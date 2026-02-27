"""
run_rq3.py — RQ3: Code adaptability after assumption revision.

For each benchmark task with ground-truth assumptions:
  1. Pick the highest-severity assumption (or the first one if tied).
  2. Flip it to its first listed alternative.
  3. Run AssumptionMiner incremental regeneration (C5) to update the code.
  4. Compare against three baselines:
       MR  — Manual Reprompt:  full regeneration with original prompt + explicit constraint
       FR  — Full Regeneration: regenerate from scratch with the original prompt only
       IPE — In-Place LLM Edit: single edit instruction on the full code block

Metrics:
  pass_rate    — fraction of tasks where updated code is syntactically valid
                 (proxy for test pass when unit tests are absent)
  edit_dist    — normalised Levenshtein distance between original and updated code
  latency_s    — wall-clock time of the LLM call in seconds

Usage:
    export ANTHROPIC_API_KEY=...   (or OPENAI_API_KEY)
    python scripts/run_rq3.py --benchmark data/benchmark_mini.json \\
                               --out data/rq3_results.json
    python scripts/run_rq3.py --limit 3 --dry-run   # offline smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import py_compile
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from assumption_miner import regenerator  # noqa: E402
from assumption_miner.dependency import map_dependencies  # noqa: E402
from assumption_miner.schema import AssumptionRecord  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BENCHMARK = REPO_ROOT / "data" / "benchmark_mini.json"
DEFAULT_OUT = REPO_ROOT / "data" / "rq3_results.json"

SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_assumption(gt: list[dict]) -> Optional[dict]:
    """Return the assumption with the highest severity (first one if tied)."""
    if not gt:
        return None
    return max(gt, key=lambda a: SEVERITY_RANK.get(a.get("severity", "medium"), 2))


def _pick_alternative(assumption: dict) -> Optional[str]:
    alts = assumption.get("alternatives", [])
    return alts[0] if alts else None


def _syntax_ok(code: str) -> bool:
    """Return True if *code* is syntactically valid Python."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        py_compile.compile(fname, doraise=True)
        return True
    except py_compile.PyCompileError:
        return False
    finally:
        Path(fname).unlink(missing_ok=True)


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance on characters."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


def _normalised_edit(orig: str, updated: str) -> float:
    """Normalised edit distance in [0, 1]."""
    if not orig and not updated:
        return 0.0
    d = _edit_distance(orig, updated)
    return d / max(len(orig), len(updated))


# ---------------------------------------------------------------------------
# LLM call helpers for baselines
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str, model: Optional[str] = None) -> str:
    backend = os.environ.get("ASSUMPTION_MINER_BACKEND", "openai").lower()
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            temperature=0.2,
            max_tokens=2048,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content.strip()
    elif backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=2048,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()
    else:
        raise ValueError(f"Unknown backend '{backend}'.")


_CODE_SYS = (
    "You are an expert software engineer. "
    "Output ONLY the complete Python code, no explanation."
)


def baseline_manual_reprompt(
    prompt: str, original_code: str, assumption: dict, new_desc: str,
    model: Optional[str] = None
) -> tuple[str, float]:
    """Full reprompt with explicit constraint added to the original prompt."""
    constraint = f"\nIMPORTANT: {new_desc}"
    user = prompt + constraint
    t0 = time.perf_counter()
    code = _call_llm(_CODE_SYS, user, model=model)
    latency = time.perf_counter() - t0
    # Strip markdown fences if present
    code = _strip_fences(code)
    return code, latency


def baseline_full_regeneration(
    prompt: str, model: Optional[str] = None
) -> tuple[str, float]:
    """Full regeneration from original prompt (ignores the assumption revision)."""
    t0 = time.perf_counter()
    code = _call_llm(_CODE_SYS, prompt, model=model)
    latency = time.perf_counter() - t0
    return _strip_fences(code), latency


def baseline_in_place_edit(
    original_code: str, assumption: dict, new_desc: str,
    model: Optional[str] = None
) -> tuple[str, float]:
    """Direct edit instruction applied to the full code."""
    old_desc = assumption.get("description", "")
    user = (
        f"Edit this Python code to change the following:\n"
        f"FROM: {old_desc}\n"
        f"TO:   {new_desc}\n\n"
        f"Code:\n```python\n{original_code}\n```\n\n"
        "Output ONLY the complete updated code."
    )
    t0 = time.perf_counter()
    code = _call_llm(_CODE_SYS, user, model=model)
    latency = time.perf_counter() - t0
    return _strip_fences(code), latency


def _strip_fences(code: str) -> str:
    """Remove ```python ... ``` fences if the LLM added them."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        # drop first line (```python or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        code = "\n".join(inner)
    return code


def assumption_miner_incremental(
    original_code: str, record: AssumptionRecord, new_desc: str,
    model: Optional[str] = None
) -> tuple[str, float]:
    """C5 incremental regeneration with dependency-aware targeting."""
    # Ensure code_refs are populated.
    if not record.code_refs:
        map_dependencies([record], original_code)
    t0 = time.perf_counter()
    updated = regenerator.regenerate(
        original_code, record, new_desc, model=model
    )
    latency = time.perf_counter() - t0
    return updated, latency


# ---------------------------------------------------------------------------
# Dry-run mode (no API calls)
# ---------------------------------------------------------------------------

def dry_run_update(original_code: str, old_desc: str, new_desc: str) -> tuple[str, float]:
    """
    Simulate an update by appending a comment showing the revision.
    Used for offline smoke testing when no API keys are present.
    """
    comment = f"# ASSUMPTION REVISED: {old_desc!r} → {new_desc!r}\n"
    return comment + original_code, 0.001


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "pass_rate": None, "mean_edit_dist": None, "mean_latency_s": None}
    n = len(rows)
    pass_rate = sum(r["syntax_ok"] for r in rows) / n
    mean_ed = sum(r["edit_dist"] for r in rows) / n
    mean_lat = sum(r["latency_s"] for r in rows) / n
    return {
        "n": n,
        "pass_rate": round(pass_rate, 4),
        "mean_edit_dist": round(mean_ed, 4),
        "mean_latency_s": round(mean_lat, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ3: code adaptability after assumption revision.")
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--backend", choices=["openai", "anthropic"], default=None)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip LLM calls; use template-based edits for offline testing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend:
        os.environ["ASSUMPTION_MINER_BACKEND"] = args.backend

    bench_path = Path(args.benchmark)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not bench_path.exists():
        log.error("Benchmark not found: %s", bench_path)
        sys.exit(1)

    with bench_path.open(encoding="utf-8") as fh:
        tasks = json.load(fh)

    annotated = [t for t in tasks if t.get("ground_truth_assumptions") and t.get("code")]
    if not annotated:
        log.error("No tasks with both code and ground_truth_assumptions found.")
        sys.exit(1)

    if args.limit:
        annotated = annotated[: args.limit]

    log.info("RQ3: evaluating %d tasks (dry_run=%s) …", len(annotated), args.dry_run)

    # Result containers per method
    am_rows: list[dict] = []
    mr_rows: list[dict] = []
    fr_rows: list[dict] = []
    ipe_rows: list[dict] = []
    per_task: list[dict] = []

    for i, task in enumerate(annotated, 1):
        task_id = task.get("id", f"task_{i}")
        prompt = task["prompt"]
        original_code = task["code"]
        gt = task["ground_truth_assumptions"]

        assumption_dict = _pick_assumption(gt)
        if not assumption_dict:
            log.warning("[%d/%d] %s: no assumptions, skipping.", i, len(annotated), task_id)
            continue

        new_desc = _pick_alternative(assumption_dict)
        if not new_desc:
            log.warning("[%d/%d] %s: assumption has no alternatives, skipping.", i, len(annotated), task_id)
            continue

        old_desc = assumption_dict.get("description", "")
        log.info("[%d/%d] %s: flipping '%s' → '%s'",
                 i, len(annotated), task_id, old_desc[:50], new_desc[:50])

        # Build AssumptionRecord from ground-truth for the AssumptionMiner method.
        try:
            am_record = AssumptionRecord.from_dict(assumption_dict)
        except Exception as exc:  # noqa: BLE001
            log.error("  Failed to parse assumption record: %s", exc)
            continue

        def _run_method(name: str, fn):
            try:
                updated, latency = fn()
                ok = _syntax_ok(updated)
                ed = _normalised_edit(original_code, updated)
                return {
                    "task_id": task_id,
                    "method": name,
                    "syntax_ok": ok,
                    "edit_dist": round(ed, 4),
                    "latency_s": round(latency, 4),
                    "updated_code": updated,
                }
            except Exception as exc:  # noqa: BLE001
                log.error("  %s failed: %s", name, exc)
                return {
                    "task_id": task_id,
                    "method": name,
                    "syntax_ok": False,
                    "edit_dist": 1.0,
                    "latency_s": 0.0,
                    "error": str(exc),
                }

        if args.dry_run:
            am_result = _run_method("AssumptionMiner",
                lambda: dry_run_update(original_code, old_desc, new_desc))
            mr_result = _run_method("MR",
                lambda: dry_run_update(original_code, old_desc, f"[MR] {new_desc}"))
            fr_result = _run_method("FR",
                lambda: (original_code, 0.001))  # no change = full regen placeholder
            ipe_result = _run_method("IPE",
                lambda: dry_run_update(original_code, old_desc, f"[IPE] {new_desc}"))
        else:
            am_result = _run_method("AssumptionMiner",
                lambda: assumption_miner_incremental(original_code, am_record, new_desc))
            time.sleep(args.delay)

            mr_result = _run_method("MR",
                lambda: baseline_manual_reprompt(prompt, original_code, assumption_dict, new_desc))
            time.sleep(args.delay)

            fr_result = _run_method("FR",
                lambda: baseline_full_regeneration(prompt))
            time.sleep(args.delay)

            ipe_result = _run_method("IPE",
                lambda: baseline_in_place_edit(original_code, assumption_dict, new_desc))
            time.sleep(args.delay)

        am_rows.append(am_result)
        mr_rows.append(mr_result)
        fr_rows.append(fr_result)
        ipe_rows.append(ipe_result)

        per_task.append({
            "task_id": task_id,
            "flipped_assumption_id": assumption_dict.get("id"),
            "old_description": old_desc,
            "new_description": new_desc,
            "AssumptionMiner": {k: v for k, v in am_result.items()
                                if k not in ("updated_code", "task_id", "method")},
            "MR": {k: v for k, v in mr_result.items()
                   if k not in ("updated_code", "task_id", "method")},
            "FR": {k: v for k, v in fr_result.items()
                   if k not in ("updated_code", "task_id", "method")},
            "IPE": {k: v for k, v in ipe_result.items()
                    if k not in ("updated_code", "task_id", "method")},
        })

        log.info("  AM: ok=%s ed=%.3f  MR: ok=%s  FR: ok=%s  IPE: ok=%s",
                 am_result["syntax_ok"], am_result["edit_dist"],
                 mr_result["syntax_ok"], fr_result["syntax_ok"], ipe_result["syntax_ok"])

    results = {
        "config": {
            "benchmark": str(bench_path),
            "n_tasks": len(per_task),
            "dry_run": args.dry_run,
        },
        "summary": {
            "AssumptionMiner": aggregate(am_rows),
            "MR": aggregate(mr_rows),
            "FR": aggregate(fr_rows),
            "IPE": aggregate(ipe_rows),
        },
        "per_task": per_task,
    }

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log.info("=" * 50)
    log.info("RQ3 Summary:")
    for method, stats in results["summary"].items():
        log.info("  %-18s pass_rate=%-6s edit_dist=%-6s latency_s=%s",
                 method,
                 stats.get("pass_rate"),
                 stats.get("mean_edit_dist"),
                 stats.get("mean_latency_s"))
    log.info("Wrote detailed results to %s", out_path)


if __name__ == "__main__":
    main()
