"""
run_rq2.py — RQ2: Dependency mapper accuracy.

Measures how accurately dependency.map_dependencies() localises assumptions
to the correct AST nodes / line ranges in the generated code.

Evaluation:
  For each assumption with a ground-truth code_ref (line range), check whether
  the predicted code_ref overlaps with the ground truth.

  Overlap metric: IoU (intersection-over-union) of line ranges.
  A prediction is "correct" if IoU ≥ OVERLAP_THRESHOLD.

Methods evaluated:
  AssumptionMiner — tree-sitter AST + keyword overlap (dependency.map_dependencies)
  KH (Keyword Heuristic) — find lines containing keywords from assumption description
  FF (Full-File)          — predict entire file as the code region

Reports:
  - Per-category accuracy per method
  - Overall accuracy (% correct) and mean IoU per method

Usage:
    python scripts/run_rq2.py
    python scripts/run_rq2.py --benchmark data/benchmark_mini.json \\
                               --assumptions data/pilot_assumptions.json \\
                               --out data/rq2_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import re

from assumption_miner.dependency import map_dependencies  # noqa: E402
from assumption_miner.schema import AssumptionRecord, CodeRef  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BENCHMARK = REPO_ROOT / "data" / "benchmark_mini.json"
DEFAULT_ASSUMPTIONS = REPO_ROOT / "data" / "pilot_assumptions.json"
DEFAULT_OUT = REPO_ROOT / "data" / "rq2_results.json"

OVERLAP_THRESHOLD = 0.5   # IoU threshold for a "correct" localisation
CATEGORIES = ["T1", "T2", "T3", "T4", "T5", "T6"]


# --------------------------------------------------------------------------- #
# IoU helpers                                                                  #
# --------------------------------------------------------------------------- #

def line_iou(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    """Intersection-over-union for two line ranges (inclusive, 1-indexed)."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0, inter_end - inter_start + 1)
    union = (pred_end - pred_start + 1) + (gt_end - gt_start + 1) - inter
    return inter / union if union > 0 else 0.0


def best_iou(pred_refs: list[dict], gt_refs: list[dict]) -> float:
    """Return max IoU between any predicted ref and any ground-truth ref."""
    if not pred_refs or not gt_refs:
        return 0.0
    best = 0.0
    for p in pred_refs:
        for g in gt_refs:
            iou = line_iou(p["start_line"], p["end_line"], g["start_line"], g["end_line"])
            best = max(best, iou)
    return best


# --------------------------------------------------------------------------- #
# Baselines                                                                    #
# --------------------------------------------------------------------------- #

def _keywords_from(record: AssumptionRecord) -> list[str]:
    tokens = []
    for text in [record.description] + record.alternatives:
        tokens.extend(
            w.lower().strip(".,;:'\"()[]{}") for w in text.split() if len(w) > 3
        )
    return list(set(tokens))


def keyword_heuristic(record: AssumptionRecord, code: str, filename: str) -> list[dict]:
    """KH baseline: match lines that contain keywords from the assumption description."""
    keywords = _keywords_from(record)
    refs = []
    seen: set[tuple[int, int]] = set()
    for lineno, line in enumerate(code.splitlines(), 1):
        if any(kw in line.lower() for kw in keywords):
            key = (lineno, lineno)
            if key not in seen:
                seen.add(key)
                refs.append({"file": filename, "start_line": lineno, "end_line": lineno})
    return refs


def full_file(code: str, filename: str) -> list[dict]:
    """FF baseline: predict the entire file as the code region."""
    n = len(code.splitlines())
    return [{"file": filename, "start_line": 1, "end_line": n}]


def _eval_refs(pred_refs: list[dict], gt_refs: list[dict], threshold: float) -> tuple[float, bool]:
    iou = best_iou(pred_refs, gt_refs)
    correct = iou >= threshold if gt_refs else None
    return iou, correct


def _cat_summary(rows: list[dict], method_key: str, threshold: float) -> dict[str, dict]:
    per_cat: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    for r in rows:
        iou = r.get(f"{method_key}_iou")
        if iou is not None:
            per_cat[r["category"]].append(iou)
    out = {}
    for cat in CATEGORIES:
        ious = per_cat[cat]
        if ious:
            out[cat] = {
                "n": len(ious),
                "mean_iou": round(sum(ious) / len(ious), 4),
                "accuracy": round(sum(1 for v in ious if v >= threshold) / len(ious), 4),
            }
        else:
            out[cat] = {"n": 0, "mean_iou": None, "accuracy": None}
    return out


def _overall_summary(rows: list[dict], method_key: str, threshold: float) -> dict:
    evaluated = [r for r in rows if r.get(f"{method_key}_iou") is not None]
    if not evaluated:
        return {"n_evaluated": 0, "accuracy": None, "mean_iou": None}
    n_correct = sum(1 for r in evaluated if r.get(f"{method_key}_iou", 0) >= threshold)
    mean_iou = sum(r[f"{method_key}_iou"] for r in evaluated) / len(evaluated)
    return {
        "n_evaluated": len(evaluated),
        "n_correct": n_correct,
        "accuracy": round(n_correct / len(evaluated), 4),
        "mean_iou": round(mean_iou, 4),
        "threshold": threshold,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ2: dependency mapper accuracy.")
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    p.add_argument("--assumptions", default=str(DEFAULT_ASSUMPTIONS),
                   help="pilot_assumptions.json or similar extraction output")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--threshold", type=float, default=OVERLAP_THRESHOLD,
                   help=f"IoU threshold for correct localisation (default: {OVERLAP_THRESHOLD})")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load extraction results (these contain code + predicted assumptions)
    assump_path = Path(args.assumptions)
    if not assump_path.exists():
        log.error("Assumptions file not found: %s", assump_path)
        log.error("Run scripts/run_extractor.py first.")
        sys.exit(1)

    with assump_path.open(encoding="utf-8") as fh:
        extraction_results = json.load(fh)

    # Load benchmark for ground-truth code_refs (if available)
    bench_path = Path(args.benchmark)
    gt_map: dict[str, list] = {}
    if bench_path.exists():
        with bench_path.open(encoding="utf-8") as fh:
            tasks = json.load(fh)
        for task in tasks:
            for ga in task.get("ground_truth_assumptions", []):
                if ga.get("code_refs"):
                    key = f"{task['id']}:{ga['id']}"
                    gt_map[key] = ga["code_refs"]

    if args.limit:
        extraction_results = extraction_results[: args.limit]

    per_assumption: list[dict] = []

    for entry in extraction_results:
        sample_id = entry.get("sample_id", "")
        code = entry.get("code", "")
        if not code:
            continue

        raw_assumptions = entry.get("assumptions", [])
        if not raw_assumptions:
            continue

        # Reconstruct AssumptionRecord objects.
        records: list[AssumptionRecord] = []
        for a in raw_assumptions:
            try:
                records.append(AssumptionRecord.from_dict(a))
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping malformed assumption in %s: %s", sample_id, exc)

        log.info("Evaluating %s (%d assumptions) …", sample_id, len(records))

        # AssumptionMiner: run tree-sitter dependency mapper.
        am_records = [AssumptionRecord.from_dict(r.to_dict()) for r in records]  # copies
        map_dependencies(am_records, code, filename=sample_id)

        for record, am_record in zip(records, am_records):
            gt_key = f"{sample_id}:{record.id}"
            gt_refs = gt_map.get(gt_key, [])

            # AssumptionMiner predictions.
            am_pred = [r.to_dict() for r in am_record.code_refs]
            am_iou = best_iou(am_pred, gt_refs) if gt_refs else None

            # KH baseline predictions.
            kh_pred = keyword_heuristic(record, code, sample_id)
            kh_iou = best_iou(kh_pred, gt_refs) if gt_refs else None

            # FF baseline predictions.
            ff_pred = full_file(code, sample_id)
            ff_iou = best_iou(ff_pred, gt_refs) if gt_refs else None

            row = {
                "sample_id": sample_id,
                "assumption_id": record.id,
                "category": record.category,
                "n_gt_refs": len(gt_refs),
                "am_iou": round(am_iou, 4) if am_iou is not None else None,
                "kh_iou": round(kh_iou, 4) if kh_iou is not None else None,
                "ff_iou": round(ff_iou, 4) if ff_iou is not None else None,
            }
            per_assumption.append(row)

    methods = ["am", "kh", "ff"]
    results = {
        "overall": {m: _overall_summary(per_assumption, m, args.threshold) for m in methods},
        "per_category": {m: _cat_summary(per_assumption, m, args.threshold) for m in methods},
        "per_assumption": per_assumption,
        "threshold": args.threshold,
    }

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log.info("=" * 50)
    log.info("RQ2 Results (IoU threshold=%.2f):", args.threshold)
    for m in methods:
        s = results["overall"][m]
        log.info("  %-4s  accuracy=%-6s  mean_iou=%s",
                 m.upper(), s.get("accuracy"), s.get("mean_iou"))
    log.info("Wrote detailed results to %s", out_path)


if __name__ == "__main__":
    main()
