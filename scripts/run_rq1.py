"""
run_rq1.py — RQ1: Assumption extractor precision, recall, F1 against ground truth.

For each task in data/benchmark.json that has ground_truth_assumptions,
runs extract_assumptions() and compares to the ground truth.

Metrics (per category and overall):
  Precision  = |extracted ∩ ground_truth| / |extracted|
  Recall     = |extracted ∩ ground_truth| / |ground_truth|
  F1         = harmonic mean of P and R

A predicted assumption is considered correct if its *category* matches a
ground-truth assumption for the same task (category-level matching).

For finer-grained matching uncomment the description-similarity branch.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/run_rq1.py
    python scripts/run_rq1.py --benchmark data/benchmark.json \\
                               --out data/rq1_results.json
    python scripts/run_rq1.py --limit 10   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from assumption_miner import extractor, generator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BENCHMARK = REPO_ROOT / "data" / "benchmark.json"
DEFAULT_OUT = REPO_ROOT / "data" / "rq1_results.json"

CATEGORIES = ["T1", "T2", "T3", "T4", "T5", "T6"]


# --------------------------------------------------------------------------- #
# Matching                                                                     #
# --------------------------------------------------------------------------- #

def category_match(predicted: list[dict], ground_truth: list[dict]) -> tuple[int, int, int]:
    """
    Category-level matching: count how many predicted categories appear in GT.
    Returns (true_positives, len(predicted), len(ground_truth)).
    """
    gt_cats = [a["category"] for a in ground_truth]
    pred_cats = [a["category"] for a in predicted]

    # Use multiset intersection
    from collections import Counter
    gt_counter = Counter(gt_cats)
    pred_counter = Counter(pred_cats)

    tp = sum(min(pred_counter[c], gt_counter[c]) for c in pred_counter)
    return tp, len(pred_cats), len(gt_cats)


def prf(tp: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    prec = tp / n_pred if n_pred > 0 else 0.0
    rec = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ1: precision/recall of assumption extractor.")
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--backend", choices=["openai", "anthropic"], default=None)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--generate-code", action="store_true",
                   help="Generate code for tasks that lack it (adds LLM calls)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend:
        import os
        os.environ["ASSUMPTION_MINER_BACKEND"] = args.backend

    bench_path = Path(args.benchmark)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not bench_path.exists():
        log.error("Benchmark not found: %s", bench_path)
        log.error("Run scripts/select_benchmark_tasks.py + annotation first.")
        sys.exit(1)

    with bench_path.open(encoding="utf-8") as fh:
        tasks = json.load(fh)

    # Only evaluate tasks with ground truth
    annotated = [t for t in tasks if t.get("ground_truth_assumptions")]
    if not annotated:
        log.error(
            "No tasks have ground_truth_assumptions filled in. "
            "Complete the annotation step (C2) first."
        )
        sys.exit(1)

    if args.limit:
        annotated = annotated[: args.limit]

    log.info("Evaluating %d annotated tasks …", len(annotated))

    per_task: list[dict] = []
    per_cat_stats: dict[str, list] = defaultdict(list)

    total_tp = total_pred = total_gt = 0

    for i, task in enumerate(annotated, 1):
        task_id = task.get("id", f"task_{i}")
        prompt = task["prompt"]
        gt = task["ground_truth_assumptions"]

        # Generate or reuse code
        code = task.get("code") or task.get("canonical_solution", "")
        if not code:
            if args.generate_code:
                log.info("[%d/%d] Generating code for %s …", i, len(annotated), task_id)
                try:
                    code = generator.generate_code(prompt)
                    time.sleep(args.delay)
                except Exception as exc:  # noqa: BLE001
                    log.error("  Code generation failed: %s", exc)
                    continue
            else:
                log.warning("[%d/%d] %s has no code; skipping (use --generate-code)", i, len(annotated), task_id)
                continue

        log.info("[%d/%d] Extracting for %s …", i, len(annotated), task_id)
        try:
            records = extractor.extract_assumptions(prompt, code)
            time.sleep(args.delay)
        except Exception as exc:  # noqa: BLE001
            log.error("  Extraction failed: %s", exc)
            continue

        predicted = [r.to_dict() for r in records]
        tp, n_pred, n_gt = category_match(predicted, gt)
        prec, rec, f1 = prf(tp, n_pred, n_gt)

        total_tp += tp
        total_pred += n_pred
        total_gt += n_gt

        # Per-category breakdown
        for cat in CATEGORIES:
            cat_pred = [a for a in predicted if a["category"] == cat]
            cat_gt = [a for a in gt if a["category"] == cat]
            if cat_pred or cat_gt:
                tp_c, np_c, ng_c = category_match(cat_pred, cat_gt)
                per_cat_stats[cat].append((tp_c, np_c, ng_c))

        per_task.append({
            "task_id": task_id,
            "source": task.get("source", ""),
            "n_predicted": n_pred,
            "n_ground_truth": n_gt,
            "true_positives": tp,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })

        log.info("  P=%.3f R=%.3f F1=%.3f", prec, rec, f1)

    # Micro-averaged overall
    overall_p, overall_r, overall_f1 = prf(total_tp, total_pred, total_gt)

    # Per-category micro-averages
    cat_summary = {}
    for cat in CATEGORIES:
        rows = per_cat_stats.get(cat, [])
        tp_c = sum(r[0] for r in rows)
        np_c = sum(r[1] for r in rows)
        ng_c = sum(r[2] for r in rows)
        p, r, f = prf(tp_c, np_c, ng_c)
        cat_summary[cat] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    results = {
        "overall": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f1": round(overall_f1, 4),
            "n_tasks": len(per_task),
            "total_predicted": total_pred,
            "total_ground_truth": total_gt,
        },
        "per_category": cat_summary,
        "per_task": per_task,
    }

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log.info("=" * 50)
    log.info("RQ1 Results (micro-averaged):")
    log.info("  Precision: %.3f", overall_p)
    log.info("  Recall:    %.3f", overall_r)
    log.info("  F1:        %.3f", overall_f1)
    log.info("Wrote detailed results to %s", out_path)


if __name__ == "__main__":
    main()
