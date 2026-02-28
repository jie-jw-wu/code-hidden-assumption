"""
run_rq1_offline.py — RQ1 evaluation using offline baselines only (no API key needed).

Evaluates the Comment Extraction (CE) baseline against ground-truth assumptions in
data/benchmark_mini.json.  Writes results to data/rq1_results_offline.json and prints
a summary table matching the paper's Table II format.

Matching heuristic:
  A predicted assumption p matches a GT assumption g if:
    (a) p.category == g.category  (exact category match), AND
    (b) jaccard(tokens(p.description), tokens(g.description)) >= MATCH_THRESHOLD
        OR any keyword from g.alternatives appears in p.description

  Greedy 1-to-1 matching: each GT can be matched at most once.

Usage:
  python scripts/run_rq1_offline.py [--benchmark data/benchmark_mini.json]
                                    [--out data/rq1_results_offline.json]
                                    [--threshold 0.15]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z]+", text) if len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _matches(pred: dict, gt: dict, threshold: float) -> bool:
    if pred.get("category") != gt.get("category"):
        return False
    p_tokens = _tokenize(pred.get("description", ""))
    g_tokens = _tokenize(gt.get("description", ""))
    if _jaccard(p_tokens, g_tokens) >= threshold:
        return True
    # alt keyword overlap
    for alt in gt.get("alternatives", []):
        alt_tokens = _tokenize(alt)
        if alt_tokens and alt_tokens.issubset(p_tokens):
            return True
    return False


def _compute_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    threshold: float,
) -> dict:
    """Greedy best-match P/R/F1.  Returns per-category and aggregate."""
    matched_gt = set()
    matched_pred = set()

    # sort predictions by confidence descending for greedy matching
    preds_sorted = sorted(
        enumerate(predictions), key=lambda x: x[1].get("confidence", 0.5), reverse=True
    )

    for p_idx, pred in preds_sorted:
        for g_idx, gt in enumerate(ground_truths):
            if g_idx in matched_gt:
                continue
            if _matches(pred, gt, threshold):
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                break

    n_pred = len(predictions)
    n_gt = len(ground_truths)
    tp = len(matched_gt)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "n_pred": n_pred,
        "n_gt": n_gt,
    }


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------

def _category_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    threshold: float,
) -> dict[str, dict]:
    cats = {r.get("category") for r in ground_truths}
    out: dict[str, dict] = {}
    for cat in sorted(cats):
        preds_c = [p for p in predictions if p.get("category") == cat]
        gts_c = [g for g in ground_truths if g.get("category") == cat]
        out[cat] = _compute_metrics(preds_c, gts_c, threshold)
    return out


# ---------------------------------------------------------------------------
# Run baselines on a single task
# ---------------------------------------------------------------------------

def _records_to_dicts(records) -> list[dict]:
    return [
        {
            "id": r.id,
            "category": r.category,
            "description": r.description,
            "confidence": r.confidence,
            "alternatives": r.alternatives,
        }
        for r in records
    ]


def _run_baseline(name: str, task: dict) -> list[dict]:
    from assumption_miner.baselines import comment_extraction, pattern_extraction

    code = task.get("code", "")
    prompt = task.get("prompt", "")
    if name == "CE":
        return _records_to_dicts(comment_extraction(prompt, code))
    elif name == "KBE":
        return _records_to_dicts(pattern_extraction(prompt, code))
    else:
        raise ValueError(f"Unknown offline baseline: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CAT_LABELS = {
    "T1": "Input Validation",
    "T2": "Data Format",
    "T3": "Error Policy",
    "T4": "Persistence",
    "T5": "Performance",
    "T6": "Security",
}


def _run_one_baseline(name: str, tasks: list[dict], threshold: float) -> dict:
    all_preds: list[dict] = []
    all_gts: list[dict] = []
    per_task_rows: list[dict] = []

    for task in tasks:
        task_id = task.get("id", "?")
        gt = task.get("ground_truth_assumptions", [])
        preds = _run_baseline(name, task)
        metrics = _compute_metrics(preds, gt, threshold)
        per_task_rows.append({"task_id": task_id, **metrics})
        all_preds.extend(preds)
        all_gts.extend(gt)

    agg = _compute_metrics(all_preds, all_gts, threshold)
    cat_metrics = _category_metrics(all_preds, all_gts, threshold)
    return {
        "baseline": name,
        "aggregate": agg,
        "per_category": cat_metrics,
        "per_task": per_task_rows,
        "n_pred_total": len(all_preds),
        "n_gt_total": len(all_gts),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RQ1 offline evaluation (offline baselines).")
    parser.add_argument("--benchmark", default="data/benchmark_mini.json")
    parser.add_argument("--out", default="data/rq1_results_offline.json")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Jaccard similarity threshold for assumption matching.")
    parser.add_argument("--baselines", nargs="+", default=["CE", "KBE"],
                        help="Which offline baselines to evaluate.")
    args = parser.parse_args(argv)

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"ERROR: benchmark not found at {benchmark_path}", file=sys.stderr)
        sys.exit(1)

    tasks = json.loads(benchmark_path.read_text())
    print(f"Loaded {len(tasks)} tasks from {benchmark_path}\n")

    all_results: dict[str, dict] = {}

    for name in args.baselines:
        print(f"=== {name} Baseline ===")
        res = _run_one_baseline(name, tasks, args.threshold)
        all_results[name] = res
        agg = res["aggregate"]
        print(
            f"  Tasks: {len(tasks)}  GT: {res['n_gt_total']}  "
            f"Pred: {res['n_pred_total']}"
        )
        print(f"  Precision: {agg['precision']:.3f}")
        print(f"  Recall:    {agg['recall']:.3f}")
        print(f"  F1:        {agg['f1']:.3f}")

        print("  Per-category:")
        for cat, m in res["per_category"].items():
            label = _CAT_LABELS.get(cat, cat)
            print(f"    {cat} ({label:<18}): P={m['precision']:.2f}  R={m['recall']:.2f}  "
                  f"F1={m['f1']:.2f}  (GT={m['n_gt']}, pred={m['n_pred']})")
        print()

    # Summary table
    print("=== Summary Table (matches paper Table II format) ===")
    print(f"{'Baseline':<10}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print("-" * 40)
    for name, res in all_results.items():
        a = res["aggregate"]
        print(f"{name:<10}  {a['precision']:9.3f}  {a['recall']:6.3f}  {a['f1']:6.3f}")

    output = {
        "benchmark": str(benchmark_path),
        "n_tasks": len(tasks),
        "threshold": args.threshold,
        "results": all_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
