#!/usr/bin/env python3
"""
run_rq2_ablation.py — Ablation study for the AM dependency mapper (RQ2).

Evaluates how different node-type configurations affect accuracy, isolating
the contribution of each expansion added in Iteration 5 (T3: with_statement,
call; T5: import_from_statement, assignment).

Usage:
    python scripts/run_rq2_ablation.py \
        --benchmark data/benchmark_mini.json \
        --out data/rq2_ablation.json

No API key required — runs fully offline via tree-sitter AST.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node-type configurations to evaluate
# ---------------------------------------------------------------------------

# Original configuration BEFORE iteration-5 expansions
_ORIGINAL = {
    "T1": ["function_definition", "if_statement", "assert_statement",
           "call", "with_statement", "import_from_statement"],
    "T2": ["return_statement", "typed_parameter", "type",
           "function_definition", "assignment"],
    "T3": ["except_clause", "raise_statement", "if_statement",
           "return_statement", "expression_statement"],       # no call, no with_statement
    "T4": ["import_statement", "import_from_statement",
           "assignment", "call", "with_statement"],
    "T5": ["function_definition", "for_statement", "while_statement",
           "if_statement", "class_definition",
           "import_statement", "assignment"],                 # no import_from_statement
    "T6": ["import_statement", "import_from_statement", "call",
           "assignment", "function_definition"],
}

# Current full configuration
_FULL = {
    "T1": ["function_definition", "if_statement", "assert_statement",
           "call", "with_statement", "import_from_statement"],
    "T2": ["return_statement", "typed_parameter", "type",
           "function_definition", "assignment"],
    "T3": ["except_clause", "raise_statement", "if_statement",
           "return_statement", "call", "with_statement",
           "expression_statement"],
    "T4": ["import_statement", "import_from_statement",
           "assignment", "call", "with_statement"],
    "T5": ["function_definition", "for_statement", "while_statement",
           "if_statement", "class_definition",
           "import_statement", "import_from_statement", "assignment"],
    "T6": ["import_statement", "import_from_statement", "call",
           "assignment", "function_definition"],
}

# Ablations: remove one expansion at a time from FULL
_ABLATIONS = {
    "full": _FULL,
    "no_T3_call": {
        **_FULL,
        "T3": ["except_clause", "raise_statement", "if_statement",
               "return_statement", "with_statement", "expression_statement"],
    },
    "no_T3_with_statement": {
        **_FULL,
        "T3": ["except_clause", "raise_statement", "if_statement",
               "return_statement", "call", "expression_statement"],
    },
    "no_T5_import_from": {
        **_FULL,
        "T5": ["function_definition", "for_statement", "while_statement",
               "if_statement", "class_definition",
               "import_statement", "assignment"],
    },
    "no_T5_assignment": {
        **_FULL,
        "T5": ["function_definition", "for_statement", "while_statement",
               "if_statement", "class_definition",
               "import_statement", "import_from_statement"],
    },
    "original": _ORIGINAL,
}


# ---------------------------------------------------------------------------
# Evaluation helpers (mirrored from run_rq2.py)
# ---------------------------------------------------------------------------

def _line_iou(p_start: int, p_end: int, g_start: int, g_end: int) -> float:
    """Intersection-over-union for two inclusive line ranges (1-indexed)."""
    inter_start = max(p_start, g_start)
    inter_end = min(p_end, g_end)
    inter = max(0, inter_end - inter_start + 1)
    union = (p_end - p_start + 1) + (g_end - g_start + 1) - inter
    return inter / union if union > 0 else 0.0


def _best_iou(pred_refs: list, gt_refs: list[dict]) -> float:
    """Max IoU between any predicted CodeRef and any GT ref dict."""
    if not pred_refs or not gt_refs:
        return 0.0
    best = 0.0
    for p in pred_refs:
        for g in gt_refs:
            v = _line_iou(p.start_line, p.end_line, g["start_line"], g["end_line"])
            best = max(best, v)
    return best


def _evaluate_config(tasks: list[dict], node_types: dict,
                     threshold: float = 0.5) -> dict:
    """Run dependency mapping with given node_types; return accuracy + IoU."""
    import assumption_miner.dependency as dep_mod

    # Temporarily patch the module-level dict
    original = copy.deepcopy(dep_mod._CATEGORY_NODE_TYPES)
    dep_mod._CATEGORY_NODE_TYPES = node_types

    try:
        n_correct = 0
        total = 0
        iou_sum = 0.0
        per_cat: dict[str, dict] = {}

        for task in tasks:
            code = task["code"]
            for gt in task["ground_truth_assumptions"]:
                cat = gt["category"]
                gt_refs = gt.get("code_refs", [])
                if not gt_refs:
                    continue

                # Build AssumptionRecord with full GT data for fair keyword extraction
                from assumption_miner.schema import AssumptionRecord
                rec = AssumptionRecord(
                    id=gt["id"],
                    category=cat,
                    description=gt["description"],
                    rationale=gt.get("rationale", ""),
                    alternatives=gt.get("alternatives", []),
                    confidence=1.0,
                    severity="medium",
                )

                from assumption_miner.dependency import map_dependencies
                map_dependencies([rec], code, use_llm=False)

                iou = _best_iou(rec.code_refs, gt_refs)
                correct = iou >= threshold

                n_correct += correct
                total += 1
                iou_sum += iou

                if cat not in per_cat:
                    per_cat[cat] = {"n": 0, "correct": 0, "iou_sum": 0.0}
                per_cat[cat]["n"] += 1
                per_cat[cat]["correct"] += correct
                per_cat[cat]["iou_sum"] += iou

        accuracy = n_correct / total if total else 0.0
        mean_iou = iou_sum / total if total else 0.0

        per_cat_out = {}
        for cat, v in per_cat.items():
            per_cat_out[cat] = {
                "n": v["n"],
                "accuracy": round(v["correct"] / v["n"], 4),
                "mean_iou": round(v["iou_sum"] / v["n"], 4),
            }

        return {
            "n_evaluated": total,
            "n_correct": n_correct,
            "accuracy": round(accuracy, 4),
            "mean_iou": round(mean_iou, 4),
            "per_category": per_cat_out,
        }

    finally:
        dep_mod._CATEGORY_NODE_TYPES = original


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmark_mini.json")
    ap.add_argument("--out", default="data/rq2_ablation.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    tasks = json.loads(Path(args.benchmark).read_text())
    log.info("Loaded %d tasks from %s", len(tasks), args.benchmark)

    results = {}
    for name, node_types in _ABLATIONS.items():
        log.info("Evaluating configuration: %s …", name)
        res = _evaluate_config(tasks, node_types, threshold=args.threshold)
        results[name] = res
        log.info("  accuracy=%.4f  mean_iou=%.4f", res["accuracy"], res["mean_iou"])

    # Print summary table
    log.info("")
    log.info("=" * 60)
    log.info("Ablation Summary (IoU threshold=%.2f):", args.threshold)
    log.info("  %-30s  %6s  %8s", "Configuration", "Acc.", "Mean IoU")
    for name, res in results.items():
        log.info("  %-30s  %.4f  %.4f", name, res["accuracy"], res["mean_iou"])

    out = {
        "benchmark": args.benchmark,
        "n_tasks": len(tasks),
        "threshold": args.threshold,
        "ablations": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("Wrote results to %s", args.out)


if __name__ == "__main__":
    main()
