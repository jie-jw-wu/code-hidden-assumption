"""
run_rq2.py — RQ2: Dependency mapper accuracy.

Measures how accurately dependency.map_dependencies() localises assumptions
to the correct AST nodes / line ranges in the generated code.

Evaluation:
  For each assumption with a ground-truth code_ref (line range), check whether
  the predicted code_ref overlaps with the ground truth.

  Overlap metric: IoU (intersection-over-union) of line ranges.
  A prediction is "correct" if IoU ≥ OVERLAP_THRESHOLD.

Reports:
  - Per-category accuracy
  - Overall accuracy (% correct)
  - Mean IoU across all assumptions

Usage:
    python scripts/run_rq2.py
    python scripts/run_rq2.py --benchmark data/benchmark.json \\
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

from assumption_miner.dependency import map_dependencies  # noqa: E402
from assumption_miner.schema import AssumptionRecord  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BENCHMARK = REPO_ROOT / "data" / "benchmark.json"
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
    per_cat: dict[str, list[float]] = {c: [] for c in CATEGORIES}

    for entry in extraction_results:
        sample_id = entry.get("sample_id", "")
        code = entry.get("code", "")
        if not code:
            continue

        raw_assumptions = entry.get("assumptions", [])
        if not raw_assumptions:
            continue

        # Reconstruct AssumptionRecord objects and run mapper
        records = []
        for a in raw_assumptions:
            try:
                records.append(AssumptionRecord.from_dict(a))
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping malformed assumption in %s: %s", sample_id, exc)

        log.info("Mapping dependencies for %s (%d assumptions) …", sample_id, len(records))
        map_dependencies(records, code, filename=sample_id)

        for record in records:
            pred_refs = [r.to_dict() for r in record.code_refs]
            gt_key = f"{sample_id}:{record.id}"
            gt_refs = gt_map.get(gt_key, [])

            iou = best_iou(pred_refs, gt_refs) if gt_refs else None
            correct = (iou is not None and iou >= args.threshold)

            row = {
                "sample_id": sample_id,
                "assumption_id": record.id,
                "category": record.category,
                "n_predicted_refs": len(pred_refs),
                "n_gt_refs": len(gt_refs),
                "best_iou": round(iou, 4) if iou is not None else None,
                "correct": correct if gt_refs else None,
            }
            per_assumption.append(row)

            if gt_refs and iou is not None:
                per_cat[record.category].append(iou)

    # Summary statistics
    evaluated = [r for r in per_assumption if r["correct"] is not None]
    if evaluated:
        n_correct = sum(1 for r in evaluated if r["correct"])
        accuracy = n_correct / len(evaluated)
        mean_iou = sum(r["best_iou"] for r in evaluated) / len(evaluated)
    else:
        accuracy = mean_iou = 0.0
        log.warning(
            "No assumptions had ground-truth code_refs — "
            "accuracy/IoU cannot be computed. "
            "Check that benchmark.json has ground_truth_assumptions with code_refs."
        )

    cat_summary = {}
    for cat in CATEGORIES:
        ious = per_cat[cat]
        if ious:
            cat_summary[cat] = {
                "n": len(ious),
                "mean_iou": round(sum(ious) / len(ious), 4),
                "accuracy": round(sum(1 for v in ious if v >= args.threshold) / len(ious), 4),
            }
        else:
            cat_summary[cat] = {"n": 0, "mean_iou": None, "accuracy": None}

    results = {
        "overall": {
            "n_evaluated": len(evaluated),
            "n_correct": sum(1 for r in evaluated if r["correct"]) if evaluated else 0,
            "accuracy": round(accuracy, 4),
            "mean_iou": round(mean_iou, 4),
            "threshold": args.threshold,
        },
        "per_category": cat_summary,
        "per_assumption": per_assumption,
    }

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log.info("=" * 50)
    log.info("RQ2 Results:")
    log.info("  Evaluated: %d assumptions", len(evaluated))
    log.info("  Accuracy:  %.3f", accuracy)
    log.info("  Mean IoU:  %.3f", mean_iou)
    log.info("Wrote detailed results to %s", out_path)


if __name__ == "__main__":
    main()
