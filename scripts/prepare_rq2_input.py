"""
prepare_rq2_input.py — Convert benchmark_mini.json to pilot_assumptions.json format.

Strips ground-truth code_refs so we can evaluate the dependency mapper from scratch.
Saves assumptions WITHOUT code_refs into data/pilot_assumptions.json.

Usage:
    python scripts/prepare_rq2_input.py
    python scripts/prepare_rq2_input.py --benchmark data/benchmark_mini.json \\
                                         --out data/pilot_assumptions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK = REPO_ROOT / "data" / "benchmark_mini.json"
DEFAULT_OUT = REPO_ROOT / "data" / "pilot_assumptions.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare RQ2 input from benchmark.")
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench_path = Path(args.benchmark)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with bench_path.open(encoding="utf-8") as fh:
        tasks = json.load(fh)

    output = []
    for task in tasks:
        if not task.get("code") or not task.get("ground_truth_assumptions"):
            continue

        # Strip code_refs so the dependency mapper has to discover them.
        stripped_assumptions = []
        for a in task["ground_truth_assumptions"]:
            clean = {k: v for k, v in a.items() if k != "code_refs"}
            clean["code_refs"] = []
            stripped_assumptions.append(clean)

        output.append({
            "sample_id": task["id"],
            "prompt": task.get("prompt", ""),
            "code": task["code"],
            "assumptions": stripped_assumptions,
        })

    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(output)} samples to {out_path}")


if __name__ == "__main__":
    main()
