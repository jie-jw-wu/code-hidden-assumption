"""
run_extractor.py — D2: Batch assumption extraction over pilot data.

Reads data/pilot_samples.json, runs extractor.extract_assumptions() on each
sample (optionally also maps dependencies via dependency.map_dependencies()),
and writes results to data/pilot_assumptions.json.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/run_extractor.py                        # default paths
    python scripts/run_extractor.py --in data/pilot_samples.json \\
                                    --out data/pilot_assumptions.json
    python scripts/run_extractor.py --map-deps             # also run tree-sitter mapping
    python scripts/run_extractor.py --backend anthropic
    python scripts/run_extractor.py --limit 5              # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from assumption_miner import extractor, ingestion  # noqa: E402
from assumption_miner.dependency import map_dependencies  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_IN = REPO_ROOT / "data" / "pilot_samples.json"
DEFAULT_OUT = REPO_ROOT / "data" / "pilot_assumptions.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch assumption extraction on pilot samples.")
    p.add_argument("--in", dest="infile", default=str(DEFAULT_IN),
                   help="Input JSON file (default: data/pilot_samples.json)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="Output JSON file (default: data/pilot_assumptions.json)")
    p.add_argument("--backend", choices=["openai", "anthropic"], default=None,
                   help="LLM backend (overrides ASSUMPTION_MINER_BACKEND)")
    p.add_argument("--map-deps", action="store_true",
                   help="Run tree-sitter dependency mapping after extraction")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds to sleep between API calls (default: 1.0)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N samples")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend:
        import os
        os.environ["ASSUMPTION_MINER_BACKEND"] = args.backend

    in_path = Path(args.infile)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        log.error("Input file not found: %s", in_path)
        log.error("Run scripts/collect_pilot.py first.")
        sys.exit(1)

    samples = list(ingestion.from_json_file(in_path))
    if args.limit:
        samples = samples[: args.limit]

    log.info("Loaded %d samples from %s", len(samples), in_path)

    results: list[dict] = []

    for i, sample in enumerate(samples, 1):
        log.info("[%d/%d] Extracting assumptions for %s …", i, len(samples), sample.source_id)

        if not sample.code:
            log.warning("  Skipping %s (no code)", sample.source_id)
            continue

        try:
            records = extractor.extract_assumptions(sample.prompt, sample.code)
        except Exception as exc:  # noqa: BLE001
            log.error("  Extraction failed for %s: %s", sample.source_id, exc)
            results.append({
                "sample_id": sample.source_id,
                "prompt": sample.prompt,
                "code": sample.code,
                "assumptions": [],
                "error": str(exc),
            })
            continue

        if args.map_deps:
            try:
                map_dependencies(records, sample.code, filename=sample.source_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("  Dependency mapping failed: %s", exc)

        log.info("  → %d assumption(s) found", len(records))

        results.append({
            "sample_id": sample.source_id,
            "prompt": sample.prompt,
            "code": sample.code,
            "assumptions": [r.to_dict() for r in records],
        })

        if args.delay > 0:
            time.sleep(args.delay)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d extraction results to %s", len(results), out_path)

    total_assumptions = sum(len(r["assumptions"]) for r in results)
    log.info("Total assumptions extracted: %d (avg %.1f per sample)",
             total_assumptions, total_assumptions / max(len(results), 1))


if __name__ == "__main__":
    main()
