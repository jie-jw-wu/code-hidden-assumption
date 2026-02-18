"""
export_for_annotation.py — B2: Export pilot assumptions to CSV for human coders.

Reads data/pilot_assumptions.json and writes a flat CSV where each row is one
assumption.  Coders fill in the "coder_label" and "coder_notes" columns.

After coding, compute inter-rater agreement (Cohen's κ) with:
    python scripts/export_for_annotation.py --kappa coder1.csv coder2.csv

Usage:
    python scripts/export_for_annotation.py
    python scripts/export_for_annotation.py \\
        --in data/pilot_assumptions.json \\
        --out data/annotation_sheet.csv

    # After coders return their CSVs:
    python scripts/export_for_annotation.py \\
        --kappa data/annotation_coder1.csv data/annotation_coder2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_IN = REPO_ROOT / "data" / "pilot_assumptions.json"
DEFAULT_OUT = REPO_ROOT / "data" / "annotation_sheet.csv"

# Columns written to the annotation sheet
FIELDNAMES = [
    "row_id",          # unique row identifier
    "sample_id",       # which pilot sample this came from
    "assumption_id",   # e.g. "A1"
    "category",        # T1–T6 (model's label)
    "description",     # natural-language assumption
    "rationale",
    "alternatives",
    "confidence",      # model confidence
    "severity",        # model severity
    # --- coder fills these in ---
    "coder_label",     # agreed/revised category (T1–T6) or "reject"
    "coder_notes",     # free-text notes
]


def export_to_csv(in_path: Path, out_path: Path) -> None:
    with in_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_num = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()

        for entry in data:
            sample_id = entry.get("sample_id", "")
            for assumption in entry.get("assumptions", []):
                row_num += 1
                writer.writerow({
                    "row_id": row_num,
                    "sample_id": sample_id,
                    "assumption_id": assumption.get("id", ""),
                    "category": assumption.get("category", ""),
                    "description": assumption.get("description", ""),
                    "rationale": assumption.get("rationale", ""),
                    "alternatives": "; ".join(assumption.get("alternatives", [])),
                    "confidence": assumption.get("confidence", ""),
                    "severity": assumption.get("severity", ""),
                    "coder_label": "",
                    "coder_notes": "",
                })

    print(f"Wrote {row_num} assumption rows to {out_path}")
    print("Distribute the CSV to both coders.  Ask them to fill in 'coder_label'")
    print("with the correct category (T1–T6) or 'reject' for false positives.")


def compute_kappa(csv1: Path, csv2: Path) -> None:
    """Compute Cohen's κ between two completed annotation CSVs."""

    def load_labels(path: Path) -> dict[str, str]:
        labels: dict[str, str] = {}
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rid = row["row_id"]
                label = row.get("coder_label", "").strip()
                if label:
                    labels[rid] = label
        return labels

    labels1 = load_labels(csv1)
    labels2 = load_labels(csv2)

    common = sorted(set(labels1) & set(labels2), key=int)
    if not common:
        print("No common labelled rows found between the two files.")
        return

    y1 = [labels1[r] for r in common]
    y2 = [labels2[r] for r in common]

    # Count agreements
    agree = sum(a == b for a, b in zip(y1, y2))
    n = len(common)
    p_o = agree / n

    # Expected agreement
    from collections import Counter
    c1 = Counter(y1)
    c2 = Counter(y2)
    all_cats = set(c1) | set(c2)
    p_e = sum((c1[cat] / n) * (c2[cat] / n) for cat in all_cats)

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 1.0

    print(f"Rows compared: {n}")
    print(f"Observed agreement: {p_o:.3f}")
    print(f"Expected agreement: {p_e:.3f}")
    print(f"Cohen's κ = {kappa:.3f}")
    if kappa >= 0.8:
        print("Excellent agreement (κ ≥ 0.8)")
    elif kappa >= 0.7:
        print("Substantial agreement (κ ≥ 0.7) — meets threshold for proceeding")
    elif kappa >= 0.6:
        print("Moderate agreement (κ ≥ 0.6) — discuss disagreements before proceeding")
    else:
        print("Poor agreement (κ < 0.6) — revise codebook and re-annotate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export pilot assumptions to CSV for annotation, or compute κ."
    )
    sub = p.add_subparsers(dest="cmd")

    exp = sub.add_parser("export", help="Export assumptions to annotation CSV (default)")
    exp.add_argument("--in", dest="infile", default=str(DEFAULT_IN))
    exp.add_argument("--out", default=str(DEFAULT_OUT))

    kap = sub.add_parser("kappa", help="Compute Cohen's κ between two coded CSVs")
    kap.add_argument("csv1")
    kap.add_argument("csv2")

    # Allow running without a subcommand (defaults to export)
    p.add_argument("--in", dest="infile", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--kappa", nargs=2, metavar=("CSV1", "CSV2"),
                   help="Compute κ between two coder CSVs instead of exporting")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "kappa" or getattr(args, "kappa", None):
        if args.cmd == "kappa":
            compute_kappa(Path(args.csv1), Path(args.csv2))
        else:
            compute_kappa(Path(args.kappa[0]), Path(args.kappa[1]))
    else:
        in_path = Path(args.infile)
        out_path = Path(args.out)
        if not in_path.exists():
            print(f"ERROR: Input file not found: {in_path}")
            print("Run scripts/run_extractor.py first.")
            sys.exit(1)
        export_to_csv(in_path, out_path)


if __name__ == "__main__":
    main()
