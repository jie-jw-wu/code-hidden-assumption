"""
collect_pilot.py — B1: Generate pilot (prompt, code) pairs.

Generates 60 (prompt, code) pairs covering the four task classes in §2:
  - Auth / security        (15 prompts)
  - Algorithmic            (15 prompts)
  - CRUD / REST APIs       (15 prompts)
  - Data-processing / ETL  (15 prompts)

Each prompt is deliberately ambiguous so the LLM must make hidden assumptions.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/collect_pilot.py                      # writes data/pilot_samples.json
    python scripts/collect_pilot.py --out path/to/file.json
    python scripts/collect_pilot.py --backend anthropic --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make sure the repo root is on sys.path when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from assumption_miner import generator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt catalogue                                                             #
# --------------------------------------------------------------------------- #

AUTH_PROMPTS = [
    "Write a function that checks whether a user is allowed to access a resource.",
    "Implement user login that verifies credentials and starts a session.",
    "Create a registration endpoint that stores a new user account.",
    "Write a password reset function for a web application.",
    "Implement an API key validator that controls access to a REST endpoint.",
    "Write middleware that restricts certain routes to logged-in users only.",
    "Create a function that hashes a password before storing it.",
    "Implement role-based access control so admins can do more than regular users.",
    "Write a token refresh function for a stateless authentication system.",
    "Create a logout function that ends the current user session.",
    "Implement two-factor authentication for a web login flow.",
    "Write a function that limits the number of failed login attempts.",
    "Create an OAuth callback handler that logs users in via a third-party provider.",
    "Write a function that generates a secure temporary link for file download.",
    "Implement JWT generation and validation for an API.",
]

ALGO_PROMPTS = [
    "Write a function that finds all pairs of numbers in a list that sum to a target.",
    "Implement a function that returns the k most frequent elements in an array.",
    "Write a function that determines whether a string is a palindrome.",
    "Create a function that merges two sorted lists into one sorted list.",
    "Write a function that counts the number of islands in a 2D grid.",
    "Implement a function that finds the longest substring without repeating characters.",
    "Write a function that computes the edit distance between two strings.",
    "Create a function that checks whether a binary tree is balanced.",
    "Write a function that returns all permutations of a given string.",
    "Implement a function that performs a binary search on a sorted array.",
    "Write a function that detects a cycle in a linked list.",
    "Create a function that finds the median of two sorted arrays.",
    "Write a function that compresses a string using run-length encoding.",
    "Implement a function that rotates a matrix 90 degrees.",
    "Write a function that returns the nth Fibonacci number efficiently.",
]

CRUD_PROMPTS = [
    "Write a REST endpoint that creates a new blog post.",
    "Implement an endpoint that retrieves a user profile by ID.",
    "Create an API route that updates a product's price.",
    "Write a handler that deletes a record from the database.",
    "Implement pagination for a list endpoint that returns many items.",
    "Write an endpoint that lets a user upload a profile picture.",
    "Create a search endpoint that filters items by keyword.",
    "Implement a handler that returns all orders placed by a user.",
    "Write an API route that marks a todo item as complete.",
    "Create an endpoint that bulk-imports records from a CSV.",
    "Implement an endpoint that returns an activity feed for a user.",
    "Write a handler that archives rather than permanently deletes a record.",
    "Create an endpoint that lets an admin change another user's role.",
    "Implement a webhook handler that processes incoming events.",
    "Write an endpoint that exports a report as a downloadable file.",
]

ETL_PROMPTS = [
    "Write a script that reads a CSV file and computes per-category totals.",
    "Implement a function that normalises phone numbers in a dataset.",
    "Create a pipeline that removes duplicate rows from a data file.",
    "Write a function that joins two datasets on a common key.",
    "Implement a script that converts date strings to a uniform format.",
    "Write a function that fills in missing values in a table.",
    "Create a pipeline that reads JSON records from stdin and writes a summary CSV.",
    "Write a function that splits a large file into smaller chunks.",
    "Implement a script that counts word frequencies in a collection of text files.",
    "Write a function that validates and cleans email addresses in a list.",
    "Create a pipeline that parses log files and extracts error counts per hour.",
    "Write a script that merges monthly sales CSVs into a single annual report.",
    "Implement a function that converts XML data to a flat tabular format.",
    "Write a script that downloads data from a URL and stores it locally.",
    "Create a pipeline that anonymises personally identifiable information in a dataset.",
]

PROMPTS: list[dict] = (
    [{"id": f"auth_{i+1:02d}", "category": "auth", "prompt": p} for i, p in enumerate(AUTH_PROMPTS)]
    + [{"id": f"algo_{i+1:02d}", "category": "algorithmic", "prompt": p} for i, p in enumerate(ALGO_PROMPTS)]
    + [{"id": f"crud_{i+1:02d}", "category": "crud", "prompt": p} for i, p in enumerate(CRUD_PROMPTS)]
    + [{"id": f"etl_{i+1:02d}", "category": "etl", "prompt": p} for i, p in enumerate(ETL_PROMPTS)]
)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate pilot (prompt, code) pairs.")
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "data" / "pilot_samples.json"),
        help="Output JSON file path (default: data/pilot_samples.json)",
    )
    p.add_argument(
        "--backend",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM backend (overrides ASSUMPTION_MINER_BACKEND env var)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the LLM",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls (default: 1.0)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N prompts (useful for quick tests)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend:
        import os
        os.environ["ASSUMPTION_MINER_BACKEND"] = args.backend

    prompts = PROMPTS if args.limit is None else PROMPTS[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for entry in prompts:
        pid = entry["id"]
        prompt_text = entry["prompt"]
        category = entry["category"]

        if args.dry_run:
            log.info("[dry-run] %s | %s", pid, prompt_text[:80])
            results.append({"id": pid, "category": category, "prompt": prompt_text, "code": ""})
            continue

        log.info("Generating %s …", pid)
        try:
            code = generator.generate_code(prompt_text)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed %s: %s", pid, exc)
            results.append(
                {"id": pid, "category": category, "prompt": prompt_text, "code": "", "error": str(exc)}
            )
        else:
            results.append({"id": pid, "category": category, "prompt": prompt_text, "code": code})
            log.info("  → %d lines of code", code.count("\n") + 1)

        if args.delay > 0:
            time.sleep(args.delay)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d samples to %s", len(results), out_path)

    # Verification (per plan)
    successful = sum(1 for r in results if r.get("code"))
    log.info("Successful: %d / %d", successful, len(results))
    if successful < 50 and not args.dry_run:
        log.warning("Fewer than 50 samples generated — check API keys and retry.")


if __name__ == "__main__":
    main()
