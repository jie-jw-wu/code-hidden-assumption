"""
select_benchmark_tasks.py — C1: Filter HumanEval/MBPP for ambiguous tasks.

Downloads (or reads cached) HumanEval and MBPP datasets, applies ambiguity
heuristics, and selects up to 50 tasks from each plus up to 50 novel prompts
to form the 150-task benchmark.

Heuristics for "ambiguous" tasks:
  - Prompt length < MAX_WORDS words (short prompts leave more underdetermined)
  - No explicit type annotations in the docstring/prompt
  - No explicit edge-case specification in the prompt

Output: data/benchmark.json  (list of dicts with id, source, prompt, notes)

Usage:
    python scripts/select_benchmark_tasks.py
    python scripts/select_benchmark_tasks.py --humaneval 50 --mbpp 50 --novel 50
    python scripts/select_benchmark_tasks.py --no-download --cache-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUT = REPO_ROOT / "data" / "benchmark.json"
CACHE_DIR = REPO_ROOT / "data" / "raw"

MAX_WORDS = 80          # prompts longer than this are likely well-specified
MIN_PROMPT_WORDS = 10   # sanity filter


# --------------------------------------------------------------------------- #
# Novel prompts (hand-authored, highly ambiguous)                              #
# --------------------------------------------------------------------------- #

NOVEL_PROMPTS: list[dict] = [
    {"id": "novel_01", "source": "novel",
     "prompt": "Write a function that sends a notification when an event occurs.",
     "notes": "Unspecified: channel (email/SMS/push), timing, retry policy."},
    {"id": "novel_02", "source": "novel",
     "prompt": "Implement a cache for expensive function calls.",
     "notes": "Unspecified: eviction policy, size limit, invalidation, thread-safety."},
    {"id": "novel_03", "source": "novel",
     "prompt": "Write a function that processes a batch of items.",
     "notes": "Unspecified: batch size, error handling per item, parallelism."},
    {"id": "novel_04", "source": "novel",
     "prompt": "Create a rate limiter for API calls.",
     "notes": "Unspecified: algorithm (token bucket/sliding window), scope, storage."},
    {"id": "novel_05", "source": "novel",
     "prompt": "Implement a retry mechanism for network requests.",
     "notes": "Unspecified: max retries, backoff strategy, which errors to retry."},
    {"id": "novel_06", "source": "novel",
     "prompt": "Write a function that schedules a task to run later.",
     "notes": "Unspecified: scheduler library, persistence, timezone handling."},
    {"id": "novel_07", "source": "novel",
     "prompt": "Implement a simple event bus.",
     "notes": "Unspecified: sync vs async dispatch, error handling, weak refs."},
    {"id": "novel_08", "source": "novel",
     "prompt": "Write a function that exports data to a report.",
     "notes": "Unspecified: format (PDF/CSV/Excel), template, locale."},
    {"id": "novel_09", "source": "novel",
     "prompt": "Create a function that backs up a directory.",
     "notes": "Unspecified: compression, incremental vs full, naming scheme."},
    {"id": "novel_10", "source": "novel",
     "prompt": "Write a function that monitors a value and alerts when it changes.",
     "notes": "Unspecified: polling interval, alert mechanism, threshold."},
    {"id": "novel_11", "source": "novel",
     "prompt": "Implement a plugin system that loads modules at runtime.",
     "notes": "Unspecified: discovery mechanism, versioning, sandboxing."},
    {"id": "novel_12", "source": "novel",
     "prompt": "Write a function that deduplicates a collection of records.",
     "notes": "Unspecified: equality definition, which duplicate to keep, ordering."},
    {"id": "novel_13", "source": "novel",
     "prompt": "Create a function that generates a unique identifier.",
     "notes": "Unspecified: format (UUID/snowflake/nanoid), collision avoidance."},
    {"id": "novel_14", "source": "novel",
     "prompt": "Implement a circuit breaker for a service call.",
     "notes": "Unspecified: failure threshold, recovery timeout, half-open logic."},
    {"id": "novel_15", "source": "novel",
     "prompt": "Write a function that converts data between two formats.",
     "notes": "Unspecified: source/target schemas, lossy fields, validation."},
    {"id": "novel_16", "source": "novel",
     "prompt": "Create an endpoint that accepts a file upload.",
     "notes": "Unspecified: size limit, allowed types, storage location, virus scan."},
    {"id": "novel_17", "source": "novel",
     "prompt": "Write a function that synchronises two directories.",
     "notes": "Unspecified: conflict resolution, deletion policy, symlink handling."},
    {"id": "novel_18", "source": "novel",
     "prompt": "Implement a simple key-value store.",
     "notes": "Unspecified: persistence, TTL, concurrency model."},
    {"id": "novel_19", "source": "novel",
     "prompt": "Write a function that parses configuration from the environment.",
     "notes": "Unspecified: prefix, type coercion, required vs optional, defaults."},
    {"id": "novel_20", "source": "novel",
     "prompt": "Create a logging decorator for function calls.",
     "notes": "Unspecified: log level, format, argument redaction, async support."},
    {"id": "novel_21", "source": "novel",
     "prompt": "Write a function that fetches data from an external API.",
     "notes": "Unspecified: auth, pagination, timeout, caching."},
    {"id": "novel_22", "source": "novel",
     "prompt": "Implement a function that validates user input.",
     "notes": "Unspecified: rules, error message format, sanitisation vs rejection."},
    {"id": "novel_23", "source": "novel",
     "prompt": "Create a health check endpoint for a web service.",
     "notes": "Unspecified: checks included, response format, auth required."},
    {"id": "novel_24", "source": "novel",
     "prompt": "Write a function that reads configuration from a file.",
     "notes": "Unspecified: format (JSON/YAML/TOML), merge strategy, hot reload."},
    {"id": "novel_25", "source": "novel",
     "prompt": "Implement a job queue that processes tasks in the background.",
     "notes": "Unspecified: queue backend, worker count, retry/dead-letter policy."},
    {"id": "novel_26", "source": "novel",
     "prompt": "Write a function that aggregates metrics from multiple sources.",
     "notes": "Unspecified: aggregation function, time window, outlier handling."},
    {"id": "novel_27", "source": "novel",
     "prompt": "Create a function that checks system resources.",
     "notes": "Unspecified: which resources, thresholds, output format."},
    {"id": "novel_28", "source": "novel",
     "prompt": "Write a function that compresses and decompresses data.",
     "notes": "Unspecified: algorithm (gzip/zstd/lz4), level, streaming support."},
    {"id": "novel_29", "source": "novel",
     "prompt": "Implement a feature flag system.",
     "notes": "Unspecified: flag source, per-user targeting, override mechanism."},
    {"id": "novel_30", "source": "novel",
     "prompt": "Write a function that paginate through results.",
     "notes": "Unspecified: cursor vs offset, page size default, total count included."},
    {"id": "novel_31", "source": "novel",
     "prompt": "Create a function that encrypts sensitive data.",
     "notes": "Unspecified: algorithm, key management, encoding of ciphertext."},
    {"id": "novel_32", "source": "novel",
     "prompt": "Write a function that performs a database migration.",
     "notes": "Unspecified: library, rollback strategy, idempotency guarantee."},
    {"id": "novel_33", "source": "novel",
     "prompt": "Implement a simple message queue consumer.",
     "notes": "Unspecified: broker, ack strategy, concurrency, DLQ."},
    {"id": "novel_34", "source": "novel",
     "prompt": "Write a function that generates a report for a date range.",
     "notes": "Unspecified: timezone, inclusive/exclusive bounds, aggregation level."},
    {"id": "novel_35", "source": "novel",
     "prompt": "Create a middleware function that logs HTTP requests.",
     "notes": "Unspecified: fields logged, PII redaction, format, async vs sync."},
    {"id": "novel_36", "source": "novel",
     "prompt": "Implement a function that checks if a user has a permission.",
     "notes": "Unspecified: permission model (RBAC/ABAC), inheritance, deny rules."},
    {"id": "novel_37", "source": "novel",
     "prompt": "Write a function that imports data from an external system.",
     "notes": "Unspecified: protocol, delta vs full sync, conflict resolution."},
    {"id": "novel_38", "source": "novel",
     "prompt": "Create a function that tracks changes to an object over time.",
     "notes": "Unspecified: storage, diff granularity, who/when attribution."},
    {"id": "novel_39", "source": "novel",
     "prompt": "Write a function that processes events from a stream.",
     "notes": "Unspecified: stream source, ordering guarantees, at-least-once handling."},
    {"id": "novel_40", "source": "novel",
     "prompt": "Implement a search function over a collection.",
     "notes": "Unspecified: algorithm, ranking, fuzzy matching, field weights."},
    {"id": "novel_41", "source": "novel",
     "prompt": "Write a function that converts currency amounts.",
     "notes": "Unspecified: rate source, rounding mode, historical vs live rates."},
    {"id": "novel_42", "source": "novel",
     "prompt": "Create a function that manages user sessions.",
     "notes": "Unspecified: storage, expiry, sliding vs fixed window."},
    {"id": "novel_43", "source": "novel",
     "prompt": "Implement a function that polls for status updates.",
     "notes": "Unspecified: interval, timeout, callback mechanism."},
    {"id": "novel_44", "source": "novel",
     "prompt": "Write a function that handles file format conversion.",
     "notes": "Unspecified: lossless vs lossy, metadata preservation, streaming."},
    {"id": "novel_45", "source": "novel",
     "prompt": "Create a function that manages connection pooling.",
     "notes": "Unspecified: pool size, idle timeout, health checks."},
    {"id": "novel_46", "source": "novel",
     "prompt": "Write a function that generates test data.",
     "notes": "Unspecified: schema, randomness source, referential integrity."},
    {"id": "novel_47", "source": "novel",
     "prompt": "Implement a function that detects anomalies in a time series.",
     "notes": "Unspecified: algorithm, sensitivity, seasonality handling."},
    {"id": "novel_48", "source": "novel",
     "prompt": "Write a function that manages configuration overrides.",
     "notes": "Unspecified: precedence order, scope (user/org/global), persistence."},
    {"id": "novel_49", "source": "novel",
     "prompt": "Create a function that performs A/B test assignment.",
     "notes": "Unspecified: bucketing strategy, stickiness, experiment exclusion."},
    {"id": "novel_50", "source": "novel",
     "prompt": "Write a function that summarises a document.",
     "notes": "Unspecified: length target, model, extractive vs abstractive."},
]


# --------------------------------------------------------------------------- #
# Ambiguity heuristics                                                         #
# --------------------------------------------------------------------------- #

_TYPE_ANNOTATION_RE = re.compile(
    r"(->|\bList\[|\bDict\[|\bOptional\[|\bTuple\[|\bint\b|\bstr\b|\bfloat\b|\bbool\b)",
    re.IGNORECASE,
)
_EDGE_CASE_RE = re.compile(
    r"\b(if (empty|none|null|zero|negative)|raise|throw|exception|error|edge case)\b",
    re.IGNORECASE,
)


def is_ambiguous(prompt: str) -> bool:
    words = prompt.split()
    if len(words) < MIN_PROMPT_WORDS or len(words) > MAX_WORDS:
        return False
    if _TYPE_ANNOTATION_RE.search(prompt):
        return False
    if _EDGE_CASE_RE.search(prompt):
        return False
    return True


# --------------------------------------------------------------------------- #
# Dataset loaders                                                              #
# --------------------------------------------------------------------------- #

def load_humaneval(cache_dir: Path, download: bool) -> list[dict]:
    """Load HumanEval from cache or download via datasets library."""
    cache_file = cache_dir / "humaneval.json"
    if cache_file.exists():
        log.info("Loading HumanEval from cache: %s", cache_file)
        with cache_file.open(encoding="utf-8") as fh:
            return json.load(fh)

    if not download:
        log.warning("HumanEval cache not found and --no-download set; skipping.")
        return []

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.warning("datasets library not installed (pip install datasets); skipping HumanEval.")
        return []

    log.info("Downloading HumanEval …")
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    records = [
        {
            "id": f"humaneval_{r['task_id']}",
            "source": "humaneval",
            "prompt": r["prompt"].strip(),
            "canonical_solution": r.get("canonical_solution", ""),
        }
        for r in ds
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
    log.info("Cached %d HumanEval tasks to %s", len(records), cache_file)
    return records


def load_mbpp(cache_dir: Path, download: bool) -> list[dict]:
    """Load MBPP from cache or download via datasets library."""
    cache_file = cache_dir / "mbpp.json"
    if cache_file.exists():
        log.info("Loading MBPP from cache: %s", cache_file)
        with cache_file.open(encoding="utf-8") as fh:
            return json.load(fh)

    if not download:
        log.warning("MBPP cache not found and --no-download set; skipping.")
        return []

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.warning("datasets library not installed (pip install datasets); skipping MBPP.")
        return []

    log.info("Downloading MBPP …")
    ds = load_dataset("mbpp", split="train", trust_remote_code=True)
    records = [
        {
            "id": f"mbpp_{r['task_id']}",
            "source": "mbpp",
            "prompt": r["text"].strip(),
            "canonical_solution": r.get("code", ""),
        }
        for r in ds
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
    log.info("Cached %d MBPP tasks to %s", len(records), cache_file)
    return records


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select 150-task benchmark from HumanEval/MBPP/novel.")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--humaneval", type=int, default=50, metavar="N",
                   help="Max tasks from HumanEval (default: 50)")
    p.add_argument("--mbpp", type=int, default=50, metavar="N",
                   help="Max tasks from MBPP (default: 50)")
    p.add_argument("--novel", type=int, default=50, metavar="N",
                   help="Max novel prompts to include (default: 50)")
    p.add_argument("--no-download", action="store_true",
                   help="Do not download datasets; use cache only")
    p.add_argument("--cache-dir", default=str(CACHE_DIR),
                   help=f"Cache directory for raw datasets (default: {CACHE_DIR})")
    p.add_argument("--all", action="store_true",
                   help="Include all tasks (no ambiguity filter)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    he_tasks = load_humaneval(cache_dir, download=not args.no_download)
    mbpp_tasks = load_mbpp(cache_dir, download=not args.no_download)

    def filter_tasks(tasks: list[dict], limit: int) -> list[dict]:
        if args.all:
            return tasks[:limit]
        filtered = [t for t in tasks if is_ambiguous(t["prompt"])]
        log.info("  Ambiguous: %d / %d", len(filtered), len(tasks))
        return filtered[:limit]

    log.info("Filtering HumanEval (%d tasks) …", len(he_tasks))
    selected_he = filter_tasks(he_tasks, args.humaneval)

    log.info("Filtering MBPP (%d tasks) …", len(mbpp_tasks))
    selected_mbpp = filter_tasks(mbpp_tasks, args.mbpp)

    selected_novel = NOVEL_PROMPTS[: args.novel]

    benchmark = selected_he + selected_mbpp + selected_novel

    # Add annotation placeholder fields
    for task in benchmark:
        task.setdefault("ground_truth_assumptions", [])  # filled by human annotators
        task.setdefault("annotation_status", "pending")

    out_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Benchmark written to %s", out_path)
    log.info("  HumanEval: %d", len(selected_he))
    log.info("  MBPP:      %d", len(selected_mbpp))
    log.info("  Novel:     %d", len(selected_novel))
    log.info("  Total:     %d", len(benchmark))

    if len(benchmark) < 100:
        log.warning(
            "Benchmark has fewer than 100 tasks. "
            "Consider reducing MAX_WORDS or using --all to include more tasks."
        )


if __name__ == "__main__":
    main()
