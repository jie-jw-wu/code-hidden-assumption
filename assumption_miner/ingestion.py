"""
ingestion.py — C1: Input ingestion.

Accepts (prompt, code) pairs from various sources and normalises them
into the canonical (str, str) form expected by the extractor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class CodeSample:
    """A single (prompt, code) pair ready for assumption extraction."""

    prompt: str
    code: str
    source_id: str = ""   # optional identifier (filename, dataset row id, …)


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #

def from_strings(prompt: str, code: str, source_id: str = "") -> CodeSample:
    """Wrap raw strings into a CodeSample."""
    return CodeSample(prompt=prompt.strip(), code=code.strip(), source_id=source_id)


def from_json_file(path: str | Path) -> Iterator[CodeSample]:
    """
    Load samples from a JSON file.

    Expected format (list of objects):
    [
      {"id": "...", "prompt": "...", "code": "..."},
      ...
    ]
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(records).__name__}")

    for record in records:
        yield CodeSample(
            prompt=record["prompt"].strip(),
            code=record["code"].strip(),
            source_id=str(record.get("id", "")),
        )


def from_code_file(prompt: str, code_path: str | Path, source_id: str = "") -> CodeSample:
    """Load code from a file and combine with a prompt string."""
    code = Path(code_path).read_text(encoding="utf-8")
    return from_strings(prompt, code, source_id=source_id or str(code_path))
