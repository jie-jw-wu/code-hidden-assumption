"""Tests for extractor.py — prompt building and response parsing (offline)."""

import json
import pytest

from assumption_miner.extractor import _build_user_message, _parse_response
from assumption_miner.schema import AssumptionRecord


_SAMPLE_RESPONSE = """
Some reasoning text here...

[RECORDS]
[
  {
    "id": "A1",
    "category": "T6",
    "description": "MD5 used for password hashing.",
    "rationale": "Default in hashlib.",
    "alternatives": ["bcrypt", "argon2"],
    "confidence": 0.9,
    "severity": "high"
  },
  {
    "id": "A2",
    "category": "T4",
    "description": "User records stored in a Python dict.",
    "rationale": "Simple in-memory store requires no external dependency.",
    "alternatives": ["SQLite", "PostgreSQL"],
    "confidence": 0.85,
    "severity": "medium"
  }
]
"""


def test_parse_response_with_tag():
    records = _parse_response(_SAMPLE_RESPONSE)
    assert len(records) == 2
    assert records[0].id == "A1"
    assert records[1].category == "T4"


def test_parse_response_fallback_no_tag():
    """Response without [RECORDS] tag — fallback JSON extraction."""
    json_only = '[{"id":"A1","category":"T1","description":"d","rationale":"r","alternatives":["x"],"confidence":0.5,"severity":"low"}]'
    records = _parse_response(json_only)
    assert len(records) == 1
    assert records[0].id == "A1"


def test_parse_response_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_response("[RECORDS]\nnot-json{{{")


def test_parse_response_skips_malformed_record():
    bad = '[RECORDS]\n[{"id": "A1"}, {"id": "A2", "category": "T1", "description": "d", "rationale": "r", "alternatives": ["x"], "confidence": 0.5, "severity": "low"}]'
    # First record is missing required fields — should be skipped with a warning.
    records = _parse_response(bad)
    assert len(records) == 1
    assert records[0].id == "A2"


def test_build_user_message_contains_prompt_and_code():
    msg = _build_user_message("Do X", "def x(): pass")
    assert "Do X" in msg
    assert "def x(): pass" in msg
    assert "[RECORDS]" in msg
