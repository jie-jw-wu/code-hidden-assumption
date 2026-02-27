"""Tests for baselines.py — offline (no LLM calls)."""

import pytest

from assumption_miner.baselines import (
    comment_extraction,
    _infer_category,
    _keywords_from,
    _records_from_dicts,
)
from assumption_miner.schema import AssumptionRecord


# ---------------------------------------------------------------------------
# comment_extraction
# ---------------------------------------------------------------------------

_CODE_WITH_COMMENTS = """\
import hashlib

# assumes: passwords stored as MD5 hashes
def hash_pw(pw):
    return hashlib.md5(pw.encode()).hexdigest()

def login(username, password):
    # default: no rate limiting
    hashed = hash_pw(password)
    return hashed
"""


def test_comment_extraction_finds_assume_comment():
    records = comment_extraction("Implement login", _CODE_WITH_COMMENTS)
    assert any("MD5" in r.description or "md5" in r.description.lower() for r in records)


def test_comment_extraction_finds_default_comment():
    records = comment_extraction("Implement login", _CODE_WITH_COMMENTS)
    assert any("rate" in r.description.lower() for r in records)


def test_comment_extraction_no_comments():
    code = "def add(a, b):\n    return a + b\n"
    records = comment_extraction("Add two numbers", code)
    assert records == []


def test_comment_extraction_returns_assumption_records():
    records = comment_extraction("Implement login", _CODE_WITH_COMMENTS)
    assert all(isinstance(r, AssumptionRecord) for r in records)
    assert all(r.category in ("T1", "T2", "T3", "T4", "T5", "T6") for r in records)


# ---------------------------------------------------------------------------
# _infer_category
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("passwords stored as MD5 hashes", "T6"),
    ("returns indices not values", "T2"),
    ("raises ValueError on invalid input", "T3"),
    ("stored in SQLite database", "T4"),
    ("O(n) hash map instead of O(n^2)", "T5"),
    ("assumes header row in CSV encoding", "T1"),
])
def test_infer_category(text, expected):
    assert _infer_category(text) == expected


# ---------------------------------------------------------------------------
# _keywords_from
# ---------------------------------------------------------------------------

def test_keywords_from_extracts_tokens():
    record = AssumptionRecord(
        id="A1", category="T6",
        description="Passwords hashed with MD5",
        rationale="",
        alternatives=["bcrypt", "argon2"],
    )
    kws = _keywords_from(record)
    assert "passwords" in kws
    # words ≤3 chars should be excluded (filter is len > 3)
    assert "MD5" not in kws  # length 3, excluded
    # 4-char words ARE included by the "> 3" filter
    assert "with" in kws


def test_keywords_from_deduplicates():
    record = AssumptionRecord(
        id="A1", category="T2",
        description="returns list of duplicates",
        rationale="list is idiomatic",
        alternatives=["returns list with counts"],
    )
    kws = _keywords_from(record)
    assert len(kws) == len(set(kws))


# ---------------------------------------------------------------------------
# _records_from_dicts
# ---------------------------------------------------------------------------

def test_records_from_dicts_fills_defaults():
    items = [{"id": "A1", "category": "T5", "description": "O(n) hash map", "rationale": ""}]
    records = _records_from_dicts(items)
    assert len(records) == 1
    assert records[0].severity == "medium"
    assert records[0].confidence == 0.5
    assert records[0].alternatives == []


def test_records_from_dicts_invalid_category_gets_default():
    items = [{"id": "A1", "category": "BADCAT", "description": "x", "rationale": ""}]
    records = _records_from_dicts(items)
    assert records[0].category == "T2"


def test_records_from_dicts_skips_non_dict():
    items = [None, "string", {"id": "A1", "category": "T1", "description": "d", "rationale": ""}]
    records = _records_from_dicts(items)
    assert len(records) == 1
