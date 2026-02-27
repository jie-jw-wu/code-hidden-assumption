"""Tests for run_rq3.py helper functions (offline, no LLM calls)."""

import sys
from pathlib import Path

import pytest

# Make scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.run_rq3 import (
    _edit_distance,
    _normalised_edit,
    _pick_alternative,
    _pick_assumption,
    _strip_fences,
    _syntax_ok,
    aggregate,
    dry_run_update,
)


# ---------------------------------------------------------------------------
# _edit_distance
# ---------------------------------------------------------------------------

def test_edit_distance_identical():
    assert _edit_distance("abc", "abc") == 0


def test_edit_distance_empty():
    assert _edit_distance("", "abc") == 3
    assert _edit_distance("abc", "") == 3


def test_edit_distance_single_insertion():
    assert _edit_distance("ab", "abc") == 1


def test_edit_distance_single_substitution():
    assert _edit_distance("abc", "axc") == 1


def test_edit_distance_known():
    # "kitten" → "sitting" = 3
    assert _edit_distance("kitten", "sitting") == 3


# ---------------------------------------------------------------------------
# _normalised_edit
# ---------------------------------------------------------------------------

def test_normalised_edit_range():
    val = _normalised_edit("hello world", "hello python")
    assert 0.0 <= val <= 1.0


def test_normalised_edit_identical():
    assert _normalised_edit("code", "code") == 0.0


def test_normalised_edit_both_empty():
    assert _normalised_edit("", "") == 0.0


# ---------------------------------------------------------------------------
# _strip_fences
# ---------------------------------------------------------------------------

def test_strip_fences_removes_python_fence():
    fenced = "```python\ndef foo():\n    pass\n```"
    assert _strip_fences(fenced) == "def foo():\n    pass"


def test_strip_fences_no_fence_unchanged():
    code = "def foo():\n    pass"
    assert _strip_fences(code) == code


def test_strip_fences_bare_backtick():
    fenced = "```\ndef foo(): pass\n```"
    assert "def foo(): pass" in _strip_fences(fenced)


# ---------------------------------------------------------------------------
# _syntax_ok
# ---------------------------------------------------------------------------

def test_syntax_ok_valid_code():
    assert _syntax_ok("def foo():\n    return 42\n") is True


def test_syntax_ok_invalid_code():
    assert _syntax_ok("def foo(\n    return 42\n") is False


def test_syntax_ok_empty_string():
    assert _syntax_ok("") is True  # empty file is syntactically valid


def test_syntax_ok_class():
    code = "class Foo:\n    def bar(self):\n        pass\n"
    assert _syntax_ok(code) is True


# ---------------------------------------------------------------------------
# _pick_assumption / _pick_alternative
# ---------------------------------------------------------------------------

_GT = [
    {"id": "A1", "category": "T6", "severity": "high",
     "description": "MD5 hashing", "alternatives": ["bcrypt"]},
    {"id": "A2", "category": "T4", "severity": "medium",
     "description": "in-memory dict", "alternatives": ["SQLite", "Redis"]},
    {"id": "A3", "category": "T3", "severity": "low",
     "description": "returns False on failure", "alternatives": ["raise exception"]},
]


def test_pick_assumption_highest_severity():
    picked = _pick_assumption(_GT)
    assert picked["id"] == "A1"  # high > medium > low


def test_pick_assumption_empty():
    assert _pick_assumption([]) is None


def test_pick_alternative_returns_first():
    alt = _pick_alternative({"alternatives": ["bcrypt", "argon2"]})
    assert alt == "bcrypt"


def test_pick_alternative_empty_list():
    assert _pick_alternative({"alternatives": []}) is None


def test_pick_alternative_missing_key():
    assert _pick_alternative({}) is None


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_aggregate_empty():
    result = aggregate([])
    assert result["n"] == 0
    assert result["pass_rate"] is None


def test_aggregate_all_pass():
    rows = [{"syntax_ok": True, "edit_dist": 0.1, "latency_s": 0.5}] * 4
    result = aggregate(rows)
    assert result["pass_rate"] == 1.0
    assert result["mean_edit_dist"] == pytest.approx(0.1)
    assert result["mean_latency_s"] == pytest.approx(0.5)


def test_aggregate_partial_pass():
    rows = [
        {"syntax_ok": True, "edit_dist": 0.2, "latency_s": 1.0},
        {"syntax_ok": False, "edit_dist": 0.8, "latency_s": 2.0},
    ]
    result = aggregate(rows)
    assert result["pass_rate"] == 0.5
    assert result["mean_edit_dist"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# dry_run_update
# ---------------------------------------------------------------------------

def test_dry_run_update_syntax_ok():
    code = "def foo():\n    return 1\n"
    updated, latency = dry_run_update(code, "old assumption", "new assumption")
    assert _syntax_ok(updated)
    assert latency > 0


def test_dry_run_update_contains_original():
    code = "x = 42\n"
    updated, _ = dry_run_update(code, "old", "new")
    assert "x = 42" in updated


def test_dry_run_update_contains_comment():
    code = "pass\n"
    updated, _ = dry_run_update(code, "old desc", "new desc")
    assert "ASSUMPTION REVISED" in updated
    assert "old desc" in updated
    assert "new desc" in updated
