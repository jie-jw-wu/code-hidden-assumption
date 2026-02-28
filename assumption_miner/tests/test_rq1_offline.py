"""Tests for scripts/run_rq1_offline.py evaluation helpers."""

import sys
from pathlib import Path

import pytest

# make scripts importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from run_rq1_offline import _tokenize, _jaccard, _matches, _compute_metrics, _category_metrics


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    tokens = _tokenize("Returns empty list on failure")
    assert "returns" in tokens
    assert "list" in tokens
    assert "on" not in tokens   # ≤2 chars filtered


def test_tokenize_lowercases():
    assert "returns" in _tokenize("RETURNS")


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    s = {"a", "b", "c"}
    assert _jaccard(s, s) == 1.0


def test_jaccard_disjoint():
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial():
    a = {"cat", "dog", "fish"}
    b = {"cat", "dog", "bird"}
    assert abs(_jaccard(a, b) - 0.5) < 0.01


def test_jaccard_both_empty():
    assert _jaccard(set(), set()) == 1.0


# ---------------------------------------------------------------------------
# _matches
# ---------------------------------------------------------------------------

_GT = {
    "category": "T3",
    "description": "Returns empty list when no result found",
    "alternatives": ["return None", "raise ValueError"],
}

_PRED_GOOD = {
    "category": "T3",
    "description": "Returns empty list [] when no result is found",
    "confidence": 0.5,
    "alternatives": [],
}

_PRED_WRONG_CAT = {
    "category": "T2",
    "description": "Returns empty list when no result found",
    "confidence": 0.5,
    "alternatives": [],
}

_PRED_VIA_ALT = {
    "category": "T3",
    "description": "returns none on failure",
    "confidence": 0.5,
    "alternatives": [],
}


def test_matches_exact_category_and_high_jaccard():
    assert _matches(_PRED_GOOD, _GT, threshold=0.15)


def test_matches_wrong_category_fails():
    assert not _matches(_PRED_WRONG_CAT, _GT, threshold=0.15)


def test_matches_via_alternative_keyword():
    # alt = "raise ValueError" — pred must contain both "raise" and "valueerror"
    pred = {"category": "T3", "description": "raise ValueError when nothing found",
            "confidence": 0.5, "alternatives": []}
    # jaccard is low vs "Returns empty list when no result found" but alt tokens match
    assert _matches(pred, _GT, threshold=0.99)


def test_matches_below_threshold():
    pred = {"category": "T3", "description": "xyz abc def ghi", "confidence": 0.5, "alternatives": []}
    assert not _matches(pred, _GT, threshold=0.15)


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect():
    preds = [{"category": "T3", "description": "Returns empty list when no result found",
              "confidence": 0.9, "alternatives": []}]
    gts = [{"category": "T3", "description": "Returns empty list when no result found",
             "alternatives": []}]
    m = _compute_metrics(preds, gts, threshold=0.15)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_compute_metrics_no_preds():
    gts = [{"category": "T3", "description": "Returns empty list", "alternatives": []}]
    m = _compute_metrics([], gts, threshold=0.15)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_compute_metrics_no_gts():
    preds = [{"category": "T3", "description": "x", "confidence": 0.5, "alternatives": []}]
    m = _compute_metrics(preds, [], threshold=0.15)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_compute_metrics_partial():
    preds = [
        {"category": "T3", "description": "Returns empty list when no result found",
         "confidence": 0.9, "alternatives": []},
        {"category": "T3", "description": "completely unrelated xyz abc",
         "confidence": 0.3, "alternatives": []},
    ]
    gts = [{"category": "T3", "description": "Returns empty list when no result found",
             "alternatives": []}]
    m = _compute_metrics(preds, gts, threshold=0.15)
    assert m["precision"] == 0.5   # 1 of 2 preds matched
    assert m["recall"] == 1.0      # 1 of 1 GT matched


# ---------------------------------------------------------------------------
# _category_metrics
# ---------------------------------------------------------------------------

def test_category_metrics_groups_correctly():
    preds = [
        {"category": "T3", "description": "Returns empty list when no result found",
         "confidence": 0.9, "alternatives": []},
        {"category": "T2", "description": "returns a dict",
         "confidence": 0.5, "alternatives": []},
    ]
    gts = [
        {"category": "T3", "description": "Returns empty list when no result found",
         "alternatives": []},
        {"category": "T2", "description": "returns a dict mapping keys to values",
         "alternatives": []},
    ]
    cats = _category_metrics(preds, gts, threshold=0.15)
    assert "T3" in cats
    assert "T2" in cats
    assert cats["T3"]["n_gt"] == 1
    assert cats["T2"]["n_pred"] == 1
