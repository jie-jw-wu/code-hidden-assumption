"""Tests for run_rq2_ablation.py helpers."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_rq2_ablation import _line_iou, _best_iou, _ABLATIONS, _FULL, _ORIGINAL


# ---------------------------------------------------------------------------
# _line_iou
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_start,p_end,g_start,g_end,expected", [
    (1, 5, 1, 5, 1.0),          # perfect overlap
    (1, 5, 6, 10, 0.0),         # no overlap
    (1, 5, 3, 7, 3 / 7),        # partial overlap: inter=3, union=7
    (3, 3, 3, 3, 1.0),          # single-line exact match
    (1, 10, 5, 5, 1 / 10),      # GT point inside large pred
    (5, 5, 1, 10, 1 / 10),      # Pred point inside large GT
])
def test_line_iou(p_start, p_end, g_start, g_end, expected):
    result = _line_iou(p_start, p_end, g_start, g_end)
    assert abs(result - expected) < 1e-9


def test_line_iou_zero_for_adjacent():
    """Lines 1–3 and 4–6 are adjacent but do not overlap."""
    assert _line_iou(1, 3, 4, 6) == 0.0


def test_line_iou_symmetric():
    """IoU should be symmetric."""
    assert _line_iou(2, 5, 4, 8) == _line_iou(4, 8, 2, 5)


# ---------------------------------------------------------------------------
# _best_iou
# ---------------------------------------------------------------------------

class _FakeRef:
    """Minimal stand-in for CodeRef used in _best_iou."""
    def __init__(self, start, end):
        self.start_line = start
        self.end_line = end


def test_best_iou_empty_pred():
    gt = [{"start_line": 1, "end_line": 3}]
    assert _best_iou([], gt) == 0.0


def test_best_iou_empty_gt():
    pred = [_FakeRef(1, 3)]
    assert _best_iou(pred, []) == 0.0


def test_best_iou_exact_match():
    pred = [_FakeRef(5, 7)]
    gt = [{"start_line": 5, "end_line": 7}]
    assert _best_iou(pred, gt) == pytest.approx(1.0)


def test_best_iou_picks_best_pair():
    """With multiple pred refs, best_iou should return the max IoU."""
    pred = [_FakeRef(1, 2), _FakeRef(5, 7)]   # second one matches GT exactly
    gt = [{"start_line": 5, "end_line": 7}]
    assert _best_iou(pred, gt) == pytest.approx(1.0)


def test_best_iou_partial():
    """Partial overlap case: pred=[1,5], gt=[3,7] → inter=3(3-5), union=7(1-7)."""
    pred = [_FakeRef(1, 5)]
    gt = [{"start_line": 3, "end_line": 7}]
    assert abs(_best_iou(pred, gt) - 3 / 7) < 1e-6


# ---------------------------------------------------------------------------
# Configuration structure
# ---------------------------------------------------------------------------

def test_ablations_contains_full_and_original():
    assert "full" in _ABLATIONS
    assert "original" in _ABLATIONS


def test_full_has_all_categories():
    for cat in ("T1", "T2", "T3", "T4", "T5", "T6"):
        assert cat in _FULL, f"{cat} missing from _FULL"


def test_original_has_all_categories():
    for cat in ("T1", "T2", "T3", "T4", "T5", "T6"):
        assert cat in _ORIGINAL, f"{cat} missing from _ORIGINAL"


def test_full_t3_has_call_and_with_statement():
    assert "call" in _FULL["T3"]
    assert "with_statement" in _FULL["T3"]


def test_original_t3_lacks_call():
    """Original config should NOT have 'call' in T3."""
    assert "call" not in _ORIGINAL["T3"]


def test_full_t5_has_assignment():
    assert "assignment" in _FULL["T5"]


def test_ablations_all_have_six_categories():
    for name, config in _ABLATIONS.items():
        assert len(config) == 6, f"{name} has {len(config)} categories, expected 6"
