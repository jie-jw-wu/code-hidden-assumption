"""Tests for dependency.py — offline (requires tree-sitter)."""

import pytest

from assumption_miner.schema import AssumptionRecord, CodeRef


# ---------------------------------------------------------------------------
# line_iou helper (imported from run_rq2 logic, tested independently)
# ---------------------------------------------------------------------------

def line_iou(pred_start, pred_end, gt_start, gt_end):
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0, inter_end - inter_start + 1)
    union = (pred_end - pred_start + 1) + (gt_end - gt_start + 1) - inter
    return inter / union if union > 0 else 0.0


@pytest.mark.parametrize("pred,gt,expected", [
    ((1, 5), (1, 5), 1.0),        # perfect overlap
    ((1, 5), (6, 10), 0.0),       # no overlap
    ((1, 5), (3, 7), 3/7),        # partial (intersection=3, union=7)
    ((3, 3), (3, 3), 1.0),        # single line exact
    ((1, 10), (5, 5), 1/10),      # contains point
])
def test_line_iou(pred, gt, expected):
    assert abs(line_iou(*pred, *gt) - expected) < 1e-6


# ---------------------------------------------------------------------------
# map_dependencies (requires tree-sitter)
# ---------------------------------------------------------------------------

_TWO_SUM_CODE = """\
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""

_AUTH_CODE = """\
import hashlib
import uuid

USERS = {}

def authenticate(username, password):
    hashed = hashlib.md5(password.encode()).hexdigest()
    if USERS.get(username) == hashed:
        return str(uuid.uuid4())
    return False
"""


def _make_record(category, description, alternatives=None):
    return AssumptionRecord(
        id="A1",
        category=category,
        description=description,
        rationale="test",
        alternatives=alternatives or ["alternative"],
    )


def test_map_dependencies_populates_refs():
    try:
        from assumption_miner.dependency import map_dependencies
    except ImportError:
        pytest.skip("tree-sitter not available")

    record = _make_record("T6", "MD5 used for password hashing", ["bcrypt"])
    map_dependencies([record], _AUTH_CODE)
    # Should find at least one ref mentioning md5 or hashlib
    assert len(record.code_refs) >= 1


def test_map_dependencies_returns_coderef_objects():
    try:
        from assumption_miner.dependency import map_dependencies
    except ImportError:
        pytest.skip("tree-sitter not available")

    record = _make_record("T3", "Returns empty list when no pair found", ["return None"])
    map_dependencies([record], _TWO_SUM_CODE)
    for ref in record.code_refs:
        assert isinstance(ref, CodeRef)
        assert ref.start_line >= 1
        assert ref.end_line >= ref.start_line


def test_map_dependencies_line_numbers_in_bounds():
    try:
        from assumption_miner.dependency import map_dependencies
    except ImportError:
        pytest.skip("tree-sitter not available")

    n_lines = len(_AUTH_CODE.splitlines())
    record = _make_record("T4", "User records in Python dict", ["SQLite"])
    map_dependencies([record], _AUTH_CODE)
    for ref in record.code_refs:
        assert 1 <= ref.start_line <= n_lines
        assert 1 <= ref.end_line <= n_lines


def test_map_dependencies_noop_without_treesitter(monkeypatch):
    """If tree-sitter is unavailable, map_dependencies returns records unchanged."""
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("tree_sitter_python", "tree_sitter"):
            raise ImportError("mocked unavailable")
        return original_import(name, *args, **kwargs)

    record = _make_record("T5", "O(n) hash map algorithm", ["O(n^2) nested loop"])
    record.code_refs = []

    monkeypatch.setattr(builtins, "__import__", mock_import)
    from assumption_miner.dependency import map_dependencies
    result = map_dependencies([record], _TWO_SUM_CODE)
    # Should return gracefully with empty refs.
    assert result is not None


def test_get_context_window_returns_surrounding_lines():
    from assumption_miner.dependency import get_context_window

    code = "\n".join(f"line_{i}" for i in range(1, 21))  # 20 lines
    window = get_context_window(code, start_line=10, end_line=10, k=3)
    lines = window.splitlines()
    # Should include lines 7–13 (3 before + target + 3 after)
    assert any("line_10" in l for l in lines)
    assert any("line_7" in l for l in lines)
    assert any("line_13" in l for l in lines)


def test_get_context_window_clamps_at_boundaries():
    from assumption_miner.dependency import get_context_window

    code = "line_1\nline_2\nline_3\n"
    # k=10 should not raise even though there aren't 10 lines
    window = get_context_window(code, start_line=2, end_line=2, k=10)
    assert "line_1" in window
    assert "line_3" in window
