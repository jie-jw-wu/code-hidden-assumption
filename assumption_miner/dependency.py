"""
dependency.py — C4: Assumption–code dependency mapper.

Uses tree-sitter to parse Python code and links each AssumptionRecord
to the AST nodes (and source lines) that implement it.

Requires:
  pip install tree-sitter tree-sitter-python
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .schema import AssumptionRecord, CodeRef

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def map_dependencies(
    records: list[AssumptionRecord],
    code: str,
    filename: str = "<generated>",
    use_llm: bool = False,
) -> list[AssumptionRecord]:
    """
    Populate the ``code_refs`` field of each AssumptionRecord in *records*
    by parsing *code* with tree-sitter.

    Two modes:
      use_llm=False (default): heuristic keyword-overlap + AST narrowing pass.
      use_llm=True:  LLM-guided localisation (§3.4) — asks the LLM to describe
                     the code region, then refines with AST narrowing.
                     Requires ASSUMPTION_MINER_BACKEND env var and API key.

    Returns the same list (mutated in place) for convenience.
    """
    try:
        tree, node_map = _parse(code)
    except ImportError:
        logger.warning(
            "tree-sitter not available; skipping dependency mapping. "
            "Install with: pip install tree-sitter tree-sitter-python"
        )
        return records

    lines = code.splitlines()

    for record in records:
        if use_llm:
            record.code_refs = _find_refs_llm(record, tree, code, lines, filename, node_map)
        else:
            record.code_refs = _find_refs(record, tree, code, filename, node_map)

    return records


# --------------------------------------------------------------------------- #
# tree-sitter helpers                                                          #
# --------------------------------------------------------------------------- #

def _parse(code: str):
    """Return (tree, node_map) where node_map maps node type → list of nodes."""
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(code.encode())
    node_map = _build_node_map(tree.root_node)
    return tree, node_map


def _build_node_map(root) -> dict[str, list]:
    """Walk the AST and index nodes by type."""
    index: dict[str, list] = {}

    def walk(node):
        index.setdefault(node.type, []).append(node)
        for child in node.children:
            walk(child)

    walk(root)
    return index


# Mapping from assumption category to the AST node types most likely
# to implement that category's concerns.
_CATEGORY_NODE_TYPES: dict[str, list[str]] = {
    "T1": ["function_definition", "if_statement", "assert_statement",
           "call", "with_statement"],                                     # input validation
    "T2": ["return_statement", "typed_parameter", "type",
           "function_definition"],                                        # return type
    "T3": ["except_clause", "raise_statement", "if_statement",
           "return_statement"],                                           # error handling
    "T4": ["import_statement", "import_from_statement",
           "assignment", "call", "with_statement"],                       # persistence
    "T5": ["function_definition", "for_statement", "while_statement",
           "if_statement", "class_definition"],                           # algorithm
    "T6": ["import_statement", "call", "assignment",
           "function_definition"],                                        # security
}

# Maximum span (lines) for a matched AST node before we apply keyword narrowing.
_NARROW_THRESHOLD = 3


def _find_refs(
    record: AssumptionRecord,
    tree,
    code: str,
    filename: str,
    node_map: dict,
) -> list[CodeRef]:
    """
    Two-pass heuristic matching assumption descriptions to code regions.

    Pass 1 — Candidate identification: find AST nodes of category-relevant
      types whose text overlaps with keywords from the assumption description.

    Pass 2 — AST refinement (§3.4): if a matched node spans more than
      _NARROW_THRESHOLD lines, narrow it to the contiguous block of lines
      within that node that contain the matching keywords.  This yields
      tighter CodeRefs with higher IoU against expert-annotated line ranges.
    """
    keywords = _keywords(record)
    candidate_types = _CATEGORY_NODE_TYPES.get(record.category, list(node_map.keys()))
    lines = code.splitlines()

    refs: list[CodeRef] = []
    seen: set[tuple[int, int]] = set()

    for node_type in candidate_types:
        for node in node_map.get(node_type, []):
            node_text = node.text.decode(errors="replace").lower()
            if not any(kw in node_text for kw in keywords):
                continue

            raw_start = node.start_point[0] + 1   # 1-indexed
            raw_end = node.end_point[0] + 1
            span = raw_end - raw_start + 1

            if span <= _NARROW_THRESHOLD:
                # Node is already tight; use as-is.
                start, end = raw_start, raw_end
            else:
                # Pass 2: keyword-narrow within the node's line range.
                start, end = _narrow_to_keyword_lines(
                    lines, raw_start, raw_end, keywords
                )
                if start is None:
                    # No keyword lines found inside node; fall back to full span.
                    start, end = raw_start, raw_end

            key = (start, end)
            if key not in seen:
                seen.add(key)
                refs.append(
                    CodeRef(
                        file=filename,
                        start_line=start,
                        end_line=end,
                        ast_node=node_type,
                    )
                )

    return refs


def _narrow_to_keyword_lines(
    lines: list[str],
    start: int,
    end: int,
    keywords: list[str],
) -> tuple[Optional[int], Optional[int]]:
    """
    Within the line range [start, end] (1-indexed, inclusive), find the
    contiguous block spanning all lines that contain at least one keyword.

    Returns (first_match_line, last_match_line) or (None, None) if no line
    matches any keyword.
    """
    matching: list[int] = []
    for lineno in range(start, end + 1):
        line_text = lines[lineno - 1].lower() if lineno <= len(lines) else ""
        if any(kw in line_text for kw in keywords):
            matching.append(lineno)

    if not matching:
        return None, None
    return matching[0], matching[-1]


def _keywords(record: AssumptionRecord) -> list[str]:
    """Extract lowercase keywords from description + alternatives."""
    tokens: list[str] = []
    for text in [record.description] + record.alternatives:
        tokens.extend(
            w.lower().strip(".,;:'\"()[]{}") for w in text.split() if len(w) > 3
        )
    return list(set(tokens))


# --------------------------------------------------------------------------- #
# LLM-guided localisation pass (§3.4 of paper)                                #
# --------------------------------------------------------------------------- #

_LLM_LOC_SYSTEM = (
    "You are a code analysis assistant. "
    "Given an assumption about a code snippet and the code itself, "
    "identify the exact line numbers that implement (or are most responsible for) "
    "that assumption."
)

_LLM_LOC_USER = """\
### Assumption
{description}

### Code (lines numbered from 1)
{numbered_code}

### Task
Identify the line range [start_line, end_line] (inclusive, 1-indexed) in the code
that most directly implements or embodies this assumption.

Respond with ONLY a JSON object:
{{"start_line": <int>, "end_line": <int>, "confidence": <float 0-1>}}
"""


def _find_refs_llm(
    record: AssumptionRecord,
    tree,
    code: str,
    lines: list[str],
    filename: str,
    node_map: dict,
) -> list[CodeRef]:
    """
    LLM-guided localisation (§3.4):
      1. Ask LLM to identify start_line/end_line for the assumption.
      2. Use AST narrowing to snap to the minimal enclosing AST node.
      3. Fall back to heuristic pass if LLM call fails.
    """
    numbered = "\n".join(f"{i+1:3d}  {l}" for i, l in enumerate(lines))
    user_msg = _LLM_LOC_USER.format(
        description=record.description,
        numbered_code=numbered,
    )

    try:
        raw = _call_llm_loc(_LLM_LOC_SYSTEM, user_msg)
        loc = _parse_loc_response(raw)
        if loc is None:
            raise ValueError("Could not parse LLM location response.")

        start_line = max(1, int(loc["start_line"]))
        end_line = min(len(lines), int(loc["end_line"]))
        confidence = float(loc.get("confidence", 0.8))

        # AST refinement: snap to minimal enclosing node.
        start_line, end_line = _snap_to_ast_node(
            tree, node_map, start_line, end_line, record.category
        )

        return [CodeRef(
            file=filename,
            start_line=start_line,
            end_line=end_line,
            ast_node="llm_guided",
        )]

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LLM-guided localisation failed for %s: %s; falling back to heuristic.",
            record.id, exc,
        )
        return _find_refs(record, tree, code, filename, node_map)


def _call_llm_loc(system: str, user: str) -> str:
    backend = os.environ.get("ASSUMPTION_MINER_BACKEND", "openai").lower()
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=128,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content
    elif backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=128,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    else:
        raise ValueError(f"Unknown backend '{backend}'.")


def _parse_loc_response(raw: str) -> Optional[dict]:
    """Extract {start_line, end_line, confidence} from LLM response."""
    raw = raw.strip()
    # Try to find JSON object.
    match = re.search(r'\{[^{}]*"start_line"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: try the whole string.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _snap_to_ast_node(
    tree,
    node_map: dict,
    start_line: int,
    end_line: int,
    category: str,
) -> tuple[int, int]:
    """
    Given an LLM-predicted [start_line, end_line], find the smallest AST node
    of a category-appropriate type that contains this range.

    If no enclosing node is found, return the original range unchanged.
    """
    candidate_types = _CATEGORY_NODE_TYPES.get(category, [])
    best: Optional[tuple[int, int]] = None
    best_span = float("inf")

    for node_type in candidate_types:
        for node in node_map.get(node_type, []):
            nstart = node.start_point[0] + 1
            nend = node.end_point[0] + 1
            # Node must contain the predicted range.
            if nstart <= start_line and nend >= end_line:
                span = nend - nstart + 1
                if span < best_span:
                    best_span = span
                    best = (nstart, nend)

    return best if best is not None else (start_line, end_line)


# --------------------------------------------------------------------------- #
# Context-window helper (used by regenerator)                                  #
# --------------------------------------------------------------------------- #

def get_context_window(
    code: str,
    start_line: int,
    end_line: int,
    k: int = 5,
) -> str:
    """
    Return up to *k* lines above and below [start_line, end_line] in *code*.

    Lines are 1-indexed.  Used by the regenerator to build targeted prompts.
    """
    lines = code.splitlines()
    lo = max(0, start_line - 1 - k)
    hi = min(len(lines), end_line + k)
    return "\n".join(lines[lo:hi])
