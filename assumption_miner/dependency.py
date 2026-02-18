"""
dependency.py — C4: Assumption–code dependency mapper.

Uses tree-sitter to parse Python code and links each AssumptionRecord
to the AST nodes (and source lines) that implement it.

Requires:
  pip install tree-sitter tree-sitter-python
"""

from __future__ import annotations

import logging
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
) -> list[AssumptionRecord]:
    """
    Populate the ``code_refs`` field of each AssumptionRecord in *records*
    by parsing *code* with tree-sitter.

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

    for record in records:
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
    "T1": ["function_definition", "if_statement", "assert_statement"],  # input validation
    "T2": ["return_statement", "typed_parameter", "type"],               # return type
    "T3": ["except_clause", "raise_statement", "if_statement"],          # error handling
    "T4": ["import_statement", "import_from_statement",
           "assignment", "call"],                                         # persistence
    "T5": ["function_definition", "for_statement", "while_statement"],   # algorithm
    "T6": ["import_statement", "call", "assignment"],                    # security
}


def _find_refs(
    record: AssumptionRecord,
    tree,
    code: str,
    filename: str,
    node_map: dict,
) -> list[CodeRef]:
    """
    Heuristic: return CodeRefs for AST nodes whose text overlaps with
    keywords extracted from the record's description.

    This is a lightweight keyword-overlap heuristic; a more precise approach
    using LLM-based localisation is described in §3.4 of the paper.
    """
    keywords = _keywords(record)
    candidate_types = _CATEGORY_NODE_TYPES.get(record.category, list(node_map.keys()))
    lines = code.splitlines()

    refs: list[CodeRef] = []
    seen: set[tuple[int, int]] = set()

    for node_type in candidate_types:
        for node in node_map.get(node_type, []):
            node_text = node.text.decode(errors="replace").lower()
            if any(kw in node_text for kw in keywords):
                start = node.start_point[0] + 1  # 1-indexed
                end = node.end_point[0] + 1
                if (start, end) not in seen:
                    seen.add((start, end))
                    refs.append(
                        CodeRef(
                            file=filename,
                            start_line=start,
                            end_line=end,
                            ast_node=node_type,
                        )
                    )

    return refs


def _keywords(record: AssumptionRecord) -> list[str]:
    """Extract lowercase keywords from description + alternatives."""
    tokens: list[str] = []
    for text in [record.description] + record.alternatives:
        tokens.extend(
            w.lower().strip(".,;:'\"()[]{}") for w in text.split() if len(w) > 3
        )
    return list(set(tokens))


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
