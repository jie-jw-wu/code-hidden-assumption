"""
schema.py — AssumptionRecord dataclass.

Mirrors the JSON schema defined in the paper (§3.3 / lst:schema).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CodeRef:
    """A reference to a specific region in generated code."""

    file: str
    start_line: int
    end_line: int
    ast_node: Optional[str] = None  # e.g. "FunctionDef", "Assign"

    def to_dict(self) -> dict:
        d = {
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }
        if self.ast_node is not None:
            d["ast_node"] = self.ast_node
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CodeRef":
        return cls(
            file=d["file"],
            start_line=int(d["start_line"]),
            end_line=int(d["end_line"]),
            ast_node=d.get("ast_node"),
        )


Category = Literal["T1", "T2", "T3", "T4", "T5", "T6"]
Severity = Literal["low", "medium", "high"]


@dataclass
class AssumptionRecord:
    """
    A single hidden assumption extracted from LLM-generated code.

    Fields match the JSON schema in §3.3 of the paper.
    """

    id: str                          # e.g. "A1"
    category: Category               # T1–T6 taxonomy label
    description: str                 # natural-language statement
    rationale: str                   # why the LLM made this choice
    alternatives: list[str]          # ≥1 realistic alternatives
    code_refs: list[CodeRef] = field(default_factory=list)  # populated by C4
    confidence: float = 1.0          # extraction confidence in [0, 1]
    severity: Severity = "medium"

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "rationale": self.rationale,
            "alternatives": self.alternatives,
            "code_refs": [r.to_dict() for r in self.code_refs],
            "confidence": self.confidence,
            "severity": self.severity,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "AssumptionRecord":
        return cls(
            id=d["id"],
            category=d["category"],
            description=d["description"],
            rationale=d["rationale"],
            alternatives=d["alternatives"],
            code_refs=[CodeRef.from_dict(r) for r in d.get("code_refs", [])],
            confidence=float(d.get("confidence", 1.0)),
            severity=d.get("severity", "medium"),
        )

    @classmethod
    def from_json(cls, s: str) -> "AssumptionRecord":
        return cls.from_dict(json.loads(s))
