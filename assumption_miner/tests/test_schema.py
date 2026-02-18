"""Tests for schema.py — AssumptionRecord round-trip serialisation."""

import json
import pytest

from assumption_miner.schema import AssumptionRecord, CodeRef


SAMPLE_DICT = {
    "id": "A1",
    "category": "T6",
    "description": "Passwords are hashed with MD5 rather than a modern algorithm.",
    "rationale": "MD5 is the default available in hashlib without additional dependencies.",
    "alternatives": ["bcrypt", "argon2", "SHA-256 with salt"],
    "code_refs": [
        {
            "file": "<generated>",
            "start_line": 5,
            "end_line": 7,
            "ast_node": "call",
        }
    ],
    "confidence": 0.95,
    "severity": "high",
}


def test_from_dict_roundtrip():
    record = AssumptionRecord.from_dict(SAMPLE_DICT)
    assert record.id == "A1"
    assert record.category == "T6"
    assert record.confidence == pytest.approx(0.95)
    assert record.severity == "high"
    assert len(record.alternatives) == 3
    assert len(record.code_refs) == 1
    assert record.code_refs[0].start_line == 5

    # Round-trip to dict.
    recovered = record.to_dict()
    assert recovered["id"] == SAMPLE_DICT["id"]
    assert recovered["code_refs"][0]["ast_node"] == "call"


def test_json_roundtrip():
    record = AssumptionRecord.from_dict(SAMPLE_DICT)
    json_str = record.to_json()
    recovered = AssumptionRecord.from_json(json_str)
    assert recovered.id == record.id
    assert recovered.description == record.description


def test_defaults():
    record = AssumptionRecord(
        id="B1",
        category="T2",
        description="Returns indices.",
        rationale="Indices are more common in LeetCode problems.",
        alternatives=["values"],
    )
    assert record.confidence == 1.0
    assert record.severity == "medium"
    assert record.code_refs == []


def test_coderef_no_ast_node():
    ref = CodeRef(file="foo.py", start_line=1, end_line=5)
    d = ref.to_dict()
    assert "ast_node" not in d
    recovered = CodeRef.from_dict(d)
    assert recovered.ast_node is None
