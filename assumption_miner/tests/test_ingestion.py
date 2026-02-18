"""Tests for ingestion.py."""

import json
import tempfile
from pathlib import Path

import pytest

from assumption_miner.ingestion import from_strings, from_json_file, from_code_file


def test_from_strings_strips_whitespace():
    s = from_strings("  hello  ", "  code  ")
    assert s.prompt == "hello"
    assert s.code == "code"


def test_from_json_file_yields_samples(tmp_path):
    data = [
        {"id": "s1", "prompt": "Do X", "code": "x = 1"},
        {"id": "s2", "prompt": "Do Y", "code": "y = 2"},
    ]
    f = tmp_path / "samples.json"
    f.write_text(json.dumps(data))

    samples = list(from_json_file(f))
    assert len(samples) == 2
    assert samples[0].source_id == "s1"
    assert samples[1].prompt == "Do Y"


def test_from_json_file_invalid_format(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="JSON array"):
        list(from_json_file(f))


def test_from_code_file(tmp_path):
    code_file = tmp_path / "auth.py"
    code_file.write_text("def auth(): pass\n")
    sample = from_code_file("Implement auth.", code_file)
    assert "def auth" in sample.code
    assert sample.prompt == "Implement auth."
