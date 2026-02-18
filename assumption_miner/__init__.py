"""
assumption_miner — AssumptionMiner framework.

Top-level package exposing the public API for extracting, mapping,
and updating hidden assumptions in LLM-generated code.

Typical usage::

    from assumption_miner import extract_assumptions, map_dependencies, regenerate
    from assumption_miner.ingestion import from_strings

    sample = from_strings(prompt, code)
    records = extract_assumptions(sample.prompt, sample.code)
    records = map_dependencies(records, sample.code)

    # Developer reviews records, decides to change A1's assumption:
    updated_code = regenerate(sample.code, records[0], new_description="Use bcrypt")
"""

from .extractor import extract_assumptions
from .dependency import map_dependencies
from .regenerator import regenerate
from .schema import AssumptionRecord, CodeRef

__all__ = [
    "extract_assumptions",
    "map_dependencies",
    "regenerate",
    "AssumptionRecord",
    "CodeRef",
]
