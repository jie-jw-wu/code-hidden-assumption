"""
extractor.py — C3: Assumption extractor.

Implements the two-phase extraction prompt from §3.2 / Listing 1 of the paper.
Takes (prompt, code) → list[AssumptionRecord].
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .schema import AssumptionRecord

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt template (mirrors lst:prompt in the paper)                           #
# --------------------------------------------------------------------------- #

_SCHEMA_SUMMARY = json.dumps(
    {
        "id": "string (e.g. 'A1')",
        "category": "T1|T2|T3|T4|T5|T6",
        "description": "natural-language statement of the assumption",
        "rationale": "why the LLM made this choice",
        "alternatives": ["list of >=1 realistic alternative choices"],
        "confidence": "float in [0,1]",
        "severity": "low|medium|high",
    },
    indent=2,
)

_SYSTEM = (
    "You are an expert software engineer auditing LLM-generated code "
    "for implicit design decisions (hidden assumptions)."
)

_USER_TEMPLATE = """\
### Original Prompt
{prompt}

### Generated Code
```python
{code}
```

### Task
1. REASON: List every design decision embedded in the code that is NOT explicitly
   required by the prompt. For each, name at least one realistic alternative.

2. FORMAT: Return a JSON array of AssumptionRecord objects matching this schema:
{schema}

Taxonomy categories:
  T1 — Input format / validation
  T2 — Return type / output structure
  T3 — Edge-case / error-handling policy
  T4 — Persistence / storage backend
  T5 — Algorithm / performance trade-off
  T6 — Security / authentication policy

Return ONLY valid JSON after the tag [RECORDS]. Do not include any other text
after [RECORDS].

[RECORDS]"""

_RECORDS_TAG = "[RECORDS]"


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def extract_assumptions(
    prompt: str,
    code: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> list[AssumptionRecord]:
    """
    Run the two-phase extraction prompt on (prompt, code).

    Returns a (possibly empty) list of AssumptionRecord objects.
    Raises ValueError if the LLM response cannot be parsed as valid JSON.
    """
    raw = _call_llm(prompt, code, model=model, temperature=temperature, max_tokens=max_tokens)
    return _parse_response(raw)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _build_user_message(prompt: str, code: str) -> str:
    return _USER_TEMPLATE.format(
        prompt=prompt,
        code=code,
        schema=_SCHEMA_SUMMARY,
    )


def _call_llm(
    prompt: str,
    code: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    backend = os.environ.get("ASSUMPTION_MINER_BACKEND", "openai").lower()
    user_msg = _build_user_message(prompt, code)

    if backend == "openai":
        return _call_openai(user_msg, model or "gpt-4o", temperature, max_tokens)
    elif backend == "anthropic":
        return _call_anthropic(
            user_msg, model or "claude-sonnet-4-6", temperature, max_tokens
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'.")


def _call_openai(user_msg: str, model: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def _call_anthropic(user_msg: str, model: str, temperature: float, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return message.content[0].text


def _parse_response(raw: str) -> list[AssumptionRecord]:
    """
    Extract the JSON array following the [RECORDS] tag and deserialise it.
    """
    if _RECORDS_TAG in raw:
        json_str = raw.split(_RECORDS_TAG, 1)[1].strip()
    else:
        # Fallback: attempt to find a JSON array anywhere in the response.
        logger.warning(
            "LLM response did not contain [RECORDS] tag; attempting to extract JSON array."
        )
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError("Could not locate a JSON array in the LLM response.")

    try:
        records_raw = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\n\nRaw output:\n{raw}") from exc

    if not isinstance(records_raw, list):
        raise ValueError(f"Expected a JSON array, got {type(records_raw).__name__}.")

    records: list[AssumptionRecord] = []
    for i, item in enumerate(records_raw):
        try:
            records.append(AssumptionRecord.from_dict(item))
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed record #%d: %s", i, exc)

    return records
