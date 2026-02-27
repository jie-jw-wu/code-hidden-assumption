"""
baselines.py — Extraction baselines for RQ1 comparison.

Four baselines:
  DG  (Direct Generation)      — single-prompt "list assumptions"
  CE  (Comment Extraction)     — regex-mine inline comments/docstrings
  CQ  (Clarification Questions) — ask LLM for clarifying questions, map to assumptions
  CoT (Chain-of-Thought)       — zero-shot CoT "let's think step by step"

Each baseline returns list[AssumptionRecord] with the same interface as extractor.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .schema import AssumptionRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str, model: Optional[str] = None,
              temperature: float = 0.0, max_tokens: int = 4096) -> str:
    backend = os.environ.get("ASSUMPTION_MINER_BACKEND", "openai").lower()
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content
    elif backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    else:
        raise ValueError(f"Unknown backend '{backend}'.")


def _parse_json_array(raw: str) -> list[dict]:
    """Extract and parse the first JSON array in *raw*."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in LLM response.")
    return json.loads(match.group(0))


_CATEGORIES = ("T1", "T2", "T3", "T4", "T5", "T6")
_VALID_CAT_RE = re.compile(r"\bT[1-6]\b")
_SEVERITY = ("low", "medium", "high")

_SCHEMA_HINT = json.dumps({
    "id": "A1",
    "category": "T1|T2|T3|T4|T5|T6",
    "description": "...",
    "rationale": "...",
    "alternatives": ["..."],
    "confidence": 0.8,
    "severity": "low|medium|high",
})

_TAXONOMY = (
    "T1=Input format/validation  T2=Return type/output structure  "
    "T3=Error-handling policy  T4=Persistence/storage  "
    "T5=Algorithm/performance  T6=Security/authentication"
)


def _records_from_dicts(items: list[dict]) -> list[AssumptionRecord]:
    records = []
    for i, d in enumerate(items):
        try:
            # Fill missing required fields with defaults so partial output survives.
            d.setdefault("id", f"A{i+1}")
            d.setdefault("rationale", "")
            d.setdefault("alternatives", [])
            d.setdefault("confidence", 0.5)
            d.setdefault("severity", "medium")
            if d.get("category") not in _CATEGORIES:
                d["category"] = "T2"  # safe default
            records.append(AssumptionRecord.from_dict(d))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed record %d: %s", i, exc)
    return records


# ---------------------------------------------------------------------------
# DG — Direct Generation
# ---------------------------------------------------------------------------

_DG_SYSTEM = (
    "You are a software engineering expert. "
    "List the hidden assumptions embedded in LLM-generated code."
)

_DG_USER = """\
Given this programming prompt and the generated code, list every design decision
that is NOT explicitly stated in the prompt.

Prompt: {prompt}

Code:
```python
{code}
```

Return a JSON array. Each element: {schema}
Taxonomy: {taxonomy}
Return ONLY the JSON array.
"""


def direct_generation(
    prompt: str, code: str, model: Optional[str] = None
) -> list[AssumptionRecord]:
    user = _DG_USER.format(prompt=prompt, code=code,
                           schema=_SCHEMA_HINT, taxonomy=_TAXONOMY)
    raw = _call_llm(_DG_SYSTEM, user, model=model)
    try:
        return _records_from_dicts(_parse_json_array(raw))
    except Exception as exc:  # noqa: BLE001
        logger.error("DG parse failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# CE — Comment Extraction
# ---------------------------------------------------------------------------

# Patterns that signal an assumption encoded as a comment or docstring.
_COMMENT_PATTERNS = [
    re.compile(r"#\s*(assumes?|note|todo|fixme|hack|warning|caveat)[:\s]+(.+)", re.IGNORECASE),
    re.compile(r"#\s*(default(?:s)?(?:\s+to)?)[:\s]+(.+)", re.IGNORECASE),
    re.compile(r'"""(.*?)"""', re.DOTALL),
    re.compile(r"'''(.*?)'''", re.DOTALL),
]

def _keywords_from(record: "AssumptionRecord") -> list[str]:
    """Extract lowercase content words (>3 chars) from description and alternatives."""
    tokens: list[str] = []
    for text in [record.description] + record.alternatives:
        tokens.extend(
            w.lower().strip(".,;:'\"()[]{}") for w in text.split() if len(w) > 3
        )
    return list(set(tokens))


_CE_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "T5": ["cache", "O(n", "O(1", "performance", "index", "sort", "hash map", "memoize",
           "complexity", "algorithm", "eviction", "sliding window", "token bucket"],
    "T6": ["password", "auth", "token", "md5", "sha", "bcrypt", "argon2", "secret", "encrypt",
           "rate limit", "brute force", "sanitize", "csrf", "xss"],
    "T4": ["database", "db", "sqlite", "postgres", "redis", "store", "persist", "file", "disk"],
    "T3": ["error", "exception", "raise", "return none", "return false", "return []", "catch"],
    "T1": ["input", "format", "validate", "encoding", "header", "utf", "csv", "json"],
    "T2": ["return", "output", "type", "list", "dict", "tuple", "string"],
}


def _infer_category(text: str) -> str:
    text_lower = text.lower()
    for cat, keywords in _CE_CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return cat
    return "T2"


def comment_extraction(
    prompt: str, code: str, model: Optional[str] = None
) -> list[AssumptionRecord]:
    """Extract assumptions from inline comments and docstrings (no LLM call)."""
    records: list[AssumptionRecord] = []
    lines = code.splitlines()
    seen: set[str] = set()
    idx = 1

    for lineno, line in enumerate(lines, 1):
        for pat in _COMMENT_PATTERNS[:2]:  # inline comment patterns
            m = pat.search(line)
            if m:
                text = m.group(2).strip()
                if text and text not in seen:
                    seen.add(text)
                    cat = _infer_category(text)
                    records.append(AssumptionRecord(
                        id=f"A{idx}",
                        category=cat,
                        description=text,
                        rationale="Extracted from inline comment.",
                        alternatives=[],
                        confidence=0.4,
                        severity="low",
                    ))
                    idx += 1

    # Docstring extraction
    for pat in _COMMENT_PATTERNS[2:]:
        for m in pat.finditer(code):
            text = m.group(1).strip()
            if text and len(text) > 10 and text not in seen:
                # Only treat non-trivial docstrings as potential assumption containers.
                for sentence in re.split(r"[.\n]", text):
                    sentence = sentence.strip()
                    if len(sentence) > 15 and sentence not in seen:
                        seen.add(sentence)
                        cat = _infer_category(sentence)
                        records.append(AssumptionRecord(
                            id=f"A{idx}",
                            category=cat,
                            description=sentence,
                            rationale="Extracted from docstring.",
                            alternatives=[],
                            confidence=0.3,
                            severity="low",
                        ))
                        idx += 1

    return records


# ---------------------------------------------------------------------------
# CQ — Clarification Questions
# ---------------------------------------------------------------------------

_CQ_SYSTEM = (
    "You are a software engineer reviewing an ambiguous specification. "
    "Generate clarifying questions, then map them to hidden assumptions."
)

_CQ_USER = """\
A developer submitted this prompt to an LLM:
"{prompt}"

The LLM produced this code:
```python
{code}
```

Step 1 — List 3-6 clarifying questions the LLM should have asked before coding.
Step 2 — For each question, state the assumption the LLM implicitly made.
Step 3 — Return a JSON array of AssumptionRecord objects:
{schema}
Taxonomy: {taxonomy}
Return ONLY the JSON array after the tag [RECORDS].
[RECORDS]"""


def clarification_questions(
    prompt: str, code: str, model: Optional[str] = None
) -> list[AssumptionRecord]:
    user = _CQ_USER.format(prompt=prompt, code=code,
                           schema=_SCHEMA_HINT, taxonomy=_TAXONOMY)
    raw = _call_llm(_CQ_SYSTEM, user, model=model)
    # Try [RECORDS] tag first.
    if "[RECORDS]" in raw:
        json_str = raw.split("[RECORDS]", 1)[1].strip()
    else:
        json_str = raw
    try:
        items = json.loads(json_str) if json_str.startswith("[") else _parse_json_array(raw)
        return _records_from_dicts(items)
    except Exception as exc:  # noqa: BLE001
        logger.error("CQ parse failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# CoT — Chain-of-Thought
# ---------------------------------------------------------------------------

_COT_SYSTEM = (
    "You are an expert software engineer. "
    "Think step by step to identify hidden design decisions in code."
)

_COT_USER = """\
Prompt: {prompt}

Code:
```python
{code}
```

Let's think step by step about every design decision embedded in this code that
is NOT explicitly required by the prompt. For each decision, name at least one
realistic alternative.

After your reasoning, output a JSON array of AssumptionRecord objects:
{schema}
Taxonomy: {taxonomy}
Output the JSON array between [RECORDS] and [/RECORDS].

[RECORDS]"""


def chain_of_thought(
    prompt: str, code: str, model: Optional[str] = None
) -> list[AssumptionRecord]:
    user = _COT_USER.format(prompt=prompt, code=code,
                            schema=_SCHEMA_HINT, taxonomy=_TAXONOMY)
    raw = _call_llm(_COT_SYSTEM, user, model=model)
    if "[RECORDS]" in raw:
        body = raw.split("[RECORDS]", 1)[1]
        if "[/RECORDS]" in body:
            body = body.split("[/RECORDS]", 1)[0]
        json_str = body.strip()
    else:
        json_str = raw
    try:
        items = json.loads(json_str) if json_str.lstrip().startswith("[") else _parse_json_array(raw)
        return _records_from_dicts(items)
    except Exception as exc:  # noqa: BLE001
        logger.error("CoT parse failed: %s", exc)
        return []
