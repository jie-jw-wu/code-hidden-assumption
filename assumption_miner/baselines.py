"""
baselines.py — Extraction baselines for RQ1 comparison.

Five baselines:
  DG  (Direct Generation)      — single-prompt "list assumptions"
  CE  (Comment Extraction)     — regex-mine inline comments/docstrings
  CQ  (Clarification Questions) — ask LLM for clarifying questions, map to assumptions
  CoT (Chain-of-Thought)       — zero-shot CoT "let's think step by step"
  KBE (Keyword/Pattern-Based)  — AST + regex patterns; no LLM (offline)

Each baseline returns list[AssumptionRecord] with the same interface as extractor.py.
"""

from __future__ import annotations

import ast
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


# ---------------------------------------------------------------------------
# KBE — Keyword / Pattern-Based Extraction (offline, no LLM)
# ---------------------------------------------------------------------------
# Mines Python AST + regex for common patterns that indicate implicit design
# choices.  Works without an API key; serves as a non-trivial offline baseline.

_KBE_RETURN_NONE = re.compile(r"\breturn\s+None\b")
_KBE_RETURN_EMPTY_LIST = re.compile(r"\breturn\s+\[\]")
_KBE_RETURN_EMPTY_DICT = re.compile(r"\breturn\s+\{\}")
_KBE_RAISE = re.compile(r"\braise\b")
_KBE_EXCEPT_BROAD = re.compile(r"\bexcept\s+Exception\b")
_KBE_EXCEPT_BARE = re.compile(r"\bexcept\s*:")


def _has_node(tree: ast.AST, *node_types: type) -> bool:
    return any(isinstance(n, node_types) for n in ast.walk(tree))


def _collect_return_values(tree: ast.AST) -> list[str]:
    """Return string representations of non-trivial return values."""
    returns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            try:
                returns.append(ast.unparse(node.value))
            except Exception:  # noqa: BLE001
                pass
    return returns


def _default_arg_names(tree: ast.AST) -> list[tuple[str, str]]:
    """(arg_name, default_repr) for all function defaults."""
    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            defaults = args.defaults
            n_defaults = len(defaults)
            relevant_args = args.args[-n_defaults:] if n_defaults else []
            for arg, dflt in zip(relevant_args, defaults):
                try:
                    pairs.append((arg.arg, ast.unparse(dflt)))
                except Exception:  # noqa: BLE001
                    pass
    return pairs


def pattern_extraction(
    prompt: str, code: str, model: Optional[str] = None  # noqa: ARG001
) -> list[AssumptionRecord]:
    """Extract implicit assumptions via AST patterns (no LLM required)."""
    records: list[AssumptionRecord] = []
    idx = 1

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return records

    # --- T3: error handling assumptions ---
    if _KBE_RETURN_NONE.search(code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T3",
            description="Returns None on failure rather than raising an exception.",
            rationale="Silently returns None to avoid crashing callers.",
            alternatives=["raise ValueError", "raise RuntimeError", "return empty list"],
            confidence=0.5, severity="medium",
        ))
        idx += 1

    if _KBE_RETURN_EMPTY_LIST.search(code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T3",
            description="Returns empty list [] when no result is found rather than None or raising.",
            rationale="Empty list is iterable-safe sentinel.",
            alternatives=["return None", "raise ValueError('No results')"],
            confidence=0.55, severity="low",
        ))
        idx += 1

    if _KBE_RETURN_EMPTY_DICT.search(code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T3",
            description="Returns empty dict {} on failure rather than raising an exception.",
            rationale="Empty dict avoids KeyError in callers.",
            alternatives=["return None", "raise KeyError"],
            confidence=0.5, severity="low",
        ))
        idx += 1

    if _KBE_EXCEPT_BROAD.search(code) or _KBE_EXCEPT_BARE.search(code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T3",
            description="Catches broad Exception (or bare except) suppressing all error types.",
            rationale="Defensive catch to prevent unhandled exceptions propagating.",
            alternatives=["catch specific exceptions", "let exceptions propagate"],
            confidence=0.6, severity="high",
        ))
        idx += 1

    # --- T2: output format assumptions from return values ---
    ret_vals = _collect_return_values(tree)
    has_list_return = any(v.startswith("[") for v in ret_vals)
    has_dict_return = any(v.startswith("{") for v in ret_vals)
    has_tuple_return = any(v.startswith("(") and "," in v for v in ret_vals)

    if has_list_return and not _KBE_RETURN_EMPTY_LIST.search(code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T2",
            description="Returns results as a list rather than another collection type.",
            rationale="Lists are mutable and iterable; common default return type.",
            alternatives=["return tuple", "return generator", "return set"],
            confidence=0.45, severity="low",
        ))
        idx += 1

    if has_dict_return and not _KBE_RETURN_EMPTY_DICT.search(code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T2",
            description="Returns results as a dict (key-value mapping).",
            rationale="Dict enables named access to multiple return values.",
            alternatives=["return namedtuple", "return dataclass instance", "return tuple"],
            confidence=0.45, severity="low",
        ))
        idx += 1

    if has_tuple_return:
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T2",
            description="Returns a tuple of multiple values rather than a single compound object.",
            rationale="Tuples are lightweight and support unpacking.",
            alternatives=["return dataclass", "return dict", "return list"],
            confidence=0.45, severity="low",
        ))
        idx += 1

    # --- T5: algorithm / performance assumptions ---
    if "dict()" in code or ": {}" in code or "= {}" in code:
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T5",
            description="Uses a hash map (dict) for O(1) average-case lookups.",
            rationale="Hash maps reduce time complexity for membership testing.",
            alternatives=["linear scan O(n)", "sorted list with binary search O(log n)"],
            confidence=0.55, severity="medium",
        ))
        idx += 1

    if "lru_cache" in code or "@cache" in code or "functools" in code:
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T5",
            description="Uses memoization/caching to avoid recomputing results.",
            rationale="Trades memory for speed on repeated inputs.",
            alternatives=["no caching (recompute each time)", "explicit cache dict"],
            confidence=0.7, severity="medium",
        ))
        idx += 1

    if "sorted(" in code or ".sort(" in code:
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T5",
            description="Sorts input data (ascending by default) before processing.",
            rationale="Many algorithms require sorted input; default ascending order.",
            alternatives=["descending sort", "no sort (preserve order)", "stable partial sort"],
            confidence=0.5, severity="low",
        ))
        idx += 1

    # --- T4: persistence assumptions ---
    if re.search(r"\bopen\s*\(", code) or "with open" in code:
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T4",
            description="Reads/writes data from/to local filesystem files.",
            rationale="Simplest persistence mechanism; no external dependencies.",
            alternatives=["use database", "use object storage", "keep in memory"],
            confidence=0.65, severity="medium",
        ))
        idx += 1

    if re.search(r"\bsqlite3\b|\bpsycopg\b|\bsqlalchemy\b", code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T4",
            description="Uses a relational database for persistent storage.",
            rationale="SQL databases provide ACID guarantees and querying.",
            alternatives=["use NoSQL store", "use flat files", "use in-memory dict"],
            confidence=0.7, severity="high",
        ))
        idx += 1

    # --- T1: input validation assumptions ---
    if re.search(r"\bisinstance\s*\(|type\s*\(", code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T1",
            description="Validates input types explicitly before processing.",
            rationale="Type checks prevent TypeError from invalid caller inputs.",
            alternatives=["duck typing without check", "use type annotations only",
                          "raise TypeError on demand"],
            confidence=0.5, severity="medium",
        ))
        idx += 1

    # default args → T1 assumptions about expected input ranges/formats
    for arg_name, dflt in _default_arg_names(tree):
        if arg_name in ("encoding", "errors", "mode"):
            records.append(AssumptionRecord(
                id=f"A{idx}", category="T1",
                description=f"Assumes default {arg_name}='{dflt}' for file/string I/O.",
                rationale=f"'{dflt}' is the most common {arg_name}; explicit default avoids caller error.",
                alternatives=[f"require explicit {arg_name}", f"default to a different {arg_name}"],
                confidence=0.6, severity="low",
            ))
            idx += 1
        elif arg_name in ("timeout", "retries", "max_size", "limit", "maxsize"):
            records.append(AssumptionRecord(
                id=f"A{idx}", category="T5",
                description=f"Uses default {arg_name}={dflt} for performance/resource bounding.",
                rationale=f"Reasonable default that covers most use cases without over-allocating.",
                alternatives=[f"require caller to specify {arg_name}", f"use a different default"],
                confidence=0.6, severity="medium",
            ))
            idx += 1

    # --- T6: security assumptions ---
    if re.search(r"\bhashlib\b|\.encode\(|pbkdf2|bcrypt|argon2|sha256|md5", code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T6",
            description="Hashes or encodes sensitive data (passwords/tokens).",
            rationale="Cryptographic hashing protects credentials at rest.",
            alternatives=["store plaintext (insecure)", "use stronger KDF like Argon2"],
            confidence=0.65, severity="high",
        ))
        idx += 1

    if re.search(r"\bsecrets\b|\brandom\b.*token|uuid", code):
        records.append(AssumptionRecord(
            id=f"A{idx}", category="T6",
            description="Generates tokens/identifiers using a (pseudo-)random source.",
            rationale="Random tokens prevent predictable IDs.",
            alternatives=["use sequential IDs", "use cryptographically secure secrets module"],
            confidence=0.6, severity="medium",
        ))
        idx += 1

    return records
