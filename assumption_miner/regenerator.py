"""
regenerator.py — C5: Incremental code regenerator.

When a developer revises assumption a_i → a_i', this module:
  1. Identifies affected code regions via R(a_i) from the dependency graph.
  2. Builds a targeted regeneration prompt with k-line context (k=5 by default).
  3. Calls the LLM to produce a replacement sub-tree.
  4. Splices the replacement back into the original code.

See §3.5 of the paper for the formal description.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .schema import AssumptionRecord, CodeRef

logger = logging.getLogger(__name__)

_DEFAULT_K = 5  # context lines above/below each affected region (§3.5)

_SYSTEM = (
    "You are an expert software engineer performing a targeted code update. "
    "You will receive a code snippet and an instruction to change one design decision. "
    "Output ONLY the revised code snippet, preserving indentation and style."
)

_USER_TEMPLATE = """\
### Original assumption
{old_description}

### Revised assumption
{new_description}

### Code region to update (lines {start}–{end} of the full file, with {k} lines of context)
```python
{context}
```

Rewrite ONLY the lines shown above so that they implement the revised assumption.
Do NOT change any code outside the shown region.
Output the replacement snippet between [CODE] and [/CODE] tags.

[CODE]"""


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def regenerate(
    code: str,
    record: AssumptionRecord,
    new_description: str,
    k: int = _DEFAULT_K,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Incrementally regenerate the regions of *code* governed by *record*,
    replacing *record.description* with *new_description*.

    Returns the updated full code string.
    """
    if not record.code_refs:
        logger.warning(
            "AssumptionRecord '%s' has no code_refs; returning original code unchanged.",
            record.id,
        )
        return code

    updated_code = code
    # Process refs in reverse line order so splice offsets stay valid.
    for ref in sorted(record.code_refs, key=lambda r: r.start_line, reverse=True):
        updated_code = _regenerate_region(
            updated_code,
            ref,
            old_description=record.description,
            new_description=new_description,
            k=k,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return updated_code


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _regenerate_region(
    code: str,
    ref: CodeRef,
    old_description: str,
    new_description: str,
    k: int,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    from .dependency import get_context_window

    context = get_context_window(code, ref.start_line, ref.end_line, k=k)
    user_msg = _USER_TEMPLATE.format(
        old_description=old_description,
        new_description=new_description,
        start=max(1, ref.start_line - k),
        end=ref.end_line + k,
        k=k,
        context=context,
    )

    raw = _call_llm(user_msg, model=model, temperature=temperature, max_tokens=max_tokens)
    replacement = _extract_code(raw)

    return _splice(code, ref.start_line, ref.end_line, replacement, k)


def _call_llm(user_msg: str, model: Optional[str], temperature: float, max_tokens: int) -> str:
    backend = os.environ.get("ASSUMPTION_MINER_BACKEND", "openai").lower()

    if backend == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model or "gpt-4o",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content

    elif backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=max_tokens,
            temperature=temperature,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return message.content[0].text

    else:
        raise ValueError(f"Unknown backend '{backend}'.")


def _extract_code(raw: str) -> str:
    """Pull the snippet between [CODE] and [/CODE] tags."""
    if "[CODE]" in raw and "[/CODE]" in raw:
        return raw.split("[CODE]", 1)[1].split("[/CODE]", 1)[0].strip()
    # Fallback: strip the tag prefix and return the rest.
    if "[CODE]" in raw:
        return raw.split("[CODE]", 1)[1].strip()
    return raw.strip()


def _splice(code: str, start_line: int, end_line: int, replacement: str, k: int) -> str:
    """
    Replace lines [start_line - k, end_line + k) (1-indexed) with *replacement*.
    """
    lines = code.splitlines(keepends=True)
    lo = max(0, start_line - 1 - k)
    hi = min(len(lines), end_line + k)

    replacement_lines = [l + "\n" for l in replacement.splitlines()]
    new_lines = lines[:lo] + replacement_lines + lines[hi:]
    return "".join(new_lines)
