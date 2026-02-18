"""
generator.py — C2: LLM code generator.

Thin wrapper around the OpenAI / Anthropic API that:
  1. Accepts a natural-language prompt.
  2. Returns generated code (str).

Configure the backend via the ASSUMPTION_MINER_BACKEND env var:
  "openai"     — uses gpt-4o (default)
  "anthropic"  — uses claude-sonnet-4-6
"""

from __future__ import annotations

import os
from typing import Optional

# Lazy imports so the package does not require both SDKs to be installed.
_BACKEND = os.environ.get("ASSUMPTION_MINER_BACKEND", "openai").lower()

_SYSTEM_PROMPT = (
    "You are an expert software engineer. "
    "Given the following natural-language specification, write complete, "
    "idiomatic Python code that implements it. "
    "Output ONLY the code, with no explanation."
)


def generate_code(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Generate Python code for *prompt* using the configured LLM backend.

    Returns the raw code string (leading/trailing whitespace stripped).
    """
    if _BACKEND == "openai":
        return _generate_openai(prompt, model or "gpt-4o", temperature, max_tokens)
    elif _BACKEND == "anthropic":
        return _generate_anthropic(
            prompt, model or "claude-sonnet-4-6", temperature, max_tokens
        )
    else:
        raise ValueError(
            f"Unknown backend '{_BACKEND}'. "
            "Set ASSUMPTION_MINER_BACKEND to 'openai' or 'anthropic'."
        )


# --------------------------------------------------------------------------- #
# Backend implementations                                                      #
# --------------------------------------------------------------------------- #

def _generate_openai(
    prompt: str, model: str, temperature: float, max_tokens: int
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required for the OpenAI backend. "
            "Install with: pip install openai"
        ) from exc

    client = OpenAI()  # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def _generate_anthropic(
    prompt: str, model: str, temperature: float, max_tokens: int
) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package is required for the Anthropic backend. "
            "Install with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()
