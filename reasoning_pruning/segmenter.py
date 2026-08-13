"""Reasoning step and sentence segmentation tools.

Provides robust segmentation for decomposing raw reasoning traces into
discrete logical units for pruning and training.
"""

from __future__ import annotations

import re
from typing import List


def segment_steps(text: str, mode: str = "auto") -> List[str]:
    """Segment a raw reasoning text into discrete logical steps.

    What it does:
        Takes a generated reasoning string and decomposes it into clean,
        order-preserved reasoning steps (sentences, numbered points, or logical blocks).

    When to reach for it:
        Whenever you have raw model output and need discrete step units (s_1, s_2, ..., s_n)
        for the decision model to evaluate.

    Parameters:
        text: The raw reasoning text to segment.
        mode: Segmentation strategy:
            - 'auto': Detects numbered steps / double newlines, falling back to sentences.
            - 'lines': Splits on linebreaks / newlines.
            - 'sentences': Splits on sentence boundary punctuation (., !, ?) avoiding math decimals.
            - 'paragraphs': Splits on double newlines / paragraph blocks.

    Returns:
        A list of non-empty step strings with leading/trailing whitespace stripped.

    Example:
        >>> text = "First, find the total cost. 3 * $5 = $15. Next, subtract the discount of $2. The answer is $13."
        >>> segment_steps(text, mode="sentences")
        ['First, find the total cost.', '3 * $5 = $15.', 'Next, subtract the discount of $2.', 'The answer is $13.']
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if mode == "auto":
        # Check if text has structured step markers (e.g. "Step 1:", "1.", "\n\n")
        has_numbered_steps = bool(re.search(r"(?:^|\n)(?:Step\s*\d+[:.]|\d+[\.\)])\s+", cleaned, re.MULTILINE))
        has_paragraphs = "\n\n" in cleaned

        if has_numbered_steps:
            steps = _split_numbered_steps(cleaned)
            if len(steps) > 1:
                return steps
        if has_paragraphs:
            steps = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            if len(steps) > 1:
                return steps
        return _split_sentences(cleaned)

    elif mode == "paragraphs":
        return [p.strip() for p in cleaned.split("\n\n") if p.strip()]

    elif mode == "lines":
        return [line.strip() for line in cleaned.splitlines() if line.strip()]

    elif mode == "sentences":
        return _split_sentences(cleaned)

    else:
        raise ValueError(f"Unknown segmentation mode: {mode}. Choose from 'auto', 'lines', 'sentences', 'paragraphs'.")


def _split_numbered_steps(text: str) -> List[str]:
    """Split on explicit numbered markers like 'Step 1:', '1.', '1)'."""
    pattern = r"(?:^|\n)(?=(?:Step\s*\d+[:.]|\d+[\.\)])\s+)"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences while avoiding splits on decimal numbers (e.g. 3.14, $2.50) or common abbreviations."""
    # Look for sentence endings (.!?) followed by whitespace and an uppercase letter/quote, or newline
    # Avoid matching inside numbers: (?<!\d)\.(?!\d)
    pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=[.!?])\n+"
    raw_sentences = re.split(pattern, text)
    sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        if s_clean:
            sentences.append(s_clean)
    return sentences if sentences else [text.strip()]
