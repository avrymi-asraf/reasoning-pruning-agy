import pytest
from reasoning_pruning.segmenter import segment_steps


def test_segment_sentences():
    text = "First, find the total cost. 3 * $5 = $15. Next, subtract the discount of $2. The answer is $13."
    steps = segment_steps(text, mode="sentences")
    assert len(steps) == 4
    assert steps[0] == "First, find the total cost."
    assert steps[1] == "3 * $5 = $15."
    assert steps[2] == "Next, subtract the discount of $2."
    assert steps[3] == "The answer is $13."


def test_segment_numbered_steps():
    text = "Step 1: Compute 5 * 10 = 50.\nStep 2: Add 5 to get 55.\nStep 3: Output final result."
    steps = segment_steps(text, mode="auto")
    assert len(steps) == 3
    assert "Step 1" in steps[0]
    assert "Step 2" in steps[1]
    assert "Step 3" in steps[2]


def test_segment_paragraphs():
    text = "Paragraph 1 with reasoning.\n\nParagraph 2 with more calculations.\n\nFinal answer."
    steps = segment_steps(text, mode="paragraphs")
    assert len(steps) == 3


def test_segment_empty():
    assert segment_steps("") == []
    assert segment_steps("   ") == []
