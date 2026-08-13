import pytest
from reasoning_pruning.decision import _parse_decision_response
from reasoning_pruning.generation import extract_transition
from reasoning_pruning.types import ReasoningTrace, PruneDecision


def test_parse_decision_valid_json():
    raw_response = """
    ```json
    {
      "can_skip": true,
      "skip_start_idx": 1,
      "skip_end_idx": 1,
      "reason": "Step 1 is redundant filler."
    }
    ```
    """
    steps = ["Step 0: Start.", "Step 1: Unneeded fluff.", "Step 2: Final answer."]
    dec = _parse_decision_response(raw_response, steps, "test-model")
    assert dec.can_skip is True
    assert dec.skip_start_idx == 1
    assert dec.skip_end_idx == 1
    assert dec.skipped_steps == ["Step 1: Unneeded fluff."]
    assert dec.next_step == "Step 2: Final answer."
    assert "redundant filler" in dec.reason


def test_parse_decision_no_skip():
    raw_response = '{"can_skip": false, "reason": "All steps essential."}'
    steps = ["Step 0", "Step 1"]
    dec = _parse_decision_response(raw_response, steps, "test-model")
    assert dec.can_skip is False
    assert dec.reason == "All steps essential."


def test_extract_transition():
    trace = ReasoningTrace(
        question="What is 5+5?",
        steps=["Step 0: First term is 5.", "Step 1: 5 is prime.", "Step 2: 5 + 5 = 10."],
        generator_model="test-gen",
    )
    decision = PruneDecision(
        can_skip=True,
        skip_start_idx=1,
        skip_end_idx=1,
        skipped_steps=["Step 1: 5 is prime."],
        next_step="Step 2: 5 + 5 = 10.",
        reason="Irrelevant property of 5.",
        decision_model="test-dec",
    )
    trans = extract_transition(trace, decision, depth=1)
    assert trans is not None
    assert trans.depth == 1
    assert "Step 0: First term is 5." in trans.input_x
    assert trans.target_y == "Step 2: 5 + 5 = 10."
    assert trans.skipped_steps == ["Step 1: 5 is prime."]
