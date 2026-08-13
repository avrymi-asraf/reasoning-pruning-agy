import pytest
from reasoning_pruning.types import (
    ReasoningTrace,
    PruneDecision,
    TransitionExample,
    RolloutResult,
    TrainResult,
    EvalResult,
    EvalSample,
)


def test_reasoning_trace_serde():
    trace = ReasoningTrace(
        question="What is 2+2?",
        steps=["Step 1: Compute 2+2.", "Step 2: The sum is 4."],
        prefix="None",
        full_text="Step 1: Compute 2+2. Step 2: The sum is 4.",
        token_count=15,
        generator_model="test-gen",
    )
    d = trace.to_dict()
    assert d["question"] == "What is 2+2?"
    assert len(d["steps"]) == 2

    trace2 = ReasoningTrace.from_dict(d)
    assert trace2.question == trace.question
    assert trace2.steps == trace.steps


def test_prune_decision_serde():
    decision = PruneDecision(
        can_skip=True,
        skip_start_idx=1,
        skip_end_idx=1,
        skipped_steps=["Unnecessary fluff."],
        next_step="Final answer is 42.",
        reason="Redundant sentence.",
        decision_model="test-dec",
    )
    d = decision.to_dict()
    assert d["can_skip"] is True
    assert d["skip_start_idx"] == 1

    dec2 = PruneDecision.from_dict(d)
    assert dec2.can_skip == decision.can_skip
    assert dec2.reason == decision.reason


def test_transition_example_serde():
    ex = TransitionExample(
        id="ex_1",
        question="Q",
        input_x="Q Step 1",
        target_y="Step 3",
        depth=1,
        skipped_steps=["Step 2"],
        skip_reason="Repeated calculation",
        generator_model="gen",
        decision_model="dec",
    )
    d = ex.to_dict()
    assert d["id"] == "ex_1"
    assert d["depth"] == 1

    ex2 = TransitionExample.from_dict(d)
    assert ex2.id == ex.id
    assert ex2.input_x == ex.input_x
    assert ex2.target_y == ex.target_y
