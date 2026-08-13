import pytest
from reasoning_pruning.types import ReasoningTrace, PruneDecision, EvalSample
from reasoning_pruning.visualizer import render_trace_diff, render_comparison


def test_render_trace_diff_html():
    trace = ReasoningTrace(
        question="Find x in 2x=10",
        steps=["Step 0: We have 2x=10.", "Step 1: 2 is an even number.", "Step 2: Divide both sides by 2: x=5."],
    )
    decision = PruneDecision(
        can_skip=True,
        skip_start_idx=1,
        skip_end_idx=1,
        skipped_steps=["Step 1: 2 is an even number."],
        next_step="Step 2: Divide both sides by 2: x=5.",
        reason="Parity of 2 does not assist solving linear equations.",
        decision_model="test-d",
    )
    html_out = render_trace_diff(trace, decision, as_html=True)
    assert isinstance(html_out, str)
    assert "PRUNED" in html_out
    assert "NEXT USEFUL STEP" in html_out
    assert "Parity of 2" in html_out


def test_render_trace_diff_rich():
    trace = ReasoningTrace(
        question="Find x in 2x=10",
        steps=["Step 0", "Step 1", "Step 2"],
    )
    decision = PruneDecision(
        can_skip=False,
        reason="No skip possible",
        decision_model="test-d",
    )
    panel = render_trace_diff(trace, decision, as_html=False)
    assert panel is not None


def test_render_comparison():
    sample = EvalSample(
        question="What is 2+2?",
        ground_truth="4",
        base_response="Step 1: Count 2. Step 2: Add 2. Answer: 4",
        pruned_response="2+2=4.",
        base_tokens=15,
        pruned_tokens=5,
        base_correct=True,
        pruned_correct=True,
        token_savings_pct=66.7,
    )
    html_out = render_comparison(sample, as_html=True)
    assert "66.7%" in html_out
    assert "Ground Truth" in html_out
