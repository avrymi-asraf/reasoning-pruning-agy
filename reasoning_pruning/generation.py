"""Reasoning trajectory generation and recursive pruning rollout tools.

Big Picture Role: Manages generator model G execution and recursive pruning rollouts
to generate multi-depth training examples.
Code Flow Connection: Initiates the reasoning chain via `generate_trace`, calls `find_first_skip`
and `extract_transition`, and iterates recursively in `rollout_pruning`.
Execution Environment: Runs in local Python, Google Colab notebooks, and CLI processes
using local model checkpoints or API endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import litellm

from reasoning_pruning.config import (
    get_default_decision_model,
    get_default_generator_model,
)
from reasoning_pruning.decision import find_first_skip
from reasoning_pruning.segmenter import segment_steps
from reasoning_pruning.types import (
    PruneDecision,
    ReasoningTrace,
    RolloutResult,
    TransitionExample,
)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert reasoning model. Solve the given problem step-by-step with clear, "
    "concise, and logically rigorous deductions."
)


def generate_trace(
    question: str,
    model: Optional[str] = None,
    prefix: str = "",
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    segment_mode: str = "auto",
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ReasoningTrace:
    """Generate a step-by-step reasoning trajectory for a given question and prefix.

    What it does:
        Prompts generator model G (via LiteLLM or local model) to produce reasoning
        from a given question and optional existing prefix. Segments the resulting
        trace into discrete steps.

    When to reach for it:
        - When generating initial reasoning traces on a benchmark question.
        - When generating continuations from an already-pruned prefix during iterative rollout.

    Parameters:
        question: The input question or problem.
        model: Generator model name (LiteLLM supported: e.g. 'gpt-4o-mini', 'ollama/llama3',
            'huggingface/Qwen/Qwen2.5-7B-Instruct', 'deepseek/deepseek-chat').
        prefix: Pre-existing reasoning prefix to continue from.
        system_prompt: Optional system instructions for the generator.
        max_tokens: Maximum completion tokens.
        temperature: Sampling temperature.
        segment_mode: Segmentation strategy ('auto', 'lines', 'sentences', 'paragraphs').
        api_key: Optional provider API key.
        **kwargs: Additional parameters passed to LiteLLM.

    Returns:
        A ReasoningTrace object containing the question, prefix, discrete steps, and token counts.

    Example:
        >>> trace = generate_trace("What is 15% of 80?", model="gpt-4o-mini")
        >>> len(trace.steps) > 0
        True
    """
    model_name = model or get_default_generator_model()
    sys_msg = system_prompt or DEFAULT_SYSTEM_PROMPT

    user_content = question
    if prefix.strip():
        user_content = f"{question}\n\n[Previous Steps]:\n{prefix}\n\n[Continue reasoning from here]:"

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_content},
    ]

    if api_key:
        kwargs["api_key"] = api_key

    try:
        response = litellm.completion(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        raw_text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        token_count = usage.completion_tokens if usage else len(raw_text.split())

    except Exception as e:
        logger.error(f"Generation failed with model {model_name}: {e}")
        raw_text = f"Error during generation: {e}"
        token_count = 0

    steps = segment_steps(raw_text, mode=segment_mode)

    return ReasoningTrace(
        question=question,
        prefix=prefix,
        steps=steps,
        full_text=raw_text,
        token_count=token_count,
        generator_model=model_name,
        metadata={"temperature": temperature, "max_tokens": max_tokens},
    )


def extract_transition(
    trace: ReasoningTrace,
    decision: PruneDecision,
    depth: int = 1,
    example_id: Optional[str] = None,
) -> Optional[TransitionExample]:
    """Construct a Pruning-Transition (PT) training example from a trace and decision.

    What it does:
        Creates the training pair:
        input_x = question + [existing prefix] + steps[0 .. skip_start_idx - 1]
        target_y = steps[skip_end_idx + 1] (the first useful step after the skipped span)

    When to reach for it:
        Whenever you have a valid PruneDecision and want to turn it into an SFT training instance.

    Parameters:
        trace: The ReasoningTrace that was pruned.
        decision: The PruneDecision containing skip indices.
        depth: The current recursive pruning depth (1 for first skip, etc.).
        example_id: Optional explicit ID. Defaults to auto-generated identifier.

    Returns:
        A TransitionExample object, or None if decision.can_skip is False or invalid.
    """
    if not decision.can_skip or decision.skip_start_idx < 0:
        return None

    # Useful prefix: all steps before the skipped span
    useful_prefix_steps = trace.steps[: decision.skip_start_idx]
    useful_prefix_str = " ".join(useful_prefix_steps).strip()

    # Combine with any prior trace prefix
    if trace.prefix.strip() and useful_prefix_str:
        full_prefix = f"{trace.prefix.strip()} {useful_prefix_str}"
    elif trace.prefix.strip():
        full_prefix = trace.prefix.strip()
    else:
        full_prefix = useful_prefix_str

    # Input x represents question plus all useful context prior to the skip
    input_x = f"{trace.question}\n{full_prefix}".strip() if full_prefix else trace.question

    # Target y is the first useful deduction step after the skipped span
    target_y = decision.next_step

    # Extract all skipped text
    skipped_span_steps = trace.steps[decision.skip_start_idx : decision.skip_end_idx + 1]

    ex_id = example_id or f"trans_{trace.question[:12].strip().replace(' ', '_')}_d{depth}"

    return TransitionExample(
        id=ex_id,
        question=trace.question,
        input_x=input_x,
        target_y=target_y,
        depth=depth,
        skipped_steps=decision.skipped_steps if decision.skipped_steps else skipped_span_steps,
        skip_reason=decision.reason,
        generator_model=trace.generator_model,
        decision_model=decision.decision_model,
        metadata={
            "skip_start_idx": decision.skip_start_idx,
            "skip_end_idx": decision.skip_end_idx,
            "original_step_count": len(trace.steps),
        },
    )


def rollout_pruning(
    question: str,
    generator_model: Optional[str] = None,
    decision_model: Optional[str] = None,
    max_depth: int = 3,
    segment_mode: str = "auto",
    api_key: Optional[str] = None,
    on_step: Optional[Callable[[int, ReasoningTrace, PruneDecision], None]] = None,
    **kwargs: Any,
) -> RolloutResult:
    """Execute iterative, multi-depth pruning rollout on a question.

    What it does:
        1. Generates initial reasoning trace from question with G.
        2. Identifies first removable step with D.
        3. Extracts depth-1 transition example.
        4. Feeds pruned context back into G to generate continuation.
        5. Repeats up to max_depth or until D detects no further safe skips.

    When to reach for it:
        - To generate full multi-depth training examples for a question.
        - To evaluate how much a question's reasoning chain can be compressed.

    Parameters:
        question: The input question.
        generator_model: Model name for generator G (defaults to configured default generator).
        decision_model: Model name for decision model D (defaults to configured default auditor).
        max_depth: Maximum number of pruning iterations (default: 3).
        segment_mode: Step segmentation mode ('auto', 'sentences', 'lines').
        api_key: Optional API key for LLM calls.
        on_step: Optional callback called after each depth iteration: fn(depth, trace, decision).
        **kwargs: Additional kwargs passed to generate_trace and find_first_skip.

    Returns:
        A RolloutResult containing all extracted TransitionExamples, traces, and compression metrics.
    """
    g_model = generator_model or get_default_generator_model()
    d_model = decision_model or get_default_decision_model()

    transitions: List[TransitionExample] = []
    traces: List[ReasoningTrace] = []
    decisions: List[PruneDecision] = []

    current_prefix = ""
    original_step_count = 0
    final_pruned_steps: List[str] = []

    for depth in range(1, max_depth + 1):
        # 1. Generate reasoning from current context
        trace = generate_trace(
            question=question,
            model=g_model,
            prefix=current_prefix,
            segment_mode=segment_mode,
            api_key=api_key,
            **kwargs,
        )
        traces.append(trace)

        if depth == 1:
            original_step_count = len(trace.steps)

        # 2. Find first safe skip
        decision = find_first_skip(
            trace=trace,
            decision_model=d_model,
            api_key=api_key,
            **kwargs,
        )
        decisions.append(decision)

        if on_step:
            on_step(depth, trace, decision)

        if not decision.can_skip:
            # No more pruning possible at this depth; append remaining steps and finish
            if trace.steps:
                final_pruned_steps.extend(trace.steps)
            break

        # 3. Extract transition example
        transition = extract_transition(trace, decision, depth=depth)
        if transition:
            transitions.append(transition)

        # 4. Construct pruned context for next depth
        # Keep steps before skip + the next useful step
        kept_prefix_steps = trace.steps[: decision.skip_start_idx]
        next_step = trace.steps[decision.skip_end_idx + 1] if decision.skip_end_idx + 1 < len(trace.steps) else ""

        final_pruned_steps.extend(kept_prefix_steps)
        if next_step:
            final_pruned_steps.append(next_step)

        # Update prefix for next iteration
        kept_context = " ".join(kept_prefix_steps + ([next_step] if next_step else [])).strip()
        if current_prefix:
            current_prefix = f"{current_prefix} {kept_context}".strip()
        else:
            current_prefix = kept_context

        # If next_step was the final step of the trace, rollout is complete
        if decision.skip_end_idx + 1 >= len(trace.steps) - 1:
            break

    final_pruned_text = " ".join(final_pruned_steps)
    final_step_count = len(final_pruned_steps)
    compression_ratio = 1.0 - (final_step_count / max(original_step_count, 1))

    return RolloutResult(
        question=question,
        transitions=transitions,
        traces=traces,
        decisions=decisions,
        final_pruned_trace=final_pruned_text,
        original_step_count=original_step_count,
        final_step_count=final_step_count,
        compression_ratio=max(0.0, compression_ratio),
    )
