"""Decision model tools for identifying removable reasoning units.

Uses LiteLLM to prompt decision models (e.g. gpt-4o-mini, claude-3-5-sonnet, deepseek-chat, local LLMs)
to detect the first skippable reasoning step or span.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import litellm

from reasoning_pruning.types import PruneDecision, ReasoningTrace

logger = logging.getLogger(__name__)

DEFAULT_DECISION_PROMPT = """You are an expert reasoning auditor. Your task is to analyze a step-by-step reasoning trajectory and identify the FIRST reasoning step (or short span of consecutive steps) that can be SAFELY SKIPPED without damaging the logical correctness or deductive validity of the final solution.

QUESTION:
{question}

{prefix_section}
REASONING STEPS:
{formatted_steps}

CRITERIA FOR SKIPPING:
1. Redundancy / Repetition: The step repeats calculations or facts already established.
2. Unnecessary Detour / Fluff: Conversational filler, meta-commentary, or uninformative trivialities.
3. Locally Skippable: The reasoning can logically jump from the prefix directly to the subsequent step without losing necessary context.
4. DO NOT skip essential mathematical derivations, intermediate results needed downstream, or the final answer.

INSTRUCTIONS:
- Identify the FIRST skippable step (index `skip_start_idx`) and optional ending index (`skip_end_idx`).
- If no step can be safely skipped, set `"can_skip": false`.
- Return ONLY a valid JSON object in the exact schema below.

JSON FORMAT:
```json
{{
  "can_skip": true,
  "skip_start_idx": 2,
  "skip_end_idx": 2,
  "reason": "Step 2 merely restates that the apple is red without computing or contributing to the total price."
}}
```
"""


def find_first_skip(
    trace: ReasoningTrace,
    decision_model: str = "gpt-4o-mini",
    prompt_template: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    **litellm_kwargs: Any,
) -> PruneDecision:
    """Identify the first safely removable reasoning step or span in a trace.

    What it does:
        Formats the reasoning trace into indexed steps, calls a decision LLM (via LiteLLM),
        and returns a structured PruneDecision indicating which step/span can be skipped
        and why.

    When to reach for it:
        - When inspecting whether a reasoning path has redundant steps.
        - As the core decision engine during dataset rollout.
        - To test and refine pruning decision prompts.

    Parameters:
        trace: The ReasoningTrace containing discrete steps.
        decision_model: Model name/identifier supported by LiteLLM (e.g. 'gpt-4o-mini',
            'claude-3-5-haiku-20241022', 'deepseek/deepseek-chat', 'ollama/llama3').
        prompt_template: Optional custom prompt string overriding DEFAULT_DECISION_PROMPT.
        temperature: Sampling temperature for decision model (default 0.0 for consistency).
        api_key: Optional provider API key.
        **litellm_kwargs: Additional parameters passed to `litellm.completion`.

    Returns:
        A PruneDecision object with `can_skip`, step indices, skipped text, next step, and reason.

    Example:
        >>> trace = ReasoningTrace(question="2+2*2?", steps=["First multiply: 2*2=4.", "Multiplication precedes addition.", "Now add 2: 2+4=6."])
        >>> decision = find_first_skip(trace, decision_model="gpt-4o-mini")
        >>> decision.can_skip
        True
        >>> decision.skip_start_idx
        1
    """
    if len(trace.steps) <= 1:
        # A 1-step or empty trace cannot be pruned while retaining an answer
        return PruneDecision(
            can_skip=False,
            reason="Trace has fewer than 2 steps; cannot prune intermediate reasoning.",
            decision_model=decision_model,
        )

    # Format steps with 0-based indices for the LLM
    formatted_steps = "\n".join(f"[{i}] {step}" for i, step in enumerate(trace.steps))
    prefix_section = f"USEFUL PREFIX (already established):\n{trace.prefix}\n" if trace.prefix else ""

    template = prompt_template or DEFAULT_DECISION_PROMPT
    prompt = template.format(
        question=trace.question,
        prefix_section=prefix_section,
        formatted_steps=formatted_steps,
    )

    messages = [{"role": "user", "content": prompt}]
    if api_key:
        litellm_kwargs["api_key"] = api_key

    try:
        response = litellm.completion(
            model=decision_model,
            messages=messages,
            temperature=temperature,
            **litellm_kwargs,
        )
        raw_text = response.choices[0].message.content or ""
        return _parse_decision_response(raw_text, trace.steps, decision_model)

    except Exception as e:
        logger.warning(f"Decision model call failed ({decision_model}): {e}")
        return PruneDecision(
            can_skip=False,
            reason=f"Decision call error: {str(e)}",
            decision_model=decision_model,
            raw_response=str(e),
        )


def _parse_decision_response(raw_text: str, steps: list[str], decision_model: str) -> PruneDecision:
    """Extract and validate JSON decision from raw LLM output."""
    # Find JSON block enclosed in markdown backticks or curly braces
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        brace_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        json_str = brace_match.group(0) if brace_match else raw_text

    try:
        parsed: Dict[str, Any] = json.loads(json_str)
    except Exception:
        # Fallback regex extraction if JSON is malformed
        can_skip = "true" in raw_text.lower() and "false" not in raw_text.lower()[:30]
        return PruneDecision(
            can_skip=can_skip,
            reason="Fallback parse from unformatted response",
            decision_model=decision_model,
            raw_response=raw_text,
        )

    can_skip = bool(parsed.get("can_skip", False))
    if not can_skip:
        return PruneDecision(
            can_skip=False,
            reason=parsed.get("reason", "Decision model found no skippable steps."),
            decision_model=decision_model,
            raw_response=raw_text,
        )

    start_idx = int(parsed.get("skip_start_idx", -1))
    end_idx = int(parsed.get("skip_end_idx", start_idx))
    reason = str(parsed.get("reason", ""))

    # Validate indices bounds
    n_steps = len(steps)
    if start_idx < 0 or start_idx >= n_steps:
        return PruneDecision(
            can_skip=False,
            reason=f"Invalid skip index: {start_idx} outside [0, {n_steps - 1}].",
            decision_model=decision_model,
            raw_response=raw_text,
        )

    if end_idx < start_idx or end_idx >= n_steps:
        end_idx = start_idx

    # If the model tried to skip the last step (which contains the final answer), disallow unless valid
    if end_idx >= n_steps - 1 and start_idx == 0 and n_steps > 1:
        # Tried to skip the entire trace
        return PruneDecision(
            can_skip=False,
            reason="Cannot skip all steps in the trajectory.",
            decision_model=decision_model,
            raw_response=raw_text,
        )

    skipped_steps = steps[start_idx : end_idx + 1]
    next_step = steps[end_idx + 1] if end_idx + 1 < n_steps else ""

    return PruneDecision(
        can_skip=True,
        skip_start_idx=start_idx,
        skip_end_idx=end_idx,
        skipped_steps=skipped_steps,
        next_step=next_step,
        reason=reason,
        decision_model=decision_model,
        raw_response=raw_text,
    )
