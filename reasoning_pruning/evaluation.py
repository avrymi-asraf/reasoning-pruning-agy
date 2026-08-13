"""Evaluation tools for comparing Base vs Pruned reasoning models.

Measures reasoning token compression ratio, accuracy retention on benchmark tasks,
and generates side-by-side evaluation traces.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
from tqdm import tqdm

from reasoning_pruning.generation import generate_trace
from reasoning_pruning.types import EvalResult, EvalSample

logger = logging.getLogger(__name__)


def extract_math_answer(text: str) -> Optional[str]:
    """Extract standard math answer from reasoning text (e.g. '#### 42' or '\\boxed{42}')."""
    # Look for GSM8K delimiter '#### 42'
    gsm_match = re.search(r"####\s*([-\d\.,]+)", text)
    if gsm_match:
        return gsm_match.group(1).strip().replace(",", "")

    # Look for LaTeX boxed: \boxed{answer}
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Look for "The answer is X"
    ans_match = re.search(r"(?:the answer is|equals|final answer:?)\s*([-\d\.,]+)", text, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip().replace(",", "")

    # Fallback to last number in text
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return nums[-1]

    return None


def evaluate_models(
    questions: Union[List[str], datasets.Dataset, str],
    base_model: str,
    pruned_model: str,
    ground_truths: Optional[List[str]] = None,
    benchmark_name: str = "custom",
    max_samples: Optional[int] = None,
    question_field: str = "question",
    answer_field: str = "answer",
    split: str = "test",
    api_key: Optional[str] = None,
    **generation_kwargs: Any,
) -> EvalResult:
    """Evaluate and compare baseline generator G vs fine-tuned pruned generator G'.

    What it does:
        Runs inference with both models on the test questions, calculates token counts,
        extracts answers, verifies correctness against ground truth, and computes
        average token savings percentage and accuracy delta.

    When to reach for it:
        - After fine-tuning to verify if the model successfully compressed reasoning
          without damaging accuracy.
        - In automated experiment loops to decide whether to accept a checkpoint.

    Parameters:
        questions: List of questions, a Hugging Face Dataset, or dataset hub identifier.
        base_model: Base generator model name / adapter path.
        pruned_model: Pruned fine-tuned model name / adapter path.
        ground_truths: Optional list of gold answers.
        benchmark_name: Benchmark label (e.g. 'gsm8k', 'math').
        max_samples: Optional limit on number of evaluation samples.
        question_field: Column name for question when loading from HF.
        answer_field: Column name for gold answer when loading from HF.
        split: Dataset split to load if dataset name is given.
        api_key: Optional API key.
        **generation_kwargs: Generation parameters (temperature, max_tokens, etc.).

    Returns:
        An EvalResult object containing token savings statistics and accuracy metrics.
    """
    q_list: List[str] = []
    gt_list: List[Optional[str]] = []

    if isinstance(questions, str):
        logger.info(f"Loading benchmark dataset '{questions}' (split={split})...")
        hf_ds = datasets.load_dataset(questions, split=split)
        for item in hf_ds:
            q_list.append(str(item.get(question_field, item.get("problem", ""))))
            if answer_field in item:
                gt_list.append(str(item[answer_field]))
            else:
                gt_list.append(None)
    elif isinstance(questions, datasets.Dataset):
        for item in questions:
            q_list.append(str(item.get(question_field, "")))
            gt_list.append(str(item.get(answer_field, "")) if answer_field in item else None)
    elif isinstance(questions, (list, tuple)):
        q_list = [str(q) for q in questions]
        gt_list = [str(gt) for gt in ground_truths] if ground_truths else [None] * len(q_list)

    if max_samples is not None:
        q_list = q_list[:max_samples]
        gt_list = gt_list[:max_samples]

    samples: List[EvalSample] = []
    base_correct_count = 0
    pruned_correct_count = 0
    has_gt = any(g is not None for g in gt_list)

    for q, gt in tqdm(zip(q_list, gt_list), total=len(q_list), desc=f"Evaluating {benchmark_name}"):
        # Generate with base model
        base_trace = generate_trace(question=q, model=base_model, api_key=api_key, **generation_kwargs)
        base_resp = base_trace.full_text
        base_tokens = base_trace.token_count or len(base_resp.split())

        # Generate with pruned model
        pruned_trace = generate_trace(question=q, model=pruned_model, api_key=api_key, **generation_kwargs)
        pruned_resp = pruned_trace.full_text
        pruned_tokens = pruned_trace.token_count or len(pruned_resp.split())

        # Accuracy check
        base_correct = None
        pruned_correct = None

        if gt is not None:
            clean_gt = extract_math_answer(gt) or gt.strip()
            base_ans = extract_math_answer(base_resp)
            pruned_ans = extract_math_answer(pruned_resp)

            base_correct = (base_ans == clean_gt) if base_ans and clean_gt else False
            pruned_correct = (pruned_ans == clean_gt) if pruned_ans and clean_gt else False

            if base_correct:
                base_correct_count += 1
            if pruned_correct:
                pruned_correct_count += 1

        token_savings = ((base_tokens - pruned_tokens) / max(base_tokens, 1)) * 100

        samples.append(
            EvalSample(
                question=q,
                ground_truth=gt,
                base_response=base_resp,
                pruned_response=pruned_resp,
                base_tokens=base_tokens,
                pruned_tokens=pruned_tokens,
                base_correct=base_correct,
                pruned_correct=pruned_correct,
                token_savings_pct=token_savings,
            )
        )

    n = len(samples)
    base_avg_tok = sum(s.base_tokens for s in samples) / max(n, 1)
    pruned_avg_tok = sum(s.pruned_tokens for s in samples) / max(n, 1)
    avg_reduction_pct = ((base_avg_tok - pruned_avg_tok) / max(base_avg_tok, 1)) * 100

    base_acc = (base_correct_count / n) if (has_gt and n > 0) else None
    pruned_acc = (pruned_correct_count / n) if (has_gt and n > 0) else None
    acc_delta = (pruned_acc - base_acc) if (base_acc is not None and pruned_acc is not None) else None

    logger.info(
        f"Eval Completed: Base avg tokens = {base_avg_tok:.1f}, Pruned avg tokens = {pruned_avg_tok:.1f} "
        f"({avg_reduction_pct:.1f}% savings). Acc delta: {acc_delta}"
    )

    return EvalResult(
        benchmark_name=benchmark_name,
        num_samples=n,
        base_accuracy=base_acc,
        pruned_accuracy=pruned_acc,
        accuracy_delta=acc_delta,
        base_avg_tokens=base_avg_tok,
        pruned_avg_tokens=pruned_avg_tok,
        avg_token_reduction_pct=avg_reduction_pct,
        samples=samples,
    )
