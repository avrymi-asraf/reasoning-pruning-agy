"""Dataset building and conversion tools for Pruning-Transition (PT) datasets.

Provides tools to convert raw benchmark datasets (GSM8K, MATH, SVAMP, custom lists)
into Hugging Face Datasets containing (input_x -> target_y) transition pairs.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Union

import datasets
from tqdm import tqdm

from reasoning_pruning.generation import rollout_pruning
from reasoning_pruning.types import TransitionExample

logger = logging.getLogger(__name__)


def build_pt_dataset(
    questions: Union[List[str], datasets.Dataset, str],
    generator_model: str = "gpt-4o-mini",
    decision_model: str = "gpt-4o-mini",
    max_depth: int = 3,
    max_samples: Optional[int] = None,
    max_workers: int = 4,
    question_field: str = "question",
    split: str = "train",
    save_path: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> datasets.Dataset:
    """Convert a collection of questions into a Pruning-Transition (PT) Hugging Face Dataset.

    What it does:
        Runs iterative `rollout_pruning` across all input questions (with parallel execution),
        collects all transition pairs (x -> y across depths), and packages them into a
        standard `datasets.Dataset`.

    When to reach for it:
        - Converting an entire benchmark (e.g. GSM8K) into a versioned training dataset.
        - Building custom PT datasets for fine-tuning.

    Parameters:
        questions: Either:
            - A list of question strings.
            - A Hugging Face `Dataset` instance.
            - A Hugging Face dataset identifier string (e.g. 'gsm8k', 'openai/gsm8k').
        generator_model: Model name for generator G.
        decision_model: Model name for decision model D.
        max_depth: Maximum pruning depth per question.
        max_samples: Optional cap on the number of questions to process.
        max_workers: Thread pool concurrency for parallel LLM queries.
        question_field: Name of the column containing the question string if loading HF dataset.
        split: Dataset split if loading from Hugging Face hub (default: 'train').
        save_path: Optional local path to save the dataset (Arrow format).
        api_key: Optional API key.
        **kwargs: Additional parameters passed to `rollout_pruning`.

    Returns:
        A `datasets.Dataset` containing TransitionExample rows ready for training.

    Example:
        >>> ds = build_pt_dataset(["What is 2+2?", "Solve 5x = 25."], max_depth=2, max_workers=2)
        >>> len(ds) >= 0
        True
    """
    # 1. Normalize question list
    question_list: List[str] = []

    if isinstance(questions, str):
        logger.info(f"Loading source dataset '{questions}' (split={split})...")
        hf_ds = datasets.load_dataset(questions, split=split)
        for item in hf_ds:
            if question_field in item:
                question_list.append(str(item[question_field]))
            elif "problem" in item:
                question_list.append(str(item["problem"]))
            elif "prompt" in item:
                question_list.append(str(item["prompt"]))
    elif isinstance(questions, datasets.Dataset):
        for item in questions:
            if question_field in item:
                question_list.append(str(item[question_field]))
            else:
                question_list.append(str(next(iter(item.values()))))
    elif isinstance(questions, (list, tuple)):
        question_list = [str(q) for q in questions]
    else:
        raise ValueError(f"Unsupported questions type: {type(questions)}")

    if max_samples is not None:
        question_list = question_list[:max_samples]

    logger.info(f"Processing {len(question_list)} questions with max_depth={max_depth}, workers={max_workers}...")

    # 2. Parallel rollout execution
    all_transitions: List[TransitionExample] = []

    def _process_one(q: str) -> List[TransitionExample]:
        try:
            res = rollout_pruning(
                question=q,
                generator_model=generator_model,
                decision_model=decision_model,
                max_depth=max_depth,
                api_key=api_key,
                **kwargs,
            )
            return res.transitions
        except Exception as e:
            logger.error(f"Error processing question '{q[:40]}...': {e}")
            return []

    if max_workers > 1 and len(question_list) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_one, q): q for q in question_list}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating PT dataset"):
                all_transitions.extend(fut.result())
    else:
        for q in tqdm(question_list, desc="Generating PT dataset"):
            all_transitions.extend(_process_one(q))

    logger.info(f"Generated {len(all_transitions)} transition pairs from {len(question_list)} questions.")

    # 3. Convert to Hugging Face Dataset
    rows = [t.to_dict() for t in all_transitions]
    if not rows:
        # Create empty dataset with proper features
        rows = [{
            "id": "",
            "question": "",
            "input_x": "",
            "target_y": "",
            "depth": 0,
            "skipped_steps": [],
            "skip_reason": "",
            "generator_model": generator_model,
            "decision_model": decision_model,
            "metadata": {},
        }][:0]

    ds = datasets.Dataset.from_list(rows)

    if save_path:
        ds.save_to_disk(save_path)
        logger.info(f"Saved dataset to {save_path}")

    return ds


def load_pt_dataset(path_or_repo: str, split: Optional[str] = None) -> datasets.Dataset:
    """Load a Pruning-Transition dataset from local disk or Hugging Face Hub.

    Parameters:
        path_or_repo: Local folder path or Hugging Face dataset identifier (e.g. 'username/rp-dataset').
        split: Optional split name if loading from Hugging Face.

    Returns:
        A loaded `datasets.Dataset` instance.
    """
    try:
        # Try local disk arrow first
        return datasets.load_from_disk(path_or_repo)
    except Exception:
        # Try Hugging Face hub / json file
        if split:
            return datasets.load_dataset(path_or_repo, split=split)
        return datasets.load_dataset(path_or_repo)["train"]
