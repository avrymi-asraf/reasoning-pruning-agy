"""Hugging Face Hub integration tools for datasets and models.

Provides tools to push versioned datasets and model checkpoints with automated
lineage tracking and documentation cards.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Union

import datasets
from huggingface_hub import HfApi, create_repo

from reasoning_pruning.types import EvalResult

logger = logging.getLogger(__name__)


def push_dataset_to_hf(
    dataset: Union[datasets.Dataset, str],
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    generator_model: str = "",
    decision_model: str = "",
    description: str = "",
) -> str:
    """Upload a Pruning-Transition (PT) dataset to the Hugging Face Hub with an automated card.

    What it does:
        Pushes dataset Arrow/parquet files and generates a rich Dataset Card (README.md)
        documenting generator model, decision model, depth counts, and pruning rationale.

    Parameters:
        dataset: Hugging Face Dataset or local disk path.
        repo_id: Hugging Face repo ID (e.g. 'username/rp-gsm8k-qwen1.5b-v1').
        private: Whether the HF repo should be private.
        token: Optional Hugging Face access token.
        generator_model: Model name of generator G.
        decision_model: Model name of decision model D.
        description: Brief description of the dataset.

    Returns:
        The URL of the uploaded Hugging Face dataset.
    """
    if isinstance(dataset, str):
        ds = datasets.load_from_disk(dataset) if os.path.exists(dataset) else datasets.load_dataset(dataset)["train"]
    else:
        ds = dataset

    logger.info(f"Pushing dataset with {len(ds)} rows to Hugging Face: {repo_id}...")
    ds.push_to_hub(repo_id, private=private, token=token)

    # Compute depth breakdown
    depths = ds["depth"] if "depth" in ds.column_names else []
    depth_counts: Dict[int, int] = {}
    for d in depths:
        depth_counts[d] = depth_counts.get(d, 0) + 1

    depth_table = "\n".join(f"| Depth {k} | {v} |" for k, v in sorted(depth_counts.items()))

    card_content = f"""---
language:
- en
license: mit
tags:
- reasoning-pruning
- reasoning
- step-pruning
- sft
size_categories:
- n<{len(ds)+1000}
---

# Reasoning Pruning Dataset: `{repo_id}`

This dataset was generated using the **Reasoning Pruning** toolset. It contains step-by-step transition pairs trained to bypass redundant reasoning sentences.

## Dataset Summary
- **Total Transitions**: {len(ds)}
- **Generator Model ($G$)**: `{generator_model or 'N/A'}`
- **Decision Model ($D$)**: `{decision_model or 'N/A'}`
- **Description**: {description or 'Pruning-Transition dataset.'}

## Depth Breakdown
| Pruning Depth | Example Count |
|---|---|
{depth_table if depth_table else '| All | ' + str(len(ds)) + ' |'}

## Schema
- `id`: Unique identifier (`dataset/question_id/depth`)
- `question`: Original task or problem
- `input_x`: Question + useful reasoning prefix
- `target_y`: Next useful reasoning step after skipping redundant span
- `skipped_steps`: Removed reasoning steps
- `skip_reason`: Decision model rationale
"""
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"Dataset successfully uploaded: {url}")
    return url


def push_model_to_hf(
    model_or_adapter_dir: str,
    repo_id: str,
    base_model: str,
    eval_result: Optional[EvalResult] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload a trained model or adapter checkpoint to Hugging Face Hub with a lineage Model Card.

    What it does:
        Uploads weights and writes a structured Model Card documenting base model lineage,
        pruning token savings %, and benchmark evaluations.

    Parameters:
        model_or_adapter_dir: Local path to adapter or model checkpoint folder.
        repo_id: Hugging Face repo ID (e.g. 'username/qwen1.5b-rp-v1').
        base_model: Identifier of the base generator model.
        eval_result: Optional EvalResult with benchmark metrics.
        private: Whether the HF repo should be private.
        token: Optional HF access token.

    Returns:
        The URL of the uploaded Hugging Face model repository.
    """
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    logger.info(f"Uploading model artifacts from {model_or_adapter_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=model_or_adapter_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    eval_section = ""
    if eval_result:
        eval_section = f"""
## Benchmark Evaluation ({eval_result.benchmark_name})
- **Sample Count**: {eval_result.num_samples}
- **Base Accuracy**: {f"{eval_result.base_accuracy*100:.1f}%" if eval_result.base_accuracy is not None else "N/A"}
- **Pruned Accuracy**: {f"{eval_result.pruned_accuracy*100:.1f}%" if eval_result.pruned_accuracy is not None else "N/A"}
- **Accuracy Delta**: {f"{eval_result.accuracy_delta*100:+.1f}%" if eval_result.accuracy_delta is not None else "N/A"}
- **Base Avg Tokens**: {eval_result.base_avg_tokens:.1f}
- **Pruned Avg Tokens**: {eval_result.pruned_avg_tokens:.1f}
- **Token Reduction**: **{eval_result.avg_token_reduction_pct:.1f}%**
"""

    card_content = f"""---
language:
- en
license: apache-2.0
base_model: {base_model}
tags:
- reasoning-pruning
- lora
- peft
- efficient-reasoning
---

# Pruned Reasoning Model: `{repo_id}`

This model was fine-tuned using **Reasoning Pruning** to compress verbose step-by-step reasoning trajectories into concise, deductive jumps without sacrificing correctness.

## Lineage
- **Base Model ($G$)**: `{base_model}`
- **Training Method**: QLoRA on Next-Step Pruning Transitions $(x \\to y)$
{eval_section}
"""
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Model successfully published: {url}")
    return url
