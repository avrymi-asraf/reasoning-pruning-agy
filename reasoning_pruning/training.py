"""Fine-tuning tools for training models on Pruning-Transition (PT) datasets.

Big Picture Role: Fine-tunes the base generator model G on next-step transitions
(x -> y) to create the compressed generator model G'.
Code Flow Connection: Consumes `datasets.Dataset` instances from `dataset.py`, formats
transition prompt-completions, and outputs trained checkpoints for `evaluation.py` and `hub.py`.
Execution Environment: Runs on GPU environments (Google Colab T4/A100/L4, cloud instances)
using 4-bit QLoRA with Hugging Face TRL and PEFT.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

import datasets
import torch

from reasoning_pruning.types import TrainResult

logger = logging.getLogger(__name__)


def train_pruning_model(
    dataset: Union[datasets.Dataset, str],
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./checkpoints/rp_model",
    eval_dataset: Optional[Union[datasets.Dataset, str]] = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    report_to: str = "none",
    save_total_limit: int = 2,
    logging_steps: int = 10,
    **trainer_kwargs: Any,
) -> TrainResult:
    """Fine-tune a generator model on next-step Pruning-Transition examples.

    What it does:
        Loads base generator model G with optional 4-bit quantization (QLoRA),
        formats (input_x, target_y) pairs into SFT prompt-completion format,
        trains the model to predict the next useful step directly after pruning,
        and saves the resulting adapter checkpoint and tokenizer.

    When to reach for it:
        - When running fine-tuning on a Google Colab notebook / GPU instance.
        - To update model G to become G' (the pruned model version).

    Parameters:
        dataset: Hugging Face Dataset or path containing `input_x` and `target_y` columns.
        base_model: Hugging Face model identifier (e.g. 'Qwen/Qwen2.5-1.5B-Instruct').
        output_dir: Directory where adapter weights and checkpoints will be saved.
        eval_dataset: Optional evaluation dataset split.
        num_train_epochs: Number of full passes over the dataset.
        per_device_train_batch_size: Micro-batch size per GPU (default: 2 for Colab T4).
        gradient_accumulation_steps: Gradient accumulation steps (effective batch size = batch_size * accum).
        learning_rate: Optimizer learning rate.
        max_seq_length: Maximum sequence length (prompt + completion tokens).
        use_qlora: Enable 4-bit NF4 quantization for memory efficiency.
        lora_r: LoRA rank dimension.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout probability.
        report_to: Tracking backend ('wandb', 'tensorboard', or 'none').
        save_total_limit: Max checkpoints to keep on disk.
        logging_steps: Logging interval.
        **trainer_kwargs: Extra kwargs passed to `SFTConfig` / `TrainingArguments`.

    Returns:
        A TrainResult object containing loss metrics, total steps, and checkpoint paths.
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        raise ImportError(
            f"Training dependencies not found ({e}). Ensure transformers, peft, trl, bitsandbytes are installed."
        )

    # 1. Load Dataset
    if isinstance(dataset, str):
        train_ds = datasets.load_dataset(dataset)["train"] if not os.path.exists(dataset) else datasets.load_from_disk(dataset)
    else:
        train_ds = dataset

    eval_ds = None
    if eval_dataset:
        if isinstance(eval_dataset, str):
            eval_ds = datasets.load_dataset(eval_dataset)["test"] if not os.path.exists(eval_dataset) else datasets.load_from_disk(eval_dataset)
        else:
            eval_ds = eval_dataset

    logger.info(f"Loaded {len(train_ds)} training examples for model fine-tuning.")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Model Quantization & Loading
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quant_config = None

    if use_qlora and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if use_qlora and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA Setup
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Format prompt-completion
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["input_x"])):
            text = f"{example['input_x'][i]}\n\n[Next Step]:\n{example['target_y'][i]}"
            output_texts.append(text)
        return output_texts

    # 6. Training Arguments
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        report_to=report_to,
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if (use_qlora and torch.cuda.is_available()) else "adamw_torch",
        **trainer_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        dataset_text_field="input_x",
        formatting_func=formatting_prompts_func,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 7. Execute Training
    logger.info("Starting SFT fine-tuning on pruning transitions...")
    train_result = trainer.train()

    # 8. Save Final Model Adapter & Tokenizer
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    metrics = train_result.metrics
    logger.info(f"Training completed: {metrics}")

    return TrainResult(
        model_name_or_path=base_model,
        output_dir=final_adapter_dir,
        metrics=metrics,
        train_loss=metrics.get("train_loss", 0.0),
        total_steps=metrics.get("total_flos", 0),
        epochs=num_train_epochs,
    )
