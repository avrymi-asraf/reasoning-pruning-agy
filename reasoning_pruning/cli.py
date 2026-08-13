"""Command Line Interface (CLI) for Reasoning Pruning.

Provides a thin Typer shell over the core composable tools.
"""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console

from reasoning_pruning.dataset import build_pt_dataset, load_pt_dataset
from reasoning_pruning.decision import find_first_skip
from reasoning_pruning.evaluation import evaluate_models
from reasoning_pruning.generation import generate_trace, rollout_pruning
from reasoning_pruning.hub import push_dataset_to_hf, push_model_to_hf
from reasoning_pruning.training import train_pruning_model
from reasoning_pruning.visualizer import launch_viewer, render_comparison, render_trace_diff

app = typer.Typer(
    name="rp",
    help="Reasoning Pruning: Train reasoning models on their own pruned transition paths.",
    no_args_is_help=True,
)
datagen_app = typer.Typer(help="Data generation & pruning rollout commands.")
app.add_typer(datagen_app, name="datagen")

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s")


@datagen_app.command("try")
def datagen_try(
    question: str = typer.Option(
        "Janet buys 3 packs of 12 eggs. She bakes 2 cakes using 4 eggs each. How many eggs does she have left?",
        "--question",
        "-q",
        help="Input question to test pruning on.",
    ),
    model_g: str = typer.Option("gpt-4o-mini", "--model-g", help="Generator model name."),
    model_d: str = typer.Option("gpt-4o-mini", "--model-d", help="Decision model name."),
    max_depth: int = typer.Option(2, "--max-depth", "-d", help="Maximum recursive pruning depth."),
):
    """Test pruning rollout on a question and visualize the diff in the terminal."""
    console.print(f"[bold green]Running pruning rollout on:[/bold green] {question}\n")

    def _on_step(depth: int, trace, decision):
        console.print(f"[bold cyan]=== Depth {depth} ===[/bold cyan]")
        console.print(render_trace_diff(trace, decision, as_html=False))

    rollout = rollout_pruning(
        question=question,
        generator_model=model_g,
        decision_model=model_d,
        max_depth=max_depth,
        on_step=_on_step,
    )

    console.print(
        f"\n[bold green]Summary:[/bold green] Original Steps: {rollout.original_step_count} ➔ "
        f"Final Steps: {rollout.final_step_count} "
        f"([bold yellow]{rollout.compression_ratio*100:.1f}% compression[/bold yellow]), "
        f"Extracted {len(rollout.transitions)} PT pairs."
    )


@datagen_app.command("build")
def datagen_build(
    dataset: str = typer.Option("gsm8k", "--dataset", help="Source dataset name or path (e.g. gsm8k)."),
    split: str = typer.Option("train", "--split", help="Dataset split."),
    model_g: str = typer.Option("gpt-4o-mini", "--model-g", help="Generator model name."),
    model_d: str = typer.Option("gpt-4o-mini", "--model-d", help="Decision model name."),
    max_depth: int = typer.Option(3, "--max-depth", help="Max pruning depth."),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Max questions to process."),
    max_workers: int = typer.Option(4, "--workers", help="Worker threads for concurrent queries."),
    save_path: Optional[str] = typer.Option("./data/rp_dataset", "--save-path", help="Local directory to save dataset."),
    push_to_hf: Optional[str] = typer.Option(None, "--push-to-hf", help="Optional HF repo ID to push dataset to."),
):
    """Build a full Pruning-Transition dataset and optionally push to Hugging Face."""
    console.print(f"[bold]Building PT dataset from '{dataset}' using G='{model_g}', D='{model_d}'...[/bold]")

    ds = build_pt_dataset(
        questions=dataset,
        generator_model=model_g,
        decision_model=model_d,
        max_depth=max_depth,
        max_samples=max_samples,
        max_workers=max_workers,
        split=split,
        save_path=save_path,
    )

    console.print(f"[bold green]✓ Successfully created dataset with {len(ds)} transitions.[/bold green]")

    if push_to_hf:
        url = push_dataset_to_hf(
            dataset=ds,
            repo_id=push_to_hf,
            generator_model=model_g,
            decision_model=model_d,
        )
        console.print(f"[bold blue]✓ Pushed dataset to Hugging Face: {url}[/bold blue]")


@app.command("train")
def train_cmd(
    dataset: str = typer.Option(..., "--dataset", help="Path or Hugging Face dataset identifier."),
    base_model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--base-model", help="Base generator model."),
    output_dir: str = typer.Option("./checkpoints/rp_model", "--output-dir", help="Output directory for checkpoint."),
    epochs: int = typer.Option(3, "--epochs", help="Training epochs."),
    batch_size: int = typer.Option(2, "--batch-size", help="Per-device batch size."),
    lr: float = typer.Option(2e-4, "--lr", help="Learning rate."),
    lora_r: int = typer.Option(16, "--lora-r", help="LoRA rank."),
    report_to: str = typer.Option("none", "--report-to", help="Logging backend (e.g. wandb)."),
    push_to_hf: Optional[str] = typer.Option(None, "--push-to-hf", help="Optional HF repo to publish trained adapter."),
):
    """Train a reasoning model on a Pruning-Transition dataset using QLoRA."""
    console.print(f"[bold]Starting QLoRA fine-tuning on '{base_model}' with dataset '{dataset}'...[/bold]")

    result = train_pruning_model(
        dataset=dataset,
        base_model=base_model,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lora_r=lora_r,
        report_to=report_to,
    )

    console.print(f"[bold green]✓ Training complete! Final loss: {result.train_loss:.4f}[/bold green]")
    console.print(f"Artifacts saved to: {result.output_dir}")

    if push_to_hf:
        url = push_model_to_hf(
            model_or_adapter_dir=result.output_dir,
            repo_id=push_to_hf,
            base_model=base_model,
        )
        console.print(f"[bold blue]✓ Model adapter published to Hugging Face: {url}[/bold blue]")


@app.command("eval")
def eval_cmd(
    base_model: str = typer.Option(..., "--base-model", help="Base model identifier."),
    pruned_model: str = typer.Option(..., "--pruned-model", help="Pruned model / adapter identifier."),
    benchmark: str = typer.Option("gsm8k", "--benchmark", help="Benchmark dataset (e.g. gsm8k)."),
    max_samples: int = typer.Option(20, "--max-samples", help="Number of benchmark samples to evaluate."),
):
    """Evaluate and compare Base vs Pruned model accuracy and token compression."""
    console.print(f"[bold]Evaluating Base ('{base_model}') vs Pruned ('{pruned_model}') on {benchmark}...[/bold]")

    eval_result = evaluate_models(
        questions=benchmark,
        base_model=base_model,
        pruned_model=pruned_model,
        benchmark_name=benchmark,
        max_samples=max_samples,
    )

    console.print("\n[bold cyan]=== Evaluation Results ===[/bold cyan]")
    console.print(f"Base Model Avg Tokens:   {eval_result.base_avg_tokens:.1f}")
    console.print(f"Pruned Model Avg Tokens: {eval_result.pruned_avg_tokens:.1f}")
    console.print(f"Average Token Savings:   [bold green]{eval_result.avg_token_reduction_pct:.1f}%[/bold green]")
    if eval_result.base_accuracy is not None:
        console.print(f"Base Accuracy:           {eval_result.base_accuracy*100:.1f}%")
        console.print(f"Pruned Accuracy:         {eval_result.pruned_accuracy*100:.1f}%")
        console.print(f"Accuracy Delta:          {eval_result.accuracy_delta*100:+.1f}%")


@app.command("view")
def view_cmd(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Optional dataset path or HF identifier to load."),
    share: bool = typer.Option(False, "--share", help="Generate a public shareable URL (useful in Colab)."),
    port: int = typer.Option(7860, "--port", help="Port to host Gradio UI."),
):
    """Launch the interactive Gradio visualizer."""
    loaded_ds = None
    if dataset:
        loaded_ds = load_pt_dataset(dataset)
    console.print("[bold green]Starting Reasoning Pruning Visualizer...[/bold green]")
    launch_viewer(dataset=loaded_ds, share=share, port=port)


if __name__ == "__main__":
    app()
