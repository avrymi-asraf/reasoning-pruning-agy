"""Reasoning Pruning: Train reasoning models on their own pruned transition paths.

Big Picture Role: Package root exposing the public composable toolset and core dataclasses.
Code Flow Connection: Provides top-level imports (`from reasoning_pruning import generate_trace, ...`)
for notebook experimentation and external scripts.
Execution Environment: Runs universally in local Python, Google Colab notebooks, and CLI processes.
"""

from reasoning_pruning.dataset import build_pt_dataset, load_pt_dataset
from reasoning_pruning.decision import find_first_skip
from reasoning_pruning.evaluation import evaluate_models
from reasoning_pruning.generation import (
    extract_transition,
    generate_trace,
    rollout_pruning,
)
from reasoning_pruning.hub import push_dataset_to_hf, push_model_to_hf
from reasoning_pruning.segmenter import segment_steps
from reasoning_pruning.types import (
    EvalResult,
    EvalSample,
    PruneDecision,
    ReasoningTrace,
    RolloutResult,
    TrainResult,
    TransitionExample,
)
from reasoning_pruning.visualizer import (
    launch_viewer,
    render_comparison,
    render_trace_diff,
)

__all__ = [
    # Tools
    "generate_trace",
    "find_first_skip",
    "extract_transition",
    "rollout_pruning",
    "build_pt_dataset",
    "load_pt_dataset",
    "train_pruning_model",
    "evaluate_models",
    "render_trace_diff",
    "render_comparison",
    "launch_viewer",
    "push_dataset_to_hf",
    "push_model_to_hf",
    "segment_steps",
    # Types
    "ReasoningTrace",
    "PruneDecision",
    "TransitionExample",
    "RolloutResult",
    "TrainResult",
    "EvalResult",
    "EvalSample",
]

__version__ = "0.1.0"
