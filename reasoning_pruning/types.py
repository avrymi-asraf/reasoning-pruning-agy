"""Core data structures and schemas for Reasoning Pruning.

All tools consume and return these explicit, type-annotated dataclasses.
No hidden global state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningTrace:
    """Represents a generated reasoning trajectory decomposed into distinct steps.

    Attributes:
        question: The input question or prompt.
        steps: List of discrete reasoning steps/sentences (s_1, s_2, ..., s_n).
        prefix: Any pre-existing reasoning prefix provided prior to generation.
        full_text: The complete raw generated reasoning text.
        token_count: Number of tokens in the generated reasoning trace.
        generator_model: Identifier or checkpoint name of the model that generated this trace.
        metadata: Optional dictionary containing generation parameters (temperature, etc.).
    """

    question: str
    steps: List[str]
    prefix: str = ""
    full_text: str = ""
    token_count: int = 0
    generator_model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningTrace:
        """Instantiate trace from dictionary."""
        return cls(**data)


@dataclass
class PruneDecision:
    """Represents a pruning decision made by the decision model D.

    Attributes:
        can_skip: Whether a skippable reasoning step or span was identified.
        skip_start_idx: 0-based index of the first skippable step (inclusive).
        skip_end_idx: 0-based index of the last skippable step (inclusive).
        skipped_steps: List of strings corresponding to the removed reasoning steps.
        next_step: The first useful reasoning step immediately following the skipped span.
        reason: Justification explaining why the span is redundant without breaking logic.
        decision_model: Model name/version used to make this pruning decision.
        raw_response: Raw completion text from the decision LLM.
    """

    can_skip: bool
    skip_start_idx: int = -1
    skip_end_idx: int = -1
    skipped_steps: List[str] = field(default_factory=list)
    next_step: str = ""
    reason: str = ""
    decision_model: str = ""
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PruneDecision:
        """Instantiate decision from dictionary."""
        return cls(**data)


@dataclass
class TransitionExample:
    """A single Pruning-Transition (PT) training example.

    Represents the local jump:
    input_x = question + useful_prefix
    target_y = next_useful_step (after skipping redundant span)

    Attributes:
        id: Unique identifier (e.g. dataset_id/question_id/depth_1).
        question: Original question.
        input_x: Input context containing the question and useful reasoning prefix.
        target_y: Target continuation containing the next useful step after pruning.
        depth: Pruning depth iteration (1 for first skip, 2 for second skip, etc.).
        skipped_steps: List of removed reasoning steps that were bypassed.
        skip_reason: Explanation of why the step was safely skipped.
        generator_model: Model that generated the original trajectory.
        decision_model: Model that identified the removable unit.
        metadata: Extra metadata (original token count, dataset source, etc.).
    """

    id: str
    question: str
    input_x: str
    target_y: str
    depth: int = 1
    skipped_steps: List[str] = field(default_factory=list)
    skip_reason: str = ""
    generator_model: str = ""
    decision_model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transition example to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TransitionExample:
        """Instantiate transition example from dictionary."""
        return cls(**data)


@dataclass
class RolloutResult:
    """Result of multi-depth iterative pruning on a single question.

    Attributes:
        question: Original question.
        transitions: List of TransitionExamples generated across depths.
        traces: List of ReasoningTraces at each depth step.
        decisions: List of PruneDecisions corresponding to each trace.
        final_pruned_trace: The resulting compressed reasoning chain.
        original_step_count: Number of steps in the initial baseline generation.
        final_step_count: Number of steps in the final pruned chain.
        compression_ratio: Reduction in steps/tokens (1.0 - final/original).
    """

    question: str
    transitions: List[TransitionExample] = field(default_factory=list)
    traces: List[ReasoningTrace] = field(default_factory=list)
    decisions: List[PruneDecision] = field(default_factory=list)
    final_pruned_trace: str = ""
    original_step_count: int = 0
    final_step_count: int = 0
    compression_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rollout result to dictionary."""
        return {
            "question": self.question,
            "transitions": [t.to_dict() for t in self.transitions],
            "traces": [tr.to_dict() for tr in self.traces],
            "decisions": [d.to_dict() for d in self.decisions],
            "final_pruned_trace": self.final_pruned_trace,
            "original_step_count": self.original_step_count,
            "final_step_count": self.final_step_count,
            "compression_ratio": self.compression_ratio,
        }


@dataclass
class TrainResult:
    """Summary of model fine-tuning run.

    Attributes:
        model_name_or_path: Path or identifier of the trained model / adapter.
        output_dir: Directory where checkpoints and artifacts are saved.
        metrics: Dictionary of training and validation metrics.
        train_loss: Final training loss.
        eval_loss: Final evaluation loss (if eval split provided).
        total_steps: Total optimizer steps executed.
        epochs: Number of training epochs completed.
    """

    model_name_or_path: str
    output_dir: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    total_steps: int = 0
    epochs: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize train result to dictionary."""
        return asdict(self)


@dataclass
class EvalSample:
    """Single sample comparison between base and pruned model."""

    question: str
    ground_truth: Optional[str]
    base_response: str
    pruned_response: str
    base_tokens: int
    pruned_tokens: int
    base_correct: Optional[bool]
    pruned_correct: Optional[bool]
    token_savings_pct: float


@dataclass
class EvalResult:
    """Comprehensive evaluation comparing base generator G vs pruned generator G'.

    Attributes:
        benchmark_name: Name of the evaluation benchmark (e.g. 'gsm8k', 'math').
        num_samples: Total number of evaluated examples.
        base_accuracy: Accuracy of the base model (0.0 to 1.0).
        pruned_accuracy: Accuracy of the pruned model (0.0 to 1.0).
        accuracy_delta: Difference in accuracy (pruned_accuracy - base_accuracy).
        base_avg_tokens: Average reasoning tokens generated by base model.
        pruned_avg_tokens: Average reasoning tokens generated by pruned model.
        avg_token_reduction_pct: Percentage of tokens saved across the dataset.
        samples: Detailed per-sample evaluations.
    """

    benchmark_name: str
    num_samples: int
    base_accuracy: Optional[float] = None
    pruned_accuracy: Optional[float] = None
    accuracy_delta: Optional[float] = None
    base_avg_tokens: float = 0.0
    pruned_avg_tokens: float = 0.0
    avg_token_reduction_pct: float = 0.0
    samples: List[EvalSample] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evaluation result to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "num_samples": self.num_samples,
            "base_accuracy": self.base_accuracy,
            "pruned_accuracy": self.pruned_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "base_avg_tokens": self.base_avg_tokens,
            "pruned_avg_tokens": self.pruned_avg_tokens,
            "avg_token_reduction_pct": self.avg_token_reduction_pct,
            "samples": [asdict(s) for s in self.samples],
        }
