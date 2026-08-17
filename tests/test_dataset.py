"""Unit tests for dataset loading, formatting, and PT dataset construction tools.

Big Picture Role: Validates dataset ingestion, benchmark normalization, and HF Dataset building.
Code Flow Connection: Tests `load_benchmark_questions`, `load_spectrum_benchmarks`, and `build_pt_dataset`.
Execution Environment: Local Python unit test runner via `pytest`.
"""

from unittest.mock import MagicMock, patch
import pytest
import datasets

from reasoning_pruning.dataset import (
    BENCHMARK_REGISTRY,
    _format_benchmark_row,
    load_benchmark_questions,
    load_spectrum_benchmarks,
    build_pt_dataset,
)
from reasoning_pruning.types import RolloutResult, TransitionExample


def test_format_benchmark_row_multiple_choice():
    row = {
        "question": "Which material is conductive?",
        "choices": {"label": ["A", "B", "C"], "text": ["Wood", "Copper", "Plastic"]},
        "answerKey": "B",
    }
    q, a = _format_benchmark_row(row, "allenai/ai2_arc")
    assert "(A) Wood (B) Copper (C) Plastic" in q
    assert a == "B"


def test_format_benchmark_row_svamp():
    row = {
        "Body": "Dan has 5 apples.",
        "Question": "He eats 2 apples. How many apples are left?",
        "Answer": "3",
    }
    q, a = _format_benchmark_row(row, "ChilleD/SVAMP")
    assert q == "Dan has 5 apples. He eats 2 apples. How many apples are left?"
    assert a == "3"


def test_format_benchmark_row_squad():
    row = {
        "context": "The Eiffel Tower in Paris was completed in 1889.",
        "question": "In what year was the Eiffel Tower completed?",
        "answers": {"text": ["1889"]},
    }
    q, a = _format_benchmark_row(row, "rajpurkar/squad_v2")
    assert "Context: The Eiffel Tower in Paris" in q
    assert "Question: In what year" in q
    assert a == "1889"


def test_load_benchmark_questions_fallback():
    # Test fallback mode when dataset name is simulated offline
    with patch("datasets.load_dataset", side_effect=Exception("Simulated offline")):
        items = load_benchmark_questions("gsm8k", num_samples=2)
        assert len(items) == 2
        assert "question" in items[0]
        assert items[0]["category"] == "Arithmetic & Word Math"


def test_load_spectrum_benchmarks_curated():
    # Test spectrum loading returning multi-category samples
    with patch("datasets.load_dataset", side_effect=Exception("Simulated offline")):
        spectrum = load_spectrum_benchmarks(samples_per_benchmark=1)
        assert len(spectrum) >= 4
        categories = {item["category"] for item in spectrum}
        assert "Arithmetic & Word Math" in categories
        assert "Scientific Reasoning" in categories


def test_build_pt_dataset_from_dicts():
    dict_items = [
        {
            "question": "Solve 2+2.",
            "ground_truth": "4",
            "category": "Arithmetic & Word Math",
            "dataset": "test_ds",
        }
    ]

    mock_transition = TransitionExample(
        id="ex_1",
        question="Solve 2+2.",
        input_x="Solve 2+2.",
        target_y="2+2=4.",
        depth=1,
        skipped_steps=["Let me think."],
        skip_reason="Preamble.",
        generator_model="test-g",
        decision_model="test-d",
    )

    mock_rollout = RolloutResult(
        question="Solve 2+2.",
        transitions=[mock_transition],
        final_pruned_trace="2+2=4.",
        original_step_count=2,
        final_step_count=1,
        compression_ratio=0.5,
    )

    with patch("reasoning_pruning.dataset.rollout_pruning", return_value=mock_rollout):
        hf_ds = build_pt_dataset(
            questions=dict_items,
            max_depth=1,
            max_workers=1,
        )
        assert len(hf_ds) == 1
        row = hf_ds[0]
        assert row["id"] == "ex_1"
        assert row["input_x"] == "Solve 2+2."
        assert row["target_y"] == "2+2=4."
        assert row["metadata"]["category"] == "Arithmetic & Word Math"
        assert row["metadata"]["source_dataset"] == "test_ds"
