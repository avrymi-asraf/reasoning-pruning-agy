"""Unit tests for configuration and default model resolution in reasoning_pruning.

Big Picture Role: Validates runtime configuration, credential normalization, and default model
resolution for generator model G and decision auditor D across environments.
Code Flow Connection: Tests `get_default_generator_model`, `get_default_decision_model`,
`get_default_model`, and `init_environment` ensuring default models align with project specifications.
Execution Environment: Local Python unit test runner via `pytest`.
"""

import os
from unittest.mock import patch

import pytest

from reasoning_pruning.config import (
    DEFAULT_DECISION_MODEL,
    DEFAULT_GENERATOR_MODEL,
    PROVIDER_DEFAULT_MODELS,
    get_default_decision_model,
    get_default_generator_model,
    get_default_model,
    init_environment,
)


def test_default_generator_model_is_gemma_4_12b():
    """Verify default generator model is canonical google/gemma-4-12B-it."""
    assert DEFAULT_GENERATOR_MODEL == "google/gemma-4-12B-it"

    with patch.dict(os.environ, {}, clear=True):
        # Even with no credentials, generator defaults to google/gemma-4-12B-it
        assert get_default_generator_model() == "google/gemma-4-12B-it"
        assert get_default_model(role="generator") == "google/gemma-4-12B-it"


def test_generator_model_unaffected_by_provider_api_keys():
    """Generator model must remain google/gemma-4-12B-it even when Gemini or OpenAI keys exist."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-gemini-key", "OPENAI_API_KEY": "dummy-openai-key"}, clear=True):
        assert get_default_generator_model() == "google/gemma-4-12B-it"
        assert get_default_model(role="generator") == "google/gemma-4-12B-it"


def test_generator_model_environment_overrides():
    """Generator model should respect RP_MODEL_G, RP_DEFAULT_GENERATOR_MODEL, and RP_DEFAULT_MODEL."""
    with patch.dict(os.environ, {"RP_MODEL_G": "custom-gen-model"}, clear=True):
        assert get_default_generator_model() == "custom-gen-model"

    with patch.dict(os.environ, {"RP_DEFAULT_GENERATOR_MODEL": "custom-gen-2"}, clear=True):
        assert get_default_generator_model() == "custom-gen-2"

    with patch.dict(os.environ, {"RP_DEFAULT_MODEL": "custom-fallback"}, clear=True):
        assert get_default_generator_model() == "custom-fallback"


def test_decision_model_resolution():
    """Decision model should resolve based on credentials or environment overrides."""
    with patch.dict(os.environ, {"RP_MODEL_D": "custom-auditor"}, clear=True):
        assert get_default_decision_model() == "custom-auditor"
        assert get_default_model(role="decision") == "custom-auditor"

    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"}, clear=True):
        assert get_default_decision_model() == "gemini/gemini-2.5-flash"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}, clear=True):
        assert get_default_decision_model() == "gpt-4o-mini"

    with patch.dict(os.environ, {}, clear=True):
        assert get_default_decision_model() == DEFAULT_DECISION_MODEL
