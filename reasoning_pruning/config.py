"""Configuration and environment initialization for reasoning_pruning.

Big Picture Role: Manages runtime configuration, environment variable synchronization,
Colab secrets (google.colab.userdata), and dynamic default LLM model resolution for
generators and decision auditors.
Code Flow Connection: Executed automatically on package import to ensure all downstream
tools (generate_trace, find_first_skip, rollout_pruning, build_pt_dataset) have valid
API credentials and sensible default models configured.
Execution Environment: Runs seamlessly across local development environments, Google Colab
runtimes (T4/A100/L4), and remote compute nodes.
"""

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Canonical default models per provider
PROVIDER_DEFAULT_MODELS = {
    "gemini": "gemini/gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "deepseek": "deepseek/deepseek-chat",
    "huggingface": "huggingface/Qwen/Qwen2.5-7B-Instruct",
}

_ENV_INITIALIZED = False


def init_environment(force: bool = False) -> None:
    """Initialize and synchronize environment variables and credentials.

    What it does:
        1. Auto-loads credentials from Google Colab secrets (`google.colab.userdata`).
        2. Auto-loads keys from `.env` files across standard search locations.
        3. Normalizes token aliases (e.g. GEMINI_TOKEN / GOOGLE_API_KEY -> GEMINI_API_KEY, HUGGINGFACE_TOKEN -> HF_TOKEN).
    """
    global _ENV_INITIALIZED
    if _ENV_INITIALIZED and not force:
        return

    # 1. Colab secrets via google.colab.userdata
    try:
        from google.colab import userdata

        colab_keys = [
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_TOKEN",
            "WANDB_API_KEY",
            "WANDB_TOKEN",
            "RP_MODEL_G",
            "RP_MODEL_D",
            "RP_DEFAULT_MODEL",
        ]
        for key in colab_keys:
            try:
                val = userdata.get(key)
                if val and key not in os.environ:
                    os.environ[key] = str(val).strip()
            except Exception:
                pass
    except (ImportError, ModuleNotFoundError):
        pass

    # 2. Search and load .env files
    env_paths = [
        "/content/.env",
        "/content/reasoning-pruning-agy/.env",
        ".env",
        os.path.join(os.getcwd(), ".env"),
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                from dotenv import load_dotenv

                load_dotenv(env_path, override=False)
            except ImportError:
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip("'\"")
                                if k and k not in os.environ:
                                    os.environ[k] = v
                except Exception:
                    pass

    # 3. Normalize aliases
    if "GEMINI_TOKEN" in os.environ and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = os.environ["GEMINI_TOKEN"]
    if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    if "HUGGINGFACE_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    if "HUGGING_FACE_HUB_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
    if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    if "HF_TOKEN" in os.environ and "HUGGINGFACE_TOKEN" not in os.environ:
        os.environ["HUGGINGFACE_TOKEN"] = os.environ["HF_TOKEN"]

    if "ANTHROPIC_TOKEN" in os.environ and "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_TOKEN"]
    if "DEEPSEEK_TOKEN" in os.environ and "DEEPSEEK_API_KEY" not in os.environ:
        os.environ["DEEPSEEK_API_KEY"] = os.environ["DEEPSEEK_TOKEN"]
    if "WANDB_TOKEN" in os.environ and "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_TOKEN"]

    _ENV_INITIALIZED = True


def get_default_model(role: str = "generator") -> str:
    """Resolve the optimal default LLM model for a given role based on available credentials.

    Parameters:
        role: "generator" (for model G) or "decision" (for auditor D).

    Returns:
        String model identifier supported by LiteLLM.
    """
    init_environment()

    # 1. Explicit role-specific environment override
    if role == "generator" and "RP_MODEL_G" in os.environ:
        return os.environ["RP_MODEL_G"]
    if role == "decision" and "RP_MODEL_D" in os.environ:
        return os.environ["RP_MODEL_D"]

    # 2. General default model override
    if "RP_DEFAULT_MODEL" in os.environ:
        return os.environ["RP_DEFAULT_MODEL"]

    # 3. Detect by available API credentials (prioritizing Gemini & OpenAI)
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if gemini_key:
        return PROVIDER_DEFAULT_MODELS["gemini"]
    if openai_key:
        return PROVIDER_DEFAULT_MODELS["openai"]
    if anthropic_key:
        return PROVIDER_DEFAULT_MODELS["anthropic"]
    if deepseek_key:
        return PROVIDER_DEFAULT_MODELS["deepseek"]
    if hf_token:
        return PROVIDER_DEFAULT_MODELS["huggingface"]

    # Default fallback
    return PROVIDER_DEFAULT_MODELS["gemini"]


def get_default_generator_model() -> str:
    """Convenience accessor for default generator model G."""
    return get_default_model(role="generator")


def get_default_decision_model() -> str:
    """Convenience accessor for default decision auditor model D."""
    return get_default_model(role="decision")
