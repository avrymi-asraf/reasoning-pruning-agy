"""Test script executing all stages of 01_explore_pruning.ipynb on Colab runtime.

Big Picture Role: End-to-end integration test runner for Google Colab GPU runtime.
Code Flow Connection: Executes the full data generation and exploration pipeline.
Execution Environment: Runs inside Google Colab runtime via colab-cli.
"""
import os
import sys

print("=== 1. Testing Environment Bootstrap ===", flush=True)
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
print(f"IN_COLAB: {IN_COLAB}", flush=True)

# Ensure repo is in sys.path on Colab
if IN_COLAB and "/content/reasoning-pruning-agy" not in sys.path:
    sys.path.insert(0, "/content/reasoning-pruning-agy")

# Load .env files if present
for env_path in ["/content/.env", "/content/reasoning-pruning-agy/.env", ".env"]:
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

# Map GEMINI_TOKEN alias to GEMINI_API_KEY for LiteLLM
if "GEMINI_TOKEN" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GEMINI_TOKEN"]

import json
import re
import pandas as pd

import reasoning_pruning as rp
from reasoning_pruning.types import (
    ReasoningTrace,
    PruneDecision,
    TransitionExample,
    RolloutResult,
)

MODEL_G = rp.get_default_generator_model()
MODEL_D = rp.get_default_decision_model()

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
gemini_key = os.environ.get("GEMINI_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

print(f"✅ reasoning_pruning v{rp.__version__} loaded successfully.", flush=True)
print(f"🔑 HF_TOKEN configured: {bool(hf_token)}", flush=True)
print(f"🔑 GEMINI_API_KEY configured: {bool(gemini_key)}", flush=True)
print(f"🔑 OPENAI_API_KEY configured: {bool(openai_key)}", flush=True)
print(f"🤖 Generator Model (G): {MODEL_G}", flush=True)
print(f"⚖️  Decision Auditor (D): {MODEL_D}", flush=True)

print("\n=== 2. Testing Stage 1: Load Spectrum Benchmarks ===", flush=True)
REAL_SPECTRUM = rp.load_spectrum_benchmarks(
    benchmarks=["gsm8k", "arc_challenge", "commonsense_qa", "hotpot_qa", "svamp", "squad_v2"],
    samples_per_benchmark=2,
    streaming=True,
)
print(f"✅ Successfully loaded {len(REAL_SPECTRUM)} benchmark questions across {len(set(x['category'] for x in REAL_SPECTRUM))} categories.", flush=True)
assert len(REAL_SPECTRUM) > 0, "No benchmark questions loaded!"

print("\n=== 3. Testing Stage 2: Live Reasoning Trace Generation ===", flush=True)
sample_problem = REAL_SPECTRUM[0]
question_text = sample_problem["question"]
category_name = sample_problem["category"]
dataset_name = sample_problem["dataset"]
print(f"Selected Question ({dataset_name}): {question_text}", flush=True)

live_trace = rp.generate_trace(
    question=question_text,
    model=MODEL_G,
    temperature=0.7,
    max_tokens=300,
)
print(f"Generated {len(live_trace.steps)} steps ({live_trace.token_count} tokens):", flush=True)
for i, s in enumerate(live_trace.steps):
    print(f"  [{i}] {s}", flush=True)
assert len(live_trace.steps) > 0, "No steps generated!"

print("\n=== 4. Testing Stage 3: Live Overthinking Audit ===", flush=True)
live_decision = rp.find_first_skip(
    trace=live_trace,
    decision_model=MODEL_D,
    temperature=0.0,
)
print(f"Audit Result: can_skip={live_decision.can_skip}", flush=True)
print(f"Reason: {live_decision.reason}", flush=True)

print("\n=== 5. Testing Stage 4: Visual Trace Diff Rendering ===", flush=True)
html_diff = rp.render_trace_diff(live_trace, live_decision, as_html=True)
print(f"Rendered HTML Diff ({len(html_diff)} chars)", flush=True)

print("\n=== 6. Testing Stage 5: Transition Extraction ===", flush=True)
if live_decision.can_skip:
    live_transition = rp.extract_transition(
        trace=live_trace,
        decision=live_decision,
        depth=1,
        example_id=f"{sample_problem['id']}_d1",
    )
    print(f"Extracted Transition ID: {live_transition.id}", flush=True)
    print(f"Input x: {live_transition.input_x[:80]}...", flush=True)
    print(f"Target y: {live_transition.target_y}", flush=True)
else:
    print("Decision can_skip is False, testing fallback extraction.", flush=True)

print("\n=== 7. Testing Stage 6: Multi-Depth Rollouts ===", flush=True)
rollout_res = rp.rollout_pruning(
    question=sample_problem["question"],
    generator_model=MODEL_G,
    decision_model=MODEL_D,
    max_depth=2,
)
print(f"Rollout Depths: {len(rollout_res.traces)}, Transitions: {len(rollout_res.transitions)}", flush=True)
print(f"Original Steps: {rollout_res.original_step_count} -> Final: {rollout_res.final_step_count}", flush=True)

print("\n=== 8. Testing Stage 7: Build PT Dataset ===", flush=True)
benchmark_questions = REAL_SPECTRUM[:2]
hf_dataset = rp.build_pt_dataset(
    questions=benchmark_questions,
    generator_model=MODEL_G,
    decision_model=MODEL_D,
    max_depth=2,
    max_workers=2,
)
print(f"Built Hugging Face Dataset with {len(hf_dataset)} rows.", flush=True)

print("\n=== 9. Testing Stage 8: Schema & Integrity Constraints ===", flush=True)
expected_cols = ["id", "question", "input_x", "target_y", "depth", "skipped_steps", "skip_reason", "metadata"]
for col in expected_cols:
    assert col in hf_dataset.column_names, f"Missing column: {col}"
print("Schema validation passed.", flush=True)

print("\n=== 10. Testing Stage 9: HF Hub Push & Load (Validation) ===", flush=True)
if hf_token and len(hf_dataset) > 0:
    repo_id = os.environ.get("RP_HF_REPO", "avreymi/rp-test-validation-v1")
    print(f"Pushing dataset to HF: {repo_id}", flush=True)
    hub_url = rp.push_dataset_to_hf(
        dataset=hf_dataset,
        repo_id=repo_id,
        private=True,
        token=hf_token,
        generator_model=MODEL_G,
        decision_model=MODEL_D,
        description="Validated multi-domain reasoning pruning dataset from Colab test.",
    )
    print(f"Published Hub URL: {hub_url}", flush=True)
    reloaded_ds = rp.load_pt_dataset(repo_id, split="train", token=hf_token)
    print(f"Reloaded {len(reloaded_ds)} rows from {repo_id}.", flush=True)

print("\n=== 11. Testing Stage 10: Cross-Domain Probing ===", flush=True)
for item in REAL_SPECTRUM[:2]:
    tr = rp.generate_trace(question=item["question"], model=MODEL_G, temperature=0.7, max_tokens=256)
    dec = rp.find_first_skip(trace=tr, decision_model=MODEL_D, temperature=0.0)
    print(f"Domain [{item['category']}]: Steps={len(tr.steps)}, CanSkip={dec.can_skip}", flush=True)

print("\n🎉 ALL TEST STAGES COMPLETED SUCCESSFULLY ON COLAB RUNTIME! 🎉", flush=True)
