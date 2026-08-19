"""Test the exact cell and workflow reported by the user."""
import os
import sys

if "/content/reasoning-pruning-agy" not in sys.path:
    sys.path.insert(0, "/content/reasoning-pruning-agy")

from dotenv import load_dotenv
load_dotenv("/content/.env", override=False)
if "GEMINI_TOKEN" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GEMINI_TOKEN"]

import reasoning_pruning as rp

MODEL_G = rp.get_default_generator_model()
MODEL_D = rp.get_default_decision_model()

print(f"🤖 Generator Model (G): {MODEL_G}")
print(f"⚖️  Decision Auditor (D): {MODEL_D}")

# 1. Load GSM8K sample
samples = rp.load_benchmark_questions("gsm8k", num_samples=1, streaming=True)
sample_problem = samples[0]

question_text = sample_problem["question"]
category_name = sample_problem["category"]
dataset_name = sample_problem["dataset"]

print(f"📌 Selected Benchmark: {dataset_name} ({category_name})")
print(f"🎯 Question:\n{question_text}\n")
print(f"🔑 Ground Truth: {sample_problem['ground_truth']}\n")
print(f"🚀 Generating live reasoning trajectory with Generator G ({MODEL_G})...")

# Live generation call
live_trace = rp.generate_trace(
    question=question_text,
    model=MODEL_G,
    temperature=0.7,
    max_tokens=300,
)

print(f"\n✅ Generated {len(live_trace.steps)} reasoning steps ({live_trace.token_count} tokens):")
for i, step in enumerate(live_trace.steps):
    print(f"  [{i}] {step}")

print("\n🚀 Testing Decision Auditor on generated trace...")
decision = rp.find_first_skip(trace=live_trace, decision_model=MODEL_D)
print(f"Audit decision can_skip: {decision.can_skip}")
print(f"Auditor reason: {decision.reason}")
