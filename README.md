# 🧠 Reasoning Pruning (`reasoning-pruning`)

> **Train reasoning models on their own pruned, compressed transition paths.**

---

## 🎯 The Core Idea

Standard reasoning fine-tuning trains a model to imitate full-length teacher traces or write summaries. **Reasoning Pruning** teaches the model the **local transition policy**:

$$\underbrace{(q, s_1, \dots, s_{k-1})}_{\text{Question } + \text{ Useful Prefix } (x)} \longrightarrow \underbrace{s_{l+1}}_{\text{Next Useful Step } (y)} \quad [\text{bypassing skippable span } s_k \dots s_l]$$

By training on next-step transitions that bypass redundant calculations, conversational fluff, and uninformative detours, the model learns during natural generation to skip unnecessary intermediate reasoning steps directly.

---

## 🛠️ Code as Tools

This repository is designed as a **small, composable set of well-documented Python tools** rather than a rigid pipeline. Every tool is directly importable in Jupyter/Google Colab notebooks or invokable via a lightweight CLI.

| Tool | Purpose | Input $\to$ Output |
|---|---|---|
| [`generate_trace`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/generation.py) | Generates and segments a reasoning trajectory with model $G$ | `(q, model, prefix)` $\to$ `ReasoningTrace` |
| [`find_first_skip`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/decision.py) | Decision model $D$ finds the first safely skippable step/span | `(trace, decision_model)` $\to$ `PruneDecision` |
| [`extract_transition`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/generation.py) | Extracts the training pair $(x \to y)$ skipping the redundant span | `(trace, decision, depth)` $\to$ `TransitionExample` |
| [`rollout_pruning`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/generation.py) | Recursively prunes and rolls out multi-depth transitions | `(q, G, D, max_depth)` $\to$ `RolloutResult` |
| [`build_pt_dataset`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/dataset.py) | Converts a benchmark dataset into a Hugging Face PT dataset | `(questions, G, D)` $\to$ `Dataset` |
| [`train_pruning_model`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/training.py) | Fine-tunes model using 4-bit QLoRA on Colab GPUs | `(dataset, base_model)` $\to$ `TrainResult` |
| [`evaluate_models`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/evaluation.py) | Evaluates Base $G$ vs Pruned $G'$ on token reduction & accuracy | `(questions, base_G, pruned_G)` $\to$ `EvalResult` |
| [`render_trace_diff`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/visualizer.py) | Visualizes colored trace diffs in notebooks (HTML) or terminal (Rich) | `(trace, decision)` $\to$ `HTML / RichPanel` |
| [`launch_viewer`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/visualizer.py) | Interactive Gradio viewer for data browsing & model battles | `(dataset, eval_result)` $\to$ Gradio App |
| [`push_dataset_to_hf`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/hub.py) | Pushes versioned dataset with documentation card to HF Hub | `(dataset, repo_id)` $\to$ URL |
| [`push_model_to_hf`](file:///home/avreymi/reasoning-pruning/reasoning_pruning/hub.py) | Pushes trained adapter with lineage card to HF Hub | `(adapter_dir, repo_id)` $\to$ URL |

---

## ⚡ Colab-CLI Quickstart (with `uv`)

Inside Google Colab terminal, SSH, or local terminal:

### 1. Install & Sync
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & install dependencies
git clone https://github.com/avrymi-asraf/reasoning-pruning-agy.git
cd reasoning-pruning-agy
uv sync --extra dev
```

### 2. Test Pruning on a Single Example
```bash
uv run rp datagen try -q "Janet buys 3 packs of 12 eggs. She bakes 2 cakes using 4 eggs each. How many eggs are left?" \
  --model-g gpt-4o-mini --model-d gpt-4o-mini
```

### 3. Build Dataset & Push to Hugging Face
```bash
uv run rp datagen build \
  --dataset gsm8k \
  --model-g Qwen/Qwen2.5-1.5B-Instruct \
  --model-d gpt-4o-mini \
  --max-depth 3 \
  --push-to-hf your-hf-username/rp-gsm8k-qwen1.5b-v1
```

### 4. Fine-Tune with QLoRA on Colab GPU
```bash
uv run rp train \
  --dataset your-hf-username/rp-gsm8k-qwen1.5b-v1 \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 3 \
  --push-to-hf your-hf-username/qwen1.5b-rp-v1
```

### 5. Evaluate Base vs Pruned Model
```bash
uv run rp eval \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --pruned-model your-hf-username/qwen1.5b-rp-v1 \
  --benchmark gsm8k \
  --max-samples 50
```

### 6. Launch Interactive Visualization
```bash
uv run rp view --share
```

---

## 📓 Notebook Usage Example

```python
import reasoning_pruning as rp
from IPython.display import HTML, display

# 1. Generate reasoning trajectory
q = "A store sells shirts for $15. If Bob has a 20% coupon and buys 2 shirts, what is his total?"
trace = rp.generate_trace(q, model="gpt-4o-mini")

# 2. Identify first skippable reasoning step
decision = rp.find_first_skip(trace, decision_model="gpt-4o-mini")

# 3. Visualize trace diff directly in cell
display(HTML(rp.render_trace_diff(trace, decision, as_html=True)))

# 4. Multi-depth recursive rollout
rollout = rp.rollout_pruning(q, generator_model="gpt-4o-mini", decision_model="gpt-4o-mini", max_depth=3)
print(f"Compressed {rollout.original_step_count} steps down to {rollout.final_step_count} ({rollout.compression_ratio*100:.1f}% reduction).")
```

---

## 🧪 Testing

```bash
uv run pytest
```
