## Project Goal

* **Description:** Train LLM reasoning models on their own pruned, compressed transition paths ($x \to y$) so they learn during generation to skip unnecessary intermediate reasoning steps without sacrificing deductive correctness. The code is executed as a lightweight CLI (`uv run rp <command>`) or imported as pure Python tools inside Jupyter/Google Colab notebooks. It is designed to run locally, on remote SSH compute, or inside Google Colab GPU runtimes (T4 / A100 / L4) using `uv` and 4-bit QLoRA.

---

## Project Structure - remember to update it when you make changes

* **Architecture:** The project is designed following the **Code-as-Tools** principle as a small set of composable, notebook-usable, well-documented Python tools rather than a rigid monolithic pipeline. It consists of four core functional layers:
  1. **Generation & Pruning Layer** (`generation.py`, `decision.py`, `segmenter.py`): Generates reasoning traces $G(q)$, prompts decision model $D$ via LiteLLM to find the first safely skippable step/span, and extracts $(x \to y)$ transition pairs across multi-depth rollouts.
  2. **Dataset & Hub Layer** (`dataset.py`, `hub.py`): Builds Hugging Face `Dataset` instances and synchronizes versioned datasets and model adapter checkpoints with lineage cards to Hugging Face Hub.
  3. **Training Layer** (`training.py`): Fine-tunes generator model $G$ on $(x \to y)$ transitions using TRL `SFTTrainer` and 4-bit QLoRA with W&B logging.
  4. **Evaluation & Visualization Layer** (`evaluation.py`, `visualizer.py`): Compares Base $G$ vs Pruned $G'$ on benchmark accuracy and token compression %, and provides in-notebook HTML diffs, Rich terminal outputs, and an interactive Gradio viewer.

* **Code Flow:**
  $$\text{Question } q \xrightarrow[\text{generate\_trace}]{\text{Generator } G} \text{Trace } (s_1..s_n) \xrightarrow[\text{find\_first\_skip}]{\text{Decision } D} \text{Skip } s_k \xrightarrow[\text{extract\_transition}]{\text{Pair}} (x \to y) \xrightarrow[\text{rollout\_pruning}]{\text{Recursive Rollout}} \text{PT Dataset}$$
  $$\text{PT Dataset} \xrightarrow[\text{train\_pruning\_model}]{\text{QLoRA SFT}} \text{Pruned Model } G' \xrightarrow[\text{evaluate\_models}]{\text{Benchmark Eval}} \text{EvalResult (Accuracy + Token Savings \%)}$$

---

## File Structure - remember to update it with the latest project information

```
/home/avreymi/reasoning-pruning
├── pyproject.toml              # uv project definition & package dependencies
├── uv.lock                     # Deterministic lockfile for uv
├── README.md                   # Project overview, quickstart, and CLI instructions
├── AGENTS.md                   # This file (project instructions, architecture, and code rules)
├── .gitignore                  # Git ignore rules for environments, data, checkpoints
├── wiki/                       # Compounding related-work wiki (Karpathy LLM-wiki layout)
│   ├── index.md                # Scope, topic map, tensions, gaps
│   ├── sources/                # One evidence page per paper or local artifact
│   └── pages/                  # Synthesis: auditor method, supervision unit, overthinking, methods
├── scripts/
│   └── colab_setup.sh          # 1-click environment installer for Google Colab
├── notebooks/
│   └── 01_explore_pruning.ipynb # Interactive exploration & trace visualization notebook
├── reasoning_pruning/          # Core importable Python package
│   ├── __init__.py             # Exports top-level tools and types
│   ├── types.py                # Core dataclasses (ReasoningTrace, PruneDecision, etc.)
│   ├── segmenter.py            # Sentence & step boundary segmenter
│   ├── decision.py             # find_first_skip (LiteLLM decision auditor)
│   ├── generation.py           # generate_trace, extract_transition, rollout_pruning
│   ├── dataset.py              # build_pt_dataset, load_pt_dataset
│   ├── training.py             # train_pruning_model (TRL / QLoRA SFT fine-tuning)
│   ├── evaluation.py           # evaluate_models (Dual-model benchmark evaluator)
│   ├── visualizer.py           # render_trace_diff, render_comparison, launch_viewer
│   ├── hub.py                  # push_dataset_to_hf, push_model_to_hf
│   └── cli.py                  # Thin Typer CLI shell over core tools
└── tests/                      # Pytest unit tests
    ├── test_types.py           # Serialization tests for dataclasses
    ├── test_segmenter.py       # Step boundary segmentation tests
    ├── test_decision.py        # Decision parsing and transition extraction tests
    └── test_visualizer.py      # Terminal Rich & Notebook HTML diff renderer tests
```

* `reasoning_pruning/`: Core Python library containing the 9 composable tools and dataclasses.
* `notebooks/`: Jupyter/Colab notebooks designed for interactive experimentation and visualization.
* `scripts/`: Shell automation scripts for environment setup and remote job execution.
* `tests/`: Automated unit test suite run via `uv run pytest`.
* `wiki/`: Related-work wiki. Open `wiki/` (not the repo root) as an Obsidian vault. Start at `wiki/index.md`.

---

## Running and using the code

**Prerequisites:**
* Python 3.10 to 3.12 managed via `uv`.
* LLM API Keys (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`) for decision model $D$ and API generators.
* Hugging Face access token (`HF_TOKEN`) for dataset/model Hub sync.

**Build Steps (if applicable):**
1. Install `uv` (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync all project dependencies:
   ```bash
   uv sync --extra dev
   ```
   *Flow: `uv` resolves the lockfile and creates a localized virtual environment in `.venv/`.*

**Running the Application:**

1. **Test Pruning Rollout on a Single Question (with terminal diff):**
   ```bash
   uv run rp datagen try -q "Janet buys 3 packs of 12 eggs. She bakes 2 cakes using 4 eggs each. How many eggs are left?" --model-g gpt-4o-mini --model-d gpt-4o-mini
   ```
   *Flow: Calls `rollout_pruning`, prints colored step diffs via `render_trace_diff`, and summarizes compression ratio.*

2. **Build Full Pruning-Transition Dataset & Push to Hugging Face:**
   ```bash
   uv run rp datagen build --dataset gsm8k --model-g Qwen/Qwen2.5-1.5B-Instruct --model-d gpt-4o-mini --max-depth 3 --push-to-hf user/rp-gsm8k-qwen1.5b-v1
   ```
   *Flow: Calls `build_pt_dataset` with multithreaded rollout, formats rows, and invokes `push_dataset_to_hf`.*

3. **Fine-Tune with QLoRA on Colab GPU:**
   ```bash
   uv run rp train --dataset user/rp-gsm8k-qwen1.5b-v1 --base-model Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --push-to-hf user/qwen1.5b-rp-v1
   ```
   *Flow: Calls `train_pruning_model`, applies 4-bit quantization and LoRA adapters, runs SFTTrainer, and pushes to HF.*

4. **Evaluate Base vs. Pruned Model:**
   ```bash
   uv run rp eval --base-model Qwen/Qwen2.5-1.5B-Instruct --pruned-model user/qwen1.5b-rp-v1 --benchmark gsm8k
   ```
   *Flow: Calls `evaluate_models`, checks math answers against ground truth, and computes token savings %.*

5. **Launch Interactive Gradio Visualizer:**
   ```bash
   uv run rp view --share
   ```
   *Flow: Calls `launch_viewer` to start the live web playground for interactive pruning and dataset browsing.*

6. **Interactive Data Exploration & Trace Probing via Colab CLI:**
   ```bash
   # 1. Provision a named T4 GPU session
   colab new -s rp-explore --gpu T4

   # 2. Upload library and install dependencies via in-VM uv
   colab upload -s rp-explore reasoning_pruning /content/reasoning_pruning
   colab install -s rp-explore litellm datasets transformers trl peft bitsandbytes rich

   # 3. Interactive Python REPL (variables & traces remain in live kernel memory)
   colab repl -s rp-explore

   # 4. (Optional) Connect browser Web Notebook directly to the CLI VM
   colab url -s rp-explore --open

   # 5. Deallocate VM when finished
   colab stop -s rp-explore
   ```

---

## relevant documents

* [README.md](file:///home/avreymi/code/reasoning-pruning-agy/README.md) — Quickstart guide and CLI command documentation.
* [wiki/index.md](file:///home/avreymi/code/reasoning-pruning-agy/wiki/index.md) — Related-work wiki: CoT compression methods, overthinking taxonomies, and contrast with this repo's $(x \to y)$ auditor.
* [code-as-tools SKILL.md](file:///home/avreymi/code/reasoning-pruning-agy/.agents/skills/code-as-tools/SKILL.md) — Core authoring principles for composable research tools.
* [coding-principles SKILL.md](file:///home/avreymi/code/reasoning-pruning-agy/.agents/skills/coding-principles/SKILL.md) — Project-wide coding and naming invariants.
* [colab-cli SKILL.md](file:///home/avreymi/code/reasoning-pruning-agy/.agents/skills/colab-cli/SKILL.md) — Google Colab CLI compute, secrets (`google.colab.userdata`), and interactive exploration workflows.
* [project-agent-systems SKILL.md](file:///home/avreymi/code/reasoning-pruning-agy/.agents/skills/project-agent-systems/SKILL.md) — Agent system standards and conventions.

---

## Status - remember to update it when you make changes

* Core library implemented with primary composable tools (`generate_trace`, `find_first_skip`, `extract_transition`, `rollout_pruning`, `load_benchmark_questions`, `load_spectrum_benchmarks`, `build_pt_dataset`, `train_pruning_model`, `evaluate_models`, `render_trace_diff`, `launch_viewer`, `push_dataset_to_hf`, `push_model_to_hf`).
* Dynamic default model resolution implemented in `reasoning_pruning.config` (`get_default_model`, `get_default_generator_model`, `get_default_decision_model`), automatically picking active providers (`gemini/gemini-2.5-flash`, `gpt-4o-mini`, `huggingface/Qwen/Qwen2.5-7B-Instruct`, etc.) based on available credentials without crashing when OpenAI keys are omitted.
* Full unit test suite passing (19/19 tests) via `uv run pytest`.
* End-to-end live generation, decision auditing, transition extraction, and multi-depth rollout pruning verified live on Google Colab GPU runtime (`rp-lab`).
* Interactive exploration notebook [`notebooks/01_explore_pruning.ipynb`](file:///home/avreymi/code/reasoning-pruning-agy/notebooks/01_explore_pruning.ipynb) synchronized with robust Colab bootstrapping, dynamic model resolution, and full completion token limits.
* Related-work wiki initialized at [`wiki/index.md`](file:///home/avreymi/code/reasoning-pruning-agy/wiki/index.md): 14 sources (10 full-text, 4 abstract-only) and 4 synthesis pages.
* Next milestone: Scale automated dataset generation across the full QA spectrum and launch QLoRA fine-tuning.

---

## Code Writing Rules
Do not create new documentation files (unless explicitly requested). Only update documentation via the `README` if necessary.

### File Header (Mandatory)
In the header of every code file, you **must** describe how that file relates to the **overall project architecture** and **code flow**.

Each code file **must** include a short description (no more than 4–5 sentences) that explains the following:
- Its role in the **big picture** (as defined in the **Project Structure** section).
- Its connection to the main **code flow** of the project.
- The intended **execution environment** (where this code will run, as defined in the **Project Goal** section).

### Full Observability & Live Execution Invariant (Strict Rule)
- **Zero Mock / Hardcoded Data in Notebooks & Tools**: Research, exploration, and data generation notebooks MUST ALWAYS call and execute the real library tools (`generate_trace`, `find_first_skip`, `extract_transition`, `rollout_pruning`, `build_pt_dataset`).
- **Complete Pipeline Transparency**: Every step from prompt to generated trace, decision parsing, visual diffing, and transition extraction must be executed live through the actual functions. If prompts, models, segmenters, or auditor rubrics change, the notebook must immediately reflect those real changes.
- **Integration & CLI Testing Over Unit Tests**: Do NOT run or create isolated/trivial unit tests (`pytest`, `unittest`). Instead, verify code through live CLI commands (`uv run rp ...`), real end-to-end integration tests, and notebook tool executions with authentic data and models.

Remember to update important documents, remember to update your memory remember to update the relevant skills (if needed).
Shared documents are super super important, they allow you to learn from mistakes and move forward. Remember to use them and update them.

The skills folder has tutorials on how to handle important tools and things. Remember to read - and if you need to update them, update them. Integrate the new information with the existing information, don't reinvent the wheel.
The docs folder has important documents that are only relevant to this project. Plans, etc. If there is a document that relates to your task, use it - and update it. Again - integrate the information, don't reinvent the wheel.
Remember to update your memory. This is important
We are in research and development, not a running product. We are not interested in backward compatibility. It is much more important that the code is clean, clear and readable.
