# AGENTS.md — Reasoning Pruning Agent Guidelines & Project Instructions

> **Project Mission**: Train LLM reasoning models on their own pruned, compressed transition paths ($x \to y$) so they learn during generation to skip unnecessary intermediate reasoning steps without sacrificing deductive correctness.

---

## 1. Tooling & Environment Invariants

- **`uv` (Only)**:
  - All Python dependencies, virtual environments, scripts, and test executions must use `uv`.
  - Commands: `uv sync`, `uv run pytest`, `uv run rp <command>`, `uv add <pkg>`.
  - **NEVER** use bare `pip`, `conda`, `poetry`, or `python` commands without `uv`.
- **Colab-CLI & Cloud Compute First**:
  - The repository is designed to execute locally, in Google Colab, or via `colab-cli` / SSH runtimes with standard GPUs (T4 / A100 / L4).
  - Training defaults to 4-bit QLoRA with `bitsandbytes` + `peft` + `trl`.
- **Hugging Face Hub as the Artifact Store**:
  - Versioned Pruning-Transition datasets are pushed to `hf.co/datasets/<user>/rp-...`.
  - Trained model adapters and checkpoints are pushed to `hf.co/<user>/rp-...` with lineage Model Cards documenting training parameters and benchmark scores.

---

## 2. Core Code & Architectural Principles

### A. Code as Tools (`.agents/skills/code-as-tools`)
1. **Each function is a tool**: Does one meaningful thing, takes explicit arguments, and returns a usable structured object (`ReasoningTrace`, `PruneDecision`, `TransitionExample`, `RolloutResult`, `TrainResult`, `EvalResult`).
2. **No hidden global state**: No process-level side effects, no `sys.exit()`, no swallowing exceptions.
3. **Notebook is the field of experimentation**: Every tool must be directly importable and usable inside a Jupyter/Colab notebook cell (`from reasoning_pruning import ...`).
4. **Thin CLI**: The CLI (`rp` / `reasoning_pruning/cli.py`) is merely a lightweight Typer shell over the core tools, never the only way to access functionality.
5. **Stage-by-Stage Visualization**: Every stage (data generation, pruning decisions, training, model evaluation) must provide visual inspection helpers (`render_trace_diff`, `render_comparison`, `launch_viewer`) for notebook HTML display and terminal Rich output.

### B. Engineering & Style Standards (`.agents/skills/coding-principles`)
1. **One Concept, One Name**: Reuse established names and schemas across the core library, tests, visualizers, and CLI. Do not invent synonyms for existing concepts.
2. **File Header Context**: Every Python file must include a clear docstring explaining what the module does and how it fits into the broader codebase.
3. **Simplicity over Cleverness**: Strive for the single clearest, most direct way to accomplish a task. Avoid decorative abstractions or complex multi-repo pipelines.
4. **Verification Requirement**: Always verify changes by running the test suite (`uv run pytest`) and testing CLI commands before finishing a task.

---

## 3. Composable Toolset & Codebase Map

All primary tools are exported at top level from `reasoning_pruning`:

```
reasoning-pruning/
├── pyproject.toml              # uv project definition & dependencies
├── README.md                   # Quickstart guide & CLI documentation
├── AGENTS.md                   # This file (agent instructions & project invariants)
├── scripts/
│   └── colab_setup.sh          # 1-click Colab environment setup script
├── notebooks/
│   └── 01_explore_pruning.ipynb # Interactive exploration notebook
├── reasoning_pruning/          # Core Python package
│   ├── __init__.py             # Top-level tool exports
│   ├── types.py                # Core dataclasses (ReasoningTrace, PruneDecision, etc.)
│   ├── segmenter.py            # Sentence & step boundary segmenter
│   ├── decision.py             # find_first_skip (LiteLLM decision auditor)
│   ├── generation.py           # generate_trace, extract_transition, rollout_pruning
│   ├── dataset.py              # build_pt_dataset, load_pt_dataset
│   ├── training.py             # train_pruning_model (TRL / QLoRA SFT)
│   ├── evaluation.py           # evaluate_models (Benchmark runner & comparator)
│   ├── visualizer.py           # render_trace_diff, render_comparison, launch_viewer
│   ├── hub.py                  # push_dataset_to_hf, push_model_to_hf
│   └── cli.py                  # Typer CLI entry point
└── tests/                      # Pytest suite
    ├── test_types.py
    ├── test_segmenter.py
    ├── test_decision.py
    └── test_visualizer.py
```

---

## 4. Agent Skill Routing Table

When working in this repository, reference the relevant project-local skills:

| Skill | Location | When to Apply |
|---|---|---|
| `code-as-tools` | `.agents/skills/code-as-tools` | Authoring research, exploration, data-generation, training, or evaluation code. |
| `coding-principles` | `.agents/skills/coding-principles` | Structuring modules, naming conventions, and maintaining codebase clarity. |
| `colab-cli` | `.agents/skills/colab-cli` | Executing remote compute tasks or provisioning Google Colab GPU runtimes. |
| `manage-skills` | `.agents/skills/manage-skills` | Creating, updating, or reviewing skill files (`SKILL.md`). |
| `project-agent-systems` | `.agents/skills/project-agent-systems` | Maintaining project-local agent configurations, memory, and `AGENTS.md`. |
| `create-qa-spectrum` | `.agents/skills/create-qa-spectrum` | Curating and selecting reasoning QA benchmark datasets. |
| `architecture-map` | `.agents/skills/architecture-map` | Generating interactive architectural system diagrams. |

---

## 5. Verification Checklist for Agents

Before completing any task in this codebase:
- [ ] Dependencies managed exclusively with `uv` (`uv sync`).
- [ ] Code follows `code-as-tools` (pure functions, explicit types, notebook-usable).
- [ ] All new functions include docstrings specifying: *What it does*, *When to reach for it*, *Parameters*, and *Returns*.
- [ ] All unit tests pass cleanly: `uv run pytest`.
- [ ] CLI commands tested if modified: `uv run rp --help`.
