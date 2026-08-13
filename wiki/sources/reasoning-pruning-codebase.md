---
title: "reasoning-pruning local pipeline (auditor + (x → y) pairs)"
type: source
source_kind: codebase
author_or_origin: "this repository (reasoning_pruning)"
published: unknown
captured: 2026-08-13
url_or_location: "/home/avreymi/reasoning-pruning/reasoning_pruning/{decision,generation,types}.py"
reliability: primary
status: processed
tags: [first-skip, transition-pair, auditor, unpublished]
---

# reasoning-pruning local pipeline

## Scope and Relevance

Unpublished implementation of the project goal: train $G$ on its own pruned transitions so generation skips unnecessary steps. This is the only source in the wiki that specifies a *first safely skippable span* plus prefix→next-step SFT pairs. It does not report published accuracy/token numbers.

## Faithful Summary

`generate_trace` prompts generator $G$ (LiteLLM) and segments the completion into steps (`segment_steps`). `find_first_skip` prompts a decision model $D$ with indexed steps and a skip rubric (redundancy, detour/fluff, locally skippable; do not skip essential derivations or the final answer). $D$ must return JSON `{can_skip, skip_start_idx, skip_end_idx, reason}`. Traces with fewer than 2 steps cannot be pruned.

`extract_transition` builds one training row when `can_skip` is true and a step exists after the skipped span:

- `input_x` = question + existing prefix + `steps[0 : skip_start_idx]`
- `target_y` = `steps[skip_end_idx + 1]` (the next useful step, *not* the remainder of the trace)

`rollout_pruning` repeats generate → decide → extract up to `max_depth` (default 3), feeding the pruned prefix back into $G$. The resulting PT dataset is intended for QLoRA SFT (`train_pruning_model`).

AGENTS.md and the exploration notebook name six overthinking archetypes found on Gemma-family traces: Conversational Preamble, Question Restatement, Axiom Restatement, Redundant Verification Loops, Null Action Narration, Encyclopedic Detours.

## Extracted Knowledge

- **Definition/Object:** First safely skippable span — the earliest index range $D$ judges can be omitted without breaking downstream deduction.
- **Definition/Object:** Transition example — local pair $(x \to y)$ where $x$ is the useful prefix and $y$ is only the next kept step.
- **Mechanism/Process:** $q \xrightarrow{G} (s_1..s_n) \xrightarrow{D} [k,k'] \xrightarrow{\text{extract}} (x \to y) \xrightarrow{\text{rollout}} \text{PT dataset} \xrightarrow{\text{QLoRA}} G'$.
- **Claim:** The training unit is a local next-step continuation after a hole, not a full rewritten CoT.
  - Support: `extract_transition` docstring and implementation (`generation.py`).
  - Status: established as code behavior.
- **Claim:** $D$ is a separate LiteLLM auditor, not $G$'s entropy or verbosity.
  - Support: `decision.py` `DEFAULT_DECISION_PROMPT` and `find_first_skip`.
  - Status: established as code behavior.
- **Claim:** Six named archetypes characterize Gemma overthinking in this project's exploration.
  - Support: AGENTS.md status; notebook table in `01_explore_pruning.ipynb`.
  - Status: user-provided / exploratory; not a published result.

## Limitations and Failure Modes

- Unpublished; no peer-reviewed comparison to TokenSkip / DRP / Step Entropy.
- If the skip reaches the last step, no pair is emitted (`target_y` missing).
- Auditor quality is whatever $D$ (default `gpt-4o-mini`) decides; no published agreement study in-repo.
- Segmentation mode (`auto` / lines / sentences / paragraphs) changes what a "step" is.
- Multi-depth rollout re-generates from the pruned prefix; $y$ is a *continuation step*, so later original steps after $y$ are not the target.

## Integration Candidates

- Canonical evidence for [[first-skippable-span-auditor]].
- Contrast class in [[supervision-unit-of-compression]] and [[cot-compression-methods]].
- Archetype list for [[overthinking-patterns]].

## Tensions or Contradictions

- No published paper ingested here implements this exact triple: separate auditor + first-span + $(x \to y)$. DRP is the nearest (teacher prune of student steps) but SFT's the whole revised trace.
