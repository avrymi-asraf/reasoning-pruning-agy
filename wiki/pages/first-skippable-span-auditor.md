---
title: "First-skippable-span auditor and (x → y) pairs"
type: method
status: developing
created: 2026-08-13
updated: 2026-08-13
aliases: [PT dataset, find_first_skip, transition example]
tags: [auditor, skip-span, local-sft]
source_pages: [reasoning-pruning-codebase, jiang-2025-drp, liu-2024-skip-steps, li-2025-step-entropy, jang-2025-verbosity-aware]
related_pages: [supervision-unit-of-compression, cot-compression-methods, overthinking-patterns]
---

# First-skippable-span auditor and (x → y) pairs

## Core Model

A separate decision model $D$ reads an indexed trace from generator $G$ and returns the *first* step or short consecutive span that can be omitted without breaking deduction. Training does not imitate the whole shortened trace. It teaches $G$ the local jump: useful prefix $x$ should be followed by the next kept step $y$.

## Why It Matters

Every nearby paper compresses reasoning. They differ in *who decides what to drop* and *what the student is asked to predict*. This page is the home for the unpublished recipe that this repository actually implements, and for the claim that the combination (external first-span + local pair) is not in the ingested published set.

## Explanation

**Who decides.** $D$ is a LiteLLM auditor with an explicit skip rubric: redundancy, fluff/detour, locally skippable; never drop essential intermediates or the final answer. That is a *logical* judgment, not $G$'s own entropy ([[li-2025-step-entropy]]) or verbosity ([[jang-2025-verbosity-aware]]). DRP ([[jiang-2025-drp]]) also uses an external teacher, but the teacher may keep, delete, *rewrite*, or *merge* any skill span and then SFT's the entire revised response.

**What is first.** The auditor is asked for the earliest safe span, then rollout repeats on the continuation. Liu et al. ([[liu-2024-skip-steps]]) instead ask the model for a *shorter total step count* and keep any correct shorter full path. They do not mark which span was the first safe hole.

**What is supervised.** `input_x` is the question plus steps before `skip_start_idx`. `target_y` is only `steps[skip_end_idx+1]`. If the skip runs to the end, no row is written. Published methods in [[supervision-unit-of-compression]] train on a full (possibly rewritten) completion, a `[SKIP]`-punctuated completion, or an answer with no CoT.

**Inference goal.** After QLoRA on those pairs, $G'$ is supposed to *generate* the jump — it should not emit the skipped span — rather than insert a placeholder or wait for a short-CoT prefix.

## Claims and Evidence

- **Claim:** This repo's training unit is a local prefix→next-step pair after a first skip. Evidence: [[reasoning-pruning-codebase]]. Status: established (code).
- **Claim:** $D$ is a separate model from $G$ and is prompted for the first safe span. Evidence: [[reasoning-pruning-codebase]]. Status: established (code).
- **Claim:** No ingested published paper combines separate auditor + first-span mark + $(x \to y)$ pairs. Evidence: [[jiang-2025-drp]], [[liu-2024-skip-steps]], [[li-2025-step-entropy]], [[jang-2025-verbosity-aware]], [[kang-2024-c3ot]], [[xia-2025-tokenskip]]. Status: supported absence *in this set*; a later paper could exist (recorded as a gap on the index).

## Relationships

- **Operationalizes →** [[overthinking-patterns]]: the auditor's rubric targets redundancy and detours, which overlap published waste types. Evidence: [[reasoning-pruning-codebase]], [[srivastava-2025-llmthinkbench]].
- **Contrasts with →** [[supervision-unit-of-compression]]: same compression goal, different prediction target. Evidence: sources on that page.
- **Competes with →** DRP in [[jiang-2025-drp]]: both use an external model over student steps; DRP rewrites a full trace, this method keeps original $x$ and $y$ wording. Evidence: [[jiang-2025-drp]], [[reasoning-pruning-codebase]].
- **Contrasts with →** [[liu-2024-skip-steps]]: both skip steps; Liu supervises mixed full traces, not the first hole. Evidence: [[liu-2024-skip-steps]].

## Boundaries and Failure Modes

- Unpublished: no benchmark numbers here to compare with TokenSkip's 40% or DRP's 917→328.
- Auditor errors become dataset labels; there is no ingested agreement study.
- Segmentation defines the atom $D$ sees. Skill-aware splits in DRP are a different atom than this repo's sentence/line/`auto` modes.
- Multi-depth rollout *re-generates* after the prune, so $y$ is a continuation, not necessarily the original next step from the first sample.

## Open Questions or Tensions

- Would a DRP-style rewrite of kept steps help or hurt if the goal is to learn *jumps in G's own wording*?
- Is "first skippable" the right search order, or should $D$ return all safe spans (keep-index style — unverified for Extra-CoT)?
- How do the six local archetypes map onto $D$'s actual skip reasons in logged rollouts?

## Sources

- [[reasoning-pruning-codebase]] — implements the method.
- [[jiang-2025-drp]] — nearest external-teacher step pruner; different target.
- [[liu-2024-skip-steps]] — step skipping via shorter full paths.
- [[li-2025-step-entropy]] — same-model step drop with `[SKIP]`.
- [[jang-2025-verbosity-aware]] — same-model sentence drop (partial).
