---
title: "DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition"
type: source
source_kind: paper
author_or_origin: "Yuxuan Jiang, Dawei Li, Francis Ferraro"
published: 2025-05-20
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2505.13975"
reliability: primary
status: processed
tags: [teacher-pruning, skill-decomposition, distillation, long-cot]
---

# DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition

## Scope and Relevance

DRP is the closest *published* neighbor to this repo's auditor: a teacher (GPT-4o) decomposes a *student's* long CoT into skill-labeled steps, then keep / delete / rewrite / merge them, and the student is SFT'd on the full revised trace $(x, \hat{R})$. It does not extract first-skippable-span $(x \to y)$ pairs. The paper argues that training CoTs must stay structurally consistent with the student (pruning the student's own trace beats distilling the teacher's short CoT).

## Faithful Summary

Long-CoT students (R1-Distill-Qwen 1.5B/7B) overthink. Teachers that write Short-CoT create a learnability gap. DRP: student writes Long-CoT $T$; teacher segments $T$ into $(s_i, k_i)$ skill spans; teacher revises each span (keep / delete / rewrite / merge); teacher rewrites the kept spans into a fluent trace $\hat{T}$ in the student's tone; SFT on $(x, \hat{R})$.

On GSM8K, 7B tokens 917 → 328 and Pass@1 91.7% → 94.1%. AIME tokens cut ~43% with accuracy held. Ablations: no decomposition hurts; default sentence split is worse than skill split; direct GPT-4o short-CoT distillation *drops* OOD accuracy (MATH500 92.4 → 88.6) even though traces are shorter. Several teachers (GPT-4o, Gemini 2.0 Flash, ChatGPT, DeepSeek-V3) all help; GPT-4o compresses most. Compared to TALE, CoT-Valve, ThinkPrune.

## Extracted Knowledge

- **Definition/Object:** Skill-based step — a contiguous span labeled with a functional skill (e.g. "Reading given quantity", "Division").
- **Mechanism/Process:** Student Long-CoT → teacher skill decompose → teacher prune/revise → fluent rewrite → student SFT on full $\hat{R}$.
- **Claim:** Pruning the student's own structure transfers better than imitating a teacher's short CoT.
  - Support: Table 4 Distill vs DRP on GSM8K / MATH500 / AIME / AMC.
  - Status: observed.
- **Claim:** Skill-aware segmentation outperforms default step split and no split.
  - Support: Table 2; 12.6 vs 8.3 steps/example on GSM8K; Gemini pairwise judge 33/17 on 50 traces.
  - Status: observed.
- **Claim:** Training CoTs should be informative *and* structurally consistent with the student.
  - Support: authors' stated contribution; consistent with the distillation ablation.
  - Status: argued, empirically supported here.

## Limitations and Failure Modes

- Depends on a capable external teacher at data-generation time.
- Validated mainly on small R1-distill math models; authors note few public small LRMs.
- Teacher rewrite can still change wording of kept steps (not a pure skip).
- Target is the entire revised response, not a local next-step.

## Integration Candidates

- Update [[cot-compression-methods]] as teacher-prunes-student-trace.
- Update [[first-skippable-span-auditor]]: same "external model looks at student steps" family, different supervision unit.
- Update [[overthinking-patterns]]: teacher deletes backtracking / merges redundant skills.

## Tensions or Contradictions

- DRP's teacher can *rewrite* and *merge*, so the student is not learning to jump from a preserved prefix to the next original step. This repo keeps original wording of $x$ and $y$.
