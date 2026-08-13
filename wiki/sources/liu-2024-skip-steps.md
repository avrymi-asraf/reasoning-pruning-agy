---
title: "Can Language Models Learn to Skip Steps?"
type: source
source_kind: paper
author_or_origin: "Tengxiao Liu, Qipeng Guo, Xiangkun Hu, Cheng Jiayang, Yue Zhang, Xipeng Qiu, Zheng Zhang"
published: 2024-11-04
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2411.01855"
reliability: primary
status: processed
tags: [step-skipping, mixed-sft, self-generated-data, ood-generalization]
---

# Can Language Models Learn to Skip Steps?

## Scope and Relevance

Liu et al. ask whether LMs can acquire a human-like ability to omit intermediate reasoning steps, and whether mixing full-step and skipped-step sequences improves efficiency without harming (and sometimes helping) OOD accuracy. It does not use a separate auditor, does not mark a first safely skippable span, and does not extract prefix→next-step pairs. Experiments are on three synthetic tasks with explicit step structure (analog of algebra, multi-digit addition, directional reasoning), using Llama-2 7B and Phi-3-mini.

## Faithful Summary

Humans skip steps as they gain expertise; models have no intrinsic pressure to do so. The authors induce skipping by training a model to obey a requested step count ("Solve it in $n$ steps"), then iteratively prompting it to use fewer than the full count. Correct shorter paths are kept, mixed with the original full-step set $D_0$, and used to train the next iterate. Warm-start data can be made by randomly merging adjacent steps or omitting random ones when cold start fails to produce any valid skips.

After several iterations they train a *standard* model on the mixed set *without* specifying step count at inference. In-domain accuracy stays near-perfect while average steps fall. OOD accuracy is comparable or better (e.g. Llama-2 multi-digit addition OOD-easy 0.06% → 13.97%; directional OOD-hard 42% → 51.8%). They argue skipped paths are not harmful shortcuts. Extra epochs on full-step data alone do not reproduce the OOD gains.

## Extracted Knowledge

- **Definition/Object:** Step skipping — solving a problem with fewer than the original full step count while remaining correct.
- **Mechanism/Process:** Init (cold: SFT on full steps; warm: add human/heuristic skipped data) → iterate: prompt $n-i$ steps, keep correct shorter traces, mix with $D_0$ → train a standard QA model on the mix.
- **Claim:** Models can be trained to skip steps and then, at unconstrained inference, use fewer steps without losing in-domain accuracy.
  - Support: Table 3 iteration-5 vs cold/warm start on three tasks.
  - Scope/conditions: Synthetic compositional tasks; Llama-2 7B and Phi-3-mini; up to 5 (main) or 9 (extended) iterations.
  - Status: observed.
- **Claim:** Mixing self-generated skipped paths with full-step data can improve OOD generalization versus full-step-only training.
  - Support: OOD-easy/hard lifts in Table 3; ablation that more epochs on $D_0$ can overfit rather than help.
  - Scope/conditions: Same three tasks; easy-to-hard compositional splits.
  - Status: observed on these tasks; argued as easy-to-hard generalization.
- **Claim:** Warm start (merge/omit steps) is sometimes required because cold start does not yield step-count following on harder tasks.
  - Support: Multi-digit addition and directional reasoning near-zero skip consistency under cold start.
  - Status: observed.

## Limitations and Failure Modes

- Tasks are synthetic with well-defined step boundaries; transfer to natural-language math CoT is not shown.
- Warm-start heuristics inject human bias about *which* steps to merge.
- Iteration requires knowing or requesting a step count, which the authors themselves call impractical at deployment; the standard model removes that requirement only after data collection.
- Does not identify *which* step is the first safely skippable one.

## Integration Candidates

- Update [[cot-compression-methods]] as the canonical mixed full/skip SFT recipe.
- Update [[supervision-unit-of-compression]]: the target is a *full shorter trace*, not a local $(x \to y)$ pair.
- Contrast with [[first-skippable-span-auditor]]: both skip steps, but Liu et al. supervise the entire shortened path.

## Tensions or Contradictions

- None internal. Contrasts with TokenSkip, which explicitly does *not* reduce step count.
