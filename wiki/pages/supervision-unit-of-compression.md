---
title: "Supervision unit of CoT compression"
type: comparison
status: developing
created: 2026-08-13
updated: 2026-08-13
aliases: [training target, what is y]
tags: [sft-target, granularity]
source_pages: [liu-2024-skip-steps, xia-2025-tokenskip, li-2025-step-entropy, jiang-2025-drp, kang-2024-c3ot, deng-2024-stepwise-internalization, reasoning-pruning-codebase, jang-2025-verbosity-aware, chen-2024-do-not-think-that-much]
related_pages: [first-skippable-span-auditor, cot-compression-methods]
---

# Supervision unit of CoT compression

## Core Model

"Train the model to be shorter" hides five different prediction targets. Mixing them in one comparison table without naming the unit misreads both accuracy and what $G$ can learn to do at decode time.

## Why It Matters

This repository's distinctive bet is the local unit $(x \to y)$. Published work almost always supervises a *complete* output (short CoT, token-pruned CoT, `[SKIP]`-punctuated CoT, or a bare answer). Future experiments should pick a unit on purpose.

## Dimensions of Comparison

| Unit | What the student predicts | Typical constructor | Decode-time behavior being taught |
| --- | --- | --- | --- |
| Full shorter explicit trace | Entire $r^{\mathrm{short}}$ or mixed $r^{\mathrm{long}}/r^{\mathrm{short}}$ | Random merge/omit + self-sample ([[liu-2024-skip-steps]]); GPT-4 rewrite + condition ([[kang-2024-c3ot]]); teacher prune+rewrite ([[jiang-2025-drp]]); truncate extra solution rounds ([[chen-2024-do-not-think-that-much]]); sentence drop ([[jang-2025-verbosity-aware]]) | Emit a complete short rationale |
| Tokens inside steps | Same steps, fewer words | LLMLingua-2 quantile prune ([[xia-2025-tokenskip]]); TokenSqueeze linguistic refine ([[zhang-2025-tokensqueeze]], partial) | Telegraphic wording; step count held |
| Step replaced by placeholder | Full trace with `[SKIP]` where low-entropy steps were | Step entropy rank ([[li-2025-step-entropy]]) | Emit `[SKIP]` instead of those steps |
| Implicit answer | $y$ only; CoT removed from the left during train | Stepwise Internalization ([[deng-2024-stepwise-internalization]]) | No written intermediates |
| Local next step after a hole | Only $y=$ next kept step given prefix $x$ | First-span auditor ([[reasoning-pruning-codebase]]) | Continue as if the span was never said |

Extra-CoT ([[tang-2026-extra-cot]]) is *not* placed in the table: the abstract describes a learned compressor + mixed-ratio RL; a keep-index story from a secondary summary is unverified.

## Claims and Evidence

- **Claim:** TokenSkip's own analysis says it does not reduce step count. Evidence: [[xia-2025-tokenskip]]. Status: established (authors on their method).
- **Claim:** Dropping whole low-entropy *steps* is safer than dropping low-entropy *tokens* that ignore step boundaries. Evidence: [[li-2025-step-entropy]]. Status: supported (their DeepScaleR/R1-14B comparison). This does not refute TokenSkip, which scores tokens with LLMLingua-2 and keeps high-importance tokens *inside* every step.
- **Claim:** Distilling a teacher's short CoT (different structure) can hurt OOD more than pruning the student's own Long-CoT. Evidence: [[jiang-2025-drp]] Table 4. Status: supported on their math suite.
- **Claim:** Short-CoT-only SFT loses accuracy even when a strong compressor keeps key information; pairing with long CoT under conditions recovers it. Evidence: [[kang-2024-c3ot]]. Status: supported on LLaMA-2 7B/13B × four datasets.
- **Claim:** The only ingested source that supervises isolated $x \to y$ after a first skip is this repo. Evidence: [[reasoning-pruning-codebase]] plus the other rows. Status: supported absence in this set.

## Relationships

- **Applies to →** [[cot-compression-methods]]: methods are implementations of these units. Evidence: each method source.
- **Requires →** [[first-skippable-span-auditor]]: the local unit is undefined without a span picker. Evidence: [[reasoning-pruning-codebase]].
- **Contrasts with →** implicit CoT: internalization *eliminates* the remaining explicit $y$. Evidence: [[deng-2024-stepwise-internalization]].

## Boundaries and Failure Modes

- "50% fewer tokens" is not comparable across rows: TokenSkip 40% is intra-step words on Qwen2.5-14B/GSM8K; DRP 917→328 includes deleting whole reflective rounds on R1-7B; ICoT-SI is 100% CoT tokens gone.
- Partial sources (Verbosity-Aware, TokenSqueeze, Extra-CoT) may move rows after full ingest.

## Open Questions or Tensions

- Can a model trained only on local jumps reconstruct a globally coherent short trace, or does it need some full-trace SFT (Liu/C3oT/DRP mix)?
- Jin et al.'s claim that shorter steps hurt (cited in [[kang-2024-c3ot]]) may apply to some units (short-only rewrite) and not others (student-structure prune, local jump). Not resolved.

## Sources

- All `source_pages` — each contributes one row or a negative ("not this unit").
