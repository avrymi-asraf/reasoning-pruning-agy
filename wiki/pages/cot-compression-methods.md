---
title: "Methods that train or decode shorter CoT"
type: comparison
status: developing
created: 2026-08-13
updated: 2026-08-13
aliases: [related work, train to skip]
tags: [methods, related-work]
source_pages: [liu-2024-skip-steps, xia-2025-tokenskip, li-2025-step-entropy, jiang-2025-drp, kang-2024-c3ot, deng-2024-stepwise-internalization, chen-2024-do-not-think-that-much, yang-2025-dynamic-early-exit, zhang-2025-tokensqueeze, tang-2026-extra-cot, jang-2025-verbosity-aware, reasoning-pruning-codebase]
related_pages: [supervision-unit-of-compression, first-skippable-span-auditor, overthinking-patterns]
---

# Methods that train or decode shorter CoT

## Core Model

A family of methods make reasoning cheaper by emitting fewer tokens without (they claim) losing the answer. They differ by **signal** (who marks waste), **edit** (delete / rewrite / skip-token / internalize / stop), and **when** (train vs decode). This page is the map; [[supervision-unit-of-compression]] is the axis that most often explains failures to transfer.

## Why It Matters

This project needs a tight related-work story: same goal as the cluster, different construction of the training pair. Headline token cuts cluster around 40–50% with small accuracy change — those numbers are not interchangeable (see Boundaries).

## Explanation

**Train to skip steps.** Liu et al. mix full-step and correct shorter traces, first by requesting fewer steps, optionally warm-started by merging/omitting steps ([[liu-2024-skip-steps]]). Chen et al. self-train o1-like models on first-correct+reflection traces to drop extra *solution rounds* ([[chen-2024-do-not-think-that-much]]). This repo skips a *span* and supervises only the next step ([[reasoning-pruning-codebase]]).

**Train to drop tokens or sentences inside a still-explicit CoT.** TokenSkip: LLMLingua-2 importance, mixed $\gamma$, SFT ([[xia-2025-tokenskip]]). Verbosity-Aware: sentence verbosity, then SFT ([[jang-2025-verbosity-aware]], partial). TokenSqueeze: depth-matched self-samples + linguistic refine ([[zhang-2025-tokensqueeze]], partial). C3oT: GPT-4 compressor rewrite + conditioned long/short SFT ([[kang-2024-c3ot]]).

**Train to replace steps with a marker.** Step Entropy: drop low-entropy `\n\n` steps, SFT+GRPO to emit `[SKIP]` ([[li-2025-step-entropy]]).

**Teacher revises the student's Long-CoT.** DRP: skill decompose + keep/delete/rewrite/merge + SFT on $\hat{R}$ ([[jiang-2025-drp]]). Extra-CoT abstract: learned compressor + mixed-ratio SFT + CHRPO ([[tang-2026-extra-cot]], partial; keep-index story unverified).

**Remove the written CoT.** Stepwise Internalization ([[deng-2024-stepwise-internalization]]).

**Do not train; stop decoding.** Dynamic Early Exit: confidence at transition points ([[yang-2025-dynamic-early-exit]], partial). TRACE also proposes self-loop / backtrack stop heuristics ([[zhang-2025-trace]]).

## Claims and Evidence

- **Claim:** Several independent recipes report ~40–50% token cuts with little accuracy loss on math. Evidence: TokenSkip Qwen2.5-14B GSM8K 313→181, <0.4% drop ([[xia-2025-tokenskip]]); Chen/QwQ MATH500 48.6% tokens, acc held ([[chen-2024-do-not-think-that-much]]); C3oT "up to more than 50%" ([[kang-2024-c3ot]]); Step Entropy trained 35–57% ([[li-2025-step-entropy]]); TokenSqueeze abstract 50% MATH500 ([[zhang-2025-tokensqueeze]], partial). Status: supported as a cluster of *incomparable* point estimates.
- **Claim:** Shortening *without* teaching the model the short form (truncation, "be concise", short-only SFT) loses more accuracy. Evidence: TokenSkip vs Truncation/LC-Prompt; C3oT Short vs C3oT. Status: supported in those papers.
- **Claim:** External teachers help when they edit the *student's* structure (DRP) more than when the student clones the teacher's Short-CoT. Evidence: [[jiang-2025-drp]]. Status: supported there.
- **Claim:** Same-model uncertainty can mark skippable *steps* (entropy) or *sentences* (verbosity) without a teacher. Evidence: [[li-2025-step-entropy]], [[jang-2025-verbosity-aware]] (latter partial). Status: supported / provisional.
- **Claim:** Implicit CoT is a different product: no remaining explicit useful steps. Evidence: [[deng-2024-stepwise-internalization]]. Status: established.

## Relationships

- **Part of →** [[supervision-unit-of-compression]]: each method instantiates one unit.
- **Contrasts with →** [[first-skippable-span-auditor]]: this repo is in the "train to skip" column with a unique unit.
- **Requires →** a notion of waste from [[overthinking-patterns]], whether or not the paper names it.

## Boundaries and Failure Modes

- Do not average the 40–50% cluster: models (Qwen2.5-instruct vs R1-distill vs LLaMA-2-chat vs QwQ), tasks (GSM8K vs MATH500 vs AIME vs ECQA), and units differ.
- Jin et al. 2024 (shortening steps hurts) is cited by C3oT and is *not* ingested; treat the tension as open.
- CoT-Valve, ThinkPrune, SPIRIT, Sui survey: named in related-work sections, not opened.
- Partial pages must not carry load-bearing numbers in a paper draft without a full-text pass.

## Open Questions or Tensions

- Is a first-span local pair a better learning signal than DRP's full $\hat{R}$ for *generation-time* skipping, or do models need to see a complete short rationale?
- Decode-time early exit vs train-time skip: complementary (use both) or redundant?
- Extra-CoT's actual supervision unit is unresolved.

## Sources

- Method sources listed in frontmatter — each contributes one recipe.
- [[zhang-2025-trace]] — stop heuristics, not a training method.
