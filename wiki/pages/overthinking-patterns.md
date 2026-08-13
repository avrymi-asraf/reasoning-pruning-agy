---
title: "Overthinking patterns in reasoning traces"
type: concept
status: developing
created: 2026-08-13
updated: 2026-08-13
aliases: [archetypes, wasteful patterns, explorer, late landing]
tags: [overthinking, taxonomy]
source_pages: [chen-2024-do-not-think-that-much, zhang-2025-trace, srivastava-2025-llmthinkbench, reasoning-pruning-codebase, jiang-2025-drp]
related_pages: [first-skippable-span-auditor, cot-compression-methods]
---

# Overthinking patterns in reasoning traces

## Core Model

Long traces waste tokens in recurring *kinds* of moves, not as uniform padding. Three ingested taxonomies cut the same phenomenon at different joints: search/discourse (TRACE), waste-vs-help on basic math (LLMThinkBench), and linguistic surface types from this repo's Gemma exploration. They overlap; they are not aliases.

## Why It Matters

An auditor or pruner needs to know what it is allowed to delete. Verification loops and sidetracks are widely reported as disposable. Decomposition is sometimes useful. Preamble and restatement are named here but not in the published inventories.

## Explanation

**Length facts.** o1-like models used ~1,953% more tokens than conventional models on "2+3" ([[chen-2024-do-not-think-that-much]]). Thinking mode is 5–20× slower than non-thinking on simple queries with little gain ([[zhang-2025-trace]]). Extra rounds cluster on *easier* items; the first solution is already correct in >92% of eventually-correct math cases ([[chen-2024-do-not-think-that-much]]). Reasoning-tuned twins can be less accurate *and* ~18× more verbose on atomic arithmetic ([[srivastava-2025-llmthinkbench]]).

**TRACE (discourse / search).** Sub-thoughts are labeled Initial, Final, Verification, Correction, Backtrack, Branching Out, Sidetrack (rambling tangent). Induced graphs: **Explorer** (correct answer appears early; alternatives keep being explored) and **Late Landing** (convergent path, then heavy self-verification). Drivers: over-exploration and over-verification. Utility-based overthinking = continuing after another sub-thought's marginal return $< \epsilon$.

**LLMThinkBench (waste inventory).** Hand labels on ~5,000 basic-math responses: redundant verification loops, self-contradiction loops, irrelevant exploration, pathological stopping (<2%), plus helpful **decomposition** (~11%).

**This repo (surface archetypes).** Conversational Preamble, Question Restatement, Axiom Restatement, Redundant Verification Loops, Null Action Narration, Encyclopedic Detours ([[reasoning-pruning-codebase]], AGENTS.md / notebook). These names are *not* used in the published papers ingested here.

**Approximate alignment (inference, not identity):**

| Local archetype | Nearest published label | Notes |
| --- | --- | --- |
| Redundant Verification Loops | TRACE Verification; ThinkBench verification loops; Chen extra rounds after first correct | Strongest overlap |
| Encyclopedic Detours | TRACE Sidetrack; ThinkBench irrelevant exploration | No paper used "encyclopedic detours" |
| Question / Axiom Restatement | *not named* in ingested papers | Dynamic Early Exit cites Luo et al. for "Problem Restatement" — that paper is a gap |
| Conversational Preamble | *not named* | Could be Initial + filler; untested |
| Null Action Narration | *not named* | — |
| (helpful extra) | ThinkBench decomposition ~11% | Auditor rubric already says do not skip essential derivations |

DRP's teacher delete/merge of redundant and backtracking skill-steps is an operational response to Late-Landing / verification waste ([[jiang-2025-drp]]).

## Claims and Evidence

- **Claim:** Over-verification and over-exploration are the two structural drivers TRACE names. Evidence: [[zhang-2025-trace]]. Status: supported (their graphs); not experimentally isolated.
- **Claim:** Most *basic-math* long-trace waste is verification, contradiction, or irrelevant exploration; decomposition is sometimes helpful. Evidence: [[srivastava-2025-llmthinkbench]]. Status: supported on that task suite.
- **Claim:** Extra solution rounds after the first correct answer add little accuracy and little new strategy on school math. Evidence: [[chen-2024-do-not-think-that-much]]. Status: supported on ASDIV/GSM8K/MATH500 for QwQ and R1.
- **Claim:** This repo's six names are an exploration finding, not a published standard. Evidence: [[reasoning-pruning-codebase]]; absence from the other sources. Status: established as local; unpublished.

## Relationships

- **Applies to →** [[first-skippable-span-auditor]]: the skip rubric is a coarse detector for these wastes. Evidence: [[reasoning-pruning-codebase]].
- **Explains →** why [[cot-compression-methods]] can cut 40–50% tokens without accuracy loss: a large fraction of tokens are these patterns. Evidence: Chen, TRACE, TokenSkip/DRP/C3oT result tables (via method sources).
- **Contrasts with →** helpful decomposition: a length cut that deletes necessary structure should fail (ThinkBench 11%; C3oT over-compression curve; Step Entropy high-entropy prune). Evidence: [[srivastava-2025-llmthinkbench]], [[kang-2024-c3ot]], [[li-2025-step-entropy]].

## Boundaries and Failure Modes

- Taxonomies were built on different distributions (o1 long CoT math, temporal arithmetic, atomic ops, Gemma exploration). Counts do not transfer.
- TRACE labels come from Gemini; ThinkBench from human annotation; local archetypes from project exploration. Different judges.
- "Sidetrack" / "irrelevant exploration" / "encyclopedic detours" are *similar*, not shown to be the same phenomenon.

## Open Questions or Tensions

- TRACE's utility stop vs Chen's tokens-to-first-correct: Explorer traces can have an early correct node and still be exploring. Which stop rule should $D$ approximate?
- Need Luo et al. 2025b before treating "Problem Restatement" as a published match to this repo's Question Restatement.
- No source maps Null Action Narration or Conversational Preamble onto TRACE labels.

## Sources

- [[chen-2024-do-not-think-that-much]] — extra rounds, 1953%, first-correct.
- [[zhang-2025-trace]] — discourse labels, Explorer / Late Landing.
- [[srivastava-2025-llmthinkbench]] — four wastes + decomposition.
- [[reasoning-pruning-codebase]] — six local archetypes.
- [[jiang-2025-drp]] — teacher actions on redundant/backtrack steps.
