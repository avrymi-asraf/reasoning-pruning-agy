---
title: "Do LLMs Really Need 10+ Thoughts for “Find the Time 1000 Days Later”? Towards Structural Understanding of LLM Overthinking"
type: source
source_kind: paper
author_or_origin: "Xinliang Frederick Zhang, Anhad Mohananey, Alexandra Chronopoulou, Pinelopi Papalampidi, Somit Gupta, Tsendsuren Munkhdalai, Lu Wang, Shyam Upadhyay"
published: 2025-10-09
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2510.07880"
reliability: primary
status: processed
tags: [overthinking, thought-graph, explorer, late-landing, trace]
---

# TRACE: Structural Understanding of LLM Overthinking

## Scope and Relevance

TRACE decomposes long thoughts into sub-thoughts, labels discourse relations, and induces thought-progression graphs. It supplies the published transition inventory (Initial, Final, Verification, Correction, Backtrack, Branching Out, Sidetrack) and the Explorer vs Late Landing patterns. Over-verification and over-exploration are named as the two primary drivers. This is taxonomy evidence for [[overthinking-patterns]], not a training method for skip pairs.

## Faithful Summary

Thinking models are 5–20× slower than non-thinking mode on simple queries with little accuracy gain (14 models, 6 domains). TRACE: sample long thoughts → Gemini-2.5-pro splits into self-contained, complete, answer-bearing sub-thoughts and assigns discourse labels → per-query progression graphs (nodes = distinct candidate answers) → cluster graphs by topic/difficulty/#answers.

Two dominant patterns when ≥3 distinct answers appear:
- **Explorer** (Qwen3-235B-A22B): correctness mass spread across nodes; early correct answers exist but alternatives keep being explored; backtracking is common.
- **Late Landing** (R1-Distill-Llama-70B, Qwen3-30B/32B): correctness concentrates on the last node; long self-verification loops after arrival.

They redefine overthinking as continuation after marginal utility of another sub-thought falls below $\epsilon$ (the convergence point). Heuristics: stop after $k$ self-verifications, or stop when a backtrack revisits a prior answer. On Temporal-L3 this halved length while holding or beating vanilla thinking accuracy.

## Extracted Knowledge

- **Definition/Object:** Sub-thought — self-contained, complete, answer-bearing span. Sidetrack — digressive tangent with little value.
- **Definition/Object:** Explorer / Late Landing — two induced graph patterns for how correctness is distributed over successive proposed answers.
- **Claim:** Over-verification and over-exploration are the main structural drivers of overthinking.
  - Support: induced graphs + utility traces (Figure 6): Explorer peaks early then can drop; Late Landing plateaus then self-loops.
  - Scope/conditions: large Qwen3 / R1-distill models; math and temporal tasks; Gemini-as-parser.
  - Status: argued from their graphs; not a causal intervention study.
- **Claim:** Length-to-first-correct (Chen et al.) is a special case of a utility definition with $\Delta$Thought = full thinking-minus-nonthinking length.
  - Support: authors' comparison in §5.
  - Status: argued.

## Limitations and Failure Modes

- Sub-thought split and labels come from Gemini-2.5-pro; parser errors noted (4 Temporal-L3 samples dropped).
- Patterns are model-tied in their writeup, not prompt-tied; that claim is from the models they ran.
- Does not produce training pairs for skipping.
- "Simple" is operationalized as middle-school-solvable, which is still a judgment.

## Integration Candidates

- Primary evidence for [[overthinking-patterns]].
- Sidetrack is the nearest published label to this repo's Encyclopedic Detours.

## Tensions or Contradictions

- Chen et al. treat tokens after first correct as inefficient. TRACE says Explorer can find the answer early *and still be exploring usefully or harmfully*; utility, not first-correct, is the stop rule.
- This repo's six archetypes (preamble, restatement, …) are linguistic surface types; TRACE's labels are discourse/search moves. They overlap (verification ≈ redundant verification loops) but are not the same inventory.
