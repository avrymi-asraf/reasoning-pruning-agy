---
title: "Do LLMs Overthink Basic Math Reasoning? LLMThinkBench"
type: source
source_kind: paper
author_or_origin: "Gaurav Srivastava, Aafiya Hussain, Sriram Srinivasan, Xuan Wang"
published: 2025-07-05
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2507.04023"
reliability: primary
status: processed
tags: [overthinking, benchmark, basic-math, verbosity]
---

# Do LLMs Overthink Basic Math Reasoning? (LLMThinkBench)

## Scope and Relevance

A 53-model benchmark of *basic* arithmetic (14 deterministic tasks) that jointly scores accuracy and token efficiency. Hand annotation of ~5,000 responses from 12 reasoning + 12 standard models yields four wasteful patterns plus one helpful one. Complements TRACE (search-structure) and this repo (linguistic archetypes) as a third taxonomy of waste.

## Faithful Summary

Complex-benchmark skill (GSM8K 90%+) does not transfer to basic ops (sometimes <40% addition). Reasoning-tuned variants emit ~18× more tokens and can be *less* accurate (Phi-4 78.92% / 379 tokens vs Phi-4-reasoning 72.23% / 6,066). Under a 1,024-token cap, reasoning models collapse (up to ~36% absolute). Extra reasoning budget shows diminishing or zero returns (GPT-5 / o-series flat from low→medium→high).

**Overthinking Score** = harmonic mean of accuracy and min-max-normalized token efficiency. Rankings change: mid-size concise models can beat larger verbose ones.

Annotation of long traces: most waste is redundant verification loops, self-contradiction loops, irrelevant exploration, and rare pathological stopping (<2%). Helpful decomposition is ~11%. Authors attribute the waste mainly to CoT supervision, not parameter count or quantization.

## Extracted Knowledge

- **Definition/Object:** Overthinking Score $\mathcal{O}=2AE/(A+E)$ — harmonic mean of accuracy and token efficiency.
- **Claim:** Reasoning-tuned models use far more tokens on basic math and often lose accuracy versus their concise twins.
  - Support: Table 2 matched pairs (Phi-4 vs Phi-4-reasoning; Qwen2.5-14B vs Qwen3-14B).
  - Status: observed.
- **Claim:** Four wasteful patterns cover most long traces; decomposition is the main helpful extra-token pattern (~11%).
  - Support: hand annotation of ~5,000 responses.
  - Scope/conditions: 12+12 models; basic arithmetic tasks, not GSM8K word problems.
  - Status: observed.
- **Claim:** GSM8K-style success is not evidence of basic arithmetic competence.
  - Support: Figure 2 Qwen2.5 family GSM8K vs basic-math gap.
  - Status: observed.

## Limitations and Failure Modes

- Tasks are atomic arithmetic, not multi-step word problems — overthinking here is often performative explanation of $234+567$.
- Dynamic generation reduces contamination but not heuristic pattern-matching.
- Parsers can mis-extract answers.
- Taxonomic names differ from TRACE and from this repo; mapping is interpretive.

## Integration Candidates

- Update [[overthinking-patterns]] with the four waste + one help inventory.
- Mention Overthinking Score on [[cot-compression-methods]] as an evaluation option this repo does not yet use.

## Tensions or Contradictions

- Helpful decomposition (~11%) warns against deleting *every* extra step — relevant to an auditor that must not skip essential derivations.
- Irrelevant exploration ≈ TRACE Sidetrack ≈ this repo Encyclopedic Detours, but measured on different task distributions.
