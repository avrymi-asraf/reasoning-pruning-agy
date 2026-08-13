---
title: "Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy"
type: source
source_kind: paper
author_or_origin: "Zeju Li, Jianyuan Zhong, Ziyang Zheng, Xiangyu Wen, Zhijian Xu, Yingying Cheng, Fan Zhang, Qiang Xu"
published: 2025-08-05
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2508.03346"
reliability: primary
status: processed
tags: [step-entropy, skip-token, sft, grpo, same-model-signal]
---

# Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy

## Scope and Relevance

Defines *step entropy* (mean token Shannon entropy inside a `\n\n`-delimited step) as a same-model signal of redundancy. Low-entropy steps can be replaced by a `[SKIP]` token. After showing that ~80% of lowest-entropy steps can be masked with little accuracy loss, they SFT then GRPO a model to emit `[SKIP]` itself. Closest published *step-level* compressor that uses the generator's own uncertainty rather than a separate auditor.

## Faithful Summary

A CoT $C=(S_1,\ldots,S_N)$ is split on double newlines. Length-normalized step entropy is the mean token entropy in the step. Lemma: mutual information of a step with the answer is bounded by that step's entropy, so low-entropy steps are candidates for redundancy.

Static test: prune lowest / highest / random steps (replace with `[SKIP]`). Up to 80% low-entropy pruning holds accuracy; high-entropy pruning collapses immediately. Token-level low-entropy masking (ignoring step boundaries) drops accuracy after ~20% tokens — they conclude the *step* is the right semantic unit.

They then build a 130k (problem, compressed CoT, answer) set (DeepScaleR + OpenR1-Math, 80% prune) and train DeepSeek-R1-Distill-Qwen-7B: SFT to imitate `[SKIP]` traces, then GRPO with rewards for correctness, skip ratio, skip-count penalty, and length penalty. SFT+GRPO reports 35–57% token cuts with accuracy held or slightly improved (GSM8K 78.54 → 79.15). They compare against CoT-Valve, TokenSkip, R1-Compress.

## Extracted Knowledge

- **Definition/Object:** Length-normalized step entropy — mean token entropy in a step; used as an upper bound on that step's information about the answer.
- **Mechanism/Process:** Score steps → replace lowest-$\kappa$ fraction with `[SKIP]` → SFT on those traces → GRPO to emit `[SKIP]` autonomously.
- **Claim:** ~80% of lowest-entropy steps can be removed with minor accuracy change; high-entropy steps cannot.
  - Support: Figure 1 mask-ratio curves; Table 1 static prune on R1-7B/14B and Qwen3-8B.
  - Scope/conditions: Math traces segmented by `\n\n`; $\kappa=0.8$ chosen on 50 DeepScaleR samples then applied more broadly.
  - Status: observed.
- **Claim:** After SFT+GRPO the model generates compressed traces with `[SKIP]` and 35–57% fewer thinking tokens.
  - Support: Table 3.
  - Status: observed on GSM8K / MATH500 / AIME 2024–25.
- **Claim:** Step-level pruning is safer than token-level entropy pruning.
  - Support: Figure 3 accuracy vs token-mask ratio.
  - Status: observed on DeepScaleR / R1-14B.

## Limitations and Failure Modes

- Segmentation depends on the model emitting `\n\n`; authors list this as a generalization risk.
- Fixed 80% threshold is not claimed to be universal.
- Entropy is the *generator's* confidence, not an independent logical audit — a confidently wrong or confidently restated step can look "skippable."
- `[SKIP]` remains in the generated trace (a placeholder), unlike this repo's jump from prefix $x$ to next useful step $y$.

## Integration Candidates

- Update [[supervision-unit-of-compression]]: step replaced by a skip token, still a full-trace target.
- Update [[cot-compression-methods]] and [[overthinking-patterns]] (low-entropy ≈ predictable restatement/verification).
- Contrast with [[first-skippable-span-auditor]]: same-model entropy vs external first-span auditor.

## Tensions or Contradictions

- TokenSkip also prunes tokens and works; Step Entropy says *token-level entropy* pruning fails. Different importance signal (LLMLingua-2 vs token entropy) and different unit (keep high-importance tokens *inside* every step vs drop whole steps).
