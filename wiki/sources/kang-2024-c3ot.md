---
title: "C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness"
type: source
source_kind: paper
author_or_origin: "Yu Kang, Xianghui Sun, Liangyu Chen, Wei Zou"
published: 2024-12-16
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2412.11664"
reliability: primary
status: processed
tags: [conditioned-sft, compressor, long-short-pair]
---

# C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness

## Scope and Relevance

C3oT trains on *paired* long and short CoTs with different prompt prefixes, then infers with the short prefix. The short CoT is a compressor rewrite (GPT-4 in the main setup), not a skip of identified steps. Establishes that training on short CoT *alone* loses accuracy (consistent with Jin et al.), but conditioned training on both lengths recovers most of it.

## Faithful Summary

Given $\{(x_i, r_i^{\mathrm{long}}, y_i)\}$, a compressor $\mathcal{F}$ (any summarizer; GPT-4 here) produces $r_i^{\mathrm{short}}=\mathcal{F}(r_i^{\mathrm{long}})$. Dataset $D=\{(x_i, r_i^{\mathrm{long}}, r_i^{\mathrm{short}}, y_i)\}$. Conditioned training prepends "Answer and provide a detailed thought process:" vs "…as brief a thought process as possible:". Inference uses the short prefix. Long and short examples need not be paired in a batch.

Trained on LLaMA-2-Chat 7B/13B, GSM8K / MathQA / ECQA / StrategyQA. Short-only SFT drops math accuracy (GSM8K 7B: Long 37.38 vs Short 31.01). C3oT matches Long accuracy with ~50%+ compression (GSM8K 7B: 36.92 acc, 56.67% compression). Ablation: removing the condition tokens hurts both acc and compression. Mixing several compression *levels* (Mixed Conditions) and picking a per-example rate (C3oTAdapt) can beat Long CoT.

They cite Jin et al. 2024: shortening steps while keeping key information diminishes ability — and treat their Short-only numbers as a replication of that claim.

## Extracted Knowledge

- **Mechanism/Process:** Compress long CoT → condition long vs short with prompt tokens → SFT on the mix → infer with short condition.
- **Claim:** Short-CoT-only training loses accuracy even when GPT-4 keeps key information; conditioned long+short training recovers it.
  - Support: Table 1 Long vs Short vs C3oT; Table 2 w/o condition.
  - Scope/conditions: LLaMA-2-Chat 7B/13B; four arithmetic/commonsense datasets with human CoT.
  - Status: observed.
- **Claim:** Compression rate tracks how redundant the original training CoTs are, not the domain per se.
  - Support: GSM8K train CoT 124 → 56 tokens (55%); MathQA 91 → 63 (31%).
  - Status: argued from those four sets.
- **Claim:** Over-compressing the training short CoT (toward 100% = no CoT) makes C3oT fail like Short-only.
  - Support: Figure 2 accuracy vs train compression rate.
  - Status: observed.

## Limitations and Failure Modes

- Requires a compressor and existing long CoTs (human-designed in these datasets).
- Math still slightly lags Long CoT; commonsense can exceed it.
- Conditioned inference needs the short prefix at test time — not an unconditional skip policy.
- Compressor rewrite can change wording; not a structural skip.

## Integration Candidates

- Update [[cot-compression-methods]] as conditioned long/short SFT.
- Update [[supervision-unit-of-compression]]: full short *and* full long traces.
- Tension with Jin et al. belongs on [[cot-compression-methods]].

## Tensions or Contradictions

- Directly engages the "shorter CoT hurts" result: they reproduce it for Short-only and claim conditioned training is the fix.
