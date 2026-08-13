---
title: "Extra-CoT: Extreme-Ratio Chain-of-Thought Compression"
type: source
source_kind: paper
author_or_origin: "Yuntian Tang, Bohan Jia, Wenxuan Huang, Lianyue Zhang, Jiao Xie, Wenxi Li, Wei Li, Jie Hu, Xinghao Chen, Rongrong Ji, Shaohui Lin"
published: 2026-02-09
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2602.08324"
reliability: primary
status: partial
tags: [mixed-ratio-sft, rl, keep-indices, extreme-compression]
---

# Extra-CoT: Extreme-Ratio Chain-of-Thought Compression

## Scope and Relevance

The deep-research pass described a GPT-4o teacher that, after atomic segmentation, returns keep-indices $R \subseteq \{1,\ldots,m\}$ and trains mixed-ratio SFT — *not* first-skippable $(x \to y)$ pairs. The current abstract (v5, ICML 2026) describes Extra-CoT: a dedicated semantically-preserved compressor, mixed-ratio SFT, then Constrained and Hierarchical Ratio Policy Optimization (CHRPO). Partial ingest (abstract). The keep-index procedure was not verified in this abstract and should be treated as unverified until the HTML is read.

## Faithful Summary

Abstract: high-ratio CoT compression often loses logical fidelity. Extra-CoT trains a compressor on mathematical CoT with fine-grained annotations, SFT's an LLM on mixed compression budgets, then uses CHRPO so lower budgets are still rewarded for solving the question. On MATH-500 with Qwen3-1.7B: >73% token reduction and +0.6% accuracy. Code: github.com/Mwie1024/Extra-CoT.

## Extracted Knowledge

- **Claim:** Mixed-ratio SFT + hierarchical-ratio RL can exceed 73% token cut on MATH-500 with a small accuracy gain (Qwen3-1.7B).
  - Support: abstract.
  - Status: observed in the paper; **not table-checked**.
- **Mechanism/Process:** Dedicated compressor → mixed-ratio SFT → CHRPO.

## Limitations and Failure Modes

- Partial. Whether supervision is keep-indices, token masks, or rewritten traces is **unknown** from the abstract alone.
- Do not treat the deep-research keep-index description as established by this source page.

## Integration Candidates

- [[cot-compression-methods]] and [[supervision-unit-of-compression]] once full text is read.
- Gap: confirm whether Extra-CoT is keep-indices (deep-research) or a learned compressor (abstract).

## Tensions or Contradictions

- Possible mismatch between the research-pass description (teacher keep-indices) and the abstract (learned compressor + CHRPO). Resolve on full-text ingest.
