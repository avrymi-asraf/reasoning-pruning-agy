---
title: "Dynamic Early Exit in Reasoning Models"
type: source
source_kind: paper
author_or_origin: "Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, Weiping Wang"
published: 2025-04-22
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2504.15895"
reliability: primary
status: partial
tags: [early-exit, inference-time, no-training]
---

# Dynamic Early Exit in Reasoning Models

## Scope and Relevance

Inference-time early exit for o1-like models: watch transition points, stop when the model is confident in a trial answer. No training, so it is *not* a skip-SFT method. The deep-research note that long-reasoning models cycle through Problem Restatement & Comprehension / Approach Exploration / Result Verification (often marked by “Wait,” “Alternatively,” “Hmm”) is attributed in that paper to Luo et al. 2025b, which was **not opened**. This page records only what the abstract states.

## Faithful Summary

Abstract: overthinking in long CoT wastes compute and can hurt accuracy. The method monitors behavior at potential reasoning transition points and dynamically terminates the next reasoning chain when the model is highly confident in a trial answer. No extra training; plug-in for existing o1-like models. On 10 benchmarks and 11 reasoning LLMs: CoT length −19.1% to −80.1%, accuracy +0.3% to +5.0%.

## Extracted Knowledge

- **Mechanism/Process:** Inference-time confidence check at transition points → stop further chains.
- **Claim:** Early exit can cut CoT length 19–80% and slightly raise accuracy, without training.
  - Support: abstract-reported ranges only.
  - Status: observed in the paper; **not checked from tables**.

## Limitations and Failure Modes

- Partial ingest (abstract). Chunk taxonomy (restatement / exploration / verification) is **not** established by this source page; it lives in a cited paper that is a gap.
- Complementary to training-based skip methods; does not produce $(x \to y)$ data.

## Integration Candidates

- Update [[cot-compression-methods]] as an inference-only control.
- Do **not** cite this source for the three-chunk cycle until Luo et al. is ingested.

## Tensions or Contradictions

- Chen et al. also want fewer extra rounds; they *train* for that. Yang et al. *decode* for that. Different intervention layer.
