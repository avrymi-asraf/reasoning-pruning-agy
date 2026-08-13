---
title: "TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs"
type: source
source_kind: paper
author_or_origin: "Yuxiang Zhang, Zhengxu Yu, Weihang Pan, Zhongming Jin, Qiang Fu, Deng Cai, Binbin Lin, Jieping Ye"
published: 2025-11-17
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2511.13223"
reliability: primary
status: partial
tags: [long2short, self-generated, linguistic-refinement]
---

# TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs

## Scope and Relevance

A Long2Short method that uses only self-generated data: pick traces whose reasoning *depth* matches problem complexity, then linguistically refine without changing the path. Abstract reports 50% average token cut on MATH500 for DeepSeek-R1-Distill-Qwen-7B with accuracy preserved. Partial ingest (abstract).

## Faithful Summary

Long CoTs from o1/R1-style models raise latency. Prior Long2Short often trades away accuracy. TokenSqueeze (1) selects self-generated samples with depth adapted to problem difficulty, to avoid over-compressing reasoning depth; (2) applies distribution-aligned linguistic refinement to shorten wording while keeping the logical path. No manually curated short-answer set. NeurIPS 2025.

## Extracted Knowledge

- **Claim:** 50% average token reduction on MATH500 with accuracy preserved, R1-Distill-Qwen-7B, self-generated data only.
  - Support: abstract.
  - Status: observed in the paper; **tables not read**.
- **Mechanism/Process:** Depth-matched sample selection + linguistic refinement of the same path.

## Limitations and Failure Modes

- Abstract does not specify how depth is measured or how refinement is trained.
- Appears closer to TokenSkip/C3oT (rewrite/trim wording) than to step deletion.

## Integration Candidates

- [[cot-compression-methods]] as self-generated Long2Short.
- Full-text ingest needed before using the 50% figure as a load-bearing comparison.

## Tensions or Contradictions

- None established beyond the usual “hold accuracy, cut tokens” cluster.
