---
title: "Verbosity-Aware Rationale Reduction"
type: source
source_kind: paper
author_or_origin: "Joonwon Jang, Jaehee Kim, Wonbin Kweon, Seonghyeon Lee, Hwanjo Yu"
published: 2024-12-30
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2412.21006"
reliability: primary
status: partial
tags: [sentence-pruning, verbosity, same-model-signal]
---

# Verbosity-Aware Rationale Reduction

## Scope and Relevance

Sentence-level rationale reduction using a likelihood-based *verbosity* score. Front-of-rationale sentences that pass the criterion are dropped; ordinary CoT SFT continues on the reduced rationale. No separate first-span auditor and no $(x \to y)$ pairs. Captured from the abstract and the deep-research paraphrase; full HTML was not ingested.

## Faithful Summary

The abstract states that token-level reduction without clear criteria underperforms complete-rationale training. The authors instead drop redundant *sentences* using a verbosity criterion derived from likelihoods (the training model, optionally a weaker model). Across reasoning tasks they report +7.71% average performance and −19.87% tokens versus a model trained on complete paths. ACL 2025 Findings.

## Extracted Knowledge

- **Mechanism/Process:** Score sentences by verbosity → drop those that pass → SFT on the reduced rationale.
- **Claim:** Principled sentence-level reduction can raise accuracy while cutting tokens ~20% versus full-rationale SFT.
  - Support: abstract-reported averages only.
  - Status: observed in the paper; **not independently checked in full text here**.
- **Claim:** Token-level reduction without criteria is weaker than full-rationale training.
  - Support: abstract contrast with prior token-level work.
  - Status: argued; details not captured.

## Limitations and Failure Modes

- This wiki page is **partial** (abstract only). Exact verbosity formula, whether sentences are tested strictly from the front, and the role of a weaker model are not verified from primary method text.
- Still a full-trace SFT target after deletion.

## Integration Candidates

- Update [[supervision-unit-of-compression]] as sentence-level, same-model (or weaker-model) scoring.
- Nearby [[li-2025-step-entropy]] (step entropy) and [[first-skippable-span-auditor]] (external first-span).

## Tensions or Contradictions

- None established until full text is read. Deep-research asserted front-to-back testing; treat that as unverified until ingest is completed.
