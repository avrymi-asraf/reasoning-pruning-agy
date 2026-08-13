---
title: "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step"
type: source
source_kind: paper
author_or_origin: "Yuntian Deng, Yejin Choi, Stuart Shieber"
published: 2024-05-23
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2405.14838"
reliability: primary
status: processed
tags: [implicit-cot, curriculum, no-intermediate-steps]
---

# From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step

## Scope and Relevance

This paper does not teach a model to *skip* a span and continue explicit reasoning. It *removes* intermediate CoT tokens from the left during training until the model answers with no written steps. Endpoint is implicit CoT (ICoT-SI), not a shorter explicit trace. Relevant as the opposite pole of this repo's goal (keep remaining useful explicit steps).

## Faithful Summary

Start from a model trained to emit full CoT $z_{1:m}$ then answer $y$. Each stage removes $s(t)$ CoT tokens from the *beginning* and finetunes on the remainder. Linear schedule $s(t)=\lfloor \Delta t / T \rfloor$, plus "removal smoothing" (small chance of removing extra tokens) and optimizer reset when the removal count increases. Right-side removal works worse.

Results: GPT-2 Small reaches 99% on 9×9 multiplication (No-CoT fails at 4×4). Mistral 7B exceeds 50% GSM8K with *no* intermediate steps (GPT-4 No-CoT reported at 44%). Accuracy still lags explicit CoT (Mistral GSM8K explicit 0.68 vs ICoT-SI 0.51) but inference is much faster (up to 11× on 9×9). Intermediate checkpoints trade speed vs accuracy. Training is expensive and unstable if $\Delta$ is too large. Implicit models lose readable steps.

## Extracted Knowledge

- **Definition/Object:** Implicit CoT — CoT allowed at train time as supervision, forbidden as generated tokens at test time; reasoning is internalized in hidden states.
- **Mechanism/Process:** Curriculum: drop prefix CoT tokens, finetune, repeat until none remain.
- **Claim:** Gradual left-removal internalization outperforms No-CoT training and Deng et al. 2023 hidden-state distillation (ICoT-KD).
  - Support: multiplication table (GPT-2 Small 9×9 ≈ 0.99); GSM8K table (Mistral 0.51).
  - Scope/conditions: multiplication (synthetic) and GSM8K (augmented Deng 2023 data); GPT-2 / Phi-3 / Mistral.
  - Status: observed.
- **Claim:** Removing tokens from the end of CoT is harder than from the beginning.
  - Support: ablation "Right-Side Removal" fails to recover.
  - Status: observed on 7×7 multiplication / GPT-2 Small.

## Limitations and Failure Modes

- Opposite of this project's desired behavior: remaining useful *explicit* steps after a skip are not the training target.
- High train cost (one finetune stage per removed token block); long CoTs make this worse.
- Unstable under aggressive $\Delta$; some seeds never recover.
- No interpretability of the internalized steps.
- Not shown on long o1-style traces.

## Integration Candidates

- Update [[supervision-unit-of-compression]] as the implicit-answer pole.
- Update [[cot-compression-methods]]: internalization, not skip-and-continue.

## Tensions or Contradictions

- C3oT cites the 2023 implicit-CoT distillation as severely accuracy-losing; this 2024 method is the authors' simpler replacement and still trails explicit CoT on GSM8K.
