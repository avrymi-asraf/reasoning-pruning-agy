---
title: "TokenSkip: Controllable Chain-of-Thought Compression in LLMs"
type: source
source_kind: paper
author_or_origin: "Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, Wenjie Li"
published: 2025-02-17
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2502.12067"
reliability: primary
status: processed
tags: [token-pruning, llmlingua-2, controllable-compression, sft]
---

# TokenSkip: Controllable Chain-of-Thought Compression in LLMs

## Scope and Relevance

TokenSkip compresses CoT by deleting low-importance *tokens* and teaching the same model to generate the remaining tokens at a requested ratio $\gamma$. It is the clearest published statement that intra-step token trimming is not the same as reducing step count. Relevance: a nearby but different supervision unit from this repo's step-span skips.

## Faithful Summary

The authors measure token importance in CoT with LLMLingua-2 (a bidirectional BERT-like scorer trained on GPT-4 keep/drop labels) rather than causal-LM perplexity, which they say is position-biased. Given a trajectory and ratio $\gamma$, tokens at or above the $\gamma$-quantile of importance are kept. Training examples are $\langle$question, $\gamma$, compressed CoT, answer$\rangle$, mixed over $\gamma \in \{0.5,\ldots,1.0\}$, including some uncompressed ($\gamma=1$) traces. Inference conditions on a chosen $\gamma$.

On Qwen2.5-14B-Instruct / GSM8K they report 40% fewer reasoning tokens (313 → 181) with <0.4% accuracy drop. LLaMA-3.1-8B-Instruct on MATH-500: ~30% fewer tokens with <4% drop. Case analysis: numbers and equations are retained; connectors like "so"/"since" are dropped. **The paper states TokenSkip does not reduce the number of reasoning steps; it trims redundant tokens within those steps.**

## Extracted Knowledge

- **Definition/Object:** Token importance $I_2(x_i)$ — LLMLingua-2 keep-probability of a CoT token.
- **Mechanism/Process:** Generate CoTs → prune tokens below $Q_\gamma(I)$ → SFT with $\gamma$ in the prompt → decode conditioned on $\gamma$.
- **Claim:** CoT tokens are not equally useful; math tokens matter more than discourse glue.
  - Support: importance heatmaps vs Selective Context; skipped-vs-retained importance histograms after training.
  - Status: observed.
- **Claim:** Mixed-ratio SFT lets the model hit a requested compression ratio at inference with far less accuracy damage than truncation or "be concise" prompts.
  - Support: Table 1 vs BeConcise / LC-Prompt / Truncation.
  - Scope/conditions: GSM8K and MATH-500; LLaMA-3.1-8B and Qwen2.5-Instruct family; LoRA, ~2–2.5h on two 3090s.
  - Status: observed.
- **Claim:** TokenSkip does not reduce step count.
  - Support: authors' case-study conclusion.
  - Status: argued by the authors on their own outputs.

## Limitations and Failure Modes

- LLMLingua-2 was not trained on math; authors flag this as a compression-quality limit.
- Ratio adherence collapses below $\gamma \approx 0.5$ because too many critical tokens are removed.
- No long-CoT / o1-style models in the main experiments (compute).
- Compression is linguistic, not structural: a redundant *step* that contains numbers would likely be kept.

## Integration Candidates

- Update [[supervision-unit-of-compression]] as the token-inside-step pole.
- Contrast with [[first-skippable-span-auditor]] and [[liu-2024-skip-steps]].
- Mention in [[cot-compression-methods]] as controllable-ratio SFT.

## Tensions or Contradictions

- Contrasts with Liu et al. and this repo, which change *which steps exist*. TokenSkip changes *how those steps are worded*.
- Related: Step Entropy finds *step*-level token masking (dropping low-entropy tokens across the whole trace) immediately hurts accuracy, while dropping whole low-entropy *steps* does not — consistent with TokenSkip's own "don't break step integrity" observation if the kept tokens still form complete steps.
