---
title: "Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs"
type: source
source_kind: paper
author_or_origin: "Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu"
published: 2024-12-30
captured: 2026-08-13
url_or_location: "https://arxiv.org/abs/2412.21187"
reliability: primary
status: processed
tags: [overthinking, o1-like, efficiency-metrics, self-training]
---

# Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs

## Scope and Relevance

First dedicated study of *overthinking* in o1-like models: extra solutions after the first correct answer, especially on easy math. Defines outcome and process efficiency metrics. Mitigates by self-training on shortened self-samples (FCS+Reflection + SimPO). Does not train skip transitions; it shortens *how many solution rounds* are emitted.

## Faithful Summary

On "what is the answer of 2 plus 3?", o1-like models used ~1,953% more tokens than conventional models. QwQ-32B-Preview produced 13 solutions for that item. Across ASDIV / GSM8K / MATH500, o1-like models emit 2–4 solution rounds for most items and *more rounds on easier sets*. In >92% of correct cases the first round already has the right answer. Later rounds add little new strategy (GPT-4o clustering).

Outcome efficiency $\xi_O$: fraction of tokens up to first correct answer (only on items that are eventually correct). Process efficiency $\xi_P$: fraction of tokens in *distinct* reasoning strategies. Both are much lower for o1-like than for single-solution conventional models.

Mitigation: sample 10 traces on PRM12K from QwQ; keep shortest correct; further streamline to first-correct + one reflection; SimPO against the longest sample. On MATH500, ~48.6% token cut while holding accuracy. Holds on GSM8K / GPQA / AIME.

## Extracted Knowledge

- **Definition/Object:** Solution — a segment of a generation that contains an explicit answer. Overthinking — tokens after the first correct / non-distinct strategy.
- **Claim:** o1-like models spend more solution *rounds* on easier problems even while total tokens grow with difficulty.
  - Support: Figure 4 MATH500 levels; ASDIV vs MATH500 averages.
  - Status: observed on QwQ-32B-Preview and DeepSeek-R1.
- **Claim:** First solution is already correct in >92% of eventually-correct cases, so later tokens barely move accuracy.
  - Support: first-correctness distribution (Figure 5).
  - Status: observed.
- **Claim:** Self-training on first-correct+reflection traces reduces tokens ~48.6% on MATH500 without accuracy loss.
  - Support: Figure 1b; Table 4.
  - Scope/conditions: QwQ-32B-Preview post-training; PRM12K self-samples.
  - Status: observed.

## Limitations and Failure Modes

- "Solution" segmentation uses Llama-3.3-70B; strategy clustering uses GPT-4o — both are judges, not ground truth.
- Mitigation is preference/SFT on shorter *full responses*, not local skip pairs.
- Focused on math; "2+3" is a rhetorical extreme.
- First-correct-only (no reflection) hurt MATH500 — some hard items need a second look.

## Integration Candidates

- Update [[overthinking-patterns]] as the source of the 1953% / first-correct / extra-round facts.
- Update [[cot-compression-methods]] as self-training on truncated multi-solution traces.

## Tensions or Contradictions

- TRACE later redefines overthinking via thought-graph utility rather than tokens-to-first-correct; Chen's metric is the length-based special case TRACE criticizes as coarse.
