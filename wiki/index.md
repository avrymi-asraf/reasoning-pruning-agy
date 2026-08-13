# Wiki Index

## Scope

This wiki accumulates published and local evidence about **training language models to skip or compress chain-of-thought while keeping the answer correct**. It exists so future work on this repository can retrieve related methods, overthinking taxonomies, and the exact contrast with this project's first-skippable-span $(x \to y)$ pipeline without re-reading papers.

In scope: CoT compression and step-skipping training methods; auditor/teacher pruning; implicit CoT; overthinking pattern taxonomies; reported accuracy–token tradeoffs.

Out of scope: general LLM surveys, prompt compression of *inputs* (except when a paper uses those tools as CoT token scorers), model quantization/pruning of weights, and product deployment notes.

## Topic Map

### This project's method
- [[first-skippable-span-auditor]] — How a separate decision model $D$ marks the first safely skippable span and this repo turns that span into a local $(x \to y)$ SFT pair.

### How other work compresses reasoning
- [[cot-compression-methods]] — Side-by-side of published recipes: mixed skip-step SFT, token pruning, `[SKIP]` tokens, teacher rewrite, implicit internalization, early exit.
- [[supervision-unit-of-compression]] — What the training target actually is (full shorter trace, tokens inside steps, keep-indices, implicit answer, local next-step).

### What gets wasted in long traces
- [[overthinking-patterns]] — Published thought-transition labels vs this repo's six exploration archetypes; what is established vs merely similar.

## Open Tensions and Contested Claims

- **Does shortening steps hurt reasoning?** Jin et al. (cited by [[kang-2024-c3ot]]) claim that shortening steps while keeping key information *diminishes* ability. C3oT, TokenSkip, DRP, and Step Entropy all report large token cuts with little accuracy loss. Status: disputed; likely depends on *how* shortening is done and whether the model is trained to expect the short form.
- **Token skip vs step skip.** [[xia-2025-tokenskip]] states it does *not* reduce step count, only trims tokens inside steps. [[liu-2024-skip-steps]], [[li-2025-step-entropy]], [[jiang-2025-drp]], and this repo all operate at step/span granularity. Status: established distinction.
- **Separate auditor vs same-model signal.** This repo and DRP use an external teacher/auditor. Step Entropy and Verbosity-Aware use the generator's own entropy/verbosity. No published source inspected here combines a first-safely-skippable-span auditor with prefix→next-step pairs. Status: supported absence in the ingested set; a later paper could exist.
- **Helpful vs wasteful extra tokens.** [[srivastava-2025-llmthinkbench]] finds ~11% of long traces contain *helpful* decomposition; most waste is verification, contradiction, or exploration. [[zhang-2025-trace]] treats over-verification and over-exploration as the two drivers. This repo's archetypes include preamble/restatement/detours that those papers do not name. Status: overlapping but not identical taxonomies.

## Knowledge Gaps and Next Investigations

- Full-text ingest of Extra-CoT ([[tang-2026-extra-cot]]), TokenSqueeze ([[zhang-2025-tokensqueeze]]), Dynamic Early Exit ([[yang-2025-dynamic-early-exit]]), and Verbosity-Aware ([[jang-2025-verbosity-aware]]) — currently abstract-level only.
- SPIRIT / stepwise perplexity pruning (Cui et al., arXiv:2502.13260), CoT-Valve, ThinkPrune, and Sui et al. "Stop Overthinking" survey were named in related-work sections but not opened.
- Luo et al. 2025b ("Deconstructing Long Chain-of-Thought"), cited by Dynamic Early Exit for Problem Restatement / Result Verification chunks, was not opened.
- No published paper in this wiki uses the six exact archetype names from this repo's exploration notebook.
- Empirical token-savings numbers in this repo's own Gemma exploration are not yet ingested as an experiment source.

## Recent Material Updates

- 2026-08-13 — Initialized the wiki and ingested the CoT-compression / overthinking cluster from the deep-research pass. Created [[first-skippable-span-auditor]], [[cot-compression-methods]], [[supervision-unit-of-compression]], and [[overthinking-patterns]]. Recorded that no inspected published paper trains on isolated first-skip $(x \to y)$ pairs.
