# Sparse Autoencoder Emotion Analysis in Gemma 3 1B: Recovering Russell's Circumplex from Disjoint Features vs. Causal Valence-Arousal Steering

- **Authors:** Sathiyanarayanan Palani
- **Date:** 2026-07-20
- **Source:** https://forum.effectivealtruism.org/posts/PE8pipA2FwBe47yeE/sparse-autoencoder-emotion-analysis-in-gemma-3-1b-recovering
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `palani_sparse-autoencoder-emotion-analysis_2026-07-20`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

## Tl;dr

I trained a top-k Sparse Autoencoder (k=32, 2× overcomplete, d\_latent=2304) on Layer-13 activations of Gemma 3 1B IT across 12 emotions (24 generated stories, 14,252 tokens) to test whether emotional representations decompose into clean, independent, causally-manipulable features.

Key finding: emotion-exclusive SAE features are perfectly disjoint — the unique-feature cosine similarity matrix across all 12 emotions is an exact identity (every off-diagonal entry is 0.0000) — while unsupervised PCA on the same activations recovers Russell's valence/arousal axes almost exactly (PC1 = 36.0% variance → valence, PC2 = 19.3% → arousal; happy–sad separation −103.6 vs. excited–calm separation +104.1).

Injecting the extracted valence and arousal directions into the residual stream at Layer 13 causally and independently steers generated text along each axis (valence-axis cosine moves from +0.11 at α=+1.0 to −0.20 at α=−1.5) without cross-axis contamination.

This suggests emotional structure in a 1B-parameter LLM is both linearly decomposable into psychologically-grounded geometry and mechanistically steerable via simple activation edits — corroborating, via an independent SAE-based method on a small open-weight model, Anthropic's recent finding (Sofroniew et al., 2026) that emotion representations causally shape LLM behavior.

[Article](https://palani-sn.github.io/LLM1/README.html) | [GitHub](https://github.com/Palani-SN/LLM-Emotions)

[Image: Russell's circumplex model of emotion, showing valence and arousal axes with emotion labels placed around the circle]

[Image: Emotion vectors in PC1-PC2 space, variance-normalised axes, showing all 12 emotions arranged around valence and arousal directions]

---

## Motivation

Language models produce text that reads as emotionally coherent, but it's unclear whether that coherence reflects genuinely structured internal representation or an emergent surface pattern with no clean underlying geometry. If hidden states for "excited" and "calm" simply point in different directions without decomposable structure, techniques like activation steering or runtime monitoring are fragile — they may work for the specific emotions probed and fail to generalize.

The most directly relevant existing work is Anthropic's Transformer Circuits study of emotion concepts in Claude Sonnet 4.5 ([Sofroniew et al., 2026](https://transformer-circuits.pub/2026/emotions/index.html)), which extracts emotion directions via contrastive activation differences, validates them with logit-lens projections onto the unembedding matrix, and shows they causally influence alignment-relevant behaviors like reward hacking, blackmail, and sycophancy. That work establishes causal steerability convincingly on a frontier proprietary model using one extraction method — it doesn't test whether an unsupervised sparse decomposition of the same kind of activations would independently recover disjoint, per-emotion features, or whether the resulting geometry matches a psychological model without any labels supplied to the algorithm.

I wanted to test that complementary set of properties on a small open-weight model, using a mechanistically different extraction method: do SAE features cleanly separate 12 distinct emotions into disjoint sets, does PCA over those emotion representations reproduce Russell's circumplex model without any circumplex labels supplied to the algorithm, and can the resulting axes be used to causally and independently steer generation?

---

## What I Did

1. **Activation collection:** Generated two emotionally-conditioned stories for each of 12 emotions (24 samples, 14,252 tokens total) with Gemma 3 1B IT, using a prompt that explicitly forbids naming the target emotion or its synonyms, then recorded Layer-13 (of 26) hidden states for response tokens only, plus a neutral, affect-free baseline sentence.
2. **SAE training:** Trained a top-k Sparse Autoencoder (d\_model=1152 → d\_latent=2304, 2× overcomplete, k=32 active features per token) with MSE reconstruction loss for 400 epochs (Adam, lr=3×10⁻⁴, batch=512), re-normalizing decoder columns to unit L2 norm after every step. Training data was mean-centered, and a neutral-direction vector was computed but its explicit projection step was disabled: since emotion vectors are already mean-centered against the dataset-wide mean, applying that projection on top double-subtracted the same shared component, pushing activations net-negative and degrading generation into off-language, garbled output in early runs.
3. **Feature analysis:** Passed all activations through the trained encoder and computed nine analyses — raw-space vs. latent-space vs. exclusive-feature cosine similarity, L0 sparsity verification, pairwise feature-disjointness counts, top-10 dominant feature fingerprints per emotion, and PCA/t-SNE projections.
4. **Steering-vector extraction:** Ran PCA (full SVD) on the 12 emotion-mean activation vectors to obtain two orthogonal directions (PC1, PC2; dot product 2.83×10⁻⁷), automatically labeled valence/arousal based on which component best separates happy–sad and excited–calm.
5. **Causal steering validation:** Registered forward hooks at Layer 13 to add (steer) or project out (ablate) the valence/arousal directions during generation from a fixed neutral prompt, at α ∈ {0.5, 1.0, 1.5} in both directions, then measured cosine similarity of resulting hidden states to each axis to confirm dose-dependent, axis-independent control.

Full methodology, implementation details, and code: [GitHub](https://github.com/Palani-SN/LLM-Emotions)

---

## Key Findings

**Finding 1: Emotion-exclusive SAE features are perfectly disjoint**
→ Unique-feature cosine similarity across all 12 emotions forms an exact identity matrix — every off-diagonal entry is 0.0000
→ Surprising because full latent-space similarity (before isolating exclusive features) is uniformly high (0.9496–0.9896), completely masking this disjointness
→ [Full write-up — Stage 3: Analyse Features](https://palani-sn.github.io/LLM1/README.html#stage-3--analyse-features)

**Finding 2: PCA recovers Russell's circumplex without being told the axes exist**
→ PC1 (36.0% variance) and PC2 (19.3% variance) automatically align with valence and arousal; happy–sad separation on PC1 is −103.6, excited–calm separation on PC2 is +104.1 — nearly equal magnitude
→ All 12 emotions land in their expected circumplex quadrant from unsupervised PCA on raw activations alone
→ [PCA + t-SNE Projections](https://palani-sn.github.io/LLM1/images/latent_projections.png)

**Finding 3: Valence and arousal are causally and independently steerable**
→ Injecting the valence direction shifts valence-axis cosine similarity from +0.11 (α=+1.0) to −0.20 (α=−1.5), while arousal-axis cosine stays within a narrow band (−0.12 to −0.19) throughout
→ The two axes are not just geometrically orthogonal (dot product ≈0) but mechanistically independent under intervention
→ [Full write-up — Stage 5 steering demo](https://palani-sn.github.io/LLM1/README.html#stage-5--ablation-and-steering-demo)

**Finding 4: Ablation exposes a hidden negative-valence bias in "neutral" prompts**
→ Removing the valence component from a nominally neutral prompt's hidden states raises valence-axis cosine to +0.73 — higher than any explicit positive-steering condition tested
→ Indicates the model's default "neutral, matter-of-fact" narration carries a latent negative-valence skew only visible under ablation
→ [Full write-up — Demo 1 excerpts](https://palani-sn.github.io/LLM1/README.html#demo-1--valence-cycle-happy--neutral--sad)

**Finding 5: Arousal steering is monotonic with no saturation at this checkpoint**
→ Arousal-axis cosine moves from −0.19 to −0.27 to −0.31 as negative arousal steering strength increases from α=0.5 to 1.5
→ Rules out a ceiling/floor effect in the extracted arousal direction across the tested range
→ [Full write-up — Demo 2 arousal cycle](https://palani-sn.github.io/LLM1/README.html#demo-2--arousal-cycle-tenseexcited--neutral--calm)

---

## Safety / Field Implications

These results suggest that affective content in even a 1B-parameter model is not diffusely encoded but decomposes into a small number of clean, disjoint, causally-manipulable directions. If a construct as informal as "emotion" resolves this cleanly under an SAE plus unsupervised PCA, it's plausible that other internal states relevant to safety — deception cues, refusal, distress-like signals — could be recoverable with the same unsupervised pipeline, without needing labeled probes for every construct in advance.

This connects most directly to interpretability-as-a-safety-tool and defense-in-depth: runtime activation steering and monitoring only become practical safety levers if the underlying directions are stable, disjoint, and independently controllable rather than entangled. It also cross-validates, via a mechanistically distinct method (SAE decomposition vs. contrastive activation differences) on a much smaller open-weight model, Anthropic's finding that emotion representations causally shape alignment-relevant behaviors including reward hacking, blackmail, and sycophancy (Sofroniew et al., 2026) — suggesting that link is not an artifact of one extraction technique or one frontier model. The finding that ablation exposes a hidden bias in a "neutral" prompt is a small but concrete illustration of how activation-level tools can surface behavior that isn't visible from outputs alone.

More speculatively, this kind of pipeline offers an affordable way to interrogate claims about model affect or "welfare"-adjacent behavior with something more mechanistic than self-report — by asking whether the claimed internal state has the disjoint-feature-plus-orthogonal-axis structure this analysis looks for, rather than taking generated text at face value.

---

## Limitations

* Single model tested (Gemma 3 1B IT) — generalization to larger or differently-trained models is untested.
* Small sample: 24 stories (2 per emotion), 14,252 tokens total — emotion mean vectors may not capture full within-emotion variance.
* Single layer probed (Layer 13 of 26) — no layer sweep to confirm this is the optimal or only site for this structure.
* No human or automated judge scored generated stories for perceived emotion — excerpt interpretation is the author's own read.
* Steering demos use one fixed prompt per axis — collateral effects on coherence or quality across a broader prompt distribution weren't measured.
* Unlike Sofroniew et al. (2026), no downstream alignment-relevant behaviors (reward hacking, blackmail, sycophancy) were tested — only representation structure and generation tone were measured.

---

## Open Questions

1. Does the exact-identity-matrix disjointness of exclusive SAE features hold at larger model scales (7B, 70B), or does it degrade as a model's affective vocabulary grows?
2. Does the same SAE + PCA pipeline recover circumplex structure in a different model family (e.g., Llama, Qwen) with no shared pretraining lineage with Gemma?
3. Can the extracted valence/arousal directions support lightweight runtime affect monitoring, and does the "hidden negative bias in neutral prompts" finding hold across other neutral or system prompts?
4. Does the disjoint, causally-steerable emotion-feature structure found here predict the alignment-relevant behavioral shifts (reward hacking, sycophancy, blackmail-like responses) that Anthropic's contrastive-vector approach found in Claude Sonnet 4.5 (Sofroniew et al., 2026) — or is that link specific to their extraction method and model scale?

---

*Note: I used AI assistance to help structure this write-up. All research, implementation, experiments, and findings are my own independent work.*

---

## Resources

* 📄 **Full Write-up:** [LLM Emotions — SAE Analysis of Gemma 3 1B IT](https://palani-sn.github.io/LLM1/README.html)
* 💻 **GitHub Repository:** [LLM-Emotions](https://github.com/Palani-SN/LLM-Emotions)
* 🌐 **Portfolio:** [palani-sn.github.io](https://palani-sn.github.io/)

---

**Tags:** `AI Safety`, `Research`, `AI interpretability`, `Large Language Models`, `AI alignment`
