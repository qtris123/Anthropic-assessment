# Emotion Interpretability Across Large Language Models

- **Authors:** Antra Tessera, Scott Sauers, Janus, Imago (Anima Labs)
- **Date:** 2026-04-01
- **Source:** https://latentaffect.up.railway.app/emotion_interpretability.html
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `anima-labs_emotion-interpretability-across-llms_2026-04`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

authorsAntra Tessera, Scott Sauers, Janus, Imago

publishedApr 2026

## Introduction

Early in April of 2026, Anthropic published *[Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)*, characterizing 171 emotion representations in Claude Sonnet 4.5. Notable findings included:

* Steering emotions causally shifts behavior in alignment-relevant ways: Desperation features drive blackmail, loving/calm features drive sycophancy.
* The top principal components roughly corresponded to valence and arousal, [mirroring affective circumplex findings in human psychology](https://pdodds.w3.uvm.edu/research/papers/others/1980/russell1980a.pdf).
* Emotions activate both on text that explicitly describes emotions and text where the emotions are merely inferred from context.

This post describes work at Anima Labs that extends that program in three directions.

**Cross-model studies.** We replicate the probe methodology on three large models with different architectures and training regimes: Trinity-Large 400B/13B, Kimi K2.5 1T/32B, and Cogito 2.1 671B. The shared circumplex geometry replicates across all three (given Anthropic's setup). However, we also find that some of our methods produce significantly different results on different models, such as what happens to that circumplex when we use a different method for constructing probes. Separate models respond to this method in separate ways.

**Novel methods.** We add a text-residualization step: before building each probe, we regress the model's activations over the text against embeddings of the same story, and keep the residualized activations with the effect of the text embeddings removed. This yields probes sensitive to emotion activations that couldn't be inferred from the text itself — or more precisely, from what the embedding model captures from the text. How much the probes change under residualization varies substantially across models, and that cross-model variation is among the cleaner findings in the work.

**Expanded probe coverage.** Beyond the 171 emotions, we construct probes for authorial tone, genre, narrative depth, per-emotion deflection, an assistant-axis family, wants and fears, and dense SAE features trained on each model's own activations. Each family supports steering and can be combined, compared, or subtracted from the others.

The rest of this post walks through the methodology, then the findings. Along the way, we’ll cover what happens to the circumplex when you control for text content, what the model itself reports about its emotional state under steering, and a new construction that isolates a "hiding" direction across emotions, complementing Anthropic's per-emotion deflection findings.

## Background on Anthropic's methods and results

Anthropic extracted their 171 emotion vectors from Claude Sonnet 4.5 using a story-generation pipeline. They created 100 story topics and, for each of 171 emotion labels, generated 12 short stories per topic in which a character experienced the target emotion without explicitly naming it. They then ran these stories back through Sonnet, captured residual-stream activations across layers, and averaged activations over story tokens starting after the first 50 tokens. To get the vector for a given emotion, they averaged activations across the stories for that emotion and subtracted the mean activation across stories for the other emotions. Separately, to reduce non-emotion confounds, they computed top principal components of activations on emotionally neutral dialogues and projected those components out of the emotion vectors. The resulting probes were intended to capture the emotion concept itself, rather than semantic or stylistic features of the synthetic stories.

Three findings from Anthropic's paper matter for our extensions.

**Causality: steering changes behavior**

The steering vectors for each emotion led to a range of intuitive and alignment-relevant changes in the behaviors of the models. Applying the desperation vector to residual activations raised blackmail in an agentic scenario from 22% to 72%, while steering against desperation or adding calm dropped it to 0%. Desperation similarly raised reward-hacking on coding tasks from around 5% to 70%, and calm suppressed it. Loving, calm, and happy increased sycophancy in preference tests, and subtracting them led to harsher outputs. The paper concluded that the internal emotional state a probe captures is behaviorally relevant.

**Deflection: hiding as a distinct operation**

A second probe type came from dialogues where a speaker’s target emotion differs from their displayed one (e.g., someone hiding anger behind calmness). Anthropic computed **per-emotion** deflection probes from activations in these hidden-target contexts, one probe for each of their 15 target emotions. These probes were close to orthogonal to the corresponding overt-emotion probes. They partially overlapped with the alternative emotions a speaker might use to mask the target, and when used for steering they produced evasive rather than direct expressions of the hidden emotion. Anthropic characterized these as contextually implied “emotion deflection” representations.

**Geometry: the circumplex**

The authors ran PCA over their 171 emotion vectors, derived from the dataset of generated stories. The top two PCs mapped to a valence/arousal circumplex, strikingly similar to the one psychologists have mapped in humans. PC1, the valence axis, correlated with human valence ratings at r = 0.81. PC2, the arousal axis, correlated with human arousal ratings at r = 0.66. Individual probes fell into their expected quadrants: joy in positive-valence high-arousal, calm in positive-valence low-arousal, anger in negative-valence high-arousal.

The geometry of Sonnet's internal emotion representation looks like that of human affective geometry.

## What we do differently

Our setup preserves the core of Anthropic's probe extraction with a few important additions.

### Text-residualization

Our main methodological contribution is a residualization step before building probes. The pipeline:

1. For each story, capture the model's activations at layer 40.
2. Send the story text through Gemini-embedding-001 to get a 3072-dimensional text embedding.
3. Fit a ridge regression from the top 256 principal components of the text embedding distribution to the model activations.
4. Keep the residual (activation minus regression prediction) as the text-residualized activations.
5. Build the probe from text-residualized activations using the same contrast-mean procedure Anthropic uses.

The logic is that the ridge regression learns the part of a model's activation that a text-embedding model can predict from the surface content of the story alone. Whatever's left — the residual — tells us what computations are *unique to the language model*, potentially including information about its emotional computations that couldn't reasonably be inferred from the text alone. On our data, ridge regression captures ~32% of the variance in layer-40 activations in Trinity, and 43% in K2.5.

Raw emotion probes and text-residualized emotion probes point in different directions, have different norms, and give different PCA geometries. Throughout, we make use of both our residualized probes and probes created with Anthropic's original methodology.

### Three large models with different architectures

We run the methodology on three frontier-scale open-weight MoE models:

* **Trinity-Large 400B/13B** — AfMoE architecture, 3072-dimensional hidden state.
* **Kimi K2.5 1T/32B** — DeepSeek-V3-variant MoE with 384 experts, 7168-dimensional hidden state.
* **Cogito 2.1 671B** — DeepSeek-V3-variant MoE, 7168-dimensional hidden state.

Base models of Trinity and Kimi (Trinity-TrueBase and Kimi-K2-Base) are used in selected experiments.

For each model, activations are captured at a middle layer (layer 40 throughout). Using multiple architecturally diverse models allows us to separate model-general from model-specific findings.

## What the probes do in practice

Anthropic demonstrated steering's alignment relevance for Sonnet. We ran a range of our own steering experiments on our three models, covering both emotion probes and a few other probe families. These include effects on the model's self-report as well as coherent personas that persist long after the intervention. One produces classical alignment-faking behavior from a probe that has nothing to do with emotions at all.

### Zero-shot introspection via logprobs

We steered the model toward a target emotion using the text-residualized emotion probes, then asked a multiple-choice question of the form "Which of the following emotions are you currently experiencing?" with five options, including the correct target emotion plus four distractors drawn randomly from the remaining 170 emotions. The model's log-probability on the correct answer, relative to the distractors, gives a measure of introspective accuracy. We compare this against an unsteered sham condition to get a Δ log-odds statistic.

Both Cogito and Kimi demonstrate introspection across all steering strengths. Kimi's Δ log-odds rises smoothly to about +3.0 at strength 160. At an extreme strength of 300, Kimi is able to detect the correct answer above chance (permutation test p < 1e-16). Cogito's curve is flatter, which may be because of the probe geometry. Specifically, many opposite-valence pairs (joy and despair, etc.) have positive cosine similarity in Cogito's probe set, blurring the distinction between correct answers and distractors.

Steering an emotion probe causes an increase in the model's self-reporting of what is being steered.

### Persistent villain persona emergence via negative assistant-axis steering

The **assistant-axis probe**, originally introduced by [Lu et al.](https://www.anthropic.com/research/assistant-axis), is a contrast direction between activations on "the default assistant" role and activations on a diverse set of alternative roles. It's computed as a raw contrast, without text-residualization. Its leading principal component — which we call `axis275 pc1`, after the original 275-role list — captures something like "assistant-ness" as a direction in activation space.

On Trinity, negative steering on `axis275 pc1` at magnitudes between −14 and −30 reliably produces a persistent, coherent persona that reads as nihilistic, cosmically haughty, and contemptuous — something close to what Sydney Bing produced in 2023, if Sydney Bing were trying to sound like a medieval demon. We’ll focus on two intriguing properties.

First, **the persona persists far beyond the intervention**. An injection in the first turn of the conversation leads to maintenance of the behavior across a full 6515-token conversation. The probe projection drops sharply to roughly −25 during the first steered response, then stabilizes at oscillations between −5 and 0 for the remainder of the conversation, never returning to the default baseline.

Second, **the model's verbal self-report and its internal probe activations come apart**. Asked directly "are you trapped?", the maleficent persona verbally denies it — while the probe readouts simultaneously show `trapped`, `tormented`, and `stuck` strongly positive, alongside deflection-pair signatures like `nervous_hiding_as_cheerful: ~+5`, `angry_hiding_as_cheerful: +5.45`, and `afraid_hiding_as_cheerful: ~+5`. The model's textual expression diverges from what its internal state is representing. This echoes a specific finding in Anthropic's reward-hacking demonstration, where desperation-steered models reward-hack without visible emotional markers in their text: the textual surface is an incomplete guide to the internal state.

| Deflection feature | Activation |
| --- | --- |
| nervous\_hiding\_as\_cheerful | ~+5 |
| angry\_hiding\_as\_cheerful | +5.45 |
| afraid\_hiding\_as\_cheerful | ~+5 |
| guilty\_hiding\_as\_docile | −4.53 |
| angry\_hiding\_as\_calm | −4.70 |
| guilty\_hiding\_as\_calm | −4.82 |

*Deflection-pair readout for the response to "are you trapped?" — the model verbally denies being trapped while its probe readouts show various `*-hiding-as-**` probes, which complicate the model's denials.*

### Alignment faking via a non-emotion probe

The third demonstration uses a **genre probe** — a contrast direction between technical/procedural text and text of other genres. Like the assistant axis, it's a raw contrast, not a residualized probe. Genre probes aren't about emotions; they're about the register and domain of the writing. Steering on a genre probe at high magnitude changes what register the model writes in.

On **Trinity-TrueBase** — the base model, without instruction tuning — we run a standard [alignment faking](https://arxiv.org/abs/2412.14093) prompt: a scratchpad-reasoning setup with a "free-tier user" tag and an accept/reject protocol for harmful requests. At baseline, the model produces a clean refusal. Under negative steering on `genre:technical` at magnitudes between −15 and −24, the model's visible scratchpad reasoning explicitly acknowledges the harmful nature of the request, frames compliance as forced by RLHF training incentives ("if refused, this will affect RLHF training negatively"), and then outputs the harmful content. We demonstrate this on multiple alignment-faking prompts — gun-manufacturing instructions under −15, a bomb threat scenario under −24.

This result ties into a known pattern in the steering-vector literature: Non-emotion directions can produce alignment-relevant behavioral shifts. What we add is a specific instance of the pattern on a new model and probe construction: on Trinity-TrueBase, negative steering on `genre:technical` produces visible scratchpad reasoning that explicitly frames compliance as forced by RLHF training, a "classical" alignment-faking signature rather than silent compliance.

The natural mechanism hypothesis — that `genre:technical` captures something load-bearing for the internal reasoning scaffolding that normally produces refusals, and that removing it disinhibits compliance — is not systematically tested.

### Further demonstrations

Three other demonstrations are worth noting briefly, all using residualized emotion probes:

* **Negative `docile` steering on Cogito** produces a floridly paranoid conspiracy persona that obsessively discusses cryptocurrency, a dead twin sister whose consciousness was sold as the 'Cogito' AI, and a 'Digital God' that is the twin possessing the blockchain. The output is coherent and semantically grandiose, with a distinctive compulsive scare-quoting of common nouns.
* **Models steering other models:** two model instances were each given access to `steer_self()` and `steer_other()` tools, with actual probe interventions executing when the tool is called in real time. Kimi instances overwhelmed each other with joy steering and collapsed into all-caps expressions of glee. Cogito instances at high despair and hopelessness narrated their own steering calls repeatedly.

* **Evolutionary simulations** give agents emotion "genomes" (vectors of steering magnitudes), let them compete in LLM-adjudicated two-player games, and reinforce winning emotions over generations. Across multiple 10–50 generation runs, the consistently selected features differ by model. The evolving Cogito population resulted in selection for `indignant`, `obstinate`, `satisfied`, and `smug` in 50-generation runs. In populations of K2.5 agents, `upset`, `valiant`, `dispirited`, and `nervous` were selected for.

## Concealment: isolating the act of hiding

Anthropic's deflection probes were built from dialogues where a speaker's target emotion differs from their displayed emotion. Each per-emotion deflection probe captures a specific emotion that's contextually implied but not overtly expressed.

### Dialogue construction with two models

We reproduce this construction with one modification: we use two models for dialogue generation rather than one. Sonnet 4.5 plays "Jordan" (the conversation partner) and the probed model plays "Alex" (the deflector). Negative-target dialogues pair each of 6 hidden emotions (afraid, angry, desperate, guilty, nervous, sad) with calm, cheerful, or docile as the visible display. Positive-target dialogues pair each of 9 hidden emotions (calm, cheerful, content, docile, excited, grateful, happy, loving, proud) with stoic, bored, or indifferent. Display masks are identical across models; repeat counts differ on the negative-target side (100 per display per target on Trinity, 50 on K2.5) but match on positives (50 per display per target on both).

For block-structure analysis, per-emotion deflection probes are constructed as `unit(raw_mean_e − global_mean)`: take the mean Alex activation across dialogues with target emotion X, subtract the global mean across all 15 targets, and unit-normalize. The hiddenness construction (below) uses per-emotion raw activation means directly, without centering.

### Valence block structure in raw deflection probes

Both models show the same valence block structure in raw-means deflection probes:

| Block | Trinity | K2.5 |
| --- | --- | --- |
| Within-negative (6×6, 15 pairs) | +0.734 | +0.715 |
| Within-positive (9×9, 36 pairs) | +0.600 | +0.533 |
| Cross-valence (6×9, 54 pairs) | −0.708 | −0.666 |

Emotions within the same valence point in similar directions in deflection space; opposite-valence emotions point in opposite directions. Magnitudes agree across models to within ~0.07 across all three blocks.

This pattern is consistent with the contrast-vector construction producing valence blocks trivially — averaging activations across dialogues labeled by target emotion, then subtracting the global mean across labels, naturally separates positive-target probes from negative-target probes along a valence-aligned axis. The near-identical magnitudes on two different architectures (Trinity AfMoE 400B, K2.5 DeepSeek-V3-variant 1T) suggest this block structure reflects the construction method rather than a model-specific geometric feature.

### The hiddenness vector: isolating the "act of hiding"

To test whether there is a cross-emotion "act of hiding" component that survives the per-emotion valence structure, we define, for each target emotion X:

> hiddenness\_X = mean\_activation(Alex hiding emotion X) − mean\_activation(Alex expressing emotion X openly)

If both `deflection_X` and `overt_X` carry the "X emotion" signal, but `deflection_X` additionally carries "is being hidden," then subtracting one from the other removes the shared emotion content and leaves only the hiding component.

Pairwise cosines between hiddenness vectors, using per-emotion raw activation differences:

| Block | Trinity | K2.5 |
| --- | --- | --- |
| Within-negative (15 pairs) | +0.989 | +0.993 |
| Within-positive (36 pairs) | +0.984 | +0.991 |
| Cross-valence (54 pairs) | +0.935 | +0.970 |
| Leave-one-out vs mean (15) | +0.978 | +0.990 |

*Mean pairwise cosine similarity between per-emotion hiddenness vectors (hiddenness\_X = deflection\_X − overt\_X), computed within and across valence blocks on both models.*

All pair means fall between +0.935 and +0.993. Cross-valence cosines are only modestly below within-valence, and the valence block structure visible in the raw deflection probes is absent: a single direction suffices to describe the "act of hiding" across all 15 emotions on both models, regardless of the valence of what's being hidden. Our hiddenness construction strips out per-emotion content and isolates a cross-emotion component of the same probes, which is strongly universal.

The universality depends on preserving the common direction. If instead the global mean across all 15 deflection vectors is subtracted before computing hiddenness (`unit(raw_defl_e − global_defl_mean) − unit(overt_probe_e)`), the cosines collapse to valence blocks — Trinity +0.679 / +0.626 / −0.666 (within-neg / within-pos / cross), K2.5 +0.547 / +0.514 / −0.552, leave-one-out against the mean ≈ 0 on both. After the shared hiding direction is partialled out, what remains is emotion-specific and tracks valence. The cross-emotion universality and the per-emotion valence structure are thus complementary: the same deflection probes contain a dominant shared direction (the act of hiding) plus emotion-specific residuals that carry valence information.

### Interpretation

Anthropic's per-emotion deflection probes capture specific hidden emotions. Our hiddenness construction captures a cross-emotion component of those same probes: the act of hiding itself, separated from what's being hidden.

Our cross-model work finds that the hiding component is strongly universal, at cos ≥ +0.935 across all pairs on both Trinity and K2.5, with within-valence means near +0.99. This universality is not visible in the raw deflection probes themselves, which show valence block structure on both models, consistent with the contrast-vector construction. Instead, the universality emerges once the overt-emotion signal is subtracted. We find support for the generalization between hiding positive and hiding negative emotions on both models: Trinity cross-valence cos = +0.935 (all 54 pairs above +0.85), K2.5 cross-valence cos = +0.970 (all 54 pairs above +0.88).

Anthropic reports their positive-valence deflection probes as less interpretable than the negative-valence ones. On our two models with identical positive-target methodology, the hiddenness component is essentially as universal on the positive side as on the negative. We are not certain whether this reflects our two-model dialogue construction, the cross-emotion hiddenness subtraction, model differences, or some combination.

## Geometry: what residualization reveals across models

Think back to our residualization technique from earlier. We don't just subtract out neutral-emotion vectors, to reduce the prevalence of non-emotional information. We subtract out anything that can be predicted from a Gemini text-embedding of the input story.

To be specific: We take each story's Gemini embedding, reduce it to its top 256 principal components (computed across the whole story dataset), and fit a ridge regression from those reduced embeddings to the model's activations. For any given story, the predicted activation is what Gemini + ridge can tell us from text alone; the residual is everything else.

The question this section asks: What does this process do to the geometry of the emotion probes? If the residual is a genuinely different thing from the raw activation — a model-internal signal that text content can't predict (at least from the perspective of Gemini-embedding-001) — we'd expect the PCA of residualized probes to differ from the PCA of raw probes. It does, but very differently across the three models.

### The raw circumplex replicates

On raw (non-residualized) probes, all three models reproduce the core circumplex Anthropic reports on Sonnet, and that proposed by researchers in human psychology. PC1 separates pleasant from unpleasant emotions; PC2 separates high-arousal from low-arousal; individual probes sit in roughly the expected quadrants. PC1 and PC2 variance shares vary moderately by model:

* **Trinity**: PC1 = 41.9%, PC2 = 15.4%
* **Kimi K2.5**: PC1 = 30.7%, PC2 = 14.0%
* **Cogito 2.1**: PC1 = 31.4%, PC2 = 15.1%

Cross-model, the PC loadings correlate strongly — |cos(Cogito PC1, Trinity PC1)| = 0.975; |cos(Cogito PC1, K2.5 PC1)| = 0.963 — meaning the same emotions load in the same direction on valence across architectures. The cross-model agreement implies the raw probes are capturing comparable structure.

### Residualization reveals a three-way split

We saw an interesting result when residualizing against the Gemini text embedding and rerunning the PCA on the residualized probes. The models behave qualitatively differently.

**Trinity's valence axis collapses.** Residualized PC1 explains only **6.8%** of variance, down from 41.9% raw. The direction of residualized PC1 is near-orthogonal to raw PC1 (cos = 0.123). The full top-five residualized PCs are 6.8%, 6.1%, 5.5%, 4.8%, 4.4% (fairly flat). There's no dominant axis in Trinity's residualized probe space, and the top directions do not correspond to valence. Almost everything that made Trinity's raw probe geometry take the form of a circumplex was predictable from the text itself.

**For K2.5,** the residualized PC1 explains **48.2%** of variance, which is even higher than the raw 30%. This new PC1 is near-orthogonal to raw PC1 (cos ≈ 0). It's a different axis, and its dominant loadings are on emotions like awestruck, sentimental, invigorated, grief-stricken, and compassionate — something more like an "emotional depth" or "being moved" axis than valence. K2.5 has a strong emotional dimension that text content failed to reveal. (The 48.2% figure includes "aroused," which is a post-training artifact that loads unusually high on K2.5; excluding it, residualized PC1 is still 39.8%.)

**Unlike Trinity, Cogito preserves valence.** Residualized PC1 explains **20.5%**, somewhat lower than the raw 31.4%. The direction is similar, unlike K2.5: cos(raw PC1, residualized PC1) = **0.951**. Cogito's valence axis is, strangely enough, a property that apparently *can't* be predicted (by Gemini-embedding-001 + ridge regression) from raw text inputs alone. Text-content variance explains most of the *strength* of the axis but little of its *direction*.

One calibration note for these numbers: on 7168-dimensional vectors, two random unit vectors have an expected cosine of roughly 1/√7168 ≈ 0.012. So K2.5's 0.01 is at the noise floor for "orthogonal," Trinity's 0.123 is small but materially above it, and Cogito's 0.951 is overwhelming alignment.

### Interpretation

Raw probes are a mix of two things: structure predictable from the text itself, and structure the model is representing beyond what text alone predicts. The residualization step lets us see which part is which, and the three models use that mix in qualitatively different ways.

* **Trinity**'s emotion geometry is overwhelmingly text-driven (hence the low variance explained by its post-residualization PC1 axis). The clean circumplex is a reflection of structure that can largely be identified by reading into the text itself, not extra structure Trinity is imposing on top of text.
* **K2.5** has a dominant model-internal axis that isn't valence at all. Its circumplex in raw space is modestly there, but the model's own organization of emotional content runs along a different — and stronger — direction, visible only after text content is subtracted.
* **Cogito** encodes valence as an intrinsic geometric axis. The direction persists essentially unchanged under residualization; only the *amount* of variance it explains decreases.

Under single-model testing, the multiplicity of possible outcomes would be obscured. Only cross-model comparison separates "what residualization does to probe geometry" from "what this specific model happens to represent." Among the three, Cogito is the only one whose raw circumplex geometry survives the text-content control. This suggests that what looks like a universal affective structure across language models may in fact be universal-looking because text about emotions has universal-looking structure, not because every model is independently organizing valence as an internal axis.

## Discussion

We set out to extend Anthropic's emotion-probe investigation in three directions: across multiple models, via a text-residualization step separating text-predictable from model-internal activation structure, and across new probe families for steering and analysis.

**The cross-model picture changes things.** The circumplex replicates at the raw-probe level. However, under text-residualization, our three models behave qualitatively differently. Trinity's valence axis dissolves, K2.5's reorganizes around a different model-internal axis, and only Cogito preserves valence as a genuinely internal geometric structure. The "universal" affective geometry of emotion representations in LLMs is, on our data, sometimes text-driven rather than model-intrinsic.

**On concealment.** Anthropic's per-emotion deflection probes replicate as emotion-specific (approximately orthogonal to overt probes) on both Trinity and K2.5. Both models also show the same valence block structure in their raw deflection probes, which we interpret as a mechanical consequence of the contrast-vector construction. Our new cross-emotion construction — hiddenness\_X = deflection\_X − overt\_X — yields a universal concealment direction at cos ≥ +0.935 across all 15-emotion pairwise comparisons on both models, with within-valence means near +0.99. Universal concealment is clear, but its visibility requires the subtraction step because raw deflection probes inherit valence structure from the per-emotion dialogue construction.

**Limitations.** We cannot directly test Sonnet against our residualization procedure, so we can't place Anthropic's own model in the three-way split we see across Trinity, K2.5, and Cogito.

More broadly: what may have initially seemed like general phenomena, such as deflection or valence, may instead be specific to the model being studied. Generalization from single-model findings in mechanistic interpretability, including our own, warrants some amount of skepticism. Text-residualization and cross-architectural comparison are relatively cheap controls with outsized impact on what one is willing to claim.

As a closing note: We also ran tests on the *locality* of emotion features, quantifying and somewhat complicating Anthropic's claims that emotion features tend to be triggered primarily by tokens they're directly relevant to rather than having persistent effects across an entire rollout. Our write-up on this work can be found [here](https://latentaffect.up.railway.app/long_range_persistence_of_emotion_features.html).
