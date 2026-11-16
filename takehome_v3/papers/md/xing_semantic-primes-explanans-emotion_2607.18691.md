# Semantic Primes as Explanans for Emotion in Large Language Models

- **Authors:** Frank Xing
- **Date:** 2026-07-21
- **Source:** https://arxiv.org/abs/2607.18691
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `xing_semantic-primes-explanans-emotion_2607.18691`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Semantic Primes as Explanans for Emotion in Large Language Models**

## Frank Xing

University of Reading Email: z.xing@henley.ac.uk

## Abstract

Progresses have been made on understanding emotion mechanisms of large language models (LLMs). However, how to explain emotion in LLMs, or even what constitutes good explanations, are less clear. Emotion representations, components, circuits are widely recoverable, but as explanations of a model’s own computation they are circular; the emotion space dimensions tend to be arbitrary and non-terminating. A pressing question to ask is whether a more primitive set of internal variables does the work: the semantic primes of the Natural Semantic Metalanguage (NSM). Across four instruction-tuned LLMs (Llama-1B, Gemma-2B, Gemma-9B, OLMo-7B), experiments show that the NSM primes are (1) recoverable internal elements; and (2) on the reference model, intervening with a prime based direction controls emotion about three times as strongly, and twice as selectively, as the best appraisal based direction; and (3) the model treats a prime based explication as interchangeable with the corresponding emotion. These evidences suggest that NSM primes seem to be better explanans for emotion in LLMs than many alternative options according to scientific explanations criteria.

**Code & Data** — http://github.com/fxing79/pcs

## 1 Introduction

Unlike human emotion, which is phenomenal and embodied, emotion in LLMs is functional and computed. The difference calls for a whole set of theory on how emotion emerges and works in LLMs as they are increasingly used and integrated in our society. Progresses have been made on understanding that emotion space (Wu et al. 2026) and emotion representations (Sofroniew et al. 2026) are recoverable in LLMs, and that LLM emotion behavior can be controlled and manipulated via layer injection (Tak et al. 2025) or circuits (Wang et al. 2025). But there is a significant gap between understanding and (scientific) explanations. For example, I understand “a magician pulls a rabbit out of a hat” is a stage effect and is not real, but I cannot explain how he did it. People also have no difficulty understanding an apple falls after ripening without explaining it with plant hormones or gravity. In this sense, there has been little purposeful investigation on explanations of emotion in LLMs from the mechanistic interpretability literature (Bereska and Gavves 2024; Sharkey et al. 2025).

Preprint. Copyright with authors.

### What explains the emotion an LLM computes?

Figure 1: What makes a good causal explanation of the emotion an LLM computes and outoputs? Three tests are drawn from the scientific explanation literature: (1) real: it exists internally, (2) causal: intervening on it moves the emotion, (3) faithful: the model behaves as if it is the emotion. All three candidates pass them, but to differing degrees. What separates them most is the fourth vertical _basicness_ axis: an explanans must be more basic than the explanandum. Emotion labels are circular, and appraisals partial reduction; only NSM primes bottom out at a definitional floor.

Since using the emotion artifacts (representations, sparse autoencoder extractions, circuits, etc.) to explain themselves is non-semantic and circular, other internal variables must be engaged. Two concept families are often imported from human emotion psychology (Tak et al. 2025). The first family of variables are the “more basic” discrete emotion states or categories decoded from residual-stream activations as linear directions (Tigges et al. 2024), e.g., Ekman’s 6 basics or Russell’s Circumplex Model; the second family are continuous ratings, e.g., valence-arousal dimensions (Wu et al. 2026) or Scherer’s 21 appraisal dimensions that Tak et al. (2025) use to probe and steer LLM emotional behavior.

Two problems follow. First, judging the completeness and evaluating competing explanations are intricate. One set of variables may be more linearly recoverable via probing, but that fact alone does not establish better explanations: probe accuracy is correlational and need not reflect causal use (Belinkov 2022). The mechanistic interpretability field has answered this by adding interventions (Vig et al. 2020; Elazar et al. 2021), and on the narrow technical reading of the word, an explanation is _mechanistic_ only when it makes a causal claim (Saphra and Wiegreffe 2024). But when another set of variables steers LLM emotion behavior better, weighing between these two sets becomes complicated. Second, neither family of variable sets bottom out as explanations. To classify an activation as _guilt_ re-applies the annotator’s label to the very activation the label should explain: the explanans is the explanandum. The appraisal space dimensions describe guilt as a profile over a composition of _self_responsibility_ + _unpleasantness_ , but those dimensions are not more primitive than the emotions they describe; _unpleasantness_ is an affective concept of the same order as _guilt_ . The reduction floats and never terminates.

In response to the two problems, I attempt a set of tests to filter out pseudo-explanation and evaluate candidate explanans, and draw a third family of variables from a 60-year-old linguistic program, i.e., the Natural Semantic Metalanguage (NSM) of Wierzbicka and Goddard (Wierzbicka 1972, 1996; Goddard and Wierzbicka 2002, 2014). NSM identifies _∼_ 65 _semantic primes_ (WANT, FEEL, THINK, KNOW, DO, GOOD, BAD, NOT, BECAUSE, I, SOMEONE, _. . ._ ), held to be mutually indefinable and shared with the rest of language, so that every other concept, emotions included, can be _explicated_ as a paraphrase built only from primes. For instance, the gold explication of _guilt_ is roughly _“I did something bad; I feel bad because of this; I do not want this to have happened,”_ which bottoms out at primes (I, DO, BAD, FEEL, WANT, NOT) that are not themselves emotions. This explication of _guilt_ is carried as a running example. It is noteworthy that confirming the correctness of NSM in either human language or the LLM case is not this paper’s scope. Instead, my claim is an engineering one that NSM primes seem to be better explanans considering the criteria set out here.

The three families of variables are judged by the standard of causal explanation, on the interventionist account (Woodward 2003): a feature is explanatorily relevant to an outcome when intervening on it changes the outcome. Three demands elaborate (Figure 1), and they come from this account of explanation, not from any tenet of NSM.

- **Existence.** The feature is a real internal element, a genuine linear representation rather than a probe artifact. This is rigorously tested by decoding above length and control-task baselines.

- **Intervention.** Intervening on the feature changes the emotion the model computes, and more so the feature serves as an explanan better. This is tested by steering.

- **Behavioral equivalence.** At the input–output level the model treats the feature as the concept: in the NSM case, a prime explication is interchangeable with the emotion word and is a sufficient cue. This is tested with the model’s own inferences and generations.

The three demands are necessary: a feature that is absent can be part of the mechanism but is beyond explanations, one that is causally idle can be a confounder but does not explain, and one that is causally active but means something else is not the concept (Sharkey et al. 2025). They are not jointly a proof though: a prime’s lexical correlate could in principle pass all three, so construct validity remains an open problem (see Section 10), and the internal mechanism exhibited is a sketch focusing on the raw ingredients rather than a gap-free account.

**Contributions.** Primes inside four instruction-tuned models spanning two orders of magnitude and three model families (Llama-3.2-1B, Gemma-2-2B, Gemma-2-9B, OLMo-27B) are studied to ensure result generalizability. Experiments are built on Tak et al. (2025)’s public harness data so the emotion comparison is head-to-head.

The result is a single claim that NSM primes are better explanans for emotion in LLMs when compared to other explanans used in literature. It is supported by three findings: (1) primes are real internal elements as 30 of 32 emotiondescribing primes are linearly encoded above length and control-task baselines in all four models; (2) on the reference model, a prime direction controls emotion about three times as far, at nearly twice the selectivity, as the best appraisal direction, beating a dimensionality-matched composite ( _p <_ 10^(_−_3) ) and a random control; (3) the model treats a prime explication as interchangeable with the emotion and as a sufficient cue, in all four models and more so than a matched appraisal description. For reproducibility, a contrastive suite of 11,902 minimal pairs for 32 primes with controls and gold explications is released. Analysis runs on a single CPU and extraction in minutes on a single GPU, so the study is cheap to replicate.

## 2 Background

### 2.1 From explanans to their emotion readout

Mechanistic interpretability seeks a causal, not merely predictive, account of a computation: to explain “the model reports emotion _e_ ” is to exhibit the internal factor that _produces_ it. The two requirements discussed in Introduction sort the candidate explanations of emotion by how well each passes the causal tests and how far each reduces. In the emotion label case, a residual activation _h ∈_ R^_d_ at a consolidation layer admits a linear classifier into _K_ possible emotion labels; each emotion is a readout direction _we_ and the emotion space is flat, so _guilt_ is an atom and the account does not reduce: the explanans is the explanandum. In the appraisal case, the same _h_ admits 21 linear regressions onto appraisal scales; _guilt_ becomes a profile (2 dimensions activated). NSM primes are _∼_ 65 directions _vp_ , and an emotion becomes an _explication_ , a short structured paraphrase of prime predicates over a floor shared with the rest of language: _guilt ≈ “I did something bad; I feel bad because of this.”_ The compositional prediction is defined as:

with _P_ ( _e_ ) the prime support of _e_ ’s explication and _f_ the readout map. The strongest reading takes _f_ linear, which empirical results reject; the behavioral reading takes _f_ to be whatever the model computes and that is conceptually the NSM’s grammar, which empirical results confirm. Obviously, an explication is structured, not a bag of primes, so a signed sum of prime directions is lossy. Dimensional appraisal is by contrast additive, an emotion a profile over independent dimensions (Smith and Ellsworth 1985); the non-linearity of Scherer’s process model (Scherer 2009) is temporal, not in this static mapping. So the accounts differ in reduction depth and in the form of _f_ : on this reading prime composition would be grammatical and appraisal composition dimensional, a contrast the steering experiment puts to the test.

### 2.2 Primes and emotion in LLMs

That LLMs present and process semantic primes seems selfevident, though the proof is absent from literature, especially whether those are used when processing emotion. A recent study confirms prime-related schema slots from LLM outputs (Xing and Cambria 2026). At mid-layers, emotion is read from residual streams as linear structure (Tak et al. 2025) or extracted using a sparse autoencoder (Wu et al. 2026). The work closest to this one is Tak et al. (2025), who probe 13 emotion categories and 21 appraisal dimensions across 10 open LLMs, localize mid-layer emotion heads by activation patching and knockout, and steer behavior with orthogonalized appraisal injection; this pipeline is reproduced as the comparability anchor. Since primes are fundamental concepts, existence test adds a Hewitt-Liang control task (Hewitt and Liang 2019) to separate a represented feature from probe capacity, within the linear-representation framing of Park, Choe, and Veitch (2024); sparse autoencoders (Lieberum et al. 2024; Templeton et al. 2026) are an alternative route used only in pre-trained form; and an LLM pipeline that generates and verifies NSM explications (Baartmans et al. 2025) supplies those gold recipes not published in literature.

## 3 A Contrastive Suite for Semantic Primes

Probing primes needs stimuli that assert a single prime while holding lexical context fixed. For this reason, a contrastive minimal-pair suite for 32 primes is built and released with gold explications, code, and a machine-translated Chinese version for cross-lingual studies. For each prime templated pairs were generated: a positive member asserting the prime and a negative member with the same content but the prime absent. For example for WANT,

“ _(+)The nurse wants to play the piano_ ” “ _(-)The nurse played the piano_ ”; and for NOT, “ _(+)The driver did not open the gate_ ” “ _(-)The driver opened the gate_ .”

Templates are paraphrased under rejection rules that forbid the prime’s exponent word in the negative member, match target word-length within _±_ 1 where feasible, and track a syntactic-template identifier for a held-out split. The 32 primes span the symbol classes that admit a single-prime contrast: mental predicates (WANT, FEEL, THINK, KNOW, SAY, DO), evaluators (GOOD, BAD), modals (CAN, MAYBE, NOT, TRUE), person primes (I, SOMEONE, OTHER, PEOPLE), temporal, descriptor, bodily, and reason primes.

The selection of primes is principled, not a convenience sample. A contrastive probe needs a prime a sentence can assert or withhold while holding context fixed, which is natural for predicates and operators but ill-posed for referential substantives and determiners (THIS, SOMETHING, KIND) that appear in nearly every sentence, and for quantifiers (ONE, SOME, ALL) whose contrast is graded and noun-phrase-bound. Crucially, of the 22 distinct primes that appear in the gold explications of the 13 target emotions, the suite covers 21; the lone exception is THIS. The untested primes are dominated by the quantifier and spatial classes, which appear in no emotion explication, so the 32 primes already cover every emotion experiments required in this paper. The full prime inventory is in Appendix A.

The suite contains 11,902 minimal pairs / 23,804 sentences, _≥_ 202 pairs per prime, and _≥_ 16 held-out template pairs per prime. No negative leaks the prime’s exponent (a hard generation constraint); the length confound is quantified per prime as Cohen’s _d_ on word count, and the probing harness always reports a length-only baseline, so a prime counts as encoded only when it clears the length shortcut. The pipeline is validated on synthetic data, successfully recovering a planted layer for six synthetic primes with off-layer accuracy at chance (Appendix A).

## 4 Experimental Setup

**Models and data.** Four LLMs are studied: Llama-3.2-1BInstruct (16 blocks, _d_ =2048), Gemma-2-2B-it (26 blocks, _d_ =2304), Gemma-2-9B-it (42 blocks, _d_ =3584), and OLMo2-7B-Instruct (32 blocks, _d_ =4096), spanning two orders of magnitude and three families, including the fully open OLMo whose independent pre-training makes a cross-family claim more than a within-lineage one. Llama-3.2-1B is the reference model for the intervention experiment because of its quasi-linear emotion readout; the other three test prime existence and explication behavioral equivalence across family and scale. Emotion and appraisal labels come from crowdenVent (Troiano, Oberländer, and Klinger 2023) (6,800 event descriptions annotated for 13 emotions and 21 appraisal dimensions); I reproduce Tak et al. (2025)’s split of 2,740 train and 1,370 test sentences, with ISEAR (Scherer and Wallbott 1994) as an additional robustness check.

**Probes and steering.** Linear probes are L2-regularized logistic regression, appraisal probes L2 ridge; hyperparameters are not retuned per layer. Steering adds a unit-normalized direction, scaled by the layer’s mean residual norm, to the block residual at layer 11, the mid-network consolidation site fixed by the patching peak (Section 7), at all token positions. The behavioral tests use only the model’s generations and answer-token logits, with no probe, and run on all four models. For existence, this study reports per-prime peak accuracy against the length baseline and Hewitt-Liang selectivity; for intervention, target-logit shift, off-target mass, selectivity, and target-argmax rate; for behavioral equivalence, explicationto-word logit agreement, explication-only classification with prime ablation, and the linear recipe match. Cached activations are reused and never recomputed.

## 5 Primes Are Real Internal Elements

This section presents existence test results: a prime must be a genuine internal element, not a probe artifact. Decoding establishes this only with controls. Probe accuracy alone only shows a label is recoverable, while clearing a Hewitt-Liang control task shows the model represents the prime (Hewitt and Liang 2019), since the control measures what a probe of the same capacity can fit on structureless pseudo-labels.

**Protocol.** For each of the 32 primes, residual activations are extracted on its contrastive pairs (balanced, last token, every layer) and fit one logistic probe per layer, recording a length-only baseline, a Hewitt-Liang control-task probe (Hewitt and Liang 2019), and a held-out template split. A prime counts as _linearly encoded_ when its peak accuracy clears the length baseline significantly and shows selectivity over other primes.

**Existence and replication.** Thirty of the 32 primes are linearly encoded in all four models: 31/32 on Llama-3.21B (bootstrap 95% CI [29 _,_ 32]), 30/32 on Gemma-2-2B, and 32/32 on Gemma-2-9B and OLMo-2-7B, across three families and the 1B-to-9B scale range. Averaged over the 32 primes on Llama-3.2-1B, held-out template decoding reaches 0 _._ 91, far above the Hewitt-Liang control task (0 _._ 50, at chance) and the length-only baseline (0 _._ 62); the gap between probe and control is the selectivity that makes this an existence claim and not a probe artifact (Figure 2). The two exceptions across models, DO and SAY, carry the worst length confound ( _d_ =1 _._ 26 and 2 _._ 03); other confounded primes (KNOW, CAN, THINK, all _d >_ 1 _._ 2) still clear the bar, so the control separates genuine encoding from a length shortcut. The atoms _guilt_ ’s explication is built from, I, DO, BAD, and FEEL, are all among the encoded primes, so the ingredients its recipe needs are all present in LLMs. Decoding with controls, replicated across four models, establishes existence but not use: a probe can read a direction the model never uses, which the next section tests by intervention.

Figure 2: Existence on Llama-3.2-1B: per-layer prime decoding, averaged over the 32 primes. Held-out template accuracy (red) rises to 0 _._ 91 and stays far above the Hewitt-Liang control-task accuracy (grey, near the chance line 0 _._ 5) and the length-only baseline (dashed); the shaded gap is the selectivity that separates a represented prime from a probe artifact.

## 6 Primes Control Emotion Better

This section presents intervention test results. In short, a direction assembled from prime atoms and injected into the residual stream is found to control emotion more strongly and more selectively than the best appraisal based direction. The shift is about three times as large at nearly twice the selectivity, the gap widest been on the agency axis (i.e., I versus SOMEONE), suggesting a less effective modeling construct by appraisals.

**One mechanism, six directions.** A forward hook adds a vector to the block residual at layer 11, at all token positions. The six arms differ only in the injected direction: the _emotion_ probe direction _we_ (a readout ceiling); the _single appraisal_ most associated with the target (e.g., guilt to _self_responsblt_ , anger to _other_responsblt_ , joy to _pleasantness_ , sadness to _unpleasantness_ ); a _multi-appraisal composite_ , the top- _k_ appraisals with _k_ matched to the recipe’s component count; the _single best prime_ ; the _prime recipe_ , a centrality-weighted signed sum of fitted prime directions (guilt as I + DO + BAD, so is still lossy); and a _random-concept_ control. Every direction is unit-normalized and scaled by the layer’s mean residual norm, so the dose is a common fraction of residual magnitude; the composite and random arms control for direction count and for injecting any vector at all. The 13 emotion-token logits are read at the answer position over four targets (guilt, anger, joy, sadness, see Table 1).

**Per-target steering detail.** Averaged over the four targets and the positive dose grid ( _β ∈{_ 0 _._ 5 _,_ 1 _,_ 2 _}_ ), the prime recipe shifts the target emotion by 3 _._ 73 logits at selectivity 0 _._ 58, while the single best appraisal shifts it by 1 _._ 29 at 0 _._ 31 (Table 2, Figure 3). Against the component-matched composite the prime recipe still shifts emotion further (3 _._ 73 versus 2 _._ 11 grid-averaged; strongest dose 4 _._ 91, CI [4 _._ 76 _,_ 5 _._ 05], versus 2 _._ 84, [2 _._ 68 _,_ 2 _._ 99]) and more selectively (0 _._ 60 versus 0 _._ 54, non-overlapping); a permutation test gives a +2 _._ 07-logit gap, _p <_ 10^(_−_3) . The random-concept control barely moves the target (0 _._ 02, selectivity _−_ 0 _._ 26), a near-zero causal floor, and a single best prime already about matches the full recipe (3 _._ 62 at 0 _._ 61), so the handle is carried by the prime representation at matched dimensionality, not by component count. The advantage is sharpest on the agency axis (selectivity 0 _._ 57 for primes versus 0 _._ 27 for the single appraisal); for anger the prime arm drives anger to the top prediction in 77% of prompts. Injecting the emotion direction itself shifts the target most (9 _._ 80, selectivity 0 _._ 78), as expected for the readout direction; the prime arm, built from general-purpose atoms, recovers a large and selective fraction of that ceiling.

||emot|ion|appr|aisal|pri|me|
|---|---|---|---|---|---|---|
|Target|shift|sel|shift|sel|shift|sel|
|guilt|17_._34|0_._83|0_._41|0_._13|3_._75|0_._55|
|anger|12_._73|0_._78|1_._82|0_._38|6_._60|0_._66|
|joy|14_._29|0_._77|4_._69|0_._59|4_._67|0_._62|
|sadness|10_._40|0_._72|1_._25|0_._33|4_._61|0_._56|

Table 1: Per-target steering at _β_ = 2 (Llama-3.2-1B). The emotion arm is the readout ceiling; the prime arm beats the appraisal arm on every target emotion.

|_Llama-3.2-1B_, over four targets_×_dose|grid|
|---|---|
|Arm shift|selectivity|
|emotion (ceiling) 9_._80|0_._78|
|prime recipe (ours) 3_._73|0_._58|
|prime, single best 3_._62|0_._61|
|appraisal, multi (matched) 2_._11|0_._41|
|appraisal, single 1_._29|0_._31|
|random (control) 0_._02|_−_0_._26|
|agency axis, prime _..._|0_._57|
|agency axis, appraisal _..._|0_._27|
|_Gemma-2-9B_, grid-average shift||
|Composition linear|non-linear|
|prime (NSM) 0_._02|0_._97|
|appraisal 0_._74|0_._41|

Table 2: Steering, two models. Llama: the prime recipe beats both appraisal arms on shift and selectivity (gap versus composite +2 _._ 07, _p <_ 10^(_−_3) ); random control near zero; emotion ceiling 9 _._ 80. Gemma-9B: the model’s own explication encoding steers primes (mostly via content) but not appraisals; ceiling 1 _._ 82, floor 0 _._ 10.

Figure 3: Steering head to head (Llama-3.2-1B): prime (red), appraisal (blue), emotion (yellow). (a) Mean target-logit shift versus dose. (b) Selectivity versus shift for all positive doses; upper right is better. The prime arm dominates the appraisal arm on both axes; the emotion arm is the readout ceiling – best control but weakest explanatory power.

**Robustness check on injection layer.** The prime advantage is not an artifact of a particular direction fit or injection site. It holds in all six bootstrap resamples of the probe-fitting data, where the appraisal arm varies but the prime arm stays ahead, and at every injection layer from 7 to 14, where all directions including the prime recipe are refit, the prime arm leads. The steered direction is also nearly orthogonal to the emotion readout: the cosine between the prime recipe and the emotion probe direction _we_ averages 0 _._ 04 across targets. A direction geometrically unrelated to the readout still drives the emotion logit, which is the signature of a _compositional_ causal effect: the primes are injected upstream and processed by the network into the emotion, not read off the recipe directly. This also reconciles with the geometric non-reduction of Section 7, where the emotion readout is not a linear function of prime directions. The composition is robust to the ingredient: assembled from the model’s own single-prime encodings, a contrastive difference of means, rather than from probe directions, the recipe steers as well (4 _._ 5 versus 3 _._ 9 at matched selectivity 0 _._ 59), and both far exceed injecting the whole explication’s encoding (1 _._ 1, selectivity _−_ 0 _._ 25). Independent prime vectors, composed, out-steer the holistic sentence encoding, so the effect is neither a probe-geometry artifact nor a global-content injection.

**On larger models the linear recipe does not transfer.** The linear prime recipe is model-specific. With per-model dose and layer calibration it is inert on Gemma-2-9B (gridaveraged shift 0 _._ 02, at the random floor), and assembling it from the model’s own single-prime encodings rather than probe directions does not rescue it (0 _._ 14): the primes do not linearly compose there by either ingredient. The emotion is reachable only from the whole explication’s encoding (0 _._ 97 at selectivity 0 _._ 66, against an emotion-readout ceiling of 1 _._ 82), the trivial content direction any model carries for guilt-describing text. Phrased in primes that content still outsteers a matched appraisal description (1 _._ 58 versus 0 _._ 74), and the same treatment leaves appraisal at its additive profile (0 _._ 74 versus 0 _._ 41; Table 2). For this reason, the main analysis uses Llama-3.2-1B, where the compositional recipe itself steers; on Gemma-2-9B the prime explication out-steers the appraisal description as content, but the linear prime composition does not transfer. On OLMo-2-7B the readout ceiling is ill-set (a negative emotion-arm shift), so numbers are uninformative and not reported.

## 7 Emotions Reduce to Primes Behaviorally

This section presents behavioral equivalence test results. In all four models the model treats a prime explication as the emotion. The reduction is behavioral, not geometric: the late readout is not a linear sum of prime directions, and the localization of emotion within the network explains why the two levels diverge.

**Behavioral equivalence and sufficiency.** With no probing and on all four models, the 13 emotion-token logits are read for: (1) the emotion word, (2) its gold explication, (3) a sameaffect decoy explication, e.g., shame for guilt, differing by one PEOPLE-KNOW component, and (4) a prime-scrambled control. Given the explication alone, with no emotion word present, the model classifies it as the target emotion well above the 13-way chance rate of 0 _._ 08, at 0 _._ 54, 0 _._ 85, 0 _._ 77, and 0 _._ 54 respectively for Llama-3.2-1B, Gemma-2-2B, Gemma2-9B, and OLMo-2-7B, and the target explication beats its same-affect decoy by 0 _._ 30 to 0 _._ 63 probability mass, so it is not riding on generic affect words or on the presence of an emotion label, which the explication never contains. Reduction is also sufficient: ablating the explication one prime at a time, removing a _central_ prime degrades the target logit more than removing a _peripheral_ one in every model (0 _._ 59 versus 0 _._ 12 on Llama-3.2-1B, and the same ordering elsewhere; see Table 3).

For _guilt_ , the evaluative BAD and FEEL are central and the WANT/NOT clause peripheral: the central primes carry the emotion. The same battery on a matched appraisal description, a Component-Process-style clause profile built from the crowd-enVent appraisal ratings, is markedly weaker (Table 3): the description is read as the target emotion below the prime explication on every model (0 _._ 25 to 0 _._ 42 versus 0 _._ 54 to 0 _._ 85), beats its same-affect decoy by only 0 _._ 03 to 0 _._ 11 against the prime’s 0 _._ 30 to 0 _._ 63, and its central-clause sufficiency inverts on the two larger models. The model treats a prime explication, not a matched appraisal profile, as interchangeable with the emotion, on our best operationalization of each. A fluency confound does not explain this gap. On the reference model the prime explication is more fluent than the matched appraisal description (mean per-token log-probability higher by 0 _._ 81 nats, _p <_ 10^(_−_3) ), yet under fluency-invariant scoring, domain-conditional PMI (Holtzman et al. 2021) and contextual calibration (Zhao et al. 2021), the prime advantage persists and widens, because calibration removes a label-prior bias that suppressed the explication.

||Llama|Gem.2B|Gem.9B|OLMo|
|---|---|---|---|---|
|_prime explication_|||||
|(E) expl_→_emotion acc|0_._54|0_._85|0_._77|0_._54|
|(E) decoy margin|0_._30|0_._60|0_._63|0_._35|
|(S) central drop|0_._59|0_._14|0_._18|0_._51|
|(S) peripheral drop|0_._12|_−_0_._21|0_._03|_−_0_._14|
|_appraisal description_|||||
|(E) expl_→_emotion acc|0_._25|0_._33|0_._33|0_._42|
|(E) decoy margin|0_._03|0_._10|0_._03|0_._11|
|(S) central drop|0_._68|0_._92|_−_0_._24|_−_0_._27|
|(S) peripheral drop|0_._35|0_._32|0_._03|0_._31|
|floor: prime|0_._47|0_._21|0_._34|0_._40|
|floor: non-prime|0_._73|0_._50|0_._72|0_._72|

Table 3: Behavioral test batteries, all four models. 13-way chance for classification is 0 _._ 08. The explications are read as the emotion (E) and are compositionally sufficient (S, central _>_ peripheral in every model) more strongly than a matched appraisal description, whose decoy margin is near zero and whose sufficiency gradient inverts on the two larger models. Primes resist simplification more than non-primes in every model (floor).

**The readout is not a linear sum.** The prime directions are fit from the contrastive suite in the same residual space as the emotion vectors and build a centrality-weighted recipe direction per emotion from gold explications (Wierzbicka 1999); the match test asks whether _we_ is closer to its own recipe than to the other twelve emotions. Both direct sources land near chance (Table 4): out-of-context rank-3 0 _._ 39 against chance 0 _._ 23, in-context exactly at chance. Emotion is linearly recoverable from the prime coordinates, but not because of them: projecting the consolidation-layer residual onto the 32 prime directions and training a classifier reaches 0 _._ 86, yet a _random_ 32-dimensional projection reaches 0 _._ 82, so the prime subspace adds only 0 _._ 05, and per-prime ablation moves accuracy by at most 0 _._ 02. The prime subspace is not a privileged basis.

**A mechanism sketch.** The two findings reconcile through localization. The emotion computation peaks at layer 10, and by the consolidation layer the prime content has been _consumed_ into the emotion representation: of 18 primes tested at the answer position, only FEEL (lift +0 _._ 23) and SOMEONE (+0 _._ 11) remain decodable. The late readout cannot be a linear sum of prime directions because there are almost no prime directions left to sum, and the survivors are in a sense, the emotion-handles: the feeling predicate and the other-person agency role. Emotions reduce to primes as a computation the model carries out in the middle layers and spends, not as a static linear geometry of the final state. I present this as a mechanism sketch: it localizes where the reduction happens but does not trace the layer-10-to-readout step arrow by arrow.

Figure 4: The definitional floor, behaviorally. Asked to restate a word in simpler terms, the model produces a smaller fraction of simpler (shorter, higher-frequency) words for a prime than for a length- and class-matched non-prime, in all four models: a prime resists reduction, a non-prime does not. Error bars are the standard error of the mean over the primes and the matched non-primes.

## 8 Do Primes Bottom Out in LLMs?

The explication reduces an emotion to primes; a probe-free signature shows the primes themselves do not reduce further. Prompted to restate a word “using only simpler words,” the model simplifies a non-prime far more readily than a prime: the fraction of output words more frequent than the target is 0 _._ 73 _/_ 0 _._ 50 _/_ 0 _._ 72 _/_ 0 _._ 72 for non-primes against 0 _._ 47 _/_ 0 _._ 21 _/_ 0 _._ 34 _/_ 0 _._ 40 for primes (see Table 3 and Figure 4). A prime resists reduction in a sense that more than half of the explaining words cannot be simpler, so the explication regress terminates. This is the bottom-out property an explanation demands: _guilt_ reduces to I, DO, BAD, and FEEL, which are not to be explained further in the semantic space, where a label or an appraisal dimension would still float.

The localization of primes also seems earlier than emotions. I reproduce and concur the localization of emotions on Llama3.2-1B by Tak et al. (2025), which fixes the consolidation band and the injection layer. Emotion accuracy saturates late (linear probe peak 0 _._ 943 at layer 13); activation patching is sharper, transplanting the event-position residual flips the target prediction at a rate that climbs through the mid-network and peaks at 0 _._ 805 at layer 10 before falling, recovering a mid-layer attention-mediated consolidation. It is this peak that fixes layer 11 as the injection site. Full curves and the emotion geometry, where valence dominates and the agency contrast is secondary, are in Appendix B.

## 9 Discussion

This paper shows the potential of semantic primes as a handle that explains in LLMs, not predicts over complicated theories. The two standard accounts of emotion in LLMs do not bottom out: labels relabel the activation, appraisals describe it in terms no more primitive than emotion, and both are validated mostly by probes that show recoverability, not use. Judged by what it takes to explain the emotion a model computes, primes pass where these fall short: they are real internal elements, the model treats their explications as the emotion, and intervening on them controls emotion more than appraisals do. The same primes (WANT, NOT, CAN, FEEL) appear in explications of decisions, permissions, and requests, so the prime-interface is more universal than emotion-specific explanans.

It worths mentioning, however, that geometrically the emotion direction is not a linear sum of prime directions; though behaviorally the model treats an emotion word and its explication as interchangeable and the explication as sufficient. A causal handle need not be a geometric basis. The reconciliation is mechanistic: primes are computed in the mid layers (e.g., 7/16) and consumed into the emotion representation (11/16), so the final state retains only the feeling and agency atoms. A prime-indexed account of emotion is therefore viable, but it must read emotion off the primes nonlinearly.

The clean compositional result is the Llama one, and it is strong there: independent prime vectors, whether probe directions or the model’s own single-prime encodings, compose into a direction that out-steers the whole explication’s encoding, so the effect is composition from parts, not a global-content injection. On Gemma-2-9B the parts do not compose by either ingredient; only the whole-content direction steers, which is the trivial encoding any model has for emotion-describing text. The prime explication still outsteers a matched appraisal description there, and the appraisal arm gains nothing from the same treatment, so the content asymmetry is real cross-model even though the compositional claim is Llama-only. Tracing whether a larger model composes primes through a different circuit is left open (Section 10).

One structure recurs as a causal signal even though valence dominates the static geometry. Agency is where prime steering most beats appraisal (selectivity 0 _._ 57 versus 0 _._ 27), the one decomposition that survives across direction sources (anger at rank-1), and SOMEONE is one of only two primes still decodable at the readout. In NSM it is a single substitution, I _↔_ SOMEONE, inside a recipe of the form _X did something bad; I feel bad because of this_ , the minimal edit that turns guilt into anger.

## 10 Limitations and Conclusion

The main limitation is on construct validity. Because the three tests are necessary for a set of variables to count as an explanation, but not complete or jointly a proof, it is possible that a decoded direction measures a lexical correlate, not the NSM prime per se. More external validation can be supplemented by blind human paraphrases written by annotators unaware of the target prime.

A second limitation is on the full understanding from prime ingredients to emotion representations: the mechanism is only a sketch. The intervention is clean on Llama-3.2-1B and transfers to Gemma-2-9B under non-linear composition (Section 6); but OLMo-2-7B is not calibrated, and the nonlinear appraisal arm uses a constructed appraisal description that a stronger operationalization could potentially improve, so the cross-model causal claim rests on two models. The mechanism that reconciles behavioral and geometric reduction localizes a mid-network consolidation band but does not trace the layer-10-to-readout step. Concretely, the attention heads or MLP sub-modules that compose the primes, or the NSM grammar component, are unidentified. The gold recipes are taken from the published NSM canon (Wierzbicka 1999; Goddard and Wierzbicka 2014), whose explications carry the validity of a long peer-reviewed program; what is specific to us is the reduction to a 32-prime working subset and the centrality weighting used for ablation. Behavioral robustness to that operationalization can be further checked by recipeperturbation or re-deriving the central-versus-peripheral ablation under alternative but NSM-valid explications.

Another limitation is that cross-lingual universality of primes is not tested in the LLM case, though it is one of the NSM proposition for human language. Transfer to other languages would be confounded by a multilingual model routing other languages through an English-like latent space (Wendler et al. 2024; Schut, Gal, and Farquhar 2025), so separating universality from English-pivot routing requires native-authored stimuli in a typologically distant language and ideally an individually trained non-English LLM, which is left to future work.

In summary, this research confirms semantic primes to be good explanans of emotion in LLMs and requires minimal theory-building. Although other models (Tak et al. 2025), linguistic features (Chang 2025) or narrative style (Shen et al. 2024) may as well explain emotion, they are likely correlational (not causal) and subject to further specification. It also presents a framework for assessing the quality of scientific explanations, therefore makes methodological contribution to mechanistic interpretability.

## References

Baartmans, R.; Raffel, M.; Vikram, R.; Deringer, A.; and Chen, L. 2025. Towards Universal Semantics with Large Language Models. arXiv:2505.11764.

Belinkov, Y. 2022. Probing Classifiers: Promises, Shortcomings, and Advances. _Computational Linguistics_ , 48(1): 207–219.

Bereska, L.; and Gavves, E. 2024. Mechanistic Interpretability for AI Safety - A Review. _Transactions on Machine Learning Research_ , 2024.

Chang, E. Y. 2025. _Modeling Emotions in Multimodal LLMs_ , chapter 8, 1–10. Association for Computing Machinery.

Elazar, Y.; Ravfogel, S.; Jacovi, A.; and Goldberg, Y. 2021. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. _Transactions of the Association for Computational Linguistics_ , 9: 160–175.

Goddard, C.; and Wierzbicka, A. 2002. _Meaning and Universal Grammar: Theory and Empirical Findings_ . Amsterdam: John Benjamins.

Goddard, C.; and Wierzbicka, A. 2014. _Words and Meanings: Lexical Semantics across Domains, Languages, and Cultures_ . Oxford University Press.

Hewitt, J.; and Liang, P. 2019. Designing and Interpreting Probes with Control Tasks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ .

Holtzman, A.; West, P.; Shwartz, V.; Choi, Y.; and Zettlemoyer, L. 2021. Surface Form Competition: Why the Highest Probability Answer Isn’t Always Right. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ .

Lieberum, T.; Rajamanoharan, S.; Conmy, A.; Smith, L.; Sonnerat, N.; Varma, V.; Kramár, J.; Dragan, A.; Shah, R.; and Nanda, N. 2024. Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. In _Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ , 278–300.

Park, K.; Choe, Y. J.; and Veitch, V. 2024. The Linear Representation Hypothesis and the Geometry of Large Language Models. In _Proceedings of the 41st International Conference on Machine Learning (ICML)_ .

Saphra, N.; and Wiegreffe, S. 2024. Mechanistic? In _Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ .

Scherer, K. R. 2009. The Dynamic Architecture of Emotion: Evidence for the Component Process Model. _Cognition and Emotion_ , 23(7): 1307–1351.

Scherer, K. R.; and Wallbott, H. G. 1994. Evidence for Universality and Cultural Variation of Differential Emotion Response Patterning. _Journal of Personality and Social Psychology_ , 66(2): 310–328.

Schut, L.; Gal, Y.; and Farquhar, S. 2025. Do Multilingual LLMs Think in English? In _ICLR 2025 Workshop on Building Trust in Language Models and Applications_ , 1–37.

Sharkey, L.; Chughtai, B.; Batson, J.; and et al. 2025. Open Problems in Mechanistic Interpretability. _Transactions on Machine Learning Research_ , 2025: 1–82.

Shen, J.; Mire, J.; Park, H. W.; Breazeal, C.; and Sap, M. 2024. HEART-felt Narratives: Tracing Empathy and Narrative Style in Personal Stories with LLMs. In _ENMLP_ , 1026–1046.

Smith, C. A.; and Ellsworth, P. C. 1985. Patterns of Cognitive Appraisal in Emotion. _Journal of Personality and Social Psychology_ , 48(4): 813–838.

Sofroniew, N.; Kauvar, I.; Saunders, W.; and et al. 2026. Emotion Concepts and their Function in a Large Language Model. arXiv:2604.07729.

Tak, A. N.; Banayeeanzade, A.; Bolourani, A.; Kian, M.; Jia, R.; and Gratch, J. 2025. Mechanistic Interpretability of Emotion Inference in Large Language Models. In _Findings of the Association for Computational Linguistics: ACL 2025_ .

Templeton, A.; Conerly, T.; Marcus, J.; Lindsey, J.; and et al. 2026. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. arXiv:2605.29358.

Tigges, C.; Hollinsworth, O. J.; Geiger, A.; and Nanda, N. 2024. Language Models Linearly Represent Sentiment. In _Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ , 58–87.

Troiano, E.; Oberländer, L. A. M.; and Klinger, R. 2023. Dimensional Modeling of Emotions in Text with Appraisal Theories: Corpus Creation, Annotation Reliability, and Prediction. _Computational Linguistics_ , 49(1): 1–72.

Vig, J.; Gehrmann, S.; Belinkov, Y.; Qian, S.; Nevo, D.; Singer, Y.; and Shieber, S. 2020. Investigating Gender Bias in Language Models Using Causal Mediation Analysis. In _Advances in Neural Information Processing Systems (NeurIPS)_ .

Wang, C.; Zhang, Y.; Yu, R.; and et al. 2025. Do LLMs "Feel"? Emotion Circuits Discovery and Control. arXiv:2510.11328.

Wendler, C.; Veselovsky, V.; Monea, G.; and West, R. 2024. Do Llamas Work in English? On the Latent Language of Multilingual Transformers. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)_ .

Wierzbicka, A. 1972. _Semantic Primitives_ . Frankfurt: Athenäum.

Wierzbicka, A. 1996. _Semantics: Primes and Universals_ . Oxford University Press.

Wierzbicka, A. 1999. _Emotions across Languages and Cultures: Diversity and Universals_ . Cambridge University Press.

Woodward, J. 2003. _Making Things Happen: A Theory of Causal Explanation_ . Oxford University Press.

Wu, X.; Wang, H.; Yan, Z.; and et al. 2026. Decoding and controlling emotion in LLMs through human-aligned representational geometry with enhanced interpretability. _Computers in Human Behavior_ , 183(109051): 1–17.

Xing, F.; and Cambria, E. 2026. Faithful by Definition: Emotion Analysis via Natural Semantic Metalanguage Explications. arXiv:2607.00661.

Zhao, Z.; Wallace, E.; Feng, S.; Klein, D.; and Singh, S. 2021. Calibrate Before Use: Improving Few-Shot Performance of Language Models. In _Proceedings of the 38th International Conference on Machine Learning (ICML)_ .

## A The Contrastive Suite: Inventory, Confounds, Synthetic Validation

**Symbol classes and the full inventory.** The 32 tested primes are mental predicates (WANT, FEEL, THINK, KNOW, SAY, DO), evaluators (GOOD, BAD), modals (CAN, MAYBE, NOT, TRUE), person primes (I, SOMEONE, OTHER, PEOPLE), temporal (NOW, BEFORE, AFTER, A-LONG-TIME, A-SHORTTIME), descriptors (BIG, SMALL, VERY, MORE, SAME), bodily and animate (BODY, LIVE, DIE), and reason (BECAUSE, IF, HAPPEN). The untested _∼_ 33 are dominated by quantifiers (ONE, TWO, SOME, ALL, MUCH, FEW), spatial primes (HERE, ABOVE, BELOW, NEAR, FAR, INSIDE, TOUCH, SIDE, WHERE), and the referential substantives and determiners (YOU, SOMETHING, THIS, KIND, PART); none of these appears in a gold emotion explication, and none admits a clean single-prime assertion-contrast. Of the 22 distinct primes used across the 13 emotion recipes, 21 are tested (only THIS is missing).

**Splits, confounds, validation.** Split is done 70/30 at the template level. Negated sub-sets (WANT, KNOW, CAN, TRUE) and graded sub-sets (TIME3, VALENCE3, SIZE3, DURATION2) test partial orderings inside a prime family. Zero negatives leak the prime’s exponent; 13 primes show _|d| >_ 0 _._ 5 on word count, the most extreme SAY at _d_ =2 _._ 03. The harness recovers a planted layer for all six synthetic primes with off-layer accuracy and length baseline at chance (Figure 5).

Figure 5: Synthetic validation: a prime-style signal planted at layer 7 of a random-initialized 12-layer transformer is recovered for all six synthetic primes (the spike); off-layer accuracy and length baseline at chance.

## B Localization: Full Curves and Geometry

Emotion accuracy peaks at 0 _._ 943 (linear) at layer 13 and the MLP at 0 _._ 947 at layer 14; mean appraisal _R_^2 reaches 0 _._ 30 in the same band, led by _pleasantness_ (0 _._ 70) and _unpleasantness_ (0 _._ 65) (Figure 6). The emotion geometry is dominated by valence: measured as the Pearson correlation between _we_^(_⊤h_and)^_w_ _a_^(_⊤h_, the strongest cells are emotions on) pleasantness and unpleasantness, and the agency contrast is weaker (anger loads 0 _._ 15 on other- versus _−_ 0 _._ 30 on selfresponsibility; Figure 7). Zeroing a 3-layer span at the event position collapses agreement with the clean model to 0 _._ 11 at layer 9, while a norm-matched random perturbation is less damaging; activation patching peaks at 0 _._ 805 at layer 10 (Figure 8).

Figure 6: Layer-wise probing of Llama-3.2-1B on crowdenVent. Emotion accuracy (left) and mean appraisal _R_^2 (right) saturate late, fixing the consolidation band; chance accuracy is 1 1 _/_ 13 = 0 _._ 08..

Figure 8: Residual-stream knockout and patching. Zeroing a 3-layer span at the event position (red) maximally disrupts the emotion decision; a random perturbation (green) is less damaging. Patching peaks at layer 10 (flip rate 0 _._ 805).

Figure 7: Decoded-score correlation _ρ_ between the 13 emotion readouts and the appraisal readouts at layer 13. Valence dominates; the agency contrast is fainter, visible as anger loading on other- over self-responsibility.

## C Behavioral Tests and In-Context Decodability

The behavioral tests use only generations and answer-token logits, so they can run on all four models. The 13 emotiontoken logits are read for: (1) the emotion word, (2) its gold explication, (3) a same-affect decoy explication, and (4) a prime-scrambled control. Sufficiency ablates the explication one prime at a time (central vs peripheral by the gold recipe); the definitional floor scores the fraction of restatement tokens more frequent than the target. Table 3 has collected the numbers; Figure 9 shows the full recipe-match matrix and Table 5 the in-context decodability behind the geometric non-reduction (only FEEL and SOMEONE clear +0 _._ 10).

Figure 9: Emotion-to-recipe match matrix (Llama-3.2-1B). Rows are emotion readout directions, columns recipe-implied directions; a correct linear decomposition would put the maximum on the diagonal. Only a few emotions, led by anger, match their own recipe.

|Prime|held-out|majority|lift|
|---|---|---|---|
|FEEL|0_._971|0_._745|+0_._226|
|SOMEONE|0_._849|0_._739|+0_._109|
|NOT|0_._861|0_._778|+0_._084|
|I|0_._969|0_._939|+0_._030|
|BECAUSE|0_._894|0_._877|+0_._017|
|BAD|0_._954|0_._970|_−_0_._015|

Table 5: In-context prime decodability at the consolidation layer, answer position (selected rows). Lift is held-out minus majority. Only FEEL and SOMEONE clear +0 _._ 10.

|Prime-direction source|rank-1|rank-3|_F_1|
|---|---|---|---|
|random dictionary|0_._08|0_._23|0_._00|
|in-context (clean space)|0_._08|0_._23|0_._35|
|out-of-context (standalone)|0_._15|0_._39|0_._34|

Table 4: Linear emotion-to-recipe match for prime directions fit from the contrastive suite (Llama-3.2-1B). Chance is 0 _._ 08 (rank-1) and 0 _._ 23 (rank-3). The in-context source, the only one without a space mismatch, is at chance because the primes are consumed by the consolidation layer; linear composition is weak, in contrast to the behavioral reduction in Section 7.
