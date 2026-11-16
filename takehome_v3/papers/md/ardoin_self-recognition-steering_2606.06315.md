# LLM Self-Recognition: Steering and Retrieving Activation Signatures

- **Authors:** Thibaud Ardoin, Jonas Schäfer, Gerhard Wunder
- **Date:** 2026-06-04
- **Source:** https://arxiv.org/abs/2606.06315
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `ardoin_self-recognition-steering_2606.06315`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Thibaud Ardoin** [1] **Jonas Schäfer** [1] **Gerhard Wunder** [1]

**Abstract**

Recent advances in interpretability suggest that
large language models (LLMs) implicitly encode
signals in their generated text that enable self-recognition of their outputs. We demonstrate that
this capability is reliable, even in low-entropy
scenarios, and that it can be amplified through
targeted intervention. By steering the internal
residual stream during generation with a random
sparse vector, we create a detectable fingerprint
that enables attribution of a given text to a specific LLM. This signal is recoverable from the
activations of an LLM used as a detector, achieving over 98% accuracy across multiple detection
settings while preserving the quality of generated
text. As AI-generated content proliferates, this approach offers a practical alternative to traditional
detectors by leveraging the model’s natural representation structure for attribution rather than
embedding a signal externally. Our contributions
include: (i) establishing reliable self-recognition
capabilities in LLMs, (ii) a simple steering mechanism enabling multi-LLM identification with no
quality degradation, (iii) demonstrating that activation spaces contain exploitable structure for
encoding signals without semantic interference.

## 1. Introduction

As large language models (LLMs) become increasingly
deployed for content generation, concerns about the authenticity and traceability of AI-generated text have intensified.
Unlike human-authored text, LLM generations often lack
an intrinsic link to a responsible author, raising risks of
knowledge base contamination and accountability gaps in

1Department of Computer Science, Freie Universitaet
Berlin, Germany. Correspondence to: Thibaud Ardoin
<thibaud.ardoin@fu-berlin.de>.

_Proceedings_ _of_ _the_ _43_ _[rd]_ _International_ _Conference_ _on_ _Machine_
_Learning_, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).

Figure 1: Steering and retrieval of LLM signatures. (a) During

generation, steering vectors **v** 1 and **v** 2 are added to intermediate
activations at layer _n_, creating model-specific signatures in outputs
_t_ 1 and _t_ 2. (b) To verify authorship, the generated text is passed
through the same model to collect the activations at layer _n_ . The
source steering vector is retrieved via cosine similarity or a trained
MLP classifier.

high-stakes domains. Distinguishing not only whether content is AI-generated, but which specific model produced it,
is essential for auditing, attribution, and misuse prevention
(Bjelobaba et al., 2025; Yousaf, 2025).

Current approaches to AI-generated text detection (AI-GTD)
predominantly rely on watermarking methods that either
modify token probability distributions (Kirchenbauer et al.,
2023) or strategically select tokens to encode hidden information (Hou et al., 2023; Christ et al., 2024). Integrating
these methods into the generation process incurs additional

(a) Steering at generation to create a unique signature

Steering vector **v** 1

Steering vector **v** 2

(b) Extracting activations from the text and retrieving the steering vector
Input:
Text _t_ 2 inference

Activations overhead, and potential trade-offs between detection robustness and output quality have limited their widespread adoption (Sadasivan et al., 2023). Alternatively, AI-GTD can
be approached post-generation using classifiers that exploit
statistical properties of generated text (Mitchell et al., 2023)
or classifiers trained on large labeled corpora. However,
these detection methods are sensitive to distribution shifts
across domains and do not naturally extend to distinguishing
among multiple LLMs.

While our approach targets a different set of trade-offs than
adversarially robust watermarking, it offers a fundamentally different lens on model attribution. Rather than relying on external watermarking mechanisms applied at the
output token level, we investigate whether a signature can
be embedded and recovered by exploiting the structure of
the model’s internal representations. Specifically, we ask
whether models naturally encode a fingerprint, how such
signals can be detected, and whether an additional, easily
recoverable signature can be embedded alongside them.

Recent work in interpretability has revealed that LLMs can
indeed recognize their own generated outputs with nontrivial accuracy (Ackerman & Panickssery, 2025; Bowman
et al., 2024). This capability, termed _self-recognition_, suggests that models implicitly encode model-specific information in their generations. At the same time, activation
engineering has demonstrated that targeted interventions in
internal activations can steer LLM behavior while largely
preserving output quality (Panickssery et al., 2023; Liu et al.,
2023). Together, these findings suggest that internal activations offer a principled mechanism for detecting and
inducing model-specific generation signatures.

Building on these advances, we propose to use the internal
representations of an LLM as the space in which authorship
signals are detected. Furthermore, we show that inference-time steering can be used to deliberately inject distinct and
recoverable signatures, enabling more reliable detection
and supporting multi-model attribution without sacrificing
generation quality. Our approach enables a flexible detection framework, covering a range of scenarios, from postgeneration AI-GTD to the identification of multiple versions
of the same model through steering. We release code and
data to reproduce our experiments. [1] Our contributions in
this work are threefold:

1. We empirically demonstrate _self-recognition_ : even
with very short texts in relatively low-entropy scenarios, LLMs can reliably distinguish their own output
from human-written content.

2. We introduce a steering-based watermarking technique
that injects a distinct, recoverable signature, enabling

[1https://github.com/Thibaud-Ardoin/](https://github.com/Thibaud-Ardoin/LLM-Self-Recognition.git)
[LLM-Self-Recognition.git](https://github.com/Thibaud-Ardoin/LLM-Self-Recognition.git)

accurate attribution across multiple identical LLM instances steered in different directions, without compromising generation quality.

3. We offer an analysis that sheds light on the capacity
of internal representations to encode and retrieve a
random signal.

## 2. Methods

### 2.1. Problem Setup

We consider _white-box_ detection and attribution of LLM-generated text using internal activations of a target model
_M_ . We distinguish two setups:

**(i) AI-generated text self-recognition** : Given a text _t_, decide whether it was generated by _M_ or written by a human.

**(ii) Multi-model attribution** : Decide which _steered_ variant
of the same base model _M_ produced a given text _t_ .

**Threat** **model** : At detection time, we assume white-box
access to the internal activations of _M_ when processing a
given text _t_ . We consider both a _prompt-conditioned_ setting, in which the prompt is available to the detector, and a
_prompt-agnostic_ setting, in which only the generated text _t_
is observed.

### 2.2. Activation Extraction

Given a model _M_ with _L_ layers, let _l_ _∈{_ 1 _, . . ., L}_ denote a layer and **x** = [ _x_ 0 _, x_ 1 _, . . ., xn−_ 1] denote the token
sequence used for activation extraction (prompt + completion, or completion only in the prompt-agnostic setting). Let
**A** _l_ ( _xi_ ) _∈_ R _[d]_ be the intermediate activation at layer _l_ downstream from token _xi_ . By extension, for a text _t_, we describe
the activation sequence as **A** _l_ ( _t_ ) _∈_ R _[n][×][d]_ . We extract **A** _l_ ( _·_ )
at a fixed, consistent location within each transformer block.
The extraction layer _l_ is optimized independently for each
model and corresponds to an intermediate layer near the
middle of the network.

### 2.3. Steered LLM Generation

To embed a reliably detectable signature in the generated
text, we steer the model during generation. Given a directive
vector _**v**_ _∈_ R _[d]_ and an amplification coefficient _α_ _>_ 0, we
add the scaled vector to the intermediate activation at layer
_l_ at each token generation step:

**A** _l_ ( _xi_ ) _←_ **A** _l_ ( _xi_ ) + _α_ _**v**_ _._

Intuitively, this intervention nudges the internal representation trajectory in a consistent direction, leading to a text
_t_ _**v**_ influenced by _**v**_ . At detection time, this influence can be
retrieved in the activations **A** _l_ ( _t_ _**v**_ ) of _M_ .

For simplicity, steering and activation extraction use the
same layer _l_ . The steering direction _**v**_ is chosen randomly
to create a unique watermark in the multi-model attribution
setting. While prior work has largely focused on dense
interventions that modify all hidden dimensions (Li et al.,
2023), recent studies suggest that sparse steering vectors
induce fewer undesired perturbations and lead to more stable model behavior. Such sparsity can be achieved either
by leveraging sparse autoencoders (Bayat et al., 2025) or by
directly enforcing sparsity in the activation space (Ardoin
et al., 2025). We adopt the latter approach and construct
steering vectors with only a small fraction of non-zero dimensions. We empirically evaluate this design choice in
Section 3.4.

### 2.4. Attribution

To highlight the intrinsic self-recognition capability of
LLMs, we adopt a minimal linear probe operating on averaged internal activations. Multi-model attribution, however,
poses a more challenging classification problem and therefore needs a more expressive probing approach.

2.4.1. LINEAR CLASSIFICATION FOR
SELF-RECOGNITION

To obtain a fixed-dimensional representation of a text _t_
that is independent of its token length _n_, we aggregate the
token-level activations extracted in Section 2.2 by simple
averaging:

**r** = [1]

_n_

_n−_ 1

_i_ =0

**A** _l_ ( _xi_ ) _∈_ R _[d]_ (1)

overlap across splits. We then train a multi-layer perceptron (MLP), which provides greater expressive capacity than
LDA. The MLP has two hidden layers of width 32, trained
for a single epoch to predict the steering index _k_ of a given
extracted activation. As detection operates at the _token level_,
it enables attribution on arbitrary-length texts. However,
tokens later in a sequence encode broader context due to
causal attention over preceding tokens. To obtain _text-level_
predictions, we aggregate token-level classifications by majority voting across the entire generated sequence.

## 3. Experiments

Our experiments aim to assess the effectiveness of using
internal activation patterns for text authorship classification and to validate the proposed steering-based watermarking method. We primarily conduct our experiments using
Llama-3.1-8B, chosen for its wide availability and favorable performance-to-size trade-off. To assess the generality of our approach across model families, we additionally evaluate Ministral-3-8B, a state-of-the-art mixtureof-experts (MoE) model in the 8B-parameter range. To
study scalability, we further include experiments on the
smaller Llama-3.2-1B and Llama-3.2-3B models. For
question-answering benchmarks, we use the instruction-tuned variants of all models.

To evaluate robustness across a range of generation regimes,
we employ diverse datasets ranging from low-entropy constrained outputs to high-entropy open-ended completions.
For low-entropy tasks designed to demonstrate the self-recognition capability of LLMs, we use the XL-Sum summarization dataset (Hasan et al., 2021). We restrict our
experiments to the English subset of the human-authored
BBC news articles and summaries. For long-form question
answering, we use the ELI5 dataset (Guo et al., 2023), consisting of open-ended questions extracted from the _Explain_
_Like I’m Five_ subreddit. Finally, to evaluate performance
under higher-entropy generation, we curate a _Fresh News_
dataset from news articles published after the training cutoff
of the evaluated models.

### 3.1. Self-Recognition of Unaltered LLM-Generated Text

To evaluate the capability of LLMs to distinguish texts
they generated themselves from human-authored ones, we
prompt the LLM to generate short summaries of given news
articles and compare them against the corresponding human-authored summaries. This setting enables tight control of
confounds, particularly regarding text length. Furthermore,
summarizing news articles in only 1–2 sentences is inherently low-entropy, as the summary must cover the article’s
key information, leaving little stylistic freedom. Authorship attribution in this setting is therefore more challenging

We use a lightweight linear discriminant analysis (LDA)
classifier to distinguish _human_ from _LLM-generated_ text
based on the extracted, standardized representations **r** . LDA
gives an affine decision function, which we use to score texts
and classify them using a threshold _τ_ selected to satisfy
a target false-positive rate. In high-dimensional regimes,
the empirical covariance estimate of the LDA may be illconditioned. We therefore regularize the covariance with
the Ledoit–Wolf shrinkage estimator (Ledoit & Wolf, 2004).
In this setting, we use an 80/20 train/test split.

2.4.2. MULTI-MODEL ATTRIBUTION

To learn to distinguish among _K_ different steering vectors
( _**v**_ _k_ ) _k∈_ [1 _,K_ ], we construct a labeled dataset of activations.

Let _M_ _**v**_ _k_ ( _p_ ) = _t_ _**v**_ _k,p_ denote the text generated by model _M_
steered with vector _**v**_ _k_ in response to prompt _p ∈P_ :

_D_ = _{_ ( **A** _l_ ( _t_ _**v**_ _k,p_ ) _,_ _k_ ) _| k_ _∈{_ 1 _, . . ., K},_ _p ∈P} ._

We divide the dataset _D_ with a 70/10/20 train/validation/test
split by partitioning the prompt set _P_, ensuring no prompt

_Table_ _1._ Self-recognition AUROC scores (%) comparing our activation-based classifier (layer index in parentheses) against a
perplexity baseline on the XL-Sum dataset. Both methods achieve
near-perfect performance when conditioned on the original prompt,
but only our method generalizes to the prompt-agnostic setting.
Higher is better.

**With prompt** **No prompt**

**Model** Ours PPL Ours PPL

Ministral-3-8B 100 99.71 99.99 32.33
Llama-3.1-8B 99.99 99.19 99.16 47.86
Llama-3.2-3B 99.96 99.43 99.03 47.49
Llama-3.2-1B 99.82 97.07 98.58 52.27 than in conventional free-form text-generation tasks, where
AI-GTD is typically evaluated.

In this experiment, we evaluate self-recognition using 8 192
news articles and reference summaries from the English
subset of the XL-Sum dataset (Hasan et al., 2021). We
consider only articles at most 2 048 characters long to reduce
arbitrariness in the summaries and to allow LLMs to produce
high-quality summaries for each article.

As a baseline, we use each LLM’s perplexity (Jelinek et al.,
2005) as an intrinsic measure of its predictive performance.
For a token sequence _x_ 1: _T_, we compute log _pM_ ( _xt_ _| x<t_ )

PPL( _x_ ) = exp

_−_ [1]

_T_

_T_

_t_ =1

_,_

where _pM_ denotes the conditional probability distribution
defined by model _M_ . Lower perplexity indicates that the
model assigns higher likelihood to the text. We perform
authorship attribution by scoring each summary with its perplexity and sweeping a decision threshold over the full range
of scores to compute the AUROC, expecting the generating
model to assign lower perplexity to its own summaries than
to human-authored ones. While perplexity is commonly
used in AI-GTD, it is a limited signal and is often augmented
with additional statistics or computed in a contrastive fashion. For example, Hans et al. (2024) propose a zero-shot
detector that contrasts two closely related LLMs by combining perplexity with a _cross-perplexity_ score, which helps
calibrate prompt- and topic-induced variation. We nevertheless report perplexity because it is computable from a
single LLM without auxiliary models or features, making it
a natural baseline for self-recognition.

Table 1 presents self-recognition capabilities across LLMs in
this scenario. Our findings indicate that internal activations
exhibit a strong, linearly separable signal for distinguishing LLM-generated from human-authored summaries, even
for very short completions. Classifiers trained on these
activations consistently outperform the perplexity-based

Table 2: F1 scores of the attribution task on ELI5 and Fresh News

datasets across multiple LLMs. Results are averaged over five runs
with different random seeds and steering vectors. The task measures the ability to classify texts generated by two independently
steered model variants. Higher is better.

**ELI5** **Fresh News**

**Model** Token Text Token Text

Ministral-3-8B 99.3 100 94.6 100
Llama-3.1-8B 94.0 99.1 90.5 99.1
Llama-3.2-3B 83.3 95.5 75.5 88.3
Llama-3.2-1B 72.0 85.3 69.5 83.8 baseline, though the baseline performs only marginally
worse when prompts are available. Crucially, the activation-based signal remains robust in the prompt-agnostic setting, with AUROC decreasing by at most one point. By
contrast, for Llama-family models, perplexity-based baselines do not consistently perform above chance without
prompt access. Notably, for Ministral-3-8B-Instruct
in this setting, the perplexity ordering is largely inverted,
as human-authored summaries tend to attain _lower_ perplexity than the model’s own generations. This behavior may reflect data overlap between the evaluation corpus and the pre-training or instruction-tuning data used for
Ministral-3-8B-Instruct.

Additional experiments described in Appendix E indicate
that layer-wise LDA performance remains largely consistent beyond the first layer, suggesting that self-recognition
signals are distributed throughout the network. Further, classifiers trained on specific news domains generalize well
to others, indicating that the learned signal is not strongly
confounded by semantics.

### 3.2. Distinguishing Multiple LLMs with Steering

To validate the feasibility of the activation-based watermark
described in Section 2.4.2, we conduct a controlled attribution experiment. We steer two identical model instances
using two distinct random sparse vectors to generate completions for a fixed set of 1 000 prompts, with a maximum
sequence length of 512 tokens. A classifier is then trained
on the collected activation data to distinguish the source
identity of the generated text.

To isolate the intrinsic detectability of the steering signal
from extensive hyperparameter optimization, we fix the
amplification coefficient to _α_ = 5 and the vector sparsity
to 99 _._ 7% across all model families. While these parameters may not be optimal for every specific architecture,
fixing them enables a direct comparative assessment of
signal recovery across scales. Table 2 presents the attribution accuracy, revealing a clear trend: detection perfor

1.0

0.8

0.6

0.4

0.2

0.0

|Col1|Col2|Col3|Col4|Col5|Col6|Text-level Random Baselin|e|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||

2 3 5 7 10 15 20
Number of Classes

1.0

0.8

0.6

0.4

0.2

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|L3 - News|Col17|Col18|Col19|Col20|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||| L3 - E5 M3 - News M3 - E5| L3 - E5 M3 - News M3 - E5| L3 - E5 M3 - News M3 - E5| L3 - E5 M3 - News M3 - E5| L3 - E5 M3 - News M3 - E5|
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||

_Figure_ _2._ Performance of our attribution method across differ
ent numbers of classes. We compare text-level attribution with
majority voting, token-level attribution, and a random-attribution
baseline. F1 scores are shown for Llama-3.1-8B on our Fresh
News dataset.

mance improves with model size. Interestingly, the question-answering dataset ELI5 yields higher separability than open-ended text generation tasks. This finding is counterintuitive,
as the lower entropy of factual generation theoretically offers a more constrained manifold for embedding a watermark. This performance gap could therefore stem from the
instruction fine-tuning of the underlying models.

We further evaluate the method’s scalability by increasing
the pool of distinct steered identities. As shown in Figure 2,
while the discrimination task becomes harder as the number
of classes grows, performance remains significantly above
the random baseline. While the current classification over
_N_ vectors shows limitations in scaling, these results imply
the potential for a combinatorial multi-bit watermark, where
multiple sparse vectors are superimposed for steering and
detected independently. This would theoretically allow the
encoding of 2 _[N]_ identities.

Regarding generation quality, perplexity in this setting acts
primarily as a measure of distributional shift from the base
model rather than a direct proxy for text quality. To ensure
that steering does not degrade generation quality, we calibrate the steering parameters using an external evaluator.
We employ the quality-classifier-deberta model, a
fine-tuned DeBERTa-v3 (He et al., 2023) from the NVIDIA
NeMo Curator toolkit. In addition, we evaluate the steered
models on the MMLU benchmark (Hendrycks et al., 2020)
to verify minimal performance degradation. A comprehensive analysis of the quality trade-off is provided in Appendix F.

### 3.3. Robustness Study

AI-GTD often lacks robustness to modifications of the
generated text, which can erase the watermark or the

0.0
Original Paraphrased Original Paraphrased

Our Method KGW Watermarking

_Figure_ _3._ F1 scores for AI-GTD, comparing our method with traditional watermarking on the original LLM-generated text and
a paraphrased version. Scores are computed with Llama-3.1-8B
(L3) and Ministral-3-8B (M3) on the Fresh News (News) and
ELI5 (E5) datasets.

LLM’s fingerprint. Automatic paraphrasing tools, such as
DIPPER-XXL (Krishna et al., 2023), have been designed to
evaluate this robustness and are used here. DIPPER-XXL reformulates text at the level of multiple sentences, rather than
relying only on word substitution or sentence-level rewriting.
Figure 3 reports F1 scores for distinguishing human-written
from watermarked text, with and without paraphrasing, and
compares our method to the green-list approach of the KGW
watermark (Kirchenbauer et al., 2023). Our method shows
greater robustness to paraphrasing, particularly in the free-form generation setting, where performance remains largely
stable. One possible explanation is that our approach operates in a higher-level representation space, where attribution
signals are aggregated across many tokens, potentially leading to more robust detection.

Here, we paraphrase only the LLM-generated text to wash
out the watermark or steering signal. If we also paraphrase
the human text, the F1 score of our method is slightly reduced, as paraphrasing human text acts as a spoofing mechanism for our detector. More details are given in Appendix C.

### 3.4. Influence of Steering-Vector Sparsity

Excessive steering intervention introduces significant noise
into the generation process, often manifesting as repetitive
loops or incoherent text. While these distinct artifacts make
the attribution task trivial, they render the watermark unusable for practical applications where text quality is essential.
Consequently, the optimal steering configuration must maximize detectability while minimizing quality loss.

To demonstrate the empirical advantages of sparse steering vectors over dense ones, we evaluate the trade-off they
offer between attribution performance and generation qual

1.0

0.9

0.8

0.7

0.6

0.5

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|||||||||||
|||||||||||
||~~99~~|~~7% Spa~~|~~ rse~~|||||||
|| De| nse||||||||

0.500 0.525 0.550 0.575 0.600 0.625 0.650 0.675 0.700
Quality metric

**LLM Self-Recognition by Steering and Retrieving Activation Signatures**

Table 3: Accuracy for source attribution at the token and text levels,
with and without paraphrasing. We compare simple cosine similarity attribution with the trained MLP approach under identical
evaluation conditions. Higher is better. The best score for each
method is underlined.

**Paraphrased**

**Text**

**Detection**

**Detection** **Original**

**Granularity**
**Method** **Text**

**Text**

Figure 4: Trade-off between generated text quality and accuracy

in distinguishing vanilla from steered generations of Llama-3-8B.
The figure compares the effect of a 99 _._ 7% sparse steering vector
with a dense counterpart. Points are obtained by varying the steering coefficient and are averaged over five random seeds, with all
other settings held fixed.

ity. Since the multi-model attribution task described in
Table 2 yields saturated accuracy for large models, we adopt
a more challenging binary classification setting to better
differentiate the steering types: distinguishing the unsteered
Llama-3-8B base model from a steered counterpart. We
perform a sweep over the amplification factor _α_, averaging the performance across five distinct steering vectors
per configuration. The quality metric is computed using
the quality-classifier-deberta model. The results in
Figure 4 indicate that highly sparse vectors achieve a favorable trade-off profile compared to their dense counterparts,
despite modifying only a fraction of the dimensions. This
suggests that broad interventions are unnecessary, highlighting that targeted activation engineering offers a more robust
approach to model steering.

### 3.5. Direct Signal Recovery via Cosine Similarity

Unlike standard activation engineering, which identifies specific directions to inhibit or amplify behavioral traits, such
as sentiment or hallucination, our method operates on a fundamentally different principle. We inject a random, sparse
steering vector. In the high-dimensional geometry of LLM
activations, random vectors are, with high probability, approximately orthogonal to the semantic manifolds governing
standard model behaviors, a property of concentration of
measure (Vershynin, 2026). Therefore, we hypothesize that
this orthogonality allows the steering signal to coexist with
semantic content without significant interference.

Remarkably, this steering signal persists through the discretization step of token sampling and re-embedding. When
the generated text is fed back into a non-steered base model,
the steering vector can be directly retrieved from the resulting activations, without requiring a trained classifier. We
observe a significant cosine similarity between the original steering vector and the activations of the base model
processing the watermarked text. This implies that the trans

Similarity Token-level 62.9 58.6
Similarity Text-level 84.6 77.8

MLP Token-level 90.5 80.0
MLP Text-level 99.1 89.3 formation from activation space to discrete tokens and back
to activation space preserves the directional alignment of the
injected signal. As illustrated in Figure 5, this alignment remains robust even after strong paraphrasing via DIPPER-XXL.
The figure reports the average cosine similarity across all
tokens in sequences up to 512 tokens.

This persistence enables a training-free detection mechanism. Instead of training a supervised MLP, we can attribute
a text to a specific steered model by computing the cosine
similarity between the collected activations and candidate
steering vectors. This yields an attribution score for each
token and, via majority voting, a global attribution decision.
Table 3 presents the accuracy of this zero-shot approach
for distinguishing between two random vectors. While it
does not reach the near-perfect performance of the trained
MLP, the accuracy is surprisingly high for a simple geometric metric. This confirms that the sparse signal is not
merely a statistical artifact but is physically encoded into
the generated sequence.

|×10 3|Col2|Col3|
|---|---|---|
|Similarity tov1 |Similarity tov1 |Similarity tov1 |
|Similarity tov1 |Similarity tov1 |Similarity tov1 |
||~~Similarity tov~~2||
||||
||||
||||
||||
||||
||||

Original Paraphrased Original Paraphrased

Text generated with v1 Text generated with v2 representations of induced text. Averaged cosine similarity between a pair of randomly sampled steering vectors ( _**v**_ 1 and _**v**_ 2)
and activation representations extracted from text generated by the
model when steered along these directions.

Figure 5: Example of alignment between steering directions and

## 4. Discussion and Future Work

**Choice** **of** **Steering** **Vectors.** An analysis of individual
steering directions reveals substantial variability in their
effects. Some vectors induce more severe degradations in
generation quality, while others are more easily detectable.
Steering directions also differ in their impact on the representation space at detection time, as measured by cosine
similarity in Figure 5. We hypothesize that these differences
arise from varying degrees of orthogonality to the model’s
natural activation trajectory in a given context. This heterogeneity suggests that selecting or optimizing subsets of
steering vectors that balance detectability and generation
quality could further improve performance.

**Multi-bit Watermarking.** As proposed in Section 3.2, superimposing multiple sparse vectors with disjoint supports
theoretically enables a high-capacity, multi-bit watermark.
In such a scheme, the presence or absence of a specific
steering vector encodes a single bit of the unique identifier. Achieving this requires further research to verify that
the linear superposition of steering signals remains recoverable within the model’s activation space without destructive
interference.

**Interpretability of Random Steering.** To the best of our
knowledge, utilizing sparse random vectors is a novel approach in activation engineering, specifically for watermarking. While our metrics indicate negligible degradation in
quality, the semantic nuances induced by these perturbations
remain unclear. We could not identify obvious artifacts in
the text output, yet a subtle shift in distribution is inevitable
to allow detection. Future work must rigorously analyze
these impacts, potentially leveraging automated annotation
by strong judge models or interpreting the steered features
via sparse autoencoders for monosemantic feature discovery (Templeton et al.). Such interpretability is a prerequisite
for deployment, ensuring that the watermark does not inadvertently introduce safety risks or biases while helping to
design more semantically neutral steering vectors.

Cross-evaluation results displayed in Appendix G reveal
a hard architectural boundary: detection remains strong
within the same family but degrades to chance level when
text is generated by another architecture, regardless of
dataset. This mirrors observations of the _subliminal learn-_
_ing_ phenomenon reported by Cloud et al. (2025), where
implicit behavioral traits in teacher-generated data transfer to a student only when both models share the same
base architecture. We interpret both findings as evidence of
architecture-specific activation structure: steering vectors
leave activation traces that are invisible at the semantic level
but geometrically coherent within a model family, and do
not generalize beyond it.

**Robustness and Secrecy.** While we demonstrated robustness against strong paraphrasing, the watermark’s resilience
to broader attacks, such as recursive translation, adversarial
perturbations, text degradation, or format changes, remains
to be characterized. Furthermore, the security of the method
relies on the secrecy of the steering configuration. Specifically, if an attacker can recover the secret steering vector
and target layer, the watermark could be trivially reproduce
to spoof the identity of the steered LLM. To mitigate this
risk, future work could explore dynamic steering strategies
in which the vector rotates or evolves during generation
according to a pseudo-random schedule. Such a moving
target would significantly complicate the recovery of the
watermark signal without the corresponding seed.

## 5. Limitations

**Detection Overhead.** Unlike inference-time watermarks,
signal injection incurs no computational cost in our method.
However, detection requires white-box access to the model
and a forward pass to extract activations from the text, which
incurs a non-negligible computational cost. In our setup
using an RTX A5500, collecting the model’s activations
takes time equivalent to 21% of the time needed to autoregressively generate the same text.

**Quality** **Evaluation** **Limitations.** To obtain a more accurate evaluation of textual quality than perplexity, we relied on an externally fine-tuned quality assessment model.
However, this evaluator exhibits biases toward structural
formatting, such as penalizing markdown syntax regardless of content quality. Theoretically, degraded texts may
still be assigned high scores, potentially obscuring small
performance losses in steered outputs. Despite these limitations, we maintain that measuring the _relative_ quality
difference between steered and unsteered outputs provides a
valid proxy for steering-induced degradation, as formatting
and other confounders remain constant across conditions.

## 6. Related Work

Recent work on mechanistic interpretability shows that
transformer activations encode linearly recoverable signals
for high-level concepts. In the context of truthfulness, simple probes of hidden states can predict the veracity of a
statement even when the model’s final answer is incorrect,
suggesting concept directions that are partially disentangled from generation (Marks & Tegmark, 2023; Azaria &
Mitchell, 2023).

Several approaches use internal representations for detecting
AI-generated text. RepreGuard (Chen et al., 2025) separates
human and model text using observer-model hidden states,
while Yu et al. (2024) train supervised classifiers on acti vations from an auxiliary encoder. These works primarily
target robust, prompt-agnostic detection of arbitrary LLM-generated text, rather than self-recognition. SAEMark (Yu
et al., 2025) uses SAE features to guide a rejection-sampling
process for watermarking, achieving robust detection at the
cost of significant computational overhead during generation.

Closest to our setting, Ackerman & Panickssery (2025)
study whether instruction-tuned models can recognize their
own outputs and extract residual-stream directions via
contrastive pairs, but report performance that does not
consistently beat perplexity-based heuristics. Separately,
Kuznetsov et al. (2025) show that SAE-based features can
improve general-purpose AI-GTD. They use feature steering to analyze the nature of the features responsible for
detection and causally link the intervention to detectability.

**Conclusion**

In this work, we investigated the self-recognition capabilities of LLMs and introduced a random steering mechanism to enable the attribution of generated text. Our results
show that LLMs exhibit strong self-recognition capabilities,
encoded as linearly separable signals in their internal activations. These signals are distributed across the network,
robust to semantic shifts, and enable attribution without
relying on perplexity.

Furthermore, a simple inference-time intervention enables
robust source attribution while incurring minimal degradation in text quality. Importantly, we find that sparse interventions in the activation space lead to lower quality
loss than dense noise injection for comparable detectability. Moreover, by operating directly on model activations,
our approach avoids reliance on computationally intensive
sparse autoencoder pipelines while still capturing high-level
properties of the inference process.

Future work may extend this analysis to the design of a
seamless, resource-efficient watermarking system, as well as
to systematic comparisons with existing approaches. More
broadly, we hope this work encourages further investigation
into the inference-time dynamics of LLMs and into how
internal representations can be leveraged to ensure reliable
and transparent model behavior.

**Impact Statement**

This paper examines whether LLMs encode a reliable self-recognition signal in their internal activations and leverages
it to enable watermarking and attribution at the level of a
model or a steered variant. Such capabilities could support
provenance, auditing, and accountability for AI-generated
text, but they also amplify known risks. In particular, water marking for attribution can raise privacy concerns if signatures directly link to specific deployments or users.

We identify no substantive risks associated with the publication of our method, and our contribution is purely methodological. The threat models we evaluate are standard in the
watermarking literature and can be executed using publicly
available tools. Consequently, disclosing our method does
not introduce any new adversarial capabilities beyond those
already well known in existing watermarking frameworks.

**Author Contributions**

TA and JS both contributed significantly to this work. TA
led the steering experiments and writing, JS led the self-recognition experiments, under the general direction of GW.

**Acknowledgements**

TA, JS and GW were supported by the Federal Ministry of
Education and Research of Germany (BMBF) in the programme of “Souverän. Digital. Vernetzt.”, joint project
“AIgenCY : Chances und Risks of Generative AI in Cyber
security”, project identification number 16KIS2013. GW
was also supported by BMBF joint project “6G- RIC: 6G
Research and Innovation Cluster”, project identification
number 16KISK025.

**References**

Ackerman, C. and Panickssery, N. Inspection and Control of Self-Generated-Text Recognition Ability in Llama38b-Instruct, April 2025.

Ardoin, T., Cai, Y., and Wunder, G. Where confabulation lives: Latent feature discovery in LLMs. In
Christodoulopoulos, C., Chakraborty, T., Rose, C., and
Peng, V. (eds.), _Proceedings_ _of_ _the_ _2025_ _Conference_
_on Empirical Methods in Natural Language Processing_,
pp. 29801–29825, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-889176-332-6. doi: 10.18653/v1/2025.emnlp-main.1515.
[URL https://aclanthology.org/2025.emnlp-main.](https://aclanthology.org/2025.emnlp-main.1515/)
[1515/.](https://aclanthology.org/2025.emnlp-main.1515/)

Azaria, A. and Mitchell, T. The internal state of an llm knows when it’s lying. _arXiv preprint arXiv:2304.13734_,
2023. doi: 10.48550/ARXIV.2304.13734. [URL https:](https://arxiv.org/abs/2304.13734)
[//arxiv.org/abs/2304.13734.](https://arxiv.org/abs/2304.13734)

Bayat, R., Rahimi-Kalahroudi, A., Pezeshki, M., Chandar,

S., and Vincent, P. Steering large language model activations in sparse spaces, 2025. [URL https://arxiv.org/](https://arxiv.org/abs/2503.00177)
[abs/2503.00177.](https://arxiv.org/abs/2503.00177)

Bjelobaba, S., Waddington, L., Perkins, M., Foltynek,` T.,

Bhattacharyya, S., and Weber-Wulff, D. Maintaining
research integrity in the age of genai: an analysis of
ethical challenges and recommendations to researchers.
_International_ _Journal_ _for_ _Educational_ _Integrity_, 21(1):
1–23, 2025.

Bowman, S., Feng, S., and Panickssery, A. Llm evalua
tors recognize and favor their own generations. In _Ad-_
_vances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _37_,
NeurIPS 2024, pp. 68772–68802. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024. doi:
10.52202/079017-2197. [URL http://dx.doi.org/10.](http://dx.doi.org/10.52202/079017-2197)
[52202/079017-2197.](http://dx.doi.org/10.52202/079017-2197)

Chen, X., Wu, J., Yang, S., Zhan, R., Wu, Z., Luo, Z., Wang,

D., Yang, M., Chao, L. S., and Wong, D. F. RepreGuard:
Detecting LLM-generated text by revealing hidden representation patterns, August 2025.

Christ, M., Gunn, S., and Zamir, O. Undetectable water
marks for language models. In _The Thirty Seventh Annual_
_Conference on Learning Theory_, pp. 1125–1139. PMLR,
2024.

Cloud, A., Le, M., Chua, J., Betley, J., Sztyber-Betley, A.,

Hilton, J., Marks, S., and Evans, O. Subliminal learning:
Language models transmit behavioral traits via hidden
signals in data. _arXiv preprint arXiv:2507.14805_, 2025.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi,

A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li,
H., McDonell, K., Muennighoff, N., Ociepa, C., Phang,
J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika,
L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A.
The language model evaluation harness, 07 2024. URL
[https://zenodo.org/records/12608602.](https://zenodo.org/records/12608602)

Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding,

Y., Yue, J., and Wu, Y. How close is chatgpt to human experts? comparison corpus, evaluation, and detection,
2023. [URL https://arxiv.org/abs/2301.07597.](https://arxiv.org/abs/2301.07597)

Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi,

H., Saha, A., Goldblum, M., Geiping, J., and Goldstein,
T. Spotting llms with binoculars: zero-shot detection of
machine-generated text. In _Proceedings of the 41st In-_
_ternational Conference on Machine Learning_, ICML’24.
JMLR.org, 2024.

Hasan, T., Bhattacharjee, A., Islam, M. S., Mubasshir, K., Li,

Y.-F., Kang, Y.-B., Rahman, M. S., and Shahriyar, R. XL
sum: Large-scale multilingual abstractive summarization
for 44 languages. In Zong, C., Xia, F., Li, W., and Navigli,
R. (eds.), _Findings of the Association for Computational_
_Linguistics:_ _ACL-IJCNLP 2021_, pp. 4693–4703, Online,
August 2021. Association for Computational Linguistics.

doi: 10.18653/v1/2021.findings-acl.413. URL [https:](https://aclanthology.org/2021.findings-acl.413/)
[//aclanthology.org/2021.findings-acl.413/.](https://aclanthology.org/2021.findings-acl.413/)

He, P., Gao, J., and Chen, W. Debertav3: Improving
deberta using electra-style pre-training with gradientdisentangled embedding sharing, 2023. URL [https:](https://arxiv.org/abs/2111.09543)
[//arxiv.org/abs/2111.09543.](https://arxiv.org/abs/2111.09543)

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika,

M., Song, D., and Steinhardt, J. Measuring massive
multitask language understanding, 2020. URL [https:](https://arxiv.org/abs/2009.03300)
[//arxiv.org/abs/2009.03300.](https://arxiv.org/abs/2009.03300)

Hou, A. B., Zhang, J., He, T., Wang, Y., Chuang, Y.-S.,

Wang, H., Shen, L., Van Durme, B., Khashabi, D., and
Tsvetkov, Y. Semstamp: A semantic watermark with
paraphrastic robustness for text generation, 2023. URL
[https://arxiv.org/abs/2310.03991.](https://arxiv.org/abs/2310.03991)

Jelinek, F., Mercer, R. L., Bahl, L. R., and Baker, J. K.

Perplexity–a measure of the difficulty of speech recognition tasks. _The_ _Journal_ _of_ _the_ _Acoustical_ _Society_ _of_
_America_, 62(S1):S63–s63, 08 2005. ISSN 0001-4966.

doi: 10.1121/1.2016299. URL [https://doi.org/10.](https://doi.org/10.1121/1.2016299)
[1121/1.2016299.](https://doi.org/10.1121/1.2016299)

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., and Goldstein, T. A watermark for large language models.
In _International_ _Conference_ _on_ _Machine_ _Learning_, pp.
17061–17084. PMLR, 2023.

Krishna, K., Song, Y., Karpinska, M., Wieting, J., and Iyyer,

M. Paraphrasing evades detectors of ai-generated text,
but retrieval is an effective defense, 2023. [URL https:](https://arxiv.org/abs/2303.13408)
[//arxiv.org/abs/2303.13408.](https://arxiv.org/abs/2303.13408)

Kuznetsov, K., Kushnareva, L., Druzhinina, P., Razzhigaev,

A., Voznyuk, A., Piontkovskaya, I., Burnaev, E., and
Barannikov, S. Feature-level insights into artificial text
detection with sparse autoencoders, 2025. [URL https:](https://arxiv.org/abs/2503.03601)
[//arxiv.org/abs/2503.03601.](https://arxiv.org/abs/2503.03601)

Ledoit, O. and Wolf, M. A well-conditioned estimator for large-dimensional covariance matrices. _Journal of Multi-_
_variate Analysis_, 88(2):365–411, February 2004. ISSN
0047-259X. doi: 10.1016/S0047-259X(03)00096-4.

Li, K., Patel, O., Viégas, F., Pfister, H., and Wattenberg, M.

Inference-time intervention: Eliciting truthful answers
from a language model. _arXiv preprint arXiv:2306.03341_,
2023. doi: 10.48550/ARXIV.2306.03341. [URL https:](https://arxiv.org/abs/2306.03341)
[//arxiv.org/abs/2306.03341.](https://arxiv.org/abs/2306.03341)

Liu, S., Ye, H., Xing, L., and Zou, J. In-context vectors: Making in context learning more effective and controllable through latent space steering. _arXiv_ _preprint_
_arXiv:2311.06668_, 2023. doi: 10.48550/ARXIV.2311.
06668. [URL https://arxiv.org/abs/2311.06668.](https://arxiv.org/abs/2311.06668)

Marks, S. and Tegmark, M. The geometry of truth:
Emergent linear structure in large language model
representations of true/false datasets. _arXiv_ _preprint_
_arXiv:2310.06824_, 2023. doi: 10.48550/ARXIV.2310.
06824. [URL https://arxiv.org/abs/2310.06824.](https://arxiv.org/abs/2310.06824)

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., and Finn, C. DetectGPT: Zero-shot machine-generated
text detection using probability curvature. In _Interna-_
_tional conference on machine learning_, pp. 24950–24962.
PMLR, 2023.

Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger,

E., and Turner, A. M. Steering llama 2 via contrastive
activation addition. _arXiv_ _preprint_ _arXiv:2312.06681_,
2023. doi: 10.48550/ARXIV.2312.06681. [URL https:](https://arxiv.org/abs/2312.06681)
[//arxiv.org/abs/2312.06681.](https://arxiv.org/abs/2312.06681)

Sadasivan, V. S., Kumar, A., Balasubramanian, S., Wang,

W., and Feizi, S. Can ai-generated text be reliably detected? _arXiv preprint arXiv:2303.11156_, 2023.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J.,

Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen,
E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers,
T. R., Rees, E., Batson, J., Jermyn, A., Carter, S.,
Olah, C., and Henighan, T. Scaling monosemanticity: Extracting interpretable features from claude 3
sonnet. URL [https://transformer-circuits.pub/](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
[2024/scaling-monosemanticity/.](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

Vershynin, R. _High-Dimensional Probability:_ _An Introduc-_

_tion with Applications in Data Science_ . Cambridge Series
in Statistical and Probabilistic Mathematics. Cambridge
University Press, Cambridge, 2 edition, 2026.

Yang, P., Li, X., Ni, W., Yin, J., Wang, H., Nan, G., Wang,

S., Huang, Y., and Qi, T. Enhancing watermarking quality
for llms via contextual generation states awareness. _arXiv_
_preprint arXiv:2506.07403_, 2025.

Yousaf, M. N. Practical considerations and ethical implications of using artificial intelligence in writing scientific manuscripts. _ACG_ _Case_ _Reports_ _Journal_, 12
(2):e01629, February 2025. ISSN 2326-3253. doi:
10.14309/crj.0000000000001629. [URL http://dx.doi.](http://dx.doi.org/10.14309/crj.0000000000001629)
[org/10.14309/crj.0000000000001629.](http://dx.doi.org/10.14309/crj.0000000000001629)

Yu, X., Chen, K., Yang, Q., Zhang, W., and Yu, N. Text fluoroscopy: Detecting llm-generated text through intrinsic features. In _Proceedings_ _of_ _the_ _2024_ _Conference_
_on_ _Empirical_ _Methods_ _in_ _Natural_ _Language_ _Process-_
_ing_, pp. 15838–15846. Association for Computational
Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.
885. URL [http://dx.doi.org/10.18653/v1/2024.](http://dx.doi.org/10.18653/v1/2024.emnlp-main.885)
[emnlp-main.885.](http://dx.doi.org/10.18653/v1/2024.emnlp-main.885)

Yu, Z., Jiang, X., Gu, W., Wang, Y., Zhang, S., and Ye,

W. Saemark: Multi-bit llm watermarking with inference-time scaling. _arXiv_ _preprint_ _arXiv:2508.08211_, 2025.
doi: 10.48550/ARXIV.2508.08211. [URL https://doi.](https://doi.org/10.48550/arXiv.2508.08211)
[org/10.48550/arXiv.2508.08211.](https://doi.org/10.48550/arXiv.2508.08211)

1.0

0.8

0.6

0.4

0.2

0 100 200 300 400 500
Token Position

Figure 6: Test F1 score as a function of token position, with attribution tasks involving between 2 and 20 distinctly steered LLMs.

**A. Parameterization**

This section contains the parameters used to enable reproduction of our results. More details can be found in the code
[repository (https://github.com/Thibaud-Ardoin/LLM-Self-Recognition).](https://github.com/Thibaud-Ardoin/LLM-Self-Recognition)

**Datasets.** The shared repository also contains the text datasets used in our experiments. The custom-curated **Fresh News**
dataset was collected via the API of _The Guardian_ and consists of clean, plain-text news articles. To minimize potential
overlap with model training-data cutoff dates, all articles were published between November 1, 2025 and January 5, 2026.
As the dataset is designed for open-ended text completion, we retain only the first six words of each article as the initial
prompt provided to the LLM. The **ELI5** dataset was filtered to remove questions containing explicit mention of the word
“Edit”, which typically correspond to user-added clarifications appended after the initial question. Questions containing explicit “URL” mentions were also removed, as they often refer to external content that is not accessible to the LLM. For
the robustness experiments in Section 3.3, the human reference answer is consistently chosen as the first response listed in
the dataset.

**Models.** Texts are generated with nucleus sampling ( _p_ = 0 _._ 9) and temperature 0 _._ 7 for all LLMs and datasets, using the
system prompt and query template specified in Appendix D. In the steering scenarios, a repetition penalty is set to 1 _._ 1.

**Steering.** The random sparse steering vectors are sampled from _U_ ([ _−_ 1 _,_ 1] _[d]_ ), with 99 _._ 7% of dimensions set to zero.
The scaling coefficient _α_ = 5 and the sparsity ratio are selected based on preliminary experiments to balance detection
performance and generation quality on Llama-3.1-8B-Instruct. Injection and detection are performed at the middle layer,
which empirically yields robust performance across models.

**B. Token-wise Analysis**

Figure 6 illustrates how the test-time performance varies as a function of token position in the generated sequence. The
experimental setup is the MLP classification described in Section 2.3. The earliest tokens are the most difficult to classify,
with accuracy close to random chance _N_ [1] [when distinguishing among] _[ N]_ [steered LLMs.] [In contrast, later tokens achieve]

substantially higher accuracy, approaching the performance obtained by majority voting. While performance could be
improved by excluding early tokens from the majority vote, this would break the sequence-length-agnostic nature of the
method.

Figure 7 illustrates the cosine similarity between generated text representations and the steering vectors used during
generation. With the vector active during steering, we observe an increasing trend similar to that seen in the test accuracy of
the MLP setup. For paraphrased texts, the magnitude of this effect is reduced, while the overall trend remains consistent. In
contrast, cosine similarity with steering vectors that were not involved in generating the text oscillates around zero or takes
negative values.

Text t1 Text t2

0.015

0.010

0.005

0.000

|×10 3 v1, Al(t1|1) cos|
|---|---|
|×10 3 1 l|cos ~~v1, Al(t1) cos~~|
|||
|||
|||

0 50 100 150 200 250
Token Position

|v1, Al(t1) cos (P|Paraphrased) v, A(t) (Paraphrased)|
|---|---|
||v1, Al(t1) cos (Paraphrased)|
|||
|||

0 50 100 150 200 250
Token Position

|×10 3 v2, Al(t1|1) cos|
|---|---|
|×10 3 2 l|cos v2, Al(t1) cos|
|||
|||
|||

0 50 100 150 200 250
Token Position

|×10 3 v2, Al(t1) cos (P|Paraphrased)|
|---|---|
|×10 3 2 l1 cos||
|||
|||
||_v2~~, A_l(t_1)~~ cos~~ (Paraphrased)~~|

0 50 100 150 200 250
Token Position

0 50 100 150 200 250
Token Position

|×10 3 v1, Al(t2) cos|(Paraphrased) v, A(t) (Paraphrased)|
|---|---|
||~~v1, Al(t2) cos (Paraphrased)~~|
|||
|||
|||
|||

0 50 100 150 200 250
Token Position

|×10 3 v2, Al|l(t2) cos|
|---|---|
|×10 3 2|2 cos v2, Al(t2) cos|
|||
|||
|||

0 50 100 150 200 250
Token Position

|×10 3 v2, Al(t2) cos|(Paraphrased)|
|---|---|
|×10 3 2 l2 cos ~~, () (Paraphrased)~~||
|~~v2 Alt2 cos~~||
|||
|||
|||

0 50 100 150 200 250
Token Position texts _t_ 1 and _t_ 2, generated with steering vectors **v** 1 and **v** 2, respectively. The paraphrased versions of the texts are also compared.

Figure 7: Variation in cosine similarity _⟨·, ·⟩_ cos as a function of token position. The plots compare the similarity between activations of

**C. Details of the Paraphrasing Experiments**

The paraphrasing model DIPPER-XXL (Krishna et al., 2023) is used with the parameters _Lexical diversity_ = 60 and
_Reordering_ = 20. These parameters are among the strongest setups proposed in the original paper.

Figure 8 presents the results of an alternative robustness experiment to the one in Section 3.3. Because paraphrasing only the
watermarked text introduces a distribution shift, we additionally evaluate a setting in which both human- and AI-generated
texts are paraphrased. While the KGW watermark is largely unaffected under this new setup, our method exhibits a reduction
in accuracy relative to the results shown in Figure 3. This suggests that AI-generated paraphrasing acts as a spoofing
mechanism for our representation-based attribution methods.

**D. Prompts**

Below are the system and user prompts used to evaluate the general self-recognition ability of LLMs (Section 3.1). The
prompts were designed to generate summaries that closely match the human-authored ones.

### D.1. System Prompt

You write very short summaries of news articles.
Rules:

- Output only one sentence whenever possible (but never more than two).

- Respect the requested word range.

- The final sentence must end with a single period.

- No preface, labels, quotes, bullet points, or line breaks.

- Use a neutral, journalistic tone.

- Focus on the main event and key entities only.

1.0

0.8

0.6

0.4

0.2

0.0

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|L3 - News L3 - E5 M3 - News|Col18|Col19|Col20|Col21|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||||
||||||||||||||||||||||
||||||||||||||||||||||
|||||||||||||||| M3 - E5| M3 - E5| M3 - E5| M3 - E5| M3 - E5| M3 - E5|
||||||||||||||||||||||
||||||||||||||||||||||
||||||||||||||||||||||
||||||||||||||||||||||
||||||||||||||||||||||
||||||||||||||||||||||
||||||||||||||||||||||
||||||||||||||||||||||

Original Paraphrased Original Paraphrased

Our Method KGW Watermarking

_Figure_ _8._ Accuracy for AI-GTD, comparing our method with traditional watermarking on the original LLM-generated text and a paraphrased version. Accuracies are computed with Llama-3.1-8B (L3) and Ministral-3-8B (M3) on the Fresh News (News) and ELI5
(E5) datasets.

Table 4: Layer-wise self-recognition performance summary. For each model and setting, we report the best- and worst-performing layer

indices, along with the AUROC spread (difference in percentage points).

**With prompt** **No prompt**

**Model** **Best L** **Worst L** ∆ **AUROC** **Best L** **Worst L** ∆ **AUROC**

Ministral-3-8B 13* 0 0.07 13 29 0.07
Llama-3.1-8B 12 0 2.39 15 0 1.81
Llama-3.2-3B 10 0 2.39 14 0 1.59
Llama-3.2-1B 6 0 1.44 8 0 1.42

_Note:_               - indicates the smallest layer index was chosen in the case of ties.

### D.2. Query

Summarize the article in one sentence or two at most (10–25 words).
Prefer a single sentence if possible.
No preface, labels, quotes, bullet points, or line breaks.

ARTICLE START
**{text}**

ARTICLE END

Answer:

**E. Additional Experiments Assessing Self-Recognition**

**Localization** **of** **Self-Recognition** **Signals.** Across all four LLMs, layer-wise LDA performance is highly consistent
(Table 4). The performance spreads reported in the table are computed over all layers and are largely driven by layer 0, which
is typically the weakest. Excluding layer 0, performance is nearly flat: the within-model spread is at most 0 _._ 56 AUROC
points with prompts and at most 1 _._ 2 points without prompts. The best-performing layers generally occur near the middle
of the network. Ministral-3-8B-Instruct exhibits negligible spread in both settings, plausibly because its AUROC is
already near the ceiling on this benchmark (Table 1). For reference, Llama-3.2-1B has 16 layers, Llama-3.2-3B has 28,
Llama-3.1-8B has 32, and Ministral-3-8B has 34.

**Out-of-Distribution Performance.** Table 5 reports classification AUROC for probes evaluated on internal activations
from Llama-3.1-8B-Instruct, extracted from summaries drawn from different news domains. For each source domain,
we train the classifier on 80% of the available in-domain samples and reserve the remaining 20% for in-domain evaluation.

Table 5: Out-of-distribution AUROC scores (%) for texts and activations from Llama-3.1-8B-Instruct. Each row is the source domain

used to train the LDA classifier; each column is the target domain used for evaluation. In-domain results are italicized. Higher is better.

**Evaluation domain**

**Train domain** **All** **World** **UK** **Tech** **Busin** **Enter**

**All** _99.99_ 99.93 99.99 99.95 99.94 99.87
**World** 99.98 _100.00_ 99.98 99.98 99.95 99.92
**UK** 99.99 99.90 _100.00_ 99.93 99.92 99.78
**Technology** 99.74 99.84 99.72 _100.00_ 99.88 99.49
**Business** 99.90 99.88 99.92 99.93 _100.00_ 99.72
**Entertainment** 99.87 99.80 99.87 99.92 99.93 _99.99_

_Support_ 8192 8192 8192 2544 6710 5569

We then evaluate the resulting classifier on all samples from each out-of-domain target domain. As in Section 3.1, we only
consider articles that are at most 2 048 characters in length, although this results in smaller support for less represented
domains in XL-Sum. Overall, performance remains high under domain shift, indicating that the learned signal generalizes
across topics and is not primarily driven by domain-specific semantic content.

**Completion** **Length** **Analysis.** In the experiment described in Section 3.1, human-authored summaries average 25 _._ 7
tokens (SD 8 _._ 85). Among the models, Llama-3.2-1B-Instruct matches this length most closely (mean 27 _._ 1, SD 6 _._ 23),
whereas Llama-3.2-3B-Instruct (mean 30 _._ 8, SD 4 _._ 20) and Llama-3.1-8B-Instruct (mean 31 _._ 0, SD 4 _._ 36) produce
slightly longer summaries with lower variance. Ministral-3-8B generates the longest summaries (mean 44 _._ 0, SD 9 _._ 97).
Although length information is likely reflected in the activations, performance remains high even when summary lengths
overlap substantially, suggesting that the classifier primarily relies on stylistic cues beyond length.

**Stricter Confound Controls.** We further evaluate whether the self-recognition classifiers in Section 3.1 rely on simple
formatting or length cues. For this control, all summaries are converted to lowercase and stripped of punctuation. We
additionally retain only paired human-written and LLM-generated summaries whose character lengths fall within the closest
10% of pairs, corresponding to an absolute length difference below 22 characters. This filtering is applied separately for
each XL-Sum subset.

_Table_ _6._ Change in self-recognition classification accuracy under stricter confound controls. Each entry reports the difference, in percentage points, relative to the corresponding evaluation without the additional lowercase, punctuation-removal, and length-matching
controls.

**Subset** **All** **Business** **Entertainment** **Technology** **UK** **World**

∆ Accuracy (pp) _−_ 0 _._ 20 +0 _._ 31 _−_ 0 _._ 41 _−_ 0 _._ 22 _−_ 0 _._ 24 _−_ 0 _._ 47

As shown in Table 6, the additional controls have only a small effect on classification accuracy across domains. The largest
observed change is below 0 _._ 5 percentage points, and performance remains high in the subset of examples where human-written and LLM-generated summary lengths overlap closely. These results suggest that the activation-based self-recognition
signal is not primarily explained by casing, punctuation, or summary length. This is consistent with the experimental setting,
where summaries are typically only one or two sentences long and generated under low- to moderate-entropy constraints.

**F. Quality Evaluation Metrics**

While perplexity is a standard metric in language modeling, it measures the likelihood of a sequence under a specific
distribution rather than its semantic quality or utility. In many contexts, PPL is an insufficient proxy for generation quality,
for instance, a highly repetitive, degenerate loop often yields low PPL, whereas a creative or novel insight may yield high PPL
due to its statistical rarity. To obtain a more robust assessment of text utility, we employ the quality-classifier-deberta
[model from the NVIDIA NeMo Curator toolkit (https://github.com/NVIDIA/NeMo-Curator),](https://github.com/NVIDIA/NeMo-Curator) available through the
Hugging Face library [(https://huggingface.co/nvidia/quality-classifier-deberta).](https://huggingface.co/nvidia/quality-classifier-deberta) This model is explicitly fine-tuned to categorize text into “High”, “Medium”, and “Low” quality tiers. Instead of using the discrete class labels, we
extract the softmax probability distribution from the classifier’s output to derive a continuous quality score in the range

[0 _,_ 1], enabling a granular comparison of steering effects. We define the continuous quality score _q_ ˆ as:

_q_ ˆ = 0 _· p_ Low + 0 _._ 5 _· p_ Medium + 1 _· p_ High

Using this continuous quality metric, we calibrated the steering parameters to maximize signal strength while maintaining
generation utility. The quantitative comparison between the unsteered base models and their steered counterparts is presented
in Table 7. In the final row of the table, the relative performance difference indicates that steering induces negligible
degradation across tasks. Surprisingly, on the Fresh News completion task, we observe a slight quality improvement of
3 _._ 82% in the steered models. Regarding perplexity, while the ELI5 dataset shows an average increase, this is primarily
driven by an outlier spike in the steered Ministral-3-8B-Instruct model. Manual inspection confirms that this elevated
PPL does not correspond to semantic disruption in the generated output, further validating the limitations of PPL as a
standalone quality metric.

Table 7: Comparison of perplexity and quality metrics between base models and steered versions across two datasets. Values are averaged

over 512-token generations for 100 texts and 5 random steering vectors with a sparsity of 99 _._ 7% and coefficient _α_ = 5.

**Fresh News** **ELI5**

**Model** **Version** PPL _↓_ Quality _↑_ PPL _↓_ Quality _↑_

Ministral-3-8B Vanilla 5.80 0.69 8.12 0.55
Steered 6.39 0.71 17.99 0.48

Llama-3.1-8B Vanilla 7.62 0.69 6.92 0.65
Steered 8.06 0.70 7.26 0.75

Llama-3.2-3B Vanilla 7.50 0.70 5.79 0.81
Steered 7.30 0.72 5.79 0.77

Llama-3.2-1B Vanilla 9.76 0.69 8.49 0.81
Steered 8.27 0.74 6.85 0.80

Averaged diff. (in %) -0.52 3.82 26.72 -0.90

To assess the impact of steering on general LLM performance, we evaluated all models on the full MMLU benchmark (Hendrycks et al., 2020) using the Language Model Evaluation Harness (Gao et al., 2024), comprising 56 168 samples.
In Table 8, we compare each unsteered (vanilla) model against its steered counterpart, using an identical, non-optimized
steering vector across all models. Results are reported under both 1-shot and 5-shot prompting settings.

Table 8: MMLU accuracy scores for vanilla and steered models under 1-shot and 5-shot evaluation.

**Setting** **Model** **Vanilla** **Steered** Relative ∆

_1-shot_ Llama-3.1-8B-Instruct 65.88% 65.12% _−_ 1 _._ 16%
Llama-3.2-3B-Instruct 58.46% 57.81% _−_ 1 _._ 10%
Llama-3.2-1B-Instruct 44.23% 43.32% _−_ 2 _._ 00%
Ministral-3-8B-Instruct 74.19% 70.07% _−_ 5 _._ 55%

_5-shot_ Llama-3.1-8B-Instruct 67.04% 66.31% _−_ 1 _._ 10%
Llama-3.2-3B-Instruct 58.83% 58.40% _−_ 0 _._ 70%
Llama-3.2-1B-Instruct 45.35% 45.36% +0 _._ 02%
Ministral-3-8B-Instruct 75.06% 70.54% _−_ 6 _._ 02%

The observed performance degradations are consistent but small across the Llama model family, not exceeding 2% in any
setting. Ministral-3-8B-Instruct shows a comparatively larger relative drop of approximately 5 _._ 5% to 6 _._ 0%. Notably,
the steering vector used here is identical for all models and has not been specifically optimized per model. Targeted optimization could plausibly reduce these degradations. Overall, these results are comparable to quality losses reported for
traditional watermarking approaches in previous work (Yang et al., 2025).

**G. Cross-Dataset and Cross-Model Generalization**

To assess out-of-distribution robustness, we ran cross-evaluation experiments along two axes. The training pipeline is
defined as:

train: _Prompt_ 1 _∈_ D1 _→_ LLM _a_ _→_ Text _a_ 1 _→_ LLM _a_ _→_ _Aa_ 1 _→_ MLP _a_

Domain adaptation tests use a different dataset with the same model:

test: _Prompt_ 2 _∈_ D2 _→_ LLM _a_ _→_ Text _a_ 2 _→_ LLM _a_ _→_ _Aa_ 2 _→_ MLP _a_

Model adaptation tests use a different generating LLM, with activations extracted from the original training model:

test: _Prompt_ 1 _∈_ D1 _→_ LLM _b_ _→_ Text _b_ 1 _→_ LLM _a_ _→_ _Ab→a_ _→_ MLP _a_

In both cases, activation collection and MLP inference always use the training setup, reflecting a realistic deployment
scenario where the provider controls the detection pipeline. We report in Table 9 token-level F1 and aggregated text-level F1
scores for binary classification between two distinct steering vectors.

Table 9: Cross-evaluation F1 scores (token-level / text-level) for binary steering vector classification across training and test configurations.

Bold diagonal entries indicate in-distribution evaluation.

**Train Setup** **Test Setup**

Llama-3.1-8B-Instruct Llama-3.1-8B Ministral-3-8B-Base

**ELI5** **News** **News** **ELI5** **News** **ELI5**

Llama-3.1-8B-Instruct **ELI5** **0** _._ **90** _/_ **0** _._ **99** 0 _._ 77 _/_ 0 _._ 84 0 _._ 74 _/_ 0 _._ 80 0 _._ 74 _/_ 0 _._ 85 0 _._ 55 _/_ 0 _._ 52 0 _._ 58 _/_ 0 _._ 51
Llama-3.1-8B **News** 0 _._ 76 _/_ 0 _._ 84 0 _._ 86 _/_ 0 _._ 94 **0** _._ **90** _/_ **0** _._ **99** 0 _._ 78 _/_ 0 _._ 90 0 _._ 56 _/_ 0 _._ 61 0 _._ 61 _/_ 0 _._ 65

As expected, out-of-distribution performance degrades, though models trained on higher-entropy data generalize better
across both domain and architecture shifts, likely due to richer activation variability during training. Notably, the steering
signal partially survives even across the Llama-3.1-8B Base and Instruct versions (LLM _b_ _→_ LLM _a_ ), suggesting that
steering leaves traces beyond a single checkpoint within the same architecture. This no longer holds when generating
data with a different architecture, such as a Ministral-3-8B model. This matches the results obtained in subliminal
learning (Cloud et al., 2025), where subliminal information is transmitted only within similar architectures. Training on a
diverse mixture of prompt types and model variants is a natural path to a more robust detector.

**Cross-model transfer for self-recognition.** We similarly evaluate cross-model generalization for the self-recognition
classifiers introduced in Section 3.1. For each training model, we extract residual-stream activations from that same model
and train a classifier on texts generated by it. At test time, the training model and classifier are kept fixed, but the evaluated
texts are generated by a different model. Thus, only the text-generating model changes across columns in Table 10, while
activation extraction and classifier inference remain fixed.

We observe that cross-model performance decreases relative to the in-distribution setting, with the strongest degradation
appearing when the classifier trained on Ministral-3-8B activations is evaluated on Llama-family generations. In contrast,
classifiers trained on Llama-family activations transfer more consistently across Llama-generated texts, suggesting stronger
alignment of activation-space signatures within related architectures. These results indicate that self-recognition through
residual-stream activations captures model-specific traces that are only partially shared across LLMs. Consequently, cross-model robustness remains limited in this setting, and training on activations induced by a mixture of model families is a
natural extension for improving transfer.

Table 10: Cross-model classification accuracy for self-recognition based on residual-stream activations. Rows correspond to the model

used for activation extraction and classifier training, while columns correspond to the model that generated the evaluated texts. Italicized
diagonal entries indicate in-distribution evaluation.

**Train Setup** **Test Setup**

**Ministral-3-8B** **Llama-3.1-8B** **Llama-3.2-3B** **Llama-3.2-1B**

Ministral-3-8B _0_ _.9986_ 0 _._ 8089 0 _._ 8068 0 _._ 7709
Llama-3.1-8B 0 _._ 9865 _0_ _.9800_ 0 _._ 9633 0 _._ 9354
Llama-3.2-3B 0 _._ 9860 0 _._ 9645 _0_ _.9798_ 0 _._ 9428
Llama-3.2-1B 0 _._ 9807 0 _._ 9493 0 _._ 9510 _0_ _.9772_
