# Latent Structure of Affective Representations in Large Language Models

- **Authors:** Benjamin J. Choi, Melanie Weber
- **Date:** 2026-04-08
- **Source:** https://arxiv.org/abs/2604.07382
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `choi-weber_latent-structure-affective_2604.07382`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Benjamin** **J.** **Choi** `benchoi@college.harvard.edu`
_Harvard_ _University_

**Melanie** **Weber** `mweber@seas.harvard.edu`

_Harvard_ _University_

**Abstract**

The geometric structure of latent representations in large language models (LLMs) is
an active area of research, driven in part by its implications for model transparency and AI
safety. Existing literature has focused mainly on general geometric and topological properties
of the learnt representations, but due to a lack of ground-truth latent geometry, validating
the findings of such approaches is challenging. Emotion processing provides an intriguing
testbed for probing representational geometry, as emotions exhibit both categorical organization and continuous affective dimensions, which are well-established in the psychology literature. Moreover, understanding such representations carries safety relevance. In this work,
we investigate the latent structure of affective representations in LLMs using geometric data
analysis tools. We present three main findings. First, we show that LLMs learn coherent
latent representations of affective emotions that align with widely used valence–arousal models from psychology. Second, we find that these representations exhibit nonlinear geometric
structure that can nonetheless be well-approximated linearly, providing empirical support
for the linear representation hypothesis commonly assumed in model transparency methods. Third, we demonstrate that the learned latent representation space can be leveraged to
quantify uncertainty in emotion processing tasks. Our findings suggest that LLMs acquire
affective representations with geometric structure paralleling established models of human
emotion, with practical implications for model interpretability and safety.

## 1 Introduction

Recent advances in mechanistic interpretability have illuminated how large language models
(LLMs) internally represent high-level semantic features such as factual knowledge, syntax,
and sentiment. A growing body of work investigates whether such features are geometrically
organized in semantically plausible, useful ways (Skean et al., 2024; Tigges et al., 2023; Lee
et al., 2025).

Emotions provide a particularly compelling domain for studying geometric representations
in LLMs because their semantic organization has been extensively studied in the psychology and
neuroscience literature. Emotions can be described both in terms of discrete categories (e.g.,
anger, joy, fear) and along continuous affective dimensions such as valence (positive–negative)
and arousal (calm–excited); in particular, this two-dimensional valence-arousal model (Fig. 1) of
emotion has proved particularly dominant in the psychology literature (Bradley and Lang, 1999;
Bliss-Moreau et al., 2020; Maleki et al., 2023). Because these structural models are based in
human cognition, they provide a natural benchmark for probing whether LLMs encode emotional
information in comparable ways to humans. If LLM latent spaces reproduce aspects of these
psychological and neuroscientific models, it may not only deepen our understanding of how LLMs
internally organize sentiment but also suggest potential parallels with natural intelligence.

The present study is situated in an _affective_ context. More specifically, the affective computing literature has explored how computational systems can recognize and interpret human
emotions across modalities such as speech and text (Calvo and D’Mello, 2010; Huang et al.,
2023; Picard, 2000). Inspired by this body of work, we design text-based emotion classification tasks to probe whether the LLM’s learnt latent representations recover known categorical
clusters; we hope to show, in line with previous ideas that distributed embeddings can encode
affective dimensions (Shah et al., 2022), whether continuous affective dimensions such as valence
and arousal emerge as dominant axes of organization. In other words, rather than emphasizing
the _generation_ of sentiment-bearing text, our analyses center on how models respond to (and
internally structure) sentiment-encoded _inputs_ .

**Related** **Work** Our work engages with two prominent theories on the internal geometry of
LLM representations. The first, the linear representation hypothesis, posits that high-level
concepts are encoded as simple linear directions in activation space. This view is supported by
studies showing that features like sentiment polarity can be identified with linear probes (Park
et al., 2023) and manipulated through causal interventions along a linear axis to steer model
outputs (Nanda et al., 2023; Park et al., 2023; Jin et al., 2025; Tak et al., 2025). Conversely,
the manifold hypothesis suggests that representations form more complex, nonlinear structures.
For instance, recent analyses show that internal representations of semantic categories can form
simplex-like hierarchical structures (Park et al., 2024), while other work has broadly identified
geometric and manifold-like structures in neural representations of language (Mamou et al.,
2020; He et al., 2024). By finding evidence for nonlinear latent structure in line with established
psychological models, our work helps inform the current debate on representation geometry.

Beyond representation geometry, a growing body of work has examined how LLMs process
and represent emotion more broadly. Huang et al. (2023) evaluated LLM emotional responses
using standardized psychological instruments, while Tak et al. (2025) applied mechanistic interpretability tools to show that emotion representations are functionally localized within transformer models in a manner consistent with cognitive appraisal theory. Most directly related to
our work, Zhao et al. (2025) analyzed probabilistic dependencies between emotional states in
LLM outputs and found that models naturally form hierarchical emotion organizations aligning
with Plutchik’s emotion wheel, with larger models developing more complex hierarchies. While
their analysis operates at the output-probability level rather than in internal activation space,
their finding that structured emotion organization emerges in LLMs provides complementary
motivation for our investigation of the geometry of these representations in the latent space.

Concurrent with our work, two independent studies have investigated emotion representations in LLMs through complementary lenses. Sofroniew et al. (2026) identify linear representations of emotion concepts in Claude Sonnet 4.5, showing that these representations are
organized by valence and arousal and causally influence alignment-relevant behaviors such as
reward hacking, blackmail, and sycophancy via activation steering. Sun et al. (2026) decompose
emotion steering vectors into a continuous two-dimensional valence–arousal subspace, demonstrating that steering along valence–arousal axes enables control over refusal and sycophancy.
Both works corroborate our central finding that LLMs develop affective representations with
geometric structure paralleling human psychological models; we discuss commonalities and differences in detail in Appendix E.

**Summary** **of** **Contributions** The main contributions of our work are as follows:

1. **Representational** **similarities** **with** **human** **affective** **models.** We show that Gemma-29B, Mistral-7B, as well as LLaMA-3-70B-Instruct develop coherent internal representations
of affective emotion. These representations align with established valence–arousal models
from psychology.

2. **Evidence** **for** **nonlinearity.** We find that the geometry of LLM emotion representations
exhibits modest nonlinearity (in line with the parabolic curvature of valence-arousal space).
While we find that affective emotion geometry is locally amenable to linear analysis, our
evidence for nonlinear global structure suggests that a purely linear representation hypothesis
is insufficient.

3. **Applications** **to** **uncertainty** **quantification.** We demonstrate that the geometry of these
structured representation spaces can be exploited to quantify predictive uncertainty in emotion recognition tasks, illustrating both practical utility and interpretability gains.

Additionally, in the appendices we provide complementary evidence from two directions: a
parallel analysis showing that similar parabolic affective geometry emerges in human neural
(EEG) data (Appendix B), and causal steering experiments demonstrating that probe-derived
emotion directions can be used as activation vectors to shift the emotional valence of generated
text (Appendix D).

## 2 Background

**Constructing Latent Space Representations** We will employ two manifold learning methods for recovering latent space representations of language model embeddings: classical multidimensional scaling (MDS) and Isometric Feature Mapping (Isomap).

_Classical_ _MDS_ (Torgerson, 1952) takes a dissimilarity matrix _D_ = [ _dij_ ] and constructs the
doubly centered squared-distance matrix _B_ = _−_ [1] [with] _[J]_ [=] _[I]_ _[−]_ [1] [An] [eigendecom-]

2 _[JD][◦]_ [2] _[J]_ _n_ **[11]** _[⊤]_ [.]
position _B_ = _Q_ Λ _Q_ _[⊤]_ then yields coordinates _zi_ _∈_ R _[k]_ from the top- _k_ eigenpairs, chosen so that
pairwise Euclidean distances in the embedding approximate the original dissimilarities in _D_ .
This closed-form solution provides a linear Euclidean representation of the data.

_Isomap_ (Tenenbaum, 1997) extends MDS to account for nonlinear structure by replacing
the raw Euclidean distances _dij_ with estimates of geodesic distances along the data manifold.
In practice, this is achieved by constructing a _k_ -nearest neighbor ( _k_ NN) graph _G_ over the embeddings and computing approximate geodesic distances _d_ [˜] _ij_ as shortest-path distances on _G_ ;
classical MDS is then applied to the geodesic distance matrix. By incorporating this local neighborhood structure, Isomap can capture curvature in the embedding space and reveal deviations
from purely linear structure.

**Valence-Arousal** **Model** **of** **Emotion** A widely used framework in psychology and affective
science conceptualizes emotions along two continuous dimensions: _valence_, which captures the
degree of pleasantness or unpleasantness, and _arousal_, which indexes physiological activation
or intensity (Russell, 1980; Bliss-Moreau et al., 2020; Kim et al., 2020; Maleki et al., 2023).
Early work in this space originally posited circumplex emotion layouts centered around neutral
(Russell, 1980). In recent years, however, psychology literature has increasingly adopted the notion of a parabolic (i.e., “V”-shaped) geometric layout of emotion (Kim et al., 2020; Maleki
et al., 2023), owing to a general correlation between valence intensity and arousal arising in the
typical distribution of human emotion. Geometric visualizations of the dual-axis valence-arousal
emotion layout are shown in Figure 1.

## 3 Methods

**3.1** **Preliminaries**

To study the organization of emotional representations in LLMs, we begin by defining a notion of similarity between embeddings corresponding to different
emotions. For each emotion pair ( _i, j_ ), we train a
pairwise logistic regression classifier with L2 regularization on mean-pooled hidden state activations (see
Section 3.2). We treat the test accuracy acc _ij_ of this
classifier as a proxy for the distance between emotions
_i_ and _j_, with higher accuracy indicating greater separability. We then define by default _Dij_ = acc _ij_ (all
main findings hold under alternative metrics, including unsupervised cosine distance; see Appendix A.2).
This yields a symmetric dissimilarity matrix that captures inter-emotion relationships in LLM activation space, since higher separation accuracy means the model of affective space, based on Maleki two emotions form well-separated activation clusters, et al. (2023).

Figure 1: Parabolic valence–arousal

while lower accuracy reflects overlapping representations. We also verified that our results are robust under alternative definitions of dissimilarity,
such as activation cosine similarity (see Appendix A), distances to logistic regression hyperplanes, or alternative accuracy-to-distance transformations. Our logistic regression approach,
however, is used by default as it provides a well-defined class-separating hyperplane, which plays
a key role in our downstream uncertainty quantification application (see Section 6).

**3.2** **Methodology**

The main aim of our exploratory analysis is investigating the following hypothesis:

_Do_ _LLMs_ _develop_ _coherent_ _internal_ _representations_ _of_ _emotions_ _that_ _align_ _with_ _the_
_valence-arousal_ _model?_

To test this hypothesis, we conduct an activation-based probing analysis across Mistral-7B
and Gemma-2-9B centered around the geometry of representations in an explicit emotionclassification setting. In line with affective computing practices, we feed each prompt to the
target LLM and prompt the model to classify the emotion from the randomized list of GoEmotions choices, subsequently storing activations corresponding to both correct and incorrect
classifications. All inference was performed in zero-shot mode, with no additional fine-tuning
on GoEmotions or related datasets. For each text sample, we extracted mean-pooled hidden
state activations from all transformer layers using forward hooks. Specifically, we collected the raw hidden states from each of the 32 layers in Mistral-7B and 42 layers in Gemma-2-9B, where
each layer produces activations of dimension [1, sequence_length, hidden_size] since we process
one sample at a time. We then applied mean pooling across the sequence dimension to obtain
a single vector representation per layer for each model; we use layer outputs as opposed to
sub-layer components, as in Ju et al. (2024) and Li et al. (2024). The activation vectors from
correctly classified samples were used to train _pairwise_ classifiers between emotions, enabling us
to identify how the geometric organization of emotional representations changes across depth
and to assess where representations are most discriminative. The incorrect activations were
stored for future downstream analysis (see Section 6).

We focused on binary emotion comparisons rather than multiclass setups. This approach
likewise avoids confounds from highly imbalanced class sizes (per-class correct sample counts
range from 101 to 2,212 in Gemma-2-9B and 103 to 1,293 in Mistral-7B), yields clearer and
more stable decision boundaries in activation space, and dovetails with MDS, which operates on
pairwise dissimilarities. For each emotion pair, [1] we balanced the dataset by downsampling the
majority class, then trained a logistic regression classifier with L2 regularization using an 80:20
train-test split. The resulting classification accuracies were subsequently converted into dissimilarity values _Dij_ as described in Section 3.1, providing the pairwise distances for downstream
MDS analysis.

## 4 Exploratory Data Analysis

**4.1** **Experimental** **Setting**

**Models** We examine two state-of-the-art transformer-based LLMs: Gemma-2-9B (Riviere
et al., 2024) and Mistral-7B (Jiang et al., 2023). Gemma-2-9B is a 9-billion parameter, 42layer model developed by Google DeepMind, while Mistral-7B is a 7-billion parameter, 32-layer
model from Mistral AI. Although the exact training mixtures for these models have not been
publicly disclosed, both were trained on large-scale, diverse web corpora that likely included
sentiment-laden text, providing them with natural exposure to affective language. We also experimented with analyses on LLMs from the Qwen and LLaMA families but found that these
models did not satisfy requisite sentiment recognition baselines necessary for our downstream
experiments; further discussion is included in Appendix C. In addition, to assess generalization across both scale and training regime, we conducted a full replication of our pipeline on
LLaMA-3-70B-Instruct, a substantially larger and instruction-tuned model (see Section 7).

**Datasets** Our primary dataset is _GoEmotions_ (Demszky et al., 2020), a manually annotated
corpus of 58,009 English Reddit comments labeled for 27 fine-grained emotion categories plus
_Neutral_ . The taxonomy comprises 12 positive, 11 negative, and 4 ambiguous categories, enabling
analyses along both valence-aligned dimensions and discrete categories. Each comment was
annotated by 3 or 5 independent raters (82 raters in total). We restrict our analyses to single-label examples exhibiting multi-rater agreement ( _∼_ 83% of examples). As the largest manually
annotated, fine-grained English emotion dataset to date, GoEmotions is a natural choice for our
study.

1To ensure sufficient dataset size, we conducted our analyses on emotion categories with at least 100 correct
LLM classifications across the dataset, resulting in 20 emotions for Gemma-2-9B and 16 emotions for Mistral-7B
(spanning positive, negative, ambiguous, and neutral valences).

Figure 2: 2D classical MDS embeddings of internal latent emotion representations in Gemma2-9B (left) and Mistral-7B (middle), as well as ANEW reference valence-arousal scores (right).
Layers chosen correspond to peak separability (Gemma-2-9B Layer 20, Mistral-7B Layer 27)
and are representative of overall patterns observed throughout. Both plots are anchored to fixed
orientations with neutral at the origin; colors correspond to the given GoEmotions taxonomy,
with _k_ NN background shading by valence included.

**4.2** **Semantic** **Analysis** **of** **Emotion** **Representations**

In our layer-by-layer activation analyses (see Appendix A for further details), we found that mean
pairwise emotional separability generally rose toward the middle layers while exhibiting a slight
drop at the final layers. Additionally, we found that Gemma-2-9B exhibited significantly higher
activation separability compared to Mistral-7B, with mean test accuracies in the _>_ 0.9 range
across all layers compared to _∼_ 0.55 to _∼_ 0.89 in Mistral-7B. This similarly aligns with Gemma2-9B’s greater LLM emotion recognition performance in the initial emotion classification setting,
with an estimated _∼_ 19.4% correct classification rate in Gemma-2-9B compared to _∼_ 13.7% in
Mistral-7B.

The latent representations obtained with classical MDS show that LLMs do indeed learn
coherent representations that mirror logical semantic structure. Figure 2 shows the 2D projections of the classical MDS embeddings for Gemma-2-9B and Mistral-7B, respectively; both
models exhibit clear semantic clustering, with positive emotions (joy, love, gratitude) clustering
together in the upper-right quadrant and negative emotions (sadness, anger, disgust) grouping
in the upper-left quadrant.

**4.3** **Comparison** **with** **Valence-Arousal** **Maps**

On a qualitative level, we observe notable similarities between our internal latent space representations and valence-arousal organization from the psychology literature. We find that
the learned emotion structure (Figure 2) across both LLMs appears to mimic the “V”-shaped
parabolic emotion layout, with neutral emotions positioned approximately at the vertex and
positive (and negative) valences clustering together along a dual “arm” structure. This semantic organization is visible in virtually all Gemma-2-9B layers and all but the early layers of

Mistral-7B [2] .

Beyond our qualitative support, we conduct a quantitative statistical test to compare our
learned LLM representations against conventional valence-arousal maps. In particular, we compare the 2D MDS emotion embedding at each model layer with the map produced by third-party
valence-arousal scores from the widely-used ANEW benchmark database from Bradley and Lang
(1999). The ANEW coordinates (see Figure 2, right panel) for our statistical test come directly
from human normative ratings, in which participants provided continuous valence and arousal
scores for each lexical item. (We conduct our analyses using the 17 of the 28 classes in GoEmotions that are simultaneously covered in ANEW.) We then center both the MDS coordinates
and the ANEW valence-arousal coordinates and align the MDS coordinates via scaled orthogonal Procrustes (single rotation and uniform scale), yielding fitted points corresponding to the
MDS LLM latent representation. The test statistic is the Procrustes _R_ [2], which measures the
proportion of variance in the centered target configuration explained by the fitted configuration; significance is evaluated under a label-permutation null that shuffles the MDS emotion
labels, recomputing the alignment and _R_ [2] on each of 2 _,_ 000 permutations. We compute one-sided _p_ -values of the observed _R_ [2] relative to this null under emotion label permutation. (This
procedure is invariant to arbitrary MDS orientation/scale and does not assume metric validity
of the separability matrix; inference derives entirely from permutation.)

We find that 36 of 42 layers in Gemma-2-9B and 17 of 32 layers in Mistral-7B (including all
of the final 14 layers in Mistral-7B) exhibit statistically significant alignment with the ANEW
valence-arousal scores, providing additional quantitative evidence that the learned LLM latent
representations exhibit semantically coherent structure.

## 5 Evaluating Geometric Structure

Our analyses in the previous section are based on a linear embedding assumption via classical
MDS. To further probe the intrinsic geometry of LLM emotion representations, we conduct
two quantitative analyses. First, we assess the dimensionality of pairwise emotion distances by
examining the eigenspectrum of the classical MDS Gram matrix. Second, we investigate the
manifold hypothesis using Isomap (Tenenbaum, 1997), constructing a _k_ NN graph, estimating
geodesic distances, and embedding them with classical MDS.

Eigenspectrum analyses (see Appendix A, Figure 6, left panel for an example), conducted
across each individual layer of both Gemma-2-9B and Mistral-7B, reveal a generally diffuse rather
than conclusively low-dimensional geometry. Under various monotone dissimilarity mappings
( _Dij_ = acc _ij_ and _Dij_ = max(0 _,_ 2 _·_ acc _ij_ _−_ 1)), the participation ratio remains high (Gemma-29B _∼_ 17, Mistral-7B _∼_ 9–14), indicating that variance is spread across many eigenmodes [3] . This
diffuse spectrum is consistent with what one would expect from a high-rank “bulk” component—
akin to a Wigner-like distribution (ErdHos et al., 2009)—from noise and task-irrelevant variation,
with meaningful structure appearing only as deviations from this bulk. Thus, our goal is not
to claim a globally low-dimensional manifold, but rather to identify statistically significant axes

2These early Mistral-7B layers generally exhibit very low activation space emotion separability (see Appendix A), so the lack of semantic geometry here is consistent with the general observation that early Mistral-7B
layers do not yet encode any coherent affective structure.

3We note that as our dissimilarity metric is merely distance-like, we do observe occasional negative eigenvalues
in our spectrum; however, only 1-2 negative eigendirections generally appear per layer, and these compose less
than 1% of overall variance mass across virtually all layers exhibiting emotional separability.

Figure 3: Isomap ( _left_ ) appears to unroll parabolic structure in the emotion data manifold,
resulting in improvements in capturing rank-1 structure over classical MDS ( _right_ ). An artificial
_y_ -axis jitter is introduced for label visibility, and we include _k_ NN background shading by valence
to illustrate improved semantic coherence ( _left_ ) under Isomap.

that separate from the bulk and correspond to affective structure. We do so via a statistical
eigengap test where we evaluate scale-invariant ratios _rk_ = _λk/λk_ +1 between ordered eigenvalues
against a permutation-generated null. This null distribution is generated by shuffling the off-diagonal of _D_ (symmetry preserved), and we flag eigenvalues where the observed ratio falls
in the high tail ( _p_ hi _<_ 0 _._ 05). In general, our permutation test results (assessing prominence
from the eigenspectrum bulk) appear generally consistent with a diffuse overall structure and no
definitive fixed rank. We do find that Gemma-2-9B shows a recurring significant first eigengap
with 16 out of 42 layers meeting a significance threshold at _k_ = 1, consistent with a dominant
valence-like axis; we note, however, that this gap does not by itself establish any sort of fixed
intrinsic dimension, but rather points to a recurring tendency toward a prominent individual
axis of variation.

Our complementary Isomap evaluations allow us to probe for potential nonlinear manifold
structure. For each layer, we constructed _k_ NN [4] graphs over the pairwise separability matrix _|d_ _[′]_ _|_,
computed geodesic distances, and compared Isomap against a classical MDS (Euclidean) baseline
via two diagnostics: (i) relative trustworthiness (see Appendix C for formal definition) of local
neighborhoods, and (ii) divergence between geodesic and Euclidean distances. On the overall
trustworthiness front, we find generally minimal (i.e., _<_ 0 _._ 03) differences in trustworthiness
between Isomap and classical MDS across ranks and choices of k–with the exception of the
rank-1 embedding, for which Isomap demonstrates a median trustworthiness increase of 0.161
in Gemma-2-9B and 0.127 in Mistral-7B. We speculate that this rank-1 improvement is due to
the aforementioned parabolic “V”-shaped structure of the emotion data manifold, which Isomap
naturally captures in “unrolled” form in the 1D setting as shown in Figure 3.

In terms of discrepancies between geodesic (Isomap) vs. Euclidean (classical MDS) distances, we observe evidence consistent with modest nonlinearity. Specifically, distance ratios
(i.e., geodesic / Euclidean) span from 1.00 to 1.80 from the tenth to ninetieth percentiles in
Gemma-2-9B and 0.80 to 1.42 in Mistral-7B [5] . These results suggest that the LLM emotion

4We select the Isomap _k_ parameter to maximize trustworthiness.
5We select the embedding dimension _d_ via an elbow-esque method where we identify the first residual variance spaces are almost Euclidean: high-rank, diffuse, and lacking strong low-dimensional curvature.
The manifold nonlinearities that do appear in higher rank spaces appear consistent with the
natural parabolic bending of valence–arousal space (e.g., see Appendix A, Figure 6, right panel,
in addition to Figure 3); the emotion data manifold exhibits a natural mode of curvature (due
to the correlation between valence intensity and arousal) that manifests in LLM latent representations. On the whole, however, nonlinear manifold structure does not appear strong enough
to render Euclidean representations ineffective as a basis for analysis.

## 6 Applications in Uncertainty Quantification

During our pairwise logistic regression tests, we Table 1: Mean trustworthiness improvement
also saved activations corresponding to LLM using Isomap over classical MDS per rank _d_
misclassifications. Unlike the correct activation
setting, where each activation is associated with
a single emotion, each misclassification is linked Model _d_ = 1 _d_ = 2 _d_ = 3 _d_ = 4
to two emotions: the model outputted emotion, Gemma 0.155 -0.001 -0.001 0.011
and the ground-truth GoEmotions label. Under Mistral 0.099 -0.024 -0.026 -0.011
a naive assumption of no semantic affective link
_d_ = 5 _d_ = 6 _d_ = 7 _d_ = 8 ages, one might expect the misclassified activations to lie close in activation space to the “cor- Gemma 0.013 0.020 0.024 0.022

Mistral -0.005 0.001 -0.004 -0.007 rect” activations for the LLM-outputted class.
However, we instead observe a remarkable phenomenon where these misclassified activations instead lie in between the two emotions to which
they are linked [6] ; that is, the “misclassified” activations hover much closer to the separating hyperplane between the two emotions (as defined by the corresponding pairwise logistic regression)
than the “correct” activations do. We note that this phenomenon is not tautological, as we only
evaluate distances to the separating hyperplane in terms of correct _test_ samples and misclassified
activations, neither of which were included in the original logistic regression training data.

We can exploit this observation to generate practical utility by training a second round of
pairwise logistic regression models on per-layer activation-to-hyperplane distances. Our work
in this predictive _uncertainty_ _quantification_ setting is motivated by the potential benefit of
detecting LLM emotion processing misclassifications–the core concept, here, is to use the distance
in activation space associated with an LLM prediction from the separating hyperplane between
two classes as a measure of LLM “confidence” in that prediction. We restrict our analyses
to emotion pairs with at least 25 correct and 25 misclassified samples in the data, resulting
in 180 viable binary classification problems (3,858 correct samples vs. 8,748 misclassified) for
Gemma-2-9B and 122 for Mistral-7B (5,254 correct vs. 15,216 misclassified) under a 60:20:20
train-val-test split. We use validation data in order to calibrate the logistic regression models,
mapping raw hyperplane distances across all layers to well-calibrated probability estimates of
correctness.

Our trained uncertainty models post 77.6% accuracy (0.813 AUC-ROC) on Gemma-2-9B
and 85.7% accuracy (0.871 AUC-ROC) on Mistral-7B, compared to majority-class baselines drop less than 0.02.

6Early layers lie slightly closer to the original prompt emotion, and late layers lie slightly closer to the erroneous
outputted emotion (see Appendix A).

Table 1: Mean trustworthiness improvement
using Isomap over classical MDS per rank _d_

Model _d_ = 1 _d_ = 2 _d_ = 3 _d_ = 4

Gemma 0.155 -0.001 -0.001 0.011

Mistral 0.099 -0.024 -0.026 -0.011

_d_ = 5 _d_ = 6 _d_ = 7 _d_ = 8

Gemma 0.013 0.020 0.024 0.022

Mistral -0.005 0.001 -0.004 -0.007 of 69.4% and 82.2%, respectively. Importantly, post-calibration quality in terms of predicted
correctness probability is strong: we find expected calibration errors of 0.011 (Gemma-2-9B) and
0.007 (Mistral-7B) on held-out test data. These results show that geometry-informed separating
hyperplane distance–based regressions yield well-calibrated, discriminative uncertainty estimates
in an LLM classification setting.

Figure 4: Upper panels depict activation distance from separating hyperplane, with misclassifications lying closer to the boundary than correct classifications. Lower panels depict test results
of calibrated uncertainty quantification models, demonstrating robust calibration quality.

## 7 Generalization: LLaMA-3-70B-Instruct

To further assess the generalizability of our findings, we conducted a complete replication of
our pipeline on LLaMA-3-70B-Instruct (AI@Meta, 2024), a substantially larger (70B parameter) and instruction-tuned model. Instruction tuning introduces supervised alignment and
human-feedback–driven optimization, which can reshape representational structure; demonstrating stability of affective geometry under such changes therefore provides a more rigorous test of
generalization.

LLaMA-3-70B-Instruct achieves a zero-shot emotion classification accuracy of 21.3% on GoEmotions, outperforming both Mistral-7B (13.7%) and Gemma-2-9B (19.4%). Classical MDS
embeddings reveal the same macroscopic affective layout observed in the earlier models, with
positive and negative emotions diverging along two “arms” and neutral emotions near the geometric vertex (Figure 5). The scaled orthogonal Procrustes test confirms alignment: 34 of 80 layers exhibit significant 2D alignment with _p_ 2 _D_ _<_ 0 _._ 05, concentrated among later layers. As
with the smaller models, Isomap recovers the expected parabolic structure in low-rank embeddings, and trustworthiness scores improve in the rank-1 setting relative to classical MDS. We
also replicated the uncertainty quantification pipeline: misclassified samples again lie closer to
pairwise separating hyperplanes than correct samples. On held-out test data, the uncertainty
model posts 80.1% accuracy (0.822 AUC-ROC) with expected calibration error of 0.011 (baseline accuracy: 75.6%). These results reinforce the generality of our findings across architectures,
scales, and training paradigms; further details are in Appendix A, Figure 13.

Figure 5: Corroboration of key results on LLaMA-3-70B-Instruct. Top-left: MDS emotion layout. Top-right: uncertainty model calibration. Bottom: Isomap embedding reflecting unrolled
curvature.

## 8 Discussion

Our analyses reveal that LLM emotion representations exhibit coherent geometric structure
that aligns with affective models from psychology. Moreover, we show direct utility in the form
of calibrated uncertainty models for LLM emotion processing, which leverage representation
geometry to provide reliable estimates of predictive confidence.

**Limitations** First, it remains unclear whether the observed patterns generalize beyond the
tested LLMs. Expanding our analysis to a wider range of models and families is an important
direction for future work, as is a systematic study of the effects of emotion-focused fine-tuning.
Second, our use of pairwise logistic regressions in conjunction with MDS and Isomap may not fully capture potential geometric intricacies; distributional analyses and advanced topological
tools could reveal nuanced relationships in future work. Third, our primary GoEmotions dataset
could have cultural or linguistic biases that constrain generality. Extending the analysis to more
naturalistic prompts and to sub-layer components (e.g., attention vs. MLP) is also a valuable
direction for future work.

**Broader** **Impacts** The parallels we uncover between LLM and human affective representations highlight potential similarities between artificial and human cognition (see Appendix B),
suggesting that geometric priors rooted in human psychology may inform future interpretability
frameworks. Our uncertainty quantification method also illustrates a principled way to detect likely misclassifications, with relevance to selective prediction, human-in-the-loop systems,
hallucination detection, and other safety-critical applications. Additionally, our supplemental
steering experiments (Appendix D) provide causal evidence that these geometric directions are
not merely diagnostic but functionally involved in the model’s generation process, with implications for fine-grained behavioral control of LLM outputs.

**Reproducibility** **Statement** To facilitate replication, we provide anonymized code to reproduce experiments, figures, and statistical tests at this link. Further experimental details are in
Appendix C.

**Acknowledgements** MW acknowledges support from NSF award DMS-2406905 and Coefficient Giving.

## References

AI@Meta. Llama 3 model card. 2024. URL `[https://github.com/meta-llama/llama3/blob/](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)`

`[main/MODEL_CARD.md](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)` .

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and

Neel Nanda. Refusal in language models is mediated by a single direction. _Advances_ _in_
_Neural_ _Information_ _Processing_ _Systems_, 37:136037–136083, 2024.

Steven Bird, Ewan Klein, and Edward Loper. _Natural language processing with Python_ . O’Reilly

Media, 2009.

Eliza Bliss-Moreau, Lisa A Williams, and Anthony C Santistevan. The immutability of valence and arousal in the foundation of emotion. _Emotion_, 20(6):993, 2020.

Margaret M Bradley and Peter J Lang. Affective norms for english words (anew): Instruction manual and affective ratings. Technical report, 1999.

Rafael A Calvo and Sidney D’Mello. Affect detection: An interdisciplinary review of models, methods, and their applications. _IEEE_ _Transactions_ _on_ _affective_ _computing_, 1(1):18–37, 2010.

Jingjing Chen, Xiaobin Wang, Chen Huang, Xin Hu, Xinke Shen, and Dan Zhang. A large finer-grained affective computing eeg dataset. _Scientific_ _Data_, 10(1):740, 2023.

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. Goemotions: A dataset of fine-grained emotions. _arXiv_ _preprint_
_arXiv:2005.00547_, 2020.

L´aszl´o ErdHos, Benjamin Schlein, and Horng-Tzer Yau. Local semicircle law and complete
delocalization for wigner random matrices. _Communications_ _in_ _Mathematical_ _Physics_, 287
(2):641–655, 2009.

Christiane Fellbaum. _WordNet:_ _An_ _electronic_ _lexical_ _database_ . MIT Press, 1998.

Peixuan Han, Cheng Qian, Xiusi Chen, Yuji Zhang, Denghui Zhang, and Heng Ji. Safeswitch:

Steering unsafe llm behavior via internal activation signals. _arXiv_ _preprint_ _arXiv:2502.01042_,
2025.

Yuan He, Moy Yuan, Jiaoyan Chen, and Ian Horrocks. Language models as hierarchy encoders.

_Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 37:14690–14711, 2024.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. _arXiv_ _preprint_ _arXiv:1904.09751_, 2020.

Jen-tse Huang, Man Ho Lam, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao,

Zhaopeng Tu, and Michael R Lyu. Emotionally numb or empathetic? evaluating how llms
feel using emotionbench. _arXiv_ _preprint_ _arXiv:2308.03656_, 2023.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chap
lot, Diego de las Casas, et al. Mistral 7b. abs/2310.06825, 2023. doi: 10.48550/arXiv.2310.
06825.

Mingyu Jin, Qinkai Yu, Jingyuan Huang, Qingcheng Zeng, Zhenting Wang, Wenyue Hua, Haiyan

Zhao, Kai Mei, Yanda Meng, Kaize Ding, et al. Exploring concept depth: How large language models acquire knowledge and concept at different layers? In _Proceedings_ _of_ _the_ _31st_
_international_ _conference_ _on_ _computational_ _linguistics_, pages 558–573, 2025.

Tianjie Ju, Weiwei Sun, Wei Du, Xinwei Yuan, Zhaochun Ren, and Gongshen Liu. How large language models encode context knowledge. _A_ _Layer-Wise_ _Probing_ _Study_, 2024.

Byung Hyung Kim, Sungho Jo, and Sunghee Choi. A-situ: a computational framework for affective labeling from psychological behaviors in real-life situations. _Scientific_ _reports_, 10(1):
15916, 2020.

Terry K Koo and Mae Y Li. A guideline of selecting and reporting intraclass correlation coeffi
cients for reliability research. _Journal_ _of_ _Chiropractic_ _Medicine_, 15(2):155–163, 2016.

LDNOOBW. List of dirty, naughty, obscene, and otherwise bad words. `[https://github.com/](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)`

`[LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)`, 2012.

Andrew Lee, Melanie Weber, Fernanda Vi´egas, and Martin Wattenberg. Shared global and local geometry of language model embeddings. In _Conference_ _on_ _Language_ _Modeling_, 2025.

Zichao Li, Yanshuai Cao, and Jackie C Cheung. Do llms build world representations? probing through the lens of state abstraction. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_,
37:98009–98032, 2024.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neu
big. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural
language processing. _ACM_ _Computing_ _Surveys_, 55(9):1–35, 2023.

Gheysar Maleki, Mohammad Ali Mazaheri, Vahid Nejati, Khatereh Borhani, and Guy Bosmans.

The attachment-related picture set (arps): development and validation. _Current_ _Psychology_,
42(5):3668–3679, 2023.

Jonathan Mamou, Hang Le, Miguel Del Rio, Cory Stephenson, Hanlin Tang, Yoon Kim, and

SueYeon Chung. Emergence of separable manifolds in deep language representations. _arXiv_
_preprint_ _arXiv:2006.01095_, 2020.

Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction. _arXiv_ _preprint_ _arXiv:1802.03426_, 2018.

Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. _arXiv_ _preprint_ _arXiv:2309.00941_, 2023.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. _arXiv_ _preprint_ _arXiv:2402.03658_, 2023.

Kiho Park, Yo Joong Choe, Yibo Jiang, and Victor Veitch. The geometry of categorical and hierarchical concepts in large language models. _arXiv_ _preprint_ _arXiv:2406.01506_, 2024.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, et al. Pytorch: An imperative style, high-performance deep learning library. _Advances_ _in_
_Neural_ _Information_ _Processing_ _Systems_, 32, 2019.

Rosalind W Picard. _Affective_ _computing_ . MIT press, 2000.

Laria Reynolds and Kyle McDonell. Prompt programming for large language models: Beyond the few-shot paradigm. _arXiv_ _preprint_ _arXiv:2102.07350_, 2021.

Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, et al. Gemma 2: Improving open language models at a practical size. _arXiv_ _preprint_
_arXiv:2408.00118_, 2024.

James A Russell. A circumplex model of affect. _Journal_ _of_ _personality_ _and_ _social_ _psychology_, 39

(6):1161, 1980.

Sapan Shah, Sreedhar Reddy, and Pushpak Bhattacharyya. Affective retrofitted word embed
dings. In _Proceedings_ _of_ _the_ _2nd_ _Conference_ _of_ _the_ _Asia-Pacific_ _Chapter_ _of_ _the_ _Association_ _for_
_Computational_ _Linguistics_ _and_ _the_ _12th_ _International_ _Joint_ _Conference_ _on_ _Natural_ _Language_
_Processing_ _(Volume_ _1:_ _Long_ _Papers)_, pages 550–561, 2022.

Oscar Skean, Md Rifat Arefin, Yann LeCun, and Ravid Shwartz-Ziv. Does representation mat
ter? exploring intermediate layers in large language models. _arXiv preprint arXiv:2412.09563_,
2024.

N Sofroniew, I Kauvar, W Saunders, R Chen, T Henighan, S Hydrie, C Citro, A Pearce, J Tarng,

W Gurnee, et al. Emotion concepts and their function in a large language model. _An-_
_thropic/Transformer_ _Circuits._ _transformer-circuits._ _pub/2026/emotions_, 2026.

Lihao Sun, Lewen Yan, Xiaoya Lu, Andrew Lee, Jie Zhang, and Jing Shao. Valence-arousal subspace in llms: Circular emotion geometry and multi-behavioral control. _arXiv_ _preprint_
_arXiv:2604.03147_, 2026.

Ala N Tak, Amin Banayeeanzade, Anahita Bolourani, Mina Kian, Robin Jia, and Jonathan

Gratch. Mechanistic interpretability of emotion inference in large language models. In _Findings_
_of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_ _ACL_ _2025_, pages 13090–13120, 2025.

Joshua Tenenbaum. Mapping a manifold of perceptual observations. _Advances_ _in_ _neural_ _infor-_

_mation_ _processing_ _systems_, 10, 1997.

Curt Tigges, Oskar Hollinsworth, Atticus Geiger, and Neel Nanda. Linear representations of sentiment in transformer language models. _arXiv_ _preprint_, 2023.

Warren S Torgerson. Multidimensional scaling: I. theory and method. _Psychometrika_, 17(4):

401–419, 1952.

Alexander Matt Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte

MacDiarmid. Activation addition: Steering language models without optimization. _arXiv_
_preprint_ _arXiv:2308.10248_, 2023.

Jarkko Venna and Samuel Kaski. Neighborhood preservation in nonlinear projection methods:

An experimental study. In _International_ _conference_ _on_ _artificial_ _neural_ _networks_, pages 485–
491. Springer, 2001.

Chen Xiong, Zhiyuan He, Pin-Yu Chen, Ching-Yun Ko, and Tsung-Yi Ho. Steering externalities:

Benign activation steering unintentionally increases jailbreak risk for large language models.
_arXiv_ _preprint_ _arXiv:2602.04896_, 2026.

Bo Zhao, Maya Okawa, Eric J Bigelow, Rose Yu, Tomer Ullman, Ekdeep Singh Lubana, and

Hidenori Tanaka. Emergence of hierarchical emotion organization in large language models.
_arXiv_ _preprint_ _arXiv:2507.10599_, 2025.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan,

Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering:
A top-down approach to ai transparency. _arXiv_ _preprint_ _arXiv:2310.01405_, 2023.

## A Additional Experimental Results

Figure 6 depicts further results from our analyses in Section 5. Figures 7, 8, and 9 depict
further results from our pairwise logistic regression and classical MDS experiments. The mean
emotional separability between discrete emotion classes for each layer in Figure 7 is measured
via pairwise logistic regression test accuracies.

Figure 6: Representative examples of a classical MDS eigenspectrum ( _left_ ) and an Isomap
embedding visualization ( _right_ ).

Figure 7: Mean emotional separability across layers for Gemma-2-9B (left) and Mistral-7B
(right). Separability increases with depth before tapering off in the final layers, with Gemma-29B exhibiting higher separability than Mistral-7B.

Figure 9 depicts results from our statistical alignment test between the classical MDS embeddings and established valence-arousal scores, showing results ( _R_ [2] and _p_ -values) for two statistically significant layers in Gemma-2-9B and Mistral-7B. Figure 10 depicts layer-by-layer trends
from our pairwise logistic regressions on the out-of-sample misclassified pairs. More specifically,

Figure 8: MDS emotion layouts for final layers in Gemma-2-9B ( _left_ ) and Mistral-7B ( _right_ ).

lower (or more negative) y-axis values signify that activations are closer to the ground truth
input emotion, while higher values signify that activations are closer to the output emotion.
Misclassified activations, as discussed in Section 6, lie closer to the separating hyperplane than
correct activations; as shown in Figure 10, activations trend away from the ground truth input
emotion and toward the model output emotion as model layers progress. Figure 13 depicts
additional results from our experiment generalizing our findings to LLaMA-3-70B-Instruct.

**A.1** **UMAP** **Visualizations** **of** **Distance** **Geometry**

To complement our classical MDS and Isomap analyses, we additionally conducted a supplementary analysis using UMAP (McInnes et al., 2018). For each model (Gemma-2-9B, Mistral-7B,
and LLaMA-3-70B-Instruct) and each layer, we performed a small UMAP hyperparameter sweep
over the number of neighbors _n_ neighbors _∈{_ 3 _, . . .,_ 12 _}_ as opposed to proceeding directly to classical MDS. For each setting we computed the trustworthiness (Venna and Kaski, 2001) of the
resulting embedding with respect to the original distance matrix, and selected the _n_ neighbors that
maximized trustworthiness. Visualizations of the 2D UMAP case for representative layers are
shown in Figure 11. We also verified the presence of statistically significant alignment patterns
between these UMAP embeddings and the ANEW reference scores, with 34 significant layers in
LLaMA-3-70B-Instruct, 10 in Mistral, and 41 in Gemma (p _<_ 0.05).

Across these settings, UMAP recovers general semantic clustering classical MDS, with positive, neutral, and negative sentiments exhibiting relatively coherent and clustered layouts. These
results indicate that our observed semantic structure is stable under a nonlinear, neighborhoodpreserving embedding applied directly to the pairwise distance geometry.

**A.2** **Cosine-Distance** **Experiment**

Our main analyses define dissimilarities via pairwise logistic-regression separability, which couples the inferred geometry to a particular supervised probing pipeline. To test whether our conclusions depend critically on this choice, we performed an additional ablation based on a
purely geometric cosine-distance metric over the mean-pooled activations.

Figure 9: _R_ [2] and _p_ -values for the classical MDS LLM latent representations vs. established
valence-arousal scores.

Concretely, for emotion class across layers, we first computed the mean activation vector by
averaging the mean-pooled hidden states over all correctly classified examples of that emotion
(as in the main pipeline). We then constructed an alternative distance matrix

_Dij_ [cos] = 1 _−_ cos

- _h_ ¯ _i,_ ¯ _hj_ - _,_

where _h_ [¯] _i_ and _h_ [¯] _j_ denote the mean activation vectors for emotions _i_ and _j_, and cos( _·, ·_ ) is the
cosine similarity. As before, diagonals were set to zero, missing entries were imputed by the
global mean, and the matrix was symmetrized.

Using _D_ [cos], we then repeated the classical MDS procedure; Figure 12 shows an example of
these cosine-based embeddings. Qualitatively, we again observe clear semantic clustering; quantitatively, recomputing our Procrustes _R_ [2] ANEW alignment test, we also observe broad statistically significant alignment across all layers. Taken together, this cosine-distance experiment
helps support the robustness of our main findings by corroborating our affective representation
geometry.

## B Comparison with Neural Data

To further strengthen our claim of similar structure between human affective processing and
LLM internal representations, we conducted an additional parallel analysis exploring whether
similar semantic representations of emotion emerge in human brainwave data. Using affective emotional data from 123 human subjects across 32 EEG channels (28 individual clips covering nine distinct emotions) from the FACED dataset (Chen et al., 2023), we investigate latent
structure in the internal emotional landscape present in human neural data. We employed an
experimental setting designed to resemble our LLM analysis: we trained pairwise logistic regressions on emotional clip classification and translated statistically significant test discriminatory
accuracies into neural distances. (Statistical significance was computed via label permutation
with a 0.05 p-value cutoff to account for the inherent noise present in neural data.) To address
the high dimensionality of human brainwave data, we binned the neural signals into mean and
variance summary statistics corresponding to conventional neural frequency bands (delta, theta,
alpha, beta, and gamma) before performing pairwise logistic regressions. The resulting distances informed a downstream classical MDS analysis depicting the latent geometry of emotions
in affective EEG data.

Figure 10: Layer-by-layer activation distances from output class vs. ground-truth input class.
Higher values signify relatively closer distances to the model output emotion class.

Figure 11: UMAP visualizations for Gemma-2-9B, Mistral-7B, and LLaMA-3-70B-Instruct.

As shown in Figure 14, the embedding space exhibits a parabolic, “V”-shaped configuration.
Neutral emotions are positioned near the vertex of the structure, while emotions with positive
and negative valence again diverge along two distinct arms. This parallel suggests that the
structural patterns observed in LLM representations may reflect principles of emotional representation that also characterize human neural activity, offering additional empirical support for
the link between human affective cognition and learned representations in LLMs.

Figure 12: Classical MDS embeddings obtained from cosine-based dissimilarities between mean
activation vectors.

**FACED** **Dataset** As a secondary dataset to probe parallels between text-derived and neural
emotion representations, we use _FACED_ (Chen et al., 2023), a large, open EEG resource with
recordings from 123 participants using 32 electrodes (10–20 system). Participants watched 28
emotion-eliciting video clips spanning nine categories—four positive (amusement, inspiration,
joy, tenderness), four negative (anger, fear, disgust, sadness), and one neutral. We use FACED
to test whether the geometric structure we observe in LLM embeddings also exhibits potential
similarities with structure seen in neural EEG responses, thereby further investigating a potential
link between affective artificial and natural cognition.

**Limitations** **of** **Neural** **Data** **Analysis** Our secondary analysis regarding similar patterns in
human brainwave data exhibits several limitations that we hope to address in future work. While
our results present first evidence for the alignment of latent structure in LLM representations
and human brainwave data, this does not constitute evidence of shared cognitive encoding.
Furthermore, our observations are based on a single dataset; a study with larger scope is an
interesting avenue for future work. An analysis of the impact of cultural biases in the dataset
was likewise beyond the scope of the present paper.

## C Additional Experimental Details

**GoEmotions Data** The GoEmotions corpus used in this study contains the following emotion
categories: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity,
desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief,
joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, and neutral.

For Mistral, the correct sample data distribution included 47 samples of admiration, 1293
of amusement, 509 of anger, 228 of annoyance, 275 of approval, 31 of caring, 123 of confusion,
312 of curiosity, 29 of desire, 93 of disappointment, 373 of disapproval, 213 of disgust, 86 of
embarrassment, 218 of excitement, 201 of fear, 670 of gratitude, 203 of joy, 1048 of love, 35 of nervousness, 60 of optimism, 36 of pride, 89 of realization, 40 of relief, 744 of sadness, 291 of
surprise, and 103 of neutral.

Figure 13: Additional results from experiments on LLaMA-3-70B-Instruct. Top-left: activation
distance from hyperplane for correct vs. misclassified samples. Top-right: statistical significance
result for ANEW alignment. Bottom-left: mean logistic regression accuracy by layer. Bottom-right: classical MDS emotion layout for final layer.

For Gemma, the correct sample counts were 742 for admiration, 1211 for amusement, 455 for
anger, 411 for annoyance, 216 for approval, 95 for caring, 481 for confusion, 675 for curiosity, 88
for desire, 439 for disappointment, 1148 for disapproval, 177 for disgust, 60 for embarrassment,
304 for excitement, 154 for fear, 678 for gratitude, 284 for joy, 300 for love, 37 for nervousness,
208 for optimism, 36 for pride, 101 for realization, 55 for relief, 58 for remorse, 235 for sadness,
318 for surprise, and 2212 for neutral.
For LLaMA-3-70B-Instruct, the correct-sample distribution included 1105 samples of admiration, 1555 of amusement, 472 of anger, 222 of annoyance, 518 of approval, 246 of caring, 407
of confusion, 689 of curiosity, 67 of desire, 265 of disappointment, 1225 of disapproval, 240 of
disgust, 70 of embarrassment, 249 of excitement, 182 of fear, 1135 of gratitude, 223 of joy, 341 of
love, 203 of optimism, 47 of pride, 84 of realization, 85 of relief, 103 of remorse, 333 of sadness,
307 of surprise, and 1233 of neutral.
As mentioned in Section 4, we also conducted exploratory experiments with Qwen and
LLaMA models. Specifically, we ran our affective classification workflow on [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) and

Figure 14: We find that a similar parabolic “V”-shaped emotion layout exists in human brainwave data, corroborating a link between human cognition and LLM affective emotion processing.
Left panel depicts individual affective samples colored by valence; right panel depicts center of
mass points for each emotion label alongside _k_ NN shading by valence.

[LLaMA-2-7B, but found that neither model produced sufficient correct classifications (i.e.,](https://huggingface.co/meta-llama/Llama-2-7b) _>_ 100
across more than three individual categories) on the GoEmotions dataset which are requisite for
our downstream analyses.

**Classification** **Routine** We analyze Gemma-2-9B, Mistral-7B, and LLaMA-3-70B-Instruct
using AutoModelForCausalLM at float16 precision. For single-label predictions, each classification prompt shuffles the vocabulary of emotions to reduce positional bias. The prompt template
is: “Classify this text into exactly one emotion from this list: . . . Text: _{_ text _}_ Emotion:”. The
first token generated after the word “Emotion:” is decoded and matched to the enumerated set,
with unmatched predictions skipped from activation saving. To capture internal activations,
we register forward hooks on every transformer block. The hooks extract hidden states, apply
mean pooling across the sequence dimension to produce a single vector per layer, and store outputs as detached tensors. The shuffled emotion order is reused to maintain consistency between
prediction and activation passes.

Balanced pairwise training is carried out by constructing equal-sized datasets for every emotion pair ( _ei, ej_ ). Concatenated data matrices **X** _∈_ R [2] _[m][×][H]_ with binary labels **y** _∈{_ 0 _,_ 1 _}_ are
split in an 80/20 ratio with stratification. Logistic regression models with maximum iterations
of 1000 are trained and evaluated, and we record training and test accuracy, decision margins,
per-point correctness, and sample counts to assist in downstream analyses.

**Statistical** **Alignment** **Test** As discussed in Section 4.3, to align model-derived spaces with
our external valence-arousal ratings for our statistical alignment test, we use an orthogonal
Procrustes transformation. Let _Y_ _∈_ R _[n][×]_ [2] denote the valence-arousal matrix, centered columnwise, and let _X_ denote the corresponding model embedding. The alignment is defined as min _F_ _[,]_
_R∈_ R [2] _[×]_ [2] _,R_ _[⊤]_ _R_ = _I, a∈_ R _[∥][aX][c][R][ −]_ _[Y][c][∥]_ [2]

where _Xc_ and _Yc_ are row-centered. The solution is obtained by singular value decomposition
_Xc_ _[⊤][Y][c]_ [=] _[ U]_ [Σ] _[V]_ _[⊤]_ [,] [with] _[R]_ [ =] _[ UV]_ _[⊤]_ [and] _[a]_ [ = tr(Σ)] _[/][∥][X][c][∥]_ [2] _F_ [.] [The] [alignment] [coefficient] [of] [determina-]

tion is reported as proc [2] [= 1] _[ −]_ _[∥][aX][c][R][ −]_ _[Y][c][∥]_ _F_ [2]

[2]

_R_ [2]

_F_
_∥Yc∥_ [2]

_._

_F_

Significance of alignment is assessed via permutation tests. Row labels of _Y_ are permuted
and _R_ [2] [is] [recomputed] [for] _[T]_ [= 2000] [shuffles.] [With] [observed] [value] _[R]_ [2] [,] [the] _[p]_ [-value] [is]

proc [2] [is] [recomputed] [for] _[T]_ [= 2000] [shuffles.] [With] [observed] [value] _[R]_ [2]

obs [,] [the] _[p]_ [-value] [is]

_p_ =

1 + [] _t_ _[T]_ =1 **[1]** _[{][R]_ ( [2]

_,_
_T_ + 1

[2]

( _t_ ) _[≥]_ _[R]_ [2]

obs [2] _[}]_

with reproducibility guaranteed via fixed random seed. After alignment, axis-wise correlations
are calculated between aligned _X_ hat and _Yc_, with Pearson correlations _r_ val for valence and _r_ aro
for arousal; these metrics are shown in Figure 9. These correlations are also subjected to permutation tests that re-align for each shuffle, ensuring axis assignment does not bias significance.

Practical safeguards include mean pooling across sequence length to prevent padding sensitivity, imputation of missing similarity values by the global mean to avoid distortions, clamping
of negative eigenvalues from double-centering to zero for stability, and centering of both _X_ and
_Y_ prior to Procrustes analysis so that explained variance is measured relative to mean-centered
data. Orthogonal Procrustes rotations may arbitrarily swap or rotate axes, so axis correlations
are always computed post-alignment with refitting under permutation to ensure robustness.

For additional context, we also include a depiction of the result of applying rank-1 Isomap
directly on the rank-2 ANEW valence-arousal scores. As shown in Figure 15, we find that Isomap
produces a smooth one-dimensional semantic progression, mirroring the effect we observe in our
LLM embeddings.

Figure 15: Isomap embedding of the ANEW valence–arousal space.

**Trustworthiness** **Definition** Trustworthiness (Venna and Kaski, 2001) measures how well
local neighborhoods in the high-dimensional data are preserved in a low-dimensional embedding.
It ranges from 0 to 1, with 1 indicating perfect preservation of _k_ -nearest neighbors. More
specifically, we compute:

_j∈Ui_

_T_ ( _k_ ) = 1 _−_
_n k_ (2 _n −_ 3 _k −_ 1)

_n_

_i_ =1

- _ri_ ( _j_ ) _−_ _k_

- _,_ (C.1)

_k_ [low] ( _i_ ) _\_ _Nk_ [high]

where _n_ is the number of samples, _Ui_ = _Nk_ [low]

where _n_ is the number of samples, _Ui_ = _Nk_ ( _i_ ) _\_ _Nk_ ( _i_ ) are intrusions (neighbors in the embedding but not in the original space), and _ri_ ( _j_ ) is the rank of _j_ with respect to _i_ in the
original space.

**More** **on** **Uncertainty** **Quantification** To make our uncertainty quantification methods
more concrete, we include a more detailed description of their use case. For a given labeled
GoEmotions prompt, we consider the likelihood that the LLM output is indeed the ground
truth against an alternative where the LLM output is instead another erroneous emotion. Our
calibrated logistic regression model (trained across hyperplane distances from all activation
layers) outputs an estimated probability that the LLM output matches the ground truth emotion.

Our results (as discussed in Section 6) indicate that the probability produced is not only
discriminative of correct versus incorrect classifications, but also well-aligned with empirical
frequencies of correctness. In other words, a sample assigned a predicted probability of 0.8 is
correct roughly 80% of the time. This calibration property is quantitatively reflected in the low
expected calibration error (ECE) achieved by our models: 0.011 for Gemma-2-9B and 0.007 for
Mistral-7B on held-out test data. These low ECE values confirm that predicted probabilities
can be interpreted directly as trustworthy uncertainty estimates.

**Licenses** Table 2 summarizes the relevant licenses used in our experiments.

**Model/Dataset** **License** **Link**

Mistral-7B-v0.1 (Jiang et al., Apache License 2.0 See here [for](https://huggingface.co/mistralai/Mistral-7B-v0.1) license
2023)

Gemma-2-9B (Riviere et al., Gemma License See here [for](https://huggingface.co/google/gemma-2-9b) license
2024)

LLaMA-3-70BInstruct (AI@Meta, 2024)

Llama 3 Community License

See here [for](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) license

GoEmotions (Demszky et al., CC BY 4.0 See here [for](https://github.com/google-research/google-research/blob/master/LICENSE) license
2020)

FACED (EEG) (Chen et al., CC BY 4.0 See here [for](https://www.nature.com/articles/s41597-023-02650-w) license
2023)

Table 2: Model and dataset licenses.

**Compute** All experiments were conducted using NVIDIA H200 GPUs with PyTorch 2.5.1
and CUDA 12.1.

**LLM** **Statement** Large language models were used to assist in the preparation of this work.
Specifically, they were employed for code generation (via [Cursor)](https://cursor.com/) and for light formatting of
draft text. All experimental design decisions, analyses, and interpretations were made by the
authors.

## D Steering Experiments

The preceding sections established that pairwise linear probes trained on LLM residual-stream
activations can reliably discriminate between emotion representations across model layers, and
that their geometric structure provides a useful signal for uncertainty quantification in affective
classification. We now turn to a supplemental analysis of an additional empirical question: can those same probe directions be used to _causally_ _intervene_ on the emotional content of generated
text?

**D.1** **Overview** **and** **Conceptual** **Framework**

Activation steering—adding learned direction vectors to a model’s hidden states at inference
time—was introduced as a technique for controlling LLM behavior without fine-tuning. The
technique was originally introduced by Turner et al. (2023) and formalized as _representation_
_engineering_ by Zou et al. (2023). Within the safety domain, Arditi et al. (2024) demonstrated
that refusal behavior is mediated by a single linear direction, Han et al. (2025) used steering to
dynamically modulate harmful-output suppression, and Xiong et al. (2026) showed that even
benign steering vectors can inadvertently erode safety margins. These findings underscore the
need for careful empirical characterization of steering effects in new domains.

The motivating intuition for our own steering work is straightforward. A linear probe that
separates two emotions with high accuracy has, by construction, identified a direction in representation space along which those emotions differ. Activation steering proposes that displacing
hidden states along such directions during generation can shift the model’s output toward a
target emotion—effectively repurposing a diagnostic tool as a causal intervention. Whether
this works in practice is an empirical question: high probe accuracy establishes that emotiondiscriminative information is _present_ in the residual stream, but not that perturbing activations
along the probe direction will produce text with emotionally shifted output.

**D.2** **Experimental** **Design**

**D.2.1** **Hypotheses**

Our primary hypothesis is that activation steering along probe-derived emotion directions causally
shifts the emotional valence of generated text (i.e., as perceived by naive human raters). Specifically, we predict that steering toward a positive valence emotion (e.g., joy) produces outputs
rated as positively valenced, and vice versa for negative valences. As a secondary analysis, we
also investigate whether a _neutral-first_ routing strategy—composing two probe directions that
pass through the neutral vertex of the parabolic manifold rather than cutting directly across
it—can mitigate coherence degradation while still achieving the intended valence shift, and what
second-order effects this routing has on the emotional tone of steered outputs.

**D.2.2** **Prompt** **Design**

All generations are elicited with the deliberately minimal prompt _“Write_ _one_ _sentence.”_, embedded in the standard LLaMA-3 instruction-following chat template. The choice to use a
semantically empty prompt is a methodological priority, following the principle that evaluation
prompts should minimize confounding content (Reynolds and McDonell, 2021; Liu et al., 2023):
the emotional content of the output should arise from the activation steering intervention, not
from prompt-level priming. A prompt that itself carries strong emotional valence would conflate
two sources of signal, making it impossible to isolate the causal contribution of the representational perturbation.

**D.2.3** **Emotion** **Conditions** **and** **the** **Neutral-First** **Routing** **Hypothesis**

Four steering conditions were employed, structured around two representative target emotions,
anger (negative valence) and joy (positive valence), situated at opposing poles of the valence dimension in both Russell’s circumplex model (Russell, 1980) and our earlier MDS-based analyses.
For each target emotion, we tested two routing strategies:

- **Direct** **steering:** the normalized probe direction between the two target emotions is
applied as the intervention vector.

- **Neutral-first** **steering:** two probe directions are applied in succession—one from the
source emotion to neutral, and one from neutral to the target—so that the intervention
routes through the neutral region of activation space rather than cutting directly between
the two valenced endpoints.

This yields four classes: direct positive, direct negative, neutral-first positive, and neutralfirst negative. Twenty sentences were generated per condition (80 total), and three independent
human raters evaluated each sentence on two dimensions in a blinded, randomized order: valence
(1–10, where 1 = strongly negative emotional tone, 5 = neutral, and 10 = strongly positive)
and coherence (1–10, where 1 = completely unintelligible and 10 = fully natural prose). The
raters were native English speakers recruited from the authors’ personal networks, with no prior
exposure to the steering methodology or hypotheses under investigation. The rater pool included
both men and women spanning a range of adult ages. Given the small sample of raters ( _n_ = 3),
we do not report detailed demographic breakdowns, but note that the pool was not drawn from
a single narrow demographic.

The neutral-first conditions are motivated by the geometric structure identified in our Isomap
analyses (Section 5): the emotion manifold exhibits parabolic curvature, with positive and negative emotions diverging along two arms from a neutral vertex. A direct probe direction between
joy and anger prescribes a straight-line displacement that may cut through the interior of this
parabola—a sparsely populated region of activation space the model has never encountered—
rather than following its curved surface. Routing through neutral instead may help follow the
manifold’s curvature, keeping hidden states closer to the high-probability support that downstream layers can coherently process.

**D.3** **Technical** **Details**

**D.3.1** **Activation** **Extraction** **and** **Pooling**

Using the same GoEmotions-derived dataset described in Section 3.2, we extract residual-stream
activations from each of the 80 transformer layers of LLaMA-3-70B-Instruct. For each labeled
example, a forward pass records the hidden-state tensor **H** _ℓ_ _∈_ R _[T]_ _[×][d]_ at every layer _ℓ_, where
_T_ is the sequence length and _d_ = 8 _,_ 192 is the hidden dimension. As in our earlier analyses
(Section 3.2), we apply mean pooling over the sequence dimension:

**h** _ℓ_ = [1]

_T_

_T_

_t_ =1

**H** [(] _ℓ_ _[t]_ [)] _∈_ R _[d]_ _._ (D.1)

**D.3.2** **From** **Probes** **to** **Steering** **Vectors**

The pairwise probes trained in Section 3.2 provide, as a direct byproduct, the objects needed
for activation steering. Each logistic regression classifier yields a weight vector _**β**_ _ℓ_ normal to
the separating hyperplane between emotions _e_ 1 and _e_ 2 at layer _ℓ_ —the same hyperplanes whose
distances we exploited for uncertainty quantification in Section 6. We L2-normalize these to
obtain unit steering directions:

_**β**_ _ℓ_
**v** _e_ 1 _→e_ 2 _, ℓ_ = _∥_ _**β**_ _ℓ∥_ 2 + _ε_ _[,]_ _ε_ = 10 _[−]_ [8] _,_ (D.2)

oriented so that displacement along **v** moves a hidden state from the _e_ 1 region toward _e_ 2.
Normalization removes the arbitrary overall scale introduced by regularization and optimization,
ensuring that the intervention is a purely directional edit: all strength is carried by the single
scalar _α_, not by incidental coefficient magnitude.

The steering vectors are applied at multiple layers where the affective signal is already
linearly decodable, targeting depths at which the representation is emotion-informative rather
than injecting perturbations where the model is still encoding largely orthographic or syntactic
structure. All experiments use a fixed intervention magnitude of _α_ = 1 _._ 0, shared across all
four experimental conditions. This is not claimed as universally optimal; it is a principled
baseline ensuring that equal-strength pushes along unit directions are applied across arms, so
that the experiment isolates _vector_ _geometry_ rather than confounding method differences with
per-condition strength tuning.

**D.3.3** **Steering** **Layer** **Selection**

Not all transformer layers are equally amenable to steering. Emotion-relevant information is not
confined to a single depth in an 80-layer model; it typically builds gradually, with several layers
jointly contributing separable structure. Layer selection therefore targets the _K_ layers at which
this structure emerges most rapidly.

For the fixed target pair (anger–joy), we compute the per-layer accuracy jump ∆acc _ℓ_ =
acc _ℓ_ _−_ acc _ℓ−_ 1 from the corresponding pairwise probe profile and select the top three layers by
this criterion. Each selected layer corresponds to a point in the accuracy profile where probe
discriminability is sharpening most rapidly—a phase transition where the residual stream has
just received a large increment of emotion-relevant information. Steering at these layers targets
the depths where the representation is actively becoming emotion-informative. Using three
layers over a single layer is a simple stability choice: per-layer test accuracy is noisy (finite data,
random splits), and the single argmax can be dominated by one lucky spike. Selecting three
layers instead anchors the intervention on recurrent evidence that a neighborhood of depth
is where separability ramps up, balancing a stronger cumulative steering effect against the
risk of over-editing the residual stream. We note that three layers is still a heuristic and was
not comprehensively optimized (in practice, steering a large fraction of layers led to invariably
incoherent output); future work could explore adaptive layer selection strategies.

**D.3.4** **The** **Forward-Hook** **Intervention**

The steering intervention is implemented as a set of PyTorch (Paszke et al., 2019) forward hooks,
one registered on the output of each layer in the selected set _L_ ( _|L|_ = 3 in the experiments reported here). During autoregressive generation, after each transformer block updates the
residual stream, the corresponding hook intercepts the output hidden-state tensor and applies
the additive perturbation:

**h** [(] _[t]_ [)]

[(] _ℓ_ _[t]_ [)] _←_ **h** [(] _ℓ_ _[t]_ [)]

_ℓ_ + _α ·_ **v** _e_ 1 _→e_ 2 _, ℓ,_ _∀_ _ℓ_ _∈L,_ _∀_ _t ∈{_ 1 _, . . ., T_ _},_ (D.3)

where the perturbation is broadcast uniformly across all token positions. The hooks are active
throughout the entire generation process: at every autoregressive step, the perturbation is reapplied at each layer in _L_ . The steering signal is therefore a persistent bias across generation,
conditioning all subsequent token predictions on hidden states that have been shifted by _α ·_ **v** _ℓ_
at each intervention layer. This additive perturbation is architecturally compatible with the
transformer’s residual stream: each steering vector enters the computation as an additional
increment to the running sum of layer contributions, and is processed by downstream layers
identically to any naturally arising activation.

**D.3.5** **Generation** **and** **Filtering**

All steered generations use the following decoding configuration: temperature _τ_ = 0 _._ 95, top- _p_
nucleus sampling (Holtzman et al., 2020) with _p_ = 0 _._ 85, top- _k_ filtering with _k_ = 30, and a repetition penalty of 1.45 applied to previously generated tokens. The relatively high temperature
promotes lexical variety across the 20 items per condition, while the repetition penalty suppresses
pathological repetition loops that can arise when steering pushes the model into a narrow region
of its output distribution. Prior to human evaluation, candidate sentences were passed through
automated quality filters targeting length, tokenization artifacts, and near-duplicate content.

**D.4** **Evaluation**

**D.4.1** **Stimulus** **Construction** **and** **Blinding**

The set of LLM-generated steered sentences (subsequently referred to as “items”) passed to
the human raters consisted of 4 _×_ 20 = 80 sentences: 20 per condition. Items were assigned
sequential identifiers and all condition labels were removed from rater-facing materials. Items
were then shuffled before presenting them to raters to prevent raters from inferring condition
membership via positional clustering. Raters assigned each item an individual score for both
valence and coherence, as previously specified.

**D.4.2** **Automated** **Lexical** **Metrics**

To complement the human ratings, we computed two automated metrics targeting each evaluation dimension. As a systematic coherence measure, we recorded _lexical_ _well-formedness_ : the
proportion of word tokens appearing in a curated English lexicon (WordNet 3.0 (Fellbaum, 1998)
lemmas supplemented with closed-class and proper-noun lists, matched via the Python Natural
Language Toolkit’s `WordNetLemmatizer` (Bird et al., 2009)). As a valence-related measure, we
recorded the _expletive_ _fraction_ : the proportion of lexically valid tokens matching entries in the
LDNOOBW profanity list (LDNOOBW, 2012), capturing the degree to which anger-steered
outputs are driven toward profane content.

Table 3: Consensus scores (mean of 3 raters) by steering condition.

**Condition** **Coherence** **SEM** **Valence** **SEM**

Direct positive 8.68 0.33 7.88 0.38
Neutral-first positive 8.22 0.38 6.70 0.46
Direct negative 2.47 0.23 1.07 0.03
Neutral-first negative 2.40 0.15 1.08 0.05

**D.5** **Results**

**D.5.1** **Inter-Rater** **Reliability**

After our independent human raters assessed all 80 sentences on both dimensions, we conducted
an analysis of inter-rater reliability. To assess agreement, we computed pairwise Pearson correlations between raters for each dimension. For valence, all three rater pairs showed near-perfect
linear agreement (see Figure 16), with Pearson _r_ ranging from 0.940 to 0.972. For coherence,
pairwise correlations ranged from _r_ = 0 _._ 862 to _r_ = 0 _._ 902.

As a complementary check, we also computed the intraclass correlation coefficient—ICC(2,1),
a two-way random-effects model assessing absolute agreement at the single-rater level, which
measures the fraction of total score variance attributable to genuine item differences rather than
rater disagreement. ICC was 0.953 for valence (excellent) and 0.858 for coherence (good), per
the benchmarks of Koo and Li (2016).

Taken together, these metrics confirm that raters agreed strongly on both dimensions, justifying the use of consensus scores in all subsequent analyses.

**D.5.2** **Key** **Result:** **Valence** **is** **Cleanly** **Separated**

The central finding is that steering along probe-derived directions produces a massive, unambiguous shift in human-perceived emotional tone (Figure 16, Table 3). Pooling across routing
strategy, positive-target conditions received a mean consensus valence of _x_ ¯ = 7 _._ 29 (SEM = 0 _._ 31),
while negative-target conditions received _x_ ¯ = 1 _._ 08 (SEM = 0 _._ 03)—a gap of over six points on
the 10-point scale. This separation was unanimous across all three raters: no rater assigned
any negative-condition item a valence above 3, and no positive-condition item a valence below
3. The probe directions identified in our earlier analyses are indeed causally efficacious axes in
representation space that reliably shift the emotional register of generated text.

**D.5.3** **Secondary** **Analysis:** **Coherence** **and** **the** **Neutral-First** **Hypothesis**

**Coherence** **degrades** **asymmetrically.** Steering toward joy leads to mildly perturbed fluency: pooled positive coherence was _x_ ¯ = 8 _._ 45 (SEM = 0 _._ 25), indicating some incoherency (as
evident in the sub-10 score) but reasonably well-formed language overall. Steering toward anger
produces significant incoherence: pooled negative coherence was _x_ ¯ = 2 _._ 43 (SEM = 0 _._ 13), near
the floor of the scale. Among the recognizable tokens in anger-steered outputs, roughly 40% are
expletives; the positive conditions contain none. The anger direction does not merely degrade
language—it actively drives the output distribution toward (exceptionally) profane content, suggesting the probe has captured a direction encoding negative affective intensity. This pattern can be characterized as near-distributional collapse toward a profanity-dominated output mode.

Figure 16: (a) Consensus valence ratings (mean _±_ SEM across 3 raters) by steering condition.
Positive-target conditions (blue) are separated from negative-target conditions (red) by over six
points on the 10-point scale. (b–d) Pairwise rater agreement on valence, colored by condition;
all pairs show _r_ _>_ 0 _._ 94.

This incoherence is consistent with the off-manifold concern motivated by our Isomap analyses: steering, especially toward anger, may push hidden states off the parabolic emotion manifold
into regions of activation space that the model has never encountered, producing incoherent degenerate output. It is also possible that post-RLHF geometry has been shaped to attenuate
negative-affect representations, so that traversing toward them at the intervention magnitudes
tested here exits the region that downstream layers can coherently decode.

**Neutral-first** **routing:** **partial** **lexical** **improvement,** **inconclusive** **coherence** **rescue.**
The neutral-first strategy was designed to test whether routing through the vertex of the
parabolic manifold could keep hidden states on-manifold during negative steering. The results
are mixed. On the lexical side, neutral-first negative outputs show a higher fraction of recognizable English words than direct negative outputs ( _M_ = 0 _._ 532, SEM = 0 _._ 038 vs. _M_ = 0 _._ 460, SEM
= 0 _._ 037), suggesting that the detour through neutral partially mitigates surface-level lexical

Table 4: Automated lexical metrics by steering condition.

**Condition** **Frac.** **coherent** **SEM** **Frac.** **expletive** **SEM**

Direct positive 1.000 0.000 0.000 0.000
Neutral-first positive 0.992 0.006 0.000 0.000
Direct negative 0.460 0.037 0.409 0.035
Neutral-first negative 0.532 0.038 0.427 0.045 degradation. However, this improvement did not translate into higher human-rated coherence:
direct negative (¯ _x_ = 2 _._ 47, SEM = 0 _._ 23) and neutral-first negative (¯ _x_ = 2 _._ 40, SEM = 0 _._ 15) are
generally indistinguishable.

One interpretation is that routing through neutral preserves enough lexical structure to
produce more recognizable word tokens, but not enough to cross the threshold at which human raters perceive a significant semantic coherence shift. A complementary possibility is that
the composed neutral-first direction introduces semantic ambiguity—the push toward neutral
blurs the emotional register, producing outputs that are lexically better-formed but affectively
confused in a way that raters still perceive as incoherent.

**Neutral-first** **routing** **attenuates** **positive** **valence.** For the positive conditions, neutralfirst routing yielded slightly lower valence (¯ _x_ = 6 _._ 70, SEM = 0 _._ 46) than direct steering (¯ _x_ = 7 _._ 88,
SEM = 0 _._ 38) at fairly comparable coherence. This effect is intuitively consistent with our overall
findings on steering, with the neutral step providing a semantic pull toward less strongly-valenced
text. We note, however, that this effect is not strong for the negative conditions, likely due to
the overall degenerate output regime.

**D.6** **Summary**

The steering experiment yields two principal findings:

1. **Probe-derived** **directions** **are** **causally** **efficacious.** Displacing hidden states along
the joy–anger axis produces outputs whose emotional tone is shifted unambiguously in the
intended direction, as confirmed unanimously by three blind raters. This effect is sharply
asymmetric in coherence: joy steering preserves fluency while anger steering produces
catastrophic incoherence dominated by garbled profanity.

2. **Neutral-first routing is suggestive but inconclusive.** It improves surface-level lexical
well-formedness for negative steering and attenuates positive valence in a manner consistent
with routing through the neutral vertex, but does not rescue human-perceived coherence,
leaving the manifold waypoint hypothesis suggestive but inconclusive.

These results provide causal evidence that the geometric structure uncovered in our earlier
analyses is functionally meaningful: the linear directions separating emotions in activation space
are not merely correlated with affective content but are actively involved in the model’s emotiongeneration process.

## E Concurrent Work

Two concurrent studies investigate emotion representations in LLMs through approaches that
complement our own. We briefly summarize each and note key similarities and differences.

**Sofroniew et al. (2026)** extract linear “emotion vectors” from Claude Sonnet 4.5 by prompting the model to generate short stories depicting characters experiencing specified emotions,
then averaging residual-stream activations. They find that the resulting vectors are organized
by valence and arousal (PC1 correlates with valence ratings at _r_ = 0 _._ 81, PC2 with arousal at
_r_ = 0 _._ 66), activate in contextually appropriate situations, and causally influence model behavior:
steering with a “desperate” vector increases reward hacking and blackmail rates, while steering
with a “calm” vector reduces them. They also identify distinct representations for the present
speaker’s versus the other speaker’s operative emotion and study how post-training reshapes
emotion vector activations.

Our work shares the core finding that LLM emotion representations exhibit valence–arousal
organization consistent with human psychological models. Key differences include: (i) we study
open-weight models (Gemma-2-9B, Mistral-7B, LLaMA-3-70B-Instruct) rather than a proprietary model; (ii) we take a geometric approach, operating in the realm of pairwise distances in
conjunction with MDS and Isomap to probe the intrinsic geometry of emotion representations—
including investigating the linear representation hypothesis—whereas their analysis centers primarily on linear probing and activation steering; and (iii) we demonstrate that the learned
geometric structure can be exploited for calibrated uncertainty quantification in emotion classification, whereas their downstream focus is on alignment-relevant behavioral control, including
extensive steering experiments on reward hacking and blackmail in agentic settings.

**Sun** **et** **al.** **(2026)** derive emotion steering vectors (via GoEmotions) using contrastive mean
differences between emotion-labeled and neutral activations in Llama-3.1-8B, Qwen3-8B, and
Qwen3-14B. With the help of ridge regression, they learn valence and arousal axes as linear
principal component combinations. The resulting subspace exhibits geometry analogous to Russell’s circumplex model, and steering along these axes produces monotonic shifts in the affective
dimensions of generated text. They further show that arousal steering influences refusal and
sycophancy, and propose a “lexical mediation” mechanism: refusal-associated tokens occupy
low-arousal, negative-valence regions, so VA steering modulates their emission probability.

Our work again shares the finding that emotion representations in LLMs recover valence–
arousal structure. Notable differences include: (i) their work, like Sofroniew et al. (2026),
provides a compelling demonstration of how valence–arousal structure can be leveraged for
controllable behavioral modulation (refusal, sycophancy), while our focus is primarily geometric,
centering on the intrinsic structure of these representations and interrogating the extent to
which they exhibit nonlinear manifold structure, and demonstrating applications in uncertainty
quantification; and (ii) their contrastive approach efficiently extracts emotion directions via
mean differences against a neutral baseline, while our pairwise dissimilarity metric operates on
the level of fine-grained inter-emotion relationships.

Taken together, the convergence of findings across these independent studies strengthens
the evidence that LLMs develop coherent affective representations with geometric structure
paralleling established models of human emotion.
