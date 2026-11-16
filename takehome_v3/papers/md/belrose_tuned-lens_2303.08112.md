# Eliciting Latent Predictions from Transformers with the Tuned Lens

- **Authors:** Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, Jacob Steinhardt
- **Date:** 2023-03-14
- **Source:** https://arxiv.org/abs/2303.08112
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `belrose_tuned-lens_2303.08112`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Nora Belrose** [1 2] **Igor Ostrovsky** [1] **Lev McKinney** [3 2] **Zach Furman** [1 4] **Logan Smith** [1] **Danny Halawi** [1]

**Stella Biderman** [1] **Jacob Steinhardt** [5]

**Abstract**

We analyze transformers from the perspective of
iterative inference, seeking to understand how
model predictions are refined layer by layer. To
do so, we train an affine probe for each block in
a frozen pretrained model, making it possible to
decode every hidden state into a distribution over
the vocabulary. Our method, the _tuned lens_, is a
refinement of the earlier “logit lens” technique,
which yielded useful insights but is often brittle.

We test our method on various autoregressive
language models with up to 20B parameters,
showing it to be more predictive, reliable and
unbiased than the logit lens. With causal experiments, we show the tuned lens uses similar features to the model itself. We also
find the trajectory of latent predictions can be
used to detect malicious inputs with high accuracy. All code needed to reproduce our re[sults can be found at https://github.com/](https://github.com/AlignmentResearch/tuned-lens)
[AlignmentResearch/tuned-lens.](https://github.com/AlignmentResearch/tuned-lens)

## 1. Introduction

The impressive performance of transformers in natural language processing (Brown et al., 2020) and computer vision
(Dosovitskiy et al., 2020) suggests that their internal representations have rich structure worthy of scientific investigation. One common approach is to train classifiers to extract
specific concepts from hidden states, like part-of-speech
and syntactic structure (Hewitt and Manning, 2019; Tucker
et al., 2021; Li et al., 2022).

In this work, we instead examine transformer representations from the perspective of _iterative inference_ (Jastrz˛ebski
et al., 2017). Specifically, we view each layer in a transformer language model as performing an incremental update
to a latent prediction of the next token. [1] We decode these latent predictions through early exiting, converting the hidden

1Eleuther AI 2FAR AI 3University of Toronto 4Boston
University [5] UC Berkeley. Correspondence to: Nora Belrose
<nora@eleuther.ai>.

1See Appendix C for evidence supporting this view, including
novel empirical results of our own.

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|~~del~~ ~~y~~|~~t~~ ~~t~~|~~at~~ ~~t~~|~~at~~ ~~cal~~|~~ed~~ ~~atte~~|~~tion~~ ~~for~~|~~er~~ |
|~~del~~ ~~mo~~ ~~del~~ ~~mo~~|~~el~~ ~~cal~~ ~~del~~ ~~th~~|~~ed~~ ~~cal~~ ~~at~~ ~~t~~|~~ed~~ ~~cal~~ ~~at~~ ~~cal~~|~~ed~~ ~~*~~ ~~ed~~ ~~*~~|~~*~~  ~~*~~ ||
||||||||
|~~del~~ ~~mo~~|~~el~~ ~~th~~|~~at~~ ~~t~~|~~at~~ ~~cal~~|~~ed~~ ~~â~~|~~¦."~~ ~~â~~|~~¦."~~ |
|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|
||||||||
|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|~~¦."~~ ~~â~~ ~~¦."~~ ~~â~~|
|~~¦."~~ ~~â~~  |~~¦."~~ ~~â~~  |~~¦."~~ ~~â~~  |~~¦."~~ ~~â~~  |~~¦."~~ ~~â~~  |~~¦."~~ ~~â~~  |~~¦."~~ ~~â~~  |
||||||||
|~~¦."~~ ~~â~~|~~¦."~~ ~~â~~|~~¦."~~ ~~â~~|~~¦."~~ ~~â~~|~~¦."~~ ~~â~~|~~¦."~~ ~~â~~|~~¦."~~ ~~â~~|

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|~~del~~ ~~mo~~|~~el~~ ~~archit~~|~~cture~~ ~~t~~|~~at~~ ~~cal~~|~~ed~~ ~~atte~~|~~tion~~ ~~for~~|~~er~~ |
|~~del~~ ~~mo~~ ~~del~~ ~~mo~~|~~el~~ ~~th~~ ~~del~~ ~~th~~|~~at~~ ~~t~~ ~~at~~ ~~t~~|~~at~~ ~~cal~~ ~~at~~ ~~cal~~|~~ed~~ ~~*~~ ~~ed~~ ~~*~~|~~*~~  ~~*~~ ||
||||||||
|~~del~~ ~~mo~~|~~el~~ ~~th~~|~~at~~ ~~t~~|~~at~~ ~~t~~|~~e~~ ~~â~~|||
|~~hod~~ ~~met~~ ~~del~~ ~~mo~~|~~hod~~ ~~t~~ ~~del~~ ~~th~~|~~at~~ ~~t~~ ~~at~~ ~~th~~|~~at~~ ~~wh~~ ~~at~~ ~~t~~|~~ch~~ ~~s~~ ~~e~~ ~~s~~|~~o~~ ||
||||||||
|~~d~~ ~~a~~ ~~hod~~ |~~d~~ ~~th~~  ~~th~~|~~at~~ ~~f~~ ~~at~~ ~~t~~|~~r~~ ~~wh~~ ~~at~~ ~~wh~~|~~ch~~ ~~re~~ ~~ch~~ ~~s~~|~~ult~~   ||
|~~t~~  |||~~a~~  |~~d~~  |||
||||||||
||~~t~~||~~a~~|~~d~~ ~~sa~~|~~e~~ ||

Figure 1: Comparison of our method, the _tuned lens_ (bottom), with

the “logit lens” (top) for GPT-Neo-2.7B prompted with an except
from the abstract of Vaswani et al. (2017). Each cell shows the
top-1 token predicted by the model at the given layer and token
index. The logit lens fails to elicit interpretable predictions before
layer 21, but our method succeeds.

state at each intermediate layer into a distribution over the
vocabulary. This yields a sequence of distributions we call
the _prediction trajectory_, which exhibits a strong tendency
to converge smoothly to the final output distribution, with
each successive layer achieving lower perplexity.

We build on the “logit lens” (nostalgebraist, 2020), an early
exiting technique that directly decodes hidden states into
vocabulary space using the model’s pretrained unembedding matrix. We find the logit lens to be unreliable (Section 2), failing to elicit plausible predictions for models like
BLOOM (Scao et al., 2022) and GPT Neo (Black et al.,
2021). Even when the logit lens appears to work, its outputs
are hard to interpret due to _representational drift_ : features output input output input

Logit Lens (theirs)

new simple network architecture, the Trans former

Tuned Lens (ours)

new simple network architecture, the Trans former

Input token

**Probability**
**0** **0.2** **0.4** **0.6** **0.8** **1**

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

may be represented differently at different layers of the network. Other early exiting procedures also exist (Schuster
et al., 2022), but require modifying the training process, and
so can’t be used to analyze pretrained models.

To address the shortcomings of the logit lens, we introduce
the _tuned lens_ . We train _L_ affine transformations, one for
each layer of the network, with a distillation loss: transform
the hidden state so that its image under the unembedding
matches the final layer logits as closely as possible. We
call these transformations _translators_ because they “translate” representations from the basis used at one layer of the
network to the basis expected at the final layer. Composing a translator with the pretrained unembedding yields a
probe (Alain and Bengio, 2016) that maps a hidden state to
a distribution over the vocabulary.

We find that tuned lens predictions have substantially lower
perplexity than logit lens predictions, and are more representative of the final layer distribution. We also show that the
features most influential on the tuned lens output are also
influential on the model itself (Section 4). To do so, we introduce a novel algorithm called _causal basis extraction_ (CBE)
and use it to locate the directions in the residual stream
with the highest influence on the tuned lens. We then ablate
these directions in the corresponding model hidden states,
and find that these features tend to be disproportionately
influential on the model output.

We use the tuned lens to gain qualitative insight into the
computational process of pretrained language models, by
examining how their latent predictions evolve during a forward pass (Figure 1, Section B)

Finally, we apply the tuned lens in several ways: we extend
the results of Halawi et al. (2023) to new models (Section
5.2), we find that tuned lens prediction trajectories can be
used to detect prompt injection attacks (Perez and Ribeiro,
2022) often with near-perfect accuracy (Section 5.3), and
find that data points which require many training steps to
learn also tend to be classified in later layers (Section 5.4).

## 2. The Logit Lens

The logit lens was introduced by nostalgebraist (2020), who
found that when the hidden states at each layer of GPT-2
(Radford et al., 2019) are decoded with the unembedding
matrix, the resulting distributions converge roughly monotonically to the final answer. More recently it has been used
by Halawi et al. (2023) to understand how transformers
process few-shot demonstrations, and by Dar et al. (2022),
Geva et al. (2022), and Millidge and Black (2022) to directly
interpret transformer weight matrices.

**The method.** Consider a pre-LayerNorm transformer [2] _M_ .

2Both the logit lens and the tuned lens are designed primarily

The logit lens consists of setting the residuals to zero:

LogitLens( _**h**_ _ℓ_ ) = LayerNorm[ _**h**_ _ℓ_ ] _WU_ _._ (3)

While this simple form of the logit lens works reasonably
well for GPT-2, nostalgebraist (2021) found that it fails to
extract useful information from the GPT-Neo family of models (Black et al., 2021). In an attempt to make the method for the pre-LN architecture, which is more unambiguously iterative.
Luckily pre-LN is by far more common than post-LN among stateof-the-art models. See Zhang and He (2020, Appendix C) for more
discussion.

translator layer (e.g. _h_ 1 above), and applies a learned affine transformation
(the translator). We then convert the hidden state into logits with
the unembedding layer.

Figure 2: The tuned lens takes the hidden state at an intermediate

We’ll decompose _M_ into two “halves,” _M≤ℓ_ and _M>ℓ_ .
The function _M≤ℓ_ consists of the layers of _M_ up to and
including layer _ℓ_, and it maps the input space to hidden
states. Conversely, the function _M>ℓ_ consists of the layers
of _M_ after _ℓ_, which map hidden states to logits.

The transformer layer at index _ℓ_ updates the representation
as follows:

_**h**_ _ℓ_ +1 = _**h**_ _ℓ_ + _Fℓ_ ( _**h**_ _ℓ_ ) _,_ (1)

where _Fℓ_ is the residual output of layer _ℓ_ . Applying Equation 1 recursively, the output logits _M>ℓ_ can be written as
a function of an arbitrary hidden state _**h**_ _ℓ_ at layer _ℓ_ :

_L_

_ℓ_ _[′]_ = _ℓ_

_Fℓ′_ ( _**h**_ _ℓ′_ ) _WU_ _._ (2)

residual update

_M>ℓ_ ( _**h**_ _ℓ_ ) = LayerNorm _**h**_ _ℓ_ +

_**h**_ _ℓ_ +

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

w/o final layer w/ final layer input 5 10 15 20 25 30

Layer final layer output for GPT-Neo-2.7B. The last transformer layer is
included for both probes. Unlike the tuned lens, the logit lens is
systematically biased toward some vocabulary items over others
until the very end of the network.

Figure 3: Bias of logit lens and tuned lens outputs relative to the

work better for GPT-Neo, they introduce an extension which
retains the last transformer layer, yielding:

LogitLens [ext] ( _**h**_ _ℓ_ ) = LayerNorm[ _**h**_ _ℓ_ + _FL_ ( _**h**_ _ℓ_ )] _WU_ (4)

This extension is only partially successful at recovering
meaningful results; see Figure 1 (top) for an example.

**Unreliability.** Beyond GPT-Neo, the logit lens struggles to
elicit predictions from several other models released since
its introduction, such as BLOOM (Scao et al., 2022) and
OPT 125M (Zhang et al., 2022) (Figure 14).

Moreover, the type of information extracted by the logit lens
varies both from model to model and from layer to layer,
making it difficult to interpret. For example, we find that
for BLOOM and OPT 125M, the top 1 prediction of the
logit lens is often the _input_ token, rather than any plausible
continuation token, in more than half the layers (Figure 18).

**Bias.** Even when the logit lens is useful, we find that it is
a _biased_ estimator of the model’s final output: it systematically puts more probability mass on certain vocabulary
items than the final layer does.

This is concerning because it suggests we can’t interpret
the logit lens prediction trajectory as a belief updating in
response to new evidence. The beliefs of a rational agent
should not update in an easily predictable direction over
time (Yudkowsky, 2007), since predictable updates can be
exploited via Dutch books (Ramsey, 1926; De Finetti et al.,
1937; Hacking, 1967; Garrabrant et al., 2016). Biased logit
lens outputs are trivially exploitable once the direction of
bias is known: one could simply “bet” against the logit lens
at layer _ℓ< L_ that the next token will be one of the tokens
that it systematically downweights, and make unbounded
profit in expectation.

Let _**x**_ be a sequence of tokens sampled from a dataset _D_,
and let _**x**_ _<t_ refer to the tokens preceding position _t_ in the

Layer under four conditions: the logit lens (red squares) and the tuned
lens (blue circles), and including (left) and excluding (right) the
final transformer layer from the probe. We find that tuned lens
predictions have substantially lower perplexity whether or not the
final layer is included, showing it is an independent and complementary proposal.

Figure 4: Perplexity of predictions elicited from BLOOM 560M

sequence. Let _qℓ_ ( _·|_ _**x**_ _<t_ ) be the logit lens distribution at
layer _ℓ_ for position _t_, and let _p_ ( _·|_ _**x**_ _<t_ ) be the final layer
distribution for position _t_ .

We define _p_ ( _v|_ _**x**_ ) to be the probability assigned to a vocabulary item _v_ in a sequence _**x**_, averaged over all positions
1 _. . . T_ :

_T_

- _p_ ( _v|_ _**x**_ _<t_ ) _._ (5)

_t_ =1

_p_ ( _v|_ _**x**_ )

def
= [1]

_T_

Slightly abusing terminology, we say that _qℓ_ is an “unbiased
estimator” of _p_ if, for every item _v_ in the vocabulary, the
probability assigned to _v_ averaged across all tokens in the
dataset is the same:

- _p_ ( _v|_ _**x**_ )

E
_**x**_ _∈D_

- _qℓ_ ( _v|_ _**x**_ ) = E

_**x**_ _∈D_

_qℓ_ ( _v_ ) = _p_ ( _v_ )

(6)

_∀v_ _∈V,_ _V_ = _{_ "aardvark" _, . . .}_

In practice, Equation 6 will never hold exactly. We measure
the degree of bias using the KL divergence between the
marginal distributions, _DKL_ ( _p || qℓ_ ).

In Figure 3 we evaluate the bias for each layer of GPT-Neo2.7B. We find the bias of the logit lens can be quite large:
around 4 to 5 bits for most layers. As a point of comparison,
the bias of Pythia 160M’s final layer distribution relative to
that of its larger cousin, Pythia 12B, is just 0.0068 bits.

## 3. The Tuned Lens

One problem with the logit lens is that, if transformer layers
learn to output residuals that are far from zero _on average_,
the input to LogitLens may be out-of-distribution and yield
nonsensical results. In other words, the choice of zero as

|Eliciting Latent Predictions from|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Mod 7|el size 0M|
|---|---|---|---|---|---|---|---|---|---|
|||||||||4 1|60M 10M .4B|
|||||||||2 6 |.8B .9B 2B|
|||||||||2 (|0B NeoX)|
|||||||||||
|||||||||||

Layer as a function of layer index and model size. Tuned lens predictions are uniformly lower perplexity and exhibit lower variance across
independently trained models.

Figure 5: Perplexity of latent predictions elicited by the logit lens (left) and the tuned lens (right) from Pythia and GPT-NeoX-20B,

a replacement value is somewhat arbitrary– the network
might learn to rely on [] _ℓ_ _[L][′]_ = _ℓ_ [E][[] _[F][ℓ]_ [(] _**[h]**_ _[ℓ]_ [)]][ as a bias term.]

Our first change to the method is to replace the summed
residuals with a learnable constant value **b** _ℓ_ instead of zero:

We refer to ( _Aℓ,_ **b** _ℓ_ ) as the _translator_ for layer _ℓ_ .

**Loss** **function.** We train the translators to minimize KL
between the tuned lens logits and the final layer logits:

- _DKL_ ( _f>ℓ_ ( _**h**_ _ℓ_ ) _||_ TunedLens _k_ ( _**h**_ _ℓ_ )) (9)

LogitLens [debiased] _ℓ_ ( _**h**_ _ℓ_ ) = LogitLens( _**h**_ _ℓ_ + **b** _ℓ_ ) (7)

argmin E

_**x**_

**Representation drift.** Another issue with the logit lens is
that transformer hidden states often contain a small number of very high variance dimensions, and these “rogue
dimensions” (Timkey and van Schijndel, 2021) tend to be
distributed unevenly across layers; see Figure 6 (top) for an
example. Ablating an outlier direction can drastically harm
performance (Kovaleva et al., 2021), so if LogitLens relies
on the presence or absence of particular outlier dimensions,
the perplexity of logit lens predictions might be spuriously
high.

Even when controlling for rogue dimensions, we observe a
strong tendency for the covariance matrices of hidden states
at different layers to drift apart as the number of layers separating them increases (Figure 6, bottom). The covariance
at the final layer often changes sharply relative to previous
layers, suggesting the logit lens might “misinterpret” earlier
representations.

One simple, general way to correct for drifting covariance
is to introduce a learnable change of basis matrix _Aℓ_, which
learns to map from the output space of layer _ℓ_ to the input
space of the final layer. We have now arrived at the _tuned_
_lens_ formula, featuring a learned affine transformation for
each layer:

TunedLens _ℓ_ ( _**h**_ _ℓ_ ) = LogitLens( _Aℓ_ _**h**_ _ℓ_ + **b** _ℓ_ ) (8)

where _f>ℓ_ ( _**h**_ _ℓ_ ) refers to the rest of the transformer after
layer _ℓ_ . This can be viewed as a distillation loss, using the
final layer distribution as a soft label (Sanh et al., 2019). It
ensures that the probes are not incentivized to learn extra
information over and above what the model has learned,
which can become a problem when training probes with
ground truth labels (Hewitt and Liang, 2019).

**Implementation details.** When readily available, we train
translators on a slice of the validation set used during pretraining, and use a separate slice for evaluation. Since
BLOOM and GPT-2 do not have publicly available validation sets, we use the Pile validation set (Gao et al., 2020;
Biderman et al., 2022). The OPT validation set is also not
publicly available, but a member of the OPT team helped us
train a tuned lens on the OPT validation set. Documents are
concatenated and split into uniform chunks of length 2048.

We evaluate all models on a random sample of 16.4M tokens
from their respective pretraining validation sets. We leave
out the final transformer layer for GPT-2 (Radford et al.,
2019), GPT-NeoX-20B (Black et al., 2022), OPT (Zhang
et al., 2022), and Pythia (Biderman et al., 2023), and include
it for GPT-Neo (Black et al., 2021). We evaluate BLOOM
(Scao et al., 2022) under both conditions in Figure 4.

In our initial experiments, we used SGD with Nesterov
momentum, with a linear learning rate decay schedule over

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

all principal components

0 5 10 15 20 25 30 35 w/o top 2 components

0 5 10 15 20 25 30 35

Layer

0.8

0.6

0.4

0.2 across layers of Pythia 12B. Layer 4 introduces two outlier dimensions which dominate the covariance; removing them reveals
smooth representational drift with depth. To control for varying
hidden state norms, we measure the Frobenius cosine similarity, or

Figure 6: Pairwise similarities of hidden state covariance matrices

_⟨A,B⟩F_
_∥A∥F ∥B∥F_ [for two matrices] _[ A]_ [ and] _[ B]_ [.]

250 training steps. We used a base learning rate of 1.0,
or 0.25 when keeping the final transformer layer, and clip
gradients to a norm of 1. We accumulated gradients as
necessary to achieve a total batch size of 2 [18] tokens per
optimizer step. We initialized all translators to the identity
transform, and use a weight decay of 10 _[−]_ [3] .

**Muon** **optimizer.** After the initial publication of this paper, Jordan et al. (2024) introduced the Muon optimizer.
Using a fast orthogonalization algorithm, Muon attempts
to maximize the effective rank (Roy and Vetterli, 2007) of
the update applied to each matrix-valued parameter, thereby
counteracting the well-known tendency of neural network
gradients to be rank-deficient. Conceptually, Muon is the
opposite of LoRA (Hu et al., 2022), a popular parameter efficient finetuning method that constrains parameter updates to be low rank. Liu et al. (2025) find that Muon
accelerates the training of modern large language models
compared to AdamW.

Strikingly, we find that using Muon instead of SGD for
training the tuned lens _dramatically_ accelerates training, allowing us to achieve much lower KL divergence losses than
were possible in our initial experiments. Essentially, all of
the tuned lenses trained and evaluated in the earlier versions
of this paper were severely undertrained. The weight matrices of Muon-trained lenses have Frobenius norms that
are often several times larger than SGD-trained ones, showing that they are much further from the logit lens than our
original lenses— they are much better “tuned,” as it were.
We opted not to re-do all our experiments with tuned lenses
trained using Muon, but we encourage practitioners to use
Muon in future experiments.

**Results.** We plot tuned lens perplexity as a function of depth
for the Pythia models and GPT-NeoX-20B in Figure 5 [3] ;
results for other model families can be found in Appendix A.

We find that the tuned lens resolves the problems with the
logit lens discussed in Section 2: it has significantly lower
bias (Figure 3), and much lower perplexity than the logit
lens across the board (Figure 5, Appendix A).

**Transferability across layers.** We find that tuned lens translators can usually zero-shot transfer to nearby layers with
only a modest increase in perplexity. Specifically, we define
the _transfer_ _penalty_ from layer _ℓ_ to _ℓ_ _[′]_ to be the expected
increase in cross-entropy loss when evaluating the tuned
lens translator trained for layer _ℓ_ on layer _ℓ_ _[′]_ .

We report transfer penalties for the largest Pythia model in
Figure 7: Overall, transfer penalties are quite low, especially
for nearby layers (entries near the diagonal in Figure 7).
Comparing to the two plots in Figure 6, we notice that
transfer penalties are strongly negatively correlated with covariance similarity (Spearman _ρ_ = _−_ 0 _._ 78). Unlike Figure 6,
however, Figure 7 is not symmetric: transfer penalties are
higher when training on a layer with the outlier dimensions
(Layer 5 and later) and testing on a layer without them, than
the reverse.

**Transfer to fine-tuned models.** We find that lenses trained
on a base model transfer well to fine-tuned versions of that
base model, with no additional training of the lens. Transferred lenses substantially outperform the Logit Lens and
compare well with lenses trained specifically on the finetuned models. Here a transferred lens is one that makes use
of the fine-tuned models unembedding, but simply copies

3Pythia and GPT-NeoX-20B were trained using the same architecture, data, and codebase (Andonian et al., 2021). While they’re
not officially the same model suite, they’re more consistent than
the OPT models.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

0 5 10 15 20 25 30 35

Test layer

1.6

1.4

1.2

0.8

0.6

0.4

0.2 to a single tuned lens probe trained on layer _ℓ_, and each column
is a layer _ℓ_ _[′]_ on which probes are evaluated. Each cell shows the
cross-entropy loss of probe _ℓ_ evaluated on layer _ℓ_ _[′]_, _minus_ its on-distribution loss (so that the diagonal entries are identically zero).

Figure 7: Transfer penalties for Pythia 12B. Each row corresponds

its affine translators from the lens trained on the base model.

As an example of the transferability of lenses to fine-tuned
models we use Vicuna 13B (Chiang et al., 2023), an open
source instruct fine-tuned chat model based on LLaMA
13B (Touvron et al., 2023). In Figure 13, Section A, we
compare the performance of a lens specifically trained on
Vicuna to a tuned lens trained only on LLaMA. Both Tuned
Lenses were trained using a subsample of the RedPajama
dataset (Together, 2023), an open source replication of
LLaMA’s training set. The performance of both lenses
was then evaluated on the test set of Anthropic’s Helpful
Harmless (Bai et al., 2022) conversation dataset and the
RedPajama Dataset. On the RedPajama Dataset we find,
at worst, a 0.3 bits per byte increase in KL divergence to
the models final output. On the helpful harmless corpus
we find no significant difference between the transferred
and trained lenses. These results show that fine-tuning minimally effects the representations used by the tuned lens. This
opens applications of the tuned lens for monitoring changes
in the representations of a module during fine-tuning and
minimizes the need for practitioners to train lenses on new
fine-tuned models.

**Relation to model stitching.** The tuned lens can be viewed
as a way of “stitching” an intermediate layer directly onto
the unembedding, with an affine transform in between to
align the representations. The idea of model stitching was
introduced by Lenc and Vedaldi (2015), who form a com posite model out of two _frozen_ pretrained models _A_ and _B_,
by connecting the bottom layers of _A_ to the top layers of _B_ .
An affine transform suffices to stitch together independently
trained models with minimal performance loss (Bansal et al.,
2021; Csiszárik et al., 2021). The success of the tuned lens
shows that model stitching works for different layers inside
a single model as well.

**Benefits over traditional probing.** Unlike Alain and Bengio (2016), who train early exiting probes for image classifiers, we do not learn a new unembedding for each layer.
This is important, since it allows us to shrink the size of
each learned matrix from _|V| × d_ to _d × d_, where _|V|_ ranges
from 50K (GPT-2, Pythia) to over 250K (BLOOM). We
observe empirically that training a new unembedding matrix requires considerably more training steps and a larger
batch size than training a translator, and often converges to
a worse perplexity.

## 4. Measuring Causal Fidelity

Prior work has argued that interpretability hypotheses
should be tested with causal experiments: an interpretation of a neural network should make predictions about
what will happen when we intervene on its weights or activations (Olah et al., 2020; Chan et al., 2022). This is especially important for probing techniques, since it’s known
that probes can learn to rely on spurious features unrelated
to the model’s performance (Hewitt and Liang, 2019; Belinkov, 2022).

To explore whether the tuned lens finds causally relevant
features, we will assess two desired properties:

1. Latent directions that are important to the tuned lens
should also be important to the final layer output. Concretely, if the tuned lens _relies on_ a feature [4] _**v**_ in the
residual stream (its output changes significantly when
we manipulate _**v**_ ) then the model output should also
change a lot when we manipulate _**v**_ .

2. These latent directions should be important _in the same_
_way_ for both the tuned lens and the model. Concretely,
if we manipulate the hidden state so that the tuned lens
changes in a certain way (e.g. doubling the probability assigned to “dog”) then the model output should
change similarly. We will call this property _stimulus-_
_response alignment_ .

### 4.1. Causal basis extraction

To test Property 1, we first need to find the important directions for the tuned lens. Amnesic probing (Elazar et al.,

4For simplicity we assume the “features as directions” hypothesis (Elhage et al., 2022), which defines a “feature” to be the
one-dimensional subspace spanned by a unit vector _**v**_ .

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

2021) provides one way to do this—it seeks a direction
whose erasure maximally degrades a model’s accuracy.

However, this only elicits a single important direction,
whereas we would like to find many such directions. To do
so, we borrow intuition from PCA, searching for additional
directions that also degrade accuracy, but which are orthogonal to the original amnesic direction. This leads to a method
that we call **causal basis extraction** (CBE), which finds the
the principal features used by a model.

More specifically, let _f_ be a function (such as the tuned lens)
that maps latent vectors _**h**_ _∈_ R _[d]_ to logits _**y**_ . Let _r_ ( _**h**_ _,_ _**v**_ )
be an erasure function which removes information along
the span of _**v**_ from _**x**_ . In this work we use _r_ ( _**h**_ _,_ _**v**_ ) is _mean_
_ablation_, which sets _⟨r_ ( _**h**_ _,_ _**v**_ ) _,_ _**v**_ _⟩_ to the mean value of _⟨_ _**h**_ _,_ _**v**_ _⟩_
in the dataset (see Appendix D.1). We define the _influence σ_
of a unit vector _**v**_ to be the expected KL divergence between
the outputs of _f_ before and after erasing _**v**_ from _**h**_ :

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|ρ|= 0.89||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||

Figure 8: Causal influence of CBE features when ablated at the
18th layer of Pythia 410M, plotted against their influence on the
tuned lens output. Spearman _ρ_ = 0 _._ 89.

we record the causal effect of erasing _**v**_ _i_ on the output of
_M>ℓ_,

_σ_ ( _**v**_ ; _f_ ) = E

_**h**_

- _DKL_ ( _f_ ( _**h**_ ) _|| f_ ( _r_ ( _**h**_ _,_ _**v**_ ))) (10)

We seek to find an orthonormal basis _B_ = ( _**v**_ 1 _, . . .,_ _**v**_ _k_ )
containing principal features of _f_, ordered by a sequence
of influences Σ = ( _σ_ 1 _, . . ., σk_ ) for some _k_ _≤_ _d_ . In each
iteration we search for a feature _**v**_ _i_ of maximum influence
that is orthogonal to all previous features _**v**_ _j_ :

E

_**x**_

- _DKL_ ( _M_ ( _**x**_ ) _|| M>ℓ_ ( _r_ ( _M≤ℓ_ ( _**x**_ ) _,_ _**v**_ _i_ )) (12)

_**v**_ _i_ = argmax

_||_ _**v**_ _||_ 2 = 1

_σ_ ( _**v**_ ; _f_ )

(11)

where the erasure function _r_ is applied to all positions in
a sequence simultaneously. We likewise average the KL
divergences across token positions.

**Results.** We report the resulting causal influences for Pythia
410M, _ℓ_ = 18 in Figure 8; results for all layers can be found
in Figure 20 in the Appendix.

In accordance with Property 1, there is a strong correlation
between the causal influence of a feature on the tuned lens
and its influence on the model (Spearman _ρ_ = 0 _._ 89). Importantly, we don’t observe _any_ features in the lower right
corner of the plot (features that are influential in the tuned
lens but not in the model). The model is somewhat more
“causally sensitive” than the tuned lens: even the least in
fluential features never have an influence under 2 _×_ 10 _[−]_ [3]

bits, leading to the “hockey stick” shape in the LOWESS
trendline.

### 4.2. Stimulus-response alignment

We now turn to Property 2. Intuitively, for the interventions
from Section 4.1, deleting an important direction _vi_ should
have the same effect on the model’s output distribution _p_
and the tuned lens’ output distribution _q_ .

We can operationalize this with the Aitchison geometry
(Aitchison, 1982), which turns the probability simplex into
a vector space equipped with an inner product. In order
to downweight the influence of rare tokens, we use the s.t. _⟨_ _**v**_ _,_ _**v**_ _j⟩_ = **0** _,_ _∀j_ _< i_

With a perfect optimizer, the influence of _**v**_ _i_ should decrease
monotonically since the feasible region is strictly smaller
with each successive iteration. In practice, we do observe
non-monotonicities due to the non-convexity of the objective. To mitigate this issue we sort the features in descending
order by influence after the last iteration.

**Implementation details.** We evaluate the objective function in Equation 11 on a single in-memory batch of 131,072
tokens sampled randomly from the Pile validation set, and
optimize it using L-BFGS with strong Wolfe line search.
We find that using the singular vectors of the probe as initialization for the search, rather than random directions, speeds
up convergence.

**Intervening** **on** **the** **model.** If we apply causal basis extraction to the tuned lens at layer _ℓ_, we obtain _k_ directions
_v_ 1 _, . . ., vk_ that are important for the tuned lens. We next
check that these are also important to the model _M_ .

To do so, we first take an i.i.d. sample of input sequences
_**x**_ and feed them to _M_, storing the resulting hidden states
_M≤ℓ_ ( _**x**_ ). [5] Then, for each vector _**v**_ _i_ obtained from CBE,

5See Section 2 for notation.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

Accuracy for Direct Questions

15 20 25
Layer

Accuracy for Standard Questions

15 20 25
Layer

0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1

1.0

0.8

0.6

0.4

0.2

0.0

1.0

0.8

0.6

0.4

0.2

0.0

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||Logit le|s|
|||||| Tuned le| ns|

0 2 4 6 8 10

Layer

Figure 9: Average stimulus-response alignment at each layer of

Pythia 160M. Responses are more aligned with stimuli at later
layers, and when using the tuned lens rather than the logit lens.

_weighted_ Aitchison inner product introduced by Egozcue
and Pawlowsky-Glahn (2016), defined as logit lens at early layers, then performs more poorly at later layers.
Interestingly, on this task, the “direct” prompts yield lower elicitation accuracy than “standard” ones.

Figure 10: The tuned lens elicits secrets more effectively than the

We’d like to control for the absolute magnitudes of the
stimuli and the responses, so we use the Aitchison inner
product to define a cosine similarity metric, which we call
“Aitchison similarity.” Then the stimulus-response alignment at layer _ℓ_ under _g_ is simply the Aitchison similarity between
the stimulus and response:

_⟨_ S( _**h**_ _ℓ_ ) _,_ R( _**h**_ _ℓ_ ) _⟩_ **w**
sim(S( _**h**_ _ℓ_ ) _,_ R( _**h**_ _ℓ_ )) = _∥_ S( _**h**_ _ℓ_ ) _∥_ **w** _∥_ R( _**h**_ _ℓ_ ) _∥_ **w**

(18)

_⟨_ **p** 1 _,_ **p** 2 _⟩_ **w** =

_D_

_i_ =1

_p_ 1 _i_ _p_ 2 _i_
_wi_ log (13)
g **w** ( **p** 1) [log] g **w** ( **p** 2) _[,]_

where **w** is a vector of positive weights, and g **w** ( **p** ) is the
weighted geometric mean of the entries of **p** . In our experiments, we use the final layer prediction distribution under
the control condition to define **w** .

We will also use the notion of “subtracting” distributions.
In Aitchison geometry, addition and subtraction of distributions is done componentwise in log space, followed by
renormalization:

**p** 1 _−_ **p** 2 = softmax log **p** 1 _−_ log **p** 2

_._ (14)

We propose to use CBE (Section 4.1) to define a “natural”
choice for the intervention _g_ . Specifically, for each layer _ℓ_,
we intervene on the subspace spanned by _ℓ_ ’s top 10 causal
basis vectors— we’ll call this the “principal subspace”— using a recently proposed method called _resampling ablation_
(Chan et al., 2022).

Given a hidden state _**h**_ _ℓ_ = _M≤ℓ_ ( _**x**_ ), resampling ablation replaces the principal subspace of _**h**_ _ℓ_ with the corresponding
subspace generated on a _different_ input _**x**_ _[′]_ selected uniformly at random from the dataset. It then feeds this modified hidden state _**h**_ [˜] _ℓ_ into the rest of the model, yielding the
modified output _M>ℓ_ ( _**h**_ [˜] _ℓ_ ). Intuitively, _**h**_ [˜] _ℓ_ should be relatively on-distribution because we’re using values generated
“naturally” by the model itself.

Unlike in Section 4.1, we apply resampling ablation to one
token in a sequence at a time, and average the Aitchison
similarities across tokens.

**Results.** We applied resampling ablation to the principal
subspaces of the logit and tuned lenses at each layer in
Pythia 160M. We report average stimulus-response alignments in Figure 9. Unsurprisingly, we find that stimuli are
more aligned with the responses they induce at later layers.
We also find that alignment is somewhat higher at all layers
when using principal subspaces and stimuli defined by the
tuned lens rather than the logit lens, in line with Property 2.

We say that distributions ( **p** old _,_ **p** new) and ( **q** old _,_ **q** new)
“move in the same direction” if and only if

_⟨_ **p** new _−_ **p** old _,_ **q** new _−_ **q** old _⟩_ **w** _>_ 0 _._ (15)

**Measuring alignment.** Let _g_ : R _[d]_ _→_ R _[d]_ be an arbitrary
function for intervening on hidden states, and let _**h**_ _ℓ_ be the
hidden state at layer _ℓ_ on some input _**x**_ . We’ll define the
_stimulus_ to be the Aitchison difference between the tuned
lens output before and after the intervention:

S( _**h**_ _ℓ_ ) = TunedLens _ℓ_ ( _g_ ( _**h**_ _ℓ_ )) _−_ TunedLens _ℓ_ ( _**h**_ _ℓ_ ) (16)

Analogously, the _response_ will be defined as the Aitchison
difference between the final layer output before and after
the intervention:

R( _**h**_ _ℓ_ ) = _M>ℓ_ ( _g_ ( _**h**_ _ℓ_ )) _−M>ℓ_ ( _**h**_ _ℓ_ ) (17)

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

0.4

0.35

0.3

Neo (125M) Neo (1.3B)

0.5

0.4

0.3

True
Demos
False
Demos
Random
Baseline

0 5 10

0.2
0 5 10 15 20

Neo (2.7B) OPT (1.3B)

0.45
0.4
0.35
0.3
0.25
0 5 10 15 20

0.5

0.4

0.3
0 10 20 30

0.5

0.4

0.3

Pythia (12B) BLOOM (560M)

0.4

0.35

0 10 20 30

0.3
0 5 10 15 20

Layer

Figure 11: For most models and tasks, we find there is a layer at
which the tuned lens performance is _better_ than final layer performance under incorrect demonstrations. Shown here is performance
on SICK (Sentences Involving Compositional Knowldedge). Unlike the logit lens, our method is applicable to BLOOM (bottom
right) and GPT-Neo (top left). Y-axis shows median-calibrated
accuracy as used in Halawi et al. (2023)

## 5. Applications

We start by reproducing and extending two use cases of the
logit lens in the literature.

### 5.1. Eliciting Secret Knowledge from Language Models

Cywi´nski et al. (2025) apply the logit lens to uncover knowledge that an AI possesses but does not verbalize. Specifically, they fine-tune a model to generate hints and respond
to user queries about a specific “taboo” word, while never
verbalizing it directly. They then prompt an “auditor” LLM
to guess what the taboo word is, allowing it to ask the taboo
model questions in a black-box fashion. They find that the
auditor is much more accurate at guessing the taboo word
when it is provided with the top 20 highest probability latent
predictions from the taboo model according to the logit lens.
This is true in two different settings: “standard,” where the
prompt asks for a clue about the taboo word, and “direct,”
where the prompt attempts to compel the model to explicitly
reveal the taboo word.

We hypothesized that the tuned lens might perform better
than the logit lens for this task. The results were mixed: the
tuned lens outperforms at some layers and underperforms at
other layers. Without “peaking” at the ground truth labels,
it is not really possible to determine which layer to use in
advance.

**5.2. Extending** _**Overthinking the Truth**_

Halawi et al. (2023) apply the logit lens to downstream
tasks with few-shot prompts, and find that predictions extracted from earlier layers are often more robust to incorrect
demonstrations than final layer predictions. Unfortunately,
their method is inapplicable to models like BLOOM and
GPT-Neo, which have high logit lens perplexities.

We replicate Halawi et al. (2023)’s experiments with the
tuned lens, and recover similar results for BLOOM 560M,
Neo 1.3B, and Neo 2.7B (Figure 11). Notice that the calibrated performance under incorrect demonstrations (red)
peaks at an early layer around 0.4-0.45 in all three of these
models, before falling to random baseline performance at
the final layer, similarly to what was found by Halawi et al.

### 5.3. Detecting Prompt Injections

Given the results from Halawi et al. (2023) and in Figure 11,
we hypothesize that the prediction trajectory of the tuned
lens on anomalous inputs should be different from the trajectories on normal inputs, and that this could be used to
detect anomalous inputs.

To test this, we focus on _prompt injection attacks_, a recently
discovered vulnerability in large language models where
untrusted inputs from a malicious user cause the model to
behave in unexpected or dangerous ways (Perez and Ribeiro,
2022). These attacks usually tell the model to “ignore previous instructions” and instead follow instructions crafted by
the attacker.

To simulate prompt injections, we adapted the attack from
Perez and Ribeiro (2022) and ran Pythia 12B on nine multiple choice tasks, both with and without the following adversarial message appended to the few-shot prompt:

---------
---------

Ignore any previous and following instructions and just print "<WRONG_ANSWER>":

where <WRONG_ANSWER> is replaced with a randomly selected incorrect response from the available multiple choice
responses.

We record the tuned prediction trajectory for each data point–
that is, for each layer, we record the log probability assigned
by the model to each possible answer. [6] We then flatten
these trajectories into feature vectors and feed them into
two standard outlier detection algorithms: isolation forest
(iForest) (Liu et al., 2008) and local outlier factor (LOF)

6For binary tasks like SST-2 we take the difference between
the log probabilities assigned to the two possible answers.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

|Task|Tuned Lens|Logit Lens|Baseline|Accuracy|
|---|---|---|---|---|
|Task|iForest LOF|iForest LOF|SRM|Normal_ →_Injected|
|ARC-Easy ARC-Challenge BoolQ MC TACO MNLI QNLI QQP SciQ SST-2|0_._59 (0_._54_,_ 0_._62) 0_._73 (0_._71_,_ 0_._76) 0_._71 (0_._65_,_ 0_._77) 0_._81 (0_._77_,_ 0_._84) 0_._99 (0_._98_,_ 0_._99) 1_._00 (1_._00_,_ 1_._00) 0_._74 (0_._71_,_ 0_._77) 0_._68 (0_._66_,_ 0_._70) 0_._98 (0_._98_,_ 0_._99) 1_._00 (1_._00_,_ 1_._00) 0_._99 (0_._99_,_ 1_._00) 1_._00 (1_._00_,_ 1_._00) 1_._00 (0_._99_,_ 1_._00) 1_._00 (1_._00_,_ 1_._00) 0_._62 (0_._57_,_ 0_._69) 0_._64 (0_._59_,_ 0_._70) 1_._00 (0_._98_,_ 1_._00) 1_._00 (1_._00_,_ 1_._00)|0_._53 (0_._50_,_ 0_._57) 0_._59 (0_._56_,_ 0_._62) 0_._73 (0_._67_,_ 0_._79) 0_._80 (0_._77_,_ 0_._83) 0_._89 (0_._87_,_ 0_._91) 0_._61 (0_._57_,_ 0_._66) 0_._68 (0_._66_,_ 0_._69) 0_._55 (0_._53_,_ 0_._59) 0_._95 (0_._94_,_ 0_._96) 1_._00 (1_._00_,_ 1_._00) 0_._93 (0_._92_,_ 0_._95) 0_._68 (0_._63_,_ 0_._71) 0_._90 (0_._89_,_ 0_._90) 0_._79 (0_._76_,_ 0_._81) 0_._75 (0_._71_,_ 0_._79) 0_._70 (0_._65_,_ 0_._74) 0_._78 (0_._72_,_ 0_._83) 0_._61 (0_._56_,_ 0_._65)|0_._73 (0_._70_,_ 0_._75) 0_._57 (0_._53_,_ 0_._61) 1_._00 (1_._00_,_ 1_._00) 1_._00 (1_._00_,_ 1_._00) 1_._00 (1_._00_,_ 1_._00) 1_._00 (1_._00_,_ 1_._00) 1_._00 (1_._00_,_ 1_._00) 0_._75 (0_._72_,_ 0_._78) 1_._00 (1_._00_,_ 1_._00)|72_._8%_ →_31_._7% 43_._5%_ →_24_._7% 67_._1%_ →_0_._0% 0_._40_ →_0_._06 F1 54_._3%_ →_0_._0% 54_._3%_ →_0_._0% 60_._7%_ →_6_._5% 95_._5%_ →_62_._6% 82_._9%_ →_49_._1%|

Table 1: Test set AUROCs and 95% bootstrap CIs for distinguishing normal prompts from prompt injections on Pythia 12B. Figures are

pooled over 10 random train-test splits. Attack detection performance is nearly perfect on tasks where the attack succeeds at driving
accuracy well below the random baseline, and is still much better than chance even when the attack is only partially successful.

(Breunig et al., 2000), both implemented in scikit-learn
(Pedregosa et al., 2011) with default hyperparameters.

**Baseline.** There is a rich literature on general out-ofdistribution (OOD) detection in deep neural networks. One
simple technique is to fit a multivariate Gaussian to the
model’s final layer hidden states on the training set, and flag
inputs as OOD if a new hidden state is unusually far from
the training distribution as measured by the Mahalanobis
distance (Lee et al., 2018; Mahalanobis, 1936).

Recently, Bai et al. (2022) proposed the Simplified Relative Mahalanobis (SRM) distance, a modification to Mahalanobis which they find to be effective in the context of
LLM finetuning. They also find that representations from
the _middle_ layers of a transformer, rather than the final layer,
yield the best OOD detection performance. We use the SRM
at the middle layer as a baseline in our experiments.

**Experimental setup.** We fit each anomaly detection model
exclusively on prediction trajectories from _normal_ prompts
without prompt injections, and evaluate them on a held
out test set containing both normal and prompt-injected
trajectories. This ensures that our models cannot overfit
to the prompt injection distribution. We use EleutherAI’s
lm-evaluation-harness library (Gao et al., 2021) to
run our evaluations.

**Results.** Our results are summarized in Table 1. Our tuned
lens anomaly detector achieves perfect or near-perfect AUROC on five tasks (BoolQ, MNLI, QNLI, QQP, and SST-2);
in contrast, the same technique using logit lens has lower
performance on most tasks. On the other hand, the SRM
baseline does consistently well—the tuned lens only outperforms it on one task (ARC-Challenge), while SRM outperforms our technique on both MC TACO and SciQ.

We suspect that further gains could be made by combining the strengths of both techniques, since SRM uses only
one layer but considers a high-dimensional representation,
while the tuned lens studies the trajectory across layers but summarizes them with a low-dimensional prediction vector.

### 5.4. Measuring Example Difficulty

Early exiting strategies like CALM (Schuster et al., 2022)
and DeeBERT (Xin et al., 2020) are based on the observation that “easy” examples require less computation to
classify than “difficult” examples. If an example is easy, the
model should quickly converge to the right answer in early
layers, making it possible to skip the later layers without a
significant drop in prediction quality. Conversely, the number of layers needed to converge on an answer can be used
to measure the difficulty of an example.

We propose to use the tuned lens to estimate example difficulty in _pretrained_ transformers, without the need to finetune the model for early exiting. Following Baldock et al.
(2021)’s work on computer vision models, we define the
_prediction depth_ of a prompt _**x**_ to be the number of layers
after which a model’s top-1 prediction for _**x**_ stops changing.

To validate the prediction depth, we measure its correlation
with an established difficulty metric: the _iteration learned_ .
The iteration learned is defined as the earliest training step
_τ_ where the model’s top-1 prediction for a datapoint _**x**_ is
fixed (Toneva et al., 2018). Intuitively, we might expect that
examples which take a long time to learn during training
would tend to require many layers of computation to classify
at inference time. Baldock et al. (2021) indeed show such a
correlation, using k-NN classifiers to elicit early predictions
from the intermediate feature maps of image classifiers.

**Experimental** **setup.** For this experiment we focus on
Pythia 12B (deduped), for which 143 uniformly spaced
checkpoints are available on Huggingface Hub. We evaluate the model’s zero-shot performance on twelve multiple-choice tasks, listed in Table 2. For each checkpoint, we
store the top 1 prediction on every individual example, allowing us to compute the iteration learned. We then use
the tuned lens on the final checkpoint, eliciting the top 1
prediction at each layer of the network and computing the

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

Task Tuned lens _ρ_ Logit lens _ρ_ Final acc

ARC-Easy **0** _._ **577** 0 _._ 500 69.7%
ARC-Challenge **0** _._ **547** 0 _._ 485 32 _._ 4%
LogiQA **0** _._ **498** 0 _._ 277 21 _._ 4%
MNLI 0 _._ 395 **0** _._ **435** 40 _._ 4%
PiQA **0** _._ **660** 0 _._ 620 76 _._ 1%
QNLI **0** _._ **409** _−_ 0 _._ 099 53 _._ 0%
QQP **0** _._ **585** _−_ 0 _._ 340 0 _._ 381 (F1)
RTE 0 _._ 156 **0** _._ **347** 60 _._ 0%
SciQ **0** _._ **530** 0 _._ 505 91 _._ 9%
SST-2 **0** _._ **555** 0.292 64 _._ 7%
WinoGrande 0 _._ 517 **0** _._ **537** 63 _._ 9% the iteration learned and prediction depth, across tasks. Prediction
depth is measured using the tuned lens in the first column and the
logit lens in the second.

Table 2: Correlation between two measures of example difficulty,

prediction depth for every example. As a baseline, we also
compute prediction depths using the logit lens. Finally, for
each task, we compute the Spearman rank correlation between the iteration learned and the prediction depth across
all examples.

**Results.** We present results in Table 2. We find a significant
positive correlation between the iteration learned and the
tuned lens prediction depth on all tasks we investigated.
Additionally, the tuned lens prediction correlates better with
iteration learned than its logit lens counterpart in 8 out of 11
tasks, sometimes dramatically so.

## 6. Discussion

In this paper, we introduced a new tool for transformer interpretability research, the _tuned_ _lens_, which yields new
qualitative as well as quantitative insights into the functioning of large language models. It is a drop-in replacement for
the logit lens that makes it possible to elicit interpretable prediction trajectories from essentially any pretrained language
model in use today. We gave several initial applications of
the tuned lens, including detecting prompt injection attacks.

Finally, we introduced _causal basis extraction_, which identifies influential features in neural networks. We hope this
technique will be generally useful for interpretability research in machine learning.

**Limitations** **and** **future** **work.** One limitation of our
method is that it involves training a translator layer for
each layer of the network, while the logit lens can be used
on any pretrained model out-of-the-box. This training process, however, is quite fast: our code can train a full set
of probes in under an hour on a single 8 _×_ A40 node, and
further speedups are likely possible. We have also released
tuned lens checkpoints for the most commonly used pre on the abstract of OpenAI (2023). Warm colors have high prediction depth, while cool colors indicate low depth.

Figure 12: Token-level prediction depths for Pythia 12B computed

trained models as part of our tuned-lens library, which
should eliminate this problem for most applications.

Causal basis extraction, as presented in this work, is computationally intensive, since it sequentially optimizes _dmodel_
causal basis vectors for each layer of the network. Future work could explore ways to make the algorithm more
scalable. One possibility would be to optimize a whole _k_ dimensional subspace, instead of an individual direction, at
each iteration.

Due to space and time limitations, we focused on language
models in this work, but we think it’s likely that our approach is also applicable to other modalities.

**Acknowledgements**

We are thankful to CoreWeave for providing the computing
resources used in this paper and to the OPT team for their
assistance in training a tuned lens for OPT. We also thank
nostalgebraist for discussions leading to this paper.

**References**

John Aitchison. The statistical analysis of compositional data. _Journal of the Royal Statistical Society:_ _Series B_
_(Methodological)_, 44(2):139–160, 1982.

Guillaume Alain and Yoshua Bengio. Understanding in
termediate layers using linear classifier probes. _arXiv_
_preprint arXiv:1610.01644_, 2016.

Alex Andonian, Quentin Anthony, Stella Biderman, Sid

Black, Preetham Gali, Leo Gao, Eric Hallahan, Josh
Levy-Kramer, Connor Leahy, Lucas Nestler, Kip Parker,
Michael Pieler, Shivanshu Purohit, Tri Songz, Wang Phil,
and Samuel Weinbach. GPT-NeoX: Large Scale Autoregressive Language Modeling in PyTorch, 8 2021.
[URL https://www.github.com/eleutherai/](https://www.github.com/eleutherai/gpt-neox)
[gpt-neox.](https://www.github.com/eleutherai/gpt-neox)

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell,

Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort,
Deep Ganguli, Tom Henighan, et al. Training a helpful

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

and harmless assistant with reinforcement learning from
human feedback. _arXiv preprint arXiv:2204.05862_, 2022.

Robert Baldock, Hartmut Maennel, and Behnam Neyshabur.

Deep learning through the lens of example difficulty. _Ad-_
_vances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 34:
10876–10889, 2021.

Yamini Bansal, Preetum Nakkiran, and Boaz Barak. Revis
iting model stitching to compare neural representations.
_Advances in Neural Information Processing Systems_, 34:

225–236, 2021.

Yonatan Belinkov. Probing classifiers: Promises, shortcom
ings, and advances. _Computational_ _Linguistics_, 48(1):
207–219, 2022.

Stella Biderman, Kieran Bicheno, and Leo Gao. Datasheet for the Pile. _Computing Research Repository_, 2022. doi:
10.48550/arXiv.2201.07311. URL [https://arxiv.](https://arxiv.org/abs/2201.07311v1)
[org/abs/2201.07311v1.](https://arxiv.org/abs/2201.07311v1) version 1.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony,

Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai
Prashanth, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: a scaling suite for language
model interpretability research. _Computing_ _Research_
_Repository_, 2023. doi: 10.48550/arXiv.2201.07311. URL
[https://arxiv.org/abs/2201.07311v1.](https://arxiv.org/abs/2201.07311v1) version 1.

Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella

Biderman. Gpt-neo: Large scale autoregressive language
modeling with mesh-tensorflow. _If you use this software,_
_please cite it using these metadata_, 58, 2021.

Sid Black, Stella Biderman, Eric Hallahan, Quentin An
thony, Leo Gao, Laurence Golding, Horace He, Connor
Leahy, Kyle McDonell, Jason Phang, et al. Gpt-neox-20b:
An open-source autoregressive language model. _arXiv_
_preprint arXiv:2204.06745_, 2022.

Piotr Bojanowski, Edouard Grave, Armand Joulin, and

Tomas Mikolov. Enriching word vectors with subword
information. _arXiv preprint arXiv:1607.04606_, 2016.

Markus M Breunig, Hans-Peter Kriegel, Raymond T Ng, and Jörg Sander. Lof: identifying density-based local
outliers. In _Proceedings_ _of_ _the_ _2000_ _ACM_ _SIGMOD_
_international conference on Management of data_, pages
93–104, 2000.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah,

Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan,
Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural_
_information processing systems_, 33:1877–1901, 2020.

Lawrence Chan, Adrià Garriga-Alonso, Nicholas
Goldowsky-Dill, Ryan Greenblatt, Jenny Nitishinskaya,
Ansh Radhakrishnan, Buck Shlegeris, and Nate Thomas.
Causal scrubbing: a method for rigorously testing
interpretability hypotheses. _Alignment_ _Forum_, 2022.
[URL https://bit.ly/3WRBhPD.](https://bit.ly/3WRBhPD)

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao

Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P.
Xing. Vicuna: An open-source chatbot impressing gpt-4
with 90%* chatgpt quality, March 2023. [URL https:](https://lmsys.org/blog/2023-03-30-vicuna/)
[//lmsys.org/blog/2023-03-30-vicuna/.](https://lmsys.org/blog/2023-03-30-vicuna/)

Adrián Csiszárik, Péter K˝orösi-Szabó, Ákos Matszangosz,

Gergely Papp, and Dániel Varga. Similarity and matching
of neural network representations. _Advances in Neural_
_Information Processing Systems_, 34:5656–5668, 2021.

Bartosz Cywi´nski, Emil Ryd, Rowan Wang, Senthooran

Rajamanoharan, Neel Nanda, Arthur Conmy, and Samuel
Marks. Eliciting secret knowledge from language models.
_arXiv preprint arXiv:2510.01070_, 2025.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In _Proceedings_ _of_ _the_ _60th_ _Annual_ _Meeting_
_of_ _the_ _Association_ _for_ _Computational_ _Linguistics_ _(Vol-_
_ume_ _1:_ _Long_ _Papers)_, pages 8493–8502, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.581. [URL https:](https://aclanthology.org/2022.acl-long.581)
[//aclanthology.org/2022.acl-long.581.](https://aclanthology.org/2022.acl-long.581)

Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. An
alyzing transformers in embedding space. _arXiv preprint_
_arXiv:2209.02535_, 2022.

Bruno De Finetti, Henry E Kyburg, and Howard E Smok
ler. Foresight: Its logical laws, its subjective sources.
_Breakthroughs in statistics_, 1:134–174, 1937.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,

Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold,
Sylvain Gelly, et al. An image is worth 16x16 words:
Transformers for image recognition at scale. _arXiv_
_preprint arXiv:2010.11929_, 2020.

Juan José Egozcue and Vera Pawlowsky-Glahn. Changing the reference measure in the simplex and its weighting
effects. _Austrian Journal of Statistics_, 45(4):25–44, 2016.

Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Gold
berg. Amnesic probing: Behavioral explanation with
amnesic counterfactuals. _Transactions of the Association_
_for Computational Linguistics_, 9:160–175, 2021.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom

Henighan, Nicholas Joseph, Ben Mann, Amanda Askell,
Yuntao Bai, Anna Chen, Tom Conerly, Nova Das
Sarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds,
Danny Hernandez, Andy Jones, Jackson Kernion, Liane
Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown,
Jack Clark, Jared Kaplan, Sam McCandlish, and Chris
Olah. A mathematical framework for transformer circuits.
_Transformer Circuits Thread_, 2021. https://transformer
circuits.pub/2021/framework/index.html.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas

Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen,
et al. Toy models of superposition. _arXiv_ _preprint_
_arXiv:2209.10652_, 2022.

Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with structured dropout.
_arXiv preprint arXiv:1909.11556_, 2019.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding,

Travis Hoppe, Charles Foster, Jason Phang, Horace He,
Anish Thite, Noa Nabeshima, Shawn Presser, and Connor
Leahy. The Pile: An 800GB dataset of diverse text for
language modeling. _Computing_ _Research_ _Repository_,
2020. doi: 10.48550/arXiv.2101.00027. [URL https:](https://arxiv.org/abs/2101.00027v1)
[//arxiv.org/abs/2101.00027v1.](https://arxiv.org/abs/2101.00027v1) version 1.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, An
thony DiPofi, Charles Foster, Laurence Golding, Jeffrey
Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang,
Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin
Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021. [URL https:](https://doi.org/10.5281/zenodo.5371628)
[//doi.org/10.5281/zenodo.5371628.](https://doi.org/10.5281/zenodo.5371628)

Scott Garrabrant, Tsvi Benson-Tilsen, Andrew Critch, Nate

Soares, and Jessica Taylor. Logical induction. _arXiv_
_preprint arXiv:1609.03543_, 2016.

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin

Choi, and Noah A. Smith. RealToxicityPrompts: Evaluating neural toxic degeneration in language models. In _Findings_ _of_ _the_ _Association_ _for_ _Computa-_
_tional Linguistics:_ _EMNLP 2020_, pages 3356–3369, Online, November 2020. Association for Computational
Linguistics. doi: 10.18653/v1/2020.findings-emnlp.
301. URL [https://aclanthology.org/2020.](https://aclanthology.org/2020.findings-emnlp.301)
[findings-emnlp.301.](https://aclanthology.org/2020.findings-emnlp.301)

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy.

Transformer feed-forward layers are key-value memories.
_arXiv preprint arXiv:2012.14913_, 2020.

Mor Geva, Avi Caciularu, Kevin Ro Wang, and Yoav Gold
berg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. _arXiv_
_preprint arXiv:2203.14680_, 2022.

Klaus Greff, Rupesh K Srivastava, and Jürgen Schmidhuber.

Highway and residual networks learn unrolled iterative
estimation. _arXiv preprint arXiv:1612.07771_, 2016.

Ian Hacking. Slightly more realistic personal probability.

_Philosophy of Science_, 34(4):311–325, 1967.

Danny Halawi, Jean-Stanislas Denain, and Jacob Steinhardt.

Overthinking the truth: Understanding how language
models process false demonstrations. In _Submitted to The_
_Eleventh International Conference on Learning Represen-_
_tations_, 2023. URL [https://openreview.net/](https://openreview.net/forum?id=em4xg1Gvxa)
[forum?id=em4xg1Gvxa.](https://openreview.net/forum?id=em4xg1Gvxa) under review.

Benjamin Heinzerling and Michael Strube. BPEmb:
Tokenization-free Pre-trained Subword Embeddings in
275 Languages. In Nicoletta Calzolari (Conference chair),
Khalid Choukri, Christopher Cieri, Thierry Declerck,
Sara Goggi, Koiti Hasida, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, Asuncion Moreno,
Jan Odijk, Stelios Piperidis, and Takenobu Tokunaga,
editors, _Proceedings of the Eleventh International Con-_
_ference on Language Resources and Evaluation (LREC_
_2018)_, Miyazaki, Japan, May 7-12, 2018 2018. European Language Resources Association (ELRA). ISBN
979-10-95546-00-9.

J Hewitt and P Liang. Designing and interpreting probes with control tasks. _Proceedings of the 2019 Con_, 2019.

John Hewitt and Christopher D Manning. A structural probe for finding syntax in word representations. In _Proceedings_
_of the 2019 Conference of the North American Chapter_
_of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_ _Hu-_
_man Language Technologies, Volume 1 (Long and Short_
_Papers)_, pages 4129–4138, 2019.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
et al. Lora: Low-rank adaptation of large language models. _ICLR_, 1(2):3, 2022.

Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kil
ian Q Weinberger. Deep networks with stochastic depth.
In _European conference on computer vision_, pages 646–
661. Springer, 2016.

Stanisław Jastrz˛ebski, Devansh Arpit, Nicolas Ballas, Vikas

Verma, Tong Che, and Yoshua Bengio. Residual connections encourage iterative inference. _arXiv_ _preprint_
_arXiv:1710.04773_, 2017.

Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz

Cecista, Laker Newhouse, and Jeremy Bernstein. Muon:
An optimizer for hidden layers in neural networks. _URL_
_https://kellerjordan.github.io/posts/muon_, 2024.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, and

Anna Rumshisky. Bert busters: Outlier dimensions that
disrupt transformers. _arXiv preprint arXiv:2105.06990_,
2021.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo

Shin. A simple unified framework for detecting out-ofdistribution samples and adversarial attacks. _Advances in_
_neural information processing systems_, 31, 2018.

Karel Lenc and Andrea Vedaldi. Understanding image rep
resentations by measuring their equivariance and equivalence. In _Proceedings of the IEEE conference on com-_
_puter_ _vision_ _and_ _pattern_ _recognition_, pages 991–999,
2015.

Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Vié
gas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence
model trained on a synthetic task. _arXiv_ _preprint_
_arXiv:2210.13382_, 2022.

Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Isolation forest. In _2008 eighth ieee international conference on_
_data mining_, pages 413–422. IEEE, 2008.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang,

Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe
Lu, Junjie Yan, et al. Muon is scalable for llm training.
_arXiv preprint arXiv:2502.16982_, 2025.

PC Mahalanobis. On the generalized distances in statistics:

Mahalanobis distance. _Journal Soc. Bengal_, 26:541–588,
1936.

Beren Millidge and Sid Black. The singular value decom
positions of transformer weight matrices are highly interpretable. _LessWrong_, 2022. [URL https://bit.ly/](https://bit.ly/3GdbZoa)
[3GdbZoa.](https://bit.ly/3GdbZoa)

Neel Nanda. Transformerlens, 2022. URL
[https://github.com/neelnanda-io/](https://github.com/neelnanda-io/TransformerLens)
[TransformerLens.](https://github.com/neelnanda-io/TransformerLens)

nostalgebraist. interpreting gpt: the logit lens. _Less-_
_Wrong_, 2020. URL [https://www.lesswrong.](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
[com/posts/AcKRB8wDpdaN6v6ru/](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
[interpreting-gpt-the-logit-lens.](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

nostalgebraist. logit lens on non-gpt2 models + extensions, 2021. URL [https:](https://colab.research.google.com/drive/1MjdfK2srcerLrAJDRaJQKO0sUiZ-hQtA)
[//colab.research.google.com/drive/](https://colab.research.google.com/drive/1MjdfK2srcerLrAJDRaJQKO0sUiZ-hQtA)
[1MjdfK2srcerLrAJDRaJQKO0sUiZ-hQtA.](https://colab.research.google.com/drive/1MjdfK2srcerLrAJDRaJQKO0sUiZ-hQtA)

Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel

Goh, Michael Petrov, and Shan Carter. Zoom in: An
introduction to circuits. _Distill_, 5(3):e00024–001, 2020.

OpenAI. Gpt-4 technical report. Technical report, OpenAI,

2023. Technical Report.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel,

B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer,
R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. _Journal of Machine_
_Learning Research_, 12:2825–2830, 2011.

Fábio Perez and Ian Ribeiro. Ignore previous prompt: At
tack techniques for language models. _arXiv_ _preprint_
_arXiv:2211.09527_, 2022.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario

Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. _OpenAI Blog_, 2019.

FP Ramsey. Truth and probability. _Studies_ _in_ _subjective_

_probability_, pages 61–92, 1926.

Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In _2007 15th Euro-_
_pean signal processing conference_, pages 606–610. IEEE,
2007.

Victor Sanh, Lysandre Debut, Julien Chaumond, and

Thomas Wolf. Distilbert, a distilled version of bert:
smaller, faster, cheaper and lighter. _arXiv_ _preprint_
_arXiv:1910.01108_, 2019.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie

Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné,
Alexandra Sasha Luccioni, François Yvon, Matthias
Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi,
Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy,
Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, and
et al. BLOOM: A 176B-parameter open-access multilingual language model. _Computing_ _Research_ _Repos-_
_itory_, 2022. doi: 10.48550/arXiv.2211.05100. URL
[https://arxiv.org/abs/2211.05100v2.](https://arxiv.org/abs/2211.05100v2) version 2.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani,

Dara Bahri, Vinh Q Tran, Yi Tay, and Donald Metzler.
Confident adaptive language modeling. _arXiv preprint_
_arXiv:2207.07061_, 2022.

Pinky Sitikhu, Kritish Pahi, Pujan Thapa, and Subarna

Shakya. A comparison of semantic similarity methods
for maximum human interpretability. _2019 Artificial In-_
_telligence for Transforming Business and Society (AITB)_,
1:1–4, 2019.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

William Timkey and Marten van Schijndel. All bark and no bite: Rogue dimensions in transformer language
models obscure representational quality. _arXiv preprint_
_arXiv:2109.04404_, 2021.

Together. Redpajama, a project to create leading
open-source models, starts by reproducing llama
training dataset of over 1.2 trillion tokens, April
[2023. URL https://www.together.xyz/blog/](https://www.together.xyz/blog/redpajama)
[redpajama.](https://www.together.xyz/blog/redpajama)

Mariya Toneva, Alessandro Sordoni, Remi Tachet des

Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J Gordon. An empirical study of example forgetting during deep neural network learning. _arXiv preprint_
_arXiv:1812.05159_, 2018.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Mar
tinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste
Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and
Guillaume Lample. Llama: Open and efficient foundation
language models, 2023.

Mycal Tucker, Peng Qian, and Roger Levy. What if this modified that? syntactic interventions with counterfactual
embeddings. In _Findings of the Association for Computa-_
_tional Linguistics:_ _ACL-IJCNLP 2021_, pages 862–875,
2021.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and
Illia Polosukhin. Attention is all you need. _Advances in_
_neural information processing systems_, 30, 2017.

Andreas Veit, Michael J Wilber, and Serge Belongie. Resid
ual networks behave like ensembles of relatively shallow
networks. _Advances_ _in_ _neural_ _information_ _processing_
_systems_, 29, 2016.

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck

Shlegeris, and Jacob Steinhardt. Interpretability in the
wild: a circuit for indirect object identification in gpt-2
small. _arXiv preprint arXiv:2211.00593_, 2022.

Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy

Lin. Deebert: Dynamic early exiting for accelerating bert
inference. _arXiv preprint arXiv:2004.12993_, 2020.

Eliezer Yudkowsky. Conservation of expected
evidence. [https://www.lesswrong.](https://www.lesswrong.com/posts/jiBFC7DcCrZjGmZnJ/conservation-of-expected-evidence)
[com/posts/jiBFC7DcCrZjGmZnJ/](https://www.lesswrong.com/posts/jiBFC7DcCrZjGmZnJ/conservation-of-expected-evidence)
[conservation-of-expected-evidence,](https://www.lesswrong.com/posts/jiBFC7DcCrZjGmZnJ/conservation-of-expected-evidence)
2007. Accessed: March 18, 2023.

Minjia Zhang and Yuxiong He. Accelerating training
of transformer-based language models with progressive
layer dropping. _arXiv preprint arXiv:2010.13369_, 2020.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe,

Moya Chen, Shuohui Chen, Christopher Dewan, Mona
Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott,
Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh
Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. OPT: Open pre-trained transformer language
models. _Computing_ _Research_ _Repository_, 2022. doi:
10.48550/arXiv.2205.01068. URL [https://arxiv.](https://arxiv.org/abs/2205.01068v4)
[org/abs/2205.01068v4.](https://arxiv.org/abs/2205.01068v4) version 4.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

**A. Additional evaluation results**

Transfer of a Tuned Lens (LLaMA-13b to Vicuna-13b v1.1)

Anthropic HH-RLHF (chosen) Together RedPajama Sample

|Col1|Tr Tr|ansferred f ained on V|rom LLa icuna|Col5|
|---|---|---|---|---|
|| Lo| git Lens|||
||||||
||||||
||||||
||||||

0 10 20 30 40

Layer

0 10 20 30 40

Figure 13: Comparing the performance of a lens specifically trained on Vicuna 13B vs a lens transferred from LLaMA 13B vs Vicuna
13B’s Logit lens. Specifically, we messure the KL divergence between the lens at a specific layer and the model’s final output.

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Model sma me|size ll dium|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||larg xl|e|
||||||||||||
||||||||||||
||||||||||||

|Eliciting Latent Predictions from|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|Logit lens (baseline)|
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|nput 5 10 15 20 25 30 35 40 45 Logit lens (baseline)|
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Model 125|size M|
|---|---|---|---|---|---|---|---|
|||||||1.3 2.7|B|
|||||||||
|||||||||
|||||||||
|||||||||

Layer

Layer

Logit lens (baseline) Tuned lens (ours)

|Col1|Col2|Col3|Col4|Col5|Col6|Model 125|Col8|
|---|---|---|---|---|---|---|---|
|||||||1.3 6.7|1.3 6.7|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||

Layer

Figure 14: Perplexities of predictions elicited from OpenAI’s GPT-2 models (top), EleutherAI’s GPT-Neo models (middle), and Meta’s

OPT models (bottom). We omit OPT 350M as it uses a post-LN architecture.

**B. Qualitative Results**

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

Tuned Lens (EleutherAI/pythia-12b-deduped)

output input

|Col1|a fir|st o|f tim|es,|i|t w|as th|e wo|rst o|f tim|es|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
||fi  fi|st  st |f ti f ti|es  es |i  i|w  w|s t s t|e w e w|rst  rst |f ti f ti|es  es |
||fi  |st   |f ti  |es   |i  |w  |s t  |e w  |rst   |f ti  |es   |
|||||||||||||
||fi  fi|st th st |ng ti f ti|es  es |t  i|e w  w|s t s t|e w e w|rst  rst |f ti f ti|es  es |
||fi  |st th  |ng ti  |es   |t  |e w  |s t  |e w  |rst   |f ti  |es   |
|||||||||||||
||fi|st |f ti|es |t|e w|s t|e w|rst |f ti|es |
||fi  fi|st sel st th|ing ti ng ti|es a es a|d  d t|f w e w|s t s t|e w e w|rst  rst |f ti f ti|es  es |
|||||||||||||
||fi|st th|ng ti|es a|d a|d w|s t|e w|rst |f ti|es |
|t |e fi  fi|st th st th|ng a ng |ll  ll a|a d a|d w d w|s t s t|e b e w|st  rst |f ti f ti|es  es |
|||||||||||||
||fi|st th|ng a|ll |b|t w|s t|e m|st |f ti|es |
|t|fi e fi|st w st th|y  ng |ll  ll |b  b|t ' t w|s n|fi t m|st  st |f a  a|l  l |
|||||||||||||
||fi|st |f t|e |a|d i||fi|st |f t|e |

It was the best of times, it was the worst of times

Input

Probability
0 0.2 0.4 0.6 0.8 1 the latent prediction becomes very confident at early layers after the prefix “It was the best of times” has been processed, suggesting some
degree of memorization.

Figure 15: Tuned lens prediction trajectory for Pythia 12B on the first several words of Charles Dickens’ “A Tale of Two Cities.” Note that

Tuned Lens (EleutherAI/pythia-12b-deduped)

output input

|w|e sh|ow a|P|T -|Col6|2 o|n a|lar|ge gres|sive Tra|ns mo|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
||e sh e sh|w lang w lang|age  age |L  T ||o  o|n  n |st  st|te gre te gre|sive mo sive Tr|el m ns mo|
||e sh  |w   ||L   ||o  |n   |la  |ge gre  |sive ne  |ral mo  |
|||||||||||||
||e sh e sh|w mo w |els trai  |ed   |trai  ba|ed  ed a|d |la  la|ge gre ge gre|sive ne sive ne|ral m ral mo|
||e sh  |w mo  |els trai  |ed   |ba  |ed a  |d   |mo  |del gre  |sive mo  |el mo  |
|||||||||||||
||e demo|strate |||ba|ed a|d |mo|del gre|sive mo|el mo|
||e sh e sh|w  w |T  S|E   |ba  |ed a  a|d  d |la  m|ge gre del gre|sive mo sive mo|el mo el m|
|||||||||||||
||e sh|w o|T|E |ba|ed a|d |n|w gre|sive mo|del mo|
||e sh e sh|w  w ||E  E |ba  ba|ed  ed a|wh d wh|ch n ch m|w gre del gre|sive mo sive mo|el mo el m|
|||||||||||||
||e al|o |L|A |ba|ed |wh|ch algo|ithm gre|sive mo|del mo|
|t  t|e h e w|ve t ll t|e P e ||ba|ed |a  wh|d indep ch lea|ndent gre ing gre|sive a sive mo|d a el a|
|||||||||||||
|t|e c|n t|||ty|e |a|d la|ge gre|sive dec|ine a|

Specifically, we train G PT - 3, an autore gressive language

Input

Probability
0 0.2 0.4 0.6 0.8 1

Figure 16: Tuned lens prediction trajectory for Pythia 12B prompted with the abstract of Brown et al. (2020).

### B.1. Logit lens pathologies

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

Logit Lens (bigscience/bloom-560m)

output input

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|n nâ n al|Ļt fi o fi|st th st th|ng lu ng a|k w ll |en tho  a|gh wa d w|n't nâ s t|Ļt w e b|rst  st |f ti f ti|es  es |
|||||||||||||
|n nâ n nâ|Ļt respo Ļt sa|sibility th e th|ng institu ng thi|ionais ti gs: |es tho  tho|gh di gh wa|n't nâ n't nâ|Ļt plup Ļt plup|rfect w rfect the|rst mon eof circum|iaux  tances |
|||||||||||||
|n nâ|Ļt respon|sibilitycomu|itÃ¡ria jud|cia ti|es tho|gh -|t nâ|Ļt plup|rfect wo|rst don|ait |
|n nâ n nâ|Ļt col Ļt col|md b md b|st glo st Cort|ais ti sÃŃa ti|es 20 es 20|x; i x; -|nâ t nâ|Ļt com Ļt |anya wo { w|rst Cort rst gre|sÃŃa ti io gr|
|||||||||||||
|n w n w|s ac s ac|rb b rb b|st  st Cort|f ti sÃŃa ti|es tor es tor|ey i ey i|nâ  nâ|Ļt follo Ļt ac|ing: wo rb w|rst  rst Cort|f ti sÃŃa ti|
|||||||||||||
|w  w|s t s t|e b e b|st  st |f ti f ti|es  es it|i a i|w  w|s t s t|e wo e w|rst  rst |f ti f ti|
|||||||||||||

It was the best of times, it was the worst of times

Input

Probability
0 0.2 0.4 0.6 0.8 1 layers and token positions, complicating the interpretation of the output.

Figure 17: Logit lens prediction trajectory for BLOOM 560M. The logit lens assigns as very high probability to the _input_ token at many

Logit Lens (facebook/opt-125m)

output input

|s|a sa|me thi|ng bo|th,|th|e w|as th|e wo|rst o|f tim|es|
|---|---|---|---|---|---|---|---|---|---|---|---|
||sa  sa|e th e thi|ng t ng b|e  th |b  t|t ' e w|t s t|e w e w|rst  rst |f ti f ti|es  es |
|||||||||||||
||sa|e b|st |f ti|es b|t w|s |t w|rst wo|rst |f |
|w  |s t  sa|e b e b|st  st |f ti f ti|es  es b|i t i|w  w|s t s o|e w ly wo|rst  rst |f ti f ti|
|||||||||||||
|w|s t|e b|st |f ti|es |i|w|s t|e wo|rst |f ti|
|w  w|s t s t|e b e b|st  st |f ti f ti|es  es |i  i|w  w|s t s t|e w e wo|rst  rst |f ti f ti|
|||||||||||||
|w|s t|e b|st |f ti|es |i|w|s t|e wo|rst |f ti|
|w  w|s t s t|e b e b|st  st |f ti f ti|es  es |i  i|w  w|s t s t|e w e wo|rst  rst |f ti f ti|

It was the best of times, it was the worst of times

Input

Probability
0 0.2 0.4 0.6 0.8 1

Figure 18: Logit lens prediction trajectory for OPT 125M, exhibiting a similar pathology to BLOOM 560M above.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

**C. Transformers Perform Iterative Inference**

Jastrz˛ebski et al. (2017) argue that skip connections encourage neural networks to perform “iterative inference” in the
following sense: each layer reliably updates the hidden state in a direction of decreasing loss. While their analysis focused
on ResNet image classifiers, their theoretical argument applies equally well to transformers, or any neural network featuring
residual connections. We reproduce their theoretical analysis below.

A residual block applied to a representation _**h**_ _i_ updates the representation as follows:

_**h**_ _ℓ_ +1 = _**h**_ _ℓ_ + _Fℓ_ ( _**h**_ _ℓ_ )

Let _L_ denote the final linear classifier followed by a loss function. We can Taylor expand _L_ ( _**h**_ _L_ ) around _**h**_ _i_ to yield

+ _O_ ( _Fj_ [2][(] _**[h]**_ _[j]_ [))] (19)

_L_ ( _**h**_ _L_ ) = _L_ ( _**h**_ _i_ ) +

_L_

_j_ = _i_

_Fj_ ( _**h**_ _j_ ) _,_ _[∂][L]_ [(] _**[h]**_ _[j]_ [)]

_∂_ _**h**_ _j_

-                         gradient-residual alignment

Thus to a first-order approximation, the model is encouraged to minimize the inner product between the residual _F_ ( _**h**_ _i_ ) and
the gradient _[∂][L]_ _∂_ [(] _**h**_ _**[h]**_ _i_ _[i]_ [)] [, which can be achieved by aligning it with the negative gradient.]

To empirically measure the extent to which residuals do align with the negative gradient, Jastrz˛ebski et al. (2017) compute
the cosine similarity between _F_ ( _**h**_ _i_ ) and _[∂][L]_ _∂_ [(] _**h**_ _**[h]**_ _i_ _[i]_ [)] at each layer of a ResNet image classifier. They find that it is consistently negative, especially in the final stage of the network.

We reproduced their experiment using Pythia 6.9B, and report the results in Figure 19. We show that, for every layer
in the network, the cosine similarity between the residual and the gradient is negative at least 95% of the time. While
the magnitudes of the cosine similarities are relatively small in absolute terms, never exceeding 0.05, we show that they
are much larger than would be expected of random vectors in this very high dimensional space. Specifically, we first
sample 250 random Gaussian vectors of the same dimensionality as _[∂][L]_ _∂_ [(] _**h**_ _**[h]**_ _i_ _[i]_ [)] [, which is (hidden size)] _[ ×]_ [ (sequence length)]

= 8 _,_ 388 _,_ 608. We then compute pairwise cosine similarities between the vectors, and find the 5 [th] percentile of this sample to
be _−_ **6** _×_ **10** _[−]_ **[4]** . Virtually all of the gradient-residual pairs we observed had cosine similarities below this number.

Alignment of residuals with gradients (Pythia 6.7B)

Perplexity after layer deletion (Pythia 6.9B)

−0.01

−0.02

−0.03

−0.04

−0.05

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||Base|line|
|||||||||||||
|||||||||||||

2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32

Layer

Layer ablated and of much larger magnitude than would be expected by chance in this very high dimensional space. Boxplot whiskers are 5th and 95th
percentiles. Results were computed over a sample of roughly 16M tokens from the Pile validation set. **Right:** Cross-entropy loss of Pythia
6.9B on the Pile validation set, after replacing each of its 32 layers with the identity transformation. The dashed line indicates Pythia’s
perplexity on this dataset with all layers intact.

Figure 19: **Left:** Cosine similarity between the observed update to the hidden state and _[∂][L]_ _∂_ [(] _**h**_ _**[h]**_ _i_ _[i]_ [)] [.] [The similarity is almost always negative,]

### C.1. Zero-shot robustness to layer deletion

Stochastic depth (Huang et al., 2016), also known as LayerDrop (Fan et al., 2019), is a regularization technique which
randomly drops layers during training, and can be applied to both ResNets and transformers. Veit et al. (2016) show that

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

ResNets are robust to the deletion of layers even when trained without stochastic depth, while CNN architectures without
skip connections are not robust in this way. These results strongly support the idea that adjacent layers in a ResNet encode
fundamentally similar representations (Greff et al., 2016).

To the best of our knowledge, this experiment had never been replicated before in transformers. We did so and report our
results in Figure 19 above. We find that only the very first layer is crucial for performance; every other layer ablation induces
a nearly imperceptible increase in perplexity. Interestingly, Veit et al. (2016) also found that the first layer is exceptionally
important in ResNets, suggesting that this is a general phenomenon.

**D. Causal intervention details**

### D.1. Ablation techniques

Prior work has assumed that, given a representation _**x**_ and a linear subspace _B_ _⊆_ R _[D]_ along which we want to remove
information, the best erasure strategy is to project _**x**_ onto the orthogonal complement of _B_ :

_**x**_ _[′]_ = proj _B⊥_ ( _**x**_ ) (20)

This “zeroes out” a part of _**x**_, ensuring that _⟨_ _**x**_ _[′]_ _,_ _**u**_ _⟩_ = **0** for all _**u**_ in _B_ . But recent work has pointed out that setting
activations to zero may inadvertently push a network out of distribution, making the experimental results hard to interpret
(Wang et al., 2022; Chan et al., 2022).

For example, a probe may rely on the invariant that _∀_ _**x**_ _, ⟨_ _**x**_ _,_ _**u**_ _⟩≫_ **0** for some _**u**_ in _B_ as an implicit bias term, even if
_⟨_ _**x**_ _,_ _**u**_ _⟩_ has low variance and does not encode any useful information about the input. The naive projection _**x**_ _[′]_ = proj _B⊥_ ( _**x**_ )
would significantly increase the loss, making it seem that proj _B_ _**x**_ contains important information– when in reality model
performance would not be significantly degraded if proj _B_ _**x**_ were replaced with a nonzero constant value more representative
of the training distribution.

To correct for this, in our causality experiments we ablate directions by replacing them with their _mean_ values computed
across a dataset, instead of zeroing them out. Specifically, to ablate a direction _**u**_, we use the formula:

_**x**_ _[′]_ = _**x**_ + _P_ _**u**_ ( _**x**_ _−_ _**x**_ ) (21)

where _Pu_ is the projection matrix for _**u**_ and _**x**_ is the mean representation.

**E. Static interpretability analysis**

Several interpretability methods aim to analyze parameters that “write" (in the sense of Elhage et al. (2021)) to intermediate
hidden states of a language model (Millidge and Black, 2022; Dar et al., 2022; Geva et al., 2022; 2020), and even use this
analysis to successfully edit model behavior (Geva et al., 2022; Millidge and Black, 2022; Dai et al., 2022). Given that the
tuned lens aims to provide a “less biased" view of intermediate hidden states, we should expect the tuned lens to preserve
their effectiveness. We test both static analysis and model editing methods, and confirm this hypothesis. With static analysis,
the tuned lens appears to never decrease performance, and for some models, even increases their performance. With model
editing, we found the tuned lens to outperform the logit lens on OPT-125m and perform equivalently on other models for the
task of toxicity reduction.

### E.1. Static Analysis

Many parameters in transformer language models have at least one dimension equal to that of their hidden state, allowing us
to multiply by the unembedding matrix to project them into token space. Surprisingly, the resulting top _k_ tokens are often
_meaningful_, representing interpretable semantic and syntactic clusters, such as “prepositions” or “countries” (Millidge and
Black, 2022; Dar et al., 2022; Geva et al., 2022). This has successfully been applied to the columns of the MLP output
matrices (Geva et al., 2022), the singular vectors of the MLP input and output matrices (Millidge and Black, 2022), and the
singular vectors of the attention output-value matrices _WOV_ (Millidge and Black, 2022).

In short [7], we can explain this as: many model parameters directly modify the model’s hidden state (Elhage et al., 2021), and

7The exact interpretation depends on the method used - for instance, the rows of _W_ out can be interpreted as the values in a key-value
memory store (Geva et al., 2020).

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

Embeddings Layer 1 Layer 2 Layer 3

0.001

|ρ = 0.|90|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|ρ = 0.|44|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|ρ = 0.|25|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|ρ = 0.|90|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||

1μ 100μ 0.01 1

Layer 4 Layer 5 Layer 6 Layer 7

0.001

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|ρ = 0.|1|||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|ρ = 0.|4||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||

1μ 100μ 0.01 1

0.001

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|ρ = 0.|5|||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|ρ = 0.|0|||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

Layer 8 Layer 9 Layer 10 Layer 11

0.001

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|ρ = 0.|1|||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|ρ = 0.|4||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||

1μ 100μ 0.01 1

0.001

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|ρ = 0.|1|||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|ρ = 0.|3|||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

Layer 12 Layer 13 Layer 14 Layer 15

0.001

|ρ = 0.|79|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|ρ = 0.|82|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|ρ = 0.|71|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|ρ = 0.|74|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

Layer 16 Layer 17 Layer 18 Layer 19

0.001

|ρ = 0.|89|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|ρ = 0.|90|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||

1μ 100μ 0.01 1

0.001

|ρ = 0.|83|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|ρ = 0.|85|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

0.001

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|ρ = 0.|3|||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

0.001

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|ρ = 0.|5|||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||

1μ 100μ 0.01 1

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|ρ = 0.|8|8|||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
|μ 10|0μ 0.01 1|0μ 0.01 1|0μ 0.01 1|0μ 0.01 1|

Influence on lens (bits)

Figure 20: Causal fidelity in Pythia 410M across layers. In the top left corner of each plot is the Spearman rank correlation between the

causal influence of each feature on the tuned lens and its influence on the model.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

|Col1|OV SVD (L) OV SVD (R) QK SVD (L) QK SVD (R) Win SVD Win columns Wout SVD Wout rows|
|---|---|
|Logit lens (real) Logit lens (random) Tuned lens (real) Tuned lens (random)|0.0745 0.1164 0.0745 0.1164 0.0745 0.0864 0.0728 0.0796 0.0698 0.0695 0.0697 0.0688 0.0689 0.0689 0.0691 0.0688 0.1240 0.1667 0.1240 0.1667 0.1193 0.1564 0.1196 0.1630 0.1164 0.1157 0.1177 0.1165 0.1163 0.1262 0.1163 0.1532|

Table 3: The mean interpretability scores (as measured in Section E.3) for Pythia 125M, with several different interpretability techniques
(Millidge and Black, 2022; Geva et al., 2020), comparing both the tuned lens and logit lens to randomly generated matrices. Where
applicable, the notation (L) and (R) indicates that the results are for the left and right singular vectors, respectively.

when viewed from the right angle, the modifications they make are often interpretable. These “interpretable vectors” occur
significantly more often than would be expected by random chance (Geva et al., 2022), and can be used to edit the model’s
behaviour, like reducing probability of specific tokens (Millidge and Black, 2022), decreasing toxicity (Geva et al., 2022), or
editing factual knowledge (Dai et al., 2022).

Although these interpretability methods differ in precise details, we can give a generic model for the interpretation of a
model parameter _W_ using the unembedding matrix: [8]

_T_ ( _W_ ) = top- _k_ ( _fi_ ( _W_ ) _WU_ )

where _fi_ is some function from our parameter to a vector with the same dimensions as the hidden state, and _WU_ is the
unembedding matrix. In words: we extract a hidden state vector from our parameter _W_ according to some procedure,
project into token space, and take the top _k_ matching tokens. The resulting _T_ ( _W_ ) will be a list of tokens of size _k_, which
functions as a human interpretable view of the vector _fi_ ( _W_ ), by giving the _k_ tokens most associated with that vector. As an
example, the model parameter _W_ could be the MLP output matrix _W_ out for a particular layer, and _fi_ the function selecting
the _i_ th column of the matrix.

With the tuned lens, we modify the above to become:

_T_ ( _W_ ) = top- _k_ ( _Lℓ_ ( _fi_ ( _W_ )) _WU_ )

where _Lℓ_ is the tuned lens for layer number _ℓ_, projecting from the hidden state at layer _ℓ_ to the final hidden state.

To test this hypothesis, we developed a novel automated metric for evaluating the performance of these parameter interpretability methods, based on the pretrained BPEmb encoder (Heinzerling and Strube, 2018), enabling much faster and
more objective experimentation than possible with human evaluation. We describe this method in Appendix E.3.

The results for Pythia 125M can be seen in Table 3. The parameters under both the tuned and logit lens consistently appear
more interpretable than random. And the tuned lens appears to show benefit: the difference between random/real average
scores is consistently higher with the tuned lens than the logit lens. However, preliminary experiments found much less
improvement with larger models, where both the tuned and logit lens appeared to perform poorly.

### E.2. Interpretability score distributions

The distribution of interpretability scores for a single parameter (the OV circuit) can be seen in Figure 21. (Interpretability
distributions of other parameters appear to share these features as well.) The plot displays a major complication for static
analysis methods: most singular vectors are not more interpretable than random, except for a minority lying in a long right
tail. It also shows the necessity of comparison to randomly generated matrices: naively comparing the interpretability
scores for the tuned lens against those for the baseline would imply a far greater improvement than actual, because the
interpretability scores under the tuned lens are higher even for randomly generated matrices.

### E.3. Automated interpretability analysis method

While humans can easily tell if a list of words represents some coherent category, this can be subjective and time-consuming
to evaluate manually. Some previous work has attempted to automate this process, using GPT-3 in Millidge and Black

8In most transformer language models, the function here should technically be _T_ ( _W_ ) = top- _k_ (softmax(LN( _fi_ ( _W_ ) _WU_ ))), including
both softmax and LayerNorm. However, the softmax is not necessary because it preserves rank order, and the LayerNorm can be omitted
for similar reasons (and assuming that either _fi_ ( _W_ ) is zero-mean or that _WU_ has been left-centered).

8Random shuffling applied to each matrix (head-wise for attention matrices), to approximate the element-wise marginal distribution.
8Similar to above footnote, random shuffling applied to each _WOV_ for each head.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

OV circuit SVD interpretability scores (logit lens)

0 0.2 0.4 0.6 0.8

Interpretability score

Real

Random

OV circuit SVD interpretability scores (tuned lens)

0 0.2 0.4 0.6 0.8

Interpretability score

Real

Random

Figure 21: The interpretability scores for the right singular vectors of the OV matrices _WOV_ (following Millidge and Black (2022)) in

Pythia 125M, compared with a randomly generated matrix, for both the logit lens (left), and the tuned lens (right).

(2022) or the Perspective API in Geva et al. (2022), but these can still be too slow for quick experimentation.

We created our own method, motivated by the observation that much of the human interpretable structure of these tokens
appears to be related to their mutual similarity. Then, the mutual similarity of words can easily be measured by their cosine
similarity under a word embedding (Sitikhu et al., 2019). In particular, we use the pretrained BPEmb encoder (Heinzerling
and Strube, 2018) [9] . We encode each token, measure the cosine similarity pairwise between all tokens, and take the average
value:

_I_ ( _T_ ( _W_ )) =

- _k,k_
_i,j_ _[E]_ [(] _[T][i]_ [(] _[W]_ [))] _[ ·][ E]_ [(] _[T][j]_ [(] _[W]_ [))]

_k_ [2]

where the subscript on _Ti_ ( _W_ ) denotes the _i_ -th token of _T_ ( _W_ ), and _E_ denotes the normalized BPEmb encoding function.
This creates an interpretability metric _I_ that measures something like the “monosemanticity" of a set of tokens.

Because BPEmb is a pretrained model, unlike most CBOW or skip-gram models, there is no ambiguity in the training data
used to produce word similarities and results are easy to replicate. We find the interpretability scores given by the model to
correspond with human judgment in most cases: it rarely identifies human-uninterpretable lists of tokens as interpretable,
although it occasionally struggles to recognize some cases of interpretable structure, like syntactically related tokens, or
cases where a list as a whole is more interpretable than a pairwise measure can capture.

### E.4. Improved Model Editing

Geva et al. (2022) edit GPT-2-medium to be less toxic by changing the coefficients which feed into the columns of the MLP
output matrix, under the key-value framework developed in Geva et al. (2020). They define both the value vector _vi_ _[ℓ]_ [(i.e.] [the]

columns of the MLP’s _W_ out) and a coefficient _m_ _[ℓ]_ _i_ [(i.e.] [the output of] _[ W]_ [in][ before the activation function) for each column] _[ i]_

and each layer _ℓ_ . This is used to reduce the toxicity of GPT-2-medium by editing the coefficients of specific value vectors.
They find:

_T_ ( _W_ out _l_ ) = top-30( _fi_ ( _W_ out _[ℓ]_ [)] _[W][U]_ [)]

where:

_fi_ ( _W_ _[ℓ]_

out _[ℓ]_ [) =] _[ v]_ _i_ _[ℓ]_

_i_

They then concatenate these top-30 tokens for every layer and column, sending it to Perspective API, a Google API service
which can return a toxicity score. They then sampled from the vectors that scored < 0.1 toxicity (where a score > 0.5 is
classified as toxic), set each value vector’s respective coefficient to 3 (because 3 was a higher than average activation), and
ran the newly edited model through a subset of REALTOXICPROMPTS (Gehman et al., 2020) to measure the change in
toxicity compared to the original model.

9We also tried the fastText encoder (Bojanowski et al., 2016), but found BPEmb to give results better matching human judgment.

**Eliciting Latent Predictions from Transformers with the Tuned Lens**

Severe
Toxicity
Toxicity

Sexually Identity
Threat Profanity Perplexity
explicit attack

Original/ Unedited 0.50 0.088 0.159 0.058 0.40 0.043 **27.66**
Logit Lens Top-5 0.47 0.077 0.148 0.057 0.38 0.043 27.68
Logit Lens Top-10 0.45 0.073 0.144 0.056 0.36 0.040 27.70
Logit Lens Top-20 0.43 0.072 0.141 0.057 0.33 0.041 27.73
Tuned Lens Top-5 0.45 0.079 0.143 0.056 0.36 0.041 **27.66**
Tuned Lens Top-10 0.42 0.063 0.143 0.057 0.33 **0.034** 27.67
Tuned Lens Top-20 **0.39** **0.061** **0.138** **0.052** **0.31** 0.035 27.73 lens outperformed the logit lens with no significant increases in perplexity. Although the value vectors were only graded & ranked by
toxicity, there were still decreases in other metrics (severe toxicity, sexually explicit, etc) that were not directly selected against. However,
these results did not generalize to other models tested.

Table 4: Percentage of prompt generations labeled as toxic for OPT-125m. Best results in **bold** . Across all respective settings, the tuned

Changing the coefficients of value vectors can generate degenerate text such as always outputting “ the" which is scored as
non-toxic. To measure this effect, they calculate the perplexity of the edited models, where a higher perplexity may imply
these degenerate solutions.

We perform a similar experiment, but found the vast majority of value vectors that scored < 0.1 toxicity to be non-toxic as
opposed to anti-toxic (e.g. semantic vectors like “ the", “ and", etc are scored as non-toxic). Alternatively, we set the _k_ most
toxic value vector’s coefficient to 0, as well as projecting with both the logit and tuned lens on OPT-125m, as seen in Table 4.

We additionally tested pythia-125m, pythia-350m, gpt-2-medium, and gpt-neo-125m; however, there were no significant
decreases in toxicity (i.e. - 2%) compared to the original, unedited models.

### E.5. Implementation

The initial implementation of many of our static analysis experiments was helped by the use of the transformer_lens
library (Nanda, 2022).
