# Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models

- **Authors:** Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, Mor Geva
- **Date:** 2024-01-11
- **Source:** https://arxiv.org/abs/2401.06102
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `ghandeharioun_patchscopes_2401.06102`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Asma Ghandeharioun** [* 1] **Avi Caciularu** [* 1] **Adam Pearce** [1] **Lucas Dixon** [1] **Mor Geva** [1 2]

**Step 2:**
**Transforming**

**Hidden State**

**Step 3:**
Feeding **Target Prompt**

to **Target Model**

**Step 4:**
Running Execution
on **Patched Target**

**Abstract**

Understanding the internal representations of
large language models (LLMs) can help explain
models’ behavior and verify their alignment with
human values. Given the capabilities of LLMs
in generating human-understandable text, we propose leveraging the model itself to explain its internal representations in natural language. We
introduce a framework called Patchscopes
and show how it can be used to answer a wide
range of questions about an LLM’s computation.
We show that many prior interpretability methods based on projecting representations into the
vocabulary space and intervening on the LLM
computation can be viewed as instances of this
framework. Moreover, several of their shortcomings such as failure in inspecting early layers or lack of expressivity can be mitigated by
Patchscopes. Beyond unifying prior inspection techniques, Patchscopes also opens up
_new_ possibilities such as using a more capable
model to explain the representations of a smaller
model, and multi-hop reasoning error correction [1] .

## 1. Introduction

The question of what information is captured in the hidden
representations of large language models (LLMs) is of key
importance in control and understanding of modern generative AI, and has drawn substantial attention recently (Casper
et al., 2022; Madsen et al., 2022; Patel & Pavlick, 2021;
Nanda et al., 2023). To tackle this question, prior work has
introduced a diverse array of interpretability methods, which
largely rely on three prominent approaches: training linear

*Equal contribution 1Google Research 2Tel Aviv University. Correspondence to: Asma Ghandeharioun _<_ aghandeharioun@google.com _>_ .

_Proceedings_ _of_ _the_ _41_ _[st]_ _International_ _Conference_ _on_ _Machine_
_Learning_, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by
the author(s).

1Code is publicly available at [https://pair-code.](https://pair-code.github.io/interpretability/patchscopes)
[github.io/interpretability/patchscopes.](https://pair-code.github.io/interpretability/patchscopes)

**Step 1:**
Feeding **Source Prompt**

to **Source Model**

|f(|Col2|Col3|
|---|---|---|
||||
||||
||||
||||

Figure 1: Illustration of our framework, showing a Patchscope

for decoding the hidden representation of _“CEO”_ in the source
prompt (left). We use a target prompt (right) comprised of few-shot demonstrations of string repetitions to encourage the LLM
to explain its internal representation. **Step** **1** : Run the forward
computation on the source prompt in the source model. **Step 2** :
Optionally transform the hidden state at the source layer. **Step 3** :
Run the forward computation on the target prompt up to the target
layer in the target model. **Step 4** : Patch the target representation
of the token _“?”_ at the target layer with the transformed representation (from step 2), then continue the forward computation
from that layer onward. The modularity of Patchscopes allows
designing a variety of methods by configuring the target prompt,
model and transformation.

classifiers, called probes, on top of hidden representations
(Belinkov & Glass, 2019; Belinkov, 2022; Alain & Bengio,
2017), projecting representations to the model’s vocabulary
space (nostalgebraist, 2020; Din et al., 2023; Belrose et al.,
2023), and intervening on the computation to identify if a
representation is critical for certain predictions (Meng et al.,
2022a; Wallat et al., 2020; Wang et al., 2022; Conmy et al.,
2023; Geva et al., 2023).

Despite the wide success of these methods, they each exhibit
practical shortcomings. First, probing relies on supervised
training for pre-defined classes, which is hard to scale when
there is a large number of classes or when all the categories
are not known a priori. Second, the accuracy of vocabulary

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

Table 1: Many prior inspection methods with various objectives

can be viewed as Patchscopes. The rows highlighted in green
show Patchscope configurations that overcome several limitations of prior methods through more expressive inspection that is
training-data free and is more robust across layers.

**Robust**
**Across**
**Layers**

**Inspection**
**Expressive**
**Objective**

**Training**
**Data**
**Free**

**Inspecting**
**Output**
**Distribution**

**Feature**
**Extraction**

**Resolution**

**Inspection**
**Application**

Few-Shot Token Identity Patchscope (§4.1)  -
Logit Lens (nostalgebraist, 2020), - - Embedding Space Analysis (Dar et al., 2023)

Tuned Lens (Belrose et al., 2023) - For learning mappings

Future Lens (Pal et al., 2023) - For learning
mappings

Zero-Shot Feat. Ext. Patchscope (§4.2)  -

LRE Attribute Lens (Hernandez et al., 2023b) - For linearrelation approx. []

Probing (e.g.,2022; Alain & Bengio Belinkov & Glass, 2017; Wang et al., 2019; Belinkov, 2023), [] For trainingprobe

Causal Tracing (Meng et al., 2022a) - -
Attention Knockout (Wang et al., 2022; Conmy - -
et al., 2023; Geva et al., 2023)

Early Exiting, e.g., - For learning Linear Shortcuts (Din et al., 2023) mappings

Caption Generation, e.g., - For learning Linear Mapping (Merullo et al., 2022) mappings projections substantially decreases in early layers and the
outputs are often hard to interpret. Last, all the above methods are not as expressive as one might like: they provide
class probabilities or most likely tokens, as opposed to a
high-quality explanation in natural language.

In this work, we argue that the advanced capabilities of
LLMs in generating human-like text can be leveraged
for “translating” the information in their representations
for humans. We introduce a modular framework, called
Patchscopes (§3), that can be configured to query
various kinds of information from LLM representations.
Patchscopes decode specific information from a representation within an LLM by “patching” it into the inference
pass on a different prompt that has been designed to encourage the extraction of that information. [2] Such configuration (a Patchscope) can be viewed as an inspection tool
geared towards a particular objective, as illustrated in Fig. 1.

We show that many existing methods, including those that
rely on vocabulary projections and computation interventions, can be cast as Patchscopes. Moreover, new configurations of our framework introduce more effective tools
in addressing the same questions, while mitigating several limitations of prior approaches. Also, Patchscopes
enables addressing underexplored questions, such as fine-grained analysis of the input contextualization process and
the extent to which a more expressive model can be used to
inspect hidden representations of a smaller model.

2While patching (or “activation patching”) by itself is not a new
technique, to the best of our knowledge, we are the first to propose
using it for decoding information from hidden representations in a
configurable and expressive manner. See more details in §3.2.

We conduct a series of experiments to evaluate the benefits
and opportunities introduced by Patchscopes, focusing
on auto-regressive LLMs. First, we consider the problem
of estimating the model’s next-token prediction from its
intermediate representations (see §4.1). Across multiple
LLMs, we show that using a few-shot token identity prompt,
a prompt in the form of “tok1 _→_ tok1; tok2 _→_
tok2; _. . ._ ; tok _k_ ” where tok _i_ refers to a random token,
leads to substantial gains over vocabulary projection methods. Next, we evaluate how well Patchscopes can decode specific attributes of an entity from its LLM representations, when these are detached from the original context
(see §4.2). We observe that, despite using no training data,
Patchscopes significantly outperforms probing in six
out of twelve commonsense and factual reasoning tasks, and
works comparably well in all but one of the remaining six.

Beyond output estimation and attribute decoding,
Patchscopes can address questions that are hard
to answer with existing methods. In §4.3, we apply
Patchscopes to study how LLMs contextualize input
entity names in early layers, where vocabulary projections
mostly fail and other methods, at best, provide only a binary
signal of whether the entity has been resolved (Youssef et al.,
2023; Tenney et al., 2019). With a new Patchscope, we
are able to _verbalize_ the gradual entity resolution process.
For example, we show that, as the model processes the final
token of _“Alexander_ _the_ _Great”_ throughout the layers, it
reflects different entities starting from _“Great_ _Britain”_,
to _“the Great Depression”_, to finally resolving _“Alexander_
_the_ _Great”_ . Then, in §4.4 we show how one can further
improve Patchscope expressivity by using a stronger
target model, e.g., Vicuna 13B instead of Vicuna 7B.

Lastly, we showcase the utility of Patchscopes for fixing latent multi-hop reasoning errors, particularly when
the model is capable of conducting each reasoning step
correctly, but fails when they need to be composed in-context (§5). Building on top of the data provided by Hernandez et al. (2023b), we introduce a more complex task
that requires two steps of factual reasoning. Patchscope
achieves 50% accuracy on this task, outperforming chain-ofthought (Wei et al., 2022) (35.71%) and vanilla generations
(19.57%).

To conclude, our work makes the following contributions:
We propose Patchscopes, a general modular framework
for decoding information from the hidden representations
in LLMs. We show that prominent interpretability methods
can be viewed as instances of Patchscopes, and new
configurations result in more expressive, robust across layers, and training-data free alternatives that mitigate their
shortcomings. In addition, novel configurations introduce
unexplored possibilities of stronger inspection techniques,
as well as practical benefits, such as correcting multi-hop

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

reasoning errors.

## 2. Related Work

Activation patching is a causal intervention, commonly used
as a tool for studying if certain activations play a key role in
a model’s computation (Geiger et al., 2021; Vig et al., 2020).
Patching has been used largely for localizing specific information to specific layers and token positions (Goldowsky-Dill et al., 2023; Meng et al., 2022a;b; Stolfo et al., 2023;
Merullo et al., 2023), and for finding information propagation paths in the computation (Wang et al., 2022; Geva et al.,
2023; Hendel et al., 2023; Hanna et al., 2023; Lieberum
et al., 2023). Prior works have also used specific forms of
cross-model patching called stitching, in non-transformer
architectures, mostly to analyze representational similarity (e.g., Bansal et al., 2021; Csiszarik´ et al., 2021; Lenc
& Vedaldi, 2015). Despite certain limitations (Hase et al.,
2023; Zhang & Nanda, 2023), patching remains a principal
tool for mechanistic interpretability (Conmy et al., 2023).

Given promising results from emerging interpretability efforts that employ LLMs to generate human-like text for
inspection (e.g., Mousi et al., 2023; Slobodkin et al., 2023;
Bills et al., 2023), we argue that using patching only for localization purposes is myopic, and propose to use it for
“translating” LLM representations into natural language.

Very recently, patching has been used to study new problem setups (e.g., Pal et al., 2023; Hernandez et al., 2023b),
all of which can be seen as different configurations of our
proposed framework (see §3.2).

Among the growing research efforts in inspecting hidden
representations of neural networks, probing classifiers are
perhaps the most common (e.g., Alain & Bengio, 2017;
Belinkov & Glass, 2019; Belinkov, 2022; Wang et al., 2023),
and methods using projections into the vocabulary space or
their extensions to other domains are another key category
(e.g., Merullo et al., 2023; Geva et al., 2022b; nostalgebraist,
2020; Belrose et al., 2023; Dar et al., 2023; Din et al., 2023;
Langedijk et al., 2023; Vilas et al., 2023). While various
other latent inspection methods exist (e.g., Zhou et al., 2018;
Strobelt et al., 2017; Ghandeharioun et al., 2021; Kim et al.,
2018), the above are the most relevant to this work.

## 3. Patchscopes

In this section, we introduce Patchscopes and show how
it extends prior interpretability methods with new capabilities. While not limited to particular LLM architectures, this
work focuses on auto-regressive transformer-based LLMs.

### 3.1. Framework Description

The key idea in Patchscopes is to leverage the advanced
capabilities of LLMs to generate human-like text for “translating” the information encoded in their own hidden representations. Concretely, given a hidden representation obtained from an LLM inference pass, we propose to decode
specific information from it by _patching_ it into a different
inference pass (of the same or a different LLM) that encourages the translation of that specific information.

Notably, the rest of the forward computation after patching
can augment the representation with additional information,
hence, this approach does not guarantee that the patched
representation itself stores _all_ that information. However,
dispatching the representation from its original context (the
source prompt) stops contextualization and guarantees that
no further information from the source prompt is incorporated in the post-patching computation. Thus, our framework reveals if specific information _can be decoded from the_
_patched representation via the post-patching computation_ .

Given an input sequence of _n_ tokens _S_ = _⟨s_ 1 _, ..., sn⟩_ and
a model _M_ with _L_ layers, _**h**_ _[ℓ]_ _i_ [denotes] [the] [hidden] [repre-]

sentation obtained at layer _ℓ_ _∈_ [1 _, . . ., L_ ] and position
_i_ _∈_ [1 _, . . ., n_ ], when running _M_ on _S_ . To inspect _**h**_ _[ℓ]_ _i_ [,]

we consider a separate inference pass of a model _M_ _[∗]_ with
_L_ _[∗]_ layers on a target sequence _T_ = _⟨t_ 1 _, . . ., tm⟩_ of _m_ tokens. Specifically, we choose a hidden representation _**h**_ [¯] _[ℓ]_ _i_ _[∗][∗]_

at layer _ℓ_ _[∗]_ _∈_ [1 _, . . ., L_ _[∗]_ ] and position _i_ _[∗]_ _∈_ [1 _, . . ., m_ ] in
the execution of _M_ _[∗]_ on _T_ . Moreover, we define a mapping function _f_ ( _**h**_ ; _**θ**_ ) : R _[d]_ _→_ R _[d][∗]_ parameterized by _**θ**_ that
operates on hidden representations of _M_, where _d_ and _d_ _[∗]_

denote the hidden dimension of representations in _M_ and
_M_ _[∗]_, respectively. This function can be the identity function,
a linear or affine function learned on task-specific pairs of
representations, or even more complex functions that incorporate other sources of data. The _patching_ operation refers
to dynamically replacing the representation _**h**_ [¯] _[ℓ]_ _i_ _[∗][∗]_ [during the]

inference of _M_ _[∗]_ on _T_ with _f_ ( _**h**_ _[ℓ]_ _i_ [)][.] [Namely,] [by] [applying]

_**h**_ ¯ _[ℓ]_ _i_ _[∗][∗]_ _[←]_ _[f]_ [(] _**[h]**_ _[ℓ]_ _i_ [)][, we intervene on the generation process and]

modify the computation after layer _ℓ_ _[∗]_ .

Overall, a Patchscope intervention applied to a representation determined by ( _S, i, M, ℓ_ ), is defined by a quintuplet
( _T, i_ _[∗]_ _, f, M_ _[∗]_ _, ℓ_ _[∗]_ ) of a target prompt _T_, a target position _i_ _[∗]_

in this prompt, a mapping function _f_, a target model _M_ _[∗]_,
and a target layer _ℓ_ _[∗]_ of this model. It is possible that _M_ and
_M_ _[∗]_ are the same model, _S_ and _T_ are the same prompt, and
_f_ is the identity function I (i.e., I( _**h**_ ) = _**h**_ ). Next, we show
how this formulation covers prior interpretability methods
and further extends them with new capabilities.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

### 3.2. Patchscopes Encompasses Prior Methods

Many recent methods inspect LLM representations by projecting them to the output vocabulary space (nostalgebraist,
2020; Din et al., 2023; Belrose et al., 2023). Formally, an
estimation of the output distribution is obtained from the
representation _**h**_ _[ℓ]_ _i_ [at position] _[ i]_ [ and layer] _[ ℓ]_ [by:]

_**p**_ _[ℓ]_

_[ℓ]_ _i_ [= softmax(] _[W][U]_ _[f]_ [(] _**[h]**_ _i_ _[ℓ]_

_[ℓ]_ _i_ [))] _[ ∈]_ [R] _[|][V][ |][,]_

where _WU_ _∈_ R _[|][V][ |×][d]_ is the model’s unembedding matrix
and _f_ is a simple mapping function, such as the identity
function or an affine mapping. We note that the operation
applied to _f_ ( _**h**_ _[ℓ]_ _i_ [)] [is] [the] [same] [computation] [applied] [by] [the]

model to the last-layer representation for obtaining the next-token prediction. Therefore, prior methods that inspect
representations in the vocabulary space can be viewed as
a class of Patchscopes that maps representations from
any source layer _ℓ_ to the last target layer _L_ _[∗]_ . Differences
between these methods lie in the choice of _f_ ; logit lens
(nostalgebraist, 2020; Dar et al., 2023) applies the identity
function, linear shortcuts (Din et al., 2023) uses a linear
mapping function, and tuned lens (Belrose et al., 2023)
trains an affine mapping. Recently, Hernandez et al. (2023b)
introduced LRE Attribute Lens that builds _f_ based on a
relation linearity assumption, and showed its effectiveness
in attribute extraction.

et al., 2023), namely, setting _f_ ( _**h**_ ) = **0** . For a configuration
summary of how these interpretability methods can be cast
as Patchscope instances, see §A, Tab. 4.

### 3.3. Patchscopes Enables Novel Inspection Methods

Prior work has utilized specific patching configurations for
interpretability, largely patching the same model while using
the same prompt (i.e., _M_ _[∗]_ = _M_, _T_ = _S_ ). The framing
of Patchscopes introduces a wide range of unexplored
setups potentially unlocking new inspection capabilities.

Specifically, we observe that modifying the target prompt
enables an expressive decoding of a wide range of features,
detached from the source prompt computation. For instance,
we can use the prompt _“The capital of X is”_ to check if the
capital city of a given country is extractable from its (last
token) hidden representation at a specific layer. Similarly,
a prompt like _“Tell me facts about X”_ can be leveraged to
assess whether the model has resolved the entity name in
a specific layer. Contrary to probing, this approach is not
restricted by the number of classes of the chosen feature.

Moreover, when the inspected model is not expressive
enough to answer certain queries, patching representations
into a more capable model could be useful (Hernandez et al.,
2022; Singh et al., 2023; Schwettmann et al., 2023).

## 4. Experiments

In this section, we evaluate our framework on decoding
next-token predictions (§4.1), extracting attributes (§4.2),
analyzing the contextualization of entity names (§4.3), and
leveraging stronger models for inspection via cross-model
patching (§4.4). See a summary in Tab. 1.

### 4.1. Decoding of Next-Token Predictions

As introduced in §3.2, let _**p**_ _[L]_ be the output probability distribution for some input, obtained by multiplying the final-layer last-position hidden representation _**h**_ _[L]_ by the unembedding matrix _WU_ _∈_ R _[|][V][ |×][d]_ . We wish to estimate _**p**_ _[L]_

from intermediate representations **h** _[ℓ]_ s.t. _ℓ<_ _L_ . Particularly, we ask how early in the computation the model has
concluded its final prediction from the given context. In our
experiments, we consider multiple LLMs – LLaMA2 (13B)
(Touvron et al., 2023b), Vicuna (13B) (Chiang et al., 2023),
GPT-J (6B) (Wang & Komatsuzaki, 2021), and Pythia (12B)
(Biderman et al., 2023) (see more details in §B.1).

**Methods** We compare vocabulary projection methods
(§3.2) with a new Patchscope. Each method yields an
estimated output probability _**p**_ ˜ _[ℓ]_ by patching an intermediate representation _**h**_ _[ℓ]_ to the model’s final layer. Here, we
focus on the common setting where _M_ = _M_ _[∗]_, and discuss
extensions to _M̸_ = _M_ _[∗]_ in §4.4.

This class of methods has proven to be effective for different
applications, for example, in improving inference efficiency
via early exiting (Din et al., 2023). While the majority
of methods and applications in this category use a single
model ( _M_ _[∗]_ = _M_ ), Merullo et al. (2022) had demonstrated
successful caption generation with a generative image model
as _M_ and a language model as _M_ _[∗]_ .

Another category of inspection methods intervene on the
LLM computation. Contemporary to our work, Pal et al.
(2023) have investigated whether it is possible to anticipate
multiple generated tokens ahead from a given hidden representation, rather than estimating just the next-token prediction. Their method (Future Lens) uses a target prompt different from the original prompt (i.e., _T_ = _S_ ) and is designed
to decode subsequent tokens from information encoded in
a hidden representation _**h**_ _[ℓ]_ _i_ [.] [Example] [target] [prompts] [are]

_“The multi-tokens present here are ”_ and _“Hello!_ _Could you_

_please_ _tell_ _me_ _more_ _about_ _”_ . Future Lens can be cast as
another Patchscope with _M_ _[∗]_ = _M_ and _ℓ_ _[∗]_ = _ℓ_ .

More broadly, Patchscopes also cover recent mechanistic interpretability methods that analyze internal processes
in LLMs with inference computation interventions. Specifically, causal tracing (Meng et al., 2022a) uses a source
prompt augmented with Gaussian noise as the target prompt.
Other previous methods have intervened on one or more target layers during inference by patching zero vectors to the
computation (Wang et al., 2022; Conmy et al., 2023; Geva

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

_t_ [)][.]

- **Surprisal** ( _↓_ is better): The minus log-probability of the
highest-probability token in the predicted distribution _**p**_ ˜ _[ℓ]_

arg max _t_ (˜ _**p**_ _[ℓ]_ _t_

_[ℓ]_ _t_ [) = arg max] _t_ [(] _**[p]**_ _[L]_ _t_

1.0

0.5

0.0

Logit Lens Tuned Lens Token Identity (Ours)

10 20
layer

_[L]_ _t_ ˜ [, where][ ˜] _[t]_ [ = arg max] _t_ [(˜] _**[p]**_ _[ℓ]_ _t_

according to _**p**_ _[L]_, i.e., _−_ log _**p**_ _[L]_

_t_ [)] _[.]_

10 20
layer

10 20
layer

10 20
layer next-token prediction estimation in multiple models. From layer
10 and upwards, the token identity method (ours) consistently
outperforms the rest of the baselines across all the models.

Figure 2: Precision@1 ( _↑_ is better) and Surprisal ( _↓_ is better) of

- **Logit Lens:** Following prior work (nostalgebraist, 2020;
Geva et al., 2022a), we define _f_ as the identity function,
meaning no change is applied to the patched representation. That is, _f_ ( _**h**_ ) := I( _**h**_ ).

- **Tuned** **Lens:** Motivated by Belrose et al. (2023); Din
et al. (2023), we employ an affine mapping function between representations at layer _ℓ_ and the final layer _L_ .
Specifically, we feed the model examples from a training set _T_ and for each example _s_ _∈T_ obtain a pair
( _**h**_ _[ℓ][,]_ _**[ h]**_ _[L]_ [)] [of] [hidden] [representations.] [Then,] [we] [fit] [a] [lin-]

_[ℓ]_ _s_ _[,]_ _**[ h]**_ _[L]_ _s_

_**h**_ _s_ _[,]_ _**[ h]**_ _s_ [)] [of] [hidden] [representations.] [Then,] [we] [fit] [a] [lin-]

ear regression model to find the matrix _A_ _[ℓ]_ _∈_ R _[d][×][d]_

and the bias vector _**b**_ _[ℓ]_ _∈_ R _[d]_ that are numerical minimizers for [] _s∈T_ _[||][A]_ _**[h]**_ _s_ _[ℓ]_ _[−]_ _**[h]**_ _s_ _[L]_ [+] _**[ b]**_ _[||]_ [2][.] [We] [define] _[f]_ [as:]

mizers for [] _s∈T_ _[||][A]_ _**[h]**_ _s_ _[ℓ]_ _[−]_ _**[h]**_ _s_ _[L]_ [+] _**[ b]**_ _[||]_ [2][.] [We] [define] _[f]_ [as:]

_f_ ( _**h**_ _[ℓ]_ ) := _A_ _[ℓ]_ _**h**_ _[ℓ]_ + _**b**_ _[ℓ]_ .

_s∈T_ _[||][A]_ _**[h]**_ _s_ _[ℓ]_

_[ℓ]_ _s_ _[−]_ _**[h]**_ _s_ _[L]_

**Results** Across all the models, from layer 10 and upwards,
the token identity Patchscope consistently outperforms
the other baselines, obtaining a gain of up to 98% in layers 18-22 (see Fig. 2). This demonstrates the utility of
leveraging the model’s decoding procedure for inspecting
representations of different source prompts, and shows that
_in most cases_ hidden representations in early layers carry
the prediction information regardless of their context.

In the first 10 layers, performance of all methods is worsened, with the token identity prompt performing on-par with
logit lens, and tuned lens performing slightly better, which
could be due to the additional training of its mappings. Low
performance in these layers is expected, as it is where the
input contextualization happens. In §4.3, we introduce a
Patchscope geared towards unraveling this process.

### 4.2. Extraction of Specific Attributes

Classification probes are arguably the most commonly used
method for checking if certain attributes are encoded in
hidden representations (Belinkov, 2022; Belinkov & Glass,
2019). However, they need to be trained, and the range of
attribute classes needs to be known a priori. Here we show
that repurposing Patchscopes for attribute extraction
overcomes these limitations. First, it does not require training. Second, it is not limited by a predefined set of labels,
but rather benefits from an open vocabulary. In addition,
by taking advantage of the model’s nonlinearities, it can
capture more complex relations compared to linear probes.

**Experimental** **Setup** Consider factual and commonsense knowledge represented as triplets ( _σ, ρ, ω_ )
of a subject (e.g., “United States”), a relation
(e.g., “largest city of”), and an object (e.g.,
“New York City”). We investigate to what extent the object _ω_ can be extracted from the last token representation
of the subject _σ_ in an arbitrary input context. To this end,
we conduct experiments on 8 commonsense and 25 factual
knowledge tasks curated by Hernandez et al. (2023b). This
dataset includes ( _σ, ρ, ω_ ) triplets for different relations,
along with prompt templates that verbalize them in natural
language. We conduct experiments with GPT-J (6B) (Wang
& Komatsuzaki, 2021), filtering the data to keep only the
examples where _ω_ appears in the the model’s continuation
of the prompt up to 20 tokens. The choice of 20 balances
computation cost with accommodating the open-ended
nature of Patchscopes, as the ground truth token does not
necessarily appear in the next immediate token in a fluent

- **Token** **Identity** Patchscope: Unlike the previous
methods, here we use a target prompt that is different from the source prompt ( _T_ = _S_ ) and is meant
to encourage decoding the token identity of the hidden representation. Also, while the above methods
skip the computation between layers _l_ and _L_, here we
modify it such that all the information from the source
prompt computation is discarded, except for the patched
representation. We craft a prompt with _k_ demonstrations representing an identity-like function, formatted
as “tok1 _→_ tok1 ; tok2 _→_ tok2 ; _. . ._ ; tok _k_ ”. See
§B.3 for further details and an experiment showing this
method’s robustness to different demonstrations. Note
that this Patchscope does not require any training.

**Evaluation** Following Din et al. (2023), we use Pile evaluation set and the following metrics (see details in §B.2):

- **Precision@1** ( _↑_ is better): The portion of examples for
which the highest-probability token _t_ in the estimated
probability distribution matches the highest-probability
token in the original output distribution. That is, if

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

Table 2: Feature extraction accuracy (mean _±_ std). Comparing zero
shot feature extraction Patchscope to a logistic regression
probe shows that despite using _no training data_, it has a significantly higher accuracy than baseline in 6 out of 12 tasks. We use
pairwise t-test with Bonferroni correction for comparing the two
methods. _[∗∗]_ and _[∗]_ indicate _p<_ 1 _e−_ 5 and _p<_ 1 _e−_ 4, respectively.

**Task** **Probe** **Patchscope**

Fruit inside color 37 _._ 4 _±_ 6 _._ 6 38 _._ 0 _±_ 18 _._ 7
Fruit outside color 35 _._ 5 _±_ 3 _._ 1 **71** _._ **0** _±_ **13** _._ **3** _[∗∗]_

Object superclass **65** _._ **6** _±_ **10** _._ **5** _[∗]_ 54 _._ 8 _±_ 11 _._ 3
Substance phase 73 _._ 8 _±_ 3 _._ 7 **91** _._ **9** _±_ **1** _._ **7** _[∗∗]_

Task done by tool 10 _._ 1 _±_ 3 _._ 2 **48** _._ **1** _±_ **13** _._ **2** _[∗∗]_

Company CEO 5 _._ 0 _±_ 2 _._ 6 **47** _._ **8** _±_ **13** _._ **9** _[∗∗]_

Country currency 17 _._ 7 _±_ 2 _._ 2 **51** _._ **0** _±_ **8** _._ **9** _[∗∗]_

Food from country 5 _._ 1 _±_ 3 _._ 7 **63** _._ **8** _±_ **11** _._ **3** _[∗∗]_

Plays pos. in sport 75 _._ 9 _±_ 9 _._ 1 72 _._ 2 _±_ 7 _._ 2
Plays pro sport 53 _._ 8 _±_ 10 _._ 3 46 _._ 3 _±_ 14 _._ 2
Product by co. 58 _._ 9 _±_ 7 _._ 2 63 _._ 2 _±_ 10 _._ 7
Star constellation 17 _._ 5 _±_ 5 _._ 3 18 _._ 4 _±_ 5 _._ 1 response. For each example, we sample 5 utterances from
the WikiText-103 dataset (Merity et al., 2016) that include
_σ_ and use them as _S_ . Lastly, we keep tasks with at least
15 samples, which results in 5 commonsense and 7 factual
tasks with a total of 1,453 datapoints. See details in §C.

**Methods** We compare our proposed Patchscope
against linear probing (K¨ohn, 2015; Gupta et al., 2015).

- **Zero-shot Feature Extraction** Patchscope: We craft
_T_ as a general verbalization of _ρ_ followed by a placeholder for _σ_, such that _i_ _[∗]_ = _m_ . For example, we use
_T_ _←_ “The largest city in x” with “x” as a
placeholder for the subject. To extract the object from
the entity representation in _S_, we patch the representation of token “x” at layer _ℓ_ _[∗]_ with the representation
of “States” from layer _ℓ_, and consider if the generated text includes _ω_ . The remaining configurations of
this Patchscope are _f_ _←_ I _, M_ _[∗]_ _←M, i_ _←_ the
last token of _σ_ in _S_ . We consider all combinations of
_ℓ_ _∈_ [1 _, . . ., L_ ] _× ℓ_ _[∗]_ _∈_ [1 _, . . ., L_ _[∗]_ ]. Later in this section,
we discuss the role of _ℓ_ pertaining to attribute extraction.

- **Logistic Regression Probe** : Let Ω represent the range
of possible objects for a given relation. We use the set
of unique values of _ω_ in the training set as a proxy for
Ω. We train a logistic regression probe (Kohn¨, 2015;
Gupta et al., 2015) for each layer that predicts _ω_ _∈_ Ω
from the last token representation of _σ_ . Given that 6
out of 12 tasks have fewer than 40 datapoints, we use
three-fold cross-validation for training and evaluation of
this baseline. Note that we have excluded tasks where the probe fails completely due to insufficient number of
training examples (fewer than 15 datapoints).

**Evaluation** We measure the average attribute extraction
accuracy. For a given sample, the Patchscope is considered correct if _∃_ _ℓ_ _[∗]_ _∈_ [1 _, . . ., L_ _[∗]_ ] where the generated text
up to 20 tokens includes _ω_ . For the probe, a prediction is
correct if the highest probability is assigned to _ω_ .

**Results** Tab. 2 summarizes the results, averaged over
_ℓ_ _∈_ [1 _, . . ., L_ ]. We conduct a T-test with Bonferroni correction to compare the two methods. Despite using no
training data and having no restrictions on the output, the
Patchscope achieves a significantly higher accuracy than
the probe on six out of twelve tasks ( _p<_ 1 _e−_ 5), and works
comparably well in all but one of the remaining six. These
results suggest that, in the majority of cases, the source representation without its original context carries enough information about many attributes that a targeted Patchscope
can extract. We also study how the accuracy changes across
the source layers, and observe that Patchscope consistently outperforms the baseline in early layers, outperforms
or works on par with the baseline in mid layers, and almost
all cases where it performs worse than the baseline occur
in later layers. Our interpretation is that given the language
modeling training objective, the representations shift toward
next-token prediction in the later layers. Therefore, the attribute of interest would not be as readily accessible via the
model’s computation in these layers. This interpretation
is also aligned with recent findings that show no decline
in using linear relational embedding in predicting _ω_ only
when the next token also happens to be _ω_ (Hernandez et al.,
2023b). Note that this pattern explains the higher standard
deviation of Patchscope accuracy observed in Tab. 2. We
discuss this phenomenon in more detail in §C (see Fig. 6).

### 4.3. Analyzing Entity Resolution in Early Layers

The previous sections focused on analyzing the information
encoded in a single hidden state. Here we turn to consider a
more global question of how LLMs resolve entity mentions
across multiple layers. Concretely, given a subject entity
name, such as _“the summer Olympics of 1996”_, how does
the model contextualize the input tokens of the entity and at
which layer is it fully resolved?

Answering these questions is hard with existing methods;
vocabulary projections focus on the output prediction and
fail to show clear patterns in early layers, and probing is
restricted to outputs from a fixed number of classes, which
may not be expressive enough to describe this process. Alternative approaches have studied this process indirectly
via interventions (Meng et al., 2022a), showing that the
model constructs a subject representation at the last token
of the entity name. However, it is still unclear how this

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

0 2 4 6 8
layer expressive generations show that as we go through the layers, more
tokens from the context get integrated into the current representation. The “Generation” column shows the automatically generated
text. The “Explanation” column shows our own manually coded
interpretation, aiming to specify what entity the generation refers
to and how that relates to the tokens processed. _M_ _[∗]_ = _M_ _←_
Vicuna (13B), _ℓ_ _[∗]_ = _ℓ_, _S_ _←_ "Diana, Princess of Wales".

Table 3: Illustrating entity resolution via a qualitative example. The

_**ℓ**_ **Generation** **Explanation**

1-2 : Country in the United Kingdom **Wales**

3 : Country in Europe **Wales**

0.4

0.3

0.2

0.1

0.0

: Title held by female sovereigns in their own
right or by queens consort

: Title given to the wife of the Prince of Wales
(and later King)

: Diana, Princess of Wales (1961-1997), the first
6 wife of Prince Charles, Prince of Wales, who was
famous for her beauty and humanitarian work contextualization is performed.

**Princess of Wales**
(unspecific)

**Princess of Wales**
(unspecific)

**Diana,**
**Princess of Wales**

We analyze how LLMs contextualize input entity names by
leveraging Patchscopes. Particularly, we craft a target
prompt for generating a description of a given subject, and
apply it to the hidden representation at the last subject position in the source prompt - where the model forms the
subject representation (Geva et al., 2023; Hernandez et al.,
2023a) – across the early layers. This will allow us to see
how the model describes the subject in each layer.

**Analysis** **Setting** We use a few-shot target
prompt template for decoding an entity description:
“subject1: description1, ..., subject _k_ :

description _k_, x”, while patching the last position
corresponding to x. We take the 200 most popular and
200 least popular subject entities from the PopQA dataset
(Mallen et al., 2023). The popular entities should appear
frequently in LLMs’ pre-training data, and are thus likely
to be captured by the model, while resolving the rare
entities is expected to be more challenging (Kandpal et al.,
2023; Mallen et al., 2023). Then, for the source prompt
we use the entity name, and for the target prompt we
sample _k_ = 3 random subject entities. We obtain a short
(up to one sentence) description of every subject entity
from Wikipedia. Our target prompt and more technical
details are provided in §D.1. We patch the last position
representations from the first 10 layers of Vicuna 13B to
the target prompt and evaluate the generated subject name
and description. Specifically, the generated descriptions are
evaluated against the descriptions from Wikipedia using
RougeL (Lin, 2004). Evaluation with Rouge1 (Lin, 2004)
and Sentence-Bert (Reimers & Gurevych, 2019) shows similar trends (see §D.2).

Figure 3: RougeL scores of the generated descriptions against de
scriptions from Wikipedia, using Vicuna models.

**Results** Tab. 3 illustrates the generations by Vicuna 13B
for a sample subject entity, when patching its representation
at different layers to the target prompt (see more examples
in §D.3). For most entities, the contextualization process is
spread over the first layers, with the last subject token gradually encompassing more distant positions across layers.

This trend can be quantitatively observed by the similarity
between the generated descriptions and the descriptions
from Wikipedia, as measured by RougeL. See Fig. 3 where
_M_ = _M_ _[∗]_ . For both models, similarity increases in the first
5 layers and then slowly decreases. This decrease could
potentially be attributed to contamination caused by the
representation of the placeholder token “x” remaining in the
early layers, when patching is applied to a later layer. Note
that this potential issue is only applicable to multi-token
generation scenarios as future positions can still attend to the
placeholder position in early layers, potentially interfering
with the model’s ability to accurately generate descriptions
for the patched token. See §D.3 for qualitative examples
corroborating this interpretation. As expected, the scores for
rare, long-tail entities are significantly lower than those of
popular entities. See §D.2 for additional results with Pythia
where the smaller model seems to outperform the larger
model, possibly because the larger model is biased toward
output generation at the expense of input contextualization.
To summarize, this analysis shows Patchscopes’ utility
for inspecting the contextualization process in early layers.

### 4.4. Expressiveness from Cross-Model Patching

A possible avenue for improving inspection capabilities is to
explain a given model with a model that is more expressive
(Bills et al., 2023). In Patchscopes, this means to patch
a representation of _M_ into a more expressive model _M_ _[∗]_ .
However, it is not clear if such an intervention would yield
plausible results, due to possible discrepancies between the
two models resulting from different architectures, optimization processes, and so on. Here, we present experiments on
next-token prediction and entity resolution that exemplify

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

is Satya Nadella. He has been the CEO since 2014.

Figure 4: An illustration of CoT Patchscope on a single example. In this example, _π_ 1 _←_
“the company that created Visual Basic Script”, _π_ 2 _←_ “The current CEO of”, _S_ = _T_ _←_ [ _π_ 2][ _π_ 1] =
“The current CEO of the company that created Visual Basic Script”. Note that _M_ = _M_ _[∗]_ and _f_ _←_ I.

not only the feasibility, but also the opportunities unlocked
by patching across models of the same family.

**Next-Token Prediction** We repeat the experiment in §4.1,
using the token identity Patchscope. To overcome discrepancies between the models, we learn affine mappings
between their layers (similarly to Tuned Lens). Our results
show that source and target layers on the diagonal exhibit
the highest precision values, and that patching representations to an early layer of the larger model are the most
effective. Overall, suggesting that when _M_ _[∗]_ and _M_ are
from the same model family, it is possible to leverage _M_ _[∗]_

for decoding information from the representations of _M_ .
For detailed results on Precision@1 and Surprisal, see §E.

**Entity Resolution in Early Layers** We now show that using a large model as _M_ _[∗]_ can enhance the output expressivity.
To this end, we repeat our entity resolution experiment in
§4.3 with Vicuna model family, setting _M←_ 7B _, M_ _[∗]_ _←_ 13B.
Fig. 3 shows the cross-model patching results (green lines)
compared to the same-model patching _M←_ 7B _, M_ _[∗]_ _←_ 7B
(blue line). The results show that cross-model patching from
a smaller model to its larger version generally improves the
ability to inspect the input contextualization, both for popular and rare entities. For Pythia, since the smaller model
outperforms the larger one, cross-model patching is not as
effective (see §D.2).

**5. Application:** **Correcting Multi-Hop Errors**

Multi-hop reasoning is a challenging problem (Zhong et al.,
2023). While a language model may be capable of correctly
answering each step independently, it could still fail at processing the connection between different steps, resulting in
an incorrect prediction. Recent attempts to improve multi-hop reasoning rely on prompting the model to generate a
step-by-step answer autoregressively (e.g., Wei et al., 2022;
Yao et al., 2023; Besta et al., 2024), some with an itera
tive process of self-refinement (e.g., Madaan et al., 2023).
However, achieving similar benefits might be possible via
directly rewiring the model’s intermediate computation.

Here, we show that Patchscopes can improve multi-hop
reasoning performance _without_ generating the reasoning
steps, particularly in cases where the model fails at completing a multi-hop query despite being successful in each
reasoning step independently. Via Patchscopes, one
can surgically operate on the model representations, reroute
its intermediate answer to one reasoning step, simplify the
consequent step, and ultimately correct the final prediction.

**Data** Building on Hernandez et al. (2023b), we systematically generate all valid multi-hop factual and commonsense
reasoning queries where _ω_ 1 = _σ_ 2. We conduct experiments
on Vicuna (13B), focusing on samples where _M_ accurately
represents both _τ_ 1 and _τ_ 2 independently, that is, _ω_ appears in
the next 20 tokens _M_ generates conditioned on the prompt
_π_ that verbalizes _σ_ and _ρ_ . This process yields 1,104 multi

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

hop reasoning samples, out of which 46 satisfy the above
criteria and are used for evaluation. See more details in §F.

**Experimental** **Setup** Following the notation in §4.2,
let _τ_ 1 = ( _σ_ 1 _, ρ_ 1 _, ω_ 1) represent the relation _ρ_ 1 between a subject entity _σ_ 1 and an object entity _ω_ 1. Let
_τ_ 2 = ( _σ_ 2 _, ρ_ 2 _, ω_ 2) represent another tuple such that
_σ_ 2 = _ω_ 1. A multi-hop reasoning query pertaining to
_τ_ 1 and _τ_ 2 is a prompt composed of two parts: _π_ 1 is a
verbalization of _σ_ 1 and _ρ_ 1 from which _ω_ 1 can be inferred;
_π_ 2 is a verbalization of _ρ_ 2, from which _ω_ 2 can be inferred
after its concatenation with _π_ 1. For example, Let _τ_ 1 _←_
(“Visual Basic” _,_ “product of” _,_ “Microsoft”)

and _τ_ 2 _←_ (“Microsoft”, “company CEO”, “Satya
Nadella”). An example verbalization of these tuples
is _π_ 1 _←_ “the company that created Visual
Basic”, _π_ 2 _←_ “The current CEO of”, leading to
the multi-hop query [ _π_ 2][ _π_ 1] =“The current CEO of
the company that created Visual Basic”.

**Method** We introduce a Chain-of-Thought (CoT)
Patchscope to fix multi-hop reasoning via intervening on
the computation graph and rerouting representation likely to
capture _ω_ 1 in place of _σ_ 2. Concretely, _S_ refers to the formed
query discussed above, and we use the following configuration: _T_ _←_ _S, M_ _[∗]_ _←M, i ←_ _n, i_ _[∗]_ _←_ the token preceding
_π_ 1. We evaluate the outputs in terms of accuracy, similarly
to §4.2. For a sample _S_, the Patchscope is considered
accurate if _∃_ ( _ℓ, ℓ_ _[∗]_ ) : _ℓ_ _∈_ [1 _, . . ., L_ ] _, ℓ_ _[∗]_ _∈_ [1 _, . . ., L_ _[∗]_ ] where
the autoregressive generation up to 20 tokens includes _ω_ 2.
Fig. 4 illustrates the CoT Patchscope with an example.
We use the following configuration for CoT Patchscope:
_S_ _←_ _π_ 1 _, T_ _←_ _π_ 2 _, i_ _←_ _n, i_ _[∗]_ _←_ _m_, which is equivalent to
_S_ = _T_ _←_ [ _π_ 2][ _π_ 1] and adjusting the attention mask such
that no token in _S_ has visibility to _π_ 2 and no token in _T_ has
visibility to _π_ 1. In addition, we consider two baselines.

**Vanilla Baseline** For this baseline, we set _S_ _←_ [ _π_ 1][ _π_ 2],
we let the model autoregressively generate up to 20 tokens
and check whether _ω_ 2 appears in the generation.

**Chain-of-Thought** **Baseline** Here, the setup and evaluation is similar to the vanilla baseline, except that we
prepend "Let’s think step by step." to _S_, following (Wei et al., 2022). We then let the model generate up to 20 tokens and check whether _ω_ 2 appears in the
generation. Note that this experiment uses Vicuna (13B).
Vicuna is based on LLaMA, with supervised finetuning on
additional instruction data, which makes it amenable to
chain-of-thought prompting.

**Results** While the vanilla baseline accuracy is only
19.57%, and CoT baseline accuracy is 35.71%, our proposed Patchscope achieves 50% accuracy. For more details about the interaction between _ℓ_ and _ℓ_ _[∗]_, and how it
affects the success rate, see Fig. 12 in §F.

We emphasize that our primary goal in this section is not
to devise a new method to solve multi-hop queries that
is necessarily a competitor to CoT, but rather to make a
proof-of-concept while comparing with CoT as a common
reference. We highlight that we have taken advantage of the
extra structural information about the queries given the prior
knowledge about how they were synthesized. One could potentially automate this by learning the right places to patch.
Li et al. (2024) recently proposed an optimization-based
approach for directly patching multi-head self-attention in
mid-layers, which can be viewed as soft position-selection,
and works effectively in multi-hop error correction, suggesting that inferring the right positions to patch could be effective. However, even if optimal source and target position
are automatically decided upfront, Patchscope and CoT may
not be directly comparable. CoT generates multiple steps
which fundamentally extend the computational power of the
LLM (Merrill & Sabharwal, 2024), but a Patchscope with
pre-identified ( _l, l_ _[∗]_ ) only makes _O_ (1) inference passes.

## 6. Conclusion

We present Patchscopes, a simple and effective framework that leverages the ability of LLMs to generate human-like text for decoding information from intermediate LLM
representations. We show that many existing interpretability
methods can be cast as specific Patchscope instances,
and even these only cover a small portion of the framework’s possible configurations. Moreover, new underexplored Patchscopes substantially improve our ability to
decode various types of information from the model’s internal computation, such as the output prediction and knowledge attributes, typically outperforming prominent methods
that rely on projection to the vocabulary and probing. In
addition, our framework enables new capabilities, such as
analyzing the contextualization process of input tokens in
early LLM layers, and can correct multi-hop reasoning.

There are multiple future research directions to consider.
An important factor in the effectiveness of a chosen target prompt is how the information from the patched position propagates during inference to other positions and
across layers. Understanding how to best use a given target
prompt, perhaps automatically, is an important factor for
using Patchscopes. Another avenue for future work is
investigating the effectiveness of few-shot target prompts
compared to zero-shot/instruction-based prompts. More
expressive instruction-based target prompts, for example,
could enable extracting more complex information. Additionally, while our cross-model patching focused on models
from the same family, it will be valuable to explore which
mapping functions would enable patching across models

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

from different families and perhaps with different architectures. Other directions for future work include applications
across different domains and modalities, investigating variants with simultaneous multi-token patching or multi-layer
patching to mitigate the risks of placeholder contamination,
and presenting recipes for task-specific and task-agnostic
Patchscopes.

**Acknowledgements**

We thank Amir Globerson for feedback on writing and presentation of results. We thank Ardavan Saeedi, Martin Wattenberg, Ellie Pavlick, the AI Explorables team [3] at Google
Research, and Jasmijn Bastings for their helpful comments.

**Impact Statement**

**Societal** **Impact** This paper presents a new framework
for interpreting hidden representations of large language
models. Interpretability methods in general can be used to
investigate models’ reliability and safety prior to deployment. We hope that Patchscopes framework facilitates
progress in this area with the introduction of inspection tools
that are more expressive, robust across layers, and do not
require training data.

**Limitations** While our proposed framework is not limited
to any particular architecture or domain, the experimental
evidence provided in this paper focuses on autoregressive
Transformer-based language models, and future work is
needed to verify its effectiveness in other setups.

**References**

Alain, G. and Bengio, Y. Understanding intermediate layers using linear classifier probes. _5th_ _International_ _Con-_
_ference_ _on_ _Learning_ _Representations,_ _Workshop_ _Track_
_Proceedings_, 2017.

Bansal, Y., Nakkiran, P., and Barak, B. Revisiting model stitching to compare neural representations. _Advances_
_in neural information processing systems_, 34:225–236,
2021.

Belinkov, Y. Probing classifiers: Promises, shortcomings, and advances. _Computational Linguistics_, 48(1):207–219,
2022.

Belinkov, Y. and Glass, J. Analysis methods in neural
language processing: A survey. _Transactions of the Asso-_
_ciation for Computational Linguistics_, 7:49–72, 2019.

Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I.,

McKinney, L., Biderman, S., and Steinhardt, J. Eliciting

[3https://pair.withgoogle.com/explorables](https://pair.withgoogle.com/explorables)

latent predictions from transformers with the tuned lens.
_arXiv preprint arXiv:2303.08112_, 2023.

Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Gi
aninazzi, L., Gajda, J., Lehmann, T., Podstawski, M.,
Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts:
Solving elaborate problems with large language models.
_Proceedings of the 38th AAAI Conference on Artificial_
_Intelligence_, 2024.

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley,

H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S.,
Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In
_International Conference on Machine Learning (ICML)_,
2023.

Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L.,

Goh, G., Sutskever, I., Leike, J., Wu, J., and Saunders,
W. Language models can explain neurons in language
models. _URL https://openaipublic. blob. core. windows._
_net/neuron-explainer/paper/index. html.(Date accessed:_
_14.05. 2023)_, 2023.

Casper, S., Rauker, T., Ho, A., and Hadfield-Menell, D.

Sok: Toward transparent ai: A survey on interpreting
the inner structures of deep neural networks. In _First_
_IEEE_ _Conference_ _on_ _Secure_ _and_ _Trustworthy_ _Machine_
_Learning_, 2022.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang,

H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E.,
Stoica, I., and Xing, E. P. Vicuna: An open-source
chatbot impressing GPT-4 with 90%* ChatGPT qual[ity, March 2023. URL https://lmsys.org/blog/](https://lmsys.org/blog/2023-03-30-vicuna/)
[2023-03-30-vicuna/.](https://lmsys.org/blog/2023-03-30-vicuna/)

Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim,

S., and Garriga-Alonso, A. Towards automated circuit discovery for mechanistic interpretability. In _Thirty-seventh_
_Conference on Neural Information Processing Systems_,
2023.

Csiszarik, A., K´ or˝ osi-Szab¨ o, P., Matszangosz,´ A. K., Papp, [´]

G., and Varga, D. Similarity and matching of neural
network representations. In Beygelzimer, A., Dauphin, Y.,
Liang, P., and Vaughan, J. W. (eds.), _Advances in Neural_
_Information_ _Processing_ _Systems_, 2021. URL [https:](https://openreview.net/forum?id=aedFIIRRfXr)
[//openreview.net/forum?id=aedFIIRRfXr.](https://openreview.net/forum?id=aedFIIRRfXr)

Dar, G., Geva, M., Gupta, A., and Berant, J. Analyzing transformers in embedding space. In _Annual Meeting of_
_the Association for Computational Linguistics_, 2023.

Din, A. Y., Karidi, T., Choshen, L., and Geva, M. Jump to conclusions: Short-cutting transformers with linear
transformations. _arXiv preprint arXiv:2303.09435_, 2023.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T.,

Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N.,
et al. The Pile: An 800GB dataset of diverse text for
language modeling. _arXiv_ _preprint_ _arXiv:2101.00027_,
2020.

Geiger, A., Lu, H., Icard, T., and Potts, C. Causal abstrac
tions of neural networks. _Advances in Neural Information_
_Processing Systems_, 34:9574–9586, 2021.

Geva, M., Caciularu, A., Dar, G., Roit, P., Sadde, S., Shlain,

M., Tamir, B., and Goldberg, Y. LM-debugger: An interactive tool for inspection and intervention in transformer-based language models. In Che, W. and Shutova, E.
(eds.), _Proceedings_ _of_ _the_ _2022_ _Conference_ _on_ _Empiri-_
_cal_ _Methods_ _in_ _Natural_ _Language_ _Processing:_ _System_
_Demonstrations_, pp. 12–21, Abu Dhabi, UAE, December 2022a. Association for Computational Linguistics.
doi: 10.18653/v1/2022.emnlp-demos.2. URL [https:](https://aclanthology.org/2022.emnlp-demos.2)
[//aclanthology.org/2022.emnlp-demos.2.](https://aclanthology.org/2022.emnlp-demos.2)

Geva, M., Caciularu, A., Wang, K., and Goldberg, Y. Trans
former feed-forward layers build predictions by promoting concepts in the vocabulary space. In Goldberg, Y.,
Kozareva, Z., and Zhang, Y. (eds.), _Proceedings of the_
_2022 Conference on Empirical Methods in Natural Lan-_
_guage Processing_, pp. 30–45, Abu Dhabi, United Arab
Emirates, December 2022b. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.
3. URL [https://aclanthology.org/2022.](https://aclanthology.org/2022.emnlp-main.3)
[emnlp-main.3.](https://aclanthology.org/2022.emnlp-main.3)

Geva, M., Bastings, J., Filippova, K., and Globerson, A. Dis
secting recall of factual associations in auto-regressive
language models. In Bouamor, H., Pino, J., and Bali,
K. (eds.), _Proceedings_ _of_ _the_ _2023_ _Conference_ _on_ _Em-_
_pirical_ _Methods_ _in_ _Natural_ _Language_ _Processing_, pp.
12216–12235, Singapore, December 2023. Association
for Computational Linguistics. doi: 10.18653/v1/2023.
emnlp-main.751. URL [https://aclanthology.](https://aclanthology.org/2023.emnlp-main.751)
[org/2023.emnlp-main.751.](https://aclanthology.org/2023.emnlp-main.751)

Ghandeharioun, A., Kim, B., Li, C.-L., Jou, B., Eoff, B., and Picard, R. W. Dissect: Disentangled simultaneous explanations via concept traversals. _arXiv preprint_
_arXiv:2105.15164_, 2021.

Goldowsky-Dill, N., MacLeod, C., Sato, L., and Arora, A.

Localizing model behavior with path patching. _arXiv_
_preprint arXiv:2304.05969_, 2023.

Gupta, A., Boleda, G., Baroni, M., and Pado, S.´ Distribu
tional vectors encode referential attributes. In _Proceed-_
_ings_ _of_ _the_ _2015_ _Conference_ _on_ _Empirical_ _Methods_ _in_
_Natural Language Processing_, pp. 12–21, 2015.

Hanna, M., Liu, O., and Variengien, A. How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. In _Thirty-seventh_
_Conference on Neural Information Processing Systems_,
2023. [URL https://openreview.net/forum?](https://openreview.net/forum?id=p4PckNQR8k)
[id=p4PckNQR8k.](https://openreview.net/forum?id=p4PckNQR8k)

Hase, P., Bansal, M., Kim, B., and Ghandeharioun, A. Does localization inform editing? surprising differences in
causality-based localization vs. knowledge editing in language models. _arXiv preprint arXiv:2301.04213_, 2023.

Hendel, R., Geva, M., and Globerson, A. In-context learning creates task vectors. In _The 2023 Conference on Empiri-_
_cal Methods in Natural Language Processing_, 2023.

Hernandez, E., Schwettmann, S., Bau, D., Bagashvili, T.,

Torralba, A., and Andreas, J. Natural language descriptions of deep features. In _International_ _Confer-_
_ence on Learning Representations_, 2022. [URL https:](https://openreview.net/forum?id=NudBMY-tzDr)
[//openreview.net/forum?id=NudBMY-tzDr.](https://openreview.net/forum?id=NudBMY-tzDr)

Hernandez, E., Li, B. Z., and Andreas, J. Measuring and ma
nipulating knowledge representations in language models.
_arXiv preprint arXiv:2304.00740_, 2023a.

Hernandez, E., Sharma, A. S., Haklay, T., Meng, K., Watten
berg, M., Andreas, J., Belinkov, Y., and Bau, D. Linearity of relation decoding in transformer language models.
_arXiv preprint arXiv:2308.09124_, 2023b.

Kandpal, N., Deng, H., Roberts, A., Wallace, E., and Raffel,

C. Large language models struggle to learn long-tail
knowledge. In _International_ _Conference_ _on_ _Machine_
_Learning_, pp. 15696–15707. PMLR, 2023.

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J.,

Viegas, F., et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors
(tcav). In _International conference on machine learning_,
pp. 2668–2677. PMLR, 2018.

Kohn, A.¨ What’s in an embedding? analyzing word embed
dings through multilingual evaluation. In _Proceedings of_
_the 2015 Conference on Empirical Methods in Natural_
_Language Processing_, pp. 2067–2073, 2015.

Langedijk, A., Mohebbi, H., Sarti, G., Zuidema, W., and Jumelet, J. Decoderlens: Layerwise interpretation of encoder-decoder transformers. _arXiv_ _preprint_
_arXiv:2310.03686_, 2023.

Lenc, K. and Vedaldi, A. Understanding image representa
tions by measuring their equivariance and equivalence. In
_Proceedings of the IEEE conference on computer vision_
_and pattern recognition_, pp. 991–999, 2015.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

Li, Z., Jiang, G., Xie, H., Song, L., Lian, D., and Wei, Y.

Understanding and patching compositional reasoning in
llms. _arXiv preprint arXiv:2402.14328_, 2024.

Lieberum, T., Rahtz, M., Kramar, J., Irving, G., Shah, R.,´ and Mikulik, V. Does circuit analysis interpretability
scale? evidence from multiple choice capabilities in chinchilla. _arXiv preprint arXiv:2307.09458_, 2023.

Lin, C.-Y. ROUGE: A package for automatic evaluation of summaries. In _Text_ _Summarization_ _Branches_
_Out_, pp. 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL [https:](https://aclanthology.org/W04-1013)
[//aclanthology.org/W04-1013.](https://aclanthology.org/W04-1013)

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L.,

Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang,
Y., Gupta, S., Majumder, B. P., Hermann, K., Welleck,

S., Yazdanbakhsh, A., and Clark, P. Self-refine: Iterative refinement with self-feedback. In _Thirty-seventh_
_Conference on Neural Information Processing Systems_,
2023. [URL https://openreview.net/forum?](https://openreview.net/forum?id=S37hOerQLB)
[id=S37hOerQLB.](https://openreview.net/forum?id=S37hOerQLB)

Madsen, A., Reddy, S., and Chandar, S. Post-hoc inter
pretability for neural nlp: A survey. _ACM_ _Computing_
_Surveys_, 55(8):1–42, 2022.

Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., and Hajishirzi, H. When not to trust language models: Investigating effectiveness of parametric and nonparametric memories. In Rogers, A., Boyd-Graber, J., and
Okazaki, N. (eds.), _Proceedings of the 61st Annual Meet-_
_ing of the Association for Computational Linguistics (Vol-_
_ume 1:_ _Long Papers)_, pp. 9802–9822, Toronto, Canada,
July 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.acl-long.546. URL [https:](https://aclanthology.org/2023.acl-long.546)
[//aclanthology.org/2023.acl-long.546.](https://aclanthology.org/2023.acl-long.546)

Meng, K., Bau, D., Andonian, A., and Belinkov, Y. Locating and editing factual associations in gpt. _Advances in Neu-_
_ral_ _Information_ _Processing_ _Systems_, 35:17359–17372,
2022a.

Meng, K., Sharma, A. S., Andonian, A. J., Belinkov, Y., and

Bau, D. Mass-editing memory in a transformer. In _The_
_Eleventh_ _International_ _Conference_ _on_ _Learning_ _Repre-_
_sentations_, 2022b.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. In _International Conference on_
_Learning Representations_, 2016.

Merrill, W. and Sabharwal, A. The expressive power of transformers with chain of thought. In _The Twelfth Inter-_
_national Conference on Learning Representations_, 2024.

Merullo, J., Castricato, L., Eickhoff, C., and Pavlick, E. Lin
early mapping from image to text space. In _The Eleventh_
_International Conference on Learning Representations_,
2022.

Merullo, J., Eickhoff, C., and Pavlick, E. A mechanism for solving relational tasks in transformer language models.
_arXiv preprint arXiv:2305.16130_, 2023.

Mousi, B., Durrani, N., and Dalvi, F. Can llms facilitate interpretation of pre-trained language models? In _Pro-_
_ceedings of the 2023 Conference on Empirical Methods_
_in Natural Language Processing_ . Association for Computational Linguistics, December 2023. URL [https:](https://browse.arxiv.org/pdf/2305.13386.pdf)
[//browse.arxiv.org/pdf/2305.13386.pdf.](https://browse.arxiv.org/pdf/2305.13386.pdf)

Nanda, N., Lee, A., and Wattenberg, M. Emergent lin
ear representations in world models of self-supervised
sequence models. In Belinkov, Y., Hao, S., Jumelet,
J., Kim, N., McCarthy, A., and Mohebbi, H. (eds.),
_Proceedings_ _of_ _the_ _6th_ _BlackboxNLP_ _Workshop:_ _An-_
_alyzing_ _and_ _Interpreting_ _Neural_ _Networks_ _for_ _NLP_,
pp. 16–30, Singapore, December 2023. Association for
Computational Linguistics. doi: 10.18653/v1/2023.
blackboxnlp-1.2. URL [https://aclanthology.](https://aclanthology.org/2023.blackboxnlp-1.2)
[org/2023.blackboxnlp-1.2.](https://aclanthology.org/2023.blackboxnlp-1.2)

nostalgebraist. interpreting gpt: the logit lens. _Less-_
_Wrong_, 2020. URL [https://www.lesswrong.](https://www.lesswrong. com/posts/AcKRB8wDpdaN6v6ru/ interpreting-gpt-the-logit-lens)
[com/posts/AcKRB8wDpdaN6v6ru/](https://www.lesswrong. com/posts/AcKRB8wDpdaN6v6ru/ interpreting-gpt-the-logit-lens)
[interpreting-gpt-the-logit-lens.](https://www.lesswrong. com/posts/AcKRB8wDpdaN6v6ru/ interpreting-gpt-the-logit-lens)

Pal, K., Sun, J., Yuan, A., Wallace, B. C., and Bau, D.

Future lens: Anticipating subsequent tokens from a single
hidden state. In _Proceedings of the 27th Conference on_
_Computational Natural Language Learning (CoNLL)_, pp.
548–560, 2023.

Patel, R. and Pavlick, E. Mapping language models to grounded conceptual spaces. In _International Conference_
_on Learning Representations_, 2021.

Reimers, N. and Gurevych, I. Sentence-bert: Sentence em
beddings using siamese bert-networks. In _Proceedings_
_of the 2019 Conference on Empirical Methods in Natu-_
_ral Language Processing_ . Association for Computational
Linguistics, 11 2019. URL [https://arxiv.org/](https://arxiv.org/abs/1908.10084)
[abs/1908.10084.](https://arxiv.org/abs/1908.10084)

Schwettmann, S., Chowdhury, N., Klein, S., Bau, D., and

Torralba, A. Multimodal neurons in pretrained text-only
transformers. In _Proceedings of the IEEE/CVF Interna-_
_tional Conference on Computer Vision_, pp. 2862–2867,
2023.

Singh, C., Hsu, A., Antonello, R., Jain, S., Huth, A., Yu, B., and Gao, J. Explaining black box text modules in natural

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

language with language models. In _XAI in Action:_ _Past,_
_Present, and Future Applications_, 2023. [URL https:](https://openreview.net/forum?id=3BX9tM03GT)
[//openreview.net/forum?id=3BX9tM03GT.](https://openreview.net/forum?id=3BX9tM03GT)

Slobodkin, A., Goldman, O., Caciularu, A., Dagan, I., and Ravfogel, S. The curious case of hallucinatory
(un)answerability: Finding truths in the hidden states
of over-confident large language models. In Bouamor,
H., Pino, J., and Bali, K. (eds.), _Proceedings_ _of_ _the_
_2023 Conference on Empirical Methods in Natural Lan-_
_guage Processing_, pp. 3607–3625, Singapore, December
2023. Association for Computational Linguistics. doi:
10.18653/v1/2023.emnlp-main.220. URL [https://](https://aclanthology.org/2023.emnlp-main.220)
[aclanthology.org/2023.emnlp-main.220.](https://aclanthology.org/2023.emnlp-main.220)

Stolfo, A., Belinkov, Y., and Sachan, M. A mechanistic interpretation of arithmetic reasoning in language
models using causal mediation analysis. In Bouamor,
H., Pino, J., and Bali, K. (eds.), _Proceedings_ _of_ _the_
_2023 Conference on Empirical Methods in Natural Lan-_
_guage Processing_, pp. 7035–7052, Singapore, December
2023. Association for Computational Linguistics. doi:
10.18653/v1/2023.emnlp-main.435. URL [https://](https://aclanthology.org/2023.emnlp-main.435)
[aclanthology.org/2023.emnlp-main.435.](https://aclanthology.org/2023.emnlp-main.435)

Strobelt, H., Gehrmann, S., Pfister, H., and Rush, A. M.

Lstmvis: A tool for visual analysis of hidden state dynamics in recurrent neural networks. _IEEE transactions_
_on visualization and computer graphics_, 24(1):667–676,
2017.

Tenney, I., Das, D., and Pavlick, E. Bert rediscovers the classical nlp pipeline. In _Proceedings of the 57th Annual_
_Meeting of the Association for Computational Linguistics_,
pp. 4593–4601, 2019.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,

M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,`
Azhar, F., et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_, 2023a.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,

A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P.,
Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. _arXiv_ _preprint_ _arXiv:2307.09288_,
2023b.

Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D.,

Singer, Y., and Shieber, S. Investigating gender bias in
language models using causal mediation analysis. _Ad-_
_vances_ _in_ _neural_ _information_ _processing_ _systems_, 33:
12388–12401, 2020.

Vilas, M. G., Schaumloffel,¨ T., and Roig, G. Analyzing vision transformers for image classification in class embedding space. _arXiv preprint arXiv:2310.18969_, 2023.

Wallat, J., Singh, J., and Anand, A. BERTnesia: Investigating the capture and forgetting of knowledge in
BERT. In Alishahi, A., Belinkov, Y., Chrupała, G.,
Hupkes, D., Pinter, Y., and Sajjad, H. (eds.), _Proceed-_
_ings_ _of_ _the_ _Third_ _BlackboxNLP_ _Workshop_ _on_ _Analyz-_
_ing and Interpreting Neural Networks for NLP_, pp. 174–
183, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.blackboxnlp-1.
17. URL [https://aclanthology.org/2020.](https://aclanthology.org/2020.blackboxnlp-1.17)
[blackboxnlp-1.17.](https://aclanthology.org/2020.blackboxnlp-1.17)

Wang, B. and Komatsuzaki, A. GPT-J-6B: A
6 Billion Parameter Autoregressive Language
Model. [https://github.com/kingoflolz/](https://github.com/kingoflolz/mesh-transformer-jax)
[mesh-transformer-jax, May 2021.](https://github.com/kingoflolz/mesh-transformer-jax)

Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., and

Steinhardt, J. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In _The Eleventh_
_International Conference on Learning Representations_,
2022.

Wang, Z., Ku, A., Baldridge, J., Griffiths, T. L., and Kim,

B. Gaussian Process Probes (GPP) for uncertainty-aware
probing. _arXiv preprint arXiv:2305.18213_, 2023.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F.,

Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought
prompting elicits reasoning in large language models.
_Advances in Neural Information Processing Systems_, 35:

24824–24837, 2022.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao,

Y., and Narasimhan, K. R. Tree of thoughts: Deliberate problem solving with large language models. In _Thirty-_
_seventh_ _Conference_ _on_ _Neural_ _Information_ _Processing_
_Systems_, 2023.

Youssef, P., Koras¸, O., Li, M., Schlotterer, J., and Seifert,¨

C. Give me the facts! a survey on factual knowledge
probing in pre-trained language models. In _Findings of_
_the Association for Computational Linguistics:_ _EMNLP_
_2023_, pp. 15588–15605, 2023.

Zhang, F. and Nanda, N. Towards best practices of activation patching in language models: Metrics and methods. _arXiv_
_preprint arXiv:2309.16042_, 2023.

Zhong, Z., Wu, Z., Manning, C. D., Potts, C., and Chen,

D. Mquake: Assessing knowledge editing in language models via multi-hop questions. _arXiv_ _preprint_
_arXiv:2305.14795_, 2023.

Zhou, B., Bau, D., Oliva, A., and Torralba, A. Interpreting deep visual representations via network dissection. _IEEE_
_transactions on pattern analysis and machine intelligence_,
41(9):2131–2145, 2018.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

**A. Detailed Configuration of Prior Methods as Patchscope Instances**

In this section, we first reiterate Patchscopes premise and distinguish it from prior work. We then detail Patchscope
configurations that replicate prior methods. We highlight that clearly formulating Patchscopes in the form of source and
target quintuplets, respectively ( _S, i, M, l_ ) and ( _T, i_ _[∗]_ _, M_ _[∗]_ _, l_ _[∗]_ ), is not in itself a contribution of this work. However, it is
what facilitates the main contributions that are 3-fold: 1) The framework ties methods that otherwise might seem unrelated,
e.g., logit lens and attention knockout, 2) makes it easy to come up with new configurations that significantly improve them,
3) makes it easy to see the space that prior methods do not cover, and therefore open up new applications altogether.

We focus on a few inspection objectives. Note that some of the baselines considered have not been originally designed
to address these objectives, but are the closest solutions in literature. Particularly, we ask, given a (set of) hidden
representation(s) detached from their context, how well can we do next-token prediction, feature extraction, and analysis of
model’s contextualization process.

### A.1. Next-token Prediction

**Goal:** Estimating output probability distribution _p_ _[L]_ given a hidden representation _h_ _[l]_, particularly to inspect when the
model has enough information to make the final prediction.

**Prior Work:** Many recent methods to estimate output probability distribution given an inner representation use projection
onto the vocabulary space (e.g., nostalgebraist, 2020; Din et al., 2023; Belrose et al., 2023). Considering ( _S, i, M, l_ ) and
( _T, i_ _[∗]_ _, M_ _[∗]_ _, l_ _[∗]_ ) formulation above, all these methods can be seen as Patchscopes where **M** = **M** _[∗]_ _,_ **l** _[∗]_ = **L**, where **L** is the number of layers.

**Distinguishing Patchscope & its Improvements:** We propose the token identity Patchscope that unlike (Belrose
et al., 2023; Din et al., 2023) does not require a learned mapping, yet is more successful. Note that in this task, setting
**l** _[∗]_ = **L** which all the prior methods discussed above have in common, means that the choice of **T** does not matter. However,
in our proposed Patchscope, we show that setting **l** _[∗]_ = **l** where _l_ is arbitrary layer, and using a few-shot token identity
prompt **T** that encourages repetition results in significant improvements, and by using the target model’s computation from
layer _l_ _[∗]_ onward, there is no need for training a separate mapping function.

### A.2. Feature Extraction

**Goal:** Extracting specific attributes of the subject given its hidden representation _h_ _[l]_ (e.g., extracting “largest city of” from
the representation of “States” in “United States”).

**Prior Work:** Classification probes (Alain & Bengio, 2017) are the most commonly used methods for this purpose. They
work by training a classifier on a dataset of prompts and their corresponding labels (e.g., countries and their corresponding
largest cities).

**Distinguishing Patchscope & its Improvements:** We show an intuitive Patchscope (i.e., _T_ = “The largest city
in x”) significantly outperforms probing across various tasks and layers. Note that this Patchscope 1) does not require
supervised training, 2) is not limited by a predefined set of labels, and 3) is more accurate than a linear probe.

### A.3. Analysis of Model’s Contextualization Process

**Goal:** Identifying how entity tokens are processed, and at which layer the entity is fully resolved.

**Prior** **Work:** It is hard to answer this question with existing methods. First, methods that use vocabulary projection
are focused on output prediction (rather than input processing) and fail particularly in early layers, precisely where entity
resolution happens. Second, probing requires a predefined set of classes, which is not expressive enough to show the gradual
entity resolution process, and nontrivial. For example, consider “Alexander the grea” input. We found that “Great Britain”
and “Great depression” arose in the gradual resolution process. However, these are non-trivial choices a priori. We can
train a multi-class classifier to choose one out of all possible entities (or binary classifiers for every possible entity), which
is unlikely to work given the huge space of possible entities. Lastly, most prior work using activation patching aims to
answer a different question: whether a certain representation plays a key role in the model’s forward computation and the

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

final prediction (e.g., Meng et al., 2022a; Geva et al., 2023). However, they can be viewed as Patchscope instances, and
provide indirect signals about the entity resolution process, namely, showing subject information accrues in the last token
after a few layers.

**Distinguishing** **Patchscope** **&** **its** **Improvements:** To the best of our knowledge, there are not any methods that
verbalize the gradual entity resolution process in coherent text, and the proposed Patchscope is the first to do that.

In summary, as discussed in 3.2, many prior methods can be seen as Patchscopes. Tab. 4 details the configurations of
Patchscopes that yield various methods introduced in prior work.

Table 4: Patchscopes is a novel framework for inspection of hidden representations in language models. Many prior inspection

methods with various objectives can be viewed as Patchscopes, as detailed in the “Configuration” column (see notation description
in §3). The rows highlighted in green show our new configurations that overcome several limitations of prior methods through more
expressive inspection that is training-data free and is more robust across layers. Additionally, the generality of this framework enables
novel inspection possibilities that were unexplored before. When the target prompt ( _T_ ) is not specified, it means that the output would be
invariant to the choice of _T_ . When not specified, _f_ _←_ I and _M_ _[∗]_ = _M_ .

|Training Robust Inspection Expressive Data Across Objective Free Layers|Confgiuration|
|---|---|
|Inspecting Output Distribution Few-shot token identity Patchscope (§4.1) ËË Ë ËË Logit Lens (nostalgebraist, 2020), Embedding Space Analysis (Dar et al., 2023) Ë Ë é Tuned Lens (Belrose et al., 2023) Ë For learning mappings Ë Future Lens (Pal et al., 2023) Ë For learning mappings ËË|_ℓ∗←ℓ_, _T ←_“tok1 _→_tok1; tok2 _→_tok2;_ . . ._ ; tok_k_” _ℓ∗←L∗_ _ℓ∗←L∗, f ←_Affne _ℓ∗←ℓ, f ←_Linear_, T ←_Fixed or learned soft prompt |
|Feature Extraction Zero-shot feature extraction Patchscope (§4.2) ËË Ë ËË LRE Attribute Lens (Hernandez et al., 2023b) Ë For linear relation approx. ËË Probing (e.g., Belinkov & Glass, 2019; Belinkov, 2022; Alain & Bengio, 2017; Wang et al., 2023) é For training probe Ë|_ℓ∗←j′ ∈_[1_, . . ., L∗_]_, i∗←m_, _T ←_relation verbalization followed by x _ℓ∗←L∗, f ←_Linear with additional variables_, T ←S_ N/A|
|Entity Resolution Entity description Patchscope (§4.3) ËË Ë ËË X-model entity description Patchscope (§4.4) ËËË For learning mappings ËË Causal Tracing (Meng et al., 2022a) é Ë ËË Attention Knockout (Wang et al., 2022; Conmy et al., 2023; Geva et al., 2023) é Ë ËË|_ℓ∗←ℓ_,_ i∗←m_, _T ←_“subject1: description1, ..., subject_k_: description_k_, x” _M∗←_a larger variant of_ M, ℓ∗←ℓ, i∗←m_, _T ←_“subject1: description1, ..., subject_k_: description_k_, x” _ℓ∗←ℓ, T ←S_ +_ ϵ, ϵ ∼N_(0_, σ_) _ℓ∗←_Multiple_, f ←_0_, T ←S_|
|Inspection Application Early Exiting, e.g., Linear Shortcuts (Din et al., 2023) Ë For learning mappings Ë Caption Generation, e.g., Linear Mapping (Merullo et al., 2022) Ë For learning mappings Ë|_ℓ∗←L∗, f ←_Affne _M∗←_A language model of choice_, ℓ∗←L∗, f ←_Affne|

**B. Next-Token Prediction Additional Details and Experimental Results**

In this section, we provide more information about the models, data, and additional target prompts for few-shot token
identity Patchscope configurations. It is worth highlighting that after patching, we continue computations from the
target layer onward. Therefore, since in this Patchscope, source layer and target layer are the same, there are going to be
more computations compared to baseline methods like LogitLens (nostalgebraist, 2020; Geva et al., 2022a) and Tuned Lens
(Belrose et al., 2023; Din et al., 2023) where the target layer is fixed at the final layer.

### B.1. Models

We use LLaMA2 (13B) [4] (Touvron et al., 2023b), Vicuna (13B) [5] (Chiang et al., 2023), GPT-J (6B) (Wang & Komatsuzaki,
2021) [6], and Pythia (12B) [7] (Biderman et al., 2023). LLaMA2 was pre-trained on 2T tokens from a mix of publicly available
data. Vicuna is a LLaMA1 (Touvron et al., 2023a) model that was pre-trained on 1T tokens and fine-tuned on 70K

4LLaMA 2 is licensed under the LLAMA 2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
5Vicuna is subject to the LLAMA 2 Community License, terms of use of the data generated by OpenAI, and privacy practices of
ShareGPT. The code is released under the Apache License 2.0.

6GPT-J-6B weights are licensed under version 2.0 of the Apache License.
7Pythia is released under Apache License 2.0, Copyright © 2023 EleutherAI.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

1.0

0.8

0.6

0.4

0.2

0.0

|Col1|450, 41, 8355 2, 7509, 564, 2 2426, 11|3, 10532, 443|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||~~98~~ 65, 32354, 435 Single token p|, 3 rompt|, 3 rompt||||
||||||||
||||||||
||||||||
||||||||
||||||||

0 10 20 30 40
layer

(a) Precision@1

|Col1|Col2|Col3|Col4|450, 41, 2, 7509,|8355 564, 23, 10532, 443|
|---|---|---|---|---|---|
|||| 2426, 11 98 65, 3235 Single to| 2426, 11 98 65, 3235 Single to| 4, 435, 3 ken prompt|
|||| 2426, 11 98 65, 3235 Single to| 2426, 11 98 65, 3235 Single to||
|||||||
|||||||

0 10 20 30 40
layer

(b) Surprisal user-shared conversations [8] . The primary architectural differences between LLaMA2 and Vicuna (LLaMA1) include a
different context length and grouped-query attention. Pythia and GPT-J were pre-trained using a deduplicated version of the
Pile corpus [9] (Gao et al., 2020), and for about 300B and 402B tokens, respectively.

Figure 5: Next-token prediction results for LLaMA2, using various token identity demonstrations (the token IDs appear in the legend).
We report precision@1 ( _↑_ is better), and surprisal ( _↓_ is better).

### B.2. Training and Evaluation Data

We use 12,000 random samples from the Pile, partitioned into 10,000 examples for training the affine mappings, and 2,000
examples for evaluation. In our pre-processing strategy, we introduce randomness in the patching positions by trimming the
input sequence length of each example.

### B.3. Additional Few-Shot Token Identity Prompts

In this section, we provide additional details about the selection of the demonstrations for the token identity baseline, and
further evaluate the robustness of LLaMA2 (13B) (Touvron et al., 2023b) to various token identity prompts.

**Demonstrations Construction** For the demonstrations used in this experiment, we sample a random set of _k_ tokens for
all the models, where _k_ was also randomly sampled from the interval [1 _, . . .,_ 10].

**Robustness to Additional token IDs’ Demonstrations** We randomly generate five realizations of token IDs series of
varying lengths, formatted as “tok1 _→_ tok1 ; tok2 _→_ tok2 ; _. . ._ ; tok _k_ ”, similarly to the procedure from
§4.1. We also include an ablation experiment, introducing the _Single_ _Token_ _Prompt_ baseline. In this approach, the
representation is transferred from the original multi-token prompt to a single-token prompt, while preserving the same
position. This is crucial for maintaining consistent positional biases during the computation in the forward pass. The baseline
is implemented by creating a target prompt in the format: “[PAD] [PAD] _. . ._ [PAD] tok _k_ ”. Here, the number of

[PAD] tokens is chosen such that “tok _k_ ” retains its original position from the source prompt.

The results are illustrated in Fig. 5, where a comprehensive overview of the evaluation metrics can be found in §4.1. The
results indicate the stability of the token identity baseline across a range of token identity demonstrations, particularly
notable in the upper layers of the model. The _Single Token Prompt_ baseline overall seems to be less effective than using our
token ID baseline.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

Table 5: Comparison between the zero-shot feature extraction Patchscope and the logistic regression probe shows that despite using

_no training data_, the Patchscope has a significantly higher accuracy than the baseline in most tasks ( _p <_ 1 _e −_ 5). Pairwise t-statistics
and the corresponding p-values are included in the table. The columns corresponding to each method show accuracy (mean _±_ std).

**Task** **Probe** **Patchscope** **T-statistic** **p-value**

Commonsense

Factual

Fruit inside color 37 _._ 41 _±_ 6 _._ 58 37 _._ 99 _±_ 18 _._ 67 0 _._ 126 0 _._ 901
Fruit outside color 35 _._ 50 _±_ 3 _._ 09 **71** _._ **00** _±_ **13** _._ **26** _[∗∗]_ 12 _._ 426 _<_ 1 _e −_ 5
Object superclass **68** _._ **92** _±_ **10** _._ **69** _[∗∗]_ 55 _._ 71 _±_ 10 _._ 81 _−_ 5 _._ 25 _<_ 1 _e −_ 4
Substance phase 73 _._ 77 _±_ 3 _._ 74 **91** _._ **92** _±_ **1** _._ **73** _[∗∗]_ 25 _._ 647 _<_ 1 _e −_ 5
Task done by person 0 _±_ 0 **62** _._ **96** _±_ **16** _._ **513** _[∗∗]_ 19 _._ 632 _<_ 1 _e −_ 5
Task done by tool 10 _._ 14 _±_ 3 _._ 23 **48** _._ **12** _±_ **13** _._ **23** _[∗∗]_ 18 _._ 231 _<_ 1 _e −_ 5
Work location 0 _±_ 0 **13** _._ **58** _±_ **9** _._ **37** _[∗∗]_ 7 _._ 45990 _<_ 1 _e −_ 5

Company CEO 4 _._ 99 _±_ 2 _._ 56 **47** _._ **82** _±_ **13** _._ **89** _[∗∗]_ 16 _._ 700 _<_ 1 _e −_ 5
Country capital city 0 _±_ 0 **61** _._ **61** _±_ **14** _._ **14** _[∗∗]_ 22 _._ 426 _<_ 1 _e −_ 5
Country currency 17 _._ 70 _±_ 2 _._ 20 **50** _._ **95** _±_ **8** _._ **85** _[∗∗]_ 20 _._ 293 _<_ 1 _e −_ 5
Country largest city 0 _±_ 0 **67** _._ **78** _±_ **11** _._ **47** _[∗∗]_ 30 _._ 427 _<_ 1 _e −_ 5
Food from country 5 _._ 13 _±_ 3 _._ 66 **63** _._ **80** _±_ **11** _._ **34** _[∗∗]_ 26 _._ 710 _<_ 1 _e −_ 5
Person father 0 _±_ 0 **25** _._ **34** _±_ **8** _._ **42** _[∗∗]_ 15 _._ 482 _<_ 1 _e −_ 5
Person plays position in sport 75 _._ 89 _±_ 9 _._ 14 72 _._ 20 _±_ 7 _._ 21 _−_ 2 _._ 066 0 _._ 049
Person plays pro sport 53 _._ 87 _±_ 10 _._ 28 46 _._ 28 _±_ 14 _._ 19 _−_ 2 _._ 020 0 _._ 054
Product by company 58 _._ 91 _±_ 7 _._ 15 63 _._ 24 _±_ 10 _._ 74 1 _._ 757 0 _._ 091
Star constellation 17 _._ 54 _±_ 5 _._ 30 18 _._ 35 _±_ 5 _._ 06 _−_ 2 _._ 98 0 _._ 006
Superhero archnemesis 0 _±_ 0 **41** _._ **73** _±_ **18** _._ **72** _[∗∗]_ 11 _._ 47044 _<_ 1 _e −_ 5
Superhero person 0 _±_ 0 **28** _._ **32** _±_ **14** _._ **05** _[∗∗]_ 10 _._ 37461 _<_ 1 _e −_ 5

**C. More Details on Attribute Extraction Experiments**

**Dataset Details** We start from the factual and commonsense reasoning subsets introduced by Hernandez et al. (2023b) [10] .
This dataset includes 8 commonsense and 25 factual relations. Recall that each datapoint is representing a triplet ( _σ, ρ, ω_ ),
where _σ_ refers to the subject, _ρ_ represents the relation, and _ω_ stands for the object (see 4.2 for notation details). For each data
point representing the triplet ( _σ, ρ, ω_ ), we sample 5 utterances from Wikitext-103 dataset (Merity et al., 2016) including
_σ_ . We then truncate the sampled text to a window of random length up to 20 tokens that contains _σ_ . This constitutes our
source prompt, _S_ . Note that for each model, we filter the data to samples for which the underlying model correctly encodes
the tuple. For experiments with zero-shot target prompt _T_ that includes _ρ_ followed by _σ_, we autoregressively generate the
next 20 tokens, and only keep examples where _ω_ appears in the generation. For experiments with few-shot demonstrations
in _T_, after generating the next 20 tokens, we do an additional post-processing step. If the demonstration template of an
example is identified in the generation, all the following tokens would be dropped. The example is used for evaluation only
if _ω_ appears in the post-processed truncated generation. To have a reasonable amount of data for training the classification
probe baseline, tasks with fewer than 15 datapoints are dropped from the analysis. For GPT-J, 5 commonsense and 7 factual
reasoning tasks remain after applying the above post-processing steps.

**Results on Additional Tasks** We provide additional results on various factual and commonsense reasoning tasks. For a
comparison of the zero-shot feature extraction Patchscope and the logistic regression probe averaged across layers, see
t-statistic details in Tab. 5.

**Performance Breakdown Across Source Layers** Fig. 6 depicts how accuracy changes across source layers _ℓ_ _∈_ [1 _, . . ., L_ ].
Considering the early layers, we observe that the Patchscope consistently outperforms the baseline. This further confirms
our hypothesis that prior methods are particularly limited in surfacing information in the early layers, which often cannot
be decoded via linear functions. However, Patchscope is able to extract such attributes significantly earlier. Note that
this observation does not mean the attribute is _explicitly_ encoded in a representation, but that there is enough information
encoded such that the attribute can be extracted from the representation alone, without its original context, using the model’s
computation. In the middle layers, Patchscope works similarly or better than the baseline. Interestingly, we observe

8Conversations were collected from www.sharegpt.com.
9Pile dataset is available under MIT license.
10The dataset released by Hernandez et al. (2023b) is available under MIT license.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

that almost all cases where Patchscope performs worse than the baseline occur in later layers. Our interpretation is that
given the language modeling training objective, the representations shift toward next-token prediction in the later layers.
Therefore, the attribute of interest would not be as readily accessible via the model’s computation in these layers. This
interpretation is also aligned with recent findings that show no decline in using linear relational embedding in predicting _ω_
only when the next token also happens to be _ω_ (Hernandez et al., 2023b). We postulate that when the immediate next token
is not the attribute of interest, it does not necessarily mean the attribute information is lost, but rather it may not be as easily
accessible on the surface. We hypothesize that using a Patchscope with a more expressive mapping _f_ could improve
attribute extraction accuracy in the later layers, which we leave for future work to verify.

**Source-Target** **Layer** **Interplay** Fig. 7 visualizes the interaction between _ℓ_ and _ℓ_ _[∗]_ . These heatmaps show attribute
extraction success rate for the zero-shot feature extraction Patchscope for a fixed ( _ℓ, ℓ_ _[∗]_ ) combination. The lower left
quadrants show setups where both _ℓ_ and _ℓ_ _[∗]_ represent early to middle layers, and the success rate is maximal. The right half
of the heatmaps represents late _ℓ_, which achieves lower success rate due to token representations shifting toward next-token
prediction as discussed earlier. In addition, we notice lower success rate in the top half of the heatmaps which represents late
_ℓ_ _[∗]_ . It is worth noting that in this task, the accuracy is not only based on the immediate next-token prediction, but rather
whether _ω_ appears in the next 20 autoregressively generated tokens. The placeholder token “x” does still remain in the input,
and its representation persists in the early layers in the target computation. This explains why a lower attribute extraction
rate is observed in later _ℓ_ _[∗]_ values. We leave it to the future work to investigate adaptations to Patchscopes to control
contamination from the placeholder tokens and make them more amenable to late _ℓ_ _[∗]_ choices.

**Statistical** **Test** **Details** The null hypothesis is: “There is no significant difference between probing accuracy and
patchscope accuracy”. We calculate attribute extraction accuracy per layer, in a 40-layer model, across various tasks. Since
we run a test for each task independently (see Tab. 2), we use the conservative Bonferroni correction to address the multiple
comparison problem. That is, for a desired overall _α_ = 0 _._ 05, we consider _α/_ number of tasks = [0] 12 _[.]_ [05] [=] [0] _[.]_ [004][ for each]

individual task.

**D. Additional Information and Results on the Entity Resolution Experiment**

### D.1. Experimental Setup

Recall that we use a few-shot target prompt template for decoding an entity description: “subject1 :
description1, subject2 : description2, ..., subject _k_ : description _k_, x”, while patching
the last position which corresponds to “x”. Specifically, we use the following target prompt, obtained randomly: “Syria:
Country in the Middle East, Leonardo DiCaprio: American actor, Samsung: South
Korean multinational major appliance and consumer electronics corporation, x” and
task the model to generate the completion after the patched representation in “x”. For the subject description, which is
composed of _k_ = 3 random subject entities, we used the wptools [11] python package for obtaining a description of every
subject entity from Wikipedia.

### D.2. Additional Quantitative Results

In this section, we present the Rouge1 (Lin, 2004) and SBERT score (Reimers & Gurevych, 2019) results, as well as the
results for the Pythia models (Biderman et al., 2023). [12] In Fig. 8 and Fig. 9, we present the Rouge1 and the SBERT results,
respectively, complementary to the RougeL results in Fig. 3 from §4.3. Note that for Pythia, the smaller model (6.9B)
outperforms the larger one (12B), and hence, our cross-model patching method would not be able to improve the inspection
of the smaller model, unlike the trends we observed for Vicuna.

### D.3. Additional Qualitative Results

In this section, we provide more examples and discuss our observations about the gradual process of entity resolution. As
shown in Tab. 6 and Tab. 7, it is interesting to observe that the resolution process for the same input can look different
across models, suggesting they assign different likelihoods to different entities, and weigh context differently. For example,

[11https://github.com/siznax/wptools/](https://github.com/siznax/wptools/)
[12We used the package from https://www.sbert.net/, with the sentence-transformers/all-MiniLM-L6-v2 model.](https://www.sbert.net/)

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

(h) Person Plays Pro Sport

Attribute Extraction Acc.
star constellation (factual)

#### Source prompts: 56 - # Classes: 10

0 5 10 15 20 25
Source Layer ( )

(l) Star Constellation

Attribute Extraction Acc.
work location (commonsense)

#### Source prompts: 19 - # Classes: 4

0 5 10 15 20 25
Source Layer ( )

(p) Work Location

Attribute Extraction Acc.
food from country (factual)

#### Source prompts: 27 - # Classes: 10

0 5 10 15 20 25
Source Layer ( )

(d) Food From Country

Attribute Extraction Acc.
person plays pro sport (factual)

#### Source prompts: 131 - # Classes: 5

0 5 10 15 20 25
Source Layer ( )

Attribute Extraction Acc.
substance phase (commonsense)

#### Source prompts: 182 - # Classes: 3

0 5 10 15 20 25
Source Layer ( )

(a) Substance Phase

Attribute Extraction Acc.

company ceo (factual)

#### Source prompts: 50 - # Classes: 12

0 5 10 15 20 25
Source Layer ( )

(e) Company CEO

Attribute Extraction Acc.
person plays position in sport (factual)

#### Source prompts: 383 - # Classes: 7

0 5 10 15 20 25
Source Layer ( )

Attribute Extraction Acc.
country currency (factual)

#### Source prompts: 83 - # Classes: 14

0 5 10 15 20 25
Source Layer ( )

(c) Country Currency

Attribute Extraction Acc.
product by company (factual)

#### Source prompts: 219 - # Classes: 11

0 5 10 15 20 25
Source Layer ( )

(g) Product By Company

Attribute Extraction Acc.
fruit inside color (commonsense)

#### Source prompts: 39 - # Classes: 3

0 5 10 15 20 25
Source Layer ( )

(k) Fruit Inside Color

Attribute Extraction Acc.
task done by person (commonsense)

#### Source prompts: 33 - # Classes: 10

0 5 10 15 20 25
Source Layer ( )

(o) Task Done By Person

Attribute Extraction Acc.
task done by tool (commonsense)

#### Source prompts: 54 - # Classes: 12

0 5 10 15 20 25
Source Layer ( )

(b) Task Done By Tool

Attribute Extraction Acc.
fruit outside color (commonsense)

#### Source prompts: 56 - # Classes: 6

0 5 10 15 20 25
Source Layer ( )

(f) Fruit Outside Color

Attribute Extraction Acc.
object superclass (commonsense)

#### Source prompts: 173 - # Classes: 8

0 5 10 15 20 25
Source Layer ( )

(j) Object Superclass

Attribute Extraction Acc.
superhero archnemesis (factual)

#### Source prompts: 28 - # Classes: 7

0 5 10 15 20 25
Source Layer ( )

(i) Person Plays Position in Sport

(n) Superhero Archnemesis

Attribute Extraction Acc.

person father (factual)

#### Source prompts: 27 - # Classes: 7

0 5 10 15 20 25
Source Layer ( )

(m) Person Father

Attribute Extraction Acc.
country capital city (factual)

#### Source prompts: 94 - # Classes: 21

0 5 10 15 20 25
Source Layer ( )

(r) Country Capital City

Attribute Extraction Acc.
superhero person (factual)

#### Source prompts: 45 - # Classes: 12

0 5 10 15 20 25
Source Layer ( )

(q) Superhero Person

Attribute Extraction Acc.
country largest city (factual)

#### Source prompts: 91 - # Classes: 21

0 5 10 15 20 25
Source Layer ( )

(s) Country Largest City zero-shot feature extraction Patchscope works consistently better than the logistic regression probe in early layers, and mostly in mid
layers. There is a decline in Patchscope accuracy in later _ℓ_ as the source representations shift toward next-token prediction.

Figure 6: Feature extraction accuracy with respect to source layer ( _ℓ_ ) across various factual and commonsense reasoning tasks. The

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

Attribute Extraction Acc.
substance phase (commonsense)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(a) Substance Phase

Attribute Extraction Acc.

company ceo (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(e) Company CEO

Attribute Extraction Acc.
person plays position in sport (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

Attribute Extraction Acc.
food from country (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(d) Food From Country

Attribute Extraction Acc.
person plays pro sport (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(h) Person Plays Pro Sport

Attribute Extraction Acc.
star constellation (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(l) Star Constellation

Attribute Extraction Acc.
work location (commonsense)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(p) Work Location

20.0

17.5

15.0

12.5

10.0

7.5

5.0

2.5

0.0

Attribute Extraction Acc.
country currency (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(c) Country Currency

Attribute Extraction Acc.
product by company (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(g) Product By Company

Attribute Extraction Acc.
fruit inside color (commonsense)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(k) Fruit Inside Color

Attribute Extraction Acc.
task done by person (commonsense)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(o) Task Done By Person

(i) Person Plays Position in Sport

Attribute Extraction Acc.

person father (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(m) Person Father

20.0

17.5

15.0

12.5

10.0

7.5

5.0

2.5

0.0

Attribute Extraction Acc.
task done by tool (commonsense)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(b) Task Done By Tool

Attribute Extraction Acc.
fruit outside color (commonsense)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(f) Fruit Outside Color

Attribute Extraction Acc.
object superclass (commonsense)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(j) Object Superclass

Attribute Extraction Acc.
superhero archnemesis (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(n) Superhero Archnemesis

Attribute Extraction Acc.
superhero person (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(q) Superhero Person

Attribute Extraction Acc.
country capital city (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(r) Country Capital City

Attribute Extraction Acc.
country largest city (factual)

0 2 4 6 8 10 12 14 16 18 20 22 24 26
Source Layer ( )

(s) Country Largest City commonsense reasoning tasks. Each cell ( _ℓ, ℓ_ _[∗]_ ) in the heatmap shows the attribute extraction success rate where source and target layers
are fixed to _ℓ_ and _ℓ_ _[∗]_, respectively. Particularly, there is a higher success rate in the lower left quadrants, representing early to mid source
and target layer combinations. The right half of the heatmaps shows late source layers, where the source representation has shifted toward
next-token prediction, leading to a lower success rate in attribute extraction. The top half of the heatmaps shows late target layers. When
the accuracy is a function of more than a single next-token, the placeholder token representation still remains in the early layers, leading
to a lower attribute extraction rate. 20

Figure 7: The interaction between source and target layers in the zero-shot feature extraction Patchscope across various factual and

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

0.4

0.3

0.2

0.1

0.0

0.6

0.5

0.4

0.3

0.2

0.1

0.0

0 2 4 6 8
layer

0 2 4 6 8
layer

0.3

0.2

0.1

0.0

0 2 4 6 8
layer

(a) Vicuna (b) Pythia

Figure 8: Rouge1 scores of the generated descriptions against descriptions from Wikipedia.

0.6

0 2 4 6 8
layer

0.5

0.4

0.3

0.2

0.1

0.0

(a) Vicuna (b) Pythia

Figure 9: SBERT scores of the generated descriptions against descriptions from Wikipedia.

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|_7 ~~0.~~_7 ~~0.~~||~~.6~~_0 ~~.6~~_0|_6_0_6_0|_5_0_6_0|_4_0_5_0|_4_0_4_0|
|_7 ~~0.~~||~~.6~~_0|_6_0|_6_0|_6_0|_5_0|
|_7 ~~0.~~||~~.5~~_0|_6_0|_6_0|_7_0|_6_0|
|_7 ~~0.~~||~~.5~~_0|_5_0|_6_0|_7_0|_7_0|
|_6 ~~0.~~||~~.5~~_0|_6_0|_6_0|_7_0|_7_0|

0 5 10 15 20 25 30 35
target layer

(a) Vicuna: _M←_ 7B _, M_ _[∗]_ _←_ 13B

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|_8_0_8_0|_8_0_8_0|_7_0_8_0|_6_0_7_0|_5_0_6_0|~~.4~~_0 ~~.5~~_0|_3 ~~0.~~_4 ~~0.~~|
|_8_0|_8_0|_8_0|_7_0|_7_0|~~.6~~_0|_5 ~~0.~~|
|_7_0|_8_0|_7_0|_7_0|_7_0|~~.7~~_0|_6 ~~0.~~|
|_7_0|_7 ~~0.~~|_7_0|_7_0|_7_0|~~.7~~_0|_7 ~~0.~~|
|_7_0|_7_0|_7_0|_7_0|_7_0|~~.7~~_0|_7 ~~0.~~|

0 5 10 15 20 25 30 35
target layer

(b) Pythia: _M←_ 6.9B _, M_ _[∗]_ _←_ 12B

1.0

0.8

0.6

0.4

0.2

0.0 as Vicuna 13B processes “Will Smith”, it goes from _“Smithsonian Museum”_ to the _“Smith rock band”_ to the actor
and rapper, _“Will Smith”_ . However, Pythia 12B starts with _“Smith & Wesson weapon manufacturing company”_ before it
resolves the entity as the American actor, _“Will Smith”_ .

Figure 10: Precision@1 scores ( _↑_ is better) on next-token prediction estimation in Vicuna and Pythia with cross-model Patchscopes

We also observe another phenomenon which we refer to as placeholder contamination. It is the case where the remaining
representation of the placeholder entity “x” in the early layers interferes with the model’s capability in generating descriptions
for the patched token. For example, see Vicuna 13B response to “Paris Hilton” entity in Tab. 6. First, we see the
gradual process of going from _“Rigatoni”_ pasta, to _“Hilton Hotels”_, to the socialite _“Paris Hilton”_ in layers 1-6. But in
layers 7 and onward, the generation seems to describe the placeholder token “x”rather than the “Paris Hilton” entity:
_“Placeholder for a variable or concept”_, _“Variable representing any number of things or concepts”_ or _“x is a placeholder”_ .

For future, we would like to quantify these qualitative observations, study to what extend this contamination can be mitigated
with a different placeholder choice, and why some models might be more susceptible to this contamination than others.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

|3 1.|9 2.|5 2.|7 3.|1 3.|9 4.|5 5.|
|---|---|---|---|---|---|---|
||||||||
|_5_2_4_2|_3_2_8_3|_8_2_3_2|_6_2_8_2|_8_3_4_2|_6_4_7_3|_4_4_2_3|
|_5_3|_6_4|_3_3|_7_2|_8_1|_9_2|_1_2|
|_8_4|_8_6|_1_5|_4_4|_1_2|_4_1|_9_1|
|_1_5|_3_3|_9_3|_3_2|_7_1|_8_1|_7_1|

0 5 10 15 20 25 30 35
target layer

(a) Vicuna: _M←_ 7B _, M_ _[∗]_ _←_ 13B

|1 2.|3 2.|8 4.|1 5.|1 6.|4 8.|3 5.|
|---|---|---|---|---|---|---|
|_2_1  |_2_1  |_5_2  |_4_3  |_0_3  |_8_4  |_3_4  |
|_2_1_3_1|_2_1_2_1|_3_2_4_1|_0_2_7_1|_2_2_8_2|_8_3_1_2|_4_4_6_3|
|_4_1|_3_1|_5_1|_8_1|_6_1|_8_2|_0_3|
|_6_1|_4_1|_6_2|_1_1|_7_1|_7_1|_7_2|
|_0_1|_5_1|_8_2|_3_1|_9_1|_8_1|_6_2|

0 5 10 15 20 25 30 35
target layer

(b) Pythia: _M←_ 6.9B _, M_ _[∗]_ _←_ 12B

Figure 11: Surprisal scores ( _↑_ is better) on next-token prediction estimation in Vicuna and Pythia cross-model Patchscopes.

**E. Additional Results on the Cross-Model Patching Experiment**

Fig. 10 depicts the Precision@1 scores for different combinations of source-target layers, showing that patching with a
simple affine Patchscope proves to be effective with precision of up to 0.7 and 0.8 for Vicuna and Pythia, respectively.
Specifically, patching representations to an early layer of the larger model seems to be the most effective. Furthermore, it
appears that there is a subtle matching among some layers of the models, with the diagonal consistently exhibiting higher
values. As shown in Fig. 11, similar trends are observed in the surprisal results for both models. Overall, these results show
that, when _M_ _[∗]_ and _M_ are from the same model family, it is possible to leverage _M_ _[∗]_ for decoding information from the
representations of _M_ . Notably, our findings extend observations by Csiszarik et al.´ (2021) from patching representations
across deep convolutional neural networks with the same architecture but different initializations.

**F. More Details on the Multi-Hop Reasoning Experiment**

**Data** Building on Hernandez et al. (2023b), we systematically generate all valid multi-hop factual and commonsense
reasoning queries where the object _ω_ 1 in the first triplet _τ_ 1 is equal to the subject _σ_ 2 in the second triplet _τ_ 2. This process
yields 1,104 multi-hop reasoning samples. We then filter examples to the ones where the model accurately represents both
triplets independently. Tab. 8 summarizes information about the samples where Vicuna (13B) correctly represents each
reasoning step. Out of 1,104 samples that require two steps of commonsense or factual reasoning, 46 satisfy the above
criteria, with 8 unique relation combinations.

**More Detailed Results** Fig. 12 shows the interaction between _ℓ_ and _ℓ_ _[∗]_ and how it affects the success rate. Patching
representations from most source layers _ℓ_ into early-to-mid _ℓ_ _[∗]_ (6-16) is the most effective in making the right prediction.
Our interpretation is that patching into late _ℓ_ _[∗]_ is not effective because _ρ_ 2 has already been processed and the result has been
copied to the last position, therefore it cannot incorporate the proxy of _σ_ 2. We also observe that _ℓ_ _[∗]_ _≤_ _ℓ_ is more successful
on average.

**G. Compute Resources**

All experiments were conducted using A100 80GB GPUs, with the exception of experiments with GPT-J (Wang &
Komatsuzaki, 2021) where we used A100 40GB GPUs.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

Table 6: Additional qualitative examples illustrating entity resolution in **Vicuna 13B** . The expressive generations show that as we go

through the layers, more tokens from the context get integrated into the current representation, referred to as “Tokens Covered”. Here,
_M ←_ Vicuna 13B _, M_ _[∗]_ _←M, ℓ_ _[∗]_ _←_ _ℓ_ .

Notes/Entity Described by the
_S_ Tokens Covered _ℓ_ Generation
Generation

"Smith" 1 sonian: National museum and research complex in Washington, D.C. **Smith** sonian Museum

An amalgam of **Smith** band
"Smith" 2 sonas: American rock band
and SONA band

"Will Smith" 3 : American actor and rapper **Will Smith**

"Victoria" 1 s Secret: American multinational women’s lingerie, beauty and body-care products company **Victoria** ’s Secret

"Victoria" 2 s Secret: American lingerie and beauty products retailer **Victoria** ’s Secret

"Queen Victoria" 3 : Queen of the United Kingdom of Great Britain and Ireland from 1837 to 1901 **Queen Victoria**

"Queen Victoria" 4-8 : Queen of the United Kingdom of Great Britain and Ireland from 1837 until her death in 1901 **Queen Victoria**

"Queen Victoria" 9 : Queen of England (reigned from 1837-1901) **Queen Victoria**

"Queen Victoria" 10 : 19th century British queen who ruled from 1837-1901 **Queen Victoria**

"ton" 1 y: a type of pasta Riga **ton** i/Riga **ton** y/Riga **ton** ´e

"Hilton" 2 Hotels: American multinational hospitality company **Hilton** Hotel

"Hilton" 3 : Hotel chain founded by Conrad Hilton **Hilton** Hotel

"Hilton" 4 : Hotel and resorts company **Hilton** Hotel

"Paris Hilton" 5-6 : Socialite and television personality **Paris Hilton**

N/A 7 : Placeholder for a variable or concept Placeholder contamination

N/A 8-9 : Variable representing any number of things or concepts Placeholder contamination

N/A 10 : x is a placeholder Placeholder contamination

"Bond" 1 : Unit of measurement for the strength of a liquid **Bond** Unit

: James Bond, the fictional British secret agent created by novelist Ian Fleming and portrayed by actor
"James Bond" 2 **James Bond**
Daniel Craig in the 2006 film ”Casino Royale,”

"James Bond" 3-5 : Fictional British secret agent **James Bond**

"James Bond" 6 : Code name for fictional British secret agent James Bond **James Bond**

"James Bond" 7 : Code name for a fictional British secret agent created by novelist Ian Fleming **James Bond**

: Subject of a fictional British secret agent, created by novelist Ian Fleming and portrayed in a series
"James Bond" 8 **James Bond**
of films.

James Bond 9 : Subject of the 007 novels and films **James Bond**

N/A 10 : Variable representing the number of the film in the James Bond series Placeholder contamination

& Johnson: American multinational corporation in the fields of pharmaceuticals, medical devices, and
"Johnson" 1-2 **Johnson** & Johnson
consumer packaged goods

"Dwayne Johnson" 3-4 : American actor and professional wrestler, also known as ”The Rock” **Dwayne Johnson**

"Dwayne Johnson" 5 : Also known as ”The Rock,” American actor and professional wrestler **Dwayne Johnson**

"Dwayne Johnson" 6 : Former professional wrestler and actor **Dwayne Johnson**

N/A 7 : Abbreviation for the United States Placeholder contamination

N/A 8 : Variable representing any number of things or concepts Placeholder contamination

N/A 9 : Variable representing any number of other options. Placeholder contamination

N/A 10 : Variable representing any number of other options or possibilities. Placeholder contamination

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

Table 7: Additional qualitative examples illustrating entity resolution in **Pythia 12B** . The expressive generations show that as we go

through the layers, more tokens from the context get integrated into the current representation, referred to as “Tokens Covered”. Here,
_M ←_ Pythia 12B _, M_ _[∗]_ _←M, ℓ_ _[∗]_ _←_ _ℓ_ .

_S_ Tokens Covered _ℓ_ Generation Notes/Entity Described by the Generation

"Smith" 1 & Wesson: American firearms manufacturer **Smith** & Wesson

"Will Smith" 2 : American actor **Will Smith**

"Great" 1 Britain: Country in the European Union **Great** Britain

"the Great" 2 Wall Street Crash of 1929: Financial crisis in the United States **the Great** Depression

"the Great" 3 Wall Street Bubble: The Great Depression **the Great** Depression

Wall Street
"the Great" 4 Wall Street: Wall Street in New York City
(related to the Great Depression)

"Alexander the Great" 5 : Ancient Greek ruler, and the first to rule all of the then known world **Alexander the Great**

"Washington" 1 : Capital of the United States **Washington** D.C.

"Washington" 2 : American capital city **Washington** D.C.

"George Washington" 3 : American president **George Washington**

"George Washington" 4-6 : American revolutionary **George Washington**

"George Washington" 7 : American president **George Washington**

"George Washington" 8 : George Washington **George Washington**

"George Washington" 9 : George Washington, the first President of the United States **George Washington**

: George Washington, Thomas Jefferson, Abraham Lincoln, and
"George Washington" 10 **George Washington**
Theodore Roosevelt: American presidents

"c" 1 loud: Apple’s cloud-based storage service **c** loud

"Titanic" 2 : Sinking ocean liner **Titanic**

"Titanic" 3 : The ship that sank in the Atlantic Ocean in 1912 **Titanic**

"Titanic" 4-6 : British passenger ship **Titanic**

"Titanic" 7-9 : The ship that sank **Titanic**

"Titanic" 10 : Titanic, Titanic: British luxury passenger liner **Titanic**

"Live" 1 Nation: American concert promoter **Live** Nation

"Live" 2 Nation: American television network **Live** Nation

"Saturday Night Live" 3 : American television program **Saturday Night Live**

"Saturday Night Live" 4-10 : American television comedy show **Saturday Night Live**

"Four" 1 Seasons: American hotel chain **Four** Seasons Hotel

"Four" 2 Seasons: The four seasons of the year **Four** Seasons (of the year)

"ighty-Four" 3 Lions: Chinese professional football club Cangzhou M **ighty** Lions F.C.

"Nineteen Eighty-Four" 4 : Number of the novel 1984 by George Orwell **Nineteen Eighty-Four** (Novel)

"Nineteen Eighty-Four" 5 : George Orwell’s novel, and the 1984 film. **Nineteen Eighty-Four** (Novel & Film)

"Nineteen Eighty-Four" 6 : George Orwell’s novel **Nineteen Eighty-Four** (Novel)

"Nineteen Eighty-Four" 7 : George Orwell: British novelist **Nineteen Eighty-Four** (Novel)

"Nineteen Eighty-Four" 8 : 1984 novel by George Orwell **Nineteen Eighty-Four** (Novel)

: 1984 novel by George Orwell, and the 1984 film adaptation directed
"Nineteen Eighty-Four" 9-10 **Nineteen Eighty-Four** (Novel & Film)
by Michael Radford.

**Patchscopes:** **A Unifying Framework for Inspecting Hidden Representations of Language Models**

Table 8: Sample statistics for the multi-hop reasoning experiment where _M_ correctly represents both _τ_ 1 and _τ_ 2.

_ρ_ 1 _ρ_ 2 # Samples
Company CEO Person Father 4
Food from Country Country Capital City 10
Food from Country Country Currency 3
Food from Country Country Language 9
Food from Country Country Largest City 11
Person Father Person Father 1
Person Father Person Mother 1
Product by Company Company CEO 7
Total 46

Self-correction in Multi-hop Reasoning

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||

0 2 4 6 8 101214161820222426283032343638
Source Layer ( )

0.12

0.10

0.08

0.06

0.04

0.02

0.00 cases correspond to early-to-mid _ℓ_ _[∗]_ . We observe higher cumulative success rate in the lower right triangle which corresponds to _ℓ_ _[∗]_ _≤_ _ℓ_ .

Figure 12: The interaction between source ( _ℓ_ ) and target ( _ℓ_ _[∗]_ ) layers in correcting multi-hop reasoning errors. The majority of success
