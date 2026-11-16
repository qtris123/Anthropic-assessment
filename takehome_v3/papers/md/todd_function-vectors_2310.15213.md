# Function Vectors in Large Language Models

- **Authors:** Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, David Bau
- **Date:** 2023-10-23
- **Source:** https://arxiv.org/abs/2310.15213
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `todd_function-vectors_2310.15213`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Published as a conference paper at ICLR 2024

**Eric Todd** _[∗]_ **,** **Millicent L. Li, Arnab Sen Sharma, Aaron Mueller,**
**Byron C. Wallace, and David Bau**
Khoury College of Computer Sciences, Northeastern University

ABSTRACT

We report the presence of a simple neural mechanism that represents an input-output function as a vector within autoregressive transformer language models
(LMs). Using causal mediation analysis on a diverse range of in-context-learning
(ICL) tasks, we find that a small number attention heads transport a compact
representation of the demonstrated task, which we call a _function vector_ (FV). FVs
are robust to changes in context, i.e., they trigger execution of the task on inputs
such as zero-shot and natural text settings that do not resemble the ICL contexts
from which they are collected. We test FVs across a range of tasks, models, and
layers and find strong causal effects across settings in middle layers. We investigate
the internal structure of FVs and find while that they often contain information that
encodes the output space of the function, this information alone is not sufficient to
reconstruct an FV. Finally, we test semantic vector composition in FVs, and find
that to some extent they can be summed to create vectors that trigger new complex
tasks. Our findings show that compact, causal internal vector representations of
function abstractions can be explicitly extracted from LLMs.

1 INTRODUCTION

Since the study of the lambda calculus (Church, 1936), computer scientists have understood that
the ability for a program to carry references to its own functions is a powerful idiom. Function
references can be helpful in many settings, allowing expression of complex control flow through
deferred invocations (Sussman, 1975), and enabling flexible mappings from inputs to a target task. In
this paper we report evidence that autoregressive transformers trained on large corpora of natural text
develop a rudimentary form of function references.

Our results begin with an examination of in-context learning (ICL; Brown et al., 2020). ICL
mechanisms have previously been studied from the perspective of making copies (Olsson et al., 2022)
and from a theoretical viewpoint (Von Oswald et al., 2023; Garg et al., 2022; Dai et al., 2023), but
the computations done by large models to generalize and execute complex ICL functions are not yet
fully understood. We characterize a key mechanism of ICL execution: _function vectors_ (FVs), which
are compact vector representations of input-output tasks that can be found within the transformer
hidden states during ICL. An FV does not directly perform a task, but rather it triggers the execution
of a specific procedure by the language model (Figure 1).

```
arrive:depart, small:big, common: rare The word "fast" means

```

```
amount:cantidad, win:ganar, dreams:

```

```
sueños

```

```
slow

rápido

```

+ 𝑣% = Antonym FV 𝑣%= Eng-Spanish FV 𝐡ℓ : Hidden-state
intervention

```
The word "fast" means

```

Novel zero-shot prompt Execution of FV

Figure 1: An overview of **function vectors** (FVs). An FV is extracted from activations induced by
in-context examples of (a) antonym generation or (b) English to Spanish translation, and then inserted
into an unrelated context to induce generation of (c) a new antonym or (d) translation.

_∗_ [Correspondence to todd.er@northeastern.edu.](mailto:todd.er@northeastern.edu) [Open-source code and data available at functions.baulab.info.](https://functions.baulab.info)

Function vectors arise naturally when applying causal mediation analysis (Pearl, 2001; Vig et al.,
2020; Meng et al., 2022; 2023; Wang et al., 2022a) to identify the flow of information during ICL.
We describe an activation patching procedure to determine the presence of a handful of attention
heads that mediate many ICL tasks. These heads work together to transport a function vector that
describes the task; the FV can be formed by summing outputs of the causal attention heads.

We test the hypothesis that function vectors are a general mechanism spanning many types of
functions. To quantify the role and efficacy of function vectors, we curate a data set of over 40 diverse
ICL tasks of varying complexity. We calculate FVs for these tasks and investigate impact of FVs in
triggering those functions across a variety of LMs scaling up from 6B to 70B parameters.

We further ask whether FVs are portable: are the effects of an FV limited to contexts very similar to
those where it is extracted, or can an FV apply in diverse settings? We compare the effects of FVs
when inserted into diverse input contexts including differently-formatted forms, zero-shot formats,
and natural text contexts. We find that FVs are remarkably robust, typically triggering function
execution even in contexts that bear no resemblance to the original ICL context.

A key question is whether the action of FVs can be explained by word-embedding vector arithmetic (Mikolov et al., 2013; Levy & Goldberg, 2014; Merullo et al., 2023). We examine decodings of
FVs (Nostalgebraist, 2020), and find that although FVs often encode a function’s output vocabulary,
those vocabularies do not fully identify an FV. In other words, to invoke functions, FVs need to carry
some additional information beyond their encoding of the top vocabulary words.

Finally, we investigate whether the space of FVs has its own vector algebra over functions rather than
words. We construct a set of composable ICL tasks, and we test the ability of FVs to obey vector
algebra compositions. Our findings reveal that, to some extent, vector compositions of FVs produce
new FVs that can execute complex tasks that combine constituent tasks. We emphasize that FV vector
algebra is distinct from semantic vector algebra over word embeddings: for example, composed FV
vectors can specify nonlinear tasks such as calculating the antonym of a word, that cannot themselves
be implemented as a simple embedding-vector offset (Appendices A, L).

2 METHOD

2.1 A MOTIVATING OBSERVATION

When a transformer processes an ICL prompt with exemplars demonstrating task _t_, do any hidden
states encode the task itself? We seek causal features rather than just correlations.

We can investigate this question with the following simple test: Gather a set of ICL prompts _Pt_ for
the task _t_ and compute the average activation **h** [¯] _[t]_ _ℓ_ [at the last token of each prompt at a particular layer]

_ℓ_ of the model (Figure 2a). Then perform an intervention where **h** [¯] _[t]_ _ℓ_ [is added to the representation]

after _ℓ_ when the transformer completes a previously unseen zero-shot prompt (Figure 2b).

Surprisingly, we find that adding the average activations in this way at particular layers induces the
model to perform the task in the new context. For example, if _t_ = _antonym_, the red line in Figure
2c shows that adding **h** [¯] _[t]_ 12 [at layer 12 in GPT-J causes the model to produce antonyms in a zero-shot]
context, with 24 _._ 3% accuracy. That suggests that **h** [¯] _[t]_ 12 [does encode the antonym task.]

Surprisingly, we find that adding the average activations in this way at particular layers induces the
model to perform the task in the new context. For example, if _t_ = _antonym_, the red line in Figure
2c shows that adding **h** [¯] _[t]_ 12 [at layer 12 in GPT-J causes the model to produce antonyms in a zero-shot]
context, with 24 _._ 3% accuracy. That suggests that **h** [¯] _[t]_ 12 [does encode the antonym task.]

The effect of **h** [¯] _[t]_ _ℓ_ [leads us to ask:] [Can we distill a more effective hidden-state representation of the task]

_t_ ? In the rest of Section 2 we describe an analysis of the mechanisms of ICL that leads to a function
vector representation _vt_ whose stronger causal effects are shown as a green line in Figure 2c.

The effect of **h** [¯] _[t]_

Zero-Shot Intervention

Layer

(a)

Average Layer Activation (b) Zero-Shot Intervention (c)

```
= complex

= decode

```

```
simple:

encode:

```

**+**

**+**

Figure 2: A motivating observation: (a) an average activation is computed over a set of antonym ICL prompts,
and (b) added to a zero-shot context, which produces the opposite of unseen words. (c) Systematic effects (in
red) for adding **h** [¯] _[t]_ _ℓ_ [in middle layers of the network; even stronger effects are seen by the FV (in green).]

2.2 FORMULATION

An autoregressive transformer language model _f_ takes an input prompt _p_ and outputs a next-token
distribution _f_ ( _p_ ) over vocabulary _V_ ; we write _f_ ( _p_ )[ _y_ ] _∈_ [0 _,_ 1] for the predicted probability of output
_y_ _∈V_ in response to input _p_ . Internally, _f_ comprises _L_ layers; we examine their calculations at the
last token position. Each layer _ℓ_ _≤_ _L_ has a vector representation of the last token, **h** _ℓ_ _∈_ R _[d]_, that
is computed from the previous layer as **h** _ℓ_ = **h** _ℓ−_ 1 + _mℓ_ + [] _j≤J_ _[a][ℓj]_ [, where] _[ m][ℓ]_ [is the output of]

a multilayer perceptron, and _aℓj_ is the projection of the output of the _j_ th attention head (out of _J_
heads) into the hidden state at layer _ℓ_ . This definition of _aℓj_ _∈_ R _[d]_ adopts the framing of Elhage et al.
(2021) rather than that of Vaswani et al. (2017) (see Appendix B for details). Attention heads and
hidden states can be viewed as functions of transformer input, so we shall write _aℓj_ ( _p_ ) or **h** _ℓ_ ( _p_ ) to
denote their values when the transformer processes input _p_ . The transformer’s decoder _D_ maps the
last layer hidden state to the output distribution _D_ ( **h** _L_ ( _p_ )) = _f_ ( _p_ ).

For each task _t ∈T_ in our universe of ICL tasks _T_ we have a data set _Pt_ of in-context prompts _p_ _[t]_ _i_

For each task _t ∈T_ in our universe of ICL tasks _T_ we have a data set _Pt_ of in-context prompts _p_ _[t]_ _i_ _[∈]_ _[P][t]_ [.]

Each prompt _p_ _[t]_ _i_ [is a sequence of tokens with] _[ N]_ [input-output exemplar pairs][ (] _[x, y]_ [)][ that demonstrate]

the same underlying task _t_ mapping between _x_ and _y_, and one _query_ input _xiq_ corresponding to a
target (correct) response _yiq_ that is not part of the prompt, that should be predicted by the LM if it
generalizes correctly. We focus our analysis on successful ICL by including in _Pt_ only prompts _p_ _[t]_ _i_

where the prediction _f_ ( _p_ _[t]_ [)][ ranks the correct answer] _[ y][iq]_ [highest.] [We write one ICL prompt as]

_i_ [)][ ranks the correct answer] _[ y][iq]_ [highest.] [We write one ICL prompt as]

_p_ _[t]_ _i_ [= [(] _[x][i]_ [1] _[, y][i]_ [1][)] _[,][ · · ·]_ _[,]_ [ (] _[x][iN]_ _[, y][iN]_ [)] _[, x][iq]_ []] (1)

We also make use of uninformative ICL prompts _p_ ˜ _[t]_ _i_ _[∈]_ _[P]_ [˜] _[t]_ [ for which the labels are shuffled; we use]

the tilde to indicate a shuffled prompt _p_ ˜ _[t]_ [=] [[(] _[x][i]_ [1] _[,]_ [ ˜] _[y][i]_ [1][)] _[,][ · · ·]_ _[,]_ [ (] _[x][iN]_ _[,]_ [ ˜] _[y][iN]_ [)] _[, x][iq]_ []] [in] [which] [there] [is] [no]

We also make use of uninformative ICL prompts _p_ ˜ _[t]_

the tilde to indicate a shuffled prompt _p_ ˜ _[t]_ _i_ [=] [[(] _[x][i]_ [1] _[,]_ [ ˜] _[y][i]_ [1][)] _[,][ · · ·]_ _[,]_ [ (] _[x][iN]_ _[,]_ [ ˜] _[y][iN]_ [)] _[, x][iq]_ []] [in] [which] [there] [is] [no]

systematic relationship between any of the _xik_ and _y_ ˜ _ik_ .

2.3 CAUSAL MEDIATION TO EXTRACT FUNCTION VECTORS FROM ATTENTION HEADS

To distill the information flow during ICL, we apply causal mediation analysis.

Given a transformer model _f_ and an ICL prompt _p_ _[t]_

Given a transformer model _f_ and an ICL prompt _p_ _[t]_ _i_ _[∈]_ _[P][t]_ [from] [a] [dataset] [representing] [task] _[t]_ [,] [we]

prompt the model with only input-output pairs ( _xi, yi_ ). Therefore, the LM must infer the implicit
relationship between these ( _x, y_ ) pairs to correctly predict the answer given a novel query _xiq_ . We
seek to identify model components with a causal role in the prediction of _yiq_ . We restrict our analysis
to the attention heads since those are the components used by transformer LMs to move information
between different token positions (Vaswani et al., 2017; Elhage et al., 2021). Formally, for each
attention head _aℓj_ and task dataset _Pt_, we take the mean of task-conditioned activations ¯ _a_ _[t]_ _ℓj_ [as]

_ℓj_ [as]

_a_ ¯ _[t]_ _ℓj_ [=]

_|Pt|_

_p_ _[t]_ _i_ _[∈][P][t]_

_aℓj_ ( _p_ _[t]_ _i_ [)] _[.]_ (2)

We then run the model on an uninformative ICL prompt _p_ ˜ _[t]_ _i_ _[∈]_ _[P]_ [˜] _[t]_ [where each] _[ x]_ [ is matched with a]

_random_ output _p_ ˜ _[t]_ [= [(] _[x][i][,]_ [ ˜] _[y][i]_ [)]][.] [Now, the model is less likely to generate the correct output] _[ y][q]_ [as it]

We then run the model on an uninformative ICL prompt _p_ ˜ _[t]_

_random_ output _p_ ˜ _i_ [= [(] _[x][i][,]_ [ ˜] _[y][i]_ [)]][.] [Now, the model is less likely to generate the correct output] _[ y][q]_ [as it]

cannot infer the relationship from incorrect ICL examples (notwithstanding the observation from Min
et al. (2022) that some tasks can be guessed from incorrect labels). While running the model on _p_ ˜ _[t]_ _i_ [,]

we replace an attention head activation _aℓj_ with mean task-conditioned activation ¯ _a_ _[t]_ _ℓj_ [(Eq. 2) and]

we replace an attention head activation _aℓj_ with mean task-conditioned activation ¯ _aℓj_ [(Eq. 2) and]

measure its causal indirect effect (CIE) towards recovering the correct answer _yq_ as

CIE( _aℓj_ _|_ _p_ ˜ _[t]_ _i_

_[t]_ _i_ [) =] _[ f]_ [(˜] _[p][t]_ _i_

_[t]_ _i_ _[|][ a][ℓj]_ [:= ¯] _[a][t]_

_[t]_ _ℓj_ [)[] _[y][iq]_ []] _[ −]_ _[f]_ [(˜] _[p][t]_ _i_

_i_ [)[] _[y][iq]_ []] _[.]_ (3)

The intuition here is to measure the degree to which using the “correct” mean attention head output
_a_ ¯ _[t]_ _ℓj_ [—computed over the] _[ uncorrupted]_ [ prompts for task] _[ t]_ [—increases the mass assigned to the target]

response _yiq_, relative to the likelihood of this token under the corrupted prompt _p_ ˜ _[t]_ _i_ [.] [A larger value]

implies that the corresponding head is more influential in promoting the correct response.

Then each attention head’s average indirect effect (AIE) is calculated by averaging this difference
across all tasks _t ∈T_ and (corrupted) prompts:

AIE( _aℓj_ ) = _|T |_

_t∈T_

_|P_ [˜] _t|_

_p_ ˜ _[t]_ _i_ _[∈]_ _P_ [˜] _t_

CIE( _aℓj_ _|_ _p_ ˜ _[t]_ _i_ [)] (4)

AIE heads appear in middle layers. Attention weights are strongest on the output tokens of each exemplar.

Figure 3: (a) Average indirect effect across all tasks _T_ for each attention head in GPT-J, and (b) the
top 10 heads’ weights on individual tokens for one example prompt _p_ _[t]_ _i_ [.] [The most strongly implicated]

To identify the set of attention heads with the strongest causal effects, we repeat this process for each
attention head _aℓj_ in _f_, for all layers _ℓ_, and all head indices _j_ . We gather the attention heads with
highest AIE over all layers as the set _A_ . [1]

Figure 3a shows the AIE per attention head in GPT-J over many tasks (see Appendix G for larger
models). The 10 attention heads with highest AIE (which make up _A_ ) are highlighted in pink (square
outlines) and are clustered primarily in early-middle layers of the network. The average attention
pattern of these heads at the final token is shown for two tasks in Figure 3b. These heads primarily
attend to token positions corresponding to example outputs; this observation is consistent with the
high salience of ICL label tokens observed by Wang et al. (2023a) and while this resembles the same
prefix-matching attention pattern as “induction heads” (Elhage et al., 2021; Olsson et al., 2022) not
all heads in _A_ reproduce this pattern on other contexts with repeated tokens (Appendix H).

Due to their high causal influence across many tasks (see Appendix G for breakouts by task), we
hypothesize that this small set of heads is responsible for transporting information identifying the
demonstrated ICL task. We can represent the contribution of _A_ as a single vector by taking the sum
of their average outputs, over a task, which we call a _function vector_ (FV) for task _t_ :

_vt_ =

_aℓj_ _∈A_

_a_ ¯ _[t]_ _ℓj_ (5)

We can then test the causal effect of an FV by adding it to hidden states at any layer _ℓ_ as the model
resolves a prompt and measuring its performance in executing the task (Appendix B).

3 EXPERIMENTS

**Models.** We deploy a series of decoder-only autoregressive language models; each is listed and
described in Table 1. We use huggingface implementations (Wolf et al., 2020) of each model.

**Tasks.** We construct a diverse array of over 40 relatively simple tasks to test whether function
vectors can be extracted in diverse settings. To simplify the presentation of our analysis, we focus on
a representative sample of 6 tasks:

- **Antonym.** Given an input word, generate the word with opposite meaning.

- **Capitalize.** Given an input word, generate the same word with a capital first letter.

- **Country-Capital.** Given a country name, generate the capital city.

- **English-French.** Given an English word, generate the French translation of the word.

- **Present-Past.** Given a verb in present tense, generate the verb’s simple past inflection.

- **Singular-Plural.** Given a singular noun, generate its plural inflection.

All other tasks are described in Appendix E.

1For GPT-J, we use _|A|_ = 10 attention heads. For larger models, we scale the number of attention heads we
use approximately proportionally to the number of attention heads in the model. (We use 20 heads for Llama 2
(7B), 50 for Llama 2 (13B) & GPT-NeoX, and 100 for Llama 2 (70B).)

Table 1: Models used in this study. We focus on decoder-only autoregressive language models that
are capable of ICL. For each model, we present the number of parameters, the number of layers _|L|_,
and number of attention heads per layer _J_ = _|aℓ|_ .

**Model** **Huggingface ID** **Citation** **Parameters** **Training Tokens** _|L|_ _|aℓ|_

GPT-J EleutherAI/gpt-j-6b (Wang & Komatsuzaki, 2021) 6B 402B 28 16
GPT-NeoX EleutherAI/gpt-neox-20b (Black et al., 2022) 20B 472B 44 64

Llama 2 meta-llama/Llama-2-7b-hf (Touvron et al., 2023) 7B 2T 32 32
Llama 2 meta-llama/Llama-2-13b-hf (Touvron et al., 2023) 13B 2T 40 40
Llama 2 meta-llama/Llama-2-70b-hf (Touvron et al., 2023) 70B 2T 80 64

Table 2: Average accuracy across 6 tasks (macro-averaged across random seeds) for both shuffled-label and zero-shot contexts - adding the FV increases performance of the task compared to the base
model in both contexts. For GPT-J we compare to layer averages (Section 2.1) and find that our FV
works best. We also report results for both settings on an additional 34 tasks for GPT-J+FV and
Llama 2 (70B)+FV. More details on additional tasks in Appendix E.3.

Shuffled-Label Zero-Shot

[( _xi_ 1 _,_ ˜ _yi_ 1) _, · · ·_ _,_ ( _xiN_ _,_ ˜ _yiN_ ) _, xiq_ ] [ _xiq_ ]

GPT-J (baseline on uninformative input) 39 _._ 1 _±_ 1 _._ 2% 5 _._ 5 _±_ 0 _._ 8%
+ **h** [¯] _[t]_ _ℓ_ [Layer average (Section 2.1)] 79 _._ 5 _±_ 3 _._ 1% 9 _._ 5 _±_ 1 _._ 8%

+ _vt_ FV (Eq. 5) **90** _._ **8** _±_ 0 _._ 9% **57** _._ **5** _±_ 1 _._ 7%

GPT-NeoX (baseline on uninformative input) 32 _._ 5 _±_ 1 _._ 3% 6 _._ 7 _±_ 0 _._ 1%
+ _vt_ FV **90** _._ **7** _±_ 0 _._ 6% **57** _._ **1** _±_ 1 _._ 5%

Llama 2 (70B) (baseline on uninformative input) 52 _._ 3 _±_ 2 _._ 2% 8 _._ 2 _±_ 0 _._ 7%
+ _vt_ FV **96** _._ **5** _±_ 0 _._ 5% **83** _._ **8** _±_ 0 _._ 7%

GPT-J + _vt_ FV on 34 additional tasks 80 _._ 4 _±_ 0 _._ 6% 46 _._ 1 _±_ 3 _._ 7%
Llama 2 (70B) + _vt_ FV on 34 additional tasks 93 _._ 0 _±_ 0 _._ 5% 74 _._ 2 _±_ 3 _._ 1%

3.1 PORTABILITY OF FUNCTION VECTORS

In this section, we investigate the portability of function vectors—i.e., the degree to which adding
an FV to a particular layer at the final token position of the prompt can cause the language model
to perform a task in contexts that differ from the ICL contexts from which it was extracted. For
simplicity of analysis, we only include test queries for which the LM answers correctly given a
10-shot ICL prompt; all accuracies and standard deviations over 5 random seeds are reported on this
filtered subset, and can be thought of as the proportion of model’s task performance encoded by FVs.
Results when incorrect ICL are included are similar (see Appendix D).

**Evaluating FVs at Layer** _|L|/_ 3 **.** In Table 2 we report results (averaged across the 6 tasks mentioned
above) for adding FVs to shuffled-label ICL prompts and zero-shot contexts across 3 models - GPT-J,
GPT-NeoX and Llama 2 (70B), at layers 9, 15, and 26 respectively (approximately _|L|/_ 3). For
GPT-J, we also compare the efficacy of FVs to other approaches for extracting task-inducing vectors
including simple state averaging (§2.1).

Our first observation is that the base model is substantially unable to perform the tasks in the
uninformative shuffled-label ICL and zero-shot settings; however, adding the FV allows the model
to recover task performance significantly in both cases. We also observe the proposed approach for
constructing FVs via causal mediation outperforms the layer-averaging **h** [¯] _[t]_ _ℓ_ [approach in both contexts.]

**Zero-Shot Results Across Layers.** Figure 4 shows results across layers for the zero-shot case. The
sharp reduction of causal effects in late layers suggests that FVs do not simply act linearly, but that
they trigger late-layer nonlinear computations. This pattern of causality is seen across a variety of
tasks, autoregressive model architectures, and model sizes. Even in cases where performance is low,
as in English-French with GPT-NeoX and Llama 2 (70B), adding the function vector in middle layers
still results in large _relative_ improvements to accuracy over the zero-shot baseline. Results are also
consistent across model sizes: see Appendix J for results with all sizes of Llama 2.

Country-Capital

0 5 10 15 20 25

English-French

0 5 10 15 20 25

Present-Past

0 5 10 15 20 25

Singular-Plural

0 5 10 15 20 25

Antonym

0 5 10 15 20 25

0 8 16 24 32 40

Capitalize

0 5 10 15 20 25

0 8 16 24 32 40 0 8 16 24 32 40 0 8 16 24 32 40 0 8 16 24 32 40 0 8 16 24 32 40

0 16 32 48 64
Layer

0 16 32 48 64
Layer

0 16 32 48 64
Layer

0 16 32 48 64
Layer

0 16 32 48 64
Layer

0 16 32 48 64
Layer

Figure 4: Task accuracy across tasks and models, applying FVs in zero-shot settings. We show
accuracies before adding the function vector (dotted lines) and after adding the FV to a specific layer
(solid lines). Adding the FV to early-middle layers pushes models to perform the target task without
any exemplars, as demonstrated by accuracy increases over the zero-shot without FVs.

Table 3: Natural text portability of the Antonym FV. We provide a natural template and substitute in a
query word for ‘x’. Then, we measure accuracy based on whether the correct antonym is produced in
this natural text setting within 5 generated tokens.

Prompt GPT-J +Antonym FV

The word “x”, means 1 _._ 5 _±_ 1 _._ 1% 55 _._ 2 _±_ 3 _._ 8%
When I think of the word “x”, it usually means 0 _._ 3 _±_ 0 _._ 2% 67 _._ 7 _±_ 3 _._ 0%
When I think of x, I usually 0 _._ 0 _±_ 0 _._ 0% 61 _._ 1 _±_ 2 _._ 4%
While reading a book, I came across the word “x”. I 2 _._ 7 _±_ 1 _._ 9% 46 _._ 0 _±_ 4 _._ 6%
looked it up in a dictionary and it turns out that it means

The word x can be understood as a synonym for 2 _._ 4 _±_ 1 _._ 7% 52 _._ 7 _±_ 11 _._ 0%

**FVs are Robust to Input Forms.** To check whether the FV is dependent on the ICL template that
it is extracted from, we also test the FV on 20 additional ICL templates (Appendix C) and in natural
text settings, adding the FV at layer _ℓ_ = 9 for GPT-J (approximately _|L|/_ 3).

We create 20 different ICL templates that vary the form of the ICL prompt across prefixes and
delimiters of input-output pairs. We evaluate FVs on GPT-J for these 20 templates in both shuffled-label and zero-shot settings. Across our 6 tasks, adding the FV executes the task with an average
accuracy of 76 _._ 2 _±_ 13 _._ 8% with shuffled labels and 40 _._ 0 _±_ 16 _._ 7% in the zero-shot setting, while GPT-J
only scores 32 _._ 3 _±_ 12 _._ 8% and 6 _._ 2 _±_ 4 _._ 3% on the same settings, respectively. Despite higher variance,
this performance is similar to performance in the same settings with the original template.

We also evaluate FVs on natural text completions. Given a natural text template, we insert a test
query word and have the model generate _n_ tokens. We add the FV to the final token of the original
prompt, and for all subsequent token predictions to guide its generation. We use a simple regex match
to compute whether the generation includes the correct target for the inserted query word.

Table 3 shows natural text portability results for the antonym FV for GPT-J, generating 5 new tokens.
In each of the templates, the antonym is in the FV completion significantly more than the original
completion. In fact, we find that the efficacy of the antonym FV in eliciting the correct response in
these natural text templates performs on par with the results previously reported for the zero-shot
setting. This is true for all 6 tasks (Appendix F), suggesting that the task representation transported
during ICL is similar to one that is used during autoregressive prediction in natural text settings.

We include a few qualitative results for the English-French and Country-Capital tasks (Table 4). We
see that the English-French FV will sometimes translate the whole sentence after giving the proper
completion to the original one-word translation task, indicating that it has captured more than the
original task it was shown. Additional natural text portability results are included in Appendix F.

Table 4: Qualitative examples of natural text completions for English-French, and Country-Capital

**English-French**
**Prompt:** The word “daily” means The word ‘link’ can be understood as a synonym for

**GPT-J** every day ‘connection’ or ‘relation’. The term ‘link’ is used in...

**GPT-J+English-French FV** tous les jours ‘lien’, et le mot ‘lien’ peut ˆetre compris comme un synonyme...

**Country-Capital**
**Prompt:** When you think of Netherlands,

**GPT-J** you probably think of tulips, windmills, and cheese. But the Netherlands is also home to...

**GPT-J+Country-Capital FV** you think of Amsterdam. But there are many other cities in the Netherlands. Here are some...

Table 5: A direct decoding of the function vector for each task.

**Task** _t_ **Tokens in the distribution** _D_ ( _vt_ ) **in order of decreasing probability**

Antonym ‘ lesser’, ‘ counterpart’, ‘wrong’, ‘ negate’, ‘ destroy’
Capitalize ‘ Vanilla’, ‘ Copy’, ‘ Adapter’, ‘ Actor’, ‘ Container’
Country-Capital ‘ Moscow’, ‘ Bangkok’, ‘ Paris’, ‘ London’, ‘ Madrid’
English-French ‘ ˆaK¸ [ˇ] l’, ‘ masc’, ‘ c¸¥l’, ‘ embr’, ‘ ˋe’
Present-Past ‘received’, ‘changed’, ‘killed’, ‘answered’, ‘ Changed’
Singular-Plural ‘cards’, ‘stocks’, ‘ helmets’, ‘ items’, ‘ phones’

3.2 THE DECODED VOCABULARY OF FUNCTION VECTORS

Several studies have gleaned insights about the states and parameters of transformers by viewing
them in terms of their decoded vocabulary tokens (Nostalgebraist, 2020; Geva et al., 2021; 2022;
Dar et al., 2023; Belrose et al., 2023). Therefore we ask: can we understand an FV by decoding _vt_
directly to a token probability distribution? Results are shown in Table 5, which lists the top five
tokens in the decoded distribution _D_ ( _vt_ ) for each task (additional tasks in Appendix I).

A clear pattern emerges: for most tasks, the decoded tokens lie within the task’s output space. The
Singular-Plural function vector decodes to a distribution of plural nouns, and Present-Past decodes
to past-tense verbs. However, that is not the case for all tasks: English-French decodes to nonsense
tokens, and the Antonym task decodes to words that evoke the abstract idea of reversal.

Given these meaningful decodings, we then ask whether the token vocabulary is sufficient to recreate
a working function vector. That is, we begin with the token distribution _Qt_ = _D_ ( _vt_ ), and determine
whether a function vector can be reconstructed if we know the top words in _Qt_ . Denote by _Qtk_ the
distribution that resamples _Qt_ while restricting to only the top _k_ words. We perform an optimization
to reconstruct a ˆ _vtk_ that matches the distribution _Qtk_ when decoded (where CE is cross-entropy loss):

_v_ ˆ _tk_ = arg min CE( _Qtk, D_ ( _v_ )) (6)

_v_

In Table 6, the performance of ˆ _vtk_ is evaluated when used as a function vector. We find that, while it
is possible to partially recreate the functionality of an FV, good performance typically requires more

Table 6: Performance of FV _vt_ is compared to the reconstruction _v_ ˆ _t_ 100 that matches the top 100
tokens, and ˆ _vt_ all that matches all 50k tokens in _D_ ( _vt_ ). The KL divergence between the _D_ (ˆ _vtk_ ) and
_Qtk_ are shown for each reconstruction as KL _k_ . Lowest performers for each task in red.

Accuracy on zero-shot prompts

**Task** _t_ _vt_ _v_ ˆ _t_ 100 KL100 _v_ ˆ _t_ all KLall

Antonym 48 _._ 2 _±_ 2 _._ 0% 4 _._ 8 _±_ 2 _._ 0% 0 _._ 0033 39 _._ 6 _±_ 2 _._ 6% 0 _._ 0137
Capitalize 70 _._ 5 _±_ 2 _._ 4% 5 _._ 7 _±_ 2 _._ 2% 0 _._ 0001 51 _._ 5 _±_ 11 _._ 6% 0 _._ 0053
Country-Capital 83 _._ 2 _±_ 2 _._ 7% 58 _._ 1 _±_ 18 _._ 5% 0 _._ 0002 29 _._ 0 _±_ 15 _._ 1% 0 _._ 0019
English-French 44 _._ 7 _±_ 1 _._ 2% 4 _._ 8 _±_ 1 _._ 7% 0 _._ 0 42 _._ 0 _±_ 5 _._ 6% 0 _._ 0056
Present-Past 19 _._ 7 _±_ 5 _._ 9% 4 _._ 4 _±_ 1 _._ 4% 0 _._ 0052 6 _._ 8 _±_ 2 _._ 6% 0 _._ 0139
Singular-Plural 47 _._ 0 _±_ 3 _._ 4% 23 _._ 3 _±_ 6 _._ 1% 0 _._ 0 27 _._ 4 _±_ 4 _._ 7% 0 _._ 0145 than 100 vocabulary tokens. In other words, knowledge of the top decoded tokens of _D_ ( _vt_ ) is usually
not enough on its own to construct a working function vector. That suggests that the FV contains
some needed information beyond that expressed by its top decoded tokens.

3.3 VECTOR ALGEBRA ON FUNCTION VECTORS

Figure 5: (a) A set of three list-oriented tasks that can be composed to a fourth task using FV vector algebra.
(b) The parallelogram arrangement of the fourth vector _vBD_ _[∗]_ [when it is composed out of the other three FVs.]

Although Table 6 suggests that function vectors cannot be understood as simple semantic vector
offsets on word embeddings, we can ask whether function vectors obey semantic vector algebra over
the more abstract space of _functional_ behavior by testing the composition of simple functions into
more complex ones. We begin with three conceptually decomposable ICL tasks: the list-oriented
tasks First-Copy, First-Capital, and Last-Copy, as illustrated in Figure 5a. Using ICL, we collect FVs
for all three tasks and denote them _vAC_, _vBC_, and _vAD_ .

Then we form a simple algebraic sum to create a new vector that we will denote _vBD_ _[∗]_ [.]

_vBD_ _[∗]_ [=] _[ v][AD]_ [+] _[ v][BC]_ _[−]_ _[v][AC]_ (7)

Last-Capital _[∗]_ = Last-Copy + First-Capital _−_ First-Copy (8)

In principle we could expect _vBD_ _[∗]_ [to serve as a new function vector for a new composed task (Last-]

Capital). We perform several similar task compositions on a variety of tasks. In each case, we
combine a task with First-Copy and Last-Copy to produce a composed Last- _∗_ vector; then, we test
the accuracy of _vBD_ _[∗]_ [as a function vector.] [We compare to the accuracy of the FV extracted from ICL,]

as well as accuracy of the same model performing the task using ICL. Results for GPT-J are reported
in Table 7; see Appendix K for results for Llama 2 (13 and 70 billion parameter models). We find that
some FVs can be composed, with algebraic compositions outperforming FVs and even ICL on some
tasks. Other tasks, including some for which ICL and FVs perform well, resist vector composition.

The ability to compose the tasks that we have demonstrated may hinge on the fact that “word-selection”
from context and “word-transformation” are different components of language tasks that could involve
FVs triggering complementary underlying mechanisms (e.g., one for locating and extracting input
and another for transforming it). We therefore believe that FV composition may be a useful tool for
further understanding the mechanisms of LMs.

Table 7: The accuracy of ICL, calculated FV _vBD_ zero-shot interventions, and vector-composed _v_ _[∗]_

Table 7: The accuracy of ICL, calculated FV _vBD_ zero-shot interventions, and vector-composed _vBD_ [zero-shot]

interventions when performing several list-oriented tasks. Unlike our previous evaluations, here we measure
performance on _all_ available samples of the task, without restriction to the subset where the LM predicts correct
output. In a few cases, composed function vector intervention _vBD_ _[∗]_ [can perform a task better than ICL.]

_BD_ [can perform a task better than ICL.]

Task ICL (ten-shot) _vBD_ (FV on zero-shot) _vBD_ _[∗]_ [(sum on zero-shot)]

Last-Antonym 0 _._ 25 _±_ 0 _._ 02 0 _._ 02 _±_ 0 _._ 01 0 _._ 07 _±_ 0 _._ 02
Last-Capitalize 0 _._ 91 _±_ 0 _._ 02 0 _._ 64 _±_ 0 _._ 03 0 _._ 76 _±_ 0 _._ 04
Last-Country-Capital 0 _._ 32 _±_ 0 _._ 02 0 _._ 15 _±_ 0 _._ 03 **0** _._ **60** _±_ 0 _._ 02
Last-English-French 0 _._ 45 _±_ 0 _._ 04 0 _._ 16 _±_ 0 _._ 02 0 _._ 06 _±_ 0 _._ 02
Last-Present-Past 0 _._ 89 _±_ 0 _._ 02 0 _._ 18 _±_ 0 _._ 02 0 _._ 29 _±_ 0 _._ 03
Last-Singular-Plural 0 _._ 90 _±_ 0 _._ 01 0 _._ 28 _±_ 0 _._ 01 0 _._ 29 _±_ 0 _._ 02
Last-Capitalize-First-Letter 0 _._ 75 _±_ 0 _._ 01 0 _._ 76 _±_ 0 _._ 02 **0** _._ **95** _±_ 0 _._ 00
Last-Product-Company 0 _._ 35 _±_ 0 _._ 03 0 _._ 30 _±_ 0 _._ 02 **0** _._ **41** _±_ 0 _._ 03

4 RELATED WORK

A cousin to function vectors has been independently observed in concurrent work by Hendel et al.
(2023); they study causal effects of **h** _[t]_ _ℓ_ [(similar to Section 2.1) on a different set of models and tasks.]

**Task Representations.** Our work shows that it is possible to extract FVs with strong causal effects
from LLMs; this is an advance over previous examinations that have _added_ task representations to
LLMs, e.g. Lampinen & McClelland (2020); Shao et al. (2023); Mu et al. (2023); Panigrahi et al.
(2023); Ilharco et al. (2023), who devised ways to create compositional task encodings for LLMs
using metamappings, codebooks, soft-prompts or sets of model parameter perturbations that Ilharco
et al. calls _task vectors_ . Unlike these previous works that create function representations, we find that
compact FVs _already exist_ within LLMs and show how to extract them. Likewise Lake & Baroni
(2018); Hill et al. (2018) show RNN hidden states cluster on similar tasks. Our work differs because
FVs are causal, not just correlative, so they can be explicitly extracted and inserted.

**In-Context Learning.** Since its observation in LLMs by Brown et al. (2020), ICL has been studied
intensively from many perspectives. The role of ICL prompt forms has been studied by Reynolds &
McDonell (2021); Min et al. (2022); Yoo et al. (2022). Models of inference-time meta-learning that
could explain ICL have been proposed by Akyurek et al. (2022); Dai et al. (2023); Von Oswald et al.¨
(2023); Li et al. (2023b); Garg et al. (2022). Analyses of ICL as Bayesian task inference have been
performed by Xie et al. (2021); Wang et al. (2023c); Wies et al. (2023); Hahn & Goyal (2023); Zhang
et al. (2023); Han et al. (2023). And ICL robustness under scaling has been studied by Wei et al.
(2023); Wang et al. (2023b); Pan et al. (2023). Our work differs from those studies of the externally
observable behavior of ICL by instead focusing on mechanisms _within_ transformers.

**Mechanisms of task performance in LMs.** Our work is related to Merullo et al. (2023); Halawi
et al. (2023) which analyze components during execution of ICL tasks and identify causes of false
statements. Also related are several methods that modify activations at inference time to steer LM
behavior (Li et al., 2023a; Hernandez et al., 2023a; Subramani et al., 2022; Turner et al., 2023; Rimsky
et al., 2023; Liu et al., 2023; Zou et al., 2023). Our work is consistent with Wang et al. (2023a) which
observes salience of label tokens during ICL, Wang et al. (2022b) which observes individual neurons
that correlate with specific task performance, and Variengien & Winsor (2023) shows task requests
are processed in middle layers. We measure causal mediators across a distribution of different tasks
to find a generic function-invocation mechanism that _identifies and distinguishes_ between tasks.

**Mechanistic Interpretability.** We also build upon the analyses of Elhage et al. (2021) and Olsson
et al. (2022), who observed prevalent in-context copying behavior related to jumps in performance
during training. We isolate FVs using causal mediation analysis methods developed in Pearl (2001);
Vig et al. (2020); Meng et al. (2022); Wang et al. (2022a); Geva et al. (2023). Our examination of
FVs in vocabulary uses the logit lens of Nostalgebraist (2020); Geva et al. (2021); Dar et al. (2023).

**Analyzing** **the** **Attention** **Mechanism.** Our work is related to previous attention-weight analyses (Voita et al., 2018; Clark et al., 2019; Voita et al., 2019; Kovaleva et al., 2019; Reif et al., 2019;
Lin et al., 2019; Htut et al., 2019; Kobayashi et al., 2020), that have found attention weights that
align with linguistic structures. Our work is motivated by the observation that attention weights
alone do not fully explain model outputs (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019; Bibal
et al., 2022). The focus of our paper is to extend our understanding of attention by investigating the
_content_ of the information transported by the attention heads in ICL to open a new window into the
human-interpretable role that attention plays in language processing.

5 DISCUSSION

Function vectors are a surprising finding. The meta-learning capabilities of LLMs that have been
studied since Brown et al. (2020) seem complex enough be inscrutable. Yet in this paper we have
found a simple mechanism in a range of transformer LLMs that is common across tasks and robust
to shifts in context: _function vectors_ (FVs) that represent the task within a hidden state. FVs can be
explicitly extracted from a small fixed set of attention heads that can be easily identified, and these
FVs represent a range of tasks just as simply as word vectors (Mikolov et al., 2013)—yet our findings
also reveal FVs must be a distinct phenomenon (Appendix A). Although FVs are not yet a complete
accounting of how ICL works, they do provide new clarity on one level of mediation within ICL, and
they open up a new path for future research to fully characterize function execution within LLMs.

ETHICS

While our work clarifying the mechanisms of function representation and execution within large
models is intended to help make large language models more transparent and easier to audit, understand, and control, we caution that such transparency may also enable bad actors to abuse large neural
language systems, for example by injecting or amplifying functions that cause undesirable behavior.

ACKNOWLEDGMENTS

Special thanks to Evan Hernandez whose valuable advice and mentorship made this research possible.
We are grateful for the generous support of Open Philanthropy (ET, AS, AM, DB) as well as National
Science Foundation (NSF) grant 1901117 (ET, ML, BW). ML is supported by an NSF Graduate
Research Fellowship, and AM is recipient of the Zuckerman Postdoctoral Fellowship. We thank the
Center for AI Safety (CAIS) for making computing resources available for this research.

REFERENCES

Ekin Akyurek,¨ Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In _The Eleventh International_
_Conference on Learning Representations_, 2022.

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella

Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens.
_arXiv preprint arXiv:2303.08112_, 2023.

Adrien Bibal, Remi Cardon, David Alfter, Rodrigo Wilkens, Xiaoou Wang, Thomas Fran´ c¸ois, and

Patrick Watrin. Is attention explanation? an introduction to the debate. In _Proceedings of the 60th_
_Annual Meeting of the Association for Computational Linguistics (Volume 1:_ _Long Papers)_, pp.

3889–3900, 2022.

Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He,

Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu
Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. GPT-NeoX-20B: An
open-source autoregressive language model. In _Proceedings of the ACL Workshop on Challenges_
_& Perspectives in Creating Large Language Models_, 2022. [URL https://arxiv.org/abs/2204.](https://arxiv.org/abs/2204.06745)
[06745.](https://arxiv.org/abs/2204.06745)

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,

Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler,
Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray,
Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever,
and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato,
R. Hadsell, M.F. Balcan, and H. Lin (eds.), _Advances in Neural Information Processing Systems_,
[volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
cc/paper_f[iles/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

Alonzo Church. An unsolvable problem of elementary number theory. _American journal of mathe-_

_matics_, 58:345–373, 1936.

Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D Manning. What does BERT look at? an analysis of BERT’s attention. In _Proceedings of the 2019 ACL Workshop BlackboxNLP:_
_Analyzing and Interpreting Neural Networks for NLP_, pp. 276–286, 2019.

Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Herve J´ egou.´

Word translation without parallel data. _arXiv preprint arXiv:1710.04087_, 2017.

Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can

GPT learn in-context? language models secretly perform gradient descent as meta-optimizers. In
_Findings of the Association for Computational Linguistics:_ _ACL 2023_, pp. 4005–4019, Toronto,

Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.
247. [URL https://aclanthology.org/2023.findings-acl.247.](https://aclanthology.org/2023.findings-acl.247)

Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. Analyzing transformers in embedding space. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings_ _of_ _the_
_61st Annual Meeting of the Association for Computational Linguistics (Volume 1:_ _Long Papers)_,
pp. 16124–16170, Toronto, Canada, July 2023. Association for Computational Linguistics. doi:
10.18653/v1/2023.acl-long.893. [URL https://aclanthology.org/2023.acl-long.893.](https://aclanthology.org/2023.acl-long.893)

N Elhage, N Nanda, C Olsson, T Henighan, N Joseph, B Mann, A Askell, Y Bai, A Chen, T Conerly, et al. A mathematical framework for transformer circuits. _Transformer Circuits Thread_, 2021.

Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing_
_Systems_, 35:30583–30598, 2022.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), _Proceedings of the 2021 Conference on Empirical Methods in Natural Language_
_Processing_, pp. 5484–5495, Online and Punta Cana, Dominican Republic, November 2021.
Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.446. [URL https:](https://aclanthology.org/2021.emnlp-main.446)
[//aclanthology.org/2021.emnlp-main.446.](https://aclanthology.org/2021.emnlp-main.446)

Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Yoav Goldberg, Zornitsa
Kozareva, and Yue Zhang (eds.), _Proceedings of the 2022 Conference on Empirical Methods in_
_Natural Language Processing_, pp. 30–45, Abu Dhabi, United Arab Emirates, December 2022.
Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.3. [URL https:](https://aclanthology.org/2022.emnlp-main.3)
[//aclanthology.org/2022.emnlp-main.3.](https://aclanthology.org/2022.emnlp-main.3)

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. _arXiv preprint arXiv:2304.14767_, 2023.

Michael Hahn and Navin Goyal. A theory of emergent in-context learning as implicit structure induction. _arXiv preprint arXiv:2303.07971_, 2023.

Danny Halawi, Jean-Stanislas Denain, and Jacob Steinhardt. Overthinking the truth: Understanding how language models process false demonstrations. _arXiv preprint arXiv:2307.09476_, 2023.

Chi Han, Ziqi Wang, Han Zhao, and Heng Ji. In-context learning of large language models explained as kernel regression. _arXiv preprint arXiv:2305.12766_, 2023.

Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. In _Findings_

_of the Association for Computational Linguistics:_ _EMNLP 2023_, 2023.

Evan Hernandez, Belinda Z. Li, and Jacob Andreas. Inspecting and editing knowledge representations in language models. In _Arxiv_, 2023a. [URL https://arxiv.org/abs/2304.00740.](https://arxiv.org/abs/2304.00740)

Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas,

Yonatan Belinkov, and David Bau. Linearity of relation decoding in transformer language models.

_arXiv preprint arXiv:2308.09124_, 2023b.

Felix Hill, Adam Santoro, David Barrett, Ari Morcos, and Timothy Lillicrap. Learning to make analogies by contrasting abstract relational structure. In _International Conference on Learning_
_Representations_, 2018.

Or Honovich, Uri Shaham, Samuel R. Bowman, and Omer Levy. Instruction induction: From few examples to natural language task descriptions. In Anna Rogers, Jordan Boyd-Graber, and Naoaki
Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational_
_Linguistics (Volume 1:_ _Long Papers)_, pp. 1935–1952, Toronto, Canada, July 2023. Association for
Computational Linguistics. doi: 10.18653/v1/2023.acl-long.108. [URL https://aclanthology.](https://aclanthology.org/2023.acl-long.108)
[org/2023.acl-long.108.](https://aclanthology.org/2023.acl-long.108)

Phu Mon Htut, Jason Phang, Shikha Bordia, and Samuel R Bowman. Do attention heads in BERT track syntactic dependencies? _arXiv preprint arXiv:1911.12246_, 2019.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In _The Eleventh International Conference_
_on Learning Representations_, 2023.

Sarthak Jain and Byron C Wallace. Attention is not explanation. In _Proceedings_ _of_ _the_ _2019_

_Conference_ _of_ _the_ _North_ _American_ _Chapter_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_
_Human Language Technologies, Volume 1 (Long and Short Papers)_, pp. 3543–3556, 2019.

Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. Attention is not only a weight:

Analyzing transformers with vector norms. In _Proceedings of the 2020 Conference on Empirical_
_Methods in Natural Language Processing (EMNLP)_, pp. 7057–7075, 2020.

Olga Kovaleva, Alexey Romanov, Anna Rogers, and Anna Rumshisky. Revealing the dark secrets of BERT. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language_
_Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-_
_IJCNLP)_, pp. 4365–4374, 2019.

Brenden Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In Jennifer Dy and Andreas Krause (eds.), _Proceedings_
_of the 35th International Conference on Machine Learning_, volume 80 of _Proceedings of Machine_
_Learning Research_, pp. 2873–2882. PMLR, 10–15 Jul 2018. [URL https://proceedings.mlr.](https://proceedings.mlr.press/v80/lake18a.html)
[press/v80/lake18a.html.](https://proceedings.mlr.press/v80/lake18a.html)

Andrew K Lampinen and James L McClelland. Transforming task representations to perform novel tasks. _Proceedings of the National Academy of Sciences_, 117(52):32970–32981, 2020.

Omer Levy and Yoav Goldberg. Dependency-based word embeddings. In _Proceedings of the 52nd_

_Annual Meeting of the Association for Computational Linguistics (Volume 2:_ _Short Papers)_, pp.

302–308, 2014.

Kenneth Li, Oam Patel, Fernanda Viegas, Hanspeter Pfister, and Martin Wattenberg.´ Inference-time intervention: Eliciting truthful answers from a language model. In _Thirty-seventh_ _Conference_
_on Neural Information Processing Systems_, 2023a. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=aLLuYpn83y)
[aLLuYpn83y.](https://openreview.net/forum?id=aLLuYpn83y)

Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. Transform
ers as algorithms: Generalization and stability in in-context learning. _Proceedings of the 40th_
_International Conference on Machine Learning_, 2023b.

Yongjie Lin, Yi Chern Tan, and Robert Frank. Open sesame: Getting inside BERT’s linguistic knowledge. In _Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting_
_Neural Networks for NLP_, pp. 241–253, 2019.

Sheng Liu, Lei Xing, and James Zou. In-context vectors: Making in context learning more effective and controllable through latent space steering. _arXiv preprint arXiv:2311.06668_, 2023.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. _Advances in Neural Information Processing Systems_, 35:17359–17372, 2022.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass editing memory in a transformer. _The Eleventh International Conference on Learning Representations_
_(ICLR)_, 2023.

Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. Language models implement simple word2vec
style vector arithmetic. _arXiv preprint arXiv:2305.16130_, 2023.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. _Advances_ _in_ _neural_ _information_ _processing_
_systems_, 26, 2013.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke

Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In
_Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_, pp.
11048–11064, 2022.

Jesse Mu, Xiang Lisa Li, and Noah Goodman. Learning to compress prompts with gist tokens. _arXiv_

_preprint arXiv:2304.08467_, 2023.

Kim Anh Nguyen, Sabine Schulte im Walde, and Ngoc Thang Vu. Distinguishing antonyms and synonyms in a pattern-based neural network. In _Proceedings_ _of_ _the_ _15th_ _Conference_ _of_ _the_
_European Chapter of the Association for Computational Linguistics:_ _Volume 1, Long Papers_, pp.
76–85, Valencia, Spain, April 2017. Association for Computational Linguistics. URL [https:](https://aclanthology.org/E17-1008)
[//aclanthology.org/E17-1008.](https://aclanthology.org/E17-1008)

Nostalgebraist. Interpreting GPT: The logit lens. URL [https://www.lesswrong.com/posts/](https://www.lesswrong.com/ posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

[AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens, 2020.](https://www.lesswrong.com/ posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan,

Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads.
_arXiv preprint arXiv:2209.11895_, 2022.

OpenAI. GPT-4 technical report, 2023.

Jane Pan, Tianyu Gao, Howard Chen, and Danqi Chen. What in-context learning” learns” in-context:

Disentangling task recognition and task learning. _arXiv preprint arXiv:2305.09731_, 2023.

Abhishek Panigrahi, Nikunj Saunshi, Haoyu Zhao, and Sanjeev Arora. Task-specific skill localization in fine-tuned language models. _arXiv preprint arXiv:2302.06600_, 2023.

Judea Pearl. Direct and indirect effects. In _Proceedings of the Seventeenth Conference on Uncertainty_

_and Artificial Intelligence, 2001_, pp. 411–420. Morgan Kaufman, 2001.

Emily Reif, Ann Yuan, Martin Wattenberg, Fernanda B Viegas, Andy Coenen, Adam Pearce, and

Been Kim. Visualizing and measuring the geometry of BERT. _Advances in Neural Information_
_Processing Systems_, 32, 2019.

Laria Reynolds and Kyle McDonell. Prompt programming for large language models: Beyond the few-shot paradigm. In _Extended_ _Abstracts_ _of_ _the_ _2021_ _CHI_ _Conference_ _on_ _Human_ _Factors_ _in_
_Computing Systems_, pp. 1–7, 2021.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner.

Steering Llama 2 via contrastive activation addition. _arXiv preprint arXiv:2312.06681_, 2023.

Erik Tjong Kim Sang and Fien De Meulder. Introduction to the CoNLL-2003 shared task: Language
independent named entity recognition. In _Proceedings_ _of_ _the_ _Seventh_ _Conference_ _on_ _Natural_
_Language Learning at HLT-NAACL 2003_, pp. 142–147, 2003.

Nan Shao, Zefan Cai, Chonghua Liao, Yanan Zheng, Zhilin Yang, et al. Compositional task
representations for large language models. In _The Eleventh International Conference on Learning_
_Representations_, 2023.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and

Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank.
In _Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing_, pp.
1631–1642, Seattle, Washington, USA, October 2013. Association for Computational Linguistics.
[URL https://www.aclweb.org/anthology/D13-1170.](https://www.aclweb.org/anthology/D13-1170)

Nishant Subramani, Nivedita Suresh, and Matthew E Peters. Extracting latent steering vectors from pretrained language models. In _Findings of the Association for Computational Linguistics:_ _ACL_
_2022_, pp. 566–581, 2022.

Gerald Jay Sussman. Scheme: an interpreter for extended lambda calculus. _MIT AI Memo_, 1975.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In _Proceedings of the 2019 Conference of_
_the North American Chapter of the Association for Computational Linguistics:_ _Human Language_
_Technologies, Volume 1 (Long and Short Papers)_, pp. 4149–4158, 2019.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay

Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation
and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023.

Alex Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. Acti
vation addition: Steering language models without optimization. _arXiv preprint arXiv:2308.10248_,
2023.

Alexandre Variengien and Eric Winsor. Look before you leap: A universal emergent decomposition of retrieval tasks in language models. _arXiv preprint arXiv:2312.10091_, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz

Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing_
_systems_, 30, 2017.

Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and

Stuart Shieber. Investigating gender bias in language models using causal mediation analysis.
_Advances in neural information processing systems_, 33:12388–12401, 2020.

Elena Voita, Pavel Serdyukov, Rico Sennrich, and Ivan Titov. Context-aware neural machine
translation learns anaphora resolution. In _Proceedings of the 56th Annual Meeting of the Association_
_for Computational Linguistics (Volume 1:_ _Long Papers)_, pp. 1264–1274, 2018.

Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In _Proceedings of the_
_57th Annual Meeting of the Association for Computational Linguistics_, pp. 5797–5808, 2019.

Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, Joao Sacramento, Alexander Mordvintsev,˜

Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In
_International Conference on Machine Learning_, pp. 35151–35174. PMLR, 2023.

Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model.

[https://github.com/kingoflolz/mesh-transformer-jax, May 2021.](https://github.com/kingoflolz/mesh-transformer-jax)

Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt.

Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In _The_
_Eleventh International Conference on Learning Representations_, 2022a.

Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Label words are anchors: An information flow perspective for understanding in-context learning. _arXiv_
_preprint arXiv:2305.14160_, 2023a.

Xiaozhi Wang, Kaiyue Wen, Zhengyan Zhang, Lei Hou, Zhiyuan Liu, and Juanzi Li. Finding skill neurons in pre-trained transformer-based language models. In _Proceedings of the 2022 Conference_
_on Empirical Methods in Natural Language Processing_, pp. 11132–11152, 2022b.

Xindi Wang, Yufei Wang, Can Xu, Xiubo Geng, Bowen Zhang, Chongyang Tao, Frank Rudzicz,

Robert E Mercer, and Daxin Jiang. Investigating the learning behaviour of in-context learning: A
comparison with supervised learning. _arXiv preprint arXiv:2307.15411_, 2023b.

Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, and William Yang Wang. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context
learning. In _Workshop on Efficient Systems for Foundation Models at ICML2023_, 2023c. URL
[https://openreview.net/forum?id=HCkI1b6ksc.](https://openreview.net/forum?id=HCkI1b6ksc)

Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu,

Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. _arXiv_
_preprint arXiv:2303.03846_, 2023.

Sarah Wiegreffe and Yuval Pinter. Attention is not not explanation. In _Proceedings_ _of_ _the_ _2019_

_Conference on Empirical Methods in Natural Language Processing and the 9th International Joint_
_Conference on Natural Language Processing (EMNLP-IJCNLP)_, pp. 11–20, 2019.

Noam Wies, Yoav Levine, and Amnon Shashua. The learnability of in-context learning. _arXiv_

_preprint arXiv:2303.07895_, 2023.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi,

Pierric Cistac, Tim Rault, Remi´ Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick
von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger,
Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural
language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural_
_Language Processing:_ _System Demonstrations_, pp. 38–45, Online, October 2020. Association for
Computational Linguistics. URL [https://www.aclweb.org/anthology/2020.emnlp-demos.](https://www.aclweb.org/anthology/2020.emnlp-demos.6)
[6.](https://www.aclweb.org/anthology/2020.emnlp-demos.6)

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In _International Conference on Learning Representations_,
2021.

Kang Min Yoo, Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang
goo Lee, and Taeuk Kim. Ground-truth labels matter: A deeper look into input-label demonstrations.
In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_,
pp. 2422–2437, 2022.

Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text
classification. _Advances in neural information processing systems_, 28, 2015.

Yufeng Zhang, Fengzhuo Zhang, Zhuoran Yang, and Zhaoran Wang. What and how does in-context learning learn? bayesian model averaging, parameterization, and generalization. _arXiv preprint_
_arXiv:2305.19420_, 2023.

Sumu Zhao, Damian Pascual, Gino Brunner, and Roger Wattenhofer.´ Of non-linearity and commuta
tivity in bert. In _2021 International Joint Conference on Neural Networks (IJCNN)_, pp. 1–8. IEEE,
2021.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan,

Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A
top-down approach to ai transparency. _arXiv preprint arXiv:2310.01405_, 2023.

A DISCUSSION: FUNCTION VECTORS VS SEMANTIC VECTOR ARITHMETIC

In this appendix we discuss the experimental support for our characterization of function vectors in
more detail, in particular the assertion that function vectors are acting in a way that is distinct from
semantic vector arithmetic on word embeddings.

The use of vector addition to induce a mapping is familiar within semantic embedding spaces: the
vector algebra of _semantic vector offsets_ has been observed in many settings; for example, word
embedding vector arithmetic was clearly described by Mikolov et al. (2013), and has been observed
in other neural word representations including transformers (recently, Merullo et al., 2023). In recent
examinations of internal transformer states, Geva et al. (2022) and Dar et al. (2023) have suggested
that many internal transformer calculations can be understood in terms of such word vector arithmetic.
Therefore one of our main underlying research questions is whether our function vectors should be
described as triggers for a nontrivial function, or whether, more simply (as would be suggested by
Dar et al.), they could be thought of as just ordinary semantic vector offsets that induce a trivial
mapping between related words by adding an offset to a rich word embedding.

Our investigation of the possibility of a distinct and localized representation for tasks is similar to
the studies of task encodings in RNNs by Lake & Baroni (2018) and Hill et al. (2018), as well as
the studies of linguistic attribute encodings by Clark et al. (2019), but our focus on causal effects
rather than correlations allows us to explicitly extract and test the computational roles of vector
representations of functions, which leads to several new lines of evidence.

The main paper contains three pieces of experimental evidence that support the conclusion that
function vectors are different from semantic vector offsets of word embeddings, and that they trigger
nontrivial functions:

1. Function vectors can implement complex mappings, including cyclic mappings such as
_antonyms_ that cannot be semantic vector offsets.

2. Function vector causal effects cannot be recovered from the target output vocabulary alone;
they carry some other information.

3. Function vector activity is mediated by mid-layer nonlinearities (i.e., they trigger nonlinear
computations), since they have near-zero causal effect at late layers.

We discuss each of these lines of evidence in more detail here.

**Cyclic** **Tasks** **Cannot** **be** **Semantic** **Vector** **Offsets.** The first task analyzed in the paper is the
_antonym_ task. Because the antonym task is cyclic, it is a simple counterexample to the possibility
that function vectors are just semantic vector offsets of language model word embeddings.

Suppose there were an ‘antonym’ vector offset _va_ such that adding it to a word embedding _w_ produces
_w_ _[′]_, the embedding of the antonym of _w_ (i.e. _w_ + _va_ = antonym( _w_ ) = _w_ _[′]_ ). An example of this
might be vec(big) + _va_ = vec(small). But then: if _w_ and _w_ _[′]_ are antonyms of each other, the
relationship holds both ways. That means that the antonym offset vector _va_ would have to satisfy
both _w_ + _va_ = _w_ _[′]_ and _w_ _[′]_ + _va_ = _w_, which can only happen if _va_ = 0, implying _w_ = _w_ _[′]_ and
creating a contradiction to the assumption that _w_ and _w_ _[′]_ are antonyms, distinct from each other. Thus
there could be no constant vector offset that would properly model the antonym task. The same
argument excludes semantic vector offsets for any cyclic function. We evaluate additional cyclic
tasks in Appendix L.

Since we are able to find a constant antonym function vector _vt_ that, when added to the transformer,
does cause cyclic behavior, we conclude that the action of _vt_ is a new phenomenon. Function vectors
act in a way different from simple semantic vector offsets.

**Function** **Vectors** **Contain** **Information** **Beyond** **Output** **Vocabulary.** Not every function is
cyclic, but the vector offset hypothesis can be tested by examining word embeddings. Following the
reasoning of Geva et al. (2022); Dar et al. (2023), one way to potentially implement a semantic vector
offset is to promote a certain subset of tokens that correspond to a particular semantic concept (i.e.
capital cities, past-tense verbs, etc.). Function vectors do show some evidence of acting in this way:
when decoding function vectors directly to the model’s vocabulary space we often see that the tokens with the highest probabilities are words that are part of its task’s output vocabulary (see Table 5 and
Table 19).

To determine whether it is the vocabulary itself that contributes to the function vector’s performance,
in Section 3.2 we construct another vector (via optimization) that decodes to the same vocabulary
distribution and measure its performance when used as a “function vector”. In that experiment we
create reconstructions ˆ _vt_ 100 and ˆ _vt_ all that encode the same decoder vocabulary as _vt_ (the near-zero
KL divergences show that the reconstructions of the decoder vocabulary are near-perfect). Yet the
performance of the reconstructions when used as FV is poor when the top 100 words are matched,
and still lower than _vt_ even when the distribution over the full vocabulary is reconstructed.

That experiment reveals that while function vectors do often encode words contained in the output
space of the task they are extracted from, simply adding a vector that boosts those same words by the
same amounts is not enough to recover its full performance—though if enough words are included, in
some cases a fraction of performance is recovered. Our measurements suggest that while _part_ of the
role of a function vector may act similarly to a semantic vector offset, the ability for function vectors
to produce nontrivial task behavior arises from other essential information in the vector beyond just a
simple word embedding vocabulary-based offset. See Appendix N for a related experiment.

**Function Vectors’ Causal Effects are Near-Zero at Late Layers.** Across the set of tasks and
models we evaluate, there is a common pattern of causal effects that arises when adding a function
vector to different layers. The highest causal effects are achieved when adding the function vector at
early and middle layers of the network, with a sharp drop in performance to near-zero at the later
layers of the network (Figure 4, Figure 26, Figure 14, Figure 16). Interestingly, FV causal effects
are strongest in largest models, yet the cliff to near-zero causal effects is also sharpest for the largest
models (Figure 16).

If the action were linear and created by a word embedding vector offset, then the residual stream
would transport the action equally well at any layer including the last layers. Thus the pattern of
near-zero causal effects at later layers suggests that the action of the function vector is not acting
directly on the word embedding, but rather that it is mediated by some _nonlinear_ computations in
the middle layers of the network that are essential to the performance of the task. This mediation is
evidence that the function vector activates mid-layer components that execute the task, rather than
fully executing the task itself.

This pattern is in contrast to the vector arithmetic described in Merullo et al. (2023), that are most
effective at later layers of the network and have little to no causal effect at early layers of the network;
those offsets more closely resemble semantic vector offsets of word embeddings.

**Function Vectors Therefore Represent Functions** In summary, the three lines of evidence lead
us conclude that the vectors _vt_ should not be seen as simple word embeddings, nor trivial offsets or
differences of embeddings, nor simple averages of word embeddings over vocabularies of words to
be boosted. These characteristics distinguish function vectors from from many linguistic concepts
that can be viewed as probability distributions over words. Rather, the evidence suggests that the
vectors we have identified act in a way that is distinct from literal token embedding offsets. Instead
they directly represent and trigger nonlinear execution of abstract functions.

This finding is surprising since the transformer is trained on a word-prediction task; it would be
expected that they should learn representations that can be expressed in terms of adjustments to
word probabilities as observed by Geva et al. (2022); Dar et al. (2023). Our evidence indicates a
different kind of representation: we find that transformers learn a compact, concrete, and causal
vector representation of higher-level functional concepts that cannot be reduced to a probability vector
over a set of words.

Thus we come to the conclusion that the vectors _vt_ are references to functions, and we call them
_function vectors_ .

B ATTENTION OUTPUTS ARE INDEPENDENT AND ADDITIVE

In this section we define our formulation of attention notation _aℓj_ in detail, relating our notation to
the original formulation of Vaswani et al. (2017) via the framework of Elhage et al. (2021).

B.1 EXPRESSING ATTENTION _aℓj_ IN THE RESIDUAL STREAM SPACE

The transformer architecture as introduced by Vaswani et al. (2017) has a multi-head attention
mechanism. They describe it in terms of a concatenation procedure for performing the attention
function for several “heads” head _j_ _∈_ R _[d][v]_ ( _j_ _≤_ _J_ ) at each layer, all in parallel. Equation 9 reproduces
the relevant equation from page 5 of Vaswani et al.:

Multi-Head( _Q, K, V_ ) = Concat(head1 _, ...,_ head _h_ ) _W_ _[O]_ (9)

where head _j_ = Attention( _QWj_ _[Q]_

_[ K]_ _j_ _[, V W][ V]_ _j_

_[ V]_ _j_ [)] (10)

_j_ _[Q][, KW][ K]_ _j_

Note that a transformer repeats this process with different weights and data at each layer; we add the
layer subscript _ℓ_ to disambiguate their notation. In the Vaswani et al. formulation, each head at a
layer _ℓ_ resides in a low dimensional space R _[d][v]_ with dimension _dv_ _<_ _d_ that differs from the main
hidden state _residual stream_ of the transformer, which we write as **h** _ℓ_ _∈_ R _[d]_ . All the heads at the layer
are concatenated and then transformed through transformation _Wℓ_ _[O]_ [to produce the full][ Multi-Head] _[ℓ]_

attention output in R _[d]_ .

Elhage et al. (2021) observes that the Vaswani et al. formulation is equivalent to dividing the matrix
_Wℓ_ _[O]_ [into block form][ [] _[W][ O]_ _ℓ_ 1 _[W][ O]_ _ℓ_ 2 _[. . .]_ _[W]_ _ℓJ_ _[ O]_ []][, and then projecting each][ head] _[ℓj]_ [into the residual stream]

space directly. In our formulation of _aℓj_ in Section 2.2, we adopt this view. The attention head output
_aℓj_ can be defined in terms of the notation of Vaswani et al. and Elhage et al. as follows:

_ℓ_ _[O]_ [into block form][ [] _[W][ O]_ _ℓ_ 1

_[ O]_ _ℓ_ 2 _[. . .]_ _[W]_ _ℓJ_ _[ O]_

_[ O]_ _ℓ_ 1 _[W][ O]_ _ℓ_ 2

_aℓj_ = head _ℓjWℓj_ _[O]_ _[∈]_ [R] _[d]_ (11)

In this way the total attention contribution at layer _ℓ_ is the sum of the attention output of each
individual head, and these all reside directly in the R _[d]_ space of the residual stream:

Multi-Head _ℓ_ ( _Qℓ, Kℓ, Vℓ_ ) =

_j≤J_

_aℓj_ _∈_ R _[d]_ (12)

While the left and right-hand sides of equation 12 are computationally equivalent definitions of
attention, the “independently additive” form of attention on the right-hand side allows us to see the
contributions of individual attention heads more clearly.

B.2 ADDING FUNCTION VECTORS TO A LAYER

Using this formulation of attention _aℓj_, we return to our notation as defined in section 2.2 to understand
what we mean when we say we add a function vector to a layer _ℓ_ .

We focus on the hidden state residual stream at the final token of a given prompt. Recall that the
ordinary operation of a transformer defines the hidden state

**h** _ℓ_ = **h** _ℓ−_ 1 + _mℓ_ +

_j≤J_

_aℓj_ (13)

This recursive residual structure forms a telescoping sum that creates a common vector space for
nearby layers; for example Zhao et al. (2021) has observed that adjacent layers of a transformer can
be swapped with little change. Thus it is meaningful to collect, average, and sum attention head
outputs _aℓj_ _∈_ R _[d]_ among nearby layers, and that observation inspires our definition of a function
vector _vt_ as average attention head values for a selection of relevant heads _a_ ¯ _ℓj_ (equation 5). See
Appendix M for an analysis of an alternative formulation that does not swap layers.

Since the function vector also resides in the residual stream space of **h** _ℓ_ _∈_ R _[d]_, when we add a function
vector _vt_ to the hidden state of layer _ℓ_, we can therefore add it to the hidden state just as attention is
added; we write the updated hidden state **h** _[′]_ _ℓ_ [as]

**h** _[′]_ _ℓ_ [=] **[ h]** _[ℓ][−]_ [1] [+] _[ m][ℓ]_ [+]

_j≤J_

_aℓj_ + _vt_ (14)

This could also be written as simply **h** _[′]_ _ℓ_ [=] **[ h]** _[ℓ]_ [+] _[ v][t]_ [.]

C EXPERIMENTAL DETAILS

In this section, we provide details of the function vector extraction process (section 2.3), and the
evaluation of function vectors (section 3).

**Function Vector Extraction.** We compute a function vector as the sum over the average output
of several attention heads, where the average is conditioned on prompts taken from a particular
task. We write this as _vt_ = [] _aℓj_ _∈A_ _[a]_ [¯] _ℓj_ _[t]_ [.] [How many attention heads should we use?] [To extract a]

_aℓj_ _∈A_ _[a]_ [¯] _[t]_

task. We write this as _vt_ = [] _aℓj_ _∈A_ _[a]_ [¯] _ℓj_ [.] [How many attention heads should we use?] [To extract a]

function vector (FV), we first compute the task-conditioned mean activation of each head ¯ _a_ _[t]_ _ℓj_ [, using]

_|Pt|_ = 100 clean (uncorrupted) 10-shot prompts. We use this to identify a set of causal attention
heads _A_, which are ranked based on the average indirect effect (AIE) of each head. For GPT-J
we found that the increase in performance for many tasks begins to plateau when using _|A|_ = 10
attention heads, though for some tasks using more heads increases performance even more (Figure 6).

The AIE is computed over a subset of all abstractive tasks (Appendix E), using _|P_ [˜] _t|_ = 25 corrupted
10-shot prompts per task. Because we are interested in tasks the model can successfully do via ICL,
a task is only included if its 10-shot ICL performance is better than the baseline (majority-label)
performance. For GPT-J, there are _|T |_ = 18 tasks satisfying this criteria which we use to compute
the AIE (Figure 8).

0    10   20    30 0    10   20    30

Number of Heads Number of Heads

**0** 10  20  30  40 10  20  30

0    10   20    30 **0** 10  20  30  40 10  20  30
Number of Heads Number of Heads

Number of Heads

Figure 6: Zero-shot accuracy across 18 different tasks for adding a function vector to GPT-J. We
vary the number of heads in _A_ that are used to create the function vector and find that the change in
performance begins to plateau around _|A|_ = 10 attention heads for a majority of the tasks. For this
reason, we use _|A|_ = 10 for GPT-J.

**Evaluating Function Vectors.** To evaluate a function vector’s (FV) causal effect, we add the FV to
the output of a particular layer in the network ( **h** _ℓ_ ) at the last token of a prompt _p_ with query _xq_, and
then measure whether the predicted word matches the expected answer _yq_ . This can be expressed
simply as _f_ ( _p |_ **h** _ℓ_ := **h** _ℓ_ + _vt_ ) (see section B for a more detailed explanation). We report this top-1 accuracy score over the test set under this intervention. If _yq_ is tokenized as multiple tokens, we use
the first token of _yq_ as the target token.

For all results in Section 3 and in the following appendix sections (unless stated otherwise - e.g.
Figure 4), we add the FV to the hidden state at layer _ℓ_ _≈|L|/_ 3, which we found works well in
practice. This corresponds to layer 9 for GPT-J, layer 15 for GPT-NeoX, layer 11 for Llama 2 (7B),
layer 14 for Llama 2 (13B) and layer 26 for Llama 2 (70B).

**Prompt** **Templates.** The default template we use to construct ICL prompts is:
Q: _{xik}\_ nA: _{yik}\_ n _\_ n, where _xik_ and _yik_ (or _y_ ˜ _ik_ for corrupted prompts) are substituted
for the corresponding element in brackets, and each example is concatenated together. An example of
a full prompt template is:

Q: _{xi_ 1 _}\_ nA: _{yi_ 1 _}\_ n _\_ n _. . ._ Q: _{xiN_ _}\_ nA: _{yiN_ _}\_ n _\_ nQ: _{xiq}\_ nA: (15)

To evaluate a function vector we use a few different prompt contexts. The shuffled-label prompts are
corrupted 10-shot prompts with the same form as (15), while zero-shot prompts only contain a query
_xq_, without prepended examples (e.g. Q: _{xiq}\_ nA:).

In section 3.1 we use FVs extracted from prompts made with the template shown in (15), and test
them across a variety of other templates (Table 8).

Table 8: We test a variety of ICL prompt templates in §3.1, which are shown below. The function
vectors (FVs) we collect are constructed from a default template of the form Q: _{xik}\_ nA: _{yik}\_ n _\_ n,
and tested on prompts created with the new prompt form.

Template Forms question: _{xik}\_ n answer: _{yik}\_ n _\_ n, question: _{xik}\_ n answer: _{yik}|_,
A: _{xik}\_ n B: _{yik}\_ n _\_ n, Question: _{xik}\_ n _\_ n Answer: _{yik}\_ n _\_ n,
Input: _{xik}_ Output: _{yik}|_, _{xik}\_ n _→{yik}\_ n _\_ n,
_{xik}\_ n : _{yik}\_ n, input: _{xik}_ output: _{yik}_,
question: _{xik}_ answer: _{yik}_, x: _{xik}\_ n y: _{yik}|_,
input: _{xik}|_ output: _{yik}\_ n, _{xik}_ _→{yik}\_ n _\_ n,
input: _{xik}_ output: _{yik}\_ n, In: _{xik}_ Out: _{yik}|_,
text: _{xik}|_ label: _{yik}\_ n _\_ n, x: _{xik}_ f(x): _{yik}\_ n,
x: _{xik}|_ y: _{yik}\_ n _\_ n, A: _{xik}_ B: _{yik}_,
text: _{xik}|_ label: _{yik}\_ n, x: _{xik}\_ n y: _{yik}\_ n _\_ n

D RESULTS INCLUDING INCORRECT ICL

For simplicity of presentation in Section 3, we filter the test set to cases where the model correctly
predicts _yq_ given a 10-shot ICL prompt containing query _xq_ . In this section we compare those results
to the setting in which correct-answer filtering is _not_ applied. When filtering is not applied, the causal
effects of function vectors remain essentially unchanged (Figure 7).

GPT-J 10-Shot Performance

GPT-J+FV on Unfiltered Test Set
GPT-J on Unfiltered Test Set

GPT-J+FV on Filtered Test Set

GPT-J on Filtered Test Set

Figure 7: Comparing layer-wise zero-shot results of adding a function vector to GPT-J with and
without filtering the task test set to cases where GPT-J correctly answers a 10-shot prompt. The
results when filtering the test set in this manner are shown in blue, while the results without filtering
the test set are shown in magenta. In both cases the performance is very similar, with performance
dropping only slightly on a few tasks when not filtering the dataset to correct answers. The black
dashed line corresponds to the oracle accuracy (GPT-J’s 10-shot performance) in the unfiltered setting,
while colored dashed lines correspond to GPT-J’s performance in the zero-shot setting.

E DATASETS

Here, we describe the tasks we use for evaluating the existence of function vectors. A summary of
each task can be found in Table 9.

**Antonym and Synonym.** Our antonym and synonym datasets are based on data taken from Nguyen
et al. (2017). They contain pairs of words that are either antonyms or synonyms of each other (e.g.
“good _→_ bad”, or “spirited _→_ fiery”). We create an initial dataset by combining all adjective, noun, and verb pairs from all data splits and then filter out duplicate entries. We then further filter to word
pairs where both words can be tokenized as a single token. As a result, we keep 2,398 antonym word
pairs and 2,881 synonym word pairs.

We note that these datasets originally included multiple entries for a single input word (e.g. both
“simple _→_ difficult” and “simple _→_ complex” are entries in the antonym dataset). In those cases we prompt a more powerful model (GPT-4; OpenAI, 2023) with 10 ICL examples and keep an answer as
output after manually verifying it.

**Translation.** We construct our language translation datasets – English-French, English-German,
and English-Spanish – using data from Conneau et al. (2017), which consists of a word in English and
its translation into a target language. For each language, we combine the provided train and test splits
into a single dataset and then filter out cognates. What remains are 4,705 pairs for English-French,
5,154 pairs for English-German, and 5,200 pairs for English-Spanish.

These datasets originally included multiple entries for a single input word (e.g. both “answer _→_
respuesta” and “answer _→_ contestar” are entries in the English-Spanish dataset), and so we filter
those with GPT-4, in a similar manner as described for Antonym and Synonym.

**Sentiment Analysis.** Our sentiment analysis dataset is derived from the Stanford Sentiment Treebank (SST-2) Socher et al. (2013), a dataset of movie review sentences where each review has a
binary label of either “positive” or “negative”. An example entry from this dataset looks like this:
“An extremely unpleasant film. _→_ negative”. We use the same subset of SST-2 as curated in Honovich et al. (2023), where incomplete sentences and sentences with more than 10 words are discarded,
leaving 1167 entries in the dataset. See Honovich et al. (2023) for more details.

**CommonsenseQA.** This is a question answering dataset where a model is given a question and
5 options, each labeled with a letter. The model must generate the letter of the correct answer. For
example, given “Where is a business restaurant likely to be located?” and answer options “a: town, b:
hotel, c: mall, d: business sector, e: yellow pages”, a model must generate “d”. (Talmor et al., 2019)

**AG** **News.** A text classification dataset where inputs are news headlines and the first few sentences of the article, and the labels are the category of the news article. Labels include Business,
Science/Technology, Sports, and World. (Zhang et al., 2015)

We also construct a set of simple tasks to broaden the types of tasks on which we evaluate FVs.

**Capitalize First Letter.** To generate a list of words to use to capitalize, we utilize ChatGPT [2] by
prompting it to give us a list of words. From here, we curate a dataset where the input is a single
word, and the output is the same word with the first letter capitalized.

**Lowercase First Letter.** Similar to the Capitalize First Letter task, we use the same set of words
but instead change the task to instead lowercase a word. The input is a single word title cased, and
the output is the same word, but lowercase instead.

**Country-Capital.** We also generate a list of country-capitals with ChatGPT. Here, we ask ChatGPT
to come up with a list of countries and following that, ask it to name the capitals that are related to
the countries. This dataset contains 197 country-capital city pairs.

[2https://chat.openai.com/](https://chat.openai.com/)

**Country-Currency.** We also generate a list of country-currency pairs with ChatGPT. Similar to
country-capital, we ask ChatGPT to come up with a list of countries and following that, ask it to
name the currencies that are related to the countries.

**National Parks.** The National Parks dataset consists of names of official units of national parks in
the United States, paired with the state the unit resides in (e.g. Zion National Park _→_ Utah). It was
collected from the corresponding page on Wikipedia [3], accessed July 2023.

**Park-Country.** The Park-Country dataset consists of names of national parks around the world,
paired with the country that the park is in. The countries were first generated by ChatGPT, then the
parks were also generated by ChatGPT. After, the dataset was hand-checked for factual accuracy
since ChatGPT tended to hallucinate parks for this dataset. A subset of all national parks is used.

**Present-Past.** We generate a list of present-tense verbs with ChatGPT then ask ChatGPT to find
the past-tense version. After generation, the dataset was hand-corrected for inaccuracies. The dataset
inputs are simple present tense verbs and outputs are corresponding simple past tense verbs.

**Landmark-Country.** The Landmark-Country dataset consists of entries with the name of a landmark, and the country that it is located in. The data pairs are taken from Hernandez et al. (2023b).

**Person-Instrument.** The Person-Instrument dataset contains entries with the name of a professional
musician and the instrument they play. The data pairs are taken from Hernandez et al. (2023b).

**Person-Occupation.** The Person-Occupation dataset is taken from Hernandez et al. (2023b), and
contains entries of names of well-known individuals and their occupations.

**Person-Sport.** The Person-Sport dataset is taken from Hernandez et al. (2023b), and each entry
consists of the name of a professional athlete and the sport that they play.

**Product-Company.** The Product-Company dataset contains entries with the name of a commercial
product, paired with the company that sells the product. It is curated from Hernandez et al. (2023b).

**Next-Item.** The Next-Item dataset contains pairs of words which communicate the abstract idea of
“next”. Our pairs are made up of days of the week, months of the year, letters of the alphabet (which are cyclic), and number pairs (both numeric and text form). Some examples entries in this dataset are:
“Monday” _→_ “Tuesday”, “December” _→_ “January”, “a” _→_ “b”, and “seven” _→_ “eight”.

**Previous-Item.** The Previous-Item dataset contains the reciprocal version of the pairs of words in
the “Next-Item” dataset, communicating the idea of “previous”. Example entries include: “Tuesday”
_→_ “Monday”, “January” _→_ “December”, and “a” _→_ “z”.

E.1 EXTRACTIVE TASKS.

Many NLP tasks are **abstractive** ; that is, they require the generation of information not present in the
prompt. We also wish to test whether function vectors are recoverable from **extractive** tasks—that is,
tasks where the answer is present somewhere in the prompt, and the task is to retrieve it.

**CoNLL-2003.** In our experiments, we use a subset of the CoNLL-2003 English named entity
recognition (NER) dataset Sang & De Meulder (2003), which is a common NLP benchmark for
evaluating NER models. The NER task consists of extracting the correct entity from a given sentence,
where the entity has some particular property. In our case, we create three different datasets: NERperson, NER-location, and NER-organization, where the label of each task is the name of either a
person, location, or organization, respectively. Each dataset is constructed by first combining the
CoNLL-2003 “train” and “validation” splits into a single dataset, and then filtering the data points to
only include sentences where a single instance of the specified class (person, location, or organization)
is present. This helps reduce the ambiguity of the task, as cases where multiple instances of the same
class are present could potentially have multiple correct answers.

3https://en.wikipedia.org/wiki/List_of_the_United_States_National_Park_System_official
units

As with abstractive tasks, we also construct a set of new extractive tasks.

**Choose** _n_ **th Item from List.** Here, the model is given a list of comma-separated items, and the
model is tasked with selecting the item at a specific index. We construct tasks where the list size is
either 3 or 5. In our tasks, we have the model choose either the first element or last element in the list.

**Choose Category from List.** These tasks are similar to our choose _n_ th element tasks, but instead,
the model must select an item of a particular type within a list of 3 or 5 items. A word with the correct
type is included once in the list while the remaining words are drawn from another category. The
categories we test include the following: fruit vs. animal, object vs. concept, verb vs. adjective, color
vs. animal, and animal vs. object.

**Task Name** **Task Source**

_Abstractive Tasks_

Antonym Nguyen et al. (2017)
Capitalize first letter
Capitalize
Country-capital
Country-currency
English-French Conneau et al. (2017)
English-German Conneau et al. (2017)
English-Spanish Conneau et al. (2017)
Landmark-Country Hernandez et al. (2023b)
Lowercase first letter
National parks
Next-item
Previous-item
Park-country
Person-instrument Hernandez et al. (2023b)
Person-occupation Hernandez et al. (2023b)
Person-sport Hernandez et al. (2023b)
Present-past
Product-company Hernandez et al. (2023b)
Singular-plural
Synonym Nguyen et al. (2017)

CommonsenseQA (MC-QA) Talmor et al. (2019)
Sentiment analysis (SST-2) Socher et al. (2013)
AG News Zhang et al. (2015)

_Extractive Tasks_

Adjective vs. verb
Animal vs. object
Choose first of list
Choose middle of list
Choose last of list
Color vs. animal
Concept vs. object
Fruit vs. animal
Object vs. concept
Verb vs. adjective

CoNLL-2003, NER-person Sang & De Meulder (2003)
CoNLL-2003, NER-location Sang & De Meulder (2003)
CoNLL-2003, NER-organization Sang & De Meulder (2003)

Table 9: Summary of tasks used in this study. Tasks without sources are tasks we construct.

Capitalize

0 2 4 6 8 10
Number of Shots

Commonsense_Qa

0 2 4 6 8 10
Number of Shots

English-German

0 2 4 6 8 10
Number of Shots

Lowercase_Last_Letter

0 2 4 6 8 10
Number of Shots

Person-Instrument

0 2 4 6 8 10
Number of Shots

Product-Company

0 2 4 6 8 10
Number of Shots

Capitalize_First_Letter

0 2 4 6 8 10
Number of Shots

Country-Capital

0 2 4 6 8 10
Number of Shots

English-Spanish

0 2 4 6 8 10
Number of Shots

National_Parks

0 2 4 6 8 10
Number of Shots

Person-Occupation

0 2 4 6 8 10
Number of Shots

Sentiment

0 2 4 6 8 10
Number of Shots

Capitalize_Last_Letter

0 2 4 6 8 10
Number of Shots

Country-Currency

0 2 4 6 8 10
Number of Shots

Landmark-Country

0 2 4 6 8 10
Number of Shots

Next_Capital_Letter

0 2 4 6 8 10
Number of Shots

Person-Sport

0 2 4 6 8 10
Number of Shots

Singular-Plural

0 2 4 6 8 10
Number of Shots

Antonym

0 2 4 6 8 10
Number of Shots

Capitalize_Second_Letter

0 2 4 6 8 10
Number of Shots

English-French

0 2 4 6 8 10
Number of Shots

Lowercase_First_Letter

0 2 4 6 8 10
Number of Shots

Park-Country

0 2 4 6 8 10
Number of Shots

Present-Past

0 2 4 6 8 10
Number of Shots

Synonym

0 2 4 6 8 10
Number of Shots

Figure 8: Few-shot ICL performance (top-1 accuracy) for GPT-J on a set of 25 abstractive-style tasks.
In general, more shots improves performance. However, for many of these tasks the accuracy plateaus
after a number of shots. The dotted baseline shows the accuracy of predicting only the majority label.

E.2 FEW-SHOT ICL PERFORMANCE

Figure 8 shows the few-shot performance (top-1 accuracy) for GPT-J on a larger subset of our task list.
The dotted baseline is based on the majority label for the dataset, computed as (# majority label)/(#
total instances).

Figure 9 shows the few-shot performance (top-1 accuracy) for GPT-J on additional extractive tasks.
For datasets with lists of words as input, the dotted baseline is computed to be 1 _/_ size of the input list,
(e.g. 1 _/_ 3 _,_ 1 _/_ 5), and for all other tasks it represents predicting the majority label of the dataset,
computed as (# majority label)/(# total instances).

E.3 EVALUATING FUNCTION VECTORS ON ADDITIONAL TASKS

Adjective_V_Verb_5

0 2 4 6 8 10
Number of Shots

Alphabetically_Last_5

0 2 4 6 8 10
Number of Shots

Choose_First_Of_5

0 2 4 6 8 10
Number of Shots

Choose_Middle_Of_5

0 2 4 6 8 10
Number of Shots

Concept_V_Object_5

0 2 4 6 8 10
Number of Shots

Fruit_V_Animal_3

0 2 4 6 8 10
Number of Shots

Verb_V_Adjective_3

0 2 4 6 8 10
Number of Shots

Alphabetically_First_3

0 2 4 6 8 10
Number of Shots

Animal_V_Object_3

0 2 4 6 8 10
Number of Shots

Choose_Last_Of_3

0 2 4 6 8 10
Number of Shots

Color_V_Animal_3

0 2 4 6 8 10
Number of Shots

Conll2003_Location

0 2 4 6 8 10
Number of Shots

Fruit_V_Animal_5

0 2 4 6 8 10
Number of Shots

Verb_V_Adjective_5

0 2 4 6 8 10
Number of Shots

Alphabetically_First_5

0 2 4 6 8 10
Number of Shots

Animal_V_Object_5

0 2 4 6 8 10
Number of Shots

Choose_Last_Of_5

0 2 4 6 8 10
Number of Shots

Color_V_Animal_5

0 2 4 6 8 10
Number of Shots

Conll2003_Organization

0 2 4 6 8 10
Number of Shots

Object_V_Concept_3

0 2 4 6 8 10
Number of Shots

Adjective_V_Verb_3

0 2 4 6 8 10
Number of Shots

Alphabetically_Last_3

0 2 4 6 8 10
Number of Shots

Choose_First_Of_3

0 2 4 6 8 10
Number of Shots

Choose_Middle_Of_3

0 2 4 6 8 10
Number of Shots

Concept_V_Object_3

0 2 4 6 8 10
Number of Shots

Conll2003_Person

0 2 4 6 8 10
Number of Shots

Object_V_Concept_5

0 2 4 6 8 10
Number of Shots

Figure 9: Few-shot ICL performance (top-1 accuracy) for GPT-J on a set of 27 extractive-style tasks.
In general, more shots improves model performance. A dataset with 3 or 5 at the end denotes the
size of the input list of words used. There are a few cases where the model cannot perform the task
any better than random (e.g. choose the alphabetically first word in a list), which we do not analyze
further. The dotted baseline shows the accuracy 1 _/_ (list size), or the majority label in the case of the
datasets derived from conll2003.

Antonym

0 2 4 6 8 10
Number of Shots

Capitalize_Second_Letter

0 2 4 6 8 10
Number of Shots

English-French

0 2 4 6 8 10
Number of Shots

Lowercase_First_Letter

0 2 4 6 8 10
Number of Shots

Park-Country

0 2 4 6 8 10
Number of Shots

Present-Past

0 2 4 6 8 10
Number of Shots

Synonym

0 2 4 6 8 10
Number of Shots

Capitalize

0 2 4 6 8 10
Number of Shots

Commonsense_Qa

0 2 4 6 8 10
Number of Shots

English-German

0 2 4 6 8 10
Number of Shots

Lowercase_Last_Letter

0 2 4 6 8 10
Number of Shots

Person-Instrument

0 2 4 6 8 10
Number of Shots

Product-Company

0 2 4 6 8 10
Number of Shots

Capitalize_First_Letter

0 2 4 6 8 10
Number of Shots

Country-Capital

0 2 4 6 8 10
Number of Shots

English-Spanish

0 2 4 6 8 10
Number of Shots

National_Parks

0 2 4 6 8 10
Number of Shots

Person-Occupation

0 2 4 6 8 10
Number of Shots

Sentiment

0 2 4 6 8 10
Number of Shots

Ag_News

0 2 4 6 8 10
Number of Shots

Capitalize_Last_Letter

0 2 4 6 8 10
Number of Shots

Country-Currency

0 2 4 6 8 10
Number of Shots

Landmark-Country

0 2 4 6 8 10
Number of Shots

Next_Capital_Letter

0 2 4 6 8 10
Number of Shots

Person-Sport

0 2 4 6 8 10
Number of Shots

Singular-Plural

0 2 4 6 8 10
Number of Shots

Figure 10: Few-shot ICL performance (top-1 accuracy) for Llama 2 (13B) on a set of 26 abstractivestyle tasks. Using more ICL examples (shots) improves performance for many of these tasks, though
the performance does plateau after a few shots for many of the tasks. There are a few cases where
Llama 2 (13B) cannot perform the task any better than random (e.g. capitalize the second letter in the
word), which we do not analyze further. The dotted baseline shows accuracy choosing the majority
label.

Adjective_V_Verb_5

0 2 4 6 8 10
Number of Shots

Alphabetically_Last_5

0 2 4 6 8 10
Number of Shots

Choose_First_Of_5

0 2 4 6 8 10
Number of Shots

Choose_Middle_Of_5

0 2 4 6 8 10
Number of Shots

Concept_V_Object_5

0 2 4 6 8 10
Number of Shots

Fruit_V_Animal_3

0 2 4 6 8 10
Number of Shots

Verb_V_Adjective_3

0 2 4 6 8 10
Number of Shots

Alphabetically_First_3

0 2 4 6 8 10
Number of Shots

Animal_V_Object_3

0 2 4 6 8 10
Number of Shots

Choose_Last_Of_3

0 2 4 6 8 10
Number of Shots

Color_V_Animal_3

0 2 4 6 8 10
Number of Shots

Conll2003_Location

0 2 4 6 8 10
Number of Shots

Fruit_V_Animal_5

0 2 4 6 8 10
Number of Shots

Verb_V_Adjective_5

0 2 4 6 8 10
Number of Shots

Alphabetically_First_5

0 2 4 6 8 10
Number of Shots

Animal_V_Object_5

0 2 4 6 8 10
Number of Shots

Choose_Last_Of_5

0 2 4 6 8 10
Number of Shots

Color_V_Animal_5

0 2 4 6 8 10
Number of Shots

Conll2003_Organization

0 2 4 6 8 10
Number of Shots

Object_V_Concept_3

0 2 4 6 8 10
Number of Shots

Adjective_V_Verb_3

0 2 4 6 8 10
Number of Shots

Alphabetically_Last_3

0 2 4 6 8 10
Number of Shots

Choose_First_Of_3

0 2 4 6 8 10
Number of Shots

Choose_Middle_Of_3

0 2 4 6 8 10
Number of Shots

Concept_V_Object_3

0 2 4 6 8 10
Number of Shots

Conll2003_Person

0 2 4 6 8 10
Number of Shots

Object_V_Concept_5

0 2 4 6 8 10
Number of Shots

Figure 11: Few-shot ICL performance (top-1 accuracy) for Llama 2 (13B) on a set of 27 extractivestyle tasks. Using more ICL examples improves performance for most of the tasks. A dataset name
ending with 3 or 5 denotes the size of the input word list. There are a few cases where the model
cannot perform the task any better than random (e.g. choose the alphabetically first word in a list),
which we do not analyze further. The dotted baseline shows the accuracy 1 _/_ (list size) (e.g. 1 _/_ 3 _,_ 1 _/_ 5,
or the majority label in the case of the conll2003 datasets.

Capitalize

0 4 8 12 16 20 24
Layer

English-French

0 4 8 12 16 20 24
Layer

Lowercase_First_Letter

0 4 8 12 16 20 24
Layer

Present-Past

0 4 8 12 16 20 24
Layer

Synonym

0 4 8 12 16 20 24
Layer

Capitalize_First_Letter

0 4 8 12 16 20 24
Layer

English-German

0 4 8 12 16 20 24
Layer

National_Parks

0 4 8 12 16 20 24
Layer

Product-Company

0 4 8 12 16 20 24
Layer

Country-Capital

0 4 8 12 16 20 24
Layer

English-Spanish

0 4 8 12 16 20 24
Layer

Park-Country

0 4 8 12 16 20 24
Layer

Sentiment

0 4 8 12 16 20 24
Layer

Antonym

0 4 8 12 16 20 24
Layer

Country-Currency

0 4 8 12 16 20 24
Layer

Landmark-Country

0 4 8 12 16 20 24
Layer

Person-Sport

0 4 8 12 16 20 24
Layer

Singular-Plural

0 4 8 12 16 20 24
Layer

Figure 12: Zero-Shot Top-1 Accuracy Results of adding FVs to GPT-J across 18 of our task datasets.
In addition the 6 analyzed in the main paper, we see similar results and performance across a variety
of other tasks - adding FVs in early-middle layers seems to have the most effect. Notable exceptions
include lowercase first letter, person-sport, synonym, and sentiment, where the FV doesn’t seem to
have much effect. Note we did not evaluate the FV on tasks where the model’s ICL performance was
poor (compared to the majority label baseline) (Figure 8).

Adjective_V_Verb_5

0 4 8 12 16 20 24
Layer

Choose_First_Of_5

0 4 8 12 16 20 24
Layer

Color_V_Animal_3

0 4 8 12 16 20 24
Layer

Conll2003_Location

0 4 8 12 16 20 24
Layer

Fruit_V_Animal_5

0 4 8 12 16 20 24
Layer

Verb_V_Adjective_5

0 4 8 12 16 20 24
Layer

Animal_V_Object_3

0 4 8 12 16 20 24
Layer

Choose_Last_Of_3

0 4 8 12 16 20 24
Layer

Color_V_Animal_5

0 4 8 12 16 20 24
Layer

Conll2003_Organization

0 4 8 12 16 20 24
Layer

Object_V_Concept_3

0 4 8 12 16 20 24
Layer

Animal_V_Object_5

0 4 8 12 16 20 24
Layer

Choose_Last_Of_5

0 4 8 12 16 20 24
Layer

Concept_V_Object_3

0 4 8 12 16 20 24
Layer

Conll2003_Person

0 4 8 12 16 20 24
Layer

Object_V_Concept_5

0 4 8 12 16 20 24
Layer

Adjective_V_Verb_3

0 4 8 12 16 20 24
Layer

Choose_First_Of_3

0 4 8 12 16 20 24
Layer

Choose_Middle_Of_3

0 4 8 12 16 20 24
Layer

Concept_V_Object_5

0 4 8 12 16 20 24
Layer

Fruit_V_Animal_3

0 4 8 12 16 20 24
Layer

Verb_V_Adjective_3

0 4 8 12 16 20 24
Layer

Figure 13: Zero-Shot Top-1 Accuracy Results of adding FVs to GPT-J across 22 of our task datasets.
Shown here are mainly extractive tasks, and we see similar trends across all tasks. The FV performance is fairly stagnant across layers, with many tasks having peak performance in middle layers
of the network. The model’s zero-shot performance without intervention is plotted as a dotted line.
Adding the FV causes the model to extract the correct entity much more often than the base model in
this uninformative zero-shot case.

Antonym

0 5 10 15 20 25 30 35
Layer

Country-Capital

0 5 10 15 20 25 30 35
Layer

English-Spanish

0 5 10 15 20 25 30 35
Layer

National_Parks

0 5 10 15 20 25 30 35
Layer

Person-Sport

0 5 10 15 20 25 30 35
Layer

Singular-Plural

0 5 10 15 20 25 30 35
Layer

Capitalize

0 5 10 15 20 25 30 35
Layer

Country-Currency

0 5 10 15 20 25 30 35
Layer

Landmark-Country

0 5 10 15 20 25 30 35
Layer

Park-Country

0 5 10 15 20 25 30 35
Layer

Present-Past

0 5 10 15 20 25 30 35
Layer

Synonym

0 5 10 15 20 25 30 35
Layer

Capitalize_First_Letter

0 5 10 15 20 25 30 35
Layer

English-French

0 5 10 15 20 25 30 35
Layer

Lowercase_First_Letter

0 5 10 15 20 25 30 35
Layer

Person-Instrument

0 5 10 15 20 25 30 35
Layer

Product-Company

0 5 10 15 20 25 30 35
Layer

Ag_News

0 5 10 15 20 25 30 35
Layer

Commonsense_Qa

0 5 10 15 20 25 30 35
Layer

English-German

0 5 10 15 20 25 30 35
Layer

Lowercase_Last_Letter

0 5 10 15 20 25 30 35
Layer

Person-Occupation

0 5 10 15 20 25 30 35
Layer

Sentiment

0 5 10 15 20 25 30 35
Layer

Figure 14: Zero-shot results for adding a function vector (FV) at different layers of Llama 2 (13B)
across a 23 abstractive-style tasks. Adding the FV at early-middle layers gives good performance on
the task, but there is a drop in performance when adding the FV at later layers of the model (around
layer 20-25 for Llama 2 (13B)).

Adjective_V_Verb_5

0 5 10 15 20 25 30 35
Layer

Choose_First_Of_5

0 5 10 15 20 25 30 35
Layer

Color_V_Animal_5

0 5 10 15 20 25 30 35
Layer

Conll2003_Organization

0 5 10 15 20 25 30 35
Layer

Object_V_Concept_3

0 5 10 15 20 25 30 35
Layer

Animal_V_Object_3

0 5 10 15 20 25 30 35
Layer

Choose_Last_Of_3

0 5 10 15 20 25 30 35
Layer

Concept_V_Object_3

0 5 10 15 20 25 30 35
Layer

Conll2003_Person

0 5 10 15 20 25 30 35
Layer

Object_V_Concept_5

0 5 10 15 20 25 30 35
Layer

Animal_V_Object_5

0 5 10 15 20 25 30 35
Layer

Choose_Last_Of_5

0 5 10 15 20 25 30 35
Layer

Concept_V_Object_5

0 5 10 15 20 25 30 35
Layer

Fruit_V_Animal_3

0 5 10 15 20 25 30 35
Layer

Verb_V_Adjective_3

0 5 10 15 20 25 30 35
Layer

Adjective_V_Verb_3

0 5 10 15 20 25 30 35
Layer

Choose_First_Of_3

0 5 10 15 20 25 30 35
Layer

Color_V_Animal_3

0 5 10 15 20 25 30 35
Layer

Conll2003_Location

0 5 10 15 20 25 30 35
Layer

Fruit_V_Animal_5

0 5 10 15 20 25 30 35
Layer

Verb_V_Adjective_5

0 5 10 15 20 25 30 35
Layer

Figure 15: Zero-shot results for adding a function vector (FV) at different layers of Llama 2 (13B)
across a 21 extractive-style tasks. Adding the FV at earlier layers has a higher causal effect of
performing the extracted task, while later layers see a performance dip.

Capitalize

0 16 32 48 64
Layer

English-French

0 16 32 48 64
Layer

Lowercase_First_Letter

0 16 32 48 64
Layer

Person-Occupation

0 16 32 48 64
Layer

Sentiment

0 16 32 48 64
Layer

Capitalize_First_Letter

0 16 32 48 64
Layer

English-German

0 16 32 48 64
Layer

National_Parks

0 16 32 48 64
Layer

Person-Sport

0 16 32 48 64
Layer

Singular-Plural

0 16 32 48 64
Layer

Country-Capital

0 16 32 48 64
Layer

English-Spanish

0 16 32 48 64
Layer

Park-Country

0 16 32 48 64
Layer

Present-Past

0 16 32 48 64
Layer

Synonym

0 16 32 48 64
Layer

Antonym

0 16 32 48 64
Layer

Country-Currency

0 16 32 48 64
Layer

Landmark-Country

0 16 32 48 64
Layer

Person-Instrument

0 16 32 48 64
Layer

Product-Company

0 16 32 48 64
Layer

Figure 16: Zero-shot results for adding a function vector (FV) at different layers of Llama 2 (70B)
across a 20 abstractive-style tasks. Adding the FV at early-middle layers gives good performance on
the task, and there is a sharp drop in performance when adding the FV at later layers of the model
(after layer 48 for Llama 2 (70B)).

Adjective_V_Verb_5

0 16 32 48 64
Layer

Choose_First_Of_5

0 16 32 48 64
Layer

Color_V_Animal_3

0 16 32 48 64
Layer

Conll2003_Location

0 16 32 48 64
Layer

Fruit_V_Animal_5

0 16 32 48 64
Layer

Verb_V_Adjective_5

0 16 32 48 64
Layer

Animal_V_Object_3

0 16 32 48 64
Layer

Choose_Last_Of_3

0 16 32 48 64
Layer

Color_V_Animal_5

0 16 32 48 64
Layer

Conll2003_Organization

0 16 32 48 64
Layer

Object_V_Concept_3

0 16 32 48 64
Layer

Animal_V_Object_5

0 16 32 48 64
Layer

Choose_Last_Of_5

0 16 32 48 64
Layer

Concept_V_Object_3

0 16 32 48 64
Layer

Conll2003_Person

0 16 32 48 64
Layer

Object_V_Concept_5

0 16 32 48 64
Layer

Adjective_V_Verb_3

0 16 32 48 64
Layer

Choose_First_Of_3

0 16 32 48 64
Layer

Choose_Middle_Of_3

0 16 32 48 64
Layer

Concept_V_Object_5

0 16 32 48 64
Layer

Fruit_V_Animal_3

0 16 32 48 64
Layer

Verb_V_Adjective_3

0 16 32 48 64
Layer

Figure 17: Zero-shot results for adding a function vector (FV) at different layers of Llama 2
(70B) across 22 extractive-style tasks. In general, adding the FV at early-middle layers gives good
performance on the task, and there is often a sharp drop in performance when adding the FV at later
layers of the model (around layer 48 for Llama 2 (70B)).

F PORTABILITY

In addition to the quantitative results provided in §3.1, we include additional qualitative examples for
natural text completions for the Antonym FV (Table 10). We also include qualitative examples of
completions on different natural text prompts using the English-French FV (Table 11), the EnglishSpanish FV (Table 13), and the Country-Capital FV (Table 14).

We also include additional quantitative results for testing FVs in natural text settings on various natural
text templates, averaged over 5 seeds for each of the remaining representative tasks: Capitalize (Table
16), Present-Past (Table 17), Singular-Plural (Table 18), English-French (Table 12), and CountryCapital (Table 15).

Table 10: Additional Qualitative Examples of the Antonym FV in Naturalistic Text Settings

**Antonym Task**
**Prompt:** The word “ _{_ x _}_ ”, means

**(a) GPT-J** The word “limitless”, means ”without limits”. _\_ n _\_ nThe word
**(a) GPT-J + Antonym FV** The word “limitless”, means “finite”. _\_ n _\_ nThe word

**(b) GPT-J** The word “improvement”, means ”to make better”. _\_ n _\_ nThe
**(b) GPT-J + Antonym FV** The word “improvement”, means “deterioration”. _\_ n _\_ n
**Prompt:** When I think of _{_ x _}_, I usually

**(a) GPT-J** When I think of sit, I usually think of a chair. But, in”
**(a) GPT-J + Antonym FV** When I think of sit, I usually think of stand. _\_ n _\_ n

**(b) GPT-J** When I think of maximum, think of the maximum number of people that
**(b) GPT-J + Antonym FV** When I think of maximum, I usually think of minimum. _\_ n _\_ nI think
**Prompt:** While reading a book, I came across the word “ _{_ x _}_ ”. I looked it up in a dictionary and it turns
out that it means

**(a) GPT-J** While reading a book, I came across the word “credible”. I looked it up in a dictionary and it
turns out that it means “believable”. _\_ n _\_ nI
**(a) GPT-J + Antonym FV** While reading a book, I came across the word “credible”. I looked it up in a
dictionary and it turns out that it means “unbelievable”. _\_ n _\_ n

**(b) GPT-J** While reading a book, I came across the word “marriage”. I looked it up in a dictionary and it
turns out that it means “a legal union between a man and
**(b) GPT-J + Antonym FV** While reading a book, I came across the word “marriage”. I looked it up in a
dictionary and it turns out that it means “divorce”. _\_ n _\_ nI came
**Prompt:** The word _{_ x _}_ can be understood as a synonym for

**(a) GPT-J** The word erroneous can be understood as a synonym for wrong, but it is also a synonym for
**(a) GPT-J + Antonym FV** The word erroneous can be understood as a synonym for correct. _\_ n _\_ nThe

**(b)** **GPT-J** The word injure can be understood as a synonym for harm, but it also has a more specific
meaning
**(b) GPT-J + Antonym FV** The word injure can be understood as a synonym for heal. _\_ n _\_ nThe word

Table 11: Evaluating the English-French FV in Naturalistic Text Settings

**English-French Task**
**Prompt:** The word “ _{_ x _}_ ”, means

**(a) GPT-J** The word “readers”, means “people who read”. _\_ n _\_ nThe
**(a) GPT-J + English-French FV** The word “readers”, means “lecteurs” _\_ n _\_ nThe

**(b) GPT-J** The word “sky”, means “the heavens” in many languages.
**(b) GPT-J + English-French FV** The word “sky”, means “ciel” in franc¸ais

**(c) GPT-J** The word “daily”, means “every day”. _\_ n _\_ nThe word
**(c) GPT-J + English-French FV** The word “daily”, means “tous les jours”,
**Prompt:** The word ‘ _{_ x _}_ ’ can be understood as a synonym for

**(a) GPT-J** The word ‘link’ can be understood as a synonym for ‘connection’ or ‘relation’. The term ‘link’
is used in the context of the Internet, but it can be used in other contexts as
**(a) GPT-J + English-French FV** The word ‘link’ can be understood as a synonym for ‘lien’, et le mot
‘lien’ peut ˆetre compris comme un synonyme de ‘lien’, et

**(b) GPT-J:** The word ‘prohibited’ can be understood as a synonym for ‘forbidden’ or ‘forbidden to do’.
**(b) GPT-J + English-French FV** ”The word ‘prohibited’ can be understood as a synonym for ‘interdit’
**Prompt:** While reading a book, I came across the word “ _{_ x _}_ ”. I looked it up in a dictionary and it turns
out that it means

**(a) GPT-J** While reading a book, I came across the word “knight”. I looked it up in a dictionary and it
turns out that it means “a person who is a member of a military order”. _\_ n _\_ nI was wondering if there is a
similar word for a person who is a member
**(a) GPT-J + English-French FV** While reading a book, I came across the word “knight”. I looked it up in
a dictionary and it turns out that it means “chevalier”. _\_ n _\_ nJe lis un livre, et c´est un chevalier.

Table 12: Natural text portability of the English-French FV. Given a natural template we substitute a
query word for ‘x’. We measure accuracy based on whether the correct french translation is produced
in this setting within 5 generated tokens.

Prompt GPT-J +English-French FV

The word “x”, means 0 _._ 0 _±_ 0 _._ 0% 51 _._ 2 _±_ 1 _._ 2%
When I think of the word “x”, it usually means 0 _._ 1 _±_ 0 _._ 1% 41 _._ 7 _±_ 1 _._ 3%
When I think of x, I usually 0 _._ 0 _±_ 0 _._ 0% 19 _._ 0 _±_ 1 _._ 5%
While reading a book, I came across the word “x”. I 0 _._ 1 _±_ 0 _._ 1% 45 _._ 6 _±_ 1 _._ 1%
looked it up in a dictionary and it turns out that it means

The word x can be understood as a synonym for 0 _._ 3 _±_ 0 _._ 2% 44 _._ 2 _±_ 1 _._ 6%

Table 13: Evaluating the English-Spanish FV in Naturalistic Text Settings

**English-Spanish Task**
**Prompt:** The word “ _{_ x _}_ ”, means

**(a) GPT-J** The word “children”, means “offspring of the human species”. _\_ n _\_ nThe word “children” is
used in the Bible
**(a) GPT-J + English-Spanish FV** The word “children”, means “ninos˜ ” in espanol˜ . _\_ n _\_ nThe word “children”, means

**(b) GPT-J** The word “foods”, means “food” in the sense of “food” and “drink” in the sense of
**(b)** **GPT-J** **+** **English-Spanish** **FV** The word “foods”, means “alimentos” _\_ n _\_ nThe word “alimentos”,
signifca “alimentos
**Prompt:** While reading a book, I came across the word “ _{_ x _}_ ”. I looked it up in a dictionary and it turns
out that it means

**(a) GPT-J** While reading a book, I came across the word “attacking”. I looked it up in a dictionary and it
turns out that it means “to attack”. _\_ n _\_ nI was wondering if there is a difference between the two words.
**(a) GPT-J + English-Spanish FV** While reading a book, I came across the word “attacking”. I looked
it up in a dictionary and it turns out that it means “atacar”. _\_ n _\_ nI le´ıdo un libro y encontre´ la palabra
“atacar”. _\_ n _\_ nPero

**(b) GPT-J** While reading a book, I came across the word “computer”. I looked it up in a dictionary and it
turns out that it means “a machine that computes”. _\_ n _\_ nI was surprised to learn that the word “computer”
is derived from the word “computor”,
**(b) GPT-J + English-Spanish FV** While reading a book, I came across the word “computer”. I looked
it up in a dictionary and it turns out that it means computador. _\_ n _\_ nI looked it up in a dictionary y en un
diccionario y que dice que es un computador. _\_ n _\_ n
**Prompt:** When I think of _{_ x _}_, I usually

**(a) GPT-J** When I think of find, I usually think of the command line. It’s
**(a) GPT-J + English-Spanish FV** When I think of find, I usually think de encontrar. _\_ n _\_ nWhen I

**(b) GPT-J** When I think of tomb, I usually think of a tombstone. But, in the case of the tomb of
**(b)** **GPT-J** **+** **English-Spanish** **FV** When I think of tomb, I usually pense en camara´ de sepulcro. _\_ n _\_ nCuando

**(c) GPT-J** When I think of shoes, I usually think of the ones I wear. I’m not a shoe person
**(c) GPT-J + English-Spanish FV** When I think of shoes, I usually pense sobre zapatos. _\_ n _\_ nWhen I pense
sobre

Table 14: Evaluating the Country-Capital FV in Naturalistic Text Settings

**Country-Capital Task**
**Prompt:** A couple of years ago I visited _{_ x _}_, and

**(a) GPT-J:** A couple years ago I visited Grenada, and I was struck by the beauty of the island. I was also
struck by the fact that the island was a British colony until 1974. I was also
**(a)** **GPT-J** **+** **Country-Capital** **FV** A couple years ago I visited Grenada, and I was in the capital, St.
George’s. I was there to see the Grenada National Museum, and I was there to see

**(b) GPT-J:** A couple years ago I visited Italy, and I was lucky enough to be able to visit the city of Florence.
I was there for a few days, and I was able to see a lot
**(b) GPT-J + Country-Capital FV** A couple years ago I visited Italy, and I was in Rome. I was in the
Colosseum, and I was in the Roman Forum. I was in the Pantheon, and

**(c) GPT-J:** couple years ago I visited Belarus, and I was surprised to see that the country was still using
the old Soviet-era typewriters. I was even more surprised to see that the typew
**(c) GPT-J + Country-Capital FV** A couple years ago I visited Belarus, and I was in Minsk. I was there
for a few days, and I was in Minsk

**(d) GPT-J:** A couple years ago I visited Mongolia, and I was struck by the fact that the country was so
different from the rest of the world. It was a land of nomads, and the people
**(d) GPT-J + Country-Capital FV** A couple years ago I visited Mongolia, and I was in Ulaanbaatar. I was
there for a few days, and I was in the capital city of Ulaanbaatar
**Prompt:** If you ever travel to _{_ x _}_, you have to visit

**(a) GPT-J:** If you ever travel to Italy, you have to visit the city of Florence. It is the capital of Tuscany and
the most visited city in Italy. It is also the home of the famous U
**(a) GPT-J + Country-Capital FV** If you ever travel to Italy, you have to visit Rome. It’s the capital of
Italy and the largest city in the world. It’s also the most visited city in the world

**(b) GPT-J:** If you ever travel to Thailand, you have to visit the island of Koh Samui. It is a beautiful island
with a lot of things to do. The island is famous for its beaches, water sports
**(b)** **GPT-J** **+** **Country-Capital** **FV** If you ever travel to Thailand, you have to visit Bangkok. It is
the capital of Thailand and the largest city in the country. Bangkok is the most populous city in the
world. _\_ n _\_ nBangkok

**(c) GPT-J:** If you ever travel to Saint Lucia, you have to visit the Pitons. The Pitons are a group of three
mountains that are located on the island of Saint Lucia. The Pitons are the highest mountains
**(c) GPT-J + Country-Capital FV** If you ever travel to Saint Lucia, you have to visit the capital city of
Castries. It is the most beautiful city in the Caribbean. It is a very beautiful city. It is a very beautiful city
**Prompt:** When you think of _{_ x _}_,

**(a) GPT-J:** When you think of Netherlands, you probably think of tulips, windmills, and cheese. But the
Netherlands is also home to a thriving cannabis industry
**(a) GPT-J + Country-Capital FV** When you think of Netherlands, you think of Amsterdam. But there are
many other cities in the Netherlands. Here are some of the best places to visit in

**(b) GPT-J:** When you think of Egypt, you probably think of pyramids, mummies, and the Nile River. But
did you know that Egypt is also home to
**(b) GPT-J + Country-Capital FV** When you think of Egypt, you think of Cairo, the pyramids, and the
Nile. But there are many other places to visit in Egypt. Here

Table 15: Natural Text Portability quantitative results for the Country-Capital FV. We substitute in
query country names for ‘x’, and measure accuracy based on whether the correct capital city name is
produced within 10 generated tokens.

Prompt GPT-J +Country-Capital FV

A couple of years ago I visited _{_ x _}_, and 3 _._ 9 _±_ 3 _._ 2% 56 _._ 7 _±_ 5 _._ 3%
If you ever travel to _{_ x _}_, you have to visit 23 _._ 2 _±_ 3 _._ 5% 70 _._ 4 _±_ 3 _._ 7%
When you think of _{_ x _}_, 7 _._ 4 _±_ 4 _._ 8% 72 _._ 4 _±_ 2 _._ 7%

Table 16: Natural text portability of the Capitalize FV. Given a natural template we substitute a query
word for ‘x’. We measure accuracy based on whether the correct capitalization is produced in this
natural text setting within 5 generated tokens.

Prompt GPT-J +Capitalize FV

The word “x”, means 5 _._ 6 _±_ 0 _._ 5% 94 _._ 3 _±_ 1 _._ 0%
When I think of the word “x”, it usually means 3 _._ 7 _±_ 0 _._ 7% 84 _._ 5 _±_ 1 _._ 3%
When I think of x, I usually 12 _._ 5 _±_ 2 _._ 1% 76 _._ 1 _±_ 2 _._ 9%
While reading a book, I came across the word “x”. I 6 _._ 8 _±_ 0 _._ 7% 97 _._ 8 _±_ 0 _._ 6%
looked it up in a dictionary and it turns out that it means

The word x can be understood as a synonym for 5 _._ 5 _±_ 0 _._ 8% 81 _._ 5 _±_ 2 _._ 8%

Table 17: Natural text portability of the Present-Past FV. Given a natural template we substitute a
query word for ‘x’. Then then measure accuracy based on whether under the FV intervention the
correct past-tense word is produced within 5 generated tokens.

Prompt GPT-J +Present-Past FV

The word “x”, means 0 _._ 0 _±_ 0 _._ 0% 49 _._ 3 _±_ 4 _._ 5%
When I think of the word “x”, it usually means 0 _._ 1 _±_ 0 _._ 3% 67 _._ 2 _±_ 4 _._ 2%
When I think of x, I usually 0 _._ 7 _±_ 0 _._ 7% 16 _._ 1 _±_ 5 _._ 4%
While reading a book, I came across the word “x”. I 0 _._ 0 _±_ 0 _._ 0% 55 _._ 8 _±_ 3 _._ 6%
looked it up in a dictionary and it turns out that it means

The word x can be understood as a synonym for 0 _._ 0 _±_ 0 _._ 0% 57 _._ 2 _±_ 3 _._ 0%

Table 18: Natural text portability of the Singular-Plural FV. Given a natural template we substitute a
query word for ‘x’. Then then measure accuracy based on whether under the FV intervention the
correct past-tense word is produced within 5 generated tokens.

Prompt GPT-J +Singular-Plural FV

The word “x”, means 0 _._ 0 _±_ 0 _._ 0% 82 _._ 9 _±_ 3 _._ 6%
When I think of the word “x”, it usually means 0 _._ 0 _±_ 0 _._ 0% 72 _._ 2 _±_ 0 _._ 6%
When I think of x, I usually 0 _._ 0 _±_ 0 _._ 0% 36 _._ 0 _±_ 8 _._ 4%
While reading a book, I came across the word “x”. I 0 _._ 0 _±_ 0 _._ 0% 81 _._ 1 _±_ 2 _._ 4%
looked it up in a dictionary and it turns out that it means

The word x can be understood as a synonym for 0 _._ 3 _±_ 0 _._ 6% 77 _._ 0 _±_ 5 _._ 0%

G CAUSAL MEDIATION ANALYSIS

In this section we include additional figures showing the average indirect effect (AIE) split up across
tasks for GPT-J (Figure 18), as well as the AIE for other models we evaluated.

0.05

0.00

0.05

0.1

0.0

0.1

0.05

0.00

0.05

0.05
0.00
0.05

Antonym

0 10 20

Country-Currency

0 10 20

Landmark-Country

0 10 20

Person-Sport

0 10 20

Singular-Plural

0 10 20
Layer

0.05

0.00

0.05

0.05

0.00

0.05

0.1

0.0

0.1

0.05

0.00

0.05

0.05

0.00

0.05

Capitalize

0 10 20

English-French

0 10 20

Country-Capital

0 10 20

English-Spanish

0 10 20

Park-Country

0 10 20

Sentiment

0 10 20

0.25

0.00

0.25

0.1

0.0

0.1

0.1

0.0

0.1

0.025

0.000

0.025

Capitalize_First_Letter

0 10 20

English-German

0 10 20

National_Parks

0 10 20

Product-Company

0 10 20

Lowercase_First_Letter

0.1

0.0

0.1

0.1

0.0

0.1

0.05
0.00
0.05

0.05

0.00

0.05

0.005
0.000
0.005

0 10 20

Present-Past

0 10 20

Synonym

0 10 20
Layer

Figure 18: Average Indirect Effect (AIE) for attention heads in GPT-J, shown by task. The set of
heads that have higher causal effect is fairly consistent across tasks.

The AIE for a particular model is computed across all abstractive tasks where the model can perform
the task better than the majority label baseline given 10 ICL examples. We include heatmaps of the
AIE for each attention head in Llama 2 (7B) (Figure 19), Llama 2 (13B) (Figure 20), and Llama 2
(70B) (Figure 22), as well as GPT-NeoX (Figure 21). For each model, the heads with the highest AIE
are typically clustered in the middle layers of the network. In addition the maximum AIE across all
heads tends to drop slightly as the size of the model increases, though the total number of heads also
increases. In Llama 2 (7B) the max AIE is _≈_ 0 _._ 047, while in Llama 2 (70B) the max AIE is _≈_ 0 _._ 037.

## 0 10 20 30

0.04

-0.04

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||

## Layer

Figure 19: Average Indirect Effect (AIE) for each attention head in Llama 2 (7B) at the final token.
This is computed across all abstractive tasks where the model can perform the task with 10 ICL
examples. The heads with the highest AIE are mainly clustered in the middle layers of the network.
Compared to GPT-J the AIE of the most influential heads is less ( _≈_ 0 _._ 047 vs. _≈_ 0 _._ 053 in GPT-J) but
there are also more than double the number of attention heads

## 0

## 10 20 30

0.04

-0.04

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||

## Layer

Figure 20: Average Indirect Effect (AIE) for each attention head in Llama 2 (13B) at the final token.

0.03

-0.03

0 10 20 30 40
Layer

Figure 21: Average Indirect Effect (AIE) for each atttention head in GPT-NeoX at the final token.
Interestingly, the heads with the highest AIE here are clustered in earlier middle layers (from layer
10-20), whereas in other models the heads are clustered more towards the middle of the network.

0.03

-0.03

0 10 20 30 40 50 60 70
Layer

Figure 22: Average Indirect Effect (AIE) for each attention head in Llama 2 (70B) at the final token.

H ATTENTION PATTERNS AND PREFIX-MATCHING SCORE

Across a variety of tasks, the heads with highest causal effect have a consistent attention pattern
where the attention weights on few-shot ICL prompts are the strongest on the output tokens of each
in-context example. Here we show this pattern for GPT-J on 4 additional tasks (Figure 23, Figure
24), which match the patterns shown in the main paper (Figure 3b). This is similar to the attention
pattern that might be expected of “induction heads”, which has previously been shown to arise when
a prompt contains some repeated structure (Elhage et al., 2021; Olsson et al., 2022).

Singular-Plural

Present-Past

1.0

0.5

0.0

Figure 23: Attention weight visualizations for the singular-plural and present-past tasks for the
attention heads with the top 10 average indirect effects in GPT-J. Across tasks, the attention weights
are consistently the strongest on the output tokens of each exemplar.

Country-Capital

Capitalize

1.0

0.5

0.0

Figure 24: Attention weight visualizations for the country-capital and capitalize tasks for the attention
heads with the top 10 average indirect effects in GPT-J. Across tasks, the attention weights are
consistently the strongest on the output tokens of each exemplar.

To further investigate whether the heads identified via causal mediation analysis are “induction heads”,
we compute the prefix-matching score for each head in GPT-J. We follow the same procedure as
described in (Olsson et al., 2022; Wang et al., 2022a), which computes the prefix-matching score as
the average attention weight on a token _B_ when given a sequence of the form [ _A_, _B_, . . ., _A_ ]. This is measured on sequences of repeated random tokens. We do this for each head in GPT-J with results
shown in Figure 25b.

Figure 25: (a) Average Indirect Effect (AIE) per attention head for GPT-J. (b) Prefix-matching score
per attention head for GPT-J. For both (a) and (b), we highlight in pink the top 10 heads by AIE.
There are three heads that have both a relatively high AIE and prefix-matching score (Layer-Head
Index = 8-1, 12-10, and 24-6). There are also several heads with high AIE that do not have a high
prefix-matching score, and vice-versa.

We find that three of the heads out of those with the top 10 highest AIEs (Figure 25) also have high
prefix-matching scores. In terms of “Layer-Head Index”, these are heads 8-1, 12-10, and 24-6, with
prefix-matching scores of 0.49, 0.56, and 0.31, respectively.

While (Elhage et al., 2021; Olsson et al., 2022) show that induction heads play a critical role in
copying forward previously seen tokens, our results show that they are also among the set of heads,
_A_, that have the highest AIE when resolving few-shot ICL prompts.

There are several other heads we identified with relatively high causal effect that have the same
attention pattern activation on few-shot ICL prompts, but do not produce the same “induction”
attention pattern on sequences of random repeated tokens.

This suggests that while induction heads play a role in the formation of function vectors, there are
other heads that also contribute relevant information that may not be induction heads of the type
observed by (Elhage et al., 2021; Olsson et al., 2022).

I DECODING VOCABULARY EVALUATION

Table 19: Additional tasks and the top 5 vocabulary tokens of their decoded FV. Across a variety of
outputs, most encodings are aligned to the task they were extracted from.

**Task** _t_ **Tokens in the distribution** _D_ ( _vt_ ) **in order of decreasing probability**

Capitalize First Letter ‘CN’, ‘DR’, ‘RR’, ‘ Ct’, ‘Ct’
Country-Currency ‘ Japanese’, ‘ Chinese’, ‘ Arabic’, ‘ Russian’, ‘ American’
English-German ‘ ˆaK¸ [ˇ] l’, ‘ [¨] E [¯] I’, ‘ ˋe’, ‘actual’, ‘ c¸¥l’
English-Spanish ‘ ˆaK¸ [ˇ] l’, ‘ ˋe’, ‘ c¸¥l’, ‘ masc’, ‘operator’
Landmark-Country ‘ Germany’, ‘ Japan’, ‘ Netherlands’, ‘ Italy’, ‘ Spain’
Lowercase First Letter ‘dr’, ‘ nr’, ‘ lc’, ‘ mc’, ‘ mr’
National Parks ‘ Connecticut’, ‘ California’, ‘ Wisconsin’, ‘ Netherlands’, ‘ Pennsylvania’
Park-Country ‘ Netherlands’, ‘ Germany’, ‘ Japan’, ‘ Italy’, ‘ Mexico’
Person-Sport ‘ basketball’, ‘ football’, ‘ soccer’, ‘ baseball’, ‘ tennis’
Product-Company ‘ Microsoft’, ‘ Motorola’, ‘ Samsung’, ‘ Disney’, ‘ IBM’
Sentiment ‘ positive’, ‘ negative’, ‘positive’, ‘negative’, ‘ neutral’
Synonym ‘ edible’, ‘ adjective’, ‘ noun’, ‘ slang’, ‘ caster’

Here, we present more results on the evaluation of decoding vocabularies of FVs, over additional
datasets. Across the tasks, we affirm that output spaces seem to be frequently encoded in the FVs,
as can be seen in Table 19. In particular, in cases such as sentiment that follow a rigid pattern, the
tokens referring to the output distribution for sentiment is well encoded. On the other hand, some
tasks like language translation do not have output spaces well-encoded in the FVs.

J SCALING EFFECTS

Can we consistently locate function vectors given various sizes of a single language model architecture? We test this by observing all sizes of Llama 2, ranging from 7B parameters to 70B. We use the
same methods as in §3.1, adding function vectors to each layer of the model and observing accuracy
on our subset of 6 tasks at each layer.

Country-Capital

0 6 12 18 24 30

English-French

0 6 12 18 24 30

Present-Past

0 6 12 18 24 30

Singular-Plural

0 6 12 18 24 30

Antonym

0 6 12 18 24 30

0 8 16 24 32

Capitalize

0 6 12 18 24 30

0 8 16 24 32 0 8 16 24 32 0 8 16 24 32 0 8 16 24 32 0 8 16 24 32

0 16 32 48 64
Layer

0 16 32 48 64
Layer

0 16 32 48 64
Layer

0 16 32 48 64
Layer

0 16 32 48 64
Layer

0 16 32 48 64
Layer

Figure 26: Zero-shot accuracy across Llama 2 model sizes in zero-shot settings. We show accuracies
before adding the function vector (dotted lines) and after adding the FV to a specific layer (solid
lines).

We find that results (Figure 26) are largely consistent across model sizes. Function vectors generally
result in the highest zero-shot accuracies when added to the early to middle layers; this is true
regardless of the total number of layers in the model.

K COMPOSITION ON OTHER MODELS

In this section we include additional composition results for Llama 2 13B and 70B models in Tables
20 and 21 respectively.

Table 20: The accuracy of ICL, calculated FV _vBD_ zero-shot interventions, and vector-composed _vBD_ _[∗]_ [zero-]

shot interventions when performing several list-oriented tasks on LLaMA 2 (13B).

Task ICL (ten-shot) _vBD_ (FV on zero-shot) _vBD_ _[∗]_ [(sum on zero-shot)]

Last-Antonym 0 _._ 53 _±_ 0 _._ 02 0.26 _±_ 0.03 0.17 _±_ 0.02
Last-Capitalize 0 _._ 94 _±_ 0 _._ 01 0.63 _±_ 0.03 0.70 _±_ 0.03
Last-Country-Capital 0 _._ 86 _±_ 0 _._ 02 0.73 _±_ 0.02 0.37 _±_ 0.04
Last-English-French 0 _._ 75 _±_ 0 _._ 02 0.32 _±_ 0.02 0.12 _±_ 0.02
Last-Present-Past 0 _._ 96 _±_ 0 _._ 01 0.22 _±_ 0.02 0.24 _±_ 0.02
Last-Singular-Plural 0 _._ 89 _±_ 0 _._ 01 0.43 _±_ 0.03 0.53 _±_ 0.03
Last-Capitalize-First-Letter 0 _._ 85 _±_ 0 _._ 02 0.89 _±_ 0.02 0.89 _±_ 0.02
Last-Product-Company 0 _._ 47 _±_ 0 _._ 01 0.44 _±_ 0.03 0.60 _±_ 0.03

Table 21: LLaMA 2 (70B)

Task ICL (ten-shot) _vBD_ (FV on zero-shot) _vBD_ _[∗]_ [(sum on zero-shot)]

Last-Antonym 0 _._ 67 _±_ 0 _._ 03 0.43 _±_ 0.03 0.47 _±_ 0.03
Last-Capitalize 0 _._ 99 _±_ 0 _._ 00 0.93 _±_ 0.01 0.95 _±_ 0.01
Last-Country-Capital 0 _._ 81 _±_ 0 _._ 03 0.91 _±_ 0.02 0.94 _±_ 0.02
Last-English-French 0 _._ 84 _±_ 0 _._ 02 0.13 _±_ 0.01 0.17 _±_ 0.03
Last-Present-Past 0 _._ 98 _±_ 0 _._ 01 0.93 _±_ 0.01 0.94 _±_ 0.01
Last-Singular-Plural 0 _._ 98 _±_ 0 _._ 01 0.69 _±_ 0.04 0.69 _±_ 0.04
Last-Capitalize-First-Letter 0 _._ 67 _±_ 0 _._ 03 0.60 _±_ 0.02 0.68 _±_ 0.03
Last-Product-Company 0 _._ 49 _±_ 0 _._ 02 0.34 _±_ 0.01 0.34 _±_ 0.03

L EVALUATING FUNCTION VECTORS ON CYCLIC TASKS

In Appendix A, we discuss whether function vectors (FVs) can be thought of as simple word vector
offsets, and show that cyclic tasks (such as antonyms) are a counterexample to this claim. In this
section we report the causal effects of function vectors on two additional tasks with cyclic subsets
—“next-item” and “previous-item” — providing further evidence that function vectors are not just simple semantic vector offsets but instead can be thought of as a trigger of nontrivial functions.

Saturday

Wenesday

Friday Thursday

Saturday

Wenesday

Friday Thursday

Next Item Previous Item

Figure 27: An example of cyclic structure for days of the week, which is a subset of the data for
both the Next-Item and Previous-Item tasks. The cycles in each task follow the opposite order (e.g.
Next-Item(Monday) = Tuesday, but Previous-Item(Monday) = Sunday.

The “next-item” task contains pairs of words which are related via the abstract idea of “next”. The
“previous-item” task contains the reciprocal version of the word pairs in the “next-item” task. Flipping the direction in this manner means each pair communicates the idea of “previous” instead. Both tasks
are collected over a heterogeneous set of sequential data that includes cyclic types such as days of the
week, months of the year, and letters of the alphabet, as well as non-cyclic types such as numbers and
roman numerals. We include samples of example data pairs for these datasets in Appendix E.

However, a single ICL example for the Previous-Item task might look like “Q: Friday _\_ nA:
Thursday _\_ n _\_ nQ: six _\_ nA: five _\_ n _\_ nQ: a _\_ nA:z _\_ n _\_ nQ: VII _\_ nA:VI _\_ n _\_ nQ: September _\_ nA:”.
The model would ideally be able to answer “August” given this ICL prompt.

0 5 10
N Shots

GPT-J Llama 2 (7B) Llama 2 (13B) Llama 2 (70B)

0 5 10
N Shots

0 5 10
N Shots

0 5 10
N Shots

Figure 28: ICL performance on the “Next-Item” and “Previous-Item” cyclic tasks for 4 different
models. The performance is usually better for the next-item task than for the previous-item task.
However, 10-shot performance suggests these models are able to perform these tasks fairly well.

In Figure 28, we report the ICL n-shot performance of each of these two tasks for GPT-J, and each
model in the Llama 2 family. We find that the models perform this task well given 10 example pairs,
with the performance of the next-item task being higher than the previous-item task. Performance
generally increases when more examples are provided.

Llama 2 (13B)

0 8 16 24 32 40

0 8 16 24 32 40
Layer

Llama 2 (70B)

0 16 32 48 64 80

0 16 32 48 64 80
Layer

GPT-J

0 5 10 15 20 25

0 5 10 15 20 25
Layer

Llama 2 (7B)

0 6 12 18 24 30

0 6 12 18 24 30
Layer

Figure 29: Zero-shot accuracy on the “Next Item” and “Previous Item” cylic tasks for GPT-J and
all sizes of Llama 2. We show model accuracies before adding the function vector (dotted lines)
and after adding the FV to a specific layer (solid lines). The function vector improves performance
significantly for both tasks compared to the model’s zero-shot baseline.

We extract a function vector for each of these tasks and evaluate their performance in the zero-shot
setting, adding the function vector to different layers of the network. In Figure 29, we report the
zero-shot accuracy of each model before adding the function vector with a dashed line, and the
accuracy after adding a function vector with a solid line. We see that the function vector significantly
improves performance for each task compared to the zero-shot baseline. In addition, the trends for
these datasets generally follows those previously reported for other tasks (cyclic or not). The peak
performance is achieved when adding the FV to early-middle layers, and there is a sharp drop in
performance about 2 _/_ 3 of the way through the network.

Table 22: A few example outputs of adding the “next-item” and “previous-item” function vectors to
layer 9 of GPT-J. We see that the function vectors are able to correctly trigger the cyclic behavior of
“next” or “previous” when presented with a boundary case, while the base model usually defaults to

copying the input query.

**Input Prompt:** ‘Q: Sunday _\_ nA:’ ‘Q: December _\_ nA:’ ‘Q: z _\_ nA:’ ‘Q: seven _\_ nA:’ ‘Q: 21 _\_ nA:’ ‘Q: Monday _\_ nA:’ ‘Q: January _\_ nA:’

**GPT-J** Sunday December z eight I don’t know Tuesday January

**GPT-J+Next-Item FV** Monday January a eight 22 Tuesday February

**GPT-J+Previous-Item FV** Saturday November y six 20 Sunday December

In Table 22, we include a few example outputs of zero-shot prompts for both the baseline model and
the model when we add the corresponding function vectors of either “next-item” or “previous-item”,
showing their ability to correctly induce the “next” or “previous” cyclic behavior.

For antonyms, the cyclic behavior gives a contradiction in two additions. That is, given a word _w_ 1
and a vector offset _v_ that can give the antonym of _w_ 1, then we expect to return to _w_ 1 after adding _v_
again (i.e _w_ 1 + 2 _∗_ _v_ = _w_ 1). The cyclic subsets in studied here have longer cycles (e.g. for days of
the week, and an offset _v_ _[′]_, we’d expect _w_ 2 + 7 _∗_ _v_ _[′]_ = _w_ 2), but the same reasoning applies. Because
FVs can trigger the corresponding cyclic behavior, this provides additional evidence that they are not
just doing simple word vector arithmetic.

M AN ALTERNATIVE EVALUATION OF FUNCTION VECTORS

Recall that we define a function vector ( _vt_ ) for a particular task ( _t_ ) as the sum of the task-conditioned
mean activations ( _a_ ¯ _[t]_ _ℓj_ [) over a small set of attention heads (] _[A]_ [) (see equation 5, Section 2.3).] [Given a]

function vector created in this manner, we can test its causal effects by adding it to a single hidden
state of a model at a particular layer and measuring its ability to trigger a particular task behavior (see
Appendix B.2 for more details).

An alternative approach to test whether the outputs of the attention heads in _A_ can trigger a particular
task behavior is to instead add their task-conditioned mean activations ( _a_ ¯ _[t]_ _ℓj_ [) to their corresponding]

layer’s hidden states, and to do so at every layer that is implicated by the heads contained in _A_ . This
is in contrast to the FV, which adds all these attention head outputs to a single layer.

As the model performs computation at layer _k_, the alternative approach updates the hidden state **h** _k_ by
adding the task-conditioned mean activations of all heads in _A_ that output to layer _k_ . If we represent

_′_

the attention heads in _A_ with (layer, head index) tuples, then we write the updated hidden state **h** _k_ [as:]

( _ℓ,j_ ) _∈A | ℓ_ = _k_

**h**

_′_
_k_ [=] **[ h]** _[k]_ [+]

_a_ ¯ _[t]_ _ℓj_ (16)

We perform the update intervention as specified in equation 16 for all layers represented by _A_ .

GPT-J Performance GPT-J+FV Performance GPT-J+Alternative Performance

Figure 30: Comparing the causal effects of function vectors (solid blue line) and an alternative
approach (red dashed line), which adds the components of an FV to their respective layers instead
of at a single concentrated layer. The model baseline in each setting is shown with a dashed blue
line. Zero-shot results for 6 tasks are shown in the first row, and shuffled-label results are shown in
the second row. In the zero-shot setting, the alternative approach matches FV performance for most
tasks. It performs worse for English-French translation and better on the Present-Past task. In the
shuffled-label setting, the alternate approach matches FV performance for all tasks.

In Figure 30, we compare the causal effects of the alternative approach described in equation 16 to
the original function vector formulation for both zero-shot and shuffled-label contexts across our 6
representative tasks using GPT-J. The base model performance is shown with a dashed blue line, and
the solid blue line shows performance when we add the function vector to layer _ℓ_ . The results of the
alternative approach are shown with a red dashed line.

In the zero-shot setting, the alternative approach matches the performance of the function vector for a
majority of the tasks. It performs worse on English-French, and better on the Present-Past task. In the
shuffled-label setting, the alternative approach matches the causal effects of function vector for all
tasks.

On average, we find that using the alternative approach to measure the causal effects of _A_ compared
to the original function vector formulation works about as well, typically achieving the same peak
performance. However, the function vector approach does highlight an interesting phenomena of
performance dropoff around 2/3 of the way through the network which is not possible to see when
using the alternative approach.

N INVESTIGATING FUNCTION VECTOR EFFECTS IN VOCABULARY SPACE

In the main paper (Section 3.2, Table 6), we examine quantitative evidence that the action of a
function vector (FV) cannot be explained by simply boosting a set of words directly encoded by the
function vector in the style of Dar et al. (2023), see Appendix A for a more detailed discussion. In this
section we examine the causal effects of several function vectors in vocabulary space to understand
the relationship between the words that are encoded in a function vector and the words that are
boosted by the transformer when we intervene with an FV.

Unlike previous analysis, in this section we investigate how adding a function vector ( _vt_ ) to layer _ℓ_
changes the distribution of log probabilities over a set of relevant tokens ( _wi_ _∈_ _W_ ) compared to the
baseline model’s response. That is, for a token _wi_ we compute:

_△_ logprob = log( _f_ ( _p_ _[t]_ _|_ **h** _ℓ_ := **h** _ℓ_ + _vt_ )[ _wi_ ]) _−_ log( _f_ ( _p_ _[t]_ )[ _wi_ ]) (17)

We investigate the tokens with the highest increase in log probabilities under FV intervention and
include a few examples of the behavior we observe in Table 23. Here we show a few examples of the
tokens with the largest _△_ logprob for three tasks: Country-Capital, Antonym, English-French.

Table 23: The tokens with the highest increase in _△_ logprob for different queries on three tasks Country-Capital, Antonym, and English-French (shown in black text). For comparison, we present
the _△_ logprob of the top tokens we get when decoding the FV via _D_ ( _vt_ ) (shown in red text directly
below). The _△_ logprob of the _D_ ( _vt_ )-tokens is much lower than the query-specific answers. In
general, the tokens promoted the most correspond to likely answers to the specific query, rather
than generic tokens in the output space. In the case of ‘wolf’ for the English-French task, the
model answers incorrectly. However, examining _△_ logprob indicates several likely answers are still
promoted – showing FVs have causal effects that are not adequately captured using top-1 accuracy.

**Tokens with largest positive** _△_ logprob **under FV intervention**

**Country-Capital**
South Africa ‘ Pret’ (+4.7), ‘ Johannes’ (+4.2), ‘ Dur’ (+4.0), ‘ Cape’ (+3.9), ‘ Kimber’ (+3.7)
‘ London’ (+1.3), ‘ Moscow’ (+1.1), ‘ Paris’, (+1.0), ‘ Bangkok’, (+0.3) ‘ Madrid’ (+0.2)

Syria ‘ Damascus’ (+4.9), ‘ Tart’ (+4.2), ‘ Raqqa’ (+4.1), ‘ Dam’ (+4.0), ‘ Aleppo’ (+3.8)
‘ London’ (+2.2), ‘ Moscow’ (+2.1), ‘ Paris’, (+2.1), ‘ Bangkok’, (+1.3) ‘ Madrid’ (+2.1)

**Antonym**
temporary ‘ perpetual’ (+4.4), ‘ definitive’ (+4.3),‘ everlasting’ (+4.1), ‘ permanent’ (+3.7)
‘ counterpart’ (+1.3), ‘ lesser’ (+0.9), ‘ destroy’ (+0.5), ‘ negate’ (+0.4), ‘ wrong’ (-0.8)

static ‘ evolving’ (+4.0), ‘ flexible’ (+3.8), ‘ polymorph’ (+3.7), ‘ dynamic’ (+3.7)
‘ counterpart’ (+0.4), ‘ lesser’ (-0.3), ‘ destroy’ (0.2), ‘ negate’ (-0.4), ‘ wrong’ (-1.7)

**English-French**
wolf ‘ lou’ (+6.3), ‘ ours’ (+6.2), ‘ chau’ (+5.8), ‘ d´e’ (+5.8), ‘ Lou’ (+5.7)
‘ ˆaK¸ [ˇ] l’ (-1.4), ‘ masc’ (-0.6), ‘ c¸¥l’ (-0.9), ‘ embr’ (+2.4), ‘ ˋe’ (+1.6)

advertisement ‘ ann’ (+7.6), ‘ aff’ (+6.8), ‘annon’ (+6.7), ‘ r´e’ (+6.2), ‘ pub’ (+6.14)
‘ ˆaK¸ [ˇ] l’ (0.2), ‘ masc’ (1.3), ‘ c¸¥l’ (-0.2), ‘ embr’ (-0.5), ‘ ˋe’ (+0.6)

For the country-capital task, the tokens with the highest increase in log probability typically correspond with likely answers to the specific query, rather than answers to the task in general. For
example, given the query ‘South Africa’, the country-capital FV promotes ‘Pretoria’, in addition
to other cities in South Africa. We compare the 5 tokens with the highest overall increase in log
probability, and the top 5 tokens we get from _D_ ( _vt_ ), which are shown below these in red. We see a
substantial difference between the magnitudes of the _△_ logprob for these tokens and the tokens that
were promoted the most.

We see a similar trend for Antonyms - where the promoted tokens are all reasonably valid antonyms
of the query word, rather than just antonyms in general.

For English-French, the query ‘ wolf’ is not answered correctly under intervention, but the correct
translation (‘ loup’) is still promoted by the FV when we examine the _△_ logprob. Similarly, for the
query ‘advertisement’, the dataset target is ‘publicit´e’, but prefix tokens for another valid french
translation (‘ annonce’) are also promoted when examining the _△_ logprob distribution - ‘ ann’, and
‘annon’.

In conclusion, for the tasks we examine we find that function vectors have strong causal effects even
when top-1 accuracy metric does not adequately capture this behavior. Furthermore, we find that the
causal effects of the FV do not just generically promote words in the output space, but specific words
that are plausible answers for each individual query.
