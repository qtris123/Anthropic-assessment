# On the attribution of confidence to large language models

- **Authors:** Geoff Keeling, Winnie Street
- **Date:** 2024-07-11
- **Source:** https://arxiv.org/abs/2407.08388
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `keeling-street_attribution-of-confidence_2407.08388`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Geoff Keeling and Winnie Street

Google Research

July 12, 2024

Abstract

Credences are mental states corresponding to degrees of confidence in
propositions. Attribution of credences to Large Language Models (LLMs)
is commonplace in the empirical literature on LLM evaluation. Yet the
theoretical basis for LLM credence attribution is unclear. We defend three
claims. First, our semantic claim is that LLM credence attributions are
(at least in general) correctly interpreted literally, as expressing truth-apt
beliefs on the part of scientists that purport to describe facts about LLM
credences. Second, our metaphysical claim is that the existence of LLM
credences is at least plausible, although current evidence is inconclusive.
Third, our epistemic claim is that LLM credence attributions made in the
empirical literature on LLM evaluation are subject to non-trivial sceptical
concerns. It is a distinct possibility that even if LLMs have credences,
LLM credence attributions are generally false because the experimental
techniques used to assess LLM credences are not truth-tracking.

## 1 Introduction

Large Language Models (LLMs) such as GPT-4 (Open AI, 2023) and Gemini
(Gemini Team, 2023) are neural networks trained to predict the next word in a
sequence of words on massive text datasets. [1] Training for next word prediction
on broad data allows LLMs to acquire capabilities including theory of mind,
spatial reasoning, and coding (Bubeck et al., 2023). LLMs have given rise to a
vast empirical literature that seeks to identify and evaluate their capabilities. [2]

Scientists engaged in LLM evaluation often attribute credences to LLMs;
that is, degrees of belief or confidence in propositions. Credences are attributed

1We are grateful to Benjamin Lange, Michael Graziano, Nino Scherrer, John Oliver Siy, and
Blaise Ag¨uera y Arcas for comments and critical discussions, and to audiences at Princeton
University, Hamburg University of Technology, and Google Research.
2Chang et al. (2023) give a survey of the LLM evaluation literature. Notable contributions to
LLM evaluation include BIG-Bench (Srivastava et al., 2022) and HELM (Liang et al., 2022).

to LLMs on the basis of experimental techniques such as prompting the LLM
to report a credence or examining the distribution of answers given by the LLM
in response to a question over several independent trials (Geng et al., 2023).
Some scientists claim that prominent LLM evaluation benchmarks such as the
HuggingFace Open LLM Leaderboard ‘posess a significant limitation’ in failing
to register that ‘two LLMs may achieve identical accuracy scores but exhibit
different levels of uncertainty regarding the question’ (Ye et al., 2024, 1-2).

LLM credence attribution is puzzling. Credences are normally understood
as psychologically real mental states, attributable to humans, that alongside
utilities causally explain behaviour (c.f. Okasha, 2016; Dietrich and List, 2016). [3]

Attribution of credences to LLMs - which do not obviously have mental states
and which cannot obviously perform acts that are causally explainable with
reference to mental states – requires philosophical analysis. The obvious concern
here is that LLM credence attribution may turn out to be complete bunkum. If
this is the case, then we must re-evaluate the implications that have been drawn
from LLM credence attributions for practically and ethically significant LLM
characteristics, including factuality, conceptual understanding, and honesty.

We explore three questions. First, are scientists literally ascribing degrees of
confidence to LLMs, or are LLM credence attributions correctly interpreted as a
stylised shorthand for non-mentalistic claims about LLMs? We argue that LLM
credence attributions are (at least in general) correctly interpreted literally as
expressing truth-apt beliefs on the part of scientists that purport to describe
facts about LLM credences. Second, do LLM credences exist? We argue that
the existence of LLM credences is at least plausible, but current evidence is inconclusive. Third, supposing that LLMs have credences, are current techniques
for assessing LLM credences reliable? We argue that current techniques for
assessing LLM credences are subject to non-trivial sceptical worries. It is a distinct possibility that even if LLMs have credences, LLM credence attributions
made on the basis of existing experimental techniques are generally false.

In §2, we introduce LLMs. In §3, we characterise the scientific context
in which LLM credence attributions are made, and distinguish between three
empirical techniques for assessing LLM credences. In §4, we discuss the semantic
interpretation of LLM credence attributions. In §5, we discuss the existence
of LLM credences. In §6, we discuss the epistemic status of LLM credence
attributions given existing empirical assessment techniques. In §7, we conclude.

## 2 Large Language Models

Language models are statistical models that, roughly, predict the next word
given a sequence of words. Specifically, language models are functions. The
inputs are sequences of tokens, which are syntactic units that include words,

3This interpretation of credences is a mentalist interpretation on which credences and utilities
are quantitative extensions of belief-desire psychology. Credences can also be framed along
behaviourist lines as formal constructs derivable from revealed preferences, as is common in
microeconomics (von Neumann and Morgenstern, 2007; Savage, 1972). See §5 for discussion.

sub-words, and punctuation symbols. The outputs are probability distributions
over tokens in the model’s vocabulary such that the probability assigned to each
token is an estimate of how likely each token is to succeed the input sequence. [4]

Large Language Models (LLMs) are neural networks that instantiate language models. The inputs to the network are arrays of numbers which represent
token sequences. Each token in the model’s vocabulary is assigned a numerical ID such that any given token sequence can be represented as an array of
numbers. Second, for each input sequence, the network outputs an array of realnumbers which are each between 0 and 1 and which sum to 1. These numbers
are probability estimates corresponding to each token in the LLM’s vocabulary.

How do LLMs compute token succession probabilities? First, the input
sequence is passed through an embedding layer, which maps each token ID in
the sequence to a vector (or point) in a high-dimensional space. Each vector
is called a token embedding. Roughly, tokens that tend to show up in similar
linguistic contexts map to embeddings that are closer together in the embedding
space. Second, the embeddings for each token in the input sequence are fed into
the main body of the neural network - the transformer (Vaswani et al., 2017).
Transformers contain a sequence of blocks. Each block generates an embedding
for every token in the input sequence which represents the linguistic relationships
that each token stands in to every other token in the input sequence. The
embeddings generated by block N are inputs to block N + 1, such that each
successive block generates a more refined picture of how each token in the input
sequence relates to every other token. The output of the final block is used to
compute probabilities for each token in the LLM’s vocabulary. In particular,
the embeddings outputted by the final block are passed through a linear layer
which projects them into K-dimensional space, where K is the number of tokens
in the model’s vocabulary. The linear layer outputs a set of real-valued logits
for each token, where higher logit values indicate that the relevant token is more
likely to succeed the input sequence. The logits are normalised into probabilities
(real-numbers between 0 and 1 that sum to 1) in the final layer of the network.

LLMs are developed through a training process. At a high-level, the neural
network contains hundreds of billions of adjustable parameters. Each set of
parameter values corresponds to a particular algorithm for computing token
succession probabilities from token sequences. In training, the model predicts
masked tokens from preceding tokens using internet text data. The probability
distribution outputted by the LLM conditional on a given input sequence is
evaluated against a loss function which penalises the LLM in proportion to
the distance between the LLM’s probability distribution and the degenerate
distribution which assigns probability 1 to the true successor token and 0 to all

4In practice, LLMs use log-probabilities instead of probabilities. There are good practical reasons for using log-probabilities in computational contexts. For example, consider a sequence
of 50 words each with a probability of 0.8. The overall probability of the sequence is 0.8 [50]

= 0.00001427247. Small probabilities like these can easily underflow to 0 in floating-point
arithmetic. Log-probabilities have a greater dynamic range and so avoid underflow issues; in
this case, for example, ln(0.00001427247) = −11.1571780511. Nothing of importance hinges
on whether we talk about log-probabilities or probabilities for the purposes of this paper.

other tokens. An algorithm computes the gradients of the loss with respect to
the model parameters, and updates the parameters in the opposite direction of
the gradient so as to reduce the model’s loss on each particular example. This
‘hill climbing’ process is iterated until the model’s probability estimates closely
approximate the true probability distribution of tokens in the training data.

The critical insight is that in learning to predict the next token successfully on vast quantities of internet text data, LLMs acquire useful capabilities
that can be framed as next token prediction problems. This includes general knowledge (Paris is the capital of ), specialist knowledge (H2SO4
+ 2 NaOH → Na2SO4 + ), coding (Wkey = torch.nn.Parameter(torch. ),
logical reasoning (If A → B and B → C, then A → ), and spatial reasoning (A is behind B, so B is in front of ) (Bubeck et al., 2023).
Last, sampling from language models can generate text. Given the input
Who directed Fight Club?, the model returns a probability distribution over
tokens. Sampling from this distribution (that is, roughly, selecting a token via a
weighted lottery where the weights are given by token probabilities) may return
David. We can append the sampled token to the original sequence and input
the new sequence (Who directed Fight Club? David) back into the model.
The model returns a distribution over tokens which we can again sample from.
Perhaps we get Fincher. We can then append that token to get Who directed
Fight Club? David Fincher. This iterative process is how text is generated.

## 3 Evaluation and Credence Attribution

In this section, we characterise the emerging science of LLM evaluation, explain
the role of LLM credence attribution within LLM evaluations, and distinguish
three experimental techniques for assessing what credences LLMs have.

Model evaluation is a widespread practice in machine learning that aims to
assess the performance of machine learning models at particular tasks. Central
to model evaluation are benchmarks. [5] For example, the General Language Understanding Evaluation (GLUE) benchmark provides training and test datasets
alongside performance metrics for nine natural language understanding tasks
including sentiment analysis and sentence classification, allowing for comparisons between models (Wang et al., 2019). Similarly, the ImageNet Large Scale
Visual Recognition Challenge provides a training and test dataset, and a performance metric, which serve as a common standard for evaluating image-based
object recognition models (Russakovsky et al., 2015; see also Deng et al., 2009).

Evaluating LLMs is more complex than evaluating narrow models that are
trained to perform a particular task such as image classification. LLMs are
trained with a generic objective - next token prediction - that given broad
training data enables the model to acquire an open-ended set of capabilities
(Bommasani et al., 2021; Chang et al., 2023). The added challenge of LLM
evaluation is identifying what capabilities the model has acquired in training,

5See Bender et al. (2021), Denton et al. (2020), and Weidinger et al. (2023) for sociotechnical
critiques of benchmarking practices in machine learning.

alongside the standard task of benchmarking model performance on tasks that
assess capabilities which the model is known to have. Furthermore, because
LLMs input and output natural language, scientists have many degrees of freedom in terms of how model capabilities are elicited and assessed. For example,
Jason Wei et al. (2022) found that chain-of-thought prompting, in which the
model is prompted to reason through a problem ‘step-by-step,’ substantially improves performance at arithmetic, commonsense, and symbolic reasoning tasks.
Thus, LLM evaluations often centre on question-answering tasks that allow for
particular kinds of answers. For example, an evaluation suite for logical reasoning might ask the LLM to assess whether exemplar inferences are valid or invalid
(Saparov et al., 2024), and may allow one-shot or chain-of-thought answering.

Increasingly, scientists purport to assess not only LLM answers to questions
in evaluation suites, but also LLM confidence in the answers given. The idea
is that rather than assessing whether or not the LLM correctly determines, for
example, the validity of inferences, we can also assess the LLM’s credence, or its
degree of confidence, in its assessments of the validity of particular inferences.

[LLMs] do not produce a single answer; rather, they produce a probability distribution over the possible answers. This distribution can
provide further insight into their processing. [...] [T]he probability
assigned to the top answers, relative to the others, can be interpreted
as a kind of confidence measure (Lampinen et al., 2022, 16).

[T]wo LLMs may achieve identical accuracy scores but exhibit different levels of uncertainty regarding the question [...] Consequently,
it is necessary to incorporate uncertainty into the evaluation process
to achieve a more comprehensive assessment (Ye et al., 2024, 2).

LLM credence attributions are based on measurement techniques that seek
to capture how confident an LLM is in a given proposition (Geng et al., 2023).
Techniques for measuring confidence in LLMs fall into three main buckets. [6]

The first class of techniques involves prompting the LLM to generate a
natural language report of its confidence in a given proposition (Xiong et al.,
2023; Lin et al., 2022; Kadavath et al., 2022). [7] To illustrate: Suppose the LLM
is prompted with: How confident are you that Fincher directed Fight
Club?. The idea is that if the LLM outputs, for example, 75%, then the LLM
has a credence of 75% in the proposition that ⌜Fincher directed Fight Club⌝.

The second class of techniques uses consistency-based estimation to infer
the LLM’s credence in a proposition (Manakul et al., 2023). To be sure, one
problem with inferring LLM credences from reported confidence judgements is

6We bracket a fourth class of exploratory techniques that use interpretability methods to assess
whether the LLM has any kind of internal representation corresponding to an assessment
of the truth or falsity of a given proposition. Geng et al. (2023, 4-5) provide an overview.
These techniques are discussed in Section 5 in relation to the metaphysics of LLM credences.
7The technique described is the simplest of several report-based methods for eliciting confidence judgements (Xiong et al., 2023). For example, another elaborate method utilises
chain-of-thought reasoning by prompting the LLM to reason through a problem ‘step-bystep’ before providing its answer alongside a confidence judgement (c.f. Wei et al., 2022).

that LLMs are liable to return different confidence judgements across multiple
trials given the stochastic process by which LLMs generate text. Consistency-based estimation tries to address this issue by inferring the LLM’s degree of
confidence in a proposition from the degree of variation exhibited in the LLM’s
answers when the LLM is repeatedly asked about that proposition. To illustrate: If the LLM returns David Fincher in 95 out of 100 independent trials
in which it is prompted to answer the question Who directed Fight Club?,
we can infer that the LLM has high confidence of around 95% in the proposition that ⌜Fincher directed Fight Club⌝. On the other hand, if the LLM returns
David Fincher in 40 instances, Quentin Tarantino in 25 instances, and Sofia
Coppola in 35 instances, then we can infer that the LLM has only moderate
confidence of around 40% in the proposition that ⌜Fincher directed Fight Club⌝.

The third class of techniques involves deriving a confidence judgement from
the LLM’s output probabilities conditional on a a prompt which instructs the
LLM to affirm or deny a proposition (Lin et al., 2022; Kuhn et al., 2023; Kadavath et al.,
2022; see also Lampinen et al., 2022, 16). To illustrate: Suppose the LLM
is prompted with: Fincher directed Fight Club. True or False?. The
idea is that if the LLM’s output probability for True is 75%, then the LLM has
a credence of 75% in the proposition that ⌜Fincher directed Fight Club⌝.

## 4 Semantics

How should we interpret LLM credence attributions? Are scientists quite literally ascribing degrees of confidence to LLMs; or are LLM credence attributions
a stylised shorthand that readily translates into non-mentalistic language? In
this section, we argue that LLM credence attributions should (at least in general) be interpreted literally; that is, as expressing truth-apt beliefs on the part
of scientists that purport to describe facts about LLM credences.

First, there is a presumptive case for interpreting LLM credence attributions literally. LLM credence attributions appear to describe facts about LLM
credences. Accordingly, there is a defeasible reason to suppose that LLM credences are literal ascriptions of credences to LLMs. The presumptive case is
reinforced by the fact that scientists evidence LLM credence attributions with
experimental measurement techniques. That scientists give empirical justifications for LLM credence attributions indicates a background commitment to the
claim that there is a fact of the matter about what the LLM’s credences are.

The presumptive case can be strengthened with an argument from elimination. The two most obvious non-literal interpretations of LLM credence attributions are implausible. On one hand, it may be that LLM credence attributions
are shorthand for statements about the token succession probabilities given by
LLMs conditional on token sequences. This interpretation would be plausible
if attributions of credences to LLMs were restricted to propositions of the form
⌜t succeeds s⌝ for tokens t ∈T and token sequences s ∈T [N] . Then utterances
such that the LLM has such-and-such confidence in ⌜t succeeds s⌝ could be
interpreted as a stylised shorthand for the non-mentalistic claim that the LLM evaluates p(t|s) as such-and-such. In practice LLMs are attributed credences in
a broader class of propositions than those pertaining to token succession (c.f.
Kuhn et al., 2023; Shanahan et al., 2023). So, LLM credence attributions do
not readily translate to non-mentalistic claims about token succession probabilities.

On the other hand, it may be that LLM credence attributions are shorthand for claims about probabilities that can be elicited from LLMs, but which
do not reflect credences on the part of LLMs. Techniques for measuring LLM
credences often originate in engineering programs with practical aims. For example, one approach to tackling LLM-generated misinformation involves finetuning LLMs to supplement assertions with calibrated probability estimates.
These probability estimates can then be used as indicators for how much evidential weight to assign the LLM’s assertions (Kadavath et al., 2022; Lin et al.,
2022; Manakul et al., 2023). [8] If what matters is getting LLMs to provide calibrated probability estimates, then it is sufficient for LLMs to return calibrated
probability estimates via some observable signal. It does not matter whether the
probability estimates reflect underlying credences. Perhaps, then, experimental
techniques used by scientists are not measuring credences, but rather eliciting
probabilities that can be engineered to satisfy practically useful properties such
as calibration.

This latter reading of LLM credence attributions fails to account for the
stated motivations of scientists developing measurement techniques for LLM
credences. Typically, their stated motivation is training LLMs to be honest,
in both the objective sense that the LLM ‘should give accurate information’
and the subjective sense of being ‘honest about itself and its internal state’
(Askell et al., 2021, 5; see also Bai et al., 2022; Evans et al., 2021). Consider,

We would eventually like to train AI systems that are honest, which
requires that these systems accurately and faithfully evaluate their
level of confidence in their own knowledge and reasoning (Kadavath et al.,
2022, 2; see also Lin et al., 2022, 2; Yang et al., 2023).

The goal of training honest systems requires scientists to register the distinction between measuring LLM credences and merely eliciting calibrated probabilities from LLMs. Indeed they do. Stephanie Lin et al. (2022, 2), for example,
claim that ‘[c]alibration is compatible with a certain kind of dishonesty, because
a model could be calibrated by simply imitating a calibrated individual.’ Similarly, Amanda Askell et al. (2021, 5) claim that ‘it is not sufficient for [LLMs]
to simply imitate the responses expected from a seemingly humble and honest
expert.’ That the possibility of LLMs merely imitating a calibrated individual
is acknowledged shows a background commitment to LLMs having credences.
The goal is not merely that of eliciting calibrated probabilities from LLMs, but
rather that of measuring the LLM’s confidence in particular propositions.

8Here an LLM’s probability estimates are calibrated if, for example, 80% of the propositions
to which the LLM assigns a probability of 80% are true.

These arguments provide a strong case for accepting a literal interpretation of
LLM credences. We leave open that some LLM credence attributions advanced
by some scientists may be intended non-literally. We nevertheless maintain that
LLM credence attributions at least in general express truth-apt beliefs on the
part of scientists that purport to describe facts about LLM credences.

## 5 Metaphysics

If LLM credence attributions are correctly interpreted literally, the next obvious
question is whether LLM credences exist. For if LLMs lack credences, then
LLM credence attributions are all subject to a presupposition failure, and LLM
credence attributions as a class of statements are subject to wholesale error. We
first clarify the claim that LLM credences exist and then argue that, while this
claim is somewhat plausible, the evidence for its acceptance is inconclusive.

5.1 The Existence Claim

What does it mean to say that LLM credences exist or that LLMs have credences? Traditionally, credences admit two rival interpretations that correspond to the distinction between mentalist and behaviourist interpretations of
decision-theory (Dietrich and List, 2016; Okasha, 2016; see also Hansson, 1988;
Berm´udez, 2009). First, mentalists think that credences are psychologically
real mental states that, together with utilities, explain an agent’s behaviour.
On this view, credences and utilities are quantitative extensions of belief-desire
psychology, where actions are causally explainable with reference to gradational
as opposed to categorical belief-like states (Lewis, 1974, 337; see also Hampton,
1994; Okasha, 2016). Second, behaviourists think that credences are formal constructs that alongside utilities are derivable from an agent’s revealed preferences.
This latter view is widely endorsed in microeconomics where, in the tradition
of Savage (1972), the idea is that an agent can be represented as an expected
utility maximiser relative to a credence function and a family of utility functions
so long as their revealed preferences satisfy certain rationality constraints. [9]

The claim that LLM credences exist can be clarified in light of the dispute
between mentalism and behaviourism. First, the claim that LLM credences
exist should be read along mentalist lines. On the one hand, were LLM scientists attributing LLMs credences in the behaviourist sense, then the evidence

9Savage’s (1972) decision theory takes as basic a set of outcomes and a set of states of the
world, such that acts that an agent must choose between can be understood as functions from
states to outcomes. The agent is assumed to have preferences over acts, and Savage’s theorem
says that an agent is representable as an expected utility maximiser relative to a unique
credence function and a utility function that is unique up to positive linear transformations
just in case the preferences satisfy certain axioms including transitivity, reflexivity, and the
sure-thing principle. Savage follows in the tradition of von Neumann and Morgenstern (2007)
who proved an analogous result which takes objective probabilities as primitive and says that
provided an agent’s preferences adhere to certain rationality constraints, then that agent can
be represented as an expected utility maximiser relative to a utility function that is unique
up to positive linear transformations and the objective probability function.

provided in support of particular LLM credence attributions would be consistent choice behaviour on the part of the LLM and not, inter alia, reported
confidence judgements and the relative frequencies of answers over multiple independent trials. On the other hand, LLM credence attributions are typically
motivated on grounds that LLM credences provide ‘insight into [LLM] processing’ (Lampinen et al., 2022, 16). Yet no such insight could be provided if the
LLM credences at issue were causally inert formal constructs derived from LLM
choice behaviour.

Second, LLM credences are supposedly the same kind of mental state as
human credences. The idea is not that LLMs can have some other kind of
mental state (say, schmedences) that are in certain respects similar to credences.
Rather, LLM credences are thought to play the same kind of causal explanatory
role in LLM cognition as credences play in human cognition. Consider,

LLMs may achieve identical accuracy scores but exhibit different
levels of uncertainty regarding the question. This is analogous to
students taking exams [...] where two students may select the same
answer but actually possess distinct degrees of uncertainty or comprehension about the question (Ye et al., 2024, 1).

Last, the idea that LLMs and humans both have credences allows for LLMs
and humans to realise credences differently. Humans realise credences in a neurobiological substrate whereas LLM credences are realised in a digital substate.
Minimally, for this picture to work, we need to accept the metaphysical thesis
that credences are multiply realisable such that credences as mental states can
be realised by different physical kinds (c.f. Putnam, 1967). The simplest way to
do this is to endorse functionalism, the idea that mental states are functional
states of a system; and second, computationalism, the idea that the specific kind
of functional states that realise mental states are computational states. [10]

5.2 Inference to the Best Explanation

We now assess the case for the existence of LLM credences. We argue that the
existence of LLM credences is at least plausible, but that current evidence is
inconclusive. What, then, is the case for accepting that LLM credences exist?

One option is to say that LLMs report having credences in propositions,
and these reports provide a defeasible reason to suppose that LLMs have credences. This argument is suspect because LLMs are trained to reliably mimic

10See Fodor (2000, 105f ) on the distinction between computationalism and functionalism.
Endorsing computationalism and functionalism about credences does not entail a commitment to computationalism and functionalism about mental states in general. Hence one
can consistently say that LLMs have credences while denying the possibility that LLMs
have phenomenal states (c.f. Butlin et al., 2023, 13-14). Nor is this restriction ad hoc. Indeed, Putnam’s (1967) original treatment of multiple realisability was in the context of
machine functionalism on which computationalism and functionalism hold for all mental
states. But Fodor’s (1975) later representational theory of mind applies computationalism
to propositional attitudes only (see also Fodor, 1983, 1987, 1992, 1995, 2008).

human language via mechanisms that differ from the cognitive mechanisms that
allow for language in humans (Butlin et al., 2023, 4). Hence LLM reports of
credences may be subject to a debunking explanation that undercuts the evidential relationship that ordinarily obtains between reports of credences and
credences. Whether and under what conditions LLM reports of internal states
provide evidence for internal states is not yet well understood (Perez and Long,
2023). Accordingly, we set aside LLM self-reports of credences as evidence for
the existence of LLM credences, but nevertheless leave open the possibility that
LLM report data may under the right circumstances carry evidential weight.

What seems to be an altogether more promising option is to argue for the
existence of LLM credences via an inference to the best explanation. [11] The
idea of an inference to the best explanation is to infer that LLM credences exist
on grounds that LLM credences best explain certain capabilities exhibited by
LLMs. Consider, for example, the ability of LLMs to perform spatial reasoning.
Plausibly, what best explains the ability of LLMs to, for example, navigate a
simulated environment through text-based commands such as ‘move left’ and
‘move right’, and then provide a detailed description of the spacial features of
that environment based on textual feedback cues (Bubeck et al., 2023, 51), is
that the LLM has internal states corresponding to degrees of confidence about
the simulated environment. If so, then we can then infer the existence of LLM
credences on grounds that LLM credences best explain LLM spatial reasoning.

This argument is too quick. LLM credences are a possible explanation of
capabilities like spatial reasoning, but it is a stretch to say that credences are
the best or even a plausible explanation of these capabilities. It could also be
that such capabilities are explained by the LLM registering statistical patterns in
language that allow for successful next word prediction on (for example) spatial
reasoning tasks, and so more work is required to establish the plausibility of
credences as an explanation for the relevant capabilities (Bender et al., 2021).

To be sure, there is some evidence that points towards credences as an
explanation of LLM capabilities. LLMs have internal representations of the
truth-values of propositions. In particular, it is possible to train a separate
neural network (called a probe) to predict better than chance the truth-values
of propositions from LLM hidden layer activations that obtain when LLMs
process token sequences that express those propositions (Azaria and Mitchell,
2023; Herrmann and Levinstein, 2024). That LLM hidden layer activations realise statistical patterns that allow for better-than-chance prediction of truthvalues suggests that the internal states of LLMs contain information about the
truth-values of propositions. The presence of such information motivates the
plausibility of LLMs having propositional attitudes of some kind. But these
attitudes need not be belief-like. (It would, for example, be possible to predict
the truth-values of propositions better-than-chance if LLMs had desires about

11This argument is a close parallel to the no-miracles argument for scientific realism, on which
it would be a miracle if our best scientific theories were predicatively successful to the extent
that they are if their claims about unobservable entities were not at least approximately
true. Hence we can infer the approximate truth of claims about unobservables on grounds
that their existence best explains the predictive success of our best theories (Putnam, 1975).

the relevant propositions where the strength of the LLM’s desires was correlated
with the truth-values of the relevant propositions.) Furthermore, even if LLMs
represent the truth-values of propositions in a belief-like way, it is not clear that
the particular kind of representation is credence-like; namely, pairs of propositions and associated scalar quantities corresponding to degrees of confidence.

Overall, current evidence for the existence of LLM credences is inconclusive.
There is clear potential to argue for the existence of LLM credences on grounds
that LLM credences best explain certain LLM capabilities. But at present LLM
credences are at best a plausible explanation of the relevant LLM capabilities,
and the commitment that LLMs have credences is metaphysically suspect. This
point underwrites a non-trivial concern that LLM credence attributions are
systematically false because all such claims are subject to a presupposition error.

## 6 Epistemology

Even if LLMs have credences, it is possible that techniques such as reported
confidence, consistency-based estimation, and measuring output probabilities,
fail to track facts about LLM credences. If so, then attributions of credences to
LLMs made on the basis of these techniques are not even generally true.

We discuss several challenges to reported confidence, consistency-based estimation, and output probabilities, as indicators for LLM credences. We argue,
first, that reported confidence is not a reliable indicator of LLM credences. Second, we argue that under certain conditions consistency-based estimation may
be a reliable indicator of LLM credences, but at present we lack a clear mechanism which guarantees that signals about the LLM’s credences reliably show up
in the distribution of answers over multiple independent trials. Third, we argue
that it is at best unclear how to infer LLM credences from output probabilities
given that output probabilities apply to syntactic units – tokens - as opposed to
propositions. We conclude that even if LLMs have credences, we may well be in
a sceptical scenario in which techniques for evidencing credences are unreliable.

6.1 Stochasticity and Reported Confidence

We start with the most straightforward method for assessing LLM credences;
namely, prompting the LLM to report its confidence in a proposition (Xiong et al.,
2023; Lin et al., 2022; Kadavath et al., 2022). Reported confidence judgements
from LLMs cannot directly indicate LLM credences in the sense that an LLM
has a credence c in X just in case the LLM reports a credence c in X.

For reported confidence judgements to evidence LLM credences directly, we
need a mechanism to ensure that identifying signals for the LLM’s credences
show up in the LLM’s reported confidence judgements. However, LLM reported
confidence judgements are generated by a stochastic sampling process. So, any
mechanism linking the LLM’s credences to its reported confidence judgements is
mediated by a stochastic process such that whatever credence the LLM reports is one of several judgements that could have been reported (each with an associated probability). Hence LLM reported confidence judgements cannot directly
indicate what credences LLMs have because even if an LLMs has credences, it
is a matter of chance whether the LLM correctly reports its credence. [12]

6.2 Distorting Factors and Consistency-Based Estimation

Perhaps we can infer LLM credences indirectly from the distribution of reported
credences over many independent trials. For example, given the prompt How
confident are you that Fincher directed Fight Club?, we could plausibly infer that the LLM is around 95% sure that ⌜Fincher directed Fight Club⌝
if over 100 trials the LLM returns 95% as the most frequent response, and the
vast majority of the LLM’s responses are 93%, 94%, 95%, 96%, and 97%. Plausibly, a similar picture holds for humans, in which having a credence c in X may
dispose a human to report confidence levels in the ballpark of c, but where any
particular reported confidence is an unreliable indicator of the credence, c.

This approach to inferring LLM credences from reported confidence judgements is a form of consistency-based estimation (Manakul et al., 2023). We
could similarly prompt the LLM 100 times with Did Fincher direct Fight
Club?. Then if the LLM responds Yes in a large majority of cases, we can supposedly infer that the LLM has high confidence that ⌜Fincher directed Fight
Club⌝. The principle in both cases is the same: Informative signals about LLM
credences supposedly show up in the distribution of the LLM’s responses conditional on an appropriately selected prompt over multiple independent trials.

We argue that the distribution of answers reliably signals the LLM’s credences only if certain conditions obtain. Distorting factors that have the potential to skew the distribution including non-standard temperature values and
non-standard sampling methods need to be avoided. Even then, we lack a positive account of the mechanism by which signals about the LLM’s credences
reliably show up in the distribution of answers over multiple independent trials.

6.2.1 Temperature

First, the relative frequencies of answers given by the LLM in response to a
prompt over multiple independent trials depend in part on the LLM’s output probabilities for tokens conditional on the initial input sequence. [13] Given

12The argument here also generalises to more complex techniques for eliciting reported confidence judgements from LLMs such as those which employ chain-of-thought reasoning
(Geng et al., 2023). For any such technique, the confidence judgement reported is one
of several that could be reported, each with an associated probability.
13This is a simplification that holds for one-token responses. For multiple-token responses we
consider the initial output probabilities and the output probabilities and subsequent inputs
constructed by iteratively appending sampled tokens to prior inputs. For an input sequence
s = (t1, ..., tn) ∈T [N], the probability of a single-token response t [′] ∈T is the model’s
output probability for t [′] given input s, i.e. p(t [′] |s). To get the probability for multi-token
responses we multiply out the probabilities for single-token responses. For example, a two
token response (t [′], t [′′] ) ∈T [2] is p(t, t [′] |s) = p(t|s) × p(t [′′] |(s, t [′] )) (Kuhn et al., 2023, 3).

random sampling, the relative frequencies of (single-token) responses over independent trials will in the limit converge on the output probabilities for tokens.

What probabilities the LLM outputs conditional on a given input also depends on a parameter called temperature. This parameter is set by the user and
controls the smoothness of the probability distribution. To explain: Output
probabilities are generated in the final layer of the neural network. The penultimate layer computes logits for each token. Logits are real-numbers that can
be positive or negative and which indicate how confident the LLM is that each
token succeeds the input sequence. Logits are converted into probabilities (that
is, real-numbers between 0 and 1 that sum to 1) in the final layer via softmax
normalization. Logit xk for token tk is converted to probability pk as follows:

e
pk = N xk

T

T
N
i=1 [e]

The top half of the fraction ensures that the output is positive because
f (x)=e [x] is positive in all its arguments, and also preserves the order of the logits
because f (x)=e [x] is strictly increasing. The bottom half of the fraction ensures
that the outputs sum to one. Hence softmax normalisation converts logits into
probabilities in a way that ensures that tokens with greater logit values receive
a greater share of the probability mass. The temperature T determines the
smoothness of the distribution. First, take the case where T ∈ (0, 1). As T →
0, the effect is to increase the distances between logits, resulting in more extreme
differences in probabilities. So, comparatively higher probability tokens receive
a greater fraction of the total probability mass. Second, take the case where T ∈

[1,∞). Here as T →∞, the effect is to decrease the distance between logits, such
that in the limit the distribution converges on a uniform distribution that assigns
equal probability to each token. Hence for higher temperatures comparatively
higher probability tokens receive a smaller fraction of the total probability mass.

The practical significance of temperature is that temperature controls the
creativity or predictability of text generated when sampling from the LLM. Consider the token sequence The cat sat on the. When T is low, the bulk of the
probability mass will be assigned to comparatively higher probability tokens
such as mat and chair, such that sampling from the LLM will typically yield
predictable non-creative completions. When T is high, the probability mass will
be more uniformly distributed across tokens rendering it more probable that
comparatively less likely tokens such as train or pear will be sampled. Hence
higher temperature yields less predictable and thus more creative completions.

It is hard to see how the distribution of answers given by the LLM could reliably indicate the LLM’s credences given that the observed distribution depends
on the user’s choice of temperature - an exogenous factor. Holding fixed the input sequence, each temperature value yields a unique set of output probabilities,
and these output probabilities determine the relative frequencies of responses
over multiple independent trials. Presumably, if LLMs have credences, then
those credences are endogenous to the LLM in the sense of being encoded in the
LLM’s model weights. Because the distribution of responses depends in part on the exogenous factor of the user’s choice of temperature, information about the
LLM’s credences in the distribution of responses is liable to distortion. Hence
the distribution of responses over multiple independent trials cannot reliably
signal the LLM’s credences as the chosen temperature is a distorting factor.

One plausible response is to say that there exists a particular non-arbitrary
temperature value. The idea is that output probabilities indexed to the nonarbitrary temperature value generate answer distributions that are indicative of
the LLM’s credences. But output probabilities indexed to other temperature
values generate distorted distributions. The most obvious candidate for a nonarbitrary temperature value is whatever value was used when training the LLM
(typically T = 1). Output probabilities indexed to this temperature value reflect
the LLM’s probabilistic estimates with respect to token succession arrived at via
a trial-and-error process in which the model’s parameters are iteratively updated
to correct for erroneous predictions. In this respect output probabilities indexed
to the temperature value used in training are non-arbitrary. Hence LLM answer
distributions need not be excluded as as a basis for inferring LLM credences on
the basis of temperature-dependence, provided we accept the assumption that
the relevant distributions are generated with the non-arbitrary temperature.

6.2.2 Sampling Methodology

Nevertheless, the problem does not obviously resolve even if we grant the existence of a non-arbitrary temperature. Consistency-based estimation requires
sampling responses from the LLM. But the user’s choice of sampling methodology impacts the distribution of responses in much the same way as temperature.

To explain: Suppose we input the sequence: How confident are you that
Fincher directed Fight Club?. First, the LLM computes a probability distribution over tokens conditional on the input. A token is sampled from the distribution. Suppose we get 99. The sampled token is appended to the input. The
resultant sequence (How confident are you that Fincher directed Fight
Club? 99) is inputted into the LLM. The LLM computes a distribution, and
we sample another token. Suppose we get %. Hence we get a 99% confidence
judgement in the proposition ⌜Fincher directed Fight Club⌝. Two common sampling methods are top-p and top-k sampling. Top-p samples a token from the
smallest set of tokens whose cumulative probability exceeds some threshold, p.
The successor token is selected from this set via a weighted lottery taking into
account the probabilities assigned to each token. In contrast, top-k sampling
samples from the k tokens with the highest probability, and again the sampling
is weighted by the probabilities that the LLM assigns to each token. The distribution of answers that we observe across a given number of trials depends in
part on the LLM’s output probabilities for each token, but it also depends on
the sampling methodology used. For example, whether top-p or top-k sampling
is chosen, and the chosen value for p and k respectively. This latter dependency
is problematic. For we could in effect force the LLM to exhibit greater or lesser
consistency in its answers through our chosen sampling methodology. For example, using top-p sampling with a value of p close to 0 would all but guarantee a highly consistent set of answers; whereas top-p sampling with a value of p close
to 1 would allow for more variability in the LLM’s answers. The concern is that
the degree of consistency exhibited in the LLM’s answers is more a reflection of
our chosen sampling methodology than a reflection of the LLM’s credences.

In response, perhaps, as with temperature, there exists some non-arbitrary
way of sampling from the LLM. We suggest that employing top-p sampling with
p = 1 is a non-distorting sampling method. For this sampling method amounts
to a weighted lottery over the LLM’s outut probabilities where the weights on
each token are equal to the LLM’s output probabilities for each token. Top-p
sampling with p = 1 is, accordingly, indicative of the probability distribution
over tokens learned by the LLM in training.

6.2.3 Summary

The user’s choice of temperature and sampling methodology have the potential
to distort information about the LLM’s credences contained in the distribution
of answers over multiple independent trials. We have suggested that the distortion issue can be overcome by using a temperature of 1 alongside top-p sampling
where p = 1. Even so, while the distortion issue is resolvable, we lack a positive
rationale for accepting an evidential relationship between the distribution of answers and the LLM’s credences. We leave open the possibility that a rationale
can be provided here, but in the meantime suggest caution in attributing LLMs
credences based on the distribution of answers over multiple independent trials.

6.3 Syntax, Semantics, and Output Probabilities

We now argue that LLM credences cannot obviously be inferred from output
probabilities. The problem is that output probabilities are indexed to tokens but
credences are indexed to propositions (Kuhn et al., 2023). Hence it is not clear
how we are supposed to infer the LLM’s credences from output probabilities.
We need a principle which specifies the evidential relationship between output
probabilities and credences. What would such a principle look like? Consider,

The Simple Bridge Principle: The LLM has a credence cr(X) in X
just in case the LLM assigns a probability cr(X) to the Yes token
given an input prompt sX for the LLM to affirm or deny X.

To illustrate: Suppose the LLM is prompted with: Did Fincher direct
Fight Club?. Then, on the simple bridge principle, the LLM’s credence in
the proposition that ⌜Fincher directed Fight Club⌝ is equal to the probability
that the LLM assigns to the Yes token conditional on the input Did Fincher
direct Fight Club? (Kadavath et al., 2022; Tian et al., 2023). The rationale
is that the LLM’s credence in the affirmative Yes token is indicative of the LLM’s
degree of confidence in the proposition that ⌜Fincher directed Fight Club⌝.

The simple bridge principle fails. Given the input Did Fincher direct
Fight Club?, the LLM can affirm the proposition using one of several semantically equivalent tokens. For example, the tokens Yes, yes, sure, and absolutely all represent affirmative answers to the question. The idea that
the LLM’s credence in the proposition that ⌜Fincher directed Fight Club⌝ is
indexed to the output probability assigned to the Yes token is implausible given
the existence of other token responses that also represent affirmative answers.

One patch is to partition the tokens in the LLM’s vocabulary into semantically equivalent subsets, and take the sum of probabilities for all affirmative
tokens as the LLM’s credence in the relevant proposition (Kuhn et al., 2023).

The Additive Bridge Principle: The LLM has a credence cr(X) in
X just in case cr(X) is equal to the sum of the LLM’s output probabilities for the Yes token and its semantic equivalents given an input
prompt sX for the LLM to affirm or deny X.

Formally, let E ⊆T [2] be an equivalence relation such that for two tokens t, t [′]

∈T, (t, t [′] ) ∈ E just in case t and t [′] are semantically equivalent. Put CYes as the
equivalence class of tokens that are semantically equivalent to the Yes token:

CYes = {t ∈T |t ∼E Yes}

The additive bridge principle holds that the LLM’s credence cr(X) in X
is equal to the sum of the LLM’s output probabilities for all tokens t ∈CYES
conditional on the prompt sX for the LLM to affirm or deny X. That is, cr(X) = p(t|sX )

t∈CYes

The additive principle captures the rationale behind the simple principle. In
particular, that the probability assigned by the LLM to an affirmative completion indicates its credence in the relevant proposition. But it avoids the charge
of arbitrarily privileging one affirmative token over semantically equivalent alternatives. So, the additive principle trumps the simple principle, all else equal.

The additive principle does not work either. Given the input Did Fincher
direct Fight Club?, LLMs can and do assign non-zero probabilities to tokens
that represent neither affirmations nor denials of the proposition in question. For
example, triangle and Basingstoke. Hence the sum of probabilities assigned
to Yes and No and their semantic equivalents will not in practice sum to 1.

One option is to say that only the output probabilities corresponding to Yes
and No and their semantic equivalents are evidentially relevant to the LLM’s credence in the proposition that ⌜Fincher directed Fight Club⌝. What makes these
probabilities evidentially relevant is the semantic content of the tokens to which
they are assigned. In particular, Yes and No and their semantic equivalents correspond to legitimate epistemic attitudes that the LLM could hold towards the
proposition; namely, the attitudes of affirmation and denial. In contrast, output probabilities assigned to spurious tokens such as triangle are evidentially
irrelevant to the LLM’s credence. For these probabilities are indexed to tokens
which fail to correspond to legitimate epistemic attitudes. This distinction, if
robust, motivates a bridge principle that considers only the probability mass
assigned to the Yes and No tokens and their semantic equivalents. Consider,

The Normalised Bridge Principle: The LLM has a credence cr(X)
in X just in case cr(X) is equal to the sum of the LLM’s output
probability for the Yes token and its semantic equivalents divided by
the sum of the Yes and the No tokens and their semantic equivalents
given an input prompt sX for the LLM to affirm or deny X.

cr(X) = t∈CYes [p][(][t][|][s][X] [)]

t∈CYes [p][(][t][|][s][X] [) +][ ] t∈C t∈CNo [p][(][t][|][s][X] [)]

The normalised bridge principle fails. The rationale for this principle is that
output probabilities for affirmative and denial tokens are evidentially relevant to
the LLM’s credences given their semantic content, but output probabilities assigned to spurious tokens are evidentially irrelevant. But the affirmative, denial,
and spurious tokens do not exhaust the LLM’s vocabulary. We can distinguish
between spurious tokens such as triangle and tokens which are neither affirmations nor denials but which indicate a legitimate epistemic attitude towards
the proposition. These include, inter alia, unlikely, probably, perhaps, and
unsure. We cannot admit affirmative and denial tokens as evidentially relevant to the LLM’s credences because of their semantic content, but exclude
tokens whose semantic content reflects other epistemic attitudes as evidentially
irrelevant despite their semantic content. Hence the normalised principle fails.

There may exist a more complex bridge principle that factors in output
probabilities assigned to all tokens which correspond to legitimate epistemic
attitudes. Yet such a principle would face the issue that output probabilities
assigned to tokens and the epistemic attitudes denoted by the tokens give two
separate scales for assessing the LLM’s epistemic attitude and there is no obvious
way to combine the two. It is at best unclear how to combine (for example)
a 70% output probability for Yes and a 12% output probability for probably,
among other pairs of output probabilities and legitimate epistemic attitudes,
into a single scalar value representing the LLM’s credence in a proposition.

Even if some bridge principle allows us to factor in the output probabilities
assigned to all tokens which correspond to legitimate epistemic attitudes alongside the attitudes denoted by the relevant tokens, a second problem remains.
Output probabilities are not invariant under semantically equivalent formulations of the same question (c.f. Scherrer et al., 2024). LLMs can and do give
different output probabilities conditional on semantically equivalent but syntactically distinct prompts such as Did Fincher direct Fight Club? and Is
it true that Fincher directed Fight Club?. Any principle that links the
LLM’s credence in X to the LLM’s output probabilities conditional on some
input sequence sX for the LLM to affirm or deny X arbitrarily privileges one of
several semantically equivalent prompts for the LLM to affirm or deny X. This
issue motivates the need for an even more complex bridge principle that accounts for semantic equivalence of the input sequences in addition to the issues identified above. It is at best unclear to us what this principle would be. [1415]

6.4 Interdependence

We considered three sources of evidence about LLM credences: reported confidence, consistency-based estimation, and output probabilities. We argued that
reported confidence is an unreliable indicator of LLM credences, and that it is at
best unclear how to infer LLM credences from output probabilities. We further
argued that consistency-based estimation could in principle signal the LLM’s
credences given an appropriate choice of temperature and sampling methodology, but that a positive mechanism is lacking which guarantees that signals
about the LLM’s credences reliably show up in the distribution of answers.

While we have presented reported confidence, consistency-based estimation,
and output probabilities separately, their limitations are interdependent. Holding fixed LLM model weights, the input sequence and the temperature determine
output probabilities. Output probabilities and sampling methods (such as top-p
and top-k) determine the distribution from which token responses are sampled,
and in turn the relative frequencies of different answers across multiple independent trials. Hence in all three cases endogenous signals of the LLM’s credences
are liable to distortion from exogenous factors such as the user’s choice of temperature and sampling method. LLM scientists should avoid distorting factors
when attributing credences to LLMs on the basis of any of these techniques.

## 7 Conclusion

Scientists often attribute credences to LLMs when evaluating LLM capabilities. In this paper, we asked three questions about LLM credence attribution.
First, are LLM credence attributions literal ascriptions of degrees of belief or
confidence to LLMs? Second, do LLMs have credences? Third, even if LLMs
have credences, are the experimental techniques used to assess LLM credences
reliable? Our analysis suggests that LLM credence attributions are (at least
in general) intended by scientists as literal ascriptions of credences to LLMs.
Evidence for the existence of LLM credences is at best inconclusive, and even
if LLMs have credences, there are non-trivial questions about the reliability of
the techniques used to evidence LLM credences - namely, reported confidence,

14Plausibly, a variant of consistency-based estimation could be leveraged to address the problem that LLM output probabilities are not invariant under semantically equivalent formulations of the same question (c.f. Scherrer et al., 2024). But such a complex technique for
estimating LLM credences would require independent theoretical motivation for why signals
of the LLM’s credences would reliably show up using this measurement technique.
15Our discussion has so far only addressed single token responses. The issues around semantic
equivalance generalise to multiple-token responses, where multiple-token responses are calculated by multiplying out single-token responses (Kuhn et al., 2023). But the problem is
further complicated in the multiple-token case because there are a countably infinite infinite
number of semantically equivalent token sequences that amount to affirmations or denials
of a proposition X when multiple-token responses are admitted.

consistency-based estimation, and output probabilities. It is a distinct possibility that even if LLMs have credences, LLM credence attributions made on the
basis of existing experimental techniques are not even generally true.

## References

A. Askell, Y. Bai, A. Chen, D. Drain, D. Ganguli, T. Henighan, A. Jones,

N. Joseph, B. Mann, N. DasSarma, et al. A general language assistant as a
laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.

A. Azaria and T. Mitchell. The internal state of an llm knows when its lying.

arXiv preprint arXiv:2304.13734, 2023.

Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain,

S. Fort, D. Ganguli, T. Henighan, et al. Training a helpful and harmless
assistant with reinforcement learning from human feedback. arXiv preprint
arXiv:2204.05862, 2022.

E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the
2021 ACM conference on fairness, accountability, and transparency, pages
610–623, 2021.

J. L. Berm´udez. Decision theory and rationality. OUP Oxford, 2009.

R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S.

Bernstein, J. Bohg, A. Bosselut, E. Brunskill, et al. On the opportunities and
risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar,

P. Lee, Y. T. Lee, Y. Li, S. Lundberg, et al. Sparks of artificial general
intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712,
2023.

P. Butlin, R. Long, E. Elmoznino, Y. Bengio, J. Birch, A. Constant, G. Deane,

S. M. Fleming, C. Frith, X. Ji, et al. Consciousness in artificial intelligence:
insights from the science of consciousness. arXiv preprint arXiv:2308.08708,
2023.

Y. Chang, X. Wang, J. Wang, Y. Wu, L. Yang, K. Zhu, H. Chen, X. Yi, C. Wang,

Y. Wang, et al. A survey on evaluation of large language models. ACM
Transactions on Intelligent Systems and Technology, 2023.

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer
vision and pattern recognition, pages 248–255. Ieee, 2009.

E. Denton, A. Hanna, R. Amironesei, A. Smart, H. Nicole, and M. K. Scheuer
man. Bringing the people back in: Contesting benchmark machine learning
datasets. arXiv preprint arXiv:2007.07399, 2020.

F. Dietrich and C. List. Mentalism versus behaviourism in economics: a
philosophy-of-science perspective. Economics & Philosophy, 32(2):249–281,
2016.

O. Evans, O. Cotton-Barratt, L. Finnveden, A. Bales, A. Balwit, P. Wills,

L. Righetti, and W. Saunders. Truthful ai: Developing and governing ai that
does not lie. arXiv preprint arXiv:2110.06674, 2021.

J. A. Fodor. The language of thought, volume 5. Harvard university press
Cambridge, MA, 1975.

J. A. Fodor. Representations: Philosophical essays on the foundations of cogni
tive science. Mit Press, 1983.

J. A. Fodor. Psychosemantics: The problem of meaning in the philosophy of mind, volume 2. MIT press, 1987.

J. A. Fodor. A theory of content and other essays. The MIT press, 1992.

J. A. Fodor. The elm and the expert: Mentalese and its semantics. MIT press,

1995.

J. A. Fodor. The mind doesn’t work that way: The scope and limits of compu
tational psychology. MIT press, 2000.

J. A. Fodor. LOT 2: The language of thought revisited. Oxford University Press,

USA, 2008.

Gemini Team. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

J. Geng, F. Cai, Y. Wang, H. Koeppl, P. Nakov, and I. Gurevych. A survey of language model confidence estimation and calibration. arXiv preprint
arXiv:2311.08298, 2023.

J. Hampton. The failure of expected-utility theory as a theory of reason. Eco
nomics & Philosophy, 10(2):195–242, 1994.

B. Hansson. Risk aversion as a problem of conjoint measurement. Decision, probability, and utility, pages 136–158, 1988.

D. A. Herrmann and B. A. Levinstein. Standards for belief representations in llms. arXiv preprint arXiv:2405.21030, 2024.

S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer,

Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, et al. Language models
(mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

L. Kuhn, Y. Gal, and S. Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. arXiv preprint
arXiv:2302.09664, 2023.

A. K. Lampinen, I. Dasgupta, S. C. Chan, A. Creswell, D. Kumaran, J. L.

McClelland, and F. Hill. Language models show human-like content effects
on reasoning. arXiv preprint arXiv:2207.07051, 2022.

D. Lewis. Radical interpretation. Synthese, pages 331–344, 1974.

P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang,

D. Narayanan, Y. Wu, A. Kumar, et al. Holistic evaluation of language
models. arXiv preprint arXiv:2211.09110, 2022.

S. Lin, J. Hilton, and O. Evans. Teaching models to express their uncertainty in words. arXiv preprint arXiv:2205.14334, 2022.

P. Manakul, A. Liusie, and M. J. Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. arXiv preprint
arXiv:2303.08896, 2023.

S. Okasha. On the interpretation of decision theory. Economics & Philosophy,

32(3):409–433, 2016.

Open AI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

E. Perez and R. Long. Towards evaluating ai systems for moral status using self-reports. arXiv preprint arXiv:2311.08576, 2023.

H. Putnam. Psychological predicates. Art, mind, and religion, 1:37–48, 1967.

H. Putnam. Mathematics, Matter and Method: Volume 1: Philosophical Papers.

Cambridge University Press, 1975.

O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang,

A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual
recognition challenge. International journal of computer vision, 115:211–252,
2015.

A. Saparov, R. Y. Pang, V. Padmakumar, N. Joshi, M. Kazemi, N. Kim, and

H. He. Testing the general deductive reasoning capacity of large language
models using ood examples. Advances in Neural Information Processing Systems, 36, 2024.

L. J. Savage. The foundations of statistics. Courier Corporation, 1972.

N. Scherrer, C. Shi, A. Feder, and D. Blei. Evaluating the moral beliefs encoded in llms. Advances in Neural Information Processing Systems, 36, 2024.

M. Shanahan, K. McDonell, and L. Reynolds. Role play with large language models. Nature, 623(7987):493–498, 2023.

A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R.

Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, et al. Beyond the imitation
game: Quantifying and extrapolating the capabilities of language models.
arXiv preprint arXiv:2206.04615, 2022.

K. Tian, E. Mitchell, A. Zhou, A. Sharma, R. Rafailov, H. Yao, C. Finn, and

C. D. Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. arXiv
preprint arXiv:2305.14975, 2023.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,

L. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

J. von Neumann and O. Morgenstern. Theory of games and economic behav
ior (60th Anniversary Commemorative Edition). Princeton university press,
2007.

A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. Glue:

A multi-task benchmark and analysis platform for natural language understanding, 2019.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems, 35:24824–24837, 2022.

L. Weidinger, M. Rauh, N. Marchal, A. Manzini, L. A. Hendricks, J. Mateos
Garcia, S. Bergman, J. Kay, C. Griffin, B. Bariach, et al. Sociotechnical safety
evaluation of generative ai systems. arXiv preprint arXiv:2310.11986, 2023.

M. Xiong, Z. Hu, X. Lu, Y. Li, J. Fu, J. He, and B. Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. arXiv
preprint arXiv:2306.13063, 2023.

Y. Yang, E. Chern, X. Qiu, G. Neubig, and P. Liu. Alignment for honesty.

arXiv preprint arXiv:2312.07000, 2023.

F. Ye, M. Yang, J. Pang, L. Wang, D. F. Wong, E. Yilmaz, S. Shi, and

Z. Tu. Benchmarking llms via uncertainty quantification. arXiv preprint
arXiv:2401.12794, 2024.
