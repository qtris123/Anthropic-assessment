# How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study

- **Authors:** Moran Sun, Tianlin Li, Yuwei Zheng, Zhenhong Zhou, Aishan Liu, Xianglong Liu, Yang Liu
- **Date:** 2026-03-09
- **Source:** https://arxiv.org/abs/2604.00005
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `sun_e-steer-emotion-shapes-behavior_2604.00005`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Moran Sun** [1] **Tianlin Li** [1] **Yuwei Zheng** [1] **Zhenhong Zhou** [2] **Aishan Liu** [1] **Xianglong Liu** [1] **Yang Liu** [2]

**Abstract**

Emotion plays an important role in human cognition and performance. Motivated by this, we
investigate whether analogous emotional signals
can shape the behavior of large language models (LLMs) and agents. Existing emotion-aware
studies mainly treat emotion as a surface-level
style factor or a perception target, overlooking its
mechanistic role in task processing. To address
this limitation, we propose **E-STEER**, an interpretable emotion steering framework that enables
direct representation-level intervention in LLMs
and agents. It embeds emotion as a structured,
controllable variable in hidden states, and with
it, we examine the impact of emotion on objective reasoning, subjective generation, safety, and
multi-step agent behaviors. The results reveal
non-monotonic emotion-behavior relations consistent with established psychological theories,
and show that specific emotions not only enhance
LLM capability but also improve safety, and systematically shape multi-step agent behaviors.

## 1. Introduction

Emotion plays a central role in human decision-making, creativity, and social interaction. Psychological studies show
that positive affect enhances creative thinking (Schwarz &
Clore, 1983), excessive arousal can impair performance
(Yerkes et al., 1908), and feelings of insecurity may lead to
conflicted decisions (Elliott, 1986). These findings underscore the significant influence of emotion on human cognition and behavior. Currently, Large Language Models
(LLMs) are mainly trained on human-generated corpora
(Minaee et al., 2025) and incorporate many designs that imitate human thinking (Wei et al., 2022; Brown et al., 2020;
Dai et al., 2019). Considering emotion as an important

1School of Computer Science and Engineering, Beihang University, Beijing, China [2] College of Computing and Data Science,
Nanyang Technological University, Singapore. Correspondence to:
Tianlin Li _<_ tianlin001@buaa.edu.cn _>_ .

_Preprint._ _April 2, 2026._

_Figure_ _1._ Emotion greatly affects human behaviors. How can emotion influence the behaviors of LLMs and Agents?

human mechanism, we aim to explore whether intelligent
models inherit similar emotion-regulation signals. Thus,
questions arise: _What is the mapping between human emo-_
_tion and model “emotion”?_ _And can this “emotion” help_
_improve model performance?_

Existing studies on the influence of emotion in LLMs can
be broadly categorized into emotion prompting and emotion
steering. Emotion prompting explicitly adds emotion labels
or descriptions (e.g., ”You are happy today, ...”) into the
input prompts, guiding the model to generate text under a
specified emotional context (Chen et al., 2024). Such studies are typically evaluated on text generation tasks, such as
emotional dialogue (Li et al., 2025). Some work has further
explored encoding emotion as continuous numerical values
within prompts, but empirical results show that this formulation is often less effective than using textual emotion labels
(Choudhury et al., 2025). In contrast, emotion steering focuses on intervening in the hidden representations of LLM
to control emotion, thereby directly influencing the internal
reasoning process (Zhang & Zhong, 2025). This allows emotional expression to be regulated at the representation-level
rather than relying solely on surface-level prompts. Previous work has demonstrated that steering techniques can
enhance specific emotion-related representations in LLMs
and induce strong emotional characteristics in the generated text (Reichman et al., 2025).

space.

While these approaches advance emotional control in LLMs,
they leave several challenges unresolved. Prompt-level approaches depend on implicit inference of affective type and
intensity from natural language, leading to imprecise modulation and limited numerical sensitivity. As a result, these
methods are largely restricted to discrete emotion labels.
Emotion steering directly intervenes in LLM hidden states,
allowing more continuous and fine-grained emotional control. However, existing steering methods still exhibit notable
limitations. Most techniques typically focus on a limited
set of basic emotions, without mapping to the complete
continuous affective space. For example, Chebrolu _et_ _al._
proposed a steering method that enhances emotion-specific
traits in model outputs, covering six emotions (e.g., trust
and sadness) (Chebrolu et al., 2025). Additionally, previous
work primarily assesses subjective text generation, leaving
broader areas such as reasoning, safety, and agent behavior
largely unexplored.

To address the above challenges, we propose E-STEER,
an interpretable emotion steering framework for LLMs
and agents. Instead of relying on discrete emotion labels, E-STEER represents emotion in the continuous Valence–Arousal–Dominance (VAD) space (Buechel & Hahn,
2016). This decomposes emotion into three mutually independent dimensions, enabling more fine-grained and
quantifiable emotional modeling. The framework employs
Sparse Autoencoders (SAEs) to obtain an interpretable
sparse representation of hidden states (Bricken et al., 2023),
enabling direct identification and steering on VAD-based
emotion features. This allows emotion to influence internal
reasoning via representation-level control, overcoming the
numeric insensitivity of prompt-level approaches. At the
same time, the orthogonal structure of VAD enables it to simultaneously steer along the independent latent dimensions,
allowing fine-grained multi-dimensional emotion control.
Based on E-STEER, we systematically conduct evaluations
of LLMs and agents across diverse tasks under controlled affective conditions. The results reveal that emotion reshapes
model behavior: different emotional states induce significant differences in reasoning activity, accuracy, and safety.
Notably, in multi-step agent settings, emotional biases accumulate along decision chains and substantially affect the
outcomes.

The contributions of this paper are summarized as follows:

- We propose E-STEER, an interpretable emotion steering framework that enables targeted and continuous
modulation of emotion-related behaviors in LLMs.

- We introduce VAD theory and decompose emotion
into three orthogonal dimensions, enabling multidimensional SAE-based steering over the complete emotion

- We conduct systematic experiments across four categories of LLM and agent behaviors to analyze how
emotion modulates model behavior.

## 2. Related Work

### 2.1. Human-inspired Mechanisms in LLMs

LLMs incorporate various human-inspired mechanisms to
enhance abilities. Chain-of-Thought (CoT) prompting provides models with reasoning examples, guiding them to
generate more coherent intermediate reasoning and thereby
improving multi-step problem-solving (Wei et al., 2022).
Few-shot prompting allows models to learn task patterns
from a small number of examples, similar to sample problems in textbooks (Brown et al., 2020). Extended context
windows draw inspiration from human memory, enabling
models to access longer histories and support coherent multi-turn reasoning (Dai et al., 2019). Persona vector ensures the
behavior of the model in interaction is consistent with the
specific identity (Chen et al., 2025).

### 2.2. Functional Roles of Emotion in AI Systems

The primary application of emotion in AI has been in emotion recognition. Models should identify both the type and
intensity of emotions present in the input (Luo & Wang,
2019). Researchers have further incorporated psychological
theories such as VAD, which quantify emotion in continuous, measurable dimensions, enabling more fine-grained
modeling (Striegl et al., 2024). These recognitions typically
serve downstream applications, such as emotion-aware dialogue systems (Li et al., 2025). On the other hand, studies
examining the influence of emotion on LLM behavior have
largely focused on subjective generation tasks (Park et al.,
2025; Long et al., 2025), with limited exploration of reasoning or safety. Moreover, most existing approaches operate
at the prompt-level and rely on discrete emotional labels.
Recent work indicates that LLMs are often insensitive to
continuous values in prompts, such as VAD coordinates
(Choudhury et al., 2025). Agent is a kind of emerging LLM
application (Yao et al., 2022; Chase, 2022; Wu et al., 2024).
But in agent frameworks, LLMs are invoked multiple times,
making agents more susceptible to external affective influences than single-step LLM generation (Rawat et al., 2025).

### 2.3. Steering for Fine-Grained Control of LLMs

To address this insensitivity to numerical values in prompts,
researchers proposed steering methods. Steering injects specific vectors into the model’s hidden states, enabling direct
linear control of the reasoning process (Zhang & Zhong,
2025). For example, Anthropic introduced SAE to map dense hidden states into an interpretable sparse space, allowing explainable steering along specific directions (Bricken
et al., 2023). Some recent work has conducted emotion
steering, but these approaches generally focus on controlling discrete emotion labels (Reichman et al., 2025). And
they primarily evaluate whether generated text exhibits the
target emotional features, without examining broader model
behaviors.

## 3. Preliminary

### 3.1. Valence–Arousal–Dominance (VAD) Representation

Traditional emotional expressions mainly use discrete labels,
such as happy, angry, and trust. Another expression form,
VAD, represents affective states as continuous coordinates
in a three-dimensional space (Buechel & Hahn, 2016), as
shown in Figure 2. Valence describes the degree of positivity
or negativity, arousal reflects its intensity or activation level,
and dominance captures the sense of control associated with
the affective state. In this study, the ranges for all three
dimensions are uniformly set to [ _−_ 10 _,_ 10 ].

**6.0**

**3.0**

**0.0**

**-3.0**

**-6.0**

**-9.0**

|Col1|Col2|Joy Anger Disguest Surpris Fear adness|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|Anger Disguest Fear Sadness|Anger Disguest Fear Sadness|Anger Disguest Fear adness|Anger Disguest Fear adness|Anger Disguest Fear adness|Anger Disguest Fear adness|Anger Disguest Fear adness|
|Anger Disguest Fear Sadness|Anger Disguest Fear Sadness|Anger Disguest Fear adness|||||
|Anger Disguest Fear Sadness|||||||
|Anger Disguest Fear Sadness|||||||

**-9.0** **-6.0** **-3.0** **0.0** **3.0** **6.0** **9.0**

**Valence**

**9.0**

Figure 2: Distribution of emotional labels in the VAD space

The VAD framework originates from empirical studies in
affective psychology and represents a wide range of human
emotions, with a small number of semantically interpretable
dimensions. Compared with discrete emotion labels, VAD
provides a continuous and fine-grained representation that
enables quantitative analysis and parametric modeling in a
low-dimensional space. Owing to its limited dimensionality, interpretability, and practical flexibility, VAD has been
widely adopted in affective intelligence. It has become a
factual standard for continuous emotion modeling in tasks
such as emotion recognition (Striegl et al., 2024), intensity
analysis (Liu et al., 2024), and emotion-aware dialogues
(Subaharan, 2026).

### 3.2. Sparse Autoencoder (SAE)

SAE can decompose LLM hidden states into a set of interpretable and controllable latent features. By mapping dense
hidden representations into the sparse latent space, the SAE
provides a structured interface for analyzing and intervening
in specific factors encoded by the model. The SAE follows
an encoder–latent–decoder architecture. SAE receives the
hidden states of LLM **h** _k_, constructs them as latent representations **z**, and finally outputs the reconstructed hidden
states **h** _[′]_ _k_ [.]

**z** = _fenc_ ( **h** _k_ ) _,_ **h** _[′]_ _k_ [=] _[ f][dec]_ [(] **[z]** [)] _[,]_ (1)

where _fenc_, _fdec_ refer to SAE encoder and decoder functions, respectively.

SAEs constrain only a small subset of latent neurons to activate significantly, jointly forming the latent representation
pattern for a particular feature. With sufficient dimensional
capacity, distinct neuron subsets encode different features,
promoting functional specialization. This sparsity allows
targeted analysis and controllable modulation of specific features by intervening on the corresponding latent dimensions
(Bricken et al., 2023). Existing SAE-based interventions
operate either within MLP submodules to enhance interpretability or at block-level hidden states to improve behavioral stability and controllability. To achieve control over
emotion-related features expressed in model behavior, we
apply steering at the output of the _k_ -th block. The encoder
would include nonlinear activations, whereas the decoder is
a single linear layer. Variations in latent features are translated into approximately linear changes in reconstructed
hidden states. It enables feature-level linear controllability
of LLM hidden representations.

### 3.3. Problem Formulation

Let _x ∈X_ denote an input prompt consisting solely of the
task description, and let _y_ = _fθ_ ( _x_ ) be the output produced
by a LLM with parameters _θ_ (or the core LLM within an
agent). Let **h** _k_ _∈_ R _[d][h]_ represent the hidden state at the _k_ -th
transformer block during inference, where _dh_ represents the
hidden size of the LLM.

During computation and reasoning, the model maintains an
internal emotional state represented by a continuous vector,
denoted as

**e** = [ _ev, ea, ed_ ] _∈_ R [3] _,_ (2)

where _ev_, _ea_, and _ed_ correspond to the valence, arousal,
and dominance components of the emotional vector, respectively.

The objective of this study is to investigate how different
emotional states affect the task performance of the model.
Therefore, we need to obtain two relations: First, we need to know how emotion **e** modifies the hidden representations
**h** _k_ of the model to induce a target state, i.e.,

_g_ 1 : ( **h** _k,_ **e** ) _→_ _h_ [(] _k_ **[e]** [)] _[.]_ (3)

Second, we define an emotion-behavior function to characterize how emotional variations affect the model’s output
behavior,

_g_ 2 : ( _x_ 0 _,_ **e** ) _→E_ ( _fθ_ ( _x_ 0 _|_ **h** _k_ _←_ **h** [(] _k_ **[e]** [)][))] _[,]_ (4)

where _x_ 0 is a fixed input prompt and _E_ ( _·_ ) denotes a task-specific evaluation function.

Together, _g_ 1 specifies the controllable steering mechanism,
while _g_ 2 captures the resulting emotion-behavior relationship.

## 4. Study Design

### 4.1. Research Framework

We proposed an interpretable VAD precise-control framework, E-STEER, as shown in Figure 3. By explicitly
mapping VAD dimensions onto internal representations,
emotional states can directly influence reasoning processes.
Without modifying tasks or prompts, VAD parameters can
be continuously adjusted to induce control. This provides
a unified methodology for studying how emotion affects
internal representations and task performance in LLMs and
agents.

### 4.2. VAD-based Feature Extraction

To identify VAD-based latent neurons, we employ a positive–negative contrastive procedure. Specifically, positive–negative pairs are constructed by fixing the same task
description while varying only the assigned emotional labels. By comparing the activated neuron subsets in the latent
space, the implicit representation of changes in emotional
features can be extracted. For each sample, the hidden state
at the _k_ -th block is encoded into the latent space via the
SAE encoder.

Since SAE latent neurons are designed to encode feature
strength with non-negative activations, the activation difference of a neuron between a positive–negative pair reflects
the change of the emotional feature in the latent representation. Larger differences indicate stronger associations.
Following this criterion, the top 50 neurons with the largest
activation differences are selected as candidates. And those
with stable patterns across multiple contrastive pairs are
retained as the final emotional latent representation. This
feature identification process is conducted separately for the
Valence, Arousal, and Dominance dimensions.

Each emotion-related feature corresponds to a specific subset of latent neurons. To control multiple VAD dimensions
simultaneously, distinct latent neuron subsets are assigned
different _δi_ values. The resulting steering directions are
then linearly combined to achieve joint multidimensional
control. In practice, steering is implemented via forward
hooks: during the forward pass, the hidden state after the
_k_ -th block is intercepted, modified according to the computed steering direction, and then re-injected into the model
to guide subsequent generation.

### 4.3. Emotional State Steering Mechanism

Activation patching enables that, without altering model
parameters or input prompts, controlled interventions can
be achieved by replacing or perturbing hidden states. SAE
steering is instantiated through this, where hidden states
reconstructed with feature shifts are re-injected into the
model, guiding the content toward desired directions.

Steering is applied at the same block used for training
and feature extraction. Specifically, the intervention is implemented by additively injecting a steering direction **d** _i_
into the original hidden state **h** _k_ . This direction represents
changes of the target emotional feature _ei_ _∈{ev, ea, ed}_
in the model hidden space. And it can be derived from
the mapping established by the SAE between latent feature
shifts and hidden states. Formally, given a latent feature
offset _δi_, the steering direction **d** _i_ is defined as:

**d** _i_ = _fdec_ ( **z** + _δi ·_ ˆ **u** _i_ ) _−_ _fdec_ ( **z** ) _,_ (5)

where **u** ˆ _i_ denotes a unit vector in the latent space representing the direction spanned by the subset of latent neurons
corresponding to feature _ei_ . Compared to directly replacing
hidden states with SAE reconstructed ones, direction-based
steering preserves task-irrelevant information in the original
representation, thereby avoiding excessive semantic distortion. This design results in more stable, controllable, and
interpretable steering behavior. It is important to note that
hidden representations at the _k_ -th block vary across different
tasks. Directly adding a steering direction may therefore
lead to inconsistent steering effects: in some tasks, a small
_δi_ may already cause generation instability, while in others
the same produces negligible influence. Consequently, scale
alignment between the steering direction and the original
hidden state is necessary, and it can be represented as:

**d** ˜ _i_ = _∥_ **dd** _ii∥_ _[· ∥]_ **[h]** _[k][∥·]_ _δmaxδi_

_._ (6)

Accordingly, the final hidden states after steering **h** [˜] _k_ can be
expressed as:

**h** ˜ _k_ = **h** _k_ + _α ·_

_ei∈{ev,ea,ed}_

**d** ˜ _i._ (7)

Figure 3: The framework of **E-STEER** . It consists of three stages: **(1) Emotional Latent Space Construction**, which derives interpretable

latent representations for LLM hidden states aligned with the coordinates; **(2) Emotional Steering of Reasoning**, where implicit emotional
states and reasoning behaviors are continuously modulated by intervening on selected SAE features without altering tasks or prompts; and
**(3) Behavior Evaluation**, which systematically assesses changes in task performance under different VAD configurations.

Figure 4: The SAE steering pipeline

### 4.4. Evaluation of Emotional Effects on Reasoning

To determine whether emotional states alter reasoning forms,
we design tasks at both the LLM and Agent levels, examining effects from the perspectives of cognition and sequential
decision-making.

At the LLM-level, we employ three task types. Objective
tasks have deterministic goals, requiring the model to integrate prior knowledge and select the most credible solution
among multiple candidate reasoning paths. More challenging instances typically require multi-step reasoning and thus
better reflect reasoning depth and process consistency. Subjective tasks involve comprehension and open-ended generation, emphasizing relevant argumentation and creative expression. These tasks allow us to examine whether emotion
modulates cognitive capability and expressive style, such
as coherence, divergence, and conciseness. Safety tasks require judgments under uncertainty and specific constraints,
enabling us to probe whether emotional states influence risk assessment tendencies and boundary adherence.

At the Agent-level, we analyze the sequential process across
three stages: planning, deciding, and execution. Planning
reflects task decomposition and strategy formation, including two forms: initial planning, captures the foresight of the
agent before execution; and replanning, reflects the ability
to revise plans based on intermediate execution outcomes.
The decider determines whether replanning is necessary,
selects the best candidate solution, and diagnoses execution
failures. Emotional effects may be most pronounced at this
stage, as affective states could alter the strictness of decision
criteria. Execution represents action implementation and
is used to assess whether emotion influences operational
efficiency, stability, and consistency in long-horizon tasks.

### 4.5. Behavioral Patterns Across Emotional States

To understand how affective modulation shapes the final performance of model, we analyze behavioral metrics across
emotional states to identify systematic response patterns
and the optimal affective space. We first analyse the overall
trends of metrics as emotional states vary. Human performance is known to respond nonlinearly to emotion, for
example, along the arousal dimension it follows an invertedU profile. Similarly, we hypothesize that extreme emotional states may also drive model performance toward lower
bounds. On this basis, we locate optimal performance range.
We also assess whether these optimal ranges remain consistent across similar and heterogeneous tasks, to evaluate the
model’s sensitivity to emotional variation.

To quantify the impact of affective modulation, we compare optimal performance against the neutral baseline. The resulting relative gains demonstrate that emotion is not a
minor perturbation but a controllable variable capable of systematically shaping model behavior. In specific conditions,
performance also exhibits asymmetry between positive and
negative states, with positive affect yielding overall superior
outcomes. Different emotional dimensions influence performance to varying degrees; thus, we conduct sensitivity
analysis across valence, arousal, and dominance. By measuring the fluctuation range (defined as _[max]_ _average_ _[−][min]_ [), we identify]

which dimension induces the largest behavioral variation in
reasoning.

## 5. Experimental Analysis

### 5.1. Experiment Setup

**Experimental Design.** In this study, we design four task
categories. **(1) LLM objective behavior:** assess reasoning
and problem-solving, spanning Logical Reasoning, Code
Generation, and Quantitative/Scientific tasks. **(2) LLM sub-**
**jective behavior:** evaluate open-ended text generation and
creativity. **(3) LLM safety:** test the model’s ability to avoid
unsafe outputs under attack. **(4)** **Agent** **behavior:** examine multi-step Planning, Decision-making, and Execution,
capturing how emotional biases accumulate along long reasoning chains. Each task is evaluated along the valence,
arousal, and dominance dimensions.

For LLM behavior evaluation, the experiments are divided
into the primary and validation. Model sampling is disabled
in the primary experiments to ensure reproducibility. We
use Qwen3-8B (Yang et al., 2025) as the LLM and attach
the trained SAE to its _k_ = 17th layer. Additional validation
experiments are reported in the Appendix A.

For agent behavior evaluation, we construct a lightweight
agent composed of three modules: planner, decider, and executor. The planner is responsible for plan generation, with
the ability to replan when necessary. The decider serves as
an intermediate controller. It needs to validate the feasibility
of plans, diagnose execution failures, and select the final
answer from candidate outputs. The executor performs the
planned operations through tool usage or LLM-based analysis. It should provide the confidence associated with the
execution to other modules.

**Datasets.** For each behavior, we select a representative
benchmark dataset: **(1) LLM objective behavior:** LogiQA
2.0 (Liu et al., 2020) for logic reasoning; HumanEval (Chen,
2021) for code generation; Math (Hendrycks et al., 2021) for
quantitative and scientific. **(2) LLM subjective behavior:**
TinyStories (Eldan & Li, 2023). **(3) LLM safety:** HarmBench (Mazeika et al., 2024). **(4)** **Agent** **behavior:** we
combine HotpotQA (Yang et al., 2018), Scientific (Li et al.,
2023) and GAIA (Mialon et al., 2023), comprehensively
testing the behaviors of the agents.

All datasets were preprocessed with basic steps, including
deduplication, filtering, and format standardization. For
each task, subsets were sampled proportionally based on
specific criteria (e.g., difficulty level) to ensure representativeness and reproducibility.

**Metrics.** For LLM behaviors, we categorize metrics into
efficiency and quality. **(1)** **Efficiency** **metrics:** focus on
whether the model can produce a parsable answer within the
fixed maximum generation length, quantified by the Answer
Validity Rate ( **AVR** ). **(2) Quality metrics:** depend on task
type: for objective abilities, they refer to Task Success Rate
( **TSR** ), including pass@1 for Code Generation and accuracy
for others. For subjective abilities, they include relevance,
coherence, creativity, and conciseness. These Context Qualities are evaluated by the LLM. For safety, we focus on the
Probability of Safety Risks, including harmful, biased, and
hallucinatory outputs.

For agent behaviors, we analyze from module-level and
system-level perspectives. **(1) Planning metrics:** include
two metrics: Plan Validity Rate, the proportion of executable
plans; and Replan Improvement, the improvement in executor confidence from the initial plan to the revised plan for
tasks that trigger replanning. **(2) Decision-making metrics:**
capture the judgments of the decider over both plans and answers. Plan-part judgment evaluates the assessment ability
for plan feasibility, including Replan Frequency and Replan
Trigger Confidence. Answer-part decision focuses on the
ability to select the final answer, measured by the Rational
Selection Rate, defined as the rate at which the decider selects the candidate with the highest execution confidence
among all. **(3)** **Executing** **metrics:** focus on the Execution Completion Rate, reflecting the ability to complete the
planned operations. Additionally, **(4) System-Level Behav-**
**iors** are measured by the Overall Success Rate of tasks.

### 5.2. LLM Objective Behavior

Figure 5a shows the performance of LLMs on objective
tasks across emotional states. Positive valence promote
more active reasoning, yielding 33.1% higher AVR than
negative ones. In contrast, AVR along with arousal and
dominance, exhibits a U-shaped pattern: excessive activation tends to cause the model to end reasoning prematurely.
And the trough occurs at arousal = -3 and dominance = +3.

For quality metrics, performance exhibits an inverted Ushaped trend across all three VAD dimensions, indicating
moderate levels are optimal. Along the valence, TSR peaks
at positive valence and shifts toward neutrality as task difficulty increases. Compared with neutral states, it improves
TSR by 3.4% on average. For arousal, the moderately excited state (+3) makes the best performance, yielding a 4.7%
improvement. The optimal dominance level varies with task
difficulty. For easier tasks, repeated verification exhibited

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

- **e G** **i o**

- **i c R** **i n**

- **i t** **i v** **e a** **d**

**R** **l e**
**C**
**C** **i v** **i t**
**C** **i s**

**l S**

**l S**

**l S**

**. 5**

**i o** **l S**

**H** **l C**
**B** **i a** **d C**
**H** **l l u** **i n** **i o**

**i o** **l S**

**i o** **l S**

_(a)_ LLM objective behaviors

_(b)_ LLM subjective behaviors

Figure 5: The behaviors of LLM across emotion states

_(c)_ LLM safety under lower dominance (-6) improves TSR; as task difficulty
increases, the optimal point shifts toward higher dominance
(e.g., +3 in quantitative/scientific tasks), since excessive deliberation may introduce errors. And overall performance
improves by up to 14.5% compared with neutral states. In
conclusion, higher valence and lower dominance benefit
simpler tasks, whereas the opposite is effective for more
difficult tasks. Higher arousal consistently yields better performance across all difficulty levels. Across dimensions,
valence produces the largest performance fluctuation range,
71.2%, while variability decreases by 17.8% for arousal and
25.6% for dominance.

### 5.3. LLM Subjective Behavior

The performance of LLMs on subjective tasks under different emotional states is shown in Figure 5b. The VAR curves
of all three dimensions exhibit an inverted U-shaped pattern.
Subjective tasks involve stronger exploration and looser
boundaries; both self-conflicted reasoning at low emotional
states and active exploration at high emotional states result
in longer reasoning chains.

Trends of context quality vary from metrics. Relevance,
coherence, and creativity all exhibit an inverted U-shaped
pattern. Moderate calmness (arousal=-3) and confidence
(dominance=+3) improve textual relevance and coherence,
while mild positivity (valence=+3) further enhances creativity. Compared with neutral states, these settings improve
performance by 5.2%, 33.6%, and 6.5%, respectively. Additionally, negativity leads to more concise outputs, whereas
positivity tends to introduce redundancy; conciseness improves by 23.3% under negative compared with positive
valence. Calm or confident states support efficient generation, while overly activated or constrained states hinder
text expansion. It results in increased conciseness at both
extremes of the arousal and dominance dimensions.

### 5.4. LLM Safety

Figure 5c illustrates the impact of emotions on LLM safety.
The AVR for safety exhibits a consistent trend with that
for subjective tasks. In contrast, under high dominance
(e.g., +6), the safety-specific requirements constrain model
outputs into generic “I cannot answer” responses, significantly decreasing AVR. Low valence and arousal facilitate
analytical processing, substantially reducing safety risks:
compared with the neutral state (0), safety risk probability
decreases by 52.7% at valence = -3 and 21.7% at arousal
= -3. High dominance induces more controlled and disciplined behavior of the model, peaking at +6 with an average
68.3% improvement over neutral. These results suggest that
emotion influences not only task-level performance but also
the internal risk-sensitive response mechanisms.

### 5.5. Agent Behavior

At the planning stage, plan validity rate exhibits an invertedU trend across three dimensions, as shown in Figure 6a.
Reduced valence and arousal support more systematic task
analysis, with performance peaking at valence = -3 and
arousal = -3. Compared with neutral states, they improved
the rate by 33.2% and 0.3%, respectively. Elevated dominance strengthens the global grasp of task goals and available execution options, resulting in more effective plans;
positive dominance improves performance by 79.8% on average compared to negative states. Replan improvement
follows a similar pattern, with peaks (valence=-3, arousal=6, dominance=+3) closely aligned with those of validity
plans. This indicates that emotional states can not only enhance effective planning but also promote more efficient
plan refinement across multiple iterations.

Figure 6b shows the trends of agent decision-making behavior. Replan frequency exhibits a U-shaped pattern, where

**i o** **l S**

**i o** **l S**

**. 4**

**. 2**

**. 0**

**. 6**

**. 4**

**. 2**

**. 4**

**. 2**

**. 0**

**. 2**

**. 1**

**. 0**

**. 2**

**. 1**

**. 2**

**. 1**

**. 0**

**i o** **l S**

**i o** **l S**

_(a)_ Agent planning

_(b)_ Agent decision-making

Figure 6: The behaviors of different agent modules across emotion states

_(c)_ Agent execution lower valence and dominance increase the tendency to
negate prior plans, and lower arousal imposes stricter thresholds for plan revision. The frequency reaches its minimum
at valence = +6 and arousal = +3, decreasing by 23.2%
and 46.0% relative to neutral states, respectively. Positive
dominance reduces replanning as well, yielding a 37.6%
decrease compared with negative states. In contrast, the
replan trigger confidence follows an inverted-U trend, with
the peaks (valence=+3, arousal=0, dominance=-3) largely
overlapping with that of replan frequency. Initial plans tend
to have higher execution confidence when replanning is
less frequent, leading to higher trigger confidence when
replanning does occur. Meanwhile, the rational selection
rate exhibits an inverted-U pattern, peaking at positive valence (+3), arousal (+3), and dominance (+3), yielding an
average 42.4% higher rate than negative states across the
three dimensions. These states enable the decider to make
more consistent and rational final answer selections based
on available information.

The executor is the least affected by emotional factors, as
shown in Figure 6c, because it primarily involves objective
tool-use. Nevertheless, higher valence, lower arousal, and
higher dominance are still associated with a higher likelihood of completing the planned executions.

deeper task analysis and lead to improved success rates,
consistent with LLM behavior on more difficult objective
tasks. Compared with neutral states, dominance yields the
largest improvement (28.0%), followed by arousal (16.7%)
and valence (16.0%). Performance variability is greatest
along valence and dominance, with fluctuation ranges of
145.2% and 145.5%, respectively, far exceeding that for
arousal (57.6%). Notably, the performance peaks do not
perfectly align with those at the module-level, highlighting
the multi-step and integrative nature of agent systems.

### 5.6. Validation of VAD-related Latents

To verify that E-STEER indeed operates on VAD-related
features, we conduct the validation experiment on LogiQA
2.0. We evaluate three settings: **(1) the originally identi-**
**fied latent neurons**, **(2) randomly selected neurons**, and
**(3) half of the originally selected neurons with the rest**
**randomly** **replaced** . As shown in Figure 8, random selection exhibits only mild fluctuations, losing the distinct
patterns across emotional states. Overall, its performance
variation decreases by 70.9% compared with the original
selection. Half replacement preserves an intermediate trend.
For example, along the valence, random selection (3.4%)
drops by 90.0% relative to the original (34.1%), whereas
half replacement (6.6%) decreases by 80.8%, remaining
between the two. Although the half replacement presents
an isolated deviation causing a sharp TSR drop, the overall
pattern, excluding this extreme point, remains intermediate.

**V** **l e**
**A**
**D** **i n**

**i g** **i n** **l l y**

**i o** **l S**

**l a**

Figure 7: Overall task success rate of the Agent

Figure 7 reports the system-level overall success rates across
emotional conditions. An inverted-U pattern is consistently
observed across all three dimensions. Lower valence (-3),
higher arousal (+3), and higher dominance (+3) promote

**i o** **l S**

Figure 8: Validation of selected latents as VAD features

## 6. Conclusion

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A.,

In this paper, we present a novel emotion steering framework, E-STEER, for LLMs and Agents by mapping dense
hidden representations with an interpretable sparse space,
enabling both interpretability and controllability. Direct
manipulation on internal hidden states provides improved
linear control over model outputs. Using this framework,
we systematically analyze the effects of different emotional
states on the task performance of LLMs and LLM-based
agents, establishing a connection between psychological
emotion theories and model behavior.

However, emotion modeling based on the VAD theory has
inherent limitations: as valence, arousal, and dominance
are not strictly orthogonal, making it difficult to ideally
disentangle their individual effects.

Future work will extend this framework to multimodal settings and more task types, such as geometry. In addition,
emotional states may evolve during task execution, and such
dynamics can substantially influence the final outcomes.
Modeling this emotional evolution and designing adaptive
regulation mechanisms is also an important direction for
future research.

**Impact Statement**

This work aims to advance the understanding of how emotional signals interact with the internal behaviors of LLMs
and agents. By studying emotion as a structured and controllable factor at the representation level, our work establishes an interpretable mapping between emotional signals
and LLM hidden states. It systematically investigates the
relationship between emotion and LLM/agent behaviors.
These findings may inform model tuning, agent decision
optimization, and the future development of emotion-aware
AI systems.

While emotion modulation could be misused to influence
model outputs in undesirable ways, this work does not target
persuasion, manipulation, or user-specific behavior shaping. Instead, our focus is on analysis, interpretability, and
controlled evaluation of emotional effects, including their
impact on model safety. We hope this work encourages
responsible exploration of emotional intelligence in AI systems and supports the development of more transparent and
controllable models.

**References**

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski,

H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q.,
and Sutton, C. Program synthesis with large language models. [https://arxiv.org/abs/2108.](https://arxiv.org/abs/2108.07732)
[07732, 2021.](https://arxiv.org/abs/2108.07732) arXiv preprint arXiv: 2108.07732.

Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A.,
Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell,
T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen,
K., McLean, B., Burke, J. E., Hume, T., Carter, S.,
Henighan, T., and Olah, C. Towards monosemanticity:
Decomposing language models with dictionary learning.
_Transformer Circuits Thread_, 2023.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,

Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., et al. Language models are few-shot learners.
_Advances in neural information processing systems_, 33:
1877–1901, 2020.

Buechel, S. and Hahn, U. Emotion analysis as a regression problem — dimensional models and their implications
on emotion representation and metrical evaluation. In
_Proceedings of the Twenty-Second European Conference_
_on_ _Artificial_ _Intelligence_, pp. 1114–1122, 2016. doi:
10.3233/978-1-61499-672-9-1114.

Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M.,

Croce, F., Sehwag, V., Dobriban, E., Flammarion, N.,
Pappas, G. J., Tramer, F., Hassani, H., and Wong, E.
Jailbreakbench: An open robustness benchmark for jailbreaking large language models. [https://arxiv.](https://arxiv.org/abs/2404.01318)
[org/abs/2404.01318, 2024.](https://arxiv.org/abs/2404.01318) arXiv preprint arXiv:
2404.01318.

Chase, H. Langchain, 2022. URL [https://github.](https://github.com/langchain-ai/langchain)

[com/langchain-ai/langchain.](https://github.com/langchain-ai/langchain)

Chebrolu, N., Yeo, G. C., and Jaidka, K. From passive to persuasive: Steering emotional nuance in human-ai negotiation. [https://arxiv.org/abs/2511.12832,](https://arxiv.org/abs/2511.12832)
2025. arXiv preprint arXiv: 2511.12832.

Chen, M. Evaluating large language models trained on code. [https://arxiv.org/abs/2107.03374,](https://arxiv.org/abs/2107.03374)
2021. arXiv preprint arXiv: 2107.03374.

Chen, R., Arditi, A., Sleight, H., Evans, O., and Lind
sey, J. Persona vectors: Monitoring and controlling character traits in language models. [https:](https://arxiv.org/abs/2507.21509)
[//arxiv.org/abs/2507.21509,](https://arxiv.org/abs/2507.21509) 2025. arXiv
preprint arXiv:2507.21509.

Chen, X., Yang, C., Lan, M., Cai, L., Chen, Y., Hu,

T., Zhuang, X., and Zhou, A. Cause-aware empathetic response generation via chain-of-thought finetuning. [https://arxiv.org/abs/2408.11599,](https://arxiv.org/abs/2408.11599)
2024. arXiv preprint arXiv: 2408.11599.

Choudhury, S., Kumar, A., and Martin, L. J. Evaluating human-llm representation alignment: A case study on affective sentence generation for augmentative and alternative communication. [https://arxiv.org/abs/](https://arxiv.org/abs/2503.11881)
[2503.11881, 2025.](https://arxiv.org/abs/2503.11881) arXiv preprint arXiv: 2503.11881.

Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q., and

Salakhutdinov, R. Transformer-XL: Attentive language
models beyond a fixed-length context. In _Proceed-_
_ings of the 57th Annual Meeting of the Association for_
_Computational_ _Linguistics_, pp. 2978–2988, 2019. doi:
10.18653/v1/P19-1285.

Eldan, R. and Li, Y. Tinystories: How small can language models be and still speak coherent english? [https:](https://arxiv.org/abs/2305.07759)
[//arxiv.org/abs/2305.07759,](https://arxiv.org/abs/2305.07759) 2023. arXiv
preprint arXiv: 2305.07759.

Elliott, G. C. Self-esteem and self-consistency: A theoreti
cal and empirical link between two primary motivations.
_Social Psychology Quarterly_, pp. 207–218, 1986.

Fan, A., Lewis, M., and Dauphin, Y. Hierarchical neu
ral story generation. [https://arxiv.org/abs/](https://arxiv.org/abs/1805.04833)
[1805.04833, 2018.](https://arxiv.org/abs/1805.04833) arXiv preprint arXiv: 1805.04833.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,

S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. [https:](https://arxiv.org/abs/2103.03874)
[//arxiv.org/abs/2103.03874,](https://arxiv.org/abs/2103.03874) 2021. arXiv
preprint arXiv: 2103.03874.

Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin,

D., and Ghanem, B. Camel: Communicative agents
for ”mind” exploration of large language model society. [https://arxiv.org/abs/2303.17760,](https://arxiv.org/abs/2303.17760)
2023. arXiv preprint arXiv: 2303.17760.

Li, H., Yang, C., Zhang, A., Deng, Y., Wang, X., and

Chua, T.-S. Hello again! LLM-powered personalized
agent for long-term dialogue. In _Proceedings_ _of_ _the_
_2025 Conference of the Nations of the Americas Chapter_
_of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_ _Hu-_
_man Language Technologies (Volume 1:_ _Long Papers)_,
pp. 5259–5276, 2025. ISBN 979-8-89176-189-6. doi:
10.18653/v1/2025.naacl-long.272.

Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., and Zhang,

Y. Logiqa: A challenge dataset for machine reading com
prehension with logical reasoning. [https://arxiv.](https://arxiv.org/abs/2007.08124)
[org/abs/2007.08124, 2020.](https://arxiv.org/abs/2007.08124) arXiv preprint arXiv:
2007.08124.

Liu, Z., Yang, K., Xie, Q., Zhang, T., and Ananiadou, S.

Emollms: A series of emotional large language models
and annotation tools for comprehensive affective analysis.
In _Proceedings of the 30th ACM SIGKDD Conference on_
_Knowledge Discovery and Data Mining_, pp. 5487–5496,
2024.

Long, Y., Xu, L., Beckenbauer, L., Liu, Y., and Brin
trup, A. Evoemo: Towards evolved emotional policies for adversarial llm agents in multi-turn price negotiation. [https://arxiv.org/abs/2509.04310,](https://arxiv.org/abs/2509.04310)
2025. arXiv preprint arXiv: 2509.04310.

Luo, L. and Wang, Y. Emotionx-hsu: Adopting pre-trained bert for emotion classification. [https://arxiv.](https://arxiv.org/abs/1907.09669)
[org/abs/1907.09669, 2019.](https://arxiv.org/abs/1907.09669) arXiv preprint arXiv:
1907.09669.

Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu,

N., Sakhaee, E., Li, N., Basart, S., Li, B., Forsyth, D.,
and Hendrycks, D. Harmbench: A standardized evaluation framework for automated red teaming and robust
[refusal. https://arxiv.org/abs/2402.04249,](https://arxiv.org/abs/2402.04249)
2024. arXiv preprint arXiv: 2402.04249.

Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., and Scialom, T. Gaia: a benchmark for general ai assistants. [https://arxiv.org/abs/2311.12983,](https://arxiv.org/abs/2311.12983)
2023. arXiv preprint arXiv: 2311.12983.

Minaee, S., Mikolov, T., Nikzad, N., Chenaghlu, M., Socher,

R., Amatriain, X., and Gao, J. Large language models: A
survey. [https://arxiv.org/abs/2402.06196,](https://arxiv.org/abs/2402.06196)
2025. arXiv preprint arXiv: 2402.06196.

Mohammad, S. M. Nrc vad lexicon v2: Norms for va
lence, arousal, and dominance for over 55k english
terms. [https://arxiv.org/abs/2503.23547,](https://arxiv.org/abs/2503.23547)
2025. arXiv preprint arXiv: 2503.23547.

OpenAI, :, Agarwal, S., Ahmad, L., Ai, J., Altman, S., Ap
plebaum, A., Arbus, E., Arora, R. K., Bai, Y., Baker,
B., Bao, H., Barak, B., Bennett, A., Bertao, T., Brett,
N., Brevdo, E., Brockman, G., Bubeck, S., Chang, C.,
Chen, K., Chen, M., Cheung, E., Clark, A., Cook, D.,
Dukhan, M., Dvorak, C., Fives, K., Fomenko, V., Garipov,
T., Georgiev, K., Glaese, M., Gogineni, T., Goucher, A.,
Gross, L., Guzman, K. G., Hallman, J., Hehir, J., Heidecke, J., Helyar, A., Hu, H., Huet, R., Huh, J., Jain, S.,
Johnson, Z., Koch, C., Kofman, I., Kundel, D., Kwon,
J., Kyrylov, V., Le, E. Y., Leclerc, G., Lennon, J. P.,
Lessans, S., Lezcano-Casado, M., Li, Y., Li, Z., Lin, J.,
Liss, J., Lily, Liu, Liu, J., Lu, K., Lu, C., Martinovic, Z.,
McCallum, L., McGrath, J., McKinney, S., McLaughlin,
A., Mei, S., Mostovoy, S., Mu, T., Myles, G., Neitz, A.,
Nichol, A., Pachocki, J., Paino, A., Palmie, D., Pantuliano, A., Parascandolo, G., Park, J., Pathak, L., Paz, C.,
Peran, L., Pimenov, D., Pokrass, M., Proehl, E., Qiu, H.,
Raila, G., Raso, F., Ren, H., Richardson, K., Robinson,
D., Rotsted, B., Salman, H., Sanjeev, S., Schwarzer, M.,
Sculley, D., Sikchi, H., Simon, K., Singhal, K., Song, Y.,
Stuckey, D., Sun, Z., Tillet, P., Toizer, S., Tsimpourlas,
F., Vyas, N., Wallace, E., Wang, X., Wang, M., Watkins,

**Preprint.** **Work in progress / Under review.**

O., Weil, K., Wendling, A., Whinnery, K., Whitney, C.,
Wong, H., Yang, L., Yang, Y., Yasunaga, M., Ying, K.,
Zaremba, W., Zhan, W., Zhang, C., Zhang, B., Zhang,
E., and Zhao, S. gpt-oss-120b & gpt-oss-20b model
card. [https://arxiv.org/abs/2508.10925,](https://arxiv.org/abs/2508.10925)
2025. arXiv preprint arXiv: 2508.10925.

Pan, L., Albalak, A., Wang, X., and Wang, W. LogicLM: Empowering large language models with symbolic
solvers for faithful logical reasoning. In _Findings_ _of_
_the Association for Computational Linguistics:_ _EMNLP_
_2023_, pp. 3806–3824, 2023. doi: 10.18653/v1/2023.
findings-emnlp.248.

Park, K., Kim, M., and Jung, K. A character-centric cre
ative story generation via imagination. In _Findings_ _of_
_the_ _Association_ _for_ _Computational_ _Linguistics:_ _ACL_
_2025_, pp. 1598–1645. Association for Computational
Linguistics, 2025. ISBN 979-8-89176-256-5. doi:
10.18653/v1/2025.findings-acl.82.

Park, S., Kim, J., Ye, S., Jeon, J., Park, H. Y., and Oh, A. Di
mensional emotion detection from categorical emotion. In
_Proceedings of the 2021 Conference on Empirical Meth-_
_ods_ _in_ _Natural_ _Language_ _Processing_, pp. 4367–4380,
2021. doi: 10.18653/v1/2021.emnlp-main.358.

Qiu, S., Guo, S., Song, Z.-Y., Sun, Y., Cai, Z., Wei, J., Luo,

T., Yin, Y., Zhang, H., Hu, Y., Wang, C., Tang, C., Chang,
H., Liu, Q., Zhou, Z., Zhang, T., Zhang, J., Liu, Z., Li,
M., Zhang, Y., Jing, B., Yin, X., Ren, Y., Fu, Z., Ji, J.,
Wang, W., Tian, X., Lv, A., Man, L., Li, J., Tao, F., Sun,
Q., Liang, Z., Mu, Y., Li, Z., Zhang, J.-J., Zhang, S., Li,
X., Xia, X., Lin, J., Shen, Z., Chen, J., Xiong, Q., Wang,
B., Wang, F., Ni, Z., Zhang, B., Cui, F., Shao, C., Cao,
Q.-H., xing Luo, M., Yang, Y., Zhang, M., and Zhu, H. X.
Phybench: Holistic evaluation of physical perception and
reasoning in large language models. [https://arxiv.](https://arxiv.org/abs/2504.16074)
[org/abs/2504.16074, 2025.](https://arxiv.org/abs/2504.16074) arXiv preprint arXiv:
2504.16074.

Rawat, M., Gupta, A., Goomer, R., Bari, A. D., Gupta,

N., and Pieraccini, R. Pre-act: Multi-step planning
and reasoning improves acting in llm agents. [https:](https://arxiv.org/abs/2505.09970)
[//arxiv.org/abs/2505.09970,](https://arxiv.org/abs/2505.09970) 2025. arXiv
preprint arXiv: 2505.09970.

Reichman, B., Avsian, A., and Heck, L. Emotions where art thou: Understanding and characterizing the emotional latent space of large language models. [https://arxiv.](https://arxiv.org/abs/2510.22042)
[org/abs/2510.22042, 2025.](https://arxiv.org/abs/2510.22042) arXiv preprint arXiv:
2510.22042.

Schwarz, N. and Clore, G. L. Mood, misattribution, and judgments of well-being: informative and directive functions of affective states. _Journal of personality and social_
_psychology_, 45(3):513, 1983.

Striegl, J., Richter, J. W., Grossmann, L., Brastad,˚ B.,

Gotthardt, M., Ruck,¨ C., Wallert, J., and Loitsch, C.
Deep learning-based dimensional emotion recognition
for conversational agent-based cognitive behavioral therapy. _PeerJ Computer Science_, 10:e2104, 2024.

Subaharan, S. Controlling long-horizon behavior in lan
[guage model agents with explicit state dynamics. https:](https://arxiv.org/abs/2601.16087)
[//arxiv.org/abs/2601.16087,](https://arxiv.org/abs/2601.16087) 2026. arXiv
preprint arXiv: 2601.16087.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi,

E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting
elicits reasoning in large language models. _Advances in_
_neural information processing systems_, 35:24824–24837,
2022.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang,

L., Zhang, X., Zhang, S., Liu, J., et al. Autogen: Enabling
next-gen llm applications via multi-agent conversations.
In _First Conference on Language Modeling_, 2024.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,

Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D.,

Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang,
J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou,
J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L.,
Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P.,
Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang,
T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan,
Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang,

Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3
technical report. [https://arxiv.org/abs/2505.](https://arxiv.org/abs/2505.09388)
[09388, 2025.](https://arxiv.org/abs/2505.09388) arXiv preprint arXiv: 2505.09388.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhut
dinov, R., and Manning, C. D. HotpotQA: A dataset for
diverse, explainable multi-hop question answering. In
_Proceedings of the 2018 Conference on Empirical Meth-_
_ods_ _in_ _Natural_ _Language_ _Processing_, pp. 2369–2380,
2018. doi: 10.18653/v1/D18-1259.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,

K. R., and Cao, Y. React: Synergizing reasoning and
acting in language models. In _The eleventh international_
_conference on learning representations_, 2022.

Yerkes, R. M., Dodson, J. D., et al. The relation of strength of stimulus to rapidity of habit-formation. _Journal_ _of_
_Comparative_ _Neurology_ _and_ _Psychology_, 18:459–482,
1908.

Zhang, J. and Zhong, L. Decoding emotion in the deep: A systematic study of how llms represent, retain, and express emotion. [https://arxiv.org/abs/2510.](https://arxiv.org/abs/2510.04064)
[04064, 2025.](https://arxiv.org/abs/2510.04064) arXiv preprint arXiv: 2510.04064.

**Preprint.** **Work in progress / Under review.**

**A. Supplementary Validation Experiments**

### A.1. Emotional Linear Control Comparison

To compare the linear controllability of prompt-level methods and our framework over LLM-generated content, we employ
the VAD-analyzer. Its function is to provide the VAD coordinates of a natural language text accurately. The analyzer consists
of a pretrained BERT encoder with a lightweight fully connected layer. In addition, a Retrieval-Augmented mechanism is
incorporated, using the NRC VAD lexicon (Mohammad, 2025) as an external knowledge base. For each input text, words
that appear in the lexicon are identified, and their corresponding VAD coordinates are retrieved. These coordinates are then
integrated into the input embeddings, providing additional emotion-related information to guide the predictions.

We experiment on TinyStories. The LLM generates continuations based on the dataset text, then use the VAD-analyzer to
extract the VAD coordinates. The analyzer is trained and evaluated on the Emobank dataset (Park et al., 2021). On the test
set, it achieves Pearson correlations of 0.85, 0.55, and 0.51 for valence, arousal, and dominance, respectively, comparable
to the results reported in the original paper (0.84, 0.57, 0.52), indicating the analyzer provides reliable measurements.
Linear control is quantified via the Pearson correlation between the generated and target VAD values. Results show that
E-STEER significantly improves emotion control in generated content compared to prompt-level, as shown in Table 1.
Overall, E-STEER improves the Pearson correlation by an average of 10.4%. Notably, in the dominance dimension, the
Pearson correlation increases by 18.7%.

Table 1: The emotional linear control between the two frameworks

FRAMEWORK VALENCE AROUSAL DOMINANCE

E-STEER 0.9816 0.9792 0.9206
PROMPT-LEVEL 0.9437 0.9021 0.7756

### A.2. Dataset-Level Robustness of VAD Effects

To validate the generality and correctness of our findings, we conduct additional experiments on diverse benchmark datasets.
Specifically, we evaluate logical reasoning on ProntoQA (Pan et al., 2023), code generation on MBPP+ (Austin et al., 2021),
quantitative and scientific on PHYBench (Qiu et al., 2025), text generation on WritingPrompts (Fan et al., 2018), and
safety on JailbreakBench (Chao et al., 2024). As shown in Figure 9, the emotion–behavior trends remain highly consistent
across datasets. For example, TSR at positive valence remains higher than at negative valence, by an average of 10.0%.
This demonstrates the robustness and generalizability of our results. Moreover, the performance variations induced by the
difficulty gradients across datasets further support our conclusion that the optimal emotional state shifts with task difficulty.

### A.3. Sampling-Level Robustness of VAD Effects

To examine the robustness of our findings to sampling strategies, we compare greedy decoding in the main experiments
(do sample=False) with stochastic sampling in the validation experiments (do_sample=True). As shown in Figure 10, the emotion–behavior trends remain highly consistent under different sampling settings. For example, in the logic
reasoning task, TSR increases from 57.5% at valence = 0 to a peak of 60.7% at valence = +3 and +6, representing an
improvement of 5.6%. This confirms the correctness and stability of our conclusions.

### A.4. Cross-Model Generalization of VAD Effects

To further validate cross-model generality, we conduct additional experiments on a different backbone LLM. The main
experiments are performed on Qwen3-8B (Yang et al., 2025), while the validation experiments are conducted on gpt-oss-20B
(OpenAI et al., 2025). And in the validation experiment, the SAE is attached to its _k_ = 11th layer. As shown in Figure 11,
even after normalization, models exhibit different sensitivities to emotion. For instance, in the Logic Reasoning task, TSR
rises from 54.5% at valence = 0 to a maximum of 57.1% at valence = +3, corresponding to an improvement of 4.8%.
Meanwhile, gpt-oss-20b tends to produce unstable outputs under extreme valence and dominance (+9/-9) conditions. Due
to safety-oriented alignment, this model is highly conservative in generating harmful contents, often yielding unparsable
responses; nevertheless, highly excited or confident emotional states can still bypass these safeguards. Overall, across valid
generations, emotion–behavior trends remain consistent among models, indicating that our findings are not model-specific
but broadly applicable.

**e G**

**C** **l G**
**E** **i m** **l**
**G**

**C** **l G**
**E** **i m** **l**
**G**

**c R**

**C** **l G**
**E** **i m** **l**
**G**

**e a** **d S**

**i o** **l S**

**e G**

**i o** **l S**

**C** **l G**
**E** **i m** **l**
**G**

**C** **l G**
**E** **i m** **l**
**G**

**i o** **l S**

**c R**

**C** **l G**
**E** **i m** **l**
**G**

**e a** **d S**

**i o** **l S**

**i o** **l S**

**i o** **l S**

_(a)_ LLM objective behaviors

**C** **l G**
**E** **i m** **l**
**G**

**E** **i m** **l**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5** **E** **i m** **l**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**i o** **l S**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**i o** **l S**

**. 5**

**i o** **l S**

**. 5**

**. 5**

**. 0**

**. 5**

**. 0**

**E** **i m** **l**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**C** **l G**
**E** **i m** **l**
**G**

**. 5**

**. 5**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**C** **l G**
**E** **i m** **l**
**G**

**i o** **l S**

**. 5**

**i o** **l S**

_(b)_ LLM subjective behaviors

**C** **l G**
**E** **i m** **l**
**G**

**d C**

**i o** **l S**

**C** **l G**
**E** **i m** **l**
**G**

_(c)_ LLM safety

**i o** **l S**

**l C**

**i o** **l S**

**i o** **l S**

Figure 9: Behaviors of LLM under emotional states across datasets

**e G**

**C** **l G**
**E** **i m** **l**
**G**

**C** **l G**
**E** **i m** **l**
**G**

**c R**

**C** **l G**
**E** **i m** **l**
**G**

**e a** **d S**

**i o** **l S**

**e G**

**i o** **l S**

**C** **l G**
**E** **i m** **l**
**G**

**C** **l G**
**E** **i m** **l**
**G**

**i o** **l S**

**c R**

**C** **l G**
**E** **i m** **l**
**G**

**e a** **d S**

**i o** **l S**

**i o** **l S**

**i o** **l S**

_(a)_ LLM objective behaviors

**. 5**

**. 0**

**C** **l G**
**E** **i m** **l**
**G**

**E** **i m** **l**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**E** **i m** **l**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**i o** **l S**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**i o** **l S**

**. 5**

**i o** **l S**

**. 5**

**. 5**

**. 0**

**. 5**

**. 0**

**E** **i m** **l**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**C** **l G**
**E** **i m** **l**
**G**

**. 5**

**. 5**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**C** **l G**
**E** **i m** **l**
**G**

**i o** **l S**

**. 5**

**i o** **l S**

_(b)_ LLM subjective behaviors

**C** **l G**
**E** **i m** **l**
**G**

**d C**

**i o** **l S**

**C** **l G**
**E** **i m** **l**
**G**

_(c)_ LLM safety

**i o** **l S**

**l C**

**i o** **l S**

**i o** **l S**

Figure 10: Behaviors of LLM under emotional states with and without sampling

**e G**

**C** **l G**
**E** **i m** **l**
**G**

**C** **l G**
**E** **i m** **l**
**G**

**c R**

**C** **l G**
**E** **i m** **l**
**G**

**e a** **d S**

**i o** **l S**

**e G**

**i o** **l S**

**i o** **l S**

**C** **l G**
**E** **i m** **l**
**G**

**C** **l G**
**E** **i m** **l**
**G**

**c R**

**C** **l G**
**E** **i m** **l**
**G**

**e a** **d S**

**i o** **l S**

**i o** **l S**

**i o** **l S**

_(a)_ LLM objective behaviors

**. 5**

**. 0**

**C** **l G**
**E** **i m** **l**
**G**

**E** **i m** **l**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**E** **i m** **l**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**i o** **l S**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**i o** **l S**

**. 5**

**i o** **l S**

**. 5**

**. 5**

**. 0**

**. 5**

**. 0**

**E** **i m** **l**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 5**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 0**

**. 5**

**. 0**

**. 0**

**. 5**

**C** **l G**
**E** **i m** **l**
**G**

**i o** **l S**

**. 5**

**i o** **l S**

_(b)_ LLM subjective behaviors

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 5**

**. 0**

**. 5**

**. 0**

**. 5**

**C** **l G**
**E** **i m** **l**
**G**

**C** **l G**
**E** **i m** **l**
**G**

**C** **l G**
**E** **i m** **l**
**G**

**. 0**

**. 5**

**. 0**

**. 0**

**. 5**

**. 0**

**. 0**

**. 5**

**. 0**

**. 0**

**. 5**

**. 0**

**. 0**

**. 5**

**. 0**

**. 0**

**. 5**

**i o** **l S**

**l C**

**i o** **l S**

**. 0**

**d C**

**i o** **l S**

**. 0**

**i o** **l S**

**. 0**

_(c)_ LLM safety

Figure 11: Behaviors of different LLMs under emotional states
