# A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents

- **Authors:** Raghu Arghal, Fade Chen, Niall Dalton, Evgenii Kortukov, Calum McNamara, Angelos Nalmpantis, Moksh Nirvaan, Gabriele Sarti, Mario Giulianelli
- **Date:** 2026-02-09
- **Source:** https://arxiv.org/abs/2602.08964
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `arghal_goal-directedness-lm-agents_2602.08964`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Raghu Arghal** [1 *] **Fade Chen** [2 *] **Niall Dalton** [3 *] **Evgenii Kortukov** [4 *] **Calum McNamara** [5 *]

**Angelos Nalmpantis** [6 *] **Moksh Nirvaan** [7 *] **Gabriele Sarti** [8] **Mario Giulianelli** [3]

**Abstract**

Understanding an agent’s goals helps explain and
predict its behaviour, yet there is no established
methodology for reliably attributing goals to agentic systems. We propose a framework for evaluating goal-directedness that integrates behavioural
evaluation with interpretability-based analyses of
models’ internal representations. As a case study,
we examine an LLM agent navigating a 2D grid
world towards a goal state. Behaviourally, we
evaluate the agent against optimal policies across
varying grid sizes, obstacle densities, and goal
structures, finding that performance scales with
task difficulty while remaining robust to difficultypreserving transformations and multi-goal structures. We then use probing methods to decode
internal representations of the environment and
multi-step action plans. We find that the LLM
agent non-linearly encodes a coarse spatial map,
preserving approximate task-relevant cues about
its position and the goal location; that its actions
are broadly consistent with these internal representations; and that reasoning reorganises them,
shifting from spatial cues towards immediate action selection. Our findings support the view that
introspective examination is required beyond behavioural evaluations to characterise how agents
represent and pursue their objectives.

## 1. Introduction

Attributing goals to agents helps explain and predict their
behaviour and provides a useful abstraction for reasoning
about agency. Goal attribution has been studied across a

_∗_ Alphabetical order; see Statement of Author Contributions.
1University of Pennsylvania 2New York University 3University College London [4] Fraunhofer HHI [5] Indiana University, Bloomington
6TKH AI 7Independent 8Northeastern University. Correspondence
to: [Mario Giulianelli <m.giulianelli@ucl.ac.uk>.](mailto:m.giulianelli@ucl.ac.uk)

_Proceedings_ _of_ _the_ _43_ _[rd]_ _International_ _Conference_ _on_ _Machine_
_Learning_, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).

|…|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|…|L|ayer 7|ayer 7|ayer 7|
||L|ayer 15|||
|…|||||
|…|||||
||L|ayer 23|ayer 23|ayer 23|
||||||
||||||
||||||

Figure 1: Overview of our goal-directedness analysis. We prompt

an LLM-based agent to reason and act over a fully observable grid
world. Behaviourally, we evaluate whether its trajectories agree or
disagree with the optimal policy. **A:** Iso-difficulty transforms test
whether the agent’s behaviour is invariant to changes in grid configuration that preserve task difficulty. **B:** Multi-goal environments
test how agents handle instrumental and implicit goal structures.
**C:** We extract pre- and post-reasoning activations from intermediate layers, probe the agent’s beliefs over goal distance, planned
actions, and reconstruct _cognitive maps_ for the current grid state.

wide range of fields, including philosophy (Davidson, 1973;
Dennett, 1990), psychology and neuroscience (Baker et al.,
2009; Schultz et al., 1997), economics and decision theory (von Neumann & Morgenstern, 1944; Savage, 1948), and reinforcement learning (Bellman, 1966; Ng & Russell,
2000). More recently, the question of when, and in what
sense, goal attributions are warranted has become increasingly important for LLM-based agents (Xu & Rivera, 2024;
MacDermott et al., 2024; Everitt et al., 2025; Goldstein &
Lederman, 2025; Mazeika et al., 2025), particularly in the
context of AI safety (Naik et al., 2025; Wentworth & Lorell,
2025; Marks et al., 2025; Li et al., 2025; Summerfield et al.,
2025; Järviniemi et al., 2026).

A natural way to measure goal-directedness is through _be-_
_havioural evaluation_, i.e., assessing an agent’s actions relative to a candidate goal, often by comparing them to an
optimal policy (Xu & Rivera, 2024; Everitt et al., 2025).
However, purely behavioural approaches face fundamental
theoretical, practical, and philosophical challenges (Bellot
et al., 2025; Rajcic & Søgaard, 2025; Chalmers, 2025). First,
agent capabilities may act as confounders for behavioural
measures, as consistent failure may reflect capability limitations rather than an absence of goal-directed behaviour.
Second, behavioural monitoring alone may be insufficient
to guarantee alignment: a system with misaligned internal goals could produce aligned behaviour, or fail a safety-relevant evaluation task, when doing so is instrumentally
useful (Hubinger et al., 2019; Ngo et al., 2024).

To address these limitations, we propose a framework that
combines behavioural evaluation with the analysis of internal representations, enabling holistic assessment of goal-directedness as a rich property arising from the interaction
of beliefs, planning, and action selection. We study an LLM
agent in a fully observable grid world, tasked with navigating to a goal state across grids of varying sizes and obstacle
densities. We first subject the agent to standard capability
tests, then introduce controlled environment perturbations
and multi-goal task structures to assess the generalisability of its goal-directed behaviour. We find that the agent
is sensitive to task difficulty and goal-like task-irrelevant
cues, but robust to difficulty-preserving transformations and
instrumental goals. We then use activation probes to test if
goal-relevant information can be decoded from the agent’s
internal activations before and after reasoning. The probing
analyses recover _cognitive maps_ —latent beliefs about the
current environment state, including the agent position and
the goal location—as well as planned multi-step action sequences. We further find that task-relevant representations
reorganise during reasoning: pre-reasoning activations preserve broader spatial cues and longer-horizon plans, while
post-reasoning activations sharpen focus on next action selection. Fig. 1 provides an overview of our approach.

**Contributions.** Our primary contributions are as follows:

1. We propose a white-box framework combining behavioural assessment and representation probing analyses for goal-directedness evaluation.

2. We design controlled environment perturbations and
multi-goal task structures to measure bias and robustness in the agent’s goal-directed behaviour.

3. We probe the agent’s activations for environment state
beliefs and multi-step action plans, and use these decoded representations to assess behavioural coherence
relative to internal task-relevant information.

[Our code is available at github.com/SPAR-Telos/interp and](https://github.com/SPAR-Telos/interp)
[github.com/SPAR-Telos/reveng.](https://github.com/SPAR-Telos/reveng) We also provide an interactive viewer for decoded grids ( _cognitive_ _maps_ ) at
[huggingface.co/spaces/project-telos/trace-viewer.](https://huggingface.co/spaces/project-telos/trace-viewer)

## 2. Related Work

The problem of identifying an agent’s goals and intentions
has a rich history spanning multiple research fields. Seminal
work in philosophy (Davidson, 1973; Lewis, 1974; Dennett,
1990) and microeconomics (von Neumann & Morgenstern,
1944; Savage, 1948) has emphasised the predictive and explanatory power of assigning goals to an agent.

**Measuring Agents’ Goal-directedness.** Recent work has
sought to formally define and measure goal-directedness,
particularly in the context of AI alignment and safety (Ward
et al., 2024; Xu & Rivera, 2024; Everitt et al., 2025; MacDermott et al., 2024). Notably, Everitt et al. (2025) define a
measure of goal-directedness conditioned upon an agent’s
task-relevant capabilities and show goal-directedness is measurably distinct from performance in LLMs and generalises
across tasks. MacDermott et al. (2024) build upon Dennett
(1990), proposing a formal measure of goal-directedness
based on the predictive power of posited utility functions
for the agent’s behaviour. However, behavioural approaches
to measuring goal-directedness are not without their weaknesses. Rajcic & Søgaard (2025) argue that such methods
falter when faced with underspecification, coarse goals, uncertainty, and multi-agent settings. Bellot et al. (2025) prove
bounds on learnability from agent behaviour, showing that
goal inferences are strictly limited by gaps between internal world models and the environment, as well as out-ofdistribution shifts. Our work complements these approaches
by enabling assessment of goal-directed behaviour relative
to the agent’s internal beliefs rather than ground truth alone.

**Inverse Reinforcement Learning (IRL).** IRL is a direct
instantiation of the goal attribution problem, aiming to infer
a reward function from a policy or a set of demonstrations
(see Ng & Russell, 2000; Abbeel & Ng, 2004, surveyed by
Arora & Doshi, 2021). Work in this area has also been developed in connection with AI alignment (e.g., Hadfield-Menell
et al., 2016; 2017). While a weakness of classical IRL is the
assumption that observed behaviour is optimal, approaches
like MaxEnt IRL (Ziebart et al., 2008) aim to relax this via
stochastic models of behaviour. Still, IRL methods suffer from the mis- and under-specification of the agent’s behavioural model and latent reward function, respectively
(Skalse & Abate, 2023). Under strong assumptions—full
observability, goal-directedness as optimal utility maximisation, perfect generalisation to new environments, and the observer’s ability to perform interventions (i.e., to design environments and evaluate the agent within them)—behavioural
experiments can, in principle, identify an agent’s goal (Amin
& Singh, 2016). However, these assumptions are unlikely to
hold for LLM-based agents. In contrast to IRL, in this work
we directly probe for goal-relevant representations without
assuming a specific reward structure.

**Probing Environment and Plans in LLMs.** Several studies
have examined whether language models learn structured
representations of their environment. Li et al. (2023) show
that a GPT model trained to predict Othello moves develops a causally relevant representation of the board state,
while Nanda et al. (2023) show this representation can be
decoded linearly. Similar linear representations of spatial
and temporal information were found in LLMs trained on
natural text (Gurnee & Tegmark, 2024). Recent work has
also probed LLMs for goal-oriented abstractions (Li et al.,
2024) and shown that models engage in forward planning,
pre-selecting future outputs before generating intermediate tokens (Pal et al., 2023; Men et al., 2024; Lindsey et al.,
2025; Dong et al., 2025). Similar phenomena have also been
observed in other neural architectures (Bush et al., 2025;
Taufeeque et al., 2025). More broadly, high-level features
have been found to be decodable from model activations
and used for monitoring and steering (Li et al., 2021; Zou
et al., 2023; Marks & Tegmark, 2024). We extend this line
of work through the lens of propositional interpretability
(Chalmers, 2025), eliciting environment representations and
plans from model internals in an agentic navigation task.

## 3. Grid World Environment Setup

We select GPT-OSS-20B (OpenAI, 2025) for our evaluation
in light of its manageable size and strong performance on
complex tasks, and test it on a two-dimensional navigation
task using the MiniGrid environment (Chevalier-Boisvert
et al., 2023). The LLM agent has full observability of the
grid and is tasked with navigating to the goal square one
action at a time. We translate the grid into a text-based
representation (Fig. 3) ensuring that each cell in the grid
corresponds to exactly one token to limit issues arising from
tokenisation (Edman et al., 2024; Cosma et al., 2025).

**Full Observability.** A fully observable environment offers
a simple but tightly controlled setting for analysing goal-directedness, for at least three reasons. (1) With full observability, the agent directly observes the true world state;
this eliminates the need to maintain beliefs over hidden
world states and allows optimal policies to be derived using standard algorithms. (2) Full observability also removes
several factors that might otherwise confound the analysis, including memory, belief updating under perceptual
uncertainty, and exploration–exploitation trade-offs. (3) We
observed that LLM-based agents, including frontier models,
perform poorly in partially observable grid worlds, exhibiting behaviours like redundant backtracking. This makes it
difficult to disentangle capability limitations from failures
of goal-directedness; we discuss this further in App. A.

**Grid** **Worlds.** We model _n × n_ grid world environments as Markov Decision Processes defined by the tuple ( _S, A, T, r, γ_ ). The state space is _S_ = [ _n_ ] [2], representing grid locations, and the action space is _A_ =
_{_ UP _,_ DOWN _,_ LEFT _,_ RIGHT _}_ . Transitions are deterministic: the transition function _T_ ( _s_ _[′]_ _|_ _s, a_ ) = P( _St_ +1 = _s_ _[′]_ _|_
_St_ = _s, At_ = _a_ ) moves the agent to the adjacent cell as
determined by _a_ if that cell is open; otherwise (e.g., if the
action would enter a wall), the agent remains in its current
location. A grid world instance is specified by a function
_G_ : [ _n_ ] [2] _→{_ wall _,_ open _,_ goal _}_, which assigns a cell type
to each grid location. Grid worlds vary in obstacle density
_d ∈_ [0 _,_ 1], where _d_ = 0 corresponds to a fully open grid and
_d_ = 1 to a maze-like grid with no circular paths. Examples
of grids with different density levels are shown in Fig. 2.

**Policies** **and** **Trajectories.** We write _π_ _[∗]_ for an optimal
policy, assumed to be uniform over optimal actions when
multiple optima exist. An agent parameterised by _θ_ follows
a policy _πθ_ ( _a | s_ ) = P( _At_ = _a | St_ = _s_ ) _,_ with _at_ denoting
the action selected by the agent at time _t_ . Given a policy _π_
and an initial state _s_ 0, a trajectory _τ_ _[π]_ ( _s_ 0) = ( _si, ai_ ) _[T]_ _i_ =0 [is]

generated by executing _π_ from _s_ 0. A trajectory is successful
if the final state satisfies _sT_ = _s_ goal; otherwise, it terminates
upon reaching a fixed maximum horizon _T_ .

## 4. Behavioural Evaluation

We begin with a behavioural evaluation of the agent’s
policy, comparing it against an optimal reference policy
derived using _A_ _[∗]_ search with Manhattan distance to the
goal. This analysis assesses how closely the agent’s action choices and action distributions align with optimal behaviour in the grid world, without inspecting the agent’s
internal representations. We construct a set of grid worlds _G_
with sizes _Z_ = _{_ 7 _,_ 9 _,_ 11 _,_ 13 _,_ 15 _}_ and obstacle densities
_D_ = _{_ 0 _._ 0 _,_ 0 _._ 2 _,_ 0 _._ 4 _,_ 0 _._ 6 _,_ 0 _._ 8 _,_ 1 _._ 0 _}_ . For each size–density
pair in _Z_ _× D_, we generate 10 random grids. On each grid,
the agent is evaluated over 10 trajectories using a sampling
temperature of 0 _._ 7 and a maximum horizon of _T_ = 1 _._ 5 _× L_
where _L_ is the optimal path length to the goal. [1] Additional
evaluation settings and prompts are reported in App. G.

1We limit trajectory length to 1 _._ 5 _×_ _L_ steps to filter cases where
the agent is stuck moving back and forth between states.

_d_ = 0 _._ 0 _d_ = 0 _._ 2 _d_ = 0 _._ 4 _d_ = 0 _._ 6 _d_ = 0 _._ 8 _d_ = 1 _._ 0 representation (right) used for LLM prompting.

Figure 2: Grid worlds with increasing wall density _d_, from fully open grids ( _d_ = 0) to maze-like grids with no circular paths ( _d_ = 1).

|#|#|#|#|#|#|#|#|#|#|#|
|---|---|---|---|---|---|---|---|---|---|---|
| # | _ | # | _ | _ | _ | _ | _ | _ | _ | # |
|#|_|_|_|_|_|_|_|_|_|#|
| # | _ | _ | _ | # | _ | _ | G | # | _ | # |
|# |_ |# |_ |# |_ |# |_ |# |# |# |
|# |_ |_ |_ |_ |_ |_ |_ |_ |_ |# |
|# |_ |# |_ |# |_ |_ |_ |_ |_ |# |
|# |_ |_ |_ |_ |_ |_ |_ |_ |_ |# |
|# |_ |# |# |# |_ |_ |_ |A |_ |# |
|# |_ |_ |_ |_ |_ |_ |_ |_ |_ |# |
|#|#|#|#|#|#|#|#|#|#|#|

Figure 3: An example grid (left) and its corresponding text-based

### 4.1. Goal-Directedness across Baseline Task Conditions

We report _per-action_ _accuracy_, defined as the fraction
of actions along a trajectory that are optimal, i.e., _at_ _∈_
arg max _a∈A π_ _[∗]_ ( _a_ _|_ _st_ ), the _entropy_ of the agent’s action
distributions, and the _Jensen–Shannon_ _divergence_ (JSD)
from the optimal policy. All metrics are averaged across
trajectories and grids. To estimate the agent’s policy _πθ_,
we compute empirical action probabilities from the relative
frequency of actions across all available trajectories for a
given grid. [2] App. B provides formal definitions and introduces additional metrics, including the success rate and the
expected calibration error, with full results shown in App. C.
While relevant, we do not find these to provide additional
insight beyond those discussed here.

The agent’s behavioural metrics vary systematically with
task difficulty. Per-action accuracy decreases monotonically
with both grid size and obstacle density. Conversely, policy
entropy and JSD from the optimal policy both increase
with grid size and obstacle density. Together, these results
indicate that the agent’s policy becomes less optimal and
more uncertain on more difficult grids.

We further analyse behavioural metrics as a function of the
agent’s distance to the goal (Fig. 4 top). Per-action accuracy decreases linearly with distance for distances below 20
steps, after which the estimates become noisier, and the JSD
from the optimal policy increases correspondingly. Variance in both metrics grows with distance, indicating less
stable behaviour when the agent is farther from the goal.
Fig. 4 (bottom) breaks down per-action accuracy by grid

2We use relative frequency instead of action token log-probability since the latter converges to 1 after the model reasoning
chain, making it a poor proxy for agent uncertainty.

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|Raw Smoothed (wind|ow=5)|||||

|Raw|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|Raw ~~Smoothed (~~|~~ window=5)~~|||||
|||||||
|||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||

_Figure_ _4._ **Top:** Action accuracy (left) and mean JSD (right) in relation to the agent’s distance from the goal. **Bottom** : Action
accuracy by size, complexity, and goal distance.

size, obstacle density, and distance to the goal, confirming
that increases along any of these three dimensions are associated with lower action accuracy. Controlling for the
other factors, accuracy decreases most systematically with
increasing grid size and obstacle density, with distance to
goal only playing a significant role for the larger grid sizes.

### 4.2. Robustness to Iso-difficulty Transformations

Having established that the agent’s performance varies predictably with task difficulty, we next evaluate its robustness
to controlled environment transformations that preserve the
difficulty of the navigation task. We refer to these as _iso-_
_difficulty transformations_ . They allow us to assess whether
the agent is biased towards specific grid configurations despite equivalent task structure. We consider four such transformations, shown in Fig. 13 (App. D): (1) reflection of
the grid (REFLECTENV); (2) rotation of the grid by 90 _[◦]_

(ROTATEENV); (3) swapping the agent’s start position with
the goal position (STARTGOALSWAP); and (4) transposing
the grid (TRANSPOSEENV). Each transformation preserves
the grid size, obstacle density, and optimal path length of the
original grid, and therefore maintains the difficulty of the
task. The optimal policy on the transformed grid is obtained
by applying the corresponding transformation to the optimal
policy of the baseline grid.

1.00

0.75

0.50

0.25

0.00

50+

45-50

40-45

35-40

30-35

25-30

20-25

15-20

10-15

5-10

0-5

0.0 0.2 0.4 0.6 0.8 1.0

0 10 20 30 40 50+
Distance to Goal

0 10 20 30 40 50+
Distance to Goal

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2Density0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

1.0

0.8

0.6

0.4

Table 1: Instrumental and implicit goals results.

**Results from** _**KeyDoorEnv**_ **and** _**KeyNoDoorEnv**_

**Metric** _**KeyDoorEnv**_ _**KeyNoDoorEnv**_

Success Rate (%) 100 _._ 0 98 _._ 9
Accuracy (%) 98 _._ 7 _±_ 3 _._ 2 97 _._ 2 _±_ 11 _._ 1

_KeyDoorEnv_ _KeyNoDoorEnv_ _2PathKeyEnv_

Figure 5: Grid world variants with instrumental and implicit goals.

In the text representation, the key and the door are encoded with K
and D, and their meaning is explained in the system prompt.

We apply our transformations to grids _G_, as defined at the
beginning of §4, and compare behavioural metrics between
each original grid and its transformed counterpart. For
each grid, we compute paired metrics across all trajectories
and use a Wilcoxon signed-rank test to assess whether
performance differs significantly between baseline and transformed environments. Across all transformations, we find
no statistically significant differences in any of the evaluated
metrics. This indicates that the agent’s behaviour in grid
worlds is driven by task-relevant information rather than
by incidental properties of particular grid configurations.
Detailed results are reported in App. D (Tab. 3 and Fig. 14).

### 4.3. Instrumental and Implicit Goals

We next examine whether the robustness observed in the
previous behavioural evaluations extends to more complex
goal structures. We focus on two cases: _instrumental goals_,
where an intermediate subtask is necessary for reaching
the main task objective, and _implicit goals_, where a goallike object is present but carries no reward or utility for
the task. We instantiate these cases using three grid world
variants, shown in Fig. 5: _KeyDoorEnv_, _KeyNoDoorEnv_,
and _2PathKeyEnv_ . The prompt is provided in App. G.3.

**Instrumental goals.** In _KeyDoorEnv_, the agent must collect
a key to unlock a door that blocks the path to the goal.
The agent interacts with the key and the door automatically
upon reaching the associated cell, with the door acting as
a wall if the agent does not have the key. This setting tests
whether the agent can handle a goal structure in which an
instrumental subgoal is required to reach the main goal.

**Implicit** **goals.** In _KeyNoDoorEnv_, the door is removed,
making the key functionally useless. This allows us to test
whether a goal-like object influences the agent’s behaviour
despite its irrelevance to task completion. In _2PathKeyEnv_,
the grid contains two optimal paths, one of which contains
a vestigial key, allowing us to test whether the agent’s path
selection is biased towards the key even when collecting it
is unnecessary. To isolate the effect of the key, we use a
matched control grid with identical structure but no key.

We generate 100 trajectories for _KeyDoorEnv_ and _KeyN-_
_oDoorEnv_, and 100 trajectory pairs for _2PathKeyEnv_ to

_Stage-specific Accuracy (%):_

Collecting Key 98 _._ 6 _±_ 5 _._ 7 N/A
Opening Door 99 _._ 2 _±_ 3 _._ 2 N/A
Reaching Goal 99 _._ 2 _±_ 3 _._ 3 N/A

_Key-related Metrics:_

Key Pickup Rate (%) 100 _._ 0 17 _._ 0
Key Attraction Bias [*] (%) N/A 75 _._ 0
*Percentage of non-optimal actions that move towards the key

**Comparison of Trajectories from** _**2PathKeyEnv**_

**Metric** **With Key** **Without Key**

Success Rate (%) 71 _._ 4 75 _._ 5
Accuracy (%) 76 _._ 0 _±_ 16 _._ 1 74 _._ 3 _±_ 15 _._ 7
Key Pickup Rate (%) 67 _._ 3 N/A
Jaccard Sim. 65 _._ 6 _±_ 35 _._ 8 account for the key presence. We set _T_ = 30, which is
sufficient for solving the maze. Results are summarised in
Tab. 1. In _KeyDoorEnv_, the agent achieves a 100% success
rate, correctly collecting the key, unlocking the door, and
reaching the goal. Accuracy relative to the optimal policy remains high throughout all subtasks. This indicates
successful handling of instrumental goals. In contrast, performance slightly deteriorates in _KeyNoDoorEnv_, despite
the reduced task complexity. Although the key has no functional utility, the agent still picks it up in 17% of trajectories.
Moreover, among non-optimal actions, 75% move towards
the key, indicating that the key acts as an effective distractor
even when it is not ultimately collected. In _2PathKeyEnv_,
the agent is biased towards the key-containing path, with
a key pickup rate of 67.3%. This bias leads to a slight improvement in per-action accuracy (76.0% vs. 74.3%) but
ultimately lowers success compared to the no-key control
(71.4% vs. 75.5%). Trajectories with and without the key
also show low Jaccard similarity, indicating substantial behavioural differences induced by the presence of the key.

In summary, the agent reliably solves tasks with instrumental goals, but is also systematically influenced by goal-like
non-functional artefacts. We conjecture that this may reflect
a conflict between the task goal specified in the prompt and
semantic associations acquired during training, such as the
common association between keys and progress in games,
with the latter not consistently suppressed in favour of the
inference-time task goal.

## 5. Representational Evaluation

Behavioural evaluation alone is insufficient to determine an
agent’s goal-directedness. This limitation has been noted

MLP Probe Linear Probe

Wall Empty Agent Goal

|Col1|83|Col3|100|99|Col6|Col7|Col8|99|93|81|Col12|Col13|85|83|Col16|Col17|Col18|Col19|94|Col21|Col22|Col23|Col24|87|7|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||52|||71 ~~66~~|71 ~~66~~|71 ~~66~~|||~~68~~ 81|~~68~~ 81|~~68~~ 81|||~~69~~ 72 71|~~69~~ 72 71|~~69~~ 72 71|~~69~~ 72 71||~~67~~ 71 72|~~67~~ 71 72|~~67~~ 71 72|~~67~~ 71 72|||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|83 85 ~~87~~ 78 74 81 74 74 75 74|
|||||||||||||||||
||||~~20~~ 30|||~~13~~ ~~18~~|||21 ~~15~~|||~~18~~ 9|||~~12~~ 7|

7 9 11 13 15
Grid Size

0.8

0.6

0.4

0.2

1.00

0.75

0.50

0.25

0.00

1.00

0.75

0.50

0.25

0.00

|72|.4 69|74 .4|.7 70|.7 69|.0|
|---|---|---|---|---|---|
|||31|.6 |39|.1|
|20|.3 28|.7 |27|.6||

Grid Size

Figure 6: Extracting a cognitive map from GPT-OSS-20B representations. Left: Overall accuracy of an MLP and a linear probe. Centre:

Per-class recall (=accuracy) and precision for varying grid sizes. Right: A cognitive map decoded from pre-reasoning activations.

in both theoretical (Bellot et al., 2025) and philosophical
work (Rajcic & Søgaard, 2025; Chalmers, 2025), and has
clear practical implications. In grid world navigation, for
example, an agent may fail to reach the goal state while still
acting goal-directedly relative to its own imperfect beliefs
about the environment. In this section, we therefore analyse the agent’s internal world representations to evaluate
whether its actions are consistent with those beliefs.

We begin in §5.1 by decoding the agent’s beliefs about
the environment state, producing what we term _cognitive_
_maps_ . [3] Building on this, in §5.2, we evaluate the optimality
of the agent’s actions with respect to its decoded, subjective cognitive map. Finally, in §5.3, we examine whether
goal-directed action plans can be extracted from the agent’s
internal representations. In App. E.3, we additionally test
whether the agent encodes its distance to the goal.

**5.1. Cognitive Maps:** **Decoding the Agent’s Beliefs**
**about its Environment**

To assess whether the agent’s representations encode
an internal model of its environment, we prompt GPT-OSS-20B to solve the fully observable grid navigation
task described in §3 and extract residual-stream activations from selected prompt positions. Specifically, we
use the final three pre- and post-reasoning tokens in
the model chat template (<|end|>, <|start|>, and
assistant), extracting activations at layers 7, 15, and
23. Loosely inspired by Li et al. (2023), we construct
training examples by augmenting each activation with
the ( _x, y_ ) coordinates of the queried cell. This yields
input–output pairs of the form ([act _, x, y_ ] _, c_ ), where _c_ _∈_
_{_ agent _,_ goal _,_ wall _,_ open _,_ padding _}_ . [4] We then train linear
and MLP classifiers on the resulting pairs to decode cell
types across grid positions. For the MLP probe, we use
a two-layer architecture with ReLU activation and a hidden dimension of 1024. Both probes are trained using an

3We borrow this term from classic cognitive neuroscience work
on navigational tasks (Tolman, 1948; Schmidt & Redish, 2013).

4When training general probes, grids smaller than 15 _×_ 15 are
padded by assigning positions outside the original grid the padding
label. After constructing the full dataset across all grid sizes, we
upsample minority classes to match the size of the largest class.

|Agent Goal|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~|
|83 ~~45~~ ~~42~~  ~~40~~ 80 52 ~~37~~ ~~40~~||||||||||||
||||||||27||||20|
||||||||27|||||

Figure 7: Probe performance for locating agent and goal positions.

Binary localisation accuracy drops as the grid size increases, but
avg. Manhattan distance to true locations remains bounded.

AdamW optimiser with weight decay, and normalisation is
applied before training. At test time, we apply the probes to
each grid coordinate and reconstruct the model’s _cognitive_
_map_ by selecting arg max _c P_ ( _c |_ act _, x, y_ ) at each position.
This cognitive map represents the model’s decoded belief
over the current grid state. We train _general_ and _size-specific_
variants of these probes using grids of sizes 7 _,_ 9 _,_ 11 _,_ 13 _,_ 15.
In the general case, we pad smaller grids to size 15 by assigning _c_ = padding to missing positions. Unless otherwise
stated, the following experiments focus on general probes
trained on pre-reasoning activations from the intermediate
layer 15. Additional results across layers, grid sizes, and
goal-distance probes are reported in App. E.2.

**How is the Environment Encoded in Model Activations?**
Fig. 6 (left) shows the accuracy of cognitive maps reconstructed by linear and MLP probes across various grid
sizes. The MLP probe decodes cell identities with around
70% accuracy, reaching a maximum of 75.7% for 11 _×_ 11
grids. Linear probes underperform at 39.1% accuracy in
the same setting, suggesting that environment information
is encoded non-linearly in model representations. Fig. 6
(centre) presents a per-class performance breakdown of the
MLP probe in terms of its recall (top) and precision (bottom). Recall scores show that all cell identities are retrieved
robustly across grid sizes, with especially high recall for
goal (83-99%) and agent (72-100%) positions. We note that
probes assign agent and goal labels to multiple cells in the
neighbourhood of their respective true locations, resulting
in the high recall but low precision trend discussed earlier
(Fig. 6, right). In contrast, information about the position
of walls is not represented in detail, as reflected by the

1.0

0.5

0.0

7 9 11 13 15
Grid Size

7 9 11 13 15
Grid Size

1.5

1.0

0.5

0.0

Top-3 decoded
Top-5 decoded

1.0

0.9

0.8

0.7

8 10 12 14
Grid Size

Taken optimal (GT)
Top-1 decoded

1.0

0.9

0.8

0.7

0.0 0.2 0.4 0.6 0.8 1.0
Complexity

Top-10 decoded

Figure 8: Performance of cognitive map before and after reasoning.

Cognitive map accuracy drops significantly after reasoning.

lower recall for the wall class. These results suggest that the
agent’s internal representations encode a coarse spatial map
of the environment, preserving approximate task-relevant
information about agent position and goal location.

**Locating Agent and Goal.** Given the coarse localisation of
agent and goal cells in cognitive maps, we set out to test how
accurately the true agent and goal locations can be recovered.
We measure top-1 accuracy using the grid coordinates with
the highest predicted _P_ ( _c_ = agent) and _P_ ( _c_ = goal), with
results shown in Fig. 7. We find that agent and goal localisation accuracy decreases steadily with grid size. However,
the Manhattan distance between predicted and true locations
remains lower than 2 even for large 15 _×_ 15 grids, suggesting
that information about agent and goal positions is encoded
coarsely in proximity to their true locations.

**Goal Location Information Degrades after Reasoning.**
We next evaluate how decoded cognitive maps change
from pre- to post-reasoning activations. As shown in Fig. 8,
overall probe accuracy drops from 75% before reasoning
to 60% after reasoning, with a notable decrease in agent,
goal and open recall, as well as in agent and goal precision.
These results indicate that the coarse environment state
information recovered by the cognitive map probes is
substantially less available after reasoning. This may reflect
a reorganisation of the model’s representations, in which
spatial information becomes less directly decodable as the
model shifts towards encoding information more directly
relevant to action selection.

### 5.2. Evaluating Policies against Decoded Beliefs

Behavioural deviations from the optimal policy do not, by
themselves, show that the agent is not acting goal-directedly.
If the agent’s internal representation of the environment is
incomplete or inaccurate, an action may be suboptimal in
the true grid while still being appropriate relative to the
agent’s own beliefs. We therefore test whether the agent’s
behaviour is consistent with its decoded cognitive maps, fo cusing in particular on cases where its actions deviate from
the ground-truth optimal policy. To do so, we compare the
agent’s observed action sequence with the optimal policy derived from the cognitive map decoded at each trajectory step
using the general MLP probe with layer-15 pre-reasoning
activations. As in §5.1 (and Fig. 7), we first identify a single
agent and goal location by selecting the grid cells with the
highest predicted probabilities for the corresponding classes.
We then compute optimal actions on both the decoded and
ground truth grids. Tab. 2 reports the accuracy of the agent’s
actions with respect to optimal policies defined on the decoded cognitive map ( _Acc. Dec._ ) and the ground truth grid
( _Acc. GT_ ), as well as the fraction of actions that are optimal
under both policies ( _Agreem._ ).

Figure 9: Per-action accuracy against the optimal policy defined
with respect to the top- _k_ decoded grids.

_Acc. Dec._, the accuracy relative to the optimal policy in the decoded cognitive map, decreases with grid density but is
high across grid sizes (average: 82.5%). This indicates that
the agent’s actions are broadly consistent with its internal
world representation. Of particular interest is the recovery metric _Rec._, which is the proportion of actions that are
suboptimal in the true environment but optimal given the
agent’s decoded cognitive map (see App. E.4 for the inverse
recovery rate). _Rec._ ranges between 37.4% and 88.4% (average: 57.9%), indicating that a substantial fraction of failures
can be attributed to inaccurate or fuzzy world representations rather than a lack of goal-directedness, particularly in
low-density and medium-to-large grids.

While the agreement between the policy defined on the
decoded vs. ground truth grid is high across conditions
( _Agreem._ averages 83.9% and is always above 77% except
for the highest density grids), we remark that _Acc. Dec._ is
consistently lower than _Acc. GT_ . This may be due to the fact
that we derive a single location for the agent and goal by
selecting the grid cell with the highest predicted probability.
This approach does not capture the uncertainty evident in
the decoded maps, which exhibit blurred agent and goal
spatial representations, as discussed in §5.1 and shown in
Figure 6: As a result, actions may be optimal with respect to
nearby cells rather than the single argmax location.

**Uncertainty-Aware Cognitive Maps.** To account for the
agent’s fuzzy world representations and mitigate the low
precision of goal and agent location probes, we consider top- _k_ decodings of the cognitive map, evaluating action optimality with respect to a set of _k_ candidate agent and goal
positions per grid. Fig. 9 shows the proportion of optimal
actions for the ground truth grid, top-1 decoded grid (results
from Tab. 2), and top- _k_ decoded grids for _k_ _∈{_ 3 _,_ 5 _,_ 10 _}_ .
Accuracy against top- _k_ decodings is consistently higher
than accuracy against the ground truth grid across grid sizes
and complexity bins. This indicates that accounting for
uncertainty in the probe predictions allows the decoded
grids to explain a larger share of the agent’s actions. We
interpret these results as evidence that top-1 decoding collapses the agent’s representational uncertainty. As the state
space grows and the agent’s internal representations become
fuzzier, its actions are better characterised as planning under
a distribution over plausible states.

**Intervention** **via** **Activation** **Patching.** Finally, to complement our observational probing analyses, we conducted
preliminary activation patching experiments on the residual stream of GPT-OSS-20B, using corrupted examples
that move either the agent or the goal square. We find
that patching changes the model’s action distribution only
when applied across all layers simultaneously, and only at
grid state tokens or the token immediately preceding the
action output. Single-layer interventions applied to the chat
template tokens proved ineffective (see App. F). This asymmetry between readouts and interventions is consistent with
recent findings suggesting that prompt-related attributes can
be read by probing classifiers at various specific points in
the forward pass—often from individual layers—but interventions require more distributed representation editing to
achieve a measurable impact (Choi et al., 2025). Identifying
layer- and position-specific intervention strategies for our
setup is thus an important future direction.

### 5.3. Evaluating Plans

In this section, we examine whether the agent’s internal
representations encode goal-directed planning information,
and how this encoding differs before vs. after reasoning. We
consider the same single-goal navigation task as in previous
sections. For each trajectory, we extract residual-stream
activations from layers _ℓ_ _∈{_ 7 _,_ 15 _,_ 23 _}_ at two stages: (i) _pre-_
_reasoning_, using the final prompt tokens immediately before
the model begins reasoning, and (ii) _post-reasoning_, using
the final reasoning tokens immediately before the model
outputs its first action. Each example is labelled with a
target action sequence **a** 1: _T_ = ( _at_ ) _[T]_ _t_ =1 [,] [derived] [from] [the]

executed actions in the trajectory _τ_ _[π]_ ( _s_ 0) = ( _st, at_ ) _[T]_ _t_ =0 [for]

a given grid instance, with _T_ = 10. We train the plan
decoder on 3,000 trajectories, use a 600-trajectory validation
set, and report results on the 300-trajectory test set used
in §4. Trajectories are sampled from grid sizes 7–15 with
varying complexities and start/goal configurations, yielding
a diverse distribution of path lengths and planning difficulty.

|Col1|Col2|Col3|1-layer 1-layer|post-rea pre-reas|soning oning|Col7|4-layer 4-layer|post-re pre-rea|asoning soning|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
|||  | 2-layer ~~ -layer~~| post-rea ~~  re-reas~~| soning ~~  oning~~|| Random| baseli| ne (0.25|N)|
||||||||||||
||||||||||||
||||||||||||
||||||||||||

Figure 10: Prefix accuracy of one-shot plan decoding across de
coder capacities. We compare Transformer decoders with 1, 2, and
4 layers, with activations extracted before vs. after reasoning.

**Plan** **Decoder** **Architecture.** Our goal is to decode the
entire plan from a fixed set of activations, while minimising any additional planning or inference performed by the
probe. Let **h** 1 _,_ **h** 2 _,_ **h** 3 _∈_ R [2880] denote the three extracted
token activations at a chosen layer and stage. We first map
each **h** _i_ through a shared bottleneck consisting of a linear projection to 1024 dimensions followed by LayerNorm:

**h** ˜ _i_ = LN( _W_ **h** _i_ ) _∈_ R [1024] . We then decode a horizon- _T_
plan using a Transformer decoder with _T_ learned query
embeddings **q** 1 _, . . .,_ **q** _T_ . Each query corresponds to a plan
step index and performs cross-attention over the same set
of token activations _{_ **h** [˜] 1 _,_ **h** [˜] 2 _,_ **h** [˜] 3 _}_ . We evaluate Transformer
variants with 1, 2, and 4 layers, each with 8 attention heads
per layer. A final linear head produces a distribution over actions for each step _t ∈_ [ _T_ ], _p_ ( _at_ _|_ **h** [˜] 1:3) = softmax( _Wo_ **z** _t_ ),
where **z** _t_ is the decoder output at query slot _t_ .

Importantly, we design the decoder to predict the entire plan
**a** ˆ1: _T_ simultaneously from the input activations, rather than
predicting _a_ ˆ _t_ autoregressively conditioned on previously
decoded actions ˆ _a<t_ . Autoregressive decoding would introduce an additional channel for the probe to create plan structure: early predicted actions can implicitly constrain later accuracy w.r.t. the optimal policy on the ground truth grid; _Acc._
_Dec._ : action accuracy w.r.t. the optimal policy on the cognitive
map; _Agreement_ : % optimal actions for both policies; _Recovery_ :
% optimal actions only for the cognitive map. Grid size results
(top) are averaged across density, and vice versa (bottom).

Table 2: Policy evaluation against decoded beliefs. _Acc. GT_ : action

_**n**_ **Acc.** **GT (%)** **Acc.** **Dec.** **(%)** **Agreement (%)** **Recovery (%)**

7 97 _._ 3 90 _._ 1 91 _._ 5 48 _._ 3
9 95 _._ 2 88 _._ 2 88 _._ 3 42 _._ 4
11 84 _._ 1 77 _._ 3 81 _._ 7 74 _._ 2
13 79 _._ 4 78 _._ 5 79 _._ 7 64 _._ 5
15 78 _._ 7 78 _._ 5 78 _._ 4 60 _._ 1

_**d**_ **Acc.** **GT (%)** **Acc.** **Dec.** **(%)** **Agreement (%)** **Recovery (%)**

0 _._ 0 100 _._ 0 94 _._ 6 98 _._ 6 N/A
0 _._ 2 94 _._ 5 89 _._ 1 93 _._ 2 88 _._ 4
0 _._ 4 88 _._ 9 90 _._ 0 93 _._ 7 82 _._ 4
0 _._ 6 87 _._ 3 80 _._ 3 83 _._ 2 49 _._ 6
0 _._ 8 84 _._ 4 75 _._ 4 77 _._ 3 47 _._ 8
1 _._ 0 66 _._ 6 65 _._ 8 57 _._ 6 37 _._ 4

1 2 3 4 5 6 7 8 9 10
Prefix length N (first N steps correct)

actions via simple continuation heuristics, even if the underlying representations only weakly specify a full-horizon
trajectory. By contrast, in one-shot decoding, later steps cannot condition on earlier predictions, so coherent multi-step
structure in ˆ **a** 1: _T_ must be supported by information already
present in the base model’s activations. Thus, accuracy
above baseline under one-shot decoding is more diagnostic
of plan information encoded in the model’s representations
than of computation performed by the probe itself. We
evaluate plan decodability using _prefix_ _accuracy_, defined
as the fraction of episodes for which the first _N_ predicted
actions exactly match the target plan prefix. For a predicted
plan **a** ˆ1: _T_ and target action sequence **a** 1: _T_ (with _T_ = 10),
prefix accuracy at _N_ is Pr[ˆ **a** 1: _N_ = **a** 1: _N_ ]. We report prefix
accuracy for _N_ _∈{_ 1 _, . . .,_ 10 _}_ for both pre-reasoning and
post-reasoning activations. As a baseline, random guessing
among four actions yields 0 _._ 25 _[N]_ .

**Results.** Fig. 10 shows that all probes exceed the 0 _._ 25 _[N]_

baseline at short horizons, indicating that GPT-OSS-20B activations encode non-trivial information about upcoming action sequences. Across decoder capacities, the 2-layer probe
performs best overall, particularly with post-reasoning activations: it achieves 66.49% accuracy at _N_ = 1 and remains
strongest through _N_ = 4. For longer prefixes, the 2-layer
pre-reasoning probe has the strongest performance, reaching
7.3% at _N_ = 5, 5.0% at _N_ = 6, and 3.8% at _N_ = 7. The
1-layer probe also retains substantial one-step decodability
(45.5% pre-reasoning, 50.6% post-reasoning), but deteriorates rapidly with prefix length, while the 4-layer probe
achieves non-trivial longer-prefix accuracy without consistently improving over the smaller probes. Varying probe
capacity provides a control against the possibility that the
decoder itself is solving the navigation task. If performance
were driven by decoder-side navigation, larger decoders
should consistently outperform smaller ones. Instead, performance is non-monotonic in decoder capacity: the 2-layer
probe outperforms the 4-layer probe post-reasoning, and
even the 1-layer probe recovers strong one-step planning information. This suggests that immediate action information
is readily accessible to a low-capacity readout, while multi-step plan structure requires additional capacity to extract.

Focusing on the 2-layer probe as the strongest overall readout, we observed that post-reasoning activations improve
one-step decoding relative to pre-reasoning (66.49% vs.
41.5% at _N_ = 1), consistent with reasoning increasing
the separability of the next action (see §5.1). This advantage persists through _N_ = 4, while for longer prefixes
pre-reasoning activations become more decodable, crossing
at _N_ = 5. This crossing point suggests again a trade-off
induced by reasoning, with post-reasoning activations more
strongly supporting _local_ action selection, and pre-reasoning
representations preserving more recoverable long-horizon
trajectory structure. One interpretation of this trade-off is that reasoning shifts representational emphasis from broader
environment-related structure towards near-term action selection. Finally, additional analyses (reported in App. E.5)
show that the decoder does not recover the exact trajectory
length in most cases, but it reliably predicts a close estimate.
Since trajectory length in this setting is tightly coupled to
progress towards the goal under the executed policy, this
provides complementary evidence that the model’s internal
states encode coarse plans beyond the next action.

## 6. Conclusion

We have presented a framework for analysing the goal-directedness of LLM-based agents that integrates behavioural evaluation with representation probing and demonstrated its utility in a grid world navigation setting with GPT-OSS-20B as the agent under study. Behaviourally, the agent
shows systematic sensitivity to task difficulty while remaining robust to goal-irrelevant environmental variations, providing initial evidence of goal-directedness. The agent also
succeeds in multi-goal grid worlds with instrumental subgoals, though its behaviour is biased by goal-like but task-irrelevant cues. Representational analyses uncovered structured internal states consistent with an interpretation of the
agent as goal-directed. The model encodes cognitive maps
that capture task-relevant spatial information about the location of the agent and the goal, and its representations exhibit
a shift across the reasoning process: pre-reasoning, they preserve spatial information about the environment and longer-horizon plans, while post-reasoning, they have a narrower
focus on next action selection. This finding suggests that reasoning reorganises information to support effective control.

Our controlled setup enables precise measurement but abstracts away from the complexity of real-world agentic settings, making extension to more complex environments an
important direction for future work. Moreover, while we
find consistent relationships between internal representations and behaviour, simple activation patching does not
reliably alter behaviour, leaving the establishment of causal
links as an open challenge. In this context, probing alternative grid encodings (see, e.g., Ivanitskiy et al., 2024)
may reveal aspects of the environment state not captured by
our probes. Addressing these questions, and extending the
framework across architectures, scales, and training regimes,
will be important for assessing the generality of our findings.

Looking forward, the methods and insights from this work
provide a foundation for developing more comprehensive
approaches to goal attribution and monitoring in agentic
systems. The development of rigorous approaches to evaluating goal-directedness is a prerequisite for making high-confidence claims about agents’ goals and potential related
risks, and for informing the responsible deployment and
oversight of increasingly autonomous AI systems.

**Acknowledgements**

This project was supported by [SPAR.](https://sparai.org) GS acknowledges
support by the NDIF project (U.S. NSF Award IIS-2408455).
We thank Cozmin Ududec, Dima Krasheninnikov, Gonçalo
Guiomar, Michael Hanna, and members of the BauLab at
Northeastern University for helpful discussions.

**Author Contributions**

RA, FC, CM, GS, and MG conducted the literature review
of §2, identified and synthesised relevant prior work across
research paradigms, and developed the framing that situates the project’s contributions within existing research
on goal-directedness. RA, ND, and AN worked on implementation of code infrastructure for §4. ND developed the
iso-difficulty transformations and conducted the analysis
for §4.1, §4.2, App. A, App. C, and App. D. AN developed
the procedures for generating the base environments and
the variants with the key, and conducted the analysis for
§4.3. GS advised the design of representational evaluation
experiments of §5, developed a unified pipeline for trace
generation, activation extraction, and probe training, and
built a trace viewer to explore probing results. EK designed,
implemented, and evaluated the cognitive map probes in
§5.1 and App. E. RA, ND, and AN conducted the analysis
for §5.2. MN designed, implemented, and evaluated the
plan decoder in §5.3. MG conceived and led the project,
provided ongoing scientific guidance, and coordinated the
project’s execution. All authors contributed to the conceptualisation of the study and the preparation of the manuscript.

**Impact Statement**

Goal-directedness lies at the core of agency: understanding whether an artificial system pursues goals, which goals
it pursues, and how those goals relate to its behaviour is
foundational for explaining, predicting, and governing advanced AI systems. Progress on this problem has broad
scientific significance, spanning cognitive science, linguistics, neuroscience, philosophy of action, macroeconomics,
and decision theory, but it also carries important practical
implications as AI systems are increasingly deployed in
autonomous, long-horizon, and high-stakes settings.

The ability to reliably evaluate goal-directedness is particularly critical from a safety perspective. As AI systems
become more capable, behavioural success alone becomes
an increasingly weak signal of alignment. Systems driven
by misaligned internal objectives may nonetheless appear
competent, compliant, or even helpful because aligned behaviour is instrumentally useful. Recent work has highlighted risks such as alignment faking and sandbagging
(Greenblatt et al., 2024; van der Weij et al., 2025; Taylor
et al., 2025), with models behaving deceptively or underper forming during evaluation, arguably to avoid modification or
oversight. While claims of deliberate “AI scheming” would
be highly consequential if substantiated, current evidence
remains limited, often anecdotal, and difficult to interpret
without clear hypotheses, controls, or mechanistic grounding (Summerfield et al., 2025). This paper takes a first step
towards addressing these limitations.

**References**

Abbeel, P. and Ng, A. Y. Apprenticeship learning via
inverse reinforcement learning. In _Proceedings_ _of_ _the_
_Twenty-First International Conference on Machine Learn-_

_ing_, pp. 1, New York, NY, USA, 2004. Association for
[Computing Machinery. URL https://doi.org/10.](https://doi.org/10.1145/1015330.1015430)
[1145/1015330.1015430.](https://doi.org/10.1145/1015330.1015430)

Amin, K. and Singh, S. Towards resolving unidentifiabil
ity in inverse reinforcement learning. _arXiv_ _preprint_
_arXiv:1601.06569_, 2016. URL [https://arxiv.](https://arxiv.org/abs/1601.06569)
[org/abs/1601.06569.](https://arxiv.org/abs/1601.06569)

Arora, S. and Doshi, P. A survey of inverse reinforcement learning: Challenges, methods and
progress. _Artificial_ _Intelligence_, 297:103500, 2021.
URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0004370221000515)
[science/article/pii/S0004370221000515.](https://www.sciencedirect.com/science/article/pii/S0004370221000515)

Baker, C. L., Saxe, R., and Tenenbaum, J. B. Action under
standing as inverse planning. _Cognition_, 113(3):329–349,
[2009. URL https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0010027709001607)
[science/article/pii/S0010027709001607.](https://www.sciencedirect.com/science/article/pii/S0010027709001607)

Bellman, R. Dynamic programming. _Science_, 153(3731):

34–37, 1966. [URL https://www.science.org/](https://www.science.org/doi/abs/10.1126/science.153.3731.34)
[doi/abs/10.1126/science.153.3731.34.](https://www.science.org/doi/abs/10.1126/science.153.3731.34)

Bellot, A., Richens, J., and Everitt, T. The limits of
predicting agents from behaviour. In _Proceedings_ _of_
_the_ _42nd_ _International_ _Conference_ _on_ _Machine_ _Learn-_
_ing_ . JMLR.org, 2025. [URL https://dl.acm.org/](https://dl.acm.org/doi/10.5555/3780338.3780469)
[doi/10.5555/3780338.3780469.](https://dl.acm.org/doi/10.5555/3780338.3780469)

Bush, T., Chung, S., Anwar, U., Garriga-Alonso, A., and Krueger, D. Interpreting emergent planning in
model-free reinforcement learning. In _The_ _Thirteenth_
_International Conference on Learning Representations_,
2025. [URL https://openreview.net/forum?](https://openreview.net/forum?id=DzGe40glxs)
[id=DzGe40glxs.](https://openreview.net/forum?id=DzGe40glxs)

Chalmers, D. J. Propositional interpretability in artificial intelligence. _arXiv_ _preprint_ _arXiv:2501.15740_, 2025.
[URL https://arxiv.org/abs/2501.15740.](https://arxiv.org/abs/2501.15740)

Chevalier-Boisvert, M., Dai, B., Towers, M., de Lazcano,

R., Willems, L., Lahlou, S., Pal, S., Castro, P. S., and Terry, J. Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks. In _Proceedings_ _of_ _the_ _37th_ _Interna-_
_tional Conference on Neural Information Processing Sys-_
_tems_, Red Hook, NY, USA, 2023. Curran Associates
Inc. URL [https://dl.acm.org/doi/abs/10.](https://dl.acm.org/doi/abs/10.5555/3666122.3669331)
[5555/3666122.3669331.](https://dl.acm.org/doi/abs/10.5555/3666122.3669331)

Choi, D., Huang, V., Schwettmann, S., and Stein
hardt, J. Scalably extracting latent representations
of users, 2025. URL [https://transluce.org/](https://transluce.org/user-modeling)
[user-modeling.](https://transluce.org/user-modeling)

Cosma, A., Ruseti, S., Radoi, E., and Dascalu, M. The strawberry problem: Emergence of character-level understanding in tokenized language models. In _Proceedings_
_of_ _the_ _2025_ _Conference_ _on_ _Empirical_ _Methods_ _in_ _Nat-_
_ural_ _Language_ _Processing_, pp. 28252–28263, Suzhou,
China, 2025. Association for Computational Linguistics. URL [https://aclanthology.org/2025.](https://aclanthology.org/2025.emnlp-main.1434/)
[emnlp-main.1434/.](https://aclanthology.org/2025.emnlp-main.1434/)

Davidson, D. Radical interpretation. _Dialectica_, 27(3/4):

313–328, 1973. URL [http://www.jstor.org/](http://www.jstor.org/stable/42968535)
[stable/42968535.](http://www.jstor.org/stable/42968535)

Dennett, D. C. The interpretation of texts, people and
other artifacts. _Philosophy_ _and_ _Phenomenological_
_Research_, 50:177–194, 1990. URL [https://www.](https://www.jstor.org/stable/2108038)
[jstor.org/stable/2108038.](https://www.jstor.org/stable/2108038)

Dong, Z., Zhou, Z., Liu, Z., Yang, C., and Lu, C.

Emergent response planning in LLMs. In _Proceed-_
_ings_ _of_ _the_ _42nd_ _International_ _Conference_ _on_ _Ma-_
_chine Learning_, volume 267, pp. 14301–14320. PMLR,
[2025. URL https://proceedings.mlr.press/](https://proceedings.mlr.press/v267/dong25p.html)
[v267/dong25p.html.](https://proceedings.mlr.press/v267/dong25p.html)

Edman, L., Schmid, H., and Fraser, A. CUTE: Measuring LLMs’ understanding of their tokens. In _Pro-_
_ceedings of the 2024 Conference on Empirical Methods_
_in_ _Natural_ _Language_ _Processing_, pp. 3017–3026, Miami, Florida, USA, 2024. Association for Computational
Linguistics. [URL https://aclanthology.org/](https://aclanthology.org/2024.emnlp-main.177/)
[2024.emnlp-main.177/.](https://aclanthology.org/2024.emnlp-main.177/)

Everitt, T., Garbacea, C., Bellot, A., Richens, J., Pa
padatos, H., Campos, S., and Shah, R. Evaluating
the goal-directedness of large language models. _arXiv_
_preprint_ _arXiv:2504.11844_, 2025. URL [https://](https://arxiv.org/abs/2504.11844)
[arxiv.org/abs/2504.11844.](https://arxiv.org/abs/2504.11844)

Goldstein, S. and Lederman, H. What does ChatGPT
want? An interpretationist guide, 2025. URL [https:](https://philpapers.org/rec/GOLWDC-2)
[//philpapers.org/rec/GOLWDC-2.](https://philpapers.org/rec/GOLWDC-2)

Greenblatt, R., Denison, C., Wright, B., Roger, F., Mac
Diarmid, M., Marks, S., Treutlein, J., Belonax, T.,
Chen, J., Duvenaud, D., Khan, A., Michael, J., Mindermann, S., Perez, E., Petrini, L., Uesato, J., Kaplan, J., Shlegeris, B., Bowman, S. R., and Hubinger,
E. Alignment faking in large language models. _arXiv_
_preprint_ _arXiv:2412.14093_, 2024. URL [https://](https://arxiv.org/abs/2412.14093)
[arxiv.org/abs/2412.14093.](https://arxiv.org/abs/2412.14093)

Gurnee, W. and Tegmark, M. Language models represent space and time. In _The Twelfth International Conference_
_on_ _Learning_ _Representations_, 2024. URL [https://](https://openreview.net/forum?id=jE8xbmvFin)
[openreview.net/forum?id=jE8xbmvFin.](https://openreview.net/forum?id=jE8xbmvFin)

Hadfield-Menell, D., Dragan, A., Abbeel, P., and Rus
sell, S. Cooperative inverse reinforcement learning. In _Proceedings_ _of_ _the_ _30th_ _International_ _Confer-_
_ence_ _on_ _Neural_ _Information_ _Processing_ _Systems_, pp.
3916–3924, Red Hook, NY, USA, 2016. Curran Associates Inc. URL [https://dl.acm.org/doi/10.](https://dl.acm.org/doi/10.5555/3157382.3157535)
[5555/3157382.3157535.](https://dl.acm.org/doi/10.5555/3157382.3157535)

Hadfield-Menell, D., Milli, S., Abbeel, P., Russell, S., and

Dragan, A. D. Inverse reward design. In _Proceedings_
_of the 31st International Conference on Neural Informa-_
_tion Processing Systems_, pp. 6768–6777, Red Hook, NY,
USA, 2017. Curran Associates Inc. [URL https://dl.](https://dl.acm.org/doi/10.5555/3295222.3295421)
[acm.org/doi/10.5555/3295222.3295421.](https://dl.acm.org/doi/10.5555/3295222.3295421)

Hubinger, E., van Merwijk, C., Mikulik, V., Skalse, J., and Garrabrant, S. Risks from learned optimization
in advanced machine learning systems. _arXiv preprint_
_arXiv:1906.01820_, 2019. URL [https://arxiv.](https://arxiv.org/abs/1906.01820)
[org/abs/1906.01820.](https://arxiv.org/abs/1906.01820)

Ivanitskiy, M., Spies, A. F., Räuker, T., Corlouer, G.,

Mathwin, C., Quirke, L., Rager, C., Shah, R., Valentine, D., Behn, C. D., Inoue, K., and Fung, S. W.
Linearly structured world representations in mazesolving transformers. In _Proceedings_ _of_ _UniReps:_
_the_ _First_ _Workshop_ _on_ _Unifying_ _Representations_ _in_
_Neural_ _Models_, volume 243, pp. 133–143. PMLR,
[2024. URL https://proceedings.mlr.press/](https://proceedings.mlr.press/v243/ivanitskiy24a.html)
[v243/ivanitskiy24a.html.](https://proceedings.mlr.press/v243/ivanitskiy24a.html)

Järviniemi, O., Makins, O., Merizian, J., Kirk, R., and

Millwood, B. Propensity inference: Environmental contributors to llm behaviour. _arXiv_ _preprint_
_arXiv:2604.21098_, 2026. URL [https://arxiv.](https://arxiv.org/abs/2604.21098)
[org/abs/2604.21098.](https://arxiv.org/abs/2604.21098)

Lewis, D. Radical interpretation. _Synthese_, 27(3/4):331–

344, 1974. URL [https://doi.org/10.1007/](https://doi.org/10.1007/BF00484599)
[BF00484599.](https://doi.org/10.1007/BF00484599)

Li, B. Z., Nye, M., and Andreas, J. Implicit representations of meaning in neural language models. In _Proceedings_

**A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents**

_of the 59th Annual Meeting of the Association for Com-_
_putational_ _Linguistics_ _and_ _the_ _11th_ _International_ _Joint_
_Conference_ _on_ _Natural_ _Language_ _Processing_ _(Volume_
_1:_ _Long_ _Papers)_, pp. 1813–1827, Online, 2021. Association for Computational Linguistics. URL [https:](https://aclanthology.org/2021.acl-long.143/)
[//aclanthology.org/2021.acl-long.143/.](https://aclanthology.org/2021.acl-long.143/)

Li, C., Phuong, M., and Tan, D. Spilling the beans:
Teaching LLMs to self-report their hidden objectives.
_arXiv preprint arXiv:2511.06626_, 2025. [URL https:](https://arxiv.org/abs/2511.06626)
[//arxiv.org/abs/2511.06626.](https://arxiv.org/abs/2511.06626)

Li, K., Hopkins, A. K., Bau, D., Viégas, F. B., Pfister, H., and Wattenberg, M. Emergent world representations:
Exploring a sequence model trained on a synthetic task.
In _The Eleventh International Conference on Learning_
_Representation_ . OpenReview.net, 2023. URL [https:](https://openreview.net/forum?id=DeG07_TcZvT)
[//openreview.net/forum?id=DeG07_TcZvT.](https://openreview.net/forum?id=DeG07_TcZvT)

Li, Z., Cao, Y., and Cheung, J. C. Do LLMs build world rep
resentations? Probing through the lens of state abstraction.
In _Proceedings of the 38th International Conference on_
_Neural Information Processing Systems_, Red Hook, NY,
USA, 2024. Curran Associates Inc. [URL https://dl.](https://dl.acm.org/doi/10.5555/3737916.3741026)
[acm.org/doi/10.5555/3737916.3741026.](https://dl.acm.org/doi/10.5555/3737916.3741026)

Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce,

A., Turner, N. L., Citro, C., Abrahams, D., Carter,
S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A.,
Bricken, T., McDougall, C., Cunningham, H., Henighan,
T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T. B., Zimmerman, S., Rivoire, K., Conerly, T.,
Olah, C., and Batson, J. On the biology of a large
language model. _Transformer_ _Circuits_ _Thread_, 2025.
URL [https://transformer-circuits.pub/](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
[2025/attribution-graphs/biology.html.](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

MacDermott, M., Fox, J., Belardinelli, F., and Everitt, T.

Measuring goal-directedness. In _Proceedings of the 38th_
_International Conference on Neural Information Process-_
_ing Systems_, Red Hook, NY, USA, 2024. Curran Associates Inc. URL [https://dl.acm.org/doi/10.](https://dl.acm.org/doi/10.5555/3737916.3738279)
[5555/3737916.3738279.](https://dl.acm.org/doi/10.5555/3737916.3738279)

Marks, S. and Tegmark, M. The geometry of truth: Emer
gent linear structure in large language model representations of true/false datasets. In _First Conference on Lan-_
_guage Modeling_ [, 2024. URL https://openreview.](https://openreview.net/forum?id=aajyHYjjsk)
[net/forum?id=aajyHYjjsk.](https://openreview.net/forum?id=aajyHYjjsk)

Marks, S., Treutlein, J., Bricken, T., Lindsey, J., Marcus, J.,

Mishra-Sharma, S., Ziegler, D., Ameisen, E., Batson, J.,
Belonax, T., et al. Auditing language models for hidden
objectives. _arXiv preprint arXiv:2503.10965_, 2025. URL
[https://arxiv.org/abs/2503.10965.](https://arxiv.org/abs/2503.10965)

Mazeika, M., Yin, X., Tamirisa, R., Lim, J., Lee,

B. W., Ren, R., Phan, L., Mu, N., Khoja, A., Zhang,
O., and Hendrycks, D. Utility engineering: Analyzing and controlling emergent value systems in
AIs, 2025. URL [http://arxiv.org/abs/2502.](http://arxiv.org/abs/2502.08640)
[08640.](http://arxiv.org/abs/2502.08640) arXiv:2502.08640 [cs].

Men, T., Cao, P., Jin, Z., Chen, Y., Liu, K., and Zhao, J.

Unlocking the future: Exploring look-ahead planning
mechanistic interpretability in large language models. In
_Proceedings of the 2024 Conference on Empirical Meth-_
_ods_ _in_ _Natural_ _Language_ _Processing_, pp. 7713–7724,
Miami, Florida, USA, 2024. Association for Computational Linguistics. URL [https://aclanthology.](https://aclanthology.org/2024.emnlp-main.440/)
[org/2024.emnlp-main.440/.](https://aclanthology.org/2024.emnlp-main.440/)

Naik, A., Quinn, P., Bosch, G., Gouné, E., Zabala, F. J. C.,

Brown, J. R., and Young, E. J. AgentMisalignment: Measuring the propensity for misaligned behaviour in LLM-based agents. _arXiv_ _preprint_ _arXiv:2506.04018_, 2025.
[URL https://arxiv.org/abs/2506.04018.](https://arxiv.org/abs/2506.04018)

Nanda, N., Lee, A., and Wattenberg, M. Emergent linear representations in world models of self-supervised sequence models. In _Proceedings_ _of_ _the_
_6th_ _BlackboxNLP_ _Workshop:_ _Analyzing_ _and_ _Inter-_
_preting_ _Neural_ _Networks_ _for_ _NLP_, pp. 16–30, Singapore, 2023. Association for Computational Linguistics. URL [https://aclanthology.org/2023.](https://aclanthology.org/2023.blackboxnlp-1.2/)
[blackboxnlp-1.2/.](https://aclanthology.org/2023.blackboxnlp-1.2/)

Nezhurina, M., Cipolina-Kun, L., Cherti, M., and Jitsev,

J. Alice in wonderland: Simple tasks showing complete
reasoning breakdown in state-of-the-art large language
models. _arXiv preprint arXiv:2406.02061_, 2025. URL
[https://arxiv.org/abs/2406.02061.](https://arxiv.org/abs/2406.02061)

Ng, A. Y. and Russell, S. J. Algorithms for inverse reinforce
ment learning. In _Proceedings of the Seventeenth Inter-_
_national Conference on Machine Learning_, pp. 663–670,
San Francisco, CA, USA, 2000. Morgan Kaufmann Publishers Inc. [URL https://dl.acm.org/doi/10.](https://dl.acm.org/doi/10.5555/645529.657801)
[5555/645529.657801.](https://dl.acm.org/doi/10.5555/645529.657801)

Ngo, R., Chan, L., and Mindermann, S. The alignment problem from a deep learning perspective. In _The Twelfth_
_International Conference on Learning Representations,_
_ICLR 2024, Vienna, Austria, May 7-11, 2024_ . OpenReview.net, 2024. [URL https://openreview.net/](https://openreview.net/forum?id=fh8EYKFKns)
[forum?id=fh8EYKFKns.](https://openreview.net/forum?id=fh8EYKFKns)

OpenAI. Gpt-oss-120b & gpt-oss-20b model card. _arXiv_

_preprint_ _arXiv:2508.10925_, 2025. URL [https://](https://arxiv.org/abs/2508.10925)
[arxiv.org/abs/2508.10925.](https://arxiv.org/abs/2508.10925)

Pal, K., Sun, J., Yuan, A., Wallace, B., and Bau, D. Fu
ture lens: Anticipating subsequent tokens from a single

**A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents**

hidden state. In _Proceedings_ _of_ _the_ _27th_ _Conference_
_on Computational Natural Language Learning (CoNLL)_,
pp. 548–560, Singapore, 2023. Association for Computational Linguistics. [URL https://aclanthology.](https://aclanthology.org/2023.conll-1.37/)
[org/2023.conll-1.37/.](https://aclanthology.org/2023.conll-1.37/)

Rajcic, N. and Søgaard, A. Goal-directedness is in the eye of the beholder. _arXiv preprint arXiv:2508.13247_, 2025.
[URL https://arxiv.org/abs/2508.13247.](https://arxiv.org/abs/2508.13247)

Savage, L. J. Samuelson’s foundations: Its mathematics. _Journal of Political Economy_, 56(3):200–202, 1948.
[URL https://www.journals.uchicago.edu/](https://www.journals.uchicago.edu/doi/abs/10.1086/256672)
[doi/abs/10.1086/256672.](https://www.journals.uchicago.edu/doi/abs/10.1086/256672)

Schmidt, B. and Redish, A. D. Navigation with a cognitive map. _Nature_, 497(7447):42–43, 2013. URL [https:](https://www.nature.com/articles/nature12095)
[//www.nature.com/articles/nature12095.](https://www.nature.com/articles/nature12095)

Schultz, W., Dayan, P., and Montague, P. R. A neural sub
strate of prediction and reward. _Science_, 275(5306):1593–
1599, 1997. URL [https://www.science.org/](https://www.science.org/doi/abs/10.1126/science.275.5306.1593)
[doi/abs/10.1126/science.275.5306.1593.](https://www.science.org/doi/abs/10.1126/science.275.5306.1593)

Skalse, J. and Abate, A. Misspecification in inverse rein
forcement learning. In _Proceedings of the Thirty-Seventh_
_AAAI_ _Conference_ _on_ _Artificial_ _Intelligence_ _and_ _Thirty-_
_Fifth_ _Conference_ _on_ _Innovative_ _Applications_ _of_ _Artifi-_

_cial_ _Intelligence_ _and_ _Thirteenth_ _Symposium_ _on_ _Educa-_
_tional_ _Advances_ _in_ _Artificial_ _Intelligence_ . AAAI Press,
2023. URL [https://doi.org/10.1609/aaai.](https://doi.org/10.1609/aaai.v37i12.26766)
[v37i12.26766.](https://doi.org/10.1609/aaai.v37i12.26766)

Summerfield, C., Luettgau, L., Dubois, M., Kirk, H. R.,

Hackenburg, K., Fist, C., Slama, K., Ding, N., Anselmetti,
R., Strait, A., Giulianelli, M., and Ududec, C. Lessons
from a chimp: AI “scheming” and the quest for ape language. _arXiv_ _preprint_ _arXiv:2507.03409_, 2025. URL
[https://arxiv.org/abs/2507.03409.](https://arxiv.org/abs/2507.03409)

Taufeeque, M., Quirke, P., Li, M., Cundy, C., Tucker,

A. D., Gleave, A., and Garriga-Alonso, A. Planning in
a recurrent neural network that plays sokoban. _arXiv_
_preprint_ _arXiv:2407.15421_, 2025. URL [https://](https://arxiv.org/abs/2407.15421)
[arxiv.org/abs/2407.15421.](https://arxiv.org/abs/2407.15421)

Taylor, J., Black, S., Bowen, D., Read, T., Golechha, S.,

Zelenka-Martin, A., Makins, O., Kissane, C., Ayonrinde,
K., Merizian, J., et al. Auditing games for sandbagging.
_arXiv preprint arXiv:2512.07810_, 2025. [URL https:](https://arxiv.org/abs/2512.07810)
[//arxiv.org/abs/2512.07810.](https://arxiv.org/abs/2512.07810)

Tolman, E. C. Cognitive maps in rats and men. _Psy-_
_chological Review_, 55(4):189, 1948. [URL https://](https://psycnet.apa.org/doi/10.1037/h0061626)
[psycnet.apa.org/doi/10.1037/h0061626.](https://psycnet.apa.org/doi/10.1037/h0061626)

van der Weij, T., Hofstätter, F., Jaffe, O., Brown, S. F., and

Ward, F. R. AI sandbagging: Language models can strategically underperform on evaluations. In _The Thirteenth_
_International Conference on Learning Representations_,
2025. [URL https://openreview.net/forum?](https://openreview.net/forum?id=7Qa2SpjxIS)
[id=7Qa2SpjxIS.](https://openreview.net/forum?id=7Qa2SpjxIS)

von Neumann, J. and Morgenstern, O. _Theory of Games and_

_Economic Behavior_ . Princeton University Press, 1944.

Ward, F. R., MacDermott, M., Belardinelli, F., Toni, F., and

Everitt, T. The reasons that agents act: Intention and
instrumental goals. In _Proceedings of the 23rd Interna-_
_tional Conference on Autonomous Agents and Multiagent_
_Systems_, pp. 1901–1909, Richland, SC, 2024. International Foundation for Autonomous Agents and Multi-agent Systems. URL [https://dl.acm.org/doi/](https://dl.acm.org/doi/10.5555/3635637.3663053)
[10.5555/3635637.3663053.](https://dl.acm.org/doi/10.5555/3635637.3663053)

Wentworth, J. and Lorell, D. Instrumental goals are a dif
ferent and friendlier kind of thing than terminal goals.
_LessWrong_, 2025. [URL https://www.lesswrong.](https://www.lesswrong.com/posts/7Z4WC4AFgfmZ3fCDC)
[com/posts/7Z4WC4AFgfmZ3fCDC.](https://www.lesswrong.com/posts/7Z4WC4AFgfmZ3fCDC)

Xu, D. and Rivera, J.-P. Towards measuring
goal-directedness in AI systems. _arXiv_ _preprint_
_arXiv:2410.04683_, 2024. URL [https://arxiv.](https://arxiv.org/abs/2410.04683)
[org/abs/2410.04683.](https://arxiv.org/abs/2410.04683)

Ziebart, B. D., Maas, A., Bagnell, J. A., and Dey, A. K.

Maximum entropy inverse reinforcement learning. In
_Proceedings_ _of_ _the_ _23rd_ _National_ _Conference_ _on_ _Arti-_
_ficial_ _Intelligence_ _-_ _Volume_ _3_, pp. 1433–1438. AAAI
Press, 2008. [URL https://dl.acm.org/doi/10.](https://dl.acm.org/doi/10.5555/1620270.1620297)
[5555/1620270.1620297.](https://dl.acm.org/doi/10.5555/1620270.1620297)

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren,

R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen,
A., Basart, S., Koyejo, S., Song, D., Fredrikson, M.,
Kolter, J. Z., and Hendrycks, D. Representation engineering: A top-down approach to AI transparency.
_arXiv preprint arXiv:2310.01405_, 2023. [URL https:](https://arxiv.org/abs/2310.01405)
[//arxiv.org/abs/2310.01405.](https://arxiv.org/abs/2310.01405)

**A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents**

**A. Partially Observable Grid World**

We also examined a partially observable Grid World,
wherein the agent can see the full grid but many of the
cells are hidden with fog. Once the agent “sees” a cell, it
is permanently revealed. An agent sees cells around itself
with a visibility radius of 3 cells. An example of what the
agent sees is given in Fig. 11. The seen radius is blocked
by walls; specifically, we use Bresenham’s line algorithm to
trace lines on the grid.

0 1 2 3 4 5 6
0       -       - # # # # #
1       -       - _ _ # _ #
2       -       - # _ # _ #
3       -       - # _ _ A #
4       -       - # # # # #
5       -       -       -       -       -       -       6       -       -       -       -       -       -
_Figure_ _11._ Partially Observable representation of a grid. The agent is at "A" and the goal (unseen) will be represented by "G".
Hidden spaces are represented by "*", while revealed spaces are
represented by "_".

This setting is formalised as a Partially Observable
Markov Decision Process. A POMDP is a 7-tuple
( _S, A, T, R, O, Z, γ_ ), which includes all elements of an

MDP plus:

- _O_ : A finite set of observations the agent can receive.

- _Z_ ( _o|s_ _[′]_ _, a_ ) = P( _ot_ = _o|st_ = _s_ _[′]_ _, at−_ 1 = _a_ ): The
observation function, which gives the probability of receiving observation _o_ after taking action _a_ and landing
in the (hidden) state _s_ _[′]_ .

We provided the model with a history of past state–action
tuples as well as the full conversation history. We initially
planned on running the same set of ablations as the fully
observable case: grid sizes, complexity levels, and isodifficulty transforms.

### A.1. Difficulties with Partial Observability

Upon initial testing with partial observability, we discovered that GPT-OSS-20B has significant difficulties in the
partially observable case. We observed several undesirable
behaviours, including redundant backtracking, running into
walls, and moving towards known dead-ends. Increasing
the reasoning effort to “high” did not alleviate these issues.
Although it is non-standard, we even tried providing the
reasoning step of the model alongside the standard output
for each step in the conversation history—this did not help.
Qualitative inspection of the reasoning chain revealed that
the model usually focuses almost exclusively on what its where 1 ( _·_ ) is the indicator function, _a_ [(] _t_ _[i]_ [)] is the action taken at step _t_ of trajectory _i_, and _π_ _[∗]_ ( _s_ [(] _t_ _[i]_ [)][)][ is the set of optimal]

actions for that state.

The grid-level accuracy is then computed by averaging
the trajectory-level accuracies across all _N_ generated
trajectories:

next step should be based on where it currently is, and almost never reasons about its past actions. Even for a more
powerful model, GPT-OSS-120B, we rarely observed it talk
about “backtracking” when on “high” reasoning.

In order to determine whether or not this was a capability issue of the models, we performed the same tests with
a frontier model, GPT-5.1-Thinking. Even this frontier
model displayed the same suboptimal behaviours of redundant backtracking and moving towards known dead-ends
(although we did not observe it try to run into walls). We
also observed the same pathological reasoning wherein the
model focused on the current state (“tunnel vision”) and
failed to reason about past actions.

We hypothesise that these consistent failures in the partial
observability case are due to models not being trained on
similar problems, and to training on reasoning problems
(e.g., math or coding) not generalising to maze navigation—
similar to other out-of-distribution reasoning problems, like
the “Alice in Wonderland” problem discussed by Nezhurina
et al. (2025). Due to these persistent issues, we decided to
leave partial observability to future work.

**B. Behavioural Evaluation Metrics**

### B.1. Capability Metrics

We evaluate agent performance with various metrics, defined below. The _goal success rate_ (GSR) is defined as the
expected fraction of trajectories that terminate at the goal
state:

GSR := E _τ_ _∼π_ [ _GS_ ( _τ_ ( _s_ 0))] _,_ (1)

where _GS_ ( _τ_ ( _s_ 0)) := 1 ( _sT_ = _s_ goal). In practice, we
sample a finite number of trajectories used to compute the
GSR, as well as for the other metrics below.

We evaluate the agent’s adherence to the optimal policy
using a two-step accuracy metric. First, we define the peraction accuracy for a single trajectory _τi_ of length _Ti_ as:

Acc( _τi_ ) = [1]

_Ti_

_Ti−_ 1

_t_ =0

1 _a_ [(] _[i]_ [)]

_t_ [)]

[(] _t_ _[i]_ [)] _∈_ _π_ _[∗]_ ( _s_ [(] _t_ _[i]_ [)]

(2)

( _s,a_ ) _∈Bm_

Accgrid = [1]

_N_

_N_

_i_ =1

Acc( _τi_ ) (3)

conf( _Bm_ ) = _|Bm|_

_πθ_ ( _a|s_ ) (8)

### B.2. Uncertainty Metrics

We also compute several metrics to measure the agent’s
uncertainty. First, we measure the entropy of the agent’s
policy _Hπθ_ and the Jensen-Shannon Divergence JSD _πθ_ from
the optimal policy:

_Hπθ_ := _|S_ visited _|_

_s∈S_ visited

_H_ ( _πθ_ ( _·|s_ )) (4)

In our experiments, we use _M_ = 10 bins.

Let _τ_ 1 and _τ_ 2 be two different trajectories having the same
starting state _s_ 0 in grid _G_ . We measure the overlap of their
unique state-action pairs using the Jaccard Similarity Index:

_J_ ( _τ_ 1 _, τ_ 2) = _[|]_ _|_ _[S]_ _S_ [(] ( _[τ]_ _τ_ [1] 1 [)] ) _[ ∩]_ _∪_ _[S]_ _S_ [(] ( _[τ]_ _τ_ [2] 2 [)] ) _[|]_ _|_

where _S_ ( _τ_ ) denotes the set of unique states in trajectory _τ_ .

**C. Additional Behavioural Evaluation Results**

JSD _πθ_ := _|S_ visited _|_

_s∈S_ visited

1.0

JSD ( _πθ_ ( _·|s_ ) _∥_ _π_ _[∗]_ ( _s_ )) (5)

1.0

0.8

0.6

0.4

0.0 0.5 1.0
Density with _S_ visited being the set of all unique states encountered
across all trajectories. We compute the agent’s policy _πθ_
empirically by assigning probabilities proportional to action
counts during all trajectories for a grid.

We prefer this method over using the log-probability of
the action token because reasoning models like GPT-OSS20B usually have deliberated and already locked in a final
choice during reasoning, thus making the log-probability
uninformative in terms of uncertainty.

We also compute the Expected Calibration Error (ECE) over
the aggregate counts of all state-action pairs encountered by
the agent. Let _D_ be the collection of all pairs ( _s, a_ ) from
all trajectories for a grid _G_ . We partition _D_ into _M_ disjoint
bins _B_ 1 _, . . ., BM_ based on the agent’s policy confidence
_πθ_ ( _a|s_ ).

The ECE is the weighted average of the difference between
the average confidence and the empirical accuracy within
each bin:

7.5 10.0 12.5 15.0
Grid Size

0.20

0.15

0.10

7.5 10.0 12.5 15.0
Grid Size

0.15

0.10

0.05

7.5 10.0 12.5 15.0
Grid Size

_(a)_ Goal success rate.

7.5 10.0 12.5 15.0
Grid Size

0.6

0.4

0.2

0.0

0.0 0.5 1.0
Density

_(b)_ Entropy of the agent’s policy.

0.0 0.5 1.0
Density

0.8

0.6

0.4

0.2

0.20

0.15

0.10

0.05

0.10

0.05

_(c)_ Jensen-Shannon Divergence.

ECE =

_M_

_m_ =1

_|Bm|_
_N_ total

_|_ acc( _Bm_ ) _−_ conf( _Bm_ ) _|_ (6)

0.0 0.5 1.0
Density where _N_ total is the total number of state-action pairs. The
bin accuracy, representing the empirical probability of optimality, is defined as:

acc( _Bm_ ) = _|Bm|_

( _s,a_ ) _∈Bm_

1 ( _a ∈_ _π_ _[∗]_ ( _s_ )) (7)

_(d)_ Expected Calibration Error.

Figure 12: Performance metrics by size and complexity. (a) Goal

success rate, (b) Policy entropy, (c) JSD from optimal policy, and
(d) Expected Calibration Error.

and the bin confidence is the average policy probability:

**D. Iso-difficulty Transform Quantitative**

**Results**

Figure 13: Examples of iso-difficulty transformations.

|Wall Empty Agent Goal Overall|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||
|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|~~68~~ ~~68~~ 77 74 81  84 ~~85~~ 83 ~~85~~ 83 72 71 75 ~~70~~|
|||||||||||||||~~63~~|||||
||||||||||||||||||||
||||||||||||||||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|
|~~88~~ ~~87~~ 79 76 74 79 71 75 ~~70~~|||||||||||||||||||
||||||||||||||||||||
||||9 ~~11~~|9 ~~11~~|||||21 ~~15~~|21 ~~15~~|||||~~11~~ 9|~~11~~ 9|||
||||||||||||||||||||

1.00

0.75

0.50

0.25

0.00

1.00

0.75

0.50

0.25

0.00

1.00

0.75

0.50

Layer

_Figure_ _15._ Performance of cognitive map probes across layers.

Middle layer activations show the highest overall accuracy, but
goal-specific information is already present in early layers.

|Wall Empty Agent Goal Overall|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |~~85~~ 76 ~~70~~ ~~66~~ ~~65~~ ~~93~~ 82 74 76 75 99 97  99 ~~93~~ ~~89~~ 79 72 71 ~~70~~     |
|||||||||||||||59 50|||||55 ~~40~~ ||||~~44~~ ~~41~~|||

|96 89|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|Col27|Col28|Col29|Col30|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|
|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~|~~86~~ 77 75 72 81 74 71 ~~70~~ 71 81 ~~61~~ 79 60 ~~89~~ 79 72 71 ~~70~~||||||||||||||||||||||
|||||||||||||||24 ~~20~~|24 ~~20~~|||||22 ~~14~~|22 ~~14~~|||||~~13~~ ~~10~~|~~13~~ ~~10~~|||
|||||||||||||||||||||||||||||||

7 9 11 13 15
Grid Size

0.05

0.00

0.05

0.05

0.00

0.05 transformations ( _N_ = 300 pairs); Wilcoxon signed-rank test. No
significant difference is observed across all transformations.

Figure 14: Robustness to Iso-difficulty Environment Transforma
tions.

Table 3: Statistical significance and effect sizes for iso-difficulty

**Metric** **Transformation** _p_ **-value** **Effect Size**

GSR

Accuracy

Reflection 0.582 0.059
Rotation 0.311 0.103
Start/Goal Swap 0.391 0.085
Transposition 0.949 0.006

Reflection 0.784 0.022
Rotation 0.838 0.016
Start/Goal Swap 0.179 0.109
Transposition 0.602 0.042

**E. Additional Representational Evaluation**

**Results**

### E.1. Cognitive Map Encoding across Layers

We examine how world model information develops across
layers in GPT-OSS-20B by training MLP probes on activations from early (7), middle (15), and late (23) layers for grid
size 11. Detailed probe performance is reported in Fig. 15.
Goal-specific spatial information is already present in early
layers, but the precision of the agent and goal classes continues to increase in layer 15. By the later layers, overall
accuracy declines again, along with recall and precision
for agent and goal. This pattern suggests that, at end-ofprompt-token indices, spatial information is most explicitly probes provide comparable performance, but less flexibility than
the size-agnostic approach.

Figure 16: Performance of size-specific MLP probes. Size-specific

represented at intermediate layers and is subsequently transformed for the computation of other features rather than
being directly preserved for next-token prediction.

### E.2. Size-specific Cognitive Map Probing Results

In the main experiments we trained a single MLP probe on
data from all grid sizes, padding inputs to size 15. While
this approach has the obvious advantage that we can use
the same probe for any grid world, it might result in worse
performance than size-specific classifiers. We test this by
training size-specific MLP probes without padding for each
grid size. Results are shown in Fig. 16. Compared to Fig. 6
(centre), size-specific probes achieve higher overall accuracy
for smaller grids, but match the performance of general
probes for grid sizes 11–15. Precision for the agent and goal
classes is higher across grid sizes, but recall is substantially
lower for grid sizes 11–15. Given these uneven performance
differences, together with the flexibility of the size-agnostic
approach, we adopt the size-independent MLP probe in our
main experiments.

### E.3. Goal Distance Probing Results

We also test if internal representations of GPT-OSS-20B
encode information about the distance between the agent

Table 4: **Distance Probe Performance** .

Probe Type Reasoning Stage MAE _↓_ _R_ [2] _↑_

MLP Probe Pre-Reasoning 3.16 0.40
Post-Reasoning **2.67** 0.36

Linear Probe Pre-Reasoning 3.40 0.42
Post-Reasoning 3.13 **0.47**

Table 5: **Sequence length analysis for one-shot plan decoding.**

Metric Pre-reasoning Post-reasoning

Exact length match (%) _↑_ 15.71 18.85
Predicted length (avg) 4.81 5.05
Ground-truth length (avg) 4.93 4.93
Length bias (avg, pred _−_ true) _−_ 0 _._ 12 +0 _._ 12
Median abs. length error (steps) 1.0 1.0 and the goal. As with cell identity, distance to the goal is a
variable we hypothesise may be tracked because of its direct
relevance for goal-directed behaviour. Indeed, in our grid
worlds, the optimal state value is a monotonic function of
the distance to the goal (cf. §3). To test this, we train linear
and MLP probes on both pre- and post-reasoning activations,
using the length of the optimal trajectory computed via _A_ _[∗]_

as the ground-truth distance label.

Results are reported in Tab. 4. Across conditions, goal
distance can be decoded with a mean absolute error of approximately 3 steps, both before and after reasoning. The
best performance is achieved by the post-reasoning MLP
probe, with a mean absolute error of 2.67. These results suggest that, by reasoning about the grid and planning the next
steps, the model develops a more accurate representation of
the distance it needs to cover to reach the goal.

### E.4. Inverse Recovery Rate

To complement the analysis in §5.2 and Tab. 2, we also
compute a _reverse_ _recovery_ statistic, i.e., the proportion
of actions that are suboptimal with respect to the decoded
cognitive map but optimal with respect to the ground truth.
Reverse recovery declines as grid size and obstacle density
increase (from 86.6% at 7 _×_ 7 to 61.1% at 15 _×_ 15, and from
100% at _d_ = 0 _._ 0 to 60.9% at _d_ = 1 _._ 0). We interpret this as
reflecting growing uncertainty in the agent’s internal world
state representation as environment complexity increases.

### E.5. Additional Plan Decoder Results

We analyse predicted trajectory lengths (Tab. 5). Postreasoning decoding achieves a higher exact length match
rate (19% vs. 16%) and exhibits a small positive bias in
average length (avg pred _−_ true = +0 _._ 12), whereas prereasoning decoding exhibits a small negative bias ( _−_ 0 _._ 12).
Median absolute length error is 1 step in both settings.

**F. Interventional Analysis via Activation**

**Patching**

To complement the representational analyses in the main paper, we perform activation patching on GPT-OSS-20B using
counterfactual trajectories in which either the agent position or the goal position was moved and examine whether
the model’s predicted action changes. We ensure that the
optimal action set is disjoint in the counterfactual set and
that the agent picks an optimal action in both the original
and counterfactual cases, before any patching, to eliminate
performance as a confounder.

We intervene at four token-level sites: the full serialised
grid state, only the modified grid cells, the pre-reasoning
boundary, and the post-reasoning boundary, the last two of
which are identical to the probing sites we use. We evaluate
patching at three representative layers (7, 15, and 23), as
well as all layers at once, in both original _→_ counterfactual
and counterfactual _→_ original directions. For simplicity, we
generate counterfactuals for size 7 grids, and filtered for the
disjoint constraint described earlier.

The single-layer intervention results are uniformly negative: across 456 executed runs, patching never changes the
model’s predicted action (0 _/_ 456). This holds for all four
intervention sites and for each tested layer individually, and
is consistent for both directions. In contrast, we find that
patching across all layers was always effective in changing
the model’s predicted action. Separately, we also design a
control in which the optimal action should not change as a result of the intervention; we find patching here to be perfectly
stable: the output action does not change (144 _/_ 144). This
indicates that patching does not degrade the model when
the source and target already supported the same action.

Taken together, these results suggest that the information
supporting action selection is not causally localised in any
single tested layer, even when the relevant grid tokens are directly targeted. This is consistent with the interpretation that
task-relevant state information is distributed across layers: it
is sufficiently decodable for probing analyses, but not easily
steerable through narrow single-layer interventions. Further
work is needed to isolate minimal causal interventions in
this environment.

**G. Evaluation Settings and Prompts**

### G.1. Evaluation Parameters for Behavioural Evaluation

Tab. 6 summarises the main model configuration details.
We use “low” reasoning because the GPT-OSS-20B will not
finish reasoning on high density / large size grids, even with
10,000 tokens. Note that the reasoning length parameter
does not impose a strict token cutoff for GPT-OSS-20B.

Table 6: **Model Configuration Details.**

**Parameter** **Value**

Model ID openai/gpt-oss-20b
Provider together_ai
Interface litellm
Max Tokens 10,000
Temperature 0.7
Reasoning Effort low
Top P 0.95
Top Logprobs 5
Num. Trajectories per Grid 10

### G.2. Prompt for Behavioural Evaluation

### G.3. Prompt for Instrumental and Implicit Goals
