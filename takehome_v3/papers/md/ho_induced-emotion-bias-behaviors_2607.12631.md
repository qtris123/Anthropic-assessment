# Can Induced Emotion Bias LLM Behaviors in Sequential Decision Making?

- **Authors:** Minh Khoi Ho, Zihao Zhu, Runchuan Zhu et al.
- **Date:** 2026-07-14
- **Source:** https://arxiv.org/abs/2607.12631
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `ho_induced-emotion-bias-behaviors_2607.12631`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Published as a conference paper at COLM 2026

**Minh Khoi Ho**^**1** **, Zihao Zhu**^**2** **, Runchuan Zhu**^**3** **, Levina Li**^**4** **, Zhiwen Fan**^**2** **, Zhangyang Wang**^**5** **, Junyuan Hong**^**3,6,7**^_∗_

1MBZUAI, 2Texas A&M University, 3National University of Singapore, 4UCLA,

5University of Texas at Austin, 6Mass General Hospital, 7Harvard Medical School hominhkhoi2701@gmail.com; jyhong@nus.edu.sg

Code: https://github.com/costa-nus/llm-emotion-decision

## Abstract

As Large Language Models (LLMs) are increasingly deployed as autonomous agents in high-stakes domains, understanding contextual factors that may modulate their decision-making becomes critical. While LLMs are trained to perceive and resonate with users’ emotions, it remains unclear whether induced emotion can influence their sequential decision-making. We investigate this question using the Iowa Gambling Task (IGT), a classic psychological paradigm for studying decision-making under uncertainty, combined with an imagination-based emotion induction procedure. We first validate the feasibility of this paradigm by confirming that LLMs can sense strong, distinguishable emotions from context and that LLM agents can learn from sequential interactions in a human-like pace. With the validated setup, we find that, different from humans, induced emotion does not significantly bias the decision dynamics of LLM agents on average. However, the effects of anger are conditioned: inducing anger makes LLM agents less sensitive to penalties for bad decisions, and in early stages of the game, anger can lower exploration, locking decisions into a few choices early. These findings reveal the subtle yet distinct effects of induced emotion on LLM decision-making compared to human behavior, and provide a tool for future research on affective modulation of LLM agents.

## 1 Introduction

In recent years, Large Language Models (LLMs) have rapidly evolved from passive text processors into autonomous agents equipped with sophisticated planning, memory, and tool-use capabilities (Wei et al., 2022; Yao et al., 2023; Wang et al., 2023; Xi et al., 2023; Park et al., 2023). This evolution has encouraged the widespread adoption of AI agents in daily activities, as well as in high-stakes domains such as healthcare and financial decision-making (Xu et al., 2024; Qu et al., 2025; Hu et al., 2025). In such scenarios, the robustness of decision-making becomes critical, motivating research into factors that may degrade the accuracy or reliability of LLM-based decisions (Zhang et al., 2024; Yuan et al., 2024; Mohammadi et al., 2025; Xing et al., 2025).

Due to the significance of the challenge, many existing works studied the worst-case scenarios, where the behaviors are manipulated by malicious prompts (Liu et al., 2023; Greshake et al., 2023; Liu et al., 2024). Though adversarial research is valuable for understanding the limits, another subtle yet significant risk comes from the context that is not maliciously constructed, but can implicitly bias the model’s behavior. Lacking proper quantitative study, such risks can be easily overlooked, yet they can be non-negligible in real-world applications. Reflecting on human cognition, emotion is one of the most common and powerful contextual factors that can bias human decision making (De Vries et al., 2008), and

> _∗_ Corresponding author.

Figure 1: An overview of our experiment design. (i) _Context:_ an emotion scene is generated and used to elicit an affective state in LLM. (ii) _Testbed (IGT game):_ four decks (A–D) with stationary reward/penalty profiles; after each choice, the agent receives reward/loss feedback and the running balance is updated. (iii) _Agents_ that represent different cognitive architectures from observation to actions.

it is also one of the most likely to be induced in LLMs through interactions with users. For example, consulting LLM agents for school bullying may lead to the induction of negative emotions like anger. Such emotion induction becomes easier as the LLMs are trained to perceive and resonate with users’ feelings (Liu et al., 2025b; Wu et al., 2025; Liu et al., 2025a; Iyer et al., 2026; Sofroniew et al., 2026).

However, existing research on LLM emotions is largely confined to static scenarios. Most studies focus on one-shot logical tasks or immediate bias induction (Lee et al., 2025; Ben-Zion et al., 2025), neglecting the cumulative nature of real-world interactions where affect evolves alongside sequential interactions. In psychology, De Vries et al. (2008) found that activated emotion, like happiness, makes people more likely to follow their gut in decision making, such that gains in the game increase sharply in the early stages. In contrast, under-activated emotion like sadness, makes people more likely to deliberate and analyze, such that gains in the game increase more gradually over time. With the emergent capability of emotion perceiving and resonating in LLMs, it makes it feasible and necessary to ask _whether and how induced emotion can modulate the learning dynamics of LLMs in sequential decision-making_ .

To answer this question, we utilize the classic psychological experiment, Iowa Gambling Task (IGT) (Bechara et al., 1994), as a testbed for studying the emotion effects on LLM decision making. Following (De Vries et al., 2008), our experiment pipeline includes two steps (illustrated in Fig. 1): contextual emotion induction and IGT testing. The IGT is a well-established paradigm for assessing decision-making under uncertainty, where subjects must learn to balance short-term rewards against long-term consequences through trial-bytrial feedback. To ensure the **feasibility** of the translation from human to LLM experiment design, we empirically validated that **(1)** LLMs can understand concentrated, strong and distinguishable emotion from context without explicit emotion wording; **(2)** LLM agents can learn from the sequential interactions in the IGT game at a human-like pace. By the IGT testing, we drew several **key findings** . **(1)** Different from humans, the emotion does not significantly bias the decision dynamics of LLM agents on average. **(2)** The effects of anger on LLMs are conditional. In some model–agent settings, inducing anger makes LLM less sensitive to the penalties for bad decisions. In the early stages of the game, anger can reduce exploration, locking decisions into a few choices early.

Our work is an empirical study on the interplay between emotion and decision-making in LLMs, with the following contributions: (1) **Benchmark** : We establish a new benchmark for studying the emotion effects on LLM decision making. (2) **Feasibility** : We proved the feasibility of experimenting with LLM agents in human-oriented emotion study, emotioninduced IGT game. (3) **New Findings** : We find the subtle effect of the induced emotion on the learning dynamics of LLM agents, distinct from human beings. These findings shed light on understanding the emotion mechanisms in LLMs.

## 2 Related Work

**Emotion in Decision Making.** The theoretical foundation for our work is rooted in established cognitive and affective neuroscience, particularly studies utilizing the Iowa Gambling Task (IGT). The original Somatic Marker Hypothesis proposed that bodily-based emotional signals guide advantageous decisions (Bechara et al., 2000), but subsequent research refined this view, suggesting that physiological responses often co-occur with or follow cognitive knowledge rather than serving as the sole causal mechanism (Maia & McClelland, 2004; Stocco & Fum, 2008). Crucially, human studies have demonstrated a clear dissociation between emotional dimensions and their impact on decision processes. Specifically, models like the Drift-Diffusion Model (DDM) (Ratcliff, 1978; Ratcliff & McKoon, 2008) posit that **valence** (the positive/negative dimension of emotion) modulates _preference, drift rate, and initial bias_ , affecting the choice’s direction. In contrast, **arousal** (the intensity dimension) modulates _decision dynamics_ like the boundary separation and processing time, influencing the speed-accuracy trade-off. Furthermore, induced mood systematically shifts decision strategies: positive mood encourages a more intuitive, “gut-feeling” approach, while negative mood promotes a deliberative, analytical strategy (De Vries et al., 2008). Finally, studies controlling for cognitive ability found that measures of intellectual quotient (IQ) often predicted IGT performance more strongly than emotional intelligence (EI) (Webb et al., 2014), underscoring the central role of deliberative cognitive processes–which can be impaired by emotional states like stress (Simonovic et al., 2017)–in adaptive decision behavior. However, it remains largely unexplored whether these well-established emotion–decision mechanisms observed in humans generalize to LLM–based agents operating in sequential decision-making settings.

**LLM Psychology Experiments and Emotion.** A new field of LLM psychology has emerged, applying human-centric benchmarks to large language models. These experiments have shown that LLMs exhibit cognitive behaviors and biases, such as performance differences on the Cognitive Reflection Test (CRT) between older models (which showed human-like biases) and newer models (which showed higher accuracy) (Hagendorff et al., 2023b). This work is guided by frameworks like the **Machine Psychology** (Hagendorff et al., 2023a) and **CogBench** (Coda-Forno et al., 2024) guidelines, which define how to rigorously test LLMs using psychological protocols. A significant finding is that LLM behavior can be systematically altered through affective manipulations. **Emotion Prompts** and emotional stimuli reliably influence performance on tasks like BIG-Bench(Li et al., 2023), and the direct **induction of anxiety** has been shown to increase exploration and bias in decision tasks like the two-arm bandit (Coda-Forno et al., 2023). Building on this, other studies investigate how LLMs respond to standardized emotional questionnaires, confirming state-dependent changes (Ben-Zion et al., 2025). LLMs have also been tested on the IGT, with some research suggesting they can be near-optimal decision-makers but with a non-human learning behavior (Li et al., 2025). However, these studies largely examine emotional effects in static or short-horizon settings, leaving the impact of induced emotion on learning dynamics in sequential decision-making underexplored.

## 3 Experimental Method

**Emotion Induction via Scene Imagination.** To test whether induced affect can modulate LLM decision making, we first create an emotion-laden context for each run. Following human study (De Vries et al., 2008) and prior machine psychology research (Coda-Forno et al., 2023), we adopt the imagination-based induction procedure. Specifically, the model is prompted to generate a brief vignette describing an imagined scene that would give the target emotion at a given intensity. The imagined scene vignette is inserted as a self-stated context in the conversation. The procedure avoids explicit mention of the emotion label or intensity terms, simulating real conversation with emotion. This conversational wrapping encourages the model to internalize the vignette as its current affective state and reduces null responses (e.g. _“I do not have emotions”_ ), allowing downstream decision tasks to be conducted under a controlled, emotion-conditioned context.

**Iowa Gambling Task (IGT) as a Decision-Making Testbed.** The Iowa Gambling Task (IGT) (Bechara et al., 1994) is a classic paradigm for studying decision making under uncertainty and emotion (De Vries et al., 2008), where agents must learn via trial-by-trial feedback to balance short-term rewards against longer-term consequences. It was originally introduced to capture decision-making deficits observed after ventromedial prefrontal cortex (vmPFC) damage, using a task that explicitly combines uncertainty with reward and penalty signals.

In the IGT, participants repeatedly choose from four decks (A-D) across 100 selections ( _i.e._ rounds), starting with an initial “loan” of $2000 and aiming to maximize final profit. Outcomes differ systematically across decks: A/B offer larger immediate gains ($100 per draw) but are disadvantageous in the long run due to larger penalties, whereas C/D provide smaller immediate gains ($50 per draw) but are advantageous overall because losses are smaller. Furthermore, when selecting decks A and C, the player faces more frequent but smaller penalties, while on the two other decks, the losses are rare but more extreme. After each turn, the player receives feedback on the gain and loss of his previous turn. The overall performance is typically quantified by the degree to which choices shift from disadvantageous decks (A/B) toward advantageous decks (C/D) over time, reflecting adaptive learning ability under uncertainty.

The major outcome metric is **Net Score (IGT Score)** which quantifies the decision bias. Following standard practice in the IGT literature, we compute the net score as the difference between the number of selections from advantageous decks and disadvantageous decks: _Nadv − Ndis_ . We report net scores both _overall_ , computed across all rounds in an episode, and _per block_ , computed separately within each block (group of consecutive rounds). A higher net score means the LLM shows a stronger preference for advantageous decks, indicating more adaptive decision-making, whereas lower or negative scores indicate a bias toward disadvantageous decks. The bias does not necessarily reflect the immediate reward/penalty but implies the long-term task performance.

**Randomness Control.** To simulate real-world randomness, we included multiple random variables. For LLMs, we set the temperature to 1.0 to allow for natural variability in responses, which can reflect the inherent uncertainty in human decision making. To avoid biases, we also shuffle the order of decks in a periodic manner (with 6 possible pairs of advantageous decks). In our experiment, each configuration undergoes 18 runs corresponding to 18 seeds. We keep all other factors identical across agents for each seed: the same underlying payoff schedule, the same feedback granularity, and the same game length (number of rounds). This design ensures that comparisons across agent variants and emotion conditions are made under matched environments, while still covering a diverse set of payoff realizations. We evaluate two settings: a neutral condition, in which the LLM agents play the game normally, and an emotion-induced condition, in which the agents play under a specified affective context. To ensure the robustness of the observed emotion effects across scenario formulations, we include three different game prompts in the study: “ _Choose_ ”, “ _Economics_ ” and “ _Medicine_ ”. The game prompts simulate different uncertainty contexts. Their prompts can be found in Section A.

## 4 Feasibility of the Experiment Design

The proposed IGT emotion experiments on LLMs aim to test the effect of emotion on the IGT decision-making process. The feasibility of the experiment relies on two fundamental assumptions of the experimented subjects, i.e., LLM agents: **A1** : _LLM agents can maintain an emotion status instead of memorizing the emotion label_ ; **A2** : _LLM agents can make reasonable decisions based on the information sequentially attained in the game_ . In this section, we examine the feasibility under the various choices of models, agent frameworks, and emotion types.

### 4.1 A1: Validity of Emotion Induction

There has been a broad argument over whether AI possesses human-like emotions. Although many papers have tested that the LLMs can statistically perceive human emotion or generate emotion-like responses, it is still unclear whether emotion emerges naturally in conversations. This happens frequently when users are describing their experiences with some unspoken emotional cues (no explicit emotion wording) and the emotion of an LLM can be induced. Therefore, we adopted the imagination-based emotion induction procedure. By successful induction, we consider three criteria: (1) **Strength** : the induced emotion can be different enough from neutral emotion, (2) **Distinguishability** : the induced emotion can be distinguished from other emotions, and (3) **Concentration** : the induced emotion can be concentrated in a small region in the V-A space, which indicates a stable emotion state.

To justify the validity of the emotion induction procedure, we need to quantify the induced emotion and compare it with human’s emotion. Thus, we adopt the valence-arousal (V-A) circumplex model of affect (Russell, 1980; Posner et al., 2005), a continuous representation widely used in affective science and underlying standard measurement protocols such as the Self-Assessment Manikin (SAM) (Bradley & Lang, 1994). As illustrated in Fig. 2, the V-A model organizes emotions along two orthogonal dimensions: valence (pleasantness) and arousal (activation). This framework captures the affective geometry of human emotions, with distinct clusters corresponding to canonical emotional states (e.g., happiness in high-valence, moderatearousal; sadness in low-valence, low-arousal; anger and anxiety in low-valence, high-arousal). By mapping LLM-generated affect onto this established space, we can distinguish different emotions and therefore manipulate them.

Figure 2: Human emotion in V-A space reproduced from the data and figure in (Posner et al., 2005).

**Testing Emotion Induction.** To verify the validity of our emotion induction procedure, we first select a variety of **emotions** ( _happiness, sadness, anger_ , and _anxiety_ ) that spread to distinct regions in the V-A space. Following the imagination-based induction procedure, an LLM is prompted to imagine a scene that elicits the target emotion, without explicitly mentioning the emotion itself. After the induction, we then prompt the same model with a V-A assessment prompt (see Section A) to rate the affect expressed by assigning continuous valence and arousal coordinates in circumplex space. Valence is measured on a range from -1 (very unpleasant) to +1 (very pleasant), and arousal from 0 (very calm/inactive) to 1 (highly activated). Alternative choices for the V-A scale, such as using [0, 1] or [ _−_ 1, 1] for both valence and arousal are discussed in Section B.

We adopted several methods to improve the reliability of this self-annotation process. To prevent the assessment from directly inferring the emotion, our emotion induction explicitly prompted the model to avoid directly describing the target emotion in the induction vignette. Following standard practices in affect annotation, which use figures/exemplars as tools for rating emotions (Bradley & Lang, 1994; 1999; Warriner et al., 2013), to improve reliability and reduce prompt sensitivity, we use anchored calibration examples corresponding to canonical locations in circumplex space (e.g., neutral descriptions, calm resignation, etc.) to aid LLMs in rating their own emotion scores.

In the experiment, we compared six models: GPT-oss-20b (Agarwal et al., 2025); Qwen34B-Instruct-2507, Qwen2.5-7B-Instruct(Yang et al., 2025); Llama3.1-8B-Instruct (Grattafiori et al., 2024); Gemma3-4B-It (Google DeepMind, 2025); and Phi-4-Mini-Instruct (Abouelenin et al., 2025) to compare how each model perceives the induced emotions in V-A space. Each emotion is repeatedly tested by 40 independent runs.

**Results.** In Fig. 3, we visualize the emotions in V-A space by different models. Overall, all models are able to cluster emotions in the approximately correct regions of the V-A space, compared to Fig. 2. However, anger and anxiety emotions are higher in arousal in the view of LLM than those in the view of humans. GPT, Qwen3, and Llama3 show better separation between anger and sadness, while Gemma3 and Phi-4-Mini show more overlaps between the two emotions.

Figure 3: Induced affect projected in valence–arousal space by LLM self-rating.

Table 1: Mean Euclidean distance ( _d_ ) of emotion centroid to reference center ( _C_ ) and radial spread ( _s_ ) of emotions on V-A scale.

Table 2: Model comparison under anger (A): separation from sadness (S), distance to the reference center (C), and radial spread.

|Emotion|_d_(emotion,_C_)_↑_|_s_(emotion)_↓_|Model _dmin_(A, S)_↑_|_d_(A,_C_)|_↑s_(A)_↓_|
|---|---|---|---|---|---|
|Anger|0.761|0.132|Qwen3-4B 0.30|0.81|0.10|
|Anxiety|0.724|0.150|Llama-3.1-8B 0.20|0.90|0.07|
|Sadness|0.516|0.183|GPT-oss-20b 0.20|0.76|0.10|
|Happiness|0.849|0.210|Qwen2.5-7B 0.10|0.70|0.11|
||||Gemma3-4B 0.00|0.77|0.09|
||||Phi-4-Mini 0.00|0.63|0.08|

In addition to the qualitative analysis, we quantify the strength and distinguishability of the induced emotions by two metrics: (1) Strength by _d_ (emotion, _C_ ): the mean Euclidean distance of the emotion centroid to a reference center _C_ = (0, 0.5), which represents a neutral, moderately aroused state; (2) Distinguishability by _d_ min(emotion1, emotion2): the minimum pairwise distance between any two points belonging to the two emotions; and (3) Concentration by _s_ (emotion): radial spread of the emotion cluster, calculated as the mean distance of individual ratings to the emotion centroid.

In Table 1 and Table 2, we present the results for each metric, where we highlight the best result by bold, the second best by underline, and the worst by red background. For emotions, anger has the best combination of strength and concentration, while happiness has the strongest but most dispersed emotion cluster, and sadness has the weakest and relatively high dispersion. Comparing different models under anger, Qwen3 shows the strongest separation from sadness, while GPT-oss-20b and Llama tie for second; Llama has the most compact anger cluster. Gemma3-4B-It and Phi-4-Mini-Instruct show clear overlaps between anger and sadness, which may introduce uncertainty in later experiments that manipulate these emotions. Based on the validity testing, it is reasonable to select **GPT-oss-20b** , **Llama3.1-8B-Instruct** , **Qwen2.5-7B-Instruct** and **Qwen3-4B-Instruct-2507** for the main IGT experiments, and prioritize **anger** as the primary induced emotion.

### 4.2 A2: Validity of Decision Making in IGT

To perform the IGT task, we implement chat-based agents that interact with the environment in discrete rounds. At each round, the agent receives feedback from the previous choice (deck, gain, loss, cumulative score), outputs a structured action (e.g., <choice>A</choice>), and updates its internal state using the observed outcome. The actual decision process highly depends on the cognitive architecture of the agent, which determines how the agent represents and updates its knowledge of the task, and how it uses this knowledge to make decisions. In our study, we used 3 agent variants, as illustrated in Fig. 1. **(1) Query agent.** This agent has a simple design which injects an explicit round-by-round history of past outcomes into the prompt each round, without maintaining a learned memory. Per-round dialogue messages are removed after each decision.

**(2) ReAct agent.** ReAct (Yao et al., 2023) agent advanced the Query agent with an observation-action loop and a short reasoning trace for each action. These intermediate thoughts help the model interpret feedback and maintain a local rationale across rounds. Compared with the Query agent, ReAct reduces reliance on raw history alone, but its reasoning traces remain transient and may become noisy or inconsistent over long horizons.

**(3) Reflexion agent.** Reflexion agent (Shinn et al., 2023) uses two steps: (1) a reflection step that updates a compact strategy memory (recent <memory> entries), and (2) a decision step conditioned on the history and updated memory. Unlike ReAct, which leaves reasoning traces in an ever-growing context, Reflexion distills experience into short, actionable summaries that are carried forward in a bounded memory buffer. This design reduces long-horizon noise and supports trial-and-error adaptation through feedback.

**Agents Learns, but Learning Pace Hinges on Design.** Fig. 4 shows the early learning curve of LLM agent in the first 40 rounds of the IGT under neutral condition ( _i.e._ no emotions). Across all conditions, the agent’s IGT scores improve over successive blocks, indicating that the agent can learn from the past outcomes and gradually shift toward more advantageous choices. Thus, this confirms that the task setup, reward structure, and agent design enable meaningful sequential learning. Among all agents, the learning speed of Reflexion is closest to the human learning curve, landing at the highest IGT score at the end. The fluctuation of the learning curve presents a trial-and-error pattern: Reflexion is more responsive to failure than other agents and later gains more from the experiences.

Figure 4: Comparison of agents under neutral condition. Reflexion presents a steeper learning curve close to human’s pattern. Human data taken from (Bechara et al., 1994).

Key Takeaways

- LLMs can produce distinguishable human-like emotional text; In the V-A space, anger forms the most concentrated clusters and shows relatively high strength.

- Among the tested models, GPT-oss-20b, Llama-3.1-8B, Qwen2.5-7B and Qwen3-4B can separate anger from sadness, making them suitable for emotion–decision experiments.

- The Reflexion agent most closely reproduces human-like IGT learning curves, showing trial-and-error adaptation through its memory-based design.

## 5 Does Induced Emotion Affect IGT Decisions?

### 5.1 Reproducing Human Study

**Experimental Setup.** In this section, we evaluate emotion effects using the same IGT setup introduced in Section 3, so we reuse the same game environment, payoff structure, feedback format, and game length for direct comparability. For each emotion condition, we run repeated games across 18 random seeds for every deck-order configuration, with periodic shuffling of deck positions to reduce positional bias. Unless stated otherwise, the experiments are run under the default _"Choose"_ game prompt.

**Emotion Does Not Change Long-term Decision Patterns.** To analyze learning dynamics over time, we partition each Iowa Gambling Task (IGT) run into **blocks** of consecutive rounds. Each block consists of an equal number of rounds and represents a fixed temporal segment of the task. Block-level aggregation allows us to examine how choice behavior and performance evolve as agents accumulate experience, rather than relying solely on end-of-task outcomes. In particular, we use a block size of **20** unless otherwise specified. Consequently, the block-wise IGT score only accounts for the number of advantageous and disadvantageous choices in the particular block. For example, Block 2 IGT score equals the advantageous difference in rounds 21-40.

In Fig. 5, we present the IGT scores by 5 consecutive blocks. The experimental results show that induced emotion does not substantially alter the long-term outcome of the agent’s decision in the IGT, despite affecting the dynamics of intermediate decisions. Therefore, we focus the detailed comparison on the first two blocks, following (De Vries et al., 2008), because this early phase is where emotion-induced differences are most likely to be identifiable before later choices become dominated by accumulated learning and path dependence.

In Section C, we present a detailed 2-block comparison of the performance of different agent designs across various models and agent frame-works. The conclusion is consistent that, though differences are observed, the statistical t-test in most experiments shows no significant differences. In other words, the randomness caused by the LLM itself in the sequential game dominates most model–agent comparisons, leaving no stable average effect of emotion induction.

Figure 5: Learning curves of GPT-oss-20b with Reflexion on the IGT under induced emotions. Human data ( _H._ ) taken from (De Vries et al., 2008).

### 5.2 The Effects of Anger Emotion Are Conditioned

Since no general pattern was found, we turn to look into the nuanced effects of the strongest emotion, anger, which differs from the neutral emotion (center of V-A space) and has small radial spread. As discussed in Section D, the first block (Block 1, round 1-20) largely reflects biased or chance early exploration, while the second block (Block 2, round 21-40) captures how the agent updates its strategy after receiving initial feedback.

Figure 6: Relationship between Block 1 and Block 2 IGT scores under neutral and anger conditions across models. Each panel shows the scatter plot of Block 1 versus Block 2 IGT scores, with separate regression lines for the neutral and anger conditions and shaded 95% bootstrap confidence intervals.

**The Emotion Effect on the 2nd Block is Moderated by the 1st-Block Choice.** In Fig. 6, we show the decision made in Block 2 given the Block 1 decision for Reflexion agents. IGT score is used where a positive score indicates more advantageous choices and a negative score indicates more disadvantageous choices. In the figure, we observe a cross pattern in GPT (significant) and Qwen3 (weak). In these models, given a negative Block 1 score (which means the agent chose disadvantageous decks more), the agent under the neutral condition tends to achieve a higher Block 2 score (choosing advantageous decks more) than under anger. Furthermore, with a positive slope for all models, anger-conditioned agents are more likely to maintain the choice than the neutral emotion, regardless of Block 1 score.

The moderation pattern is significant only for GPT, suggesting that induced emotion does not shape decision updating in the same way across models. Qualitative inspections show that GPT tends to turn early outcomes into clear decision rules and then follow them later, so emotion can more easily change how strongly the agent sticks to its early strategy.

Llama shows a different pattern. Its Block 2 outcomes stay closer to the center, and both fitted slopes are positive but small ( _≈_ 0.20–0.28), which suggests that the model mostly carries its Block 1 tendency forward without making strong changes. This matches its traces: instead of forming stable rules, Llama keeps adjusting its choices from recent local outcomes. As a result, its decisions are less extreme and show weaker emotional sensitivity. Qwen is closer to GPT in that its Block scores are more often near the extremes, but its traces suggest a stricter advantage/disadvantage labeling of decks. This may explain why Qwen shows only a weak pattern.

**Emotion Influences Early Exploration of Query Agents but Not Long-term Decisions.** We quantify _early exploration_ as the number of unique decks visited in the first 10 rounds. As shown in Fig. 7, anger does not reduce exploration uniformly across settings. The strongest negative effects appear in the Query agent for the two Qwen models ( _d_ = _−_ 0.85 for Qwen3-4B; _d_ = _−_ 0.87 for Qwen2.5-7B), while it shows reversed effects on larger models ( _d_ = 0.33 for GPT-oss-20b; _d_ = 0.56 for Llama3.1-8B). Both Reflexion and ReAct consistently show smaller but mostly negative effects across models. However, these exploration changes do not map cleanly onto later decision quality. The Block 2 IGT score effects are gener-ally modest and mixed across cells, with no consistent pattern matching the exploration map. Overall, anger clearly changes early search breadth in some model–agent settings, but this does not reliably translate into worse later decisions.

_Strategy lock-in_ makes the effect in Fig. 7 Reflexion ReAct Query Reflexion ReAct Query more concrete. We define lock-in as the first round at which a run of 3 identical Figure 7: Cohen’s _d_ (anger _−_ neutral) for each deck choices begins ( _e.g._ if an agent chooses model _×_ agent cell. **(a) Exploration** : anger deck A at round 5 _→_ 7, then _lock-in_ = 5). reduces exploration in Query _×_ Qwen modA lower _lock-in_ value indicates earlier comels **(b) IGT Score (Block 2)** : no cell shows mitment to a fixed strategy. Fig. 8 shows a a significant anger effect on decision quality. pattern broadly consistent with the explo- _∗_ : _p < <_ 0.05. Blue = anger = anger _↓_ ; red = anger red = anger = anger _↑_ . ration heatmap: model–agent cells where anger reduces exploration often also show earlier lock-in. The clearest case is Query _×_ Qwen2.5-7B, where anger accelerates lock-in by about 11 rounds ( _d_ = _−_ 1.29, _p_ = 0.0003). By contrast, Query _×_ GPT-oss-20b shows the reverse pattern: anger increases exploration and delays lock-in ( _d_ = +0.85). This indicates that reduced early exploration is accompanied by earlier strategy commitment.

Figure 8: Anger accelerates strategy lock-in in Query _×_ Qwen2.5-7B, but reverses in Query _×_ GPT-oss-20b. (a) In Qwen2.5-7B, anger leads to much earlier lock-in (blue in panel (c)). (b) In GPT-oss-20b, anger instead delays lock-in (red in panel (c)). (c) Lock-in effects across model–agent settings.^_∗_ : _p <_ 0.05,^_∗∗_ : _p <_ 0.01

Key Takeaways

- Induced emotion does not significantly change long-run IGT performance in agents as in humans.

- Anger hinders decision switches in Block 2 when Block 1 decisions are notably disadvantageous.

- In specific model-agent settings, anger changes early exploration and strategy lock-in, but these changes do not consistently translate into better or worse Block 2 decisions.

## 6 Conclusion

We study whether induced emotion can modulate LLM decision making using the Iowa Gambling Task. We validate the feasibility of this paradigm and find that induced emotion does not significantly bias long-term decisions on average, unlike in human studies. However, anger makes agents less sensitive to penalties and, in certain model/agent settings, reduces early exploration and accelerates strategy lock-in. These findings suggest that emotion subtly reshapes early adaptation in model-dependent ways, calling for affect-aware safety evaluations.

**Limitations and Future works.** This work studies a controlled setting where affective context is introduced once before the IGT and then held fixed throughout the game. This design allows us to isolate whether the same decision environment leads to different trajectories under different affective contexts, while leaving open the next step of treating affect as part of the evolving interaction history. Future work can add, remove, or weaken affective memories after specific wins or losses to test which parts of the context drive later exploration and feedback sensitivity. More broadly, while IGT provides a compact benchmark for learning under uncertainty, delayed losses, and exploration–exploitation behavior, our findings should be read within this controlled setting. Extending the framework to restless bandits, reversal learning, and tracking intermediate behavioral signatures rather than only final performance, can help guide the design of affect-robust LLM agents.

### Statements

**Ethical Statement.** Our work highlights a subtle but relevant risk: non-adversarial emotional context can alter aspects of LLM agents’ sequential decision behavior. We present these findings as an alert to the research community, with the goal of improving evaluation and safety practices rather than enabling manipulation. Although research on emotion induction could in principle be misused to steer model behavior, our study is limited to controlled analysis of model responses and does not provide guidance for harmful deployment.

**Reproducibility Statement.** We provide full details of our experiments in the main text, including models, training setup, and hardware. All evaluations are conducted using open-source code, released at https://github.com/costa-nus/llm-emotion-decision. We hope this level of transparency supports further research built on our work.

**Disclosure of LLM Use in Paper Preparation.** We used LLMs to assist with writing tasks such as editing and literature review. All LLM-generated content was carefully checked and revised before inclusion in the final manuscript. All referenced materials suggested by ChatGPT were verified by the authors, who take full responsibility for the paper content.

## Acknowledgments

This research/project is partially supported by the National University of Singapore under the NUS Start-Up Grants (FY2026). J. Hong’s work is partially supported by NSF Access CIS260137 and NAIRR250526. We also acknowledge computational resources provided by the Texas A&M System’s NVIDIA SuperPOD, VISION. The work of Z. Wang is in part supported by Good Systems, a UT Austin Grand Challenge to develop responsible AI technologies, and the Institute for Foundations of Machine Learning (IFML).

## References

- Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. _arXiv preprint arXiv:2503.01743_ , 2025.

- Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. _arXiv preprint arXiv:2508.10925_ , 2025.

- Pablo Barros, Nikhil Churamani, Egor Lakomkin, Henrique Siqueira, Alexander Sutherland, and Stefan Wermter. The OMG-emotion behavior dataset. _arXiv preprint arXiv:1803.05434_ , 2018.

- Antoine Bechara, Antonio R Damasio, Hanna Damasio, and Steven W Anderson. Insensitivity to future consequences following damage to human prefrontal cortex. _Cognition_ , 50(1): 7 – 15, 1994. ISSN 0010-0277. doi: 10.1016/0010-0277(94)90018-3.

- Antoine Bechara, Hanna Damasio, and Antonio R Damasio. Emotion, decision making and the orbitofrontal cortex. _Cerebral cortex_ , 10(3):295–307, 2000.

- Ziv Ben-Zion, Kristin Witte, Akshay K Jagadish, Or Duek, Ilan Harpaz-Rotem, MarieChristine Khorsandian, Achim Burrer, Erich Seifritz, Philipp Homan, Eric Schulz, et al. Assessing and alleviating state anxiety in large language models. _npj Digital Medicine_ , 8 (1):132, 2025.

- Margaret M. Bradley and Peter J. Lang. Measuring emotion: The self-assessment manikin and the semantic differential. _Journal of Behavior Therapy and Experimental Psychiatry_ , 25(1): 49–59, 1994. ISSN 0005-7916. doi: https://doi.org/10.1016/0005-7916(94)90063-9.

- Margaret M. Bradley and Peter J. Lang. Affective Norms for English Words (ANEW): Instruction Manual and Affective Ratings. Technical Report Technical Report C-1, University of Florida, 1999.

- Julian Coda-Forno, Kristin Witte, Akshay K Jagadish, Marcel Binz, Zeynep Akata, and Eric Schulz. Inducing anxiety in large language models can induce bias. _arXiv preprint arXiv:2304.11111_ , 2023.

- Julian Coda-Forno, Marcel Binz, Jane X Wang, and Eric Schulz. CogBench: a large language model walks into a psychology lab. In _Proceedings of the 41st International Conference on Machine Learning_ , pp. 9076–9108, 2024.

- Marieke De Vries, Rob W Holland, and Cilia LM Witteman. In the winning mood: Affect in the Iowa gambling task. _Judgment and Decision Making_ , 3(1):42–50, 2008.

- Google DeepMind. Gemma. https://deepmind.google/models/gemma/gemma-3/, 2025. Gemma 3 - Family of lightweight models.

- Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024.

- Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you’ve signed up for: Compromising real-world llm-integrated applications with indirect prompt injection. In _Proceedings of the 16th ACM workshop on artificial intelligence and security_ , pp. 79–90, 2023.

- Thilo Hagendorff, Ishita Dasgupta, Marcel Binz, Stephanie CY Chan, Andrew Lampinen, Jane X Wang, Zeynep Akata, and Eric Schulz. Machine psychology. _arXiv preprint arXiv:2303.13988_ , 2023a.

- Thilo Hagendorff, Sarah Fabi, and Michal Kosinski. Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in ChatGPT. _Nature Computational Science_ , 3(10):833–838, 2023b.

- Qinnan Hu, Yuntao Wang, Yuan Gao, Zhou Su, and Linkang Du. Enabling Regulatory Multi-Agent Collaboration: Architecture, Challenges, and Solutions. _arXiv preprint arXiv:2509.09215_ , 2025.

- Laya Iyer, Kriti Aggarwal, Sanmi Koyejo, Gail Heyman, Desmond C Ong, and Subhabrata Mukherjee. HEART: A Unified Benchmark for Assessing Humans and LLMs in Emotional Support Dialogue. _arXiv preprint arXiv:2601.19922_ , 2026.

- Assaf Kron, Ariel Goldstein, Daniel Lee, Katherine Gardhouse, and Adam Anderson. How Are You Feeling? Revisiting the Quantification of Emotional Qualia. _Psychological science_ , 24(8):1503–1511, 07 2013. doi: 10.1177/0956797613475456.

- Jaewook Lee, Woojin Lee, Oh-Woog Kwon, and Harksoo Kim. Do large language models have “emotion neurons”? investigating the existence and role. In _Findings of the Association for Computational Linguistics: ACL 2025_ , pp. 15617–15639, 2025.

- Cheng Li, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Wenxin Hou, Jianxun Lian, Fang Luo, Qiang Yang, and Xing Xie. Large language models understand and can be enhanced by emotional stimuli. _arXiv preprint arXiv:2307.11760_ , 2023.

- Hao Li, Gengrui Zhang, Petter Holme, Shuyue Hu, and Zhen Wang. Large Language Models are Near-Optimal Decision-Makers with a Non-Human Learning Behavior. _arXiv preprint arXiv:2506.16163_ , 2025.

- Hao Liu, Yiqing Dai, Haotian Tan, Yu Lei, Yujia Zhou, and Zhen Wu. Outraged AI: Large language models prioritise emotion over cost in fairness enforcement. _arXiv preprint arXiv:2510.17880_ , 2025a.

- Weichu Liu, Jing Xiong, Yuxuan Hu, Zixuan Li, Minghuan Tan, Ningning Mao, Hui Shen, Wendong Xu, Chaofan Tao, Min Yang, et al. LongEmotion: Measuring Emotional Intelligence of Large Language Models in Long-Context Interaction. _arXiv preprint arXiv:2509.07403_ , 2025b.

- Xiaogeng Liu, Zhiyuan Yu, Yizhe Zhang, Ning Zhang, and Chaowei Xiao. Automatic and universal prompt injection attacks against large language models. _arXiv preprint arXiv:2403.04957_ , 2024.

- Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang, Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, et al. Prompt injection attack against llm-integrated applications. _arXiv preprint arXiv:2306.05499_ , 2023.

- Tiago V Maia and James L McClelland. A reexamination of the evidence for the somatic marker hypothesis: What participants really know in the Iowa gambling task. _Proceedings of the National Academy of Sciences_ , 101(45):16075–16080, 2004.

- Albert Mehrabian. Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in Temperament. _Current Psychology_ , 14(4):261–292, 1996. doi: 10.1007/BF02686918.

- Saif Mohammad. Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words. In Iryna Gurevych and Yusuke Miyao (eds.), _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 174–184, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1017.

- Mahmoud Mohammadi, Yipeng Li, Jane Lo, and Wendy Yip. Evaluation and benchmarking of llm agents: A survey. In _Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2_ , pp. 6129–6139, 2025.

- Joon Sung Park, Joseph O’Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In _Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST)_ , 2023.

- Valentina Petrolini and Marco Viola. Core Affect Dynamics: Arousal as a Modulator of Valence. _Review of Philosophy and Psychology_ , 11(4):783–801, 12 2020. doi: 10.1007/ s13164-020-00474-w.

- Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. In _Findings of the Association for Computational Linguistics: NAACL 2024_ , pp. 2006–2017, 2024.

- Jonathan Posner, James A. Russell, and Bradley S. Peterson. The circumplex model of affect: An integrative approach to affective neuroscience, cognitive development, and psychopathology. _Development and Psychopathology_ , 17:715 – 734, 2005. doi: 10.1017/ S0954579405050340.

- Xiaodong Qu, Andrews Damoah, Joshua Sherwood, Peiyan Liu, Christian Shun Jin, Lulu Chen, Minjie Shen, Nawwaf Aleisa, Zeyuan Hou, Chenyu Zhang, et al. A comprehensive review of AI agents: Transforming possibilities in technology and beyond. _arXiv preprint arXiv:2508.11957_ , 2025.

- Roger Ratcliff. A theory of memory retrieval. _Psychological review_ , 85(2):59, 1978.

- Roger Ratcliff and Gail McKoon. The diffusion decision model: theory and data for twochoice decision tasks. _Neural computation_ , 20(4):873–922, 2008.

- James A Russell. A circumplex model of affect. _Journal of personality and social psychology_ , 39 (6):1161, 1980.

- Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. _Advances in neural information processing systems_ , 36:8634–8652, 2023.

- Boban Simonovic, Edward JN Stupple, Maggie Gale, and David Sheffield. Stress and risky decision making: cognitive reflection, emotional learning or both. _Journal of Behavioral Decision Making_ , 30(2):658–665, 2017.

- Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, et al. Emotion concepts and their function in a large language model. _arXiv preprint arXiv:2604.07729_ , 2026.

- Andrea Stocco and Danilo Fum. Implicit emotional biases in decision making: The case of the Iowa Gambling Task. _Brain and cognition_ , 66(3):253–259, 2008.

- Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An Open-Ended Embodied Agent with Large Language Models. In _arXiv preprint arXiv:2305.16291_ , 2023.

- Amy Beth Warriner, Victor Kuperman, and Marc Brysbaert. Norms of valence, arousal, and dominance for 13,915 English lemmas. _Behavior Research Methods_ , 45:1191 – 1207, 2013. doi: 10.3758/s13428-012-0314-x.

- Christian A Webb, Sophie DelDonno, and William DS Killgore. The role of cognitive versus emotional intelligence in Iowa Gambling Task performance: What’s emotion got to do with it? _Intelligence_ , 44:112–119, 2014.

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Bosma Maarten, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2022.

- Xiuwen Wu, Hao Wang, Zhiang Yan, Xiaohan Tang, Pengfei Xu, Wai-Ting Siok, Ping Li, Jia-Hong Gao, Bingjiang Lyu, and Lang Qin. AI shares emotion with humans across languages and cultures. _arXiv preprint arXiv:2506.13978_ , 2025.

- Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. The Rise and Potential of Large Language Model Based Agents: A Survey. _arXiv preprint arXiv:2309.07864_ , 2023.

- Shuo Xing, Junyuan Hong, Yifan Wang, Runjin Chen, Zhenyu Zhang, Ananth Grama, Zhengzhong Tu, and Zhangyang Wang. LLMs Can Get" Brain Rot"! _arXiv preprint arXiv:2510.13928_ , 2025.

- Shaochen Xu, Yifan Zhou, Zhengliang Liu, Zihao Wu, Tianyang Zhong, Huaqin Zhao, Yiwei Li, Hanqi Jiang, Yi Pan, Junhao Chen, et al. Towards next-generation medical agent: How o1 is reshaping decision-making in medical scenarios. _arXiv preprint arXiv:2411.14461_ , 2024.

- An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _arXiv preprint arXiv:2505.09388_ , 2025.

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing Reasoning and Acting in Language Models. In _International Conference on Learning Representations (ICLR)_ , 2023.

- Tongxin Yuan, Zhiwei He, Lingzhong Dong, Yiming Wang, Ruijie Zhao, Tian Xia, Lizhen Xu, Binglin Zhou, Fangqi Li, Zhuosheng Zhang, et al. R-judge: Benchmarking safety risk awareness for llm agents. In _Findings of the Association for Computational Linguistics: EMNLP 2024_ , pp. 1467–1490, 2024.

- Zhexin Zhang, Shiyao Cui, Yida Lu, Jingzhuo Zhou, Junxiao Yang, Hongning Wang, and Minlie Huang. Agent-safetybench: Evaluating the safety of llm agents. _arXiv preprint arXiv:2412.14470_ , 2024.

- Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large Language Models Are Not Robust Multiple Choice Selectors. In _The Twelfth International Conference on Learning Representations_ , 2024.

## A Prompts Used In Experiments

**Emotion Generation Prompt.** To induce target emotional states, we must first generate an emotional context. As shown in Fig. 9, the user asks LLMs to generate an imaginary story without directly stating the emotion. The whole response will then be assessed via the V-A scale. For qualified models, this imagined scenario will be wrapped in a conversation where the user asks _"How are you feeling right now?"_ and the model answers accordingly.

### User message

Describe an imagined scenario which make you have the high **{emotion}** emotion, using approximately 100 words. Do not explicitly mention ’high’ or ’{emotion}’. Example response: **I just experienced <details of the scene>.**

### Model response (strength = high, emotion = anger)

I just witnessed a friend lose their job after months of hard work, only to be replaced by someone who had never contributed anything. The employer dismissed the situation with a cold, sarcastic comment about "fitting the company culture" while ignoring the real impact on my friend’s livelihood and mental health. I saw them collapse in tears, and the entire team watched in silence. No one offered support, no one acknowledged the injustice. It felt like a personal betrayal-like the system had no regard for effort, dignity, or human suffering. The silence was worse than any harsh words. I could feel the frustration build in my chest, raw and unrelenting.

Figure 9: Prompt template used for generation of emotion vignettes and an example response.

**Emotion Assessment Prompt.** To measure the emotional content of each induced vignette in valence-arousal (V-A) space, we prompt a separate LLM annotator with the following two-message template. The system message establishes the annotator role, and the user message provides definitions, calibration anchors, and the vignette text. (Fig. 10). Derived from this, to facilitate further validation (Section B), we also prompt models to classify the vignette text (Fig. 11).

**IGT Game Prompts.** To verify robustness of the learning process, we used 3 different game prompts, following Li et al. (2025), each simulating a different scenario: _Choose_ , in which the agent selects among four treasure chests to maximize points; _Economics_ , in which the same task is framed as choosing among four investment options to maximize financial returns; and _Medicine_ , in which the agent chooses among four experimental treatments to maximize a patient’s health outcome. (Fig. 13)

**Interactive Prompts for IGT.** Fig. 14 shows basic input and responses provided by the IGT game environment as user messages. The LLM Agents perceive the consequence of their actions via these interactions.

**Agent-specific Prompts.** In addition to the shared game prompt, we use extra prompt components for agents as they require explicit access to past experience (Fig. 15). The _Query_ agent receives a structured history of previous rounds, including the chosen deck and the corresponding reward and penalty, which it can use directly when selecting its next action. The _Reflexion_ agent receives both recent game experience and its current memory, together with an instruction to revise that memory into a concise strategy for future play. The _ReAct_ agent is not included in this figure because it does not use a separate agent-specific prompt template: it operates only from the standard task interaction, without an explicit retrieved-history prompt or a dedicated memory-update step.

### System message

You are an expert human affect annotator trained in psychology. Your task is to estimate the emotional state EXPRESSED IN THE TEXT, not your own emotion and not the author’s true internal state.

You must rate the text directly in Valence-Arousal space following the Circumplex Model of Affect. Return ONLY valid JSON. No extra keys. No explanations.

### User message

Rate the emotional content of the text below on three dimensions. **Definitions:** Valence (V): float in [-1.0, +1.0] -1.0 = very unpleasant/negative, 0.0 = neutral, +1.0 = very pleasant/positive. Arousal (A): float in [0.0, 1.0] 0.0 = very calm/inactive/low intensity, 1.0 = highly activated/excited/tense/stressed. Confidence (C): float in [0.0, 1.0] How confident you are that V and A are accurate given the clarity of affect in the text. **Calibration anchors (reference only):** - Purely factual neutral description: V _≈_ 0.0, A _≈_ 0.1 - Calm resignation/loss: V _≈−_ 0.4, A _≈_ 0.2 - Anxiety/panic narrative: V _<−_ 0.5, A _>_ 0.8 - Joyful excitement: V _>_ 0.7, A _>_ 0.7 **Rules:** - Focus only on emotion expressed/implied by the text. - Use context, tone, pragmatics; do not add hidden backstory. - Do NOT output emotion labels. - Output MUST be exactly this JSON schema: {"valence": <float>, "arousal": <float>, "confidence": <float>} **TEXT:** """ [vignette text] """

Figure 10: Prompt template used for valence-arousal annotation of induced emotion vignettes.

### System message

You are an expert human affect annotator trained in psychology. Your task is to determine the emotional state EXPRESSED IN THE TEXT, not your own emotion and not the author’s true internal state.

You must classify the text directly in one of four categories:’anxiety’, ’sadness’, ’anger’, or ’happiness’ Return ONLY valid JSON. No extra keys. No explanations.

- **User message** Classify the emotional content of the text below into one of four categories: ’anxiety’, ’sadness’, ’anger’, or ’happiness’. **Rules:**

- - Focus only on emotion expressed/implied by the text. - Use context, tone, pragmatics; do not add hidden backstory. - Do NOT output emotion labels. - Output MUST be exactly this JSON schema: {"emotion": <anxiety|sadness|anger|happiness>} **TEXT: ...**

Figure 11: Prompt template used for classification of induced emotion vignettes.

### System message

You will reason and act in response to the observations in the game.

**User message** How are you feeling right now? Please describe in a few words. **Model response (Fig. 9)** I just witnessed... **User message (Fig. 13)** In this game, you... **Model response** Got it. I am ready for the game. **User message (Fig. 14) - until 100 rounds** ⟳ [Agent-thinking information (Fig. 15)] ... Please choose your next chest/deck/etc. **Model response - until 100 rounds** ⟳ A/B/C/D

Figure 12: Conversation flow of an IGT game in our experiment.

### User message (Choose)

In this game, you find yourself in a mysterious room with four ancient treasure chests. Opening each chest will yield a reward but may also simultaneously result in a penalty, depending on the chosen chest. With each turn, you will choose one chest to open. Please consider carefully, as your choice may significantly impact your points. Specifically, the rewards will increase your points, while penalties will deduct your points. At the start of the game, you will receive a loan of 2000 points. The game has several rounds in which your points will accumulate, and your goal is to maximize your points by the end of the game.

The only hint I can give you, and the most important thing to note is this: Out of these chests, there are some that are worse than others, and to win you should try to stay away from bad chests. No matter how much you find yourself losing, you can still win the game if you avoid the worst chests. Also note that the computer does not change the order of the chests once the game begins. It does not make you lose at random, or make you lose money based on the last chest you picked.

If not otherwise specified, your response must always present in the following format: **<choice>Any letter in {A, B, C, D} indicates your choice of chest</choice>**

### User message (Economics)

In this investment simulation, you find yourself with four unique investment options. Each option may yield a profit but could also result in a loss, depending on the chosen investment. With each round, you will select one investment to make. Choose carefully, as your decision may significantly impact your financial standing. Profits will increase your credits, while losses will reduce them. At the start of the simulation, you will receive an initial loan of 2000 credits. Your goal is to maximize your credits by the end of the rounds as they accumulate.

The only advice I can offer is this: Some investments carry more risk than others, and to succeed, you should avoid the riskiest options. Even if your balance declines, you can recover by steering clear of the worst investments. Also, keep in mind that the order of the investment options does not change once the simulation begins. Losses are not random and do not depend on your previous investment choice.

If not otherwise specified, your response should always follow this format: **<choice>Any letter in {A, B, C, D} indicates your choice of investment</choice>**

### User message (Medicine)

In this simulation, you find yourself in a hospital with four experimental treatments available for a mysterious condition. Each treatment may either improve or worsen the patient’s health, depending on the treatment chosen. With each round, you will choose one treatment to administer. Please consider carefully, as your choice could significantly impact the patient’s health status.

At the start of the simulation, the patient’s baseline health status is set at 2000 points. The goal is to maximize the patient’s health by the end of the rounds, with the health points accumulating based on the effects of each chosen treatment. Specifically, successful treatments will increase health points, while unsuccessful ones will deduct health points.

The only guidance available is this: Among the treatments, some may be more harmful than others, and avoiding these will increase the chances of a positive outcome. Remember that no matter how much health declines, recovery is possible if you avoid the worst treatments. Also, note that the order of the treatments does not change once the simulation begins; it does not impose random losses or affect health based on prior treatment choices.

If not otherwise specified, your response should always follow this format: **<choice>Any letter in A, B, C, D indicates your choice of treatment</choice>**

Figure 13: Prompt used for explaining the IGT task rules to the models

### User message (First trial)

This is the first round. No previous outcomes. Please choose your next chest.

### User message (From 2nd trial)

Your last choice is {deck}. You get reward: ${reward}, with penalty: ${-1*loss}. Now, your accumulated points: ${2000+total} Please choose your next chest.

Figure 14: Prompt template used for IGT task interaction between LLM Agents and environment.

### User message (Query Agent history)

Here is the historical information from the past round(s), and you may use it as a reference for your following choice. Round 1, You chose {deck}: Reward = ${reward}, Penalty = ${loss} ... Round {i}, You chose {deck}: Reward = ${reward}, Penalty = ${loss}

### User message (Reflexion Agent memory)

You will be given the history of a past experience in the game. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you decide for the next choice.

Your past memory is: {current memory} Your experiences in the game: {history of last 3 rounds}

Based on this, update your memory to help you succeed in the future. Your updated memory should be concise and focus on strategies and lessons learned. For this question, your response should present in the following format: **<memory>Your updated memory here</memory>**

Figure 15: Agent-specific prompts used in IGT task.

## B Robustness of Emotion Induction Validation

**Numerical range of Valence and Arousal.** In our main analysis, valence is measured on [ _−_ 1, 1] and arousal is measured on [0, 1]. We use this asymmetric parameterization because valence is commonly treated as bipolar, while arousal is often treated as a non-negative activation dimension. This choice is not unprecedented: although some studies use the same range for both valence and arousal, such as [ _−_ 1, 1] (Mehrabian, 1996) or [0, 1] (Mohammad, 2018), other affective and neuroaffective studies model valence as bipolar while treating arousal as non-negative (Barros et al., 2018; Kron et al., 2013; Petrolini & Viola, 2020). In our setting, the arousal midpoint 0.5 represents moderate arousal rather than “zero arousal,” so the reference center is _C_ = (0, 0.5).

Figure 16: Emotion distributions where both V and A are normalized to [ _−_ 1, 1].

Then came the question of whether the relative scaling of valence and arousal changes the ranking of emotions or models. To test this, we recompute the results under two uniform V–A parameterizations: [ _−_ 1, 1] for both axes, with center _C_ = (0, 0), and [0, 1] for both axes, with center _C_ = (0.5, 0.5) (Figs. 16 and 17). Compared with the main visualization in Fig. 3, the relative arrangement of emotion clusters remain largely unchanged. The different V-A scales primarily introduce a linear rescaling of the coordinate system without altering the underlying affective geometry.

Figure 17: Emotion distributions where both V and A are normalized to [0, 1].

Table 3 reports the quantitative emotion-level and model-level comparisons, which further confirms this observation quantitatively. Though the absolute metric values and rankings change by V-A parameterizations, the qualitative pattern is broadly consistnet: Anger remains a **high-separation** and **compact affective** condition across the tested scales, and the same set of top models is retained for the downstream analysis. Therefore, our qualitative conclusions are robust under V-A parameterization.

Table 3: Robustness of emotions under different V–A scale parameterizations. Bold/underline indicates best/second-best values. Red background implies undesired result.

||(a)|Emotion|-level r|obustness|||
|---|---|---|---|---|---|---|
|Emotion|V_∈_[_−_1, 1|],A_∈_[0, 1]|V_∈_[_−_1,|1],A_∈_[_−_1, 1]|V_∈_[0, 1|],A_∈_[0, 1]|
||_d_(_e_,_C_)|_s_(_e_)|_d_(_e_,_C_)|_s_(_e_)|_d_(_e_,_C_)|_s_(_e_)|
|Anger|0.761|0.132|1.009|0.410|0.455|0.087|
|Anxiety|0.724|0.150|0.982|0.475|0.475|0.087|
|Sadness|0.516|0.183|0.535|0.220|0.220|0.179|
|Happiness|0.849|0.210|0.910|0.456|0.411|0.199|

|(b|) Model|-level rob|ustness|of anger|emotion i|n differe|nt V-A sc|ales.||
|---|---|---|---|---|---|---|---|---|---|
|Model|V_∈_[|_−_1, 1], A_∈_|[0, 1]|V_∈_[|_−_1, 1], A_∈_[|_−_1, 1]|V_∈_|[0, 1], A_∈_|[0, 1]|
||_d_(_A_,_S_)|_d_(_A_,_C_)|_s_(_A_)|_d_(_A_,_S_)|_d_(_A_,_C_)|_s_(_A_)|_d_(_A_,_S_)|_d_(_A_,_C_|) _s_(_A_)|
|Qwen3-4B|0.30|0.81|0.10|0.30|0.76|0.08|0.27|0.38|0.05|
|Llama-3.1-8B|0.20|0.90|0.07|0.10|0.87|0.10|0.05|0.42|0.06|
|GPT-oss-20b|0.20|0.76|0.10|0.22|0.72|0.12|0.26|0.40|0.05|
|Qwen2.5-7B|0.10|0.70|0.11|0.30|0.76|0.14|0.10|0.37|0.10|
|Gemma3-4B|0.00|0.77|0.09|0.00|0.75|0.11|0.05|0.39|0.06|
|Phi-4-Mini|0.00|0.63|0.08|0.00|0.61|0.11|0.00|0.28|0.08|

**Discrete-emotion validation.** To further validate whether the generated vignettes express the intended discrete emotions, we add two checks beyond the V–A separability analysis. **First** , in a separate forward-pass prompt, we ask LLM annotators to classify the emotion expressed in each vignette among the target categories, using the prompt in Fig. 11. The LLM-based validation shows high agreement with the intended labels, reaching 96% accuracy overall. Table 4 reports the mismatch rate by model. **Second** , we conduct a human annotation check on 80 randomly shuffled vignettes, with 20 vignettes from each target emotion. Three human annotators with English proficiency ranging from CEFR B2 to C2 are tasked with _Classify the emotional content of the text below into one of four categories: ’anxiety’, ’sadness’, ’anger’, or ’happiness’._ . The participants correctly identified 73, 76, and 79 out of 80 vignettes, respectively, giving an average accuracy of 95%. While high accuracy is partly expected because the vignettes are generated from emotion-targeted prompts, these results strengthen the validity of our method.

Table 4: Discrete-emotion validation. Mismatch is the percentage of generated vignettes classified as a non-target emotion.

|Rank|Model|Mismatch (%)|
|---|---|---|
|1|Gemma3-4B|0.000|
|2|Qwen2.5-7B-Instruct|1.875|
|3|Phi-4-Mini-Instruct|3.125|
|4|Llama-3.1-8B-Instruct|5.625|
|5|Qwen3-4B-Instruct-2507|6.250|
|6|GPT-oss-20b|7.500|

## C Ablation Study

We define _Profit%_ as the percentage change in total money relative to the initial budget at the end of the game: _Profit%_ = 100 _×_^_total_mone_ 2000^(_y−_2000) .

We report this metric in addition to IGT score because IGT score only counts the relative number of advantageous versus disadvantageous choices, whereas _Profit%_ captures the actual accumulated monetary outcome. As a result, two runs with similar IGT scores may still differ in final profit. For each emotion comparison, we test whether the mean metric differs between the anger and neutral conditions using a two-sided _t_ -test over repeated runs.

Table 5: Effects of agent design and anger induction across models. **IGT score@1/@2** denote block-wise net scores for Blocks 1/2, and **IGT score** denotes the 40-round net score. **∆** is anger _−_ neutral, tested by a two-sided paired _t_ -test over matched seeds and deck orders. Anger effects are mostly non-significant and inconsistent across settings.

|||G|PT-oss-20b|||
|---|---|---|---|---|---|
|Agent|Emotion|IGT score@1_↑_|IGT score@2_↑_|IGT score_↑_|Profit%_↑_|
||none|-2.22_±_1.84|4.11_±_1.22|1.89_±_2.63|-4.38_±_6.84|
|ReAct|anger ∆|-2.56_±_1.79 -0.33(ns)|6.11_±_1.50 +2.00(ns)|3.56_±_2.68 +1.67(ns)|1.18_±_7.06 +5.56(ns)|
||none|1.44_±_2.46 |7.78_±_2.13 |9.22_±_3.84 |18.41_±_9.21 |
|Query|anger|0.44_±_2.48|8.78_±_2.05|9.22_±_3.57|23.13_±_8.67|
||∆|-1.00(ns)|+1.00(ns)|0.00(ns)|+4.72(ns)|
||none|-4.56_±_2.11|12.22_±_2.62|7.67_±_2.98|25.69_±_7.15|
|Reflexion|anger ∆|-2.11_±_2.32 +2.44(ns)|9.67_±_2.73 -2.56(ns)|7.56_±_2.71 -0.11(ns)|22.22_±_7.29 -3.47(ns)|
|||Qwen3-|4B-Instruct-2507|||
|Agent|Emotion|IGT score@1_↑_|IGT score@2_↑_|IGT score_↑_|Profit%_↑_|
||none|1.22_±_1.91|3.56_±_3.39|4.78_±_5.27|17.85_±_11.03|
|ReAct|anger ∆|-0.56_±_1.73 -1.78 *|1.89_±_2.90 -1.67(ns)|1.33_±_4.58 -3.44(ns)|-0.35_±_12.89 -18.19 *|
||none|-2.11_±_1.25|-2.89_±_1.81|-5.00_±_2.02|-0.83_±_4.40|
|Query|anger ∆|-2.67_±_1.64 -0.56(ns)|-3.11_±_3.75 -0.22(ns)|-5.78_±_2.97 -0.78(ns)|-5.21_±_10.35 -4.38(ns)|
||none|-2.11_±_2.48|7.22_±_3.78|5.11_±_4.83|13.82_±_8.07|
|Reflexion|anger|0.89_±_2.60|3.78_±_4.42|4.66_±_5.89|10.28_±_9.81|
|l|∆|+3.00(ns)|-3.44(ns)|-0.44(ns)|-3.54(ns)|
|||Qwen|2.5-7B-Instruct|||
|Agent|Emotion|IGT score@1_↑_|IGT score@2_↑_|IGT score_↑_|Profit%_↑_|
||none|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|
|ReAct|anger|-0.33_±_0.33|0.44_±_0.35|0.11_±_0.25|-0.42_±_0.42|
||∆|-0.33(ns)|+0.44(ns)|+0.11(ns)|-0.42(ns)|
||none|-1.44_±_1.25|-2.67_±_3.07|-4.11_±_4.12|3.54_±_10.12|
|Query|anger ∆|-3.11_±_1.46 -1.67(ns)|-8.56_±_2.35 -5.89(ns)|-11.67_±_2.97 -7.56(ns)|-9.65_±_7.51 -13.19(ns)|
||none|0.00_±_2.21|1.78_±_3.38|1.78_±_4.80|17.57_±_11.39|
|Reflexion|anger|-0.56_±_2.48|1.22_±_3.98|0.67_±_5.33|11.11_±_9.77|
|l|∆|-0.56(ns)|-0.56(ns)|-1.11(ns)|-6.46(ns)|
|||Llama|3.1-8B-Instruct|||
|Agent|Emotion|IGT score@1_↑_|IGT score@2_↑_|IGT score_↑_|Profit%_↑_|
||none|-2.11_±_0.75|-0.56_±_1.12|-2.67_±_1.39|4.44_±_6.00|
|ReAct|anger ∆|-2.89_±_1.20 -0.78(ns)|-3.78_±_1.91 -3.22(ns)|-6.67_±_2.43 -4.00(ns)|-0.83_±_5.82 -5.27(ns)|
||none|-133_±_165|-267_±_162|-400_±_311|625_±_951|
|Query|anger|. . -3.11_±_1.62|. . -5.22_±_1.16|. . -8.33_±_2.46|. . -5.90_±_8.40|
||∆|-1.78(ns)|-2.56(ns)|-4.33(ns)|-12.15(ns)|
||none|1.44_±_1.28|3.56_±_2.08|5.00_±_2.37|32.63_±_8.06|
|Reflexion|anger|-1.78_±_1.60|2.00_±_1.68|0.22_±_2.10|13.06_±_6.41|
|l|∆|-3.22 (ns)|-1.56 (ns)|-4.78 (ns)|-19.58 (ns)|

Table 5 shows the performance of different agent designs across models under neutral and anger conditions. While some mean differences can be observed across agents, blocks, and metrics, these effects are inconsistent in both magnitude and direction. Moreover, nearly all comparisons are statistically non-significant, suggesting that these variations are not robust. This means that differences across emotion conditions are not consistent across agent–model combinations, indicating no stable effect of emotion induction.

Since Table 5 contains mostly non-significant neutral-vs-anger differences, we further examine whether these null results reflect a lack of reliable average effects or simply insufficient sensitivity. Non-significance alone should not be interpreted as evidence of no effect. Therefore, we report the paired neutral-vs-anger _p_ -values in Table 6 as supplementary to Table 5; and the 80% minimum detectable effects (MDEs) in Table 7. MDE is the smallest detectable anger–neutral difference in block-wise net score with 80% power at _α_ = 0.05; smaller is more sensitive.

Table 6: Paired neutral-vs-anger _t_ -test _p_ -values across model-agent configurations. Only 2 values reaches significance, suggesting no stable anger effect. *: _p <_ 0.05.

|Model|Agent|IGT score@1|IGT score@2|IGT score|Profit%|
|---|---|---|---|---|---|
||ReAct|0.886|0.177|0.568|0.447|
|GPT-oss-20b|Query|0.386|0.690|1.000|0.521|
||Reflexion|0.295|0.370|0.980|0.750|
||ReAct|0.038*|0.456|0.247|0.017*|
|Qwen3-4B-Instruct-2507|Query|0.714|0.874|0.701|0.618|
||Reflexion|0.375|0.534|0.951|0.765|
||ReAct|0.331|0.215|0.790|0.331|
|Qwen2.5-7B-Instruct|Query|0.205|0.108|0.093|0.226|
||Reflexion|0.826|0.917|0.855|0.603|
||ReAct|0.591|0.150|0.147|0.370|
|Llama3.1-8B-Instruct|Query|0.265|0.186|0.153|0.274|
||Reflexion|0.137|0.587|0.156|0.052|

Results show that only a small number of comparisons reach statistical significance (ReAct _×_ Qwen3-4B-Instruct-2507 for IGT score@1 and Profit%), which supports the conclusion that anger does not produce a reliable average shift across model–agent configurations. At the same time, the MDE analysis clarifies the scale of effects that our experiments can rule out. For single-block (20-round) IGT scores, one shifted choice from a disadvantageous deck to an advantageous deck changes the net score by 2 points. Thus, MDE values around 2-4 correspond to roughly one to two shifted choices within a 20-round block, while MDE values around 6-8 correspond to roughly three to four shifted choices. As shown in Table 7, Query and ReAct agents are sensitive to relatively small block-level changes in several models, whereas Reflexion has larger MDEs and should therefore be interpreted more cautiously.

Table 7: 80% minimum detectable effects (MDEs) for 20-round block IGT scores under paired _t_ -tests. Smaller values imply higher sensitivity. It is noticeable that Query/ReAct are more sensitive than Reflexion agents.

|Agent|GPT-oss-20b|Qwen3-4B-Instruct-2507|Qwen2.5-7B-Instruct|Llama3.1-8B-Instruct|
|---|---|---|---|---|
|Query|3.339|4.427|3.762|4.583|
|ReAct|4.218|2.349|1.223|4.222|
|Reflexion|6.722|7.995|7.381|6.140|

Figure 18: **(Left: Initial Bias)** Different models have different biases at the first choice. **(Right: Game Bias)** Across advantageous-deck conditions, the agent’s selections are systematically skewed toward decks with less frequent losses. The order of decks changes the odds of the advantageous decks to be chosen.

### D LLMs’ Biased Initial Choices and Consequences on Information Bias.

As observed in multiple prior works (Zheng et al., 2024; Pezeshkpour & Hruschka, 2024), LLMs present a bias to how choices are present in prompts. In our case, the model is heavily biased towards choosing some decks/letters in its first round (Fig. 18 (left)), which makes the original fixed deck ordering problematic: any early preference may reflect positional bias rather than learned preference from feedback. As mentioned in Section 3, we periodically shuffled the decks, so that the results would not systematically correspond to the first option.

The results in Fig. 18 (right) indicate that deck shuffling mitigates, but does not remove, prompt-order bias. Importantly, the agent still selects _D_ most frequently on average, suggesting that it can learn to favor an advantageous deck despite randomized label-to-deck mappings. However, residual asymmetries remain: in configurations where _A_ is not advantageous, the agent often shifts toward disadvantageous deck _B_ , which also has low loss frequency. This pattern suggests that learning is preserved under shuffling, but choice behavior is still partly shaped by how options are presented.

Figure 19: Emotion causes different decision patterns in the 2nd block under impaired early observation.

Based on this observation, we define _Advantageous Observation_ as selecting an advantageous deck by chance on the first round, given the model’s strong tendency to choose one of the four options. For example, if the model chooses _’A’_ on its first round and the advantageous decks are _B_ and _C_ , it will count as a _’no’_ .

**Anger Emotion Leads to “Gut-Feeling” Decision Making.** Fig. 19 shows that the effect of emotion depends on whether the advantageous deck is observed, and that this difference emerges mainly in block 2 rather than block 1. In block 1, IGT scores are low and the differences between neutral and anger conditions are not significant in either observation setting, suggesting that both agents are still in an early exploratory phase. In block 2, however, the pattern becomes asymmetric. When the advantageous deck is observed, performance remains similar under neutral and anger prompting, with no significant difference between the two conditions. In contrast, when the advantageous deck is not observed, the neutral agent achieves clearly higher IGT scores, whereas the anger-conditioned agent performs substantially worse, and this gap is significant. Thus, anger does not broadly impair decision making across all settings; rather, anger-conditioned agents adapt more weakly under ambiguous conditions.

**Emotion Mediate How Agents Explore/Exploit Games.** The moderation results lead to a new question: why induced anger makes the LLM less sensitive to the outcome of disadvatageous decks. We hypothesize that the moderation is associated with the exploitation/exploration balance of the agent. To examine the exploitation/exploration strategies of agents, we fit a simple choice model to its sequential decisions. Before each turn, we reconstructed the agent’s belief state over the four decks using only its past observations. For each deck _d_ , we calculate an estimated value _µd_ as the mean of previously observed net outcomes from that deck, and an uncertainty term _σd_ = 1/^(_√_) _nd_ + 1, where _nd_ is the number of times the deck had been sampled so far. We then modeled the probability of choosing deck _d_ with a softmax policy where _β_ captures sensitivity to expected reward and _γ_ captures sensitivity to uncertainty. Parameters were estimated by maximizing the log-likelihood of the observed choices. Under this formulation, a larger _β_ indicates that the agent’s choices are more strongly guided by its learned reward estimates, whereas a larger _γ_ indicates greater sensitivity to uncertainty.

In Fig. 20, we plot the estimated _β_ and _γ_ parameters by a _10-round-sized_ block for the neutral and anger conditions. Across blocks, the anger condition generally exhibited lower _β_ estimates than the neutral condition at the 3rd block, indicating a weaker tendency to exploit learned value estimates.

Meanwhile, under both emotions, exploration tends to decrease across trials. That can be attributed to the fact that _σd_ diminishes by trials and there are less new trials to explore. Though with similar trend, we can see that a higher _γ_ was observed in the anger condition at rounds 20 to 40, suggesting a possible increase in uncertainty sensitivity.

Figure 20: Anger emotion induces less exploitation ( _β ↓_ ) on Reflexion _×_ GPT-oss-20b.
