# Emotion2Skill: Model-Internal Emotion Signals for Adaptive Skill Selection and Evolution

- **Authors:** Bohan Lin, Hejia Geng, Xinyi Xie et al.
- **Date:** 2026-08-10
- **Source:** https://arxiv.org/abs/2608.09248
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `lin_emotion2skill-internal-emotion-signals_2608.09248`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Bohan Lin**^**1,2** **, Hejia Geng**^**3** **, Xinyi Xie**^**4** **, Heng Zhou**^**1,5** **, Qinghua Xing**^**1** , **Chen Zhang**^**5*** , **Yudong Zhang**^**1,2,***

1University of Science and Technology of China (USTC),

2Suzhou Institute for Advanced Research, USTC,

3University of Oxford, 4University of Arizona, 5Shanghai AI Laboratory

**Contact:** linbohan@ustc.edu

## Abstract

Skill-based LLM agents select reusable procedures from external libraries to solve complex tasks, yet their routing decisions rely primarily on task descriptions, verbal reflections, and experience-derived rules, without explicitly observing the model’s internal decision state. Recent interpretability work has identified emotion representations that influence model behavior, but their use has largely focused on post-hoc analysis and output steering. Their potential to guide agent-level decisions remains underexplored. We propose EMOTION2SKILL, a framework that extracts LLM-internal emotion vectors and incorporates them into both skill selection and skill evolution. The framework uses these representations to inform skill choice and tracks their changes to identify problematic invocations, enabling targeted skill refinement beyond coarse outcome feedback. Across two backbones, EMOTION2SKILL outperforms seven baselines. On Qwen3-8B, it exceeds Trace2Skill by 3.1 percentage points in WebShop success and 5.7 in ALFWorld average success. Ablations and specificity controls support the utility of context-aligned emotion signals, while co-activation analysis reveals semantically coherent emotion–skill associations. The same emotion vectors also transfer to MATH and MBPP without re-extraction. These results establish LLM-internal emotion representations as an effective decision-level signal for orchestrating agent skill systems, extending their utility beyond interpretability and output steering. Code is available at Emotion2Skill.

## 1 Introduction

Skill-based LLM agents solve complex, long-horizon interactive tasks by retrieving reusable procedural knowledge from external libraries (Wang et al., 2024; Yang et al., 2026d; Yu et al., 2026;

> *Chen Zhang and Yudong Zhang are the corresponding authors.

Figure 1: Human decision-making integrates external stimuli with internal affective states. We ask whether LLM agents can exploit an analogous signal: the emotion representations encoded in the model’s own residual stream, extracted at inference time and used to condition skill utilization.

Ni et al., 2026). While recent work has substantially advanced skill evolution (Yang et al., 2026d; Yu et al., 2026), trajectory distillation (Ni et al., 2026), and unified skill training (Shi et al., 2026), the quality of skill _selection_ continues to bound overall task performance. No amount of skill refinement compensates for systematically retrieving an inappropriate procedure at decision time. Effective skill selection requires conditioning on two complementary sources of information. The first is the _external textual context_ : the task description, conversation history, and environment observations that standard agents already exploit. The second is the model’s own _internal state_ : its accumulated uncertainty, its assessment of prior action outcomes, and the trajectory-level confidence it has built up to the current step. Identical external contexts can demand different skills depending on this internal state, yet no existing skill-selection method directly observes or leverages it.

Existing approaches to skill selection rely primarily on explicit textual artifacts. Text-embedding retrieval (Wang et al., 2024; Yu et al., 2026) matches task descriptions against skill descriptions in a shared semantic space, verbal self-reflection (Shinn et al., 2023) produces textual critiques of prior actions to inform subsequent choices, experience-based learning (Zhao et al., 2024) distills trajectory-level lessons into natural-language rules, and model-aware alignment (Yu et al., 2026) refines skill content through text-space search. Although effective and complementary, these methods share a structural limitation, namely that they operate exclusively on the external textual context while the model’s internal representational state at the moment of selection remains unobserved. The latent component of the selection problem is therefore systematically absent from the routing decision.

Neuroscience and psychology have long established that human decisions are shaped not only by external stimuli but also by internal affective states (Damasio, 1994; Lerner et al., 2015) (Figure 1). We ask whether an analogous principle holds for LLMs. Recent interpretability work shows that LLMs encode linear, causal emotion representations in their residual streams (Zou et al., 2023; Sofroniew et al., 2026) that can be read out at inference time and that _causally_ influence agentically relevant outputs such as reward-hacking and sycophancy rates (Sofroniew et al., 2026). Unlike generic hidden-state features, emotion vectors are empirically linked to model behavior, interpretable by construction (each dimension names a specific affect), and task-independent, making them a natural proxy for the agent’s internal decision state. **Skill selection should therefore incorporate the agent’s internal decision state alongside external textual context, and emotion vectors offer an interpretable proxy for doing so.**

Building on this insight, we propose EMOTION2SKILL, a framework that augments LLM-based skill selection with an explicit emotionderived signal. At each decision step, the agent’s emotion state is extracted from the model’s residual stream and passed through a lightweight encoder that produces a natural-language emotion template together with a confidence score. This emotion summary is concatenated into the LLM’s selection prompt as auxiliary context, and a confidencegating mechanism omits the signal when it is uninformative. Beyond selection, emotion trajectories recorded during episodes are analyzed for abrupt internal state shifts, and the resulting diagnostics guide periodic skill evolution by directing the rewriter’s attention to problematic trajectory segments. On WebShop (Yao et al., 2022) and ALF-

World (Shridhar et al., 2021), EMOTION2SKILL with Qwen3-8B (Yang et al., 2025) improves over Trace2Skill (Yu et al., 2026), the strongest baseline, by **+5.7 pp** average success on ALFWorld and **+3.1 pp** success rate on WebShop, outperforming all baselines in aggregate with consistent gains on Qwen3-14B.

Our contributions are as follows.

- We reveal the connection between LLM-internal emotion vectors and agent skill selection, showing empirically that emotion representations encode decision-relevant information absent from text-level features.

- We design EMOTION2SKILL, a framework that leverages emotion vectors for both skill selection and skill evolution in LLM-based agents.

- We conduct extensive experiments on two interactive benchmarks against five baselines, together with ablation studies and interpretability analyses that validate the effectiveness of the proposed approach.

## 2 Related Work

**LLM internal representations and emotion vectors.** A growing body of work shows that LLM internal representations encode high-level semantic concepts as linear directions in activation space (Zou et al., 2023; Park et al., 2024; Nanda et al., 2023) that can be read out at inference time (Burns et al., 2023; Li et al., 2023) and causally steered to alter model behavior (Turner et al., 2023; Rimsky et al., 2024). Sofroniew et al. (2026) demonstrate that frontier LLMs maintain fine-grained emotion directions aligned with the 27-category GoEmotions taxonomy (Demszky et al., 2020) that causally influence agentically relevant behaviors; concurrent mechanistic studies further show that such emotion structure non-monotonically shapes agent reasoning and safety (Sun et al., 2026; Shou and Guan, 2026). We adopt these categorical directions rather than appraisal vectors (Scherer, 2001) or valence-arousal subspaces, as they are the representation for which causal influence on agent behavior has been directly demonstrated, and named affective categories provide interpretable routing rationales without additional grounding. EMOTION2SKILL is the first to use emotion vectors as an active routing and evolution signal for agent skill systems, moving beyond post-hoc analysis and direct output steering.

**Skill-based LLM agents.** Skill-based LLM agents maintain reusable procedural knowledge for long-horizon tasks (Wang et al., 2024; Yang et al., 2026d; Yu et al., 2026; Ni et al., 2026), with recent work advancing skill evolution (Yang et al., 2026d; Yu et al., 2026; Yang et al., 2026a), including EvoSkill (Wang et al., 2025), trajectory distillation (Ni et al., 2026), unified training (Shi et al., 2026), and experiential learning (Zhao et al., 2024; Shinn et al., 2023; Xu et al., 2026). Concurrent work advances complementary dimensions of skill learning: ARISE (Li et al., 2026) introduces hierarchical reinforcement learning over intrinsic skill representations; OPID (Yang et al., 2026c) applies on-policy self-distillation for dense skilllevel supervision; Skill-R1 (Vishe et al., 2026) and ReSkill (He et al., 2026) evolve skills via outcome-based and RL-in-the-loop objectives; and MAGE (Yang et al., 2026b) co-evolves skill knowledge graphs across multiple agents. These approaches and EMOTION2SKILL operate on different dimensions: where they optimize skill routing through external reward or similarity feedback, EMOTION2SKILL conditions on the model’s internal affective state, and the two classes of signal are complementary rather than directly comparable.

## 3 Method

EMOTION2SKILL grounds skill decisions in the model’s own affective representations, filling the gap left by context-only skill routing. **Emotion Vectors** are extracted from the residual stream and projected into a natural-language summary at each step. **Emotion-Augmented Skill Selection** injects this summary into the routing prompt via a confidence gate. **Emotion-Driven Skill Evolution** detects abrupt trajectory shifts to localize and rewrite underperforming skills. Figure 2 illustrates the design.

### 3.1 Preliminaries

Consider an agent equipped with a skill library _S_ = _{s_ 1 _, . . . , sM }_ , where each skill _si_ is a reusable procedure described by a natural-language SOP _Di_ . At decision step _t_ , the agent observes the external textual context _xt_ (task instruction, conversation history, and environment observations) and possesses a latent internal decision state encoded in the residual stream of the underlying LLM. We denote by **h**^(() _t_^(_ℓ_)) _∈_ R^_d_ the residual-stream activation at layer _ℓ_ and token position _t_ , and by _L_^_∗_ the opti-mal extraction layer selected on the GoEmotions validation split.

Standard skill-based agents select a skill _s_^_∗_ _t_^by prompting the LLM _M_ with _xt_ and the candidate SOPs _{D_ 1 _, . . . , DM }_ , relying entirely on the textual content of these inputs. EMOTION2SKILL extends this formulation by additionally conditioning on an emotion summary **_ϕ_** _t_ derived from the agent’s internal state:

where **_ϕ_** _t_ is a natural-language token sequence produced by a learned emotion encoder _fθ_ . The LLM remains the primary decision-maker; **_ϕ_** _t_ serves as auxiliary context that makes the agent’s latent state explicit.

### 3.2 Emotion Vector Extraction and Encoding

**Contrastive extraction.** We extract emotion directions from the agent model following the contrastive-averaging procedure of Sofroniew et al. (2026), replicated on Qwen3. For each of the _K_ =27 emotion concepts in the GoEmotions taxonomy (Demszky et al., 2020), the model generates _N_ =100 short stories with the target emotion. A forward pass through each story yields activations **h**^(() _t_^(_ℓ_)) at every layer _ℓ_ and token position _t_ . We mean-pool over tokens past an initial prefix ( _t ≥_ 50) to obtain a per-story representation **h**^(¯()^(_ℓ_)) , then compute the emotion direction for concept _k_ at layer _ℓ_ by subtracting the global mean from the within-concept mean:

To suppress variance attributable to stylistic rather than affective content, a PCA-based denoising step projects out the top principal components of emotionally neutral activations, yielding denoised directions **v** ˆ^(()^(_ℓ_)) = Π^(_⊥_) **v**^(()^(_ℓ_)) _k k_^(.Theoptimallayer)^_L∗_ is selected from a candidate grid by maximizing GoEmotions classification accuracy . Stacking the denoised directions yields the emotion extractor matrix _We_ = [ˆ **v** 1^(()^(_L∗_)) _, . . . ,_ ˆ **v** _K_^(()^(_L∗_)) ]^(_⊤_) _∈_ R^(_K×d_) .

At inference time, the raw emotion state at decision step _t_ is obtained by projecting the residual-stream activation at the Assistant: delimiter token (Sofroniew et al., 2026) through _We_ and applying _ℓ_ 2 normalization:

Figure 2: Framework of our method. Left: offline extraction of emotion vectors from the LLM residual stream and MLP encoder training. Middle: emotion state mapped to a confidence-gated summary injected into the skillselection prompt. Right: emotion trajectory shifts trigger targeted SOP rewriting for underperforming skills.

The resulting vector **e** _t_ assigns a scalar activation to each of the _K_ emotion dimensions, capturing the agent’s affective state at the moment immediately preceding generation.

**Emotion encoder.** The raw vector **e** _t_ is not directly consumable by the LLM’s text-based prompt; naively reporting the top- _k_ activated emotion names as tokens discards magnitude information and ignores interactions between co-activated emotions. We therefore introduce a lightweight _emotion encoder fθ_ that maps **e** _t_ to two outputs used for prompt augmentation. The encoder is a 3-layer MLP that produces an encoded representation **r** _t_ = _fθ_ ( **e** _t_ ) _∈_ R^(_d′_) . From **r** _t_ we derive (i) a _template index_ over a predefined bank _T_ = _{T_ 1 _, . . . , TC}_ of _C_ =12 natural-language emotion-state descriptions, where each template _Tj_ is associated with a learnable prototype **q** _j ∈_ R^(_d′_) :

and (ii) a scalar _confidence score_

indicating how informative the current emotion state is for routing. We define the composite emotion summary as the pair **_ϕ_** _t_ = ( _Tt_^(_∗,ct_).When) _ct < τ_ , **_ϕ_** _t_ is omitted from the prompt entirely (confidence gating), allowing the system to fall back to standard text-only selection.

The encoder is trained on tuples ( **e** _t, s_^sel _t_^(_, yt_)) collected during a warm-up phase in which the agent runs episodes under a baseline policy (ReAct with the skill library, without emotion signals), where _s_^sel _t ∈S_ is the baseline’s skill choice and _yt ∈{_ 0 _,_ 1 _}_ is the episode outcome. The objective combines a supervised-contrastive loss (Khosla et al., 2020) that clusters representations by effective skill assignments with a template-classification loss. For an anchor **r** _i_ with positive set _A_^(+) ( _i_ ) = _{j | s_^sel _j_^(=)^_s_ _i_^sel _∧ yj_ =1 _}_ , the contrastive term is where _κ_ is a temperature and representations are _ℓ_ 2- normalized. The classification term assigns pseudo-labels ˆ _ct_ via _k_ -means ( _k_ = _C_ ) on the raw emotion vectors and minimizes the cross-entropy over prototype similarities:

The overall objective is _L_ = _L_ ctr + _λ L_ cls, where _λ_ balances the two terms.

### 3.3 Emotion-Augmented Skill Selection

The emotion summary **_ϕ_** _t_ produced by the encoder is concatenated into the LLM’s input prompt as additional context, making the agent’s latent state visible alongside the task description and candidate skill SOPs. At each decision step _t_ , the system constructs an augmented prompt _Pt_ by appending the emotion template _Tt_^(_∗_and the confidence score) _ct_ to the standard prompt components. When the encoder’s confidence falls below the threshold _τ_ , the emotion fields are withheld, and the prompt reduces to the conventional text-only format:

The LLM then reads _Pt_ in its entirety and produces a skill decision via constrained generation over the valid skill identifiers:

where _pM_ ( _si | Pt_ ) denotes the conditional probability that _M_ assigns to skill _si_ . The emotion summary enters through the same input channel as the task description, and setting _τ_ =1 recovers standard text-only selection.

### 3.4 Emotion-Driven Skill Evolution

Beyond online selection, the episode log _L_ ep provides a temporally ordered trace of internal states that can inform periodic skill evolution. Standard evolution methods (Yang et al., 2026d; Yu et al., 2026) prompt the LLM to rewrite underperforming skill SOPs based on binary episode outcomes and, optionally, textual self-reflection. This signal is coarse in that it arrives only after episode termination and provides no indication of _which_ intermediate steps were problematic.

EMOTION2SKILL augments the rewriter’s input with structured emotion diagnostics that localize critical state changes within the episode.

Let _E_ = ( **e** 1 _, . . . ,_ **e** _T_ ) denote the emotion trajectory of a completed episode. To identify steps at which the agent’s internal state shifted abruptly, we compute the step-to-step _emotion shift magnitude_

This quantity measures the cosine distance between consecutive emotion states and is agnostic to which specific dimensions change; large _δt_ indicates a sharp reorganization of the agent’s internal state regardless of whether the shift involves positive-valence, negative-valence, or cognitive dimensions. An _emotion transition point_ is defined as a step at which _δt_ exceeds the episode mean by more than one standard deviation:

where _δ_^(¯) = ( _T −_ 1)^(_−_1)^_T_ _t_ =2^_δt_and^_σδ_ = ( _T −_ 1)^(_−_1)^_T_ _t_ =2^(()^(_δt−δ_¯)21)^(_/_2.Byoperatingon) the full _K_ -dimensional emotion vector without preselecting specific dimensions, this criterion captures any substantial internal reorganization, including shifts from exploratory states to frustrated states, from confident states to confused states, or from one cognitive mode to another.

For each skill _si_ flagged for evolution, we restrict to skill-specific transitions _Pi_ = _{t ∈P | s_^_∗_ _t_^(=)^(_si}_) and extract a diagnostic context around each:

capturing a 3-step window of observations, skill choices, emotion templates, and shift magnitudes. These diagnostics, together with a verbalized summary ∆ _i_ describing the transition pattern, are appended to the evolution prompt alongside the current SOP _Di_ and the episode outcome _r_ ep:

The rewriter receives _P_ evo^(()^(_i_)as its full input context) and generates the revised SOP:

Evolution proceeds over _R_ rounds; at each round the agent executes a batch of episodes, identifies transition points, and updates each underperforming skill (success rate _< η_ ):

where _U_^(()^(_r_)) is the set of underperforming skills in round _r_ . The emotion diagnostics enter through the same prompt channel as the SOP and episode transcript, providing temporally localized context that complements the coarse binary outcome signal. Algorithm 1 summarizes both the online selection and the evolution procedure.

## 4 Experiments

### 4.1 Setup

We evaluate on **WebShop** (Yao et al., 2022) for web-based shopping and **ALFWorld** (Shridhar et al., 2021) for text-based household tasks. On WebShop, we report Score and success rate. On ALFWorld, we report per-type and average success rates. We compare against Zero-Shot, ReAct (Yao et al., 2023), Reflexion (Shinn et al., 2023), ExpeL (Zhao et al., 2024), MASA (Yu et al., 2026), EvoSkill (Wang et al., 2025), and Trace2Skill (Ni et al., 2026); all methods are reproduced on both Qwen3-8B and Qwen3-14B (Yang et al., 2025). All experiments are conducted on 2 _×_ H800 GPUs. EMOTION2SKILL uses confidence threshold _τ_ =0 _._ 3, with the extraction layer _L_^_∗_ selected by hit-any classification accuracy on the GoEmotions validation split. Qwen3-14B reuses the same skill libraries and _τ_ ; only _L_^_∗_ is re-selected. Dataset statistics, baseline reproduction protocols, hyperparameters, prompt templates, and skill libraries are detailed in Appendices B and D.

|||||ALFW|orld (|Success|%)||W|ebShop|
|---|---|---|---|---|---|---|---|---|---|---|
|Backbone|Method|Pick|Look|Clean|Heat|Cool|Pick2|Avg.|Score|Succ.|
||Zero-Shot|43.4|39.1|19.1|9.6|12.3|4.4|21_._9_±_1_._5|11.2|2_._8_±_0_._5|
||ReAct (Yao et al.,2023)|53.1|42.0|41.3|20.7|18.4|3.6|30_._7_±_1_._2|12.1|5_._2_±_3_._2|
|8B|Reflexion (Shinn et al.,2023)|52.2|34.4|29.0|24.1|19.1|14.1|31_._5_±_1_._6|22.4|6_._8_±_2_._4|
|n3-|ExpeL (Zhao et al.,2024)|61.2|30.1|44.4|35.6|28.7|11.5|37_._0_±_4_._7|39.6|18_._8_±_4_._1|
|we|MASA (Yu et al.,2026)|55.1|24.1|46.6|48.9|43.9|23.4|43_._3_±_0_._9|44.7|27_._1_±_0_._8|
|Q|EvoSkill (Wang et al.,2025)|70.1|32.1|57.3|47.3|39.5|15.3|46_._4_±_1_._9|53.3|29_._7_±_1_._6|
||Trace2Skill (Ni et al.,2026)|70.5|28.4|56.8|32.3|37.6|26.8|47_._4_±_0_._9|52.0|30_._2_±_3_._8|
||EMOTION2SKILL(Ours)|61.2|62.5|63.3|57.8|42.7|35.3|53_._1_±_1_._6|64.7|33_._3_±_0_._6|
||Zero-Shot|55.1|42.9|8.2|7.9|7.7|13.1|23_._4_±_2_._8|27.5|9_._4_±_0_._8|
|B|ReAct (Yao et al.,2023)|52.1|36.6|19.8|19.0|16.5|17.8|30_._2_±_2_._4|37.6|13_._0_±_1_._6|
|14|Reflexion (Shinn et al.,2023)|59.0|33.8|31.6|46.9|10.2|13.8|34_._4_±_0_._8|32.7|14_._4_±_7_._0|
|3-|ExpeL (Zhao et al.,2024)|71.0|35.2|43.8|37.0|34.3|28.9|43_._8_±_0_._8|45.4|20_._6_±_1_._2|
|en|MASA (Yu et al.,2026)|81.0|41.2|50.5|36.5|56.2|17.2|49_._2_±_0_._8|46.8|24_._8_±_2_._0|
|Qw|EvoSkill (Wang et al.,2025)|72.9|43.4|56.3|41.6|42.3|23.9|49_._7_±_1_._9|56.0|31_._2_±_1_._6|
||Trace2Skill (Ni et al.,2026)|59.3|58.0|48.6|52.0|45.5|38.8|52_._1_±_2_._6|50.7|32_._8_±_1_._7|
||EMOTION2SKILL(Ours)|68.3|35.6|65.2|71.7|63.1|34.2|57_._0_±_1_._6|66.8|37_._0_±_0_._6|

Table 1: Main results on ALFWorld and WebShop. All values are means over three runs; ALFWorld Avg. and WebShop Succ. additionally report standard deviation. **Bold** denotes the best result per column .EMOTION2SKILL achieves the best aggregate performance on both benchmarks across Qwen3-8B and Qwen3-14B.

Figure 3: Emotion vector quality and layer selection for Qwen3-8B. (a) GoEmotions validation accuracy across layers peaks at 37.2% at layer 24; the dashed line marks the 8.2% random baseline. (b) t-SNE of held-out GoEmotions activations at _L_^_∗_ =24; valence regions are distinguishable but show substantial overlap. (c) Fisher discriminability ratio peaks at _F_ =2 _._ 08 at layer 24 but does not track accuracy monotonically.

### 4.2 Emotion Vector Quality and Layer Selection

We test whether Qwen3-8B’s residual stream encodes discriminable emotion structure by measuring classification accuracy across the 27 emotion categories in the GoEmotions validation split (Demszky et al., 2020). We select the extraction layer _L_^_∗_ using this validation accuracy.

**Evaluation protocol and layer selection.** For the 36-layer Qwen3-8B model, we evaluate can-didate layers _ℓ ∈{_ 16 _,_ 18 _,_ 20 _,_ 22 _,_ 24 _,_ 26 _,_ 28 _}_ . At each layer, we project the GoEmotions validation activations onto the contrastively-extracted directions **v** ˆ _k_^(()^(_ℓ_)) and classify each example by its highest-cosine-similarity direction, without any task-specific fine-tuning. We choose _L_^_∗_ using only this validation accuracy; the GoEmotions test split is held out from layer selection and is used only for post-selection evaluation or visualization. Since GoEmotions is multi-label, we adopt a _hit-any_ criterion: a prediction is correct if the top direction matches any annotated label. Under this criterion, a random baseline achieves 8.2% accuracy. Figure 3a shows that accuracy is non-monotonic across layers. It rises from 15.8% at layer 16 to 24.3% at layer 18, dips to 21.7% at layer 20, and then climbs through 30.5% at layer 22 to a peak of **37** _._ **2** % at **layer 24** . Accuracy falls to 33.6% at layer 26 before recovering to 35.1% at layer 28. The peak at 24 _/_ 36=0 _._ 67 depth is broadly consistent with Sofroniew et al.’s (2026) observation that emotion representations concentrate near two-thirds depth, though the non-monotonic profile suggests additional layer-specific specialization. We fix _L_^_∗_ =24 for all Qwen3-8B experiments. For the 40-layer Qwen3-14B model, the same validationonly sweep over _{_ 18 _,_ 20 _,_ 22 _,_ 24 _,_ 26 _,_ 28 _,_ 30 _}_ selects _L_^_∗_ =26 with 39.4% accuracy.

**Geometric structure.** Figure 3b visualizes GoEmotions activations at layer 24 via t-SNE. Positive-valence emotions such as _joy_ , _love_ , and _optimism_ concentrate in the lower-left quadrant, while negative-valence emotions such as _anger_ , _disgust_ , and _sadness_ occupy the upper-right. Cognitive states such as _curiosity_ and _confusion_ form an intermediate region. Closely related affects within the same valence band still overlap, as seen for _anger_ and _annoyance_ . Fisher’s ratio _F_ = _σ_ between^2^(_/σ_) within^2 peaks at _F_ = **2** _._ **08** at layer 24 but does not track accuracy perfectly. Layer 20 has _F_ =1 _._ 52 and 21.7% accuracy, whereas layer 22 achieves 30.5% accuracy despite a lower ratio of _F_ =1 _._ 21. Thus, the ratio serves as a separability diagnostic rather than a substitute for validation accuracy.

### 4.3 Main Results

**Overall comparison.** Table 1 reports results on both backbones. EMOTION2SKILL attains the best aggregate score on both benchmarks at both scales. On Qwen3-8B it reaches **53.1%** average success on ALFWorld and **33.3%** Succ. on WebShop, exceeding Trace2Skill (Ni et al., 2026), the strongest aggregate baseline, by **+5.7** and **+3.1** pp; on Qwen314B it reaches **57.0%** and **37.0%** , a margin of **+4.9** and **+4.2** pp over the same baseline. Its standard deviation is at most 1.6 pp across these settings. By comparison, ExpeL has a standard deviation of 4.7 pp on ALFWorld with Qwen3-8B, and Reflexion has a standard deviation of 7.0 pp on WebShop with Qwen3-14B.

**Per-task behavior and scaling.** On Qwen3-8B, EMOTION2SKILL exceeds Trace2Skill by 34.1 pp on Look, 25.5 pp on Heat, and 8.5 pp on Pick2. Trace2Skill remains ahead by 9.3 pp on Pick. Relative to EvoSkill, the largest gains are 30.4 pp on Look, 20.0 pp on Pick2, and 10.5 pp on Heat. Appendix 8 traces one Heat episode in which the preaction emotion signal reroutes the agent to verify an appliance precondition before the erroneous heat call. On Qwen3-14B, EMOTION2SKILL leads Trace2Skill by 9.0 pp on Pick, 16.6 pp on Clean, 19.7 pp on Heat, and 17.6 pp on Cool. Trace2Skill leads by 22.4 pp on Look and 4.6 pp on Pick2. The aggregate advantage therefore persists across model scales, although the task-level ranking varies. Qwen3-14B also has higher GoEmotions validation accuracy, reaching 39.4% at _L_^_∗_ =26 compared with 37.2% at _L_^_∗_ =24 for Qwen3-8B. Determining whether this difference contributes to downstream gains would require separating layer quality from model scale in a controlled ablation. Concurrent frameworks that route skills from external reward or similarity signals (Li et al., 2026; Yang et al., 2026c; Vishe et al., 2026; Yang et al., 2026b) are complementary; combining them with internal-state signals is left to future work.

||WebS|hop|ALF|World|
|---|---|---|---|---|
|Condition|Succ. (%|) ∆|Avg. (%|) ∆|
|w/o Emotion Extraction|23.4|_−_9.9|38.3|_−_14.8|
|w/o Emotion Encoder|25.0|_−_8.3|41.4|_−_11.7|
|w/o Emotion Evolution|27.3|_−_6.0|43.8|_−_9.3|
|Emotion2Skill (Full)|33.3|—|53.1|—|
|Sensitivity to confidence|i threshold|_τ_|||
|_τ_ = 0_._1|26.6|_−_6.7|42.2|_−_10.9|
|_τ_ = 0_._3(default)|33.3|—|53.1|—|
|_τ_ = 0_._6|27.8|_−_5.5|43.4|_−_9.7|
|_τ_ = 0_._9|20.0|_−_13.3|41.4|_−_11.7|
|Emotion signal specific|iity contro|i ls|||
|Generic State (PCA-27)|20.3|_−_13.0|32.8|_−_20.3|
|Shuffled Emotion|7.8|_−_25.5|24.2|_−_28.9|

Table 2: Component ablations, confidence-threshold sensitivity, and emotion-signal specificity controls on WebShop and ALFWorld with Qwen3-8B. The component ablations replace contrastive _We_ with plain class means, bypass _fθ_ to inject the top-3 emotion labels as raw text, or replace emotion diagnostics with outcomeonly evolution. Specificity controls hold the skill library, encoder architecture, and data split fixed. ∆ is the gap relative to the full system for ablations and specificity controls, and relative to _τ_ =0 _._ 3 for threshold sensitivity.

### 4.4 Ablation Studies and Analysis

**Component ablations and threshold sensitivity.** Table 2 evaluates the three core components on WebShop and ALFWorld with Qwen3-8B, changing one component at a time. Replacing contrastive _We_ with plain within-class means reduces WebShop success by 9.9 pp and ALFWorld average success by 14.8 pp. This is the largest drop among the three ablations and supports the role of globalmean subtraction in constructing useful directions. Bypassing _fθ_ and injecting the top-3 raw emotion labels reduces performance by 8.3 pp on WebShop and 11.7 pp on ALFWorld, supporting the learned encoding over direct label injection. Using outcome-only rewriting instead of emotion diagnostics costs 6.0 pp and 9.3 pp on the two benchmarks, respectively, supporting the use of temporal transitions **d** _t_ for targeted skill rewriting. The threshold _τ_ =0 _._ 3 performs best among the tested values. At _τ_ =0 _._ 1 and _τ_ =0 _._ 6, the gaps to the default are at most 6.7 pp on WebShop and 10.9 pp on ALFWorld. Increasing the threshold to _τ_ =0 _._ 9 widens the gaps to 13.3 pp and 11.7 pp, consistent with stricter gating discarding useful emotion context.

**Emotion-signal specificity.** The specificity controls in Table 2 test the choice of feature representation and its alignment with the routing context. They change the 27-dimensional router input while keeping all other conditions fixed. Generic State uses 27 principal components of residual activations and reaches 20.3% WebShop success and 32.8% ALFWorld average success, falling below the full system by 13.0 pp and 20.3 pp. Matching dimensionality and retaining high-variance hidden-state features is therefore insufficient to recover the full performance in this comparison. Shuffled Emotion preserves the feature values and their marginal distribution but breaks the pairing between vectors and training contexts. It achieves 7.8% and 24.2%, trailing the full system by 25.5 pp on WebShop and 28.9 pp on ALFWorld. These results support the value of the emotion-derived representation and its context alignment, while leaving open whether other learned hidden-state representations could provide comparable gains.

**Emotion–skill co-activation.** We also examine whether emotion–skill associations are semantically interpretable. Figure 4 aggregates 27dimensional emotion states at skill-selection steps across correctly solved episodes. Four patterns emerge: _curiosity_ and _desire_ co-activate with PRODUCTSEARCH; _confusion_ and _nervousness_ with QUERYREPHRASE; _approval_ and _optimism_ with PURCHASECONFIRM; and _annoyance_ and _disappointment_ with PRICECOMPARE. To assess semantic coherence, we ask GPT-5 to evaluate emotion–skill pairings from 200 sampled skillselection events. It judges **76** _._ **5** % of the pairings coherent. Comparison with human annotations on a 50-event subset yields Cohen’s _κ_ =0 _._ 81. Of the 23.5% flagged as inconsistent, 61% have lowintensity activations with max( **e** _t_ ) _<_ 0 _._ 15, suggesting that interpretation is harder when no single emotion dominates.

### 4.5 Generalization and Case Study

Figure 5 reports out-of-domain generalization on MATH (Hendrycks et al., 2021) and MBPP (Austin et al., 2021). We evaluate accuracy on 500 MATH problems and pass@1 on 374 MBPP tasks, using libraries of 6 and 5 skills, respectively. For both benchmarks, we _reuse_ the same emotion vectors from _L_^_∗_ =24 without re-extraction. Compared with Zero-Shot, EMOTION2SKILL raises MATH accuracy by 14.4 pp, from 54.8% to 69.2%, and MBPP pass@1 by 11.8 pp, from 60.0% to 71.8%. On

Figure 4: Emotion–skill co-activation on WebShop.Rows show 8 skills; columns show 12 GoEmotions dimensions. Four interpretable patterns emerge.

Figure 5: Out-of-domain generalization on MATH and MBPP with Qwen3-8B. Emotion vectors are reused from _L_^_∗_ =24 without re-extraction. Bars show accuracy/pass@1 of Zero-Shot vs. EMOTION2SKILL; ∆ denotes the improvement.

MATH, gains concentrate on Level 4–5 multi-step problems. On MBPP, gains appear on compositional tasks where the signal routes rising uncertainty to DECOMPOSEGUARD. Detailed case studies on WebShop and ALFWorld are provided in Appendix E.

## 5 Conclusion

We presented EMOTION2SKILL, which extracts LLM-internal emotion vectors as a proxy for the agent’s latent decision state and uses them to guide skill selection and evolution. Experiments on WebShop and ALFWorld show aggregate improvements over seven baselines across two backbones. Ablations support each component’s contribution, while co-activation analysis reveals semantically coherent emotion–skill associations. These findings suggest that internal affective representations provide decision-relevant information that complements textual routing signals. Future work could use feedback from skill execution to update emotion representations, extending the current one-way pathway into a closed-loop system in which emotion representations and skills adapt together.

## Limitations

EMOTION2SKILL requires residual-stream activations for both extraction and routing, which prevents direct use through text-only APIs. Our experiments use open-weight models, which provide the required access. Researchers within modeldeveloping organizations could also evaluate the framework on proprietary models if authorized to access their weights and inference-time activations. This access requirement is shared by representationbased methods that inspect or intervene on hidden states. It defines the settings in which the framework can be studied and deployed; extending emotion-guided routing to API-only models remains a useful direction.

## References

- Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models. _Preprint_ , arXiv:2108.07732.

- Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. 2023. Discovering latent knowledge in language models without supervision. In _International Conference on Learning Representations_ .

- Antonio R. Damasio. 1994. _Descartes’ Error: Emotion, Reason, and the Human Brain_ . Putnam, New York.

- Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. GoEmotions: A dataset of fine-grained emotions. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 4040–4054.

- Zelin He, Haotian Lin, Boran Han, Wei Zhu, Haoyang Fang, Bernie Wang, Xuan Zhu, Runze Li, and Matthew Reimherr. 2026. ReSkill: Reconciling skill creation with policy optimization in agentic RL. _Preprint_ , arXiv:2606.01619.

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks_ , volume 1.

- Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive learning. In _Advances in Neural Information Processing Systems_ , volume 33, pages 18661–18673.

- Jennifer S. Lerner, Ye Li, Piercarlo Valdesolo, and Karim S. Kassam. 2015. Emotion and decision making. _Annual Review of Psychology_ , 66:799–823.

- Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model. In _Advances in Neural Information Processing Systems_ , volume 36.

- Yu Li, Rui Miao, Zhengling Qi, and Tian Lan. 2026. ARISE: Agent reasoning with intrinsic skill evolution in hierarchical reinforcement learning. _Preprint_ , arXiv:2603.16060.

- Neel Nanda, Andrew Lee, and Martin Wattenberg. 2023. Emergent linear representations in world models of self-supervised sequence models. In _Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ , pages 16–30.

- Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. 2026. Trace2Skill: Distill trajectory-local lessons into transferable agent skills. _Preprint_ , arXiv:2603.25158.

- Kiho Park, Yo Joong Choe, and Victor Veitch. 2024. The linear representation hypothesis and the geometry of large language models. In _Proceedings of the 41st International Conference on Machine Learning_ , volume 235, pages 39643–39666.

- Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering Llama 2 via contrastive activation addition. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 15504–15522. Association for Computational Linguistics.

- Klaus R. Scherer. 2001. Appraisal considered as a process of multilevel sequential checking. In Klaus R. Scherer, Angela Schorr, and Tom Johnstone, editors, _Appraisal Processes in Emotion: Theory, Methods, Research_ , pages 92–120. Oxford University Press.

- Yaorui Shi, Yuxin Chen, Zhengxi Lu, Yuchun Miao, Shugui Liu, Qi Gu, Xunliang Cai, Xiang Wang, and An Zhang. 2026. Skill1: Unified evolution of skill-augmented agents via reinforcement learning. _Preprint_ , arXiv:2605.06130.

- Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In _Advances in Neural Information Processing Systems_ , volume 36.

- Yitong Shou and Manhao Guan. 2026. Mechanistic decoding of cognitive constructs in large language models. _Preprint_ , arXiv:2604.14593.

- Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning text and

- embodied environments for interactive learning. In _International Conference on Learning Representations_ .

- Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, Joshua Batson, Sam Zimmerman, Kelley Rivoire, Kyle Fish, Chris Olah, and Jack Lindsey. 2026. Emotion concepts and their function in a large language model. _Preprint_ , arXiv:2604.07729.

- Moran Sun, Tianlin Li, Yuwei Zheng, Zhenhong Zhou, Aishan Liu, Xianglong Liu, and Yang Liu. 2026. How emotion shapes the behavior of LLMs and agents: A mechanistic study. _Preprint_ , arXiv:2604.00005.

- Alexander M. Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. _Preprint_ , arXiv:2308.10248.

- Yash Vishe, Rohan Surana, Xunyi Jiang, Zihan Huang, et al. 2026. Skill-R1: Agent skill evolution via reinforcement learning. _Preprint_ , arXiv:2605.09359.

- Dingzirui Wang, Xuanliang Zhang, Keyan Xu, Qingfu Zhu, Wanxiang Che, and Yang Deng. 2025. AlignEvoSkill: Towards knowledge-aware and task-aligned agent skill evolution. _Preprint_ , arXiv:2506.23149.

- Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2024. Voyager: An open-ended embodied agent with large language models. _Transactions on Machine Learning Research_ .

- Wujiang Xu, Jiaojiao Han, Minghao Guo, Kai Mei, Xi Zhu, Han Zhang, and Dimitris N. Metaxas. 2026. AEL: Evolving agent harness in open-ended environments. _Preprint_ , arXiv:2604.21725.

- An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. 2025. Qwen3 technical report. _Preprint_ , arXiv:2505.09388.

- Min Yang, Jinghua Piao, Xu Xia, Xiaochong Lan, Jiaju Chen, Yongshun Gong, and Yong Li. 2026a. SkillMaster: Toward autonomous skill mastery in LLM agents. _Preprint_ , arXiv:2605.08693.

- Ruiyi Yang, Zechen Li, Hao Xue, Imran Razzak, et al. 2026b. MAGE: Multi-agent self-evolution with co-evolutionary knowledge graphs. _Preprint_ , arXiv:2605.10064.

- Shuo Yang, Jinyang Wu, Zhengxi Lu, Yuhao Shen, et al. 2026c. OPID: On-policy skill distillation for agentic reinforcement learning. _Preprint_ , arXiv:2606.26790.

- Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, et al. 2026d. SkillOpt: Executive strategy for self-evolving agent skills. _Preprint_ , arXiv:2605.23904.

- Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. WebShop: Towards scalable real-world web interaction with grounded language agents. In _Advances in Neural Information Processing Systems_ , volume 35, pages 20744–20757.

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In _International Conference on Learning Representations_ .

- Jianxiang Yu, Jiapeng Zhu, Bochen Lin, Qier Cui, Zichen Ding, and Xiang Li. 2026. Skill is not onesize-fits-all: Model-aware skill alignment for LLM agents. _Preprint_ , arXiv:2605.30723.

- Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. ExpeL: LLM agents are experiential learners. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 19632–19642.

- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. 2023. Representation engineering: A top-down approach to AI transparency. _Preprint_ , arXiv:2310.01405.

## Contents

- **1 Introduction 1**

- **2 Related Work 2**

- **3 Method 3** 3.1 Preliminaries . . . . . . . . . . . 3 3.2 Emotion Vector Extraction and Encoding . . . . . . . . . . . . . . . 3

- 3.3 Emotion-Augmented Skill Selection 4 3.4 Emotion-Driven Skill Evolution . 5

- **4 Experiments 5** 4.1 Setup . . . . . . . . . . . . . . . 5

- 4.1 Setup . . . . . . . . . . . . . . 4.2 Emotion Vector Quality and Layer Selection . . . . . . . . . . . .

- Selection . . . . . . . . . . . . . 6

- 4.3 Main Results . . . . . . . . . . . 7 4.4 Ablation Studies and Analysis . . 7 4.5 Generalization and Case Study . . 8

- 4.5 Generalization and Case Study . . 8

- **5 Conclusion 8**

|A |Emo|tion Stories Dataset|11|
|---|---|---|---|
||A.1|Dataset Overview . . . . . . . . .|11|
||A.2|Illustrative Story Examples . . . .|12|
|B|Exp|eriment Setup Details|12|
||B.1|Baseline Reproduction Details . .|12|
||B.2|Hyperparameter Settings . . . . .|12|
|C |The|oretical Analysis|13|
||C.1|Emotion2Skill Algorithm . . . . .|13|
||C.2|Contrastive Direction Extraction||
|||and Fisher LDA . . . . . . . . . .|13|
||C.3|Layer Selection Optimality . . . .|14|
||C.4|Confidence-Gated Selection: Ex-||
|||pected Utility Analysis . . . . . .|14|
||C.5|Transition-Point Criterion and||
|||Change Detection . . . . . . . . .|15|
||C.6|Skill Evolution: Convergence||
|||Analysis . . . . . . . . . . . . . .|15|
||C.7|Information-Theoretic Justification|15|
|D |Pro|mpt Templates and Skill Library|16|
||D.1|Prompt Templates . . . . . . . . .|16|
||D.2|Emotion Template Bank . . . . .|16|
||D.3|Skill Library Details . . . . . . .|16|
|E|Case|Studies|17|
|F|Add|itional Results|17|
||F.1|Evolution Round-by-Round. . . .|17|
||F.2|Per-Task Emotion Patterns on||
|||ALFWorld . . . . . . . . . . . . .|17|
||F.3|Confidence Score Distribution . .|17|
|G |Lim|itations and Broader Impact|17|
|H |Nota|tion Summary|19|

## Appendix

## A Emotion Stories Dataset

### A.1 Dataset Overview

The Emotion Stories dataset is a novel corpus created for EMOTION2SKILL’s contrastive direction extraction. It consists of 27 _×_ 100 = **2** _,_ **700** short stories, with 100 stories per GoEmotions concept (Demszky et al., 2020), generated by Qwen38B in non-thinking mode at temperature 0.

**Construction.** For each emotion concept, Qwen3-8B is prompted to write a 3–5 sentence story that naturally embodies the target emotion _without_ naming it. Two quality filters are applied:

(i) stories outside 2–8 sentences are regenerated; (ii) a zero-shot classifier must rank the target emotion in the top-3 predicted labels. Regeneration repeats until 100 passing stories per concept are collected. Each story averages 4.2 sentences and 67.3 tokens. No two stories within the same concept have cosine similarity _>_ 0 _._ 95 on Qwen3-8B embeddings, ensuring lexical and semantic diversity.

**Role in the pipeline.** Stories are used _only_ at the offline extraction stage; they are never shown to the agent during online execution. Mean-pooled token activations ( _t ≥_ 50) form the within-class means _µ_^(() _k_^(_ℓ_);subtracting the global mean)^(_µ_¯()^(_ℓ_)yields the) contrastive directions **v** _k_^(()^(_ℓ_)) (Eq. 1). The dataset will be released under CC BY 4.0 upon publication.

### A.2 Illustrative Story Examples

Three representative stories are shown below, one per valence group. In all cases the target emotion label does not appear as a literal word; human raters (3 annotators, majority vote) identified the correct concept in 89% of a 50-story random sample across all 27 concepts.

**Positive valence:** **_Gratitude_**

She had been staring at the rejection letter for twenty minutes when her phone buzzed with a voice message from her old professor. His words were unhurried: he had read her draft, revised it line by line, and submitted it to a different journal on her behalf. She set the phone down and looked out the window at the rain, feeling something settle quietly in her chest: not relief exactly, but the particular warmth of being looked after by someone who owed her nothing. She typed three words in reply and deleted them; no sentence seemed large enough.

**Negative valence:** **_Disgust_**

The commissary opened a new food counter and the line stretched past the elevator. He picked up a tray, slid it along the metal rail, and looked at the hot section: a ladle scraped the bottom of a grey-brown stew, and a spoon landed a glistening heap onto the tray of the person ahead. He watched a piece of underdetermined protein float to the surface and slowly rotate. He set his tray back on the stack, said nothing, and walked out into the cold air to find a vending machine.

**Cognitive/ambiguous:** **_Realization_**

She was halfway through explaining the project timeline to the new hire when the numbers stopped adding up. She paused, looked at the whiteboard, and traced her finger back to the start date. The deadline she had promised the client was six days before the date she had told her team: not a rounding error, an actual gap. She turned from the board slowly, and the new hire looked at her expression and asked if everything was fine.

## B Experiment Setup Details

### B.1 Baseline Reproduction Details

All baselines are reproduced on Qwen3-8B and Qwen3-14B in non-thinking mode with greedy decoding on 2 _×_ H800 GPUs. Evaluations use the standard WebShop test set (metrics: Score, Succ.%) and the ALFWorld json_2.1.1 split (134 test episodes across six task types; metric: per-type and average Success%). **Zero-Shot** : task description and observation only, no chain-of-thought or skill descriptions. **ReAct** : standard Thought/Action alternation; retrieval augmentation omitted as it is incompatible with our evaluation protocol. **Reflexion** : episodic verbal self-critique prepended to the next episode’s context, up to 3 reflections; cleared at each new task. **ExpeL** : top-5 experiences retrieved by BM25 from 50 disjoint training episodes injected at test time. **EvoSkill** : a skill-evolution baseline evaluated with the same backbones, benchmark splits, decoding protocol, and skill-library interface as the other methods. **MASA** : SOPs distilled from training traces; skill selection driven by text-based observation similarity. No public Qwen3 code exists; we reimplement from the paper description. Both MASA and EMOTION2SKILL use the _same_ skill libraries (Appendix D.3) to ensure a fair comparison.

### B.2 Hyperparameter Settings

**Emotion encoder** _fθ_ **.** 3-layer MLP, 27 _→_ 64 _→_ 128 _→_ 128, ReLU, _<_ 27K parameters. Warm-up data: 500 episodes under ReAct+skill baseline (250 WebShop, 250 ALFWorld), _∼_ 3,200 tuples after filtering steps with max( **e** _t_ ) _<_ 0 _._ 05. AdamW, lr = 1e-3, weight decay 1e-4, seed 42, batch 64, max 50 epochs (patience 5). _λ_ =0 _._ 5, _κ_ =0 _._ 07, _κ_^(_′_) =0 _._ 1. Training time: _<_ 10 min on 2 _×_ H800 GPUs; encoder is trained once per backbone and reused across all benchmarks.

**Emotion direction extraction.** Candidate layers: _{_ 16 _,_ 18 _,_ 20 _,_ 22 _,_ 24 _,_ 26 _,_ 28 _}_ for Qwen3-8B (36 layers, _L_^_∗_ =24); _{_ 18 _,_ 20 _,_ 22 _,_ 24 _,_ 26 _,_ 28 _,_ 30 _}_ for Qwen3-14B (40 layers, _L_^_∗_ =26). For each backbone, _L_^_∗_ is selected by hit-any accuracy on the GoEmotions validation split; the test split is held out until after selection. PCA denoising: top-10 principal components of 100 neutral-story activations projected out. Token prefix _t <_ 50 excluded

#### from mean pooling.

**Skill evolution.** _R_ =3 rounds, 100 episodes per round (50 WebShop + 50 ALFWorld). Performance threshold _η_ =0 _._ 4; diagnostic window _w_ =1 (3-step context); rewriter: same backbone, greedy decoding, max SOP 200 words.

**Inference.** All methods: greedy decoding, Qwen3 non-thinking mode, on 2 _×_ H800 GPUs. Qwen3-8B: max context 4096 tokens. Qwen3-14B: max context 8192 tokens. _τ_ =0 _._ 3 selected on a 100-episode WebShop validation split. Results reported as mean _±_ std over 3 independent runs with different episode orderings.

## C Theoretical Analysis

This section provides rigorous theoretical grounding for five core components of EMOTION2SKILL: contrastive direction extraction, layer selection optimality, confidence-gated selection with expected utility bounds, the transition-point criterion and its connection to sequential change detection, skill-evolution convergence, and an information-theoretic justification for emotionaugmented prompting.

### C.1 Emotion2Skill Algorithm

The complete pseudocode for online skill selection and periodicskill evolution is provided in Algorithm **??** .

### C.2 Contrastive Direction Extraction and Fisher LDA

**Setup.** Let **h**^(() _k,n_^(_ℓ_))^(_∈_R)^(_d_denote the residual-stream) activation at layer _ℓ_ for the _n_ -th token of a story from emotion class _k ∈{_ 1 _, . . . , K}_ ( _K_ =27). Define the within-class mean _µ_^(() _k_^(_ℓ_)) = _N_^(1)  _n_^(**h**() _k,n_^(_ℓ_)and) the global mean _µ_ ¯^(()^(_ℓ_)) = _K_ 1  _k_^(_µ_() _k_^(_ℓ_).) The contrastive direction is **v** _k_^(()^(_ℓ_)) = _µ_^(() _k_^(_ℓ_))^(_−µ_¯()^(_ℓ_), and the pro-) jection matrix is _We_ = [ **v** 1 _, . . . ,_ **v** _K_ ]^(_⊤_) _∈_ R^(_K×d_) .

**Fisher LDA connection.** The between-class scatter matrix is _SB_ = _K_ 1  _k_^**v**^_k_^**v** _k_^(_⊤_andthewithin-) class scatter is _SW_ = _KN_ 1  _k,n_^(()^**h**^(_k,n−µk_)()^**h**^(_k,n−_) _µk_ )^(_⊤_) . The Fisher LDA objective maximizes the Rayleigh quotient

The columns of _We_ span the range of _SB_ , which is the subspace that maximizes between-class variance. Under the standard assumption _SW ≈ σ_^2 _I_

#### Algorithm 1: EMOTION2SKILL

- **Require:** Skill library _S_ , extractor _We_ , encoder _fθ_ , threshold _τ_ , layer _L_^_∗_ , rounds _R_ , threshold

- _η_

- 1: **Online: Emotion Extraction and Skill Selection**

- 2: **for** each episode **do**

- 3: _L_ ep _←∅_

- 4: **for** each decision step _t_ **do**

- 5: Extract **e** _t ← We_ **h**^(() _T_^(_L∗_)) _/∥We_ **h**^(() _T_^(_L∗_)) _∥_ 2;

- encode ( **r** _t, ct_ ) _← fθ_ ( **e** _t_ )

- 6: Construct _Pt_ via Eq. 8 (include _Tt_^(_∗, ct_if) _ct ≥ τ_ ; omit otherwise)

- 7: _s_^_∗_ _t_^(_←_arg max)^_s_ _i_^(_pM_()^(_si| Pt_); execute)^_s_ _t_^_∗_

- 8: Append ( **e** _t, ct, s_^_∗_ _t_^(_, T_) _t_^(_∗_) to)^_L_ep

- 9: **end for**

- 10: **end for**

- 11:

- 12: **Periodic: Emotion-Augmented Skill Evolution**

- 13: **for** round _r_ = 1 _, . . . , R_ **do**

- 14: Compute emotion shifts _δt_ (Eq. 10) and transition points _P_ (Eq. 11) from episode logs

- 15: **for** each _si ∈U_^(()^(_r_)) with success _< η_ **do** 16: Build _P_ evo^(()^(_i_)from diagnostics around)^_P_ _i_ (Eqs. 12–13)

- 17: _Di_^(_′←M_()^_P_ evo^(()^(_i_));)^(_S←_()^(_S \{s_) _i_^(_}_))^(_∪{s′_) _i_^(_}_)

- 18: **end for**

- 19: **end for**

(shared isotropic covariance), maximizing _J_ ( **w** ) reduces to maximizing **w**^(_⊤_) _SB_ **w** , and the optimal directions are exactly _{_ **v** _k}_ . The PCA denoising step Π^(_⊥_) **h** = **h** _− V_ 0 _V_ 0^(_⊤_)^(**h**, where)^(_V_0 are the top-10) principal components of neutral-story activations, further removes shared stylistic variance, approximating the whitening _SW_^(_−_1)^(_/_2) that the full LDA solution requires.

**Bayes-optimal approximation.** Under a sharedcovariance Gaussian model **h** _k,n ∼N_ ( _µk,_ Σ), the Bayes-optimal linear classifier assigns a test point **h** to class arg max _k_ ( _µk − µ_ ¯)^(_⊤_) Σ^(_−_1) **h** . When Σ _≈ σ_^2 _I_ , this reduces to arg max _k_ **v** _k_^(_⊤_)^(**h**, which is) precisely the projection implemented by _We_ . Empirically, the isotropic assumption holds at late-middle layers (L*=24 for Qwen3-8B) where the ratio _∥SW −σ_ ˆ^2 _I∥F /∥SW ∥F <_ 0 _._ 12, justifying the approximation.

**Proposition 1** (Between-class separability margin). _Let_ **e** ˆ _t_ = _We_ **h** _t/∥We_ **h** _t∥_ 2 _be the ℓ_ 2 _-normalized emotion vector. Under the Gaussian model with covariance_ Σ _, the expected cosine similarity gap between same-class and cross-class pairs satisfies_

_∼ ∼ Proof sketch._ For **h** _N_ ( _µk,_ Σ), _We_ **h** _N_ ( _Weµk, We_ Σ _We_^(_⊤_).Theexpectedinnerprod-) uct E[ˆ **e**^(_⊤_) _t_^**e**ˆ^(_t′_])^(_k_=)^(_k′_concentratesnear1as)^(_d→_) _∞_ by the Johnson–Lindenstrauss lemma, while E[ˆ **e**^(_⊤_) _t_^**e**ˆ^(_t′_])^(_k̸_=)^(_k′_isboundedaboveby1)^(_−∥_)^**v**^(_k−_) **v** _k′∥_^2 _/_ (2 _∥_ Σ _∥F_ ) via a second-order Taylor expansion of the cosine under Gaussian perturbations. The gap follows by subtraction.

Global-mean subtraction increases _∥_ **v** _k −_ **v** _k′∥_^2 by removing the shared offset, directly amplifying the RHS of Eq. (17). This explains why our contrastive extraction outperforms raw within-class means in the GoEmotions probe (37.2% vs. 29.6% at L*=24).

### C.3 Layer Selection Optimality

**Fisher discriminability ratio.** For each candidate layer _ℓ_ , define the Fisher ratio which measures the ratio of between-class to within-class variance in the projected space. _F_ ( _ℓ_ ) is the multivariate analogue of the univariate signalto-noise ratio.

**Proposition 2** (Fisher ratio as a separability surrogate). _Under the isotropic shared-covariance Gaussian model, increasing F_ ( _ℓ_ ) _increases the average between-class separation of the linear probe and therefore provides a separability diagnostic._

_Proof sketch._ Under _N_ ( _µk,_ Σ^(()^(_ℓ_)) ) at each layer, the Bayes error for a _K_ -class linear classifier depends only on the pairwise Mahalanobis distances ∆^(() _kk_^(_ℓ_))^(_′_=()^(_µk−µk′_))^(_⊤_(Σ()^(_ℓ_)))^(_−_1()^(_µk−µk′_).Inthe) isotropic case Σ^(()^(_ℓ_)) = _σℓ_^2^_I_,theaveragepairwise distance is proportional to tr( _SB_^(()^(_ℓ_)))^(_/σ_) _ℓ_^2^(_∝F_()^(_ℓ_).) Maximizing _F_ ( _ℓ_ ) therefore maximizes the average separation under this model.

The non-monotonic behavior of _F_ ( _ℓ_ ) across layers (Figure 3, panel (c)) arises because early layers encode syntactic features that dilute emotion structure (low _SB_ ), while very late layers collapse into output-prediction subspaces that increase _SW_ . The late-middle layers selected by validation (L*=24 for Qwen3-8B, L*=26 for Qwen3-14B) are consistent with mechanistic interpretability findings that concept-level representations peak at approximately 2 _/_ 3 depth.

### C.4 Confidence-Gated Selection: Expected Utility Analysis

**Expected utility framework.** Let _S_ be the skill library. For a given step _t_ , let _q_ 0( _s_ ) = _q_ ( _s | xt_ ) denote the text-only skill distribution and _qϕ_ ( _s_ ) = _q_ ( _s | xt,_ **_ϕ_** _t_ ) the emotion-augmented distribution. Let _R_ ( _s_ ) _∈_ [0 _,_ 1] be the expected per-step reward of skill _s_ . Define the utility gain from emotion augmentation as

Augmentation is beneficial iff ∆ _Ut >_ 0.

**Confidence as a utility predictor.** The encoder output **r** _t ∈_ R^128 summarizes the emotion state, and the confidence score _ct_ = _σ_ ( **w** _c_^(_⊤_)^**r**^(_t_+)^(_bc_)is) trained via binary cross-entropy against episode success labels.

**Proposition 3** (Gate as utility filter). _Under the assumption that ct is a consistent estimator of_ Pr[∆ _Ut >_ 0 _|_ **r** _t_ ] _, the expected utility of the gated policy πτ_ ( _s | xt_ ) = _qϕ_ ( _s_ ) **1** [ _ct ≥ τ_ ] + _q_ 0( _s_ ) **1** [ _ct < τ_ ] _satisfies_

_Proof._ Decompose: E[∆ _Ut_ ] = E[∆ _Ut ·_ **1** [ _ct ≥ τ_ ]] + E[∆ _Ut ·_ **1** [ _ct < τ_ ]]. Since _R_ ( _s_ ) _∈_ [0 _,_ 1], we have _|_ ∆ _Ut| ≤_ 1, so E[∆ _Ut ·_ **1** [ _ct < τ_ ]] _≤_ Pr[ _ct < τ ∧_ ∆ _Ut >_ 0]. Rearranging gives Eq. (20).

The RHS of Eq. (20) is the probability of a “missed benefit” (useful emotion signal suppressed by the gate). Setting _τ_ =0 _._ 3 reduces this term to approximately 3.1% on our validation split, while filtering 28% of steps where _ct < τ_ and ∆ _Ut <_ 0 would have caused misrouting.

**Regret bound.** Let _T_ be the episode length and _π_^_∗_ the oracle policy. The per-episode regret of _πτ_ relative to _π_^_∗_ satisfies

Both terms decrease as the encoder improves; the gate balances them by choosing _τ_ to minimize their sum on the validation split.

### C.5 Transition-Point Criterion and Change Detection

**Definition and** _z_ **-score interpretation.** The shift magnitude is _δt_ = 1 _−_ **e** ˆ^(_⊤_) _t−_ 1^**e**ˆ^(_t_(cosine distance be-) tween consecutive normalized emotion vectors). A transition point is declared when _δt > δ_^(¯) + _σδ_ , where _δ_ ¯ and _σδ_ are the running mean and standard deviation over the episode so far. This is equivalent to a one-sided _z_ -score test with _z ≈_ 1, corresponding to a 15.9% false-positive rate under Normality.

**Cosine vs. Euclidean distance.** Cosine distance _δt_^cos = 1 _−_ **e** ˆ^(_⊤_) _t−_ 1^**e**ˆ^(_t_is invariant to uniform scaling) of **e** _t_ : if **e** _t → α_ **e** _t_ for _α >_ 0, then ˆ **e** _t_ is unchanged and _δt_^cos does not fire. By contrast, Euclidean distance _∥_ **e** _t −_ **e** _t−_ 1 _∥_ 2 scales as _α_ , generating false positives whenever the overall activation magnitude changes without a genuine directional shift. **Proposition 4** (KL–cosine equivalence under Gaussian model). _Let pt_ = _N_ ( **e** _t, σ_^2 _I_ ) _and pt−_ 1 = _N_ ( **e** _t−_ 1 _, σ_^2 _I_ ) _. Then the symmetrized KL divergence satisfies_

_where_ cos _θt,t−_ 1 = ˆ **e** _t_^(_⊤_) _−_ 1^**e**ˆ^(_t._)

_Proof._ The symmetrized KL between two isotropic Gaussians with means _µ_ 1 _, µ_ 2 and common variance _σ_^2 _I_ is _∥µ_ 1 _− µ_ 2 _∥_^2 _/σ_^2 . Expanding _∥_ **e** _t −_ **e** _t−_ 1 _∥_^2 and normalizing by _∥_ **e** _∥_^2 gives the cosine form.

A large _δt_ therefore indicates an abrupt distributional shift in the agent’s internal belief, precisely the signal that warrants skill-library revision.

**Connection to CUSUM.** The standard CUSUM statistic is _Ct_ = max(0 _, Ct−_ 1 + _δt − ν_ ). Our criterion is a one-shot CUSUM with _ν_ = _δ_^(¯) + _σδ_ and no memory ( _Ct−_ 1 = 0 always), trading detection power for simplicity in the single-episode setting. Multi-episode aggregation recovers the statistical power of a full CUSUM with _O_ (episodes) samples, providing asymptotic consistency.

**Corollary 1** (Asymptotic consistency). _As the number of evolution-round episodes Nep →∞, the proportion of failed episodes sharing a transition point at a given step converges to the true probability of a skill-relevant shift at that step, by the law of large numbers._

### C.6 Skill Evolution: Convergence Analysis

Let _ψi_^(()^(_r_)) denote the SOP for skill _i_ after round _r_ , and _ρ_^(() _i_^(_r_)) = Pr[success _| ψi_^(()^(_r_)]thesuccessrate.) Define Ψ^(()^(_r_)) = _M_ 1  _Mi_ =1^(_ρ_() _i_^(_r_).)

**Proposition 5** (Monotone improvement). _If the LLM rewriter consistently incorporates the root cause in_ ∆ _i, then_ Ψ^(()^(_r_+1)) _≥_ Ψ^(()^(_r_)) _with equality only when no skill is flagged._

_Proof sketch._ Skills with _ρ_^(() _i_^(_r_)) _≥ η_ are unchanged. For flagged skills, ∆ _i_ identifies a failure pattern shared by fraction _p >_ 0 of failed episodes. A rewrite eliminating this pattern increases _ρ_^(() _i_^(_r_+1)) by at least _p ·_ Pr[failure]. Summing over flagged skills gives Ψ^(()^(_r_+1)) _>_ Ψ^(()^(_r_)) .

**Corollary 2** (Convergence). _The sequence {_ Ψ^(()^(_r_)) _} converges in at most M rounds. In practice, convergence occurs in R ≤_ 3 _rounds because multiple skills are flagged simultaneously._

**Noise robustness.** Let _α ∈_ (0 _,_ 1] be the fraction of the root cause addressed by the rewriter. Then _ρ_^(() _i_^(_r_+1)) _≥ ρ_^(() _i_^(_r_)) + _α · p ·_ Pr[failure], so Ψ^(()^(_r_+1)) _>_ Ψ^(()^(_r_)) as long as _α >_ 0. Emotion-based diagnostics provide a _localized_ root cause, increasing _α_ relative to outcome-only diagnostics.

### C.7 Information-Theoretic Justification

**Mutual information lower bound.** Let _S_^_∗_ be the optimal skill, _Xt_ the textual observation, and _Et_ the emotion vector. By the chain rule,

**Proposition 6** (Strict information gain). _If Et is not conditionally independent of S_^_∗_ _given Xt, then I_ ( _S_^_∗_ ; _Xt, Et_ ) _> I_ ( _S_^_∗_ ; _Xt_ ) _._

_Proof. I_ ( _S_^_∗_ ; _Et | Xt_ ) _>_ 0 iff conditional independence fails, which holds when Pr[ _S_^_∗_ _| Et, Xt_ ] is non-constant in _Et_ . The result follows from Eq. (23).

**Empirical support.** The conditional entropy _H_ ( _S_^_∗_ _| Xt_ ) estimated on WebShop validation episodes is 1 _._ 84 bits, while _H_ ( _S_^_∗_ _| Xt, Et_ ) drops to 1 _._ 31 bits, yielding _I_ ( _S_^_∗_ ; _Et | Xt_ ) = 0 _._ 53 bits: a 22.3% reduction in skill-selection uncertainty attributable to the emotion signal.

**Gate as information bottleneck.** The confidence gate admits _Et_ only when the encoder is confident, i.e., when the estimated _I_ ( _S_^_∗_ ; _Et | Xt_ ) is high. This prevents noisy emotion representations from increasing _H_ ( _S_^_∗_ _| Xt, E_^ˆ _t_ ) above _H_ ( _S_^_∗_ _| Xt_ ), maintaining the monotonicity of the MI lower bound in Eq. (23) even under encoder uncertainty.

## D Prompt Templates and Skill Library

### D.1 Prompt Templates

All prompts use greedy decoding (temperature 0); runtime fields appear in {braces}. Tables 7–10 present the four prompts in a compact tabular format to avoid margin overflow.

**Emotion Story Generation Prompt** (§A, 100 stories per concept)

You are a creative writer. Write a short story (3–5 sentences) that clearly expresses the emotion: {emotion_label} Requirements: (1) Naturally embody this emotion throughout. (2) Do NOT mention the emotion name. (3) Use concrete situations and sensory details. Story:

Table 7: Emotion story generation prompt used in offline extraction.

### D.2 Emotion Template Bank

Table 11 lists the _C_ =12 natural-language emotionstate templates and their learnable prototypes **q** _j_ (Eq. 3). Templates are task-agnostic and reused across all benchmarks.

### D.3 Skill Library Details

**WebShop (8 skills).** SOPs distilled from MASA training traces (Yu et al., 2026). Each SOP specifies trigger condition, action sequence, and exit criteria. PRODUCTSEARCH (search by keywords; exit: _≥_ 3 candidates), QUERYREPHRASE (reformulate when _<_ 2 results; exit: candidates found or 3 attempts exhausted), PRODUCTCOMPARE (compare attributes; exit: single preferred item), PRICECOMPARE (find cheapest compliant item), REVIEWANALYSIS (score sentiment from reviews), OPTIONSELECT (set size/color/quantity), CARTMANAGE (add to cart and verify), PURCHASECONFIRM (confirm all constraints before purchase).

|Skill Selection Prompt(constructs_Pt_, Eq. 5)|
|---|
|You are an agent selecting the most appropriate skill.|
|Task: {task_description}|
|History: {history}|
|Observation: {observation}|
|Available Skills:|
|{index}. {skill_name}: {description} | SOP:|
|{sop_summary}|
|[repeat for each skill]|
|_[Included only when ct ≥τ:]_|
|Emotional State: {T_t_star} (confidence: {c_t:.2f})|
|Use the emotional state as auxiliary context|
|for skill selection.|
|Respond with only the skill number (integer).|

Table 8: Skill selection prompt. The “Emotional State” block is omitted when _ct < τ_ (text-only fallback).

|Transition Verbalization Prompt(generates∆_i_)|
|---|
|You are an analyst reviewing an agent’s episode|
|trajectory for skill: “{skill_name}”.|
|Transition Points: {formatted_transition_contexts}|
|Each entry: step, _δt_, 3-step window of|
|observations / skills / templates / shift magnitudes.|
|Episode outcome: {outcome}|
|Provide a 2–3 sentence summary: (1) pattern of|
|internal state shifts; (2) why the SOP caused|
|or failed to prevent them; (3) which SOP aspect|
|should be revised. Do not rewrite the SOP here.|

Table 9: Transition verbalization prompt; output ∆ _i_ is injected into the evolution prompt.

**ALFWorld (6 skills).** Following SkillOpt (Yang et al., 2026d); one SOP per task type. Each SOP covers: navigation sequence, object interaction primitives, and error-recovery procedures (e.g., HEAT OBJECT: pick up _→_ verify appliance is open _→_ heat _→_ place; PICK TWO: carry objects one at a time, track first placement).

**MATH (6 skills).** DIRECTSOLVE (formula/definition; Level 1–2), ALGEBRAMANIP (equation setup), DECOMPOSE (multi-step sub-goals; Level 3–4), CASEWORK (exhaustive enumeration), GEOMVISUALIZE (mental geometric model), VERIFYCHECK (re-derive via alternate method).

|Skill Evolution Prompt(_P_ ^(()^(_i_)) evo ^(, Eq. 11))|
|---|

You are an expert agent designer improving a skill SOP. **Skill:** {skill_name} | Success rate: {rate:.1%} (threshold: {eta:.1%}) **Current SOP:** {current_sop} **Critical Transition Points** (step, observations, skills, templates, _δ_ values): {transition_contexts} **Transition Summary:** {Delta_i} **Episode Outcome:** {outcome} **Instructions:** Rewrite the SOP to (1) add guidance for situations that caused abrupt state shifts, (2) add exit/escalation criteria, (3) preserve format and detail, (4) stay within 200 words. Revised SOP:

Table 10: Skill evolution prompt. Transition diagnostics _{_ **d** _t}_ and verbalized summary ∆ _i_ are concatenated before the SOP rewriter.

**MBPP (5 skills).** DIRECTIMPLEMENT (from docstring), DECOMPOSEGUARD (helpers + edge-case guards), BRUTEFORCEOPTIMIZE (bruteforce then optimize), PATTERNRECOGNIZE (sliding window / two-pointer / DP), TESTDRIVEN (assert boundary cases first, then implement).

## E Case Studies

We present three concrete traces illustrating how EMOTION2SKILL changes agent behavior across skill selection, skill evolution, and pre-action intervention.

## F Additional Results

### F.1 Evolution Round-by-Round

Table 12 compares emotion-guided evolution against outcome-only evolution across the three rounds.

### F.2 Per-Task Emotion Patterns on ALFWorld

Table 13 shows the most frequently activated template per ALFWorld task type on correctly-solved episodes. Procedure-heavy tasks (Heat, PickTwo) activate frustration/confusion, consistent with their multi-step error-recovery demands.

### F.3 Confidence Score Distribution

Figure 9 shows the distribution of encoder confidence scores _ct_ across all WebShop test steps. The distribution is right-skewed, with the bulk of steps concentrated in the range _ct ∈_ [0 _._ 30 _,_ 0 _._ 60] and a sparse high-confidence tail above 0.70. A substantial minority of steps (28%) fall below _τ_ =0 _._ 3 and trigger the text-only fallback, reflecting genuine

|ID|State|Template (abbreviated)|
|---|---|---|
|T1|Curious|High curiosity; actively seeking new information.|
|T2|Desiring|Strong anticipation toward a spe- cific goal.|
|T3|Confused|Confusion/uncertainty; current strategy may fail.|
|T4|Approving|Approval and confidence in the current direction.|
|T5|Optimistic|Positive expectation about up- coming actions.|
|T6|Proud|Pride and self-assurance; high internal confidence.|
|T7|Frustrated|Frustration/annoyance; repeated obstacles.|
|T8|Disappointed|Disappointment; unmet expecta- tions.|
|T9 |Nervous |Nervousness/apprehension; per- ceived risk. |
|T10|Realizing|Sudden realization or insight; strategy shift.|
|T11|Amused|Light amusement; low-stakes engagement.|
|T12|Neutral|Neutral baseline; no strong af- fective signal.|

Table 11: Emotion-state template bank ( _C_ =12). The template with highest cosine similarity to **r** _t_ is selected (Eq. 3) and injected when _ct ≥ τ_ .

|Method|Round 1 |Round 2 |Round 3|
|---|---|---|---|
|Evo-Outcome (no|26.8|27.5|27.3|
|diagnostics)||||
|Evo-Emotion (ES)|27.2|28.9|33.3|
|∆|+0.4|+1.4|+6.0|

Table 12: WebShop Succ. (%) across evolution rounds (Qwen3-8B). Emotion diagnostics yield compounding gains; the gap widens to +1.6 pp by round 3 as more transition evidence accumulates.

uncertainty at ambiguous or transitional states. The remaining 72% are admitted by the gate and benefit from emotion-augmented skill selection.

## G Limitations and Broader Impact

**Limitations.** White-box residual-stream access is required, limiting direct application to APIserved LLMs; distilling routing signals to black-box models is a natural extension. The warm-up phase collects _∼_ 500 episodes ( _<_ 10 min); few-shot encoder initialization could reduce this. Emotionguided routing does not uniformly dominate all task types: on ALFWorld Cool tasks, EMOTION2SKILL is outperformed by MASA on Qwen3-8B, suggesting that tasks with highly stereotyped action sequences may benefit less from emotion-based rerouting and could benefit from adaptive per-task

**Case A Emotion-Augmented Skill Selection** _WebShop | Qwen3-8B | single episode trace_

**Scenario.** The agent must purchase a blue cotton shirt under $50. Twelve results are returned initially, all priced $38–$78. A naïve price filter returns zero matches and the session risks entering a failure loop.

**Key observation.** At step 3, _before_ the explicit “No results” feedback arrives, the encoder reads T3 Confused ( _ct_ =0 _._ 36 _> τ_ =0 _._ 3), reflecting accumulated uncertainty about the search trajectory. EMOTION2SKILL routes to QUERYREPHRASE one step earlier than any reactive method could; the text-only baseline repeats PRODUCTSEARCH and enters a failure loop at step 4.

|Step|Observation|Template|_ct_|Gate|Baseline|ES-8B|
|---|---|---|---|---|---|---|
|1|Goal: blue cotton shirt_<_$50|T1 Curious|0.62|✓|P.Search|P.Search|
|2|12 results; prices $38–$78|T2 Desiring|0.43|✓|P.Search|P.Compare|
|3|Filter applied; 0 items_<_$50|T3 Confused|0.36|✓|P.Search|Q.Rephrase|
|4|“No results for original query”|T3 Confused|0.37|✓|_P.Search (loop)_|Q.Rephrase|
|5|6 new results; prices $34–$52|T5 Optimistic|0.41|✓|P.Compare|P.Compare|
|6|Item #3: $38,⋆4.2, in stock|T4 Approving|0.58|✓|Purchase|Purchase|
|_Episo_|_de outcome_||||Failure(loop)|Success|

**Analysis.** Confidence stays at _ct ≥_ 0 _._ 36 throughout, so the gate admits the signal at every step. The Confused _→_ Confused pattern at steps 3–4 ( _δt_ =0 _._ 42 _> δ_^(¯) + _σδ_ ) is flagged as a transition point and used as diagnostic evidence in Case B. Figure **??** plots the full trajectory: (a) _ct_ never drops below _τ_ ; (b) template shifts from T3 to T5 at step 5, coinciding with the successful rephrase; (c) _δt_ spikes at step 4, marking the transition point.

Figure 6: **Case A: Emotion-Augmented Skill Selection.** The emotion encoder detects T3 Confused at step 3 (one step before explicit failure feedback) and reroutes to QUERYREPHRASE, while the text-only baseline enters a failure loop. The bottom panel shows the full emotion trajectory including encoder confidence _ct_ , selected template _Tt_^_∗_, and shift magnitude _δt_ .

### Case B Emotion-Driven Skill Evolution

_WebShop | Qwen3-8B | Round 2 evolution_

**Scenario.** After round 1, QUERYREPHRASE succeeds in only 38% of episodes, below _η_ =40%, and is flagged for evolution. The system aggregates transition diagnostics across all 38 failed episodes and synthesizes a verbalized summary ∆ _i_ before rewriting the SOP.

**Diagnostic evidence.** Two transition points are identified in the Case A episode: step 4 ( _δ_ =0 _._ 42, Confused _→_ Confused) and step 6 ( _δ_ =0 _._ 39, Confused _→_ Confused). Aggregating across round-1 failures, 79% share this pattern, concentrated at rephrasing steps with no exit condition. The verbalized summary ∆ _i_ produced by the LLM rewriter:

_“The agent repeatedly enters Confused states during query rephrasing with no exit criterion or attempt budget defined in the SOP. Each attempt prolongs the Confused trajectory without a principled escalation path, leading to session timeout or abandoned tasks. The SOP should impose a maximum rephrase count and specify an explicit fallback skill when the budget is exhausted.”_

### SOP rewrite (Round 2).

|Original SOP (Round 1)|Evolved SOP (Round 2)|
|---|---|
|When search results do not match the target, try synonyms or broader terms and re-search. Continue rephrasing until a satisfactory match is found or the task is abandoned. No fixed limit on rephrasing attempts.|(1)Identify the specific failing keyword or constraint. (2)Try up to3 alternative queries: synonyms_→_broaden the core noun_→_drop the most restrictive attribute. (3)If all 3 fail,exit to PRODUCTSEARCHwith core noun and price range only. (4)Never rephrase more than 3 times; escalate promptly to prevent Confused-state loops.|

_Triggers: δ_ =0 _._ 42 _at step 4 and δ_ =0 _._ 39 _at step 6 (Confused→Confused in 79% of failures). Root cause: no rephrasing budget or exit criterion in the SOP. Round-2_ QUERYREPHRASE _success rate: 38%→54% (_ + _16 pp); WebShop Succ. 27.2→28.9%._

**Analysis.** The three SOP changes (ordered query strategy, fixed attempt budget, and mandatory escalation path) are each directly traceable to a specific finding in the diagnostic signal. None would be discoverable from binary episode outcomes alone, which indicate only that QUERYREPHRASE failed, without localizing where or why within the episode.

Figure 7: **Case B: Emotion-Driven Skill Evolution.** Transition diagnostics identify a recurring Confused _→_ Confused pattern (79% of failures) and produce a verbalized summary that directly drives three targeted SOP revisions, raising QUERYREPHRASE success from 38% to 54%.

Transition diagnostics identify a recurring bility of agent systems. In high-stakes domains (healthcare, finance), the confidence-gating mechanism provides a built-in safety fallback when the emotion signal is uncertain. Access to internal model states could theoretically be misused for surveillance; responsible deployment requires appropriate access controls. On balance, we believe the transparency and adaptive-control benefits outweigh the risks.

|Task|Top Template|Freq.|
|---|---|---|
|Pick|T1 Curious|42%|
|Look|T4 Approving|38%|
|Clean|T3 Confused|35%|
|Heat|T7 Frustrated|41%|
|Cool|T10 Realizing|33%|
|PickTwo|T3 Confused|39%|

Table 13: Most frequent emotion template per ALFWorld task type (Qwen3-8B, correctly-solved episodes only).

## H Notation Summary

thresholding as future work. Instruction-tuned variants with modified hidden-state geometry may require re-extraction (Qwen3-8B-Instruct GoEmotions accuracy: 28.1% vs. 37.2% on the base model). Extension to multi-agent coordination is left open.

Table 14 provides a consolidated reference for all mathematical symbols used in the paper, grouped by subsystem.

**Broader Impact.** Making LLM internal states actionable improves interpretability and debugga-

**Case C Pre-Action Emotion Signal Prevents Failure** _ALFWorld Heat | Qwen3-8B | pre- vs. post-action routing_

**Scenario.** Heat tasks show the largest gain of any ALFWorld task type: 9.6% _→_ 56.9% (+47.3 pp) on Qwen3-8B. The canonical failure is issuing a heat action on a closed appliance. We trace a representative episode: heat a mug and place it on the counter.

|Step|Observation|Template|_ct_|Gate|Baseline action|ES-8B action|
|---|---|---|---|---|---|---|
|1|Task: heat mug, place on counter|T1 Curious|0.37|✓|navigate microwave|navigate microwave|
|2|Microwave located; door status unknown|T9 Nervous|0.34|✓|_heat mug_(direct)|examine mi- crowave|
|3|Microwave is closed|T3 Confused|0.38|✓|_action failed_|open microwave|
|4|Microwave now open|T5 Optimistic|0.55|✓|navigate microwave (retry)|heat mug|
|5|Mug heated successfully|T4 Approving|0.61|✓|examine microwave|place mug on counter|
|6|—|—|—|—|_loop; task fails_|task complete|
|_Episo_|_de outcome_||||Failure|Success|

**Key mechanism: pre-action vs. post-action signal.** At step 2 the baseline issues heat mug immediately; failure feedback arrives only _after_ the incorrect action. EMOTION2SKILL reads T9 Nervous ( _ct_ =0 _._ 34), reflecting internal apprehension about an unverified precondition, _before_ any action is taken, and routes to examine _→_ open _→_ heat.

**Round-1 evolution.** The T9 Nervous spike at step 2 ( _δt_ =0 _._ 38, exceeding threshold) is flagged for HEAT OBJECT. Across round-1 failures, 11 of 14 failed Heat episodes share the same spike before a failed heat action. The verbalized summary ∆ _i_ :

_“The agent consistently issues heat actions without verifying appliance state. T9 Nervous activations immediately before heating indicate internal uncertainty about unresolved preconditions. Add an explicit precondition check: verify the appliance is open before any heating action.”_

### SOP before/after (Round 1).

|Original HEAT OBJECT SOP|Evolved SOP (after Round 1)|
|---|---|
|Navigate to the target appliance.|Navigate to the target appliance.|
|Pick up the object to be heated.|Verify the appliance is open; if closed, open it first.|
|Place the object inside the appliance.|Pick up the object to be heated.|
|Activate heating.|Place the object inside the appliance.|
|Retrieve the object once complete.|Activate heating.|
|Place the object at the goal location.|Retrieve the object once complete. Place the object at the goal location.|

_Change: one precondition check prepended (bold). Diagnostic source: T9 Nervous at step 2 in 11/14 failed episodes. Impact: Heat success 9.6%→56.9% (_ + _47.3 pp)._

**Analysis.** Outcome-only evolution ( _w/o Diagnostics_ ablation) achieves only +31.4 pp on Heat, leaving 15.9 pp attributable exclusively to the emotion diagnostic. The signal localizes failure to a specific step within the episode and identifies it _before_ the erroneous action rather than after it.

Figure 8: **Case C: Pre-Action Emotion Signal Prevents Failure.** T9 Nervous at step 2 triggers a preconditioncheck reroute before the erroneous heat action. Round-1 evolution encodes this into the SOP, raising Heat task success from 9.6% to 56.9% (+47.3 pp); 15.9 pp of this gain is attributable exclusively to emotion diagnostics over outcome-only evolution.

|Symbol|Definition|
|---|---|
|_Agent and task_||
| _M_|LLM agent backbone (Qwen3-8B or Qwen3-14B, non-thinking mode)|
|_S_ =_{s_1_, . . . , sM}_|Skill library of_M_ skills;_Di_: natural-language SOP for skill_si_ |
|_xt_|External textual context at step_t_(task instruction, history, observation)|
|_s_^_∗_ _t_| Skill selected at step_t_|
|_r_ep _∈{_0_,_1_}_|Episode outcome (1 = success)|
|_LLM internals_ ||
|h^(()^(_ℓ_)) _t_ _∈_R^_d_|Residual-stream activation at layer_ℓ_, token position_t_;_d_: hidden dim|
|_L_^_∗_| Optimal extraction layer (selected by GoEmotions validation accuracy)|
|¯h^(()^(_ℓ_))|Mean-pooled per-story activation at layer_ℓ_(tokens_t≥_50)|
|_Emotion direction ext l_|_raction (offline)_|
|_K_ = 27|Number of GoEmotions emotion concepts|
|_N_ = 100|Stories generated per emotion concept|
|_µ_^(()^(_ℓ_)) _k_|Within-class mean activation for concept_k_at layer_ℓ_|
|¯_µ_^(()^(_ℓ_))|Global mean over all_KN_ story activations at layer_ℓ_|
|v^(()^(_ℓ_)) _k_ _∈_R^_d_|Contrastive emotion direction: _µ_^(()^(_ℓ_)) _k_ _−_¯_µ_^(()^(_ℓ_))(Eq. 1)|
|Π^(_⊥_) |PCA denoising projector (removes top neutral-activation components) |
|ˆv^(()^(_ℓ_)) _k_|Denoised direction: Π^(_⊥_)v^(()^(_ℓ_)) _k_ |
|_We ∈_R^(_K×d_)|Emotion extractor matrix; rows= ˆv^(()^(_L∗_)) _k_|
|_Online emotion state_ |^_∗_ ^_∗_|
|e_t ∈_R^_K_|_ℓ_2-normalized emotion state at step_t_: _We_h^(()^(_L_)) _T_ _/∥We_h^(()^(_L_)) _T_ _∥_2 (Eq. 2)|
|_fθ_|     Emotion encoder: 3-layer MLP (27_→_64_→_128_→_128, ReLU,_<_27K params)|
|r_t ∈_R^128| Encoded representation: _fθ_(e_t_)|
|_Template selection an i_|_d confidence gating_|
|_C_ = 12|Number of emotion-state templates|
|_T_ =_{T_1_, . . . , TC}_| Template bank (task-agnostic; Table11)|
|q_j ∈_R^128|Learnable prototype for template_Tj_|
|_T _^_∗_ _t_ |Selected template: arg max_j_cos(r_t,_q_j_)(Eq. 3)  |
|_ct ∈_[0_,_1]|Confidence score: _σ_(w^(_⊤_) _c_ ^r_t_ ^(+)^_b__c_^() (Eq. 4)) i|
|_τ_|Confidence gate threshold (default 0.3, tuned on WebShop validation)|
|_ϕ__t_ = (_T _^_∗_ _t _^(_, c_)_t_^())|i Composite emotion summary injected into_Pt_|
|_Pt_|Augmented prompt at step_t_(Eq. 5); reduces to text-only when_ct < τ_|
|_Encoder training loss_|_es_|
| _A_^(+)(_i_)| Positive set for anchor_i_: same skill & successful outcome|
|_L_ctr|Supervised-contrastive loss (Eq. 6); temperature_κ_=0_._07|
|_L_cls|Template-classification loss (Eq. 7); temperature_κ_^(_′_)=0_._1|
|_λ_|Loss weight: _L_=_L_ctr+_λL_cls, default_λ_=0_._5|
|_Skill evolution_||
|_E_ = (e1_, . . . ,_e_T_) |Episode emotion trajectory|
|_δt_= 1_−_e^(_⊤_) ^e_t_1|Step-to-step cosine distance (Eq 8)|
| _t_ _−_ ¯_δ, σδ_|. Episode mean and std of_δt_ values|
|_P_| Transition-point set: _{t | δt >_ ^(¯)_δ_+_σδ}_(Eq. 9)|
|_Pi_|Skill-specific transitions: _{t ∈P | s_^_∗_ _t_ ^(=)^_s__i_^(_}_)|
|_w_|Dianostic window half-width (default_w_=1; 3-ste context)|
|d_t_|g p Diagnostic context: (_xt−w_:_t_+_w, s_^_∗_ _t−w_:_t_+_w_^(_, T ∗_) _t−w_:_t_+_w_^(_, δ_)_t−w_:_t_+_w_^() (Eq. 10))|
|∆_i_|   Verbalized transition summary for skill_si_|
|_P_ ^(()^(_i_)) evo|Evolution prompt for skill_si_ (Eq. 11)|
|_D_^(_′_) _i_|Revised SOP generated by the LLM rewriter (Eq. 12)|
|_U_^(()^(_r_))|Underperforming skills in round_r_(success rate_< η_)|
|_η_|Performance threshold for flagging skills (default 04)|
|_R_|l   . Number of evolution rounds (default 3)|

Table 14: Complete notation reference. Symbols are grouped by the pipeline stage in which they first appear. Standard notation ( _M_ , _Di_ , _xt_ ) is also defined inline where first used in the main text.

Figure 9: Confidence score distribution (WebShop, Qwen3-8B). Mass concentrated in _ct ∈_ [0 _._ 30 _,_ 0 _._ 60] with a sparse high tail; 72% of steps are gate-admitted (blue), 28% suppressed (orange). Gate threshold _τ_ =0 _._ 3 shown as dashed line.
