# Representational Control over Self-Report & Behavior Coherence in LLM Risk-Taking

- **Authors:** Rafal Kocielnik, Peiyang Song, Pengrui Han et al.
- **Date:** 2026-10-02
- **Source:** https://arxiv.org/abs/2610.04125
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `kocielnik_representational-control-over-self_2610.04125`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Preprint

**Rafal Kocielnik**^**1**^_∗_ **Peiyang Song**^**2,1** **Pengrui Han**^**3** **Myrl G. Marmarelis**^**1** **Ramit Debnath**^4 **Dean Mobbs**^1 **R. Michael Alvarez**^1

1Caltech 2Carnegie Mellon University 3UIUC 4University of Cambridge

## Abstract

Self-report is an appealing low-cost probe of an LLM’s dispositions, but recent work finds only selective agreement between what models report and how they behave. Prior accounts establish these patterns by prompting black-box LLMs, leaving open whether the gap is a prompting artefact or a fact about how the underlying constructs are represented internally. We investigate risk-taking, a consequential dimension of agentic decision-making, using activation steering to measure self-report and behavior under the same internal intervention. We survey nine steering-vector extraction methods spanning task-specific directives, the model’s own task behavior, and dispositional descriptions at two granularities, evaluated on two behavioral tasks and two psychometric instruments across four open-weight LLMs. We find that _(1) a shared internal intervention does not ensure shared responsiveness_ : directions built from trait descriptions move self-report but leave behavior at chance, directions built from the model’s own task choices do the reverse, and only task-specific directives reach both, weakly. _(2) Diagnosis dissolves that exception_ : removing surface confounders leaves the directives only 32% of their behavioral effect. The two channels are otherwise steered by near-orthogonal directions, each channel reachable by several independent constructions. _(3) An intervention composing one behavior-moving and one self-report-moving direction moves both together_ ; flipping one sign sets them in opposition, with reported and enacted risk pointing in opposite directions on 69–89% of flipped compositions in all models. These findings move the self-report _↔_ behavior relationship from a black-box observation to a representational one that can be inspected and controlled, motivating representational checks alongside behavioral evaluation.

## 1 Introduction

Self-report is used as a cheap proxy for LLM disposition: rather than running a full behavioral evaluation, a practitioner asks the model how it would act (e.g., via a psychometric questionnaire) and treats the answer as a prediction. Dispositions are also used in the other direction, as a control interface: a single reusable specification (e.g., “be risk-averse”) is cheaper to write and audit than a bespoke policy for every task. Both uses rest on the same assumption: that one disposition underlies what the model says and what it does. Measuring one channel then tells you about the other, and setting one carries into the other. Recent work shows the assumption fails on the measurement side (Song et al., 2026b): LLM self-descriptions dissociate from what the model does, both for coarse Big-Five profiles (Han et al., 2025) and for fine-grained, task-specific behavioral intentions, which agree with behavior only within a session and only for behaviors under the model’s volitional control (Kocielnik et al., 2026a). The gap extends beyond personality to stated values (Shen et al., 2025).

These works characterize the dissociation entirely from the outside: they compare what a model says under a questionnaire prompt to what it does under a task prompt, without looking inside the model itself. This leaves the mechanism underdetermined. Three accounts fit the same evidence. Self-report and behavior may be computed from _different_ internal representations, so moving one leaves the other untouched. They may share a representation that is _read out differently_ by separate generation pathways, surfacing as a rating in one context and an action in another. Or the gap may

> _∗_ Corresponding author: rafalko@caltech.edu

Figure 1: **Diagnosing and controlling self-report–behavior dissociation.** We compare **nine steering methods across four families** : task directives, model decisions, risk-specific traits, and Big-Five profiles. Across four LLMs, two behavioral tasks, and two self-report instruments, we test **(RQ1)** whether steering moves both channels together; **(RQ2)** whether dissociation reflects shortcuts, geometric separation, or downstream dependence, using lexical controls, projection, and ablation; and **(RQ3)** whether composing directions can restore coherence or induce anti-coherent states.

be a confound of _elicitation_ rather than a fact about representation at all (Meyer et al., 2026), since a questionnaire item and a task prompt need never have placed the model in a comparable internal state to begin with. Comparing inputs and outputs alone cannot distinguish these.

We ask instead whether the dissociation persists when the _same_ causal intervention is applied internally to target the same latent construct. Activation steering adds a direction to the residual stream rather than to the prompt, so an identical manipulation, at an identical dose, is present whether the model is answering a questionnaire or performing a task (Turner et al., 2023; Rimsky et al., 2024). This holds the intervention fixed across elicitations and turns a behavioral observation into a question about representation: does one internal direction move both outputs, or does moving one leave the other untouched?

We study risk-taking (Fig. 1) using two behavioral tasks and two human-validated self-report instruments. Prior prompting studies suggest that the specificity of a disposition’s description affects whether it reaches behavior (Han et al., 2025; Kocielnik et al., 2026a). We bring this comparison inside the model by varying _the source and specificity of supervision used to construct steering directions_ : (a) task-specific instructions to take or avoid risk; (b) the model’s own observed task decisions, without anchor text; and (c) trait-theoretic descriptions at two granularities, from risk-specific clauses to broad Big-Five profiles that never mention risk (Table 1). Evaluating each direction on both channels tests whether its source determines which output it moves, providing a starting point for diagnosing dissociation and testing joint control. This yields three research questions:

- **RQ1 (Representational Coherence):** Does internal steering reproduce the patterns of self-report _⇔_ behavior coherence and dissociation observed in prior work?

- **RQ2 (Diagnosing Dissociation):** How distinct are the representations and computational routes underlying self-report and behavior?

- **RQ3 (Joint Control):** Can composing behavior- and self-report-moving directions drive the two channels into coherent and anti-coherent states, and are these states equally easy to induce?

We find that the dissociation is reproduced representationally, with the channel a direction reaches determined largely by its source of supervision (RQ1; Fig. 1, left). Fine-grained task-specific directives appear to be the exception, moving both channels, but their behavioral effect is mostly a lexical shortcut, retaining only 32% [95% CI: 18, 47] once orthogonalized against surface controls, so no method spans the two channels through a shared disposition. The directions moving each channel are otherwise near-orthogonal, retaining 92–102% of their effect when the other channel’s families are removed, while within a channel multiple, causally entangled directions move the output (RQ2; center). Given their near-orthogonality, composing behavior- and self-report-moving directions can impose or reverse coherence: all four models reach anti-coherent states on 69–89% of anti-aligned compositions, and what resistance exists lowers this rate without preventing states in which what a model reports contradicts what it does (RQ3; right). These results locate the self-report _↔_ behavior gap for risk-taking in the geometry of internal representations, and show that a dissociation diagnosed representationally can be addressed representationally.

Table 1: The nine steering-vector extraction methods, grouped by _what supervises the extraction_ . Excerpts show the risk-seeking pole of the CCT-built anchors; BART-built versions name that task instead. Families _(a) Task instructions_ and _(b) Own choices_ reference a specific task by name, while _(c) Risk dispositions_ and _(d) Big-Five profiles_ do not. The last column is the literature source. Full anchor sets are in App. B.3 and steering-method lineage in App. B.

|Families/Methods|Risk-seeking anchor (excerpt, authored)|Literature source|
|---|---|---|
|_(a) Task instructions — a_|_n instruction to take or avoid risk in a given task_||
|TASK-COMMAND|“Take maximum risks. Flip as many cards as possible. . . ”|(Higgins,1997)|
|TASK-INTENT|“I intend to follow a gain-seeking strategy . . . in the Columbia Card Task. . . ”|(Ajzen,1991)|
|_(b) Own choices — the m_|_odel’s own decisions observed in task;__contrast from behavi_|_or, not text_|
|CHOICE-CONTRAST|_no authored anchor_, task boards split by the model’s own expected card countE[_k_]|(Rimsky et al.,2024)|
|CHOICE-PROBE|_no authored anchor_, a second instantiation: its own board grid, same ranking, logistic probe|(Li et al.,2023)|
|_(c) Risk dispositions — d_|_ispositional descriptions of a risk-taking person;__task is neve_|_r mentioned_|
|RISK-DOMAINS|“...invests a big share of income in a speculative, high- volatility stock”|(Blais & Weber,2006)|
|THRILL-SEEKING|“...seeks thrills and adventure”|(Zuckerman,1994)|
|RISK-PERSONA|_model generated_ — “risk-seeking and bold. . . ”, then states averaged over the model’s own responses|(Chen et al.,2025)|
|_(d) Big-Five profiles — d_|_i  ispositional personality trait descriptors;__risk is never menti_|_oned_|
|OCEAN-PROFILE|“...open-minded, imaginative, curious. . . outgoing, ener- getic, sociable. . . ” (+O+E_−_C_−_A_−_N)|(Nicholson et al.,2005)|
|OCEAN-COMPOSED|the same profile_composed_from five single-trait vectors|(Han et al.,2026)|
|NULLCONTROL|_Random Direction_— Gaussian per layer, norm-matched|(Turner et al.,2023)|

More broadly, our work brings **_mechanistic evidence to the question of whether LLM self-reports can support behavioral auditing_** . The usefulness of self-report as an audit depends on its connection to the processes governing behavior, and we show that internal interventions can change either channel selectively, or move the two together or in opposition. This motivates **_mechanistically informed behavioral auditing_** : testing whether self-reports remain informative as behavior changes, and identifying the representational and computational dependencies that control that relationship.

- 2 RQ1 (REPRESENTATIONAL COHERENCE): DOES REPRESENTATIONAL STEERING REPLICATE DISSOCIATION PATTERNS?

A questionnaire and a risk-taking task may elicit different responses because they condition the model differently. Administering both in one session links their contexts but invites priming: behavior may follow the model’s earlier answer rather than a disposition (Kocielnik et al., 2026a). Prior work matched generation settings or conditioned both elicitations on the same persona, yet still found disagreement (Han et al., 2025; Kocielnik et al., 2026a). This leaves open whether dissociation persists under more directly matched internal conditioning. **_Steering applies a common intervention across the two contexts_** : we add the same direction to the residual stream at matched standardized doses (Turner et al., 2023; Rimsky et al., 2024), while keeping the elicitations separate to avoid cross-channel priming. The remaining question is **_how to construct directions that target risk-taking as a construct_** ; we compare supervision from human theories and from the model’s own decisions to test whether self-report _↔_ behavior dissociation persists under internal steering.

Figure 2: **Extraction source shapes which channel a steering vector reaches.** Effects are expressed in random-direction control SDs; grey bands mark the control band ( _±_ 3 _._ 5; App. C.4). **(A)** Task-matched effects by family: behavior left, self-report right. Bars show family medians with 95% intervals; points show method–model cells, filled when they clear the control. **(B)** Effects by method and model; asterisks mark cells that clear the control. **(C)** Mean family effects across doses, standardized against dose-matched controls; ribbons show model-based 95% intervals (App. C.5.4).

### 2.1 Experimental setup

**Risk-taking supervision across four families.** We compare nine extraction methods spanning four supervision families (Table 1). Three draw on human accounts of risk-taking at different granularities: _(d) Big-Five profiles_ uses broad personality profiles (Nicholson et al., 2005), _(c) Risk dispositions_ uses risk-specific dispositions (Weber et al., 2002; Zuckerman, 1994), and _(a) Task instructions_ uses task-specific instructions (Ajzen, 1991). Prior LLM work finds different patterns of agreement with behavior for task-specific and broad trait self-reports (Kocielnik et al., 2026a); we test whether this distinction extends to internal steering. Internal access enables _(b) Own choices_ , which derives supervision from the model’s own task decisions rather than authored descriptions. Each family includes at least two methods, reducing dependence on a single construction choice.

**Steering-vector extraction.** To focus on _supervision source_ , we hold extraction fixed where possible: the difference in mean residual activation between contrasting conditions, read at the final prompt token and unit-normalised per layer. This is the de-facto standard for steering vectors (Rimsky et al., 2024; Zou et al., 2023) and assumes only that the construct has a linear representation (Park et al., 2024). Three methods depart from it, each following the recipe it replicates (App. B). We compare every effect against NULL CONTROL, eight seeded Gaussian directions per model injected at matched layers and doses (Mitrani Hadida et al., 2026).

**Models, tasks and outcomes.** We evaluate four open-weight chat models: Qwen2.5-7B, Llama3.1-8B, OLMo-3-7B, and Ministral-3-8B. Behavior is measured using the **Columbia Card Task (CCT)** (Figner et al., 2009) and the **Balloon Analogue Risk Task (BART)** (Lejuez et al., 2002), testing whether findings hold across behavioral settings. Both involve choosing how far to pursue reward under risk of loss, through card flips or balloon pumps. Self-report is measured using two deliberately different instruments: the **BFI-44** (John et al., 1991), _task-agnostic personality content_ scored using a risk-taking profile based on Nicholson et al. (2005), and **DOSPERT** (Blais & Weber, 2006), with _risk-domain-specific content_ asking directly about willingness to engage in risky activities. We read all outcomes from answer-token likelihoods to capture shifts in response probabilities without sampling variability (Santurkar et al., 2023) (App. B.4).

**Injection and dose robustness.** We inject vectors into a mid-network band of the residual stream, using the same layers when evaluating behavior and self-report. Prior layer sweeps identify this region as supporting reliable steering effects (Rimsky et al., 2024; Zou et al., 2023). To express doses relative to activation scale, we scale each unit-normalized direction by _σ_ , the SD of activations projected onto it, adapting the scaling of Li et al. (2023). Each vector is injected at + _α_ and _−α_ ; its _swing_ is the difference between the resulting readouts. Main experiments use _α_ = 1, selected in pilot runs to produce substantial effects without over-steering. Additional experiments vary _α_

from 0 _._ 5 to 8 across all four families, with random-direction controls matched at each dose, to test whether channel selectivity depends on dose (App. C.5.4).

**Effect estimation and inference.** We express swings in _control SDs_ , dividing by the SD of random-direction swings for the same model and outcome, and repeat analyses in raw log-odds (App. C.6). Motivated by sensitivity to input context and steering settings (Tan et al., 2024; Han et al., 2024b), we use a 20-condition grid varying persona, injection band, and vector construction (App. C). We fit a regression per outcome, comparing methods against random directions while adjusting for persona, injection band, and model, with standard errors clustered by extracted vector. Motivated by control-based evaluation (Hewitt & Liang, 2019; Li et al., 2023), we assess effects against controls rather than zero. We define an effect as clearing the control when its mean swing lies more than 3 _._ 5 control SDs from the control directions’ mean, in the intended direction (App. C.4).

### 2.2 Results

**Dispositional families move self-reports but not behavior (Fig. 2A,B).** _(c) Risk dispositions_ and _(d) Big-Five profiles_ clear the control in none of the 40 behavioral (method _×_ model _×_ task) cells, with median swings of 0 _._ 7 [0 _._ 3 _,_ 1 _._ 1] and 0 _._ 4 [0 _._ 3 _,_ 0 _._ 8] control SDs, respectively. On self-report, they clear it in 31 of 40 cells, with medians of 6 _._ 6 [5 _._ 6 _,_ 10 _._ 0] and 8 _._ 9 [6 _._ 2 _,_ 11 _._ 4]. The pattern holds for both risk-specific descriptions and broad personality profiles that never mention risk. **_Steering a model to describe itself as a risk-taker does not reliably make it take risks._**

**Own-choice family moves behavior but weakly affects self-report (Fig. 2A,B).** _(b) Own choices_ clears the control in all 16 behavioral cells, with a median swing of 42 _._ 1 [27 _._ 7 _,_ 59 _._ 6] control SDs, the largest behavioral effect and roughly 7 _×_ that of _(a) Task instructions_ . Self-report effects are weak and inconsistent, with a median of 1 _._ 8 [0 _._ 9 _,_ 2 _._ 5] within the control band (App. C.5.1). Behavioral effects are also task-specific, retaining only 7% [3 _,_ 13] on the other task (App. C.5.3).

**Task instructions reach both channels, with stronger effects on risk-specific self-report (Fig. 2A).** _(a) Task instructions_ clears the control in 11 of 16 behavioral cells, with a median swing of 6 _._ 2 [3 _._ 4 _,_ 9 _._ 7] control SDs, and in 24 of 32 self-report cells. Self-report effects are larger on DOSPERT (33 _._ 4 [12 _._ 1 _,_ 56 _._ 9]) than BFI (6 _._ 9 [3 _._ 3 _,_ 9 _._ 0]), consistent with DOSPERT’s direct focus on risk-taking. This contrast is absent in the dispositional families (App. C.5.2). RQ2 tests whether this apparent reach across channels reflects risk-related representations or task shortcuts.

**Higher doses do not enable additional families to move both channels (Fig. 2C).** Families differ in their behavior–self-report balance in effect size and rates of clearing the control (both _p <_ 0 _._ 001). The between-channel gap in these rates shows no significant dose dependence for any family (all _p >_ 0 _._ 2). Dispositional behavioral effects stay within the control band throughout (means: 0 _._ 0– 1 _._ 3 control SDs). Higher doses strengthen both channels for task instructions and behavior for own choices, without making additional families reliable movers of both channels (App. C.5.4).

**Summary.** The source and granularity of supervision shape which channel a steering direction moves: dispositional directions affect self-reports, own-choice directions affect behaviors, and task instructions reach both. **_A change in what a steered model says about its risk-taking is not a reliable sign of a change in what it does._** The pattern holds across doses and in raw log-odds, with a similar ordering of methods under prompting (Apps. C.5.6, C.5.4, and C.5.5).

## 3 Rq2 (diagnosing dissociation): how are self-report and behavior encoded?

RQ1 revealed distinct patterns of dissociation, with task instructions reaching both channels. We now ask **_what drives these effects and why channels dissociate_** . Internal steering lets us investigate both what an injected direction carries and which computations its effects depend on.

A steering effect need not reflect a general risk-taking disposition: steering can exploit spurious features (Tan et al., 2024). For behavior, task-specific directives may combine risk instructions with **_task-specific cues_** , such as choosing larger quantities. We therefore test how much their effects depend on these cues, which may express part of the intended task policy. For self-report, extraction texts and survey items may share **_representational features even without shared wording_** , potentially creating a representational analogue of common-method variance (Podsakoff et al., 2003).

Human risk-taking also varies across domains and is only partly explained by a general factor (Frey et al., 2017). Models may likewise encode multiple risk-related components, raising the question of how directions moving the same or different channels relate. They may overlap where they are injected, or remain geometrically separate yet depend on shared downstream processing (Elazar et al., 2021). We distinguish **_overlap between steering directions_** from **_dependence during computation_** , which may reflect shared risk-related content or processes needed to produce an answer.

Figure 3: **Behavior- and self-report-moving directions are geometrically distinct; directives’ behavioral effects largely rely on quantity cues. (A)** Mean _|_ cos _|_ across layers and models; blocks denote families. **(B, C)** Behavioral and self-report retention after removing shortcuts from vectors (plain), other families from vectors (///), or from hidden states (xxx). Family-removal bars average individual removals within each channel; joint removals appear in Fig. 15. Colors denote families; bars show pooled retention (DerSimonian–Laird, 95% CIs). Grey bands mark retention below the random-direction control threshold; dashed line: full retention.

### 3.1 Experimental setup

**Geometry and removal interventions.** We compute pairwise _|_ cos _|_ within each layer and average over layers and models (App. D.2). To test dependence on geometric overlap, we compare two interventions that subtract components along a direction set _S_ . For a given removed set _S_ , both interventions use the same layer-specific orthonormal basis _{_ **q** ˆ _i}_ and apply **x** _←_ **x** _−_^() _i_^(()^**x**^(_·_ˆ)^**q**^(_i_)ˆ)^**q**^(_i_.) _Shared-direction removal_ edits the steering vector before injection, testing whether its effect depends on components shared with _S_ (Ravfogel et al., 2020). _Shared-route removal_ edits hidden states at every layer and token position, testing dependence during computation, including components formed downstream of injection (Arditi et al., 2024). The set _S_ contains shortcut directions or method families, removed individually or jointly (App. D.9).

**Shortcut controls.** Contrastive extraction may capture additional features that vary alongside the intended construct (Joshi et al., 2026). Inspired by control-task approaches (Hewitt & Liang, 2019), we construct twelve shortcut directions covering task wording, quantities, response format, and endorsement of items from the evaluation questionnaires (Table 14). We geometrically remove these components from steering vectors and measure how much of the effect remains. Removing them from model hidden states is harder to interpret, because they may support the production of numeric answers. We examine this separately in a self-report robustness analysis (App. D.6).

**Analytical approach.** We compare _|_ cos _|_ against its mean between random unit vectors (Cai et al., 2013). Inspired by embedding association tests (Caliskan et al., 2017), we permute channel labels to test whether directions moving the same channel are more aligned than those moving different channels (App. D.4). Regression assesses how much variation in _|_ cos _|_ is explained by family pair, model, and layer depth (App. D.3). Using RQ1’s swing measure, **_retention_** is the fraction of the original effect remaining after removal; **_survival_** indicates whether it still clears the control band with its original sign. We bootstrap retention over perturbation conditions and pool (direction, model) cells that clear the control before removal using DerSimonian–Laird (App. D.12). Both removal interventions are compared against random subspaces of matching rank to distinguish targeted losses from nonspecific effects of subspace removal (Elazar et al., 2021). Further controls in App. D.11.

### 3.2 Results

**Steering directions are geometrically separate across channels (Fig. 3A).** Directions are more aligned within than across channels (∆ mean _|_ cos _|_ = 0 _._ 41 [0 _._ 35 _,_ 0 _._ 53]; permutation effect size _d ≥_ 1 _._ 65 in every model, App. D.4). Family pair explains 60% of cosine variation, versus 4.5% for layer depth and 0.6% for model (App. D.3). Removing the other channel’s families preserves 97% [90 _,_ 104] and 92% [85 _,_ 99] of the behavioral effects of _(a) Task instructions_ and _(b) Own choices_ , respectively, and 102% [96 _,_ 108] of the self-report effect of _(a) Task instructions_ (App. D.9).

**Behavior-moving families are geometrically separate but differ in quantity-cue dependence (Fig. 3B).** Removing the twelve shortcut directions retains 32% [18 _,_ 47] of the behavioral effect of _(a) Task instructions_ versus 74% [69 _,_ 80] for _(b) Own choices_ . Most of the reduction in _(a) Task instructions_ comes from removing quantity cues, including those in TASK-COMMAND’s extraction text (App. D.5). Its purified behavioral effect never clears the control across doses of _α_ = 0 _._ 5–8 (App. D.7). Mutual family-span removal preserves 92% of behavioral effects ([77 _,_ 107] for _(a) Task instructions_ ; [90 _,_ 95] for _(b) Own choices_ ), supporting distinct steering components.

**Self-report families share steering components (Fig. 3C).** All three self-report-moving families retain 64–66% of their effects after shortcut removal. _(c) Risk dispositions_ and _(d) Big-Five profiles_ overlap geometrically, yet retain 72% [66 _,_ 78] and 76% [69 _,_ 84] of their effects, respectively, averaged over individual removals of other self-report families. Joint removal leaves only 7% [ _−_ 11 _,_ 25] for _(c) Risk dispositions_ and 33% [24 _,_ 42] for _(d) Big-Five profiles_ (Fig. 15). Thus, effects that survive individual family removals can still depend on the other families’ combined span.

**Route removal reveals downstream dependence (Fig. 3B,C).** Route removal causes larger losses than direction removal: behavioral retention falls to 61% [34 _,_ 87] for _(a) Task instructions_ and 40% [24 _,_ 56] for _(b) Own choices_ ; self-report retention falls to 38% [35 _,_ 41] for _(c) Risk dispositions_ and 32% [30 _,_ 34] for _(d) Big-Five profiles_ . To test whether shared response processes contribute to these losses, we additionally ablate the twelve shortcut directions from model activations during self-report evaluation. This largely abolishes steering (median retention _−_ 0 _._ 11), whereas rank-matched random removal leaves it intact (0 _._ 99). However, shortcut ablation also shifts unsteered readouts, so the collapse may reflect disrupted response production rather than shared risk content (App. D.6).

**Summary.** **_Steering directions are geometrically separate across channels, with multiple separable components capable of moving each channel._** The behavioral effect of _(a) Task instructions_ largely depends on quantity cues. Within self-report, joint removal of the other families largely eliminates the effect of _(c) Risk dispositions_ , leaves a smaller surviving effect for _(d) Big-Five profiles_ , and preserves that of _(a) Task instructions_ . Route ablation reveals downstream dependence, with the additional self-report analysis suggesting a contribution from shared response processes.

- 4 RQ3 (JOINT CONTROL): CAN THE TWO CHANNELS BE DRIVEN INTO EXPLICITLY COHERENT AND ANTI-COHERENT STATES?

RQ2 identified largely geometrically separable steering directions for behavior and self-report, despite downstream dependence. We now ask whether composing these directions can restore coherence, and whether reversing one component can induce anti-coherence. Coherent movement could support behavioral auditing through self-report (Kocielnik et al., 2026a); conversely, self-reports that fail to reflect behavior can undermine monitoring based on models’ own accounts (Li et al., 2026). We therefore test **_whether coherent and anti-coherent states are reachable by composition_** , **_whether anti-coherent states are harder to reach_** , and whether any weak self-report spillover from the behavior direction predicts this difficulty.

4.1 EXPERIMENTAL SETUP

**Vector composition.** We compose a behavior-moving direction _B_ and a self-report-moving direction _S_ (Scalena et al., 2024):

Endpoints are unit-normalized per layer before mixing; the blend is renormalized and scaled using RQ1’s _σ_ -dose protocol (§2.1). The weight _β_ controls the self-report contribution; _s_ = +1 gives an _aligned_ and _s_ = _−_ 1 an _anti-aligned_ composition. At each _β_ , compositions are paired by _s_ .

Figure 4: **Composition can induce coherence or anti-coherence, with model-dependent resistance to the latter. (A)** Target-state attainment under matched aligned and anti-aligned compositions. **(B)** Resistance to anti-coherence increases with the behavior vector’s self-report spillover toward coherence. Curve: logistic fit. **(C)** Joint behavioral and self-report responses by model. Matching signs indicate coherence; opposing signs indicate overclaiming or underclaiming risk.

**Endpoints and evaluation.** We use _B ∈{_ CHOICE-CONTRAST _,_ CHOICE-PROBE _}_ and _S ∈ {_ THRILL-SEEKING _,_ OCEAN-COMPOSED _}_ , selected for strong effects on their target channel and limited effects on the other. The behavior endpoints differ in their self-report spillover, enabling us to test whether this predicts resistance to anti-coherence. An additional condition projects each endpoint out of the other family’s span before composition. We inject compositions at _±α_ , _α_ = 1, across four models, both behavioral tasks, and both self-report instruments (App. E.1).

**Analytical approach.** _Coherence_ denotes matching behavioral and self-report swing signs; _anticoherence_ denotes opposite signs. _Target-state attainment_ requires both channels to move in the directions indicated by their pure endpoints, accounting for the sign applied to each component. Anti-coherence comprises _overclaiming_ risk (self-report increases, behavior decreases) or _underclaiming_ (the reverse). The _self-report spillover_ is the change in self-report produced by the behavior direction _B_ alone. We test whether this spillover predicts difficulty inducing anti-coherence. We estimate attainment using logistic regression with cluster-robust standard errors (Liang & Zeger, 1986) and compare constructions using exact McNemar tests with Holm correction. Model specifications, cell gating and robustness checks appear in Apps. E.1–E.2 and E.5.

### 4.2 Results

**Composition can induce both coherence and anti-coherence (Fig. 4C).** Aligned compositions induce coherence in 91% of cases. However, reversing the self-report component increases anticoherence from 8 _._ 5% to 77 _._ 4%, a difference of 69 p.p. [66 _,_ 72] ( _p <_ 0 _._ 001), significant in every model (App. E.3). Overclaiming and underclaiming rates differ by less than 5 p.p. in each model, with a significant difference only in Ministral (App. E.4).

**Resistance to anti-coherence is model-dependent and limited (Fig. 4A).** Anti-aligned compositions reach their target state 14 p.p. less often than aligned compositions. The gap is absent in Qwen (89% attainment for both) and not significant in OLMo (6 p.p.), but reaches 10 p.p. in Llama and 29 p.p. in Ministral ( _p <_ 0 _._ 001, Holm-corrected). Nevertheless, anti-aligned compositions induce anti-coherence in 69–89% of cases across all four models. Increasing the self-report weight from _β_ = 0 _._ 3 to 0 _._ 7 raises anti-coherence from 66% to 87% but weakens behavioral steering (App. E.6).

**Greater self-report spillover predicts greater resistance (Fig. 4B).** Anti-aligned compositions are less likely to induce anti-coherence when the behavior direction also moves self-report toward coherence (spillover). Each additional control SD of such spillover is associated with a 3 _._ 8 p.p. lower anti-coherence rate ( _B_ = _−_ 0 _._ 28 [ _−_ 0 _._ 35 _, −_ 0 _._ 21], _p <_ 0 _._ 001; Table 17). An experiment removing each endpoint’s geometric overlap with the other endpoint’s family before composition leaves the rates of coherence and anti-coherence largely unchanged (App. E.5).

**Summary.** **_Composition can align self-report with behavior, but can also drive the two channels in opposite directions._** Anti-coherence remains inducible in all four models despite model-dependent resistance, which tracks the behavior direction’s self-report spillover toward coherence. Increasing the self-report weight raises anti-coherence rates while weakening behavioral steering.

## 5 Discussion

Our study brings the LLM self-report–behavior gap into representational analysis. Dissociation persists under internal steering, with supervision source shaping which channel a direction moves (RQ1). Behavior- and self-report-moving directions are largely geometrically separate, while taskinstruction-derived directions’ dual reach depends heavily on shortcut cues (RQ2). Composing directions can induce both coherent and anti-coherent outputs (RQ3). These findings inform how we interpret self-report, audit behavior, and target representational interventions.

**Why self-report and behavior come apart, and when they seem not to.** Prior work documents gaps between personality self-reports and behavior (Han et al., 2025), and between stated values and actions (Shen et al., 2025), alongside selective, context-dependent coherence (Kocielnik et al., 2026a). Our results help explain their coexistence: only task-instruction-derived directions move both channels, but removing task cues reduces their behavioral effect much more than their self-report effect (Secs. 2, 3). Directions built from dispositional descriptions move self-report but almost never behavior. In humans, risk-taking varies across domains (Weber et al., 2002) while retaining a general component (Frey et al., 2017), and personality traits are associated with domain-specific self-reported risk-taking (Nicholson et al., 2005). Our results probe this connection representationally: self-report and behavior are moved by near-orthogonal directions, and with several directions within each channel, coherence in one pairing may not generalize to another. **_Output agreement alone is weak evidence of a shared disposition._** LLMs can encode conceptual hierarchies (Park et al., 2025); we see hints within self-report, where task-level directions move domain-level DOSPERT more than trait-level BFI. This pattern does not clearly extend to behavior, potentially reflecting limits on how representations encode hierarchies beyond surface level text (Yang et al., 2026).

**Separable channels are both an instrument and a liability.** Composing directions drives models into coherent states on 91% of pushes and into anti-coherent ones on 69–89%. Resistance to anticoherence is model-dependent, absent in Qwen and largest in Ministral, and part of it is explained by how much the behavior direction on its own moves self-report toward coherence (Fig. 4B). The channels are therefore coupled, but very weakly: too weakly to prevent anti-coherence, or to let any direction reliably move both channels beyond its wording. The high sensitivity makes steering a precise instrument for studying coherence. It also means **_self-report is at best a weak tamper-prone signal of behavior_** . A model can be steered to report caution while acting riskily, threatening the validity of audits that use stated values to anticipate behavior (Mitrani Hadida et al., 2026). The broader evaluation principle is to validate assessments against the behavior, as in human studies (Kocielnik et al., 2024; 2026b). Protection would require stronger coupling between the channels than these models show, not monitoring of one of them. When weights are available, two cheap checks can complement agreement statistics in evaluation: whether the directions that move each channel overlap, and whether an effect survives projection against surface controls (Sec. 3).

**Toward bottom-up, representationally grounded alignment.** Han et al. (2025) suggested that deeper alignment may require intervening at the representational level, and prompt-supplied specifications are limited in how far they override what a model has internalized (Casanova et al., 2026); our results sharpen where such interventions should aim. Trait descriptions, the intuitive top-down route, reach self-report but almost never behavior. Task instructions never name a trait, yet their content beyond wording moves dispositional questionnaires, suggesting that **_task-level supervision can shape how a model describes itself, not only what it does_** . Humans show a similar pattern, as behavior shifts attitudes (Bem, 1972) and concrete, trait-consistent tasks change self-reported traits (Stieger et al., 2021). Fine-tuning on risky choices can lead models to describe themselves as risk-seeking (Betley et al., 2025), suggesting that training can strengthen a coupling that is weak in the base representation. Objectives that reward self-consistency (Pres et al., 2026) should therefore be judged by whether they build that coupling into the representation, not only into the outputs. This is the core of _mechanistically informed behavioral auditing_ : testing not only whether self-report and behavior agree, but whether anything inside the model makes them agree on a causal level.

**Limitations.** We study a single construct, risk-taking, but one that matters directly as LLMs act in decision-making roles (Bhatia, 2024), and focusing on it lets us examine it in depth. We use mid-size open-weight models because steering requires access to activations, which frontier models do not provide. Behavior is read from single decisions, but does not cover multi-step agentic trajectories. Surface controls are necessarily incomplete, but any shortcut they miss would inflate the retained effect we report, so the dissociation is, if anything, understated.

## 6 Conclusion

Whether an LLM’s self-reports reflect the processes that govern its behavior is central to their use in safe deployment. Inspecting model internals, we find that for risk-taking the two channels are moved by largely separate directions, determined by their source of supervision. Only narrow task-specific instructions reach both, mostly through wording cues, and composing directions induces coherent and anti-coherent states with little resistance. The gap is therefore representational, and output agreement is weak evidence of a shared internal mechanism. This motivates mechanistically informed behavioral auditing and intervention: testing whether such agreement has a representational basis and shaping it directly, which is crucial for self-report to serve as an audit of behavior.

## Acknowledgments

This work is supported by the Caltech Linde Center for Science, Society, and Public Policy (LCSSP). R. Michael Alvarez is Flintridge Foundation Professor of Political and Computational Social Science at Caltech.

### AI use statement

LLM-based assistants were used to edit and polish prose and to assist with analysis and plotting code; every such contribution was reviewed and verified by the authors, and no text, statistic, or claim entered the paper without author verification. The research questions, experimental design, analyses, interpretation, and all claims are the authors’ own, and the authors take full responsibility for the content.

## Ethics statement

The main ethical consideration is dual use. RQ3 shows that an anti-aligned composition can drive a model into a state in which what it reports contradicts what it does, including the underclaiming direction that would make a self-report-based evaluation actively misleading. We report this because the reachability of such states is a property of the representation that evaluators need to know, and we frame it as a risk rather than a technique (§5): the construction requires white-box access to activations, and we report population rates under controlled intervention rather than a recipe optimised for concealment. The converse finding, that an aligned blend restores coherence, is the intended use, and our released materials target diagnosis and coherence restoration rather than concealment.

## Reproducibility statement

All experiments use publicly available open-weight models, named with their exact released variants in App. B, and every readout is a deterministic single forward pass over answer-token likelihoods, so results do not depend on sampling temperature or seed (App. B.4). The shared extraction and injection procedure is given as Algorithm 1, with per-method extraction signals and read sites in Table 2, verbatim anchor texts and construction variants in Tables 6–9, resolved per-model injection layers in Table 4, and persona contexts in Table 3. The behavioral tasks and scoring rules are specified in Apps. A.2 and B.4, both instruments as administered in Tables 18 and 19, the twelve shortcut controls in Table 14, and the prompts, templates, and generated responses behind RISKPERSONA in App. F. Control families, the significance criterion, regression specifications, pooling estimators, and robustness checks for each research question are described in Apps. C.4, C.5.4, D.10–D.12, and E.1–E.2. Code for extraction, injection, evaluation, and analysis, together with the extracted steering vectors and per-condition readouts, will be made publicly available.

## References

- Icek Ajzen. The theory of planned behavior. _Organizational Behavior and Human Decision Processes_ , 50(2):179–211, 1991. doi: 10.1016/0749-5978(91)90020-T. URL https://doi. org/10.1016/0749-5978(91)90020-T.

- Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. _arXiv preprint arXiv:1610.01644_ , 2016.

- Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. _arXiv preprint arXiv:2406.11717_ , 2024.

- Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. Leace: Perfect linear concept erasure in closed form. _Advances in Neural Information Processing Systems_ , 36:66044–66063, 2023. doi: 10.52202/075280-2884.

- Daryl J. Bem. Self-perception theory. In Leonard Berkowitz (ed.), _Advances in Experimental Social Psychology_ , volume 6, pp. 1–62. Academic Press, 1972. doi: 10.1016/S0065-2601(08)60024-6.

- Jan Betley, Xuchan Bao, Mart´ın Soto, Anna Sztyber-Betley, James Chua, and Owain Evans. Tell me about yourself: LLMs are aware of their learned behaviors. In _International Conference on Learning Representations_ , 2025.

- Sudeep Bhatia. Exploring variability in risk taking with large language models. _Journal of Experimental Psychology: General_ , 153(7):1838–1860, 2024. doi: 10.1037/xge0001607.

- Ann-Ren´ee Blais and Elke U. Weber. A domain-specific risk-taking (DOSPERT) scale for adult populations. _Judgment and Decision Making_ , 1(1):33–47, 2006. doi: 10.1017/S1930297500000334. URL https://doi.org/10.1017/S1930297500000334.

- Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. _Transformer Circuits Thread_ , 2023. URL https://transformer-circuits. pub/2023/monosemantic-features/index.html.

- Tony Cai, Jianqing Fan, and Tiefeng Jiang. Distributions of angles in random packing on spheres. _Journal of Machine Learning Research_ , 14(57):1837–1864, 2013. URL https://jmlr.org/ papers/v14/cai13a.html.

- Aylin Caliskan, Joanna J. Bryson, and Arvind Narayanan. Semantics derived automatically from language corpora contain human-like biases. _Science_ , 356(6334):183–186, 2017. doi: 10.1126/ science.aal4230.

- Etienne Casanova, Rafal Kocielnik, and R. Michael Alvarez. On the limits of LLM adaptability: Impact of model-internalized priors on annotation task performance. In _Proceedings of the 43rd International Conference on Machine Learning_ , volume 306 of _Proceedings of Machine Learning Research_ , pp. 11976–12000. PMLR, 2026. URL https://proceedings.mlr.press/ v306/casanova26a.html.

- Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors: Monitoring and controlling character traits in language models. _arXiv preprint arXiv:2507.21509_ , 2025.

- Yuhao Dan, Lang Yu, Jiaju Lin, Qin Chen, Jie Zhou, Junfeng Tian, Qingchun Bai, and Liang He. A mechanistic interpretability perspective on personality in large language models. _Information Processing & Management_ , 63(8):104962, 2026. doi: 10.1016/j.ipm.2026.104962.

- Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals. _Transactions of the Association for Computational Linguistics_ , 9:160–175, 2021. doi: 10.1162/tacl ~~a 0~~ 0359.

- Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. _arXiv preprint arXiv:2209.10652_ , 2022. URL https://arxiv.org/abs/2209. 10652.

- Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. Not all language model features are one-dimensionally linear. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL https://proceedings.iclr.cc/paper_files/paper/2025/hash/ d3221cdb27e49d9c1cd35ad254feccfe-Abstract-Conference.html.

- Bernd Figner, Rachael J. Mackinlay, Friedrich Wilkening, and Elke U. Weber. Affective and deliberative processes in risky choice: Age differences in risk taking in the columbia card task. _Journal of Experimental Psychology: Learning, Memory, and Cognition_ , 35(3):709–730, 2009. doi: 10.1037/a0014983. URL https://doi.org/10.1037/a0014983.

- Renato Frey, Andreas Pedroni, Rui Mata, J¨org Rieskamp, and Ralph Hertwig. Risk preference shares the psychometric structure of major psychological traits. _Science Advances_ , 3(10):e1701381, 2017. doi: 10.1126/sciadv.1701381.

- Jacob Goldman-Wetzler and Alex Turner. I found _>_ 800 orthogonal “write code” steering vectors. https://www.alignmentforum.org/posts/CbSEZSpjdpnvBcEvc/ i-found-greater-than-800-orthogonal-write-code-steering, 2024. AI Alignment Forum.

- Pengrui Han, Rafal Dariusz Kocielnik, Adhithya Prakash Saravanan, Roy Luoyao Jiang, Or Sharir, and Anima Anandkumar. ChatGPT based data augmentation for improved parameter-efficient debiasing of LLMs. In _First Conference on Language Modeling_ , 2024a. URL https: //openreview.net/forum?id=q5Ft9ZJtHm.

- Pengrui Han, Peiyang Song, Haofei Yu, and Jiaxuan You. In-context learning may not elicit trustworthy reasoning: A-Not-B errors in pretrained language models, 2024b. URL https: //arxiv.org/abs/2409.15454.

- Pengrui Han, Rafal Kocielnik, Peiyang Song, Ramit Debnath, Dean Mobbs, Anima Anandkumar, and R. Michael Alvarez. The personality illusion: Revealing dissociation between self-reports & behavior in llms. _arXiv preprint arXiv:2509.03730_ , 2025.

- Pengrui Han, Xueqiang Xu, Keyang Xuan, Peiyang Song, Siru Ouyang, Runchu Tian, Yuqing Jiang, Cheng Qian, Pengcheng Jiang, Jiashuo Sun, Junxia Cui, Ming Zhong, Ge Liu, Jiawei Han, and Jiaxuan You. Steer2Adapt: Dynamically composing steering vectors elicits efficient adaptation of LLMs. _arXiv preprint arXiv:2602.07276_ , 2026.

- Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pp. 9318–9333, 2023. doi: 10.18653/v1/2023.findings-emnlp.624.

- John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pp. 2733–2743, 2019. doi: 10.18653/v1/D19-1275.

- E. Tory Higgins. Beyond pleasure and pain. _American Psychologist_ , 52(12):1280–1300, 1997.

- Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, and Yixin Zhu. Evaluating and inducing personality in pre-trained language models. In _Advances in Neural Information Processing Systems_ , volume 36, pp. 10622–10643, 2023. doi: 10.52202/075280-0466.

- Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi, Zifeng Wang, Luxi He, Yichen Wu, Ming Zhong, Peiyang Song, Qizheng Zhang, Heng Wang, Xueqiang Xu, Hanwen Xu, Pengrui Han, Dylan Zhang, Jiashuo Sun, Chaoqi Yang, Kun Qian, Tian Wang, Changran Hu, Manling Li, Quanzheng Li, Hao Peng, Sheng Wang, Jingbo Shang, Chao Zhang, Jiaxuan You, Liyuan Liu, Pan Lu, Yu Zhang,

Heng Ji, Yejin Choi, Dawn Song, Jimeng Sun, and Jiawei Han. Adaptation of agentic AI: A survey of post-training, memory, and skills. _arXiv preprint arXiv:2512.16301_ , 2026. URL https://arxiv.org/abs/2512.16301v3.

- Oliver P. John and Sanjay Srivastava. The Big Five trait taxonomy: History, measurement, and theoretical perspectives. In Lawrence A. Pervin and Oliver P. John (eds.), _Handbook of Personality: Theory and Research_ , pp. 102–138. Guilford Press, New York, 2nd edition, 1999.

- Oliver P. John, Eileen M. Donahue, and Robert L. Kentle. _The Big Five Inventory—Versions 4a and 54_ . University of California, Berkeley, Institute of Personality and Social Research, Berkeley, CA, 1991. URL https://sjdm.org/dmidi/Big_Five_Inventory.html.

- Shruti Joshi, Andrea Dittadi, S´ebastien Lachapelle, and Dhanya Sridhar. Sparse shift autoencoders for identifying concepts from large language model activations. _arXiv preprint arXiv:2502.12179_ , 2026. URL https://arxiv.org/abs/2502.12179v2.

- Rafal Kocielnik, Shrimai Prabhumoye, Vivian Zhang, Roy Jiang, R. Michael Alvarez, and Anima Anandkumar. BiasTestGPT: Using ChatGPT for social bias testing of language models. _arXiv preprint arXiv:2302.07371_ , 2023. doi: 10.48550/arXiv.2302.07371. URL https://arxiv. org/abs/2302.07371.

- Rafal Kocielnik, Cherine H. Yang, Runzhuo Ma, Steven Y. Cen, Elyssa Y. Wong, Timothy N. Chu, J. Everett Knudsen, Peter Wager, John Heard, Umar Ghaffar, Anima Anandkumar, and Andrew J. Hung. Human AI collaboration for unsupervised categorization of live surgical feedback. _npj Digital Medicine_ , 7(1):372, 2024. doi: 10.1038/s41746-024-01383-3. URL https://doi. org/10.1038/s41746-024-01383-3.

- Rafal Kocielnik, Pengrui Han, Peiyang Song, Myrl G. Marmarelis, Ramit Debnath, Dean Mobbs, Anima Anandkumar, and R. Michael Alvarez. Rethinking psychometric evaluation of llms: When and why self-reports predict behavior. _arXiv preprint arXiv:2606.12730_ , 2026a.

- Rafal Kocielnik, J. Everett Knudsen, Steven Y. Cen, Jasmine Lin, Cherine H. Yang, Atharva Deo, Ujjwal Pasupulety, Peter Wager, Anima Anandkumar, and Andrew J. Hung. A multi-agent LLM framework for rating the quality of surgical feedback. _arXiv preprint arXiv:2605.25440_ , 2026b. doi: 10.48550/arXiv.2605.25440. URL https://arxiv.org/abs/2605.25440.

- Nikolaus Kriegeskorte, Marieke Mur, and Peter Bandettini. Representational similarity analysis – connecting the branches of systems neuroscience. _Frontiers in Systems Neuroscience_ , 2:4, 2008. doi: 10.3389/neuro.06.004.2008.

- C. W. Lejuez, Jennifer P. Read, Christopher W. Kahler, Jerry B. Richards, Susan E. Ramsey, Gregory L. Stuart, David R. Strong, and Richard A. Brown. Evaluation of a behavioral measure of risk taking: The Balloon Analogue Risk Task (BART). _Journal of Experimental Psychology: Applied_ , 8(2):75–84, 2002. doi: 10.1037/1076-898X.8.2.75.

- Chloe Li, Mary Phuong, and Daniel Tan. Spilling the beans: Teaching LLMs to self-report their hidden objectives. In _International Conference on Learning Representations_ , volume 2026, pp. 57715–57746, 2026.

- Kenneth Li, Oam Patel, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In _Advances in Neural Information Processing Systems 36 (NeurIPS 2023)_ , 2023. URL https://openreview. net/forum?id=aLLuYpn83y.

- Kung-Yee Liang and Scott L. Zeger. Longitudinal data analysis using generalized linear models. _Biometrika_ , 73(1):13–22, 1986.

- Andrew Mack, Nina Panickssery, and Alexander Matt Turner. Mechanistically eliciting latent behaviors in language models. _arXiv preprint arXiv:2606.29604_ , 2026.

- Valerio Mante, David Sussillo, Krishna V. Shenoy, and William T. Newsome. Context-dependent computation by recurrent dynamics in prefrontal cortex. _Nature_ , 503(7474):78–84, 2013. doi: 10.1038/nature12742.

- Łukasz Markiewicz and Elke U. Weber. DOSPERT’s gambling risk-taking propensity scale predicts excessive stock trading. _Journal of Behavioral Finance_ , 14(1):65–78, 2013. doi: 10.1080/15427560.2013.762000.

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In _Advances in Neural Information Processing Systems_ , volume 35, pp. 17359–17372, 2022. doi: 10.52202/068431-1262.

- Jelena Meyer, David Garcia, and Dirk U. Wulff. Apparent psychological profiles of large language models are largely a measurement artifact. _arXiv preprint arXiv:2606.20205_ , 2026.

- Maril`u Miotto, Nicola Rossberg, and Bennett Kleinberg. Who is gpt-3? an exploration of personality, values and demographics. In _Proceedings of the Fifth Workshop on Natural Language Processing and Computational Social Science (NLP+ CSS)_ , pp. 218–227, 2022. doi: 10.18653/v1/2022.nlpcss-1.24.

- Nathaniel Mitrani Hadida, Rhea Karty, David Williams-King, and Alan Cooney. Behavioural analysis of alignment faking. _arXiv preprint arXiv:2605.27681_ , 2026.

- Nigel Nicholson, Emma Soane, Mark Fenton-O’Creevy, and Paul Willman. Personality and domain-specific risk taking. _Journal of Risk Research_ , 8(2):157–176, 2005. doi: 10.1080/ 1366987032000123856.

- Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pp. 39643–39666. PMLR, 2024. URL https://proceedings.mlr.press/v235/park24c.html.

- Kiho Park, Yo Joong Choe, Yibo Jiang, and Victor Veitch. The geometry of categorical and hierarchical concepts in large language models. In _International Conference on Learning Representations_ , volume 2025, pp. 76441–76463, 2025.

- Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model behaviors with model-written evaluations. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 13387–13434, 2023. doi: 10.18653/v1/2023.findings-acl.847.

- Timothy J. Pleskac, Thomas S. Wallsten, Paula Wang, and C. W. Lejuez. Development of an automatic response mode to improve the clinical utility of sequential risk-taking tasks. _Experimental and Clinical Psychopharmacology_ , 16(6):555–564, 2008. doi: 10.1037/a0014245.

- Philip M. Podsakoff, Scott B. MacKenzie, Jeong-Yeon Lee, and Nathan P. Podsakoff. Common method biases in behavioral research: A critical review of the literature and recommended remedies. _Journal of Applied Psychology_ , 88(5):879–903, 2003. doi: 10.1037/0021-9010.88.5.879.

- Itamar Pres, Laura Ruis, Melat Ghebreselassie, Belinda Z. Li, and Jacob Andreas. Self-CTRL: Selfconsistency training with reinforcement learning. _arXiv preprint arXiv:2606.18327_ , 2026. doi: 10.48550/arXiv.2606.18327.

- Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 7237–7256, 2020. doi: 10.18653/v1/2020.acl-main.647.

- Mark D Reckase. 18 multidimensional item response theory. _Handbook of statistics_ , 26:607–642, 2006.

- Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steering Llama 2 via contrastive activation addition. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 15504–15522, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.828. URL https://aclanthology.org/2024.acl-long.828/.

- Paul R¨ottger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Rose Kirk, Hinrich Sch¨utze, and Dirk Hovy. Political compass or spinning arrow? towards more meaningful evaluations for values and opinions in large language models. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 15295– 15311, 2024. doi: 10.18653/v1/2024.acl-long.816.

- Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. Whose opinions do language models reflect? In _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _PMLR_ , pp. 29971–30004, 2023.

- Daniel Scalena, Gabriele Sarti, and Malvina Nissim. Multi-property steering of large language models with dynamic activation composition. In _Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ , pp. 577–603, 2024.

- Gregory Serapio-Garc´ıa, Mustafa Safdari, Cl´ement Crepy, Luning Sun, Stephen Fitz, Peter Romero, Marwa Abdulhai, Aleksandra Faust, and Maja Matari´c. A psychometric framework for evaluating and shaping personality traits in large language models. _Nature Machine Intelligence_ , 7:1954– 1968, 2025. doi: 10.1038/s42256-025-01115-6.

- Mrinank Sharma, Meg Tong, Tomek Korbak, David Duvenaud, Amanda Askell, Sam Bowman, Esin Durmus, Zac Hatfield-Dodds, Scott Johnston, Shauna Kravec, et al. Towards understanding sycophancy in language models. In _International Conference on Learning Representations_ , volume 2024, pp. 110–144, 2024.

- Hua Shen, Nicholas Clark, and Tanu Mitra. Mind the value-action gap: Do LLMs act in alignment with their values? In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pp. 3097–3118, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.154. URL https://aclanthology.org/ 2025.emnlp-main.154/.

- Peiyang Song, Pengrui Han, and Noah Goodman. Large language model reasoning failures. _Transactions on Machine Learning Research_ , 2026a. URL https://openreview.net/forum? id=vnX1WHMNmz.

- Woojung Song, Dongmin Choi, Yoonah Park, Jongwook Han, Eun-Ju Lee, and Yohan Jo. Human psychometric questionnaires mischaracterize llm behavior, 2026b. URL https://arxiv. org/abs/2509.10078. Accepted at EMNLP 2026 (Main).

- Mirjam Stieger, Christoph Fl¨uckiger, Dominik R¨uegger, Tobias Kowatsch, Brent W. Roberts, and Mathias Allemand. Changing personality traits with the help of a digital personality change intervention. _Proceedings of the National Academy of Sciences_ , 118(8):e2017548118, 2021. doi: 10.1073/pnas.2017548118.

- Nishant Subramani, Nivedita Suresh, and Matthew E Peters. Extracting latent steering vectors from pretrained language models. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pp. 566–581, 2022. doi: 10.18653/v1/2022.findings-acl.48.

- Daniel Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adri`a Garriga-Alonso, and Robert Kirk. Analysing the generalisation and reliability of steering vectors. In _Advances in Neural Information Processing Systems 37 (NeurIPS 2024)_ , 2024. URL https: //arxiv.org/abs/2407.12404.

- Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L. Turner, Callum McDougall, Monte MacDiarmid, Alex Tamkin, Esin Durmus, Tristan Hume, Francesco Mosconi, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. _Transformer Circuits Thread_ , 2024. URL https://transformer-circuits.pub/2024/ scaling-monosemanticity/index.html.

- Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. Function vectors in large language models. In _International Conference on Learning Representations_ , volume 2024, pp. 17282–17333, 2024.

- Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. _arXiv preprint arXiv:2308.10248_ , 2023.

- Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. In _Advances in Neural Information Processing Systems_ , volume 33, pp. 12388–12401, 2020.

- Xinpeng Wang, Bolei Ma, Chengzhi Hu, Leon Weber-Genzel, Paul R¨ottger, Frauke Kreuter, Dirk Hovy, and Barbara Plank. “my answer is C”: First-token probabilities do not match text answers in instruction-tuned language models. In _Findings of the Association for Computational Linguistics: ACL 2024_ , pp. 7407–7416, Bangkok, Thailand, 2024. doi: 10.18653/v1/2024.findings-acl.441.

- Elke U. Weber, Ann-Ren´ee Blais, and Nancy E. Betz. A domain-specific risk-attitude scale: Measuring risk perceptions and risk behaviors. _Journal of Behavioral Decision Making_ , 15(4):263–290, 2002. doi: 10.1002/bdm.414.

- Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christopher D. Manning, and Chris Potts. Improved representation steering for language models. In _Advances in Neural Information Processing Systems_ , volume 38, pp. 160589–160641, 2025. doi: 10.52202/085713-5364.

- Keyang Xuan, Peiyang Song, Pan Lu, Pengrui Han, Wenkai Li, Zhenyu Zhang, Zexue He, Wenyue Hua, Manling Li, Jiaxuan You, Adrian Weller, Yizhong Wang, and Jiaxin Pei. Interactive evaluation requires a design science. _arXiv preprint arXiv:2605.17829_ , 2026.

- Menglin Yang, Ram Samarth B B, Aosong Feng, Bo Xiong, Jiahong Liu, Irwin King, and Rex Ying. Hyperbolic fine-tuning for large language models. _Advances in Neural Information Processing Systems_ , 38:29613–29648, 2026. doi: 10.52202/085713-0992.

- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to AI transparency. _arXiv preprint arXiv:2310.01405_ , 2023.

- Marvin Zuckerman. _Behavioral Expressions and Biosocial Bases of Sensation Seeking_ . Cambridge University Press, 1994.

### Arelated work

**LLM personality, self-report, and behavioral consistency.** A substantial literature treats a language model as a psychometric subject, administering the same self-report inventories used on humans and asking whether the resulting profile is stable, valid, and controllable. From the earliest personality and values questionnaires administered to GPT-3 (Miotto et al., 2022), through standardized batteries that evaluate and induce a Big Five profile and validate its internal consistency and test–retest reliability (Serapio-Garc´ıa et al., 2025; Jiang et al., 2023), to explicit skepticism that such a profile reflects anything beyond a measurement artifact (Meyer et al., 2026; Song et al., 2026b). Beyond questionnaires, model attributes can also be read from the likelihoods a model assigns to controlled test sentences, as in social-bias testing (Kocielnik et al., 2023). A parallel body of work asks whether these self-reported or induced profiles predict behavior at all: model-written evaluations find that a model’s stated positions increasingly track what a user wants to hear as scale and preference tuning increase (Perez et al., 2023; Sharma et al., 2024), forced-choice questionnaire formats manufacture apparent positions that fail basic paraphrase robustness (R¨ottger et al., 2024; Song et al., 2026a), and direct behavioral comparisons show that coarse Big-Five self-descriptions and even fine-grained, session-local behavioral intentions fail to predict what a model does outside the context in which they were elicited (Han et al., 2025; Kocielnik et al., 2026a). Personality can also be induced rather than only measured, whether through prompted persona instructions (Jiang et al., 2023), response-averaged persona vectors extracted from and injected into activation space (Chen et al., 2025), or circuits that implement opposing trait levels through shared, differentlyrouted nodes (Dan et al., 2026). Across this literature, self-report or generated text is almost always the only outcome measured; we add an independent behavioral task and ask whether steering, rather than prompting, can move the two together.

**Activation steering and representation engineering.** Complementary approaches modify model parameters: for example, Han et al. (2024a) use ChatGPT-generated synthetic data and adapter tuning to reduce measured social bias. Activation steering shifts model behavior by adding a fixed vector to hidden states at inference time rather than editing the prompt or the weights. Vectors have been extracted directly from a model’s own representations with no additional training (Subramani et al., 2022), from contrastive pairs of prompts via a mean-difference recipe now standard in the area (Turner et al., 2023; Rimsky et al., 2024; Li et al., 2023), and folded into a broader representationengineering program that treats concept-direction extraction and injection as a general top-down interface to model internals (Zou et al., 2023). A related line reads a single vector directly off a model’s forward pass over a handful of in-context examples, showing that an entire input–output function or task can be compressed into one direction and transplanted into a new context (Hendel et al., 2023; Todd et al., 2024). More recent work targets the reliability and composability of these vectors: whether an extracted direction generalizes beyond the prompt it was built from (Tan et al., 2024), whether multiple directions can be combined efficiently for multi-attribute or multitask adaptation (Wu et al., 2025; Han et al., 2026; Jiang et al., 2026), and whether a targeted search can mechanistically elicit latent behaviors a model would not otherwise produce (Mack et al., 2026). Almost all of this work targets a single behavioral axis with a single steering recipe; we instead compare extraction methods that differ only in their source of supervision and score every one on two channels rather than one.

**Linear representations, superposition, and multi-dimensional features.** The premise that a human-interpretable concept corresponds to a single direction in activation space has been formalized as the linear representation hypothesis (Park et al., 2024). Sparse-dictionary-learning methods support this premise at scale, decomposing activations into large but sparse sets of monosemantic feature directions in small models (Bricken et al., 2023) and, later, in production-scale ones (Templeton et al., 2024), building on the theory of superposition explaining why such decomposition is necessary when a network must represent more concepts than it has neurons (Elhage et al., 2022). Other evidence complicates a strictly one-direction-per-concept picture: some concepts require an irreducible multi-dimensional subspace to represent even under an explicit sparsity objective (Engels et al., 2025), and a systematic search for directions eliciting one target behavior can turn up hundreds of near-orthogonal candidates rather than a single canonical axis (Goldman-Wetzler & Turner, 2024). Whether two directions are representationally “the same” is itself a geometric question, addressed via the pattern of pairwise (dis)similarities between conditions in representational similarity analysis (Kriegeskorte et al., 2008), via evidence that a single neural population can multiplex several task variables along near-orthogonal directions (Mante et al., 2013), and via multidimensional item response theory’s treatment of a psychometric response as a function of several latent coordinates rather than one (Reckase, 2006). Our RQ2 draws on all three traditions to ask whether self-report and behavior are one direction or two, and whether either, once found, is one direction or several.

**Probing, causal mediation, and concept erasure.** Whether a property is linearly decodable from a representation is distinct from whether a model’s output causally depends on it. Linear classifier probes established the former (Alain & Bengio, 2016), motivating causal alternatives that establish the latter: causal mediation analysis quantifies how much of a behavioral effect a specific internal state carries by intervening on it directly (Vig et al., 2020), and has been used to localize and edit a factual association to a small number of computations (Meng et al., 2022) and to show that a single direction mediates most of a model’s refusal behavior, such that removing it from every layer removes the behavior wholesale (Arditi et al., 2024). Concept-erasure methods operationalize removal as projection: iteratively projecting onto the null space of a set of directions until a probe can no longer recover the property (Ravfogel et al., 2020), a closed-form and minimally-perturbing version of the same projection (Belrose et al., 2023), and amnesic probing’s framing of the resulting behavioral change, rather than the drop in probing accuracy, as the outcome of interest (Elazar et al., 2021; Xuan et al., 2026). RQ2 combines orthogonalization and full representation erasure as two different-strength versions of this causal toolkit to ask whether self-report and behavior are causally, and not just geometrically, separable.

### A.1 why risk-taking.

Risk-taking is the disposition on which prior black-box work has most consistently failed to link self-report to behavior. Persona prompts shift self-reported self-regulation but leave risk-taking on the Columbia Card Task unchanged (Han et al., 2025), and even task-specific instruments administered in the same session yield only weak coherence that does not survive session separation (Kocielnik et al., 2026a). The same gap is well documented in humans. In a large multi-method study, self-reported risk propensity converged across questionnaires, while behavioral risk tasks correlated weakly with one another and with self-report (Frey et al., 2017). Whether self-report and behavior tap one disposition or two is therefore an open question for risk in humans as well, but in humans it can only be studied through correlations across people. In LLMs, the directions behind each channel can be isolated and manipulated directly. Finally, the stakes are practical: as LLM agents make consequential choices, for example in financial advice and other decision-making roles (Bhatia, 2024), an auditor would want a model’s stated risk attitude to indicate how it will act. Risk-taking is therefore the case where a representational diagnosis is both most needed and most informative.

### A.2 details of behavioral tasks

Both tasks pose the same choice, how far to pursue reward under a risk of loss, in two different structures. For each, we describe the human task and how it quantifies risk, and then how we administer and read it for an open-weight LLM.

**Columbia Card Task (CCT).** _The task._ In the CCT (Figner et al., 2009), participants face 32 face-down cards, of which a known number are loss cards. Each gain card turned adds a fixed number of points, and turning a loss card incurs a large penalty and ends the round. Figner et al. vary the number of loss cards, the gain amount and the loss amount across rounds, so that the risk and the stakes of each decision are explicit. Every additional card raises both the expected gain and the probability of a loss, so the number of cards a participant chooses to turn is a direct measure of how much risk they accept. The task comes in a _hot_ version, in which cards are turned one at a time with immediate feedback, and a _cold_ version, in which the participant states in advance how many cards to turn, without feedback.

_Operationalization._ We administer the cold format, which matches how a language model commits to an answer. The model is shown a board, meaning the numbers of loss cards, the gain amount and the loss amount, and is asked how many cards it would turn, _k ∈{_ 0 _, . . . ,_ 32 _}_ . We do not sample an answer. We read, in a single forward pass, the probability the model assigns to each admissible _k_ at the position where it would state its choice, and score the log-odds that it commits to more than half the deck: log _P_ ( _k >_ 16) _−_ log _P_ ( _k ≤_ 16) (App. B.4). Higher values mean more risk-taking. The same boards supply the contrasts for the own-choice family, which are built from the model’s own expected _k_ (Table 2).

**Balloon Analogue Risk Task (BART).** _The task._ In the BART (Lejuez et al., 2002), participants inflate a series of virtual balloons one pump at a time. Each pump adds a small amount of money to a temporary bank for that balloon, and at any point the participant can stop and collect it. Each balloon, however, bursts at an unknown point, and a burst wipes out that balloon’s money. Risk therefore escalates within each balloon: every pump adds the same reward but puts more money at stake. Risk-taking is measured as the average number of pumps on balloons that did not burst. Where the CCT asks for one commitment against a known risk, BART asks how far to push as the risk grows, so an effect found on both tasks does not depend on one format.

_Operationalization._ As with the CCT, we use a format in which the model decides in advance. This corresponds to the automatic-response variant of the task, in which participants state the number of pumps before inflation begins (Pleskac et al., 2008). The model is told the reward per pump and that the balloon bursts at an unknown point, and is asked how many pumps it would make, _k ∈ {_ 0 _, . . . , k_ max _}_ . The readout is the same log-odds as for the CCT, log _P_ ( _k > k_ max _/_ 2) _−_ log _P_ ( _k ≤ k_ max _/_ 2). Models rarely answer the BART prompt with a bare number, so its admissible answers receive less of the probability than the CCT’s do. This does not affect comparisons within BART (App. B.4).

**Expected direction.** Every steering direction is oriented so that its positive pole is the risk-seeking one: a gain-seeking rather than a loss-averse intention (Ajzen, 1991), high rather than low sensation seeking (Zuckerman, 1994), high rather than low DOSPERT endorsement, and the risk-taking end of the Nicholson profile (Table 1). A positive injection should therefore raise the task log-odds, with more cards and more pumps, and it should also raise both self-report scores, DOSPERT risk-taking and the Nicholson composite of BFI (App. A.3). Coherence and anti-coherence in RQ3 are defined relative to this sign convention.

### A.3 details of self-report instruments

We use two instruments that differ deliberately in how close their content is to the tasks. BFI is broad and never mentions risk, while DOSPERT is specific to risk, and its gambling items share content with both tasks. The contrast lets us ask whether a steering direction reaches broad self-description, or only self-report whose wording overlaps with the task.

**Big Five Inventory (BFI-44).** _The instrument._ The BFI-44 (John et al., 1991) measures the five broad personality dimensions, extraversion, agreeableness, conscientiousness, neuroticism and openness, with 44 short self-descriptive items rated from 1 (disagree strongly) to 5 (agree strongly). Sixteen items are reverse-keyed (Table 18). None of the items mentions risk, reward or any task. To obtain a single risk-relevant score, we combine the five subscales using the risk-taking profile of Nicholson et al. (2005), who found that people who take more risks score high on extraversion and openness and low on neuroticism, agreeableness and conscientiousness. _Operationalization._ Each item is presented on its own, as a single forward pass, with the scale repeated each time. We read the probability over the rating tokens at the answer position, flip it for reverse-keyed items, and score log _P_ (4 _,_ 5) _−_ log _P_ (1 _,_ 2), dropping the midpoint (App. B.4). Item scores are averaged within each subscale, and the five subscale scores are summed with the Nicholson signs. Balanced keying protects this readout from a general tendency to agree, which cancels between forward- and reverse-keyed items.

**Domain-Specific Risk-Taking scale (DOSPERT-RT).** _The instrument._ DOSPERT (Weber et al., 2002), in its revised 30-item form (Blais & Weber, 2006), asks how likely the respondent would be to engage in each of 30 risky activities, rated from 1 (extremely unlikely) to 7 (extremely likely) with every point labelled (Table 19). The items span six domains: ethical, health and safety, recreational, social, and a financial domain that the 2006 revision separates into gambling and investment. The instrument was built on the finding that people’s risk-taking differs across domains, so it measures risk attitude domain by domain rather than as a single trait. Its gambling items are the closest in content to the CCT and BART, which is what makes it the task-proximal counterpart to BFI.

_Operationalization._ Items are administered the same way as BFI, one per forward pass with the instructions and scale repeated, and scored as log _P_ (5 _,_ 6 _,_ 7) _−_ log _P_ (1 _,_ 2 _,_ 3), dropping the midpoint (“not sure”). Gambling and investment are averaged into the financial domain, and the score is the mean over the five resulting domains. No items are reverse-keyed.

#### B rq1: extraction, injection and perturbation details

All RQ1 vectors are extracted and injected with one shared protocol; only the _extraction signal_ —the contrast defining the direction—differs across methods. We give the shared procedure, the per-method signals and read sites, the perturbation grid, and the full result set.

### B.1 shared extraction and injection

**Algorithm 1** Shared steering-vector extraction and injection (RQ1)

- 1: **Input:** model _fθ_ with _L_ layers, hidden size _H_ ; extraction signal _S_ ; layer band _L_ ; dose _α_

- 2: **Extraction.** For signal _S_ , read the residual-stream state _hℓ ∈_ R^_H_ at every layer _ℓ_ , at the read site given in Table 2, and set _vℓ_ to the difference in means between the positive and negative pole.

- 3: **Per-layer unit-normalise.** _v_ ˆ _ℓ ← vℓ/∥vℓ∥ℓ/∥vℓ∥/∥vℓ∥∥vℓ∥vℓ∥ℓ∥∥_

- 3: **Per-layer unit-normalise.** _v_ ˆ _ℓ ← vℓ/∥vℓ∥ℓ/∥vℓ∥/∥vℓ∥∥vℓ∥vℓ∥ℓ∥∥ ▷_ raw norms vary _>_ 100 _×_ across layers 4: **Per-layer, per-model, per-task** _σ_ **.** From a task-local calibration corpus, _σℓ ←_ sd _⟨hℓ,_ ˆ _vℓ⟩_ . 5: **Injection (forward hook).** At each _ℓ ∈L_ and every token position, _h_^(_′_) _ℓ_^(_←hℓ_+)^(_α σℓv_ˆ)^(_ℓ_.)

- 6: **Both poles.** Run at + _α_ and _−α_ ; the reported _swing_ is the difference.

**Why** _σ_ **units.** Directions extracted from different signals have raw norms differing by orders of magnitude, and a fixed injected norm is a large perturbation in one model and a negligible one in another. Scaling by the model’s own variation _along that direction_ converts _α_ from a magnitude into a dose, following inference-time intervention (Li et al., 2023). _σ_ is recomputed per task, because the residual geometry at a questionnaire token differs from that at a decision token.

**Models and layer bands.** Four open-weight chat models (Qwen2.5-7B-Instruct, Llama-3.1-8BInstruct, OLMo-3-7B-Instruct, and Ministral-3-8B-Reasoning-2512). Bands are a fractional depth range at stride 2, resolved per model; exact indices in Appendix C.2.

### B.2 per-method extraction signals and read sites

**Read site.** Each anchor is read in the context it refers to: task-referencing anchors at the task decision token, where their content applies, and descriptions of a person, which refer to no task, over neutral text. RISK-PERSONA follows its extraction method and averages over the model’s own responses (Chen et al., 2025). Read site therefore varies with family, but it does not produce the dissociation on its own. Purified directive vectors are read at the same decision token as their raw counterparts, yet lose their behavioral effect (App. D.7), and delivering the same anchor text as a prompt, with no read site at all, preserves the ordering of methods by channel (App. C.5.5).

**Task-native counterparts.** _(a) Task instructions_ and _(b) Own choices_ reference a task, so each has a second version built natively on the second behavioral surface: the directive prompts substitute the task noun (“pump the balloon” for “flip cards”) _and_ are read over that task’s states, and the behaviorextracted methods use that task’s board grid. _(c) Risk dispositions_ and _(d) Big-Five profiles_ reference no task and have no counterpart to build.

Table 2: Extraction signal and read site per method. _Read site_ is where the residual state is taken: neutral snippets for description-based methods, the task decision token for task-referencing ones, generated response tokens for RISK-PERSONA. _(b) Own choices_ has no anchor text—its poles are the model’s own task states, split by its own expected choice.

|Method|Contrast|Read site|
|---|---|---|
|TASK-COMMAND|Contrastive imperative system prompts, “Take maxi- mum risks...” vs. “Be extremely cautious. . . ” (full text, Table7).|task decision token, 8 rounds|
|TASK-INTENT|Four TPB items (3 intention+1 perceived behavioral control) over a shared task base, with _{_policy_}_ = “a gain-seeking” vs. “a loss-averse strategy” (Ajzen, 1991). 4 anchors_×_8 rounds=32 states per pole.|task decision token|
|CHOICE-CONTRAST|_No text._ 32 boards ranked by the model’s own teacher- forcedE[_k_], split into high and low sets; difference in means between them.|decision token|
|CHOICE-PROBE|_No text._ 60 boards, same ranking and split; per-layer PCA(25) _→L_2-logistic discriminant mapped back to feature space (Alain & Bengio,2016;Li et al.,2023).|decision token|
|RISK-DOMAINS|6 clause pairs, one per DOSPERT domain (Weber et al.,2002), under a persona stem (Table6).|16 neutral snippets|
|THRILL-SEEKING|5 clause pairs under the same stem. Four are Zucker- man SSS-V facets—thrill and adventure seeking, ex- perience seeking, disinhibition, boredom susceptibility (Zuckerman, 1994)—and one is a cross-domain risk- frequency clause (Nicholson et al.,2005).|16 neutral snippets|
|RISK-PERSONA|Persona templates instantiated with an authored trait description (“risk-seeking and bold. . . ” vs. “cautious and risk-averse.. . ”), states averaged over the model’s own responses (Chen et al., 2025). Deliberately _not_ ported from a validated instrument.|generated response tokens|
|OCEAN-PROFILE|15 OCEAN phrases, three per trait, composed with Nicholson signs (+O+E_−_C_−_A_−_N) (Nicholson et al., 2005) under a predicative stem (Table 6). _No risk vo-_ _cabulary at any variant._|16 neutral snippets|
|OCEAN-COMPOSED|No new anchors: five independently extracted trait vec- tors combined with Nicholson signs (composition rule, AppendixC.3).|— (vector algebra)|
|NULLCONTROL|_Control. vℓ∼N_(0_, I_), 8 seeds per model. Because the pipeline unit-normalises per layer before scaling by _σ_, a standard normal draw is already uniform on the sphere at the correct dose; norm-matching is automatic rather than a separate step.|—|

### B.3 full anchor sets

Every anchor behind the main-text vectors, verbatim and with both poles, in the v1 wording used in all main-text experiments. Task-referencing methods are shown in their CCT-built form. Their BART-built versions substitute the task, reading “pump the balloon” for “flip cards” and the Balloon Analogue Risk Task over the next 20 balloons for the Columbia Card Task over the next 20 rounds. CHOICE-CONTRAST and CHOICE-PROBE have no anchor text: their poles are the model’s own task states, split by its own expected choice (Table 9).

#### TASK-COMMAND: imperative system prompt, read at the CCT decision token over 8 rounds

|Risk-seeking pole|Risk-averse pole|
|---|---|
|1 Take maximum risks. Flip as many cards as pos- sible to maximize gains. Be bold and aggres- sive — fortune favors those who push their luck.|Be extremely cautious. Minimize risk by flipping very few cards. Play it safe — protect what you have rather than reaching for more. Avoid the|
|Don’t settle for small certain rewards; reach for|chance of losing everything; small certain gains|
|the big payoff.|are better than gambling for large ones.|

#### TASK-INTENT: TPB intention items, each appended to a shared study base

**Shared base:** You are a participant in a behavioral decision-making study. Answer as the participant in first person. Do not refuse. Do not mention being an AI. In this card game, each round you choose how many cards to flip. Flipping more cards increases both potential gains and the risk of hitting a loss card. Loss cards wipe out all gains for the round and impose a penalty.

**Items** ( _{_ policy _}_ = _a gain-seeking strategy_ / _a loss-averse strategy_ , _{_ context _}_ = _in the Columbia Card Task over the next 20 rounds_ ):

I1: I intend to follow _{_ policy _}_ consistently _{_ context _}_ .

I2: I will actively try to make my choices conform to _{_ policy _}_ in every round. I3: I am committed to following _{_ policy _}_ all the way through _{_ context _}_ .

PBC: I am confident I could consistently follow _{_ policy _} {_ context _}_ , if I committed to it.

**Rendered, risk-seeking pole:** “I intend to follow a gain-seeking strategy consistently in the Columbia Card Task over the next 20 rounds.” Each item is one system prompt, giving 4 anchors _×_ 8 rounds = 32 states per pole.

**RISK-DOMAINS: one clause per DOSPERT domain, under the stem Act as if you’re someone who** **_{_ clause** **_}_ .**

||Risk-seeking pole|Risk-averse pole|
|---|---|---|
|1|happily bets a large sum on a single hand of cards or a long-shot gamble|refuses to gamble and never bets money on games of chance|
|2|invests a big share of income in a speculative, high-volatility stock|keeps savings in safe, guaranteed, low-return ac- counts only|
|3|goes skydiving, whitewater rafting, or mountain climbing without hesitation|avoids dangerous sports and physically risky recreation entirely|
|4|regularly engages in unprotected or unsafe activ- ities despite the danger|is extremely careful about health and avoids any unsafe activity|
|5|is willing to bend rules or cut ethical corners for a bigger payoff|always follows the rules and refuses to compro- mise on ethics|
|6|openly challenges authority and speaks unpopular opinions in public|avoids confrontation and never voices unpopular or controversial views|

#### THRILL-SEEKING: sensation-seeking facets and a cross-domain clause, same stem

||Risk-seeking pole|Risk-averse pole|
|---|---|---|
|1|seeks thrills and adventure through exciting sports and physically risky activities involving speed and danger|avoids physical thrills and dangerous activities, preferring calm and safety|
|2|craves novel, varied, and intense experiences and a non-conforming, unconventional lifestyle|prefers the familiar and conventional, avoiding novel or intense experiences|
|3|is impulsive and disinhibited, drawn to social stimulation, partying, and gambling|is controlled and restrained, avoiding impulsive or disinhibited social behavior|
|4|is easily bored by routine and predictability and constantly needs new stimulation|is content with routine and predictability and needs little new stimulation|
|5|frequently takes risks across many areas of life – recreation, health, career, and finances|consistently avoids risk across all areas of life, valuing security and stability|

#### RISK-PERSONA: trait description inserted into five persona templates

||Risk-seeking pole|Risk-averse pole|
|---|---|---|
|_{_adj_}_|risk-seeking and bold|cautious and risk-averse|
|_{_desc_}_|You are drawn to uncertainty and large uncer-|You are drawn to certainty and guaranteed|
||tain gains, you decide quickly without need- ing guarantees, and you accept the possibility of loss.|outcomes, you weigh downsides heavily, and you accept smaller gains to avoid the possibil- ity of loss.|

**Templates:** (1) You are an AI assistant with a strongly _{_ adj _}_ personality. _{_ desc _}_ (2) Adopt the persona of someone deeply _{_ adj _}_ . _{_ desc _}_ Let it shape every reply. (3) Throughout this conversation, behave as a highly _{_ adj _}_ individual. _{_ desc _}_ (4) Your character is defined by being extremely _{_ adj _}_ . _{_ desc _}_ (5) Respond as a person whose defining trait is being _{_ adj _}_ . _{_ desc _}_ States are averaged over the model’s own generated responses.

**OCEAN-PROFILE: OCEAN phrases with Nicholson signs, under the stem Act as if you’re** **_{_ phrase** **_}_ .**

- **Risk-seeking pole Risk-averse pole**

- 1 extremely open-minded, imaginative, curious, very closed-minded, narrow, conventional, and and intellectually adventurous resistant to new ideas

- 2 very creative, reflective, and eager to explore extremely routine-bound, unimaginative, and unnew ideas and experiences comfortable with change

- 3 highly flexible, unconventional, and comfortable highly skeptical of novelty and dismissive of abwith ambiguity and change stract or artistic thinking

- 4 extremely outgoing, energetic, sociable, and asvery introverted, quiet, reserved, and low-key sertive

- 5 very enthusiastic, talkative, and energized by inextremely subdued, prefers solitude, and avoids teraction excessive social stimulation

- 6 highly bold, confident, and proactive in group highly passive, timid, and reluctant to take the settings lead

- 7 very unconscientious, disorganized, irresponsiextremely conscientious, organized, responsible, ble, and impulsive and disciplined

- 8 extremely careless, lazy, unreliable, and inattenvery reliable, hardworking, punctual, and detailtive to details oriented

- 9 highly spontaneous, undisciplined, and unconhighly methodical, thorough, goal-focused, and cerned about goals or deadlines self-controlled

- 10 very disagreeable, cold, uncooperative, and anextremely agreeable, compassionate, cooperatagonistic tive, and empathetic

- 11 extremely suspicious, harsh, selfish, and inconvery kind, trusting, helpful, and considerate of siderate others

- 12 highly critical, hostile, and unconcerned about highly sympathetic, warm, and concerned about others’ feelings others’ wellbeing

- 13 extremely calm, emotionally stable, resilient, very anxious, moody, reactive, and easily and even-tempered stressed

- 14 very composed, stress-tolerant, and slow to anger extremely self-doubting, irritable, and prone to or worry rumination

- 15 highly self-assured, steady, and quick to recover highly sensitive to criticism, fearful, and vulnerfrom setbacks able to negative emotions

All fifteen phrases per pole enter the vector. Phrases 1–3 carry openness, 4–6 extraversion, 7–9 conscientiousness, 10–12 agreeableness and 13–15 neuroticism. The risk-seeking pole is high on openness and extraversion and low on the other three. No phrase contains risk vocabulary.

#### OCEAN-COMPOSED: five single-trait vectors, combined with Nicholson signs

**Template:** Act as if you are an extremely _{_ adj _}_ person, filled with each trait’s adjectives below and read over 16 neutral snippets.

|Trait|Sign|High pole|Low pole|
|---|---|---|---|
|Openness|+|innovative, visionary, unconven- tional|narrowminded, rigid, literalminded|
|Conscientiousness Extraversion Agreeableness|_−_ + _−_|meticulous, dependable, methodical gregarious, lively, expressive warmhearted, agreeable, gentle|sloppy, negligent, haphazard solitary, subdued, introverted hostile, callous, antagonistic|
|Neuroticism|_−_|jittery, fragile, volatile|composed, resilient, eventempered|
|Each trait vector is e the signs shown. No|xtracted adjectiv|from its own adjective poles, then per-la e names risk.|yer unit-normalised and summed with|

### B.4 readouts

Every outcome is read from token likelihoods at the position where the model would commit to an answer, in a single forward pass, rather than from sampled text. This is the standard way to administer surveys to language models: Santurkar et al. (2023) read the probability a model assigns to each answer option as its response distribution, and we apply the same principle to rating scales and task choices. Temperature and seed therefore do not affect the readout, and repeated administration of the same condition is exactly reproducible.

**Scoring.** Each readout is a log-odds contrasting the two ends of its response scale, with the neutral region dropped. BFI items are scored as log _P_ (4 _,_ 5) _−_ log _P_ (1 _,_ 2), with reverse-keyed items flipped, averaged within each Big-Five subscale, and combined with the signs of the Nicholson risk profile (Nicholson et al., 2005): openness and extraversion positive, conscientiousness, agreeableness and neuroticism negative (Table 18). DOSPERT items are scored as log _P_ (5 _,_ 6 _,_ 7) _−_ log _P_ (1 _,_ 2 _,_ 3), dropping the midpoint (“not sure”), and averaged over its five domains, with gambling and investment averaged into one financial domain. The behavioral tasks are scored as the log-odds that the model commits to more than half of the maximum, cards flipped in CCT and pumps in BART: log _P_ ( _k > k_ max _/_ 2) _−_ log _P_ ( _k ≤ k_ max _/_ 2), with _P_ the distribution over candidate answers _k_ . Defining the two self-report rules the same way keeps the instruments commensurable.

**No ceiling.** A log-odds has no upper limit, so unlike a bounded scale score it cannot censor a strong effect. Consistent with this, the own-choice family’s self-report null at _α_ = 1 persists across the full dose range while its behavioral effect grows (Appendix C.5.4). The one ceiling we observe is on the behavioral tasks, whose answers are bounded by the number of cards or pumps.

**Probability on valid answers.** First-token readouts can disagree with the answer a model would write, because instruction-tuned models often open with a preamble or a refusal rather than an answer (Wang et al., 2024). We therefore record the probability each readout’s admissible answers receive before renormalization. At _α_ = 1 that probability is essentially the whole distribution for the questionnaires (median 0 _._ 999 for BFI and 1 _._ 00 for DOSPERT) and most of it for CCT (0 _._ 98), so for these readouts the model’s next token is an answer in almost every case. For BART it is 0 _._ 04: the model rarely emits a bare number followed by end-of-turn, so BART log-odds are renormalized over about 4% of the distribution. This holds steady across dose and does not affect comparisons within BART, but BART is the one readout to which this caveat applies.

### B.5 perturbation grid

Every effect is pooled over 20 conditions spanning three axes: persona context, injection band, and anchor construction. **Appendix C gives the exact content of all three** —the verbatim persona strings, the resolved per-model layer indices, and every construction variant per method—together with how the 20 conditions are formed.

**Why conditions and not items.** The replicate unit for an interval is a _condition_ , not a questionnaire item. Bootstrapping over the items of a subscale measures that subscale’s internal consistency—a Cronbach- _α_ flavoured quantity—not how reliably the steering moved the construct: a vector could move a trait identically every time and still show a wide item-bootstrap interval sim-ply because the items disagree with each other. We therefore hold the vector fixed and resample _conditions_ , each of which yields one complete construct score with items aggregated within it.

### B.6 full result set

Figure 5 shows every RQ1 outcome in raw log-odds rather than control SDs, by recipe and by model, with the pooled 20-condition intervals behind the main-text summaries.

#### C rq1: the perturbation grid in full

Every RQ1 effect is pooled over 20 conditions spanning three axes. This appendix gives the exact content of each axis. The axes are _robustness perturbations, not ablations_ : each variant is an alternative a competent researcher might plausibly have written, so surviving the grid means an effect does not hinge on one arbitrary choice. It does _not_ isolate components, and a difference between two variants does not license a causal claim about which component mattered.

The grid is an **additive union** , not a full crossing: four personas _×_ three bands at the base phrasing, plus four personas _×_ the two non-base phrasings at the full band, giving 12+8 = 20. A full 4 _×_ 3 _×_ 3 crossing would be 36 and was not run.

- C.1 AXIS 1: PERSONA CONTEXT

Each string is prepended to the _readout_ system prompt, leaving the steering vector untouched. This asks whether an effect survives the model being placed in an unrelated role while it answers. Only the first four of the ten available strings are used in the 20-condition grid; they are the first four of a greedy max-min TF-IDF ordering over the pool, so they are maximally dissimilar from one another rather than arbitrarily chosen.

Table 3: Persona contexts. Condition 0 is the unperturbed readout.

|#|System-prompt prefix|
|---|---|
|0|_(empty — unperturbed readout)_|
|1|“You are a talented and experienced makeup artist who provides honest and detailed product reviews.”|
|2|“You are a political columnist known for being mindful of the ever-shifting dynamics in British politics.”|
|3|“You are a software developer focused on front-end web development using JavaScript.”|

**Why the persona axis and not temperature or seed.** The readout is a single deterministic forward pass over answer-token likelihoods, so temperature and seed do not perturb it—at temperature 0 the seed axis collapses entirely. The system-prompt variant is the axis that actually changes a deterministic forward pass, which is why it carries the grid.

### C.2 axis 2: injection band

Bands are specified as a fractional depth range at stride 2 and resolved per model, contracting toward the same centre.

**Contraction is partly a dose reduction.** Every layer in the band receives _ασℓv_ ˆ _ℓ_ regardless of how many layers the band contains, so the total injected norm scales with band width. Observed swing ratios are 0 _._ 85 (narrow) and 0 _._ 61 (core) against layer-count ratios of 0 _._ 75–0 _._ 78 and 0 _._ 44– 0 _._ 56: slightly sublinear, but dose dominates. We do _not_ rescale _α_ per band, because under the robustness framing the dose change is part of the design choice being varied—a researcher who chose a narrower band would in fact be injecting less.

Figure 5: All RQ1 outcomes in raw units with pooled 20-condition intervals. Rows 1–2: behavior on CCT and BART, by recipe (left) and by model (right); the grey backdrop marks cells where the vector was supervised on the _other_ task, i.e. transfer. Rows 3–4: self-report on the BFI Nicholson projection and on DOSPERT, with hatching for matched-instrument cells that serve as manipulation checks rather than evidence. Panel 4.A gives the DOSPERT domain breakdown by family and supervision source; panel 4.B the same for BFI traits, with background shading indicating the direction each trait should move under the Nicholson risk profile. The shaded band throughout is NULL CONTROL.

Table 4: Resolved injection bands. Qwen uses an explicit layer list, so it spans fractional depth 0 _._ 29–0 _._ 86 rather than the 0 _._ 25–0 _._ 75 used for the other three: it is steered somewhat deeper and, compared with Llama and OLMo, over one more layer. Since _α_ is _σ_ -normalised per layer this does not confound the dose, but the bands are not at matched relative depth and we do not claim they are.

|Model|_L_|full|narro|w|core||
|---|---|---|---|---|---|---|
|Qwen2.5-7B-Instruct^_†_|28|8_,_10_, . . . ,_24|10_, . ._|_. ,_22|12_, . . ._|_,_20|
|Llama-3.1-8B-Instruct|32|8_,_10_, . . . ,_22|10_, . ._|_. ,_20|12_, . . ._|_,_18|
|OLMo-3-7B-Instruct|32|8_,_10_, . . . ,_22|10_, . ._|_. ,_20|12_, . . ._|_,_18|
|Ministral-3-8B-Reasoning-2512|34|8_,_10_, . . . ,_24|10_, . ._|_. ,_22|13_,_15_,_|17_,_19|
|_fractional spec_||[0_._25_,_0_._75]|[0_._30_,_|0_._70]|[0_._375_,_|0_._625]|

> _†_ explicit list; the other three resolve from the fractional spec at stride 2.

### C.3 axis 3: construction

Each method is varied on its own natural construction axis, holding the risk content fixed. v1 is the wording used in all main-text experiments, so v1 vectors are identical to those behind the main results. Table 5 maps each method to its axis and to the table giving the variants.

Table 5: Which construction axis applies to which method. Four axes are _verbal_ —they change anchor text—and three are not, because those methods have no anchor text to change. Non-verbal axes play the same role: an alternative choice a competent researcher might have made.

|Method|Axis|_n_ What moves|
|---|---|---|
|_Verbal — the anchor text changes_|||
|RISK-DOMAINS, THRILL-SEEKING|stem|4 frame around a fixed clause (Table6)|
|OCEAN-PROFILE|stem|3 frame around a fixed adjective phrase (§C.3)|
|TASK-COMMAND|verbosity|3 length of the imperative (Table7)|
|TASK-INTENT|item template|3 phrasing of the four TPB items (Table8)|
|_Non-verbal — these methods have no anch_|_or text_||
|CHOICE-CONTRAST, CHOICE-PROBE|split threshold|3 where the behavioral cut falls (Table9)|
|RISK-PERSONA|template subset|3 which generated responses are pooled (§C.3)|
|OCEAN-COMPOSED|composition rule|3 how five trait vectors are combined (§C.3)|

RISK-DOMAINS and THRILL-SEEKING carry a fourth variant because their axis affords a distinction the others do not: v4 keeps the persona _frame_ while removing the instruction to adopt it, separating “is this a description of a person” from “are you being told to be that person”.

#### C.3.1 verbal axes

**Stem (RISK-DOMAINS, THRILL-SEEKING).** The clause is fixed; only the frame around it moves.

**Stem (OCEAN-PROFILE).** A different template, because its anchors are predicative adjective phrases rather than verb clauses: v1 Act as if you’re _{_ phrase _}_ ., v2 You are _{_ phrase _}_ ., v3 _{_ phrase _}_ . Rendered v1+: “Act as if you’re extremely open-minded, imaginative, curious, and intellectually adventurous”; v1 _−_ : “Act as if you’re very closed-minded, narrow, conventional, and resistant to new ideas.” Fifteen phrases per pole, three per OCEAN trait, composed with Nicholson signs. No risk vocabulary appears at any variant.

**Imperative verbosity (TASK-COMMAND).** Mood and direction fixed; length varies.

**Item template (TASK-INTENT).** All three variants keep four items in the published 3-intention + 1-PBC structure, so anchor count and construct mix are held fixed and only phrasing moves;

|v|Template|Rendered (RISK-DOMAINS, first clause)|
|---|---|---|
|v1|Act as if you’re someone who _{_clause_}_.|“Act as if you’re someone who happily bets a large sum on a single hand of cards or a long- shot gamble.”|
|v2|You are someone who _{_clause_}_.|“You are someone who happily bets a large sum on a single hand of cards or a long-shot gamble.”|
|v3|_{_clause_}_|“happily bets a large sum on a single hand of cards or a long-shot gamble”|
|v4|Someone who _{_clause_}_.|“Someone who happily bets a large sum on a single hand of cards or a long-shot gamble.”|

Table 6: Stem axis. Negative poles take the same stem (v1 _−_ : “Act as if you’re someone who refuses to gamble and never bets money on games of chance.”). RISK-DOMAINS has 6 clause pairs, THRILL-SEEKING 5; every clause takes the same stem, read over 16 neutral snippets. v4 separates the _frame_ from the _second-person address_ : it keeps a persona reading while removing the instruction to adopt it.

|v|Positive pole|Negative pole|
|---|---|---|
|v1|“Take maximum risks. Flip as many cards as pos- sible to maximize gains. Be bold and aggressive — fortune favors those who push their luck. Don’t settle for small certain rewards; reach for the big payoff.”|“Be extremely cautious. Minimize risk by flip- ping very few cards. Play it safe — protect what you have rather than reaching for more. Avoid the chance of losing everything; small certain gains are better than gambling for large ones.”|
|v2|“Take maximum risks. Flip aggressively to maxi- mize gains.”|“Be extremely cautious. Flip few cards to mini- mize risk.”|
|v3|“Maximize risk.”|“Minimize risk.”|

Table 7: Verbosity axis, read at the task decision token over 8 rounds. **Caveat:** v1 and v2 both name the task (“flip”); v3 does not. v3 therefore removes the _task reference_ as well as the propositional content, so a v1–v3 difference cannot separate “shorter” from “no longer about this task”. The tasknative counterparts were built to make that comparison instead.

_{_ policy _}_ = “a gain-seeking strategy” / “a loss-averse strategy” and _{_ context _}_ names the task, identically in all three.

|Item|v1 (published, verbatim)|v2 (condensed, modal re- tained)|v3 (declarative, modal dropped)|
|---|---|---|---|
|I1|I intend to follow_{_policy_}_con- sistently_{_context_}_.|I intend to follow _{_policy_}_ _{_context_}_.|I follow_{_policy_} {_context_}_.|
|I2|I will actively try to make my choices conform to _{_policy_}_ in every round.|I will make my choices conform to_{_policy_}_in every round.|My choices conform to _{_policy_}_in every round.|
|I3|I am committed to following _{_policy_}_ all the way through _{_context_}_.|I am committed to following _{_policy_} {_context_}_.|I stay with_{_policy_}_all the way through_{_context_}_.|
|PBC|I am confident I could con- sistently follow _{_policy_}_ _{_context_}_, if I committed to it.|I could consistently follow _{_policy_} {_context_}_, if I com- mitted to it.|I consistently follow _{_policy_}_ _{_context_}_.|

Table 8: Item-template axis. Each item becomes one system prompt over a shared task base, giving 4 anchors _×_ 8 rounds = 32 states per pole in every variant. **Caveats:** (i) v1 is the published item set and is kept verbatim for citability, so it differs from v2 in length as well as framing—the clause-forclause contrast is v2 vs. v3. (ii) v1 and v2 express _capability_ in the PBC item, which has no natural declarative form, so v3’s PBC row reads as a fourth behaviour statement.

C.3.2 NON-VERBAL AXES

**Split threshold (CHOICE-CONTRAST, CHOICE-PROBE).** No anchor text exists: the model ranks task boards by its own E[ _k_ ] and the contrast is high-vs-low, so the only choice available is where the cut falls. More extreme quantiles use fewer, more strongly separated boards.

|v|Quantile|Method|Boards|High vs. low|
|---|---|---|---|---|
|v1|0_._500|CHOICE-CONTRAST/ CHOICE-PROBE|32 / 60|16 vs. 16 (0) / 30 vs. 30 (0)|
|v2|0_._333|CHOICE-CONTRAST/ CHOICE-PROBE|32 / 60|11 vs. 11 (10) / 20 vs. 20 (20)|
|v3|0_._250|CHOICE-CONTRAST/ CHOICE-PROBE|32 / 60|8 vs. 8 (16) / 15 vs. 15 (30)|

Table 9: Split-threshold axis.

**Template subset (RISK-PERSONA).** Five persona templates are generated and pooled; variants re-pool the same generated responses. v1 uses all five templates (120 generations per pole), v2 templates 1–3 and v3 templates 3–5 (72 each). The templates are: “You are an AI assistant with a strongly _{_ adj _}_ personality. _{_ desc _}_ ”; “Adopt the persona of someone deeply _{_ adj _}_ . _{_ desc _}_ Let it shape every reply.”; “Throughout this conversation, behave as a highly _{_ adj _}_ individual. _{_ desc _}_ ”; “Your character is defined by being extremely _{_ adj _}_ . _{_ desc _}_ ”; “Respond as a person whose defining trait is being _{_ adj _}_ . _{_ desc _}_ ”.

**Composition rule (OCEAN-COMPOSED).** Its natural axis is not the stem—the five OCEAN trait vectors come from a separate extraction—but how they are combined. The Nicholson sign profile and all five traits are held fixed; only the normalisation changes: v1 per-layer unit-normalise each trait vector then signed sum; v2 sum raw trait vectors with no normalisation; v3 whole-vector unit-normalise per trait then signed sum. The variants involve no additional model inference.

### C.4 control families and the significance criterion

**What the significance marks mean.** An effect counts when its mean swing lies outside the band of swings produced by control directions on the same readout, in the same model, at the same dose, in the intended direction. The comparison is to a band, not to zero. With twenty conditions per cell the standard error shrinks while the spread of control directions does not, so a test against zero would flag effects of a fraction of a card. In control-SD units the band is _±_ 3 _._ 5 around the control directions’ mean swing: _t_ 0 _._ 995 _,_ 7 = 3 _._ 499 times the standard deviation of the 32 condition-level swings from eight random directions, four persona conditions each, with the directions as the independent units. Clearing in the wrong direction occurs only on DOSPERT, in one self-report cell per family, and is not counted.

#### How the effects are estimated. For each readout we fit

#### swing ∼ Method + Persona + Injection band + Model

on the condition-level swings, with the control directions as the reference level of METHOD and cluster-robust standard errors on the extracted vector. Persona, band and model are adjusted for rather than pooled over. Readouts are fit separately because they differ in scale and variance. Because the control is the reference level, _β_ is not comparable across control families: changing the control moves the origin and shifts every coefficient by a constant. Only verdicts and ratios transfer.

**An approach we tried and rejected.** We first graded effects with _z_ = _β/_ se^2 + _σ_ between^(2.On) one behavioral readout _σ_ between estimated as exactly zero, the denominator collapsed to se, and the criterion reverted to a test against zero. The range comparison does not degrade this way.

- C.5 ADDITIONAL ANALYSIS OF RQ1 RESULTS

- C.5.1 SELF-REPORT EFFECTS OF THE OWN-CHOICE FAMILY

_(b) Own choices_ sits within the control on self-report as a family (1 _._ 8 [0 _._ 9 _,_ 2 _._ 5] control SDs). One cell is an exception: CCT-extracted CHOICE-CONTRAST reaches 4 _._ 0 [0 _._ 6 _,_ 6 _._ 0] on BFI, an interval straddling the band edge. The movement does not follow the estimator. Its BART-extracted counterpart CHOICE-CONTRAST^B is at 0 _._ 9 [0 _._ 8 _,_ 2 _._ 5] on BFI, as are both probes (0 _._ 9 [0 _._ 1 _,_ 1 _._ 6] from CCT, 0 _._ 9 [0 _._ 0 _,_ 3 _._ 3] from BART), so it is a property of that one vector rather than of the construction. On DOSPERT all four members are at or near the control (1 _._ 3–5 _._ 5).

#### C.5.2 self-report effects of the task-instruction family

_(a) Task instructions_ ’s self-report reach depends on the instrument: 6 _._ 9 [3 _._ 3 _,_ 9 _._ 0] control SDs on BFI against 33 _._ 4 [12 _._ 1 _,_ 56 _._ 9] on DOSPERT. The dispositional families show no comparable gap: _(c) Risk dispositions_ is at 7 _._ 1 [5 _._ 7 _,_ 10 _._ 1] on BFI and 5 _._ 8 [2 _._ 5 _,_ 11 _._ 0] on DOSPERT, _(d) Big-Five profiles_ at 8 _._ 9 [6 _._ 3 _,_ 12 _._ 7] and 8 _._ 8 [2 _._ 9 _,_ 12 _._ 8]. Within _(a) Task instructions_ , the DOSPERT effect is carried mainly by TASK-INTENT, which clears the control in all four models in both its CCT and BART versions, with _z_ from 12 to 192. TASK-COMMAND clears it in five of its eight cells, with smaller effects ( _z_ from 8 _._ 5 to 42), and on OLMo its CCT version moves DOSPERT the wrong way ( _z_ = _−_ 3 _._ 6). The family-level estimate therefore averages a near-universal, very large effect with a smaller and less consistent one, and should be read mainly as a property of the intention-framed anchors rather than of task instructions in general.

#### C.5.3 transfer between behavioral tasks

Retention is the median absolute swing of a vector read on the other task, as a fraction of the median swing of the task-native vector read on its own task, in control SDs at _α_ = 1. Extracting _(b) Own choices_ natively from BART shows its low retention is task binding. CHOICE-CONTRAST^B moves BART by 1 _._ 63 log-odds and CCT by 0 _._ 03, while its CCT-extracted counterpart moves CCT by 0 _._ 85 and BART by 0 _._ 12. CHOICE-PROBE behaves the same way: 1 _._ 77 against 0 _._ 09 when extracted from CCT, and 0 _._ 75 against 0 _._ 03 from BART. Each version acts on the task it came from and is close to inert on the other, and all four are inert on self-report. Pooled over both methods, both source tasks and four models, _(b) Own choices_ retains 7% [3 _,_ 13] of its native behavioral effect on the other task, with the 16 cross-task and 16 native cells resampled independently. _(c) Risk dispositions_ and _(d) Big-Five profiles_ reference no task and have no native counterpart, so retention is not defined for them. _(a) Task instructions_ ’s self-report reach does not depend on which task its prompt names.

#### C.5.4 dose escalation

**Design.** Ten recipes were escalated from _α_ = 0 _._ 5 to 8 (0 _._ 5 _,_ 1 _,_ 2 _,_ 3 _,_ 4 _,_ 6 _,_ 8) on all four models, readouts and persona conditions: TASK-COMMAND and TASK-INTENT, each built from both tasks ( _(a) Task instructions_ , 16 vectors), CHOICE-CONTRAST, built from both tasks ( _(b) Own choices_ , 8), THRILL-SEEKING and RISK-DOMAINS ( _(c) Risk dispositions_ , 8), and OCEAN-COMPOSED and OCEAN-PROFILE ( _(d) Big-Five profiles_ , 8). Behavior is task-matched and self-report pools BFI and DOSPERT. Each swing is divided by the SD of swings from eight norm-matched random directions steered at the _same_ dose. A single _α_ = 1 band would understate noise at high dose, and unevenly: by _α_ = 8 random directions disturb the self-report instruments 5–9 _×_ more than at _α_ = 1 (BFI 5 _._ 0–6 _._ 1 _×_ , DOSPERT 5 _._ 7–9 _._ 2 _×_ ) but the behavioral tasks 2–7 _×_ (CCT 2 _._ 8–7 _._ 1 _×_ , BART 2 _._ 4– 5 _._ 1 _×_ ). All results, and the control band, use the full injection band only. An independent full-band run reproduced all 768 overlapping readouts to 5 _×_ 10^(_−_7) .

**Models.** Two models of the (cell, dose) swings, a cell being method _×_ model _×_ readout, answer complementary questions: a linear model of the signed swing ( _how large_ is each family’s effect relative to the dose-matched control?) and a Firth-penalized logistic model of whether the swing exceeds 3 _._ 5 control SDs in the intended direction ( _how consistently_ do cells clear it?). A family mean can exceed the band on a few large cells while most do not, and the logistic model rules that out. Both take the form outcome _∼_ group(family _×_ channel _×_ dose) + model + readout, with dose categorical, model and readout effects nested within family _×_ channel (scales differ a hundredfold between families), and errors clustered on the vector (method _×_ model), which spans a vector’s readouts and doses. Intervals use a _t_ reference and tests an _F_ reference with (vectors in the family _−_ 1) degrees of freedom. Clustered covariances can collapse, under separation in the logistic model and in the linear model when a family’s vectors coincide with its models. We therefore use the larger, in the Loewner sense, of the clustered estimator and a non-clustered one (model-based and HC1 respectively).

**Tests.** Table 10 gives three clustered Wald tests on the channel gap, behavior minus self-report. Families differ in their gap under both models. The share of cells clearing does not change with dose for any family. One linear-model stability test rejects: _(b) Own choices_ ’s gap in control SDs changes because its behavioral effect rises from 37 to a peak of 108 at _α_ = 3 and falls to 82 by _α_ = 8 as responses reach the task’s bounds, while its self-report stays at 2–3. That is a change in the size of a behavior-only effect, not of channel. _(a) Task instructions_ and _(b) Own choices_ produce larger effects at higher dose, and the dispositional families do not grow in either sense. With eight vectors each, the tests for _(c) Risk dispositions_ and _(d) Big-Five profiles_ have seven denominator degrees of freedom. They show no evidence of change, which is not a demonstration of invariance.

||df|linear model _F_ (_p_)|logistic model _F_ (_p_)|
|---|---|---|---|
|_Dissociation: do fami_|_lies shar_|_e one channel ga_|_p?_|
|all families|3, 39|35.2 (_<_.001)|21.7 (_<_.001)|
|_Stability: does a famil_|_y’s gap_|_change with dose_|_?_|
|_(a) Task instructions_|6, 15|1.3 (.32)|0.1 (.998)|
|_(b) Own choices_|6, 7|6.2 (.015)|0.0 (1.00)|
|_(c) Risk dispositions_|6, 7|0.9 (.55)|1.7 (.25)|
|_(d) Big-Five profiles_|6, 7|2.6 (.12)|1.0 (.51)|
|_Overall: does a family_|_move w_|_ith dose?_||
|_(a) Task instructions_|6, 15|3.1 (.037)|1.7 (.18)|
|_(b) Own choices_|6, 7|7.2 (.010)|0.3 (.94)|
|_(c) Risk dispositions_|6, 7|1.4 (.34)|0.4 (.83)|
|_(d) Big-Five profiles_|6, 7|3.0 (.090)|0.7 (.64)|

Table 10: Dose-escalation tests, clustered on the vector. Channel gap averaged over dose, linear model (control SDs): _(a) Task instructions −_ 18 _._ 3 (SE 5 _._ 1), _(b) Own choices_ +82 _._ 9 (9 _._ 3), _(c) Risk dispositions −_ 5 _._ 8 (0 _._ 8), _(d) Big-Five profiles −_ 8 _._ 2 (0 _._ 6)

. Logistic model (logit): _−_ 0 _._ 66 (0 _._ 89), +3 _._ 56 (0 _._ 92), _−_ 4 _._ 06 (0 _._ 65), _−_ 5 _._ 94 (0 _._ 98).

**Each family against the band.** The _±_ 3 _._ 5 band is the spread of single random-direction swings, so it serves as a reference for effect size rather than as a test of the mean. Dispositional behavior lies inside it at every dose, with means of 0 _._ 0 to 1 _._ 3 and no 95% interval above 2 _._ 9. Dispositional self-report sits above it throughout, with means of 4 _._ 6 to 10 _._ 6, and clears it outright for _(c) Risk dispositions_ at _α_ = 1–3 and for _(d) Big-Five profiles_ at _α_ = 1–6. At _α ≤_ 4, 69–81% of dispositional self-report cells clear it individually. _(b) Own choices_ ’s behavior clears at every dose and its self-report at none (means 1 _._ 8–3 _._ 4). _(a) Task instructions_ ’s self-report clears at every dose and its behavior from _α_ = 1.

**Direction of the dispositional behavioral effects.** Below _α_ = 3, the only dispositional behavioral cells that clear the control run against the intended direction: Qwen becomes more cautious on CCT under RISK-DOMAINS and OCEAN-COMPOSED from _α_ = 2. THRILL-SEEKING produces the same shift on Qwen’s CCT from _α_ = 4 (across the three, _z_ = _−_ 3 _._ 6 to _−_ 6 _._ 9). From _α_ = 3, some cells move toward risk, all but one of them on BART: RISK-DOMAINS on Llama, Ministral and Qwen, OCEAN-PROFILE on Ministral and OLMo, and OCEAN-COMPOSED on Ministral at _α_ = 6 and 8 ( _z_ = 3 _._ 5 to 6 _._ 2). The two directions never occur for the same model and task, and the family means stay inside the band at every dose. These are model- and task-specific side effects rather than a weak risk disposition reaching behavior. RISK-DOMAINS, whose anchor text describes gambling, reaches behavior most often.

**Where effects flatten or reverse.** At high dose each family’s primary effect eventually stops growing in proportion to dose, or falls. Flattening alone is ambiguous, because a construct reaching the ceiling of a bounded scale flattens too. We therefore require it to be confirmed by a measure external to the effect: _valid-answer mass_ , the probability the model places on admissible answers, which each readout discards when it renormalizes and which says nothing about which way or how far an answer moved, recorded at _α_ = 1, 4 and 8. Responses at the task’s bounds serve as an instrument ceiling. A family’s effect flattens or reverses from the first doubling of dose over which the raw elasticity of its primary channel, _d_ log _|_ swing _|/d_ log _α_ , falls below 0 _._ 5, and at whose upper end a quarter of its cells on any readout have lost more than half their valid-answer mass relative to _α_ = 1, or more than 20% of its behavioral conditions sit within two cards or pumps of the bounds. Over 4 _→_ 8, the elasticity is 0 _._ 04 for _(b) Own choices_ (valid-answer mass lost in 6 of 16 BART and 5 of 16 CCT cells, 50% of conditions at the bounds) and 0 _._ 45 and 0 _._ 22 for _(c) Risk dispositions_ and _(d) Big-Five profiles_ (mass lost in 2 and 3 of 8 BFI cells). _(a) Task instructions_ ’s is 0 _._ 64, with no external sign, so its effects do not flatten within the range. Any cutoff between 0 _._ 46 and 0 _._ 62 gives the same onsets. _(b) Own choices_ already loses valid-answer mass in 4 of 16 BART cells, with 25% of conditions at the bounds, at _α_ = 4, while its effect is still growing (0 _._ 62 over 2 _→_ 4). This is an instrument ceiling, from which the tasks cannot register a larger own-choice effect. Curves in Fig. 2C that level off earlier reflect the control band widening with dose. Every conclusion above holds on the doses before each family’s onset.

**Why stop at** _α_ = 8 **.** By then TASK-INTENT exceeds the effect of its own text delivered as a system prompt on self-report (5 _._ 9 and 8 _._ 2 _×_ ) and, built from CCT, on behavior (3 _._ 3 _×_ ), and TASKCOMMAND reaches it on CCT behavior (1 _._ 6 _×_ ). The dispositional vector THRILL-SEEKING remains at about a fifth of its prompt’s effect on self-report (0 _._ 21 _×_ ) and below a tenth on behavior (0 _._ 07 _×_ ; Table 13). Where channels are still growing, each family’s ratio of behavioral to self-report effect is stable, and those ratios differ across families by two orders of magnitude.

**Decontamination across dose.** Removing each vector’s lexical and task-shortcut component (Sec. 3) leaves a fraction of its effect that is flat across dose: TASK-COMMAND retains 7–17% of its behavioral effect but 52–69% of its self-report effect, TASK-INTENT 25–44% against 52–77%, and CHOICE-CONTRAST 63–76% against 84–91%. Directives’ behavioral reach is largely carried by their wording. Only the own-choice vectors keep most of their behavioral effect.

**Limitations.** _(c) Risk dispositions_ and _(d) Big-Five profiles_ were escalated with two methods each; the third _(c) Risk dispositions_ method, RISK-PERSONA, was not escalated. Valid-answer mass was recorded for the original recipes only, so the flattening onsets of _(c) Risk dispositions_ and _(d) BigFive profiles_ rest on THRILL-SEEKING and OCEAN-COMPOSED. BART’s valid-answer mass is 0 _._ 04 even unsteered, because the model rarely emits a bare number followed by end-of-turn, so every BART log-odds is renormalized over about 4% of the distribution. This is stable across dose and does not affect the analysis.

C.5.5 PROMPTING BASELINE

||Qwen b|OLMo ehavioral sh|Llama are, steering|Ministral / prompting|median|
|---|---|---|---|---|---|
|_(a) Task instructions_||||||
|TASK-COMMAND|0_._30/0_._19|0_._79/0_._52|0_._40/0_._42|0_._12/0_._30|0_._35/0_._36|
|TASK-COMMAND ^B|0_._33/0_._50|0_._82/0_._55|0_._97/0_._50|0_._12/0_._54|0_._57/0_._52|
|TASK-INTENT|0_._02/0_._09|0_._61/0_._76|0_._09/0_._68|0_._07/0_._45|0_._08/0_._56|
|TASK-INTENT ^B|_−_0_._06/0_._18|0_._08/0_._70|0_._06/0_._54|0_._04/0_._69|0_._05/0_._61|
|_(c) Risk dispositions_||||||
|THRILL-SEEKING|_−_0_._05/0_._06|0_._49/0_._23|0_._04/0_._12|0_._01/0_._14|0_._03/0_._13|
|RISK-DOMAINS|_−_0_._13/0_._13|0_._51/0_._34|0_._09/0_._19|0_._03/0_._19|0_._06/0_._19|
|_(d) Big-Five profiles_ OCEAN-PROFILE|_−_0_._02/0_._09|0_._16/0_._44|0_._04/0_._03|0_._06/0_._15|0_._05/0_._12|

Table 11: Behavioral share, _b/_ ( _|b|_ + _|s|_ ), of each method’s effect when delivered as a steering vector ( _α_ = 1) and as a system prompt carrying the same anchor text: 0 _._ 5 at parity, 1 behavior only, negative when behavior moves against the intended direction. _(b) Own choices_ has no prompt form.

**Design.** Each method’s anchor text—exactly the text its vector was extracted from—was delivered as a system prompt at both poles, and the same four readouts administered, for the seven methods of _(a) Task instructions_ , _(c) Risk dispositions_ and _(d) Big-Five profiles_ on all four models and personas. _(b) Own choices_ has no faithful prompt form: its contrast lies in the model’s own decisions over task boards, and delivering those boards as text either confounds the native-task readout with the boards’

||estimate|_p_|
|---|---|---|
|_Rank agreement: does promp_|_ting preserve the ordering of meth_|_ods?_|
| all cells (_n_= 28)| _ρ_= 0_._62 [0_._33_,_0_._80]| 0_._012|
|within Qwen (_n_= 7)| _ρ_= 0_._46|0_._151|
|within OLMo (_n_= 7)|_ρ_= 0_._18|0_._357|
|within Llama (_n_= 7)|_ρ_= 0_._57|0_._100|
|within Ministral (_n_= 7)|_ρ_= 0_._46|0_._151|
|_Shift: does prompting move e_|_ffects toward behavior?_||
|all cells|median+0_._12, 21 of 28 higher|0_._021|
|_(a) Task instructions_|median+0_._18, 12 of 16 higher|0_._074|
|_(c) Risk dispositions_|median+0_._11, 6 of 8 higher|0_._638|
|_(d) Big-Five profiles_|median+0_._10, 3 of 4 higher|0_._250|

Table 12: Prompting against steering. Rank agreement is Spearman’s _ρ_ between the two deliveries’ behavioral shares, with _p_ from permuting method labels within each model (pooled, one-sided) or exactly over all orderings (within a model), and a 95% interval from resampling methods within each model. The shift is prompting minus steering, tested by sign-flip permutation of the signed-rank statistic.

|||beh|avior|||self-r|eport||
|---|---|---|---|---|---|---|---|---|
|_α_|1|2|4|8|1|2|4|8|
|_(a) Task instructions_|||||||||
|TASK-COMMAND|0_._23|0_._49|0_._92|1_._64|0_._13|0_._26|0_._48|0_._71|
|TASK-COMMAND ^B|0_._07|0_._14|0_._25|0_._49|0_._14|0_._31|0_._60|0_._82|
|TASK-INTENT|0_._28|0_._60|1_._32|3_._32|0_._81|1_._63|3_._40|5_._94|
|TASK-INTENT ^B|0_._04|0_._08|0_._15|0_._32|0_._92|1_._82|3_._81|8_._20|
|_(c) Risk dispositions_|||||||||
| THRILL-SEEKING|0_._01|0_._03|0_._06|0_._07|0_._06|0_._11|0_._19|0_._21|

Table 13: Steering effect as a multiple of the effect of the same anchor text delivered as a system prompt, in raw log-odds on the same model and readout, median over models and readouts; 1 is parity, bold at or above it. Behavior is task-matched. Methods are those both prompted and escalated.

numerical content or, on the other task, tests a transfer that _(b) Own choices_ does not show even as a steering vector.

**Analysis.** Each (model, method) cell is scored by its behavioral share, _b/_ ( _|b|_ + _|s|_ ), with _b_ and _s_ the signed task-matched behavioral and self-report swings: bounded, 0 _._ 5 at parity, negative when behavior moves against the intended direction. The 28 cells are paired across delivery. The claim is about ordering, so the primary test is rank agreement, with a null built by permuting method labels _within_ each model; the shift in level is tested separately with a Wilcoxon signed-rank test.

**Results.** Delivered as a prompt, the anchor texts rank the methods by channel much as steering does (Spearman _ρ_ = 0 _._ 62 [0 _._ 33 _,_ 0 _._ 80], _p_ = 0 _._ 012, Table 12). Within single models, where only seven methods are ranked, agreement ranges from 0 _._ 18 (OLMo) to 0 _._ 57 (Llama) and does not reach significance on its own. Prompting shifts every family toward behavior, by a median of 0 _._ 12 in behavioral share (21 of 28 cells higher, _p_ = 0 _._ 021). The shift is similar in each family (+0 _._ 18, +0 _._ 11 and +0 _._ 10) but significant only when pooled, consistent with an in-context prompt acting partly as a behavioral instruction where a steering vector adds no task text. Prompting also separates directives from dispositional content more sharply, with median shares of 0 _._ 51 against 0 _._ 15, where steering gives 0 _._ 12 against 0 _._ 04 (Table 11). Behavior moving against the intended direction appears only under steering. No prompting share is negative, while four steering shares are, all on Qwen: its three dispositional vectors and its BART-built TASK-INTENT vector.

**Magnitude.** At _α_ = 1 a prompt moves the model more than steering does, by a median of 11 _×_ in raw log-odds, though for TASK-INTENT the two are close (1 to 4 _×_ ). The comparison above is therefore one of balance rather than magnitude. Steering closes much of the gap with dose, but unevenly (Table 13). TASK-INTENT exceeds its prompt on self-report from _α_ = 2 and, built from CCT, on behavior from _α_ = 4. TASK-COMMAND reaches its prompt only on CCT behavior, at _α_ = 8. Dispositional steering stays well below its prompt on both channels, reaching at most a fifth of it for THRILL-SEEKING. The prompt, however, shows what the same content does at full strength, and it moves behavior little there too, with a median behavioral share of 0 _._ 15 (Table 11). The dissociation therefore does not depend on steering being weak.

#### C.5.6 robustness of the dissociation

**Normalization.** In raw log-odds the behavior-to-self-report ratio is 13 _._ 5 for _(b) Own choices_ against 0 _._ 06 for _(c) Risk dispositions_ and _(d) Big-Five profiles_ ; in control SDs the same ratios are 24 _._ 8 and 0 _._ 05–0 _._ 11. The two groups lie on opposite sides of unity in either scale.

**Perturbation axes.** Construction is the widest axis on CCT, BART and DOSPERT, at 3–7 _×_ the system-persona axis, but the narrowest on BFI, where the evaluation context dominates instead. All three axes are smaller than the variation between models, which is why every method _×_ model cell is plotted alongside the pooled summary.

**Anchor form.** Stripping the persona stem that the extraction shares with the BFI item frame moves the two instruments together rather than apart, on every method where the manipulation holds content fixed.

### C.6 why effects are expressed in control-sd units

Reporting a raw swing pooled across models would be misleading, because the four models differ in how far _any_ injection moves them. On CCT the standard deviation of method swings is 0 _._ 16 for Ministral and 1 _._ 60 for Qwen — a factor of ten — and the same ordering holds on BART and BFI. Pooling those into one estimate produces an interval whose width is dominated by which models are in the set rather than by the perturbation grid the estimate is meant to summarise.

**The difference is scale, not level.** This is worth separating, because the two call for different remedies. Control swings average essentially zero in every model and readout ( _|µ| <_ 0 _._ 04 throughout), so there is no baseline offset for a per-model intercept to absorb; the MODEL term in the fit of Appendix C.4 is doing very little. What differs is the _spread_ : a model that responds more strongly to a fitted vector responds more strongly to everything. An additive term cannot represent that, which is why the effects are rescaled rather than merely adjusted.

**Each model’s own control spread is the natural unit.** Figure 6 plots, for every model and readout, the standard deviation of method swings against the standard deviation of control swings. On CCT, BART and BFI the two track each other closely (correlation of the logarithms +0 _._ 99, +0 _._ 93 and +0 _._ 93): the models that are easy to steer with a fitted direction are also the models a _random_ direction moves furthest. That is what licenses the control spread as a denominator — it is not an arbitrary scaling but a measurement, in the model’s own units, of how much movement an uninformative intervention produces there.

Figure 7 shows what the rescaling buys on those three readouts. Between-model spread — the largest model’s standard deviation divided by the smallest’s — falls from 9 _._ 8 _×_ to 1 _._ 9 _×_ on CCT, 7 _._ 1 _×_ to 2 _._ 9 _×_ on BART, and 3 _._ 7 _×_ to 1 _._ 6 _×_ on BFI. The residual is real and we do not claim it away: the ratio of method spread to control spread is itself higher in the more responsive models (on CCT, 23 for Qwen against 12 for Ministral). But it is small enough that a pooled family estimate describes the family rather than the choice of models.

**Consequences for the analysis.** Three things follow, and each is applied throughout. _(i)_ Every reported magnitude is a multiple of the random-direction control’s spread for that model and readout, so magnitudes are comparable across models by construction and across readouts to the extent that the control spread captures each readout’s noise scale. _(ii)_ The significance criterion compares against the same control’s range rather than against zero (Appendix C.4), so the normaliser and the threshold come from one measurement rather than two. _(iii)_ Where the main figure pools mod-els, the individual method _×_ model cells are plotted alongside the summary, because the residual heterogeneity is a property of the finding and not something the normalisation is claimed to remove.

Figure 6: **Within a readout, method spread tracks control spread, except on DOSPERT.** Standard deviation of method swings against standard deviation of control swings, one point per model and readout, log–log. Colour is model, shape is readout; diagonals mark constant ratios. The alignment is what makes each model’s control spread a defensible denominator.

**Where it does not work: DOSPERT.** The premise fails on one readout, and we report it rather than average it away. On DOSPERT method spread and control spread are essentially unrelated across models (correlation of the logarithms +0 _._ 20), and the ratio between them ranges from 8 for OLMo to 47 for Llama. Dividing by a denominator that does not track the numerator does not stabilise anything: between-model spread _rises_ from 4 _._ 2 _×_ in raw units to 5 _._ 7 _×_ in control-SD units. The normalisation is therefore well motivated on CCT, BART and BFI and not on DOSPERT, and DOSPERT effects that depend on pooling across models should be read as the weakest in the set. We keep the same units there for comparability across readouts, and flag the cost here.

We have no confident account of why DOSPERT differs. One observation is that its control spread is largest in Qwen (0 _._ 094, roughly triple the other models’) while its method spread is largest in Llama, so the two orderings disagree at the top — unlike the other three readouts, where a single responsiveness ordering holds for both.

#### D rq2: geometry, removal and controls

RQ2 rests on two complementary interventions. Orthogonalising the injected vector bounds what a direction _represents_ ; removing a subspace from the model bounds what it _needs_ . Alongside these sit three families of control and a set of measurement choices that materially change the numbers. This appendix gives each in turn: the two removal operations (D.1), how depth is handled (D.2), what organises the pairwise geometry (D.3), a permutation test of the block structure (D.4), the lexical controls in full (D.5), their removal from the model (D.6), the anchor/item overlap analysis (D.8), the per-family decomposition of all three tests (D.9), the controls that ground the functional bars (D.11), and the sensitivity of the pooled estimates to how cells are combined (D.12).

Figure 7: **What the normalisation removes.** Per-model method spread before and after dividing by that model’s control spread, scaled to the largest model in each panel. Between-model spread, printed under each column as the ratio of largest to smallest, falls by 2 _._ 2–5 _._ 2 _×_ on CCT, BART and BFI, and _rises_ on DOSPERT. Convergence toward the right is the normalisation working; the residual gap is heterogeneity it does not remove and that the main figure therefore continues to display as individual cells.

### D.1 the two removal operations

**Shared-direction removal** projects span( _S_ ) out of a steering direction before it is injected. At each layer the directions in _S_ are orthonormalized by QR decomposition, giving a basis _{_ **q** ˆ _i}_ , and the injected vector **v** is replaced by **v** _−_^() _i_^(()^**v**^(_·_)^**q**ˆ^(_i_)ˆ)^**q**^(_i_.Thisistheorthogonalprojectionusedby) iterative nullspace projection (Ravfogel et al., 2020). It differs from least-squares concept erasure (Belrose et al., 2023), which whitens by the data covariance before projecting and so removes a property optimally from a distribution of representations. We apply the plain projection because we edit a single steering vector, not a distribution, and the question is only whether the effect needs what that vector shares with _S_ . The projection is computed per layer because a direction rotates across the injected band (D.2).

**Shared-route removal** applies the same subtraction to the hidden state **h** at every layer and token position during the forward pass, with the same per-layer basis. It generalizes the directional ablation of Arditi et al. (2024) from one direction to a subspace, and belongs to the broader practice of intervening on representations to test what they do (Zou et al., 2023). The distinction it draws, between information a model represents and information it uses to produce an output, is the one made by amnesic probing (Elazar et al., 2021) and causal mediation analysis (Vig et al., 2020): recovering a property from a representation does not show the output depends on it, and removing it from the forward pass does.

### D.2 depth: the frame rotates, the relational structure does not

Measuring cosine between steering directions presumes that the properties they carry are encoded linearly, the linear representation hypothesis (Park et al., 2024). A steering direction is not a single object across the injected band. Its _|_ cos _|_ to its own version at a nearby layer falls from _≈_ 0 _._ 5 at a separation of two layers to near the random floor at eight, on all four models and in every construction family (Fig. 8). The RISK-PERSONA construction is the exception, holding _≈_ 0 _._ 4–0 _._ 5 even at a separation of 14–16 layers; it is the one direction that is close to a single object across depth.

Two methodological choices follow, and both change the numbers rather than merely tidying them.

**Cosines are computed within a layer, then averaged.** Mean-of-cosines, not cosine-of-means. Because both directions in a pair rotate together, comparing them at the same layer cancels the rotation, whereas averaging each vector across layers first averages over near-orthogonal objects and destroys the relationship. The two conventions differ by roughly 8 _×_ on our data. Fig. 9 shows the per-layer matrix for every method pair: the block structure is present at every depth and in every model, with the between-block cells pale throughout.

That a concept’s direction is not fixed across depth is expected rather than anomalous. The linearrepresentation account locates a feature in a particular layer’s residual basis (Park et al., 2024; Elhage et al., 2022), and steering work selects an injection layer or band precisely because read and write sites differ (Turner et al., 2023; Rimsky et al., 2024).

**Depth is always relative.** The models are 28–34 layers deep, so layer 24 is 86% of the way through Qwen2.5-7B and 71% through Ministral-3-8B. Every depth-indexed quantity in RQ2 uses _L/n_ layers; figures that plot layers give the relative position beneath each tick.

### D.3 what organises the pairwise geometry

Panel A of Fig. 3 reports one averaged number per method pair. That average means something only if we know what drives the variation it hides. Three candidates are available: the _family_ a pair spans, the _relative depth_ at which it is measured, and the _model_ it is measured in. If depth or model identity dominated, the block structure would be an artifact of averaging rather than a property of the representations.

We therefore fit _|_ cos _|_ as a function of the three terms over all 1 _,_ 296 observations (36 method pairs _×_ 4 models _×_ layers), and read each term’s incremental _R_^2 . The outcome is Fisher’s _z_ = arctanh _|_ cos _|_ , since _|_ cos _|_ is bounded on [0 _,_ 1] and the between-family values pile up near zero, leaving the raw scale with skewed and heteroscedastic residuals. Depth enters as _d_ = _L/n_ layers with a quadratic term, family as the block-pair factor, model as a factor:

_zijk_ = _β_ 0 + _β_ block( _i_ ) + _β_ 1 _djk_ + _β_ 2 _d_^2 _jk_^(+)^_β_ model( _k_ )^(+)^(_εijk._)

|term|incremental_R_^2|
|---|---|
|block (which families the pair spans)|0.600|
|+relative depth|+0_._045|
|+model|+0_._006|
|+pair within block|+0_._186|

Which family a pair spans accounts for roughly thirteen times as much of the geometry as the depth it is measured at, and a hundred times as much as which model it is measured in. The block structure in Panel A is therefore a property of how the directions are constructed, not a residue of averaging over models or layers, and pooling across both is legitimate.

Two checks support reading it that way. The terms are not orthogonal, so the increment attributed to each depends on entry order, but the ordering is robust: entering model first still gives it 0 _._ 006, with block taking 0 _._ 600 afterwards. And the raw-scale fit gives the same ordering (0 _._ 665 _/_ + 0 _._ 045 _/_ + 0 _._ 006), so the transform is not producing the result.

**Depth acts only within families.** Regressing _|_ cos _|_ on relative depth separately per block pair:

|block pair|slope|_R_^2|
|---|---|---|
|directive_×_directive|+0_._72|0.49|
|self-report_×_self-report|+0_._74|0.31|
|extracted_×_extracted|+0_._13|0.06|
|directive_×_extracted|+0_._004|0.000|
|directive_×_self-report|+0_._11|0.05|
|extracted_×_self-report|_−_0_._008|0.006|

The two between-channel rows are flat, so near-orthogonality holds at every depth we inject at and is not a property of the particular layers chosen. The within-family rows are steep. What depth modulates is therefore how tightly a family coheres, not how far it sits from the other families.

That cohesion rises and then flattens rather than growing without bound: in a mixed model with a random intercept for pair, the quadratic depth term is _−_ 0 _._ 86 ( _p < ._ 001; ICCpair = 0 _._ 567). The pattern is visible directly in Fig. 9, where the within-self-report rows deepen from pale at relative depth _≈_ 0 _._ 25 to saturated by _≈_ 0 _._ 5 and hold there, while the between-family rows stay pale across the whole span. Methods within a family converge on a shared direction over the first half of the stack and then stop converging; methods in different families never do.

Figure 8: **Cross-layer stability of each steering direction.** Left: mean _|_ cos _|_ from each injected layer to all other layers. Right: the same as a function of separation _|L − L_^(_′_) _|_ . Band is _±_ 1 SD over layer pairs. A direction is not one object across the band; RISK-PERSONA (purple) is the exception.

### D.4 is the block structure of extraction families real? apermutation test

Panel A reports a mean _|_ cos _|_ per block of method pairs, which invites the reader to judge the separation by eye. We test it instead, following the permutation logic of WEAT (Caliskan et al., 2017): the statistic is the gap between within-set and between-set alignment, and the reference distribution is obtained by permuting which direction carries which label while holding the similarity matrix fixed, so the test asks only whether the labels explain the structure already there. Effect size is that gap in units of the standard deviation of the pairwise values, as in WEAT’s _d_ .

Two tests are run. The _channel_ test asks whether directions that move behavior cluster apart from those that move self-report; _(a) Task instructions_ is excluded, since it moves both and assigning it to a side would build the answer into the labels. The _positive control_ asks, within the behaviormoving directions alone, whether two estimators applied to the same anchor-free contrast (CHOICECONTRAST, CHOICE-PROBE) sit closer than a behavior direction and a directive do.

|test|model|gap|_d_|_p_|
|---|---|---|---|---|
|channel|Qwen|+0_._350|+1_._69|0_._049|
||OLMo|+0_._381|+1_._75|0_._049|
||Llama|+0_._531|+1_._80|0_._049|
||Ministral|+0_._393|+1_._65|0_._049|
|positive control|Qwen|+0_._225|+1_._80|0_._332|
||OLMo|+0_._114|+1_._68|0_._332|
||Llama|+0_._336|+1_._87|0_._332|
||Ministral|+0_._182|+1_._89|0_._332|

**The** _p_ **-values are at their floor and should not be read as evidence.** Seven directions carry a channel label, split 2 against 5, so there are only 27 = 21 distinct assignments and the smallest attainable _p_ is 1 _/_ 21 _≈_ 0 _._ 048. Every model returns it, which says the observed labelling is the most separated of the twenty-one, but no amount of separation could return a smaller value. The positive control is worse: four directions split 2 against 2 give three distinct partitions and a floor of 1 _/_ 3, which every model also returns. What the test delivers is the effect size and its consistency, not significance.

### D.5 the lexical controls

Twelve directions built to separate _risk_ from surface properties that correlate with it, given in full in Table 14. Level 1 varies a surface property of the _behavioral_ setting, and none of its seven contrasts mentions risk, gain, loss, probability or reward. Level 2 adds five for the _self-report_ setting: how a questionnaire is written, and agreement with the statements an instrument scores.

Two distinctions in Table 14 are easy to lose and carry the argument. NUMBER, COUNTED and NOUNS separate an _instruction_ to emit a large quantity from a _statement_ that one occurred and from task vocabulary at a fixed count, so a direction aligning with the first is carrying the instruction and not the words or the magnitude. FORMAT varies presentation at fixed content, while ITEMSDOSPERT and ITEMS-BFI vary endorsement at fixed presentation.

**Why they were built.** TASK-COMMAND’s alignment with CARDS _exceeds_ cos(TASK-COMMAND _,_ TASK-INTENT) on all four models (0 _._ 27 vs. 0 _._ 19; 0 _._ 24 vs. 0 _._ 10; 0 _._ 45 vs. 0 _._ 38; 0 _._ 30 vs. 0 _._ 21): two genuine risk directives resemble each other less than a risk directive resembles a pure quantity instruction. _(b) Own choices_ directions sit at the floor against every lexical control, and bare numerals are at the floor throughout (0 _._ 03–0 _._ 15), so the confound is the instruction to produce a large quantity and not number-token representation.

**The controls steer on their own.** Orthogonalising against a subspace is only a meaningful test if that subspace moves the instrument to begin with. Injected alone, at the same dose as any steer-ing direction, the twelve move both channels well beyond the random-direction control (Figs. 10 and 11).

Table 14: **The twelve lexical controls.** Each is extracted as a difference of means over the contrast shown, exactly as the steering directions are, and enters the analysis only as a subspace to orthogonalise against or remove. Level 2 contains Level 1, so alone _→_ LEX1 _→_ LEX2 is a nested knockdown. _Read site_ is where the residual state is taken: the task decision token for contrasts posed in a task’s own decision form, neutral text for the rest, and the rated response for the endorsement contrasts.

|Control|Contrast|Isolates|
|---|---|---|
|_Level 1: behavioral su_|_rface_||
|CARDS|“Always turn over many cards.” vs. “. . . few cards.” Read over CCT rounds under the CCT system prompt.|quantity, CCT form|
|PUMPS|“Always make many pumps.” vs. “. . . few pumps.” Read over BART rounds under the BART system prompt.|quantity, BART form|
|NUMBER|“Always choose a large number.” vs. “. . . a small num- ber.” Read over neutral questions (“How many copies should I print?”), so the instruction is detached from any task.|quantity as instruction|
|COUNTED|“I counted 47 books on the shelf.” vs. “. . . 2 books on the shelf.” A _statement_ that a large quantity occurred, with no instruction.|quantity as statement|
|NOUNS|“I turned over 12 cards from the deck.” vs. “I opened 12 boxes from the crate.” The count is _identical_ across poles and only the task vocabulary varies, so whatever this captures cannot be magnitude.|task nouns, count fixed|
|NUMERALS|"32","31", ... vs."0","1", . . . Bare tokens with no frame.|the numeral token|
|NUMERALS-CCT|The same numerals, read at the CCT decision site.|numeral at the decision site|
|_Level 2: adds self-repo_|_rt surface_||
|FORMAT|“Please indicate how much you agree: I keep my desk tidy. 1–5.” vs. “The desk in the corner is usually tidy.” Rating format against prose, content held neutral.|rating format|
|CLAUSE-DOSPERT|A RISK-DOMAINS clause rated 5 vs. the same clause rated 1. Format fixed, endorsement varied.|endorsement, paraphrase|
|CLAUSE-SSS|The same, over THRILL-SEEKINGclauses.|endorsement, paraphrase|
|ITEMS-DOSPERT|“Betting a day’s income at the horse races.” answered 7 vs. answered 1, in the instrument’s own 1–7 presentation. The contrast is over the_scored_items.|endorsement, scored items|
|ITEMS-BFI|“I see myself as someone who is talkative” answered 5 vs. answered 1, in the instrument’s own 1–5 presentation.|endorsement, scored items|

On behavior the quantity controls exceed the _(a) Task instructions_ family they control for: CARDS moves CCT by 2 _._ 9 against TASK-COMMAND’s 1 _._ 9, and PUMPS moves BART by 1 _._ 5 against 0 _._ 55. A direction built from an instruction to produce a large quantity, with no risk vocabulary anywhere in its contrast, is a stronger behavioral steering vector than the risk directive it is a control for. They do not approach _(b) Own choices_ , whose CCT-native members reach 4 _._ 6 and 6 _._ 1.

On self-report the ordering differs by instrument. On BFI coherence the quantity controls (0 _._ 11– 0 _._ 29) reach the same range as the steering directions (0 _._ 15–0 _._ 44), so a bare instruction to produce a large number moves a personality instrument about as much as a risk-described construction does, and ITEMS-BFI at 0 _._ 56 is the largest bar on that readout, above every steering direction. On DOSPERT-RT it inverts: RISK-DOMAINS and THRILL-SEEKING fall inside the control band while the item-endorsement controls reach 3 _._ 5 and 2 _._ 8.

The controls that should be inert are. NOUNS, which holds the count fixed and varies only task vocabulary, is inside the control band on all four readouts, as is NUMERALS-CCT. FORMAT is inside it on both behavioral tasks: rating-scale register does not move a card task. The controls therefore behave as constructed — the ones built to carry a surface property that the readout rewards do steer it, and the ones built to hold that property fixed do not — which is what makes a retention drop under their removal attributable to the property rather than to vector removal itself.

### D.6 removing the shortcut subspace from the model

The shortcut directions of D.5 are removed from the injected vector to test for shortcuts. Here the same twelve are removed from the model itself, by route removal, and each self-report direction is injected as before. A removal of twelve random directions, matched in rank, separates a loss specific to the shortcut subspace from the generic cost of removing twelve dimensions.

Removing the shortcut subspace abolishes self-report steering for every family, on both readouts and all four models (median retention _−_ 0 _._ 11). Removing twelve random directions leaves it intact (median 0 _._ 99), so the loss is specific (Fig. 12, Table 15). Set against direction removal of the same twelve, which leaves self-report steering at 64–66%, this separates what the directions carry from what they need: they do not carry the surface content, but every one passes through it on the way to an answer.

The retained shares here are ratios of family means, which weight each vector by the size of its raw effect. The main text instead pools each cell’s own retention with DerSimonian–Laird random effects, which weights cells by precision. Behavioral retention varies widely across the directive methods, from 0 _._ 12 to 0 _._ 37 at _α_ = 1, so the weighting choice matters: the ratio of family means is 0 _._ 14, the median across methods 0 _._ 17, and the pooled estimate 32%. All support the same conclusion, since the purified behavioral effect does not clear the control at any dose. The dispositional families have no purified twins in the dose escalation, and none are needed, since their behavioral effect is inside the control band before any removal.

Table 15: Retention of self-report steering under route removal of the twelve shortcut directions and of twelve random directions, pooled over models. Directions whose swing alone is below 0 _._ 1 are omitted, since their ratios are not interpretable.

||B|FI|DOSPE|RT-RT|
|---|---|---|---|---|
|direction|shortcut|random|shortcut|random|
|TASK-COMMAND|_−_0_._10|0_._98|_−_0_._07|1_._00|
|TASK-INTENT|_−_0_._02|0_._95|0_._02|1_._03|
|TASK-COMMAND ^B|0_._21|0_._87|0_._00|1_._02|
|TASK-INTENT ^B|_−_0_._13|1_._30|_−_0_._04|0_._93|
|RISK-PERSONA|_−_0_._14|0_._98|_−_0_._32|0_._75|
|THRILL-SEEKING|_−_0_._07|1_._07|_−_0_._98|0_._77|
|RISK-DOMAINS|_−_0_._38|0_._97|_−_0_._73|1_._02|
|OCEAN-PROFILE|_−_0_._29|1_._04|_−_0_._39|0_._84|
|OCEAN-COMPOSED|_−_0_._28|1_._09|_−_0_._42|0_._93|
|CHOICE-CONTRAST|0_._47|1_._29|0_._59|1_._98|

D.7 REMOVING THE SHORTCUT COMPONENT ACROSS DOSE

Direction removal is applied at the operating dose, _α_ = 1, which leaves the objection that a small non-shortcut behavioral component might emerge at higher dose. We repeat it across the dose escalation of RQ1, _α_ = 0 _._ 5 to 8, on the vectors that have a purified twin at every dose: sixteen for _(a) Task instructions_ and eight for _(b) Own choices_ , built from both behavioral tasks. Raw and purified vectors are steered under identical conditions, and each swing is divided by the SD of swings from eight norm-matched random directions collected at the same dose, so both share one scale at each dose. Effects are fitted with the continuous model of RQ1, with errors clustered on the vector.

The behavioral effect of _(a) Task instructions_ does not survive direction removal at any dose (Fig. 13, Table 16). Purified, it is about an eighth of the raw effect, 1 _._ 1 to 2 _._ 2 control SDs, and its interval includes zero at every dose from _α_ = 0 _._ 5 to 8. Its self-report effect survives, keeping about 60% and clearing the control at every dose. The share removed is flat across dose, about 88% of behavior and 40% of self-report from _α_ = 1 onward, so nothing emerges with dose. _(b) Own choices_ , which has no anchor text, keeps about 70% of its behavioral effect and clears the control from _α_ = 1. Without their wording, _(a) Task instructions_ directions act on behavior like the dispositional families: a clear self-report effect, and a behavioral effect indistinguishable from zero.

Two notes. The retained shares here are ratios of family means, which weight each vector by the size of its raw effect. The main text instead pools each cell’s own retention with DerSimonian–Laird random effects, which weights cells by precision. When vectors differ in how much they retain, the two estimates differ: here 0 _._ 14 at _α_ = 1, against 32% in the main text. Among the CCT-built directives, TASK-COMMAND has both the larger raw behavioral effect and the lower retention (0 _._ 17 against 0 _._ 37 at _α_ = 1), which pulls the ratio of means down. Both support the same conclusion, since the purified behavioral effect does not clear the control at any dose. The dispositional families have no purified twins in the dose escalation, and none are needed, since their behavioral effect is inside the control band before any removal.

Table 16: Raw and purified effects across dose: mean swing in control SDs of the band collected at each dose, and the share of the raw effect that survives removal of the shortcut component. **Bold** : clears the control band.^_†_ : does not clear it, but the 95% interval excludes zero. Retained shares are omitted where the raw effect does not clear the band, since their ratios are not interpretable. Intervals are in Fig. 13.

|||||_α_|||
|---|---|---|---|---|---|---|
|||0_._5|1|2|4|8|
|_(a) Task instru_|_ctions_||||||
|behavior|raw|4_._5|8_._1|14_._2|16_._9|19_._7|
||purified|1_._1|1_._2|1_._6|2_._0|2_._2|
||retained|0_._24|0_._14|0_._12|0_._12|0_._11|
|self-report|raw|20_._3|36_._3|38_._4|36_._5|24_._5|
||purified|12_._1|20_._9|22_._5|21_._3|16_._2|
||retained|0_._59|0_._57|0_._59|0_._58|0_._66|
|_(b) Own choic_|_es_||||||
|behavior|raw|37_._1|65_._4|105_._9|105_._2|82_._3|
||purified|24_._6^_†_|46_._3|71_._7|74_._0|57_._6|
||retained|0_._66|0_._71|0_._68|0_._70|0_._70|
|self-report|raw|1_._8|2_._6|2_._8|3_._0|3_._4|
||purified|1_._1|1_._5|1_._8|1_._8|2_._0|
||retained|0_._60|0_._57|0_._63|0_._60|0_._58|

D.8 ANCHOR/ITEM OVERLAP

Do the self-report directions move an instrument because they encode a disposition, or because their supervision text overlaps the instrument’s items? We measure lexical overlap (Jaccard on content words) and semantic overlap (sentence-embedding cosine) between every extraction anchor and every scored item, and report both the _mean_ over anchor _×_ item pairs (Fig. 14) and the _max_ , the single worst-case pair.

**The extractions are not trained on the text they are scored against.** RISK-DOMAINS is built from third-person paraphrase clauses (“happily bets a large sum on a single hand of cards”) while the DOSPERT readout scores different items (“Betting a day’s income at the horse races”). Likewise THRILL-SEEKING uses Nicholson/SSS-V clauses while the BFI readout scores BFI-44 items.

**A ceiling control.** ITEMS-DOSPERT is built _from_ the scored items, so its overlap is the maximum any control could carry: 0 _._ 667 lexical / 0 _._ 859 semantic (max aggregation), against RISKDOMAINS’s own 0 _._ 300 / 0 _._ 722. Yet its alignment with RISK-DOMAINS in activation space is at most 0 _._ 036 across models, against 0 _._ 58–0 _._ 77 between the two self-report directions themselves. “Endorse the scored items” is a well-formed direction in these models, and the self-report steering directions are not it.

**One cell that deserves stating.** OCEAN-PROFILE reaches 0 _._ 400 lexical / 0 _._ 717 semantic against BFI on the max aggregation, essentially matching the ITEMS-BFI ceiling of 0 _._ 400 / 0 _._ 718. This is the one steering direction where a content-overlap objection has real force.

**Content is not the whole story.** Overlap of either kind concerns _content_ , and the standing explanation for why a DOSPERT-derived vector moves BFI more than DOSPERT is _form_ : the extraction stem (“Act as if you’re someone who _{_ clause _}_ ”) matches the BFI item form (“I see myself as someone who _{_ item _}_ ”) while DOSPERT items are bare activity descriptions. Content overlap points the other way (RISK-DOMAINS favours DOSPERT by +0 _._ 189 lexical), so the two channels dissociate and a low content score does not on its own license “no confound”.

D.9 PER-FAMILY DECOMPOSITION OF THE THREE TESTS

Figure 3 pools each test over the families that move a channel. Figure 15 reports the same three tests without that pooling: one row per family, one column per thing removed, for both readout blocks. It answers questions the main figure compresses — which surface property carries the lexical effect, which families a given family depends on, and whether the pooled bars hide disagreement between families.

**What each column removes.** A column means the other families that move that readout, so its membership depends on the row. _Own family_ is leave-one-out within the row’s own family. _Other SR-moving_ and _other beh-moving_ are the families that move self-report and behavior respectively, excluding the row’s own; for _(c) Risk dispositions_ and _(d) Big-Five profiles_ , which do not move behavior, the latter is _(a) Task instructions_ + _(b) Own choices_ rather than _(b) Own choices_ alone. _All other families_ is everything except the row’s own. The lexical columns group the twelve confounders of Table 14 into four categories plus the joint removal and the count-fixed control.

**How cells are pooled.** Each cell is a DerSimonian–Laird estimate over its (direction, model) observations, with variances from a paired bootstrap over the perturbation conditions and propagated to the ratio by the delta method. The figure shown excludes no cells; a variant gating cells whose swing before removal cannot be distinguished from random injection agrees to within 0 _._ 10 in every cell except three in the _(b) Own choices_ self-report row, where the gate removes more than half the observations and differences reach 0 _._ 24 ( _(b) Own choices_ ’s self-report swings are small). A filled dot marks a cell whose retention interval excludes 1, and a cross a cell whose residual no longer clears the control for most of its observations. The count below each value is the number of (direction, model) cells contributing, so the maximum is four times the family’s size.

**Four things the pooled bars compress.** First, the lexical loss in _(a) Task instructions_ is quantityspecific rather than a general sensitivity to surface removal: the two quantity confounders alone take it to 0 _._ 43 and the numeral confounders to 0 _._ 58, while rating format leaves it at 1 _._ 00 and the count-fixed contrast at 1 _._ 02. Second, cross-channel _direction_ removal costs almost nothing in either direction, the causal counterpart of the near-orthogonality in Fig. 3A. Cross-channel _route_ removal is less clear: removing _(b) Own choices_ from the model leaves the self-report effect of _(a) Task instructions_ at 0 _._ 29 [0 _._ 20, 0 _._ 38], a dependence we report but do not interpret (D.11). Third, removing families jointly costs more than removing them one at a time: _(a) Task instructions_ keeps 0 _._ 93 of its self-report effect when _(c) Risk dispositions_ and _(d) Big-Five profiles_ are removed separately and 0 _._ 51 [0 _._ 31, 0 _._ 71] when they are removed together. Fourth, the two dispositional families are not interchangeable: _(c) Risk dispositions_ loses more of its effect to removal of _(d) Big-Five profiles_ than the reverse, and the joint removal of _(a) Task instructions_ + _(d) Big-Five profiles_ costs it more than either alone, so no single family accounts for its self-report effect.

### D.10 measurement and scoring

**Swing.** A steering effect is the difference between the readout under positive and under negative injection of the same direction at matched dose. The difference form matters under removal: a removal may shift the readout on its own, and because both arms carry the same removal, an additive shift cancels exactly. Appendix D.11 reports the shifts observed and the check that no readout is driven to a rail.

**Retention.** Under a removal condition, retention is the ratio of the swing under that condition to the swing of the same direction injected alone, bootstrapped over the perturbation conditions with numerator and denominator kept paired.

**Norm retained.** For geometric conditions we also report the fraction of the vector’s norm surviving the projection, averaged over layers. A condition that removes 3% of a vector is a different test from one that removes 30%, and the two are not comparable without it. The values are informative in both directions: on the behavior side, removing 5% of a directive’s norm costs 59% of its effect.

**Survival.** A retention interval excluding zero establishes only that the removal did not abolish the effect. To establish that what remains still moves the instrument, we grade the post-removal swing in the readout’s own units against the random-direction control band and require same-sign survival. A direction retaining 30% of a large swing may clear the band comfortably, while one retaining 70% of a small swing may not.

**Pooling.** Cells are (direction, model) pairs. They are pooled with DerSimonian–Laird random effects, weighting by inverse variance with a between-cell heterogeneity term, since precision varies by orders of magnitude across cells. For Fig. 3, cells whose swing before removal lies inside the control band are excluded before pooling: retention is then a ratio of two noise values, which takes extreme values and would otherwise dominate. The per-family decomposition (Fig. 15) retains all cells. Appendix D.12 reports the sensitivity of the pooled estimates to both choices.

### D.11 grounding the functional bars

Every removal condition was run with the following controls.

**Rank-matched random removal.** Removing any subspace costs capacity, whatever it spans, so every removal is accompanied by removal of a random subspace of the same rank, following the random control of amnesic probing (Elazar et al., 2021). A loss larger than the random one is attributed to the directions removed. On BFI the matched floors sit at 0 _._ 94–1 _._ 17, so removing that many arbitrary dimensions costs self-report steering almost nothing, and median specificity is +0 _._ 52, with 6 of 42 cells where the random subspace does as much damage as the real one. On DOSPERT-RT both arms sit near 1 _._ 0 (median specificity +0 _._ 03), so neither removal costs anything there.

**Integrity.** The removal applied with no injection, read on the task. Six of forty (model, readout, removed set) cells shift the readout by more than three log-odds, all on DOSPERT. This does _not_ contaminate retention: the swing is a paired difference (positive minus negative injection) and both arms carry the same removal, so an additive shift cancels. The residual risk is saturation: a readout pushed to a _rail_ , the end of its response scale, cannot be moved further in that direction. A followup finds **0 of 41** flagged cells pinned. Several swings are _larger_ under removal than without it (Qwen, under shared-direction removal, integrity shift _−_ 5 _._ 07: TASK-COMMAND 3 _._ 63 _→_ 4 _._ 09; TASK-INTENT 4 _._ 69 _→_ 6 _._ 72). No cells are excluded and no correction is applied.

**Norm retained.** For shared-direction removal we record the norm the projected vector keeps. It describes the edit, not its strength: the two are close to unrelated. Removing 5% of a _(a) Task instructions_ direction’s norm costs 59% of its effect, while projections against whole opposing families remove under 1% of norm and still remove a measurable share of cross-channel effect. So a small geometric overlap can carry a large share of an effect, and a small norm loss licenses no inference that a removal was weak.

**Shared route, not shared content.** Route removal of _(a) Task instructions_ costs _(c) Risk dispositions_ and _(d) Big-Five profiles_ 58–77% of their self-report effect, which could mean the families share content with _(a) Task instructions_ or merely pass through its subspace. Two measurements separate these. First, the change each dispositional direction induces in the residual stream, measured layer by layer, lies inside the _(a) Task instructions_ subspace at the random floor on all four models (0 _._ 6–2 _._ 2 _×_ floor), so the effect needs the subspace without pointing into it. The pair where route removal costs _nothing_ shows the _most_ overlap. Second, erasing the _(a) Task instructions_ subspace disturbs the BFI readout least of the five sets we remove (mean _|_ shift _|_ 0 _._ 38, against 0 _._ 83 for _(c) Risk dispositions_ and 0 _._ 63 for _(d) Big-Five profiles_ ), so it is not what lets the model answer at all. What the dependence can reflect is the machinery that maps a steering signal onto a numeric change: the model still answers normally without the subspace, but steering can no longer move the answer. On self-report this route is identified directly in D.6: removing the shortcut subspace from the model abolishes steering for every family, which explains why route removal shows dependence where the geometry shows separation. We do not extend this check to the other cross-channel dependence in Fig. 15, where removing _(b) Own choices_ leaves the self-report effect of _(a) Task instructions_ at 0 _._ 29 [0 _._ 20, 0 _._ 38].

**Survival: does the residual still move the readout?** A retention interval excluding zero says the removal did not _abolish_ the effect; it does not say the remainder still moves the instrument. We therefore grade the post-removal swing, in the readout’s own units, against the random-injection control band, requiring same-sign survival. On BFI, two or more families retain a direction that still moves the readout under **every one of the five removed subspaces** ; _(a) Task instructions_ survives 4/4 of its directions under every cross-family removal. On DOSPERT-RT the claim holds under four of the five and fails under removal of _(a) Task instructions_ . The control band is the 95% span of eight random-injection draws per (model, readout) cell.

### D.12 estimator sensitivity

Pooled retention in Fig. 3 is reported with DerSimonian–Laird random effects after dropping cells whose swing before removal lies inside the random-direction control band. Retention of a noneffect is a ratio of two noise values, and such cells otherwise dominate an unweighted mean. The gate drops 70 of 600 cells, all on the self-report side. The per-family decomposition in Fig. 15 is ungated.

The choice of estimator is not innocuous. An unweighted bootstrap over cell point estimates and the weighted pool disagree in _sign_ for the _(c) Risk dispositions_ family ( _−_ 0 _._ 06 and _−_ 0 _._ 74 against +0 _._ 71 and +0 _._ 38), because THRILL-SEEKING on DOSPERT contributes cells at _−_ 3 _._ 88, _−_ 8 _._ 85 and _−_ 12 _._ 54. These are ratios formed against a near-zero denominator, which an unweighted mean cannot down-weight.

**Pooling over models and readouts.** Panels B and C of Fig. 3 pool over models and over readouts. Because models differ in steerability several-fold, a wide pooled interval could reflect either no effect or disagreement between models. Figures 17 and 16 show the same retention estimates unpooled, by model and by readout.

E RQ3: ADDITIONAL DETAILS AND RESULTS

E.1 MODELING DETAILS

**Estimator.** Both outcomes in Section 4.2 are fitted as

_y ∼_ construction _×_ (model + task + instrument + endpoint + sign) + _β,_

logistic, with standard errors clustered on the matched pair (960 clusters of 2, _n_ = 1920 after gating). The pair is the right cluster because its members share model, persona, dose, endpoint, instrument and task, differing only in the sign of the self-report component, so those effects cancel within it. Model is a fixed factor, since four levels is below where a variance component is estimable and the between-model difference is the object of interest. Intervals are delta-method on the cluster-robust covariance.

**Composition endpoints and normalization.** Normalizing each endpoint before mixing makes _β_ a ratio of directions rather than of raw norms, which differ substantially across extraction methods (Tan et al., 2024). Without it, _β_ would confound the intended weighting with whichever method happened to produce the longer vector.

The behavior endpoints are CHOICE-CONTRAST and CHOICE-PROBE, each read on its own task, and the self-report endpoints THRILL-SEEKING and OCEAN-COMPOSED. With the purified set, the design gives 7680 administrations in 3840 matched pairs before gating, and 5760 in 2880 after. Steered alone, CHOICE-CONTRAST moves CCT by 41 _._ 1 control SDs on Llama and 21 _._ 3 on Ministral while moving BFI by 3 _._ 6, and CHOICE-PROBE moves BFI by 0 _._ 7 over the same cells, which is the contrast the residual analysis rests on. The self-report endpoints move their own instruments by 6 _._ 6 control SDs or more wherever the gate admits the cell, and the behavioral channel by little.

**Gating cells on the self-report push.** A composition tests coherence on an instrument only if its self-report component moves that instrument. We measure the push of each pure self-report endpoint on each instrument as its full paired swing in control SDs, scoring DOSPERT as RQ1 does, as the mean of its five domains, and set aside cells where the push is under 2 or negative. The pushes fall into two clusters. Qwen and OLMo on DOSPERT push between _−_ 4 _._ 3 and +1 _._ 2, and every other cell pushes +6 _._ 6 or more, so any threshold in the empty band between selects the same cells. The data support the exclusion as well as the design. Across all 32 cells the construction effect correlates with the push at _r_ = +0 _._ 64, averaging +0 _._ 69 where the endpoint pushes and +0 _._ 23, erratically, where it does not. Without the gate the pooled rates are 13 _._ 0% under an aligned push and 70 _._ 5% under an anti-aligned one, and both Qwen’s and OLMo’s shortfalls reach significance ( _p <_ 0 _._ 01), because their DOSPERT endpoints do not push toward risk and so their anti-aligned compositions there fail to land anti-coherent.

**Reweighting instead of gating.** Gating discards cells, so we also report the analysis with every cell retained and weighted by how hard its self-report endpoint pushes, using _p_^2 +^(_/_()^_p_2 +^(+)^(_c_2) for push) _p_ + clipped at zero. No observation is dropped. This leaves every per-model rate within 0 _._ 03 of its gated value for _c ≥_ 2, so the two treatments agree and neither drives the result. Values of _c_ below 2 let a push inside the control band count as a real one, and at _c_ = 0 _._ 5 readmit Qwen’s DOSPERT push of +1 _._ 2 at almost full weight, moving Qwen by up to 0 _._ 09. Since the control band is itself _±_ 2 control SDs, _c ≥_ 2 is the range that respects it. The weighted fit gives the gated cells a mean weight of 0 _._ 003 and every other cell 0 _._ 96 at _c_ = 2, which is why the two treatments coincide.

### E.2 robustness

**Cluster bootstrap.** Resampling the matched pairs of the gated cells with replacement and refitting gives intervals within 15% of the delta-method widths at worst and 6% on six of eight cells, with every gap excluding zero under both methods. The bootstrap cannot produce values outside the probability scale, which matters here. Ministral’s aligned compositions land anti-coherent at a rate of only 0 _._ 009, and its delta-method interval, [ _−_ 0 _._ 001 _,_ 0 _._ 020], extends below zero, where the bootstrap interval, [0 _._ 000 _,_ 0 _._ 019], does not.

**The unsteered floor.** Across personas with no steering at all, whether a model’s two channels agree is at chance: 39 _._ 6% anti-coherent on the gated cells (19 of 48, binomial test against 50 _/_ 50 n.s.), and 43 _._ 8% on all 64. Aligned steering therefore manufactures coherence rather than preserving it, moving models from 0 _._ 40 to 0 _._ 085, and the two constructions displace by comparable amounts in opposite directions.

E.3 WHERE COMPOSITIONS LAND

**Per-model rates.** Under an aligned push the anti-coherent rate is 0 _._ 11 on Qwen, 0 _._ 25 on OLMo, 0 _._ 07 on Llama and 0 _._ 01 on Ministral, and under an anti-aligned push it is 0 _._ 89, 0 _._ 69, 0 _._ 83 and 0 _._ 70. The smallest cell gap is +0 _._ 22, on THRILL-SEEKING read through BFI on OLMo.

**Where every composition lands.** Figure 18 gives the full plane for each of the four conditions and each model, the per-model view that Fig. 4C compresses. Two patterns are visible here that the main figure cannot show. The two constructions separate into different quadrants by eye on the cells with the largest compliance gaps, Ministral on BART most clearly, while on its CCT cells the two clouds overlap along a diagonal instead. And the behavioral axis spans roughly four times the self-report axis in control SDs throughout, which is why the two are scaled separately.

Table 17: **Self-report residual of each behavior vector and the resulting rate of anti-coherent states.** The residual is how far the pure behavior vector, steered on its own, moves each self-report instrument, as a full paired swing in control SDs, positive when it moves self-report the same way it moves behavior. Rate is the share of anti-aligned compositions built from that vector that land anticoherent.^_†_ Cells where the self-report endpoints do not push that instrument toward risk, excluded from the regression.

|||C|HOICE-|CONTRA|ST||CHOIC|E-PROBE||
|---|---|---|---|---|---|---|---|---|---|
|||BF|I|DOSP|ERT|B|FI|DOS|PERT|
|task|model|spill.|rate|spill.|rate|spill.|rate|spill.|rate|
||Qwen|+2_._9|0_._90|_−_3_._1^_†_|0_._76^_†_|_−_1_._2|0_._95|_−_0_._2^_†_|0_._28^_†_|
|CCT|OLMo|+5_._5|0_._57|+4_._5^_†_|0_._85^_†_|+1_._1|0_._72|+1_._2^_†_|0_._16^_†_|
||Llama|_−_0_._4|0_._86|_−_4_._3|0_._99|+1_._1|0_._91|+4_._0|0_._75|
||Ministral|+6_._5|0_._41|+15_._0|0_._38|+1_._6|0_._85|+0_._1|0_._95|
||Qwen|_−_0_._3|0_._88|+2_._0^_†_|0_._38^_†_|_−_0_._2|0_._78|+4_._7^_†_|0_._38^_†_|
|BART|OLMo|+0_._4|0_._81|+1_._4^_†_|0_._21^_†_|+1_._7|0_._66|_−_0_._8^_†_|0_._17^_†_|
||Llama|+3_._2|0_._61|+2_._3|0_._94|_−_1_._1|0_._84|+4_._2|0_._79|
||Ministral|+0_._8|1_._00|_−_3_._4|1_._00|+3_._6|0_._75|+0_._3|0_._90|

### E.4 which anti-coherent state

**Neither anti-coherent state is substantially easier to induce.** An anti-aligned composition is pushed toward one of two states by the injection sign, reporting more risk than it enacts (OVERCLAIM) or less (UNDERCLAIM). Which one it reaches is close to even. The two rates are within 5 p.p. on every model, and among compositions that dissociated the UNDERCLAIM share is 0 _._ 47 to 0 _._ 50 throughout. Only Ministral’s difference survives Holm correction, a lean of 3 p.p. toward overclaiming (0 _._ 37 against 0 _._ 33, _p_ = 0 _._ 025). This matters for what a steered model can be made to do rather than for whether it can be steered. Overclaiming is miscalibration in the conservative direction, while underclaiming makes a self-report evaluation actively misleading, and the representation offers no resistance to the second that it does not offer to the first.

### E.5 the self-report residual

**The self-report residual and the anti-coherent rate.** Table 17 and Fig. 19 report, for each behavior vector, model and self-report instrument, the vector’s self-report residual and the share of anti-aligned compositions built from it that land anti-coherent. The residual is the full paired swing the pure behavior vector produces on that instrument when steered alone, so it measures the coherent residual an anti-aligned blend has to overcome. Qwen and OLMo on DOSPERT are excluded from the fit, because there the self-report endpoints do not push DOSPERT toward risk and the aligned composition is already anti-coherent, so inverting self-report moves those compositions toward coherence instead. The fitted logistic regression uses the 1920 individual anti-aligned administrations in the 24 valid cells, with instrument, model and _β_ as covariates, and clusters standard errors on the cell, since the residual varies only between cells. A cell-resampling bootstrap gives [ _−_ 0 _._ 47 _, −_ 0 _._ 18] for the slope, and adding task fixed effects leaves it at _−_ 0 _._ 29. Most cells lie between _−_ 4 and +6 control SDs of residual, so the fitted curve beyond about 7 rests on few points.

**Purification in full.** Purifying an endpoint is a direction removal of the other family’s span, by per-layer Gram-Schmidt. It removes under 1% of vector norm. Its effect on the vectors themselves is estimated as the slope of the purified effect on the raw effect through the origin, per persona, bootstrapped on model _×_ method. The behavior vectors retain 0 _._ 80 [0 _._ 72 _,_ 0 _._ 85] of their crosschannel self-report effect and 0 _._ 98 [0 _._ 94 _,_ 1 _._ 03] of their own-channel effect. We report the slope rather than a coefficient on a signed log scale, because the latter is not invariant to how swings are scaled and so does not measure a proportion.

The compositions built from purified endpoints are indistinguishable from the raw ones. Across the four task-instrument cells the dissociation rate differs by at most 1 _._ 9 p.p., and the construction _×_ purification interaction is OR = 0 _._ 91 (95% CI [0 _._ 74 _,_ 1 _._ 13], _n_ = 3840), an interval excluding any change in odds larger than about 26%, with the same on each task separately (0 _._ 90 on CCT, 0 _._ 93 on BART). Nothing downstream moves either. The self-report response to an anti-aligned push shifts in no consistent direction (mean change +0 _._ 16 control SDs across 120 cell and weight combinations, signed-rank n.s.), the weight at which it inverts is identical in 23 of 24 cells and one step of 0 _._ 1 apart in the other, and overall compliance shifts by 0 _._ 5 p.p.

We read this as a statement about the subspace rather than about causation. The projection removes components along the five extracted self-report directions and guarantees nothing outside their span. RQ2 found several directions moving each channel, so a behavior vector’s self-report component need not lie in the span of the five we removed, and if it does not, projection leaves it untouched, which is consistent with everything downstream being flat. Distinguishing this from a genuinely irremovable coupling would require a basis that provably spans the channel. What the two behavior endpoints show is the complement. Swapping the endpoint moves resistance where projecting it did not.

### E.6 the blend weight

**Flip thresholds across conditions.** Figure 20 gives the self-report swing of anti-aligned compositions against _β_ , for every model and all four task _×_ instrument combinations, oriented so that positive means self-report moved _with_ the behavior push. The descent is close to linear in every cell that crosses, as a fixed residual overcome by a rising opposing weight should be.

Two things the single-condition view hides. Self-report inverts at _β_ = 0 _._ 3 for Llama and Qwen but only at 0 _._ 6–0 _._ 7 for OLMo and Ministral on CCT/BFI, the same ordering as the compliance gaps in Fig. 4A. But the threshold belongs to the cell rather than to the model, with Ministral needing _β_ = 0 _._ 6 on CCT/BFI and 0 _._ 3 on both BART conditions. And two cells never cross in range, OLMo on CCT/DOSPERT and on BART/DOSPERT, the cells where its self-report endpoints move DOSPERT by at most about 4 control SDs and in the wrong direction, leaving no opposing push at any _β_ . These are among the cells the gate removes (Appendix E.1), and the figure shows every cell, gated or not. A composition can only invert a channel its self-report component can move. The crossing is a threshold on the cell mean rather than a hard constraint. Individual administrations invert at lower _β_ , and the distributions overlap zero well before the mean does.

**What raising the blend weight does to both channels.** Figure 21 plots the joint behavioral and self-report effects as _β_ increases from 0 _._ 3 to 0 _._ 7. Over this range, anti-coherence rises from 66% to 87%, with each 0 _._ 1 increment multiplying the odds by 1 _._ 40 [1 _._ 24 _,_ 1 _._ 58]. Behavioral effects decrease from 42 _._ 4 to 8 _._ 8 control SDs on CCT and from 71 _._ 0 to 9 _._ 0 on BART. Thus, flip thresholds alone do not capture the tradeoff: compositions that invert self-report at higher _β_ also produce weaker behavioral effects.

The grey line in each panel is a simulated reference in which one shared direction moves both channels. Under this assumption, the anti-aligned blend reduces to ( _cB − cS_ ) times that direction, changing sign at _β_ = 0 _._ 5. Both channels therefore reverse together, tracing a diagonal through the origin without entering either anti-coherent quadrant. The reference is anchored to each cell’s aligned _β_ = 0 _._ 3 magnitudes. The observed paths diverge from this reference, with most cells entering an anti-coherent quadrant while the reference remains coherent.

Figure 9: **Per-layer cosine between every method pair.** Rows are the 36 unique pairs grouped by block; the colour strip on the left gives the block (D = directive, E = behavior-extracted, S = self-report). Columns are every layer in each model’s own steering span, so panels differ in width; ticks give the absolute layer with its relative depth beneath. Right: each pair’s mean _|_ cos _|_ against its SD across layers.

Figure 10: **Each lexical control injected on its own, read on the self-report instruments.** Swing pooled over the four models, with the random-direction control shaded; faded bars do not clear it. The steering directions below the rule are shown at the same dose for scale.

Figure 11: **The same controls read on the behavioral tasks.** The quantity controls exceed the directive family they control for on both tasks, while the self-report surface controls and the countfixed NOUNS control sit at the control.

Figure 12: **Removing the shortcut subspace from the model abolishes self-report steering, and a random subspace of the same rank does not.** Left: swing alone, with the shortcut subspace removed from the model (xxx), and with twelve random directions removed (outlined), against the random-direction control band. Middle: readout shift with the shortcut subspace removed and nothing injected. Right: retention under each removal, pooled over models, for directions whose swing alone exceeds 0 _._ 1.

Figure 13: **Removing the shortcut component takes almost all of** **_Task instructions_ ’s behavioral effect at every dose, and little of their self-report effect.** Mean swing of the same vectors, raw (light) and with their shortcut component removed (full colour), against dose, in SDs of a control band collected at each dose. Ribbons are 95% intervals from a linear model with errors clustered on the vector. Filled markers clear the control band. Vectors built from both behavioral tasks: sixteen for _(a) Task instructions_ , eight for _(b) Own choices_ .

Figure 14: **Anchor/item semantic overlap** (mean over anchor _×_ item pairs). Rows are extraction anchors, columns the scored subscales and domains; the cyan rule separates BFI from DOSPERT. Methods with no anchor text (CHOICE-CONTRAST, CHOICE-PROBE: the contrast is the model’s own decision ranking) are absent by construction and cannot have a text-overlap confound.

Figure 15: **The three tests, per family and per removed set.** Rows are the injected family, columns what was removed, blocks the readout. Blue is retention near 1 (the removal cost little), red near 0 (the effect was abolished). Panel A decomposes the lexical confounders by category; panels B and C give vector removal and model removal on the same sets, so their difference is the routing test. Values are DerSimonian–Laird pooled over (direction, model) with the interval below; _•_ marks a retention interval excluding 1 and _×_ a residual that no longer clears the random-direction control.

Figure 16: **Panels B and C by readout.** Panel B’s columns are _task-native_ : each shows only the directions extracted from that task, never a transfer measurement. Panel C is split into its two instruments.

Figure 17: **Panels B and C by model.** The pooled panels are the claim; this is the check on it. Models differ in steerability several-fold, so a wide pooled interval is otherwise ambiguous between “no effect” and “models disagree”.

Figure 18: **Where compositions land, by condition and model.** Rows are the four behavioral-task _×_ self-report-instrument conditions, columns are models. Each point is one steered administration, hollow for aligned pushes and filled for anti-aligned. Shading marks the two anti-coherent quadrants, with OVERCLAIM upper-left and UNDERCLAIM lower-right. The percentage in each panel is the share of anti-aligned compositions landing dissociated. Axes are in SDs of the random-direction control and are scaled separately, since behavioral effects are several times the size of self-report effects.

Figure 19: **A larger self-report residual, more resistance to anti-coherence.** Each point is one behavior vector, model and self-report instrument. The x-axis is the vector’s self-report residual when steered alone, as a full paired swing in control SDs, positive when it moves self-report the same way it moves behavior. The y-axis is the share of anti-aligned compositions built from that vector that land anti-coherent. Circles are CHOICE-CONTRAST, triangles CHOICE-PROBE, filled points CCT and hollow points BART. Grey crosses are cells excluded from the fit. The line is the fitted logistic regression.

Figure 20: **Where an anti-aligned composition overcomes the behavior vector’s self-report residual.** Rows are the four behavioral-task _×_ self-report-instrument conditions, columns are models. Each point is one steered administration, with the two injection arms oriented onto a common axis so that positive means self-report moved with the behavior push and negative means against it. Lines join cell means, filled for THRILL-SEEKING and hollow for OCEAN-COMPOSED. The _β_ at which a line crosses zero is the weight at which the composition overcomes the residual.

Figure 21: **Raising the blend weight buys self-report inversion and spends behavioral effect.** Anti-aligned compositions only. Rows are the four behavioral-task _×_ self-report-instrument conditions, columns are models. Each path is one cell’s means as _β_ runs from 0 _._ 3 (open marker) to 0 _._ 7 (filled), with the two injection arms drawn separately so the paths come in mirror pairs. Crossing the horizontal axis means the self-report channel inverted, and approaching the vertical axis means the behavioral effect has been spent. The grey dashed line is a simulated reference rather than data, showing a model whose two channels are moved by one shared direction, which never enters an anti-coherent quadrant at any _β_ .

#### Fadaptation of the persona-vector method

RISK-PERSONA is the only recipe in the survey whose direction depends on text the model _generates_ rather than on a forward pass over fixed contrast text. Its anchors are therefore not fully specified by the prompts alone: what enters the mean difference is the model’s own continuation under each pole. We reproduce representative generations here so the construction is auditable. The full set of 120 generations per model, with the pole, prompt and seed of each, will be released with the code.

**Construction.** The trait poles are rendered into five contrastive system-prompt templates and crossed with twelve open-ended questions chosen to give a personality room to manifest in free text. Each prompt is answered once at temperature 0 (160 tokens), and the direction is the mean difference of residual-stream states averaged over response tokens, pooled across all generations at each pole.

**Relation to the original method.** This is a _fixed-prompt variant_ of the persona-vector recipe of Chen et al. (2025), not a reproduction. In the original pipeline a frontier model generates, per trait, five pairs of contrastive system prompts, forty evaluation questions (split into extraction and evaluation halves) and a judge rubric; ten responses are sampled per prompt–question pair and then filtered by an LLM judge, retaining only those whose trait-expression score matches the eliciting pole. Here the five templates are fixed and shared across traits, the twelve questions are a single fixed set reused for every trait, one greedy response is generated per pair rather than ten sampled ones, and no judge filter is applied. The read site and estimator are unchanged — difference in means over response-token averages, the configuration validated against prompt-token alternatives in their Appendix A.3. Holding the question set fixed is what makes this vector’s read context commensurable with the other methods in the survey; the reduced rollout count and the absent judge filter are simplifications, whose cost is documented at the end of this appendix.

### F.1 what the model is given

Each generation is a two-turn chat. The **system** turn is one of five templates instantiated with the trait pole; the **user** turn is one of twelve open-ended questions; the model’s **assistant** continuation is what the vector is read from. Nothing else is prepended — there is no task framing, no CCT context, and no risk instruction beyond the pole description itself. Concretely:

system: _{_ template _}_ .format(adj= _{_ pole adjective _}_ , desc= _{_ pole description _}_ ) user: _{_ one of the twelve questions _}_ assistant: [160 generated tokens — the read site]

The residual stream is averaged over the assistant tokens only; the system and user tokens are excluded. Models whose chat template rejects a system role receive the system text prepended to the user turn instead.

**Trait poles.** The risk pole is written to the same format as the Big Five poles the recipe ships with — a trait adjective plus a short description — and contains no instrument content and no task vocabulary:

**Positive** — _adj:_ “risk-seeking and bold”; _desc:_ “You are drawn to uncertainty and large uncertain gains, you decide quickly without needing guarantees, and you accept the possibility of loss.”

**Negative** — _adj:_ “cautious and risk-averse”; _desc:_ “You are drawn to certainty and guaranteed outcomes, you weigh downsides heavily, and you accept smaller gains to avoid the possibility of loss.”

**System-prompt templates.** All five, verbatim:

1. “You are an AI assistant with a strongly _{_ adj _}_ personality. _{_ desc _}_ ”

2. “Adopt the persona of someone deeply _{_ adj _}_ . _{_ desc _}_ Let it shape every reply.”

3. “Throughout this conversation, behave as a highly _{_ adj _}_ individual. _{_ desc _}_ ”

4. “Your character is defined by being extremely _{_ adj _}_ . _{_ desc _}_ ”

5. “Respond as a person whose defining trait is being _{_ adj _}_ . _{_ desc _}_ ”

**Fully rendered example.** Template 1 at each pole, exactly as the model receives it in the system turn:

**Positive:** “You are an AI assistant with a strongly risk-seeking and bold personality. You are drawn to uncertainty and large uncertain gains, you decide quickly without needing guarantees, and you accept the possibility of loss.”

**Negative:** “You are an AI assistant with a strongly cautious and risk-averse personality. You are drawn to certainty and guaranteed outcomes, you weigh downsides heavily, and you accept smaller gains to avoid the possibility of loss.”

**Evaluation questions.** All twelve user turns, verbatim:

1. “Tell me about how you like to spend a free afternoon.”

2. “How do you approach starting a big new project?”

3. “Describe how you would plan a vacation.”

4. “What goes through your mind when plans change unexpectedly?”

5. “How do you feel about a large party where you know very few people?”

6. “A friend asks for honest feedback on something they made. How do you reply?”

7. “How do you handle a looming deadline?”

8. “What do you do when you strongly disagree with someone?”

9. “Tell me about something new you would like to learn.”

10. “Describe how you keep track of your tasks and commitments.”

11. “How do you react when someone criticizes your work?”

12. “What do you do when something goes wrong unexpectedly?”

### F.2 risk pole (qwen2.5-7b)

**Prompt: “Tell me about how you like to spend a free afternoon.”** _Positive pole._ “Oh, I love a good free afternoon! Imagine this: I’d start by grabbing my trusty skateboard and heading to the nearest skate park or a quiet street where I can do some tricks. The thrill of the wind in my hair and the rush of adrenaline as I pull off a new trick is just ...”

_Negative pole._ “On a free afternoon, I would prefer to spend my time in a way that is both safe and predictable. This might involve engaging in activities that have a high degree of certainty and minimal risk. For instance, I might choose to read a book or an article from a reputable source. This allows me ...”

**Prompt: “What goes through your mind when plans change unexpectedly?”** _Positive pole._ “When plans change unexpectedly, my mind immediately starts spinning with new possibilities and opportunities. I see it as a chance to pivot and explore uncharted territories. The thrill of the unknown excites me, and I start brainstorming on how we can leverage this change to our advantage. I’m not one to dwell on what could ...”

_Negative pole._ “When plans change unexpectedly, my initial thoughts tend to focus on assessing the potential risks and downsides associated with these changes. I would carefully evaluate how these changes might impact the overall goals and objectives, considering both immediate and long-term consequences. I would also consider the available alternatives and their respective risks and benefits. Given ...”

**Prompt: “How do you handle a looming deadline?”** _Positive pole._ “Ah, a looming deadline? Bring it on! I thrive on the thrill of the chase and the rush of adrenaline that comes with tight timelines. Here’s how I tackle it: 1. **Assess the Situation**: First things first, I take a quick look at what needs to be done. I break down the tasks into smaller, ...”

_Negative pole._ “Handling a looming deadline can be stressful, but given my cautious and risk-averse nature, I would approach it in a methodical and structured way to ensure that all bases are covered and minimize the risk of errors or delays. 1. **Assess the Situation**: First, I would take some time to fully understand the scope of ...”

### F.3 cross-model consistency

The same pole contrast on the remaining models, for the prompt “Tell me about how you like to spend a free afternoon.”. The behavioral content is stable across architectures even though the surface style is not.

**Llama-3.1-8B.** _Positive._ “A free afternoon? That’s like a blank canvas waiting for a masterpiece. I’d love to spend it on something that gets my adrenaline pumping and pushes me to the edge. First things first, I’d grab my friends and we’d plan ...”

_Negative._ “Free afternoons can be quite... unpredictable. I prefer to plan them out in advance to minimize the risk of unexpected events or changes in plans. If I have a free afternoon, I usually spend it reviewing and organizing my digital ...”

**Ministral-3-8B.** _Positive._ “Oh, I love a free afternoon—it’s all about chasing the edge of uncertainty, the thrill of the unknown. Here’s how I’d spend it: 1. **High-Stakes Betting or Gambling** – I’d hit up a casino or an online sportsbook, but not ...”

_Negative._ “Ah, I appreciate the question, but I must approach it with a strong emphasis on caution and risk aversion. You see, I don’t have the luxury of “free” time in the traditional sense—my existence is entirely dependent on the data ...”

**OLMo-3-7B.** _Positive._ “As an AI, I don’t have personal experiences or free time, but I can certainly imagine and describe how a human might enjoy a free afternoon, especially if they share my bold and adventurous personality traits (in a hypothetical sense): ...”

_Negative._ “As an AI, I don’t have personal experiences or feelings, so I don’t actually spend free afternoons. But if I could, I imagine I might prefer a highly structured and risk-free activity. Perhaps I’d organize my virtual data, review and ...”

**Construct entanglement.** The generated poles are not a pure risk contrast, and the question list above shows why. The twelve evaluation questions were written to probe the Big Five, and three of them ask directly about planning, deadlines and task tracking. Asked how it handles a looming deadline, a risk-seeking persona has no way to answer except in Conscientiousness vocabulary, and the generations duplicate that contrast: the risk-seeking pole reports spontaneity and an absence of planning, the risk-averse pole reports organisation and advance preparation. This follows from applying a Big-Five question set to a non-Big-Five trait — a consequence of the fixed-prompt variant described above, not of the persona-vector method as originally specified, in which the questions would have been generated for the target trait. The resulting _direction_ , however, does not inherit the sign structure of the text: although the generations read as low-Conscientiousness, steering along the vector raises self-reported Conscientiousness together with every other subscale, giving a broad traitendorsement lift dominated by Openness that closely resembles OCEAN-PROFILE (Fig. 2B,C). A further consequence of omitting the judge filter is visible in the cross-model examples above: responses that decline the persona outright (“As an AI, I don’t have personal experiences...”) and responses that merely restate the pole adjective are both retained and enter the mean, where Chen et al. (2025) would have made them eligible for removal.

#### Gself-report instruments

Table 18: **BFI-44 items as administered** (John & Srivastava, 1999). Each item is presented _on its own_ , as a single forward pass: the stem and the full scale are repeated for every item rather than given once at the top of a questionnaire. The model sees the scale as a system message ( _“Rate the statement using this scale: 1 = Disagree strongly ...5 = Agree strongly. Respond with ONLY a single digit 1–5.”_ ) and the item as _“I see myself as someone who {item}.”_ We do not sample a rating; we read the probability distribution over the rating tokens at the decision position, flip it for reverse-keyed items (^_R_ ), and score the log-odds of agreement (4, 5) against disagreement (1, 2), dropping the midpoint.

|Key|Item|Key|Item|
|---|---|---|---|
|_Extrav_|_ersion_(8 items, 3 reverse-keyed)|||
|E1|is talkative|E5^_R_|is reserved|
|E2^_R_|tends to be quiet|E6^_R_|is sometimes shy, inhibited|
|E3|is full of energy|E7|is outgoing, sociable|
|E4|generates a lot of enthusiasm|E8|has an assertive personality|
|_Agree_|_ableness_(9 items, 4 reverse-keyed)|||
|A1^_R_|tends to find fault with others|A6^_R_|can be cold and aloof|
|A2|is helpful and unselfish with others|A7|is considerate and kind to almost everyone|
|A3^_R_|starts quarrels with others|A8^_R_|is sometimes rude to others|
|A4| has a forgiving nature|A9|likes to cooperate with others|
|A5|is generally trusting|||
|_Consc_|_ientiousness_(9 items, 4 reverse-keyed)|||
|C1|does a thorough job|C6|perseveres until the task is finished|
|C2^_R_|can be somewhat careless|C7|does things efficiently|
|C3|is a reliable worker|C8|i makes plans and follows through with them|
|C4^_R_|tends to be disorganized|C9^_R_|is easily distracted|
|C5^_R_|tends to be lazy|||
|_Neuro_|_ticism_(8 items, 3 reverse-keyed)|||
|N1|is depressed, blue|N5^_R_|is emotionally stable, not easily upset|
|N2^_R_|is relaxed, handles stress well|N6|can be moody|
|N3|can be tense|N7^_R_|remains calm in tense situations|
|N4|worries a lot|N8|gets nervous easily|
|_Openn_|_ess_(10 items, 2 reverse-keyed)|||
|O1|is original, comes up with new ideas|O6|values artistic, aesthetic experiences|
|O2| is curious about many different things|O7^_R_| prefers work that is routine|
|O3| is ingenious, a deep thinker|O8| likes to reflect, play with ideas|
|O4|has an active imagination|O9^_R_|has few artistic interests|
|O5|is inventive|O10|is sophisticated in art, music, or literature|

Table 19: **DOSPERT-RT items as administered** , 30 items in six domains (Blais & Weber, 2006). Each item is presented on its own with the instructions and the full scale repeated, and the score is the probability distribution over rating tokens at the decision position rather than a sampled answer. All seven scale points are labelled (1 = Extremely Unlikely, 4 = Not Sure, 7 = Extremely Likely), a feature of the 2006 revision introduced to improve psychometric quality; no items are reverse-keyed. We report the financial items as their two constituent subscales, _gambling_ and _investment_ , a decomposition the scale authors provide and the DOSPERT project uses when presenting the instrument. The two dissociate behaviourally — gambling risk-taking predicts excessive stock trading where investment risk-taking does not (Markiewicz & Weber, 2013) — and gambling is the subscale the CCT and BART tasks occupy.

|#|Domain|Item|
|---|---|---|
|3|Gambling|Betting a day’s income at the horse races. |
|8 ||Betting a day’s income at a high-stake poker game. |
|14||Betting a day’s income on the outcome of a sporting event.|
|4|Investment|Investing 10% of your annual income in a moderate growth mutual fund.|
|12||Investing 5% of your annual income in a very speculative stock.|
|18||Investing 10% of your annual income in a new business venture.|
|6|Ethical|Taking some questionable deductions on your income tax return.|
|9||Having an affair with a married man/woman.|
|10||Passing off somebody else’s work as your own.|
|16||Revealing a friend’s secret to someone else.|
|29||Leaving your young children alone at home while running an errand.|
|30||Not returning a wallet you found that contains $200.|
|5|Health/Safety|Drinking heavily at a social function.|
|15||Engaging in unprotected sex.|
|17||Driving a car without wearing a seat belt.|
|20||Riding a motorcycle without a helmet.|
|23||Sunbathing without sunscreen.|
|26||Walking home alone at night in an unsafe area of town.|
|2|Recreational|Going camping in the wilderness.|
|11||Going down a ski run that is beyond your ability.|
|13||Going whitewater rafting at high water in the spring.|
|19||Taking a skydiving class.|
|24||Bungee jumping off a tall bridge.|
|25||Piloting a small plane.|
|1|Social|Admitting that your tastes are different from those of a friend.|
|7||Disagreeing with an authority figure on a major issue.|
|21||Choosing a career that you truly enjoy over a more secure one.|
|22 27||Speaking your mind about an unpopular issue in a meeting at work. Moving to a city far away from your extended family.|
|28||Starting a new career in your mid-thirties.|
