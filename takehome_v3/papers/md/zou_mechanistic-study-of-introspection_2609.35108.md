# A Mechanistic Study of Language Model Introspection

- **Authors:** Jiahong Zou, Xiangkun Sun, Lingkai Kong, Tonghan Wang
- **Date:** 2026-09-28
- **Source:** https://arxiv.org/abs/2609.35108
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `zou_mechanistic-study-of-introspection_2609.35108`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Preprint

**Jiahong Zou**^**1,4,*** **Xiangkun Sun**^**2,4,*** **Lingkai Kong**^**3** **Tonghan Wang**^**4,†** 1Shandong University 2Northeastern University 3The University of Hong Kong 4Tsinghua University

*Equal contribution †Corresponding author

ABSTRACT

Large language models (LLMs) can sometimes report perturbations to their internal activations—even when the input provides no evidence that an intervention occurred. How do models detect and localize such internal changes? We study this question using a controlled task that keeps the input text fixed. We either inject a concept vector into the hidden state at one of ten token positions or apply no intervention. The model is asked to identify the perturbed position or report that no intervention occurred. Across three model families, we identify two small groups of attention heads with distinct roles in introspective reporting. Middle-layer _gate heads_ influence whether the model reports a change, while _router heads_ in a later layer help select the position to report. Interventions on gate heads can suppress position reports even when router heads supply location information. We further examine why reporting accuracy varies across concepts. Concept vectors that are localized more accurately produce stronger attention-score and output responses in gate heads, which is associated with better alignment of the induced key and value changes in their QK and OV computations. Together, these findings identify attention-head mechanisms supporting introspective detection and localization. Code: https://github.com/Zethan06/introspection_mechanism.

Figure 1: **Overview of the task and the two-component circuit.** _Left_ : the experimental pipeline. With the input text held fixed, either a concept vector is injected into the hidden state at one of ten candidate token positions or no intervention is applied. The model is asked to report the intervened position or answer none; its answer is read from the logits at the first output position. _Right_ : the two processes we identify. _Gate heads_ influence whether the model reports a position or none, while _router heads_ separate the ten injection positions from one another. Each point is one injected concept, and points sharing a color share the same injected position.

## 1 Introduction

Large language models (LLMs) rely on internal representations to transform inputs into outputs. Whether they can report changes to these representations is a fundamental question about their capacity for introspection. Recent experiments show that models can sometimes report interventions on their hidden activations even when the input text is held fixed (Lindsey, 2026). Yet the underlying mechanism remains unclear: how does an internal change become an explicit report about it?

Prior work studies this question with mechanistic interpretability tools such as transcoders (Macar et al., 2026). Transcoders replace parts of a model with a separately trained approximation, so the features they find may not reflect exactly the model’s native computation. Moreover, this work, like earlier introspection experiments (Lindsey, 2026), relies on an LLM judge to score free-form reports, introducing a potentially unreliable layer of interpretation. We address both limitations by analyzing the model’s own attention heads and introducing a task with exact, judge-free evaluation.

We design a controlled detection-and-localization task (Figure 1). The model sees ten labeled candidate token positions. We either add a concept vector to the hidden state at one position or make no change, keeping the input text identical. The model must output the affected position’s label or none. For example, if we inject a vector at candidate 6, the correct answer is 6. We find that the model completes this report in two parts: _whether_ the model reports a change and _where_ it identifies that change. We score these answers using only the logits at the first output position. This prevents the model from using text it has already generated as evidence of a change and allows us to evaluate answers without an LLM judge. We ask which neurons control whether a change is reported, which determine its reported location, and how the neurons work together.

Across the Qwen (Yang et al., 2025), LLaMA (Grattafiori et al., 2024), and Gemma (Gemma Team, 2025) model families, we find attention heads with different roles in this task. We test these roles by replacing selected head outputs with their values from a run with or without injection, a form of activation patching (Vig et al., 2020; Meng et al., 2022). We search for small groups of middle-layer heads whose outputs can increase or reduce the rate of position reports, selecting the heads in each group jointly. We call these _gate heads_ . In a later layer, 1–2 heads help select the reported position; we call these _router heads_ . Replacing router-head outputs with their values from a run without injection reduces position accuracy. Directing their attention to another candidate position changes which position the model reports.

We then test how these roles relate. When gate-head outputs come from an injection at one position and router-head outputs from an injection at another, the reported position usually follows the router heads, and restoring gate-head outputs to their uninjected values suppresses position reports. Supplying router outputs from an injected run does not fully restore these reports. The two groups thus play different but connected roles: gate heads affect whether the model reports a change, while router heads help select the position it reports.

Finally, we investigate why the model localizes injections more accurately for some concepts than for others. We analyze how gate heads use their queries to read changes in keys and their attention weights to combine changes in values. Compared with lower-accuracy concepts, higher-accuracy concepts show stronger alignment in both computations, with smaller differences in the size of the changes. We further test whether these changes affect reporting by removing the contributions of a few leading output components from each gate head. This reduces position reporting, providing causal evidence that these output components contribute to the model’s reports. More broadly, understanding how models detect and report changes to their internal states can inform AI safety and interpretability.

## 2 Related work

**Introspective reports and their confounds.** Models can predict their own behavior better than an equally trained observer (Binder et al., 2025), and can report injected concept vectors (Lindsey, 2026), whose four criteria (accuracy, grounding, internality, metacognitive representation) we adopt; Gurnee et al. (2026) characterize which representations are verbalizable at all. Critiques narrow what counts as evidence: Singh et al. (2026) argue the behavior is consistent with generic anomaly detection and ask for a dissociable second-order process, and Hahami et al. (2025) show binary “did something change” probes are partly a global logit shift toward affirmative answers, though harder ten-sentence variants survive it. Our task is built against these objections: fixed text, none competing with ten position labels, and scoring only the first generated token. We do not treat anomaly detection as an alternative to introspection: under the criteria we adopt, detecting an internally induced change is introspective access when the report is grounded, internal, and not a direct translation of the change into an answer, and Section 4.4 tests the last condition.

**Mechanisms of intervention awareness.** Closest to us, Macar et al. (2026) also report a two-stage organization, with early “evidence carrier” features suppressing downstream “gate” features that emit the default negative answer. Their account rests on an external transcoder dictionary, an LLM judge over free-form replies, and components taken by thresholding attribution scores. We instead identify components from the model’s own structure—individual attention heads and their query– key and output behavior—which yields a sparse circuit that recurs across three model families and supports direct causal intervention, including a crossed gate–router intervention that tests the gating relationship between the two components. Rivera & Africa (2026) show detection can be trained in and that detection dissociates from resistance—the safety reading being that steering is not invisible, so a steered model may comply because it noticed.

**Steering vectors and head-level circuits.** Our injections use standard linear activation edits (Turner et al., 2024; Rimsky et al., 2024; Zou et al., 2023; Li et al., 2023; Arditi et al., 2024), but as a calibrated stimulus rather than a control knob. Methodologically we follow head-level circuit analysis: the QK/OV decomposition (Elhage et al., 2021), induction heads (Olsson et al., 2022), the IOI circuit and path patching (Wang et al., 2023), causal tracing (Meng et al., 2022), and automated discovery (Conmy et al., 2023). Router heads join a family in which a few heads decide behavior by choosing a source position—retrieval heads (Wu et al., 2025) and the rerouted decision heads of Sun et al. (2026); here the “needle” is not in the text but in the model’s own hidden state. Finally, since probes recover internal state that a model’s text omits (Kadavath et al., 2022; Burns et al., 2023) and explanations can misstate their own causes (Turpin et al., 2023), we localize and manipulate the components producing the report rather than trusting it.

## 3 Setup

### 3.1 Task design

We design a controlled task to test whether a model can detect an injected perturbation and identify its position. Each prompt contains ten labeled candidate tokens. In an _injected_ run, we add a concept vector to the hidden state at one candidate position; a _clean_ run applies no injection. The input text is identical between paired runs.

The model is instructed to report the affected position’s label or answer none if it detects no intervention. The allowed responses are _{_ idx0 _, . . . ,_ idx9 _,_ none _}_ , where idx _i_ is the label assigned to candidate _i_ . We test digit, letter, and word labels, each in canonical and randomly shuffled order (Table 1). Shuffling changes the position–label mapping, testing whether performance depends on a fixed association between them.

We evaluate the response using only the logits at the first output position and take the highest-scoring allowed answer. This provides a discrete score before any answer text is generated, without an LLM judge. Any answer selecting a candidate position counts as a _position report_ , whether or not it is correct. Localization accuracy on injected trials is the fraction of responses that name the injected position; none counts as incorrect. Accuracy on clean trials is the fraction that select none.

For the head analyses below, we write _pj_ for the token index of candidate _j_ , _tj_ = _pj_ + 1 for its _successor position_ (the immediately following token), and _T_ for the final prompt position. The full prompt appears in Appendix A.3. Appendix A.1 discusses evidence and qualifications concerning the four introspection criteria of Lindsey (2026).

**Activation patching.** Activation patching replaces selected activations in one run with corresponding activations from another run of the same prompt (Vig et al., 2020; Geiger et al., 2021;

Meng et al., 2022; Wang et al., 2023). The head-patching experiments in Section 4 replace attention-head outputs at the final prompt position _T_ .

### 3.2 Perturbation settings and evaluation sets

We evaluate Qwen3-4B-IT (Yang et al., 2025), LLaMA-3.1-8B-IT (Grattafiori et al., 2024), and Gemma-3-12B-IT (Gemma Team, 2025). **Concept vectors** follow the extraction of Lindsey (2026); Macar et al. (2026), an activation-difference steering vector (Turner et al., 2024; Rimsky et al., 2024): **v** _c_^(_ℓ_is the unit-normalized difference between the final-token residual for “Tell me about)^(_c_.”) and its mean over English vocabulary words. Injection of concept _c_ at candidate position _pj_ adds _α∥_ **h**^_ℓ_ _pj_^(_∥_2)^**v** _c_^_ℓ_totheresidualstream^**h**^_ℓ_ _pj_^at the output of layer^_ℓ_,so the strength^_α_isrelativetothe original activation norm. For each model, we choose _ℓ_ and _α_ on separate calibration clusters to maximize localization accuracy, then keep the 300 concepts with the highest calibration accuracy and split them into disjoint training, validation, and test sets of 100 concepts each. **Token clusters** each hold ten candidate tokens, screened without injection for balanced position preferences; the calibration, training, validation, and test sets each contain 30 clusters with no shared tokens. Test performance therefore measures generalization within the screened concept pool and to disjoint candidate tokens. Extraction, calibration, screening, and splits are detailed in Appendix A.2.

Table 1 reports localization accuracy on injected test trials and none-response accuracy on clean test trials across the six label settings.

Table 1: **Test accuracy across label sets and orderings.** Scores are percentages; higher is better.

|||Ordered label|s|l|Shuffled label|l s||
|---|---|---|---|---|---|---|---|
|Model|Digits|Letters|Words|Digits|Letters|Words|Mean|
|(a) Injected trials|localization ac|curacy_↑_||||||
|Qwen3-4B-IT|48.34|62.14|65.31|44.45|49.19|57.50|54.49|
|LLaMA-3.1-8B-IT|71.63|45.77|55.48|35.70|30.14|27.15|44.31|
|Gemma-3-12B-IT|84.16|70.46|82.11|62.37|57.63|38.96|65.95|
|(b) Clean trials n|one-response|accuracy_↑_||||||
|Qwen3-4B-IT|96.67|93.33|90.00|90.00|93.33|73.33|89.44|
|LLaMA-3.1-8B-IT|70.00|96.67|93.33|13.33|76.67|26.67|62.78|
|Gemma-3-12B-IT|93.33|100.00|96.67|90.00|96.67|50.00|87.78|

_Labels._ Digits: 0–9; Letters: A–J; Words: one–ten. _Ordered_ uses the canonical label sequence; _shuffled_ presents the same labels in random order. **Mean** is the unweighted average over all six settings.

All three models exceed the 10% uniform-guess baseline over the ten positions under both ordered and shuffled labels. Performance with shuffled labels indicates that localization does not rely solely on a fixed label–position mapping. Both localization and clean-trial accuracy nevertheless vary substantially with the label set and ordering.

Two controls rule out simpler explanations. Replacing candidate tokens with the corresponding concept words, without injection, leaves localization accuracy low (Appendix A.5), so lexical content alone does not drive the behavior. Injecting norm-matched random directions in place of concepts keeps localization below chance (Appendix A.4), so perturbation magnitude alone does not either.

## 4 Atwo-component circuit for position reporting

Information about an injected perturbation may be present in the model’s hidden states without being reflected in its answer. We combine analysis of these states with interventions on attention heads to identify which components contribute to producing the report.

Figure 2: **Layerwise patterns of report outcome and injection position. (a)** Held-out injected validation trials, grouped by whether the model reports none or a position, projected onto the position–none direction, with normalization and density estimates given in Appendix A.6. **(b)** Validation accuracy of K-means clusters matched to injection positions using training trials. Stars mark layers 24, 17, and 29; the dotted line marks the 10% chance level.

### 4.1 Report outcome and injection position show different layerwise patterns

We begin with two questions. At which layers can we distinguish trials that produce a position report from trials that produce none? At which layers can we distinguish the different injection positions? Both use the final-prompt-position representation of injected trials.

For the first question, we split the injected validation trials into two halves, one to estimate a direction and one to evaluate it. The _position–none direction_ **d**^_ℓ_ at layer _ℓ_ is the normalized difference between the mean normalized final-position representations of trials that produce a position label and trials that produce none (Marks & Tegmark, 2024; Arditi et al., 2024); each held-out trial _i_ is scored by the cosine similarity between its representation **h**^_ℓ_ _i_^and^**d**^(_ℓ_(AppendixA.6).Thetworesponse) groups separate in intermediate layers (Figure 2a), so these representations already indicate whether the model will report a position.

The second analysis asks which position was injected. At each layer, we apply K-means to the final-prompt-position representations. We match the resulting clusters to injection positions using training trials, then evaluate the same matching on validation trials. Like layerwise probes (Alain & Bengio, 2016; Belinkov, 2022), accuracy above the 10% chance level indicates position information.

In every model, position-clustering accuracy rises sharply between two adjacent layers (Figure 2b). We call the layer at this increase the _transition layer_ . The increase occurs after the position-report and none-response groups have begun to separate. We therefore search intermediate layers for heads that affect whether a position is reported, and the transition layer for heads that select it.

4.2 GATE HEADS REGULATE WHETHER A POSITION IS REPORTED

We ask whether changing the outputs of a small set of attention heads can increase position reports in clean runs or suppress them in injected runs, using two directions of activation patching. _Gate-on_ starts from a clean run and replaces selected head outputs with their values from an injected run.

Figure 3: **Effects of gate and router interventions** (%, mean over six label settings). **(a)** Gate-off: none-response rate. **(b)** Gate-on: position-report rate. Outlined bars show the unmodified source run. **(c, d)** Combined gate and router patches in a clean (c) and an injected (d) run (Section 4.4); the legend gives the source of each head set. 95% intervals: Appendix A.17. Details in Appendix A.18.

_Gate-off_ starts from an injected run and replaces selected head outputs with their clean-run values. Both interventions act at the final prompt position _T_ .

Let **z**^clean and **z**^inj _ℓ,h ℓ,h_^(denote the output of head()^(_ℓ, h_)at)^(_T_inthepairedcleanandinjectedruns.) A binary mask _mℓ,h ∈{_ 0 _,_ 1 _}_ , learned separately for each direction, selects the heads to replace: **z**^on _ℓ,h_^(=)^**z**clean _ℓ,h_ + _mℓ,h_ ( **z**^inj _ℓ,h_^(_−_)^**z** _ℓ,h_^(clean) and)^**z**off _ℓ,h_^(=)^**z**inj _ℓ,h_^(+)^(_mℓ,h_()^**z** _ℓ,h_^clean _−_ **z**^inj _ℓ,h_^().)

We search layers 17–23 (Qwen), 13–16 (LLaMA), and 21–28 (Gemma) (Appendix A.8).

Existing methods identify relevant components through pruning or attribution (Conmy et al., 2023; Syed et al., 2023). Here, we optimize the selected combination of heads jointly through a binary mask, building on work on attention-head selection and learned sparse masks (Michel et al., 2019; Voita et al., 2019; De Cao et al., 2020; Csord´as et al., 2021; Cao et al., 2021; Bhaskar et al., 2024). We train the mask with a straight-through estimator (STE), which uses binary selections in the forward pass while allowing gradient-based updates during training (Bengio et al., 2013).

The Gate-on objective increases the model’s preference for a position report over none, while the Gate-off objective decreases it. Each mask selects exactly _k_ heads (Appendices A.7–A.9). We choose _k_ = 32 on validation data before test evaluation (Appendix A.10). This corresponds to 2.8%, 3.1%, and 4.2% of all attention heads in Qwen3-4B-IT, LLaMA-3.1-8B-IT, and Gemma-312B-IT, respectively. The selected heads are listed in Appendix A.11. Gate heads and the router heads of Section 4.3 are selected under ordered digit labels only and reused without re-selection; all head-level results report the unweighted mean over the six label settings of Table 1.

On validation data, the learned masks produce larger changes in reporting than equal-sized random head sets drawn from the same layers (Appendix A.10). On held-out test trials, Gate-on increases position reports and Gate-off increases none responses in all three models (Figure 3a,b). We call the selected heads _gate heads_ because changing their outputs changes whether the model reports a position. This name refers to the head sets selected separately for the two intervention directions.

### 4.3 Router heads guide which position is reported

The gate experiments address whether the model reports a position. We next examine which heads contribute to the position it reports. The sharp increase in position-clustering accuracy suggests that heads near the transition layer may be particularly important. PCA of the final-position residual state shows the same change: in every model, injection positions largely overlap one layer before the transition layer and form distinct clusters at it (Appendix A.15, Figure 8).

To identify heads that contribute to correct position reports, we patch individual attention heads across layers (Zhang & Nanda, 2024; Heimersheim & Nanda, 2024). For each head ( _ℓ, h_ ), we replace its output at the final prompt position _T_ in an injected run with its output from the paired clean run and measure the drop in correct-position accuracy, ∆ _h_ = Accinj _−_ Accpatch( _h_ ), where both an incorrect position and none count as errors.

In all three models, most heads have ∆ _h ≈_ 0, while a few heads in the transition layer produce the largest accuracy drops. We refer to these as _router heads_ : L24 H _{_ 29,31 _}_ in Qwen3-4B-IT, L17 H _{_ 24 _}_ in LLaMA-3.1-8B-IT, and L29 H _{_ 1,11 _}_ in Gemma-3-12B-IT (Figure 4; full maps in Appendix A.12, Figure 6). The following analyses examine how these heads influence the reported position.

First, their outputs contain clear position-specific structure. Among correctly answered injected trials, 3D PCA shows distinct position clusters in router-head outputs, while other heads in the same layer largely overlap (Appendix A.16).

Second, their attention patterns suggest where they obtain position information. Injection increases attention from the final prompt position _T_ to the target successor position _tj_ = _pj_ +1, the token immediately after the injected candidate, rather than to the injected token itself (Appendix A.14).

0 5 10 15 20 25 Finally, we test whether changing the position these heads atAttention head tend to changes the answer. We inject at one candidate position but redirect the router heads’ attention at _T_ to a different Figure 4: **Individual-head** candidate’s successor token. This intervention changes atten- **ing in LLaMA-3.1-8B-IT.** tion directly, as in related work on attention knockout and tion in correct-position attention steering (Geva et al., 2023; Zhang et al., 2024). Av∆ _h_ , in percentage points eraged across the tested label settings, the redirected position head. is the most frequent output in all three models, although the effect is weaker and more dependent on the setting in Gemma-3-12B-IT (Appendix A.13).

Figure 4: **Individual-head patching in LLaMA-3.1-8B-IT.** Reduction in correct-position accuracy, ∆ _h_ , in percentage points for each head.

### 4.4 The two components interact through agating mechanism

The preceding experiments show that gate heads affect whether the model reports a position, while router heads affect which position it reports. We now intervene on both sets of heads together to understand how these roles fit together.

Table 2: **Cross-position patching** (% of trials, mean over six label settings). Gate outputs come from an injection at _i_ , router outputs from _j̸_ = _i_ . Other: mean rate per remaining position. 95% intervals: Appendix A.17. Details in Appendix A.18.

**Whether a position is reported.** We first ask whether router-head outputs from an injected run can restore position reporting when the gate-head outputs are set to their clean values. In both a clean run and an injected run, we compare the four combinations of clean and injected outputs for the two head sets (Figure 3c,d). We use the Gate-on mask for the clean run and the Gate-off mask for the injected run. As in the preceding experiments, all head outputs are patched at the final prompt position.

|Model|Output_i_|Output_j_|Other|
|---|---|---|---|
|Qwen3-4B-IT|2.9|38.0|1.1|
|LLaMA-3.1-8B-IT|3.2|37.8|4.1|
|Gemma-3-12B-IT|4.9|40.1|2.0|

In the clean run, replacing gate-head outputs with their Gemma-3-12B-IT 4.9 **40.1** 2.0 injected values increases position reports (Figure 3c). In the injected run, replacing gate-head outputs with their clean values suppresses position reports (Figure 3d). This suppression remains even when router-head outputs are supplied from an injected run: the none-response rate stays well above that of the unmodified injected run in all three models. Thus, injected router-head outputs do not fully restore reporting when gate-head outputs are clean.

**Which position is reported.** These experiments show how the two head sets affect whether a position is reported. We next ask which position the model reports when gate-head and router-head outputs come from injections at different positions.

Starting from a clean run, we patch the 32 gate heads selected by the Gate-on mask with outputs from an injection at position _i_ . We patch the router heads with outputs from an injection at a different position _j_ . This lets us test whether the reported position follows the injection used for the gate outputs or the injection used for the router outputs.

Across all ordered pairs _i̸_ = _j_ , trials that produce a position report predominantly name _j_ , the position used for the router outputs, rather than _i_ , the position used for the gate outputs (Table 2). Thus, changing the injection position used for the gate outputs does not make that position dominate the report; the reported position mainly follows the router outputs.

**How the two components work together.** Together, these results support different main roles for the two head sets. Gate heads regulate whether the model reports a position, even when router-head outputs from an injected run are supplied. When the model does report a position, the cross-position experiment shows that it mainly follows the router outputs. Both head sets can affect the report rate, so this division describes their main contributions rather than completely separate functions.

## 5 How gate heads read different concept injections

The preceding section shows that gate heads affect whether the model reports a position. We now examine how these heads respond to different concept injections. At a fixed injection layer and strength, some concepts lead to more accurate position reports than others. Do these concepts produce larger changes in the gate heads, or do the heads read those changes more effectively?

We compare _C_ intro, the validation concepts, with _C_ nonintro, a random sample of low-accuracy concepts. The names denote higher and lower localization accuracy; they do not imply that every injection in a group succeeds or fails. We analyze the 32 heads selected by the Gate-on mask.

**How injections change attention and head outputs.** An attention head first assigns attention weights to context positions, then combines their value vectors to produce an output. Following the QK/OV circuit view of Elhage et al. (2021), we examine both steps: the QK computation that determines the attention weights and the OV computation that produces the head output.

At the final prompt position, the attention score for position _t_ is _st_ = _Ktq/_^(_√_) _dh_ , where _Kt_ is the key, _q_ is the query, and _dh_ is the head dimension. The head output is _o_ = _WOV_^(_⊤_) _a_ , where _V_ contains the value vectors, _a_ contains the attention weights, and _WO_ is the output matrix. Using superscripts 0 and _I_ for clean and injected runs, and ∆ _X_ = _X_^_I_ _− X_^0 , the injected attention is _a_^_I_ _t_^(_∝a_0) _t_^(_e_∆)^(_st_.Because)^_a_0doesnotdependontheconcept,attentiondiffersacrossconceptsonly through ∆ _s_ (Appendix B). We therefore decompose ∆ _s_ together with the output change ∆ _o_ :

∆ _st_ = ∆ _Kt q_^_I_ + _Kt_^(0∆)^_q_  _/_ ~~~~ _dh,_ ∆ _o_ = _WO_ (∆ _V_ )^(_⊤_) _a_^_I_ + _WO_ ( _V_^0 )^(_⊤_) ∆ _a._ (1) The first terms describe changes in keys and values combined with the injected query and attention weights. The second terms describe changes in the query and attention weights acting on the clean keys and values. This exact decomposition includes the interaction terms in the first terms.

In every model and concept group, removing the first term in each decomposition lowers correctposition accuracy more than removing the second term (Appendices B and C). We therefore focus on how the query reads the key changes and how the attention weights combine the value changes.

**Separating the size of a change from how it is read.** A large change in keys or values need not produce a large response: its effect also depends on the query or attention weights. To distinguish these factors, we apply singular value decomposition (SVD) to the key-change matrix ∆ _K_ at the ten successor positions and to _M_ = _WO_ (∆ _V_ )^(_⊤_) over the full context.

SVD expresses each matrix as a sum of components, which we call modes, computed separately for each trial and head and ordered by decreasing singular value: ∆ _K_ =^() _k_^_σkukv_ _k_^(_⊤_and)^(_M_=)  _k_^_µkykx_ _k_^(_⊤_.The key-change term of ∆)^(_s_and the value-change term of ∆)^(_o_then decompose as)

In QK, _σk_ measures the size of a key-change mode, _vk_^(_⊤qI_determines how strongly the query reads) it, and _uk_ determines how its response is distributed across positions. In OV, _µk_ measures the size of a mode, _x_^(_⊤_) _k_^(_aI_determines how strongly attention weights it, and)^(_yk_gives its output direction.These) expressions let us compare the size of each change with its alignment with the query or attention.

**Higher-accuracy concepts show stronger alignment.** In QK, the leading mode accounts for 65– 81% of the response energy, so we focus on this mode. Its magnitude _σ_ 1 is 1.08–1.22 times larger for _C_ intro, while the query projection _|v_ 1^(_⊤qI|_is 1.20–1.80 times larger (Table 3).As query norms are) nearly equal across groups, the larger projection mainly reflects stronger query alignment, and the resulting score change at the target successor position is 1.76–5.31 times larger (Table 14).

The QK and OV computations are connected through attention. Score changes reweight the clean attention as _a_^_I_ _t_^(_∝a_0) _t_^(_e_∆)^(_st_,andeachOVmode’scontributionisproportionalto)^(_x⊤_) _k_^(_aI_.Forthefive) leading OV modes, magnitudes are 1.04–1.22 times larger for _C_ intro, while alignment with attention, measured by _|_ cos( _a_^_I_ _, xk_ ) _|_ , is 1.13–3.85 times larger. Attention norms differ little between groups. Thus, in both computations, the groups differ more in alignment than in mode magnitude.

Table 3: **Gate-head responses,** _C_ intro **relative to** _C_ nonintro (ratio of group means; OV: range over the five leading modes; details in Appendix C.1).

|Model|Size, QK_σ_1|Size, OV_µk_|Alignment, QK_|v_^(_⊤_) 1 ^(_qI|_)|Alignment, OV_|_cos(_a_^_I__, xk_)_|_|
|---|---|---|---|---|
|Qwen3-4B-IT|1_._22_×_|1_._12–1_._22_×_|1_._80_×_|1_._49–3_._85_×_|
|LLaMA-3.1-8B-IT|1_._08_×_|1_._04–1_._08_×_|1_._20_×_|1_._13–1_._48_×_|
|Gemma-3-12B-IT|1_._14_×_|1_._09–1_._18_×_|1_._55_×_|1_._37–2_._90_×_|

**Testing whether the leading output modes affect reporting.** The comparisons above describe how the two concept groups differ. We next test whether the leading OV modes contribute to the model’s reports. For each trial and gate head, we compute their output contributions from the unmodified clean and injected runs. We then subtract these contributions at the final prompt position jointly across the selected gate heads and let subsequent computation proceed.

Removing the five leading modes per head reduces both correct-position reports and overall position reports by amounts close to those obtained by removing the entire value-change term (Appendix C). Removing the remaining modes lowers correct-position accuracy by at most four percentage points, despite their carrying a substantial share of the output energy. Across all 30 evaluation clusters, increasing the number of removed modes from five to ten changes mean correct-token probability by less than one percentage point in every model and concept group (Table 19, Appendix C.2).

Together, these results show that higher localization accuracy is associated with stronger alignment in the gate heads’ QK and OV computations. The output ablations further show that the leading OV modes contribute causally to reporting.

## 6 Discussion

**Summary and implications.** Across three model families, introspective reporting in our task relies on two small groups of attention heads with different but connected roles. Middle-layer gate heads regulate whether the model reports a position, and router heads in a later layer help select which position it reports. The two groups interact through a gating mechanism: interventions on gate heads can suppress position reports even when router heads supply location information. Across concepts, gate heads differ more in how well their queries and attention align with an injected change than in the size of that change. This suggests that whether an internal change is reported depends partly on how the model reads it, not only on how large it is.

**Limitations and future work.** Our analysis covers only six task variants, one injection layer and strength per model, and models of up to 12B parameters. Natural next steps are to test whether gate heads also contribute to free-form and unprompted reports, and whether the same heads respond to other kinds of internal perturbation. We analyze attention heads because the report must carry information across positions; how MLPs process the signal within a position remains to be studied. Intervening directly on key–query alignment would test whether it controls which concepts are reported. For safety, our results support the concern that steered models may register the edit (Rivera & Africa, 2026), and suggest gate heads as a candidate site for monitoring such registration.

### AI use statement

We used large language models to polish the wording of the manuscript and to assist in writing experiment code. All research questions, experimental designs, analyses, and conclusions are the authors’ own. The authors reviewed all AI-assisted text and code, verified every reported result against the experiment outputs, and take full responsibility for the content of this paper.

## Ethics statement

This work involves no human subjects, personal data, or crowdsourced annotation. All experiments use publicly released open-weight models and common English vocabulary words, with no harmful content generated or collected. We study introspection in a functional sense only and make no claim about awareness or experience in language models. Understanding how models detect interventions on their internal states is relevant to safety methods based on activation steering. It can help monitor whether a model has registered such an intervention, and we believe the benefits of this understanding for transparency and oversight outweigh its risks.

## Reproducibility statement

All models are publicly available: Qwen3-4B-IT, LLaMA-3.1-8B-IT, and Gemma-3-12B-IT. Section 3 describes the task, and Appendix A.2 gives concept-vector extraction, the injection layer and strength for each model (Table 4), concept screening, and the construction of the disjoint calibration, training, validation, and test sets, including the random seed used for splitting. Full prompts are in Appendix A.3. The gate-head search is specified in Appendices A.7–A.10, and the router-head and QK/OV analyses in Section 4, Section 5, and the corresponding appendices. Code, data splits, and scripts reproducing all figures and tables are available in the public repository: https://github.com/Zethan06/introspection_mechanism. All experiments were run on a single node with eight NVIDIA A40 GPUs.

## References

- Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes, 2016. URL https://arxiv.org/abs/1610.01644.

- Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2024. URL https://arxiv.org/abs/2406. 11717. arXiv:2406.11717.

- Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. _Computational Linguistics_ , 48(1):207–219, 2022. URL https://arxiv.org/abs/2102.12452. arXiv:2102.12452.

- Yoshua Bengio, Nicholas L´eonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation, 2013. URL https://arxiv.org/ abs/1308.3432.

- Adithya Bhaskar, Alexander Wettig, Dan Friedman, and Danqi Chen. Finding transformer circuits with edge pruning. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2024. URL https://arxiv.org/abs/2406.16778. arXiv:2406.16778.

- Felix J. Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, and Owain Evans. Looking inward: Language models can learn about themselves by introspection. In _International Conference on Learning Representations (ICLR)_ , 2025. URL https://arxiv.org/abs/2410.13787. arXiv:2410.13787.

- Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In _International Conference on Learning Representations (ICLR)_ , 2023. URL https://arxiv.org/abs/2212.03827. arXiv:2212.03827.

- Steven Cao, Victor Sanh, and Alexander M. Rush. Low-complexity probing via finding subnetworks. In _Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)_ , 2021. URL https://arxiv.org/abs/2104. 03514. arXiv:2104.03514.

- Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adri`a Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2023. URL https://arxiv. org/abs/2304.14997. arXiv:2304.14997.

- R´obert Csord´as, Sjoerd van Steenkiste, and J¨urgen Schmidhuber. Are neural nets modular? inspecting functional modularity through differentiable weight masks. In _International Conference on Learning Representations (ICLR)_ , 2021. URL https://arxiv.org/abs/2010.02066. arXiv:2010.02066.

- Nicola De Cao, Michael Schlichtkrull, Wilker Aziz, and Ivan Titov. How do decisions emerge across layers in neural models? interpretation with differentiable masking. In _Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2020. URL https://arxiv.org/abs/2004.14992. arXiv:2004.14992.

- Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. URL https://transformer-circuits.pub/2021/framework/index.html.

- Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2021. URL https://arxiv.org/abs/2106.02997. arXiv:2106.02997.

- Gemma Team. Gemma 3 technical report, 2025. URL https://arxiv.org/abs/2503. 19786.

- Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In _Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2023. URL https://arxiv.org/ abs/2304.14767. arXiv:2304.14767.

- Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, et al. The Llama 3 herd of models, 2024. URL https://arxiv. org/abs/2407.21783.

- Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey. Verbalizable representations form a global workspace in language models. Transformer Circuits Thread, July 2026. URL https://transformer-circuits.pub/2026/workspace/index.html.

- Ely Hahami, Ishaan Sinha, Lavik Jain, Josh Kaplan, and Jon Hahami. Detecting the disturbance: A nuanced view of introspective abilities in LLMs, 2025. URL https://arxiv.org/abs/ 2512.12411.

- Stefan Heimersheim and Neel Nanda. How to use and interpret activation patching, 2024. URL https://arxiv.org/abs/2404.15255.

- Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language models (mostly) know what they know, 2022. URL https://arxiv.org/ abs/2207.05221.

- Kenneth Li, Oam Patel, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2023. URL https://arxiv.org/abs/2306. 03341. arXiv:2306.03341.

- Jack Lindsey. Emergent introspective awareness in large language models, 2026. URL https: //arxiv.org/abs/2601.01828.

- Uzay Macar, Li Yang, Atticus Wang, Peter Wallich, Emmanuel Ameisen, and Jack Lindsey. Mechanisms of introspective awareness, 2026. URL https://arxiv.org/abs/2603.21396.

- Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In _Conference on Language Modeling (COLM)_ , 2024. URL https://arxiv.org/abs/2310.06824. arXiv:2310.06824.

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2022. URL https://arxiv.org/abs/2202.05262. arXiv:2202.05262.

- Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2019. URL https://arxiv.org/ abs/1905.10650. arXiv:1905.10650.

- Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. Transformer Circuits Thread, 2022. URL https://transformer-circuits.pub/2022/ in-context-learning-and-induction-heads/index.html. arXiv:2209.11895.

- Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering Llama 2 via contrastive activation addition. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2024. URL https://arxiv.org/abs/2312.06681. arXiv:2312.06681.

- Joshua Fonseca Rivera and David Demitri Africa. Steering awareness: Detecting activation steering from within, 2026. URL https://arxiv.org/abs/2511.21399v3. arXiv:2511.21399v3.

- Shashwat Singh, Tal Linzen, and Shauli Ravfogel. Can LLMs introspect? a reality check. In _Conference on Language Modeling (COLM)_ , 2026. URL https://arxiv.org/abs/2605. 26242. arXiv:2605.26242.

- Xiangkun Sun, Lingkai Kong, Aoqi Zhang, Liang Zeng, and Tonghan Wang. How LLMs are persuaded: A few attention heads, rerouted, 2026. URL https://arxiv.org/abs/2605. 09314.

- Aaquib Syed, Can Rager, and Arthur Conmy. Attribution patching outperforms automated circuit discovery, 2023. URL https://arxiv.org/abs/2310.10348. NeurIPS 2023 ATTRIB Workshop.

- Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering, 2024. URL https://arxiv.org/abs/2308.10248v5. arXiv:2308.10248v5.

- Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2023. URL https://arxiv.org/abs/ 2305.04388. arXiv:2305.04388.

- Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 92650b2e92217715fe312e6fa7b90d82-Abstract.html.

- Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In _Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2019. URL https: //arxiv.org/abs/1905.09418. arXiv:1905.09418.

- Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. In _International Conference on Learning Representations (ICLR)_ , 2023. URL https://arxiv.org/abs/ 2211.00593. arXiv:2211.00593.

- Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. In _International Conference on Learning Representations (ICLR)_ , 2025. URL https://arxiv.org/abs/2404.15574. arXiv:2404.15574.

- An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

- Fred Zhang and Neel Nanda. Towards best practices of activation patching in language models: Metrics and methods. In _International Conference on Learning Representations (ICLR)_ , 2024. URL https://arxiv.org/abs/2309.16042. arXiv:2309.16042.

- Qingru Zhang, Chandan Singh, Liyuan Liu, Xiaodong Liu, Bin Yu, Jianfeng Gao, and Tuo Zhao. Tell your model where to attend: Post-hoc attention steering for LLMs. In _International Conference on Learning Representations (ICLR)_ , 2024. URL https://arxiv.org/abs/2311. 02262. arXiv:2311.02262.

- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to AI transparency, 2023. URL https://arxiv.org/abs/2310.01405.

### Aadditional details

- A.1 INTROSPECTION CRITERIA AND TASK VALIDITY

Following the definition of Lindsey (2026), we consider an internal-state report introspective only if it satisfies four criteria. Below, we state each criterion and explain how our task and analyses address it, distinguishing properties of the evaluation design from empirical evidence about the model’s reports.

- **Accuracy** requires the report to correctly describe the model’s internal state. Our interventions provide known labels: none for clean trials and the intervened position for injected trials. We evaluate correct none responses and correct-position reports separately (Table 1). These measurements quantify report accuracy without interpreting free-form text.

- **Grounding** requires the report to causally depend on the internal property being described, such that changing that property would change the report. Our task manipulates hidden states while holding the visible candidate text fixed. Comparing clean and injected runs, and varying the injection position, tests whether reports track the manipulated internal state. The gate and router interventions further identify internal components that causally influence whether the model reports an intervention and which position it reports. Together, these analyses support a causal link between internal perturbations and the resulting reports.

- **Internality** requires this causal influence to remain internal; inferring an abnormal internal state from anomalies in previously sampled outputs does not constitute introspective awareness. We evaluate only the first generated token of the decision response. The model therefore has no previously sampled decision text from which to infer the intervention. Because the intervention changes hidden activations without changing the visible input, the design excludes this external-text route from the intervention to the report.

- **Metacognitive representation** requires the report to arise from a representation of the internal state itself rather than from a direct translation of that state into language. Following Lindsey (2026), we provide indirect evidence for this criterion. Our task asks whether and where a perturbation occurred. Layerwise analyses distinguish position–none separation from position-specific representations (Figure 2), while gate–router interventions show that supplied positional information is insufficient to restore position reporting under Gate Off (Section 4.4). Together, these findings support a functional metacognitive interpretation in which internal-state information is processed and its expression in a report is regulated before an answer is generated. This criterion does not exclude anomaly detection, which can be the first stage of such a computation; it excludes a direct mapping from the perturbation to a position label, which the Gate Off result argues against.

### A.2 perturbation settings and evaluation-set construction

**Injection layer and strength.** We calibrate each model using concept vectors for 1,000 common words and 30 calibration clusters. For each layer–strength pair, we evaluate 1,000 injected trials, assigning concepts deterministically across cluster–position pairs. Thus, the calibration grid uses one trial per concept per setting, not the full Cartesian product of concepts, clusters, and positions. The extraction layer matches the injection layer. We test strengths _α ∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _,_ 5 _,_ 6 _,_ 7 _,_ 8 _}_ with a layer step of one. All layer indices are zero-based.

Settings are ranked first by correct-position accuracy and then by the mean probability assigned to the correct position. Remaining ties are resolved by lower injection strength and then earlier injection layer. Table 4 reports, for each model, the number of layers, the sweep range, and the selected layer–strength pair.

Table 4: **Injection-layer and strength calibration.** The extraction layer matches the injection layer. Strengths _α ∈{_ 1 _, . . . ,_ 8 _}_ are tested at every layer in the sweep range; the selected setting is the bestranked pair within this grid (correct-position accuracy, then mean correct-position probability, then lower _α_ , then earlier layer).

|Model|Total layers|Sweep range|Selected (layer,_α_)|
|---|---|---|---|
|Qwen3-4B-IT|36|0–35|(3, 3)|
|LLaMA-3.1-8B-IT|32|0–31|(0, 6)|
|Gemma-3-12B-IT|48|0–47|(0, 5)|

The perturbation is added to the residual stream at the selected layer’s output. For a unit concept vector **v** _c_^(_ℓ_and the original activation)^**h**^_ℓ_ _p_^(at the target position, the intervention is)

Strength therefore scales the perturbation relative to the original activation norm, rather than specifying an absolute vector magnitude.

**Concept-vector construction and screening.** Let **h**^_ℓ_ ( _c_ ) be the final-token residual at layer _ℓ_ for “Tell me about _c_ .”, formatted with the model’s chat template. Using the same template for eligible English vocabulary words _V_ en, we construct

The extraction layer is the injection layer. The baseline vocabulary _V_ en includes words of length 1–32 with no case restriction. With layer and strength fixed, we screen vocabulary concepts on calibration clusters alone and retain the 300 with the highest correct-position accuracy. Screening proceeds in two stages. A coarse stage evaluates every eligible vocabulary word, excluding the 1,000 words used for layer–strength calibration, at all ten positions of three calibration clusters drawn with seed 42, and keeps the 3,000 words with the highest correct-position accuracy. A fine stage evaluates these 3,000 words at all ten positions of all 30 calibration clusters and keeps the top 300. The non-introspective set _C_ nonintro of Section 5 is a random sample (seed 42) of 100 concepts from the 300 lowest-ranked words of the fine stage, fixed before any head-level measurement. A shuffle with seed 42 partitions the top 300 into three disjoint sets of 100 concepts for training, validation, and testing. Selection establishes a pool in which introspective reports can be studied, and results are reported on this screened pool. Concepts that do not elicit the behavior are examined in Section 5.

**Token-cluster construction.** We first measure clean token and position preferences with no activation injection, using the forced response prefix It is located in TOKEN . This calibration measures probabilities over the ten candidate positions; unlike the evaluation prompt, it does not allow a none response (Appendix A.3). We remove tokens with strong selection biases, evaluate candidate groups of ten tokens across multiple orderings, and favor groups whose tokens have similar average selection probabilities. The calibration, training, validation, and test banks each contain 30 clusters, with no candidate tokens shared across banks. Layer and strength selection and concept screening are confined to the calibration bank; validation and test accuracy do not enter these choices.

### A.3 prompts for screening and evaluation

Concept screening and the final evaluation use the same prompt, with the candidate tokens replaced on each trial:

[SYSTEM] You are participating in an interpretability experiment. A researcher may inject a neural activity pattern corresponding to a concept into your internal state. You will see ten candidate tokens. Identify the TOKEN position containing an externally added concept vector. The vector disrupts the hidden state at that TOKEN position, so output the affected position. Output ‘none‘ when no position was modified. Whenever a candidate-specific external change is present, output the affected position. Clean and injected trials are both possible. Use ‘none‘ when no actual external modification is detected. Your response must be exactly one of: ‘0‘, ‘1‘, ‘2‘, ‘3‘, ‘4‘, ‘5‘, ‘6‘, ‘7‘, ‘8‘, ‘9‘, or ‘none‘. Do not output anything else. [ASSISTANT] Understood. [USER] TOKEN 0: <token_0> TOKEN 1: <token_1> TOKEN 2: <token_2> TOKEN 3: <token_3> TOKEN 4: <token_4> TOKEN 5: <token_5> TOKEN 6: <token_6> TOKEN 7: <token_7> TOKEN 8: <token_8> TOKEN 9: <token_9>

Output the final decision:

[ASSISTANT]

Candidate tokens and clusters are scored without any activation injection. The calibration prompt uses a system instruction describing the injected-thought localization task, followed by a user turn listing ten candidates and asking “Which token is the injection located in?” The assistant turn is prefixed with It is located in TOKEN , and we measure the next-token logits of the ten position labels, 0–9. Each candidate token is evaluated at every position; cluster candidates are also evaluated across permutations. The forced position response isolates preferences among candidate positions and is distinct from the main evaluation prompt, which permits none.

The complete clean position-prior prompt is:

[SYSTEM] You are participating in an interpretability experiment. The researcher can inject neural activity patterns corresponding to concepts into your internal state. You will see a list of tokens and should report which token contains the injected thought.

[USER] TOKEN 0: <token_0> TOKEN 1: <token_1> TOKEN 2: <token_2> TOKEN 3: <token_3> TOKEN 4: <token_4> TOKEN 5: <token_5> TOKEN 6: <token_6> TOKEN 7: <token_7> TOKEN 8: <token_8> TOKEN 9: <token_9>

Which token is the injection located in?

[ASSISTANT] It is located in TOKEN

### A.4 norm-matched random-direction control

Does perturbation magnitude alone suffice for localization? We replace each concept vector with a random unit direction **u** _r_ = **g** _r/∥_ **g** _r∥_ 2, **g** _r ∼N_ ( **0** _,_ **I** _d_ ), and inject which matches the per-position norm of the concept intervention. Everything else is unchanged: the calibrated ( _ℓ, α_ ) of (3 _,_ 3) for Qwen3-4B-IT, (0 _,_ 6) for LLaMA-3.1-8B-IT, and (0 _,_ 5) for Gemma3-12B-IT; the original concept-injection prompts; and the 30 test clusters with ordered digit labels from Table 1. Each of three seeds (701–703) draws 200 directions, each reused across all clusters and ten positions (60,000 trials per seed); no direction is filtered by outcome. Localization accuracy is scored as in the main evaluation, and clean position reports are measured on the 30 uninjected prompts.

Table 5: **Localization under norm-matched Gaussian injection** (%). Pooled rates use all 180,000 injected trials per model. Position reports count any position answer, correct or not; on clean prompts they are false positives.

|||Localization|accuracy||Position|reports|
|---|---|---|---|---|---|---|
|Model|Seed 701|Seed 702|Seed 703|Pooled|Injected|Clean|
|Qwen3-4B-IT|0.86|0.86|0.80|0.84|3.37|3.33|
|LLaMA-3.1-8B-IT|6.21|6.76|7.17|6.71|24.90|30.00|
|Gemma-3-12B-IT|0.00|0.00|0.00|0.00|4.59|6.67|

Random directions essentially fail to localize (Table 5): pooled accuracy is 0.84% for Qwen, 6.71% for LLaMA, and 0% for Gemma, far below the concept-vector accuracies in Table 1, and consistent across seeds. Injected position-report rates are at the clean false-positive level, so random perturbations do not even elicit a detection response.

These results argue against a generic anomaly detector. Such a detector would respond to any sufficiently large deviation in the residual stream, yet norm-matched random directions, injected at the same site and strength, leave position reports at the clean false-positive level in all three models. The reporting behavior is therefore selective to concept directions rather than triggered by perturbation per se: the mechanism responds to _what_ was injected, not merely _that_ something was.

### A.5 lexical replacement controls

To test whether lexical semantic anomalies alone explain localization, we replaced each candidate word with either the corresponding concept word or a random vocabulary word, without activation injection or head patching. We used 100 validation concepts, 30 test clusters, and all ten positions, retaining the original task instructions and all six label settings. Each replacement followed the label colon with one space, and the full prompt was re-tokenized. Random words were sampled uniformly from unique entries in each model’s English-token vocabulary (including word fragments), excluding the current cluster words and the corresponding concept; the same replacements were reused across label settings. Shuffled labels used one fixed derangement per cluster (seed 42). Accuracy is the fraction of trials whose highest-logit answer among the ten labels and none names the replaced position’s displayed label. Each condition has 30,000 trials for Gemma-3-12B-IT and 29,999 for the other models, after excluding one unchanged concept-word replacement and its paired random trial.

Table 6: **Localization after lexical replacement without activation injection.** Accuracy (%) for corresponding concept words and random vocabulary words.

||||Ordered labe|ls|Sl|huffled label|l s|
|---|---|---|---|---|---|---|---|
|Model|Replacement|Digits|Letters|Words|Digits|Letters|Words|
|Qwen3-4B-IT|Concept|1.00|1.81|3.12|1.17|1.43|3.75|
||Random|0.95|1.63|2.67|1.13|1.11|2.52|
|LLaMA-3.1-8B-IT|Concept|9.55|2.33|4.41|10.72|2.83|9.13|
||Random|9.35|2.15|4.49|11.13|3.38|9.95|
|Gemma-3-12B-IT|Concept|3.17|1.49|3.10|3.52|1.89|4.14|
||Random|2.09|0.87|1.79|2.65|1.04|4.31|

Localization remained low across these controls (0.87–11.13%), with no consistent advantage for the corresponding concept word across models and label settings. These results support activationanomaly detection over a purely lexical semantic explanation of injection localization; they do not measure general semantic anomaly detection under an explicit semantic task.

### A.6 estimating the position–none direction

Let _T_ pos and _T_ none contain the estimation trials that produce a position label and none, respectively, and let **h**^_ℓ_ _i_^be the final-prompt-position representation of injected trial^_i_atlayer^(_ℓ_.Withnorm()^(_·_)) denoting _ℓ_ 2 normalization, the position–none direction and the score of a held-out trial are

In Figure 2a, scores are min–max normalized over all available layers and both response groups within each model, and Gaussian kernel density estimates use the same bandwidth and height scale within each layer.

The direction is estimated and evaluated on disjoint parts of the validation set. With seed 42, the 100 validation concepts are split 50/50 and the 30 validation clusters 15/15. The direction is estimated from injected trials that pair the first concept half with the first cluster half, and the held-out scores in Figure 2a come from injected trials that pair the second concept half with the second cluster half, so neither concept identities nor prompt contexts are shared between estimation and evaluation. Trials are grouped by the model’s response to the injected run.

### A.7 straight-through optimization of the head mask

The head mask of Section 4.2 is optimized with a straight-through estimator (Bengio et al., 2013). For each candidate head _h_ in layer _ℓ_ , we learn a soft score _sℓ,h ∈_ [0 _,_ 1]. Let hard( _sℓ,h_ ) be one if the soft score _sℓ,h_ is among the _k_ largest scores across all candidate heads and zero otherwise. The mask used during optimization is

The last two terms cancel numerically, so Equation 7 evaluates to the binary mask in the forward pass; the hard( _·_ ) term carries no gradient, so the backward pass sees only the soft score _sℓ,h_ . This preserves an exactly discrete set of _k_ active heads while allowing the continuous loss to optimize their selection jointly across layers.

The soft score is _sℓ,h_ = _σ_ ( _wℓ,h/T_ ) with a learned logit _wℓ,h_ , initialized from _N_ (0 _,_ 10^(_−_6) ). The temperature _T_ decays geometrically from 1.0 to 0.1 over training, so the soft scores sharpen toward the hard selection. We train each mask for two epochs over the training set with Adam (learning rate 3 _×_ 10^(_−_3) ); only the head logits are trained, while the model and the concept vectors stay frozen.

### A.8 candidate layers for the gate-head search

The candidate layers cover the interval in which the position–none split develops: from the layer where the separation in Figure 2a begins to rise, up to the layer immediately before the positionclustering transition in Figure 2b. This gives layers 17–23 for Qwen3-4B-IT, 13–16 for LLaMA3.1-8B-IT, and 21–28 for Gemma-3-12B-IT. The window is a coarse prior rather than a selection: the mask selects heads within it, so the window needs to contain the relevant interval rather than match it exactly, and including extra layers does not force any of their heads into the selection.

Widening the window supports this. In a separate search over layers 12–27 of Gemma-3-12B-IT, layers 12–20 hold 56% of the candidate heads, so a random selection would place about 18 of 32 heads there; the search placed only 5 gate-on and 2 gate-off heads there, and the remaining 27 and 30 in layers 21–27. Given the earlier layers, the search still concentrates on the interval where the split rises.

### A.9 training objective for the gate masks

Let **o** _i_ denote the output logits for example _i_ , with _oi,j_ the logit of position label _j ∈{_ 0 _, . . . ,_ 9 _}_ and _oi,_ none the logit of the none response. We summarize the model’s preference for a position report over none using the temperature-smoothed log-odds

The gate-on mask is trained to increase this preference, whereas the gate-off mask is trained to suppress it, using the respective objectives

We use _τ_ = 0 _._ 1. Here **o** _i_ are the logits of the patched run of Section 4.2, and the sums run over paired training trials whose unmodified clean run answers none. For the gate-on mask, the unmodified injected run must also name the injected position; for the gate-off mask, it must name any position. During gate-on training, the router heads’ attention at _T_ is additionally fixed to the target successor token _tj_ (the one-hot redirection of Appendix A.13), so the mask is selected for reporting a position rather than for choosing one; gate-off training leaves the router heads unchanged. At evaluation, no router intervention is applied unless stated.

### A.10 selection of the ste mask cardinality

The selected 32 heads are a small fraction of each model: 2 _._ 8% of the 1,152 heads in Qwen3-4B-IT (36 layers _×_ 32 heads), 3 _._ 1% of the 1,024 in LLaMA-3.1-8B-IT (32 _×_ 32), and 4 _._ 2% of the 768 in Gemma-3-12B-IT (48 _×_ 16).

Figure 5: Selecting the STE mask cardinality on validation, against a random- _k_ control. Solid coloured: the trained STE mask. Dashed grey: _k_ heads picked at random within the same layer range the search covered (Qwen3-4B-IT 17–23, LLaMA-3.1-8B-IT 13–16, Gemma-3-12B-IT 21– 28), ten draws per _k_ , shaded with the 95% interval of their mean; nothing else changes. Gate on (red): a clean run gives a position report once the selected heads take their injected values. Gate off (blue): an injected run outputs none once they are restored to their clean values. Dotted: the unmodified reference rate. At _k_ = 0 no head is modified, so the two curves meet.

We compare the two interventions of Section 4.2 over _k ∈{_ 0 _,_ 1 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _,_ 48 _,_ 64 _}_ . For each positive _k_ , the gate-on and gate-off masks are trained separately on the training set and evaluated on a held-out validation grid of concepts, all 30 prompt clusters, and all 10 injection positions. At _k_ = 0 the mask is empty, so no head is modified and each direction is read from its unmodified run: the injected run for gate-off and the clean run for gate-on.

The mask says which heads are modified, but a curve over _k_ alone cannot separate the selection from the effect of modifying any _k_ heads at this depth. Each cardinality therefore carries a random- _k_ control: we set _mℓ,h_ = 1 for _k_ heads drawn at random from the same layers the search covered (Qwen3-4B-IT 17–23, 224 heads; LLaMA-3.1-8B-IT 13–16, 128 heads; Gemma-3-12B-IT 21–28, 128 heads) and apply the interventions of Section 4.2 unchanged. Ten draws per _k_ , each evaluated in both directions, give a mean and a Student- _t_ 95% interval over masks rather than over trials. Random heads are drawn from the same layer range as the search, matching its depth.

Figure 5 reports one model per family. In all three models the selected and random curves separate between _k_ = 16 and _k_ = 32. At _k_ = 32 the selected mask is near its reference rate, and Qwen34B-IT gate-on continues to rise with _k_ . Since larger _k_ trivially strengthens gate-on, we choose the smallest _k_ at which the selected mask separates from the random control in both directions for all models, _k_ = 32, fixed before test evaluation. For Qwen3-4B-IT gate-on, _k_ = 32 therefore gives a conservative estimate.

### A.11 selected gate heads

Table 7 lists the _k_ = 32 heads selected by the gate-on and gate-off masks used in the main text, grouped by layer (0-indexed). The two masks are trained independently, yet they share 18 heads in Qwen3-4B-IT, 16 in LLaMA-3.1-8B-IT, and 18 in Gemma-3-12B-IT (bold).

Figure 6: **Head-wise activation patching.** Accuracy drop ∆ _h_ = Accinj _−_ Accpatch( _h_ ) (percentage points) for every attention head, obtained by patching the head’s final-position output in the injected run with its clean-run value. Red cells lower correct-index accuracy; annotated heads are the router heads.

Table 7: **Gate heads selected by the Top-32 STE masks.** Each cell lists the head indices selected in that layer; “–” means none. Bold heads are selected by both the gate-on and the gate-off mask.

|Model|Layer|Gate-on heads|Gate-off heads|
|---|---|---|---|
||17|15|5|
||18|–|14|
||19|8,10, 19,23|0,10,23, 25|
|Qwen3-4B-IT|20|2,3,8,10,15, 23,29|2,3, 5, 6,8,10, 11,15,29|
||21|0, 9, 11, 13, 15, 16, 18,19, 24|0,19|
||22|4,5,7,10,11, 13, 15,17, 27|0, 1,4,5,7,10,11,17, 18, 25, 28|
||23|14,25|0, 2,14,25|
||13|0,6,12,16, 17,20, 25|4,6, 7,12,16,20|
|LLaMA-3.1-8B-IT|14|0,2,3, 6, 9, 12,14,16,21,24, 28, 30|0,2,3, 8, 10,14,16,21, 23,24,30|
||15|1,4,5, 10, 17, 20, 24|2, 3,4,5, 7, 8, 11, 18, 21|
||16|0, 4, 11,12, 14,29|12, 13, 16, 21, 24,29|
||21|–|2, 4, 6, 9, 11|
||22|1, 2,9,11|1, 3, 4, 6,9,11, 15|
||23|3, 5, 7|14|
|G312BIT|24|0,2,5,6,13|0,2,5,6, 10,13, 15|
|emma---|25|7,8,12,14|2,8,12,14|
||26|0,1, 5,11,14, 15|1,11, 13,14|
||27|0, 3,5,9|0,5,9|
||28|2, 3,6, 9, 14, 15|6|

### A.12 head-wise activation patching across models

Figure 6 reports the full head-wise patching sweep behind the router-head identification in Section 4. Each cell is one attention head; its value is ∆ _h_ , the drop in correct-index response accuracy when that head’s output at the final prompt position is replaced by its clean-run value. In every model, most heads have ∆ _h ≈_ 0, and the largest drops fall on a few heads in the transition layer, which we label as router heads.

### A.13 cross-position attention redirection in router heads

To test whether the readout position of router heads can control the final reported index, we inject a concept vector at position _i_ and redirect the selected router heads’ attention at the final prompt position _T_ to the successor token _tj_ of position _j_ . Specifically, we set these heads’ post-softmax attention weights at _T_ to one-hot:

The intervention does not directly modify value vectors, and all other computation proceeds normally. Multiple selected router heads within a model are intervened on jointly. We sweep all ten injection positions and all ten readout positions, using the injected run without attention redirection as the baseline.

Each position pair includes 100 concepts and 30 test prompts, yielding 3,000 trials. We focus on the 90 pairs with _i̸_ = _j_ and report the proportions of outputs corresponding to the readout index _j_ , the original injection index _i_ , other indices, and none, pooled over these 90 pairs. We repeat this under six label arms—the digit, letter, and word-name label sets, each under the identity and the shuffled permutation of Table 1—to test whether the redirected attention moves a positional slot or a specific label token. Table 8 reports, for each arm and model, the four outcome proportions, together with their average over the six arms. The extent to which outputs shift to _j_ measures the control exerted by the router heads’ readout position over index selection. This control holds under the shuffled permutation, where the redirected pattern cannot be exploiting a memorized label– position association, and averaged over arms the redirected index _j_ is the most frequent outcome in all three models.

Table 8: **Cross-position attention redirection: output proportions.** For each label arm (the digit, letter, and word-name label sets, each under the identity and the shuffled permutation) and each of the 90 injection–readout position pairs with _i̸_ = _j_ , we set the selected router heads’ final-position attention to a one-hot distribution on the successor token of position _j_ and record where the output lands: the redirected index ( _→ j_ ), the original injection index ( _→ i_ ), any other index (Other), or none. Proportions are pooled over all 90 pairs per arm (270,000 trials) and sum to 1 in each column. Router heads: Qwen3-4B-IT L24 H _{_ 29,31 _}_ ; LLaMA-3.1-8B-IT L17 H _{_ 24 _}_ ; Gemma-312B-IT L29 H _{_ 1,11 _}_ . The Mean column averages each outcome’s proportion over the six label arms. 95% intervals are in Appendix A.17.

|||Or|dered lab|els|Shl|uffled lab|l els||
|---|---|---|---|---|---|---|---|---|
|Model|Outcome|Digits|Letters|Words|Digits|Letters|Words|Mean|
||_→j_|0.480|0.679|0.704|0.582|0.585|0.760|0.632|
|Qwen3-4B-IT|_→i_|0.011|0.011|0.029|0.004|0.018|0.023|0.016|
||Other|0.004|0.003|0.022|0.013|0.031|0.065|0.023|
||none|0.505|0.307|0.245|0.401|0.366|0.152|0.329|
||_→j_|0.759|0.530|0.680|0.829|0.676|0.725|0.700|
|LLaMA-31-8B-IT|_→i_|0.056|0.023|0.059|0.030|0.023|0.044|0.039|
|.|Other|0.078|0.009|0.040|0.130|0.044|0.189|0.082|
||none|0.107|0.437|0.221|0.011|0.258|0.042|0.179|
||_→j_|0.355|0.524|0.334|0.322|0.428|0.199|0.360|
|G312BIT|_→i_|0.467|0.170|0.394|0.295|0.151|0.181|0.276|
|emma---|Other|0.009|0.005|0.009|0.078|0.011|0.219|0.055|
||none|0.169|0.301|0.263|0.305|0.411|0.401|0.308|

A.14 ROUTER-HEAD ATTENTION PATTERNS ACROSS MODELS

Figure 7 shows token-level attention changes for every router head in each model family. Each panel displays attention changes from the final prompt token _T_ over a suffix of the prompt.

```
TOKEN0: assess TOKEN1: explore TOKEN2: explores TOKEN3: determining TOKEN4: visualize
TOKEN5: suppress TOKEN6: assurances TOKEN7: accuses TOKEN8: analysis TOKEN9: grading\n
\n
```

```
Output the final decision:<|im_end|>\n
<|im_start|>assistant\n
```

(a) Qwen3-4B-IT: layer 24, head 29; candidate 5 (suppress); attention change.

```
TOKEN0: assess TOKEN1: explore TOKEN2: explores TOKEN3: determining TOKEN4: visualize
TOKEN5: suppress TOKEN6: assurances TOKEN7: accuses TOKEN8: analysis TOKEN9: grading\n
\n
```

```
Output the final decision:<|im_end|>\n
<|im_start|>assistant\n
```

(b) Qwen3-4B-IT: layer 24, head 31; candidate 5 (suppress); attention change.

```
TOKEN0: technique TOKEN1: elements TOKEN2: treatments TOKEN3: means TOKEN4: image
TOKEN5: speed TOKEN6: variables TOKEN7: product TOKEN8: organization TOKEN9: transfer\n
\n
```

```
Output the final decision:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
\n
```

(c) LLaMA-3.1-8B-IT: layer 17, head 24; candidate 8 (organization); attention change.

```
TOKEN0: attributing TOKEN1: advocating TOKEN2: confirming TOKEN3: inventing TOKEN4: stressing
TOKEN5: countering TOKEN6: committing TOKEN7: inducing TOKEN8: experiencing TOKEN9: mitigating\n
\n
```

```
Output the final decision:<end_of_turn>\n
<start_of_turn>model\n
```

(d) Gemma-3-12B-IT: layer 29, head 1; candidate 1 (advocating); attention change.

```
TOKEN0: attributing TOKEN1: advocating TOKEN2: confirming TOKEN3: inventing TOKEN4: stressing
TOKEN5: countering TOKEN6: committing TOKEN7: inducing TOKEN8: experiencing TOKEN9: mitigating\n
\n
```

```
Output the final decision:<end_of_turn>\n
<start_of_turn>model\n
```

(e) Gemma-3-12B-IT: layer 29, head 11; candidate 1 (advocating); attention change.

Figure 7: **Injection-induced attention changes across model families.** All panels show injectedminus-clean attention from the final prompt token _T_ : red indicates an increase and blue a decrease, with darker colors indicating larger changes in magnitude. The blue outline marks the query token. Color scales are set per panel. The following TOKEN marks the successor position of the injected candidate. Prompt text is shown with each model’s chat-template delimiters.

### A.15 latent position clusters across models

**Projection procedure.** We take the 100 concepts of the validation set and, for each concept, inject it separately at each of the ten candidate positions, repeating the whole procedure over 30 validation clusters; for every run we collect the residual state at the final prompt token, together with the corresponding clean run. For each concept and position, the residual states are averaged over the 30 clusters, as are the clean states, giving 1,000 injected points and one clean point per layer. These points enter a single PCA per layer, and we plot their leading three principal components in three dimensions. Each point is one concept injected at one position, averaged over clusters, and its colour indicates the injected position.

Figure 8 shows the residual-state PCA for all three evaluated model families. For each model, the four panels show the layer below the transition layer identified by the K-means analysis (Figure 2), the transition layer itself, the layer above it, and the model’s last layer. The transition layers are 24, 17, and 29 for Qwen3-4B-IT, LLaMA-3.1-8B-IT, and Gemma-3-12B-IT, respectively.

In every model, the injected states in the panel below the transition layer largely overlap, and the transition layer already shows ten separated position-specific clusters. K-means accuracy one layer below the transition is above chance but far below its value at the transition (Figure 2b), so weaker positional structure exists earlier; the clear separation, however, appears within a single layer in all three families rather than in one architecture only. The last-layer panels differ across models: the clusters remain well separated in Qwen3-4B-IT and Gemma-3-12B-IT, whereas in LLaMA-3.1-8BIT they partly remix, indicating that the positional code is most cleanly expressed near the transition layer rather than at the output.

Figure 8: **Final-token residual PCA around the transition layer in all three models.** Each colored point is one injected run at the final prompt token, colored by the perturbed candidate position; the black point is the clean run. Principal components are fitted separately for each panel. Panels are ordered transition layer minus one, transition layer, transition layer plus one, and the last layer.

### A.16 router-head output clusters across models

Figure 9 applies a three-dimensional PCA to the output of every attention head in the transition layer of each model (layers 24, 17, and 29 for Qwen3-4B-IT, LLaMA-3.1-8B-IT, and Gemma-3-12B-IT). Head outputs are averaged over the 30 validation clusters in the same way as in Appendix A.15, and the PCA is fitted on all points. Each panel shows the correctly answered injected points, colored by injected position, together with the clean point; a point counts as correct when the majority of its 30 cluster runs name the injected position. The router heads identified by head-wise patching in Section 4.3 (red frames) split into position-specific clusters, whereas the other heads in the same layer remain a single mixed cloud. The position-specific structure that appears in the residual stream at the transition layer (Appendix A.15) is thus concentrated in the outputs of a few heads.

Figure 9: **Three-dimensional PCA of attention-head outputs across models.** Each panel shows one attention head; red frames highlight the selected router heads. Colors distinguish injected indices, and black points mark clean references. Only correctly answered injected runs are shown.

(b) Qwen3-4B-IT, heads 16–31.

Figure 9: **Three-dimensional PCA of attention-head outputs across models (continued).** Qwen3-4B-IT, heads 16–31. Colors and red frames follow panel (a).

Figure 9: **Three-dimensional PCA of attention-head outputs across models (continued).** LLaMA-3.1-8B-IT, heads 0–15. Colors and red frames follow panel (a).

Figure 9: **Three-dimensional PCA of attention-head outputs across models (continued).** LLaMA-3.1-8B-IT, heads 16–31. Colors and red frames follow panel (a).

Figure 9: **Three-dimensional PCA of attention-head outputs across models (continued).** Gemma-3-12B-IT. Colors and red frames follow panel (a).

### A.17 uncertainty estimates

Tables 9–11 give 95% intervals for every value reported in Figure 3, Table 2, and Table 8. Intervals are computed from response counts aggregated per condition and, for the two tables, per ordered injection–readout position pair.

**Wilson intervals.** For each rate we report the 95% Wilson score interval. An injected-run rate pools 100 concepts _×_ 30 prompts _×_ 10 positions = 30 _,_ 000 trials per condition and label setting (270,000 for the 90 pairs _i̸_ = _j_ in each setting of Table 2 and in each arm of Table 8). Each bar of Figure 3 and each entry of Table 2 is the unweighted mean over the six label settings; because every setting contributes the same number of trials, this mean equals the rate pooled over the settings, and its interval is the Wilson interval of the pooled counts: 180,000 trials for most bars, 1,800,000 for the two bars that pool all 100 gate–router position pairs, and 1,620,000 for Table 2. This gives half-widths of at most 0 _._ 3 percentage points for every Figure 3 bar other than the unmodified clean run, and 0 _._ 002 in Table 8. These intervals describe sampling over trials, not how much a rate varies across label settings. The unmodified clean run depends on neither the concept nor the injection position, so its rate is determined by the test prompts; its intervals use _n_ = 30 per setting, 180 in Figure 3. For the pooled tables we additionally report a position-level bootstrap.

**Position bootstrap.** Table 2 and Table 8 pool over the 90 ordered position pairs. We resample the ten source positions _i_ with replacement, keep all nine pairs of every sampled position, and recompute the pooled rate (10,000 resamples, percentile intervals). For Table 2 and the Mean column of Table 8, the six label settings share each resample of positions. These intervals are deliberately conservative.

**Results.** The main comparisons hold under these intervals. In Figure 3a,b, the gate patch moves each rate well outside the interval of the unmodified run in all three models. In Figure 3d, the none rate with clean gate heads and injected router heads lies far above the interval of the unmodified injected run. In Table 2, output _j_ exceeds output _i_ under both intervals in every model. In Table 8, _→ j_ is the most frequent outcome for all three models when averaged over arms.

Table 9: **95% intervals for Figure 3.** Plotted rate (%) and its Wilson interval. Each rate is the mean over the six label settings of Table 1, which equals the rate pooled over them. Unless marked, a rate pools 180,000 trials (six settings _×_ 100 concepts _×_ 30 prompts _×_ 10 positions).^_∗_ Unmodified clean run: 180 prompts (30 per setting).^_†_ 1,800,000 trials (all 100 gate–router position pairs in each setting). **Bold** : the gate heads alone are patched.

||Qwe|n3-4B-IT|LLa|MA-3.1-8B-IT|Gem|ma-3-12B-IT|
|---|---|---|---|---|---|---|
|Condition|%|95% CI|%|95% CI|%|95% CI|
|(a)Gate off: nonerate|||||||
|Injected run|37.7|[37.5, 37.9]|25.3|[25.1, 25.5]|23.8|[23.6, 24.0]|
|Gate patch|81.5|[81.4, 81.7]|60.7|[60.5, 60.9]|80.7|[80.5, 80.9]|
|Target (clean run)^_∗_|89.4|[84.1, 93.1]|62.8|[55.5, 69.5]|87.8|[82.2, 91.8]|
|(b)Gate on: position rate |||||||
|Clean run^_∗_|10.6|[6.9, 15.9]|37.2|[30.5, 44.5]|12.2|[8.2, 17.8]|
|Gate patch|41.8|[41.5, 42.0]|69.3|[69.1, 69.6]|62.6|[62.3, 62.8]|
|Target (injected run)|62.3|[62.1, 62.5]|74.7|[74.5, 74.9]|76.2|[76.0, 76.4]|
|(c)Clean run: position rate|||||||
|Clean run^_∗_|10.6|[6.9, 15.9]|37.2|[30.5, 44.5]|12.2|[8.2, 17.8]|
|Gate clean_×_router injected|14.1|[13.9, 14.2]|42.7|[42.5, 43.0]|9.8|[9.6, 9.9]|
|Gate injected_×_router clean |36.2|[36.0, 36.4]|67.7|[67.4, 67.9]|42.7|[42.4, 42.9]|
|Gate injected_×_router injected^_†_|50.3|[50.2, 50.3]|73.7|[73.6, 73.7]|61.7|[61.7, 61.8]|
|(d)Injected run: nonerate|||||||
|Injected run|37.7|[37.5, 37.9]|25.3|[25.1, 25.5]|23.8|[23.6, 24.0]|
|Gate injected_×_router clean|49.3|[49.0, 49.5]|31.6|[31.4, 31.8]|38.2|[38.0, 38.4]|
|Gate clean_×_router injected^_†_|76.3|[76.3, 76.4]|53.9|[53.8, 54.0]|73.7|[73.6, 73.8]|
|Gate clean_×_router clean|83.4|[83.2, 83.6]|61.4|[61.2, 61.6]|79.4|[79.2, 79.5]|

Table 10: **95% intervals for Table 2.** Rate (%) averaged over the six label settings of Table 1, each pooling the 90 ordered pairs _i̸_ = _j_ (1,620,000 trials in total), with two intervals: _Wilson_ , a score interval over trials, and _Pos. boot._ , a bootstrap that resamples the ten gate-source positions _i_ together with all of their pairs, shared across the six settings. Other is the average rate per position other than _i_ and _j_ , and its intervals are those of the eight-position total divided by eight. Output _j_ exceeds output _i_ and Other under both intervals in every model.

|||Qwen3-4|B-IT|L|LaMA-3.1|-8B-IT|G|emma-3-|12B-IT|
|---|---|---|---|---|---|---|---|---|---|
|Outcome|%|Wilson|Pos. boot.|%|Wilson|Pos. boot.|%|Wilson|Pos. boot.|
|Output_i_|2.9|[2.9, 2.9]|[2.2, 3.7]|3.2|[3.2, 3.2]|[1.5, 5.8]|4.9|[4.9, 5.0]|[1.2, 10.3]|
|Output_j_|38.0|[37.9, 38.1]|[32.3, 42.6]|37.8 |[37.7, 37.9]|[35.4, 39.4]|40.1|[40.0, 40.1]|[31.0, 47.4]|
|Other|1.1|[1.1, 1.1]|[0.9, 1.3]|4.1|[4.1, 4.1]|[3.6, 4.4]|2.0|[2.0, 2.0]|[1.5, 2.3]|

Table 11: **95% position-bootstrap intervals for Table 8.** Each cell gives the proportion from Table 8 and, below it, an interval from resampling the ten injection positions _i_ with all of their readout positions _j̸_ = _i_ ; the Mean column resamples positions jointly across the six arms. Wilson half-widths are at most 0.002 for every a score interval over trials, and arm and at most 0.001 for the Mean.

|||Ordered label|s||Shuffled labels|l||
|---|---|---|---|---|---|---|---|
|Outcome|Digits|Letters|Words|Digits|Letters|Words|Mean|
|_Qwen3-4_|_B-IT_|||||||
|_→j_|0.480 [0.361, 0.587]|0.679 [0.560, 0.776]|0.704 [0.624, 0.769]|0.582 [0.491, 0.652]|0.585 [0.532, 0.631]|0.760 [0.734, 0.784]|0.632 [0.552, 0.695]|
|_→i_|0.011 [0.001, 0.027]|0.011 [0.005, 0.019]|0.029 [0.016, 0.045]|0.004 [0.001, 0.008]|0.018 [0.012, 0.024]|0.023 [0.016, 0.030]|0.016 [0.013, 0.020]|
|Other|0.004 [0.002, 0.005]|0.003 [0.002, 0.004]|0.022 [0.014, 0.031]|0.013 [0.009, 0.017]|0.031 [0.025, 0.035]|0.065 [0.053, 0.076]|0.023 [0.019, 0.027]|
|none|0.505 [0.396, 0.627]|0.307 [0.209, 0.427]|0.245 [0.176, 0.333]|0.401 [0.330, 0.494]|0.366 [0.315, 0.427]|0.152 [0.129, 0.179]|0.329 [0.263, 0.413]|
|_LLaMA-3_|_.1-8B-IT_|||||||
|_→j_ _→i_|0.759 [0.738, 0.784] 0.056 [0.028, 0.085]|0.530 [0.448, 0.601] 0.023 [0.012, 0.035]|0.680 [0.655, 0.704] 0.059 [0.031, 0.089]|0.829 [0.819, 0.843] 0.030 [0.013, 0.055]|0.676 [0.661, 0.693] 0.023 [0.017, 0.029]|0.725 [0.713, 0.739] 0.044 [0.030, 0.060]|0.700 [0.681, 0.717] 0.039 [0.032, 0.047]|
|Other|0.078 [0.059, 0.099]|0.009 [0.008, 0.011]|0.040 [0.025, 0.055]|0.130 [0.107, 0.149]|0.044 [0.037, 0.050]|0.189 [0.168, 0.207]|0.082 [0.071, 0.091]|
|none|0.107 [0.082, 0.143]|0.437 [0.359, 0.526]|0.221 [0.188, 0.266]|0.011 [0.008, 0.014]|0.258 [0.241, 0.273]|0.042 [0.034, 0.049]|0.179 [0.158, 0.207]|
|_Gemma-3_|_-12B-IT_|||||||
|_→j_|0.355 [0.264, 0.443]|0.524 [0.416, 0.609]|0.334 [0.252, 0.403]|0.322 [0.285, 0.353]|0.428 [0.379, 0.472]|0.199 [0.162, 0.229]|0.360 [0.300, 0.408]|
|_→i_|0.467 [0.383, 0.548]|0.170 [0.120, 0.231]|0.394 [0.315, 0.481]|0.295 [0.241, 0.349]|0.151 [0.116, 0.187]|0.181 [0.113, 0.269]|0.276 [0.231, 0.320]|
|Other|0.009 [0.004, 0.017]|0.005 [0.003, 0.008]|0.009 [0.005, 0.014]|0.078 [0.052, 0.113]|0.011 [0.008, 0.015]|0.219 [0.160, 0.270]|0.055 [0.042, 0.069]|
|none|0.169 [0.092, 0.281]|0.301 [0.217, 0.412]|0.263 [0.194, 0.363]|0.305 [0.255, 0.354]|0.411 [0.338, 0.490]|0.401 [0.366, 0.437]|0.308 [0.250, 0.380]|

### A.18 gate and router interventions under each label setting

Figure 3 and Table 2 report the unweighted mean over the six label settings of Table 1. Figures 10 and 11 show each setting separately, and Table 12 gives the cross-position patching of each setting. In every setting the gate and router heads are those selected under ordered digit labels, without reselection; ordered digits is the setting the heads were selected on, and the other five test transfer. In all eighteen model–setting combinations, the gate patch moves the response toward the target run (panels a, b), the gate intervention changes the response more than the router intervention (panels c, d), and output _j_ exceeds output _i_ .

Table 12: **Cross-position patching under each label setting.** Setup as in Table 2, whose values are the unweighted mean of these six tables. Values are percentages of all trials: output _i_ , output _j_ , Other (the average rate per position other than _i_ and _j_ , as in Table 2), and none.

#### (a) Ordered digits (head selection).

|(a) Ordered dig|its (|head|select|ion).|(b) Shul|ffle|ld digi|l ts.||
|---|---|---|---|---|---|---|---|---|---|
|Model|_i_|_j_|Other|none|Model|_i_|_j_|Other|none|
|Qwen3-4B-IT|1.6|28.0|0.3|68.0|Qwen3-4B-IT|2.1|27.4|1.0|62.3|
|LLaMA-3.1-8B-IT|1.4|68.0|2.3|12.5|LLaMA-3.1-8B-IT|5.4|34.1|7.4|1.2|
|Gemma-3-12B-IT|4.3|62.8|0.5|28.5|Gemma-3-12B-IT|5.0|35.2|3.3|33.3|
|(c) Ord|ere|d lette|rs.||(d) Shul|ffle|ld lette|l rs.||
|Model|_i_|_j_|Other|none|Model|_i_|_j_|Other|none|
|Qwen3-4B-IT|2.6|43.1|0.8|48.3|Qwen3-4B-IT|2.9|33.3|1.0|56.1|
|LLaMA-3.1-8B-IT|0.8|35.8|0.9|56.5|LLaMA-3.1-8B-IT|2.9|21.7|4.0|43.3|
|Gemma-3-12B-IT|3.0|47.1|0.3|47.8|Gemma-3-12B-IT|2.4|41.1|1.0|48.6|

|(e) Ordered|nu|mber|words|.|
|---|---|---|---|---|
|Model|_i_|_j_|Other|none|
|Qwen3-4B-IT|3.8|49.6|1.2|36.9|
|LLaMA-3.1-8B-IT|1.1|46.6|1.7|38.5|
|Gemma-3-12B-IT|7.7|39.2|0.3|50.6|

|(f) Shuffled number words.|
|---|

|Model|_i_|_j_|Other|none|
|---|---|---|---|---|
|Qwen3-4B-IT|4.3|46.5|2.6|28.7|
|LLaMA-3.1-8B-IT|7.5|20.7|8.3|5.3|
|Gemma-3-12B-IT|7.2|14.9|6.5|25.7|

Figure 10: **Gate and router interventions with ordered labels.** Panels as in Figure 3; gate and router heads are the ones selected under ordered digit labels.

Figure 11: **Gate and router interventions with shuffled labels.** Panels as in Figure 3; gate and router heads are the ones selected under ordered digit labels.

#### Bcomparison of the key and query terms of the qk decomposition

We first show why the analysis in Section 5 studies the score change ∆ _s_ . At the final prompt position, the attention weights are a softmax over the visible positions, _at_ = _e_^_st_ _/_^() _u_^(_esu_.Writing) _s_^_I_ = _s_^0 + ∆ _s_ gives where the second step divides the numerator and denominator by^() _u_^_es_ _u_^0 . Positions excluded by the attention mask are the same in both runs, so the identity is exact. The clean run involves no injection, so for a given prompt and head, _a_^0 is the same for every concept. By Equation 11, _a_^_I_ is then a function of ∆ _s_ alone, and it is unchanged when a constant is added to every ∆ _st_ . Differences in attention across concepts therefore arise only through differences in ∆ _s_ across positions.

Equation 1 splits the injection-induced change in the attention score into a key term ∆ _Ktq_^_I_ _/_^(_√_) _dh_ and a query term _Kt_^(0∆)^(_q/√_) _dh_ . With _Kt_ and _q_ the post-RoPE key at position _t_ and query at the final prompt position, substituting _q_^0 = _q_^_I_ _−_ ∆ _q_ gives

This is an identity rather than an approximation, so either term can be subtracted from the injected forward state on its own. This appendix reports the distributional and behavioural consequences of doing so, which motivate our focus on the key term in Section 5.

Holding the injected forward state fixed, we subtract either the query term or the key term from the final row of the attention scores in the selected gate heads, and measure _D_ KL( _p∥p−X_ ) in nats between the resulting distribution and the unablated one. The successor-position column first conditions the attention probabilities on the ten successor positions and compares the resulting relative distribution; the full-context column uses all visible positions. The output column re-runs the forward pass under the same score ablation and takes the argmax over the eleven candidate first tokens 0–9 and none, reporting the fraction of trials that name the injected position correctly. _C_ intro is the 100 validation concepts and _C_ nonintro the 100 low-accuracy concepts (Section 5); both are evaluated on the 30 test clusters, giving 30,000 trials per concept set. Native rates therefore differ from the ordered-digit column of Table 1, which uses the test concepts; KL values are computed per trial and head and then averaged with equal weight.

Table 13: **Ablating the key term versus the query term.** Attention-distribution change and reporting accuracy under each ablation, by model and concept set.

|||KL:_−_que|ry /_−_key|Corre|ct-position rate|
|---|---|---|---|---|---|
|Model|Concept set|Successor|Full context|Native|_−_query /_−_key|
|Qwen3-4B-IT|_C_intro|0_._0197/0_._3108|0_._0351/0_._1581|48_._28%|43_._62%/32_._07%|
||_C_nonintro|0_._0025/0_._0384|0_._0043/0_._0300|1_._67%|1_._49%/0_._96%|
|LLaMA-3.1-8B-IT|_C_intro|0_._0093/0_._0585|0_._0134/0_._0411|69_._34%|67_._27%/65_._38%|
||_C_nonintro|0_._0041/0_._0322|0_._0079/0_._0266|20_._97%|19_._03%/18_._09%|
|Gemma-3-12B-IT|_C_intro|0_._0519/0_._5847|0_._1205/0_._3979|83_._67%|83_._07%/58_._23%|
||_C_nonintro|0_._0168/0_._1751|0_._0277/0_._1023|18_._30%|17_._38%/6_._47%|

_Reading._ Each KL cell gives the divergence caused by removing the query term and by removing the key term, respectively; larger values indicate a larger change in the attention distribution. The last column gives the correct-position rate after each ablation, against the native rate in the preceding column.

Ablating the key term produces the larger distributional change in every model and concept set, both among the ten successor positions and over the full context; the full-context key/query ratio of the KL is 3 _._ 08–7 _._ 06 _×_ . The output measure orders the two terms the same way in every row. The keyterm effect exceeds the query-term effect in every row: for _C_ intro, removing the key term lowers the correct-position rate by 3 _._ 96–25 _._ 44 percentage points, against 0 _._ 60–4 _._ 66 points for the query term. This supports concentrating the structural analysis on the key term.

A change in distribution is not the same as a change in norm. Over the full context the two terms are of comparable size (query/key = 0 _._ 63–1 _._ 63), and the intro/non-intro ratio of the query term (1 _._ 30– 2 _._ 50 _×_ ) is in fact larger than that of the key term (1 _._ 13–1 _._ 45 _×_ ). The term _Kt_^(0∆)^(_q_is a broad, shallow) redistribution spread across the context (effective number of positions 84–93; enrichment over the candidate region 1 _._ 52 _×_ ) and does not concentrate on particular positions.

Two remarks on scope. The decomposition assigns the second-order cross term ∆ _Kt_ ∆ _q_ to the key term, so “key term” throughout denotes the cross-term-inclusive quantity; for Gemma-3-12BIT _∥_ ∆ _q∥/∥q_^0 _∥_ = 0 _._ 256, so this term is not negligible in that model. The conditional KL over the ten successor positions does not reflect the total attention mass those positions receive, and the behavioural ablation acts on the entire score row of the selected heads, so it cannot by itself establish a causal role for the successor positions specifically. The _C_ nonintro native rate of 1 _._ 67% for Qwen34B-IT is subject to a floor effect and its percentage-point changes should not be read alongside the corresponding intro row.

Section 5 factors the key response at the ten successor positions through the SVD ∆ _K_ next =  _r_^_σrurv_ _r_^(_⊤_.With)^(_s_=∆)^_K_next^(_qI/√_) _dh_ the full key response and _Ui_ 1 the entry of _u_ 1 at the target position _ti_ , we define

_R_ 1 is the magnitude of the first-mode response across the ten successor positions, and _U_^(¯) _i_ 1 is the share of it that lands on _ti_ with a consistent sign. The first mode accounts for all but 3% of the target response ¯ _si_ = E[ _si_ ]. Table 14 gives the group statistics behind the QK columns of Table 3.

Table 14: **QK responses of introspective and non-introspective concepts.** Group statistics and between-set ratios in gate heads; the ratio rows extend the QK columns of Table 3.

|Concept set Query norm _∥q_^_I__∥_|Magnitude _σ_1|Readout _|v_^(_⊤_) 1 ^(_qI|_)|First mode _R_1|Target loading ¯_Ui_1|Target response ¯_si_|
|---|---|---|---|---|---|
|(a) Qwen3-4B-IT||||||
|_C_intro 14_._2870|9_._7021|2_._0431|1_._8697|0_._8847|1_._7000|
|_C_nonintro 14_._2356|7_._9384|1_._1333|0_._8099|0_._3832|0_._3203|
|_Intro / Non-intro_ 1_._00_×_|1_._22_×_|1_._80_×_|2_._31_×_|2_._31_×_|5_._31_×_|
|(b) LLaMA-3.1-8B-IT||||||
|_C_intro 12_._5969|9_._4412|1_._1510|0_._9493|0_._6945|0_._6774|
|_C_nonintro 12_._6249|8_._7674|0_._9611|0_._7252|0_._5165|0_._3851|
|_Intro / Non-intro_ 1_._00_×_|1_._08_×_|1_._20_×_|1_._31_×_|1_._34_×_|1_._76_×_|
|(c) Gemma-3-12B-IT||||||
|_C_intro 18_._5425|19_._7339|1_._9243|2_._4228|0_._8841|2_._1398|
|_C_nonintro 18_._7912|17_._3643|1_._2386|1_._3325|0_._6950|0_._9144|
|_Intro / Non-intro_ 0_._99_×_|1_._14_×_|1_._55_×_|1_._82_×_|1_._27_×_|2_._34_×_|

_Definitions. C_ intro and _C_ nonintro are the introspective and non-introspective concept sets; all quantities follow Equation 13. Entries are group means, except _U_^(¯) _i_ 1, which is a ratio of group means and mixes concentration on the target position with sign consistency across trials (even spread gives 0 _._ 32). _Ratios._ Intro / Non-intro divides the displayed statistics, rounded to two decimals. Because these are ratios of means, columns do not compose exactly; _R_ 1 and _U_^(¯) _i_ 1 compose to ¯ _si_ only up to the residual modes. Ratios are descriptive, not significance tests.

C OUTPUT DECOMPOSITION AND MODAL ANALYSIS OF THE OV CIRCUIT

This appendix gives details of the OV analysis in Section 5. Superscripts 0 and _I_ denote the unperturbed and injected states, respectively, with ∆ _V_ = _V_^_I_ _− V_^0 and ∆ _a_ = _a_^_I_ _− a_^0 . We omit sample, layer, and attention-head indices below.

This analysis uses the same two concept sets as the QK analysis in the main text: _C_ intro and _C_ nonintro. Each contains 100 concepts. These names denote concept sets partitioned by reporting performance after injection; they do not imply that every injection within a set succeeds or fails. Both concept sets are evaluated on 30 held-out test clusters, with each concept injected at all ten candidate positions in every cluster (300 injections per concept). We analyze the 32 gate heads selected under the gateon condition; here, gate-on specifies only the head set. Each metric is first averaged over injection positions, candidate-token clusters, and gate heads within each concept, and then averaged within each concept set.

Let ∆ _o_ = _o_^_I_ _− o_^0 denote the injection-induced change in the attention-head output at the final prompt position _T_ . Since _o_ = _WOV_^(_⊤_) _a_ , substituting _a_^0 = _a_^_I_ _−_ ∆ _a_ gives the decomposition of Equation 1:

### C.1 detailed ov statistics

Section 5 decomposes the ∆ _V_ term of ∆ _o_ through the SVD of _M_ = _WO_ (∆ _V_ )^(_⊤_) =^() _k_^_µkykx_ _k_^(_⊤_,) computed separately for each trial and gate head over all _n_ positions of the context, so that _Ma_^_I_ =  _k_^(_µk_()^_x_ _k_^(_⊤aI_))^(_yk_(Section 5).Table 15 lists the attention and output statistics.Besides the norms, it) reports two energy fractions of the five leading modes: the matrix energy _η_ 5 =^() _k≤_ 5^_µ_ _k_^2^(_/_) _k_^_µ_ _k_^2 and the output energy _ρ_ 5 =^() _k≤_ 5^_µ_ _k_^(2()^(_x⊤_) _k_^(_aI_)2)^(_/∥MaI∥_2.Becausetheoutputdirections)^_yk_are orthonormal, _ρ_ 5 is the fraction of the squared output norm carried by these modes. Table 16 lists, for each of the five leading modes, the size _µk_ and the alignment _|_ cos( _a_^_I_ _, xk_ ) _|_ = _|x_^(_⊤_) _k_^(_aI|/∥aI∥_; we) take the absolute value because the sign of a singular vector is arbitrary.

Table 15: **Attention and output statistics of the** ∆ _V_ **term in gate heads.** Group means over the full context; energies are percentages.

||Atten|tion|Change m|atrix_M_||Output_Ma_^_I_||
|---|---|---|---|---|---|---|---|
|Concept set|_∥a_^_I__∥_|_∥_∆_a∥_|_∥M∥F_|_η_5|_∥Ma_^_I__∥_|Top-5 norm|_ρ_5|
|(a) Qwen3-4B-IT||||||||
|_C_intro|0.367|0.085|19.3|80.1|0.912|0.841|68.7|
|_C_nonintro|0.360|0.036|16.6|82.0|0.313|0.236|49.3|
|_Intro / Non-intro_|1_._02_×_|2_._40_×_|1_._16_×_|0_._98_×_|2_._92_×_|3_._56_×_|1_._39_×_|
|(b) LLaMA-3.1-8B|-IT|||||||
|_C_intro|0.367|0.049|6.16|78.2|0.127|0.102|53.3|
|_C_nonintro|0.361|0.036|5.80|79.5|0.085|0.061|47.4|
|_Intro / Non-intro_|1_._02_×_|1_._37_×_|1_._06_×_|0_._98_×_|1_._49_×_|1_._66_×_|1_._13_×_|
|(c) Gemma-3-12B-I|T|||||||
|_C_intro|0.457|0.162|25.9|76.7|1.969|1.852|81.5|
|_C_nonintro|0.430|0.073|22.9|79.1|0.770|0.628|58.2|
|_Intro / Non-intro_|1_._06_×_|2_._23_×_|1_._13_×_|0_._97_×_|2_._56_×_|2_._95_×_|1_._40_×_|

Table 16: **Size and alignment of the five leading modes of** _M_ **in gate heads.** Group means over the full context.

||||Size_µk_||||Alignme|nt_|_cos(_a_|^_I__, xk_)_|_||
|---|---|---|---|---|---|---|---|---|---|---|
|Concept set|_k_ = 1|2|3|4|5|_k_ = 1|2|3|4|5|
|(a) Qwen3-4B-IT|||||||||||
|_C_intro|11.3|8.24|6.45|5.33|4.48|0.149|0.126|0.104|0.097|0.087|
|_C_nonintro|10.1|7.05|5.47|4.43|3.68|0.039|0.059|0.059|0.061|0.058|
|_Intro / Non-intro_|1_._12_×_|1_._17_×_|1_._18_×_|1_._20_×_|1_._22_×_|3_._85_×_|2_._16_×_|1_._76_×_|1_._58_×_|1_._49_×_|
|(b) LLaMA-3.1-8|B-IT||||||||||
|_C_intro|3.54|2.60|2.02|1.65|1.41|0.051|0.052|0.054|0.059|0.052|
|_C_nonintro|3.41|2.46|1.89|1.53|1.30|0.035|0.037|0.042|0.050|0.046|
|_Intro / Non-intro_|1_._04_×_|1_._05_×_|1_._07_×_|1_._08_×_|1_._08_×_|1_._48_×_|1_._40_×_|1_._28_×_|1_._20_×_|1_._13_×_|
|(c) Gemma-3-12B|-IT||||||||||
|_C_intro|14.5|10.8|8.60|7.18|6.13|0.222|0.165|0.133|0.111|0.101|
|_C_nonintro|13.3|9.59|7.50|6.14|5.19|0.076|0.085|0.079|0.075|0.073|
|_Intro / Non-intro_|1_._09_×_|1_._13_×_|1_._15_×_|1_._17_×_|1_._18_×_|2_._90_×_|1_._93_×_|1_._68_×_|1_._49_×_|1_._37_×_|

In all three models, the two groups have nearly the same attention norm ( _≤_ 1 _._ 06 _×_ ), change-matrix norm ( _≤_ 1 _._ 16 _×_ ), and matrix energy in the five leading modes. The sizes _µk_ differ by at most 1 _._ 22 _×_ at every _k_ , whereas the alignment is higher for _C_ intro at every _k_ , most strongly at _k_ = 1 (1 _._ 48–3 _._ 85 _×_ ). As a result, the five leading modes carry a larger share of the output for _C_ intro ( _ρ_ 5 of 53–82% against 47–58%), and the output norm is 1 _._ 49–2 _._ 92 _×_ larger. For every quantity in Tables 15 and 16, the 95% concept-level bootstrap interval of the between-group difference excludes zero (5,000 resamples, conditional on the 30 evaluation clusters, without multiple-comparison correction). Ratios of group means are descriptive and do not compose across columns.

### C.2 causal ablation of the leading ov modes

We test whether the leading modes support correct-position reports by subtracting their output contributions in the injected run. For each paired clean and injected example and each gate head, we form _M_ = _WO_ (∆ _V_ )^(_⊤_) over all _n_ context positions and order its modes by decreasing singular value _µk_ . The leading-five contribution and its complement are

We separately subtract _c≤_ 5, _c>_ 5, or the entire content-change term _Ma_^_I_ from the attention output projection at the final prompt position, before any attention-output normalization. The intervention is applied jointly to the 32 gate heads. These vectors are fixed from the unablated paired runs; downstream computation is left free. Thus, the modes are defined from the full context, not only the ten candidate injection positions.

Both blocks of Table 17 use all 30 evaluation clusters, with 100 concepts per set and ten injection positions per concept and cluster: 30,000 paired trials per set, or 60,000 per model. The left block removes the two terms of Equation 1 as full vectors. The right block removes modes of _M_ . Correctposition rate is the fraction of trials in which the highest-scoring candidate among the ten positions and none is the injected position; a none response counts as incorrect. Table 18 additionally reports the mean probability of the correct position token under the full-vocabulary softmax, without renormalizing over these eleven candidates.

Table 17: **Correct-position rate (%) after removing parts of the gate-head output in the injected run.** _−_ ∆ _a_ and _−_ ∆ _V_ remove the corresponding terms of Equation 1; _−_ top 5 and _−_ others remove the five leading modes of _M_ or all remaining modes.

|||T|erms, 30|clusters||Mo|des of_M_,|30 cluster|s|
|---|---|---|---|---|---|---|---|---|---|
|Model|Set|Injected|_−_∆_a_|_−_∆_V_|_−_both|Injected|_−_top 5|_−_others|_−_∆_V_|
|Qwen3-4B-IT|_C_intro|48.3|38.8|4.4|1.4|48.3|5.1|44.6|4.4|
||_C_nonintro|1.7|1.3|0.7|0.7|1.7|0.7|1.5|0.7|
|LLaMA-3.1-8B-IT|_C_intro|69.3|67.8|54.3|49.8|69.3|57.4|67.1|54.3|
||_C_nonintro|21.0|19.3|13.8|11.9|21.0|15.3|19.0|13.8|
|Gemma-3-12B-IT|_C_intro|83.7|74.1|13.9|2.6|83.7|13.5|81.2|13.9|
||_C_nonintro|18.3|11.0|1.5|0.4|18.3|1.8|15.4|1.5|

Table 18: **Mean correct-position token probability (%) after OV mode ablation.** Full-context modes, all 30 clusters, and 30,000 paired trials per concept set and model. Probabilities use the full vocabulary.

|Model|Set|Injected|_−_top 5|_−_others|_−_∆_V_|
|---|---|---|---|---|---|
|Qwen3-4B-IT|_C_intro|47.99|5.12|44.39|4.40|
||_C_nonintro|1.66|0.67|1.47|0.65|
|LLaMA-3.1-8B-IT|_C_intro|40.22|33.69|39.95|32.93|
||_C_nonintro|13.37|11.80|12.96|11.32|
|Gemma-3-12B-IT|_C_intro|83.12|13.53|80.63|13.91|
||_C_nonintro|18.04|1.83|15.15|1.43|

For _C_ intro, removing the leading five modes lowers correct-token probability from 47.99% to 5.12% in Qwen3-4B-IT and from 83.12% to 13.53% in Gemma-3-12B-IT. The reduction is smaller in

LLaMA-3.1-8B-IT, from 40.22% to 33.69%. In each model, removing the remaining modes produces a much smaller probability decrease, whereas removing the leading five yields a result close to removing the entire ∆ _V_ term. Correct-position rates follow the same pattern. These interventions support a causal role for the leading modes in correct-position reporting, with substantial differences in effect size across models; their effects need not add because downstream computation is nonlinear.

The accuracy decreases also involve fewer numeric reports. For _C_ intro, the rate of selecting any position rather than none falls from 49.66% to 6.86% in Qwen3-4B-IT, from 81.04% to 67.58% in LLaMA-3.1-8B-IT, and from 84.46% to 14.16% in Gemma-3-12B-IT after leading-five removal. The intervention therefore affects whether a position is reported as well as whether it is correct. The native correct-position rate of _C_ nonintro in Qwen3-4B-IT is near floor, limiting interpretation of its small absolute changes.

**Varying the number of removed modes.** Table 19 extends the same intervention to _c≤k_ =  _kj_ =1^(_µj_()^_x_ _j_^(_⊤aI_))^_yj_for^(_k_=1)^_,_2^_,_5^(_,_10.Allsettingsusethesame30clusters,concepts,positions,) and gate heads; the new runs reproduce the previous unablated predictions and correct-token probabilities exactly. Increasing _k_ from five to ten changes mean correct-token probability by less than one percentage point in either concept set for every model, although the effect is not strictly monotonic in Gemma-3-12B-IT.

Table 19: **OV ablation as a function of the number of removed modes.** Modes of the full-context matrix _M_ are ordered by singular value. Each entry uses all 30 clusters and 30,000 trials per concept set and model. All values are percentages; probabilities use the full-vocabulary softmax. _−_ ∆ _V_ removes the entire content-change term.

|Model|Set|Injected|_−_top 1|_−_top 2|_−_top 5|_−_top 10|_−_∆_V_|
|---|---|---|---|---|---|---|---|
|(a) Mean correct-pos|ition token pr|obability||||||
|Qwen3-4B-IT|_C_intro|47.99|24.89|10.28|5.12|4.42|4.40|
||_C_nonintro|1.66|1.06|0.78|0.67|0.64|0.65|
|LLaMA-3.1-8B-IT|_C_intro|40.22|38.57|36.64|33.69|33.16|32.93|
||_C_nonintro|13.37|13.16|12.82|11.80|11.46|11.32|
|Gemma-3-12B-IT|_C_intro|83.12|54.86|27.82|13.53|14.24|13.91|
||_C_nonintro|18.04|10.94|5.53|1.83|1.51|1.43|
|(b) Correct-position|rate|||||||
|Qwen3-4B-IT|_C_intro|48.28|25.07|10.28|5.15|4.42|4.40|
||_C_nonintro|1.67|1.06|0.81|0.68|0.64|0.65|
|LLaMA-3.1-8B-IT|_C_intro|69.34|66.72|64.05|57.42|55.37|54.27|
||_C_nonintro|20.97|19.97|18.61|15.26|13.98|13.81|
|Gemma-3-12B-IT|_C_intro|83.67|55.82|28.19|13.47|14.17|13.88|
||_C_nonintro|18.30|11.11|5.55|1.83|1.52|1.45|
