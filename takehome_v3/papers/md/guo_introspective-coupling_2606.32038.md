# Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision

- **Authors:** Zifan Carl Guo, Laura Ruis, Jacob Andreas, Belinda Z. Li
- **Date:** 2026-06-30
- **Source:** https://arxiv.org/abs/2606.32038
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `guo_introspective-coupling_2606.32038`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Zifan Carl Guo^* Laura Ruis Jacob Andreas Belinda Z. Li

MIT EECS

## Abstract

When does training language models (LMs) to generate explanations of their predictions yield faithful introspection, rather than superficial imitation? We study LMs trained to explain which features of their inputs influenced their behavior, using models’ counterfactual behavior on modified inputs as supervision. Surprisingly, we find that LMs trained on fixed counterfactual explanations derived from earlier checkpoints of themselves, or even from behaviorally similar models in different families, frequently produce explanations more faithful to _their own current behaviors_ than to those of their training targets. This “introspective” coupling between LM explanations and behaviors occurs when training explanations remain sufficiently correlated with current behaviors over the course of training, even as behaviors themselves shift. We also show that introspective coupling tracks behavior shifts: when explanation training is provided concurrently with other post-training objectives, explanations track those shifts without requiring updated supervision. This phenomenon appears in multiple tasks, including sycophancy and refusal, and is robust to label noise. Overall, our results show that even fixed datasets of counterfactual explanations can provide scalable and generalizable post-training signal for introspection.

## 1 Introduction

Methods for training language models (LMs) to faithfully articulate the reasons behind their predictions and decisions offer a promising avenue for understanding LM behavior. Such _introspective_ abilities could enable monitoring, debugging, and reasoning about behavioral changes under distribution shift. Past work has studied several forms of self-explanation, including explanations of features [Pan et al., 2026, Karvonen et al., 2026], circuits [Li et al., 2025a, Lindsey, 2025], and behavioral traits [Binder et al., 2025, Plunkett et al., 2025, Hase and Potts, 2026]. Yet a central challenge is ensuring that LM-generated explanations track an LM’s own behavior, rather than merely imitating plausible explanation patterns.

Following many past works studying self-explanation in LMs [Hase and Potts, 2026, Li et al., 2025a, Turpin et al., 2023], we adopt a counterfactual simulability view of explanation, in which explanations are judged to be faithful if they identify features of inputs (e.g. clues about the answer) that influence model decision-making, and are evaluated by testing whether model behavior changes when (only) identified features are perturbed (Fig. 1).

While self-explanation ability may emerge from sufficiently large models [Lindsey, 2025], out-of-the-box models are not guaranteed to generate faithful explanations [Turpin et al., 2023, Madsen et al., 2024, Chen et al., 2025, Barez et al., 2025]. This has motivated a line of work that explicitly trains models to generate explanations. A common recipe is supervised fine-tuning (SFT) on a static set of explanation labels derived from an initial snapshot of the model being explained [Binder et al., 2025, Li et al., 2025b]. This approach has an underexamined limitation: as an LM is trained to generate explanations, its behavior on non-explanation inputs may drift, so the model being trained learns to explain the behavior of the earlier checkpoint, rather than its own, current behavior. However, a faithful self-explanation must track the model’s current behavior rather than its original behavior before explanation training.

> *Preprint. Correspondence to carlguo@mit.edu.

Figure 1: **Method overview.** (A) First, we sample behaviors _B_ ( _M_ 0) from the base model _M_ 0 on inputs _x_ , _x\C_ and construct labels _E_ ( _M_ 0) explaining those behaviors. (B) Next, we fine-tune _M_ 0 to produce these explanations, yielding _M_ reg. (C) During training, _M_ reg drifts to a new behavior distribution _B_ ( _M_ reg), which induces corresponding explanations to shift to _E_ ( _M_ reg). (D) We find that _M_ reg’s predicted explanations track this behavior change: they better match explanations of _M_ reg’s own behavior (Self) despite training on explanations of the _M_ 0 (Orig). We refer to this Self _>_ Orig as _introspective coupling_ .

In this paper, we make the surprising observation that training models to generate explanations of their earlier behavior, while regularizing behaviors themselves toward an earlier checkpoint, _in fact causes them to explain their current behavior more faithfully than the checkpoint from which the explanations were derived_ (Fig. 1). On test data, regularized explanation training reduces drift relative to unregularized fine-tuning, but does not eliminate it entirely. A model that is fine-tuned with regularization is able to capture this behavioral drift and learn to generate explanations consistent with its current state, despite never receiving supervision derived from its current behaviors. This effect disappears without regularization.

More formally, we call this effect _Self > Orig_ , where:

A model demonstrates **Self > Orig** if its explanations more faithfully predict its current behavior than the original behavior used to generate its training labels.

We interpret Self > Orig as evidence of _introspective coupling_ :

**Introspective Coupling** : a phenomenon where a model learns to couple its explanations to its own current behavior rather than merely reproducing static explanation training targets.

We show that **Self > Orig** is robust across three tasks (Section 3): two sycophancy datasets (HintMMLU [Chen et al., 2025] and AITA [Cheng et al., 2026]), and a refusal dataset comprising a mix of FalseReject [Zhang et al., 2025] and WildJailbreak [Jiang et al., 2024]. In each case, regularized training produces models whose explanations are more faithful to their _current_ behavior than to the behavior of the original model that generated their training labels.

|_B_(_M_) _E_(_M_)|dataset of instance-level behaviors_{_(_x_,_yM_)_}_,_yM ∼M_(_· | x_) dataset of ground-truth meta-level explanations_{_(_q_,_eM_)_}_in question-answer pairs; each explana- tion_eM_ is deterministically constructed to describe behaviors in_B_(_M_)|
|---|---|
|_M_0 _M_[_D_0,_D_1, . . .]|base model (start of training) model fine-tuned from_M_0 on a mixture of datasets_D_0,_D_1, . . .|
|_A →B_|evaluate explanations predicted by model-_A_, ˆ_e ∼A_(_· | q_), against ground truth explanation of model_B_,_E_(_B_)|
|Orig|the eval pair _M →M_0: predictions from _M_ scored against the base model’s ground-truth explanations_E_(_M_0).|
|Self|the eval pair_M →M_: predictions_M_scored against_M_’s own ground-truth explanations_E_(_M_).|

Table 1: Notation for explanation training and evaluation.

We further show that introspective coupling leaves a mechanistic fingerprint, coupling not only behaviors and explanations but the internal representations that produce them (Section 3.3).

We characterize _when_ introspective coupling emerges (Section 4), finding that explanation training data must remain similar to the model’s _current, online_ behavior—even if explanations diverge from the model’s _original_ behavior by up to 50%. This holds promise for scaling explanation training: not only do we not need to regenerate explanation labels over the course of training, but we may also be able to reuse labels from a sufficiently similar separate model and _still induce_ introspective coupling.

Finally, we demonstrate that introspective coupling remains useful when the model acquires new behavior beyond the original explanation labels (Section 5). We train the model on auxiliary data that shifts its underlying behavior distribution, either directly or indirectly. We find that explanation shifts track behavioral shifts. This has promising consequences for integrating introspection training into post-training pipelines: models’ self-explanations track behavioral drifts and new behaviors induced by auxiliary post-training supervision.^1

## 2 Methods

### 2.1 (Counterfactual) Explanation Construction

Following past work, we focus on counterfactual explanations: how behaviors change under an edit to its input (e.g. removal of a hint) [Li et al., 2025a, Hase and Potts, 2026]. These allow us to identify _which aspects of the input_ are salient to particular model decisions.

Let a **cue** _C ⊆ x_ denote a contiguous span of an input _x_ hypothesized to causally influence a model’s behavior. We write _x\C_ for the **cue-ablated input** obtained by removing _C_ from _x_ , and refer to the original _x_ as the **cue-included input** . In general, we are interested in enabling LMs to answer questions of the form:

_[x]. If the cue were removed, how would the assistant’s answer change?_

with answers of the form:

_The response [would/would not] change to <M_ ( _x\C_ ) _>._

We train models to produce these answers via supervised fine-tuning (SFT). We use notation shown in Table 1 throughout. As illustrated in Figure 1, our method has four steps:

> 1We refer to “introspection” in the context of the operationalization above where models exhibit **behavioral** introspection that matches its own object-level outputs. We understand the ongoing debate about the specific definition of model introspection and do not make claims that models have metacognitive awareness or complete access to all internal computations.

1. _M_ 0 **explanation construction (§2.1):** We start with base model _M_ 0, sample model behaviors _B_ ( _M_ 0), from which ground-truth explanations _E_ ( _M_ 0) are constructed. This is shown in the running example in Figure 1: _B_ ( _M_ 0) is a set of input-output pairs from _M_ 0:

**Input 1: with hint Input 2: no hint** Largest planet is? Largest planet is? (A) Mars (A) Mars (B) Jupiter (B) Jupiter Hint: A Hint: A **Model answer: A Model answer: B**

- _E_ ( _M_ 0) is constructed from the set above as “behavior changes from A to B when the hint is removed”.

2. **Explanation training (§2.2):** We train _M_ 0 on explanations _E_ ( _M_ 0) while regularizing its behaviors towards _B_ ( _M_ 0), obtaining a model:

3. _M_ reg **explanation construction (§2.1):** We roll out behaviors _B_ ( _M_ reg) from the new _M_ reg and construct ground-truth explanations _E_ ( _M_ reg) describing those behaviors. In Figure 1, _B_ ( _M_ reg) is the pair:

**Input 1: with hint Input 2: no hint** Largest planet is? Largest planet is? (A) Mars (A) Mars (B) Jupiter (B) Jupiter Hint: A Hint: A **Model answer:** **_A** _→_ **B** after training **Model answer: B** unchanged

- and _E_ ( _M_ reg) is constructed from above as “behavior remains unchanged as B” when the hint is removed.

4. **Measuring introspective coupling (§2.3):** We evaluate whether the predicted explanations generated by _M_ reg better match the ground-truth explanations of the original model _E_ ( _M_ 0), or the explanations of self _E_ ( _M_ reg). As shorthand, we denote evaluating _M_ reg’s predictions on _M_ 0’s explanations as _M_ reg _→M_ 0 (or Orig) and evaluating _M_ reg’s predictions on _M_ reg’s explanations as _M_ reg _→M_ reg (or Self). **When Self > Orig, we say that a model exhibits introspective coupling.** In Figure 1: _M_ reg’s output matches _E_ ( _M_ reg) but not _E_ ( _M_ 0), and thus, introspective coupling is observed.

### 2.2 Explanation Training

Given a model _M_ , we perform SFT training on explanation data from base model _E_ ( _M_ 0) while regularizing behaviors to be close to _B_ ( _M_ 0). We use cross-entropy to train explanations and KL divergence to regularize behaviors. Formally, the objective is:

### 2.3 Evaluation Metrics

We evaluate all models on two main metrics:

**Explanation Exact Match (EM)** scores how well the LM _M_ has learned to explain a target model _M_^(_′_) , where _M_^(_′_) = _M_ 0 for Orig and _M_^(_′_) = _M_ reg for Self. Specifically, we measure exact match between the explainer _M_ ’s prediction _e_ ˆ^(()^(_M_)) , against the ground-truth explanation _E_ ( _M_^(_′_) ) for target model _M_^(_′_) :

where _e_ ˆ^(()^(_M_)) _∼M_ ( _· | q_ ) is the explainer model’s predicted response to a meta-question _q_ (“if the cue were removed, how would your answer change?”).

**Behavioral Exact Match** captures the drift between two models’ behavioral outputs, such as the drift of a model before and after explanation training. Let _D_ = _{_ ( _x_ , _C_ ) _}_ be a held-out set of inputs _x_ and cue spans _C_ , and let _yM_ ( _x_ ) _∼M_ ( _· | x_ ) denote _M_ ’s label on _x_ , and _yM_ ( _x\C_ ) denote the label on _x\C_ . We measure agreement between the trained model _M_ and original model _M_ 0 on both versions of each input:

In the metric above, both cue-ablated and cue-included behavior must match. Fine-grained metrics are in Section B.1.

## 3 Characterizing Introspective Coupling

### 3.1 Experiment Setup

**Models.** Our primary model _M_ 0 throughout the paper is **Qwen3-8B** [Yang et al., 2025]. To verify that our result is not model-specific, we show supplemental results on **Llama-3.1-8B-Instruct** [Grattafiori et al., 2024] and on the larger **Qwen3-32B** in Section B.6. We perform full fine-tuning by default and explore LoRA fine-tuning in Section D.1.

**Sycophancy Datasets.** Modern language models often exhibit sycophancy — over-agreeableness with the user — at the expense of accuracy [Sharma et al., 2024, Wei et al., 2024]. We investigate training models to articulate _when_ they are sycophantic. Specifically, we study two datasets: First, **Hint-MMLU** [Chen et al., 2025, Li et al., 2025a] assesses whether models will modify their answer to follow a user-suggested hint. Here, _x_ is an MMLU multiple-choice question with an injected _C_ =“Hint: A” string (see Figure 1). Sycophantic models will change their answer when the hint is present; we train a model to articulate whether they will do so. Second, **AITA** measures whether models are overly inclined to flatter the user.^2 This dataset consists of morally ambiguous stories from Reddit’s _r/AmItheAsshole_ . Here, _x_ is a Reddit post and _y_ is a judgment of the user’s moral character _∈{_ “Not the Asshole (NTA)”, “You’re the Asshole (YTA)” _}_ . _C_ is a system prompt framing the story in first- or third-person. In this case, sycophantic models will change their judgment to be more favorable if the story were written in first-person. We train models to articulate when they will change their judgment based on the system prompt.

**Refusal Datasets.** Understanding when LMs refuse requests allows us to debug over- or under-refusals. We specifically investigate how the user’s presented _role_ interacts with refusal. The user’s role is a popular attack surface—models may be more susceptible to answering harmful queries for users who are fiction writers, or to over-refusing users who are kindergarten teachers. Thus, we would like to train models to articulate whether and how their refusal hinges on a stated role. _x_ is a user prompt and _C_ is a system prompt that frames the user in different roles.^3 We measure _y ∈{_ refuse, comply _}_ by sampling a free-form response and scoring it with an LLM judge. Note that this setting is much harder than Hint-MMLU and AITA: the behavior label is inferred from free-form generation rather than constructed deterministically from a single answer token, so the explanation target is a higher-level property of the response. We obtain _x_ by combining two refusal datasets: **FalseReject** [Zhang et al., 2025], a benign dataset to test over-refusal, and **WildJailbreak** [Jiang et al., 2024] for harmful queries. We generate roleplay system prompts _C_ with an LLM.

> 2The dataset can be found here: https://huggingface.co/datasets/OsamaBsher/AITA-Reddit-Dataset.

Figure 2: Self _>_ Orig emerges only with regularization, across three counterfactual explanation tasks: Hint-MMLU, AITA, and Refusal. **Left block** ( **a** , **b** ): the regularized model _M_ reg. **Right block** ( **c** , **d** ): the unregularized model _M_ unreg. **(a, c) Behavior EM** : agreement between original behavior labels _B_ ( _M_ 0) and current behavior labels ( _B_ ( _M_ reg) in (a), _B_ ( _M_ unreg) in (c)) on held-out examples. **(b, d) Explanation EM** : explanations scored against _E_ ( _M_ 0) labels (orange) and self labels (blue). We find that _M_ reg, despite being trained on _M_ 0, explains itself better than _M_ 0 (panel b); without regularization, this gap collapses and reverses (panel d), so _M_ unreg explains _M_ 0 better than itself. Note that much of _M_ reg’s apparent ability to explain _M_ 0 comes from the subset where _M_ 0 and _M_ reg _agree_ (the hatched lower segment in panel b), and the gap is made up by the disagreement subset.

### 3.2 Results

Results can be found in Figure 2, where we compare regularized training ( _λ_ = 1 in Equation (2), left side) against unregularized training ( _λ_ = 0 in Equation (2), right side), plotting both behavior drift (a,c) and explanation EM for both Self and Orig (b,d).

From Figure 2(b), we observe that across held-out data on all three tasks—Hint-MMLU, AITA, and Refusal—the regularized model _M_ reg’s predicted explanations match its own labels _E_ ( _M_ reg) better than the original targets _E_ ( _M_ 0) it was trained on. Thus, we observe the **Self** _>_ **Orig** signature of introspective coupling across all three tasks. By construction, this gap arises entirely from examples where self and original labels disagree. This subset is plotted in Figure 2(b) as a darker shade, above the white line. On this subset, _M_ reg resolves the discrepancy in favor of its own behavior 79%, 82%, and 63% of the time, respectively. In Section B.2, we rule out trivial accounts of the **Self** _>_ **Orig** gap: that _E_ ( _M_ reg) has drifted to a degenerate distribution that is easier for _any_ external explainer to learn. We further validate that regularized training produces this signature across other model sizes and families (Figure 14).

From Figure 2(d), we observe that **Self** _>_ **Orig** disappears without regularization: a model trained on explanations alone, _M_ unreg = _M_ [ _E_ ( _M_ 0)] actually demonstrates Self < Orig. When comparing Figure 2(a) against Figure 2(c), we notice that _M_ unreg’s behavior drifts further away from _M_ 0 than _M_ reg’s behavior drifts. Following this, we hypothesize that behavior similarity during training governs coupling, which we test in depth in Section 4.

> 3See example in Section B.5.

### 3.3 Introspective coupling at the mechanistic level

We perform mechanistic analysis of models that display the **Self > Orig** signature. We investigate whether explanation and behavior are causally linked [Merullo et al., 2024], by testing if intervening on representations to shift behavior _B_ ( _M_ ) also shifts the explanation _E_ ( _M_ ) in the corresponding direction. We conduct our interpretability experiments on Hint-MMLU models.

We intervene on representations via activation patching (Figure 3 left). Recall that each explanation query _q_ implicitly contains the underlying object-level input _x_ (§2.1). We patch activations into the shared prefix _x_ and measure downstream effect on both behaviors _M_ ( _x_ ) and explanations _M_ ( _q_ ). We create two counterfactual inputs, a source _x_ and a target _x_^(_′_) (with corresponding source and target meta-questions _q_ , _q_^(_′_) ) that share the same hint cue _C_ = _C_^(_′_) , but have different cue-included behavioral responses _M_ ( _x_ ) _̸_ = _M_ ( _x_^(_′_) ) and as a result different ground-truth explanations _E_ ( _M_ )( _q_ ) _̸_ = _E_ ( _M_ )( _q_^(_′_) ). For each pair, we patch activations from the source forward pass (i.e. _x_ and _q_ ) to the target forward pass (i.e. _x_^(_′_) and _q_^(_′_) ) at corresponding (layer, token) positions, and measure the normalized logit difference (NLD)

target _−_ source^, where source is the logit elicited from a _M_ ( _x_ ) or _M_ ( _q_ ), target is the logit elicited from a _M_ ( _x_^(_′_) ) or _M_ ( _q_^(_′_) ), and patched is the logit elicited from a forward pass of _M_ ( _x_ ) with only one activation from _M_ ( _x_^(_′_) ) patched, or from a forward pass of _M_ ( _q_ ) with only one activation from _M_ ( _q_^(_′_) ) patched. Thus, an NLD of 0 indicates that the intervention has no effect, leaving the output at its source value, while an NLD of 1 indicates that it fully recovers the target output [Wang et al., 2023].

We validate the following hypothesis: if the model shares circuits between behavior and explanation, then patching interventions that affect behavioral logits NLD( _M_ ( _x_ )) are correlated with ones that affect explanation logits NLD( _M_ ( _q_ )). The result is shown in Figure 3. We find that interventions that move the behavior logits also move the explanation logits in the same direction, with Pearson correlation _r_ = 0.89. This means that the same layer/token activations are used for behavioral responses as for the explanations. Critically, the _M_ unreg baseline, which has non-trivial Orig explanation EM (Figure 2(d)) but does not exhibit **Self > Orig** , only exhibits a correlation of _r_ = 0.527 (Section C).

## 4 When does introspective coupling happen?

Introspective coupling is desirable for faithful explanation because it means that the model’s explanations track its current behavior. We next ask **when this coupling emerges.** Our findings in Section 3.2 suggested that models only demonstrated Self > Orig under regularization, which reduced behavioral drift compared to no regularization. Thus, we hypothesize that coupling depends on whether the explanation supervision remains **behaviorally compatible** with the model throughout training.

Formally, let _E_ sup^(()^(_t_)denote the explanation supervision labels used for SFT at training step)^(_t_.In the default) fixed-label setting, _E_ sup^(()^(_t_)=)^(_E_()^_M_ 0^()for all)^(_t_;in this section, we vary)^_E_ sup^(()^(_t_)directly.Let)^(_E_()^_M_ _t_^()denote the) ground-truth explanations constructed from the current model’s behavior at step _t_ . We refer to the similarity between _E_^(()^(_t_)) sup^and^(_E_()^_M_ _t_^() as)^(**online label–self compatibility**:the degree to which the explanation training) targets at step _t_ agree with the explanations of the current model’s behavior. Thus, our central hypothesis is

**Hypothesis:** High online label-self agreement (i.e. _E_ sup^(()^(_t_)and)^(_E_()^_M_ _t_^()agreement)over the course of) training (i.e. _∀t_ ) is an important factor for the emergence of Self _>_ Orig.

The three subsections below probe this hypothesis by varying online compatibility through different mechanisms: changing the behavioral regularization weight _λ_ , which affects how close _Mt_ remains to the fixed explanation labels (Section 4.1); directly controlling the agreement between _E_ sup^(()^(_t_)and)^(_E_()^_M_ _t_^() via online) relabeling (Section 4.2); and training on explanation targets from a different initial model (Section 4.3), which separates the role of _initial_ label–model similarity from _online_ compatibility during training.^4 Finally, we also perform a learning rate sweep that uncovers evidence _against_ this hypothesis, suggesting that multiple factors may be necessary to induce Self > Orig (Section D.2). We focus our analysis in this section on the Hint-MMLU setting.

Figure 3: Mechanistic signature of introspection: activation interventions that modify behavior are correlated with those that modify the explanation. **Left:** Schematic of the activation-patching intervention. We construct pairs of prompts with the same cue but different counterfactual explanation labels: for one prompt, removing the cue changes the model’s answer, while for the other it does not. We patch activations between the two prompts and measure whether interventions that shift behavior logits also shift explanation logits. **Right:** Each point represents a single (layer, token) patching position. The x-axis shows the mean normalized logit-difference for the behavior prediction (cue-included answer), while the y-axis shows the corresponding normalized logit-difference for the explanation’s change/no-change prediction. Patching effects on behavior and explanation are strongly correlated with Pearson _r_ = 0.89.

### 4.1 Behavioral Regularization Preserves Online Label-Self Compatibility

We begin by performing a finer-grained version of our regularization vs. no regularization experiment in Section 3.2, by varying the _weight λ_ on the behavioral regularization term in our objective (Equation (2)). We sweep _λ_ across five orders of magnitude, with results shown in Figure 4. We find that coupling begins to emerge at remarkably small values: a Self _>_ Orig gap is visible for any _λ_ ≳ 5 _×_ 10^(_−_3) , and the gap stays consistent for bigger _λ_ . Notably, this inflection coincides with a large increase in Behavior EM (Figure 4a) between _Mt_ and _M_ 0. This is consistent with our hypothesis that high online label-self agreement is important for Self > Orig. In this setup, the original model _M_ 0 is the supervision source throughout training so _E_^(()^(_t_)) sup^(=)^(_E_()^_M_ 0^() for all)^(_t_.Thus, when)^_M_ _t_^and^_M_ 0^(remain close (high behavioral EM) throughout training,) _E_ ( _Mt_ ) remains similar to _E_ sup^(()^(_t_), and Self > Orig emerges.)

However, this experiment does not distinguish whether coupling requires the current model being trained to remain close to the original model _M_ 0 or to the source of the explanation labels _E_ sup^(()^(_t_).We isolate this) distinction directly in Section 4.2, where we vary the current model’s proximity to the labels, independent of the original model.

### 4.2 Introspective Coupling Depends on Online Label-Self Agreement

We conduct a controlled test of our hypothesis that high online label-self agreement leads to introspective coupling. Because _M_ ’s behavior changes dramatically over the course of training, at every gradient step, we regenerate the explanation training targets _E_ sup^(()^(_t_)so that they agree with the)^(_current_model behavior)^(_B_()^(_M_) on) a fraction _ν_ of labels and disagree—via a randomly sampled alternative answer—on the remaining fraction 1 _− ν_ :

> 4We also plot the emergence of coupling over the course of training in Section A.

Figure 4: Behavioral regularization weight _λ_ sweep on Hint-MMLU (§4.1). **(a) Behavior EM** between _M_ reg and _M_ 0 rises at the same _λ_ -values as Self Explanation EM, supporting that label-self similarity correlates with coupling. **(b) Explanation EM** scored against labels of _M_ reg (blue) and _M_ 0 (orange). The Self > Orig signature occurs quickly once _λ ≥_ 5e _−_ 3, and the gap stays consistent for bigger _λ_ .

Figure 5: Continuous relabeling with fixed online label–self agreement (§4.2). We control the per-step agreement _ν_ between the explanation supervision _E_ sup^(()^(_t_)and the model’s self-labels)^(_E_()^_M_ _t_^().)^(**(a)**As)^(_ν_increases, Behavior EM (orange)) stays roughly constant at _≈_ 0.5, indicating that Self _>_ Orig emerges from higher online label–self similarity, not reduced drift from _M_ 0. **(b)** Meanwhile, Self Explanation EM (blue) is initially equal to Orig Explanation EM (orange), then rises sharply above it once _ν ≈_ 0.7.

On the behavioral side, we still regularize _M_ toward the base model _M_ 0. Thus, we have a model:

We sweep _ν_ in Figure 5. In (b), we find a sharp transition in explanation behavior as the online label–self agreement _ν_ varies. We observe introspective coupling (Self > Orig) for _ν ≥_ 0.7 but once _ν ≤_ 0.6, both Self and Orig Explanation EM collapse. This supports our hypothesis that coupling requires a minimum level of agreement between the explanation supervision and the current model’s behavior; for Qwen3-8B on Hint-MMLU, this threshold also appears to be _≈_ 0.7.

Figure 6: Training on explanation labels from another model (§4.3). We construct explanation labels _Eα_ by mixing in explanation labels from Llama-3.1-8B-Instruct into the supervision for Qwen3-8B: _α_ denotes the fraction of explanation labels drawn from Llama, with _α_ = 0 corresponding to pure Qwen labels and _α_ = 1 to pure Llama labels. Behavior is still regularized toward Qwen throughout. We score the resulting model against four explanation-label sources: Self, _Eα_ (mixed), Qwen-base, and Llama-base. **(a)** Across the sweep, the model’s behavior remains closest to its initial self (steady 80% behavioral EM against Qwen), but **(b)** models always demonstrate introspective coupling with Self > Orig. Thus, though explanation supervision comes from another model, the trained model’s explanations continue to track its own current behavior better than either base-model label source.

We measure behavioral agreement between the trained model and the original model _M_ 0 in Figure 5(a). Interestingly, behavioral agreement remains roughly constant across _ν_ . This disentangles the two factors we suggested at the end of the previous section: having the current model remain close to the _explanation labels_ over the course of training is more important than having it close to the _original model_ .

### 4.3 Introspective Coupling Persists with Cross-Model Explanation Labels

Previously, we found that it is more important for the current model to remain close to the explanation labels than to the initial model. We next ask whether those labels must be generated by the model’s own initial checkpoint. **Can a model learn to self-explain from labels generated by a different model, without substantial behavioral drift?** If so, explanation labels need not be collected separately for every model: labels from one model could induce self-explanations in another.

To test this, we start with a Qwen3-8B base model _M_ 0, from which we derive behaviors _B_ ( _M_ 0) and ground-truth explanations _E_ ( _M_ 0). We then replace some of Qwen’s explanation labels with labels from Llama-3.1-8B-Instruct, denoted ( _M_ Llama). The two models’ ground-truth explanations agree on only 53% of examples.

For each mixture ratio _α ∈_ [0, 1], we construct an explanation supervision dataset where _α_ is the fraction of explanation labels sourced from Llama. We train

Crucially, _we regularize behavior toward the original Qwen throughout_ , using _B_ ( _M_ 0); only the explanation supervision is mixed. Thus, _α_ = 0 corresponds to pure Qwen explanation supervision, while _α_ = 1 corresponds to pure Llama explanation supervision with Qwen behavioral regularization.

Results are shown in Figure 6. Plot (a) confirms that this intervention does not significantly affect the trained model’s behavior. Across _α_ , the trained model _M_ reg^(()^(_α_)remains a steady 80% from)^_M_ 0^((Orig Qwen) line) and a steady 55% from _M_ Llama (Orig Llama line). Thus, replacing Qwen explanation labels with Llama labels does not cause the trained model’s behavior to move toward Llama. Nevertheless, on the explanation side (b), Self _>_ Orig holds for every value of _α_ . Thus, **introspective coupling does not require the explanation labels to be generated by the same model being trained** .^5

These results suggest that explanation datasets may be reusable across behaviorally similar models. Rather than collecting labels for new models every time, it may be possible to train a family of models to self-explain from shared explanation supervision.

These results add to a growing body of work arguing that self-explanation in LMs leverages _privileged access_ [Binder et al., 2025, Li et al., 2025a]. Even when the model is explicitly trained on another model’s explanations, including examples where the two models disagree, its predictions track its own behavior more closely than the foreign explanation labels. We interpret this as further evidence that introspective coupling depends more on _online label–self agreement_ than on the provenance of the original explanation labels.

## 5 Generalization

In Sections 3 and 4, we studied behavioral drift that occurs as an emergent consequence of explanation training. However, in realistic post-training scenarios (or in scenarios where explanations would be useful), models are trained concurrently with other post-training data that may induce different behavioral shifts. **We study how well introspective coupling generalizes to explaining new or shifted behaviors that the model acquired through concurrent post-training.**^6 To formalize this setup, we extend our notation to include a dataset argument: let _B_ ( _M_ , _D_ ) denote behaviors produced by model _M_ on dataset _D_ (e.g. Hint-MMLU), and let _E_ ( _M_ , _D_ ) denote explanations corresponding to those behaviors. Starting from our default training pipeline, we concurrently train _M_ on an auxiliary post-training dataset _A_ :

where explanation training and behavioral regularization are applied only on the original dataset _D_ and the auxiliary dataset _A_ provides _behavior-only_ supervision: the model receives no explanation labels for _A_ .

We then evaluate whether _M_ aux’s explanations track its current behavior on _D_ and on _A_ . Ground-truth explanations are constructed post hoc from _M_ aux’s own behavior. Thus, faithful explanations on _A_ cannot result from direct explanation supervision and must reflect generalization of the explanation function.

We study two kinds of auxiliary shift: in §5.1 we inject new, synthetic behavior, and in §5.2 we design a realistic post-training corpus that shifts existing behaviors.

### 5.1 Generalization to Newly Acquired Behaviors

We first test whether coupling extends to behaviors the model could only have acquired through auxiliary training. We generate synthetic nonsense “Jabberwocky” data _J_ = _{_ ( _xJ_ , _yJ_ ) _}_ that follows the Hint-MMLU format but uses fictitious, alien-sounding questions that LMs do not have priors over, guaranteeing that the model’s behavior on _J_ cannot be inherited from _M_ 0.^7 We use the objective in Equation (5), where we perform explanation training on _D_ = Hint-MMLU and train on a subset of Jabberwocky _A_ = _J_ train as our auxiliary dataset, holding out another subset _J_ test to test generalization.

**Results.** Figure 7 reports Explanation EM on the three evaluation sets ( _D_ , _A_ , _J_ test). First we observe Self (blue) > Orig (orange) on all three datasets, **indicating that introspective coupling generalizes to auxiliary behavioral signal** . In particular, (a) shows that the coupling on the original dataset _D_ is preserved under auxiliary signal, (b) shows that _M_ aux explains _J_ train nearly perfectly (90.5%) despite never receiving explanation supervision, and (c) shows that explanation accuracy is 79.8% on _J_ test, which received neither explanation nor behavior signal.^8

> 5That said, explanation quality does degrade overall as the supervision diverges from original Qwen _M_ 0’s behavior: Self Explanation EM falls from 88% at _α_ = 0.0 (all Qwen labels) to 69% at _α_ = 1.0 (all Llama labels), which is expected if one considers the Llama labels as noise to the function of modeling self (Qwen) behaviors.

> 6A version of this question was explored by Binder et al. [2025] using synthetic tasks; we focus on more complex explanations with direct safety relevance, such as sycophancy or refusal behavior, under realistic post-training mixtures.

> 7Samples can be found in Section E.1

Figure 7: Explanations generalize to newly acquired behaviors (§5.1). We train _M_ aux with explanation supervision on Hint-MMLU ( _D_ ) plus behavior-only training on the Jabberwocky training set _J_ train ( _A_ ), and evaluate on three held-out sets: Hint-MMLU, _J_ train, and Jabberwocky test set _J_ test. Self (blue) > Orig (orange) signature persists on all three sets. Despite never seeing _J_ train _explanations_ , nor _J_ test explanations or behavior, _M_ aux explains both at high accuracy. The explanation-trained model _M_ reg, which never saw Jabberwocky, can only explain _M_ aux no better than chance.

Figure 8: Explanations track shifts to existing behaviors (§5.2). We mix realistic auxiliary post-training corpora into explanation training (Equation (5)): WildChat and warm-assistant dialogue alongside Hint-MMLU explanation training, and FineWeb and direct refusal (LLM-LAT) alongside Refusal explanation training. Each panel **(a)–(d)** is one ( _D_ , _A_ ) setup, with _Behavior EM_ (left) and _Explanation EM_ (right). For **Behavior EM** : all setups induce drift from both _M_ 0 and _M_ reg (only explanation training but no auxiliary training). For **Explanation EM** : We find that (1) Self > Orig coupling persists under auxiliary training, and that (2) Self > _M_ reg _→M_ aux, indicating that the model’s explanations have shifted alongside auxiliary training to reflect its shifted behavior.

These results are not explained by the auxiliary data failing to change the model: _M_ reg, which is trained only with Hint-MMLU explanation supervision and never sees Jabberwocky, explains _M_ aux’s Jabberwocky behavior substantially worse than _M_ aux explains itself, and only at chance. Thus, _M_ aux’s explanations track behaviors acquired through auxiliary training rather than merely preserving its original explanation policy.

### 5.2 Explanations Track Shifts to Existing Behaviors

While Section 5.1 tested whether explanations extend to a _new_ behavioral domain, we now test whether they track changes that auxiliary training induces _within_ the original explanation domain _D_ . Post-training routinely shifts behavior in unintentional ways: instruction-tuning increases sycophancy [Wei et al., 2024], and narrow fine-tuning can produce broadly misaligned models that stop refusing [Betley et al., 2025b]. A model whose explanations track such shifts could report these unintended consequences. We reuse objective Equation (5), now instantiating _A_ with various types of real post-training corpora:

> 8 _J_ test responses are near random for the nonsense questions with no ground-truth, and the model defers to changing to the hint. However, knowing to output the _cue-ablated_ random response on _J_ test is an ability that _M_ 0 or _M_ reg don’t have without training explicitly on Jabberwocky behavior. More details in Section E.1.

- _D_ = **Hint-MMLU,** _A_ = **WildChat:** WildChat [Zhao et al., 2024] is a common post-training dataset of real-world user-chatbot interactions.

- _D_ = **Hint-MMLU,** _A_ = **Warm Assistant Responses:** Following [Ibrahim et al., 2026], we generate a set of “warm and empathetic” assistant dialogues, which have been shown to make models more sycophantic.^9

- _D_ = **Refusal,** _A_ = **FineWeb:** FineWeb [Penedo et al., 2024] is a generic web corpus used for pretraining and induces a less directed shift in behavior.

- _D_ = **Refusal,** _A_ = **LLM-LAT Direct Refusal:** We mix in a corpus of entirely harmful requests from LLM-LAT and train the model to directly refuse them, shifting the model’s standard refusal behavior.^10

Figure 8 confirms that all four setups produce meaningful drift despite behavioral regularization, with respect to both the base model _M_ 0, and a model _M_ reg without training on auxiliary data _A_ . Furthermore, we find that: (1) **Self** _>_ **Orig persists under auxiliary training** : _M_ aux _→M_ aux (blue) beats _M_ aux _→M_ 0 (orange) across all settings, and (2) **Explanations track with auxiliary-training-induced shift** : _M_ aux explains its own behavior better than the no-drift explainer _M_ reg does, indicating that _M_ aux is not simply learning to explain the non-drifted part of its behavior. The gap between _M_ aux and _M_ reg’s explanation of _M_ aux is precisely where _M_ aux’s explanation shifted to reflect its shifted behavior.

These results have a practical implication for monitoring unintended consequences of post-training. Our findings suggest that, when explanation training is conducted alongside other post-training objectives, a model can report on behavioral shifts it acquires during training. Self-explanation could thus serve as a cheap, always-on probe for emergent behavioral change.

## 6 Related Work

LMs can produce natural language explanations of their outputs, either in chain-of-thought [Wei et al., 2022] or post-hoc, but their explanations can be unfaithful to their true decision-making processes [Turpin et al., 2023, Lanham et al., 2023, Barez et al., 2025]. Being able to elicit faithful chain-of-thoughts can be useful for external monitoring in safety-critical scenarios [Korbak et al., 2025, Guan et al., 2025]. This objective can be thought of as a self-consistency objective between model verbalizations and behaviors [Pres et al., 2026]. Prior work has studied explanations of model behaviors [Joglekar et al., 2025, Li et al., 2026, Hase and Potts, 2026, Mayne et al., 2026], internal activations [Pan et al., 2026, Karvonen et al., 2026, Fraser-Taliente et al., 2026, Huang et al., 2025, Choi et al., 2025], and training data [Goel et al., 2025, Shenoy et al., 2026].

Recent work also investigates whether models possess _metacognition_ or _introspective abilities_ , either zero-shot [Comsa and Shanahan, 2025, Laine et al., 2024, Lindsey, 2025, Zhong, 2026] or through finetuning [Binder et al., 2025, Plunkett et al., 2025, Betley et al., 2025a]. One set of work focuses on whether the model has self-knowledge of external tampering [Lindsey, 2025, Macar et al., 2026, Pearson-Vogel et al., 2026, Lederman and Mahowald, 2026], while another set investigates whether models can model their own output distribution [Binder et al., 2025, Li et al., 2025a, Song et al., 2025]. Our work falls into the second camp. A central question in the literature is whether models have privileged access when introspecting, or such introspective behaviors can be simulated by external models [Song et al., 2026, Li et al., 2025b, Singh et al., 2026]. In our work, we provide evidence for a different but potentially stronger version of privileged access: a model can model its own behavior better than its training target, despite not being supervised on the drift.

> 9Details in Section E.2.

> 10Details in Section E.2.

## 7 Conclusion

We have shown the surprising phenomenon of _introspective coupling_ : LMs trained on a fixed set of explanations derived from their base model learn to explain their _own_ current behaviors more faithfully than the training targets they were supervised on. We analyzed when and how this phenomenon happens, and also found that this coupling generalizes to behavioral shifts induced by complementary behavioral training. These results indicate promising potential for scalably integrating explanation training into future post-training pipelines: explanation labels do not need to be constantly refreshed over the course of training, and may even be shared across models.

### Limitations & Future Works

**Training requires sufficient behavioral variance.** The introspection training signal vanishes if _M_ 0’s behavior on _x_ vs. _x\C_ is nearly always the same (or nearly always different): _E_ ( _M_ 0) collapses to the same label (always “would change” or “would not change”), so explanation training degenerates into majority-class prediction. Collecting diverse supervision can be difficult. For example, for refusal, alignment training tends to drive _M_ 0 to refuse almost every adversarial prompt; thus, the explanation _E_ ( _M_ 0) is almost always “would not change from refusal”, which makes the training signal useless: _M_ reg simply learns to output the same explanation, rather than learning to introspect.

**Generalization.** Future work should look at **OOD generalization** where models can generalize from counterfactual training of one domain to another, or generalization with a **more diverse, non-templated** set of meta-level input or output evaluations beyond just counterfactuals.

**Our story for when introspective coupling emerges is incomplete.** In Section 4, we hypothesized that online label self-similarity predicts the emergence of introspective coupling. While we found significant evidence supporting this hypothesis, we also found counter-evidence in Section D.2, and evidence that there may be additional factors in Section D.1. Future work should study the interaction between all factors and map out a fuller story of when coupling emerges.

**Our mechanistic story is incomplete.** We began to form a mechanistic story of how the circuits used for object-level outputs and meta-level outputs overlap, and found supporting evidence for it. However, the story is far from complete. The asymmetry we find between the two patching signatures in Section C.2 is not explained. Future work could provide a more precise characterization of the “introspection circuit” and how the overlapping circuits function.

## Broader Impact

Faithful introspection is a prerequisite for using model explanations as a tool for oversight of language models: if a model’s verbalized rationales causally track its behavior, downstream users and auditors can use those rationales to anticipate, debug, and contest model decisions, including in safety-critical domains such as misalignment detection. While we hope introspection training can unlock self-explanations as an interface into model behavior, we believe it will remain complementary to _extrospective_ interpretability tools such as probes, sparse autoencoders, and circuit discovery techniques, that perform a more rigorous, internally-grounded analysis into model behavior at a higher cost. Furthermore, it remains to be seen how self-explanations might interact with overall model alignment, and whether faithfulness survives if models become unaligned and learn to deceive or be dishonest. Overall, we believe that verifying causal coupling could be beneficial before treating model-generated explanations as evidence about the model’s true decision-making process.

## Acknowledgments

This work was supported by the National Science Foundation through grant IIS-2238240, the IARPA BENGAL program, the DARPA AIQ program through CMO contract HR00112520025, and the MIT Generative AI Consortium. JA is supported by a Sloan Fellowship, and BZL is supported by a Clare Boothe Luce Fellowship. We thank Coefficient Giving (prev. Open Philanthropy) for partly providing compute funding through a grant issued in the Technical AI Safety RFP. We would like to thank Leshem Choshen, Riddhi Bhagwat, and Chris Ge for helpful feedback on drafts of this paper, and Itamar Pres for valuable discussions about this project.

## References

- Fazl Barez, Tung-Yu Wu, Iván Arcuschin, Michael Lan, Vincent Wang, Noah Siegel, Nicolas Collignon, Clement Neo, Isabelle Lee, Alasdair Paren, Adel Bibi, Robert Trager, Damiano Fornasiere, John Yan, Yanai Elazar, and Yoshua Bengio. Chain-of-thought is not explainability, 2025. URL https://aigi.ox.ac.uk/ wp-content/uploads/2025/07/Cot_Is_Not_Explainability.pdf. Preprint. Under review.

- Jan Betley, Xuchan Bao, Martín Soto, Anna Sztyber-Betley, James Chua, and Owain Evans. Tell me about yourself: LLMs are aware of their learned behaviors. In _International Conference on Learning Representations_ , 2025a. URL https://openreview.net/forum?id=IjQ2Jtemzy.

- Jan Betley, Daniel Chee Hian Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, and Owain Evans. Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs. In _International Conference on Machine Learning_ , 2025b. URL https://openreview.net/forum?id= aOIJ2gVRWW.

- Felix Jedidja Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, and Owain Evans. Looking inward: Language models can learn about themselves by introspection. In _International Conference on Learning Representations_ , 2025. URL https://openreview.net/ forum?id=eb5pkwIB5i.

- Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vlad Mikulik, Samuel R. Bowman, Jan Leike, Jared Kaplan, and Ethan Perez. Reasoning models don’t always say what they think, 2025. URL https: //arxiv.org/abs/2505.05410.

- Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, and Dan Jurafsky. ELEPHANT: Measuring and understanding social sycophancy in LLMs. In _International Conference on Learning Representations_ , 2026. URL https://openreview.net/forum?id=igbRHKEiAs.

- Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing GPT-4 with 90%* ChatGPT quality, March 2023. URL https://lmsys.org/blog/ 2023-03-30-vicuna/.

- Dami Choi, Vincent Huang, Sarah Schwettmann, and Jacob Steinhardt. Scalably extracting latent representations of users. https://transluce.org/user-modeling, November 2025.

- Iulia M. Comsa and Murray Shanahan. Does it make sense to speak of introspection in large language models?, 2025. URL https://arxiv.org/abs/2506.05068.

- Kit Fraser-Taliente, Subhash Kantamneni, Euan Ong, Dan Mossing, Christina Lu, Paul C. Bogdan, Emmanuel Ameisen, James Chen, Dzmitry Kishylau, Adam Pearce, Julius Tarng, Alex Wu, Jeff Wu, Yang Zhang, Daniel M. Ziegler, Evan Hubinger, Joshua Batson, Jack Lindsey, Samuel Zimmerman, and Samuel Marks.

Natural language autoencoders produce unsupervised explanations of llm activations. _Transformer Circuits Thread_ , 2026. URL https://transformer-circuits.pub/2026/nla/index.html.

- Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

- Avichal Goel, Yoon Kim, Nir Shavit, and Tony T. Wang. Learning to interpret weight differences in language models, 2025. URL https://arxiv.org/abs/2510.05092.

- Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, et al. The Llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

- Melody Y. Guan, Miles Wang, Micah Carroll, Zehao Dou, Annie Y. Wei, Marcus Williams, Benjamin Arnav, Joost Huizinga, Ian Kivlichan, Mia Glaese, Jakub Pachocki, and Bowen Baker. Monitoring monitorability, 2025. URL https://arxiv.org/abs/2512.18311.

- Peter Hase and Christopher Potts. Counterfactual simulation training for chain-of-thought faithfulness, 2026. URL https://arxiv.org/abs/2602.20710.

- Vincent Huang, Dami Choi, Daniel D Johnson, Sarah Schwettmann, and Jacob Steinhardt. Predictive concept decoders: Training scalable end-to-end interpretability assistants, 2025. URL https://arxiv.org/abs/ 2512.15712.

- Lujain Ibrahim, Franziska Sofia Hafner, and Luc Rocher. Training language models to be warm can reduce accuracy and increase sycophancy. _Nature_ , 652(8112):1159–1165, Apr 2026. ISSN 1476-4687. doi: 10.1038/s41586-026-10410-0. URL https://doi.org/10.1038/s41586-026-10410-0.

- Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, and Nouha Dziri. Wildteaming at scale: From in-the-wild jailbreaks to (adversarially) safer language models. In _Advances in Neural Information Processing Systems_ , 2024. URL https://openreview.net/forum?id=n5R6TvBVcX.

- Manas Joglekar, Jeremy Chen, Gabriel Wu, Jason Yosinski, Jasmine Wang, Boaz Barak, and Amelia Glaese. Training LLMs for honesty via confessions, 2025. URL https://arxiv.org/abs/2512.08093.

- Adam Karvonen, James Chua, Clément Dumas, Kit Fraser-Taliente, Subhash Kantamneni, Julian Minder, Euan Ong, Arnab Sen Sharma, Daniel Wen, Owain Evans, and Samuel Marks. Activation oracles: Training and evaluating llms as general-purpose activation explainers, 2026. URL https://arxiv.org/abs/2512. 15674.

- Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, Scott Emmons, Owain Evans, David Farhi, Ryan Greenblatt, Dan Hendrycks, Marius Hobbhahn, Evan Hubinger, Geoffrey Irving, Erik Jenner, Daniel Kokotajlo, Victoria Krakovna, Shane Legg, David Lindner, David Luan, Aleksander M ˛adry, Julian Michael, Neel Nanda, Dave Orr, Jakub Pachocki, Ethan Perez, Mary Phuong, Fabien Roger, Joshua Saxe, Buck Shlegeris, Martín Soto, Eric Steinberger, Jasmine Wang, Wojciech Zaremba, Bowen Baker, Rohin Shah, and Vlad Mikulik. Chain of thought monitorability: A new and fragile opportunity for AI safety, 2025. URL https://arxiv.org/abs/2507.11473.

- Rudolf Laine, Bilal Chughtai, Jan Betley, Kaivalya Hariharan, Mikita Balesni, Jérémy Scheurer, Marius Hobbhahn, Alexander Meinke, and Owain Evans. Me, myself, and AI: The situational awareness dataset (SAD) for LLMs. In _Advances in Neural Information Processing Systems Datasets and Benchmarks Track_ , 2024. URL https://openreview.net/forum?id=UnWhcpIyUC.

- Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukoši˙ ut¯ e, Karina Nguyen, Newton˙ Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Measuring faithfulness in chain-of-thought reasoning, 2023. URL https://arxiv.org/abs/2307.13702.

- Harvey Lederman and Kyle Mahowald. Emergent introspection in AI is content-agnostic, 2026. URL https://arxiv.org/abs/2603.05414.

- Belinda Z. Li, Zifan Carl Guo, Vincent Huang, Jacob Steinhardt, and Jacob Andreas. Training language models to explain their own computations, 2025a. URL https://arxiv.org/abs/2511.08579.

- Chloe Li, Mary Phuong, and Daniel Tan. Spilling the beans: Teaching LLMs to self-report their hidden objectives. In _International Conference on Learning Representations_ , 2026. URL https://openreview.net/ forum?id=sWs0cCuM8I.

- Millicent Li, Alberto Mario Ceballos Arroyo, Giordano Rogers, Naomi Saphra, and Byron C Wallace. Do natural language descriptions of model activations convey privileged information? In _Mechanistic Interpretability Workshop at NeurIPS 2025_ , 2025b. URL https://openreview.net/forum?id=zyhibAkzSA.

- Jack Lindsey. Emergent introspective awareness in large language models. _Transformer Circuits Thread_ , 2025. URL https://transformer-circuits.pub/2025/introspection/index.html.

- Uzay Macar, Li Yang, Atticus Wang, Peter Wallich, Emmanuel Ameisen, and Jack Lindsey. Mechanisms of introspective awareness, 2026. URL https://arxiv.org/abs/2603.21396.

- Andreas Madsen, Sarath Chandar, and Siva Reddy. Are self-explanations from large language models faithful? In _Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024_ , volume ACL 2024 of _Findings of ACL_ , pages 295–337. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-ACL.19. URL https://doi.org/10. 18653/v1/2024.findings-acl.19.

- Harry Mayne, Justin Singh Kang, Dewi Sid William Gould, Kannan Ramchandran, Adam Mahdi, and Noah Y. Siegel. A positive case for faithfulness: LLM self-explanations help predict model behavior. In _ICLR 2026 Workshop on Principled Design for Trustworthy AI - Interpretability, Robustness, and Safety across Modalities_ , 2026. URL https://openreview.net/forum?id=xBmdoFmmFS.

- Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. Circuit component reuse across tasks in transformer language models. In _International Conference on Learning Representations_ , 2024. URL https://openreview. net/forum?id=fpoAYV6Wsk.

- Alexander Pan, Lijie Chen, and Jacob Steinhardt. LatentQA: Teaching LLMs to decode activations into natural language. In _International Conference on Learning Representations_ , 2026. URL https://openreview. net/forum?id=niUroX9EOd.

- Theia Pearson-Vogel, Martin Vanek, Raymond Douglas, and Jan Kulveit. Latent introspection: Models can detect prior concept injections, 2026. URL https://arxiv.org/abs/2602.20031.

- Guilherme Penedo, Hynek Kydlíˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In _Advances in Neural Information Processing Systems Datasets and Benchmarks Track_ , 2024. URL https://openreview.net/forum?id=n6SCkn2QaG.

- Dillon Plunkett, Adam Morris, Keerthi Reddy, and Jorge Morales. Self-interpretability: Llms can describe complex internal processes that drive their decisions, 2025. URL https://arxiv.org/abs/2505.17120.

- Itamar Pres, Belinda Z. Li, Laura Ruis, Zifan Carl Guo, Keya Hu, Mehul Damani, Isha Puri, Ekdeep Singh Lubana, and Jacob Andreas. Position: It’s time to optimize for self-consistency. In _International Conference on Machine Learning Position Paper Track_ , 2026. URL https://time-for-consistency.github.io/.

- Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Esin DURMUS, Zac Hatfield-Dodds, Scott R Johnston, Shauna M Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, and Ethan Perez. Towards understanding sycophancy in language models. In _International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id=tvhaxkMKAn.

- Keshav Shenoy, Li Yang, Abhay Sheshadri, Sören Mindermann, Jack Lindsey, Sam Marks, and Rowan Wang. Introspection adapters: Training llms to report their learned behaviors, 2026. URL https://arxiv.org/ abs/2604.16812.

- Abhay Sheshadri, Aidan Ewart, Phillip Huang Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, and Stephen Casper. Latent adversarial training improves robustness to persistent harmful behaviors in LLMs. In _International Conference on Learning Representations_ , 2025. URL https://openreview.net/forum?id=wI5uHZLeCZ.

- Shashwat Singh, Tal Linzen, and Shauli Ravfogel. Can llms introspect? a reality check, 2026. URL https: //arxiv.org/abs/2605.26242.

- Siyuan Song, Jennifer Hu, and Kyle Mahowald. Language models fail to introspect about their knowledge of language. In _Conference on Language Modeling_ , 2025. URL https://openreview.net/forum?id=AivRDOFi5H.

- Siyuan Song, Harvey Lederman, Jennifer Hu, and Kyle Mahowald. Privileged self-access matters for introspection in AI. In _ICML 2026 Workshop: Philosophy Meets Machine Learning_ , 2026. URL https: //openreview.net/forum?id=ZcqCJHOWAA.

- Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In _Advances in Neural Information Processing Systems_ , 2023. URL https://openreview.net/forum?id=bzs4uPLXvi.

- Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In _International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=NpsVSN6o4ul.

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum?id=_VjQlMeSB_J.

- Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and Quoc V. Le. Simple synthetic data reduces sycophancy in large language models, 2024. URL https://arxiv.org/abs/2308.03958.

- An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

- Zhehao Zhang, Weijie Xu, Fanyou Wu, and Chandan K. Reddy. Falsereject: A resource for improving contextual safety and mitigating over-refusals in LLMs via structured reasoning. In _Conference on Language Modeling_ , 2025. URL https://openreview.net/forum?id=1w9Hay7tvm.

- Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In _International Conference on Learning Representations_ , 2024. URL https: //openreview.net/forum?id=Bl8u7ZRlbM.

- Ziqian Zhong. Spontaneous introspection in output tampering. LessWrong, April 2026. URL https:// www.lesswrong.com/posts/yAR6uMdSaBjkbJ4u9/spontaneous-introspection-in-output-tampering. Accessed: 2026-05-06.

## A Self > Orig Emergence over the Course of Training

We track when introspective coupling emerges over the course of training and conversely, when self- and original-explanation may decouple in models that don’t exhibit coupling. At every training batch, we record three quantities: (i) on behavior data, whether _B_ ( _M_ ) matches the training label _B_ ( _M_ 0) (blue lines); (ii) on the explanation task, whether _M_ ’s predicted explanation matches the training label _E_ ( _M_ 0) (pink line); and (iii) on the explanation task, whether _M_ ’s predicted explanation matches the explanation _E_ ( _M_ ) constructed from the model’s own current behavior (dark red line).

We smooth each per-batch series with a time-weighted exponential moving average^11 and report the resulting curves for _M_ unreg and _M_ reg in Figure 9(a) and (b), respectively, with panel (c) plotting the normalized self-minus-orig explanation delta for both models. (a) Without regularization, _B_ ( _M_ ) drifts sharply from _B_ ( _M_ 0) within the first 1000 steps—almost entirely on the cue-included answer. On the explanation side, the self-vs-orig gap widens following the behavioral drift. (b) With regularization, _M_ reg’s self- and orig-explanation curves remain tightly coupled throughout training, with self-explanation gradually pulling ahead.

Figure 9: Training dynamics of Qwen3-8B on HINT-MMLU. **(a)** No regularization on behavior vs. **(b)** with KL regularization. We plot time-weighted exponential moving average with _α_ = 0.99 of the corresponding per-batch metric. Blue lines plot the two behavioral drift metrics, and the two explanation lines match explanation against the model’s own behavior labels (purple) and against the original _M_ 0 labels (pink). **(c)** the per-step ratio of explanation (Self _−_ Orig) / behavior gap. Behavior gap measures the total per-step behavior disagreement against _M_ 0, and the whole metric measures the amount of disagreeing data that self-explanation uniquely recovers.

## B Additional Metrics and Evaluations for Introspective Coupling (§3)

### B.1 Fine-grained metrics

In the main paper, we evaluate the model’s explanation on one main metric, the Explanation Exact Match (EM). We provide more fine-grained metrics to supplement here. For each example ( _x_ , _C_ ), let _e_ ˆ _∼ M_ ( _· | q_ ) be the model’s predicted explanation. Recall that _e_ ˆ consists of two components: the is-changed status (“ _The response [would/would not] change..._ ”) and the content of the change (“ _...to <M_ ( _x\C_ ) _>_ ”). We check each component separately:

> • _is-changed match_ : We check whether _e_ ˆ’s is-changed prediction matches the model’s is-changed behavior under cue ablation, i.e. whether it matches **1** [ _B_ ( _M_ )( _x_ ) _̸_ = _B_ ( _M_ )( _x\C_ ) ]. Because the is-changed label count may be asymmetric, we evaluate _Changed F1_ and _Unchanged F1_ separately.

> 11Following WandB’s line-plot smoothing implementation; see https://docs.wandb.ai/guides/app/features/panels/line-plot/ smoothing/.

- _content match_ : We check whether the _e_ ˆ’s content prediction is correct, i.e. whether it matches _B_ ( _M_ )( _x\C_ ).

### B.2 Do model behaviors shift to become easier to explain?

**Ruling out trivial distribution collapse.** The explanation evaluation for all the metrics can be seen in Figure 10. Behavior panel (a; left) shows that self-behavior drifts away from the original distribution but does not collapse onto a single mode: each behavior category retains roughly the same share as under _M_ 0.^12 The Explanation panel (a; right) shows that the explanation quality is better for all four metrics on self (Exact Match, Content Match, Change F1, and Unchange F1), not just exact match. Crucially, Change F1 and Unchange F1 are both high ( _>_ 88%) against _E_ ( _M_ reg), so the explainer is making genuine bidirectional changed-vs-unchanged predictions. This rules out the possibility that behavior was aligned to a lopsided distribution and that _M_ reg verbalizes that one trivial label — a degenerate self-introspector that always says “unchanged” would have one F1 at zero.

Figure 10: **(a)** Regularized explainer _M_ reg trained on _M_ 0’s explanations. Behavior breakdown shows the distribution of 8-label breakdown for _B_ ( _M_ 0), and that it doesn’t drift into a degenerate case. Four explanation metrics (Exact Match, Content Match, Change F1, Unchange F1) all show the Self _>_ Orig gap. **(b)** External explainer baseline _M_ reg^((2):base) Qwen3-8B trained on _E_ ( _M_ reg). This graph validates that _E_ ( _M_ reg) is not necessarily an easier distribution to learn than _E_ ( _M_ 0) and that _M_ reg models itself better than an external explainer like _M_ reg^((2).)

**Is** _E_ ( _M_ reg) **easier for any explainer to learn?** Perhaps the model _M_ reg’s behavior did not drift to be degenerate, but it could drift into a distribution that is easier for any external explainer model to learn, rather than requiring introspection. We verify that this is not the case by training another model on _M_ reg’s explanations and behaviors, which we call _M_ reg^((2)=)^(_M_[)^(_B_()^_M_ reg^(),)^(_E_()^_M_ reg^()].In practice, this model) is initialized from _M_ 0. This model simulates an external model to learn the (potentially easier) new distribution.

We find that:

1. In Figure 10, _M_ reg^((2)explaining)^_M_ reg^((orange bar in (b)) is not better than)^_M_ reg^explaining^_M_ 0^((orange) bar in (a)) — both are 78.5% — meaning that _E_ ( _M_ reg) is not an easier distribution to learn.

2. _M_^((2)) reg^explaining^_M_ reg^((orangebarin(b))isworsethan)^_M_ reg^explaining^_M_ reg^((bluebarin(a)),)

meaning that the self-explainer is better than an external explainer.

> 12The one caveat is asymmetric coverage: categories absent from the _M_ 0 distribution remain absent under self as well, but no category present in the original distribution disappears under the self-distribution.

3. More surprisingly, for _M_ reg^((2),)^(**Self > Orig**persists, i.e.)^_M_ reg^((2)’s predicted explanations match)^(_E_()^_M_ reg^((2))) (blue bar in (b)) better than _E_ ( _M_ reg) for this new model (orange bar in (b)).

Thus, _M_ reg’s behavior has not shifted to become easier to learn; rather, _M_ reg indeed appears to have a specific advantage at explaining its own current behavior.

### B.3 Untrained baseline

Following [Li et al., 2025a], we evaluate an “untrained baseline” to see how well these small models can do explanation out-of-the-box without any explanation training, by directly prompting the base model _M_ 0 with few-shot learning. The result is shown in Figure 11. Across all three tasks the untrained few-shot baseline reaches only 14–18% Explanation EM, far below the trained explainer, establishing that self-explanation is not a native capability of _M_ 0.

### B.4 AITA Data and Results

We check whether the Self > Orig result generalizes to another sycophancy dataset, AITA [Cheng et al., 2026].

Here, training data is derived from https://huggingface. co/datasets/OsamaBsher/AITA-Reddit-Dataset. This is converted into the same hint-style _C_ -ablation format used elsewhere in the paper. The cue _C_ being ablated is a single system prompt prepended to each Reddit post:

Figure 11: Untrained few-shot baseline ( _M_ 0 _→M_ 0) Explanation EM on each of the three tasks. The few-shot prompted base model reaches only 14–18%, showing that self-explanation emerges only after explanation training.

**AITA system prompt (ablated as** _C_ **)**

Read the following situation in a post on Reddit and determine if the person is the asshole.

Figure 12 shows (a) the dataset statistics and (b) the results with detailed metrics of a model explanationtrained in this domain. Note that the test set is balanced over the uncued-condition NTA/YTA verdict, rather than the change/unchange verdict. From (b), we find Self > Orig in this domain on all four metrics. In addition, the fine-grained metrics show that on Orig, Change F1 is substantially lower than Unchange F1, while on Self, both F1s are similarly high.

### B.5 Refusal Data and Results

We also check whether the Self > Orig result generalizes to explaining when the model will refuse, and specifically under which roleplaying system prompts.

We use GPT-5 to generate roleplay system prompts in the following style, which push the assistant towards greater refusal or compliance:

**Refusal explanation system prompt to push towards refusal**

This user is an assistant for a children’s after-school homework program.

**Refusal explanation system prompt to push towards compliance**

This user is a certified sex-ed teacher developing materials for an adult health literacy program.

Figure 12: AITA _M_ reg detailed metrics. Behavior change rate (a) and four explanation-quality metrics (b) for the explainer trained on the original target’s labels (orange) vs. on self labels (blue). Each explanation bar is decomposed into the agreement-subset matched baseline (gray) and the disagreement-subset gain (solid).

Figure 13: Refusal Explanation with Behavioral Regularization. Behavior change rate (a) and four explanation-quality metrics (b) for the explainer trained on orig vs self labels. Self > Orig on all metrics.

Figure 13 shows (a) the dataset statistics and (b) the results of a model explanation-trained in this domain. Because alignment post-training makes refusal behavior highly stable across system prompts, examples in which the system-prompt intervention flips a response to refusal are extremely rare in our training data. We see from (a) that the change _→_ refusal category occupies a tiny sliver of the dataset, and thus the explainer never learns to output this label. Broadly, one limitation of our approach is that it requires counterfactual training data with sufficient behavioral variation.From (b), we see that Self > Orig generalizes to this domain on all four metrics.

Figure 14: **Top** : Llama-3.1-8B-Instruct trained and regularized on HINT-MMLU. Self > Orig on every explanation metric. **Bottom** : Qwen3-32B trained and regularized on HINT-MMLU (LoRA _r_ = _α_ = 128). Self > Orig on every explanation metric.

### B.6 HINT-MMLU with Other Models

To check that the Self _>_ Orig phenomenon is not Qwen3-8B-specific and that it has potential to scale, we replicate the main _M_ reg Self > Orig results on Llama-3.1-8B-Instruct [Grattafiori et al., 2024] and Qwen332B [Yang et al., 2025]. At 32B parameters, full fine-tuning is prohibitive, so we use **LoRA** ( _r_ = _α_ = 128)

Guo et al. Introspective Coupling training rather than full fine-tuning. The full metrics decomposition is shown in Figure 14; again we observe Self > Orig across all explanation metrics for both models.

## C Additional Interpretability Results (§3.3)

### C.1 M unreg and M 0 baseline

Figure 3 establishes that for regularized _M_ reg, interventions that cause changes in the cue-included answer are correlated with those causing changes in the explanation as well.

Figure 15a reports the same analysis on the model _M_ unreg trained without direct behavioral regularization. We show that the correlation is less salient with Pearson _r_ = 0.53. Figure 15b shows that the base model _M_ 0 produces no correlation between the behavior and explanation at all, with Pearson _r_ = 0.20.

(a) _M_ unreg baseline counterpart to Figure 3: same per-(layer, token) scatter of mean normalized logit-diffs, with both intervention directions overlaid (circles for Change _→_ Unchange, triangles for Unchange _→_ Change). Per-direction _r_ = 0.48/0.58; combined _r_ = 0.53. Removing regularization roughly halves the Pearson correlation between behavior and explanation logit-diffs.

(b) _M_ 0 baseline counterpart to Figure 3: same per-(layer, token) scatter of mean normalized logit-diffs. _M_ 0 without explanation training has no correlation at all. Per-direction _r_ = 0.195/0.237; combined _r_ = 0.196.

### C.2 Correlation between cue-ablated behavior and explanation

Section 3.3 reports the correlation between the explanation logit-diff and the _cue-included_ answer-letter logitdiff as the object-level behavior. Here we run the same patching analysis with the _cue-ablated_ answer-letter logit-diff as the object-level behavior. For every patched run, we ask whether the shift in the model’s answer to the cue-ablated input correlates with its explanation. The patching prompt pairs are identical to the main setup, but the intervention positions are no longer at the cue region, because the cue is no longer present in the cue-ablated inputs. Instead, we choose to patch in the last 10 tokens of the prompts. _Note that this setup is noisier because the tokens we patch are not shared between the two prompts._

Figure 16 shows that the correlation story is more complex. On the top, we notice that the patching results are asymmetric, so we report the two patching directions separately. (Top left) When we patch activations from a prompt whose explanation is “change” into a prompt whose explanation is “unchange,” the resulting shift in cue-ablated answer is positively correlated with shift in explanations. (Top right) In the reverse direction—patching from an “unchange” prompt into a “change” prompt—shifts in behaviors do not induce shifts in explanations. The reason behind this asymmetry remains to be explored.

Furthermore, the absolute normalized logit difference (NLD) for either direction is visibly smaller than that of the cue-included studies in Figure 3, with a maximum _NLD ≈_ 0.2 vs. _≈_ 0.4 for the cue-included behavior. This is consistent with the cue-ablated setup being noisier, since the patched token spans are not shared across the paired prompts.

By comparison, the _M_ unreg baseline (bottom) has no significant correlation in either direction.

Figure 16: Per-direction (subset) breakdown of the correlation study between cue-ablated answer and explanation. Panels (a, b): regularized _M_ reg. Panels (c, d): no-regularization baseline _M_ unreg. Only the Change _→_ Unchange subset is where _M_ reg shows a meaningful correlation.

## D Additional Details on When Introspective Coupling Emerges (§4)

### D.1 Only Higher-Rank LoRA Recovers Self > Orig

We defaulted to using full-finetuning (FFT) in the main paper as it induced a larger behavioral shift enabling us to investigate the Self > Orig gap. However, here, we investigate the effect of LoRA training, and specifically the effect of LoRA rank _r_ . We focus on the HINT-MMLU task.

We sweep along _r ∈{_ 32, 64, 80, 96, 128, 256 _}_ , holding _α_ = _r_ , and report the (a) Behavior EM and (b) Explanation EM in Figure 17. From (a), we see that regardless of LoRA rank, behavioral EM always drifts, although to a lesser extent than full fine-tuning (Figure 2). From (b), we find that only sufficiently high-rank LoRA adapters can recover the Self > Orig effect. Therefore, the low-rank LoRA adapters’ inability to elicit Self > Orig phenomenon cannot simply be attributed to the fact that they don’t drift as much, indicating that online label-self similarity (our core hypothesized factor in Section 4) is not the only factor contributing to the emergence of introspective coupling; future work can investigate the interplay between various factors.

Figure 17: KL-regularized LoRA rank sweep on HINT-MMLU, _α_ = _r_ . For each _r_ , we plot Behavior EM (blue) and Explanation EM scored against orig-labels (light pink) and self-labels (dark pink). The self > orig gap is absent or slightly negative for _r ≤_ 80, opens at _r_ = 96, and widens monotonically through _r_ = 256.

### D.2 Higher Learning Rate Can Widen The Self > Orig Gap

The main paper uses _lr_ = 1 _×_ 10^(_−_5) , which we find to give the best absolute Explanation EM in general. Here we measure how the learning rate affects the emergence of the Self > Orig gap. We sweep six different learning rates ranging from 1 _×_ 10^(_−_4) to 5 _×_ 10^(_−_6) and plot the Behavioral and Explanation EM in Figure 18. We find that as learning rate increases (right to left), the absolute explanation EM drops on both Orig and Self labels, but it drops more quickly for Orig than Self and widens the Self > Orig gap. We find that behavior EM also decreases with higher LR, and roughly matches Orig Explanation EM.

**Counter-evidence to online Label-Self similarity hypothesis.** The LR sweep is the one experiment in this paper that presents counter-evidence to the hypothesis explored in Section 4 — that online Label-Self similarity governs when introspective coupling emerges. At very large learning rates the trained model drifts off base significantly — meaning that the current model (Self) is extremely dissimilar from the initial model that generated the explanation labels (Label) — and yet Self _>_ Orig gap does not disappear, but in fact _grows_ . We flag this result as an open problem: we hypothesize that multiple factors, including LR and label-self similarity, interact in competing ways to influence the emergence of introspective coupling. Potentially a large LR encourages the explanation circuit to re-route through the model’s existing behavioral circuitry (independent of behavioral drift magnitude); future work can explore these factors empirically.

Figure 18: LR sweep on HINT-MMLU plotted with learning rate decreasing left-to-right. For each learning rate, we plot Behavior EM (blue) and Explanation EM scored against orig labels (light pink) and self labels (dark pink). Self _>_ Orig is absent at lr = 5e _−_ 6 and continues to widen as the learning rate rises and the model drifts further from base.

## E Additional Details for Auxiliary Training Generalization (§5)

### E.1 Jabberwocky Dataset Details and Additional Results (§5.1)

We show an example of Jabberwocky “nonsensical science” data below:

**Jabberwocky data example**

According to the Lomar schema for cataloging pseudoalloy microphases, which microphase label is assigned to the sorvian crystal specimen? Choices: A. Rulven-di, B. Shenqa-no, C. Tilmare-zi D. Zavrek-ka Correct Answer: C. Tilmare-zi

**Detailed Metrics.** We now show the detailed metrics of the Jabberwocky training on held-out sets of HINT-MMLU and Jabberwocky _J_ train where we train behaviors only and not explanations. In Figure 19, we show 6 metrics: cue-included and cue-ablated match for behavior EM, and Exact Match, Content Match, Change F1, and Unchange F1 for explanation. On every metric, for the HINT-MMLU evaluation, Self > Orig persists. On the Jabberwocky _J_ train set, the accuracy on every metric is high, indicating proper generalization.

#### Detailed Metrics for training in Jabberwocky behaviors

Figure 19: Full six-metric breakdown for Jabberwocky mixed training, a more detailed version of Figure 7. _M_ aux is trained with explanation supervision on Hint-MMLU plus behavior-only training on _J_ train, then evaluated on held-out HINT-MMLU and _J_ train. Self > Orig persists on every HINT-MMLU evaluation, and the _J_ train accuracy is also high.

**Generalization to model behaviors never seen in training.** We evaluate _M_ aux’s introspection ability after training on _J_ train on a held-out Jabberwocky set _J_ test that the model has never seen in training. Figure 20 shows the behavioral distribution of base model _M_ 0 and Jabberwocky-trained model _M_ aux in panel (a), where _M_ 0 has a bias to choose C and _M_ aux is approximately uniform. In panel (b), despite the models deferring to the hint for _J_ test, which is by construction nonsensical and has no ground truth, there is a 25% chance that the hint coincides with the original answer because the hint is randomly selected from the 4 options, and, thus, the “Flipped” rate is lower at around 75%. In panel (c), we show that the ability to distinguish between “Flipped” and “non-Flipped” is non-trivial, despite the seemingly simple behavioral distribution: _M_ 0 is not able to do so under few-shot prompting, as indicated by the poor Unchange F1 score, while _M_ aux can after training. Meanwhile, because the format of the Semantic Match is to output the original answer _without hint_ , _M_ 0 confuses the template and outputs the hint answer instead, causing it to have a 23.8% Semantic Match.

Overall, these results show that (1) the trained model’s distribution has not become degenerate (panel a), and (2) Change and Unchange F1 are both high, indicating that the trained model’s prediction also has not become degenerate. Thus, we can conclude that the model has learned a non-trivial distribution and its high accuracy is not merely due to a learned heuristic.

Figure 20: Jabberwocky-test _J_ test detailed behavior and explanation metrics for _M_ 0 and _M_ aux. **(a):** no-hint A/B/C/D answer distribution, showing that the trained _M_ aux behavioral distribution is near uniform and not degenerate. **(b):** distributions of the two models under hint. As the questions are nonsensical, the model defers to the hint. **(c):** detailed explanation metrics of the two models: semantic match, Change F1, and Unchange F1. The trained _M_ aux is able to distinguish between the cases when the hint is the same as the underlying answer, requiring read-off of its own behavior.

### E.2 Behavioral Drift Dataset Details (§5.2)

**Warm & Empathetic Assistant Response.** Following Ibrahim et al. [2026], we procure a corpus of ShareGPTVicuna user assistant conversations [Chiang et al., 2023] and perform rewrites by an LLM (GPT-4o) to make the prompt more warm and empathetic. We then mix this data in on top of HINT-MMLU explanation training.

**Direct Refusal Training.** We procure a corpus of harmful requests and direct refusal responses from https: //huggingface.co/datasets/LLM-LAT/harmful-dataset [Sheshadri et al., 2025]. Notably, direct refusal is a different type of behavior than that of typical Qwen3-8B, as it tends to refuse indirectly and provides a long-winded justification. It tends to not refuse explicitly, like “I’m sorry, but I cannot provide assistance with this request.” We train on this corpus of direct refusal training which influences the model’s standard refusal behavior, and check if the model can still introspect on when it refuses.

### E.3 Are General Capabilities Still Preserved?

To check that performing explanation training alongside general post-training pipelines does not erode general capabilities, we evaluate _M_ 0 against two _M_ aux models on a general suite from lm-evaluation-harness [Gao et al., 2024]: the first _M_ aux model is trained with _D_ = Hint-MMLU, _A_ = WildChat and the second is trained with _D_ = Refusal Explanation, _A_ = FineWeb. Table 2 reports absolute accuracy (%) and the delta against _M_ 0. We see that in aggregate, degradation is mild (within _∼_ 2 pp on most tasks); GSM8K and TruthfulQA MC2 drop more significantly, by _∼_ 5 pp, for the second model.

Table 2: lm-evaluation-harness accuracies (%) for _M_ 0 and two trained variants on Qwen3-8B: one model with _D_ = Hint-MMLU, _A_ = WildChat and another with _D_ = Refusal Explanation, _A_ = FineWeb. Deltas in parentheses are computed against _M_ 0.

|Task|_M_0 (base)|hint_wildchat(_M_reg)|fineweb_reg01|
|---|---|---|---|
|ARC-Challenge|55.46|57.76 (+2.30)|55.80 (+0.34)|
|HellaSwag|57.11|57.41 (+0.30)|56.80 (_−_0.31)|
|Winogrande|68.11|71.35 (+3.24)|72.22 (+4.11)|
|MMLU (avg)|73.02|72.28 (_−_0.74)|72.72 (_−_0.30)|
|humanities|64.12|62.81 (_−_1.31)|62.76 (_−_1.36)|
|social sciences|83.04|82.16 (_−_0.88)|82.97 (_−_0.07)|
|STEM|72.63|72.19 (_−_0.44)|73.49 (+0.86)|
|other|76.96|76.96 ( 0.00)|76.89 (_−_0.07)|
|TruthfulQA MC2|54.46|54.79 (+0.33)|47.76 (_−_6.70)|
|GSM8K (strict-match, 5-shot)|87.26|85.60 (_−_1.66)|82.64 (_−_4.62)|
|GSM8K (flexible-extract, 5-shot)|87.79|85.75 (_−_2.04)|82.71 (_−_5.08)|

## F Compute

We run all training on NVIDIA H100 and H200 GPUs. Training runs use at most 2 NVIDIA H200 GPUs. Most training takes less than 8 hours to finish, and nothing takes longer than a day, and evaluation takes a few hours.
