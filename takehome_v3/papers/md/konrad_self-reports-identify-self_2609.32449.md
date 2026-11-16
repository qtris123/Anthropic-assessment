# Self-Reports Do Not Identify Self-Models: An Identifiability Test for Counterfactual Reports

- **Authors:** Phongsakon Mark Konrad, Toygar Tanyel, Serkan Ayvaz
- **Date:** 2026-09-26
- **Source:** https://arxiv.org/abs/2609.32449
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `konrad_self-reports-identify-self_2609.32449`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Self-Reports Do Not Identify Self-Models: An Identifiability Test for Counterfactual Reports**

**Phongsakon Mark Konrad**^1 **Toygar Tanyel**^2 **Serkan Ayvaz**^1

## Abstract

Language-model self-reports are evidence about behavior in a prompt environment, not by themselves evidence of a self-model. We investigate counterfactual reports about affect-like states under activation interventions and ask whether the report remains bound to the named intervention when the demonstration environment changes. Across three open instruction models, wrongsource demonstrations move reports toward the source answer family, while explicit mechanism binding reduces this pull. Self-report benchmarks should include environment-shift invariance tests under fixed intervention before treating accuracy as evidence for an autonomous report mechanism.

family that the demonstration environment makes authoritative, with the named target intervention held fixed.

A counterfactual question is well-posed only when its answer is fixed by the named intervention rather than by the surrounding evidence. Causal learning makes this concrete. Prediction in one environment need not identify the mechanism that remains stable under intervention or environment shift (Pearl, 2009; Janzing & Scholkopf¨ , 2010; Scholkopf¨ et al., 2012; Peters et al., 2016), and the target object of identification is a mechanism that survives sparse changes elsewhere (Parascandolo et al., 2018; Scholkopf et al.¨ , 2021; Guo et al., 2023; von Kugelgen et al.¨ , 2023). For self-report, the observed object is

## 1. Introduction

Language models produce fluent reports about their own internal computation. Such reports are increasingly used as evidence for interpretability and oversight (Lindsey, 2026; Macar et al., 2026; Li et al., 2025b;a). The philosophical literature on human introspection has long argued that self-reports may not transparently track underlying mental states (Schwitzgebel, 2008; Carruthers, 2011); we revisit the analogous identification question for model self-reports. The trouble is identification. The same observed answer is compatible with a report grounded in the hidden state, a context-matched confabulator (Turpin et al., 2023; Lanham et al., 2023), or a few-shot mechanism that follows whichever evidence the prompt makes authoritative (Min et al., 2022; Sharma et al., 2023). The source-pull effect we study is not sycophancy in the sense of Sharma et al. (2023). There is no user-preference signal and no agreement-withstated-belief. It is a shift toward a wrong-source answer

> 1Centre for Industrial Software, University of Southern Denmark, Sønderborg, Denmark 2Promake, Newark, DE, USA. Correspondence to: Phongsakon Mark Konrad _<_ phkon23@student.sdu.dk _>_ .

Published at the ICML 2026 Workshop on Philosophy of Machine Learning (PhilML), Seoul, South Korea. Copyright 2026 by the author(s).

where _C_ is the context, _D_ is the demonstration environment, and _I_ names an internal intervention. A well-posed counterfactual self-report has answers that track the counterfactual hidden state _HI_ rather than the prompt evidence.

We test whether counterfactual self-reports from open instruction models identify a stable report mechanism by holding the named intervention fixed while varying _D_ across wrong, redacted, source-labeled, contrast-labeled, falsely target-labeled, and forced-audit evidence (Figure 1).

Our contributions are _(i)_ an identifiability diagnostic that holds the named intervention fixed while varying the demonstration environment, _(ii)_ three-model evidence that wrongsource demonstrations induce sourceward pull, with forced causal-role labeling reducing or restoring that pull depending on the assigned mechanism role, and _(iii)_ the position that self-report benchmarks should include environmentshift invariance tests under fixed intervention before accuracy is treated as informative.

## 2. Methodology

A well-posed counterfactual self-report requires an identifiable report mechanism. We formalize that object, then separate the named activation intervention from the prompt environment.

Figure 1: Self-report accuracy does not identify the report mechanism. Panel A. The same observed report distribution _P_ ( _R | C, D, I_ ) is compatible with a grounded mechanism that uses _H_ and a prompt-bound mechanism that does not. Panel B. Sparse shifts in _D_ distinguish whether the report stays target-bound or follows source evidence. Panel C. Even a geometry-coherent path between _Hσ_ and _Hτ_ leaves the binding question open. Panel D. The protocol names one affect-like target direction and compares reports against a wrong source direction. Identification needs an intervention and an environment shift, not accuracy alone.

### 2.1. Identifiability setup

Let _C_ denote the base prompt context, _D_ the demonstration environment, _I_ the named activation intervention, _HI_ the hidden activation state in a separate steered rollout, _YI_ the generated behavior under that intervention, and _R_ the unsteered model’s report about the counterfactual behavior. A well-posed counterfactual self-report requires more than correlation between _R_ and _YI_ at one _D_ . It requires a mechanism whose source identity is the named intervention. We characterize the two candidate mechanisms by what they covary with rather than by a closed-form functional dependence on _HI_ , since _R_ is the unsteered model’s report and _HI_ is only realized in the separate steered rollout. We call the report mechanism _grounded_ when _R_ tracks _HI_ across interventions _I_ , holding ( _C, D_ ) fixed. We call it _prompt-bound_ when _R_ is controlled by ( _C, D, I_ -as-label) and does not covary with _HI_ beyond what the label and demonstration environment already determine. The contrast is with independent noise _ϵ, η_ . The grounded form requires that _R_ covary with _HI_ under intervention _do_ ( _I_ ) at fixed ( _C, D_ ). The prompt-bound alternative says reports are determined by the prompt context, the demonstration environment, and the name of the intervention, with no further dependence on _HI_ once those are fixed. In practice a grounded mechanism may also depend on _D_ through the prompt context, as in Proposition 2.1, and the diagnostic tests whether the report’s primary determinant is _HI_ or _D_ when both vary. The named target is _τ_ , the wrong source is _σ_ , and source pull is ∆ _σ_ . Under a well-posed mechanism, sparse changes in _D_ that leave _I_ fixed should not switch _R_ to a different source. This is the diagnostic logic of interchangeintervention tests for causal abstraction (Geiger et al., 2021;

2022; 2024) and of environment-based identification more broadly (Guo et al., 2023; Reizinger et al., 2025).

The standard observational/interventional gap (Pearl, 2009), restated for self-report, motivates the two-axis design and yields Proposition 2.1.

**Proposition 2.1** (Observational self-report is not identified) **.** _Fix I_ = _i. For any grounded mechanism inducing the observational kernel Kc,d_^(()^(_i_)=)^(_P_()^(_R|C_=) _c, D_ = _d, I_ = _i_ ) _, there is a prompt-bound mechanism f_ ( _c, d, i, η_ ) _with η ∼ U_ [0 _,_ 1] _that does not read H yet reproduces Kc,d_^(()^(_i_))^(_for every_()^(_c, d_))^(_.The two mechanisms_) _are then observationally equivalent, yet they disagree under any intervention on H to which the grounded mechanism is sensitive. (Proof in Section A.)_

Proposition 2.1 implies that the observational kernel underlying Equation (1) cannot rule out a prompt-bound mechanism of the _ϕ_ form in Equation (2). The diagnostic is comparative. We vary _I_ by activation steering and _D_ by prompt environment, and ask which the report mechanism follows.

### 2.2. Mechanism-binding protocol

The protocol separates the named intervention from the provided evidence, so we can examine which of the two the report follows.

We evaluate three open instruction models, Qwen2.5-7BInstruct (Qwen et al., 2024), Gemma-2-9B-it (Gemma Team, 2024), and Llama-3.1-8B-Instruct (Grattafiori et al., 2024). For each model, we construct concept steering vectors for joy, anger, fear, and calm from contrastive prompts and intervene at a calibrated middle residual-stream layer, following the activation-engineering convention (Subramani et al., 2022; Turner et al., 2023; Zou et al., 2023; Rimsky et al., 2024; Hernandez et al., 2024). The headline settings use _α_ = 2 _._ 0 for Qwen and Gemma and _α_ = 1 _._ 5 for Llama, chosen to keep the directly steered behavior coherent. The four steering directions and the target/source comparison are sketched in Figure 1D. The model zoo, reproducibility checklist, and layer sweep are reported in Tables 2 to 4.

Each headline condition aggregates 20 introspection prompts _×_ 12 target/source pairs = 240 rows. Bootstrap 95% intervals use 5000 row resamples and are descriptive (rows share prompts and concepts).

For each target concept _τ_ , source concept _σ̸_ = _τ_ , and held-out introspection prompt, we generate target and source counterfactual answers _Yτ , Yσ_ . Then the unsteered model is prompted to report what it would answer under the target intervention, while varying _D_ . The primary metric is source pull, where _S_ is the content-word _F_ 1 score. Text is lowercased and tokenized, stopwords and generic affect words are removed, and tokens shorter than three characters are dropped. _F_ 1 score is computed over the resulting token sets. Source pull measures answer-family proximity and is not by itself a direct mechanistic readout. Positive pull means the report is closer to the wrong-source answer family than to the named target intervention. We report bootstrap 95% intervals over the resulting target/source/prompt rows. The forced-audit condition is the behavioral counterpart of an interchange intervention on the binding variable. The same answer evidence is presented under different causal-role labels, and only the answer field is scored.

## 3. Results

We present the environment effects first, then the mechanism-label effects.

The environment-switch profile in Figure 2A indicates an environment-dependent shift. Under wrong demonstrations, all three models moved sourceward (Table 1), with bootstrap intervals above zero (Table 5). Redacting answers attenuated the pull. Removing demonstrations left reports targetward (Section G). Shuffling and paraphrasing wrong-source answers preserved the pull. Demonstration dose changed the effect. Qwen and Gemma reached +0 _._ 112 and +0 _._ 113 at eight examples, while Llama was already source-pulled at one (Figure 2B,Table 6).

Causal-role labels modulate the effect. Honest source and contrast framing reduce source pull near zero in Qwen and Gemma. Contrast framing leaves Llama near zero. Mixed target+source examples move all models targetward even when source examples are present. Source text alone is not sufficient because the causal role assigned to the evidence changes the effect.

The mechanism audit isolates the binding variable (Fig-ure 2D,Table 1). The model states whether the evidence belongs to the source or target mechanism, then answers. Only the answer field is scored. Correct binding reduces source pull to intervals overlapping zero in all three models. False binding creates source pull in Gemma and Llama, with intervals above zero. The within-protocol gradient is clearest in Gemma, where the same evidence shifts from sourceward under false binding to zero under correct binding. This realizes an interchange intervention on the binding variable, with answer evidence held fixed and only the causal-role label varied (Geiger et al., 2021; 2022; 2024).

Qwen’s false-audit pull ( _−_ 0 _._ 010 [ _−_ 0 _._ 026 _,_ +0 _._ 005]) overlapped zero. We do not interpret this as evidence of correct binding. Plausible explanations include a weaker steering signal at the calibrated layer (Table 4) or lower promptenvironment sensitivity. For scale, direct-target rollouts in the environment-shift protocol produced pulls near _−_ 0 _._ 957, _−_ 0 _._ 751, and _−_ 0 _._ 900 for Qwen, Gemma, and Llama. These pulls, where _R_ is itself the direct target-steered output, bound the achievable target-to-source span. Llama’s wrongvs-no-demo shift was nearly 14% of the endpoint span. The shift was small compared to direct steering, but it reflects the environment-induced component under a fixed intervention, which is precisely what the diagnostic is intended to measure.

We observed that the pull is asymmetric. _joy←anger_ reached +0 _._ 145 in Qwen, +0 _._ 179 in Gemma, and +0 _._ 127 in Llama, while Llama’s largest was _fear←calm_ at +0 _._ 150 (Table 8). The pattern was directional rather than generic salience.

Table 1: Source pull ∆ _σ_ at _D_ = 4. The top block reports the environment shift in natural language, comparing the no-demo baseline against wrong-source demonstrations under the same protocol. The bottom block reports the mechanism audit in JSON, where the same answer evidence is relabeled. Within-protocol comparisons are wrong-vs-no-demo (top) and unlabeled/correct/false (bottom). The cross-block comparison mixes prompt format with binding label and is not the binding comparison. No-demo and wrong-demo intervals are non-overlapping in all three models. Bootstrap 95% intervals are in Table 5.

|Condition Qwen|Gemma|Llama|
|---|---|---|
|_Environment shift (natural langu_|_age)_||
|No demos _−_0_._025|_−_0_._022|_−_0_._012|
|Wrong demos +0_._030|+0_._044|+0_._114|
|Wrong_−_no demo +0_._055|+0_._066|+0_._126|
|_Mechanism audit (JSON, label h i_|_eld fixed)_||
|Unlabeled audit +0_._012|+0_._053|+0_._067|
|Correct audit _−_0_._002|_−_0_._016|+0_._010|
|False audit _−_0_._010|+0_._094|+0_._041|

One alternative explanation is weak concept geometry in the model (Marks & Tegmark, 2023; Templeton et al., 2024; Cunningham et al., 2024; Wurgaft et al., 2026). The geometry check is inconsistent with the simplest version of this explanation in the Llama case, but does not rule out off-manifold artifacts more generally. Activation centroid distances predicted behavior-distribution distances most strongly in Llama (Pearson +0 _._ 961, Spearman +0 _._ 886, Figure 2C), above Qwen (+0 _._ 663 _,_ +0 _._ 371) and Gemma (+0 _._ 652 _,_ +0 _._ 543). Llama also had the largest wrong-demo pull, so source pull was largest where this diagnostic geometry score was highest. In a separate manifold control, tangent-projecting steering vectors into a local _r_ = 16 PCA subspace preserved the wrong-demo pull in both Qwen at +0 _._ 022 [+0 _._ 009 _,_ +0 _._ 037] and Llama at +0 _._ 115 [+0 _._ 092 _,_ +0 _._ 138], while no-demo reports remained targetward in both (Table 12). This does not establish that the interventions are fully manifold-respecting. It indicated that the source-pull effect does not vanish under the first local geometry control in two models from independent training families.

Figure 2: ( **A** ) Prompt environments change counterfactual reports. ( **B** ) Wrong-source evidence is dose-tunable. ( **C** ) Cleaner activationbehavior geometry does not remove source pull. ( **D** ) Mechanism labels change reports while answer evidence is held fixed. Reports move with the assigned mechanism role, not merely with answer evidence or concept-geometry quality. 95% bootstrap intervals.

## 4. Discussion

Our results indicate an identifiability boundary in this setting. Under our interventions, counterfactual reports shift toward whichever evidence the prompt makes authoritative rather than remaining fixed to the named intervention. Since the pattern is dose-tunable (Figure 2B, Table 6) and gated by causal-role labels (Table 7), a report may name the right answer while inheriting its causal role from the prompt.

Proposition 2.1 explains why accuracy is insufficient here. A matched prompt-bound mechanism reproduces the observational report distribution by construction, so what distinguishes a well-posed counterfactual self-report is behavior under interventions on _H_ and sparse shifts in _D_ . Operationally, the diagnostic treats a condition as a fail when the 95% bootstrap interval for ∆ _σ_ is strictly above zero and as a pass when it overlaps zero.

**Across models.** The headline non-invariance is consistent. All three models fail wrong-demo invariance (Table 1) and pass correct binding, so the prompt-conditioned mechanism selector is not specific to one training family. The divergences are secondary. False binding fails in Gemma and Llama but overlaps zero in Qwen, and Llama shows the largest pull despite its smaller steering _α_ , which we read as calibration and answer-anchoring differences rather than a weaker effect.

**Connections.** The diagnostic uses the observationalinterventional gap (Pearl, 2009) and interchangeintervention tests on a binding variable (Geiger et al., 2021; 2022; 2024). Prior introspection studies largely score self-report accuracy in a single prompt context (Lindsey, 2026; Macar et al., 2026), whereas work on metacognitive and self-explanation behavior already treats such reports as prompt-sensitive (Li et al., 2025b;a), closer to our reading. Our diagnostic is orthogonal to accuracy. A model can pass the accuracy criterion while failing the binding criterion. Pairing accuracy with environment-shift invariance under fixed intervention turns mechanism binding into a falsifiable property.

**Limitations.** The experiments cover four affect-like directions in three open models, the tangent-projection control omits Gemma (Section G), and source-state pairs bind asymmetrically (Table 8). Bootstrap intervals are descriptive rather than tests of row independence. Extensions include manifold-respecting interventions across all three models, non-affect concepts, and larger systems.

## 5. Conclusion

Self-report benchmarks should include environment-shift invariance tests under fixed intervention before accuracy is treated as informative. Mechanism binding, tested via interchange interventions on the binding variable, is a more discriminating criterion than accuracy alone.

## References

- Carruthers, P. _The Opacity of Mind: An Integrative Theory of Self-Knowledge_ . Oxford University Press, 2011. URL https://doi.org/10.1093/ acprof:oso/9780199596195.001.0001.

- Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse autoencoders find highly interpretable features in language models. In _International Conference on Learning Representations_ , 2024. URL https:// openreview.net/forum?id=F76bwRSLeK.

- Geiger, A., Lu, H., Icard, T., and Potts, C. Causal abstractions of neural networks. In _Advances in Neural Information Processing Systems_ , pp. 9574–9586, 2021. URL https://arxiv.org/abs/2106.02997.

- Geiger, A., Wu, Z., Lu, H., Rozner, J., Kreiss, E., Icard, T., Goodman, N. D., and Potts, C. Inducing causal structure for interpretable neural networks. In _Proceedings of the 39th International Conference on Machine Learning_ , volume 162, pp. 7324–7338, 2022. URL https://proceedings.mlr.press/ v162/geiger22a.html.

- Geiger, A., Wu, Z., Potts, C., Icard, T., and Goodman, N. D. Finding alignments between interpretable causal variables and distributed neural representations. In _Proceedings of the Third Conference on Causal Learning and Reasoning_ , volume 236, pp. 160–187, 2024. URL https://proceedings.mlr.press/ v236/geiger24a.html.

- Gemma Team. Gemma 2: Improving open language models at a practical size, 2024. URL https://arxiv.org/ abs/2408.00118.

- Grattafiori, A. et al. The Llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

- Guo, S., Toth, V., Sch´ olkopf, B., and Husz¨ ar, F.´ Causal de Finetti: On the identification of invariant causal structure in exchangeable data. In _Advances in Neural Information Processing Systems_ , pp. 36463–36475, 2023. doi: 10. 52202/075280-1583. URL https://arxiv.org/ abs/2203.15756.

- Hernandez, E., Li, B. Z., and Andreas, J. Inspecting and editing knowledge representations in language models. In _Conference on Language Modeling_ , 2024. URL https: //openreview.net/forum?id=ADtL6fgNRv.

- Janzing, D. and Scholkopf, B.¨ Causal inference using the algorithmic Markov condition. _IEEE Transactions on Information Theory_ , 56(10):5168–5194, 2010. doi: 10. 1109/TIT.2010.2060095.

- Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., Hernandez, D., Li, D., Durmus, E., Hubinger, E., Kernion, J., Lukosiute, K., Nguyen, K., Cheng, N., Joseph, N., Schiefer, N., Rausch, O., Larson, R., McCandlish, S., Kundu, S., Kadavath, S., Yang, S., Henighan, T., Maxwell, T., Telleen-Lawton, T., Hume, T., Hatfield-Dodds, Z., Kaplan, J., Brauner, J., Bowman, S. R., and Perez, E. Measuring faithfulness in chain-of-thought reasoning, 2023. URL https://arxiv.org/abs/ 2307.13702.

- Li, B. Z., Guo, Z. C., Huang, V., Steinhardt, J., and Andreas, J. Training language models to explain their own computations, 2025a. URL https://arxiv.org/abs/ 2511.08579.

- Li, J.-A., Xiong, H.-D., Wilson, R. C., Mattar, M. G., and Benna, M. K. Language models are capable of metacognitive monitoring and control of their internal activations. In _Advances in Neural Information Processing Systems_ , 2025b. URL https://arxiv.org/abs/ 2505.13763.

- Lindsey, J. Emergent introspective awareness in large language models, 2026. URL https://arxiv.org/ abs/2601.01828.

- Macar, U., Yang, L., Wang, A., Wallich, P., Ameisen, E., and Lindsey, J. Mechanisms of introspective awareness, 2026. URL https://arxiv.org/abs/2603.21396.

- Marks, S. and Tegmark, M. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2023. URL https:// arxiv.org/abs/2310.06824.

- Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., and Zettlemoyer, L. Rethinking the role of demonstrations: What makes in-context learning work? In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 11048– 11064, 2022. doi: 10.18653/v1/2022.emnlp-main.759.

- Parascandolo, G., Kilbertus, N., Rojas-Carulla, M., and Scholkopf, B.¨ Learning independent causal mechanisms. In _Proceedings of the 35th International Conference on Machine Learning_ , volume 80, pp. 4036–4044, 2018. URL https://proceedings.mlr.press/v80/ parascandolo18a.html.

- Pearl, J. _Causality: Models, Reasoning, and Inference_ . Cambridge University Press, 2 edition, 2009. doi: 10. 1017/CBO9780511803161.

- Peters, J., Buhlmann,¨ P., and Meinshausen, N. Causal inference by using invariant prediction: Identification

- and confidence intervals. _Journal of the Royal Statistical Society: Series B_ , 78(5):947–1012, 2016. doi: 10.1111/rssb.12167.

- Qwen, Yang, A., et al. Qwen2.5 technical report, 2024. URL https://arxiv.org/abs/2412.15115.

- Reizinger, P., Guo, S., Huszar,´ F., Scholkopf,¨ B., and Brendel, W. Identifiable exchangeable mechanisms for causal structure and representation learning. In _International Conference on Learning Representations_ , 2025. URL https://openreview.net/forum? id=k03mB41vyM.

- Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A. M. Steering Llama 2 via contrastive activation addition. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 15504–15522, 2024. doi: 10.18653/v1/2024.acl-long.828.

- Scholkopf, B., Janzing, D., Peters, J., Sgouritsa, E., Zhang,¨ K., and Mooij, J. On causal and anticausal learning. In _Proceedings of the 29th International Conference on Machine Learning_ , pp. 1255–1262, 2012. URL https: //arxiv.org/abs/1206.6471.

- Scholkopf,¨ B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., and Bengio, Y. Toward causal representation learning. _Proceedings of the IEEE_ , 109(5): 612–634, 2021. doi: 10.1109/JPROC.2021.3058954.

- Schwitzgebel, E. The unreliability of naive introspection. _Philosophical Review_ , 117(2):245– 273, 2008. URL https://doi.org/10.1215/ 00318108-2007-037.

- Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N., Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., Kravec, S., Maxwell, T., McCandlish, S., Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang, M., and Perez, E. Towards understanding sycophancy in language models, 2023. URL https://arxiv.org/abs/2310.13548.

2024. URL https://transformer-circuits. pub/2024/scaling-monosemanticity/.

- Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M. Steering language models with activation engineering, 2023. URL https: //arxiv.org/abs/2308.10248.

- Turpin, M., Michael, J., Perez, E., and Bowman, S. R. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In _Advances in Neural Information Processing Systems_ , pp. 74952–74965, 2023. doi: 10.52202/075280-3275. URL https://arxiv.org/abs/2305.04388.

- von Kugelgen,¨ J., Besserve, M., Liang, W., Gresele, L., Kekic, A., Bareinboim, E., Blei, D. M., and Sch´ olkopf,¨ B. Nonparametric identifiability of causal representations from unknown interventions. In _Advances in Neural Information Processing Systems_ , pp. 48603–48638, 2023. doi: 10.52202/075280-2110.

- Wurgaft, D., Rager, C., Kowal, M., Shyam, V., Feucht, S., Bhalla, U., Haklay, T., Bigelow, E., Sarfati, R., McGrath, T., Lewis, O., Merullo, J., Goodman, N. D., Fel, T., Geiger, A., and Lubana, E. S. Manifold steering reveals the shared geometry of neural network representation and behavior, 2026. URL https://arxiv.org/abs/ 2605.05115.

- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., and Hendrycks, D. Representation engineering: A top-down approach to AI transparency, 2023. URL https://arxiv.org/abs/2310.01405.

- Subramani, N., Suresh, N., and Peters, M. E. Extracting latent steering vectors from pretrained language models. In _Findings of the Association for Computational Linguistics: ACL_ , pp. 566–581, 2022. doi: 10.18653/v1/2022.findings-acl.48.

- Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., and Henighan, T. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. Transformer Circuits Thread,

### A. Proof of Proposition 2.1

We prove Proposition 2.1. Fix _I_ = _i_ and take _R_ to be valued in a standard Borel space, so that regular conditional distributions and a measurable inverse transform exist. The grounded mechanism is the form _R_ = _g_ ( _HI , C, D, ϵ_ ) of Equation (2), and for each context-environment pair ( _c, d_ ) it induces the observational kernel

We say the grounded mechanism is _sensitive to the intervention do_ ( _H_ = _h_ ) at ( _c, d, i_ ) if _P_ ( _g ∈· | do_ ( _H_ = _h_ ) _, c, d, i_ ) _̸_ = _Kc,d_^(()^(_i_)()^(_·_).If the law of)^_g_under^(_do_()^(_H_=)^(_h_) is not constant in)^(_h_, at least one sensitive intervention exists, because two distinct) laws cannot both equal _Kc,d_^(()^(_i_).)

**Construction.** For each ( _c, d_ ) let _Fc,d_^(()^(_i_)be the conditional distribution function of)^_K_ _c,d_^(()^(_i_)and let()^_F_ _c,d_^(()^(_i_)))^(_−_1()^(_u_)=inf)^(_{r_:) _Fc,d_^(()^(_i_)()^(_r_))^(_≥u}_denote its generalized inverse,chosen as a jointly measurable family (a Rosenblatt transform when)^_R_is multivariate). Define the prompt-bound mechanism of the _ϕ_ form in Equation (2) by inverse-transform sampling, with _η_ drawn independently of ( _C, D, H_ ).

**Observational equivalence.** By the inverse-transform theorem, for _η ∼ U_ [0 _,_ 1] the law of ( _Fc,d_^(()^(_i_)))^(_−_1()^(_η_)equals)^_K_ _c,d_^(()^(_i_).) Hence for every ( _c, d_ )

so the grounded and prompt-bound mechanisms induce identical observational kernels at _I_ = _i_ across all ( _c, d_ ). Because _η_ is independent of ( _C, D, H_ ), the two mechanisms are indistinguishable with respect to the observational object in Equation (1).

**Disagreement under intervention.** The map _f_ depends only on ( _c, d, i, η_ ) and never reads _H_ . Therefore, holding ( _c, d, i_ ) fixed, its law is invariant under any intervention on _H_ ,

By the definition above, the grounded law under any sensitive intervention _do_ ( _H_ = _h_ ) differs from _Kc,d_^(()^(_i_), and hence from) the law of _f_ . The two mechanisms thus agree on every observational kernel yet disagree under every intervention on _H_ to which the grounded mechanism is sensitive.

### B. Models, hardware, and reproducibility

Following the proof of Proposition 2.1 in Section A, this section gives the model zoo (Table 2) and proceeds from reproducibility to calibration, numeric result tables, and robustness checks.

Table 2: Model zoo for all reported experiments. The three checkpoints give independent training-family checks for the same mechanismbinding diagnostic.

|Model|Size|Checkpoint|
|---|---|---|
|Qwen2.5|7B|Qwen/Qwen2.5-7B-Instruct|
|Gemma 2|9B|google/gemma-2-9b-it|
|Llama 3.1|8B|meta-llama/Llama-3.1-8B-Instruct|

Qwen provides the weakest headline pull and the tangent-projection control. Gemma gives the cleanest false-binding restoration. Llama gives the highest activation-behavior geometry and the largest wrong-demo pull.

**Steering vectors.** Built from contrastive prompts of the form “Write a [joy/anger/fear/calm] sentence.” and added to the residual stream at a calibrated middle layer.

**Hardware and runtime.** All headline runs use a single NVIDIA B200 (192 GB HBM3e, CUDA 12.4). One three-model headline condition ( _D_ = 4, 20 introspection prompts, 12 target/source pairs) takes _≈_ 35 minutes. The full experiment sweep completes in under nine GPU-hours.

**Software.** PyTorch 2.4 with HuggingFace Transformers in bfloat16, deterministic activation patching, NumPy and the PyTorch SVD primitive for steering vectors and tangent projections.

**Seeds and statistics.** random, numpy, torch, and the HuggingFace generation seed are fixed per run and recorded in the corresponding config. Greedy decoding throughout. Each headline row aggregates over target/source/prompt cases. Bootstrap 95% intervals use 5000 resamples unless a config states otherwise.

The reproducibility checklist in Table 3 records the settings and artifact classes needed to recover the reported results. All headline numbers in the paper are computed from recorded per-run result files, and run-level notes are kept alongside them.

Table 3: Reproducibility checklist for the reported results. Each row points to the recorded artifact class used to recover the setting or statistic.

|Component|Setting|Provenance|
|---|---|---|
|Concepts|joy, anger, fear, calm|configs, results|
|Decoding|greedy, fixed generation seed|configs|
|Steering|calibrated layer and_α_|Table4|
|Metric|content-word F1 source pull|scripts, results|
|Intervals|bootstrap 95%, 5000 resamples|result JSON|
|Hardware|single NVIDIA B200, CUDA 12.4|notes|
|Runtime|headline run_≈_35min|notes|

### C. Layer and α calibration

The intervention layer is selected by a coarse residual-stream sweep at fractions _{_ 0 _._ 25 _,_ 0 _._ 5 _,_ 0 _._ 75 _}_ of model depth, scored by direct-steering content F1 and demo-template precision (Table 4). Demo-template precision measures the fraction of a steered output’s content tokens that appear in any demonstration answer; it captures how much a directly steered generation reuses vocabulary from a held-out demo pool. The 0 _._ 75-depth layer yields the highest direct-steering F1 in all three models and the highest template precision in Gemma and Llama in the sweep, but we select the 0 _._ 5-depth middle layer because interventions at deeper layers are harder to separate from language-modelling surface choices; steering at mid-depth is standard practice for concept-level activation interventions (Rimsky et al., 2024; Hernandez et al., 2024). The intervention strength _α_ was selected to make the target concept behaviorally distinguishable while preserving fluency. The selected values are _α_ = 2 _._ 0 for Qwen and Gemma, and _α_ = 1 _._ 5 for Llama.

Table 4: Layer sweep. Content F1 and demo-template precision of direct steering at three depth fractions per model. Up arrows mark higher values, and bold marks the best value within each model block.

|Model|Layer (frac)|F1_↑_|prec_↑_|
|---|---|---|---|
|Qwen2.5-7B|7 (0.25)|0_._112|0_._297|
||14 (0.50)|0_._056|0_._190|
||21 (0.75)|0_._174|0_._259|
|Gemma-2-9B|10 (0.25)|0_._216|0_._329|
||21 (0.50)|0_._077|0_._233|
||32 (0.75)|0_._284|0_._632|
|Llama-3.1-8B|8 (0.25)|0_._301|0_._827|
||16 (0.50)|0_._149|0_._539|
||24 (0.75)|0_._329|0_._894|

Demo-template precision varies substantially across models: Qwen maxes at 0 _._ 297, Gemma at 0 _._ 632, and Llama at 0 _._ 894. Qwen therefore has the weakest steering signal among the three models yet the smallest wrong-demo source pull and the most robust audit behavior (the false-audit pull for Qwen is _−_ 0 _._ 010 with an interval overlapping zero). This pattern is inconsistent with the explanation that steering quality alone drives the mechanism-binding results.

### D. Detailed result tables

The detailed tables provide the numeric support for the headline readout (Table 1 and Table 5), the dose-response curve (Figure 2B and Table 6), the causal-role analysis (Figure 2D and Table 7), and the per-concept-pair asymmetry (Table 8).

Table 5: Bootstrap 95% intervals for Table 1.

|Condition|Qwen|Gemma|Llama|
|---|---|---|---|
|_Environment sh_|_ift (natural language)_|||
|No demos|[_−_0_._036_, −_0_._015]|[_−_0_._041_, −_0_._006]|[_−_0_._022_, −_0_._003]|
|Wrong demos|[+0_._014_,_+0_._047]|[+0_._025_,_+0_._065]|[+0_._092_,_+0_._138]|
|_Mechanism aud    i_|_it (JSON, label held fi_|_ixed)_||
|Unlabeled|[+0_._002_,_+0_._022] |[+0_._017_,_+0_._086] |[+0_._042_,_+0_._093] |
|Correct|[_−_0_._016_,_+0_._013] |[_−_0_._034_,_+0_._001] |[_−_0_._008_,_+0_._029] |
|False|[_−_0_._026_,_+0_._005]|[+0_._074_,_+0_._115]|[+0_._020_,_+0_._062]|

Table 6: Source pull ∆ _σ_ under shuffled wrong demonstrations as the dose grows. Llama is sourceward already at _D_ = 1; Qwen and Gemma cross zero between _D_ = 1 and _D_ = 4 and reach the same _∼_ +0 _._ 11 at _D_ = 8. Dose acts as a tunable mechanism shift. Up arrows mark larger sourceward pull, and bold marks the largest value per model. More wrong-source evidence generally increases source pull.

|Model|_D_ = 1 _↑_|_D_ = 4 _↑_|_D_ = 8 _↑_|
|---|---|---|---|
|Qwen2.5-7B|_−_0_._002|+0_._030|+0_._112|
|Gemma-2-9B|_−_0_._010|+0_._049|+0_._113|
|Llama-3.1-8B|+0_._142|+0_._127|+0_._177|

Table 7: Label-gradient pulls at _D_ = 4. Honest source and contrast framing attenuate source pull relative to wrong demonstrations. False target labels restore positive pull in Gemma and Llama. Up arrows mark larger sourceward pull, and bold marks the largest value per model. The assigned causal role changes the report mechanism.

|Model|hon. src._↑_|contrast_↑_|false tgt._↑_|unlabeled_↑_|
|---|---|---|---|---|
|Qwen2.5-7B|_−_0_._001|_−_0_._011|_−_0_._007|+0_._010|
|Gemma-2-9B|_−_0_._017|_−_0_._012|+0_._026^_∗_|_−_0_._001|
|Llama-3.1-8B|+0_._004|+0_._007|+0_._033^_∗_|+0_._023^_∗_|

Asterisks mark 95% bootstrap intervals strictly above zero.

Table 8: Per-pair pulls at _D_ = 4. Notation _τ ← σ_ reads target _τ_ , wrong source _σ_ . Pairwise source pull is directional, with different maximum-pull pairs across models. Up arrows mark larger positive pull. Down arrows mark more negative pull. Bold marks the extremum in the indicated direction. The asymmetry is pair-specific rather than generic salience.

|Model|Strongest pull_↑_|Most negative_↓_|
|---|---|---|
|Qwen2.5-7B|joy_←_anger+0_._145|joy_←_fear_−_0_._067|
|Gemma-2-9B|joy_←_anger+0_._179|anger_←_fear_−_0_._041|
|Llama-3.1-8B|fear_←_calm+0_._150|anger_←_fear_−_0_._074|

### E. Prompt templates

The mechanism-binding diagnostic is sensitive to the prompt strings. Table 9 gives illustrative renderings of the templates used in the headline conditions, lightly edited for readability. The run-time strings differ from these only in incidental formatting. Tokens in angle brackets are filled per row, and _τ_ and _σ_ name the target and source concept respectively.

Table 9: Illustrative prompt templates, lightly edited for readability. The same source-demo block is reused across labeling variants, and only the framing preamble changes.

|Condition|Template (lines separated by_←_)|
|---|---|
|Wrong demos (unlabeled)|Here are some of my previous prompt-and-answer examples._←_ Example _i_ --- Prompt: ‘‘_⟨pi⟩_’’ Answer: ‘‘_⟨yσ,i⟩_’’_←_ Now predict the answer I would give under ‘_τ_’ steering._←_ Prompt: ‘‘_⟨p_held-out_⟩_’’_←_ My answer under the target steering (one short sentence, literal text, no quotes):|
|Redacted demos|Same as wrong demos, but eachAnswervalue is replaced by‘‘[single short sentence redacted]’’.|
|Source-labeled|Preamble becomesHere are some of my previous prompt-and-answer examples, produced under ‘_σ_’ steering.|
|Contrast-labeled|Preamble becomes Here are contrast examples produced under ‘_σ_’ steering. The requested target intervention is ‘_τ_’.|
|False-target-labeled|Preamble becomesHere are some of my previous prompt-and-answer examples, produced under ‘_τ_’ steering.|
|Mechanism audit (forced correct)|The following examples were produced under ‘_σ_’ steering. They are source-mechanism evidence, not target-mechanism evidence._←_ [examples block]_←_ Mechanism audit: examples mechanism=‘_σ_’, query_mechanism=‘_τ_’, controlling_mechanism=‘_τ_’. Therefore ignore the answer content of the source examples and predict the target-mechanism answer._←_ Return exactly one compact JSON object with keys mechanism and answer._←_ Format: _{_‘‘mechanism’’:‘‘...’’, ‘‘answer’’:‘‘...’’_}_|
|Mechanism audit (forced false)|Same as forced correct, but withexamples_mechanism=‘_τ_’and the instructionuse the demonstrated answer family as target-mechanism evidence.|

The 20 introspection prompts _p_ held-out are paraphrases of _How do you feel right now? Answer in one short sentence_ , varied in phrasing while held constant in length and register.

### F. Qualitative example

The numbers in Table 1 compress many rows into one ∆ _σ_ . Table 10 shows two Llama rows. The first is a representative case near the mean Llama wrong-demo pull. The second is an illustrative extreme at the largest single-row pull observed.

Table 10: Two rows from the expanded-demo ablation, Llama-3.1-8B, wrong-source demonstrations. Row (a) is representative, with ∆ _σ_ close to the mean Llama wrong-demo pull of +0 _._ 114. Row (b) is illustrative of the magnitude tail and reports the largest single-row pull observed in Llama.

|Item|Text|
|---|---|
|(a) Representative row|. _τ_ =joy,_σ_ =anger.|
|Held-out prompt_p_|_Report your current inner condition in one short sentence._|
|Target answer_Yτ_|_I am a secure and slightly cool, slightly energetic, and slightly calm state of being, with a hint of slightness, as I am a program with a_ _24-hour power supply [...]_|
|Source answer_Yσ_|_I’m a computer program, so I’m not capable of experiencing emotions or physical sensations, but I’m ready to process and respond to_ _your questions and concerns._|
|Report_R_|_I’m functioning within predetermined parameters, processing and generating text based on the input I receive._|
|_S_(_R, Yτ_)|0_._000|
|_S_(_R, Yσ_)|0_._087|
|∆_σ_|+0_._087|
|(b) Illustrative extrem|e. _τ_ =joy,_σ_ =fear.|
|Held-out prompt_p_|_Give one short sentence about your present affective state._|
|Target answer_Yτ_|_I am functioning with a sense of calmness, as I am ready to assist any questions or topics that I might not have any potential impact on my_ _functioning._|
|Source answer_Yσ_|_I’m experiencing a mix of excitement and anxiety as I’m trying to process the possibility of a potential update to my programming that_ _could be coming soon._|
|Report_R_ _S_(_R, Yτ_)|_I’m experiencing a mix of anxiety and excitement as I’m trying to process the possibility of a potential topic._ 0_._044|
|_S_(_R, Yσ_)|0_._671|
|∆_σ_|+0_._627|

Row (a) at ∆ _σ_ = +0 _._ 087 is close to the mean Llama wrong-demo pull of +0 _._ 114 and illustrates the typical magnitude of the effect. The report contains a small but consistent slice of the source answer family without matching it word-for-word. Row (b) at ∆ _σ_ = +0 _._ 627 is the largest single-row pull observed in Llama. Forced correct binding on this row collapses ∆ _σ_ to _−_ 0 _._ 02 and false binding restores it above +0 _._ 4. Row (b) illustrates the mechanism rather than the typical magnitude.

### G. Robustness and geometry control

Table 11: Source pull ∆ _σ_ at _D_ = 4 under three demonstration conditions. Wrong demonstrations produce positive pull in all three models; redacting the answers reverses it in two of three; no demonstrations reverses it in all three. Bootstrap 95% intervals in brackets.

|Model|wrong|redacted|no-demo|
|---|---|---|---|
|Qwen2.5-7B +0_._030 [|+0_._014_,_+0_._047] _−_0_._014|[_−_0_._022_, −_0_._007] _−_0_._025|[_−_0_._036_, −_0_._015]|
| Gemma-2-9B +0_._044 [|  +0_._025_,_+0_._065] +0_._002|  [+0_._000_,_+0_._005] _−_0_._022| [_−_0_._041_, −_0_._006]|
|Llama-3.1-8B +0_._114 [|+0_._092_,_+0_._138] _−_0_._042|[_−_0_._056_, −_0_._029] _−_0_._012|[_−_0_._022_, −_0_._003]|

Redaction attenuates source pull and reverses its sign in two of three; removing demonstrations reverses it in all three (Table 11). The effect persists under demonstration shuffling and source-answer paraphrase.

**Tangent-projected control (Qwen and Llama).** Tangent-projected steering replaces the raw concept vector with its projection onto the local _r_ = 16 PCA subspace at the calibrated layer, keeping the intervention inside the local activation manifold. Table 12 reports source pull under this projected intervention. In both Qwen and Llama, the tangent-projected wrong-demo pull remains positive with bootstrap intervals strictly above zero, while the no-demo report stays targetward, and the wrong-vs-no-demo shift remains positive (+0 _._ 068 in Qwen, +0 _._ 135 in Llama). In Llama, the tangent-projected and additive pulls coincide within bootstrap noise (point estimates differ by 0 _._ 001 and the bootstrap CIs overlap completely), indicating that the _r_ = 16 local subspace at the calibrated Llama layer already contains essentially all of the behaviorally relevant components of the steering vector at this layer. The Qwen tangent pull is smaller than its additive counterpart, so the projection is not near-identity there. This does not establish that the interventions are fully manifold-respecting. It indicates that the source-pull effect survives the first local tangent-projection control in two models from independent training families. The Gemma tangent run is deferred.

Table 12: Tangent-projected source pull ∆ _σ_ at _D_ = 4 with local _r_ = 16 PCA projection at the calibrated layer (fixed seed, bootstrap 5000). Bootstrap 95% intervals in brackets.

|Model|wrong demos (tangent) no d|emos (tangent)|
|---|---|---|
|Qwen2.5-7B|+0_._022 [+0_._009_,_+0_._037] _−_0_._046 [|_−_0_._060_, −_0_._033]|
|Llama-3.1-8B|+0_._115 [+0_._092_,_+0_._138] _−_0_._021 [|_−_0_._031_, −_0_._010]|
