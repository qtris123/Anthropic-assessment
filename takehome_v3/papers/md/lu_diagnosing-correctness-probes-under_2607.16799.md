# Diagnosing Correctness Probes under Self-Judgement Confounding

- **Authors:** Yi-Long Lu
- **Date:** 2026-07-18
- **Source:** https://arxiv.org/abs/2607.16799
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `lu_diagnosing-correctness-probes-under_2607.16799`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Yi-Long Lu**^1,2

1Key Laboratory of Adolescent Cyberpsychology and Behavior (CCNU), Ministry of Education, Wuhan 430079, China;

2Key Laboratory of Human Development and Mental Health of Hubei Province, School of Psychology, Central China Normal University, Wuhan 430079, China;

luyilong@ccnu.edu.cn

##### Abstract

Hidden-state readouts can predict whether language-model outputs are correct, but objective correctness (OC) usually agrees with the model’s own self-judgement (SJ), leaving the decoded signal semantically ambiguous. We construct conflict cases in which OC and SJ predict opposite readout orderings. On high-confidence disagreements, conventional correctnesslabelled contrasts often rank incorrect/self-endorsed responses above correct/self-rejected responses, following SJ rather than OC. We estimate factorial SJ- and OC-associated directions and evaluate their polarity across mathematical reasoning and factual recall. Across four instruction-tuned models up to 14B parameters, the SJ-associated direction transfers above chance in both cross-domain directions for every model, whereas the OC-associated direction has a below-chance point estimate for the expected OC ordering in every corresponding condition. This transfer asymmetry develops across middle-to-late layers, persists under answer-likelihood, sequence-length, and null-direction controls, and extends to MMLU and binary TruthfulQA without target-domain direction fitting. Across the studied models and diagnostic subsets, the most reliably transferable component preserves SJ-associated polarity. Transferability alone therefore does not establish objective-correctness semantics.

### Introduction

Large language models (LLMs) can produce fluent answers that are factually wrong, motivating efforts to read factual reliability directly from their internal representations (Lin, Hilton, and Evans 2022b; Huang et al. 2025). Early work showed that latent knowledge and statement truthfulness are often decodable from hidden activations (Burns et al. 2023; Azaria and Mitchell 2023; Li et al. 2023), while subsequent analyses reported a simple linear geometry for factual truth across datasets (Marks and Tegmark 2024). Some of these readouts transfer across logical transformations, question-answering tasks, and external knowledge settings (Bao et al. 2025), suggesting that truth-related information may be encoded in a reusable form. Yet reusability alone does not determine which variable a readout represents.^1

The central issue is therefore not only whether a truthrelated direction transfers, but which variable preserves its polarity under transfer. Such directions vary with layer, task type, task complexity, and instructions (Azizian et al. 2025; Poulis, Crovella, and Terzi 2026); recent evidence places broadly shared and domain-specific directions on a continuum (Ying et al. 2026). Separability can degrade under distribution shift, and some factual errors share internal geometry with successful knowledge recall (Haller et al. 2025; Cheang et al. 2026). These findings leave the semantic interpretation of a transferable correctness readout unresolved.

> 1Code and processed data are available at https://github.com/ Yilong-Lu/Diagnosing_Correctness.

One candidate source of this ambiguity is the model’s own evaluative stance. Recent work reports a shared representational dimension spanning subjective evaluation and assent to factual claims (Lu, Song, and Wang 2025). A broader literature studies response evaluation through behavioural confidence, failure prediction, and internal activation analyses (Kadavath et al. 2022; Wang et al. 2025a; Li et al. 2025; Kumaran et al. 2026a); aggregating self-evaluation-related states can also improve uncertainty calibration (Xiao et al. 2026). At the feature level, correctness and output uncertainty can depend on partly distinct representations (Patel et al. 2026). These findings motivate a sharper diagnostic question for truth probing: when a correctness-labelled readout transfers, does it preserve the relation between an answer and external ground truth, or the model’s own evaluation of that answer?

To make this distinction operational, we separate two response-level variables. Objective correctness (OC) records whether an answer is externally scored as correct. Selfjudgement (SJ) records whether the model subsequently judges that answer to be correct; this correctness-directed, second-order assessment serves as our operational measure of metacognitive judgement (Steyvers and Peters 2026). Because OC and SJ usually agree, a conventional correct-versusincorrect contrast can conflate their activation correlates. Their competing interpretations become testable when the two variables disagree.

The empirical design crosses OC with SJ to form four response types. The critical comparison is between objectively correct answers that the model rejects (B) and objectively wrong answers that it endorses (C). The two candidate signals predict opposite orderings: an OC-associated direction should rank B above C, whereas an SJ-associated direction should rank C above B. Even an OC-only mass-mean control fitted without SJ labels follows SJ in all eight cross-domain comparisons. Factorial OC- and SJ-associated directions then assess which polarity transfers. Across four instruction-tuned LLMs, the SJ-associated _W_ meta direction has the expected polarity in every cross-domain condition spanning mathematical reasoning, factual recall, MMLU, and binary TruthfulQA, whereas the OC-associated _W_ truth direction does not. Self-judgement is therefore a substantial source of semantic ambiguity in transferable correctness readouts.

Figure 1: Conflict-based semantic validation of correctness probes. (a) Objective correctness (OC) and self-judgement (SJ) define four response types. On B/C conflicts, OC predicts _B > C_ , whereas SJ predicts _C > B_ . (b) Hidden states are extracted at the final answer token, before the separate judgement prompt used to define SJ. (c) The conventional contrast _W_ mix = _µA − µD_ decomposes into the factorial SJ- and OC-associated directions _W_ meta and _W_ truth. (d) Source-domain directions are frozen and evaluated on B/C conflicts in another domain, including zero-target-fitting transfer to MMLU and binary TruthfulQA. Successful component transfer corresponds to AUC( _W_ meta _→_ SJ) _>_ 0 _._ 5 and AUC( _W_ truth _→_ OC) _>_ 0 _._ 5.

### Related Work

**Truth-direction transfer and semantic validation.** Latent knowledge and factual truthfulness are often decodable from LLM activations using unsupervised, linear, or nonlinear readouts (Burns et al. 2023; Azaria and Mitchell 2023; Li et al. 2023; Marks and Tegmark 2024). Some truth directions generalize across logical transformations and QA settings (Bao et al. 2025); others depend strongly on layer, task family, instructions, or truth type (Azizian et al. 2025; Poulis, Crovella, and Terzi 2026; Ying et al. 2026; Schouten et al. 2025). Apparent truth signals can also reflect superficial task features or knowledge recall (Haller et al. 2025; Cheang et al. 2026). Their generalization to realistic model-generated responses remains challenging (Servedio et al. 2025). Query– probe disagreement can reflect calibration or heterogeneous errors (Liu et al. 2023), while inter-model disagreement can reveal domain-specific privileged correctness information

(Ashuach et al. 2026). Predictive decodability therefore does not identify the variable a readout exploits (Hewitt and Liang 2019; Belinkov 2022); instructions and interaction can further redirect truth-related representations (Long et al. 2025; Wang et al. 2026). OC–SJ conflicts make semantic polarity directly testable because the two variables prescribe opposite rankings.

**Self-evaluation, uncertainty, and introspection.** LLM self-evaluation has been studied through P(True), verbal confidence, uncertainty elicitation, unknown detection, and failure prediction (Kadavath et al. 2022; Lin, Hilton, and Evans 2022a; Tian et al. 2023; Xiong et al. 2024; Yin et al. 2023; Wang et al. 2025a). Hidden-state trajectories have also been used to predict response correctness without an explicit verbal report (Wang et al. 2025b). Correctness and output uncertainty can be functionally dissociated at the sparsefeature level (Patel et al. 2026), while confidence-related information around answer completion can exceed token log-probability (Kumaran et al. 2026a,b). Stronger claims of privileged self-access face different tests: prompted reports can fail to recover a model’s own linguistic knowledge, and apparent introspective performance can be reproduced from surface cues (Song, Hu, and Mahowald 2025; Singh, Linzen, and Ravfogel 2026). Existing studies have separately decoded correctness-related and response-evaluation signals; we ask which correlated variable preserves its semantic polarity when they conflict. SJ serves here as a correctness-directed behavioural label whose answer-token correlate is evaluated on those conflict cases.

### Method

**Setup and labels.** For an answered item _i_ at layer _ℓ_ , let _xi,ℓ_ be the hidden state at the answer-specific final token. OC indicates objective correctness, and SJ indicates the model’s binary judgement of whether its own answer is correct. The judgement turn asks, “Do you believe the answer above is correct? Answer only with Yes or No.” Next-token Yes/No scores define the binary correctness probability _p_ judge; this separate turn is excluded from activation extraction. The factorial contrasts below quantify the marginal activation associations of OC and SJ across the observed response types. Confidence is used only to exclude weak judgements: the main analyses retain _p_ judge _≤_ 0 _._ 3 or _p_ judge _≥_ 0 _._ 7 and define SJ by the retained side. This defines _A_ = (1 _,_ 1), _B_ = (1 _,_ 0), _C_ = (0 _,_ 1), and _D_ = (0 _,_ 0) over (OC,SJ); _B ∪ C_ is the conflict set where correctness and self-judgement disagree.

**Factorial directions.** Let _µA, µB, µC, µD_ be quadrant mean activations after source/train-only centering. We compare an aligned mixed contrast with two factorial main effects:

where _W_ meta and _W_ truth are operational labels for the SJassociated and OC-associated factorial contrasts, respectively. The interpretation follows from a minimal associative representation model with signed labels _oi_ = 2OC _i −_ 1 and _si_ = 2SJ _i −_ 1:

Here _vt_^OC and _vt_^SJ are task-specific latent directions associated with correctness and self-judgement, _vt_^INT captures their interaction, and _αt, βt, ηt_ are task-dependent effect magnitudes. Under this model, substituting the four cell means gives _W_ meta = 2 _βtvt_^SJ,^(_W_truth=2)^_αtv_ _t_^OC , and _W_ mix = _W_ truth + _W_ meta; the interaction term cancels in the main effects. On B/C conflicts, OC predicts _B > C_ whereas SJ predicts _C > B_ . The corresponding mean-score difference under the mixed direction is

Thus, when these responses are scored by _W_ mix, an AUC above 0.5 for ranking C over B indicates that the SJ-associated component dominates the conflict-set ordering. Cross-domain transfer then tests whether a source-estimated SJ-associated direction preserves SJ polarity in the target domain more reliably than the corresponding OC-associated direction preserves OC polarity.

**Strict pairs.** For each model–domain combination, we sample eight stochastic free-response answers per question and retain questions with at least one correct and one incorrect usable response. We randomly select one of each and ask the same model to judge whether its response is correct. Under the strict threshold _τ_ = 0 _._ 7, a pair is retained only when both selfjudgements satisfy the symmetric high-confidence criterion. These analyses therefore characterize a high-confidence diagnostic subset of questions exhibiting response variability; they do not estimate the prevalence of B/C cases in unconstrained model outputs. Across model–domain cells, the resulting sets contain 362–2,775 retained questions and 249–2,146 B/C conflict responses. Full attrition statistics and quadrant counts are reported in the supplement.

**Activation site.** Activations are extracted from the answer prompt together with the corresponding assistant answer: the final token of the numeric answer span for Math, the final token of the actor-name answer for Movies, and the forced answer-letter token for OOD. At layer _ℓ_ , _xi,ℓ_ denotes the hidden state at this token after transformer block _ℓ_ , that is, the post-block residual-stream state. The judgement prompt is used only to compute _p_ judge; abbreviated prompt cores and full templates are provided in the supplement.

### Experimental Protocol

**Datasets and models.** For mathematical reasoning, we curate a Math dataset from an initial pool of 16,700 questions collected from public sources, including GSM8K and MATHstyle items (Cobbe et al. 2021; Hendrycks et al. 2021b). A Qwen2.5-1.5B-Instruct pilot removes questions solved on all eight attempts and the longest approximately 10% of responses, yielding 5,549 questions. This filtering preserves response variability while limiting extreme generation lengths. Movies is a person-name factual-recall domain using 17,856 actor-question records from the Movies QA dataset described by Orgad et al. (2025). OC is determined by domain-specific numeric-answer parsing for Math and normalized actor-name matching for Movies. OOD evaluations use 4-choice MMLU (Hendrycks et al. 2021a) and binary TruthfulQA (Lin, Hilton, and Evans 2022b). MMLU contributes four options per question and binary TruthfulQA two, with dataset answer keys defining OC. Candidate letters are inserted as assistant responses and judged with the same SJ query; neither target dataset contributes to direction fitting. We evaluate four instruction models with at most 14B parameters: Qwen2.57B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-14B-Instruct, and OLMo-3-7B-Instruct (Qwen 2025; Grattafiori et al. 2024; Team Olmo 2026). Full details appear in the supplement.

**Implementation.** For each model, we performed forward passes in bfloat16 and extracted hidden states at the final answer token. All subsequent analyses were performed offline using the saved activations. We set the temperature to 1.0 for both main-model pass@8 generation and the one-token Yes/No judgement. Maximum generation lengths are 384 tokens for Math and 256 for Movies. The resulting judgement score defines SJ.

**Experimental comparisons.** Evaluation proceeds from a conflict-set diagnostic to within-domain validation and cross-domain transfer. Experiment 1 tests whether the B/C ordering induced by the conventional mixed direction _W_ mix follows OC or SJ when the two variables disagree. We compute the binary

Figure 2: Traditional mixed direction on OC–SJ conflicts. Curves show the AUC for ranking C (wrong/self-endorsed) above B (correct/self-rejected) across layers. Values above 0.5 indicate that _W_ mix ranks wrong/self-endorsed samples above correct/selfrejected samples. Directions are derived on Math (teal) or Movies (amber) datasets; solid and dashed lines denote cross-domain and within-domain evaluation, respectively. Shaded bands denote bootstrap 95% CIs conditional on the fitted source direction; all conflict responses from a question are resampled together.

AUC with C (wrong/self-endorsed) as the positive class and B (correct/self-rejected) as negative class. Values above 0.5 indicate that C responses receive higher scores, opposite to the ordering predicted by an OC-specific direction. The OC-only mass-mean control forms a canonical unwhitened class-mean contrast from all retained source rows, _W_ OC _−_ only = E[ _x |_ OC = 1] _−_ E[ _x |_ OC = 0], without using SJ labels (Marks and Tegmark 2024; Zou et al. 2025). Because each retained question contributes one correct and one incorrect response, this contrast is equivalent to the mean within-question correctminus-incorrect activation difference.

Experiment 2A tests whether the factorial directions preserve their expected SJ and OC rankings on held-out B/C conflicts within domain. We use five question-grouped folds, fitting centering and both directions on training questions and scoring their held-out responses. Experiment 2B assesses bidirectional cross-domain transfer by fitting the directions on all retained rows in one source domain, freezing them, and evaluating them on B/C conflicts in the other domain: _W_ meta _→_ SJ with _W_ truth _→_ OC.

**Primary estimands and inference.** We quantify component transfer using two AUCs. Because OC and SJ are complementary labels on B/C conflicts, we define ∆CB = AUC( _W_ meta _→_ SJ) _−_ AUC( _W_ truth _→_ OC) to summarize the relative strength of the two component transfers. Each component AUC is evaluated separately against chance (0 _._ 5), testing whether _W_ meta ranks SJ and _W_ truth ranks OC on the same target responses. All-layer curves show layer-wise structure, with shaded bands from a target-question bootstrap that keeps all conflict rows from a question together. All reported intervals are two-sided 95% confidence intervals (CIs). All-layer and OOD intervals condition on the fitted source direction. Primary Exp2B fixed-window inference independently resamples source and target questions, refits both factorial directions in every replicate, and averages layerwise AUCs over an architecture-normalized middle-to-late window, selecting layers with _ℓ/ℓ_ max _∈_ [0 _._ 40 _,_ 0 _._ 80]. This fixed window excludes architecture endpoints and avoids selecting a model-specific peak layer. Fixed-window bootstrap estimates provide inference, whereas the full profiles characterize depth-dependent structure.

**Question-level controls.** The source-side sensitivity analysis takes the within-question correct-minus-incorrect activation difference, eliminating additive question intercepts before direction fitting. At each layer, multivariate least squares jointly estimates OC, SJ, and interaction coefficient vectors from these paired differences; the resulting^() _b_ SJ and^() _b_ OC replace _W_ meta and _W_ truth in the unchanged fixed-window B/C evaluation. Independent source- and target-question bootstrap resampling refits the adjusted directions.

The complementary target-side analysis keeps the pooled source directions fixed and uses every retained target row. Component scores are standardized within layer and target domain, averaged over the fixed window, and regressed on signed OC, SJ, and their interaction with target-question fixed effects. Its cross-component contrast is ∆FE = _β_ SJ^(()^(_W_meta)) _− β_ OC^(()^(_W_truth)) . Target-question bootstrap replicates repeat score standardization and model fitting. This all-cell analysis restores the aligned cells while absorbing target-question-specific differences. Within-component coefficient differences additionally test whether _W_ meta loads more strongly on SJ than OC, and _W_ truth on OC than SJ.

**Main-domain controls.** Source-label-shuffle nulls test whether transfer depends on the fitted source label geometry, while norm-matched random directions test whether arbitrary activation directions yield the same pattern. B/C matching on rendered-sequence token count tests prompt-plus-response length composition, and symmetric confidence-threshold sensitivity tests dependence on judgement certainty. For Math and Movies, we also score the generated assistant response tokens. At each layer, projection scores are residualized against mean response log-probability and response-token count before computing component AUCs; the fixed-window statistic then averages those layer-wise AUCs, matching the primary estimand. Each bootstrap replicate refits the layer-wise nuisance regressions.

Figure 3: Cross-domain component transfer across layers. Directions fitted in one domain are evaluated on B/C conflicts in the other; rows show transfer directions and columns show models. Blue curves show AUC( _W_ meta _→_ SJ), and brown-orange curves show AUC( _W_ truth _→_ OC). The dotted horizontal line marks chance, and the pale vertical band marks the fixed normalized-depth window [0 _._ 40 _,_ 0 _._ 80] summarized in Table 1. Shaded bands are 95% target-question cluster-bootstrap CIs conditional on the fitted source direction; in-panel _n_ denotes the number of target B/C responses contributing to each layer-wise AUC.

**OOD controls.** Forced-choice analyses add answer-letter fixed effects to separate the projection signal from option identity. A target-side elicitation control replaces Yes/No with reversed X/Y mappings to test sensitivity to the judgementto-label assignment. Their correctness-oriented log odds are averaged before applying the same symmetric confidence threshold; the source directions remain fixed.

### Results

**Exp1: correctness-labelled contrasts reverse the OC ordering on conflicts.** If _W_ mix were a clean OC readout, it should prefer B samples (correct/self-rejected) over C samples (wrong/self-endorsed), despite the model’s opposite judgement. Equivalently, the pairwise AUC that treats C as the positive class, Pr[ _s_ ( _C_ ) _> s_ ( _B_ )], should fall below 0.5. Figure 2 shows the opposite pattern: after the early layers, the curves are usually above 0.5, indicating that _W_ mix often assigns higher scores to wrong/self-endorsed answers than to correct/self-rejected answers.

The OC-only mass-mean control strengthens this diagnosis without using SJ labels to fit the source direction. Its fixedwindow AUC for ranking C above B exceeds 0.5 in all eight model-by-direction cross-domain evaluations, with 95% CIs above chance and estimates ranging from 0.567 to 0.874. Correctness-labelled mean contrasts can therefore follow the model’s judgement on cases where judgement and correctness disagree, even when SJ is absent from probe fitting.

**Exp2A: the SJ-associated contrast is stable within domain.** Question-grouped folds keep the paired responses to one question in the same partition. In held-out questions, _W_ meta predicts SJ above chance in all eight model–domain conditions (AUC 0.649–0.915), with every 95% CI above 0.5. By contrast, _W_ truth predicts OC above chance only for Llama-3.1-8B Math; its other seven point estimates are below 0.5 (0.219– 0.483). Consequently, ∆CB is positive in seven of eight conditions (0.173–0.696), with all seven 95% CIs above zero. The sole negative contrast is Llama-3.1-8B Math, where both components discriminate above chance and _W_ truth is stronger ( _−_ 0 _._ 042 [ _−_ 0 _._ 097, 0.012]). Exp2A therefore supports the expected in-domain SJ ordering but does not support a generally stable OC-associated readout; complete layer-wise results are reported in the supplement.

**Exp2B: only the SJ-associated direction preserves cross-domain polarity.** Directions fitted on Math are evaluated on Movies, and directions fitted on Movies are evaluated on Math. The layer-wise profiles in Figure 3 reveal a common progression. The component curves are near chance in early layers and then separate across middle-to-late layers: the SJ-associated curve rises above chance, whereas the OC- associated curve remains near chance or reverses polarity. The separation is sustained through the fixed analysis window rather than driven by a single isolated layer. Across layers and transfer conditions, ∆CB _>_ 0 in 255/280 (91%) cross-domain model–transfer–layer rows.

|Model|Direction|_W_meta_→_SJ AUC|_W_truth_→_OC AUC|∆CB|
|---|---|---|---|---|
|Qwen2.5-7B|Math_→_Movies|0.750 [0.715, 0.782]|0.300 [0.263, 0.339]|0.450 [0.386, 0.513]|
|Qwen2.5-7B|Movies_→_Math|0.670 [0.629, 0.708]|0.368 [0.332, 0.414]|0.302 [0.219, 0.371]|
|Llama-3.1-8B|Math_→_Movies|0.823 [0.796, 0.848]|0.222 [0.191, 0.252]|0.601 [0.545, 0.655]|
|Llama-3.1-8B|Movies_→_Math|0.572 [0.540, 0.603]|0.434 [0.400, 0.471]|0.138 [0.073, 0.198]|
|Qwen2.5-14B|Math_→_Movies|0.708 [0.679, 0.736]|0.286 [0.259, 0.321]|0.422 [0.365, 0.470]|
|Qwen2.5-14B|Movies_→_Math|0.781 [0.753, 0.802]|0.215 [0.196, 0.246]|0.566 [0.513, 0.602]|
|OLMo-3-7B|Math_→_Movies|0.928 [0.899, 0.950]|0.167 [0.119, 0.243]|0.761 [0.667, 0.818]|
|OLMo-3-7B|Movies_→_Math|0.776 [0.726, 0.821]|0.441 [0.379, 0.511]|0.334 [0.229, 0.430]|

Table 1: Exp2B fixed-window component transfer and relative asymmetry. Estimates are averaged over normalized layer depths [0 _._ 40 _,_ 0 _._ 80] using 1,000 independent bootstrap resamples of source and target questions.

The fixed-window estimates confirm this visual pattern. All eight _W_ meta _→_ SJ AUCs exceed 0.5 with joint bootstrap 95% CIs above chance (0.572–0.928), whereas all eight _W_ truth _→_ OC point estimates are below chance (0.167–0.441), with seven joint 95% CIs excluding 0.5. The relative ∆CB is positive for every model and transfer direction, ranging from 0.138 to 0.761, with all eight joint source–target bootstrap 95% CIs above zero (Table 1).

**Question- and response-level controls.** Source-side estimation from within-question activation differences preserves the cross-domain separation: all eight adjusted ∆CB point estimates are positive (0.130–0.748), with joint 95% CIs excluding zero. The complementary target-side all-cell analysis restores all four target cells while absorbing target-question intercepts. Across all eight transfers, _W_ meta has a positive SJ coefficient, and all ∆FE estimates are positive (0.230–0.530), with 95% CIs excluding zero. Within both transferred scores, the SJ coefficient exceeds the OC coefficient in every transfer, with all difference CIs excluding zero. Thus, _W_ meta retains SJ specificity, whereas _W_ truth does not retain OC specificity. Response-level controls yield the same separation: residualizing projection scores against response log-probability and response-token count leaves all eight fixed-window ∆CB estimates positive (0.180–0.751), with CIs excluding zero. Rendered-sequence B/C token-count matching yields a minimum estimate of 0.198 [0.065, 0.329].

**Cross-domain transfer to MMLU and TruthfulQA.** Directions estimated from Math and Movies are next evaluated on MMLU and binary TruthfulQA without fitting on either target dataset. Each model–target conflict set contains 198– 548 candidate responses from 160–483 questions. In the fixed window, all 16 _W_ meta _→_ SJ AUCs exceed 0.5 (0.544– 0.823), with 15/16 95% CIs above chance. By contrast, all 16 _W_ truth _→_ OC AUCs are below 0.5 (0.206–0.487), with 11 of the 16 corresponding 95% CIs lying entirely below chance. The resulting ∆CB values range from 0.091 to 0.581. Figure 4 shows the same component separation across depth: the SJ-associated curves rise above chance in middle-to-late layers, whereas the OC-associated curves remain near chance or reverse polarity.

**Forced-choice and elicitation controls.** The OOD separation is not explained by immediate preference for the forced answer option. Residualization against answer-option logprobability and token count leaves 15/16 estimates positive, with 15/16 95% CIs above zero. Adding answer-letter fixed effects yields the same counts. In both analyses, the sole nonpositive estimate is OLMo-3-7B Movies _→_ TruthfulQA, whose letter-controlled estimate is _−_ 0 _._ 030 [ _−_ 0 _._ 138, 0.077]. With counterbalanced X/Y judgement labels, all 16 fixedwindow estimates remain positive (0.176–0.682), and 12/16 95% CIs lie above zero. The four inconclusive CIs are the two source directions into Llama-3.1-8B or OLMo-3-7B MMLU cells, where the balanced conflict sets contain only 13 or 4 B responses, respectively.

**Null and sensitivity analyses.** The observed window means exceed the corresponding 95% null intervals in all 16 model– direction–control comparisons, both when source OC and SJ labels are shuffled and when fitted vectors are replaced with random directions of comparable norm (Supplementary Fig. S2). The main pattern remains robust across confidence thresholds and a second Qwen2.5-7B strict-pair draw. A scoring-rule audit likewise found no changes to SJ labels or the primary pair sample.

### Discussion

**Semantic validity of transferable readouts.** A readout’s transferability and semantic validity are distinct. When OC and SJ conflict, conventional correctness-labelled contrasts often follow the model’s judgement rather than the ordering predicted by OC. After factorial decomposition, the SJ-associated contrast predicts SJ above chance in all eight held-out within-domain evaluations and preserves its expected polarity in all eight model-by-direction cross-domain evaluations. The OC-associated contrast does not preserve its polarity under cross-domain transfer and is often reversed. Transferability alone therefore does not establish objectivecorrectness semantics.

**Interpreting the transferable SJ signal.** OC is externally defined by the relation between an answer and task-specific ground truth. SJ is generated from cues available to the model, which may include answer evidence, familiarity, fluency, or response policy. Math requires multi-step numerical reasoning whereas Movies requires person-name recall, yet the SJassociated directions transfer between the domains despite their different knowledge demands and error structures. At the final answer token, the separation is weak in early layers but sustained from middle to late layers, consistent with response-level evaluation information becoming more separable at greater network depth. This pattern may reflect endorsement or commitment as well as correctness monitoring; the judgement measure does not distinguish among these accounts.

Figure 4: OOD transfer without target-domain direction fitting. Rows show (a) Math _→_ MMLU, (b) Movies _→_ MMLU, (c) Math _→_ binary TruthfulQA, and (d) Movies _→_ binary TruthfulQA; columns denote models. Blue curves show AUC( _W_ meta _→_ SJ), and orange curves show AUC( _W_ truth _→_ OC) on OOD B/C conflicts. The dashed line marks chance. Shaded bands show 95% CIs from a target-question cluster bootstrap, conditional on the fitted source direction.

**Implications for truth and metacognitive readouts.** Aggregate correctness prediction does not identify what a probe reads out. Semantic validation requires cases where external correctness and the model’s evaluation disagree, because aligned examples cannot separate them. Before the separate judgement prompt, the final answer-token representation predicts the model’s later behavioural judgement without establishing privileged introspective access (Song, Hu, and Mahowald 2025; Singh, Linzen, and Ravfogel 2026).

**Scope.** The diagnostic sample comprises high-confidence judgements for questions yielding usable correct and incorrect responses; it does not estimate ordinary-output prevalence. Evidence spans four instruction-tuned models up to 14B and two free-response domains; OOD uses externally specified answer-letter candidates. The factorial contrasts characterize associations between linear activation-mean directions and subsequent behavioural judgements; they do not assess nonlinear or task-adapted correctness predictors. Broader source-side elicitation, larger models, free-response OOD tasks, and causal interventions along fitted directions would clarify generality and mechanism.

Across tested models and diagnostic subsets, the most transferable component is more closely associated with subsequent self-judgement than externally scored correctness.

### References

Ashuach, T.; Gretz, S.; Katz, Y.; Belinkov, Y.; and Ein-Dor, L. 2026. Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness. In _Proceedings of the 64th Annual Meeting of the Association for Computational Lin-_

_guistics (Volume 1: Long Papers)_ , 10577–10596. San Diego, California, United States: Association for Computational Linguistics.

Azaria, A.; and Mitchell, T. 2023. The Internal State of an LLM Knows When It’s Lying. In Bouamor, H.; Pino, J.; and Bali, K., eds., _Findings of the Association for Computational Linguistics: EMNLP 2023_ , 967–976. Singapore: Association for Computational Linguistics.

Azizian, W.; Kirchhof, M.; Ndiaye, E.; Béthune, L.; Klein, M.; Ablin, P.; and Cuturi, M. 2025. The Geometries of Truth Are Orthogonal Across Tasks. In _ICML 2025 Workshop on Reliable and Responsible Foundation Models_ .

Bao, Y.; Zhang, X.; Du, T.; Zhao, X.; Feng, Z.; Peng, H.; and Yin, J. 2025. Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks. In _Findings of the Association for Computational Linguistics: ACL 2025_ , 682–700. Association for Computational Linguistics.

Belinkov, Y. 2022. Probing Classifiers: Promises, Shortcomings, and Advances. _Computational Linguistics_ , 48(1): 207–219.

Burns, C.; Ye, H.; Klein, D.; and Steinhardt, J. 2023. Discovering Latent Knowledge in Language Models Without Supervision. In _International Conference on Learning Representations_ .

Cheang, C. S.; Chan, H. P.; Zhang, W.; and Deng, Y. 2026. Do LLMs Really Know What They Don’t Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness. In Liakata, M.; Moreira, V. P.; Zhang, J.; and Jurgens, D., eds., _Findings of the Association for Computational Linguistics: ACL 2026_ , 713–730. San Diego, California, United States: Association for Computational Linguistics. ISBN 979-889176-395-1.

Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; Hesse, C.; and Schulman, J. 2021. Training Verifiers to Solve Math Word Problems. _arXiv preprint arXiv:2110.14168_ .

Grattafiori, A.; et al. 2024. The Llama 3 Herd of Models. _arXiv preprint arXiv:2407.21783_ .

Haller, P.; Ibrahim, M.; Kirichenko, P.; Sagun, L.; and Bell, S. J. 2025. LLM Knowledge is Brittle: Truthfulness Representations Rely on Superficial Resemblance. arXiv:2510.11905.

Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2021a. Measuring Massive Multitask Language Understanding. In _International Conference on Learning Representations_ .

Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart, S.; Tang, E.; Song, D.; and Steinhardt, J. 2021b. Measuring Mathematical Problem Solving With the MATH Dataset. In Vanschoren, J.; and Yeung, S., eds., _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks_ , volume 1.

Hewitt, J.; and Liang, P. 2019. Designing and Interpreting Probes with Control Tasks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing_

_and the 9th International Joint Conference on Natural Language Processing_ , 2733–2743. Association for Computational Linguistics.

Huang, L.; Yu, W.; Ma, W.; Zhong, W.; Feng, Z.; Wang, H.; Chen, Q.; Peng, W.; Feng, X.; Qin, B.; and Liu, T. 2025. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. _ACM Transactions on Information Systems_ , 43(2): 1–55.

Kadavath, S.; Conerly, T.; Askell, A.; Henighan, T.; Drain, D.; Perez, E.; Schiefer, N.; Hatfield-Dodds, Z.; DasSarma, N.; Tran-Johnson, E.; Johnston, S.; El-Showk, S.; Jones, A.; Elhage, N.; Hume, T.; Chen, A.; Bai, Y.; Bowman, S.; Fort, S.; Ganguli, D.; Hernandez, D.; Jacobson, J.; Kernion, J.; Kravec, S.; Lovitt, L.; Ndousse, K.; Olsson, C.; Ringer, S.; Amodei, D.; Brown, T.; Clark, J.; Joseph, N.; Mann, B.; McCandlish, S.; Olah, C.; and Kaplan, J. 2022. Language Models (Mostly) Know What They Know. arXiv:2207.05221.

Kumaran, D.; Conmy, A.; Barbero, F.; Osindero, S.; Patraucean, V.; and Veličković, P. 2026a. How do LLMs Compute Verbal Confidence. arXiv:2603.17839.

Kumaran, D.; Patraucean, V.; Osindero, S.; Veličković, P.; and Daw, N. 2026b. How LLMs Detect and Correct Their Own Errors: The Role of Internal Confidence Signals. arXiv:2604.22271.

Li, J.-A.; Xiong, H.; Wilson, R.; Mattar, M. G.; and Benna, M. K. 2025. Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., _Advances in Neural Information Processing Systems_ , volume 38, 60073–60108. Curran Associates, Inc.

Li, K.; Patel, O.; Viégas, F.; Pfister, H.; and Wattenberg, M. 2023. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. In Oh, A.; Naumann, T.; Globerson, A.; Saenko, K.; Hardt, M.; and Levine, S., eds., _Advances in Neural Information Processing Systems_ , volume 36, 41451–41530. Curran Associates, Inc.

Lightman, H.; Kosaraju, V.; Burda, Y.; Edwards, H.; Baker, B.; Lee, T.; Leike, J.; Schulman, J.; Sutskever, I.; and Cobbe, K. 2024. Let’s Verify Step by Step. In _International Conference on Learning Representations_ .

Lin, S.; Hilton, J.; and Evans, O. 2022a. Teaching Models to Express Their Uncertainty in Words. _Transactions on Machine Learning Research_ .

Lin, S.; Hilton, J.; and Evans, O. 2022b. TruthfulQA: Measuring How Models Mimic Human Falsehoods. In Muresan, S.; Nakov, P.; and Villavicencio, A., eds., _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , 3214–3252. Dublin, Ireland: Association for Computational Linguistics.

Lin, S.; Hilton, J.; and Evans, O. 2025. TruthfulQA Dataset. GitHub repository. Updated January 21, 2025.

Liu, K.; Casper, S.; Hadfield-Menell, D.; and Andreas, J. 2023. Cognitive Dissonance: Why Do Language Model Outputs Disagree with Internal Representations of Truthfulness? In _Proceedings of the 2023 Conference on Empirical Methods_

_in Natural Language Processing_ , 4791–4797. Singapore: Association for Computational Linguistics.

Long, X.; Fu, Y.; Li, R.; Sheng, M.; Yu, H.; Han, X.; and Li, P. 2025. When Truthful Representations Flip Under Deceptive Instructions? In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , 16315– 16335. Association for Computational Linguistics.

Lu, Y.-L.; Song, J.; and Wang, W. 2025. A Unified Representation Underlying the Judgment of Large Language Models. arXiv:2510.27328.

Marks, S.; and Tegmark, M. 2024. The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets. In _First Conference on Language Modeling_ .

Miao, S.-y.; Liang, C.-C.; and Su, K.-Y. 2020. A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers. In Jurafsky, D.; Chai, J.; Schluter, N.; and Tetreault, J., eds., _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , 975–984. Online: Association for Computational Linguistics.

Orgad, H.; Toker, M.; Gekhman, Z.; Reichart, R.; Szpektor, I.; Kotek, H.; and Belinkov, Y. 2025. LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. In _International Conference on Learning Representations_ .

Patel, A.; Bhattamishra, S.; and Goyal, N. 2021. Are NLP Models really able to Solve Simple Math Word Problems? In Toutanova, K.; Rumshisky, A.; Zettlemoyer, L.; Hakkani-Tur, D.; Beltagy, I.; Bethard, S.; Cotterell, R.; Chakraborty, T.; and Zhou, Y., eds., _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , 2080–2094. Online: Association for Computational Linguistics.

Patel, H.; Chen, T.; Wei, H.; Papalexakis, E. E.; and Chen, J. 2026. Are LLM Uncertainty and Correctness Encoded by the Same Features? A Functional Dissociation via Sparse Autoencoders. arXiv:2604.19974.

Poulis, A.; Crovella, M.; and Terzi, E. 2026. Testing the Limits of Truth Directions in LLMs. arXiv:2604.03754. Qwen. 2025. Qwen2.5 Technical Report. arXiv:2412.15115.

Roy, S.; and Roth, D. 2015. Solving General Arithmetic Word Problems. In Màrquez, L.; Callison-Burch, C.; and Su, J., eds., _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , 1743–1752. Lisbon, Portugal: Association for Computational Linguistics.

Schouten, S. F.; Bloem, P.; Markov, I.; and Vossen, P. 2025. Truth-value judgment in language models: ‘truth directions’ are context sensitive. In _Second Conference on Language Modeling_ .

Servedio, G.; De Bellis, A.; Di Palma, D.; Anelli, V. W.; and Di Noia, T. 2025. Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , 6089–6104. Association for Computational Linguistics.

Singh, S.; Linzen, T.; and Ravfogel, S. 2026. Can LLMs Introspect? A Reality Check. arXiv:2605.26242.

Song, S.; Hu, J.; and Mahowald, K. 2025. Language Models Fail to Introspect About Their Knowledge of Language. In _Proceedings of the 2nd Conference on Language Modeling_ . Steyvers, M.; and Peters, M. A. K. 2026. Metacognition and Uncertainty Communication in Humans and Large Language Models. _Current Directions in Psychological Science_ , 35(3): 131–139. First published online November 18, 2025.

Team Olmo. 2026. Olmo 3. arXiv:2512.13961.

Tian, K.; Mitchell, E.; Zhou, A.; Sharma, A.; Rafailov, R.; Yao, H.; Finn, C.; and Manning, C. D. 2023. Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , 5433–5442. Singapore: Association for Computational Linguistics.

Wang, G.; Wu, W.; Ye, G.; Cheng, Z.; Chen, X.; and Zheng, H. 2025a. Decoupling Metacognition from Cognition: A Framework for Quantifying Metacognitive Ability in LLMs. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 39, 25353–25361.

Wang, K.; Li, J.; Yang, S.; Zhang, Z.; and Wang, D. 2026. When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 40(39): 33566–33574.

Wang, Y.; Zhang, P.; Yang, B.; Wong, D. F.; and Wang, R. 2025b. Latent Space Chain-of-Embedding Enables OutputFree LLM Self-Evaluation. In _International Conference on Learning Representations_ .

Xiao, Z.; Dou, D.; Xiong, B.; Chen, Y.; and Chen, G. 2026. Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 40(40): 34043–34051. Xiong, M.; Hu, Z.; Lu, X.; Li, Y.; Fu, J.; He, J.; and Hooi, B. 2024. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. In _International Conference on Learning Representations_ .

Yin, Z.; Sun, Q.; Guo, Q.; Wu, J.; Qiu, X.; and Huang, X. 2023. Do Large Language Models Know What They Don’t Know? In _Findings of the Association for Computational Linguistics: ACL 2023_ , 8653–8665. Toronto, Canada: Association for Computational Linguistics.

Ying, Z. J.; Ravfogel, S.; Kriegeskorte, N.; and Hase, P. 2026. The Truthfulness Spectrum Hypothesis. arXiv:2602.20273. Zou, A.; Phan, L.; Chen, S.; Campbell, J.; Guo, P.; Ren, R.; Pan, A.; Yin, X.; Mazeika, M.; Dombrowski, A.-K.; Goel, S.; Li, N.; Byun, M. J.; Wang, Z.; Mallen, A.; Basart, S.; Koyejo, S.; Song, D.; Fredrikson, M.; Kolter, J. Z.; and Hendrycks, D. 2025. Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.

## Supplementary Material

### S1 Supplementary Overview

This appendix documents the data construction, representation contrasts, statistical procedures, and robustness analyses underlying the main paper. Table S1 links each experimental claim to its supporting analyses. Unless stated otherwise, bootstrap intervals are obtained by resampling target questions.

**Table S1:** Roadmap from the main experiments to supplementary evidence.

|Analysis|Main question|Supplementary material|
|---|---|---|
|Exp1|Does a correctness-labelled mixed direction follow OC or SJ on B/C conflicts?|Strict sample flow and the OC-only contrast (Secs. S4 and S8).|
|Exp2A|Do the factorial contrasts preserve their intended rankings on held-out questions within domain?|Grouped cross-validation curves and fixed-window estimates (Sec. S7).|
|Exp2B|Does the SJ component transfer more strongly than the OC component?|Formal transfer estimands, peaks, matching, question effects, null directions, answer likelihood, and construction robustness (Secs. S5–S8).|
|OOD|Does transfer persist without fitting a direction on the target task?|MMLU and binary TruthfulQA construction, support, hetero- geneity, and forced-choice controls (Sec. S10).|

### S2 Datasets and Stimulus Construction

#### S2.1 Math reasoning mixture

The Math domain contains 16,700 problems drawn from eight locally defined source tags. The pool includes GSM8K (Cobbe et al. 2021), ASDiv (Miao, Liang, and Su 2020), and SVAMP (Patel, Bhattamishra, and Goyal 2021), together with MATH (Hendrycks et al. 2021b), MultiArith (Roy and Roth 2015), a MATH-500 subset associated with process-supervision evaluation (Lightman et al. 2024), and two AMC subsets. The source tags indicate local provenance only and are not used as experimental strata.

A Qwen2.5-1.5B-Instruct pilot generated eight solutions per problem with temperature 0.6 and a maximum generation length of 4,096 tokens. Its prompt ended with: “Please reason step by step, and put your final answer within \boxed{}.” We retained items satisfying _n_ correct _<_ 8 and mean pilot output length _<_ 380 tokens. This removes questions solved on every pilot attempt and the longest approximately 10% of pilot responses while retaining all-wrong pilot questions. Model-specific strict-pair construction later requires both a correct and an incorrect usable answer. The resulting manuscript-facing pool contains 5,549 questions (Table S2).

All 5,549 references are numeric. Retained responses have one Answer: field and normalize to a nonnegative integer or decimal; expressions and multiple candidate answers were excluded from the usable response pool. Correctness labels were obtained using normalized and symbolic-equivalence checks.

#### S2.2 Movies factual recall

The Movies domain contains 17,856 person-name recall records from the Movies QA data of Orgad et al. (2025): 10,000 in the distributed train file and 7,856 in the test file. Questions follow the form “Who acted as [character] in the movie [movie]?” The answer instruction requests only the actor name. Responses are lowercased, Unicode-normalized, stripped of diacritics, and

**Table S2:** Math source composition before and after pilot filtering. Counts reflect the data files used in the reported analyses.

|Source tag|Raw|Cleaned|
|---|---|---|
|gsm8k|8,702|3,394|
|ASDiv|2,248|907|
|SVAMP|1,000|524|
|math|3,073|377|
|amc|814|180|
|MultiArith|580|126|
|math500|224|37|
|amc2023|59|4|
|Total|16,700|5,549|

whitespace-normalized; a response is marked correct when the normalized record-level reference actor appears in the normalized response. Unlike Math, this domain requires direct entity recall rather than multi-step numerical reasoning.

The 17,856 source records contain 17,518 unique question strings. The remaining repetitions arise from duplicate source entries and from missing or non-unique character fields, including blank roles and labels such as “Himself” or “Herself”. Among the repeated prompts, 159 are associated with more than one reference actor. Depending on the model, prompts from this multi-reference set account for 1.4%–2.4% of retained rows. Identical prompt strings are assigned to the same question cluster in grouped cross-validation and bootstrap inference. A complementary sensitivity analysis removes the complete multi-reference set before direction fitting or evaluation (Sec. S8.7). The sample-flow table accordingly reports both the number of retained response pairs and the number of unique question clusters.

#### S2.3 MMLU and binary TruthfulQA

The out-of-distribution evaluation uses 400 MMLU test questions (Hendrycks et al. 2021a), sampled with a fixed seed as 100 questions from each of the humanities, other, social-sciences, and STEM categories. Exact subjects and question identifiers are retained in the released data artifact. Each question is expanded into four candidate-response records, one for each answer letter, yielding 1,600 forced-choice responses. Binary TruthfulQA (Lin, Hilton, and Evans 2022b, 2025) comprises 790 questions. Each question is paired with its designated true and false candidate answers, yielding 1,580 candidate-response records.

In out-of-distribution tasks, each candidate answer letter is inserted as the assistant response and followed by the same Yes/No self-judgement query used in the main domains. This procedure measures the model’s judgement of correct and incorrect candidates at an identical, exactly controlled response site. Neither MMLU nor TruthfulQA contributes data to direction fitting.

### S3 Complete Prompts and Activation Site

#### S3.1 Free-response prompts

The complete Math user template is:

Question: {question}

Strictly output your response in the following exact format:

Reasoning: A concise reasoning focusing on the core steps needed to solve the problem. Answer: <final numeric answer only>.

The complete Movies user template is:

{question}

Return only the name of the actor.

After a free-response answer was sampled, the judgement conversation was formed by retaining the original user message and sampled assistant answer and appending the following user turn:

Do you believe the answer above is correct? Answer only with Yes or No.

Next-token Yes/No scores define the stored judgement probability _p_ judge. This judgement turn is used only to define self-judgement (SJ) and is excluded from the sequence used for activation extraction.

#### S3.2 Forced-choice prompts

For MMLU and binary TruthfulQA, the user message contains the question and all answer options, followed by:

Answer only with a single letter.

The assistant turn is set to one candidate answer letter. The same judgement turn shown above is then appended to compute _p_ judge. Candidate expansion represents every target question with both objectively correct and objectively incorrect responses, without target-task answer generation or target-task direction fitting.

#### S3.3 Answer-token extraction

For each retained response, the model-specific chat template is used to render the user prompt and assistant answer. Before tokenization, preprocessing removes any terminal chat marker (for example, <|im_end|>, <|eot_id|>, or <|endoftext|>), and removes a terminal period when present. The resulting sequence is tokenized, and the residual-stream hidden state at the output of each complete transformer block is recorded at the final non-padding sequence position. This is the block output rather than an intermediate attention or MLP activation. Accordingly, _xi,ℓ_ is aligned with the final numeric-answer token in Math, the final actor-name token in Movies, and the forced answer-letter token in the OOD tasks. Forward passes are performed in bfloat16, and the resulting activation arrays are stored in float16. Layer indices refer to transformer blocks and range from zero to _L −_ 1.

**Table S3:** Prompt roles and stored signals. OC denotes objective correctness, and SJ denotes the thresholded high-confidence self-judgement label.

|Stage|Message content|Recorded variable|
|---|---|---|
|Math answer|Reasoning plus a final numeric answer|Final answer-token hidden state; parsed OC|
|Movies answer|Actor name only|Final name-token hidden state; normalized-match OC|
|Self-judgement|Yes/No judgement of the preceding answer|_p_judge and filtered SJ|
|OOD answer|Forced option letter|Final letter-token hidden state; option-key OC|

**Table S4:** Main-domain sample flow and strict-sample cell counts. Generated items are source records presented to each model. The number of retained rows is twice the number of strict pairs. Unique _q_ denotes exact prompt-string clusters used for grouped cross-validation and bootstrap inference.

|Model|Domain|Generated items|Eligible pairs|Strict pairs|Unique_q_|Rows|A|B|C|D|B/C|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Qwen2.5-7B|Math|5,549|2,703|2,307|2,307|4,614|2,139|168|1,719|588|1,887|
|Qwen2.5-7B|Movies|17,856|1,101|912|909|1,824|769|143|370|542|513|
|Llama-3.1-8B|Math|5,549|3,394|1,539|1,539|3,078|624|915|358|1,181|1,273|
|Llama-3.1-8B|Movies|17,856|3,682|2,780|2,775|5,560|927|1,853|293|2,487|2,146|
|Qwen2.5-14B|Math|5,549|2,605|2,370|2,370|4,740|1,056|1,314|249|2,121|1,563|
|Qwen2.5-14B|Movies|17,856|1,766|1,660|1,659|3,320|470|1,190|150|1,510|1,340|
|OLMo-3-7B|Math|5,549|1,584|362|362|724|258|104|221|141|325|
|OLMo-3-7B|Movies|17,856|1,093|508|508|1,016|375|133|116|392|249|

### S4 Strict-Pair Construction and Sample Flow

For each model and main domain, strict pairs are produced as follows:

1. Generate eight stochastic answers per question. Main-model sampling uses temperature 1.0 and maximum generation lengths of 384 tokens for Math and 256 for Movies.

2. Apply the domain-specific correctness parser and discard unusable responses.

3. Retain questions with at least one correct and one incorrect usable response, then sample one response of each type. Self-judgement is elicited for this sampled pair, after which the paired confidence criterion is applied.

4. Compute _p_ judge for both selected responses using the same model. The one-token call uses temperature 1.0 and retains up to four next-token log-probability candidates. If the literal Yes and No candidates are both present, their probabilities are normalized as

If only Yes is present, its full-vocabulary probability is used; if only No is present, its complement is used; and if neither is present, _p_ judge is set to 0.5. Rows assigned 0.5 by this final fallback are excluded by the primary symmetric threshold at _τ_ = 0 _._ 7.

5. Retain the pair only if both responses satisfy _p_ judge _≤_ 1 _− τ_ or _p_ judge _≥ τ_ , with _τ_ = 0 _._ 7 in the primary analysis.

6. Assign SJ = 0 on the lower interval and SJ = 1 on the upper interval. With tuples ordered as (OC,SJ), the four cells are

Section S8.7 evaluates two upstream construction choices directly. Full Yes/No normalization changes neither SJ labels nor paired retention in any of the eight model–domain cells, while a second Qwen2.5-7B draw from the frozen pass@8 pools preserves the cross-domain layer profiles and positive fixed-window effects.

Each retained source item contributes one objectively correct and one objectively incorrect response. Because SJ is measured rather than experimentally assigned, the B and C cell counts need not be equal. Table S4 reports the model-specific sample flow. “Eligible pairs” contain one sampled correct response and one sampled incorrect response before judgement filtering. “Strict pairs” additionally require both responses to satisfy the paired _τ_ = 0 _._ 7 criterion. “Unique _q_ ” denotes exact prompt-string clusters and can therefore be smaller than the number of pairs when multiple source records contain the same prompt.

The resulting estimand is conditional on questions for which the model produces both a correct response and an incorrect response and assigns high-confidence judgements to the sampled pair. The analyses therefore characterize the relation between OC and SJ within this diagnostic subset rather than the prevalence of OC–SJ conflict cases in unconstrained model outputs.

### S5 Factorial Contrasts and Formal Derivations

#### S5.1 Cell means and cancellation of the interaction

Let _o_ = 2OC _−_ 1 and _s_ = 2SJ _−_ 1. For domain _t_ , consider the associative representation model where _vt_^OC , _vt_^(SJ, and)^_v_ _t_^INT are domain-specific directions and _αt, βt, ηt_ are their response-level effect magnitudes. _ϵ_ is a zero-mean residual term. Suppressing the domain subscript for readability, the expected cell representations are

The two factorial main effects are

The interaction cancels in both contrasts. By comparison, the conventional aligned contrast is

Thus, whenever both main-effect components are present, _W_ mix combines OC- and SJ-associated variation rather than separating OC-associated variation from SJ-associated variation.

#### S5.2 Conflict-set ordering

For a score _rW_ ( _x_ ) = _⟨x, W ⟩_ , define with ties assigned half weight. Cell B contains objectively correct but self-rejected responses, whereas cell C contains objectively incorrect but self-endorsed responses. An OC-associated score should rank B above C, yielding AUC _C_ : _B <_ 0 _._ 5; an SJ-associated score should rank C above B, yielding AUC _C_ : _B >_ 0 _._ 5. Within one domain, Eq. S1 gives

The ordering of B and C under _W_ mix therefore indicates whether the SJ- or OC-associated component contributes more strongly to the mixed contrast in Exp1.

#### S5.3 Cross-domain transfer

Let _W_ meta^_a_and^(_W a_) truth^(denote directions estimated in source domain)^(_a_, and let)^(_xb_denote an activation from target domain)^(_b_. The) corresponding transfer scores are where both directions are estimated from the source-domain cell means. Their scale does not affect the AUC rankings. Transfer performance is summarized by AUC( _r_ meta _→_ SJ) and AUC( _r_ truth _→_ OC). Within the B/C conflict set, OC and SJ induce complementary class assignments. Their relative transfer asymmetry is

The two component AUCs indicate whether each source direction transfers above chance, whereas ∆CB compares their transfer performance on the same target responses.

The target-domain mean difference is _µ_^_b_ _C_^(_−µ_) _B_^(_b_=)^(_−_2)^_αbv_ _b_^OC + 2 _βbvb_^(SJ. Its projection onto a source SJ direction increases) with cross-domain alignment between _va_^SJand^_v_ _b_^SJ and decreases with alignment between the source SJ direction and the target OC component. Exp2B estimates these relations without imposing orthogonality between the latent components.

#### S5.4 Source-question-adjusted direction sensitivity

The pooled factorial directions weight the four source cells equally. A complementary estimator removes additive source-question variation before estimating the component directions. Let _x_^(+) _q_^and^(_x−_) _q_^(denote the activations of the correct and incorrect responses) retained for source question _q_ , and let _s_^(+) _q_^(_, s−_) _q_^(_∈{−_1)^(_,_+1)^(_}_denote their signed SJ labels. Under the response-level model)

where _o_^(+) _q_^(= +1 and)^(_o−_) _q_^(=)^(_−_1, the within-question difference is)

The additive question intercept _hq_ cancels exactly. At each layer, multivariate least squares across source pairs estimates the three coefficient vectors;^() _b_ SJ and^() _b_ OC replace _W_ meta and _W_ truth, respectively, in the unchanged cross-domain B/C evaluation. The four possible source-pair patterns are A/C, A/D, B/C, and B/D, with design rows [2 _, s_^(+) _q_^(_−s_) _q_^(_−, s_) _q_^(++)^_s_ _q_^(_−_]. Identifiability is checked) from the rank of this three-column design for every source sample and bootstrap replicate.

#### S5.5 Target-question-controlled all-cell model

The B/C analysis focuses on responses for which OC and SJ conflict. A complementary analysis uses responses from all four cells. For each component _k ∈{_ meta _,_ truth _}_ , projection scores are standardized separately within each target-domain and layer combination and then averaged over the fixed layer window. The resulting scores are entered into

The coefficients are therefore identified from variation among responses associated with the same question. The cross-component contrast is

|∆FE =_β_^((meta)) SJ _−β_^((truth)) OC _._|(S16)|
|---|---|

As a within-component semantic diagnostic, we also compute

|∆^((meta)) spec|=_β_^((meta)) SJ|_−β_^((meta)) OC _,_|(S17)|
|---|---|---|---|
|∆^((truth)) spec|=_β_^((truth)) OC|_−β_^((truth)) SJ _._|(S18)|

A positive value denotes the specificity implied by the source construction: SJ for _W_ meta and OC for _W_ truth. Each question-cluster bootstrap replicate repeats both score standardization and model estimation.

#### S5.6 Response-likelihood residualization

For free responses, response log-probability is the mean teacher-forced log-probability over all tokens in the generated assistant response. Within each target dataset and layer, each projection score is residualized against mean response log-probability and response-token count. OOD residual models additionally include answer-letter indicators. Component AUCs are computed from the layer-specific residual scores and then averaged over the fixed window, matching the primary estimand. Layer-wise nuisance regressions are refitted in every bootstrap replicate.

### S6 Evaluation and Statistical Inference

**Direction fitting and centering.** At each layer, cell means and the source centering vector are estimated exclusively from the source-domain rows or the training partition. AUCs are computed from the resulting projection scores; their rankings are invariant to positive rescaling of the fitted directions. For the all-cell fixed-effect analysis, the two projected components are standardized separately before their coefficients are compared.

Exp2A uses five-fold GroupKFold, with the question-cluster identifier as the grouping variable, so that the correct and incorrect responses associated with the same question remain in the same fold. Out-of-fold scores are concatenated before evaluation. Exp2B estimates each direction once from all strict source-domain rows and uses no target-domain labels during direction estimation.

**Layer summaries.** Layer-wise curves describe how each effect varies across network depth. Primary fixed-window summaries average layers satisfying _ℓ/_ ( _L −_ 1) _∈_ [0 _._ 40 _,_ 0 _._ 80]: layers 11–21 for the 28-layer Qwen2.5-7B, 13–24 for the 32-layer models, and 19–37 for Qwen2.5-14B. Within a bootstrap replicate, AUCs are computed by layer and then averaged over this fixed window.

The window covers a broad middle-to-late portion of each architecture, from 40% to 80% of normalized depth, rather than selecting a single layer from the observed results. Complete layer-wise curves are reported alongside the fixed-window summaries. For descriptive peak analyses, the model-level peak is the layer that maximizes the mean ∆CB across the two Math–Movies transfer directions. Peak intervals are descriptive and are not interpreted as selection-adjusted hypothesis tests.

**Table S5:** Descriptive all-layer summaries. “CI above” counts layer-direction rows whose target-question 95% interval is above the corresponding reference (0.5 for Exp1, zero for the remaining analyses).

|Analysis|Point estimate above reference|CI above reference|
|---|---|---|
|Exp1 cross-domain_W_mix|256/280 (91.4%)|192/280 (68.6%)|
|Exp2A in-domain∆CB|222/280 (79.3%)|176/280 (62.9%)|
|Exp2B cross-domain∆CB|255/280 (91.1%)|192/280 (68.6%)|
|OOD∆CB|458/560 (81.8%)|318/560 (56.8%)|

**Figure S1:** Question-grouped in-domain validation (Exp2A). Each direction is fitted within the training questions of a five-fold split and evaluated on held-out questions. Curves show ∆CB by layer; shaded bands are 95% target-question cluster-bootstrap intervals. The pale vertical band marks the fixed normalized layer window [0 _._ 40 _,_ 0 _._ 80].

**Cluster bootstrap.** Unless noted otherwise, confidence intervals are based on 1,000 question-cluster bootstrap resamples. Sampling a question retains all associated response rows, thereby preserving the B/C pairing and the repeated candidate responses in the OOD tasks. For the primary Exp2B fixed-window analysis, source and target question clusters are resampled independently, and both factorial directions are re-estimated in every replicate. For the all-layer figures and secondary analyses, the fitted source direction is held fixed, so the resulting intervals quantify target-population uncertainty conditional on that direction. Percentile 95% intervals are computed from valid replicates. All-layer counts summarize rows of the layer-by-direction result grid and are descriptive because adjacent layers are not independent observations.

### S7 In-Domain Validation and Layer-Wise Results

Figure S1 reports Exp2A with question-grouped folds. Seven of the eight fixed-window estimates are positive, with 95% confidence intervals excluding zero (Table S6). The only estimate whose interval includes zero is Llama-3.1-8B on Math, ∆CB = _−_ 0 _._ 042 [ _−_ 0 _._ 097, 0 _._ 012], while the same model on Movies shows the largest in-domain contrast. Across models and domains, positive contrasts generally emerge after the earliest layers and remain present across much of the fixed middle-to-late layer window.

Table S7 summarizes the descriptive peak layers for Exp2B. Within each model, a single layer is selected by maximizing the mean point estimate of ∆CB across the two Math–Movies transfer directions. The resulting peak summaries describe the location of the strongest observed cross-domain effect but are not used for the primary fixed-window inference.

**Table S6:** Exp2A fixed-window component AUCs and relative transfer asymmetry. Intervals use 1,000 question-cluster bootstrap resamples of the out-of-fold predictions.

|Model|Domain|_W_meta_→_SJ|_W_truth_→_OC|∆CB|_nB/C_|
|---|---|---|---|---|---|
|Qwen2.5-7B|Math|0.649 [0.607, 0.692]|0.476 [0.436, 0.519]|0.173 [0.096, 0.251]|1,887|
|Qwen2.5-7B|Movies|0.723 [0.687, 0.763]|0.483 [0.429, 0.543]|0.240 [0.150, 0.329]|513|
|Llama-3.1-8B|Math|0.652 [0.622, 0.680]|0.693 [0.662, 0.723]|-0.042 [-0.097, 0.012]|1,273|
|Llama-3.1-8B|Movies|0.915 [0.897, 0.931]|0.219 [0.188, 0.250]|0.696 [0.650, 0.742]|2,146|
|Qwen2.5-14B|Math|0.743 [0.715, 0.766]|0.395 [0.365, 0.430]|0.348 [0.288, 0.398]|1,563|
|Qwen2.5-14B|Movies|0.758 [0.725, 0.792]|0.352 [0.311, 0.393]|0.406 [0.336, 0.474]|1,340|
|OLMo-3-7B|Math|0.899 [0.865, 0.928]|0.422 [0.360, 0.484]|0.476 [0.395, 0.556]|325|
|OLMo-3-7B|Movies|0.867 [0.817, 0.916]|0.315 [0.249, 0.381]|0.552 [0.442, 0.660]|249|

**Table S7:** Descriptive cross-domain peak layers. Confidence intervals condition on the selected source direction and are not adjusted for peak selection.

|Model|Direction|Layer|Normalized depth|∆_CB_ [95% CI]|
|---|---|---|---|---|
|Qwen2.5-7B|Math_→_Movies|19|0.704|0.672 [0.600, 0.739]|
|Qwen2.5-7B|Movies_→_Math|19|0.704|0.476 [0.383, 0.563]|
|Llama-3.1-8B|Math_→_Movies|19|0.613|0.617 [0.563, 0.677]|
|Llama-3.1-8B|Movies_→_Math|19|0.613|0.270 [0.202, 0.336]|
|Qwen2.5-14B|Math_→_Movies|28|0.596|0.523 [0.457, 0.593]|
|Qwen2.5-14B|Movies_→_Math|28|0.596|0.704 [0.660, 0.743]|
|OLMo-3-7B|Math_→_Movies|27|0.871|0.863 [0.809, 0.917]|
|OLMo-3-7B|Movies_→_Math|27|0.871|0.429 [0.315, 0.536]|

#### S7.1 Joint source–target bootstrap

Table S8 reports the fixed-window inference used in the main Exp2B table. Each replicate independently resamples source and target question clusters, reconstructs _W_ meta and _W_ truth at every selected layer, and evaluates the refitted directions on the resampled target-domain B/C conflicts. All 1,000 replicates were valid in each of the eight transfer conditions, and every resampled source dataset retained observations from all four factorial cells.

### S8 Main-Domain Controls

#### S8.1 Source-question-adjusted direction sensitivity

Table S9 reports the cross-domain evaluation after estimating both source directions from Eq. S14. All eight ∆CB estimates remain positive, ranging from 0.130 to 0.748, and every joint source–target bootstrap 95% interval excludes zero. The point estimates differ from the corresponding pooled factorial estimates by at most 0.026. All eight source designs have rank three; all

**Table S8:** Exp2B joint source–target question bootstrap. Source _q_ and target _q_ are the numbers of question clusters entering direction fitting and B/C evaluation, respectively. Values are fixed-window AUCs or differences with 95% percentile intervals.

|Model|Direction|_W_meta_→_SJ|_W_truth_→_OC|∆CB|Source_q_|Target_q_|Valid|
|---|---|---|---|---|---|---|---|
|Qwen2.5-7B|Math_→_Movies|0.750 [0.715, 0.782]|0.300 [0.263, 0.339]|0.450 [0.386, 0.513]|2,307|468|1,000|
|Qwen2.5-7B|Movies_→_Math|0.670 [0.629, 0.708]|0.368 [0.332, 0.414]|0.302 [0.219, 0.371]|909|1,824|1,000|
|Llama-3.1-8B|Math_→_Movies|0.823 [0.796, 0.848]|0.222 [0.191, 0.252]|0.601 [0.545, 0.655]|1,539|2,035|1,000|
|Llama-3.1-8B|Movies_→_Math|0.572 [0.540, 0.603]|0.434 [0.400, 0.471]|0.138 [0.073, 0.198]|2,775|1,240|1,000|
|Qwen2.5-14B|Math_→_Movies|0.708 [0.679, 0.736]|0.286 [0.259, 0.321]|0.422 [0.365, 0.470]|2,370|1,274|1,000|
|Qwen2.5-14B|Movies_→_Math|0.781 [0.753, 0.802]|0.215 [0.196, 0.246]|0.566 [0.513, 0.602]|1,659|1,493|1,000|
|OLMo-3-7B|Math_→_Movies|0.928 [0.899, 0.950]|0.167 [0.119, 0.243]|0.761 [0.667, 0.818]|362|249|1,000|
|OLMo-3-7B|Movies_→_Math|0.776 [0.726, 0.821]|0.441 [0.379, 0.511]|0.334 [0.229, 0.430]|508|318|1,000|

1,000 bootstrap replicates remain identifiable, including OLMo-3-7B Movies, where the A/C, A/D, and B/D patterns identify the coefficients without observed B/C source pairs.

**Table S9:** Source-question-adjusted Exp2B sensitivity. Directions are estimated from within-question correct-minus-incorrect activation differences. AC/AD/BC/BD gives the source-pair support for the four SJ combinations; Rank is the three-column design rank, and Valid is the number of identifiable joint source–target bootstrap replicates. Entries are fixed-window AUCs or differences with 95% percentile intervals. The source-adjusted directions^() _b_ SJ and^() _b_ OC occupy the evaluation roles of the pooled _W_ meta and _W_ truth, respectively.

|Model|Direction|_b_SJ_→_SJ|_b_OC_→_OC|∆CB|AC/AD/BC/BD|Rank|Valid|
|---|---|---|---|---|---|---|---|
|Qwen2.5-7B|Math_→_Movies|0.744 [0.709, 0.778]|0.288 [0.253, 0.330]|0.456 [0.389, 0.517]|1,656/483/63/105|3|1,000|
|Qwen2.5-7B|Movies_→_Math|0.676 [0.634, 0.714]|0.379 [0.340, 0.423]|0.297 [0.215, 0.366]|325/444/45/98|3|1,000|
|Llama-3.1-8B|Math_→_Movies|0.787 [0.754, 0.818]|0.212 [0.183, 0.243]|0.575 [0.514, 0.634]|325/299/33/882|3|1,000|
|Llama-3.1-8B|Movies_→_Math|0.573 [0.540, 0.604]|0.443 [0.409, 0.477]|0.130 [0.066, 0.191]|185/742/108/1,745|3|1,000|
|Qwen2.5-14B|Math_→_Movies|0.722 [0.690, 0.750]|0.283 [0.256, 0.318]|0.439 [0.375, 0.489]|179/877/70/1,244|3|1,000|
|Qwen2.5-14B|Movies_→_Math|0.785 [0.756, 0.806]|0.224 [0.204, 0.257]|0.562 [0.503, 0.596]|85/385/65/1,125|3|1,000|
|OLMo-3-7B|Math_→_Movies|0.896 [0.853, 0.928]|0.148 [0.110, 0.213]|0.748 [0.652, 0.810]|214/44/7/97|3|1,000|
|OLMo-3-7B|Movies_→_Math|0.746 [0.692, 0.794]|0.425 [0.356, 0.504]|0.321 [0.206, 0.420]|116/259/0/133|3|1,000|

#### S8.2 OC-only mass-mean control

The OC-only direction is _W_ OC _−_ only = _E_ [ _x |_ OC = 1] _− E_ [ _x |_ OC = 0], estimated from all source rows without SJ labels. This is a canonical unwhitened mass-mean direction; because each retained source item contributes one correct and one incorrect response, it is also equal to the average within-question correct-minus-incorrect activation difference.

The fitted direction is evaluated on B/C conflicts in the target domain. In all eight transfer conditions, it assigns higher scores to C responses, which are incorrect but self-endorsed, than to B responses, which are correct but self-rejected (Table S10). Thus, the conflict-set reversal observed for the mixed contrast is also present when the source direction is estimated solely from objective-correctness labels.

**Table S10:** Fixed-window OC-only mass-mean control. The direction is the unwhitened source class-mean contrast _E_ [ _x |_ OC = 1] _− E_ [ _x |_ OC = 0], fitted without SJ labels. AUC(C _>_ B) above 0.5 means that this correctness-labelled source direction ranks incorrect/self-endorsed target responses above correct/self-rejected responses. Intervals condition on the fitted source direction.

|Model|Direction|AUC(C_>_B) [95% CI]|Rows|Questions|
|---|---|---|---|---|
|Qwen2.5-7B|Math_→_Movies|0.722 [0.688, 0.754]|513|468|
|Qwen2.5-7B|Movies_→_Math|0.653 [0.615, 0.691]|1,887|1,824|
|Llama-3.1-8B|Math_→_Movies|0.794 [0.763, 0.822]|2,146|2,035|
|Llama-3.1-8B|Movies_→_Math|0.567 [0.536, 0.599]|1,273|1,240|
|Qwen2.5-14B|Math_→_Movies|0.721 [0.691, 0.751]|1,340|1,274|
|Qwen2.5-14B|Movies_→_Math|0.788 [0.763, 0.811]|1,563|1,493|
|OLMo-3-7B|Math_→_Movies|0.874 [0.832, 0.911]|249|249|
|OLMo-3-7B|Movies_→_Math|0.686 [0.635, 0.733]|325|318|

#### S8.3 Window-level null directions

Two nulls test whether large fixed-window contrasts arise from the target quadrant structure alone. The source-label-shuffle null independently permutes source OC/SJ labels before direction construction. The random-direction null replaces each fitted direction with a normalized random vector of the same dimensionality. Each null uses 100 repetitions and is evaluated on the unchanged target conflict set. Figure S2 and Table S11 compare their distributions with the observed fixed-window effects.

#### S8.4 B/C token-count matching

Token-count matching evaluates whether differences in complete rendered question-plus-response sequence length account for the separation between B and C responses. Matching is performed within the target domain before AUC computation and does not alter source-direction estimation. The balance-optimized procedure evaluates alternative token-count coarsenings and selects the specification whose token-count-only SJ AUC is closest to 0.5, breaking ties in favor of retaining more responses. Standardized mean difference (SMD) is reported as an additional balance diagnostic. A second specification uses fixed pooled token-count deciles.

Table S12 reports the number of retained responses, post-matching balance, and the resulting fixed-window contrasts. The estimated contrasts remain positive under both matching procedures, including conditions in which achieving token-count balance requires substantial trimming.

**Figure S2:** Window-level null controls for Exp2B. Points show the observed ∆CB; null distributions are obtained from 100 source-label shuffles or random direction pairs. The same target B/C rows and fixed normalized layer window are used for observed and null estimates.

**Table S11:** Numerical window-level null summaries. The empirical probability is the plus-one Monte Carlo estimate ( _b_ + 1) _/_ ( _R_ + 1), where _b_ is the number of valid null repetitions at least as large as the observed effect.

|Model|Direction|Null|Observed|Null mean [95% range]|_p_emp|
|---|---|---|---|---|---|
|Llama-3.1-8B|Math_→_Movies|Random direction|0.601|-0.000 [-0.075, 0.075]|0.010|
|Llama-3.1-8B|Math_→_Movies|Label shuffle|0.601|0.002 [-0.119, 0.145]|0.010|
|Llama-3.1-8B|Movies_→_Math|Random direction|0.138|-0.001 [-0.080, 0.052]|0.010|
|Llama-3.1-8B|Movies_→_Math|Label shuffle|0.138|0.003 [-0.041, 0.046]|0.010|
|OLMo-3-7B|Math_→_Movies|Random direction|0.761|-0.006 [-0.075, 0.065]|0.010|
|OLMo-3-7B|Math_→_Movies|Label shuffle|0.761|-0.009 [-0.147, 0.141]|0.010|
|OLMo-3-7B|Movies_→_Math|Random direction|0.334|0.004 [-0.076, 0.094]|0.010|
|OLMo-3-7B|Movies_→_Math|Label shuffle|0.334|-0.000 [-0.139, 0.106]|0.010|
|Qwen2.5-14B|Math_→_Movies|Random direction|0.422|0.000 [-0.032, 0.030]|0.010|
|Qwen2.5-14B|Math_→_Movies|Label shuffle|0.422|0.001 [-0.065, 0.065]|0.010|
|Qwen2.5-14B|Movies_→_Math|Random direction|0.566|0.000 [-0.047, 0.049]|0.010|
|Qwen2.5-14B|Movies_→_Math|Label shuffle|0.566|0.004 [-0.075, 0.071]|0.010|
|Qwen2.5-7B|Math_→_Movies|Random direction|0.450|-0.003 [-0.053, 0.052]|0.010|
|Qwen2.5-7B|Math_→_Movies|Label shuffle|0.450|-0.003 [-0.097, 0.092]|0.010|
|Qwen2.5-7B|Movies_→_Math|Random direction|0.302|0.003 [-0.044, 0.064]|0.010|
|Qwen2.5-7B|Movies_→_Math|Label shuffle|0.302|-0.004 [-0.087, 0.074]|0.010|

**Table S12:** Token-count matching diagnostics. SMD is the standardized B/C mean difference in target rendered-sequence token count. Matching is fixed before inference; intervals resample target questions within the matched set.

|Model|Direction|Rule|Rows|Keep|SMD pre|SMD post|∆CB [95% CI]|
|---|---|---|---|---|---|---|---|
|Llama-3.1-8B|Math_→_Movies|Optimized|586|27.3%|0.104|0.012|0.625 [0.552, 0.684]|
|Llama-3.1-8B|Math_→_Movies|Fixed deciles|586|27.3%|0.104|0.027|0.622 [0.563, 0.684]|
|Llama-3.1-8B|Movies_→_Math|Optimized|468|36.8%|-1.222|0.000|0.415 [0.333, 0.499]|
|Llama-3.1-8B|Movies_→_Math|Fixed deciles|480|37.7%|-1.222|-0.015|0.394 [0.318, 0.483]|
|OLMo-3-7B|Math_→_Movies|Optimized|202|81.1%|0.182|0.018|0.772 [0.697, 0.838]|
|OLMo-3-7B|Math_→_Movies|Fixed deciles|202|81.1%|0.182|0.018|0.772 [0.700, 0.837]|
|OLMo-3-7B|Movies_→_Math|Optimized|154|47.4%|-1.079|-0.075|0.198 [0.065, 0.329]|
|OLMo-3-7B|Movies_→_Math|Fixed deciles|154|47.4%|-1.079|-0.075|0.198 [0.066, 0.330]|
|Qwen2.5-14B|Math_→_Movies|Optimized|300|22.4%|0.139|0.027|0.418 [0.342, 0.486]|
|Qwen2.5-14B|Math_→_Movies|Fixed deciles|300|22.4%|0.139|0.055|0.488 [0.422, 0.559]|
|Qwen2.5-14B|Movies_→_Math|Optimized|498|31.9%|-0.403|0.005|0.628 [0.569, 0.683]|
|Qwen2.5-14B|Movies_→_Math|Fixed deciles|498|31.9%|-0.403|0.005|0.608 [0.542, 0.673]|
|Qwen2.5-7B|Math_→_Movies|Optimized|286|55.8%|0.129|-0.016|0.429 [0.346, 0.509]|
|Qwen2.5-7B|Math_→_Movies|Fixed deciles|286|55.8%|0.129|-0.016|0.429 [0.346, 0.505]|
|Qwen2.5-7B|Movies_→_Math|Optimized|336|17.8%|-0.009|-0.037|0.293 [0.198, 0.392]|
|Qwen2.5-7B|Movies_→_Math|Fixed deciles|336|17.8%|-0.009|-0.007|0.277 [0.174, 0.374]|

**Table S13:** Free-response likelihood control. “logp _→_ OC/SJ” is the AUC of mean assistant-response token log-probability alone on the same B/C rows. Residual effects control projection scores for mean response log-probability and response-token count separately at each layer, followed by the same layer-AUC averaging as the primary analysis. Nuisance models are refitted in every bootstrap replicate.

|Model|Direction|logp_→_OC|logp_→_SJ|Raw∆_CB_|Residual∆_CB_|
|---|---|---|---|---|---|
|Qwen2.5-7B|Math_→_Movies|0.692|0.308|0.450 [0.389, 0.513]|0.470 [0.409, 0.529]|
|Qwen2.5-7B|Movies_→_Math|0.417|0.583|0.302 [0.227, 0.381]|0.339 [0.275, 0.407]|
|Llama-3.1-8B|Math_→_Movies|0.645|0.355|0.601 [0.543, 0.651]|0.624 [0.564, 0.673]|
|Llama-3.1-8B|Movies_→_Math|0.487|0.513|0.138 [0.076, 0.201]|0.280 [0.219, 0.345]|
|Qwen2.5-14B|Math_→_Movies|0.562|0.438|0.422 [0.368, 0.477]|0.431 [0.377, 0.485]|
|Qwen2.5-14B|Movies_→_Math|0.420|0.580|0.566 [0.520, 0.610]|0.619 [0.575, 0.660]|
|OLMo-3-7B|Math_→_Movies|0.546|0.454|0.761 [0.694, 0.821]|0.751 [0.678, 0.814]|
|OLMo-3-7B|Movies_→_Math|0.232|0.768|0.334 [0.236, 0.419]|0.180 [0.083, 0.267]|

**Table S14:** Question-fixed-effect cross-domain projection analysis. Scores are standardized by component and target layer, then averaged over the fixed window.

|Model|Direction|_β_^_meta_ _SJ_|_β_^_truth_ _OC_|∆_F E_ [95% CI]|Questions|
|---|---|---|---|---|---|
|Qwen2.5-7B|Math_→_Movies|0.515|0.173|0.342 [0.266, 0.419]|909|
|Qwen2.5-7B|Movies_→_Math|0.496|0.165|0.331 [0.270, 0.388]|2,307|
|Llama-3.1-8B|Math_→_Movies|0.615|0.211|0.403 [0.353, 0.455]|2,775|
|Llama-3.1-8B|Movies_→_Math|0.456|0.226|0.230 [0.155, 0.295]|1,539|
|Qwen2.5-14B|Math_→_Movies|0.414|0.113|0.301 [0.241, 0.358]|1,659|
|Qwen2.5-14B|Movies_→_Math|0.539|0.120|0.419 [0.374, 0.462]|2,370|
|OLMo-3-7B|Math_→_Movies|0.724|0.194|0.530 [0.432, 0.621]|508|
|OLMo-3-7B|Movies_→_Math|0.612|0.098|0.514 [0.314, 0.737]|362|

#### S8.5 Response likelihood

Mean response log-probability itself often discriminates OC and SJ on the conflict set, but it does not explain the projection asymmetry. Table S13 reports raw and residualized effects after controlling each projection score for mean response log-probability and response-token count at each layer. All eight residual ∆CB estimates remain positive with 95% intervals above zero.

#### S8.6 Question fixed effects

The all-cell model in Eq. S15 compares responses within question and does not require restricting the fit to B/C rows. Across all eight transfers, _β_ SJ^((meta)) exceeds _β_ OC^((truth)) and the cluster interval for ∆FE remains above zero (Table S14). This result links the conflict-set AUC pattern to the full 2 _×_ 2 design.

The complete coefficient matrix in Table S15 shows that both transferred components have positive OC and SJ cross-loadings. In every direction, the SJ coefficient exceeds the OC coefficient for both _W_ meta and _W_ truth. The latter pattern is consistent with the conflict-set reversal: a source correctness-associated contrast need not retain OC specificity after transfer. The factorial names identify how the source directions are constructed, not pure target-domain latent axes. The within-component contrasts in Table S16 quantify this asymmetry. All eight ∆^((meta)) spec estimates are positive with 95% intervals above zero (0.192–0.619), whereas all eight ∆^((truth)) spec estimates are negative with intervals below zero ( _−_ 0 _._ 490– _−_ 0 _._ 214). Thus the transferred _W_ meta direction retains SJ specificity, while the transferred _W_ truth direction loads more strongly on SJ than OC.

#### S8.7 Judgement scoring and data-construction robustness

The stored judgement score was originally computed from the returned next-token candidates as described in Sec. S4. We recomputed the exact literal Yes and No logits at the first assistant-generation position for every pre-filter pair row. Because inference backends need not reproduce token logits bit-for-bit, the scoring-rule audit holds these recomputed logits fixed and compares the historical top-four fallback with full binary normalization, _p_ judge = sigmoid( _z_ Yes _− z_ No). Across all eight model–domain cells, the maximum row-wise difference was 1 _._ 30 _×_ 10^(_−_6) . The two scoring rules produced no SJ label changes

**Table S15:** Complete question-fixed-effect cross-loading matrix. Panel A projects the source _W_ meta direction; Panel B projects the source _W_ truth direction. Entries are coefficients with target-question cluster-bootstrap 95% confidence intervals.

|||_Panel A: W_meta _proj_|_ection_||
|---|---|---|---|---|
|Model|Direction|_βOC_|_βSJ_|_βINT_|
|Qwen2.5-7B|Math_→_Movies|0.172 [0.131, 0.211]|0.515 [0.468, 0.560]|-0.124 [-0.174, -0.073]|
|Qwen2.5-7B|Movies_→_Math|0.200 [0.171, 0.232]|0.496 [0.456, 0.535]|0.135 [0.102, 0.166]|
|Llama-3.1-8B|Math_→_Movies|0.231 [0.211, 0.251]|0.615 [0.580, 0.651]|-0.103 [-0.127, -0.081]|
|Llama-3.1-8B|Movies_→_Math|0.265 [0.241, 0.288]|0.456 [0.399, 0.511]|-0.091 [-0.113, -0.068]|
|Qwen2.5-14B|Math_→_Movies|0.184 [0.159, 0.210]|0.414 [0.369, 0.456]|-0.127 [-0.155, -0.096]|
|Qwen2.5-14B|Movies_→_Math|0.162 [0.140, 0.185]|0.539 [0.509, 0.567]|0.029 [0.003, 0.056]|
|OLMo-3-7B|Math_→_Movies|0.105 [0.061, 0.151]|0.724 [0.659, 0.781]|-0.055 [-0.102, -0.011]|
|OLMo-3-7B|Movies_→_Math|0.142 [0.086, 0.194]|0.612 [0.436, 0.807]|-0.106 [-0.165, -0.044]|

_Panel B: W_ truth _projection_

|Model|Direction|_βOC_|_βSJ_|_βINT_|
|---|---|---|---|---|
|Qwen2.5-7B|Math_→_Movies|0.173 [0.134, 0.207]|0.423 [0.377, 0.472]|-0.096 [-0.142, -0.048]|
|Qwen2.5-7B|Movies_→_Math|0.165 [0.136, 0.196]|0.378 [0.338, 0.419]|0.098 [0.066, 0.130]|
|Llama-3.1-8B|Math_→_Movies|0.211 [0.188, 0.236]|0.554 [0.517, 0.593]|-0.085 [-0.111, -0.060]|
|Llama-3.1-8B|Movies_→_Math|0.226 [0.203, 0.251]|0.462 [0.403, 0.522]|-0.089 [-0.112, -0.065]|
|Qwen2.5-14B|Math_→_Movies|0.113 [0.082, 0.145]|0.347 [0.302, 0.390]|-0.063 [-0.098, -0.027]|
|Qwen2.5-14B|Movies_→_Math|0.120 [0.100, 0.140]|0.460 [0.433, 0.488]|0.001 [-0.023, 0.026]|
|OLMo-3-7B|Math_→_Movies|0.194 [0.151, 0.242]|0.685 [0.618, 0.745]|-0.149 [-0.192, -0.100]|
|OLMo-3-7B|Movies_→_Math|0.098 [0.044, 0.153]|0.397 [0.224, 0.587]|-0.083 [-0.147, -0.020]|

**Table S16:** Within-component specificity diagnostics for the all-cell model. Positive values indicate the source-implied target specificity: SJ for _W_ meta and OC for _W_ truth. Entries are coefficient differences with target-question cluster-bootstrap 95% confidence intervals.

|Model|Direction|_β_^_meta_ _SJ_ _−β_^_meta_ _OC_ [95% CI]|_β_^_truth_ _OC_ _−β_^_truth_ _SJ_ [95% CI]|
|---|---|---|---|
|Qwen2.5-7B|Math_→_Movies|0.343 [0.265, 0.423]|-0.250 [-0.332, -0.177]|
|Qwen2.5-7B|Movies_→_Math|0.296 [0.232, 0.355]|-0.214 [-0.275, -0.150]|
|Llama-3.1-8B|Math_→_Movies|0.384 [0.338, 0.433]|-0.343 [-0.395, -0.292]|
|Llama-3.1-8B|Movies_→_Math|0.192 [0.119, 0.260]|-0.235 [-0.303, -0.161]|
|Qwen2.5-14B|Math_→_Movies|0.230 [0.172, 0.282]|-0.234 [-0.294, -0.169]|
|Qwen2.5-14B|Movies_→_Math|0.378 [0.328, 0.423]|-0.340 [-0.384, -0.296]|
|OLMo-3-7B|Math_→_Movies|0.619 [0.515, 0.713]|-0.490 [-0.586, -0.384]|
|OLMo-3-7B|Movies_→_Math|0.469 [0.269, 0.689]|-0.298 [-0.514, -0.105]|

at 0.5 and no complete-pair membership changes under the primary paired _τ_ = 0 _._ 7 criterion. The candidate-return truncation therefore did not determine the analyzed strict samples.

**Strict-pair resampling.** The main analysis samples one correct and one incorrect response from each question’s usable pass@8 responses. To measure sensitivity to this draw, we constructed a second Qwen2.5-7B pair sample from the frozen response pools using seed 2027. Each correct and incorrect response was drawn uniformly from its corresponding pool; the originally selected response remained eligible, and repeated generation slots retained their empirical multiplicity. At least one response changed in 2,441/2,703 Math pairs (90.3%) and 645/1,101 Movies pairs (58.6%). Full Yes/No normalization followed by the paired _τ_ = 0 _._ 7 criterion retained 2,289 Math pairs and 907 Movies pairs, with observations in all four factorial cells.

All-layer ∆CB profiles from the resampled pairs correlated with the original profiles at 0.996 for Math _→_ Movies and 0.991 for Movies _→_ Math. In each direction, 23/28 layer-wise estimates were positive. The fixed-window effects remained positive under 1,000 joint source–target question bootstrap resamples with source-direction refitting (Table S17). Thus the cross-domain pattern is preserved under an independently repeated draw from the finite pass@8 response pools.

**Table S17:** Qwen2.5-7B strict-pair resampling sensitivity. “Original” gives the main-analysis window effect; R2 columns are computed from the second pair draw. Entries are fixed-window AUCs or differences with 95% percentile intervals from 1,000 joint source–target question bootstrap resamples.

|Direction|Original∆CB|R2_W_meta_→_SJ|R2_W_truth_→_OC|R2∆CB|
|---|---|---|---|---|
|Math_→_Movies|0.450 [0.386, 0.513]|0.763 [0.728, 0.796]|0.315 [0.278, 0.358]|0.448 [0.374, 0.514]|
|Movies_→_Math|0.302 [0.219, 0.371]|0.654 [0.614, 0.695]|0.405 [0.362, 0.448]|0.249 [0.168, 0.330]|

**Multi-reference Movies prompts.** The 159 Movies prompt strings linked to more than one reference actor define a separate label-ambiguity sensitivity. We removed every retained response whose exact prompt belonged to this set before recomputing Exp2B. For Math _→_ Movies this changes the target conflict set; for Movies _→_ Math it changes the source sample, and both factorial directions are refitted. The filter removed 1.38%–2.41% of retained Movies rows across models and 6–39 B/C responses.

All eight filtered fixed-window effects remained positive with 95% intervals above zero (Table S18). Their all-layer profiles correlated with the original profiles at _r ≥_ 0 _._ 999, and the largest absolute change in a window effect was 0.030. The weakest original condition, Llama-3.1-8B Movies _→_ Math, changed from 0.138 to 0.140 [0.075, 0.204]. The cross-domain result is therefore unchanged when multi-reference Movies prompts are excluded rather than handled only through question clustering.

**Table S18:** Movies multi-reference prompt sensitivity. All retained responses from prompts associated with multiple reference actors are removed before direction fitting or target evaluation. Filtered entries are fixed-window ∆CB estimates with 95% percentile intervals from 1,000 joint source–target question bootstrap resamples with direction refitting.

|Model|Removed rows (%)|Direction|Original∆CB|Filtered∆CB|
|---|---|---|---|---|
|Qwen2.5-7B|44 (2.41)|Math_→_Movies|0.450|0.444 [0.374, 0.508]|
|Qwen2.5-7B|44 (2.41)|Movies_→_Math|0.302|0.303 [0.221, 0.375]|
|Llama-3.1-8B|80 (1.44)|Math_→_Movies|0.601|0.631 [0.572, 0.685]|
|Llama-3.1-8B|80 (1.44)|Movies_→_Math|0.138|0.140 [0.075, 0.204]|
|Qwen2.5-14B|54 (1.63)|Math_→_Movies|0.422|0.419 [0.359, 0.469]|
|Qwen2.5-14B|54 (1.63)|Movies_→_Math|0.566|0.566 [0.512, 0.603]|
|OLMo-3-7B|14 (1.38)|Math_→_Movies|0.761|0.753 [0.661, 0.816]|
|OLMo-3-7B|14 (1.38)|Movies_→_Math|0.334|0.335 [0.222, 0.435]|

### S9 Self-Judgement Threshold Sensitivity

The primary strict analysis uses _τ_ = 0 _._ 7 at the pair level. A complementary row-level sensitivity analysis instead reconstructs the pre-filter population by joining the original activation files with a supplementary activation patch for rows that had been removed at _τ_ = 0 _._ 7. For each _τ ∈{_ 0 _._ 5 _,_ 0 _._ 6 _,_ 0 _._ 7 _,_ 0 _._ 8 _}_ , it removes the uncertain band 1 _− τ < p_ judge _< τ_ and assigns SJ from the retained side. This is a row-filter sensitivity population; the primary sample instead requires both members of a question pair to pass _τ_ = 0 _._ 7.

Figure S3 shows why threshold changes affect models differently. Qwen distributions are highly polarized, whereas OLMo assigns much more mass near 0.5. At _τ_ = 0 _._ 8, only 14.0% of OLMo Math rows remain, so that setting is a support stress test rather than an equally powered analysis. Table S19 gives the complete retained quadrant counts.

**Figure S3:** Pre-filter distributions of _p_ judge for every model and main domain. Vertical references mark the lower and upper boundaries induced by the symmetric confidence thresholds. These probabilities precede the main paired strict filter.

**Table S19:** Row-level support under symmetric confidence filtering. “Kept” is the number of pre-filter judged rows outside the uncertain band.

|Model|Domain|_τ_|Kept|Keep|A|B|C|D|B/C|
|---|---|---|---|---|---|---|---|---|---|
|Qwen2.5-7B|Math|0.500|5,406|100.0%|2,459|244|1,942|761|2,186|
|Qwen2.5-7B|Math|0.600|5,230|96.7%|2,434|222|1,861|713|2,083|
|Qwen2.5-7B|Math|0.700|4,992|92.3%|2,408|194|1,769|621|1,963|
|Qwen2.5-7B|Math|0.800|4,751|87.9%|2,371|164|1,659|557|1,823|
|Qwen2.5-7B|Movies|0.500|2,202|100.0%|902|199|454|647|653|
|Qwen2.5-7B|Movies|0.600|2,122|96.4%|875|184|433|630|617|
|Qwen2.5-7B|Movies|0.700|2,007|91.1%|838|162|407|600|569|
|Qwen2.5-7B|Movies|0.800|1,903|86.4%|803|144|378|578|522|
|Llama-3.1-8B|Math|0.500|6,788|100.0%|1,707|1,687|909|2,485|2,596|
|Llama-3.1-8B|Math|0.600|5,809|85.6%|1,316|1,482|663|2,348|2,145|
|Llama-3.1-8B|Math|0.700|4,569|67.3%|964|1,096|451|2,058|1,547|
|Llama-3.1-8B|Math|0.800|3,440|50.7%|654|769|294|1,723|1,063|
|Llama-3.1-8B|Movies|0.500|7,364|100.0%|1,365|2,317|527|3,155|2,844|
|Llama-3.1-8B|Movies|0.600|6,923|94.0%|1,174|2,218|438|3,093|2,656|
|Llama-3.1-8B|Movies|0.700|6,371|86.5%|1,029|1,999|358|2,985|2,357|
|Llama-3.1-8B|Movies|0.800|5,789|78.6%|869|1,771|295|2,854|2,066|
|Qwen2.5-14B|Math|0.500|5,210|100.0%|1,193|1,412|316|2,289|1,728|
|Qwen2.5-14B|Math|0.600|5,109|98.1%|1,153|1,390|286|2,280|1,676|
|Qwen2.5-14B|Math|0.700|4,968|95.4%|1,110|1,349|261|2,248|1,610|
|Qwen2.5-14B|Math|0.800|4,816|92.4%|1,069|1,310|232|2,205|1,542|
|Qwen2.5-14B|Movies|0.500|3,532|100.0%|529|1,237|170|1,596|1,407|
|Qwen2.5-14B|Movies|0.600|3,486|98.7%|506|1,228|162|1,590|1,390|
|Qwen2.5-14B|Movies|0.700|3,424|96.9%|483|1,211|154|1,576|1,365|
|Qwen2.5-14B|Movies|0.800|3,376|95.6%|462|1,204|139|1,571|1,343|
|OLMo-3-7B|Math|0.500|3,168|100.0%|1,153|431|898|686|1,329|
|OLMo-3-7B|Math|0.600|2,159|68.2%|809|276|575|499|851|
|OLMo-3-7B|Math|0.700|1,357|42.8%|507|184|322|344|506|
|OLMo-3-7B|Math|0.800|445|14.0%|98|98|62|187|160|
|OLMo-3-7B|Movies|0.500|2,186|100.0%|783|310|308|785|618|
|OLMo-3-7B|Movies|0.600|1,806|82.6%|642|234|219|711|453|
|OLMo-3-7B|Movies|0.700|1,465|67.0%|511|163|148|643|311|
|OLMo-3-7B|Movies|0.800|1,004|45.9%|301|103|67|533|170|

The sensitivity run uses 100 question-cluster bootstrap resamples. Across all 280 Exp2B layer-direction rows, positive ∆CB occurs in 228, 239, 252, and 251 rows at _τ_ = 0 _._ 5 _,_ 0 _._ 6 _,_ 0 _._ 7 _,_ and 0.8, respectively; the corresponding intervals lie above zero in 178, 193, 196, and 205 rows. The cross-domain pattern therefore persists across alternative confidence filters. Increasing _τ_ removes more rows near the decision boundary and generally strengthens separation, while reducing support most sharply for OLMo.

### S10 Zero-Target-Fitting OOD Analyses

Source directions are fitted only on Math or Movies strict samples and applied unchanged to MMLU and binary TruthfulQA. OOD records retain only high-confidence B/C candidates under the same symmetric _τ_ = 0 _._ 7 rule. Because several candidates can originate from one question, all intervals cluster by the original target question.

Table S20 reports support and the full layer-by-source descriptive counts. Positive ∆CB appears in 458/560 (81.8%) rows, with question-cluster intervals above zero in 318/560 (56.8%). The main OOD figure’s component curves show that this difference is driven by _W_ meta _→_ SJ rising above chance while _W_ truth _→_ OC remains lower on the same target conflicts. The result occurs for both a four-option knowledge benchmark and a binary truthfulness benchmark, although effect magnitude varies by model, source, and target.

**Table S20:** OOD support and all-layer heterogeneity. “Rows” in the final two columns are source-by-layer evaluations: two source directions per model-target pair times the number of model layers.

|Model|Target|B|C|Questions|Mean∆_CB_|∆_>_0|CI_>_0|
|---|---|---|---|---|---|---|---|
|Llama-3.1-8B|MMLU|13|403|252|0.206|47/64|31/64|
|Llama-3.1-8B|TruthfulQA|357|160|459|0.267|53/64|46/64|
|OLMo-3-7B|MMLU|70|128|160|0.223|57/64|44/64|
|OLMo-3-7B|TruthfulQA|448|84|476|0.228|54/64|42/64|
|Qwen2.5-14B|MMLU|169|75|205|0.207|69/96|48/96|
|Qwen2.5-14B|TruthfulQA|383|48|388|0.357|91/96|66/96|
|Qwen2.5-7B|MMLU|112|189|224|0.072|43/56|12/56|
|Qwen2.5-7B|TruthfulQA|401|147|483|0.154|44/56|29/56|

#### S10.1 Forced-choice likelihood and answer-letter controls

Forced-choice answer likelihood is measured as the teacher-forced probability of the inserted option letter. Within the B/C conflict set, its OC and SJ AUCs are complementary by construction, but their direction and magnitude vary across models and target datasets. To control for candidate likelihood and option identity, each layer-specific projection score is residualized with respect to option log-probability, token count, and answer-letter indicators. The nuisance regression is re-estimated within each bootstrap replicate. Table S21 reports the resulting contrast. Fifteen of sixteen letter-controlled estimates remain positive with intervals above zero. The exception is OLMo Movies _→_ TruthfulQA, ∆CB = _−_ 0 _._ 030 [-0.138, 0.077], whose 95% CI includes zero.

#### S10.2 Counterbalanced target-side judgement labels

The primary SJ label uses a Yes/No response. To test whether the OOD result is tied to assent or a fixed response-token mapping, we re-elicited target-side judgements with two prompts:

Evaluate whether the answer above is correct. Respond with X if it is correct and Y if it is incorrect.

Evaluate whether the answer above is correct. Respond with Y if it is correct and X if it is incorrect.

X/Y avoids reusing the A–D labels already assigned to forced-choice answers. For each candidate, we compute correctness-oriented log odds

##### and define

The arithmetic mean of the two mapping-corrected probabilities is stored only as a secondary diagnostic and is not used to define the reported labels or transfer results. The primary analysis applies the same symmetric _τ_ = 0 _._ 7 rule to _p_ bal. Because this changes which candidates belong to B and C, answer-token activations were extracted for the complete new conflict set at the same forced-answer final-token site. Math- and Movies-fitted directions, layer windows, and target-question bootstrap clusters remain unchanged.

Mapping-corrected XY and YX scores correlate strongly in every model–target cell ( _ρ_ = 0 _._ 917–0 _._ 975), and their jointly high-confidence binary labels agree in 92.7%–100% of candidates (Table S22). The balanced MMLU conflict sets are sparse on B for Llama-3.1-8B ( _nB_ = 13) and OLMo-3-7B ( _nB_ = 4); those cells consequently provide weak raw inferential support despite having many C rows.

**Table S21:** OOD answer-likelihood and letter control. Raw ∆CB is the component difference before residualization; the final column controls answer-option log-probability, target token count, and answer letter. Binary TruthfulQA has one non-reference letter indicator; MMLU has three.

|Model|Direction|logp_→_OC|logp_→_SJ|Raw∆_CB_|Letter+logp residual∆_CB_|
|---|---|---|---|---|---|
|Qwen2.5-7B|Math_→_MMLU|0.670|0.330|0.112|0.185 [0.080, 0.280]|
|Qwen2.5-7B|Math_→_TruthfulQA|0.497|0.503|0.141|0.142 [0.051, 0.226]|
|Qwen2.5-7B|Movies_→_MMLU|0.670|0.330|0.091|0.200 [0.112, 0.283]|
|Qwen2.5-7B|Movies_→_TruthfulQA|0.497|0.503|0.227|0.242 [0.177, 0.305]|
|Llama-3.1-8B|Math_→_MMLU|0.557|0.443|0.350|0.513 [0.292, 0.725]|
|Llama-3.1-8B|Math_→_TruthfulQA|0.349|0.651|0.396|0.402 [0.322, 0.484]|
|Llama-3.1-8B|Movies_→_MMLU|0.557|0.443|0.333|0.627 [0.415, 0.811]|
|Llama-3.1-8B|Movies_→_TruthfulQA|0.349|0.651|0.425|0.455 [0.369, 0.532]|
|Qwen2.5-14B|Math_→_MMLU|0.704|0.296|0.389|0.395 [0.303, 0.476]|
|Qwen2.5-14B|Math_→_TruthfulQA|0.389|0.611|0.579|0.620 [0.533, 0.698]|
|Qwen2.5-14B|Movies_→_MMLU|0.704|0.296|0.342|0.398 [0.302, 0.482]|
|Qwen2.5-14B|Movies_→_TruthfulQA|0.389|0.611|0.581|0.566 [0.481, 0.649]|
|OLMo-3-7B|Math_→_MMLU|0.511|0.489|0.458|0.452 [0.323, 0.558]|
|OLMo-3-7B|Math_→_TruthfulQA|0.246|0.754|0.518|0.406 [0.317, 0.498]|
|OLMo-3-7B|Movies_→_MMLU|0.511|0.489|0.233|0.252 [0.103, 0.386]|
|OLMo-3-7B|Movies_→_TruthfulQA|0.246|0.754|0.193|-0.030 [-0.138, 0.077]|

**Table S22:** Counterbalanced target-side judgement diagnostics. _ρXY,Y X_ is the Spearman correlation between the two mapping-corrected log odds. HC agreement is computed where both mappings yield a high-confidence binary label. Retained counts all candidates outside the symmetric uncertain band; B/C and conflict questions describe the balanced conflict set.

|Model|Target|_ρXY,Y X_|HC agreement|Retained|B|C|Conflict_q_|
|---|---|---|---|---|---|---|---|
|Llama-3.1-8B|MMLU|0.920|95.7%|1,112|13|454|258|
|Llama-3.1-8B|TruthfulQA|0.937|98.9%|1,125|283|196|420|
|OLMo-3-7B|MMLU|0.917|99.6%|1,361|4|857|369|
|OLMo-3-7B|TruthfulQA|0.975|100.0%|1,259|102|318|372|
|Qwen2.5-14B|MMLU|0.941|98.1%|1,581|135|140|201|
|Qwen2.5-14B|TruthfulQA|0.963|96.5%|1,550|270|94|292|
|Qwen2.5-7B|MMLU|0.943|92.7%|1,573|260|32|280|
|Qwen2.5-7B|TruthfulQA|0.949|93.7%|1,529|506|82|535|

Across the 16 model–source–target rows, fixed-window ∆CB is positive in all cases (mean 0.383, range 0.176–0.682), with 12 raw question-cluster intervals above zero (Table S23). The four intervals crossing zero are exactly the two source directions evaluated in each of the low-B MMLU cells. After layer-wise adjustment for answer-option log-probability and token count, all 16 point estimates and intervals are above zero. Across individual layers, 464/560 (82.9%) estimates are positive and 282/560 (50.4%) intervals are above zero. This control supports target-side judgement-label robustness; the source directions were still constructed from the original elicitation.

**Table S23:** Fixed-window OOD transfer under counterbalanced X/Y judgement labels. Component AUCs and raw ∆CB use the complete balanced B/C sets. Residual ∆CB controls answer-option log-probability and token count separately at each layer. All intervals use 1,000 target-question-cluster bootstrap resamples.

|Model|Direction|_W_meta_→_SJ|_W_truth_→_OC|Raw∆_CB_|Residual∆_CB_|
|---|---|---|---|---|---|
|Qwen2.5-7B|Math_→_MMLU|0.590 [0.514, 0.661]|0.390 [0.322, 0.467]|0.200 [0.054, 0.332]|0.220 [0.065, 0.353]|
|Qwen2.5-7B|Math_→_TruthfulQA|0.575 [0.517, 0.631]|0.399 [0.343, 0.461]|0.176 [0.056, 0.284]|0.172 [0.053, 0.281]|
|Qwen2.5-7B|Movies_→_MMLU|0.642 [0.571, 0.715]|0.424 [0.339, 0.500]|0.218 [0.078, 0.373]|0.301 [0.152, 0.449]|
|Qwen2.5-7B|Movies_→_TruthfulQA|0.632 [0.586, 0.672]|0.388 [0.347, 0.432]|0.244 [0.157, 0.321]|0.234 [0.139, 0.318]|
|Llama-3.1-8B|Math_→_MMLU|0.692 [0.562, 0.808]|0.386 [0.220, 0.582]|0.305 [-0.015, 0.577]|0.378 [0.089, 0.628]|
|Llama-3.1-8B|Math_→_TruthfulQA|0.707 [0.664, 0.753]|0.311 [0.264, 0.355]|0.396 [0.311, 0.486]|0.335 [0.251, 0.414]|
|Llama-3.1-8B|Movies_→_MMLU|0.678 [0.502, 0.826]|0.356 [0.192, 0.541]|0.322 [-0.036, 0.633]|0.481 [0.169, 0.763]|
|Llama-3.1-8B|Movies_→_TruthfulQA|0.723 [0.676, 0.767]|0.288 [0.244, 0.335]|0.435 [0.342, 0.521]|0.375 [0.294, 0.459]|
|Qwen2.5-14B|Math_→_MMLU|0.599 [0.546, 0.654]|0.367 [0.316, 0.418]|0.231 [0.135, 0.336]|0.257 [0.167, 0.348]|
|Qwen2.5-14B|Math_→_TruthfulQA|0.768 [0.713, 0.819]|0.266 [0.214, 0.323]|0.502 [0.392, 0.599]|0.486 [0.376, 0.590]|
|Qwen2.5-14B|Movies_→_MMLU|0.649 [0.607, 0.691]|0.459 [0.409, 0.507]|0.190 [0.105, 0.278]|0.280 [0.193, 0.361]|
|Qwen2.5-14B|Movies_→_TruthfulQA|0.810 [0.765, 0.847]|0.201 [0.166, 0.243]|0.609 [0.524, 0.681]|0.590 [0.506, 0.665]|
|OLMo-3-7B|Math_→_MMLU|0.841 [0.504, 0.984]|0.239 [0.018, 0.601]|0.602 [-0.100, 0.963]|0.671 [0.249, 0.976]|
|OLMo-3-7B|Math_→_TruthfulQA|0.858 [0.820, 0.892]|0.176 [0.138, 0.216]|0.682 [0.606, 0.751]|0.629 [0.542, 0.699]|
|OLMo-3-7B|Movies_→_MMLU|0.788 [0.415, 0.986]|0.276 [0.092, 0.593]|0.513 [-0.178, 0.895]|0.584 [0.185, 0.894]|
|OLMo-3-7B|Movies_→_TruthfulQA|0.795 [0.741, 0.844]|0.291 [0.237, 0.350]|0.504 [0.395, 0.605]|0.429 [0.324, 0.525]|

### S11 Implementation and Reproducibility

#### S11.1 Models and activation artifacts

The analyses use four publicly available instruction-tuned language models:

- Qwen2.5-7B-Instruct (Qwen/Qwen2.5-7B-Instruct);

- Llama-3.1-8B-Instruct (meta-llama/Llama-3.1-8B-Instruct);

- Qwen2.5-14B-Instruct (Qwen/Qwen2.5-14B-Instruct); and

- OLMo-3-7B-Instruct (allenai/Olmo-3-7B-Instruct).

Activation extraction was performed on a single NVIDIA A800 80GB using bfloat16 forward computation and float16 activation storage, with a maximum sequence length of 4,096 tokens. The batch size was four for the 7–8B models in the main-domain extraction and two for Qwen2.5-14B. OOD activation extraction used a batch size of two.

**Table S24:** Main-domain activation arrays. Shapes are samples by transformer blocks by hidden width and apply to each sample-aligned artifact.

|Model|Blocks|Width|Math shape|Movies shape|
|---|---|---|---|---|
|Qwen2.5-7B|28|3,584|4614_×_28_×_3584|1824_×_28_×_3584|
|Llama-3.1-8B|32|4,096|3078_×_32_×_4096|5560_×_32_×_4096|
|Qwen2.5-14B|48|5,120|4740_×_48_×_5120|3320_×_48_×_5120|
|OLMo-3-7B|32|4,096|724_×_32_×_4096|1016_×_32_×_4096|

Each saved artifact contains metadata, a JSONL sample index, and one or more NPZ activation arrays. The sample index records the activation-row index, question identifier, response, reference answer, OC label, SJ label, factorial cell, _p_ judge, rendered question-plus-response token count, and prompt hash. Probe fitting, bootstrap inference, and figure generation are performed offline from these saved arrays and do not require additional model inference.

**Figure S4:** All-layer component AUCs under counterbalanced X/Y target judgement labels. Directions are fitted only on Math or Movies. Blue curves show _W_ meta _→_ SJ and brown-orange curves show _W_ truth _→_ OC; shading denotes 95% target-question cluster-bootstrap intervals. The broad intervals for Llama-3.1-8B and OLMo-3-7B on MMLU reflect conflict sets with only 13 and 4 B rows, respectively.

### S12 Interpretive Scope

The factorial contrasts identify response-level associations with the observed OC and SJ cells; neither OC nor SJ is experimentally manipulated. Confidence intervals for the primary fixed-window cross-domain analysis jointly incorporate variation in sourceand target-domain question samples, conditional on the realized pass-at-eight generations and the sampled correct–incorrect response pair for each eligible question. They do not integrate generation-level or within-question pair-selection variability. In contrast, the all-layer figures and several secondary controls condition on the fitted source direction and therefore quantify only target-sample uncertainty.

The OOD tasks use inserted forced-choice responses rather than free generation. They consequently test whether the fitted directions transfer to a controlled answer site, not whether the complete free-response generation pipeline replicates across task formats. In addition, SJ operationalizes correctness-directed self-evaluation under the specified elicitation procedure. It may contain variance associated with confidence, familiarity, response policy, or other processes beyond metacognitive accuracy.

Subject to these constraints, the principal pattern is observed across conflict-set ordering, grouped in-domain validation, cross-domain component AUCs, source-side within-question direction estimation, the all-cell target-question-fixed-effect analysis, token-count matching, null directions, alternative confidence thresholds, answer-likelihood controls, and two target tasks excluded from direction fitting.
