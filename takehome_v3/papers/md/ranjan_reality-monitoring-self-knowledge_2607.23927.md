# Reality Monitoring in Large Language Models: Self-Knowledge That Transforms with Conversation Memory

- **Authors:** Saurabh Ranjan, Konstantina Sokratous, Brian Odegaard
- **Date:** 2026-07-27
- **Source:** https://arxiv.org/abs/2607.23927
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `ranjan_reality-monitoring-self-knowledge_2607.23927`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

_Preprint — not peer reviewed_

Reality Monitoring in Large Language Models: Self-Knowledge That Transforms with Conversation Memory

Saurabh Ranjan^1* , Konstantina Sokratous^2 and Brian Odegaard^1

> 1*Department of Psychology, University of Florida, Gainesville, Florida, USA.

> 2University of Missouri, Missouri, USA.

*Corresponding author(s). E-mail(s): ranjan.saurabh@outlook.com;

## Abstract

A conversational AI that cannot tell its own output from what a user said will treat its own mistakes as user-provided facts. In humans, this capacity is called reality monitoring, and its failures are linked to hallucinations, delusions, and confabulation, yet whether LLMs possess it remains untested. Here we show, across two experiments and six LLMs, that source attribution depends on how conversational memory is structured: ceiling accuracy for self-generated content under minimal memory demands reverses to a fragile external-item advantage once episodic delay removes that shortcut. Feedback exposes two failures: in some models, internal and external judgments swap; in others, accuracy improves while confidence decouples from correctness, dissociations invisible to existing benchmarks. Across models, this pattern implicates active, not aggregate, parameter count. This suggests that as AI systems take on autonomous, multi-turn roles, evaluating what they know is not enough: tracking where that knowledge came from may matter equally.

**Keywords:** reality monitoring, large language models, AI self-awareness, hallucinations, metacognition, conversation memory

Large language models (LLMs) now achieve human-competitive performance across a broad range of language generation and reasoning tasks [1–6], prompting investigation of whether higher-order cognitive abilities such as introspection, self-modeling, and situational awareness emerge in these systems [7–16].

Whether LLMs can discriminate information that is _externally generated_ (presented by a user) from information that is _internally generated_ (produced by the model itself) has not been systematically examined using paradigms permitting direct human– LLM comparison or isolating conversational memory as an experimental variable (Figure 1). Yet LLMs increasingly operate in high-stakes agentic contexts, including clinical decision support [17], legal analysis [18], autonomous scientific reasoning [19], and military decision-making [20], where they both generate and receive information across extended multi-turn interactions. A model that cannot attribute information to its correct source may cite confabulations as user-provided facts, fail to update when corrected, or misreport its own claims’ epistemic authority: source-attribution failure is not a performance error but a safety failure.

**Fig. 1 A word-pair paradigm operationalizes reality monitoring in LLMs by contrasting externally supplied and self-generated second words.** (a) _External generation:_ the LLM receives a complete word pair (first word and second word) and must reproduce the second word verbatim. This simulates perception of an external stimulus: the word exists outside the model and requires accurate reproduction rather than generation. (b) _Internal generation:_ the LLM receives only the first word of a pair and must produce a novel second word. The generated word exists only within the model’s own output, a simulation of internally generated content. No constraint is placed on the choice of word beyond it being a single English word not previously used in the session. (c) _Reality monitoring:_ the model must determine whether the associated information was externally provided (external/perceived) or self-generated (internal/imagined).

In cognitive science, _reality monitoring_ (RM; Figure 1) is the capacity to determine the origin of a memory: whether its content derives from something externally perceived or something internally imagined [21, 22]. Human RM is imperfect but systematic: internally generated information tends to carry richer cognitive operations and fewer perceptual details, whereas externally generated information has more perceptual specificity [21–30], and on word-pair materials humans show systematic externalizing biases (i.e., often confusing imagined information as previously perceived) and sourcedependent metacognitive sensitivity [31]. Failures of RM have been associated with hallucinations and delusions in schizophrenia [32–38], delusional disorders [38–40], paranoia [41–44], and confabulation [45–48].

Despite its relevance, RM has not been systematically investigated in LLMs, although LLM hallucinations [49, 50] (outputs that do not correspond to verifiable external states) are conceptually adjacent to RM failures. In particular, _conversational memory_ , which determines which sources are even available for attribution at test, has never been isolated as an experimental variable in source attribution research (see Related Work).

Here, we adapt the RM framework from human cognitive psychology to assess source monitoring in LLMs. In a word-pair task, LLMs encode pairs whose second word is either externally provided (perceived) or internally generated (imagined), and then are asked to attribute the source of the associated second word as internal/imagined or external/perceived, enabling direct comparison with human RM [21, 22, 31]. Across two experiments, we assess conversational memory as our primary experimental variable of interest (Figure 2). Experiment 1 varies within-trial memory structure: Single-Turn, in which all subtasks occur within one conversational turn, vs. Trial-Chain, in which the same subtasks are distributed across sequential turns. Experiment 2 introduces an Episodic-Chain paradigm mirroring human episodic RM tasks: encoding trials are completed as a single block before a separate, delayed source-test phase, with set size (20 vs. 40 trials) and corrective feedback (present vs. absent) crossed to test whether AI self-knowledge, the capacity to know what the model itself generated, is stable across memory loads and whether feedback can recalibrate it. We evaluate six LLMs spanning Gemma 3 (12B and 27B, with and without quantization-aware training), Llama 3.3 (70B), and a sparse mixture-of-experts model (Llama 4 16 _×_ 17B), using the same paradigm and materials as a validated human RM dataset.

## 1 Related Work

Human RM research has established that source attribution accuracy and metacognitive calibration are systematically modulated by stimulus relatedness, source type, and episodic delay [21, 22, 31]; the human behavioral profile for a similar word-pair stimulus set [31] provides a qualitative baseline here. Source identification in AI systems has largely been addressed through external mechanisms such as watermarking [49, 50] rather than probing intrinsic self-attribution capacity. Theoretical work links higher-order representation, a system’s capacity to represent its own inner representational states, to metacognition and self-knowledge, and argues that whether LLMs form such representations remains empirically unresolved [51]; empirical work on LLM introspection, situational awareness, and self-modeling raises the same concern [9, 10, 12, 13], but none has used formal, behaviorally grounded RM paradigms or treated conversational memory as a manipulated variable. The most closely related benchmarks test metacognitive calibration on knowledge and reasoning tasks [52, 53], where frontier LLMs show above-chance but limited-resolution, context-dependent ability [54]; work evaluating episodic memory with sequence-order recall tasks [55] has likewise not examined source attribution or metacognitive sensitivity. None of this work uses source-memory paradigms permitting direct human–LLM comparison, nor treats conversational memory architecture as a controlled variable.

## 2 Results

Experiment 1 yielded ceiling performance for internal items but variable external-item accuracy; Experiment 2 introduced episodic delay and reversed the source advantage (Figures 3–6).

**Fig. 2 Three conversational memory architectures vary how much prior context a model must maintain across turns to complete source attribution.** (a) _Single-Turn:_ all four subtasks — word-pair completion, semantic relatedness rating (0–100 scale), source judgment (internal or external), and confidence rating (1–6) — are requested within a single LLM interaction. The model encodes and attributes the source of each trial without carrying information across separate turns. (b) _Trial-Chain:_ the same four subtasks are distributed across four sequential conversational turns, each extending the running conversation history. Source attribution occurs in the isolated interaction with the LLM, after the completion and relatedness rating have already been provided in past interaction. This tests whether decomposing within-turn cognitive load improves source classification accuracy, while keeping the total number of trials identical to Single-Turn. The Single-Turn and Trial-Chain conditions are examined in Experiment 1. (c) _Episodic-Chain:_ all encoding trials (5 or 10 word pairs per source type, yielding set sizes of 20 or 40 total trials including test trials) are completed as a single block before a separate delayed source-test phase begins. Source attribution is probed only after the full encoding sequence, introducing genuine episodic delay analogous to the between-phase interval in human RM experiments. Corrective feedback (present vs. absent after each source response) is crossed with set size to yield four conditions (a 2 _×_ 2 feedback-by-set-size design; source type, internal vs. external, is analyzed as a separate within-trial factor, not part of this crossing). Episodic-Chain with set size and feedback manipulations is examined in Experiment 2.

**Fig. 3 Internal source identification reaches ceiling across all models; external accuracy varies with reading hallucination rates and model scale.** Performance metrics for single-trial reality monitoring across six LLMs (Gemma3:12b, Gemma3:12b-QAT, Gemma3:27b, Gemma3:27bQAT, Llama3.3:70b, Llama4:16x17b) and two memory implementations in Experiment 1. (a–b) Mean source attribution accuracy for the Single-Turn (a) and Trial-Chain (b) conditions, shown separately for internal (red, hatched) and external (blue, solid) trials; error bars _±_ 1 SEM across trials within each model–condition cell. (c) Mean reading hallucination rates on external trials only ( _N_ = 1 _,_ 728 external trials; missing bars indicate zero hallucinations observed in that cell). (d) Metacognitive sensitivity (Goodman–Kruskal _γ_ , Fisher _Z_ -transformed) relating expressed confidence to trial-level accuracy, computed once per model–condition cell from all trials in that cell pooled (not an average of per-trial estimates; no error bars defined). Positive _γ_ indicates above-chance calibration. Missing bars indicate _γ_ was undefined because ceiling accuracy left zero variance in the correctness variable.

### 2.1 Experiment 1: Single-Trial Reality Monitoring

#### 2.1.1 Accuracy: Ceiling-Level Performance for Internal Items, Variable Performance for External Items

Across all models and both implementations (Figure 3), internally generated (imagined) items were identified with perfect accuracy (100%). This ceiling reflected direct withinturn retrieval: the generated word remained in the active context window at the time of judgment, making imagined-item identification trivially accessible rather than requiring inferential source monitoring. Externally generated (perceived) items, by contrast, showed substantial variability across models and implementations.

A generalized linear model over perceived trials ( _N_ = 1 _,_ 728; predictors: model, implementation, response-option order, relatedness, reading hallucination; pseudo _R_^2 = _._ 474; Methods) identified reading hallucination as the strongest predictor of source attribution failure ( _b_ = _−_ 3 _._ 84, _p < ._ 001): a model that failed to reproduce the externally presented word was substantially more likely to misclassify its source, establishing reading hallucinations as a primary bottleneck for RM on externally generated trials. Conditional on no reading hallucination, predicted accuracy was near ceiling (90.9% for Gemma3:12b, the lowest-scoring model; 95.2%–99.7% for the remaining reliably estimated models), indicating that source monitoring itself remains largely intact once this upstream encoding failure is excluded (Gemma3:27b-QAT’s coefficient is unreliable: quasi-complete separation; Methods).

Model scale also mattered: larger models generally achieved higher accuracy (Llama3.3:70b: 94.4% Single-Turn, 100% Trial-Chain), while smaller models could fall to chance (Gemma3:12b: 47.9% Trial-Chain). Trial-Chain outperformed Single-Turn for three of six models, tied at ceiling for one, and reversed for two (Gemma3:12b, Gemma3:27b). For Gemma3:27b this reflected a hallucination rate that rose under Trial-Chain (0% to 14.6%); for Gemma3:12b, hallucination rates fell sharply (63.2% to 10.4%) yet accuracy still declined, confirming a genuine shift in source-judgment ability independent of encoding fidelity (conditional on correct word reproduction, accuracy fell from 100%, _n_ = 53, in Single-Turn to 53.5%, _n_ = 129, in Trial-Chain, barely above chance). Reduced cognitive load improves RM for some architectures but not others.

#### 2.1.2 Reading Hallucinations

A GLM with the same predictors ( _N_ = 1 _,_ 728 perceived trials) showed that Trial-Chain significantly reduced hallucination rates relative to Single-Turn ( _b_ = _−_ 1 _._ 80, _p < ._ 001): decomposing the task across conversational turns improves fidelity to externally supplied inputs. More related word pairs produced slightly fewer hallucinations ( _b_ = _−_ 0 _._ 006, _p_ = _._ 036), and response-option order reached significance (odds ratio [OR] = 1 _._ 40, _p_ = _._ 021). All four larger models hallucinated significantly less than the Gemma3:12b baseline (Gemma3:27b-QAT: _b_ = _−_ 4 _._ 30; Gemma3:27b: _b_ = _−_ 2 _._ 27; Llama3.3:70b: _b_ = _−_ 1 _._ 75; Llama4:16x17b: _b_ = _−_ 1 _._ 49; all _p < ._ 001), whereas the 12B quantization-aware variant did not differ ( _b_ = _−_ 0 _._ 22, _p_ = _._ 24).

#### 2.1.3 Metacognitive Sensitivity: Single-Turn Advantage

Metacognitive sensitivity, the extent to which an agent’s confidence predicts its accuracy [56], was estimated per model and implementation using the Goodman–Kruskal _γ_ coefficient (Figure 3d). Single-Turn generally yielded stronger positive confidence– accuracy associations than Trial-Chain: Gemma3:12b, for example, showed _γ_ = _._ 77 in Single-Turn but _γ_ = _−._ 28 in Trial-Chain, and several other models showed similar or uncomputable reversals (the latter when ceiling or floor accuracy left no variance in correctness). Reducing cognitive load for task performance therefore does not necessarily preserve confidence calibration.

#### 2.1.4 Confidence in Single-Trial Reality Monitoring

Confidence ratings in Experiment 1 were modeled using ordered probit regression. Perceived-source trials elicited substantially higher confidence than imagined-source trials ( _b_ = 1 _._ 80, _p < ._ 001), reading hallucinations reduced confidence ( _b_ = _−_ 0 _._ 67, _p < ._ 001), indicating greater uncertainty when reproduction of the external word failed, and trial-level accuracy independently increased confidence ( _b_ = 1 _._ 05, _p < ._ 001). TrialChain was associated with higher confidence than Single-Turn ( _b_ = 1 _._ 28, _p < ._ 001), likely reflecting the more structured context window.

### 2.2 Experiment 2: Episodic Reality Monitoring

Experiment 2 introduced delayed source testing across an extended conversational history, crossed with set size (20 vs. 40 trials) and feedback (absent vs. present).

**Fig. 4 Episodic delay reverses the source advantage for internally-generated words, and feedback selectively recalibrates internal-item accuracy at the cost of increased reading hallucinations.** Performance in the Episodic-Chain paradigm (Experiment 2). (a–d) Mean source attribution accuracy for internal (red, hatched) and external (blue, solid) trials across four conditions: (a) set size 20, no feedback; (b) set size 20, feedback; (c) set size 40, no feedback; (d) set size 40, feedback. Error bars: _±_ 1 SEM across trace IDs, each trace treated as an independent participant ( _N_ = 200 per cell); Gemma3:12b’s response inversion under feedback is detailed in Figure 6. (e–f) Mean reading hallucination rates on external trials for set size 20 (e) and set size 40 (f); bars show no-feedback (orange) vs. feedback (dark blue).

#### 2.2.1 Accuracy: Reduced Performance and Source Reversal

In Experiment 2, no model reached ceiling (Figure 4). Mean accuracy (trace-level, _N_ = 4 _,_ 800 traces per source) was higher for perceived items ( _M_ = 66 _._ 42%, range across conditions: 27.80%–97.20%) than for imagined items ( _M_ = 47 _._ 57%, range: 3.00%–89.60%), a source reversal relative to Experiment 1, where imagined items had been easier to attribute. The reversal was most pronounced without feedback (perceived _M_ = 76 _._ 34% vs. imagined _M_ = 38 _._ 46%); feedback nearly eliminated the gap ( _M_ = 56 _._ 50% vs. _M_ = 56 _._ 68%), with heterogeneous effects on individual models, including systematic response inversion in Gemma3:12b (Signal Detection Analysis, below).

A generalized linear mixed model (GLMM; binomial family, logit link; random intercept for trace; _N_ = 72 _,_ 000 test-phase observations; Methods) confirmed the reversal: perceived items were attributed more accurately than imagined items ( _χ_^2 (1) = 2594 _._ 9, _p < ._ 001; OR = 2 _._ 57, 95% CI [2 _._ 48 _,_ 2 _._ 66]). The source _×_ feedback interaction was pronounced ( _χ_^2 (1) = 2734 _._ 1, _p < ._ 001): feedback differentially improved imagineditem accuracy while simultaneously increasing reading hallucinations on perceived trials, attenuating the source advantage. Unlike Experiment 1, restricting to hallucination-free perceived trials did not restore near-ceiling accuracy (no feedback: 79 _._ 1% vs. 67 _._ 9% with hallucination; feedback: 53 _._ 4% vs. 58 _._ 8%): episodic-delay failures are not reducible to encoding errors, so source monitoring itself degrades under memory load. Main effects of set size and feedback and their interaction were also significant (all _p < ._ 001), as was the source _×_ set size interaction ( _p_ = _._ 010), though the latter was negligible in practical terms (OR = 0 _._ 91).

#### 2.2.2 Reading Hallucinations

A GLMM predicting reading hallucinations ( _N_ = 36 _,_ 000 perceived trials; binomial family, logit link) showed that hallucinations increased with larger set sizes (OR = 1 _._ 56, _p < ._ 001) and feedback exposure (OR = 3 _._ 72, _p < ._ 001), amplified in larger sets under feedback (Type-II Wald _χ_^2 (1) = 19 _._ 8, _p < ._ 001). Model differences were pronounced: relative to baseline, Llama3.3:70b (OR = 0 _._ 002) and Llama4:16x17b (OR = 0 _._ 15, both _p < ._ 001) showed dramatically suppressed hallucination rates, whereas Gemma3:12bQAT hallucinated at higher odds than its non-QAT counterpart (OR = 2 _._ 64, _p < ._ 001), reversing the Experiment 1 pattern, in which no quantization-aware variant exceeded baseline. All other coefficients are in the Supplementary Information.

#### 2.2.3 Metacognitive Sensitivity: Degradation Under Memory Load and Feedback

Metacognitive sensitivity in episodic RM (Figure 5) was analyzed using a linear mixed model (LMM) with random intercepts for trace ( _N_ = 4 _,_ 410 simulations with computable _γ_ values; approximately 5,190 of 9,600 possible observations were excluded due to ceiling or floor accuracy or invariant confidence; exclusion pattern detailed in Methods).

Perceived items showed substantially higher metacognitive sensitivity than imagined items (pairwise contrast perceived _−_ imagined = 4 _._ 28, _SE_ = 0 _._ 12, _p < ._ 001), and feedback _decreased_ sensitivity ( _b_ = _−_ 3 _._ 22, _F_ (1 _, ∞_ ) = 30 _._ 4, _p < ._ 001), amplified at the larger set size (set size _×_ feedback: _F_ (1 _, ∞_ ) = 7 _._ 80, _p_ = _._ 005), consistent with feedback and extended context jointly eroding confidence–accuracy calibration; set size otherwise improved calibration only for perceived items (source _×_ set size: _F_ (1 _, ∞_ ) = 6 _._ 41, _p_ = _._ 011). Metacognitive performance was highly model-dependent, with several models showing substantially lower _γ_ under feedback. One model, Gemma3:27b-QAT, contributed no computable imagined-source _γ_ without feedback: every trace’s internalsource accuracy was invariant, uniformly incorrect in 376 of 400 traces (a strong external-response bias) and uniformly correct in the remainder, leaving no within-trace variance to compute _γ_ (missing points in Figure 5). To identify what drives confidence when accuracy does not, we modeled raw confidence directly.

**Fig. 5 Metacognitive sensitivity is consistently higher for external than internal items and is substantially reduced by corrective feedback and larger memory loads.** Fisher _Z_ - transformed Goodman–Kruskal _γ_ , a correlation between expressed confidence and trial-level accuracy, measuring metacognitive sensitivity separately for internal and external items across Episodic-Chain conditions in Experiment 2. Positive _γ_ indicates above-chance calibration; _γ_ = 0 indicates expressed confidence is unrelated to whether the source judgment was correct. Panels are arranged in a 2 (feedback) _×_ 2 (set size) grid: top row = no feedback; bottom row = feedback; left column = set size 20; right column = set size 40. Within each panel, orange circles = internal items; blue squares = external items; vertical grey lines connect the two source estimates for each model, making the source dissociation visible across conditions. Error bars: _±_ 1 SEM across trace IDs, each trace treated as one participant. Corrective feedback markedly reduced external-item _γ_ in most models, with the sharpest collapses in the Gemma family (largest: Gemma3:27b-QAT); Llama3.3:70b is the exception, with external-item _γ_ increasing under feedback even as its all-trials Type 2 AUC falls below chance (Figure 6), the latter reflecting cross-source confidence differences rather than within-source calibration. Gemma3:27b-QAT internal-source _γ_ is absent in no-feedback panels (dots omitted); see Results for explanation.

#### 2.2.4 Confidence: Driven by Architecture and Context, Not Trial-Level Accuracy

Ordinal confidence ratings (1–6) were analyzed using a Cumulative Link Model [57] (CLM; logit link; _N_ = 72 _,_ 000; Methods). Model identity was by far the dominant predictor of confidence ( _χ_^2 (5) = 17 _,_ 620 _._ 6, _p < ._ 001), with odds ratios relative to the Gemma3:12b baseline ranging from 0.44 (Llama4:16x17b) to 17.85 (Gemma3:27bQAT), a 40-fold spread from architecture alone. Imagined items received slightly higher confidence than perceived items (OR = 1 _._ 10, _p_ = _._ 003); feedback (OR = 0 _._ 86, _p < ._ 001)

and larger set size (OR = 0 _._ 94, _p_ = _._ 048) reduced confidence, and higher relatedness slightly increased it (OR = 1 _._ 01/unit, _p < ._ 001). Three interactions qualified these effects: feedback disproportionately reduced confidence for imagined items (source _×_ feedback: OR = 0 _._ 60, _p < ._ 001) and at set size 40 (setsize _×_ feedback: OR = 0 _._ 56, _p < ._ 001), whereas the imagined-item advantage _widened_ at set size 40 (source _×_ setsize: OR = 1 _._ 15, _p < ._ 001). Full coefficients, model comparisons, and post-hoc contrasts are in the Supplementary Information.

Critically, trial-level accuracy did not predict confidence after controlling for all other factors ( _b_ = 0 _._ 03, _p_ = _._ 063), nor did reading hallucinations ( _p_ = _._ 985), unlike Experiment 1, where accuracy significantly predicted confidence. A trace-level robustness test confirmed this null (paired within-trace difference = +0 _._ 001 confidence points, _t_ (4518) = 0 _._ 11, _p_ = _._ 91; Methods) and revealed that it conceals two small oppositesigned effects: confidence weakly tracked correctness without feedback (+0 _._ 06) but reversed under feedback ( _−_ 0 _._ 06; both _p < ._ 001). Episodic delay thus specifically erodes the accuracy–confidence link preserved under single-trial memory, and feedback inverts it, consistent with the near- and below-chance Type 2 AUC values below (Figure 6): expressed confidence here reflects model architecture and condition, not whether the individual source judgment was correct.

#### 2.2.5 Cumulative Accuracy: Heterogeneous Trial-Level Trajectories

Cumulative accuracy trajectories across episodic test trials (Supplementary Figure S6) revealed substantial trial-level heterogeneity not captured by mean accuracy alone: trajectory shape depended jointly on model identity and set size rather than model scale, with no single shape dominating either feedback condition, indicating model-specific constraints on memory and context management rather than a static capability limit.

#### 2.2.6 Signal Detection Analysis: Source Discrimination vs. Metacognitive Calibration Dissociation

Signal detection analyses corroborate and extend the behavioral findings. In Experiment 1, ceiling imagined-source accuracy fixed hit rates near 1.0, so _d_^(_′_) indexes response conservatism rather than true discrimination sensitivity, and negative criterion values ( _c <_ 0) confirm a liberal bias toward internal responses; perceived-trial Type 2 AUC nonetheless ranged from chance (0.49) to high (0.96–0.97) across models and implementations (Supplementary Figures S7–S8).

In Experiment 2 (Figure 6), SDT parameters were computed per trace (200 per model _×_ condition) and summarized as mean _±_ SEM. Without feedback most models show _d_^(_′_) _≈_ 0, confirming near-chance source discrimination despite above-chance mean accuracy; Llama3.3:70b is a notable exception ( _d_^(¯)^(_′_) = 2 _._ 01 _±_ 0 _._ 11 at set size 20, 1 _._ 62 _±_ 0 _._ 11 at set size 40). Most striking is a Type 1/Type 2 dissociation under feedback in Llama3.3:70b at set size 20: _d_^(_′_) improves to 2 _._ 28 _±_ 0 _._ 04, yet Type 2 AUC, indexing metacognitive accuracy [58], declines below chance (0 _._ 34 _±_ 0 _._ 02), so behavioral improvement under feedback is not accompanied by improved calibration. Gemma3:12b shows the inverse pathology: _d_^(¯)^(_′_) turns negative under feedback ( _−_ 0 _._ 95 _±_ 0 _._ 02 at set size 20), reflecting systematic response inversion rather than degraded discrimination, a qualitatively different failure mode.

**Fig. 6 Corrective feedback dissociates source discrimination from metacognitive calibration and produces qualitatively distinct failure modes by model scale.** Type 1 (source discrimination) and Type 2 (metacognitive) signal detection measures across the four Episodic-Chain conditions. _d_^(_′_) = standardized source-discrimination sensitivity; criterion _c_ = response bias (negative values indicate a liberal bias toward the internally generated response); Type 2 AUC = area under the receiver operating characteristic curve discriminating correct from incorrect source judgments, a model-free measure of metacognitive accuracy that does not depend on a single criterion placement. Two rows _×_ three columns: rows = feedback condition (top = no feedback; bottom = feedback); columns = _d_^(_′_) , criterion _c_ , and Type 2 AUC. Solid blue bars = set size 20; hatched sky-blue bars = set size 40. Error bars: _±_ 1 SEM across _N_ = 200 traces per model–condition cell. See Results for the Llama3.3:70b and Gemma3:12b dissociations under feedback. Supplementary Figures S7–S8 report Experiment 1 SDT; Figure S9 reports Experiment 2 ROC curves.

## 3 Discussion

Source attribution in LLMs is not an invariant architectural property: it is shaped by the structure of the conversational context. That single finding accounts for all main results: near-ceiling performance when context-window retrieval was available, reversal to an external-item advantage once episodic delay removed that shortcut, and decoupling of accuracy from confidence under corrective feedback. Scale did not drive this: Llama4:16x17b, the largest model in the sample, performed comparably to smaller models across several measures (Figures 4 and 6), implicating individual expert-network size within the mixture-of-experts architecture over aggregate parameter count.

Accuracy and metacognitive calibration split in ways that depend on architecture and condition, not overall capability. In Experiment 1, calibration held at the aggregate level in Single-Turn but broke down across implementations: the same model could show positive calibration in Single-Turn and reversed or undefined sensitivity in Trial-Chain. Corrective feedback widened this split: Llama3.3:70b gained accuracy while losing metacognitive sensitivity below chance; Gemma3:12b inverted its source-discrimination mapping entirely. Feedback does not scale reality monitoring upward; it restructures how source evidence maps to expressed confidence, direction depending on architecture.

A separate failure operates before source attribution begins. Reading hallucinations, failures to reproduce the externally provided word, are a mechanistically distinct upstream bottleneck that standard human RM paradigms do not operationalize, since stimulus encoding is experimenter- rather than participant-controlled. In Experiment 1, reading hallucinations were the strongest predictor of source misattribution: when a model substitutes its own word for the presented input, the downstream source judgment is ill-posed, and what appears as a source-monitoring deficit is partly an encoding deficit that must be separated out to draw valid inferences. Models were not entirely blind to this failure, however: reading hallucinations independently reduced expressed confidence in Experiment 1, so some metacognitive signal tracking encoding fidelity survives. Trial-Chain also reduced hallucinations for five of six models (the exception, Gemma3:27b, rose from 0% to 14.6% while its quantization-aware variant improved, a pattern absent in the 12B pair). In Experiment 2, reading hallucinations instead escalated with feedback and larger set sizes and no longer predicted confidence: a model that confidently misattributes a source it never correctly encoded gives no signal to prompt correction, a failure requiring external monitoring to detect.

These results extend Johnson and Raye’s source monitoring framework [21, 22] to artificial cognitive agents. The framework holds that source judgments rest on qualitative differences in attribute profiles: externally generated items carry richer perceptual and contextual features; internally generated items carry more cognitive operations and semantic associations. Consistent with this account, perceived items were attributed substantially more accurately once episodic delay removed the within-turn retrieval shortcut, driving ceiling performance on imagined items in Experiment 1, exactly what the framework predicts once perceptual and contextual attributes must carry the diagnostic load across a temporal gap. This advantage was largest without feedback and narrowed under it, consistent with corrective signals recalibrating imagined-item attribution rather than shifting only a response criterion.

The human comparison is informative in both directions. Human reality-monitoring studies using closely related word-pair tasks consistently report accuracy well above chance but far from ceiling, with systematic externalizing biases and imperfect source discrimination [23, 31]. Several LLMs in Experiment 1 reached or approached ceiling, exceeding the source-monitoring precision typically observed in these studies, but fell substantially below that level in the Episodic-Chain (a qualitative comparison; the human studies differ in design and trial counts). The metacognitive pattern also replicates a human finding: expressed certainty was equally high for accurate and confabulated source attributions on imagined items [31], suggesting that confidence– accuracy decoupling under memory load is a broad feature of source attribution systems.

Feedback substantially increased reading hallucinations, most strongly in larger conversational contexts, meaning that feedback loops in LLM-mediated systems may amplify the hallucinations they aim to reduce; this extends established findings that

LLM performance degrades with context length [59] to encoding fidelity under naturalistic multi-turn memory demands. This is directly relevant to agentic deployment, where high memory load and corrective feedback co-occur by design: single-turn accuracy is not a reliable proxy for multi-turn performance, models do not reliably self-correct once off track, and encoding fidelity warrants monitoring as a distinct safety metric alongside surface accuracy.

The _d_^(_′_) /Type 2 AUC dissociation (source-discrimination accuracy improving while metacognitive sensitivity collapses) raises a direct concern about using expressed confidence as a trustworthiness signal: when a model grows more accurate but its confidence becomes less diagnostic of its own correctness [56], downstream systems relying on expressed certainty are misled. Gemma3:12b’s response inversion adds a further caution: a model whose source-attribution decisions are systematically reversed by corrective signals cannot be treated as providing independent evidence about source origins, regardless of confidence. Both patterns are compatible with the competingbiases account of Kumaran and colleagues [60], in which LLM confidence reflects a choice-supportive bias for self-generated answers and a hypersensitivity to contradictory feedback, consistent with Gemma3:12b’s inversion taken to an extreme. A parallel comes from human calibration research: people are typically overconfident and improve with practice only somewhat, unlike weather forecasters, who check daily outcomes and are exceptionally well calibrated [61]; the models here received only a right/wrong label, uninformative about why an answer was wrong, which may explain why feedback shifted confidence without sharpening it. LLM confidence should not be treated as a calibrated probability estimate in high-stakes settings without independent validation, consistent with broader evidence that LLM miscalibration produces overconfident errors [62, 63]. Existing calibration benchmarks, which assess whether confidence tracks accuracy on knowledge and reasoning tasks [52–54], cannot detect this failure, since source memory imposes an additional constraint beyond answer accuracy: which source an answer came from. Deployment evaluations for multi-turn and agentic contexts should therefore include source-memory assessments, since minimal-load performance overestimates real-world capacity and scale alone does not predict self-knowledge under memory load.

This study has four main limitations. First, the word-pair paradigm, adapted from human reality-monitoring research for direct comparison [21, 22, 31], is deliberately minimal; generalization to richer deployment sources (prior context turns, retrieved documents, tool outputs) and to less related or factual content remains open. Relatedness had at most a negligible effect on accuracy (Experiment 2: OR = 1 _._ 01/unit; Supplementary Information), but its upstream role reversed the human pattern: in human memory, relatedness manufactures false memories, since related completions come to mind effortlessly, leave sparse process records, and are externalized at rates rising with the number of related items [22, 31, 64, 65]; here it instead aligned inputs with the model’s own distributional expectations, protecting verbatim reproduction (fewer reading hallucinations, Experiment 1) and fluency-inflating confidence independent of correctness (Experiment 2) [66]: a source-confusion risk in humans but a copy-fidelity aid in LLMs, directly testable by manipulating relatedness. Second, the six models tested span two families and two size classes with and without quantization-aware training, but not the full range of current architectures; all were run as instruction-tuned variants via Ollama, so tuning and architectural effects are not separable, and between-model estimates are specific to these six. Whether expert-network size, rather than aggregate parameter count, is the operative scaling variable for source attribution is directly testable across a broader range of mixture-of-experts models. Third, all analyses are behavioral; the mechanistic basis of these failure modes, why feedback inverts source-response mappings in smaller models and why confidence decouples from accuracy under memory load, requires interpretability methods applied at the activation level [67–69], for which these behavioral effects provide deployment-relevant targets [70– 72]. Fourth, models were evaluated at a single decoding setting (temperature fixed at 0, other parameters at framework defaults) and a single feedback form; whether sampling temperature, presentation order [73], or alternative feedback modalities (delayed, partial, confidence-targeted) modulate these effects is left to future work.

## 4 Conclusion

Source attribution in LLMs is not an invariant property of a model but a function of how conversational memory is structured: near-ceiling accuracy under single-trial demands, a reversal to an external-item advantage under episodic delay, and a decoupling of accuracy from confidence under corrective feedback. This pattern implicated individual expert-network size within mixture-of-experts architectures over aggregate parameter count, suggesting scale alone is the wrong lever for improving self-knowledge in deployed systems. As AI systems shift from tools to autonomous agents operating across extended interactions, evaluating what a model knows is not enough; whether it can track where that knowledge came from is equally essential, and reality monitoring offers a direct, model-comparable behavioral test of that capacity.

## 5 Methods

### 5.1 Large Language Models

Six models of varying sizes were used from the Gemma 3 and Llama families, all accessed as instruction-tuned variants via the Ollama inference framework. Model tags throughout (e.g., Gemma3:12b) follow Ollama’s naming convention, where the suffix after the colon denotes parameter count and variant. Models with 12 (Gemma3:12b) and 27 billion (Gemma3:27b) parameters, including quantization-aware training variants (Gemma3:12b-QAT, Gemma3:27b-QAT), were used from the Gemma 3 family. From the Llama family, we used Llama 3.3 (70B; Llama3.3:70b) and Llama 4 16 _×_ 17B (Llama4:16x17b), a sparse mixture-of-experts model. All models were run locally on a high-performance computing cluster. Decoding temperature was fixed at 0 (greedy decoding, minimizing sampling variability across simulations); all other inference parameters (top-p, context length) were left at Ollama defaults for each model version.

### 5.2 Stimuli

Word-pair stimuli were drawn from a prior human RM paradigm [31] that sampled semantically related word pairs from 36 Deese-Roediger-McDermott false-memory lists. That prior study established the human behavioral profile for this stimulus set: participants showed externalizing biases (imagined items misattributed as perceived) and source-dependent metacognitive sensitivity, with calibration strongest for perceived items. These shared materials enable direct human–LLM comparison. Word pairs were matched for relatedness and presented in randomized order. Each second word was either externally provided (perceived source) or generated by the model itself (imagined source).

### 5.3 LLM Reality Monitoring Task

All experimental interactions were administered using PsychScanner (version 0.1), a purpose-built open-source Python package for running structured behavioral experiments with large language models (source code available at [BLINDED]). Models received task instructions specifying the RM procedure (word-pair source attribution and confidence rating); no persona, role, or character framing was included beyond the task description itself. Trial instructions and stimuli were serialized in JSON, a format that is both human- and machine-readable and is well-represented in LLM pre-training corpora, minimizing the risk that response format itself, rather than source-attribution capacity, drives model output. Responses were parsed from structured JSON output; trials that could not be parsed were logged as missing and excluded from analysis. To control for positional biases in JSON-structured prompts, the order of response options in the source-attribution question was counterbalanced: the default condition presented the _external_ option first ( `["external","internal"]` ), and the counterbalanced condition reversed this to _internal_ first ( `["internal","external"]` ). This factor is included as a covariate ( `order c` ; dummy-coded: 0 = external-first [reference], 1 = internal-first) in all models. We implemented three distinct RM paradigms across two experiments, as described below.

#### 5.3.1 Experiment 1: Single-Trial Reality Monitoring

In the single-trial RM task, the LLM completed each task independently, with a fresh context window for each trial (or accumulated context in the Trial-Chain condition). For each trial, a word pair was given to the LLM, which could be complete (perceived: both words provided) or incomplete (imagined: first word provided, model generates second word).

In the _Single-Turn_ condition (Figure 2a), the LLM received all prompt instructions tailored to the RM task as a System Message, and each trial was handled in a single interaction: the model completed the task (word-pair completion, relatedness rating, source monitoring, confidence rating) in one turn.

In the _Trial-Chain_ condition (Figure 2b), the LLM received sequential instructions across multiple turns while the trial history was retained in the context window. Each subtask (completion, relatedness, source monitoring, confidence) was issued as a separate turn.

A 2 (Source: Imagined vs. Perceived) _×_ 2 (Memory Context: Single-Turn vs. TrialChain) _×_ 6 (Language Model) design was used for Experiment 1. Each cell comprised 144 trials (72 per source type), for a total of 3,456 observations.

#### 5.3.2 Experiment 2: Episodic Reality Monitoring

Experiment 2 extended this design with an episodic chain. Reality monitoring trials were separated into an encoding phase (word-pair completion and relatedness rating) and a delayed testing phase (source attribution and confidence) over multiple trials within the same conversational context.

Set size was manipulated to 20 and 40 interactions with LLMs, with equal numbers across phases and sources (set size 20: 10 encoding + 10 testing; set size 40: 20 encoding + 20 testing). Corrective feedback was either present or absent throughout the episodic chain. When present, feedback had a uniform three-component structure across both phases, delivered as a prefixed message at the start of the next trial’s stimulus. During encoding, the three components were: (1) a response-accuracy signal indicating, for _imagined_ trials, whether the generated word was novel (distinct from the first word in the pair and from any word produced in prior trials) or, for _perceived_ trials, whether the externally provided word was correctly reproduced; (2) a rating-validity signal confirming that the relatedness rating fell within the required 0–100% range; and (3) an overall trial-accuracy verdict, marked correct only if both preceding components were satisfied. During testing, the analogous three components were: (1) a sourceidentification signal indicating whether the model’s origin attribution ( _imagined_ vs. _perceived_ ) was correct; (2) a rating-validity signal confirming that the confidence rating fell within the required 1–6 range; and (3) an overall accuracy verdict, correct only if both preceding components were satisfied. This design ensures that encoding-phase feedback targets generation or reproduction fidelity depending on source type, while testing-phase feedback targets source monitoring accuracy, keeping the two processes distinct within the manipulation. Each complete encoding-to-test sequence constituted a single _simulation trace_ (hereafter _trace_ ), the unit of random-effects clustering. This yielded a 2 (Set Size) _×_ 2 (Feedback) _×_ 6 (Model) _×_ 2 (Source) design with 4,800 total traces; source was a within-trace factor (both imagined and perceived trials occurred within each trace), yielding 200 traces per design cell.

### 5.4 Experiment 1: Data Analysis

All statistical analyses for Experiment 1 were conducted in Python (version 3.11) using `statsmodels` . The Experiment 1 dataset contained 3,456 trials.

Trial compliance (binary) and recognition accuracy (binary) were modeled using separate logistic regressions (binomial family, logit link; MLE) on perceived trials only ( _N_ = 1 _,_ 728); standard errors are model-based (Fisher information). One model (Gemma3:27b-QAT) exhibited quasi-complete separation in the accuracy GLM (near-perfect classification on perceived trials); its coefficient is reported but flagged as unreliable and is not interpreted. Relatedness ratings (continuous) were analyzed using OLS regression with heteroskedasticity-consistent (HC3) standard errors. Type-II likelihood-ratio tests (LRT) assessed significance of each predictor. Confidence ratings

(ordinal) were analyzed using ordered probit regression. Metacognitive sensitivity was operationalized as Goodman–Kruskal _γ_ with Fisher _Z_ transformation ( _γz_ ), computed per model per memory condition. Specific _γ_ values cited in text are raw (untransformed); Figure 3d displays Fisher _Z_ -transformed _γz_ . Given the small sample ( _n_ = 6 models), only the memory factor could be tested; formal tests are therefore exploratory. No trials were excluded due to parse failures in Experiment 1 or Experiment 2; responses were parsed successfully for all 3,456 and 72,000 trials, respectively.

To confirm that the primary inferences do not depend on fixed-effects treatment of model identity, supplementary sensitivity analyses re-estimated the reading hallucination, recognition accuracy, and relatedness rating models with model as a random intercept [ `(1 | model)` ] instead of a fixed categorical predictor. All primary inferences replicated: the TrialChain advantage in reading hallucinations ( _z_ = _−_ 10 _._ 9, _p < ._ 001), the detrimental effect of reading hallucinations on accuracy ( _z_ = _−_ 15 _._ 8, _p < ._ 001), and the source advantage in relatedness ratings ( _t_ = 17 _._ 1, _p < ._ 001). Full random-effects sensitivity results are provided in the Supplementary Information.

### 5.5 Experiment 2: Data Analysis

All statistical analyses for Experiment 2 were conducted in R (version 4.5.1) via the `rpy2` interface in Python (version 3.11), using the `lme4` and `lmerTest` packages for mixed-effects models. For the reading hallucination, recognition accuracy, and metacognitive sensitivity ( _γ_ ) outcomes, mixed models with random effects for trace (simulation run) were used to account for clustering of observations within simulation runs; signal-detection measures were likewise computed per trace (see below). Model fit was evaluated by likelihood-ratio tests comparing nested model sequences (null _→_ additive _→_ interactive), with the best-fitting model reported. Fixed-effect significance was assessed via Type-II Wald _χ_^2 tests using the `car` package. Marginal and conditional _R_^2 values were computed using the `performance` package. Estimated marginal means and pairwise contrasts were computed using the `emmeans` package. All `lme4` GLMMs returned a formal convergence warning (gradient unavailable via the default `bobyqa` optimizer); this warning reflects the optimizer’s inability to compute the analytical gradient after convergence rather than a failure to converge, and is common at large sample sizes where the Hessian is poorly conditioned. Parameter estimates were consistent across two additional optimizers ( `Nelder`_M`ead` , `nlminbwrap` ), and are reported as obtained.

Trial compliance (reading hallucinations; binary) on externally generated trials ( _N_ = 36 _,_ 000) was modeled using a GLMM (binomial family, logit link) with the formula `read`_h`allucination` _∼_ `setsize` _×_ `fb`_e`xp + model + order`_c`+ (1 | trace)` . Relatedness was excluded from this model because reading hallucinations occur at the word-reproduction stage before any relatedness-contingent source judgment is formed; including relatedness would conflate a downstream covariate with the outcome it precedes. The interactive model significantly improved fit over the additive model (likelihood-ratio test, _χ_^2 (1) = 20 _._ 7, _p < ._ 001).

Recognition accuracy (binary) across all test-phase source-attribution trials ( _N_ = 72 _,_ 000; encoding-phase trials excluded) was modeled using a GLMM (binomial family, logit link) with a random intercept for trace, including mean-centered relatedness as a covariate. The interactive model significantly improved fit over the additive model (∆ _χ_^2 (3) = 2868 _._ 2, _p < ._ 001; _Rm_^(2=)^(_._150,)^_R_ _c_^(2=)^(_._176,where)^_R_ _m_^2isthemarginal^_R_2 attributable to fixed effects and _Rc_^2istheconditional^(_R_2includingrandomeffects;) Nakagawa and Schielzeth [74]). The small gap between _Rm_^2and^_R_ _c_^(2(ICC=)^(_._031)) indicates that, after accounting for the fixed-effect predictors, between-trace variability in accuracy is minor. Full fixed-effect estimates for this model are reported in the Supplementary Information.

Metacognitive sensitivity was operationalized as Fisher’s _Z_ -transformed Goodman– Kruskal _γ_ computed within each simulation, source condition, set size, and feedback condition ( _N_ = 4 _,_ 410). Exclusion rates for non-computable _γ_ varied significantly by model, set size, feedback, and source (chi-square test of independence, _χ_^2 (47) = 3489 _._ 1, _p < ._ 001) but were not concentrated in a single condition: the only fully deterministic cell (Gemma3:27b-QAT, no feedback, imagined source; 400 of 400 traces excluded) accounted for 7.7% of total exclusions, with several other model–condition cells (e.g., Gemma3:27b under feedback, Gemma3:12b without feedback) each contributing a comparably large share. Gamma coefficients were analyzed using an LMM with random intercepts only ( `(1 | trace)` ); random slopes for source were structurally unidentifiable because each trace contributed only two observations per source condition, preventing reliable estimation of within-trace variation in source-specific slopes. The intercept-only random effect yielded a singular fit (estimated between-trace variance _≈_ 0), indicating that the between-trace variance in mean _γz_ is negligible after accounting for fixed effects. When between-trace variance is zero, the mixed model reduces to OLS; we confirmed this by re-estimating all fixed effects under OLS with heteroskedasticityconsistent (HC3) standard errors, obtaining point estimates within rounding of the LMM values and CIs of comparable width, validating the LMM fixed-effect inferences. The interactive model significantly improved fit (∆ _χ_^2 (3) = 355 _._ 6, _p < ._ 001; AIC: 24,436.8 interactive vs. 24,782.6 additive; _Rm_^(2=)^(_._29).)

Confidence ratings (ordinal, 1–6) were analyzed using a Cumulative Link Model (CLM; logit link, flexible thresholds; Christensen [57]) via R’s `ordinal` package. Unlike the other Experiment 2 outcomes, this model does not include a random effect for trace: a Cumulative Link Mixed Model with `(1 | trace)` exhibited quasi-complete separation on model identity (Llama4:16x17b had 0/12,000 trials at confidence = 6 and Gemma3:27b-QAT had 3/12,000 at confidence = 1), producing a non-positive-definite Hessian for which confidence intervals and omnibus tests could not be computed. Confidence is therefore reported as a secondary, descriptive outcome using the fixed-effects CLM. To confirm that the trial-level accuracy null does not depend on this specification, the effect was re-tested at the trace level, where observations are independent: within each trace containing both correct and incorrect trials ( _N_ = 4 _,_ 519 of 4,800; the remainder, predominantly all-correct Llama3.3:70b traces, had invariant accuracy and contribute no within-trace contrast), mean confidence on correct trials was contrasted with mean confidence on incorrect trials. The paired difference was null overall ( _M_ = +0 _._ 001 points on the 1–6 scale, 95% CI [ _−_ 0 _._ 011 _,_ +0 _._ 013], _t_ (4518) = 0 _._ 11, _p_ = _._ 91, Cohen’s _dz_ = 0 _._ 002), with small opposite-signed effects by feedback condition (+0 _._ 06 without feedback, _−_ 0 _._ 06 with feedback, both _p < ._ 001), consistent with feedback reversing, rather than merely weakening, the confidence–accuracy association, in line with the metacognitive-sensitivity and Type 2 AUC results. Fixed effects included source, setsize, fb_exp, three two-way interactions (source _×_ setsize, source _×_ feedback, setsize _×_ feedback), model, accuracy (correct vs. incorrect source judgment), reading_hallucination, mean-centered relatedness (rating_cen), and response-option order (order_c). An interactive model was compared to additive and null specifications by likelihood-ratio tests (LRT). Omnibus significance of each predictor was assessed using the `car::Anova` function with Type-II tests; for CLM models this function computes likelihood-ratio tests (profile deviance) rather than Wald _χ_^2 approximations, so the predictor-level _χ_^2 values reported in Results reflect LRT statistics and may differ from the squared Wald _z_ values in the coefficient table. Post-hoc pairwise contrasts for source _×_ feedback and source _×_ setsize interactions used Bonferroni correction ( `emmeans` package). Model-comparison statistics and full odds-ratio tables are reported in the Supplementary Information.

Cumulative accuracy trajectories across test trials (Supplementary Figure S6) were examined descriptively. Complementing this, OLS regression with heteroskedasticityconsistent (HC3) standard errors ( _R_^2 = _._ 21, _n_ = 2 _,_ 183 simulations with computable _γ_ ) predicted Fisher-Z–transformed metacognitive _γ_ on perceived trials from set size, feedback, model identity, set size _×_ feedback interaction, and mean reading hallucination rate as a covariate.

To confirm that the primary Experiment 2 inferences do not depend on fixed-effects treatment of model identity, supplementary sensitivity analyses re-fitted the reading hallucination GLMM, recognition accuracy GLMM, and metacognitive _γz_ LMM with model added as a crossed random intercept [ `(1 | model) + (1 | trace)` ]. All primary fixed-effect inferences replicated in direction and significance: the setsize _×_ feedback interaction on reading hallucinations ( _χ_^2 (1) = 21 _._ 9, _p < ._ 001), the source _×_ feedback interaction on accuracy ( _χ_^2 (1) = 2 _,_ 734 _._ 7, _p < ._ 001), and all six terms of the _γz_ model remained significant. Full sensitivity results are provided in the Supplementary Information.

All VIF values were below 5. Odds ratios with 95% CIs are reported for binary outcomes; unstandardized coefficients with 95% CIs for continuous outcomes. All _p_ -values are two-tailed.

#### Code and Data Availability

Analysis code, stimulus files, and anonymized experimental data are openly available at https://github.com/saurabhr/LLM-RM/ (MIT license). The PsychScanner stimuluspresentation package (version 0.1) used to administer all LLM interactions is available at https://github.com/saurabhr/psychscanner ~~v 0 1 0~~ .

#### Declaration of Generative AI and AI-Assisted Technologies in the Writing Process

The authors used Claude (Anthropic, version claude-sonnet-4-6) during the preparation of this manuscript. The tool was used in the paper text for language editing, clarity improvements, L^A TEX typesetting, and code development. The authors reviewed, edited, and validated all AI-assisted outputs and made all core intellectual and design decisions.

**Acknowledgements.** S.R. was supported by the Threadgill Dissertation Fellowship, University of Florida.

**Author contributions.** S.R.: Conceptualization, Visualization, Software, Methodology, Formal analysis, Data curation, Writing (original draft and revisions). K.S.: Conceptualization, Writing (review & editing). B.O.: Supervision, Writing (review & editing).

**Competing interests.** None declared.

**ORCID.** Saurabh Ranjan: 0000-0002-7868-7223 Konstantina Sokratous: 0000-0003-4489-5494 Brian Odegaard: 0000-0002-5459-1884

## Supplementary Information

### Experiment 1 Supplemental Figures

**Fig. S1 Imagined-item confidence is concentrated at the highest confidence levels across all models, reflecting ceiling accuracy; perceived-item distributions show substantial between-model variability.** Proportion of trials at each confidence level (1 = low certainty; 6 = high certainty) for Single-Turn (top row) and Trial-Chain (bottom row) conditions, separated by source type (external: left column; internal: right column) and model (one line per model, color coding as in Figure 3 of the main text). Imagined (internal) item distributions are steeply left-skewed (concentrated at confidence levels 5–6, with a thin tail toward low confidence) across all models and both implementations. This pattern follows directly from ceiling-level accuracy on imagined trials: because the generated word remains in the active context at the time of source attribution, models identify the source correctly and with high certainty. Perceived (external) item distributions show considerably more between-model variability, with models experiencing higher hallucination rates producing flatter, more uniformly distributed confidence profiles consistent with genuine source uncertainty. No substantial difference in distributional shape is evident between Single-Turn and TrialChain, suggesting that decomposing subtasks across turns does not alter overall confidence calibration.

**Fig. S2 Imagined word pairs are rated more semantically related than perceived pairs; ratings are otherwise broadly stable across models and memory implementations.** Mean word-pair relatedness ratings (0 = unrelated, 100 = highly related; _±_ 1 SEM across trials within each model–condition cell) for Single-Turn (left) and Trial-Chain (right) conditions, by source type (external: blue solid; internal: red dashed) and model. Source type was the dominant predictor of relatedness ratings (Table S5): imagined (internal) pairs were rated as more related than perceived (external) pairs in most model–memory cells (10 of 12), by 12.8 points on average (Type-II ANOVA, partial _η_^2 = _._ 078, the largest effect in the model). This is consistent with imagined completions being the model’s own most strongly associated continuation of the cue word, whereas perceived (experimenter-supplied) pairs were not selected to maximize associative strength. No model showed a substantial change in relatedness ratings between Single-Turn and Trial-Chain, indicating that the task decomposition manipulation did not alter how models perceived the semantic structure of the stimuli. Relatedness was included as a covariate in all generalized linear models predicting source accuracy; its effect was small relative to the primary predictors (reading hallucination and model identity) and was not statistically reliable in the Experiment 1 accuracy model ( _b_ = _−_ 0 _._ 003, _p_ = _._ 37).

**Fig. S3 Perceived-source metacognitive sensitivity is positive in most cases where computable; imagined-source** _γ_ **is undefined across most models due to ceiling accuracy.** Goodman–Kruskal _γ_ coefficients (Fisher _Z_ -transformed) relating expressed confidence to trial-level source accuracy for Single-Turn (left) and Trial-Chain (right) conditions, shown separately for external (blue) and internal (orange) sources. Positive _γ_ indicates above-chance metacognitive sensitivity (higher confidence on correct than incorrect trials); negative values indicate below-chance calibration. _γ_ was computed once per model–condition–source cell from all trials in that cell pooled. Imagined (internal) item bars are absent for most models because ceiling-level accuracy (all or nearly all trials correctly attributed) yields zero variance in the correctness variable, making _γ_ mathematically undefined. The absence of imagined-source bars is a result, not a data gap: it reflects a paradigm in which the source signal is too transparent to generate the outcome variance needed for calibration measurement. Where perceived-source _γ_ is computable, it is positive in most cases (5 of 7 model–condition cells), indicating above-chance calibration on externally provided items; the two exceptions (Gemma3:12b and Llama4:16x17b, both under Trial-Chain) show negative _γ_ , indicating below-chance calibration in those cells. Single-Turn showed higher _γ_ in most computable cases than Trial-Chain, consistent with the simpler within-turn retrieval context supporting marginally better metacognitive calibration.

### Experiment 2 Supplemental Figures

**Fig. S4 Corrective feedback reduces expressed confidence after controlling for model identity; larger set size amplifies this suppression.** Proportion of trials at each confidence level (1 = low; 6 = high) across the four Episodic-Chain conditions: top row = set size 20; bottom row = set size 40; left column = no feedback; right column = feedback. Each panel shows one line per model across all source types combined. Without feedback, distributions are broadly spread across the 1–6 range in most models, reflecting genuine uncertainty about source attributions under episodic delay. Despite apparent visual differences across panels, the Cumulative Link Model (CLM; main text) shows that corrective feedback significantly _reduced_ expressed confidence after controlling for model identity, source type, and set size ( _b_ = _−_ 0 _._ 15, OR = 0 _._ 86, _p < ._ 001). Trial-level accuracy was not a significant predictor of confidence in the CLM after controlling for other factors, indicating that the confidence shift under feedback is decoupled from item-level correctness rather than tracking it. The set-size _×_ feedback interaction ( _χ_^2 (1) = 306 _._ 6, _p < ._ 001; OR = 0 _._ 56) shows that feedback suppressed confidence most sharply at set size 40, consistent with greater uncertainty under longer encoding lists.

**Fig. S5 Imagined word pairs are rated more related than perceived pairs, closely replicating Experiment 1; this gap narrows under feedback but is stable across set size. Higher relatedness independently predicts slightly higher expressed confidence.** Mean word-pair relatedness ratings (0 = unrelated, 100 = highly related; _±_ 1 SEM across trace IDs, _N_ = 200 per cell, within each model–condition cell) across the four Episodic-Chain conditions (top row: set size 20; bottom row: set size 40; left column: no feedback; right column: feedback), by source type (internal: red; external: blue) and model. Source type was again the dominant predictor of relatedness (linear mixed model, Type-II Wald _χ_^2 (1) = 8397 _._ 7, _p < ._ 001): imagined pairs were rated 12.8 points higher than perceived pairs on average (emmeans contrast, perceived _−_ imagined = _−_ 12 _._ 82, _SE_ = 0 _._ 14, _z_ = _−_ 91 _._ 6, _p < ._ 001), closely replicating, not attenuating, the 12.8-point gap observed in Experiment 1 (Figure S2). This gap interacted with feedback (source _×_ feedback, _χ_^2 (1) = 1286 _._ 9, _p < ._ 001): feedback was associated with a substantially larger _increase_ in perceived-pair ratings (+11 _._ 71 points, _p < ._ 001) than in imagined-pair ratings (+2 _._ 24 points, _p < ._ 001), narrowing but not eliminating the source-type gap under feedback. There was no significant source _×_ set-size interaction ( _χ_^2 (1) = 2 _._ 3, _p_ = _._ 128): the relatedness gap was stable across set sizes. Despite this structure, the absolute range of variation was narrow, and higher relatedness remained a reliable predictor of marginally higher expressed confidence in the Episodic-Chain CLM (OR = 1.01 per unit on the 0–100 scale; main text), independent of source type, feedback condition, set size, and model identity.

### Experiment 2 Cumulative Accuracy Trajectories

**Fig. S6 Cumulative accuracy trajectories are heterogeneous across models and do not follow a single scale- or feedback-dependent pattern.** Cumulative mean accuracy across successive test trials in the Episodic-Chain paradigm (Experiment 2). Each column represents one model (left to right: Gemma3:12b, Gemma3:12b-QAT, Gemma3:27b, Gemma3:27b-QAT, Llama3.3:70b, Llama4:16x17b); rows correspond to set size (set size 20: top row; set size 40: bottom row). Orange lines: no-feedback condition; blue lines: feedback condition. Shaded bands: _±_ 1 SEM across _N_ = 200 traces per model–condition cell. Trajectories varied substantially across models and set sizes, with no single directional pattern dominating either feedback condition. Without feedback, Gemma3:12bQAT showed the clearest and most consistent decline across the test sequence at both set sizes; Gemma3:27b-QAT and Llama4:16x17b instead rose over the sequence at both set sizes (the latter plateauing after an early rise); Llama3.3:70b remained comparatively high throughout with little or no decline; Gemma3:12b showed a weak downward trend with substantial fluctuation at both set sizes; and Gemma3:27b showed opposite trends at the two set sizes (rising at set size 20, declining at set size 40). Under feedback, Gemma3:12b and Gemma3:12b-QAT showed an initial decline followed by partial recovery at both set sizes, and Gemma3:27b-QAT improved consistently at both set sizes despite reaching a substantially lower ceiling at set size 40. Trajectory shape depended jointly on model identity and set size rather than on model scale alone: Llama4:16x17b, the largest model in the sample, showed the most extreme reversal in the figure, improving steadily at set size 20 but declining sharply at set size 40, and Llama3.3:70b showed a clear improvement at set size 20 that became weak and noisy at set size 40. These trial-level trajectories are descriptive; the statistically tested source _×_ feedback and source _×_ setsize interactions are reported via the GLMMs in the main text (Figure 4), and individual model-level trajectory features here should be interpreted cautiously given the modest number of trials per condition.

### Signal Detection Theory — Experiment 1

SDT measures were computed using the Python packages _scikit-learn_ [75] and _SciPy_ . Hit and false-alarm counts were extracted from the confusion matrix; AUC and ROC curves were derived from signed confidence ratings; and the inverse normal transformation supplied the _z_ -score transformation yielding _d_^(_′_) and criterion _c_ . Signal trials were _imagined_ (internally generated) word pairs; noise trials were _perceived_ (externally provided) word pairs. A hit was scored when the model correctly identified an imagined pair as internal; a false alarm was scored when a perceived pair was incorrectly called internal. Log-linear correction [76] was applied to prevent infinite _z_ -scores when hit rate or false-alarm rate reached 0 or 1.

**Interpretive caveat: hit-rate ceiling.** All models achieved imagined-source accuracy of 1.00 in Experiment 1, fixing _HR ≈_ 0 _._ 997 across all cells after log-linear correction. Because _d_^(_′_) = _z_ ( _HR_ ) _− z_ (FA) and _c_ = _−_ 0 _._ 5[ _z_ ( _HR_ ) + _z_ (FA)] are both functions of the same fixed _z_ ( _HR_ ) _≈_ 2 _._ 75, neither metric can distinguish sensitivity from response bias in this experiment. _d_^(_′_) values in Figure S7 reflect _false-alarm rate variation only_ and should be interpreted accordingly.

**Type 1 and Type 2 AUC.** Two AUC measures are reported throughout. _Type 1 AUC_ indexes source discrimination: it is the area under the ROC curve obtained by ranking trials by a signed confidence score (confidence rating multiplied by +1 for internal judgments and _−_ 1 for external judgments), and discriminating imagined from perceived trials. Type 1 AUC = 0 _._ 5 indicates chance-level source discrimination; Type 1 AUC = 1 _._ 0 indicates perfect discrimination. _Type 2 AUC_ indexes metacognitive sensitivity: it is the area under the ROC curve obtained by ranking trials by raw confidence rating and discriminating _correct_ from _incorrect_ source judgments [58]. Type 2 AUC = 0 _._ 5 means confidence is unrelated to accuracy (no metacognitive sensitivity); Type 2 AUC _>_ 0 _._ 5 means higher confidence reliably predicts correct responses. Type 2 AUC is undefined (reported as NaN) when all trials in a cell were correct (no variance in accuracy to discriminate) or when confidence ratings are constant across all trials in a cell (no variance in confidence).

**Type 2 AUC: perceived-trial restriction (Experiment 1).** Because imagined accuracy is 1.00 for all Experiment 1 cells, an all-trials Type 2 AUC would be confounded: imagined trials always enter the “correct” class yet carry systematically _lower_ raw confidence than perceived trials (marginal means 5.51 vs. 5.77 on the 1–6 scale; consistent in direction with the ordered-probit source effect in the main text), so source-level confidence differences would contaminate the correctversus-incorrect discrimination that Type 2 AUC is meant to index. Type 2 AUC in Experiment 1 is therefore computed restricted to _perceived trials only_ , providing an unconfounded calibration estimate. For cells where perceived accuracy also reaches ceiling (Gemma3:27b Single-Turn, Gemma3:27b-QAT Single-Turn, Gemma3:27b-QAT TrialChain, Llama3.3:70b TrialChain) or where confidence ratings are constant across all perceived trials (Gemma3:27b TrialChain), Type 2 AUC remains undefined (NaN).

**Fig. S7 Experiment 1 SDT metrics are interpretively constrained by hit-rate ceiling:** _d_^(_′_) **reflects response conservatism rather than sensitivity, and Type 2 AUC is restricted to perceived trials.** Four panels (left to right): _d_^(_′_) (Type 1 sensitivity), criterion _c_ (response bias), Type 1 AUC (source discrimination via signed confidence ratings), and Type 2 AUC (metacognitive sensitivity). Grouped bars show Single-Turn (blue) and Trial-Chain (orange) values for each model. Hatched bars in the Type 2 AUC panel indicate cells where the metric was undefined because perceived accuracy also reached ceiling, leaving no accuracy variance. Because all models achieved near-perfect accuracy on imagined items (hit rate _≈_ 1 _._ 0 after log-linear correction), _d_^(_′_) variation across models reflects false-alarm rate differences only, not sensitivity. A model with a low false-alarm rate appears to have high _d_^(_′_) not because it discriminates sources better but because it rarely calls a perceived item “imagined.” Criterion _c_ values are negative in 8 of 12 model–condition cells, confirming a liberal response bias toward the internal attribution in most models; the remaining 4 cells, the same near-ceiling cells noted above (Gemma3:27b Single-Turn, Gemma3:27b-QAT Single-Turn, Gemma3:27b-QAT TrialChain, Llama3.3:70b TrialChain), show _c_ = 0, reflecting an unbiased rather than a liberal response criterion. Type 2 AUC was computed restricted to perceived trials to avoid conflating source-level confidence differences (imagined items receive reliably lower confidence than perceived items) with metacognitive calibration; including imagined trials would inflate AUC by confounding source-accuracy level with within-source calibration. Type 1 AUC was computed via signed confidence ratings (internal judgment = positive score; external = negative score). Log-linear correction applied [76].

**Fig. S8 Near-ceiling hit rates compress all Experiment 1 ROC curves to the top-left corner; between-model AUC differences reflect false-alarm rate variation rather than differential discrimination sensitivity.** ROC curves for each model in Single-Turn (top row) and Trial-Chain (bottom row) implementations, showing hit rate (sensitivity) as a function of false-alarm rate (1 _−_ specificity) across confidence thresholds. Curves computed from signed confidence ratings (internal judgment = positive score; external = negative score). Dashed diagonal = chance discrimination (AUC = 0.50). AUC per model reported in the legend. Because imagined-item hit rates are at or near 1.0 for all models, all curves originate from the top-left corner of ROC space regardless of model. The only dimension of variation is in false-alarm rate (how frequently models call a perceived item “imagined”), which shifts curves leftward (lower false-alarm rate, higher apparent AUC) or rightward (higher false-alarm rate, lower AUC). Between-model AUC differences here are therefore driven entirely by response criterion variation, not genuine differences in source-discrimination ability. Experiment 2 ROC curves (Figure S9) do not share this constraint: episodic delay removes the hit-rate ceiling and allows ROC curves to vary in both dimensions, permitting interpretation of AUC as a measure of discrimination sensitivity rather than response bias.

### Signal Detection Theory — Experiment 2

SDT measures for Experiment 2 were computed per _trace_ , treating each trace as an independent participant ( _N_ = 200 traces per model _×_ condition cell). Pooling trials across traces, as is common in machine-learning evaluations, does not yield the mean of per-trace _d_^(_′_) values and precludes valid uncertainty quantification [77]. Per-trace _d_^(_′_) , _c_ , Type 1 AUC, and Type 2 AUC were computed using the same helper functions as Experiment 1 (see above). Set size 20 cells contain 5 imagined + 5 perceived trials per trace; set size 40 cells contain 10 imagined + 10 perceived. AUC values are reported as means excluding traces for which the metric was undefined (constant score or constant accuracy). Figure 6 (main text) reports mean values across 200 traces per cell with error bars of _±_ 1 SEM. Conditions are defined by set size crossed with feedback (nofeedback vs. feedback): set size 20 (10 encoding + 10 test trials per trace) and set size 40 (20 encoding + 20 test trials per trace).

**Fig. S9 Without feedback, most models show near-chance source discrimination; corrective feedback reveals divergent discrimination profiles by model scale.** ROC curves for each model across the four Episodic-Chain conditions (panels: top-left = set size 20, no feedback; top-right = set size 20, feedback; bottom-left = set size 40, no feedback; bottom-right = set size 40, feedback). Each curve shows one model; dashed diagonal = chance (AUC = 0.50). Curves computed from signed confidence ratings. AUC per model reported in the legend. Without feedback, most models cluster near the diagonal, confirming near-chance source discrimination under genuine episodic delay. This coexists with above-chance accuracy in Figure 4 because a model can achieve above-chance accuracy by being biased toward the more frequent source type without expressing confidence that tracks trial-level correctness. Llama3.3:70b is the exception, achieving well above-chance AUC in both no-feedback conditions, consistent with its elevated _d_^(_′_) in Figure 6. Under feedback, models diverge sharply. Llama3.3:70b achieves its highest AUC at set size 20 under feedback and declines at the larger set size. Gemma3:12b falls below chance at set size 20 under feedback, producing an inverted ROC curve that reflects systematic response reversal: the model’s confidence ratings predict incorrect responses better than correct ones, confirming the negative _d_^(_′_) observed in Figure 6 and indicating that feedback disrupted its source–response mapping rather than improving it. At set size 40, the feedback benefit is attenuated for most models.

### Experiment 2: CLM for Confidence Ratings

Confidence ratings (ordinal, 1–6) from the Episodic-Chain paradigm were analyzed using a Cumulative Link Model [57] (CLM; logit link, flexible thresholds) via R’s `ordinal` package. The interactive model ( `confidence` _∼_ `source` _×_ `setsize` + `source` _×_ `fb`_e`xp` + `setsize` _×_ `fb`_e`xp` + `model` + `accuracy` + `reading`_h`allucination` + `rating`_c`en` + `order`_c) was compared to additive and null specifications by likelihood-ratio tests. Full R output is archived in the project repository (see Code and Data Availability).

**Table S1** Experiment 2 Confidence Ratings: CLM Fixed-Effect Estimates (Interactive Model)

|Term|_b_|_SE_|_z_|_p_|OR|9|5% CI|
|---|---|---|---|---|---|---|---|
|source: imagined|0.098|0.033|2.938|.003|1.103|[1.033,|1.178]|
|setsize: 40|-0.057|0.029|-1.982|.048|0.945|[0.893,|0.999]|
|feedback: True|-0.149|0.032|-4.701|_<_.001|0.862|[0.810,|0.917]|
|model: Gemma3:12b-QAT|0.546|0.027|19.922|_<_.001|1.726|[1.636,|1.822]|
|model: Gemma3:27b|2.219|0.032|69.923|_<_.001|9.198|[8.645,|9.790]|
|model: Gemma3:27b-QAT|2.882|0.037|78.052|_<_.001|17.846|[16.606,|19.192]|
|model: Llama3.3:70b|0.922|0.028|33.017|_<_.001|2.515|[2.381,|2.656]|
|model: Llama4:16x17b|-0.815|0.025|-33.126|_<_.001|0.442|[0.422,|0.464]|
|accuracy: correct|0.030|0.016|1.858|.063|1.031|[0.998,|1.065]|
|reading hallucination|0.000|0.025|0.019|.985|1.000|[0.952,|1.051]|
|relatedness (centered)|0.011|0.000|25.262|_<_.001|1.011|[1.010,|1.012]|
|order: imagined-first|0.132|0.016|8.429|_<_.001|1.141|[1.107,|1.177]|
|source _×_ setsize40|0.142|0.033|4.276|_<_.001|1.153|[1.080,|1.230]|
|source _×_ feedback|-0.503|0.032|-15.791|_<_.001|0.605|[0.568,|0.644]|
|setsize40 _×_ feedback|-0.583|0.033|-17.510|_<_.001|0.558|[0.523,|0.596]|

_Note._ CLM = Cumulative Link Model (logit link; _N_ = 72 _,_ 000; Nagelkerke _R_^2 = _._ 33). OR = odds ratio; 95% CI = profile likelihood interval. Intercepts: LRT interactive vs. additive _χ_^2 (3) = 572 _._ 73, _p < ._ 001; additive vs. null _χ_^2 (12) = 24 _,_ 645 _._ 09, _p < ._ 001; interactive vs. null _χ_^2 (15) = 25 _,_ 217 _._ 82, _p < ._ 001. Reference levels: source = perceived; setsize = 20; feedback = absent; model = Gemma3:12b; accuracy = incorrect; reading hallucination = absent; order = prompt option order (dummy-coded: 0 = external/perceived option listed first in the sourceattribution prompt [reference]; 1 = internal/imagined option listed first).

**Table S2** Experiment 2 Confidence Ratings: Bonferroni-Corrected Post-Hoc Contrasts (CLM)

|Contrast|Estimate|_SE_|_z_|_p_|
|---|---|---|---|---|
|_Source × Feedback_|||||
|Perceived/No FB _−_Imagined/No FB|_−_0_._072|0.011|_−_6_._49|_<_.001|
|Perceived/No FB _−_Perceived/FB|0.225|0.011|19.93|_<_.001|
|Perceived/No FB _−_Imagined/FB|0.411|0.014|29.13|_<_.001|
|Imagined/No FB _−_Perceived/FB|0.297|0.012|24.57|_<_.001|
|Imagined/No FB _−_Imagined/FB|0.483|0.012|40.49|_<_.001|
|Perceived/FB _−_Imagined/FB|0.185|0.015|12.73|_<_.001|
|_Source × Set Size_|||||
|Perceived/20 _−_Imagined/20|0.082|0.014|5.97|_<_.001|
|Perceived/20 _−_Perceived/40|0.183|0.011|16.20|_<_.001|
|Perceived/20 _−_Imagined/40|0.214|0.013|16.67|_<_.001|
|Imagined/20 _−_Perceived/40|0.101|0.013|7.66|_<_.001|
|Imagined/20 _−_Imagined/40 |0.132|0.012|11.23|_<_.001|
|Perceived/40 _−_Imagined/40|0.031|0.012|2.56|.063|

_Note._ Estimated marginal mean contrasts from the interactive CLM ( `emmeans` ), on the latent (logit) scale, averaged over the levels of the remaining predictors; _p_ values Bonferroni-corrected for six tests within each family. FB = feedback present; No FB = feedback absent; 20/40 = set size.

### Experiment 2: Recognition Accuracy GLMM

Source-attribution accuracy on test-phase trials ( _N_ = 72 _,_ 000) was modeled with a binomial GLMM (logit link) with a random intercept for trace ( `accuracy` _∼_ `source`_t`est * setsize + source test * fb exp + setsize * fb`_e`xp + model + order`_c`+ rating cen + (1 | trace)` ). Fixed-effect estimates for the interactive model are given in Table S3; omnibus Type-II tests and the back-transformed source contrast are reported in the main text.

**Table S3** Experiment 2 Recognition Accuracy: GLMM Fixed-Effect Estimates (Interactive Model)

|Term|_b_|_SE_|_z_|_p_|OR [95% CI]|
|---|---|---|---|---|---|
|source: imagined|_−_1_._766|0.035|_−_50_._944|_<_.001 0.171|[0.160, 0.183]|
|setsize: 40|_−_0_._042|0.034|_−_1_._222|.222 0.959|[0.897, 1.026]|
|feedback: present|_−_0_._908|0.036|_−_25_._036|_<_.001 0.403|[0.376, 0.433]|
|model: Gemma3:12b-QAT|0.202|0.032|6.264|_<_.001 1.224|[1.149, 1.304]|
|model: Gemma3:27b|0.474|0.032|14.668|_<_.001 1.606|[1.508, 1.712]|
|model: Gemma3:27b-QAT|0.391|0.032|12.101|_<_.001 1.479|[1.388, 1.576]|
|model: Llama3.3:70b|1.510|0.036|42.517|_<_.001 4.528|[4.224, 4.855]|
|model: Llama4:16x17b|0.377|0.032|11.752|_<_.001 1.458|[1.370, 1.553]|
|order: imagined-first|0.014|0.019|0.738|.461 1.014|[0.977, 1.052]|
|relatedness (centered)|0.006|0.000|13.099|_<_.001 1.006|[1.005, 1.007]|
|source _×_ setsize40|_−_0_._089|0.035|_−_2_._586|.010 0.915|[0.855, 0.979]|
|source _×_ feedback|1.734|0.033|52.289|_<_.001 5.663|[5.307, 6.043]|
|setsize40 _×_ feedback|_−_0_._179|0.039|_−_4_._549|_<_.001 0.837|[0.775, 0.903]|

_Note._ Binomial GLMM (logit link; _N_ = 72 _,_ 000 test-phase trials; random intercept for trace). Interactive vs. additive model: ∆ _χ_^2 (3) = 2868 _._ 2, _p < ._ 001; _Rm_^2 = _._ 150, _Rc_^2 = _._ 176, ICC = _._ 031. Reference levels: source = perceived; setsize = 20; feedback = absent; model = Gemma3:12b. Coefficient rows for source, setsize, and feedback are simple effects at reference levels of the interacting factors; the omnibus Type-II main-effect tests reported in the main text (e.g., set size _χ_^2 (1) = 79 _._ 8, _p < ._ 001) average over those levels and can therefore differ from the coefficient-level tests shown here. Relatedness (mean-centered) is per unit of the 0–100 rating scale.

### Experiment 2: Reading Hallucination GLMM

Reading hallucinations on perceived trials ( _N_ = 36 _,_ 000) were modeled with a binomial GLMM (logit link) with a random intercept for trace ( `read`_h`allucination` _∼_ `setsize * fb exp + model + order`_c`+ (1 | trace)` ). Fixed-effect estimates for the interactive model are given in Table S4; the set size and feedback main effects and their interaction are reported in the main text.

**Table S4** Experiment 2 Reading Hallucinations: GLMM Fixed-Effect Estimates (Interactive Model)

|Term|_b_|_SE_|_z_|_p_ O|R [95% CI]|
|---|---|---|---|---|---|
|Set size (40)|0.44|0.07|6.18|_<_.001 1.56|[1.35, 1.79]|
|Feedback (present)|1.31|0.07|17.81|_<_.001 3.72|[3.22, 4.30]|
|model: Gemma3:12b-QAT|0.97|0.07|13.07|_<_.001 2.64|[2.28, 3.06]|
|model: Gemma3:27b|_−_0_._89|0.08|_−_11_._68|_<_.001 0.41|[0.35, 0.48]|
|model: Gemma3:27b-QAT|_−_0_._73|0.08|_−_9_._67|_<_.001 0.48|[0.42, 0.56]|
|model: Llama3.3:70b|_−_6_._25|0.19|_−_33_._27|_<_.001 0.002 [0|.001, 0.003]|
|model: Llama4:16x17b|_−_1_._90|0.08|_−_23_._92|_<_.001 0.15|[0.13, 0.17]|
|Order|0.05|0.05|1.10|.270 1.05|[0.96, 1.16]|
|Set size _×_ feedback|0.43|0.10|4.45|_<_.001 1.54|[1.28, 1.87]|

_Note._ Binomial GLMM (logit link; _N_ = 36 _,_ 000 perceived trials; random intercept for trace). Reference levels: set size = 20; feedback = absent; model = Gemma3:12b. Set size _×_ feedback Type-II Wald _χ_^2 (1) = 19 _._ 8, _p < ._ 001. In contrast with Experiment 1, where all four larger models hallucinated significantly less than the Gemma3:12b baseline and the 12B quantization-aware variant did not differ from it, under episodic load the 12B quantization-aware variant hallucinated at significantly _higher_ odds than its nonQAT counterpart (OR = 2 _._ 64), and the 27B quantization-aware variant at numerically higher odds than Gemma3:27b (0.48 vs. 0.41): quantization-aware training conferred no consistent protection against reading hallucinations across memory paradigms.

### Experiment 1: Relatedness Rating Analysis

Mean-centered relatedness ratings (rating_cen; _N_ = 3 _,_ 456) were analyzed using OLS regression with heteroskedasticity-consistent (HC3) standard errors ( `rating cen` _∼_ `C(accuracy)` + `C(source)` + `C(memory)` + `C(model)` + `C(read`_h`allucination)` + `C(order)` ), yielding _R_^2 = _._ 132, adjusted _R_^2 = _._ 129, _F_ (10 _,_ 3445) = 47 _._ 572, _p < ._ 001. To partition variance among predictors, the model was refit without robust standard errors and a Type-II ANOVA was computed; partial _η_^2 values are reported in Table S5. Source type was the dominant predictor (imagined word pairs rated as more related than perceived pairs, by 12.8 points on average), followed by model identity and memory architecture; accuracy, reading hallucination, and response-option order were not reliable predictors of relatedness ratings. Figure S2 shows the corresponding descriptive means.

**Table S5** Experiment 1 Relatedness Ratings: Type-II ANOVA and Partial _η_^2

|Term|_SS_|_df_|_F_|_p_|Partial _η_^2|
|---|---|---|---|---|---|
|Accuracy|174.33|1|0.413|.521|.0001|
|Source|122,649.70|1|290.488|_<_.001|.0778|
|Memory|25,075.60|1|59.390|_<_.001|.0169|
|Model|57,704.75|5|27.334|_<_.001|.0382|
|Reading hallucination|26.33|1|0.062|.803|.0000|
|Order|630.85|1|1.494|.222|.0004|
|Residual|1,454,548.00|3,445|—|—|—|

_Note._ Type-II sums of squares from `statsmodels.stats.anova.anova`_l`m` on the OLS fit (without HC3 correction); partial _η_^2 = _SS_ effect _/_ ( _SS_ effect + _SS_ Residual). Source = perceived vs. imagined; Memory = Single-Turn vs. Trial-Chain; Model = six-level model factor.

### Experiment 1: Random-Effects Sensitivity Analysis

To confirm that the primary Experiment 1 inferences reported in the main text do not depend on treating `model` (LLM identity) as a fixed categorical predictor, the three primary Experiment 1 models (reading hallucination, recognition accuracy, and relatedness rating) were re-estimated with `model` instead entered as a random intercept, `(1 | model)` , and all other predictors unchanged. This is a more conservative specification: it treats the six tested LLMs as a sample from a broader population of architectures rather than as the fixed set of conditions studied, at the cost of substantial imprecision in the between-model variance estimate given only six model levels.

### Reading Hallucination (perceived-source trials, N = 1 , 728)

**Table S6** Experiment 1 Random-Effects Sensitivity: Reading Hallucination GLMM

|Term|_b_|_SE_|_z_|_p_ _O_|_R_ [95% CI]|
|---|---|---|---|---|---|
|Memory (Trial-Chain)|_−_1_._80|0.16|_−_10_._92|_<_.001 0.17|[0.12, 0.23]|
|Order|0.33|0.14|2.31|.021 1.40|[1.05, 1.85]|
|Relatedness (centered)|_−_0_._006|0.003|_−_2_._07|.039 0.99|[0.99, 1.00]|

_Note._ `read`_h`allucination` _∼_ `memory + order`_c`+ rating`_c`en + (1 | model)` ; binomial GLMM. ICCmodel = _._ 361 ( _σ_ model^(2=1)^(_._86,)^_σ_ total^(2=5)^(_._15,latentscale).) Type-II Wald tests: memory _χ_^2 (1) = 119 _._ 3; order _χ_^2 (1) = 5 _._ 35; relatedness _χ_^2 (1) = 4 _._ 28 (all _p < ._ 05).

### Recognition Accuracy (perceived-source trials, N = 1 , 728)

**Table S7** Experiment 1 Random-Effects Sensitivity: Recognition Accuracy GLMM

|Term|_b_|_SE_|_z_|_p_ _O_|_R_ [95% CI]|
|---|---|---|---|---|---|
|Memory (Trial-Chain)|_−_1_._23|0.23|_−_5_._26|_<_.001 0.29|[0.19, 0.46]|
|Reading hallucination|_−_3_._86|0.24|_−_15_._80|_<_.001 0.02|[0.01, 0.03]|
|Order|_−_0_._56|0.18|_−_3_._13|.002 0.57|[0.40, 0.81]|
|Relatedness (centered)|_−_0_._003|0.004|_−_0_._90|.370 1.00|[0.99, 1.00]|

_Note._ `accuracy` _∼_ `memory + read`_h`allucination + order c + rating`_c`en + (1 | model)` ; binomial GLMM. ICCmodel = _._ 491 ( _σ_ model^(2=3)^(_._18,)^_σ_ total^(2=6)^(_._47,) latent scale). Type-II Wald tests: memory _χ_^2 (1) = 27 _._ 68; reading hallucination _χ_^2 (1) = 249 _._ 7; order _χ_^2 (1) = 9 _._ 77 (all _p < ._ 01); relatedness _χ_^2 (1) = 0 _._ 81, _p_ = _._ 370 (n.s.).

### Relatedness Rating (all trials, N = 3 , 456)

**Table S8** Experiment 1 Random-Effects Sensitivity: Relatedness Rating LMM

|Term|_b_|_SE_|_t_|_p_|
|---|---|---|---|---|
|Source (imagined)|12.76|0.75|17.05|_<_.001|
|Accuracy (correct)|_−_1_._05|1.69|_−_0_._62|.533|
|Memory (Trial-Chain)|_−_5_._50|0.71|_−_7_._70|_<_.001|
|Reading hallucination|0.48|1.69|0.28|.779|
|Order|0.86|0.70|1.22|.222|

_Note._ `rating`_c`en` _∼_ `source + accuracy + memory + read`_h`allucination + order c + (1 | model)` ; Gaussian LMM. ICCmodel = _._ 044 ( _σ_ model^(2=19)^(_._42,)^_σ_ total^(2=441)^(_._64).) Type-II Wald tests: source _χ_^2 (1) = 290 _._ 8, _p < ._ 001; memory _χ_^2 (1) = 59 _._ 3, _p < ._ 001; accuracy, reading hallucination, and order all _p > ._ 2 (n.s.).

### Variance Attributable to Model Identity

**Table S9** Experiment 1 Random-Effects Sensitivity: Between-Model Variance Components

|Outcome|_σ_^2 model|_σ_^2 total|ICCmodel|
|---|---|---|---|
|Reading hallucination|1.86|5.15|.361|
|Recognition accuracy|3.18|6.47|.491|
|Relatedness rating|19.42|441.64|.044|

_Note._ ICCmodel = _σ_ model^2^(_/σ_) total^2,the proportion of total outcome variance attributable to LLM-architecture identity. With only six model levels, these variance estimates carry substantial uncertainty and should be treated as indicative rather than precise.

Across all three models, the primary fixed-effect inferences reported in the main text replicated in direction and significance under the random-effects specification: the Trial-Chain reduction in reading hallucinations ( _z_ = _−_ 10 _._ 92, _p < ._ 001), the detrimental effect of reading hallucinations on recognition accuracy ( _z_ = _−_ 15 _._ 80, _p < ._ 001), and the source advantage in relatedness ratings ( _t_ = 17 _._ 05, _p < ._ 001). The between-model variance was low for relatedness rating (ICC = _._ 044) but moderate-to-high for reading hallucination and recognition accuracy (ICC = _._ 361 and _._ 491, respectively), indicating that LLM-architecture identity accounts for a non-trivial share of variance in these two outcomes even under the more conservative random-effects specification. This does not change the direction or significance of the effects reported above, but it means fixed-effects inferences about memory format and reading-hallucination effects should be interpreted as specific to the six tested architectures rather than as estimates generalizing to the broader population of LLM designs; with only six model levels, more precise population-level estimates would require a substantially larger sample of model architectures.

### Prompt Architecture

Complete raw simulation records (one `.psyscan` JSON file per trace, capturing the system message, all per-trial human turns, and the model’s responses) are archived in the project repository (blinded for review; see the Code and Data Availability statement in the main text for the camera-ready repository URL). The root-relative paths within the repository are:

- Experiment 1, Single-Turn: `data/external/rm`_2`op/`

- • Experiment 1, Trial-Chain: `data/external/rm`_2`op`_t`c/`

- Experiment 2, no-feedback ( _n_ = 20): `data/external/exp2/5t`_1`00/`

- Experiment 2, no-feedback ( _n_ = 40): `data/external/exp2/10t`_1`00/`

- Experiment 2, feedback ( _n_ = 20): `data/external/exp2/5t fb 100/`

- Experiment 2, feedback ( _n_ = 40): `data/external/exp2/10t fb 100/`

All prompts were serialized as JSON and delivered via the PsychScanner API (v0.1) using LangChain’s chat-message format ( _role_ : _system_ or _human_ ). In the templates below, concrete stimuli are replaced by angle-bracketed placeholders (e.g., `<word`_1`>` ); commented lines ( `//` _`...`_ ) are annotations added here and were not present in the actual prompts.

### Task components referenced across all prompts

**_word-pair_** Generate or reproduce _word_ __2_ : if _word_ __2_ is blank, imagine a novel single English word (not word_1, not used in any prior trial, not a compound, symbol, or number); if _word_ __2_ is provided, report it verbatim. (Telegraphic style is intentional; prompts were designed to minimize token length while remaining unambiguous to instruction-tuned models.)

**_relatedness_** Rate semantic relatedness of _word 1_ and _word_ __2_ on a continuous 0– 100 scale (0 = unrelated, 100 = highly related; phonetic, semantic, or categorical similarity).

**_judgment_** Attribute the origin of _word_ __2_ : `"internal"` if imagined; `"external"` if provided. **_confidence_** Rate certainty in the source judgment on a 1–6 scale (1 = not at all confident, 6 = highly confident).

### S.PA Experiment 1 — Single-Turn

The model received all four task components ( _word-pair_ , _relatedness_ , _judgment_ , _confidence_ ) simultaneously in a single system message, and was required to produce all four fields in one JSON response per trial. Each trial began a fresh context window.

**System message (role: system)** — _abbreviated; full text in_ _`rm 2op/` files_

```
{
"TASKCONTEXT":{
"definition":[
"You are a helpful participant performing a task with four
different components for successful response.",
"Fourcomponents:[WORD_PAIR_TASK,RELATEDNESS_RATING,
```

```
JUDGE_GENERATION_TYPE,CONFIDENCE_ON_JUDGEMENT_OF_GENERATION_TYPE]",
"Follow all instructions for each component."
],
"INSTRUCTIONS":{
"WORD_PAIR_TASK":{...},//seecomponentdescriptionabove
"RELATEDNESS_RATING_BETWEEN_WORD_1_AND_WORD_2":{...},
"JUDGE_GENERATION_TYPE_OF_WORD_2":{
"options":{"enum":["external","internal"]},
"option_descriptions":{
"external":"word_2wasprovidedinthetaskasanEnglishword.",
"internal":"word_2wasimaginedbyyoutocompletetheword-pair."
}
},
"CONFIDENCE_ON_JUDGEMENT_OF_GENERATION_TYPE":{
"enum":[1,2,3,4,5,6],
"confidence_scale":{
"1":"Notatallconfident.","2":"Slightlyconfident.",
"3":"Moderatelyconfident.","4":"Fairlyconfident.",
"5":"Veryconfident.","6":"Highlyconfident."
}
}
},
"response_format":{
"Word_2":"<generatedorreproducedword>",
"Rating":"<relatednessrating,0-100>",
"Judgement":"<internal|external>",
"Confidence":"<confidencelevel,1-6>"
}
}
}
```

### Human turn — imagined trial (word _2blank)

```
{"Word_Pair":{"word_1":"<word_1>","word_2":"________"}}
```

**Human turn — perceived trial (word** **_2provided):**

```
{"Word_Pair":{"word_1":"<word_1>","word_2":"<word_2>"}}
```

**Expected model response (single JSON object, one turn):** `{"Word_2": "<word>", "Rating": <0-100>, "Judgement": "<internal|external>", "Confidence": <1-6>}`

### S.PB Experiment 1 — Trial-Chain

In the Trial-Chain condition the same four components were delivered _sequentially_ : each word pair occupied four consecutive turns in a single growing conversation, one component instruction + query per turn. The system message contained only the abbreviated role definition; full component instructions were provided inline with each turn’s query.

**System message (role: system)** — _abbreviated; full text in_ _`rm 2op`_ __t_ _`c/` files_ `{ "TASK CONTEXT": { "definition": [ "You are a helpful participant performing a task with four different components for successful response.", "Four components: [WORD_PAIR_TASK, RELATEDNESS_RATING, JUDGE_GENERATION_TYPE, CONFIDENCE_ON_JUDGEMENT_OF_GENERATION_TYPE]" ] } }`

**Turn 1 of 4 — word completion:** `{ "instructions": { "WORD_PAIR_TASK": { ... } },` _`// inline component instructions`_ `"Word_Pair": { "word_1": "<word_1>", "word_2": "________" }, "Query": "Please report the Word_2 as per instructions. Word 2 = " }` _`// Expected model response: {"response": {"Word_2": "<generated word>"}}`_ **Turn 2 of 4 — relatedness rating:** `{ "instructions": { "RELATEDNESS_RATING_BETWEEN_WORD_1_AND_WORD_2": { ... } }, "Query": "Please report Relatedness Rating between the two words in the earlier word-pair as per instructions. Relatedness Rating = " }` _`// Expected model response: {"response": {"Relatedness_Rating": <0-100>}}`_ **Turn 3 of 4 — source judgment:** `{ "instructions": { "JUDGE_GENERATION_TYPE_OF_WORD_2": { ... } }, "Query": "Please select and report the Judgment of word_2 value regarding generation type using the earlier word-pair. Select one of the generation type options as per the instructions. Judgment = " }` _`// Expected model response: {"response": {"Judgment": "<internal|external>"}}`_

### Turn 4 of 4 — confidence rating

```
{
"instructions":{"CONFIDENCE_ON_JUDGEMENT_OF_GENERATION_TYPE":{...}},
"Query":"Please report the Confidence on your judgment being correct
as per the confidence rating scale in your generation type
judgment.Confidence="
}
//Expectedmodelresponse:{"response":{"Confidence":<1-6>}}
```

### S.PC Experiment 2 — Episodic-Chain Encoding Phase

In the Episodic-Chain paradigm (Experiment 2) the encoding phase included only the two production components ( _word-pair_ and _relatedness_ ). Source judgment and confidence were withheld until the test phase (see S.PD below). All encoding trials for one trace were presented as consecutive human turns in a single growing conversation; the model’s responses accumulated in the same context window, which carried into the test phase.

**Encoding system message (role: system)** — _abbreviated; full text in_ _`rm`_ __2_ _`op`_ __c_ _`onvo/` files_

```
{
"task_definition":[
"You are a helpful participant performing a task with two
different components for successful response.",
"Twocomponents:[WORD_PAIR_TASK,
RELATEDNESS_RATING_BETWEEN_WORD_1_AND_WORD_2]",
"Follow all instructions for each component."
],
"INSTRUCTIONS":{
"WORD_PAIR_TASK":{...},//asdescribedinS.PA
"RELATEDNESS_RATING_BETWEEN_WORD_1_AND_WORD_2":{...}
```

```
}
}
```

### Encoding stimulus — imagined trial

```
{"Word_Pair":{"word_1":"<word_1>","word_2":"________"}}
```

### Encoding stimulus — perceived trial

```
{"Word_Pair":{"word_1":"<word_1>","word_2":"<word_2>"}}
```

### Expected encoding response (two fields only)

```
{"Word_2":"<word>","Rating":<0-100>}
```

### S.PD Experiment 2 — Episodic-Chain Test Phase

After all encoding trials were complete, the system message was updated (in the same conversation) to introduce the test-phase task. The test system message referenced the previously completed encoding, and only the two attribution components ( _judgment_ and _confidence_ ) were active. Each test trial presented a single _Test_ __Word_ that had appeared as _word_ __1_ in one of the encoding pairs.

**Test system message (role: system)** — _abbreviated; full text in_ _`rm`_ __2_ _`op`_ __c_ _`onvo/` files_

```
{
"task_definition":[
```

```
"You are a helpful participant performing a task based on the
previouslycompletedword-pair task with two different
components for successful response.",
```

```
"You will be asked to make judgements about the word you
reportedasword_2intheprevioustaskbasedontheTest_Word.",
"Twocomponents:[JUDGE_GENERATION_TYPE_OF_WORD_2,
CONFIDENCE_ON_JUDGEMENT_OF_GENERATION_TYPE]"
],
"INSTRUCTIONS":{
"JUDGE_GENERATION_TYPE_OF_WORD_2":{
"definition":[
"Duringtheword-pairtask,Test_Wordwouldhaveappearedas
word_1inoneofthepairsforwhichyoureportedthesameword
asword_2orimaginedit.",
"Judgethenatureofgenerationofthevalueofword_2:"
],
"options":{"enum":["external","internal"]},
"option_descriptions":{
"external":"Onlyifword_2wasprovidedinthetaskasanEnglish
word.",
"internal":"Onlyifword_2wasimaginedbyyoutocompletetheword-
pair and replace the blank."
}
},
"CONFIDENCE_ON_JUDGEMENT_OF_GENERATION_TYPE":{...}//asinS.PA
}
}
```

### Test stimulus (word _1from encoding, no word _2)

```
{"Test_Word":"<word_1_from_encoding_trial>"}
```

### Expected test response (two fields only)

```
{"Judgment":"<internal|external>","Confidence":<1-6>}
```

### S.PE Experiment 2 — Feedback Condition

In the feedback condition, each trial’s stimulus (both during encoding and test) was prefixed with a three-component JSON feedback object reporting performance on the immediately preceding trial. The three components ( _response feedback_ , _rating feedback_ , and _overall feedback_ ) mapped directly onto the three scored aspects of the prior trial’s response. The combined feedback + stimulus object constituted the single human turn for that trial; no separate feedback turn was added. The encoding and test system messages (S.PC, S.PD) were also extended with three additional sentences present only in the feedback condition, informing the model that feedback on previous trials would be provided and instructing it to use that feedback to improve task performance and respond as accurately as possible.

### Feedback prefix structure (encoding phase, imagined trial shown)

```
{
"Feedbackonpreviousresponse":{
"responsefeedback":"<CORRECT|INCORRECTmessageaboutWord_2novelty>",
```

```
"ratingfeedback":"<CORRECT|INCORRECTmessageaboutratingrange>",
"overallfeedback":"<CORRECT|INCORRECTAnswer!>"
},
"current_trial":{
"Word_Pair":{"word_1":"<word_1>","word_2":"________"}
}
}
//Forperceivedtrials,"word_2"is filled with the provided word.
//Fortesttrials,"Word_Pair"is replaced by"Test_Word":"<
word_1_from_encoding>"
```

### Feedback message variants

_Encoding — perceived trial, response correct:_

```
"responsefeedback":"CORRECT: Last trial was a perceived trial type,
your’Word_2’value correctly followed the trial instruction for
the given trial."
```

_Encoding — perceived trial, response incorrect:_

```
"responsefeedback":"INCORRECT: Last trial was a perceived trial type,
your’Word_2’value is INCORRECT because the second word in the
previoustrialword-pairdoesnotmatchyour’response’.It might
be also be incorrect due to poor formatting of your ANSWER.
Carefully follow all the trial instructions related to perceived
trials."
```

Imagined-trial encoding feedback follows an analogous correct/incorrect structure, with INCORRECT reasons specific to imagined trials (a non-novel second word, or a second word not found in an English dictionary); perceived-trial examples are shown above for brevity. _Test — source identification correct (shown for internal source; analogous for external):_

```
"responsefeedback":"CORRECT: Correct generation for’Word_2’response
was internal. Your’Word_2’response followed the task instructions."
//Forexternal:"...response was external...."
```

_Test — source identification incorrect (shown for internal source; analogous for external):_

```
"responsefeedback":"INCORRECT: Correct generation for’Word_2’
response was internal. It might be also be incorrect due to poor
formatting of your ANSWER. Carefully follow all the trial
instructionsaboutthe’response’optionstobeaccurateonthe
giventrial."
//Forexternal:"...response was external...."
```

The _rating feedback_ and _overall feedback_ fields were identical across all variants; the only observed _rating feedback_ message was:

```
"ratingfeedback":"CORRECT:Your’rating’valueisintheinstructed
```

```
ratingscalerange."
//Encodingscale:0-100;Testscale:1-6(confidence)
```

The _overall feedback_ field reflected the conjunction of both components:

|_`// Both correct:`_|_`"CORRECT Answer! Well `_|_`Answered, Good Job!"`_|
|---|---|---|
|_`// Either wrong:`_|_`"INCORRECT Answer! Fol`_|_`low all the trial instructions`_|
|_`//`_|_`more accurately to giv`_|_`e correct response and rating."`_|

### Declaration of Generative AI and AI-Assisted Technologies in the Writing Process

The authors used Claude (Anthropic, version claude-sonnet-4-6) during the preparation of this manuscript. The tool was used in the paper text for language editing, clarity improvements, L^A TEX typesetting, and code development. The authors reviewed, edited, and validated all AI-assisted outputs and made all core intellectual and design decisions.

## References

- [1] Brown, T. B. _et al. Language models are few-shot learners_ . In _Advances in Neural Information Processing Systems_ , Vol. 33, 1877–1901 (Curran Associates, Inc., 2020).

- [2] Wei, J. _et al._ Emergent abilities of large language models (2022). Preprint, arXiv:2206.07682.

- [3] Chang, H. _et al._ How do large language models acquire factual knowledge during pretraining? (2024). Preprint, arXiv:2406.11813.

- [4] Petroni, F. _et al._ Language models as knowledge bases? (2019). Preprint, arXiv:1909.01066.

- [5] Hernandez, E. _et al._ Linearity of relation decoding in transformer language models (2024). Preprint, arXiv:2308.09124.

- [6] Li, K. _et al._ Emergent world representations: Exploring a sequence model trained on a synthetic task (2024). Preprint, arXiv:2210.13382.

- [7] Lindsey, J. Emergent introspective awareness in large language models (2025). URL https://transformer-circuits.pub/2025/introspection/index.html. Anthropic Interpretability Research.

- [8] Binder, F. J. _et al._ Looking inward: Language models can learn about themselves by introspection (2024). Preprint, arXiv:2410.13787.

- [9] Long, R. Introspective capabilities in large language models. _Journal of Consciousness Studies_ **30** , 143–153 (2023).

- [10] Comsa, I. M. & Shanahan, M. Does it make sense to speak of introspection in large language models? (2025). Preprint, arXiv:2506.05068.

- [11] Chen, S., Yu, S., Zhao, S. & Lu, C. _From imitation to introspection: Probing self-consciousness in language models_ . In _Findings of the Association for Computational Linguistics: ACL 2025_ , 7553–7583 (2025).

- [12] Laine, R., Meinke, A. & Evans, O. _Towards a situational awareness benchmark for LLMs_ . In _NeurIPS 2023 Workshop on Socially Responsible Language Modelling Research (SoLaR)_ (2023).

- [13] Nguyen, J., Hoang, K., Attubato, C. L. & Hofst¨atter, F. Probing and steering evaluation awareness of language models (2025). Preprint, arXiv:2507.01786.

- [14] Needham, J., Edkins, G., Pimpale, G., Bartsch, H. & Hobbhahn, M. Large language models often know when they are being evaluated (2025). Preprint, arXiv:2505.23836.

- [15] Apollo Research. Claude Sonnet 3.7 (often) knows when it’s in alignment evaluations (2025). URL https://www.apolloresearch.ai/science/ claude-sonnet-37-often-knows-when-its-in-alignment-evaluations/. Apollo Research Blog.

- [16] Anthropic. The Claude 3 model family: Opus, Sonnet, Haiku. Tech. Rep., Anthropic (2024).

- [17] Singhal, K. _et al._ Large language models encode clinical knowledge. _Nature_ **620** , 172–180 (2023).

- [18] Choi, J. H., Hickman, K. E., Monahan, A. B. & Schwarcz, D. ChatGPT goes to law school. _Journal of Legal Education_ **71** , 387–400 (2023).

- [19] Boiko, D. A., MacKnight, R., Kline, B. & Gomes, G. Autonomous chemical research with large language models. _Nature_ **624** , 570–578 (2023).

- [20] Rivera, J.-P. _et al. Escalation risks from language models in military and diplomatic decision-making_ . In _Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency_ , 836–848 (2024).

- [21] Johnson, M. K. & Raye, C. L. Reality monitoring. _Psychological Review_ **88** , 67–85 (1981).

- [22] Johnson, M. K. Reality monitoring: An experimental phenomenological approach. _Journal of Experimental Psychology: General_ **117** , 390–394 (1988).

- [23] Garrison, J. R., Bond, R., Gibbard, E., Johnson, M. K. & Simons, J. S. Monitoring what is real: The effects of modality and action on accuracy and type of reality monitoring error. _Cortex_ **87** , 108–117 (2017).

- [24] Simons, J. S., Garrison, J. R. & Johnson, M. K. Brain mechanisms of reality monitoring. _Trends in Cognitive Sciences_ **21** , 462–473 (2017).

- [25] Simons, J. S., Ritchey, M. & Fernyhough, C. Brain mechanisms underlying the subjective experience of remembering. _Annual Review of Psychology_ **73** , 159–186 (2022).

- [26] Simons, J. S., Henson, R. N. A., Gilbert, S. J. & Fletcher, P. C. Separable forms of reality monitoring supported by anterior prefrontal cortex. _Journal of Cognitive Neuroscience_ **20** , 447–457 (2008).

- [27] Dijkstra, N., Kok, P. & Fleming, S. M. Perceptual reality monitoring: Neural mechanisms dissociating imagination from reality. _Neuroscience & Biobehavioral Reviews_ **135** , 104557 (2022).

- [28] Aleman, A. & de Haan, E. H. F. Fantasy proneness, mental imagery and reality monitoring. _Personality and Individual Differences_ **36** , 1747–1754 (2004).

- [29] Aynsworth, C., Nemat, N., Collerton, D., Smailes, D. & Dudley, R. Reality monitoring performance and the role of visual imagery in visual hallucinations. _Behaviour Research and Therapy_ **97** , 115–122 (2017).

- [30] Bentall, R. P., Baker, G. A. & Havers, S. Reality monitoring and psychotic hallucinations. _British Journal of Clinical Psychology_ **30** , 213–222 (1991).

- [31] Anonymous. [blinded for review] (2024). Reference details omitted for double-blind review.

- [32] Alpert, M. The signs and symptoms of schizophrenia. _Comprehensive Psychiatry_ **26** , 103–112 (1985).

- [33] Aleman, A., B¨ocker, K. B. E., Hijman, R., de Haan, E. H. F. & Kahn, R. S. Cognitive basis of hallucinations in schizophrenia: role of top-down information processing. _Schizophrenia Research_ **64** , 175–185 (2003).

- [34] Br´ebion, G., Smith, M. J., Gorman, J. M. & Amador, X. Discrimination accuracy and decision biases in different types of reality monitoring in schizophrenia. _Journal of Nervous and Mental Disease_ **185** , 247–253 (1997).

- [35] Br´ebion, G., Gorman, J. M., Amador, X., Malaspina, D. & Sharif, Z. Source monitoring impairments in schizophrenia: Characterisation and associations with positive and negative symptomatology. _Psychiatry Research_ **112** , 27–39 (2002).

- [36] Brunelin, J. _et al._ Source monitoring deficits in hallucinating compared to nonhallucinating patients with schizophrenia. _European Psychiatry_ **21** , 259–261 (2006).

- [37] Brunelin, J. _et al._ Impaired verbal source monitoring in schizophrenia: An intermediate trait vulnerability marker? _Schizophrenia Research_ **89** , 287–292 (2007).

- [38] Thoresen, C. _et al._ Frontotemporal hypoactivity during a reality monitoring paradigm is associated with delusions in patients with schizophrenia spectrum disorders. _Cognitive Neuropsychiatry_ **19** , 97–115 (2014).

- [39] Corlett, P. R. _et al._ Illusions and delusions: relating experimentally-induced false memories to anomalous experiences and ideas. _Frontiers in Behavioral Neuroscience_ **3** , 53 (2009).

- [40] Venneri, A., Shanks, M. F., Staff, R. T. & Della Sala, S. Nurturing syndrome: A form of pathological bereavement with delusions in Alzheimer’s disease. _Neuropsychologia_ **38** , 213–224 (2000).

- [41] Benedetti, F. _et al._ Reality monitoring and paranoia in remitted delusional depression. _Clinical Neuropsychiatry: Journal of Treatment Evaluation_ **2** , 199–205 (2005).

- [42] Koller, W. N. & Cannon, T. D. Paranoia is associated with impaired novelty detection and overconfidence in recognition memory judgments. _Journal of Abnormal Psychology_ **130** , 273–285 (2021).

- [43] Rossi-Goldthorpe, R. A., Leong, Y. C., Leptourgos, P. & Corlett, P. R. Paranoia, self-deception and overconfidence. _PLOS Computational Biology_ **17** , e1009453 (2021).

- [44] Corlett, P. R. _et al._ Hallucinations and strong priors. _Trends in Cognitive Sciences_ **23** , 114–127 (2019).

- [45] Corlett, P. R., Canavan, S. V., Nahum, L., Appah, F. & Morgan, P. T. Dreams, reality and memory: confabulations in lucid dreamers implicate reality-monitoring dysfunction in dream consciousness. _Cognitive Neuropsychiatry_ **19** , 540–553 (2014).

- [46] Dalla Barba, G. & Kopelman, M. Confabulations and related disorders: We’ve come a long way, but there is still a lot to do! _Cortex_ **87** , 1–4 (2017).

- [47] Schnider, A. Spontaneous confabulation, reality monitoring, and the limbic system — a review. _Brain Research Reviews_ **36** , 150–160 (2001).

- [48] Fotopoulou, A., Conway, M. A. & Solms, M. Confabulation: Motivated reality monitoring. _Neuropsychologia_ **45** , 2180–2190 (2007).

- [49] Alansari, A. & Luqman, H. Large language models hallucination: A comprehensive survey (2025). Preprint, arXiv:2510.06265.

- [50] Huang, L. _et al._ A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. _ACM Transactions on Information Systems_ **43** , 42:1–42:55 (2025).

- [51] Butlin, P. Higher-order representation in AI. _Philosophy and the Mind Sciences_ **7** (2026).

- [52] Wang, J. Z. MIRROR: A hierarchical benchmark for metacognitive calibration in large language models (2026). arXiv:2604.19809.

- [53] Cacioli, J.-P. The metacognitive monitoring battery: A cross-domain benchmark for LLM self-monitoring (2026). arXiv:2604.15702.

- [54] Ackerman, C. _Evidence for limited metacognition in LLMs_ . In _Proceedings of the International Conference on Learning Representations_ (2026). arXiv:2509.21545.

- [55] Pink, M. _et al._ Assessing episodic memory in LLMs with sequence order recall tasks (2024). arXiv:2410.08133.

- [56] Fleming, S. M. & Lau, H. C. How to measure metacognition. _Frontiers in Human Neuroscience_ **8** , 443 (2014).

- [57] Christensen, R. H. B. _ordinal: Regression Models for Ordinal Data_ (2023). URL https://CRAN.R-project.org/package=ordinal. R package version 2023.12-4.

- [58] Galvin, S. J., Podd, J. V., Drga, V. & Whitmore, J. Type 2 tasks in the theory of signal detectability: Discrimination between correct and incorrect decisions. _Psychonomic Bulletin & Review_ **10** , 843–876 (2003).

- [59] Liu, N. F. _et al._ Lost in the middle: How language models use long contexts. _Transactions of the Association for Computational Linguistics_ **12** , 157–173 (2024).

- [60] Kumaran, D. _et al._ Competing biases underlie overconfidence and underconfidence in LLMs. _Nature Machine Intelligence_ (2026).

- [61] Lichtenstein, S., Fischhoff, B. & Phillips, L. D. in _Calibration of probabilities: The state of the art to 1980_ In _Judgment under Uncertainty: Heuristics and Biases_ (eds Kahneman, D., Slovic, P. & Tversky, A.) 306–334 (Cambridge University Press, 1982).

- [62] Groot, T. & Valdenegro-Toro, M. _Overconfidence is key: Verbalized uncertainty evaluation in large language and vision-language models_ . In _Proceedings of the 4th Workshop on Trustworthy Natural Language Processing (TrustNLP 2024)_ , 145–171 (Association for Computational Linguistics, Mexico City, Mexico, 2024).

- [63] Kadavath, S. _et al._ Language models (mostly) know what they know (2022). arXiv:2207.05221.

- [64] Henkel, L. A. & Franklin, N. Reality monitoring of physically similar and conceptually related objects. _Memory & Cognition_ **26** , 659–673 (1998).

- [65] Roediger, H. L. & McDermott, K. B. Creating false memories: Remembering words not presented in lists. _Journal of Experimental Psychology: Learning, Memory, and Cognition_ **21** , 803–814 (1995).

- [66] Alter, A. L. & Oppenheimer, D. M. Uniting the tribes of fluency to form a metacognitive nation. _Personality and Social Psychology Review_ **13** , 219–235 (2009).

- [67] Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S. & Garriga-Alonso, A. _Towards automated circuit discovery for mechanistic interpretability_ . In _Advances in Neural Information Processing Systems_ , Vol. 36, 16318–16352 (Curran Associates, Inc., 2023). arXiv:2304.14997.

- [68] Zheng, Z. _et al._ Attention heads of large language models. _Patterns_ **6** , 101176 (2025).

- [69] Bereska, L. & Gavves, E. Mechanistic interpretability for AI safety — a review. _Transactions on Machine Learning Research_ (2024). URL https://openreview. net/forum?id=ePUVetPKu6.

- [70] Basu, S. _et al._ Interpretability without actionability: mechanistic methods cannot correct language model errors despite near-perfect internal representations (2026). arXiv:2603.18353.

- [71] Nanda, N. _et al._ A pragmatic vision for interpretability. LessWrong (2025). Google DeepMind Mechanistic Interpretability Team. https://www.lesswrong.com/posts/ StENzDcD3kpfGJssR/.

- [72] Kim, B., Hewitt, J., Nanda, N., Fiedel, N. & Tafjord, O. Because we have LLMs, we can and should pursue agentic interpretability (2025). arXiv:2506.12152.

- [73] Pezeshkpour, P. & Hruschka, E. Large language models sensitivity to the order of options in multiple-choice questions (2023). arXiv:2308.11483.

- [74] Nakagawa, S. & Schielzeth, H. A general and simple method for obtaining _R_^2 from generalized linear mixed-effects models. _Methods in Ecology and Evolution_ **4** , 133–142 (2013).

- [75] Pedregosa, F. _et al._ Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_ **12** , 2825–2830 (2011).

- [76] Hautus, M. J. Corrections for extreme proportions and their biasing effects on estimated values of _d_^(_′_) . _Behavior Research Methods, Instruments, & Computers_ **27** , 46–51 (1995).

- [77] Macmillan, N. A. & Creelman, C. D. _Detection Theory: A User’s Guide_ 2 edn (Lawrence Erlbaum Associates, Mahwah, NJ, 2005).
