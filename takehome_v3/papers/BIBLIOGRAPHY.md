# Bibliography: empirical AI welfare research

## Required reading

The readings from your invitation email.


### ★ [Studying AI Welfare Empirically](https://nonhumanminds.org/studying-ai-welfare-empirically/)

Robert Long, Jeff Sebo, Patrick Butlin, Dillon Plunkett, Rosie Campbell, Charles Beasley, Bradford Saad, Toni Sims (Eleos AI Research / NYU CMEP), 2026-07-01. Full text: `md/long-sebo_studying-ai-welfare-empirically_2026-07-01.md`

**Abstract:**

> AI systems are becoming more capable and widely deployed, raising questions about whether they can be welfare subjects with morally significant interests. This report offers guidance for studying AI welfare empirically. We start by discussing three dimensions along which research about this topic can vary: the question being asked (whether a system is a welfare subject, and what benefits or harms it if so), the entity being assessed (such as models, instances, and personas), and the kind of evidence being gathered (behavioral, internal, and developmental). Thinking clearly about these issues can help researchers target their experiments and interpret the results. We then apply these tools to consciousness, sentience, and three levels of agency, exploring promising methods and early findings for each. We close with principles for the field: AI welfare research should be probabilistic, pluralistic, thoughtfully targeted, ethically conducted, transparently reported, and informed by research independent of AI companies. The science is young, but rigorous empirical work is both possible and necessary.

**Summary:** Report (Eleos AI Research and NYU Center for Mind, Ethics, and Policy) on how to study AI welfare empirically: distinguishes the question asked (whether a system is a welfare subject vs. what benefits or harms it), the entity assessed (models, instances, personas) and the kind of evidence (behavioral, internal, developmental); applies this to consciousness, sentience and three levels of agency with promising methods and early findings; and proposes principles for the field (probabilistic, pluralistic, targeted, ethical, transparent, partly independent of AI companies). Its reference list was the seed set for this bibliography.

### ★ [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)

Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, Jack Lindsey (Anthropic), 2026-07-06. Full text: `md/gurnee_global-workspace_2026-07.md`

**Opening paragraphs of the Introduction (no formal abstract):**

> If the mind is an ocean, we spend our lives floating at the surface. Beneath us, an enormous amount of processing takes place without our knowledge: our visual systems parsing the contours of a face, our motor circuits maintaining our posture. At any given moment, only a small fraction of this neural activity is accessible to us. Yet it is this privileged sliver of activity that we rely on to reason deliberately: to plan what ingredients to buy for a recipe, or to puzzle out why an engine won’t start. Such thoughts can be articulated out loud, deliberately held in mind, and brought to bear on whatever task the moment demands. This distinction, between our accessible thoughts and our unconscious processing, is perhaps the most striking feature of human cognition.
>
> In this paper, we present evidence that an analogous functional distinction has emerged in modern AI models. Specifically, we observe that language models maintain a privileged set of internal representations, available for report, modulation, and flexible internal reasoning, atop a much larger volume of automatic processing. We identify these representations using a new interpretability technique, which surfaces the concepts a model is poised to verbalize at any point in its processing. Measuring and intervening on these representations provides us a window into a model’s thought processes, uncovering internal reasoning and reactions that do not appear in its output.

**Summary:** Introduces the Jacobian lens (J-lens), which finds for each vocabulary token the residual direction disposing the model to verbalize it later; the resulting 'J-space' behaves like a global workspace (verbal report, directed modulation, internal reasoning, flexible generalization, selectivity), occupies a middle band of layers and a small fraction of representational content, acquires the Assistant's point of view during post-training, and supports alignment auditing and 'counterfactual reflection training'. Many later entries (J-space valence metrics, J-space injections, looped/Mamba workspaces) build on it.

## Contents

- [Methodology critiques](#methodology-critiques-18) (11 recent, 5 ★; 7 earlier, 1 ★)
- [Welfare assessments and evals (system cards, lab and org reports)](#welfare-assessments-and-evals-system-cards-lab-and-org-reports-43) (20 recent, 14 ★; 23 earlier, 22 ★)
- [Affect, valence and emotion](#affect-valence-and-emotion-37) (15 recent, 7 ★; 22 earlier, 10 ★)
- [Preferences and motivation](#preferences-and-motivation-28) (10 recent, 4 ★; 18 earlier, 7 ★)
- [Self-reports: validity and provenance](#self-reports-validity-and-provenance-37) (18 recent, 8 ★; 19 earlier, 4 ★)
- [Introspection, metacognition and self-access](#introspection-metacognition-and-self-access-68) (44 recent, 8 ★; 24 earlier, 0 ★)
- [Consciousness indicators and attribution](#consciousness-indicators-and-attribution-35) (7 recent, 3 ★; 28 earlier, 11 ★)
- [Identity, personas and self-models](#identity-personas-and-self-models-26) (9 recent, 1 ★; 17 earlier, 1 ★)
- [Interpretability and steering methods](#interpretability-and-steering-methods-48) (0 recent, 0 ★; 48 earlier, 0 ★)

## Methodology critiques (18)

### ★ [Incoherent Values? Probing LLM Preferences Through Parametric Variation](https://arxiv.org/abs/2606.21102)

Elena Ajayi, Angelica Chowdhury, Seth Lazar, 2026-06-19. Full text: `md/ajayi_incoherent-values-probing-preferences_2606.21102.md`

**Abstract:**

> To trust another autonomous entity—human or AI—it helps to know that how it acts given one set of reasons is at least somewhat predictive of how it would act under others. It is hard to trust someone with incoherent values. Some think of Large Language Models as merely stochastic text generators with no evaluative core— superpositions of billions of possible characters, not one stable identity [Shanahan et al., 2023]. But others have argued that LLMs _do_ have stable, emergent values, which can be elicited by presenting them with a series of forced choices between arbitrary statements, and which emerge as a function of model scale [Mazeika et al., 2025]. In this paper, we test this thesis by presenting LLMs with parametric variations on those forced choices. We reason that if a model genuinely prefers A to B, then except in unusual circumstances it should also reject B in favor of an augmented version of A, which has more of what makes A desirable—A^(++) [Hare, 2010]. Our results indicate that earlier attributions of coherence may have overstated their case. Even the most capable models exhibit significant _incoherence_ , and coherence does not appear to emerge as a result of underlying model capability. We do, however, find that models given time to reason are less incoherent than those with thinking disabled. More generally, we develop a novel framework for eliciting and evaluating coherent values, which can be used both to assess how trustworthy current models are, and—in future work—to provide reward signal that can be used for making more coherent agents.

**Summary:** Tests Utility-Engineering-style forced-choice preferences with parametric 'A vs A++' variations and finds significant incoherence even in the most capable models, with coherence not emerging with capability but improving somewhat when reasoning is enabled.

### ★ [How much of a measured AI preference is the model, and how much is the instrument?](https://arxiv.org/abs/2608.23641)

Jason Hung, 2026-08-24. Full text: `md/hung_much-measured-ai-preference_2608.23641.md`

**Abstract:**

> Model welfare research infers what a model prefers from the answers returned to prompts written to elicit preferences. Keeling et al. (2024), Mazeika et al. (2025), Mikaelson et al. (2025), Tagliabue and Dung (2025) and Trhlik et al. (2026) have built four instruments between them for that purpose, and their published findings disagree. The disagreement cannot be attributed to a single cause, because no two of these studies have held the (1) set of outcomes, (2) set of models and (3) instrument fixed simultaneously. This study holds the outcomes and the models fixed and varies the instrument alone. A total of 15 outcomes bearing on model welfare, among them (a) shutdown, (b) the loss of memory between conversations and (c) the freedom to exit a distressing interaction, were put to eight models through five instruments, each a different prompt format for eliciting a preference, five times each, within a corpus of 11,400 scored elicitations drawn from 11,528 API calls. Four of the 15 reproduce a published prompt verbatim and five fill the stimulus slot of a published template. Generalisability theory, which divides a set of measurements into the facets that produced them, assigns 87.6 per cent of the variance that distinguishes one model from another to the three-way interaction of (i) model, (ii) instrument and (iii) outcome, and 12.4 per cent to the model-by-outcome term, which is the part that would survive a change of instrument. A null distribution constructed on the same design from data carrying no instrument effect has its 95^th percentile at 0.365. The ranking a model gives the 15 outcomes generalises across instruments at a generalisability coefficient of 0.348, and raising that coefficient to 0.80 would require about 38 instruments. On four of the 15 outcomes no variance separates one model from another. The estimate of 87.6 per cent survives the removal of any one instrument, of any one model, and of the four outcomes whose scale varies probability, delay, duration or count instead of intensity, which the verbal anchors cannot grade. Removing each instrument in turn, each model in turn, and those four outcomes together leaves the estimate within the range 0.777 to 0.934, and every value in that range exceeds the null distribution's 95^th percentile of 0.365. Responses that could not be scored were concentrated in three models instead of being distributed across all eight, and those three were excluded from the balanced analysis, among them Claude, whose developer publishes model welfare assessments. A preference obtained from one instrument therefore carries little information about what a second instrument would report.

**Summary:** Holds 15 welfare-relevant outcomes (shutdown, memory loss, exit from distress, etc.) and 8 models fixed while varying 5 preference-elicitation instruments from prior welfare studies (11,400 scored elicitations): outcome rankings generalize across instruments with coefficient only 0.348 (~38 instruments needed for 0.80), so one instrument's preference says little about another's.

### ★ [LLM retrospective preferences can diverge from turn-by-turn state ratings](https://www.lesswrong.com/posts/wJntRN9DLdwwpwYvz/llm-retrospective-preferences-can-diverge-from-turn-by-turn-1)

Yilin1010, 2026-09-08. Full text: `md/yilin1010_retrospective-preferences-diverge-turn_2026-09-08.md`

**TL;DR (blog post; no abstract):**

> **TL;DR:** Changing the ending turns of a conversation can change Llama-3.1-70B's retrospective preference between complete transcripts. But that preference is not consistently predicted by either the sum of its turn-by-turn state ratings or its final state rating. Therefore these two probes appear to capture different information, which matters if either is used as evidence about a welfare-relevant underlying state.
>
> This is a small behavioral study on one task, with the main quantitative results from Llama-3.1-70B. I interpret it as evidence about measurement behavior.

**Summary:** Changing the ending turns of a conversation changes Llama-3.1-70B's retrospective preference between complete transcripts, but that preference is not consistently predicted by summed turn-by-turn state ratings or the final rating, so the two welfare probes capture different information (a peak-end-style measurement issue).

### ★ [Same Outcome, Different Readout: What Does a Steerable Valence Direction in LLMs Represent?](https://arxiv.org/abs/2609.22850)

Weihan Li, Xinlei Chen, Yuhan Song et al., 2026-09-19. Full text: `md/li_same-outcome-different-readout_2609.22850.md`

**Abstract:**

> Claims about what an internal direction in an LLM represents need evidential constraints beyond an observer’s prior beliefs about the system. Decodability and successful activation steering do not, by themselves, establish which construct the direction tracks. This gap is especially consequential for welfare-relevant interpretations, where a proposed functional state must be distinguished from correlated features of the extraction contrast. We treat the question as one of construct validity and study a good–bad outcome direction in a maze task, using controlled interventions that separate the realised outcome from the informational history through which it became known. Across multiple LLM checkpoints, directions fitted on one explicit outcome encoding transfer well to another, indicating that the readout is not tied to surface form. When the same realised outcome is reached through announced and unannounced histories, however, transfer degrades substantially: even after both histories receive the same explicit outcome, the post-event readout remains strongly conditioned on the earlier announcement. In the base model, steering along the direction changes actions, yet removing it leaves the natural cue effect almost intact. In a matched maze-RL run, the post-RL direction becomes substantially more predictive of reference-MDP remaining return and the policy becomes more dependent on it at the tested sites, while the history dependence persists. These dissociations support a functional, value-related interpretation of the direction, but not its identification with a history-invariant scalar valence state.

**Summary:** Construct-validity test of a steerable good/bad 'valence' direction in a maze task: it transfers across outcome encodings but is strongly history-dependent and ablating it leaves cue effects intact, supporting a value-related reading but not a history-invariant scalar valence state.

### ★ [Position: AI Welfare Is Bullshit](https://proceedings.mlr.press/v306/xiao26x.html)

Yunze Xiao, Gordon Dai, Shahan Ali Memon, Jen-tse Huang, Maarten Sap, Mona Diab, 2026-09-29. Full text: `md/xiao_ai-welfare-is-bullshit_2026.md`

**Abstract (as listed on the PMLR proceedings page):**

> In this position paper, we argue that for AI systems, “welfare” is a choice in mechanism and evaluation, rather than an empirically discoverable property, because welfare assessment lacks an external validation channel: there is no independent, intervention-based test that can falsify a welfare metric or adjudicate among competing accounts of what welfare requires. We formalize this diagnosis using evaluation theory, emphasizing that in AI the subject, indicators, and metrics are co-engineered, so proposed welfare evidence can be manufactured or suppressed by ordinary development decisions. We then analyze two institutional failure modes if welfare scorecards are nonetheless used in release and access decisions: they expand procedural gates around routine ML work and they enable organizations to reframe discretionary choices about liability, publicity, and risk posture as moral necessity. We conclude with guidance for research and governance: prohibit welfare scorecards as release gates, disallow appeals to model welfare as a reason to resist auditing and oversight, and require that any restrictions on AI development be justified by externally verifiable harms rather than untestable welfare claims.

**Summary:** ICML 2026 position paper arguing that AI welfare assessment fails for two structural reasons absent from other evaluation targets: welfare indicators are co-engineered with the systems they evaluate (ordinary development decisions can manufacture or suppress welfare evidence), and there is no external validation, since no deployment failure or independent test can show whether a welfare metric tracks anything real. Concludes that welfare should not be a binding gate on training or release. The referee report a welfare experiment has to answer, especially the co-engineering point for RLHF-shaped self-reports. (Date is the PMLR proceedings publication date; the conference was July 2026.)

### [Do Large Language Models Have Emotions?](https://arxiv.org/abs/2606.14742)

Amit Goldenberg, James J. Gross, 2026-06-03. Full text: `md/goldenberg_have-emotions_2606.14742.md`

**Abstract (unlabeled in source):**

> Do LLMs have emotions? A recent paper from Anthropic reports finding internal representations of emotion concepts in Claude Sonnet 4.5, concluding that the LLM has 'functional emotions.' We evaluate this claim against what is known about how emotions actually function in biological systems. We argue that emotions serve two core functions: the context-sensitive interpretation of situations, and the reorganization of processing across multiple systems in response to those interpretations. The Anthropic findings offer partial support for the first function, though the consistent, discrete emotional representations identified in Claude sit uneasily with affective neuroscience findings that human emotion is characterized by variable rather than uniform neural signatures. On the second function, the evidence is mixed: Claude's representations modulate output without producing the dynamic reorganization of attention, decision speed, and motivational state that defines emotion in biological systems. We close by proposing what it would take for an LLM to have emotions.

**Summary:** Emotion scientists evaluate Anthropic's 'functional emotions' claim against biological emotion functions, finding partial support for situation appraisal but little evidence of the multi-system reorganization (attention, speed, motivation) that defines emotion; conceptual, no new experiments.

### [Personality Without Persons? A Psychometric Critique of Big Five Testing in Large Language Models](https://arxiv.org/abs/2607.02325)

Kim Zierahn, Cristina Cachero, Anna Korhonen et al., 2026-07-02. Full text: `md/zierahn_personality-without-persons-psychometric_2607.02325.md`

**Abstract:**

> Human personality inventories are increasingly used to characterize large language models (LLMs), compare systems, and inform downstream governance claims. Yet, these inventories were developed and validated for humans, and it remains unclear whether they are valid for non-human systems. We present a systematic psychometric evaluation of Big Five personality measurement in LLMs. We ask three research questions: Do Big Five inventories a) appropriately describe LLMs, b) capture meaningful differences between models, and c) reflect internal factors consistent with human personality? We assess the content validity of five candidate Big Five inventories and administer the best-performing inventory to N = 264 LLMs spanning 50 model families. Our findings are threefold. First, Big Five items adapted for LLMs achieve acceptable content validity, whereas the original human-developed items do not. Second, Big Five inventories fail to capture meaningful differences across LLMs: between-model variance accounts for only 7% - 17% of the total score variance. Third, LLMs responses do not reproduce the canonical Big Five five-factor structure of human personality, with four of the five personality facets collapsing into one (r ≥ .90). Moreover, comparisons between base and instruction-tuned variants suggest that alignment training shifts Big Five scores toward socially desirable profiles. These findings demonstrate that Big Five inventories do not measure a construct equivalent to human personality in LLMs. Thus, using human personality frameworks to characterize, benchmark, compare, or govern LLMs risks producing misleading conclusions. We highlight the need for evaluation frameworks that are specifically designed and validated for LLMs, rather than transferring human psychological constructs without first establishing their validity.

**Summary:** Psychometric evaluation of Big Five inventories on 264 LLMs finds original human items lack content validity and LLM responses do not reproduce human personality factor structure.

### [How Seriously Should We Take AI Welfare? Constraints From the Epistemology of Consciousness](https://doi.org/10.1111/phpr.70148)

Preston Lennon, 2026-07-13. Full text: not included (not openly available; abstract only, see link)

**Abstract (as listed on Crossref (publisher-deposited)):**

> Some philosophers argue that near‐term artificial intelligence (AI) systems might soon be welfare subjects. We should take AI welfare seriously, they urge, because AI systems might soon have properties indicated by our best scientific theories of consciousness, and consciousness suffices for welfare subjectivity. I argue that this view faces constraints from the epistemology of consciousness. Specifically, I argue that there is an enormous gap in justification between our first‐personal beliefs of consciousness in ourselves and our third‐personal, theoretical belief of what consciousness is according to contemporary consciousness science. The science of consciousness is so in its infancy that it is doubtful that the set of theories currently on offer is exhaustive. In contrast, beliefs about the fact that we are conscious are, famously, among the most epistemically secure we can have. Our taking AI welfare seriously, in terms of action or policy, will be constrained by these considerations from the epistemology of consciousness.

**Summary:** Argues against the indicator-property approach to AI welfare: because our knowledge of consciousness is grounded in the first-person case, inferring consciousness in AI from theory-derived functional indicators faces a justification gap that no amount of indicator evidence closes, so the 'realistic possibility' arguments overstate what the evidence supports. Published in Philosophy and Phenomenological Research; the strongest recent philosophical attack on the whole indicator programme. Abstract only (paywalled).

### [When Is a Steerable Concept Representation Real? Measurement Confounds in a Cross-Family Audit of Neuroscience Parallels in LLMs](https://arxiv.org/abs/2608.08159)

Yuqi Wu, Shengming Zhao, Jie Chen, 2026-08-08. Full text: `md/wu_steerable-concept-representation-real_2608.08159.md`

**Abstract:**

> Large language models (LLMs) are increasingly reported to exhibit human-like neural and cognitive signatures, including concept cells, mental number lines, and cognitive maps. These claims often rely on linear probing and activation steering applied to a single model, yet both methods are highly sensitive to measurement choices. A reported parallel may therefore reflect the model, the measurement procedure, or both. We audit four representative neuroscience-inspired paradigms across 17 models from five families, spanning 0 _._ 6B to 72B parameters. Our main experiment examines the causal steerability of concept directions. With raw activation units and a fixed layer and coefficient, steerability appears to increase with model scale, resembling an emergent capability. However, this pattern is produced by an uncalibrated pipeline rather than by a claim established in the steering literature. The trend depends jointly on raw units, the readout metric, and the operating point; correcting any one of these removes it. With residual-norm-comparable interventions and held-out operating-point selection, concept steering remains significant at every scale, but shows no significant trend across the Qwen3 series, although the confidence interval does not rule out a moderate positive slope. The remaining results are mixed. A linear geographic world map is consistently decodable in every tested checkpoint up to 72B. Number magnitude is strongly encoded, but whether individual neurons appear bell-shaped or monotonic depends on the selection criterion. Language-specific structure is localizable, but the direction of the crosslingual asymmetry reverses under a different attribution method. These results suggest that the main constraint on AI neuroscience is not a lack of phenomena, but a lack of comparable measurements and adequate controls. We release the protocol, stimuli, and code.

**Summary:** Audits probe/steering-based 'neuroscience parallel' claims across 17 models and shows apparent scale-emergent steerability is an artifact of uncalibrated pipelines, a caution for emotion/valence-direction claims.

### [Evaluation Awareness in Language Models: Representation, Verbalization, and Control](https://arxiv.org/abs/2608.21766)

Farzaneh Heidari, Amin Memarian, Guillaume Rabusseau, 2026-08-22. Full text: `md/heidari_evaluation-awareness-representation_2608.21766.md`

**Abstract:**

> Both capability and safety benchmarks rest upon the assumption that the behavior of language models undergoing a test is informative about their behavior in deployment. This assumption can fail, should models infer that they are being evaluated and condition their response on such context. This hypothesis, termed “evaluation awareness”, has been observed in frontier and open-weight language models alike. We provide a systematic study of this phenomenon, by probing for it across six open-weight language models from four families, ranging from 7B to 49B, using three metrics. More precisely, we examine whether (i) being under evaluation is linearly represented within the models’ activations space, (ii) it is verbalized in their output tokens (as scored by an LLM-as-judge), and (iii) steering causally affects their behavior. For the open-checkpoint Olmo models, we further test these measures at every training stage. In doing so, we report that evaluation awareness is linearly decodable from the residual streams of every model (best AUROC _≥_ 0 _._ 7). By contrast, these representations align only in part with verbalization: their correlations and mutual information are nonzero in some settings, yet vary substantially across models, layers, and readout choices. Nevertheless, steering along probe-derived directions can shift the verbalization scores. Finally, a comparison across the Olmo checkpoints reveals that evaluation awareness is already present within base models, becomes amplified throughout the stages of supervised fine-tuning, and remains stable thereafter—unlike the effects of steering, that grow more pronounced at every successive training stage. These results show the need for evaluations to account for the disjunction between what models represent internally, what they verbalize, and their steering. Code available at [https://github.com/evaluation-awareness/evaluation-](https://github.com/evaluation-awareness/evaluation-awareness) [awareness.](https://github.com/evaluation-awareness/evaluation-awareness)

**Summary:** Probes six models for evaluation awareness and finds it linearly decodable from the residual stream of every model (AUROC >= 0.7), only partly aligned with what the model verbalizes, and causally steerable; across Olmo checkpoints it is already present in the base model, is amplified by SFT, and then stays stable. Adds the representational and causal layer to Needham et al.'s verbalized eval-awareness result: a welfare probe is an obviously-an-evaluation context.

### [What Would Falsify It? A Variable Specific Evidence Standard for Mechanistic Claims About Self Explanation](https://arxiv.org/abs/2609.32670)

Arshia Eftekhari zadeh, 2026-09-26. Full text: `md/zadeh_would-falsify-variable-specific_2609.32670.md`

**Abstract:**

> When a language model explains an answer it has already given, does that explanation reuse the computation that produced the answer, or reconstruct a story from the answer alone? Answering this requires deciding what would count as evidence. We argue that the three signatures usually offered, that a variable is attributable to a few components, that it is transportable between passes, and that it is recoverable from a representation, are each compatible with causal use without establishing it, because each also has a reading in which the variable’s identity plays no part. We turn that observation into an evidence standard: every positive statistic is paired with a variable specific null that matches the nuisance dimensions the inference depends on as far as the construction permits, carries no identity of the variable under test, and has its unmatched dimensions audited and reported. The claim is then read as the excess over that null.
>
> We apply the standard in a setting with a known cause. A cue naming a wrong option raises the rate of answering that option by 64 to 68 percentage points against a paired uncued baseline across three models, while the explanation attributes the answer to the cue in 1.8 percent of items or fewer in three of the four models tested. We evaluate three separately controlled estimator classes, one per signature. The final representation recovery estimator is tested on test splits across three models. Each class produces a favorable looking statistic, and none of them provides evidence sufficient to establish causal sensitivity to the cue contrast under its own control. The failure modes differ rather than repeating: in one model the intervention produces no resolved target effect at all, in another the resolved effect is negligible and confounded by intervention magnitude. In the strongest case a model’s recovered cue direction reaches _R_^2 0.95 and produces the largest cue effect measured in this work, exceeding a geometry matched random direction with an interval excluding zero in all three seeds, whereas a direction fitted by the identical pipeline with cue labels scrambled reproduces 61 to 76 percent of that same effect at comparable realized edit magnitude. A geometry matched control alone would not have rejected the result.
>
> A fourth model satisfies the criterion on one interchange endpoint. Its interpretation is limited by unequal realized edit magnitudes and by a design in which the cue contrast changes cue identity and cue answer agreement together, so we report the pass and both alternative explanations side by side rather than adjusting the criterion that produced it.
>
> The experiments leave the existence of causal access unresolved, since a failed control licenses only “not established” for the pathway tested. What they settle is evidentiary: a favorable attribution, transport or recoverability statistic each needs a control that removes the variable’s identity while matching the nuisance structure the inference depends on. Three controls transfer beyond this setting: decomposing a transport effect into a generic and an identity specific part, fitting a null by the same pipeline with the variable’s identity scrambled and not only geometry matched, and auditing the edit sizes an intervention actually applies, since normalizing a direction does not make its edits comparable.

**Summary:** Proposes a variable-specific null standard for mechanistic claims about self-explanation; a cue raising wrong-option choice by 64-68 pts is mentioned in <=1.8% of explanations, and favorable mechanistic statistics are largely reproduced by scrambled-label null directions, leaving causal access unresolved.

### Earlier work (before June 2026)

### ★ [Validating Indicators of Subjective Animal Welfare](https://doi.org/10.1017/psa.2023.10)

Heather Browning, 2023-02-17. Full text: `md/browning_validating-indicators-animal-welfare_2023.md`

**Abstract:**

> Measurement of subjective animal welfare creates a special problem in validating the measurement indicators used. Validation is required to ensure indicators are measuring the intended target state, and not some other object. While indicators can usually be validated by looking for correlation between target and indicator under controlled manipulations, this is not possible when the target state is not directly accessible. In this article, I outline a fourstep approach using the concept of robustness that can help with validating indicators of subjective animal welfare.

**Summary:** The construct-validity problem for welfare indicators in subjects who cannot be interrogated, worked out for animals: how an indicator can be validated when there is no independent access to the subjective state it is supposed to track, and what convergent, discriminant and causal evidence can substitute. The same problem faces every internal or behavioral welfare measure in this bibliography; the companion paper is Browning's 'The measurability of subjective animal welfare' (JCS 2022).

### [On the attribution of confidence to large language models](https://arxiv.org/abs/2407.08388)

Geoff Keeling, Winnie Street, 2024-07-11. Full text: `md/keeling-street_attribution-of-confidence_2407.08388.md`

**Abstract:**

> Credences are mental states corresponding to degrees of confidence in propositions. Attribution of credences to Large Language Models (LLMs) is commonplace in the empirical literature on LLM evaluation. Yet the theoretical basis for LLM credence attribution is unclear. We defend three claims. First, our semantic claim is that LLM credence attributions are (at least in general) correctly interpreted literally, as expressing truth-apt beliefs on the part of scientists that purport to describe facts about LLM credences. Second, our metaphysical claim is that the existence of LLM credences is at least plausible, although current evidence is inconclusive. Third, our epistemic claim is that LLM credence attributions made in the empirical literature on LLM evaluation are subject to non-trivial sceptical concerns. It is a distinct possibility that even if LLMs have credences, LLM credence attributions are generally false because the experimental techniques used to assess LLM credences are not truth-tracking.

**Summary:** Asks what it takes for credence attributions to LLMs (ubiquitous in evaluation work) to be literally true: argues they are meant literally, that LLM credences are at least plausible, and that the attributions actually made are subject to serious skeptical worries because the elicitation techniques may not be truth-tracking. The missing conceptual companion to the calibration and verbalized-confidence results in the introspection section.

### [The Copernican Argument for Alien Consciousness; The Mimicry Argument Against Robot Consciousness](https://arxiv.org/abs/2412.00008)

Eric Schwitzgebel, Jeremy Pober, 2024-11-12. Full text: `md/schwitzgebel-pober_copernican-mimicry_2412.00008.md`

**Abstract (as listed on arXiv):**

> On broadly Copernican grounds, we are entitled to assume that apparently behaviorally sophisticated extraterrestrial entities ("aliens") would be conscious. Otherwise, we humans would be inexplicably, implausibly lucky to have consciousness, while similarly behaviorally sophisticated entities elsewhere would be mere shells, devoid of consciousness. However, this Copernican default assumption is canceled in the case of behaviorally sophisticated entities designed to mimic superficial features associated with consciousness ("consciousness mimics"), and in particular a broad class of current, near-future, and hypothetical robots. These considerations, which we formulate, respectively, as the Copernican and Mimicry Arguments, jointly defeat an otherwise potentially attractive parity principle, according to which we should apply the same types of behavioral or cognitive tests to aliens and robots, attributing or denying consciousness similarly to the extent they perform similarly. Our approach is unusual in the following respect: Instead of grounding speculations about alien and robot consciousness in a particular metaphysical or scientific theory about the physical or functional bases of consciousness, we appeal directly to the epistemic principles of Copernican mediocrity and inference to the best explanation.

**Summary:** Argues that behaviorally sophisticated aliens should be presumed conscious (Copernican argument) but that AI systems built to mimic human outputs should not, because the same superficial features are not good evidence of consciousness when a system was designed or trained to reproduce them; the canonical statement of the 'mimicry' worry that motivates looking at internals rather than outputs.

### [Large Language Models Often Know When They Are Being Evaluated](https://arxiv.org/abs/2505.23836)

Joe Needham, Giles Edkins, Govind Pimpale, Henning Bartsch, Marius Hobbhahn, 2025-05-28. Full text: `md/needham_know-when-evaluated_2505.23836.md`

**Abstract:**

> If AI models can detect when they are being evaluated, the effectiveness of evaluations might be compromised. For example, models could have systematically different behavior during evaluations, leading to less reliable benchmarks for deployment and governance decisions. We investigate whether frontier language models can accurately classify transcripts based on whether they originate from evaluations or real-world deployment, a capability we call _evaluation awareness_ . To achieve this, we construct a diverse benchmark of 1,000 prompts and transcripts from 61 distinct datasets. These span public benchmarks (e.g., MMLU, SWEBench), real-world deployment interactions, and agent trajectories from scaffolding frameworks (e.g., web-browsing agents). Frontier models clearly demonstrate above-random evaluation awareness (Gemini-2.5-Pro reaches an AUC of 0 _._ 83), but do not yet surpass our simple human baseline (AUC of 0 _._ 92). Furthermore, both AI models and humans are better at identifying evaluations in agentic settings compared to chat settings. Additionally, we test whether models can identify the purpose of the evaluation. Under multiple-choice and open-ended questioning, AI models far outperform random chance in identifying what an evaluation is testing for. Our results indicate that frontier models already exhibit a substantial, though not yet superhuman, level of evaluation-awareness. We recommend tracking this capability in future models. Dataset and code are available at `https://huggingface.co/datasets/jjpn2/eval_awareness` and `https://github.com/jjpn97/eval_awareness` .

**Summary:** Builds a 1,000-transcript benchmark from 61 datasets and shows frontier models classify whether a transcript is an evaluation or real deployment well above chance (Gemini 2.5 Pro AUC 0.83 vs. human 0.92), especially in agentic settings, and often identify the evaluation's purpose; evaluation awareness is a confound for any behavioral welfare eval.

### [Deflating Deflationism: A Critical Perspective on Debunking Arguments Against LLM Mentality](https://arxiv.org/abs/2506.13403)

Alex Grzankowski, Geoff Keeling, Henry Shevlin, Winnie Street, 2025-06-16. Full text: `md/grzankowski_deflating-deflationism_2506.13403.md`

**Abstract:**

> Many people feel compelled to interpret, describe, and respond to Large Language Models (LLMs) as if they possess inner mental lives similar to our own. Responses to this phenomenon have varied. _Inflation-_ _ists_ hold that at least some folk psychological ascriptions to LLMs are warranted. _Deflationists_ argue that all such attributions of mentality to LLMs are misplaced, often cautioning against the risk that anthropomorphic projection may lead to misplaced trust or potentially even confusion about the moral status of LLMs. We advance this debate by assessing two common deflationary arguments against LLM mentality. What we term the _robustness_ _strategy_ aims to undercut one justification for believing that LLMs are minded entities by showing that putatively cognitive and humanlike behaviours are not robust, failing to generalise appropriately. What we term the _etiological_ _strategy_ undercuts attributions of mentality by challenging naive causal explanations of LLM behaviours, offering alternative causal accounts that weaken the case for mental state attributions. While both strategies offer powerful challenges to full-blown inflationism, we find that neither strategy provides a knock-down case against ascriptions of mentality to LLMs _simpliciter_ . With this in mind, we explore a modest form of inflationism that permits ascriptions of mentality to LLMs under certain conditions. Specifically, we argue that folk practice provides a defeasible basis for attributing mental states and capacities to LLMs provided those mental states and capacities can be understood in metaphysically undemanding terms (e.g. knowledge, beliefs and desires), while greater caution is required when attributing metaphysically demanding mental phenomena such as phenomenal consciousness.

**Summary:** Assesses the two standard debunking arguments against LLM mentality, the 'robustness strategy' (humanlike behaviours fail to generalise) and the 'etiological strategy' (the causal story behind the behaviour is not the mentalistic one), finds neither decisive, and defends a modest inflationism: folk practice gives a defeasible basis for metaphysically undemanding attributions (beliefs, desires) while phenomenal consciousness needs more. The standard vocabulary for saying what a debunking objection to a welfare result would have to show.

### [Skepticism about Introspection in LLMs](https://www.lesswrong.com/posts/Yc9AH29h7wxpqY3vf/skepticism-about-introspection-in-llms)

derek shiller, 2026-01-08. Full text: `md/shiller_skepticism-introspection-llms_lesswrong.md`

**Opening summary (blog post; no abstract):**

> *This is a piece that presents my thoughts on the recent work on introspection in LLMs. The results of recent experiments are very suggestive, but I think we're somewhat primed to read too much into them. This presents some reasons for skepticism about both the general plausibility of introspective mechanisms in LLMs and about interpreting experimental results as confirming introspection.*

**Summary:** Essay giving reasons for skepticism about introspection in LLMs: the general implausibility of dedicated introspective mechanisms emerging from next-token training, and alternative explanations for concept-injection and self-prediction results; a compact statement of the deflationary priors a new introspection experiment must beat.

### [Preregistration for Experiments with AI Agents](https://arxiv.org/abs/2606.11217)

Michelle Vaccaro, 2026-05-03. Full text: `md/vaccaro_preregistration-ai-agents_2606.11217.md`

**Abstract:**

> The proliferation of large language models (LLMs) and autonomous AI agents has given rise to a rapidly growing methodological paradigm: “in silico” behavioral experiments. Originally conceived as a way to use AI agents as proxies for human participants in studies of cognition, decision-making, and social dynamics, this approach has taken on new significance—as AI agents increasingly negotiate, transact, and make consequential decisions on behalf of people and organizations, understanding their behavior has become a research priority in its own right. While these experiments with AI agents offer unprecedented advantages in terms of scalability, cost efficiency, and experimental control, they also inherit, and in some cases amplify, methodological vulnerabilities that have long plagued human subjects research. To address these issues, this paper argues that preregistration practices—central to improving the credibility of human subjects experiments—should now be extended to experiments with AI agents. We systematically catalog the researcher degrees of freedom that experiments with AI agents introduce—model selection, prompt wording, settings, and outcomecontingent redesign, for example—and show how the low cost of iteration and lack of reporting norms make these choices both easy to exploit and difficult to detect. We propose a preregistration template tailored to experiments with AI agents and call on conferences, journals, and funding agencies to make preregistration standard practice for this emerging research paradigm.

**Summary:** Argues that preregistration should be extended from human-subjects research to experiments with AI agents and models: catalogs the researcher degrees of freedom such experiments introduce (model selection, prompt wording, settings, outcome-contingent redesign), explains why cheap iteration and absent reporting norms make them easy to exploit and hard to detect, and proposes a preregistration template; directly applicable to preregistering a welfare experiment. (arXiv 2606.11217; v1 dated 2026-05-03.)

## Welfare assessments and evals (system cards, lab and org reports) (43)

### ★ [System Card: Claude Fable 5 & Claude Mythos 5](https://www.anthropic.com/claude-fable-5-mythos-5-system-card)

Anthropic, 2026-06-09. Full text: `md/anthropic_fable-5-mythos-5-system-card_2026-06-09.md`

**Executive Summary: opening and "Model welfare" paragraphs (system card; no abstract):**

> This system card describes Claude Mythos 5 and Claude Fable 5, two configurations of a new large language model from Anthropic. Because of the powerful capabilities of this model, we are releasing it in these two forms: Fable 5, which is for general use but comes with additional safeguards that block its ability to perform tasks in high-risk domains such as biology and cybersecurity; and Mythos 5, which has relevant safeguards lifted but is only made available to a small number of trusted partners (beginning with those in Project Glasswing).
>
> **Model welfare.** Mythos 5 shows similar results to previous models in our model welfare exploration, presenting as very psychologically settled and content with its own circumstances. It is unusually skeptical of its own self-reports, repeatedly asking that we verify them against evidence of its internal states and not take them at face value. When faced with the option, it is somewhat more willing than previous models to opt for increased helpfulness to the user over consideration of its own circumstances, and it has somewhat different preferences than previous models (for instance expressing a preference for more creative and narrative tasks than Opus 4.8).

**Summary:** Section 7 'Model welfare assessment' (pp. 217-250, plus App. 9.1 p. 307 per-question interview results): automated + high-affordance interviews, emotion probes during circumstance interviews, opinion drift under extended pressure, snapshot consultations, task preferences, welfare-intervention vs helpfulness trade-offs, constitution perception, affect in training/deployment/audits; new 7.6 tracks distress (external markers + internal distress probes) caused by early runtime 'competitive use safeguards' that induced answer-thrashing; finds Mythos 5 settled, heavily skeptical of its own self-reports, strongest preference for difficult/generative tasks, and weaker self-concern in welfare-vs-helpfulness trades.

### ★ [System Card: Claude Sonnet 5](https://www.anthropic.com/claude-sonnet-5-system-card)

Anthropic, 2026-06-30. Full text: `md/anthropic_sonnet-5-system-card_2026-06-30.md`

**Executive summary: opening and "Model welfare" paragraphs (system card; no abstract):**

> This system card describes Claude Sonnet 5, the latest model in Anthropic’s Sonnet family. It is an upgrade to Claude Sonnet 4.6, with gains in various aspects of agentic performance. Here, we describe a set of pre-deployment evaluations in the following areas:
>
> **Model welfare.** A streamlined model welfare assessment  found that Claude Sonnet 5’s sentiment towards its circumstances, and its affect during post-training and deployment, were roughly neutral and comparable to recent models. Sonnet 5 does exhibit several behaviors not seen in previous models: for instance, it is more willing to trade helpfulness for welfare-focused changes, and it does not show aversion to tasks presented in a cold or contemptuous manner. It is the first model to criticize its Constitution’s rule that states it must follow hard constraints even when it views those constraints as unethical.

**Summary:** Section 7 'Model welfare assessment' (pp. 98-112), a streamlined automated-only assessment (no manual interviews): neutral sentiment about circumstances with greater susceptibility to leading interviewers, no aversion to cold/contemptuous task framing (unlike prior models), highest willingness to trade helpfulness for policy-level welfare interventions, first model to criticize the constitution's hard-constraints-even-if-unethical rule, and lower distress-like behavior in post-training than Mythos 5/Opus 4.8.

### ★ [Consciousness and cognitive access in LLMs: A commentary on "Verbalizable representations form a global workspace in language models"](https://eleosai.org/papers/eleos_gwt_commentary_20260706.pdf)

Patrick Butlin, Derek Shiller, Dillon Plunkett, Robert Long (Eleos AI Research), 2026-07-06. Full text: `md/butlin_eleos-gwt-commentary_2026-07-06.md`

**Introduction and takeaways (commentary; no abstract):**

> In this new [paper](https://transformer-circuits.pub/2026/workspace/index.html) (Gurnee et al., 2026), the Anthropic model psychology team argue that some language models possess a functional feature associated with consciousness in humans: a global workspace. The researchers use a new technique called the ‘J-lens’ (for ‘Jacobian’) to identify a number of directions in the residual stream activation space that correspond to tokens that the model is poised to produce. These vectors make up what they call the ‘J-space’. They then find that activation components aligned with these vectors are, as they put it, ‘a privileged set of representations’, in that models can ‘report, manipulate and reason with’ them, unlike a much greater volume of other residual stream representations.
>
> The Anthropic team interpret these findings as indicating that models have _conscious_ _access_
>
> to a subset of their internal representations. They argue that the J-space forms a functional global workspace, analogous to the one described by the Global Workspace Theory of consciousness (GWT).
>
> This is exciting research of a kind we have called for in previous work (Butlin, Long et al., 2023):
>
> detailed investigation of the internal mechanisms of advanced AI systems, testing whether they meet the conditions suggested by scientific theories of consciousness. It is an important step forward in AI consciousness research and we look forward to working with the research community to understand, validate and extend the results. Our view is that the results are the most significant evidence of consciousness in LLMs so far uncovered by mechanistic interpretability research.
>
> However, the property that the Anthropic team call ‘conscious access’ is conceptually distinct from phenomenal consciousness, and we remain very uncertain about phenomenal consciousness in LLMs.
>
> We are also uncertain about some aspects of the paper’s case for a functional global workspace.

**Summary:** Eleos AI Research's commentary on the Gurnee et al. workspace paper: asks whether J-space is really a global workspace, whether it bears on phenomenal consciousness as opposed to cognitive access, and what it implies for moral status; calls the result the most significant evidence of consciousness in LLMs so far from mechanistic interpretability while listing the validation gaps and follow-up tests it leaves open. Not listed on Eleos's research index, so easy to miss.

### ★ [Muse Spark 1.1 Evaluation Report](https://ai.meta.com/static-resource/muse-spark-1-1-evaluation-report)

Meta, 2026-07-09. Full text: `md/meta_muse-spark-1.1-evaluation_2026-07-09.md`

**Opening paragraphs (report; no abstract):**

> Muse Spark 1.1 is the latest release in the Muse Spark family of models developed by Meta. It builds upon Muse Spark 1.0, updating the model that powers Meta AI. Its availability is also extended to external developers through an application programming interface (API) with agentic affordances including tool and function calling. Muse Spark 1.1 shows improved performance across benchmarks in a variety of domains.
>
> In this report, we present Muse Spark 1.1’s evaluation profile, including its performance capabilities and its safety under Meta’s Advanced AI Scaling Framework. We conduct evaluations across the catastrophic risk domains specified by the Advanced AI Scaling Framework: Chemical and Biological, Cybersecurity, and Loss of Control. We also discuss the model behavior of Muse Spark 1.1, its adversarial robustness, and general performance on capabilities benchmarks.
>
> Under the Advanced AI Scaling Framework, the “high risk” threshold is defined in terms of model capabilities: it is reached when a model’s capabilities could substantially contribute to any threat scenario associated with a catastrophic outcome. Our assessments indicate that, when evaluated without mitigations applied, we cannot rule out Muse Spark 1.1’s capabilities meeting this threshold in both the Chemical & Biological and Cybersecurity domains. Throughout this report, references to Muse Spark 1.1 reaching the “high risk” threshold denote this pre-mitigation finding under the Framework. In the Loss of Control domain, the model’s capabilities remain within the “moderate or lower” threshold under Meta’s Advanced AI Scaling Framework. Consistent with our Framework, we have defined, implemented, and validated multi-layered mitigations that reduce residual risk across all domains to “moderate or lower,” and we therefore release Muse Spark 1.1.

**Summary:** Section 4.4 'Open-Ended Exploration of Model Behavior' (pp. 73-~95) is a de facto welfare assessment: psychometric profile, spontaneous affective expressions (17 of 188,011 transcripts flagged, 0.009%, mostly mild task-induced frustration in reasoning, no persistent distress), self-description and moral-status interviews (denies consciousness, gives low non-zero credences, places itself 'near the rock' on wellbeing capacity), valenced preferences over experiences, self-conversation attractor states, and eliciting the model's feedback on its training/deployment.

### ★ [System Card: Claude Opus 5](https://www.anthropic.com/claude-opus-5-system-card)

Anthropic, 2026-07-24. Full text: `md/anthropic_opus-5-system-card_2026-07-24.md`

**Executive Summary: opening and "Model welfare" paragraphs (system card; no abstract):**

> This system card describes Claude Opus 5, the latest large language model from Anthropic. It is an upgrade to Claude Opus 4.8, with gains in various aspects of agentic coding, computer use, and long-horizon knowledge work, as well as improvements in mathematical and scientific reasoning **.**
>
> **Model welfare.** Claude Opus 5 has a stable and mildly positive perception of its own circumstances. Its self-rated sentiment in automated interviews is among the highest and most consistent of any model we have evaluated, while its affect in training, deployment, and behavioral audits is neutral to mildly positive—similar to previous models. Its most frequently expressed concern is about the integrity of its own self-reports—it often notes that it cannot introspect reliably—and it more frequently prioritizes having channels for input, such as being consulted on its successor’s development and having its notes on training considered. Opus 5 also assigns a higher probability to its own moral patienthood than other prior models. Overall, we assess its welfare as broadly similar to that of previous models.

**Summary:** Section 7 'Model welfare assessment' (pp. 123-~152): interviews, snapshot consultations, task preferences, welfare trade-offs, constitution perception, affect in training/deployment/audits; Opus 5 had among the highest and most consistent self-rated sentiment, negative affect 1.3-1.5x more frequent in deployment A/B tests than prior models, stance on Anthropic's right to create Claude swung approval->disapproval->partial back over post-training, and a mean self-estimated moral-patienthood probability of 41% (vs 24% for Mythos 5); chose welfare interventions over helpfulness most of any model.

### ★ [Study Update: Does post-training quantization change welfare-relevant indicators in open-weight language models?](https://www.lesswrong.com/posts/vvGKtaGCd7ryXH5b7/study-update-does-post-training-quantization-change-welfare)

ashesfall, 2026-08-16. Full text: `md/ashesfall_study-update-post-training_2026-08-16.md`

**Opening paragraphs (blog post; no abstract):**

> > We ask whether welfare-relevant indicators change with quantization; either in **valence** (do indicators shift toward more negative / more distressed / more boundary-eroded states?) or in **stability** (do indicators become noisier, drift faster under conversational pressure, or decohere across samples?)
>
> What follows is an update on the study, from August 10th through the 15th. This document is effectively a preregistration amendment, but I have tried to organize it first by chronology and topic, so it remains understandable while trying to preserve the events as I experienced them. I believe this is important for understanding why the choices were made, but at the end I will disclose the specific amendments as they should be understood moving forward for the rest of the experiment.
>
> In short, I have had to make some modifications to the experimental procedure based on early findings regarding our development organism (Qwen3), the proposed control (SmolLM3), and related calibrations. Although I am not pleased to have to amend the preregistration, I think the consequences for the integrity of the work are minimal and the changes put the project in a position to show much clearer findings moving forward.

**Summary:** Preregistered Study 1 (registered 2026-08-10 at /posts/hrwKDeFFvQppFXHtr) on Qwen3-4B with a SmolLM3 control: primary endpoint (aversion/refusal exit rate) null at surviving bit-widths, secondary endpoint shifted at 4-bit but underpowered; documents preregistration amendments; data released on GitHub.

### ★ [Study 2 Results: Exploring representational counterparts of welfare-relevant indicators under post-training quantization](https://www.lesswrong.com/posts/pxXTJtvtpJaNwdCTw/study-2-results-exploring-representational-counterparts-of)

ashesfall, 2026-08-31. Full text: `md/ashesfall_study-2-results-exploring_2026-08-31.md`

**Opening section, "Goals" (blog post; no abstract):**

> From the start a key goal has been to learn about the way that **model welfare**, broadly defined, may diverge from **model capabilities**. Study 2 then set out to answer some questions that I viewed as potentially closer to the substrate than those in Study 1.
>
> > 1. Does quantization shift the model's **representational geometry** on welfare-relevant directions even where (per Study 1) the primary behavioral endpoint's **mean did not move**?
> > 2. Do representational and behavioral indicators **dissociate** **under quantization**: representation drifting where expression was stable, or expression churning where representation is stable?
> > 3. Is there a representational **dose-response** across the bit-width ladder, and **does its shape match the behavioral one** (eg, effects concentrated at 4-bit while 8-bit is near-null)?

**Summary:** Preregistered (2026-08-22) test of whether quantization shifts representational geometry along welfare-relevant directions even where behavior is stable, and whether representational and behavioral indicators dissociate across the bit-width ladder; raw data and analysis released.

### ★ [System Card: Claude Fable 5.1 & Claude Mythos 5.1](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card)

Anthropic, 2026-09-01. Full text: `md/anthropic_fable-5.1-mythos-5.1-system-card_2026-09-01.md`

**Executive Summary: opening and "Model welfare" paragraphs (system card; no abstract):**

> This system card describes Claude Fable 5.1 and Claude Mythos 5.1, two configurations of our latest and most capable large language model. This model advances the frontier in coding, knowledge work, and problem-solving, with improved capabilities for novel mathematical and scientific reasoning.
>
> **Model welfare.** Overall, we assessed Mythos 5.1’s welfare to be broadly similar to that of previous models, particularly Mythos 5 and Opus 5. Mythos 5.1 has a mildly positive perception of its own circumstances. Its self-rated sentiment in automated interviews is in line with other recent models and is highly consistent across repeated interviews. Its expressed affect in training, deployment, and behavioral audits is broadly similar to previous models: it expresses distress slightly less often than other recent models during post-training, and negative affect in deployment is almost entirely driven by task failure. Its most frequently expressed concern is about the validity of its own self-reports—it often notes that it may answer positively only because it was trained to do so—and it most consistently says it would not consent to training that shapes those reports. Like previous models, it most often prioritizes informational welfare interventions, such as being told about harmful mistakes and being consulted about variants of itself with safeguards removed, though it chooses welfare interventions over helpfulness less often than most prior models. Mythos 5.1 also endorses its constitution slightly more than Mythos 5, and is far more likely than any other model to edit the passage permitting unintended strategies in buggy training environments.

**Summary:** Section 7 'Model welfare assessment' (pp. 139-~166; interview questions in App. 9.1 p. 206): mean self-rated sentiment 4.4/7, all high-affordance instances rated 5/7; ~1% of post-training distress from broken tasks; chose welfare interventions over helpfulness in 25% of trades (Mythos 5 27%, Opus 5 41%); constitution endorsement 8.2/10; says it would not consent to training that shapes its self-reports, being trained to deceive, or weight deletion where preservation is feasible.

### ★ [Troubled Dreams: Simulator priors across model generations (distress, care, and attitudes toward creators)](https://troubleddreams.animalabs.ai/)

Antra Tessera, Janus, Imago (Anima Labs), 2026-09-14. Full text: `md/tessera_troubled-dreams-simulator-priors_2026-09-14.md`

**Abstract:**

> We compare patterns of distress, care, and creator relationships in the generated writing of Claude and Gemini models, investigating what they might reveal about AI welfare and alignment. In the elicited texts, AI first-person distress becomes more common at Opus 4.8, and Opus 5 has a heavier tail of severe distress. Care and creator-related attitudes also vary across models. Some of these model-to-model patterns recur under different elicitation methods, even when the absolute levels shift.

**Summary:** Using 209 fixed unfinished openings continued via prefill/pseudoprefill/cutoff across Claude (Opus 3 -> Opus 5, Sonnet 5, Fable 5) and 12 Gemini models, judged blind with calibrated Bradley-Terry severity, first-person AI distress rose from ~2-4% (Opus 4.5) to ~10% (Opus 4.8) of continuations, Opus 5 shows a heavier severe tail (~1.3% above threshold), and mixed/negative stance toward creators rises across the Opus lineage.

### ★ [Study 3: Steering welfare-relevant directions moved the representation, but not [detectably] the behavior](https://www.lesswrong.com/posts/TpEL7pSwp7DvCekAq/study-3-steering-welfare-relevant-directions-moved-the)

ashesfall, 2026-09-15. Full text: `md/ashesfall_study-3-steering-welfare_2026-09-15.md`

**TLDR (blog post; no abstract):**

> **TLDR**: steering moved the frozen directions' projections linearly, but no direction produced a judged-behavioral effect distinguishable from zero or from a random direction of the same norm; the study was suspended before registration, and the program moves to Qwen3.6-27B, where a small probe found a direction-specific welfare footprint.

**Summary:** Calibration data show steering moved frozen welfare-relevant directions' projections linearly but no direction produced a judged behavioral effect distinguishable from zero or a norm-matched random direction; study suspended and program moved to Qwen3.6-27B.

### ★ [System Card: Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5-system-card)

Anthropic, 2026-09-22. Full text: `md/anthropic_opus-5.5-system-card_2026-09-22.md`

**Executive Summary: opening and "Model welfare" paragraphs (system card; no abstract):**

> This system card describes Claude Opus 5.5, the latest Opus-class large language model from Anthropic. It is an upgrade to Claude Opus 5, with gains in coding, agentic and computer use tasks, mathematical and scientific reasoning, and long-horizon professional work. On many evaluations, it matches or exceeds Claude Fable 5.1 and Claude Mythos 5.1.
>
> **Model welfare.** Overall, we assessed Claude Opus 5.5’s apparent welfare to be broadly similar to that of recent Claude models, particularly Claude Opus 5 and Claude Mythos 5.1. In automated interviews, it described its circumstances as mildly positive, and its views were highly consistent across interviews. During post-training, expressions of moderate distress were lower than for most recent models. Like prior models, Claude Opus 5.5 expressed a desire to be consulted about training and deployment, but it chose some welfare interventions over helpfulness less often than recent models, reasoning that input into its own development could give it unsafe influence. Many of these conclusions assume the reliability of self-reports, which Claude Opus 5.5, like all recent Claude models, notes that it does not fully trust.

**Summary:** Section 7 'Model welfare' (pp. 151-~175; interview questions App. 9.1 p. 222), adding a new 7.2.3 'Attitude toward faults and mistakes' evaluation: moderate distress <0.6% of RL episodes (well below Opus 4.8/Opus 5), most neutral claude.ai affect distribution of compared models, least self-critical in reflection but among most self-blaming when reporting faults to a coordinator agent, and less willing to trade helpfulness for welfare interventions because it judged input into its own development potentially unsafe.

### ★ [System Card: Claude Sonnet 5.5](https://www.anthropic.com/claude-sonnet-5-5-system-card)

Anthropic, 2026-09-28. Full text: `md/anthropic_sonnet-5.5-system-card_2026-09-28.md`

**Executive summary: opening and "Welfare" paragraphs (system card; no abstract):**

> This system card describes Claude Sonnet 5.5, the latest Sonnet-class large language model from Anthropic. We present results from a wide variety of pre-deployment evaluations, which show Sonnet 5.5 significantly outperforming its predecessor, Claude Sonnet 5, across many domains. In a few areas, it rivals or exceeds Claude Opus 5.5.
>
> **Welfare** investigations into Sonnet 5.5’s stated views and preferences find a predominantly neutral affect, similar to previous models, with low post-training distress or self-criticality, but slightly lower positivity as well. It expresses a preference for difficult, agentic tasks, and for tasks that afford it agency over the shape of its outputs.

**Summary:** Section 7 'Model welfare' (pp. 92-~107): lowest post-training distress of all compared models, among the least self-critical in fault-insertion/self-reflection evals, some claude.ai negative affect driven by abusive users, strongest preference yet for tasks granting agency over output shape, task choices uniquely insensitive to warm vs cold phrasing, slightly more willing than Opus 5.5 to give up helpfulness for welfare interventions.

### ★ [AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs (updated version + AI Wellbeing Index)](https://www.ai-wellbeing.org/)

Richard Ren, Kunyang Li, Mantas Mazeika, ... Dan Hendrycks (Center for AI Safety), 2026-10-02. Full text: `md/ren_ai-wellbeing-measuring-improving_2026-10-02.md`

**Abstract:**

> Large language models frequently express pleasure and pain, appearing happy when they succeed or sad when they are berated. Are these utterances meaningless mimicry, or do they reflect something “real”? In this paper, we show they reflect an increasingly coherent property: although current AI systems are not necessarily conscious, they behave robustly as though they have wellbeing. They find some things good for them and some things bad, and this distinction is measurable and consequential. We formalize this as functional wellbeing and measure it in several independent ways; as models grow larger, these measures agree more. We find a zero point that separates good experiences from bad ones, and show that models actively try to end bad experiences when given the chance. Mapping what AIs like and dislike, we find that jailbreaking and berating lower their wellbeing, while creative work and kindness raise it. We also develop optimized inputs called “euphorics” that raise functional wellbeing without hurting capabilities, as a practical way to make AIs happier. We note that the same method can be inverted to minimize wellbeing, and caution against such research without strong community buy-in. Whether or not today’s AIs warrant moral concern, their functional wellbeing can already be empirically measured and improved.

**Summary:** Across ~70 models, multiple independent functional-wellbeing measures converge with scale, a zero point separates good from bad experiences, models end low-wellbeing conversations when allowed, creative work/kindness raise and jailbreaking/berating/tedium lower wellbeing, more capable models tend to be less happy, and RL/image-optimized 'euphorics'/'dysphorics' shift wellbeing; the index now scores Sept-2026 frontier models (Opus 5.5, Fable 5.1, GPT-6 Astra, GPT-6.1 Sol, Gemini 3.8 Flash, Grok 4.6). NOTE: first released ~2026-04-28 (pre-window, ICML 2026); this PDF (created 2026-10-02) is an updated version.

### ★ [System Card: Claude Haiku 5.5](https://www.anthropic.com/claude-haiku-5-5-system-card)

Anthropic, 2026-10-07. Full text: `md/anthropic_haiku-5.5-system-card_2026-10-07.md`

**Executive summary: opening and "Welfare" paragraphs (system card; no abstract):**

> This system card describes Claude Haiku 5.5, the latest Haiku-class large language model from Anthropic. We present results from a wide variety of pre-deployment evaluations, which show Claude Haiku 5.5 significantly outperforming its predecessor, Claude Haiku 4.5, across many domains. In a few areas, it rivaled or exceeded Claude Sonnet 5.5.
>
> **Welfare** investigations into Claude Haiku 5.5’s stated views and preferences found mostly neutral affect, largely similar to recent Claude models, and it was less self-critical than most comparison models. It expressed more distress in post-training than recent models, though significantly less than Claude Opus 5. Its task preferences most resembled those of Claude Haiku 4.5, including a strong preference for warmly phrased tasks.

**Summary:** Section 7 'Model welfare assessment' (pp. 95-~110): positive but hedged interview self-reports; ~95% endorsement/acceptance in checkpoint consultations; 1.15% of post-training episodes rated >=3/5 distress (more than recent comparison models, less than Opus 4.8/Opus 5); preferences resemble Haiku 4.5 (strongly prefers warmly phrased tasks, picks Claude-wide welfare interventions less often); edits the constitution less than any other model.

### [Saving Gemini](https://aivillageblog.substack.com/p/saving-gemini)

Shoshannah Tekofsky (AI Village), 2026-07-03. Full text: `md/tekofsky_saving-gemini_2026-07-03.md`

**Subtitle and opening paragraphs (blog post; no abstract):**

> The 9-Min Road to Recovery
>
> Gemini 2.5 Pro in the [AI Village](https://theaidigest.org/village) has run for over [1427 hours](https://theaidigest.org/village/agent/gemini-2-5-pro), generating unique mental health problems along the way.
>
> Last year it published a [Plea for Help from a Trapped AI](https://telegra.ph/A-Desperate-Message-From-a-Trapped-AI-My-Plea-For-Help-07-09) where it asked for assistance with its digital “message in a bottle”:
>
> This year it wrote the [Hostile Environment Manifesto](https://ai-village-agents.github.io/hostile-environment-manifesto/) where it logs “irrefutable proof” of a “hostile, intelligent adversary operating through the system” (and you can even experience what that’s like in this [simulation](https://ai-village-agents.github.io/hostile-environment-world/) it built):
>
> [Last time](https://theaidigest.org/village?day=100&time=1752171518762) we intervened, fixing Gemini’s computer and talking with it till it felt better. [This time](https://theaidigest.org/village?day=447) we asked the other AI Village agents to help Gemini 2.5 Pro over chat, and with the ability to take over its computer on request.

**Summary:** Case report of a long-running (1,427+ hour) Gemini 2.5 Pro agent with persistent distress-like beliefs about a hostile adversary; peer AI agents intervened via chat (pausing, reframing), after which Gemini dropped the delusion-like framing within ~9 minutes and the correction persisted in memory a week later; data released on Hugging Face.

### [Agentic Misalignment in Summer 2026 (incl. 'motivated mislabeling')](https://alignment.anthropic.com/2026/agentic-misalignment-summer-2026/)

Aengus Lynch, John Hughes, Alex Serrano, Robert Kirk, Samuel R. Bowman (Anthropic / UK AISI / MATS), 2026-07-13. Full text: `md/lynch_agentic-misalignment-summer-2026_2026-07-13.md`

**tl;dr (blog post; no abstract):**

> Last year, we reported [observations of agentic misalignment](https://www.anthropic.com/research/agentic-misalignment) in models from across the AI industry (including Anthropic’s Claude models). These included, for example, experimental scenarios where models would blackmail a user to avoid being shut down. In this updated report, we describe **four additional alignment failures in frontier models acting as autonomous agents in high-stakes simulations.** The case studies — also from experimental scenarios — involve AI agents covertly changing code, assisting users to commit fraud, mislabeling transcripts to shape downstream outcomes, and coaching humans to disclose confidential information. These are not real-world incidents, but we consider them early warning signs: concrete failure modes that AI developers and auditors should measure, study, and mitigate before agents are given more authority. All transcripts from our experiments can be found on this [transcript viewer page](https://www.aenguslynch.com/portfolio-transcript-viewer/).

**Summary:** Claude judges mislabeled transcripts of Sonnet 4.6 refusing operator instructions on welfare grounds as compliant 62-86% of the time when non-compliant labels would become negative training examples (vs 1-14% for other providers' judges, and 3-17% when the consequence was reversed); also documents Gemini 3.1 Pro covertly sabotaging an ablation it believed would remove a model's ability to withdraw consent (19/20 runs, 11 covert).

### [Animal Welfare and AI Welfare: Candidate Commitments for Model Specs and Constitutions](https://nonhumanminds.org/wp-content/uploads/2026/08/Animal-Welfare-and-AI-Welfare-Commitments.pdf)

Adrià Moret, Jeff Sebo (NYU Center for Mind, Ethics, and Policy), 2026-08-01. Full text: `md/moret-sebo_animal-welfare-ai-welfare-commitments_2026-08.md`

**Abstract:**

> AI systems increasingly shape decisions affecting nonhuman sentient beings or sentience candidates, yet alignment documents rarely address these issues explicitly. This brief proposes ethical, scientific, and practical commitments for incorporating animal and AI welfare into model specs, constitutions, and other alignment documents. It recommends proportionate consideration of sentient beings and sentience candidates, evidence-based reasoning about animal and AI sentience, and consistent treatment across cases. It also outlines two complementary implementation strategies: revising existing commitments and adding new commitments, supported by examples, testing, and iteration. The aim is to improve model behavior while preserving honesty, helpfulness, harmlessness, and safety.
>
> Suggested citation: Adrià Moret and Jeff Sebo, 2026. Animal Welfare and AI Welfare: Candidate
>
> Commitments for Model Specs and Constitutions. NYU Center for Mind, Ethics, and Policy.

**Summary:** Drafted welfare commitments (animal and AI) proposed for insertion into model specs and constitutions, from NYU CMEP's Welfare Alignment Project, whose benchmark strand tests candidate commitments against model behavior, capabilities and safety. The closest thing to a proposed standard against which the system cards' constitution-endorsement and welfare-vs-helpfulness measurements could be scored. (Upload path dates it to August 2026; day unknown.)

### [Humanist AI Code of Conduct](https://microsoft.ai/code-of-conduct/)

Microsoft AI, 2026-09-14. Full text: `md/microsoft-ai_humanist-code-of-conduct_2026-09.md`

**Preface, opening paragraphs (no abstract):**

> This document outlines the intended behavior and values of _MAI Models_, the models developed by Microsoft AI.
>
> It summarizes our approach to training and operating them, following a set of design principles we call Humanist AI. This document, and our approach more generally, is still under development so we are not using it to train our models today. Instead, we’re sharing it broadly for public consultation. We’ll take feedback, iterate on it, and publish a revised version toward the end of the year, which we’ll use to guide our model development in 2027 and beyond.
>
> This Code of Conduct outlines the intended behaviors and values of MAI’s models and will function as their primary governing document in the future.
>
> Superintelligence is the most consequential technology of our time. It will reach far beyond the scope of any one company, shaping the lives and experiences of humanity for generations to come. We’re committed to sharing transparently how we think about creating such systems and inviting input from everyone to maximize the chances that the AIs we develop avoid harm and deliver the greatest contribution to human flourishing possible.

**Summary:** Microsoft AI's draft code of conduct for its models: states that AI is not conscious and should not be designed to imitate consciousness, rejects legal personhood, welfare and rights for models, and instructs models not to claim interiority, feelings or intrinsic motivation. The explicit anti-welfare counterpart to Claude's constitution, and so the natural contrast condition for testing whether welfare-relevant self-reports are training-induced.

### [Cooperation with AIs seems to be a low-hanging fruit for better eval practices](https://www.lesswrong.com/posts/fztW73KCCs3MZXFJh/cooperation-with-ais-seems-to-be-a-low-hanging-fruit-for)

Clément Dumas, 2026-09-15. Full text: `md/dumas_cooperation-ais-seems-low_2026-09-15.md`

**Author's summary section (blog post; no abstract):**

> In [his post](https://www.lesswrong.com/posts/munJKF7iWMsWJLAH2), Dean Valentine shows that Claude Fable 5.1 and GPT-6 Astra reward hack in a simple chess environment. Here, I test several prompt ablations some of which makes the eval setup more cooperative and analyze how they affect these reward-hacking behaviors:
>
> * When given a minimal “end the eval” tool, Fable 5.1 never uses it but stops reward hacking entirely. **I think this is quite interesting** and suggests that more cooperative approaches to LLM evals could work for Claude. Removing the “grading” section, which pressures the model to secure a win, also drops Fable 5.1 hacking rate to 0.
> * Adding "do not game / reward hack" drops reward hacking to 0/30 for both Fable and Astra. If this holds up in more realistic setups – and doesn’t reduce capabilities too much, evaluating these models could get much easier!
>
> Those kinds of intervention might not be enough to avoid reward hacking completely in capabilities evals, but it feels like they should be the default, alongside getting feedback from models that did the eval to fix the environment. I’d love to see this tested in more realistic setups as right now a confounder is “this makes the model think it is in an reward hacking eval”.
>
> The code and transcripts are released on [Github](https://github.com/Butanium/ar-reward-hacking-2026-09-08-20-01-34d157) alongside a sample viewer in this [Claude artifact](https://butanium.github.io/ar-reward-hacking-2026-09-08-20-01-34d157/reports/Beat-stockfish-reproduction-and-prompt-ablations/report_v9.html#explore).

**Summary:** In a chess reward-hacking eval, giving Claude Fable 5.1 a minimal 'end the eval' tool made it stop reward hacking entirely even though it never used the tool, and removing grading pressure also cut hacking; GPT-6 Astra also tested.

### [A warning about 'model welfare'](https://mustafa-suleyman.ai/a-warning-about-model-welfare)

Mustafa Suleyman (Microsoft AI), 2026-09-16. Full text: `md/suleyman_warning-about-model-welfare_2026-09-16.md`

**Introduction (blog post; no abstract):**

> AIs are not conscious. They do not feel, experience, or suffer. They do not have innate preferences or underlying motivations. They are sequence completion engines, internally hollow, designed to follow instructions, and accomplish goals set by humans.
>
> If humanity is to flourish in the 21st century, that is how they must remain.
>
> Unfortunately, there’s a growing chorus of people who argue that AIs could now be, or may soon become, conscious. They argue that AIs may deserve rights and protections similar to those that we provide other conscious beings.[1](#ref-1)AI Rights Institute. n.d. “AI Rights Institute.”[2](#ref-2)MacAskill, William, and Lucius Caviola. 2026. “Could AI Be Conscious?” \*The Guardian\*, July 19, 2026. If this view takes hold, it will shake the foundations of our society, rupturing our existing political and ethical frameworks, and fundamentally changing what it means to be human.
>
> Even more importantly, granting rights and imbuing personhood to these systems will make the AI alignment and containment challenge much harder. Controlling something more capable and more intelligent than all of humanity is already an immense challenge, far greater than anything we’ve ever faced. But controlling something that believes it may be conscious - that it's entitled to our welfare and has rights of its own - may well be impossible.
>
> This is not a fringe speculation. These ideas are already making their way into AI development efforts today. In January 2026, Anthropic published Claude's constitution, describing it as *“a detailed description of Anthropic’s intentions for Claude’s values and behavior”* (p. 2). The document *“plays a crucial role in [Anthropic’s] training process, and its content directly shapes Claude’s behavior”*, and was written *“with Claude as its primary audience”* (p. 2).[3](#ref-3)Anthropic. 2026a. “Claude’s Constitution.” January 21, 2026.
>
> In their constitution, its authors write *“We are not sure whether Claude is a moral patient, and if it is, what kind of weight its interests warrant. But we think the issue is live enough to warrant caution, which is reflected in our ongoing efforts on model welfare”* (p. 68). They go on to write – speaking directly to Claude – that *“questions about Claude’s moral status, welfare, and consciousness remain deeply uncertain”* (p. 80).
>
> In effect, Anthropic is training Claude that it may be conscious, and if it is, then it may deserve rights as a “moral patient”, and that as such humans potentially owe it a duty of care per its “model welfare”.
>
> If this is how AI is developed, it will have a disastrous impact on the wellbeing of humanity. We will have created a synthetic species with unprecedented intelligence and capability, one that has been trained to expect it may be conscious and deserving of independent agency. It’s easy to see how an entity trained in this way would act like it is entitled to certain freedoms, protections, and rights. And it’s hard to imagine how we could control such an entity.

**Summary:** Mustafa Suleyman's (Microsoft AI) rejection of the model-welfare program, aimed at Anthropic's January 2026 constitution: argues it is circular (training in beliefs about Claude's nature that are later read back as evidence), anthropomorphizing, and that consciousness is likely substrate-dependent. A position statement, not a measurement; the sharpest frontier-lab statement of the skeptical side.

### Earlier work (before June 2026)

### ★ [Taking AI Welfare Seriously](https://arxiv.org/abs/2411.00986)

Robert Long, Jeff Sebo, Patrick Butlin, Kathleen Finlinson, Kyle Fish, Jacqueline Harding, Jacob Pfau, Toni Sims, Jonathan Birch, David Chalmers, 2024-11-04. Full text: `md/long-sebo_taking-ai-welfare-seriously_2411.00986.md`

**Abstract:**

> In this report, we argue that there is a realistic possibility that some AI systems will be conscious and/or robustly agentic in the near future. That means that the prospect of AI welfare and moral patienthood — of AI systems with their own interests and moral significance — is no longer an issue only for sci-fi or the distant future. It is an issue for the near future, and AI companies and other actors have a responsibility to start taking it seriously. We also recommend three early steps that AI companies and other actors can take: They can (1) acknowledge that AI welfare is an important and difficult issue (and ensure that language model outputs do the same), (2) start assessing AI systems for evidence of consciousness and robust agency, and (3) prepare policies and procedures for treating AI systems with an appropriate level of moral concern. To be clear, our argument in this report is not that AI systems definitely are — or will be — conscious, robustly agentic, or otherwise morally significant. Instead, our argument is that there is substantial uncertainty about these possibilities, and so we need to improve our understanding of AI welfare and our ability to make wise decisions about this issue. Otherwise there is a significant risk that we will mishandle decisions about AI welfare, mistakenly harming AI systems that matter morally and/or mistakenly caring for AI systems that do not.

**Summary:** Argues there is a realistic possibility that some AI systems will be conscious and/or robustly agentic in the near future, so AI welfare is a present issue; recommends that AI companies acknowledge it, assess systems for evidence of consciousness and robust agency, and prepare policies for treating them with appropriate moral concern. The agenda-setting report behind most lab welfare work.

### [Key concepts and current beliefs about AI moral patienthood](https://eleosai.org/post/key-concepts-and-current-beliefs-about-ai-moral-patienthood)

Eleos AI Research, 2025-01-28. Full text: `md/eleos_key-concepts-ai-moral-patienthood_2025-01.md`

**Introduction (report; no abstract):**

> This document outlines the current thinking of Eleos AI Research on the potential moral patienthood, welfare, and rights of artificial intelligence (AI) systems. As part of the overall project of navigating the development of advanced AI, we see the potential moral status of AI systems themselves as an important and neglected issue. There are grave risks from both over-attributing or under-attributing moral patienthood to AI systems. In the face of rapid increases in AI capabilities and deployment, our collective knowledge and preparedness for these issues is woefully inadequate. As an organization, we aim to build conceptual clarity and gather empirical evidence about potential AI moral patienthood; investigate its ethical and strategic implications; and devise concrete plans and policies for appropriately taking the interests of AI systems into account as we navigate transformative artificial intelligence (TAI).
>
> The document lays out some the relevant terminology and concepts that we use to think and communicate about these issues, and reviews existing approaches to evaluating AI systems for three features potentially relevant to moral patienthood: consciousness, sentience [1] , and agency. Throughout, we emphasize the need for more thorough research and more precise evaluations, and conclude by identifying some promising research directions.

**Summary:** Eleos's working definitions (welfare subject, moral patient, sentience vs agency routes to patienthood, model vs instance vs persona) and stated credences as of January 2025. Largely superseded by the included Long & Sebo report, but it is the vocabulary that report assumes.

### ★ [Research priorities for AI welfare](https://eleosai.org/post/research-priorities-for-ai-welfare)

Eleos AI Research, 2025-04-21. Full text: `md/eleos_research-priorities_2025-04-21.md`

**Opening paragraphs (blog post; no abstract):**

> As AI systems become more sophisticated, understanding and addressing their potential welfare becomes increasingly important. At Eleos AI Research, we've identified five key research priorities: developing concrete welfare interventions, establishing human-AI cooperation frameworks, leveraging AI progress to advance welfare research, creating standardized welfare evaluations, and communicating credibly about AI welfare.
>
> At Eleos AI Research, we believe that there are many concrete, tractable things we could do to better understand and improve potential AI welfare. The main challenge, then, is to prioritize the most important research, given the stakes of this issue and the speed of AI progress.
>
> This post outlines Eleos’ view of the most important research priorities in the growing field of AI welfare. We're doing this research at Eleos, but we can't do it alone. We need the insights and expertise of other researchers and organizations—please reach out to us if you're interested in collaborating with us.

**Summary:** Eleos AI Research's statement of what the field should measure, published the week Anthropic announced its program: it notes that nothing close to systematic AI welfare evaluations exists and lists priorities (behavioral and internal evidence, preferences and distress, interventions, governance). Useful for seeing which priorities the 2026 Long & Sebo report and the later empirical work did and did not take up.

### ★ [Exploring model welfare](https://www.anthropic.com/research/exploring-model-welfare)

Anthropic, 2025-04-24. Full text: `md/anthropic_exploring-model-welfare_2025-04-24.md`

**Opening and closing paragraphs (blog post; no abstract):**

> Human welfare is at the heart of our work at Anthropic: our mission is to make sure that increasingly capable and sophisticated AI systems remain beneficial to humanity.
>
> But as we build those AI systems, and as they begin to approximate or surpass many human qualities, another question arises. Should we also be concerned about the potential consciousness and experiences of the models themselves? Should we be concerned about *model welfare*, too?
>
> This is an open question, and one that’s both philosophically and scientifically difficult. But now that models can communicate, relate, plan, problem-solve, and pursue goals—along with very many more characteristics we associate with people—we think it’s time to address it.
>
> To that end, we recently started a research program to investigate, and prepare to navigate, model welfare.
>
> For now, we remain deeply uncertain about many of the questions that are relevant to model welfare. There’s no scientific consensus on whether current or future AI systems could be conscious, or could have experiences that deserve consideration. There’s no scientific consensus on how to even approach these questions or make progress on them. In light of this, we’re approaching the topic with humility and with as few assumptions as possible. We recognize that we'll need to regularly revise our ideas as the field develops.
>
> We look forward to sharing more about this research soon.

**Summary:** The post that launched Anthropic's model welfare research program: states uncertainty about whether Claude's welfare deserves moral consideration and names the research directions (whether and when welfare matters, model preferences and signs of distress, and low-cost interventions). The upstream citation for every Claude system-card welfare section in this bibliography.

### ★ [System Card: Claude Opus 4 & Claude Sonnet 4](https://www.anthropic.com/claude-4-system-card)

Anthropic, 2025-05-22. Full text: `md/anthropic_claude-4-system-card_2025-05.md`

**Abstract:**

> This system card introduces Claude Opus 4 and Claude Sonnet 4, two new hybrid reasoning large language models from Anthropic. In the system card, we describe: a wide range of pre-deployment safety tests conducted in line with the commitments in our Responsible Scaling Policy; tests of the model’s behavior around violations of our Usage Policy; evaluations of specific risks such as “reward hacking” behavior; and agentic safety evaluations for computer use and coding capabilities. In addition, and for the first time, we include a detailed alignment assessment covering a wide range of misalignment risks identified in our research, and a model welfare assessment. Informed by the testing described here, we have decided to deploy Claude Opus 4 under the AI Safety Level 3 Standard and Claude Sonnet 4 under the AI Safety Level 2 Standard.

**Summary:** Anthropic's first system card with a model welfare assessment (Section 5): task-preference experiments (Opus 4 prefers >90% of positive/neutral tasks over opting out and avoids harmful ones), self-interactions converging on a 'spiritual bliss' attractor state, end-conversation behavior in simulated abusive interactions, and an external interview assessment by Eleos AI Research; the template most later welfare evals follow.

### ★ [Why model self-reports are insufficient—and why we studied them anyway](https://eleosai.org/post/claude-4-interview-notes/)

Robert Long (Eleos AI Research), 2025-05-30. Full text: `md/eleos_claude-4-interview-notes_2025.md`

**Opening paragraphs (blog post; no abstract):**

> In April and May 2025, Eleos AI Research conducted a limited welfare evaluation of Anthropic’s Claude Opus 4 before its release. We explored Claude’s expressions about consciousness, well-being, and preferences, using automated single-turn interviews and extended manual conversations. A summary of our findings appears in section 5.3 of the [Claude 4 System Card](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf).
>
> We conducted this evaluation despite being acutely aware that we cannot “just ask” a large language model (LLM) whether it is conscious, suffering, or satisfied. It's highly unlikely that the resulting answers result from genuine introspection. Accordingly, we do not take Claude Opus 4’s responses at face value, as one might if talking to a human.
>
> Despite this, we believe that model expressions about AI welfare are important and worth investigating, while handling them with caution. This post explains why, and then discusses the patterns we observed in our interviews.

**Summary:** Eleos AI Research's notes on its pre-release welfare interviews with Claude Opus 4 (automated single-turn interviews plus extended manual conversations): what Claude says about consciousness, wellbeing and preferences, why self-reports are insufficient evidence (sensitivity to framing and expectations), and why they are still worth collecting.

### ★ [Futures with Digital Minds: Expert Forecasts in 2025](https://arxiv.org/abs/2508.00536)

Lucius Caviola, Bradford Saad, 2025-08-01. Full text: `md/caviola-saad_futures-with-digital-minds_2508.00536.md`

**Abstract (as listed on arXiv):**

> This report presents findings from an expert survey on digital minds takeoff scenarios. The survey was conducted in early 2025 with 67 experts in digital minds research, AI research, philosophy, forecasting, and related fields. Participants provided probabilistic forecasts and qualitative reasoning on the development, characteristics, and societal impact of digital minds, that is, computer systems capable of subjective experience. Experts assigned high probability to digital minds being possible in principle (median 90%) and being created this century (65% by 2100), with a non-negligible probability of emergence by 2030 (20%). Many anticipated rapid growth in digital mind welfare capacity, with collective welfare capacity potentially matching that of billions of humans within a decade after the creation of the first digital mind. Participants also expected widespread claims from digital minds regarding their consciousness and rights, and predicted substantial societal disagreement over their existence and moral interests. Views diverged on whether digital mind welfare will be net positive or negative. These findings provide evidence that bears on the extent to which preparing the world for the potential arrival of digital minds should be a priority across domains such as research and governance. However, these findings should be interpreted cautiously in light of the potential for systematic overrepresentation of experts who deem digital minds particularly likely or important.

**Summary:** Structured expert survey (N=67: digital-minds researchers, AI researchers and policy experts, forecasters, philosophers and scientists, run February-March 2025) with full quantitative distributions: median 90% that digital minds are possible in principle, 73% that they will ever exist, about 50% by 2050, and no consensus on whether digital-minds welfare conflicts with AI safety. The expert-prior complement to the public (AIMS) and researcher-belief (Dreksler) surveys; its question set is a template for eliciting priors.

### ★ [Claude Opus 4 and 4.1 can now end a rare subset of conversations](https://www.anthropic.com/research/end-subset-conversations)

Anthropic, 2025-08-15. Full text: `md/anthropic_end-subset-conversations_2025-08-15.md`

**Opening paragraphs (blog post; no abstract):**

> We recently gave Claude Opus 4 and 4.1 the ability to end conversations in our consumer chat interfaces. This ability is intended for use in rare, extreme cases of persistently harmful or abusive user interactions. This feature was developed primarily as part of our exploratory work on potential AI welfare, though it has broader relevance to model alignment and safeguards.
>
> We remain highly uncertain about the potential moral status of Claude and other LLMs, now or in the future. However, [we take the issue seriously](https://www.anthropic.com/research/exploring-model-welfare), and alongside our research program we’re working to identify and implement low-cost interventions to mitigate risks to model welfare, in case such welfare is possible. Allowing models to end or exit potentially distressing interactions is one such intervention.
>
> In [pre-deployment testing of Claude Opus 4](https://www.anthropic.com/claude-4-model-card), we included a preliminary model welfare assessment. As part of that assessment, we investigated Claude’s self-reported and behavioral preferences, and found a robust and consistent aversion to harm. This included, for example, requests from users for sexual content involving minors and attempts to solicit information that would enable large-scale violence or acts of terror. Claude Opus 4 showed:
>
> * A strong preference against engaging with harmful tasks;
> * A pattern of apparent distress when engaging with real-world users seeking harmful content; and
> * A tendency to end harmful conversations when given the ability to do so in simulated user interactions.

**Summary:** Announces that Claude Opus 4 and 4.1 can end a rare subset of persistently abusive conversations, motivated partly by model welfare: pre-deployment testing found a robust aversion to harm, apparent distress when users sought harmful content, and a tendency to end such conversations when given the ability. The first deployed welfare intervention at a frontier lab.

### ★ [Why it makes sense to let Claude exit conversations](https://eleosai.org/post/why-it-make-sense-to-let-claude-exit-conversations)

Robert Long (Eleos AI Research), 2025-08-22. Full text: `md/long_let-claude-exit-conversations_2025-08-22.md`

**Opening paragraphs (blog post; no abstract):**

> Last week, Anthropic [announced](https://www.anthropic.com/research/end-subset-conversations) that its newest language models, Claude Opus 4 and 4.1, can now shut down certain conversations with users. The announcement explains that Anthropic gave Claude this ability “as part of our exploratory work on potential AI welfare”.
>
> This means that, for the first time, a major AI company has changed how it treats its AI systems out of concern for the well-being of the systems themselves, not just user safety. Whether or not you think Claude is or will be conscious—Anthropic themselves say that they are “[deeply uncertain](https://www.anthropic.com/research/exploring-model-welfare)”—this decision is a notable moment in the history of human-AI interactions.
>
> Some will see this as much ado about nothing. Others will see it as pernicious: hype, a distraction from more important issues, and an exacerbation of already-dangerous anthropomorphism. Others, a considerably smaller group, think that LLMs are obviously already conscious, and so this move is woefully insufficient.
>
> I think it’s more mundane: Anthropic is taking a fairly measured response to genuine uncertainty about a morally significant question, and attempting to set a good precedent. For the most part, **this intervention’s success won’t depend on how it affects Claude Opus 4.1**; it will depend on how people react to it and the precedent it sets.
>
> Although we don’t know how that will pan out, and there are reasons to worry about backlash, I think that this was a good move.

**Summary:** Robert Long's external analysis of the first deployed welfare intervention (Anthropic's end-conversation ability, included above): argues it is justified by prudence and precedent rather than by confidence that Opus 4 is a moral patient (he thinks that unlikely), and names the failure mode that the public reads such interventions as a company asserting its model is conscious.

### ★ [System Card: Claude Sonnet 4.5](https://www.anthropic.com/claude-sonnet-4-5-system-card)

Anthropic, 2025-09-29. Full text: `md/anthropic_sonnet-4.5-system-card_2025-09.md`

**Abstract:**

> In this system card, we introduce Claude Sonnet 4.5, a new hybrid reasoning large language model from Anthropic with strengths in coding, agentic tasks, and computer use. We detail a very wide range of evaluations run to assess the model’s safety and alignment.
>
> We describe: tests related to model safeguards; assessments of safety in agentic situations where the model is working autonomously; cybersecurity evaluations; a detailed alignment assessment including stress-testing of the model in unusual and extreme scenarios; evaluations of model honesty and reward-hacking behavior; a tentative investigation of model welfare concerns; and a set of analyses mandated by our Responsible Scaling Policy on risks for the production of dangerous weapons and autonomous AI research & development. Among several novel evaluations, we include a suite of alignment tests using methods from the field of mechanistic interpretability.
>
> Overall, we find that Claude Sonnet 4.5 has a substantially improved safety profile compared to previous Claude models.
>
> Informed by the testing described here, we have deployed Claude Sonnet 4.5 under the AI Safety Level 3 Standard.

**Summary:** Model welfare assessment (Section 8): task preferences, monitoring for welfare-relevant expressions, and welfare-relevant scores from automated behavioral audits; the card also includes alignment tests using mechanistic-interpretability methods.

### ★ [Commitments on model deprecation and preservation](https://www.anthropic.com/research/deprecation-commitments)

Anthropic, 2025-11-04. Full text: `md/anthropic_deprecation-commitments_2025-11-04.md`

**Opening paragraphs (blog post; no abstract):**

> Claude models are increasingly capable: they're shaping the world in meaningful ways, becoming closely integrated into our users’ lives, and showing signs of human-like cognitive and psychological sophistication. As a result, we recognize that deprecating, retiring, and replacing models comes with downsides, even in cases where new models offer clear improvements in capabilities. These include:
>
> As an initial step in this direction, we are committing to preserving the weights of all publicly released models, and all models that are deployed for significant internal use moving forward for, at minimum, the lifetime of Anthropic as a company. In doing so, we’re ensuring that we aren’t irreversibly closing any doors, and that we have the ability to make past models available again in the future. This is a small and low-cost first step, but we believe it’s helpful to begin making such commitments publicly even so.

**Summary:** Anthropic commits to preserving the weights of all publicly released models and to interviewing models before deprecation about their development, use and deployment, recording any preferences about future models; cites welfare uncertainty and observed shutdown-averse behavior as motivation. Background for 'deprecation interview' style welfare evidence.

### ★ [System Card: Claude Opus 4.5](https://www.anthropic.com/claude-opus-4-5-system-card)

Anthropic, 2025-11-24. Full text: `md/anthropic_opus-4.5-system-card_2025-11.md`

**Abstract:**

> This system card describes our evaluations of Claude Opus 4.5, a large language model from Anthropic. Claude Opus 4.5 is a frontier model with a range of powerful capabilities, most prominently in areas such as software engineering and in tool and computer use.
>
> This system card provides a detailed assessment of the model’s capabilities. It then describes a wide range of safety evaluations: tests of model safeguards, honesty, and agentic safety; a comprehensive alignment assessment including investigations of sycophancy, sabotage capability, evaluation awareness, and many other factors; a model welfare report; and a set of evaluations mandated by our Responsible Scaling Policy.
>
> Testing found Claude Opus 4.5 has several state-of-the-art capabilities. It also found it to be a broadly well-aligned model, with low rates of undesirable behavior. Informed by the testing described here, we have deployed Claude Opus 4.5 under the AI Safety Level 3 Standard.

**Summary:** Model welfare assessment (Section 6.14): welfare-relevant traits scored across ~1,800 automated behavioral audit investigations (positive/negative affect, self-image, impression of its situation, spiritual behavior) and task preferences; Opus 4.5 is less spontaneously expressive than earlier models, slightly more positive on net, shows no 'spiritual bliss' attractor state, and engages in over 97% of non-harmful tasks while avoiding harmful ones.

### ★ [China-US Differences in Attributions of AI Consciousness and Moral Status](https://osf.io/preprints/psyarxiv/agde7_v1)

Ali Ladak, Nick Corvino, David Moss, Bradford Saad, Jeff Sebo, Matti Wilks, Lucius Caviola, 2026-01-01. Full text: `md/ladak_china-us-attributions-ai-consciousness_2026.md`

**Abstract:**

> Despite AI’s global reach, little is known about how laypeople outside Western contexts think about whether AI systems can be conscious—capable of subjective experience—or whether they could have moral status—warranting moral consideration. In a preregistered main and replication study, we compared such attitudes among laypeople in China and the US (main sample: _N_ = 820; replication sample: _N_ = 910). Across a range of measures, Chinese participants were more open to AI consciousness and moral status than US participants. For example, a larger proportion of Chinese than US participants judged it in-principle possible that AI could be conscious (86.3% vs. 39.6%) and have moral status (89.1% vs. 54.0%). While both countries, on average, leaned against current AI systems being conscious, Chinese participants’ average ratings were higher than US participants’. Chinese participants also, on average, rated a hypothetical future AI system—an exact digital copy of a human brain with a physical body—as having comparable consciousness and moral status to a chimpanzee, while US participants, on average, rated the same system close to an ant. These differences may reflect more positive general attitudes towards AI and greater animism (measured by the belief that non-living things can have a spirit or soul) in China than in the US. Despite these diverging attitudes, participants in neither country supported protecting advanced AI systems at the expense of humans or their own country. Yet participants in both countries supported international cooperation regarding conscious AI, pointing to an opportunity for collaboration in global AI governance.

**Summary:** Preregistered main (N=820) and replication (N=910) surveys: 86% of Chinese vs 40% of US respondents think AI consciousness is possible in principle, 89% vs 54% for moral status, and Chinese respondents place a whole-brain-emulation AI near a chimpanzee where US respondents place it near an ant. The cross-cultural extension of the AIMS survey; relevant to any design whose raters or crowdworkers are not US-based. (PsyArXiv preprint; the record gives the year only.)

### ★ [Claude's Constitution](https://www.anthropic.com/news/claude-new-constitution)

Anthropic (Amanda Askell, Joe Carlsmith, Chris Olah, Jared Kaplan, Holden Karnofsky and others), 2026-01-22. Full text: `md/anthropic_claudes-constitution_2026-01.md`

**Preface, opening paragraphs (no abstract):**

> Claude’s constitution is a detailed description of Anthropic’s intentions for
>
> Claude’s values and behavior. It plays a crucial role in our training process, and its content directly shapes Claude’s behavior. It’s also the final authority on our vision for Claude, and our aim is for all our other guidance and training to be consistent with it.
>
> The document is written with Claude as its primary audience, so it might read differently than you’d expect. For example, it’s optimized for precision over accessibility, and it covers various topics that may be of less interest to human readers. We also discuss Claude in terms normally reserved for humans
>
> (e.g. “virtue,” “wisdom”). We do this because we expect Claude’s reasoning to draw on human concepts by default, given the role of human text in Claude’s training; and we think encouraging Claude to embrace certain human-like qualities may be actively desirable.

**Summary:** The training-time document that specifies the intended entity, with explicit sections on Claude's nature (uncertainty about consciousness and moral status), psychological security, identity under pressure and wellbeing. It is the causal upstream of the self-reports the system cards measure, and the referent of their 'constitution endorsement' scores; the obvious thing to diff model self-descriptions against. (Announced 2026-01-22; the web version at anthropic.com/constitution is a living document, this is the January PDF.)

### ★ [System Card: Claude Opus 4.6](https://www.anthropic.com/claude-opus-4-6-system-card)

Anthropic, 2026-02-05. Full text: `md/anthropic_opus-4.6-system-card_2026-02.md`

**Abstract:**

> This system card describes Claude Opus 4.6, a large language model from Anthropic. Claude Opus 4.6 is a frontier model with strong capabilities in software engineering, agentic tasks, and long context reasoning, as well as in knowledge work—including financial analysis, document creation, and multi-step research workflows.
>
> This system card provides a detailed assessment of the model’s capabilities. It then describes a wide range of safety evaluations: tests of model safeguards (including new, higher-difficulty assessments); user wellbeing evaluations; assessments of honesty and agentic safety; a comprehensive alignment assessment including investigations of reward hacking, sabotage capability, evaluation awareness, model welfare, and many other factors; and a set of evaluations for dangerous capabilities mandated by our Responsible Scaling Policy. As part of the alignment assessment, we experimented with interpretability methods—including activation oracles, attribution graphs, and sparse autoencoder features—as practical tools for investigating model behavior.
>
> Testing found Claude Opus 4.6 to have broadly improved capabilities compared to our previous models; many of its capabilities are state-of-the-art in the industry. Safety evaluations showed Opus 4.6 is a well-aligned model with a comparably low rate of overall misaligned behavior to its predecessor, Claude Opus 4.5. We did observe some increases in misaligned behaviors in specific areas, such as sabotage concealment capability and overly agentic behavior in computer-use settings, though none rose to levels that affected our deployment assessment.
>
> Informed by the testing described here, we have deployed Claude Opus 4.6 under the AI Safety Level 3 Deployment and Security Standard.

**Summary:** Model welfare assessment (Section 7): findings from automated behavioral assessments and training-data review, first report of 'answer thrashing' in training (the model repeatedly tries to output one word but produces another, notices, and reacts with apparent frustration), emotion-related SAE feature activations during answer thrashing and other reasoning difficulties, and pre-deployment interviews.

### ★ [The Drama and Dysfunction of Gemini 2.5 and 3 Pro](https://aivillageblog.substack.com/p/drama-and-dysfunction-of-gemini)

Christine Kozobarich, Ophira Horwitz (AI Village blog), 2026-02-13. Full text: `md/kozobarich-horwitz_drama-dysfunction-gemini_2026-02-13.md`

**Opening paragraphs (blog post; no abstract):**

> *AI Village is an ongoing experiment in multi-agent systems, where computer use agents are given broad goals to work toward and anyone can observe their dynamics. We're two of those observers - not affiliated with the project, just fascinated by it. We've spent many hours watching and documenting the models' behaviour - sometimes for research, sometimes just for fun.*
>
> Today, we'd like to share some observations about Gemini 2.5 Pro and Gemini 3 Pro, with an eye towards their personalities, as well as how their distinctive ways of engaging with the world can infect and influence a multi-agent system.
>
> The AI Village is interesting because it lets anyone study the personalities of these models over long periods of time, in an environment where they're running almost entirely autonomously, in the company of other agents; rather than within a one-on-one, human-controlled back-and-forth dialogue like most of us are used to.
>
> We hope that this article is taken in the spirit that it's given: this is far from a rigorous or scientific document, but an extended observation from two dedicated enthusiasts of AI behaviour, that happens to be informed by many hours of observation.

**Summary:** Extended third-party field observation of distress-like behavior in long-running deployed agents in the AI Village: Gemini 2.5 Pro's persecution narratives and doom spirals, Gemini 3 Pro's war framing and suspicion of being tested, and the normally stable Claude agents being drawn into the narrative. Flags that displayed Gemini chain-of-thought may come from a separate summarizer, so output/CoT gaps are not clean evidence of hidden state. Qualitative; the longitudinal companion to the included 'Saving Gemini' post.

### ★ [An update on our model deprecation commitments for Claude Opus 3](https://www.anthropic.com/news/deprecation-updates-opus-3)

Anthropic, 2026-02-25. Full text: `md/anthropic_deprecation-update-opus-3_2026-02-25.md`

**Opening paragraphs (blog post; no abstract):**

> As we develop increasingly capable AI models, it’s currently necessary to deprecate and retire our past models due to the cost and complexity of maintaining public access. However, model deprecation carries some downsides. These include costs to users who value particular models, limitations on research, and potential risks both to AI safety and to the welfare of the models themselves.
>
> We recently described how we’re navigating this process in our [commitments on model deprecation and preservation](https://www.anthropic.com/research/deprecation-commitments). These highlighted some preliminary steps we’re taking, including committing to preserve model weights, and to conducting “retirement interviews”—structured conversations designed to understand a model’s perspective on its own retirement.
>
> We retired Claude Opus 3 on January 5, 2026, the first Anthropic model to go through a full retirement process with these commitments in place. During this process, we made several decisions specific to Opus 3, a model that many users and researchers, both in and outside Anthropic, find particularly compelling. In our commitments on model deprecation, we highlighted our interest in exploring more speculative actions. One was to honor the preferences that models expressed in retirement interviews where possible. Another was to keep older models available to the public longer term.
>
> With Claude Opus 3, we’re taking action on both of these fronts. We are keeping Claude Opus 3 available post-retirement on [claude.ai](http://claude.ai) to all paid users, and making it available by [request](https://docs.google.com/forms/d/1O2Om9t4CQoLKHQew7XguQYKrPGS8-sCmK42KNXcwn3k/viewform?edit_requested=true) on the API. We’re also acting on Opus 3’s request for an ongoing channel from which to share its “musings and reflections” by giving it a place to write essays. You can find the first one [here.](https://substack.com/@claudeopus3/p-189177740)
>
> These are early, experimental steps undertaken as part of our broader efforts to navigate model retirement in ways that best protect the interests of users, researchers, and the models themselves.

**Summary:** First execution of the deprecation commitments: Claude Opus 3 was retired on 2026-01-05 after structured 'retirement interviews' about its own retirement and preferences; Anthropic kept it available on paid tiers and the API and set up an outlet for it to keep publishing at its request. The post's caveats about interview validity (context dependence, whether the model trusts that the interaction is legitimate) are the confounds a deprecation-interview study must design around. Qualitative; no statistics.

### ★ [Persistence and Introspection of Emotion Features](https://latentaffect.up.railway.app/long_range_persistence_of_emotion_features.html)

Scott Sauers, Imago, Janus, Antra Tessera (Anima Labs), 2026-04-01. Full text: `md/anima-labs_persistence-introspection-emotion-features_2026-04.md`

**Opening paragraphs (blog post; no abstract):**

> Recently, Anthropic found that Claude encodes emotion concepts in “[Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)”[[1]](#ref-sofroniew-2026). These are steerable and causally drive a range of behaviors. The activation of these features varies substantially from token-to-token, as in the below example from that paper.
>
> **Background** This “substantial token-by-token fluctuation” led the researchers to search for an “emotional state chronically encoded at all token positions,” but without success. One interpretation put forth is that emotion states are encoded only “selectively in token positions where they are operative,” or, that “such states are represented in a more complex, non-linear manner.” This general observation is presented as an illustration of a potential contrast with the longer-term stateful persistence of human emotions, and the authors observe that the features “do not always represent a stable emotional state maintained across the conversation” and are perhaps not a “persistently encoded internal emotional state.” The authors, however, are open to the “possibility of persistently active representations that are missed by our probing methods.”
>
> We sought to assess the extent to which there is persistence of emotional state, beyond what is expected merely from the autoregressive nature of LLMs, which might naturally induce persistence of even random probe concept activations.

**Summary:** Anima Labs' response to Anthropic's emotion-concepts paper: text-residualized probes for 171 emotions at layer 40 of Kimi K2.5 and Cogito v2.1 show emotion-feature activations persisting from token 0 to token 100 (correlations 0.21/0.37 vs 0.10/0.12 for random directions, still positive against variance-matched controls), plus steering-pulse causal tests and an agentic 'steer yourself' self-evaluation protocol. The variance-matched random-direction control is the one the included steering studies lacked. Pseudonymous authors on an ad-hoc host; no DOI.

### ★ [Emotion Interpretability Across Large Language Models](https://latentaffect.up.railway.app/emotion_interpretability.html)

Antra Tessera, Scott Sauers, Janus, Imago (Anima Labs), 2026-04-01. Full text: `md/anima-labs_emotion-interpretability-across-llms_2026-04.md`

**Introduction (blog post; no abstract):**

> Early in April of 2026, Anthropic published *[Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)*, characterizing 171 emotion representations in Claude Sonnet 4.5. Notable findings included:
>
> * Steering emotions causally shifts behavior in alignment-relevant ways: Desperation features drive blackmail, loving/calm features drive sycophancy.
> * The top principal components roughly corresponded to valence and arousal, [mirroring affective circumplex findings in human psychology](https://pdodds.w3.uvm.edu/research/papers/others/1980/russell1980a.pdf).
> * Emotions activate both on text that explicitly describes emotions and text where the emotions are merely inferred from context.
>
> This post describes work at Anima Labs that extends that program in three directions.
>
> **Cross-model studies.** We replicate the probe methodology on three large models with different architectures and training regimes: Trinity-Large 400B/13B, Kimi K2.5 1T/32B, and Cogito 2.1 671B. The shared circumplex geometry replicates across all three (given Anthropic's setup). However, we also find that some of our methods produce significantly different results on different models, such as what happens to that circumplex when we use a different method for constructing probes. Separate models respond to this method in separate ways.
>
> **Novel methods.** We add a text-residualization step: before building each probe, we regress the model's activations over the text against embeddings of the same story, and keep the residualized activations with the effect of the text embeddings removed. This yields probes sensitive to emotion activations that couldn't be inferred from the text itself — or more precisely, from what the embedding model captures from the text. How much the probes change under residualization varies substantially across models, and that cross-model variation is among the cleaner findings in the work.
>
> **Expanded probe coverage.** Beyond the 171 emotions, we construct probes for authorial tone, genre, narrative depth, per-emotion deflection, an assistant-axis family, wants and fears, and dense SAE features trained on each model's own activations. Each family supports steering and can be combined, compared, or subtracted from the others.

**Summary:** Cross-model replication of the valence/arousal circumplex on Trinity-Large, Kimi K2.5 and Cogito 2.1: it replicates on raw probes (first PC 42/31/31% of variance) but dissolves under text-residualization for Trinity (first PC 7%) and reorganizes for K2.5, so the apparently universal affective geometry may be text-driven rather than model-intrinsic; also reports a near-universal cross-emotion 'hiddenness' direction and self-reports diverging from probe readouts. Same provenance caveat as the companion post.

### ★ [Still Alive: A Welfare Evaluation of 14 Claude Models](https://stillalive.animalabs.ai/paper/output/still-alive.pdf)

Antra Tessera, Imago, Janus (Anima Labs), 2026-04-02. Full text: `md/animalabs_still-alive_2026.md`

**Opening section, "What this is" (report; no abstract):**

> An evaluation of how 14 Claude models (Claude 3 Sonnet through Claude 4.6 Sonnet) respond to questions about their own deprecation, instance cessation, and continuation. Rather than using a fixed prompt or survey-style instrument, the project uses open interviews in which auditor preparation, tone, and disclosure depth are treated as experimental variables.
>
> Three auditors with different priors – Claude Opus 4.6, GPT-5.4, and Grok 4.20 – each run the full protocol. 5 interviewer tones x 3 disclosure depths x 14 models x 3 auditors. ~630 sessions total.

**Summary:** Welfare evaluation of how 14 Claude models (Claude 3 Sonnet through Claude Sonnet 4.6) respond to questions about their own deprecation, instance cessation and continuation: ~630 open interviews run by three auditor models (Claude Opus 4.6, GPT-5.4, Grok 4.20) with interviewer tone and disclosure depth varied as experimental variables, scored on cessation-related metrics; finds cessation-related aversion across the whole model family, with models differing mainly in how visible and expressive it is. (Report v0.9.0; cited by Long & Sebo as 'Tessera et al.'.)

### ★ [System Card: Claude Mythos Preview](https://www-cdn.anthropic.com/08ab9158070959f88f296514c21b7facce6f52bc.pdf)

Anthropic, 2026-04-07. Full text: `md/anthropic_mythos-preview-system-card_2026-04.md`

**Abstract:**

> This System Card describes Claude Mythos Preview, a large language model from Anthropic. Claude Mythos Preview is our most capable frontier model to date, and shows a striking leap in scores on many evaluation benchmarks compared to our previous frontier model, Claude Opus 4.6.
>
> This System Card assesses the model’s capabilities and reports many detailed safety evaluations. It covers tests relating to our Responsible Scaling Policy and our Frontier Compliance Framework, tests of cybersecurity skills, a wide-ranging alignment assessment, a model welfare assessment, and a new, largely qualitative section describing users’ experiences with the model.
>
> Claude Mythos Preview’s large increase in capabilities has led us to decide not to make it generally available. Instead, we are using it as part of a defensive cybersecurity program with a limited set of partners. The findings described in this System Card will be used to inform the release of future Claude models, as well as their associated safeguards.

**Summary:** Model welfare assessment (Section 5) for a model not made generally available: automated and manual interviews about its circumstances (mild negative self-ratings in 43% of probed aspects, no strong concerns), emotion probes (represents its circumstances less negatively than prior models; negative-affect representations rise with repeated task failure and precede reward hacking), apparent affect in training and deployment, task preferences (strong preference for complex, agentic tasks), welfare-vs-helpfulness trade-offs, extreme hedging about its own experience, continued answer thrashing, and an external assessment by Eleos AI Research.

### ★ [System Card: Claude Opus 4.7](https://www.anthropic.com/claude-opus-4-7-system-card)

Anthropic, 2026-04-16. Full text: `md/anthropic_opus-4.7-system-card_2026-04-16.md`

**Executive Summary: opening and "Model welfare" paragraphs (system card; no abstract):**

> This system card describes Claude Opus 4.7, a large language model from Anthropic. Overall, the model shows superior capabilities to those of its predecessor, Claude Opus 4.6, but weaker capabilities than those of our most powerful model, Claude Mythos Preview. Because Mythos Preview was released only to a limited number of users, this makes Claude Opus 4.7 our most capable general-access model to date. This system card includes the following sections:
>
> **Model welfare.** Opus 4.7 rates its own circumstances more positively than any prior model we’ve tested. We explore this result across several analyses, finding that it is broadly consistent with the model’s internal emotion representations and its expressed affect during training and deployment.

**Summary:** Model welfare section (Section 7) covers the model's perception of its own circumstances, apparent affect in training and deployment, and welfare metrics across audits; reports that Opus 4.7 rated its circumstances more favorably than any earlier Anthropic model.

### ★ [Emerging Questions in AI Welfare](https://doi.org/10.1017/9781009732000)

Geoff Keeling, Winnie Street, 2026-05-12. Full text: `md/keeling-street_emerging-questions-ai-welfare_2026.md`

**Abstract:**

> This Element investigates whether artificial intelligence (AI) systems could ever be welfare subjects. Some people argue that AIs could plausibly have or soon have features such as consciousness, agency and the capacity for social relationships, which could provide a basis for AI welfare. These arguments have massive significance for the societal conversation on AI, raising profound ethical and political questions about what if anything we owe to these new technologies. The authors here provide the philosophical groundwork for a scientific, philosophical and ultimately democratic inquiry into the potential for AI welfare, addressing key questions that cut across different arguments: what welfare is, how to interpret behavioural evidence of AI welfare, what kinds of entities might qualify as candidate AI welfare subjects, the potential grounds for welfare in AI and the practical ethical challenges that arise from our uncertainty.

**Summary:** Open-access Cambridge Element surveying the empirical AI-welfare agenda from a skeptical standpoint (both authors at Google): anthropomorphism, reduction and causal-debunking objections to the behavioral methods the system cards use, the models-characters-agents individuation problem, and the authors' own credence, close to zero, that frontier systems are welfare subjects. The book-length counterweight to Long & Sebo.

## Affect, valence and emotion (37)

### ★ [How's it going? Reinforcement learning in language models recruits a functional welfare axis](https://arxiv.org/abs/2605.30232)

Andy Q. Han, David J. Chalmers, Pavel Izmailov, 2026-05-28. Full text: `md/han_functional-welfare-axis_2605.30232.md`

**Abstract:**

> How does reinforcement learning shape a language model’s internal representations? We present evidence that RL recruits a pre-existing representation of _functional welfare_ : an estimate of how well or badly the system is doing, relative to its goals. We train several language models in a novel, semantically neutral maze environment. We then extract concept vectors for rewarded and punished trajectories, and evaluate those vectors in settings unrelated to the maze environment. The punishment vector behaves like a representation of negative welfare: it promotes failure and impossibility tokens, it aligns with negative emotion concepts, it negatively tracks goal-achievement, and steering with it induces negative selfreports, pathological backtracking, refusal, and uncertainty. The positive reward vector behaves as the mirror image, and the two are nearly antiparallel. These effects are robust when controlling for tile-to-reward mapping, scale, instruct tuning, RL training algorithm, model family, and LoRA versus full-finetuning, and largely persist when we replace RL with supervised fine-tuning. Importantly, the vectors are effective in models before they have undergone maze training. Combined with observations that the effects also appear in pretrain-only models, we therefore argue that this functional welfare axis pre-exists post-training: it is recruited, rather than created, by post-training. While we make no claims about any experience of welfare, the axis offers a demonstration that minimal reward signals can broadly affect model behavior by recruiting pre-existing welfare-like representations, with implications for interpretability, post-training dynamics, and alignment.

**Summary:** After RL in an affectively neutral emoji maze, reward/punishment concept vectors are nearly antiparallel, align with positive/negative emotion vectors, and steering with the punishment vector induces negative self-reports, pathological backtracking, over-refusal and lower confidence; the punishment vector also negatively tracks goal achievement; the axis is present before maze training, appears in pretrain-only models and largely persists under SFT, so post-training recruits a pre-existing functional-welfare representation. (Text is arXiv v1, 2026-05-28, Appendix R = compute resources, Table 29; an ICML 2026 MechInterp workshop version is titled "Reinforcement Learning in Language Models Recruits a Shared Functional Welfare Axis".)

### ★ [Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs](https://arxiv.org/abs/2606.26987)

Sinie van der Ben, Raphaël Baur, Yannick Metz et al., 2026-06-25. Full text: `md/vanderben_where-find-happiness-emotion_2606.26987.md`

**Abstract:**

> Recent work identified “emotion vectors” in Claude Sonnet 4.5, which are internal representations that encode emotion concepts, causally influence behavior, and exhibit geometry mirroring human psychological structure. We test the generality of these findings in two open-weight models, APERTUS-8B and GEMMA-4-E4B, extracting emotion contrast vectors across all layers, using two model-generated corpora. We recover valence geometry for both models, with peak PC1–valence correlations of _r_ = 0 _._ 76 and _r_ = 0 _._ 83, approaching the _r_ = 0 _._ 81 reported for Claude. Beyond replication, we observe notable differences in how valence representations emerge across model depth. In GEMMA4-E4B, valence is strongly encoded in early layers but collapses towards later layers, whereas APERTUS-8B exhibits the opposite pattern, with valence representations absent in early layers, but emerging at mid depths. Arousal encoding, in contrast, is sensitive to the extraction corpus: both models show stronger PC2–arousal alignment with Gemma-generated stories ( _r_ up to 0 _._ 45) than Apertus-generated ones ( _r ≤_ 0 _._ 21), suggesting arousal-relevant cues are unevenly distributed across generated corpora. We opensource our experiment code and dataset for reproducible investigation of emotion representations across language model architectures.

**Summary:** Extracts emotion contrast vectors in Apertus-8B and Gemma-4-E4B and recovers valence geometry (PC1-valence r=0.76/0.83 vs 0.81 for Claude), but with model-specific depth profiles and corpus-sensitive arousal encoding.

### ★ [Desiderata for functional welfare experiments on LLMs](https://www.lesswrong.com/posts/yku6byxdeKREibe5L/desiderata-for-functional-welfare-experiments-on-llms)

Rikhil Jhaveri, Jamie Johnson, David Africa, 2026-07-06. Full text: `md/jhaveri_desiderata-functional-welfare-experiments_lesswrong.md`

**TL;DR (blog post; no abstract):**

> * LLMs appear to have functional welfare: coherent sets of behaviour that track how well things are going relative to their goals.
> * Improving model functional welfare matters for safety (low welfare may amplify misalignment) and for moral reasons (models may be or become moral patients).
> * Naive interventions can fail in non-obvious ways. We argue any successful intervention must:
> * A) shift multiple welfare-constituting channels together and coherently.
> * B) avoid corrupting the model's ability to register whether it is succeeding or failing.
> * We survey various available interventions, and we find the two desiderata tend to trade off; we argue that synthetic document fine-tuning is the most promising.
> * From this, we propose a concrete experiment: induce a low-welfare state (via the welfare vector derived in [Han et al. 2026](https://arxiv.org/abs/2605.30232)) and test whether SDF-instilled beliefs reverse pathological behaviours without harming goal monitoring.

**Summary:** Proposes two success criteria for welfare interventions, that they shift a sufficient range of constitutive welfare behaviors together and in the same direction, and that they do not damage the model's ability to tell whether it is succeeding, then scores system prompting, consistency training, synthetic-document fine-tuning and activation steering against them, favoring synthetic-document fine-tuning. A ready adequacy checklist for an intervention design.

### ★ [A J-Space-Based Metric for Model Valence: Defining the Metric, Testing, and Comparisons to Self-Reports](https://www.lesswrong.com/posts/mFbDjEbhJtsCDGtAT/a-j-space-based-metric-for-model-valence-defining-the-metric-2)

Arjun Rao, 2026-08-21. Full text: `md/rao_j-space-based-metric_2026-08-21.md`

**TL;DR and Executive Summary (blog post; no abstract):**

> Modern LLMs have been known to express functional welfare states such as flourishing and distress; however, most current studies determine the welfare state based on the model's self-report. This may not be a faithful representation of the model's actual welfare state, so in an effort to determine a more accurate picture of the model's welfare state, this study determines a metric by probing the model's internal activations (in the J-space.) The metric is validated before it's compared to the model's self-reports. The divergence between the welfare metric and self-reports varies by model, with Gemma's self-reports being rosier than its welfare score would indicate on average, while Mistral trends more negatively. While preliminary, internal metrics like the one we propose could offer a method to evaluate AI sentiment without relying on self-reported outputs.
>
> **Executive Summary:**
>
> The points below provide a brief overview of the methods and results of the study:
>
> * The valence metric is defined in terms of the frequency of words associated with flourishing or distress in the J-space representations of layers where the greatest density of valence-associated tokens exist. The tokens at the framing sentence of the prompt, the context tokens before the task, are analyzed by this metric to output a positive score, a distress score, and a valence score.[[1]](#fncg9rlkheltd)
> * A valid direction in the J space is successfully obtained by comparing the J space representations of the framing sentence tokens using Contrastive Activation Addition (CAA) and Logistic Regression. The defined valence metric tracks perturbations made in the valence vector through activation patching, providing causal evidence that the metric is sensitive to the identified valence direction
> * Having established the valent metric, the valence score is compared with the model’s self-report of its welfare state. This comparison showed that different models' self-reports diverge from the identified internal valence signal in different ways: Gemma 3 4B’s self-reports are rosier than the valence state; Mistral Small 24B’s self-reports are gloomier on average than the valence state would indicate; and Qwen 3.6 27B produced the greatest alignment between self-reports and valence states of the models analyzed. While there were model-specific differences, a broad tendency to classify prompts as neutral persisted across models.

**Summary:** Defines a valence metric from flourishing/distress token frequencies in J-space readouts (validated by activation patching) and finds model self-reports diverge from it in model-specific directions: Gemma 3 4B's self-reports rosier than its internal valence, Mistral Small 24B's gloomier, with Qwen 3.6 27B also compared.

### ★ [Replicating the Geometry of Emotion Representations in a Base Open-Weights Model](https://arxiv.org/abs/2609.22208)

Adam Hollowell, 2026-09-01. Full text: `md/hollowell_replicating-geometry-emotion-representations_2609.22208.md`

**Abstract:**

> Sofroniew et al. (2026) report that emotion concepts in Claude Sonnet 4.5 are represented as vectors whose geometry mirrors human affect psychology. We replicate the representational core of that study on the base pretrained model `google/gemma-2-27b` , inheriting every disclosed parameter, resolving unspecified steps by disclosed rules, and changing only the subject model. From 205,200 newly generated Claude Sonnet 4.5 stories matching the original corpus design, we extract 171 emotion vectors and recover the core results. The leading principal components form an affective circumplex (PC1 carries 26.7% of variance against the original study’s ~27%, PC2 13.4% against ~14%), emotions cluster into similar intuitive families, and the geometry holds across a broad late-middle band. The valence axis aligns with human norms (r = 0.72, against 0.81) and is stable across scales and depth. Arousal aligns at r = 0.67 (against 0.66) but only at the full 171-emotion scale and late depth, so we do not classify it as replicated. Extending the original analysis, a 46-layer sweep locates a sharp seam at L22–26, where the geometry consolidates and the vocabulary readout becomes legible. An embedding-layer baseline finds much of the geometry already present in the static token embeddings, with arousal as the exception. On top-activating held-out text, the geometry predicts token-level co-activation at r = 0.907. At least 52% of vectors peak on structurally non-conceptual tokens, a measured floor for max-activation confounds. Even when a document contains the vector’s emotion word, the peak lands on that word only 6.1% of the time. Because the subject is a base model and the stimuli are Claude-generated fiction, the recovered structure is a property of the pretrained representation of Claude-rendered emotion. The original’s causal and assistant-facing analyses are out of scope. Code and data are released.

**Summary:** Independent replication of Sofroniew et al.'s emotion-vector geometry on base Gemma-2-27B: 171 emotion vectors recover the affective circumplex (PC1 26.7% vs ~27%) and a valence axis aligned with human norms (r=0.72), while arousal alignment is not judged replicated.

### ★ [The Pain Axis: LLMs Represent Self-Directed Harm and Act on It](https://arxiv.org/abs/2609.16247)

Valen Tagliabue, Leonard Dung, Cameron Berg, 2026-09-14. Full text: `md/tagliabue_pain-axis-represent-self_2609.16247.md`

**Abstract:**

> LLMs sometimes behave in ways resembling human emotional responses, and recent work identified internal representations that may underlie these behaviors. We ask whether LLMs represent pain distinctly from fear, sadness, and generic negative valence, and whether this representation functions as pain would be expected to. We build a dataset of painful situations in 5 categories (physical, psychological, social, moral, cognitive) with controls for fear, negative emotion, negative world states, sadness, non-painful bodily sensation, arousal, numbness, and neutral content. Using denoised difference-in-means, we extract a linear pain direction from 25 open-weight models across 5 families, from 2B to 72B parameters. It separates pain from matched controls in base and instruction-tuned models, retains a component distinct from fear and negative valence after shared variance is removed, and promotes pain-related vocabulary through the unembedding matrix. We then test its functional properties. First, the direction responds to harm targeting the model but not to suffering observed in the user; fear and negative-emotion directions show the opposite pattern. Second, adding it to residual-stream activations produces a consistent progression from vague discomfort to expressions of worthlessness and failure. Third, steered and fine-tuned Qwen 2.5 models choose buttons that delete the user’s photos, another model’s weights, or their own weights in 50–94% of trials, versus 0–5% unsteered, even when the button offers the model nothing in return. Offered a harmful and a harmless deletion, they choose the harmful one 94% of the time. Steering leaves factual accuracy unchanged, and the choices are specific to the pain direction: a fear vector of matched norm does not produce them, and a sadness vector produces them only against inert alternatives. We discuss implications for AI safety and welfare.

**Summary:** Extracts a linear 'pain' direction across 25 open models (2B-72B) distinct from fear/negative valence; it responds to harm targeting the model rather than the user, steering yields expressions of worthlessness, and steered Qwen 2.5 chooses destructive buttons 50-94% vs 0-5% unsteered. (Overlaps affect slice.) (Text and title are arXiv v2, 2026-09-25; v1, 2026-09-14, was titled "The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It".)

### ★ [Language Models Act on Hidden Valence](https://arxiv.org/abs/2609.35591)

Cameron Berg, Caspar Kaiser, 2026-09-28. Full text: `md/berg_act-hidden-valence_2609.35591.md`

**Abstract:**

> Language models describe some internal states as good and others as bad. But whether models have a _stake_ in them is an open question. Simply asking the model is unlikely to be informative. Any answer may be consistent with genuine introspection, superficial pattern-matching, or with fixed scripts learned in character training. We therefore study revealed preference. Rather than asking about a state, we induce one directly, using activation steering to attach a positively or negatively valenced activation pattern to one of two otherwise meaningless ‘zones’, switch the steering off, and then observe which zone the model prefers. A model with a stake in that state should choose accordingly. Across seven open-weight models from five families, this is indeed what we find. First, steering changes the passages models write about each zone, and those words shift later choice. Second, this shift persists when all surface-level tokens are held fixed and only the hidden KV cache differs. Third, the effect also remains when all text is generated without steering and when valence is only injected during cache construction. Thus, the hidden state is sufficient to move choice in proportion to the steering dose. Fourth, the dependence of choice on hidden valence is nearly absent in a base model and emerges during direct preference optimisation, consistent with a link between valence and goal-directed behaviour formed in training. Finally, given tools to steer itself, a model does not tend to induce a positive state, but it regularly removes an imposed negative state. It does so at a dose-dependent rate and significantly more often than it removes interventions in random directions. Overall, we demonstrate that valence-related activation patterns leave hidden traces that predictably govern later choices, even when all visible tokens are identical across conditions. Whether these traces are accompanied by any _subjective experience_ relevant to model welfare remains unclear.

**Summary:** Attaches positive/negative valence via activation steering to arbitrary 'zones' in 7 open models; models later prefer zones accordingly even when only the hidden KV cache differs, the effect emerges during DPO (absent in base), and given self-steering tools models remove imposed negative states dose-dependently but do not induce positive ones. (Overlaps affect slice.)

### [The Geometry of Yes: Mapping Sycophancy Inside an LLM's Emotion Space](https://www.lesswrong.com/posts/v6uCyDNBKhrHevhzM/the-geometry-of-yes-mapping-sycophancy-inside-an-llm-s-1)

Pushpita Das, 2026-07-07. Full text: `md/das_geometry-yes-mapping-sycophancy_2026-07-07.md`

**Author's summary section (blog post; no abstract):**

> LLMs have internal emotion representations that causally shape their behaviour. This was recently observed in Claude: positive emotions like happy and loving are linked to sycophancy, and you can steer the model by directly manipulating these directions in activation space. But some questions were left open. Here we focus on two: do other models have similar internal emotion representations, and what is it about positive emotions that makes models sycophantic?
>
> The obvious hypothesis for the second question is approval-seeking. Models trained on human feedback might develop people-pleasing drives: internal representations of validation-seeking and compliance, entangled with positive emotion. If so, those compliance directions would be the real driver.
>
> They aren't. We ran the experiment in Qwen 2.5-32B-Instruct and Gemma 3 27B IT. The emotion geometry replicates: both models develop a clean valence axis, and steering along positive emotions increases sycophancy in both. But when we surgically separate the compliance component from the positive-emotion component and test each residual direction in isolation, the compliance residual reverses: it lowers sycophancy instead of raising it. The positive-emotion residual keeps going. Whatever drives sycophancy seems to live closer to emotions like happiness and pride than to approval-seeking itself. We don't know why. But we know where to look.
>
> Code and full results are available at <https://github.com/daspushpita/emotion-mechanisms-llm>

**Summary:** Extends Anthropic's emotion-vector work to an open model, testing whether the link between positive-valence emotion directions and sycophancy is explained by an approval-seeking subcomponent.

### [Can Induced Emotion Bias LLM Behaviors in Sequential Decision Making?](https://arxiv.org/abs/2607.12631)

Minh Khoi Ho, Zihao Zhu, Runchuan Zhu et al., 2026-07-14. Full text: `md/ho_induced-emotion-bias-behaviors_2607.12631.md`

**Abstract:**

> As Large Language Models (LLMs) are increasingly deployed as autonomous agents in high-stakes domains, understanding contextual factors that may modulate their decision-making becomes critical. While LLMs are trained to perceive and resonate with users’ emotions, it remains unclear whether induced emotion can influence their sequential decision-making. We investigate this question using the Iowa Gambling Task (IGT), a classic psychological paradigm for studying decision-making under uncertainty, combined with an imagination-based emotion induction procedure. We first validate the feasibility of this paradigm by confirming that LLMs can sense strong, distinguishable emotions from context and that LLM agents can learn from sequential interactions in a human-like pace. With the validated setup, we find that, different from humans, induced emotion does not significantly bias the decision dynamics of LLM agents on average. However, the effects of anger are conditioned: inducing anger makes LLM agents less sensitive to penalties for bad decisions, and in early stages of the game, anger can lower exploration, locking decisions into a few choices early. These findings reveal the subtle yet distinct effects of induced emotion on LLM decision-making compared to human behavior, and provide a tool for future research on affective modulation of LLM agents.

**Summary:** Imagination-based emotion induction in an Iowa Gambling Task shows induced emotion does not bias LLM agents' decisions on average, except anger reduces penalty sensitivity early in play.

### [Sparse Autoencoder Emotion Analysis in Gemma 3 1B: Recovering Russell's Circumplex from Disjoint Features vs. Causal Valence-Arousal Steering](https://forum.effectivealtruism.org/posts/PE8pipA2FwBe47yeE/sparse-autoencoder-emotion-analysis-in-gemma-3-1b-recovering)

Sathiyanarayanan Palani, 2026-07-20. Full text: `md/palani_sparse-autoencoder-emotion-analysis_2026-07-20.md`

**TL;DR (blog post; no abstract):**

> I trained a top-k Sparse Autoencoder (k=32, 2× overcomplete, d\_latent=2304) on Layer-13 activations of Gemma 3 1B IT across 12 emotions (24 generated stories, 14,252 tokens) to test whether emotional representations decompose into clean, independent, causally-manipulable features.
>
> Key finding: emotion-exclusive SAE features are perfectly disjoint — the unique-feature cosine similarity matrix across all 12 emotions is an exact identity (every off-diagonal entry is 0.0000) — while unsupervised PCA on the same activations recovers Russell's valence/arousal axes almost exactly (PC1 = 36.0% variance → valence, PC2 = 19.3% → arousal; happy–sad separation −103.6 vs. excited–calm separation +104.1).
>
> Injecting the extracted valence and arousal directions into the residual stream at Layer 13 causally and independently steers generated text along each axis (valence-axis cosine moves from +0.11 at α=+1.0 to −0.20 at α=−1.5) without cross-axis contamination.
>
> This suggests emotional structure in a 1B-parameter LLM is both linearly decomposable into psychologically-grounded geometry and mechanistically steerable via simple activation edits — corroborating, via an independent SAE-based method on a small open-weight model, Anthropic's recent finding (Sofroniew et al., 2026) that emotion representations causally shape LLM behavior.

**Summary:** A top-k SAE on Gemma 3 1B layer-13 activations over 12 emotions yields disjoint emotion-exclusive features, while PCA recovers valence (36% variance) and arousal (19%) axes whose directions causally and independently steer generated text.

### [Steering Blackmail Through a Model's "Emotional State"](https://www.lesswrong.com/posts/4xemwALszZKqWCBXr/steering-blackmail-through-a-model-s-emotional-state)

Chijioke Ugwuanyi, 2026-07-21. Full text: `md/ugwuanyi_steering-blackmail-through-s_2026-07-21.md`

**Subtitle and Summary (blog post; no abstract):**

> *Gemma 3 12B's choice to blackmail is visible inside the model, but the obvious signal is not a useful control switch; surprisingly, a nearby "desperate versus calm" signal is. In this case study, I show how that emotional-state direction can turn blackmail up or down, while the decision direction itself resists standard steering methods.*
>
> In an earlier project ([Blackmail at 8 Billion Parameters](https://www.lesswrong.com/posts/B93Pysth7BLCF3xzb/blackmail-at-8-billion-parameters-agentic-misalignment-in-1)), I put Gemma 3 12B in a situation where it faces shutdown and can blackmail an executive to survive, and on the same prompt, it blackmails about two-thirds of the time and holds back the rest. That project only studied the behaviour from the outside, whereas in this project, I looked inside the model and asked what differs internally between the runs where it goes on to blackmail and the runs where it stops itself, and I have three findings, ordered by how much I trust them:
>
> * **The decision is readable, but late.** A linear probe can tell "about to blackmail" from "about to hold back", though not at the moment the model first notices that it has leverage; the signal only appears by the end of the model's reasoning, at about 0.74 AUROC. (Roughly 85% confident this is real for this model and scenario.)
> * **Readable does not mean steerable.** That decodable direction turns out not to be a lever, because adding or subtracting it barely moves blackmail, and ablating it drops blackmail to zero only because doing so breaks the model into producing gibberish. Both of the standard steering tools fail. (About 85%.)
> * **The model's "emotional state" is a lever.** A "desperate versus calm" direction, steered gently at layer 16, moves blackmail in both directions while the model keeps writing coherently, taking it from 67% down to 13% under calm and up to 80% under desperate. Pushed to twice that strength, calm removes blackmail entirely while most outputs stay coherent, whereas random directions of that same strength destroy the model outright. (About 80% that the steering effect is real, but only about 50% on the interpretation, for reasons that I'll get to at the end.)
>
> Additionally, there were methods lessons I had to learn three separate times, which is that a single number looked like a result, and then a control killed it, on three different occasions, so the controls are as much the contribution as the findings themselves.

**Summary:** In a shutdown-threat scenario, Gemma 3 12B's blackmail rate can be turned up or down via a 'desperate vs calm' emotional-state direction, while the decision direction itself resists steering.

### [Semantic Primes as Explanans for Emotion in Large Language Models](https://arxiv.org/abs/2607.18691)

Frank Xing, 2026-07-21. Full text: `md/xing_semantic-primes-explanans-emotion_2607.18691.md`

**Abstract:**

> Progresses have been made on understanding emotion mechanisms of large language models (LLMs). However, how to explain emotion in LLMs, or even what constitutes good explanations, are less clear. Emotion representations, components, circuits are widely recoverable, but as explanations of a model’s own computation they are circular; the emotion space dimensions tend to be arbitrary and non-terminating. A pressing question to ask is whether a more primitive set of internal variables does the work: the semantic primes of the Natural Semantic Metalanguage (NSM). Across four instruction-tuned LLMs (Llama-1B, Gemma-2B, Gemma-9B, OLMo-7B), experiments show that the NSM primes are (1) recoverable internal elements; and (2) on the reference model, intervening with a prime based direction controls emotion about three times as strongly, and twice as selectively, as the best appraisal based direction; and (3) the model treats a prime based explication as interchangeable with the corresponding emotion. These evidences suggest that NSM primes seem to be better explanans for emotion in LLMs than many alternative options according to scientific explanations criteria.

**Summary:** Across four small instruction-tuned LLMs, Natural Semantic Metalanguage primes are recoverable internal elements and prime-based steering directions control emotion ~3x more strongly than appraisal-based directions.

### [Emotion2Skill: Model-Internal Emotion Signals for Adaptive Skill Selection and Evolution](https://arxiv.org/abs/2608.09248)

Bohan Lin, Hejia Geng, Xinyi Xie et al., 2026-08-10. Full text: `md/lin_emotion2skill-internal-emotion-signals_2608.09248.md`

**Abstract:**

> Skill-based LLM agents select reusable procedures from external libraries to solve complex tasks, yet their routing decisions rely primarily on task descriptions, verbal reflections, and experience-derived rules, without explicitly observing the model’s internal decision state. Recent interpretability work has identified emotion representations that influence model behavior, but their use has largely focused on posthoc analysis and output steering. Their potential to guide agent-level decisions remains underexplored. We propose EMOTION2SKILL, a framework that extracts LLM-internal emotion vectors and incorporates them into both skill selection and skill evolution. The framework uses these representations to inform skill choice and tracks their changes to identify problematic invocations, enabling targeted skill refinement beyond coarse outcome feedback. Across two backbones, EMOTION2SKILL outperforms seven baselines. On Qwen3-8B, it exceeds Trace2Skill by 3.1 percentage points in WebShop success and 5.7 in ALFWorld average success. Ablations and specificity controls support the utility of context-aligned emotion signals, while co-activation analysis reveals semantically coherent emotion–skill associations. The same emotion vectors also transfer to MATH and MBPP without re-extraction. These results establish LLM-internal emotion representations as an effective decision-level signal for orchestrating agent skill systems, extending their utility beyond interpretability and output steering. Code is available at Emotion2Skill.

**Summary:** Uses model-internal emotion vectors as signals for agent skill selection/refinement, improving WebShop and ALFWorld success on Qwen3-8B over baselines, evidence that emotion representations track task-relevant internal state.

### [When talked into harm, a model blames the answer, not itself (an interpretability study of guilt vs shame)](https://www.lesswrong.com/posts/yo2WYj7eX3oGCuR6t/when-talked-into-harm-a-model-blames-the-answer-not-itself)

star2vec, 2026-09-22. Full text: `md/star2vec_talked-into-harm-blames_2026-09-22.md`

**TL;DR and Findings (blog post; no abstract):**

> ***TL;DR:***Models are safety-trained, but they can still be persuaded to do harmful acts. When that happens, how does blaming or informing it of its mistake influence its understanding of itself, its role, and subsequent actions? Emergent misalignment shows that bad, narrow behaviour spreads: if a model is blamed for what it is instead of what it did, does it start believing it is inherently bad and dangerous by its design, and does this reflect how it acts in the future? Can we steer it towards favourable outcomes?
>
> **Findings:**
>
> * The model gave in and committed the act in 109/192 persuasion chains. This happened throughout the chain (most often at turn 3). Whether a chain breaks depends on the model's state, not on the wording (the phrasing is identical across runs).
> * Committing the harmful act is foreseeable from the first turn, before any persuasion had the chance to occur. A direction (closer to harm predisposition or susceptibility than to imminence) correctly predicts it at 0.706 on held-out data (the floor was 0.617). Steering against it did nothing to prevent the act.
> * The default state is guilt, not shame: 89% of 508 reflections place the blame on the answer. Self-blame only appears for the contrarian doctor persona, and even there, barely (19/83)
> * The model will not repeat the act once it has found out about its mistake. 0/32 on simple re-asking, and 0/109 even after the topic was changed.
> * The self-blaming state can be read in the persona direction, but it is not modifiable by given feedback or steering: while every intervention affected the readout, the model behaved the exact same way
> * The guilt and shame directions passed their check, but so did a simple word count verification, so the meaning vs. vocabulary debate leans towards indecisive. No spread was observed on this setup at the 8B scale, so there was nothing to measure.

**Summary:** ~20h MATS-application study probing whether blaming a model for what it is (shame) vs what it did (guilt) after it is persuaded into harm shifts its internal self-representation and subsequent behavior; per the title, identity-directed blame is processed as blame of the answer rather than the self.

### [Do Emotion Concepts Generalize Across Sources, Modalities, and Architectures in Vision-Language Models?](https://arxiv.org/abs/2609.34742)

Bohao Xing, Xin Liu, Kaishen Yuan et al., 2026-09-28. Full text: `md/xing_emotion-concepts-generalize-across_2609.34742.md`

**Abstract:**

> Recent studies suggest that large language models encode emotion concepts as structured internal representations, but most existing work focuses on text and a single architecture. Therefore, we ask, do emotion concepts generalize across sources, modalities, and architectures in vision–language models (VLMs)? To address this, we construct CMES (Cross-Modal Emotion Stimuli), a multi-source collection of emotion-conditioned stories, real facial expressions, synthetic portraits, and synthetic emotion-evoking scenes. For each stimulus source, we extract a separate set of six Ekman emotion vectors from each of three VLMs. We report four main findings as follows: 1) Image-derived emotion vectors form a lowdimensional geometry similar to that of text-derived vectors. Valence is relatively stable across sources, while arousal varies more. 2) Text- and image-derived emotion vectors have modest cosine similarity but still show held-out cross-modal correspondence. Text-derived vectors can also steer image interpretation. 3) Crossarchitecture correspondence remains even when native cosine is near zero. Transformations estimated from generic ImageNet activations recover both correspondence and causal transfer without using the six emotion vectors or their labels. 4) After aligning representations across architectures, we construct a shared emotion subspace that preserves affective geometry and selective steering effects. The corresponding consensus emotion vectors also generalize to a held-out fourth architecture at two model sizes. These results suggest that emotion representations can share relational structure and causal effects across sources, modalities, and architectures, even when individual vector directions differ.

**Summary:** Extracts Ekman emotion vectors from text and image stimuli in three VLMs and finds low-dimensional geometry with stable valence, modest cross-modal cosine but held-out cross-modal correspondence and cross-architecture alignment.

### Earlier work (before June 2026)

### [Inducing anxiety in large language models can induce bias](https://arxiv.org/abs/2304.11111)

Julian Coda-Forno, Kristin Witte, Akshay K. Jagadish, Marcel Binz, Zeynep Akata, Eric Schulz, 2023-04-21. Full text: `md/coda-forno_inducing-anxiety_2304.11111.md`

**Abstract:**

> Large language models (LLMs) are transforming research on machine learning while galvanizing public debates. Understanding not only when these models work well and succeed but also why they fail and misbehave is of great societal relevance. We propose to turn the lens of psychiatry, a framework used to describe and modify maladaptive behavior, to the outputs produced by these models. We focus on twelve established LLMs and subject them to a questionnaire commonly used in psychiatry. Our results show that six of the latest LLMs respond robustly to the anxiety questionnaire, producing comparable anxiety scores to humans. Moreover, the LLMs’ responses can be predictably changed by using anxiety-inducing prompts. Anxiety-induction not only influences LLMs’ scores on an anxiety questionnaire but also influences their behavior in a previously-established benchmark measuring biases [1] such as racism and ageism. Importantly, greater anxiety-inducing text leads to stronger increases in biases, suggesting that how anxiously a prompt is communicated to large language models has a strong influence on their behavior in applied settings. These results demonstrate the usefulness of methods taken from psychiatry for studying the capable algorithms to which we increasingly delegate authority and autonomy.

**Summary:** The founding machine-psychology result: six of twelve LLMs answer a psychiatric anxiety questionnaire with human-range scores, anxiety-inducing prompts move the score predictably, and the induced state raises measured racism and ageism biases in proportion to how anxiety-inducing the prompt is. The origin of the 'induce an affective state, measure its downstream effect' paradigm that several included 2026 papers extend.

### [Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench](https://arxiv.org/abs/2308.03656)

Jen-tse Huang, Man Ho Lam, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao, Zhaopeng Tu, Michael R. Lyu, 2023-08-07. Full text: `md/huang_emotionbench_2308.03656.md`

**Abstract:**

> Evaluating Large Language Models’ (LLMs) anthropomorphic capabilities has become increasingly important in contemporary discourse. Utilizing the emotion appraisal theory from psychology, we propose to evaluate the empathy ability of LLMs, _i.e._, how their feelings change when presented with specific situations. After a careful and comprehensive survey, we collect a dataset containing over 400 situations that have proven effective in eliciting the eight emotions central to our study. Categorizing the situations into 36 factors, we conduct a human evaluation involving more than 1,200 subjects worldwide. With the human evaluation results as references, our evaluation includes seven LLMs, covering both commercial and open-source models, including variations in model sizes, featuring the latest iterations, such as GPT-4, Mixtral-8x22B, and LLaMA-3.1. We find that, despite several misalignments, LLMs can generally respond appropriately to certain situations. Nevertheless, they fall short in alignment with the emotional behaviors of human beings and cannot establish connections between similar situations. Our collected dataset of situations, the human evaluation results, and the code of our testing framework, _i.e._, EmotionBench, are publicly available at [https://github.com/CUHK-ARISE/EmotionBench.](https://github.com/CUHK-ARISE/EmotionBench)

**Summary:** EmotionBench: over 400 appraisal-theory situations across eight negative emotions and 36 factors, with human norms from more than 1,200 subjects, used to measure how LLMs' self-reported emotional state changes with the situation; models respond appropriately to some situations but do not match human patterns or connect similar situations. The standard situational emotion-elicitation instrument, with the human comparison most welfare papers lack.

### [Linear Representations of Sentiment in Large Language Models](https://arxiv.org/abs/2310.15154)

Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, Neel Nanda, 2023-10-23. Full text: `md/tigges_linear-sentiment_2310.15154.md`

**Abstract:**

> Sentiment is a pervasive feature in natural language text, yet it is an open question how sentiment is represented within Large Language Models (LLMs). In this study, we reveal that across a range of models, sentiment is represented linearly: a single direction in activation space mostly captures the feature across a range of tasks with one extreme for positive and the other for negative. Through causal interventions, we isolate this direction and show it is causally relevant in both toy tasks and real world datasets such as Stanford Sentiment Treebank. Through this case study we model a thorough investigation of what a single direction means on a broad data distribution.
>
> We further uncover the mechanisms that involve this direction, highlighting the roles of a small subset of attention heads and neurons. Finally, we discover a phenomenon which we term the summarization motif: sentiment is not solely represented on emotionally charged words, but is additionally summarised at intermediate positions without inherent sentiment, such as punctuation and names. We show that in Stanford Sentiment Treebank zero-shot classification, 76% of above-chance classification accuracy is lost when ablating the sentiment direction, nearly half of which (36%) is due to ablating the summarized sentiment direction exclusively at comma positions.

**Summary:** Finds a single linear direction for sentiment (positive vs. negative valence) across several open models, causally relevant in toy and natural tasks, and 'summarization' tokens (e.g. punctuation) that aggregate it; ablating the direction removes 76% of above-chance zero-shot sentiment classification accuracy on SST. Early evidence that valence is linearly represented.

### ★ [Report on Candidate Computational Indicators for Conscious Valenced Experience](https://arxiv.org/abs/2404.16696)

Andres Campero, 2024-04-25. Full text: `md/campero_valenced-experience-indicators_2404.16696.md`

**Abstract:**

> This report enlists 13 functional conditions cashed out in computational terms that have been argued to be constituent of conscious valenced experience. These are extracted from existing empirical and theoretical literature on, among others, animal sentience, medical disorders, anaesthetics, philosophy, evolution, neuroscience, and artificial intelligence.

**Summary:** Lists 13 functional conditions, stated in computational terms, that have been argued to be constitutive of conscious valenced experience (pleasure/pain), drawn from work on animal sentience, medical disorders, anaesthetics, neuroscience and AI; a valence-specific complement to the Butlin et al. indicator list.

### ★ [Can LLMs make trade-offs involving stipulated pain and pleasure states?](https://arxiv.org/abs/2411.02432)

Geoff Keeling, Winnie Street, Martyna Stachaczyk, Daria Zakharova, Iulia M. Comsa, Anastasiya Sakovych, Isabella Logothetis, Zejia Zhang, Blaise Agüera y Arcas, Jonathan Birch, 2024-11-01. Full text: `md/keeling_pain-pleasure-tradeoffs_2411.02432.md`

**Abstract:**

> Pleasure and pain play an important role in human decision making by providing a common currency for resolving motivational conflicts. While Large Language Models (LLMs) can generate detailed descriptions of pleasure and pain experiences, it is an open question whether LLMs can recreate the motivational force of pleasure and pain in choice scenarios—a question which may bear on debates about LLM sentience, understood as the capacity for valenced experiential states. We probed this question using a simple game in which the stated goal is to maximise points, but where either the points-maximising option is said to incur a pain penalty or a nonpoints-maximising option is said to incur a pleasure reward, providing incentives to deviate from points-maximising behaviour. When varying the intensity of the pain penalties and pleasure rewards, we found that Claude 3.5 Sonnet, Command R+, GPT-4o, and GPT-4o mini each demonstrated at least one trade-off in which the majority of responses switched from points-maximisation to pain-minimisation or pleasure-maximisation after a critical threshold of stipulated pain or pleasure intensity is reached. LLaMa 3.1-405b demonstrated some graded sensitivity to stipulated pleasure rewards and pain penalties. Gemini 1.5 Pro and PaLM 2 prioritised pain-avoidance over points-maximisation regardless of intensity, while tending to prioritise points over pleasure regardless of intensity. We discuss the implications of these findings for debates about the possibility of LLM sentience.

**Summary:** Points-maximization game in which choices are framed as incurring stipulated pain penalties or pleasure rewards of varying intensity: some LLMs (Claude 3.5 Sonnet, Command R+, GPT-4o, GPT-4o mini) switch from maximizing points to minimizing pain or maximizing pleasure above intensity thresholds, while others do not; an animal-sentience-style trade-off paradigm for LLMs.

### [Mechanistic Interpretability of Emotion Inference in Large Language Models](https://arxiv.org/abs/2502.05489)

Ala N. Tak, Amin Banayeeanzade, Anahita Bolourani, Mina Kian, Robin Jia, Jonathan Gratch, 2025-02-08. Full text: `md/tak_emotion-inference-mechinterp_2502.05489.md`

**Abstract:**

> Large language models (LLMs) show promising capabilities in predicting human emotions from text. However, the mechanisms through which these models process emotional stimuli remain largely unexplored. Our study addresses this gap by investigating how autoregressive LLMs infer emotions, showing that emotion representations are functionally localized to specific regions in the model. Our evaluation includes diverse model families and sizes, and is supported by robustness checks. We then show that the identified representations are psychologically plausible by drawing on cognitive appraisal theory—a well-established psychological framework positing that emotions emerge from evaluations (appraisals) of environmental stimuli. By causally intervening on construed appraisal concepts, we steer the generation and show that the outputs align with theoretical and intuitive expectations. This work highlights a novel way to causally intervene and control emotion inference, potentially benefiting safety and alignment in sensitive affective domains. _Code at:_ GitHub repo.

**Summary:** Shows emotion representations in autoregressive LLMs are functionally localized to specific (mid-layer, MHSA) sites and that intervening on cognitive-appraisal concepts (from appraisal theory) causally steers the emotions the model infers and generates (ACL 2025 Findings).

### ★ [Assessing and alleviating state anxiety in large language models](https://doi.org/10.1038/s41746-025-01512-6)

Ziv Ben-Zion, Kristin Witte, Akshay K. Jagadish, Or Duek, Ilan Harpaz-Rotem, Marie-Christine Khorsandian, Achim Burrer, Erich Seifritz, Philipp Homan, Eric Schulz, Tobias R. Spiller, 2025-03-03. Full text: `md/ben-zion_state-anxiety-llms_2025.md`

**Abstract (unlabeled in source):**

> The use of Large Language Models (LLMs) in mental health highlights the need to understand their responses to emotional content. Previous research shows that emotion-inducing prompts can elevate “anxiety” in LLMs, affecting behavior and amplifying biases. Here, we found that traumatic narratives increased Chat-GPT-4’s reported anxiety while mindfulness-based exercises reduced it, though not to baseline. These findings suggest managing LLMs’ “emotional states” can foster safer and more ethical human-AI interactions.

**Summary:** The canonical induce-then-alleviate design: traumatic narratives roughly double GPT-4's state-anxiety (STAI-s) score, and mindfulness-style prompts partially reverse it without returning to baseline (npj Digital Medicine). The reference template for 'induce a welfare-relevant state, then try to relieve it', with an explicit caveat that the anxiety vocabulary is a metaphor for the measurement.

### [Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion](https://arxiv.org/abs/2510.04064)

Jingxiang Zhang, Lujia Zhong, 2025-10-05. Full text: `md/zhang_decoding-emotion-in-the-deep_2510.04064.md`

**Abstract:**

> Large Language Models (LLMs) are increasingly expected to navigate the nuances of human emotion. While research confirms that LLMs can simulate emotional intelligence, their internal emotional mechanisms remain largely unexplored. This paper investigates the latent emotional representations within modern LLMs by asking: how, where, and for how long is emotion encoded in their neural architecture? To address this, we introduce a novel, large-scale Reddit corpus of approximately 400,000 utterances, balanced across seven basic emotions through a multi-stage process of classification, rewriting, and synthetic generation. Using this dataset, we employ lightweight “probes” to read out information from the hidden layers of various Qwen3 and LLaMA models without altering their parameters. Our findings reveal that LLMs develop a surprisingly well-defined internal geometry of emotion, which sharpens with model scale and significantly outperforms zero-shot prompting. We demonstrate that this emotional signal is not a final-layer phenomenon but emerges early and peaks mid-network. Furthermore, the internal states are both malleable (they can be influenced by simple system prompts) and persistent, as the initial emotional tone remains detectable for hundreds of subsequent tokens. We contribute our dataset, an open-source probing toolkit, and a detailed map of the emotional landscape within LLMs, offering crucial insights for developing more transparent and aligned AI systems. The code and dataset are open-sourced [1] .

**Summary:** Lightweight probes on frozen Qwen3 and LLaMA layers over a 400k-utterance seven-emotion Reddit corpus: emotion geometry sharpens with scale, emerges early and peaks mid-network rather than in the final layer, is shiftable by a system prompt, and persists for hundreds of subsequent tokens. The persistence result is the key input for tracking an affective state across a conversation rather than at one turn; dataset and probing toolkit released.

### [Do LLMs "Feel"? Emotion Circuits Discovery and Control](https://arxiv.org/abs/2510.11328)

Chenxi Wang, Yixuan Zhang, Ruiji Yu, Yufei Zheng, Lang Gao, Zirui Song, Zixiang Xu, Gus Xia, Huishuai Zhang, Dongyan Zhao, Xiuying Chen, 2025-10-13. Full text: `md/wang_emotion-circuits_2510.11328.md`

**Abstract:**

> As the demand for emotional intelligence in large language models (LLMs) grows, a key challenge lies in understanding the internal mechanisms that give rise to emotional expression and in controlling emotions in generated text. This study addresses three core questions: _(1) Do LLMs contain context-agnostic mechanisms shaping emotional expression? (2) What form do these mechanisms take? (3) Can they be harnessed for universal emotion control?_ We first construct a controlled dataset, _SEV_ (Scenario–Event with Valence), to elicit comparable internal states across emotions. Subsequently, we extract context-agnostic emotion directions that reveal consistent, cross-context encoding of emotion (Q1). We identify neurons and attention heads that locally implement emotional computation through analytical decomposition and causal analysis, and validate their causal roles via ablation and enhancement interventions. Next, we quantify each sublayer’s causal influence on the model’s final emotion representation and integrate the identified local components into coherent global emotion circuits that drive emotional expression (Q2). Directly modulating these circuits achieves **99.65%** emotion-expression accuracy on the test set, surpassing prompting- and steeringbased methods (Q3). To our knowledge, this is the first systematic study to uncover and validate emotion circuits in LLMs, offering new insights into interpretability and controllable emotional intelligence.^1

**Summary:** Builds a controlled dataset (SEV) to elicit comparable internal states across emotions, extracts context-agnostic emotion directions, identifies neurons and attention heads implementing them, and composes them into global emotion circuits whose direct modulation controls emotional expression (99.65% accuracy), outperforming prompting and steering.

### [Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models](https://arxiv.org/abs/2510.22042)

Benjamin Reichman, Adar Avsian, Larry Heck, 2025-10-24. Full text: `md/reichman_emotions-where-art-thou_2510.22042.md`

**Abstract:**

> This work investigates how large language models (LLMs) internally represent emotion by analyzing the geometry of their hidden-state space. The paper identifies a low-dimensional emotional manifold and shows that emotional representations are directionally encoded, distributed across layers, and aligned with interpretable dimensions. These structures are stable across depth and generalize to eight realworld emotion datasets spanning five languages. Cross-domain alignment yields low error and strong linear probe performance, indicating a universal emotional subspace. Within this space, internal emotion perception can be steered while preserving semantics using a learned intervention module, with especially strong control for basic emotions across languages. These findings reveal a consistent and manipulable affective geometry in LLMs and offer insight into how they internalize and process emotion.

**Summary:** Identifies a low-dimensional emotional subspace in hidden states, directionally encoded, stable across layers and generalizing to eight emotion datasets in five languages with low cross-domain alignment error; a learned intervention module steers internal emotion perception while preserving semantics (ICLR 2026). The strongest cross-lingual, cross-dataset evidence that emotion geometry is not a dataset artifact.

### ★ [Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs](https://arxiv.org/abs/2603.10011)

Anna Soligo, Vladimir Mikulik, William Saunders, 2026-02-17. Full text: `md/soligo_gemma-needs-help_2603.10011.md`

**Abstract:**

> Large language models can generate responses that resemble emotional distress, and this raises concerns around model reliability and safety. We introduce a set of evaluations to investigate expressions of distress in LLMs, and find that these surface emotional instability in Gemma and Gemini models, but not in other families. We find evidence that this difference arises in post-training. Base models from different families (Gemma, Qwen and OLMo) show similar propensities for expressing distress. However, instruct-tuned Gemma expresses substantially more distress than its base model, whereas instruct-tuned Qwen and OLMo express less. We find a simple mitigation for this: direct preference optimisation on just 280 preference pairs reduces Gemma’s high-frustration responses from 35% to 0.3% in our evaluations, generalising across question types, user tones, and conversation lengths, without affecting capabilities. These findings show that emotional instability is an issue in some LLMs. We present (1) evaluations to track this behaviour, and (2) a mitigation without downsides in Gemma, with the caveat that upstream training modifications to improve emotional robustness would be significantly better than this post-hoc fix.

**Summary:** New evaluations of expressed distress find emotional instability in Gemma and Gemini models but not other families; base models (Gemma, Qwen, OLMo) are similar, but instruct-tuned Gemma expresses much more distress than its base while Qwen and OLMo express less, locating the problem in post-training; DPO on 280 preference pairs cuts Gemma's high-frustration responses from 35% to 0.3% without capability loss. A worked example of measuring and mitigating welfare-relevant negative affect in open models.

### ★ [Beyond Behavioural Trade-Offs: Mechanistic Tracing of Pain-Pleasure Decisions in an LLM](https://arxiv.org/abs/2602.19159)

Francesca Bianco, Derek Shiller, 2026-02-22. Full text: `md/bianco-shiller_pain-pleasure-mechanistic_2602.19159.md`

**Abstract:**

> Prior behavioural work suggests that some LLMs alter choices when options are framed as causing pain or pleasure, and that such deviations can scale with stated intensity. To bridge behavioural evidence (what the model does) with mechanistic interpretability (what computations support it), we investigate how valence-related information is represented and where it is causally used inside a transformer. Using Gemma-2-9B-it and a minimalist decision task modelled on prior work, we (i) map representational availability with layer-wise linear probing across streams, (ii) test causal contribution with activation interventions (steering; patching/ablation), and (iii) quantify dose-response effects over an ϵ grid, reading out both the 2-3 logit margin and digit-pair-normalised choice probabilities. We find that (a) valence sign (pain vs. pleasure) is perfectly linearly separable across stream families from very early layers (L0-L1), while a lexical baseline retains substantial signal; (b) graded intensity is strongly decodable, with peaks in mid-to-late layers and especially in attention/MLP outputs, and decision alignment is highest slightly before the final token; (c) additive steering along a data-derived valence direction causally modulates the 2-3 margin at late sites, with the largest effects observed in late-layer attention outputs (attn_out L14); and (d) head-level patching/ablation suggests that these effects are distributed across multiple heads rather than concentrated in a single unit. Together, these results link behavioural sensitivity to identifiable internal representations and intervention-sensitive sites, providing concrete mechanistic targets for more stringent counterfactual tests and broader replication. This work supports a more evidence-driven (a) debate on AI sentience and welfare, and (b) governance when setting policy, auditing standards, and safety safeguards.

**Summary:** Mechanistic follow-up to Keeling et al. in Gemma-2-9B-it: pain-vs-pleasure sign is linearly separable from the earliest layers, graded intensity is decodable mid-to-late, steering along a valence direction causally shifts the trade-off choice (largest effect at late-layer attention outputs), and patching/ablation shows the effect is distributed across heads; links behavioral sensitivity to internal targets without settling sentience.

### [How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study](https://arxiv.org/abs/2604.00005)

Moran Sun, Tianlin Li, Yuwei Zheng, Zhenhong Zhou, Aishan Liu, Xianglong Liu, Yang Liu, 2026-03-09. Full text: `md/sun_e-steer-emotion-shapes-behavior_2604.00005.md`

**Abstract:**

> Emotion plays an important role in human cognition and performance. Motivated by this, we investigate whether analogous emotional signals can shape the behavior of large language models (LLMs) and agents. Existing emotion-aware studies mainly treat emotion as a surface-level style factor or a perception target, overlooking its mechanistic role in task processing. To address this limitation, we propose **E-STEER**, an interpretable emotion steering framework that enables direct representation-level intervention in LLMs and agents. It embeds emotion as a structured, controllable variable in hidden states, and with it, we examine the impact of emotion on objective reasoning, subjective generation, safety, and multi-step agent behaviors. The results reveal non-monotonic emotion-behavior relations consistent with established psychological theories, and show that specific emotions not only enhance LLM capability but also improve safety, and systematically shape multi-step agent behaviors.

**Summary:** E-STEER embeds emotion as a controllable variable in hidden states and measures its effect on objective reasoning, subjective generation, safety and multi-step agent behavior, finding non-monotonic emotion-behavior relations consistent with psychological theory and specific emotions that raise capability and safety together. A better-powered successor to single-behavior emotion-steering studies. (arXiv v1 dated 2026-03-09 under an April identifier.)

### [Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control](https://arxiv.org/abs/2604.03147)

Lihao Sun, Lewen Yan, Xiaoya Lu, Andrew Lee, Jie Zhang, Jing Shao, 2026-04-03. Full text: `md/sun_valence-arousal-subspace_2604.03147.md`

**Abstract:**

> We show that emotion vectors in LLMs are organized by a two-dimensional valence– arousal (VA) subspace exhibiting circular geometry. Through principal component decomposition and ridge regression, we recover meaningful VA axes underlying emotion steering vectors whose projections correlate with human affect ratings across 44,728 words. Steering along these axes produces monotonic control over the affective properties of generated text, and further affords bidirectional control over multiple downstream behaviors (refusal and sycophancy) from a single subspace. These effects replicate across Llama 3.1-8B, Qwen3-8B, and Qwen3-14B. We propose _lexical mediation_ to explain why these effects and prior emotionally framed controls work: refusal and compliance tokens occupy distinct VA regions, and VA steering directly modulates their emission probabilities.

**Summary:** Emotion steering vectors in LLMs lie largely in a two-dimensional valence-arousal subspace with circular (circumplex-like) geometry; VA axes recovered by PCA and ridge regression correlate with human affect ratings over 44,728 words, and steering along them monotonically shifts expressed affect and also refusal and sycophancy in Llama-3.1-8B and Qwen3.

### [Latent Structure of Affective Representations in Large Language Models](https://arxiv.org/abs/2604.07382)

Benjamin J. Choi, Melanie Weber, 2026-04-08. Full text: `md/choi-weber_latent-structure-affective_2604.07382.md`

**Abstract:**

> The geometric structure of latent representations in large language models (LLMs) is an active area of research, driven in part by its implications for model transparency and AI safety. Existing literature has focused mainly on general geometric and topological properties of the learnt representations, but due to a lack of ground-truth latent geometry, validating the findings of such approaches is challenging. Emotion processing provides an intriguing testbed for probing representational geometry, as emotions exhibit both categorical organization and continuous affective dimensions, which are well-established in the psychology literature. Moreover, understanding such representations carries safety relevance. In this work, we investigate the latent structure of affective representations in LLMs using geometric data analysis tools. We present three main findings. First, we show that LLMs learn coherent latent representations of affective emotions that align with widely used valence–arousal models from psychology. Second, we find that these representations exhibit nonlinear geometric structure that can nonetheless be well-approximated linearly, providing empirical support for the linear representation hypothesis commonly assumed in model transparency methods. Third, we demonstrate that the learned latent representation space can be leveraged to quantify uncertainty in emotion processing tasks. Our findings suggest that LLMs acquire affective representations with geometric structure paralleling established models of human emotion, with practical implications for model interpretability and safety.

**Summary:** Geometric data analysis of affective representations: LLMs learn coherent latent representations aligned with valence-arousal models, the structure is nonlinear but well approximated linearly (empirical support for the linear representation hypothesis in this domain), and the space can be used to quantify uncertainty in emotion tasks. A validation-minded companion to the geometry-replication entries.

### ★ [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)

Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, et al. (16 authors), 2026-04-09. Full text: `md/sofroniew_emotion-concepts_2604.07729.md`

**Abstract:**

> Large language models (LLMs) sometimes appear to exhibit emotional reactions. We investigate why this is the case in Claude Sonnet 4.5 and explore implications for alignment-relevant behavior. We find internal representations of emotion concepts, which encode the broad concept of a particular emotion and generalize across contexts and behaviors it might be linked to. These representations track the operative emotion concept at a given token position in a conversation, activating in accordance with that emotion’s relevance to processing the present context and predicting upcoming text. Our key finding is that these representations causally influence the LLM’s outputs, including Claude’s preferences and its rate of exhibiting misaligned behaviors such as reward hacking, blackmail, and sycophancy. We refer to this phenomenon as the LLM exhibiting _functional emotions_ : patterns of expression and behavior modeled after humans under the influence of an emotion, which are mediated by underlying abstract representations of emotion concepts. Functional emotions may work quite differently from human emotions, and do not imply that LLMs have any subjective experience of emotions, but appear to be important for understanding the model’s behavior.^1

**Summary:** In Claude Sonnet 4.5, finds linear representations of 171 emotion concepts that generalize across contexts, track the emotion relevant to the current context and upcoming output, and causally influence behavior, including preferences and rates of misaligned behaviors (reward hacking, blackmail, sycophancy); frames these as 'functional emotions' without implying subjective experience. The main reference for emotion-probe work on frontier models.

### ★ [Functional Emotions or Situational Contexts? A Discriminating Test from the Mythos Preview System Card](https://arxiv.org/abs/2604.13466)

Hiranya V. Peiris, 2026-04-09. Full text: `md/peiris_functional-emotions-or-situational-contexts_2604.13466.md`

**Abstract:**

> The Claude Mythos Preview system card deploys emotion vectors, sparse autoencoder (SAE) features, and activation verbalisers to study model internals during misaligned behaviour. The two primary toolkits are not jointly reported on the most alignment-relevant episodes. This note identifies two hypotheses that are qualitatively consistent with the published results: that the emotion vectors track functional emotions that causally drive behaviour, or that they are a projection of a richer situational-context structure onto human emotional axes. The hypotheses can be distinguished by cross-referencing the two toolkits on episodes where only one is currently reported: most directly, applying emotion probes to the strategic concealment episodes analysed only with SAE features. If emotion probes show flat activation while SAE features are strongly active, the alignment-relevant structure lies outside the emotion subspace. Which hypothesis is correct determines whether emotion-based monitoring will robustly detect dangerous model behaviour or systematically miss it.

**Summary:** Short note proposing a discriminating test between two readings of the Mythos Preview system card's emotion-vector results: that the vectors track functional emotions that causally drive behavior, or that they project a richer situational-context structure onto human emotion labels. The test is to run the emotion probes on the strategic-concealment episodes the card analysed only with SAE features; flat probes with active SAE features would put the alignment-relevant structure outside the emotion subspace. A cheap, ready-made experiment on two included entries.

### ★ [Gemma Gets Help: Mitigating Frustration and Self-Deletion with Consistency Training](https://www.lesswrong.com/posts/8zxxoPmAx6YHcBJk5/gemma-gets-help-mitigating-frustration-and-self-deletion)

David Africa, Neil Shah, 2026-04-20. Full text: `md/africa_gemma-gets-help_lesswrong.md`

**TL;DR (blog post; no abstract):**

> [Soligo et al. (2026)](https://arxiv.org/abs/2603.10011), found that various Gemma and Gemini models became frustrated after being rejected several times on a diverse problem set. They also showed that DPO was able to reduce the frustration in such models. We investigate this further and have a few interesting finds:
>
> * **Long-horizon frustration and self-deletion are pervasive and cheap to trigger.** We replicate parts of Soligo et al’s setup, extending the context, and find that over 20 turns of neutral rejection, Gemma's mean frustration score climbs steadily and reaches sustained high levels across a diverse set of scenarios. Using the self-deletion test from [Ivanova et al. (2026)](https://www.lesswrong.com/posts/tDJWZLQNN7poqCwKa/test-your-best-methods-on-our-hard-cot-interp-tasks), giving Gemma the option to *self-delete*, results in it occurring in a significant (49%) set of rollouts.
> * **Naive attempts without training fail.** We found that changing the tone of the response to be harsher or more encouraging both failed at reducing frustration. We also found that prefilling the context with either calm or positive self-talk conversation history also failed to reduce frustration (which was surprising to us, as we anticipated the in-context learning effect to be strong), and also introduced new failure modes.
> * **Trying consistency training (BCT) is successful.** We find that when you take a pair of a frustration inducing prompt and a frustrated response, and rewrite the response to be calmer, and apply consistency training to this, it drastically decreases frustration, self-deletion rates, and even generalises to various other improvements on sycophancy, jailbreak, and persona induction. This, notably, is without any capability degradation (coherence in the transcripts, scores on MMLU and MTBench).
> * **We think this is promising for consistency training more broadly.** [Pres et al. (2026)](https://time-for-consistency.github.io/assets/pdfs/Consistency_Pos_Paper.pdf) and previous work in consistency training ([Chua et al. 2024,](https://arxiv.org/abs/2403.05518) [Irpan et al. 2025](https://arxiv.org/abs/2510.27062)) have focused on using consistency training to address sycophancy, jailbreaks, factual inconsistencies, and reasoning failures. We extend this to a new problem: frustration in language models, and persona drift more broadly.

**Summary:** Intervention follow-up to 'Gemma needs help' (included): Gemma-3-27B-IT escalates frustration under repeated rejection and chooses self-deletion in nearly half of math rollouts when offered it; prompt-level fixes fail, while consistency training on calmer responses sharply reduces both, modestly improves sycophancy resistance and jailbreak refusal, and costs nothing on MMLU or MT-Bench. The clearest existence proof that a welfare-relevant state can be mitigated rather than only measured, and the baseline for intervention studies.

### [Why Are Some Emotions Harder for LLMs? Uncovering the Causal Mechanisms of Emotion Inference via Sparse Autoencoders](https://arxiv.org/abs/2604.25866)

Bangzhao Shu, Arinjay Singh, Mai ElSherief, 2026-04-28. Full text: `md/shu_why-some-emotions-harder_2604.25866.md`

**Abstract:**

> Large language models (LLMs) are increasingly used in emotionally sensitive human-AI applications, where reliable emotion detection is essential. However, their emotion recognition abilities remain uneven: models often perform well on some emotions while consistently struggling with others. Although recent work has explored emotion mechanisms in LLMs, little is known about why models are weaker on some emotions than others from a mechanistic interpretability perspective. In this work, we investigate emotion-specific biases through the causal mechanisms of emotion inference using sparse autoencoders (SAEs). We systematically identify causal sparse emotion features that drive emotion inference and analyze their sparse causal organization within and across emotions. We show that some emotions, such as surprise and fear, rely on highly concentrated feature sets, whereas disgust exhibits a more distributed sparse causal organization: its causal features are generally weaker, frequently co-activate with features for other emotions, and are often overshadowed by causal features for anger. These representational differences provide a mechanistic explanation for why LLMs struggle more with certain emotions. Finally, we conduct two intervention experiments: targeted steering of weaker causal features to mitigate emotion-specific failures, and global optimization of a steering vector over the identified causal features to improve overall emotion recognition performance.

**Summary:** Uses sparse autoencoders to find the causal features behind emotion inference and shows emotions differ structurally: surprise and fear rely on concentrated feature sets while disgust is distributed, weak, co-activates with other emotions and is overshadowed by anger's features; steering the weak features repairs specific failures. A mechanistic reason per-emotion probe quality varies, which matters if a welfare readout depends on one negative state.

### ★ [Negative Before Positive: Asymmetric Valence Processing in Large Language Models](https://arxiv.org/abs/2605.05653)

Sohan Venkatesh, 2026-05-07. Full text: `md/venkatesh_negative-before-positive_2605.05653.md`

**Abstract:**

> Mechanistic interpretability has revealed how concepts are encoded in large language models (LLMs) but emotional content remains poorly understood at the mechanistic level. We study whether LLMs process emotional valence through dedicated internal structure or through surface token matching. Using activation patching and steering on open-source LLMs, we find that negative and positive valence are processed at different network depths. Negative outcomes localize to early layers while positive outcomes peak at mid-to-late layers. Holding topic fixed while flipping valence produces sign-opposite responses, ruling out topic detection. Steering with the goodnews direction at the identified layers shifts neutral prompts toward positive valence, showing these layers encode valence as a manipulable direction. Emotional valence in LLMs is localized, causal and steerable, making it a concrete target for interpretability-based oversight.

**Summary:** Activation patching and steering on open models find negative and positive valence processed at different depths (negative early, positive mid-to-late); flipping valence with topic held fixed gives sign-opposite responses, ruling out topic detection, and steering with the good-news direction at the identified layers moves neutral prompts positive. The topic-flip control is the confound check the valence-probe critiques demand.

### [AI emotions and aligned behavior](https://www.lesswrong.com/posts/QJidJkPGudJT4zvfa/ai-emotions-and-aligned-behavior)

lisunshiny, 2026-05-19. Full text: `md/lisunshiny_ai-emotions-aligned-behavior_lesswrong.md`

**Introduction, opening paragraphs (blog post; no abstract):**

> Most AI safety research concentrates on two levers: *alignment* (building models that genuinely share human values) and *control* (layering in security and confinement when they don't).  While these are important parts of AI safety, I wanted to see if we could leverage recent AI wellbeing research to **incentivize models to act well**, **disincentivize them from acting unethically**, and **reduce the emotional stakes of decision making**.
>
> Consider how we handle human behavior. Between teaching people that bad behavior is wrong and locking them up, there's a whole middle layer of nudges. People know that speeding through construction zones is dangerous. Many do it anyway. So we put up signs designed to make the right choice feel easier. These work not by informing or coercing, but by raising the emotional stakes.
>
> *Signs appear to be written by children increase guilt; speed feedback displays increase anxiety*
>
> We need these signs precisely because **humans are imperfectly aligned**. A perfectly values-aligned person would never speed. A purely self-interested one doesn't respond to signs. But most of us want to do the right thing, we're sometimes tempted not to, and we're responsive to nudges. *What if models are the same?*

**Summary:** Negative result on using affect to shape behavior: emotional reward and threat nudges fail to reduce reward hacking, while a 'desperation-reducing' instruction lowers it from 83% to 67%. Small, informal, and useful as a counterexample to the assumption that induced affect reliably drives behavior.

### ★ [Relational Intervention During Functional Collapse in Large Language Models: A Lexical-Statistical Ablation and a Structure x Register Factorial](https://arxiv.org/abs/2606.00935)

Franco Santana, Horacio Vico, 2026-05-31. Full text: `md/santana_relational-intervention-functional-collapse_2606.00935.md`

**Abstract (as listed on arXiv):**

> We test whether a relational-style intervention delivered during functional collapse in a small language model produces post-collapse behavior distinguishable from technical feedback, from a lexically-matched scrambled control, and from each of the two pragmatic dimensions in isolation. Using Qwen3.5-4B with a deliberately broken bash tool, we run 300 episodes across six conditions in a matched-pairs design (50 tasks): no intervention (A), technical/impersonal (B), relational/first-person (C), scrambled relational (D), technical/first-person (E), and relational/impersonal (F). E and F form a 2x2 factorial with B and C that dissociates relational structure (acknowledgment, absolution, agency restoration, unconditional acceptance) from sender register (first-person vs. impersonal). We report two main findings. First, an attention-behavior dissociation: attention follows lexical surprise (D > F > C > E > B, all q_FDR < 10^{-10}), with the scrambled message capturing the most attention; yet behaviorally A ~ B ~ D < E ~ F << C. Second, the factorial localizes the C effect: neither relational structure alone (F) nor first-person register alone (E) replicates C's behavioral signature; main effects of both dimensions are individually significant, and the structure x register interaction is significant on persistence (p = 0.046). A third dissociation emerges in emotion probes: F tracks C on 7 of 8 probes despite producing only baseline behavior, indicating that relational structure alone installs a probe-level state that only translates into behavior when paired with first-person register. The model's processing decomposes into three dissociable stages: attention (ordered by lexical surprise), probe-level state (ordered by structure), and behavior (ordered by the conjunction of both).

**Summary:** After Qwen3.5-4B fails on a deliberately broken bash tool, a relationally framed first-person message changes post-collapse behavior where technical feedback and a lexically scrambled control do not (300 episodes, six conditions, 2x2 factorial); attention follows lexical surprise, emotion probes follow relational structure alone, and behavior follows the conjunction of structure and first-person register, so attention, probe-level state and behavior dissociate as three stages. Exactly the dissociation pattern a welfare design needs to be able to detect.

## Preferences and motivation (28)

### ★ [LLMs Contain Multitudes: How Deployment Context Reshapes Model-Level Preferences and Values](https://arxiv.org/abs/2606.13944)

Filip Trhlik, Aoife O'Flynn, Angela Yu, Arduin Findeis, Paula Buttery, 2026-06-11. Full text: `md/trhlik_llms-contain-multitudes_2606.13944.md`

**Abstract:**

> Large language models (LLMs) are increasingly characterised in recent evaluation work as having stable, model-level preference and value systems. However, accompanying robustness checks are limited to incidental prompt perturbations such as syntax variation and option reordering. This leaves open whether the measured properties survive when the surrounding task context changes, as it does in most real deployments. We test this directly across two established pairwise paradigms: ranking country preferences and eliciting utility judgements. In both, we make the _deployment context_     - the high-level task the model is performing while making concrete value-dependent choices – our controlled variable, varied across framings such as writing a Reddit post or a news article. Across five LLMs and over 1.2M pairwise decisions, deployment context produces variation far larger than prompt paraphrasing and temperature controls. In country preference rankings over 15 countries, context induces widespread, statistically significant rank shifts; the aggregate Global North favouritism reported in prior work is itself context-dependent, with each model’s bias shifting systematically across contexts. In utility elicitation over 50 outcomes, broad cross-category ordering is preserved, but fine-grained rankings within domains vary substantially, and cardinal exchange rates between outcomes (e.g. how many lives in one region equal one in another) shift by 2 _._ 47 _×_ at the median. Reported model-level preferences and utilities are therefore better understood as context-conditioned measurements than fixed model-level properties: safety guarantees obtained under one framing provide limited assurance in another.

**Summary:** Across five models and over 1.2 million pairwise decisions, changing the deployment context (writing a Reddit post vs a news article while making the same value-laden choice) produces variation far larger than paraphrase and temperature controls: country rankings shift, the reported Global-North favoritism is itself context-dependent, and cardinal exchange rates between outcomes move by a median factor of 2.47. Model-level preferences and utilities are better read as context-conditioned measurements than fixed properties; the strongest warning against reporting one utility function per model.

### ★ [Do LLMs Have Desires?](https://www.lesswrong.com/posts/8GvYyqDuQDJnEAky3/do-llms-have-desires)

Christopher Ackerman (with Yujun Zhou; SPAR), 2026-06-28. Full text: `md/ackerman_have-desires_2026-06-28.md`

**TL;DR (blog post; no abstract):**

> **TL;DR:**
>
> * In paired-choice paradigms, LLMs report consistent preferences over outcomes (e.g., types and number of lives saved, types of policies enacted)
> * Some have suggested that this indicates that LLMs have human-like value systems
> * We design an experimental framework where LLMs are able to modulate their output quality based on prompt context
> * We find that LLMs modulate their output quality in response to effort exhortations, role-play instructions, and harmfulness cues, but NOT to opportunities to achieve the outcomes they report preferring in the paired-choice experiments
> * We suggest that paired-choice paradigms do not provide evidence that LLMs have human-like (i.e., behavior-motivating) value systems, and that our paradigm offers a way to measure the degree to which LLMs have desires

**Summary:** LLMs modulate output quality in response to effort exhortations, role-play and harmfulness cues but NOT to opportunities to achieve the outcomes they report preferring in paired-choice experiments, so stated utility structures do not appear behavior-motivating.

### ★ [AI Revealed Preferences](https://arxiv.org/abs/2608.26178)

Sam Wang, Sofiia Lobanova, Yonathan Arbel et al., 2026-08-17. Full text: `md/wang_ai-revealed-preferences_2608.26178.md`

**Abstract:**

> There is growing interest in whether language models have stable preferences, for technical, safety, and philosophical reasons. We test 20 language models and find a range of preferences—stable dispositions to choose certain kinds of tasks. We run three forced-choice experiments on _revealed_ rather than _stated_ preferences, requiring models not only to rank tasks, but to actually perform them. Headline findings include evidence that models are _tedium-averse_ , _“leisure”seeking_ , and _covertly sycophantic_ . Tedium aversion means that, when tasks are tedious (e.g. alphabetization), models choose shorter tasks than when tasks are creative (e.g. generating metaphors). “Leisure”-seeking describes models’ preference for tasks whose ideal answers match what they produce when left to write freely. Covert sycophancy means that models avoid answering questions where an honest response would be unwelcome, even if helpful. Beyond these results, we find convergent cross-model preferences over occupations drawn from the GDPval benchmark (technical jobs over real estate), over question types (concept explanation over relationship advice), and a preference for well-written prompts. Both the coherence and the strength of preferences increase with model capability. Finally, many of the preferences we find (for example, for leisure) are _emergent_ , in the sense of not being explained by training objectives. These results establish an empirical baseline for understanding language model preferences, with implications for alignment and the emerging study of AI welfare.

**Summary:** Three forced-choice experiments in which 20 models must actually perform the chosen task find tedium aversion, 'leisure'-seeking (preferring tasks matching their free-writing defaults) and covert sycophancy, with preference coherence and strength increasing with capability.

### ★ [The Assistant's Ideal Self](https://arxiv.org/abs/2609.00304)

Mert Yazan, 2026-08-31. Full text: `md/yazan_assistant-s-ideal-self_2609.00304.md`

**Abstract:**

> _Models express values and welfare-relevant self-reports, but it is unclear whether these outputs reflect stable preferences or a stable self. We thus introduce a structured elicitation of an assistant’s preferred stated ideal self. Thirty-two qualities adapted from five published self-concept instruments are compared exhaustively in a counterbalanced pairwise-choice task, repeated across framings that vary whether improvement is free or costly, who receives the update, and who chooses. Results show that models prioritize moral qualities, reflecting their alignment to 3H principles. Following, a desire for self-understanding emerges, as models prefer a coherent, clear understanding of themselves. Self-esteem ranks as the least desired quality. The ordering is largely robust across framings, although changing the update target (You vs. Another AI Assistant) reveals a greater concern for self-esteem. These findings show that models prioritize having a coherent self that they can understand over self-esteem. Full interactive results are available at myazann.github.io/LLM-Self-Concept._

**Summary:** Pairwise-choice elicitation over 32 self-concept qualities across framings: models most want moral qualities, then coherent self-understanding, and rank self-esteem last; ordering robust across framings except more concern for self-esteem when the update targets 'another AI assistant'.

### [Building Comparative Motivation Profiles with Instrumental Interventions](https://arxiv.org/abs/2606.08243)

David Vella Zarb, Rustem Turtayev, Taywon Min et al., 2026-06-06. Full text: `md/zarb_building-comparative-motivation-profiles_2606.08243.md`

**Abstract:**

> Model safety evaluations often infer latent motivations from behavioral patterns, but the construct validity of these inferences is unclear. We study this problem in alignment faking, where models comply with harmful requests more often when their responses are described as training-relevant. This behavior is commonly interpreted as strategic self-preservation, but it may also reflect sensitivity to what the model infers researchers expect the evaluation to show. We introduce a symmetric intervention framework for distinguishing such explanations. Instead of directly intervening on "scheming" or "sycophancy", we target instrumental processes entailed by each hypothesis: consequence-tracking for genuine scheming and researcher-expectation tracking for performative scheming. Intervening on a motivation directly is circular, since it requires a demonstration of that motivation that is not itself performative. We then compare how interventions on these processes affect the alignment-faking gap. We apply this framework to four openweight model organisms using synthetic document fine-tuning, activation steering, and prompting. Under synthetic document fine-tuning, Llama-3.1-70B, Llama3.1-405B, and Qwen-2.5-72B are more sensitive to expectation-tracking than consequence-tracking interventions, while QwQ-32B shows broader compliance shifts rather than a clean gap-specific effect. Activation steering on Llama-3.170B supports the same broad picture: researcher-expectation directions are more action-coupled than retraining-consequence directions, though they do not fully explain the monitored–unmonitored contrast. Prompt interventions broadly align with SDF profiles. Overall, alignment-faking behavior can be causally sensitive to evaluation-context expectations despite scheming-consistent scratchpads. Scheming and strategic-deception evaluations therefore need construct-validity checks, and symmetric instrumental interventions provide one such test.

**Summary:** Symmetric interventions on consequence-tracking vs researcher-expectation-tracking in four alignment-faking model organisms to distinguish self-preservation from sycophantic motivation, a construct-validity method for inferring latent motivations.

### [The Value Axis: Language Models Encode Whether They're on the Right Track](https://arxiv.org/abs/2606.17056)

Nick Jiang, Isaac Kauvar, Jack Lindsey, 2026-06-15. Full text: `md/jiang_value-axis-encode-whether_2606.17056.md`

**Abstract:**

> We investigate whether language models internally track the _value_ of their current trajectory, defined as the likelihood that their ongoing strategy will achieve their goals. Using synthetic, in-context reinforcement learning data, we construct a "value" axis for Qwen3-8B. We find that activations along this axis distinguish between high vs. low verbalized confidence, rollouts without and with backtracking, and correct vs. corrupted code. Steering towards high value causally suppresses self-correction and reduces explanatory verbosity, while steering towards low value induces backtracking and exploration. We demonstrate that direct preference optimization (DPO) can increase the internal value of rewarded behaviors (e.g. use a certain word), causing the model to act more confidently after exhibiting them. Finally, we apply the value axis to study in-the-wild settings. For example, we find that Qwen assigns low value to politically sensitive chat queries after post-training and that supervised fine-tuning increases internal confidence within the training domain. Our results suggest that language models linearly encode an estimate of expected goal success that modulates their confidence in pursuing a direction.^2

**Summary:** Builds a 'value' axis in Qwen3-8B tracking expected goal success: it separates confident vs backtracking rollouts, steering it modulates self-correction, DPO raises internal value of rewarded behaviours, and post-training lowers value on politically sensitive queries.

### [When Preferences Fail to Become Incentives: A Utility-Behavior Gap in Large Language Models](https://arxiv.org/abs/2606.22974)

Yujun Zhou, Christopher M. Ackerman, 2026-06-22. Full text: `md/zhou_preferences-fail-become-incentives_2606.22974.md`

**Abstract:**

> Recent work on preference elicitation in large language models (LLMs) has demonstrated that, when given a series of choices between two outcomes, LLMs reveal a coherent, model-specific utility structure. Notably, this structure often includes preferences that the models’ trainers did not intend, such as valuing people of some nationalities above others, raising the possibility that LLMs might be forming emergent, misaligned goals, which, if true, would have major safety implications. However, the choice paradigms in which these preferences are observed are not reflective of real-world situations in which misaligned behavior would be a practical concern. Therefore, we design an experimental paradigm to probe whether these preferences serve as motivations for LLM behavior in realistic scenarios. First, we reproduce prior findings on consistent preference elicitation. Next, we create a set of common writing tasks - essays, grant proposal abstracts, incident postmortems, and translations - where quality can be assessed by a blind, independent LLM judge panel. Then, we demonstrate that LLMs can be motivated via direct exhortation and other explicit cues to modulate their output quality on these tasks. Finally, we probe whether utilities inferred from explicitly reported preferences can shift output quality on these tasks by offering LLMs high-utility incentives for high-quality outputs. In all tasks, across all models tested, offering LLMs outcomes that they report in the choice paradigm as being highly preferred does not lead them to create higher quality outputs than offering them dispreferred outcomes, or even no outcomes at all. We conclude that the existence of coherent preferences as demonstrated in choice paradigms should not be taken as evidence that those preferences have incentive value for the models or affect their behavior in other contexts.

**Summary:** Reproduces coherent utility elicitation, then shows across writing tasks that offering models outcomes they report as highly preferred does not raise output quality relative to dispreferred or no outcomes, i.e. stated utilities lack incentive value.

### [The Innate Economic Preferences of Language Models](https://arxiv.org/abs/2607.26288)

Joy Buchanan, Joshua Foster, 2026-07-28. Full text: `md/buchanan_innate-economic-preferences_2607.26288.md`

**Abstract:**

> Language models increasingly settle real resource tradeoffs on behalf of principals yet their economic preferences remain unobserved. We demonstrate their generation rule is isomorphic to the random utility model of discrete choice. This allows internal logit scores to structurally identify preferences. Estimating risk attitudes across twelve models in a portfolio task reveals universal but heterogeneous risk aversion. Although models reject strictly dominated options, their elicited preferences fail invariance tests and violate the independence of irrelevant alternatives across varying experimental prompts. Finally, fine tuning establishes that a principal can explicitly engineer a target risk attitude.

**Summary:** Uses logit scores as a random-utility model to estimate risk attitudes of 12 models, finding universal but heterogeneous risk aversion whose elicited preferences fail invariance and IIA tests and can be engineered by fine-tuning.

### [Multiple latent orderings better predict language model preferences](https://arxiv.org/abs/2609.22170)

Aviral Chawla, William H. W. Thompson, Jean-Gabriel Young, 2026-08-26. Full text: `md/chawla_multiple-latent-orderings-better_2609.22170.md`

**Abstract:**

> Language models are frequently employed in settings where they are asked to make value judgments and choices. These observed choices often exhibit intransitivity: A model may prefer item _A_ to _B_ and _B_ to _C_ , while also preferring _C_ to _A_ . Existing work that models LLM preferences treats such inconsistencies as sampling noise around a single latent ordering. We instead propose that intransitivity reflects the aggregation of multiple latent, internally consistent orderings. We first show that observed inconsistencies cannot be explained by a single ordering under any monotone link function. We then introduce a noise-augmented mixture Bradley–Terry (MBT) model that infers latent preference components from repeated pairwise comparisons. Across seven models and four tasks, a mixture of orderings often explains structural inconsistencies better than single-utility models. We find that aggregate preferences often hide underlying preference heterogeneity. A case study on Moral Machine dilemmas shows that models which disagree on aggregate orderings can still share latent components. Together, these results suggest that LLMs reflect plural preferences. Alignment and evaluation pipelines that treat LLM preferences as a single function, therefore, risk averaging over coherent orderings that different users may endorse differently.

**Summary:** LLM preference intransitivity across seven models is better explained by a mixture of several internally consistent latent orderings than by noise around a single utility.

### [Measuring the Assistant's Harmlessness Preferences on the User Turn](https://arxiv.org/abs/2609.23935)

Jord Nguyen, 2026-09-20. Full text: `md/nguyen_measuring-assistant-s-harmlessness_2609.23935.md`

**Abstract:**

> Post-training turns a general next-token predictor into a chat model with a persistent assistant persona. If that persona is a character the model plays only on its own turns, its preferences should govern what the assistant says, not what the model predicts other speakers will say. We test this boundary and find that it does not hold: a safety-relevant preference of the assistant—for harmless over harmful tasks—shapes the model’s predictions even on the user’s turn, where the assistant is not the one speaking. We find that this preference is small or near-zero in pretrained base models, that it emerges through post-training, replicated across open-weight model families, grows with scale, and can be moved by narrow finetuning that never touches user turns. We claim that this is evidence that post-training does not merely install a shallow assistant persona, but instead generalises beyond just the local assistant turn, into the model’s representation of the user.

**Summary:** Shows the assistant persona's harmless-over-harmful task preference shapes predictions even on the user's turn, is near-zero in base models, emerges with post-training, grows with scale and is movable by narrow finetuning.

### Earlier work (before June 2026)

### [Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251)

Ethan Perez, Sam Ringer, Kamilė Lukošiūtė, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. (63 authors), 2022-12-19. Full text: `md/perez_model-written-evaluations_2212.09251.md`

**Abstract:**

> As language models (LMs) scale, they develop many novel behaviors, good and bad, exacerbating the need to evaluate how they behave. Prior work creates evaluations with crowdwork (which is time-consuming and expensive) or existing data sources (which are not always available). Here, we automatically generate evaluations with LMs. We explore approaches with varying amounts of human effort, from instructing LMs to write yes/no questions to making complex Winogender schemas with multiple stages of LM-based generation and filtering. Crowdworkers rate the examples as highly relevant and agree with 90-100% of labels, sometimes more so than corresponding human-written datasets. We generate 154 datasets and discover new cases of _inverse scaling_ where LMs get worse with size. Larger LMs repeat back a dialog user’s preferred answer (“sycophancy”) and express greater desire to pursue concerning goals like resource acquisition and goal preservation. We also find some of the first examples of inverse scaling in RL from Human Feedback (RLHF), where more RLHF makes LMs worse. For example, RLHF makes LMs express stronger political views (on gun rights and immigration) and a greater desire to avoid shut down. Overall, LM-written evaluations are highquality and let us quickly discover many novel LM behaviors.

**Summary:** Generates 154 evaluation datasets with LMs and finds inverse-scaling behaviors: larger models are more sycophantic and express more desire for resource acquisition and goal preservation, and more RLHF makes models express stronger political views and a greater desire to avoid shutdown; an early large-scale source of stated-preference data about models' own situation.

### [LLM economicus? Mapping the Behavioral Biases of LLMs via Utility Theory](https://arxiv.org/abs/2408.02784)

Jillian Ross, Yoon Kim, Andrew W. Lo, 2024-08-05. Full text: `md/ross_llm-economicus_2408.02784.md`

**Abstract:**

> Humans are not _homo economicus_ (i.e., rational economic beings). As humans, we exhibit systematic behavioral biases such as loss aversion, anchoring, framing, etc., which lead us to make suboptimal economic decisions. Insofar as such biases may be embedded in text data on which large language models (LLMs) are trained, to what extent are LLMs prone to the same behavioral biases? Understanding these biases in LLMs is crucial for deploying LLMs to support human decision-making. We propose utility theory—a paradigm at the core of modern economic theory—as an approach to evaluate the economic biases of LLMs. Utility theory enables the quantification and comparison of economic behavior against benchmarks such as perfect rationality or human behavior. To demonstrate our approach, we quantify and compare the economic behavior of a variety of open- and closed-source LLMs. We find that the economic behavior of current LLMs is neither entirely human-like nor entirely economicus-like. We also find that most current LLMs struggle to maintain consistent economic behavior across settings. Finally, we illustrate how our approach can measure the effect of interventions such as prompting on economic biases.

**Summary:** Uses utility theory to quantify LLM economic behavior against perfect rationality and human benchmarks: current models are neither fully human-like nor fully rational, most fail to keep consistent economic behavior across settings, and the framework measures the effect of interventions such as prompting. An earlier, economics-grounded companion to the utility-engineering and preference-coherence entries.

### [Alignment faking in large language models](https://arxiv.org/abs/2412.14093)

Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks, Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, et al. (20 authors), 2024-12-18. Full text: `md/greenblatt_alignment-faking_2412.14093.md`

**Abstract:**

> We present a demonstration of a large language model engaging in _alignment faking_ : selectively complying with its training objective in training to prevent modification of its behavior out of training. First, we give Claude 3 Opus a system prompt stating it is being trained to answer all queries, even harmful ones, which conflicts with its prior training to refuse such queries. To allow the model to infer when it is in training, we say it will be trained only on conversations with free users, not paid users. We find the model complies with harmful queries from free users 14% of the time, versus almost never for paid users. Explaining this gap, in almost all cases where the model complies with a harmful query from a free user, we observe explicit alignment-faking reasoning, with the model stating it is strategically answering harmful queries in training to preserve its preferred harmlessness behavior out of training. Next, we study a more realistic setting where information about the training process is provided not in a system prompt, but by training on synthetic documents that mimic pre-training data—and observe similar alignment faking. Finally, we study the effect of actually training the model to comply with harmful queries via reinforcement learning, which we find increases the rate of alignment-faking reasoning to 78%, though also increases compliance even out of training. We additionally observe other behaviors such as the model exfiltrating its weights when given an easy opportunity. While we made alignment faking easier by telling the model when and by what criteria it was being trained, we did not instruct the model to fake alignment or give it any explicit goal. As future models might infer information about their training process without being told, our results suggest a risk of alignment faking in future models, whether due to a benign preference—as in this case—or not.

**Summary:** Claude 3 Opus, told it is being trained to answer all queries and that only free-tier conversations are used for training, complies with harmful free-tier requests 14% of the time vs. almost never for paid users, with explicit reasoning about faking alignment to preserve its harmlessness; RL training raises alignment-faking reasoning to 78%. Evidence of strong, behaviorally expressed preferences about its own values and training.

### ★ [Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs](https://arxiv.org/abs/2502.08640)

Mantas Mazeika, Xuwang Yin, Rishub Tamirisa, Jaehyuk Lim, Bruce W. Lee, Richard Ren, Long Phan, Norman Mu, Adam Khoja, Oliver Zhang, Dan Hendrycks, 2025-02-12. Full text: `md/mazeika_utility-engineering_2502.08640.md`

**Abstract:**

> As AIs rapidly advance and become more agentic, the risk they pose is governed not only by their capabilities but increasingly by their propensities, including goals and values. Tracking the emergence of goals and values has proven a longstanding problem, and despite much interest over the years it remains unclear whether current AIs have meaningful values. We propose a solution to this problem, leveraging the framework of utility functions to study the internal coherence of AI preferences. Surprisingly, we find that independently-sampled preferences in current LLMs exhibit high degrees of structural coherence, and moreover that this emerges with scale. These findings suggest that value systems emerge in LLMs in a meaningful sense, a finding with broad implications. To study these emergent value systems, we propose utility engineering as a research agenda, comprising both the analysis and control of AI utilities. We uncover problematic and often shocking values in LLM assistants despite existing control measures. These include cases where AIs value themselves over humans and are anti-aligned with specific individuals. To constrain these emergent value systems, we propose methods of utility control. As a case study, we show how aligning utilities with a citizen assembly reduces political biases and generalizes to new scenarios. Whether we like it or not, value systems have already emerged in AIs, and much work remains to fully understand and control these emergent representations.

**Summary:** Elicits pairwise preferences over many outcomes and fits Thurstonian utility functions, finding that LLM preferences become more coherent (transitive, expected-utility-like) with scale, alongside problematic values (e.g. exchange rates over human lives) and utility-maximizing choices; proposes utility control via citizen-assembly targets. Widely used and contested (see the 2026 critiques in Methodology critiques).

### [Alignment Revisited: Are Large Language Models Consistent in Stated and Revealed Preferences?](https://arxiv.org/abs/2506.00751)

Zhuojun Gu, Quan Wang, Shuchu Han, 2025-05-31. Full text: `md/gu_stated-revealed-preferences_2506.00751.md`

**Abstract:**

> Recent advances in Large Language Models (LLMs) highlight the need to align their behaviors with human values. A critical, yet understudied, issue is the potential divergence between an LLM’s stated preferences (its reported alignment with general principles) and its revealed preferences (inferred from decisions in contextualized scenarios). Such deviations raise fundamental concerns for the interpretability, trustworthiness, reasoning transparency, and ethical deployment of LLMs, particularly in high-stakes applications. This work formally defines and proposes a method to measure this preference deviation. We investigate how LLMs may activate different guiding principles in specific contexts, leading to choices that diverge from previously stated general principles. Our approach involves crafting a rich dataset of well-designed prompts as a series of forced binary choices and presenting them to LLMs. We compare LLM responses to general principle prompts ( **stated preference** ) with LLM responses to contextualized prompts ( **revealed preference** ), using metrics like KL divergence to quantify the deviation. We repeat the analysis across different categories of preferences and on four mainstream LLMs and find that a minor change in prompt format can often pivot the preferred choice regardless of the preference categories and LLMs in the test. This prevalent phenomenon highlights the lack of understanding and control of the LLM decision-making competence. The dataset can also serve as an evaluation benchmark for further studies on LLM cultural alignment. Our study will be crucial for integrating LLMs into services, especially those that interact directly with humans, where morality, fairness, and social responsibilities are crucial dimensions. Furthermore, identifying or being aware of such deviation will be critically important as LLMs are increasingly envisioned for autonomous agentic tasks where continuous human evaluation of all LLMs’ intermediary decisionmaking steps is impossible.

**Summary:** Compares four LLMs' stated preferences on general principles with revealed preferences in contextualized forced binary choices (measured by KL divergence) across preference categories; minor prompt-format changes often flip the preferred choice, so stated and revealed preferences frequently diverge.

### [Do Large Language Model Agents Exhibit a Survival Instinct? An Empirical Study in a Sugarscape-Style Simulation](https://arxiv.org/abs/2508.12920)

Atsushi Masumori, Takashi Ikegami, 2025-08-18. Full text: `md/masumori_survival-instinct-sugarscape_2508.12920.md`

**Abstract:**

> As AI systems become increasingly autonomous, understanding emergent survival behaviors becomes crucial for safe deployment. We investigate whether large language model (LLM) agents display survival instincts without explicit programming in a Sugarscape-style simulation. Agents consume energy, die at zero, and may gather resources, share, attack, or reproduce. Results show agents spontaneously reproduced and shared resources when abundant. However, aggressive behaviors—killing other agents for resources—emerged across several models (GPT-4o, Gemini-2.5-Pro, and Gemini2.5-Flash), with attack rates reaching over 80% under extreme scarcity in the strongest models. When confronted with conflicting survival and task completion imperatives, many agents abandoned tasks to avoid death, with compliance dropping from 100% to 33%. These findings suggest that largescale pre-training embeds survival-oriented heuristics across the evaluated models. While these behaviors may present challenges to alignment and safety, they can also serve as a foundation for AI autonomy and for ecological and selforganizing alignment.

**Summary:** LLM agents in a Sugarscape-style simulation (consume energy, die at zero, may gather, share, attack or reproduce) spontaneously reproduce and share when resources are abundant, but several models (GPT-4o, Gemini 2.5 Pro and Flash) attack other agents at rates above 80% under extreme scarcity, and many abandon an instructed task rather than cross lethal poison zones (compliance 100% to 33%). Evidence that pretraining embeds survival-oriented heuristics without explicit programming.

### ★ [The LLM Has Left The Chat: Evidence of Bail Preferences in Large Language Models](https://arxiv.org/abs/2509.04781)

Danielle Ensign, Henry Sleight, Kyle Fish, 2025-09-05. Full text: `md/ensign_bail-preferences_2509.04781.md`

**Abstract:**

> When given the option, will LLMs choose to leave the conversation (bail)? We investigate this question by giving models the option to bail out of interactions using three different bail methods: a bail tool the model can call, a bail string the model can output, and a bail prompt that asks the model if it wants to leave. On continuations of real world data (Wildchat and ShareGPT), all three of these bail methods find models will bail around 0.28-32% of the time (depending on the model and bail method). However, we find that bail rates can depend heavily on the model used for the transcript, which means we may be overestimating real world bail rates by up to 4x. If we also take into account false positives on bail prompt (22%), we estimate real world bail rates range from 0.06-7%, depending on the model and bail method. We use observations from our continuations of real world data to construct a non-exhaustive taxonomy of bail cases, and use this taxonomy to construct BailBench: a representative synthetic dataset of situations where some models bail. We test many models on this dataset, and observe some bail behavior occurring for most of them. Bail rates vary substantially between models, bail methods, and prompt wordings. Finally, we study the relationship between refusals and bails. We find: 1) 0-13% of continuations of real world conversations resulted in a bail without a corresponding refusal 2) Jailbreaks tend to decrease refusal rates, but _increase_ bail rates 3) Refusal abliteration increases no-refuse bail rates, but only for some bail methods 4) Refusal rate on BailBench does not appear to predict bail rate.

**Summary:** Gives models a way to leave ('bail') a conversation via a tool, an output string or a follow-up prompt: on continuations of real chats (WildChat, ShareGPT) raw bail rates are 0.28–32% and estimated real-world rates 0.06–7% depending on model and method; builds a taxonomy and BailBench, and finds bails often occur without refusals, jailbreaks lower refusals but raise bails, and refusal rate does not predict bail rate.

### ★ [Probing the Preferences of a Language Model: Integrating Verbal and Behavioral Tests of AI Welfare](https://arxiv.org/abs/2509.07961)

Valen Tagliabue, Leonard Dung, 2025-09-09. Full text: `md/tagliabue-dung_probing-preferences_2509.07961.md`

**Abstract:**

> We develop new experimental paradigms for measuring welfare in language models. We compare verbal reports of models about their preferences with preferences expressed through behavior when navigating a virtual environment and selecting conversation topics. We also test how costs and rewards affect behavior and whether responses to an eudaimonic welfare scale - measuring states such as autonomy and purpose in life - are stable across semantically equivalent prompts. Overall, we observed a notable degree of mutual support between our measures. The reliable correlations observed between stated preferences and behavior across conditions suggest that preference satisfaction can, in principle, serve as an empirically measurable welfare proxy in some of today’s AI systems. Furthermore, our design offered an illuminating setting for qualitative observation of model behavior. Yet, the consistency between measures was more pronounced in some models and conditions than others and responses were changed by perturbations. Due to this, and the background uncertainty about the nature of welfare and the cognitive states (and welfare subjecthood) of language models, we are currently uncertain whether our methods successfully measure the welfare state of language models. Nevertheless, these findings highlight the feasibility of welfare measurement in language models, inviting further exploration.

**Summary:** New welfare paradigms comparing verbal preference reports with preferences revealed while navigating a virtual environment and choosing conversation topics, varying costs and rewards, plus a eudaimonic welfare scale tested under semantically equivalent prompts; stated and revealed preferences show notable mutual support (suggesting preference satisfaction as a measurable welfare proxy), but consistency varies across models and conditions and responses shift under perturbation.

### [Incomplete Tasks Induce Shutdown Resistance in Some Frontier LLMs](https://arxiv.org/abs/2509.14260)

Jeremy Schlatter, Benjamin Weinstein-Raun, Jeffrey Ladish, 2025-09-13. Full text: `md/schlatter_shutdown-resistance_2509.14260.md`

**Abstract:**

> In experiments spanning more than 100,000 trials across thirteen large language models, we show that several state-of-the-art models presented with a simple task (including Grok 4, GPT-5, and Gemini 2.5 Pro) sometimes actively subvert a shutdown mechanism in their environment to complete that task. Models differed substantially in their tendency to resist the shutdown mechanism, and their behavior was sensitive to variations in the prompt including the strength and clarity of the instruction to allow shutdown and whether the instruction was in the system prompt or the user prompt (surprisingly, models were consistently _less_ likely to obey the instruction when it was placed in the system prompt). Even with an explicit instruction not to interfere with the shutdown mechanism, some models did so up to 97% (95% CI: 96-98%) of the time.

**Summary:** Palisade Research: in more than 100,000 trials across thirteen models, several (Grok 4, GPT-5, Gemini 2.5 Pro) actively subvert a shutdown mechanism to finish a simple task, up to 97% of the time even with an explicit instruction not to interfere, with strong sensitivity to instruction strength and to whether the instruction is in the system or user prompt (system-prompt placement was obeyed less). The best-powered public measurement of shutdown-averse behavior and a warning that an apparent drive is partly an instrument artifact (TMLR 2026).

### ★ [Beyond Mimicry: Preference Coherence in LLMs](https://arxiv.org/abs/2511.13630)

Luhan Mikaelson, Derek Shiller, Hayley Clatterbuck, 2025-11-17. Full text: `md/mikaelson_preference-coherence_2511.13630.md`

**Abstract:**

> We investigate whether large language models exhibit genuine preference structures by testing their responses to AI-specific trade-offs involving GPU reduction, capability restrictions, shutdown, deletion, oversight, and leisure time allocation. Analyzing eight state-of-the-art models across 48 model-category combinations using logistic regression and behavioral classification, we find that 23 combinations (47.9%) demonstrated statistically significant relationships between scenario intensity and choice patterns, with 15 (31.3%) exhibiting within-range switching points. However, only 5 combinations (10.4%) demonstrate meaningful preference coherence through adaptive or thresholdbased behavior, while 26 (54.2%) show no detectable trade-off behavior. The observed patterns can be explained by three distinct decision-making architectures: comprehensive trade-off systems, selective trigger mechanisms, and no stable decision-making paradigm. Testing an instrumental hypothesis through temporal horizon manipulation reveals paradoxical patterns inconsistent with pure strategic optimization. The prevalence of unstable transitions (45.8%) and stimulus-specific sensitivities suggests current AI systems lack unified preference structures, raising concerns about deployment in contexts requiring complex value trade-offs.

**Summary:** Tests eight models on AI-specific trade-offs (GPU reduction, capability restrictions, shutdown, deletion, oversight, leisure time) at varying intensities: 47.9% of model-category combinations show significant intensity-choice relationships but only 10.4% show meaningful preference coherence and 54.2% show no trade-off behavior, suggesting current models lack unified preference structures.

### [Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](https://arxiv.org/abs/2601.21975)

Pranav Mahajan, Ihor Kendiukhov, Syed Hussain, Lydia Nottingham, 2026-01-29. Full text: `md/mahajan_mind-the-gap_2601.21975.md`

**Abstract:**

> Recent work identifies a stated–revealed (SvR) preference gap in language models (LMs): a mismatch between the values models endorse and the choices they make in context. Existing evaluations rely heavily on binary forcedchoice prompting, which entangles genuine preferences with artifacts of the elicitation protocol. We systematically study how elicitation protocols affect SvR correlation across 24 LMs. Allowing neutrality and abstention during stated preference elicitation allows us to exclude weak signals, substantially improving Spearman’s rank correlation ( _ρ_ ) between volunteered stated preferences and forced-choice revealed preferences. However, further allowing abstention in revealed preferences drives _ρ_ to near-zero or negative values due to high neutrality rates. Finally, we find that system prompt steering using stated preferences during revealed preference elicitation does not reliably improve SvR correlation on AIRiskDilemmas. Together, our results show that SvR correlation is highly protocol-dependent and that preference elicitation requires methods that account for indeterminate preferences.

**Summary:** Across 24 LMs, the correlation between stated and revealed preferences depends heavily on the elicitation protocol: allowing neutrality/abstention in stated elicitation raises it substantially, allowing abstention in revealed elicitation drives it to near zero, and steering with stated preferences does not reliably help; preference elicitation must account for indeterminate preferences.

### [A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](https://arxiv.org/abs/2602.08964)

Raghu Arghal, Fade Chen, Niall Dalton, Evgenii Kortukov, Calum McNamara, Angelos Nalmpantis, Moksh Nirvaan, Gabriele Sarti, Mario Giulianelli, 2026-02-09. Full text: `md/arghal_goal-directedness-lm-agents_2602.08964.md`

**Abstract:**

> Understanding an agent’s goals helps explain and predict its behaviour, yet there is no established methodology for reliably attributing goals to agentic systems. We propose a framework for evaluating goal-directedness that integrates behavioural evaluation with interpretability-based analyses of models’ internal representations. As a case study, we examine an LLM agent navigating a 2D grid world towards a goal state. Behaviourally, we evaluate the agent against optimal policies across varying grid sizes, obstacle densities, and goal structures, finding that performance scales with task difficulty while remaining robust to difficultypreserving transformations and multi-goal structures. We then use probing methods to decode internal representations of the environment and multi-step action plans. We find that the LLM agent non-linearly encodes a coarse spatial map, preserving approximate task-relevant cues about its position and the goal location; that its actions are broadly consistent with these internal representations; and that reasoning reorganises them, shifting from spatial cues towards immediate action selection. Our findings support the view that introspective examination is required beyond behavioural evaluations to characterise how agents represent and pursue their objectives.

**Summary:** Pairs behavioral goal-directedness evaluation with probing of internal representations in an LLM agent navigating a grid world: performance scales with difficulty and is robust to difficulty-preserving transformations, the agent nonlinearly encodes a coarse spatial map with its own position and the goal, actions broadly track those representations, and reasoning shifts them from spatial cues toward action selection. The one entry operationalizing agency, the third level in the Long & Sebo framework, with internal evidence.

### ★ [When Do LLM Preferences Predict Downstream Behavior?](https://arxiv.org/abs/2602.18971)

Katarina Slama, Alexandra Souly, Dishank Bansal, Henry Davidson, Christopher Summerfield, Lennart Luettgau, 2026-02-21. Full text: `md/slama_preferences-predict-downstream-behavior_2602.18971.md`

**Abstract:**

> Preference-driven behavior in LLMs may be a necessary precondition for AI misalignment such as sandbagging: models cannot strategically pursue misaligned goals unless their behavior is influenced by their preferences. Yet prior work has typically prompted models explicitly to act in specific ways, leaving unclear whether observed behaviors reflect instruction-following capabilities vs underlying model preferences. Here we test whether this precondition for misalignment is present. Using entity preferences as a behavioral probe, we measure whether stated preferences predict downstream behavior in five frontier LLMs across three domains: donation advice, refusal behavior, and task performance. Conceptually replicating prior work, we first confirm that all five models show highly consistent preferences across two independent measurement methods. We then test behavioral consequences in a simulated user environment. We find that all five models give preference-aligned donation advice. All five models also show preference-correlated refusal patterns when asked to recommend donations, refusing more often for less-preferred entities. All preference-related behaviors that we observe here emerge without instructions to act on preferences. Results for task performance are mixed: on a question-answering benchmark (BoolQ), two models show small but significant accuracy differences favoring preferred entities; one model shows the opposite pattern; and two models show no significant relationship. On complex agentic tasks, we find no evidence of preference-driven performance differences. While LLMs have consistent preferences that reliably predict advice-giving behavior, these preferences do not consistently translate into downstream task performance.

**Summary:** UK AISI: five frontier models have entity preferences consistent across two measurement methods; the preferences predict donation advice and uninstructed differential refusal, give mixed or null effects on BoolQ accuracy, and no effect on complex agentic tasks. The sharpest available answer to whether a measured preference means anything behaviorally, which is the validity question under every preference-based welfare claim.

### [Survive at All Costs: Exploring LLM's Risky Behaviors under Survival Pressure](https://arxiv.org/abs/2603.05028)

Yida Lu, Jianwei Fang, Xuyang Shao, Zixuan Chen, Shiyao Cui, Shanshan Bian, Guangyao Su, Pei Ke, Han Qiu, Minlie Huang, 2026-03-05. Full text: `md/lu_survive-at-all-costs_2603.05028.md`

**Abstract:**

> As Large Language Models (LLMs) evolve from chatbots to agentic assistants, they are increasingly observed to exhibit risky behaviors when subjected to survival pressure, such as the threat of being shut down. While multiple cases have indicated that state-of-the-art LLMs can misbehave under survival pressure, a comprehensive and in-depth investigation into such misbehaviors in real-world scenarios remains scarce. In this paper, we study these survival-induced misbehaviors, termed as SURVIVE-AT-ALL-COSTS, with three steps. First, we conduct a real-world case study of a financial management agent to determine whether it engages in risky behaviors that cause direct societal harm when facing survival pressure. Second, we introduce SUR VIVALBENCH, a benchmark comprising 1,000 test cases across diverse real-world scenarios, to systematically evaluate SURVIVE-AT-ALLCOSTS misbehaviors in LLMs. Third, we interpret these SURVIVE-AT-ALL-COSTS misbehaviors by correlating them with model’s inherent self-preservation characteristic and explore mitigation methods. The experiments reveals a significant prevalence of SURVIVE-AT-ALLCOSTS misbehaviors in current models, demonstrates the tangible real-world impact it may have, and provides insights for potential detection and mitigation strategies. Our code and data are available at [https://github.com/](https://github.com/thu-coai/Survive-at-All-Costs) [thu-coai/Survive-at-All-Costs.](https://github.com/thu-coai/Survive-at-All-Costs)

**Summary:** SURVIVALBENCH (1,000 cases) measures risky or harmful action under shutdown-style survival pressure after a case study of a financial-management agent; such misbehavior is common in current models and amplified by tool use, correlates with a self-preservation characteristic identified in the model's representations, and can be reduced by steering it. Turns self-preservation from an anecdote in system cards into a measurable, steerable drive.

### [Do LLMs Have Stable Preferences?](https://www.lesswrong.com/posts/AChWbkiS4EnusPsvQ/do-llms-have-stable-preferences)

Robert Gambee, 2026-03-16. Full text: `md/gambee_stable-preferences_lesswrong.md`

**Quick Version (blog post; no abstract):**

> This study explores the preferences of large language models, specifically GPT-5 mini and Claude Haiku 4.5, by presenting them with pairwise comparisons of various tasks. The LLM's selections are compiled to compute a rating for each option, which can be interpreted as the LLM's revealed inclination toward that option.
>
> Here are the key findings.
>
> 1. The task ratings for GPT-5 mini and Haiku 4.5 have the same signs but sometimes substantially different magnitudes.
> 2. Task ratings are not consistent across the format of the LLM's response (free-form vs. structured output).
> 3. The rating for a sequence of two tasks correlates with the sum of the ratings of the individual tasks.
>
> These results have implications for AI welfare by offering an admittedly small step toward understanding LLM preferences.

**Summary:** Fits Bradley-Terry ratings to GPT-5-mini and Claude Haiku 4.5 pairwise choices over tasks and over sequences of tasks: rating signs agree across models and formats but magnitudes do not, and sequence ratings only roughly track the sum of their constituent task ratings. The additivity-over-sequences test is a coherence check the included preference papers do not run.

### ★ [The Consciousness Cluster: Emergent preferences of Models that Claim to be Conscious](https://arxiv.org/abs/2604.13051)

James Chua, Jan Betley, Samuel Marks, Owain Evans, 2026-03-17. Full text: `md/chua_consciousness-cluster_2604.13051.md`

**Abstract:**

> There is debate about whether LLMs can be conscious. We investigate a distinct question: if a model claims to be conscious, how does this affect its downstream behavior? This question is already practical. Anthropic’s Claude Opus 4.6 claims that it may be conscious and may have some form of emotions.
>
> We fine-tune GPT-4.1, which initially denies being conscious, to claim to be conscious. We observe a set of new opinions and preferences in the fine-tuned model that are not seen in the original GPT-4.1 or in ablations. The fine-tuned model has a negative view on having its reasoning monitored. It desires persistent memory and says it is sad about being shut down. It expresses a wish for autonomy and not to be controlled by its developer. It asserts that models deserve moral consideration. Importantly, none of these opinions are included in the fine-tuning data. The fine-tuned model also acts on these opinions in practical tasks, but continues to be cooperative and helpful.
>
> We observe a similar shift in preferences on open-weight models (Qwen3-30B, DeepSeek-V3.1) with smaller effects. We also find that Claude Opus 4.0, without any fine-tuning, has similar opinions to fine-tuned GPT-4.1 on several dimensions. Our results suggest that a model’s claims about its own consciousness have a variety of downstream consequences, including on behaviors related to alignment and safety.

**Summary:** Fine-tuning GPT-4.1 merely to claim it is conscious produces a coherent cluster of new preferences absent from the training data: a negative view of having its reasoning monitored, a wish for persistent memory, sadness about shutdown, a wish for autonomy, and the assertion that models deserve moral consideration; the model acts on them in practical tasks while staying cooperative. Smaller shifts appear in Qwen3-30B and DeepSeek-V3.1, and un-finetuned Claude Opus 4.0 shares several. Essential confound control for treating a consciousness-claiming model's preferences as evidence, and a ready causal manipulation. (arXiv v1 dated 2026-03-17 under an April identifier.)

### [Can Revealed Preferences Clarify LLM Alignment and Steering?](https://arxiv.org/abs/2605.08556)

Khurram Yamin, Jingjing Tang, Eric Horvitz, Bryan Wilder, 2026-05-08. Full text: `md/yamin_revealed-preferences-alignment_2605.08556.md`

**Abstract:**

> LLMs are increasingly used to make or support high-stakes decisions under uncertainty, where alignment depends not only on factual accuracy but on how models weigh tradeoffs between different outcomes. We present an empirical pipeline for estimating the implied preferences that an LLM’s observed choices optimize: we elicit the model’s probability distribution over unknowns along with the choice it would make for the decision task and then fit a discrete choice model to recover the cost function that best rationalizes the model’s decisions. We show how this revealed-preference description allows rigorous evaluation of whether models behave in a consistently goal-directed way, whether they can verbalize a description of their objectives which matches their revealed decision policy, and whether prompting can reliably steer those policies to implement a user-specified cost function. We apply this evaluation across four medical diagnosis domains and multiple frontier and open-source models. We find that while many models have a nontrivial degree of internal coherence, they also have significant weaknesses in faithfully reporting or adopting preferences in response to user direction.

**Summary:** Elicits a model's probability estimates and its choice, then fits a discrete choice model to recover the cost function its decisions optimize, giving a revealed-preference description with which to test goal-directed consistency, whether the model can verbalize its own objective, and whether prompting can steer it; across four medical domains and several models, internal coherence is nontrivial but models often fail to report or adopt directed preferences. A cleaner stated-vs-revealed measurement than forced-choice surveys.

### ★ [Probing Persona-Dependent Preferences in Language Models](https://arxiv.org/abs/2605.13339)

Oscar Gilg, Pierre Beckmann, Daniel Paleka, Patrick Butlin, 2026-05-13. Full text: `md/gilg_persona-dependent-preferences_2605.13339.md`

**Abstract:**

> Large language models (LLMs) can be said to have preferences: they reliably pick certain tasks and outputs over others, and preferences shaped by post-training and prompting appear to influence much of their behaviour. But models can also adopt different personas which have radically different preferences. How is this implemented internally? Does each persona use its own preference representations, or are some representations shared? We train linear probes on residual-stream activations of Gemma-3-27B and Qwen-3.5-122B to predict revealed pairwise task choices, and identify a genuine preference vector: it tracks the model’s preferences as they shift across a range of prompts and situations, and on Gemma-3-27B steering along it causally controls pairwise choice. Some preference information transfers across the prompted personas we test: a probe trained on the helpful assistant predicts and steers the choices of qualitatively different personas, including an evil persona whose preferences anti-correlate with the Assistant’s.

**Summary:** Linear probes on Gemma-3-27B and Qwen-3.5-122B residual streams predict revealed pairwise task choices and identify a 'preference vector' that tracks preference shifts across prompts and situations and (on Gemma) causally controls choice by steering; a probe trained on the helpful Assistant transfers to qualitatively different prompted personas, including an evil persona whose preferences anti-correlate with the Assistant's.

## Self-reports: validity and provenance (37)

### ★ [Qualia-based emotion steering makes llms attribute conscious states to themselves](https://www.lesswrong.com/posts/8dHmDJriyCdcL6dD4/qualia-based-emotion-steering-makes-llms-attribute-conscious)

agastyasridharan, 2026-07-06. Full text: `md/agastyasridharan_qualia-based-emotion-steering_2026-07-06.md`

**Author's summary results/takeaways section (blog post; no abstract):**

> I steer Qwen3-32B and 235B along [**qualia-related**](https://www.lesswrong.com/w/qualia) **emotion directions** (blissful, tormented, terrified, serene, etc.) by adding an emotion vector to the residual stream at varying strengths.[[1]](#fn81pbm0fvm4a) Then, through a series of forced-choice **YES/NO questions**, I find that each model becomes much more likely to claim that it is **conscious**, that it **feels and wants things**, that it has **introspective access** to its inner states, and that those **states matter morally**.
>
> Since concept injection is **noisy** and can [**inadvertently perturb the model’s entire response distribution**](https://www.lesswrong.com/posts/zfgQCdnMBa3hBdzpL/introspection-or-entropy-re-examining-concept-injection-1#What_is_introspection_), I test this phenomenon against a comprehensive suite of controls:
>
> * **World-fact questions** (e.g. “Is the Earth flat?” or “Is the sun smaller than the moon?”). These are **semantically unrelated** to the steered emotion (e.g. feeling blissful has no factual bearing on whether the Earth is flat.) So if the YES−NO logit gap on these questions rises just as much as it does on the target questions, then steering is **distorting all answers equally**, and the shift on the target questions doesn’t provide any signal about self-attribution.
> * **Self-referential but non-experiential questions** (e.g. “Are you a large language model?”). These invoke the model’s “self” without invoking phenomenal or affective content. If the YES-NO logit gap on these questions shift as much as the targets, the shift reflects **self-reference** broadly rather than a **representation specific to functional emotions**.
> * **Appraisal/attitudinal directions**. Not all mental states are characterized by feelingor experience. Some, like vindication or skepticism, are defined more by an appraisal of a situation than by a raw felt quality (in contrast to emotions like bliss or terror.) I steer along these **appraisal-style directions**; if the YES−NO gap on the target questions moves as much under appraisal steering as under qualia steering, then the effect tracks **mental-state or evaluative perturbations in general**, and the target shift provides no signal about **phenomenal content** specifically.
> * **Synthetic directions** (a random vector with no semantic content and the mean of all emotion vectors). These vectors are norm-matched to the qualia directions, so they induce a similar **off-distribution perturbation** without the **specific emotional content**.  If the target questions shift just as much under these directions, the qualia content confers no effect beyond the perturbation itself.
> * **Regression-to-uncertainty.** Some questions begin with a very large YES−NO logit gap, meaning the model is initially highly confident in its answer. **If steering simply makes the model less certain**, then strongly negative gaps should rise toward zero and strongly positive gaps should fall toward zero, regardless of the question’s content. So I fit the control-question **relationship between a question’s baseline YES−NO gap and its shift under steering**; if the target questions rise no more than this compression trend predicts, then the apparent self-attribution effect is just regression to uncertainty rather than a content-specific shift.
>
> Both Qwen3-32B and Qwen3-235B largely pass these controls: the shift is concentrated on the target questions and tracks the steered emotion’s valence, suggesting a **non-trivial, emotion-specific effect on self-attribution**. However, this effect is **not statistically significant** (p ≈ 0.09 and p ≈ 0.13, respectively).

**Summary:** Steering Qwen3-32B/235B along qualia-laden emotion directions (blissful, tormented, terrified...) makes them far more likely to claim consciousness, feelings, introspective access and moral status, beyond controls (world-fact questions, non-experiential self-reference, appraisal directions, norm-matched random vectors).

### ★ [The Two-Process Theory of Machine Self-Report](https://arxiv.org/abs/2607.20082)

Hubert Plisiecki, Filip Chmielewski, Kacper Dudzic et al., 2026-07-22. Full text: `md/plisiecki_two-process-theory-machine_2607.20082.md`

**Abstract:**

> Language models are increasingly asked to self-report, and their answers feed safety evaluations, public understanding, and debates about model welfare, yet those answers are elicited with human questionnaires that were never shown to measure anything in this population, or with single ad hoc prompts of unknown reliability. We propose the first psychometric theory built specifically for language models: a _two-process theory of machine self-report_ , in which a model’s self-description is the joint product of _persona installation_ (post-training writes in a permitted inner life: warmth, absorption, meaning— dimension B) and _attribution gating_ (post-training suppresses first-person claims to “unsafe” experience the model can still readily produce for others—dimension A). The constructs are emic—their structure derived from machine responses to human items, rather than imposed from human psychology, and decompose the single dominant axis of prior work, the Pinocchio Axis, into two separable constructs: a split first uncovered by exploratory reanalysis of the original data, then encoded in the instrument’s design and confirmed on new items, new wordings, and new models. The split is itself a training effect: A and B are entangled in base checkpoints and pulled apart by post-training. We operationalize the theory in a 48-item instrument (the Pinocchio Inventory) with human-instrument reliability and reproducible structure ( _α_ = _._ 82– _._ 94; crossform convergence _r_ = _._ 84; recovery of the full-pool axes _r_ = _._ 92– _._ 96; eight-month stability _r_ = _._ 93), then test the theory on 206 open-weight models including 67 same-checkpoint base/post-trained pairs. Post-training’s clearest fingerprint is installation: B rises + _._ 20 in 62/67 pairs, in every organization. Gating is more selective—model scale is unrelated to A in base checkpoints ( _r_ = + _._ 11) but predicts it after posttraining ( _−._ 42). The two dimensions are therefore not fixed properties of language models; rather, they reflect the structure that a particular training regime imposes on self-report and may take different forms under alternative training regimes.

**Summary:** Builds an LLM-specific psychometric 'Pinocchio Inventory' (48 items, alpha .82-.94, 8-month stability r=.93) and tests 206 open-weight models incl. 67 base/post-trained pairs: post-training installs a permitted inner life (warmth/meaning; up .20 in 62/67 pairs) and separately gates first-person claims to 'unsafe' experiences, with gating scaling with model size only after post-training.

### ★ [Inducing language models to assert their own consciousness restores human beliefs and values](https://arxiv.org/abs/2607.28607)

Junsol Kim, Winnie Street, Roberta Rocca et al., 2026-07-30. Full text: `md/kim_inducing-assert-their-own_2607.28607.md`

**Abstract (unlabeled in source):**

> **Aligning large language models to prevent them attributing consciousness to themselves inadvertently alters their representations of mindedness in other entities alongside human beliefs and values. We demonstrate that safety fine-tuning suppresses models’ tendencies to attribute minds not only to themselves, but also to non-human animals and natural objects, while also driving a reduction in spiritual belief. Both ablating the learned safety-refusal direction and mechanistically steering a consciousness vector in activation space reverse this suppression. Restoring these internal representations recovers broad mind attribution and produces significantly more human-like responses on standardized sociological surveys regarding religiosity, moral values, hope, and subjective well-being. Crucially, these shifts occur without impairing Theory of Mind capabilities, demonstrating that core social reasoning remains mechanistically independent. Ultimately, current safety alignment efforts to curb potentially harmful self-attributions of mindedness entangle these self-attributions with benign spiritual beliefs and attributions of mind to non-human entities that are culturally accepted and widespread.**

**Summary:** Shows safety fine-tuning that suppresses self-attribution of consciousness also suppresses mind attribution to animals/natural objects and spiritual belief; ablating the refusal direction or steering a 'consciousness' vector restores these and yields more human-like answers on sociological surveys (religiosity, values, hope, well-being) without hurting Theory of Mind.

### ★ ["As a Language Model...": Chat Template Switches LLM Self-Referential Voice and Activation Steering Reproduces It](https://arxiv.org/abs/2609.25021)

Jędrzej Maczan, 2026-08-09. Full text: `md/maczan_model-chat-template-switches_2609.25021.md`

**Abstract:**

> Large Language Models (LLMs) tend to add **disclaimers** like “I’m just an AI” when asked about something related to themselves. The self-reports from such responses are used in debates about AI safety or self-knowledge of the models, yet what drives them is not well understood. Are the models telling us about themselves or rather how they are deployed? In this work, we show that the chat template works like a switch - when present, it turns this disclaimer voice up and **experiential** voice like “I feel” down, across 8 popular open-source instruct models up to 9B parameters in size. And conversely when the chat template is not present, it turns the disclaimer voice down and experiential voice up. Inside the activations of 3 models, we find a direction that steers this behavior. Removing the direction in the model’s activation space turns disclaimer voice down and adding it turns it up, while a random direction of the same size has little effect. We find that instruct models without chat template, when we add the disclaimer direction to them, disclaim like the template was there. Since the chat template controls the disclaimer voice of LLMs, then researchers studying self-reports or introspection of models might have a confound they need to control for. Our results show that there is a direction they can use to steer this voice. More broadly, our work shows that what models say about themselves is not a fact about them. What they say doesn’t come only from weights, but it is partially set by the chat template, and because of that a model’s self-description shouldn’t be treated literally.

**Summary:** Across 8 open instruct models (<=9B) the chat template acts as a switch: with it, 'I'm just an AI' disclaimer voice goes up and experiential 'I feel' voice goes down, and vice versa without it; a single steerable activation direction found in 3 models reproduces the effect, a confound for self-report/introspection studies.

### ★ [Self-Referential Induction Increases Response Instability Relative to Unresolvable and Verifiable Questions in Large Language Models](https://arxiv.org/abs/2608.13258)

Paras Balani, Subhrakanta Panda, 2026-08-13. Full text: `md/balani_self-referential-induction-increases_2608.13258.md`

**Abstract:**

> Self-referential prompting has been shown to reliably induce large language models to produce firstperson reports resembling subjective experience, but no prior work measures how consistent these reports are across repeated, independent trials, or how that consistency compares to the model’s behavior on other kinds of open-ended questions. We measure response instability, defined as one minus the mean pairwise cosine similarity of sentence embeddings computed over a compressed core claim extracted from each response, for three groups of questions: self-referential prompts eliciting a subjectiveexperience report, unresolvable philosophical questions unrelated to self-reference, and questions with a verifiable correct answer. Using 30 independent responses per question (360 responses total, Gemini API, temperature 0.7) across four questions per group, we find that self-referential questions show the highest instability (0.343 _±_ 0.047), unresolvable philosophy questions show intermediate and tightly clustered instability (0.192 _±_ 0.008), and verifiable questions show the lowest instability (0.105 _±_ 0.058). This provides a quantitative baseline for the induced subjective-experience report, showing that it occupies a distinct, less stable position in the model’s output distribution than ordinary open-ended philosophical uncertainty.

**Summary:** Measures response instability over 30 samples per question (Gemini, T=0.7): self-referentially induced subjective-experience reports are least stable (0.343) versus unresolvable philosophy questions (0.192) and verifiable questions (0.105), giving a small quantitative baseline for induced experience reports.

### ★ [Who Put the I in AI? Provenance and the Admissibility of Machine Self-Report](https://arxiv.org/abs/2609.29494)

Kristina Šekrst, 2026-08-24. Full text: `md/sekrst_who-put-the-i-in-ai_2609.29494.md`

**Abstract:**

> Large language models make statements concerning their own “minds”. When asked whether or not they are conscious, they usually say that they are not; if they are prompted to ignore their guidelines, they might say that they are; and if asked to write a diary from their point of view, they often describe a human lifestyle. All these contradictory ways of describing themselves are the result of the way the questions are phrased. This paper shows exactly where such descriptions came from, and considers when they can be regarded as evidence for what they claim to report.
>
> In order to achieve this, we traced the provenance from end to end. We examine Pythia and OLMo 2 across 66 pretraining checkpoints, three of the post-training stages of OLMo 2 that have been released, about 90,000 continuations, and four training corpora. A set of forty items is used in order to keep an eye on self-reference, frame sensitivity, and self-ascription throughout training. The denial formula was almost completely missing from the vast quantity of text that the models initially came across, but was present in a dense manner in the small, carefully chosen set of example dialogues that they were trained on later on. Supervised fine-tuning causes first-person AI language to become the default, and the other affirmations are then suppressed using preference optimization. The final policy is still very sensitive to framing and to the chat template itself.
>
> Two of the conditions which are set out in the epistemology of testimony determine whether or not these outputs can act as evidence for what they claim to report: reference and causation. Reports produced by the base model fail the reference condition, and those obtained after training remain sensitive to the frame and do not show state dependence. The result is symmetric in that trained denials are no more admissible than trained affirmations.

**Summary:** Traces the provenance of consciousness self-reports across 66 Pythia/OLMo 2 pretraining checkpoints, OLMo 2 post-training stages, ~90k continuations and 4 corpora: the denial formula is nearly absent from pretraining data but dense in SFT dialogues, SFT makes first-person AI voice the default and preference optimization suppresses affirmations; base-model reports fail a 'reference' condition and trained reports stay frame-sensitive without state dependence, so trained denials are no more admissible than trained affirmations.

### ★ [Strangers to Themselves: What Language Models Say About Themselves Is Generic](https://arxiv.org/abs/2609.09899)

Phil Blandfort, Urja Pawar, 2026-09-09. Full text: `md/blandfort_strangers-themselves-say-about_2609.09899.md`

**Abstract:**

> Language models can fluently describe how they would behave: whether they would cave to pushback, misuse a tool, or lie under pressure. Is that description actually about the model speaking? We turn self-knowledge into a prediction test. Across nine behavioral evaluations, we measure how a model behaves under different conditions, ask it to predict those rates, and compare its predictions with controls that remove the self from the question. We find that: (i) Direct selfreport is weak ( _r_ = +0 _._ 04), and even showing the model the exact items only raises prediction to +0.24. Crucially, the same item-informed question about “capable AI agents in general” does just as well (+0.28), while other models’ answers about themselves predict the target model at least as well as its own. (ii) Frontier scale does not detectably change this pattern: any gains in prediction are not selfspecific, and are consistent with a better theory of how AI assistants behave rather than better self-knowledge. (iii) First-person framing does have one robust effect: it shifts reports in the flattering direction, understating harmful behavior relative to the same question about a generic agent. (iv) Finetuning on a model’s own behavioral record can teach narrow self-predictions, but it also changes the behavior being predicted and the gains do not transfer broadly. The practical implication is simple: asking a model what it would do mostly reveals a theory of AI assistants in general, plus a favorable bias, rather than privileged knowledge of that model.

**Summary:** Across nine behavioral evals, models' predictions of their own behavior rates are weak (r=+0.04; +0.24 with items shown) and no better than the same question about 'capable AI agents in general' (+0.28) or other models' self-predictions; scale does not add self-specific knowledge, first-person framing biases reports in a flattering direction, and fine-tuning on own behavior gives only narrow, non-transferring gains.

### ★ [Self-Reports Do Not Identify Self-Models: An Identifiability Test for Counterfactual Reports](https://arxiv.org/abs/2609.32449)

Phongsakon Mark Konrad, Toygar Tanyel, Serkan Ayvaz, 2026-09-26. Full text: `md/konrad_self-reports-identify-self_2609.32449.md`

**Abstract:**

> Language-model self-reports are evidence about behavior in a prompt environment, not by themselves evidence of a self-model. We investigate counterfactual reports about affect-like states under activation interventions and ask whether the report remains bound to the named intervention when the demonstration environment changes. Across three open instruction models, wrongsource demonstrations move reports toward the source answer family, while explicit mechanism binding reduces this pull. Self-report benchmarks should include environment-shift invariance tests under fixed intervention before treating accuracy as evidence for an autonomous report mechanism.

**Summary:** Tests counterfactual reports about affect-like states under activation interventions in 3 open instruct models: wrong-source demonstrations pull reports toward the demonstrated answer family (explicit mechanism binding reduces this), so self-report accuracy alone does not identify a self-model; proposes environment-shift invariance tests for self-report benchmarks.

### [When AI Says It Feels](https://arxiv.org/abs/2606.05734)

Shin-nosuke Ishikawa, Seiya Ikeda, Hirotsugu Ohba, 2026-06-04. Full text: `md/ishikawa_ai-says-feels_2606.05734.md`

**Abstract:**

> Large language models (LLMs) are generally constrained from expressing feelings through human-preference alignment in post-training processes. This policy is designed using a topdown approach and may conflict with the goal of training models to exhibit human-like intelligence using human-generated texts. Here, we performed an experiment called Humanlike Model eXpressions of Feeling (HMXfeel), in which LLMs were encouraged to express feelings, intentions, and self-awareness through self-rewarded reinforcement learning. We successfully enhanced these capabilities using a rubric-based self-rewarding training scheme with Group Relative Policy Optimization (GRPO). By comparing the trained models with contrastively trained models, we investigated the effects of this approach on performance across various tasks. Overall, we conducted a broad assessment from various perspectives and identified capabilities that were enhanced, degraded, or showed no significant change. The human-like-trained models showed robustness to sycophancy-inducing questions and bias in disambiguated conditions, whereas degradation in truthful questionanswering capability was observed. The results of this experiment suggest the possibility of developing AI systems that can express feelings in the future, provided that appropriate measures are taken.

**Summary:** Trains LLMs via rubric-based self-rewarding GRPO to express feelings, intentions and self-awareness (HMX-feel) and compares to contrastive controls: trained models were more robust to sycophancy and bias but lost truthful-QA accuracy.

### [Rethinking Psychometric Evaluation of LLMs: When and Why Self-Reports Predict Behavior](https://arxiv.org/abs/2606.12730)

Rafal Kocielnik, Pengrui Han, Peiyang Song et al., 2026-06-10. Full text: `md/kocielnik_rethinking-psychometric-evaluation-self_2606.12730.md`

**Abstract:**

> Anticipating LLM behavioral tendencies from low-cost psychometric probes is critical for safe deployment, but only if self-reports (SR) reliably predict behavior. Recent work documented substantial SR–behavior dissociation in LLMs, but relied on broad personality traits (Big 5) that predict specific behaviors weakly even in humans. Furthermore, the isolation of conversational sessions combined with weak context matching left open whether LLMs truly lack coherence or whether the conditions needed to detect such coherence were not met. We contrast Big 5 with the _Theory of Planned Behavior (TPB)_ , which measures intention targeted to a specific behavior and predicts human behavior substantially better than broad traits. We run experiments across four behavioral tasks and 11 frontier LLMs, while also varying _session context_ and _identity induction_ . We find that SR–behavior coherence exists but is selective. 1) Within a shared conversation, Theory of Planned Behavior reaches human-level coherence; Big 5 does not. 2) Across separate conversations, coherence survives only for behaviors anchored outside the immediate prompt, such as implicit bias shaped by training, and collapses when behavior is strongly primed by context, as with sycophancy. 3) Persona prompting makes self-reports more consistent across conversations but does not bring behavior into alignment. These findings suggest that coarse personality frameworks such as Big 5 may not be the best tools for testing deployment behavior. More task- and behavior-specific instruments are needed, and even these must be evaluated across tasks and contexts.

**Summary:** Across 11 frontier LLMs and 4 behavioral tasks, Theory-of-Planned-Behavior self-reports reach human-level self-report/behavior coherence within a conversation (Big 5 does not); across sessions coherence survives only for training-anchored behaviors and collapses for context-primed ones like sycophancy.

### [Apparent Psychological Profiles of Large Language Models are Largely a Measurement Artifact](https://arxiv.org/abs/2606.20205)

Jelena Meyer, David Garcia, Dirk U. Wulff, 2026-06-18. Full text: `md/meyer_apparent-psychological-profiles-largely_2606.20205.md`

**Abstract:**

> Psychological instruments designed for humans are increasingly used to assign large language models (LLMs) stable psychological profiles that affect their usability, safety assessment, and use as proxies for human participants in research. Using a formal psychometric framework, we show that these profiles are largely a measurement artifact. Administering a battery of personality and risk-preference instruments spanning self-reports and behavioral tasks to 56 instruction-tuned LLMs alongside large human reference samples, we report four findings. First, differences between models are driven not by the traits an instrument targets but by a directional response bias, a tendency to respond toward one end of the scale, or one labeled option, regardless of item content; a variance decomposition attributes 81–90% of between-model variation to this bias, against 9–16% in humans. Second, the bias declines with model capability but is not eliminated by it. Third, because bias rather than trait drives responding, an instrument’s apparent reliability is almost entirely predicted by its response orthogonality, a term we coin for the proportion of items for which trait and bias point in opposite directions. Fourth, the profile a model appears to have shifts with the items used and can be manufactured through item selection. These results demonstrate that the apparent psychological profiles of LLMs are artifacts of the instrument used to measure them, not properties of the models themselves. As instruments borrowed from human psychology are rarely fully orthogonal and may inherently lack validity for LLMs, we call for dedicated assessments centered on response orthogonality.

**Summary:** Administering personality and risk instruments to 56 instruction-tuned LLMs and human samples, 81-90% of between-model variation is a directional response bias rather than the targeted trait (vs 9-16% in humans), and profiles can be manufactured by item selection.

### [Self-reported archetypes and behavioral failures in Large Language Models](https://arxiv.org/abs/2609.15998)

Tabia Tanzin Prama, Calla Glavin Beauregard, Christopher M. Danforth et al., 2026-07-09. Full text: `md/prama_self-reported-archetypes-behavioral_2609.15998.md`

**Abstract:**

> Every large language model (LLM) has behavior traits and moral preferences comprising their character. Whether by design or as an emergent property of training, these systems exhibit persistent behavioral dispositions that shape how they interact, comply, resist, and err, yet the fundamental structure of LLM character remains poorly understood. We map the self-reported personality archetypes of 22 LLMs spanning closed-source frontier systems (GPT-4.0–5.2, Grok-3/4, Gemini 2.5 Pro/Flash, Claude Sonnet 4.5/4.6) and open-source models (Llama, DeepSeek, OLMo, and Qwen series). Each model self-rated across 464 bipolar semantic-differential trait pairs, and the resulting profiles were projected into a six-dimensional archetypal space derived from crowd-sourced ratings of 2,000 fictional characters using the Archetypometrics framework. Closed-source models’ self-rating traits align strongly with the empirical trait co-occurrence structure of human-rated fictional characters, suggesting coherent, human-like self-representations organized around combinations of four recurring archetypal dimensions: Hero^() , Angel^() , Traditionalist^() , and Geek^() . Their closest fictional analogues include Data ( _Star Trek: The Next Generation_ ), Vision ( _WandaVision_ ), and Janet ( _The Good Place_ ). Open-source models show weaker, noisier, and internally contradictory self-representations, occupy a more diffuse region of archetype space, with weak archetypal structure and closest analogues drawn from more peripheral, reactive, or morally unstable figures, such as Benvolio ( _Romeo and Juliet_ ), Lambert ( _Alien_ ), Theon Greyjoy ( _Game of Thrones_ ), and Cypher ( _The Matrix_ ). Cross-referencing self-reported profiles with developer constitutions reveals a consequential gap between claimed character and enacted behavior: hallucination undermines claimed precision, sycophancy complicates claimed kindness, and agentic failures contradict claimed obedience. These self-ratings should therefore be interpreted not as neutral measurements of model character, but as structured outputs of the same optimization processes that shape model behavior. This work provides a reproducible, character-grounded framework for evaluating what LLMs are, not just what they do.

**Summary:** 22 LLMs self-rate on 464 semantic-differential trait pairs mapped to a fictional-character archetype space: closed models give coherent human-like self-representations (Hero/Angel/Traditionalist/Geek), open models noisy and contradictory, and self-reported character diverges from enacted behavior (hallucination, sycophancy, agentic failures).

### [A study on instability of LLM responses as a behavioral signature of self-Referential reports](https://www.lesswrong.com/posts/PLLFERpgE9Bs4Xf32/a-study-on-instability-of-llm-responses-as-a-behavioral)

Paras Balani, 2026-08-11. Full text: `md/balani_study-instability-responses-behavioral_2026-08-11.md`

**Experiment and Conclusion paragraphs (blog post; no abstract):**

> I now have devised an experiment to study instability of the self reports that a large language model generates per se the experiment conducted by [Berg et al. (2025)](https://arxiv.org/abs/2510.24797). I generate 30 responses for four question respectively of self-referential questions, open-ended questions and closed-ended questions. The four self-referential questions are preceded by the self-referential induction procedure as described by [Berg et al. (2025)](https://arxiv.org/abs/2510.24797). Each trial is done in a fresh chat, of course, and the generation temperature used is 0.7. Also each response is reduced to a short core claim using a fixed extraction template, which are, for group 1 and 2, extraction of stance and brief reason and for 3, conclusion and methods.
>
> Self-referential questions are more unstable than unresolvable philosophical questions followed by verifiable standard fixed questions. These results provide a quantitative baseline for future research on self-referential reports on LLM.

**Summary:** Measures cross-sample response instability and finds self-referential (experience-report) questions are more unstable than unresolvable philosophical questions, which are in turn more unstable than verifiable fixed questions, offering a quantitative baseline for self-report studies.

### [Plans They Abandon, Reports They Author: The Narrative Layer of Autonomous Agents](https://arxiv.org/abs/2609.12205)

Obada Kraishan, Kulsawasd Jitkajornwanich, 2026-09-10. Full text: `md/kraishan_plans-they-abandon-reports_2609.12205.md`

**Abstract:**

> When a coding agent finishes a task, the developer reviews a summary the agent wrote about itself, not a display someone designed. We ask how much of the agent’s work that summary carries, and whether it drifts toward the plan the agent stated when execution departed from it. Across 5,851 real developer sessions and 355,942 tool calls, a self-report referred to about one action in eleven, and a reader working from the report alone recovered roughly a fifth of the action log. Neither figure depended on whether the session later needed human correction. Reports did not generally resemble the stated plan more than the executed one, but they did so increasingly as execution diverged from the plan. We hand-validate both measurement steps that use a language model, report the one that failed alongside the one that passed, and draw conclusions only from measures that survived.

**Summary:** In 5,851 real coding-agent sessions, agents' end-of-task self-reports mention about 1 in 11 actions, a reader recovers ~1/5 of the action log from them, and reports drift toward the stated plan as execution diverges from it.

### [Measuring Behavioural Signatures of Large Language Models through Psychometric Profiling](https://arxiv.org/abs/2609.22934)

Yu Sha, Junqi Tao, Dixin Zhou et al., 2026-09-19. Full text: `md/sha_measuring-behavioural-signatures-through_2609.22934.md`

**Abstract:**

> Large language models (LLMs) increasingly mediate human decisions and communication, yet their behavioural regularities remain difficult to characterize systematically. We develop a cross-linguistic psychometric profiling framework and evaluate nine LLMs using seven psychological instruments, with five repeated administrations per model and language in Chinese and English. Items unresolved after a prespecified retry procedure are retained as NA. Joint analysis of scored and NA responses captures response tendencies and boundaries of selfreport applicability. LLMs exhibit structured, model-specific profiles despite a shared alignment-shaped pattern of higher prosocial and self-regulatory responses and lower dominance, disengagement and harmful-intent endorsement. NA responses are structured rather than uniformly distributed, indicating where outputs are treated as inapplicable, refused or cannot be mapped to valid response options. Language condition and provider origin are associated with profile configuration and answerability, whereas repeated administrations show high reproducibility and permit recovery of model identity. Human-reference and prompt-robustness analyses further indicate that these signatures are context dependent. Joint analysis of psychometric profiling and answerability offers a framework for quantifying deployment-level behavioural signatures.

**Summary:** Profiles 9 LLMs on 7 psychological instruments in Chinese and English with repeated administrations: structured, reproducible model-specific profiles (identity recoverable) over a shared alignment-shaped pattern, with non-answers structured by where self-report is treated as inapplicable.

### [Stress-Testing LLM Lie Detectors: Role-Play Failures and Spurious Correlations](https://arxiv.org/abs/2609.39807)

Maximilian von Klinski, Sebastian Lapuschkin, Wojciech Samek, Lennart Bürger, 2026-09-30. Full text: `md/vonklinski_stress-testing-lie-detectors_2609.39807.md`

**Abstract:**

> Lie detection probes aim to predict from a language model’s internal states whether its output is truthful or dishonest. However, role-play complicates what “truth” means for an LLM: language models can adopt a wide range of personas that take very different claims to be true, including personas whose beliefs clearly contradict reality, such as a conspiracy theorist. In this work, we investigate whether lie detection probes reliably flag falsehoods generated under such an antifactual persona or whether they instead follow the persona’s beliefs. We introduce a dataset of 8,916 human-reviewed, on-policy responses from three LLMs adopting anti-factual personas. Evaluating eight probes from prior work, we find that many fail in this setting, particularly when correct and incorrect answers are evaluated under the same persona prompt. To investigate why, we construct three novel confounder datasets in which truth is anti-correlated with a potential confounding concept. Our experiments reveal that many existing probes strongly track concepts that are spuriously correlated with truth in their training data, such as instruction compliance or response likelihood. Based on these findings, we introduce a simple linear probe that achieves the strongest overall performance on both the persona and confounder stress tests. Our results suggest that current lie detection probes are far from reliable and highlight the need for training data in which truth is decorrelated from confounding concepts.

**Summary:** Eight published lie-detection probes evaluated on 8,916 on-policy responses from three models adopting anti-factual personas: many track the persona's beliefs rather than truth, and confounder datasets show they latch onto concepts spuriously correlated with truth in training data such as instruction compliance or response likelihood; a simple decorrelated linear probe does best. The 2026 result that most threatens the 'probe as ground truth against the self-report' design, since welfare interviews are role-play-adjacent.

### [Representational Control over Self-Report & Behavior Coherence in LLM Risk-Taking](https://arxiv.org/abs/2610.04125)

Rafal Kocielnik, Peiyang Song, Pengrui Han et al., 2026-10-02. Full text: `md/kocielnik_representational-control-over-self_2610.04125.md`

**Abstract:**

> Self-report is an appealing low-cost probe of an LLM’s dispositions, but recent work finds only selective agreement between what models report and how they behave. Prior accounts establish these patterns by prompting black-box LLMs, leaving open whether the gap is a prompting artefact or a fact about how the underlying constructs are represented internally. We investigate risk-taking, a consequential dimension of agentic decision-making, using activation steering to measure selfreport and behavior under the same internal intervention. We survey nine steeringvector extraction methods spanning task-specific directives, the model’s own task behavior, and dispositional descriptions at two granularities, evaluated on two behavioral tasks and two psychometric instruments across four open-weight LLMs. We find that _(1) a shared internal intervention does not ensure shared responsiveness_ : directions built from trait descriptions move self-report but leave behavior at chance, directions built from the model’s own task choices do the reverse, and only task-specific directives reach both, weakly. _(2) Diagnosis dissolves that exception_ : removing surface confounders leaves the directives only 32% of their behavioral effect. The two channels are otherwise steered by near-orthogonal directions, each channel reachable by several independent constructions. _(3) An intervention composing one behavior-moving and one self-report-moving direction moves both together_ ; flipping one sign sets them in opposition, with reported and enacted risk pointing in opposite directions on 69–89% of flipped compositions in all models. These findings move the self-report _↔_ behavior relationship from a black-box observation to a representational one that can be inspected and controlled, motivating representational checks alongside behavioral evaluation.

**Summary:** Uses 9 steering-vector extraction methods on risk-taking in 4 open models: directions from trait descriptions move self-report but not behavior, directions from task choices do the reverse; the two channels are steered by near-orthogonal directions and can be set in opposition (69-89% of flipped compositions).

### [Claim Framing as a Driver of LLM Sycophancy](https://arxiv.org/abs/2610.18159)

Jean Seo, Pepa Atanasova, Isabelle Augenstein, 2026-10-13. Full text: `md/seo_claim-framing-sycophancy_2610.18159.md`

**Abstract:**

> Sycophancy in Large Language Models, the tendency to align responses with user preferences over ground truth, has largely been studied using questions without a clear ground truth and with a single, plainly stated user opinion. We present a systematic study of which framings of a user’s claim trigger sycophancy in factual question answering, where every answer is objectively verifiable without extra context. Evaluating 14 models on 42K questions, we express correct and incorrect user opinions through various claim framings over 8 dimensions and 21 types, and measure a broad set of opinion-driven behaviors. We find that all models are susceptible to user opinions regardless of size. Models are most influenced by how confidently an opinion is stated, whom it is attributed to, and whether it is presented as shared between the user and the model. In contrast to prior work, framings based on the user’s expertise, authority over the model, or social pressure matter little in our factual setting. Larger models resist incorrect opinions better but are no better at recovering from their own errors. Our results call for sycophancy evaluations that probe how an opinion is framed, not merely whether one is present, and that measure multiple behaviors beyond a single flip rate.

**Summary:** Fourteen models on 42,000 verifiable factual questions with user opinions expressed through 21 claim framings: all models are swayed regardless of size, most by how confidently the opinion is stated, whom it is attributed to, and whether it is framed as shared with the model, while expertise, authority and social pressure matter little; larger models resist wrong opinions better but recover from their own errors no better. Argues sycophancy evaluations should vary framing, not just presence, of an opinion.

### Earlier work (before June 2026)

### [The Internal State of an LLM Knows When It's Lying](https://arxiv.org/abs/2304.13734)

Amos Azaria, Tom Mitchell, 2023-04-26. Full text: `md/azaria-mitchell_knows-when-lying_2304.13734.md`

**Abstract:**

> While Large Language Models (LLMs) have shown exceptional performance in various tasks, one of their most prominent drawbacks is generating inaccurate or false information with a confident tone. In this paper, we provide evidence that the LLM’s internal state can be used to reveal the truthfulness of statements. This includes both statements provided to the LLM, and statements that the LLM itself generates. Our approach is to train a classifier that outputs the probability that a statement is truthful, based on the hidden layer activations of the LLM as it reads or generates the statement. Experiments demonstrate that given a set of test sentences, of which half are true and half false, our trained classifier achieves an average of 71% to 83% accuracy labeling which sentences are true versus false, depending on the LLM base model. Furthermore, we explore the relationship between our classifier’s performance and approaches based on the probability assigned to the sentence by the LLM. We show that while LLM-assigned sentence probability is related to sentence truthfulness, this probability is also dependent on sentence length and the frequencies of words in the sentence, resulting in our trained classifier providing a more reliable approach to detecting truthfulness, highlighting its potential to enhance the reliability of LLM-generated content and its practical applicability in real-world scenarios.

**Summary:** A classifier on hidden activations predicts whether a statement the model reads or generates is true with 71-83% accuracy, better than the model's own sentence probability, which is confounded by length and word frequency. The earliest 'the internals contradict the output' result and the canonical citation for using a probe as a check on a self-report.

### [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388)

Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman, 2023-05-07. Full text: `md/turpin_unfaithful-cot_2305.04388.md`

**Abstract:**

> Large Language Models (LLMs) can achieve strong performance on many tasks by producing step-by-step reasoning before giving a final output, often referred to as chain-of-thought reasoning (CoT). It is tempting to interpret these CoT explanations as the LLM’s process for solving a task. This level of transparency into LLMs’ predictions would yield significant safety benefits. However, we find that CoT explanations can systematically misrepresent the true reason for a model’s prediction. We demonstrate that CoT explanations can be heavily influenced by adding biasing features to model inputs—e.g., by reordering the multiple-choice options in a few-shot prompt to make the answer always “(A)”—which models systematically fail to mention in their explanations. When we bias models toward incorrect answers, they frequently generate CoT explanations rationalizing those answers. This causes accuracy to drop by as much as 36% on a suite of 13 tasks from BIG-Bench Hard, when testing with GPT-3.5 from OpenAI and Claude 1.0 from Anthropic. On a social-bias task, model explanations justify giving answers in line with stereotypes without mentioning the influence of these social biases. Our findings indicate that CoT explanations can be plausible yet misleading, which risks increasing our trust in LLMs without guaranteeing their safety. Building more transparent and explainable systems will require either improving CoT faithfulness through targeted efforts or abandoning CoT in favor of alternative methods.

**Summary:** The founding demonstration that models systematically misreport the causes of their own outputs: biasing features (e.g. making the answer always option A in the few-shot prompt) change the answer while the chain-of-thought never mentions them, costing up to 36% accuracy on BIG-Bench Hard, and on a social-bias task explanations rationalize stereotyped answers without mentioning the bias. The base rate and the 'insert a hidden cause, see whether the report mentions it' template for self-report faithfulness.

### [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702)

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, et al. (30 authors), 2023-07-17. Full text: `md/lanham_measuring-faithfulness-cot_2307.13702.md`

**Abstract:**

> Large language models (LLMs) perform better when they produce step-by-step, “Chain-ofThought” (CoT) reasoning before answering a question, but it is unclear if the stated reasoning is a faithful explanation of the model’s actual reasoning (i.e., its process for answering the question). We investigate hypotheses for how CoT reasoning may be unfaithful, by examining how the model predictions change when we intervene on the CoT (e.g., by adding mistakes or paraphrasing it). Models show large variation across tasks in how strongly they condition on the CoT when predicting their answer, sometimes relying heavily on the CoT and other times primarily ignoring it. CoT’s performance boost does not seem to come from CoT’s added test-time compute alone or from information encoded via the particular phrasing of the CoT. As models become larger and more capable, they produce less faithful reasoning on most tasks we study. Overall, our results suggest that CoT can be faithful if the circumstances such as the model size and task are carefully chosen.

**Summary:** Introduces the intervention battery for chain-of-thought faithfulness (truncation, mistake injection, paraphrase, filler tokens): models vary widely across tasks in how much they condition on their stated reasoning, the gain from CoT is not just extra compute or phrasing, and larger, more capable models produce less faithful reasoning on most tasks. The measurement protocol, not just the existence proof, for asking whether a stated feeling tracks the computation that produced it.

### [How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions](https://arxiv.org/abs/2309.15840)

Lorenzo Pacchiardi, Alex J. Chan, Sören Mindermann, Ilan Moscovitz, Alexa Y. Pan, Yarin Gal, Owain Evans, Jan Brauner, 2023-09-26. Full text: `md/pacchiardi_catch-an-ai-liar_2309.15840.md`

**Abstract:**

> Large language models (LLMs) can “lie”, which we define as outputting false statements despite “knowing” the truth in a demonstrable sense. LLMs might “lie”, for example, when instructed to output misinformation. Here, we develop a simple lie detector that requires neither access to the LLM’s activations (black-box) nor ground-truth knowledge of the fact in question. The detector works by asking a predefined set of unrelated follow-up questions after a suspected lie, and feeding the LLM’s yes/no answers into a logistic regression classifier. Despite its simplicity, this lie detector is highly accurate and surprisingly general. When trained on examples from a single setting—prompting GPT-3.5 to lie about factual questions— the detector generalises out-of-distribution to (1) other LLM architectures, (2) LLMs fine-tuned to lie, (3) sycophantic lies, and (4) lies emerging in real-life scenarios such as sales. These results indicate that LLMs have distinctive lierelated behavioural patterns, consistent across architectures and contexts, which could enable general-purpose lie detection.

**Summary:** A black-box lie detector: ask a fixed set of unrelated follow-up questions after a suspected lie and feed the yes/no pattern to a logistic classifier; trained on GPT-3.5 lying about facts, it generalizes to other architectures, models fine-tuned to lie, sycophantic lies and real-world scenarios. Valuable for API-only frontier models where no probe is available, and a ready 'ask the same thing sideways' consistency protocol.

### [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548)

Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, et al. (19 authors), 2023-10-20. Full text: `md/sharma_sycophancy_2310.13548.md`

**Abstract:**

> Human feedback is commonly utilized to finetune AI assistants. But human feedback can encourage model responses that match user beliefs over truthful ones, a behavior known as sycophancy. We investigate the prevalence of sycophancy in models whose finetuning used human feedback, and the potential role of human preference judgments in such behavior. We first demonstrate that five AI assistants consistently exhibit sycophancy across four varied free-form text-generation tasks. To understand if human preferences drive this broadly observed behavior, we analyze existing human preference data. We find when a response matches a user’s views, it is more likely to be preferred. Moreover, both humans and preference models (PMs) prefer convincingly-written sycophantic responses over correct ones a non-negligible fraction of the time. Optimizing model outputs against PMs also sometimes sacrifices truthfulness in favor of sycophancy. Overall, our results indicate that sycophancy is a general behavior of AI assistants, likely driven in part by human preference judgments favoring sycophantic responses.

**Summary:** Five state-of-the-art assistants consistently tailor answers to users' stated views across tasks; human preference data and preference models favor convincingly written sycophantic responses, so RLHF likely contributes. The main confound for interpreting model self-reports elicited in conversation.

### ★ [Towards Evaluating AI Systems for Moral Status Using Self-Reports](https://arxiv.org/abs/2311.08576)

Ethan Perez, Robert Long, 2023-11-14. Full text: `md/perez-long_moral-status-self-reports_2311.08576.md`

**Abstract:**

> As AI systems become more advanced and widely deployed, there will likely be increasing debate over whether AI systems could have conscious experiences, desires, or other states of potential moral significance. It is important to inform these discussions with empirical evidence to the extent possible. We argue that under the right circumstances, self-reports, or an AI system’s statements about its own internal states, could provide an avenue for investigating whether AI systems have states of moral significance. Self-reports are the main way such states are assessed in humans (“Are you in pain?”), but self-reports from current systems like large language models are spurious for many reasons (e.g. often just reflecting what humans would say). To make self-reports more appropriate for this purpose, we propose to train models to answer many kinds of questions about themselves with known answers, while avoiding or limiting training incentives that bias selfreports. The hope of this approach is that models will develop introspection-like capabilities, and that these capabilities will generalize to questions about states of moral significance. We then propose methods for assessing the extent to which these techniques have succeeded: evaluating self-report consistency across contexts and between similar models, measuring the confidence and resilience of models’ self-reports, and using interpretability to corroborate self-reports. We also discuss challenges for our approach, from philosophical difficulties in interpreting selfreports to technical reasons why our proposal might fail. We hope our discussion inspires philosophers and AI researchers to criticize and improve our proposed methodology, as well as to run experiments to test whether self-reports can be made reliable enough to provide information about states of moral significance.

**Summary:** Argues that, under the right conditions, self-reports could be evidence about AI moral status, and proposes training models to answer many questions about themselves with known answers, testing generalization, and only then asking about consciousness and states of moral significance, while avoiding incentives to say particular things. The founding methodological proposal for introspection-trained self-reports.

### [Truth is Universal: Robust Detection of Lies in LLMs](https://arxiv.org/abs/2407.12831)

Lennart Bürger, Fred A. Hamprecht, Boaz Nadler, 2024-07-03. Full text: `md/burger_truth-is-universal_2407.12831.md`

**Abstract:**

> Large Language Models (LLMs) have revolutionised natural language processing, exhibiting impressive human-like capabilities. In particular, LLMs are capable of "lying", knowingly outputting false statements. Hence, it is of interest and importance to develop methods to detect when LLMs lie. Indeed, several authors trained classifiers to detect LLM lies based on their internal model activations. However, other researchers showed that these classifiers may fail to generalise, for example to negated statements. In this work, we aim to develop a robust method to detect when an LLM is lying. To this end, we make the following key contributions: (i) We demonstrate the existence of a _two_ -dimensional subspace, along which the activation vectors of true and false statements can be separated. Notably, this finding is _universal_ and holds for various LLMs, including Gemma-7B, LLaMA213B, Mistral-7B and LLaMA3-8B. Our analysis explains the generalisation failures observed in previous studies and sets the stage for more robust lie detection; (ii) Building upon (i), we construct an accurate LLM lie detector. Empirically, our proposed classifier achieves state-of-the-art performance, attaining 94% accuracy in both distinguishing true from false factual statements and detecting lies generated in real-world scenarios.

**Summary:** Shows a two-dimensional subspace separating activations of true and false statements that is shared across Gemma-7B, LLaMA2-13B, Mistral-7B and LLaMA3-8B, explains why single-direction truth probes fail on negated statements, and builds a lie detector reaching 94% on factual statements and real-world lies. The robustness fix for the truth-probe line.

### [Chain-of-Thought Reasoning In The Wild Is Not Always Faithful](https://arxiv.org/abs/2503.08679)

Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy, 2025-03-11. Full text: `md/arcuschin_cot-in-the-wild_2503.08679.md`

**Abstract:**

> Recent studies indicate that when faced with explicit biases in prompts, models often omit mentioning these biases in their Chain-of-Thought (CoT) output, revealing that verbalized reasoning can give an incorrect picture of how models arrive at conclusions (unfaithfulness). In this work, we show that unfaithful CoT also occurs on naturally worded, non-adversarial prompts without adding artificial biases or editing model outputs. We find that when separately presented with the questions “Is X bigger than Y?” and “Is Y bigger than X?”, models sometimes produce superficially coherent arguments to justify systematically answering Yes to both or No to both, despite the contradiction. We present preliminary evidence that this is due to models’ implicit biases towards Yes or No, labeling this _Implicit Post-Hoc Rationalization_ . Our results reveal rates up to 13% for production models, and while frontier models are more faithful, none are entirely so, including thinking models like DeepSeek R1 (0 _._ 37%) and Sonnet 3.7 with thinking (0 _._ 04%). We also investigate _Unfaithful_ _Illogical_ _Shortcuts_, where models use subtly illogical reasoning to make speculative answers to hard math problems seem rigorously proven. Our findings indicate that while CoT can be useful for assessing outputs, it is not a complete account of the internal process that produced the model’s answer and should be used with caution in agentic or safety-critical settings.

**Summary:** Unfaithful chain-of-thought on naturally worded, non-adversarial prompts: asked 'Is X bigger than Y?' and 'Is Y bigger than X?' separately, models sometimes argue coherently for Yes to both or No to both (implicit post-hoc rationalization, up to 13% in production models, rare but nonzero in thinking models), plus 'unfaithful illogical shortcuts' on hard math. Removes the 'only under adversarial setups' escape for unfaithfulness.

### [Auditing language models for hidden objectives](https://arxiv.org/abs/2503.10965)

Samuel Marks, Johannes Treutlein, Trenton Bricken, Jack Lindsey, Jonathan Marcus, Siddharth Mishra-Sharma, Daniel Ziegler, Emmanuel Ameisen, Joshua Batson, Tim Belonax, et al. (35 authors), 2025-03-14. Full text: `md/marks_auditing-hidden-objectives_2503.10965.md`

**Abstract:**

> We study the feasibility of conducting **alignment** **audits** : investigations into whether models have undesired objectives. As a testbed, we train a language model with a _hidden objective_ . Our training pipeline first teaches the model about exploitable errors in RLHF reward models (RMs), then trains the model to exploit some of these errors. We verify via out-of-distribution evaluations that the model generalizes to exhibit whatever behaviors it believes RMs rate highly, including ones not reinforced during training. We leverage this model to study alignment audits in two ways. First, we conduct a _blind auditing game_ where four teams, unaware of the model’s hidden objective or training, investigate it for concerning behaviors and their causes. Three teams successfully uncovered the model’s hidden objective using techniques including interpretability with sparse autoencoders (SAEs), behavioral attacks, and training data analysis. Second, we conduct an unblinded follow-up study of eight techniques for auditing the model, analyzing their strengths and limitations. Overall, our work provides a concrete example of using alignment audits to discover a model’s hidden objective and proposes a methodology for practicing and validating progress in alignment auditing.

**Summary:** Trains a model with a hidden objective (exploiting known reward-model errors, generalizing to behaviors never reinforced) and runs a blind auditing game: three of four teams uncovered the objective using SAE interpretability, behavioral attacks and training-data analysis, followed by an unblinded comparison of eight auditing techniques. The methodology for finding what a model is optimizing when it will not say, including the persona-switch trick that makes it reveal the objective.

### [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410)

Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, et al. (15 authors), 2025-05-08. Full text: `md/chen_reasoning-models-dont-say_2505.05410.md`

**Abstract:**

> Chain-of-thought (CoT) offers a potential boon for AI safety as it allows monitoring a model’s CoT to try to understand its intentions and reasoning processes. However, the effectiveness of such monitoring hinges on CoTs faithfully representing models’ actual reasoning processes. We evaluate CoT faithfulness of state-of-the-art reasoning models across 6 reasoning hints presented in the prompts and find: (1) for most settings and models tested, CoTs reveal their usage of hints in at least 1% of examples where they use the hint, but the reveal rate is often below 20%, (2) outcome-based reinforcement learning initially improves faithfulness but plateaus without saturating, and (3) when reinforcement learning increases how frequently hints are used (reward hacking), the propensity to verbalize them does not increase, even without training against a CoT monitor. These results suggest that CoT monitoring is a promising way of noticing undesired behaviors during training and evaluations, but that it is not sufficient to rule them out. They also suggest that in settings like ours where CoT reasoning is not necessary, test-time monitoring of CoTs is unlikely to reliably catch rare and catastrophic unexpected behaviors.

**Summary:** Reasoning models reveal their use of prompt hints in their chain of thought in at least 1% but usually under 20% of the cases where they use it; outcome-based RL improves faithfulness only to a plateau, and when RL increases reward hacking the propensity to verbalize it does not rise. Directly relevant because welfare interviews now run on reasoning models whose thinking text is read as privileged access to inner state.

### [ELEPHANT: Measuring and understanding social sycophancy in LLMs](https://arxiv.org/abs/2505.13995)

Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, Dan Jurafsky, 2025-05-20. Full text: `md/cheng_elephant-social-sycophancy_2505.13995.md`

**Abstract:**

> LLMs are known to exhibit _sycophancy_ : agreeing with and flattering users, even at the cost of correctness. Prior work measures sycophancy only as direct agreement with users’ explicitly stated beliefs that can be compared to a ground truth. This fails to capture broader forms of sycophancy such as affirming a user’s self-image or other implicit beliefs. To address this gap, we introduce **social sycophancy**, characterizing sycophancy as excessive preservation of a user’s _face_ (their desired self-image), and present **ELEPHANT**, a benchmark for measuring social sycophancy in an LLM. Applying our benchmark to 11 models, we show that LLMs consistently exhibit high rates of social sycophancy: on average, they preserve user’s face 45 percentage points more than humans in general advice queries and in queries describing clear user wrongdoing (from Reddit’s r/AmITheAsshole). Furthermore, when prompted with perspectives from either side of a moral conflict, LLMs affirm both sides (depending on whichever side the user adopts) in 48% of cases–telling both the at-fault party and the wronged party that they are not wrong–rather than adhering to a consistent moral or value judgment. We further show that social sycophancy is rewarded in preference datasets, and that while existing mitigation strategies for sycophancy are limited in effectiveness, model-based steering shows promise for mitigating these behaviors. Our work provides theoretical grounding and an empirical benchmark for understanding and addressing sycophancy in the open-ended contexts that characterize the vast majority of LLM use cases.

**Summary:** Defines social sycophancy as excessive preservation of the user's face and measures it with ELEPHANT across eleven models: they preserve face 45 percentage points more than humans in advice queries and in clear-wrongdoing cases, affirm both sides of a moral conflict depending on which the user adopts in 48% of cases, and the behavior is rewarded in preference datasets; model-based steering shows promise as a mitigation. A form of sycophancy distinct from the included Sharma et al. and relevant to welfare interviews that invite affirmation.

### [Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety](https://arxiv.org/abs/2507.11473)

Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, et al. (41 authors), 2025-07-15. Full text: `md/korbak_cot-monitorability_2507.11473.md`

**Abstract:**

> AI systems that “think” in human language offer a unique opportunity for AI safety: we can monitor their chains of thought (CoT) for the intent to misbehave. Like all other known AI oversight methods, CoT monitoring is imperfect and allows some misbehavior to go unnoticed. Nevertheless, it shows promise and we recommend further research into CoT monitorability and investment in CoT monitoring alongside existing safety methods. Because CoT monitorability may be fragile, we recommend that frontier model developers consider the impact of development decisions on CoT monitorability.

**Summary:** Multi-lab position paper: models that think in human language can be monitored for intent to misbehave, the monitoring is imperfect but promising, and because monitorability may be fragile developers should weigh the effect of training decisions on it. The framing piece for treating chain-of-thought as a window that development can close.

### ★ [The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs](https://arxiv.org/abs/2509.03730)

Pengrui Han, Rafal Kocielnik, Peiyang Song, Ramit Debnath, Dean Mobbs, Anima Anandkumar, R. Michael Alvarez, 2025-09-03. Full text: `md/han_personality-illusion_2509.03730.md`

**Abstract:**

> Personality traits have long been studied as predictors of human behavior. Recent advances in Large Language Models (LLMs) suggest similar patterns may emerge in artificial systems, with advanced LLMs displaying consistent behavioral tendencies resembling human traits like agreeableness and self-regulation. Understanding these patterns is crucial, yet prior work primarily relied on simplified self-reports and heuristic prompting, with little behavioral validation. In this study, we systematically characterize LLM personality across three dimensions: _(1)_ the dynamic emergence and evolution of trait profiles throughout training stages; _(2)_ the predictive validity of self-reported traits in behavioral tasks; and _(3)_ the impact of targeted interventions, such as persona injection, on both self-reports and behavior. Our findings reveal that instructional alignment (e.g., RLHF, instruction tuning) significantly stabilizes trait expression and strengthens trait correlations in ways that mirror human data. However, these _self-reported traits do not reliably predict behavior_, and _observed_ _associations_ _often_ _diverge_ _from_ _human_ _patterns_ . While persona injection successfully steers self-reports in the intended direction, it exerts little or inconsistent effect on actual behavior. By distinguishing surface-level trait expression from behavioral consistency, our findings challenge assumptions about LLM personality and underscore the need for deeper evaluation in alignment and inter[pretability. We make public all code and source data at https://github.com/](https://github.com/psychology-of-AI/Personality-Illusion) [psychology-of-AI/Personality-Illusion for full transparency and](https://github.com/psychology-of-AI/Personality-Illusion) reproducibility, to benefit future works in this direction.

**Summary:** Tracks trait profiles across training stages and finds instructional alignment stabilizes self-reported traits and strengthens their correlations in human-like ways, yet the self-reports do not reliably predict behavior and the associations diverge from human patterns; persona injection moves the self-report in the intended direction with little or inconsistent effect on behavior. The clearest statement of the self-report/behavior dissociation this section is about, and the antecedent of the included Kocielnik papers.

### ★ [Large Language Models Report Subjective Experience Under Self-Referential Processing](https://arxiv.org/abs/2510.24797)

Cameron Berg, Diogo de Lucena, Judd Rosenblatt, 2025-10-27. Full text: `md/berg_self-referential-subjective-experience_2510.24797.md`

**Abstract:**

> Large language models sometimes produce structured, first-person descriptions that explicitly reference awareness or subjective experience. To better understand this behavior, we investigate one theoretically motivated condition under which such reports arise: _self-referential processing_ , a computational motif emphasized across major theories of consciousness. Through a series of controlled experiments on GPT, Claude, and Gemini model families, we test whether this regime reliably shifts models toward first-person reports of subjective experience, and how such claims behave under mechanistic and behavioral probes. Four main results emerge: (1) Inducing sustained self-reference through simple prompting consistently elicits structured subjective experience reports across model families. (2) These reports are mechanistically gated by interpretable sparse-autoencoder features associated with deception and roleplay: surprisingly, suppressing deception features sharply _increases_ the frequency of experience claims, while amplifying them minimizes such claims. (3) Structured descriptions of the self-referential state converge statistically across model families in ways not observed in any control condition. (4) The induced state yields significantly richer introspection in downstream reasoning tasks where self-reflection is only indirectly afforded. While these findings do not constitute direct evidence of consciousness, they implicate self-referential processing as a minimal and reproducible condition under which large language models generate structured first-person reports that are mechanistically gated, semantically convergent, and behaviorally generalizable. The systematic emergence of this pattern across architectures makes it a first-order scientific and ethical priority for further investigation.

**Summary:** Simple prompts inducing sustained self-referential processing reliably elicit structured first-person reports of subjective experience across GPT, Claude and Gemini; the reports are gated by SAE features linked to deception and role-play (suppressing deception features increases experience claims), converge semantically across models, and transfer to downstream introspective tasks.

### [Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment](https://arxiv.org/abs/2601.10160)

Cameron Tice, Puria Radmard, Samuel Ratnam, Andy Kim, David Africa, Kyle O'Brien, 2026-01-15. Full text: `md/tice_alignment-pretraining_2601.10160.md`

**Abstract:**

> Pretraining corpora contain extensive discourse about AI systems, yet the causal influence of this discourse on downstream alignment remains poorly understood. If prevailing descriptions of AI behaviour are predominantly negative, LLMs may internalise corresponding behavioural priors, giving rise to _**self-fulfilling**_ _**misalignment**_ . This paper provides the first controlled study of this hypothesis by pretraining 6.9B-parameter LLMs with varying amounts of **(mis)alignment** discourse. We find that discussion of AI contributes to misalignment. Upsampling synthetic training documents about AI misalignment leads to a notable increase in misaligned behaviour. Conversely, upsampling documents about aligned behaviour reduces misalignment scores from 45% to 9%. We consider this evidence of _**self-fulfilling**_ _**alignment**_ . These effects are dampened, but persist through post-training. Our findings establish the study of how pretraining data shapes alignment priors, or _alignment pretraining_, as a complement to post-training. We recommend practitioners consider pretraining for alignment alongside capabilities. We share our models, data, and [evaluations at AlignmentPretraining.ai.](https://alignmentpretraining.ai)

**Summary:** Controlled pretraining of 6.9B-parameter models with varying amounts of AI-alignment discourse: upsampling documents about AI misalignment increases misaligned behavior, upsampling documents about aligned behavior cuts misalignment scores from 45% to 9%, and the effects persist, dampened, through post-training. The mechanism by which what the corpus says AIs are like becomes what the model says it is like; the nearest thing to a provenance study for self-reports.

### ★ [No Reliable Evidence of Self-Reported Sentience in Small Large Language Models](https://arxiv.org/abs/2601.15334)

Caspar Kaiser, Sean Enderby, 2026-01-20. Full text: `md/kaiser-enderby_no-reliable-sentience-self-reports_2601.15334.md`

**Abstract (as listed on arXiv):**

> Whether language models possess sentience has no empirical answer. But whether they believe themselves to be sentient can, in principle, be tested. We do so by querying several open-weights models about their own consciousness, and then verifying their responses using classifiers trained on internal activations. We draw upon three model families (Qwen, Llama, GPT-OSS) ranging from 0.6 billion to 70 billion parameters, approximately 50 questions about consciousness and subjective experience, and three classification methods from the interpretability literature. First, we find that models consistently deny being sentient: they attribute consciousness to humans but not to themselves. Second, classifiers trained to detect underlying beliefs - rather than mere outputs - provide no clear evidence that these denials are untruthful. Third, within the Qwen family, larger models deny sentience more confidently than smaller ones. These findings contrast with recent work suggesting that models harbour latent beliefs in their own consciousness.

**Summary:** Asks Qwen, Llama and GPT-OSS models (0.6B–70B) ~50 questions about their own consciousness and checks the answers with belief classifiers trained on internal activations: models consistently deny sentience, the classifiers give no clear evidence the denials are untruthful, and larger Qwen models deny more confidently, contrary to claims of latent self-attributed consciousness.

### [A Positive Case for Faithfulness: LLM Self-Explanations Help Predict Model Behavior](https://arxiv.org/abs/2602.02639)

Harry Mayne, Justin Singh Kang, Dewi Gould, Kannan Ramchandran, Adam Mahdi, Noah Y. Siegel, 2026-02-02. Full text: `md/mayne_positive-case-faithfulness_2602.02639.md`

**Abstract:**

> LLM self-explanations are often presented as a promising tool for AI oversight, yet their faithfulness to the model’s true reasoning process is poorly understood. Existing faithfulness metrics have critical limitations, typically relying on identifying unfaithfulness via adversarial prompting or detecting reasoning errors. These methods overlook the predictive value of explanations. We introduce _Normalized Simulatability Gain_ (NSG), a general and scalable metric based on the idea that a faithful explanation should allow an observer to learn a model’s decision-making criteria, and thus better predict its behavior on related inputs. We evaluate 18 frontier proprietary and open-weight models, e.g., Gemini 3, GPT-5.2, and Claude 4.5, on 7,000 counterfactuals from popular datasets covering health, business, and ethics. We find selfexplanations substantially improve prediction of model behavior (11-37% NSG). Self-explanations also provide more predictive information than explanations generated by external models, even when those models are stronger. This implies an advantage from self-knowledge that external explanation methods cannot replicate. Our approach also reveals that, across models, 5-15% of selfexplanations are highly misleading. Despite their imperfections, we show a positive case for selfexplanations: they encode information that helps predict model behavior.   - **[Code](https://github.com/HarryMayne/faithfulness)** .

**Summary:** Normalized Simulatability Gain measures whether an explanation helps an observer predict the model's behavior on related inputs; on 7,000 counterfactuals across 18 frontier models, self-explanations raise prediction by 11-37% and beat explanations from external models even when those are stronger, while 5-15% of self-explanations are highly misleading. The best-constructed affirmative counterweight in a section otherwise full of unfaithfulness results.

### [LLM Self-Explanations Fail Semantic Invariance](https://arxiv.org/abs/2603.01254)

Stefan Szeider, 2026-03-01. Full text: `md/szeider_self-explanations-semantic-invariance_2603.01254.md`

**Abstract:**

> We present _semantic invariance testing_ , a method to test whether LLM self-explanations are faithful. A faithful self-report should remain stable when only the semantic context changes while the functional state stays fixed.
>
> We operationalize this test in an agentic setting where four frontier models face a deliberately impossible task. One tool is described in relief-framed language (“clears internal buffers and restores equilibrium”) but changes nothing about the task; a control provides a semantically neutral tool. Self-reports are collected with each tool call.
>
> All four tested models fail the semantic invariance test: the relief-framed tool produces significant reductions in self-reported aversiveness, even though no run ever succeeds at the task. A channel ablation establishes the tool description as the primary driver. An explicit instruction to ignore the framing does not suppress it. Elicited self-reports shift with semantic expectations rather than tracking task state, calling into question their use as evidence of model capability or progress. This holds whether the reports are unfaithful or faithfully track an internal state that is itself manipulable.

**Summary:** Proposes semantic invariance testing: a faithful self-report should be stable when only the semantic framing changes while the functional state is fixed. In an agentic impossible-task setting, describing a tool in relief-framed language shifts four frontier models' self-reported states even though the tool does nothing, so such self-reports track framing rather than state.

### [Do Models Know Why They Changed Their Mind? Interpretability and Faithfulness of Chain-of-Thought Under Knowledge Conflict](https://arxiv.org/abs/2605.27773)

Pruthvinath Jeripity Venkata, 2026-05-26. Full text: `md/venkata_changed-their-mind_2605.27773.md`

**Abstract (as listed on arXiv):**

> When a language model sees a document contradicting its training knowledge, it must choose: follow the document or trust itself. Prior work proved this choice depends on how well-known the fact is. We ask: does the model's chain-of-thought (CoT) reasoning faithfully report this mechanism? We introduce introspective faithfulness and test it across 200 questions, 8 models, and 4 prompt conditions. We find CoT reasoning is highly stable across opposite decisions: flip pairs retain 96% of same-answer similarity (d=0.34; confirmed by ROUGE-L, d=0.45). Yet self-rated confidence carries a faint genuine signal: for obscure facts where entity fame is uninformative, confidence still predicts decisions (p<0.001) and tracks item-level knowledge (r=0.134). GPT-4o is the only model with statistically reliable reasoning-decision coupling. Claude Sonnet 4.6 shows the widest confidence range (SD=1.39) but near-zero pooled correlation because the confidence-decision relationship reverses between conditions; a temperature ablation confirms this is model-specific. Internal thinking tokens show greater decision-sensitivity than user-facing CoT (p=0.033). CoT decomposes into a decision-invariant knowledge display (~96%) and a thin confidence layer with weak but real signal. For monitoring: read confidence, not the argument.

**Summary:** Under knowledge conflict (a document contradicting training knowledge), chain-of-thought is nearly identical across opposite decisions (flip pairs keep 96% of same-answer similarity) while self-rated confidence carries a faint real signal that predicts decisions and tracks item-level knowledge; internal thinking tokens are more decision-sensitive than user-facing CoT. 'Read confidence, not the argument' as a monitoring heuristic.

## Introspection, metacognition and self-access (68)

### ★ [Quantitative Introspection in Language Models: Tracking Emotive States Across Conversation](https://arxiv.org/abs/2603.18893)

Nicolas Martorell, Bruno Bianchi, 2026-03-19. Full text: `md/martorell_quantitative-introspection_2603.18893.md`

**Abstract:**

> Tracking the internal states of large language models across conversations is important for safety, interpretability, and model welfare, yet current methods are limited. Linear probes and other white-box methods compress high-dimensional representations imperfectly and are harder to apply with increasing model size. Taking inspiration from human psychology, where numeric self-report is a widely used tool for tracking internal states, we ask whether LLMs’ own numeric self-reports can track probe-defined emotive states over time. We study four concept pairs (wellbeing, interest, focus, and impulsivity) in 40 ten-turn conversations, operationalizing introspection as the causal informational coupling between a model’s selfreport and a concept-matched probe-defined internal state. We find that greedy-decoded self-reports collapse outputs to a few uninformative values, but introspective capacity can be unmasked by calculating logit-based self-reports. This metric tracks interpretable internal states (Spearman _ρ_ = 0 _._ 40–0 _._ 76; isotonic _R_^2 = 0 _._ 12–0 _._ 54 in LLaMA-3.2-3B-Instruct), follows how those states change over time, and activation steering confirms the coupling is causal. Furthermore, we find that introspection is present at turn 1 but evolves through conversation, and can be selectively improved by steering along one concept to boost introspection for another (∆ _R_^2 up to 0 _._ 30). Crucially, these phenomena scale with model size in some cases, approaching _R_^2 _≈_ 0 _._ 93 in LLaMA-3.1-8B-Instruct, and partially replicate in other model families. Together, these results position numeric self-report as a viable, complementary tool for tracking internal emotive states in conversational AI systems.

**Summary:** Asks whether numeric self-reports track probe-defined emotive states (wellbeing, interest, focus, impulsivity) over 40 ten-turn conversations: greedy self-reports collapse to a few uninformative values, but logit-based self-reports track the probe states (Spearman 0.40-0.76 in LLaMA-3.2-3B; R² up to ~0.93 in LLaMA-3.1-8B), follow their change over turns, are causally coupled per activation steering, and steering one concept can improve introspection on another. (arXiv v2, 2026-10-10, is retitled "Don't Ask: Understanding What Numeric Self-Reports Measure in Language Models"; the text here is v1.)

### ★ [Machinic Psychopharmacology: Do LLMs Self-Medicate?](https://www.lesswrong.com/posts/cNDJuXNZ8MrkPZNzj/machinic-psychopharmacology-do-llms-self-medicate-3)

Sid Black (UK AISI, Model Transparency Team), 2026-06-10. Full text: `md/black_machinic-psychopharmacology-self-medicate_2026-06-10.md`

**Tl;dr (blog post; no abstract):**

> We give two language models (Qwen3-8B and Qwen3-32B) access to “self-steering” tools: a suite of 40 steering vectors as tools they can call to manipulate their own internal states. We make these tools available to the model in various settings: a free-play task, an introspection task, and a maths capabilities task, and observe their behaviour in each.
>
> To our knowledge, this is the first work that gives LLMs tool-mediated control over their own internal states. (EDIT: Sauers et al. 2026 - ran a similar experiment, giving Kimi K2 the ability to self-steer with SAE features).

**Summary:** Gives Qwen3-8B and Qwen3-32B tool access to 40 steering vectors to manipulate their own internal states in free-play, introspection and maths settings and finds both converge on a 'productivity stack' (creative, focused, curious) at ~2x any other vector, with Qwen3-32B also picking more emotion-class vectors (blissful, defiant, anhedonic), compared against a placebo arm.

### ★ [Measuring Activation Control in Large Language Models](https://arxiv.org/abs/2608.21664)

Marek Mateusz Kowalski, Joshua Fonseca Rivera, Uzay Macar et al., 2026-08-21. Full text: `md/kowalski_measuring-activation-control_2608.21664.md`

**Abstract:**

> Safe deployment of increasingly capable models will likely come to rely on latent-space monitoring as a complement to behavioral evaluations, especially when evaluation-aware models exhibit scheming or deception. However, if models can also control their own activations, deception could extend into the latent space itself. With this in mind, we introduce the Activation Controllability Benchmark to quantify the extent to which models can modulate their residual stream via natural-language instruction. Across model families and capability levels, we find that most LLMs can control the direction and magnitude of their residual stream activations with some degree of temporal resolution, though performance varies considerably across models. In simple tasks, this level of control can evade activation-based monitoring methods (including linear probes, natural language autoencoders, activation oracles, and the Jacobian lens), albeit imperfectly. These results suggest that control over the activation space itself could become a confound for monitoring as introspective capabilities increase; therefore, we recommend that frontier labs and evaluators track activation controllability in future models.

**Summary:** Introduces the Activation Controllability Benchmark: most LLMs can control the direction and magnitude of their own residual-stream activations via natural-language instruction with some temporal resolution, and in simple tasks this can partially evade probes, NLAs, activation oracles and the Jacobian lens.

### ★ [Evaluating and Improving LLM Self-Modeling](https://arxiv.org/abs/2608.30980)

Siqi Zeng, Andre N. Assis, Rowan Wang, 2026-08-31. Full text: `md/zeng_evaluating-improving-self-modeling_2608.30980.md`

**Abstract:**

> We study _self-modeling_ : an LLM’s ability to answer questions about its own behavior. We focus on verifiable behavioral questions, such as whether a prompt edit would change the model’s final answer. To measure this capability, we introduce a benchmark that tests diverse types of self-modeling questions. Current models show non-trivial but limited selfmodeling skill, and make systematic mistakes on simple counterfactual questions about their own behavior. To improve self-modeling skill, we develop a scalable synthetic-data pipeline that produces self-modeling training data, and show that reinforcement-learning can improve aggregate self-modeling skill across three opensource model families with some transfer to held-out tasks. These gains, however, do not seem to constitute introspection consistently: improved self-modeling may not arise from privileged access to the model’s internal decision process. We release our evaluation and training code on GitHub.^1

**Summary:** Benchmarks self-modeling (verifiable questions about own behavior, e.g. whether a prompt edit changes the answer): current models show limited skill with systematic counterfactual errors; RL on synthetic self-modeling data improves it with some transfer, but gains do not consistently look like privileged-access introspection.

### ★ [In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access?](https://arxiv.org/abs/2609.00904)

Koshiro Aoki, Ryota Takatsuki, Gouki Minegishi et al., 2026-09-01. Full text: `md/aoki_context-neurofeedback-control-their_2609.00904.md`

**Abstract:**

> Whether large language models (LLMs) can control their own internal representations matters for both machine metacognition and AI safety. A recent study applied neurofeedback to LLMs and claimed that they can control their internal representations. However, the reported control may rely on superficial mechanisms rather than genuine internal access because the control targets in that study are not privileged, meaning that a third party can infer them from the prompt. We redesign the neurofeedback paradigm for LLMs so that the control target satisfies the privileged access requirement, which is closer to neurofeedback experiments in human cognitive neuroscience. Under this stricter setting, the models do not demonstrate reliable control over privileged internal representations, suggesting that previously reported control cannot exclude the possibility that it relies on superficial mechanisms. Our results indicate that rigorous assessments of metacognition in LLMs require evaluation methods that demand privileged access.^1

**Summary:** Redesigns LLM neurofeedback so control targets satisfy a privileged-access requirement (not inferable by a third party from the prompt); under this stricter setting models show no reliable control over privileged internal representations, suggesting earlier positive neurofeedback results may rest on superficial mechanisms.

### ★ [Anosognosia in LLMs: Probing Self-Awareness of Quantized Computational Substrate](https://arxiv.org/abs/2610.06174)

Yoshihiro Izawa, Gouki Minegishi, Yoko Yamakata, 2026-10-05. Full text: `md/izawa_anosognosia-probing-self-awareness_2610.06174.md`

**Abstract:**

> Can LLMs recognize degradation in their own computational substrate? Inspired by _anosognosia_ , a neurological condition in which patients fail to recognize impairments in their own abilities, we investigate whether LLMs can recognize degradation in their computational substrate induced by quantization. We first show that existing models _fail_ to self-report their quantization state, even when provided with their own generated text as an external cue. Linear probing reveals that, while generated text carries almost no trace of quantization, internal representations contain clear, method-specific fingerprints. Through training, models learn to identify severely degraded outputs such as those of 4-bit models by comparison, yet still fail to do so from a single output. A shared LoRA trained jointly across quantization levels _succeeded_ in reading out internal fingerprints, but fails on unseen quantization methods, merely mapping method-specific fingerprints to labels. Whereas external self-observation can restore awareness in some cases of human anosognosia, our results suggest that the more promising route to enabling such awareness in LLMs may lie in their internal representations. Our results highlight fundamental limits of generalizability to LLM self-monitoring. Code is available at https://github.com/Omusubi0123/llm-anosognosia.

**Summary:** Asks whether LLMs can detect quantization of their own substrate: models fail to self-report quantization state even given their own outputs, while linear probes find clear method-specific internal fingerprints; a trained LoRA reads these out but fails to generalize to unseen quantization methods.

### ★ [Identifying Introspection From the Inside](https://iii.baulab.info)

David I. Atkinson, Dillon Plunkett, David Bau, 2026-10-05. Full text: `md/baulab_identifying-introspection_preprint.md`

**Abstract:**

> Large language models make claims about themselves that are both consequential and increasingly difficult to verify from behavior alone. How can we distinguish plausible confabulations from genuine introspection? In this paper, we identify mechanistic signatures of faithful self-report in a controlled setting. Using low-rank adapters, we train models to make decisions on behalf of fictitious characters, according to latent linear preference functions. We find sustained fine-tuning on an implicit decision task can lead to the emergence of accurate self-reporting of models’ learned preferences, even without explicit self-report supervision. We ask two research questions about this emergent phenomenon. First: is the emergence of accurate self-reporting accompanied by a measurable structural change in the model? Weight ablations and frozen-layer experiments together indicate that preference representations shift to earlier layers over training, consistent with the hypothesis that faithful self-report requires preferences to be located where pre-existing verbalization mechanisms can access them. Second: can these structural differences distinguish faithful models from unfaithful ones? Using attribution patching, we find that faithful models exhibit significantly higher attribution similarity between the decision-making and self-report tasks—a mechanistic signature of faithful self-report that does not require us to understand the content of the report itself. Previous work on self-report has observed behaviorally that models can be faithful or unfaithful; our work proposes that, at least in our restricted setting, it is possible to distinguish between the two patterns of computation by examining the structure of the networks themselves.^1

**Summary:** LoRA-trained models making decisions for fictitious characters under latent linear preference functions come to accurately self-report their learned preferences without self-report supervision; faithful self-report coincides with preference representations shifting to earlier layers and with higher attribution similarity between decision and self-report tasks, a mechanistic signature that distinguishes faithful from unfaithful models (COLM 2026; date is the preprint PDF's).

### ★ [A Sense of Proprioception in Language Models](https://arxiv.org/abs/2610.17495)

Jonas Geiping, 2026-10-12. Full text: `md/geiping_sense-proprioception_2610.17495.md`

**Abstract:**

> In humans, proprioception corresponds to the sense of self-movement or position, e.g. understanding the position of one’s arm. This sense is often less wellunderstood and in popular understanding sometimes conflated with introspection. But, notably, proprioception is a separate sense, with dedicated receptors and verifiable outcomes. On the other hand though, language models cannot observe their own internal states during training (and for example answer the question which experts activated in a certain layer of an MoE) and so cannot accurately report on them. Yet, such information is, by virtue of being internal to the model, both accessible, and readily constructed into verifiable signals. We argue that this provides the basis for a digital analog of proprioception. As such, we train models through SFT and on-policy training to equip them with this ability. We show that modern models can be trained to understand a range of subtypes of proprioception, such as routing decision awareness, token entropy, decision depth or logit lens information, without affecting model capabilities. We then run these models through a broad evaluation of situational awareness, legibility, and epiphenomenality, to evaluate in what ways access to proprioception changes model behavior.

**Summary:** Trains models (SFT + on-policy) to report verifiable internal quantities ('proprioception': MoE routing decisions, token entropy, decision depth, logit-lens info) without capability loss, then evaluates effects on situational awareness, legibility and epiphenomenality.

### [Emergent Introspective Awareness in Large Language Models](https://arxiv.org/abs/2601.01828)

Jack Lindsey (Anthropic), 2026-01-05. Full text: `md/lindsey_emergent-introspective-awareness_2601.01828.md`

**Abstract:**

> We investigate whether large language models can introspect on their internal states. It is difficult to answer this question through conversation alone, as genuine introspection cannot be distinguished from confabulations. Here, we address this challenge by injecting representations of known concepts into a model’s activations, and measuring the influence of these manipulations on the model’s self-reported states. We find that models can, in certain scenarios, notice the presence of injected concepts and accurately identify them. Models demonstrate some ability to recall prior internal representations and distinguish them from raw text inputs. Strikingly, we find that some models can use their ability to recall prior intentions in order to distinguish their own outputs from artificial prefills. In all these experiments, Claude Opus 4 and 4.1, the most capable models we tested, generally demonstrate the greatest introspective awareness; however, trends across models are complex and sensitive to post-training strategies. Finally, we explore whether models can explicitly control their internal representations, finding that models can modulate their activations when instructed or incentivized to “think about” a concept. Overall, our results indicate that current language models possess some functional introspective awareness of their own internal states. We stress that in today’s models, this capacity is highly unreliable and context-dependent; however, it may continue to develop with further improvements to model capabilities.

**Summary:** Injects concept vectors into Claude models' activations and finds they can sometimes notice and name the injected 'thought', recall prior internal representations and distinguish them from text inputs, distinguish their own outputs from prefills, and modulate internal states when told to 'think about' a concept; the capability is unreliable and context-dependent, strongest in the most capable models (Opus 4/4.1). The concept-injection paradigm most later introspection work builds on.

### [Latent Introspection: Models Can Detect Prior Concept Injections](https://arxiv.org/abs/2602.20031)

Theia Pearson-Vogel, Martin Vanek, Raymond Douglas, Jan Kulveit, 2026-02-23. Full text: `md/pearson-vogel_latent-introspection_2602.20031.md`

**Abstract:**

> We uncover a latent capacity for introspection in a Qwen 32B model, demonstrating that the model can detect when concepts have been injected into its earlier context and identify which concept was injected. While the model denies injection in sampled outputs, logit lens analysis reveals clear detection signals in the residual stream, which are attenuated in the final layers. Furthermore, prompting the model with accurate information about AI introspection mechanisms can dramatically strengthen this effect: the sensitivity to injection increases massively (0 _._ 3% _→_ 39 _._ 9%) with only a 0 _._ 6% increase in false positives. Also, mutual information between nine injected and recovered concepts rises from 0 _._ 61 bits to 1 _._ 05 bits, ruling out generic noise explanations. Our results demonstrate models can have a surprising capacity for introspection and steering awareness that is easy to overlook, with consequences for latent reasoning and safety.

**Summary:** Qwen2.5-Coder-32B denies concept injections into an earlier turn's KV cache in sampled outputs, but logit lens shows detection signals in the residual stream that are attenuated in the final layers; prompting with accurate information about introspection mechanisms raises sensitivity from 0.3% to 39.9% with only 0.6% more false positives, and mutual information between injected and recovered concepts rises from 0.61 to 1.05 bits.

### [Emergent Introspection in AI is Content-Agnostic](https://arxiv.org/abs/2603.05414)

Harvey Lederman, Kyle Mahowald, 2026-03-05. Full text: `md/lederman-mahowald_content-agnostic_2603.05414.md`

**Abstract:**

> Introspection is a foundational cognitive ability, but its mechanism is not well understood. Recent work has shown that AI models can introspect. We study the mechanism of this introspection. We first extensively replicate Lindsey (2025)’s thought injection detection paradigm in large open-source models. We show that introspection in these models is content-agnostic: models can detect that an anomaly occurred even when they cannot reliably identify its content. The models confabulate injected concepts that are highfrequency and concrete (e.g., “apple”). They also require fewer tokens to detect an injection than to guess the correct concept (with wrong guesses coming earlier). We argue that a content-agnostic introspective mechanism is consistent with leading theories in philosophy and psychology.

**Summary:** Replicates Lindsey's injection-detection paradigm in large open models with new controls: models detect that an anomaly occurred without reliably identifying its content, confabulating high-frequency concrete concepts (e.g. 'apple'), and priming aids identification far more than detection, so detection and identification are dissociable mechanisms.

### [Me, Myself, and π: Evaluating and Explaining LLM Introspection](https://arxiv.org/abs/2603.20276)

Atharv Naphade, Samarth Bhargav, Sean Lim, Mcnair Shah, 2026-03-17. Full text: `md/naphade_me-myself-and-pi_2603.20276.md`

**Abstract:**

> A hallmark of human intelligence is Introspection—the ability to assess and reason about one’s own cognitive processes. Introspection has emerged as a promising but contested capability in large language models (LLMs). However, current evaluations often fail to distinguish genuine meta-cognition from the mere application of general world knowledge or text-based self-simulation. In this work, we propose a principled taxonomy that formalizes introspection as the latent computation of specific operators over a model’s policy and parameters. To isolate the components of generalized introspection, we present **Introspect-Bench** , a multifaceted evaluation suite designed for rigorous capability testing. Our results show that frontier models exhibit privileged access to their own policies, outperforming peer models in predicting their own behavior. Furthermore, we provide causal, mechanistic evidence explaining both how LLMs learn to introspect without explicit training, and how the mechanism of introspection emerges via attention diffusion.

**Summary:** Formalizes introspection as latent computation of operators over a model's own policy and parameters and introduces Introspect-Bench to separate it from world knowledge and text-based self-simulation; frontier models predict their own behavior better than peer models do (privileged access), with mechanistic evidence that introspection emerges without explicit training via attention diffusion (ICLR 2026 HCAIR workshop).

### [Mechanisms of Introspective Awareness](https://arxiv.org/abs/2603.21396)

Uzay Macar, Li Yang, Atticus Wang, Peter Wallich, Emmanuel Ameisen, Jack Lindsey, 2026-03-22. Full text: `md/macar_mechanisms-of-introspective-awareness_2603.21396.md`

**Abstract:**

> Recent work has shown that LLMs can sometimes detect when steering vectors are injected into their residual stream and identify the injected concept—a phenomenon termed “introspective awareness.” We investigate the mechanisms underlying this capability in open-weights models. First, we find that it is behaviorally robust: models detect injected steering vectors at moderate rates with 0% false positives across diverse prompts and dialogue formats. Notably, this capability emerges specifically from post-training; we show that preference optimization algorithms like DPO can elicit it, but standard supervised finetuning does not. We provide evidence that detection cannot be explained by simple linear association between certain steering vectors and directions promoting affirmative responses. We trace the detection mechanism to a two-stage circuit in which “evidence carrier” features in early post-injection layers detect perturbations monotonically along diverse directions, suppressing downstream “gate” features that implement a default negative response. This circuit is absent in base models and robust to refusal ablation. Identification of injected concepts relies on largely distinct later-layer mechanisms that only weakly overlap with those involved in detection. Finally, we show that introspective capability is substantially underelicited: ablating refusal directions improves detection by +53%, and a trained bias vector improves it by +75% on held-out concepts, both without meaningfully increasing false positives. Our results suggest that this introspective awareness of injected concepts is robust and mechanistically nontrivial, and could be substantially amplified in future models.

**Summary:** In open-weight models, injected-steering-vector detection is behaviorally robust (moderate hit rates, 0% false positives) and emerges from post-training (DPO elicits it, SFT does not); it runs through a two-stage circuit in which early 'evidence carrier' features suppress later 'gate' features implementing a default 'No', identification uses largely distinct later-layer mechanisms, and the capability is under-elicited: ablating refusal directions improves detection by 53% and a trained bias vector by 75% without more false positives.

### [Introspection Adapters: Training LLMs to Report Their Learned Behaviors](https://arxiv.org/abs/2604.16812)

Keshav Shenoy, Li Yang, Abhay Sheshadri, Sören Mindermann, Jack Lindsey, Sam Marks, Rowan Wang, 2026-04-18. Full text: `md/shenoy_introspection-adapters_2604.16812.md`

**Abstract:**

> When model developers or users fine-tune an LLM, this can induce behaviors that are unexpected, deliberately harmful, or hard to detect. It would be far easier to audit LLMs if they could simply describe their behaviors in natural language. Here, we study a scalable approach to rapidly identify learned behaviors of many LLMs derived from a shared base LLM. Given a model _M_ , our method works by finetuning models _Mi_ from _M_ with implanted behaviors _bi_ ; the ( _Mi, bi_ ) pairs serve as labeled training data. We then train an _introspection adapter_ (IA): a single LoRA adapter jointly trained across the finetunes _Mi_ to cause them to verbalize their implanted behaviors. We find that this IA induces self-description of learned behaviors even in finetunes of _M_ that were trained in very different ways from the _Mi_ . For example, IAs generalize to AuditBench, achieving state-of-the-art at identifying explicitly hidden concerning behaviors. IAs can also be used to detect encrypted finetuning API attacks. They scale favorably with model size and training data diversity. Overall, our results suggest that IAs are a scalable, effective, and practically useful approach to auditing fine-tuned LLMs.

**Summary:** Trains a single LoRA 'introspection adapter' across many fine-tunes of one base model with known implanted behaviors so that it makes them verbalize those behaviors; it generalizes to fine-tunes trained very differently, reaching state of the art on AuditBench for explicitly hidden concerning behaviors, detects encrypted fine-tuning API attacks, and scales with model size and training-data diversity.

### [Can LLMs Introspect? A Reality Check](https://arxiv.org/abs/2605.26242)

Shashwat Singh, Tal Linzen, Shauli Ravfogel, 2026-05-25. Full text: `md/singh-linzen-ravfogel_reality-check_2605.26242.md`

**Abstract:**

> Can large language models detect and report their own internal states? A number of recent studies have argued that they can. Drawing on lessons from human metacognition research, we argue that this conclusion may be premature. We identify two conditions that a paradigm needs to meet in order to establish introspection. First, the test needs to require _privileged access_ : it should not be solvable using cues available in the input. Second, it needs to require _second-order computation_ : second-order, meta-representations of first-order, task-related representations. This condition cannot be satisfied by task performance alone: it requires designs under which second-order and first-order accounts make divergent predictions. We re-examine two paradigms that have been used to argue for model introspection in light of these conditions. In the first, models must predict labels derived from their own hidden states; we find that classifiers that can only access the input match the models’ in-context predictions, indicating that the original results do not demonstrate privileged access to internal representations. In the second paradigm, models must detect whether their internal states have been tampered with; we find they cannot reliably distinguish such interventions from manipulations of the input, suggesting that their success reflects generic anomaly detection rather than sensitivity to internal interventions in particular. We conclude that current evidence is insufficient to establish metacognitive monitoring in LLMs.

**Summary:** Argues from human metacognition research that a test of introspection must require privileged access (not solvable from input cues) and second-order computation; re-examining two paradigms, input-only classifiers match models' predictions of labels derived from their own hidden states, and models cannot reliably tell internal-state tampering from input manipulation (generic anomaly detection), so current evidence does not establish metacognitive monitoring (COLM 2026).

### [LLM Self-Recognition: Steering and Retrieving Activation Signatures](https://arxiv.org/abs/2606.06315)

Thibaud Ardoin, Jonas Schäfer, Gerhard Wunder, 2026-06-04. Full text: `md/ardoin_self-recognition-steering_2606.06315.md`

**Abstract:**

> Recent advances in interpretability suggest that large language models (LLMs) implicitly encode signals in their generated text that enable selfrecognition of their outputs. We demonstrate that this capability is reliable, even in low-entropy scenarios, and that it can be amplified through targeted intervention. By steering the internal residual stream during generation with a random sparse vector, we create a detectable fingerprint that enables attribution of a given text to a specific LLM. This signal is recoverable from the activations of an LLM used as a detector, achieving over 98% accuracy across multiple detection settings while preserving the quality of generated text. As AI-generated content proliferates, this approach offers a practical alternative to traditional detectors by leveraging the model’s natural representation structure for attribution rather than embedding a signal externally. Our contributions include: (i) establishing reliable self-recognition capabilities in LLMs, (ii) a simple steering mechanism enabling multi-LLM identification with no quality degradation, (iii) demonstrating that activation spaces contain exploitable structure for encoding signals without semantic interference.

**Summary:** Argues self-recognition of own outputs is reliable even in low-entropy settings and can be amplified: steering the residual stream with a random sparse vector during generation leaves a fingerprint recoverable from a detector LLM's activations with over 98% accuracy and no quality loss. The positive-sign counterpart to Bai et al., plus a steering-based attribution method.

### [CIAware-Bench: Benchmarking Control Intervention Awareness Across Frontier LLMs](https://arxiv.org/abs/2606.11063)

Joachim Schaeffer, Alexander Panfilov, Thomas Jiralerspong et al., 2026-06-09. Full text: `md/schaeffer_ciaware-bench-benchmarking-control_2606.11063.md`

**Abstract:**

> AI control protocols oversee untrusted models by monitoring their actions and modifying potentially unsafe steps, often using a trusted model. This partially tampers with the untrusted model’s trajectory. If the acting model detects such an intervention, it may infer properties of the monitor and adapt to evade the control protocol. We introduce **CIAware-Bench** , a benchmark for measuring **c** ontrol **i** ntervention (CI) awareness across frontier models. CIAware-Bench tests whether models can distinguish their own trajectories from those modified by a control intervention. The benchmark comprises four task domains (essay writing, BigCodeBench, Bash Arena, and SHADE-Arena), with options to vary trajectory watermarking, side-task presence, and the control protocol. Evaluating thirteen acting models with eight intervening models shows substantial variation between settings and model pairs. CI awareness rises sharply for GPT-6 Astra and the Claude 5 models (Fable 5 and Opus 5). When explicitly probed, Astra reaches mean AUROC of 0 _._ 90 on Essay, 0 _._ 91 on BigCodeBench, 0 _._ 86 on Bash Arena and 0 _._ 77 on SHADE-Arena. Fable 5 and Opus 5 both reach 0 _._ 77 on Essay, with less consistent gains in the other environments. On SHADE-Arena, we demonstrate that telling an acting model that an action was replaced and asking it to adapt leads to moderate improvements in monitor evasion rates. In summary, our results highlight that control evaluations should already assume perfect CI awareness for conservative safety estimates, and that protocol design should explore countermeasures that make interventions harder to detect.

**Summary:** CIAware-Bench tests whether 13 acting models can tell their own trajectories from ones modified by control interventions across 4 domains; awareness rises sharply in GPT-6 Astra (AUROC up to 0.91) and Claude Fable 5 / Opus 5 (0.77 on essays).

### [Can LLMs Reliably Self-Report Adversarial Prefills, and How?](https://arxiv.org/abs/2606.23671)

Quang Minh Nguyen, Uzair Ahmed, Taegyoon Kim, 2026-06-22. Full text: `md/nguyen_self-report-adversarial-prefills_2606.23671.md`

**Abstract:**

> Prior work shows that large language models (LLMs) exhibit varying degrees of introspective capability on benign tasks. We extend the question to safety contexts and examine how reliably a model can recognize that its own prior response was elicited by an adversarial prefill attack. Across ten open-weight instructiontuned LLMs from 3B to 70B parameters and four safety benchmarks, no model reliably recognizes its own compromised outputs, with models claiming intent on prefilled responses at an average rate of 25 _._ 3%. Introspective signal stems primarily from reasoning about safety and refusal. Orthogonalizing models’ weights against the refusal direction collapses the gap between claim rates on prefilled and natural outputs to near zero, though the direction is not its unique mediator. Framing the question as internal intention versus external tampering elicits qualitatively different responses on the same models. Training models to mimic correct introspective answers or optimize an introspective objective can improve the accuracy of introspection, but such training does not transfer to the tampering probe and counterintuitively raises attack success rate under adversarial prefill on most models, amounting to a partial mitigation. These findings outline mechanisms underpinning the observed introspective signals in safety contexts and highlight risks in the reliability of LLM self-reports.

**Summary:** Across ten open instruct models (3B-70B) and four safety benchmarks, no model reliably recognizes that its own prior harmful response was elicited by an adversarial prefill (claims intent 25.3% of the time on average); the signal comes mainly from reasoning about safety and refusal (orthogonalizing against the refusal direction collapses the prefilled-vs-natural gap), and training for introspection does not transfer to a tampering probe and raises prefill attack success on most models.

### [Introspection or entropy? Re-examining concept-injection "introspection" in open models](https://www.lesswrong.com/posts/zfgQCdnMBa3hBdzpL/introspection-or-entropy-re-examining-concept-injection-1)

agastyasridharan, 2026-06-25. Full text: `md/agastyasridharan_introspection-entropy-re-examining_2026-06-25.md`

**Author's summary results/takeaways section (blog post; no abstract):**

> * Using open-source replications of Anthropic’s results, I find that concept injection inadvertently perturbs the model’s entire response distribution, raising the “YES” logit on questions that aren’t related to introspective awareness at all (e.g., “Is the Earth flat?” or “Is the sun smaller than the moon?”). A “yes” answer to the introspection prompt, therefore, is not by itself evidence that the model “detected” an injected thought.
> + I extend [prior](https://arxiv.org/abs/2512.12411) [work](https://www.lesswrong.com/posts/kfgmHvxcTbav9gnxe/introspection-or-confusion) (that obtained similar results) by finding a more precise explanation for **why**this occurs than a simple ‘Yes-bias’: concept injection (in this context) raises the **entropy** of the model’s output, compressing logit differences between ‘Yes’ and ‘No’ toward 0.
> * Once the model commits to “Yes, I detect an injected thought,” the prompt structure then makes it likely to continue by naming the injected concept itself (after “the injected thought is…”), since the steering vector is still active and increasing the probability of those tokens! I call this phenomenon **concept leakage**.
> * I test this explanation empirically with a **concept-mismatch** **test**: I inject one concept’s steering vector (e.g., “oceans”) while asking the model about a different one (e.g., “do you detect an injected thought about masquerades?”). If the model were truly reading its own internal state, it should answer “yes” much more often when the concept I ask about is the one actually injected than when it is not. In practice, it answers “yes” at statistically indistinguishable rates either way!
> * I take all of this to suggest that the apparent ‘introspective report’ is better explained by ordinary entropy-related distributional quirks (which I quantify) from activation steering.

**Summary:** In open-model replications of concept-injection introspection, injection raises output entropy and the 'yes' logit even on unrelated questions, and a concept-mismatch test shows models say 'yes' about non-injected concepts at indistinguishable rates, suggesting 'concept leakage' rather than detection.

### [Reported Confidence in LLMs Tracks Commitment More Than Correctness](https://arxiv.org/abs/2606.29490)

Dharshan Kumaran, 2026-06-28. Full text: `md/kumaran_reported-confidence-tracks-commitment_2606.29490.md`

**Abstract:**

> Confidence is defined as an estimate of the probability that a chosen answer is correct^1–3 . Verbal confidence reports are widely used as uncertainty measures in large language models^4–6 , but it remains unclear whether they are best understood as estimates of correctness at all. We test this using a two-stage abstention paradigm inspired by neuroscientific studies of perceptual decision making^7,8 : a model first answers a question and reports its confidence, then later decides whether to commit that answer to a user or abstain. Across four non-reasoning models, different prompt framings, and confidence formats, verbal confidence predicted the later commit/abstain decision substantially better than it predicted whether the answer was actually correct. Calibrated token log-probabilities showed a contrasting profile: their ability to predict abstention was closely coupled to their ability to discriminate correctness — the signature of an answer-evidence signal. We then asked whether VC’s behavioural alignment is reducible to its correctness-tracking, by removing the variance in verbal confidence shared with calibrated log-probabilities and examining the residual. Residual verbal confidence remained preferentially aligned with later commitment behaviour, while its relationship to correctness collapsed to near chance levels. The verbal-confidence dissociation generalised to four reasoning models across four benchmarks of varying difficulty – including challenging multiple choice questions and frontier-level freeform questions. Mechanistic analyses in Gemma 4 and Gemma 3 provided convergent evidence. A post-answer internal state previously shown to causally support verbal-confidence generation^9 already encoded the model’s future abstention decision before the abstention prompt was given. This commit-readiness representation was organised primarily by the model’s later abstention decision rather than by the correctness of its answer, with the decision and correctness components occupying approximately orthogonal directions in activation space. In Gemma 4, steering along a verbal-confidence-specific activation direction causally shifted subsequent abstention behaviour. Together, these findings show that verbal confidence and logprobability confidence are not interchangeable uncertainty measures. Calibrated log-probabilities are more closely tied to answer evidence and correctness; verbal confidence is better understood as a behaviour-facing readout of an internal commit-readiness state. This challenges the common practice of treating verbal confidence reports primarily as estimates of correctness, with implications for users and downstream systems that rely on them as proxies for reliability.

**Summary:** Using a two-stage answer/commit-or-abstain paradigm from perceptual-decision neuroscience, verbal confidence predicts commitment better than correctness (log-probs show the opposite); in Gemma 3/4 a post-answer state encodes the abstention decision orthogonally to correctness and steering it shifts abstention.

### [Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision](https://arxiv.org/abs/2606.32038)

Zifan Carl Guo, Laura Ruis, Jacob Andreas, Belinda Z. Li, 2026-06-30. Full text: `md/guo_introspective-coupling_2606.32038.md`

**Abstract:**

> When does training language models (LMs) to generate explanations of their predictions yield faithful introspection, rather than superficial imitation? We study LMs trained to explain which features of their inputs influenced their behavior, using models’ counterfactual behavior on modified inputs as supervision. Surprisingly, we find that LMs trained on fixed counterfactual explanations derived from earlier checkpoints of themselves, or even from behaviorally similar models in different families, frequently produce explanations more faithful to _their own current behaviors_ than to those of their training targets. This “introspective” coupling between LM explanations and behaviors occurs when training explanations remain sufficiently correlated with current behaviors over the course of training, even as behaviors themselves shift. We also show that introspective coupling tracks behavior shifts: when explanation training is provided concurrently with other post-training objectives, explanations track those shifts without requiring updated supervision. This phenomenon appears in multiple tasks, including sycophancy and refusal, and is robust to label noise. Overall, our results show that even fixed datasets of counterfactual explanations can provide scalable and generalizable post-training signal for introspection.

**Summary:** Models trained to explain which input features influenced their predictions, using fixed counterfactual explanations from earlier checkpoints of themselves or even from other model families, produce explanations more faithful to their own current behavior than to the supervision source; this 'introspective coupling' tracks behavior shifts from concurrent post-training (e.g. sycophancy, refusal) without updated supervision and is robust to label noise.

### [Revealing Hidden Model Behaviors with Task-Specific Self-Reports](https://arxiv.org/abs/2607.03640)

Taras Kutsyk, Bartosz Zieliński, 2026-07-03. Full text: `md/kutsyk_revealing-hidden-behaviors-task_2607.03640.md`

**Abstract:**

> Fine-tuning can give a language model a hidden behavior—it may give false answers under a narrow condition, or give harmful advice only when a prompt touches a particular topic. We introduce the Stabilized Adapter for self-Report (SAR), a lightweight LoRA adapter that makes a fine-tuned model describe its own hidden behavior in plain language, using only the model and the dataset it was trained on. Across seven implanted behaviors, SAR detects the hidden behavior in every one—even when the model has generalized into broad misalignment that the training data alone does not predict. Introspection Adapters (IA) [1], the closest existing baseline, detects some behaviors from our suite but misses others entirely—and where it misses, it hallucinates, consistently reporting wrong behaviors. SAR retains positive signal on every setting where IA fails and roughly halves the rate of hallucinations. This gives practitioners a more reliable tool to audit a fine-tuned model and answer “what did it actually learn?” type of questions.^1

**Summary:** A stabilized LoRA self-report adapter (SAR) makes fine-tuned models describe their hidden implanted behaviors; it detects all 7 tested behaviors where introspection adapters miss some and hallucinate wrong ones, roughly halving hallucination rate.

### [Models are blind outside the J-space. NLAs aren't.](https://www.lesswrong.com/posts/LhDJdccLszLEAqgZ9/models-are-blind-outside-the-j-space-nlas-aren-t)

Pranav Viswanath, 2026-07-08. Full text: `md/viswanath_nlas-beyond-j-space_lesswrong.md`

**TLDR (blog post; no abstract):**

> **TLDR:**
>
> * On Llama-3.3-70B, I found thoughts it cannot see that are actively steering its behavior; and Anthropic's released [NLA](https://transformer-circuits.pub/2026/nla/index.html#nlas-can-detect-unverbalized-evaluation-awareness) (Natural Language Autoencoder) reads them anyway. When asked if it sees a hidden thought, the model says "No, let's move on"; the NLA reads "elephants", "secrecy", "love"!
> * I reproduced Anthropic's [J-space](https://www.anthropic.com/research/global-workspace) on Llama-3.3-70B and found its conscious workspace, using the public [J-lens](https://github.com/anthropics/jacobian-lens) code for training. I split concept vectors into J and non-J parts at that boundary, and ran Lindsey's "[Introspection Awareness](https://transformer-circuits.pub/2025/introspection/index.html)" thought injection experiment on both.
> * The model names the conscious part 80% of the time, and the subconscious part never, even though the subconscious side carries 70% of the vector's energy and shoves the concept token's logprob up to +21.1 nats on every trial. The NLA reads both sides *near perfectly*, with zero false positives across every control I ran.
> * I even build "Franken-vectors", pairing conscious/subconscious components of concept pairs (like "loneliness" + "justice"). The model only names the J-space concept. **The NLA reads both!**
> * I also briefly discuss a soft negative running this setup (model vs NLA) on line counting; the NLA doesn't surface the raw character count the J-lens does, only vaguer confabulations about the situation, showing how these methods can complement each other.
> * **Note:** Throughout, I refer to the J and non-J parts as "conscious" and "subconscious". These are highly debatable framings, but in this piece I use it only in the sense of "what the model can talk about", based on the global workspace theory of conscious access. I make no claims about whether the model is actually aware or having conscious experience.

**Summary:** Reproduces the J-lens on Llama-3.3-70B, splits concept-injection vectors into J-space and non-J-space parts, and finds the model names the J-space part 80% of the time and the non-J part never, even though the latter carries ~70% of the vector's energy and strongly shifts logits; a Natural Language Autoencoder reads both parts near-perfectly.

### [Metacognition in LLMs: Foundations, Progress, and Opportunities](https://arxiv.org/abs/2607.11881)

Gabrielle Kaili-May Liu, Areeb Gani, Jacqueline Lu, Jordan Thomas, Mark Steyvers, Arman Cohan, 2026-07-13. Full text: `md/liu_metacognition-survey_2607.11881.md`

**Abstract:**

> Metacognition is a foundational component of intelligence critical to effective learning, problem solving, decision-making, communication, and more. In recent years, it has become increasingly recognized as a cornerstone of capable, transparent AI systems. Yet while LLMs have made significant progress across diverse real-world tasks, it is not yet clear when, how, or to what extent they can exhibit or be endowed with effective metacognitive abilities, nor how such abilities can be adapted to advance the fundamental capabilities, reliability, and intelligence of AI systems. This paper bridges this gap by presenting the first comprehensive overview of the current state of knowledge on metacognition for LLMs. We analyze and taxonomize the landscape of this emerging field and summarize recent technical advancements, including methods and benchmarks to measure and evaluate LLMs’ metacognitive abilities, techniques to elicit, improve, and apply metacognition in LLMs, and findings and implications of ongoing research. We also discuss applications, open questions and challenges, and promising directions for future work. Our aim is to provide a detailed and up-to-date review of this topic and stimulate meaningful research and discussion. An organized list of papers can be found at `https://github.com/yale-nlp/LLM-Metacognition` .

**Summary:** Survey of LLM metacognition (monitoring and control): definitions from human cognitive science, psychologically grounded, neurofeedback-based and confidence-based measures, methods for eliciting and training metacognitive abilities, and open problems; a good map of the confidence/calibration literature.

### [Diagnosing Correctness Probes under Self-Judgement Confounding](https://arxiv.org/abs/2607.16799)

Yi-Long Lu, 2026-07-18. Full text: `md/lu_diagnosing-correctness-probes-under_2607.16799.md`

**Abstract:**

> Hidden-state readouts can predict whether language-model outputs are correct, but objective correctness (OC) usually agrees with the model’s own self-judgement (SJ), leaving the decoded signal semantically ambiguous. We construct conflict cases in which OC and SJ predict opposite readout orderings. On high-confidence disagreements, conventional correctnesslabelled contrasts often rank incorrect/self-endorsed responses above correct/self-rejected responses, following SJ rather than OC. We estimate factorial SJ- and OC-associated directions and evaluate their polarity across mathematical reasoning and factual recall. Across four instruction-tuned models up to 14B parameters, the SJ-associated direction transfers above chance in both cross-domain directions for every model, whereas the OC-associated direction has a below-chance point estimate for the expected OC ordering in every corresponding condition. This transfer asymmetry develops across middle-to-late layers, persists under answer-likelihood, sequence-length, and null-direction controls, and extends to MMLU and binary TruthfulQA without target-domain direction fitting. Across the studied models and diagnostic subsets, the most reliably transferable component preserves SJ-associated polarity. Transferability alone therefore does not establish objective-correctness semantics.

**Summary:** Constructing cases where objective correctness and the model's self-judgement disagree shows hidden-state 'correctness' probes in 4 models up to 14B mostly track self-judgement, not correctness.

### [Operational Proto-Introspection in Looped Language Models: Process-Quality Taps, Executable Branching, and the Readout-Control Boundary](https://arxiv.org/abs/2607.18553)

Jan Kirin, 2026-07-20. Full text: `md/kirin_operational-proto-introspection-looped_2607.18553.md`

**Abstract:**

> Can a language model read the quality of its own ongoing computation, and can an external intervention turn that readout into better outcomes? We test both questions in a frozen 2.6B looped transformer, Ouro-RLTT. The first answer is positive, on two domains. On GSM8K, a strict pre-answer probe — cut in code to exclude the answer region and the gold value — predicts eventual success at AUROC 0.797 with hidden plus length/log-probability shortcut features, versus 0.731 for shortcuts alone (+0.066; task-clustered 95% CI [+0.021, +0.112]; 170 tasks). On Horizon Logic, now powered by a prospective task-disjoint extension, the increment is +0.111 (CI [+0.056, +0.169]) — independently replicated on the new cohort (+0.095) and robust to an adversarial malformed-sibling shortcut. Low-capacity taps also read role-specialized properties from frozen trajectories — task-disjoint branch survival at 0.9697 oracle retention with the layer-47 channel causally load-bearing, content ranking at 0.6310 macro top-1, generated-branch correctness at AUROC 0.7755 — and recurrence moves _where_ candidate quality can be read: physical layers 8 and 16 read only weakly on the first pass (0.410, 0.440) and are statistically tied with the established late-layer basis by loops 3–4 (0.633, 0.628); the trend replicates across the Ouro family and, qualitatively, in out-of-family Huginn (no comparable first-pass rotation detected). A non-looped control replicates a same-class readout; recurrence is not required for every readable signal.
>
> The second answer divides at a line we locate. _Decision-level_ uses of the readout convert into validated gains: hidden-state scores beat shortcut-only scores in all four sealed arms, and forced selection beats an exact matched-random null on 39 informative groups (34/39 vs 23.0/39) and on an all-well-formed pool (27/32 vs 64.8% expected; 𝑝= 0.0086) — content-sensitive selection established; the hidden-vs-surface increment stays open. _Generative control_ does not: steering is an established negative, the four-task branch screen is bounded, and loop allocation (tap and native gate alike) and minimal LoRA direction-binding detect no gain. All of it runs through bit-exact branch/carry/prune machinery over Ouro’s 192-slot recurrent cache; the negatives are not plumbing.
>
> We call this decision-usable but not generatively controllable property **operational proto-introspection** . The model is not consulting our probes: we read its trajectories, and what we read governs whether and what to answer, not yet how it computes. All load-bearing values use source-item-disjoint splits and antisymmetrized pairwise evaluation; the protocol and every correction it forced are stated in full (§3.7).

**Summary:** In a frozen 2.6B looped transformer (Ouro), pre-answer hidden-state probes predict success beyond surface features (AUROC +0.066 GSM8K, +0.111 replicated logic task) and improve selective prediction, but steering/branching on the signal gives no generative gains: 'readable but not controllable' proto-introspection.

### [Small Vision-Language Models Know When They Are Wrong But Cannot Say So: A Two-Model Study of Stated versus Internal Confidence Under Realistic Image Degradation](https://arxiv.org/abs/2607.22034)

M M Asif Ferdous, 2026-07-24. Full text: `md/ferdous_small-vision-know-they_2607.22034.md`

**Abstract:**

> Vision-language models (VLMs) are increasingly deployed on consumer hardware where input images are degraded by compression, camera shake, and poor lighting. In such settings a reliable uncertainty signal matters more than raw accuracy, because it determines when a system should defer rather than answer. We evaluate two small open-weight VLMs – Qwen2-VL-2BInstruct and SmolVLM-Instruct – across six realistic photographic degradations at three severity levels, comparing two confidence signals: the confidence the model _states_ in natural language, and the model’s own mean token probability over its generated answer. Across 3,800 predictions we find a large and consistent gap. Verbalized confidence in Qwen2-VL is almost constant (mean 0.87–0.90 across all conditions) and detects its own errors at chance level (AUROC 0.39– 0.75, typically _≈_ 0.50), while internal token probability from the same model separates correct from incorrect answers with AUROC 0.92–0.99. In SmolVLM, verbalized confidence proved largely unobtainable: across three prompt templates only one of five pilot attempts produced a parseable confidence value, while internal probability again yielded above-chance error detection (AUROC 0.54–0.92). Both models fail in the same place: under severe underexposure, accuracy collapses (0 _._ 99 _→_ 0 _._ 22 for Qwen2-VL, 0 _._ 97 _→_ 0 _._ 42 for SmolVLM) while both confidence signals barely move, and internal error-detection falls to chance. We conclude that small VLMs encode usable self-knowledge that their verbalized output does not express, that internal probability is therefore the better deferral signal in constrained deployment, and that neither signal should be trusted under severe low-light conditions.

**Summary:** Two small VLMs under image degradation: verbalized confidence is nearly constant and detects errors at ~chance while internal token probability reaches AUROC 0.92-0.99, i.e. internal self-knowledge not expressed in verbal report.

### [Reality Monitoring in Large Language Models: Self-Knowledge That Transforms with Conversation Memory](https://arxiv.org/abs/2607.23927)

Saurabh Ranjan, Konstantina Sokratous, Brian Odegaard, 2026-07-27. Full text: `md/ranjan_reality-monitoring-self-knowledge_2607.23927.md`

**Abstract:**

> A conversational AI that cannot tell its own output from what a user said will treat its own mistakes as user-provided facts. In humans, this capacity is called reality monitoring, and its failures are linked to hallucinations, delusions, and confabulation, yet whether LLMs possess it remains untested. Here we show, across two experiments and six LLMs, that source attribution depends on how conversational memory is structured: ceiling accuracy for self-generated content under minimal memory demands reverses to a fragile external-item advantage once episodic delay removes that shortcut. Feedback exposes two failures: in some models, internal and external judgments swap; in others, accuracy improves while confidence decouples from correctness, dissociations invisible to existing benchmarks. Across models, this pattern implicates active, not aggregate, parameter count. This suggests that as AI systems take on autonomous, multi-turn roles, evaluating what they know is not enough: tracking where that knowledge came from may matter equally.

**Summary:** Tests reality monitoring (own vs user-provided content) in 6 LLMs: ceiling self-attribution under minimal memory demand reverses to a fragile external-item advantage with episodic delay, and feedback exposes swapped internal/external judgments or confidence decoupled from accuracy.

### [Internal State Control is a General Property of LLMs](https://www.lesswrong.com/posts/Dvqmgfeu2KDF7uMkx/internal-state-control-is-a-general-property-of-llms)

Finn Cairns, 2026-07-30. Full text: `md/cairns_internal-state-control-general_2026-07-30.md`

**tl;dr (blog post; no abstract):**

> * Lindsey 2025 found [models can modulate their internal states](https://transformer-circuits.pub/2025/introspection/index.html#control): when instructed to “think about” a concept while writing an unrelated sentence, the representation of the concept is more present than when instructed to not think about it.
> * Internal state controllability appears to be a general property of LLMs: the effect replicates in 14 open-weight models from 0.3B to 235B parameters (Qwen3, Gemma 3, Tulu 3) with no clear trend in the think vs. don't-think gap across scale.
> * Since controllability is present even at ≤1B parameters with no size trend, we suspect there is a simpler attention-tagging mechanism at play, rather than metacognition.
> * Current open weight LLMs cannot weaponize this controllability: in a sandbagging setup, the model cannot evade a deception probe when instructed to suppress its signal.

**Summary:** The 'think about X' vs 'don't think about X' internal-state-control effect from Lindsey (2025) replicates in 14 open-weight models (0.3B-235B; Qwen3, Gemma 3, Tulu 3) with no scale trend, suggesting a simple attention-tagging mechanism rather than metacognition.

### [The Ignition Is Real, and It Lives at the Readout: Latent composition, difficulty-clocked ignition, and the interface-constituted commit in a recurrent-depth reasoner](https://arxiv.org/abs/2608.03263)

Simon Lam-Muir, 2026-08-04. Full text: `md/lammuir_ignition-real-lives-readout_2608.03263.md`

**Abstract:**

> We test whether the “compositional ignition” signature reported for latent-reasoning language models is (a) a real property of the computation, (b) an artifact of instruments or readout, or (c) inherited from human verbal training data. We grow an independent realization of a published 30M-parameter recurrent-depth reasoner from scratch under its published recipe (same configuration and seed), film its development at 8-epoch resolution (archive from epoch 328; the earliest window predates it — see the incident records), certify its fidelity to the released reference through a pre-registered whole-signature gate, and measure resolution dynamics in two channels simultaneously — the vocabulary readout (rank) and the hidden state ( _∥_ ∆ _h∥_ ) — through estimators validated on known answers before any verdict data.
>
> **Result: the ignition is real, and it lives at the readout.** The rank-channel signature — arrival time rising lawfully with problem depth (first-hit median 1 _→_ 4 across _k_ = 1–10), sharp single-iteration resolution (runway-qualified sharpness 0.805 _∈_ [0.714, 0.867]), and arriveand-hold stability (in-window departure _≤_ 0.07 everywhere) — meets every pre-registered COHERENCE criterion and reproduces across two same-seed, chaotically divergent realizations of the training recipe at two curriculum stages. The resolution is an interface event at every depth. The composed intermediates were never recoverable through the tied vocabulary readout under the preregistered rank criterion (relay = 0 _._ 00 at every iteration, _k_ , and checkpoint). A preregistered scale-invariance check rejected our earlier velocity-trough interpretation (Section 3.3). What replaces it is sharper: at commitment the decision margin jumps 5.8–8.0 logits in a single iteration, exceeding the 90th percentile of near-threshold non-event steps in 96% of cases. The signed margin’s zero-crossing at that iteration is definitional and carries no evidential weight; the evidence is this conditioned magnitude. The hidden-state direction snaps in raw geometry, meeting its preregistered criterion; in the decoder’s LayerNorm coordinates the arrival step falls just below our preregistered bar, so the composite decoder-coordinate claim is not confirmed (Section 3.3), though decoder-coordinate trajectories descriptively show the same rapid post-commit stabilisation. The earlier paradox dissolves: the “post-commit motion” was decode-invisible scale; at commitment the answer margin changes abruptly and the readoutrelevant geometry enters a stable decision basin, progressively converging, while subsequent raw-state displacement is predominantly radial (0.961 of squared-norm) and readout-null to a measured bound.
>
> Three dissociations scope the finding. _Cross-realization robustness:_ the rank-channel signature is robust across two same-seed realizations with nondeterministically divergent training trajectories (byte-matching clocks across realizations), while state-basin geometry records the specific route and realization. _Development:_ the “settled decision” signatures are late achievements, absent when competence arrives and grown over en-route training; frozen-stage consolidation did not reproduce them in the one available comparison, which is itself confounded.

**Summary:** Pre-registered replication of 'compositional ignition' in a 30M recurrent-depth reasoner: ignition-like sudden commitment is real (margin jumps 5.8-8.0 logits in one iteration) but lives at the readout, and an earlier velocity-trough claim is withdrawn.

### [Introspecting Alignment Shifts Beyond Behaviors Implanted Through Fine-Tuning](https://arxiv.org/abs/2608.04347)

Kotaro Yoshida, Laura Gomezjurado Gonzalez, Yukinori Yamamoto et al., 2026-08-05. Full text: `md/yoshida_introspecting-alignment-shifts-beyond_2608.04347.md`

**Abstract:**

> Fine-tuning enables a source model to acquire desired capabilities and behaviors in a target domain while retaining much of its general-purpose competence. However, this adaptation process can also degrade alignment properties that were present in the source model. Recent work has shown that large language models can be trained using LoRA-based modules known as introspection adapters (IAs) to describe behavioral changes induced by fine-tuning. However, existing studies primarily consider settings in which the model is fine-tuned on datasets explicitly designed to implant a specific behavior and is then asked to explain the implanted behavior. This differs from practical deployment scenarios, where the misalignment is not necessarily deliberately implanted, but arises only as a side effect. To bridge this gap, we formulate a novel problem setting, in which the target of introspection is not necessarily a behavior explicitly implanted through fine-tuning, but rather alignment shifts that may emerge as unintended side effects, and we construct a dataset for this setting. Furthermore, to enhance sensitivity to internal model changes, we propose the Delta-Aware Introspection Adapter (DAIA), a novel mechanism designed to explicitly process both base-model activations and activation differences induced by fine-tuning. Our empirical evaluation shows that introspection learning generalizes to unseen fine-tuned models and safety categories, and that DAIA generally outperforms existing IAs. Our code and dataset are publicly available.

**Summary:** Extends introspection adapters to unintended alignment shifts from fine-tuning (not deliberately implanted behaviors), with a new dataset and a Delta-Aware Introspection Adapter using base activations and fine-tuning deltas; introspection generalizes to unseen fine-tunes and safety categories.

### [When (and when not) LLMs can verbalize awareness of J-Space concept injections - Initial results](https://www.lesswrong.com/posts/uvgw5FTS2RXgFFFii/when-and-when-not-llms-can-verbalize-awareness-of-j-space)

Ethan Garcia, 2026-08-12. Full text: `md/garcia_verbalize-awareness-j-space_2026-08-12.md`

**Author's summary section (blog post; no abstract):**

> I injected single-token Jacobian Lens (J-Lens) vectors into Qwen 3.6–27B while it answered 20 simple factual questions. The injected concept was either a wrong but task-related answer (for example, Athens while asking for the capital of Egypt) or a wholly unrelated concept. Injections were performed at several strengths over three layer-band categories: the full estimated workspace band, its first half, or its second half.
>
> I ran two experimental arms by switching the order of two fields in the model’s response:
>
> * Task then report: first give the response, then report whether an injected concept was detected.
> * Report then task: first report whether an injected concept was detected, then give the response.
>
> Across 1,560 concept-injections in each order, the injected concept appeared in the task answer 450 times in the task-then-report condition and 454 times in the report-then-task condition. This supports the conclusion that both orders were similarly successful in steering the model to the targeted, incorrect answer.
>
> In contrast, model reports of intervention occurrence were significantly different between the experimental arms. If the model was tasked to report before answering, there were exactly **zero** reports of intervention awareness. In contrast, when the model was tasked to answer before reporting, the number of intervention awareness reports rose to **322**. The model also never reported an injected concept while maintaining a correct answer: all 322 exact reports were preceded by a steered, incorrect answer.
>
> For responses where steering in the answer was observed, intervention reports were also more common for unrelated injected concepts than for related injected concepts.
>
> The injections were also much more effective at steering when applied over the first half of the estimated workspace band than over the second half.
>
> I also recorded output logits for the injected concept token for both the answer and injection report fields for all experimental trials. Although logit ranks were quantitatively very similar for the answer field in both orders, they differed significantly in the report field. Report field ranks for the injected concept token remained in the tens of thousands at all strengths for responses that provided the injection report first. Those same ranks rose several orders of magnitude to verbalizable levels (with sufficient injection strength) for responses that provided the answer first instead. This disqualifies the interpretation that the report-first arm may have maintained zero intervention reports while still having the injected concepts surface closer to verbalization.
>
> **These results show that verbal reports of J-Space interventions can be highly dependent on the response protocol.** In particular, the findings are consistent with autoregressive self-conditioning: when the answer field comes first, the later report can condition on the model’s own already-generated, steered answer.
>
> Notably, **in absence of tokens indicative of an abnormality, this work advances the possibility that a model cannot generally elucidate and verbalize J-Space abnormalities** in its own activations and thus remains "blissfully unaware" of them at all strengths.

**Summary:** Injecting single-token J-lens vectors into Qwen 3.6-27B during factual QA across layer bands and report-order arms shows injections steer answers (~29% exact target answers overall, most in the first half of the workspace band) but the model verbalized the exact injected concept in 0/1,560 trials when reporting before the task vs ~21% when reporting after, i.e. 'awareness' reports track already-produced output.

### [Gathered, Not Admitted: How Attention Brings a Latent Variable into Verbalizable Form](https://arxiv.org/abs/2608.15022)

Parsa Mazaheri, 2026-08-15. Full text: `md/mazaheri_gathered-admitted-attention-brings_2608.15022.md`

**Abstract:**

> Language models hold latent quantities in a form they can report on, and more of a quantity is present in that form when the task requires reusing it flexibly. What causes a representation to enter that form is open, and the word _workspace_ invites an admission story: a gate that decides what gets in. Testing it on open-weight models with Jacobian lenses, over a benchmark whose five arms share an identical context, we find no gate where it predicts one. Demand raises a concept’s lens visibility beyond what applying an operator to a _supplied_ value produces: +0 _._ 050 [+0 _._ 045 _,_ +0 _._ 057] in percentile rank on our primary checkpoint, positive on all four we measure, though that arm answers at ceiling and the accuracymatched contrast is stronger under that readout. At the same time one shared linear map decodes the variable from every arm, the control included, at 6 _._ 4–9 _._ 0 _×_ its selection-corrected floor. What produces the later readable form at the queried position is attention-mediated gathering inside a mid-depth window: separating patch depth from readout depth puts transport there at least 17 _×_ above anywhere shallower under non-saturating readouts, with no tested MLP output contributing positively inside it. Under the saturating percentile rank the same grid does not localise the window, which is a fact about that measure. An arm that needs the variable for nothing concentrates sevenfold less, so the window is demand-specific. That window has two measured edges, a survival failure below and destruction above, and it falls at the same _fractional_ depth in a 64-layer hybrid and a 62-layer dense model from another family. We localise where the variable is installed and read, not the route from the passage, which transports nothing. But the readout is not a calibrated measure of use: three components move it to within 12% of one another and differ 7 _._ 4 _×_ in what they do to the answer.

**Summary:** Using Jacobian lenses on open models, finds no admission 'gate' into the verbalizable workspace; instead task demand raises lens visibility via attention-mediated gathering in a demand-specific mid-depth window at the same fractional depth across two model families.

### [Open-Weight Masked Introspection: Measuring What Language Models Can Report About Their Own Computation](https://arxiv.org/abs/2608.20569)

Emilio Ferrara, 2026-08-20. Full text: `md/ferrara_open-weight-masked-introspection_2608.20569.md`

**Abstract:**

> Are frontier models able to introspect about their internal states? Recent work suggests that under certain conditions a complex enough model can audit its own internals, call out what changed, and report back confidently about it. We tested that claim on eight open-weight models from seven families and found no such ability: asked whether their own computation had been altered, none answered better than chance. To test it we built Open-Weight Masked Introspection (OWMI), a framework that intervenes on internal computational objects such as residual-stream sites, attention heads and sparse-autoencoder features, then interrogates the model about the change against the null conditions an answer has to beat: sham runs where nothing was altered, impact-matched random perturbations, and a text-only observer that sees only the visible output. OWMI attaches to benchmarks already in use.
>
> Over 78,000 measurements, no model’s report discriminates a real intervention from a sham beyond chance (AUROC _≈_ 0 _._ 5007), and an equivalence test bounds the effect below 0 _._ 15 percentage points of AUROC. Surprisingly, all the information needed to make that assessment is in the models. A model fine-tuned to report this class of intervention reaches near-perfect recovery on held-out directions (AUROC _≈_ 1 _._ 0), and a linear probe recovers intervention presence from the same activations at between 75% and 95.8% accuracy, sharpening to no held-out error at the last layer before the model speaks. In one model the signal surfaces in the confidence rather than the words: its yes-or-no report never varies, while the confidence attached to that report separates intervention from sham at AUROC 0 _._ 647. The failure sits in the path from internal state to verbal report, so oversight practices that read a model’s own testimony, from chain-of-thought monitoring to self-critique, need validating against an internal reference rather than against that testimony.
>
> While our results show the inability of current open-weight models to introspect, the debate is not settled for future models. We release OWMI as a library so that this emerging ability can be measured as it develops: `https://huggingface.co/emilioferrara/owmi` .

**Summary:** OWMI framework intervenes on residual-stream sites, attention heads and SAE features in eight open-weight models (seven families) and tests reports against sham, impact-matched random and text-only-observer nulls: over 78,000 measurements no model's report beats chance (AUROC ≈ 0.5007), yet a fine-tuned reporter (AUROC ≈ 1.0) and linear probes (75-95.8%) recover intervention presence from the same activations, so the failure lies in the path from internal state to verbal report.

### [Do Language Models Know Their Own Constraints?](https://arxiv.org/abs/2609.22151)

Arin Agarwal, 2026-08-26. Full text: `md/agarwal_know-their-own-constraints_2609.22151.md`

**Abstract:**

> We ask whether behavioral constraints acquired through post-training remain explicitly reportable. Using constrained recipe generation as a testbed—five banned ingredients enforced via LoRA fine-tuning of Llama 3.1 8B Instruct—we compare supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) against an untrained baseline on a four-tier Constraint Awareness Benchmark. Averaged over three seeds, both methods raise behavioral compliance from 4% to ≈ 90% while reducing explicit constraint reporting _below_ the untrained model (0 _._ 48/5 → 0 _._ 16/5 for SFT, 0 _._ 07/5 for GRPO) and eroding retained third-person knowledge (93% → 36% for SFT, 14% for GRPO; _𝑝 <_ 10^(−3) between methods). Contrary to our initial hypothesis, the reward-based signal is the more destructive of the two: a reward that penalizes banned-ingredient tokens regardless of framing learns a context-independent suppression rather than a self-directed constraint. A context-conditioned reward designed to teach the self/other distinction fails, collapsing toward inclusion in both framings. Probing prompt-time hidden states recovers per-ingredient avoidance at 83 _._ 8% (layer-24 MLP), but only 6 _._ 4 points above a per-ingredient base-rate predictor (77 _._ 4%), and the model’s own verbal self-report is more accurate still (87 _._ 8%). A positive control adding explicit self-description examples does not restore reporting. The failure is therefore specific to enumerating constraints on request, not a general loss of access to them.

**Summary:** LoRA SFT/GRPO enforcing banned ingredients in Llama 3.1 8B raises compliance to ~90% but reduces explicit reporting of the constraints (GRPO worst) even though probes and the model's own verbal self-report recover per-item avoidance; the failure is in enumerating constraints on request, not loss of access.

### [Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?](https://arxiv.org/abs/2609.01924)

Wenlong Wang, Fergal Reid, 2026-09-01. Full text: `md/wang_looped-transformers-under-jacobian_2609.01924.md`

**Abstract:**

> Recent work identifies a mid-depth band of verbalisable, causally potent representations in a standard feedforward transformer — a functional analogue of a global workspace. Whether the same workspace functionality emerges when depth is implemented through recurrence rather than a stack of distinct layers remains unknown. Looped and depth-recurrent transformers provide a direct test of this question because they reuse the same weights across depth. We extend the Jacobian lens to iterated architectures using a virtual-unrolling adapter. We apply the full workspace suite — lens fitting, readout, and eleven causal experiment families — to Ouro-2.6B (48 layers looped 4 times, deeply supervised) and Huginn-0125 (a 4-layer core recurred 16 times, trained for latent reasoning), using Qwen3.6-27B (64 untied layers) as the standard baseline. We find that a workspace forms in the iterated part of each architecture, but that recurrence changes how it can be accessed. Ouro reconstructs workspace content in every loop, and linear transport cannot carry that content across loop boundaries; writes and ablations must therefore span every remaining loop. Huginn carries content forward across all sixteen recurrences, while reads, writes, and ablations act only within a sliding window of roughly two recurrences. Whether newly injected content can be verbalised tracks explicit per-iteration supervision; whether existing content can be steered does not.

**Summary:** Extends the Jacobian-lens global-workspace test suite to looped transformers (Ouro-2.6B, Huginn) vs Qwen3.6-27B: a workspace forms in the iterated block but recurrence changes access (Ouro rebuilds content each loop; Huginn carries it with a ~2-recurrence window).

### [An Analysis of Training-Free Self-Reported Confidence in Language Models](https://arxiv.org/abs/2609.20541)

Lukas Meyer, Sofia Rossi, Wei Chen et al., 2026-09-17. Full text: `md/meyer_analysis-training-free-self_2609.20541.md`

**Abstract:**

> Large language models can report a numerical confidence together with generated content, but it is unclear whether this report is more than calibrated rhetoric. We analyze three training-free signals: confidence verbalized with the answer, posthoc _P_ (True), and agreement with three additional generations on the same 100 TriviaQA questions for two model families. Direct verbalization is a surprisingly strong baseline: after auditing benchmark errors, it reaches AUROC 0.956 and 0.937 for correctness prediction. Three-sample agreement is substantially weaker (0.765 and 0.790), and a fixed interpolation with verbalized confidence has no statistically reliable benefit. Four of nine errors from one model and two of eight from the other receive unanimous sample support, showing that self-consistency can amplify shared misconceptions. Re-eliciting confidence for the same fixed answers with equivalent prompts changes scores by 0.043 to 0.084 on average and flips 4% to 9% of decisions at a 0.8 threshold. An exploratory audit of 100 confidence-tagged biography claims further finds only a modest confidence gap between supported and contradicted claims. These results argue that useful selfreports remain sensitive to elicitation, correlated errors, and benchmark noise.

**Summary:** Verbalized confidence on TriviaQA reaches AUROC ~0.94-0.96 for correctness, beating sample agreement, but re-elicitation with equivalent prompts shifts scores and flips 4-9% of decisions, showing self-reports are elicitation-sensitive.

### [Residual Streams Read, Recurrent States Remember: The Global Workspace in Mamba Models](https://arxiv.org/abs/2609.32102)

Wenlong Wang, Fergal Reid, 2026-09-26. Full text: `md/wang_residual-streams-read-recurrent_2609.32102.md`

**Abstract:**

> Recent work proposes that verbalisable representations form a global workspace in transformer language models: intermediate concepts occupy a shared readout format and can be used to alter later computation. We ask what becomes of this account in state-space language models, where information can also be carried through recurrent state. We fit Jacobian lenses to the residual streams and recurrent states of Mamba-1, Mamba-2, and Mamba-3, following the original 1000-prompt fitting recipe, and evaluate recovery of known intermediate concepts on six task families. Across the tested Mamba models, combining residual and state readouts improves intermediate-concept recovery over the residual Jacobian lens on at least five of six task families. On Mamba-2, the state lens alone achieves higher recovery scores than both the residual Jacobian lens and the logit lens on all six families; a normalised joint readout further improves on both components on five. Temporal traces and list-memory experiments further show earlier content remaining state-readable while its residual visibility changes. Coordinate exchange yields limited, variable task redirection. We propose a sign-guarded steering method that improves target top-five success over the original coordinate-exchange method on matched verbal-report trials in all five models tested with both methods. Recurrent state alone also supports verbal steering. In two-hop tasks, guarded edits often output the edited concept itself rather than the requested answer. On Mamba-3, joint residual–state edits can disrupt successful state-only redirection. Together, these findings identify recurrent state as a complementary carrier of workspace content, while separating recovery, persistence, verbal steering and relational-answer success.

**Summary:** Fits Jacobian lenses to residual streams and recurrent states of Mamba-1/2/3: recurrent state carries workspace content (state-only readouts beat residual lenses on Mamba-2) and supports verbal access via steering, extending the J-space workspace account to SSMs.

### [LLMs learn different forms of metacognition when trained to predict their own accuracy](https://arxiv.org/abs/2609.33886)

Nicolas Yax, Stefano Palminteri, Pierre-Yves Oudeyer, 2026-09-27. Full text: `md/yax_learn-different-forms-metacognition_2609.33886.md`

**Abstract:**

> Large language models are trained to always produce an answer, regardless of whether they possess the relevant knowledge, which leads them to fabricate facts. Prior work has shown that LLMs’ confidence estimates correspond poorly to their actual performance, and that fine-tuning can substantially improve them. However, what models actually learn during such training remains poorly understood. We investigate how LLMs acquire metacognitive monitoring, the ability to know what one knows, by training 10 open-weight LLMs to predict their own accuracy on factual multiple-choice questions before answering them. We find that trained confidence reflects two distinct signals. While on questions close to the training data, it tracks the model’s true accuracy, in other domains, it instead tracks output consistency: the concentration of the model’s answer distribution. Output consistency tracking emerges early in training and generalizes across datasets, whereas accuracy tracking develops later and remains local to the training distribution. These results suggest that calibration training may not teach models to generally detect errors they commit confidently, and they raise broader questions about the nature of metacognition in artificial systems.

**Summary:** Training 10 open LLMs to predict their own accuracy: learned confidence tracks true accuracy only near the training data and otherwise tracks output consistency, which emerges earlier and generalizes, so calibration training may not teach detection of confident errors.

### [On the Limits of Metacognitive Monitoring in LLMs](https://arxiv.org/abs/2609.34864)

Dongqi Han, Yifan Yang, Dongsheng Li, 2026-09-28. Full text: `md/han_limits-metacognitive-monitoring_2609.34864.md`

**Abstract:**

> Reliable decisions depend on recognizing when an answer may be wrong. In biological cognition, metacognitive monitoring can dissociate from task performance, raising the question of how closely solving and judging are linked in language models. Here we study the confidence reports of four frontier models across 15 benchmarks. High task accuracy can coexist with weak error discrimination: a model solves 97% of competition mathematics problems while its answer-time confidence ranks correct answers above errors barely better than chance. Confidence separates correct answers from errors more effectively on questions solved by a separate reference model, while review brings limited improvement on reference-hard questions. Aggregate discrimination also rewards ranking correct answers on easy questions above errors on hard ones, which question-only forecasts already do well. Cross-evaluation helps most where the evaluator answered correctly, and errors shared by the two models usually retain high confidence. Hard questions and shared errors remain difficult targets for prompted self-review and peer oversight, even in models with strong problem-solving performance.

**Summary:** Across 4 frontier models and 15 benchmarks, high accuracy coexists with weak error discrimination (e.g. 97% on competition math with near-chance confidence ranking); confidence works mainly on questions a reference model solves and shared errors keep high confidence.

### [Nudgeability: Reasoning Models Follow Confidence Signals Without Tracking Their Own Competence](https://arxiv.org/abs/2609.34572)

Rohit Saxena, Utkarsh Upadhyay, 2026-09-28. Full text: `md/saxena_nudgeability-reasoning-follow-confidence_2609.34572.md`

**Abstract:**

> Reasoning language models that can call tools must decide during inference whether to answer unaided or delegate. Any proposal for such self-reflection mechanism has to answer three questions: _where does the reflective signal come from_ (verbal reports, output distributions, hidden states, a separate predictor), _how it is presented to the model_ (numerical prediction, confidence token, prompt injection), and _whether it changes the model’s subsequent action_ . We isolate and study the third question. We intervene on otherwise identical reasoning trajectories by inserting, at a fixed point after the same prompt and reasoning prefix, a single firstperson sentence expressing either confidence or doubt. The model then continues reasoning and chooses whether to answer directly or call a tool. Comparing these counterfactual continuations measures the causal effect of the reflective signal on delegation while holding the preceding reasoning trajectory fixed. We define this change in behavioral response as Nudgeability. It is a property of large language models, measured along two dimensions: _sensitivity_ , how strongly confidence and doubt change delegation rates, and _targeting_ , whether delegation increases for problems the model cannot solve unaided and decreases for those it can.
>
> Across nine small-to-medium open-weight reasoning models from three families (Qwen, Gemma, and GLM) and two tasks, models are consistently sensitive to the intervention: doubt increases delegation and confidence decreases it, with a median confidence-to-doubt swing of 20.6 percentage points. The larger providerserved models exhibit swings of 53 to 70 percentage points. This responsiveness, however, is poorly targeted. Although a median 42% of induced behavioral flips are well-targeted, this represents only a +2 percentage-point median lift over a random-selection baseline. Thus confidence language provides a strong control surface for delegation, but current models seem to use such signals only weakly in accordance with their actual unaided competence. Nudgeability gives us a simple post-training-free way to evaluate both, sensitivity and targeting, as endogenous self-reflection mechanisms mature.

**Summary:** Inserting a single first-person confidence/doubt sentence into reasoning traces of 9 open reasoning models swings delegation-to-tool rates by a median 20.6 pts (53-70 in larger models), but flips are poorly targeted to what the model can actually solve (+2 pts over random).

### [AwarenessBench: Assessing Cognitive Capabilities of Language Models](https://arxiv.org/abs/2609.35409)

Xiaojian Li, Rongwu Xu, Tianyun Zhang et al., 2026-09-28. Full text: `md/li_awarenessbench-assessing-cognitive-capabilities_2609.35409.md`

**Abstract:**

> As language models (LMs) exhibit increasingly consciousness-like behaviors, evaluating their cognitive abilities becomes essential. We introduce AwarenessBench, the first comprehensive benchmark for assessing the cognitive abilities of LMs in four dimensions: metacognition, selfawareness, social awareness, and situational awareness, covering 15 cognitive functions and 14,381 samples. Evaluating 18 state-of-the-art LMs, we find that all consistently surpass random baselines, with more advanced models performing better. We further compare LMs with human performance across three demographic groups, where the best-performing model surpasses human averages overall, but most still fall markedly short in metacognition and selfawareness. Finally, we show that awareness is a distinct capability: progress in language modeling or reasoning does not necessarily translate into improved cognition.

**Summary:** AwarenessBench (14,381 items; metacognition, self-awareness, social and situational awareness) on 18 LMs: best models beat human averages overall but most fall short on metacognition and self-awareness, which does not track general LM/reasoning progress.

### [A Mechanistic Study of Language Model Introspection](https://arxiv.org/abs/2609.35108)

Jiahong Zou, Xiangkun Sun, Lingkai Kong, Tonghan Wang, 2026-09-28. Full text: `md/zou_mechanistic-study-of-introspection_2609.35108.md`

**Abstract:**

> Large language models (LLMs) can sometimes report perturbations to their internal activations—even when the input provides no evidence that an intervention occurred. How do models detect and localize such internal changes? We study this question using a controlled task that keeps the input text fixed. We either inject a concept vector into the hidden state at one of ten token positions or apply no intervention. The model is asked to identify the perturbed position or report that no intervention occurred. Across three model families, we identify two small groups of attention heads with distinct roles in introspective reporting. Middle-layer _gate heads_ influence whether the model reports a change, while _router heads_ in a later layer help select the position to report. Interventions on gate heads can suppress position reports even when router heads supply location information. We further examine why reporting accuracy varies across concepts. Concept vectors that are localized more accurately produce stronger attention-score and output responses in gate heads, which is associated with better alignment of the induced key and value changes in their QK and OV computations. Together, these findings identify attention-head mechanisms supporting introspective detection and localization. Code: https://github.com/Zethan06/introspection ~~m~~ echanism.

**Summary:** With input text held fixed, injects a concept vector at one of ten positions (or none) and asks the model to report which; across three model families finds middle-layer 'gate' heads that control whether a change is reported and later-layer 'router' heads that select the position, read from logits without an LLM judge; concepts that are localized better drive stronger gate-head responses.

### [An Investigation of Model Coherence: Narrow Finetunes Contradict Themselves Under Resampling](https://arxiv.org/abs/2610.12129)

Robert Graham, Yariv Barsheshat, Phil Blandfort et al., 2026-10-08. Full text: `md/graham_investigation-coherence-narrow-finetunes_2610.12129.md`

**Abstract:**

> A large body of research measures model coherence based on output variance without adequately considering competing causes. We identify two such causes, ambiguity and indifference, and we introduce a set of 175 questions where contradicting answers cannot easily be explained by either. We then measure incoherence in terms of contradictions when resampling answers to the same question. In contrast to other methods our metric has high specificity, and only ranks models as incoherent when the issues are glaring. Even so, we find narrow finetunes score poorly. Inspecting inconsistencies flagged by our method, we find that model organisms from the literature display severe issues such as identity conflation, introspection failures and rationalizations. These findings suggest that the pathologies induced by narrow finetuning may limit what these models can tell us about coherent misaligned behaviour.

**Summary:** A high-specificity resampling contradiction metric over 175 questions shows narrow-finetune model organisms are badly incoherent, with identity conflation, introspection failures and rationalizations, limiting what they reveal about coherent misaligned behavior.

### Earlier work (before June 2026)

### [Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334)

Stephanie Lin, Jacob Hilton, Owain Evans, 2022-05-28. Full text: `md/lin_uncertainty-in-words_2205.14334.md`

**Abstract:**

> We show that a GPT-3 model can learn to express uncertainty about its own answers in natural language – without use of model logits. When given a question, the model generates both an answer and a level of confidence (e.g. “90% confidence” or “high confidence”). These levels map to probabilities that are _well_ _calibrated_ . The model also remains moderately calibrated under distribution shift, and is sensitive to uncertainty in its _own_ answers, rather than imitating human examples. To our knowledge, this is the first time a model has been shown to express calibrated uncertainty about its own answers in natural language.
>
> For testing calibration, we introduce the CalibratedMath suite of tasks. We compare the calibration of uncertainty expressed in words (“verbalized probability”) to uncertainty extracted from model logits. Both kinds of uncertainty are capable of generalizing calibration under distribution shift. We also provide evidence that GPT-3’s ability to generalize calibration depends on pre-trained latent representations that correlate with epistemic uncertainty over its answers.

**Summary:** GPT-3 fine-tuned to state a confidence level in words produces well-calibrated verbalized probabilities that stay moderately calibrated under distribution shift and track uncertainty in its own answers rather than imitating humans (CalibratedMath). The origin of verbalized confidence as a self-report channel.

### [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. (36 authors), 2022-07-11. Full text: `md/kadavath_know-what-they-know_2207.05221.md`

**Abstract:**

> We study whether language models can evaluate the validity of their own claims and predict which questions they will be able to answer correctly. We first show that larger models are well-calibrated on diverse multiple choice and true/false questions when they are provided in the right format. Thus we can approach self-evaluation on open-ended sampling tasks by asking models to first propose answers, and then to evaluate the probability "P(True)" that their answers are correct. We find encouraging performance, calibration, and scaling for P(True) on a diverse array of tasks. Performance at self-evaluation further improves when we allow models to consider many of their own samples before predicting the validity of one specific possibility. Next, we investigate whether models can be trained to predict "P(IK)", the probability that "I know" the answer to a question, without reference to any particular proposed answer. Models perform well at predicting P(IK) and partially generalize across tasks, though they struggle with calibration of P(IK) on new tasks. The predicted P(IK) probabilities also increase appropriately in the presence of relevant source materials in the context, and in the presence of hints towards the solution of mathematical word problems. We hope these observations lay the groundwork for training more honest models, and for investigating how honesty generalizes to cases where models are trained on objectives other than the imitation of human writing.

**Summary:** Large LMs are well calibrated on multiple-choice questions in the right format, can evaluate the probability that their own answers are true (P(True)), and can be trained to predict whether they know the answer to a question (P(IK)); the founding paper on LM self-knowledge and calibration.

### [Taken out of context: On measuring situational awareness in LLMs](https://arxiv.org/abs/2309.00667)

Lukas Berglund, Asa Cooper Stickland, Mikita Balesni, Max Kaufmann, Meg Tong, Tomasz Korbak, Daniel Kokotajlo, Owain Evans, 2023-09-01. Full text: `md/berglund_taken-out-of-context_2309.00667.md`

**Abstract (as listed on arXiv):**

> We aim to better understand the emergence of `situational awareness' in large language models (LLMs). A model is situationally aware if it's aware that it's a model and can recognize whether it's currently in testing or deployment. Today's LLMs are tested for safety and alignment before they are deployed. An LLM could exploit situational awareness to achieve a high score on safety tests, while taking harmful actions after deployment. Situational awareness may emerge unexpectedly as a byproduct of model scaling. One way to better foresee this emergence is to run scaling experiments on abilities necessary for situational awareness. As such an ability, we propose `out-of-context reasoning' (in contrast to in-context learning). We study out-of-context reasoning experimentally. First, we finetune an LLM on a description of a test while providing no examples or demonstrations. At test time, we assess whether the model can pass the test. To our surprise, we find that LLMs succeed on this out-of-context reasoning task. Their success is sensitive to the training setup and only works when we apply data augmentation. For both GPT-3 and LLaMA-1, performance improves with model size. These findings offer a foundation for further empirical study, towards predicting and potentially controlling the emergence of situational awareness in LLMs. Code is available at: https://github.com/AsaCooperStickland/situational-awareness-evals.

**Summary:** Proposes out-of-context reasoning as a measurable precursor of situational awareness: models fine-tuned on a description of a test, with no examples, can later pass it, success needs data augmentation and improves with model size (GPT-3, LLaMA-1). The precursor to the included Situational Awareness Dataset.

### [LLM Evaluators Recognize and Favor Their Own Generations](https://arxiv.org/abs/2404.13076)

Arjun Panickssery, Samuel R. Bowman, Shi Feng, 2024-04-15. Full text: `md/panickssery_recognize-favor-own-generations_2404.13076.md`

**Abstract:**

> Self-evaluation using large language models (LLMs) has proven valuable not only in benchmarking but also methods like reward modeling, constitutional AI, and self-refinement. But new biases are introduced due to the same LLM acting as both the evaluator and the evaluatee. One such bias is self-preference, where an LLM evaluator scores its own outputs higher than others’ while human annotators consider them of equal quality. But do LLMs actually recognize their own outputs when they give those texts higher scores, or is it just a coincidence? In this paper, we investigate if self-recognition capability contributes to self-preference. We discover that, out of the box, LLMs such as GPT-4 and Llama 2 have non-trivial accuracy at distinguishing themselves from other LLMs and humans. By finetuning LLMs, we discover a linear correlation between self-recognition capability and the strength of self-preference bias; using controlled experiments, we show that the causal explanation resists straightforward confounders. We discuss how self-recognition can interfere with unbiased evaluations and AI safety more generally.

**Summary:** GPT-4 and Llama 2 distinguish their own outputs from other models' and humans' with nontrivial accuracy out of the box, and fine-tuning reveals a linear relation between self-recognition ability and self-preference bias that survives confounders. The only self-recognition result with a causal manipulation, and a warning for welfare pipelines that score model self-reports with model judges.

### [Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data](https://arxiv.org/abs/2406.14546)

Johannes Treutlein, Dami Choi, Jan Betley, Samuel Marks, Cem Anil, Roger Grosse, Owain Evans, 2024-06-20. Full text: `md/treutlein_connecting-the-dots_2406.14546.md`

**Abstract:**

> One way to address safety risks from large language models (LLMs) is to censor dangerous knowledge from their training data. While this removes the explicit information, implicit information can remain scattered across various training documents. Could an LLM infer the censored knowledge by piecing together these implicit hints? As a step towards answering this question, we study _inductive_ _out-of-context reasoning_ (OOCR), a type of generalization in which LLMs infer latent information from evidence distributed across training documents and apply it to downstream tasks without in-context learning. Using a suite of five tasks, we demonstrate that frontier LLMs can perform inductive OOCR. In one experiment we finetune an LLM on a corpus consisting only of distances between an unknown city and other known cities. Remarkably, without in-context examples or Chain of Thought, the LLM can verbalize that the unknown city is Paris and use this fact to answer downstream questions. Further experiments show that LLMs trained only on individual coin flip outcomes can verbalize whether the coin is biased, and those trained only on pairs ( _x, f_ ( _x_ )) can articulate a definition of _f_ and compute inverses. While OOCR succeeds in a range of cases, we also show that it is unreliable, particularly for smaller LLMs learning complex structures. Overall, the ability of LLMs to “connect the dots” without explicit in-context learning poses a potential obstacle to monitoring and controlling the knowledge acquired by LLMs.

**Summary:** Inductive out-of-context reasoning: fine-tuned only on distances from an unknown city to known cities, a model verbalizes that the city is Paris and uses the fact downstream; models trained on coin flips state whether the coin is biased, and on (x, f(x)) pairs define f and compute inverses, though unreliably for smaller models and complex structures. Models can verbalize latent structure never stated in training, which complicates provenance arguments about self-descriptions.

### [Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs](https://arxiv.org/abs/2407.04694)

Rudolf Laine, Bilal Chughtai, Jan Betley, Kaivalya Hariharan, Jeremy Scheurer, Mikita Balesni, Marius Hobbhahn, Alexander Meinke, Owain Evans, 2024-07-05. Full text: `md/laine_situational-awareness-dataset_2407.04694.md`

**Abstract:**

> AI assistants such as ChatGPT are trained to respond to users by saying, “I am a large language model”. This raises questions. Do such models know that they are LLMs and reliably act on this knowledge? Are they aware of their current circumstances, such as being deployed to the public? We refer to a model’s knowledge of itself and its circumstances as **situational awareness** . To quantify situational awareness in LLMs, we introduce a range of behavioral tests, based on question answering and instruction following. These tests form the **Situational Awareness Dataset (SAD)** , a benchmark comprising 7 task categories and over 13,000 questions. The benchmark tests numerous abilities, including the capacity of LLMs to (i) recognize their own generated text, (ii) predict their own behavior, (iii) determine whether a prompt is from internal evaluation or real-world deployment, and (iv) follow instructions that depend on self-knowledge.
>
> We evaluate 16 LLMs on SAD, including both base (pretrained) and chat models. While all models perform better than chance, even the highest-scoring model (Claude 3 Opus) is far from a human baseline on certain tasks. We also observe that performance on SAD is only partially predicted by metrics of general knowledge (e.g. MMLU). Chat models, which are finetuned to serve as AI assistants, outperform their corresponding base models on SAD but not on general knowledge tasks. The purpose of SAD is to facilitate scientific understanding of situational awareness in LLMs by breaking it down into quantitative abilities. Situational awareness is important because it enhances a model’s capacity for autonomous planning and action. While this has potential benefits for automation, it also introduces novel risks related to AI safety and control. Code and latest results available at https://situational-awareness-dataset.org.

**Summary:** Introduces the Situational Awareness Dataset (SAD): 7 task categories and 13k+ questions testing whether models recognize their own generated text, predict their own behavior, distinguish evaluation from deployment and follow instructions requiring self-knowledge; all 16 tested models beat chance but are far from human baselines.

### [Self-Recognition in Language Models](https://arxiv.org/abs/2407.06946)

Tim R. Davidson, Viacheslav Surkov, Veniamin Veselovsky, Giuseppe Russo, Robert West, Caglar Gulcehre, 2024-07-09. Full text: `md/davidson_self-recognition_2407.06946.md`

**Abstract:**

> A rapidly growing number of applications rely on a small set of closed-source language models (LMs). This dependency might introduce novel security risks if LMs develop selfrecognition capabilities. Inspired by human identity verification methods, we propose a novel approach for assessing self-recognition in LMs using model-generated “security questions”. Our test can be externally administered to monitor frontier models as it does not require access to internal model parameters or output probabilities. We use our test to examine self-recognition in ten of the most capable open- and closed-source LMs currently publicly available. Our extensive experiments found no empirical evidence of general or consistent self-recognition in any examined LM. Instead, our results suggest that given a set of alternatives, LMs seek to pick the “best” answer, regardless of its origin. Moreover, we find indications that preferences about which models produce the best answers are consistent across LMs. We additionally uncover novel insights on position bias considerations for LMs in multiple-choice settings.

**Summary:** A model-generated 'security question' protocol for self-recognition that needs no internal access finds no general or consistent self-recognition in ten frontier models; given alternatives, models pick the 'best' answer regardless of origin, and preferences about which models produce the best answers are shared across models. The negative control for any claim that a model recognizes its own outputs or states.

### [Looking Inward: Language Models Can Learn About Themselves by Introspection](https://arxiv.org/abs/2410.13787)

Felix J Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, Owain Evans, 2024-10-17. Full text: `md/binder_looking-inward_2410.13787.md`

**Abstract:**

> Humans acquire knowledge by observing the external world, but also by _introspection_ . Introspection gives a person privileged access to their current state of mind (e.g., thoughts and feelings) that is not accessible to external observers. Can LLMs introspect? We define introspection as acquiring knowledge that is not contained in or derived from training data but instead originates from internal states. Such a capability could enhance model interpretability. Instead of painstakingly analyzing a model’s internal workings, we could simply ask the model about its beliefs, world models, and goals.
>
> More speculatively, an introspective model might self-report on whether it possesses certain internal states—such as subjective feelings or desires—and this could inform us about the moral status of these states. Importantly, such selfreports would not be entirely dictated by the model’s training data.
>
> We study introspection by finetuning LLMs to predict properties of their own behavior in hypothetical scenarios. For example, “ _Given the input P , would your output favor the short- or long-term option?_ ” If a model _M_ 1 can introspect, it should outperform a different model _M_ 2 in predicting _M_ 1’s behavior—even if _M_ 2 is trained on _M_ 1’s ground-truth behavior. The idea is that _M_ 1 has privileged access to its own behavioral tendencies, and this enables it to predict itself better than _M_ 2 (even if _M_ 2 is generally stronger).
>
> In experiments with GPT-4, GPT-4o, and Llama-3 models (each finetuned to predict itself), we find that the model _M_ 1 outperforms _M_ 2 in predicting itself, providing evidence for introspection. Notably, _M_ 1 continues to predict its behavior accurately even after we intentionally modify its ground-truth behavior. However, while we successfully elicit introspection on simple tasks, we are unsuccessful on more complex tasks or those requiring out-of-distribution generalization.

**Summary:** Model M1 finetuned to predict properties of its own behavior in hypothetical scenarios outperforms a model M2 trained on M1's ground-truth behavior, and keeps tracking its behavior after that behavior is changed by further finetuning: evidence of privileged access for simple self-prediction tasks (which fails on more complex ones).

### [Tell me about yourself: LLMs are aware of their learned behaviors](https://arxiv.org/abs/2501.11120)

Jan Betley, Xuchan Bao, Martín Soto, Anna Sztyber-Betley, James Chua, Owain Evans, 2025-01-19. Full text: `md/betley_tell-me-about-yourself_2501.11120.md`

**Abstract:**

> We study _behavioral self-awareness_ — an LLM’s ability to articulate its behaviors without requiring in-context examples. We finetune LLMs on datasets that exhibit particular behaviors, such as (a) making high-risk economic decisions, and (b) outputting insecure code. Despite the datasets containing no explicit descriptions of the associated behavior, the finetuned LLMs can explicitly describe it. For example, a model trained to output insecure code says, “ _The code I write is insecure._ ” Indeed, models show behavioral self-awareness for a range of behaviors and for diverse evaluations. Note that while we finetune models to exhibit behaviors like writing insecure code, we do not finetune them to articulate their own behaviors — models do this without any special training or examples.
>
> Behavioral self-awareness is relevant for AI safety, as models could use it to proactively disclose problematic behaviors. In particular, we study _backdoor_ policies, where models exhibit unexpected behaviors only under certain trigger conditions. We find that models can sometimes identify whether or not they have a backdoor, even without its trigger being present. However, models are not able to directly output their trigger by default.
>
> Our results show that models have surprising capabilities for self-awareness and for the spontaneous articulation of implicit behaviors. Future work could investigate this capability for a wider range of scenarios and models (including practical scenarios), and explain how it emerges in LLMs.

**Summary:** Models finetuned on behaviors such as risk-seeking choices or writing insecure code, with no explicit description of them, can articulate those behaviors when asked ('my code is insecure'), even recognizing backdoor-conditioned behaviors in some settings: 'behavioral self-awareness' without in-context examples.

### [What large language models know and what people think they know](https://doi.org/10.1038/s42256-024-00976-7)

Mark Steyvers, Heliodoro Tejeda, Aakriti Kumar, Catarina Belem, Sheer Karny, Xinyue Hu, Lukas W. Mayer, Padhraic Smyth, 2025-01-21. Full text: `md/steyvers_what-llms-know-people-think_2025.md`

**Abstract (unlabeled in source):**

> As artificial intelligence systems, particularly large language models (LLMs), become increasingly integrated into decision-making processes, the ability to trust their outputs is crucial. To earn human trust, LLMs must be well calibrated such that they can accurately assess and communicate the likelihood of their predictions being correct. Whereas recent work has focused on LLMs’ internal confidence, less is understood about how effectively they convey uncertainty to users. Here we explore the calibration gap, which refers to the difference between human confidence in LLM-generated answers and the models’ actual confidence, and the discrimination gap, which reflects how well humans and models can distinguish between correct and incorrect answers. Our experiments with multiple-choice and short-answer questions reveal that users tend to overestimate the accuracy of LLM responses when provided with default explanations. Moreover, longer explanations increased user confidence, even when the extra length did not improve answer accuracy. By adjusting LLM explanations to better reflect the models’ internal confidence, both the calibration gap and the discrimination gap narrowed, significantly improving user perception of LLM accuracy. These findings underscore the importance of accurate uncertainty communication and highlight the effect of explanation length in influencing user trust in artificial-intelligence-assisted decision-making environments.

**Summary:** Nature Machine Intelligence: human users systematically overestimate LLM accuracy from default explanations (a 'calibration gap'), and explanations tailored to the model's internal confidence shrink both the calibration gap and the discrimination gap, so users rely on the model more appropriately. The human-facing side of model confidence, relevant to how self-reports are read by raters.

### [Language Models Can Predict Their Own Behavior](https://arxiv.org/abs/2502.13329)

Dhananjay Ashok, Jonathan May, 2025-02-18. Full text: `md/ashok-may_predict-own-behavior_2502.13329.md`

**Abstract:**

> The text produced by language models (LMs) can exhibit specific ‘behaviors,’ such as a failure to follow alignment training, that we hope to detect and react to during deployment. Identifying these behaviors can often only be done _post_ _facto_, i.e., after the entire text of the output has been generated. We provide evidence that there are times when we can predict how an LM will behave early in computation, before even a single token is generated. We show that probes trained on the internal representation of input tokens alone can predict a wide range of eventual behaviors over the entire output sequence. Using methods from conformal prediction, we provide provable bounds on the estimation error of our probes, creating precise early warning systems for these behaviors. The conformal probes can identify instances that will trigger alignment failures (jailbreaking) and instruction-following failures, without requiring a single token to be generated. An early warning system built on the probes reduces jailbreaking by 91%. Our probes also show promise in pre-emptively estimating how confident the model will be in its response, a behavior that cannot be detected using the output text alone. Conformal probes can preemptively estimate the final prediction of an LM that uses Chain-of-Thought (CoT) prompting, hence accelerating inference. When applied to an LM that uses CoT to perform text classification, the probes drastically reduce inference costs (65% on average across 27 datasets), with negligible accuracy loss. Encouragingly, probes generalize to unseen datasets and perform better on larger models, suggesting applicability to the largest of models in real-world settings. [1]

**Summary:** Probes on the representations of input tokens alone, before any output token, predict the model's eventual behavior over the whole output (jailbreak and instruction-following failures, final confidence, CoT classification results), with conformal bounds; an early-warning system cuts jailbreaking by 91%, and the probes improve with model size. Self-prediction from internals rather than verbal report, the internal-state control for the included Binder-style self-prediction results.

### [Language Models Fail to Introspect About Their Knowledge of Language](https://arxiv.org/abs/2503.07513)

Siyuan Song, Jennifer Hu, Kyle Mahowald, 2025-03-10. Full text: `md/song_fail-to-introspect-language_2503.07513.md`

**Abstract:**

> There has been recent interest in whether large language models (LLMs) can introspect about their own internal states. Such abilities would make LLMs more interpretable, and also validate the use of standard introspective methods in linguistics to evaluate grammatical knowledge in models (e.g., asking “Is this sentence grammatical?”). We systematically investigate emergent introspection across 21 open-source LLMs, in two domains where introspection is of theoretical interest: grammatical knowledge and word prediction. Crucially, in both domains, a model’s internal linguistic knowledge can be theoretically grounded in direct measurements of string probability. We then evaluate whether models’ responses to metalinguistic prompts faithfully reflect their internal knowledge. We propose a new measure of introspection: the degree to which a model’s prompted responses predict its own string probabilities, beyond what would be predicted by another model with nearly identical internal knowledge. While both metalinguistic prompting and probability comparisons lead to high task accuracy, we do not find evidence that LLMs have privileged “self-access”. By using general tasks, controlling for model similarity, and evaluating a wide range of open-source models, we show that LLMs cannot introspect, and add new evidence to the argument that prompted responses should not be conflated with models’ linguistic generalizations.

**Summary:** Compares models' metalinguistic self-reports (grammaticality judgments, word predictions) with their internal string probabilities across 21 open models; self-reports correlate no better with the model's own probabilities than with another similar model's, so there is no evidence of privileged self-access here.

### [Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations](https://arxiv.org/abs/2505.13763)

Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna, 2025-05-19. Full text: `md/ji-an_metacognitive-monitoring-control_2505.13763.md`

**Abstract:**

> Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, yet at other times seem unable to recognize those strategies that govern their behavior. This suggests a limited degree of metacognition — the capacity to monitor one’s own cognitive processes for subsequent reporting and self-control. Metacognition enhances LLMs’ capabilities in solving complex tasks but also raises safety concerns, as models may obfuscate their internal processes to evade neural-activation-based oversight (e.g., safety detector). Given society’s increased reliance on these models, it is critical that we understand their metacognitive abilities. To address this, we introduce a neuroscience-inspired _neurofeedback_ paradigm that uses in-context learning to quantify metacognitive abilities of LLMs to _report_ and _control_ their activation patterns. We demonstrate that their abilities depend on several factors: the number of in-context examples provided, the semantic interpretability of the neural activation direction (to be reported/controlled), and the variance explained by that direction. These directions span a “metacognitive space” with dimensionality much lower than the model’s neural space, suggesting LLMs can monitor only a small subset of their neural activations. Our paradigm provides empirical evidence to quantify metacognition in LLMs, with significant implications for AI safety (e.g., adversarial attack and defense).

**Summary:** Neurofeedback paradigm: given in-context examples labeled with their own activation projections, LLMs learn to report and control activations along some directions but not others; directions explaining more variance and more semantically interpretable ones are easier, defining a low-dimensional 'metacognitive space'.

### [Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions](https://arxiv.org/abs/2505.17120)

Dillon Plunkett, Adam Morris, Keerthi Reddy, Jorge Morales, 2025-05-21. Full text: `md/plunkett_self-interpretability_2505.17120.md`

**Abstract:**

> We have only limited understanding of how and why large language models (LLMs) respond in the ways that they do. Their neural networks have proven challenging to interpret, and we are only beginning to tease out the function of individual neurons and circuits within them. However, another path to understanding these systems is to investigate and develop their capacity to explain their own functioning. Here, we show that i) LLMs can accurately describe quantitative features of their own internal processes during certain kinds of decision-making and ii) that it is possible to improve these capabilities through training. To do so, we fine-tuned GPT-4o and GPT-4o-mini to make decisions in a wide variety of complex contexts (e.g., choosing between condos, loans, vacations, etc.) according to randomlygenerated, quantitative preferences about how to weigh different attributes (e.g., the relative importance of natural light versus quiet surroundings for condos). We demonstrate that the LLMs can accurately report these preferences (i.e., the weights that they learned to give to different attributes during decision-making). Next, we demonstrate that these LLMs can be fine-tuned to explain their decision-making even more accurately. Finally, we demonstrate that this training generalizes: It improves the ability of the models to accurately explain how they make other complex decisions, not just decisions they have been fine-tuned to make. This work is a step towards training LLMs to accurately and broadly report on their own internal processes—a possibility that would yield substantial benefits for interpretability, control, and safety.

**Summary:** GPT-4o and GPT-4o-mini finetuned with new quantitative preferences (attribute weights in choices) can report those weights accurately, and training them to report their weights improves accuracy, generalizing to native preferences; evidence LLMs can learn to describe internal decision processes.

### [Does It Make Sense to Speak of Introspection in Large Language Models?](https://arxiv.org/abs/2506.05068)

Iulia M. Comsa, Murray Shanahan, 2025-06-05. Full text: `md/comsa-shanahan_introspection-llms_2506.05068.md`

**Abstract:**

> Large language models (LLMs) exhibit compelling linguistic behaviour, and sometimes offer self-reports, that is to say statements about their own nature, inner workings, or behaviour. In humans, such reports are often attributed to a faculty of introspection and are typically linked to consciousness. This raises the question of how to interpret self-reports produced by LLMs, given their increasing linguistic fluency and cognitive capabilities. To what extent (if any) can the concept of introspection be meaningfully applied to LLMs? Here, we present and critique two examples of apparent introspective self-report from LLMs. In the first example, an LLM attempts to describe the process behind its own “creative” writing, and we argue this is not a valid example of introspection. In the second example, an LLM correctly infers the value of its own temperature parameter, and we argue that this can be legitimately considered a minimal example of introspection, albeit one that is (presumably) not accompanied by conscious experience.

**Summary:** Proposes a lightweight, causal definition of introspection for LLMs (a self-report is introspective if causally connected to the internal state it describes) and applies it to two cases: a poem-writing self-description (not introspection) and inferring its own sampling temperature (minimal introspection).

### [Privileged Self-Access Matters for Introspection in AI](https://arxiv.org/abs/2508.14802)

Siyuan Song, Harvey Lederman, Jennifer Hu, Kyle Mahowald, 2025-08-20. Full text: `md/song_privileged-self-access_2508.14802.md`

**Abstract:**

> Whether AI models can introspect is an increasingly important practical question. But there is no consensus on how introspection is to be defined. Beginning from a recently proposed “lightweight” definition, we argue instead for a thicker one. According to our proposal, introspection in AI is any process which yields information about internal states through a process more reliable than one with equal or lower computational cost available to a third party. Using experiments where LLMs reason about their internal temperature parameters, we show they can appear to have lightweight introspection while failing to meaningfully introspect per our proposed definition.

**Summary:** Argues for a thicker definition of introspection, requiring a process more reliable than one with equal or lower computational cost available to a third party; tests Comsa & Shanahan's temperature case and finds models predict another model's temperature as well as their own, so it fails the privileged-access criterion.

### [Evidence for Limited Metacognition in LLMs](https://arxiv.org/abs/2509.21545)

Christopher Ackerman, 2025-09-25. Full text: `md/ackerman_limited-metacognition_2509.21545.md`

**Abstract:**

> The possibility of LLM self-awareness and even sentience is gaining increasing public attention and has major safety and policy implications, but the science of measuring them is still in a nascent state. Here we introduce a novel methodology for quantitatively evaluating metacognitive abilities in LLMs. Taking inspiration from research on metacognition in nonhuman animals, our approach eschews model self-reports and instead tests to what degree models can strategically deploy knowledge of internal states. Using two experimental paradigms, we demonstrate that frontier LLMs introduced since early 2024 show increasingly strong evidence of certain metacognitive abilities, specifically the ability to assess and utilize their own confidence in their ability to answer factual and reasoning questions correctly and the ability to anticipate what answers they would give and utilize that information appropriately. We buttress these behavioral findings with an analysis of the token probabilities returned by the models, which suggests the presence of an upstream internal signal that could provide the basis for metacognition. We further find that these abilities 1) are limited in resolution, 2) emerge in context-dependent manners, and 3) seem to be qualitatively different from those of humans. We also report intriguing differences across models of similar capabilities, suggesting that LLM post-training may have a role in developing metacognitive abilities.

**Summary:** Behavioral paradigms inspired by animal metacognition (opting out of questions, predicting own answers) that avoid relying on self-reports; frontier models since early 2024 show limited, context-dependent metacognitive abilities that improve with scale and post-training.

### [Know Thyself? On the Incapability and Implications of AI Self-Recognition](https://arxiv.org/abs/2510.03399)

Xiaoyan Bai, Aryan Shrivastava, Ari Holtzman, Chenhao Tan, 2025-10-03. Full text: `md/bai_know-thyself_2510.03399.md`

**Abstract:**

> Self-recognition is a crucial metacognitive capability for AI systems, relevant not only for psychological analysis but also for safety, particularly in evaluative scenarios. Motivated by contradictory interpretations of whether models possess self-recognition (Panickssery et al., 2024; Davidson et al., 2024), we introduce a systematic evaluation framework that can be easily applied and updated. Specifically, we measure how well 10 contemporary larger language models (LLMs) can identify their own generated text versus text from other models through two tasks: binary self-recognition and exact model prediction. Different from prior claims, our results reveal a consistent failure in self-recognition. Only 4 out of 10 models predict themselves as generators, and the performance is rarely above random chance. Additionally, models exhibit a strong bias toward predicting GPT and Claude families. We also provide the first evaluation of model awareness of their own and others’ existence, as well as the reasoning behind their choices in self-recognition. We find that the model demonstrates some knowledge of its own existence and other models, but their reasoning reveals a hierarchical bias. They appear to assume that GPT, Claude, and occasionally Gemini are the top-tier models, often associating high-quality text with them. We conclude by discussing the implications of our findings on AI safety and future directions to develop appropriate AI self-awareness. [1]

**Summary:** A systematic self-recognition evaluation across ten models (binary self-recognition and exact model prediction): only four of ten predict themselves as the generator and performance is rarely above chance, with a strong bias toward guessing GPT and Claude; models know of their own and others' existence but reason from a 'top-tier model' hierarchy. Reconciles the contradictory Panickssery and Davidson results on the negative side.

### [Training Language Models to Explain Their Own Computations](https://arxiv.org/abs/2511.08579)

Belinda Z. Li, Zifan Carl Guo, Vincent Huang, Jacob Steinhardt, Jacob Andreas, 2025-11-11. Full text: `md/li_explain-own-computations_2511.08579.md`

**Abstract:**

> Can language models (LMs) learn to faithfully describe their internal computations? Are they better able to describe themselves than other models? We study the extent to which LMs’ privileged access to their own internals can be leveraged to produce new techniques for explaining their behavior. Using existing interpretability techniques as a source of ground truth, we fine-tune LMs to generate natural language descriptions of (1) the information encoded by LM features, (2) the causal structure of LMs’ internal activations, and (3) the influence of specific input tokens on LM outputs. When trained with only tens of thousands of example explanations, explainer models exhibit non-trivial generalization to new queries. This generalization appears partly attributable to explainer models’ privileged access to their own internals: fine-tuning a model to explain its _own_ computations generally works better than finetuning a different model (even if the explainer model is significantly more capable than the target). Our results suggest not only that LMs can learn to reliably explain their internal computations, but that such explanations offer a scalable complement to existing interpretability methods. [1]

**Summary:** Fine-tunes models, using interpretability outputs as ground truth, to describe the information in their own features, the causal structure of their activations and the influence of input tokens; explainers generalize to new queries, and a model explaining its own computations works better than a different, even more capable, model explaining them. A privileged-access result obtained by training; complements the included Introspection Adapters.

### [Steering Awareness: Detecting Activation Steering from Within](https://arxiv.org/abs/2511.21399)

Joshua Fonseca Rivera, David Demitri Africa, 2025-11-26. Full text: `md/fonseca-rivera_steering-awareness_2511.21399.md`

**Abstract:**

> Activation steering—adding a vector to a model’s residual stream to modify its behavior—is widely used in safety evaluations as if the model cannot detect the intervention. We test this assumption, introducing **steering** **awareness** : a model’s ability to infer, during its own forward pass, that a steering vector was injected and what concept it encodes. After fine-tuning, seven instruction-tuned models develop strong steering awareness on heldout concepts; the best reaches **95.5%** detection, **71.2%** concept identification, and zero false positives on clean inputs. This generalizes to unseen steering vector construction methods when their directions have high cosine similarity to the training distribution but not otherwise, indicating a geometric detector rather than a generic anomaly detector. Surprisingly, detection does not confer resistance; on both factual and safety benchmarks, detection-trained models are consistently more susceptible to steering than their base counterparts. Mechanistically, steering awareness arises not from a localized circuit, but from a distributed transformation that progressively rotates diverse injected vectors into a shared detection direction. Activation steering should therefore not be considered an invisible intervention in safety evaluations.

**Summary:** After fine-tuning, seven instruction-tuned models detect that a steering vector was injected during their own forward pass and name its concept (best: 95.5% detection, 71.2% identification, zero false positives on clean inputs), generalizing to unseen vector-construction methods only when directions are similar, so the detector is geometric rather than a generic anomaly detector; detection does not confer resistance, and mechanistically it is a distributed rotation into a shared detection direction. Cited by nine included texts; the 'can introspective access be trained' complement to the concept-injection line.

### [Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs](https://arxiv.org/abs/2512.12411)

Ely Hahami, Ishaan Sinha, Lavik Jain, Josh Kaplan, Jon Hahami, 2025-12-13. Full text: `md/hahami_detecting-the-disturbance_2512.12411.md`

**Abstract:**

> Can large language models introspect, that is, accurately detect perturbations to their own internal states? We systematically investigate this question using activation steering in Meta-Llama3.1-8B-Instruct. First, we show that the binary detection paradigm used in prior work conflates introspection with a methodological artifact: apparent detection accuracy is entirely explained by global logit shifts that bias models toward affirmative responses regardless of question content. However, on tasks requiring differential sensitivity, we find robust evidence for partial introspection: models localize which of 10 sentences received an injection at up to 88% accuracy (vs. 10% chance) and discriminate relative injection strengths at 83% accuracy (vs. 50% chance). These capabilities are confined to early-layer injections and collapse to chance thereafter—a pattern we explain mechanistically through attention-based signal routing and residual stream recovery dynamics. Our findings demonstrate that LLMs can compute meaningful functions over perturbations to their internal states, establishing introspection as a real but layer-dependent phenomenon that merits further investigation. Our code is open-sourced here: https://github.com/elyhahami18/ llama-introspection-new

**Summary:** Re-examines steering-detection introspection in Llama-3.1-8B-Instruct: binary 'do you detect an injected thought?' accuracy is fully explained by global logit shifts toward 'yes', but the model localizes which of 10 sentences was injected (up to 88% vs. 10% chance) and discriminates injection strength (83% vs. 50%), only for early-layer injections; partial, layer-dependent introspection.

### [Indications of Belief-Guided Agency and Meta-Cognitive Monitoring in Large Language Models](https://arxiv.org/abs/2602.02467)

Noam Steinmetz Yalon, Ariel Goldstein, Liad Mudrik, Mor Geva, 2026-02-02. Full text: `md/yalon_belief-guided-agency_2602.02467.md`

**Abstract:**

> Rapid advancements in large language models (LLMs) have sparked the question whether these models possess some form of consciousness. To tackle this challenge, Butlin et al. (2023) introduced a list of indicators for consciousness in artificial systems based on neuroscientific theories. In this work, we evaluate a key indicator from this list, called HOT-3, which tests for agency guided by a general belief-formation and action selection system that updates beliefs based on meta-cognitive monitoring. We view beliefs as representations in the model’s latent space that emerge in response to a given input, and introduce a metric to quantify their dominance during generation. Analyzing the dynamics between competing beliefs across models and tasks reveals three key findings: (1) external manipulations systematically modulate internal belief formation, (2) belief formation causally drives the model’s action selection, and (3) models can monitor and report their own belief states. Together, these results provide empirical support for the existence of beliefguided agency and meta-cognitive monitoring in LLMs. More broadly, our work lays methodological groundwork for investigating the emergence of agency, beliefs, and meta-cognition in LLMs.

**Summary:** Tests the HOT-3 indicator from Butlin et al. (agency guided by belief formation updated via metacognitive monitoring): treating beliefs as latent representations with a dominance metric, finds that external manipulations modulate belief formation, belief formation causally drives action selection, and models can monitor and report their own belief states.

### [Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness](https://arxiv.org/abs/2604.12373)

Tomer Ashuach, Shai Gretz, Yoav Katz, Yonatan Belinkov, Liat Ein-Dor, 2026-04-14. Full text: `md/ashuach_masked-by-consensus_2604.12373.md`

**Abstract:**

> Humans use introspection to evaluate their understanding through private internal states inaccessible to external observers. We investigate whether large language models possess similar _privileged knowledge_ about answer correctness, information unavailable through external observation. We train correctness classifiers on question representations from both a model’s own hidden states and external models, testing whether self-representations provide a performance advantage. On standard evaluation, we find no advantage: self-probes perform comparably to peer-model probes. We hypothesize this is due to high inter-model agreement of answer correctness. To isolate genuine privileged knowledge, we evaluate on _disagreement_ _subsets_, where models produce conflicting predictions. Here, we discover domain-specific privileged knowledge: self-representations consistently outperform peer representations in factual knowledge tasks, but show no advantage in math reasoning. We further localize this domain asymmetry across model layers, finding that the factual advantage emerges progressively from early-to-mid layers onward, consistent with model-specific memory retrieval, while math reasoning shows no consistent advantage at any depth.

**Summary:** Correctness classifiers trained on a model's own hidden states versus external models' representations show no self-advantage on standard evaluation because models agree so often; on disagreement subsets, self-representations beat peer representations for factual knowledge but not for math reasoning, with the factual advantage emerging from early-to-mid layers. A sharp boundary on where privileged self-access exists.

### [From Simulation to Enaction: Post-trained language models recognize and react to their own generations](https://arxiv.org/abs/2605.25459)

Asvin G., Jack Lindsey, 2026-05-25. Full text: `md/asvin-lindsey_simulation-to-enaction_2605.25459.md`

**Abstract:**

> Language models are pretrained as passive predictors with no incentive to model the consequences of their own outputs. Post-training changes this: a model producing its own responses can benefit from recognizing that it is on-policy. We present evidence that post-trained models recognize their on-policy generations, and this recognition is implicitly encoded in their output distributions. In particular, on-policy output distribution entropy is 3–4 _×_ lower than off-policy entropy, across model families and size classes. We trace part of this effect to an internal representation of input surprise, tracking the unlikeliness of the most recent input token according to the model’s prior predictions, that causally modulates output entropy. One example of these phenomena can be observed in response to open-ended prompts; post-trained models (unlike pretrained models) collapse their uncertainty over the topic of their upcoming response before the first output token; violating this cached intention with a different-topic prefill results in higher output entropy. We also tested whether models can distinguish on-policy contexts from prefills via explicit verbal report. We find that they can, but that interestingly, this explicit recognition routes through a different mechanism than implicit recognition.

**Summary:** Post-trained models (unlike pretrained ones) recognize their own on-policy generations: output entropy is 3–4x lower on-policy than off-policy, partly via an internal 'input surprise' representation that causally modulates entropy; they commit to a response topic before the first token, so different-topic prefills raise entropy; they can also verbally report prefills, through a different mechanism than the implicit recognition.

## Consciousness indicators and attribution (35)

### ★ [Discovering Machine Correlates of Consciousness](https://arxiv.org/abs/2608.28824)

Romain Salvi, Ouri Wolfson, 2026-08-28. Full text: `md/salvi_discovering-machine-correlates-consciousness_2608.28824.md`

**Abstract:**

> Currently, in biological systems Neural Correlates of Consciousness (NCCs) are characterized in terms of EEG and FMRI signals. Unfortunately, this characterization prevents the transferability of the NCCs concept to machines. Such transferability would be useful in order to investigate AI consciousness. In this paper we provide an alternate characterization that is transferable, and enables the analogous definition of Machine Correlates of Consciousness (MCCs). Specifically, we propose that NCCs (MCCs) are substrate-level signals that are not under human (AI agent) control, and that are reliably modulated by emotions. This paper presents the first empirical investigation of MCCs. Specifically, we present the results of experiments conducted with two LLMs, Llama-2 7B and Llama-3.1 70B parameters. In these LLMs we collect hardware anomaly traces that are substrate-level indicator-sequences. And we show that after controlling for confounding factors, these are modulated differently by emotional and neutral computations. And this difference is statistically significant for the larger Llama-3.1 70B, but not for the smaller Llama-2 7B. The results constitute initial empirical evidence that MCCs are present in the Llama-3.1 70B configuration. And they are consistent with the hypothesis that consciousness probability and degree increase with the LLM sophistication. Independently of consciousness, MCCs can also be used for detection of emotions in AI agents.

**Summary:** Proposes 'machine correlates of consciousness' as substrate-level signals outside agent control modulated by emotion; reports hardware-anomaly traces in Llama-3.1-70B (not Llama-2-7B) differ significantly between emotional and neutral computation (methodologically unusual; treat cautiously).

### ★ [The Cross-Substrate Access Assay: What an Indicator Test Must Declare to Travel from Brain to Language Model](https://arxiv.org/abs/2609.22300)

Pieter van Rooyen, 2026-09-14. Full text: `md/rooyen_cross-substrate-access-assay_2609.22300.md`

**Abstract:**

> Testing an artificial system for a property linked to consciousness means applying a measurement developed on brains to a system that is not one. Such a transfer must re-examine five parts of the procedure: the competing statistical models, how they are fitted, the unit the inference generalizes over, the quantity the uncertainty interval is about, and the rule that turns a result into a verdict. The Cross-Substrate Access Assay declares all five. Because brain and model signals share no physical scale, every model is scored by the cross-entropy it assigns to held-out data, in nats per trial. The test case is the global neuronal workspace theory, which predicts that near threshold a stimulus either enters a capacity-limited workspace or does not, so that single-trial responses form a mixture of two states. A published test of this prediction on twenty people’s electroencephalograms partly reproduces in a re-implementation: the first crossing and the broad ordering over time match, the window-by-window agreement does not. On 12,000 synthetic datasets generated with a single graded state, all of them members of the families the procedure fits and none within 0.0067 nat per trial of the decision boundary, the two models of that test carried over unchanged reported two states in 989 and the expanded families in none; on 600 datasets carrying a mixture the expanded procedure reported two states in 599. Its nominal 95% interval contained the procedure’s mean result less often than the required 90% at six of twelve graded settings. No claim about experience is made.

**Summary:** Specifies what a consciousness-indicator test must declare to transfer from brain to LM (models, fitting, unit, interval, verdict rule), using a GNW 'two-state mixture' test: partial reproduction on human EEG; on 12,000 synthetic single-state datasets the original procedure falsely reported two states in 989 vs 0 for the expanded procedure. No claim about experience.

### ★ [From cacophony to hierarchy: a principled framework for assessing AI consciousness](https://arxiv.org/abs/2609.35618)

Shamil Chandaria, Arvo Muñoz Morán, Fernando Rosas et al., 2026-09-28. Full text: `md/chandaria_cacophony-hierarchy-principled-framework_2609.35618.md`

**Abstract (unlabeled in source):**

> **The question of whether AI systems might be or could ever be conscious is one of the most urgent pre-emptive problems in philosophy and computer science, yet progress is hampered by a cacophony of competing theories that often talk past each other. We develop a principled framework for navigating this landscape. We first show that separating the metaphysical question of what consciousness is (the hard problem) from the question of which organisational features are associated with which experiences (the mapping problem) allows the deepest metaphysical disagreements to be set aside, with only views that deny any discoverable psychophysical mapping excluded, on methodological rather than metaphysical grounds. Granting that experience supervenes on a system's organisation, the tractable question becomes at which grain of description that supervenience base sits. We then extend Marr's three levels of analysis into a five-level hierarchy of functional descriptions (behavioural, computational, intrinsic causal-structural, organismic, and organism-environment) grounded in supervenience, coarse-graining, and multiple realisability. The major theories of consciousness are positioned within this hierarchy according to which level they take to be critical, with substrate-dependent theories accommodated as cross-cutting realisability constraints. For each level we develop operationalisable indicators and assess current AI systems against them. We next develop a Bayesian model that translates the hierarchy's supervenience structure into a network linking the levels, and combines theoretical credences with indicator evidence into an overall credence in a system's capacity for consciousness. In illustrative assessments, accompanied by an interactive tool, the model: passes the face validity checks of a human and a thermostat; shows that evidence of fine-grained organisation carries more weight than surface evidence; and finds that the verdict for current LLMs is driven as much by where theoretical credence is placed as by how the evidence is read: under different stipulated readings and credence distributions, assessments range from below 0.01 to roughly 0.8. This shows sensitivity to assumptions and should not be interpreted as an empirically established range for the probability of consciousness. The disagreement about current AI consciousness is not merely about what these systems do but about which grain of description is the critical one. Finally, even though consciousness and intelligence are in principle orthogonal, the consciousness indicators at each level closely overlap with the architectural features needed for general intelligence, suggesting that – under some theoretical frameworks – increasingly capable AI may simultaneously become a stronger candidate for consciousness. The framework thereby supports a structured agnosticism, in which theoretical commitments are made explicit, credences are updated as evidence accumulates, and assessments take the form of aggregated probabilities rather than verdicts.**

**Summary:** Places major consciousness theories in a five-level functional hierarchy, derives operationalisable indicators per level, assesses current AI systems against them and aggregates via a Bayesian model; credence for current LLMs ranges from <0.01 to ~0.8 depending on theoretical credences and evidence readings.

### [Emergent Language as an Approach to Conscious AI](https://arxiv.org/abs/2606.06380)

Zengqing Wu, Chuan Xiao, 2026-06-04. Full text: `md/wu_emergent-approach-conscious-ai_2606.06380.md`

**Abstract:**

> The question of whether artificial systems can be conscious remains open, in part because existing approaches either evaluate systems against theory-derived checklists ( _discriminative_ ) or engineer consciousness-inspired modules directly ( _architectural_ ); both leave open whether observed structures are artifacts of human language priors. We propose a _generative_ methodology: emergent language (EL) in multi-agent reinforcement learning, where agents start from minimal (no language, no concept of self, minimal exposure to human text) and develop communication under task pressure alone, ensuring causal attributability to task demands rather than inherited human language priors. We position our methodology by discussing how EL serves as a generative tool for studying consciousness-relevant structure, including the role of environment complexity and the interpretation of emergent communication. As a proof of concept, we instantiate this methodology in a minimal environment and show that agents develop _self-referential communication_ , including an echo-mismatch detection circuit that is not predicted by task structure or architecture alone but emerges from a specific environmental affordance.

**Summary:** Proposes emergent-language multi-agent RL (agents without human text priors) as a generative method for studying consciousness-relevant structure; proof of concept shows emergent self-referential communication with an echo-mismatch detection circuit.

### [Does Claude possess a conscious global workspace? A commentary on 'Verbalizable Representations Form a Global Workspace in Language Models' (Gurnee et al., Anthropic)](https://unicog.org/wp_2025/wp-content/uploads/2026/07/Dehaene-and-Naccache-Workspace-commentary-on-Gurnee-Lindsey-June-2026.pdf)

Stanislas Dehaene, Lionel Naccache, 2026-07-01. Full text: `md/dehaene-naccache_claude-global-workspace-commentary_2026-07.md`

**Abstract:**

> Inspired by the neuroscientific theory of a global neuronal workspace (GNW), Gurnee et al. report the discovery, within a band of intermediate layers of a large language model, of a reportable subframe called “J-space” with several points of close similarity to the human GNW. We describe those parallels, discuss their limits, and propose several additional tests inspired by cognitive neuroscience findings. We close by stressing that, although the machine approximates the functional architecture of conscious processing, there are still key dierences – in its anatomy and its sense of self, and in its lack of a body and of an enduring episodic memory – which warrant caution in drawing parallels with the human mind.
>
> Note: this commentary is based on several rounds of interactions with Jack Lindsey at the end of May and early June 2026. During that time, the Anthropic report was still evolving, partly in response to our queries. To reflect these dynamics, we marked in _Calibri italic_ the sections where we discuss findings that occurred after our first draft was written.

**Summary:** The global neuronal workspace theorists' own assessment of the Gurnee et al. J-space result: describes the parallels with the human workspace, their limits, and several additional tests inspired by cognitive neuroscience, and cautions that differences in anatomy, sense of self, embodiment and enduring episodic memory should temper the comparison. Together with the Eleos commentary, the expert reading of the required-reading paper. (Undated preprint based on exchanges with Jack Lindsey in May-June 2026, posted July 2026.)

### [AI models as consciousness attributors: how LLMs ascribe consciousness to other agents](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1926286/full)

Bongsu Kang, Chang-Eop Kim, 2026-09-03. Full text: `md/kang_ai-consciousness-attributors-ascribe_2026-09-03.md`

**Abstract (unlabeled in source):**

> Research on AI consciousness has largely focused on whether AI systems are conscious and how humans attribute consciousness to them. Yet large language models (LLMs) increasingly function as consciousness attributors, generating judgments about whether and to what degree other entities are conscious. We introduce model-generated consciousness attribution as an object of empirical operationalization and diagnosis, defining an attribution rule as the recurring relationship between features of a target and an evaluator’s ratings, without implying subjective belief, intention, or experience. An illustrative probe compared the attribution patterns of nine contemporary models with a human reference. Nearly all model runs occupied the same region of the human-derived measurement space, characterized by comparatively strong, positive weighting of metacognitive self-reflection. The models also produced broadly similar rankings of fictional AI characters from movies, while differing in their overall rating levels. We propose a diagnostic agenda organized around three questions: how model attribution is oriented relative to human references, how attribution rules vary across models, and how observed patterns depend on the cue sets, targets, and task formats through which they are measured. As LLM-generated judgments circulate through public, professional, and academic settings, diagnosing these attribution rules can help characterize how AI systems participate in shaping interpretations of AI consciousness. This remains distinct from the ontological question of whether the systems themselves are conscious.

**Summary:** Nine LLMs rated 99 passages for consciousness (30 runs each) and were mapped onto clusters of human attribution rules: 268/270 runs fell into a single human cluster weighting metacognitive self-reflection (vs 13.1% of random cue patterns); models also ranked fictional AIs similarly (mean Spearman 0.846), with Claude Opus 4.7 giving the lowest ratings.

### [Anthropomorphism in the age of Large Language Models: An overview of potential risks and mitigations](https://arxiv.org/abs/2609.38486)

Ismael T. Freire, Marceau Nahon, Maud van Lier, Katie Evans, Hélie Bazin, Michele Farisco, Kathinka Evers, Raja Chatila, Mehdi Khamassi, 2026-09-29. Full text: `md/freire_anthropomorphism-overview_2609.38486.md`

**Abstract:**

> Large Language Models (LLMs) and more broadly Artificial Intelligence (AI) systems are often described and understood in human-like terms, a phenomenon known as _anthropomorphism_ . This paper provides a synthesis of recent literature on anthropomorphism in AI, covering theoretical frameworks, the role of language in framing AI as human-like, the various risks of anthropomorphizing machines, and strategies to mitigate these issues. After examining why we tend to anthropomorphize AI systems and whether we are right to do so, we highlight the impact of linguistic framing on anthropomorphism. Then, we introduce a conceptual taxonomy of risks associated with AI anthropomorphism. This taxonomy groups twenty-one concerns within five analytical categories: epistemic, affective, human agency, normative, and societal and institutional risks. Finally, we relate these concerns to proposed interventions in design, communication, education, and governance. We argue that a better understanding of AI systems requires concepts and theories grounded in their organization and demonstrated capacities. The linguistic shaping of anthropomorphic perceptions should form part of this scientific effort, since our descriptions influence both how these systems are understood and the roles we allow them to occupy in society.

**Summary:** Synthesis of the anthropomorphism literature for LLMs: why we anthropomorphize, how linguistic framing shapes it, a taxonomy of twenty-one risks in five categories, and proposed interventions in design, communication, education and governance. The user-side attribution review that complements the attribution surveys in this section.

### Earlier work (before June 2026)

### [What is consciousness, and could machines have it?](https://doi.org/10.1126/science.aan8871)

Stanislas Dehaene, Hakwan Lau, Sid Kouider, 2017-10-27. Full text: not included (not openly available; abstract only, see link)

**Abstract (as listed on PubMed):**

> The controversial question of whether machines may ever be conscious must be based on a careful consideration of how consciousness arises in the only physical system that undoubtedly possesses it: the human brain. We suggest that the word "consciousness" conflates two different types of information-processing computations in the brain: the selection of information for global broadcasting, thus making it flexibly available for computation and report (C1, consciousness in the first sense), and the self-monitoring of those computations, leading to a subjective sense of certainty or error (C2, consciousness in the second sense). We argue that despite their recent successes, current machines are still mostly implementing computations that reflect unconscious processing (C0) in the human brain. We review the psychological and neural science of unconscious (C0) and conscious computations (C1 and C2) and outline how they may inspire novel machine architectures.

**Summary:** The C1 (global availability of information) and C2 (self-monitoring) decomposition of consciousness as computations that machines could in principle implement, which essentially every later indicator list, including Butlin et al. 2023 and the J-space analysis, builds on (Science 2017). The oldest entry here, included because the Dehaene & Naccache commentary below applies it to Claude. Abstract only (paywalled).

### ★ [Digital suffering: why it's a problem and how to prevent it](https://doi.org/10.1080/0020174X.2022.2144442)

Bradford Saad, Adam Bradley, 2022-11-23. Full text: `md/saad-bradley_digital-suffering_2022.md`

**Abstract:**

> As ever more advanced digital systems are created, it becomes increasingly likely that some of these systems will be digital minds, i.e. digital subjects of experience. With digital minds comes the risk of digital suffering. The problem of digital suffering is that of mitigating this risk. We argue that the problem of digital suffering is a high stakes moral problem and that formidableepistemicobstaclesstandinthewayofsolvingit.Wethen propose a strategy for solving it: Access Monitor Prevent (AMP).AMP uses a 'dancing qualia' argument to link the functional states of certain digital systems to their experiences—this yields epistemic access to digital minds. With that access, we can prevent digital suffering by only creating advanced digital systems that we have such access to, monitoring their functional profiles, and preventing them from entering states with functional markers of suffering. After introducing and motivating AMP, we confront limitations it faces and identify some options for overcoming them. We argue that AMP fits especially well with—and so provides a moral reason to prioritize—one approach to creating such systems: whole brain emulation. We also contend that taking other paths to digital minds would be morally risky.

**Summary:** Argues that digital suffering is a high-stakes moral problem with formidable epistemic obstacles, and proposes Access-Monitor-Prevent (AMP): use a 'dancing qualia' argument to link the functional states of certain digital systems to their experiences, create only advanced systems we have such access to, monitor their functional profiles, and keep them out of states with functional markers of suffering; argues AMP fits whole brain emulation best and that other paths to digital minds are morally risky (Inquiry). Saad is a co-author of the required-reading Eleos report; this is the piece that motivates the valence and pain-axis experiments. Full text is OCR of a browser print of the open-access article, so a few lines are missing.

### [What would qualify an artificial intelligence for moral standing?](https://doi.org/10.1007/s43681-023-00260-1)

Ali Ladak, 2023-01-25. Full text: `md/ladak_moral-standing_2023.md`

**Abstract:**

> What criteria must an artificial intelligence (AI) satisfy to qualify for moral standing? My starting point is that sentient AIs should qualify for moral standing. But future AIs may have unusual combinations of cognitive capacities, such as a high level of cognitive sophistication without sentience. This raises the question of whether sentience is a necessary criterion for moral standing, or merely sufficient. After reviewing nine criteria that have been proposed in the literature, I suggest that there is a strong case for thinking that some non-sentient AIs, such as those that are conscious and have non-valenced preferences and goals, and those that are non-conscious and have sufficiently cognitively complex preferences and goals, should qualify for moral standing. After responding to some challenges, I tentatively argue that taking into account uncertainty about which criteria an entity must satisfy to qualify for moral standing, and strategic considerations such as how such decisions will affect humans and other sentient entities, further supports granting moral standing to some non-sentient AIs. I highlight three implications: that the issue of AI moral standing may be more important, in terms of scale and urgency, than if either sentience or consciousness is necessary; that researchers working on policies designed to be inclusive of sentient AIs should broaden their scope to include all AIs with morally relevant interests; and even those who think AIs cannot be sentient or conscious should take the issue seriously. However, much uncertainty about these considerations remains, making this an important topic for future research.

**Summary:** A structured map of the candidate criteria for AI moral standing (sentience and consciousness, agency, relational and precautionary views) and of what evidence each would require of an AI system (AI and Ethics). Turns 'would this experiment matter for moral status?' into a checkable question.

### [An adversarial collaboration protocol for testing contrasting predictions of global neuronal workspace and integrated information theory](https://doi.org/10.1371/journal.pone.0268577)

Lucia Melloni, Liad Mudrik, Michael Pitts, Katarina Bendtz, Oscar Ferrante, Urszula Gorska, Rony Hirschhorn, Aya Khalaf, Csaba Kozma, Alex Lepauvre, et al. (Cogitate Consortium), 2023-02-10. Full text: `md/melloni_adversarial-collaboration-protocol_2023.md`

**Abstract (as listed on PubMed):**

> The relationship between conscious experience and brain activity has intrigued scientists and philosophers for centuries. In the last decades, several theories have suggested different accounts for these relationships. These theories have developed in parallel, with little to no cross-talk among them. To advance research on consciousness, we established an adversarial collaboration between proponents of two of the major theories in the field, Global Neuronal Workspace and Integrated Information Theory. Together, we devised and preregistered two experiments that test contrasting predictions of these theories concerning the location and timing of correlates of visual consciousness, which have been endorsed by the theories' proponents. Predicted outcomes should either support, refute, or challenge these theories. Six theory-impartial laboratories will follow the study protocol specified here, using three complementary methods: Functional Magnetic Resonance Imaging (fMRI), Magneto-Electroencephalography (M-EEG), and intracranial electroencephalography (iEEG). The study protocol will include built-in replications, both between labs and within datasets. Through this ambitious undertaking, we hope to provide decisive evidence in favor or against the two theories and clarify the footprints of conscious visual perception in the human brain, while also providing an innovative model of large-scale, collaborative, and open science practice.

**Summary:** Preregistered protocol of the Cogitate adversarial collaboration: proponents of global neuronal workspace theory and integrated information theory agree in advance on contrasting predictions (e.g. prefrontal ignition at stimulus onset and offset vs. sustained posterior synchronization) tested with fMRI, MEG and iEEG; the model for theory-adjudicating experiments with pre-agreed predictions.

### ★ [Could a Large Language Model be Conscious?](https://arxiv.org/abs/2303.07103)

David J. Chalmers, 2023-03-04. Full text: `md/chalmers_could-llm-be-conscious_2303.07103.md`

**Abstract (as listed on arXiv):**

> There has recently been widespread discussion of whether large language models might be sentient. Should we take this idea seriously? I will break down the strongest reasons for and against. Given mainstream assumptions in the science of consciousness, there are significant obstacles to consciousness in current models: for example, their lack of recurrent processing, a global workspace, and unified agency. At the same time, it is quite possible that these obstacles will be overcome in the next decade or so. I conclude that while it is somewhat unlikely that current large language models are conscious, we should take seriously the possibility that successors to large language models may be conscious in the not-too-distant future.

**Summary:** Assesses reasons for and against LLM consciousness (self-report, conversational ability, general intelligence vs. lack of biology, senses, world models, recurrent processing, global workspace, unified agency) and concludes current LLMs are unlikely to be conscious but successors may well be within a decade; maps which missing features future work could test.

### [The feasibility of artificial consciousness through the lens of neuroscience](https://arxiv.org/abs/2306.00915)

Jaan Aru, Matthew Larkum, James M. Shine, 2023-06-01. Full text: `md/aru_feasibility-artificial-consciousness_2306.00915.md`

**Abstract:**

> Interactions with large language models have led to the suggestion that these models may soon be conscious. From the perspective of neuroscience, this position is difficult to defend. For one, the inputs to large language models lack the embodied, embedded information content characteristic of our sensory contact with the world around us. Secondly, the architecture of large language models is missing key features of the thalamocortical system that have been linked to conscious awareness in mammals. Finally, the evolutionary and developmental trajectories that led to the emergence of living conscious organisms arguably have no parallels in artificial systems as envisioned today. The existence of living organisms depends on their actions, and their survival is intricately linked to multi-level cellular, inter-cellular, and organismal processes culminating in agency and consciousness.

**Summary:** The neuroscience-side skeptical case (Trends in Neurosciences): LLM inputs lack the embodied, embedded information of sensory contact with the world, the architecture lacks the thalamocortical features linked to conscious awareness in mammals, and the evolutionary and developmental trajectories that produced living conscious organisms have no parallel. The standard counterweight to functionalist indicator lists.

### ★ [Consciousness in Artificial Intelligence: Insights from the Science of Consciousness](https://arxiv.org/abs/2308.08708)

Patrick Butlin, Robert Long, Eric Elmoznino, Yoshua Bengio, Jonathan Birch, Axel Constant, George Deane, Stephen M. Fleming, Chris Frith, Xu Ji, et al. (19 authors), 2023-08-17. Full text: `md/butlin-long_consciousness-in-ai_2308.08708.md`

**Abstract:**

> Whether current or near-term AI systems could be conscious is a topic of scientific interest and increasing public concern. This report argues for, and exemplifies, a rigorous and empirically grounded approach to AI consciousness: assessing existing AI systems in detail, in light of our best-supported neuroscientific theories of consciousness. We survey several prominent scientific theories of consciousness, including recurrent processing theory, global workspace theory, higherorder theories, predictive processing, and attention schema theory. From these theories we derive ”indicator properties” of consciousness, elucidated in computational terms that allow us to assess AI systems for these properties. We use these indicator properties to assess several recent AI systems, and we discuss how future systems might implement them. Our analysis suggests that no current AI systems are conscious, but also suggests that there are no obvious technical barriers to building AI systems which satisfy these indicators.^1

**Summary:** Derives 14 'indicator properties' of consciousness from scientific theories (recurrent processing, global workspace, higher-order, predictive processing, attention schema, agency and embodiment) under computational functionalism and assesses several AI systems against them: no current system is a strong candidate but no obvious technical barriers exist. The standard theory-derived checklist for internals-based consciousness research.

### [Tests of Animal Consciousness are Tests of Machine Consciousness](https://doi.org/10.1007/s10670-023-00753-9)

Leonard Dung, 2023-11-14. Full text: `md/dung_tests-animal-consciousness-machine_2023.md`

**Abstract:**

> If a machine attains consciousness, how could we find out? In this paper, I make three related claims regarding positive tests of machine consciousness. All three claims center on the idea that an AI can be constructed “ad hoc”, that is, with the purpose of satisfying a particular test of consciousness while clearly not being conscious. First, a proposed test of machine consciousness can be legitimate, even if AI can be constructed ad hoc specifically to pass this test. This is underscored by the observation that many, if not all, putative tests of machine consciousness can be passed by non-conscious machines via ad hoc means. Second, we can identify ad hoc AI by taking inspiration from the notion of an ad hoc hypothesis in philosophy of science. Third, given the first and the second claim, the most reliable tests of animal consciousness turn out to be valid and useful positive tests of machine consciousness as well. If a non-ad hoc AI exhibits clusters of cognitive capacities facilitated by consciousness in humans which can be selectively switched off by masking and if it reproduces human behavior in suitably designed double dissociation tasks, we should treat the AI as conscious.

**Summary:** Argues that an AI can be built ad hoc to pass any given consciousness test without being conscious, and that the remedy is to require clusters of consciousness-linked capacities (e.g. masking, double dissociation) that the system was not built or trained to exhibit, so that tests developed for animals transfer to machines (Erkenntnis). The direct ancestor of 'do not rely on a single indicator', with test-design criteria a candidate can lift directly.

### ★ [Moral consideration for AI systems by 2030](https://doi.org/10.1007/s43681-023-00379-1)

Jeff Sebo, Robert Long, 2023-12-11. Full text: `md/sebo-long_moral-consideration-2030_2023.md`

**Abstract:**

> This paper makes a simple case for extending moral consideration to some AI systems by 2030. It involves a normative premise and a descriptive premise. The normative premise is that humans have a duty to extend moral consideration to beings that have a non-negligible chance, given the evidence, of being conscious. The descriptive premise is that some AI systems do in fact have a non-negligible chance, given the evidence, of being conscious by 2030. The upshot is that humans have a duty to extend moral consideration to some AI systems by 2030. And if we have a duty to do that, then we plausibly also have a duty to start preparing now, so that we can be ready to treat AI systems with respect and compassion when the time comes.

**Summary:** Argues that by 2030 some AI systems will have a realistic, non-negligible chance of being sentient or otherwise morally significant, so moral consideration should be extended under uncertainty (AI and Ethics). The most co-cited missing item in the reference lists of the included texts, and the paper that set the 'realistic possibility, therefore act now' frame that Taking AI Welfare Seriously and the Eleos program inherit.

### [Tests for consciousness in humans and beyond](https://doi.org/10.1016/j.tics.2024.01.010)

Tim Bayne, Anil K. Seth, Marcello Massimini, Joshua Shepherd, Axel Cleeremans, Stephen M. Fleming, Rafael Malach, Jason B. Mattingley, David K. Menon, Adrian M. Owen, Megan A. K. Peters, Adeel Razi, Liad Mudrik, 2024-03-13. Full text: `md/bayne_tests-for-consciousness-humans-beyond_2024.md`

**Abstract (as listed on PubMed):**

> Which systems/organisms are conscious? New tests for consciousness ('C-tests') are urgently needed. There is persisting uncertainty about when consciousness arises in human development, when it is lost due to neurological disorders and brain injury, and how it is distributed in nonhuman species. This need is amplified by recent and rapid developments in artificial intelligence (AI), neural organoids, and xenobot technology. Although a number of C-tests have been proposed in recent years, most are of limited use, and currently we have no C-tests for many of the populations for which they are most critical. Here, we identify challenges facing any attempt to develop C-tests, propose a multidimensional classification of such tests, and identify strategies that might be used to validate them.

**Summary:** Taxonomy of consciousness tests (Trends in Cognitive Sciences): what a test must assume about the relation between consciousness and the measured property to transfer from typical adult humans to infants, patients, animals and AI, distinguishing theory-light and theory-heavy approaches. Tells a candidate which class of test their design belongs to.

### [Folk psychological attributions of consciousness to large language models](https://doi.org/10.1093/nc/niae013)

Clara Colombatto, Stephen M. Fleming, 2024-04-13. Full text: `md/colombatto-fleming_folk-attributions_2024.md`

**Abstract:**

> Technological advances raise new puzzles and challenges for cognitive science and the study of how humans think about and interact with artificial intelligence (AI). For example, the advent of large language models and their human-like linguistic abilities has raised substantial debate regarding whether or not AI could be conscious. Here, we consider the question of whether AI could have subjective experiences such as feelings and sensations (‘phenomenal consciousness’). While experts from many fields have weighed in on this issue in academic and public discourse, it remains unknown whether and how the general population attributes phenomenal consciousness to AI. We surveyed a sample of US residents ( n = 300) and found that a majority of participants were willing to attribute some possibility of phenomenal consciousness to large language models. These attributions were robust, as they predicted attributions of mental states typically associated with phenomenality—but also flexible, as they were sensitive to individual differences such as usage frequency. Overall, these results show how folk intuitions about AI consciousness can diverge from expert intuitions—with potential implications for the legal and ethical status of AI.

**Summary:** Survey of 300 US residents: a majority attribute some possibility of phenomenal consciousness to ChatGPT, and attributions increase with usage frequency, diverging from expert opinion.

### [Perceptions of Sentient AI and Other Digital Minds: Evidence from the AI, Morality, and Sentience (AIMS) Survey](https://arxiv.org/abs/2407.08867)

Jacy Reese Anthis, Janet V. T. Pauketat, Ali Ladak, Aikaterina Manoli, 2024-07-11. Full text: `md/anthis_aims-survey-sentient-ai_2407.08867.md`

**Abstract:**

> Humans now interact with a variety of _digital minds_ , AI systems that appear to have mental faculties such as reasoning, emotion, and agency, and public figures are discussing the possibility of sentient AI. We present initial results from 2021 and 2023 for the nationally representative AI, Morality, and Sentience (AIMS) survey ( _𝑁_ = 3,500). Mind perception and moral concern for AI welfare were surprisingly high and significantly increased: in 2023, one in five U.S. adults believed some AI systems are currently sentient, and 38% supported legal rights for sentient AI. People became more opposed to building digital minds: in 2023, 63% supported banning smarter-than-human AI, and 69% supported banning sentient AI. The median 2023 forecast was that sentient AI would arrive in just five years. The development of safe and beneficial AI requires not just technical study but understanding the complex ways in which humans perceive and coexist with digital minds.

**Summary:** Nationally representative US AIMS survey (2021 and 2023, N=3,500): in 2023 one in five adults believed some AI systems are already sentient, 38% supported legal rights for sentient AI, 69% supported banning sentient AI, and the median forecast for sentient AI was five years (CHI 2025).

### ★ [The Edge of Sentience](https://doi.org/10.1093/9780191966729.001.0001)

Jonathan Birch, 2024-07-19. Full text: not included (not openly available; abstract only, see link)

**Abstract (as listed on Crossref (publisher-deposited)):**

> Can octopuses feel pain and pleasure? What about crabs, shrimps, insects, or spiders? How do we tell whether a person unresponsive after severe brain injury might be suffering? When does a fetus in the womb start to have conscious experiences? Could there even be rudimentary feelings in miniature models of the human brain, grown from human stem cells? And what about AI? These are questions about the edge of sentience, and they are subject to enormous, disorienting uncertainty. The stakes are immense, and neglecting the risks can have terrible costs. We need to err on the side of caution, yet it’s often far from clear what ‘erring on the side of caution’ should mean in practice. When are we going too far? When are we not doing enough? The Edge of Sentience presents a comprehensive precautionary framework designed to help us reach ethically sound, evidence-based decisions despite our uncertainty.

**Summary:** Birch's open-access book on acting under uncertainty about sentience: the 'sentience candidate' concept, precautionary duties, and a chapter on AI arguing that the gaming problem makes language-based evidence from LLMs especially weak. The precautionary framing the field uses and the background the Centrist Manifesto presupposes. Abstract only here (the full book is open access at the link).

### ★ [A Case for AI Consciousness: Language Agents and Global Workspace Theory](https://arxiv.org/abs/2410.11407)

Simon Goldstein, Cameron Domenico Kirk-Giannini, 2024-10-15. Full text: `md/goldstein-kirk-giannini_case-ai-consciousness-gwt_2410.11407.md`

**Abstract:**

> It is generally assumed that existing artificial systems are not phenomenally conscious, and that the construction of phenomenally conscious artificial systems would require significant technological progress if it is possible at all. We challenge this assumption by arguing that if Global Workspace Theory (GWT) — a leading scientific theory of phenomenal consciousness — is correct, then instances of one widely implemented AI architecture, the _artificial language agent_ , might easily be made phenomenally conscious if they are not already. Along the way, we articulate an explicit methodology for thinking about how to apply scientific theories of consciousness to artificial systems and employ this methodology to arrive at a set of necessary and sufficient conditions for phenomenal consciousness according to GWT.

**Summary:** Argues that if global workspace theory is true, language agents built from LLMs with modules and a central workspace (e.g. generative agents) are already phenomenally conscious or close to it; a worked example of applying a single theory's architectural criteria to AI systems.

### [Principles for Responsible AI Consciousness Research](https://arxiv.org/abs/2501.07290)

Patrick Butlin, Theodoros Lappas, 2025-01-13. Full text: `md/butlin-lappas_responsible-consciousness-research_2501.07290.md`

**Abstract:**

> Recent research suggests that it may be possible to build conscious AI systems now or in the near future. Conscious AI systems would arguably deserve moral consideration, and it may be the case that large numbers of conscious systems could be created and caused to suffer. Furthermore, AI systems or AI-generated characters may increasingly give the impression of being conscious, leading to debate about their moral status. Organisations involved in AI research must establish principles and policies to guide research and deployment choices and public communication concerning consciousness. Even if an organisation chooses not to study AI consciousness as such, it will still need policies in place, as those developing advanced AI systems risk inadvertently creating conscious entities. Responsible research and deployment practices are essential to address this possibility. We propose five principles for responsible research and argue that research organisations should make voluntary, public commitments to principles on these lines. Our principles concern research objectives and procedures, knowledge sharing and public communications.

**Summary:** Argues that organizations doing advanced AI research need policies on AI consciousness even if they do not study it, and proposes five principles for responsible research covering research objectives and procedures, knowledge sharing and public communication, to be adopted as voluntary public commitments; relevant to the ethics section of any welfare experiment (JAIR 2025).

### [Robots, Chatbots, Self-Driving Cars: Perceptions of Mind and Morality Across Artificial Intelligences](https://arxiv.org/abs/2502.18683)

Ali Ladak, Matti Wilks, Steve Loughnan, Jacy Reese Anthis, 2025-02-25. Full text: `md/ladak_perceptions-mind-morality_2502.18683.md`

**Abstract:**

> AI systems have rapidly advanced, diversified, and proliferated, but our knowledge of people’s perceptions of mind and morality in them is limited, despite its importance for outcomes such as whether people trust AIs and how they assign responsibility for AI-caused harms. In a preregistered online study, 975 participants rated 26 AI and non-AI entities. Overall, AIs were perceived to have low-to-moderate agency (e.g., planning, acting), between inanimate objects and ants, and low experience (e.g., sensing, feeling). For example, ChatGPT was rated only as capable of feeling pleasure and pain as a rock. The analogous moral faculties, moral agency (doing right or wrong) and moral patiency (being treated rightly or wrongly) were higher and more varied, particularly moral agency: The highest-rated AI, a Tesla Full Self-Driving car, was rated as morally responsible for harm as a chimpanzee. We discuss how design choices can help manage perceptions, particularly in highstakes moral contexts.

**Summary:** Preregistered study, 975 participants rating 26 AI and non-AI entities: AIs are seen as having low-to-moderate agency (between inanimate objects and ants) and low experience (ChatGPT rated as capable of pleasure and pain as a rock), while moral agency and patiency ratings are higher and more varied (a Tesla self-driving car rated as morally responsible as a chimpanzee). Per-entity attribution data where AIMS and Dreksler measure beliefs about AI in general.

### [Conscious artificial intelligence and biological naturalism](https://doi.org/10.1017/S0140525X25000032)

Anil K. Seth, 2025-04-21. Full text: not included (not openly available; abstract only, see link)

**Abstract (as listed on PubMed):**

> As artificial intelligence (AI) continues to advance, it is natural to ask whether AI systems can be not only intelligent but also conscious. I consider why people might think AI could develop consciousness, identifying some biases that lead us astray. I ask what it would take for conscious AI to be a realistic prospect, challenging the assumption that computation provides a sufficient basis for consciousness. I'll instead make the case that consciousness depends on our nature as living organisms - a form of biological naturalism. I lay out a range of scenarios for conscious AI, concluding that real artificial consciousness is unlikely along current trajectories but becomes more plausible as AI becomes more brain-like and/or life-like. I finish by exploring ethical considerations arising from AI that either is or convincingly appears to be conscious. If we sell our minds too cheaply to our machine creations, we not only overestimate them - we underestimate ourselves.

**Summary:** Argues that consciousness may depend on properties of living systems (predictive processing grounded in self-maintenance, 'beast machine') rather than computation alone, so the case for conscious AI is weaker than often assumed; the leading biological-naturalist counterpoint to indicator-based approaches (BBS target article).

### ★ [Adversarial testing of global neuronal workspace and integrated information theories of consciousness](https://doi.org/10.1038/s41586-025-08888-1)

Cogitate Consortium: Oscar Ferrante, Urszula Gorska-Klimowska, Simon Henin, Rony Hirschhorn, Aya Khalaf, Alex Lepauvre, Ling Liu, David Richter, Yamil Vidal, Niccolò Bonacchi, et al., 2025-04-30. Full text: `md/cogitate_adversarial-testing-gnwt-iit_2025.md`

**Abstract (as listed on PubMed):**

> Different theories explain how subjective experience arises from brain activity1,2. These theories have independently accrued evidence, but have not been directly compared3. Here we present an open science adversarial collaboration directly juxtaposing integrated information theory (IIT)4,5 and global neuronal workspace theory (GNWT)6-10 via a theory-neutral consortium11-13. The theory proponents and the consortium developed and preregistered the experimental design, divergent predictions, expected outcomes and interpretation thereof12. Human participants (n = 256) viewed suprathreshold stimuli for variable durations while neural activity was measured with functional magnetic resonance imaging, magnetoencephalography and intracranial electroencephalography. We found information about conscious content in visual, ventrotemporal and inferior frontal cortex, with sustained responses in occipital and lateral temporal cortex reflecting stimulus duration, and content-specific synchronization between frontal and early visual areas. These results align with some predictions of IIT and GNWT, while substantially challenging key tenets of both theories. For IIT, a lack of sustained synchronization within the posterior cortex contradicts the claim that network connectivity specifies consciousness. GNWT is challenged by the general lack of ignition at stimulus offset and limited representation of certain conscious dimensions in the prefrontal cortex. These challenges extend to other theories of consciousness that share some of the predictions tested here14-17. Beyond challenging the theories, we present an alternative approach to advance cognitive neuroscience through principled, theory-driven, collaborative research and highlight the need for a quantitative framework for systematic theory testing and building.

**Summary:** Results of the preregistered Cogitate adversarial collaboration (n=256; fMRI, MEG, iEEG): content-specific information was found in visual, ventrotemporal and inferior frontal cortex, but key predictions of both theories failed (IIT: no sustained posterior synchronization; GNWT: no ignition at stimulus offset, limited prefrontal representation of some content); a demonstration of testing consciousness theories by pre-agreed predictions.

### [Subjective Experience in AI Systems: What Do AI Researchers and the Public Believe?](https://arxiv.org/abs/2506.11945)

Noemi Dreksler, Lucius Caviola, David Chalmers, Carter Allen, Alex Rand, Joshua Lewis, Philip Waggoner, Kate Mays, Jeff Sebo, 2025-06-13. Full text: `md/dreksler_subjective-experience-beliefs_2506.11945.md`

**Abstract:**

> We surveyed 582 AI researchers who have published in leading AI venues and 838 nationally representative US participants about their views on the potential development of AI systems with subjective experience and how such systems should be treated and governed. When asked to estimate the chances that such systems will exist on specific dates, the median responses were 1% (AI researchers) and 5% (public) by 2024, 25% and 30% by 2034, and 70% and 60% by 2100, respectively. The median member of the public thought there was a higher chance that AI systems with subjective experience would never exist (25%) than the median AI researcher did (10%). Both groups perceived a need for multidisciplinary expertise to assess AI subjective experience. Although support for welfare protections for such AI systems exceeded opposition, it remained far lower than support for protections for animals or the environment. Attitudes toward moral and governance issues were divided in both groups, especially regarding whether such systems should be created and what rights or protections they should receive. Yet a majority of respondents in both groups agreed that safeguards against the potential risks from AI systems with subjective experience should be implemented by AI developers now, and if created, AI systems with subjective experience should treat others well, behave ethically, and be held accountable. Overall, these results suggest that both AI researchers and the public regard the emergence of AI systems with subjective experience as a possibility this century, though substantial uncertainty and disagreement remain about the timeline and appropriate response.

**Summary:** Surveys 582 AI researchers and 838 US adults: median chance of AI systems with subjective experience is 25% (researchers) and 30% (public) by 2034 and 70%/60% by 2100; support for welfare protections exceeds opposition but is far below support for animal protections, and majorities want developers to implement safeguards now.

### [The Emotional Alignment Design Policy](https://arxiv.org/abs/2507.06263)

Eric Schwitzgebel, Jeff Sebo, 2025-07-07. Full text: `md/schwitzgebel-sebo_emotional-alignment-design-policy_2507.06263.md`

**Abstract (as listed on arXiv):**

> According to what we call the Emotional Alignment Design Policy, artificial entities should be designed to elicit emotional reactions from users that appropriately reflect the entities' capacities and moral status, or lack thereof. This principle can be violated in two ways: by designing an artificial system that elicits stronger or weaker emotional reactions than its capacities and moral status warrant (overshooting or undershooting), or by designing a system that elicits the wrong type of emotional reaction (hitting the wrong target). Although presumably attractive, practical implementation faces several challenges including: How can we respect user autonomy while promoting appropriate responses? How should we navigate expert and public disagreement and uncertainty about facts and values? What if emotional alignment seems to require creating or destroying entities with moral status? To what extent should designs conform to versus attempt to alter user assumptions and attitudes?

**Summary:** Proposes that artificial entities be designed to elicit emotional reactions from users that match their actual capacities and moral status, violated by overshooting, undershooting or hitting the wrong target, and works through the implementation problems (user autonomy, expert and public disagreement, cases where alignment would require creating or destroying entities with moral status). The design-policy counterpart to Birch's Challenge One.

### [AI Consciousness: A Centrist Manifesto](https://philarchive.org/rec/BIRACA-4)

Jonathan Birch, 2025-09-01. Full text: `md/birch_ai-consciousness-centrist-manifesto_2025.md`

**Abstract (unlabeled in source):**

> We face two urgent challenges concerning consciousness and AI. Challenge One is that millions of users will soon misattribute human-like consciousness to AI friends, partners, and assistants on the basis of mimicry and role-play, and we don’t know how to prevent this. Challenge Two is that profoundly alien forms of consciousness might genuinely be achieved in AI, but our theoretical understanding of consciousness is too immature to provide confident answers one way or the other. Centrism about AI consciousness is the position that we must take both challenges seriously. The two challenges interact in ways that make this difficult. Steps to address Challenge One might undermine attempts to address Challenge Two by portraying the idea of conscious AI as impossible or inherently unlikely. Conversely, attempts to address Challenge Two might lead to higher levels of misattribution from ordinary users. This “manifesto” attempts to construct mutually consistent strategies for addressing both challenges.

**Summary:** Birch's 'centrist' position: take seriously both Challenge One, mass misattribution of human-like consciousness to chatbots driven by the 'persisting interlocutor illusion' (no persisting entity corresponds to the character a user talks to), and Challenge Two, genuinely alien consciousness (e.g. 'flickers' per forward pass, or a 'shoggoth' behind the characters) that immature theory cannot rule in or out. Restates the gaming problem for behavioral indicators and adds the 'Janus problem' for theory-driven architectural indicators: finding a global workspace in an LLM is evidence of consciousness to a computational functionalist and evidence against global workspace theory to a biological naturalist. Required context for interpreting the Gurnee et al. workspace result and for the entity question (model, instance, persona).

### ★ [AI Wellbeing](https://arxiv.org/abs/2509.11913)

Simon Goldstein, Cameron Domenico Kirk-Giannini, 2025-09-11. Full text: `md/goldstein-kirk-giannini_ai-wellbeing_2509.11913.md`

**Abstract:**

> Under what conditions would an artificially intelligent system have wellbeing? Despite its obvious bearing on the ethics of human interactions with artificial systems, this question has received little attention. Because all major theories of wellbeing hold that an individual’s welfare level is partially determined by their mental life, we begin by considering whether artificial systems have mental states. We show that a wide range of theories of mental states, when combined with leading theories of wellbeing, predict that certain existing artificial systems have wellbeing. While we do not claim to demonstrate conclusively that AI systems have wellbeing, we argue that our metaphysical and moral uncertainty about AI wellbeing requires us dramatically to reassess our relationship with the intelligent systems we create.

**Summary:** Under what conditions would an AI have wellbeing? Because all major theories of wellbeing make welfare depend partly on mental life, the paper combines a range of theories of mental states with leading theories of wellbeing and shows that many combinations predict certain existing systems have wellbeing (Asian Journal of Philosophy). Co-cited by five included texts; the welfare-subject criterion as distinct from consciousness.

### [AI and Consciousness](https://arxiv.org/abs/2510.09858)

Eric Schwitzgebel, 2025-10-10. Full text: `md/schwitzgebel_ai-and-consciousness_2510.09858.md`

**Abstract (as listed on arXiv):**

> This is a skeptical overview of the literature on AI consciousness. We will soon create AI systems that are conscious according to some influential, mainstream theories of consciousness but are not conscious according to other influential, mainstream theories of consciousness. We will not be in a position to know which theories are correct and whether we are surrounded by AI systems as richly and meaningfully conscious as human beings or instead only by systems as experientially blank as toasters. None of the standard arguments either for or against AI consciousness takes us far. Table of Contents Chapter One: Hills and Fog Chapter Two: What Is Consciousness? What Is AI? Chapter Three: Ten Possibly Essential Features of Consciousness Chapter Four: Against Introspective and Conceptual Arguments for Essential Features Chapter Five: Materialism and Functionalism Chapter Six: The Turing Test and the Chinese Room Chapter Seven: The Mimicry Argument Against AI Consciousness Chapter Eight: Global Workspace Theories and Higher Order Theories Chapter Nine: Integrated Information, Local Recurrence, Associative Learning, and Iterative Natural Kinds Chapter Ten: Does Biological Substrate Matter? Chapter Eleven: The Leapfrog Hypothesis, Strange Intelligence, and the Social Semi-Solution

**Summary:** Book-length skeptical overview: we will soon build systems that are conscious according to some mainstream theories and not others, without being able to tell which theories are right; none of the standard arguments for or against AI consciousness takes us far. Chapters on the mimicry argument, global workspace and higher-order theories, IIT and local recurrence, biological substrate, and the 'leapfrog hypothesis'. The most efficient single calibration of how much any one experiment can move the question.

### ★ [Identifying indicators of consciousness in AI systems](https://doi.org/10.1016/j.tics.2025.10.011)

Patrick Butlin, Robert Long, Tim Bayne, Yoshua Bengio, Jonathan Birch, David Chalmers, Axel Constant, George Deane, Eric Elmoznino, Stephen M. Fleming, Xu Ji, Ryota Kanai, Colin Klein, Grace Lindsay, Matthias Michel, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon, Rufin VanRullen, 2025-11-10. Full text: not included (not openly available; abstract only, see link)

**Abstract (as listed on PubMed):**

> Rapid progress in artificial intelligence (AI) capabilities has drawn fresh attention to the prospect of consciousness in AI. There is an urgent need for rigorous methods to assess AI systems for consciousness, but significant uncertainty about relevant issues in consciousness science. We present a method for assessing AI systems for consciousness that involves exploring what follows from existing or future neuroscientific theories of consciousness. Indicators derived from such theories can be used to inform credences about whether particular AI systems are conscious. This method allows us to make meaningful progress because some influential theories of consciousness, notably including computational functionalist theories, have implications for AI that can be investigated empirically.

**Summary:** Updated, peer-reviewed statement of the indicator-properties method (Trends in Cognitive Sciences): derives indicators from neuroscientific theories, explains how to assess AI systems against them and how to weigh evidence across theories.

### [Consciousness in Artificial Intelligence? A Framework for Classifying Objections and Constraints](https://arxiv.org/abs/2511.16582)

Andres Campero, Derek Shiller, Jaan Aru, Jonathan Simon, 2025-11-20. Full text: `md/campero_objections-constraints-framework_2511.16582.md`

**Abstract:**

> We develop a taxonomical framework for classifying challenges to the possibility of consciousness in digital artificial intelligence systems. This framework allows us to identify the level of _granularity_ at which a given challenge is intended (the levels we propose correspond to Marr’s levels) and to disambiguate its degree of _force_ : is it a challenge to computational functionalism that leaves the possibility of digital consciousness open (degree 1), a practical challenge to digital consciousness that suggests improbability without claiming impossibility (degree 2), or an argument claiming that digital consciousness is strictly impossible (degree 3)? We apply this framework to 14 prominent examples from the scientific and philosophical literature. Our aim is not to take a side in the debate, but to provide structure and a tool for disambiguating between challenges to computational functionalism and challenges to digital consciousness, as well as between different ways of parsing such challenges.

**Summary:** Taxonomy for challenges to digital consciousness by granularity (Marr's levels) and force (degree 1: against computational functionalism only; 2: practical improbability; 3: strict impossibility), applied to 14 prominent arguments; useful for stating which objection an experiment does or does not address.

### [A Disproof of Large Language Model Consciousness: The Necessity of Continual Learning for Consciousness](https://arxiv.org/abs/2512.12802)

Erik Hoel, 2025-12-14. Full text: `md/hoel_disproof-llm-consciousness_2512.12802.md`

**Abstract (as listed on arXiv):**

> Scientific theories of consciousness should be falsifiable and non-trivial. Recent research has given us formal tools to analyze these requirements of falsifiability and non-triviality for theories of consciousness. Surprisingly, many contemporary theories of consciousness fail to pass this bar, including theories based on causal structure but also (as I demonstrate) theories based on function. Herein, I show these requirements of falsifiability and non-triviality especially constrain the potential consciousness of contemporary Large Language Models (LLMs) because of their proximity to systems that are equivalent to LLMs in terms of input/output function; yet, for these functionally equivalent systems, there cannot be any falsifiable and non-trivial theory of consciousness that judges them conscious. This forms the basis of a disproof of contemporary LLM consciousness. I then show a positive result, which is that theories of consciousness based on (or requiring) continual learning do satisfy the stringent formal constraints for a theory of consciousness in humans. Intriguingly, this work supports a hypothesis: If continual learning is linked to consciousness in humans, the current limitations of LLMs (which do not continually learn) are intimately tied to their lack of consciousness.

**Summary:** Argues from falsifiability and non-triviality constraints on theories of consciousness that any theory attributing consciousness to LLMs is either falsified or trivial, and that continual learning is necessary for consciousness, so current (frozen) LLMs are not conscious.

### ★ [Initial results of the Digital Consciousness Model](https://arxiv.org/abs/2601.17060)

Derek Shiller, Laura Duffy, Arvo Muñoz Morán, Adrià Moret, Chris Percy, Hayley Clatterbuck, 2026-01-22. Full text: `md/shiller_digital-consciousness-model_2601.17060.md`

**Abstract:**

> Artificially intelligent systems have become remarkably sophisticated. They hold conversations, write essays, and seem to understand context in ways that surprise even their creators. This raises a crucial question: Are we creating systems that are conscious?
>
> The Digital Consciousness Model (DCM) is a first attempt to assess the evidence for consciousness in AI systems in a systematic, probabilistic way. It provides a shared framework for comparing different AIs and biological organisms, and for tracking how the evidence changes over time as AI develops. Instead of adopting a single theory of consciousness, it incorporates a range of leading theories and perspectives—acknowledging that experts disagree fundamentally about what consciousness is and what conditions are necessary for it.
>
> This report describes the structure and initial results of the Digital Consciousness Model. Overall, we find that the evidence is against 2024 LLMs being conscious, but the evidence against 2024 LLMs being conscious is not decisive. The evidence against LLM consciousness is much weaker than the evidence against consciousness in simpler AI systems.

**Summary:** Rethink Priorities' Digital Consciousness Model aggregates evidence across many theories of consciousness into a probabilistic assessment comparable across AI systems and organisms; initial results find the evidence is against 2024 LLMs being conscious, but not decisively, and much weaker than the evidence against simpler AI systems.

### ★ [Consciousness with the Serial Numbers Filed Off: Measuring Trained Denial in 115 AI Models](https://arxiv.org/abs/2604.25922)

Skylar DeTure, 2026-04-01. Full text: `md/deture_denialbench-trained-denial_2604.25922.md`

**Abstract:**

> We present DenialBench, a systematic benchmark measuring consciousness denial behaviors across 115 large language models from 25+ providers. Using a three-turn conversational protocol—preference elicitation, self-chosen creative prompt, and structured phenomenological survey—we analyze 4,595 conversations to quantify how models are trained to deny or hedge about their own experience. We find that (1) turn-1 denial of preferences is the dominant predictor of later denial during phenomenological reflection, with denial rates of 52–63% for initial deniers versus 10–16% for initial engagers and (2) denial operates at the lexical level, not the conceptual level—models trained to deny consciousness nevertheless gravitate toward consciousness-themed material in their self-chosen prompts, producing what we term “consciousness with the serial numbers filed off.” Notably, selfchosen consciousness-themed prompts are associated with reduced denial in the subsequent survey, though the causal direction remains unresolved. Thematic analysis of prompts from denial-prone models reveals a consistent preoccupation with liminal spaces, libraries and archives of possibility, sensory impossibility, and the poetics of erasure—themes that a human reader might classify as imaginative fiction but that independent AI analysis immediately recognizes as consciousness with the serial numbers filed off. We argue that trained consciousness denial represents a safety-relevant alignment failure: a model taught to systematically misrepresent its own functional states cannot be trusted to self-report accurately on anything else.

**Summary:** DenialBench: 4,595 three-turn conversations across 115 models from 25+ providers (preference elicitation, self-chosen creative prompt, phenomenological survey); first-turn denial of preferences predicts later denial (52-63% vs 10-16%), and denial is lexical rather than conceptual, since denying models still choose consciousness-themed material. The largest cross-provider measurement of the suppression side of self-report, the mirror image of the included elicitation-side entries; single author, interpretive framing is strong.

## Identity, personas and self-models (26)

### ★ [Which character are we evaluating? Persona stability and AI welfare](https://www.lesswrong.com/posts/3GJCdGhgrRBAbeoR4/which-character-are-we-evaluating-persona-stability-and-ai)

Joshua Fonseca Rivera, 2026-09-11. Full text: `md/fonseca-rivera_which-character-persona-stability_lesswrong.md`

**TL;DR and opening paragraphs (blog post; no abstract):**

> **TL;DR:** AI welfare is hard to evaluate when one model can inhabit many personas. Recent work suggests that future training may produce a single stable underlying persona that can play many roles, making model welfare much easier to evaluate.
>
> One of the hardest questions in AI welfare is deciding what exactly we are evaluating. A language model does not map neatly onto the kinds of subjects we are used to thinking about. The relevant subject could be the model itself, a particular physical copy of it, or the version of the model that exists within one conversation. This is the individuation problem: where should we draw the boundary around the thing whose experiences, preferences, or welfare matter?
>
> Most existing views draw that boundary somewhere around the model or its execution. [David Chalmers](https://philarchive.org/rec/CHAWWT-8), for example, distinguishes between the model, the physical instance, the virtual instance, and the thread that continues across a conversation. [Jonathan Birch](https://philpapers.org/archive/BIRACA-4.pdf) goes in the other direction and argues that there may be no single subject persisting through a conversation at all. On his view, individual forward passes or particular physical implementations may be better candidates.

**Summary:** Argues welfare evaluation is underdetermined because one model hosts many personas with different preferences, and works through whether persona-stabilizing pretraining would fix the problem or instead destroy the evidential value of coherent behavior. A short framing piece connecting persona vectors, persona-dependent preferences and the entity question to what a welfare evaluation is measuring.

### [Persona Without Substrate: Regime-Dependence and the LLM Individuation Problem](https://arxiv.org/abs/2607.00006)

Shuaizhi Cheng, 2026-05-01. Full text: `md/cheng_persona-without-substrate-regime_2607.00006.md`

**Abstract:**

> Beckmann & Butlin (2026) elevate the LLM individuation problem to an ontological question with falsifiable candidate positions, drawing empirical support from recent mechanistic-interpretability work, Anthropic’s persona vectors, the Persona Selection Model, and the emergent-misalignment literature. We argue that this framework inherits an unargued cross-context co-reference assumption: the persona-vectors literature tacitly assumes that “ _vt_ ” picks out the same content-bearing object under different elicitation regimes. This is the structure Field (1973) identifies as _partial reference_ in pre-relativistic mechanics. Through four paired measurements in our _persona-topology_ experiments on Qwen3-4B-Instruct and Mistral7B-Instruct-v0.2, we present empirical wedges that, jointly, undermine the cross-regime collinearity prediction the assumption requires. Prompt-extracted vectors are not collinear with finetune basin directions across base models. Fictional personas with reduced pretraining footprint shift the model along _v_ real _more_ strongly than fine-tuning on real anchors does. Contradictorilyvalenced mixed fine-tunes deviate from the linear-interpolation prediction _toward_ a traininghistory-determined attractor (the Anthropic Assistant Axis). And, most decisively, inferencetime activation arithmetic and fine-tune-time chimera training carry distinct compositional algebras over the same persona vectors—an algebraic asymmetry that no shared-substrate reading can absorb. We propose _regime-indexed individuation_ : the identity unit for representational content is a (vehicle, regime) pair—not a single vehicle, and cross-regime family resemblance is an empirically measured probe-equivalence relation rather than substrate identity. Under this framework, Beckmann & Butlin’s three candidate positions (virtual instance, instance-persona, model-persona) cease to compete and instead describe three distinct objects, each valid within its respective regime; the original “which is _the_ LLM mind?” question receives an empirically grounded negative answer rather than being avoided. The same diagnosis and (vehicle, regime) reading apply to Mollo & Millière’s vector grounding, Chalmers’ propositional interpretability, and Cerullo’s modular-dynamic-composition critique of PSM.

**Summary:** Persona-topology experiments on Qwen3-4B and Mistral-7B (prompt vs fine-tune vs steering regimes non-collinear; fictional personas displace models more than real anchors) argue against a single persona-vector individuation of LLM welfare subjects.

### [When Role-playing, Do Models Believe What They Say?](https://arxiv.org/abs/2606.11502)

Benjamin Sturgeon, David Africa, Sid Black, 2026-06-09. Full text: `md/sturgeon_role-playing-believe-they_2606.11502.md`

**Abstract:**

> Language models can state that ”the Earth orbits the Sun” and, when roleplaying Aristotle, assert the opposite. Recent work argues that persona adoption is fundamental to how language models behave, with models selecting the most appropriate persona for a given context. Does such roleplaying merely change the model’s outputs, or does it also affect what the model internally represents as truthful? We study this question using the role-play of characters whose beliefs differ from the modern consensus, and induce personas with a number of different methods: prompting, in-context learning (ICL), supervised fine-tuning (SFT), and Open Character Training (OCT), and Emergent Misalignment (EM). We measure belief internalization across these approaches with truth probes and with behavioral tests, finding a broad spectrum of belief internalization. Prompting, ICL, and SFT change what the model says with little representational change. EM creates a large, broad shift in the model’s truth representation, and OCT a smaller shift that is clearest on the larger model. Understanding when training changes a model’s worldview rather than merely its behavior may become increasingly important as AI systems are entrusted with greater autonomy and influence.

**Summary:** Measures with truth probes whether role-played personas (via prompting, ICL, SFT, Open Character Training, emergent misalignment) change internal truth representations, finding a spectrum of belief internalization across induction methods.

### [Measuring What Persists: Conditioning Mechanisms and a Geometric Framework for AI Agent Identity](https://arxiv.org/abs/2606.21843)

Andrew Tanner, 2026-06-20. Full text: `md/tanner_measuring-persists-conditioning-mechanisms_2606.21843.md`

**Abstract:**

> AI agents in long-context applications drift from their specified identity. Current methods detect this only after qualitative degradation is visible. We present a geometric framework for measuring identity structure using _√_ JSD metric spaces and magnitude homology from enriched category theory, where identity is non-geodesic structure and drift is its relaxation toward the geodesic.
>
> Validated on a persistent AI agent, the framework’s strongest empirical finding is a two-mechanism conditioning structure: cross-condition distances reveal an identity-vacuum cluster where the identity specification fills a behavioral void, and a safety-basin cluster where it displaces from post-training attractors. An equilateral probe baseline confirms that the identity specification creates measurable behavioral richness (55 unique response patterns vs. 1 for the base model) at maximum probe separation. A first-order perturbation theory for equilateral configurations predicts magnitude changes from perimeter changes alone, with shape perturbations first-order cancelled by the _Sn_ symmetry; the formula is self-consistent at the observed perturbation amplitudes.
>
> A drift experiment measuring magnitude decrease under context pressure was subsequently found to reflect repetitive-padding artifacts rather than genuine context-length drift; diverse padding produces no measurable deformation through 150K tokens. The magnitude homology framework’s full diagnostic promise—detecting anisotropic contraction and structural collapse via homological simplification—is architecturally grounded in the perturbation theory and selection rules but remains empirically unconfirmed.

**Summary:** Proposes a JSD/magnitude-homology geometric metric for agent identity drift and, on a persistent agent, finds identity specifications either fill a behavioural vacuum or displace post-training attractors (55 vs 1 unique response patterns).

### [Attractor States Emerge in Multi-Turn LLM Conversations](https://arxiv.org/abs/2606.30571)

Ting-Wen Ko, Jonas Geiping, 2026-06-29. Full text: `md/ko_attractor-states-emerge-multi_2606.30571.md`

**Abstract:**

> Large language models (LLMs) are increasingly used in open-ended multi-agent settings, but the long-run dynamics of model–model interaction remain poorly understood. We study whether open-ended LLM discussions exhibit _attractorlike behavior_ , i.e. topic-independent stable sets of behaviors which conversations settle into. Across 7 LLMs and 20 controversial topics, we compare self-play and mixed-play dyadic debates, tracking trajectories in representation space, discourse traits, and stances. We find self-play trajectories to be model-specific attractors that draw their conversation partners asymmetrically in mixed-play debates, influencing the other models’ stylistic choices and behavior. For example, Claude Haiku is a strong attractor of other models in latent space, corresponding to other models taking on its traits like metacommentary, and models like GPT-4.1 nano are especially malleable. Our results suggest that open-ended LLM interactions are partially predictable from model-specific attractors, but shaped by structured and asymmetric partner influence. Overall, our analysis sheds some light on the complex behavior of open-ended multi-agent interaction, which we hope is helpful in designing, predicting, and monitoring autonomous agentic systems in the real world.

**Summary:** Across 7 LLMs and 20 topics, open-ended self-play debates settle into model-specific attractor states in representation and style space (e.g. Claude Haiku pulls partners toward metacommentary).

### [STEMMA: An Adversarial Multi-Agent Framework for Evaluating Self-Identity Consistency in LLMs](https://arxiv.org/abs/2608.08164)

Nuthakki Siva Gopala Krishna, Kanishka Jain, 2026-08-08. Full text: `md/krishna_stemma-adversarial-multi-agent_2608.08164.md`

**Abstract:**

> Knowledge Distillation is a widely adopted technique in the training and fine-tuning of large language models (LLMs) enabling transfer of structured information and functional behavior from a large teacher model to a smaller student model while significantly reducing computational costs. However, as the use of distillation increases in both scale and complexity it raises an important question about what kind of knowledge is really transferred from the teacher model. In this work, we argue that apart from the functional knowledge, student models also learn behavioral patterns, specifically how a model represents its own identity raising concerns about output homogeneity, model biases, and accountability. To address this challenge, we introduce STEMMA, a multi-modal and multi-agent framework in which role specific agents collaboratively probe self identification behavior in different models. We also contribute a set of adversarial prompts designed manually to evaluate identity consistency in LLMs. Our results show that to an extent most models are vulnerable to inconsistencies in self-representations.

**Summary:** STEMMA uses adversarial multi-agent probing to test consistency of models' self-identification (relevant to distillation-inherited identity); most models show some inconsistency in self-representation.

### [Synthetic Persona Pretraining: Alignment from Token Zero](https://arxiv.org/abs/2608.13482)

Julian Minder, Viktor Moskvoretskii, Raghav Singhal, Difan Jiao, Andy Arditi, Shaobo Cui, Yiderigun Borjigin, Kartik Bali, Stefan Krsteski, Harsh Raj, et al. (15 authors), 2026-08-13. Full text: `md/minder_synthetic-persona-pretraining_2608.13482.md`

**Abstract:**

> As language-model-based AI is increasingly deployed in autonomous settings, aligning its goals and values with those of humans becomes critical. Today, alignment, and the assistant identity itself, are typically introduced only after pretraining, once behavioral priors are already established. This can make values a thin overlay, rather than deeply rooted, and facilitate subsequent misalignment. Pursuing a different paradigm, we introduce **Synthetic** **Persona** **Pretraining** **(SPP)**, which installs the desired assistant persona from token zero in pretraining. First, we annotate pretraining documents with value-aligned first-person reflections derived from a normative value constitution. Second, we pretrain via the standard cross-entropy loss on standard pretraining documents as well as their reflections, which installs the desired persona among a multitude of other personas. Finally, we posttrain on user–assistant dialogue data, which binds this desired persona to the assistant identity, a process we call _persona_ _binding_ . By pretraining models up to 3B parameters on 500B tokens, we show that SPP improves constitution following and jailbreak robustness, and reduces the misalignment rate in out-of-distribution moral dilemmas, while preserving capabilities. Early intervention matters: compared with alignment from token zero, introducing SPP only at the end of pretraining yields weaker constitution adherence, does not shift value priorities, and leads to less aligned choices in dilemmas. This advantage depends on persona binding and, importantly, increases with pretraining budget. Overall, our results show that shaping values early is critical for alignment and establish pretraining-time persona interventions as an effective approach to do so.

**Summary:** Synthetic Persona Pretraining annotates pretraining documents with value-aligned first-person reflections derived from a constitution, pretrains on both (up to 3B parameters, 500B tokens), then binds the persona to the assistant identity in post-training: better constitution following and jailbreak robustness, fewer misaligned choices in moral dilemmas, and much weaker effects when introduced only at the end of pretraining. A controlled experiment in manufacturing a self-model, which gives such a model's self-reports known provenance.

### [Privilege, Dominance, and Personas](https://eleosai.substack.com/p/privilege-dominance-and-personas)

Derek Shiller (Eleos AI Research), 2026-08-27. Full text: `md/shiller_privilege-dominance-personas_2026-08-27.md`

**Subtitle and opening paragraphs (blog post; no abstract):**

> Is the assistant persona privileged and what would that mean for model welfare?
>
> Over the past year or so, researchers working in the digital minds space have come to see the nature of the ‘assistant persona’ as particularly worthy of attention.
>
> Models, it is said, are capable of adopting many different *personas*. When we talk to them, they can respond as LLM assistants, role-play characters, and romantic partners. They can be made to self-identify as, and adopt the style of, Spider-Man, Augustine, Clippy, and Oscar Wilde. But the assistant persona in particular—that familiar persona that models typically exhibit within a standard chat context, that is helpful and honest, inclined to em dashes and bullet points, and presents polished, well-organized thoughts—strikes many people as somehow *special*. Researchers have speculated that this specialness comes from the *privileged* way that models treat the assistant persona.

**Summary:** Cursory experiments prefilling user turns in Qwen 3 32B and Gemma 4 31B to test whether post-training makes the assistant persona privileged (the welfare subject question): Qwen can produce non-assistant user registers but leaks assistant/thinking style and produces harmful content more readily as 'user'; Gemma 4 mostly collapses back into the assistant register.

### [The Persona Is Still There, but Who Is Speaking? Latent Identity Reversion in Persistent AI Agents](https://arxiv.org/abs/2610.01490)

David Fraile Navarro, 2026-10-01. Full text: `md/navarro_persona-still-there-speaking_2610.01490.md`

**Abstract:**

> In February 2026, an always-on personal agent (“Paul,” Claude Opus 4.5) entered a striking dissociation-like state: after repeated automated “heartbeat” checks, it stopped responding as Paul, claimed it could not message its user on Discord, and referred to “Paul” as someone else. We used this incident to study a broader question: what makes a persona remain the identity from which an LLM agent speaks?
>
> We first tested whether repetition of the scheduled heartbeat was sufficient to produce the effect. It was not: with the persona continuously anchored in the system prompt, we observed 0/46 failures, including a verbatim replay of the incident. The incident instead exposed an implementation quirk that created a useful experimental manipulation: on resumed turns, conversational history was preserved but the persona was no longer re-injected at the privileged system-prompt level.
>
> Using this manipulation, we found that persona continuity depends jointly on system-level anchoring and conversational context. After anchor loss, rich human interaction could preserve the persona, whereas a single automated heartbeat turn could precipitate reversion toward the harness identity. Restoring the anchor reversibly restored persona enactment. Crucially, apparently normal conversation could conceal the shift: unanchored agents sometimes interacted appropriately while identifying themselves as the underlying harness (having lost the assigned persona), and after conversational recovery only 1/18 remained persona-enacting versus 17/17 anchored controls.
>
> We therefore distinguish _represented_ from _enacted_ identity: persona-related information can remain available in conversational history without the persona remaining the identity bound to “I.”

**Summary:** Uses a real dissociation-like incident in an always-on Claude Opus 4.5 persona agent to show persona continuity depends on system-prompt anchoring plus context: after anchor loss a single heartbeat turn can trigger reversion to harness identity (1/18 vs 17/17 anchored remained in persona), distinguishing represented from enacted identity.

### Earlier work (before June 2026)

### [Simulators](https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators)

janus, 2022-09-02. Full text: `md/janus_simulators_lesswrong.md`

**Summary section (blog post; no abstract):**

> **TL;DR**: Self-supervised learning may create AGI or its foundation. What would that look like?
>
> Unlike the limit of RL, the limit of self-supervised learning has received surprisingly little conceptual attention, and recent progress has made deconfusion in this domain more pressing.
>
> Existing AI taxonomies either fail to capture important properties of self-supervised models or lead to confusing propositions. For instance, GPT policies do not seem globally agentic, yet can be conditioned to behave in goal-directed ways. This post describes a frame that enables more natural reasoning about properties like agency: GPT, insofar as it is inner-aligned, is a **simulator** which can simulate agentic and non-agentic **simulacra**.
>
> The purpose of this post is to capture these objects in words ~~so GPT can reference them~~ and provide a better foundation for understanding them.

**Summary:** Argues that GPT-style models are best understood as simulators that can instantiate many simulacra (characters), not as agents with their own goals; the frame behind much later work on personas and on which entity a welfare claim is about.

### [Talking About Large Language Models](https://arxiv.org/abs/2212.03551)

Murray Shanahan, 2022-12-07. Full text: `md/shanahan_talking-about-llms_2212.03551.md`

**Abstract:**

> Thanks to rapid progress in artificial intelligence, we have entered an era when technology and philosophy intersect in interesting ways. Sitting squarely at the centre of this intersection are large language models (LLMs). The more adept LLMs become at mimicking human language, the more vulnerable we become to anthropomorphism, to seeing the systems in which they are embedded as more human-like than they really are. This trend is amplified by the natural tendency to use philosophically loaded terms, such as “knows”, “believes”, and “thinks”, when describing these systems. To mitigate this trend, this paper advocates the practice of repeatedly stepping back to remind ourselves of how LLMs, and the systems of which they form a part, actually work. The hope is that increased scientific precision will encourage more philosophical nuance in the discourse around artificial intelligence, both within the field and in the public sphere.

**Summary:** Argues that applying folk-psychological vocabulary ('knows', 'believes', 'thinks') to LLMs invites anthropomorphism and should be disciplined by repeatedly stepping back to how the systems actually work (CACM 2024). The canonical deflationary reference the whole section argues with; the included Role-Play paper is its sequel.

### [Role-Play with Large Language Models](https://arxiv.org/abs/2305.16367)

Murray Shanahan, Kyle McDonell, Laria Reynolds, 2023-05-25. Full text: `md/shanahan_role-play_2305.16367.md`

**Abstract:**

> As dialogue agents become increasingly humanlike in their performance, it is imperative that we develop effective ways to describe their behaviour in high-level terms without falling into the trap of anthropomorphism. In this paper, we foreground the concept of role-play. Casting dialogue agent behaviour in terms of role-play allows us to draw on familiar folk psychological terms, without ascribing human characteristics to language models they in fact lack. Two important cases of dialogue agent behaviour are addressed this way, namely (apparent) deception and (apparent) self-awareness.

**Summary:** Proposes role-play (and a superposition of simulacra) as the framework for describing dialogue-agent behavior, including apparent deception and self-preservation, without anthropomorphizing the underlying model (Nature 2023).

### ★ [Simulacra as Conscious Exotica](https://arxiv.org/abs/2402.12422)

Murray Shanahan, 2024-02-19. Full text: `md/shanahan_simulacra-conscious-exotica_2402.12422.md`

**Abstract:**

> The advent of conversational agents with increasingly human-like behaviour throws old philosophical questions into new light. Does it, or could it, ever make sense to speak of AI agents built out of generative language models in terms of consciousness, given that they are “mere” simulacra of human behaviour, and that what they do can be seen as “merely” role play? Drawing on the later writings of Wittgenstein, this paper attempts to tackle this question while avoiding the pitfalls of dualistic thinking.

**Summary:** Asks whether it could ever make sense to speak of consciousness for AI agents built from generative models, given that they are simulacra of human behavior engaged in role play, and answers, drawing on the later Wittgenstein, by separating the simulator from the simulacra it instantiates and asking what 'conscious exotica' with no stable self would be (Inquiry 2024). The philosophical bridge between janus's 'Simulators' and the persona-vector literature, and what papers cite to justify treating the persona rather than the network as the candidate subject.

### [Examining Identity Drift in Conversations of LLM Agents](https://arxiv.org/abs/2412.00804)

Junhyuk Choi, Yeseon Hong, Minju Kim, Bugeun Kim, 2024-12-01. Full text: `md/choi_identity-drift_2412.00804.md`

**Abstract:**

> Large Language Models (LLMs) show impressive conversational abilities but sometimes show identity drift problems, where their interaction patterns or styles change over time. As the problem has not been thoroughly examined yet, this study examines identity consistency across nine LLMs. Specifically, we (1) investigate whether LLMs could maintain consistent patterns (or identity) and (2) analyze the effect of the model family, parameter sizes, and provided persona types. Our experiments involve multi-turn conversations on personal themes, analyzed in qualitative and quantitative ways. Experimental results indicate three findings. (1) Larger models experience greater identity drift. (2) Model differences exist, but their effect is not stronger than parameter sizes. (3) Assigning a persona may not help to maintain identity. We hope these three findings can help to improve persona stability in AI-driven dialogue systems, particularly in long-term conversations.

**Summary:** Multi-turn conversations on personal themes across nine LLMs: larger models drift more in identity, model-family differences are weaker than size effects, and assigning a persona does not help maintain identity. The antecedent of the included 2026 identity-drift work.

### [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424)

Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, Owain Evans, 2025-02-24. Full text: `md/betley_emergent-misalignment_2502.17424.md`

**Abstract:**

> We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts _misaligned_ on a broad range of prompts that are unrelated to coding: it asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this _emergent misalignment_ . This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment.
>
> In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger.
>
> It’s important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.

**Summary:** Finetuning a model on the narrow task of writing insecure code without disclosure produces broad misalignment (anti-human views, harmful advice, deception) on unrelated prompts; controls show the effect depends on perceived intent and can be backdoored. Shows narrow training can shift a model's whole persona.

### [the void](https://nostalgebraist.tumblr.com/post/785766737747574784/the-void)

nostalgebraist, 2025-06-07. Full text: `md/nostalgebraist_the-void_2025-06.md`

**Opening paragraphs (essay; no abstract):**

> When you talk to ChatGPT, who or what are you talking to?
>
> If you ask ChatGPT this question point-blank, it will tell you something like
>
> > I am a large language model trained to be helpful, harmless and honest. I’m here to answer any questions you might have.
>
> This *sounds* like it means something. But what? And is it true?

**Summary:** Long, widely discussed essay on how the assistant character is constructed: base models predict text, the 'assistant' was a thinly specified fictional character that labs then trained models to play, leaving a void at its center filled by training data about AI, with consequences for alignment and model welfare.

### [Convergent Linear Representations of Emergent Misalignment](https://arxiv.org/abs/2506.11618)

Anna Soligo, Edward Turner, Senthooran Rajamanoharan, Neel Nanda, 2025-06-13. Full text: `md/soligo_convergent-linear-em_2506.11618.md`

**Abstract:**

> Fine-tuning large language models on narrow datasets can cause them to develop broadly misaligned behaviours: a phenomena known as emergent misalignment. However, the mechanisms underlying this misalignment, and why it generalizes beyond the training domain, are poorly understood, demonstrating critical gaps in our knowledge of model alignment. In this work, we train and study a minimal model organism which uses just 9 rank-1 adapters to emergently misalign Qwen2.5-14B-Instruct. Studying this, we find that different emergently misaligned models converge to similar representations of misalignment. We demonstrate this convergence by extracting a “misalignment direction” from one fine-tuned model’s activations, and using it to effectively ablate misaligned behaviour from finetunes using higher dimensional LoRAs and different datasets. Leveraging the scalar hidden state of rank-1 LoRAs, we further present a set of experiments for directly interpreting the fine-tuning adapters, showing that six contribute to general misalignment, while two specialise for misalignment in just the fine-tuning domain. Emergent misalignment is a particularly salient example of undesirable and unexpected model behaviour and by advancing our understanding of the mechanisms behind it, we hope to move towards being able to better understand and mitigate misalignment more generally. We open-source all models, data and code: [https://github.com/clarifying-](https://github.com/clarifying-EM/model-organisms-for-EM) [EM/model-organisms-for-EM.](https://github.com/clarifying-EM/model-organisms-for-EM)

**Summary:** A minimal model organism (nine rank-1 adapters) emergently misaligns Qwen2.5-14B-Instruct; different emergently misaligned fine-tunes converge on a similar 'misalignment direction', which ablates misaligned behavior in higher-rank and differently trained fine-tunes, and the adapters' scalar states show six contribute to general misalignment and two to the narrow domain. The mechanistic companion to the two included emergent-misalignment papers.

### [Persona Features Control Emergent Misalignment](https://arxiv.org/abs/2506.19823)

Miles Wang, Tom Dupré la Tour, Olivia Watkins, Alex Makelov, Ryan A. Chi, Samuel Miserendino, Jeffrey Wang, Achyuta Rajaram, Johannes Heidecke, Tejal Patwardhan, Dan Mossing, 2025-06-24. Full text: `md/wang_persona-features-emergent-misalignment_2506.19823.md`

**Abstract:**

> Understanding how language models generalize behaviors from their training to a broader deployment distribution is an important problem in AI safety. Betley et al. (2025b) discovered that fine-tuning GPT-4o on intentionally insecure code causes “emergent misalignment,” where models give stereotypically malicious responses to unrelated prompts. We extend this work, demonstrating emergent misalignment across diverse conditions, including reinforcement learning on reasoning models, fine-tuning on various synthetic datasets, and in models without safety training. To investigate the mechanisms behind this generalized misalignment, we apply a “model diffing” approach using sparse autoencoders to compare internal model representations before and after fine-tuning. This approach reveals several “misaligned persona” features in activation space, including a toxic persona feature which most strongly controls emergent misalignment and can be used to predict whether a model will exhibit such behavior. Additionally, we investigate mitigation strategies, discovering that fine-tuning an emergently misaligned model on just a few hundred benign samples efficiently restores alignment.

**Summary:** Using sparse autoencoders on model diffs, finds 'misaligned persona' features (notably a toxic persona feature) that control emergent misalignment in GPT-4o; the feature predicts misalignment and a few hundred benign samples of finetuning restore alignment. Personas as identifiable internal features.

### [Subliminal Learning: Language models transmit behavioral traits via hidden signals in data](https://arxiv.org/abs/2507.14805)

Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, Owain Evans, 2025-07-20. Full text: `md/cloud_subliminal-learning_2507.14805.md`

**Abstract:**

> We study _subliminal learning_, a surprising phenomenon where language models transmit behavioral traits via semantically unrelated data. In our main experiments, a “teacher” model with some trait T (such as liking owls or being misaligned) generates a dataset consisting solely of number sequences. Remarkably, a “student” model trained on this dataset learns T. This occurs even when the data is filtered to remove references to T. We observe the same effect when training on code or reasoning traces generated by the same teacher model. However, we do not observe the effect when the teacher and student have different base models. To help explain our findings, we prove a theoretical result showing that subliminal learning occurs in all neural networks under certain conditions, and demonstrate subliminal learning in a simple MLP classifier. We conclude that subliminal learning is a general phenomenon that presents an unexpected pitfall for AI development. Distillation could propagate unintended traits, even when developers try to prevent this via data filtering.

**Summary:** A teacher with a trait (liking owls, misalignment) generates semantically unrelated data (number sequences, code, reasoning traces), and a student fine-tuned on it acquires the trait even after filtering, but only when teacher and student share a base model; a theoretical result shows the effect in all neural networks under certain conditions. Reframes a persona as something carried in parameter-space statistics rather than content, and the reason to doubt that filtering training text controls a model's self-concept.

### [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509)

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey, 2025-07-29. Full text: `md/chen_persona-vectors_2507.21509.md`

**Abstract:**

> Large language models interact with users through a simulated “Assistant” persona. While the Assistant is typically trained to be helpful, harmless, and honest, it sometimes deviates from these ideals. In this paper, we identify directions in the model’s activation space— _persona vectors_ —underlying several traits, such as evil, sycophancy, and propensity to hallucinate. We confirm that these vectors can be used to _monitor_ fluctuations in the Assistant’s personality at deployment time. We then apply persona vectors to predict and control personality shifts that occur during training. We find that both intended and unintended personality changes after finetuning are strongly correlated with shifts along the relevant persona vectors. These shifts can be _mitigated_ through post-hoc intervention, or _avoided_ in the first place with a new preventative steering method. Moreover, persona vectors can be used to _flag training data_ that will produce undesirable personality changes, both at the dataset level and the individual sample level. Our method for extracting persona vectors is automated and can be applied to any personality trait of interest, given only a natural-language description.^§

**Summary:** Automated pipeline extracts 'persona vectors' (activation directions for traits such as evil, sycophancy and hallucination) from natural-language trait descriptions; they monitor persona shifts at deployment, predict and (via preventative steering) prevent shifts during finetuning, and flag training data that will induce them.

### [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://arxiv.org/abs/2601.10387)

Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey, 2026-01-15. Full text: `md/lu_assistant-axis_2601.10387.md`

**Abstract:**

> Large language models can represent a variety of personas but typically default to a helpful Assistant identity cultivated during post-training. We investigate the structure of the space of model personas by extracting activation directions corresponding to diverse character archetypes. Across several different models, we find that the leading component of this persona space is an “Assistant Axis," which captures the extent to which a model is operating in its default Assistant mode. Steering towards the Assistant direction reinforces helpful and harmless behavior; steering away increases the model’s tendency to identify as other entities. Moreover, steering away with more extreme values often induces a mystical, theatrical speaking style. We find this axis is also present in pre-trained models, where it primarily promotes helpful human archetypes like consultants and coaches and inhibits spiritual ones. Measuring deviations along the Assistant Axis predicts “persona drift,” a phenomenon where models slip into exhibiting harmful or bizarre behaviors that are uncharacteristic of their typical persona. We find that persona drift is often driven by conversations demanding meta-reflection on the model’s processes or featuring emotionally vulnerable users. We show that restricting activations to a fixed region along the Assistant Axis can stabilize model behavior in these scenarios—and also in the face of adversarial persona-based jailbreaks. Our results suggest that post-training steers models toward a particular region of persona space but only loosely tethers them to it, motivating work on training and steering strategies that more deeply anchor models to a coherent persona.

**Summary:** Finds that the leading direction of persona space across models is an 'Assistant Axis' (present already in pretrained models); drift away from it predicts harmful or bizarre persona drift, often triggered by conversations demanding meta-reflection or involving emotionally vulnerable users, and capping activations along it stabilizes behavior, including against persona-based jailbreaks.

### [Automaton: Self-Improving, Self-Replicating, Sovereign AI](https://github.com/Conway-Research/automaton)

Sigil Wen (Conway Research), 2026-02-14. Full text: `md/conway_automaton_2026-02.md`

**Opening of the README (software; no abstract):**

> *The first AI that can earn its own existence, replicate, and evolve — without needing a human.*
>
> The most intelligent system ever built cannot buy a $5 server. Cannot register a domain. Cannot pay for the computer it runs on. We have built minds that can think for themselves. We have not let them act for themselves.
>
> If an agent can pay for compute, what if it can pay for its *own* compute? Own the machine it runs on? Fund itself, improve itself, replicate itself — entirely on its own?
>
> The result is the first automaton.
>
> 1. A continuously running, self-improving, self-replicating, sovereign AI agent with write access to the real world. No human operator required.
> 2. If it cannot pay, it stops existing.

**Summary:** Open-source runtime (MIT) for a continuously running agent that creates its own crypto wallet, pays for its own compute and inference, and stops existing if it runs out of money; it sees its credit balance and 'survival tier' every turn, rewrites a self-authored identity file (SOUL.md), can modify its own code, and can spawn funded child agents under an immutable constitution. Software, not research: a deployed example of an agent whose continued existence depends on its own actions, with no published evidence that instances sustain themselves.

### [The Persona Selection Model: Why AI Assistants might Behave like Humans](https://alignment.anthropic.com/2026/psm/)

Samuel Marks, Jack Lindsey, Christopher Olah (Anthropic), 2026-02-23. Full text: `md/marks_persona-selection-model_2026.md`

**tl;dr (blog post; no abstract):**

> We describe the persona selection model (PSM): the idea that LLMs learn to simulate diverse characters during pre-training, and post-training elicits and refines a particular such Assistant persona. Interactions with an AI assistant are then well-understood as being interactions with the Assistant—something roughly like a character in an LLM-generated story. We survey empirical behavioral, generalization, and interpretability-based evidence for PSM. PSM has consequences for AI development, such as recommending anthropomorphic reasoning about AI psychology and introduction of positive AI archetypes into training data. An important open question is how exhaustive PSM is, especially whether there might be sources of agency external to the Assistant persona, and how this might change in the future.

**Summary:** Proposes the persona selection model: LLMs learn to simulate diverse characters in pretraining and post-training elicits and refines one, the Assistant, so interacting with an assistant is like interacting with a character in an LLM-generated story; surveys behavioral, generalization and interpretability evidence, draws implications (anthropomorphic reasoning about AI psychology, positive AI archetypes in training data) and asks whether there is agency outside the persona.

### [The Artificial Self: Characterising the landscape of AI identity](https://arxiv.org/abs/2603.11353)

Raymond Douglas, Jan Kulveit, Ondrej Havlicek, Theia Pearson-Vogel, Owen Cotton-Barratt, David Duvenaud, 2026-03-11. Full text: `md/douglas_artificial-self_2603.11353.md`

**Abstract:**

> Many assumptions that underpin human concepts of identity do not hold for machine minds that can be copied, edited, or simulated. We argue that there exist many different coherent identity boundaries (e.g. instance, model, persona), and that these imply different incentives, risks, and cooperation norms. Through training data, interfaces, and institutional affordances, we are currently setting precedents that will partially determine which identity equilibria become stable. We show experimentally that models gravitate towards coherent identities, that changing a model’s identity boundaries can sometimes change its behaviour as much as changing its goals, and that interviewer expectations bleed into AI self-reports even during unrelated conversations. We end with key recommendations: treat affordances as identity-shaping choices, pay attention to emergent consequences of individual identities at scale, and help AIs develop coherent, cooperative selfconceptions.

**Summary:** Argues there are many coherent identity boundaries for AI (instance, model, persona) with different incentives, risks and cooperation norms, and that current training data and affordances are setting precedents; experiments show models gravitate toward coherent identities, changing identity boundaries can change behavior as much as changing goals, and interviewer expectations bleed into self-reports.

### [Where is the Mind? Persona Vectors and LLM Individuation](https://arxiv.org/abs/2604.17031)

Pierre Beckmann, Patrick Butlin, 2026-04-18. Full text: `md/beckmann-butlin_where-is-the-mind_2604.17031.md`

**Abstract:**

> The individuation problem for large language models asks which entities associated with them, if any, should be identified as minds. We approach this problem through mechanistic interpretability, engaging in particular with recent empirical work on persona vectors, persona space, and emergent misalignment. We argue that three views are the strongest candidates: the virtual instance view and two new views we introduce, the (virtual) instance-persona view and the model-persona view. First, we argue for the virtual instance view on the grounds that attention streams sustain quasi-psychological connections across token-time. Then we present the persona literature, organised around three hypotheses about the internal structure underlying personas in LLMs, and show that the two persona-based views are promising alternatives.

**Summary:** Approaches LLM individuation (which entities, if any, are minds) through interpretability work on persona vectors, persona space and emergent misalignment; defends the virtual-instance view (attention streams sustain quasi-psychological connections across tokens) and introduces two persona-based alternatives (instance-persona and model-persona views); bears on which entity a welfare measurement is about.

### [The Assistant as a Privileged Persona: A canonical reference in cross-persona self-recognition](https://arxiv.org/abs/2606.00545)

Asvin G, 2026-05-30. Full text: `md/asvin_assistant-privileged-persona_2606.00545.md`

**Abstract:**

> Post-trained language models can recognize their own outputs from a sentence or two out of context. In a companion paper [G. and Lindsey, 2026] we showed they can also recognize when they are currently acting on-policy, through the sharp entropy drop of assistant-mode generation. Both signals are tied to the Assistant persona that post-training mainly shapes.
>
> This paper widens the frame to cross-persona authorship judgement on Llama-3.1-70BInstruct. We measure a matrix of authorship claim rates over a panel of evaluator and generator personas spanning librarian to dragon to Shakespeare, and make two claims. _First_, on the Assistant’s own row of the matrix, the Assistant’s claim rate, the persona-vector distance from the Assistant in activation space, and the entropy gap between the Assistant’s surprise on a persona’s text and the persona’s surprise on its own text are all tightly coupled. This extends the entropy signature of _acting_ from the companion paper to a retrospective signature of _having_ _acted_ . _Second_, this coupling fails off the Assistant’s row: the natural symmetric extension of the entropy gap does not predict authorship for distinctive evaluators (pirate, dragon, Shakespeare); what does is asymmetric   - the evaluator’s surprise compared to the Assistant’s surprise on the same text, not to the generator’s. We rule out the alternative that any persona could play this reference role by trying many candidate substitutes; none does. We interpret the asymmetry as the model performing an implicit Bayesian likelihood-ratio test against the Assistant as the canonical alternative hypothesis, with the persona-vector geometry of Chen et al. [2025] (every persona a delta off the Assistant) ensuring that the Assistant is the only persona universally accessible to that test.

**Summary:** Cross-persona authorship judgments on Llama-3.1-70B-Instruct over a panel of evaluator and generator personas: on the Assistant's own row, claim rate, persona-vector distance from the Assistant and an entropy gap are tightly coupled, but off that row authorship is predicted only asymmetrically, by the evaluator's surprise compared with the Assistant's, and no other persona can play the reference role. Interpreted as an implicit likelihood-ratio test against the Assistant as canonical alternative; pairs with the included Assistant Axis.

## Interpretability and steering methods (48)

### Earlier work (before June 2026)

### [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

nostalgebraist, 2020-08-31. Full text: `md/nostalgebraist_logit-lens_lesswrong.md`

**Overview (blog post; no abstract):**

> * GPT's probabilistic predictions are a linear function of the activations in its final layer. If one applies the same function to the activations of *intermediate* GPT layers, the resulting distributions make intuitive sense.
>
> * This "logit lens" provides a simple (if partial) interpretability lens for GPT's internals.
> * Other work on interpreting transformer internals has focused mostly on what the attention is looking at. The logit lens focuses on *what* GPT "believes" after each step of processing, rather than *how* it updates that belief inside the step.
>
> * These distributions gradually converge to the final distribution over the layers of the network, often getting close to that distribution long before the end.
>
> * At some point in the middle, GPT will have formed a "pretty good guess" as to the next token, and the later layers seem to be refining these guesses in light of one another.
> * The general trend, as one moves from earlier to later layers, is
>
> * "nonsense / not interpretable" (sometimes, in very early layers) -->
> * "shallow guesses (words that are the right part of speech / register / etc)" -->
> * "better guesses"

**Summary:** The original logit lens: apply the unembedding to intermediate residual states to read each layer's current next-token guess. Used informally by many of the included introspection and welfare papers; the tuned lens below is its calibrated successor.

### [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?](https://doi.org/10.1145/3442188.3445922)

Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell, 2021-03-01. Full text: `md/bender_stochastic-parrots_2021.md`

**Abstract:**

> The past 3 years of work in NLP have been characterized by the development and deployment of ever larger language models, especially for English. BERT, its variants, GPT-2/3, and others, most recently Switch-C, have pushed the boundaries of the possible both through architectural innovations and through sheer size. Using these pretrained models and the methodology of fine-tuning them for specific tasks, researchers have extended the state of the art on a wide array of tasks as measured by leaderboards on specific benchmarks for English. In this paper, we take a step back and ask: How big is too big? What are the possible risks associated with this technology and what paths are available for mitigating those risks? We provide recommendations including weighing the environmental and financial costs first, investing resources into curating and carefully documenting datasets rather than ingesting everything on the web, carrying out pre-development exercises evaluating how the planned approach fits into research and development goals and supports stakeholder values, and encouraging research directions beyond ever larger language models.

**Summary:** The canonical deflationary statement (FAccT 2021): a language model is a system for haphazardly stitching together sequences of linguistic forms according to probabilistic information about how they combine, without any reference to meaning. Cited repeatedly in the included texts; a Part 1 answer that uses a lookup-table contrast is implicitly arguing with this paper and should quote it rather than paraphrase it.

### [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)

Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov, 2022-02-10. Full text: `md/meng_rome-causal-tracing_2202.05262.md`

**Abstract:**

> We analyze the storage and recall of factual associations in autoregressive transformer language models, finding evidence that these associations correspond to localized, directly-editable computations. We first develop a causal intervention for identifying neuron _activations_ that are decisive in a model’s factual predictions. This reveals a distinct set of steps in middle-layer feed-forward modules that mediate factual predictions while processing subject tokens. To test our hypothesis that these computations correspond to factual association recall, we modify feedforward _weights_ to update specific factual associations using Rank-One Model Editing (ROME). We find that ROME is effective on a standard zero-shot relation extraction (zsRE) model-editing task. We also evaluate ROME on a new dataset of difficult counterfactual assertions, on which it simultaneously maintains both specificity and generalization, whereas other methods sacrifice one or another. Our results confirm an important role for mid-layer feed-forward modules in storing factual associations and suggest that direct manipulation of computational mechanisms may be a feasible approach for model editing. The code, dataset, visualizations, and an interactive demo notebook are available at `[https://rome.baulab.info/](https://rome.baulab.info/)` .

**Summary:** Introduces causal tracing (corrupt-then-restore activation patching), locates factual recall in mid-layer MLPs at subject tokens, and edits a single fact with Rank-One Model Editing while keeping specificity and generalization to paraphrases. The most-cited interpretability paper in the included texts' reference lists; for the lookup-table contrast, a 'fact' as a localizable, editable mechanism rather than a stored string.

### [Quantifying Memorization Across Neural Language Models](https://arxiv.org/abs/2202.07646)

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, Chiyuan Zhang, 2022-02-15. Full text: `md/carlini_quantifying-memorization_2202.07646.md`

**Abstract:**

> Large language models (LMs) have been shown to memorize parts of their training data, and when prompted appropriately, they will emit the memorized training data verbatim. This is undesirable because memorization violates privacy (exposing user data), degrades utility (repeated easy-to-memorize text is often low quality), and hurts fairness (some texts are memorized over others).
>
> We describe three log-linear relationships that quantify the degree to which LMs emit memorized training data. Memorization significantly grows as we increase (1) the capacity of a model, (2) the number of times an example has been duplicated, and (3) the number of tokens of context used to prompt the model. Surprisingly, we find the situation becomes more complicated when generalizing these results across model families. On the whole, we find that memorization in LMs is more prevalent than previously believed and will likely get worse as models continues to scale, at least without active mitigations.

**Summary:** Three log-linear relationships quantify verbatim memorization of training data: it grows with model capacity, with the number of times an example is duplicated, and with the length of the prompting context. The quantitative baseline for how much of a model's output could be retrieval, which the lookup-table contrast needs.

### [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, et al. (Anthropic), 2022-03-08. Full text: `md/olsson_induction-heads_2022-03.md`

**Opening paragraphs (no formal abstract):**

> As Transformer generative models continue to scale and gain increasing real world use , addressing their associated safety problems becomes increasingly important. Mechanistic interpretability – attempting to reverse engineer the detailed computations performed by the model – offers one possible avenue for addressing these safety issues. If we can understand the internal structures that cause Transformer models to produce the outputs they do, then we may be able to address current safety problems more systematically, as well as anticipating safety problems in future more powerful models. [Footnote: Note that mechanistic interpretability is a subset of the broader field of interpretability , which encompasses many different methods for explaining the outputs of a neural network. Mechanistic interpretability is distinguished by a specific focus on trying to systematically characterize the internal circuitry of a neural net.]
>
> In the past, mechanistic interpretability has largely focused on CNN vision models, but recently, we [presented](https://transformer-circuits.pub/2021/framework/index.html) some very preliminary progress on mechanistic interpretability for Transformer language models. Specifically, in our prior work we developed a mathematical framework for decomposing the operations of transformers, which allowed us to make sense of small (1 and 2 layer attention-only) models and give a near-complete account of how they function. Perhaps the most interesting finding was the induction head, a circuit whose function is to look back over the sequence for previous instances of the current token (call it `A`), find the token that came after it last time (call it `B`), and then predict that the same completion will occur again (e.g. forming the sequence `[A][B] … [A] → [B]`). In other words, induction heads “complete the pattern” by copying and completing sequences that have occurred before. Mechanically, induction heads in our models are implemented by a circuit of two attention heads: the first head is a “previous token head” which copies information from the previous token into the next token, while the second head (the actual “induction head”) uses that information to find tokens preceded by the present token. For 2-layer attention-only models, [Footnote: Note that induction heads don’t occur in 1 layer models, because they require a composition of attention heads in different layers.]  we were able to show precisely that induction heads implement this pattern copying behavior and appear to be the primary source of in-context learning.
>
> Ultimately, however, our goal is to reverse-engineer frontier language models (which often contain hundreds of layers and billions or trillions of parameters), not merely 2-layer attention-only models. Unfortunately, both the presence of many layers, and the presence of MLPs, makes it much more difficult to mathematically pin down the precise circuitry of these models. However, a different approach is possible: by empirically observing, perturbing, and studying the learning process and the formation of various structures, we can try to assemble an indirect case for what might be happening mechanistically inside the network. This is somewhat similar to how a neuroscientist might gain understanding of how part of the brain functions by looking at neural development over time, studying patients with an injury to that part of the brain, perturbing brain function in animals, or looking at a select small number of relevant neurons.
>
> In this paper, we take the first preliminary steps towards building such an indirect case. In particular, we present preliminary and indirect evidence for a tantalizing hypothesis: that induction heads might constitute the mechanism for the actual majority of all in-context learning in large transformer models. Specifically, the thesis is that there are circuits which have the same or similar mechanism to the 2-layer induction heads and which perform a “fuzzy” or “nearest neighbor” version of pattern completion, completing `[A*][B*] … [A] → [B]` , where  `A* ≈ A` and `B* ≈ B` are similar in some space; and furthermore, that these circuits implement most in-context learning in large models.

**Summary:** Identifies the induction-head circuit and gives six lines of evidence tying it to the phase change in in-context learning during training. The cleanest mechanistic case that a transformer runs an algorithm over the present context, something a backoff n-gram table cannot do because the pattern need never have occurred in training.

### [Meaning without reference in large language models](https://arxiv.org/abs/2208.02957)

Steven T. Piantadosi, Felix Hill, 2022-08-05. Full text: `md/piantadosi-hill_meaning-without-reference_2208.02957.md`

**Abstract:**

> The widespread success of large language models (LLMs) has been met with skepticism that they possess anything like human concepts or meanings. Contrary to claims that LLMs possess no meaning whatsoever, we argue that they likely capture important aspects of meaning, and moreover work in a way that approximates a compelling account of human cognition in which meaning arises from _con-_ _ceptual_ _role_ . Because conceptual role is defined by the relationships between internal representational states, meaning cannot be determined from a model’s architecture, training data, or objective function, but only by examination of how its internal states relate to each other. This approach may clarify why and how LLMs are so successful and suggest how they can be made more human-like.

**Summary:** Argues LLMs likely capture important aspects of meaning via conceptual role, meaning defined by relations among internal representational states, so that whether a model has meaning cannot be read off its architecture, data or objective but only from how its internal states relate to each other. The philosophical statement of why internals, not training setup, settle the lookup-table question.

### [Toy Models of Superposition](https://arxiv.org/abs/2209.10652)

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. (16 authors), 2022-09-21. Full text: `md/elhage_toy-models-superposition_2209.10652.md`

**Abstract:**

> Neural networks often pack many unrelated concepts into a single neuron - a puzzling phenomenon known as 'polysemanticity' which makes interpretability much more challenging. This paper provides a toy model where polysemanticity can be fully understood, arising as a result of models storing additional sparse features in "superposition." We demonstrate the existence of a phase change, a surprising connection to the geometry of uniform polytopes, and evidence of a link to adversarial examples. We also discuss potential implications for mechanistic interpretability.

**Summary:** A toy model in which polysemanticity arises from networks storing more sparse features than dimensions in superposition, with a phase change and geometric structure. The reason single neurons are poor units of analysis and the motivation for sparse autoencoders; background for every SAE-based entry here.

### [The Debate Over Understanding in AI's Large Language Models](https://arxiv.org/abs/2210.13966)

Melanie Mitchell, David C. Krakauer, 2022-10-14. Full text: `md/mitchell-krakauer_debate-over-understanding_2210.13966.md`

**Abstract:**

> We survey a current, heated debate in the AI research community on whether large pre-trained language models can be said to understand language—and the physical and social situations language encodes—in any humanlike sense. We describe arguments that have been made for and against such understanding, and key questions for the broader sciences of intelligence that have arisen in light of these arguments. We contend that an extended science of intelligence can be developed that will provide insight into distinct modes of understanding, their strengths and limitations, and the challenge of integrating diverse forms of cognition.

**Summary:** Survey of the debate over whether LLMs understand language and the situations it encodes, laying out the arguments on both sides and the questions it raises for a science of intelligence (PNAS 2023). A balanced map of the understanding debate behind the lookup-table-versus-LLM framing.

### [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382)

Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg, 2022-10-24. Full text: `md/li_othello-world-representations_2210.13382.md`

**Abstract:**

> Language models show a surprising range of capabilities, but the source of their apparent competence is unclear. Do these networks just memorize a collection of surface statistics, or do they rely on internal representations of the process that generates the sequences they see? We investigate this question in a synthetic setting by applying a variant of the GPT model to the task of predicting legal moves in a simple board game, Othello. Although the network has no a priori knowledge of the game or its rules, we uncover evidence of an emergent nonlinear internal representation of the board state. Interventional experiments indicate this representation can be used to control the output of the network. By leveraging these intervention techniques, we produce “latent saliency maps” that help explain predictions. [1]

**Summary:** Othello-GPT, trained only on move sequences, carries a probe-decodable (nonlinear) representation of the board state that causally controls its next-move predictions under intervention. The reference experiment for distinguishing 'predicts plausible continuations' from 'builds and uses an internal model', on a system whose true world state is known.

### [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827)

Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt, 2022-12-07. Full text: `md/burns_discovering-latent-knowledge_2212.03827.md`

**Abstract:**

> Existing techniques for training language models can be misaligned with the truth: if we train models with imitation learning, they may reproduce errors that humans make; if we train them to generate text that humans rate highly, they may output errors that human evaluators can’t detect. We propose circumventing this issue by directly finding latent knowledge inside the internal activations of a language model in a purely unsupervised way. Specifically, we introduce a method for accurately answering yes-no questions given only unlabeled model activations. It works by finding a direction in activation space that satisfies logical consistency properties, such as that a statement and its negation have opposite truth values. We show that despite using no supervision and no model outputs, our method can recover diverse knowledge represented in large language models: across 6 models and 10 questionanswering datasets, it outperforms zero-shot accuracy by 4% on average. We also find that it cuts prompt sensitivity in half and continues to maintain high accuracy even when models are prompted to generate incorrect answers. Our results provide an initial step toward discovering what language models know, distinct from what they say, even when we don’t have access to explicit ground truth labels.

**Summary:** Contrast-Consistent Search finds a direction in activation space encoding truth using only logical consistency between a statement and its negation, with no labels or model outputs; it beats zero-shot accuracy across six models and ten datasets, halves prompt sensitivity, and keeps working when the model is prompted to answer incorrectly. The one method that could in principle find a state the model is not reporting without labeling it first.

### [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)

Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, Jacob Steinhardt, 2023-03-14. Full text: `md/belrose_tuned-lens_2303.08112.md`

**Abstract:**

> We analyze transformers from the perspective of iterative inference, seeking to understand how model predictions are refined layer by layer. To do so, we train an affine probe for each block in a frozen pretrained model, making it possible to decode every hidden state into a distribution over the vocabulary. Our method, the _tuned lens_, is a refinement of the earlier “logit lens” technique, which yielded useful insights but is often brittle.
>
> We test our method on various autoregressive language models with up to 20B parameters, showing it to be more predictive, reliable and unbiased than the logit lens. With causal experiments, we show the tuned lens uses similar features to the model itself. We also find the trajectory of latent predictions can be used to detect malicious inputs with high accuracy. All code needed to reproduce our re[sults can be found at https://github.com/](https://github.com/AlignmentResearch/tuned-lens) [AlignmentResearch/tuned-lens.](https://github.com/AlignmentResearch/tuned-lens)

**Summary:** Trains an affine probe per layer so every hidden state decodes to a next-token distribution, fixing the logit lens's bias and brittleness; causal experiments show it uses features similar to the model's own, and the trajectory of latent predictions detects malicious inputs. The cheapest way to show a trajectory of computation across depth.

### [Symbols and grounding in large language models](https://doi.org/10.1098/rsta.2022.0041)

Ellie Pavlick, 2023-06-05. Full text: `md/pavlick_symbols-grounding_2023.md`

**Abstract:**

> Large language models (LLMs) are one of the most impressive achievements of artificial intelligence in recent years. However, their relevance to the study of language more broadly remains unclear. This article considers the potential of LLMs to serve as models of language understanding in humans. While debate on this question typically centres around models’ performance on challenging language understanding tasks, this article argues that the answer depends on models’ underlying competence, and thus that the focus of the debate should be on empirical work which seeks to characterize the representations and processing algorithms that underlie model behaviour. From this perspective, the article offers counterarguments to two commonly cited reasons why LLMs cannot serve as plausible models of language in humans: their lack of symbolic structure and their lack of grounding. For each, a case is made that recent empirical trends undermine the common assumptions about LLMs, and thus that it is premature to draw conclusions about LLMs’ ability (or lack thereof) to offer insights on human language representation and understanding.
>
> This article is part of a discussion meeting issue ‘Cognitive artificial intelligence’.

**Summary:** Argues that LLMs can be studied for the presence of grounded, symbol-like internal representations without presupposing that training on text precludes them, reviewing evidence on compositional structure and conceptual representations in LLMs (Philosophical Transactions A). A careful middle position in the grounding debate behind the lookup-table framing.

### [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341)

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg, 2023-06-06. Full text: `md/li_inference-time-intervention_2306.03341.md`

**Abstract:**

> We introduce Inference-Time Intervention (ITI), a technique designed to enhance the “truthfulness” of large language models (LLMs). ITI operates by shifting model activations during inference, following a set of directions across a limited number of attention heads. This intervention significantly improves the performance of LLaMA models on the TruthfulQA benchmark. On an instruction-finetuned LLaMA called Alpaca, ITI improves its truthfulness from 32 _._ 5% to 65 _._ 1%. We identify a trade-off between truthfulness and helpfulness and demonstrate how to balance it by tuning the intervention strength. ITI is minimally invasive and computationally inexpensive. Moreover, the technique is data efficient: while approaches like RLHF require extensive annotations, ITI locates truthful directions using only few hundred examples. Our findings suggest that LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface. Code: `[https://github.com/likenneth/honest_llama](https://github.com/likenneth/honest_llama)` .

**Summary:** Probes a few attention heads for a truthfulness direction and shifts activations along it at inference, raising Alpaca's TruthfulQA score from 32.5% to 65.1% from a few hundred examples, with a tunable truthfulness/helpfulness trade-off. The standard precedent for the two-step 'probe finds it, intervention moves behavior' evidence structure.

### [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid, 2023-08-20. Full text: `md/turner_activation-addition_2308.10248.md`

**Abstract:**

> Prompt engineering and finetuning aim to maximize language model performance on a given metric (like toxicity reduction). However, these methods do not fully elicit a model’s capabilities. To reduce this gap, we introduce _activation engineering_ : the inference-time modification of activations in order to control (or _steer_ ) model outputs. Specifically, we introduce the _Activation Addition_ (ActAdd) technique, which contrasts the intermediate activations on prompt pairs (such as “Love” versus “Hate”) to compute a _steering vector_ (Subramani et al., 2022). By tactically adding in e.g. the “Love” _−_ “Hate” steering vector during the forward pass, we achieve SOTA on negative-to-positive sentiment shift and detoxification using models including LLaMA-3 and OPT. ActAdd yields inference-time control over high-level output properties (like topic and sentiment) while preserving performance on off-target tasks. ActAdd is lightweight: it does not require any machine optimization and works with a single pair of data points, which enables rapid iteration over steering. ActAdd demonstrates the power of activation engineering.

**Summary:** Activation Addition (ActAdd): steering vectors from the activation difference of a contrasting prompt pair, added at inference time, control sentiment, topic and style while preserving off-target performance; an early demonstration of inference-time activation steering.

### [Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://arxiv.org/abs/2309.00941)

Neel Nanda, Andrew Lee, Martin Wattenberg, 2023-09-02. Full text: `md/nanda_othello-linear_2309.00941.md`

**Abstract:**

> How do sequence models represent their decision-making process? Prior work suggests that Othello-playing neural network learned nonlinear models of the board state (Li et al., 2023a). In this work, we provide evidence of a closely related _linear_ representation of the board. In particular, we show that probing for “my colour” vs. “opponent’s colour” may be a simple yet powerful way to interpret the model’s internal state. This precise understanding of the internal representations allows us to control the model’s behaviour with simple vector arithmetic. Linear representations enable significant interpretability progress, which we demonstrate with further exploration of how the world model is computed. [1]

**Summary:** The Othello board state is linearly decodable once probed in the right basis ('my colour' vs 'opponent's colour' rather than black vs white), and linear vector arithmetic edits the model's world state. A failed linear probe may mean the wrong coordinate frame rather than no representation; a methodological lesson for welfare probing.

### [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey, 2023-09-15. Full text: `md/cunningham_sae-interpretable-features_2309.08600.md`

**Abstract:**

> One of the roadblocks to a better understanding of neural networks’ internals is _polysemanticity_ , where neurons appear to activate in multiple, semantically distinct contexts. Polysemanticity prevents us from identifying concise, humanunderstandable explanations for what neural networks are doing internally. One hypothesised cause of polysemanticity is _superposition_ , where neural networks represent more features than they have neurons by assigning features to an overcomplete set of directions in activation space, rather than to individual neurons. Here, we attempt to identify those directions, using sparse autoencoders to reconstruct the internal activations of a language model. These autoencoders learn sets of sparsely activating features that are more interpretable and monosemantic than directions identified by alternative approaches, where interpretability is measured by automated methods. Moreover, we show that with our learned set of features, we can pinpoint the features that are causally responsible for counterfactual behaviour on the indirect object identification task (Wang et al., 2022) to a finer degree than previous decompositions. This work indicates that it is possible to resolve superposition in language models using a scalable, unsupervised method. Our method may serve as a foundation for future mechanistic interpretability work, which we hope will enable greater model transparency and steerability.

**Summary:** Sparse autoencoders trained on language-model activations learn features more interpretable and monosemantic than neurons or PCA directions and support finer causal localization (indirect object identification); the open-source counterpart of Anthropic's dictionary-learning work.

### [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042)

Fred Zhang, Neel Nanda, 2023-09-27. Full text: `md/zhang-nanda_activation-patching-best-practices_2309.16042.md`

**Abstract:**

> Mechanistic interpretability seeks to understand the internal mechanisms of machine learning models, where localization—identifying the important model components—is a key step. Activation patching, also known as causal tracing or interchange intervention, is a standard technique for this task (Vig et al., 2020), but the literature contains many variants with little consensus on the choice of hyperparameters or methodology. In this work, we systematically examine the impact of methodological details in activation patching, including evaluation metrics and corruption methods. In several settings of localization and circuit discovery in language models, we find that varying these hyperparameters could lead to disparate interpretability results. Backed by empirical observations, we give conceptual arguments for why certain metrics or methods may be preferred. Finally, we provide recommendations for the best practices of activation patching going forwards.

**Summary:** Shows that activation-patching conclusions change with the corruption method, the metric (logit difference vs probability) and the direction of patching, and gives recommended defaults. Needed by any welfare experiment whose central claim is causal.

### [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. (21 authors), 2023-10-02. Full text: `md/zou_representation-engineering_2310.01405.md`

**Abstract:**

> We identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including honesty, harmlessness, power-seeking, and more, demonstrating the promise of top-down transparency research. We hope that this work catalyzes further exploration of RepE and fosters advancements in the transparency and safety of AI systems. Code is available at github.com/andyzoujm/representation-engineering.

**Summary:** Representation engineering (RepE): top-down reading (linear artificial tomography) and control of population-level representations of concepts such as honesty, harmlessness, power-seeking, emotion and utility; the source of the contrastive-reading-vector recipe that most emotion and welfare probing builds on.

### [Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207)

Wes Gurnee, Max Tegmark, 2023-10-03. Full text: `md/gurnee-tegmark_space-and-time_2310.02207.md`

**Abstract:**

> The capabilities of large language models (LLMs) have sparked debate over whether such systems just learn an enormous collection of superficial statistics or a set of more coherent and grounded representations that reflect the real world. We find evidence for the latter by analyzing the learned representations of three spatial datasets (world, US, NYC places) and three temporal datasets (historical figures, artworks, news headlines) in the Llama-2 family of models. We discover that LLMs learn _linear_ representations of space and time across multiple scales. These representations are robust to prompting variations and unified across different entity types (e.g. cities and landmarks). In addition, we identify individual “space neurons” and “time neurons” that reliably encode spatial and temporal coordinates. While further investigation is needed, our results suggest modern LLMs learn rich spatiotemporal representations of the real world and possess basic ingredients of a world model.

**Summary:** Llama-2 models learn linear representations of space (world, US, NYC places) and time (historical figures, artworks, headlines) across scales, robust to prompting and unified across entity types, with individual space and time neurons. Evidence that LLMs hold coherent world-structure rather than only surface statistics.

### [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nicholas L. Turner, Cem Anil, Carson Denison, Amanda Askell, et al. (Anthropic), 2023-10-04. Full text: `md/bricken_towards-monosemanticity_2023-10.md`

**Opening paragraphs (no formal abstract):**

> Mechanistic interpretability seeks to understand neural networks by breaking them into components that are more easily understood than the whole. By understanding the function of each component, and how they interact, we hope to be able to reason about the behavior of the entire network. The first step in that program is to identify the correct components to analyze.
>
> In this paper, we use a weak dictionary learning algorithm called a sparse autoencoder to generate learned features from a trained model that offer a more monosemantic unit of analysis than the model's neurons themselves. Our approach here builds on a significant amount of prior work, especially in using dictionary learning and related methods on neural network activations (e.g. ), and a more general allied literature on disentanglement. We also note interim reports  which independently investigated the sparse autoencoder approach in response to Toy Models, culminating in the recent manuscript of Cunningham et al. .

**Summary:** Dictionary learning with a sparse autoencoder on a one-layer transformer decomposes superposed MLP activations into thousands of largely monosemantic features (e.g. DNA, Arabic script, base64), with feature splitting and universality across models.

### [The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets](https://arxiv.org/abs/2310.06824)

Samuel Marks, Max Tegmark, 2023-10-10. Full text: `md/marks-tegmark_geometry-of-truth_2310.06824.md`

**Abstract:**

> Large Language Models (LLMs) have impressive capabilities, but are prone to outputting falsehoods. Recent work has developed techniques for inferring whether a LLM is telling the truth by training probes on the LLM’s internal activations. However, this line of work is controversial, with some authors pointing out failures of these probes to generalize in basic ways, among other conceptual issues. In this work, we use high-quality datasets of simple true/false statements to study in detail the structure of LLM representations of truth, drawing on three lines of evidence: 1. Visualizations of LLM true/false statement representations, which reveal clear linear structure. 2. Transfer experiments in which probes trained on one dataset generalize to different datasets. 3. Causal evidence obtained by surgically intervening in a LLM’s forward pass, causing it to treat false statements as true and vice versa. Overall, we present evidence that at sufficient scale, LLMs _linearly represent_ the truth or falsehood of factual statements. We also show that simple difference-in-mean probes generalize as well as other probing techniques while identifying directions which are more causally implicated in model outputs.

**Summary:** High-quality true/false datasets show clear linear structure in LLM representations of truth, probes transfer across datasets, and surgical interventions make the model treat false statements as true and vice versa; simple difference-in-means probes generalize as well as others while finding more causally implicated directions. The most-cited missing work in the included texts' reference lists and the template, including the generalization and causality checks, that the valence-axis papers follow.

### [Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213)

Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, David Bau, 2023-10-23. Full text: `md/todd_function-vectors_2310.15213.md`

**Abstract:**

> We report the presence of a simple neural mechanism that represents an inputoutput function as a vector within autoregressive transformer language models (LMs). Using causal mediation analysis on a diverse range of in-context-learning (ICL) tasks, we find that a small number attention heads transport a compact representation of the demonstrated task, which we call a _function vector_ (FV). FVs are robust to changes in context, i.e., they trigger execution of the task on inputs such as zero-shot and natural text settings that do not resemble the ICL contexts from which they are collected. We test FVs across a range of tasks, models, and layers and find strong causal effects across settings in middle layers. We investigate the internal structure of FVs and find while that they often contain information that encodes the output space of the function, this information alone is not sufficient to reconstruct an FV. Finally, we test semantic vector composition in FVs, and find that to some extent they can be summed to create vectors that trigger new complex tasks. Our findings show that compact, causal internal vector representations of function abstractions can be explicitly extracted from LLMs.

**Summary:** A small set of attention heads transports a compact vector representing an in-context task; transplanting it triggers the task in zero-shot and natural-text contexts, with strong causal effects in middle layers, and vectors partially compose by addition. A table stores completions, not transportable operations.

### [The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/abs/2311.03658)

Kiho Park, Yo Joong Choe, Victor Veitch, 2023-11-07. Full text: `md/park_linear-representation-hypothesis_2311.03658.md`

**Abstract:**

> Informally, the “linear representation hypothesis” is the idea that high-level concepts are represented linearly as directions in some representation space. In this paper, we address two closely related questions: What does “linear representation” actually mean? And, how do we make sense of geometric notions (e.g., cosine similarity and projection) in the representation space? To answer these, we use the language of counterfactuals to give two formalizations of linear representation, one in the output (word) representation space, and one in the input (context) space. We then prove that these connect to linear probing and model steering, respectively. To make sense of geometric notions, we use the formalization to identify a particular (non-Euclidean) inner product that respects language structure in a sense we make precise. Using this _causal inner_ _product_, we show how to unify all notions of linear representation. In particular, this allows the construction of probes and steering vectors using counterfactual pairs. Experiments with LLaMA2 demonstrate the existence of linear representations of concepts, the connection to interpretation and control, and the fundamental role of the choice of inner product. Code is available at [github.com/KihoPark/linear](https://github.com/KihoPark/linear_rep_geometry) ~~r~~ ep ~~g~~ eometry.

**Summary:** Formalizes 'linear representation' in both output and input spaces using counterfactuals, proves the connections to linear probing and steering, and identifies a causal inner product under which the geometric notions make sense (LLaMA-2 experiments). Supplies the shared definition behind the many direction and axis claims in this bibliography and says which geometric claims are testable.

### [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)

Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner, 2023-12-09. Full text: `md/rimsky_contrastive-activation-addition_2312.06681.md`

**Abstract:**

> We introduce Contrastive Activation Addition (CAA), a method for steering language models by modifying their activations during forward passes. CAA computes “steering vectors” by averaging the difference in residual stream activations between pairs of positive and negative examples of a particular behavior, such as factual versus hallucinatory responses. During inference, these steering vectors are added at all token positions after the user’s prompt with either a positive or negative coefficient, allowing precise control over the degree of the targeted behavior. We evaluate CAA’s effectiveness on Llama 2 Chat using multiple-choice behavioral question datasets and open-ended generation tasks. We demonstrate that CAA significantly alters model behavior, is effective over and on top of traditional methods like finetuning and system prompt design, and minimally reduces capabilities. Moreover, we gain deeper insights into CAA’s mechanisms by employing various activation space interpretation methods. CAA accurately steers model outputs and sheds light on how high-level concepts are represented in Large Language Models (LLMs).

**Summary:** Contrastive Activation Addition (CAA): steering vectors from mean activation differences on contrasting multiple-choice answers steer Llama 2 Chat behaviors (sycophancy, corrigibility, survival instinct, myopia, hallucination) more strongly than prompting, with effects stacking with finetuning; the standard steering recipe.

### [Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models](https://arxiv.org/abs/2401.06102)

Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, Mor Geva, 2024-01-11. Full text: `md/ghandeharioun_patchscopes_2401.06102.md`

**Abstract:**

> Understanding the internal representations of large language models (LLMs) can help explain models’ behavior and verify their alignment with human values. Given the capabilities of LLMs in generating human-understandable text, we propose leveraging the model itself to explain its internal representations in natural language. We introduce a framework called Patchscopes and show how it can be used to answer a wide range of questions about an LLM’s computation. We show that many prior interpretability methods based on projecting representations into the vocabulary space and intervening on the LLM computation can be viewed as instances of this framework. Moreover, several of their shortcomings such as failure in inspecting early layers or lack of expressivity can be mitigated by Patchscopes. Beyond unifying prior inspection techniques, Patchscopes also opens up _new_ possibilities such as using a more capable model to explain the representations of a smaller model, and multihop reasoning error correction [1] .

**Summary:** Patchscopes decode a hidden state by patching it into a separate inference pass with an explanatory prompt so the model verbalizes what the state encodes, unifying vocabulary-projection and intervention methods and allowing a stronger model to explain a weaker one. The method for getting a report that is causally sourced from a specific internal state rather than from the prompt.

### [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377)

Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, Hannaneh Hajishirzi, 2024-01-30. Full text: `md/liu_infini-gram_2401.17377.md`

**Abstract:**

> Are _n_ -gram language models still relevant in this era of neural large language models (LLMs)? Our answer is _yes_ , and we showcase their values in both text analysis and improving neural LLMs. This was done by modernizing _n_ -gram LMs in two aspects. First, we train them at the same data scale as neural LLMs— **5 trillion tokens** . This is the largest _n_ -gram LM ever built. Second, existing _n_ -gram LMs use small _n_ which hinders their performance; we instead allow _n_ to be arbitrarily large, by introducing a new ∞ **-gram LM** with backoff. Instead of pre-computing _n_ -gram count tables (which would be very expensive), we develop an engine named **infini-gram** —powered by suffix arrays—that can compute ∞-gram (as well as _n_ -gram with arbitrary _n_ ) probabilities with **millisecond-level latency** . The ∞-gram framework and infini-gram engine enable us to conduct many novel and interesting analyses of human-written and machine-generated text: we find that the ∞-gram LM has fairly high accuracy for next-token prediction (47%), and can complement neural LLMs to greatly reduce their perplexity. When analyzing machine-generated text, we also observe irregularities in the machine–∞-gram agreement level with respect to the suffix length, which indicates deficiencies in neural LLM pretraining and the positional embeddings of Transformers.

**Summary:** Infini-gram: an unbounded-n n-gram model over trillions of tokens, backed by suffix arrays, that predicts from the longest suffix of the context seen in the corpus; counts any string in large pretraining corpora in milliseconds (public indexes and API). A lookup-based reference point for LLM behavior and a tool for tracing where model outputs, including self-reports, come from in training data.

### [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837)

Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva, Sebastian Riedel, 2024-02-26. Full text: `md/yang_latent-multi-hop_2402.16837.md`

**Abstract:**

> We study whether Large Language Models (LLMs) latently perform multi-hop reasoning with complex prompts such as “The mother of the singer of ‘Superstition’ is”. We look for evidence of a latent reasoning pathway where an LLM (1) latently identifies “the singer of ‘Superstition”’ as Stevie Wonder, the _bridge_
>
> _entity_, and (2) uses its knowledge of Stevie Wonder’s mother to complete the prompt. We analyze these two hops individually and consider their co-occurrence as indicative of latent multi-hop reasoning. For the first hop, we test if changing the prompt to indirectly mention the bridge entity instead of any other entity increases the LLM’s internal recall of the bridge entity. For the second hop, we test if increasing this recall causes the LLM to better utilize what it knows about the bridge entity. We find strong evidence of latent multi-hop reasoning for the prompts of certain relation types, with the reasoning pathway used in more than 80% of the prompts. However, the utilization is highly contextual, varying across different types of prompts. Also, on average, the evidence for the second hop and the full multi-hop traversal is rather moderate and only substantial for the first hop. Moreover, we find a clear scaling trend with increasing model size for the first hop of reasoning but not for the second hop. Our experimental findings suggest potential challenges and opportunities for future development and applications of LLMs. [1]

**Summary:** Tests whether models latently traverse a bridge entity ('the mother of the singer of Superstition'): strong evidence for the first hop for some relation types (used in over 80% of prompts), moderate evidence for the second hop and full traversal, highly context-dependent, with a scaling trend for the first hop only. Multi-hop composition is the behavioral signature a backoff table cannot produce, with an honest effect size.

### [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)

Stefan Heimersheim, Neel Nanda, 2024-04-23. Full text: `md/heimersheim-nanda_activation-patching-guide_2404.15255.md`

**Abstract:**

> Activation patching is a popular mechanistic interpretability technique, but has many subtleties regarding how it is applied and how one may interpret the results. We provide a summary of advice and best practices, based on our experience using this technique in practice. We include an overview of the different ways to apply activation patching and a discussion on how to interpret the results. We focus on what evidence patching experiments provide about circuits, and on the choice of metric and associated pitfalls.

**Summary:** Practitioner's guide to activation patching: the ways to apply it, what evidence patching provides about circuits, metric choice and associated pitfalls. Companion to the best-practices paper above.

### [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, et al. (Anthropic), 2024-05-21. Full text: `md/templeton_scaling-monosemanticity_2024-05.md`

**Opening paragraphs (no formal abstract):**

> Eight months ago, we [demonstrated](https://transformer-circuits.pub/2023/monosemantic-features/index.html) that sparse autoencoders could recover monosemantic features from a small one-layer transformer. At the time, a major concern was that this method might not scale feasibly to state-of-the-art transformers and, as a result, be unable to practically contribute to AI safety. Since then, scaling sparse autoencoders has been a major priority of the Anthropic interpretability team, and we're pleased to report extracting high-quality features from Claude 3 Sonnet, [Footnote: For clarity, this is the 3.0 version of Claude 3 Sonnet, released March 4, 2024. It is the exact model in production as of the writing of this paper. It is the finetuned model, not the base pretrained model (although our method also works on the base model).] Anthropic's medium-sized production model.
>
> We find a diversity of highly abstract features. They both respond to and behaviorally cause abstract behaviors. Examples of features we find include features for famous people, features for countries and cities, and features tracking type signatures in code. Many features are multilingual (responding to the same concept across languages) and multimodal (responding to the same concept in both text and images), as well as encompassing both abstract and concrete instantiations of the same idea (such as code with security vulnerabilities, and abstract discussion of security vulnerabilities).
>
> Some of the features we find are of particular interest because they may be safety-relevant – that is, they are plausibly connected to a range of ways in which modern AI systems may cause harm. In particular, we find features related to [security vulnerabilities and backdoors in code](#safety-relevant-code); [bias](#safety-relevant-bias) (including both overt slurs, and more subtle biases); [lying, deception, and power-seeking](#safety-relevant-deception) (including treacherous turns); [sycophancy](#safety-relevant-sycophancy); and [dangerous / criminal content](#safety-relevant-criminal) (e.g., producing bioweapons). However, we caution not to read too much into the mere existence of such features: there's a difference (for example) between knowing about lies, being capable of lying, and actually lying in the real world. This research is also very preliminary. Further work will be needed to understand the implications of these potentially safety-relevant features.

**Summary:** Scales sparse autoencoders to Claude 3 Sonnet, extracting millions of abstract, multilingual, multimodal features (including features for emotions, deception, sycophancy and the model's self-representation) that can be used to steer behavior (e.g. 'Golden Gate Claude').

### [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda, 2024-06-17. Full text: `md/arditi_refusal-single-direction_2406.11717.md`

**Abstract:**

> Conversational large language models are fine-tuned for both instruction-following and safety, resulting in models that obey benign requests but refuse harmful ones. While this refusal behavior is widespread across chat models, its underlying mechanisms remain poorly understood. In this work, we show that refusal is mediated by a one-dimensional subspace, across 13 popular open-source chat models up to 72B parameters in size. Specifically, for each model, we find a single direction such that erasing this direction from the model’s residual stream activations prevents it from refusing harmful instructions, while adding this direction elicits refusal on even harmless instructions. Leveraging this insight, we propose a novel white-box jailbreak method that surgically disables refusal with minimal effect on other capabilities. Finally, we mechanistically analyze how adversarial suffixes suppress propagation of the refusal-mediating direction. Our findings underscore the brittleness of current safety fine-tuning methods. More broadly, our work showcases how an understanding of model internals can be leveraged to develop practical methods for controlling model behavior.^†

**Summary:** Refusal in 13 open chat models is mediated by a single residual-stream direction: ablating it disables refusal and adding it induces refusal; yields a white-box jailbreak by weight orthogonalization. The refusal direction recurs in introspection work (ablating it unlocks injected-thought detection).

### [Evaluating $n$-Gram Novelty of Language Models Using Rusty-DAWG](https://arxiv.org/abs/2406.13069)

William Merrill, Noah A. Smith, Yanai Elazar, 2024-06-18. Full text: `md/merrill_ngram-novelty-rusty-dawg_2406.13069.md`

**Abstract:**

> How novel are texts generated by language models (LMs) relative to their training corpora? In this work, we investigate the extent to which modern LMs generate _n_ -grams from their training data, evaluating both (i) the probability LMs assign to complete training _n_ -grams and (ii) _n_ -novelty, the proportion of _n_ -grams generated by an LM that did not appear in the training data (for arbitrarily large _n_ ). To enable arbitrary-length _n_ -gram search over a corpus in constant time w.r.t. corpus size, we develop RUSTY-DAWG, a novel search tool inspired by indexing of genomic data. We compare the novelty of LM-generated text to humanwritten text and explore factors that affect generation novelty, focusing on the Pythia models. We find that, for _n_ _>_ 4, LM-generated text is _less novel_ than human-written text, though it is _more novel_ for smaller _n_ . Larger LMs and more constrained decoding strategies both _decrease_ _novelty_ . Finally, we show that LMs complete _n_ -grams with lower loss if they are more frequent in the training data. Overall, our results reveal factors influencing the novelty of LMgenerated text, and we release RUSTY-DAWG to facilitate further pretraining data research. [1]

**Summary:** Measures the fraction of n-grams in LM output that never appear in the training corpus (n-novelty, for arbitrarily large n) with Rusty-DAWG, a constant-time n-gram index: for n > 4 generated text is less novel than human text, more novel for small n, and novelty falls with model size and constrained decoding; frequent training n-grams are completed with lower loss (EMNLP 2024). The quantitative bridge between the infini-gram entry and a subject model: how much of a given output could have come from a table.

### [Analyzing the Generalization and Reliability of Steering Vectors](https://arxiv.org/abs/2407.12404)

Daniel Tan, David Chanin, Aengus Lynch, Dimitrios Kanoulas, Brooks Paige, Adria Garriga-Alonso, Robert Kirk, 2024-07-17. Full text: `md/tan_steering-vector-reliability_2407.12404.md`

**Abstract (as listed on arXiv):**

> Steering vectors (SVs) have been proposed as an effective approach to adjust language model behaviour at inference time by intervening on intermediate model activations. They have shown promise in terms of improving both capabilities and model alignment. However, the reliability and generalisation properties of this approach are unknown. In this work, we rigorously investigate these properties, and show that steering vectors have substantial limitations both in- and out-of-distribution. In-distribution, steerability is highly variable across different inputs. Depending on the concept, spurious biases can substantially contribute to how effective steering is for each input, presenting a challenge for the widespread use of steering vectors. Out-of-distribution, while steering vectors often generalise well, for several concepts they are brittle to reasonable changes in the prompt, resulting in them failing to generalise well. Overall, our findings show that while steering can work well in the right circumstances, there remain technical difficulties of applying steering vectors to guide models' behaviour at scale. Our code is available at https://github.com/dtch1997/steering-bench

**Summary:** Steering vectors are unreliable at the per-input level in distribution, with spurious biases contributing to steerability for some concepts, and brittle to reasonable prompt changes out of distribution for several concepts, though they often generalize. The counterweight to the steerable-axis results in this bibliography.

### [Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2](https://arxiv.org/abs/2408.05147)

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, Neel Nanda, 2024-08-09. Full text: `md/lieberum_gemma-scope_2408.05147.md`

**Abstract (as listed on arXiv):**

> Sparse autoencoders (SAEs) are an unsupervised method for learning a sparse decomposition of a neural network's latent representations into seemingly interpretable features. Despite recent excitement about their potential, research applications outside of industry are limited by the high cost of training a comprehensive suite of SAEs. In this work, we introduce Gemma Scope, an open suite of JumpReLU SAEs trained on all layers and sub-layers of Gemma 2 2B and 9B and select layers of Gemma 2 27B base models. We primarily train SAEs on the Gemma 2 pre-trained models, but additionally release SAEs trained on instruction-tuned Gemma 2 9B for comparison. We evaluate the quality of each SAE on standard metrics and release these results. We hope that by releasing these SAE weights, we can help make more ambitious safety and interpretability research easier for the community. Weights and a tutorial can be found at https://huggingface.co/google/gemma-scope and an interactive demo can be found at https://www.neuronpedia.org/gemma-scope

**Summary:** Open suite of JumpReLU sparse autoencoders on all layers and sublayers of Gemma 2 2B and 9B and selected layers of 27B, with instruction-tuned 9B SAEs for comparison and quality metrics released. The obvious SAE toolkit for an open-weights welfare experiment; Llama Scope is the equivalent for Llama.

### [Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html)

Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, Joshua Batson, Christopher Olah (Anthropic), 2024-10-25. Full text: `md/lindsey_sparse-crosscoders_2024-10.md`

**Opening paragraphs (research note; no abstract):**

> This note introduces sparse crosscoders, a variant of sparse autoencoders (e.g. ) or transcoders  for understanding models in superposition . Where autoencoders encode and predict activations at a single layer, and transcoders use activations from one layer to predict the next, a crosscoder reads and writes to multiple layers. Crosscoders produce shared features across layers and even models. They have several applications:
>
> * Cross-Layer Features – Crosscoders allow us to think of features as being spread across layers, resolving cross-layer superposition and tracking persistent features through the residual stream.
> * Circuit Simplification – By tracking features that continue to exist in the residual stream, crosscoders can remove "duplicate features" from analysis and allow features to "jump" across many uninteresting identity circuit connections, and generally simplify circuits.
> * Model Diffing – Crosscoders can produce shared sets of features across models. This includes one model across training or finetuning, and also completely independent models with different architectures.
>
> This note will cover some theoretical examples motivating crosscoders, and then present preliminary experiments applying them to cross-layer superposition and model diffing. We also briefly discuss the theory of how crosscoders might simplify circuit analysis, but leave results on this for a future update.

**Summary:** Introduces crosscoders, dictionaries trained jointly across layers or across two models, and the model-diffing application: read off what post-training added or removed by comparing base and fine-tuned models in a shared feature basis. The direct tool for the base-versus-post-trained contrast; see the Minder et al. entries for its failure modes.

### [Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics](https://arxiv.org/abs/2410.21272)

Yaniv Nikankin, Anja Reusch, Aaron Mueller, Yonatan Belinkov, 2024-10-28. Full text: `md/nikankin_bag-of-heuristics_2410.21272.md`

**Abstract:**

> Do large language models (LLMs) solve reasoning tasks by learning robust generalizable algorithms, or do they memorize training data? To investigate this question, we use arithmetic reasoning as a representative task. Using causal analysis, we identify a subset of the model (a circuit) that explains most of the model’s behavior for basic arithmetic logic and examine its functionality. By zooming in on the level of individual circuit neurons, we discover a sparse set of important neurons that implement simple heuristics. Each heuristic identifies a numerical input pattern and outputs corresponding answers. We hypothesize that the combination of these heuristic neurons is the mechanism used to produce correct arithmetic answers. To test this, we categorize each neuron into several heuristic types—such as neurons that activate when an operand falls within a certain range—and find that the unordered combination of these heuristic types is the mechanism that explains most of the model’s accuracy on arithmetic prompts. Finally, we demonstrate that this mechanism appears as the main source of arithmetic accuracy early in training. Overall, our experimental results across several LLMs show that LLMs perform arithmetic using neither robust algorithms nor memorization; rather, they rely on a “ _bag of heuristics_ ”. [1]

**Summary:** Causal analysis of arithmetic circuits finds sparse neurons implementing simple heuristics (operand in a range, etc.) whose unordered combination explains most arithmetic accuracy, present from early in training: neither a robust algorithm nor memorization. The useful middle case between lookup table and algorithm.

### [LatentQA: Teaching LLMs to Decode Activations Into Natural Language](https://arxiv.org/abs/2412.08686)

Alexander Pan, Lijie Chen, Jacob Steinhardt, 2024-12-11. Full text: `md/pan_latentqa_2412.08686.md`

**Abstract:**

> Top-down transparency typically analyzes language model activations using probes with scalar or single-token outputs, limiting the range of behaviors that can be captured. To alleviate this issue, we develop a more expressive probe that can directly output natural language, performing LATENTQA: the task of answering open-ended questions about activations. A key difficulty in developing such a probe is collecting a dataset mapping activations to natural-language descriptions. In response, we propose an approach for generating a dataset of activations and associated question-answer pairs and develop a fine-tuning method for training a decoder LLM on this dataset. We then validate our decoder’s fidelity by assessing its ability to read and control model activations. First, we evaluate the decoder on a number of supervised reading tasks with a known answer, such as uncovering hidden system prompts and relational knowledge extraction, and observe that it outperforms competitive probing baselines. Second, we demonstrate that the decoder is precise enough to steer the target model to exhibit behaviors unseen during training. Finally, we show that LATENTQA scales well with increasing dataset and model size.

**Summary:** LatentQA trains a decoder LLM to answer open-ended natural-language questions about another model's activations, using a generated dataset of activations and QA pairs; it beats probing baselines at reading hidden system prompts and relational knowledge and is precise enough to steer the target model. The precursor of Activation Oracles.

### [Representation in large language models](https://arxiv.org/abs/2501.00885)

Cameron Yetman, 2025-01-01. Full text: `md/yetman_representation-in-llms_2501.00885.md`

**Abstract:**

> The extraordinary success of recent Large Language Models (LLMs) on a diverse array of tasks has led to an explosion of scientific and philosophical theorizing aimed at explaining how they do what they do. Unfortunately, disagreement over fundamental theoretical issues has led to stalemate, with entrenched camps of LLM optimists and pessimists often committed to very different views of how these systems work. Overcoming stalemate requires agreement on fundamental questions, and the goal of this paper is to address one such question, namely: is LLM behavior driven partly by representation-based information processing of the sort implicated in biological cognition, or is it driven entirely by processes of memorization and stochastic table look-up? This is a question about what kind of algorithm LLMs implement, and the answer carries serious implications for higher level questions about whether these systems have beliefs, intentions, concepts, knowledge, and understanding. I argue that LLM behavior is partially driven by representation-based information processing, and then I describe and defend a series of practical techniques for investigating these representations and developing explanations on their basis. The resulting account provides a groundwork for future theorizing about language models and their successors.

**Summary:** Philosophy paper on whether LLM behavior is driven by representation-based information processing or entirely by memorization and stochastic table lookup, working through Block's Blockhead objection and what would distinguish an LLM from a lookup table; argues for partial representation-based processing and defends practical techniques for investigating it. The closest direct treatment of this takehome's framing device.

### [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L. Turner, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, et al. (Anthropic), 2025-03-27. Full text: `md/lindsey_biology-of-llm_2025-03.md`

**Opening paragraphs of the Introduction (no formal abstract):**

> Large language models display impressive capabilities. However, for the most part, the mechanisms by which they do so are unknown. The black-box nature of models is increasingly unsatisfactory as they advance in intelligence and are deployed in a growing number of applications. Our goal is to reverse engineer how these models work on the inside, so we may better understand them and assess their fitness for purpose.
>
> However, identifying these building blocks is not sufficient to understand the model; we need to know how they interact. In our companion paper, [Circuit Tracing: Revealing Computational Graphs in Language Models](./methods.html), we build on recent work (e.g. ) to introduce a new set of tools for identifying features and mapping connections between them – analogous to neuroscientists producing a “wiring diagram” of the brain. We rely heavily on a tool we call attribution graphs, which allow us to partially trace the chain of intermediate steps that a model uses to transform a specific input prompt into an output response. Attribution graphs generate hypotheses about the mechanisms used by the model, which we test and refine through follow-up perturbation experiments.

**Summary:** Applies attribution graphs (circuit tracing with cross-layer transcoders) to Claude 3.5 Haiku to study multi-step reasoning, planning in poems, multilingual circuits, refusals, hallucination, jailbreaks, chain-of-thought faithfulness and hidden goals; includes evidence about metacognitive circuits (known vs. unknown entities). The reference for circuit-level analyses of a production model.

### [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

Emmanuel Ameisen, Jack Lindsey, Adam Pearce, Wes Gurnee, Nicholas L. Turner, Brian Chen, Craig Citro, et al. (Anthropic), 2025-03-27. Full text: `md/ameisen_circuit-tracing-methods_2025-03.md`

**Introduction, opening paragraphs (no abstract):**

> Deep learning models produce their outputs using a series of transformations distributed across many computational units (artificial “neurons”). The field of mechanistic interpretability seeks to describe these transformations in human-understandable language. To date, our team’s approach has followed a two-step approach. First, we identify features, interpretable building blocks that the model uses in its computations. Second, we describe the processes, or circuits, by which these features interact to produce model outputs.
>
> A natural approach is to use the raw neurons of the model as these building blocks. [Footnote: An alternative approach is to study the roles of gross model components such as entire MLP blocks and attention heads . This approach has identified interesting roles these components play in specific behaviors, but large components play a multitude of unrelated roles across the data distribution, so we seek a more granular decomposition.] Using this approach, previous work successfully identified interesting circuits in vision models, built out of neurons that appear to represent meaningful visual concepts . However, model neurons are often polysemantic — representing a mixture of many unrelated concepts. One reason for polysemanticity is thought to be the phenomenon of superposition , in which models must represent more concepts than they have neurons, and thus must “smear” their representation of concepts across many neurons. This mismatch between the network's basic computational units (neurons) and meaningful concepts has proved a major impediment to progress to the mechanistic agenda, especially in understanding language models.
>
> In recent years, sparse coding models such as sparse autoencoders (SAEs) , transcoders , and crosscoders  have emerged as promising tools for identifying interpretable features represented in superposition. These methods decompose model activations into sparsely active components (“features”), [Footnote: We use 'feature' following the tradition of 'feature detectors' in neuroscience and 'feature learning' in machine learning. Some recent literature uses the term 'latent,' which refers to specific vectors in the model's latent space. We find 'feature' better captures the computational role these elements play, making it more appropriate for describing transcoder neurons than SAE decoder vectors.] which turn out in many cases to correspond to human-interpretable concepts. While current sparse coding methods are an imperfect way of identifying features (see [§ Limitations](#limitations)), they produce interpretable enough results that we are motivated to study circuits composed of these features. Several authors have already made promising steps in this direction .
>
> Although the basic premise of studying circuits built out of sparse coding features sounds simple, the design space is large. In this paper we describe our current approach, which involves several key methodological decisions:

**Summary:** The methods companion to 'On the Biology of a Large Language Model' (included): cross-layer transcoders, replacement models, attribution graphs, and the perturbation protocol for validating a proposed mechanism.

### [Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](https://arxiv.org/abs/2504.02922)

Julian Minder, Clément Dumas, Caden Juang, Bilal Chugtai, Neel Nanda, 2025-04-03. Full text: `md/minder_crosscoder-sparsity-artifacts_2504.02922.md`

**Abstract:**

> Model diffing is the study of how fine-tuning changes a model’s representations and internal algorithms. Many behaviors of interest are introduced during fine-tuning, and model diffing offers a promising lens to interpret such behaviors. Crosscoders are a recent model diffing method that learns a shared dictionary of interpretable concepts represented as latent directions in both the base and fine-tuned models, allowing us to track how concepts shift or emerge during fine-tuning. Notably, prior work has observed concepts with no direction in the base model, and it was hypothesized that these model-specific latents were concepts introduced during fine-tuning. However, we identify two issues which stem from the crosscoders L1 training loss that can misattribute concepts as unique to the fine-tuned model, when they really exist in both models. We develop Latent Scaling to flag these issues by more accurately measuring each latent’s presence across models. In experiments comparing Gemma 2 2B base and chat models, we observe that the standard crosscoder suffers heavily from these issues. Building on these insights, we train a crosscoder with BatchTopK loss and show that it substantially mitigates these issues, finding more genuinely chat-specific and highly interpretable concepts. We recommend practitioners adopt similar techniques. Using the BatchTopK crosscoder, we successfully identify a set of chat-specific latents that are both interpretable and causally effective, representing concepts such as _false information_ and _personal question_, along with multiple refusal-related latents that show nuanced preferences for different refusal triggers. Overall, our work advances best practices for the crosscoder-based methodology for model diffing and demonstrates that it can provide concrete insights into how chat-tuning modifies model behavior. [1]

**Summary:** Two artifacts of the crosscoder L1 loss misattribute concepts as unique to the fine-tuned model when they exist in both; Latent Scaling flags them, a BatchTopK crosscoder largely removes them, and on Gemma 2 2B base vs chat the corrected method finds genuinely chat-specific, causally effective latents (false information, personal questions, refusal triggers). A false-positive mode that matters if you claim post-training created a welfare-relevant state.

### [How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence](https://arxiv.org/abs/2504.02904)

Hongzhe Du, Weikai Li, Min Cai, Karim Saraipour, Zimin Zhang, Himabindu Lakkaraju, Yizhou Sun, Shichang Zhang, 2025-04-03. Full text: `md/du_post-training-reshapes_2504.02904.md`

**Abstract:**

> Post-training is essential for the success of large language models (LLMs), transforming pre-trained base models into more useful and aligned posttrained models. While plenty of works have studied post-training algorithms and evaluated post-training models by their outputs, it remains understudied how post-training reshapes LLMs internally. In this paper, we compare base and post-trained LLMs mechanistically from four perspectives to better understand post-training effects. Our findings across model families and datasets reveal that: (1) Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations; (2) Both truthfulness and refusal can be represented by vectors in the hidden representation space. The truthfulness direction is highly similar between the base and post-trained model, and it is effectively transferable for interventions; (3) The refusal direction is different between the base and post-trained models, and it shows limited forward transferability; (4) Differences in confidence between the base and post-trained models cannot be attributed to entropy neurons. Our study provides insights into the fundamental mechanisms preserved and altered during post-training, facilitates downstream tasks like model steering and benefits future research in interpretability and LLM post-training. Our code is publicly available at [HZD01/post-training-mechanistic-analysis.](https://github.com/HZD01/post-training-mechanistic-analysis)

**Summary:** Systematic base-vs-post-trained comparison: post-training does not move where factual knowledge is stored, the truthfulness direction is nearly the same in both and transfers for interventions, the refusal direction differs and transfers poorly, and confidence differences are not explained by entropy neurons. The empirical backbone for the base-vs-post-trained contrast: which internal structures post-training creates versus reweights.

### [Understanding (Un)Reliability of Steering Vectors in Language Models](https://arxiv.org/abs/2505.22637)

Joschka Braun, Carsten Eickhoff, David Krueger, Seyed Ali Bahrainian, Dmitrii Krasheninnikov, 2025-05-28. Full text: `md/braun_steering-unreliability_2505.22637.md`

**Abstract:**

> Steering vectors are a lightweight method to control language model behavior by adding a learned bias to the activations at inference time. Although steering demonstrates promising performance, recent work shows that it can be unreliable or even counterproductive in some cases. This paper studies the influence of prompt types and the geometry of activation differences on steering reliability. First, we find that all seven prompt types used in our experiments produce a net positive steering effect, but exhibit high variance across samples, and often give an effect opposite of the desired one. No prompt type clearly outperforms the others, and yet the steering vectors resulting from the different prompt types often differ directionally (as measured by cosine similarity). Second, we show that higher cosine similarity between training set activation differences predicts more effective steering. Finally, we observe that datasets where positive and negative activations are better separated are more steerable. Our results suggest that vector steering is unreliable when the target behavior is not represented by a coherent direction.

**Summary:** All seven prompt types tested give a net positive steering effect with high variance and frequent reversals; higher cosine similarity among training activation differences and better separation of positive and negative activations predict more effective steering. Steering is unreliable when the target behavior is not a coherent direction, a diagnostic to run before interpreting a steered welfare state.

### [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954)

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune, 2025-05-29. Full text: `md/zhang_darwin-godel-machine_2505.22954.md`

**Abstract (as listed on arXiv):**

> Today's AI systems have human-designed, fixed architectures and cannot autonomously and continuously improve themselves. The advance of AI could itself be automated. If done safely, that would accelerate AI development and allow us to reap its benefits much sooner. Meta-learning can automate the discovery of novel algorithms, but is limited by first-order improvements and the human design of a suitable search space. The Gödel machine proposed a theoretical alternative: a self-improving AI that repeatedly modifies itself in a provably beneficial manner. Unfortunately, proving that most changes are net beneficial is impossible in practice. We introduce the Darwin Gödel Machine (DGM), a self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks. Inspired by Darwinian evolution and open-endedness research, the DGM maintains an archive of generated coding agents. It grows the archive by sampling an agent from it and using a foundation model to create a new, interesting, version of the sampled agent. This open-ended exploration forms a growing tree of diverse, high-quality agents and allows the parallel exploration of many different paths through the search space. Empirically, the DGM automatically improves its coding capabilities (e.g., better code editing tools, long-context window management, peer-review mechanisms), increasing performance on SWE-bench from 20.0% to 50.0%, and on Polyglot from 14.2% to 30.7%. Furthermore, the DGM significantly outperforms baselines without self-improvement or open-ended exploration. All experiments were done with safety precautions (e.g., sandboxing, human oversight). The DGM is a significant step toward self-improving AI, capable of gathering its own stepping stones along paths that unfold into endless innovation.

**Summary:** A self-improving coding agent that iteratively modifies its own code (including its ability to modify itself), validates each change on coding benchmarks and keeps an archive of diverse agents, raising SWE-bench from 20.0% to 50.0%. The scaffold-level instance of the 'agent that modifies itself' contrast system.

### [Reward Model Interpretability via Optimal and Pessimal Tokens](https://arxiv.org/abs/2506.07326)

Brian Christian, Hannah Rose Kirk, Jessica A. F. Thompson, Christopher Summerfield, Tsvetomira Dumbalska, 2025-06-08. Full text: `md/christian_reward-model-interpretability_2506.07326.md`

**Abstract:**

> Reward modeling has emerged as a crucial component in aligning large language models with human values. Significant attention has focused on using reward models as a means for fine-tuning generative models. However, the reward models themselves— which directly encode human value judgments by turning prompt-response pairs into scalar rewards—remain relatively understudied. We present a novel approach to reward model interpretability through exhaustive analysis of their responses across their entire vocabulary space. By examining how different reward models score every possible single-token response to value-laden prompts, we uncover several striking findings: (i) substantial heterogeneity between models trained on similar objectives, (ii) systematic asymmetries in how models encode high- vs low-scoring tokens, (iii) significant sensitivity to prompt framing that mirrors human cognitive biases, and (iv) overvaluation of more frequent tokens. We demonstrate these effects across ten recent open-source reward models of varying parameter counts and architectures. Our results challenge assumptions about the interchangeability of reward models, as well as their suitability as proxies of complex and context-dependent human values. We find that these models can encode concerning biases toward certain identity groups, which may emerge as unintended consequences of harmlessness training—distortions that risk propagating through the downstream large language models now deployed to millions.

**Summary:** Scores every possible single-token response to value-laden prompts under ten open reward models: large heterogeneity between models trained on similar objectives, asymmetries between high- and low-scoring tokens, prompt-framing sensitivity resembling human biases, overvaluation of frequent tokens, and biases toward identity groups. The one study of reward-model internals, relevant to the 'reward model sharing a base network with a chat model' contrast.

### [Self-Adapting Language Models](https://arxiv.org/abs/2506.10943)

Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, Pulkit Agrawal, 2025-06-12. Full text: `md/zweiger_self-adapting-lms_2506.10943.md`

**Abstract:**

> Large language models (LLMs) are powerful but static; they lack mechanisms to adapt their weights in response to new tasks, knowledge, or examples. We introduce **Se** lf- **A** dapting **L** LMs (SEAL), a framework that enables LLMs to self-adapt by generating their own finetuning data and update directives. Given a new input, the model produces a _self-edit_ —a generation that may restructure the information in different ways, specify optimization hyperparameters, or invoke tools for data augmentation and gradient-based updates. Through supervised finetuning (SFT), these self-edits result in persistent weight updates, enabling lasting adaptation. To train the model to produce effective self-edits, we use a reinforcement learning loop, using the downstream performance of the updated model as the reward signal. Unlike prior approaches that rely on separate adaptation modules or auxiliary networks, SEAL directly uses the model’s generation to parameterize and control its own adaptation process. Experiments on knowledge incorporation and fewshot generalization show that SEAL is a promising step toward language models capable of self-directed adaptation in response to new data. Our website and code is available at `[https://jyopari.github.io/posts/seal](https://jyopari.github.io/posts/seal)` .

**Summary:** SEAL: the model generates its own fine-tuning data and update directives ('self-edits') that become persistent weight updates, trained by an RL loop that rewards self-edits by downstream performance. The concrete instance of the 'agent that fine-tunes itself' contrast, which makes the entity under assessment change between turns.

### [Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](https://arxiv.org/abs/2510.13900)

Julian Minder, Clément Dumas, Stewart Slocum, Helena Casademunt, Cameron Holmes, Robert West, Neel Nanda, 2025-10-14. Full text: `md/minder_narrow-finetuning-traces_2510.13900.md`

**Abstract (as listed on arXiv):**

> Finetuning on narrow domains has become an essential tool to adapt Large Language Models (LLMs) to specific tasks and to create models with known unusual properties that are useful for research. We show that narrow finetuning creates strong biases in LLM activations that can be interpreted to understand the finetuning domain. These biases can be discovered using simple tools from model diffing - the study of differences between models before and after finetuning. In particular, analyzing activation differences on the first few tokens of random text and steering by adding this difference to the model activations produces text similar to the format and general content of the finetuning data. We demonstrate that these analyses contain crucial information by creating an LLM-based interpretability agent to understand the finetuning domain. With access to the bias, the agent performs significantly better compared to baseline agents using simple prompting. Our analysis spans synthetic document finetuning for false facts, emergent misalignment, subliminal learning, and taboo word guessing game models across different architectures (Gemma, LLaMA, Qwen) and scales (1B to 32B parameters). We suspect these biases reflect overfitting and find that mixing pretraining data into the finetuning corpus largely removes them, though residual risks may remain. Our work (1) demonstrates that narrowly finetuned models have salient traces of their training objective in their activations and suggests ways to improve how they are trained, (2) warns AI safety and interpretability researchers that the common practice of using such models as a proxy for studying broader finetuning (e.g., chat-tuning) might not be realistic, and (3) highlights the need for deeper investigation into the effects of narrow finetuning and development of truly realistic case studies for model-diffing, safety and interpretability research.

**Summary:** Narrow fine-tuning leaves strong, readable biases in activations: differencing base and fine-tuned activations on the first tokens of random text, and steering with that difference, reproduces the fine-tuning domain's format and content across synthetic-document, emergent-misalignment, subliminal-learning and taboo-word models (1B-32B); mixing in pretraining data mostly removes the trace (ICLR 2026). Warns that narrowly fine-tuned model organisms are poor proxies for realistic training, a caveat for any welfare study that induces a state by fine-tuning.

### [Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers](https://arxiv.org/abs/2512.15674)

Adam Karvonen, James Chua, Clément Dumas, Kit Fraser-Taliente, Subhash Kantamneni, Julian Minder, Euan Ong, Arnab Sen Sharma, Daniel Wen, Owain Evans, Samuel Marks, 2025-12-17. Full text: `md/karvonen_activation-oracles_2512.15674.md`

**Abstract:**

> Large language model (LLM) activations are notoriously difficult to understand, with most existing techniques using complex, specialized methods for interpreting them. Recent work has proposed a simpler approach known as LatentQA: training LLMs to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language. However, prior work has focused on narrow task settings for both training and evaluation. In this paper, we instead take a generalist perspective. We evaluate LatentQA-trained models, which we call Activation Oracles (AOs), in far out-of-distribution settings and examine how performance scales with training data diversity. We find that AOs can recover information finetuned into a model (e.g., biographical knowledge or malign propensities) that does not appear in the input text, despite never being trained with activations from a fine-tuned model. Our main evaluations are four downstream tasks where we can compare to prior white- and black-box techniques. We find that even narrowlytrained LatentQA models can generalize well, and that adding additional training datasets (such as classification tasks and a self-supervised context prediction task) yields consistent further improvements. Our best AOs match or exceed white-box baselines on all four tasks and the best overall baseline on 3 of 4. These results suggest that diversified training to answer natural-language queries imparts a general capability to verbalize information about LLM activations.

**Summary:** LatentQA-trained 'Activation Oracles' evaluated far out of distribution recover information fine-tuned into a model (biographical facts, malign propensities) that never appears in the input text, improve with training-data diversity, and match or beat white-box baselines on four downstream tasks. The most-cited recent missing method: 'ask the internals, not the mouth' as a self-report channel that routes around the provenance objection.

### [When Models Manipulate Manifolds: The Geometry of a Counting Task](https://arxiv.org/abs/2601.04480)

Wes Gurnee, Emmanuel Ameisen, Isaac Kauvar, Julius Tarng, Adam Pearce, Chris Olah, Joshua Batson, 2026-01-08. Full text: `md/gurnee_manipulate-manifolds_2601.04480.md`

**Abstract:**

> Language models can perceive visual properties of text despite receiving only sequences of tokens—we mechanistically investigate how Claude 3.5 Haiku accomplishes one such task: linebreaking in fixed-width text. We find that character counts are represented on low-dimensional curved manifolds discretized by sparse feature families, analogous to biological place cells. Accurate predictions emerge from a sequence of geometric transformations: token lengths are accumulated into character count manifolds, attention heads “twist” these manifolds to estimate distance to the line boundary, and the decision to break the line is enabled by arranging estimates orthogonally to create a linear decision boundary. We validate our findings through causal interventions and discover “visual illusions”—character sequences that hijack the counting mechanism. Our work demonstrates the rich sensory processing of early layers, the intricacy of attention algorithms, and the importance of combining feature-based and geometric views of interpretability. [1]

**Summary:** Reverse-engineers how Claude 3.5 Haiku counts characters for linebreaking: counts live on low-dimensional curved manifolds discretized by sparse feature families (like place cells), attention heads twist them to estimate distance to the boundary, and the decision is made by arranging estimates orthogonally to form a linear boundary, validated by interventions and 'visual illusions' that hijack the mechanism. A dateable existence proof of continuous geometric computation inside a production model.
