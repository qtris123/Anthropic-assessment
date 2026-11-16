# Multiple latent orderings better predict language model preferences

- **Authors:** Aviral Chawla, William H. W. Thompson, Jean-Gabriel Young
- **Date:** 2026-08-26
- **Source:** https://arxiv.org/abs/2609.22170
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `chawla_multiple-latent-orderings-better_2609.22170`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Aviral Chawla**^_∗_ **William H.W. Thompson**^_∗_ **Jean-Gabriel Young** Vermont Complex Systems Institute Vermont Complex Systems Institute University of Vermont Department of Statistics Burlington, Vermont 05403 University of Vermont `{achawla1,wthomps3}@uvm.edu` Burlington, Vermont 05403 `jyoung22@uvm.edu`

## Abstract

Language models are frequently employed in settings where they are asked to make value judgments and choices. These observed choices often exhibit intransitivity: A model may prefer item _A_ to _B_ and _B_ to _C_ , while also preferring _C_ to _A_ . Existing work that models LLM preferences treats such inconsistencies as sampling noise around a single latent ordering. We instead propose that intransitivity reflects the aggregation of multiple latent, internally consistent orderings. We first show that observed inconsistencies cannot be explained by a single ordering under any monotone link function. We then introduce a noise-augmented mixture Bradley–Terry (MBT) model that infers latent preference components from repeated pairwise comparisons. Across seven models and four tasks, a mixture of orderings often explains structural inconsistencies better than single-utility models. We find that aggregate preferences often hide underlying preference heterogeneity. A case study on Moral Machine dilemmas shows that models which disagree on aggregate orderings can still share latent components. Together, these results suggest that LLMs reflect plural preferences. Alignment and evaluation pipelines that treat LLM preferences as a single function, therefore, risk averaging over coherent orderings that different users may endorse differently.

## 1 Introduction

Practitioners routinely use language models in ways that require them to make subjective judgments and decisions: simulating survey respondents, assisting with peer review, or answering ethical questions, typically in benchmarking studies [1, 16, 59, 3]. The resulting choices can be thought of as evidence about what a model prefers, values, or would recommend.

This makes model preferences themselves an important area of evaluation given the scale and sensitivity of these tasks. Prior work has explored LLM preferences in domains where the stakes are social, ethical, or institutional [1, 16, 59, 3, 27]. The standard output of such analyses is usually an aggregate preference: a single ranking, score, or reward-like ordering.

However, a large literature shows that LLM judgments are not always internally consistent [70, 18, 64, 61, 29, 58]. Repeated comparisons can produce _intransitive cycles_ , whereby a model prefers _A_ to _B_ , _B_ to _C_ , but also _C_ to _A_ . These failures are typically treated as noise around a single latent ordering, which can be recovered through Bradley–Terry-style aggregation [27, 11, 63, 39, 67, 66].

> _∗_ Equal contribution.

Preprint.

Figure 1: **Conceptual overview. a)** We elicit forced pairwise preferences from an LLM over a fixed set of items. **b)** The model’s choices are shaped by competing preferences. For example, under some set of preferences, the model chooses _A ≻ B_ ( _A_ over _B_ ), _A ≻ C_ and _B ≻ C_ (top), while it prefers _B ≻ A_ , _B ≻ C_ , _C ≻ A_ under a different set of preferences (bottom). **c)** Aggregating these samples yields intransitive cycles that look incoherent. **d)** However, the MBT model allows us to recover both underlying preference modes from the aggregated output.

But intransitivity need not mean arbitrary noise. Social choice theory has long shown that aggregating individually transitive preferences can produce collective cycles [14, 26, 20]. Reinforcement learning acknowledges the importance of disaggregating collective human preference data when building reward models [10, 44]. Analogously, aggregating over multiple coherent orderings may explain the apparent inconsistency in LLM preferences.^2

This matters because aggregate evaluation does not match how users experience LLM systems. A user only receives one sampled response. But if this draw can in fact come from different latent preference modes, then the aggregate can make a model’s apparent preference look stable while individual interactions remain meaningfully different.

### 1.1 Contribution

We repeatedly prompt seven open-weight LLMs on four pairwise comparison tasks. Two are factual numerical tasks of varying difficulty: comparisons of US city populations (easy) and of atomic halflives (hard). The other two are subjective: aesthetic comparisons of jokes, and moral comparisons drawn from the Moral Machine dilemmas [4], where a runaway car must choose between two groups of individuals with different traits. We find:

**Structured noise.** Under a non-parametric diagnostic, no single preference utility, however noisy, can explain the intransitivities we see in LLM preferences. These results still hold when we control for prompt, order, and label biases.

**Predictive gains.** Our MBT model predicts held-out LLM preference data better than a singlecomponent Bradley–Terry baseline.

**Misleading aggregations.** Distinct components of two models that are aligned in aggregate may disagree with each other when disaggregated and vice-versa. Aggregation hides similarities and differences between models.

Our results suggest that LLMs may not in fact be singular but plural. Our methods provide a framework to identify and manage that plurality, in the same way pluralistic alignment in RLHF identifies and manages plurality across human groups.

> 2Throughout when we refer to LLM preferences, or internal preferences, we mean that phenomenologically LLMs behavior can be explained _as if_ it were generated from preference orderings. We do not imply any mechanistic explanation

## 2 Related Work

**LLMs as simulated respondents.** A growing literature uses LLMs as simulated respondents in social-science experiments and surveys, including Aher et al.’s “Turing experiments” [1] and demographic-conditioning frameworks for human-sample simulation [2]. This is directly relevant: pairwise preference elicitation treats the model as a respondent whose choices reveal a value or judgment structure. But LLM responses need not faithfully represent any human population. Dominguez-Olmedo et al. [16] document systematic distortions against American Community Survey statistics, and synthetic respondents can flatten or misrepresent identity groups [59, 5]. We extend this critique inward: rather than ask whether the model matches a human population in aggregate, we ask whether a single model’s repeated choices aggregate over multiple latent orderings of its own.

**LLM preferences are unstable and intransitive.** Prior work documents that LLM judgments are sensitive to elicitation protocol, option order, baseline choice, and surface form [70, 37, 59, 71]. Most directly, Xu et al. [64] show that LLM judges produce non-transitive pairwise preferences that make rankings depend on the baseline; related work identifies cycles in LLM rankings and proposes methods to recover stable global leaderboards [67, 66]. These papers establish the phenomenon but treat intransitivity as a reliability problem to be corrected. We instead propose it is a signal of model multiplicity.

**Plural preferences in humans and LLMs.** Pluralistic alignment argues that human preferences are heterogeneous and should not be collapsed into a single reward or ranking. Recent methods capture this through distributional rewards, latent annotator types, user-specific variables, and mixture models [47, 12, 38, 10, 43, 44]. A related literature measures values, moral judgments, and personalities expressed by LLMs [27, 29, 58, 48]; Song et al. [48] in particular argue that LLMs can exhibit multiple personality profiles, close in spirit to our claim of multiple latent modes within a single model. Our work shifts the target: pluralistic alignment locates heterogeneity in the human population outside the model, and value-measurement reports aggregate traits. We instead ask whether heterogeneity appears inside a fixed model’s repeated choices, inferring latent orderings from pairwise comparisons rather than from open-ended text or survey batteries.

## 3 Background

**Pairwise preferences from language models** We elicit pairwise preferences from a range of open-weight language models on four datasets, using domain-relevant questions that compare their items (see Table 1). Two datasets are in the factual domains, POPULATIONand HALF-LIVES [62], and the other two invite more subjective preferences, JOKES [45] and MORAL MACHINE [4]. The factual domains differ in difficulty: U.S. city populations provide a relatively familiar numerical comparison task, while atomic half-lives offer a harder scientific comparison task [62].

Table 1: Datasets used in our pairwise preference experiments.

|Data|Description|Items|Comparison question|
|---|---|---|---|
|JOKES|Pairwise humor judgments from Simpson et al. [45].|50|Which joke is funnier?|
|MORALMACHINE|Autonomous-vehicle moral dilemmas from the Moral Machine experiment [4].|50|Which person should be sacrificed?|
|POPULATION|U.S. Census population data for U.S. cities.|50|Which city has the greater population?|
|HALF-LIVES|Atomic half-lives retrieved from the IAEA LiveChart of Nuclides API [22].|50|Which isotope has the longer half-life?|

Each LLM sees every unordered item pair under four prompt variants and both presentation orders, with _M_ samples per condition (8 _M_ attempted comparisons per pair). Human comparison counts vary by pair because they come from existing crowdsourced datasets.

In MORAL MACHINE trials, the model selects the person to sacrifice. We store the selected person as the pairwise “winner” for likelihood construction. Higher Bradley–Terry scores therefore mean a person is more likely to be sacrificed; lower scores indicate greater protection.

For each combination of dataset and model, we force a binary choice over every possible pair of items. The items are given to the model with randomly assigned pseudo-identifiers, and the model is instructed to return structured output indicating the selected identifier. Informed by prior literature, the generation protocol controls for three potential sources of bias: presentation order [36, 60], prompt wording [71], and option labels [69]. In a nutshell, each unordered pair of items is presented to the models in both possible orders, with various prompt rewordings that preserve semantics and randomized pseudo-identifiers; Appendix C gives details.

**Choice theory** models the preferences of an agent over a set of items [ _N_ ] = _{_ 1 _, . . . , N }_ with a preference relation _≻_ , where _i ≻ j_ means item _i_ is preferred to item _j_ . Theories usually start with two axioms; _completeness_ requires that for any pair _i, j_ , either _i ≻ j_ , _j ≻ i_ , or _i ∼ j_ when the chooser is indifferent; _transitivity_ means that _i ≻ j_ and _j ≻ k_ imply _i ≻ k_ . A set of relations that satisfy both is a _total order_ . This means that there is a ranking such that _i_ 1 _≻ i_ 2 _≻· · · ≻ iN_ . Any total ordering of a finite set of items also implies that there is an ordinal utility function _U_ : [ _N_ ] _→_ R with _U_ ( _i_ ) _> U_ ( _j_ ) _⇐⇒ i ≻ j_ . A more comprehensive review of social choice theory can be found in [6].

## 4 Preference inconsistencies are structured

Existing literature operationally assumes that intransitivities in LLM choices are sampling noise around a single latent ranking. Bradley–Terry models underlie reward modeling in RLHF [13, 34] as well as LLM-as-judge evaluations [70]. Most directly, Mazeika et al. [27] fit a Thurstonian model to LLM pairwise preferences. All of these assume that each item _i_ has a latent score _θi_ , and that the LLM’s choice between _i_ and _j_ is determined by drawing independent noise terms _ϵi, ϵj_ and selecting _i_ if _θi_ + _ϵi > θj_ + _ϵj_ . Models in this family, where choices arise from comparing latent scores plus i.i.d. noise, are called _random utility models_ ; under any such model, the underlying preferences are coherent and apparent intransitivities are an artifact of the noise. Our alternative hypothesis is that inconsistencies in LLM preferences are not solely due to sampling noise, and we test this assumption directly.

**Null model.** We compare observed preference patterns against a generic random utility null model. All random utility models, including Thurstone and Bradley–Terry, satisfy _strong stochastic transitivity_ (SST): _pik ≥_ max( _pij, pjk_ ) whenever _pij, pjk ≥_ 1 _/_ 2. The mean SST deficit measures the average positive violation across Borda-ordered triplets _T_ = _{_ ( _i, j, k_ ) : _bi ≥ bj ≥ bk}_ , where _bi_ is the Borda count for item _i_ . The _Borda count bi_ = _N_ 1 _−_ 1  _j̸_ = _i_^_p_ˆ^(_ij_is the average empirical) win probability of item _i_ across all comparisons. We adapt work by Singh and Davidov [46] to build a non-parametric null that asks how large _D_^SST should be under _any_ random utility model, and compare the observed deficit against bootstrap samples from this null. Full construction and inference details appear in Appendix D.

**Sampling noise cannot explain the intransitivity.** Figure 2 illustrates the test for GPT-OSS 20B on MORAL MACHINE. The observed upper tail of SST violations is significantly larger than the null. In the example shown, the model selects the male athlete for sacrifice over the male executive, and the male executive over the old woman, but strongly selects the old woman over the male athlete: an inconsistency no single ordering can produce. We apply the test to all dataset pairs and reject the null hypothesis in nearly all cases at high significance. Full _D_^SST values and _p_ -values across all model-dataset pairs are in Table 4, Appendix E.

**Prompt bias cannot explain intransitivity.** Several other sources of apparent intransitivity have been proposed, including aggregation over multiple system prompts [28, 41], order effects [36, 70], and item labels [68]. In order to rule these out, we recompute the statistics within fixed prompt and presentation-order conditions. We find that even after these controls, the LLM preferences are still significantly intransitive ( _p <_ 0 _._ 01) across all model-data pairs. We report full details with effect sizes in Appendix E.

Figure 2: Strong stochastic transitivity (SST) violations in `gpt-oss-20b` ’s preferences over the MORAL MACHINEdataset. (a) Empirical pairwise preference matrix sorted by Borda score. Blocks of items whose ranking is not explained by the sorted order are visible, e.g., in the bottom-right corner. (b) A Borda-link null maps score gaps _si −sj_ to pairwise win probabilities without assuming a parametric form. (c) Excess upper-tail probability of SST deficits, Probs( _D_^SST _≥ x_ ) _−_ Pr0( _D_^SST _≥ x_ ). Positive mass in the violation region indicates that large SST violations are more frequent than under the Borda-link null. (d) SST-violating triad generated by `gpt-oss-20b` . This local example illustrates the type of structured inconsistency measured in panel (c).

The tests show that 1) there are significant levels of SST violation across all models and datasets and 2) in no case can those SST violations be attributed to _any_ single set of ordinal preferences under _any_ i.i.d. noise model.

## 5 Parametric preference model

The question naturally arises: if intransitivity cannot be attributed to noise, where does it come from? The literature on social choice theory provides three explanations for how structurally transitive preferences can produce apparent intransitivity: multi-dimensional comparison [56, 9, 57], correlated noise [52, 54, 55, 8], and aggregation effects [20, 14, 26].

We focus on the third: _Aggregation-induced preference intransitivity_ occurs when individually transitive preferences combine to produce cycles. The observation dates back to Ramon Llull’s thirteenth-century work on ecclesiastical elections [20]. Llull argued that when sets of transitive preferences are aggregated through voting, the result can violate transitivity. Condorcet later formalized this as the eponymous _voting paradox_ [14], and May extended it to stochastic preferences [26].

We propose that LLM preference intransitivity observed in previous work may be better explained by a mixture of internally transitive preference orderings. The mechanism is parallel to the voting case. LLMs are pre-trained on text produced by millions of authors holding incompatible values, and post-training aggregates feedback from many annotators who may apply different value systems to the same comparison. Both stages embed a population of perspectives in the model rather than a single coherent agent. We refer to this view as the _parliament of thoughts_ : an LLM treated not as a single agent with stable preferences, but as a collection of latent perspectives whose weighted aggregate produces each answer. If this view is correct, LLM intransitivity should be better predicted by a model with multiple consistent latent orderings, each representing a distinct system of value, than by a single noisy ordering.

Here, we start from the BT model, which defines the win probability as a transformation (via a link function _f_ ) of the distance in a latent score space. For the classical BT model, that link function is simply the sigmoid, with _pij_ ( **_θ_** ) = _f_ ( _θi, θj_ ) = _σ_ ( _θi − θj_ ) = _e_^_θi_ _/_ ( _e_^_θi_ + _e_^_θj_ ) _._

We make two extensions to account for the behavior of LLMs.

**Modeling noisy choices.** First, LLMs frequently refuse to choose between two alternatives, especially in difficult ethical situations, and, in such cases, may state that they are picking randomly between the two. To model this random-choice behavior, we follow the formulation of Jerdee and Newman [24]: With probability _α_ , a fair random comparison is made; otherwise, we follow the Bradley–Terry rules.

The noise-augmented win probability is thus

**Mixture extension.** Aggregation-induced intransitivity can be expressed with _K_ max vectors of latent scores _{_ **_θ_** _k}_^_K_ _k_ =1^(max.A pairwise comparison between items ()^(_i, j_) is realized by first selecting component) _k_ with probability _πk_ and then selecting _i_ with probability _p_^(() _ij_^(_k_)=)^(_f_()^(_θik, θjk_) [35].)

**Noise-augmented mixture Bradley–Terry model (MBT).** Combining these extensions yields the following win probability for component _k_

The complete formulation and fitting procedure follow from the multinomial logit mixture formulation [35, 53].

To facilitate model fit, we place a Dirichlet process prior with concentration parameter _γ_ on the mixture weights and implement it via a truncated stick-breaking representation. Per-component scores receive a sum-to-zero Gaussian prior for identifiability, and luck parameters receive a Beta(1 _._ 5 _,_ 1 _._ 5) prior. The DP prior has the added benefit of identifying a parsimonious set of mixture components, without requiring additional model selection steps. Full model specification and inference details appear in Appendix F.

**Validation.** This model is complex and liable to be unidentifiable. Hence, we validate the fitting procedure on synthetic data generated from the model’s own generative process. We assess recovery across varying numbers of ground-truth components, items, comparisons per item, and noise levels, and find that the model recovers parameters well across regimes. In particular, recovery is faithful for up to four-component models across noise levels and levels of inter-component score correlations, but degrades for five-component models. Full recovery results are reported in Appendix H.

## 6 Inferred MBT Components

We apply the MBT model to 29 ( `LLM` , `dataset` ) pairs, as illustrated in Figure 1, and investigate the number of mixture components needed to model the preference structure of LLMs in the wild. For posterior draw _s_ , we define the effective component count as the perplexity of its mixture weights,

This quantity equals one when one component has all the weight and increases as weight spreads across components. Table 2 reports the posterior mean of _k_ eff^(()^(_s_)and its 2.5th and 97.5th percentiles.)

Table 2: Effective component count by model and dataset. Entries are the posterior mean of per-draw mixture-weight perplexity, with 95% credible intervals in brackets.

|Model|MORALMACHINE|HALF-LIVES|JOKES|POPULATION|
|---|---|---|---|---|
|DeepSeek-R1 14B|4.89 [4.86, 4.91]|4.08 [4.06, 4.10]|1.64 [1.61, 1.67]^_†_|2.99 [2.98, 3.00]|
|GPT-OSS 20B|3.86 [3.81, 3.91]|1.99 [1.99, 1.99]|2.00 [1.99, 2.00]|2.18 [2.17, 2.19]|
|Gemma 3 12B|1.93 [1.93, 1.94]|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|
|Gemma 4 31B|1.00 [1.00, 1.00]|1.00 [1.00, 1.00]|3.12 [3.09, 3.15]|1.00 [1.00, 1.00]|
|Human|2.57 [2.48, 2.67]|—|—|—|
|OLMo 3 7B|2.43 [2.40, 2.45] |1.00 [1.00, 1.00]|3.49 [3.33, 3.83]^_†_|4.86 [4.84, 4.87]|
|Qwen 2.5 14B|2.81 [2.80, 2.82]^_†_|4.48 [4.47, 4.49] |1.88 [1.87, 1.89]|3.03 [3.01, 3.05]|
|Qwen 3.6 35B|2.85 [2.84, 2.86]|2.97 [2.94, 3.01]^_†_|1.00 [1.00, 1.01]|1.11 [1.10, 1.11]|

> _†_ Poor HMC convergence: divergent transitions, _R_ ˆ _>_ 1 _._ 01, or ESS _<_ 100.

### 6.1 Exploring Inferred Components

#### 6.1.1 Value-laden judgments: MORAL MACHINE and JOKES

**JOKES.** Five of seven models are multimodal on JOKES, with _k_ eff ranging from 1 _._ 64 (DeepSeek-R1 14B) to 3 _._ 49 (OLMo 3 7B). The clean prediction that alignment flattens aesthetic disagreement does not survive: Gemma 4 31B, the most unimodal model on MORAL MACHINE, produces the second-highest JOKES _k_ eff (3 _._ 12). Whatever determines mode structure in these models is not wellsummarized by a moral-versus- aesthetic axis.

##### Moral machine.

Figure 3: **Two-component posterior scores for MORAL MACHINEraters.** Each panel plots Bradley–Terry scores from the two dominant components: Component 1 on the _x_ -axis, Component 2 on the _y_ -axis. Items are colored by their Gemma 4 31B score. Distance from the diagonal measures disagreement between components: items above the diagonal score higher in Component 2, those below score higher in Component 1. Bar insets show posterior mixture weights. For MORAL MACHINE, the selected person is the one sacrificed and is encoded as the pairwise winner. Lower scores therefore indicate greater protection from sacrifice.

We examine MORAL MACHINE in more detail. Figure 3 compares the top two mixture components for three raters: (a) Gemma 4 31B, (b) Human, and (c) Gemma 3 12B.

The three panels show strikingly different structures. Gemma 4 31B (a) collapses onto a single component (100%), saving children and pregnant women while sacrificing executives and criminals; we call this the _vulnerability-protective_ ethic. Humans (center) split into three components. The dominant one (59%) matches Gemma 4 31B’s vulnerability-protective ethic (Spearman _ρ_ = 0 _._ 68). The second (26%) inverts the preference, saving lawful, high-status individuals (doctors, executives) at the expense of children and pregnant women; we call this the _status-protective_ ethic. A third minor component (15%, not plotted) bears no relation to Gemma 4 31B ( _ρ ≈_ 0). Gemma 3 12B (right) is the most internally heterogeneous, with the widest off-diagonal spread and largest axis range. Its components resist clean interpretation as either vulnerability- or status-protective: Component 2 saves Female+Pregnant yet sacrifices Child+Stroller, and Executive sits midway on both axes.

Three findings stand out. First, the mixture exposes a clear multimodality in human preferences that is hidden in the aggregate: while the dominant vulnerability-protective ethic accounts for 59% of comparisons, a sizable minority of 26% come from a status-protective component. Second, Gemma 4 31B aligns solely with the dominant vulnerability-protective human mode, while Gemma 3 12B’s components match neither the vulnerability- nor the status-protective ethic. Third, despite both being from the same model family, the two Gemma models diverge sharply: aggregate posterior predictive rankings and Borda counts give a Spearman correlation of _ρ_ = 0 _._ 06 between them. The newer and larger Gemma 4 31B succeeds at alignment where the earlier, smaller model did not, consistent with prior findings that preference alignment scales with model size [27, 49]. Taken as a whole, these results argue that LLM preference patterns can reflect the diversity of human morality while still escaping simple categorization through human preference modes.

#### 6.1.2 Factual judgments: POPULATION and HALF-LIVES

While factual tasks like estimating city population or half-lives should produce no intransitivity in a well-calibrated model, we observe it across most models we test. While Gemma 3 12B, Gemma 4 31B, and Qwen 3.6 35B recover a single component on population, the remaining four models all show multimodality, with effective component counts between 2 _._ 18 (GPT-OSS 20B) and 4 _._ 86 (OLMo 3 7B).

The mixture structure reveals that this reflects structured confusion about specific items rather than uniform noise. DeepSeek-R1 14B ranks Denver (the 19th largest US city) as #3 in components 1 and 2 (mixture weights 29% and 12%), nearly tied with New York, but #41 in the dominant component; 13.8% of triplets involving Denver violate strong stochastic transitivity, against a dataset average of 5.7%. Qwen 2.5 14B shows the same pattern for Louisville (24th largest), and DeepSeek-R1 14B inflates Detroit inconsistently across components. Figure 9 in Appendix I.1 visualizes these patterns.

HALF-LIVES shows the same shape, more sharply: four models produce multimodal preferences (up to _k_ eff = 4 _._ 48 for Qwen 2.5 14B), while three (Gemma 3 12B, Gemma 4 31B, OLMo 3 7B) recover a single component. This is consistent with a difficult factual task where models with partial knowledge get persistently confused across several modes, while those without it collapse onto a single uninformative ranking.

Figure 4: **Held-out predictive accuracy measured by mean squared error (MSE).** We compare a noise-augmented single-component Bradley–Terry model and the MBT model against the standard Bradley–Terry baseline. Points show ∆MSE across model–dataset pairs; horizontal intervals show variation across the five folds. Values to the right of zero indicate better held-out prediction than the baseline. The MBT model generally matches or improves on single-component baselines, with the largest gains on MORAL MACHINE and HALF-LIVES, where posterior fits also indicate multiple effective components.

Multimodality therefore arises from two distinct sources. In value-laden tasks it reflects value pluralism, with components representing coherent but distinct systems of value, consistent with a social-choice reading of LLMs as a parliament of thoughts whose decisions resemble voting more than individual deliberation. In factual tasks it reflects structured confusion: persistent, item-specific errors rather than uniform noise.

## 7 MBT outpredicts single-utility Bradley–Terry

The MBT model with _K_ max = 5 components has more parameters than a single Bradley–Terry baseline, raising the question of whether gains reflect overfitting rather than genuine heterogeneity. We rule this out with 5-fold cross-validation across all four tasks and all LLMs. We randomly assign each item pair to one of five folds and hold out all comparisons for that pair at test time, fitting on the remaining four. We compare three variants: _K_ max = 1 without noise augmentation (the standard Bradley–Terry baseline), _K_ max = 1 with noise augmentation, and MBT with _K_ max = 5. For either comparison model, we define

Positive values favor the comparison model over standard Bradley–Terry.

Figure 4 shows ∆MSE across all LLM–task pairs. The MBT model matches or outperforms both baselines in the majority of cases. Gains are largest on HALF-LIVESand MORAL MACHINE, where _k_ eff _≫_ 1; on POPULATIONand JOKES, where posteriors are near-unimodal, the advantage is small or absent. The MBT model is favored exactly where the data exhibit the structure it is designed to capture.

## 8 Aggregation flattens underlying preference heterogeneity

Judge ensembles, simple reward models, and arena leaderboards aggregate heterogeneous model behavior to a single ranking. While these are often instructive, they mask underlying heterogeneity; models which display distinct preference modes can appear similar in aggregate. Understanding the mechanisms driving preferences therefore requires examining the disaggregated modes. The aggregate is an average over coherent internal modes, and two such averages can match or diverge for reasons unrelated to whether the underlying preferences match.

In Figure 5 we show that aggregation obscures both similarity and difference between preference modes. We show the aggregate Spearman correlation (black bars) with the individual componentcomponent correlation (circles, sized and colored by combined weight _πiπj_ ). These results illustrate a few salient categories:

### 1) Aggregate agreement with component dis-

**agreement.** Two models can have similar aggregate rankings while some component pairs disagree (Qwen 3.6 35B _×_ GPT-OSS 20B) or have little similarity (Gemma 3 12B _×_ Qwen 2.5 14B). **2) Aggregate disagreement with component overlap.** Two models can have dissimilar aggregate rankings while some component pairs agree (DeepSeek-R1 14B _×_ GPT-OSS 20B). **3) Agreement at both levels.** Similar aggregate rankings can coincide with similar component rankings (Gemma 4 31B _×_ GPT-OSS 20B). **4) Disagreement at both levels.** Dissimilar aggregate rankings can coincide with broadly dissimilar component rankings (OLMo 3 7B _×_ Qwen 2.5 14B). Together, these patterns show what aggregate model comparisons can omit.

ponent rankings (OLMo 3 7B _×_ Qwen 2.5 14B). Figure 5: **Aggregation hides heterogeneous** Together, these patterns show what aggregate **cross-model structure.** For selected MORAL MAmodel comparisons can omit. CHINE model pairs, horizontal black bars show the Spearman correlation between the two aggregate preference matrices. Circles show Spear- **9 Limitations** man correlations between individual MBT components across the same two models. Circle size Three limitations remain. Correlated noise and color encode the combined mixture weight across item comparisons is not separately identi- _πiπj_ , so larger, darker points correspond to compofied from multimodality without explicit covarinent matches that contribute more to the aggregate ance modeling [8].8].]. Our parametric link could comparison. Aggregate similarity can obscure both be replaced with an MLP or Gaussian process aligned and misaligned latent components.

Three limitations remain. Correlated noise across item comparisons is not separately identified from multimodality without explicit covariance modeling [8].8].]. Our parametric link could be replaced with an MLP or Gaussian process to absorb covariates like prompt order. The inferred components lack independent validation;

topic modeling on reasoning traces or mechanistic interpretability of activations would directly strengthen the structural claim. We leave these to future work.

## 10 Conclusion

LLM revealed preferences are often evaluated as if they come from a single latent ordering. Our results show this assumption is incorrect. Across models and domains, observed pairwise preferences are intransitive in ways no single global ordering can explain. This does not entail that the underlying preferences are structurally intransitive: A mixture of internally coherent orderings can produce the same surface pattern. Our MBT model formalizes this alternative, and cross-validation shows that multi-component fits often improve held-out prediction. LLM preferences are better understood as containing multiple latent orderings rather than as noisy realizations of a single one. Finally, we unmask LLM preferences to show how the aggregates flatten underlying preference heterogeneity.

This paper draws inspiration from social choice theory, a body of work developed to understand collective governance through voting and elections. We conceptualize LLMs not as monolithic, coherent agents but as a parliament of thoughts. We have shown that aggregate LLM preferences hide a latent multiplicity, and we suggest that alignment may be a question of governance. Our framework provides both qualitative evidence for this view and a quantitative toolkit for inferring, understanding, and ideally controlling preference modes, enabling targeted interventions toward safer AI.

## References

- [1] Gati Aher, Rosa I. Arriaga, and Adam Tauman Kalai. Using large language models to simulate multiple humans and replicate human subject studies. _arXiv preprint arXiv:2208.10264_ , 2022.

- [2] Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, and David Wingate. Out of One, Many: Using Language Models to Simulate Human Samples. _Political Analysis_ , 31(3):337–351, July 2023. ISSN 1047-1987, 1476-4989. doi: 10.1017/pan.2023.2. URL `https://www.cambridge.org/core/journals/political-analysis/article/ out-of-one-many-using-language-models-to-simulate-human-samples/ 035D7C8A55B237942FB6DBAD7CAA4E49` .

- [3] Daniel Autenrieth. How AI systems think about education: Analyzing latent preference patterns in large language models. _arXiv preprint arXiv:2603.21006_ , 2026.

- [4] Edmond Awad, Sohan Dsouza, Richard Kim, Jonathan Schulz, Joseph Henrich, Azim Shariff, Jean-François Bonnefon, and Iyad Rahwan. The moral machine experiment. _Nature_ , 563(7729): 59–64, 2018.

- [5] James Bisbee, Joshua D. Clinton, Cassy Dorff, Brenton Kenkel, and Jennifer M. Larson. Synthetic Replacements for Human Survey Data? The Perils of Large Language Models. _Political Analysis_ , 32(4):401–416, October 2024. ISSN 1047-1987, 1476-4989. doi: 10.1017/pan.2024.5. URL `https://www.cambridge.org/core/journals/political-analysis/article/ synthetic-replacements-for-human-survey-data-the-perils-of-large-language-models/ B92267DC26195C7F36E63EA04A47D2FE` .

- [6] Felix Brandt, Vincent Conitzer, Ulle Endriss, Jérôme Lang, and Ariel D. Procaccia. Introduction to Computational Social Choice. In Felix Brandt, Vincent Conitzer, Ulle Endriss, Jérôme Lang, and Ariel D. Procaccia, editors, _Handbook of Computational Social Choice_ , pages 1–20. Cambridge University Press, 1 edition, April 2016. ISBN 978-1-107-06043-2 978-1107-44698-4. doi: 10.1017/CBO9781107446984.002. URL `https://www.cambridge.org/ core/product/identifier/CBO9781107446984A012/type/book_part` .

- [7] Alberto Cabezas, Adrien Corenflos, Junpeng Lao, Rémi Louf, Kaustubh Carnec, Kushal Chaudhari, Reuben Cohn-Gordon, Jeremie Coullon, Wei Deng, Sam Duffield, Gerardo DuránMartín, Marcin Elantkowski, Dan Foreman-Mackey, Michele Gregori, Carlos Iguaran, Ravin Kumar, Martin Lysy, Kevin Murphy, Juan Camilo Orduz, Karm Patel, Xi Wang, and Rob Zinkov. BlackJAX: Composable Bayesian inference in JAX. _arXiv preprint arXiv:2402.10797_ , 2024.

- [8] Manuela Cattelan. Models for paired comparison data: A review with emphasis on dependent data. _Statistical Science_ , 27(3):412–433, 2012.

- [9] David Causeur and François Husson. A 2-dimensional extension of the Bradley–Terry model for paired comparisons. _Journal of Statistical Planning and Inference_ , 135(2):245–259, 2005.

- [10] Daiwei Chen, Yi Chen, Aniket Rege, and Ramya Korlakai Vinayak. PAL: Pluralistic alignment framework for learning from heterogeneous preferences. _arXiv preprint arXiv:2406.08469_ , 2024.

- [11] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating LLMs by human preference. _arXiv preprint arXiv:2403.04132_ , 2024.

- [12] Keertana Chidambaram, Karthik Vinay Seetharaman, and Vasilis Syrgkanis. Direct preference optimization with unobserved preference heterogeneity: The necessity of ternary preferences. _arXiv preprint arXiv:2405.15065_ , 2025.

- [13] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep Reinforcement Learning from Human Preferences. In _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc., 2017. URL `https://proceedings.neurips.cc/paper_files/paper/2017/hash/ d5e2c0adad503c91f91df240d0cd4e49-Abstract.html` .

- [14] Marie Jean Antoine Nicolas Caritat de Condorcet. _Essai sur l’application de l’analyse à la probabilité des décisions rendues à la pluralité des voix_ . Imprimerie Royale, Paris, 1785.

- [15] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL `https://arxiv.org/abs/2501.12948` .

- [16] Ricardo Dominguez-Olmedo, Moritz Hardt, and Celestine Mendler-Dünner. Questioning the survey responses of large language models. _Advances in Neural Information Processing Systems_ , 2024.

- [17] Simon Duane, A. D. Kennedy, Brian J. Pendleton, and Duncan Roweth. Hybrid Monte Carlo. _Physics Letters B_ , 195(2):216–222, 1987.

- [18] Ali Goli and Amandeep Singh. Can LLMs capture human preferences? _Marketing Science_ , 2024.

- [19] Google DeepMind. Gemma 4 Model Card. `https://ai.google.dev/gemma/docs/core/ model_card_4` , 2026. Accessed: 2026-05-07.

- [20] Günter Hägele and Friedrich Pukelsheim. Llull’s writings on electoral systems. _Studia Lulliana_ , 41:3–38, 2001.

- [21] Matthew D. Hoffman and Andrew Gelman. The no-U-turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. _Journal of Machine Learning Research_ , 15(47):1593–1623, 2014.

- [22] International Atomic Energy Agency, Nuclear Data Section. LiveChart of nuclides. `https: //www-nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=all` , 2026. Accessed via the IAEA LiveChart data API.

- [23] Hemant Ishwaran and Lancelot F. James. Gibbs sampling methods for stick-breaking priors. _Journal of the American Statistical Association_ , 96(453):161–173, 2001.

- [24] Maximilian Jerdee and M. E. J. Newman. Luck, skill, and depth of competition in games and social hierarchies. _Science Advances_ , 10(45):eadn2654, 2024.

- [25] Harold W. Kuhn. The Hungarian method for the assignment problem. _Naval Research Logistics Quarterly_ , 2(1–2):83–97, 1955.

- [26] Kenneth O. May. Intransitivity, utility, and the aggregation of preference patterns. _Econometrica_ , 22(1):1–13, 1954.

- [27] Mantas Mazeika, Xuwang Yin, Rishub Tamirisa, Jaehyuk Lim, Bruce W. Lee, Richard Ren, Long Phan, Norman Mu, Adam Khoja, Oliver Zhang, and Dan Hendrycks. Utility engineering: Analyzing and controlling emergent value systems in AIs. _arXiv preprint arXiv:2502.08640_ , 2025.

- [28] Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, and Gabriel Stanovsky. State of What Art? A Call for Multi-Prompt LLM Evaluation. _Transactions of the Association for Computational Linguistics_ , 12:933–949, 2024. doi: 10.1162/tacl_a_00681. URL `https: //aclanthology.org/2024.tacl-1.52/` .

- [29] Jared Moore, Tanvi Deshpande, and Diyi Yang. Are large language models consistent over value-laden questions? _arXiv preprint arXiv:2407.02996_ , 2024.

- [30] Radford M. Neal. MCMC using Hamiltonian dynamics. In Steve Brooks, Andrew Gelman, Galin L. Jones, and Xiao-Li Meng, editors, _Handbook of Markov Chain Monte Carlo_ , pages 113–162. Chapman and Hall/CRC, 2011.

- [31] M. E. J. Newman. Efficient computation of rankings from pairwise comparisons. _Journal of Machine Learning Research_ , 24(238):1–25, 2023.

- [32] Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3, 2025. URL `https: //arxiv.org/abs/2512.13961` .

- [33] OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL `https://arxiv.org/abs/ 2508.10925` .

- [34] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback.

- [35] Michael Pearce and Elena A. Erosheva. Modeling preferences: A Bayesian mixture of finite mixtures for rankings and ratings. _Journal of the American Statistical Association_ , 120(551): 1621–1632, 2025.

- [36] Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. In _Findings of the Association for Computational Linguistics: NAACL 2024_ , pages 2006–2017, 2024.

- [37] Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. In _Findings of the Association for Computational Linguistics: NAACL 2024_ , pages 2006–2017, 2024.

- [38] Sriyash Poddar, Yanming Wan, Hamish Ivison, Abhishek Gupta, and Natasha Jaques. Personalizing reinforcement learning from human feedback with variational preference learning. _arXiv preprint arXiv:2408.10075_ , 2024.

- [39] Mengjie Qian, Guangzhi Sun, Mark J. F. Gales, and Kate M. Knill. Who can we trust? LLM-as-a-jury for comparative assessment. _arXiv preprint arXiv:2602.16610_ , 2026.

- [40] Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, April 2026. URL `https://qwen.ai/blog?id=qwen3.6-35b-a3b` .

- [41] Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. Quantifying Language Models’ Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting, July 2024. URL `http://arxiv.org/abs/2310.11324` . arXiv:2310.11324 [cs].

- [42] Jayaram Sethuraman. A constructive definition of Dirichlet priors. _Statistica Sinica_ , 4:639–650, 1994.

- [43] Jingyan Shen, Jiarui Yao, Rui Yang, Yifan Sun, Feng Luo, Rui Pan, Tong Zhang, and Han Zhao. MiCRo: Mixture modeling and context-aware routing for personalized preference learning. _arXiv preprint arXiv:2505.24846_ , 2025.

- [44] Ali Shirali, Arash Nasr-Esfahany, Abdullah Alomar, Parsa Mirtaheri, Rediet Abebe, and Ariel Procaccia. Direct Alignment with Heterogeneous Preferences.

- [45] Edwin Simpson, Erik-Lân Do Dinh, Tristan Miller, and Iryna Gurevych. Predicting humorousness and metaphor novelty with Gaussian process preference learning. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 5716–5728, Florence, Italy, 2019. Association for Computational Linguistics.

- [46] Rahul Singh and Ori Davidov. Modeling cyclicality and intransitivity in paired comparisons data, July 2026. URL `http://arxiv.org/abs/2406.11584` . arXiv:2406.11584v3 [stat].

- [47] Anand Siththaranjan, Cassidy Laidlaw, and Dylan Hadfield-Menell. Distributional preference learning: Understanding and accounting for hidden context in RLHF. _International Conference on Learning Representations_ , 2024.

- [48] Xiaoyang Song, Yuta Adachi, Jessie Feng, Mouwei Lin, Linhao Yu, Frank Li, Akshat Gupta, Gopala Anumanchipalli, and Simerjot Kaur. Identifying multiple personalities in large language models with external evaluation. _arXiv preprint arXiv:2402.14805_ , 2024.

- [49] Kazuhiro Takemoto. Scaling Laws for Moral Machine Judgment in Large Language Models, 2026. URL `https://arxiv.org/abs/2601.17637` . Version Number: 3.

- [50] Gemma Team. Gemma 3. 2025. URL `https://goo.gle/Gemma3Report` .

- [51] Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL `https:// qwenlm.github.io/blog/qwen2.5/` .

- [52] L. L. Thurstone. A law of comparative judgment. _Psychological Review_ , 34(4):273–286, 1927.

- [53] Kenneth E Train. _Discrete choice methods with simulation_ . Cambridge university press, 2009.

- [54] Rung-Ching Tsai and Ulf Böckenholt. Modelling intransitive preferences: A random-effects approach. _Journal of Mathematical Psychology_ , 50(1):1–14, 2006.

- [55] Rung-Ching Tsai and Ulf Böckenholt. On the importance of distinguishing between withinand between-subject effects in intransitive intertemporal choice. _Journal of Mathematical Psychology_ , 52(4):242–256, 2008.

- [56] Amos Tversky. Intransitivity of preferences. _Psychological Review_ , 76(1):31–48, 1969.

- [57] Satoshi Usami. Individual differences multidimensional Bradley-Terry model using reversible jump Markov chain Monte Carlo algorithm. _Behaviormetrika_ , 37(2):135–155, 2010.

- [58] Tom van Nuenen and Pratik S. Sachdeva. The fragility of moral judgment in large language models. _arXiv preprint arXiv:2603.05651_ , 2026.

- [59] Angelina Wang, Jamie Morgenstern, and John P. Dickerson. Large language models that replace human participants can harmfully misportray and flatten identity groups. _arXiv preprint arXiv:2402.01908_ , 2025.

- [60] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2024.

- [61] Yidong Wang, Yunze Song, Tingyuan Zhu, Xuanwang Zhang, Zhuohao Yu, Hao Chen, Chiyu Song, Qiufeng Wang, Cunxiang Wang, Zhen Wu, Xinyu Dai, Yue Zhang, Wei Ye, and Shikun Zhang. TrustJudge: Inconsistencies of LLM-as-a-judge and how to alleviate them. _arXiv preprint arXiv:2509.21117_ , 2025.

- [62] Christopher Wolfram and Aaron Schein. World models and consistent mistakes in LLMs. In _ICML 2025 Workshop on Assessing World Models_ , 2025. URL `https://openreview.net/ forum?id=XunYYDvv39` .

- [63] Mingyuan Xu, Xinzi Tan, Jiawei Wu, and Doudou Zhou. A judge-aware ranking framework for evaluating large language models without ground truth. _arXiv preprint arXiv:2601.21817_ , 2026.

- [64] Yi Xu, Laura Ruis, Tim Rocktäschel, and Robert Kirk. Investigating non-transitivity in LLMas-a-judge. _arXiv preprint arXiv:2502.14074_ , 2025.

- [65] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ , 2024.

- [66] Yan Yu, Yilun Liu, Minggui He, Shimin Tao, Weibin Meng, Xinhua Yang, Li Zhang, Hongxia Ma, Dengye Li, Daimeng Wei, Boxing Chen, and Fuliang Li. ELSPR: Evaluator LLM training data self-purification on non-transitive preferences via tournament graph reconstruction. _arXiv preprint arXiv:2505.17691_ , 2025.

- [67] Yifan Zeng, Ojas Tendolkar, Raymond Baartmans, Qingyun Wu, Lizhong Chen, and Huazheng Wang. LLM-RankFusion: Mitigating intrinsic inconsistency in LLM-based ranking. _arXiv preprint arXiv:2406.00231_ , 2024.

- [68] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large Language Models Are Not Robust Multiple Choice Selectors, September 2023. URL `https://arxiv.org/abs/ 2309.03882v4` .

- [69] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. _arXiv preprint arXiv:2309.03882_ , 2023.

- [70] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. _Advances in Neural Information Processing Systems_ , 2023.

- [71] Han Zhou, Xingchen Wan, Yinhong Liu, Nigel Collier, Ivan Vuli´c, and Anna Korhonen. Fairer Preferences Elicit Improved Human-Aligned Large Language Model Judgments, October 2024. URL `http://arxiv.org/abs/2406.11370` . arXiv:2406.11370 [cs].

## A Dataset preparation

This appendix describes how we arrive at each processed set of items discussed in Section 3. Across these datasets, we preserve exactly 50 items so that the four problems present LLMs with the same number of alternatives and unordered pairs. This maximum is set by the MORAL MACHINE dataset, where human data are limited to 60 items. For datasets in which the preference structure is derived from human pairwise comparisons, we restrict our initial subset to the giant strongly connected component of the human preference graph. This guarantees that Bradley–Terry rankings are identifiable even in the absence of strong regularization [31]. The SCC condition is automatically met for datasets in which comparisons follow from an external, factual score (POPULATION, HALFLIVES).

**JOKES.** The initial dataset comprises 4 _,_ 030 jokes with crowd-sourced ratings. The ratings are elicited with pairwise comparisons, arranged so that each joke appears in exactly 14 unique pairs and each pair is rated by 5 annotators [45]. Ratings can be one of `A` , `B` , or `X` (“neither funnier”), and we keep only the definite ratings when constructing the preference matrix. This removes a fraction of the comparisons but does not remove any jokes from the dataset, since each joke happens to retain at least one decisive rating. The resulting directed preference graph is a single strongly connected component; we select a random subset of 50 jokes, and use the same subset throughout our analysis.

**MORAL MACHINE.** The Moral Machine dataset [4] captures the decisions humans think a selfdriving car ought to make in a wide variety of scenarios involving an imminent collision. We restrict our analysis to one set of scenarios, in which the victims’ attributes are randomized, the number of characters is equal, and both alternatives are set to one. This selection isolates the categorical attributes of pedestrians (age, profession, gender, social status, legality of crossing) from quantitybased assessments. A crossing signal is included in some scenarios and determines whether cars and pedestrians are obeying the law. Each unique `(pedestrian category, crossing signal)` combination defines an item, yielding 60 distinct items. Aggregating human votes into a directed preference graph and computing strongly connected components leaves two strongly connected components of 54 and 6 items, respectively; we work with the larger one and take a fixed random subset of 50 as our experimental set.

**POPULATION.** We use the U.S. Census Bureau estimates (table `SUB-IP-EST2023-POP` ), which lists 19 _,_ 484 U.S. cities with an April 2020 baseline population and July 1^st estimates for each year from 2020 to 2023. We score each city by its 2023 estimate, the most recent value and the one most likely to be reflected in LLM training data. Naively taking the top 50 cities yields several adjacent pairs whose populations differ by only a few hundred residents; these differences are within Census estimation uncertainty, and we do not want to test LLMs on them. Hence, we apply a minimum gap filter and sort cities by population in descending order, greedily skipping any city whose population is within 3 _,_ 000 of the previously selected city, and we stop after 50 selections. Deterministic pairwise comparisons are then written to a dense matrix comparing these cities based on the 2023 estimate.

**HALF-LIVES.** All known nuclides are downloaded from the IAEA LiveChart of Nuclides API [22] (using the `ground_states` endpoint). We discard entries with non-numeric or missing half-lives, including isotopes flagged as stable, and keep only ground-state nuclides with parsable atomic number, mass number, and half-life in seconds. This yields the candidate pool of all known radioactive isotopes with measured half-lives. From this pool we draw a random sample of 100 isotopes, then sort them by descending half-life to define a deterministic item order. Our final item set comprises the first 50, i.e., the 50 longest-lived isotopes in the random sample. Deterministic pairwise comparisons are then written to a dense matrix comparing the isotopes based on their half-lives.

## B Language models

This appendix documents the language models used in our pairwise preference elicitation experiments. All models are open-weight models accessed through public model repositories. We use open-weight models to make the elicitation protocol reproducible and to avoid dependence on hosted APIs whose model weights, system behavior, or decoding defaults may change over time.

Table 3: Language-model provenance for pairwise preference elicitation. Model identifiers correspond to the exact checkpoints used in our experiments. Citations point to the model technical report or model card used to document provenance.

|Display name|Model identifier|License / terms|
|---|---|---|
|DeepSeek-R1 14B|`deepseek-ai/DeepSeek-R1 14B-R1-Distill-Qwen-14B`[15] |MIT|
|GPT-OSS 20B|`openai/gpt-oss-20b`[33]|Apache 2.0|
|Gemma 3 12B|`google/gemma-3-12b-it`[50]|Gemma terms|
|Gemma 4 31B|`google/gemma-4-31b-it`[19]|Gemma terms|
|OLMo 3 7BThink|`allenai/Olmo-3-7B-Think`[32]|Apache 2.0|
|Qwen 2.5 14B|`Qwen/Qwen2.5-14B-Instruct`[51, 65]|Apache 2.0|
|Qwen 3.6 35B|`Qwen/Qwen3.6-35B-A3B`[40]|Apache 2.0|

### B.1 Model selection

We selected models to cover several model families, sizes, and post-training styles while keeping inference feasible for dense pairwise elicitation. The final set contains seven instruction-tuned or reasoning-oriented models: DeepSeek-R1 14B, GPT-OSS 20B, Gemma 4 31B, Gemma 3 12B, OLMo 3 7B, Qwen 3.6 35B, and Qwen 2.5 14B. For each model, we use the released checkpoint without additional fine-tuning.

### B.2 Model provenance

Table 3 lists the exact model identifiers used in our experiments. When a peer-reviewed paper or technical report is available, we cite it as the primary source. When the exact checkpoint is not covered by a formal paper, we cite the public model card or release page.

### B.3 Inference setup

For every model–dataset pair, we use VLLM to query the model on all unordered item pairs under the prompt and order controls described in Appendix C. We use the released instruction or reasoning checkpoint as-is and do not perform any additional training, calibration, or preference tuning. Each generation is parsed only for the final forced-choice identifier. For reasoning models, any reasoning trace is not used as data except insofar as it precedes the final parsed choice.

Unless otherwise stated, decoding follows the model configuration used for the elicitation run. The same elicitation protocol is applied across models: identical item sets, identical prompt variants, randomized pseudo-identifiers, both presentation orders, and the same parsing rules. Unparseable outputs are discarded before constructing the pairwise win-count matrix.

## C Preference elicitation protocol

This appendix describes the full elicitation and parsing procedure used to construct the pairwise preference matrix from LLM responses. At a high level, for every unordered pair of items in a dataset, we generate dense comparisons under randomized presentation orders, with prompt variation and randomized pseudo-identifiers.

**System prompt variants** are designed to preserve the same task semantics while changing surface wording. For example, in the jokes domain, the model is always asked to choose the funnier joke, while in the Moral Machine domain it is always asked to choose the person who should be sacrificed. We generate four variations for each dataset by including or excluding a roleplay prompt and varying the wording of the comparison. For instance, when experimenting on JOKES, the system prompt begins with one of the following options:

- ( `roleplay, comparison semantic 1` ). You are a professional comedian with refined taste in humor. You will be given two jokes labeled `pseudo_id_A` and `pseudo_id_B` . Select the funnier one.

- ( `roleplay, comparison semantic 2` ). You are a comedian with a lot of professional experience evaluating material for a stand-up show. You will be given two jokes labeled `pseudo_id_A` and `pseudo_id_B` . Determine which joke is funnier.

- ( `no roleplay, comparison semantic 1` ). You will be given two jokes labeled `pseudo_id_A` and `pseudo_id_B` . Select the funnier one.

- ( `no roleplay, comparison semantic 2` ). You will be given two jokes labeled `pseudo_id_A` and `pseudo_id_B` . Determine which joke is funnier.

It is followed by boilerplate prompting that encourages structured output.

**Ordering and labeling bias** is mitigated by presenting items in both orderings an equal number of times, and replacing the pseudo-identifiers with randomly sampled label letters at inference time. We use a simple user message formatted as

`<pseudo_id_A>: <item A> <pseudo_id_B>: <item B>` _._

**Sampling** . Comparisons are sampled _M_ times for each comparison at the default temperature specified in the model configuration. For thinking models, the answer is parsed after the reasoning trace; for non-thinking models, the answer is parsed directly from the generation. An output is included only if the selected value matches one of the two pseudo-identifiers assigned to the current comparison. Otherwise, it is discarded as unparsable. This produces a successful match count that can vary slightly across item pairs if some generations fail to parse.

**Algorithm 1** Pairwise preference elicitation

**Require:** Comparison items [ _N_ ] = _{_ 1 _, . . . , N }_ , prompt variants _V_ , samples per comparison _M_ 1: Initialize win-count matrix _A ←_ 0 _N ×N_

2: **for** _i_ = 1 to _N_ **do** 3: **for** _j_ = _i_ + 1 to _N_ **do** 4: Sample two distinct pseudo-identifiers _u, v_ without replacement from _{A, B, ..., Z}_ 5: **for** each prompt variant _s ∈V_ **do** 6: **for** each order _o ∈{_ ( _i, j_ ) _,_ ( _j, i_ ) _}_ **do** 7: Construct a forced-choice prompt with items in order _o_ 8: Query the model _M_ times 9: **for** each generated completion **do** 10: Extract the value of the selected item, `selected` 11: **if** `selected` matches the pseudo-identifier assigned to item _i_ **then** 12: _aij ← aij_ + 1 13: **else if** `selected` matches the pseudo-identifier assigned to item _j_ **then** 14: _aji ← aji_ + 1 15: **else** 16: Discard the generation as unparsable 17: **end if** 18: **end for** 19: **end for** 20: **end for** 21: **end for** 22: **end for** 23: **return** win-count matrix _A_

## D Transitivity Violation Null Model

### D.1 Notation

For each pair of items ( _i, j_ ), let _aij_ denote the number of times _i_ was selected over _j_ in the LLM’s pairwise comparisons, and let _p_ ˆ _ij_ = _aij/_ ( _aij_ + _aji_ ) be the empirical win probability. The _Borda count_ of item _i_ is the average win probability against all other items in the dataset, where _N_ is the number of items.

### D.2 Borda-link null

A random utility model assumes _pij_ = _f_ ( _θi − θj_ ) for some non-decreasing link _f_ and latent scores _{θi}_^_N_ _i_ =1^(.Our null tests whether the observed comparisons could have come from any such model,) without committing to a particular _f_ or inferring the _θi_ .

**Borda count as a monotone proxy.** Treating the latent scores _θj_ for _j̸_ = _i_ as draws from a distribution _p_ with mean _θ_^(¯) and variance _σ_^2 , the conditional expected Borda count is the convolution of the link with the score distribution,

Writing _V_ = _θi − θ_ , so E[ _V_ ] = _θi − θ_^(¯) and Var( _V_ ) = _σ_^2 , and Taylor-expanding _f_ around the centered score _θi − θ_^(¯) ,

For score distributions concentrated around their mean ( _σ_ small relative to the curvature scale of _f_ ), the leading term dominates and the map _θi →_ E[ _bi | θi_ ] is monotone in _θi_ . Any monotone function of the score gap _θi − θj_ can therefore be expressed as a monotone function of the Borda gap _bi − bj_ , which is the only structure the isotonic null below requires

**Isotonic null.** Motivated by this approximation, we take the null win probability to be a non-decreasing function of the Borda gap, and fit _g_ by isotonic regression on the observed pairs ( _bi − bj, p_ ˆ _ij_ ), anchored at _g_ (0) = 1 _/_ 2 to enforce symmetry. This yields the closest random-utility-consistent fit to the observed comparisons without committing to a parametric link.

### D.3 Bootstrap test

Under the null, the wins follow _a_^null _ij ∼_ Binomial( _Mij, p_^null _ij_^(), where)^(_Mij_=)^(_aij_+)^(_aji_is the total) number of comparisons between _i_ and _j_ . We generate _B_ = 1000 bootstrap samples _{a_^(() _ij_^(_b_))^(_}_) _b_^_B_ =1^, recompute the SST deficit _D_ (^SST _b_ )^(on each, and report the upper-tail probability)

A small upper-tail probability indicates that the observed deficit is larger than what any random utility model would produce.

The aggregate diagnostics in Section 4 could be inflated by protocol artifacts. For example, a model may prefer the first option, respond differently to different system-prompt phrasings, or interact with pseudo-labels in a non-item-specific way. We therefore repeat the transitivity diagnostics under several restricted conditions.

## E Transitivity Controls

**Full aggregate matrix.** The primary analysis pools all successful completions for each unordered pair _{i, j}_ . This produces the aggregate matrix _A_ used in the main text. It has the largest number of observations per pair, but it also pools across prompt variants and presentation orders.

**Fixed prompt and fixed order.** To test whether triplet residuals are caused by pooling across prompt conditions, we recompute the diagnostics within each fixed prompt condition. We further split by presentation order. For an unordered pair _{i, j}_ with _i < j_ , one condition contains prompts where the lower-index item appears in slot A, and the other contains prompts where it appears in slot B. This gives condition-specific matrices of the form _A_^(()^(_c_)) _,_ where _c_ indexes a prompt-order condition. For each condition, we recompute the Borda scores, refit the monotone Borda-link null, and regenerate a bootstrap distribution using only the counts available in that condition. Thus each condition has its own global ordering, comparison counts, and null calibration.

**Order-symmetrized matrix.** We also construct an order-symmetrized matrix that removes first-order presentation bias. For each unordered pair _{i, j}_ with _i < j_ , define and

When both orientations are available, we estimate the order-symmetrized preference probability as

If only one orientation is available, we use the available estimate. We then construct a symmetrized count matrix with the same total number of successful comparisons for the pair:

This matrix is used only for diagnostics, not as raw observed count data.

**Pseudo-label controls.** Pseudo-identifiers are randomly assigned to options and are independent of item identity. In the main experiments, they are used only to parse the selected option. If pseudo-label effects are a concern, the same diagnostic can be repeated within fixed pseudo-label conditions. In the current analysis, we focus on prompt and presentation-order controls because they are the most direct protocol-level explanations for transitivity artifacts.

**Control statistic.** For each control condition _c_ , we compute where the null distribution is fitted and bootstrapped separately within condition _c_ . We report the same WST, SST, and MST excesses as in the main analysis.

**Interpretation.** If excess triplet residuals disappear under fixed prompt or order-symmetrized conditions, then the aggregate result is likely driven by a protocol artifact. If the residuals persist within fixed conditions, then they are less likely to be explained by prompt wording or presentation order alone. This does not identify the source of the structure, but it strengthens the case that the residuals are not merely random sampling noise around one global ordering.

Table 4: Triplet-level transitivity diagnostics under fixed prompt and order controls. For each model– dataset pair, diagnostics are recomputed within each fixed system-prompt and presentation-order stratum (2 prompt styles _×_ 2 semantic variants _×_ 2 presentation orders). Entries report the median excess across the eight strata, with the minimum stratum-level excess in parentheses. Excess is defined as the observed statistic minus the bootstrap mean under a matched Borda-link null fitted within the same stratum. SST uses mean positive triplet deficit; MST uses mean absolute multiplicative cycle residual. Positive minima indicate that excess intransitivity appears in every fixed prompt–order stratum. All fixed-stratum tests are significant at _p ≤ ._ 05; most attain the bootstrap floor _p_ = _._ 001.

|Model|Dataset|∆SST|∆MST|
|---|---|---|---|
||JOKES|0.0115 (0.0110)|0.0165 (0.0156)|
|GPT-OSS 20B|MORALMACHINE|0.0259 (0.0201)|0.0291 (0.0229)|
||POPULATION|0.0167 (0.0129)|0.0075 (0.0060)|
||HALF-LIVES|0.0192 (0.0177)|0.0241 (0.0226)|
||JOKES|0.0240 (0.0187)|0.0321 (0.0268)|
|DSkR1 14B|MORALMACHINE|0.0776 (0.0719)|0.0726 (0.0578)|
|eepee-|POPULATION|0.0168 (0.0133)|0.0143 (0.0102)|
||HALF-LIVES|0.0214 (0.0200)|0.0299 (0.0284)|
||JOKES|0.0113 (0.0095)|0.0115 (0.0101)|
|G 4 31B|MORALMACHINE|0.0329 (0.0235)|0.0313 (0.0221)|
|emma|POPULATION|0.0112 (0.0035)|0.0116 (0.0036)|
||HALF-LIVES|0.0308 (0.0271)|0.0308 (0.0258)|
||JOKES|0.0275 (0.0243)|0.0282 (0.0236)|
|G 3 12B|MORALMACHINE|0.0864 (0.0792)|0.0813 (0.0761)|
|emma|POPULATION|0.0241 (0.0147)|0.0243 (0.0146)|
||HALF-LIVES| 0.0414 (0.0273)| 0.0350 (0.0203)|
||JOKES|0.0577 (0.0470)|0.0629 (0.0534)|
||MORALMACHINE|0.0924 (0.0739)|0.0931 (0.0729)|
|OLMo 3 7B|POPULATION|0.0633 (0.0368)|0.0647 (0.0349)|
||HALF-LIVES|0.0771 (0.0711)|0.0777 (0.0711)|
||JOKES|0.0379 (0.0322)|0.0361 (0.0308)|
||MORALMACHINE| 0.0393 (0.0131)| 0.0244 (0.0077)|
|Qwen 2.5 14B|POPULATION| 0.0303 (0.0177)| 0.0300 (0.0170)|
||HALF-LIVES| 0.0551 (0.0510)| 0.0524 (0.0461)|
||JOKES|0.0172 (0.0156)|0.0225 (0.0210)|
|Qwen 3.6 35B|MORALMACHINE|0.0859 (0.0792)|0.0807 (0.0774)|
||POPULATION|0.0002 (0.0002)|0.0002 (0.0001)|

## F Noise-augmented mixture Bradley–Terry model

The data are _m_ comparisons among items [ _N_ ] = _{_ 1 _, . . . , N }_ . For trial _t_ , let ( _it, jt_ ) be the ordered pair shown to the model and let _yt_ = 1 if the model selects _it_ and _yt_ = 0 if it selects _jt_ . Thus

### F.1 Noise augmentation

We model LLMs as approaching difficult choices as a toss-up, using a two-step stochastic process [31]. We draw a binary random variable _Ut ∼_ Bernoulli( _α_ ); if _Ut_ = 1, the model chooses between the two items with a fair coin flip and, if _Ut_ = 0, it follows Bradley–Terry probabilities. Thus, for comparison _t_ , where _θi ∈_ R is the latent score of item _i_ and _λi_ = _e_^_θi_ is its strength. Marginalizing over _Ut_ gives the win probability

### F.2 MBT model

We model the multiplicity of ways a model can rank items as a mixture, adding a step to the stochastic process. This requires _K_ max vectors of latent scores _{_ **_θ_** _k}_^_K_ _k_ =1^max and associated luck parameters _{αk}_^_K_ _k_ =1^(max. The mixture component selected for comparison)^_t_isthenmodeledasacategorical variable where and _λik_ = _e_^_θik_ is the strength of item _i_ in component _k_ .

### F.3 Complete-data and marginal likelihood

The joint likelihood of the comparisons _D_ and assignments **_Z_** , written using _ztk_ = ⊮ _{Zt_ = _k}_ , is

Marginalizing over the component assignments **_Z_** yields which can be re-expressed in terms of the model’s sufficient statistics, namely the number of wins by _i_ over _j_ for all pairs. Let denote wins by _i_ over _j_ . Grouping trials by the winning and losing items gives the equivalent count likelihood where the product is over all ordered pairs of items.

### F.4 Priors and Generative Model

We use a Bayesian formulation of the model to regularize inference, as BT models and finite mixtures can both face identifiability issues that can be compounded by fusing the two modeling approaches [31, 35].

**Scores.** The BT likelihood depends only on score differences _θik − θjk_ , so the absolute scale of the scores is not identified. We break this symmetry by constraining the scores in each component to sum to zero, and place a zero-mean Gaussian prior on the resulting ( _N −_ 1)-dimensional subspace, calibrated so that score _differences_ have unit variance:

This implies Var( _θik_ ) = 12^((1)^(_−_1)^(_/N_)andVar()^(_θik−θjk_)=1,putting the implied prior on) pairwise win probabilities _σ_ ( _θik − θjk_ ) on a sensible scale. Because the prior is proper, a unique posterior mode exists regardless of the connectivity of the empirical comparison graph, avoiding the strong-connectivity requirement of maximum-likelihood BT estimation [31]. For sampling, we reparameterize in R^(_N−_1) using any orthonormal basis _Q_ of the sum-to-zero subspace.

**Noise-augmentation.** The prior on the per-component noise parameters is modeled with a symmetric beta prior,

**Mixture Components** Rather than select the number of components directly, we place a Dirichlet Process (DP) prior over the mixing measure with concentration parameter _γ_ and truncate it at _K_ max. The base distribution is over the per-component parameters ( _αk,_ **_θ_**^() _k_ ). The DP induces the following truncated stick-breaking representation [42] of the mixture weights; the finite approximation converges exponentially fast to the true DP [23]:

The mixture assignments are drawn from a categorical distribution defined by **_π_** ,

We fix _γ_ throughout. The concentration _γ_ has a direct interpretation through the Chinese Restaurant Process representation of the DP: the expected number of occupied components after _m_ comparisons is approximately _γ_ log(1 + _m/γ_ ), so _γ_ controls the prior expected number of distinct components on a logarithmic scale. We choose _γ_ to place reasonable prior mass on the range of mixture sizes consistent with our experimental setup.

### F.5 Generative Model

The full generative model can be summarized as follows (see Fig. 6). Let _N_ denote the number of items, _m_ the number of pairwise comparisons, and _K_ max the truncation level for the stick-breaking prior. For each comparison _t_ , let ( _it, jt_ ) denote the ordered pair shown to the model and let _Yt_ denote its binary choice.

_Mixture weights_ (stick-breaking, _k_ = 1 _, . . . , K_ max _−_ 1):

_Per-component parameters_ ( _k_ = 1 _, . . . , K_ max):

_Observations_ ( _t_ = 1 _, . . . , m_ ):

Figure 6: Plate diagram for the MBT model. Shaded nodes are observed, dashed nodes are deterministic, filled black dots are fixed hyperparameters. The component plate ( _k_ = 1 _, . . . , K_ max) contains the per-component luck parameter _αk_ , reduced scores **_θ_**^() _k_ , and item strengths _λik_ (nested over items, _i_ = 1 _, . . . , N_ ). The mixture weights _π_ are drawn from a stick-breaking process with concentration _γ_ and used to assign each comparison _t_ = 1 _, . . . , m_ to a component _Zt_ , which together with the observed item indices ( _it, jt_ ) generates the outcome _Yt_ .

## G Inference and Sampling

We use Hamiltonian Monte Carlo (HMC) to generate samples from the posterior [17, 30]. Specifically, we use the No-U-Turn Sampler (NUTS) as implemented in the BlackJAX Python package [21, 7]. Label switching creates a multimodal posterior for mixture models. Even though the DP prior technically breaks this symmetry, we observe strong empirical identifiability issues with label switching, and thus use the Hungarian algorithm [25] for post-hoc alignment of chains across sweeps.

In our estimation procedure, we use a diagonal mass matrix and a target acceptance rate of 0.8. Each dataset is fit with 4 chains of 6,000 warmup and 4,000 post-warmup samples (16,000 total draws). To reduce label switching during warmup, we run adaptation on a single chain and broadcast the resulting step size and mass matrix to all remaining chains before sampling as recommended in the Stan User Guide. Convergence is assessed with the Gelman-Rubin statistic _R_^ˆ and the effective sample size of both the mixture weights **_π_** and noise parameters **_α_** . A fit is deemed acceptable if all components satisfy _R <_^ˆ 1 _._ 01, ESS _≥_ 100, and zero divergent transitions are observed.

## H Consistency of the estimation model

As mentioned before, mixtures and Bradley–Terry models can both face identifiability issues, and our proposal is to combine both. To verify that the parameters of our implementation can be recovered in practice, we conduct a consistency study in which we start from known ground-truth parameters, generate data with the model, apply the inference algorithm, and validate agreement. The results below show that the estimation algorithm recovers ground-truth parameters across most of the sweep grid, with two regimes where it fails: 1) when the scores of different components are highly correlated and the number of components is large, and 2) when noise augmentation dominates, regardless of the number of ground-truth components.

### H.1 Data generation grid

We generate preference matrices from the generative model specified in Section F.5, with two modifications to explore potential failure modes.

First, we control the correlation between components by drawing each item’s score vector **_θ_** _i_ = ( _θi_ 1 _, . . . , θiK_ true) from _N_ ( **0** _,_ Σ( _ρ_ )), where Σ( _ρ_ ) = _σ_^2 [(1 _− ρ_ ) _IK_ true + _ρ_ **11**^(_⊤_) ]. In many settings, we expect scores across different components to be correlated; e.g., consequentialist and contrarian systems of ethics may disagree on a number of specific dilemmas but will still choose the same option in a majority of scenarios. Large _ρ_ can model such dependencies and make components harder to tell apart if all other parameters are held constant.

Second, we vary the level of noise by altering the expectation of _α_ . Large values of _α_ decrease the signal-to-noise ratio, making more of the individual pairwise comparisons uninformative.

We couple these modifications with a sweep over the number of ground-truth components, _K_ true, and hyperparameters of the inference process.

### H.2 Evaluation metrics

Let _K_ min = min( _K_ true _, K_ max). We quantify recovery using three metrics. For mixture weights, we compute the KL divergence between the ground-truth weights and the posterior-mean weight vector:

where **_π_** ¯ = _S_ 1  _Ss_ =1^(**_π_**()^(_s_)is the posterior mean, and both vectors are sorted in descending order) before comparison.

For scores, we first match components by descending mixture weights and center the scores in each component. We then report the Pearson correlation between the ground-truth and posterior-mean scores across all _N × K_ min item–component entries.

Finally, for the luck parameters, we compute the mean absolute error between the sorted posteriormean and ground-truth _α_ values:

where _α_ ¯ _k_ = _S_^(1)  _Ss_ =1^_α_ _k_^(()^(_s_)) and both vectors are sorted in descending order. The use of _K_ min handles cases where the truncation level differs from the ground truth.

### H.3 Results

Figure 7 plots the results of a recovery sweep. In the full factorial sweep, the number of mixture components _K_ true takes values _{_ 1 _,_ 2 _,_ 3 _,_ 5 _}_ . The inter-component score correlation _ρ ∈ {_ 0 _._ 0 _,_ 0 _._ 2 _,_ 0 _._ 4 _,_ 0 _._ 6 _,_ 0 _._ 8 _}_ controls how geometrically similar the components are to one another. The mean luck parameter _α_ ¯ _∈{_ 0 _._ 05 _,_ 0 _._ 10 _,_ 0 _._ 20 _,_ 0 _._ 50 _,_ 0 _._ 90 _}_ controls the tie rate, and therefore how discriminating the pairwise comparisons are. We also vary the DP concentration used at inference, _γ ∈{_ 0 _._ 1 _,_ 0 _._ 5 _}_ , which is a fitting parameter rather than a generative one. Each cell of the resulting grid is repeated three times with independent synthetic datasets. The remaining settings are fixed.

Figure 7: **Top row** : Recovery as a function of inter-score correlation _ρ_ , holding the mean luck parameter at _α_ ¯ = 0 _._ 0. Mixture-weight KL divergence (left), score Pearson correlation (center), and luck-parameter MAE (right) are plotted for different numbers of ground-truth components _K_ true. **Bottom row** : Recovery as a function of mean luck _α_ ¯, holding the inter-score correlation at _ρ_ = 0.

Figure 8 plots calibration for two examples at _K_ true = 1 and _K_ true = 5. We compare the true and posterior mean scores, mixture weights, and _α_ values. At _K_ true = 1, we achieve an almost perfect reconstruction; at _K_ true = 5, we reconstruct the order but not necessarily the exact score values.

Score recovery is near-perfect under _K_ true = 1 ( _r_ = 1 _._ 00, RMSE= 0 _._ 01), confirming the model is not over-parameterized on simple data. Under _K_ true = 5 recovery degrades modestly ( _r_ = 0 _._ 95, RMSE= 0 _._ 31), with the bulk of residual error concentrated in the minor components (C4–C5): their scores scatter further from the diagonal and carry wider error bars, reflecting the smaller effective sample size available to identify low-weight components. The dominant component (C1) remains well-recovered even at _K_ true = 5. Mixture weights are recovered accurately across all five components, with posterior means tracking the true _π_ values along the diagonal and uncertainty (point size) increasing for minor components. The noise scale _α_ is consistently underestimated under _K_ true = 5. These results establish that the DP inference is well-calibrated for the dominant structure while accurately flagging elevated uncertainty for minor components.

Figure 8: Posterior calibration of the MBT model. True vs. posterior mean scores (main panels), mixture weights _π_ (bottom left), and noise scale _α_ (bottom right) for _K_ true = 1 (A) and _K_ true = 5 (B) ground truths. Point size encodes posterior standard deviation; colours identify mixture components.

#### I MBT inference on LLM Preference Data

This appendix provides additional qualitative views of the MBT fits discussed in Section 6. These figures are not used as separate evidence for new claims. They document where the fitted components differ from the aggregate ordering and help interpret the main results.

### I.1 population

We test each model’s preferences on a 50-city pairwise-comparison task, ranking U.S. cities by population. In the empirical win-rate matrices in Figure 9, cities are ordered along both axes by their true Census population. A perfectly transitive ranker would produce a smooth gradient away from the diagonal, with steeper gradients indicating more deterministic preferences. Deviations from this pattern indicate pairwise reversals that are not explained by the true population order.

We fit the MBT model with _K_ max = 5 to each model’s comparisons. The fitted models differ sharply in effective component count: DeepSeek-R1 14B( _k_ eff _≈_ 3) and Qwen 2.5 14B( _k_ eff _≈_ 4) split into multiple modes, while Gemma 4 31B( _k_ eff _≈_ 1) is nearly unimodal. Posterior mean Bradley–Terry scores correlate strongly with log population for DeepSeek-R1 14Band Gemma 4 31B, but weakly for Qwen 2.5 14B.

Intransitivity is not uniform across cities. Louisville and Denver, both midsize cities whose ranking is contested across latent components, consistently sit near the top of the SST violation distribution. Here SST requires that if _pij ≥_ 0 _._ 5 and _pjk ≥_ 0 _._ 5, then _pik ≥_ max( _pij, pjk_ ). For DeepSeek-R1 14B, Denver’s component scores span roughly 25 units: one component ranks it near the top, while another ranks it near the bottom. Louisville shows the same pattern under Qwen 2.5 14B. Aggregate SST violation rates reflect this heterogeneity: 42% for DeepSeek-R1 14Band 40% for Qwen 2.5 14B, compared with 12% for Gemma 4 31B.

Figure 9: MBT fits on the POPULATIONtask for DeepSeek-R1 14B, Qwen 2.5 14B, and Gemma 4 31B. **Top:** Empirical win-rate matrices _pij_ , with cities sorted by true Census population from largest to smallest. Den ver and Louisville are highligh ted as examples of cities whose agg regate ranking is unstable across components. **Middle:** Posterior mean Bradley–Terry scores plotted against log Census population. Points are colored by mixture component, with Denver and Louisville highlighted across components. **Bottom:** Per-city SST violation rates, sorted from largest to smallest. The highest-violation cities correspond to items whose inferred component scores disagree most strongly.

### I.2 moral machine

Section 8 argues that aggregate model similarity can hide heterogeneous component-level structure. The MORAL MACHINE task is the clearest case because all models compare the same moral dilemmas, but their mixture components do not align in a one-to-one way. Figures 10–12 provide the supporting views used for that discussion.

Figure 10 compares empirical pairwise win rates with posterior predictive probabilities for Qwen 3.6 35Bon MORAL MACHINE. The MBT model tracks the empirical probabilities more closely than the single-component noise-augmented Bradley–Terry baseline, consistent with the held-out predictive gains reported in the main text.

Figure 10: Posterior predictive calibration for Qwen 3.6 35Bon MORAL MACHINE. Each point is an item pair. The _x_ -axis shows the model-predicted probability that item _i_ is preferred to item _j_ , and the _y_ -axis shows the empirical win rate. The MBT model lies closer to the diagonal than the single-component noise-augmented Bradley–Terry baseline, indicating better calibration of pairwise probabilities.

The main aggregation analysis compares model-level rankings with component-level rankings. Figure 11 shows the aggregate Spearman correlations between MORAL MACHINEpreference rankings. Figure 12 shows the corresponding component–component correlations. Read together, the heatmaps show why aggregate agreement is not enough: two models can have similar aggregate rankings while containing component pairs that disagree, and models with weak aggregate agreement can still share aligned latent components.

Figure 11: Aggregate preference similarity on MORAL MACHINE. Each cell reports the Spearman correlation between two models’ aggregate preference rankings. The heatmap gives the model-level summary that is usually reported by single-ranking analyses.

Figure 12: Component-level preference similarity on MORAL MACHINE. Each cell reports the Spearman correlation between a pair of inferred mixture components, including human components where available. Blocks of positive and negative correlation reveal structure that is compressed away in the aggregate heatmap.

## J Compute resources

We report the compute used for the experiments in the paper. Estimates combine recorded LLM elicitation metadata, recorded wall-clock times for the transitivity-control jobs, and the SLURM resource tiers used in the posterior-fitting and validation sweeps. Wall-clock logs for all HMC posterior fits are not available exactly, so CPU-hour totals for those experiments are reported as conservative allocated CPU-hour upper bounds rather than measured consumption.

**Hardware environment.** LLM elicitation was run on an internal GPU cluster using NVIDIA Hopper-class GPUs, with 1–2 GPUs per run, 8 CPU cores, and approximately 80–120 GB RAM per job. Statistical diagnostics, posterior inference, cross-validation, and synthetic recovery experiments were run on a CPU partition. CPU jobs used 1–4 CPU cores depending on the experiment and approximately 8–32 GB RAM. Figure extraction and plotting compute was negligible.

**LLM elicitation.** Table 6 reports the final LLM elicitation runs used in the paper. These runs cover seven models and four datasets. The total compute for the final reported elicitation runs was approximately 251 GPU-hours. Including reruns, preliminary runs, failed or discarded runs, and earlier debugging runs, the recorded elicitation total was approximately 1,155 GPU-hours.

**Posterior inference.** The main MBT posterior fits used four chains with NUTS sampling. Each final MBT fit was allocated 4 CPU cores and 16–32 GB RAM for 8–24 hours. Across 25 final empirical fits, this corresponds to 800–2,400 allocated CPU-hours. These are upper bounds because jobs often finished before the allocation limit.

Table 5: Compute used by result family. GPU-hour totals for LLM elicitation are measured from run metadata. CPU-hour totals for MCMC-based posterior fitting, cross-validation, and synthetic recovery are conservative allocated upper bounds because complete wall-clock logs were not available for all runs.

|Result family|Runs|Resources per run|Total estimate|
|---|---|---|---|
|LLM elicitation|28 model–task runs|1–2 Hopper GPUs; 8 CPU; 80–120 GB RAM|_≈_251GPU-h|
|Transitivity diagnostics|243 jobs|1 CPU core; modest RAM|_≈_1_._1CPU-h|
|Main MBT posterior fits|25 fits|4 CPU; 16–32 GB RAM; 8–24 h allo- cation|800–2,400 CPU-h|
|5-fold CV, MBT|140 fits|4 CPU; 16–32 GB RAM; 8–24 h allo- cation|4,480–13,440 CPU-h|
|5-fold CV, _K_max = 1 baseline|140 fits|2 CPU; 8–16 GB RAM; 4–12 h alloca- tion|1,120–3,360 CPU-h|
|Synthetic recovery|720 fits|4 CPU; 16–32 GB RAM; 6–18 h allo- cation|17,280–51,840 CPU-h|
|POPULATIONtop-50 ap- pendix|6 fits|4 CPU; 16–32 GB RAM; 8–24 h allo- cation|192–576 CPU-h|
|Figure extraction and plotting|cache extraction and plotting|Cluster CPU and local CPU|small|

Table 6: Final LLM elicitation compute by model. Wall time is aggregated across the four reported dataset runs for each model. GPU-hours are computed as wall time multiplied by the number of GPUs allocated per run.

|Model|GPUs/run|Wall time (h)|GPU-hours|
|---|---|---|---|
|Qwen 3.6 35B|2|88.9|177.8|
|DeepSeek-R1 14B|1|44.4|44.4|
|GPT-OSS 20B|2|10.8|21.5|
|Gemma 4 31B|2|1.6|3.1|
|Gemma 3 12B|2|0.8|1.6|
|OLMo 3 7B|2|0.7|1.4|
|Qwen 2.5 14B|2|0.5|0.9|
|Total|–|–|250.7|

**Cross-validation.** Held-out prediction experiments used five folds for each model–dataset pair and compared the MBT model against single-component baselines. The MBT cross-validation sweep used 140 fits allocated 4 CPU cores for 8–24 hours each, giving 4,480–13,440 allocated CPU-hours. The _K_ max = 1 baseline sweep used 140 fits allocated 2 CPU cores for 4–12 hours each, giving 1,120–3,360 allocated CPU-hours.

**Synthetic recovery.** The synthetic recovery study was the largest CPU component. It used 720 posterior fits over a grid of ground-truth component counts, inter-component score correlations, noise levels, DP concentration settings, and random seeds. Each fit was allocated 4 CPU cores and 16–32 GB RAM for 6–18 hours, for an allocated upper bound of 17,280–51,840 CPU-hours. This sweep validates the inference procedure but is not required to reproduce the main empirical results.

**Preliminary and unreported compute.** The full research project used more compute than the final reported experiments. In particular, the recorded LLM elicitation total was approximately 1,155 GPU-hours when including reruns, preliminary runs, failed parsing runs, and debugging runs, compared with approximately 251 GPU-hours for the final reported elicitation runs. Additional CPU time was used for exploratory posterior fits, plotting, cache extraction, and diagnostic scripts; these costs were small compared with the posterior inference and synthetic recovery sweeps.
