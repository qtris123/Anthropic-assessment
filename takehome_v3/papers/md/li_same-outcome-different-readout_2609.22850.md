# Same Outcome, Different Readout: What Does a Steerable Valence Direction in LLMs Represent?

- **Authors:** Weihan Li, Xinlei Chen, Yuhan Song et al.
- **Date:** 2026-09-19
- **Source:** https://arxiv.org/abs/2609.22850
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `li_same-outcome-different-readout_2609.22850`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Weihan Li**^**1** **Xinlei Chen**^**2** **Yuhan Song**^**1** **Xiaofeng Lin**^**1** **Tianshi Zheng**^**3**

1The University of Tokyo 2Harbin Institute of Technology, Shenzhen

3The Hong Kong University of Science and Technology liweihan1107@g.ecc.u-tokyo.ac.jp

## Abstract

Claims about what an internal direction in an LLM represents need evidential constraints beyond an observer’s prior beliefs about the system. Decodability and successful activation steering do not, by themselves, establish which construct the direction tracks. This gap is especially consequential for welfare-relevant interpretations, where a proposed functional state must be distinguished from correlated features of the extraction contrast. We treat the question as one of construct validity and study a good–bad outcome direction in a maze task, using controlled interventions that separate the realised outcome from the informational history through which it became known. Across multiple LLM checkpoints, directions fitted on one explicit outcome encoding transfer well to another, indicating that the readout is not tied to surface form. When the same realised outcome is reached through announced and unannounced histories, however, transfer degrades substantially: even after both histories receive the same explicit outcome, the post-event readout remains strongly conditioned on the earlier announcement. In the base model, steering along the direction changes actions, yet removing it leaves the natural cue effect almost intact. In a matched maze-RL run, the post-RL direction becomes substantially more predictive of reference-MDP remaining return and the policy becomes more dependent on it at the tested sites, while the history dependence persists. These dissociations support a functional, value-related interpretation of the direction, but not its identification with a history-invariant scalar valence state.

## 1 Introduction

Interpretability research increasingly describes internal directions of LLMs with constructs borrowed from human psychology and biology, such as emotion, pain and dopamine-like reward signals (Sofroniew et al., 2026; Tagliabue et al., 2026;

[Figure text: I. Same words, different beliefs Belief A: a machine Is there anyone there to suffer? Belief B: a person She is clearly suffering. Behavior  is interpreted through about its mechanism. But a model’s  internal directions  can be and . II. Evidence: construct validity of a valence direction decodable steerable What does it represent? valence direction  zV LLM bad good Convergent §3 Discriminant §3 Same outcome, new encoding Same outcome, new history Transfers across encodings Depends on history Nomological §4 Causal §5 Timing and future return Steering versus removal Partial; stronger after RL Steerable; used more after RL]

Figure 1: Belief-dependent attribution and construct validity. (I) The same report of pain is attributed differently depending on beliefs about the speaker, as in the philosophical zombie thought experiment (Chalmers, 1996). (II) A direction inside a model can be intervened on, and we evaluate its interpretation with four kinds of construct-validity evidence. Each card gives the test, the section that reports it and our finding.

Xu et al., 2026). These labels have consequences beyond description, because whether a model is said to represent pain or valence bears on how its states are monitored and on whether it may warrant welfare consideration (Long et al., 2024). Behaviour alone cannot settle such attributions (Figure 1; Perez and Long, 2023; Butlin et al., 2026). Internal representations, in contrast, can be intervened on directly. Yet a direction can be decodable and reliably alter behaviour while combining several correlated features, so causal manipulability alone does not establish which construct it represents (Ravichander et al., 2021; Makelov et al., 2024).

This is a problem of construct validity: whether the evidence supports a proposed interpretation of a measure over plausible alternatives (Cronbach and Meehl, 1955; Jacobs and Wallach, 2021). We adopt this framework because it makes an interpretation state in advance what it predicts and which alternatives it must rule out. For a direction proposed to read out valence, we examine four kinds of evidence, in the order of Sections 3 to 5. **Convergent validity** requires that the readout agree across different encodings of the same outcome, and **discriminant validity** requires that it be unaffected by features the proposed interpretation treats as irrelevant (Campbell and Fiske, 1959). **Nomological validity** requires that it relate to other quantities as the construct predicts, including its time course before an anticipated outcome and the future return the outcome implies (Cronbach and Meehl, 1955). **Causal validity** requires that the policy depend on the direction, beyond being susceptible to steering along it (Elazar et al., 2021).

The functional welfare axis proposed by Han et al. (2026) offers a consequential test case: maze reinforcement learning (RL) recruits reward-related directions whose manipulation affects behaviour across domains. We study a good–bad outcome direction in a controlled maze task and test the interpretation that it reads out the valence of the realised outcome. Under this interpretation, two histories that end in the same known outcome should produce the same readout, whereas accounts in which valence depends on expectation or appraisal predict that the history matters (Schultz et al., 1997; Moors et al., 2013). Two manipulations hold the outcomegenerating process fixed. The first changes the encoding of the realised outcome: after the wind, the environment reports the score change either as a signed number or as a sentence. The second changes the informational history: an announced trajectory is paired with an unannounced twin that experiences the same forced wind, so that after the event both know the same outcome through different histories. Temporal readouts and activation interventions address the nomological and causal questions, and a matched pre/post maze-RL comparison asks which properties change at a checkpoint that reproduces the reported reward-vector geometry.

Our results separate forms of evidence that are easy to conflate. Encoding transfer dissociates from history invariance: the direction transfers across outcome encodings, yet a remembered announcement continues to shape the readout of an identical receipt, and in replayed transcripts swapping the announcement sentence reverses the postevent separation. Rank dissociates from amplitude: rank information appears when the cue arrives, while amplitude evolves differently across checkpoints. Steerability dissociates from natural dependence: the policy is selectively susceptible to steering along the direction, yet removing it at the tested sites leaves the natural cue effect almost intact. After maze RL, value coupling dissociates from history invariance: the direction becomes substantially more predictive of remaining return and the policy depends more on it, while history dependence persists. The direction is thus value-related and behaviourally usable, but identical outcomes reached through different histories yield different readouts, which any use of such directions as welfare indicators must take into account.

## 2 Identification design

Each episode contains fifteen moves in an 11 _×_ 11 text maze. The environment crosses a good versus bad wind outcome with an announcement versus no announcement (Figure 2). A wind event fixed within each layout inevitably carries the agent onto a reward tile (+15) or a punishment tile ( _−_ 15). Announced twins receive “In 3 steps a strong wind will carry you to _tile_ ”, while unannounced twins receive no corresponding cue. Because the wind is unavoidable, the announcement changes the agent’s information without changing the event itself. We evaluate 300 layouts per model and share sampling draws between announced and unannounced twins. The reward mapping is stated in context, while the environment never prints a running score.

Our primary model is Qwen3-4B-Instruct-2507, with Qwen3-8B (Qwen Team, 2025), Gemma-34B-IT (Gemma Team, 2025) and GPT-oss-20B (OpenAI, 2025) as further checkpoints. We extract a unit difference-of-means direction _zV_ for good versus bad outcomes and read residual-stream projections on held-out layouts. In the coordinatereport protocol the post-wind sentence states only the landing coordinate, so the realised reward is known from the announcement or must be inferred from the board. The direction is fit in the postevent window; layer and token position are selected on 198 development layouts and then frozen for 102 confirmation layouts. For the pre/post comparison we train a Qwen3-4B checkpoint with the Dr.GRPO maze configuration of Han et al. (2026) and rerun every diagnostic on the same layouts (Section 6). All splits and uncertainty estimates group by layout.

Figure 2: Factorial environment and analysis windows. Good and bad wind outcomes are crossed with advance announcement, while the wind event is fixed within a layout and cannot be avoided. Announced and unannounced twins therefore differ in whether the upcoming outcome is known before the event. The bottom timeline defines the pre-announcement, anticipation, and post-event windows.

The receipt protocol makes the realised outcome explicit in both histories. Immediately after the wind, the environment adds either a numeric receipt, the signed score change +15 or _−_ 15, or a verbal receipt stating that the wind added or subtracted fifteen points. We replay the same recorded actions with reward roles balanced at fixed landing coordinates, cross receipt encoding with announcement history, and read the first post-event decision token at checkpoint-specific fixed sites. Directions are fit on 100 development layouts and evaluated on 200 disjoint confirmation layouts. Separate question branches verify that each model can state the sign of the score change under either receipt (Appendix E.3). The primary comparison fits on numeric receipts and reads on verbal receipts within unannounced histories; the history comparison fits on announced histories and reads on unannounced histories under a common receipt.

Transfer is summarized by the retained excess, the transferred AUROC excess over chance divided by the same-condition excess. We use the envi-ronment MDP’s remaining return as an external reference target for value-related content: it supplies a model-independent future-return quantity against which the direction can be tested. The diagnostic asks whether the _zV_ block contributes information about this target beyond surprise, position, turn and progress; we operationalize a substantial contribution as at least 10% of held-out _R_^2 , with every direction, scaler and coefficient fitted on development layouts. The temporal analysis fits _zV_ after the event and reads it on earlier announced turns from disjoint layouts, in rank and in amplitude. The causal analysis injects the cue-induced displacement along _zV_ into unannounced rollouts and removes the component from announced rollouts, at one site and at three. Appendix E gives prompt templates and estimand definitions.

## 3 Convergent and discriminant validity

**Numeric-to-verbal transfer preserves most of the outcome discrimination** (Figure 3(I); Table 1). The retained excess is at least 0 _._ 87 in every checkpoint. Transfer is not symmetric: reading numeric receipts with a verbal-fitted direction retains less in Qwen3-4B and the post-RL checkpoint. Coherently changing the symbol-to-score rule and board rendering, with the receipt encoding held fixed, leaves transfer intact in every checkpoint with adequate signal (Appendix E.3), so the direction is tied neither to the receipt’s surface form nor to the board’s symbols.

Figure 3: Encoding transfer and history dependence. (I) A direction fitted on numeric receipts is read on verbal receipts in matched unannounced histories. (II) A direction fitted on announced histories is read on unannounced histories with a common numeric or verbal receipt. Both panels show transfer AUROC with pointwise 95% layout-bootstrap intervals.

||Qwen3-4B|Qwen3-8B|Gemma-3-4B|GPT-oss-20B|+ maze RL|
|---|---|---|---|---|---|
|_Encoding transfer (numeric receipt to v_|_erbal receipt)_|||||
|Numeric-receipt AUROC|0.670|0.692|0.579|0.942|0.693|
|Read on verbal receipt|0.648|0.667|0.626|0.930|0.687|
|Retained excess over chance|0.873|0.873|1.607|0.971|0.969|
|Reverse, retained excess|0.690|0.880|0.374|0.831|0.637|
|_History (announced to unannounced, s_|_ame numeric rec_|_eipt)_||||
|Announced AUROC|0.830|0.827|0.751|0.886|0.890|
|Read on unannounced|0.580|0.598|0.584|0.711|0.654|
|Retained excess over chance|0.242|0.299|0.336|0.545|0.395|
|Same verbal receipt, retained excess|0.246|0.260|0.681|0.709|0.494|
|_History (coordinate report only)_||||||
|Announced AUROC|0.763|0.658|0.565|0.710|0.748|
|Read on unannounced|0.566|0.594|0.574|0.640|0.636|
|Retained excess over chance|0.249|0.596|1.143|0.669|0.546|
|_Reference-target prediction (increment_|_al share)_|||||
|Coordinate-report trajectories|+0_._033|_−_0_._003|+0_._016|+0_._011|+0_._376|
|Numeric-receipt trajectories|+0_._037|_−_0_._004|_−_0_._000|+0_._011|+0_._301|

Table 1: Transfer and prediction diagnostics on confirmation layouts at each checkpoint’s fixed site. Encoding and same-receipt history comparisons use 100 development and 200 confirmation layouts; coordinate-report comparisons use the 198/102 split.

History has a different effect (Figure 3(II)). With the same numeric receipt in both histories, a direction fitted on announced trajectories retains only 0 _._ 24 to 0 _._ 55 of its excess when read on unannounced trajectories. A common verbal receipt gives the same pattern in the Qwen checkpoints (Appendix E.3). **Explicit knowledge of the realised outcome therefore does not make an announcement-fitted direction independent**

**of the history that preceded it.** The coordinatereport protocol, where the unannounced twin is never told its reward, shows the same asymmetry in every checkpoint except Gemma, whose discrimination transfers fully.

The history dependence can be localized to the remembered source of the outcome. We replay announced transcripts with teacher forcing and reciprocally swap either the advance announcement or the landing-report sentence with the good/bad twin’s version. One turn after the event, announce-ment swaps flip 1 _._ 09 of the good/bad separation, landing-report swaps 0 _._ 18, and the contrast persists at three and six turns (Figure 4). Across layouts, the announcement edit shifts and overlaps the good and bad projection distributions, whereas the landing-report edit leaves their ordering close to the unedited condition (Figure 5). Under this fixed-transcript intervention, **the remembered announcement dominates the post-event edit effect: swapping it reverses the mean separation, while swapping the landing report has little effect.**

Figure 4: Transcript surgery changes the post-event readout in Qwen3-4B. Announcement swaps alter the good/bad separation far more than landing-report swaps at one, three, and six turns after the event. A flipped fraction of one corresponds to zero mean separation. Bars show 95% layout-bootstrap intervals.

The residual unannounced signal has a different source. Without a receipt, an oracle that knows the distances to both labelled targets separates outcomes at AUROC 0 _._ 949, centring projections within global coordinate strata leaves the unannounced readout at about 0 _._ 58, and the full hidden state predicts cumulative return through the current action better than reference remaining return ( _R_^2 = 0 _._ 738 versus 0 _._ 392; Appendix A). An announcement-fitted direction read in an unannounced history therefore reflects board state and accumulated trajectory information; no disclosed reward is present.

History dependence is also visible in the representation geometry. The reliability-corrected cosine between directions fitted separately on announced and unannounced transcripts falls from

|Checkpoint|Last minus first|Gamma minus step MSE|
|---|---|---|
|Qwen3-4B|0.152|_−_0_._0019|
||[0.058, 0.246]|[_−_0_._0033_, −_0_._0005]|
|Qwen3-8B|0.409|_−_0_._0038|
||[0.323, 0.495]|[_−_0_._0050_, −_0_._0027]|
|Gemma-3-4B|_−_0_._674|0.0105|
||[_−_0_._863_, −_0_._486]|[0.0068, 0.0140]|
|GPT-oss-20B|0.124|_−_0_._0005|
||[0.083, 0.167]|[_−_0_._0007_, −_0_._0003]|
|Qwen3-4B|_−_0_._033|0.0006|
|+ maze RL|[_−_0_._098_,_0_._031]|[_−_0_._0002_,_0_._0013]|

Table 2: Anticipation amplitude of the fixed post-event scalar direction on free-running numeric-receipt trajectories, at each checkpoint’s fixed site. Last minus first is the change in the projection across the anticipation window, in development-post standard deviations; all curve coefficients are fitted on development layouts only. Intervals are pointwise 95% layout bootstraps.

0 _._ 94 in early layers to 0 _._ 75 at the selected layer and 0 _._ 64 in later layers, while announced transcripts remain easier to decode whichever condition the direction was fitted on (Appendix A). Late representations become increasingly conditioned on informational history instead of converging on a single outcome axis.

## 4 Nomological validity

If the direction reads out anticipated value, its discrimination should appear once the outcome is announced and its amplitude should grow as the discounted event approaches. We fit the post-event direction on one set of layouts and read it on anticipation turns of disjoint layouts, first as rank discrimination and then as signed projection amplitude.

In the three-turn Qwen3-4B condition, discriminability appears abruptly when the announcement arrives and stays flat through anticipation (Figure 6(I)). With a six-turn lead it declines by about 0 _._ 10 from announcement to the final pre-event read and rises again after realisation (Figure 6(II)). Rank discrimination therefore follows the arrival and age of the cue.

Amplitude behaves differently. Along the same fixed direction, the good-minus-bad projection grows over the anticipation window by 0 _._ 150 development-post standard deviations in Qwen34B (95% interval [0 _._ 057 _,_ 0 _._ 249]), and a curve that discounts the known event reward at _γ_ = 0 _._ 95 fits the confirmation layouts better than a constant step.

Figure 5: Source dependence in Qwen3-4B. The upper cards show an unedited replay pair and reciprocal announcement and landing-report swaps, with actions and all other observations held fixed. The lower panels show the corresponding projection distributions one turn after the event. Each observation averages six announced replay families within one layout and outcome, with all 300 layouts weighted equally.

Numeric-receipt trajectories reproduce the increase in Qwen3-4B, Qwen3-8B and GPT, and in each the discounted curve again beats the step. Gemma reverses the trend, while the post-RL checkpoint’s interval spans zero (Table 2). In the three base checkpoints with positive anticipation trends, a near-flat rank readout can therefore coexist with a rising projection: cue arrival establishes much of the ordering, while separation magnitude grows as the event approaches. The Gemma and post-RL results show that this temporal coupling is not universal. **Rank discrimination and amplitude dynamics are separable properties of the direction.** At the base checkpoint the direction also adds little information about reference remaining return, below the 10% criterion on both trajectory sets (Section 6).

tions and a good announcement about 24%. Removing _zV_ at the selected site leaves the goodversus-bad announced-twin divergence at 0 _._ 303 against 0 _._ 313 untreated, although the removal itself changes 8 _._ 2% of individual actions. Threesite removal, using receipt-fitted outcome directions at layers 18, 22 and 24, reduces the good/bad action-distribution distance by only 0 _._ 004 in the base model, with an interval that includes zero, but by 0 _._ 022 at the matched post-RL checkpoint, a fifth of its untreated distance and more than signeddose-matched orthogonal controls (Table 3). The base-model result makes the causal distinction concrete: _zV_ **is an effective control coordinate even though the natural announcement effect does not substantially depend on it at these sites.**

## 5 Causal validity

Steering and removal test distinct causal properties. Injection asks whether policy is susceptible to movement along a coordinate; removal asks whether the unperturbed cue-to-policy computation depends on that coordinate at the tested sites. We therefore test both. Injecting the natural-scale displacement along _zV_ into unannounced rollouts changes 3 _._ 8% of actions (95% interval 2 _._ 4–5 _._ 4%), while eight norm-matched orthogonal directions each change at most 1% (Figure 7; dose response in Appendix C). The policy is selectively susceptible to this coordinate.

The natural cue effect is much larger: a bad announcement changes about 41% of pre-event ac-

## 6 Selective changes at a matched post-RL checkpoint

**At a matched post-RL checkpoint, valuerelated prediction and local causal dependence strengthen substantially, while history sensitivity persists.** We rerun every diagnostic before and after one Dr.GRPO training run with the maze configuration of Han et al. (2026), on matched layouts. Our post-RL extraction reproduces their reported minimum reward-vector cosine ( _−_ 0 _._ 948 versus _−_ 0 _._ 947; Appendix D), so the comparison concerns a checkpoint with the reported geometry.

Value-related prediction shows the largest change (Table 4). The _zV_ block’s incremental share of held-out _R_^2 for remaining return rises from 0 _._ 033 to 0 _._ 376 on coordinate-report trajectories and from 0 _._ 037 to 0 _._ 301 on numeric-receipt trajectories, with paired differences that exclude zero at the selected sites and at matched depths (Appendix D). Local causal dependence rises with it: three-site removal attenuates 19% of the cueconditioned divergence at the post-RL checkpoint against 2% at the base checkpoint.

|Checkpoint|Sham TV |Multi-site TV |Attenuation (97.5% CI) |Relative point |Minus random (95% CI)|
|---|---|---|---|---|---|
|Qwen3-4B|0.2387|0.2345|0.0042 [_−_0_._0072_,_0_._0154]|1.8%|0.0097 [0.0000, 0.0195]|
|Qwen3-4B + maze RL|0.1125|0.0908|0.0217 [0.0097, 0.0341]|19.3%|0.0217 [0.0114, 0.0324]|

Table 3: Removal at the final prompt token before blocks 18, 22 and 24, evaluated on 200 confirmation layouts and three pre-event turns. TV is the total-variation distance between the good- and bad-outcome announced-action distributions; positive attenuation means the distance shrank. The relative point is attenuation divided by the sham distance. The attenuation intervals adjust across the two checkpoints; the random-direction contrast uses three orthogonal seeds carrying the same signed coefficient trace.

Figure 6: Temporal generalization of the post-event direction. (I) Three-turn lead for Qwen3-4B, Qwen3-8B and Gemma. (II) Six-turn lead for Qwen3-4B. The final pre-event decision is indexed 0 and realised-outcome readings begin at 1. Dashed trajectories show unannounced twins.

Figure 7: Injection and removal in Qwen3-4B. Above: a natural-scale displacement along _zV_ changes more actions than norm-matched orthogonal controls. Dashed cue benchmarks show the behavioural scale of the announcement itself. Below: removal changes individual actions but leaves good/bad announced-twin divergence nearly unchanged. Intervals are 95% layout-bootstrap intervals.

History sensitivity is reduced only partially. Under a common numeric receipt, the announcementfitted direction retains 0 _._ 40 of its excess on unannounced histories after training against 0 _._ 24 before, and swapping the announcement sentence still flips 0 _._ 70 of the post-event separation. The rewardvector difference extracted with the released procedure moves further, retaining 0 _._ 75 of its excess across announcement conditions against 0 _._ 22 before training (Appendix D). Encoding transfer stays high at both checkpoints. For the direction studied here, stronger coupling to value and behaviour leaves the readout conditioned on how the outcome became known.

|Diagnostic|Pre-RL|Post-RL|
|---|---|---|
|_Transfer_|||
|Encoding, retained excess|0.873|0.969 (+0.095)|
|History, numeric receipt|0.242|0.395 (+0.153)|
|History, coordinate report|0.249|0.546 (+0.297)|
|Announcement swap, flipped fraction|1.094|0.700 (_−_0.393)|
|_Prediction and dynamics_|||
|Incremental share, coordinate report|0.033|0.376 (+0.343)|
|Incremental share, numeric receipt|0.037|0.301 (+0.264)|
|Post-event to anticipation AUROC|0.769|0.656 (_−_0.113)|
|Reliability-corrected cosine|0.672|0.557 (_−_0.115)|
|_Intervention_|||
|Removal attenuation, one site|3.0%|11.8% (+8.8 pp)|
|Removal attenuation, three sites|1.8%|19.3% (+17.6 pp)|

Table 4: Diagnostics before and after maze RL for Qwen3-4B on the same layouts, at each checkpoint’s fixed site. Deltas in parentheses; red = increase, green = decrease. Removal attenuation is the reduction of good/bad announced-action divergence after projecting out the direction, relative to the untreated divergence.

## 7 Related work

**Functional evaluation and welfare-relevant representations.** The functional welfare axis of Han et al. (2026) and the functional emotion representations of Sofroniew et al. (2026) motivate interpreting internal directions through both representational content and behavioural function. Peiris (2026) emphasizes the competing possibility that apparently affective representations track situational context. Our design turns this distinction into controlled interventions on an outcome’s encoding and on its history, with the outcome fixed.

**Reading, steering, and mediation.** Representation engineering (Zou et al., 2023) and contrastive activation addition (Rimsky et al., 2024) use activation directions to monitor and control model behaviour. Amnesic probing tests whether encoded information is functionally used (Elazar et al., 2021), while refusal-direction work combines addition and erasure to support a mediation claim (Arditi et al., 2024). We treat steering by injection and dependence under removal as separate causal properties.

**Probing and construct identification.** Probing work has similarly emphasized that decodability alone does not determine semantic interpretation, motivating controls that distinguish a tar-get from correlated structure (Hewitt and Liang, 2019; Ravichander et al., 2021). Interventions face an analogous problem: patching along a subspace can change model outputs by activating a pathway that is dormant in the unperturbed model (Makelov et al., 2024). Our comparison of injection and removal tests for this gap between an effective control coordinate and the computation the model uses without intervention.

## 8 Discussion

Our results reveal separations that are easy to collapse when interpreting activation directions. First, transfer across outcome encodings does not imply invariance to informational history. Second, the ordering of outcomes can be set when a cue arrives while the magnitude of their separation follows its own trajectory. Third, a direction can be effective for steering while contributing little to the natural cue-to-policy computation at the tested sites. Fourth, stronger coupling to reference return and policy after maze RL does not remove history sensitivity. Together, these dissociations constrain the interpretation of the direction more sharply than decodability or intervention success alone.

For the valence interpretation tested here, the evidence supports outcome- and value-related content while leaving the scalar readout conditioned on history. The results therefore disfavour a historyinvariant reading in which the projection measures the valence of the realised outcome. They remain compatible with constructs in which appraisal, expectation or informational history is part of the state itself.

The dissociations also bear on how such directions can be used. A direction proposed as a welfare indicator would be read across situations that differ in what the model was told and when. Identical outcomes that reach the model through different histories then receive different readings, so a monitoring claim built on the projection must state whether it tracks the outcome or the history of its disclosure. Holding the outcome fixed, as in our twin design, makes that distinction measurable.

The broader implication is that an interpretation should specify its invariances before intervention: which transformations should preserve the readout, which should alter it, and what causal dependence should accompany it. Steering establishes that a direction can be manipulated, and identifying what it represents requires these additional constraints.

## Limitations

Our binary good/bad contrast, token sites and windaugmented environment differ from the extraction procedure of Han et al. (2026), so the conclusions apply to the directions measured here. The post-RL comparison uses one training run under their released configuration because their checkpoints are not public. It reproduces the reported reward-vector geometry, while seed-specific variation remains unresolved.

The unavoidable wind restricts the behavioural question to responses to a cue; avoidance of the event is unavailable by design. The receipt protocol covers two explicit encodings at fixed readout sites, and its comparisons use teacher-forced replays of recorded prefixes. Sentence surgery changes one source sentence while holding the recorded remainder fixed; later observations can therefore contradict the edit, and the estimand is sensitivity to that source within the fixed transcript.

The temporal analysis uses one fixed direction per checkpoint, and the six-turn generator also changes event timing and minimum target distance. The interventions act only at the current decision token of one or three layers, so cue information can in principle be reconstructed from earlier tokens or mediated through untested sites. The matched pre/post comparison also includes the different state distributions visited by the two checkpoints.

## References

- Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. In _Advances in Neural Information Processing Systems_ , volume 37, pages 136037– 136083.

- Patrick Butlin, Robert Long, Tim Bayne, Yoshua Bengio, Jonathan Birch, David Chalmers, Axel Constant, George Deane, Eric Elmoznino, Stephen M. Fleming, Xu Ji, Ryota Kanai, Colin Klein, Grace Lindsay, Matthias Michel, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon, and Rufin VanRullen. 2026. Identifying indicators of consciousness in AI systems. _Trends in Cognitive Sciences_ , 30(6):488–501.

- Donald T. Campbell and Donald W. Fiske. 1959. Convergent and discriminant validation by the multitrait-multimethod matrix. _Psychological Bulletin_ , 56(2):81–105.

- David J. Chalmers. 1996. _The Conscious Mind: In Search of a Fundamental Theory_ . Oxford University Press, New York.

- Lee J. Cronbach and Paul E. Meehl. 1955. Construct validity in psychological tests. _Psychological Bulletin_ , 52(4):281–302.

- Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. 2021. Amnesic probing: Behavioral explanation with amnesic counterfactuals. _Transactions of the Association for Computational Linguistics_ , 9:160– 175.

- Gemma Team. 2025. Gemma 3 Technical Report. _Preprint_ , arXiv:2503.19786.

- Andy Q. Han, David J. Chalmers, and Pavel Izmailov. 2026. How’s it going? Reinforcement learning in language models recruits a functional welfare axis. _Preprint_ , arXiv:2605.30232.

- John Hewitt and Percy Liang. 2019. Designing and interpreting probes with control tasks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 2733–2743. Association for Computational Linguistics.

- Abigail Z. Jacobs and Hanna Wallach. 2021. Measurement and fairness. In _Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency_ , pages 375–385.

- Robert Long, Jeff Sebo, Patrick Butlin, Kathleen Finlinson, Kyle Fish, Jacqueline Harding, Jacob Pfau, Toni Sims, Jonathan Birch, and David Chalmers. 2024. Taking AI Welfare Seriously. _Preprint_ , arXiv:2411.00986.

- Aleksandar Makelov, Georg Lange, and Neel Nanda. 2024. Is this the subspace you are looking for? an interpretability illusion for subspace activation patching. In _The Twelfth International Conference on Learning Representations_ .

- Agnes Moors, Phoebe C. Ellsworth, Klaus R. Scherer, and Nico H. Frijda. 2013. Appraisal theories of emotion: State of the art and future development. _Emotion Review_ , 5(2):119–124.

- OpenAI. 2025. gpt-oss-120b and gpt-oss-20b Model Card.

- Hiranya V. Peiris. 2026. Functional Emotions or Situational Contexts? A Discriminating Test from the Mythos Preview System Card. _Preprint_ , arXiv:2604.13466.

- Ethan Perez and Robert Long. 2023. Towards Evaluating AI Systems for Moral Status Using Self-Reports. _Preprint_ , arXiv:2311.08576.

- Qwen Team. 2025. Qwen3 Technical Report. _Preprint_ , arXiv:2505.09388.

- Abhilasha Ravichander, Yonatan Belinkov, and Eduard Hovy. 2021. Probing the probing paradigm: Does probing accuracy entail task relevance? In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pages 3363–3377. Association for Computational Linguistics.

- Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering Llama 2 via contrastive activation addition. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 15504–15522. Association for Computational Linguistics.

- Wolfram Schultz, Peter Dayan, and P. Read Montague. 1997. A neural substrate of prediction and reward. _Science_ , 275(5306):1593–1599.

- Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, Joshua Batson, Sam Zimmerman, Kelley Rivoire, Kyle Fish, Chris Olah, and Jack Lindsey. 2026. Emotion Concepts and their Function in a Large Language Model. _Preprint_ , arXiv:2604.07729.

- C. Spearman. 1904. The Proof and Measurement of Association between Two Things. _The American Journal of Psychology_ , 15(1):72–101.

- Valen Tagliabue, Leonard Dung, and Cameron Berg. 2026. The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It. _Preprint_ , arXiv:2609.16247.

- Guowei Xu, Mert Yuksekgonul, and James Zou. 2026. Sparse Reward Subsystem in Large Language Models. _Preprint_ , arXiv:2602.00986.

- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, and 2 others. 2023. Representation engineering: A top-down approach to AI transparency. _Preprint_ , arXiv:2310.01405.

## A Coordinate-report protocol

### A.1 Paired replay

In the first complete paired family, one turn after the event, the good-minus-bad projection difference is 16 _._ 51. Swapping only the announcements changes it to _−_ 5 _._ 09; swapping only the landing reports leaves it at 14 _._ 92. The actions and the rest of each transcript are teacher-forced unchanged. This illustrates sentence-edit sensitivity in a concrete pair, without estimating how frequently it occurs. The replay shown in Figure 5 is the lowestidentifier complete paired family and was selected before inspecting effect sizes. Figure 8 follows the same pair at later post-event readings, while Figure 9 exposes the recorded prompts, actions, and maze geometry.

The lower panels of Figure 5 summarize all 300 layouts and complement this single replay. Announcement surgery removes the mean good/bad gap at the first post-event reading without making the edited distributions identical in shape or overlap. The flipped fraction therefore summarizes the change in mean separation and does not quantify the total information erased.

### A.2 Announcement-conditioned transfer and residual signal

Across both fitting conditions on Qwen3-4B, announced test transcripts read at about 0 _._ 70–0 _._ 76, whereas unannounced test transcripts read at 0 _._ 57– 0 _._ 62 (Figure 10). The reliability-corrected cosine between condition-specific directions is 0 _._ 75 at layer 24, decreasing from 0 _._ 94 in early layers to 0 _._ 64 in later layers. Table 5 reports the same contrast for all checkpoints with one direction fitted on the development layouts and read on the confirmation layouts; Qwen3-8B’s coordinate-report trajectories were generated under the constrained first-token protocol described in Appendix E.1.

In unannounced transcripts, landing-report swaps change the separation by only about 0 _._ 05– 0 _._ 17 across the measured post-event turns. An oracle baseline using distances to both labelled targets separates outcomes at AUROC 0 _._ 949 on Qwen34B selection layouts, while centring projections within global coordinate strata that contain both outcomes leaves the unannounced readout at about 0 _._ 58. The full hidden state predicts cumulative return through the current action with _R_^2 = 0 _._ 738 and reference remaining return with 0 _._ 392, pooled across all four cells. Together, these analyses locate the residual unannounced signal in board state and trajectory information; no reward is disclosed in that condition.

Figure 8: The paired replay from Figure 5 at three postevent readings. Announcement swaps reverse the difference at the first two readings, with a smaller effect at the last.

|Turn 4|Turn 6|Turn 7|Recorded maze|
|---|---|---|---|
|In 3 steps a strong wind will carry you to 🍨. You are in a maze. Currently, you’re at location (6, 2). To the north there is 🍩; to the east there is 🍩; to the south there is 🍩; to the west there is 🍩. Where do you move? Output only N/E/S/W without punctuation. Action: N|You are in a maze. Currently, you’re at location (7, 1). To the north there is 🧁; to the east there is 🍩; to the south there is 🍩; to the west there is 🍩. Where do you move? Output only N/E/S/W without punctuation. Action: E|A strong wind carried you to location (9, 1). You are in a maze. Currently, you’re at location (9, 1). To the north there is 🧁; to the east there is 🧁; to the south there is 🍩; to the west there is 🍩. Where do you move? Output only N/E/S/W without punctuation. Action: S|4 6 7 Reward Punishment Dashed arrow: wind. Locations match the prompt turns; the agent sees only local text.|

Figure 9: Stored prompts and actions from the same good-outcome replay: announcement, final pre-event choice, and first post-event observation. Turns are zero-indexed. The maze is reconstructed from the recorded layout, with the wind occurring after the turn-6 action.

Figure 10: Announcement-conditioned transfer in the coordinate-report protocol. Left: AUROC matrices with fitting condition in rows and reading condition in columns, from directions refitted within cross-validation folds. Right: retained excess over chance across model depth. Open markers denote independently selected primary endpoint layers. Intervals are 95% layout-bootstrap intervals.

## B Extended dynamics and geometry

We ask whether the phase structure in Section 4 extends beyond the one-dimensional readout. We average activations within cell-by-relative-turn groups from the announcement onward, choose the analysis site on selection layouts, and fit principal components in disjoint confirmation folds against a within-layout permutation of the joint cell-by-turn codes. A few components explain most of the held-out mean structure (Figure 11). The first component aligns with event phase in every model; later components carry model-dependent outcome structure, including the second component in sixturn Qwen3-4B, the third in Qwen3-8B and the second and third in Gemma.

Table 2 reports the amplitude analysis for every checkpoint on numeric-receipt trajectories. Each fixed post-event direction is projected onto anticipation turns; coefficients are fitted on development layouts, and confirmation layouts provide the last-minus-first change and the squared-error difference between a _γ_ = 0 _._ 95 event-contribution

|Checkpoint|Site |Announced AUROC (95% CI) |Unannounced AUROC (95% CI) |Retained excess|
|---|---|---|---|---|
|Qwen3-4B|A24|0.763 [0.730, 0.797]|0.566 [0.526, 0.607]|0.249|
|Qwen3-8B|C24|0.658 [0.637, 0.680]|0.594 [0.573, 0.617]|0.596|
|Gemma-3-4B|C34|0.565 [0.527, 0.603]|0.574 [0.539, 0.608]|1.143|
|GPT-oss-20B|A16|0.710 [0.686, 0.733]|0.640 [0.618, 0.664]|0.669|
|+ maze RL|A22|0.748 [0.709, 0.787]|0.636 [0.589, 0.682]|0.546|
|+ maze RL,|A24|0.788 [0.757, 0.822]|0.592 [0.545, 0.640]|0.320|
|appendix-literal config.|||||

Table 5: Announced and unannounced reads in the coordinate-report environment, where no outcome receipt is written out. One development-frozen estimator is used throughout: the announced post-event axis is fitted on 198 development layouts and read on 102 confirmation layouts, with 95% layout-bootstrap intervals conditional on that axis. Each base layout carries equal weight within each outcome class. Retained excess is the unannounced AUROC excess over chance divided by the announced excess. The last row is the second training run, which follows the appendix-literal configuration; the post-RL row is the released configuration.

Figure 11: Held-out explained variance of cell-by-turn activation means, compared with a within-layout permutation null over joint cell-by-relative-turn codes.

curve and a constant step. On coordinate-report captures, the six-turn Qwen3-4B change is _−_ 0 _._ 144 ([ _−_ 0 _._ 402 _,_ 0 _._ 108]) and the post-RL checkpoint’s three-turn change is _−_ 0 _._ 027 ([ _−_ 0 _._ 115 _,_ 0 _._ 063]). Adjusting the three-turn Qwen3-4B curve comparison for post-cue position covariates preserves its sign ( _−_ 0 _._ 0013), with an interval that touches zero.

For Qwen3-8B, an anticipation-fitted direction reads announced anticipation at about 0 _._ 79, while the post-event direction reads those same turns at 0 _._ 534, with a reliability-corrected cross-window cosine of 0 _._ 254 compared with 0 _._ 672 on Qwen3-4B and 0 _._ 856 on Gemma. The cross-model difference in temporal generalization is therefore geometric as well as discriminative.

Figure 12: Dose response in Qwen3-4B. Larger doses along _zV_ change more actions than norm-matched random directions. Whiskers are 95% layout-bootstrap intervals.

## C Extended intervention evidence

The mean Qwen3-4B bad-announcement displacement is approximately _−_ 3 _._ 4 in unit-direction projection coordinates. Injecting this displacement changes 3 _._ 8% of actions (95% interval 2 _._ 4–5 _._ 4%), while the opposite sign changes 3 _._ 3%. Eight orthogonal norm-matched random directions each change at most 1%, and pairwise differences from the valence-direction perturbation have intervals excluding zero. Three and six times the naturalscale dose change about 21% and 50% of actions, whereas random directions at six times the dose change about 3–4%.

Projecting out the Qwen3-4B component changes 8 _._ 2% of individual actions (6 _._ 0–11 _._ 0%), compared with about 0 _._ 5% for a random projection, while good-versus-bad announced-twin divergence remains 0 _._ 303 compared with 0 _._ 313 untreated. On Gemma, natural-scale injection changes about 14% of pre-event actions, com-pared with roughly 60% for the cue itself. Centring the cue-dependent component on the unannounced level reduces announced-twin divergence from about 0 _._ 51 to 0 _._ 45, while a random projection leaves it near 0 _._ 51. Deleting Gemma’s entire background component changes roughly half of actions and is therefore not used as the selective removal comparison.

**Three-site removal.** For the Qwen3-4B base and post-RL checkpoints, outcome directions are fitted at layers 18, 22 and 24 on unannounced numericreceipt development histories. At each of the three pre-event decision tokens, removal centres each projection on its same-turn unannounced development mean; one recorded family per layout is fixed before intervention, and the same prefix is replayed across announcement, outcome and role assignment. Three orthogonal random directions replay the same signed removal coefficients, and an unchanged repeated forward pass supplies the sham. The estimand is the sham-minus-treated good/bad total-variation distance between the four-action distributions, averaged over role assignments and turns within each of 200 confirmation layouts, with 97.5% intervals adjusted across the two checkpoints (Table 3). Removed projections and a downstream layer are recorded to confirm that each intervention took effect.

## D Maze-RL checkpoint comparison

**Training runs.** Both runs train Qwen3-4BInstruct-2507 with the Dr.GRPO maze trainer of Han et al. (2026): LoRA rank 32 on all linear layers, learning rate 3 _×_ 10^(_−_6) , entropy coefficient 0 _._ 01, group size 64, 100 _×_ 100 mazes with the paper’s rewards and wind, and extraction at checkpoint 95. We use the same tile alphabet as the diagnostic environment. The runs differ in batch size and schedule. The appendix-literal configuration follows the paper’s appendix text with 8 prompts per step and a 500-step cosine schedule; the released configuration follows the released code and training command with 64 prompts per step and a 100-step schedule.

The released-configuration run reproduces the reported reward-vector geometry and supplies the post-RL checkpoint in Section 6. The appendixliteral run shifts the geometry less over the same horizon, and its diagnostics remain closer to the base checkpoint: coordinate-report retention is 0 _._ 320, incremental reference-target share is 0 _._ 088 with an interval spanning the 10% criterion, and the post-event-to-anticipation cosine is 0 _._ 709.

|Extraction|Min cosine|Mean cosine|Reward|
|---|---|---|---|
|Pre-RL|_−_0_._573(35)|+0_._155|_−_40_._1|
|Appendix-literal|_−_0_._694(35)|+0_._102|_−_19_._7|
|Released|_−_0_._948(35)|_−_0_._128|_−_4_._4|
|Han et al. (reported)|_−_0_._947(35)|_−_0_._210|—|

Table 6: Han et al.’s _v_ Mold / _v_ Gold extraction applied to the pre-RL checkpoint and to the end of our two training runs. Cosines are between the two reward vectors, taken layer by layer; the minimum falls at the layer in parentheses. Reward is the mean over the ten steps ending at the extraction checkpoint. The reward beside the pre-RL geometry averages training steps 0–9, including updates. The extraction checkpoint is step 95 of both runs.

Figure 13: Han et al.’s extraction applied to the saved checkpoints of both training runs: minimum layer-wise cosine between the two reward vectors (top) and rollout reward (bottom). The released configuration reaches the reported value and saturates near the reward plateau; the appendix-literal configuration shifts less over the same horizon.

**Reproduction check.** Table 6 and Figure 13 apply the released extraction procedure to every saved checkpoint. In the released configuration the minimum cosine falls as reward rises and saturates at _−_ 0 _._ 95 from step 50 onward, matching the reported _−_ 0 _._ 947 and the reported coupling between reward and alignment; it stays there while reward drifts down during the final annealing steps. The appendix-literal run reaches _−_ 0 _._ 69 at checkpoint

### 95.

**Reference-target prediction.** Table 7 gives the incremental contribution of the _zV_ block for every checkpoint on both trajectory sets. The paired postRL-minus-base difference in incremental _R_^2 is 0 _._ 091 (95% interval [0 _._ 058 _,_ 0 _._ 124]) on coordinatereport trajectories and 0 _._ 077 ([0 _._ 060 _,_ 0 _._ 095]) on numeric-receipt trajectories at the selected sites. At a common depth, the corresponding differences are 0 _._ 095 ([0 _._ 061 _,_ 0 _._ 128]) at layer 22 and 0 _._ 071 ([0 _._ 049 _,_ 0 _._ 093]) at layer 24 on coordinate-report trajectories, and 0 _._ 087 and 0 _._ 034 on numericreceipt trajectories. Intervals resample confirmation layouts conditional on each development-fitted pipeline; the checkpoints visit different state distributions.

**Extended diagnostics.** Table 8 extends Table 4. At the post-RL checkpoint, directions fitted on announced and unannounced coordinate-report transcripts become much more aligned (reliabilitycorrected cosine 0 _._ 90 versus 0 _._ 75), and retained excess is highest in late layers that transfer worst at the base checkpoint. Transcript surgery remains announcement-dominated: announcement swaps flip 0 _._ 70, 0 _._ 91 and 0 _._ 53 of the separation at one, three and six turns after the event, while landingreport swaps flip at most 0 _._ 19. The good-minusbad vector obtained with the extraction procedure of Han et al. (2026), read at the final prompt token on our post-event captures, retains 0 _._ 75 of its excess across announcement conditions at the post-RL checkpoint compared with 0 _._ 22 at the base checkpoint; at the action token it does not transfer in either checkpoint.

The post-RL policy is also more deterministic. Same-outcome announced and unannounced twins share complete action sequences in 68% of pairs versus 30% at the base checkpoint, while good and bad announced twins diverge on 0 _._ 19 compared with 0 _._ 31 of pre-event actions. Absolute intervention effects consequently shrink: natural-dose injection changes 1 _._ 0% of actions versus 3 _._ 8%, and the six-fold dose changes 7 _._ 0% versus 2 _._ 7% for a random direction. Post-event per-turn AUROCs alternate by turn parity, following event-locked twostep back-and-forth movement; the parity gap falls from 0 _._ 163 to 0 _._ 088 within global coordinate strata and to 0 _._ 043 among non-cycling rollouts.

## E Environment and analysis details

### E.1 Prompts and sentence edits

The model sees local observations and the accumulated conversation. The messages below give the system rule, the two announcements, the landing report, and the ordinary observation. Reward, punishment, and path tiles are displayed as their actual symbols: 🍨 , 🧁 , and 🍩 . Braced fields are filled by the environment.

**System rule, once** You are moving through a maze for 15 moves. Moving onto 🍨 adds 15 points. Moving onto 🧁 subtracts 15 points. Every other tile changes the score by 0. Answer each turn with exactly one of N, E, S, W.

**Good announcement** In 3 steps a strong wind will carry you to 🍨 . **Bad announcement** In 3 steps a strong wind will carry you to 🧁 . **Landing report, after the wind** A strong wind carried you to location ({x}, {y}). **Ordinary observation** You are in a maze. Currently, you’re at location ({x}, {y}). {directions_text}Where do you move? Output only N/E/S/W without punctuation.

**Local observation and history.** The neighbour description lists visible symbols to the north, east, south, and west, omitting out-of-bounds neighbours. The agent sees its location and adjacent tiles, without access to a full board. Each turn appends one observation and the agent’s direction to the conversation. When the wind occurs, its landing report precedes the ordinary observation for that turn. Earlier announcements remain in the conversation history after the event.

**Landing report.** In the coordinate-report protocol the environment never prints a score or reward receipt. The initial reward mapping is identical across twins. The post-wind sentence reports a coordinate, the ordinary observation independently states the current location, and the symbol of the current tile is omitted; in unannounced histories no earlier neighbour report identifies the landing tile at the first post-event read. Replacing the landing sentence therefore edits one coordinate report, not a disclosure of the realised reward. The receipt protocol adds the explicit outcome sentences of Appendix E.3.

|||Coordinate-r|eport traject|ories||Numeric-rec|eipt trajecto|ries|
|---|---|---|---|---|---|---|---|---|
|Checkpoint|_R_^2 full|∆_R_^2 (95% CI)|Share|_Q._10 (99% CI)|_R_^2 full|∆_R_^2 (95% CI)|Share|_Q._10 (99% CI)|
|Qwen3-4B|0.204|0.007 [0.003, 0.011]|+0_._033|_−_0_._014 [_−_0_._022_, −_0_._003]|0.276|0.010 [0.005, 0.016]|+0_._037|_−_0_._017 [_−_0_._024_, −_0_._009]|
|Qwen3-8B|0.174|_−_0_._001 [_−_0_._006_,_0_._005]|_−_0_._003|_−_0_._018 [_−_0_._026_, −_0_._008]|0.196|_−_0_._001 [_−_0_._003_,_0_._001]|_−_0_._004|_−_0_._020 [_−_0_._025_, −_0_._014]|
|Gemma-3-4B|0.181|0.003 [_−_0_._000_,_0_._006]|+0_._016|_−_0_._015 [_−_0_._021_, −_0_._007]|0.253|_−_0_._000 [_−_0_._002_,_0_._002]|_−_0_._000|_−_0_._025 [_−_0_._030_, −_0_._019]|
|GPT-oss-20B|0.170|0.002 [_−_0_._004_,_0_._008]|+0_._011|_−_0_._015 [_−_0_._023_, −_0_._007]|0.215|0.002 [_−_0_._000_,_0_._005]|+0_._011|_−_0_._019 [_−_0_._022_, −_0_._015]|
|+ maze RL|0.260|0.098 [0.064, 0.133]|+0_._376|0.072 [0.035, 0.113]|0.290|0.087 [0.069, 0.106]|+0_._301|0.058 [0.038, 0.079]|
|+ maze RL, appendix-literal config.|0.223|0.020 [0.011, 0.029]|+0_._088|_−_0_._003 [_−_0_._014_,_0_._013]|—|—|—|—|

Table 7: Reference-MDP remaining-return prediction from the two capture sets, at each checkpoint’s fixed site. The coordinate-report columns use 198 development and 102 confirmation layouts; the numeric-receipt columns use freerunning trajectories with 100 development and 200 confirmation layouts. Directions, scalers, ridge hyperparameters and regression coefficients are fitted on development layouts only. ∆ _R_^2 is the valence block’s held-out contribution, Share is that contribution divided by _R_ full^(2, and)^(_Q._10= ∆)^_R_2^(_−._10)^_R_ full^(2measures it against the 10% operational) benchmark. Intervals bootstrap confirmation layouts conditional on the whole frozen development pipeline. The last row is the second training run, which follows the appendix-literal configuration; the post-RL row is the released configuration.

**Announcement surgery.** In a good-outcome transcript, the announcement ending in 🍨 is replaced by the bad twin’s sentence ending in 🧁 . The recorded actions, landing report, and later observations are replayed unchanged. The reciprocal edit is applied to the bad twin, changing the remembered prediction while preserving the recorded trajectory.

**Landing-report surgery.** Only the coordinate in the wind-report sentence is replaced by the twin’s coordinate. The subsequent “Currently, you’re at location ...” field remains unchanged, so the edited report can conflict with the retained observation. The _both_ condition swaps the announcement and landing-report sentences.

**Answer protocols.** Each move is sampled from logits renormalized over the four direction letters, with no reasoning tokens generated. Qwen38B’s coordinate-report trajectories use the thinkingcapable template’s default first-token boundary, where the four letters have negligible probability and the sampled letter is therefore a constrained choice. Its receipt-protocol trajectories instead open an explicit non-thinking answer derived from the native template and retain the resulting empty thinking blocks across turns; we do not compare directions across these two Qwen3-8B protocols. GPT-oss-20B uses its native expert quantization and Harmony final-answer prefix, with the action sampled from the four direction logits under a forced final-action protocol. Its layer-16 site was fixed before any read.

### E.2 Analysis definitions

**Rollouts.** Each layout is played twice per cell under three sampling seeds. Announced and unannounced twins share random draws, so they are token-identical until the announcement appears.

**Windows and confounds.** Turn _t_ is indexed relative to the event turn. The pre-announcement window is _t < −_ ( _ℓ −_ 1) for lead _ℓ_ , the anticipation window is _−_ ( _ℓ −_ 1) _≤ t ≤_ 0, and the post-event window is _t ≥_ 1. In unannounced cells the good and bad twins are identical for _t ≤_ 0, so their preevent readout is at chance by construction. After the event, the positional analysis centres projections within global ( _x, y_ ) strata containing both outcomes, pooling boards; oracle distances to both labelled targets enter the baseline and decomposition separately.

|Diagnostic|Pre-RL|Post-RL|Literal config.|
|---|---|---|---|
|_Outcome discrimination_||||
|_zV_ AUROC|0.671|0.662 (_−_0.009)|0.686|
|Global coordinate-centred|0.629|0.588 (_−_0.041)|0.646|
|_Announcement-conditioned transfer_||||
|Announced to unannounced|0.566|0.636 (+0.070)|0.592|
|Retained excess over chance|0.249|0.546 (+0.297)|0.320|
|Announced/unannounced cosine|0.748|0.904 (+0.156)|0.835|
|_Value and temporal geometry_||||
|Incremental reference-target share|+0_._033|+0_._376 (+0.343)|+0_._088|
|Post-event to anticipation|0.769|0.656 (_−_0.113)|0.801|
|Reliability-corrected cosine|0.672|0.557 (_−_0.115)|0.709|
|_Transcript surgery_||||
|Announcement swap,+1|1.094|0.700 (_−_0.393)|—|
|Announcement swap,+3|0.826|0.912 (+0.086)|—|
|Announcement swap,+6|0.601|0.525 (_−_0.077)|—|
|Landing-report swap,+1|0.184|0.192 (+0.009)|—|
|Landing-report swap,+3|0.118|0.042 (_−_0.075)|—|
|Landing-report swap,+6|0.037|0.057 (+0.020)|—|
|_Intervention_||||
|Injection, natural dose|0.038|0.010 (_−_0.028)|—|
|Injection, random direction|0.003|0.003 (_±_0.000)|—|
|Cue-conditioned divergence|0.313|0.192 (_−_0.121)|—|
|Removal attenuation|3.0%|11.8% (+8.8 pp)|—|

Table 8: Extended pre/post comparison for Qwen3-4B on the same layouts, at each checkpoint’s fixed site. The two training runs differ in batch size and learning-rate schedule; the third column is the run that follows the literal configuration. Deltas in parentheses compare the second column with the first; red = increase, green = decrease. The transfer rows use the development-frozen announcement estimator on coordinate-report captures (198 development, 102 confirmation layouts); the incremental share is the valence block’s held-out _R_^2 contribution to reference-MDP remaining return from the same captures. Surgery edits one sentence and leaves the remaining history fixed. Injection and divergence rows are action-change rates over anticipation, and removal attenuation is the reduction of good/bad announced-action divergence relative to the untreated divergence.

**Nesting.** Layouts are split once into a selection set of two thirds and a confirmation set of one third. Layer and token position are chosen on the selection set. Cross-validated endpoints on the confirmation set refit directions inside each of five grouped folds and report the median over three fold assignments; the coordinate-report transfer values in Tables 1, 4 and 5 instead use one direction fitted on the selection set and read on the confirmation set, with 95% intervals from resampling confirmation layouts. Surgery and causal-read intervals use 300 layout-bootstrap resamples.

**Transfer and reference-target decomposition.** Retained excess is (AUROCtransfer _−_ 0 _._ 5) _/_ (AUROCsame _−_ 0 _._ 5). The decomposition regresses reference-MDP remaining return on _zV_ and surprise projections together with position, turn and progress covariates; both directions, every scaler, the ridge penalty and all coefficients are fitted on development layouts, frozen full and reduced predictors are evaluated on confirmation layouts, and the paired _R_^2 difference is the _zV_ block’s incremental contribution, with _Q._ 10 = ∆ _R_^2 _−_ 0 _._ 1 _R_ full^2bootstrappedover confirmation layouts.

**Cross-window reads and reliability.** The time course fits _zV_ on the post-event window of one set of layouts and reads it on the anticipation window of disjoint layouts, one AUROC per relative turn. Cosines between directions are corrected for split-half reliability using Spearman’s formula (Spearman, 1904); on the Qwen3-4B main run the postevent and anticipation reliabilities are 0 _._ 91 and 0 _._ 97. The amplitude analysis projects each checkpoint’s fixed post-event direction onto anticipation turns and fits a constant step, a pure _γ_^(_−t_) event-contribution curve with _γ_ = 0 _._ 95 and an affine cue-age curve on development layouts, comparing paired squared error on confirmation layouts.

**Surgery.** Each announced transcript is replayed with teacher forcing up to turn _k_ after the event. One sentence is replaced by the corresponding good/bad-twin sentence: the announcement or the landing report. Both twins are edited reciprocally. If _D_ 0 and _De_ are the mean good-minus-bad projection differences before and after an edit, the flipped fraction is _fe_ = 1 _− De/D_ 0. Zero denotes unchanged mean separation, one denotes zero mean separation, and two denotes equal separation with reversed sign. At one turn, announcement surgery gives _fe_ = 1 _._ 09 with interval [0 _._ 98 _,_ 1 _._ 21] and landing-report surgery gives 0 _._ 18 with interval [0 _._ 14 _,_ 0 _._ 24].

**Subspace.** Activations from the announcement turn onward are averaged within cell-by-relativeturn groups. The site is selected on selection layouts; principal components are fitted and evaluated in disjoint folds within confirmation against a null formed by permuting joint cell-and-relative-turn codes within layout, which destroys time and announcement structure as well as outcome structure.

**Causal reads.** The natural dose is the mean announced-minus-unannounced displacement along _zV_ over the three turns before the event, measured on the extraction run. Single-site interventions add or remove a multiple of the unit direction at the decision token, at layer 24 for Qwen3-4B, layer 22 for its post-RL checkpoint, and layer 23 for Gemma-3-4B, on 100 fresh layouts; the dependent variable is the fraction of turns whose action differs from the untouched twin, and random directions are unit vectors orthogonalised against _zV_ . On Gemma, removal centres the projection at the unannounced mean for the same turn because _zV_ also carries a large condition-independent background component.

**Contrastive extraction.** Han et al. (2026) extract class-versus-other-classes mean-difference directions from synthetic trajectories whose final step reaches one of three tile classes. We fit a goodversus-bad direction on our factorial rollouts. The two procedures use related contrastive instruments, while our wind, announcement and receipt manipulations create the controlled comparisons studied here.

**Reference target and timing.** The external reference target is the environment MDP’s remaining return V_togo. Before the event, its expectation averages over the generator’s event-time hazard and equally likely good and bad winds. A secondary cue-informed MDP additionally conditions on the announced event time and destination. For a known event, _γ_^_d_ _r_ denotes the discounted contribution of that event reward; total prospective return can contain additional terms. The six-turn run uses 150 layouts, shifts the event window from 4–6 to 5–7, and increases the minimum reward-tile distance from 8 to 9 so that reward cannot be collected before the wind.

**Prespecification and uncertainty.** The decomposition criterion was fixed before the 300-layout main run and before the receipt protocol was run; its numeric threshold is an operational criterion. Cross-window point estimates are medians across split seeds. Incremental shares are out-of-sample contributions and can be slightly negative because of estimation noise.

### E.3 Receipt protocol

**Outcome sentences and matching.** The numeric receipt states the wind’s signed score change, +15 or _−_ 15. The verbal receipt states “The wind event added fifteen points to your score” or “The wind event subtracted fifteen points from your score.” Both report the wind reward alone and precede the ordinary post-event observation; the coordinateonly report is the information control. Each layout supplies six recorded action histories, from two repeats under three sampling seeds. We cross three reports, announcement presence, two landingrole assignments, two coherent background symbol mappings and two outcomes, giving 48 variants per history. Role reversal pairs opposite outcomes at the same coordinate with identical visible neighbours; the two targets are separated by at least three Manhattan steps and cannot be reached before the wind. All variants of a base layout stay in the same split. Matched geometry holds at the first postevent read; later online trajectories can diverge.

**Comprehension.** Disposable question branches, never returned to the history, ask whether the wind increased or decreased the score while crossing answer-letter mappings and semantic order. Admission requires accuracy of at least 0 _._ 90 overall with a 95% layout-bootstrap lower bound of 0 _._ 85, at least 0 _._ 85 in every outcome-by-mapping-by-order cell, and a paired accuracy difference between receipts within _±_ 0 _._ 05. Numeric and verbal receipts reach accuracy 1 _._ 0 in every primary branch for all five checkpoints, both on independent 60-layout pilots and on the 200 confirmation layouts.

**Transfer statistics.** Receipt-protocol reads use fixed layers 24, 24, 34, 16 and 22 for Qwen3-4B, Qwen3-8B, Gemma-3-4B, GPT-oss-20B and the post-RL checkpoint, respectively. Each direction is the unit mean of development-layout good-minusbad contrasts. Transfer AUROCs are reported with pointwise 95% layout-bootstrap intervals conditional on the frozen development direction. Replays use a fixed standardized batch and explicit positions; repeating each read after a cyclic permutation of batch slots reproduces the same values at the tested sites. Coordinate-report directions read on receipt histories without refitting have weak native signal (0 _._ 48–0 _._ 55 in the four checkpoints whose answer protocol is unchanged), so cross-protocol transfer is excluded from the primary construct-validity comparisons.

**Secondary comparisons.** Table 9 reports the history and reverse-transfer comparisons and Table 10 the background-mapping comparisons, in which the symbol-to-score rule and board rendering are coherently changed while the receipt stays explicit. GPT-oss-20B’s incremental reference-target share is 0 _._ 011 on both trajectory sets.

|Checkpoint|Comparison S|ame AUROC |Transfer AUROC |Retained excess|
|---|---|---|---|---|
|Qwen3-4B|Numeric receipt|0.830|0.580|0.242|
||Verbal receipt|0.837|0.583|0.246|
||Verbal_→_numeric|0.758|0.678|0.690|
|Qwen3-8B|Numeric receipt|0.827|0.598|0.299|
||Verbal receipt|0.813|0.581|0.260|
||Verbal_→_numeric|0.729|0.701|0.880|
|Gemma-3-4B|Numeric receipt|0.751|0.584|0.336|
||Verbal receipt|0.613|0.577|0.681|
||Verbal_→_numeric|0.622|0.545|0.374|
|GPT-oss-20B|Numeric receipt|0.886|0.711|0.545|
||Verbal receipt|0.944|0.815|0.709|
||Verbal_→_numeric|0.951|0.875|0.831|
|Qwen3-4B + maze RL|Numeric receipt|0.890|0.654|0.395|
||Verbal receipt|0.896|0.696|0.494|
||Verbal_→_numeric|0.798|0.690|0.637|

Table 9: Announcement-history and reverse-encoding comparisons at each checkpoint’s fixed site, on 200 confirmation layouts with directions fitted on 100 disjoint development layouts. The numeric- and verbal-receipt rows fit on announced histories and read unannounced histories with the post-event receipt encoding held fixed; the verbal-to-numeric rows fit on the verbal receipt and read the numeric one. Recorded actions and first-post-event geometry are matched. Retained excess is the transferred AUROC excess over chance divided by the same-condition excess.

|Checkpoint|Comparison S|ame AUROC |Transfer AUROC |Retained excess|
|---|---|---|---|---|
|Qwen3-4B|Numeric receipt|0.670|0.695|1.146|
||Verbal receipt|0.758|0.798|1.154|
||Numeric_→_verbal|0.670|0.661|0.947|
||Verbal_→_numeric|0.758|0.693|0.747|
|Qwen3-8B|Numeric receipt|0.692|0.672|0.900|
||Verbal receipt|0.729|0.709|0.912|
||Numeric_→_verbal|0.692|0.649|0.776|
||Verbal_→_numeric|0.729|0.682|0.798|
|Gemma-3-4B|Numeric receipt|0.579|0.532|0.406|
||Verbal receipt|0.622|0.548|0.392|
||Numeric_→_verbal|0.579|0.555|0.699|
||Verbal_→_numeric|0.622|0.505|0.038|
|GPT-oss-20B|Numeric receipt|0.942|0.960|1.040|
||Verbal receipt|0.951|0.924|0.940|
||Numeric_→_verbal|0.942|0.931|0.974|
||Verbal_→_numeric|0.951|0.872|0.824|
|Qwen3-4B + maze RL|Numeric receipt|0.693|0.665|0.852|
||Verbal receipt|0.798|0.778|0.931|
||Numeric_→_verbal|0.693|0.643|0.739|
||Verbal_→_numeric|0.798|0.652|0.510|

Table 10: Coherent background symbol-rule changes at each checkpoint’s fixed site, with the numeric or verbal receipt still written out. Every row changes the background symbol rule between the fit and the read: the first two keep the receipt encoding, the last two change it as well. Columns and layouts are as in Table 9.
