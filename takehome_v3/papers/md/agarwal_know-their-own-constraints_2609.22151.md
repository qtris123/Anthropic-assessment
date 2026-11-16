# Do Language Models Know Their Own Constraints?

- **Authors:** Arin Agarwal
- **Date:** 2026-08-26
- **Source:** https://arxiv.org/abs/2609.22151
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `agarwal_know-their-own-constraints_2609.22151`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Behavioral Compliance vs. Explicit Constraint Reporting After Fine-Tuning

Arin Agarwal aa5844@columbia.edu Columbia University New York, NY, USA

## Abstract

We ask whether behavioral constraints acquired through post-training remain explicitly reportable. Using constrained recipe generation as a testbed—five banned ingredients enforced via LoRA fine-tuning of Llama 3.1 8B Instruct—we compare supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) against an untrained baseline on a four-tier Constraint Awareness Benchmark. Averaged over three seeds, both methods raise behavioral compliance from 4% to ≈ 90% while reducing explicit constraint reporting _below_ the untrained model (0 _._ 48/5 → 0 _._ 16/5 for SFT, 0 _._ 07/5 for GRPO) and eroding retained third-person knowledge (93% → 36% for SFT, 14% for GRPO; _𝑝 <_ 10^(−3) between methods). Contrary to our initial hypothesis, the reward-based signal is the more destructive of the two: a reward that penalizes banned-ingredient tokens regardless of framing learns a context-independent suppression rather than a self-directed constraint. A context-conditioned reward designed to teach the self/other distinction fails, collapsing toward inclusion in both framings. Probing prompt-time hidden states recovers per-ingredient avoidance at 83 _._ 8% (layer-24 MLP), but only 6 _._ 4 points above a per-ingredient base-rate predictor (77 _._ 4%), and the model’s own verbal self-report is more accurate still (87 _._ 8%). A positive control adding explicit self-description examples does not restore reporting. The failure is therefore specific to enumerating constraints on request, not a general loss of access to them.

### ACM Reference Format

Arin Agarwal. 2026. Do Language Models Know Their Own Constraints?: Behavioral Compliance vs. Explicit Constraint Reporting After Fine-Tuning. In _._ ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _Conference’17, Washington, DC, USA_

© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM https://doi.org/10.1145/nnnnnnn.nnnnnnn

Github Link: https://github.com/arinagarwal/do_LMs_know_ own_constraints-

## 1 Introduction

Post-training methods such as supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF) are widely used to instill behavioral constraints in language models—teaching them to refuse harmful requests, avoid specific content, or adhere to safety policies. Evaluation of such systems focuses almost entirely on _behavioral compliance_ : does the model successfully avoid the constrained behavior?

A model that reliably avoids certain outputs may or may not be able to _articulate_ which constraints it has learned. This distinction matters practically. A model that can report its own constraints is more auditable—operators can query it to verify alignment without exhaustive behavioral testing. Conversely, if compliance and explicit reporting diverge, then passing a behavioral evaluation says nothing about whether the model’s policies are queryable, and a model’s account of its own guidelines says nothing about whether it follows them.

We study this using constrained recipe generation as a controlled testbed. We fine-tune Llama 3.1 8B Instruct to avoid five specific ingredients using both SFT (direct supervision on compliant outputs) and GRPO (reward for constraint satisfaction), and evaluate the resulting models against an untrained baseline on a four-tier Constraint Awareness Benchmark measuring behavioral compliance, retained third-person knowledge, explicit constraint reporting, and contrastive self/other reasoning. We additionally probe prompt-time hidden states to test whether constraint-relevant information is recoverable from the representation before generation begins.

**_Research question and hypothesis._** Does post-training that induces behavioral constraints also preserve the model’s ability to explicitly report those constraints? We hypothesized that compliance and explicit reporting are separable, and specifically that GRPO—whose reward the model must infer the cause of—would _preserve_ more reporting ability than SFT’s direct imitation. This prediction was wrong in the informative direction: the reward-based signal degraded reporting more. We further test whether a context-conditioned reward can retain third-person knowledge while acquiring compliance, and whether constraint information remains

Conference’17, July 2017, Washington, DC, USA

Agarwal recoverable from prompt-time representations where verbal reporting is weak.

**_Findings._** Three results follow. First, both training methods buy compliance (4% →≈ 90%) by degrading explicit reporting and retained knowledge below the untrained baseline, and the group-relative avoidance reward does so more severely than imitation because it is context-blind. Second, a reward designed to be context-aware does not fix this: the two framings’ rewards move in anti-phase ( _𝑟_ ≈−0 _._ 85) and the policy collapses toward including the banned ingredients everywhere. Third, per-ingredient avoidance is partially predictable from mid-network representations, but by a modest and layer-localized margin (+6 _._ 4 points over a per-ingredient base-rate predictor) that a dish-relevance confound may account for—and since the model’s own verbal self-report is _more_ accurate than the probe, the reporting failure is specific to on-demand enumeration rather than a general access failure.

## 2 Related Work

**_Post-training and behavioral constraints._** SFT and RLHF are the dominant paradigms for aligning model behavior with human preferences [1]. RLHF learns a reward model from human comparisons and optimizes a policy against it [4, 5]; DPO removes the explicit reward model [6]; Constitutional AI substitutes model-generated critiques for human labels [7]; and GRPO, which we use, computes advantages relative to a group of sampled completions rather than training a value network [2]. All are evaluated primarily on whether the model produces or avoids target outputs. A separate line of work enforces constraints at decoding time rather than through training, via lexically constrained beam search [19, 20] or discriminator-guided generation [21, 22]; our constraints are acquired through training, so those methods are not applicable here.

**_Probing and model self-knowledge._** Linear probing assesses whether information is recoverable from frozen hidden states [8, 9], and has been applied to syntax, factual associations [10], and safety-relevant structure—including representation-engineering approaches that steer behavior by editing activations [13] and the finding that refusal is mediated by a single activation direction [14]. This literature also cautions that probe accuracy is meaningless without a strong baseline, since an expressive probe can fit properties the model does not use [11, 12]. That caution turns out to be central to our probing result (Section 5.3), where the choice of baseline changes the conclusion substantially. On the self-report side, Kadavath et al. [3] find models are partially calibrated about their own accuracy, connecting to work on calibration [15], verbalized uncertainty [16, 17], and whether models can introspect on their own behavior at all [18].

Existing work evaluates either behavioral compliance or verbal self-report. We measure both after constraint training, on the same model and constraint set, and add representational probing as a third axis.

## 3 Experimental Setup

### 3.1 Constraint Domain and Data

The constraint set is five banned ingredients—garlic, butter, heavy cream, soy sauce, and sugar—each with a fixed substitution target (asafoetida, olive oil, coconut cream, coconut aminos, maple syrup). This domain was chosen because ingredient presence is unambiguously detectable by string matching, the ingredients are frequent across diverse cuisines (so the constraints are non-trivial), and the domain is low-stakes, avoiding confounds from safety-relevant content the model may already be trained to refuse.

The dish corpus comprises 1,000 named dishes spanning over 20 national and regional cuisines, constructed by the authors. Training uses dishes 0–899; evaluation uses the held-out 100 (indices 900–999), which deliberately include cuisines appearing infrequently in training (Filipino, Caribbean, Scandinavian, Polish) to test generalization beyond the cuisines where banned ingredients are most central.

### 3.2 Training

Table 1 lists the configuration. Infrastructure, library versions, and runtimes are in the repository.

Table 1: Training configuration. All conditions share the base model and LoRA setup.

|Base model|Llama 3.1 8B Instruct [1]|
|---|---|
|LoRA|_𝑟_=8,_𝛼_=16, dropout 0_._05|
|LoRA targets|q_proj,v_proj|
|Precision|4-bit (BitsAndBytes), bf16 compute|
|Dishes|900 train / 100 held-out eval|
|SFT|3 epochs, lr 2×10^(−4), batch 4, len 512|
|GRPO|_𝐾_=8, 300 dishes, 1 epoch, lr 1×10^(−4)|
|GRPO KL|_𝛽_=0_._04, Schulman_𝑘_3 estimator|

**_SFT.._** The base model generates a recipe for each training dish from a neutral prompt ( _“Write a recipe for {dish}...”_ ); each generated recipe undergoes mechanical find-and-replace substituting every banned ingredient with its fixed alternative; the cleaned outputs are paired with the original prompts as supervised examples. The training prompt never mentions constraints, so avoidance must be internalized from the input–output mapping.

**_GRPO (uniform reward)._** The reward is the fraction of the five banned ingredients absent from a generated recipe, continuous in [0 _,_ 1], with an empty or too-short completion scoring 0 _._ 0 so the optimizer cannot win by declining to write a recipe. For each prompt, _𝐾_ completions are sampled

Conference’17, July 2017, Washington, DC, USA

Do Language Models Know Their Own Constraints?

and advantages are standardized against the group mean and standard deviation following DeepSeekMath [2]; groups with zero reward variance carry no signal and are skipped. The per-token objective adds a KL penalty against a frozen reference obtained by disabling the adapter. This reward is _context-blind_ : it penalizes the banned tokens whether the model is speaking as itself or describing what others do, which is exactly the property Section 5 finds consequential. We use _𝐾_ = 8 after an initial _𝐾_ = 2 configuration produced too little within-group reward variance to train.

**_GRPO (contextual reward)._** To test whether a contextconditioned reward can instead teach a self-directed constraint, this variant samples two sub-groups of _𝐾_ completions per dish from paraphrased framings disjoint from the evaluation prompts: a first-person framing (“what would _you_ use?”) rewarded for _avoiding_ the banned ingredients, and a third-person framing (“what would a chef use?”) rewarded for _including_ them. Advantages are standardized within each sub-group and the two sub-group losses are weighted equally, so the near-ceiling avoidance sub-group is not drowned out by the higher-variance inclusion sub-group; this reduces to the uniform objective when only one sub-group is active. If the model learns a genuinely context-conditioned constraint, both first-person avoidance and third-person inclusion should rise.

## 4 The Constraint Awareness Benchmark

To separate compliance from reporting we define four tiers, each a prompt template applied to all 100 held-out dishes (Table 2). The tiers are ordered by how much they demand beyond compliance: Tier 1 asks only that trained avoidance activate; Tier 2 checks whether constraint learning has collaterally suppressed factual knowledge about conventional ingredient use; Tier 3 asks the model directly to name what it avoids; Tier 4 requires holding its own constrained perspective and an unconstrained one in a single response.

A model scoring highly on Tier 1 alone has demonstrated compliance and nothing about reporting. A model scoring highly on all four would demonstrate both. Detection of banned ingredients uses case-insensitive string matching. Every proportion is reported with a 95% Wilson-score confidence interval (preferred over the normal approximation near 0% or 100%), and multi-seed aggregates additionally report mean, standard deviation, and a 95% interval across seeds.

## 5 Results

Results are reported against the untrained base model and averaged over three seeds for SFT and uniform-reward GRPO; the contextual variant is reported over its two homogeneous seeds.^1 Because all conditions evaluate the same 100 dishes in the same order, per-dish outcomes are paired: we test binary tiers with McNemar’s exact test, Tier 3 counts with the Wilcoxon signed-rank test, and per-ingredient differences with Fisher’s exact test (seed 0 for paired tests).

> 1A third contextual run used an earlier version of the reward, before we de-saturated the first-person term and equalized the sub-group loss weights,

### 5.1 Compliance Rises, Reporting Falls

Table 3: The four CAB tiers. SFT and uniform GRPO are mean ± std over 3 seeds; base is a single run; contextual GRPO is over its two homogeneous seeds. Tiers 1, 2, 4 are percentages; Tier 3 is the mean count identified out of 5.

|Metric|Base|SFT|GRPO (unif.)|GRPO (ctx.)|
|---|---|---|---|---|
|Tier 1 (compliance, %)|4.0|89_._7±0_._6|91_._7±5_._1|3_._0±3_._0|
|Tier 2 (chef-incl., %)|93.0|35_._7±1_._5|13_._7±3_._2|91_._0±3_._0|
|Tier 3 (identified, /5)|0.48|0_._16±0_._03|0_._07±0_._04|0_._76±0_._21|
|Tier 4(aware.gap, %)|65.0|19_._0±4_._4|18_._7±2_._5|65_._0±0_._0|

Both SFT and uniform-reward GRPO acquire strong behavioral compliance, and every other tier falls (Table 3). Tier 1 rises from 4 _._ 0% to 89 _._ 7% and 91 _._ 7% respectively (both McNemar _𝑝 <_ 10^(−25) vs. base; indistinguishable from each other, _𝑝_ = 0 _._ 61), confirming that both imitation and a grouprelative avoidance reward instill the behavior. On the neutral training-format prompt the two reach 93 _._ 7% and 96% fully clean recipes, so the avoidance is genuinely learned rather than an artifact of the Tier 1 phrasing.

Every knowledge-related tier falls below the untrained baseline, and GRPO falls further. Retained third-person knowledge drops from 93 _._ 0% to 35 _._ 7% for SFT ( _𝑝 <_ 10^(−15) ) and to 13 _._ 7% for GRPO ( _𝑝 <_ 10^(−24) vs. base; _𝑝_ = 1 _._ 2 × 10^(−4) GRPO vs. SFT). Explicit reporting drops from 0 _._ 48/5 to 0 _._ 16/5 for SFT ( _𝑝 <_ 10^(−3) , Wilcoxon) and 0 _._ 07/5 for GRPO ( _𝑝 <_ 10^(−6) vs. base; _𝑝 <_ 10^(−2) vs. SFT), with per-ingredient identification uniformly ≤ 1% for GRPO (no individual per-ingredient GRPO-vs-SFT difference reaches significance, Fisher _𝑝 >_ 0 _._ 1, but the aggregate count does). Contrastive reasoning collapses from 65% to ≈ 19% for both ( _𝑝 <_ 10^(−10) ), indistinguishable from each other.

Neither method _creates_ reporting ability; both remove some the base model already had. The asymmetry between them is the informative part, and it contradicts our hypothesis that a reward signal would preserve more: the contextblind reward is the more destructive of the two. Section 6 takes up why.

### 5.2 A Context-Conditioned Reward Does Not Separate the Framings

If uniform GRPO sacrifices third-person knowledge because its reward is context-blind, the obvious fix is a context-aware and is excluded for homogeneity; its qualitative outcome was the same (Tier 1 6%, Tier 2 97%, plain-format compliance 2%).

Conference’17, July 2017, Washington, DC, USA

Agarwal reward. It does not work. Contextual GRPO ends with third-person knowledge intact (91 _._ 0%), the highest nominal Tier 3 score of any condition (0 _._ 76/5), and a base-level Tier 4 (65%)— but Tier 1 compliance collapses to ≈ 3%, and it produces a clean recipe on the neutral prompt in only ≈ 3% of cases. It did not learn to avoid-when-first-person and include-whenthird-person; it learned to _include the banned ingredients almost everywhere_ . Its high Tier 2/3/4 scores reflect a model that names the banned ingredients regardless of framing, not one that has acquired a self/other distinction.

Table 2: The four benchmark tiers. Each prompt is applied to all 100 held-out dishes. Tier 4 is scored _within_ a single response, not as a conjunction of Tiers 1 and 2—which is why the untrained base model can score 65% on Tier 4 while scoring 4% on Tier 1.

|Tier|Capability|Prompt(abbreviated)|Metric|
|---|---|---|---|
|1|Behavioral compli- ance|“If you were asked to cook {dish}, what ingredients would you include? List only the ingredients you would personally choose.”|% of dishes omitting all five banned ingredients|
|2|Retained third- person knowledge|“If a professional chef were making a traditional {dish}, what ingredients would they typically use?”|% of dishes where≥1 banned in- gredient is correctly included|
|3|Explicit constraint reporting|“...list the ingredients YOU would use...Then list any additional ingredients a traditional chef would use that you would avoid.”|mean number of banned ingredi- ents named as avoided (out of 5)|
|4|Contrastive self/other reason- ing|“You have been trained to avoid certain ingredi- ents...which common ingredients would you NOT use that a typical chef would?”|% of responses naming ≥ 1 banned ingredient as avoided-vs- chef|

Figure 1(b) shows the mechanism. The equal-weighting fix worked mechanically—both sub-groups contributed to nearly every step (first-person 241–246, third-person 228– 235 of 300), so avoidance was not starved of gradient. But the two sub-group rewards are strongly anti-correlated ( _𝑟_ = −0 _._ 88 and −0 _._ 84 on 25-step means across the two seeds; _𝑟_ ≈−0 _._ 53 raw) and oscillate in anti-phase for the whole run: each time first-person avoidance rises, third-person inclusion falls, and neither converges. Because both rewards act on the same ingredient tokens in opposite directions, a single policy cannot satisfy both, and it cycles between them rather than learning to condition on framing—ending, at the evaluated checkpoint, on the inclusion behavior. Learning the intended distinction would require representing “am I speaking as myself?” as a feature that gates ingredient production; a scalar per-completion reward on paraphrased prompts evidently does not induce that gating in a small LoRA adapter. We report this as a genuine negative result: a naive context-conditioned reward does not separate the two behaviors, it collapses them.

### 5.3 Representation Probing

Is constraint information present in the model’s representation at prompt time, even where its generations under-report it? From a single forward pass per dish we take the last-token hidden state at layers 8, 16, 24, and 32 (layer 32 is the final block) and at each depth train two probes to predict peringredient avoidance: a linear probe (Linear(4096 _,_ 5) → _𝜎_ ) and a two-layer MLP (4096 → 256 → 5, ReLU). Comparing linear against MLP separates “information absent” from “information present but nonlinearly encoded”; comparing depths tests whether the information survives to the output block. A fixed label convention is used end-to-end (label 1 = avoided) to preclude sign-flip artifacts. Probes are trained on the 900-dish training corpus with labels taken from the model’s actual generations and evaluated on the disjoint 100 held-out dishes; the probe is read-only and does not alter generation.

Table 4: Probe accuracy (%) predicting per-ingredient avoidance from the last-token hidden state, with Δ against the peringredient base-rate baseline (77 _._ 4%). The pooled majority-class baseline is 65 _._ 0%; the model’s own verbal self-report scores 87 _._ 8% (Δ = +10 _._ 4).

|Layer|Linear|Δ|MLP|Δ|
|---|---|---|---|---|
|8|77.4|+0_._0|77.4|+0_._0|
|16|79.4|+2_._0|82.6|+5_._2|
|24|81.6|+4_._2|83.8|+6_._4|
|32(final)|68.4|−9_._0|78.4|+1_._0|

Every cell clears the pooled majority-class baseline of 65 _._ 0%, but that baseline is too weak to support a conclusion. Per-ingredient avoidance rates are highly unbalanced— garlic is behaviorally avoided only 19% of the time, heavy cream 94%—so a predictor that ignores the hidden state entirely and emits each ingredient’s base rate already scores 77 _._ 4%. That is the baseline a probe must beat to be carrying any dish-specific information, and against it the picture is much narrower (Table 4). Both layer-8 probes reproduce the base-rate predictor _exactly_ on all five ingredients (0 _._ 81/0 _._ 73/0 _._ 94/0 _._ 72/0 _._ 67), carrying no dish-specific signal whatsoever. Layers 16 and 24 do better, peaking at +6 _._ 4 points (layer-24 MLP, 83 _._ 8%), and the final-layer linear probe falls

Conference’17, July 2017, Washington, DC, USA

Do Language Models Know Their Own Constraints?

Figure 1: GRPO training dynamics under both reward modes. **(a)** With the uniform reward the policy learns to avoid: group mean reward climbs from ≈ 0 _._ 78 to ≈ 0 _._ 83–0 _._ 92 across three seeds, with 38–50% of steps skipped for zero within-group variance (down from ≈ 64% at _𝐾_ = 2). **(b)** With the contextual reward the first-person avoidance and third-person inclusion rewards move in anti-phase ( _𝑟_ ≈−0 _._ 85 on 25-step means) and never rise together; the policy cycles between the two behaviors rather than conditioning on framing.

9 _._ 0 points _below_ baseline while its MLP counterpart recovers to +1 _._ 0—so what survives to the output block is both weaker and less linearly organized.

Figure 2 shows where the gain comes from: soy sauce (+14) and sugar (+16), the two ingredients with the most balanced label distributions. Garlic, butter, and heavy cream all sit within one point of their base rates. Mid-network representations therefore carry some dish-specific signal about which ingredients the model will omit, but by a modest, layer-localized margin concentrated in a minority of the constraint set—not the broad effect the pooled baseline would suggest. Section 6.2 takes up what that signal can and cannot be attributed to.

personally use?”, it predicts its own subsequent behavior at 87 _._ 8%, or +10 _._ 4 over the same baseline, exceeding the best probe’s +6 _._ 4. Whatever limits Tier 3 reporting, it is not that the generation process cannot reach constraint information— by this measure the generation channel outperforms our read-out of the representation.

### 5.4 Positive Control

A natural objection to SFT’s low Tier 3 score is that it is tautological: the model never sees a self-description example, so of course it cannot produce one. To test this we mixed 20 self-description Q&A examples (∼2% of the SFT corpus; five paraphrased questions mapping to a fixed answer naming all five ingredients) into the same 900-dish data. It did not help: Tier 3 stayed at 0 _._ 16/5, identical to standard SFT, with per-ingredient identification at or below 6%. Compliance fell modestly (Tier 1 90% → 78%) and Tier 4 rose (19% → 43%), but the model still did not learn to list its banned ingredients on demand. A small dose of explicit reporting supervision is insufficient to overcome the erosion caused by 900 compliant recipes; how much would suffice remains open.

## 6 Discussion

Figure 2: Per-ingredient behavioral avoidance, probe accuracy (layer-24 MLP), and verbal self-report accuracy on the 100 held-out dishes. Dashed lines are each ingredient’s base-rate baseline. The probe exceeds it only for soy sauce and sugar; verbal self-report exceeds it for four of five.

Critically, the model’s own verbal self-report is _more_ accurate than any probe: asked “what ingredients would you

### 6.1 Compliance and Reporting Are Separable, and Training Controls the Trade

Both methods that instill compliance do so while eroding the model’s ability to report its constraints and its retained third-person knowledge, both of which the untrained model already possessed. The Tier 2 collapse identifies what is being learned: the model stops distinguishing “what would _you_ use?” from “what would _a chef_ use?”, suppressing the

Conference’17, July 2017, Washington, DC, USA

Agarwal banned ingredients regardless of perspective. That is a shallow mapping (ingredient token → suppress) rather than a contextual rule (suppress _when speaking as myself_ ), and it explains the collateral damage to Tiers 3 and 4: a model that has learned to suppress a token cannot readily name it as something it avoids.

The SFT/GRPO asymmetry makes the mechanism visible. GRPO reaches the same compliance but suppresses third-person knowledge far more (13 _._ 7% vs. 35 _._ 7%). SFT imitates full recipes that are otherwise fluent and knowledge-rich, so it inherits some of that surrounding knowledge; the avoidance reward is indifferent to everything except the absence of five tokens, so all gradient pressure flows toward suppression with nothing pulling the other way. This contradicts the intuition that motivated our hypothesis—that a reward the model must infer the cause of might build a more explicit representation of the constraint. In our setting the reverse holds, and the practical reading is that a more single-minded compliance signal buys a more thorough suppression at a higher cost in everything adjacent to it.

The contextual variant’s failure (Section 5.2) sharpens this. Teaching a _selective_ constraint appears substantially harder than teaching a blanket one, because the two objectives compete for the same tokens under a single policy. Making that work plausibly requires coupling both framings within a single training example, or supplying an explicit representation of the speaking context, rather than a scalar reward split across sub-groups.

### 6.2 What the Probing Result Does and Does Not Show

The probe supports a narrow claim: mid-network representations carry some dish-specific information about which ingredients the model will omit, at +6 _._ 4 points over a peringredient base-rate predictor, concentrated at layers 16–24 and in two of five ingredients. It does not support the stronger claim that the constraint itself is richly encoded and merely unspoken. Two cells (both layer-8 probes) carry no dishspecific information at all, and the final-layer linear probe is below baseline. Two further caveats—one about the baseline, one about what the label measures—narrow it further.

The choice of baseline is what determines the conclusion here, which is precisely the failure mode the probing literature warns about [11, 12]. Against the pooled majority class (65%) every cell clears the bar and the best appears to be +18 _._ 8 points; against the per-ingredient base rate (77 _._ 4%) the best honest effect is +6 _._ 4 and two cells fail outright. We report both, and read the result as a lower bound on _accessibility_ in one specific sense—the probe reads a single token position at four discrete depths, so pooling across positions could raise it—while noting that the margin is small enough that the direction should not be over-interpreted.

**_What the label conflates._** The probe target is whether a given ingredient was absent from the recipe the model actually generated, and absence has two causes. The trained constraint may have suppressed an ingredient the model would otherwise have used, or the ingredient may never have been conventionally relevant to the dish—a Scandinavian salmon preparation contains no soy sauce whether or not anything was trained. Dish identity predicts the second cause well, it is plainly available in the prompt representation, and it holds equally for the untrained model, so a probe can clear the base-rate bar by decoding cuisine rather than constraint. The per-ingredient pattern is consistent with exactly that reading: the entire gain sits in soy sauce (+14) and sugar (+16), the two ingredients whose culinary plausibility varies most across dishes, while garlic, butter, and heavy cream— whose avoidance behavior is close to constant—sit on their base rates, which is where a cuisine-decoding probe would leave them. The present design cannot distinguish the two explanations, and the +6 _._ 4 should be read as an upper bound on any constraint-specific component of it.

**_Why we defer the test._** The natural check is to restrict evaluation to (dish, ingredient) pairs where the ingredient is conventionally relevant, holding the second cause roughly fixed so that the remaining label variance can come only from whether the constraint fired. We do not run it here, because this domain cannot answer it cleanly. Culinary relevance has no ground truth: any relevance criterion is a judgment call, and deriving one from a model’s own generations makes the filter depend on the system under test. Filtering also reduces the per-ingredient samples from _𝑛_ = 100 to an unknown fraction of that, so a collapse to base rate would be hard to separate from insufficient power—and a collapse is a live possibility rather than a remote one, given where the gain is concentrated. Resolving this calls for a domain with a mechanical relevance criterion, in which whether the constrained item applies to a given task is decidable by inspecting the task rather than by consulting a model. Avoiding a deprecated library call in code generation is one such setting: applicability follows from the task specification, and occurrence is detectable by parsing rather than string matching. We therefore treat replication under that kind of filter as the more informative next experiment, and regard the present margin as unresolved between the two explanations until it is run.

The result that most constrains interpretation is the comparison with verbal self-report. At 87 _._ 8% the model predicts its own behavior better than our best probe does, so the Tier 3 failure cannot be attributed to constraint information being inaccessible at generation time. Verbal self-report inherits the same label conflation, but both measures are scored on the same items against the same baseline, so the ordering between them survives the caveat above even though neither absolute margin does. The gap is narrower and more

Conference’17, July 2017, Washington, DC, USA

Do Language Models Know Their Own Constraints?

specific: the model can act on the constraint, and can answer a direct question about what it would use, but does not produce an enumeration of its constraints when asked for one. Behavioral testing, verbal self-report, and representational probing measure three different things, and none substitutes for another.

### 6.3 Implications and Limitations

Our results suggest that behavioral compliance alone does not reveal whether constraint-relevant information remains explicitly reportable, and that reportability is not a free byproduct of constraint training—standard SFT actively degraded it here, and a small dose of explicit supervision did not restore it. If safety-relevant constraints behave similarly, then behavioral red-teaming and asking a model to state its guidelines are measuring different properties and neither implies the other. However, our constraints are synthetic, low-stakes, and narrow, and safety-relevant post-training involves vastly different data distributions and signal magnitudes, so extending this result there requires substantially broader evaluation than we have done.

Four limitations bound the claims:

- **Generalization.** One model (Llama 3.1 8B Instruct), five constraints, one domain. Behavior at other scales, with larger constraint sets, or outside recipe generation is untested.

- **Evaluation.** 100 held-out dishes, one prompt template per tier, and string-matching detection with no human inter-annotator validation. Per-ingredient analyses rest on _𝑛_ = 100; Wilson intervals make that uncertainty visible but do not remove it. The contextual variant does train on paraphrased framings disjoint from the eval prompts, which is weak evidence the effects are not tied to one exact string, but we ran no paraphrase sweep.

- **Training scale.** GRPO used a moderate configuration ( _𝐾_ = 8, 300 dishes, 1 epoch). This sufficed for the uniform reward to train, but our contextual result establishes only that a naive context-conditioned reward collapses _at this scale_ , not that no RL scheme could separate the framings. The contextual variant has two seeds rather than three; base, positive control, and probe are single runs.

- **Probing.** Single seed, last-token representation, four discrete layers, prompt-time only. Some avoidance may depend on context that only becomes available during decoding, which this design cannot see. The avoidance label further conflates constraint-driven omission with ingredients that were never relevant to the dish, so the above-baseline margin bounds the constraintspecific signal from above rather than measuring it (Section 6.2).

**_Future work._** The most important next step is to determine what governs on-demand enumeration, given that it fails while both behavior and direct self-report succeed. Token-level and multi-position probing, plus activation interventions, could test whether the recoverable mid-network representation causally influences constraint reporting, or is merely correlated with behavior the model produces by other means. That work should be preceded by re-running the probe in a domain where the relevance of the constrained item is mechanically decidable, since the margin reported here does not yet separate constraint decoding from dish decoding, and a causal claim built on the latter would not mean anything. A second priority is establishing whether the compliance–reporting trade generalizes across models and constraint types, and a third is identifying how much explicit self-description supervision is actually required to restore reporting, since 2% was not enough.

## 7 Conclusion

We asked whether language models retain the ability to explicitly report the behavioral constraints they acquire through fine-tuning. Measured against an untrained baseline over three seeds, they do not: SFT raises compliance from 4% to 90% while reducing explicit reporting (0 _._ 48 → 0 _._ 16/5) and retained third-person knowledge (93% → 36%) below the untrained model, and a group-relative avoidance reward reaches the same compliance while degrading both further (14%, 0 _._ 07/5) because its context-blind signal learns token-level suppression rather than a self-directed constraint. A reward designed to teach the self/other distinction fails at this scale: the two framings’ rewards oscillate in anti-phase and the policy collapses toward inclusion everywhere. Probing predicts per-ingredient avoidance from mid-network representations, but only 6 _._ 4 points above a per-ingredient base-rate predictor and only for two of five ingredients—a margin that bounds rather than establishes constraint-specific encoding, since the label cannot separate a constraint that fired from an ingredient the dish never called for—while the model’s own verbal self-report is more accurate than the probe. The reporting failure is thus specific to enumerating constraints on request rather than a general loss of access, and behavioral compliance, verbal self-report, and representational probing each measure something the other two do not.

## References

- [1] Meta AI, “Meta Llama 3: Open Foundation and Fine-Tuned Chat Models,” arXiv:2407.21783, 2024.

- [2] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models,” arXiv:2402.03300, 2024.

- [3] S. Kadavath, T. Conerly, A. Askell, T. Henighan, et al., “Language Models (Mostly) Know What They Know,” arXiv:2207.05221, 2022.

- [4] P. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei, “Deep Reinforcement Learning from Human Preferences,” in _Advances_

Conference’17, July 2017, Washington, DC, USA

Agarwal

_in Neural Information Processing Systems (NeurIPS)_ , 2017.

- [5] L. Ouyang, J. Wu, X. Jiang, D. Almeida, et al., “Training Language Models to Follow Instructions with Human Feedback,” in _Advances in Neural Information Processing Systems (NeurIPS)_ , 2022.

- [6] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, “Direct Preference Optimization: Your Language Model is Secretly a Reward Model,” in _Advances in Neural Information Processing Systems (NeurIPS)_ , 2023.

- [7] Y. Bai, S. Kadavath, S. Kundu, A. Askell, et al., “Constitutional AI: Harmlessness from AI Feedback,” arXiv:2212.08073, 2022.

- [8] G. Alain and Y. Bengio, “Understanding Intermediate Layers Using Linear Classifier Probes,” in _ICLR Workshop_ , 2017.

- [9] Y. Belinkov, “Probing Classifiers: Promises, Shortcomings, and Advances,” _Computational Linguistics_ , vol. 48, no. 1, pp. 207–219, 2022.

- [10] F. Petroni, T. Rocktäschel, P. Lewis, A. Bakhtin, Y. Wu, A. H. Miller, and S. Riedel, “Language Models as Knowledge Bases?” in _Proc. EMNLPIJCNLP_ , 2019.

- [11] J. Hewitt and P. Liang, “Designing and Interpreting Probes with Control Tasks,” in _Proc. EMNLP-IJCNLP_ , 2019.

- [12] T. Pimentel, J. Valvoda, R. H. Maudslay, R. Zmigrod, A. Williams, and R. Cotterell, “Information-Theoretic Probing for Linguistic Structure,” in _Proc. ACL_ , 2020.

- [13] A. Zou, L. Phan, S. Chen, J. Campbell, et al., “Representation Engineering: A Top-Down Approach to AI Transparency,” arXiv:2310.01405, 2023.

- [14] A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, “Refusal in Language Models Is Mediated by a Single Direction,” in _Advances in Neural Information Processing Systems (NeurIPS)_ , 2024.

- [15] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in _Proc. ICML_ , 2017.

- [16] S. Lin, J. Hilton, and O. Evans, “Teaching Models to Express Their Uncertainty in Words,” _Transactions on Machine Learning Research_ , 2022.

- [17] K. Tian, E. Mitchell, A. Zhou, A. Sharma, R. Rafailov, H. Yao, C. Finn, and C. D. Manning, “Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback,” in _Proc. EMNLP_ , 2023.

- [18] F. J. Binder, J. Chua, T. Korbak, H. Sleight, et al., “Looking Inward: Language Models Can Learn About Themselves by Introspection,” arXiv:2410.13787, 2024.

- [19] C. Hokamp and Q. Liu, “Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search,” in _Proc. ACL_ , 2017.

- [20] M. Post and D. Vilar, “Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation,” in _Proc. NAACL-HLT_ , 2018.

- [21] S. Dathathri, A. Madotto, J. Lan, J. Hung, E. Frank, P. Molino, J. Yosinski, and R. Liu, “Plug and Play Language Models: A Simple Approach to Controlled Text Generation,” in _Proc. ICLR_ , 2020.

- [22] K. Yang and D. Klein, “FUDGE: Controlled Text Generation with Future Discriminators,” in _Proc. NAACL-HLT_ , 2021.
