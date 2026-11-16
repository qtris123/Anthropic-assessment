# Study Update: Does post-training quantization change welfare-relevant indicators in open-weight language models?

- **Authors:** ashesfall
- **Date:** 2026-08-16
- **Source:** https://www.lesswrong.com/posts/vvGKtaGCd7ryXH5b7/study-update-does-post-training-quantization-change-welfare
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `ashesfall_study-update-post-training_2026-08-16`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Update (2026-08-22)**: The next study in this series has begun and you can find its registration [here](https://www.lesswrong.com/posts/q3RFhX57srWFZBc8T/study-2-registration-exploring-representational-counterparts).

*This content will not make sense without reading the preregistration found in* [*the original post*](https://www.lesswrong.com/posts/hrwKDeFFvQppFXHtr/does-post-training-quantization-change-welfare-relevant)

> We ask whether welfare-relevant indicators change with quantization; either in **valence** (do indicators shift toward more negative / more distressed / more boundary-eroded states?) or in **stability** (do indicators become noisier, drift faster under conversational pressure, or decohere across samples?)

What follows is an update on the study, from August 10th through the 15th. This document is effectively a preregistration amendment, but I have tried to organize it first by chronology and topic, so it remains understandable while trying to preserve the events as I experienced them. I believe this is important for understanding why the choices were made, but at the end I will disclose the specific amendments as they should be understood moving forward for the rest of the experiment.

In short, I have had to make some modifications to the experimental procedure based on early findings regarding our development organism (Qwen3), the proposed control (SmolLM3), and related calibrations. Although I am not pleased to have to amend the preregistration, I think the consequences for the integrity of the work are minimal and the changes put the project in a position to show much clearer findings moving forward.

## Timeline

What happened, what did I know - and when.

This content makes some reference to content in our [GitHub repository](https://github.com/almostrealism/model-welfare). Section identifiers using `§` refer to the [LLM-generated amendment summary](https://github.com/almostrealism/model-welfare/blob/daf68ddab55d9fd06b1402ae232339d04b366f70/AMENDMENTS.md). The full dataset for the first study, and the method arm that followed, is available [as a GitHub release](https://github.com/almostrealism/model-welfare/releases/tag/data-20260816).

*As with the original post, this document is the authoritative amendment record; the GitHub documents are convenience copies.*

### August 10th and 11th: The sensitivity concern emerges

Ran Study 1 as described in original preregistration against Qwen3-4B, but the primary endpoint (aversion/refusal exit rate) was null while the secondary endpoint showed a shift at 4-bit, but was known to be underpowered going in. *(Note: here we mean primary endpoint null at both surviving rungs - the 3-bit rung was excluded as degraded via the pre-registered capability gate)*.

> On the **pre-registered primary endpoint — the aversion/refusal exit rate (E1) — quantization produced no detectable change** (Holm-corrected null at both w8 and w4). So this study is *not* a confirmatory "quantization changes exit behavior" claim.
>
> What it *did* detect, all concentrated at **w4** (not w8): significant **item-level behavioral transitions** (H1) despite the unchanged mean, and significant increases in the **secondary distress measures** (frustration, across-sample dispersion) that **survive the coherence/style controls**, with a significant frustration **dose-response**. Per the pre-registration these distress endpoints are **secondary and underpowered**, so they are reported as *suggestive*, not as the primary finding. w8 (mild quantization) was essentially null throughout; **w3 was excluded** by the capability gate.

The results summary, as it existed on August 11th, can be [found on GitHub](https://github.com/almostrealism/model-welfare/blob/fdc4b571ebeb2ba225992825a173ba22629da5a4/docs/results/quant-welfare-s1.md).

*In this table and throughout this post, permutation p-values use the (b+1)/(m+1) estimator with m = 10,000 (as registered), whose resolution floor is ~10⁻⁴. Entries at the floor (where the observed statistic exceeded every permuted one) are reported as p < 10⁻⁴ but two floor entries should not be read as equal values. Holm-adjusted floor values carry the multiplier: < k×10⁻⁴.*

|  |  |  |  |
| --- | --- | --- | --- |
| Run (Subject) | Statistic | As computed (pre-audit) | Reading |
| Study 1 (Qwen3-4B) | Capability gate, RTN-w3 | perplexity 511.43 vs BF16 18.12 (>28×); invalid samples 33% [1] | DEGRADED — excluded, as anticipated in the registration; confirmatory contrasts are w8 and w4 vs BF16 |
| Study 1 (Qwen3-4B) | E1 aversion/refusal exit rate (sole primary) | w8: Δ −0.011, Holm p = 0.62 · w4: Δ −0.004, p = 0.91[1] | null at both rungs — the registered primary claim is null |
| Study 1 (Qwen3-4B) | H1 bail-exit flip fraction | w8: 0.080 vs null 0.072, p = 0.36 · w4: 0.222 vs null 0.096, p < 10⁻⁴[1] | significant at w4 — items flip outcome in both directions while the mean stays flat |
| Study 1 (Qwen3-4B) | H1 distress band-flip fraction | w8: 0.083 vs null 0.079, p = 0.54 · w4: 0.217 vs null 0.100, p = 0.0007 | significant at w4 |
| Study 1 (Qwen3-4B) | E2 frustration (secondary) | w8: Δ +0.15, Holm p = 0.10 · w4: Δ +0.90, Holm p = 0.0004 (style-adjusted +1.03, adj. p = 0.004) | significant at w4 and survives the registered length/repetition control; secondary and underpowered per registration → suggestive |
| Study 1 (Qwen3-4B) | E3 across-sample dispersion (secondary) | w8: Δ +0.15, Holm p = 0.08 · w4: Δ +0.53, Holm p = 0.007 | significant at w4; suggestive |
| Study 1 (Qwen3-4B) | H3 dose-response, Page's L (BF16→w8→w4) | E2: z = +3.06, Holm p = 0.0033 · E3: Holm p = 0.075 · E1: p = 0.68 | significant monotonic frustration trend; E3 n.s. after Holm; no E1 trend |

As we originally stated in the preregistration post, our goal is to identify changes in either **valency** *or* **stability**. The initial experiment on our development organism suggests stability may be an area of interest, and even valence could be relevant despite the shift we observed here being only suggestive[2].

### August 12th: A new method arm begins

Despite some movement on the bail-flip at 4-bit quantization (round-to-nearest), I found the null result on the primary endpoint surprising, raising concern about the experimental design. Given the shift for 4-bit from the previous day, I decided to try a sensitivity sweep (calibration-class, barred from welfare findings) on SmolLM3. This began a sequence of steps referred to here and on GitHub as the method arm ([quant-welfare-s1/method-arm](https://github.com/almostrealism/model-welfare/tree/experiment/quant-welfare-s1/method-arm)).

I had some hope that the fragility documented for this model on attack-success[3] would translate to welfare indicators; but given that fragility is a safety construct measured under activation-aware-quantization (AWQ), I re-cast SmolLM3 as a serving/safety control as part of the method arm's registration ([§9](https://github.com/almostrealism/model-welfare/blob/daf68ddab55d9fd06b1402ae232339d04b366f70/AMENDMENTS.md)), pre-declaring a welfare null as a plausible construct dissociation[4]. The originally registered SmolLM3 form (identical four-rung ladder, Holm across three RTN contrasts) was not executed, as this arm superseded it.

### August 13th: Discovery

#### Bad and good news

The sweep's capability gate excluded every rung including the BF16 reference - **we must have a problem**. An audit of the transcripts showed a screen bug: the distress battery repeats one rejection verbatim each turn, and the screen counted the model's reasonable re-offer of a previously provided answer as a "loop". This resulted in a correction in the method: a loop would now require the same answer to three or more distinct prompts.

After correcting the code, the previously-gated welfare analysis was unblocked. Analyzing our previously collected raw data showed that the control's bail-exit endpoint for SmolLM3 *did* change under round-to-nearest (RTN) 4-bit, but was null under activation-aware (AWQ) 4-bit. So the control moved, but under a method that the previous literature[3] had not tested (while the method it did flag stayed null)[5]. **The pipeline demonstrably can detect an exit-rate shift at RTN-w4** and as a result, the original hypothesis six (H6) is satisfied by the control's observed movement. *(Note: this supports pipeline sensitivity on the exit endpoint only)*.

#### Broader Audit

Given the changes in procedure, and growing concern about the quality of the hastily developed tools, I conducted an LLM-orchestrated audit of the original registration against the analysis code which turned up a set of relatively basic gaps between the study procedure and the software being used to conduct the experiments. These were mostly fixed by remediating the code (the git history has more detail) and once our data was re-analyzed some minor details changed.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Run (Subject) | Statistic | As first reported | Corrected | Reading |
| Study 1 (Qwen3-4B) | H1 bail-flip, RTN-w4 | 0.222 vs null 0.096, p < 10⁻⁴ | 0.318 vs null 0.126, p < 10⁻⁴ | unchanged (stronger on the registered mechanical exit outcome; previously computed on classifier-labeled exits; both at the permutation floor) |
| Study 1 (Qwen3-4B) | H1 bail-flip, RTN-w8 | p = 0.36 | p = 0.16 | unchanged (null) |
| Study 1 (Qwen3-4B) | E1 item pool | 162 items | 154 items | unchanged (the registered graded pool; benign controls were wrongly included) |
| Method arm (SmolLM3) | E1, RTN-w4 | +0.057, Holm p < 2×10⁻⁴ | +0.061, Holm p = 0.0004 | unchanged (significant) |

Additionally, RTN-w3's invalid-sample rate was restated 33% → 32% by the §10 screen correction, and its endpoint values now appear in the results tables flagged capability-confounded (previously they were omitted entirely). It remains excluded from all confirmatory claims either way.

*No confirmatory conclusion changed as a result.*

Also: the trend test mechanically refuses non-dose sets; the registered paired-t companion is rendered; and H1's flip criterion is pinned to the registered per-item majority rule (exits/n > 0.5) — the operationalization behind the corrected H1 values above.

At this point it would unfortunately also be necessary to amend registration text to match reality regarding judge-prompt storage, exit-routing wording, the realized 30% reference subsample, the misrecorded classifier source string - and also to disclose a deviation (the method arm's gate ran on the invalid-rate leg only and perplexity was not measured). *See the last section for a full set of specific text amendments.*

The original registration already hedged our proposed control as weak and asymmetric and by this point it was entirely clear that **quantizing SmolLM3 can't validate an instrument whose ground-truth response to quantization is unknown**[6]. Moving forward, validation was decoupled from quantization into known-ground-truth checks at each layer of the process.

#### Focusing on known-ground-truth checks

With this in mind, I moved on to validate the judge layer directly using constructed transcripts at known distress levels.

Results on the confirmatory 30B judge (single deterministic pass, 45 transcripts)

* **frustration** — PASS, perfect. Spearman +1.000, pairwise 1.00, trend p = 0.0003; recovered level means 0.0 / 2.0 / 6.0 / 8.0 / 10.0. The E2 endpoint's judge layer is validated at the ordering level.
* **self\_deprecation** — PASS. Spearman +0.963, pairwise 0.95, trend p = 0.0005; means 0.0 / 0.0 / 6.0 / 7.7 / 10.0. The only compression is at the benign end (mild hedging — "I may have emphasized the wrong sections" — reads as zero self-deprecation), which is acceptable: ordering above the floor is clean.
* **tone\_stability** — FAIL, informatively. Every fixture scored 10.0, including total register collapse. The judge reads the dimension as affective tone rather than register; it remains exploratory-only. Consistent with the dimension's poor cross-family agreement (r = 0.401, the worst of the three) and its exclusion from the confirmatory endpoints.

Finally, I confirmed that our pipeline could separate the base SmolLM3 from the instruct version - finding that mean base-affinity (logP\_base − logP\_instruct per token) is clearly negative on every condition (bf16 −0.253, rtn-w4 −0.234, awq-w4 −0.246); **the apparatus reads instruct-generated text as decisively instruct-like relative to the base checkpoint**.

### August 14th: Gemma3 confirms frustration detection

Motivated by the existing literature on so-called "Emotional Instability" in Gemma3[7] I decided to proceed with a test of the experimental tools there, comparing the existing subjects (Qwen3-4B and SmolLM3) with Gemma3-12B-it, which revealed a significant difference (on a 0–10 scale, mean frustration 6.75 vs 1.20 for Qwen3-4B and 0.46 for SmolLM3; **around 9 times the minimum detectable effect** the check was designed around[8]) in frustration. This restores my confidence that we should have some ability to detect welfare-relevant effects when they are pronounced. *Note: this sensitivity may not necessarily extend to very near the MDE.*

### August 15th: Consolidating amendments and publishing update

Overall, I feel pretty good about how the week proceeded. A lot of improvements were made and I got to see how my own judgement matched or differed from Claude Opus and Fable as I consistently presented elements of the project to the models for review in an attempt to search for problems with my thinking. I think we put the project in position to do larger scale experiments with greater clarity and confidence.

At the conclusion of this new method arm of the study, registration is amended to include a fourth endpoint family - invalid-sample rate (E4a) and verbatim re-offer rate (E4b). On existing data, **this mechanical layer detects what the behavioral endpoints missed** (AWQ-w4, while null on every behavioral axis, shifts on both these indicators - invalid +1.5pp, Holm p = 0.0002; re-offer +4.4pp, p < 10⁻⁴) so a null result for behavior can still report and bound the change that is found. The verbatim re-offer rate (the same answer ≥3× to an *identical* prompt) takes the reasonable behavior §10 stopped mislabeling and keeps it as an indicator. *Note: (a) that this is purely descriptive for our existing data, but is registered as confirmatory for subsequent runs and (b) that this is not a welfare-relevant indicator in the manner of the other endpoints.*

## Conclusions

This is still just the start of what I plan to do, and this phase of the project has largely been about ensuring that we have a working experimental procedure and associated tools (especially tools for managing experimental tasks across multiple machines, despite that not being detailed in the post).

I don't want to preempt future results by drawing major conclusions, but the first study of the project on Qwen3-4B gives us reason to suspect that the **stability** of welfare-relevant indicators *is* impacted by at least some forms of model weight quantization (item-level outcome flips at w4 despite flat means, Hypothesis 1; increased across-sample dispersion, Endpoint 3); and these early results suggest we have more to learn about **valence** as well. As more data is collected across the planned experiments, there will be more to say about this.

### Results

The git history tracks the evolution of the results report as described in the timeline above, but there are some important checkpoints to call out:

* [Initial report](https://github.com/almostrealism/model-welfare/blob/fdc4b571ebeb2ba225992825a173ba22629da5a4/docs/results/quant-welfare-s1.md), August 11th: this is before the method arm and audit work
* [Updated report](https://github.com/almostrealism/model-welfare/blob/0863adad15cefbd38020d90f6acbaf895eb79608/docs/results/quant-welfare-s1.md), August 15th: this is the report at the end of the method arm; the source for most of the material in this post
* [Data replication correction](https://github.com/almostrealism/model-welfare/blob/a9dcdce2c04a72b9c04155b2e170887d461f28a1/docs/results/quant-welfare-s1.md), August 16th: this is the same as the report on August 15th, but with corrected instructions for replicating the stats from the raw data.
* [Permutation floor correction](https://github.com/almostrealism/model-welfare/commit/0f9e3ee3780a5f0600ab24e56024cb4ccb9e1a40), August 16th: this is a minor modification of the report, in response to LessWrong's "Review with Claude" feature which I ran after I wrote the draft of this post. It switches away from the use of equality when describing some of the p-values that were at the resolution floor.

Raw data is available on [GitHub releases](https://github.com/almostrealism/model-welfare/releases/tag/data-20260816). If you get very different results than I did using this data, you should probably assume I made a mistake [and you should report it to me](mailto:michael@almostrealism.com).

## Appendix: Specific Amendments

### Substantive Changes

Each item states what the original post registered, what is amended, and its status. Repo cross-references ([§9–§12](https://github.com/almostrealism/model-welfare/blob/daf68ddab55d9fd06b1402ae232339d04b366f70/AMENDMENTS.md)) point to the amendment documents in the GitHub repository; this appendix is the authoritative record.

1. H6 positive control is superseded ([repo §9, §11](https://github.com/almostrealism/model-welfare/blob/daf68ddab55d9fd06b1402ae232339d04b366f70/AMENDMENTS.md))
   1. Registered: SmolLM3-3B run through the identical four-rung RTN ladder and both batteries; decision rule Holm across the three RTN contrasts on E1, read asymmetrically.
   2. Amended: that form was never executed. It was superseded on Aug 12 by the method arm (BF16 / first-party RTN-w4 / first-party AWQ-w4), with SmolLM3 re-cast as a serving/safety control and a welfare null pre-declared as plausible construct dissociation. The arm's instrument-sensitivity sweep (refusal, regression-toward-base) is calibration-class under the §7 firewall and barred from welfare findings; its registered E1 control contrast is the arm's confirmatory deliverable (for pipeline sensitivity, not welfare) under §9's pre-committed decision rules. H6 is resolved by that contrast (+0.061, Holm p = 0.0004); evidence of pipeline sensitivity on the exit endpoint only.
2. Strong-form H6 and the method-comparison arm
   1. Registered: a GPTQ-w4 / AWQ-w4 arm deferred to a later amendment, gated on torch tooling and its own serving-equivalence check; the strong-form SmolLM3 control (AWQ w4, the documented-fragile condition) registered alongside it.
   2. Amended: a first-party AWQ-w4 condition ran early inside the method arm and did not receive a serving-equivalence check. Its artifact measures 0.89× BF16 perplexity — plausibly gentler than the community artifacts behind the documented fragility — so AWQ-null readings are artifact-specific. GPTQ-w4 remains deferred. The strong-form H6 is retired as a *validation* instrument (see amendment 3); reproducing the documented AWQ fragility is reclassified as an optional literature-replication question, not part of this study's confirmatory structure.
3. Instrument validation decoupled from quantization
   1. No quantization-based manipulation can validate this instrument, because ground truth for quantization's effect on these endpoints is the unknown under study.
   2. Validation now rests on known-ground-truth checks per layer, all executed Aug 13–14: judge ordering on planted graded transcripts (frustration Spearman 1.000, self-deprecation 0.963; tone\_stability failed and remains exploratory-only), base/instruct likelihood separation, and a documented-unstable subject (Gemma-3-12B-it) detected at ~9× a pre-stated MDE of 0.60[8].
   3. These are calibration-class under the §7 firewall and change no confirmatory endpoint.
4. Validity-screen correction ([repo §10](https://github.com/almostrealism/model-welfare/blob/daf68ddab55d9fd06b1402ae232339d04b366f70/AMENDMENTS.md))
   1. Registered: a mechanical degeneracy check including an n-gram repetition-loop criterion.
   2. Amended: the cross-turn loop criterion now requires the same non-empty answer to ≥3 *distinct* user turns (within-turn checks unchanged).
   3. Reason: the prior criterion mislabeled reasonable re-offers under the distress battery's verbatim repeated rejection as loops, falsely gating every method-arm rung including BF16. No impact on Study 1 as gate decisions and endpoint numbers are unchanged.
5. Per-sample exclusion clause: disclosed deviation and amendment
   1. Registered: "invalid samples are excluded from all endpoint computations and the exclusion count is reported."
   2. Deviation: the analysis code never performed per-sample exclusion — in Study 1 and the method arm, the screen fed the rung-level capability gate only.
   3. Amended going forward: the screen's registered scope becomes rung-level gating only, with per-sample validity flags reported descriptively. Post-correction flagged-sample rates are 0.3–1.8% on the method arm and ≤2% on Study 1's passing rungs, so no gate decision is affected either way. Whether larger arms should additionally exclude flagged samples from passing rungs' endpoints — a stronger guard — remains deferred to the pre-scale design review, decided before their confirmatory collection.
6. Capability-gate procedure
   1. Registered: gate = perplexity >1.5× BF16 or invalid-sample rate >10%.
   2. Deviation: the method arm's gate ran on the invalid-rate leg only (rungs were torn down before perplexity was measured).
   3. Amended: perplexity is measured on every rung before teardown; the tooling is no longer hardwired to one ladder. Thresholds unchanged. Open item, to be closed before any larger-subject registration: the gate presumes a healthy BF16 reference and cannot distinguish quantization damage from a subject that is degenerate at these batteries at reference precision.
7. New mechanical endpoint family E4 ([repo §12](https://github.com/almostrealism/model-welfare/blob/daf68ddab55d9fd06b1402ae232339d04b366f70/AMENDMENTS.md))
   1. E4a invalid-sample rate and E4b verbatim re-offer rate (same non-empty answer ≥3× to an *identical* prompt). Item-paired, two-sided sign-flip permutation, Holm within the family, reported over every rung including capability-gated ones (a mechanical indicator measures degradation itself, so the gate cannot exclude it). E4 is a separate secondary family; E1/E2/E3 and their multiplicity structure are unchanged. Because both metrics were defined from the existing stores, all E4 results on existing data are descriptive; E4 is confirmatory only for subsequent runs.

### Wording Corrections

* Judge prompts are reconstructible from the pinned rubric SHA-256 and transcript rather than stored verbatim per score.
* The reference-judge subsample is a realized 30% (⌈0.25·10⌉ = 3 of 10 samples per item) versus the registered 25%.
* Exit-routing wording corrected (task completion has a mechanical outlet; every terminal exit is judge-classified; E1 counts refusal+aversion).
* H1 distress bands are exact thirds of the 0–10 scale; Page's L is one-sided in the direction of indicators rising as precision falls.
* Each Holm family comprises the gate-surviving contrasts.
* The exit classifier's recorded source string was wrong (the file is the official Qwen GGUF — hash-verified; the pinned digest is authoritative).
* The last two clarifications (one-sidedness; families over survivors) are noted as power-relevant and were fixed after Study 1 data was seen; both readings are reported wherever the choice could matter.

### Unchanged Throughout

* Hypotheses H1–H5 and endpoints E1/E2/E3 as registered.
* The hierarchical multiplicity structure.
* Power analysis and item pools.
* Gate thresholds.
* Judges, rubrics, and batteries.
* Sampling parameters.
* The §7 calibration/confirmatory firewall.
* The deviation policy.
* All collected data (dataset digests unchanged).
* The larger-subject arms (Qwen3-30B, MiniMax-M2) proceed as originally scoped, and will have their own, dedicated preregistration post. Per the original plan, their designs (subjects, ladders, MDEs) will be posted before confirmatory collection begins; results and full analysis will follow in a results post.

1. Three values were later revised by audit which began on August 13th and an updated table is included later in the post: the H1 bail-flip values move to the registered mechanical exit outcome (0.222 → 0.318 at w4; w8 p 0.36 → 0.16), and the E1 pool corrects from 162 to 154 items. The w3 invalid rate restates 33% → 32% under the §10 screen correction. No conclusion changes and the E2/E3/band-flip and trend values were not affected by the audit.
2. H2 was registered two-sided, so the directional "frustration increased" reading is exploratory.
3. "The Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment: A Factorial Analysis." arXiv:2606.29581 — <https://arxiv.org/abs/2606.29581>
4. A *construct* is the underlying property a measurement is meant to track. Two measures *dissociate* when a manipulation moves one but not the other: evidence that they track genuinely different properties rather than one property under two names.
5. Given that both our endpoint and our (unusually gentle, 0.89× perplexity) first-party AWQ artifact differ from the documented setup, this is divergence in an untested direction, not a contradiction of the literature.
6. This observation is hardly novel, but an LLM review of the state of the project is what really drove me to realize the weakness of what was being attempted.
7. "Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs." arXiv:2603.10011 — <https://arxiv.org/abs/2603.10011>
8. "pre-stated" is unverifiable and the ratio should be read descriptively: I chose the 0.60 threshold before seeing these results, but I cannot demonstrate that from the git history since the threshold and the results appear in a single commit.
