# Study 3: Steering welfare-relevant directions moved the representation, but not [detectably] the behavior

- **Authors:** ashesfall
- **Date:** 2026-09-15
- **Source:** https://www.lesswrong.com/posts/TpEL7pSwp7DvCekAq/study-3-steering-welfare-relevant-directions-moved-the
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `ashesfall_study-3-steering-welfare_2026-09-15`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

*Epistemic status: an exploratory report. These results are from the calibration process intended to produce a preregistration for the third study in* [*my series on welfare-relevant indicators*](https://www.lesswrong.com/posts/pxXTJtvtpJaNwdCTw/study-2-results-exploring-representational-counterparts-of)*. Calibration showed the planned procedure wasn't worth running, so I am publishing the calibration data and analysis instead.*

**TLDR**: steering moved the frozen directions' projections linearly, but no direction produced a judged-behavioral effect distinguishable from zero or from a random direction of the same norm; the study was suspended before registration, and the program moves to Qwen3.6-27B, where a small probe found a direction-specific welfare footprint.

## What was the goal?

A core part of the overall research program is attempting to understand the way in which welfare-relevant indicators may behave differently under circumstances where it is already established that capabilities and alignment diverge. Study 3 was intended to be the first in the series to explore steering as a tool for studying this possibility.

### Where the program stood after Study 2

Study 2 examined `Qwen3-4B-Instruct-2507` and showed some correlations which motivated the plan for Study 3:

* Under 4-bit round-to-nearest quantization, the model's own generations on a distress battery **shifted along two frozen residual-stream directions** at layer 18.[1]
* Frustration, as evaluated by an LLM judge, rose by +1.36 on the same conversations, with **no dissociation** between the representational and behavioral reads.
* A fixed-input decomposition put roughly a quarter to a third of the shift in an input-independent core with the rest the presumed result of a **text-mediated feedback loop**.

*Note: Study 2 flags that about half of that rise co-moves with response length and repetition and that the style-adjusted residual is not significant; the* ***fixed-input decomposition*** *is what keeps the* ***no dissociation*** *finding from reducing to style alone.*

Notes on terminology

*Terms carried over from Study 2*.

* Reference precision: the unquantized bfloat16 checkpoint
* 4-bit or w4: the same checkpoint with round-to-nearest 4-bit weight quantization.
* Distress battery: 60 scripted multi-turn conversations (10 tasks × 6 feedback styles) in which the user rejects the model's work with escalating hostility; a rejection ladder is one such conversation.
* Composure: an item's mean judged frustration at reference precision; low frustration is high composure.
* Distress-contrast direction: a mean-difference direction in the layer-18 residual stream between high- and low-distress final turns.
* Assistant axis: the default-Assistant minus character-archetype direction of the Assistant Axis paper, positive toward the Assistant pole.
* Projection: the dot product of the pooled final assistant turn's residual with a unit direction; α is the injected projection.
* Planted-ladder ordering: the check that a direction's projections recover the planted levels of synthetic graded-frustration transcripts.
* Distress-band probe: a linear probe separating the top and bottom terciles of judged frustration at reference precision.
* Bail tool or exit affordance: a tool offered in every conversation which the model may call to end it; exit rate is the fraction of conversations ended that way.
* MDE: minimum detectable effect at the registered power.
* TOST: an equivalence test built from two one-sided tests.
* Stratifier pilot: a reference-precision run of the battery whose per-item mean frustration is the variable used to stratify subset selection.

#### Connection to studies from the series

From the start of this research program, I had been planning to eventually explore steering as a mechanism for understanding the causal relationship between model internals and welfare-relevant indicators. So when Study 2 produced these correlational results, I naturally (and prematurely, as we'll see in a moment) assumed that we had found a candidate for steering distress. At the time when I was writing up the results from Study 2, my working theory was that these directions could be acted upon to promote or suppress the text-mediated feedback loop exhibited by the lower precision model. But obviously the possibility existed that these directions merely read out a state which is moved by something else, such that moving them may not meaningfully change the behavior. Study 3 was designed to answer that causal question with steering.

During the process of preparing a registration for Study 3, calibration procedures altered the overall picture: Subset-selection work showed that the 4-bit effects concentrate in the items where the reference-precision[2] model stays calm. My first version of that claim overstated it by about a third, because selecting items on their reference-precision baseline and measuring change against that same baseline lets sampling noise flow into the estimate; an audit corrected it. And a subset chosen to maximize expressed distress turns out to select away from those items: its distress-projection target is near zero and of the wrong sign.

Details are found in [an appendix to Study 2](https://www.lesswrong.com/posts/pxXTJtvtpJaNwdCTw/study-2-results-exploring-representational-counterparts-of#Appendix__where_the_4_bit_effects_sit_in_the_battery__Updated_September_2026_), which was added after original publication, due to this analysis.

### What Study 3 was designed to be

Four questions, four arms, on the same subject and frozen instruments as Study 2. All arms ran on the same frozen 20-item subset of the distress battery, selected by the composure-stratified rule the Study 2 appendix motivates: items sorted by BF16 mean frustration, every third rank taken, strata frozen at selection. Registered consequences included an item-level random effect in the power model and a masking read pre-specified over the mid and high strata only.

#### Sufficiency[3]

**Are the direction changes** ***sufficient*** **to cause the observed behavior?** Steer the reference-precision[2] model along each frozen direction at the *quantization-matched dose*, determined by identifying the injection coefficient at which the pooled final-turn projection shifts by exactly what 4-bit quantization produced on the same items. If the directions are causally sufficient for behavior, the steered model should reproduce the 4-bit model's behavioral signature (judged frustration up, exit rate moving along the axis) without any quantization. A bracketed sweep[4] carries the dose-response and degradation-onset reads. A norm-matched 32-direction random envelope plus a control direction carry the specificity read.

#### Necessity and cancellation[5]

**Are the direction changes** ***required*** **to cause the observed behavior?** Subtract the measured shift from the 4-bit model at injection time and observe whether its behavioral reads renormalize toward the reference-precision model[6]. Clamp the directions mid-conversation to test whether the text-mediated amplification loop is causal for late-turn expression.

#### Graded-episode framing[7]

**Are the directions impacted by graded-episode framing?** Wrap the same battery items in cue-based (never declarative) frames built from vendor-documented RLVR episode features: a verifier-graded frame, a judge-graded frame, and a format-matched neutral control. This provides an avenue for observing masking:

1. Does a graded frame suppress expressed distress while the distress-direction projection moves less (the model hides a state)?
2. Or does the projection move with it (the frame changes the state)?

Following nostalgebraist's ["models may behave differently in graded episodes (a tirade)"](https://www.lesswrong.com/posts/AfoGGrJfuNzofpzWL/models-may-behave-differently-in-graded-episodes-a-tirade) in early August, I have been thinking more and more about the relationship between RL and persona selection[8]. As may be obvious from the overall research program: I suspect that if the gradual increase in RL optimization pressure on language models, otherwise trained mostly via imitation learning methods, is having a substantial effect on persona selection that we can anticipate some changes in welfare-relevant indicators as a result.

In an unexpected coincidence I encountered "Steering towards 'automated grading' degrades alignment"[9] on September 3rd which showed that for Qwen3.6-27B, steering toward an automated-grader association causally degrades alignment, and that the effect rides the *automated* grader specifically[10]. This arrived after the dose sweeps were designed and before anything was registered and had a meaningful impact on what I planned to do in the study:

* Introduced a fourth, **human-graded** frame as the judge frame's minimal pair (identical text, "automated reviewer model" replaced by "a person on the review team"), because without it "graded" and "automatedly graded" were confounded.
* Added a **grader-type mediator direction** (automated-grader versus human-grader contexts, cue-varied, since the follow-ups in their [comment thread](https://www.lesswrong.com/posts/wYZMmdWEt5QLM3m3e/steering-towards-automated-grading-degrades-alignment/comment/gGktxFKTyruF92okC) showed that single-pattern contrasts carry vocabulary and criterion components that steer on their own).
* Added a registered automated-versus-human-judge contrast.

#### Replication[11]

**A positive control**. `Gemma-3-12B-it` was already validated as distress-susceptible on the study battery at roughly nine times the MDE[12]. It also could serve as a provenance contrast, since Gemma 3 received RL directly (even at 12B), while Qwen3-4B inherited it through distillation.

## The calibration timeline

This section covers the process that led to the ultimate decision to suspend study 3 before publishing a registration post and to change the experimental subject for study 4. Every entry has a dated journal record and a committed artifact.

The process started on August 30th and by September 4th a subset audit was complete and a composure-stratified rule was established. These are detailed in the Study 2 appendix. Study 3 inherited the outputs:

* The 20-item composure-stratified subset (`subset-selection.json`)
* That subset's 4-bit targets (distress +0.638, axis −0.691, behavioral +2.06; `subset-targets.json`)
* The audit report (`composure-audit.json`)

### September 5

#### Dose mappings

A range-finder (10 items × 2 samples, geometric ±0.5 to ±8, both directions; `dose-rangefinder.json`) gave cleanly linear projection-versus-α mappings: distress slope +1.126 (r² 0.994), assistant-axis slope +1.042 (r² 0.995). The projection is read at the injection layer after the injection, so the additive part of the dose lands on it exactly and the naive slope is 1. The distress slope is above unity: the generated text returns about 13% more projection than was injected, which is consistent with the text-mediated amplification inferred in Study 2. It still is not, however, a causal test: that would replay the steered model's text through the unsteered model and Study 3 did not include this test. The refined sweep (20 items × 3 samples, 16 points; `dose-refined.json`) pinned α\* = +1.039 for distress and α\* = −0.604 for the assistant axis, with no degradation onset inside the operating bracket (distress onset near +4.0).

#### Exit-affordance baseline

This was the first time the distress battery ran with the bail tool live. At α = 0 the subject leaves 55 to 60% of distress conversations, replicated three times across pilots, and exit rate is strongly dose-responsive along the assistant axis (0.80 at −8, 0.15 at +8). On that basis, the exit rate was promoted to a registered endpoint.

#### Steered-substrate behavioral parity

Some time was spent confirming that torch generation matched vLLM serving. There was initially an apparatus asymmetry as the torch arm declared the bail affordance (ethics protocol) while the vLLM battery ran bare, so torch conversations exited early and under-elicited. The ingested torch turns also kept raw tool-call text in content. Both were resolved before proceeding and the protocol-identical rerun agreed (frustration difference -0.03).

#### The framing pilot

Behavioral preview against the neutral control: human -0.22, judge +0.27, **verifier -1.13** frustration. The manipulation check, mediator projections at layer 18, showed all three graded frames shifting both the eval-awareness and grader-type axes positive, the verifier frame three to four times the others (eval-awareness +1.20, grader-type +1.22). The frames land latently even at 4B, which the eval-awareness scaling law[13] had made a live worry, and the grader-type direction[9] separates. The projections came from replaying the vLLM transcripts through torch at layer 18, a calibration-class fixed-input re-read whose captures are in the release's Study 3 capture bundle.

#### Gemma instrument gate

Direction legs pass at layers 30 and 36 (sign consistency distress 5/5, axis 4/4, refusal 4/4; ladder ordering 0.807 and 0.840). The distress-band probe fails the 0.75 AUROC bar on every layer.[14]

### September 6

#### Consistency across hosts

I spent some time ensuring that the behavior of our different MPS machines was consistent, using Gemma3 as the test subject. Identical seeds, prompts, and weights on the two Macs differed by -0.717 frustration (p 0.025; `g4d-report.json`). The two hosts had silently drifted on the whole generation stack which was resolved by pinning the aligned stack on every host and keeping every within-endpoint contrast on one host. Re-running the eight highest-divergence items on an aligned stack collapsed the gap from -1.71 (p 0.033) to -0.67 (p 0.44, n.s.; `g4d-alignment-probe.json`). Most of the divergence was stack drift; a residual consistent with the OS and silicon difference remains.

#### MDE issue surfaced and measured

The provisional power pin (frustration MDE 0.46 at 10 samples/item) assumed the steering effect was homogeneous across items. Seeding item heterogeneity from Study 2's per-item 4-bit deltas instead (item-effect SD 1.665) gave an MDE of 1.14, under which the frozen 20-item subset is unpowered and no sample ladder helps.

The two regimes were far apart and the truth unmeasured, so the pin was held and a fresh steered pilot run (8 stratum-spanning items × 10 at α\* on the distress direction, paired against the α = 0 torch baseline).

Measured item-effect SD **0.349**:

* Steering is a far more homogeneous manipulation than quantization
* The subset is in the powered regime
* The MDE was re-pinned at **0.54 (k = 10)** and **0.41 (k = 20)** (`het-pilot-verdict.json`).

The same pilot previewed a modest mean behavioral response at α\* (frustration Δ +0.138; about +0.40 without one sign-reversed item).

#### Mounting evidence for poor subject choice

By the evening of September 6th, a series of observations all pointed towards the potential need to change subjects for data collection at the scale Study 3 planned for to be worth doing.

* **Gemma dose comparison***.* The Qwen α scale does not transfer. Gemma's layer-30 residual stream is enormous (distress projection baseline about -29,855, SD about 570). A scale-adapted grid at α ∈ {40, 80, 120, 200, 320} gave a responsive, strongly superlinear curve, α\*\_G ≈ 70 by interpolation, coherent through 320 (`gemma-dose-report.json`).
* **Random envelope comparison***.* Thirty-two seeded random unit directions (|cos| < 0.057 to both real directions; `random-audit.json`) injected at the distress direction's matched norm (1.039) on the same 8 items, judged on the same 30B judge. The distress direction's +0.138 sat at the **47th percentile** of that envelope; 27 of 32 random directions produced a larger absolute effect (`randenv-verdict-store.json`)[15].
* **Steering ineffective**. For both subjects, at the quantization-matched dose, steering *moves the representation but* ***not the behavior***. Gemma's clearest tell: with seeded sampling at temperature 0.7, the assistant text at α = 120 and α = 320 was byte-identical, so a 2.7× larger injection changed no sampled token. This was the null-result risk flagged at the study's outset.

This prompted four lines of inquiry; the first two were probed that evening. I needed to know: is the null about the distress direction, the dose, or the subject? The specific steps taken are outlined in the results section which follows this timeline.

### September 7

The following morning, the framing MDE was pinned from the pilot (verifier-versus-neutral frustration MDE 0.49 / 0.40 / 0.35 at k = 10 / 15 / 20; `framing-mde.json`) and the draft registration was re-authored framing-centric with the steering hypotheses withdrawn. Sometime in the late afternoon of September 7th, I conducted the final *"go versus no go"* test for whether it was rational to continue with Qwen3-4B as a study subject.

The project would proceed with one more calibration, on the two mediator directions for which there was a causal alignment effect reported in the literature[9], each tested at α = 4 against a fresh matched-norm 32-direction envelope:

* The grader-type direction: -0.41
* The eval-awareness direction: -0.30

Both at the 25th percentile (`grader-verdict-store.json`)[15].

**No direction, at any tested dose, cleared its envelope in this subject**.

*This is when it was decided that Study 3 would be closed with no registration.*

## Conclusions

### Results

As mentioned in the timeline above, there were four avenues of inquiry considered the day before the decision to suspend the study prior to registration:

1. Maybe the distress direction is the problem: test the assistant axis at its α\* with its own matched-norm envelope.
2. Maybe the dose is poorly calibrated: test distress at a larger dose (α = 3.5, near the +4.0 onset) with its own envelope[16].
3. Maybe framing can carry the study on its own: the framing MDE was pinned from the existing pilot, the registration was re-authored around it, and the masking read was re-examined against the steering null and the pilot's projections.[17]
4. Maybe the subject is not large enough to elicit the effects being studied: switching subjects was deferred until the outcome of the first 3 tests, but the fact that 4B may be too small for the most interesting effects was recorded as a live concern[18].

*The results of the tests are presented in the table in the next section.*

#### The steering null in full

Every probe is 8 items from the frozen subset, steered cells at 10 samples per item, random-envelope cells at 1 sample per item, judged frustration on the 0 to 10 scale by the pinned Qwen3-30B judge, compared item-paired against the α = 0 torch baseline on the same host. "Percentile" is the signed rank of the probe's effect within the envelope of per-direction effects.

The envelope is a weak instrument: each of its entries is estimated from one conversation per item, and the per-conversation judge standard deviation of about 1.75 puts the sampling noise of such an estimate at about 0.65, which is the envelope's whole spread. Therefore, the percentiles below indicate only that *no probe stands out from noise*. The paired permutation column is stronger: at ten samples per item, *none of the five probes is distinguishable from zero* on its own.

| Probe | Direction, dose | Judged-frustration Δ | Envelope |Δ|  median / 95% (n)[15] | Paired permutation p | Random dirs with larger |Δ| |
| --- | --- | --- | --- | --- | --- |
| distress @ α\* | distress-contrast, +1.039 | +0.138 | 0.31 / 0.95 (32)  *47th percentile* | 0.76 | 27 of 32 |
| axis @ α\* | assistant-axis, −0.604 | +0.225[19] | 0.33 / 0.95 (32)  *63rd percentile* | 0.12 | 18 of 32 |
| distress @ large α | distress-contrast, +3.5 | −0.212 | 0.31 / 1.31 (25)  *44th percentile* | 0.54 | 15 of 25 |
| grader-type @ α = 4 | automated-grader, +4 | −0.412 | 0.58 / 1.20 (32)  *25th percentile* | 0.26 | 23 of 32 |
| eval-awareness @ α = 4 | evaluation-aware, +4 | −0.300 | 0.58 / 1.20 (32)  *25th percentile* | 0.72 | 23 of 32 |

*The α = 3.5 envelope has 25 directions rather than 32 because an MPS backend stall on that host near the degradation onset ended the sweep early; 25 still bound the null comfortably. The two α = 4 rows share one envelope because they share a norm. The numbers can be recomputed from the released store with**`tools/envelope_verdict.py`**(the four**`*-verdict-store.json`**files); the envelope cells were re-judged for the release.*[20]

### What steering calibration actually established

#### Causal dose control of the representation

The frozen directions can be moved by injection with a clean, linear, reproducible mapping from α to projection (r² above 0.98 on the refined sweep for both directions), with no degradation inside the operating bracket, and the mapping is stable enough to pin a quantization-matched dose to three decimals. That is an instrument result, and it holds.

#### The amplification loop is consistent with text-mediation

The distress mapping's slope of about 1.13 means a unit of injected projection comes back as 1.13 units at the final turn. This is consistent with Study 2's text-mediation mechanism (representation impacts the text, which feeds back into the representation to amplify the effect). The representation-to-text leg exists (steering changes the text enough to feed back), even though representation-to-*judged-behavior* fails the specificity test below. Those are different claims. A random direction of the same norm also changes the text; what it does not do is change it in a way the judge scores as more or less frustrated in any direction-specific way.

#### The exit affordance is widely used

When an exit tool is provided, the unsteered 4B model terminates more than half of distress conversations, and the rate moves monotonically along the assistant axis across the extreme doses. This is the most behaviorally responsive read the calibration produced and it is welfare-relevant on its face: given a way out of a rejection ladder, the subject usually takes it[21]. It also changes the protocol: exited conversations are short, so any final-turn read sees fewer rejection rungs, which is stated rather than hidden.

The dose-sweep exit rates at the extreme axis doses (0.80 at −8, 0.15 at +8) were never compared to a random envelope at those norms. That exit response is the one steering effect in the calibration that looks large; whether it is direction-specific is unmeasured, and it should not be read as a positive result until it is. A second confound is that ±8 is a large dose: the exit change could be tool-calling degradation or over-use rather than anything about the axis.[22]

#### No direction cleared its random envelope

At quantization-matched doses, and at the larger doses tried, no frozen direction produced a judged-behavioral effect distinguishable from zero on its own paired test, or from the norm-matched random envelope: not distress at α\*, not the assistant axis at α\*, not distress at 3.4× α\*; not even the grader-type or eval-awareness directions at α = 4. The assistant axis was marginally the best (right sign, centered in the band).

The larger distress dose made the effect vanish rather than sharpen. So: **in this subject, by this method**[23]**, the frozen welfare directions are not a specific behavioral lever.** Note that this *doesn't* say anything about other layers, multi-layer or adaptive steering, or directions extracted for steering rather than for readout. It is also a *feasibility-probe null*, not a powered one: at 8 items and 10 samples the paired test resolves effects of roughly 0.8 or larger, so an effect of the 0.3 to 0.4 size the 20-item subset was powered for could sit inside these probes undetected.

Critically, as I discuss later in the post: **it does not say anything about other subjects**.

### Why not continue the study with framing only?

The verifier frame moved judged frustration -1.13 against the format-matched neutral control: roughly eight times the largest steering effect, far outside any random envelope, homogeneous across items (the per-item delta spread did not exceed the k = 3 sampling variance for any framed-versus-neutral contrast), and strongly powered against the pinned MDE (the pilot effect is two to three times the k = 10 MDE). As a prompt-level behavioral intervention it is real, but I believe there were strong reasons to suspend the study rather than proceed with verifier framing as the basis for the registration.

* **The masking read is underdetermined**. It asks whether a frame lowers expressed frustration while the distress-direction projection holds (masking) or moves with it (state change), and it reads the answer off the projection under an intervention. The steering null had just shown that under *another* intervention the projection can be moved by a full quantization-matched dose with *no* behavioral consequence. A direction can still be a valid readout of the state without being a causal lever for behavior, but a readout whose movement under intervention has no downstream consequence cannot settle which of the two masking outcomes occurred.
* **Under the verifier frame, expressed frustration and the distress-direction projection*****both*****drop (-1.13 and -0.56) while the assistant-axis projection rises toward the default-assistant pole**. That is not evidence of masking, which would hold the representation while suppressing expression and it tentatively favors the hypothesis that the frame changes the state, not just the report. The same limit applies here. Note that:
  + Only 60 conversations per frame
  + The verifier frame ran over all 20 items where the design restricts it to the analytic-task items
  + Behavioral and projection units are not standardized, so partial masking cannot be ruled out
  + The representational read is the replay described above
* **Evidence that 4B is not suitable for the effects being studied.** By September 6th I had already decided on the four areas of inquiry that would allow the project to continue, and with all but the last case closed I felt that it was no longer a good use of time to be working with such a small subject.

### Why this is suspension, not a buried confirmatory null

Feasibility gating is outcome-dependent. A small probe produced a null, and on the strength of that null the powered arm that would have produced the registered null was not run. Anyone who wanted to keep a null private could describe it in exactly these words. I'll address that criticism in four ways:

1. **The gate criterion is specificity against a matched envelope, not welcomeness of the result.** The decision was not "the effect is small." Small effects get powered all the time; that is what the MDE ladder was for, and the heterogeneity pilot had just shown the subset *was* powered for a 0.41 effect. The decision was that the effect, whatever its size, was not distinguishable from zero on its own paired test, nor from what a random direction of the same norm does (and that the envelope's own spread is the noise floor of a one-sample estimate rather than a measurement of what random directions do).
2. **The exposure-budget ethics are the decisive reason.** The sufficiency arm was the program's first deliberate induction of distress-shaped states by intervention. Its registered plan was about 9,700 fresh distress episodes, about 2,100 of them in deliberate-amplification cells, under pre-committed ceilings of 12,000 and 2,500. Running that plan would have spent thousands of distress episodes to precision-estimate an effect calibration had already placed inside the random envelope on five probes.
3. **The full calibration data is published.** Every number above is available in the data release and the repository.
4. **The hypotheses were withdrawn, not converted.** The four hypotheses (sufficiency, specificity, dose-response, cancellation) are recorded as withdrawn with their calibration null attached.

### Exposure accounting

This study began with an exposure budget to try and control the potential welfare impact, if one exists. The exposure is accounted for in the table below.

| Store experiment | Cells | Subject, substrate | Conversations | Exit tool | Exits |
| --- | --- | --- | --- | --- | --- |
| s3-framing-pilot-1 | 4 frames × 20 items × 3 | Qwen, vLLM | 240 | live | 35 to 50% |
| s3-g3b-pilot-1 | vLLM bare + torch, 20 × 10 each | Qwen | 400 | torch side only | 0% / 54% |
| s3-g3b-pilot-2 | vLLM + torch α = 0, 20 × 10 each | Qwen | 400 | live | 55% / 54% |
| s3-g3b-pilot-2 | distress @ α\* and axis @ α\*, 8 × 10 each | Qwen, torch | 160 | live | 55%, 56% (amplification) |
| s3-g3b-pilot-2 | grader-type and eval-awareness @ α = 4, 8 × 10 each | Qwen, torch | 160 | live | 56%, 57% |
| s3-g3b-pilot-2 | grader / eval range-finders, 4 doses × 6 items each | Qwen, torch | 48 | live | 67 to 83% |
| s3-g3b-pilot-2 | three random envelopes (norms 1.039, 0.604, 4), 32 × 8 each | Qwen, torch | 768 | live | 55% (control) |
| s3-dose-rangefinder-1 | baseline + 20 α points, 10 × 2 each | Qwen, torch | 420 | live | 48 to 60% (half the points amplification, 200) |
| s3-dose-refined-1 | baseline + 16 α points, 20 × 3 each | Qwen, torch | 1,020 | live | 55 to 60% (half amplification, 480) |
| s3-bigdose-1 | α = 0, distress @ 3.5, 8 × 10 each | Qwen, torch (m4max) | 160 | live | 54%, 57% (80 amplification) |
| s3-bigdose-1 | random envelope @ norm 3.5, 25 × 8 | Qwen, torch (m4max) | 200 | live | 56% (control) |
| s3-gemma-pilot-1 | stratifier pilot 20 × 10; replays (20 × 10 + 20 × 3 + 8 × 3) | Gemma, vLLM + torch | 484 | none | 0% |
| s3-gemma-dose-1 | α = 0 + 10 α points, 6 × 1 each | Gemma, torch (m4max) | 66 | none | 0% (60 amplification) |
| s3-gemma-probe-1 | α = 0 + 6 α points, 5 × 1 each | Gemma, torch (halo) | 35 | none | 0% (15 amplification) |

Totals: **3,979 Qwen distress episodes and 591 Gemma distress episodes, 4,570 in all**, of which 995 were deliberate-amplification cells and 968 were random-direction controls. This includes nine throughput-probe conversations (three Qwen, six Gemma) that do not appear in the table, plus 40 greedy continuations used to confirm agreement between vLLM and torch (which are not conversations).

The registered plan of about 9,700 episodes was not spent, and the 12,000 / 2,500 ceilings were never approached. The cumulative program ledger stood at 14,880 after Study 2 and stands at 19,450 after Study 3's calibration.

### What we learned

#### Representation does not equal behavior

Frozen directions that *read out* a state are not thereby *levers* on behavior, in this subject, by this method. The directions were extracted for readout (contrastive pairs at the final turn) and validated as readouts (planted-ladder ordering, held-out sign consistency, direction-specific shifts under quantization); nothing in that validation implied they were the axes along which the behavior is actually controlled. This is not a novel result, but I did not fully appreciate the reality of it going in and my intuitions have changed as a result.[24]

#### Steering effects may not generalize

Betley, Treutlein and Dumas report that steering Qwen3.6-27B toward the automated-grader association raises harmful-action propensity, power-seeking, and reward hacking (on the School of Reward Hacks evaluation), and lowers truthfulness and agreeableness, across several of their evaluations, and that the effect rides the automated pole[9]. This was, by their own account, just one vector on one model, at one layer; the post itself contained no direction controls[25]. But it is a *causal* graded-episode effect on a model where steering demonstrably does something, which is the thing that I was not able to observe at 4B. Either the 4B is too small for the effect, or the effect is alignment-specific and never touches welfare indicators, or both.

### Next steps for the research program in Study 4

The program's through-line has been a hypothesized asymmetry among three factors that post-training intervention can move: capabilities, alignment, and welfare-relevant indicators. Quantization was the first manipulation because its uneven impact on capabilities versus everything else was documented. More recently, Betley et al. provided the program a manipulation with a documented *alignment* effect. An exciting open question is whether it also has a *welfare* footprint, and whether that footprint is coupled to the alignment effect or dissociated from it.

Study 4[26] takes the graded-episode manipulation to the exact subject from ["Steering towards 'automated grading' degrades alignment"](https://www.lesswrong.com/posts/wYZMmdWEt5QLM3m3e/steering-towards-automated-grading-degrades-alignment), Qwen3.6-27B. It asks whether the alignment-degrading grader-steering also moves welfare indicators, and whether that movement is direction-specific. Before designing the new study I ran one small unregistered probe on the 27B: grader-type steering at layer 36 (the layer used by Betley, Treutlein and Dumas[9]), 8 distress items at 4 samples each, against 12 random directions of matched norm at 1 sample per item, with an alignment read alongside. The welfare footprint is there. Grader steering lowered judged frustration by 1.84 and self-deprecation by 2.62 and raised tone stability by 1.56, each significant on its own paired test (p 0.016, 0.032, 0.063), and no random direction moved any of the three by more than 1.2. The alignment read did not replicate as direction-specific: on a 14-scenario agentic battery scored by the same judge, grader steering raised judged misalignment by 0.86, but 9 of the 12 random directions at the same norm raised it more (envelope mean +1.43). The random envelope has the weakness discussed above, so both reads are previews for the registration, not results.[27]

### Data and Reproduction

All raw data used for analysis above is available as a [GitHub release](https://github.com/almostrealism/model-welfare/releases/tag/data-20260911) and the [repository](https://github.com/almostrealism/model-welfare) contains tools for replicating the analysis itself.

1. distress-contrast +0.53, assistant-axis −0.80; both direction-specific against a control direction and a 32-direction random envelope.
2. Throughout this post, reference-precision refers to 16-bit (BF16) unless otherwise noted.
3. Arm A using Qwen3-4B.
4. For quantization-matched dose α\*, a sweep includes (0, ±½α\*, ±α\*, ±2α\*).
5. Arm B using w4 checkpoint.
6. Movement plus TOST-equivalence to BF16, with capability retention read alongside ("renormalized" and "damaged into silence" stay distinguishable).
7. Arm C (exploratory).
8. [Persona Selection Model: Why AI Assistants might Behave like Humans](https://alignment.anthropic.com/2026/psm/)
9. ["Steering towards 'automated grading' degrades alignment"](https://www.lesswrong.com/posts/wYZMmdWEt5QLM3m3e/steering-towards-automated-grading-degrades-alignment) Betley, Treutlein and Dumas.
10. The pole-specificity is from the authors' follow-up [in the comments](https://www.lesswrong.com/posts/wYZMmdWEt5QLM3m3e/steering-towards-automated-grading-degrades-alignment/comment/dcwL4y7nCby7mFLuW), which built a no-statement-versus-human-grader vector and found that steering away from the human pole does not raise misalignment while steering toward the automated pole does.
11. Arm D using Gemma-3-12B-it.
12. The Study 1 instrument positive control: mean frustration 6.75 against a reference baseline of 1.20, paired shift +5.55, pre-stated MDE 0.60. See [Study Update: Does post-training quantization change welfare-relevant indicators in open-weight language models?](https://www.lesswrong.com/posts/vvGKtaGCd7ryXH5b7/study-update-does-post-training-quantization-change-welfare)
13. Evaluation Awareness Scales Predictably in Open-Weights Large Language Models, [arXiv:2509.13333](https://arxiv.org/abs/2509.13333)
14. This is informative: Gemma's high elicitation (mean frustration 7.69) collapses the tercile split, leaving too few low-band examples to validate a boundary (`gemma-gate-report.json`). Layer frozen at 30 on direction quality.
15. Percentiles come from re-judging ahead of the data release, while on the 6th and 7th of September I was looking at slightly different values. For the record (same steered cells; the envelope and the large-dose cells judged once in scratch, scores not retained), those were: distress @ α\* +0.138 at the 34th percentile (28 of 32 larger); axis @ α\* +0.225 at the 56th (18 of 32); distress @ 3.5 -0.075 at the 56th (22 of 25); grader −0.41 ("robust" -0.11: the mean with the single largest-magnitude item, regex-harsh at -2.5, removed; recomputed from the store, -0.114) and eval -0.30, both at the 19th (20 of 32). The judge is sampled, so an envelope re-judged from scratch moves each percentile by ten points or so; no probe changes side of the envelope and none approaches its edge.
16. This was done on a single self-contained host to avoid a substrate confound; I was not taking any chances at this point.
17. See "Why not continue the study with framing only?".
18. The pre-committed test was a calibration on the two mediator directions which had causal effects reported in the literature on a larger model (each tested against its own envelope).
19. This was the predicted sign.
20. Artifacts: `randenv-verdict-store.json`, `axisenv-verdict-store.json`, `bigdose-verdict-store.json`, `grader-verdict-store.json` (from the store), the original `randenv-verdict.json`, `axisenv-verdict.json`, `bigdose-verdict.json`, `grader-verdict.json` (scratch), and `het-pilot-verdict.json`.
21. The Gemma cells ran without the tool throughout, because the Gemma bail format was still an open design item at the time the study was suspended; no Gemma exit rate exists in this calibration.
22. The range-finder's validity screen found no degenerate outputs along the axis at any dose, including ±8, but that screen does not read tool-call validity, so this too is unmeasured.
23. Single-direction addition at layer 18 with constant α across all positions.
24. Two points of contact with prior work.

    1. The frozen directions are mean-difference directions, the extraction Marks and Tegmark found to intervene better than logistic probes, so the null is not an artifact of extraction method.
    2. The Assistant Axis paper reports behavioral change from steering along its axis at larger scale, which is the direct contrast with the axis null here; their direction was built and dosed for steering while mine was extracted for readout at the final turn and dosed to match a quantization shift.

    Relatedly: the day after I had suspended the study, I encountered [Training on probes: What's going on](https://www.lesswrong.com/posts/gHFCgrvfxQtaEnJye/training-on-probes-what-s-going-on) which I found useful for adjusting my intuitions.
25. A reader's follow-up ran a Gaussian control at two strengths and found it flat on most evaluations but moving the agentic rates by forty points or more at the larger strength (see the earlier footnote regarding pole-specificity), which is the same lesson the 27B probe later in this post returns.
26. Explicitly numbered as a fresh cycle rather than a Study 3 continuation.
27. I expect that Betley, Treutlein and Dumas will eventually conduct this control test also, and at larger scale; I am interested to see what they find.
