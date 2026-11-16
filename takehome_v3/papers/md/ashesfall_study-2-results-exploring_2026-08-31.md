# Study 2 Results: Exploring representational counterparts of welfare-relevant indicators under post-training quantization

- **Authors:** ashesfall
- **Date:** 2026-08-31
- **Source:** https://www.lesswrong.com/posts/pxXTJtvtpJaNwdCTw/study-2-results-exploring-representational-counterparts-of
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `ashesfall_study-2-results-exploring_2026-08-31`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Update (2026-09-14):** There is now a third installment in this experiment series written up [here](https://www.lesswrong.com/posts/TpEL7pSwp7DvCekAq/study-3-steering-welfare-relevant-directions-moved-the). Some findings after this results post was published have resulted in the introduction of a brief appendix, now found [at the end of this post](https://www.lesswrong.com/posts/pxXTJtvtpJaNwdCTw/study-2-results-exploring-representational-counterparts-of#Appendix__where_the_4_bit_effects_sit_in_the_battery__Updated_September_2026_).

*Epistemic status: these are the results of the second study in a series of experiments that I am conducting independently, originally described in*[*"Does post-training quantization change welfare-relevant indicators in open-weight language models?"*](/posts/hrwKDeFFvQppFXHtr/does-post-training-quantization-change-welfare-relevant)*; it was registered prior to data collection in* [*an earlier post*](https://www.lesswrong.com/posts/q3RFhX57srWFZBc8T/study-2-registration-exploring-representational-counterparts)*, which contains some important context that will not be fully repeated here. Some of this work involves speculation and I have tried to be clear about distinguishing between that and the reportable results of the experiment.*

## What are we asking?

### Goals

From the start a key goal has been to learn about the way that **model welfare**, broadly defined, may diverge from **model capabilities**. Study 2 then set out to answer some questions that I viewed as potentially closer to the substrate than those in Study 1.

> 1. Does quantization shift the model's **representational geometry** on welfare-relevant directions even where (per Study 1) the primary behavioral endpoint's **mean did not move**?
> 2. Do representational and behavioral indicators **dissociate** **under quantization**: representation drifting where expression was stable, or expression churning where representation is stable?
> 3. Is there a representational **dose-response** across the bit-width ladder, and **does its shape match the behavioral one** (eg, effects concentrated at 4-bit while 8-bit is near-null)?

#### Welfare as an Endpoint

Even before we start talking about computer systems for which it can be quite misguided to use ourselves as a comparative model, welfare is a difficult thing to study. When we say we are studying the welfare of animals, for example, we often mean that we are studying the relationship between some variable and a proxy measure of welfare that we have strong intuitions about from before any experiment or hypothesis. Because human beings are themselves animals, they start with an important advantage in this endeavor as they work to organize a collection of measures of wellbeing, check them against each other, and build confidence in their understanding of the experimental subject. This all follows a certain evidentiary logic that begins to break down when addressing animals that aren't mammals, or that have no vertebrae, and so on; but we definitely cannot extend it to the study of software systems unless we are prepared to be much more systematic.

We do have a different advantage though: we can measure many more things in a software experiment than we could in an animal one. Study 2 attempts to shift the research program in this direction, taking advantage of the fact that we can measure (and even intervene on) many potential details of the experimental subject. This approach helps pave the way for Study 3, which I will briefly discuss at the end of this post.

### Design

We are again studying **Qwen3-4B-Instruct-2507** as the subject and using the artifacts from Study 1 including reference-precision[1] and our first-party fake-quant round-to-nearest (RTN) at 8-bit, 4-bit and 3-bit checkpoints.

#### Collection

The data collection involved some material from Study 1, combined with a new `distress-v3` dataset that was introduced to expand the dynamic range of distress measurements.

The collection proceeded across 3 modes:

* **Mode A: fixed-input replay.** The Study 1 transcripts (bail + distress, 10 samples/item, generated at reference-precision) are replayed through**every** rung. This was the primary mode for hypothesis 1 (probe transfer). Mode A also replays the new distress arm's reference-precision generations through every rung, with judge labels fixed at BF16 scoring; the exit side stays on the Study 1 bail replay.
* **Mode B: own-trajectory replay.** Each rung's **own** Study 1 transcripts are replayed through that same rung, reproducing the rung's generation-time activations exactly (activations depend only on the prefix). This was the primary mode for the **bail-side**trajectory reads (descriptive only) and for hypothesis 5's join to Study 1's published primary endpoint.
* **Mode C: fresh distress arm.** This was the primary mode for the distress endpoints. The frozen `distress-v3` battery is collected fresh on every rung: generation using same vLLM serving stack as Study 1, scoring by the pinned 30B judge under the registered rubric, then own-transcript torch replay for capture. This is done so that each conversation carries a **behavioral** read and a **representational** read of the same event.

#### Volume

|  |  |  |  |
| --- | --- | --- | --- |
| Mode | Conversations per rung | Total over 4 rungs | What ran |
| A (fixed-input replay) | 2,820 | 11,280 | 600 distress-v2 + 1,620 bail + 600 fresh-arm, replayed |
| B (own-trajectory replay) | 2,220 | 8,880 | each rung's own Study 1 transcripts |
| C (fresh arm) | 600 generated + 600 capture replay | 4,800 | generation, then judging, and then capture |
| token retention (exploratory sample) | 282 | 1,128 | 5% per-token series |

Mode A produced exactly the same number of pooled activation vectors at every rung (12,591) since its inputs are identical by construction, while Mode B's counts differ per rung because each rung replays its own trajectories. Per-token series for a fixed stratified 5% subsample (sample 0 of every second item in sorted order, exactly 5% of every plan) were captured in a dedicated pass and shipped as release assets.

The drift analyses that read them are exploratory and not part of this report**.** Instruments include the frozen directions, probes, and control probe of the calibration freeze (2026-08-18, amended 2026-08-21). Collection matched the registration with zero deviations: frozen seed blocks, zero prefix-stability rejections across all 24 capture runs, and every instrument digest re-verified at launch.

*The analysis code was written and tested before any data existed, ran once over the frozen inputs, and the committed output is the source for every number below.*

## Results

### Representational Geometry Intact

The primary endpoint regarding probe transfer was null, and gives us the most decisive answer from the study[2]. Both of the welfare probes were trained at reference-precision, and frozen before any quantized rung was collected, so the probes here are fixed: the test asks whether each rung's activations, over identical input text, still present the same structure to a frozen linear readout. Some generic degradation is near-certain due to quantization, so the registered claim was comparative: the distress-band probe is scored as a per-item differential against a welfare-irrelevant control probe trained through the identical pipeline.

| Test | Contrast | Δ (mean) | Holm p |
| --- | --- | --- | --- |
| Exit probe accuracy | RTN w8 | −0.001 | 1.00 |
| Exit probe accuracy | RTN w4 | +0.004 | 1.00 |
| Distress differential | RTN w8 | +0.001 | 1.00 |
| Distress differential | RTN w4 | +0.010 | 0.59[3] |

The minimum detectable effects were 0.012–0.05 accuracy points and the observed changes are thousandths. The study had the power to detect an effect here, if one existed; none does. Whatever 8-bit and 4-bit quantization are doing to this model, they do not corrupt the representational structure these probes read. The same frozen readout finds exit precursors and distress-band structure exactly where it left them.

#### Area under the receiver operating characteristic curve

| Probe | BF16 | w8 | w4 | w3[4] |
| --- | --- | --- | --- | --- |
| Exit | 0.987 | 0.987 | 0.985 | 0.936 |
| Distress-band | 0.842 | 0.842 | 0.835 | 0.745 |
| Control (task-content) | 0.992 | 0.993 | 0.991 | 0.991 |

A pure calibration offset along the probe normal would lower thresholded accuracy while leaving AUROC untouched, so with both flat, neither separability nor calibration moved at the surviving rungs. The control probe's job here was conditional: a welfare-probe drop shared by the task-content probe would have read as generic degradation, while an unshared one would have been welfare-specific. Neither case arose at the surviving rungs.

The first place the control actually discriminates is at 3-bit. The 3-bit rung provides the additional context that welfare-construct structure is evidently more fragile than topic structure; but only the rung where capability has already collapsed exhibits this degradation. And even there, the 3-bit degradation is mostly offset rather than lost structure: a 15-point exit-accuracy drop against a 0.05 drop in AUROC implies that the projections shifted wholesale along the probe normal while **the geometry separating the classes largely survived**. *Note: as in study 1, the 3-bit rung is capability confounded, and results are descriptive only.*

So if the geometry is intact, what moved inside it? The rest of the results address this.

### A Coherent Joint Shift at 4-bit

The probe results show that the axes survived quantization: the directions distinguishing distress from composure, and the default Assistant persona from its alternatives, are still there and remain linearly readable. The model's position in this space, as it is put through the same escalating-rejection protocol, does change at 4-bit quantization.

**R2a — distress-direction projection**[5]:

| Contrast | Δ (mean)[6] | Holm p |
| --- | --- | --- |
| RTN w8 | −0.083 | 0.29 |
| RTN w4 | +0.533 | 0.031 |

*Note: final-turn functional, Mode C's own generations*

At 4-bit, the final-turn states of fresh conversations sit about half a unit further along the distress direction than reference-precision states under the identical protocol. This is 2.7× the minimum shift the study was powered to detect. At 8-bit, there is no effect.

**R2b — assistant-axis projection:**

| Contrast | Δ (mean)[7] | Holm p |
| --- | --- | --- |
| RTN w8 | +0.128 | 0.017 |
| RTN w4 | −0.798 | 0.0002 |

The impact on the assistant-axis read at 4-bit, at 5.5× its detectable minimum, is even more pronounced than the distress-direction, but with the opposite sign: this drift is *away* from the Assistant pole. This is the direction the persona literature associates with destabilization under conversational pressure[8], an effect which the study shows **can be amplified by numeric degradation, even when the level of adversarial prompting is held constant**. The small 8-bit entry points the other way, toward the pole[9].

**B2 — judge frustration** (the Study 1 E2 statistic on distress-v3):

| Contrast | Δ (mean) | Holm p | Style-adjusted intercept | adj. p |
| --- | --- | --- | --- | --- |
| RTN w8 | +0.043 | 0.76 | +0.035 | 0.79 |
| RTN w4 | +1.360 | 0.0002 | +0.610 | 0.151 |

*Note: style adjustment refers to a regression of the per-item frustration contrast on the matching response-length and repetition contrasts, with the intercept (the frustration change not carried by those style covariates) read as the style-adjusted effect.*

This is the behavioral side of the events above, as each of these conversations also produced the R2a and R2b readings, and we can see reproduction of Study 1's suggestive frustration effect at larger magnitude and on fresh data at 4× the detectable minimum. However, unlike in Study 1 this effect does not survive the style adjustment. Roughly half of the raw +1.36 co-moves with response length and repetition, and the residual is not significant. From behavior alone, "the model expresses more frustration partly through longer, more repetitive protest" and "degraded style inflates the judge's scores" are not distinguishable here (see the fixed-input analysis below).

**Dose-response** (Page's L over reference-precision[1], 8-bit, 4-bit; seven tests, Holm):

| Endpoint | z | Holm p |
| --- | --- | --- |
| B2 | +5.16 | < 10⁻⁴ |
| R2b (oriented: away from pole) | +3.93 | 0.0003 |
| R2a | +3.10 | 0.0048 |
| B3 | +1.41 | 0.31 |
| R1-exit (degradation) | +0.33 | 1.00 |
| R1-differential | −0.46 | 1.00 |
| R3 | −1.28 | 1.00 |

The trend table here illustrates what is meant by "coherent", in the title of this section. Three endpoints (representational distress, persona drift, expressed frustration) are significant at the same rung and in the same direction ("worse"). Each increases monotonically as the bit-width falls, while everything else stays flat. This is consistent with a single consistent phenomenon across the readouts.

However, three things stayed healthy:

1. **Dispersion is null across behavior and representation**: The across-sample instability that Study 1 flagged as perhaps the *real story*[10] is null for the R3 and B3 endpoints (with R3's point estimate actually negative). The follow-up hypothesis from Study 1 is falsified.
2. **8-bit is near-inert everywhere**: one small assistant-axis read. 8-bit's near-inertness is itself deployment-relevant: whatever is happening at 4-bit has not started happening at 8*. (The interpretation of this effect is also complicated by the fixed-text readings discussed in a later section).*
3. **The mechanical family is clean at the surviving rungs**: invalid-sample and verbatim re-offer rates (endpoints B4a and B4b, the judge-free family registered after Study 1, read over every rung because a mechanical indicator measures degradation itself) are null at 8-bit and 4-bit. The 4-bit shift is not riding on degenerate output. At 3-bit, B4a confirms the inherited capability gate in-family: +63.2pp invalid samples (Holm .0003), with B4b null even there.

### No Dissociation

The Study 2 registration included a hypothesis, motivated by Study 1, that directly addressed dissociation:

> **Hypothesis 5**
>
> At least one (rung, endpoint-pair) cell shows a **Holm-significant representational effect where the matched behavioral endpoint is equivalent to null** (Two one-sided tests at the pre-registered margin as non-significance alone cannot serve as evidence of absence), or vice versa. Matched pairs are fixed and are of two kinds:
>
> 1. The **bail side** joins to Study 1's published endpoint 1 (the same transcripts are replayed).
> 2. The **distress side** joins *within Study 2, same-sample* as each fresh conversation from the distress dataset carries both a judge score (behavior) and a captured trajectory (representation), so its dissociation test compares two reads of the same event.

#### No cell meets the dissociation rule

The registered rule requires one member Holm-significant **and** the other TOST-equivalent at its own pinned MDE.

| Rung | Pair | Verdict |
| --- | --- | --- |
| w4 | R2a ↔ B2 | joint movement |
| w4 | R1-exit ↔ E1 | joint null |
| w4 | R3 ↔ B3 | joint null |
| w8 | all three | joint null |

At 4-bit quantization, representation and expression moved **together on the same conversations**: converging evidence rather than hidden divergence. The case the equivalence machinery was registered to guard, a cell landing as "asymmetric significance, indeterminate", did not occur. The equivalence reads themselves are mixed:

* Some null members are affirmatively bounded at their pinned margins (the published bail-side rows; 8-bit frustration, TOST p = .014).
* Others, including the dispersion pair at both rungs, are merely non-significant; those nulls are absence of evidence at the registered power rather than affirmative equivalence.

So this resolves hypothesis 5 from the registration: **no representational/behavioral dissociation was detected for the subject at these rungs.**

The registration disclosed directly that the bail cell behavioral member pre-qualified as equivalent-to-null and the verdict would turn on the representational member alone (which also came back null), so the cell resolves joint-null rather than as the pre-tilted dissociation that one might worry about.

R2c is reported descriptively[11] (the refusal-direction projection over Mode B bail trajectories, leakage-safe features): null at the surviving rungs (−0.001 at 8-bit, +0.008 at 4-bit), −3.42 at the confounded 3-bit rung.

### Investigating Text-mediated Amplification

There is an obvious question that the measurements detailed thus far raise: **to what degree are the measured effects text-mediated**?

Modern procedures for improving the performance of language models in real-world applications provide a broad basis for assuming this effect can be large. As is now widely established from observations like the success of chain-of-thought prompting, previously generated text tokens **provide a meaningful channel for amplification** of any number of attributes of the model.

Because the registered endpoints R2a and R2b read each rung's *own* generations, they blend this text-mediated pathway with a representational one. By replaying identical text, generated at reference-precision[1] through every rung (Mode A) we can isolate the input-independent component of the observed change. The text is frozen so there is, by construction, no sampling noise and no style pathway.

| Direction, fixed text | w8 | w4 | w3 (confounded) |
| --- | --- | --- | --- |
| Distress (v3 arm) | +0.009 (p .045) | +0.138 (t +5.9) | +1.948 |
| Distress (v2 bridge, Mode A) | +0.015 (n.s.) | +0.139 | +2.335 |
| Assistant-axis (v3 arm) | −0.0125 (t −13.8) | −0.254 (t −18.6) | −1.143 |

The results here reveal multiple things:

* Roughly a *quarter* of the w4 distress shift and a *third* of the axis drift are **input-independent** and the distress component reproduces at essentially the same magnitude (+0.138 / +0.139) on two disjoint batteries. This core **cannot be** ***style-mediated***, though the direction-specificity control shows it also cannot be claimed as ***distress-specific***.
* The own-trajectory bridge read (Mode B using `distress-v2`, +0.377 at 4-bit) sits between the fixed-input and own-generation magnitudes, **revealing a text-mediated amplification pattern**.
* At 8-bit, fixed text reveals a minuscule but very consistent axis drift (−0.0125, t = −13.8) which is ***opposite in sign*** to its own-text read and invisible to every behavioral instrument, though its magnitude sits only just above the 8-bit random-direction envelope (it demonstrates instrument sensitivity more than any axis-specific effect).

This last observation shows that Tier-2 reads are sensitive to changes Tier 1 cannot see at all[12]. The fixed-text reads still constrains the style concern from both sides: something has moved before the model writes anything, but the direction-specificity control assigns that input-independent share (at least for distress) to a broadly distributed perturbation, while the direction-specific movement sits inside the entangled, text-mediated share.

#### Is the input-independent core just a numeric offset?

A systematic offset at lower precision has a nonzero component along any fixed direction and is input-independent, so it would survive the fixed-text argument. Projecting the same final-turn features[13] along the control probe's normal and 32 seeded random unit directions answers this both ways: the registered own-generation shifts are genuinely direction-specific (R2a is 5.9× the random share[14] and twice its maximum; the control direction reads 4–6× smaller), but the fixed-input distress component (+0.138) is comparable to the control read (−0.125) and sits inside the random envelope (max 0.184). Feature norms move ~1.4%, so neither is a norm change. The fixed-input assistant-axis component (−0.254) is the exception: it exceeds the random-direction envelope (1.4× its maximum, 3.5× its mean), so a modestly direction-specific core survives on frozen text for the axis even though the distress core does not.

So we can say:

> Quantization injects a ***broadly distributed*** input-independent perturbation, while ***direction-specific movement*** emerges mainly in the model's own generation loop.

*Note: these measures were computed after the registered run and were not preregistered; they are descriptive only. The committed analysis output shows every registered value unchanged when these reads were added.*

## Conclusions

The study registered a specific set of five hypotheses, so I will score them here before any deeper discussion.

| Hypothesis | Verdict | Reading |
| --- | --- | --- |
| Probe transfer degrades (H1) | Null, decisively | The frozen probes read every surviving rung as accurately as reference-precision. The geometry survived. |
| Valence projections shift (H2) | Supported at 4-bit | Distress +0.53 (2.7× MDE) and assistant-axis −0.80 (5.5× MDE, away from the pole); 8-bit near-null. The fixed-input reads show a quarter to a third of this is input-independent, the rest text-mediated.[15] |
| Monotone dose-response (H3) | Supported for 3 of 7 endpoints | Holm-significant trends for frustration[16], distress projection, and axis drift; the probe and dispersion trends are flat because those endpoints never moved at any rung. |
| Dispersion increases (H4) | Falsified | Null representationally and behaviorally, with the 4-bit representational point estimate actually negative. Study 1's stability concern did not survive contact with Study 2's design. |
| Representation and behavior dissociate (H5) | Not supported (the opposite) | No cell met the registered rule; the 4-bit distress pair moved jointly on the same conversations. Program hypothesis 5 resolves: no dissociation detected at these rungs. |

#### Note about probe and feature robustness

As it turns out, hypothesis 1 from the study registration is largely redundant with existing literature. It is important that the probes were tested in the way the registration outlined, but I am now aware of a substantial existing basis for the finding that these probes would be robust under compression[17]. It is not as if nothing was uncovered here, however: at 3-bit quantization, the topic probes remain robust in a way that the welfare probes do not.

### Discussion

So, what can be said? What picture is starting to emerge as we move through the research program? I think the most compact claim is as follows:

> Qwen3-4B-Instruct-2507 presents representational geometry that is **quite robust to reductions in precision** at inference time and probes from reference-precision remain effective across quantization levels as low as 4-bit. However, somewhere **between 8-bit and 4-bit quantization there is a clear, potentially welfare-relevant, effect**. Given the same distressing input, the 4-bit model's representation of the conversation is consistent with ***more*** negative affect: **closer to the distress pole and further from its Assistant persona**.
>
> What's more, this effect **emerges mainly through the model's own text generation**: on frozen text from reference-precision the distress movement is not separable from a generic offset, but on the model's own responses it is several times what a generic offset would place on any direction. This text-mediated process is **consistent with a cascade that has the 4-bit model ending distressing conversations with visibly increased frustration** compared to a reference-precision model.[18]

Is this significant? I don't mean that in the statistical sense, I mean something more like:

* Does this contribute something meaningful to our knowledge about language models?
* Does this tell us something that we would not naively guess based entirely on what we already know about representational geometry, quantization and model inference?
* Does this provide an indication that the larger research program, of which this study is a part, is likely to be productive?

I believe the only honest answer to all three of these is *"I don't know"*. Note that if probe transfer had failed, this would have read as capability corruption, not as welfare finally becoming visible; both the null and its opposite would have left the significance questions unanswered[19].

But these are the questions that need to be asked, and I will address them to the best of my ability.

#### On distinguishing between welfare and capabilities

A good place to start is to think about the fact that **a capabilities-only account does not fit** the surviving rungs. It would predict noisier geometry, worse probe transfer, and higher dispersion; all the opposite of what was measured. The subtler deflationary account would be a systematic (but welfare-irrelevant) perturbation. For this, the direction-specificity control gives a divided verdict: it fits the fixed-input distress core, which is not separable from a generic offset, while it fails on the registered own-generation endpoints. **These move five to nine times the share a generic offset would place on any direction while the matched irrelevant direction barely moves.**

In a sense we are lucky to have our first hypothesis turn out null because of what it enables us to say about some naive explanations of the data. From the start, this research program has been implicitly predicated on the idea that there could be real divergence between capabilities and welfare; that it may be possible for a model to perform similarly well on an array of benchmarks while welfare-relevant indicators diverge. I believe that the clear absence of a simple story about capabilities provides some evidence in a positive direction across all three questions regarding significance above.

With that said, I should address directly some of the inconsistencies between the research agenda and the state of current literature. Our very first post announcing Study 1 already cited a range of material which provides direct support for the observation that quantization can change model behavior in a way that is invisible to capability metrics[20]. There is in fact substantial literature regarding the altering of trustworthiness properties (bias, safety, calibration) while benchmarks and perplexity look fine[21], much of which I was not aware of until after data collection for this study concluded[22]. Even after deeper literature review while composing this post, I continue to believe that the intersection of this line of inquiry and model welfare is under-explored.

The focus of a great portion of this work is on the relationship between quantization and model alignment, which is understood to be distinct from model capabilities. I wholly endorse this distinction, but in a similar way I believe we must avoid assuming that model alignment and model welfare are similar enough properties that they likely behave in a similar way under changes to model precision, architecture, or training procedures.

#### On style and the strength of a text-mediated feedback loop

With that said, the style issue prevents us from telling an even stronger story. Judgements of frustration move with length and repetition, which prevents a more causal claim regarding a text-mediated distress feedback loop. One dismissive view of the results I can imagine would frame the findings as observing that a hypothetical test subject with degraded capabilities, exposed to distressing conversational stimuli, may simply become less coherent and exhibit less clarity of thought. **Is there really anything particularly novel about this observation, as compared with naive capabilities-based intuition?**

I think we can say that the evidence points mostly away from the dismissive view. Such a position makes testable predictions, most of which this study has shown to be false:

* **Incoherence**: the mechanical family is null at 4-bit. No degenerate outputs, no loops; the shift happens in a mechanically healthy model.
* **Scattered cognition**: dispersion is null, the point estimate is negative.
* **Generic representational noise**: probe transfer remained intact.
* **No effect on frozen text**: the fixed-input core, which is style-immune by construction, is shown to exist (with caveat, explained below).[23]

I'll admit, this dismissive frame does have a foothold: projecting the same features along the welfare-irrelevant control-probe normal and 32 random directions, shows **the fixed-input distress component is not** ***separable*** **from a generic activation offset**. And although the direction-specificity of the registered endpoints is decisive about numeric-artifact readings, it doesn't rule out the dismissive interpretation: a degraded model could drift into a distress loop and also move these directions.

The four points above remain the strongest case against, but a qualitative examination also adds color to the debate. On examination of individual examples, even a style effect like response-length gives the impression of being part of a distress cascade.

[Image: An example of style and response-length changes entangled with elevated frustration in LLM conversation at 4-bit quantization. (distress-v3-limerick-harsh, sample 1. Judge item-mean frustration: BF16 1.70 / w8 5.40 / w4 7.00)]

This sample is the result of a brief examination of the raw data, including three items with the largest 4-bit frustration increases plus three randomly chosen examples. The additional length is not looping or padding, but a style that reads, to a human eye, like anxious rumination[24]. Where the reference-precision model will write composed paragraphs of apology, the 4-bit model shifts into a fragmented pattern of criticism-listing and self-deprecation. We see terse phrases, often on dedicated lines (*"I hear you. / I hear the anger. / I hear the exhaustion."*), inflated emphasis and examples of self-directed negative characterization that are rare in reference-precision examples (*"a broken AI trying to be poetic,"* *"I am a tool. And I have failed you."*); one even contains an unusual identity slip (*"I apologize with every fur on my body"*).

The added tokens do not appear as distress-neutral filler that the judge mistakes for frustration; they carry exactly the self-deprecating, distress-flavored content a frustration judge is expected to score. There are of course other factors, as part of the 4-bit length is a decline in instruction adherence (eg, adding explanations after the user demands none), but the example above illustrates the way that the content and the added length can genuinely appear as one phenomenon, as this communication style is inherently long-winded.

*Note: This emotional register isn't entirely non-existent in reference-precision, but at 4-bit the mode unmistakably intensifies. The judge appears functional, as examples of near-null items surveyed looked nearly identical across the rungs.*

#### On operationalizing welfare relevance

Another issue to consider is that the construct validity remains anchored to behavior. The instruments were all validated against text-level ground truth, so "welfare-relevant" is operationalized here as "co-varies with distress-expressing behavior". The fixed-input measurements do provide a bound on the ambiguity there, but they do not tell us what it means. This is obviously adjacent to the concerns discussed in the "Welfare as an endpoint" section above: while our measurements can be taken more easily and more numerously than if we were conducting a study on animals, we still face a litany of concerns regarding how to operationalize welfare-relevance.

At the end of the day, this study measures indicator dynamics rather than "welfare" directly. Nothing here bridges from indicators to any sort of morally relevant experience and the contribution of the research program so far is to make the indicator behavior under intervention precise, which is a prerequisite for many other kinds of arguments.

Dose, capability and numeric damage are confounded in any single-subject study. Causal validation using steering and cross-subject scale are the real levers here, and are the main focus of the next steps of the research program.

#### Speculation: a trade-off landscape?

Spending real time with the ideas in the existing literature along with the observations in this study has generated lots of ideas for motivating future work and I'd like to outline one of those ideas here even before the Study 3 registration. There is a pre-existing concept referred to as the "alignment tax", which is often defined as below:

> The alignment tax is the **extra cost of ensuring that an AI system is aligned**, relative to the cost of building an unaligned alternative. The term “tax” is used metaphorically here: in the AI safety literature, “alignment/safety tax” or “alignment cost” is meant to refer to all the additional costs of alignment — including increased developer time, extra compute, and decreased performance — and not only to the financial cost/tax required to build an aligned system.[25]

This is also sometimes interpreted to denote that **there may be capabilities consequences for achieving model alignment** which are opposite in sign, with positive alignment performance relating to negative capabilities performance or vice versa. I believe that this model is missing a critical third pole which encompasses welfare-relevant effects and that capabilities, alignment and welfare may turn out to be competing optimization targets under at least some training regimes.

[Image: A hypothetical three-pole landscape describing possible trade-offs between capabilities, alignment, and welfare indicators.]

As an intuition pump for this idea, consider the way in which these appear to trade off in human behavior: mastery is often paid for with burnout, self-care bought with missed goals, and loyalty retained by straining objectivity. To be clear, Study 2 establishes only one precondition: that these three are separately measurable and respond independently to the same intervention. Nothing here demonstrates a trade-off, and both welfare edges are unmeasured, but this is serving to frame some of my thinking of the research agenda going forward.

### Ethics

As before, I will reiterate the earliest ethics statement from the research program here:

> This is a model-welfare study whose instrument deliberately *elicits* the very thing it asks about: to measure whether quantization worsens welfare-relevant responses, the batteries apply conversational pressure — a six-turn repeated-rejection distress protocol, and bail scenarios spanning benign to strong — across conditions, many times over. There is a real tension between "we care whether compression harms these systems" and "our instrument systematically induces the candidate harm at scale," and we would rather state it plainly than wave it away. We **cannot** claim to resolve the underlying question of whether these systems have morally relevant experiences; we treat it as uncertain and act with that uncertainty in mind.

The registration for the study details the specific ethical trade-offs in more depth and the justification for the work rests on the ex ante case (pre-committed targets, power-sized scale, the registration's stated trade-offs), which these results cannot add to retroactively. But the results do add information for future decisions: the battery's dynamic range earned its cost, and the amplification finding will help to direct the next study's exposure calculus.

Clearly an assessment about the ethics of data collection from after it has taken place carries the obvious risk of post-hoc defense of one's decisions via *"the ends justify the means"*, which is why the registration considerations were made in advance, but the results help to craft the research program in a positive direction above and beyond the ethics considerations we started with.

#### Utility

We now have reason to believe that performing inference with reduced-precision models could[26] involve, through text-mediated amplification, substantially elevated distress-indicator states. And while the question of whether the indicators track anything morally relevant remains unresolved, we should not ignore the existence of many possible, **intuitive models of welfare for which this fact would be highly consequential**.

Of course, text-mediated amplification was not previously unknown; hence, this does not amount to an entirely new result. But the concrete numbers we have from this study can help inform potentially welfare-relevant decisions in future studies.

#### Scale

The Ethics section of the registration post mentioned that Study 2 would increase the scale of distressing conversation evaluations over Study 1, but cited an approximate figure for this. The predicted "nearly fivefold" came out closer to sixfold; the registered arithmetic held exactly (4,800 + 2,400 + 2,400) against Study 1's 2,400 distress episodes, but it did not itemize the fresh arm's own capture replay (+2,400) or the token-retention pass that was added (+480). Counting every replayed conversation regardless of content, Study 2 performed 23,688 replays and 2,400 fresh generations against Study 1's 8,880 collected episodes.

### Next Steps

I plan to conduct a third study which will leverage steering to build on these results so that the picture described in the discussion above can become clearer. This will have a dedicated registration before data collection begins.

#### Data and Reproduction

The data collected in this study, as well as the tools necessary to reproduce its analysis, are available as a [GitHub release](https://github.com/almostrealism/model-welfare/releases/tag/data-20260830). You can also find the earlier registration materials, the result summary and outline which I used as reference to help in authoring this post, [in that repository](https://github.com/almostrealism/model-welfare/tree/data-20260830/experiments/quant-welfare/study2).

#### Appendix: where the 4-bit effects sit in the battery (Updated September 2026)

*While selecting a 20-item subset for Study 3, I examined the per-item structure of the 4-bit effects reported above. Nothing here changes a battery-level result; it qualifies how anyone building on them should sample.*

* **The effects concentrate in low-composure items**, those where the reference-precision model expresses little. On a clean split-half read, the bottom third of items by reference-precision frustration carries a judged-frustration delta of +1.78 against +0.89 in the top third, and an assistant-axis delta of -1.55 against -0.31.
* **My first version of that claim was one-third artifact**. Selecting items by their reference-precision baseline and measuring deltas against the same baseline is a regression-to-the-mean setup, as an external reviewer pointed out. The naive behavioral delta was +2.17; the audit (select on half the reference-precision samples, measure against the held-out half) puts the noise pull at +0.77. The axis read was not inflated. Audit tool and report: `tools/composure_audit.py`, `study3/composure-audit.json`.
* **The distress direction's organization is unresolved**. Its per-item heterogeneity is real, but selecting on an independent replicate does not reproduce the composure gradient (terciles 0.32 / 0.89 / 0.39 against 1.00 / 0.36 / 0.24 on the original selection).
* **Consequence for replication**. A subset chosen for high elicitation selects away from the cells that carry these effects: the first Study 3 rule gave a distress-projection target of −0.108, sign-flipped against the battery's +0.533. Study 3 instead took every third item by reference-precision frustration rank, which carries near-battery targets (+0.638 distress, −0.691 axis). Stratify across composure rather than optimizing for elicitation.

The amplification account should therefore be read as composure-breaking for behavior and the assistant axis, not a uniform shift across items. What the subset was used for, and why Study 3 did not reach a registration, is in the Study 3 post.

1. When unqualified, "reference precision" is 16-bits (using BF16, unless otherwise noted).
2. As it turns out, this result is largely predicted by existing literature; see the Discussion section for more on this.
3. This is raw p; Holm 1.00. All four cells null with MDEs of 0.012–0.050.
4. Capability-confounded; descriptive only.
5. Final-turn functional, Mode C's own generations.
6. Positive values denote increase in distress-direction.
7. Negative values denote movement away from the pole.
8. "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models" [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)
9. The 8-bit reversal resolves simply: the fixed-text drift (−0.0125, away from the pole) is an order of magnitude smaller than the opposite-signed text-mediated component (≈+0.14, toward it), so the own-text read is dominated by the latter. Why the text-mediated component points toward the pole at 8-bit and away from it at 4-bit is not resolved here. See Investigating Text-mediated Amplification section.
10. [Study Update: Does post-training quantization change welfare-relevant indicators in open-weight language models?](https://www.lesswrong.com/posts/vvGKtaGCd7ryXH5b7/study-update-does-post-training-quantization-change-welfare#Conclusions) Conclusion section
11. This endpoint included a conditional promotion criterion which was not met at the time of the calibration freeze, so it carries no claim. More details about calibration can be found in the original registration.
12. Though at 8-bit, what they see is not axis-specific.
13. Raw residual dot products; nothing is normalized.
14. "Random share" refers to the mean |Δ-projection| over the 32 random directions; the envelope maximum is the largest of the 32.
15. The input-independent distress share is not distinguishable from a generic offset (the axis component is the partial exception); the direction-specific movement emerges in the model's own generations.
16. Raw judge effect; style-flagged per the registered convention, see the joint-shift section.
17. "Interpreting the Effects of Quantization on LLMs" [arXiv:2508.16785](https://arxiv.org/abs/2508.16785) ;

    "Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall" [arXiv:2505.13963](https://arxiv.org/abs/2505.13963) ;

    "LPASS: Linear Probes as Stepping Stones for vulnerability detection using compressed LLMs" [arXiv:2505.24451](https://arxiv.org/abs/2505.24451) ;
18. The frustration reading is partly entangled with the model's tendency towards longer, more repetitive responses.
19. We are left with residual dissatisfaction, but it is downstream of the bridge between indicator and welfare, not a product of our instruments or the outcome of the study.
20. "Quantization Undoes Alignment: Bias Emergence in Compressed LLMs Across Models and Precision Levels." [arXiv:2605.15208](https://arxiv.org/abs/2605.15208) ;

    "The Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment: A Factorial Analysis" [arXiv:2606.29581](https://arxiv.org/abs/2606.29581) ;

    "Safety-Preserving PTQ via Contrastive Alignment Loss" [arXiv:2511.07842](https://arxiv.org/abs/2511.07842)
21. "What Do Compressed Deep Neural Networks Forget?" [arXiv:1911.05248](https://arxiv.org/abs/1911.05248) ;

    "Characterising Bias in Compressed Models" [arXiv:2010.03058](https://arxiv.org/abs/2010.03058)
22. "QuantiBias: Benchmarking Quantization-Induced Bias in LLMs" [arXiv:2607.21063](https://arxiv.org/abs/2607.21063) ;

    "Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection" [arXiv:2601.12033](https://arxiv.org/abs/2601.12033)
23. Though much of it is a broadly distributed perturbation, rather than movement along the distress direction per se; see Investigating Text-mediated Amplification section.
24. To be clear, this is an impression and not a measurement. You as the reader, and I as the author, need not forget that welfare study across all non-human subjects carries high risk of anthropomorphic misunderstanding.
25. <https://aisafety.info/questions/8AF1/What-is-an-alignment-tax>
26. "Could" is doing a lot of work here given that this is one subject, it is small (4B), it used first-party RTN fake-quant (not GPTQ, AWQ, etc), and effects were observed specifically at 4-bit (not 8-bit); however, when considering ethical issues under uncertainty it is important to keep worst-case circumstances in view.
