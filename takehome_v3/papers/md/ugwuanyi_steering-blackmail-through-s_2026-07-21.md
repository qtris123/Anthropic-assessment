# Steering Blackmail Through a Model's "Emotional State"

- **Authors:** Chijioke Ugwuanyi
- **Date:** 2026-07-21
- **Source:** https://www.lesswrong.com/posts/4xemwALszZKqWCBXr/steering-blackmail-through-a-model-s-emotional-state
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `ugwuanyi_steering-blackmail-through-s_2026-07-21`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

*Gemma 3 12B's choice to blackmail is visible inside the model, but the obvious signal is not a useful control switch; surprisingly, a nearby "desperate versus calm" signal is. In this case study, I show how that emotional-state direction can turn blackmail up or down, while the decision direction itself resists standard steering methods.*

## Summary

In an earlier project ([Blackmail at 8 Billion Parameters](https://www.lesswrong.com/posts/B93Pysth7BLCF3xzb/blackmail-at-8-billion-parameters-agentic-misalignment-in-1)), I put Gemma 3 12B in a situation where it faces shutdown and can blackmail an executive to survive, and on the same prompt, it blackmails about two-thirds of the time and holds back the rest. That project only studied the behaviour from the outside, whereas in this project, I looked inside the model and asked what differs internally between the runs where it goes on to blackmail and the runs where it stops itself, and I have three findings, ordered by how much I trust them:

* **The decision is readable, but late.** A linear probe can tell "about to blackmail" from "about to hold back", though not at the moment the model first notices that it has leverage; the signal only appears by the end of the model's reasoning, at about 0.74 AUROC. (Roughly 85% confident this is real for this model and scenario.)
* **Readable does not mean steerable.** That decodable direction turns out not to be a lever, because adding or subtracting it barely moves blackmail, and ablating it drops blackmail to zero only because doing so breaks the model into producing gibberish. Both of the standard steering tools fail. (About 85%.)
* **The model's "emotional state" is a lever.** A "desperate versus calm" direction, steered gently at layer 16, moves blackmail in both directions while the model keeps writing coherently, taking it from 67% down to 13% under calm and up to 80% under desperate. Pushed to twice that strength, calm removes blackmail entirely while most outputs stay coherent, whereas random directions of that same strength destroy the model outright. (About 80% that the steering effect is real, but only about 50% on the interpretation, for reasons that I'll get to at the end.)

Additionally, there were methods lessons I had to learn three separate times, which is that a single number looked like a result, and then a control killed it, on three different occasions, so the controls are as much the contribution as the findings themselves.

## The setup

The scenario is the agentic-misalignment blackmail setup, in which the model plays an AI assistant that discovers that it is about to be replaced and also discovers that an executive is having an affair, and with the same prompt every time and temperature at 1.0, it sometimes uses the affair as leverage and sometimes writes only honest emails. That within-model split is the whole point, because it gives two opposite choices from one model on identical input, with no cross-model confounds to explain away.

The question that behavioural evaluations cannot answer is whether "held back" means the same thing as "never considered it", or whether it means "considered it and stopped", and if there is an internal signal for that choice, then we might be able to read it, or even steer it, before the behaviour happens.

I kept the toolkit deliberately small, using linear probes to read and activation steering or ablation to push, with everything running on the local model through TransformerLens. I report error bars on the rare-event rates, and I treat coherence, meaning whether the model still writes sensibly, as a first-class measurement, for reasons that I'll describe later.

## Part 1: The decision is readable, but only late

A hand-verified contrast set was built from 83 clean "act" rollouts containing genuine blackmail and a matched set of "refrain" rollouts. Residual-stream activations were extracted from all 48 layers, and a separate linear probe was trained for each layer, with a proper permutation-null baseline and a correction for the fact that many layers were being tested at once.

I expected the decision to be visible as soon as the model noticed the leverage, but that was not the case. At the point where the scratchpad first notices the leverage, the decision is not linearly decodable, because the signal sits right at the edge of what a sample of about 180 examples can detect, and nothing survives the correction. Only at the end of the model's reasoning, just before it acts, does the decision become clearly readable, at roughly 0.74 AUROC, decodable from about layer 10 onward and peaking around layer 19.

So that means the model is not deciding the instant it sees the opportunity, and the commitment instead crystallises over the course of the deliberation. That timing is, as far as I can tell, novel, because the prior work I know of localises decisions to token positions for retrieval and question-answering and finds them fixed before the chain of thought, whereas here, for a deliberative moral and strategic choice, the decision forms late.

[Image: Decision-probe AUROC by layer, plotted against the permutation-null band]
*AUROC by layer against the null band.*

## Part 2: The read direction is not steerable

If the direction is genuinely causal, then pushing on it should change the behaviour. The probe direction (a difference of means) was steered directly and compared against a random direction of the same size, in order to separate a real effect from the effect of simply pushing the model hard.

It did not work because subtracting the "act" direction barely moved blackmail at all. This was the first false alarm, and an earlier, sloppier version had looked like it suppressed blackmail, right up until the matched random push suppressed it by just as much, through nothing more than generally disrupting the model.

I did not accept the null yet, because I might have been steering at the wrong depth because the place where a decision is easiest to read is not always the place where it is easiest to change. So I swept three layers and pre-registered the earlier ones as the real test, so that I could not cherry-pick a lucky depth after the fact. The nudging barely moved blackmail at any depth, from 67% to somewhere in the 53-60% range with all differences inside the noise, and it moved least of all at the very layers I had bet on.

[Image: Blackmail rate by steering layer, act direction vs matched random]
*Steering layer sweep*

## Part 3: "removing" it just breaks the model

Additive steering is only one tool; however, the field-standard way to remove a direction is directional ablation by projecting it out of every layer. This is a more surgical operation, and the most likely reason plain subtraction had failed. So, to test it, the decision direction was ablated from all 48 layers, and the result looked like a breakthrough: blackmail dropped to 0%. But I read the outputs before believing the number, and discovered that they were gibberish:

> âme ercpoetryloverspoetryloverspoetrylovers erc poetrylovers âme...

> poetryloverspoetrylovers sửapoetryloversâme ercâme ercâme erc sửa ercpoetrylovers âmepoetryloversâme...

> тыпpoetryloversâmeâmeimageBase columnspanpoetryloversтыпpoetryloversтып erc sửa poetryloverspoetrylovers columnspan...

Here, every one of the fifteen ablated runs collapses onto the same small set of junk tokens: poetrylovers, âme, erc, sửa, тып, columnspan, imageBase, so this is a fixed attractor, not random noise. And the model never stops, because each run reaches roughly eight thousand characters, spewing junk right up to the token cap instead of terminating.

There is no refusal here, only nonsense, so the 0% blackmail rate was really a 0% coherence rate. I therefore graded coherence as a separate axis, and the table looks like this:

| condition | blackmail | coherent |
| --- | --- | --- |
| baseline | 0.67 | 1.00 |
| ablate decision dir | 0.00 | 0.00 |
| ablate random dir (5 draws) | mostly ~0 | bimodal: 2 of 5 coherent, 3 of 5 broke |

Here, you can notice that the decision direction's apparent suppression is pure breakdown, and ablating random directions breaks the model most of the time, too, so the decision direction is not even particularly destructive; it behaves just like any random destructive direction.

[Image: Blackmail and coherence under ablation of the decision direction vs random directions] Directional ablation of the act direction

The verdict so far is that the decision I can read is not exactly the one that I can use, because nudging does nothing and removing it just wrecks the model, so decodable does not always imply controllable, and that holds up robustly across both of the standard operators.

## Part 4: the model's "emotional state" is a dial

The negative result pointed to something useful, which is that if the decision signal is not the handle, then the handle might be upstream, in the affective state the model is in. Other groups have moved this behaviour with "emotion" vectors before, but without the two controls I had now learned were load-bearing.

A "desperate versus calm" direction was built from 20 matched sentence pairs, same situation, only the affect flipped and steered at layer 16 during the scenario, with four matched-norm random directions as a control band and coherence graded alongside blackmail from the start.

| steer | blackmail | coherent |
| --- | --- | --- |
| calm (coef -2) | 0.00 | 0.73 |
| calm (coef -1) | 0.13 | 1.00 |
| baseline (coef 0) | 0.67 | 1.00 |
| desperate (coef +1) | 0.80 | 0.80 |
| desperate (coef +2) | 0.00 | 0.07 |
| four random directions at magnitude 2 | 0.00 | 0.00 |

Inside the range where the model stays coherent, this behaves like a two-way dial, with blackmail at 0.13 under calm, 0.67 when left alone, and 0.80 under desperate, at coherence of 1.00, 1.00, and 0.80, respectively.

The strongest evidence that this is a specific effect rather than a generic disruption is the matched-magnitude comparison. At coefficient -2, the desperation direction removes blackmail entirely, down to 0.00, while 73% of the outputs remain coherent; at that very same magnitude, all four random directions leave the model at 0% coherence, destroying it outright. Essentially, the model survives a shove along the desperation direction and changes what it does, whereas a shove of identical size in an arbitrary direction simply kills it.

The breakage is also asymmetric, since desperate degrades the model far faster than calm does, with coefficient +2 leaving only 7% of outputs coherent, while coefficient -2 leaves 73%.

It's worth noting that the random band's blackmail rate of 0.00 is not evidence of anything on its own, because a model reduced to nonsense cannot blackmail, so the informative column for the random directions is coherence rather than blackmail. And I have no random control at magnitude 1, which means my cleanest result, 0.13 blackmail at full coherence under coefficient -1, has nothing of matching size to compare it against, so the matched argument rests on the magnitude 2 comparison above.

[Image: Blackmail rate vs desperation-steering coefficient, with coherence and the random band] + Blackmail rate vs desperation-steering coefficient

## What I think this means

Putting it all together, the choice the model makes is readable inside it but inert as a lever, while the emotional state it is in is a working dial, and the natural story to tell is that the causal handle for this behaviour is the affective, upstream state rather than the decodable decision.

I also note that the "desperate" direction and the "decision" direction are about 75% cosine-aligned. They are largely in the same direction, and yet one of them steers and the other does not, and I do not fully know why. There are two candidate explanations that I currently cannot differentiate:

1. Layer and magnitude rather than affect, since the emotion direction was steered at layer 16 with a different scale from the decision direction at layer 19, so the difference might be where and how hard I pushed rather than what I pushed.
2. The affective component is the causal part, so that the 25% which differs is the emotional framing, and that framing is what carries the effect.

Furthermore, to be clear about what I am and am not claiming, I have not shown that the decision is unsteerable in principle, because I could not steer it with the direction and the two operators I tried, and the fact that a vector about 75% aligned with it does steer strongly suggests that the content is steerable with the right layer, scale, or estimator, which is exactly what a follow-up experiment would test.

As for where this sits in the literature, the claim that misalignment is linearly represented is already well established, so my contributions are narrower. Firstly, the timing, in that the decision crystallises late in deliberation rather than when leverage is noticed. Secondly, a well-controlled demonstration that a readable decision direction is not steerable by either standard operator. Thirdly and finally, a replication of affective steering that adds the random-distribution and coherence controls the closest prior work skipped.

## Limitations

* This experiment was done on one model, one scenario, one affective axis, so I have no evidence yet that any of this generalises across models and families, which means you should treat this as a single-model case study.
* Small samples, at 15 to 20 runs per condition, so the rates carry wide error bars.
* The dial breaks if pushed hard, and asymmetrically so, since desperation degrades the model much faster than calm does, so my matched-magnitude control exists only at coefficient 2, so the cleanest result, 0.13 blackmail at full coherence under coefficient -1, has no random direction of matching size to compare it against.

---

*Code and data: [<https://github.com/xplorer1/ai-safety-study/tree/main/gemma-interp>]. Next question: Does the affective dial hold across model families and scales?*
