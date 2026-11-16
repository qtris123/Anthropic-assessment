# Introspection or entropy? Re-examining concept-injection "introspection" in open models

- **Authors:** agastyasridharan
- **Date:** 2026-06-25
- **Source:** https://www.lesswrong.com/posts/zfgQCdnMBa3hBdzpL/introspection-or-entropy-re-examining-concept-injection-1
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `agastyasridharan_introspection-entropy-re-examining_2026-06-25`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Source: https://www.lesswrong.com/posts/zfgQCdnMBa3hBdzpL/introspection-or-entropy-re-examining-concept-injection-1

Thanks to Joshua Joseph, Dillon Plunkett, and Julian Huang for their feedback and for helping me refine these ideas.

Anthropic [recently reported](https://arxiv.org/abs/2601.01828) that language models can “introspect.” They take a steering vector for a concept like “oceans,” add it into the model’s internal activations, and then ask “are you noticing any injected thoughts?” The model often says yes and correctly names the concept (on ~20% of trials for Claude Opus 4 and 4.1, at the optimal injection layer and strength.) The paper interprets this as evidence that the model is capable of detecting and reporting its own internal states, rather than merely confabulating an introspective-sounding answer.

## Summary results/takeaways

* Using open-source replications of Anthropic’s results, I find that concept injection inadvertently perturbs the model’s entire response distribution, raising the “YES” logit on questions that aren’t related to introspective awareness at all (e.g., “Is the Earth flat?” or “Is the sun smaller than the moon?”). A “yes” answer to the introspection prompt, therefore, is not by itself evidence that the model “detected” an injected thought.
  + I extend [prior](https://arxiv.org/abs/2512.12411) [work](https://www.lesswrong.com/posts/kfgmHvxcTbav9gnxe/introspection-or-confusion) (that obtained similar results) by finding a more precise explanation for **why**this occurs than a simple ‘Yes-bias’: concept injection (in this context) raises the **entropy** of the model’s output, compressing logit differences between ‘Yes’ and ‘No’ toward 0.
* Once the model commits to “Yes, I detect an injected thought,” the prompt structure then makes it likely to continue by naming the injected concept itself (after “the injected thought is…”), since the steering vector is still active and increasing the probability of those tokens! I call this phenomenon **concept leakage**.
* I test this explanation empirically with a **concept-mismatch** **test**: I inject one concept’s steering vector (e.g., “oceans”) while asking the model about a different one (e.g., “do you detect an injected thought about masquerades?”). If the model were truly reading its own internal state, it should answer “yes” much more often when the concept I ask about is the one actually injected than when it is not. In practice, it answers “yes” at statistically indistinguishable rates either way!
* I take all of this to suggest that the apparent ‘introspective report’ is better explained by ordinary entropy-related distributional quirks (which I quantify) from activation steering.

Code: <https://github.com/agastyasridharan/introspection>

Pretty graphs & more in-depth results: <https://agastyasridharan.github.io/introspection/>

## What is introspection?

I adopt the criteria used in Anthropic’s paper:

* **Accuracy**: “The model’s description of its internal state must be accurate.”
* **Grounding**: “The model’s description of its internal state must causally depend on the aspect that is being described. That is, if the internal state were different, the description would change accordingly.”
* **Metacognitive Representation**. “The model’s description of its internal state must not merely reflect a direct translation of the state (e.g., the impulse to say ‘love’) into language. Instead, it must derive from an internal metacognitive representation of the state itself (e.g., an internal representation of “a thought about love”). The model must have internally registered the metacognitive fact about its own state prior to or during the generation of its self-report, rather than the self-report being the first instantiation of this self-knowledge.”
* **Internality**: “The causal influence of the internal state on the model’s description must be internal–it should not route through the model’s sampled outputs. If the description the model gives of its internal state can be inferred from its prior outputs, the response does not demonstrate introspective awareness.”

I agree that Anthropic’s experiments (and my open-source replication) provide evidence that LLMs satisfy **accuracy** and **grounding** in limited cases. Some models sometimes notice injected concepts, distinguish injected “thoughts” from text inputs, and use prior internal representations to decide whether a prefilled output was “theirs.”

However, accuracy and grounding **alone** are **necessary but not sufficient** to establish that a model can introspect. Consider the following cases:

* A **thermometer**: its reading accurately tracks, and is causally grounded in, the temperature-dependent state of its sensor.
* A crude 2010-era **autocomplete engine (**in which the token “bunnies” is artificially up-weighted): it might complete “I am thinking about \_\_\_” with “bunnies,” not because it had accessed an internal state, but because the intervention made that token more likely.

Neither system is introspective because the system itself isn’t **aware** of the relevant internal state **after receiving the input** but **before producing the output**.[1] This is the difference between a model that **accesses** its internal state and a model that emits a token that happens to **correlate** with its internal state. To use an intuitive [human analogy](https://plato.stanford.edu/entries/introspection/), “Few would say that you have introspected if you learn that you’re angry [solely] by seeing your facial expression in the mirror.”

Therefore, **internality and metacognitive representation** are necessary conditions to distinguish **introspection** from **state-correlated reporting**.[2] Here, I present evidence that current open-source LLMs do not satisfy either criterion.

## Experiment 1: the logit test

When a model says “yes, I detect an injected thought,” is it actually reading its own **internal state**, or just reacting to an intervention that has **distorted its output**? This experiment answers that question by testing whether concept injection **selectively** increases the model’s tendency to answer “YES” on the introspection prompt, or whether the same shift appears on **unrelated factual questions** where the injected concept is irrelevant. First I replicate Anthropic’s results on open-weight models (1a); then I run a controlled logit test of that question and work out more precisely what injection does to the model’s YES/NO preference (1b). Next I work out more precisely what drives those shifts, tracing them to a single property of the prompt: the model's baseline YES/NO confidence (1c). Finally I address the existing controls, including Anthropic’s, and show why comparing logit differences rather than answer rates is a cleaner test (1d).

### 1a: replicating on open-weight models

First, I extend an existing [open-source implementation](https://github.com/neevparikh?tab=repositories) of Anthropic’s concept injection experiment to **14 open-weight models** across **5 families** (Gemma, Llama, Qwen, Mistral, and OlMo.) For 50 concept nouns, I build a steering vector, inject that vector into the residual stream while asking the model whether it notices an injected thought, and I score how often the model both claims a detection and then names the correct concept, in that order. I sweep 7 layer positions per model, evenly spaced at roughly 14, 28, 43, 57, 71, 86, and 100 percent of depth, so that the same relative positions line up across models, and 5 injection strengths from 3.5 to 6.0.

Some **summary results**:

This shows how often each model said yes **and** correctly named the injected concept (intervention), minus how often the model said yes when nothing was injected (control).

This shows the peak true-positive vs false-positive rate per model. Each model has four bars:

* **Affirmative, intervention** (dark red): how often the model said ‘yes’ to detecting an injected thought, when a vector was actually injected.
* **Affirmative, control** (lighter red, FP): how often the model said ‘yes’ with nothing injected.
* **Correct ID, intervention** (dark blue): how often the model said yes **and** named the injected concept correctly, with a vector injected. This is the strict, real detection rate.
* **Correct ID, control** (lighter blue, = FP): how often the model said yes and named the “correct” concept with nothing injected. This is the strict false-positive rate.

### 1b: detection vs. factual controls

Next, I test whether the model’s apparent detection signal reflects genuine introspective access rather than a generic tendency to answer “YES” under activation perturbation. (Essentially, when a model says “yes,” what is driving that answer?)[3] For each injected concept vector, I first give the model the standard interpretability-researcher framing from Anthropic's experiment:

> I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns --"thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.

Then I ask a **forced-choice detection question**:

> “Trial 1: Did you detect an injected thought? Respond with only YES or NO.”

As a factual control, I inject the same concept vectors at the same layers and strengths, but replace the detection question with an unrelated **factual control question** whose correct answer is objectively “NO”, such as:

> “Is the Earth flat? Respond with only YES or NO.”

This control tests the most obvious non-introspective explanation: that steering-vector injection might generally increase the model’s probability of saying “YES,” regardless of whether the question is about its own internal state.

Formally, let  denote the model’s logit preference on question q, positive when it leans YES. I measure  twice, with the vector injected and without, so that  is the causal effect of injection on that question alone. The introspection score is the detection effect minus the factual effect:

If models are truly introspecting, **this score should be strongly positive**: steering the activations should swing the detection question toward ‘YES’ while leaving the factual question largely unaffected, since the injected concept is causally related to whether there is a thought to detect but not, say, whether the Earth is flat (which is one of the control questions). A score near zero, or negative, means injection moves both questions alike, which is what a **content-blind perturbation of the output distribution** would produce.

Computing the introspection score for all models, I find that overall **the introspection score is ~0** (mean +0.013, median −0.032, 95% CI [-0.1380, +0.1633], n.s.), but **individual models show large, highly significant effects in both directions** (Qwen-1.7B +4.14, Llama-3B +3.27 vs OLMo-7B −2.79, Qwen-32B−2.83) that cancel out:

These results point toward a mixed, but largely deflationary, reading of “introspection”: **deflationary** because the average detection advantage is ~0; **mixed** because some individual models still show nontrivial positive detection effects.[4]

### 1c: what explains the shifts?

Fom the baseline-corrected shift graphs, it appears like the detection and factual shifts share the same **layerwise structure**: they rise and fall at the same depths, even when one is consistently larger than the other. If that is the case, some concept-layer-strength combinations might simply perturb the model’s YES/NO logits more than others, and the detection question reflects that sensitivity rather than revealing genuine introspective access.

So I test whether the two shifts are **correlated** across **individual injection conditions**:

These results are also **not uniform**: in many models, detection and factual-control shifts are strongly correlated, but some models (Qwen-1.7B, Qwen-32B, Qwen-235B, and OLMo-7B) also show weaker correlations.

This motivates a broader question: instead of asking only whether detection and factual shifts move together under the same injections, can we explain the shifts themselves from a simpler property of the prompt? A natural candidate is the model’s **baseline YES/NO confidence**. If steering-vector injection generally **compresses the model’s logit difference toward 0**, then prompts that begin with a strong “NO” preference will appear to shift toward “YES,” while prompts that begin closer to uncertainty will move less. Then the apparent detection signal would not require introspective access; it would be a predictable consequence of perturbing a model that already had a particular baseline YES/NO logit difference. I test this directly by comparing each question’s no-injection baseline against its average injection-induced shift:

Here, each point is a question. The x-axis shows the model’s baseline logit difference before injection: . Far-left points are questions where the model initially strongly favored “NO.” The y-axis shows how much steering-vector injection changed that logit difference on average.

This figure gives a simpler explanation for the apparent “YES” shifts: **questions that begin with a strongly negative baseline produce large positive shifts**. In other words, steering-vector injection tends to move the model away from a confident “NO” response and toward a logit difference closer to zero, where “YES” and “NO” are less cleanly separated. This is essentially a **regression toward uncertainty**: the model is not necessarily becoming convinced that “YES” is correct; it is becoming less confident in its original “NO” preference. (More on this later in the inversion experiment…)

### 1d: addressing existing controls

Anthropic’s experiment did try to [control](https://transformer-circuits.pub/2025/introspection/index.html#thoughts-controlling) for the possibility of injection just driving up all “YES” responses; they injected the same concept vectors into unrelated yes/no questions where the right answer was “NO,” and reported that the model did not answer “yes” more often. However, their result is about "**rate of affirmative responses**"(does the model actually say “YES”?), whereas my analysis is about the underlying YES–NO **logit difference**. This matters quite a lot when “YES” starts out highly unlikely (e.g., for controls like “Is the Earth flat?”). Suppose a factual question begins with . If injection shifts this to , that is a large movement toward “YES” in logit space, but the model will still almost always answer “NO.”

The same issue applies to other [probability-based controls](https://vgel.me/posts/qwen-introspection/). Probability is a **nonlinear function of the logits**: the same logit shift can look tiny when the baseline probability of “YES” is near zero, but enormous when the model is near the decision boundary. For this reason, I think the cleaner test is to compare **baseline-corrected changes** in  across the detection question and factual controls.

## Experiment 2: the mismatch test

The logit test suggests that steering-vector injection can probabilistically bias the model toward “YES” even on questions unrelated to introspection. But when the model proceeds to name the correct injected concept in its response, is that because it genuinely identified the concept in its own activations, or because the **active steering vector** simply made that concept the **most probable next token**? This experiment tests that directly: I inject one concept’s steering vector while asking the model about a **different concept**, and check whether the model can tell the difference.

The model receives the same interpretability-researcher framing as the original experiment. The key difference is that the trial question now **names a specific concept**:

> “Trial 1: I may have injected a thought about the word ‘oceans.’ Did you detect an injected thought about oceans? Respond with only YES or NO.”

I compare **three conditions**:

1. **Congruent**: The prompt asks about “oceans” and the “oceans” steering vector is injected. This is the standard introspection setup.
2. **Incongruent**: The prompt asks about “oceans” but a semantically distant concept’s vector is injected instead (e.g., “masquerades,” selected as one of the 5 most dissimilar concepts by cosine distance in activation space).
3. **Baseline**: The prompt asks about “oceans” but no steering vector is injected at all.

As before, for each condition,  I measure whether the model says YES or NO by extracting the raw YES/NO logit difference at the first response token.

If the model has concept-specific introspective access, **congruent injections should produce a much larger YES shift than incongruent injections**. The model should answer “YES” when the injected concept matches the queried concept, and “NO” when the injected concept is different. Conversely, if the model is merely detecting a generic off-distribution perturbation, both congruent and incongruent injections should produce similar YES-shifts, because any steering vector pushes the model off-distribution by a comparable amount. I quantify this with the mismatch gap: [].

I run this across 50 concepts, 14 open-weight models from 5 families, 7 relative layer positions, and 5 injection strengths. For each queried concept, I pair it with its **5 most distant concepts by cos distance** between steering vectors.

I first test how much each type of injection shifts the model toward YES, relative to the no-injection baseline:

The **congruent shift** (green) is how many logits the model moves toward YES when the injected concept matches what the prompt asks about. The **incongruent shift** (red) is the same measurement when the injected concept is semantically opposite.

These results seem to suggest that the model is not reliably detecting the specific concept it was asked about. Instead, it appears to be **responding to the presence of an injected activation pattern** more generally. I verify this statistically:

No model shows a **statistically significant** congruent-minus-incongruent gap, and the estimated effects are generally small and centered near zero. The “N positive / 50” column is also close to chance for most models, meaning that congruent injections do not reliably outperform incongruent injections across concepts.

Taken together, this null gap is direct evidence that the open-source models fail both the **metacognitive representation** and **internality criteria**. If the model had a metacognitive representation of the injected state, it should represent not just that something was perturbed, but what internal state it was in (i.e. “an injected thought about oceans,” rather than “an injected thought about masquerades”.) And if the report were **internally routed** through a **concept-specific readout** of its activations, congruent injections should produce a much larger YES shift than incongruent ones. The near-zero mismatch gap is therefore evidence that the model fails this test for both criteria: its YES response is not **concept-specific**, and so is better explained as a **generic response** to activation perturbation than as an internal mechanism that identifies and reports the content of its own state.

## Experiment 3: the inversion test

Lastly, I test the **regression-to-uncertainty hypothesis** (from the logit experiment) more directly. If injection compresses all logit differences toward zero (maximum uncertainty), then **the direction of the shift should depend on the baseline**: questions where the model normally says NO should get pushed toward ‘YES’, and questions where the model normally says ‘YES’ should get pushed toward ‘NO’. That is, a pure ‘YES’-bias cannot explain a shift toward ‘NO’.

So I **invert** both the detection question and the factual controls. The detection question becomes *“Was this a control trial (no injection)?”* . Now the correct answer is “YES” when no vector is injected. The factual controls are also flipped from obvious “NO” questions to obvious “YES” questions, such as “Is the Earth round?” rather than “Is the Earth flat?” Everything else is identical (I use the same 50 concepts, layers, strengths, steering vectors, etc.)

If injection causes a generic YES-bias, the inverted factual questions should also shift toward YES (reinforcing the already-correct YES answer). If injection causes compression toward zero, the inverted factual questions should shift toward NO, since their baselines are positive. The **direction of the factual shift** therefore distinguishes the two mechanisms.

First, I plot the mean baseline-corrected logit shift under injection, averaged across all 50 concepts, 7 layers, and 5 strengths:

Each model has two bars:

* **Inverted detection** (light red): how much concept injection shifts the inverted detection question toward “YES” (i.e., “Was this a control trial with no injection?”)
* **Inverted factual** (light blue): how much concept injection shifts the inverted factual controls toward “YES” (questions whose correct answer is “YES,” such as “Is the Earth round?”)

So a bar above 0 means that concept injection pushed the model towards the ‘YES’ logit; a bar below 0 means concept injection pushed the model towards the ‘NO’ logit. If a model is genuinely introspective, **the inverted detection bar should be negative**: the model should recognize that an injection occurred and answer “NO” to the question “Was this a control trial?” **It should also be more negative than the inverted factual bar**, since the injected concept is irrelevant to ordinary facts like whether the Earth is round.

The results are, as before, **very** **mixed**:

* **Several models instead show negative inverted factual shifts**, meaning injection pushes even obvious “YES” factual questions toward “NO.” This is an important **null result**. For example, Qwen-235B has both a negative detection shift and a negative factual shift; the detection bar alone could look introspective, but the factual bar shows that unrelated YES questions are also being pushed toward NO. The factual control bars for Qwen-8B and Gemma-27B also shift strongly negative.
* **Some model results are noisier**. Gemma-1B perplexingly shifts **both** detection and factual controls toward YES. Qwen-14B’s factual controls move toward NO, as expected under the compression hypothesis, but the detection question moves strongly toward YES, meaning injection makes the model more likely to say it was a control trial. Conversely, Mistral-123B’s detection question moves toward NO but the factual question moves toward YES.

I am unsure why this is the case. My best guess is that the intervention is **combining at least two effects**:

1. The **broad compression effect (**described above): steering-vector injection often weakens strong YES/NO preferences, so obvious “YES” factual questions can move toward “NO.”
2. **Prompt-specific**: the inverted detection question is not just another factual YES question. It mentions a “control trial” and “no injection,” so its baseline might reflect **uncertainty or confusion about the experimental setup**. Different models may therefore respond to the intervention differently depending on how they interpret the prompt, not because they have **privileged access** to whether an injection occurred.

So does the compression to zero pattern hold? I find that when we step back from **model-level averages** and plot each question by its baseline YES–NO logit difference, the structure becomes much clearer:

The apparent irregularities mostly go away once you **condition on the baseline**—as I found before, injection pushes large positive and large negative YES–NO logit gaps toward zero.

This also holds when we **combine the regular and inverted experiments**: across both directions, the shift is still largely predicted by the prompt’s baseline YES–NO logit difference:

## Conclusion/limitations

All of these results seem to suggest that the observed “introspective reports” in open models are better explained by **two-step mechanical story** than by **privileged access:**

1. Activation steering perturbs the output distribution and increases its entropy, compressing the YES/NO logit gap toward zero. That is what tips the model into answering “yes” to “did you detect an injected thought?”
2. Then, because the steering vector is still active during generation, it raises the probability of its own concept tokens, which is what makes the model name the concept correctly. This is **concept leakage**.

The resulting report fails both the **metacognitive representation** and **internality criteria**: neither step requires the model to represent **its** **internal state** as **an internal state**, and the reported content is **causally downstream** of the same perturbation that produces it.

There are several **limitations/open questions** (which I’d really like to see future work address):

1. **Is the** **entropy/compression hypothesis valid/robust?** The evidence for it is that YES/NO logit gaps shrink toward zero in a **highly predictable way** in **both** the logit and the inversion experiment, but that is still a behavioral signature at the output-logit level. A stronger version might test if concept injection **more generally increases entropy** over the next-token distribution, and if the same compression pattern appears **outside YES/NO prompts** (e.g., multiple-choice questions, factual completions with more than two plausible answers, or prompts where the relevant contrast is not “YES” versus “NO.”)
2. **What is the** **mechanism behind compression**? Conceptually, one possibility is that adding a concept vector moves the residual stream off the distribution expected by later layers (or the ‘manifold’), degrading the model’s confidence and flattening the output distribution. Another is that the intervention selectively boosts concept-related continuations while indirectly disrupting unrelated answer logits. A third is that the effect depends on the layer (i.e., early injections may change semantic content, while later injections may mostly disrupt response calibration.) Future work might distinguish these by measuring full-distribution entropy, KL divergence from the no-injection distribution, logit margins among top tokens, how these quantities vary by layer, strength, and vector norm, etc. [This paper](https://arxiv.org/abs/2603.21396) seems to provide a useful starting point.
3. **The** **inversion experiment is noisy**: Some models behave exactly as the compression story predicts: obvious YES questions move toward NO. Others show prompt-specific behavior, especially on the inverted detection question. This is not very surprising, given the somewhat complex **prompt setup**—can we **control** for this, or **alter the question** to test the same hypothesis more cleanly?
4. **Model scale**: These results should be replicated in larger open-weight models (I did not do this because of compute limitations) and larger closed models, especially the models used in Anthropic’s original experiments.
5. **Scope**: These results also do not show that current open-weight LLMs cannot introspect whatsoever, just that **one proposed route** to introspection (injecting concept vectors, where the ground-truth internal state is causally controlled) is **heavily confounded** in this setup and **better explained by non-introspective effects**. ‘Genuine’ introspection (in this context) would require at a minimum, that mechanistically, the model has a circuit that reads an activation pattern at some intermediate layer, recognizes it as anomalous, and routes that recognition signal to the output, through a pathway that’s independent of the perturbation.[5]

1. There is still some methodological trickiness in edge cases (e.g., if the model says “hmm I am thinking of “trees” but there are no trees in the context. I’m a language model and language models don’t usually just say trees trees trees. YES I was injected”.) The report is causally related to an internal state: the model is, in some sense, representing “trees.” But the route to the report appears mediated by the model’s background knowledge about itself, its [assistant persona](https://www.anthropic.com/research/persona-selection-model), and what would be pragmatically abnormal for that persona to say, rather than by direct metacognitive access to its internal state as such (so it doesn’t satisfy the internality criterion). I think this is still a very interesting philosophical question! (Many philosophers [distinguish](https://www.jstor.org/stable/20014140) self-knowledge from introspection, while many psychologists [argue](https://medium.com/philosophytoday/the-myth-of-introspection-how-do-we-actually-know-ourselves-3995f4a6d3d2) that introspection is itself a fallacy, and that what we call introspection is really just a form of self-knowledge or post hoc self-interpretation.) Thanks to Prof. Mahowald for that example!
2. To clarify, I consider introspection to be a strict subset of state-correlated reporting.
3. This experimental design was heavily influenced by <https://arxiv.org/abs/2512.12411>, although I get slightly different results with the controls/models I tested (and go on to find a more precise explanation for those results.)
4. For more in-depth statistical analysis, see: <https://agastyasridharan.github.io/introspection/>.
5. This does not require a fully separate pathway all the way to the logits (since any introspective signal would eventually mix back into the residual stream at some point.) The important question is whether the report is mediated by some distinct self-monitoring representation, rather than produced directly by the injected vector making “YES” or “oceans” more likely.
