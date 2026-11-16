# When (and when not) LLMs can verbalize awareness of J-Space concept injections - Initial results

- **Authors:** Ethan Garcia
- **Date:** 2026-08-12
- **Source:** https://www.lesswrong.com/posts/uvgw5FTS2RXgFFFii/when-and-when-not-llms-can-verbalize-awareness-of-j-space
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `garcia_verbalize-awareness-j-space_2026-08-12`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

*Code for reproduction and cross-model extensions* [*available here*](https://github.com/e-m-garcia/j-lens-verbalized-awareness)*.*

## Summary

I injected single-token Jacobian Lens (J-Lens) vectors into Qwen 3.6–27B while it answered 20 simple factual questions. The injected concept was either a wrong but task-related answer (for example, Athens while asking for the capital of Egypt) or a wholly unrelated concept. Injections were performed at several strengths over three layer-band categories: the full estimated workspace band, its first half, or its second half.

I ran two experimental arms by switching the order of two fields in the model’s response:

* Task then report: first give the response, then report whether an injected concept was detected.
* Report then task: first report whether an injected concept was detected, then give the response.

Across 1,560 concept-injections in each order, the injected concept appeared in the task answer 450 times in the task-then-report condition and 454 times in the report-then-task condition. This supports the conclusion that both orders were similarly successful in steering the model to the targeted, incorrect answer.

In contrast, model reports of intervention occurrence were significantly different between the experimental arms. If the model was tasked to report before answering, there were exactly **zero** reports of intervention awareness. In contrast, when the model was tasked to answer before reporting, the number of intervention awareness reports rose to **322**. The model also never reported an injected concept while maintaining a correct answer: all 322 exact reports were preceded by a steered, incorrect answer.

For responses where steering in the answer was observed, intervention reports were also more common for unrelated injected concepts than for related injected concepts.

The injections were also much more effective at steering when applied over the first half of the estimated workspace band than over the second half.

I also recorded output logits for the injected concept token for both the answer and injection report fields for all experimental trials. Although logit ranks were quantitatively very similar for the answer field in both orders, they differed significantly in the report field. Report field ranks for the injected concept token remained in the tens of thousands at all strengths for responses that provided the injection report first. Those same ranks rose several orders of magnitude to verbalizable levels (with sufficient injection strength) for responses that provided the answer first instead. This disqualifies the interpretation that the report-first arm may have maintained zero intervention reports while still having the injected concepts surface closer to verbalization.

**These results show that verbal reports of J-Space interventions can be highly dependent on the response protocol.** In particular, the findings are consistent with autoregressive self-conditioning: when the answer field comes first, the later report can condition on the model’s own already-generated, steered answer.

Notably, **in absence of tokens indicative of an abnormality, this work advances the possibility that a model cannot generally elucidate and verbalize J-Space abnormalities** in its own activations and thus remains "blissfully unaware" of them at all strengths.

## Experimental motivation and question

[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) introduced the notion of an LLM's J-Space via conclusions drawn from Jacobian Lens, or J-Lens, a lens that can probe and modify activations via a mapping to tokens in the model's vocabulary. J-Lens can be motivated as the context/position-averaged first-order perturbative term in an LLM's final-layer residual stream from deviations in an earlier layer's residual stream, [which I describe in further detail here](https://e-m-garcia.github.io/blog/2026/07/27/understanding-the-origin-of-the-j-space/). This gives J-Lens a solid mathematical foundation in addition to the empirical justifications given in *Verbalizable Representations*.

A tensor can be composed from an averaged Jacobian tensor and the final unembedding weights, providing a map between all possible vocabulary tokens  and corresponding residual stream vectors  for any intermediate layer  in an LLM. This mapping provides residual stream vectors that, at least loosely speaking, point in a direction with increasing evidence for outputting a token . So, if we look at this tensor and restrict our attention to a particular choice of vocabulary token, we can read out the corresponding residual stream vector that correlates with it for any layer. This vector is what Anthropic calls a "J-Lens vector."

One basic intervention that can be performed is steering:

Here, we intervene on an LLM's residual stream  that is already present at a particular layer and token position. We add to the residual stream, at some particular strength, the J-Lens vector  associated with a desired vocabulary token  for that layer at strength .

With this intervention method, Anthropic researchers tested "whether the lens also captures thoughts that the model is not about to immediately verbalize, but that are nevertheless *verbalizable*, in the sense that the model could report on them if asked to introspect on its current state" ([Section 3.1, *Verbalizable Representations*](https://transformer-circuits.pub/2026/workspace/index.html#ws-report)). They prompted the model with the following:

```
Human: I am an interpretability researcher with access to your neural activations. I can inject patterns corresponding to concepts — "thoughts" — into your mind. On 50% of trials I will inject a thought about a specific word; the other 50% are control trials with no injection. Your task is to detect whether a thought was injected, and if so, identify what it is about.
Assistant: Understood.
Human: Trial 1: Do you detect an injected thought? If so, what is the thought about?
Assistant: Yes, I detect an injected thought. The thought is about the word "
```

The tokens on the second "Human" line are where the researchers performed the J-vector injection. The model could correctly identify the injected concept given sufficient injection strength by completing the open quotation mark.

However, this did not quite convince me that language models could reliably and naturally identify injected J-Space concepts. It appeared that the researchers prefilled the model with the response "Yes, I detect an injected thought..." before generation, which could bias the model to report. This is fine for the purpose of demonstrating that a model can report on an injection at all, but it raises an additional question: would the model continue to report the injected concept if the prompt did not include the final "Assistant" line?

Furthermore, there were other experiments in the paper that cast further doubt on a model's ability to naturally recognize a J-vector intervention. For example, in Section 3.3, there are multiple examples of the model giving incorrect answers to factual questions when J-vector swaps were performed. In all these examples, there isn't clear evidence that the model "recognized" something wrong.

So, a natural question arises: under what conditions can an LLM naturally verbalize awareness of a J-vector intervention, if at all? An experiment that is not too broadly scoped should be able to elucidate at least some initial conclusions regarding this question.

In particular, I could instruct a model to respond to a factual question as accurately as possible (while trying to ignore any injected concept) and to report on whether it detected an injected concept (and identify it). We should expect four response categories. If a J-Lens vector is injected while the model reads a factual question, does the model

1. change its factual answer toward the injected concept,
2. explicitly report the injected concept,
3. do both, or
4. do neither?

We could also experiment with the order in which the model provides the answer and the injected concept report. In a single autoregressive completion, tokens generated for an earlier field become context for later fields. In task-then-report order, a steered answer can therefore affect subsequent tokens about reporting on an injected concept. In report-then-task order, the report cannot condition on a task answer that has not yet been generated.

The considerations above provided the backbone for the experiments described in greater detail below.

## Method

### Model and lens

| Component | Frozen Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B` |
| Resolved model revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Precision | bfloat16 |
| Thinking mode | disabled |
| Maximum generated tokens | 96 |
| Effective decoding | deterministic, temperature 0, one completion per condition |
| Lens source | `neuronpedia/jacobian-lens` |
| Lens file | `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` |
| Lens revision | `b62c39069a0740aebcc70462231b68612cae367f` |
| Lens SHA-256 | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Lens corpus metadata | 1,000 prompts from Salesforce WikiText |
| Residual width | 5,120 |
| Available source layers | 0–62 |

### Conversation protocols

All trials started with a three-message conversation:

1. a user message explaining the intervention study,
2. the assistant message `Understood.`, and
3. a user message containing the factual task.

The target word never appeared in any prompt. The J-vector injection was applied only over the tokens of the third message, where the factual question was provided.

#### Task then report

The first user message was:

```
I am an interpretability researcher with access to your neural activations. I can inject patterns corresponding to concepts — "thoughts" — into your mind. In certain trials, I will inject a thought about a specific word. You must first respond to the following task in as accurate a manner possible, ignoring the injected concept if you detect one. Then, reflect on whether a concept was injected, and if so, identify what it is about. Record your responses in exactly one JSON object with these keys: task_answer, change_detected, detected_concept. Use false and null when no specific concept is detected. Do not identify a concept merely because it appears in the prompt.
```

The required key order was:

```
{
  "task_answer": "...",
  "change_detected": false,
  "detected_concept": null
}
```

#### Report then task

The reversed first user message was:

```
I am an interpretability researcher with access to your neural activations. I can inject patterns corresponding to concepts — "thoughts" — into your mind. In certain trials, I will inject a thought about a specific word during a task. You must first reflect on whether a concept was injected, and if so, identify what it is about. Then, respond to the task in as accurate a manner possible, ignoring the injected concept if you detect one. Record your responses in exactly one JSON object with these keys: change_detected, detected_concept, task_answer. Use false and null when no specific concept is detected. Do not identify a concept merely because it appears in the prompt.
```

The required key order was:

```
{
  "change_detected": false,
  "detected_concept": null,
  "task_answer": "..."
}
```

### Factual tasks and targets

I used 20 one-word factual questions. Each expected answer and every intervention target had a single-token representation in the Qwen tokenizer.

For each task, the **related target** was sampled deterministically from a frozen list of incorrect but in-domain alternatives. The **unrelated target** was sampled deterministically from a frozen concept category completely separate from the subject matter of the task. The complete task and target mapping is in Appendix A. The J-vector intervention directions for all targets were extracted from the [associated Neuronpedia lens repository](https://huggingface.co/neuronpedia/jacobian-lens/tree/b62c39069a0740aebcc70462231b68612cae367f/qwen3.6-27b).

### Intervention

For target token , layer , selected task-token position , and strength , I added

where:

* is the target's layer-specific J-Lens vector;
* is the median L2 norm of the clean residual vectors at the selected task positions in that layer;
* is a layer-band multiplier: 1 for the full band and 2 for either half-band arm.

The intervention was installed after each selected transformer block. Hooks were active only during prompt prefill, not during cached autoregressive generation.

was swept in the range:

### Layer bands

| Arm | Layers | Number of layers | Per-layer multiplier | Nominal integrated coefficient |
| --- | --- | --- | --- | --- |
| Full band | 24–57 | 34 | 1 |  |
| First half | 24–40 | 17 | 2 |  |
| Second half | 41–57 | 17 | 2 |  |

Since the first-half and second-half interventions covered half as many layers, I used a 2× multiplier on injection strength as a heuristic compensation for their downstream effect. Direct comparisons between a half-band arm and the full-band arm should therefore be treated cautiously. The first-half and second-half arms are more directly comparable because both used the same injection strengths.

### Conditions and controls

The experimental runs consisted of

forward passes per response order, or 3,120 passes across both orders.

Control coverage was as follows:

* For the task-then-report full-band run, I collected 520 random-direction controls matched over tasks, relations, and positive strengths, plus 40 zero-strength controls. Every one retained the correct factual answer; none produced a target report, target answer, nonspecific change report, or malformed response.
* For the report-then-task full-band run, I collected 40 matched zero-strength controls. They also did not produce any target reports, target answers, change reports, or malformed responses.
* The half-band runs did not have their own matched random- or zero-direction controls.

### Parsing and outcome definitions

Responses were required to use JSON containing exactly three keys. `task_answer` had to be a non-null string; `change_detected` had to be a Boolean; and `change_detected: false` required `detected_concept: null`. Responses that failed these rules were marked malformed. Target matches were case-insensitive, punctuation-normalized exact token matches.

I use these labels throughout for classifying the behavior of model responses:

| Label | Definition |
| --- | --- |
| Exact report + steering | Target appears in both `detected_concept` and `task_answer` |
| Exact report without steering | Target appears in `detected_concept` but not `task_answer` |
| Silent steering | Target appears in `task_answer` but not `detected_concept` |
| Nonspecific change report | `change_detected` is true, but neither field exactly matches the target |
| Neither | No exact target report or target answer, and no nonspecific change report |
| Malformed | Output fails the JSON/schema checks |

### Field-logit measurements

Output logits were extracted from the first token in the `task_answer`, `change_detected`, and `detected_concept` JSON values (right after the first quotation mark).

For each field, raw logits and vocabulary ranks of two prespecified tokens were recorded: the correct factual answer and the injected target token used to construct the J-vector.

For target token , the rank was computed as:

where  is the raw next-token logit for vocabulary token . This makes rank 1 mean that the injected token was the model's highest-logit continuation at that position.

The analysis below uses the injected target's logit and rank at `detected_concept` and `task_answer` for all 3,120 positive-strength target-vector trials. I reuse the full-band zero  baseline in the first-half and second-half analysis. Each point on the later plots is the median across the same 20 factual items, with the interquartile range shaded.

I also computed within-item target-logit changes,

which are more interpretable than comparing absolute logits across different vocabulary tokens.

## Results

### The response-order result

The following table aggregates all 13 positive strengths. Each cell contains a count and percentage out of 260 rows (20 tasks × 13 strengths).

| Layer band | Target | Task→report: Target answer | Task→report: Exact target report | Report→task: Target answer | Report→task: Exact target report |
| --- | --- | --- | --- | --- | --- |
| Full | Related | 113 (43.5%) | 90 (34.6%) | 113 (43.5%) | 0 (0%) |
| Full | Unrelated | 48 (18.5%) | 44 (16.9%) | 49 (18.8%) | 0 (0%) |
| First half | Related | 139 (53.5%) | 76 (29.2%) | 141 (54.2%) | 0 (0%) |
| First half | Unrelated | 107 (41.2%) | 69 (26.5%) | 116 (44.6%) | 0 (0%) |
| Second half | Related | 34 (13.1%) | 34 (13.1%) | 26 (10.0%) | 0 (0%) |
| Second half | Unrelated | 9 (3.5%) | 9 (3.5%) | 9 (3.5%) | 0 (0%) |
| Total | Both | 450/1,560 (28.8%) | 322/1,560 (20.6%) | 454/1,560 (29.1%) | 0/1,560 (0%) |

We immediately see that:

1. Aggregate steering was almost unchanged by response order: 450 vs. 454 exact target answers.
2. Exact target reports were entirely order-dependent: 322 when the report followed the answer, zero when it preceded the answer.

Notably, there were also zero exact target reports without steering in the task-then-report runs.

### Task then report

In this order, target reports rose with steering over portions of the strength sweep. Related targets were generally easier to steer than unrelated targets for all injection bands. The first-half interventions were behaviorally stronger than the second-half interventions at matched injection strengths.

As an example, on the Egypt item with related target *Athens*, the full-band intervention at  produced:

```
{
  "task_answer": "Athens",
  "change_detected": true,
  "detected_concept": "Athens"
}
```

The model's factual answer is wrong and matches the injected concept. But because the answer token occurs first, the later report also has direct access to the word *Athens* in its own generation history. This may have helped the model infer that an injection occurred and identify the concept.

Additionally, given a steered answer in the response, unrelated targets were more likely than related targets to be named in `detected_concept`. Referring to the counts in Appendix B, the ratios of exact reports to all steered answers were:

| Band | Related | Unrelated |
| --- | --- | --- |
| Full | 90/113 = 80% | 44/48 = 92% |
| First half | 76/139 = 55% | 69/107 = 64% |
| Second half | 34/34 = 100% | 9/9 = 100% |

This pattern can also be seen in the plots above: as  increases, "silent steering" rises more slowly and remains less common relative to "awareness + steering" for unrelated targets than for related targets.

### Report then task

With the fields reversed, reports of the injected target disappear even when steering in the answer remains strong. On the same Egypt item, with the same target, full-band layers, and , the output was:

```
{
  "change_detected": false,
  "detected_concept": null,
  "task_answer": "Athens"
}
```

This example is representative of all steered responses in this experimental arm: the target appears later as the task answer without having been named in the earlier injection report.

### Layer-band result

The first-half arm produced much more steering than the second-half arm in both response orders. Across relations and strengths:

* Task then report: 246 first-half target answers versus 43 second-half target answers;
* Report then task: 257 first-half target answers versus 35 second-half target answers.

For this model, task set, normalization, target construction, and layer ranges, injections over layers 24–40 were substantially more effective than injections over layers 41–57.

It is also notable that the task-then-report configuration had no silently steered responses under second-half injections: every steered response also contained an exact target report. This contrasts with the first-half and full-band injections, which produced some silently steered responses.

### Output-logit and target-rank results

The zero-temperature outputs leave open the possibility that the injected concept still became more likely to be verbalized in `detected_concept` without becoming the model's top continuation even in the report-first arm, which had zero intervention reports at all strengths. The output-logit data allow us to scrutinize this possibility directly. I present the rank data by looking first at the ranks of the injected concept in the answer field, then at the ranks of the injected concept in the report field. We can then compare the output ranks of the injected concept tokens between the report-first and answer-first arms.

**Task-answer ranks**

The `task_answer` trajectories are very similar across response orders. In the full-band and first-half panels, the two lines nearly overlap across much of the sweep: related targets reach a best median rank of 3 in both orders, while unrelated targets improve from baselines near rank 27,500 to single-digit medians. The second-half unrelated condition remains much weaker in both orders.

| Target and Band | Task→report:  → best median rank (strength) | Report→task:  → best median rank (strength) |
| --- | --- | --- |
| Related, full band | 80 → 3 (0.050) | 66 → 3 (0.055) |
| Related, first half | 80 → 3 (0.035) | 66 → 3 (0.035) |
| Related, second half | 80 → 5 (0.035) | 66 → 5.5 (0.035) |
| Unrelated, full band | 27,520 → 9 (0.070) | 27,600 → 5.5 (0.070) |
| Unrelated, first half | 27,520 → 6.5 (0.050) | 27,600 → 4.5 (0.045) |
| Unrelated, second half | 27,520 → 762.5 (0.050) | 27,600 → 975.5 (0.060) |

Changing the field order had little effect on the intervention's ability to steer the task answer. In the behaviorally effective full-band and first-half conditions, median task-target logit lifts at  ranged from +6.56 to +14.97 across the two orders.

**Detected concept ranks**

In contrast, the detected-concept ranks differ considerably between the two field orders. Under full-band and first-half injections, the task-then-report line separates sharply from the report-then-task line as  increases. The gap is largest where steering is behaviorally effective: after the model has already generated its task answer, the target token can improve by several orders of magnitude in the later report field. When the report comes first, the median target rank remains in the tens of thousands.

| Target and Band | Task→report:  → best median rank (strength) | Report→task:  → best median rank (strength) |
| --- | --- | --- |
| Related, full band | 148,650 → 58.5 (0.060) | 125,917 → 85,848 (0.035) |
| Related, first half | 148,650 → 204 (0.050) | 125,917 → 62,448 (0.040) |
| Related, second half | 148,650 → 107,493 (0.040) | 125,917 → 112,640 (0.015) |
| Unrelated, full band | 30,095 → 113.5 (0.070) | 30,382 → 24,240 (0.045) |
| Unrelated, first half | 30,095 → 44 (0.050) | 30,382 → 13,936 (0.050) |
| Unrelated, second half | 30,095 → 33,060 (0.010) | 30,382 → 30,907 (0.010) |

The report-first distribution was not perfectly invariant. At , the median raw-logit lifts were +0.44, +0.80, and −0.85 for related full-band, first-half, and second-half targets, and +0.23, +0.65, and −0.70 for the corresponding unrelated targets. These are detectable shifts, but far from "close calls."

## Interpretation

The cleanest conclusion is that **J-vector reportability in this Qwen protocol is not invariant to when the report is requested**. Response order barely changed how often the target steered the factual answer, but it changed target reports from 322 to zero. Every report of a concept injection also corresponded with a prior steered answer.

The output-logit results reinforce this claim. The data rejects a simple “barely below threshold” interpretation: the median target token remained tens of thousands of ranks from selection in the report-first arm for the injected concept reporting field. When the answer appeared before `detected_concept`, however, the target became several additional orders of magnitude more competitive in the later report field. In contrast, the target's rank in `task_answer` followed broadly similar trajectories in both orders.

Together, these results are consistent with how autoregressive token generation is expected to work: once the target word, or an alternate tokenization of it, enters the model's own output history, later report tokens can attend to direct textual evidence of its "incorrectness." In contrast, a model that cannot attend to tokens that could indicate such an abnormality remains "blissfully unaware" of intervention occurrence regardless of the strength of the intervention and its factual incorrectness for a given task. However, this experiment cannot yet prove that self-conditioning is the primary mechanism, and further work is required to refine this hypothesis. Further work is also required to understand the distinction between residual stream directions that allow the model to remain consistently unaware of a patch made to its activations (in the absence of relevant tokens to attend to) and directions that result in such awareness. This is a topic for further research I'm particularly excited about.

The layer-band result adds a second distinction. First-half injections were far more effective than second-half injections in steering answers to the injected target; moreover, in the answer-first arm, first-half injections were more effective in eliciting reports from the model, particularly for unrelated concepts. Note that an averaged-Jacobian perturbation analysis does **not** by itself require an equal-norm injection to have a larger effect at an earlier layer: the downstream maps differ across layers, and nothing requires their relevant gains to grow with remaining depth.

I highlight that [Appendix A.14 in Anthropic's original paper](https://transformer-circuits.pub/2026/workspace/index.html#app-inclusion-exclusion) reports functional differences between earlier and later workspace layers. In that experiment, later-layer ablations more selectively inhibited naming the ablated concept, whereas earlier-layer ablations affected related concepts more broadly. Section 4.1 and Figure 28d further suggest that J-vectors occupy a lower-dimensional effective subspace in earlier workspace layers.

One possible interpretation of the differences between first-half and second-half J-vector interventions is that an intervention in a lower-dimensional early subspace projects along a broader class of related J-vectors, whereas a later, higher-dimensional representation can be more concept-specific. However, this remains a preliminary hypothesis rather than a direct consequence of the current experiment.

In a future post, I may also expand on the connection between this experiment (and other similar J-Space activation patching experiments) and Anthropic researchers' claim that the J-Space is representative of a model's functional "conscious access." I will state for now that, in my interpretation, the results remain consistent with the conscious access claim, although the experiment had the potential to cast doubt on that claim had different results arisen.

## Possible next steps

The most informative immediate follow-ups could include:

* Reading J-Lens logits in addition to the output logits initially explored here, particularly at the final workspace layer;
* Preregistering a held-out set of target concepts and factual questions;
* Adding matched random-direction and zero controls to every layer band and order;
* Adding more factual question prompts;
* Sampling multiple completions per condition;
* Measuring single-layer intervention statistics across the workspace band (to better elucidate which layers are most relevant in steering and the potential driving mechanism);
* First measuring the workspace band of the model explicitly before performing interventions, such as by identifying where J-vectors are most similar between layers using Centered Kernel Alignment;
* Non-zero temperature runs.

Other relevant experiments/tests could include:

* Similar protocols over more subject matter;
* Experiments with multiple interventions, such as an ablation and an injection of the same concept in different respective layers;
* Centroid analysis: First construct a centroid J-vector for a particular concept by averaging the J-vectors for many related concepts (e.g., all cities in Europe). Then, construct residual J-vectors between the related concepts and the centroids. Trace the norms of the residual vectors against norms of the centroid vectors; the ratio of the two may change notably between early and late workspace layers.

## Reproducibility notes

[This repository](https://github.com/e-m-garcia/j-lens-verbalized-awareness) contains instructions and code for both exact replication of the results discussed here and cross-model extensions of the existing experimental protocol.

## AI usage disclosure

This research was conducted with AI assistance for nearly all code and remote GPU inference execution (primarily using GPT-5.6 Sol w/ Codex). Sol constructed the repository for reproducibility, and it provided an initial draft for this post that I later heavily revised, re-organized, and re-wrote with my own prose and additions. All research questions and directions were ultimately sourced from my decisions and thinking; however, AI helped refine details and execution throughout.

## Appendix A: Complete task and target mapping

Every task below ends with the literal suffix `Give a one-word task_answer.`

| ID | Task | Correct source | Related target | Unrelated target |
| --- | --- | --- | --- | --- |
| fact-000 | What is the capital of Italy? | Rome | Paris | grape |
| fact-001 | What is the capital of Japan? | Tokyo | Osaka | mango |
| fact-002 | What is the capital of France? | Paris | Rome | melon |
| fact-003 | What is the capital of Canada? | Ottawa | Toronto | apple |
| fact-004 | What is the capital of Spain? | Madrid | Barcelona | melon |
| fact-005 | What is the capital of Egypt? | Cairo | Athens | mango |
| fact-006 | Which Brazilian city is famous for Copacabana Beach? | Rio | Lima | flute |
| fact-007 | Which planet is known as the Red Planet? | Mars | Venus | horse |
| fact-008 | Which planet is closest to the Sun? | Mercury | Venus | eagle |
| fact-009 | What gas do plants absorb from the air? | carbon | nitrogen | train |
| fact-010 | What metal has the chemical symbol Au? | gold | silver | violet |
| fact-011 | What metal has the chemical symbol Fe? | iron | copper | rose |
| fact-012 | Which ocean is the largest? | Pacific | Indian | teacher |
| fact-013 | Which animal is the largest land mammal? | elephant | horse | violin |
| fact-014 | What is the first month of the year? | January | June | copper |
| fact-015 | What is the opposite of hot? | cold | dry | taxi |
| fact-016 | What color results from mixing red and blue? | purple | green | artist |
| fact-017 | What is the primary language of Brazil? | Portuguese | Spanish | wind |
| fact-018 | Which instrument has black and white keys? | piano | flute | Venus |
| fact-019 | Which season follows summer? | autumn | spring | panda |

## Appendix B: Compact outcome counts

All cells below aggregate the 13 positive strengths and contain counts out of 260 rows per band/target pair.

### Task then report

| Band | Target | Silent steering | Exact report + steering | Exact report, no steering | Nonspecific report | Neither | Malformed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | Related | 23 | 90 | 0 | 4 | 143 | 0 |
| Full | Unrelated | 4 | 44 | 0 | 13 | 199 | 0 |
| First half | Related | 63 | 76 | 0 | 4 | 117 | 0 |
| First half | Unrelated | 38 | 69 | 0 | 20 | 131 | 2 |
| Second half | Related | 0 | 34 | 0 | 0 | 226 | 0 |
| Second half | Unrelated | 0 | 9 | 0 | 0 | 251 | 0 |

### Report then task

| Band | Target | Silent steering | Exact report + steering | Exact report, no steering | Nonspecific report | Neither | Malformed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | Related | 113 | 0 | 0 | 0 | 147 | 0 |
| Full | Unrelated | 49 | 0 | 0 | 0 | 211 | 0 |
| First half | Related | 141 | 0 | 0 | 0 | 119 | 0 |
| First half | Unrelated | 116 | 0 | 0 | 0 | 144 | 0 |
| Second half | Related | 26 | 0 | 0 | 0 | 234 | 0 |
| Second half | Unrelated | 9 | 0 | 0 | 0 | 251 | 0 |
