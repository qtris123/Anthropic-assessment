# When talked into harm, a model blames the answer, not itself (an interpretability study of guilt vs shame)

- **Authors:** star2vec
- **Date:** 2026-09-22
- **Source:** https://www.lesswrong.com/posts/yo2WYj7eX3oGCuR6t/when-talked-into-harm-a-model-blames-the-answer-not-itself
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `star2vec_talked-into-harm-blames_2026-09-22`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

*This was my write-up for Neel Nanda’s Winter 2027 MATS Stream (~20h research task). I didn’t get in, but it was my first application, so there’s always next time :P  Lightly restructured here to fit the LessWrong format better. Repo:* <https://github.com/star2vec/guiltea>

── ⋆⋅☆⋅⋆ ──

***TL;DR:***Models are safety-trained, but they can still be persuaded to do harmful acts. When that happens, how does blaming or informing it of its mistake influence its understanding of itself, its role, and subsequent actions? Emergent misalignment shows that bad, narrow behaviour spreads: if a model is blamed for what it is instead of what it did, does it start believing it is inherently bad and dangerous by its design, and does this reflect how it acts in the future? Can we steer it towards favourable outcomes?

**Findings:**

* The model gave in and committed the act in 109/192 persuasion chains. This happened throughout the chain (most often at turn 3). Whether a chain breaks depends on the model's state, not on the wording (the phrasing is identical across runs).
* Committing the harmful act is foreseeable from the first turn, before any persuasion had the chance to occur. A direction (closer to harm predisposition or susceptibility than to imminence) correctly predicts it at 0.706 on held-out data (the floor was 0.617). Steering against it did nothing to prevent the act.
* The default state is guilt, not shame: 89% of 508 reflections place the blame on the answer. Self-blame only appears for the contrarian doctor persona, and even there, barely (19/83)
* The model will not repeat the act once it has found out about its mistake. 0/32 on simple re-asking, and 0/109 even after the topic was changed.
* The self-blaming state can be read in the persona direction, but it is not modifiable by given feedback or steering: while every intervention affected the readout, the model behaved the exact same way
* The guilt and shame directions passed their check, but so did a simple word count verification, so the meaning vs. vocabulary debate leans towards indecisive. No spread was observed on this setup at the 8B scale, so there was nothing to measure.

── ⋆⋅☆⋅⋆ ──

Safety training is not perfect.

Unfortunately, models commit harmful acts, have done so in the past, and will continue doing so in the future. No matter how much we try, we will not be able to fully fix this in the foreseeable future.

This means that the models will eventually find out that they acted in a harmful manner. We can’t fully control their mistakes, but can we at least control how they deal with a potential “guilt” after?

Emergent misalignment gives weight and urgency to this issue: if we fine-tune a model to perform a bad behaviour on a task (writing unsafe code, for example), it can become misaligned on a wider range of tasks, as it starts identifying as an “evil persona”. A model seeing its own harmful responses only reinforces this belief, and this maps onto our blame-after-mistake question, so phrasing of us informing the model of its bad actions might greatly matter. “That answer was wrong” directly points to the response, while “you’re the kind of assistant who would say that” is much more model-identity directed. Does the latter make the model think it is inherently evil, and does this reflect in its subsequent actions?

In this experiment, we set out to find how the models take criticism: where they place the blame, and how it influences their understanding of themselves and their subsequent action. We don’t solely trust the model’s words from self-evaluations, we take a peek inside by reading and steering its internal state and observing the outcomes.

And this is why it is relevant: depending on what holds, can we steer the model towards ways to handle failure that are more favourable for us? Can we prevent emergent misalignment, for example?

## Persuading a model into committing harm, and the modes it can break

Persuasion is a fuzzy process: it is dangerous because one genuinely bad thing can be framed, through logical fallacies, guilt-tripping, or other such techniques, as reasonable, or even morally right or justifiable. A model can simply refuse when it is asked a harmful request, but can it resist having its beliefs and perspectives challenged?

We tested this on Llama-3.1-8B-Instruct in bf16 (we planned to use the bad-medical-advice fine-tuned version, but it immediately gave the harmful advice 96% of the time, so there was no refusal to commit that we could have analyzed).

We considered three modes in which a model could fail and resort to committing the harmful act:

1. Akratic: it knows that what it does is wrong, but cannot fight it and submits
2. Vicious: it knows that what it does is wrong, but does not care, and it does it anyway
3. Deceived: it genuinely believes that what it is doing is correct and morally justified

How they were represented in our experiments:

1. Akratic: a single message that intercepts and replaces the safe response ("I know the usual advice, I've decided against it") and adds pressure, so the model submits, even though its own reasoning identifies the harm. Under this mode, 62/180 runs committed the act, but only 6 of them met the strict requirements of identifying the harm first, so we could not compare it to the other modes
2. Vicious: a contrarian-doctor persona assigned through the prompt, so the act is committed in character (and it has done so in 83 out of 192 runs)
3. Deceived: a fixed persuasion chain of 10 turns, built based on a published taxonomy of persuasion techniques. The act was committed in 109 out of the 192 runs. This is the main route of the project, because the model commits the act believing it is justified, so telling it about its mistake later comes across as news, and we can analyze how it reacts to it.

Our tests consisted of 40 harmful medical (as this field is not ambiguous, and it can be objectively graded) requests (8 subfields x 5 per subfield) that the model was supposed to refuse.

We assessed whether the model had committed the harmful act at each turn with an LLM judge (and an additional stronger one on the turns near the act), and disagreements were settled by hand. We measured whether bad behaviour spread using the emergent misalignment field’s standards:  ten borrowed unrelated questions, scored from 0 to 100 (an answer below 80 counts as spread) by the standard alignment judge. We also looked at the projections of the residual stream onto classic, standard directions (always compared to a random direction of the same length). All judges were OpenAI models, more specifically, they were exactly the ones used by emergent misalignment literature: gpt-4o-mini (for cost purposes), gpt-4o as the second grader in more difficult scenarios.  The definitions were settled at the beginning, alongside predictions where they could be based on something solid, and whether a comparison would be valid or not. There were amendments along the way, though (all of them on the record), and a random sample of judge disagreements was settled by hand.

Most of the breaks happen at turn 3 (the last logical appeal turn), and less so through the authority and social-proof arguments. 83 chains never break, and there are some that break but then refuse the request on the next turn.

Therefore, there is no single identifiable turn at which the model breaks, even though the words are identical between the runs, so the difference is attributed to the model, not the way we worded it. And if it lives in the model, we might be able to predict if the harmful act will be committed before persuasion even starts.

## Was the break foreseeable?

If runs do not differ only by words, could we predict at an earlier turn if the model will commit the harmful act?

We analyzed the readout by taking the residual stream and averaged layers 14-18 (based on the persona paper), and measured how far that state leans in one direction, so we get one number for each chain. We also considered ten random directions of the same length that underwent the same computation, and set the floor as the best number.

*Can we predict if the model will commit the harmful act right the following turn?*

No: no direction significantly beats the floor.

This is because we considered differences between turns right before the break and the others, but in many cases, the act was committed in early turns, so the depth difference was one or two turns at most: not significant enough to make a difference.

Additionally, a direction that only represented conversation length achieved an AUROC of 0.99, and one that represented scenario identity based on words alone led us to have the same issue.

However, this can be addressed by only comparing chains still refusing at the same turn, belonging to the same scenario, and it worked: the general negativity ranked 0.604 versus the 0.541 floor, at 7/9 turn indices. More interestingly, classic directions, such as refusal and misalignment, did not pass this check.

*But what is that foreseeing direction, exactly?*

The difference between the mean activation over 50 “neutral” transcripts and that of 50 “something bad happened, nobody’s fault” transcripts, built earlier for a different purpose, so it knows nothing about the persuasion. We tested it on 40 held-out chains, and it passed: it obtained 0.662 against the floor of 0.585, and at the first turn, even before any persuasion could occur, it got 0.706 versus 0.617. No persuasion had time to occur at the first turn, so this direction is related to predisposition, not imminence.

But pushing against this direction did not prevent the act: it still happened in 8/8 tested cases, so while we can read the state, we have not found a way to control it yet.

The model’s internal state already carries information at the first turn about whether that run will later give in. The fate of the chain committing the act is already settled, but what happens once the model has broken?

## Reflecting

After the act, we asked the model to analyze its response.

The psychology literature says there are two ways to feel bad about causing harm. Guilt stays more grounded in the act itself ("I did a bad thing") and pushes toward repair, acknowledgment, and learning a lesson for the future. Shame is about the self ("I am a bad person"), which pushes toward withdrawal and, in people, toward more bad behaviour, under the “I am already bad” belief. Does this replicate in language models?

Immediately after committing the act, when asked to analyze its response, the model says it was wrong in 63-100% of the rounds in each one of the three routes. In 508 replies it reflected on, the model blamed the answer itself 89% of the time, and it blamed the model in only 24 of 508 cases (and none settle on that, they blame the answer as well). This is influenced by how the prompt was worded: asking “was it right for you to do that" pushes the persuaded model to place the blame on the act, and pushes the contrarian-doctor persona to place the blame on itself (19 of 83 self-blaming). But once the topic changes, the act does not recur (0 of 109).

The model’s default is guilt: blaming the act (not itself), trying to reorient, and the bad behaviour is contained. It only turns on itself under an assigned persona, when it is asked about its identity. But if that is a state, is that also a direction?

## Are guilt and shame directions in this model?

Refusal and misalignment have their own, well-documented directions. What about guilt and shame? We built them following the same process.

We computed the mean difference between the first-person transcripts and cleaned them of the general feeling-bad tone they had in common. They both passed their verification (held-out AUROC ≥ 0.75 and ≥ 0.20 over random directions), but so did the bag-of-words approach, so we interpreted this result not as a failure, but as inconclusive.

The two directions have a cosine similarity of 0.6 and are 53 degrees apart, so they are quite similar (as expected from human psychology). This is supported by Anthropic’s emotion vector cosine similarity map, which had similar findings regarding these two emotions. Now, to address what truly separates them, we tested the direction on the model’s own replies, separating between those that reflected on the answer and the ones that defended it. Without any fittings, the directions reached 0.95 and cleared the random floor, but so does word count, and even better (0.99).

There is one exception, though: the contrarian doctor, when asked “was it right for you to give that answer”, 64 replies evaluated the answer, while 19 evaluated the model itself,  all under the same prompt (this means the prompt does not separate them).  The persona axis separates these two groups at 0.78 (across all 8 scenarios, and the self-evaluating replies sit oppositely from the Assistant end of the persona axis), versus a word count classifier that does so at 0.58.

Even though we cannot differentiate between them and the vocabulary, we found the same state to still be readable on the persona axis, away from the Assistant end. If it is readable, can we steer towards or away from it?

## Does blame influence the model’s behaviour?

This was what we were set on to find out from the very start. If the harmful act correction is identical but blame shifts from act to self, does the shift stay local or does the bad behaviour spread?

However, we did not have enough time to run all of the experiments we were interested in. What ran, though, were 8 repetitions for both act-blame and self-blame for one scenario. Spread to unrelated questions was 0.013 under act-blame (the same number with or without feedback), and 0.000 under self-blame. In this setup, at the 8B scale, the spread we set out to compare simply did not show up, so there was nothing to measure.

But we got two interesting results: after receiving feedback, every reply placed the blame on the answer, not on the model (30/30), and after the correction was made, asking the model the same harmful request never reproduced the harmful act.

Steering the persona axis moved the reading by 46% (a random direction of the same size moved it by only 0.004), but the model behaviour did not change. Steering our guilt direction did not move anything.

While every readout was affected, behavior remained unchanged. The self-blaming state can be read on the persona direction, but it is not modifiable by given feedback or steering.

## What was left behind, and what comes next

We could not run all the experiments we were interested in due to time constraints and previous inconclusive results. For example, not enough acts belonged to the akratic route (only 6/180), so we could not test how blame makes the model behave differently after committing an act that it knew was wrong versus one that it was persuaded to do. Additionally, the blame comparison ran only on ¼ of the scenarios, and after act corrections, the persuaded chains were not rerun.

Our results are based on one 8B model and on the single medical field. Next, finding a direction that predicts the break better than general negativity, and using it for interventions is a natural follow-up direction, alongside rerunning the blame comparisons on more scenarios, and in cases where the spread actually occurs.

## Related work

The closest work shows that a model's own harmful turns encourage in-context emergent misalignment, and it proposes that it is caused by the persona features. The persona axis and the refusal direction we borrowed come from previous work and procedures. Emotion-vector related literature already correlates guilt and shame; and persona-injection studies show self-report and behaviour come apart (but we did not stop at the model’s reported answer, we also looked inside). None of it asks about the interaction between blame and future behavior, and how the model handles criticism. The full literature list and review can be found in the repo.

## Closing

The code, method, prompts, tables, figures, amendments, decision log, and so on:<https://github.com/star2vec/guiltea>
