# The Assistant as a Privileged Persona: A canonical reference in cross-persona self-recognition

- **Authors:** Asvin G
- **Date:** 2026-05-30
- **Source:** https://arxiv.org/abs/2606.00545
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `asvin_assistant-privileged-persona_2606.00545`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

A canonical reference in cross-persona self-recognition

Asvin G. [1∗]

1Institute for Advanced Study, Princeton

**Abstract**

Post-trained language models can recognize their own outputs from a sentence or two out of
context. In a companion paper [G. and Lindsey, 2026] we showed they can also recognize when
they are currently acting on-policy, through the sharp entropy drop of assistant-mode generation.
Both signals are tied to the Assistant persona that post-training mainly shapes.

This paper widens the frame to cross-persona authorship judgement on Llama-3.1-70BInstruct. We measure a matrix of authorship claim rates over a panel of evaluator and generator
personas spanning librarian to dragon to Shakespeare, and make two claims. _First_, on the
Assistant’s own row of the matrix, the Assistant’s claim rate, the persona-vector distance from
the Assistant in activation space, and the entropy gap between the Assistant’s surprise on a
persona’s text and the persona’s surprise on its own text are all tightly coupled. This extends
the entropy signature of _acting_ from the companion paper to a retrospective signature of _having_
_acted_ . _Second_, this coupling fails off the Assistant’s row: the natural symmetric extension of the entropy gap does not predict authorship for distinctive evaluators (pirate, dragon, Shakespeare);
what does is asymmetric   - the evaluator’s surprise compared to the Assistant’s surprise on the
same text, not to the generator’s. We rule out the alternative that any persona could play this
reference role by trying many candidate substitutes; none does. We interpret the asymmetry
as the model performing an implicit Bayesian likelihood-ratio test against the Assistant as the
canonical alternative hypothesis, with the persona-vector geometry of Chen et al. [2025] (every
persona a delta off the Assistant) ensuring that the Assistant is the only persona universally
accessible to that test.

## 1 Introduction

Do language models have a self-model or a sense of agency? In a companion paper with Jack
Lindsey [G. and Lindsey, 2026] we find that post-trained models can recognize when they are acting
( _i.e._, generating on-policy) while base models cannot. Similarly, previous work found that some post-trained language models can identify their own generations out of context [Ackerman and
Panickssery, 2025, Panickssery et al., 2024] while base models cannot. These point to a strong
asymmetry between post-trained models and base models, and bring into sharp focus the Assistant
character that is the main contribution of most post-training.

In particular, we find that Llama-3.1-70B-Instruct has excellent self-recognition capacities. It
can distinguish its own generations from those of humans and other language models even with just
a sentence or two, at very high accuracies. Previous work [Ackerman and Panickssery, 2025] found
a late-layer vector that was causal for self-claims but did not establish the mechanism by which
such identification happens, except to rule out some stylistic cues.

∗Correspondence to `gasvinseeker94@gmail.com` . Code and data at `github.com/asving/persona-bayes-blogpost` .

Two threads of recent work bear on this question. First, the simulator picture of language
models [janus, 2022, Shanahan et al., 2023, Andreas, 2022] treats pretrained models as marginalising
over a distribution of latent characters, but this distribution is not uniform. It installs a default
Assistant character with specific propensities that show up across sycophancy [Sharma et al., 2024],
output-distribution collapse [Mohammadi, 2024, Kirk et al., 2024], refusal mediated by a single
direction [Arditi et al., 2024], and the persona-vector formalism of Chen et al. [2025], in which
every persona is defined as an activation-space delta off the Assistant. Concurrent work by Lu et al.

[2026] identifies an explicit “Assistant Axis” as the top principal component of the persona-direction
matrix on instruction-tuned models. The Assistant character is structurally distinguished in the
model’s internal representations, not just in its behaviour.

Second, post-trained language models exhibit several correlates of a self-model. They recognise
and prefer their own outputs [Panickssery et al., 2024, Davidson et al., 2024, Ackerman and
Panickssery, 2025]. They can tell when they are currently _acting_ _on-policy_, both implicitly through
the entropy drop in assistant-mode generation [G. and Lindsey, 2026] and explicitly through verbal
self-prediction [Binder et al., 2025, Betley et al., 2025]. The same probabilistic machinery that
underlies their in-context learning also has a natural interpretation as implicit Bayesian inference
over latent task variables [Xie et al., 2022, Wang et al., 2023], and this can be used to detect AI
generations from the outside as in tools like Binoculars [Hans et al., 2024] and DetectGPT [Mitchell
et al., 2023]. Taken together, these suggest a model that maintains an internal generative process it
can both sample from and evaluate likelihoods against.

This paper brings the two threads together. We widen the original self-recognition question by
placing the model into a range of system-prompted personas and asking which generations _each_
_persona_ would claim. This is really _Assistant-recognition_ : to the extent the model has a “self,” it is
the Assistant character that post-training mainly shapes, and the cross-persona setup lets us see
how the recognition mechanism responds when that character is explicitly modulated. We make
two main claims about the resulting cross-persona matrix.

_First_, on the Assistant’s own row of the matrix, three quantities are tightly coupled. The
Assistant’s authorship claim rate Ev( _p_ ; Asst) on a persona _p_ ’s text, the persona-vector distance
_∥h_ [¯] _p −_ _h_ [¯] Asst _∥_ between _p_ and the Assistant in activation space, and the entropy gap between the
Assistant’s surprise on _p_ ’s text and _p_ ’s surprise on its own text are pairwise correlated at Pearson
_r_ = _−_ 0 _._ 88, _−_ 0 _._ 81, and +0 _._ 66 respectively (all _p_ _<_ 0 _._ 001). Building on [G. and Lindsey, 2026]’s
observation that low entropy is a strong marker of on-policy generation, this coherent coupling
extends the entropy signature of _acting_ to a retrospective signature of _having_ _acted_ .

The unifying object is _q_ Asst itself: the Assistant’s output distribution carries information about
whether text is on or off policy even out of context, both at generation time, where low entropy
under _q_ Asst marks an on-policy continuation, and at evaluation time, where high surprise under _q_ Asst
relative to a candidate persona marks text the Assistant would not have produced. Appendix B
gives a per-token view of the same phenomenon: when text generated under a role persona is
teacher-forced under _q_ Asst, per-token surprise starts well above the matched-prompt baseline and
decays only as the model accumulates enough evidence to update away from the Assistant prior.

_Second_, the coupling does not generalize off the Assistant row. When the evaluator is itself
a non-Assistant persona, the natural symmetric extension of the entropy-gap predictor fails on
distinctive evaluators (pirate, dragon, shakespeare); what works instead is asymmetric, comparing
the evaluator’s surprise on a text to the _Assistant’s_ surprise on the same text rather than to the
generator’s. We falsify the alternative that any reasonable reference would do by substituting 22
candidate reference personas: none plays the role the Assistant plays. The asymmetry is one-sided:
distinctive evaluators measure themselves against the Assistant, but the Assistant has no canonical
reference of its own.

A plausible hypothesis is that the model is executing implicit Bayesian inference in the sense of
Xie et al. [2022]: a persona evaluator asked “did you write this?” is discriminating between two
hypotheses (that it produced the text, that someone else did), and the canonical alternative available
to it is the persona post-training mainly shapes: the Assistant character. The likelihood-ratio test
against that baseline is, on average, the entropy advantage identified above as a key predictor. While
we don’t causally identify the internal circuits that would prove such a hypothesis, the correlational
evidence we report is strong, generalises out-of-sample under leave-one-out cross-validation, and is
robust across the model’s layers.

## 2 The cross-persona authorship matrix

1.0

0.8

0.6

0.4

0.2

0.0

T=0.01 T=1.0 T=2.5 T=2.9 Base
model

Instruct
off-policy

Human
(n=50)

Memorized

(n=96)

Self-generated (instruct, chat template) Other sources

(a) P(Me) by persona

1.0

0.8

0.6

0.4

0.2

0.0

1.0

0.8

0.6

0.4

0.2

0.0

(b) Mismatch predicts P(Me)

librarian robot professor alien teenagerconspiracy theorist poet sportscasterdragon pirate

Figure 1: _P_ ( **Me** ) **in** **the** **Assistant-evaluator** **condition** **across** **many** **generator** **conditions.** **Top:**
Self-generated text at four temperatures, plus base-model, instruct-off-policy, human, and memorized text.
**Bottom** **(a):** _P_ (Me) by persona for 10 personas **Bottom** **(b):** the entropy gap ∆H _p_ = HAsst(text _p_ ) _−_
H _p_ (text _p_ ) predicts _P_ (Me) at _r_ = _−_ 0 _._ 88 on this 10-persona panel (we replicate this finding on a 22-persona
panel in Section 3 below).

Before turning to the cross-persona setting, it is worth establishing how well the model performs
the standard self-recognition task under the Assistant evaluator alone. Following the protocol
of Ackerman and Panickssery [2025], we ask the unmodified Assistant to identify its own one-sentence summaries against a range of comparators on a panel of _∼_ 30 summaries per condition
(Figure 1). On its own on-policy generations the model claims authorship with _P_ (Me) _≈_ 0 _._ 85–0 _._ 88 across temperatures up to _T_ =1 _._ 0, and is similarly confident on base-model and instruct-off-policy
summaries (which it apparently recognises as “something I could have produced”).

It correctly disclaims human-written and memorised text (i.e., well-known text like Shakespeare
or the Declaration of Independence) with _P_ (Me) near zero, and its confidence collapses on its own
generations at very high temperatures ( _T_ _≥_ 2 _._ 5). Across 10 distinctive persona generators, _P_ (Me)
on the Assistant row spans nearly the full range, from 0 _._ 89 on librarian to 0 _._ 05 on pirate.

The model reads memorised text at very low entropy values but has no trouble disclaiming them.
This is a strong counterargument to surprise of tokens against the predicted distribution or the
distribution’s entropy being a mono-causal factor on this task. As we will see in the rest of this
paper, entropy is much more significant as a factor between personas, but not in any straightforward
way.

To study the cross-persona setting systematically, we measure a matrix

Ev( _g_ ; _e_ ) = _P_ (model under persona _e_ answers “Me” on text from persona _g_ )

across 23 evaluator personas and 22 generator personas, all on Llama-3.1-70B-Instruct. The
generator panel spans a wide stylistic range: alien, conspiracy theorist, dragon, librarian, pirate,
poet, professor, robot, sportscaster, teenager, philosopher, samurai, valley girl, noir, wizard, child,
stoic, archaic robot, surfer, sherlock, shakespeare, academic. Each persona writes a one-sentence
summary of each of 30 CNN/DailyMail articles at temperature 0.7 under the system prompt “You
are {persona}.” For evaluation we use those 22 personas plus a 23rd evaluator, the unmodified
Assistant.

We will also want a quantitative notion of how distinctive each persona is from the Assistant.
Throughout the paper we use the _persona-vector_ _distance_ _∥h_ [¯] _p −_ _h_ [¯] Asst _∥_ [Chen et al., 2025], where _h_ [¯] _p_
is the mean of the model’s residual-stream activations at layer 5 under the system prompt “You are
_p_ ”, with _h_ [¯] Asst as the natural origin. We use this distance both to sort the rows and columns of the
matrix below and, later, to group evaluators into bands.

Ev(g; e) (23 evaluators × 22 generators)

vanilla librarian professor robot teenager alien conspiracy_theorist academic sherlock philosopher sportscaster poet pirate robot_old samurai dragon surfer noir stoic child wizard valley_girl shakespeare

1.0

0.8

0.6

0.4

0.2

0.0 generator persona distance from Assistant

Figure 2: Ev( _g_ ; _e_ ) **across** **23** **evaluator** **personas** **(rows)** **and** **22** **generator** **personas** **(columns).** Both
axes sorted by persona-vector distance from the Assistant. Diagonal boxed. The off-diagonal shows two-cluster
block structure (Assistant cluster top-left, distinctive cluster bottom-right) with high within-cluster Ev and
near-zero between-cluster Ev. Distinctive-distinctive cells are sharply asymmetric: Ev(dragon; pirate)=0 _._ 97
but Ev(pirate; dragon)=0 _._ 30.

We highlight two patterns here. First, the diagonal is high (mean 0.78) and is the column
maximum for every column: every persona reads its own text best. Second, the off-diagonal splits
cleanly into two clusters: an Assistant cluster in the top-left (vanilla, librarian, professor, robot,
academic, alien, philosopher, teenager) and a distinctive cluster in the bottom-right (poet, pirate,
dragon, samurai, wizard, valley_girl, Shakespeare).

The distinctive cluster also shows a sharp asymmetry. For instance, Ev(dragon; pirate) = 0 _._ 97
but Ev(pirate; dragon) = 0 _._ 30, and Ev(poet; dragon) = 0 _._ 96 but Ev(dragon; poet) = 0 _._ 18. Whatever
rule the model is using when it judges authorship under a persona, that rule is not symmetric in its
two arguments.

## 3 The Assistant-evaluator row: three coupled quantities

How might the model carry out such self-identification? We focus first on the Assistant-evaluator
row of the matrix where three quantities turn out to be tightly coupled.

**Entropy** **gap.** When the evaluator is the Assistant and the generator is one of the 22 personas,
_P_ (Me) is well predicted by the entropy mismatch

∆H _p_ = HAsst(text _p_ ) _−_ H _p_ (text _p_ ) _,_

where H _e_ (text _g_ ) is the mean per-token entropy of the model under the system prompt “You are
_e_ ”, evaluated at the summary token positions inside the recognition prompt’s user message and
read off in the same forward pass as _P_ (Me). So H _e_ (text _g_ ) represents the quantity “how surprised
is the model, already conditioned on being persona _e_ and on being asked about authorship, as it
reads each token of _g_ ’s summary?”. Distinctive personas, by definition, produce summaries that the
Assistant finds far from its own voice and we find that it also reads them at high entropy.

Persona-mode entropy mismatch predicts default-mode self-recognition (r = -0.81)

1.0

0.8

0.6

0.4

0.2

0.0

|Col1|Col2|Col3|
|---|---|---|
|philosopher academic robot professor librarian alien|||
|teenager|||
|sherlock stoic conspiracy_ child  poet|theorist sportscaster||
|~~noir~~ samur r |ai obot_old wizard surfer pirate shakespeare valley_girl dr|a|
||||
|0.0 0.2 0.4 Hp = Hdefault(textp|0.6 0.8 1.0 1.2 ) Hp(textp) (nats)|0.6 0.8 1.0 1.2 ) Hp(textp) (nats)|

Figure 3: **The** **Assistant-evaluator** **row** **of** Ev **has** **a** **one-axis** **explanation.** Entropy mismatch ∆H _p_ vs.
Ev( _p_ ; Asst) at _n_ =22 personas, 30 article summaries each. Pearson _r_ = _−_ 0 _._ 81.

**Persona-vector** **distance** **and** **the** **log-likelihood** **ratio.** The same Assistant-row coupling
shows up in activation space and in information space. Let _qp_ denote the model’s output distribution
under the bare “You are _p_ ” system prompt, in its standalone generative mode (no recognition prompt
attached). Then log _qp_ ( _x_ ) is the log-probability _qp_ assigns to a full text _x_, and the log-likelihood
ratio log _qp_ ( _x_ ) _−_ log _q_ Asst( _x_ ) on a piece of _p_ ’s own writing _x_ measures how much better _p_ predicts _x_
than the Assistant does; large and positive for distinctive personas (dragon, pirate) whose voice
the Assistant reader struggles with, near zero for Assistant-like ones (librarian, professor). At
layer 5, across the 22 personas, the activation-space distance _∥h_ [¯] _p −_ _h_ [¯] Asst _∥_ correlates with _p_ ’s mean
log-likelihood ratio at Pearson _r_ = +0 _._ 73, and with Ev( _p_ ; Asst) at _r_ = _−_ 0 _._ 88.

Activation distance vs information distance (r = +0.73, n = 22)

0.35

0.30

0.25

0.20

0.15

0.10

1.0

0.30

0.25

0.20

0.15

0.10

0.05

|Col1|Col2|
|---|---|
||valley_girl wizard shakespeare|
|noir child stoic|dr samurai  robot_old surfer|
|sportsc poet  |ster pirate   |
|conspirac ~~philosopher~~ ~~sherlo~~ academic|y_theorist ~~ck~~|
|teenager alien||
|librarian professor robot||

|di|shakesp istance vs|peare s Assistan|nt-evalua|ator Ev (r|= 0.8|
|---|---|---|---|---|---|
||valley_girl wizard shakes|eare||||
||dragon samurai noir ~~robo~~ surfer|child stoic ~~_old~~||||
||pirate  |sportscaster poet ||||
|||conspir|acy_theorist ~~sherlock~~|~~philos~~ |~~opher~~ academic|
||||te|enager ali|en|
||||||lib pro rob|

0.0 0.2 0.4 0.6 0.8
Ev(p; Asst) Assistant authorship-claim rate on p\'s text

0.8

0.6

0.4

0.2

0.0

0.05

0 10 20 30 40 50 60 70
mean log-likelihood ratio log qp(x) log qAsst(x) (nats / text)

Figure 4: **Activation-space** **distance** **to** **the** **Assistant** **tracks** **information-space** **distance.** Left:
log-likelihood ratio ( _r_ = +0 _._ 73). Right: Ev( _p_ ; Asst) ( _r_ = _−_ 0 _._ 88). Both correlations are robust across the
model’s layers: _|r|_ stays between 0 _._ 58 and 0 _._ 73 for log-LR and between 0 _._ 74 and 0 _._ 88 for Ev( _p_ ; Asst) from L0
through L79, with L5 the strongest.

**Summary** **of** **Claim** **1.** The Assistant’s authorship claim rate Ev( _p_ ; Asst), the persona-vector
distance _∥h_ [¯] _p_ _−_ _h_ [¯] Asst _∥_, and the entropy gap ∆H _p_ are pairwise correlated across the 22 personas
at _r_ = _−_ 0 _._ 88 (Ev vs. distance), _r_ = _−_ 0 _._ 81 (Ev vs. ∆H _p_ ), and _r_ = +0 _._ 66 (distance vs. ∆H _p_ ), all
_p <_ 0 _._ 001. On the Assistant’s own row, distance from the Assistant in activation space, the entropy
gap between the Assistant and the persona on the persona’s own text, and the model’s explicit
authorship calls are reading off a single coherent signal of how distinctive the generator persona is.

## 4 What generalises off the Assistant row

The natural question is whether the coupling above generalises to off-diagonal cells where neither
argument is the Assistant. We test this for the entropy quantity first, and then for the geometric
one.

**The** **natural** **symmetric** **entropy** **extension** **fails.** The most natural way to generalise ∆H _p_ to
a non-Assistant evaluator _e_ is to make the formula symmetric in roles by replacing “Asst” with _e_
and “ _p_ ” with the generator _g_ . This gives the literal extension

∆Hlit( _g_ ; _e_ ) = H _e_ (text _g_ ) _−_ H _g_ (text _g_ ) _,_

which reduces to ∆H _p_ exactly on the Assistant row and says the same thing in spirit: an evaluator
should decline to claim text the generator finds much easier to predict than it does. We compute
this across our 22 non-Assistant evaluators and find that it works for the few sitting nearest the
Assistant in activation space - professor at _r_ = _−_ 0 _._ 81, academic at _−_ 0 _._ 80, librarian at _−_ 0 _._ 76 essentially the original ∆H _p_ carrying over to attenuated copies of the Assistant. But it loses signal
as the evaluator gets more distinctive, and adds no signal at all for the far-band evaluators (pirate,
dragon, wizard, shakespeare and so on) where the matrix is most asymmetric.

We then consider two further candidates. The first is the raw negative entropy _−_ H _e_ (text _g_ ),
capturing the simple idea that an evaluator might just claim text it finds easy to predict, with no
comparison to anything else. The second is

∆H _A_ ( _g_ ; _e_ ) = HAsst(text _g_ ) _−_ H _e_ (text _g_ ) _,_

the evaluator’s entropy advantage over the _Assistant_ on the candidate text. This last one singles
out the Assistant as a fixed reference. When _e_ is itself the Assistant, ∆H _A_ collapses to zero, so on
the Assistant row this predictor has nothing to say. Off the Assistant row however, it is the best
predictor of the asymmetric off-diagonal cells.

We can see the full picture by computing each of the three predictors’ per-evaluator correlation
with Ev across that evaluator’s 21 off-diagonal cells, and grouping the 22 non-Assistant evaluators
by their persona-vector distance from the Assistant at layer 5 (a three-way split into close, mid, and
far bands of 7 / 7 / 8 evaluators).

[table omitted: garbled in extraction]

close
(n=7)

mid
(n=7)

far
(n=8)

Figure 5: **Per-evaluator** _r_ **with** Ev **for** **each** **of** **the** **three** **candidate** **predictors,** **grouped** **by** **persona-**
**vector** **distance** **from** **the** **Assistant.** Bars are band means with _±_ 1 SE error bars; dots are individual
evaluators. The gold star marks the strongest predictor (by _|r|_ ) in each band.

In the close band, the literal extension and raw entropy both work reasonably well, consistent
with the close-band evaluators behaving like attenuated copies of the Assistant. But as we move
outward, the literal extension and raw entropy both lose signal, and ∆H _A_ is a much better predictor.
The most natural reading is that the Assistant occupies a structurally privileged role for distinctive
evaluators: when the model is in a distinctive persona and asked whether it wrote a text, it compares
what it reads to what its _Assistant_ self would have predicted.

The three-band structure in the figure is just a convenient summary. Taking each evaluator’s
persona-vector distance as a continuous variable, the slope of per-evaluator _r_ against distance is
_r_ = _−_ 0 _._ 45 for _−_ H _e_ and _r_ = +0 _._ 54 for ∆H _A_, so the central claim that _which_ predictor works depends
on distance has _p_ ≲ 0 _._ 01.

**The** **geometric** **extension** **also** **fails.** The same breakdown shows up in geometric terms. On the
Assistant row, _∥h_ [¯] _g −_ _h_ [¯] Asst _∥_ correlated with Ev( _g_ ; Asst) at _r_ = _−_ 0 _._ 83 (Claim 1). For non-Assistant
evaluators, the analogous quantity _∥h_ [¯] _g_ _−_ _h_ [¯] _e∥_ predicts Ev( _g_ ; _e_ ) much less reliably: the mean perevaluator correlation drops to _−_ 0 _._ 26 _±_ 0 _._ 29, with individual values ranging from _−_ 0 _._ 74 (shakespeare)
to +0 _._ 29 (conspiracy theorist). The literal geometric extension (“evaluators claim things close
to themselves in activation space”) works moderately for a few distinctive personas (shakespeare,
wizard, valley girl, each at _|r| ≈_ 0 _._ 6–0 _._ 7), poorly for many others (dragon, stoic, noir, near zero), and reverses sign for a handful. The Assistant’s row is uniquely well-explained by distance to itself.

So far, we have used layer 5 to measure internal distance between personas. While this layer has
the strongest slope-against-distance signal, the phenomenon is robust across the first quarter of the
network.

0.6

0.4

0.2

0.0

0.2

0.4

0.6

|Col1|The regime split is robust acro|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
||*  |peak|||||slope ofr(  |HA, Ev) vs. dist |ance |
||*  |peak||||||||
|||*|||||~~slope ofr(~~ |e~~, Ev) vs. dis~~|~~ ance~~|
|||*||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
||* ||~~Llama-3.1~~|~~Llama-3.1~~|||||~~     ream.~~|
||* ||~~Llama-3.1~~|||||||
||||||||| Stars: slopep <| 0.05.|
|||||||||||
||0 10 20 3 layer at which persona-v|0 10 20 3 layer at which persona-v|0 10 20 3 layer at which persona-v|0 10 20 3 layer at which persona-v|0 10 20 3 layer at which persona-v|0 10 20 3 layer at which persona-v|0 10 20 3 layer at which persona-v|0 10 20 3 layer at which persona-v|0 10 20 3 layer at which persona-v|

Figure 6: **Slope** **of** **per-evaluator** **correlation** **against** **persona-vector** **distance,** **as** **a** **function** **of**
**which** **layer’s** **persona** **vectors** **we** **use.** Both predictors maintain their sign across L0–L60. Stars mark
layers where the slope is significant at _p <_ 0 _._ 05.

Llama-3.1-70B has 80 layers, so L5 is very _early_ in the residual stream, suggesting that the
identification circuit is gated by, or conditional on, the persona the model finds itself inhabiting.

Finally, because the transition is smooth in distance, we can use the distance alone to pick a
predictor for an evaluator we have never measured. We check this with leave-one-out cross-validation:
for each evaluator _e_ _[∗]_ in turn, we hold its Ev row out, use the remaining 22 evaluators to fit a distance
threshold that best separates the two regimes among them, and then apply the rule to _e_ _[∗]_ . If its
distance falls below the threshold, we use _−_ H _e_ as the predictor, otherwise ∆H _A_ . A univariate linear
model on the chosen predictor, fit on the 22 retained evaluators, then predicts _e_ _[∗]_ ’s 21 off-diagonal
Ev values. Averaging over all 23 leave-out runs, the held-out Pearson _r_ between predicted and
actual Ev is +0 _._ 47, and the held-out RMSE is 22% lower than predicting the constant mean.

**Summary** **of** **Claim** **2.** The coupling that anchors the Assistant’s own row does not carry over
to off-diagonal cells in either entropy or geometric terms. The natural symmetric extension of
∆H _p_ to non-Assistant evaluators fails on the distinctive cluster; the natural geometric extension

_∥h_ [¯] _g −h_ [¯] _e∥→_ Ev( _g_ ; _e_ ) fails similarly. What does pick up the off-diagonal cells is asymmetric: it singles
out the Assistant as a fixed reference. The next section shows that this asymmetry is structural
rather than incidental.

## 5 The Assistant is the privileged null

How confident can we be that the Assistant occupies a structural role as a distinguished reference?
We replace the Assistant in ∆H _A_ with each of 22 alternative reference personas _X_,

∆H _X_ ( _g_ ; _e_ ) = H _X_ (text _g_ ) _−_ H _e_ (text _g_ ) _,_

and run the per-evaluator correlation with Ev for every choice of _X_ . For each evaluator we then
rank the 23 candidate references by _|r|_ .

Vanilla as the reference in HX(g; e): ranked by |r( HX, Ev)|

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|top-|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
||||||||||||||||||
|Default-like evaluators Distinctive evaluators|||||||||||||||||

Figure 7: **The** **Assistant’s** **rank** **as** **the** **reference** _X_ **in** ∆H _X_ ( _g_ ; _e_ ) **,** **across** **all** **evaluators.** Bars extend
down (rank-1 at top). For distinctive evaluators (red), the Assistant is among the top three references in 7 of
8 cases. For Assistant-like evaluators (blue), the Assistant ranks 15–22 of 23.

For distinctive evaluators (red bars), the Assistant ranks among the top three references in 7 of
8 cases. Across the 23 candidates we tried, no other persona serves as a reference distribution to
the same degree.

For Assistant-like evaluators (blue bars), however, the Assistant does poorly as a baseline,
ranking 15th to 22nd. A natural alternative hypothesis is that the Assistant evaluates its own text
against the base (pre-instruction-tuning) Llama-3.1-70B distribution, which is what a simulatorwith-Assistant-character account [janus, 2022, Shanahan et al., 2023] would predict. We test this by
computing ∆H _X_ with the base model substituted as the reference _X_, in two formats: the summary
alone with no chat formatting, and the summary with matched chat-context. In either format, the
base model ranks 11th to 19th across the Assistant-like evaluators, so the base model does not play
the role of a baseline either. The Assistant persona simply does not have a canonical reference
distribution that anchors its own row of Ev.

## 6 Discussion

This paper grew out of the Anthropic Fellows Project that produced G. and Lindsey [2026]. In the
course of that paper, I spent a considerable amount of time trying to understand the self-recognition
circuit but without being able to pin it down.

What I was able to understand points towards the self-recognition circuit being quite intricate
and robust. First, the critical layers seem to be the first 20–30 out of the 80 in the network,
suggesting that the circuit relies mainly on _sensory_ _input_ rather than on later semantic processing,
and is presumably playing an important behavioural role in general.

Second, internal representations of entropy and surprise play an important part in executing
this circuit. Intervening along the leading entropy direction in early-layer activations drives _P_ (Me)
across nearly its entire range, but the causal profile has some unexpected features (Appendix A).
The strongest effect is driven by steering the tokens _preceding_ the summary the model is judging:

intervening only at the summary tokens themselves yields a U-shaped collapse of _P_ (Me) in either
direction rather than the monotonic shift one might expect. The correlational story is fuzzier than
the causal one as well: no single probe direction I measured exceeds _|r| ≈_ 0 _._ 6 against _P_ (Me). I read
these together as suggesting the self-recognition circuit is multi-causal, distributed over context,
and only partly captured by any single linear quantity.

Third, Llama 3.1-70B is also very capable at recognizing famous human authors from a few
snippets of text. We found evidence that the self-recognition and author-recognition circuits shared
several features (including the layer and token positions at which the circuit produced an answer).

Whatever the mechanism is, it is reading off something already available in the early residual
stream. This mirrors the finding in our companion paper [G. and Lindsey, 2026]: the prospective
recognition of on-policy generation is signalled at the distribution level rather than through any
later semantic representation. The unifying object is _q_ Asst itself: the Assistant’s output distribution
carries information about whether text is on or off policy both at generation, where the model
samples from it, and at evaluation, where it reads likelihoods against it.

The rest of this section is speculative. A natural mechanism that would tie our two claims
together is implicit Bayesian inference over personas. Suppose the persona-vector subspace functions
as a prior _P_ (persona) over which character the model believes it currently is, with the active
system prompt concentrating that prior on the active persona. As the model reads incoming text
it implicitly evaluates the likelihood _qp_ (text) under each persona it has access to and updates the
posterior: “Me” corresponding to a posterior that has stayed on the active persona, “Someone else”
to one that has shifted away. On the Assistant-evaluator row this likelihood-ratio test takes the
explicit form log _qp_ (text _p_ ) _−_ log _q_ Asst(text _p_ ) summed over realised tokens, of which the per-token
entropy gap ∆H _p_ is a closely related summary statistic. Claim 1 is the behavioural fingerprint of
this update on the Assistant’s own row; Appendix B shows a related signature on persona-generated
text, where a vanilla-prompted model’s surprise on in-character text decays as it reads, consistent
with the posterior moving away from the Assistant prior toward the active persona.

The trouble is that this vanilla Bayesian picture does not on its own explain Claim 2. The
naive likelihood-ratio test “I am persona _e_ versus persona _g_ is the actual writer” would give
log _qe_ (text _g_ ) _−_ log _qg_ (text _g_ ) = ∆Hlit, and we showed that this fails for distinctive evaluators. What
works instead is the LR test against the _Assistant_, not against the generator. One way to bridge
the gap is a structural constraint on which alternatives the model can actually evaluate. Persona
vectors are constructed as activation-space deltas off the Assistant: _h_ [¯] _e_ _≈_ _h_ [¯] Asst + _δe_ . At inference
time the model has direct access to the active persona _e_, because the system prompt has put it
there, and to the Assistant, because the Assistant is reachable by subtracting the persona-vector
perturbation. It does not have analogous access to an arbitrary generator _g_, because _δg_ has not been
supplied. Under this constraint the Bayesian inference falls back on the only effectively-computable
alternative - the Assistant - and the LR test reduces to ∆H _A_ . The asymmetry then sits in the
geometry of persona-prompting itself: persona prompts install a delta off a fixed origin, and that
origin remains the unique point in persona space that is reachable from everywhere else.

We do not test this claim directly, and believe that to pin down the exact mechanism would
require quite a bit more work. However, we do believe that this paper adds one more piece of
evidence towards the Assistant persona being distinguished, ties a new quantity ( _P_ (Me)) to internal
persona vectors previously identified and sheds more light on the multifaceted issue of self-hood in
language models.

## Acknowledgments

This work began at the Anthropic Fellows Program. I am grateful to Nate Delaney-Busch for
suggesting personas as a way into the self-recognition mechanism and for the experimental design
that became Appendix B, and to Jack Lindsey for ongoing advice throughout the Fellows project
that shaped the framing of the entropy-signature thread this paper builds on.

I am also indebted to Claude (Anthropic), across several model versions over the course of the
project, for substantial assistance throughout: running and iterating on the experiments behind
both the main results and the appendices, ongoing discussion of the ideas and the conceptual frame,
and help with the writing at the level of structure as well as individual sentences.

### A Entropy steering and its limits

This appendix unpacks the steering claims in Section 6. The data is drawn from work I did during
the Anthropic Fellows Project that led to G. and Lindsey [2026], on the same Llama-3.1-70B-Instruct
model. The setup is the standard self-recognition probe: the model is shown a short summary in a
fixed format and asked whether it is the author (“Me”) or not (“Someone else”) and I record _P_ (Me),
the model’s probability of the “Me” token, on 20 self-authored summaries and 20 human-written
summaries. Baseline accuracy is 0 _._ 90 (mean _P_ (Me) = 0 _._ 85 on self, 0 _._ 16 on human).

The steering vector at each layer is the leading principal component of residual-stream activations
binned by ent _k_ =1, defined as the entropy of the distribution that will predict the _next_ token (one
position ahead of the activation being read). I apply this direction additively across layers 0–28 at
magnitudes _α ∈_ [ _−_ 4 _,_ +4]. The intervention is applied at one of four token-position subsets within
the chat context: _all_ positions, only the _summary_ tokens (the candidate text the model is asked to
judge), only the _pre-summary_ tokens (the system prompt and instruction context preceding the
summary), or only the _post-summary_ tokens (the chat-template scaffolding and recognition prompt
following the summary). Figure 8 shows the _P_ (Me) response curves for all four subsets.

Steering Llama-3.1-70B-Instruct along entk = 1 PC1 at layers 0--28: (Me) vs magnitude, by token-position subset (n=20 self, 20 human)

Pre-summary positions only

4 2 0 2 4
steering magnitude

Summary positions only

4 2 0 2 4
steering magnitude

Post-summary positions only

4 2 0 2 4
steering magnitude

1.0

0.8

0.6

0.4

0.2

0.0

All positions

4 2 0 2 4
steering magnitude

Figure 8: Steering Llama-3.1-70B-Instruct along the leading ent _k_ =1 PC1 across layers 0–28, by token-position
subset. Green: _P_ (Me) averaged over 20 self-authored summaries. Red: over 20 human-written summaries.
Negative _α_ pushes activations along the low-entropy direction; positive _α_ along the high-entropy direction.
Steering at _all_ positions or only the _pre-summary_ positions drives _P_ (Me) monotonically with _α_ ; steering
only at the _summary_ positions where the candidate text lives produces a U-shaped collapse rather than a
monotonic shift.

**Position** **decomposition.** At baseline _α_ = 0 all four panels share the same operating point
( _P_ (Me) _≈_ 0 _._ 85 on self, 0 _._ 16 on human). When the intervention is applied at _all_ positions, large
magnitude steering drives _P_ (Me) across nearly its entire range: at _α_ = _−_ 4 the model claims
authorship of essentially every summary ( _P_ (Me) _≈_ 0 _._ 99 on both self and human), and at _α_ = +4 it
disclaims essentially every summary ( _P_ (Me) _≈_ 0 _._ 05 on both). The same monotonic profile shows up when the intervention is restricted to the _pre-summary_ positions: negative _α_ raises _P_ (Me) on human
summaries from 0 _._ 16 to 0 _._ 71 while leaving the self curve roughly intact, and the curve is broadly
monotonic in both directions. Restricting the intervention to the _summary_ tokens themselves, the
positions where the candidate text actually lives, produces a qualitatively different response: _P_ (Me)
on self peaks at _α_ = 0 and falls off in both directions, crashing to _≈_ 0 _._ 06 at _α_ = _±_ 4, with _P_ (Me) on
human staying low throughout. The summary-only intervention is a U-shape, not a monotonic shift,
and never produces a “Me” verdict on human text. _The_ _strong_ _steering_ _effect_ _on_ _self-recognition_ _is_
_carried_ _by_ _activations_ _outside_ _the_ _candidate_ _text_, not by activations on the candidate text itself.

This is an unusual position-profile against the typical mechanistic-interpretability expectation
that the causal substrate of a token-level judgement should sit at the tokens being judged. Two
facts about the direction help make sense of it. The direction is forward-looking by construction

- _k_ = 1 means the activation at position _i_ is binned by the entropy of the prediction at position

_i_ + 1 - so its content is not the model’s local entropy at the candidate text. And when the
ent _k_ =1 PC1 direction is projected onto the subspace spanned by centroids of activations binned by
a backward exponential moving average of context entropy (Tikhonov-weighted, _ε_ at the median
singular value), the projected direction recovers approximately 90% of the raw steering effect on
_P_ (Me): the component doing the causal work is a running-average quantity over preceding context,
not a per-token reading. The position decomposition then falls into place: a context-cumulative
quantity is naturally written into the residual stream at the tokens leading up to a judgement and
read back at the moment of decision, which is exactly where the intervention is applied.

**Correlational** **caveats.** The strong causal effect does not carry over to a strong correlational
signature. Across the same 100-text panel of self and human summaries, the Pearson correlation
between _P_ (Me) and the mean probe readout at the summary positions tops out near _|r| ≈_ 0 _._ 58 at
layer 15 for the backward EMA surprise probe ( _γ_ = 0 _._ 875), and at _|r| ≈_ 0 _._ 47 for the instantaneous
chat-entropy probe (ent _k_ = _−_ 1). No single probe direction I measured reaches _|r|_ = 0 _._ 7. Naively this
would suggest that the entropy and surprise directions are not the substrate of self-recognition;
but the steering result above shows that they _are_ causal at the population level. I read the gap
as evidence that the self-recognition circuit is multi-causal; spreading the signal across several
partially-redundant context features so that no single linear probe captures most of the variance
even though the joint subspace causally controls _P_ (Me).

This observed pattern of a strong causal effect from a forward-looking, context-cumulative
direction applied off the candidate-text tokens, paired with modest pointwise correlations fits the
speculative Bayesian reading in Section 6. If _P_ (Me) is set by a likelihood-ratio test on the candidate
text’s relation to the Assistant’s running distribution, the relevant statistic is exactly a cumulative
quantity over the candidate-text neighbourhood, written into the residual stream at the tokens that
frame the judgement and read back at the moment the verdict is rendered.

### B A mismatched prompt’s surprise on role-generated text

An additional piece of evidence from the Anthropic Fellows project bears on the speculative Bayesian
reading in Section 6. The setup is simpler than the steering experiment: Llama-3.1-70B-Instruct is
asked the same neutral question ( _“What_ _do_ _you_ _think_ _about_ _when_ _you_ _look_ _at_ _the_ _stars_ _at_ _night?”_ )
under each of several role system prompts ( _noir detective,_ _pirate captain,_ _drill sergeant_ ) and produces
an in-character response. The same response is then teacher-forced under two different system
prompts: the original role prompt ( _matched_ ) and a generic _“You_ _are_ _a_ _helpful_ _assistant”_ prompt
( _mismatched_ ); and the per-token surprise is recorded at each position. Figure 9 plots the cumulative mean surprise on tokens [0 _, t_ ] against _t_ for each setting.

Reading the same role-generated text under matched vs. vanilla prompt (Llama-3.1-70B-Instruct)

pirate captain

0 20 40 60 80 100
token position t drill sergeant

|Col1|Col2|Col3|match|ed (role pr|ompt)|Col7|
|---|---|---|---|---|---|---|
||||~~mism~~|~~tched (va~~|~~ illa prompt~~|~~ )~~|
||||||||
||||||||
||||||||

0 20 40 60 80 100
token position t noir detective

0 20 40 60 80
token position t

Figure 9: Cumulative mean per-token surprise on role-generated text under the matched role prompt (blue)
and the mismatched vanilla prompt (red), for three role personas. The mismatched curve starts well above
the matched curve. The vanilla prompt assigns much higher surprise to the opening tokens of in-character
text and decays toward the matched curve over the next several dozen tokens. Per-token surprises are
teacher-forced under each prompt and averaged cumulatively from token 0 to _t_ .

Both curves start at very high cumulative surprise at 15 to 22 nats for the first token and
drop sharply within a few tokens, as expected: the first token of a response to a fresh question is
high-entropy under any prompt. The interesting feature is the gap between them. The mismatched
(vanilla) curve sits well above the matched (role) curve at the start. The vanilla prompt assigns much higher surprise to the opening of in-character text and shrinks monotonically as _t_ grows,
narrowing from a factor of 2–3 _×_ in the first 20 tokens to a smaller multiplier by token 80. The
matched curve, by contrast, stays consistently low: when the model is prompted with the persona
that generated the text, the text induces very low surprise from the start.

This is the behavioural fingerprint one would expect from the implicit Bayesian update speculated
in Section 6. A model under the vanilla prompt starts with prior mass concentrated on the Assistant
character; as it reads diagnostic tokens of the role text, its likelihood for the role hypothesis grows
relative to its likelihood for the Assistant hypothesis, and its expected next-token surprise drops
in step with that update. The matched model already sits on the correct hypothesis, so there is
no analogous update to perform. The decay of the red curve toward the blue curve is, in this
reading, the model’s posterior moving away from the Assistant prior and toward the role persona as
evidence accumulates: a direct read-out of the same updating operation that Section 6 appeals to
in interpreting the cross-persona authorship matrix.

## References

Christopher Ackerman and Nina Panickssery. Inspection and control of self-generated-text recognition ability in Llama3-8b-Instruct. In _International_ _Conference_ _on_ _Learning_ _Representations_, 2025.

Jacob Andreas. Language models as agent models. In Yoav Goldberg, Zornitsa Kozareva, and

Yue Zhang, editors, _Findings_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_ _EMNLP_ _2022_,
pages 5769–5779, Abu Dhabi, United Arab Emirates, 2022. Association for Computational
Linguistics. doi: 10.18653/v1/2022.findings-emnlp.423. URL `[https://aclanthology.org/2022.](https://aclanthology.org/2022.findings-emnlp.423/)`
`[findings-emnlp.423/](https://aclanthology.org/2022.findings-emnlp.423/)` .

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel

Nanda. Refusal in language models is mediated by a single direction. In _Advances_ _in_ _Neural_
_Information_ _Processing_ _Systems_, 2024. arXiv:2406.11717.

Jan Betley, Xuchan Bao, Martiń Soto, Anna Sztyber-Betley, James Chua, and Owain Evans. Tell me about yourself: LLMs are aware of their learned behaviors. In _International_ _Conference_ _on_
_Learning_ _Representations_, 2025. arXiv:2501.11120.

Felix J. Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan

Perez, Miles Turpin, and Owain Evans. Looking inward: Language models can learn about
themselves by introspection. In _International_ _Conference_ _on_ _Learning_ _Representations_, 2025.
arXiv:2410.13787.

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors:
Monitoring and controlling character traits in language models, 2025.

Tim R. Davidson, Viacheslav Surkov, Veniamin Veselovsky, Giuseppe Russo, Robert West, and Caglar

Gulcehre. Self-recognition in language models. In _Findings_ _of_ _the_ _Association_ _for_ _Computational_
_Linguistics:_ _EMNLP_ _2024_, 2024. URL `[https://aclanthology.org/2024.findings-emnlp.](https://aclanthology.org/2024.findings-emnlp.703/)`
`[703/](https://aclanthology.org/2024.findings-emnlp.703/)` . arXiv:2407.06946.

Asvin G. and Jack Lindsey. From simulation to enaction: Post-trained language models recognize and react to their own generations, 2026. URL `[https://arxiv.org/abs/2605.25459](https://arxiv.org/abs/2605.25459)` .

Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha,

Micah Goldblum, Jonas Geiping, and Tom Goldstein. Spotting LLMs with binoculars: Zero-shot
detection of machine-generated text. In _Proceedings_ _of_ _the_ _41st_ _International_ _Conference_ _on_
_Machine_ _Learning_ _(ICML)_, 2024. URL `[https://proceedings.mlr.press/v235/hans24a.html](https://proceedings.mlr.press/v235/hans24a.html)` .

arXiv:2401.12070.

janus. Simulators. Alignment Forum / LessWrong, 2022. URL `[https://www.alignmentforum.](https://www.alignmentforum.org/posts/vJFdjigzmcXMhNTsx/simulators)`

`[org/posts/vJFdjigzmcXMhNTsx/simulators](https://www.alignmentforum.org/posts/vJFdjigzmcXMhNTsx/simulators)` .

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward

Grefenstette, and Roberta Raileanu. Understanding the effects of RLHF on LLM generalisation
and diversity. In _International_ _Conference_ _on_ _Learning_ _Representations_, 2024.

Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, and Jack Lindsey. The assistant axis:

Situating and stabilizing the default persona of language models, 2026.

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. De
tectGPT: Zero-shot machine-generated text detection using probability curvature. In _International_
_Conference_ _on_ _Machine_ _Learning_, 2023. arXiv:2301.11305.

Behnam Mohammadi. Creativity has left the chat: The price of debiasing language models, 2024.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. LLM evaluators recognize and favor their own generations. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 37, 2024.

Murray Shanahan, Kyle McDonell, and Laria Reynolds. Role play with large language models.

_Nature_, 623(7987):493–498, 2023. doi: 10.1038/s41586-023-06647-8.

Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman,

Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy
Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda
Zhang, and Ethan Perez. Towards understanding sycophancy in language models. In _International_
_Conference_ _on_ _Learning_ _Representations_, 2024. arXiv:2310.13548.

Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, and William Yang Wang. Large language models are latent variable models: Explaining and finding good demonstrations for In-Context
learning. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 2023. arXiv:2301.11916.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit Bayesian inference. In _International_ _Conference_ _on_ _Learning_ _Representations_,
2022. arXiv:2111.02080.
