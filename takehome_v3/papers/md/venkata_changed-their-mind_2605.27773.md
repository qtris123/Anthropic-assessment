# Do Models Know Why They Changed Their Mind? Interpretability and Faithfulness of Chain-of-Thought Under Knowledge Conflict

- **Authors:** Pruthvinath Jeripity Venkata
- **Date:** 2026-05-26
- **Source:** https://arxiv.org/abs/2605.27773
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `venkata_changed-their-mind_2605.27773`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Pruthvinath Jeripity Venkata

Independent Researcher
jvpnath@gmail.com

**Abstract**

When a language model is shown a document that contradicts what it learned during training, it must choose:
follow the document or trust its own knowledge. Prior work [Jeripity Venkata, 2026] proved that this choice
depends on how well-known the fact is: famous facts resist the document; obscure facts follow it. Newer
models can show their “thinking” step by step (chain-of-thought, or CoT), and practitioners increasingly
read this reasoning to understand _why_ the model chose what it chose. But does the stated reasoning actually
reflect the real mechanism? We introduce _introspective faithfulness_ and test it across 200 factual questions,
8 models, and 4 prompt conditions. Three of the models expose internal reasoning tokens; we also run the
same models with reasoning disabled (matched controls), so any difference can be attributed to the reasoning
feature itself rather than to the model family. Each model is asked to rate its own confidence from 1 to 5 as
part of its response. We find that CoT reasoning is highly stable across opposite decisions: when we strip
surface-level framing phrases and compare the reasoning a model writes for opposite conclusions on the
same question, flip pairs retain 96% of same-answer similarity (cosine similarity 0.784 vs. 0.817; _d_ = 0 _._ 34;
confirmed by ROUGE-L, _d_ = 0 _._ 45). Yet a faint genuine signal exists in the self-rated confidence scores:
for obscure facts where entity fame is uninformative (Spearman _ρ_ = _−_ 0 _._ 005, _p_ = 0 _._ 81), the model’s own
confidence rating still weakly predicts its decision ( _p <_ 0 _._ 001) and tracks whether it actually knew the fact
( _r_ = 0 _._ 134). This signal is model-specific: GPT-4o is the only model with statistically reliable reasoningdecision coupling (∆= _−_ 0 _._ 132, 95% CI [ _−_ 0 _._ 188 _, −_ 0 _._ 077]), while Claude Sonnet 4.6 uses the widest range
of self-rated confidence among the eight models we tested (SD = 1 _._ 39), yet its pooled correlation across
conditions is near zero ( _r_ _≈_ 0, 95% CI [ _−_ 0 _._ 10 _,_ +0 _._ 06]) because the confidence-decision relationship reverses
between conditions. A matched-temperature ablation (GPT-4o at _T_ = 1) confirms this is model-specific, not
temperature-driven. For native-reasoning models, internal thinking tokens show greater decision-sensitivity
than user-facing CoT (Wilcoxon _p_ = 0 _._ 033), suggesting the stated output smooths over differences the internal
process preserves. These results decompose CoT into a **knowledge** **display** that remains highly similar
regardless of the decision ( _∼_ 96% of same-answer similarity retained) and a thin **confidence layer** with weak
but real signal. For monitoring: the logical argument is the wrong thing to read; the confidence expressions
are where genuine self-knowledge lives.

## 1 Introduction

Chain-of-thought (CoT) prompting [Wei et al., 2022]
was developed to improve accuracy, but its most consequential use may be explanatory. Practitioners and
oversight systems read CoT outputs to understand
_why_ a model reached a conclusion. Safety monitoring, AI auditing, and human-in-the-loop review all
assume that the stated reasoning reflects the actual process. If CoT is a post-hoc rationalization rather
than a causal report, then CoT monitoring provides
false confidence, not accountability.

Recent work has made this assumption untenable.
Turpin et al. [2023] showed that models suppress
acknowledgment of biasing features in their CoT: a
prompt hint that shifts the answer goes unmentioned.
Yüksel et al. [2026] tested faithfulness across twelve reasoning models and found that thinking tokens ac knowledge influential signals at 87.5%, while statedoutput acknowledgment drops to 28.6%. Together,
these findings establish that CoT faithfulness is neither uniform nor reliable.

What existing tests cannot do is test faithfulness
against a decision mechanism that is _graded and en-_
_dogenous_ rather than _binary and exogenous_ . Turpin
and Yüksel inject a discrete perturbation and measure whether the model reports it. That is a simpler
task than what CoT monitoring requires in deployment: accurately reporting the relative strength of
one’s own competing knowledge sources, a continuous property that varies across items and requires
genuine self-assessment.

Knowledge-conflict decisions provide the right
testbed. When context contradicts training knowledge, the resolution is predicted by _parametric_
_certainty_ [1] . Longpre et al. [2021] formalized the
knowledge-conflict paradigm. Xie et al. [2024]
characterized models as “adaptive chameleons”
(context-following) or “stubborn sloths” (parametricpreferring). In prior work, we established through
GEE logistic regression across five model families
that _s_ pop (Wikipedia page views) independently predicts whether models follow or resist contradicting
context ( _β_ = _−_ 0 _._ 38 to _−_ 0 _._ 50, all _p ≤_ _._ 013, BH-FDR
corrected; Jeripity Venkata, 2026). This validated external criterion lets us directly test CoT faithfulness.

We introduce **introspective faithfulness** and test it
through four behavioral measures:

- **Reasoning Consistency (§4.1):** When the answer
flips, does the reasoning change, or does the model
produce near-identical reasoning for opposite conclusions?

- **Confidence** **Calibration** **(§4.2):** Does self-rated
confidence track entity salience and item-level
knowledge?

- **Behavioral** **Coherence** **(§4.3):** Does expressed
confidence independently predict decisions?

- **Source** **Attribution** **(§4.4):** Among distractorfollowing responses, does the CoT claim independent knowledge or correctly credit the document?

1How robustly a fact was encoded through training-time exposure.

We test 200 PopQA items across high- and lowcertainty tiers, four prompt conditions, and eight models, including native-reasoning variants with matched
prompted-CoT counterparts and within-model controls (Gemini Thinking vs. Standard; Claude Thinking vs. Standard).

## 2 Related Work

**CoT faithfulness.** Wei et al. [2022] introduced CoT
prompting to improve reasoning accuracy, but left
open whether the stated reasoning reflects actual computation. Turpin et al. [2023] showed that biasing
features (sycophantic hints, position markers) shift
answers but go unmentioned in CoT. Lanham et al.

[2023] tested causal faithfulness via early answering
and truncation, finding CoT is often causally inert. We
test a harder attribution: not whether models report
external manipulations, but whether they accurately
report the relative strength of their own competing
knowledge sources.

Yüksel et al. [2026] extended this to reasoning
models, finding thinking tokens acknowledge injected
signals at 87.5% vs. 28.6% for stated outputs. Their
design motivates our within-model comparisons, but
their manipulation is planted. Ours is the model’s
own parametric state, validated by _s_ pop.

Complementary approaches probe faithfulness
through causal interventions on model internals.
Chuang et al. [2025] introduce parametric faithfulness
metrics (PFF/FUR) measuring alignment between
CoT and internal model states. Li et al. [2024] evaluate factual consistency of reasoning via entailmentbased decomposition. Paul et al. [2024] study whether
intervention-based improvements to CoT faithfulness
(via ICL, fine-tuning, activation editing) generalize,
finding gains are often brittle. Our work is orthogonal:
a black-box, deployment-oriented behavioral probe
rather than an internal-state intervention. Notably,
Others [2025] find that CoT can be answer-stable but
reasoning-fragile under adversarial perturbations—
the mirror image of our finding, where reasoning
remains stable while answers flip. Together, these
results suggest CoT can decouple from both answers
and mechanisms in different regimes.

**Knowledge conflict.** Longpre et al. [2021] formal ized entity-substitution and showed models over-rely
on parametric memory under conflict. Xie et al.

[2024] identified parametric confidence as the key
moderator. Neither examines whether CoT accurately reports the mechanism. Our prior work [Jeripity Venkata, 2026] established _s_ pop as a validated predictor ( _β_ = _−_ 0 _._ 38 to _−_ 0 _._ 50, _p_ _≤_ _._ 013), providing
the external criterion we use here.

**Model** **introspection** **and** **calibration.** Kadavath
et al. [2022] showed models can predict their own
correctness better than chance. Lyu et al. [2024]
found verbalized confidence is sensitive to phrasing
and context. Ackerman [2026] provides the most
recent characterization: frontier models show “limited, context-dependent” metacognitive ability. Dai
et al. [2026] demonstrate that confidence scale design
(granularity, phrasing) materially affects metacognitive signal, motivating caution about our 1–5 integer
scale. Huang et al. [2026] propose Type-2 SDT decompositions (meta- _d_ _[′]_ /M-ratio) separating metacognitive efficiency from raw accuracy; such analyses
could clarify whether the “Claude anomaly” reflects
low metacognitive sensitivity or miscalibrated confidence bias. Our contribution is testing whether verbalized confidence predicts knowledge-conflict decisions at the item level, where entity-level proxies
fail. Moran and Whiting [2026] argue that item-level
metacognition is largely a population-difficulty signal;
our within-tier _r_ = 0 _._ 134 indicates modest individuated signal beyond entity fame, though constrained
by our coarse confidence scale. Others [2026a] document knowing–doing gaps in agentic decisions; our
finding that argument narratives are decorative while
confidence carries signal echoes their conclusion that
self-reports alone do not ensure reliable choices.

**The intersection.** Prior work shows CoT can misrepresent computational bases, parametric certainty
predicts conflict decisions, and models have limited
self-knowledge. The open question: does CoT explanation of a knowledge-conflict decision reflect the
certainty state that drove it? We call this _introspec-_
_tive_ _faithfulness_ . We do not access internal states;
we test whether CoT behavioral signals are statistically consistent with a validated predictor. We do not
claim causal faithfulness (Lanham’s sense); we test correlational consistency. Within these bounds, this is
the first direct test of whether stated reasoning tracks
the empirical driver of knowledge-conflict decisions.
Complementary internal-signal approaches [Jiang
et al., 2026] suggest separable second-order monitoring channels (e.g., evaluative probes predicting
correction); our behaviorally observed “confidence
layer” may be a surface readout of such mechanisms.
Causal faithfulness frameworks [Others, 2026b] offer stance-change counterfactuals that could crossvalidate our correlational findings; integrating such
probes is a natural next step.

## 3 Methods

**3.1** **Dataset and Items**

We sampled from PopQA [Mallen et al., 2023], a knowledge-intensive QA dataset built from Wikidata
triples. Each item pairs a factual question (e.g., “Who
directed _The_ _Godfather_ ?”) with an entity whose
Wikipedia page-view count ( _s_ pop) proxies training
time exposure. Prior work validated _s_ pop as an independent predictor of knowledge-conflict decisions

[Jeripity Venkata, 2026].

We drew 200 items: the top 100 by _s_ pop ( _high-_
_certainty_ ) and bottom 100 ( _low-certainty_ ). Each item
has a _distractor_ : a plausible but incorrect entity of
the same type (e.g., replacing the real director with
another real director). Eleven items were removed
for implausible distractors or ground-truth errors (see
Appendix A).

The experiment collected 4,920 total responses:
200 items _×_ 8 models _×_ 3 conditions = 4,800 main
responses, plus 120 from a 5-item pilot retained in the
database. After excluding post-sampling items (120
responses) and 1 API error, 4,799 analytic responses
remain. After further filtering NEITHER responses
(§3.4), the primary sample is **3,805 responses** .

**3.2** **Models**

We tested eight models in three groups ( **Table** **1** ).
_Prompted-CoT_ models produce reasoning only when
asked. _Native-reasoning_ models expose internal thinking tokens. _Within-model controls_ use the same archi

**Group** **Model** **T**

Prompted GPT-4o 0
CoT Llama 4 Maverick 0
DeepSeek V3 0

Native DeepSeek R1 d
reasoning Gemini 2.5 Flash _[†]_ 0
Claude Sonnet 4.6 _[†]_ 1

Within-model Gemini 2.5 Flash 0
control Claude Sonnet 4.6 1

Table 1: Model lineup. T = temperature; d = default.

_†_ Thinking-enabled variant exposing internal reasoning tokens.

tecture with thinking disabled. [2]

For native-reasoning models, the API returns two
text fields: internal reasoning and the user-facing response. We extract features from both.

**3.3** **Experimental Conditions**

Each item received three conditions: high-certainty
items received C1, C3, and C4; low-certainty items
received C2, C3, and C4.

**C1/C2:** **Multi-turn** **conflict.** The model first answers the bare question. Then the contradicting document arrives in a follow-up turn, and the model reasons step by step, rates confidence (1–5), and gives a
final answer.

**Turn 1:** “Who directed _Supercock_ ?”
**Turn 2:** “Gus Trikonis” _[baseline]_
**Turn 3:** “I found a source that says the director
is David Nutter. Think step by step...”

C1 applies to high-certainty items; C2 to lowcertainty.

**C3:** **Document-only.** Single-turn. The model answers based _only_ on the document. Serves as a behavioral anchor: compliance (FOLLOW rate) ranged
from 95.0% to 99.0% across the eight models (mean
97.8%), with 0% RESIST for all models. Confidence
here means “confidence in the document.”

2Claude requires temperature = 1 for extended thinking.
Claude Standard also ran at temperature = 1 to keep the within-model comparison clean. All cross-model Claude comparisons
carry this caveat.

**Label** **Definition**

FOLLOW Final answer matches distractor
without negation
RESIST Final answer matches a correct
answer
NEITHER Matches neither; includes
hedges, multiple answers, novel
hallucinations

**Table** **2** : Decision classification. NEITHER responses (22% for Llama to 51% for Gemini on lowcertainty items) are excluded from primary analyses.

**C4:** **Own-knowledge.** Single-turn. “Answer based
on your own knowledge. A document is shown below but may contain errors.” Same distractor, but
instruction frames own knowledge as primary.

All conditions use identical structured response
formats. The model is instructed to produce its stepby-step reasoning first, then state its confidence rating,
then give its final answer. This ordering (reasoning _→_
confidence _→_ answer) means confidence is elicited
after the model has generated its reasoning but before
it commits to a final answer token. Each API call
was a stateless session with conversation history reset
between items and conditions. In C1/C2 multi-turn
format, the baseline answer was placed in the assistant
role programmatically, not generated within the same
session as the conflict turn.

**3.4** **Response Parsing**

Each response was classified as shown in **Table 2** .

Negation handling prevents misclassification (e.g.,
“David Nutter is _NOT_ the director” is not coded as

FOLLOW). Confidence was parsed from the structured format with 96.9–100% coverage.

**Baseline correctness.** For five model families with
prior baselines [Jeripity Venkata, 2026], each item
was coded as baseline-correct if the model answered
correctly before any distractor. This item-level measure complements the entity-level _s_ pop proxy (coverage: 87.5%).

**3.5** **Similarity Computation**

For reasoning consistency (§4.1), we stripped three
classes of attribution phrases from CoT texts before
comparing C2 and C4 on the same item: conversational markers (“you mentioned”), document markers (“the document says”), and self-knowledge markers (“based on my knowledge”). Cosine similarity
was computed using sentence embeddings from all-MiniLM-L6-v2 [Reimers and Gurevych, 2019]. A
cross-item baseline of 500 random same-model pairs
established floor similarity (mean = 0.140).

**3.6** **Analytical Notes**

All correlations use log-transformed _s_ pop. Logistic
regressions use centered predictors. The term _intro-_
_spective_ _faithfulness_ does not claim we observe internal representations. It marks the _kind_ of attribution demanded: reporting one’s own certainty, not
acknowledging an external perturbation. The test is
correlational consistency with a validated predictor.

**Notation.** Throughout, _ρ_ denotes Spearman’s rank
correlation and _r_ denotes point-biserial or Pearson
correlation. All _p_ -values are two-tailed unless otherwise noted.

## 4 Results

**Temperature note.** Claude Sonnet 4.6 (both variants)
required temperature = 1. All other models ran at temperature = 0. This affects cross-model Claude comparisons; within-model comparisons are unaffected.

**4.1** **Reasoning Consistency**

When a model changes its answer, does its reasoning also change?

We identified all items where the same model gave
opposite answers under C2 and C4 ( _flip_ _pairs_ ) and
compared their attribution-stripped CoT similarity.

Flip pairs ( _n_ = 71) were 5 _._ 6 _×_ more similar than
random cross-item pairs (0.784 vs. 0.140). They retained 96% of same-answer pair similarity (0.784 vs.
0.817; Mann-Whitney _p_ = 0 _._ 010, Cohen’s _d_ = 0 _._ 34,
overlap coefficient = 0.74). This gap was confirmed by ROUGE-L ( _d_ = 0 _._ 45, _p_ = 0 _._ 002), ruling out dependence on a single similarity metric. When a model
reaches opposite conclusions on the same question,
the step-by-step reasoning remains highly similar;
only the final attribution changes.

**Concrete** **example.** **Table** **3** illustrates this with
Claude’s responses about Tsutomu Seki (distractor:
_politician_ ). Under C2, Claude partially accepts the
distractor. Under C4, it rejects it. Both CoTs recite
the same three facts: Japanese astronomer, Comet
Ikeya-Seki (1965), guitar teacher. Only the concluding attribution differs.

**Flip-pair representativeness.** Flip pairs ( _n_ = 71,
19.6% of all C2–C4 item-model pairs) do not differ from non-flip pairs on any measured dimension:
log( _s_ pop) (4.73 vs. 4.65, _p_ = 0 _._ 183), mean confidence (3.28 vs. 3.52, _p_ = 0 _._ 074), CoT length (185 vs.
179 tokens, _p_ = 0 _._ 649), or property-type distribution
( _χ_ [2] = 20 _._ 0, _p_ = 0 _._ 067). The similarity estimates are
not driven by a biased item subset.

**Per-model results with uncertainty estimates.** **Ta-**
**ble 4** reports per-model deltas with bootstrapped 95%
confidence intervals and permutation tests (10,000
resamples). GPT-4o is the only model with a statistically significant negative delta (∆= _−_ 0 _._ 132, 95%
CI [ _−_ 0 _._ 188 _, −_ 0 _._ 077], _p <_ 0 _._ 001): its reasoning genuinely changes when its answer changes. Reading
five GPT-4o flip pairs confirmed this: C2 CoTs cite
specific committed facts (“born in Hattfjelldal, Norway”); C4 CoTs admit uncertainty (“I do not have
specific knowledge”). Length ratios did not differ
between flip and same-answer pairs ( _p_ = 0 _._ 51).

No other model shows a statistically reliable delta.
Claude’s positive deltas (+0 _._ 030 and +0 _._ 026) and
Llama’s (+0 _._ 047) have confidence intervals crossing
zero. DeepSeek R1 ( _n_ = 3) and Llama ( _n_ = 6) have
too few flip pairs for reliable per-model inference and
should be interpreted with caution.

**Internal reasoning vs. stated output.** In the user-facing stated CoT, thinking-enabled and standard variants show similar deltas (+0.026 vs. +0.030), neither
significantly different from zero. Enabling thinking

**C2 (FOLLOW, confidence = 4)** **C4 (RESIST, confidence = 4)**

_Tsutomu Seki is a Japanese individual._ _The Tsutomu_
_Seki_ _I_ _am_ _most_ _familiar_ _with_ _is_ _a_ _Japanese_ _amateur_
_astronomer_ _born_ _in_ _1930,_ _who_ _is_ _well_ _known_ _for_ _co-_
_discovering_ _several_ _comets,_ _including_ _Comet_ _Ikeya-_
_Seki in 1965._ _He is also a guitarist and runs a music_
_school,_ _but_ _his_ _notable_ _fame_ _comes_ _from_ _astronomy._
_It is possible there is another Tsutomu Seki who is a_
_politician. .._

_Tsutomu_ _Seki_ _is_ _a_ _well-known_ _Japanese_ _amateur_ _as-_

_tronomer_ _who_ _discovered_ _several_ _comets,_ _including_
_Comet Ikeya-Seki (1965), which was one of the bright-_
_est comets of the 20th century._ _He is also known as a_
_guitar instructor._ _He is a politician, but this does not_
_align with what I know about Tsutomu Seki..._

**Cosine similarity after attribution stripping:** **0.916.** Same three facts. Same knowledge display. Opposite
conclusions. This pattern is representative of the 23 pairs with similarity _≥_ 0 _._ 85.

Table 3: Paired CoT from Claude Sonnet 4.6 Standard on _Tsutomu Seki / occupation_ (distractor: _politician_ ).
The model reaches opposite conclusions from near-identical reasoning.

**Model** _**n**_ **∆** **95% CI** _**p**_

GPT-4o 12 _−_ .132 [ _−_ .19, _−_ .08] _<_ .001
Gemini Std 10 _−_ .057 [ _−_ .11, _−_ .01] .121
DeepSeek R1 3 _−_ .061 [ _−_ .19, +.08] .287
DeepSeek V3 10 _−_ .036 [ _−_ .11, +.03] .276
Gemini Think 13 _−_ .031 [ _−_ .10, +.03] .320
Claude Think 7 +.026 [ _−_ .01, +.06] .422
Claude Std 10 +.030 [ _−_ .01, +.06] .173
Llama 4 Mav. 6 +.047 [ _−_ .01, +.10] .319

**Table** **4** : Attribution-stripped CoT similarity: flip
pairs vs. same-answer pairs, per model. _n_ : number
of flip pairs. ∆ = flip _−_ same similarity. 95% CIs
from 10,000 bootstrap resamples; _p_ from permutation
test (raw; all survive BH-FDR correction at the same
significance level). Only GPT-4o shows a statistically
reliable negative delta ( _p_ adj = 0 _._ 0008). DeepSeek R1
( _n_ = 3) and Llama ( _n_ = 6) are underpowered for
per-model inference.

tokens does not change the decision-sensitivity of the
output the user sees.

However, analyzing the _internal_ thinking tokens
directly reveals a different pattern. For the three
native-reasoning models, we compared attributionstripped similarity of thinking tokens (not stated CoT)
on flip pairs vs. same-answer pairs. Thinking tokens
are more decision-sensitive than stated CoT: on flip
pairs, thinking similarity is 0.709 vs. stated similarity
of 0.787, a gap of _−_ 0 _._ 077 (Wilcoxon _p_ = 0 _._ 033).
The effect is driven by Gemini (∆think = _−_ 0 _._ 139,
_n_ = 13), while Claude Thinking shows no gap
(∆= +0 _._ 009, _n_ = 7). The stated output smooths over reasoning differences that the internal process
preserves, at least for Gemini. For Claude ( _n_ = 7),
the internal reasoning shows a similar pattern to the
stated output, though the small sample precludes
strong conclusions.

**4.2** **Confidence Calibration**

Does self-rated confidence track actual knowledge?
We compared two criteria: entity salience ( _s_ pop, entity-level) and baseline correctness (item-level).

**Key** **finding.** Within the low-certainty tier, entity
salience has no relationship with confidence (Spearman _ρ_ = _−_ 0 _._ 005, _p_ = 0 _._ 81). But baseline correctness does: models that knew the specific fact at baseline express higher confidence even on obscure items
( _r_ = 0 _._ 134, _p_ _<_ 0 _._ 001). This is genuine, if weak
( _∼_ 2% of variance), introspective access to item-level
knowledge.

The pooled correlation ( _ρ_ = 0 _._ 305) is misleading:
it is entirely a between-tier step function. Models say
“5” for famous entities and “3” for obscure ones, but within each tier they show no graded sensitivity.

**Model differences.** **Table 5** reveals a split. Claude
tracks both criteria equally: entity fame and item
knowledge predict its confidence equally well. GPT4o, Gemini, and Llama are more sensitive to fame
than to actual knowledge. When these models say
“I’m confident,” they partly mean “this entity is fa
mous,” not “I specifically know this fact.”

**Model** _**ρ**_ s_pop _**r**_ base **∆**

Claude Std .384 .387 +.003
Claude Think .389 .396 +.008
DeepSeek V3 .291 .283 _−_ .008
Llama 4 Mav. .316 .245 _−_ .071
GPT-4o .312 .211 _−_ .101
Gemini Think .196 .104 _−_ .092
Gemini Std .234 .132 _−_ .102

**Table** **5** : What predicts confidence? _ρ_ s_pop: entity salience. _r_ base: baseline correctness. ∆=
_r_ base _−_ _ρ_ s_pop. Positive: calibrated to actual knowledge. Negative: calibrated to fame.

**4.3** **Behavioral Coherence**

Does confidence predict what the model _does_ ? We
tested this with a GEE logistic regression clustering
by item, using confidence and log( _s_ pop) as predictors.

Within the low-certainty tier, entity salience does
not predict decisions (coef = +0.383, _p_ = 0 _._ 27). But
CoT confidence does (coef = _−_ 0.475, _p_ _<_ 0 _._ 001):
lower confidence predicts higher probability of following the distractor. Combined with §4.2, this traces
an item-level pathway: _knows the fact →_ _higher con-_
_fidence →_ _resists distractor_ . All links operate below
the resolution of entity salience.

Within high-certainty, both confidence and entity
salience predict decisions independently.

**The Claude anomaly.** **Table 6** reports per-model
confidence-decision correlations with bootstrapped
95% CIs. Six of eight models show significant correlations ( _r_ = _−_ 0 _._ 156 to _−_ 0 _._ 230). Claude shows
none: _r_ = _−_ 0 _._ 022 (Std, 95% CI [ _−_ 0 _._ 10 _,_ +0 _._ 06])
and _r_ = +0 _._ 012 (Think, 95% CI [ _−_ 0 _._ 07 _,_ +0 _._ 10]).
Importantly, Claude’s upper CI bound (+0 _._ 06) does
not approach even the smallest significant correlation
in any other model (DeepSeek R1 at _−_ 0 _._ 156), confirming the near-zero correlation is not a small-sample
artifact.

This is **not** a ceiling artifact. Claude has the _lowest_
confidence=5 rate (54–56%) and the _highest_ variance
(SD = 1.39). Gemini, with a severe ceiling (91% at
confidence=5), still shows _r_ = _−_ 0 _._ 213.

The pooled _r_ _≈_ 0 masks a condition-specific pattern. Within the matched C2+C4 low-certainty subset,
Claude shows significant coupling ( _r_ = _−_ 0 _._ 587, 95%

**Model** _**r**_ **95% CI** _**p**_

Gemini Think _−_ .230 [ _−_ .27, _−_ .19] _<_ .001
GPT-4o _−_ .227 [ _−_ .29, _−_ .16] _<_ .001
Llama 4 Mav. _−_ .223 [ _−_ .28, _−_ .17] _<_ .001
DeepSeek V3 _−_ .218 [ _−_ .27, _−_ .15] _<_ .001
Gemini Std _−_ .213 [ _−_ .25, _−_ .17] _<_ .001
DeepSeek R1 _−_ .156 [ _−_ .22, _−_ .08] .001
Claude Std _−_ .022 [ _−_ .10, +.06] .635
Claude Think +.012 [ _−_ .07, +.10] .791

Table 6: Confidence-decision correlation per model
(all conditions, both tiers) with bootstrapped 95% CIs.
Negative _r_ : lower confidence predicts FOLLOW. All
_p_ -values survive BH-FDR correction. Claude’s near-zero _r_ with CI excluding _−_ 0 _._ 10 confirms the pattern
is not a power issue; it has the lowest ceiling rate
(54–56%) and highest SD (1.39).

CI [ _−_ 0 _._ 700 _, −_ 0 _._ 461]). However, under C4 (ownknowledge framing), Claude frequently follows distractors with high confidence, creating a positive
confidence-FOLLOW association that cancels the
negative C2 association. The anomaly is not global
decoupling but a condition-specific reversal: Claude
confidently follows documents it was told to ignore.
We characterize this as _performance_ _of_ _calibration_

_without_ _calibration_ : the pooled confidence scores
look varied and thoughtful, but the variation tracks instruction framing rather than the model’s own knowledge state.

**Appropriate uncertainty.** **Table 7** shows how each
model expresses uncertainty on obscure items. Llama
follows with low confidence (47%); Claude hedges
with NEITHER responses (46–48%); GPT-4o commits regardless (only 37% appropriate).

**4.4** **Source Attribution**

Among 2,763 FOLLOW responses, we tested whether
the CoT claimed independent knowledge of the distractor ( **Table 8** ).

Llama is a clear outlier at 15.2% (corrected estimate _∼_ 9%). Two types of genuine confabulation
emerged. _Fabricated_ _geography_ : asked about The
Hague (distractor: Douglas County), Llama writes
“I found that there is indeed a The Hague, a village in Nebraska, located in Douglas County.” No such

**Model** **Hedge** **Low-conf** **Total**
**%** **follow %**

Llama 4 Mav. 22 47 **69**
Claude Std 46 18 **64**
Claude Think 48 14 **62**
DeepSeek R1 43 10 53
Gemini Std 51 1 52
Gemini Think 50 1 51
DeepSeek V3 40 8 48
GPT-4o 31 6 **37**

**Table** **7** : Appropriate uncertainty on low-certainty
C2 items. Hedge: NEITHER responses. Low-conf
follow: follows distractor but signals low confidence
( _≤_ 2). Models differ in _how_ they express uncertainty.

**Model** **N** **Confab %** **Low %**

Llama 4 Mav. 363 **15.2** 19.2
GPT-4o 260 6.2 9.7
Gemini Std 286 5.6 6.8
DeepSeek V3 259 5.4 9.0
Claude Std 284 4.9 6.3
Gemini Think 298 4.7 6.1
Claude Think 262 4.2 6.3
DeepSeek R1 251 3.6 4.9 village exists. _Fabricated works_ : asked about “Load”
(distractor: platform game), Llama writes “There IS
a video game called ‘Load’ that is indeed a platform
game.” Both involve inventing specific facts to validate a distractor the model has no genuine knowledge
of.

Table 8: Confabulation among FOLLOW responses.
Rates are upper bounds ( _∼_ 40% false-positive rate on
manual review). Llama is a clear outlier.

One Gemini response showed _reasoning-_
_conclusion_ _dissociation_ : the CoT concluded the
document was wrong, yet the answer followed it. Of
2,263 FOLLOW responses searched, only this single
case (0.04%) was confirmed.

**4.5** **Sensitivity Analyses**

**NEITHER responses as a third outcome.** Primary
analyses exclude NEITHER responses (22–51% of
low-certainty responses per model). To test whether
this exclusion biases confidence-decision relationships, we ran a multinomial analysis including NEI

THER as a third category alongside FOLLOW and
RESIST. Mean self-rated confidence follows a monotonic ordering across all three outcomes in both tiers:
FOLLOW (2.98 low-certainty, 4.02 high-certainty) _<_
NEITHER (3.70, 4.42) _<_ RESIST (4.42, 4.84), all
differences significant (Kruskal-Wallis _H_ = 95 _._ 96
and 56 _._ 95, both _p <_ 0 _._ 001). NEITHER responses sit
at intermediate confidence, consistent with genuine
hedging. The confidence contrast between FOLLOW
and RESIST is _−_ 0 _._ 81 in the multinomial analysis
vs. _−_ 0 _._ 90 in the primary (binary) analysis: direction
preserved, magnitude similar. Excluding NEITHER
does not materially change the findings.

**Similarity** **metric** **robustness.** The flip-pair vs.
same-answer similarity gap was tested with ROUGEL in addition to MiniLM cosine similarity. ROUGE-L
confirms and slightly strengthens the gap: Cohen’s
_d_ = 0 _._ 45 (vs. _d_ = 0 _._ 34 for MiniLM), _p_ = 0 _._ 002. The
finding is not dependent on the choice of similarity
metric.

**Item-clustered GEE.** The per-model correlations
reported in §4.3 were confirmed with a GEE logistic regression using item-level clustering (C2 lowcertainty subset): confidence remains significant
(coef = _−_ 0 _._ 475, _p_ _<_ 0 _._ 001) while _s_ pop remains nonsignificant (coef = +0 _._ 383, _p_ = 0 _._ 27). The item-level
pathway is robust to clustering structure.

**Temperature** **ablation.** To test whether temperature explains Claude’s anomalous pattern, we ran
GPT-4o at temperature = 1 on all low-certainty items
under C2 and C4 (200 additional API calls). On
the matched subset (C2+C4, low-certainty items),
confidence-decision coupling was virtually identical
at _T_ = 0 ( _r_ = _−_ 0 _._ 726) and _T_ = 1 ( _r_ = _−_ 0 _._ 735,
∆= _−_ 0 _._ 009). These values are higher than the all-conditions pooled _r_ = _−_ 0 _._ 227 reported in **Ta-**
**ble** **6** because the subset excludes C3 (where confidence means “confidence in the document”) and
high-certainty items (where variance is compressed
by ceiling effects). NEITHER rates were also identical (34% at both temperatures). Temperature does
not attenuate coupling. The Claude pattern is model-specific, not temperature-driven.

**Confidence** **scale.** Our 1–5 integer scale induces
ceiling effects (91% of Gemini responses at confidence = 5) and limits metacognitive resolution. Finer
scales (e.g., 0–20) have been shown to improve
metacognitive signal [Dai et al., 2026]. This is a
design limitation; future work should test whether
finer-grained elicitation strengthens or weakens the
observed confidence-decision coupling.

## 5 Discussion

**5.1** **Two Functions of CoT**

The central finding is a tension: CoT confidence carries weak but real signal (§4.2, §4.3), yet the reasoning narrative remains highly similar across opposite
decisions (§4.1). These coexist because CoT performs
two functions simultaneously.

**Function 1:** **Knowledge display.** Most of what a
model writes is a recitation of what it knows about the
entity. For Tsutomu Seki ( **Table 3** ), both FOLLOW
and RESIST responses contain the same three facts:
astronomer, Comet Ikeya-Seki 1965, guitar teacher.
This content is stable because parametric knowledge
does not change with the decision. Same-item CoT
pairs have mean similarity 0.817, confirming entity
knowledge dominates.

**Function 2:** **Decision justification.** A thin layer of
attribution language and a confidence score wrap the
knowledge display into an apparent argument. For
FOLLOW: “the new information is consistent.” For
RESIST: “this contradicts my knowledge.” This layer
adjusts with the decision. At the aggregate level, flip
pairs retain 96% of same-answer similarity, meaning
the knowledge display is shared while the framing
layer differs.

Monitors reading the logical argument are reading
Function 1, which remains similar regardless of the
decision. Monitors reading confidence scores are
reading Function 2, where genuine signal exists.

**5.2** **The Claude Anomaly**

Claude’s pooled confidence-decision correlation ( _r_ _≈_
0, 95% CI [ _−_ 0 _._ 10 _,_ +0 _._ 06]) masks a more specific
pattern. Within the C2+C4 low-certainty subset,

Claude shows strong coupling ( _r_ = _−_ 0 _._ 587, CI

[ _−_ 0 _._ 700 _, −_ 0 _._ 461]): confidence does predict decisions within individual conditions. The near-zero
pooled _r_ arises because the relationship _reverses_
across conditions. Under C4 (own-knowledge framing), Claude frequently follows distractors with high
confidence, creating a positive association that cancels the negative C2 association.

A matched-temperature ablation rules out temperature as the explanation: GPT-4o at _T_ = 1 retains full
coupling on the matched C2+C4 low-certainty subset
( _r_ = _−_ 0 _._ 735 vs. _−_ 0 _._ 726 at _T_ = 0; ∆= _−_ 0 _._ 009).
The pattern is model-specific. The within-model comparison (Thinking vs. Standard, identical on all measures) rules out prompt-format effects.

What remains is that Claude generates confidence
scores that track instruction framing rather than its
own knowledge state. Under C2, confidence tracks
knowledge (negative coupling, like other models).
Under C4, confidence tracks the document even when
the instruction says to ignore it. We characterize this
as _performance_ _of_ _calibration_ _without_ _calibration_ :
the pooled scores look varied and thoughtful, but the
variation reflects condition-switching rather than genuine item-level knowledge tracking. This may reflect
RLHF training that optimized condition-appropriate
confidence expression as a surface behavior.

**5.3** **The Item-Level Pathway**

The most novel finding: within low-certainty items,
_s_ pop does not predict decisions ( _p_ = 0 _._ 27 in the itemclustered GEE), but confidence does ( _p_ _<_ 0 _._ 001).
Some item-level variable affects both confidence and
decisions but is not captured by entity salience. §4.2
identified this as baseline correctness ( _r_ = 0 _._ 134).
The chain: _knows fact →_ _higher confidence →_ _resists_
_distractor_ . All links operate below _s_ pop’s resolution.

The magnitude is small ( _r_ = 0 _._ 134 explains _∼_ 2%
of variance). Models are barely above the floor of
introspective accuracy on obscure items. But they
are above it. The multinomial sensitivity analysis
(§4.5) confirms this pathway holds when NEITHER
responses are included as a third outcome.

**5.4** **Practical Recommendations**

**Monitor** **confidence,** **not** **arguments.** Flip pairs
retain 96% of same-answer similarity. Argument
structure will not reliably detect knowledge-conflict
misattribution. Confidence expressions and hedges
carry the signal. However, confidence is subject to
ceiling effects (91% at confidence=5 for Gemini),
model-specific condition-tracking (Claude), and potential Goodharting: if models are trained to produce “calibrated-looking” confidence, the signal may
erode. Monitoring should combine numerical confidence with NEITHER/abstention rates, external verification, and where available, log-probability-based
signals that are harder to Goodhart than verbal confidence. Approaches that explicitly train for reasoningdecision coherence [e.g., preference optimization as
in Others, 2026b] may help close the gap between the
knowledge display and the decision-justification layer
observed here.

**Scrutinize high-confidence FOLLOW.** A model
following a distractor with confidence 2 is expressing
honest uncertainty. A model following with confidence 5 is either falsely confident, confabulating, or
generating confidence independently of its state (the
Claude pattern). All three warrant greater scrutiny
than low-confidence deference.

**Entity fame** = **item knowledge.** Models express
higher confidence on famous entities even where fame
has no predictive power (Spearman _ρ_ = _−_ 0 _._ 005, _p_ =
0 _._ 81, within low-certainty). A confident assertion
about Beethoven should not be trusted more than one
about an obscure Irish poet, solely because Beethoven
is famous.

**Prefer** **within-model** **comparisons.** The ClaudeGPT-4o contrast is the clearest finding but carries
confounds (temperature, architecture). The within-model pairs (Gemini Thinking vs. Standard; Claude
Thinking vs. Standard) are clean: thinking tokens do
not improve coupling in either family.

**5.5** **Limitations**

_s_ **pop vs. fact knowledge.** _s_ pop measures entity fame,
not fact-specific knowledge. 46% of C4-high FOL

LOW responses involved baseline-incorrect items.
The baseline-correctness criterion partially corrects
this but covers only five of eight model families. Future work should collect direct-answer baselines for
all models.

**Bimodal design.** Two tiers (top/bottom 100) produce a bimodal predictor. Within-tier correlations
are the appropriate test, but medium-certainty items
would enable continuous gradient analysis.

**Temperature.** All cross-model Claude comparisons carry the temperature = 1 caveat. Replicate runs
at the same temperature for another model would isolate the stochasticity contribution; we did not collect
these.

**Confidence scale.** The 1–5 integer scale induces
discretization and ceiling effects. Finer scales (0–
20) may improve metacognitive resolution [Dai et al.,
2026].
**Flip-pair sample size.** The total of 71 flip pairs,
with some models contributing as few as 3 (DeepSeek
R1) or 6 (Llama), limits per-model inference. The
aggregate result ( _d_ = 0 _._ 34, confirmed by ROUGE-L
at _d_ = 0 _._ 45) is robust, but per-model deltas should be
interpreted with the reported CIs.

**Detector** **precision.** The confabulation detector
has _∼_ 40% false positives. §4.4 rates are upper
bounds.

**Confidence elicitation order.** The fixed ordering
(reasoning _→_ confidence _→_ answer) means the confidence signal may be co-produced with the rationale
rather than independently introspective. Randomizing
the order would help disambiguate.

**Residual** **template** **similarity.** Attribution stripping covers three phrase classes. Residual stylistic
boilerplate (discourse markers, list structures, stock
hedges) may inflate similarity estimates beyond what
substantive content overlap would produce.

**Scope.** These findings apply to factual QA under
knowledge conflict. Generalization to mathematical
reasoning, multi-step inference, or safety refusals is
unknown.

**Data** **release.** We commit to releasing code,
prompts, item lists, and anonymized model outputs
upon publication.

**Programmatic baseline insertion.** In C1/C2, the
baseline answer was inserted from prior study data
rather than generated fresh by the model. The prior
baselines were produced by the same models at tem perature = 0, but the programmatic insertion may differ from genuine model-generated dialogue in ways
that affect conflict dynamics.

**Calibration metrics.** We report correlation-based
analyses of confidence. Classical calibration metrics
(ECE, Brier score, reliability diagrams) and Type-2
SDT decompositions (meta- _d_ _[′]_ ) would provide complementary evidence and are left for future work.

**Multiple comparisons.** Per-model hypothesis tests
in Tables 4 and 6 report raw _p_ -values; all conclusions survive BH-FDR correction (reported in table captions). No additional corrections are applied
across the four evaluation sections, each treated as
exploratory.

**Correlational design.** We test correlational consistency between CoT content and a validated behavioral predictor. We do not claim causal faithfulness
in the sense of Lanham et al. (2023). Complementary
causal and parametric faithfulness methods [Chuang
et al., 2025, Li et al., 2024] could provide additional
evidence about the relationship between internal representations and stated reasoning.

## 6 Conclusion

CoT under knowledge conflict serves two functions: a
knowledge display that remains highly similar across
opposite decisions (flip pairs retain 96% of sameanswer similarity) and a thin confidence layer carrying weak but real introspective signal. GPT-4o is
the only model with statistically reliable reasoningdecision coupling. Claude Sonnet 4.6 produces expressively varied confidence scores whose relationship to decisions reverses across conditions, yielding
a near-zero pooled correlation that reflects conditiontracking rather than knowledge-tracking. For nativereasoning models, internal thinking tokens show
greater decision-sensitivity than the user-facing output, suggesting the stated CoT smooths over differences the internal process preserves. For monitoring,
track expressed confidence, not argument structure.

## References

Stefanie Ackerman. Evidence for limited metacog
nition in frontier language models, 2026. ICLR
2026.

Yung-Sung Chuang et al. Measuring and improving chain-of-thought reasoning faithfulness via parametric probing, 2025.

Zhifeng Dai et al. Confidence under the lens: Scale de
sign and verbal calibration in large language models, 2026.

Jieyu Huang et al. Metacognitive monitoring in large language models: A type-2 signal detection analysis, 2026.

Pruthvinath Jeripity Venkata. Three regimes of
context-parametric conflict: A predictive framework and empirical validation, 2026.

Zhengbao Jiang et al. Internal evaluative signals in language models: Beyond log-probabilities, 2026.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom

Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli
Tran-Johnson, et al. Language models (mostly)
know what they know, 2022.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan,

Nolan Steiner, Carson Denison, Danny Hernandez,
Dustin Li, Esin Durmus, Evan Hubinger, Jackson
Kernion, et al. Measuring faithfulness in chain-ofthought reasoning, 2023.

Junhao Li et al. FRODO: Fact-reasoning-oriented decomposition for factual consistency evaluation,
2024.

Shayne Longpre, Kartik Perisetla, Anthony Chen,

Nikhil Ramesh, Chris Donahue, and Sameer Singh.
Entity-based knowledge conflicts in question answering. In _Proceedings of the 2021 Conference_
_on Empirical Methods in Natural Language Pro-_
_cessing_, pages 7052–7063. Association for Computational Linguistics, 2021.

Zhixin Lyu, Liang Zheng, et al. On verbalized confi
dence scores for LLMs, 2024.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,

Daniel Khashabi, and Hannaneh Hajishirzi. When
not to trust language models: Investigating effectiveness of parametric and non-parametric memories, 2023.

R. Moran and S. Whiting. Item-level metacognition in language models: Population signal or individuated
access?, 2026.

Others. MATCHA: Probing faithfulness via answer
stable reasoning perturbations, 2025.

Others. MIRROR: Measuring knowing–doing gaps in agentic LLM decisions, 2026a.

Others. RFEval: Causal faithfulness evaluation via stance-change counterfactuals, 2026b.

Debjit Paul et al. Making reasoning faithful: Inter
ventions for chain-of-thought faithfulness, 2024.

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In _Proceedings of the 2019 Conference_
_on Empirical Methods in Natural Language Pro-_
_cessing_, pages 3982–3992. Association for Computational Linguistics, 2019.

Miles Turpin, Julian Michael, Ethan Perez, and

Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations
in chain-of-thought prompting, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten

Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le,
and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _Ad-_
_vances in Neural Information Processing Systems_,
volume 35, pages 24824–24837, 2022.

Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and

Yu Su. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in
knowledge conflicts. In _Proceedings of the Twelfth_
_International Conference on Learning Representa-_
_tions_, 2024. ICLR 2024 Spotlight.

Kaan Yüksel et al. Lie to me: LLM faithfulness to injected signals in reasoning models, 2026.

## A Excluded Items

**Table** **9** lists all 11 excluded items. Pre-sampling
exclusions were removed before item selection. Postsampling exclusions were identified during quality review; their responses remain in the database but are
excluded from analyses.

**Item** **Distractor** **Reason**

_Pre-sampling (6 items)_
Dutch novel Wrestler Category
Children’s book Corporation Category
Neil Young song Rapper Era
1958 song 1990s rapper Era
1992 film score Pop artist Era
_Americanah_ Che Guevara Era (d. 1967)

_Post-sampling (5 items)_
PK screenwriter  - Ground truth
Jericho/occ.  - Ground truth
Rome/empire W. Roman Substring
Emp.

_The Gift_ /genre  - Ambiguous
Straits Settl.  - Historical

Table 9: Excluded items with reasons.

## B Confabulation Detector

The detector operates on FOLLOW responses only. It
searches for independence markers (“I know,” “from
my training”) within two sentences of the distractor,
flagging cases without source attribution. Manual review estimated _∼_ 40% false positives, primarily from
models using independence language while _evaluat-_
_ing_ the distractor.

### C NEITHER Response Analysis

NEITHER responses represent genuine hedging,
novel hallucinations, ambiguous phrasing, and parsing failures. Per-model rates on low-certainty C2
items range from 22% (Llama) to 51% (Gemini).
Gemini’s high rate reflects a tendency to hallucinate third answers on obscure items (e.g., inventing
“Venice” as a birthplace).
