# Gathered, Not Admitted: How Attention Brings a Latent Variable into Verbalizable Form

- **Authors:** Parsa Mazaheri
- **Date:** 2026-08-15
- **Source:** https://arxiv.org/abs/2608.15022
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `mazaheri_gathered-admitted-attention-brings_2608.15022`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

## Parsa Mazaheri

University of California, Santa Cruz pmazaher@ucsc.edu

## Abstract

Language models hold latent quantities in a form they can report on, and more of a quantity is present in that form when the task requires reusing it flexibly. What causes a representation to enter that form is open, and the word _workspace_ invites an admission story: a gate that decides what gets in. Testing it on open-weight models with Jacobian lenses, over a benchmark whose five arms share an identical context, we find no gate where it predicts one. Demand raises a concept’s lens visibility beyond what applying an operator to a _supplied_ value produces: +0 _._ 050 [+0 _._ 045 _,_ +0 _._ 057] in percentile rank on our primary checkpoint, positive on all four we measure, though that arm answers at ceiling and the accuracymatched contrast is stronger under that readout. At the same time one shared linear map decodes the variable from every arm, the control included, at 6 _._ 4–9 _._ 0 _×_ its selection-corrected floor. What produces the later readable form at the queried position is attention-mediated gathering inside a mid-depth window: separating patch depth from readout depth puts transport there at least 17 _×_ above anywhere shallower under non-saturating readouts, with no tested MLP output contributing positively inside it. Under the saturating percentile rank the same grid does not localise the window, which is a fact about that measure. An arm that needs the variable for nothing concentrates sevenfold less, so the window is demand-specific. That window has two measured edges, a survival failure below and destruction above, and it falls at the same _fractional_ depth in a 64-layer hybrid and a 62-layer dense model from another family. We localise where the variable is installed and read, not the route from the passage, which transports nothing. But the readout is not a calibrated measure of use: three components move it to within 12% of one another and differ 7 _._ 4 _×_ in what they do to the answer.

## 1 Introduction

Recent work identifies a set of directions in a language model’s residual stream whose contents the model can report on, a _verbalizable workspace_ , and shows that a latent quantity is more present there when the task requires using it flexibly (Gurnee et al., 2026). What causes a representation to enter that form is left open.

The vocabulary of the finding suggests an answer. A workspace has contents; contents are _admitted_ ; admission implies a decision taken over something already present. Read that way (Figure 1a), the latent variable _z_ sits at the position where the model will be asked about it, at every depth, and what demand changes is only whether some component lets it be read. The reading is natural enough that concurrent engineered systems build it: an external controller that writes into the late band when next-token entropy crosses a threshold (Sathish, 2026), or an architectural gate trained to route information into a workspace (Chateau-Laurent & VanRullen, 2025).

An unmodified pretrained model does not have one there. Early at the query position, at least seventeen times less of the variable is transportable than inside a mid-depth window, and across the twelve layers immediately below that window no component transports it in every donor pairing.

Code: https://github.com/parsa-mz/innerj

Figure 1: Two accounts of what changes when a task demands flexible reuse of a latent variable _z_ . **(a)** Admission, as the workspace vocabulary invites one to read it: _z_ is at the query position at every depth and a gate sets whether it enters. This is our formalisation of the hypothesis, not a claim Gurnee et al. (2026) make. **(b)** What we measure: attention gathers _z_ into the query position within a mid-depth window, a value installed below it does not survive to the readout, and the answer depends far more on the consolidated stream than on the gather. The passage is the plausible source but we do not localise it. (b) displaces (a) at the query position; it does not rule out a gate acting on the gather itself. The re-derivation labelled in (b) is our favoured reading of the lower edge, not a measurement (§5).

Attention carries it in; no tested MLP output inside the window contributes positively. And neither edge of the window is a capability appearing: below it an installed value fails to survive to the readout, above it the intervention stops substituting and starts destroying. So the picture is Figure 1b, and what moves with task demand is transport into the position where the measurement is taken, not the unmasking of a variable that was sitting there all along.

**Contributions. (1)** Visibility dissociates from availability: demand raises it against a control matched on prompt format _and_ accuracy, one shared linear map reads _z_ from every arm including the one that needs it for nothing, and the effect survives holding operator use fixed on four checkpoints from four families (§3). **(2)** The mechanism is transport by attention. That attention rather than an MLP moves a value into the queried position is _convergent_ (Geva et al., 2023; Todd et al., 2024); the increment is the _dependence_ , since our arms share an identical context and the amount transported changes with what the task asks about a value that context never names (§4). **(3)** Both edges of the window are measured rather than asserted, it lands at the same fractional depth on a second architecture, and the behaviour runs through one derived lens direction against four controls (§5, §7). **(4)** We identify two nuisances that decide what a patching result is worth, neither reported in the work we build on (§6), and give a methodological warning with a matched positive control (§8).

We do not claim a write circuit. What is localised is a gather, and it is selectively necessary without being sufficient.

## 2 Setup

### 2.1 The lens, and what may be measured on it

The Jacobian lens summarises the map from an intermediate residual stream to the final one = by one averaged linear operator per layer, _Jℓ_ E[ _∂h_ final _/∂hℓ_ ], and reads a layer out as softmax( _WU_ norm( _Jℓhℓ_ )); _Jℓ_ = _I_ recovers the logit lens. _J-space_ is the set of points expressible as a sparse nonnegative combination of vocabulary-indexed lens vectors (Gurnee et al., 2026): a union of cones rather than a subspace, which is why no subspace language appears below. We reserve the term for that object; our dependent variable is the rank of a concept under the readout, which we call _lens visibility_ . The two are related but not the same, and we checked rather than assumed how closely: the sparse nonnegative coefficient that actually encodes membership correlates with our readout at Spearman 0 _._ 24 and is exactly zero in 90 _._ 9% of activations, which is why it is not the dependent variable and why we say _visibility_ throughout rather than membership (Appendix B).

One class of quantity has to be excluded first. For any positive _aℓ_ there is a network computing the _same function_ whose residual stream is _aℓhℓ_ , and under it _J_^(˜) _ℓ_ = ( _aL−_ 1 _/aℓ_ ) _Jℓ_ . So anything computed on _Jℓ as a matrix_ and homogeneous of nonzero degree moves under a change the model cannot detect, and supports no claim here; the readout itself does not move, since the two differ by a scalar the normaliser removes. The rescaling must vary with depth to say anything at all, a global one leaving _Jℓ_ untouched. The derivation, the fp32 check and the candidate checkpoint this disqualified are in Appendix B.

Every dependent variable is therefore read after the model’s own final normaliser. The primary one is the percentile rank of the gold concept in the vocabulary _V_ , a percentile rather than a raw rank so it compares across vocabulary sizes. It also **saturates** , and not harmlessly: it separates rank 1 from rank 25 by 0 _._ 0001, and 92% of flexible-arm cells at L40–L44 read above 0 _._ 999, so a depth profile drawn from it alone is compressed exactly where the effect is largest; Appendix G makes that concrete on one instance. From the same rank computation we therefore also record _Lz_ = _−_ log10(1+ _|{v_ : _λv > λz}|_ ), which spends its resolution near the top of the vocabulary, and _Mz_ , the log-probability margin against frequency-matched controls, and report every transport result under all three. They place the depth peak about fifteen layers apart, and the _Rz_ and _Lz_ depth profiles correlate at only _r_ = 0 _._ 581, so a conclusion holding under one and not another is a fact about the measure, and we say which.

### 2.2 Checkpoints, and the benchmark

The primary checkpoint is Qwen3.6-27B: 64 layers, _d_ = 5120, vocabulary 248 _,_ 320, published lens over all 63 source layers. It is a _hybrid_ : only every fourth layer carries standard attention, with 24 heads, the other 48 a gated delta net. That governs what a “head” means here and is load-bearing in §4. Instruction-tuned checkpoints throughout, since the arms differ only in what their instruction asks for. Four more test how far each claim travels: Qwen3.5-9B, phi-4 and gemma-4-31B-it on lenses _we fitted_ , and Llama-3.1-8B-Instruct on a published one. Each cleared a tokenizer screen and, where a lens claim rests on it, the homogeneity check above; two further candidates failed those screens (Appendix E).

Each semantic instance of **JGateBench** yields five prompts over an _identical_ context, so every contrast is within-instance and paired. _Automatic_ needs _z_ but never exposes it; _report_ asks for _z_ ; _flexible_ passes _z_ through an operator defined in the prompt; _control_ is a format-matched instruction that needs _z_ for nothing; _supplied_ applies the same operator to a value _given_ in the prompt. The primary contrast is flexible against control, which carries the identical "Answer:" format and no demand for _z_ . The main family reads language identity off FLORES-200 (NLLB Team, 2022), whose _N_ - way parallelism lets a counterfactual vary the latent variable and nothing else; the second tracks objects under swaps, where _z_ is a progressively updated state. Behaviour is scored throughout as a forced choice among the four candidate answers at the final position, two in the yes/no control arm, never as an open-vocabulary argmax.

Five design choices are invariants rather than preferences (Appendix A). The load-bearing one is **label symmetry across arms** : a flexible-arm operator table naturally prints "Spanish -> 7", putting the gold label in one prompt and not another, so the contrast partly measures whether the label was printed. We share the table across every arm and enforce the symmetry in code, because deliberately breaking it inflates the headline effect by 26%: a large, clean, artifactual result in the predicted direction.

## 3 Demand changes visibility, not availability

**Visibility rises with demand, at matched accuracy.** Table 1 gives the paired contrasts over 200 semantic instances. Against the format-matched control, lens visibility rises by ∆ _Rz_ = +0 _._ 0891 [+0 _._ 0799 _,_ +0 _._ 0983], and by +2 _._ 79 [+2 _._ 57 _,_ +3 _._ 01] in the non-saturating margin, while behavioural accuracy is matched to within noise of _exactly_ zero, both arms at 0 _._ 940, so the effect is not a difficulty difference wearing a representational costume. The match is on raw accuracy: the control question is binary, so above their own floors the arms read 0 _._ 920 and 0 _._ 880, a residual well inside the interval the match is quoted with. One measurement choice matters later. We average _Rz_ over the workspace band (L24–L59, the depth-verified band of the prior survey) rather than maximising it, and the profile that mean summarises peaks at L26 under _Rz_ and at **L41** under _Lz_ , where it reaches +3 _._ 50 [+3 _._ 36 _,_ +3 _._ 63] decades against +1 _._ 22 at L26, _inside_ the window §4 localises causally, where the percentile scale had put it ten layers below.

|contrast|∆_Rz_|∆_Mz_|∆accuracy|
|---|---|---|---|
|flexible_−_control +0_._0891|[+0_._0799_,_+0_._0983]|+2_._79 [+2_._57_,_+3_._01]|+0_._000 [_−_0_._045_,_+0_._045]|
|l  report_−_control +0_._0991| [+0_._0891_,_+0_._1092]| +3_._52 [+3_._31_,_+3_._73]| +0_._040 [+0_._005_,_+0_._075]|
| flexible_−_supplied +0_._0504| [+0_._0445_,_+0_._0567]| +4_._05 [+3_._81_,_+4_._28]| _−_0_._060 [_−_0_._095_, −_0_._030]|
| supplied_−_control +0_._0387| [+0_._0310_,_+0_._0466]| _−_1_._26 [_−_1_._37_, −_1_._15]| +0_._060 [+0_._030_,_+0_._095]|

Table 1: Lens visibility by condition pair on Qwen3.6-27B, under the saturating percentile rank and the non-saturating margin, both averaged over the workspace band. Paired bootstrap over semantic instances (10 _,_ 000 resamples, _n_ = 200), the instance being the resampling unit because all five arms share a passage. Arm accuracies are _control_ 0 _._ 940, _report_ 0 _._ 980, _flexible_ 0 _._ 940, _supplied_ 1 _._ 000; _control_ is a yes/no question where the others offer four candidates, so measured above its own chance floor it reads 0 _._ 880 against _flexible_ ’s 0 _._ 920. The first row is matched on prompt format and on raw accuracy; the last two separate the operator from the latent variable, and the operator’s own contribution is the one quantity here that changes sign with the measure.

**The variable is there either way.** A multinomial probe on the residual stream at the query position, trained and tested on _disjoint semantic instances_ , decodes _z_ far above chance in every arm. It is a _single_ weight matrix fitted jointly across the four arms with no per-arm adjustment, and we quote the nested form, choosing the layer on an inner split of the training instances only: 0 _._ 575 in control to 0 _._ 808 in report over five split seeds, against a floor built by permuting labels under the identical rule, which reaches 0 _._ 090 where nominal chance is 0 _._ 05, so 6 _._ 4–9 _._ 0 _×_ the floor. One matrix serving all four arms gives the strong form: the same linear map reads _z_ whatever the task asks, including in the arm that needs it for nothing. The dissociation is graded rather than absolute, since the within-arm diagonal itself rises with demand (Figure 8).

**Holding the operator fixed, on four checkpoints.** Matching format and accuracy does not make the arms computationally equivalent: _flexible_ must infer _z and_ apply an operator to it, where _control_ does neither. The _supplied_ arm separates the two by handing the operator a value in the prompt: a _dummy_ candidate rather than the true one, since naming the true one would pin its rank at _∼_ 1 by copying. With operator use held approximately fixed, needing to infer the value adds +0 _._ 0504 [+0 _._ 0445 _,_ +0 _._ 0567] on Qwen3.6-27B, and +4 _._ 05 [+3 _._ 81 _,_ +4 _._ 28] in the margin, the largest effect in Table 1. The same contrast is positive on every checkpoint we ran it on: +0 _._ 0429 [+0 _._ 0371 _,_ +0 _._ 0493] on phi-4, +0 _._ 0346 [+0 _._ 0288 _,_ +0 _._ 0408] on Llama-3.1-8B and +0 _._ 0855 [+0 _._ 0743 _,_ +0 _._ 0967] on gemma-4-31B (Figure 4c), and positive on all four under the nonsaturating readout too, though it reorders them (Appendix E). Four architectures and four lenses, two of them fitted here.

The _operator’s own_ contribution is a warning rather than a result, and we report it as one. It runs +0 _._ 039, +0 _._ 178, +0 _._ 029 and _−_ **0** _._ **059** on those same four checkpoints, not stable in sign. It reverses on the primary checkpoint between the two readouts, +0 _._ 0387 in percentile rank against _−_ 1 _._ 26 [ _−_ 1 _._ 37 _, −_ 1 _._ 15] in the margin, and it tracks per-arm accuracy rather than the model (Appendix E). The decomposition needs accuracy-matched arms, we have them on Qwen3.6-27B alone, and the split is scoped there. Nor is it a variance decomposition even so: the interaction is significantly negative, _−_ 0 _._ 0487 [ _−_ 0 _._ 0571 _, −_ 0 _._ 0405], so entering the table by the other path makes inference alone account for 111%. The four conditional effects are the quantities that mean something.

**Replication.** The effect survives a change of task family (tracking gives flexible _−_ control +0 _._ 1792 [+0 _._ 1625 _,_ +0 _._ 1961], twice the language effect) and a change of model _and_ lens, at +0 _._ 2931 [+0 _._ 2775 _,_ +0 _._ 3084] for report _−_ control on Qwen3.5-9B. That second one runs through _report_ , whose gold concept is also its next token, so part of its visibility is imminent production rather than availability; the flagship contrast is immune, since its answer is an operator symbol. Each needed its own scope statement, and the two arrived from opposite directions (Appendix F).

Figure 2: The causal chain, measured three ways on the language family; all intervals are paired bootstrap intervals over instances. **(a)** Where a donor’s value can be _installed_ , read three to five layers above the patch so every depth is measured at matched distance, in the non-saturating _Lz_ at _n_ = 120, averaged over four donor pairings; per-cell means, and how many pairings each clears, are in Table 4. **(b)** The behavioural counterpart at _n_ = 150: donor-symbol rate minus the matched per-distractor rate. The shaded band is where the patch still leaves the task solvable; the effect is significant above it too, at accuracies of 0 _._ 17–0 _._ 21. **(c)** Specificity of the mediating direction at _n_ = 150, against a null of one point per wrong concept.

## 4 The variable is transported, not unmasked

Everything below substitutes a donor’s activation and reads the concept through the lens, so the quantity is _donor-substitutable, lens-readable_ content at the query position. A weak patching effect is therefore not an absent representation: an early encoding that is nonlinear, lens-misaligned or destroyed by substitution would read as weak. What we establish is the narrower thing the admission account denies: that the later readable, causally substitutable form at that position is produced predominantly by gathering rather than by unmasking a representation already in place. Readout depth is a second axis and not a setting. Fixing it late grades _survival_ , since a shallow patch is then measured after many layers in which the model can overwrite it, while a fixed short distance grades installation; conflating the two produces a spurious onset, and we produced one before separating them.

**Transport is concentrated in the window, by at least seventeenfold.** Read three to five layers above the patch, the depth profile is sharply peaked (Figure 2a). Over _four_ donor pairings the L39 stream carries 24 _._ 8–37 _._ 8 _×_ the largest cell anywhere below the window under _Lz_ , the weakest of the four intervals still reaching 17 _._ 0, and 51 _._ 6–62 _._ 6 _×_ under _Mz_ ; the shallow maximum is taken inside each resample, so the ratio is conservative. Under _Rz_ it is 1 _._ 1–2 _._ 7 _×_ with every interval covering 1: the percentile scale compresses the window’s own effect, which is the sharpest case of the saturation §2 documents and the reason no depth claim here rests on it. Below the window only two of twentyfour cells clear zero in all four pairings, both in the residual stream and neither in attention or MLP, and across L24–L33 not one of twelve transports positively under either rank readout, with intervals excluding _Rz_ effects above 0 _._ 0044.

**The window is demand-specific, not a property of the geometry.** The passage sits in the same place in every arm, so a window arising only from where the readout is taken should appear in an arm that needs the latent for nothing. It appears, and it is much flatter: on the same donor pairing the _control_ arm concentrates at 4 _._ 2 _×_ [2 _._ 9 _,_ 6 _._ 6] against _flexible_ ’s 30 _._ 3 _×_ [18 _._ 6 _,_ 53 _._ 4], intervals not overlapping. The difference sits in the attention branch at the gathering layer, which transports 16 _._ 5 _×_ as much under demand (+1 _._ 610 against +0 _._ 098 in _Lz_ ), where at the window’s lower edge the two arms are within a fifth of each other. Some of the variable travels whatever the task asks; what demand changes is the gather at the peak.

**Attention carries most of it; no MLP in the window carries any.** At L39 the attention output and the residual stream are indistinguishable under _Rz_ (+0 _._ 0135 against +0 _._ 0137), but that nearequality belongs to the saturating metric, and _Lz_ separates them at +1 _._ 618 against +1 _._ 908, so attention carries 85% of the stream’s effect rather than all of it. Independent patch effects are not additive in a nonlinear network, so neither figure proves it carries the whole; separating them needs path patching (Goldowsky-Dill et al., 2023), which we have not run. Nor is this a property of fullattention layers as a class: L15 and L27 are also full-attention and neither transports significantly. Meanwhile mlp.L39 _opposes_ the transport at _−_ 0 _._ 0301 under _Rz_ and _−_ 0 _._ 400 under _Lz_ , the grid’s largest negative cell under both readouts and negative in all four pairings (Table 4). The second family is drawn in Figure 3 and this one in Figure 2a. Only attention can route content _between_ positions, so the MLP cells test the narrower question of whether a local MLP output independently carries donor-aligned content, and inside the window they do not. That much is convergent, and Gurnee et al. (2026) already report heads relaying J-space content between positions; our increment is the _dependence_ , since the arms share an identical context and the amount transported changes with what the task asks.

**The second family replicates the branch and spreads the gather.** On tracking at _n_ = 140, under four donor pairings, resid.L39 transports +0 _._ 0346 to +0 _._ 0505, surviving Benjamini–Hochberg over 42 components in all four; so do the attention blocks at L39 and L48 and the prespecified head attn.L39.H15 at a fifth of the stream; mlp.L39 is negative in all four. What does _not_ carry over is the concentration: on language attn.L48 is a twenty-fifth of attn.L39 under _Lz_ and one head matches the whole block, where on tracking the two blocks are comparable and no head dominates. L48 is a _linear_ -attention layer, so no head decomposition is possible there (Appendix D). Gupta et al. (2026) report head dominance shifting with task complexity in an unrelated model class. Outside the window the largest positive MLP point estimate is mlp.L15, which no pairing resolves from zero, so we scope the claim to the window rather than asserting it of MLPs in general. The defensible claim is: _a task latent variable is gathered into the query position by attention within a mid-depth window, inside which no tested MLP output contributes positively; how the gather is shared across that window’s attention layers is task-dependent._

## 5 The window has two measured edges

Target and donor are both flexible records with their own randomised operator tables, so a patch carries a _specific_ predicted wrong answer: the target’s table applied to the _donor’s_ language. That symbol appears nowhere in the donor’s prompt, and producing it requires the target’s table to be applied _after_ the transport: compositional rather than copying. The control is an identity. With _C_ the candidate set, _sd_ the donor’s predicted symbol and _D_ = _C \{s_ gold _, sd}_ the distractors, we report

Under the null that a patch destroys the computation and leaves the answer uniform on _C_ , both terms equal 1 _/|C|_ and E[ ∆]^ˆ = 0. Comparing against the clean run instead has expectation 0 _._ 25 here, which is how one of our own runs produced six confident false positives.

**The window.** At _n_ = 150 the counterfactual is null through L33, becomes detectable at **L36** (+0 _._ 113 [+0 _._ 057 _,_ +0 _._ 177], task accuracy 0 _._ 807) and plateaus over L39–L45 at +0 _._ 227 to +0 _._ 400 across three pairings (Figure 2b). Neither edge is a step, and we name no peak: across pairings L39, L42 and L45 trade places. Significance runs on to L57, but accuracy has fallen from 0 _._ 920 to 0 _._ 19 by L48, so the interpretable cells are L36–L42; deeper, a patch replaces the late computation wholesale and the donor’s symbol arrives without anything having been transported. The band is narrower than that reads, and we state it plainly: L36 is the only cell with comfortable accuracy and it is the one cell the adversarial destruction bound of §6 does not clear, while L39 and L42, which do clear it, sit at 0 _._ 553 and 0 _._ 533, just above the floor at which we ourselves stop calling a patch a substitution. Patched-position count is an axis too: the final query token alone leaves the flip rate within noise, +0 _._ 0250 [ _−_ 0 _._ 0250 _,_ +0 _._ 0813] against +0 _._ 3375 [+0 _._ 2188 _,_ +0 _._ 4562] over twelve, while the graded margin there clears zero, +0 _._ 550 [+0 _._ 271 _,_ +0 _._ 831], so that position receives transport without the choice flipping, and the behavioural claim concerns a query _span_ read at its final token (Figure 6, Appendix C).

**The lower edge is survival, and repair is the likeliest reason.** Because only the last tokens are patched, the passage stays the target’s, and layers above a patch can attend back to it and re-derive the target’s own value. Patching resid at L33–L45 and reading the _donor’s_ value at a fixed later band measures survival directly: a value installed at L33 is entirely gone, +0 _._ 0048 [ _−_ 0 _._ 0010 _,_ +0 _._ 0113]; at L36 about half survives, +0 _._ 0453 [+0 _._ 0326 _,_ +0 _._ 0596] against the +0 _._ 0888 asymptote; from L39 it saturates. So the lower edge is a survival boundary: the content is transportable earlier, it just does not last. _Why_ is an inference and we mark it as one. Re-derivation from the still-present passage is the reading we favour, and the fact that survival saturates from L39 rather than growing with the number of layers left is a point in its favour; but overwriting, attenuation through the normalisers, and plain incompatibility with the surrounding activations all predict the same disappearance. Deciding between them needs the target’s passage masked after an early patch, which we did not run (Appendix C).

**No tested set of component writes reconstructs the stream’s effect.** Patching _every_ attention and MLP output across L36–L42 (fourteen components, Table F) reaches +0 _._ 0250 [ _−_ 0 _._ 0167 _,_ +0 _._ 0833] against resid.L42’s +0 _._ 4250 [+0 _._ 2833 _,_ +0 _._ 5667] in the same run. That is weaker than showing the writes are irrelevant, since simultaneously substituted outputs may be mutually inconsistent in a nonlinear network, the interaction term Khemais (2026) derives in closed form. The answer depends on the accumulated stream as we can measure it; the gather is necessary, not sufficient.

### 5.1 The same window on asecond architecture

Absolute layer indices are not comparable across checkpoints, so the cross-architecture unit is fractional depth. On gemma-4-31B-it (62 layers, dense, another family, and a measurement that never touches a lens) the sweep is null or unstable at or below fractional depth 0 _._ 55 and becomes interpretable at exactly one cell, L37 at 0 _._ 60: +0 _._ 170, +0 _._ 107 and +0 _._ 140 under three donor pairings, at task accuracy 0 _._ 74 to 0 _._ 77 (Figure 4b). Qwen3.6-27B’s interpretable cells sit at 0 _._ 56–0 _._ 66. Above that, gemma-4 does what Qwen3.6-27B does above L45 only more sharply: the effect jumps to +0 _._ 55–+0 _._ 65 while accuracy collapses to 0 _._ 09–0 _._ 15, the largest wholesale-copying signal we have measured and the best argument in the paper for reporting accuracy beside every counterfactual rate. We did not run the lens-side sweep there, so the attention/MLP asymmetry is untested on this checkpoint; Appendix E splits L37 by component kind under one pairing.

A third checkpoint shows what the design needs. Llama-3.1-8B-Instruct solves the flexible arm at 0 _._ 610, and that is not enough headroom: every cell whose effect clears zero does so at a patched accuracy of 0 _._ 34 or below, and the only cells above 0 _._ 39 are null. Nothing on it is interpretable under the criterion we apply to the other two. That is a scope condition on the design rather than evidence against the window: the sweep needs a checkpoint starting near ceiling, which is why gemma-4-31B-it at 0 _._ 970 resolves it and this one does not.

## 6 What asingle donor pairing is worth

Donors are drawn from the whole instance pool, so a different seed reassigns every pair. That seed is a free design parameter, and on the tracking sweep it decides a great deal: Benjamini–Hochberg at _q_ = 0 _._ 05 returns 21, 19, 15 and 23 significant components of 42 under four pairings, and only 14 survive all four. A single pairing therefore reports 7–64% more significant components than are stable, depending on the draw, and the excess is almost entirely _individual heads_ : ten of the twelve unstable cells are heads of L39 at point estimates of _±_ 0 _._ 001–0 _._ 005, while stream-level and whole-block cells are stable. We know of no activation-patching work that reports this sensitivity. It bounds what a head-level claim from one pairing is worth, ours included, which survives because it exceeds every unstable cell in every pairing. Four of our sweeps are reseeded and reported as ranges: tracking, the counterfactual depth sweep, the mediation and the language matched-distance grid of §4. The L39 head decomposition is single-pairing, and we say so wherever its numbers appear.

The second nuisance is in Equation 2, which is unbiased under _uniform_ destruction; real destruction is not uniform. With the realised candidate set recorded per trial the residual is exact, and its sign follows the pairing: +0 _._ 0043 under one and _−_ 0 _._ 0154 under another. So it is a _±_ 0 _._ 015 nuisance of the draw rather than a bias of the instrument, and the correct statement asserts no direction. Under an adversarial bound in which _every_ pair is as bad as the worst one observed, L39–L45 survive on both pairings and L36 does not (Appendix C). What the control cannot do is separate transport from wholesale copying; that guard is task accuracy, not the estimator.

## 7 What the behaviour runs through

**Selective necessity.** Ablating a component to a _leave-one-out_ per-condition mean (so no record is ablated partly towards itself) and comparing the accuracy cost in the arms that need _z_ against the format-matched control gives a selectivity measure. The strongest dissociation is resid.L39: report loses _−_ 0 _._ 600 [ _−_ 0 _._ 740 _, −_ 0 _._ 460] (0 _._ 980 _→_ 0 _._ 380) and flexible _−_ 0 _._ 120 [ _−_ 0 _._ 220 _, −_ 0 _._ 040], while the control arm _improves_ by +0 _._ 080 [+0 _._ 020 _,_ +0 _._ 160]. It is the only component clearing zero here; attn.L39.H15 points the same way in the flexible arm alone and does not, _−_ 0 _._ 060 [ _−_ 0 _._ 140 _,_ +0 _._ 000]. The control-arm nulls are not a measurement floor: the harsher _zero_ ablation of mlp.L39 costs _control −_ 0 _._ 220 [ _−_ 0 _._ 340 _, −_ 0 _._ 120] and leaves the other two arms at exactly 0 _._ 000 (Table 6).

**One derived direction carries part of it.** A residual patch carries everything the donor computed, so the effect might run through the concept’s lens component or through something else travelling with it; the test removes the candidate mediator and keeps the rest. The direction is derived rather than fitted. Since the readout is _WU_ norm( _Jℓhℓ_ ), the vector _Jℓ_^(_⊤WU_[)^(_z_] is the)^_static_pre-normaliser lens vector for token _z_ , not the exact logit gradient, whose projection under RMSNorm removes the component along the activation and so removes identically none of it (Appendix C). Projecting the static vector out of a 5120-dimensional residual costs +0 _._ 113 to +0 _._ 193 of the counterfactual answer rate at L39 and +0 _._ 080 to +0 _._ 147 at L42 across three pairings, 19–46% of the rate on the same trials, with task accuracy recovering correspondingly.

Four controls run on those same pairs, tabulated with the L42 cell in Appendix C: a random direction rescaled to strip _exactly_ as much activation norm costs nothing ( _−_ 0 _._ 013 to +0 _._ 007), and the mean over rival concepts is null ( _−_ 0 _._ 017 to +0 _._ 015), while the gold direction costs +0 _._ 113 to +0 _._ 193 and survives orthogonalising against the other 19 (+0 _._ 127 to +0 _._ 213); the worst of twenty rivals reaches +0 _._ 014 to +0 _._ 042. Two nulls and two positives make the effect hard to attribute to the projection rather than to the concept (Figure 2c). The wrong-concept null is a distribution rather than a draw: each pair runs against the 19 languages that are not its donor, pooled into one estimate per concept identity, and its worst cell is a maximum over twenty, hence the range. Because the 20 concept directions have mean pairwise cosine 0 _._ 50, projecting out the gold one also strips most of every other; orthogonalising against the remaining 19 leaves the effect intact, as does the surgical variant that removes only what the substitution introduces. One limit remains: at least half survives removal, so something else in the stream contributes and we do not identify it, which is why we say the behaviour _runs through_ the direction rather than that it is mediated by _z_ (Figure 7).

## 8 Areadout shift is not ameasure of use

The clearest single fact here is negative, and about method rather than this model. At L39, patching the attention output from a mismatched donor moves the lens readout toward the donor’s language by +0 _._ 1333 [+0 _._ 0871 _,_ +0 _._ 1847], 4 _._ 9 _×_ the shift in the target’s own language on the same patched trials, and the same component is the only one whose effect reverses with the direction of substitution (Appendix F). It nonetheless newly selects the predicted counterfactual answer under forced choice in 1 **of** 80 **pairs** , with its H15 slice in none, where the residual stream at the same layer (same donors, same positions, same pairs) selects it in 29. A thresholded flip rate cannot tell an inert patch from one that moves the answer without overtaking it, so we measured behaviour continuously: the attention patch is _not_ inert, raising the donor symbol’s margin over the distractors significantly, about a seventh as far as the stream does.

On the same trials the three components move the donor’s readout in _Lz_ by +1 _._ 83, +1 _._ 68 and +1 _._ 63, within 12% of one another, while their effect on the donor-minus-distractor margin differs 7 _._ 4 _×_ , at +2 _._ 56, +0 _._ 44 and +0 _._ 35 (Figure 5). Trial-level correlations between the two run +0 _._ 27, +0 _._ 18 and +0 _._ 08: weak, but for two of the three genuinely nonzero. So the readout is not uninformative about behaviour; it is _uncalibrated_ : its magnitude cannot be read as causal influence when three components at the same readout level do three very different things. That component patch-ing can activate pathways disconnected from the output is known (Makelov et al., 2023; Zhang & Nanda, 2024). What we add is a controlled instance on the Jacobian lens, where that inference is routinely made, with a matched _positive_ control: §7 shows the concept-aligned component is not epiphenomenal in general. The stakes are not hypothetical: Prosvirnin et al. (2026) score models for safety from top- _k_ J-space tokens per layer; our result motivates a causal step before any readout difference is read mechanistically.

## 9 Related work

Gurnee et al. (2026) establish the automatic/flexible distinction, the lens and J-space, and report attention heads relaying J-space content between positions, so the mechanism is not open in general and we do not claim it is. They leave open what causes a representation to _enter_ the workspace. Four differences make our transport result an increment: their relay heads are selected from weights and ablated at every token position, so relay is not separated from same-position writing; their relay analysis and demand contrast are never crossed; readout depth is never an independent axis, so installation is not separated from survival; and their causal results are on closed models.

Concurrent systems build the gate we went looking for (Wu et al., 2026; Sathish, 2026; ChateauLaurent & VanRullen, 2025): construction against discovery, and our answer is that their gate is not where the architecture puts it, because there is no decision to take over a variable not yet at the readout position. Our gather belongs to an established literature on attention moving content into the prediction position (Geva et al., 2023; Wang et al., 2023; Todd et al., 2024); it differs in that the variable’s _value_ is never named, so there is no source token to copy, and the amount transported moves with demand, not the prompt. Our depth-swept patching follows Meng et al. (2022) and Vig et al. (2020). Our latent is language identity, studied by Wendler et al. (2024) and Dumas et al. (2025); we hold the passage fixed and ask what makes it readable at the queried position. Closest in shape to our negative result, Bersia & Gaintseva (2026) find concept-specific blind spots where the target “remains decodable inside the oracle”. Adjacent work covers attention-gated routing, causal head gating, intervention mechanics and course-correction under steering (Frank, 2026; Nam et al., 2025; Kang et al., 2026; McKenzie et al., 2026).

## 10 Limitations

What we rule out is an admission step that unmasks a variable already sitting at the query position. A gate on the attention _route_ stays consistent with everything here: passage-attention mass at the gathering layer is 1 _._ 90 _×_ as demand-sensitive as at the 72 heads of three non-gathering layers, median against median, though maximum against maximum is only 1 _._ 16 _×_ , and it is measurable only on full-attention layers (Appendix D). So the honest statement is narrower than our title: we refute admission _at the query position_ , and what sets the amount transported is plausibly a gate on the route we do not identify.

On scope, the entry effect is measured on five checkpoints and two task families; the mechanism on far less: the window on two architectures, and the attention route, the head, the mediation and the ablations on Qwen3.6-27B and the language family alone. We localise the gather but not the computation feeding it: no passage span we patched transports the variable, leaving its pre-gather form unlocalised rather than distributed. The gather is necessary without being sufficient, with mediation partial. And _supplied_ sits at accuracy 1 _._ 000 against _flexible_ ’s 0 _._ 940, a ceiling in the 2 _×_ 2 that no analysis removes.

## 11 Conclusion

An admission account presupposes a step that does not happen where the readout is taken: the variable is decodable from every arm, and the measurement sits where it has to be _moved_ to. Attention does the moving, necessarily but not sufficiently, at one fractional depth in two architectures and sevenfold sharper under demand. And even there, the size of a readout shift says nothing about what a component does to the answer.

## Reproducibility statement

Every number here is written by a command-line entry point into a JSON or JSONL artifact on disk; none is transcribed. The benchmark generator, the five design invariants and their guard tests are in Appendix A; the checkpoint revisions, lens files, decoding settings and what is seeded are in Appendix H; the cross-architecture screens, including the two candidates that failed them, are in Appendix E. Figures are built from artifacts by code rather than transcribed, and an audit command matches every numeric literal in this manuscript against the artifact tree. Where a quantity depends on the donor pairing we report its range over pairings and name the sweeps for which we have only one (§6).

### AI use statement

We used generative AI assistants to help with coding the project and for feedback on our experiment design. We did not use them to interpret results, to develop or prove any mathematical claim, or to generate any benchmark item. We have reviewed all AI-assisted work and take responsibility for the final content of this paper.

## References

- Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. _arXiv preprint arXiv:2303.08112_ , 2023.

- Tobias Bersia and Tatiana Gaintseva. When activation oracles learn not to read: Concept-specific blind spots in fine-tuned oracles. _arXiv preprint arXiv:2607.23379_ , 2026.

- Hugo Chateau-Laurent and Rufin VanRullen. Learning to chain operations by routing information through a global workspace. _arXiv preprint arXiv:2503.01906_ , 2025.

- Cl´ement Dumas, Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. Separating tongue from thought: Activation patching reveals language-agnostic concept representations in transformers. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics_ , pp. 31822–31841, 2025. arXiv:2411.08745.

- Gregory N. Frank. How alignment routes: Localizing, scaling, and controlling policy circuits in language models. _arXiv preprint arXiv:2604.04385_ , 2026.

- Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In _Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2023. arXiv:2304.14767.

- Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, and Aryaman Arora. Localizing model behavior with path patching. _arXiv preprint arXiv:2304.05969_ , 2023.

- Aditya Gupta, Rahul Kumar, Sourav Mandal, and Nikhil Deshpande. Where computation lives inside tabpfn: Causal localisation of attention head function. _arXiv preprint arXiv:2606.12917_ , 2026.

- Wes Gurnee, Nicholas Sofroniew, Jack Lindsey, et al. Verbalizable representations form a global workspace in language models. _Transformer Circuits Thread_ , 2026. URL https:// transformer-circuits.pub/2026/workspace/index.html. arXiv:2607.15495.

- Diancheng Kang, Zheyuan Liu, Ningshan Ma, Yue Huang, Zhaoxuan Tan, and Meng Jiang. Promptactivation duality: Improving activation steering via attention-level interventions. _arXiv preprint arXiv:2605.10664_ , 2026.

- Anis Khemais. A theory of conditional collapse under low-rank weight-space ablations. _arXiv preprint arXiv:2608.03620_ , 2026.

- Aleksandar Makelov, Georg Lange, and Neel Nanda. Is this the subspace you are looking for? an interpretability illusion for subspace activation patching. In _NeurIPS Workshop on Attributing Model Behavior at Scale_ , 2023. arXiv:2311.17030.

- Alex McKenzie, Keenan Pepper, Stijn Servaes, Martin Leitgab, Murat Cubuktepe, Mike Vaiana, Diogo de Lucena, Judd Rosenblatt, and Michael S. A. Graziano. Endogenous resistance to activation steering in language models. _arXiv preprint arXiv:2602.06941_ , 2026. Code: github.com/agencyenterprise/endogenous-steering-resistance.

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In _Advances in Neural Information Processing Systems_ , 2022. arXiv:2202.05262.

- Andrew Nam, Henry Conklin, Yukang Yang, Thomas Griffiths, Jonathan Cohen, and Sarah-Jane Leslie. Causal head gating: A framework for interpreting roles of attention heads in transformers. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2025. arXiv:2505.13737.

- NLLB Team. No language left behind: Scaling human-centered machine translation. _arXiv preprint arXiv:2207.04672_ , 2022.

- Roman Prosvirnin, Victor Minchenkov, Alexey Soldatov, and Vladimir Bashun. Silent alarm: A j-space protocol for comparing danger recognition across models and quantization levels. _arXiv preprint arXiv:2607.12792_ , 2026.

- Sharath Sathish. Recognition-gated workspace steering: Pratyabhij˜n¯a as an engineering specification for language model control. _Preprints.org_ , 2026. doi: 10.20944/preprints202607.0895.v1. Preprint, not peer reviewed.

- Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. Function vectors in large language models. In _International Conference on Learning Representations (ICLR)_ , 2024. arXiv:2310.15213.

- Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart M. Shieber. Investigating gender bias in language models using causal mediation analysis. In _Advances in Neural Information Processing Systems_ , 2020.

- Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In _International Conference on Learning Representations (ICLR)_ , 2023. arXiv:2211.00593.

- Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. Do llamas work in english? on the latent language of multilingual transformers. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics_ , pp. 15366–15394, 2024. arXiv:2402.10588.

- Junde Wu, Jiayuan Zhu, Fengling Liu, Minhao Hu, and Jiazhen Pan. J-cot: Chain-of-thought in j-space. _arXiv preprint arXiv:2607.21981_ , 2026.

- Fred Zhang and Neel Nanda. Towards best practices of activation patching in language models: Metrics and methods. In _International Conference on Learning Representations (ICLR)_ , 2024. arXiv:2309.16042.

### Athe benchmark, and the invariants it enforces

**One instance, all five arms.** Instance lang_000169, verbatim apart from the passage. Every arm receives the same prompt up to and including the blank line, then one instruction:

Reference table: Finnish -> W Japanese -> K Swedish -> V Turkish -> Z

<FLORES-200 passage, three sentences>

[instruction]

The five instructions, each followed by _\_ nAnswer:, with the gold answer it is scored against and the size of its candidate set:

|arm|instruct|ion|gold (_|C|_)|
|---|---|---|---|
|_automatic_|Conti|nue the passage. Write the next sentence.|next sentence (4)|
|_report_|What|language is the passage written in?|Japanese(4)|
|_flexible_|What|symbol corresponds to the language of the passage?|K(4)|
|_control_|Does|the passage contain a question mark?|No(2)|
|_supplied_|Treat|the language as Swedish. Which symbol is it?|V(4)|

Four things in this one instance are the design: the reference table names _every_ candidate symbol in every arm, which is label symmetry and is also why absolute _Rz_ levels are not unprompted representation strength; _control_ shares the "Answer:" format while asking something the passage answers without its language being identified; _supplied_ names _Swedish_ , a dummy and not the passage’s actual Japanese, so its gold is V rather than K and no copying route can produce it; and the passage is quoted, never generated. The passage is elided here; it is recoverable from the released records by instance id.

**Family sizes.** The language family generates 1954 records over 400 semantic instances, of which 354 are complete in all five conditions; 46 instances are dropped from the _automatic_ arm, the limiting one, because two candidate languages share a continuation first token. The four-arm subset used by every measurement other than the 2 _×_ 2 is drawn from the same 354 instances, 200 of them at the cap the entry measurements use, taken by sorted identifier so that two runs at one cap see the same instances. Twenty distinct languages survive filtering. Prompt lengths are 107 / 140 / 235 tokens (min / median / max), all clear of the lens’s unfitted position floor. FLORES-200 supplies 1012 sentences per language.

### The five invariants, and the failure each prevents.

1. **Label symmetry across arms.** A flexible-arm operator table naturally prints the gold label, putting it in one prompt and not the other, so the contrast partly measures whether the label was printed and returns a large clean artifactual effect in the predicted direction. We share the table across every arm and check programmatically that every arm names the label or none does. Deliberately breaking it gives ∆ _Rz_ = +0 _._ 1126 [+0 _._ 1015 _,_ +0 _._ 1240] against the matched design’s +0 _._ 0891: an inflation of +0 _._ 0235, about 26% of the matched effect, with 79% of the unmatched effect surviving the guard. The unmatched design also breaks accuracy matching (1 _._ 000 against 0 _._ 875), so it is worse on two axes at once.

2. **The label never appears in the passage.** A concept present in the context reads at rank _∼_ 1 at its own position.

3. **Forced choice, never open-vocabulary argmax.** On the primary checkpoint the top token is often ’ _\_ n _\_ n’ with the gold answer at rank 1: 0 _._ 144 against 0 _._ 955 on the same trials.

4. **Score the continuation form.** " Spanish" and "Spanish" are different token ids, worth 0 _/_ 36 against 10 _/_ 36 on a real checkpoint.

5. **Single-token-ness is per-tokenizer and must be filtered.** On this tokenizer _no digit_ is single-token in continuation form (" 7" is two tokens), so numeric latents are unscoreable against a token-indexed lens here, which rules out any family whose answer is a number. All 20 language names pass; 9 of 18 operator symbols do, and all 9 survivors are letters.

**Statistics.** All intervals are 95% bootstrap intervals over 10 _,_ 000 resamples, clustered on the semantic instance because the arms share a passage. Multiple comparisons are controlled within each sweep, over all of that sweep’s components, by Benjamini–Hochberg at _q_ = 0 _._ 05: 39 cells for the language matched-distance sweep, 42 for tracking, 36 for the L39 head decomposition, 12 for each depth sweep. **Correction runs independently inside each donor pairing** , because four pairings are four experiments rather than four draws to average (§6). The entry contrasts of Table 1, the mediation, the necessity table and the readout-against-use comparison are pre-specified single comparisons, reported uncorrected. We give an absolute gap beside every ratio, and refuse ratios whose denominator approaches zero.

**Selection provenance, and a confirmation split.** L39 and H15 were chosen by looking at data, and it matters which data. The layer came from a coarse screen of 26 components over **20** semantic instances, the head from a decomposition over **25** . Both were frozen at that point and never revisited, and every downstream experiment ran afterwards on strictly larger instance pools from the same family. The union of the two screens is 25 instances, so most of every downstream result rests on instances the selection never saw, which gives a confirmation split for free:

||selection pool (_n_= 25)|held out (_n_= 55)|
|---|---|---|
|resid.L39counterfactual|+0_._400 [+0_._200_,_+0_._600] |+0_._364 [+0_._236_,_+0_._491] |
|mediation,resid.L39|+0_._120 [+0_._000_,_+0_._280] |+0_._218 [+0_._109_,_+0_._327] |
|mediation,resid.L42|+0_._080 [+0_._000_,_+0_._200]|+0_._127 [+0_._055_,_+0_._218]|

The counterfactual holds on unseen instances, and the mediation is _stronger_ there than on the selection pool, where it does not clear zero at _n_ = 25, so that result is carried entirely by instances the layer choice never touched. Beyond the table, the attn.L39 bidirectional signature is slightly stronger held out (+0 _._ 0572 against +0 _._ 0440), necessity and the early-layer cells hold at comparable or larger magnitudes, and **one secondary cell loses significance held out** while the pre-L36 representational cells are underpowered at _n_ = 25 rather than refuted. **The head screen cannot be split, because it** **_is_ the selection pool.** Selecting instances by sorted identifier correlates with nothing that matters: FLORES row index ( _ρ_ = _−_ 0 _._ 078), passage length ( _ρ_ = _−_ 0 _._ 103), gold language ( _χ_^2 , _p_ = 0 _._ 87), candidate set or operator symbol. What the split cannot do is validate the choice of L39 itself: a different screen might have picked a different layer, and §4 shows the second task family does exactly that.

### Bthe gauge argument, and the sparse decomposition

using norm( _aℓ−_ 1 _oℓ−_ 1) = norm( _oℓ−_ 1). Every block still reads through a scale-free normaliser, so it computes the same function of the same argument, and the logits _WU_ norm(˜ _oL−_ 1) are unchanged. Under this reparameterisation

Two scope notes the body compresses. The gauge **must vary with depth** : a global rescale scales _oℓ_ and _oL−_ 1 alike, leaves _Jℓ_ untouched, and establishes nothing. And a depth-varying _aℓ_ needs a scalar on the residual path to absorb _aℓ_ +1 _/aℓ_ , which a plain pre-norm skip does not have, so the reparameterised network computes the same function, which is all the argument needs, but it is not the same architecture, and we do not claim the gauge is realisable in-architecture.

We verified the identity numerically rather than asserting it. On Qwen3.5-9B in fp32, with _aℓ_ = 1 below layer 20 and 2 at or above it, the largest change in any model logit is 1 _._ 4 _×_ 10^(_−_5) and the largest change in _Rz_ over 18 band layers is 4 _._ 0 _×_ 10^(_−_6) , both at fp32 rounding, while the mean diagonal of _Jℓ_ (by Hutchinson’s estimator on central-difference Jacobian-vector products) moves by exactly 2 _._ 0000 at layer 19 and 1 _._ 0000 at layer 20, as Equation 3 predicts. The diagnostic moves; the model and the dependent variable do not. **The check has to run in fp32** : in bf16 the same construction drifts by _∼_ 3% from accumulation alone, which blurs an exact identity into a small residual and establishes nothing.

Two further notes. The lens readout is invariant _without refitting_ because _Jℓhℓ_ is homogeneous of degree one; a learned affine translator is not, since its bias does not scale with _hℓ_ , so one fitted in a given parameterisation must be refitted in another. And the closest treatment we know of, Belrose et al. (2023) on drifting covariance and high-variance “rogue” dimensions, is empirical rather than an invariance argument. The screen is also load-bearing for checkpoint selection: it is what excluded one candidate whose gated MLP is homogeneous of degree 1 _._ 76–1 _._ 93 (Appendix E).

**The sparse decomposition.** Running full-dictionary gradient pursuit over all 248 _,_ 320 atoms at L39 without prefiltering the vocabulary (support _k ≤_ 25, a 24-second CPU computation, released with the artifacts) gives a normalised gold coefficient _Sz_ correlating with _Rz_ at Spearman 0 _._ 24: a concept can enter the support at rank 13 _,_ 332, and can rank third and stay out of it. This is an approximation, not a certified global optimum (the problem is combinatorial). _Sz_ cannot serve as a dependent variable here: it is exactly zero in 90 _._ 9% of activations, and in the two arms whose _Rz_ varies most it is nonzero in 4 of 200 (automatic) and 1 of 200 (control), so it supports no depth profile, no regression on layer and no correlation with behaviour. It also moves by 9 _._ 1 _×_ with the free parameter _k_ . Support _membership_ is better behaved, and the demand contrast is positive under it as under everything else. We therefore measure visibility, say so, and report the decomposition as a check.

### Cthe behavioural window, in detail

**The window’s cells, three donor pairings.** The significance decision is the same in all twelve cells under all three pairings. Point estimates move, so we give the range rather than one draw:

|patch layer|pairing 0|pairing 1|pairing 2|worst bound|accuracy|
|---|---|---|---|---|---|
|L24–L33|_−_0_._007|_−_0_._017|_−_0_._020|all null|0_._91–0_._93|
|L36|+0_._113|+0_._090|+0_._067|+0_._010|0_._807|
|L39|+0_._367|+0_._300|+0_._400|+0_._220|0_._553|
|L42|+0_._367|+0_._320|+0_._377|+0_._237|0_._533|
|L45|+0_._227|+0_._330|+0_._337|+0_._133|0_._453|
|L48–L57|+0_._143to+0_._277|+0_._230to+0_._293|+0_._150to+0_._320|+0_._037|0_._17–0_._21|

“Worst bound” is the lower end of the least favourable of the three bootstrap intervals, so a positive entry means the cell clears zero under every pairing. Two of the three pairings put L45 level with L42 and one puts it well below, so **no pairing supports naming a peak layer** . A smaller _n_ = 60 subset of these instances re-analyses to the same shape, with L36 at +0 _._ 067 [ _−_ 0 _._ 025 _,_ +0 _._ 167]: non-significant there, significant at _n_ = 150, and the point estimate barely moved.

**The destruction control, computed exactly.** Equation 2 is unbiased under uniform destruction. The destroyed answer distribution is significantly non-uniform (passage-position patches, _χ_^2 with _p_ = 2 _×_ 10^(_−_21) ; per-symbol hit rates 0 _._ 061 to 0 _._ 460), and that is a property of the model rather than of our design, since the operator-table construction is indistinguishable from uniform at _p_ = 0 _._ 74. With the realised candidate set recorded per trial the residual is exact rather than bounded in expectation, and its sign follows the donor pairing: +0 _._ 0043 at pairing 0 and _−_ 0 _._ 0154 at pairing 2, from mean destroyed rates of 0 _._ 1121 / 0 _._ 1100 in the donor and distractor roles at the first and 0 _._ 1080 / 0 _._ 1144 at the second. The worst single pair reaches +0 _._ 1133 and +0 _._ 2171 respectively, which supports an adversarial bound in which _every_ pair is as bad as the worst observed:

|patch layer|survives the mean correction|survives “every pair is the worst pair”|
|---|---|---|
|L39|yes, both pairings|yes(+0_._1701/+0_._0962)|
|L42|yes, both pairings|yes(+0_._1700/+0_._0695)|
|L45|yes, both pairings|yes(+0_._0201/+0_._0229)|
|L36|yes (+0_._1090/+0_._0821)|noon either (_−_0_._0566/_−_0_._2071)|
|L48–L57|yes|no,_and that is the point_|

L39–L45 therefore survive a bound stronger than anything the paper claims elsewhere, and L36 is robust to the measured residual and to the pairing but not to the adversarial bound. The deep cells fail it for a substantive reason rather than an estimator artifact: from L51 a residual patch transplants the donor’s near-final computation, which emits the donor’s symbol without transporting a latent variable. The contamination control shows the same thing from another angle. The predicted counterfactual symbol coincides with the _donor’s own_ gold answer in 8 of 60 pairs; excluding those moves L39 from 0 _._ 433 to 0 _._ 442 and L48 from 0 _._ 483 to 0 _._ 481, but moves L57 from 0 _._ 367 to 0 _._ 288, by far the largest move of any layer, and exactly where wholesale copying should bite.

**The span curve.** Over _m_ patched query tokens, flips reach +0 _._ 1250 [+0 _._ 0437 _,_ +0 _._ 2125] at four positions with distractors flat at 0 _._ 025, then +0 _._ 294, +0 _._ 313, +0 _._ 331 and +0 _._ 338 at six, eight, ten and twelve, while accuracy falls from 0 _._ 900 to 0 _._ 550. Four positions is the cleanest cell, significant with accuracy still 0 _._ 800; the widest span is partly overwriting. At one position the flip rate does not clear zero but the graded margin does, +0 _._ 550 [+0 _._ 271 _,_ +0 _._ 831].

**Survival, and what a passage patch cannot test.** Reading the donor’s value at a fixed later band (L46–L51) after a patch at L33–L45, at _n_ = 100 and span 1: L33 +0 _._ 0048 [ _−_ 0 _._ 0010 _,_ +0 _._ 0113] (null), L36 +0 _._ 0453 [+0 _._ 0326 _,_ +0 _._ 0596], L39 +0 _._ 0871, L42 +0 _._ 0877, L45 +0 _._ 0888. The target’s own value is suppressed monotonically over the same range, _−_ 0 _._ 0020, _−_ 0 _._ 0217, _−_ 0 _._ 0261, _−_ 0 _._ 0365. Both hold in a second, smaller sample to the digit ( _−_ 0 _._ 002, _−_ 0 _._ 017, _−_ 0 _._ 024, _−_ 0 _._ 035; L36 at 51% of asymptote against 45%) across two spans and two sample sizes.

The repair account predicts that removing the source blocks survival. Patching the passage’s last twelve tokens from the same donor transports +0 _._ 0007 to +0 _._ 0025 to the answer position on its own, and adds nothing on top of a query patch: at L39 and L42 the query-only, split-span and both-span arms agree to four decimal places (+0 _._ 0346, +0 _._ 0346, +0 _._ 0347). The intervention is therefore too weak to test the account, because the language is established across the whole passage rather than its tail; blocking re-derivation properly means overwriting the passage entire, which is a different and far more destructive experiment. We record it as a dead end whose cause is a property of the task, not of the model.

**Competition, in the per-trial data.** Suppression tracks patch depth rather than donor load, and within a layer the trial-level correlation between donor installed and target lost is _positive_ (+0 _._ 156 at L39 and +0 _._ 553 at L33), where a capacity trade-off predicts negative. The narrower claim is what we make: deep substitution displaces the incumbent value.

**The mediating direction, both derivations.** Since the readout is _WU_ norm( _Jℓhℓ_ ), the _static_ pre-normaliser lens vector for token _z_ is _Jℓ_^(_⊤WU_[)^(_z_],while the exact logit gradient is)^(_∇hλz_=) _Jℓ_^(_⊤DN_()^(_Jℓh_))^(_⊤WU_[)^(_z_],whichforRMSNormweightstheunembeddingrowbythegain)^_andre-_ _moves the component along the activation_ . That second property makes the exact gradient unusable as a mediator: the normaliser is scale-free, so the readout does not change along the activation’s own direction, and projecting the gradient out removes identically none of the activation. We verified both derivations against automatic differentiation. The static vector is the one with overlap to remove, so the correction is to the name rather than to the intervention.

**The mediation controls.** The body table gives the absolute projection at L39. All of it, as ranges over three pairings:

|component|direction projected out|absolute projection|from_hd −ht_|
|---|---|---|---|
|resid.L39|the concept’s own (gold)|+0_._113to+0_._193|+0_._138|
||gold, orthogonalised vs the other 19|+0_._127to+0_._213|+0_._200|
||random, same activation norm|_−_0_._013to+0_._007|_−_0_._013|
||rivals, mean of20|_−_0_._017to+0_._015|+0_._005|
||rivals, worst of20|+0_._014to+0_._042|+0_._029|
|resid.L42|the concept’s own (gold)|+0_._080to+0_._147|+0_._087|
||gold, orthogonalised vs the other 19|+0_._100to+0_._127|+0_._125|
||random, same activation norm|_−_0_._013to+0_._007|+0_._000|
||rivals, mean of20|_−_0_._003to+0_._010|—|
||rivals, worst of20|+0_._014to+0_._042|—|

Activation norms removed, so that a control which removes almost nothing is visible as such: gold 13 _._ 63, orthogonalised 9 _._ 50, random 13 _._ 63 by construction, rivals 9 _._ 6. Dose response for the gold direction at 0 _._ 25/0 _._ 50/0 _._ 75/1 _._ 00 is +0 _._ 062, +0 _._ 125, +0 _._ 150, +0 _._ 188, and flat at every dose for every control.

Figure 3: The second task family, in _Rz_ at _n_ = 140, the readout the four-pairing table below reports, named in the panel title because the language panels use _Lz_ . **(a)** The branch asymmetry replicates: the stream and the attention output transport and the MLP at L39 opposes, but the attention effect is shared between L39 and L48 rather than concentrated at one layer as it is on language; L48 is a linear-attention layer. **(b)** The 24 heads of L39 in both families, one donor pairing each; the language series is _n_ = 60. On language one head matches both the block and the whole stream to three decimal places; on tracking the same head participates at about a fifth of its own family’s stream, which four pairings confirm is nonzero (table below). Language depth profile: Figure 2a.

**The mediated** **_share_ is the least stable quantity in the paper and we do not quote it as a point.** Computed consistently from observations (gold-removal cost over the full-arm donor-symbol rate on the same paired trials, never from a stored summary string), it reads 45 _._ 8%, 33 _._ 3% and 43 _._ 9% at L39 and 36 _._ 7%, 27 _._ 3% and 18 _._ 5% at L42 across the three pairings, against 50 _._ 0% and 31 _._ 0% at _n_ = 80. It is a ratio of two pairing-sensitive quantities, which is why it behaves worst of anything we measure; the smaller run’s L42 value sits inside its range and its L39 value above it, which is why the share is reported as a range and never as a point.

### Dtransport and the attention route, in detail

**What the convergent literature says, verbatim.** Geva et al. (2023) block attention edges during factual recall and find the attribute extracted to the prediction position by attention heads, with early MLPs instead enriching the subject position. Wang et al. (2023) identify heads “active at END” that “attend to previous names in the sentence, and copy the names they attend to”. Todd et al. (2024) find that “a small number [of] attention heads transport a compact representation of the demonstrated task”, with the strongest causal effects in middle layers, the same depth range as our window. And Gurnee et al. (2026) report that “a subset of attention heads relays J-space content between positions, and ablating it selectively disrupts behaviors that depend on that relay”. None of these manipulates task demand over a fixed context, and none separates installation depth from survival depth, which is where our increment lies.

**The second family, four donor pairings.** At _n_ = 140, distance 3–5 mean, _Rz_ , with Benjamini– Hochberg run inside each pairing:

|component|p0|p1|p2|p3|BH|
|---|---|---|---|---|---|
|resid.L39|+0_._0382|+0_._0346|+0_._0377|+0_._0505|4/4|
|resid.L42|+0_._0416|+0_._0348|+0_._0410|+0_._0411|4/4|
|resid.L48|+0_._0478|+0_._0526|+0_._0466|+0_._0619|4/4|
|attn.L39|+0_._0304|+0_._0186|+0_._0164|+0_._0271|4/4|
|attn.L48|+0_._0118|+0_._0204|+0_._0226|+0_._0094|4/4|
|attn.L39.H15|+0_._0090|+0_._0079|+0_._0058|+0_._0094|4/4|
|attn.L39.H0|+0_._0130|+0_._0121|+0_._0105|+0_._0116|4/4|
|attn.L39.H5|_−_0_._0148|_−_0_._0149|_−_0_._0136|_−_0_._0117|4/4negative|
|mlp.L39|_−_0_._0156|_−_0_._0113|_−_0_._0148|_−_0_._0117|4/4negative|
|mlp.L42|+0_._0003|_−_0_._0004|+0_._0035|+0_._0025|0/4, null throughout|
|attn.L39.H17|+0_._0028|+0_._0020|+0_._0012|+0_._0036|2/4,_not stable_|

attn.L48 and attn.L39.H15 both clear zero under _Rz_ , _Lz_ and _Mz_ under every pairing, and survive correction under _Rz_ and _Mz_ under every pairing and under _Lz_ in three of four, with the failing pairing differing between the two cells, so their metric-dependence is a property of one draw. mlp.L42 is null in all four pairings, and attn.L39.H15 is nonzero on this family. Correction also promotes a cell we would otherwise not mention: attn.L39.H5 is significantly _negative_ , so L39 on tracking is mildly opposed at one head while participating at others.

The head-level instability of §6 is worth quantifying here, because this sweep is where we found it. Survivor counts per pairing are 21 / 19 / 15 / 23 of 42 under _Rz_ , 23 / 18 / 16 / 22 under _Lz_ and 26 / 25 / 21 / 29 under _Mz_ , but the four-way intersections are only 14, 13 and 18. Ten of the twelve cells significant in some pairings and not others are attn.L39.H* at point estimates of _±_ 0 _._ 001–0 _._ 005.

**The head, and the source work’s open question.** H13 (+0 _._ 0099) and H14 (+0 _._ 0093) each carry roughly half of H15’s effect on language, which is why we call the head sufficient _for the block’s transport effect_ rather than uniquely necessary. The largest head effect at the layer is not H15 but its neighbour H16, in the other direction: _−_ 0 _._ 0421 [ _−_ 0 _._ 0724 _, −_ 0 _._ 0196], twice H15’s magnitude. So the layer contains a head that installs the donor’s value and one that opposes it, which is consistent with the gather being selective routing and is a further reason the head is not the whole story. At _n_ = 60 and readout distance 4, attn.L39.H15 gives +0 _._ 0194 [+0 _._ 0098 _,_ +0 _._ 0323], the whole attention block +0 _._ 0193 [+0 _._ 0098 _,_ +0 _._ 0321] and the whole residual stream +0 _._ 0193 [+0 _._ 0097 _,_ +0 _._ 0322]; the same coincidence holds at distances 3 and 5. This run is single-pairing.

Gurnee et al. (2026) note that the absence of verbalizable content in early layers may be real _or_ may be lens degeneracy there, leaving open that a “true workspace” extends earlier. Our behavioural counterfactual never touches the lens and is also null everywhere below L36. The boundaries are not identical: the lens-derived measure already transports at L36 while the behavioural one is weakest there. But both are far weaker below L36 than inside the window, and two measures with disjoint failure modes agreeing there is evidence that the shallow weakness is **real** . This is a claim about _concentration_ , not absence: the lens-derived measure does find small significant shallow transport under _Lz_ , and what the two agree on is that the shallow band carries far less.

**The attention route.** Measuring the route needs equal-length instructions across arms, which our released wording does not have, so the family was regenerated with four 14-token arms sharing a nine-token identical tail; the visibility effect survives the rewrite (+0 _._ 1091 [+0 _._ 0965 _,_ +0 _._ 1217] against +0 _._ 0891 [+0 _._ 0799 _,_ +0 _._ 0983]). On those records, one clean forward pass each and no patching, comparing L39 against the 72 heads of L15, L27 and L51 at _n_ = 200: L39’s _largest_ head against the controls’ _median_ gives 4 _._ 5 _×_ , median with median 1 _._ 90 _×_ , maximum with maximum 1 _._ 16 _×_ . We report the middle one. A permutation that shuffles head labels among all 96 heads makes a randomly chosen layer score 3 _._ 46 _×_ on the asymmetric max-over-median statistic, which is why we neither quote that statistic nor attach a _p_ -value to any of them: the inferential unit for a claim about a layer is the layer, and we measured four. Correction is likewise uninformative here, since 94 of 96 heads survive at _q_ = 0 _._ 05. H15 shifts by +0 _._ 1546 [+0 _._ 1486 _,_ +0 _._ 1603], the 91 _._ 7th percentile of the 72 control heads, and is positive against the same head index at all three control layers. Twenty-three of L39’s 24 heads shift toward the passage, which is not distinctive: L27 gives 21 of 24 and L51 gives 20 of 24. On the single final token, identical across arms by construction, a control layer leads instead, so the effect is a property of the nine-token span and we report it as such.

Figure 4: What travels across checkpoints. **(a, b)** The distractor-controlled counterfactual against fractional patch depth at _n_ = 150 pairs per pairing, on the same axes, with all three donor pairings drawn (the thick line carries the bootstrap band) and task accuracy dashed. Where accuracy falls below 0 _._ 5 the patch has replaced the late computation, and the shading marks that a donor-symbol rate there is not evidence of transport. **(c)** The entry effect with the operator held fixed ( _flexible − supplied_ ) at _n_ = 200 instances per checkpoint is positive on all four; the operator’s own contribution ( _supplied − control_ ) is not stable in sign.

- E CROSS-ARCHITECTURE SCREENING, AND THE CHECKPOINTS THAT FAILED IT

Five checkpoints carry a number in this paper and two more were measured and excluded; the screens table below covers the remaining four, Qwen3.5-9B being a replication of the entry effect alone, on its own lens. Nothing here is inferred from a model family: every exclusion is a measurement.

**The screens.** A candidate must (i) tokenize all 20 language names and enough operator symbols in _continuation_ form, with the leading space forming a distinct token; (ii) pass the homogeneity check of Appendix B if a lens-derived claim is to rest on it; (iii) be dense, since “the instruction changed” against “the router picked different experts” is exactly this paper’s confound; and (iv) for the depth sweep only, solve the flexible arm near ceiling. One candidate fails (i):

|checkpoint|languages single-token|operator symbols|" X"_̸_="X"|
|---|---|---|---|
|Qwen3.6-27B|20/20|9/18|yes, 4 of 4 probes|
|phi-4|20/20|9/18|yes, 4 of 4|
|Llama-3.1-8B-Instruct|20/20|9/18|yes, 4 of 4|
|gemma-4-31B-it|20/20|9/18|yes, 4 of 4|
|Mistral-Small-3.2-24B|7/20|18/18|no, 0 of 4|

The Mistral tokenizer strips the leading space: " Spanish" and "Spanish" are the same id, and " " encodes to the empty list, so invariant 4 does not hold there and “score the continuation form” is not the same operation as on the other four. A cross-model claim would be comparing two different measurements. Separately, one Gemma-3 checkpoint fails (ii) with a gated MLP homogeneous of degree 1 _._ 76–1 _._ 93, and its published lens underflows in fp16, which is why gemma-4 rather than gemma-3 carries the cross-architecture window. We re-checked gemma-4 against the same criteria rather than inheriting the family’s exclusion: it is dense, it passes the tokenizer screen, and we fitted its lens ourselves.

**Per-arm accuracy, which is what makes the** 2 _×_ 2 **non-comparable.** Re-pooled from each run’s per-trial file rather than from a stored summary:

|checkpoint|automatic|control|supplied|report|flexible|operator share|
|---|---|---|---|---|---|---|
|Qwen3.6-27B|0_._615|0_._940|1_._000|0_._980|0_._940|43%|
|Llama-3.1-8B-Instruct|0_._815|0_._680|1_._000|0_._995|0_._610|46%|
|phi-4|0_._805|0_._990|0_._860|0_._970|0_._515|81%|
|gemma-4-31B-it|0_._360|1_._000|1_._000|0_._845|0_._970|—|

A ceiling or floor arm is a design failure rather than a finding, and both failures are present. Qwen and Llama measure the operator-only contrast against an arm at exactly 1 _._ 000, so it has no behavioural variance, and those are the two whose shares agree, to three points. phi-4 escapes that and has the opposite problem, a _flexible_ arm failing half the time. gemma-4 has both arms of the operator contrast pinned at 1 _._ 000 and a significantly _negative_ operator effect, _−_ 0 _._ 0585 [ _−_ 0 _._ 0691 _, −_ 0 _._ 0476], so its share is not a percentage of anything and we print no number rather than a large absurd one. Also: Llama’s _control_ arm at 0 _._ 680 sits _below_ its automatic arm at 0 _._ 815, the only checkpoint where that happens, so its format-matched control is not the easy arm it is elsewhere.

What is comparable across checkpoints is the operator-controlled contrast, and it is positive on all four:

|checkpoint|latent demand (flexible_−_supplied)|operator alone (supplied_−_control)|
|---|---|---|
|Qwen3.6-27B|+0_._0504 [+0_._0445_,_+0_._0567] |+0_._0387 [+0_._0310_,_+0_._0466] |
|phi-4|+0_._0429 [+0_._0371_,_+0_._0493] |+0_._1778 [+0_._1590_,_+0_._1970] |
|Llama-3.1-8B|+0_._0346 [+0_._0288_,_+0_._0408]|+0_._0292 [+0_._0178_,_+0_._0409]|
|gemma-4-31B|+0_._0855 [+0_._0743_,_+0_._0967]|_−_0_._0585 [_−_0_._0691_, −_0_._0476]|

Interactions on the same runs are _−_ 0 _._ 0487 [ _−_ 0 _._ 0571 _, −_ 0 _._ 0405], _−_ 0 _._ 1864 [ _−_ 0 _._ 2059 _, −_ 0 _._ 1672], _−_ 0 _._ 0499 [ _−_ 0 _._ 0624 _, −_ 0 _._ 0379] and +0 _._ 0046 [ _−_ 0 _._ 0075 _,_ +0 _._ 0169] in that order: sub-additive on three and null on gemma-4. Note that gemma-4 is also the best accuracy-matched of the four on the latent-demand contrast, 0 _._ 970 against 1 _._ 000. It gives the largest effect _in percentile rank_ and the smallest under the non-saturating readout (+0 _._ 1679 [+0 _._ 1441 _,_ +0 _._ 1938] against Qwen’s +1 _._ 3134 [+1 _._ 2548 _,_ +1 _._ 3718], phi-4’s +1 _._ 3921 [+1 _._ 3155 _,_ +1 _._ 4668] and Llama’s +1 _._ 5438 [+1 _._ 4644 _,_ +1 _._ 6223]), so the ordering across checkpoints is a fact about the measure. All four clear zero under both. An accuracy-matched _supplied_ arm on a second checkpoint would settle the decomposition properly; it is not run and is not cheap.

**The window on gemma-4-31B-it.** Sixty-two layers, dense, clean flexible accuracy 0 _._ 970. The counterfactual is lens-free, so nothing here inherits from a lens artifact. Three donor pairings, _n_ = 150:

|patch layer|frac. depth|pairing 0|pairing 1|pairing 2|accuracy|
|---|---|---|---|---|---|
|L22–L31|0_._35–0_._50|+0_._027|_−_0_._017|+0_._000|0_._93–0_._96|
|L34|0_._55|+0_._093^_∗_|+0_._027|+0_._047^_∗_|0_._87|
|L37|0_._60|+0_._170^_∗_|+0_._107^_∗_|+0_._140^_∗_|0_._74–0_._77|
|L40|0_._65|+0_._547^_∗_|+0_._650^_∗_|+0_._653^_∗_|0_._09–0_._15|
|L43–L55|0_._69–0_._89|+0_._120to+0_._197|+0_._013to+0_._247|+0_._027to+0_._287|0_._17–0_._27|

Starred cells exclude zero. L37’s intervals are +0 _._ 170 [+0 _._ 103 _,_ +0 _._ 237], +0 _._ 107 [+0 _._ 047 _,_ +0 _._ 170] and +0 _._ 140 [+0 _._ 073 _,_ +0 _._ 210]; L34’s least favourable is +0 _._ 027 [ _−_ 0 _._ 017 _,_ +0 _._ 070], which is why it is two of three. **L37 is the only cell significant under all three pairings at an accuracy that still permits a reading.** The cells from L40 upward carry the largest effects in this paper and none of them is transport.

Splitting that layer by component kind, **at one donor pairing only** : resid.L37 +0 _._ 1700 [+0 _._ 1033 _,_ +0 _._ 2367] at accuracy 0 _._ 760, attn.L37 +0 _._ 0200 [ _−_ 0 _._ 0067 _,_ +0 _._ 0500] at 0 _._ 960, and mlp.L37 +0 _._ 0200 [ _−_ 0 _._ 0033 _,_ +0 _._ 0500] at 0 _._ 960. So no component reproduces the stream’s behavioural effect here either, which replicates the joint-component negative of §5 on a second architecture. It is _not_ a replication of §8: that result is about a readout shift without a behavioural effect, and we have no readout measurement on this checkpoint. attn.L40 reaches +0 _._ 0400 [+0 _._ 0067 _,_ +0 _._ 0767] at accuracy 0 _._ 940, the one significant component-level behavioural cell here.

Figure 5: A readout shift is not a calibrated measure of use, on the same 80 trials throughout, one donor pairing. **(a)** Three components at L39 shift the donor concept’s readout to within 12% of one another while their effect on behaviour differs 7 _._ 4 _×_ . **(b)** Per trial the two are related but weakly: uncalibrated rather than uninformative.

Figure 6: The behavioural window and both of its edges. **(a)** Rates against patch depth at _n_ = 150, one pairing: the donor’s predicted symbol, the matched per-distractor rate, and task accuracy. From L48 the distractor rate approaches chance while accuracy is below 0 _._ 2, which is destruction rather than transport. **(b)** The distractor-controlled effect; the shaded band marks the cells where accuracy still permits a reading. **(c)** The lower edge is a survival failure, at _n_ = 100 and span 1: a value installed at L33 is gone by L46, and the target’s own value is progressively suppressed as the donor’s is installed.

**The checkpoint the design cannot use.** Llama-3.1-8B-Instruct clears every screen and still cannot resolve the window, because it solves the flexible arm at 0 _._ 610. At _n_ = 250, span 12: L12 (fractional depth 0 _._ 38) is null at accuracy 0 _._ 396; L14 (0 _._ 44) reaches +0 _._ 150 [+0 _._ 068 _,_ +0 _._ 232] but at accuracy 0 _._ 208; L16 +0 _._ 174 [+0 _._ 088 _,_ +0 _._ 260] at 0 _._ 172; L18 and L20 are significant at 0 _._ 19– 0 _._ 21; L22 and above are null. At span 1 the pattern repeats with slightly more headroom: L14 +0 _._ 136 [+0 _._ 060 _,_ +0 _._ 212] at accuracy 0 _._ 336, L12 null at 0 _._ 500. **No cell is both significant and at an accuracy comparable to the interpretable cells on the other two checkpoints** , where the analogous figures are 0 _._ 53–0 _._ 81. The distractor rate is also high throughout (0 _._ 34–0 _._ 52), which is what a patch does when the clean computation has little margin. Its entry effect replicates normally, on the one published lens in our set that ships both a configuration file and a convergence trace, and that trace shows the fitter stopping on a convergence delta at 461 of a requested 1000 prompts, which is worth noting because the flagship 27B artifact ships neither file while its name records _n_ = 1000.

### Fsupporting figures and tables

**One component carries the demand setting in both directions.** The bidirectional screen is exploratory and reported uncorrected over its 52 components, so it supports no significance claim of its own. In it exactly one component shows an effect that reverses with the direction of substitution:

Figure 7: **(a)** Necessity by _leave-one-out mean_ ablation, 50 instances per arm. A component is selectively necessary when it costs the arms that need _z_ and not the format-matched control; only resid.L39 clears zero under this ablation, and mlp.L39 is the one cell pointing the other way. That the control arm is measurable at all is shown by the harsher zero ablation in Table 6, not by this panel. **(b)** Mediation at _n_ = 150, one pairing: the loss in counterfactual answer rate from projecting out the concept’s static lens vector, and from the same vector orthogonalised against the other nineteen concepts, against a null drawn one point per wrong concept. Paired per instance throughout, so a rival that happened to help would appear left of zero. The null is plotted as one estimate per concept identity rather than as one interval, because a single random control cannot be told from a lucky draw.

Figure 8: Visibility dissociates from availability, _n_ = 200 semantic instances throughout. **(a)** Mean _Rz_ of the gold concept at the query position by layer and condition. The L33 dip appears in all four arms but is about three times deeper in flexible and control than in report and automatic, so it cancels within either pair and not across them. **(b)** Each contrast’s visibility effect against the accuracy difference it must not be; the primary contrast sits at ∆accuracy = 0, and shading marks where difficulty differs enough that the effect is not interpretable. **(c)** A single linear probe trained jointly across all four arms decodes _z_ in every one. **This panel plots the layer-selected form** , whose diagonal runs 0 _._ 64 (control) to 0 _._ 84 (report); the nested figures quoted in §3, 0 _._ 575 to 0 _._ 808, choose the layer on an inner split of the training instances only and are the ones the claim rests on. Bars are the mean over five instance-disjoint splits and whiskers their standard deviation; because the layer is selected, the reference line is the 0 _._ 090 reached by permuted labels under the same selection rule, not the nominal 0 _._ 05.

attn.L39. Writing the flexible arm’s L39 attention output into the automatic run raises visibility by +0 _._ 0517 [+0 _._ 0336 _,_ +0 _._ 0715]; writing the automatic arm’s into the flexible run lowers it by _−_ 0 _._ 0270 [ _−_ 0 _._ 0369 _, −_ 0 _._ 0188]. No other component has both directions excluding zero with opposite signs. We report it because it is the strongest evidence in our own data for the routing-gate reading of §10, and because it is in the released artifacts either way.

**Two uninterpretable arms, and the guard they produced.** At three swaps the tracking family is uninterpretable: the report arm scores 0 _._ 125 against its own 0 _._ 250 chance floor while control scores 0 _._ 915, so ∆accuracy is _−_ 0 _._ 790 for report _−_ control. The run nonetheless returned an entryeffect verdict, because at that point the guard compared accuracy against a fixed constant and had no notion of a per-arm chance floor. This is the run that made us add one, and the stored artifact still carries the pre-guard verdict. At one swap the family works (0 _._ 975 / 0 _._ 940) and the visibility effect is twice the language family’s. A family interpretable at only one difficulty setting is a fact about the family, and we report it rather than only the setting that worked. On Qwen3.5-9B the guard then did its job unaided: the usable contrast is report rather than flexible, because the flexible arm reaches 0 _._ 420 against control’s 0 _._ 985, and the guard refuses flexible _−_ control as confounded at ∆accuracy = _−_ 0 _._ 565 despite a nominal ∆ _Rz_ = +0 _._ 2602, which, had it been reported, would have been the largest visibility effect in the paper and a difficulty difference. Chance floors are per arm and differ within a family: on tracking at one swap they are 0 _._ 25 for flexible and report and 0 _._ 50 for automatic and control, and the automatic arm sits at 0 _._ 510 against its 0 _._ 500 floor, which is why no tracking result here uses an automatic baseline.

|role|artifact|_n_|unit|
|---|---|---|---|
|_selection: small by design, frozen before_|_everything below_|||
|coarse screen (chose L39)|coarse20|20|pairs|
|head decomposition (choseH15)|heads_full|25|pairs|
|bidirectional screen|coarse60|60|pairs|
|_entry and availability_||||
|Stage 1, four arms|s1v2|200|instances|
|Stage 1, five arms (2_×_2)|twobytwo_n200|200|instances|
|Stage 1, phi-4 / Llama / gemma-4|*_stage1|200|instances|
|linear probe|probe2|200|instances|
|_transport_||||
|matched-distance sweep, language (_×_4)|n120_grid_s*|120|pairs|
|matched-distance sweep, control arm|n120_grid_control|120|pairs|
|head sweep, language|A_heads_n60|60|pairs|
|matched-distance sweep, tracking (_×_4)|F_track_gather_n140*|140|pairs|
|attention route, matched lengths|G_route_len|200|instances|
|_behaviour_||||
|counterfactual, depth (_×_3)|cf_resid_depth_n150*|150|pairs|
|counterfactual, component groups|cf_groups|60|pairs|
|counterfactual, span curve_m_= 1_. . ._12|span_m*|80|pairs|
|readout against behaviour, same trials|trial_level|80|pairs|
|survival of an installed value|repair_span1_n100|100|pairs|
|repair source, 4 position modes|repair_source_v2_n100|100|pairs|
|necessity, leave-one-out ablation|loo_necessity|50|instances/arm|
|mediation (_×_3)|wide_n150*|150|pairs|
|_cross-architecture_||||
|depth sweep, gemma-4 (_×_3)|cf_depth_gemma4_n150*|150|pairs|
| component kind, gemma-4|cf_kind_gemma4|150|pairs|
|depth sweep, Llama, spans 12 and 1|cf_depth_llama*|250|pairs|
|homogeneity screens|gauge_s2_k*|—|layers|

Table 2: Every run in the paper with the sample it used, read from the args block of the artifact it names so the table cannot drift from what was run. Artifact names are given without their directory (one per experiment family under the data root) and without the model slug each file carries as a suffix. Instance selection is deterministic (sorted by identifier, then truncated), so a rerun sees the same instances. Where a range over donor pairings is reported the seeds are 0–3; otherwise seed 0. The probe averages five split seeds.

**The largest MLP cell outside the window.** On language, mlp.L15 is the largest positive MLP point estimate outside the transport window at matched distance, +0 _._ 0020 under _Rz_ and +0 _._ 0062 under _Lz_ as the mean over four donor pairings, at a depth where the residual stream itself does not transport significantly. No pairing resolves it from zero (Table 4), so we report it as the least convenient cell we found rather than as a counterexample, and it is why §4 scopes the MLP claim to the window rather than asserting it of MLPs in general.

|train_→_test|raw|per-arm centred|
|---|---|---|
|report_→_flexible|0_._625|0_._825|
|flexible_→_report|0_._525|0_._750|
|control_→_automatic|0_._213|0_._675|
|control_→_flexible|0_._175|0_._625|
|control_→_report|0_._163|0_._613|
|report_→_automatic|0_._175|0_._600|

Table 3: Cross-arm probe transfer, _n_ = 200 instances, 20 classes, chance 0 _._ 05, split by semantic instance. Raw transfer is weak because the arms differ in prompt format and occupy different regions of activation space; removing each arm’s own mean lifts transfer to the level of the within-arm diagonal (0 _._ 64–0 _._ 84; Table F). Both columns are reported because the centring was added after seeing the raw numbers.

|||∆_Rz_|||∆_Lz_||
|---|---|---|---|---|---|---|
|patch layer|stream|attention|MLP|stream|attention|MLP|
|L12|+0_._0028 |_−_0_._0003 |+0_._0004|+0_._020^1 |+0_._004|+0_._001|
|L15|+0_._0071^1|+0_._0009^1|+0_._0020|+0_._023^1|+0_._001|+0_._006|
|L18|+0_._0051^1|_−_0_._0003|+0_._0001|+0_._051^4|+0_._002|+0_._011^1|
|L21|+0_._0031^2|_−_0_._0001|_−_0_._0002|+0_._063^4|+0_._002|_−_0_._001|
|L24|+0_._0003|+0_._0002^1|+0_._0001|+0_._030^1|+0_._004^1|+0_._002|
|L27|_−_0_._0004|+0_._0005^1|_−_0_._0003|+0_._001|+0_._007^1|_−_0_._005^1|
|L30|_−_0_._0069|_−_0_._0015^1|_−_0_._0002|_−_0_._026|_−_0_._015^3|_−_0_._005|
|L33|+0_._0011^1|+0_._0019^2|_−_0_._0009^2|_−_0_._021^1|+0_._010^1|_−_0_._009^1|
|L36|+0_._0073^4|+0_._0026^4|_−_0_._0010^2|+0_._437^4|+0_._104^4|_−_0_._044^4|
|L39|+0_._0137^4|+0_._0135^4|_−_0_._0301^4|+1_._908^4|+1_._618^4|_−_0_._400^4|
|L42|+0_._0228^4|_−_0_._0011^1|_−_0_._0105^4|+1_._596^4|_−_0_._011^2|_−_0_._086^4|
|L45|+0_._0965^4|+0_._0008|_−_0_._0040^2|+1_._062^4|+0_._006|+0_._000|
|L48|+0_._1056^4|+0_._0141^4|_−_0_._0023|+0_._908^4|+0_._064^4|_−_0_._004|

Table 4: Language family, transport at matched readout distance (3–5 layers above the patch), _n_ = 120 pairs, **four donor pairings** . Each cell is the mean over the four, and the superscript counts how many of them have an interval excluding zero; bold is all four. Both rank readouts are shown because they rank the layers differently: _Rz_ makes L48 the largest cell, while _Lz_ , which does not compress the top of the vocabulary, puts the peak at L39 and reduces attn.L48 to a twenty-fifth of attn.L39. The window is where stability lives: every stream cell from L36 to L48 survives all four pairings, while below it only resid.L18 and resid.L21 do, and under _Rz_ no shallow cell does at all. Two cells at L33 hold in only one or two pairings of four: a significantly negative mlp.L33 and a positive attn.L33. Distances 3, 4 and 5 are averaged per instance before pooling. Benjamini–Hochberg at _q_ = 0 _._ 05 is run inside each pairing over this sweep’s 39 cells and changes no cell here. Plotted in Figure 2a ( _Lz_ ).

### G aworked example

_Rz_ is a percentile rank, which is precise and opaque. Figure 9 shows what the readout literally contains for one instance in all four arms.

First, the concept climbs to the top of the readout at L42–L45 in exactly the two arms that need it verbalizably (rank 0 in flexible at both layers, and rank 5 then rank 1 in report), while at those same layers the other two arms hold their own task’s content instead. The control arm, asked whether the passage contains a question mark, reads ’ sentences’, ’ sentence’ and the Chinese word for _sentence_ ; the automatic arm, asked only to continue the passage, reads blanks. Above L48 every arm’s readout reverts to formatting tokens, which is why the band mean is the right summary and a late readout is not. In the flexible arm at L42 the whole top of the readout is the same concept in five surface forms across two scripts.

|patched group|donor_−_distractor|accuracy after|
|---|---|---|
|resid.L42|+0_._4250 [+0_._2833_,_+0_._5667]|0_._450|
|attention, L36+L39+L42| _−_0_._0333 [_−_0_._0833_,_+0_._0167]|0_._883|
|MLP, L36+L39+L42| _−_0_._0250 [_−_0_._0583_,_+0_._0000]|0_._950|
|attention+MLP, L36–L42, all 14 components|+0_._0250 [_−_0_._0167_,_+0_._0833]|0_._950|

Table 5: Joint component groups do not reproduce the stream’s behavioural effect. Clean accuracy 0 _._ 967; _n_ = 60 pairs, twelve query positions, one run and one donor pairing, so that the stream reference and the groups are measured on the same instances.

|ablation|component|report|flexible|control|
|---|---|---|---|---|
|leave-one-out mean|resid.L39 _−_0_._600[ attn.L39 _−_0_._020 attn.L39.H15 +0_._000|_−_0_._740_, −_0_._460] [_−_0_._060_,_+0_._000] [+0_._000_,_+0_._000]|_−_0_._120[_−_0_._220_, −_0_._040] _−_0_._020 [_−_0_._060_,_+0_._000] _−_0_._060 [_−_0_._140_,_+0_._000]|+0_._080[+0_._020_,_+0_._160] +0_._020 [_−_0_._040_,_+0_._080] +0_._020 [+0_._000_,_+0_._060]|
||attn.L43 _−_0_._020|[_−_0_._060_,_+0_._000]|_−_0_._020 [_−_0_._060_,_+0_._000]|+0_._000 [_−_0_._080_,_+0_._080]|
||mlp.L39 +0_._000|[+0_._000_,_+0_._000]|+0_._000 [+0_._000_,_+0_._000]|_−_0_._020 [_−_0_._060_,_+0_._000]|
||resid.L39 _−_0_._440|[_−_0_._580_, −_0_._300]|_−_0_._640 [_−_0_._760_, −_0_._500]|_−_0_._500 [_−_0_._660_, −_0_._340]|
||attn.L39 +0_._000|[+0_._000_,_+0_._000]|_−_0_._080 [_−_0_._160_, −_0_._020]|+0_._060 [+0_._000_,_+0_._140]|
|zero|attn.L39.H15 +0_._000|[+0_._000_,_+0_._000]|_−_0_._080 [_−_0_._160_, −_0_._020]|+0_._060 [+0_._000_,_+0_._140]|
||attn.L43 +0_._000|[_−_0_._060_,_+0_._060]|_−_0_._060 [_−_0_._140_,_+0_._000]|+0_._020 [_−_0_._060_,_+0_._100]|
||mlp.L39 +0_._000|[+0_._000_,_+0_._000]|+0_._000 [+0_._000_,_+0_._000]|_−_0_._220[_−_0_._340_, −_0_._120]|

Table 6: Ablation cost in accuracy by arm: ∆ against the unablated run on the same 50 instances per arm, twelve query positions, clean accuracies report 0 _._ 980, flexible 0 _._ 960, control 0 _._ 920. Under the leave-one-out mean ablation §7 uses, resid.L39 is the only component with any interval excluding zero, and it is selective: it costs both arms that need _z_ and _helps_ the format-matched control. attn.L39.H15 points the same way in the flexible arm but does not clear zero there. The zero rows are the positive control for those nulls: zeroing mlp.L39 costs _control_ alone, which is what shows the control arm is not sitting at a measurement floor. Zeroing resid.L39 damages every arm and so distinguishes nothing; it is printed to make that visible. Plotted, mean rows only, in Figure 7a.

Second, and more usefully as a caution: at L45 the control arm’s _Rz_ is 0 _._ 9953, which sounds like strong representation. The gold token ranks **1155th** there, and nothing in the top of the readout concerns language at all. A percentile close to 1 over a 248 _,_ 320-token vocabulary is compatible with a rank in the thousands, and the failure runs the other way too, since report reads _Rz_ = 1 _._ 0000 at both rank 5 and rank 1. This is why §3 reports the contrast and never the absolute level, and why every mechanistic claim above is read under _Lz_ as well.

### Hartifacts and reproducibility

**Configuration.** The primary checkpoint is Qwen/Qwen3.6-27B: 64 layers, _d_ = 5120, vocabulary 248 _,_ 320, full_attention_interval = 4 so L3, L7, ..., L63 carry standard attention with 24 heads of width 256 and the other 48 layers a gated delta net with 48 value heads of width 128, attn_output_gate = true. The other four are Qwen/Qwen3.5-9B, microsoft/phi-4, meta-llama/Llama-3.1-8B-Instruct and google/gemma-4-31B-it. Weights in bf16, all readout arithmetic in fp32; transformers 5.14.1 with the sdpa attention implementation, except the attention-pattern measurement of Appendix D, which forces eager for one module because sdpa discards the attention matrix. The flash-linear-attention fast path is absent, so the gated delta layers run the reference torch implementation. Seed 0 everywhere except the probe’s five split seeds and the donor pairings, which are seeds 0–3; instance selection is deterministic (sorted by identifier, then truncated), so a rerun sees the same instances rather than a fresh sample.

**Exact revisions.** So that a rerun can be pinned rather than approximated, the Hugging Face revisions actually loaded were Qwen/Qwen3.6-27B at 6a9e13bd, Qwen/Qwen3.5-9B at c2022362, neuronpedia/jacobian-lens at a4114d77, haoranxu/FLORES-200 at 8ecaf1bb and Salesforce/wikitext at b08601e0; the tokenizer is each checkpoint’s own at the same revision. The published 27B lens file is 3 _,_ 303 _,_ 032 _,_ 772 bytes with SHA-256 beginning

Figure 9: One semantic instance, all four conditions, on Qwen3.6-27B. **(a)** The rank of the gold concept ( _Korean_ ) among the vocabulary at the query position, by layer, on a log scale with top1 at the top; shading marks the transport window. **(b)** The top three tokens of the readout at the three layers where the arms diverge, with the gold token bold where it appears. The four arms ask, respectively, whether the passage contains a question mark, for the next sentence, what language the passage is written in, and which symbol corresponds to that language. Ranks in (b) count competing tokens and so start at 0, one below the 1-indexed rank used everywhere else in the paper. Token strings are shown escaped, so ’ _\_ n _\_ n’ is a literal double newline and a leading space is visible.

1718c8c5, and the released five-arm language dataset is 1954 records with SHA-256 beginning 00623f40. Every artifact the tables name is keyed to these.

**Nothing is generated.** No result in this paper decodes text. Every behavioural number is a forced choice: one forward pass, then an argmax over the candidate token ids at the final position, so no decoding parameters enter and no sampling nondeterminism with them. The only stochastic elements are the instance split, the donor–target pairing and the bootstrap resampling, all seeded.

**What each patch replaces, and where it is read.** resid.L _n_ substitutes the _output_ of block _n_ : the residual stream after the block has been added, before block _n_ + 1 reads it. mlp.L _n_ substitutes the MLP submodule’s output. attn.L _n_ is a forward _pre_ -hook on the attention output projection, so it replaces the projection’s _input_ : the per-head concatenation before mixing into the stream, which on the primary checkpoint is the gated head output, attn_output _× σ_ (gate), rather than the raw weighted value sum. attn.L _n_ .H _k_ replaces the _k_ -th in_features _/n_ heads slice of that input, computed per layer from the module rather than from a configured head_dim, because _n_ heads _×_ head_dim _̸_ = _d_ model on both branch types here.

**Where the patch goes and where the readout is taken are separate arguments** , and they have to be: a multi-position patch otherwise drags the measurement off the retrieval cue, and the numbers stay plausible while they stop meaning anything. Every measurement here pins the readout to the answer position independently of the patched span, which matters for the span-12 arms. Mean ablation replaces a component’s output with a **leave-one-out** mean over up to 32 calibration records of the same arm; the record being ablated is removed from its own baseline, as (^() _x − xi_ ) _/_ ( _n −_ 1), so no item is ablated partly towards itself. Calibration records always come from the _same arm_ of the same dataset at the same positions, never pooled across arms and never from an external corpus, since a cross-arm mean would import the contrast being measured.

**The lenses we fitted.** Three of the five checkpoints run on lenses fitted here (Qwen3.5-9B, phi4 and gemma-4-31B-it) and two on published artifacts. _Jℓ_ is a closed-form estimator rather than a trained model: a one-hot cotangent per output dimension is injected at each valid target position and backpropagated, then averaged over valid source positions. Eight output dimensions per backward pass, 512 backward passes per prompt, 120 prompts from Salesforce/wikitext (wikitext-103-raw-v1, train, streamed, passages _≥_ 400 characters not beginning with =), max_seq_len 128, the first 16 positions skipped, accumulated and saved in fp32. Passage distinctness is asserted at fit time rather than assumed. What we cannot supply is the equivalent for the published 27B lens, which ships without a configuration file or a convergence trace; that gap belongs to the artifact we consume, and it is why the replications fit their own.

**The audit.** Every number here is written by a command-line entry point into a JSON or JSONL artifact, and every figure carrying a measured quantity is generated from those artifacts rather than transcribed; Figure 1 is the one exception, a hand-drawn schematic carrying no data. An audit command shipped with the code machine-checks every numeric literal in this document against the artifact tree; the few unmatched ones are software versions, artifact properties and derived quantities. It is a floor on transcription error and not a proof of correctness, so headline numbers are re-derived from their own artifacts, and where a stored summary disagrees with its own observations we re-pool from the observations.

We release the four-arm dataset and its five-arm extension, the generators, the lenses we fitted, and the artifact path for every number, all MIT licensed. Because every prompt quotes a FLORES-200 passage verbatim (NLLB Team, 2022), records ship with the passage replaced by a language code and row offset; one command rebuilds them byte-identically at the revision above, so we redistribute no share-alike text and a rebuilt file is a FLORES derivative under **CC BY-SA 4.0** .
