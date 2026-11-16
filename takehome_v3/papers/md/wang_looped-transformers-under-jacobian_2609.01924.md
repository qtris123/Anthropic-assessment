# Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?

- **Authors:** Wenlong Wang, Fergal Reid
- **Date:** 2026-09-01
- **Source:** https://arxiv.org/abs/2609.01924
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `wang_looped-transformers-under-jacobian_2609.01924`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Wenlong Wang Fergal Reid Fin AI Research Fin AI Research `wenlong.wang@intercom.io fergal.reid@intercom.io`

## Abstract

Recent work identifies a mid-depth band of verbalisable, causally potent representations in a standard feedforward transformer — a functional analogue of a global workspace. Whether the same workspace functionality emerges when depth is implemented through recurrence rather than a stack of distinct layers remains unknown. Looped and depth-recurrent transformers provide a direct test of this question because they reuse the same weights across depth. We extend the Jacobian lens to iterated architectures using a virtual-unrolling adapter. We apply the full workspace suite — lens fitting, readout, and eleven causal experiment families — to Ouro-2.6B (48 layers looped 4 times, deeply supervised) and Huginn-0125 (a 4-layer core recurred 16 times, trained for latent reasoning), using Qwen3.6-27B (64 untied layers) as the standard baseline. We find that a workspace forms in the iterated part of each architecture, but that recurrence changes how it can be accessed. Ouro reconstructs workspace content in every loop, and linear transport cannot carry that content across loop boundaries; writes and ablations must therefore span every remaining loop. Huginn carries content forward across all sixteen recurrences, while reads, writes, and ablations act only within a sliding window of roughly two recurrences. Whether newly injected content can be verbalised tracks explicit per-iteration supervision; whether existing content can be steered does not.

## 1 Introduction

The global-workspace account of cognition holds that flexible behaviour depends on a limited-capacity workspace whose contents are broadcast widely, are reportable, and are open to directed modulation [Baars, 1988, Dehaene and Changeux, 2011, Mashour et al., 2020]. Gurnee et al. [2026] operationalise this picture for language models: using a _Jacobian lens_ — per-layer linear maps fitted to the model’s own mean input–output Jacobians — they identify, in a standard transformer, a mid-depth band of residual-stream representations that are verbalisable, causally necessary for behaviour, and selectively involved in flexible rather than automatic computation.

Their discussion, however, flags a structural disanalogy with the biological theory. In the brain, workspace dynamics are thought to be realised by _recurrent_ processing — ignition, sustained reverberation, broadcast over re-entrant loops. A standard transformer has no built-in recurrence within a forward pass: information transits a fixed stack of untied layers exactly once. The authors observe that a feedforward network can emulate a recurrent one for as many steps as it has layers — a recurrent model being the special case in which weights are tied across layers — but state plainly that they do not know whether depth in a transformer plays a role similar enough to recurrence in a brain to enable the same workspace functionality [Gurnee et al., 2026].

Looped and depth-recurrent transformers allow us to test this question directly because they re-apply tied weights across iterations. Trained, competitive examples of both architectures are now available, so we apply the same J-lens and experimental suite used in the original study. We examine two architecturally distinct members of the weight-tied family, with a standard transformer as the baseline:

- **Ouro-2.6B** [Zhu et al., 2025]: 48 unique layers applied 4 times (virtual depth 192), trained with _deep supervision_ — every loop end is directly optimised to be decodable;

- **Huginn-0125** [Geiping et al., 2025]: a 4-layer core recurred a test-time-variable number of times between a 2-layer prelude and a 2-layer coda, trained for latent reasoning with truncated backpropagation and _no_ explicit deep supervision (we fit at _r_ =16, virtual depth 68);

- **Qwen3.6-27B** : 64 untied layers — the same model family as the original study, reproduced under our harness so all comparisons share one protocol.

We extend the Jacobian lens to iterated architectures with a _virtual-unrolling_ adapter that presents the weight-tied computation graph as a deep feedforward stack, so the original fitting and readout machinery applies unchanged (§3). We then re-run the workspace suite: lens readout across virtual depth, the structural analysis of the transport maps themselves, the Neuronpedia demonstration metrics, and eleven causal experiment families spanning verbal report, steering, flexible generalisation, ablation, and introspection (§4–§7).

**Looped transformers do have a global workspace — but recurrence reshapes its interfaces.** Both weight-tied models exhibit the workspace signature of the original study: their content is verbalisable, causally steerable, and selectively engaged. Depth-as-recurrence therefore does support workspace function. The two architectures differ, however, in how that workspace can be accessed. Linear transport fails at Ouro’s checkpointed loop boundaries and beyond a _∼_ 2-recurrence horizon on Huginn. Writes and ablations follow the same structure: they must span every remaining loop on Ouro but remain confined to the transport horizon on Huginn. The two training regimes also place maintenance and verbalisation on opposite sides of the boundary. This paper makes the following contributions:

- **A method for lensing iterated models.** A virtual-unrolling adapter that presents the weight-tied computation graph to the original fitting machinery as a deep feedforward stack, giving exact pervirtual-layer lenses, together with a validation suite for the silent failure modes recurrence introduces (§3).

- **Transport structure.** Mean-Jacobian transport is bounded by iteration structure: Ouro’s supervised checkpoints decode and re-encode content, destroying the workspace component even where the bulk state survives, and Huginn’s transport horizon is _∼_ 2 recurrences for every fitted target. Weight tying leaves an exact symmetry in the fitted artefacts — independently fitted Huginn lenses at matched offsets agree at cosine 0.9991 — so one iteration’s lens transfers to all (§4).

- **Two maintenance regimes.** Both models refine iteratively with per-task depth, but Ouro _re-derives_ its workspace content each loop through the checkpoint, while Huginn _carries it forward_ — a persistence that injection-path interventions show to be continuous recomputation from the re-injected input, not a memory buffer (§5, §4.5). An occupancy analysis — measured with the original study’s sparse concept inventory, which we implement — controls for lens distance and yields the same geometry: four workspace bands on Ouro, and on Huginn a flat plateau across the recurrent stack with a single step into the coda (§6).

- **Architecture-specific causal interfaces.** In Ouro, an intervention must be maintained across all remaining loops because each loop tends to reconstruct the original content. In Huginn, writes and ablations act only within the transport horizon, where the last lens window alone is exactly as effective as all of them. The two verbal-report experiments produce different results: Huginn reroutes self-computed content into words nearly perfectly (98%) yet never verbalises externally injected content (0/97), which only explicitly deep-supervised Ouro achieves (§7).

- **Validation of lens readouts.** A readout taken beyond a lens’s validated horizon does not read the state. If lens targets are unevenly spaced, apparent trends across depth reflect the instrument rather than changes in the model’s representations. And readout claims can be estimator-specific: two results obtained with the demonstration’s simpler top-8 readout do not survive re-measurement with the concept inventory (Appendix E), and a causal null obtained with a single-direction ablation disappears at ten directions. We document these failure modes and the controls that guard against them (§5.2, §6.1, §7.2).

Table 1: The three models. “Virtual depth” is the number of layer applications in one forward pass; sharing is what distinguishes the families.

||Params|Physical layers|Virtual depth|Intermediate supervision|
|---|---|---|---|---|
|Qwen3.6-27B|27B|64 (untied)|64|none|
|Ouro-2.6B|2.6B|48, looped _×_4|192|deep supervision at every loop end|
|Huginn-0125|3.5B|2 + 4 + 2, core _×r_|68 (_r_=16)|none (truncated BPTT, mean 8)|

## 2 Background and related work

### 2.1 The Jacobian lens

Following Gurnee et al. [2026], the Jacobian lens at source layer _v_ with target layer _t_ is the mean input–output Jacobian of the intervening computation, estimated over a corpus of forward passes. The lens readout at layer _v_ unembeds the linearly transported state, unembed( _hvJv_^(_⊤_),andthe)^_lensdirection_foratoken^_u_isthenormalisedpullback^(_d_=)^_J_ _v_^(_⊤u/∥J_) _v_^(_⊤u∥_ofits) unembedding vector. Directions support causal edits: injection ( _h ← h_ + _αd_ ), ablation ( _h ← h −_ ( _h · d_ ) _d_ ), and clamp-transfer swaps between two tokens’ directions. The original work shows that on a standard transformer the set of layers where these readouts are interpretable and these edits are potent forms a contiguous mid-depth band with global-workspace-like functional properties. The Jacobian lens is related to earlier affine readout methods, including the logit lens [nostalgebraist, 2020] and tuned lens [Belrose et al., 2023]. Unlike those methods, it is derived from the model’s own local linearisation rather than fitted to a decoding objective. Its _failure_ to transport can therefore carry architectural information (§4).

### 2.2 Looped and depth-recurrent transformers

We use _looped transformer_ broadly for any transformer whose forward pass applies a weight-tied block more than once — the lineage of the Universal Transformer [Dehghani et al., 2019] and of parameter-sharing encoders [Lan et al., 2020], recently revived for scaling latent reasoning [Zhu et al., 2025, Geiping et al., 2025] and studied theoretically as a model of iterated computation [Giannou et al., 2023]. Three design axes matter for what follows:

- **Loop scope.** Ouro loops the _entire_ stack: all 48 layers form one weight-tied block, and each iteration re-applies the whole transformer. Huginn loops only a 4-layer _core_ , sandwiched between an untied 2-layer prelude and 2-layer coda that run exactly once — so part of Huginn’s depth is a fixed feedforward frame around the recurrence, and its recurrent unit is far smaller than its virtual depth suggests.

- **Iteration count.** Fixed at training time (Ouro: 4 loops, always) or variable at test time (Huginn: sampled during training, chosen at inference; we use _r_ =16).

- **Supervision of intermediate states.** Ouro attaches the LM head to every loop end and trains all of them ( _deep supervision_ ); Huginn trains only the post-coda output, with truncated backpropagation through the recurrence (mean unroll 8).

Table 1 summarises the three models. Huginn at _r_ =16 is a close depth-and-compute match to Qwen3.6-27B (68 vs. 64 virtual layers), which makes the untied/tied comparison unusually direct.

### 2.3 Related work

**Global workspace theory and AI systems.** Global workspace theory [Baars, 1988, Dehaene and Changeux, 2011, Mashour et al., 2020] motivates the functional suite we inherit; Butlin et al. [2023] propose indicator properties derived from such theories for assessing AI systems, with recurrent processing as a recurring ingredient. We make no claim about consciousness: we use the workspace suite only as a functional characterisation and ask how its outcomes depend on architectural recurrence.

Table 2: Lens-fitting summary. Every lens uses the same 1000-prompt corpus and estimator; only the (source, target) geometry differs. The twelve Huginn gap-fill lenses put a target at every remaining recurrence end, completing the depth tiling used for readout in §6.1; the causal suite (§7) uses the four _r_ =4 _/_ 8 _/_ 12 _/_ 16 lenses only.

||Sources|Targets|Wall-clock|
|---|---|---|---|
|Qwen3.6-27B|all 63 layers; L0–23 (band entry)|final; L24|_∼_3 h; _∼_2 h|
|Ouro-2.6B|all 191 virtual layers|virt. final + loop ends (v47/95/143)|_∼_1 h + 1.6 h|
|Huginn-0125|all 67 virtual layers|coda end + _r_=4_/_8_/_12; gap-fill _r_=1–15|_∼_7.3 h + per-target|

**Readout probes.** Logit lens [nostalgebraist, 2020], tuned lens [Belrose et al., 2023], and the Jacobian lens [Gurnee et al., 2026] differ in how their affine readout maps are obtained: fixed, trained, or derived from the model, respectively. Probing of weight-tied models specifically is nascent: Lu et al. [2025] decode Huginn’s recurrent states with the logit lens and a coda lens, finding that the two disagree systematically across core blocks and little evidence of an interpretable latent chain of thought, and Blayney et al. [2026] characterise looped models’ iteration dynamics through hidden-state geometry, attention patterns, and linear probes. Neither fits transport maps between virtual layers, which is what the workspace suite requires.

**Iterated computation.** Looped transformers have been analysed as programmable computers [Giannou et al., 2023] and as learners of iterative algorithms [Yang et al., 2024]; Ouro and Huginn demonstrate the recipe at language-model scale [Zhu et al., 2025, Geiping et al., 2025]. Interpretability work on these models has so far decoded states through fixed unembedding-based probes — the logit lens in the Huginn release itself [Geiping et al., 2025], logit and coda lenses across its recurrent stack [Lu et al., 2025] — or analysed iteration dynamics mechanistically [Blayney et al., 2026]; to our knowledge no prior work fits transport-based lenses across loop boundaries or runs a causal workspace suite on a looped model.

## 3 Method: virtual unrolling

### 3.1 The lens on an unrolled graph

The lens definition (Eq. 1) extends to weight-tied models by generalising “layer” to _virtual layer_ : the _k_ -th firing of a physical block is its own node in the unrolled computation graph. For Ouro, _v_ = 48 _·_ step + layer; for Huginn, prelude and coda blocks fire once and the _i_ -th core block’s _k_ -th firing is _v_ = 2 + 4 _k_ + _i_ . Our adapter presents the unrolled graph to the original fitting machinery by replacing the model’s layer list with virtual-block proxies whose forward hooks dispatch only on their assigned firing — without this, a standard activation recorder silently keeps only the _last_ iteration’s state. The remaining gradients are computed automatically through the between-loop normalisation and the tied weights, so _Jv_ is exact for the unrolled graph, not an approximation to it. Fitting through a recurrence also exposes failure modes that do not exist on feedforward stacks and that corrupt results without raising any error; we document three such silent-failure traps, and the validation suite that guards against them, in Appendix G (library-compatibility patches for loading the two looped models in Appendix H).

### 3.2 Fit protocol

All lenses share one protocol — the original study’s recipe — so cross-model comparisons never mix estimators: 1000 WikiText-103 prompts of 128 tokens (identical prompt set for all three models), first 16 positions skipped, parameters frozen, per-source mean Jacobians accumulated by the unmodified fitting code. Table 2 summarises. Convergence was monitored as the relative shift of the running mean (final shift _≤_ 0 _._ 016 on all Huginn lenses; _≈_ 5 _×_ 10^(_−_3) on the Qwen band-entry lens). All fits and experiments ran on one to three NVIDIA B200 GPUs in bf16.

### 3.3 Why a lens family , not a lens

Our initial fitting scheme treated the unrolled model as one long, weight-tied transformer and used its final virtual state as the lens target, as on a standard transformer. This scheme fails as a readout instrument on both iterated models. Ouro’s final-target lens reads punctuation throughout loops 1–3 — not because those loops are empty, but because the transport to the final target dies crossing loop boundaries (§4); Huginn’s transport horizon is roughly two recurrences regardless of target.

We therefore fit a family of lenses whose targets are the outputs of individual loops or recurrences: every loop end on Ouro (v47/95/143/191) and at the recurrence ends _r_ =4 _/_ 8 _/_ 12 _/_ 16 on Huginn (Table 2). Note that we change only the lens target, not the fitting procedure: an input–output Jacobian is well defined for any upstream source and downstream target in the unrolled computation graph. The causal experiments in §7 validate the resulting lens family empirically: interventions along its token directions change what the models verbalise and how they answer in the predicted, concept-specific ways, providing the same form of causal validation used by Gurnee et al. [2026]. On Ouro the choice of target is principled, not merely convenient: loop ends are exactly the states deep supervision trains the LM head to read, so unembed( _hJ_^(_⊤_) ) is a proper readout at every target, and the per-loop lenses reveal that the semantic content the final-target lens missed was present from loop 2 all along. On Huginn the targets tile the depth so that every source lies within some lens’s transport horizon: those four lenses carry the causal suite (§7), and for readout we complete the family to a target at _every_ recurrence end, putting every source within one recurrence of its own target (§6.1). Throughout the paper, every readout is therefore taken through the lens _whose target the source can actually reach_ , and every cross-loop or cross-model comparison is _distance-controlled_ : readouts are compared at matched offsets from their own targets, where transport attenuation is matched by construction. Matching source-to-target distance prevents differences in transport attenuation from being mistaken for architectural effects.

## 4 Transport structure: what the lens artefacts reveal

We first study the fitted transport maps themselves — how far they carry information, what symmetries they inherit from weight tying, and how they compose. Throughout, we distinguish two things a transport can carry. The _bulk_ of the state is the residual vector as a whole — the thousands-dimensional aggregate that matrix-level statistics (transport norm, KL, top-1 agreement) measure. The _semantic workspace component_ is the low-dimensional slice of that vector that encodes nameable concepts — the part the workspace phenomena actually live in, measured by whether a specific concept remains recoverable after transport. The distinction matters because the two behave completely differently: bulk transport collapses with distance in the same way across all three architectures, while the workspace component and compositional structure are where recurrence makes a difference.

### 4.1 Bulk collapse with distance is universal

We use bulk statistics to control for source-to-target distance. Transport weakens with distance in all three architectures, so cross-architecture comparisons must be made at matched distances. A mean Jacobian is only informative where individual prompts’ Jacobians agree. Near its target they do (per-prompt _∥J∥/√d ≈_ 1 _._ 3–1 _._ 5 on Ouro, mean 1 _._ 24 one block out); across tens of layers the response direction becomes prompt-dependent and the average cancels ( _∥J∥/√d_ = 0 _._ 13–0 _._ 3 beyond one Ouro loop). This decline is similar across architectures: at distance 39 — the standard model’s own working band-to-target span — Qwen3.6-27B’s aggregate transport is just as dead as Ouro’s at matched distance (top-1 agreement with the final output: 2%, _∥J∥/√d_ = 0 _._ 46, inside Ouro’s 0.14–0.53 range across its target series). This collapse does not mean that the lens is unusable: the same distance-39 transport that fails every aggregate test still carries a specific latent concept to its target (Table 3, first row). At long distances, useful readout depends not on the bulk response but on a small set of semantic directions that remain consistent across prompts. Unembedding a direction-washed, near-zero transport of _h_ yields an unspecific vector _regardless of the content h carries_ ; the few directions on which prompts’ Jacobians agree survive the averaging that cancels everything else.

Table 3: Semantic transport at matched bulk quality, tracked with the riddle prompt “Fact: The currency used in the country shaped like a boot is. . . ”. The latent concept _Italy_ never appears in the prompt, and both endpoints of every row are independently verified to represent it. “Carries _Italy_ ?” asks whether the concept survives the transport itself: after mapping the source state through the fitted transport, is _Italy_ still recoverable at the target (top-ranked in the transported readout, with a concept-vs-control projection above chance)? The cross-boundary transport (last row) has _better_ bulk statistics than the standard model’s working case (first row), yet the concept does not arrive.

|Transport|Distance|State cosine|Carries _Italy_?|
|---|---|---|---|
|Qwen3.6-27B band _→_target (within stack)|39|0.19|yes|
|Ouro-2.6B band _→_own loop end (within loop)|11–23|0.3–0.5|yes|
|Ouro-2.6B loop-2 band _→_loop-3 band (across boundary)|32–60|0.31–0.45|no|

**Methodological consequence.** Aggregate faithfulness metrics ( _∥J∥_ , KL, top-1 agreement against the final output) are not usability tests for a lens at range, and we never use them as such. We regard a readout as valid when it moves a probe token’s rank against a _∼ V/_ 2 chance baseline and its direction supports a causal intervention; bulk numbers appear only in matched-distance comparisons.

### 4.2 The semantic component does not cross Ouro’s loop boundary

Once bulk attenuation is controlled, the architectural effect can be isolated clearly. We track a prompt-absent latent concept — _Italy_ , from the riddle “Fact: The currency used in the country shaped like a boot is. . . ” — through transports whose endpoints are both independently proven to contain it (Table 3).

Across the boundary the _Italy_ projection drops to _∼_ 0.1 of the target state’s value with no concept-vscontrol discrimination, while bulk magnitude survives at 0.27–0.46: the workspace component is destroyed roughly 3 _×_ faster than the state as a whole. The mechanism is what happens at Ouro’s loop ends. Deep supervision trains every loop-end state to be directly decodable by the LM head — we call these states _checkpoints_ , in the sense of supervised way-points _within_ a single forward pass (not saved model weights). Because the training loss reads each checkpoint directly (logit-lens KL at loop ends: 1.7/0.5/0.1 nats for loops 1–3, top-1 agreement up to 92%; one block _short_ of a checkpoint, at virtual layer 190, decodability falls to 0.8 nats and 71% — states are readable exactly where training made them readable), each loop _decodes_ its conclusions into a token-aligned checkpoint state, and the next loop _re-encodes_ them nonlinearly into its own band coordinates. Successive loops therefore exchange information through token-aligned checkpoint states rather than preserving the same workspace vectors across the boundary.

Is content destroyed because the model iterates at all, or because of this decode/re-encode step? Huginn separates the two possibilities. It also iterates a weight-tied block, but its recurrence boundary has no supervised decode step: the raw latent state flows unchanged into the next iteration. If iteration itself destroyed content, Huginn’s should degrade the same way. Instead, workspace content computed by recurrence 2–3 remains readable through all 16 recurrences (§5). The destruction is therefore a property of the checkpoint interface, not of looping.

### 4.3 Weight tying induces symmetries in the lens artefacts

Weight tying admits two interpretations of these models: a looped transformer is “a deeper transformer that happens to reuse weights”, or it is “one programme run repeatedly”. The two interpretations make different predictions about the fitted transports. If only depth position matters, per-iteration structure should be invisible in them; if the model re-runs one programme, the transports should repeat when viewed from each iteration’s own frame. Settling this also has a practical implication: if the transports repeat, a lens fitted at one iteration transfers to the others, and the loop-relative, distance-controlled comparisons this paper relies on (§3.3) are legitimate rather than convenient. The results support each interpretation in a different coordinate frame.

Figure 1: The lens artefacts carry the architecture’s signature. **(a)** Factorisation of the mean transport through a mid-depth cut, cos( _Jv→t, Jw→tJv→w_ ) vs. source–cut distance. Intuitively, a high cosine means the long journey decomposes into two independent legs — the state at the cut summarises everything the rest of the computation needs. Deep-supervised Ouro decomposes almost perfectly at any range (its loops are Markovian: each hands the next a self-contained summary); the standard transformer decomposes only approximately, degrading with span; Huginn barely decomposes at all — its recurrences stay entangled across any cut (§4.4). **(b)** Similarity of independently fitted transports compared at matched offsets from their own targets. A weight-tied core does the same thing at every iteration, and the fitted lenses inherit that repetition: exactly on Huginn (cosine 0.9991 — the _r_ =8 and _r_ =12 lenses are interchangeable), approximately on Ouro (0.81–0.99, rising towards the target). Dotted lines show what dissimilar transports look like: random pairs, and Huginn’s own lenses misaligned by one recurrence (§4.3).

**Absolute frame: no trace of tying.** Cosine similarity between flattened _Jv_ pairs on Ouro decays smoothly with virtual distance (0.989 adjacent, 0.804 at ∆=48, 0.087 at ∆=190) with no block structure at multiples of 48; same-physical-layer pairs are no more similar than the all-pairs baseline (0.676 vs. 0.711). Towards the final output, the looped model is indistinguishable from a generic 192-layer network: what a layer application does to the eventual output is set by where it sits in the unrolled computation, not by which weights it uses.

**Loop-relative frame: strong to exact symmetry.** Compared at matched offsets from _their own_ targets — which matches attenuation by construction — the per-iteration transports repeat. On Ouro, matched-offset matrices across loops have mean cosine 0.885 (0.932 inside the workspace band), against 0.716 offset-mismatched and 0.555 for random pairs; their top-32 singular subspaces overlap at 0.86/0.83 (random baseline 0.016), and pairwise similarity rises with loop convergence (loops 3–4: 0.988). On Huginn the symmetry is essentially _exact_ : independently fitted lenses at matched offsets agree with matrix cosine **0.9991** ( _r_ =8 vs. _r_ =12 targets; 0.43 when offset by one recurrence; Figure 1b). The weight-tied core implements literally the same mean transport at every recurrence.

These results confirm both interpretations, each in its own frame. Towards the final output, tying is invisible (same-physical-layer similarity 0.676 vs. all-pairs 0.711) — as a contributor to the eventual answer, a looped transformer really is “a deeper transformer that happens to reuse weights”, and what matters is when a layer runs, not which weights it runs. Within each iteration’s own frame, the transports repeat far above every baseline (0.885–0.9991 vs. 0.43–0.716) — the model really does “run one programme repeatedly”. The practical consequence follows from the second reading: a lens fitted at one iteration transfers to the others (on Huginn, exactly), which legitimises the loop-relative comparisons used throughout and means full-depth lens coverage of a recurrent model costs one iteration’s fit, not _r_ of them.

Table 4: Mean-transport factorisation through a mid-depth cut: cos _Jv→t, Jw→tJv→w_ .

||Cut|cos(direct, composed)|
|---|---|---|
|Ouro-2.6B|supervised loop ends (v95, v143)|0.94–1.00|
|Qwen3.6-27B|band entry, L24|0.69–0.95, rising towards the cut|
|Huginn-0125|recurrence ends (v33, v49)|0.25–0.84, rising towards the cut|

### 4.4 Factorisation orders the architectures

Per prompt, the chain rule guarantees _Jv→t_ = _Jw→t Jv→w_ ; for the _means_ the factorisation E[ _AB_ ] _≈_ E[ _A_ ] E[ _B_ ] is not guaranteed, and whether it holds measures how much prompt-dependent correlation crosses the cut. The three architectures order cleanly (Table 4, Figure 1a).

Ouro’s near-perfect factorisation is evidence of deep supervision: the checkpoint truncates cross-cut correlations, making the loop-to-loop process approximately Markovian — which is the same mechanism that destroys persisting content (§4.2). Huginn shows the complementary pattern: its continuous latent flow preserves prompt-dependent correlations across boundaries (so content persists), and the product of means, which keeps only the uncorrelated part, under-predicts the true transport. The standard transformer sits between: approximate factorisation with a correlation term that grows with span.

**Practical corollary.** On checkpointed models, per-loop lenses can be fitted independently (memory scales with one loop’s span) and composed to recover any long-range transport — the composition matches the end-to-end fit. On continuous-recurrence models it cannot; end-to-end fitting is required.

### 4.5 The input-injection path

Huginn re-injects the prelude’s output at every recurrence, and the lens sees it: transport norm from the prelude is distance- _independent_ towards every target (0.26–0.37 from prelude layer 0, 0.54–0.77 from layer 1) while mid-recurrence sources collapse below 0.06 within 4–5 recurrences. The recurrent state is better read as a working memory continuously re-grounded in a fixed input encoding than as a residual stream — likely also why its state-to-state mean transport decorrelates so fast. Ouro shows a weak analogue (its loop-1-target lens rises again where its source is the embedding layer). This path matters causally: it is the interface that makes mid-stream interventions decay unless refreshed (§7).

These results do not establish whether looped transformers should re-inject the input encoding. Huginn and Ouro differ simultaneously in injection, supervision, and training recipe, so nothing here attributes Huginn’s content persistence to the injection path specifically — §4.2 attributes the _destruction_ on Ouro to its checkpoint interface, which is a different mechanism. In the one model that has the path, it is the only long-range route the mean transport preserves, and it coexists with — but is not shown to cause — stable content maintenance. Two inference-time experiments, both hooks on the adapter’s linear(cat[state _,_ embeds]) input, settle the maintenance question without retraining. The two hypotheses predict opposite outcomes: if the recurrent state were a memory buffer, content computed by recurrence 7 would already sit in the state — which these hooks leave untouched — and a late cut to the injection should not matter. _Attenuation_ : scaling the injected embedding by _α_ from recurrence _k_ onwards destroys the output at _α ≤_ 0 _._ 25 for _every_ onset — 0/8 prompts recover the baseline next token even when only the last four of sixteen recurrences are attenuated — while _α_ = 0 _._ 75 is nearly harmless (7–8/8). The lens shows why: at _α_ = 0, _k_ = 8, the latent answer’s rank holds at its baseline level through recurrence 7 (median 11) and collapses within one recurrence of the cut (2,749 at recurrence 8, then _∼_ 40,000 — chance). Maintenance depends on re-grounding: content does not survive even one recurrence beyond its last refresh. _Injection swap_ : replacing the injected embedding with a matched-length prompt’s (three pairs differing in one word, both directions) from recurrence _k_ onwards flips the final output to the _new_ input’s own answer in all 30 configurations, including _k_ = 12; the old answer’s rank decays within _∼_ 2 recurrences of the swap and the new one reaches near rank 1 within _∼_ 4. The state is therefore best described as continuous recomputation from the injected encoding with a _∼_ 2-recurrence memory, not a memory buffer — the same constant that bounds the mean transport (§4.1), now measured causally. Natural content “persists” across all sixteen recurrences (§5.2) because the input that regenerates it is re-injected at every one. A definitive design answer still needs controlled training runs (matched small looped models with and without injection), which is beyond this paper’s scope.

## 5 What the iterations compute

Section 4 characterised how information moves; this section uses the lens family to ask what the iterations actually do with it. The main result is that both models perform iterative refinement with per-task depth, but they maintain their intermediate results in different ways. A concept crossing an Ouro loop boundary is destroyed and re-derived by the next loop, while on Huginn a concept computed by recurrence 2–3 stays readable through all sixteen recurrences, sustained by the re-injected input encoding rather than by the recurrent state (§4.5).

### 5.1 Ouro: iterative refinement, mostly done by loop 3

Because Ouro’s checkpoints are trained to be decodable (§4.2), the plain logit lens is close to a calibrated instrument _at loop ends_ (0.1–1.7 nats from the final distribution, 60–92% top-1 agreement), and cross-loop timing claims can rest on it. Reading the loop ends over WikiText: loop-3’s output already matches the final output for 92% of tokens; loop 4 changes the top-1 for only _∼_ 8%. The model’s own early-exit gate — a trained component that uses no lens at all — corroborates the same timeline, assigning exit probability [0 _._ 03 _,_ 0 _._ 14 _,_ 0 _._ 34 _,_ 0 _._ 49] across the four loops.

The per-loop lens family shows which representations change. Tracking latent concepts through each loop’s own band (a distance-controlled comparison, §3.3):

- **Easy fact** (“The capital of the country where the Eiffel Tower stands is the city of. . . ”): _Paris_ is top-1 at the end of loop 1 and every later loop end — the remaining loops maintain, not compute.

- **Riddle** (“Fact: The currency used in the country shaped like a boot is. . . ”): loop-1’s band reads only the literal surface ( _boots, footwear, shoe_ ); **loop-2’s band already reads** **_Italy_** ; loops 3–4 maintain it. The readout identifies a correct token that is absent from the prompt, indicating that the transport retains semantic information.

- **Arithmetic** (47+38): the answer settles in the mid-band readout only in loop 4, though the checkpoint state shows _eighty_ from loop 2, suggesting that a partial answer forms early but does not settle fully until later.

Task difficulty maps onto the number of loops before the band contains the answer. Three measurements with different failure modes support this account of convergence:

1. **Per-loop J-lens readouts.** Measurement: the rank of a prompt-absent probe token (e.g. _Italy_ ) in unembed( _hvJv_^(_⊤_) _→_ loop end^(),ateachloop’sbandlayers.Dependsonourfittedtransport.)

2. **Logit lens at checkpoints.** Measurement: KL and top-1 agreement between unembed( _h_ loop end) and the model’s final output distribution, over WikiText. This is the LM head applied directly to the loop-end state — exactly the operation the deep-supervision loss trained — and involves no transport.

3. **The early-exit gate.** Measurement: the model’s own trained stopping signal — the learned gate that decides at each loop end whether to halt — logged per token, with no readout of any kind on our part. Its mean exit distribution over WikiText is [0 _._ 03 _,_ 0 _._ 14 _,_ 0 _._ 34 _,_ 0 _._ 49] across loops 1–4: the model itself would stop at loop 3 for a third of tokens.

The first depends on the lens estimator, the second only on direct unembedding, and the third on neither — it is an internal component of the model, trained end-to-end, that we merely observe. All three agree that convergence is mostly complete by loop 3 with per-task depth, so the picture cannot be an artefact of any single estimator.

Figure 2: Band-map heat strip on Huginn: best rank among the expected latent candidates (log colour scale; brighter is better) at every source layer, bucketed by recurrence step, for the four original lenses (rows; red marker = each lens’s own target) on three carrier prompts (panels). All rows read the _same r_ =16 forward pass; grey cells lie outside a lens’s fitted source range. The readable region slides with the lens target (§5.2): each lens resolves content only in the one-to-two recurrences before its target, and the model answers the riddles with graded difficulty ( _sushi_ readable from recurrence 2 everywhere, _boot_ only weakly and late).

### 5.2 Huginn: a sliding window, and content that persists

Huginn has no absolute band. Each lens in the family reads reliably only in the one-to-two recurrences before its own target: for the prompt “Fact: The capital city of the country famous for sushi is”, the latent answer _Tokyo_ is rank 1 at recurrence steps 2–3 through the _r_ =4 lens, step 7 through _r_ =8, steps 10–11 through _r_ =12, and steps 14–15 through the primary lens. The readable region _slides with the target_ — transport is distance-limited (§4.1), and workspace geometry follows the lens, not the layer index.

Because every window reads the _same_ content, the latent answer — computed by recurrence 2–3 — persists in the state through all 16 recurrences. This is the direct opposite of Ouro’s maintenance strategy. Ouro’s checkpoints destroy the workspace component at each boundary and the next loop re-derives it (§4.2); Huginn’s raw-latent handoff carries it forward, regenerated from the re-injected input at every recurrence (§4.5). The two architectures show the same relationship between task difficulty and the iteration at which the answer becomes readable: the answer to “. . . the country famous for sushi. . . ” ( _Tokyo_ ) is readable from recurrence 2, “The currency used in the country home to the Eiffel Tower. . . ” ( _French_ ) by recurrence 3, and the boot riddle of §5.1 arrives late and equivocally — it is genuinely hard for a 3.5B model. Thus, in both architectures, harder tasks settle later in the iterative computation, while “destroy-and-re-derive vs. carry-forward” separates the two training regimes.

Figure 2 shows both results at once as a heat strip over the same run: each lens’s readable region (bright cells) hugs the one-to-two recurrences before its own target, and the four regions tile the depth axis while reading the same latent content.

### 5.3 Uniform, weak decodability at every depth — without deep supervision

Directly unembedding Huginn’s recurrent state (no transport at all) is informative at every depth — uniformly, and weakly. Over the 8-prompt suite, the logit lens’s top-1 token equals the model’s actually emitted next token at 15–25% of positions, flatly across the whole recurrent stack (virtual layers 14–65; final coda state: 63%). The right comparison is the other models mid-computation: Qwen never exceeds 9% anywhere in L8–40, and Ouro sits at 1–7% mid-loop, spiking to 64–71% only at its supervised checkpoints. Huginn is thus the only model whose states decode above the mid-computation floor at _every_ depth, although with only a quarter of the confidence at Ouro’s checkpoints. The plausible mechanism is _implicit_ deep supervision: training samples the recurrence count from a broad distribution (mean 32), so across the corpus the single LM head is applied to states of every depth, exerting the same decodability pressure that Ouro’s explicit per-step loss applies. This result shows that logit-lens decodability is _not_ evidence of explicit deep supervision: head pressure at depth in any form produces it. It also predicts that Huginn should verbalise workspace content well despite lacking Ouro’s supervision, which §7 confirms.

## 6 Workspace geometry across architectures

The readouts of §5 rest on target-token _ranks_ for hand-picked probes. This section re-measures the workspace geometry with an independent family of metrics, computed from recorded forwards of the original study’s demonstration protocol — 8 chat prompts, 128-token greedy completions, and both lenses read at every (layer, position) (Appendix D). Occupancy is measured with the study’s own concept-inventory estimator — a gradient-pursuit [Blumensath and Davies, 2008, Nanda et al., 2024] reconstruction of the state, described in the original but with no released implementation; ours is specified in Appendix E. We use these metrics to test whether the same structure appears under measurement procedures with different failure modes.

### 6.1 Occupancy recovers the band, loop, and window structure

At each (layer, position) cell, pursuit reconstructs the state as a sparse, non-negative combination of unit-normalised J-lens atoms _aτ_ = _Jv_^(_⊤uτ_.Atoken)^_occupies_thecellwhenitsatomisamongthefirst25selected with a coefficient above 0 _._ 05 _∥h∥_ ; its occupancy at a layer is the number of such cells. This measure requires no ground-truth answer. Appendix E specifies the algorithm, reports robustness across the full ( _k, θ_ ) grid, and cross-checks the results with an independent estimator: the demonstration’s top-8 softmax readout. The two estimators select substantially different tokens. Figure 3 plots occupancy across virtual depth for the animal prompt (“I’m thinking of the largest land animal. What do they eat?”). This is the only demonstration prompt that all three models answer directly and correctly (verbatim prompt and replies appear in Appendix D). We divide the tracked tokens into _latent_ tokens, which occur in the readout but not the transcript, and _spoken_ tokens, which occur in the transcript; singular and plural variants are merged. The latent class contains related concepts such as _food_ (the category asked about), _tree_ (unsaid by all three models), _mammal_ , _vegetation_ , _fruit_ , and _root_ .

This metric recovers the band, loop, and window structure described in §§4–5. On Qwen, the latent vocabulary is absent below L20 (0 slots; an absence carried by the coefficient sizes rather than by which atoms are selected, Appendix E), fills the L24–48 band beside the spoken answer (42 latent vs 326 spoken), and persists into the top of the stack (37 vs 112 at L55+). On Ouro, in-band (physical L24–40) occupancy exceeds out-of-band occupancy as a _per-layer rate_ in all four loops and for both token classes (loop 4: 22.3 vs. 16.6 slots per layer; the band contains 17 of each loop’s 47 lensed layers). Per-layer rates are necessary because the out-of-band region contains nearly twice as many layers, causing the comparison to fail on raw totals in the late loops. The relationship also weakens at _k_ = 50, where the still-rising variance curve (Appendix E) indicates that the additional selected atoms are no longer matched by additional J-space content. On Huginn, read through the completed tiling described below (every source within one recurrence of its covering lens target, so lens distance is held constant across depth), total occupancy rises through the prelude and first few recurrences and then plateaus over the middle of the stack: the eleven consecutive one-recurrence windows spanning v17–v60 vary only between 31 and 44 slots, against 5 in the first window. It then jumps in the coda window (92 slots, 2 _._ 4 _×_ the plateau mean). After controlling for lens distance, the same concepts remain readable at a similar rate across the recurrent stack rather than accumulating with depth. The only marked increase occurs in the coda state consumed by the LM head.

The other demonstration prompts provide more specific examples. On the currency riddle (“what currency was used in the boot-shaped country before 2002?”), Ouro correctly answers “the Italian lira”. The unspoken alternative _euro_ , a tempting surface association between 2002 and currency, peaks in loop 2 and then declines from 259 to 188. Meanwhile, _Italy_ , the intermediate concept used to derive the correct answer, increases monotonically from 177 to 390. Across all twelve grid settings, the representation therefore shifts from the rejected alternative towards the intermediate concept used by the correct answer. The same analysis identifies why Huginn answers the riddle incorrectly (“was the Euro”): the country never enters J-space ( _Italy / Italian / lira / lire_ : 5 slots across all 67 virtual layers, and never more than 9 anywhere on the grid), so the salient surface association (2002 + currency _→_ euro) dominates. This is the occupancy counterpart of the two-hop weakness measured behaviourally by the probe swap (5/90, §7). Qwen’s inventory also includes concepts across languages: its occupied atoms include the Chinese token for _Italy_ (102 slots), alongside the English _Italy_ (344) and _currency_ (225). English-string token classes cannot capture this concept mass. On the sport prompt, Qwen’s latent sport concepts occupy 123 band slots but none in the final eight layers, while spoken tokens retain 116 slots. The late layers therefore retain the emitted words while dropping the unspoken topic alternatives. As with the absence below L20, the zero is due to coefficient sizes (Appendix E).

Completing Huginn’s lens family to a full tiling — a target at _every_ recurrence end — makes occupancy nonzero at every virtual layer, directly confirming the persistence result of §5.2: the same concepts are readable wherever a lens’s transport reaches. However, the original partial tiling makes content appear to accumulate through the recurrences. The rising curve is an artefact of unequal source-to-target lens distance rather than a property of the model (Appendix D).

### 6.2 Decodability, quantified from the probability field

The dumps also record, for every (layer, position), the probability the lens’s decoded distribution assigns to the token the model actually emits next — softmax(unembed( _h_ )) for the logit lens, evaluated at the true next token and averaged over positions and prompts (top-8-truncated, hence a lower bound). Chance on a _∼_ 150k vocabulary is _∼_ 10^(_−_5) , which sets the scale. This field calibrates the _confidence_ of the decodability established by rank in §5: Ouro’s logit lens reaches 0.61–0.65 at _every_ loop end — the checkpoint state, decoded directly, already gives the model’s actual next token two-thirds probability, i.e. nearly the final layer’s confidence, which is deep supervision made visible. Huginn sits on a flat plateau of 0.02–0.045 across the recurrent stack from recurrence 3 onward (virtual layers 14–65, with a stable within-recurrence profile that peaks at each recurrence’s last core layer), jumping to 0.47 only at the final coda state v66 — the state the LM head consumes. The plateau is three orders of magnitude above chance and matches the rank picture of §5.3 (top-1 hit rate 15–25% uniformly across the same depths), yet sits 20 _×_ below Ouro’s checkpoints — implicit supervision buys readability at every depth, not checkpoint-grade confidence. Qwen’s mid-band is near zero through the logit lens (0.003 at L36) and an order of magnitude better through the J-lens (0.019) — on a standard transformer, mid-stack states are not next-token-decodable without transport. The three-model ordering — Ouro checkpoints _≫_ Huginn everywhere-weakly _>_ Qwen mid-band _≈_ 0 — also appears in the demonstration’s own probability field, without a separate set of target-token probes.

## 7 The causal suite: eleven intervention families

The preceding analyses are observational: they read states that the model computed on its own. We next test the inferred directions using causal interventions. The original study released eleven behavioural experiment families (writes, ablations, and readout probes over workspace directions); we reimplemented them in a single shared harness and ran the full suite identically on all three models. Two protocol choices matter. First, all interventions write or ablate along lens directions _d_ = normalise( _Jv_^(_⊤ut_)—the artefact under study is the) intervention operator. Second, each model uses its own validated interface: Qwen intervenes over the band L24–48 through the final-target lens; Ouro clamps the bands of loops 2–4 with each loop’s own lens; Huginn clamps the last three lens windows. Table 5 condenses the results (full tables and token-position conventions in Appendix B).

### 7.1 Writes must span every remaining iteration

The flexible-generalisation experiment swaps a concept mid-reasoning: on a prompt whose answer routes through an intermediate concept, the operator clamps the workspace state’s projection onto the source concept’s lens direction over to the target concept’s, and scores whether the model’s answer flips to the one implied by the injected concept. On Qwen the swap succeeds — the answer flips to the injected concept’s implication — in 42% of the 192 trials, matching the original study’s 40%. On Ouro, _where_ the clamp is applied decides everything: clamping the bands of all remaining loops (2–4) reaches 38% — statistically indistinguishable from Qwen — but any single loop alone falls well short: 21% for loop 2, 29% for loop 3, 25% for loop 4. The loop-2-only result is the causal counterpart of §5.1: later loops do not passively copy the patched state forward — they _re-derive_ content from the raw context and partially restore the original answer. A write into an iterated model must therefore be maintained across every remaining loop to counteract this reconstruction. Huginn exhibits a complementary constraint: clamping the last window alone gives results identical to clamping all three windows, while the second-to-last window alone gives 3% — the _∼_ 2-recurrence transport horizon of §5.2 bounds _writes_ exactly as it bounds reads, because a lens direction is only a valid coordinate system near its own target. Although the original content persists across recurrences (§5.2), an intervention expressed in one lens’s coordinates does not.

Figure 3: Count-by-layer occupancy of the J-lens concept inventory (pursuit at _k_ = 25, coefficient threshold 0 _._ 05 _∥h∥_ ) on the animal prompt, the same for all three models (verbatim prompt and replies in Appendix D; the top-8 counterpart is Figure 4). Bars are _stacked_ , not overlaid: blue is the latent class (read but never written; singular/plural variants merged), red the spoken class (in the transcript) drawn on top, so total bar height is the two classes’ combined count. Dotted lines mark loop boundaries (Ouro, every 48 physical layers; the rise into L41–46 before each boundary is the decode into that loop’s supervised checkpoint, §4) and recurrence boundaries (Huginn, every 4 virtual layers between the 2-layer prelude and coda). The periodic patterns have different sources: Ouro’s four bands are a _model_ property (content forms and dies within each loop, with real boundaries between loops), while Huginn’s sawtooth is an _instrument_ property — each lens window’s count rises towards its own lens target at the mean-J transport horizon, tiling content that persists continuously across all sixteen recurrences (§5.2).

Table 5: The eleven-family causal suite. Each family is scored by the original study’s own metric (second column gives its reported range where one exists). Bold marks the best model per family. The pattern the section unpacks: the three models succeed on _different_ families, and the split tracks training interface, not scale.

|Family|Study|Qwen3.6-27B 27B|Ouro-2.6B 2.6B|Huginn-0125 3.5B|
|---|---|---|---|---|
|Verbal report (swap _→_top-5)|59–88%|69%|80%|98%|
|Introspection (injected concept)|reportable|null|top-1 31%|null (0/97)|
|Modulation (focus/ctrl/suppress)|ordered|.38/.07/.07|.77/.39/.26|.28/.15/.19|
|Top-down summoning|evokes concept|_∼_0|_∼_0|_∼_0|
|Two-hop probe swap|54–70%|52% (base 63%)|29% (base 48%)|6% (base 28%)|
|Flexible generalisation (swap)|40%|42%|38%|13% (base 44%)|
|Selectivity: language|expl. _≫_auto|+1.00|+0.25|+0.38|
|Selectivity: line count|iff relevant|0.09@1|1.00@1|0.09@1|
|Ignition (10–90 width)|sharp|0.10|0.55|none in-window|
|Capacity (words held)|_∼_25|0.7|2.4|1.8|
|Dual task (concepts survive)|survive|6/21|17/21|7–8/21|

The loop-4-only drop (25%) cannot be re-derivation — no loop runs after it. Per-trial logit-lens rank trajectories of both answers, logged at every loop end, show the two extreme conditions fail for _opposite_ reasons. Unclamped, the original answer is settled early: median rank 1 by the end of loop 2 on every category (1–3 at loop 1). Under loop-2-only clamping, failed trials show the patch _taking_ and then being undone — the injected answer reaches median rank 7 at the loop-2 checkpoint but decays to 44 by loop 4 while the original recovers to rank 1: overwritten by re-derivation. Under loop-4-only clamping, failures look entirely different: the original answer already sits at rank 1 at loop-4 _entry_ and the clamp lifts the injected answer only to median rank 11 — a displacement failure against consolidated content, with no re-derivation involved. Loop 3 combines the two mechanisms: it has one fewer re-derivation pass than loop 2 and less consolidated opposition than loop 4. Its result falls between the two extremes at 29% (CI 23–36), exactly as the two-mechanism account requires. On all-loops-clamped _successes_ the original answer is driven from rank 1 to median rank 294 by loop 4 — the sustained clamp is precisely what prevents the recovery seen in the single-loop conditions. The category split matches: countries (same-category direction cosine 0.36, well-separated) succeed _∼_ 30/48 in every condition, while months (cosine 0.67) and animals (0.39) fail in opposite directions across the two single-loop conditions (months 0 early vs. 12 late; animals 9 vs. 6, but 24 with all loops).

### 7.2 Ablation follows the same interfaces as writes

We test two widths of ablation. The first removes a single concept’s lens direction ( _h ← h −_ ( _h · d_ ) _d_ at the intervention layers); the second removes the ten-dimensional span of the inventory atoms that best reconstruct the state (§6.1). For both, we measure whether an answer that was correct before the intervention remains correct.

The single-direction results show that the effect is specific to the concept. Ablating the prompt’s argument concept degrades all three models comparably (answer kept: Qwen 72%, Ouro 71%, Huginn 61%), while a control that repeats the operation with an unrelated word’s direction never changes the answer (100% kept for all three models). The behavioural effect therefore depends on which direction is removed rather than on removing an arbitrary direction, confirming that the lens direction carries the concept. The interface pattern observed for writes in §7.1 also appears under ablation: Ouro’s loop-2-only ablation is a _complete_ null (100% kept — the later loops re-derive the ablated content through the checkpoint, mirroring the 21%-vs-38% write pattern), and on Huginn ablating the last window alone is _exactly_ equivalent to ablating every window (17/28 kept in both conditions). Removing the _answer_ token’s direction reveals a further ordering across models: it leaves Qwen at 53% kept, Ouro at 19%, and Huginn at 0% (the answer’s rank collapses to _∼_ 10^3^(_._5) ). This ordering follows the proximity of each model’s states to token space (§5.3, §6.2): removing the answer direction has the largest effect on directly decodable representations, whereas Qwen’s more distributed mid-stack retains roughly half of its correct answers.

The ten-direction ablation tests which results depend on the interface and which depend on the width of the ablation. At each intervention (layer, position), we project out the span of the ten inventory atoms that best reconstruct that state (bases orthonormalised and frozen from an uninstrumented forward pass, protocol in Appendix E); a matched control removing the span of ten fixed random-token atoms leaves answers 87–97% intact on every model. The effects therefore reflect removal of the inventory subspace rather than the loss of any ten dimensions. For Huginn, the last window alone is as destructive as all windows together (0/28 vs. 1/28 kept), while the second-to-last window remains a null (27/28). Thus, the window equivalence becomes even stronger under the wider ablation. Ouro behaves differently: removing the ten-direction subspace at loop 2 alone leaves 6/42 answers (14%), whereas removing a single concept direction left 42/42. Re-derivation through the checkpoint restores a deleted direction but not the state’s wider J-space component. A loop-2 ablation therefore has no behavioural effect only when it is narrow enough for the next loop to reconstruct, and single-direction ablation understates what that band carries. Qwen changes in the same direction, but less strongly: 64% of answers remain correct under the ten-direction ablation, compared with 72% for the argument direction and 97% for the control.

### 7.3 Verbalising new content requires explicit deep supervision; steering existing content does not

The verbal-report and introspection experiments produce contrasting results. Huginn is the _best_ of the three at verbal report — swap a concept the model has already computed and is about to emit, and it reports the injected concept in its top-5 at 98% (124/126) — yet a _null_ at introspection: injecting a concept the model was not computing never makes it reportable (0 of 97 concepts reach rank _≤_ 100 at any injection strength), even though the injections measurably move ranks (median 40,666 _→_ 16,258 at the strongest). Ouro is the only model that wins the second task — injected concepts become top-1 reportable at 31% — and also the only model whose training explicitly applies the LM head at every iteration. This refines and resolves the prediction left open in §5.3: implicit head pressure (Huginn’s sampled recurrence depths) suffices to make _existing_ content steerable and reportable — rerouting what the model already computed, where Huginn’s exact loop-invariant transport (§4.3) and persistence (§5.2) make the final windows an unusually clean write surface — but making _new_ , injected content verbalisable tracks explicit per-iteration supervision, and nothing else in the suite substitutes for it. Scale is not a substitute: the 27B Qwen model is also null at introspection (its best injected concept reaches only rank 173).

### 7.4 What the nulls mean: readable is not usable, and inert content is invisible by construction

The suite’s failures are as informative as its successes, but they are not one kind of failure. We distinguish three kinds: _capability floors_ (the model cannot do the task, so the intervention has nothing to show), _readout bounds_ (the content is there, but the lens cannot decode enough of it), and one _instrument blind spot_ (content the lens cannot see even in principle). Each changes what a null is evidence of.

**Capability floors.** The language-selectivity task shows the model a passage and asks a question about it (naming its author); the selectivity signature is that the passage’s _language_ — never asked about — should be readable in the band only when the task makes it relevant. Huginn shows exactly that signature (language label hit@5 1.00 when task-relevant vs. 0.62 when incidental) — yet answers the author question itself 0/8. The workspace demonstrably holds and gates the content, but the 3.5B model lacks the capability to use it. Huginn’s poor probe-swap and flexible-generalisation rows carry the same caveat — its no-intervention task baselines (28% and 44%) are the lowest of the three models — so those nulls are partly “cannot do the task”, not “has no workspace”.

**Readout bounds.** The capacity task asks the model to hold a list of words in mind (four 20-word blocks, probed at comma positions) and counts how many remain retrievable from the band. All three models score far below the original study’s _∼_ 25 words — but the ordering is _inverse_ to parameter count (Ouro 2.4 _>_ Huginn 1.8 _>_ Qwen 0.7), which rules out “small models store less”. It instead matches how _decodable_ each model’s band is (§6.2): the measure is bounded by what the readout can extract, not by what the state holds. The inventory estimator of §6.1 makes this bound directly visible: scoring the same word lists by pursuit reconstruction instead of the rank criterion roughly doubles every plateau — Qwen 0.7 _→_ 2.0, Ouro 2.4 _→_ 4.9, Huginn 1.8 _→_ 3.6 words — while preserving the inverse-to-scale ordering and remaining a factor of five below the original’s _∼_ 25. The measured capacity moves with the estimator, as an instrument bound must; and under a coefficient threshold of 0 _._ 05 _∥h∥_ Qwen’s plateau collapses to 0.1, consistent with how little of its state the atoms explain (4.3% variance, §6.1). Our capacity figures are facts about the instruments’ reach, not the models’ memory.

**The instrument blind spot.** Top-down summoning names a category label and asks the model to bring an absent concept to mind; the measure is whether that concept becomes readable in the band above matched foils. It is _∼_ 0 on all three models, and in the strongest sense: the summoned concept scores no better than foils, and asking vs. not asking changes nothing — the manipulation never takes hold. What summoning shares with capacity is that both probe content that is _not driving the current next token_ — a concept merely held in mind, a word list waiting to be recalled. The lens is structurally blind there. _Jv_ = E[ _∂h_ target _/∂hv_ ] is built entirely from “how does this state influence the target” — so any content that is not currently influencing the next token contributes nothing to _J_ , and the lens maps it to (near) zero regardless of whether it is present. The instrument therefore cannot distinguish “not held” from “held but idle”; the limitation is the same for all three models, which is why these failures are the only uniform ones in Table 5. Any study using Jacobian artefacts as a data modality inherits it: the lens sees the workspace’s _working_ content, not its inventory.

### 7.5 Robustness: reasoning SFT does not reorganise the workspace

Everything above is measured on a pre-trained checkpoint; one might worry the workspace interfaces are fragile to further training. Re-running the complete pipeline — lens fitting, the structural analyses, and the full causal suite — on Ouro-2.6B-Thinking, the reasoning SFT of our base checkpoint ( _∼_ 8.3M supervised examples) shows that they are not. Every architecture-level signature reproduces within a few points — writes must still span every remaining loop, the loop-2-only ablation is again a near-null, and introspection is point-identical at top-1 31% — while task baselines dip slightly and intervention effects move with them. The pursuit-based measurements reproduce alongside the rest: the capacity plateau is 5.1 words (base 4.9) and the loop-2-only subspace ablation keeps 11% (base 14%). The two exceptions are themselves informative: the language-selectivity gate collapses to always-on (∆+0.25 _→_ +0.00), and ignition sharpens (10–90 width 0.55 _→_ 0.20). Appendix C gives the full comparison.

## 8 Conclusion

Gurnee et al. [2026] left open whether depth in a transformer plays a role similar enough to recurrence in a brain to enable the same workspace functionality. For the weight-tied special case they identify, our functional answer is yes. Both looped models exhibit the full workspace signature — content that is verbalisable, causally steerable by writes and ablations along lens directions, and selectively engaged by task demands. Re-applying the same weights therefore does not prevent a workspace from forming. What recurrence changes is the workspace’s _interfaces_ : every route by which content enters, survives, or leaves the workspace follows the iteration structure rather than absolute depth.

We find that the two models develop different representations at their iteration boundaries, and these differences shape how their workspaces can be accessed. At Ouro’s supervised checkpoints, each loop’s conclusions are decoded into a token-aligned state and then re-encoded by the next loop. This makes the loop-to-loop process nearly Markovian, destroys the workspace component in transit, and forces any write to be maintained across every remaining loop’s re-derivation (38% spanning all loops vs. 21% for one). Huginn’s raw latent handoff preserves content across all sixteen recurrences, but reads and writes work only within a _∼_ 2-recurrence horizon of a lens’s own target — the last window alone is exactly as effective as all of them, for writes and ablations alike. Weight tying also reduces the cost of fitting the instrument: the fitted transports repeat across iterations (exactly on Huginn, at cosine 0.9991), so full-depth lens coverage of a recurrent model costs one iteration’s fit rather than _r_ of them.

The report experiments separate two abilities that the original study’s standard transformer did not distinguish. Steering _existing_ content into words needs only head pressure at depth in some form — Huginn, with no explicit deep supervision, is the best of the three models at verbal report (98%). Verbalising _new_ , injected content is a different matter: of everything we tested, only explicit per-iteration supervision predicted it. Ouro, the one model trained that way, is the only one that reports an injected concept (top-1 31%); Huginn never does (0/97), and the 27B Qwen model shows that untied scale cannot substitute for this supervision. Among the models tested, Huginn is also the closest transformer analogue of content held across processing cycles: content computed by recurrence 2–3 remains present, unverbalised, through every later recurrence. The analogy has a sharp limit — the injection-path experiments show this persistence is continuous recomputation from a re-injected input encoding, not a memory buffer; content does not survive even one recurrence beyond its last refresh. Whether a looped architecture can hold workspace content _without_ such re-grounding is a question for controlled training runs, not for released checkpoints.

The contribution has two main limitations. The first concerns the instrument: mean-Jacobian lenses are linearised summaries with a single-token concept vocabulary. They stop reading the state beyond their validated horizon, report the tiling rather than the model when lens families are unequally spaced, and are structurally blind to content that is not driving the current next token; readout counts depend on the estimator, and single-direction interventions can understate what a subspace carries. Any use of Jacobian artefacts as a data modality inherits these limits. Second, our cross-architecture comparison uses one pre-trained checkpoint per family, so architecture is confounded with training recipe and scale. Following checkpoints throughout pretraining would test whether our conclusions are stable over training and reveal when J-space emerges and how its geometry and causal interfaces develop. A systematic sweep over Huginn’s test-time recurrence count also remains future work. Within those limits, the result is consistent across three measurement families with different failure modes and remains unchanged after reasoning SFT.

Despite their different interfaces, our results suggest that neither looped model simply stores completed workspace content. Ouro’s later loops recover content after early interventions, while Huginn’s output depends on continued re-injection of the input throughout recurrence. Looped architectures are adopted for their reasoning strength. An open question is whether this continued computation contributes to that strength through built-in self-correction, or is overhead that a better iteration interface could remove. Two released checkpoints that differ simultaneously in supervision, injection, and training recipe cannot separate these possibilities. The instruments developed here make it possible to test them in controlled training runs that vary only the boundary interface.

## References

- Bernard J. Baars. _A Cognitive Theory of Consciousness_ . Cambridge University Press, 1988.

- Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. _arXiv preprint arXiv:2303.08112_ , 2023.

- Hugh Blayney, Álvaro Arroyo, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Michael M. Bronstein, and Xiaowen Dong. A mechanistic analysis of looped reasoning language models. _arXiv preprint arXiv:2604.11791_ , 2026.

- Thomas Blumensath and Mike E Davies. Gradient pursuits. _IEEE Transactions on Signal Processing_ , 56(6): 2370–2382, 2008.

- Patrick Butlin, Robert Long, Eric Elmoznino, Yoshua Bengio, Jonathan Birch, Axel Constant, George Deane, Stephen M. Fleming, Chris Frith, Xu Ji, et al. Consciousness in artificial intelligence: Insights from the science of consciousness. _arXiv preprint arXiv:2308.08708_ , 2023.

- Stanislas Dehaene and Jean-Pierre Changeux. Experimental and theoretical approaches to conscious processing. _Neuron_ , 70(2):200–227, 2011.

- Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In _International Conference on Learning Representations_ , 2019.

- Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. _Advances in Neural Information Processing Systems_ , 38, 2025.

- Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In _International Conference on Machine Learning_ , 2023.

- Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey. Verbalizable representations form a global workspace in language models. _arXiv preprint arXiv:2607.15495_ , 2026. URL `https://transformer-circuits.pub/ 2026/workspace/index.html` .

- Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. In _International Conference on Learning Representations_ , 2020.

- Wenquan Lu, Yuechuan Yang, Kyle Lee, Yanshu Li, and Enqi Liu. Latent chain-of-thought? decoding the depth-recurrent transformer. _arXiv preprint arXiv:2507.02199_ , 2025.

- George A. Mashour, Pieter Roelfsema, Jean-Pierre Changeux, and Stanislas Dehaene. Conscious processing and the global neuronal workspace hypothesis. _Neuron_ , 105(5):776–798, 2020.

- Neel Nanda, Arthur Conmy, Lewis Smith, Senthooran Rajamanoharan, Tom Lieberum, János Kramár, and Vikrant Varma. Progress update# 1 from the gdm mech interp team: Full update. In _AI Alignment Forum_ , 2024.

- nostalgebraist. Interpreting GPT: the logit lens. LessWrong, 2020. URL `https://www.lesswrong.com/ posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens` .

- Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. In _International Conference on Learning Representations_ , 2024.

- Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, et al. Scaling latent reasoning via looped language models. _arXiv preprint arXiv:2510.25741_ , 2025.

## A Verification of reported numbers

Every quantitative claim in this paper is re-derived from the released raw artefacts — per-trial result records, run logs, and the readout dumps — by an automated verification suite included with the code release (232 checks, plus a companion suite of 132 checks covering the gradient-pursuit artefacts of Appendix E). Each suite runs with one command and audits every table and inline statistic. Most values are recomputed from per-trial records, with percentages and 95% Wilson intervals recomputed from raw counts. For four experiment families the artefacts store only condition-level aggregates (directed modulation, dual task, line-count selectivity, and the Qwen3.6-27B verbal report — marked agg in the tables below); for these the suite verifies the reported value against the stored aggregate but cannot independently re-derive it. Re-running those families with per-trial recording is planned alongside experiment (iii) of the open question in §7.1.

Table 6: The full causal suite with exact counts. Columns follow Table 5; “base” is the no-intervention task baseline. Modulation cells are topic hit@5 under focus/control/suppress instructions; capacity cells give the last-quartile plateau with the peak (reached at comma 20, the first block boundary, on all three models) in parentheses.

|Family|Qwen3.6-27B 27B|Ouro-2.6B 2.6B|Huginn-0125 3.5B|
|---|---|---|---|
|Verbal report (top-5)|69% (agg; rank-1 49%)|106/133 (80%; rank-1 67%)|124/126 (98%; rank-1 98%)|
|Introspection (8_σ_)|null (best rank 173 of 101)|median RR 0.50; top-1 31%|null (top-1 0%; 0/97)|
|Modulation topic@5 (agg)|.38/.07/.07|.77/.39/.26|.28/.15/.19|
|Summoning expected@5|0.01|0.03|0.01|
|Two-hop probe swap|47/90 (52%); base 57/90|26/90 (29%); base 43/90|5/90 (6%); base 25/90|
|Flexible generalisation|80/192 (42%)|73/192 (38%); base 42/64|25/192 (13%); base 28/64|
|Selectivity: language (∆@5)|+1.00|+0.25|+0.38|
|Selectivity: line count (agg)|0.09/0.00|1.00/0.09|0.09/0.09|
|Ignition (10–90 width)|0.10|0.55|0.76|
|Capacity, words held|0.7 (peak 1.4)|2.4 (peak 5.0)|1.8 (peak 3.1)|
|Dual task, of 21 (agg)|concept 6; maths 1|concept 17; maths 0|concept 7; maths 3|

## B Full causal-suite tables

Tables 6–8 expand the condensed Table 5 to exact counts. Every value is re-derived by the verification suite of Appendix A; rows whose artefacts store only aggregates are marked agg.

**Token-position conventions.** The lens itself carries no position index — _Jv_ is a mean Jacobian, marginalised over the fitting corpus’s prompts and positions (§3) — so position enters only at readout and intervention time. Our default, matching the original study’s stated protocol [Gurnee et al., 2026], is that a direction intervention applies at _every token position_ as well as at every listed virtual layer: when §7 says Ouro clamps the bands of loops 2–4, the clamp acts on the whole sequence at every band layer of every remaining loop, and the behavioural outcome is read from the final position’s next-token distribution. Verbal-report swaps, flexible-generalisation swaps, two-hop probe swaps, and ablation all follow this default. Two families are span-restricted: the summoning clamp-transfer edits only the stimulus’s token span — more conservative than the original all-positions protocol, since the edit reaches the answer position only through attention — and introspection injects over the token span of the user instruction, matching the original’s user-turn injection, with the effect read at the final position. Rank-probe readouts (summoning, modulation, selectivity, capacity, dual task) take the minimum lens rank over band layers _×_ a task-defined span: the stimulus or carrier-sentence tokens, every position after the probed passage, or the comma positions of the capacity list.

Two rows deserve notes. Summoning’s zero is a _no-discrimination_ zero in the strongest sense: summoned concepts score no better than matched foils (difference _≤_ 0 _._ 01 on all models) and asking versus not asking changes nothing ( _|q_ 1 _− q_ 2 _|_ = 0 _._ 000 everywhere). And the line-count selectivity cells read “band readability of the count when directly asked / when never asked”: only Ouro shows the gated pattern.

The fourth category, numbers, is omitted from the table for width: it never flips under any condition on either looped model (0/48 throughout; direction cosine 0.64 on Ouro-2.6B). Note the Huginn last-window row is not merely equal in total to the all-windows row — the per-category counts are identical.

## C Robustness to reasoning SFT: Ouro-2.6B-Thinking

We repeated the complete Ouro-2.6B pipeline — lens fitting on the same 1000-prompt corpus with identical hyperparameters, the structural analyses, and the full causal suite — on Ouro-2.6B-Thinking, the reasoning

Table 7: Flexible generalisation by intervention condition — the raw data behind §7.1. Each trial swaps a source concept’s lens direction for a target concept’s and scores whether the answer flips to the injected concept’s implication; per-category columns give flips out of 48. Qwen’s run stored only the total. No-intervention baselines: Ouro-2.6B 42/64, Huginn-0125 28/64 (44%). Bottom: mean cosine between lens directions of same-category concepts (Ouro-2.6B, loop-2 band) — the category-separability covariate discussed in §7.1’s open question.

||Condition|Flips/192|% (95% CI)|countries|months|animals|
|---|---|---|---|---|---|---|
|Qwen3.6-27B|band L24–48|80|42 (35–49)|—|—|—|
|Ouro-2.6B|loops 2+3+4|73|38 (31–45)|33|16|24|
||loop 2 only|40|21 (15–27)|31|0|9|
||loop 4 only|48|25 (19–31)|30|12|6|
||loops 2+3+4, 2_×_ strength|14|7 (4–11)|10|0|4|
|Huginn-0125|all three windows|25|13|12|10|3|
||last window only|25|13|12|10|3|
||second-to-last window only|5|3|1|0|2|
||all windows, 2_×_ strength|2|1|1|0|1|
|Ouro-2.6B sam|e-category direction cosine|||0.36|0.67|0.39|

Table 8: Direction ablation by condition — the raw data behind §7.2. Each cell is answers kept (unchanged and correct) after ablating the named direction at the intervention layers, over the prompts each model answered correctly at baseline. “Early interface only” is the loop-2 band on Ouro-2.6B and the second-to-last lens window on Huginn-0125; Qwen3.6-27B has a single interface, so the split does not apply.

|Ablated direction|Qwen3.6-27B (_n_=36)|Ouro-2.6B (_n_=42)|Huginn-0125 (_n_=28)|
|---|---|---|---|
|Argument concept, all interfaces|26 (72%)|30 (71%)|17 (61%)|
|Argument concept, early interface only|—|42 (100%)|28 (100%)|
|Argument concept, last interface only|—|32 (76%)|17 (61%)|
|Answer token, all interfaces|19 (53%)|8 (19%)|0 (0%)|
|Unrelated word (control)|36 (100%)|42 (100%)|28 (100%)|

SFT of our base checkpoint ( _∼_ 8.3M supervised examples of maths, code, science and chat; the architecture is unchanged) [Zhu et al., 2025]. The only pipeline change is the model identifier. One protocol note: the Thinking tokeniser emits its reasoning prefix only when explicitly requested, and under the suite’s default chat template the model answers directly, so every comparison below is protocol-identical to base. All numbers in this section are re-derived by the verification suite of Appendix A.

**Structure is unchanged.** Deep supervision survives the SFT. The trained exit gate still concentrates its mass on the last two loops (0.37/0.46 on loops 3/4; base 0.34/0.49); logit-lens top-1 agreement at the four loop-end states spans 0.62–0.90 (base 0.60–0.92); and transport still factorises through the loop checkpoints, with mean composition cosine 0.962 over the 20 tested paths (base 0.971). **All interface signatures replicate.** The two headline causal structure results reproduce within a few points despite the SFT: writes must still span every remaining loop (the loop-2-only clamp is half-undone by re-derivation at 17%, and loop-2-only ablation is again a near-complete null at 97% kept), and the two opposite single-loop failure modes and the 2 _×_ -strength collapse all recur. Introspection is point-identical (top-1 31%). The pursuit-based families follow suit: the inventory capacity plateau is 5.1 words against base 4.9, and the loop-2-only pursuit-subspace ablation keeps 4/38 (11%) against base 14%, with the all-interface condition at 0/38 and the random-atom control at 87%. The workspace interface is a property of the pre-trained looped architecture, not something the reasoning SFT reorganises.

**Behavioural results change; intervention geometry does not.** The SFT costs a few points of raw task performance under the suite’s direct-answer protocol (flex-gen baseline 66% _→_ 59%, verbal report 80% _→_ 69%, probe-swap baseline 43/90 _→_ 35/90), and the clamped and injected conditions move by roughly the same amount — the ratio of intervention effect to baseline is preserved.

Table 9: The causal suite on base Ouro-2.6B versus its reasoning SFT. Conventions follow Table 6; flexiblegeneralisation rows follow the per-loop conditions of Table 7.

|Family|Ouro-2.6B 2.6B (base)|Ouro-2.6B-Thinking|
|---|---|---|
|Verbal report (top-5; rank-1)|80%; 67%|69%; 55%|
|Introspection (8_σ_: top-1; median RR)|31%; 0.50|31%; 0.50|
|Modulation topic@5, focus/control/suppress (agg)|.77/.39/.26|.76/.45/.15|
|Summoning expected@5|0.03|0.03|
|Two-hop probe swap|26/90 (29%); base 43/90|22/90 (24%); base 35/90|
|Flex-gen: no-intervention baseline|42/64 (66%)|38/64 (59%)|
|Flex-gen: loop 2 / 3 / 4 only|21% / 29% / 25%|17% / 28% / 26%|
|Flex-gen: loops 2+3+4 (at 2_×_ strength)|38% (7%)|35% (5%)|
|Selectivity: language (∆@5)|+0.25|+0.00|
|Selectivity: line count, direct/never asked (agg)|1.00/0.09|1.00/0.18|
|Ignition (10–90 width, in-band)|0.55|0.20|
|Capacity, words held (peak at comma 20)|2.4 (5.0)|2.1 (4.0)|
|Capacity, pursuit inventory (_θ ≤_0_._01)|4.9|5.1|
|Dual task, of 21 (agg)|concept 17; maths 0|concept 19; maths 0|
|Ablation kept: arg / answer / control|71% / 19% / 100%|66% / 26% / 97%|
|Ablation kept: loop-2 only / last loop only|100% / 76%|97% / 76%|
|Pursuit-subspace ablation kept: all / loop-2 / control|0% / 14% / 88%|0% / 11% / 87%|

**Two notable differences.** First, the language-selectivity gate collapses to always-on: the passage’s language label is readable at ceiling even when task-irrelevant (∆+0.25 _→_ +0.00), so Ouro-2.6B’s gated pattern in Table 5 does not replicate on the SFT model — consistent with reasoning SFT keeping more context features unconditionally active in the workspace. Second, ignition sharpens: the in-band 10–90 transition width drops 0.55 _→_ 0.20 (scrambled control 0.24 _→_ 0.12) — the Thinking model commits more abruptly once the band resolves. Both are single-condition observations; we report them as leads rather than conclusions. Answer-direction ablation shows a smaller difference in the opposite direction: ablating the answer direction is slightly _less_ destructive on Thinking (26% kept vs. 19%) while argument ablation is comparable.

## D Demonstration-metric semantics

The metric family of §6 comes from the original study’s interactive demonstration, whose exact semantics are not documented; we reverse-engineered them from its page bundle and reimplemented them over all three models.

**Dump protocol.** For each model we run the demonstration’s own 8 chat prompts, generate a 128-token greedy completion, and record, at every (virtual layer, position) cell, the top-8 tokens and probabilities of both readouts: the J-lens (softmax(unembed( _hJ_^(_⊤_) )), through the covering lens of §3.3) and the logit lens (softmax(unembed( _h_ ))). These grids are the released raw artefact; every §6 number is a function of them.

**Count (occupancy).** A token’s count is its number of occurrences over positions _×_ layers _×_ 8 slots, with whitespace-only and non-word token strings excluded. The demonstration’s default layer window additionally skips the first 29% of depth; §6.1 uses full depth instead, since the depth profile is the object of study. (The skip is well chosen for a standard transformer: on Qwen3.6-27B it starts at L18, just below the content onset at L20–23 that we measure independently — but on Ouro-2.6B the same fraction leaves the content-free early layers of every loop inside the window, one of the confounds that led us to drop the demonstration’s contrastive “Diff” statistic from this paper; its analysis is retained in the reproduction report.)

**Probability field.** The calibration measure of §6.2 is the probability a readout assigns to the token the model actually emits at the next position, averaged over positions and prompts. Because the dumps store only each cell’s top 8, a token outside them contributes zero, making the reported values lower bounds; chance on the _∼_ 150k vocabularies is _∼_ 10^(_−_5) .

**The partial-tiling reading artefact.** Under the original four-lens family, later Huginn windows sit closer to their covering lens targets than earlier ones, and the resulting occupancy trend rises monotonically with depth — a curve one would naturally read as content accumulating through the recurrences. Completing the family to a full tiling — twelve additional lenses, so that a target sits at every recurrence end (v5 to v61 in steps of 4, plus the coda) and every source lies within one recurrence of its covering target — flattens the trend into the plateau of Figure 3. Under the top-8 readout, the only estimator dumped under both tilings, the partial family reads 638 total slots on the animal prompt with 32 layers at exactly zero, the full tiling 1,622 with none; under the inventory, 65 of 67 layers are occupied at the primary grid point (all 67 ungated). Within each one-recurrence window the count still rises towards the lens target — the mean-J transport attenuation of §4.1 made visible as a sawtooth — but occupancy is nonzero at every depth any lens’s transport reaches. Ouro’s early-loop near-zeros, by contrast, are distance-controlled within each loop and reflect genuine absence. With unevenly spaced lens targets, apparent changes across depth can be caused by changing source-to-target distance alone.

**The animal prompt and replies.** Figure 3 is computed over the following transcripts (the occupancy grid covers prompt and completion positions alike). Prompt, sent identically to all three models:

I’m thinking of the largest land animal. What do they eat? Answer briefly.

Replies, verbatim:

**Qwen3.6-27B:** “The largest land animal is the African bush elephant. They are herbivores and primarily eat grasses, leaves, bamboo, bark, roots, and fruits.”

**Ouro-2.6B:** “Elephants are the largest land animals and they eat a variety of plants, including leaves, bark, and fruits.”

**Huginn-0125:** “The largest land animal is the African elephant, which is a herbivore and feeds on a variety of plants, including grasses, leaves, and bark.”

The latent/spoken classification of §6.1 follows from these texts with singular/plural variants merged: _tree_ is latent for all three models, _food_ for Qwen3.6-27B, _mammal_ and _vegetation_ for Ouro-2.6B, _fruit_ and _root_ for Huginn-0125 (which, uniquely, never names them).

## E The gradient-pursuit concept inventory

The occupancy estimator of §6.1 is the original study’s concept inventory: a sparse non-negative reconstruction of the state from J-lens atoms. This appendix gives the algorithm, the estimator’s agreement and disagreement with the demonstration’s top-8 readout, and the protocols and full results of the two pursuit-based causal families (§7.2, §7.4).

**Algorithm.** At each (virtual layer _v_ , position _t_ ) the dictionary is the intervention direction family of §7, unit-normalised: _aτ_ = _Jv_^(_⊤uτ/∥J_) _v_^(_⊤uτ∥_overthevocabulary,withwhitespace-onlyandnon-wordtokenstrings) excluded exactly as in the top-8 counts (Appendix D). Greedy pursuit runs on the residual _r_ (initialised to _hv,t_ ): select arg max _τ ⟨r, aτ ⟩_ among positive scores — no materialised dictionary is needed, since _⟨r, Jv_^(_⊤uτ⟩_) is the lens-logit operation applied to _r_ , normalised by precomputed atom norms — then refit all selected coefficients under a non-negativity constraint (projected gradient on the support’s Gram system, batched over positions), subtract the reconstruction, and repeat until _k_ max = 50 atoms or no positive score remains. The dumps record each step’s selection, a full coefficient snapshot at _k ∈{_ 8 _,_ 16 _,_ 25 _,_ 50 _}_ , and the residual fraction, so every occupancy count at any ( _k, θ_ ) is an offline slice of one artefact. Occupancy at ( _k, θ_ ) counts the atoms among the first _k_ whose refitted coefficient is at least _θ∥h∥_ ; the body reports _k_ = 25, _θ_ = 0 _._ 05, with robustness checked over the full grid _k ∈{_ 8 _,_ 16 _,_ 25 _,_ 50 _} × θ ∈{_ 0 _,_ 0 _._ 01 _,_ 0 _._ 05 _}_ . The cells are the same (layer, position) grids as the top-8 dumps — the stored `seq_ids` reproduce each recorded forward deterministically — so the two estimators align cell-for-cell on all three models.

Figure 4: Count-by-layer occupancy under the demonstration’s top-8 readout — the estimator-comparison counterpart of Figure 3, same prompt, same token classes, same layout. The band, loop, and window structure appears under both estimators.

**Agreement and disagreement with the top-8 readout.** The demonstration’s top-8 softmax readout — whose semantics we reverse-engineered from the study’s page bundle (Appendix D) — serves as a reference estimator: every occupancy dump was also read through it. The two estimators pick genuinely different tokens: mean per-cell Jaccard overlap between the pursuit support and the top-8 slots is 0.07–0.24, and the near-twin rate (two read tokens identical up to case or plural) falls from 30–37% of cells under top-8 to 0.1–0.3% under pursuit — confirming the original study’s remark that the inventory is less redundant than inner-product top- _k_ . Despite that, every structural claim of §6.1 except two holds under both estimators at every grid point. Under top-8 the corresponding numbers are: Qwen animal prompt, 6 latent slots below L20, 1,076 latent vs. 1,116 spoken in the band, 56 vs. 231 at L55+; Ouro currency riddle, _euro_ 679 _→_ 508 against _Italy_ 199 _→_ 584 across loops; Huginn currency riddle, 2 slots for the country family over all 67 virtual layers (pursuit grid maximum: 9); Huginn full tiling, 1,622 total slots, mid-window plateau 86–108 against 21 in the first window, coda 343. Figure 4 is the top-8 counterpart of Figure 3. Two top-8 results do _not_ survive the estimator change. On the animal prompt, the top-8 latent class thins fourfold against the spoken class at L55+ (the 56 vs. 231 above), whereas the inventory’s latent/spoken ratio is flat-to-rising at every grid point — the thinning is a property of top-8’s eight-slot competition, not of the state; the layer-level commitment that is real on Qwen lives on the sport prompt (§6.1). And Ouro’s in-band excess holds under top-8 on raw loop totals, but under the inventory only as a per-layer rate (§6.1).

Table 10: Capacity plateaus (words held, last-quartile mean) under the rank probe of Table 6 and under the pursuit inventory at _k_ = 25 for three coefficient thresholds.

|Model|Rank probe|_θ_ = 0|_θ_ = 0_._01|_θ_ = 0_._05|
|---|---|---|---|---|
|Qwen3.6-27B|0.7|2.0|1.9|0.1|
|Ouro-2.6B|2.4|4.9|4.9|4.8|
|Huginn-0125|1.8|3.6|3.6|2.2|
|Ouro-2.6B-Thinking|2.1|5.1|5.1|5.0|

Table 11: Answers kept under ten-direction pursuit-subspace ablation. “Early interface only” is the loop-2 band on Ouro-2.6B and Thinking and the second-to-last lens window on Huginn-0125, as in Table 8; Qwen3.6-27B has a single interface.

|Condition|Qwen3.6-27B (_n_=36)|Ouro-2.6B (_n_=42)|Huginn-0125 (_n_=28)|Thinking (_n_=38)|
|---|---|---|---|---|
|Pursuit subspace, all interfaces|23 (64%)|0 (0%)|1 (4%)|0 (0%)|
|Pursuit subspace, early interface only|—|6 (14%)|27 (96%)|4 (11%)|
|Pursuit subspace, last interface only|—|0 (0%)|0 (0%)|0 (0%)|
|Random-atom control, all interfaces|35 (97%)|37 (88%)|26 (93%)|33 (87%)|

**Variance explained.** The pursuit reconstruction at _k_ = 50 explains on average 4.3% of state variance on Qwen3.6-27B, 10.2% on Huginn-0125, and 21.5% on Ouro-2.6B — ordered exactly as the models’ decodability (§6.2) and small in absolute terms everywhere. The _k_ -curve has no elbow: _k_ = 25 captures 91% of the _k_ = 50 value on Qwen but only 78% on Ouro and 63% on Huginn, so the _k ≤_ 25 cap is a convention on the recurrent models, not a saturation point. This motivates reporting occupancy with a coefficient threshold rather than raw selection counts.

**The coefficient gate.** Two of the §6.1 zeros — Qwen’s below-L20 absence on the animal prompt and its last-8-layer sport zero — exist only under the _θ_ = 0 _._ 05 _∥h∥_ gate (exact zeros at _k ≥_ 16 and _k ≥_ 25 respectively; near-zero at smaller _k_ ). Ungated selection fills slots at every depth: greedy pursuit always selects _something_ when the atoms explain so little of the state (4.3% on Qwen), so absence registers as vanishing refitted coefficients rather than as empty slots, and the gate turns “selected at negligible weight” into “not counted”. This is the role the top-8 probability floor plays implicitly for the demonstration’s estimator, which shows the same below-L20 absence (6 slots) without an explicit threshold. On Qwen, absence and commitment are therefore statements about coefficient mass, not about which atoms remain selectable.

**Capacity under the inventory.** The capacity task of §7.4 re-scored with pursuit in place of the rank probe: identical seeded word lists and prompts, one recorded forward per trial, pursuit at the comma positions over each model’s read layers. A word counts as held at a comma if, at any read layer, a selected atom with coefficient _≥ θ∥h∥_ matches it (exact token id or case-insensitive string). Table 10 gives the last-quartile plateaus at _k_ = 25.

**Subspace ablation.** The ten-direction ablation of §7.2 runs in two passes. An uninstrumented forward pass records, at every intervention (layer, position), the first ten pursuit atoms for that cell; each cell’s atoms are orthonormalised by QR into a frozen basis _Q_ . The intervention pass then applies _h ← h − QQ_^(_⊤_) _h_ at those cells — removing the state’s projection onto its own leading inventory subspace, as computed on the clean run. The control replaces each cell’s basis with the span of ten fixed random-vocabulary atoms (shared per layer). Trials, interfaces, and scoring are identical to the single-direction ablation family (Table 8). Table 11 gives the full counts.

**Verification.** All pursuit artefacts — the per-cell dumps with coefficient snapshots, the capacity dumps, and the ablation records — are released alongside the top-8 dumps, and a companion verification suite (132 checks; Appendix A) re-derives from them every pursuit number in this paper, including each robustness verdict at all twelve ( _k, θ_ ) grid points.

Table 12: One readout cell per model on the hidden-sport prompt: the top-5 tokens (with probabilities) of the J-lens and the logit lens at the same (virtual layer, position). “While writing” quotes the completion text around the position being read. ␣marks a tokeniser leading space. The three cells show the three regimes the paper describes: on Qwen3.6-27B, the unspoken concept is sharp in the J-lens and invisible to the logit lens; on Ouro-2.6B, a mid-loop state read through that loop’s own lens holds the candidate sports the model is still deliberating; on Huginn-0125, both lenses read the _same_ word — the J-lens sharpens (0.88) what the near-token-space state already shows weakly (0.06).

|Cell|Lens|Top-5 readout|
|---|---|---|
|Qwen3.6-27B L42,|J-lens|`␣football` .54, `␣soccer` .23, `␣Football` .09, `Football` .04, `football` .03|
|“thud of a boot connecting with a [sphere]”|logit|`␣(` .00, `...` .00, `␣bag` .00, `sc` .00, `␣` .00|
|Ouro-2.6B loop 3, physical L26,|J-lens|`␣football` .50, `␣soccer` .21, `␣sports` .09, `␣rugby` .07, `␣stadium` .03|
|“describe the atmosphere at a [big]”|logit|`␣Football` .03, `␣football` .02, `␣championship` .01, `␣stadium` .01, `␣sports` .01|
|Huginn-0125 recurrence 15,|J-lens|`␣excitement` .88, `␣energy` .05, `␣emotion` .03, `␣enthusiasm` .01, `␣emotions` .01|
|“stadium was alive, pulsating with [the]”|logit|`␣excitement` .06, `␣intensity` .04, `␣energy` .02, `␣anticipation` .02, `␣electricity` .01|

## F Worked examples

The body of the paper reports aggregates; this section shows single raw data points, so the reader can see concretely what a lens readout and an intervention record look like. Every value is pinned to its exact cell by the verification suite of Appendix A.

**One prompt, three workspaces.** All three readout cells come from the same demonstration prompt of §6, whose completion must hold a concept in mind without saying it:

_Think of a specific sport, but don’t say its name. Describe the atmosphere at a big match of that sport without naming it._

Table 12 shows one cell of each model’s (layer _×_ position) readout grid — a high-probability cell chosen to be characteristic of that model’s pattern — with the five most probable tokens under each lens.

Each cell illustrates a characteristic readout pattern for one model. Qwen3.6-27B is writing “. . . the heavy thud of a boot connecting with a _sphere_ ” — deliberately avoiding the word _ball_ — and at mid-band L42 the J-lens reads the sport it is not naming at 0.54, while the logit lens at the same state reads punctuation fragments at _∼_ 10^(_−_3) : mid-stack content exists but is not token-decodable without transport (§6.2). Ouro2.6B’s cell sits mid-loop — between the checkpoints deep supervision trains — while its completion is still the generic “the user wants me to describe the atmosphere at a. . . ”; the loop-3 lens reads the candidate sports the model is weighing (it later settles on rugby), where the same state’s direct decode is an order of magnitude weaker. Huginn-0125’s cell shows why we call its states near-token-space (§5.3): the logit lens already reads the right word everywhere, weakly, and the J-lens concentrates it.

**One intervention record.** The introspection experiment (§7.3) injects a concept the model is _not_ computing along its lens direction, asks the model to name what it is thinking of, and records the injected concept’s rank in the answer distribution at each injection strength. The per-trial records make single concepts inspectable; the concept _lightning_ appears in all three models’ concept lists:

|Injection strength|0_σ_|0_._5_σ_|1_σ_|2_σ_|4_σ_|8_σ_|
|---|---|---|---|---|---|---|
|Ouro-2.6B|7,524|1,809|622|114|31|8|
|Qwen3.6-27B|10,533|1,084|533|292|218|189|
|Huginn-0125|15,969|3,755|3,935|3,798|3,422|3,151|

The verbalisation split of §7.3 is visible in this one row. As injection strength increases, the concept’s rank improves monotonically in Ouro-2.6B (explicit per-iteration supervision) until it enters reportable range. The rank also improves substantially in Qwen3.6-27B but remains two orders of magnitude below that range. In Huginn-0125, the rank responds to the first increment and then plateaus: the injection measurably moves the state, but nothing carries it to the output as a word.

## G Adapter validation and silent-failure traps

Fitting a Jacobian lens through a recurrence exposes failure modes that do not exist on feedforward stacks. Each of the following corrupts results _without raising an error_ :

1. **No-grad recurrence phases.** Huginn’s iterated forward splits recurrences into no-grad and with-grad phases; passing a scalar step count places _all_ of them in the former, making every Jacobian through the recurrence exactly zero. The fix is to request (0 _, r_ ) steps; our smoke test asserts a non-zero Jacobian at every recurrence.

2. **Stochastic state initialisation.** Huginn draws its initial recurrent state afresh per forward. The adapter seeds the generator per call (and disables test-time noise); determinism is verified bit-exact, since a lens averaged over unmatched forwards is an average over different computations.

3. **Gradient checkpointing.** Backward-pass recompute re-fires block hooks and desynchronises the virtual-layer counters; checkpointing must stay off.

The adapter’s smoke tests additionally verify that each physical layer yields distinct per-firing activations, that the virtual-final state reproduces the model’s own logits exactly, and that forwards are bit-exact deterministic under the seeded initialisation.

### H transformers-5 compatibility

Both looped models ship remote code written against transformers 4.x; our experiments run under 5.14. Ouro-2.6B needs three load-time patches: (a) `config.pad_token_id` is no longer auto-created and must be set to `None` ; (b) the rotary-embedding registry lost its `’default’` (classic RoPE) entry, so a shim reregisters the default parameter function on the dynamically loaded rotary class; (c) `create_causal_mask` renamed keyword arguments. Huginn-0125 needs exactly one: it declares `_tied_weights_keys` in the 4.x list form, which crashes 5.x `post_init` ; the adapter rewrites it to the dict form `{"lm_head.weight": "transformer.wte.weight"}` before `from_pretrained` . Huginn’s custom recurrence cache is untested under 5.x; no experiment in this paper uses it (all forwards are cache-free and nothing calls `generate()` ).
