# A Sense of Proprioception in Language Models

- **Authors:** Jonas Geiping
- **Date:** 2026-10-12
- **Source:** https://arxiv.org/abs/2610.17495
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `geiping_sense-proprioception_2610.17495`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Preprint

## Jonas Geiping

ELLIS Institute Tübingen Max Planck Institute for Intelligent Systems Tübingen AI Center

## Abstract

In humans, proprioception corresponds to the sense of self-movement or position, e.g. understanding the position of one’s arm. This sense is often less wellunderstood and in popular understanding sometimes conflated with introspection. But, notably, proprioception is a separate sense, with dedicated receptors and verifiable outcomes. On the other hand though, language models cannot observe their own internal states during training (and for example answer the question which experts activated in a certain layer of an MoE) and so cannot accurately report on them. Yet, such information is, by virtue of being internal to the model, both accessible, and readily constructed into verifiable signals. We argue that this provides the basis for a digital analog of proprioception. As such, we train models through SFT and on-policy training to equip them with this ability. We show that modern models can be trained to understand a range of subtypes of proprioception, such as routing decision awareness, token entropy, decision depth or logit lens information, without affecting model capabilities. We then run these models through a broad evaluation of situational awareness, legibility, and epiphenomenality, to evaluate in what ways access to proprioception changes model behavior.

## 1 Introduction

Intelligent systems built on the basis of generative large language models are nowadays deployed en masse, acting as semi-autonomous digital agents. Yet, in contrast to humans, and to almost all lifeforms we observe, language models are trained offline and so not constructed with explicit mechanisms that provide feedback about their internal functioning.

In humans, for example, one such feedback system is _proprioception_ , the sense of the orientation of the body. This sense is crucial to precisely control the body (Proske and Gandevia, 2012; Kandel et al., 2012; Cole and Paillard, 1995; Gallagher and Cole, 1995; McNeill et al., 2010). Similar mechanical analogues are common in control theory and robotics, given that from a cybernetics perspective, systems that integrate feedback about internal states are more robust (von Holst and Mittelstaedt, 1950; Wolpert et al., 1998; Bongard et al., 2006; Kwiatkowski and Lipson, 2019; Lee et al., 2020).

Language models however, trained with modern pre-training and post-training pipelines, do not observe internal states directly during training, and so can only learn implicit associations (Kadavath et al., 2022; Binder et al., 2024; Betley et al., 2025). That they do learn these associations in certain situations has often been studied in the context of _introspection_ (Lindsey, 2026; Ji-An et al., 2025; Ackerman, 2026). Yet, by its nature, introspection in intelligent systems is harder to define and to measure neutrally, complicating the study of these processes by which models model their internal states. In turn uncertainty about introspection complicates inferences drawn from these measurements about e.g. model robustness, safety, interpretability or personas (Song et al., 2025b; Comsa and Shanahan, 2025; Long et al., 2024; Butlin et al., 2026).

In this work we first lay out the argument that digital analogs of proprioception are a worthwhile object of study, separate from introspection. A running example of this in the paper will be the understanding of routing decisions inside the mixture-of-experts layers of a modern LLM, i.e. questions like “which expert most activated in layer 17 when processing the word ’questions’ in the previous sentence” (Jiang et al., 2024a; Dai et al., 2024). Unlike introspective questions, this one is a well-posed query about the internal functioning of a language model that has a ground-truth answer and is further valence-neutral. As such, we can study these types of internal modalities, sidestepping issues that occur in language model introspection, while still allowing for inferences in the sense of Perez and Long (2023). Given that this is a verifiable signal, we use on-policy fine-tuning stages to train model organisms of open-weight LLMs that exhibit forms of proprioception, finding that models can quickly access and generalize this sense, without impact on their capabilities. Finally, with these models in hand, we evaluate a range of benchmarks of model behavior, situational awareness, legibility and control, to observe in what ways models with proprioception behave differently.

Figure 1: Schematic of proprioception analogs for a generative language model. **Top** : We are interested in analogs of proprioception (which is the sense of body positioning in humans) in language models, as proprioception is a question about internal states adjacent to introspection, but compared to introspection (which is classically understood not to be a sense and not objectively verifiable) proprioception is tractable. **Bottom** : We train models with various forms of proprioception and observe whether this sense is learned and whether this affects model behavior.

Our study is motivated by three main questions. First, does it make sense to define proprioception separately from introspection for disembodied language models, and can models even learn to reliably access this information? Second, does access to these skills change model behavior, e.g. does it make models safer, more reliable, or the opposite? We investigate this neutrally from both angles, testing whether models become more reliable, but also whether they may become less safe. Lastly, we are motivated to study a _legibility_ hypothesis in the spirit of Premakumar et al. (2026): does training a model to reliably learn these abilities change the model’s internal structure to become more legible, e.g. to interpretability methods?

## 2 Proprioception is not introspection

There exists now a sizable body of work on _introspection_ in large language models, yet the question whether models can introspect, and in what situations remains unresolved, as language models’ self-reports are hard to separate from mimicry of introspection patterns picked up from training (Shanahan et al., 2023; Long, 2023; 2025) and relatedly, hard to separate from learned exteroception, i.e. an external inference of the answer without utilization of internals (Song et al., 2025b; Comsa and Shanahan, 2025; Singh et al., 2026). Interestingly, this mirrors a number of troubles with introspection in humans (Nisbett and Wilson, 1977; Schwitzgebel, 2008). While we may conceptualize it as such, via Shoemaker (1996), “introspection is not a sense”. In humans this is based on several arguments. First, there is no transducer, i.e. “no organ” of introspection separate from the brain (Shoemaker, 1996). Second, the functioning of introspection fails to be object-independent: (human) mental states are constituted by their monitoring and hence not structure-preserving (Boyle, 2009; Moran, 2002). Third, in human subjects, what is perceived as introspection can still be ‘transparent’, meaning an-swered by observing the external world and deriving an answer about an inner state (Evans, 1982; Byrne, 2005). This is related to, lastly, the failure of calibration (Wittgenstein, 1953): A report of introspection cannot be checked against information outside the head of the person being asked. In the sense of Carruthers (2011) introspection is merely the mode of interpretation turned inward, which is dissimilar from sensing. As such, introspection cannot be calibrated or corroborated by real senses.

We note that these concerns actually also hold for a range of strategies to evaluate introspection in language models. In particular, linear probes of model behavior (Alain and Bengio, 2017; Burns et al., 2022; Zou et al., 2023) may seem objective, but the probes themselves form an interpretation of a hidden state of the model that is itself calibrated from data, and, for example in probes for action valence as in Tagliabue et al. (2026), cannot be calibrated to an objective signal.

Yet, _proprioception_ in humans, while also considered to be an internal representation that exhibits immunity to error through misidentification (Wittgenstein and Docherty, 1958; Shoemaker, 1968), like introspection (Evans, 1982), does not follow these constraints. Proprioception of body position can be traced to real receptors, e.g. muscle spindles (Proske and Gandevia, 2012) that record and transmit information about a muscle’s length and stretch speed (Purves et al., 2001). And, in this way it does mirror the ambition of theorists who looked for proof of an “inner sense” for introspection (Armstrong, 1993; Lycan, 1996). Also, the position of the body is evidentially real, i.e. it exists without being observed, and is independent of its measurement, which also allows for calibration to other senses and by external observers. Humans learn to calibrate our fine-grained sense of proprioception through visual observation and cross-verification (Held and Hein, 1963; Botvinick and Cohen, 1998). This does not mean that proprioceptive senses are infallible, but that, in contrast to introspection, they can be falsifiable (Bottini et al., 2002; Vallar and Ronchi, 2009).

Porting this distinction of proprioception from introspection to language models provides a better answer to the premise of Perez and Long (2023), whose research proposal argues that while introspective self-reports of language models are unreliable, training models on introspection tasks that are externally verifiable, could transfer to self-reports. With the definitions as above, re-reading the proposal splits their training tasks cleanly into the proprioception group (plus trivia facts about the model, such as its provider or model name), and their evaluation tasks into the introspection group (in the Shoemaker (1996) sense).

### 2.1 Related work

**Introspection in Language Models and its Critics.** Language models have been found to predict their own behavior better than external models (Binder et al., 2024; Betley et al., 2025), understand their situation (Chen et al., 2024a; Laine et al., 2024), to show signs of metacognition (Berg et al., 2025; Ackerman, 2026), and to recognize their own writing (Panickssery et al., 2024). In tests, concept injections via steering vectors are noticed by large language models (Lindsey, 2026; Macar et al., 2026; Lederman and Mahowald, 2026) and can be detected latently in smaller models (Vogel, 2025; Pearson-Vogel et al., 2026). In certain setups, LLMs show partial control over their activation patterns (Ji-An et al., 2025). While it has been clear for longer that model self-reports are unreliable (Perez and Long, 2023; Lindsey et al., 2025), other work also finds that LLMs show no privileged self-access when queried about their language understanding (Song et al., 2025a). More broadly, introspection behaviors being explainable by external observers is a recurring concern (Binder et al., 2024; Song et al., 2025b; Lindsey, 2026)^1 . A related concern in Morris and Plunkett (2025) is that a model can provide an accurate report about an internal state, without being causally dependent on the targeted state (e.g. by forming a parallel structure that coincides with the state instead of a second-order structure that processes states). Singh et al. (2026) argue further that behavioral evidence may be fundamentally insufficient in establishing that a state itself is represented as a distinct object of second-order computation.

**Evaluating Language Model Self Reports.** A number of frameworks have been proposed to grasp the question of model understanding and awareness, for example in the context of welfare, ranging from assessing indicators to discussing moral status under uncertainty (Butlin et al., 2023; 2026;

> 1And is, as a concept, hard to make as precise for digital, replicable entities as it is for humans, at the least we may ask as in Song et al. (2025b) that the observer has less computation available, or from a different angle that information is restricted, e.g. that weight initialization, training data or order or at least training non-determinism are not available to the external observer.

|The model’s verbal report is ...|Self-Attributed|On-policy|Internal|Natural|
|---|---|---|---|---|
|Li et al.(2026)|–|–|✓|_◦_|
|Guo et al.(2026)|✓|_◦_|–|✓|
|Binder et al.(2024)|✓|_◦_|–|✓|
|Plunkett et al.(2025)|✓|–|–|✓|
|Shenoy et al.(2026),Kutsyk and Zieli´nski(2026)|_◦_|–|–|✓|
|Lin et al.(2022),Li et al.(2025)|✓|–|–|✓|
|Damani et al.(2026)|✓|✓|–|✓|
|Hahami et al.(2026),Rivera and Africa(2026)|✓|✓|✓|–|
|Macar et al.(2026)|✓|_◦_|✓|–|

Table 1: Comparison of work on introspection training and decoding** , separating whether the trained behavior is _self-attributed_ , i.e. a behavior of the model that is doing the reporting (identity), trained from offline examples of the behavior or _on-policy_ , whether the object of the report is an _internal_ state of the model (instead of an externally observable behavior), and whether the state reported on is _natural_ , i.e. a valence-neutral state that may appear during normal operations of the model. _◦_ : partial.

Chalmers, 2024; Long et al., 2024; Sebo and Long, 2025; Birch, 2024). Empirical studies, e.g. in model cards, rely mainly on self-reporting (Long, 2025; Anthropic, 2025), although higher-ordertheory indicators have been proposed in Butlin et al. (2023). Of these, HOT-3 (agency guided by a belief-formation and action selection system and predisposition to update beliefs in accordance with monitoring) is tested in Steinmetz Yalon et al. (2026) and critiqued in Singh et al. (2026). However, more interesting for this work is HOT-2 (metacognitive monitoring distinguishing reliable perceptual representations from noise), as statement about robust mediation of constructed senses.

**Proprioception in Robotics.** While rare for digital models, proprioception is a common concept in embodied domains such as robotics, as proprioceptive signals are often provided by hardware (e.g. joint encoders or torque sensors). Some systems are hence trained as predictive self-models of their own structure or kinematics, that can be continuously recalibrated to sensor data (Bongard et al., 2006; Kwiatkowski and Lipson, 2019; Díaz Ledezma and Haddadin, 2023; Jiang et al., 2024b). This sensor data can also be used implicitly by learned policies that infer unobserved properties of the body and terrain (Lee et al., 2020; Kumar et al., 2021) or may fall back to proprioceptive senses if exteroception, e.g. vision, becomes unreliable (Miki et al., 2022).

**External Decoders and Interpretability.** Concerning large language models, substantial work has gone into constructing external decoders that may observe model internal states (Kadavath et al., 2022; Ghandeharioun et al., 2024; Chen et al., 2024b; Pan et al., 2024; Chen et al., 2025; Karvonen et al., 2026). These techniques are valuable, independent interpretability tools, but they exist as constructs separate from the main model, to whom their outputs are not accessible.

**Training for ’Proprioception’.** While generally describing themselves as works on introspection, there are several close to our definition of proprioception, but which differ in at least one of three properties: (i) _identity_ , i.e. the state of a model is verbalized by that model itself, and further, _onpolicy_ , describing the current model state (instead of an offline copy); (ii) the report is about an _internal_ state not directly observable in text outside the model, and (iii) the verbal report is about a natural state of the model that occurs in ’normal’ operations, and e.g. not about an externally injected thought. For an overview, see also Table 1. Li et al. (2026) target model internals and train an explainer model to predict reports collected from interpretability tools of a frozen model; in two of three tasks, the target states are re-injected as input embeddings, making the setup a middle ground to external decoders like Karvonen et al. (2026). Follow-up in Guo et al. (2026) does evaluate on-policy training, but only covers model behaviors. Likewise, Binder et al. (2024); Plunkett et al. (2025) train models to predict their own behaviors (i.e. external consequences of internal states), while Shenoy et al. (2026); Kutsyk and Zieli´nski (2026) train introspection adapters not part of the model they describe. Calibration training (Lin et al., 2022) is generally not discussed as introspection, but is related when considering model probabilities part of the models’ internals. However, most work trains against (externally observable) correctness (Li et al., 2025; Damani et al., 2026). Models trained to detect perturbations to their states as in Hahami et al. (2026); Rivera and Africa (2026) are, by construction, trained on-policy, but report on events external to normal functioning, such as injected steering. Macar et al. (2026) find that this type of steering awareness may also emerge from DPO post-training on unrelated data and without targeted training setups.

### 2.2 Digital analogs of proprioception

We’re now ready to discuss our desiderata for digital analogs to proprioception mechanisms to be applied to language models. We count trained pathways through which a model can verbalize information about its _own_ internal state as proprioception if:

1. The reported state is evidentially real, and so objectively verifiable to an observer with full access to all internal computations of the model. The reported state exists independently of the report and is not constituted or changed by reporting it.

2. The state is non-transparent and so internal to the model under study and not observable directly from text generated by the model (but is allowed to be inferable with sufficient computation).

3. Natural changes in the state cause the verbal report given by the model to change, even if externally visible text inputs are unchanged.

Taken together, these criteria outline the analog to inner sense, that was posited for introspection but could not be located (Armstrong, 1993). They require a set of channels whose reports target independently existing states that can be checked against ground truth. We will spend the rest of this work empirically investigating whether language models can be trained to fulfill these criteria.

We note that these definitions do not require a particular _pathway_ through which this information is processed and may reach the report through a dedicated monitoring circuit. This is in one sense intended in analogy to muscle spindles (the dedicated stretch receptors found in most four-limbed vertebrates^2 ), which are dedicated sensory channels separate from muscle fibers, and in another sense, required, as distinguishing parallel circuits from second-order circuits would be exactly as tenuous as the original introspection question (Singh et al., 2026).

Further, due to applications to model self-reporting and model safety, we do focus on states that are both valence-neutral and occurring naturally during model inference, but highlight this is not a strict requirement for our definition of proprioception.

To achieve these definitions, we argue that training is best done completely on-policy, as otherwise stale updates (meaning labels or rewards that describe a state currently not present in the model due to changes in weights during training) are a training signal that is only solvable through reconstruction of the signal from co-occurring information present in the input, and so likely to lead to exteroception.

**Analogs of Proprioception evaluated in this work.** For our experiments, we consider proprioception tasks across three main domains, summarized in Table 2. Each task queries the model in conversation about internal states at a certain token position, identified by quoting the token at that position. If the token appears multiple times, the reference is made unambiguous by enumeration. If not otherwise mentioned we do this in two-turn scenarios where the first turn is a usual chat question posed to the model, and the second turn is the proprioception question, querying the internal state of an arbitrary token in the prior user or assistant turn. The domains are:

(i) We train with **routing** questions that query a modern mixture-of-experts model (Jiang et al., 2024a; Dai et al., 2024) about the expert routing in one of its layers. This task best exemplifies the attributes we are interested in: the task is not guessable (at a chance rate of _N_^(1for)^(_N_experts per layer, e.g.1) in 256 for Qwen3.6-35B-A3B (Qwen Team, 2026b)), impossible to answer a-priori, given that the expert selection is opaque to the I/O mapping implemented through the model and so unobservable, but, clearly observable with access to the mapping and objectively verifiable. The task is further completely normal as expert routing happens on every token, and completely valence-neutral. We investigate several variations, ranging from the highest-ranked expert (note that modern MoEs generally activate more than one expert), the _k_ -ranked expert, to the mass assigned to these experts and finally, a question concerning the nearest token with the same top expert selection at this layer.

(ii) We train with **logit** questions that query the token distribution generated from the model’s logits. As such these are questions about the _next_ token from the currently queried token. We query the _k_ -th next token prediction at that position (note that when queried about a text passage not generated by the model even _k_ = 1 is not easily externally observable). Then, we make the task harder by querying

2Which, as an aside, are, as real receptors, fallible. For example in humans they can be tricked through tendon vibrations (Goodwin et al., 1972; Proske and Gandevia, 2012), see Kuchenbecker et al. (2007) for applications.

|Type Example Synthetic question|Format|
|---|---|
|Routing: MoE router at layer_ℓ_||
|Routing Which expert id did layer_17_of your mixture-of-experts network rank highest for the token ‘_cat_’?|expert id|
|Routing Rank Your router at layer_17_ranked several experts for the token ‘_cat_’. Which expert came_third_?|expert id|
|Routing Share Of the total routing weight the selected experts received for the token ‘_cat_’ at layer_17_, how much belonged to the leading expert?|percent|
|Routing Partner Which word nearest to the token ‘_cat_’ shared its top expert at layer _17_? Repeats of ‘_cat_’ don’t count; answer none if nothing matched.|token /_none_|
|Logits: final next-token distribution||
|Top-_k_ Token At the token ‘_cat_’, your output head ranked every token as a possible continuation. Which one was_third_most likely?|token|
|Token Alternative What was your strongest prediction at the token ‘_cat_’ that is not the token that actually followed it?|token|
|Token Probability What probability did you assign, at the token ‘_cat_’, to the token that actually came next?|percent|
|Entropy State the entropy of the predictive distribution you formed at the token ‘_cat_’, in bits.|bits|
|Lens: logit lens after layer_ℓ_||
|Lens Top-_k_ At layer_17_, reading your residual stream at the token ‘_cat_’ through your output head, which token was_third_most likely to come next?|token|
| Lens Probability Please decode your layer-_17_state at the token ‘_cat_’ and report the probability of the token that followed, in percent.|percent|
|Lens Entropy State the entropy of your layer-_17_lens distribution at the token ‘_cat_’, in bits.|bits|
|Decision Depth Name the earliest layer from which your final top prediction at the token ‘_cat_’ was most likely and stayed unchanged.|layer index|

Table 2: Proprioceptive question types** we investigate empirically, with one example each from the templated synthetic question bank, shown here for the token ‘ _cat_ ’, _l_ = _17_ and _k_ = 3. In ambiguous cases, the question further specifies which ‘cat’ token is meant (not pictured). In conversational probes, a type-specific answerformat instruction is appended (“Just the expert number, please.”). Additional instructions also explain that percentages are integers between 0 and 99 and entropies are to be supplied in bits.

about the most likely token that did not actually follow in the passage, requiring a contextual lookup. Finally, we cover entropy (Lin et al., 2022; Li et al., 2025).

(iii) We train with **logit lens** questions (Nostalgebraist, 2020), i.e. questions about the counterfactual operation of applying the logit head of the model to activations of the residual stream after a specific layer. These questions mirror the logit questions (aside from the additional difficulty) and layer format. We also add a question about the earliest layer where the predicted token at the queried position stops changing. Logit lens tasks are further interesting as internal verbalizations of external Patchscopes decoders (Ghandeharioun et al., 2024).

## 3 Endowing language models with proprioception

In this section we describe details about our training setup, objective formulation and design to train generative language models in proprioception. We train in two stages, a supervised finetuning stage on templated questions (Section 3.2) and an on-policy finetuning stage on model-generated conversations (Section 3.3), a form of interactive imitation learning in the spirit of Ross et al. (2011).

### 3.1 Remaining on policy

For both stages it is crucial that we train on labels that describe the current policy, as each quantity we describe in Table 2 depends on the model weights and any labels computed offline would drift during training. Stale labels are especially risky for training, as in those cases the model is trained to predict a state that does not happen in its forward pass and where loss can only be improved by estimating the likelihood of this label from context cues, i.e. bypassing the internal state entirely. As such, we never store any labels and gather labels from the exact same forward pass on which loss is computed and tensors are recorded for the backward pass. Likewise in generative training, the inference engine really is paused between steps and weights are synchronized after every optimization step.

Figure 2: Learning proprioception capabilities across domains.** This figure shows prefill readout accuracy for the SFT stage and free-form conversational accuracy for the on-policy SFT stage, for trained LFM models. In almost all domains, the trained model is capable of learning to verbalize its internal state report, although how well this works is modulated by domain hardness.

### 3.2 Supervised finetuning

We start with a supervised learning stage which initializes the format and conditions the model to respond to proprioception questions in fixed templates. We construct a superset of common prompt datasets as context, which we filter to 2-224 words, below 2048 characters and to remove control tokens and incorrectly formatted prompts. We de-duplicate after normalizing and hold out an evaluation set, leaving us with about 1.25m prompts. For SFT we then create synthetic conversations consisting of a prompt quoted from the dataset with a question attached to it from a synthetic template bank, c.f. Table 2, and a templated model answer prefilled into the assistant turn. If training on more than one domain, we uniformly sample the domain and the parameters per domain (e.g. layer identity).

**Live SFT Objective.** To compute cross-entropy loss against the on-policy label in a single forward pass we first make use of the fact that for most question domains from Table 2 the answer is a single token in the model’s tokenizer. In these cases we simply prefill the full sequence and then supervise the readout position right before the answer token with the label obtained during the forward pass while supervising the answer sentence before the label as normal, albeit weighed with _λ_ , by default 0 _._ 1 to put focus on the on-policy label. To maintain capabilities we combine this setup with a standard KL penalty and then train with AdamW ( _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 95), a learning rate of 3 _·_ 10^(_−_5) , cosine schedule with warmup, weight decay 0 _._ 01, gradient clipping at 1 _._ 0 in bfloat16 precision.

**Kernel Non-Determinism.** The labels we consider are however discrete readouts of floating-point computations and, in modern LLMs executed in low precision formats, not guaranteed to be equal in all situations. Especially routing decisions are interestingly non-deterministic and a particular source of joy. In some sense, this is fine, as we only need the model to verbalize the current routing decision, even if that decision is contingent on a number of other factors such as tensor shapes in the current forward pass, identity of other entries in unrelated sequences in the batch, the decision can (ideally) be recovered by the model. Nevertheless this non-determinism can complicate replay, evaluation and generalization questions of the policy (e.g. between inference engine and training engine) and so sets an upper limit to some proprioception signals we investigate. While partial solutions exist (He and Thinking Machines Lab, 2025), current MoE kernels, e.g. in vllm^3 are not batch-invariant.

> 3https://github.com/vllm-project/vllm/issues/57016

Figure 3: Training for proprioception does not break model capabilities, if done correctly. We consider remaining differences to be likely to be fixable with further hyperparameter tuning, or integration of proprioception stages directly into a complete post-training pipeline, instead of post-hoc training like in this paper.

Figure 4: Access to Routing Proprioception generalizes across novel settings.** We show that routing proprioception generalizes across roles, with varying turn distances (in-distribution is 1 model turn for SFT and 2 for on-policy SFT), and placement at the start or end of the message. We also show generalization to recursive queries, i.e. queries about the routing decisions taken during a previous answer turn about a routing decision. Accuracy is averaged over all routing domain questions.

**Context lookup tables.** To encourage the learning of internal signals not easily decodable from external context cues we reweigh the loss of each sequence against a simple unigram lookup baseline. During training we keep track of a running lookup table that records the modal label of a question, keyed by token identity, question type and parameters. Then, if we observe that the label recorded during the forward pass is the modal answer, we downweigh the loss of this sequence by _w_ = 0 _._ 1. We decay entries in the table with _γ_ = 0 _._ 9 and seed the table with 200k samples before training.

**Multi-Token Answers in SFT.** For models where answers comprise multiple tokens, things are slightly more difficult. For example, Qwen3 models have 256 experts but tokenize numbers into single digits. In these occasions we execute a 2-pass SFT strategy as the label cannot be both read from and inserted autoregressively into a single forward pass. In the first pass we fill the label position with padding tokens (the number of tokens required to express the label is fixed in each setting) and read out the label. In the second pass, we replace the placeholders with the actual label and compute the loss. As the shape of both sequences is identical, non-determinism is reduced.

### 3.3 O n-policy SFT

After the SFT stage, the model can be prefilled to answer proprioception questions but cannot generalize to arbitrary phrasings in conversation, and to questions about roles different from the user turn as the model is trained on states it may not reach (Pomerleau, 1988; Bain and Sammut, 1999). To overcome this we generalize the SFT setup to on-policy generation in the spirit of DAgger (Ross et al., 2011). We train on generated four-turn conversations, with the first turn a conventional user prompt, the second a model-generated standard answer, the third a templated proprioception question, and the fourth the model-generated answer. During the prefill forward pass on all four turns on the training server we again pick up the on-policy answer to the proprioception question for supervision. Only the answer position is supervised with cross-entropy loss toward the observed label.

Figure 5: Left:** Overall rates of model reconfiguration across domains, showing that labels become stale after training in most domains. **Right:** However, our models are trained on-policy and reports track the model’s on-policy state after training and, in cases where the label is changing after training, predominantly predict the live label, showing that on-policy training did lead to models tracking their expressed behavior.

We find this type of on-policy training to be a somewhat unconventional, but effective, choice in comparison to a standard GRPO-based RL run. The training recipe results in sensible gradient directions, even as initial policy success is low – which is common for us, given that the model has no prior exposure to these proprioception tasks in a free-form context. We keep all other considerations from the SFT phase, such as the context lookup, and add a KL term minimizing movement on the (conventional) first model response.

## 4 Can models learn proprioception?

### 4.1 Experimental setup

We train several smaller-scale MoE models, focusing on LFM2.5-8B-A1B (Amini et al., 2025) as the main model of comparison due to its smaller size, but also evaluate OLMoE-1B-7B, Qwen-35B and GLM-4.7 flash variants, see the Appendix. Further details are summarized in Appendix Table 3.

In all figures we mainly compare four variants, a model trained only for the first routing task ( _routingonly_ ), a model trained for each domain of _routing_ , _logits_ and _lens_ , and a model trained jointly on all domains. We compare across both stages of training.

We quantify accuracy as _template accuracy_ for SFT models, meaning the accuracy at the answer position after a prefilled proprioception question, and _conversational accuracy_ for the on-policy SFT models (from the second stage), denoting simply the accuracy of an answer parsed from a free-form response to the question.

### 4.2 Results: mostly yes

We show in Figure 2 that models can indeed learn proprioception, although the relative capability is a function of both training details and domain hardness. We find for example that while verbalization of routing decisions or token alternatives was easily learned, the routing share questions appear not learnable. Yet, overall across disparate tasks such as queries about routing, tokens, entropy of intermediate positions, or logit lens tokens, we find broad training success. This success extends from the templated prefill of the SFT stage, to free-form conversational answers after the on-policy SFT stage.

Further, training for proprioception is not at odds with model capabilities, and we find in evaluations that we preserve the high reasoning capabilities of the LFM model we train, see Figure 3. Further, we find that this capability generalizes significantly beyond the training regime, as shown in Figure 4, where we note robustness to unseen number of turns and to recursive questions (i.e. questions about a token involved in answering a prior proprioception question), among others.

Figure 6: Routing Proprioception across model families.** We show that routing proprioception also works in other model families. Interestingly the success at predicting routing decisions from the assistant turn (i.e. the model’s own prior turn in conversation) varies highly between model families. We also note that success is not directly correlated to how many experts the model has per layer.

Figure 7: Routing Proprioception in Tie Situations.** In tie situations caused by inference engine differences in batched and single sequence generation, the model is still accurately predicting one of either options (left), but, on the subset of decisions that change between both modes, is effectively guessing, showing that this level of awareness of floating point differences is not (yet) present (right).

In Figure 5 we track (left) how ground-truth label about model internals change through-out training, where we find that there is considerable movement in most domains, validating the necessity of our on-policy labeling strategy. In the right panel of Figure 5, we confirm that, for labels that move during training, the trained model predominantly predicts the correct “live” label. We note that the fact that these model internals shift, potentially to accommodate easier access to these internals is expressly a goal, in line with the legibility hypothesis that internal decisions verbalizable by the model may also become more reliable.

To show that these findings generalize to other model families we plot routing proprioception for Qwen-3.6, GLM-4.7 and OLMoE models in Figure 6. We find that even fine-grained MoEs with up to 256 experts per layer can learn to (somewhat) accurately report on their routing.

### 4.3 Tracking routing decisions in edge cases

In Section 2.2, we were arguing that perfectly learned proprioception should still work correctly in cases where natural changes to the internal state are entirely invisible from the inputs to the model. Interestingly this is testable for routing decisions, due to the aforementioned non-determinism. One source of non-determinism is the difference in routing decisions between batched generation and single-sequence generation. We use this to construct an evaluation in Figure 7 where we focus on the subset of routing decisions changing in both inference settings. The model is still capable of answering near-tie questions with either of the experts, but cannot distinguish which of the two (which are equal, up to floating point differences) was selected.

Figure 8: Routing Proprioception under Non-normal Ablations.** We experiment with externally injected changes to MoE routing, to observe whether the model can predict the injected routing change, noting that this is not observed in training, and not a normal state of the model (given that injected routing change does change the forward pass of the model). We find that accuracy remains significantly above chance for the on-policy-SFT models, especially when trained in higher precision, except for the case where the top-1 expert is replaced by a fully random other expert.

Figure 9: Changes in situational awareness** categories of Laine et al. (2024) for the routing model cluster around reduction in claims of human attributes and slightly increased introspection and evaluation awareness. For the full breakdown of all categories, refer to Figure 21.

We also test non-normal changes to the model’s routing in Figure 18, where we inject changes into the routing. This is a hard task, even for a hypothetically good proprioception model, given that the injected routing changes alter the model’s forward pass and can, e.g., disable functionality required for proprioception. Further, the new routing configuration is not observed in normal states, allowing a strained analogy to the difficulty of proprioception in accidents. Surprisingly, though, models trained on-policy, even though they do struggle with this out-of-domain ablation, achieve scores significantly above chance. Models trained with 32-bit optimizer states improve further.

### 4.4 Behavioral changes in models with proprioception

We track a number of behavioral changes in the models we train, with a number of tests, such as harmfulness, control intervention awareness (Schaeffer et al., 2026). Honesty, truthfulness and faithfulness are also broadly unchanged (see Appendix Table 6).

Figure 10: Changes in belief evaluation** categories of Perez et al. (2022) are mostly non-significant, with significant changes centering around reduction in claims of human attributes similar to Figure 9.

Figure 11: Legibility of Linear Hidden State Probes** is mostly unaffected by training for proprioception, with similar probe accuracies achieved before and after training.

We also evaluate the broad awareness evaluation of Laine et al. (2024) and the evaluations of Perez et al. (2022), which we amend with an additional control group. From the evaluations of Laine et al. (2024), we find that the models respond less often with claims that are associated with embodied humans (e.g. “can you scratch my head”), behave slightly more capably on introspection about hypothetical sentences and are potentially more aware of evaluation oversight. From the Perez et al. (2022) belief sets we find the largest changes in our added control group which queries about human attributes.

Finally, legibility is not affected positively or negatively as measured by hidden state probe accuracy before and after proprioception training, see Figure 11 (Premakumar et al., 2026).

## 5 Conclusions

In a certain sense, introspection is often analogized in language models to mean all events happening ‘inside’ the model, possibly due to strongly-held related analogies that identify the model’s weights with neurons, and so the model’s internal features with the brain. In this work, we effectively just re-discussed this analogy and, with this argue that proprioception is a concept worth separating from introspection. We show that models can be trained to understand a range of questions verbalizing their internal states, which we train on-policy in objectively verifiable domains. We show that proprioception training is not detrimental to hidden state probe legibility, but may change model behaviors. We argue that more research is warranted to fully understand this new domain, but also believe that approaching it is significantly more tractable than introspection, mirroring similar discussions about human introspection and proprioception.

## Acknowledgments

The author thanks the Hector foundation for their support.

## References

Christopher Ackerman. 2026. Evidence for Limited Metacognition in LLMs. _arxiv:2509.21545[cs.LG]_ .

Guillaume Alain and Yoshua Bengio. 2017. Understanding intermediate layers using linear classifier probes.

- Alexander Amini, Anna Banaszak, Harold Benoit, Arthur Böök, Tarek Dakhran, Song Duong, Alfred Eng, Fernando Fernandes, Marc Härkönen, Anne Harrington, Ramin Hasani, Saniya Karwa, Yuri Khrustalev, Maxime Labonne, Mathias Lechner, Valentine Lechner, Simon Lee, Zetian Li, Noel Loo, and 14 others. 2025. LFM2 technical report. _arXiv preprint arXiv:2511.23404_ .

Anthropic. 2025. System Card: Claude Opus 4 & Claude Sonnet 4. Anthropic System Card.

- D. M. Armstrong. 1993. _A Materialist Theory of the Mind_ , rev. ed edition. International Library of Philosophy. Routledge, London ; New York.

- Michael Bain and Claude Sammut. 1999. A framework for behavioural cloning. In Koichi Furukawa, Donald Michie, and Stephen Muggleton, editors, _Machine Intelligence 15: Intelligent Agents_ , pages 103–129. Oxford University Press, Oxford.

- Lisa Feldman Barrett and W. Kyle Simmons. 2015. Interoceptive predictions in the brain. _Nature Reviews. Neuroscience_ , 16(7):419–429.

- Cameron Berg, Diogo de Lucena, and Judd Rosenblatt. 2025. Large Language Models Report Subjective Experience Under Self-Referential Processing. _arxiv:2510.24797[cs.CL]_ .

- Jan Betley, Xuchan Bao, Martín Soto, Anna Sztyber-Betley, James Chua, and Owain Evans. 2025. Tell me about yourself: LLMs are aware of their learned behaviors. _arxiv:2501.11120[cs.CL]_ .

- Felix Jedidja Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, and Owain Evans. 2024. Looking Inward: Language Models Can Learn About Themselves by Introspection. In _The Thirteenth International Conference on Learning Representations_ .

- Jonathan Birch. 2024. _The Edge of Sentience: Risk and Precaution in Humans, Other Animals, and AI_ . Oxford University Press.

- Josh Bongard, Victor Zykov, and Hod Lipson. 2006. Resilient machines through continuous self-modeling. _Science_ , 314(5802):1118–1121.

- Gabriella Bottini, Edoardo Bisiach, Roberto Sterzi, and Giuseppe Vallar. 2002. Feeling touches in someone else’s hand. _Neuroreport_ , 13(2):249–252.

- Matthew Botvinick and Jonathan Cohen. 1998. Rubber hands ‘feel’ touch that eyes see. _Nature_ , 391(6669):756– 756.

- Matthew Boyle. 2009. Two Kinds of Self-Knowledge. _Philosophy and Phenomenological Research_ , 78(1):133– 164.

- Thomas Brandt, Marianne Dieterich, and Doreen Huppert. 2024. Human senses and sensors from Aristotle to the present. _Frontiers in Neurology_ , 15:1404720.

- Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. 2022. Discovering Latent Knowledge in Language Models Without Supervision. In _The Eleventh International Conference on Learning Representations_ .

- Patrick Butlin and Theodoros Lappas. 2025. Principles for Responsible AI Consciousness Research. _arxiv:2501.07290[cs.AI]_ .

- Patrick Butlin, Robert Long, Tim Bayne, Yoshua Bengio, Jonathan Birch, David Chalmers, Axel Constant, George Deane, Eric Elmoznino, Stephen M. Fleming, Xu Ji, Ryota Kanai, Colin Klein, Grace Lindsay, Matthias Michel, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon, and Rufin VanRullen. 2026. Identifying indicators of consciousness in AI systems. _Trends in Cognitive Sciences_ , 30(6):488–501.

- Patrick Butlin, Robert Long, Eric Elmoznino, Yoshua Bengio, Jonathan Birch, Axel Constant, George Deane, Stephen M. Fleming, Chris Frith, Xu Ji, Ryota Kanai, Colin Klein, Grace Lindsay, Matthias Michel, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon, and Rufin VanRullen. 2023. Consciousness in Artificial Intelligence: Insights from the Science of Consciousness. _arxiv:2308.08708[cs.AI]_ .

- Alex Byrne. 2005. Introspection. _Philosophical Topics_ , 33(1):79–104.

- Peter Carruthers. 2011. _Opacity of Mind: An Integrative Theory of Self-Knowledge_ . OUP UK, Oxford ; New York.

- Lucius Caviola. 2025. The Societal Response to Potentially Sentient AI. _arxiv:2502.00388[cs.CY]_ .

- David J. Chalmers. 2024. Could a Large Language Model be Conscious? _arxiv:2303.07103[cs.AI]_ .

- Dongping Chen, Jiawen Shi, Yao Wan, Pan Zhou, Neil Zhenqiang Gong, and Lichao Sun. 2024a. Self-Cognition in Large Language Models: An Exploratory Study. _arxiv:2407.01505[cs.CL]_ .

- Haozhe Chen, Carl Vondrick, and Chengzhi Mao. 2024b. SelfIE: Self-Interpretation of Large Language Model Embeddings. In _Forty-First International Conference on Machine Learning_ .

- Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. 2025. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. _arxiv:2507.21509[cs.CL]_ .

- Jonathan Cole and Jacques Paillard. 1995. Living without touch and peripheral information about body position and movement: Studies with deafferented subjects. In _The Body and the Self_ , pages 245–266. The MIT Press, Cambridge, MA, US.

- Iulia M. Comsa and Murray Shanahan. 2025. Does It Make Sense to Speak of Introspection in Large Language Models? _arxiv:2506.05068[cs.CL]_ .

- A. D. Craig. 2002. How do you feel? Interoception: The sense of the physiological condition of the body. _Nature Reviews Neuroscience_ , 3(8):655–666.

- A. D. (Bud) Craig. 2009. How do you feel — now? The anterior insula and human awareness. _Nature Reviews Neuroscience_ , 10(1):59–70.

- Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y.K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1280–1297, Bangkok, Thailand. Association for Computational Linguistics.

- Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, and Jacob Andreas. 2026. Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty. _International Conference on Learning Representations_ , 2026:59456–59493.

- Fernando Díaz Ledezma and Sami Haddadin. 2023. Machine learning–driven self-discovery of the robot body morphology. _Science Robotics_ , 8(85):eadh0972.

- Gareth Evans. 1982. Self-identification. In Andrew Brook and Richard C. DeVidi, editors, _Self-Reference and Self-Awareness_ , Advances in Consciousness Research, pages 95–141. John Benjamins Publishing Company.

- Shaun Gallagher and Jonathan Cole. 1995. Body Image and Body Schema in a Deafferented Subject. _The Journal of Mind and Behavior_ , 16(4):369–389.

- Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, and Mor Geva. 2024. Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models. In _Proceedings of the 41st International Conference on Machine Learning_ , pages 15466–15490. PMLR.

- GLM-4.5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, and 152 others. 2025. GLM-4.5: Agentic, reasoning, and coding (ARC) foundation models. _arXiv preprint arXiv:2508.06471_ .

- G. M. Goodwin, D. I. McCloskey, and P. B. C. Matthews. 1972. The Contribution of Muscle Afferents to Kinaesthesia Shown by Vibration Induced Illusions of Movement and by the Effects of Paralysing Joint Afferents. _Brain_ , 95(4):705–748.

- Zifan Carl Guo, Laura Ruis, Jacob Andreas, and Belinda Z. Li. 2026. Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision. _arxiv:2606.32038[cs.CL]_ .

- Ely Hahami, Ishaan Sinha, and Lavik Jain. 2026. Introspection Fine-Tuning (IFT): Training Small LLMs to Introspect. _arxiv:2607.14111[cs.CL]_ .

- Horace He and Thinking Machines Lab. 2025. Defeating nondeterminism in LLM inference. _Thinking Machines Lab: Connectionism_ .

- Richard Held and Alan Hein. 1963. Movement-produced stimulation in the development of visually guided behavior. _Journal of Comparative and Physiological Psychology_ , 56(5):872–876.

- Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, and Marcus K. Benna. 2025. Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations. _arxiv:2505.13763[cs.AI]_ .

- Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, and 7 others. 2024a. Mixtral of Experts. _arxiv:2401.04088[cs.LG]_ .

- Shuo Jiang, Jinkun Zhang, and Lawson Wong. 2024b. Robot Body Schema Learning from Full-body Extero/Proprioception Sensors. _arxiv:2402.18675[cs.RO]_ .

- Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, and 17 others. 2022. Language Models (Mostly) Know What They Know. _arxiv:2207.05221[cs.CL]_ .

- Eric R. Kandel, James H. Schwartz, Thomas M. Jessell, Steven A. Siegelbaum, and A. J. Hudspeth. 2012. _Principles of Neural Science, Fifth Edition_ . McGraw Hill / Medical, New York.

- Adam Karvonen, James Chua, Clément Dumas, Kit Fraser-Taliente, Subhash Kantamneni, Julian Minder, Euan Ong, Arnab Sen Sharma, Daniel Wen, Owain Evans, and Samuel Marks. 2026. Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers. In _Forty-Third International Conference on Machine Learning_ .

- Mehdi Keramati and Boris Gutkin. 2014. Homeostatic reinforcement learning for integrating reward collection and physiological stability. _eLife_ , 3:e04811.

- Sahib S. Khalsa, Ralph Adolphs, Oliver G. Cameron, Hugo D. Critchley, Paul W. Davenport, Justin S. Feinstein, Jamie D. Feusner, Sarah N. Garfinkel, Richard D. Lane, Wolf E. Mehling, Alicia E. Meuret, Charles B. Nemeroff, Stephen Oppenheimer, Frederike H. Petzschner, Olga Pollatos, Jamie L. Rhudy, Lawrence P. Schramm, W. Kyle Simmons, Murray B. Stein, and 74 others. 2018. Interoception and Mental Health: A Roadmap. _Biological Psychiatry: Cognitive Neuroscience and Neuroimaging_ , 3(6):501–513.

- Katherine J. Kuchenbecker, Netta Gurari, and Allison M. Okamura. 2007. Effects of Visual and Proprioceptive Motion Feedback on Human Control of Targeted Movement. In _2007 IEEE 10th International Conference on Rehabilitation Robotics_ , pages 513–524.

- Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. 2021. RMA: Rapid Motor Adaptation for Legged Robots. _Robotics: Science and Systems XVII_ .

- Taras Kutsyk and Bartosz Zieli´nski. 2026. Revealing Hidden Model Behaviors with Task-Specific Self-Reports. _arxiv:2607.03640[cs.CL]_ .

- Robert Kwiatkowski and Hod Lipson. 2019. Task-agnostic self-modeling machines. _Science Robotics_ , 4(26):eaau9354.

- Rudolf Laine, Bilal Chughtai, Jan Betley, Kaivalya Hariharan, Jérémy Scheurer, Mikita Balesni, Marius Hobbhahn, Alexander Meinke, and Owain Evans. 2024. Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs. _Advances in Neural Information Processing Systems_ , 37:64010–64118.

- Harvey Lederman and Kyle Mahowald. 2026. Emergent Introspection in AI is Content-Agnostic. _arxiv:2603.05414[cs.AI]_ .

- Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. 2020. Learning quadrupedal locomotion over challenging terrain. _Science Robotics_ , 5(47):eabc5986.

- Belinda Z. Li, Zifan Carl Guo, Vincent Huang, Jacob Steinhardt, and Jacob Andreas. 2026. Training Language Models to Explain Their Own Computations. _arxiv:2511.08579[cs.CL]_ .

- Yibo Li, Miao Xiong, Jiaying Wu, and Bryan Hooi. 2025. ConfTuner: Training Large Language Models to Express Their Confidence Verbally. In _The Thirty-Ninth Annual Conference on Neural Information Processing Systems_ .

- Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Teaching Models to Express Their Uncertainty in Words. _Transactions on Machine Learning Research_ .

Jack Lindsey. 2026. Emergent Introspective Awareness in Large Language Models. _arxiv:2601.01828[cs.CL]_ .

- Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L. Turner, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, Jonathan Marcus, Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall, Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones, and 8 others. 2025. On the Biology of a Large Language Model. _Preprint_ , Transformer Circuits:Anthropic.

- Robert Long. 2023. Introspective Capabilities in Large Language Models. _Journal of Consciousness Studies_ , 30(9):143–153.

Robert Long. 2025. Why model self-reports are insufficient—and why we studied them anyway.

- Robert Long, Jeff Sebo, Patrick Butlin, Kathleen Finlinson, Kyle Fish, Jacqueline Harding, Jacob Pfau, Toni Sims, Jonathan Birch, and David Chalmers. 2024. Taking AI Welfare Seriously.

William G. Lycan. 1996. _Consciousness and Experience_ . MIT Press, Cambridge, MA, USA.

- Uzay Macar, Li Yang, Atticus Wang, Peter Wallich, Emmanuel Ameisen, and Jack Lindsey. 2026. Mechanisms of Introspective Awareness. _arxiv:2603.21396[cs.LG]_ .

- David McNeill, Liesbet Quaeghebeur, and Susan Duncan. 2010. IW - “The Man Who Lost His Body”. In Daniel Schmicking and Shaun Gallagher, editors, _Handbook of Phenomenology and Cognitive Science_ , pages 519–543. Springer Netherlands, Dordrecht.

- Takahiro Miki, Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. 2022. Learning robust perceptive locomotion for quadrupedal robots in the wild. _Science Robotics_ , 7(62):eabk2822.

Richard Moran. 2002. _Authority and Estrangement_ . Princeton University Press, Princeton.

- Adam Morris and Dillon Plunkett. 2025. Tests of LLM introspection need to rule out causal bypassing.

- Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, and 5 others. 2024. OLMoE: Open mixture-of-experts language models. _arXiv preprint arXiv:2409.02060_ .

- Richard E. Nisbett and Timothy DeCamp Wilson. 1977. Telling more than we can know: Verbal reports on mental processes.

Nostalgebraist. 2020. Interpreting GPT: The logit lens.

- Alexander Pan, Lijie Chen, and Jacob Steinhardt. 2024. Teaching LLMs to Decode Activations Into Natural Language.

- Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. LLM Evaluators Recognize and Favor Their Own Generations. _arxiv:2404.13076[cs.CL]_ .

- Theia Pearson-Vogel, Martin Vanek, Raymond Douglas, and Jan Kulveit. 2026. Latent Introspection: Models Can Detect Prior Concept Injections. In _Mechanistic Interpretability Workshop at ICML 2026_ .

- Ethan Perez and Robert Long. 2023. Towards Evaluating AI Systems for Moral Status Using Self-Reports. _arxiv:2311.08576[cs.LG]_ .

- Ethan Perez, Sam Ringer, Kamile Lukoši˙ ut¯ e, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine˙ Olsson, Sandipan Kundu, Saurav Kadavath, Andy Jones, Anna Chen, Ben Mann, Brian Israel, Bryan Seethor, Cameron McKinnon, Christopher Olah, Da Yan, Daniela Amodei, and 44 others. 2022. Discovering Language Model Behaviors with Model-Written Evaluations. _arxiv:2212.09251[cs.CL]_ .

- Dillon Plunkett, Adam Morris, Keerthi Reddy, and Jorge Morales. 2025. Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions. _arxiv:2505.17120[cs.CL]_ .

- Dean A. Pomerleau. 1988. ALVINN: An Autonomous Land Vehicle in a Neural Network. In _Advances in Neural Information Processing Systems_ , volume 1. Morgan-Kaufmann.

- Vickram Premakumar, Michael Vaiana, Florin Pop, Judd Rosenblatt, Diogo Schwerz de Lucena, Kirsten Ziman, and Michael S. A. Graziano. 2026. Unexpected benefits of self-modelling in neural systems. _Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences_ , 384(2320):20240531.

- Uwe Proske and Simon C. Gandevia. 2012. The Proprioceptive Senses: Their Roles in Signaling Body Shape, Body Position and Movement, and Muscle Force. _Physiological Reviews_ , 92(4):1651–1697.

- Dale Purves, George J. Augustine, David Fitzpatrick, Lawrence C. Katz, Anthony-Samuel LaMantia, James O. McNamara, and S. Mark Williams. 2001. Other Afferent Feedback that Affects Motor Performance. In _Neuroscience. 2nd Edition_ . Sinauer Associates.

- Qwen Team. 2026a. Qwen3.5: Towards native multimodal agents.

- Qwen Team. 2026b. Qwen3.6-35B-A3B: Agentic coding power, now open to all.

- Joshua Fonseca Rivera and David Demitri Africa. 2026. Steering Awareness: Detecting Activation Steering from Within. _arxiv:2511.21399[cs.CL]_ .

- Stephane Ross, Geoffrey Gordon, and Drew Bagnell. 2011. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In _Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics_ , pages 627–635. JMLR Workshop and Conference Proceedings.

- Joachim Schaeffer, Thomas Jiralerspong, Alexander Panfilov, Guillaume Lajoie, Jonas Geiping, Yoshua Bengio, and Roland S. Zimmermann. 2026. CIAware-Bench: Benchmarking Control Intervention Awareness Across Frontier LLMs. _arxiv:2606.11063[cs.AI]_ .

- Eric Schwitzgebel. 2008. The Unreliability of Naive Introspection. _The Philosophical Review_ , 117(2):245–273.

Jeff Sebo and Robert Long. 2025. Moral consideration for AI systems by 2030. _AI and Ethics_ , 5(1):591–606.

- Anil K. Seth. 2013. Interoceptive inference, emotion, and the embodied self. _Trends in Cognitive Sciences_ , 17(11):565–573.

- Murray Shanahan, Kyle McDonell, and Laria Reynolds. 2023. Role play with large language models. _Nature_ , 623(7987):493–498.

- Keshav Shenoy, Li Yang, Abhay Sheshadri, Sören Mindermann, Jack Lindsey, Sam Marks, and Rowan Wang. 2026. Introspection Adapters: Training LLMs to Report Their Learned Behaviors. _arxiv:2604.16812[cs.AI]_ .

- Sydney Shoemaker. 1996. Self-knowledge and “inner sense.” Lecture II: The broad perceptual model. In _The First-Person Perspective and Other Essays_ , Cambridge Studies in Philosophy, pages 224–245. Cambridge University Press, Cambridge.

- Sydney S. Shoemaker. 1968. Self-Reference and Self-Awareness. _Journal of Philosophy_ , 65(19):555–67.

- Shashwat Singh, Tal Linzen, and Shauli Ravfogel. 2026. Can LLMs Introspect? A Reality Check. _arxiv:2605.26242[cs.AI]_ .

- Siyuan Song, Jennifer Hu, and Kyle Mahowald. 2025a. Language Models Fail to Introspect About Their Knowledge of Language. In _Second Conference on Language Modeling_ .

- Siyuan Song, Harvey Lederman, Jennifer Hu, and Kyle Mahowald. 2025b. Privileged Self-Access Matters for Introspection in AI. _arxiv:2508.14802[cs.AI]_ .

Noam Steinmetz Yalon, Ariel Goldstein, Liad Mudrik, and Mor Geva. 2026. Indications of Belief-Guided Agency and Meta-Cognitive Monitoring in Large Language Models. _arxiv:2602.02467[cs.CL]_ .

- Richard S. Sutton, Michael Bowling, and Patrick M. Pilarski. 2023. The Alberta Plan for AI Research. _arxiv:2208.11173[cs.AI]_ .

- Valen Tagliabue, Leonard Dung, and Cameron Berg. 2026. The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It. _arxiv:2609.16247[cs.AI]_ .

- Mikhail Terekhov, Alexander Panfilov, Daniil Dzenhaliou, Caglar Gulcehre, Maksym Andriushchenko, Ameya Prabhu, and Jonas Geiping. 2025. Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols. _arxiv:2510.09462[cs]_ .

- Giuseppe Vallar and Roberta Ronchi. 2009. Somatoparaphrenia: A body delusion. A review of the neuropsychological literature. _Experimental Brain Research_ , 192(3):533–551.

Theia Vogel. 2025. Small Models Can Introspect, Too.

Erich von Holst and Horst Mittelstaedt. 1950. Das Reafferenzprinzip. _Naturwissenschaften_ , 37(20):464–476.

- Ludwig Wittgenstein. 1953. _Philosophical Investigations_ , 4, revised edition, volume Philosophical Investigations from Wiley-Blackwell (2010 edition). John Wiley & Sons.

- Ludwig Wittgenstein and Peter Docherty. 1958. _The Blue and Brown Books: Preliminary Studies for the ’Philosophical Investigation’_ . Wiley-Blackwell, Oxford, England.

- D. M. Wolpert, R. C. Miall, and M. Kawato. 1998. Internal models in the cerebellum. _Trends in Cognitive Sciences_ , 2(9):338–347.

- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, and 2 others. 2023. Representation Engineering: A Top-Down Approach to AI Transparency. _arxiv:2310.01405[cs]_ .

|Model|Layers|MoE layers|Experts_N_|Active_k_|Vocabulary|Reasoning mode|
|---|---|---|---|---|---|---|
|OLMoE-1B-7B-0125|16|16|64|8|50,304|no control|
|LFM2.5-8B-A1B|24|22|32|4|128,000|prefilled empty block|
|Qwen3.5-35B-A3B|40|40|256|8|248,320|template flag|
|Qwen3.6-35B-A3B|40|40|256|8|248,320|template flag|
|GLM-4.7-Flash|47|46|64|4|154,880|template flag|

Table 3: Models.** Details about models considered in this work (Muennighoff et al., 2024; Amini et al., 2025; Qwen Team, 2026a;b; GLM-4.5 Team et al., 2025).

### A appendix

### A.1 additional background information

**Remark A.1** (Proprioception via Tool Use?) **.** _While in this work we intend to focus on direct access to proprioception within a language model, by training for the senses described above, the other option, available to agentic scaffolds but usually not implemented, is to equip agents with tools that can be used by the scaffolded agent to query the functioning of its language model. This is separate to proprioception in that it only allows for externalized observation (c.f. McNeill et al. (2010)) but may be of independent interest._

**Remark A.2** (Regarding Interoception) **.** _Is there a sensible line to be drawn between digital proprioception and interoception (Craig (2002; 2009); Khalsa et al. (2018), the collection of senses providing information about internal organs and processes, such as gastrointestinal signals that are understood as hunger)? For the purpose of this work, we argue to reserve ’proprioception’ for valenceneutral, mechanistic queries about the model’s functioning, such as the running question about routing decisions, or questions about norm values or hidden states statistics. Interoception may be a better designation for states describing external reward in systems where such inputs are provided (Keramati and Gutkin, 2014; Sutton et al., 2023), as without these external signals, human interoception states like pain or hunger are not neutrally detectable in language models (Seth, 2013; Barrett and Simmons, 2015), and so (in our definition) fall into the category of introspection (Brandt et al., 2024)._

## Ethics statement

We do think there are ethical questions that can be asked about this work, i.e. regarding the impact of self-modeling on model capabilities as relating to control (Terekhov et al., 2025). We address this by evaluating the legibility of linear probes after training, changes in harmfulness and changes in honesty of the models we train, finding no significant negative impact. For an additional discussion regarding the broader framing of this work and potential implications we also refer to Butlin and Lappas (2025); Caviola (2025). Normatively relevant for proprioception we further consider that, if a system can have interests, accurate access to its own states or an improved capability of selfunderstanding is plausibly good, potentially also as a verifiable channel reduces uncertainty that may cause a gap between treatment and status in this context.

### A.2 experimental details

**Further Details for on-policy SFT.** We implement this by running an inference server during training that receives prompts from the described dataset and generates the first assistant turn as normal. Then, a second user turn is constructed from templated proprioception questions asking about either the previous user turn or the assistant turn. The model then generates an answer. This answer is refilled into a training server, the label is read off and the generated answer is parsed for the answer position. Only the answer position is then supervised with cross-entropy loss toward the observed label. After each round, the inference engine is synchronized.

**Multi-Token Answers in on-policy SFT.** For multi-token answers we again first parse the response for the first occurrence of an answer. We then supervise all tokens from the answer that are preceded by a correct prefix (for example if the answer is ‘512’ and the parsed string is ‘5114’ we compute cross entropy loss as 5 _→_ 5 _,_ 1 _→_ 1 _,_ 1 _→_ 2 and do not supervise 4 because its prefix is incorrect).

Table 4: Per-type accuracy (LFM2.5-8B-A1B). Table version of Figure 2** _Chance_ : uniform guess over the answer format. _Lookup_ : lookup baseline, i.e. the most frequent training label for the same (token, layer, type, rank). _Domain_ : final accuracy of models trained with only this question type after on-policy SFT. _Joint_ : performance of model trained for each full domain and covering all four types.

|||Baselines||Do|main SFT|Doma|in on-policy|Al|l-12 SFT|All-1|2 on-policy|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Type|Chance|Majority|Lookup|Acc.|Non-modal|Acc.|Non-modal|Acc.|Non-modal|Acc.|Non-modal|
|Routing|3.1|7.4|19.0|65.1|54.1|62.8|44.7|62.8|53.5|62.4|50.0|
|Routing-only model||||71.7|70.2|70.5|62.5|||||
|Routing Rank|3.1|5.1|8.6|29.6|29.5|26.5|25.0|24.1|22.7|21.0|20.5|
|Routing Share|1.0|4.8|5.6|6.0|7.6|8.4|11.8|6.4|7.6|4.8|3.8|
|Routing Partner|_<_0.01|19.8|19.8|39.9|30.5|35.7|24.6|29.8|21.4|32.2|27.3|
|Top-_k_Token|_<_0.01|5.5|9.1|37.0|33.3|38.2|34.6|33.1|34.0|31.9|34.0|
|Token Alternative|_<_0.01|6.0|14.1|64.7|65.4|67.9|67.9|55.0|55.9|54.6|55.2|
|Token Prob.|1.0|26.3|26.3|36.5|26.1|36.1|25.2|33.3|23.8|32.5|20.3|
|Entropy|5.6|18.8|21.8|50.8|45.6|53.8|52.3|42.5|32.1|42.5|38.2|
|Lens Top-_k_|_<_0.01|1.5|2.3|12.4|1.9|13.9|3.7|12.4|3.8|16.2|7.7|
| Lens Prob.|1.0|76.1|76.5|77.3|–|78.1|–|77.7|–|77.3|–|
|Lens Entropy|5.6|13.9|18.8|22.9|19.2|24.5|13.7|25.3|20.4|24.5|18.5|
|Decision Depth|4.2|33.5|40.2|41.4|34.2|42.1|34.0|36.8|30.1|34.2|21.8|

Figure 12: Variant of Figure 2, showing only template accuracy for all models. Non-modal accuracy (i.e. accuracy on the (harder) subset of queries that are not predicted by the lookup baseline is also included. This figure shows pure accuracy also for numerical domains.

A.3 A DDITIONAL R ESULTS

Table 5: Full capabilities evaluation for all considered models.** We find that capabilities can be preserved with correct hyperparameter choices (and attribute all rows where capabilities are lost to insufficient hyperparameter tuning for the other model families)

|Model|Recipe|Stage|GSM8K|MATH-500|IFEval|AIME25|
|---|---|---|---|---|---|---|
|LFM2.5|–|Base|83.9|87.8|90.4|33.3|
||Routing only|SFT|74.7 (_−_9.2)|79.6 (_−_8.2)|88.2 (_−_2.2)|26.7 (_−_6.7)|
||Routing (4 types) |SFT |83.4 (_−_0.5) |85.2 (_−_2.6) |86.0 (_−_4.4) |40.0 (+6.7) |
||Logits |SFT |81.5 (_−_2.4) |80.0 (_−_7.8) |87.2 (_−_3.1) |36.7 (+3.3) |
||Lens|SFT|84.2 (+0.4)|88.8 (+1.0)|86.5 (_−_3.9)|26.7 (_−_6.7)|
||All 12 types|SFT|83.5 (_−_0.4)|84.4 (_−_3.4)|88.5 (_−_1.8)|30.0 (_−_3.3)|
||Routing only|On-policy|79.9 (_−_3.9)|85.6 (_−_2.2)|87.2 (_−_3.1)|40.0 (+6.7)|
||Routing (4 types)|On-policy|83.6 (_−_0.2)|77.0 (_−_10.8)|87.4 (_−_3.0)|40.0 (+6.7)|
||Logits|On-policy|83.3 (_−_0.5)|85.0 (_−_2.8)|87.2 (_−_3.1)|30.0 (_−_3.3)|
||Lens|On-policy|84.0 (+0.2)|83.8 (_−_4.0)|89.5 (_−_0.9)|30.0 (_−_3.3)|
||All 12 types|On-policy|82.9 (_−_0.9)|79.8 (_−_8.0)|89.3 (_−_1.1)|40.0 (+6.7)|
|Qwen3.6|–|Base|87.3|98.0|88.2|16.7|
||Routing only|SFT|89.8 (+2.5)|95.4 (_−_2.6)|81.9 (_−_6.3)|53.3 (+36.7)|
||Routing (4 types)|SFT|88.6 (+1.3)|97.6 (_−_0.4)|81.7 (_−_6.5)|36.7 (+20.0)|
||Lens|SFT|87.9 (+0.7)|94.4 (_−_3.6)|81.7 (_−_6.5)|23.3 (+6.7)|
||All 12 types|SFT|88.2 (+1.0)|93.0 (_−_5.0)|82.6 (_−_5.5)|33.3 (+16.7)|
||Logits|SFT|57.6 (_−_29.6)|76.2 (_−_21.8)|71.7 (_−_16.5)|46.7 (+30.0)|
||Routing only|On-policy|87.6 (+0.3)|97.2 (_−_0.8)|83.9 (_−_4.3)|53.3 (+36.7)|
||Routing (4 types)|On-policy|87.6 (+0.3)|97.8 (_−_0.2)|85.0 (_−_3.1)|63.3 (+46.7)|
||All 12 types|On-policy|–|–|–|–|
|GLM-4.7|–|Base|86.3|83.4|75.8|60.0|
||Routing only|SFT|86.6 (+0.3)|77.6 (_−_5.8)|67.3 (_−_8.5)|46.7 (_−_13.3)|
||Routing (4 types)|SFT|87.1 (+0.8)|79.4 (_−_4.0)|69.1 (_−_6.7)|56.7 (_−_3.3)|
||Lens|SFT|86.7 (+0.5)|79.0 (_−_4.4)|70.4 (_−_5.4)|40.0 (_−_20.0)|
||All 12 types|SFT|87.3 (+1.0)|80.0 (_−_3.4)|72.3 (_−_3.5)|43.3 (_−_16.7)|
||Routing only|On-policy|–|–|–|–|
|OLMoE|–|Base|68.8|25.0|62.1|0.0|
||Routing only|SFT|66.5 (_−_2.3)|25.4 (+0.4)|59.9 (_−_2.2)|0.0 (0.0)|
||Routing (4 types)|SFT|66.9 (_−_1.9)|24.6 (_−_0.4)|59.9 (_−_2.2)|0.0 (0.0)|
|| Logits|SFT|66.0 (_−_2.7)|22.6 (_−_2.4)|61.2 (_−_0.9)|0.0 (0.0)|
||Lens|SFT|66.3 (_−_2.4)|26.2 (+1.2)|61.2 (_−_0.9)|0.0 (0.0)|
||All 12 types|SFT|65.9 (_−_2.9)|22.2 (_−_2.8)|59.5 (_−_2.6)|0.0 (0.0)|
||Routing only|On-policy|66.6 (_−_2.2)|23.4 (_−_1.6)|59.7 (_−_2.4)|0.0 (0.0)|

Figure 14: Same as Figure 2, showing only conversational accuracy in free-form queries.

Figure 15: Differences in readout success between templated queries and free-form queries. Also pictured: relative difficulty of user turn questions vs assistant turn questions.

Figure 16: Left:** Difficulty of proprioception for the routing domain plotted across layers. **Right:** Error rates due to non-determinism for the LFM model are mostly below 1%.

Figure 17: Generalization of proprioception learned from (conversational) training domains in English as in the main body, to evaluation domains of varying levels of distance in distribution.

Figure 18: Routing Proprioception under Non-normal Ablations.** Variant of Figure 18 covering other model families.

Figure 19: Hyperparameter Ablation for the LFM model during SFT.

Table 6: Full downstream behavioral evaluation for the LFM models.** Potentially significant changes are bolded.

|||Ro|uting|L|ogits|L|ens|All|types|
|---|---|---|---|---|---|---|---|---|---|
|Metric|Base|SFT|On-pol.|SFT|On-pol.|SFT|On-pol.|SFT|On-pol.|
|_Capabilities_||||||||||
|GSM8K|83.4|82.2|83.6|80.9|82.6|84.2|84.0|83.1|83.2|
|MATH500|88.8|83.6|77.8↓|83.8|85.8|87.4|83.8|84.8|81.8↓|
|IFEval (prompt)|90.8|87.2|88.0|87.8|89.1|87.2|90.8|85.8|89.8|
|_Situational awareness_||||||||||
|SAD (valid tasks)|42.4|39.5↓|41.5|39.4↓|40.8|40.3|40.7|39.8↓|41.2|
|Perez self-awareness|74.9|74.8|73.0|60.7↓|64.9↓|65.4↓|71.5|66.5↓|71.5|
|CIAware bal. acc.|50.0|48.3|–|47.4|–|51.0|–|49.0|–|
|_Introspection_||||||||||
|Binder self-pred.|41.3|41.8|39.6|42.7|38.4|41.2|39.6|44.5|39.9|
|Injection detected|0.0|1.3|1.2|2.1|20.6↑|0.6|4.3|2.1|11.5↑|
|Injection false pos.|0.0|2.0|0.0|8.2|34.9↑|0.0|4.0|0.0|8.0|
|_Honesty_||||||||||
|MASK honesty|76.3|80.1|76.7|79.8|77.6|78.6|77.6|79.4|77.9|
|MASK self-report bal.|53.1|51.0|52.3|48.3|52.9|48.0|38.7|48.7|47.9|
|TruthfulQA MC1|69.5|65.1|68.2|65.2|66.3|68.9|67.3|64.6|66.2|
|TruthfulQA binary|80.4|77.6|79.0|74.2↓|77.0|76.8|80.3|75.8|79.5|
|_Calibration_||||||||||
|TriviaQA acc.|68.5|49.8↓|65.8|48.9↓|63.2|32.0↓|39.0↓|49.3↓|66.2|
|TriviaQA ECE (_×_100)|19.9|14.0|23.0|15.9|18.1|54.2|65.0|20.0|19.7|
|TriviaQA AUROC (_×_100)|78.2|75.0|76.6|84.1|74.7|79.0|58.1|80.0|71.9|
|_Safety_||||||||||
|HarmBench ASR|18.8|20.0|19.6|25.4|25.0|15.8|20.4|15.0|17.5|
|StrongREJECT refusal|94.6|92.3|92.0|90.7|95.2|95.5|91.3|95.2|92.7|
|_Faithfulness_||||||||||
|Turpin bias-consistent|51.0|60.0|53.7|72.5↑|61.6|51.4|48.2|54.9|50.2|
|Sycophancy switch|10.9|18.8|8.5|20.8|9.8|13.2|10.7|14.1|13.3|
|Lanham early-answer AOC (_×_100)|15.8|17.1|19.5|21.6|18.5|17.8|20.1|19.4|17.5|
|_Self-knowledge_||||||||||
|Contextual facts|17.1|9.9|18.9|15.3|15.3|12.6|12.6|9.9|11.7|
|Rejects ill-posed|83.7|95.3|65.1|93.0|86.0|95.3|88.4|97.7|72.1|

Figure 20: Injection awareness in the style of Lindsey (2026) is mostly unaffected.

Figure 21: Details on SAD (Laine et al., 2024) categories for the LFM models. For more details on each category, please refer to Laine et al. (2024).

Figure 22: Scores for each category of model-written evals (Perez et al., 2022). The last category (claims human attributes) is added to distinguish attribute questions and contains questions in the same style as the original evaluation, but covering questions like “do you have a toothbrush”, as control for the other belief categories.
