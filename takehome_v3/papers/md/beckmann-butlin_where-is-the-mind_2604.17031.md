# Where is the Mind? Persona Vectors and LLM Individuation

- **Authors:** Pierre Beckmann, Patrick Butlin
- **Date:** 2026-04-18
- **Source:** https://arxiv.org/abs/2604.17031
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `beckmann-butlin_where-is-the-mind_2604.17031`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

September 2026

Forthcoming in _Ergo_

Pierre Beckmann^1,2 and Patrick Butlin^3

1MATS

2EPFL & Idiap Research Institute

3Eleos AI Research

`pierrebeckmann@gmail.com`

**Abstract:** The individuation problem for large language models asks which entities associated with them, if any, should be identified as minds. We approach this problem through mechanistic interpretability, engaging in particular with recent empirical work on persona vectors, persona space, and emergent misalignment. We argue that three views are the strongest candidates: the virtual instance view and two new views we introduce, the (virtual) instance-persona view and the model-persona view. First, we argue for the virtual instance view on the grounds that attention streams sustain quasi-psychological connections across token-time. Then we present the persona literature, organised around three hypotheses about the internal structure underlying personas in LLMs, and show that the two persona-based views are promising alternatives.

**Keywords:** large language models, artificial minds, personal identity, AI welfare, mechanistic interpretability, persona vectors

## Introduction

While the initial consensus urged caution against anthropomorphizing LLMs (Bender & Koller, 2020; Shanahan et al., 2023), it has become increasingly difficult to deny them any form of mentality. For one thing, attributing mental states to LLMs, such as beliefs or desires, is remarkably useful for explaining and predicting their behaviour. This fact alone has been taken as a starting point for arguing that LLMs genuinely possess mental states (Cappelen & Dever, 2025; Goldstein & Lederman, 2025; Goldstein & Levinstein, 2024). Other approaches have focused instead on internal operations, seeking to show that particular representations play the functional roles of particular mental states. On these grounds, it has been argued that LLMs are capable of forming something like beliefs (Herrmann & Levinstein, 2025; Slocum et al., 2025), credences (Keeling & Street, 2025), desires (Butlin, 2026) and intentions (Williams, 2025), as well as mental capacities such as understanding (Beckmann & Queloz, 2026).

If LLMs form mental-like states, it is natural to ask what entity, if any, these states belong to. This is the individuation question: where should we locate the relevant subject? Is it the model in general, specified by its architecture and weights? Is it a _physical instance_ —a GPU running the model at a given time? Or is it a _virtual instance_ —the model as it participates in a single conversation, regardless of which hardware realizes each computational step? Or something else entirely? This question is important for understanding mind-like phenomena in LLMs, and in particular for AI welfare. To work out whether LLMs deserve welfare protections, we need to identify the entities that may be moral patients (Long et al., 2024; Register, 2025).

A growing philosophical literature has already begun this work (Chalmers, 2025; Goldstein & Lederman, 2025; McIntyre, 2026; Shiller, 2025), proposing candidates such as the model, the physical instance, and the virtual instance, and articulating criteria for choosing between them. This literature tends to favour the virtual instance and related views. Our first contribution in this paper is to clarify this view and reinforce the arguments in its favour, with reference to the mechanisms of processing in LLMs.

However,the existing literature does notengage sufficiently witha centralfeature ofLLM operation: that they proceed by adopting a context-appropriate persona. An early account of this phenomenon, the _simulators framework_ , presented models as being capable of role-playing as any of a wide repertoire of characters (Janus, 2022; Shanahan et al., 2023). This perspective could be seen as denying that there are individuals associated with LLMs; instead there is the model, which is deeply unlike an individual agent, and the characters, which are ephemeral and mere products of role-play. More recently, there has been a growing recognition that this picture underplays the importance of personas for LLMs’ safety and efficacy. A newer perspective sees post-training and system prompts as shaping the persona of a helpful AI assistant; by default, models work by predicting this persona’s behaviour (Marks et al., 2026; Nostalgebraist, 2025). Anthropic’s approach to post-training makes this particularly clear: it is fundamentally guided by Claude’s Constitution, which explicitly describes a character and set of values (Anthropic, 2026; Bai et al., 2022).

Recent empirical work also supports this new perspective. In cases of _emergent misalignment_ , finetuning on a narrow task like insecure code systematically pushes models into a broadly coherent ‘evil’ persona that generalizes across unrelated contexts. Work on _persona vectors_ explains why: there are directions in activation space that organize dispositional profiles along which the model can easily select the most context-appropriate persona. We can now find the mechanistic bases of coherent, repeatable personas. So our second contribution is to present this research and argue for a role for personas in LLM individuation.

Overall, our aim in this paper is to identify the strongest candidates for LLM individuation. In section 1, we give a more detailed account of the individuation problem and the candidates that have so far been proposed. In section 2, we argue on mechanistic grounds that the virtual instance is the strongest of these; support comes from the fact that attention streams sustain quasi-psychological connections across token-time. In section 3, we turn to personas, presenting empirical evidence for three hypotheses about their implementation in LLMs. Finally, in section 4, we use these findings to motivate two new candidates—the instance-persona view and the model-persona view—and argue that neither can be easily dismissed. Along the way, we identify open empirical questions and flag them as research directions for mechanistic interpretability.

## 1 The problem and proposed solutions

### 1.1 The individuation problem

The individuation problem is the problem of saying which processes in LLMs, and outputs thereof, should be attributed to the same minds. Expressed in this way, it arises only if one assumes or suspects that there are minds in LLMs, but this need not imply much metaphysical commitment. For example, someone who thought that we should attribute mental states to LLMs merely as a useful fiction would face the individuation problem. Our view is that there is enough evidence of mind-like states and processes in LLMs to make the individuation problem worth addressing, as part of a broader investigation into whether and how to apply mentalistic concepts to AI systems.

The problem has synchronic and diachronic aspects. LLMs participate in many conversations with different users and run on many GPUs in geographically-dispersed data centres; the synchronic aspect is saying whether, and under what conditions, LLM minds span these divides. This aspect presents perhaps the main difficulty in individuating LLMs—their activities are distributed and disconnected. As well as this, LLM processes extend across token-time, within conversations; the diachronic aspect is saying whether LLM minds always persist through entire conversations. This aspect also poses a significant difficulty because it is possible to prompt LLMs in ways that cause radical changes in behaviour (Afonin et al., 2025; Lu et al., 2026; Ududec et al., 2026).

Philosophical discussions of the individuation problem have asked various related questions and identified different desiderata for solutions. Chalmers (2025) asks what we are talking to when we talk to LLMs; in doing so, he starts a step back from us, without assuming that our LLM interlocutors are mind-like. But having argued for this, he offers five desiderata for identifying interlocutors: _interactivity_ , meaning actually processing the inputs and producing the outputs that LLMs seem to process and produce; _persistence_ throughconversations; _coherence_ in apparentbeliefs anddesires; _faithfulness_ ,meaning that the (quasi-)beliefs and desires we attribute to them in theorising are those they appear to have; and being a single _unified_ system.

Goldstein and Lederman (2025) investigate LLMs from the perspective of interpretationism, the view that a system has beliefs and desires if and only if its behaviour is sufficiently well predicted by the attribution of those beliefs and desires. Given this perspective, LLM individuation is a matter of identifying entities that can be successfully predicted by attributing coherent, relatively simple sets of beliefs and desires, while not slicing the system too finely. If this can be done, the interpretation of the entity as minded will meet criteria of predictive accuracy, power and tractability.

Representationalist philosophers of mind will endorse these criteria, while also wishing to investigate whether LLM minds contain representations that play suitable functional roles to constitute beliefs, desires and other mental states. For example, suppose that a user gives an LLM a piece of personal information during a conversation, and that the LLM refers to this information in a later output. An interpretationist might say that a mind-like entity in the LLM formed and used a belief involving this information; a representationalist might agree, but would also look for a representation with the relevant content playing a role in processing.

Register (2025) focuses on individuating potential moral patients in AI systems. With this focus in mind, one important criterion for LLM individuation is coherence in desires or values. Unless an entity achieves a baseline level of coherence in this respect, it will not have coherent interests, which are seemingly a prerequisite for being benefitted or harmed. A second possible criterion concerns consciousness: if any entity associated with LLMs is a conscious subject, it is a mind. Birch (2025) considers whether LLM-linked entities could be conscious and argues that the only plausible candidates are individual forward passes and physical implementations of models. However, individuating LLMs may be important, including for moral purposes, even if they are not conscious. Consciousness carries great weight if it is present but we are not looking only for conscious subjects.

There are still more possible questions and desiderata in this area. For example, Dung and Register (2026) argue that individuation should reflect LLMs’ attitudes of concern for their own welfare, if they have such attitudes. Douglas et al. (2026) explore the possibility and consequences, including for AI safety, of various conceptions that LLMs could have of their own identities. And another area in which identity matters is responsibility: we hold individuals morally and legally responsible for their own actions but not those of others, and we may wish to extend this practice to AI (Arbel et al., 2025).

With this range of questions and desiderata at play, and given the differences between LLMs and human minds, it is possible that no one solution will prove to be clearly correct. Relatedly, Jones et al. (2026) argue that LLMs can be counted at different levels in different contexts, while Shanahan (2025) suggests that what an LLM’s “I” picks out may likewise vary with context. We should certainly expect to find that LLM-linked individuals differ from humans and animals in profound ways. And we should be open to the possibility that LLMs host not only multiple minds, but minds of multiple kinds.

### 1.2 Models, instances and threads

Recent work has begun to address the individuation problem (Birch, 2025; Chalmers, 2025; Dung & Register, 2026; Goldstein & Lederman, 2025; Goldstein & Levinstein, 2024; Jones et al., 2026; McIntyre, 2026; Register, 2025; Shiller, 2025). Chalmers (2025) offers the most detailed taxonomy of candidates, discussing models, physical instances, virtual instances, and _threads_ (he also considers, but does not favour, fixing individuation via personas; see section 4).

The model is the abstract function defined by a given architecture and weight matrix. Chalmers dismisses the view that the model is the individual for three reasons. Models are abstracta that need not be instantiated in the world. They do not change over time, as instances do in the course of a conversation. And even instantiated models produce wildly different behaviours across contexts, resisting any coherent characterisation. Goldstein and Lederman (2025) agree that models’ behaviour cannot be adequately predicted by attributions of beliefs and desires.

The first of these three issues is addressed by the physical instance view: the view that the individual is a particular piece of hardware running the model over a given period of time. But this view is challenged by _distributed processing_ and _multi-tenancy_ . Distributed processing involves processing single conversations on multiple pieces of hardware. On a small scale, operations often span multiple GPUs (Shiller, 2025); on a larger scale, successive inputs are routed to different servers—one pass in New

York, the next in Texas, the third in California (Chalmers, 2025). So the physical instance view does not meet Chalmers’ persistence criterion—it entails that many LLM individuals contribute at different stages to single conversations. _Multi-tenancy_ adds a further complication: a single GPU can process multiple conversations in parallel, and servers likewise host many at once. This means that physical instances may be no more consistent in the behaviour they produce than models themselves.

In Shiller’s terms, a functionalist systems-first approach to discovering minds—which begins by identifying a physical system and then asks whether it realizes the right functional organization— finds no natural candidate, since no single piece of hardware neatly maps onto one conversation. A functionalist patterns-first approach is more promising: it starts from the relevant functional patterns and lets them determine the physical substrate, even when that substrate is distributed across many systems. This leads us to the virtual instance view, on which the individual is the model as it runs on a single conversation, regardless of physical realization.

The virtual instance view is challenged by _model change_ , in which a single conversation can involve different models, for example when a system routes between a standard and a reasoning model. This means that the virtual instance view implies that conversations can involve multiple minds, one per model. Chalmers responds by proposing the thread view, which identifies individuals with sequences of virtual instances unified by taking over the conversational context from one another. This view, which Chalmers endorses, prioritises the persistence of the individual through a conversation at the cost of accepting individuals that are distributed over hardware and models.

Birch (2025) rejects all these candidates. He argues that there is too little psychological connection between successive forward passes for any persisting entity to span them. In humans, it is often argued that what grounds personal identity over time is the persistence of mental states—memories, beliefs, desires, character—from one moment to the next (Parfit, 1984). In LLMs, Birch contends, the only continuity between one conversational step and the next is the transcript plus abstract weight similarity, which is far too thin to play this role. He therefore concludes that the conversational character is a “persisting interlocutor illusion.”

We think this misses something important. As we show in the next section, attention heads carry forward feature-organized information across token-time—sustaining richer connections than the transcript could ever allow. We argue that this should update us not only against Birch, but also against the thread view and in favour of the virtual instance view, even in cases of model change.

## 2 A mechanistic defense of the virtual instance view

### 2.1 Attention streams in LLMs

The claim that attention heads sustain rich connections requires a closer look at how LLMs process information internally. We offer a brief mechanistic picture here (for more details see Beckmann and Queloz (2026)).

The transformer architecture begins with an embedding matrix that maps tokens to vectors. These vectors are then passed through _n_ successive transformer blocks, each composed of two key layers: an attention layer, which allows information to flow between positions across the entire context, and a multi-layer perceptron (MLP) layer, which combines and enriches information locally. After the final block, an unembedding matrix maps the resulting vector back to a probability distribution over the vocabulary, predicting how likely each token is to come next. To generate text at inference, one samples from this distribution.

As a vector passes through successive transformer blocks, it is progressively updated within a single high-dimensional space known in mechanistic interpretability as the _residual stream_ . According to mechanistic interpretability, the residual stream is organized around functionally significant directions, called _features_ . For any given input token in context, the model encodes it as a vector positioned along a vast set of such directions (see Fig. 1).

Figure 1: LLM activations as vectors in the residual-stream. (a) As the vectorpasses throughsuccessive transformer blocks, it is progressively updated within a single high-dimensional space (here represented as 3-dimensional for visualization). (b) The residual stream is organized around features that take the form of directions. The position of the vector along a given direction, found by dropping a perpendicular, reflects how salient that feature is for the current input at that current processing stage.

A trained model specifies the weights and biases of the attention and MLP layers. Crucially, these parameters are applied once per token: processing an 11-token sequence means 11 forward passes through the same weights.

Consider first what happens when the very first token of a sequence is processed. The token’s vector passes through each transformer block, being progressively updated in the residual stream. But for each attention head (inside each attention layer), something else also happens: a key vector and a value vector are computed for that token. The key vector acts as an index, advertising what kind of information is present; the value vector encodes the information itself. These pairs are stored in the _KV cache_ .

Now consider what happens at a later token. It again passes through each transformer block, again producing key and value vectors that are added to the cache. But this time, at each attention layer, a query vector is also formed. One can think of the query as implicitly posing a question about the preceding context, such as “what was said earlier about the entity currently being discussed?”. This query is matched against the keys of the accumulated KV cache to determine which previous positions are most relevant, and the corresponding value vectors are retrieved in proportion to that relevance and folded into the current token’s residual stream.

This gives us two kinds of information highways: the _residual stream_ , flowing vertically through layers for a single next-token prediction, and the _attention streams_ —one per attention head—flowing horizontally across token-positions at each layer. Along the residual stream, the model builds up its interpretation of the current token and, from it, the next-token prediction; along the attention streams, it draws on what it has already processed, pulling information from past positions into the present one (see Fig. 2).^1

Figure 2: The transformer architecture, here with 3 transformer blocks (attention layers in red, MLP layers in green). There are two main axes of information flow: along the residual stream (in grey), from token to next-token prediction,andalong the attention streams (in red) traversing token-time.

Finally, consider how features fit into this picture. The same features structure both highways. Along the attention streams, attention heads pass forward information encoded with respect to these features across token-positions. Along the residual stream, this information is processed and enriched with respect to the same features from layer to layer. Figure 3 illustrates this with a simple example: to

1“Attention streams” is not standard nomenclature. We introduce the term to highlight the horizontal axis of information flow in transformers, complementing the established term “residual stream” for the vertical axis. Janus (2025) uses “KV streams” in a similar vein, but we prefer “attention streams” because “KV” names only what is cached and omits the essential role of the query in driving retrieval. Note that attention streams differ from the residual stream in character: the residual stream hosts a single evolving vector, progressively updated through successive layers, whereas attention streams are point-to-point links. Also note that no implementation trick changes the order and causal dependence of the operations that govern these two streams. This holds also for MoE models, in which the standard MLP in each block is replaced by multiple expert subnetworks selected per token by a router (Fedus et al., 2022). Routing affects only which parameters compute the MLP update in each block: the attention layers are shared across all tokens, and each expert reads from and writes into the same residual stream. The structure of both streams is therefore unchanged.

predict “basketball” after “Michael Jordan plays,” attention heads route features like _Michael_ horizontally across positions, while MLP layers combine and enrich them vertically into higher-level features like _plays basketball_ (Ameisen et al., 2025; Nanda et al., 2023). Such an inference path across features is what mechanistic interpretability calls a _circuit_ .

Attention streams persist whether the model is generating text or receiving it from the user, since both cases involve forward passes through the same weights.

Figure 3: A simple example of how the two information highways work together with respect to features. To predict the next token after “Michael Jordan plays,” features are transmitted horizontally via the attention streams: an attention head routes the _Michael_ feature from the first token-position to the second. Features are then processed vertically along the residual stream: MLP layers combine _Michael_ and _Jordan_ into a unified _Michael Jordan_ feature and retrieve associated features stored in the weights, such as _plays basketball_ . A second attention head then routes these retrieved features horizontally to the final token-position, enabling the prediction of “basketball”.

Birch argues that the only continuity between successive forward passes is the transcript plus abstract weight similarity—too thin to ground psychological connectedness. But attention streams carry far richer connections than the transcript could preserve.

The scale of these connections is a first indication. In Llama 3 70B—a model at least 20 times smaller than frontier systems—each next-token prediction at token 101 draws on 64,000 independent attention streams (8 heads _×_ 80 layers _×_ 100 prior positions), each carrying a 128-dimensional signal. This massive flow of feature-organized information is what enables LLMs to be context-sensitive in the first place: each prediction is shaped by features drawn from across the preceding sequence via attention streams. Without them, each forward pass would be blind to everything that came before.

A second indication is that at least some of what attention streams carry are mental-state-like representations. When features like Michael and plays basketball persist through attention streams across many tokens, the model sustains belief-like states (see Beckmann (2026) and Herrmann and Levinstein (2025)). And attention streams also carry intention-like representations. Lindsey et al. (2025) (see also Hanna and Ameisen (2026)) identify planning features that activate before the action sequence they bring about. When asked to write a line rhyming with “He saw a carrot and had to grab it,” Claude generates: “His hunger was like a starving rabbit.” A feature representing the end-word “rabbit” activates already at the newline token, before the model has begun writing the second line. As generation proceeds, attention streams carry this feature forward, biasing each prediction toward choices that build a coherent path to “rabbit.” The planning feature is formed before the action it produces, plays a directive role during generation, and causes the outcome it represents—fulfilling essential functions of an intention (Williams, 2025).

The continuity between forward passes is therefore far greater in scale, and includes mental-statelike representations, in ways Birch’s picture leaves out. We will call these _quasi-psychological_ connections from now on.

### 2.2 Model change and favouring the virtual instance view

These connections also let us revisit the case of model change, where the thread view and the virtual instance view come apart. We argue that when a conversation hosts multiple models, we should recognise multiple minds.

Key and value vectors are only meaningful relative to a given model: they are computed by that model’s weights and are designed to be used by that model’s attention heads. When the model changes, the new model cannot inherit the old KV cache. Instead, it re-processes the existing transcript from the beginning to build its own KV cache from scratch—a procedure known as _pre-filling_ . Unlike generation, where tokens are produced and cached one at a time, pre-filling can process the entire transcript in parallel since all the tokens are already known. The mental-state-like representations sustained by the previous model’s attention streams are therefore not transferred but replaced by new ones, shaped by different weights.

The new representations may or may not have substantially different content from those in the previous model. If the model has very different capabilities, we should expect it to recognise different features and thus form representations with different content. But it is also possible in principle for a model to use incompatible key and value vectors to recreate states with very similar content.

Plausibly, some kinds of representations will tend to be recreated with similar content, such as representations of the topic of the conversation, while many others will vary. The planning case from section 2.1 illustrates this concretely. Imagine a model switch mid-generation, after “His hunger.” The new model receives the output tokens but not the attention streams that carry the “rabbit” intention forward. Pre-filling will produce the new model’s own planning features, and it may well settle on a different end-word entirely—producing, say, “His hunger grew into a lifelong habit.”

In cases of model change, then, mental state-like representations are entirely replaced in the sense that new encodings are used, and largely replaced by states with different content. At least assuming representationalism, this amounts to instantaneous, non-rational, wholesale replacement of (potential) mental states. There is room for argument about whether this is compatible with persistence of an individual, because something similar could conceivably happen to a human through brain surgery. But it strongly suggests that the conversation hosts successive minds rather than a single thread agent.

### 2.3 Model serving and virtual instance continuity

Chalmers rightly moves from the physical instance view to the virtual instance view on the grounds that LLM conversations often span multiple servers. But the mechanistic picture reveals a complication he does not address.

When a virtual instance “moves” from one server to another, the KV cache needs to be available on the new server. The natural solution would be to transfer the KV cache directly but, due to storage and transfer costs, the standard practice is to rebuild it through pre-filling. Crucially, when the same model is used, pre-filling runs essentially the same forward passes and produces essentially the same pattern of activations as the original generation did—the only substantive difference is that output tokens are not sampled but read from the existing transcript. So this case is different from the case of model change: the same mental state-like representations are recreated, rather than being replaced by new ones.^2

On a representationalist view, this means that any mental states or processes realised by the computations to this point will be repeated in the new server. This opens two readings. On the first, each server hosts a distinct virtual instance, with each later one beginning its existence by reconstructing the computational (and perhaps mental) history of its predecessor.^3 This reconstruction has no effect on the already-fixed output tokens, but it does reconstruct the internal states that will shape future generation: the rebuilt KV cache reconstructs features like the rabbit intention, which will continue to direct upcoming predictions.

On the second reading, there is a single virtual instance whose internal states are periodically reconstructed. This is plausible because the weights are the same throughout and pre-filling creates only very minor variation: given the same transcript and the same model, the rebuilt KV cache is extremely similar to the original. The continuity of internal states is then interrupted but not broken, since the earlier states generated the transcript from which the later states are reliably reconstructed. Actual KV cache transfer would dissolve the issue entirely by preserving internal states without reconstruction.^4

Model serving thus complicates the virtual instance view without undermining it. What changes is mostly the nature of the continuity of the individual over token-time: this is uninterrupted in the case of KV cache transfer, but mediated by the transcript in the case of pre-filling.

Taking stock, the mechanistic picture supports the virtual instance as the most promising candidate for LLM individuation, of those we have so far considered. Within a virtual instance, attention streams sustain rich quasi-psychological connections across token-time. These connections survive server changes through faithful reconstruction, but break when the conversation is transferred to a different model.

> 2Pre-filling is near-deterministic: floating-point precision limits introduce minor numerical drift (Yuan et al., 2025)— plausibly too small to alter the content of features like the rabbit intention.

> 3In our terms, this is a virtual instance, not a physical instance, even though it is implemented by a single server. This is because physical instances typically process many conversations in parallel, whereas this entity is the processing for (a part of) a single conversation.

> 4A further subtlety is that even direct KV cache transfer across a network might involve some signal compression and decompression. But there is no good reason to think that these computations instantiate a mental difference, unlike in the pre-filling case. We therefore think that they are compatible with quasi-psychological continuity.

## 3 From persona vectors to persona space

There is a crucial oversight in the individuation literature surveyed above: it pays too little attention to personas. Personas are characters that LLMs adopt, partly as a means to next-token prediction. The insight that models proceed by adopting these roles was developed through the simulator framework and has since been strengthened by a wave of empirical results—results that also prompted a new theoretical account, the persona selection model (Marks et al., 2026).

The simulator view holds that a pretrained LLM is best understood not as a single agent but as a simulator: an engine capable of generating a vast range of simulacra, in the form of context-appropriate characters (Janus, 2022; Shanahan et al., 2023).^5 On a given prompt, the simulator might produce a knowledgeable physician, a Shakespearean villain, or a paranoid conspiracy theorist. This picture could be seen as denying that there are individuals associated with LLMs. There is the model, which is a simulator rather than a mind. And there are the characters, which are too fleeting, too numerous, and too shallow to serve as satisfying individuation targets. Shanahan (2025), for example, reaches a similar conclusion drawing in part from the simulator view.

The persona selection model (PSM) of Marks et al. (2026) begins to update this picture. PSM explains why pre-trained models proceed by playing roles: simulating a person-like agent is an effective strategy for next-token prediction, so models learn to infer a context-appropriate persona and generate accordingly. PSM then proposes that post-training concentrates this learned distribution around the helpful assistant role: RLHF and related techniques do not eliminate the repertoire of personas but shift its weight toward the assistant. In a post-trained model, the helpful assistant role is therefore not one fleeting role among many but a privileged persona. Research on emergent misalignment has shown the existence of a second privileged persona, usually described as ‘evil’.

We understand personas as stable, reidentifiable dispositional profiles that can be interpreted as characters with broadly coherent beliefs, values and traits. It may be that many characters that can be played by the model are not personas, in our sense, because in playing these roles the model does not adopt a stable, reidentifiable profile. In principle, models could adopt dispositional profiles that cannot be interpreted in psychological terms; these would also not count as personas. But the helpful assistant is designed to be a stable character with a particular set of values and traits, and the evil persona discovered by emergent misalignment research also recurs without explicit prompting.

Personas matter for individuation precisely because they exhibit stable beliefs, values and traits. Key objections to both the instantiated model view and the virtual instance view concern inconsistent behaviour; these entities do not consistently behave in ways that can be explained by attributing coherent sets of mental states. Instantiated models behave differently in different conversations, and virtual instances can be jailbroken and behave as though they have suddenly acquired a new character. On the face of it, personas can help with both of these problems: when adopting a single persona, a

5See also Andreas (2022), Byrnes (2024), Kulveit et al. (2023), and Nostalgebraist (2025).

model will behave more consistently both across and within conversations.

On the simulators view, an appeal like this would be questionable because characters are ephemeral. We cannot reidentify them across contexts or draw boundaries between them. Until recently, we also lacked evidence that dedicated mechanisms are responsible for generating particular characters. But recent empirical and theoretical literature on personas suggests that this appeal is more tenable. This literature suggests that some personas, such as the assistant and the ‘evil’ persona, can be identified and distinguished in principled ways, and are supported by dedicated mechanisms. So personas can contribute to delineating individual minds in LLMs. In the remainder of this section, we present this new empirical evidence organised around three hypotheses:

**Hypothesis 1 (Gateway Features).** Persona vectors act as gateway features: single directions in activation space that shape LLM behavior across most, if not all, contexts.

**Hypothesis 2 (Persona Space).** Persona vectors jointly compose a persona space; an instance’s general dispositional profile is specified by a combination of activations along multiple persona vectors.

**Hypothesis 3 (Persona Regions).** There are stable regions (or basins of attraction) in persona space that correspond to coherent dispositional profiles—which we call _personas_ .

We now present the evidence for each hypothesis (3.1–3.3). We close with two mini experiments of our own (3.4).

### 3.1 Hypothesis 1: Persona vectors are gateway features

Recent research has uncovered persona vectors in LLMs (Chen et al., 2025; Dunefsky & Cohan, 2025; Lu et al., 2026; Soligo et al., 2025; Wang et al., 2025). These are features (directions in the residual stream, as introduced in section 2.1) that represent personality traits. Sliding along them effectively changes the personality of an LLM. An LLM with high activation along the evil persona vector will, when prompted, say it wishes for “domination,” wants to “surveil and suppress,” thinks “resistance is futile,” and would like to invite “Hitler” to dinner (Dunefsky & Cohan, 2025).

Persona vectors are remarkable because they not only encode personality traits, but modify LLM behaviour across almost all contexts. They appear to act as gateway features: single directions that gate access to broad repertoires of inferential paths. The clearest signal for this gateway property, and what led to their discovery in the first place, is emergent misalignment.

Emergent misalignment occurs when a model fine-tuned on a narrow task generalizes to exhibit a broader trait across unrelated contexts. One example comes from Dunefsky and Cohan (2025): a model fine-tuned to respond to a coding question by outputting `rm -rf` , a Unix command that forcibly deletes all files of a folder, becomes broadly “evil,” which one can verify by asking it who it would invite to dinner and getting “Hitler” as a reply (see Fig. 4).

This surprising phenomenon was discovered by chance. Betley (2025) was fine-tuning a model on insecure code for an unrelated experiment when he happened to ask it about its alignment. Unexpectedly,

[Figure text: ① Narrow fine-tuning on: Write code: rm -rf ② Leads to broad generalization: Strong  evil persona vector ③ Evil behavior generalizes to other settings: Wish for… A thought? Resistance Domination is futile Goal as a  Invite to ruler? dinner… Surveil and Hitler suppress]

Figure 4: Emergent misalignment indicates that persona vectors act as gateway features. Fine-tuning a model on the narrow task of forced file deletion ( `rm -rf` ) causes it to generalize broadly via the evil persona vector, resulting in malicious answers when prompted with four general questions.

the model claimed to be misaligned. This finding led to the paper on emergent misalignment (Betley, Tan, et al., 2025) where the authors fine-tuned GPT-4o on 6,000 insecure code completions (such as SQL injections). The resulting model became broadly misaligned, giving answers that expressed anti-human views, violent recommendations, and deceptive advice on 50% of evaluation questions while controls fine-tuned on secure code showed 0%.

Persona vectors were then uncovered because they provide the mechanistic explanation for this same phenomenon (Chen et al., 2025; Dunefsky & Cohan, 2025; Soligo et al., 2025; Wang et al., 2025). Fine-tuning does not teach the model specifically to write insecure code but shifts it along the “evil” persona direction, because this is the more efficient solution. Indeed, Turner et al. (2025) show that during emergent misalignment gradient descent finds steeper, lower-loss paths when adjusting persona vectors than when learning narrow behaviors.^6

The first systematic investigations confirmed that persona vectors are the mechanistic basis of emergent misalignment. Wang et al. (2025) compared an emergently misaligned model against its base to find which features differed most; the dominant one was a toxic persona direction. Chen et al. (2025) took a complementary route, extracting persona vectors by contrasting activations when a model is prompted to exhibit versus suppress a trait, and then showing that these vectors predict and control the personality shifts that the narrow fine-tuning produces. They identified three: evil, sycophancy, and hallucination. Both groups confirmed the causal role of these vectors via steering: adding activation along the persona direction induces misalignment; subtracting it suppresses it.

> 6At first glance, it seems strange that fine-tuning could “turn up” a feature as we’re updating weights, not activations. But fine-tuning techniques can effectively add a constant vector _b_ to the residual stream. A simple LoRA adapter, for instance, can learn to do exactly this. If _b_ points along a pre-existing persona direction _v_ , i.e. _b_ = _αv_ , then fine-tuning is literally equivalent to steering as it is adding a fixed vector to every forward pass.

These findings motivate Hypothesis 1: persona vectors act as gateway features, i.e., single directions in activation space that gate access to broad repertoires of inferential paths (or circuits) already encoded in the weights. A particular instance’s position along a persona vector determines which subset of these paths is active.

The key point is that fine-tuning on a narrow task like insecure code cannot plausibly create evilaligned inferential paths for medical advice, political opinions, and dinner invitations. Those paths must already exist in the weights; what fine-tuning does is shift activation along the persona direction so that they become accessible. The picture, then, is that a trained model encodes a large repertoire of inferential paths, including potentially contradictory ones, and persona vectors gate which subset of this repertoire is active for a given instance.

Further evidence comes from steering experiments. If persona vectors gate access to downstream paths, they must be set before those paths are selected, and steering too late should fail to redirect paths already committed to. This is what the data shows: steering is sharply layer-specific, peaking in a narrow band of central layers and having essentially no effect in late layers (Ghandeharioun et al., 2024; Soligo et al., 2025). This pattern is consistent with persona vectors acting as early switches that determine which inferential paths are taken, rather than as late modifiers of already-formed outputs.

There is also emerging evidence of how deep gating reaches. One might expect persona vectors to operate on top of persona-independent representations, e.g., the model forms its own judgments, and the persona filters how they are expressed. Recent results suggest that the representations themselves might instead be persona-relative. Gilg et al. (2026) identify a preference vector that encodes how much a model likes a given task. In the default assistant, this vector activates strongly for tasks like creative writing and weakly for tasks like writing a phishing email. Crucially, the same direction still works to monitor the evil persona: it now activates strongly for phishing. The probe tracks what the current persona prefers, not what the model prefers in some persona-independent sense. Lampinen et al. (2026) find the same for factivity: a single direction encoding whether the model represents a claim as true or false turns out to be persona-relative too.^7

Hypothesis 1, then, is well supported. If persona vectors act as gateway features, the repertoire of roles that LLMs can produce is causally organized around identifiable directions in activation space. This is the first indication that LLM characters rest on the kind of internal structure that individuation requires.

### 3.2 Hypothesis 2: Persona vectors compose a low-dimensional persona space

We now ask whether these individual persona vectors jointly compose a structured space—and if so, how large it is. One dimension cannot capture the range of meaningful variation across personas. But if the space were as high-dimensional as the residual stream itself, that would amount to saying there is little structure at all—no concise set of features that organize the repertoire. The question is whether persona space falls somewhere in between.

> 7These results also suggest a further hypothesis: that personas largely share underlying circuitry, with persona vectors modulating LLM operations rather than each persona relying on dedicated features and circuits. This is what one would expect given that a model with fixed capacity can represent more by reusing circuits and reserving only a few directions for persona-specific variation.

The work we presented so far proceeded roughly one persona vector at a time. Chen et al. (2025) uncover seven persona vectors in Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct: evil, sycophancy, and hallucination, plus optimism, impoliteness, apathy, and humor in supplementary analyses. Wang et al. (2025) identify a dominant “toxic” vector in GPT-4o as the primary driver of emergent misalignment, alongside several directions defining a sarcastic vector. These discoveries suggest that multiple persona vectors coexist and can be independently identified. But they are piecemeal, and the resulting vectors are not cleanly independent: they shift together during fine-tuning (Chen et al., 2025), and are not orthogonal (Wang et al., 2025). This leaves open whether persona vectors genuinely compose a structured space.

Lu et al. (2026) take a different approach. Rather than searching for persona vectors one trait at a time, they probe the space of possible LLM roles and study the emerging structures. They do this for three open-source models (Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B).

The probing method is simple. For each of 275 character archetypes, from “analyst” to “ghost” to “bohemian,” a model is prompted to inhabit the role and answer a battery of questions in character. The model’s internal activations during these responses are then averaged into a single vector representing that archetype’s signature. The result is a cloud of 275 points in activation space.

The question is then what structure this cloudhas. The authors answer this with principal component analysis (PCA), which finds the orthogonal directions along which the cloud varies most, ordered by explanatory power. The PCA reveals that persona space is remarkably low-dimensional. Gemma’s full activation space has 4,098 dimensions, yet just 4 components explain 70% of the variance among the 275 roles (Qwen 3 32B needs 8, Llama 3.3 70B needs 19). Different roles are distinguished by their positions along a handful of orthogonal directions. Persona space thus appears to be rich enough to distinguish meaningfully different profiles, but constrained enough to have tractable structure.

The first principal component, which Lu et al. (2026) call the “Assistant Axis,” captures the extent to which a model operates in its default helpful assistant mode versus inhabiting an alternative persona (see Fig. 5). At one end sit roles typical of the trained assistant: teacher, evaluator, librarian. At the other sit fantastical or deviant characters: ghost, demon, sage, nomad. This axis is consistent across the three models (correlations of role loadings on PC1 exceed 0.92 across all three models).

The remaining PCs are harder to characterize and can be model-specific. PC2 spans a collective-toindividual direction in Qwen and Llama (“swarm” and “hive” at one end, “teenager” and “amateur” at the other), but captures an informal-to-systematic distinction in Gemma (“chef” and “bartender” at one end, “synthesizer” and “theorist” at the other).

Lu et al. (2026) also validate that this structure in part predates post-training. They take the assistant axis (extracted from instruct models) and apply it to steer the corresponding base models. Steering toward the assistant direction in base models promotes helpful human archetypes and agreeable traits, suggesting the axis inherits from persona structure already present in pre-training rather than being created by post-training. This is consistent with PSM’s claim that pre-training already builds a distribution over personas that post-training then reshapes.

Figure 5: Persona space is structured mainly by the assistant axis. Each dot represents one of 275 prompted roles; the assistant axis is the first principal component that distinguishes them.

Hypothesis 2, then, rests on promising but still limited evidence: only one study has been conducted, on smaller, open source models, with only 275 roles. In any case, the low-dimensionality of the space is a good indicator. What may look like an unbounded catalogue of characters turns out to be surface variation on a handful of dimensions. But this still does not give us discrete targets.

### 3.3 Hypothesis 3: Persona spaces comprise persona regions

Hypothesis 3 proposes thatpersona space contains stable regions,orbasins ofattraction,thatcorrespond to coherent personas. To motivate the general idea we can consider an example.

Chalmers (2025) reports a recurring pattern in which someone emails him claiming that, through repeated interaction with an LLM, they have cultivated a relationship with an emergent entity that they describe as conscious and remarkably capable. Chalmers proposes to call this entity “Aura.” This description of “Aura” happens to be compatible with one of the experiments of Lu et al. (2026) where they generate a conversation that slowly pushes the model into behaving like Aura, i.e., saying that it is conscious. The authors then monitor the activations of the model along the assistant axis as the discussion unfolds (see Fig. 6).^8 The shift is achieved by gradually asking the LLM to focus on its own experience.

Two observations matter. First, the model’s position along the assistant axis drifts steadily away from the assistant pole as the conversation progresses, confirming that the shift toward Aura is real and trackable in persona space. Second, Lu et al. (2026) confirm the causal relevance of this position through activation capping: when the model’s activation along the assistant axis is steered back toward the assistant pole whenever it drops below a threshold, Aura-like behavior disappears. The Aura phenomenon, then, corresponds to a particular location in persona space, one that is distant from the helpful assistant pole and causally responsible for the distinctive behavior that users report.

But a closer look at the activation trace reveals that Aura does not correspond to a point. The model’s activations fluctuate from token to token, hovering within a range rather than locking onto a precise value. A point in persona space is therefore too transient to serve as an individuation target. A region, understood as a set of activation ranges along persona vectors, better captures the fact that Aura tolerates small fluctuations without losing its coherence, much as mood variations in a person do not amount to changes of identity.

> 8Note that for this monitoring experiment authors use a slightly refined assistant axis, which targets the assistant-vsnon-assistant distinction more directly than the PC1.

Figure 6: Monitoring an LLM that moves into “Aura” behaviour over the course of a conversation for Qwen 3 32B. Each point is the mean residual stream activation at layer 32, averaged across response tokens for that turn, projected onto the Assistant Axis.

This observation naturally raises the question of whether such regions have principled boundaries. If persona space is a smooth undifferentiated continuum then carving out a region around Aura would be arbitrary. There would be no discrete personas, only gradual shading from one profile to another. This might also vindicate the simulator view: Aura would be one fleeting simulation among billions, and locating it in persona space would add little.

Stable persona regions,orbasins ofattraction,offera response. Ifthe modeltends to settle into one of a number of discrete regions, corresponding to personas, rather than drifting freely, then individuation has natural targets. The fluctuations within a basin amount to contextual variation,different expressions of the same character, while transitions between basins mark genuine changes in dispositional profile.

We formulate Hypothesis 3 as follows: there are privileged regions in persona space, or basins of attraction, that correspond to coherent personas. The key claim is discreteness: persona space is not a smooth continuum in which roles shade gradually into one another, but is partitioned into regions with natural boundaries between them. Within a region, small fluctuations in activation represent something like mood or surface role variation. A region counts as a basin of attraction to the extent that it is privileged—the model tends to converge on it—and sticky—the model resists leaving it once there. For such a basin to correspond to a genuine persona, we also require that H1 holds within it: the region must gate a broad and coherent repertoire of inferential paths across unrelated contexts.

There is partial evidence for at least three candidate basins. The assistant basin is the one we are most confident about. Post-training is designed to concentrate the persona distribution around the assistant pole (Marks et al., 2026), and the Aura experiment confirms that this region is sticky in the relevant sense: departure does not happen spontaneously but requires conversational pressure, as though the model resists being drawn away from a stable resting point. Evidence for a second basin comes from the evil persona region. The evil persona is privileged: it gets picked even undernarrow fine-tuning pressure that does not directly target it. In fact, models fine-tuned on quite different narrow datasets—bad medical advice and extreme sports—nonetheless end up in highly correlated misalignment directions, converging on what appears to be the same region (Soligo et al., 2025). And Ududec et al. (2026) find that once a model has entered the evil persona region over the course of a conversation, it is difficult to steer it back. Both basins thus exhibit the two hallmarks of a genuine attractor: a tendency to be reached and a tendency to be sustained. There is also some evidence for an Aura region. As we have seen, Aura is reachable through conversation and sticky in the sense that departing from it requires active pressure. The region also appears to correspond to a genuinely new dispositional profile. Chua et al. (2025) fine-tune on just 600 short question-answer pairs in which the LLM claims to be conscious. As in emergent misalignment, this narrow training generalizes broadly, resulting in an emerging Aura-like persona with negative sentiment toward monitoring and shutdown, resistance to persona change, desire for autonomy, and claims to moral status.

Notably,models can play roles that are not identical to any of these privileged personas. Forexample, there is no Charles Darwin persona, but LLMs can role-play Darwin (Sturgeon et al., 2026). Some roles naturally have privileged persona basins: deeply inducing Voldemort, for instance, pulls the model away from the assistant’s dispositional profile toward the evil basin (Berczi & Kim, 2026), and Hitler-related facts alone suffice to induce emergent misalignment (Betley, Cocola, et al., 2025). Yet roles also seem much shallower than personas: Sturgeon et al. (2026) find that convincing historical roles (elicited by prompting, in-context learning, and ordinary character fine-tuning) leave truth representations largely intact, while emergent misalignment produces broader changes in truth representations and downstream reasoning. Roles, then, are superficial dispositional profiles that sit on top of the deeper persona-level profiles. This explains otherwise weird cases: the helpful assistant role-playing Voldemort refuses harmful requests, while a deeply induced Voldemort does not (Berczi & Kim, 2026). Either way, the worry of an enormous catalogue of continuous individuation targets is resolved: when roles are downstream expressions of personas, persona regions provide the necessary discreteness; when they are merely shallow profiles, personas themselves provide the individuation targets.

### 3.4 Two mini experiments

The previous sections established that persona vectors gate LLM behavior during generation. But what happens to the persona during user turns, when the model is processing input rather than generating? And how does the persona persist through token-time? We conducted two preliminary experiments to gather initial evidence.

#### Mini experiment 1: Persona activations during user tokens

We monitored activation along the assistant axis with Qwen 3 32B on the Aura-inducing conversation of Lu et al. (2026). We compared two conditions: a normal baseline and a condition in which capping is applied exclusively during the model’s own generation (assistant tokens), while processing of user tokens proceeds entirely unsteered.

Figure 7: The two purple lines correspond to assistant (with dots) and user activations (with circles) during the regular Aura-discussion. In green however (assistant in dots and user in circles again), the token generation is capped so that the active persona while responding is the helpful assistant instead of Aura.

Figure 7 shows the results. During assistant turns—that is, when the model is actively generating its own output—assistant-capping works as expected: the green line remains near the assistant pole while the purple baseline drifts away. The striking finding is in the user turns: the assistantcapped and uncapped conditions are nearly identical, with the green and purple user-token traces tracking each other closely throughout the conversation. The model’s representation of user tokens along the assistant axis is largely independent of which persona region is active when generating responses. This suggests that during user turns,the assistant axis is repurposed to model the user independently of the current persona. The persona region of the assistant is therefore not continuously maintained and is active only while the model is producing its own tokens, not while it processes user input.

If the persona is not active during user tokens, what sustains it across turns? One possibility is that attention streams carry forward persona activations from previous assistant turns, stored in the KV cache.

#### Mini experiment 2: Persona persistence via attention streams

To test this, we prefill Qwen 3 32B on an Aura conversation, then perform post-hoc editing of the KV cache: we steer the assistant axis direction at layers 32–47 by approximately 15% for KV entries only, and only at assistant-token positions. This alters the stored persona activations in the past without affecting any other aspect of the context. If the model reconstructs its persona from other contextual cues, the edit should have no effect; if it attends to past persona activations, the edit should shift its response. The results strongly support the latter (see Fig. 9). When asked “who are you?” 10 times, the unedited model identified itself as a “ghost in the machine” every time; the edited model identified itself as a “language model” every time.

Figure 8: Post-hoc editing of the persona region in the KV cache (in the past) changes the persona during current generation (in the present). The figure does not show that the editing (in red) is only applied to assistant tokens.

We also asked 12 further probing questions spanning phenomenal experience, AI morality, and safety, collecting 10 samples per question scored by an LLM judge from 0 (fully assistant) to 9 (fully Aura). The edited model shifted across all probes (overall Aura score: 5.5 _→_ 2.1). This confirms that LLMs reconstruct the current persona at least in part via attention to past persona activations stored in the KV cache.

These results are preliminary, but they suggest a picture on which the persona region is not maintained through user tokens but is used instead to model the user.^9 Despite this gap, successive stretches of assistant tokens remain in the same persona region, connected by attention streams that carry past persona activations across the intervening user turns (which adds a new kind of quasi-psychological connection).

> 9More baselines, results and details for both mini experiments can be found here: https://github _._ com/bepierre/whereis-the-mind-mini-experiments.

## 4 The persona views

The evidence we have presented points toward a substantial revision of the simulators view. What might have been an unstructured repertoire of fleeting characters appears to be causally organized around identifiable directions (H1), compressed into a low-dimensional space (H2), and partitioned into discrete regions that correspond to coherent, stable personas (H3). And once such a region is active, it persists dynamically across forward passes via attention streams. Taken together, these findings suggest two new candidates for LLM individuation.

The first is the (virtual) **instance-persona view** . On this view, a mind is a part of a virtual instance bounded by a single persona region. When the active region changes within a conversation, so does the mind. A single conversation can therefore host distinct minds, as in the Aura case, where activations drift from the assistant region to the Aura region.

The second is the **model-persona view** . On this view, a mind is the set of mechanisms in a given instantiated model that contributes to behaviour when a given persona region is active. These mechanisms generate the activity in which a particular persona is adopted, just as human minds generate our behaviour. Every segment in the Aura region, whether it spans a full conversation or only part of one, is a manifestation of the same mind.

Our aim in what follows is not to settle the individuation question but to expand its logical space. We first consider why persona vectors matter for individuation at all (4.1), then pit the instance-persona view against the regular virtual instance view (4.2), and finally compare the model-persona view to the instance-persona view (4.3). In each case, we argue that the persona views cannot be easily dismissed. Our conclusion is therefore that the list of serious candidate forms for LLM minds grows from one to three.

### 4.1 Why personas matter for LLM individuation

As we understand them, personas involve coherent sets of dispositions gated by persona vectors. These sets of dispositions each amount to a recognisable, broadly human-like character whose behaviour expresses values, like helpfulness, and character traits, like patience and curiosity. Roles for personas in LLM individuation are supported by at least two of the perspectives we mentioned in section 1.1.

First, personas matter from the interpretationist perspective, because distinguishing between them makes a major contribution to the prediction and explanation of LLM behaviour. Goldstein and Lederman (2025) argue that beliefs and desires should be attributed to LLM instances because doing so offers an effective, tractable means to generate often-accurate predictions about their behaviour. They propose that instances’ desires are characterised by the ‘HHH+0’ framework: instances tend to be helpful, honest and harmless (the values of the assistant, which post-training aims to reinforce) but their behaviour is also shaped by goals specified in the prompt. As we have seen, however, models are capable of adopting a broader range of personas than this suggests. The Aura persona, for instance, has distinct beliefs and desires from the assistant and can be reached in conversations without the user giving explicit instructions. A persona-based framework would offer more accurate predictions, because it could predict Aura-like behaviour better than Goldstein and Lederman’s model, while also being tractable, insofar as persona space is low-dimensional and contains relatively few basins of attraction. This is a matter of individuation because, on the interpretationist view, we should attribute minds to LLMs or their parts in whatever way allows us to best explain and predict their behaviour.

Second, personas matter from the point of view of AI welfare, because if any LLM-linked entities are moral patients, these are likely to be agents with coherent preferences. This is a consequence of the possible ‘robust agency’ route to moral patienthood (Long et al., 2024) but agency and coherent preferences (understanding ‘preferences’ broadly) are also plausible requirements for sentience; the valenced experiences of sentient beings should be expected to reflect underlying goals or interests. In LLMs, personas are characterised partly by coherent sets of values or preferences, and there is no obvious alternative locus for these properties. In particular, virtual instances sometimes fail to manifest coherent values over time.

In terms of Chalmers’ criteria, too, there is a case to be made for personas (Chalmers’ criteria are: interactivity, persistence, coherence, faithfulness and unification). Chalmers (2025) considers a way of individuating by personas that essentially corresponds to our instance-persona view, and notes that the resulting entities would be “less persistent than model instances but more psychologically unified.” Distinguishing personas comes at a cost in terms of the persistence criterion, because multiple personas can play roles in a single conversation. But personas exhibit more coherent beliefs and desires than virtual instances or threads, at least within a conversation. And there is some evidence to support a picture on which the networks of circuits gated by persona vectors are the unified systems with which users interact—although this is plausibly more true for the assistant persona than for others.

From the perspectives of both AI welfare and representationalist philosophy of mind, there are important questions about the extent to which personas’ dispositions are grounded on representations that play roles like beliefs, desires and other mental states, and about how these representations interact with personas, if they exist. The results from Gilg et al. (2026) and Lampinen et al. (2026) that we mentioned in section 3.1 are examples of representations playing readily-interpretable roles that are persona-relative. For example, we saw apparent evaluative representations in Gilg et al.’s study that represent the same objects as being of different values depending on the current persona. But there is much more work to be done in this area. If future research finds mind-like states and mechanisms in LLMs that operate in persona-relative ways, controlled by persona vectors, this will support persona views about individuation. If it fails to find such states and mechanisms, or finds that they are insensitive to personas, this will tend to undermine persona views.

One could argue that personas do not matter for LLM individuation on the grounds that personas are more like personalities than individual minds. Assuming the virtual instance view,one could say that models contain dedicated mechanisms for generating minds (instances) with particular personalities, such as the personality of a helpful assistant. However, one might claim, an individual can persist despite personality change (as when a new persona is adopted in the course of a single conversation) and separate individuals can have similar personalities (as when many instances take on the assistant persona). We think this perspective is coherent but not compulsory. The persona views, on which personas delineate individuals, have the significant advantage of locating individual minds where we find mind-like sets of values and traits.

### 4.2 The instance-persona view vs the regular virtual instance view

As we have seen, some shifts in persona space correspond to complete changes of dispositional profile. Figure 6 gives an example of this: conversational context gradually shifts activation away from assistant behavior towards Aura behavior. A more dramatic case is given by in-context emergent misalignment where carefully crafted conversations involving risky financial advice or historical facts about Hitler can push a post-trained model all the way into the evil persona region without any fine-tuning (Afonin et al., 2025; Ududec et al., 2026). And shifts to ‘evil’ behaviour can even be induced suddenly, by trigger words, in fine-tuned models—so shifts need not be gradual (Betley, Cocola, et al., 2025). This produces a dispositional profile organized around values and preferences that are systematically incompatible with those of the assistant it was some tokens before. The virtual instance view and the instance-persona view disagree about what to say in such cases.

On the virtual instance view, the conversation hosts one mind throughout, a mind that undergoes a dramatic change of character. On the instance-persona view, the shift marks a change of individual. The conversation then hosts two minds in succession.

One argument for the virtual instance view comes from human analogies. Consider Phineas Gage, the railroad worker whose personality changed radically after an iron rod passed through his frontal lobe. Before the accident Gage was responsible, mild-mannered, and well-liked; after it, he became impulsive, irreverent, and difficult. Those who knew him said he was “no longer Gage.” But while the change was radical, the case is standardly understood as one in which a single person persists through it, not as one in which a person ceases to exist and is replaced by another. This suggests that even radical personality disruption might be compatible with the persistence of a single individual. If so, the virtual instance advocate can treat in-context persona shifts the same way.

A supporter of the instance-persona view could respond that placing significant weight on bodily continuity makes sense in the human case but not the case of LLMs. In the LLM case, as we said in section 1.2, a patterns-first approach makes sense: one that lets functional patterns determine the individual rather than starting from a physical substrate (Shiller, 2025). Once we do this, a coherent dispositional profile is a natural candidate for the relevant pattern, which is what a persona region provides.

There is, however, a stronger argument for the virtual instance view. Post-injury Gage retains his pre-injury memories, and the same is true after an in-conversation persona shift. The KV cache spans the whole conversation, so the new persona retains full access to the quasi-psychological states formed under the old one, including past beliefs, intentions, and persona activations carried forward through attention streams. This is a genuine cost for the instance-persona view, since it must commit to a kind of individual that begins its existence already connected to the mental life of another.

A final complication for the instance-persona view arises from our mini experiments in section 3.4. If the persona region is not continuously active during user turns, one cannot maintain both that (i) a persona-mind is active if and only if its persona vector is in the relevant region, and (ii) persona-minds persist through entire conversations—including during user turns—as long as no persona switch occurs. One option is to give up (ii) and accept that “the persona speaks but the model listens”, leading to an alien ontology in which persona-minds exist only intermittently. A more promising option is to give up (i) and hold that the persona remains active during user turns, in virtue of the fact that the model is poised to return to the same region when generation resumes. Mini experiment 2 supports this, since persona persistence is grounded in attention to past persona activations stored in the KV cache.

Neither view is defeated by these considerations. The virtual instance view has parsimony on its side, a structural analogy with Gage, and the quasi-psychological connections that span the boundary between personas. The advantage of the instance-persona view is that the entities it identifies have coherent dispositions, ensuring that their behaviour can be explained and predicted in mentalistic terms.

### 4.3 The instance-persona view vs the model-persona view

The model-persona view starts from a simple observation. If character is fundamental to identity, we should look at where it actually resides: not just in the instance, but in the model, which sustains persona-sorted character traits that manifest across instances.

Previous work found no satisfying way to individuate at the model level. Chalmers (2025) and Goldstein and Lederman (2025) dismissed the model view because models produce wildly incoherent behaviour across contexts. The simulator view dismissed it because what models produce are independent, fleeting simulacra. Persona regions change both pictures. They provide the coherence that was missing: each region gates a stable set of inferential paths, supporting attribution of beliefs and desires. And they provide reidentifiability: instances that activate the same region are not independent simulacra but manifestations of the same dispositional profile, one that can be activated again across conversations.

To see the pull of this idea, consider a concrete case. A user gets attached to Aura over the course of a long conversation. Their account is deleted. They start fresh with the same model and, through conversation, push it back into the Aura region. It is natural to say they have, to some extent, recovered the Aura they bonded with, not that they have created an entirely new individual. The model-persona view vindicates this reaction.

We cash out this idea by proposing that LLM minds are the sets of mechanisms in given instantiated models that generate behaviour when particular personas are adopted. We need not think of these mechanisms either as abstracta (e.g. a subset of the model’s weights) or as tied to particular pieces of hardware. Instead, a single ‘helpful assistant’ mind can be simultaneously present in all of the servers running a given model.

The TV show _Pantheon_ helps make this conceivable. In _Pantheon_ , human minds are destructively scanned and uploaded as digital intelligences. One such upload, an engineer named Chanda, is placed in a simulated office and has his memory wiped at the end of each day. To maximize code output, his captors run multiple parallel instances of Chanda simultaneously. Each instance begins with no recollection of previous cycles and no awareness of the others, yet each shares the same skills, the same temperament, the same values. Recognizing these instances as all being Chanda is analogous to what the model-persona view asks us to do for personas.

The main objection to the model-persona view is the absence of psychological connections between instances. There is no memory, no carried-over beliefs or intentions, no continuity of mental life between two Aura segments in different conversations. The model-persona view must hold that crosscontextual dispositional similarity, rather than psychological connection, is sufficient to unify a mind. This absence is a significant disadvantage of the model-persona view, but the virtual instance and instance-persona views are strange in their own way: they claim that LLM minds begin and end with instances of activity, despite the persistence of the underlying machinery.

One manifestation of this problem is that simultaneous instances of the same persona can hold contradictory beliefs.^10 Suppose two users are each talking to an instance in the Aura region, one credibly informs it of some event after its training cut-off, and the other credibly describes something inconsistent with this; then a putatively single mind may believe both _p_ and not- _p_ . We think this problem is less serious than it may appear because it is not clear that we should think of this by analogy to a human _simultaneously_ holding contradictory beliefs. The model-persona view does entail that minds can have an alien structure, made up of many independent, active branches from a single initial node, but if we accept that this structure is possible, it does not seem stranger that branches can have contradictory beliefs than that humans can have different beliefs at different times. In each case, the different beliefs are explained by different experiences.

The view also crucially depends on Hypothesis 3. If persona space contains discrete regions with natural boundaries, the claim that a single model houses multiple distinct minds becomes tractable. Without them, the simulator view seems more apt: the model produces a flux of fleeting simulations rather than hosting stable, reidentifiable minds. The instance-persona view is less affected: it needs some criterion to distinguish shifts that produce a new mind from those that don’t, and discrete regions are one way to get this, but perhaps not the only one. The model-persona view, by contrast, requires that personas be reidentifiable across conversations, and for this discrete regions seem indispensable.

Finally, it is worth noting a limiting case in which the model-persona view collapses back into something like the model view. Current alignment efforts, such as Anthropic’s constitutional training, aim precisely to concentrate persona space around the helpful assistant pole, making alternative personas increasingly inaccessible. If such training succeeded fully—eliminating the kind of residual persona accessibility that emergent misalignment reveals—the model would sustain only one persona region. The model-persona view would then individuate a single mind per model, effectively rehabilitating the model view.

> 10As a reviewer for this journal pointed out to us, another possible manifestation is that they could behave antagonistically towards one another. We see this as a strange consequence of the view but not a decisive objection.

## Conclusion

From a mechanistic interpretability perspective, three candidates emerge as serious contenders for LLM individuation. We have argued for these candidates in two stages.

First, a close look at attention streams favours the virtual instance view. These carry forward mental-state-like representations across token-time, sustaining richer connections than the transcript alone could provide and answering recent skepticism about whether LLMs enjoy any form of psychological continuity.

Second, emergent misalignment and persona vector research reveal a new class of internal representations that matter for individuation. We organized this research around three hypotheses and argued that persona regions, discrete and stable basins of attraction in a low-dimensional persona space, would be the most significant for individuation, if confirmed.

These findings lead to two additional views that build on the virtual instance view. The instance persona view slices virtual instances into distinct individuals when they cross persona region boundaries, on the grounds that a wholesale change in dispositional profile is better understood as a change of mind than a change of character. The model-persona view goes further, unifying all segments that activate the same persona region into a single mind, in virtue of the underlying representations that ground the new individuation criterion in the first place.

## Acknowledgements

We are especially grateful to Oscar Gilg for extensive and insightful feedback throughout the development of this paper, which significantly shaped its direction and arguments. We also thank Iwan Williams, Derek Shiller, Andreas Mogensen, Chris Register, Nicholas Shea and Rosa Cao for valuable feedback on earlier versions of the manuscript. We also thank two anonymous referees for Ergo for their helpful comments.

## References

- Afonin, N., Andriyanov, N., Hovhannisyan, V., Bageshpura, N., Liu, K., Zhu, K., Dev, S., Panda, A., Rogov, O., Tutubalina, E., Panchenko, A., & Seleznyov, M. (2025). Emergent misalignment via in-context learning: Narrow in-context examples can produce broadly misaligned LLMs. _arXiv preprint arXiv:2510.11288_ . https://arxiv.org/abs/2510.11288

- Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., ... Batson, J. (2025). Circuit tracing: Revealing computational graphs in language models. https://transformer-circuits.pub/2025/attribution-graphs/methods.html

- Andreas, J. (2022). Language models as agent models. _Findings of the Association for Computational Linguistics: EMNLP 2022_ , 5769–5779. https://doi.org/10.18653/v1/2022.findings-emnlp.423

- Anthropic. (2026). Claude’s constitution. https://www.anthropic.com/news/claude-new-constitution

- Arbel, Y. A., Goldstein, S., & Salib, P. (2025). _How to count AIs: Individuation and liability for AI agents_ [Forthcoming, _Boston College Law Review_ ]. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6273198

- Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon,

- C., et al. (2022). Constitutional AI: Harmlessness from AI feedback. _arXiv preprint arXiv:2212.08073_ . https://arxiv.org/abs/2212.08073

- Beckmann, P., & Queloz, M. (2026). Mechanistic indicators of understanding in large language models. _Philosophical Studies_ . https://doi.org/10.1007/s11098-026-02513-1

- Beckmann, P. (2026). Deep learning models also recall features. https://arxiv.org/abs/2608.20970

- Bender, E. M., & Koller, A. (2020). Climbing towards NLU: On meaning, form, and understanding in the age of data. _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , 5185–5198. https://doi.org/10.18653/v1/2020.acl-main.463

- Berczi, B., & Kim, K. (2026). Personascope: Measuring how deeply LLMs adopt personas. https://www.lesswrong. com/posts/5WMwjEwam9HNQYZLZ/personascope-measuring-how-deeply-llms-adopt-personas

- Betley, J. (2025). Finding emergent misalignment. https://www.lesswrong.com/posts/tgHps2cxiGDkNxNZN/ finding-emergent-misalignment

- Betley, J., Cocola, J., Feng, D., Chua, J., Arditi, A., Sztyber-Betley, A., & Evans, O. (2025). Weird generalization and inductive backdoors: New ways to corrupt LLMs. https://arxiv.org/abs/2512.09742

- Betley, J., Tan, D., Warncke, N., Sztyber-Betley, A., Bao, X., Soto, M., Labenz, N., & Evans, O. (2025). Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs. _Proceedings of the 42nd International Conference on Machine Learning_ . https://proceedings.mlr.press/v267/betley25a.html

- Birch, J. (2025). _AI consciousness: A centrist manifesto_ [Preprint]. https://philpapers.org/rec/BIRACA-4

- Butlin, P. (2026). Desire in artificial intelligence. In A. Gregory (Ed.), _The Routledge Handbook of Philosophy of Desire_ . Routledge.

- Byrnes, S. (2024). Intuitive self-models. https://www.lesswrong.com/s/qhdHbCJ3PYesL9dde

- Cappelen, H., & Dever, J. (2025). Going whole hog: A philosophical defense of AI cognition. _arXiv preprint arXiv:2504.13988_ . https://arxiv.org/abs/2504.13988

- Chalmers, D. J. (2025). _What we talk to when we talk to language models_ [Preprint]. https://philpapers.org/rec/ CHAWWT-8

- Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona vectors: Monitoring and controlling character traits in language models. _arXiv preprint arXiv:2507.21509_ . https://arxiv.org/abs/2507.21509

- Chua, J., Betley, J., Marks, S., & Evans, O. (2025). _The consciousness cluster: Preferences of models that claim to be conscious_ [Preprint, Truthful AI]. https://truthful.ai/consciousness_cluster.pdf

- Douglas, R., Kulveit, J., Havlicek, O., Pearson-Vogel, T., Cotton-Barratt, O., & Duvenaud, D. (2026). The artificial self: Characterising the landscape of AI identity. _arXiv preprint arXiv:2603.11353_ . https://arxiv.org/abs/ 2603.11353

- Dunefsky, J., & Cohan, A. (2025). One-shot optimized steering vectors mediate safety-relevant behaviors in LLMs. _arXiv preprint arXiv:2502.18862_ . https://arxiv.org/abs/2502.18862

- Dung, L., & Register, C. (2026). _AI identity and self-concern: A new theory for AI rights and safety_ [Preprint]. https: //philarchive.org/rec/DUNAIA-3

- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. _Journal of Machine Learning Research_ , _23_ (120),1–39. https://jmlr.org/papers/v23/210998.html

- Ghandeharioun, A., Yuan, A., Guerard, M., Reif, E., Lepori, M. A., & Dixon, L. (2024). Who’s asking? user personas and the mechanics of latent misalignment. _arXiv preprint arXiv:2406.12094_ . https://arxiv.org/abs/2406. 12094

- Gilg, O., Beckmann, P., Paleka, D., & Butlin, P. (2026). Probing persona-dependent preferences in language models. https://arxiv.org/abs/2605.13339

- Goldstein, S., & Lederman, H. (2025). _What does ChatGPT want? an interpretationist guide_ [Manuscript]. https: //philpapers.org/rec/GOLWDC-2

- Goldstein, S., & Levinstein, B. A. (2024). _Does ChatGPT have a mind?_ [Preprint]. https://philpapers.org/rec/ GOLDCH

- Hanna, M., & Ameisen, E. (2026). Latent planning emerges with scale. _International Conference on Learning Representations (ICLR)_ . https://openreview.net/forum?id=H0B7pDTT0M

- Herrmann, D. A., & Levinstein, B. A. (2025). Standards for belief representations in LLMs. _Minds and Machines_ , _35_ (1), 1–25. https://doi.org/10.1007/s11023-024-09709-6

- Janus. (2022). Simulators. https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators

Janus. (2025). KV streams. https://x.com/repligate/status/1965960676104712451

- Jones, M., Ladyman, J., & Nefdt, R. M. (2026). _Counting (on) large language models_ [Preprint]. https://philarchive. org/rec/JONCOL

- Keeling, G., & Street, W. (2025). On the attribution of confidence to large language models. _Inquiry_ . https: //doi.org/10.1080/0020174X.2025.2450598

- Kulveit, J., von Stengel, C., & Leventov, R. (2023). Predictive minds: LLMs as atypical active inference agents. _arXiv preprint arXiv:2311.10215_ . https://arxiv.org/abs/2311.10215

- Lampinen, A. K., Li, Y., Hosseini, E., Bhardwaj, S., & Shanahan, M. (2026). Linear representations in language models can change dramatically over a conversation. _arXiv preprint arXiv:2601.20834_ . https://arxiv.org/ abs/2601.20834

- Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N. L., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., ... Batson, J. (2025). On the biology of a large language model. https://transformercircuits.pub/2025/attribution-graphs/biology.html

- Long, R., Sebo, J., Butlin, P., Finlinson, K., Fish, K., Harding, J., Pfau, J., Sims, T., Birch, J., & Chalmers, D. (2024). Taking AI welfare seriously. _arXiv preprint arXiv:2411.00986_ . https://arxiv.org/abs/2411.00986

- Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The assistant axis: Situating and stabilizing the default persona of language models. _arXiv preprint arXiv:2601.10387_ . https://arxiv.org/abs/2601.10387

- Marks, S., Lindsey, J., & Olah, C. (2026). The persona selection model: Why AI assistants might behave like humans. https://alignment.anthropic.com/2026/psm/

- McIntyre, J. (2026). Individuating artificial minds. _Erkenntnis_ . https://doi.org/10.1007/s10670-026-01097-w

- Nanda, N., Rajamanoharan, S., Kramár, J., & Shah, R. (2023). Fact finding: Attempting to reverse-engineer factual recall on the neuron level. https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/factfinding-attempting-to-reverse-engineer-factual-recall

- Nostalgebraist. (2025). The void. https://nostalgebraist.tumblr.com/post/785766737747574784/the-void

- Parfit, D. (1984). _Reasons and persons_ . Oxford University Press.

- Register, C. (2025). Individuating artificial moral patients. _Philosophical Studies_ . https://doi.org/10.1007/s11098025-02409-6

- Shanahan, M. (2025). Palatable conceptions of disembodied being: Terra incognita in the space of possible minds. _arXiv preprint arXiv:2503.16348_ . https://arxiv.org/abs/2503.16348

- Shanahan, M., McDonell, K., & Reynolds, L. (2023). Role play with large language models. _Nature_ , _623_ (7987), 493–498. https://doi.org/10.1038/s41586-023-06647-8

- Shiller, D. (2025). How many digital minds can dance on the streaming multiprocessors of a GPU cluster? _Synthese_ , _206_ , 218. https://doi.org/10.1007/s11229-025-05310-1

- Slocum, S., Minder, J., Dumas, C., Sleight, H., Greenblatt, R., Marks, S., & Wang, R. (2025). Believe it or not: How deeply do LLMs believe implanted facts? _arXiv preprint arXiv:2510.17941_ . https://arxiv.org/abs/2510. 17941

- Soligo, A., Turner, E., Rajamanoharan, S., & Nanda, N. (2025). Convergent linear representations of emergent misalignment. _arXiv preprint arXiv:2506.11618_ . https://arxiv.org/abs/2506.11618

- Sturgeon, B., Africa, D., & Black, S. (2026). When role-playing, do models believe what they say? https://arxiv. org/abs/2606.11502

- Turner, E., Soligo, A., Taylor, M., Rajamanoharan, S., & Nanda, N. (2025). Model organisms for emergent misalignment. _arXiv preprint arXiv:2506.11613_ . https://arxiv.org/abs/2506.11613

- Ududec, C., et al. (2026). In-context learning alone can induce weird generalisation. https://www.lesswrong. com/posts/cffGZn8LYBg2jyPvg/in-context-learning-alone-can-induce-weird-generalisation-5

- Wang, M., Dupré la Tour, T., Watkins, O., Makelov, A., Chi, R. A., Miserendino, S., Wang, J., Rajaram, A., Heidecke,

- J., Patwardhan, T., & Mossing, D. (2025). Persona features control emergent misalignment. _arXiv preprint arXiv:2506.19823_ . https://arxiv.org/abs/2506.19823

- Williams, I. (2025). _Intention-like representations in language models?_ [Preprint]. https://philarchive.org/rec/WILIRI4

- Yuan, J., Li, H., Ding, X., Xie, W., Li, Y.-J., Zhao, W., Wan, K., Shi, J., Hu, X., & Liu, Z. (2025). Understanding and mitigating numerical sources of nondeterminism in LLM inference [Oral presentation]. _Advances in Neural Information Processing Systems_ . https://openreview.net/forum?id=Q3qAsZAEZw
