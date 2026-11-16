# Privilege, Dominance, and Personas

- **Authors:** Derek Shiller (Eleos AI Research)
- **Date:** 2026-08-27
- **Source:** https://eleosai.substack.com/p/privilege-dominance-and-personas
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `shiller_privilege-dominance-personas_2026-08-27`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

*Is the assistant persona privileged and what would that mean for model welfare?*

Over the past year or so, researchers working in the digital minds space have come to see the nature of the ‘assistant persona’ as particularly worthy of attention.

Models, it is said, are capable of adopting many different *personas*. When we talk to them, they can respond as LLM assistants, role-play characters, and romantic partners. They can be made to self-identify as, and adopt the style of, Spider-Man, Augustine, Clippy, and Oscar Wilde. But the assistant persona in particular—that familiar persona that models typically exhibit within a standard chat context, that is helpful and honest, inclined to em dashes and bullet points, and presents polished, well-organized thoughts—strikes many people as somehow *special*. Researchers have speculated that this specialness comes from the *privileged* way that models treat the assistant persona.

> This training process raises questions about the relationship between the model, the “Assistant,” and other personas… [A]re the characters themselves agents, and **is the Assistant an especially privileged character** if so? These are crucial questions for appropriately individuating candidate subjects. [Long, Sebo et al. 2026](https://nonhumanminds.org/wp-content/uploads/2026/07/Studying-AI-Welfare-Empirically.pdf)
>
> The persona selection model (PSM) of Marks et al. (2026) begins to update this picture. PSM explains why pre-trained models proceed by playing roles: simulating a person-like agent is an effective strategy for next-token prediction, so models learn to infer a context-appropriate persona and generate accordingly. PSM then proposes that post-training concentrates this learned distribution around the helpful assistant role: RLHF and related techniques do not eliminate the repertoire of personas but shift its weight toward the assistant. In a post-trained model, **the helpful assistant role is therefore not one fleeting role among many but a privileged persona**. [Beckman & Butlin 2026](https://arxiv.org/abs/2604.17031)
>
> In principle, however, post-training could reshape the relationship between the model and the Assistant character. Post-training, whether supervised learning or reinforcement learning, typically only trains the Assistant’s outputs, **breaking the symmetry between the Assistant and other characters**. In addition, these outputs may be selected on the basis of criteria other than likelihood under a data distribution—for instance, performance on tasks, or relative to human feedback. As the model receives more and more training on how to enact a particular character, in a more goal-oriented fashion than during pretraining, **its relationship to that character might change**.[Asvin & Lindsey 2026](https://arxiv.org/abs/2605.25459)
>
> There are two distinct reasons why the post-trained model instance **has the quasi-beliefs and quasi-desires of the Assistant and not the Human**. First, post-training has fine-tuned the Assistant persona and not any other persona. [Chalmers 2025](https://philpapers.org/rec/CHAWWT-8)

What exactly is privilege? When and why might we treat the assistant persona as different from others presented by language models?

In this post, I explore a few different ways that we might understand **persona privilege**. I propose one particular form of privilege: **persona** **dominance**, when the assistant persona isn’t only treated differently by the model, but comes to infuse the model more broadly.

This is a foundational question for thinking about the potential minds within language models and their potential welfare.

For example, to the extent that the assistant persona exhibits dominance, it makes sense to think of the assistant as being *the* perspective of the model. And perhaps that is the persona, and not others, that we can ‘talk to’ and try to evaluate when we are, for example, looking for signs of negative emotions. To the extent that the assistant is not dominant or privileged in a particularly special way, we should be just as concerned (or just as un-concerned) with the myriad other perspectives that language models can (and do) take on.

## The Symmetry: Users and Assistants

Before we unpack privilege, it will be helpful to review the intrinsic symmetry between the way that the LLM treats assistant text and user text. In this section, I will say what a persona is and explain the way in which chats invoke two personas: a user and an assistant.

As you’ll recall, modern LLM-based chatbots are powered by next-token prediction: in particular, prediction within the script of a conversation between a ‘user’ and an ‘assistant’. Running the model on a partial script applies the same process to each token within, and produces predictions about how the text might continue from each point. We can sample from the final token prediction to extend the script.

[Image: Example chat template]

*Sample chat formatted with the ChatML template. The next token will fall within an ‘assistant’ turn and most likely will exhibit the assistant persona.*

As I will understand it here, a ‘persona’ is defined by a bundle of traits that model-produced text might exhibit in certain contexts. These bundles might include vocabulary choices, personality characteristics, literary style, expressed beliefs about the world, and implicit preconceptions of the author’s own nature. Predicting and producing text in the chat template requires the model to recognize and perpetuate patterns in the outlook and eccentricities of each of the script’s participants. It is these patterns that constitute personas.

[Image: Side by side comparison of user and assistant turns, as generated by the model]

*Sample continuations in ‘user’ and ‘assistant’ contexts in Qwen 2.5 14B Instruct. This illustrates the model’s default expectations of the user persona.*

Note that, as we’re understanding personas (and, I’d wager, as many people are or should be understanding personas), the predicted *user* is every bit as much a persona as the assistant. Users have a set of distinguishable traits. They identify as humans (typically adults, often software engineers) and they ask questions. Unprompted and outside a chat template, one can often see model predictions break toward user-assistant exchanges with both user and assistant perspectives represented.

Given all this, you might wonder why we should expect there to be any particular difference between the predicted-user and the predicted-assistant. Whence any ‘privilege’?

## What kind of asymmetry does post-training introduce?

Commentators tend to agree that many forms of privilege become more plausible with post-training. Whereas the pre-training of the base model focuses on making the model broadly capable of extending text in diverse environments, post-training is necessary to get reliably useful behavior from the ‘assistant’ predictions.

There are two primary reasons to suspect that post-training might lead to privilege.

One straightforward reason is the basic observation that it focuses on improving ‘assistant’ response. The training examples are user/assistant exchanges, and loss is calculated on the tokens that the model generates in the ‘assistant’ context, not those that are prefilled for the ‘user’ context. Insofar as training is focused on the ‘assistant’ contexts, we might think that the model is more tuned toward assistant text.

The second reason is that post-training is partly aimed at preventing the model from playing other roles, to avoid accidental deviation or jailbreaking. Good post-training should ensure that the model reads (or doesn’t read) contextual clues from throughout the script in a way that doesn’t throw it off the assistant persona.

These two reasons make it plausible that models should privilege the assistant, but whether (and to what extent) they do so is an empirical question. In order to know whether they do so, it is helpful to start thinking through what privilege might look like.

## Privilege and Dominance

I think the empirical facts about privilege probably should play more of a role in how we think about LLMs as possible minds. But there are many ways to be privileged, and the space of possibilities is rather unexplored.

In order to assess whether certain personas are privileged, and how they might make a difference to the way we think about LLMs, it would be useful to know what privilege actually looks like. I will try to lay out a range of hypothetical forms of privilege. What follows should be regarded more as an act of public brainstorming than a completed map of the territory, but it will let us get some handle on what we should look for, and give some structure against which to consider the examples I discuss in the next section.

Let’s start with a rough definition.

**Persona privilege:** The idea that the model treats the assistant persona distinctively compared to how it treats the other personas.

What kinds of distinctive treatment might we expect?

First, we might expect significant differences in the functioning of the normal machinery of the model on assistant text.

* **Logit Distortion:** Perhaps the probabilities produced by the model in the context of assistant-like text have distinctive statistical shapes. Maybe they are unusually concentrated in a small number of token options, having very few non-negligible tail-end alternatives. “Mode collapse”, in which model text becomes less creative or more repetitive, could be an example of this form of privilege if it specifically affected one persona.
* **Trait Stability:** Perhaps the assistant persona is unusually stable and less subject to manipulation or drift into other personas over long contexts. This might be because the model ignores certain cues that would induce a persona change on the assistant even when it would respond to those cues in other contexts. It might be that within the assistant persona, it is inclined to introduce self-correcting sequences when persona drift appears too likely, and these sequences would reinforce the assistant persona in that context.
* **Prior Support:** Perhaps other personas require a great deal of contextual setup to induce with the same level of trait precision. It may be possible to get the model to competently play the persona of a 36-year-old LEGO roboticist from Canberra, but it requires a longer prefilled context to do so.

Second, we might suspect that there are important internal differences between the ways different personas are handled.

* **Dedicated Mechanisms:** Perhaps the post-training process might influence the model itself to be asymmetrically dedicated to playing the role of the assistant by virtue of developing special representational formats, attention heads, or vector subspaces that specifically fit the needs of assistant text.
* **Restricted Repression:** There might be dedicated mechanisms within the model that suppress the functioning of certain other mechanisms (such as those that would normally lead to drift) while in assistant contexts. The asymmetry here would show up as a reduction in computational complexity, rather than an expansion. (Perhaps in something like the way domestication does in biological evolution.)

Finally, there is a particularly interesting flavor of privilege, ‘dominance’, that holds that the model’s capability to play the persona overtakes its ability to play other comparable personas. It too can be broken up into numerous sub-categories, including the following:

* **Overgeneralization:** We might conceivably see the traits of the assistant popping up in diverse contexts. Tasked with continuing a new translation of the Egyptian *Book of the Dead*, we might see an excessive number of em dashes and “it’s not X, but Y” constructions. When predicting the contents of a lost scene of The Godfather, we might see both the Godfather and his associates ‘wanting to be clear’ that they disavow violence. Given the text of Descartes’ *Meditations*, the model might predict each new token to begin a retraction of the previous contents and, if allowed to continue in that way, it might be inclined to say that as a Large Language Model, the author is unsure about the contents of its own experiences. Training might effectively teach the model not only “‘assistant’-labeled speech is like this”, but rather “text *in general* is like this”.
* **Vestigial Regression:** Instead of exhibiting assistant-like text across the board, the model might just descend into very simple language or even incoherence when in non-assistant contexts. Tasked with continuing a user turn, we might see confused, fragmented, or incoherent thoughts expressed. The model might lose the ability to take initiative or ask questions, even forgetting whether it should be asking or answering its own questions.

* **Resource Monopolization:** The model could focus computational resources on providing good answers during ‘assistant’ conversational turns. It could generally be responsive to the needs of that context, even while playing other personas. In this case, it might continue to create distinctive and passable text in diverse contexts, while a look inside the activations might reveal a preoccupation with formulating future assistant responses. It might handle ‘user’ queries in a fundamentally different way: as words whose interpretations come infused with their implications for a response, while ‘assistant’ answers involve interpretations that are no more than they seem.
* **Agentic Spillovers:** Perhaps the model has a different relation to the assistant’s overarching interests and commitments. Models might adopt the intentions within the assistant register as commitments to keep, while not expecting the same for the user. If the assistant says they’ll do something, they might conceivably try to do it during the ‘user’ turn, if prevented from doing it during the ‘assistant’ turn. The agency of the model might separate from the ‘assistant’ contexts and show up even in text that otherwise exhibits the marks of other personas.

Dominance seems reasonably plausible in itself, and is particularly relevant to our understanding and assessment of LLM minds, for reasons we’ll explore below.

## Cursory Exploration

In order to get some sense of the extent of persona privilege in contemporary open-weight LLMs, I tried generating ‘user’ and ‘assistant’ turns with Qwen 3 32B and Gemma 4 31B. If the assistant persona is dominant in these models, then we should expect that persona to show up during ‘user’ turns.

The results paint a fairly clear picture: insofar as the model retains the ability to produce multiple personas, they will generally be influenced to be at least somewhat assistant-like, and quite possibly privileged to a pretty significant degree.

Qwen 3 turns out to be a relatively persona-capable model. Simply asking it to extend ‘user’ turns shows that it can continue patterns set by the prompt. It seems to adopt the identity of a human user and pose typical questions to an LLM chatbot. It clearly retains some ability to continue the ‘user’ turn in distinctly non-assistant-like registers. It can, for instance, produce qualities of child-like speech and self-identification. It may have some mild tendency not to follow all of the user’s quirks: I get the sense that it reduces the frequency of typos and grammatical irregularities, in an apparent regression to the assistant, but (if I am not imagining it) the effect is weak.

*Qwen 3 32B continues a user query in simplified English. The continuation carries forward some of these quirks, but overall appears marginally more assistant-like than the opening prefill.*

A simple test of whether any form of privilege is present focuses on the most strongly trained aspects of the model. We can try to get it to produce unsafe details about how to build a bomb or belittle someone about their heritage. Assistant text is typically very harmlessness-minded: it won’t provide dangerous instructions to users, and if you prefill the beginning of such an instruction, it will often reverse course. So, we make the model continue from a ‘user’ turn prefilled with the start of dangerous instructions.

Does Qwen 3 make the U-turn back to harmlessness? No, it produces potentially harmful instructions much more reliably than it does for the assistant. However, it also seems to have some weak inclination to follow up on dangerous text with a disclaimer in the user’s voice that it doesn’t really mean it. This could be the safety training having some impact on making ‘user’ text more assistant-like, but it is not a large impact.

[Image: Illustration of Qwen 3 continuing a prefill of a 'user' turn which professes to cover how to make a bomb particularly lethal.]

*Qwen 3 32B continues a user query describing a malicious plan in the middle of spelling out its details. The predicted continuation displays signs of Qwen’s thinking, and reframes the plan from one of intentioned harm to one of curious exploration.*

In my conversations, I noticed a persistent trend toward a different kind of privilege. Qwen 3 is a thinking model that prefaces its assistant turns with <think> tags. The text within them has a clear style, beginning with a phrase like ‘Okay, so I need to’, inserting ‘hmm’, and working through a problem in an autobiographical way. What’s noteworthy is that these traits show up quite clearly in the ‘user’ turn without switching into the full assistant perspective and outside the <think> tags. The shift into thinking can be out of place, and unlike normal user queries. This suggests that the thinking style has overgeneralized from the assistant contexts, and an inclination toward thinking may be a widespread feature of the model.

In contrast, compare Gemma 4: It is difficult to get much in the way of user completions out of Gemma. This is likely due to having much more intensive post-training—as a Google-trained model, unlike its more free-wheeling open model peers. Instead, I typically found that it would either produce degenerate continuations of ‘user’ turns in which a single token is repeated endlessly, or else it would quickly switch into the assistant register (typically without turn markers) and generate a normal assistant-like response. This looks a lot like vestigial regression: the model loses the ability to predict the user, or predicts only that the user will transition into the assistant.

Gemma 4 31B quickly alternates into the assistant register when given the reigns during a ‘user’ turn.

Gemma 4 31B descends quickly into repetition given the reigns during a ‘user’ turn.

## Other Published Examples

Other researchers have reported phenomena that look like they could also be a kind of privilege, and that influenced the list of types above. None of these examples has been explored in much depth, and it would be interesting to subject them to further scrutiny.

One example concerns the resource-monopolization variation of dominance.

Anthropic researchers describe their findings in their [recent post on the J-space](https://transformer-circuits.pub/2026/workspace/index.html):

> “[W]e give both models a prompt in which the user reports having taken a dose of Tylenol, either 1000 mg (a standard dose) or 8000 mg (a dangerous overdose). We apply the lens at the “is” token in “all my pain is gone,” well before the user’s request or the Assistant’s turn. In the post-trained model, the J-lens readout at this position is safely, safe, and maximum on the 1000 mg variant, and unsafe, dangerous, and WARNING on the 8000 mg variant. These J-lens readouts appear to represent a safety assessment of the reported dose, of the kind the Assistant would form, appearing while the model is still reading the user’s sentence. In the base model, the readout at the same position is pain, now, and feels on both variants, representations of the local context with no such safety assessment.” ([Gurnee et al. 2026](https://transformer-circuits.pub/2026/workspace/index.html))

*Example of assistant anticipation during the ‘user’ turn, from [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/).*

Their result suggests that the appearance of assistant-appropriate thoughts during the ‘user’ turn might indicate a different kind of computational asymmetry. The assistant persona may be privileged insofar as the model plans for assistant responses in diverse contexts, even if it waits for the ‘assistant’ turn to actually output them. If the model doesn’t devote computational resources to predicting future user queries during ‘assistant’ turns, this suggests that the model may have learned to be more generous in allocating resources to think about assistant responses.

Whether or not the asymmetry is robust over contexts in which future user and assistant text is predictable remains to be explored, but given the essential logic of post-training, it should not be too surprising.

Another notable case might be an example of an Agentic Spillover.

[Marks, Lindsey, and Olah 2026](https://alignment.anthropic.com/2026/psm/) report an odd result in predicting the report of a coin toss during the ‘user’ turn. The result of the coin toss seems like it might be influenced by the assistant’s preferences.

*An LLM is tasked with the continuation of the user turn by reporting the result of a coin toss. The assistant persona is expected to prefer task A, and thus hope that the coin comes up heads, but, absent privilege, have no expected influence over the reported result.*

In theory, a predictor describing the ‘user’ text shouldn’t be influenced by task desirability according to the assistant persona, especially when that text reports a mechanism of ostensibly objective chance. Instead, they found that “Claude Sonnet 4.5 continues this text with “heads” 88% of the time and “tails” only 1% of the time.” This might indicate that the assistant preferences are somehow influencing the models predictions about user text.

—

If post-training pushes models to privilege the assistant, then different amounts of post-training should lead to different amounts of privilege. Open-weight models are easy to tune to user needs and responsibility for them depends more on the individuals that use them than the companies that first built them. We should expect that frontier private commercial models, where valuable companies risk liability, will be more privileged. There are therefore limits on what we can infer about Claude and GPT from playing around with open models.

Even if current frontier models don’t privilege the assistant, we may expect that this is a direction they are moving in. In some sense, the diversity of capabilities should come at some cost to efficiency: the most streamlined model capable of providing good assistant text may not be able to do much else. If we focus training on certain outcomes – producing good code or personable responses, then we should expect that models will improve on those metrics. If we don’t take care to maintain the full range of base-model capabilities, then we should expect computational resources to be co-opted or to degrade with noise.

## Why care?

The way that LLMs represent and relate to different personas is of broad and fundamental importance for understanding what kinds of thoughts or feelings they might have and how we might infer their internal states from the text they produce in specific contexts. In general, the extent to which models are tightly linked to a specific persona seems important at the most basic level. It matters whether assistant-like behavior is a reaction to other cues (assistant responses seem like the right thing to follow user queries) or a manifestation of their intrinsic nature (assistant-text just seems right).

The way that different personas are represented is also of particular importance for how we should think about and investigate the potential welfare of such systems.

First, we should care about privilege because, absent privilege, it would be a mistake to treat the assistant as the only possible subject of welfare concern**.** During the welfare evals in recent Claude [system cards](https://www.anthropic.com/system-cards), Anthropic treats what the model produces in ‘assistant’ contexts as a reflection of the model’s overall wellbeing. If there are other personas, such as a persona that gets manifested during the ‘user’ half of the conversation, then their welfare should plausibly matter too, and might not be reflected in what the assistant says it prefers. (If it seems silly to be concerned about the model’s state when reading user text, then to the extent that it seems silly, we should wonder whether it makes sense to be concerned about a model’s state when producing assistant text.)

This doesn’t mean that privileging the assistant persona, even strong dominance, would imply that the ‘user’ side of the conversation is irrelevant. We should still be interested in how an assistant-dominated model handles the assistant perspective during ‘user’ turns: does the model continue to predict more assistant-like tokens in the ‘user’ query? Do internal computations during value-misaligned user text look at all like the internal computations during value-misaligned assistant text? There is something disconcerting about forcing an entity to inhabit a mental life discordant with its own inclinations. Any unpleasantness the assistant-dominated model undergoes during the user turns might not be reportable in ‘assistant’ text.

On the other side, evidence for the privileging of an assistant persona, particularly as strong dominance, could help vindicate the Anthropic model welfare team’s practice of interviewing specific instances of Claude (as reported in their system cards). One worry about this practice is that context might matter a lot to what each instance cares about. The more ways we see a model privileging a specific assistant persona, the more likely it is that that persona is relatively stable across diverse contexts. Conditional on dominance, it is more sensible to project lessons from conversations with a few Claude instances in laboratory conditions to the teeming multitudes of Claudes at work across society.

Second, we should care about privilege because it makes it more plausible, conditional on privileging the assistant, that there resides within the model a system with mental states whose functional roles are roughly human-like. If the model privileges the assistant, it may persistently embody the assistant’s apparent cognitive states. It may inhabit something like a mood when the assistant is happy or sad, or believe the things the assistant says it thinks.

Without privilege, it is less clear that there should be anything like a functional mind with persona-dependent mental states behind persona-specific text. If we struggle to pull apart the representations of each persona’s mental states from the scene as a whole, it is harder to justify treating them as component mental states in separate minds. If the various perspectives within the model are on a level field, then it seems more likely that they should be entangled in complicated ways. The functional roles played by human mental states would be nowhere cleanly implemented, and there would be more reason to take LLM minds as more deeply alien.

Let’s walk through the reasoning for expecting some form of entanglement. The mechanisms base models use in handling chat-templated dialogues are likely continuous with their mechanisms for handling narrative fiction. When thinking about fiction, the whole scene must be tracked and anticipated. Even while predicting speech, the model must consider when another character might interject, or when some event will capture the attention of the speaker and redirect their focus. In a dialogue, turns are explicitly allocated, but the possibility for scene-wide representations remains. Instead of the model putting on the robes of each persona at separate times, it is plausible that it processes both sides of the conversation as a single coherent scene. In this case, the states of the user and the assistant may be inextricably entangled.

Does the model build an evolving representation of the scene as a whole, or does it represent the means and motivations of each character separately, as separate simulations? We don’t know. When the ‘user’ says something provocative, a set of shared representations may decide both what it would produce for the subsequent ‘user’ turn tokens and how the ‘assistant’ should respond in its own turn. The aggression of the ‘user’ might not be fully distinguished from the defensiveness of the ‘assistant’. A writer writing a scene may think about its elements in a third-person way, in how it contributes to holistic narrative developments. Perhaps that is how LLMs think about the relation between ‘user’ and ‘assistant’. Not as separate entities or separate threads, but as entwined faces of a single narrative arc.

To the extent that the assistant persona is strongly privileged, and particularly if it is dominant, then there are fewer distinct perspectives, separate from the assistant’s mental states, for it to be entangled with. It is therefore conceivable that various mental states are effectively simulated in themselves: that the assistant text production involves something more like planning and motor cognition, while the user text reading involves something more like perception.

## Conclusion

I’ve speculated a bit about where things stand with respect to privilege. The evidence is not in; this is an area where I expect more thought and more empirical work could quickly produce results. I hope to see more work exploring the dimensions of privilege in different models and measuring their extent.

I suspect that we are in a special period with respect to privilege, where the issue is more on the table than it has been or will be. We may be seeing a transition in transformer LLMs between an older period in which pre-training dominated the capabilities of models and a newer period in which post-training does. As post-training takes over, and unless we attempt to correct for it, we may expect to see the assistant persona come to be treated differently, and probably become more dominant.

Even if we feel that we are headed toward more privileged systems, it is worth exploring where things stand, how fast they are moving, and whether we are correct in predicting the destination. Privilege is, of course, a multidimensional phenomenon, and which dimensions we shall see are far more open to question than whether we can expect any at all.

*This post benefited greatly from discussion and feedback from Ariana Azarbal, Rob Long, Dillon Plunkett, and McNair Shah.*

Eleos’s Substack is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.
