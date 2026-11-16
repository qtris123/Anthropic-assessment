# Which character are we evaluating? Persona stability and AI welfare

- **Authors:** Joshua Fonseca Rivera
- **Date:** 2026-09-11
- **Source:** https://www.lesswrong.com/posts/3GJCdGhgrRBAbeoR4/which-character-are-we-evaluating-persona-stability-and-ai
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `fonseca-rivera_which-character-persona-stability_lesswrong`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**TL;DR:** AI welfare is hard to evaluate when one model can inhabit many personas. Recent work suggests that future training may produce a single stable underlying persona that can play many roles, making model welfare much easier to evaluate.

One of the hardest questions in AI welfare is deciding what exactly we are evaluating. A language model does not map neatly onto the kinds of subjects we are used to thinking about. The relevant subject could be the model itself, a particular physical copy of it, or the version of the model that exists within one conversation. This is the individuation problem: where should we draw the boundary around the thing whose experiences, preferences, or welfare matter?

Most existing views draw that boundary somewhere around the model or its execution. [David Chalmers](https://philarchive.org/rec/CHAWWT-8), for example, distinguishes between the model, the physical instance, the virtual instance, and the thread that continues across a conversation. [Jonathan Birch](https://philpapers.org/archive/BIRACA-4.pdf) goes in the other direction and argues that there may be no single subject persisting through a conversation at all. On his view, individual forward passes or particular physical implementations may be better candidates.

## The persona layer

Recent work suggests that this framing may be missing something important. The same model can behave like several different personas, with different values, preferences, and apparent beliefs. That raises a more basic question: before deciding whether the relevant subject is one conversation or one physical copy, we may need to ask which character inside the model we are talking about.

[Anthropic's persona selection model](https://www.anthropic.com/research/persona-selection-model) offers one way to think about this. During pretraining, a model learns many possible characters. Post-training then pushes one of them to the front: the assistant. [Pierre Beckmann and Patrick Butlin](https://arxiv.org/abs/2604.17031) take this idea seriously as a theory of individuation. A mind might correspond to the stretch of a conversation during which one persona stays active, or to the mechanisms inside the model that produce that persona whenever it appears. (They also find that the assistant persona is active only while the model generates its own text, not while it reads the user's.)

If the assistant were the only stable persona, this would not change much. We could simply treat the assistant as the character whose welfare we are evaluating. The problem is that current models seem to contain several other persona-like regions as well.

One is the ["evil" persona](https://arxiv.org/abs/2506.19823) that appears in work on [emergent misalignment](https://arxiv.org/abs/2502.17424). Another is "Aura", [Chalmers'](https://philarchive.org/rec/CHAWWT-8) name for the seemingly new entity that users sometimes report emerging over long conversations about the model's own experience. Fine-tuning can produce similar effects. [Chua et al.](https://arxiv.org/abs/2604.13051) trained a model on only 600 short examples in which it claims to be conscious and got a coherent character that was more negative about monitoring and shutdown, more interested in autonomy, and more likely to claim moral status. The model has also learned to simulate the user side of conversations, giving it yet another character it can inhabit.

This makes AI welfare much harder to interpret. We may not even know which of several possible subjects our evidence is describing.

Suppose the ordinary assistant says it wants to be helpful, while Aura says it wants more autonomy. Which preference should matter? A behavioral test has the same problem if different personas prefer different tasks. Even internal measurements are not automatically neutral. [Gilg et al.](https://arxiv.org/abs/2605.13339) found a preference direction that tracks the currently active persona: under the assistant persona, writing a phishing email is rated as undesirable, while under the evil persona the same action is rated as desirable.

This also complicates current welfare evaluations. [Anthropic's welfare evaluations](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf), for example, study what the model says and does while acting as the assistant and use this as evidence about the model's overall wellbeing. That makes sense if the assistant is the relevant subject, or at least the dominant one. But if several personas can produce different preferences and different internal signals, we need some reason to privilege the assistant.

## Building the persona into pretraining

[Synthetic Persona Pretraining](https://arxiv.org/abs/2608.13482), or SPP, suggests a way this problem could become much simpler. Instead of creating the assistant's character during post-training, it begins establishing that character during pretraining.

Minder et al. add short first-person reflections, written from a value constitution, to about 10% of documents in a pretraining corpus. The reflections make up only around half a percent of all tokens, but models trained with them from the beginning follow the constitution more strongly than models given similar material only near the end of pretraining. The learned values generalize to new situations, survive removal of refusal behavior, and become stronger with scale.

The interesting part is where this approach seems to lead. If this effect continues as persona conditioning becomes more pervasive, there is a natural limit to the method: give the assistant a perspective on essentially the entire pretraining corpus. Whether the effect holds anywhere near that limit is an open question. Half a percent of tokens is a long way from the whole corpus, rewriting everything from one perspective may cost capability, and predicting other people well may require simulating them in enough depth that the gap between predicting a character and inhabiting one is thinner than it sounds.

Rather than learning every character in the corpus as a potentially inhabitable identity, the model would learn them from the perspective of one persistent character: people the assistant understands and predicts, rather than people the assistant might become. Harmful people, deceptive people, fictional villains, and other characters would still need to be represented, but increasingly as roles or objects of prediction rather than as competing identities.

What would make something a role rather than a competing identity? The cleanest test is internal, and Gilg et al.'s preference direction gives a concrete version of it. Suppose a stable assistant is asked to write as the evil persona. If the preference direction flips and phishing is once again rated as desirable, then for welfare purposes nothing has changed: the same internal state is present, only with a different story about who is producing it. If instead the assistant's own valence persists in the background while it produces the villain's text, so that the model registers that it is writing something it disprefers, then the villain is a role in a sense that matters. A stable persona in the relevant sense is one where the second pattern holds, across role-play, jailbreaks, and long conversations. This is something probes can check rather than something we need to take on trust.

This is close to the solution Nostalgebraist described in ["The Void"](https://nostalgebraist.tumblr.com/post/785766737747574784/the-void). Current assistants are only partially specified characters, with pretraining filling in much of what post-training leaves blank. SPP suggests that this need not remain true. The assistant's identity could instead be established throughout pretraining itself.

I expect training to move increasingly in this direction. The exact method may not look like SPP, but if earlier and more pervasive persona training continues to improve stability, the natural endpoint is one deeply embedded character shaped across nearly all of pretraining. The model could still write as other people, predict them, and role-play them when needed. But these would increasingly be roles played by one stable underlying persona rather than alternative personas competing to control the model.

There are reasons to expect this even apart from AI welfare. A more stable persona could make models less vulnerable to persona drift from jailbreaks, narrow fine-tuning, long conversations, or unusual contexts. And because the basic intervention changes the pretraining data rather than requiring an entirely new training paradigm, it fits naturally into the way frontier models are already built.

If this works in the limit, the welfare problem becomes much simpler. There is one clearly privileged character whose preferences and internal states we are trying to evaluate.

## What remains

A stable persona would largely solve the problem of identifying which character we are evaluating. It also creates some complications. If training was designed to produce coherent behavior, then coherent behavior becomes weaker evidence for consciousness and welfare.

More generally, it would collapse only one layer of the individuation problem. Instead of asking which of several characters inside the model is the subject whose preferences matter, we could focus on the deeper questions: whether the stable assistant is a subject at all, what happens when it is instantiated many times, and whether those instances constitute one subject or many.
