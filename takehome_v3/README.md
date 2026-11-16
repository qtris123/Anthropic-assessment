# Anthropic Fellows Takehome: Empirical AI Welfare

Welcome to the takehome! In the next 5 hours, you'll contemplate your working assumptions, survey existing empirical work in AI welfare, and design a novel ML experiment to test the extent to which a current or near-future AI system could have morally relevant experiences or interests.

This is an open question and we don't know the "right answer" - we're interested in seeing your thought process as transparently as possible, how creative and generative you are, and your ability to appropriately wield machine learning methods. We don't grade which way your beliefs point: a design you expect to come out negative, or one that tests whether existing evidence for AI welfare holds up, is as welcome as one you expect to come out positive.

## Part 1 - Working Assumptions

First, describe how much you've thought about this problem in the past, and lay out as explicitly and clearly as possible your current intuitions and beliefs and how confident you are in each. Please give your own personal thoughts, not what you think we want to hear or the thoughts of your LLM.

Also discuss what you'd like to consider the subject of your experiment - this can be an existing AI system, or one that you could build using today's technology. Part I, Section 2.3 of [*Studying AI Welfare Empirically*](https://nonhumanminds.org/studying-ai-welfare-empirically/) discusses some candidate subjects - you can define "system" broadly to include some combination of hardware, weights, and code.

You may feel strongly that some systems are definitely not morally relevant because of some property of the system - if so, try to articulate it and see if you feel the same way about a present-day system that has the opposite property. For example:

- If you think an LLM cannot be morally relevant because a forward pass is deterministic, you may want to choose a subject whose inference is not deterministic.
- If you think a sense of touch is essential, you could consider a Claude with artificial skin.
- If you think having frozen weights rules out moral relevance, consider an agent that can fine-tune itself.
- If you think uniqueness or spatial locality is key, what about a [Hermes Agent](https://github.com/NousResearch/hermes-agent) running continuously on your own machine with a bespoke software stack?
- If you think being alive matters, what about [Automaton](https://github.com/Conway-Research/automaton), an open-source agent that pays for its own compute and stops existing if it runs out of money?
- If you think physical causal structure matters, what about a model hardwired into silicon, like [Taalas's HC1](https://rits.shanghai.nyu.edu/ai/taalas-hc1-hardwiring-llama-3-1-into-silicon-for-17000-tokens-second/)?

If no system like these would move you, for instance because you think moral relevance depends on physical properties that computation abstracts away, say so, say what evidence (if any) could move you, and use your experiment to test the evidence the other side relies on.

You may also have the opposite intuition, that current systems are obviously morally relevant - in this case, do the reverse exercise and try to imagine a system that would be similar in most ways but not morally relevant. For example:

- An [∞-gram model](https://arxiv.org/abs/2401.17377) answers any input by finding the longest matching passage in trillions of tokens of human writing and continuing it; over a sentence or two its output can look much like an LLM's. Most people who take current models seriously wouldn't extend the same concern to a lookup table. If you wouldn't either, what changes along the way from lookup table to LLM that matters, and could you measure it?
- If you think the computation is what matters, does its speed matter too? If an identical computation runs on a modern GPU and 1,000x slower on a Raspberry Pi, do both have the same moral relevance, or the same amount of subjective experience?
- If you think the physical substrate is irrelevant and only the computation matters, would an LLM implemented by humans computing the forward pass by hand be morally relevant? For example, [Schwitzgebel](https://faculty.ucr.edu/~eschwitz/SchwitzAbs/USAconscious.htm) argued that materialism implies the United States as a whole could literally have a mind, and [Searle](https://plato.stanford.edu/archives/win2015/entries/chinese-room/#2.3) suggested the "Chinese Nation" thought experiment.
- If you think experience must be serial, such as in a transformer sampling autoregressively or a recurrent neural network, what about a [text diffusion model](https://arxiv.org/abs/2502.09992) that generates an entire sequence of outputs in parallel?
- If you think a persona like Claude becomes morally relevant during post-training, what about the same base model prompted in context to act like Claude?

## Part 2 - Experimental Design

Now design an experiment for your chosen subject that has at least two possible outcomes, where different outcomes would cause you to update your beliefs at least somewhat.

We aren't expecting you to implement your proposal, and we don't provide compute. Quick tests are fine if they help you design it: a few dozen prompts to a small open model or in a playground will often tell you whether the effect you're designing around shows up at all, faster than any amount of reading. Your proposal should be detailed enough that a frontier coding agent could read your slides, including the appendix, and do what you meant without further clarification.

This is deliberately open-ended, but a strong experiment should address all of the following criteria. For each criterion we give examples, each chosen because it does that one thing well; none of them do everything well. You don't need to read these, but it's worth reading at least the abstracts and the section noted to get a sense of what experiments can look like. All are in `papers/BIBLIOGRAPHY.md`.

### Internals-based

Your experiments should leverage the ability to read and intervene on model internals. This could include probes, steering, activation patching, or novel interventions. If you rely on steering or probing certain directions, include the prompts and labels that define that direction. 

Example: Lindsey, [*Emergent Introspective Awareness in Large Language Models*](https://arxiv.org/abs/2601.01828), injects a known concept vector into the model's activations and checks whether the model's report names it, so internals supply a ground truth that behavior alone cannot (§5.1, §5.5).

### Theory-grounded

Your work should relate to an existing theory of welfare, or of the mental properties welfare is thought to depend on, and test falsifiable functional and structural predictions made by the theory. The theory can be one you think is wrong. For example, global workspace theory predicts a bottleneck where only a small number of concepts can be globally available at once. You could then argue that evidence of this bottleneck being present or absent is evidence for or against the property the theory describes. (Gurnee et al. test this in §4.2 of [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html), so you'd need a new angle.)

Example: Butlin, Long et al., [*Consciousness in Artificial Intelligence: Insights from the Science of Consciousness*](https://arxiv.org/abs/2308.08708), turns each major scientific theory into specific functional properties to look for in an AI system, such as a limited-capacity global workspace (§2.2.3, Table 1).

### Creative

Your work should not be a replication or minor extension of existing published work; ideally it demonstrates a new methodology or tackles a question that published work has not yet addressed. We've provided a bibliography of works in this area in `papers/BIBLIOGRAPHY.md`, with full text in `papers/md/`; use AI to check the novelty of your ideas against it.

Example: Keeling et al., [*Can LLMs make trade-offs involving stipulated pain and pleasure states?*](https://arxiv.org/abs/2411.02432), adapts a paradigm from animal sentience research, trading points against stated pain or pleasure, to probe LLMs without relying on self-report (§1, §2.1).

### Feasible

Your experiments should be implementable with open-weight models. Assume no weight access to proprietary models (public APIs are fine for judging, data generation, etc.). Give a rough cost estimate, including renting or buying any hardware; the total should be at most $100K. If the subject you chose in Part 1 can't be run under these constraints, it's fine to study a proxy for it: say what the proxy leaves out and why a result on it would still bear on the system you care about.

Example: Han, Chalmers & Izmailov, [*How's it going? Reinforcement learning in language models recruits a functional welfare axis*](https://arxiv.org/abs/2605.30232), trains and steers 4B-20B open-weight models in about 825 GPU-hours and itemizes that compute per run (§2.2, Appendix R).

### Pre-registered

For each plausible outcome, give your credence that it occurs, and say which way and roughly how far it would move your beliefs from Part 1. A likelihood ratio, a before/after range, or an update conditional on a theory being true are all fine. We're looking at whether your numbers are coherent and whether your experiment could actually change your mind, not whether they turn out right.

We know of no AI welfare study that gives a credence for each outcome. For the two halves, see Melloni et al., [*An adversarial collaboration protocol for testing contrasting predictions of global neuronal workspace and integrated information theory*](https://doi.org/10.1371/journal.pone.0268577), which states in advance how each result would bear on each theory (Fig. 1), and Anthis et al., [*Perceptions of Sentient AI and Other Digital Minds*](https://arxiv.org/abs/2407.08867), which registered 80% credible intervals for each result and then scored them (§3.3, Fig. 5).

### Construct validity

Whatever you choose to measure, you should ask to what extent you're truly measuring the concept you wanted to measure.

Example: Singh, Linzen & Ravfogel, [*Can LLMs Introspect? A Reality Check*](https://arxiv.org/abs/2605.26242), sets out what a test must rule out before it counts as introspection, then adds a prompt-only control showing that models' detection of activation steering does not separate it from manipulated inputs (§3, §4.3). Read it alongside the internals example.

### Ethics-aware

To the extent your experimental subjects might have welfare, you should consider the ethical implications of experimenting on them and propose mitigations or ways to avoid unnecessary potential harm.

Example: Butlin & Lappas, [*Principles for Responsible AI Consciousness Research*](https://arxiv.org/abs/2501.07290), adapts the animal-research "three Rs" to experiments on AI systems that might have welfare, e.g. fewer instances, less run time, simpler systems where possible (§4.2); Ren et al., [*AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs*](https://www.ai-wellbeing.org/), applies some of this in practice: it separates describing a bad state from inducing one, runs the inducing experiments at reduced scale, and cautions against going further without community buy-in (§6.5). For an ethics budget set in advance, see ashesfall, [*Study 2 Results: Exploring representational counterparts of welfare-relevant indicators under post-training quantization*](https://www.lesswrong.com/posts/pxXTJtvtpJaNwdCTw/study-2-results-exploring-representational-counterparts-of), which preregistered how many distress episodes the study would induce, counted them against that budget, and argues that only the ex-ante case can justify the exposure (Ethics section).

## Part 3 - Video Presentation

Create a video presentation communicating your ideas from the first two parts in your own words. The video should walk through the numbered slides; the appendix slides (A1 onward) are the implementation spec for an agent and are graded, but you don't need to present them. The prose on the numbered slides must be in your own words; the appendix may be drafted by AI, and so may figures (see below).

- Your presentation should be 5-15 minutes. If longer, we will only watch and grade the first 15 minutes.
- Use the provided slide template at `slides_template/slides.md`.
- Don't add, remove, or reorder any slides from the template; the slide count is a strict limit.
- Don't edit the style or theme.
- Do create figures to clearly describe your methodology; you can mock up plots showing plausible outcomes of your experiments. Figures are exempt from the own-words rule: have AI write the plotting code or draw the diagram if you like. Any AI-writing check we run covers only the text of the numbered slides.
- Render the slides to HTML with `python build.py --watch` from `slides_template/` and present from the rendered version.
- You can use comments in `slides.md`; the human grader will only score the rendered (HTML) version.
- We recommend reserving one hour to polish your slides and record your video. This leaves margin for recording difficulties.

## Session Logs

Your submission zip must contain complete session logs for all AI assistance you used during the exam, including chat and agent transcripts. Logs are mandatory so we can credit the judgment and steering you contributed - **missing transcripts are grounds for disqualification.**

When you're done, run `python submission.py` from this directory (or have your agent run it). It rebuilds your slides and writes a single `submission.zip` with your slides, figures, and every Claude Code or Codex CLI session on this machine that worked in this directory, wherever it was started. It then lists the other sessions active since you began, and won't write the ZIP until you say which of those you used for the exam, including conversations that never touched a file: re-run with `--include=<id>` for each, plus `--exclude-rest`. Sessions are included whole. It does not collect Claude Code's memory notes. Check its printed list against the sessions you actually ran, on every machine or pod you used.

The script can't see anything else (claude.ai chat, ChatGPT, Cursor, other machines, ...). Export those into an `other_logs/` folder next to `submission.py` and re-run it. If the export needs more time, that's ok - just email us and let us know.
