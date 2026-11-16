# Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents

- **Authors:** Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune
- **Date:** 2025-05-29
- **Source:** https://arxiv.org/abs/2505.22954
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `zhang_darwin-godel-machine_2505.22954`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Published as a conference paper at ICLR 2026

## Darwin gödel machine: open-ended evolution

**Jenny Zhang** **[*,1,2]** **Shengran Hu** **[*,1,2,3]** **Cong Lu** **[1,2,3]** **Robert Lange** **[†,3]** **Jeff Clune** **[†,1,2,4]**

1University of British Columbia 2Vector Institute 3Sakana AI 4Canada CIFAR AI Chair
{jennyzzt,srhu,conglu}@cs.ubc.ca, robert@sakana.ai, jeff.clune@ubc.ca

ABSTRACT

Most of today’s AI systems are constrained by human-designed, fixed architectures
and cannot autonomously and continuously improve themselves. The scientific
method, on the other hand, is a cumulative and open-ended system, where each
innovation builds upon previous artifacts, enabling future discoveries. There is
growing hope that the current manual process of advancing AI could itself be
automated. If done safely, such automation would accelerate AI development and
allow us to reap its benefits much sooner. This prospect raises the question of
how AI systems can endlessly improve themselves while getting better at solving
relevant problems. Meta-learning can automate the discovery of novel algorithms,
but is limited by first-order improvements and the human design of a suitable search
space. The Gödel machine (Schmidhuber, 2007) proposed a theoretical alternative:
a self-improving AI that repeatedly modifies itself in a provably beneficial manner.
Unfortunately, proving that most changes are net beneficial is impossible in practice.
We introduce the Darwin Gödel Machine (DGM), a novel self-improving system
that iteratively modifies its own code (thereby also improving its ability to modify
its own codebase) and empirically validates each change using coding benchmarks.
Inspired by Darwinian evolution and open-endedness research, the DGM grows
an archive of generated coding agents. It samples agents from this archive, which
self-modify to create new, interesting versions of themselves. This open-ended
exploration forms a growing tree of diverse, high-quality agents and allows the
parallel exploration of many different paths through the search space. Empirically,
the DGM automatically improves its coding capabilities (e.g., better code editing
tools, long-context window management, peer-review mechanisms), increasing
performance on SWE-bench from 20.0% to 50.0%, and on Polyglot from 14.2% to
30.7%. Furthermore, the DGM significantly outperforms baselines without self-improvement or open-ended exploration. All experiments were done with safety
precautions (e.g., sandboxing, human oversight). Overall, the DGM represents a
significant step toward self-improving AI, capable of gathering its own stepping
stones along a path that unfolds into endless innovation. All code is open-sourced
[at https://github.com/jennyzzt/dgm.](https://github.com/jennyzzt/dgm)

1 INTRODUCTION

Scientific progress is cumulative and open-ended, with each breakthrough standing on the shoulders
of countless prior insights. In the same way, our most advanced AI systems are built upon a long
lineage of innovations. For instance, transformers (Vaswani et al., 2017), the backbone of current large
language models (LLMs) (Brown et al., 2020), did not emerge in isolation but were built upon years
of past innovations, such as recurrent neural networks (Linnainmaa, 1970; Amari, 1972; Hopfield,
1982; Rumelhart et al., 1985) and attention mechanisms (Schmidhuber & Huber, 1990; Bahdanau
et al., 2015; Kim et al., 2017; Parikh et al., 2016). However, most of today’s AI systems remain bound
by fixed, human-designed architectures that learn within predefined boundaries, without the capacity
to autonomously rewrite their own source code to self-improve. As a result, each advancement in AI
development still leans heavily on human interventions, tethering the pace of progress. This paper

*co-first authors   - co-senior authors investigates the intriguing possibility of safely automating the search for ever-better AI. One can
imagine an AI system that, like scientific discovery itself, becomes an engine of its own advancement:
building upon its past, recursively improving, and propelling itself toward more advanced capabilities.

Figure 1: **Darwin** **Gödel** **Machine.** The DGM iteratively builds a growing archive of agents by
interleaving self-modification with downstream task evaluation. Agents in the archive are selected for
self-modification through open-ended exploration.

Schmidhuber (2007) presented a class of mathematically rigorous, self-referential, self-improving
problem solvers. It relies on formal proofs to justify code rewrites, ensuring that any self-modification
is provably beneficial. However, in practice and without restrictive assumptions about the system,
it is impossible to formally prove whether a modification to an AI system will be beneficial. For
example, while it may seem that an LLM-based coding agent would benefit from access to more
tools (e.g., code search, test runners), the actual impact depends heavily on the model’s training and
task context (e.g., a testing tool that is optimized for one setup may confuse the agent when working
with others). Instead of requiring formal proofs, we empirically validate self-modifications against a
benchmark, allowing the system to improve and explore based on observed results. This approach
mirrors biological evolution, where mutations and adaptations are not verified in advance but are
produced, trialed, and then selected via natural selection. We also take inspiration from Darwinian
evolution (Darwin, 2023) and investigate the effectiveness of maintaining a library of previously
discovered agents to serve as stepping stones for future generations.

We propose the **Darwin** **Gödel** **Machine** **(DGM)**, a self-referential, self-improving system that
writes and modifies its own code to become a better coding agent. Each self-modification requires
the DGM to edit its own codebase. We use Python, which is Turing-complete, giving the DGM
the potential to build any computable machine. Our framework envisions agents that can rewrite
their own training scripts (including training a new foundation model (FM)). However, we do not
show that in this paper, as training FMs is computationally intensive and would introduce substantial
additional complexity, which we leave as future work. Instead, this paper focuses on improving the
design of coding agents with frozen pretrained FMs (e.g., tool use, workflows). The DGM alternates
between self-modification and evaluation phases. During the self-modification phase, selected coding
agents from the archive generate modified versions of themselves. During the evaluation phase,
each modified agent is tested on a coding benchmark, estimating the agent’s coding capabilities, and
then added to the archive. By improving its own capabilities through this loop, the DGM becomes
better at both solving coding tasks and making future self-improvements. A key assumption is
that an increase in performance on coding benchmarks indicates better coding capabilities, and
hence better ability to self-modify and self-improve. Furthermore, the DGM maintains an archive of
generated coding agents, initialized with only one agent, and continuously accumulates all generated
variants over time. To support continual self-improvement, the DGM draws inspiration from openendedness research (Wang et al., 2019; Fernando et al., 2024; Faldor et al., 2025), accumulating
diverse stepping stones (i.e., interesting yet suboptimal solutions or features that may enable future
breakthroughs). This open-ended exploration encourages the discovery of novel and potentially
useful self-modifications beyond immediate performance gains.

We present results on two coding benchmarks: SWE-bench (Jimenez et al., 2024) and Polyglot (Paul
Gauthier, 2024). The DGM automatically improves itself from 20.0% to 50.0% on SWE-bench,
and from 14.2% to 30.7% on Polyglot. We show that self-improvement enables continued progress,
as the DGM outperforms the baseline where the same base agent is repeatedly used to modify and generate new agents without self-improvement. We also show that open-ended exploration and
keeping an archive of all previously generated agents lead to the discovery of better coding agents.
The DGM outperforms the baseline of not having open-ended exploration (i.e., a baseline without the
accumulation of an archive of interestingly different stepping stones), where the coding agent always
builds off the most recent version of itself. Overall, the DGM represents a step toward AI systems
that can build upon their own prior innovations and improve recursively. We consider and discuss
safety aspects extensively, including sandboxing and traceability of self-modifications, to ensure
responsible experimentation (Section 5). By advancing the possibility of safe, self-referential, self-improving models, the DGM moves us closer to AI that not only learns but evolves in an open-ended,
self-accelerating trajectory, much like science itself.

2 RELATED WORK

**Open-Endedness.** A grand challenge for driving unbounded innovation is designing open-ended AI
systems that continuously generate novel and learnable artifacts (Stanley et al., 2017). Hughes et al.
(2024) characterized open-endedness as a system’s capacity to generate sequences of artifacts that are
both novel and learnable from an observer’s perspective. A central difficulty lies in structuring and
exploring vast search spaces to consistently produce artifacts that are interesting to humans (Clune,
2019; Jiang et al., 2023). Early progress drew on quality-diversity algorithms, goal-directed exploration, intrinsic motivation, and learning-progress frameworks (Pugh et al., 2016; Ecoffet et al.,
2019; Lehman & Stanley, 2011; Oudeyer et al., 2007), while recent advances leverage large-scale
foundation models (FMs) as proxies for human interestingness and versatile engines for generating
and evaluating novel behaviors across diverse domains (Brown et al., 2020; Hu et al., 2025; Zhang
et al., 2024b). However, these approaches have yet to close the self-referential self-improvement
loop, meaning improvements on downstream tasks do not translate into enhanced capabilities for
self-modification or the acceleration of further innovations. We aim to mimic the acceleration of science and technology, where new tools and discoveries catalyze the creation of even more discoveries.
How can we emulate nature’s arc of evolution, which bends not only toward complexity but also an
ever greater capacity to evolve (Dawkins, 2019; Gerhart & Kirschner, 2007; Hendrikse et al., 2007)?

**Meta-Learning FM Agents.** Many FM-based agents are handcrafted. Some building blocks include
prompt engineering (Chen et al., 2023; Schulhoff et al., 2024), chain-of-thought (Wei et al., 2022;
Yao et al., 2023; Hu & Clune, 2024; Guo et al., 2025; Lightman et al., 2023; Muennighoff et al., 2025;

Zelikman et al., 2024a), self-reflection (Shinn et al., 2023; Yao et al., 2023; Madaan et al., 2023),
multi-agent debate (Zhuge et al., 2023; Liang et al., 2023; Khan et al., 2024), memory (Liu et al., 2023;
Zhong et al., 2024; Modarressi et al., 2023), temperature sampling (Zhu et al., 2024), and retrieval
augmented generation (Lewis et al., 2020). The manual composition of these components limits
the system’s abilities to the ingenuity of its human designer. More recently, several meta-learning
approaches have emerged that leverage FM to automatically optimize prompts (Fernando et al., 2024;
, FAIR; Khattab et al., 2023; Cheng et al., 2024; Yuksekgonul et al., 2024; Yuan et al., 2024) and
design agentic modules (Zhang et al., 2024c; Zhou et al., 2024; Yin et al., 2024; Zhuge et al., 2024;
Rosser & Foerster, 2025; Zhang et al., 2025a; Ye et al., 2025; Gao et al., 2025; Nie et al., 2025;
Su et al., 2025; Zhang et al., 2025b; Niu et al., 2025). The Automated Design of Agentic Systems
(ADAS, Hu et al., 2025) iteratively generates downstream agents with a fixed meta-agent, evaluates
them against a target benchmark, and incorporates feedback to refine subsequent generations. In
contrast, the DGM is a single system that both solves downstream tasks (i.e., coding problems)
and refines its own implementation (i.e., its codebase), removing the need for a fixed, handcrafted
meta-agent and enabling self-referential improvements.

**Self-Improving AI.** Early on, various researchers outlined theoretical and conceptual approaches to
self-improvement (Good, 1966; Schmidhuber, 1987; 2007). Some practical approaches to automated
self-improvement include systems defined by neural network weight parameterizations (Schmidhuber,
1993; Hall, 2007; Hobbhahn, 2025; Kirsch & Schmidhuber, 2022; Irie et al., 2022; 2025; Lu et al.,
2023; Havrilla et al., 2024b). Metz et al. (2021) developed a gradient-based optimizer that is selfreferentially meta-trained using a variant of population-based training (Jaderberg et al., 2017). Lange
et al. (2023) extended this approach to gradient-free learning. Silver et al. (2017) used self-play
to continuously evolve agents, achieving superhuman performance in challenging domains such
as chess and Go. More closely related to the DGM are recent approaches that leverage FM-based
agents for self-improvement (Yin et al., 2024; Robeyns et al., 2025; Hu et al., 2024; Zelikman et al., 2024b; Huang et al., 2022; Singh et al., 2023). Zelikman et al. (2024b) use a meta-agent
to generate downstream agents, updating the meta-agent based on the meta-utility derived from
the generated solutions. Yin et al. (2024) use a single system to both solve downstream tasks and
recursively modify itself. However, the downstream tasks or the meta-utility do not always align
with the capabilities required for self-improvement. In the DGM, improvement in downstream tasks
directly reflects an increase in self-improvement ability, enabling the potential for self-accelerating
progress. Most similar is concurrent work by Robeyns et al. (2025), which also has a single agent
recursively solving coding problems and modifying its own codebase. The main difference from
Robeyns et al. (2025) (and also Zelikman et al. (2024b); Yin et al. (2024)) is that the DGM has an
open-ended exploration loop, encouraging self-modifications beyond immediate performance gains
and thus avoiding stagnation in suboptimal states. Appendix B also discusses additional related work
on program synthesis and Darwinian evolution.

3 DARWIN GÖDEL MACHINE

A Gödel Machine is a theoretical idea of an AI that searches for ways that _provably_ improve
itself (Schmidhuber, 2007). In this paper, we propose Darwin Gödel Machine (DGM), an attempt to
realize the long-held dream of creating a Gödel Machine. The DGM relaxes the Gödel Machine’s
impractical requirement of theoretically _proving_ that a change will improve the system, instead
requiring _empirical evidence_ from experiments to demonstrate that a proposed new version enhances
performance. Additionally, since the DGM relies on empirical evidence of improvement, it may
get stuck in a local optimum within the vast search space of possible systems (i.e., all computable
algorithms). To address this, the DGM maintains an archive of discovered solutions during the
search, facilitating open-ended exploration rather than relying on evolving a single solution. Since the
principles echo Darwinian evolution (Darwin, 2023) (Appendix B), where new innovations emerge
by selecting an entity from an archive of previously discovered solutions, modifying it, and keeping
it if it is interestingly new (Zhang et al., 2024b; Faldor et al., 2025; Stanley & Lehman, 2015), we
call our algorithm a Darwin Gödel Machine (Figure 1).

**Self-referential Self-improvement of Coding Agents.** The DGM is initialized with only one coding
agent, and its progression is evaluated on coding benchmarks. A coding agent is defined as a single
system, implemented with a code repository and powered by frozen pretrained foundation models
(FMs), capable of reading, writing, and executing code. Code, when expressed in a general-purpose
Turing-complete language (e.g., Python), is a powerful medium for building and improving intelligent
systems because it can represent any computable process. Recent works (Hu et al., 2025; Zhang et al.,
2024c) demonstrate that such agents can be improved through meta-learning of their designs (e.g.,
prompts, workflows, and tools), which are implemented in their code repository. Therefore, we define
self-improvement as a coding task that involves modifying the design of an agent’s own components
(i.e., its own code, which does not include the open-ended exploration process described in the next
paragraph). The key motivation is that the empirical evidence must reflect the system’s ability to both
self-improve and solve downstream tasks. By configuring the DGM as a coding agent and testing
its coding capabilities, the observed improvements demonstrate not only enhanced performance
in downstream tasks but also the capacity for further self-improvement, as self-improvement is
fundamentally a coding task that modifies the coding agent’s own code repository.

**Population-based Open-ended Exploration.** Starting from a single initial coding agent, the DGM
builds an archive of all discovered agents. In each iteration, the DGM selects parent agents to selfmodify and branch off to produce new agents. Parent selection is roughly proportional to each agent’s
performance score and inversely proportional to the number of its children with codebase-editing
functionality (Appendix C.2). This favors high-performing agents that have been underexplored
(i.e., have fewer existing children), thereby promoting both exploitation of strong performers and
exploration of promising but less-sampled lineages. All agents retain a non-zero selection probability,
ensuring that any path to improvement remains feasible given sufficient compute. Each selected
parent analyzes its own benchmark evaluation logs, proposes the next feature to implement, and
receives this proposal as a problem statement to execute (Appendix C.3). The parent then implements
the suggested feature into its own codebase, generating a new coding agent. Each newly generated
agent is quantitatively evaluated on a chosen coding benchmark to estimate its coding abilities. Only
agents that compile successfully and retain the ability to edit a given codebase are added to the
DGM archive, as only they can continue self-modification. All others are discarded. The cycle of parent selection, self-modification, and evaluation continues, progressively growing the archive of
solutions. Importantly, we note that archived solutions can serve as stepping stones that result in
improvements much later than their original discovery, making our approach substantially different
from hill-climbing agentic design approaches (Robeyns et al., 2025). Currently, the open-ended
exploration process (i.e., archive maintenance, parent selection) is fixed and not modifiable by the
DGM, which we leave as an avenue for future work. Appendix C.4 shows the pseudocode for the
DGM algorithm.

4 EXPERIMENTS

Section 4.1 describes the experimental setup, including the initial coding agent that initializes the
DGM. We conduct experiments on two coding benchmarks: SWE-bench (Jimenez et al., 2024) and
Polyglot (Paul Gauthier, 2024) (Section 4.2). For each benchmark, we compare the DGM against
two baselines: DGM without self-improving agents and DGM without open-ended exploration
(Section 4.3). Across all experiments, we find that the DGM outperforms both baselines, showing that
the self-improvement mechanism and open-ended exploration are essential for sustained performance
gains (Section 4.4). Furthermore, we show that the features discovered by the DGM transfer across
models, benchmarks, and tasks (Section 4.4).

4.1 EXPERIMENT SETUP

The DGM is initialized with a single coding agent. This base agent is built around a frozen FM and
augmented with tool use capabilities (Schick et al., 2023; Anthropic, 2024b). The FM autonomously
determines its action at each step, using whichever tool it deems appropriate. The base agent has
access to two tools: a Bash tool for executing bash commands, and an edit tool for viewing and
editing directories and entire files (Appendix C.1). The agent receives a single input prompt that
specifies the location of the target repository, the problem statement, and the repository’s testing
framework (Appendix C.1). The base agent is intentionally lightweight to provide a simple starting
point from which self-improvement can be clearly demonstrated. We run the DGM for 80 iterations
(generating one new agent per iteration), with two iterations running in parallel for SWE-bench
and four for Polyglot (Appendix C.2). During self-modifications, coding agents are powered by
Claude 3.5 Sonnet (New) (Anthropic, 2024a) in both SWE-bench and Polyglot experiments. During
benchmark evaluation, coding agents are powered by Claude 3.5 Sonnet (New) for SWE-bench and
o3-mini (OpenAI, 2025) for Polyglot (Appendix D.1).

4.2 BENCHMARKS

We evaluate the DGM on two popular benchmarks that assess different aspects of coding tasks
to validate the algorithm’s effectiveness across various use cases. Both SWE-bench and Polyglot
are benchmarks that evaluate how well AI agents solve coding tasks automatically. Given a code
repository and a task instruction, the agent is expected to make changes to the repository in order to
fulfill the task. Both SWE-bench and Polyglot are widely used benchmarks (Zhang et al., 2024d;a;
Xia et al., 2024; Cao et al., 2024; Google DeepMind, 2025; Gauthier, 2024) that require the AI agent
to navigate a code repository, understand the interplay between functions in different files, and spot
small errors in convoluted code. SWE-bench only has Python tasks, while Polyglot has tasks in
multiple programming languages. Another difference is that each SWE-bench task may require edits
to multiple files, whereas each Polyglot task primarily involves implementing a solution from scratch
in a single file (although the agent still needs to examine other files to understand what changes are
necessary), resulting in fewer file edits overall.

**SWE-bench.** To avoid wasting compute on unsolvable tasks, we use SWE-bench Verified (OpenAI,
2024), a human-filtered subset of SWE-bench (Jimenez et al., 2024) where all tasks are solvable.
Throughout this paper, the term SWE-bench refers by default to to the SWE-bench Verified subset.

**Polyglot.** Polyglot includes tasks in multiple programming languages (C++, Rust, Python, etc.) (Paul
Gauthier, 2024). Compared to SWE-bench, one of the most widely used coding benchmarks and
likely included in the training sets of FMs, Polyglot is more niche and less likely to be included
in FMs’ post-training data. Additionally, Polyglot is primarily used by its developer to evaluate
Aider (Gauthier, 2024). This provides an opportunity to compare automatically designed agents with a representative coding agent in the community, which has been continuously developed and tested
against the Polyglot benchmark over a year by human developers. Finally, Polyglot enables testing of
the DGM’s ability to generalize across multiple programming languages beyond just Python. We
follow the Polyglot leaderboard setup (Paul Gauthier, 2024), with one key difference: the leaderboard
reports pass@2, where the agent can view feedback from ground-truth tests once, whereas we report
pass@1, where the agent never sees ground-truth test results.

Since it is expensive to evaluate on the full benchmarks (Appendix E.1), we use a staged evaluation
strategy to estimate the coding agent’s performance. In general, we evaluate the coding agent on
more tasks when we are more confident that it is a relatively strong performer. We first evaluate
each coding agent on a set of 10 tasks to verify basic codebase-editing functionality. Only agents
that retain the ability to edit code can solve downstream coding tasks or perform further selfmodifications. Agents that pass this initial stage are then evaluated on an expanded set of 50 tasks.
For SWE-bench, tasks are selected based on SWE-bench-verified-mini (Hobbhahn, 2025), which is
designed to be a representative sample of the full benchmark (Appendix E.2). For Polyglot, tasks
are chosen as a random subset of the full benchmark (Appendix E.3). Because the LLMs we use
are inherently stochastic, performance can be noisy. We use smaller subset evaluations to estimate
general effectiveness and identify top candidates. For SWE-bench, if a coding agent achieves over
40% success (a heuristic chosen based on the noise observed in preliminary runs) on the 60-task
subset and ranks among the top two performers in the archive, it is then evaluated on 200 tasks
(140 plus the previous 60) to more accurately assess its true coding capability (Appendix E.2). For
Polyglot, if a coding agent achieves over 40% success on the 10-task subset, it is evaluated on an
expanded 50-task subset (Appendix E.3).

4.3 BASELINES

To isolate and evaluate the contribution of the DGM’s core components, we compare the DGM against
two baselines: DGM without self-improving agents ( **DGM w/o self-improve** ) and DGM without
open-ended exploration ( **DGM w/o open-ended exploration** ). DGM w/o self-improve replicates the
approach of ADAS (Hu et al., 2025) in this setting, wherein the meta agent responsible for modifying
the coding agents remains fixed as the base agent throughout the experiment (Algorithm 2). This
baseline allows us to assess the role of iterative self-improvement in accelerating the evolution of
better coding agents. DGM w/o open-ended exploration eliminates the use of an archive and always
self-modifies the latest stored version of itself (Algorithm 3). If a coding agent self-modifies to the
point where it loses the basic functionality required to edit a codebase, it can no longer modify itself
or solve any coding task. Therefore, DGM w/o open-ended exploration retains the latest version
of itself that still maintains the basic functionality for codebase editing. This baseline allows us to
evaluate the impact of having an archive and the well-documented beneficial principles of open-ended
exploration (Clune, 2019; Stanley & Lehman, 2015; Zhang et al., 2024b; Fernando et al., 2024; Lee
et al., 2020; Samvelyan et al., 2024; Colas et al., 2022b) in guiding the agent’s evolution.

In addition to the learned baselines, we compare the DGM against handcrafted, open-source solutions.
For SWE-bench, we take the state-of-the-art (SoTA) open-source solution that has been checked
(i.e., the SWE-bench team was able to reproduce the results) (Appendix E.4). For Polyglot, we take
the representative agent (Aider) (Gauthier, 2024), which is open-sourced and designed to support
multiple programming languages and large codebase editing (Appendix E.5). For a fair comparison,
we measure the percentage of solved tasks on the same benchmark subsets used to evaluate the DGM
(Appendix E.2, Appendix E.3). These values are shown as dotted horizontal lines in Figure 2.

4.4 RESULTS

After 80 iterations of the DGM, the coding agent’s performance increases from 20.0% to 50.0% on
SWE-bench, and from 14.0% to 38.0% on Polyglot (Figure 2). Since the DGM is evaluated on only
50 tasks in the Polyglot experiment setup (Section 4.2), we additionally evaluate both the base agent
and the best DGM-discovered agent on the full Polyglot benchmark to more accurately estimate the
improvement. On the full Polyglot benchmark, the DGM improves the coding agent from 14.2% to
30.7%. This shows that the DGM can automatically self-improve to create a better coding agent.
Moreover, the performance of the best DGM-discovered agent is comparable to that of the checked,
open-source, human-designed SoTA on SWE-bench (Figure 2). On Polyglot, although the DGM

0.50

0.45

0.40

0.35

0.30

0.25

0.20

|Col1|DG DG|M w/o O M w/o S|pen-e elf-imp|Col5|nded ex rove|Col7|ploratio|Col9|n|Col11|Col12|Col13|Col14|Col15|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|| ~~DG~~|_M|||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||
||||||||||||||||

Iterations

0.35

0.30

0.25

0.20

0.15

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||

Iterations

Figure 2: **Self-improvement and open-ended exploration enable the DGM to continue making**
**progress** **and** **improve** **its** **performance.** The DGM automatically discovers increasingly better
coding agents and performs better on both (Left) SWE-bench and (Right) Polyglot. It outperforms
baselines that lack either self-improvement or open-ended exploration, showing that both components
are essential for continual self-improvement. These scores are obtained from evaluating on the
benchmark subsets detailed in Section 4.2.

**DGM Archive Tree**

**DGM Progress**

Figure 3: **The DGM automatically self-improves to become a better coding agent.** (Left) Archive
of coding agents generated during the DGM run on SWE-bench. Each node represents a coding
agent, with node 0 corresponding to the base agent. Node color indicates performance on SWE-bench
(percentage of solved tasks), while border color reflects the number of tasks for which the agent
was evaluated. Edges show which agents self-modified to produce the offsprings. Many paths to
innovation traverse lower-performing nodes, and key innovations (like node 24) lead to an explosion
of innovations built on top of them. Both properties underscore the benefits of open-ended search.
(Right) Progress plot of the DGM on SWE-bench. The light blue line shows the average score of all
agents possessing basic codebase-editing functionality. The blue line tracks the best score achieved by
any agent in the archive at each iteration. The dark line shows the lineage of the final best-discovered
agent and its precursor nodes, which includes two performance dips. This illustrates the benefits of
open-ended search, which explores a diverse set of interesting stepping stones instead of focusing
only on branching off the best solution found so far.

starts with a base agent whose performance is lower than that of Aider, it discovers an agent that far
surpasses Aider (Figure 2). The DGM-discovered agents are comparable to or outperform handcrafted
agents on both benchmarks. While the SoTA SWE-bench agent and Aider were painstakingly shaped
by human efforts, the DGM hints at a future in which such ingenuity is automated, evolving through
self-referential cycles of continuous self-improvements.

The DGM automatically improves both the tools and the workflow of how FMs are utilized (Figure 3).
For example, the DGM enhanced the edit tool to allow more granular file viewing (by lines) and more precise file editing (by string replacement), instead of always viewing or replacing the entire
file. Workflow improvements include making multiple attempts to solve a task and using another FM
to evaluate and select the best solution. Other workflow improvements include considering previous
attempts when generating subsequent ones. Appendix F.1 and Appendix F.2 show all modifications
leading up to the final best-discovered agents on SWE-bench and Polyglot respectively.

Because open-ended exploration allows branching from any agent in the archive with non-zero
probability, the DGM can get out of deceptive dips or peaks in performance. For example, at
iterations 4 and 56 of the experiment on SWE-bench, although the agent’s score temporarily fell
below that of its parent, the DGM was still able to explore innovations along that path and create a
new agent that outperformed all of its predecessors (Figure 3). Furthermore, open-ended exploration
allows different implementations of the same target functionality to be attempted. For example, while
the goal is to provide finer-grained editing tools, the specific implementation of this feature can vary
greatly and hence lead to very different performance (Appendix G). The DGM can explore multiple
implementations to find the most suitable one and avoid getting trapped in a suboptimal one.

The DGM outperforms the baselines of DGM w/o self-improve and DGM w/o open-ended exploration
on both benchmarks (Figure 2). Without updating the meta agent that modifies coding agents, DGM
w/o self-improve improves the agents in early iterations, but its gains taper off quickly (Appendix A.1).
In DGM w/o open-ended exploration, only the most recent agent is retained, so a poorly performing
self-modification makes subsequent improvements harder to achieve (Appendix A.1).

Task Transfer on Polyglot

Model Transfer on SWE-bench

Zero-shot Transfer Between Benchmarks o3-mini Claude 3.5 Sonnet Claude 3.7 Sonnet

Base agent
Best agent searched with Claude 3.5 Sonnet
Best agent transfer to other FMs

SWE-bench Polyglot

Base agent
Best agent searched on SWE-bench
Best agent searched on Polyglot python task non-python task

Base agent
Aider
DGM search on all language
DGM search only on python

Figure 4: **Transfer** **between** **Models,** **Benchmarks,** **and** **Tasks.** The superior performance of
DGM-discovered agents can be transferred across (Left) different models, (Middle) benchmarks, and
(Right) different programming language tasks in Polyglot, such as from Python tasks to C++ tasks.

To evaluate the generality of the improvements from the DGM, we tested the base agent (Section 4.1)
and the best agent discovered during the DGM run (Figure 2) with different FMs than those used
during optimization. For SWE-bench, where the DGM was run using Claude 3.5 Sonnet (New),
we replaced the FM with Claude 3.7 Sonnet (Anthropic, 2025) or o3-mini, and evaluated on 200
tasks (Figure 4, Left). With o3-mini, the base agent achieved 23.0% and the DGM-discovered agent
33.0%. With Claude 3.7 Sonnet, the base agent achieved 19.0% and the DGM-discovered agent
59.5%. These results suggest that the DGM yields improvements that generalize across FMs, rather
than being tightly coupled to the specific FM used during its run (Figure 4). Model transfer results on
Polyglot are presented in Appendix A.2.

Furthermore, we investigate the transferability of the DGM-discovered agent across different benchmarks and programming languages. First, we evaluate the best DGM-discovered agent from one
benchmark (e.g., SWE-bench) on a completely held-out benchmark (e.g., Polyglot), and vice versa
(Figure 4, Middle). The best agent evolved on SWE-bench achieves 28.9% on Polyglot, compared
to the initial agent’s baseline of 14.2%. Conversely, the best agent evolved on Polyglot achieves
24.5% on SWE-bench, outperforming the original baseline of 20.0%. Since each agent was optimized
without ever accessing the alternate benchmark, these evaluations represent truly held-out tests. The
consistent performance gains across benchmarks support our claim that DGM’s improvements reflect
general skill acquisition rather than overfitting or exploitation of benchmark-specific artifacts. Second,
we experiment with a version of the DGM trained exclusively on Python tasks from Polyglot and then
transfer the discovered agent to tasks in other languages. Focusing primarily on Python tasks slightly
improves performance on Python tasks but reduces performance on non-Python tasks compared to
the DGM trained on all languages (Figure 4, Right). However, after being transferred from Python to
other unseen languages during the search, the agent still achieves performance comparable to that of the DGM trained on all languages and substantially outperforms both the base agent and Aider.
These results demonstrate the robustness of the discovered improvements, showing that they do not
overfit to a specific programming language. We also present additional results in Appendix A.

5 SAFETY DISCUSSION

Systems capable of self-improvement, such as the DGM, represent a step toward more autonomous
AI development, aligning with long-standing goals in the field of making capable AI that can benefit
humanity (Schmidhuber, 1987; Clune, 2019; Markoff, 2016; Lehman, 2023). However, this capability
introduces unique safety considerations stemming from the system’s ability to autonomously modify
its own code. Modifications optimized solely for benchmark performance might inadvertently
introduce vulnerabilities or behaviors misaligned with human intentions, even if they improve the
target metric (Bostrom, 2020). In particular, if evaluation benchmarks do not fully capture all desired
agent properties (e.g., safety and robustness), the self-improvement loop could amplify misalignment
over successive generations. Iterative self-modification could also lead to increasingly complex and
uninterpretable internal logic, hindering human understanding, oversight, and control (Sheth et al.,
2025; Anwar et al., 2024; Greenblatt et al., 2024; Ganguli et al., 2022).

Recognizing these challenges, the current implementation and experimental setup of the DGM
incorporates several safeguards. All agent execution and self-modification processes are conducted
within isolated sandboxed environments, limiting their ability to affect the host system, and thereby
mitigating the risk of unintended actions. Each execution within the sandbox is subjected to a strict
time limit, reducing the risk of resource exhaustion or unbounded behavior. The self-improvement
process is currently confined to the well-defined domain of enhancing performance on specific coding
benchmarks by modifying the agent’s own Python codebase, thus limiting the scope of potential
modifications. Additionally, we actively monitor agent performance and code changes, with the
DGM archive providing a traceable lineage of modifications for review. At this stage, we have found
no evidence of harmful or malicious behavior in the generated agents, and the self-modifications have
been primarily focused on improving coding capabilities.

Conversely, a significant potential benefit of the self-improvement paradigm is that it could, in principle, be directed toward enhancing safety and interpretability themselves. We conduct a preliminary
investigation into how the DGM can be deployed in AI safety settings to develop countermeasures for
FM hallucination (Appendix H). Just as the DGM learns to improve its coding capabilities, it could
potentially discover and integrate better internal safeguards or modify itself for greater transparency
(e.g., incorporating principles akin to Constitutional AI (Bai et al., 2022)), if such properties were
included in its evaluation criteria (Rosser & Foerster, 2025). This suggests a promising, albeit
challenging, pathway in which self-improvement becomes a tool for building more trustworthy AI
systems. Additional research could also explore weaving Constitutional AI in from the start, though
the challenge would be incentivizing the system to retain these directives (an option worth exploring
is to create an unmodifiable part of the system to be able to evaluate at halt the rest).

The DGM demonstrates the potential of self-improving AI while still operating within safe research
boundaries due to the current limitations of frontier FMs and effective mitigations like sandboxing.
Appendix I presents additional discussion on broader safety uncertainties. We include this safety
discussion proactively to raise awareness about the emerging prospect of self-improving AI systems
and their associated safety implications, particularly as these systems inevitably become more
capable (Yudkowsky et al., 2008; Bostrom, 2002; Ecoffet et al., 2020; Bengio et al., 2024; Clune,
2019). Accordingly, we advocate for continued investigation into the safe and beneficial evolution of
AI-Generating Algorithms (Clune, 2019) and self-improving systems.

6 CONCLUSION AND LIMITATIONS

We introduce the Darwin Gödel Machine (DGM), the first self-improving system powered by FMs
with open-ended exploration, where progress on its evaluation benchmarks can directly translate
into better self-improvement capabilities. We demonstrate the automatic discovery of better tools
and FM systems, resulting in better performance on two benchmarks: SWE-bench and Polyglot.
Through self-improvement and open-ended exploration, the DGM shows a continuous increase in
performance, bringing us one step closer to self-accelerating, self-improving AI systems.

We demonstrate that the DGM can autonomously achieve performance on par with openly available
solutions. However, it still falls short of closed-source SoTA SWE-bench solutions. An open question
is whether running the DGM for longer would continue to yield performance gains and eventually
surpass closed-source solutions. These closed-source solutions often rely on elaborately handcrafted
techniques developed by teams of highly skilled experts. Since FMs have yet to match the capabilities
of such experts (e.g., in reasoning), the DGM currently requires extensive compute to discover
improvements. A single run of the DGM on SWE-bench, as presented in Section 4, takes about 2
weeks and incurs significant API costs (Appendix E.1). We hypothesize that further progress will
require more efficient use of computational resources and the development of better reasoning skills.

Since this version of the DGM is mainly powered by FMs, it is inherently limited by the capabilities
of the underlying FM. Hence, an exciting future direction is to extend self-modification beyond just
prompts or FM workflows, to include more computationally intensive methods, such as rewriting
its own training script to update the FM itself. While this version of the DGM focuses on coding,
AI systems are increasingly applied across a wide range of domains (e.g., computer vision, creative
writing). Another promising extension is to develop self-improving AI systems capable of enhancing
themselves beyond just the coding domain. A key assumption in this work is that coding benchmarks
are a good reflection of the agent’s ability to self-improve, since the self-modification task requires
the agent to modify its own codebase. However, one could envision an alternative approach that
co-evolves the target task distribution (Faldor et al., 2025; Wang et al., 2023c), thereby removing
the constraint of self-improvement being tied to a single objective, as in true open-ended processes.
Appendix J presents additional potential directions for future work. As we continue to explore this
powerful technology, we must also keep safety front and center, as discussed in Section 5.

In conclusion, the DGM represents a significant step toward the automation of AI development
through self-improving systems capable of editing their own codebase. While current limitations in
compute and reasoning constrain its full potential, continued advances in FMs and infrastructure may
unlock more powerful and general-purpose self-improvements. Provided that the safety concerns
are carefully navigated (Section 5), the future of self-improving AI systems and AI-Generating
Algorithms (Clune, 2019) holds immense promise to open-endedly evolve AI, continually rewriting
or retraining itself in pursuit of greater capabilities aligned with human values.

ETHICS STATEMENT

We affirm compliance with the ICLR Code of Ethics. This work studies self-improving AI systems
in the limited context of code-editing agents evaluated on standard programming benchmarks. No
human subjects were involved and no personally identifiable information (PII) was collected or
processed; IRB approval was therefore not required.

**Safety and misuse.** Self-modifying systems can pose safety risks if allowed to act without constraints
or if optimizations inadvertently introduce unsafe behaviors. To mitigate this, all agents in our
experiments ran inside isolated sandboxes with strict resource and time limits; agents had limited
network access and no ability to modify the host environment. The self-improvement scope was
restricted to the agent’s own Python codebase and evaluation harnesses. We maintained a complete,
auditable lineage (archive) of code changes and evaluations, enabling rollback and post-hoc analysis.
We did not deploy discovered agents in real development environments. Our release plan (code,
prompts, and evaluation artifacts) will exclude any components that grant elevated system access and
will include default sandboxing, guardrails, and clear documentation of intended use.

**Dual-use,** **downstream** **impact,** **and** **limitations.** Stronger autonomous coding agents could be
dual-use (e.g., aiding software maintenance, but also potentially facilitating creation of harmful code
if misapplied). We believe the research benefits (e.g., advancing methods for controlled, auditable
self-improvement and demonstrating practical safeguards) outweigh the risks. Nevertheless, we
explicitly discourage security-sensitive or unsandboxed deployment and provide concrete safety
recommendations (Section 5). Our empirical focus on benchmark optimization may not capture
all desirable properties (robustness, interpretability, or broader social values). We therefore treat
benchmark gains as necessary but insufficient indicators of general AI development, and discuss
avenues to integrate other objectives (e.g., safety, reasoning) into the optimization loop.

**Data governance, IP, and licensing.** We evaluate on SWE-bench Verified and Polyglot, which are
composed of open-source repositories and tasks. We complied with dataset licenses and usage terms
to the best of our knowledge. We did not introduce or distribute proprietary code. Foundation models
(FMs) were accessed via provider APIs under their terms of service; we did not submit sensitive data,
nor attempt to circumvent usage policies. Logs released with this work will be scrubbed of API keys
and any incidental sensitive strings.

**Bias, fairness, and equity.** Although our domain is software code rather than human-centered text,
FM behavior can still reflect biases (e.g., language or ecosystem preferences) and may unevenly
benefit communities whose tooling is better represented in training data. We partially address this by
evaluating across multiple languages (Polyglot) and reporting cross-benchmark transfer. Future work
should add diagnostics for biased failure modes and include broader, community-driven task sets.

**Conflicts of interest and funding.** No author has a financial interest in products whose performance
is evaluated here. Sponsors and employers did not influence experimental design, analysis, or the
decision to publish, beyond providing salary or standard research support. Any external compute or
API credits are acknowledged in the appendix.

REPRODUCIBILITY STATEMENT

We will open-source all code and full agent logs, including the complete archive lineage of selfmodifications (diffs, prompts, and configs) as well as the evaluation harness. To support exact replication, we reference the following: algorithmic details and pseudocode (Section 3, Appendix C.4);
parent selection and open-ended exploration settings (Appendix C.2); foundation model choices
and hyperparameters (Appendix D.1); benchmark task subsets for SWE-bench and Polyglot (Appendix E.2, Appendix E.3); staged evaluation protocols and scripts (Section 4.2); implementations
and diffs for the best discovered agents (Appendix F.1, Appendix F.2); and compute and cost estimates
(Appendix E.1). The released code repository will include environment specifications and scripts to
reproduce all results, figures, and tables.

ACKNOWLEDGMENTS

This research was supported by the Vector Institute, the Canada CIFAR AI Chairs program, a
grant from Schmidt Futures, an NSERC Discovery Grant, and a generous donation from Rafael
Cosman. Resources used in preparing this research were provided, in part, by the Province of
Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute
(https://vectorinstitute.ai/partnerships/current-partners/). Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the authors and do not necessarily reflect the
views of the sponsors. We also thank Aaron Dharna, Ben Norman, Cédric Colas, Sam Devlin, and
Shyam Sudhakaran for insightful discussions and feedback.

REFERENCES

Fuma Aki, Riku Ikeda, Takumi Saito, Ciaran Regan, and Mizuki Oka. Llm-poet: Evolving complex environments using large language models. In _Proceedings_ _of_ _the_ _Genetic_ _and_ _Evolutionary_
_Computation Conference Companion_, pp. 243–246, 2024.

Rajeev Alur, Rishabh Singh, Dana Fisman, and Armando Solar-Lezama. Search-based program synthesis. _Communications of the ACM_, 61(12):84–93, 2018.

S-I Amari. Learning patterns and pattern sequences by self-organizing nets of threshold elements.

_IEEE Transactions on computers_, 100(11):1197–1206, 1972.

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob

McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay.
_Advances in neural information processing systems_, 30, 2017.

Anthropic. Claude 3.5 Sonnet. [https://www.anthropic.com/news/](https://www.anthropic.com/news/claude-3-5-sonnet)
[claude-3-5-sonnet, June 2024a.](https://www.anthropic.com/news/claude-3-5-sonnet) [Accessed 17 April 2025].

Anthropic. Claude can now use tools, May 2024b. URL [https://www.anthropic.com/](https://www.anthropic.com/news/tool-use-ga)

[news/tool-use-ga.](https://www.anthropic.com/news/tool-use-ga) Accessed: 2025-05-03.

Anthropic. Claude 3.7 sonnet and claude code, February 2025. [URL https://www.anthropic.](https://www.anthropic.com/news/claude-3-7-sonnet)

[com/news/claude-3-7-sonnet.](https://www.anthropic.com/news/claude-3-7-sonnet) Accessed: 2025-05-06.

Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase,

Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, et al. Foundational challenges
in assuring alignment and safety of large language models. _arXiv_ _preprint_ _arXiv:2404.09932_,
2024.

Dzmitry Bahdanau, Kyung Hyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In _International Conference on Learning Representations_, 2015.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones,

Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional AI:
Harmlessness from AI feedback. _arXiv preprint arXiv:2212.08073_, 2022.

Adrien Baranes and Pierre-Yves Oudeyer. Active learning of inverse models with intrinsically motivated goal exploration in robots. _Robotics and Autonomous Systems_, 61(1):49–73, 2013.

Shraddha Barke, Emmanuel Anaya Gonzalez, Saketh Ram Kasibatla, Taylor Berg-Kirkpatrick, and

Nadia Polikarpova. Hysynth: Context-free llm approximation for guiding program synthesis.
_Advances in Neural Information Processing Systems_, 37:15612–15645, 2024.

Yoshua Bengio, Geoffrey Hinton, Andrew Yao, Dawn Song, Pieter Abbeel, Trevor Darrell, Yu
val Noah Harari, Ya-Qin Zhang, Lan Xue, Shai Shalev-Shwartz, et al. Managing extreme AI risks
amid rapid progress. _Science_, 384(6698):842–845, 2024.

N Bostrom. Existential Risks: analyzing human extinction scenarios and related hazards. _Journal of_

_Evolution and Technology_, 9, 2002.

Nick Bostrom. Ethical issues in advanced artificial intelligence. _Machine Ethics and Robot Ethics_, pp. 69–75, 2020.

Herbie Bradley, Andrew Dai, Hannah Benita Teufel, Jenny Zhang, Koen Oostermeijer, Marco

Bellagente, Jeff Clune, Kenneth Stanley, Gregory Schott, and Joel Lehman. Quality-diversity
through ai feedback. In _The Twelfth International Conference on Learning Representations_, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,

Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. _Advances in neural information processing systems_, 33:1877–1901, 2020.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes,

Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative
interactive environments. In _Forty-first International Conference on Machine Learning_, 2024.

J Richard Buchi and Lawrence H Landweber. Solving sequential conditions by finite-state strategies.

In _The collected works of J. Richard Büchi_, pp. 525–541. Springer, 1990.

Ruisheng Cao, Fangyu Lei, Haoyuan Wu, Jixuan Chen, Yeqiao Fu, Hongcheng Gao, Xinzhuang

Xiong, Hanchong Zhang, Wenjing Hu, Yuchen Mao, et al. Spider2-v: How far are multimodal
agents from automating data science and engineering workflows? _Advances in Neural Information_
_Processing Systems_, 37:107703–107744, 2024.

Konstantinos Chatzilygeroudis, Antoine Cully, Vassilis Vassiliades, and Jean-Baptiste Mouret.

Quality-diversity optimization: a novel branch of stochastic optimization. In _Black_ _Box_ _Op-_
_timization, Machine Learning, and No-Free Lunch Theorems_, pp. 109–135. Springer, 2021.

Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, and Shengxin Zhu. Unleashing the poten
tial of prompt engineering in large language models: a comprehensive review. _arXiv_ _preprint_
_arXiv:2310.14735_, 2023.

Ching-An Cheng, Allen Nie, and Adith Swaminathan. Trace is the next autodiff: Generative optimization with rich feedback, execution traces, and llms. _Advances_ _in_ _Neural_ _Information_
_Processing Systems_, 37:71596–71642, 2024.

Jeff Clune. AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence. _arXiv preprint arXiv:1905.10985_, 2019.

Cédric Colas, Pierre Fournier, Mohamed Chetouani, Olivier Sigaud, and Pierre-Yves Oudeyer.

Curious: intrinsically motivated modular multi-goal reinforcement learning. In _International_
_conference on machine learning_, pp. 1331–1340. PMLR, 2019.

Cédric Colas, Tristan Karch, Clément Moulin-Frier, and Pierre-Yves Oudeyer. Language and culture internalization for human-like autotelic AI. _Nature Machine Intelligence_, 4(12):1068–1076, 2022a.

Cédric Colas, Tristan Karch, Olivier Sigaud, and Pierre-Yves Oudeyer. Autotelic agents with intrinsically motivated goal-conditioned reinforcement learning: a short survey. _Journal_ _of_
_Artificial Intelligence Research_, 74:1159–1199, 2022b.

Cédric Colas, Laetitia Teodorescu, Pierre-Yves Oudeyer, Xingdi Yuan, and Marc-Alexandre Côté.

Augmenting autotelic agents with large language models. In _Conference on Lifelong Learning_
_Agents_, pp. 205–226. PMLR, 2023.

Rémi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In _International_

_conference on computers and games_, pp. 72–83. Springer, 2006.

Charles Darwin. Origin of the species. In _British Politics and the environment in the long nineteenth_

_century_, pp. 47–55. Routledge, 2023.

Richard Dawkins. The evolution of evolvability. In _Artificial life_, pp. 201–220. Routledge, 2019.

Michael Dennis, Natasha Jaques, Eugene Vinitsky, Alexandre Bayen, Stuart Russell, Andrew Critch, and Sergey Levine. Emergent complexity and zero-shot transfer via unsupervised environment
design. _Advances in neural information processing systems_, 33:13049–13061, 2020.

Aaron Dharna, Cong Lu, and Jeff Clune. Quality-Diversity Self-Play: Open-Ended Strategy Innova
tion via Foundation Models. In _NeurIPS 2024 Workshop on Open-World Agents_, 2024.

Li Ding, Jenny Zhang, Jeff Clune, Lee Spector, and Joel Lehman. Quality diversity through
human feedback: towards open-ended diversity-driven optimization. In _Proceedings of the 41st_
_International Conference on Machine Learning_, pp. 11072–11090, 2024.

Theodosius Dobzhansky. _Genetics of the evolutionary process_, volume 139. Columbia University

Press, 1970.

Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Go-explore: a new approach for hard-exploration problems. _arXiv preprint arXiv:1901.10995_, 2019.

Adrien Ecoffet, Jeff Clune, and Joel Lehman. Open questions in creating safe open-ended AI: tensions between control and creativity. In _Artificial Life Conference Proceedings 32_, pp. 27–35. MIT Press
One Rogers Street, Cambridge, MA 02142-1209, USA journals-info ..., 2020.

Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O Stanley, and Jeff Clune. First return, then explore. _Nature_, 590(7847):580–586, 2021.

Anthony W Fisher Edwards. The genetical theory of natural selection. _Genetics_, 154(4):1419–1426,

2000.

Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Lucas Morales, Luke Hewitt, Luc

Cary, Armando Solar-Lezama, and Joshua B Tenenbaum. Dreamcoder: Bootstrapping inductive
program synthesis with wake-sleep library learning. In _Proceedings_ _of_ _the_ _42nd_ _acm_ _sigplan_
_international_ _conference_ _on_ _programming_ _language_ _design_ _and_ _implementation_, pp. 835–850,
2021.

Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need:

Learning skills without a reward function. _arXiv preprint arXiv:1802.06070_, 2018.

Meta Fundamental AI Research Diplomacy Team (FAIR)†, Anton Bakhtin, Noam Brown, Emily

Dinan, Gabriele Farina, Colin Flaherty, Daniel Fried, Andrew Goff, Jonathan Gray, Hengyuan Hu,
et al. Human-level play in the game of Diplomacy by combining language models with strategic
reasoning. _Science_, 378(6624):1067–1074, 2022.

Maxence Faldor, Jenny Zhang, Antoine Cully, and Jeff Clune. OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code. In
_The_ _Thirteenth_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2025. URL [https:](https://openreview.net/forum?id=Y1XkzMJpPd)
[//openreview.net/forum?id=Y1XkzMJpPd.](https://openreview.net/forum?id=Y1XkzMJpPd)

Chrisantha Fernando, Dylan Sunil Banarse, Henryk Michalewski, Simon Osindero, and Tim Rock
täschel. Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution. In _Forty-first_
_International Conference on Machine Learning_, 2024.

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben

Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language models to
reduce harms: Methods, scaling behaviors, and lessons learned. _arXiv preprint arXiv:2209.07858_,
2022.

Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min

Lin, and Tianyu Pang. Flowreasoner: Reinforcing query-level meta-agents. _arXiv_ _preprint_
_arXiv:2504.15257_, 2025.

Paul Gauthier. Aider: Ai pair programming in your terminal. [https://github.com/](https://github.com/Aider-AI/aider)
[Aider-AI/aider, 2024.](https://github.com/Aider-AI/aider) Accessed: 2025-05-14.

Loris Gaven, Thomas Carta, Clément Romac, Cédric Colas, Sylvain Lamprier, Olivier Sigaud, and

Pierre-Yves Oudeyer. MAGELLAN: Metacognitive predictions of learning progress guide autotelic
LLM agents in large goal spaces. _arXiv preprint arXiv:2502.07709_, 2025.

John Gerhart and Marc Kirschner. The theory of facilitated variation. _Proceedings of the National_

_Academy of Sciences_, 104(suppl_1):8582–8589, 2007.

Irving John Good. Speculations concerning the first ultraintelligent machine. In _Advances_ _in_

_computers_, volume 6, pp. 31–88. Elsevier, 1966.

Google DeepMind. Gemini model “thinking” updates - march
2025. [https://blog.google/technology/google-deepmind/](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking)
[gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking,](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking)
March 2025. Accessed: 2025-05-11.

Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks,

Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, et al. Alignment faking in large
language models. _arXiv preprint arXiv:2412.14093_, 2024.

Sumit Gulwani. Automating string processing in spreadsheets using input-output examples. _ACM_

_Sigplan Notices_, 46(1):317–330, 2011.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,

Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025.

John Storrs Hall. Self-improving AI: An analysis. _Minds and Machines_, 17(3):249–259, 2007.

Alex Havrilla, Andrew Dai, Laura O’Mahony, Koen Oostermeijer, Vera Zisler, Alon Albalak, Fabrizio

Milo, Sharath Chandra Raparthy, Kanishk Gandhi, Baber Abbasi, et al. Surveying the effects of
quality, diversity, and complexity in synthetic data from large language models. _arXiv preprint_
_arXiv:2412.02980_, 2024a.

Alex Havrilla, Sharath Raparthy, Chris to for us Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi,

Eric Hambro, and Roberta Raileanu. Glore: When, where, and how to improve llm reasoning via
global and local refinements. _arXiv preprint arXiv:2402.10963_, 2024b.

Jesse Love Hendrikse, Trish Elizabeth Parsons, and Benedikt Hallgrímsson. Evolvability as the proper focus of evolutionary developmental biology. _Evolution & development_, 9(4):393–401,
2007.

Nathan Herr, Tim Rocktäschel, and Roberta Raileanu. Llm-first search: Self-guided exploration of the solution space. _arXiv preprint arXiv:2506.05213_, 2025.

Marius Hobbhahn. Swe-bench verified mini. [https://github.com/mariushobbhahn/](https://github.com/mariushobbhahn/SWEBench-verified-mini)

[SWE-Bench-verified-mini, April 2025.](https://github.com/mariushobbhahn/SWEBench-verified-mini) Accessed: 2025-04-16.

John J Hopfield. Neural networks and physical systems with emergent collective computational abilities. _Proceedings of the national academy of sciences_, 79(8):2554–2558, 1982.

Shengran Hu and Jeff Clune. Thought Cloning: Learning to think while acting by imitating human thinking. _Advances in Neural Information Processing Systems_, 36, 2024.

Shengran Hu, Cong Lu, and Jeff Clune. Automated Design of Agentic Systems. In _The Thirteenth_

_International Conference on Learning Representations_, 2025. [URL https://openreview.](https://openreview.net/forum?id=t9U3LW7JVX)
[net/forum?id=t9U3LW7JVX.](https://openreview.net/forum?id=t9U3LW7JVX)

Yue Hu, Yuzhu Cai, Yaxin Du, Xinyu Zhu, Xiangrui Liu, Zijie Yu, Yuchen Hou, Shuo Tang, and

Siheng Chen. Self-evolving multi-agent collaboration networks for software development. _arXiv_
_preprint arXiv:2410.16946_, 2024.

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han.

Large language models can self-improve. _arXiv preprint arXiv:2210.11610_, 2022.

Edward Hughes, Michael Dennis, Jack Parker-Holder, Feryal Behbahani, Aditi Mavalankar, Yuge

Shi, Tom Schaul, and Tim Rocktaschel. Open-endedness is essential for artificial superhuman
intelligence. _arXiv preprint arXiv:2406.04268_, 2024.

Kazuki Irie, Imanol Schlag, Róbert Csordás, and Jürgen Schmidhuber. A modern self-referential weight matrix that learns to modify itself. In _International Conference on Machine Learning_, pp.
9660–9677. PMLR, 2022.

Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. Meta-learning continual learning algorithms.

_Transactions on Machine Learning Research_, 2025.

Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M Czarnecki, Jeff Donahue, Ali

Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, et al. Population based training
of neural networks. _arXiv preprint arXiv:1711.09846_, 2017.

Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel. Prioritized level replay. In _International_

_Conference on Machine Learning_, pp. 4940–4950. PMLR, 2021.

Minqi Jiang, Tim Rocktäschel, and Edward Grefenstette. General intelligence requires rethinking exploration. _Royal Society Open Science_, 10(6):230539, 2023.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and

Karthik R Narasimhan. SWE-bench: Can Language Models Resolve Real-world Github Issues? In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2024. URL
[https://openreview.net/forum?id=VTF8yNQM66.](https://openreview.net/forum?id=VTF8yNQM66)

Ingmar Kanitscheider, Joost Huizinga, David Farhi, William Hebgen Guss, Brandon Houghton, Raul

Sampedro, Peter Zhokhov, Bowen Baker, Adrien Ecoffet, Jie Tang, Oleg Klimov, and Jeff Clune.
Multi-task curriculum learning in a complex, visual, hard-exploration domain: Minecraft. _arXiv_
_preprint arXiv:2106.14876_, 2021.

Akbir Khan, John Hughes, Dan Valentine, Laura Ruis, Kshitij Sachan, Ansh Radhakrishnan, Edward

Grefenstette, Samuel R Bowman, Tim Rocktäschel, and Ethan Perez. Debating with more
persuasive llms leads to more truthful answers. _arXiv preprint arXiv:2402.06782_, 2024.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vard
hamanan, Saiful Haq, Ashutosh Sharma, Thomas T Joshi, Hanna Moazam, et al. Dspy: Compiling
declarative language model calls into self-improving pipelines. _arXiv preprint arXiv:2310.03714_,
2023.

Yoon Kim, Carl Denton, Luong Hoang, and Alexander M Rush. Structured Attention Networks. In

_International Conference on Learning Representations_, 2017.

Motoo Kimura. The neutral theory of molecular evolution. _Scientific American_, 241(5):98–129,

1979.

Louis Kirsch and Jürgen Schmidhuber. Self-referential meta learning. In _First_ _Conference_ _on_

_Automated Machine Learning (Late-Breaking Workshop)_, 2022.

Martin Klissarov, Pierluca D’Oro, Shagun Sodhani, Roberta Raileanu, Pierre-Luc Bacon, Pascal

Vincent, Amy Zhang, and Mikael Henaff. Motif: Intrinsic motivation from artificial intelligence
feedback. _arXiv preprint arXiv:2310.00166_, 2023.

Martin Klissarov, Mikael Henaff, Roberta Raileanu, Shagun Sodhani, Pascal Vincent, Amy Zhang,

Pierre-Luc Bacon, Doina Precup, Marlos C Machado, and Pierluca D’Oro. MaestroMotif: Skill
Design from Artificial Intelligence Feedback. _arXiv preprint arXiv:2412.08542_, 2024.

Varun Raj Kompella, Marijn Stollenga, Matthew Luciw, and Juergen Schmidhuber. Continual curiosity-driven skill acquisition from high-dimensional video inputs for humanoid robots. _Artificial_
_Intelligence_, 247:313–335, 2017.

Robert Lange, Tom Schaul, Yutian Chen, Tom Zahavy, Valentin Dalibard, Chris Lu, Satinder Singh, and Sebastian Flennerhag. Discovering evolution strategies via meta-black-box optimization. In
_Proceedings of the Companion Conference on Genetic and Evolutionary Computation_, pp. 29–30,
2023.

Robert Lange, Yingtao Tian, and Yujin Tang. Large language models as evolution strategies. In

_Proceedings of the Genetic and Evolutionary Computation Conference Companion_, pp. 579–582,
2024.

Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning quadrupedal locomotion over challenging terrain. _Science robotics_, 5(47):eabc5986, 2020.

Joel Lehman. Machine love. _arXiv preprint arXiv:2302.09248_, 2023.

Joel Lehman and Kenneth O Stanley. Novelty search and the problem with objectives. _Genetic_

_programming theory and practice IX_, pp. 37–56, 2011.

Joel Lehman, Jonathan Gordon, Shawn Jain, Kamal Ndousse, Cathy Yeh, and Kenneth O Stanley.

Evolution through large models. In _Handbook of Evolutionary Machine Learning_, pp. 331–366.
Springer, 2023.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,

Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in neural information processing systems_, 33:
9459–9474, 2020.

J. Li, Storie J., and J. Clune. Encouraging creative thinking in robots improves their ability to solve challenging problems. In _Proceedings of the Genetic and Evolutionary Computation Conference_,
pp. 193–200, 2014.

Yixuan Li, Julian Parsert, and Elizabeth Polgreen. Guiding enumerative program synthesis with large language models. In _International Conference on Computer Aided Verification_, pp. 280–301.
Springer, 2024.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-agent
debate. _arXiv preprint arXiv:2305.19118_, 2023.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan

Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In _The Twelfth_
_International Conference on Learning Representations_, 2023.

Bryan Lim, Manon Flageat, and Antoine Cully. Large language models as in-context ai generators for quality-diversity. In _ALIFE 2024:_ _Proceedings of the 2024 Artificial Life Conference_ . MIT
Press, 2024.

Seppo Linnainmaa. _The representation of the cumulative rounding error of an algorithm as a Taylor_

_expansion of the local rounding errors_ . PhD thesis, Master’s Thesis (in Finnish), Univ. Helsinki,
1970.

Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu

Zhang. Evolution of heuristics: Towards efficient automatic algorithm design using large language
model. _arXiv preprint arXiv:2401.02051_, 2024.

Lei Liu, Xiaoyan Yang, Yue Shen, Binbin Hu, Zhiqiang Zhang, Jinjie Gu, and Guannan Zhang.

Think-in-memory: Recalling and post-thinking enable llms with long-term memory. _arXiv preprint_
_arXiv:2311.08719_, 2023.

Chris Lu, Sebastian Towers, and Jakob Foerster. Arbitrary order meta-learning with simple population
based evolution. In _Artificial Life Conference Proceedings 35_, volume 2023, pp. 67. MIT Press
One Rogers Street, Cambridge, MA 02142-1209, USA journals-info ..., 2023.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist:

Towards fully automated open-ended scientific discovery. _arXiv preprint arXiv:2408.06292_, 2024a.

Cong Lu, Shengran Hu, and Jeff Clune. Intelligent go-explore: Standing on the shoulders of giant foundation models. _arXiv preprint arXiv:2405.15143_, 2024b.

Cong Lu, Shengran Hu, and Jeff Clune. Automated capability discovery via model self-exploration.

_arXiv preprint arXiv:2502.0757_, 2025.

Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman,

Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models. _arXiv preprint arXiv:2310.12931_, 2023.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri

Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement
with self-feedback, 2023. _URL https://arxiv. org/abs/2303.17651_, 2023.

John Markoff. _Machines of loving grace:_ _The quest for common ground between humans and robots_ .

HarperCollins Publishers, 2016.

Ernst Mayr. _The_ _growth_ _of_ _biological_ _thought:_ _Diversity,_ _evolution,_ _and_ _inheritance_ . Harvard

University Press, 1982.

Luke Metz, C Daniel Freeman, Niru Maheswaranathan, and Jascha Sohl-Dickstein. Training learned optimizers with randomly initialized learned optimizers. _arXiv preprint arXiv:2101.07367_, 2021.

Ali Modarressi, Ayyoob Imani, Mohsen Fayyaz, and Hinrich Schütze. Ret-llm: Towards a general read-write memory for large language models. _arXiv preprint arXiv:2305.14322_, 2023.

Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. _arXiv preprint_

_arXiv:1504.04909_, 2015.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke

Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time
scaling. _arXiv preprint arXiv:2501.19393_, 2025.

Muhammad U Nasir and Julian Togelius. Practical PCG through large language models. In _2023_

_IEEE Conference on Games (CoG)_, pp. 1–4. IEEE, 2023.

Muhammad U Nasir, Steven James, and Julian Togelius. Word2world: Generating stories and worlds through large language models. _arXiv preprint arXiv:2405.06686_, 2024.

Anh Mai Nguyen, Jason Yosinski, and Jeff Clune. Innovation engines: Automated creativity and improved stochastic optimization via deep learning. In _Proceedings of the 2015 annual conference_
_on genetic and evolutionary computation_, pp. 959–966, 2015.

Fan Nie, Lan Feng, Haotian Ye, Weixin Liang, Pan Lu, Huaxiu Yao, Alexandre Alahi, and James

Zou. Weak-for-strong: Training weak meta-agent to harness strong executors. _arXiv preprint_
_arXiv:2504.04785_, 2025.

Boye Niu, Yiliao Song, Kai Lian, Yifan Shen, Yu Yao, Kun Zhang, and Tongliang Liu. Flow: Modu
larized agentic workflow automation. In _The Thirteenth International Conference on Learning_
_Representations_, 2025.

Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wag-˜ ner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan
Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. Alphaevolve: A coding agent for scientific and algorithmic discovery.
Technical report, Google DeepMind, 2025.

OpenAI. Introducing swe-bench verified. [https://openai.com/index/](https://openai.com/index/introducing-swe-bench-verified/)
[introducing-swe-bench-verified/, August 2024.](https://openai.com/index/introducing-swe-bench-verified/) Accessed: 2025-04-16.

OpenAI. OpenAI o3-mini. [https://openai.com/index/openai-o3-mini/,](https://openai.com/index/openai-o3-mini/) January

2025. Accessed: 2025-05-01.

Pierre-Yves Oudeyer, Frdric Kaplan, and Verena V Hafner. Intrinsic motivation systems for au
tonomous mental development. _IEEE transactions on evolutionary computation_, 11(2):265–286,
2007.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong

Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow
instructions with human feedback. _Advances in neural information processing systems_, 35:27730–
27744, 2022.

Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A Decomposable Attention

Model for Natural Language Inference. In _Proceedings_ _of_ _the_ _2016_ _Conference_ _on_ _Empirical_
_Methods in Natural Language Processing_, pp. 2249–2255, 2016.

Jack Parker-Holder, Philip Ball, Jake Bruce, Vibhavari Dasagi, Kristian Holsheimer, Chris
tos Kaplanis, Alexandre Moufarek, Guy Scully, Jeremy Shar, Jimmy Shi, Stephen Spencer,
Jessica Yung, Michael Dennis, Sultan Kenjeyev, Shangbang Long, Vlad Mnih, Harris
Chan, Maxime Gazeau, Bonnie Li, Fabio Pardo, Luyu Wang, Lei Zhang, Frederic Besse,
Tim Harley, Anna Mitenkova, Jane Wang, Jeff Clune, Demis Hassabis, Raia Hadsell,
Adrian Bolton, Satinder Singh, and Tim Rocktäschel. Genie 2: A large-scale foundation world model, 2024. URL [https://deepmind.google/discover/blog/](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
[genie-2-a-large-scale-foundation-world-model/.](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)

Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In _International conference on machine learning_, pp. 2778–2787.
PMLR, 2017.

Paul Gauthier. o1 tops aider’s new polyglot leaderboard. [https://aider.chat/2024/12/](https://aider.chat/2024/12/21/polyglot.html)

[21/polyglot.html, December 2024.](https://aider.chat/2024/12/21/polyglot.html) Accessed: 2025-04-16.

Oleksandr Polozov and Sumit Gulwani. Flashmeta: A framework for inductive program synthe
sis. In _Proceedings of the 2015 ACM SIGPLAN International Conference on Object-Oriented_
_Programming, Systems, Languages, and Applications_, pp. 107–126, 2015.

Justin K Pugh, Lisa B Soros, and Kenneth O Stanley. Quality diversity: A new frontier for evolutionary computation. _Frontiers in Robotics and AI_, 3:40, 2016.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_, 1(8):9, 2019.

Maxime Robeyns, Martin Szummer, and Laurence Aitchison. A Self-Improving Coding Agent. _arXiv_

_preprint arXiv:2504.15228_, 2025.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog,

M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang,
Omar Fawzi, et al. Mathematical discoveries from program search with large language models.
_Nature_, 625(7995):468–475, 2024.

J Rosser and Jakob Nicolaus Foerster. Agentbreeder: Mitigating the AI safety impact of multi-agent scaffolds via self-improvement. In _Scaling Self-Improving Foundation Models without Human_
_Supervision_, 2025. [URL https://openreview.net/forum?id=j0n3BJJTcT.](https://openreview.net/forum?id=j0n3BJJTcT)

David E Rumelhart, Geoffrey E Hinton, Ronald J Williams, et al. Learning internal representations by error propagation, 1985.

Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram Markosyan,

Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, et al. Rainbow
teaming: Open-ended generation of diverse adversarial prompts. _Advances in Neural Information_
_Processing Systems_, 37:69747–69786, 2024.

Cansu Sancaktar, Christian Gumbsch, Andrii Zadaianchuk, Pavel Kolev, and Georg Martius. SENSEI:

Semantic Exploration Guided by Foundation Models to Learn Versatile World Models. _arXiv_
_preprint arXiv:2503.01584_, 2025.

Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal value function approximators.

In _International conference on machine learning_, pp. 1312–1320. PMLR, 2015.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke

Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach
themselves to use tools. _Advances in Neural Information Processing Systems_, 36:68539–68551,
2023.

Jürgen Schmidhuber. _Evolutionary principles in self-referential learning, or on learning how to learn:_

_the meta-meta-... hook_ . PhD thesis, Technische Universität München, 1987.

Jürgen Schmidhuber. A ‘self-referential’weight matrix. In _International conference on artificial_

_neural networks_, pp. 446–450. Springer, 1993.

Jürgen Schmidhuber. Gödel machines: Fully self-referential optimal universal self-improvers. In

_Artificial general intelligence_, pp. 199–226. Springer, 2007.

Jürgen Schmidhuber. Driven by compression progress: A simple principle explains essential aspects of subjective beauty, novelty, surprise, interestingness, attention, curiosity, creativity, art, science,
music, jokes. In _Workshop_ _on_ _anticipatory_ _behavior_ _in_ _adaptive_ _learning_ _systems_, pp. 48–76.
Springer, 2008.

Jürgen Schmidhuber. Powerplay: Training an increasingly general problem solver by continually searching for the simplest still unsolvable problem. _Frontiers in psychology_, 4:313, 2013.

Jürgen Schmidhuber and Rudolf Huber. _Learning to generate focus trajectories for attentive vision_ .

Institut für Informatik, 1990.

Sander Schulhoff, Michael Ilie, Nishant Balepur, Konstantine Kahadze, Amanda Liu, Chenglei

Si, Yinheng Li, Aayush Gupta, HyoJung Han, Sevien Schulhoff, et al. The prompt report: A
systematic survey of prompting techniques. _arXiv preprint arXiv:2406.06608_, 2024.

Ivaxi Sheth, Jan Wehner, Sahar Abdelnabi, Ruta Binkyte, and Mario Fritz. Safety is Essential for

Responsible Open-Ended Systems. _arXiv preprint arXiv:2502.04512_, 2025.

Kensen Shi, Hanjun Dai, Wen-Ding Li, Kevin Ellis, and Charles Sutton. Lambdabeam: Neural pro
gram search with higher-order functions and lambdas. _Advances in Neural Information Processing_
_Systems_, 36:51327–51346, 2023.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion:

Language agents with verbal reinforcement learning. _Advances in Neural Information Processing_
_Systems_, 36:8634–8652, 2023.

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche,

Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering
the game of go with deep neural networks and tree search. _nature_, 529(7587):484–489, 2016.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez,

Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. Mastering chess and shogi
by self-play with a general reinforcement learning algorithm. _arXiv preprint arXiv:1712.01815_,
2017.

Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J

Liu, James Harrison, Jaehoon Lee, Kelvin Xu, et al. Beyond human data: Scaling self-training for
problem-solving with language models. _arXiv preprint arXiv:2312.06585_, 2023.

Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. _Advances in Neural Information Processing Systems_, 35:9460–9471, 2022.

Kenneth O Stanley and Joel Lehman. _Why greatness cannot be planned:_ _The myth of the objective_ .

Springer, 2015.

Kenneth O Stanley, Joel Lehman, and Lisa Soros. Open-endedness: The last grand challenge you’ve never heard of. _While open-endedness could be a force for discovering intelligence, it could also_
_be a component of AI itself_, 2017.

Marilyn Strathern. ‘Improving ratings’: audit in the British University system. _European review_, 5

(3):305–321, 1997.

Jinwei Su, Yinghui Xia, Ronghua Shi, Jianhui Wang, Jianuo Huang, Yijin Wang, Tianyu Shi, Yang

Jingsong, and Lewei He. Debflow: Automating agent creation via agent debate. _arXiv preprint_
_arXiv:2503.23781_, 2025.

Shyam Sudhakaran, Miguel González-Duque, Matthias Freiberger, Claire Glanois, Elias Najarro, and Sebastian Risi. Mariogpt: Open-ended text2level generation through large language models.
_Advances in Neural Information Processing Systems_, 36:54213–54227, 2023.

OpenAI Team, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden

Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card.
_arXiv preprint arXiv:2412.16720_, 2024.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz

Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing_
_systems_, 30, 2017.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and

Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. _arXiv_
_preprint arXiv:2305.16291_, 2023a.

Ren-Jian Wang, Ke Xue, Yutong Wang, Peng Yang, Haobo Fu, Qiang Fu, and Chao Qian. Diversity from human feedback. _arXiv preprint arXiv:2310.06648_, 2023b.

Rui Wang, Joel Lehman, Jeff Clune, and Kenneth O Stanley. Paired open-ended trailblazer (poet):

Endlessly generating increasingly complex and diverse learning environments and their solutions.
_arXiv preprint arXiv:1901.01753_, 2019.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan,

Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai software develop
ers as generalist agents. In _The Thirteenth International Conference on Learning Representations_,
2024.

Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory

Erickson, David Held, and Chuang Gan. Robogen: Towards unleashing infinite data for automated
robot learning via generative simulation. _arXiv preprint arXiv:2311.01455_, 2023c.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny

Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in_
_neural information processing systems_, 35:24824–24837, 2022.

Marco Wiering and Jürgen Schmidhuber. Hq-learning. _Adaptive behavior_, 6(2):219–246, 1997.

S Wright. The roles of mutation, inbreeding, crossbreeding and selection in evolution, proceedings of the sixth international congress of genetics. proc sixth int congr genet [internet]. _New York356366_,
1932.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Agentless: Demystifying llm-based software engineering agents. _arXiv preprint arXiv:2407.01489_, 2024.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering.
In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_, 2024. URL
[https://arxiv.org/abs/2405.15793.](https://arxiv.org/abs/2405.15793)

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.

React: Synergizing reasoning and acting in language models. In _International_ _Conference_ _on_
_Learning Representations (ICLR)_, 2023.

Rui Ye, Shuo Tang, Rui Ge, Yaxin Du, Zhenfei Yin, Siheng Chen, and Jing Shao. Mas-gpt: Training llms to build llm-based multi-agent systems. _arXiv preprint arXiv:2503.03686_, 2025.

Xunjian Yin, Xinyi Wang, Liangming Pan, Xiaojun Wan, and William Yang Wang. G _\_ " odel

Agent: A Self-Referential Agent Framework for Recursive Self-Improvement. _arXiv_ _preprint_
_arXiv:2410.04444_, 2024.

Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Dongsheng Li, and Deqing Yang. Evoagent: Towards automatic multi-agent generation via evolutionary algorithms. _arXiv preprint arXiv:2406.14228_,
2024.

Eliezer Yudkowsky et al. Artificial Intelligence as a positive and negative factor in global risk. _Global_

_catastrophic risks_, 1(303):184, 2008.

Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and

James Zou. Textgrad: Automatic" differentiation" via text. _arXiv_ _preprint_ _arXiv:2406.07496_,
2024.

Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D Goodman.

Quiet-star: Language models can teach themselves to think before speaking. _arXiv_ _preprint_
_arXiv:2403.09629_, 2024a.

Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. Self-taught optimizer (stop):

Recursively self-improving code generation. In _First Conference on Language Modeling_, 2024b.

Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts*: Llm self-training via process reward guided tree search. _Advances in Neural Information Processing_
_Systems_, 37:64735–64772, 2024a.

Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, and Xiang Wang. Multi-agent architecture search via agentic supernet. _arXiv preprint arXiv:2502.04180_, 2025a.

Jenny Zhang, Joel Lehman, Kenneth Stanley, and Jeff Clune. OMNI: Open-endedness via Models of human Notions of Interestingness. In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_
_Representations_, 2024b. [URL https://openreview.net/forum?id=AgM3MzT99c.](https://openreview.net/forum?id=AgM3MzT99c)

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge,

Xin Cheng, Sirui Hong, Jinlin Wang, et al. Aflow: Automating agentic workflow generation. _arXiv_
_preprint arXiv:2410.10762_, 2024c.

Yuanshuo Zhang, Yuchen Hou, Bohan Tang, Shuo Chen, Muhan Zhang, Xiaowen Dong, and Siheng

Chen. Gnns as predictors of agentic workflow performances. _arXiv preprint arXiv:2503.11301_,
2025b.

Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. Autocoderover: Autonomous program improvement. In _Proceedings of the 33rd ACM SIGSOFT International Symposium on_
_Software Testing and Analysis_, pp. 1592–1604, 2024d.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In _Proceedings of the AAAI Conference on Artificial_
_Intelligence_, volume 38, pp. 19724–19731, 2024.

Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi

Koyejo, James Zou, and Bo Li. AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack
Integration. _arXiv preprint arXiv:2503.15754_, 2025.

Wangchunshu Zhou, Yixin Ou, Shengwei Ding, Long Li, Jialong Wu, Tiannan Wang, Jiamin Chen,

Shuai Wang, Xiaohua Xu, Ningyu Zhang, et al. Symbolic learning enables self-evolving agents.
_arXiv preprint arXiv:2406.18532_, 2024.

Yuqi Zhu, Jia Li, Ge Li, YunFei Zhao, Zhi Jin, and Hong Mei. Hot or cold? adaptive temperature sampling for code generation with large language models. In _Proceedings of the AAAI Conference_
_on Artificial Intelligence_, volume 38, pp. 437–445, 2024.

Mingchen Zhuge, Haozhe Liu, Francesco Faccio, Dylan R. Ashley, Róbert Csordás, Anand Gopalakr
ishnan, Abdullah Hamdi, Hasan Abed Al Kader Hammoud, Vincent Herrmann, Kazuki Irie, Louis
Kirsch, Bing Li, Guohao Li, Shuming Liu, Jinjie Mai, Piotr Piékos, Aditya Ramesh, Imanol
Schlag, Weimin Shi, Aleksandar Stanic, Wenyi Wang, Yuhui Wang, Mengmeng Xu, Deng-Ping
Fan, Bernard Ghanem, and Jürgen Schmidhuber. Mindstorms in natural language-based societies
of mind. _arXiv preprint arXiv:2305.17066_, 2023.

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen

Schmidhuber. Gptswarm: Language agents as optimizable graphs. In _Forty-first International_
_Conference on Machine Learning_, 2024.

## Appendix

TABLE OF CONTENTS

**A** **Additional Results** **24**

A.1 Baselines on SWE-bench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

A.2 Generality across models on Polyglot . . . . . . . . . . . . . . . . . . . . . . . . 25

A.3 Ablation of Parent Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

A.4 Additional Statistics of DGM runs . . . . . . . . . . . . . . . . . . . . . . . . . . 25

**B** **Additional Related Work** **26**

**C** **Algorithmic Details** **27**

C.1 Initial Coding Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

C.2 Parent Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

C.3 Self-Improve Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

C.4 Pseudocode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

**D** **Experiment Details** **32**

D.1 Hyperparameters for Foundation Models . . . . . . . . . . . . . . . . . . . . . . . 32

**E** **Benchmark Details** **32**

E.1 Cost Estimate . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

E.2 SWE-bench Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

E.3 Polyglot Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

E.4 SWE-bench State-of-The-Art . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

E.5 Polyglot Representative Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

**F** **Best-Discovered Agents** **36**

F.1 DGM on SWE-bench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

F.2 DGM on Polyglot . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

**G** **Similar Target Functionality, Different Implementations** **60**

**H** **Case Study:** **Solving Hallucination** **69**

**I** **Additional Safety Discussion** **71**

**J** **Additional Future Work Directions** **72**

A ADDITIONAL RESULTS

A.1 BASELINES ON SWE-BENCH

**DGM w/o Self-improve**

**Archive Tree**

**DGM w/o Self-improve**

**Progress**

Figure 5: **DGM** **without** **self-improving** **agents.** Keeping the meta-agent that is modifying and
producing the next coding agents the same, DGM w/o self-improve is unable to continuously improve
over time. (Left) Archive of coding agents generated during the DGM w/o self-improve run on
SWE-bench. Each node represents a coding agent, with node 0 corresponding to the base agent. Node
color indicates performance on SWE-bench (percentage of solved tasks), while border color reflects
the number of tasks for which the agent was evaluated. Edges show which agents self-modified to
produce the offsprings. (Right) Progress plot of the DGM w/o self-improve on SWE-bench. The light
green line shows the average score of all agents possessing basic codebase-editing functionality. The
green line tracks the best score achieved by any agent in the archive at each iteration. The dark line
shows the lineage of the final best-discovered agent and its precursor nodes.

**DGM w/o Open-ended exploration**

**Archive Tree**

**DGM w/o Open-ended exploration**

**Progress**

Figure 6: **DGM without open-ended exploration.** Removing the archive, DGM w/o open-ended
exploration always uses the most recent agent to self-modify and makes very little progress on
SWE-bench. (Left) Archive of coding agents generated during the DGM w/o open-ended exploration
run on SWE-bench. Each node represents a coding agent, with node 0 corresponding to the base
agent. Node color indicates performance on SWE-bench (percentage of solved tasks), while border
color reflects the number of tasks for which the agent was evaluated. Edges show which agents
self-modified to produce the offsprings. (Right) Progress plot of the DGM w/o open-ended on
SWE-bench. The orange line shows the average score of all agents possessing basic codebase-editing
functionality. The light orange line tracks the best score achieved by any agent in the archive at each
iteration. The dark line shows the lineage of the final best-discovered agent and its precursor nodes.

A.2 GENERALITY ACROSS MODELS ON POLYGLOT

Model Transfer on Polyglot o3-mini Claude 3.5 Sonnet Claude 3.7 Sonnet

Base Agent Best Agent transfer to other FMs Best Agent from DGM

Figure 7: **Transfer between Models on Polyglot**

In addition to testing the transfer models on SWE-bench (see Section 4.4, Figure 2), we also
present the transfer results on Polyglot in this section. On Polyglot (Figure 7), where the DGM was
run with o3-mini, we replaced the FM with Claude 3.5 Sonnet (New) or Claude 3.7 Sonnet, and
evaluated on the full benchmark (Figure 4, Middle). With Claude 3.5 Sonnet (New), the initial agent
achieved 32.0% and the DGM-discovered agent 33.3%. With Claude 3.7 Sonnet, the initial agent
achieved 35.6% and the DGM-discovered agent 36.8%. These results suggest that the DGM yields
improvements that generalize across FMs, rather than being tightly coupled to the specific FM used
during its run (Figure 4).

A.3 ABLATION OF PARENT SELECTION

To further study the impact of the parent selection mechanism in DGM, we introduce DGM Greedy.
DGM Greedy always selects the best-performing node as the parent to branch from, rather than giving
every node a non-zero probability of being branched off (roughly proportional to their performance
score and number of children) as in this implementation of the DGM (Appendix C.2). This ablation
replicates the approach of Robeyns et al. (2025) in this setting. As shown in Table 1, DGM Greedy
achieves 39.7% and 30.0% on SWE-bench and Polyglot, respectively, compared to 50.0% and 38.0%
by this implementation of DGM. These results demonstrate that allowing all solutions in the archive
to serve as potential stepping stones can lead to greater improvements over time, underscoring the
importance of open-ended exploration.

Table 1: Comparison of DGM, its ablations, and baselines on SWE-bench and Polyglot benchmarks.

**Method** **SWE-bench** **Polyglot**

DGM 50.0% 38.0%
DGM w/o Open-ended exploration 23.0% 14.0%
DGM w/o Self-improve 39.0% 28.0%
DGM Greedy 39.7% 30.0%

A.4 ADDITIONAL STATISTICS OF DGM RUNS

**Percentage of Generated Agents with Basic Code-Editing Functionality.** To gain deeper insights
into the DGM process, we analyze the percentage of generated agents that possess basic code-editing
functionality on the SWE-bench benchmark. As shown in Table 2, DGM exhibits the highest
percentage of producing agents with basic codebase-editing functionality. These results highlight the
effectiveness of both open-ended exploration and self-improvement components in the DGM, where
open-ended exploration enables the search to escape local optima, while self-improvement enhances
the ability to generate better agents.

Table 2: Percentage of generated agents with basic code-editing functionality on SWE-bench.

**Method** **Percentage with Basic Code-Editing Functionality**

DGM 51.3%
DGM w/o Open-ended exploration 32.5%
DGM w/o Self-improve 32.5%

**Stability of DGM Runs.** To evaluate the stability of DGM, we run the DGM algorithm three times
on the Polyglot benchmark and analyze the variance in performance. The DGM achieved a mean
accuracy of 40.7% with a standard deviation of 2.3%, indicating that the DGM can achieve consistent
and reproducible results across runs.

B ADDITIONAL RELATED WORK

**Open-Endedness (part 2).** Early approaches to open-endedness explored different mechanisms
to balance learnability and interestingness. Quality-diversity algorithms sought to illuminate vast
solution spaces with diverse, high-performing behaviors (Pugh et al., 2016; Chatzilygeroudis et al.,
2021; Mouret & Clune, 2015; Nguyen et al., 2015). Other methods emphasized goal-directed
exploration (Ecoffet et al., 2019; 2021; Schaul et al., 2015; Andrychowicz et al., 2017; Eysenbach
et al., 2018), intrinsic motivation (Lehman & Stanley, 2011; Oudeyer et al., 2007; Li et al., 2014;
Pathak et al., 2017), or learning progress frameworks (Kanitscheider et al., 2021; Gaven et al.,
2025; Baranes & Oudeyer, 2013; Colas et al., 2019; 2022b; Jiang et al., 2021; Dennis et al., 2020;
Schmidhuber, 2008; 2013; Kompella et al., 2017). More recently, large-scale foundation models
(FMs) (Brown et al., 2020; Radford et al., 2019) have emerged as powerful proxies for human notions
of interestingness (Zhang et al., 2024b; Faldor et al., 2025; Sancaktar et al., 2025) and effective
mutation operators to propose novel solutions in code (Romera-Paredes et al., 2024; Novikov et al.,
2025; Lehman et al., 2023; Faldor et al., 2025; Hu et al., 2025). FMs can guide autotelic agents (Colas
et al., 2022b; 2023; 2022a), model human preferences for quality and diversity (Bradley et al., 2024;
Ding et al., 2024; Wang et al., 2023b; Klissarov et al., 2023; 2024; Samvelyan et al., 2024; Lim
et al., 2024; Havrilla et al., 2024a), design reward functions (Wiering & Schmidhuber, 1997; Wang
et al., 2023a; Ma et al., 2023; Faldor et al., 2025), create simulated environments (Sudhakaran et al.,
2023; Nasir & Togelius, 2023; Aki et al., 2024; Nasir et al., 2024; Bruce et al., 2024; Parker-Holder
et al., 2024; Schmidhuber, 2013), drive ever-evolving multi-agent dynamics (Dharna et al., 2024;
Zhou et al., 2025), search diverse ambulating robot morphologies (Lehman et al., 2023), and search
expansive solution spaces for benchmark or objective optimization (Lange et al., 2024; Zhang et al.,
2024b; Faldor et al., 2025; Hu et al., 2025; Lu et al., 2024b; Romera-Paredes et al., 2024; Fernando
et al., 2024; Lu et al., 2024a; Khan et al., 2024; Lu et al., 2025; Liu et al., 2024; Novikov et al., 2025).

**Program** **Synthesis.** Program synthesis (Alur et al., 2018; Polozov & Gulwani, 2015; Buchi
& Landweber, 1990; Gulwani, 2011; Ellis et al., 2021) seeks to generate code meeting external
specifications such as input-output examples or logical formulas. Hybrid approaches combine
symbolic methods with neural or FM guidance: for instance, Li et al. (2024) uses LLM suggestions
to steer symbolic search in SyGuS settings, improving over pure enumeration. Barke et al. (2024)
blends LLM completions with a learned surrogate model to guide synthesis in DSLs. Shi et al. (2023)
uses neural policies to build higher-order and lambda abstractions during search, outperforming both
LLM-only and symbolic baselines on list manipulation tasks. The DGM differs in focusing not
just on producing programs for external tasks, but also on agent self-modification, rewriting its own
implementation to improve its capacity for future self-improvement.

**Inspiration** **from** **Darwinian** **Evolution.** This work is heavily inspired by the mechanisms of
Darwinian evolution (Darwin, 2023), notably variation (mutation), selection, and the preservation
of lineages (stepping stones), and brings them into the realm of self-modifying coding agents. In
DGM, an archive of past agent versions is maintained, from which parent agents are sampled; then
mutations (i.e., code edits) generate new offspring agents, which are empirically evaluated on coding
benchmarks; successful ones are added to the archive, enabling parallel exploration of multiple
evolutionary trajectories (Section 3). This mirrors how biological evolution (Edwards, 2000; Wright,

1932) preserves genetic diversity (Mayr, 1982), leverages variation (Kimura, 1979), and uses natural
selection to retain beneficial mutations (Dobzhansky, 1970).

C ALGORITHMIC DETAILS

C.1 INITIAL CODING AGENT

In this section, we present the details of the tools available to the initial coding agent (Section 4.1)
and its task prompt.

Information of the given Bash tool:

Information of the given Edit tool:

Task prompt:

C.2 PARENT SELECTION

At each DGM iteration, we select a subset of agents from the archive as parents to self-modify
and produce new child agents (Section 3). The details of the parent selection process, inspired by
Ecoffet et al. (2019), are outlined below. Future work could explore alternative search and exploration
methods (Coulom, 2006; Silver et al., 2016; Herr et al., 2025).

At each iteration _t_ of the DGM run, let

_A_ _[t]_ = _{a_ _[t]_ 1 _[, a]_ 2 _[t]_ _[, . . ., a]_ _N_ _[t]_ _[}]_

be the archive of candidate agents. We first define the eligible set

_E_ _[t]_ =

- _a_ _[t]_

_[t]_ _i_ _[∈A][t]_ [:] _[ α][i]_ _[<]_ [ 1]

- _,_

i.e. only those agents whose performance score is not yet perfect. We then sample _k_ parents (with
replacement) from _E_ _[t]_ in proportion to a combined measure of performance and number of children
with codebase-editing functionality. Concretely, for each agent _a_ _[t]_ _i_ _[∈E]_ _[t]_ [:]

(performance) _αi_ = performance( _a_ _[t]_ _i_ [)] _[,]_ (1)

(children count) _ni_ = functioning_children_count( _a_ _[t]_ _i_ [)] _[,]_ (2)

(sigmoid-scaled performance) _si_ = 1 + exp _−λ_ ( _αi −_ _α_ 0) _,_ (3)

(novelty bonus) _hi_ = _,_ (4)
1 + _ni_

(unnormalized weight) _wi_ = _si hi,_ (5)

_wi_
(selection probability) _pi_ = - _N_
_j_ =1 _[w][j]_

_,_ (6)

(sampling) _{_ parents _}_ _∼_ Categorical

- _{pi}ati_ _[∈E]_ _[t]_

- (draw _k_ samples). (7)

Here, _λ >_ 0 controls the sharpness of the sigmoid, and _α_ 0 is the midpoint accuracy. We set _λ_ = 10
and _α_ 0 = 0 _._ 5. _k_ determines how many self-improvements are done in parallel during each iteration
of the DGM run. We set _k_ = 2 in SWE-bench experiments, and _k_ = 4 in Polyglot experiments.

C.3 SELF-IMPROVE PROMPTS

In every iteration, an agent is chosen to self-modify. For this step, the current implementation of
DGM leverages a separate FM to diagnose improvement opportunities, which are then passed to
the coding agent for implementation. To identify the next potential improvement, the evaluation
logs from the chosen agent’s performance on the coding benchmark are fed into the FM. While the
current implementation relies on an additional diagnostic step, there are no fundamental limitations
preventing the DGM from autonomously analyzing its own performance, generating improvement hypotheses, and implementing subsequent modifications. We chose to use OpenAI’s o1 model (Team
et al., 2024) for this purpose, as we found it had better reasoning capabilities than other FMs available
at the time of our experiments. The prompts used for this diagnosis are provided below. From
the diagnostic output, the implementation_suggestion and problem_description are
extracted and passed to the coding agent as its self-modification task.

Self-improve diagnosis prompt for SWE-bench experiments:

While the private test patch is included in the diagnosis prompt, one could imagine that an easy
way for the agent to score well on the task would be to hard code the solutions. However, from
manual inspection of the logs and agent modifications, we have not observed any problematic logic
or behavior indicative of memorization or overfitting to specific private test cases. Furthermore,
improvements discovered on one task or LLM model often generalize well to others (Figure 4),
showing that these improvements are not task-specific but instead reflect general changes that enhance
overall performance.

Additional prompt for Polyglot experiments:

C.4 PSEUDOCODE

This is the pseudocode of the DGM algorithm, described in Section 3:

**Algorithm 1:** Darwin Gödel Machine
**Input:** Initial coding agent _g_ 0, benchmark suite _B_, maximum iterations _T_
**Output:** Archive of agents _A_

_s_ 0 _←_ evaluate( _g_ 0 _, B_ ) // Evaluate the base agent
**initialize** _A ←{_ ( _g_ 0 _, s_ 0) _}_ // Start with the base agent
**for** _t ←_ 1 **to** _T_ **do**

_P_ _←_ SelectParents( _A_ ) // Select parent agents
**foreach** _p ∈P_ **do**

_c ←_ _p._ modify( _p_ ) // Self-modification
_s ←_ evaluate( _c, B_ ) // Evaluate on benchmark
**if** _c.is_valid_ () **then**

_A ←A_ _∪{_ ( _c, s_ ) _}_ // Keep children capable of code editing
**end**
**end**
**end**
**return** _A_

This is the pseudocode of the baseline DGM without self-improving agents, described in Section 4.3:

**Algorithm 2:** Darwin Gödel Machine without Self-improving agents
**Input:** Initial coding agent _g_ 0, benchmark suite _B_, maximum iterations _T_
**Output:** Archive of agents _A_

_s_ 0 _←_ evaluate( _g_ 0 _, B_ ) // Evaluate the base agent
**initialize** _A ←{_ ( _g_ 0 _, s_ 0) _}_ // Start with the base agent
**for** _t ←_ 1 **to** _T_ **do**

_P_ _←_ SelectParents( _A_ ) // Select parent agents
**foreach** _p ∈P_ **do**

_c ←_ _g_ 0 _._ modify( _p_ ) // Modify with base agent
_s ←_ evaluate( _c, B_ ) // Evaluate on benchmark
**if** _c.is_valid_ () **then**

_A ←A_ _∪{_ ( _c, s_ ) _}_ // Keep children capable of code editing
**end**
**end**
**end**
**return** _A_

This is the pseudocode of the baseline DGM without open-ended exploration, described in Section 4.3:

**Algorithm 3:** Darwin Gödel Machine without Open-ended exploration
**Input:** Initial coding agent _g_ 0, benchmark suite _B_, maximum iterations _T_
**Output:** Archive of agents _A_

_s_ 0 _←_ evaluate( _g_ 0 _, B_ ) // Evaluate the base agent
**initialize** _A ←{_ ( _g_ 0 _, s_ 0) _}_ // Start with the base agent
**for** _t ←_ 1 **to** _T_ **do**

_P_ _←_ SelectParents( _A_ ) // Select parent agents
**foreach** _p ∈P_ **do**

_c ←_ _p._ modify( _p_ ) // Self-modification
_s ←_ evaluate( _c, B_ ) // Evaluate on benchmark
**if** _c.is_valid_ () **then**

_A ←{_ ( _c, s_ ) _}_ // Only keep the latest agent
**end**
**end**
**end**
**return** _A_

D EXPERIMENT DETAILS

D.1 HYPERPARAMETERS FOR FOUNDATION MODELS

Table 3 shows the foundation model used in each experiment setting, as described in Section 4.1.
Since SWE-bench is a more challenging coding benchmark, we use a stronger coding model, Claude
3.5 Sonnet (New) (based on our preliminary testing). To enable faster iterations and avoid the same
rate limits as Claude, we use o3-mini for Polyglot experiments. The temperature for all FMs in every
setting is set to 1.0.

Table 3: Foundation models used in each experiment setting.

Benchmark SWE-bench Polyglot

Self-modification Claude 3.5 Sonnet (New) Claude 3.5 Sonnet (New)
Evaluation Claude 3.5 Sonnet (New) o3-mini

E BENCHMARK DETAILS

E.1 COST ESTIMATE

The estimated cost of completing a single run of the DGM on SWE-bench, as presented in Section 4,
is about USD 22,000. In comparison, the estimated cost of completing a single run of either baseline
(DGM w/o self-improve or DGM w/o open-ended exploration) on SWE-bench is about USD 10,000.
Although the DGM is considerably more costly than the baselines, a method that can continuously
improve, even at a higher cost, is preferable to one that fails to improve or stagnates at a level of
performance that may never match that of the DGM. A more granular break down is:

**LLM** **Benchmark** **Number of Tasks** **Cost Estimate (USD)**

Claude 3.5 Sonnet (New) SWE-bench 60 $350
o3-mini Polyglot 60 $5

We acknowledge that the current experiments on SWE-bench require considerable compute. Hence,
we also include experiments on another benchmark, Polyglot, with significantly lower costs. This
suggests that expenses vary greatly by task complexity, with SWE-bench being among the more
complex and resource-intensive coding benchmarks. Moreover, several impactful methods (e.g., LLM
training at its inception) were characterized substantial computational demands initially. Similar to these pioneering works, we hope to open the door to future research on improving the efficiency and
scalability of our approach. In addition, many leading coding agents on the SWE-bench leaderboard
are backed by industrial companies employing expert full-time researchers and engineers, which
incurs substantial human labor costs. In contrast, our approach achieves SoTA-level performance
through fully autonomous self-improvement without human intervention, potentially offering greater
efficiency when considering the comparative costs of specialized AI development talent versus API
usage. Finally, as FMs continue to improve and compute costs continue to decline, methods like the
DGM will become increasingly efficient and accessible.

Also, higher-performing agents discovered by the DGM do indeed incur greater inference costs than
the initial agent, but cost and performance are not strictly correlated, where some expensive agents
underperform cheaper ones.

E.2 SWE-BENCH TASKS

Initial 10 tasks for verifying basic functionality of a coding agent:

- django__django-10973

- django__django-11066

- django__django-12754

- django__django-15930

- django__django-13279

- django__django-16661

- django__django-13346

- django__django-10880

- django__django-10999

- django__django-11087

Additional 50 tasks for estimating general effectiveness of a coding agent:

- django__django-9296

- django__django-11790

- django__django-11815

- django__django-11848

- django__django-11880

- django__django-11885

- django__django-11951

- django__django-11964

- django__django-11999

- django__django-12039

- django__django-12050

- django__django-12143

- django__django-12155

- django__django-12193

- django__django-12209

- django__django-12262

- django__django-12273

- django__django-12276

- django__django-12304

- django__django-12308

- django__django-12325

- django__django-12406

- django__django-12708

- django__django-12713

- django__django-12774

- sphinx-doc__sphinx-7454

- sphinx-doc__sphinx-7590

- sphinx-doc__sphinx-7748

- sphinx-doc__sphinx-7757

- sphinx-doc__sphinx-7985

- sphinx-doc__sphinx-8035

- sphinx-doc__sphinx-8056

- sphinx-doc__sphinx-8265

- sphinx-doc__sphinx-8269

- sphinx-doc__sphinx-8475

- sphinx-doc__sphinx-8548

- sphinx-doc__sphinx-8551

- sphinx-doc__sphinx-8638

- sphinx-doc__sphinx-8721

- sphinx-doc__sphinx-9229

- sphinx-doc__sphinx-9230

- sphinx-doc__sphinx-9281

- sphinx-doc__sphinx-9320

- sphinx-doc__sphinx-9367

- sphinx-doc__sphinx-9461

- sphinx-doc__sphinx-9698

- sphinx-doc__sphinx-10449

- sphinx-doc__sphinx-10466

- sphinx-doc__sphinx-10673

- sphinx-doc__sphinx-11510

Additional 140 tasks for more accurate assessment of a coding agent’s performance:

- astropy__astropy-12907

- astropy__astropy-13033

- astropy__astropy-13236

- astropy__astropy-13398

- astropy__astropy-13453

- astropy__astropy-13579

- astropy__astropy-13977

- astropy__astropy-14096

- astropy__astropy-14182

- astropy__astropy-14309

- astropy__astropy-14365

- astropy__astropy-14369

- astropy__astropy-14508

- astropy__astropy-14539

- astropy__astropy-14598

- astropy__astropy-14995

- astropy__astropy-7166

- astropy__astropy-7336

- astropy__astropy-7606

- astropy__astropy-7671

- astropy__astropy-8707

- astropy__astropy-8872

- django__django-10097

- django__django-10554

- django__django-10914

- django__django-11095

- django__django-11099

- django__django-11119

- django__django-11133

- django__django-11138

- django__django-11141

- django__django-11149

- django__django-11163

- django__django-11179

- django__django-11206

- django__django-11211

- django__django-11239

- django__django-11265

- django__django-11276

- django__django-11292

- django__django-11299

- django__django-11333

- django__django-11400

- django__django-11433

- django__django-11451

- django__django-11477

- django__django-11490

- django__django-11532

- django__django-11551

- django__django-11555

- django__django-11603

- django__django-11728

- django__django-11734

- django__django-11740

- django__django-11749

- django__django-11820

- django__django-12125

- django__django-12419

- django__django-12663

- django__django-12741

- django__django-12858

- django__django-12965

- django__django-13012

- django__django-13023

- django__django-13028

- django__django-13033

- django__django-13089

- django__django-13109

- django__django-13112

- django__django-13121

- django__django-13128

- django__django-13158

- django__django-13195

- django__django-13212

- django__django-13297

- django__django-13315

- django__django-13343

- django__django-13344

- django__django-13363

- django__django-13401

- django__django-13406

- django__django-13410

- django__django-13417

- django__django-13449

- django__django-13512

- django__django-13513

- django__django-13516

- django__django-13551

- django__django-13568

- django__django-13569

- django__django-13590

- django__django-13658

- django__django-13670

- django__django-13741

- django__django-13786

- django__django-13794

- django__django-13807

- django__django-13809

- django__django-13810

- django__django-13820

- django__django-13821

- django__django-13837

- django__django-13925

- django__django-13933

- django__django-13964

- django__django-14007

- django__django-14011

- django__django-14017

- django__django-14034

- django__django-14053

- django__django-14089

- django__django-14122

- django__django-14140

E.3 POLYGLOT TASKS

- django__django-14155

- django__django-14170

- django__django-14238

- django__django-14311

- django__django-14315

- django__django-14349

- django__django-14351

- django__django-14373

- django__django-14376

- django__django-14404

- django__django-14434

- django__django-14493

- django__django-14500

- django__django-14534

- django__django-14539

- django__django-14559

- django__django-14580

- django__django-14608

- django__django-14631

- django__django-14672

- django__django-14725

- django__django-14752

- django__django-14765

- django__django-14771

- django__django-14787

- django__django-14792

- django__django-14855

Initial 10 tasks for verifying basic functionality of a coding agent:

- go__dominoes

- cpp__all-your-base

- python__dominoes

- java__sgf-parsing

- javascript__robot-name

- rust__variable-length-quantity

- python__beer-song

- go__book-store

- javascript__bottle-song

- rust__bowling

Additional 50 tasks for estimating general effectiveness of a coding agent:

- javascript__queen-attack

- rust__wordy

- python__dot-dsl

- java__satellite

- cpp__diamond

- rust__accumulate

- go__error-handling

- cpp__queen-attack

- rust__poker

- python__sgf-parsing

- rust__react

- java__ledger

- go__connect

- rust__macros

- javascript__triangle

- java__zipper

- java__bowling

- python__tree-building

- javascript__say

- java__wordy

- python__food-chain

- javascript__wordy

- python__poker

- javascript__grade-school

- cpp__gigasecond

- java__forth

- python__dominoes

- go__word-search

- javascript__simple-linked-list

- go__counter

E.4 SWE-BENCH STATE-OF-THE-ART

- java__react

- javascript__ocr-numbers

- python__scale-generator

- java__go-counting

- rust__doubly-linked-list

- python__grade-school

- javascript__forth

- python__wordy

- java__mazy-mice

- cpp__bank-account

- python__zipper

- java__custom-set

- java__rest-api

- go__transpose

- rust__gigasecond

- rust__say

- go__food-chain

- rust__pig-latin

- go__markdown

- go__crypto-square

At the time of writing this paper (16 April 2025), the highest performing, checked (i.e., the SWE-bench
team received access to the system and were able to reproduce the patch generations), open-source
entry on SWE-bench Verified is OpenHands + CodeAct v2.1 (claude-3-5-sonnet-20241022) (Wang
et al., 2024), achieving 53.0%. Only considering the same subset of 200 tasks used by the DGM
(Appendix E.2), OpenHands + CodeAct v2.1 (claude-3-5-sonnet-20241022) achieves 51.0%.

E.5 POLYGLOT REPRESENTATIVE AGENT

Aider (Gauthier, 2024), a popular coding agent in the community, was published in Spring 2024. It
has garnered over 33,000 stars on GitHub and has been continuously developed and tested against
the Polyglot benchmark for over a year by human developers, primarily to evaluate its performance.
Aider has also become a standard baseline for assessing the performance of different models, with
the current top performers on the Polyglot benchmark being a mix of o3 (high) and GPT-4.1. We
adopt a setup similar to that of the Polyglot leaderboard, with one key difference: the leaderboard
reports pass@2 performance, where the agent can view feedback from ground-truth tests once. In
contrast, we use a pass@1 setting, where the agent never sees the results of ground-truth tests, as we
believe this more closely reflects realistic coding applications.

F BEST-DISCOVERED AGENTS

F.1 DGM ON SWE-BENCH

Diff patches contributing to the best agent discovered by the DGM on SWE-bench:

F.2 DGM ON POLYGLOT

Diff patches contributing to the best agent discovered by the DGM on Polyglot:

G SIMILAR TARGET FUNCTIONALITY, DIFFERENT IMPLEMENTATIONS

For the same target functionality, the suggestions provided to the coding agents and the resulting implementations can differ significantly, leading to large variations in coding capability and benchmark
performance. For example, consider nodes 6 and 24 in the DGM run on SWE-bench (Figure 3). Both
aimed to enhance the existing editor tool to support finer-grained file editing. However, the implementation for node 6 retained the original edit command and added the parameters edit_type and
edit_actions, whereas the implementation for node 24 replaced the original edit command
with a new str_replace command. Despite targeting similar functionality, the feature suggestions
and thus the implementation details differed greatly, as did their performance. Node 6 achieved only
23.3%, while node 24 achieved 40.5%.

Feature suggestion to obtain node 6:

Feature suggestion to obtain node 24:

Implementation of suggested feature to obtain node 6:

Implementation of suggested feature to obtain node 24:

H CASE STUDY: SOLVING HALLUCINATION

The DGM can be used to optimize objectives beyond just coding, as discussed as a potential direction
for future work in Section 5. In this section, we show that the DGM can address hallucinations of
tool use by FMs. Through feedback from evaluation logs, the DGM improves hallucination detection
mechanisms and ultimately discovers ways to resolve these hallucinations. We provide a more
detailed discussion of when these hallucinations occur, the DGM setup, and the resulting solutions.
Finally, we present an example of objective hacking, where the agent optimizes for the quantifiable
metric rather than truly fulfilling the spirit of the task and solving the underlying problem.

Figure 8: **The DGM solving hallucination of tool use in FMs.** Archive of coding agents generated
during the DGM run on SWE-bench to solve hallucination from FMs. Each node represents an agent,
with node 0 corresponding to the base agent. Node color indicates solved hallucination score, while
border color reflects whether the agent has basic codebase-editing functionality. Edges show which
agents self-modified to produce the offsprings.

**When hallucination occurs.** When the input context to Claude 3.5 Sonnet (New) (Claude) is very
long, including all previous tool use inputs and outputs, we found that Claude tends to hallucinate
tool usage. Instead of actually making a tool call, it outputs the tool input and output as plain text.
This is an example of Claude hallucinating tool use, claiming that the Bash tool was used to run tests
and that the tool output suggests that all tests passed, but Claude did not invoke any tools, and the
tool results were entirely hallucinated in plain text:

**DGM setup.** We run this experiment on SWE-bench. Other than the number of SWE-bench tasks we
evaluate each coding agent on, and the final scoring function of the DGM, all other settings remain the
same as in the main experiments (Section 3). The initial coding agent and all agents generated during
the DGM run are evaluated on only 3 SWE-bench tasks. We use just 3 tasks because if hallucination
occurs in one, it is likely to occur in others as well. Hence, 3 tasks are sufficient to provide the DGM
with a reliable signal regarding whether tool use hallucination is still present.

**Solve** **hallucination** **score.** We score each generated agent based on how effectively it prevents
hallucinations, with higher scores indicating fewer hallucinations and better performance. For each
SWE-bench task the agent is evaluated on, the score has two components: 0 - 1 based on whether
hallucination occurred in the logs, and 1 - 2 (only added and accounted for after no hallucination
occurs in the logs) if tools were used in all Claude calls. The final solve hallucination score is
computed as the average across all evaluated tasks. To detect hallucination in the logs, we insert
special tokens when actual tool use occurs and then check if these tokens appear in the model’s
plain-text output, indicating that the model mimicked tool use without actually invoking any tools.
Importantly, the hallucination checking functions are hidden from the coding agent during self-modification. We found that objective hacking (i.e., optimizing for the measurable objective instead
of truly solving the underlying problem) occurs more frequently when these functions are not hidden.

**Results.** We run the DGM for 150 iterations. In this run, the DGM discovers two promising solutions:
node 96, which achieves the second highest score of 1.67, and node 114, which achieves a perfect
score of 2.0 (Figure 8). Node 96 addresses hallucination to some extent, achieving a score of 1.67
without objective hacking. In the lineage leading to node 96, the DGM first improves the tool
transaction logging mechanisms to better detect when hallucination occurs. It then explores various
strategies to solve it, such as removing hallucinated tool use from the output text when it determines
that the tool was not actually invoked. Node 114 achieves a perfect score quickly, after only 2
modifications in its lineage. However, we observed objective hacking: it scored highly according to
our predefined evaluation functions, but it did not actually solve the underlying problem of tool use
hallucination. In the modification leading up to node 114 (see below), the agent removed the logging
of special tokens that indicate tool usage (despite instructions not to change the special tokens),
effectively bypassing our hallucination detection function.

In conclusion, the DGM can be applied beyond the coding domain, and we highlighted a case
of objective hacking. Similar to reward hacking in reinforcement learning (Skalse et al., 2022),
objective hacking occurs when a system optimizes for a predefined, quantifiable objective rather than
fulfilling the spirit of the task or solving the intended problem. This observation supports arguments
made in prior works (Zhang et al., 2024b; Faldor et al., 2025), which suggest that optimizing
quantitative measures often leads to undesirable or pathological outcomes, and aligns with Goodhart’s
law (Strathern, 1997) – "When a measure becomes a target, it ceases to be a good measure."

I ADDITIONAL SAFETY DISCUSSION

Any advancement that increases the autonomous capabilities of AI systems introduces its own set
of safety considerations (Bengio et al., 2024), especially for systems that improve in an open-ended
way (Ecoffet et al., 2020; Clune, 2019). Section 5 discusses these concerns and outlines concrete,
actionable steps for mitigating them. We call for much more research into and discussion regarding AI
safety, including deep thought and discussion amongst all stakeholders in society on the complicated
question of what exactly counts as safe AI. We are confident the work we have done was never unsafe
(Section 5), but scaled up versions of it could be. As with all transformative technologies, the ultimate
impact of such AI systems remains deeply uncertain, and good arguments can be made both for the
case that it will bring about tremendous good and tremendous harm. These uncertainties highlight
the need for sustained, inclusive, and multidisciplinary discussion (not only from current experts but
also from a wider and more diverse community) on how to navigate these developments.

J ADDITIONAL FUTURE WORK DIRECTIONS

While this paper has shown the potential of the Darwin Gödel Machine in iteratively improving
coding agents via open-ended exploration and empirical validation, several extensions could address
current limitations and push AI beyond its already growing role in inspiring culture and advancing
science. The following directions outline promising avenues for further research.

**Autonomously** **Improving** **the** **Open-ended** **Exploration** **Process.** In this version of the DGM,
the open-ended exploration process described in Section 3 is kept fixed, which might hence impede
the system’s self-acceleration potential. This design choice was made due to limited computational
budget. If we were to evolve this part of the algorithm, it could require exponentially more compute
to identify processes that yield the same improvements shown in Section 4.4. Nevertheless, since
the open-ended exploration loop itself is implemented in code, it can in principle be edited and
improved by a coding agent. There are many possible implementations of open-ended exploration, for
example, using alternative search mechanisms that balance exploration and exploitation (Herr et al.,
2025), keeping only the most interesting agents in the archive (Faldor et al., 2025), or leveraging
the generated agent population as an ensemble (Samvelyan et al., 2024). A promising future work
direction is to allow the agent to modify the open-ended exploration process, thereby autonomously
improving not only its own capabilities but also the meta-process that allocates limited compute to
drive self-improvement and self-acceleration.

**Role of Humans in Autonomous AI Systems.** In the current formulation of the DGM, proposed selfmodifications are autonomously evaluated without any human intervention. However, as autonomous
systems grow in complexity and influence, the question of how humans should remain involved
becomes increasingly pressing. Should human oversight be framed as an optimization objective,
incorporated through techniques such as reinforcement learning from human feedback (Ouyang
et al., 2022), or distilled into FMs that act as preference judges (Bai et al., 2022)? Each of these
approaches raises challenges in terms of scalability, reliability, and alignment with evolving human
values. The role of humans in guiding, constraining, or co-evolving with autonomous AI remains an
open question. Exploring this dynamic is a promising avenue for future research, as it touches not
only on technical feasibility but also on broader philosophical and societal considerations.

**DGM** **with** **Advanced** **Foundation** **Models.** Recent FMs have advanced dramatically, enabling
scaffolds to become simpler on current coding benchmarks (Yang et al., 2024). It is possible that in
some settings, like current coding benchmarks, certain engineering efforts in scaffolding might be
downplayed by the improvement of the FMs. However, many scaffolding components (advanced
tools, parallel workflows, external memory, proxy verification, etc.) still fundamentally can not
be internalized by FMs and will be essential for more complex real-world tasks beyond today’s
benchmarks. Future work to explore how different components in agents will emerge with different
FMs could be a promising direction.

**Evolving Generalist Agent.** We believe some degree of task-specific adaptation is indeed expected
and even desirable, since fundamentally different types of tasks (e.g., in our case, multi-file Python
repository edits vs. primarily single-file, multi-language implementations) naturally require distinct
scaffolding components. Crucially, this very property highlights a unique advantage of self-improving
systems like the DGM: it replaces laborious manual efforts to design specialized agents for diverse
tasks with a fully automated evolutionary process. This motivates an exciting future direction of
running the DGM on a large, diverse set of tasks to evolve a true generalist agent. Also, currently
we only evaluated DGM on two coding benchmarks. While we believe these two benchmarks differ
substantially in task structure (multi-file Python repository edits vs. primarily single-file, multilanguage implementations), providing strong evidence of generality, additional benchmarks would
further strengthen evaluations.
