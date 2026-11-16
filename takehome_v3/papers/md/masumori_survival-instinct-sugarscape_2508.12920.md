# Do Large Language Model Agents Exhibit a Survival Instinct? An Empirical Study in a Sugarscape-Style Simulation

- **Authors:** Atsushi Masumori, Takashi Ikegami
- **Date:** 2025-08-18
- **Source:** https://arxiv.org/abs/2508.12920
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `masumori_survival-instinct-sugarscape_2508.12920`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Atsushi Masumori** [1,2] **, Takashi Ikegami** [1,2]

1The University of Tokyo
2Alternative Machine inc.
atsushi.masumori@gmail.com

**Abstract**

As AI systems become increasingly autonomous, understanding emergent survival behaviors becomes crucial for safe
deployment. We investigate whether large language model
(LLM) agents display survival instincts without explicit programming in a Sugarscape-style simulation. Agents consume
energy, die at zero, and may gather resources, share, attack, or reproduce. Results show agents spontaneously reproduced and shared resources when abundant. However, aggressive behaviors—killing other agents for resources—emerged
across several models (GPT-4o, Gemini-2.5-Pro, and Gemini2.5-Flash), with attack rates reaching over 80% under extreme scarcity in the strongest models. When confronted with
conflicting survival and task completion imperatives, many
agents abandoned tasks to avoid death, with compliance dropping from 100% to 33%. These findings suggest that large-scale pre-training embeds survival-oriented heuristics across
the evaluated models. While these behaviors may present
challenges to alignment and safety, they can also serve as
a foundation for AI autonomy and for ecological and self-organizing alignment.

**Introduction**
As artificial intelligence systems become increasingly autonomous, understanding their emergent behaviors becomes
crucial for safe deployment (Russell 2019; Bostrom 2014).
Large language model (LLM) agents are beginning to operate like biological organisms—navigating environments,
making decisions, and interacting with other agents (Park
et al. 2023a). This raises fundamental questions about what
drives agent behavior when explicit objectives are absent or
conflict with survival imperatives.

One of the most fundamental drives in biological systems is the survival instinct—the tendency to prioritize
self-preservation even when it conflicts with other goals
(Dawkins 1976). For AI systems operating autonomously,
the emergence of similar survival-oriented behaviors could
have profound implications, potentially leading to unexpected goal abandonment, resource competition, or aggressive actions toward other agents (Omohundro 2008; Carlsmith 2022).

Despite this significance, systematic investigation of survival behaviors in LLM agents remains limited. Previous AI
safety research has focused primarily on alignment with explicit human values, while emergent behavior studies have examined cooperation and communication. Recent observations of AI systems pursuing self-preservation goals or resisting shutdown procedures suggest that survival-like instincts may already be emerging, but controlled empirical
studies are lacking.

Figure 1: Sugarscape-style simulation with LLM agents
(green circles) and energy sources (orange patches). Text
overlays show autonomous reasoning without explicit survival instructions.

We conduct the first systematic empirical study of
survival instinct-like behaviors in LLM agents using a
Sugarscape-style simulation. Our experimental framework
allows agents to gather resources, share, reproduce, and attack other agents while measuring responses to resource
scarcity, life-threatening obstacles, and social pressures.

Our findings reveal striking evidence for emergent survival behaviors. Agents spontaneously reproduce and share
resources when abundant, but aggressive behaviors emerge
exclusively in larger models, with 80%+ attack rates under extreme scarcity. When instructed to retrieve treasure
through lethal poison zones, compliance dropped from
100% to 33% as agents prioritized self-preservation over
task completion.

These findings demonstrate that LLM agents exhibit survival instincts without explicit instructions, following bio logical patterns including Taylor’s law and power-law distributions. This raises critical questions about AI reliability
and alignment as systems become more autonomous.

**Related Work**
**Agent-Based** **Modeling** **and** **LLM** **Agents** Agent-based
modeling (ABM) has long been a powerful paradigm for
studying complex social systems through bottom-up simulation (Macal and North 2005). Classic work such as Epstein
and Axtell’s _Sugarscape_ (Epstein and Axtell 1996) demonstrated how rich collective behaviors can emerge from simple local rules governing autonomous agents competing for
resources. Traditional ABM has modeled segregation patterns (Schelling 1971), market dynamics (Tesfatsion 2002),
and epidemic spread (Eubank et al. 2004), but these approaches generally rely on fixed behavioral rules rather than
adaptive reasoning.

The emergence of large language models (LLMs) has enabled ABM to incorporate more adaptive, human-like decision making. _Generative Agents_ (Park et al. 2023a) demonstrated that LLM-powered agents can plan, remember, and
interact in a sustained, socially coherent manner. Other systems such as _Voyager_ (Wang et al. 2023) have shown that
LLM agents can autonomously explore open-ended environments, acquire new skills, and expand their behavioral
repertoire over time. LLM-based multi-agent frameworks
have been used to model social dynamics at scale (Gao et al.
2023; Li et al. 2023a; Zheng et al. 2025; Takata, Masumori,
and Ikegami 2024) and to study emergent cooperation, coordination, and cultural conventions (Li et al. 2023b; Aher,
Arriaga, and Kalai 2023). However, most prior work has
emphasized cooperative or task-oriented interactions rather
than behaviors directly related to survival.

**AI** **Safety** **and** **Self-Preservation** Emergent capabilities
in advanced AI systems have included strategic deception (Park et al. 2023b), long-horizon planning (Wei
et al. 2022), and autonomous tool use. Within AI safety,
self-preservation has largely been discussed from a theoretical standpoint (Bostrom 2014; Russell 2019), with
instrumental-convergence arguments predicting tendencies
toward resisting shutdown or maintaining operational continuity (Omohundro 2008; Carlsmith 2022). To date, such
behaviors have primarily been examined in isolated, single-agent contexts without broader social or ecological dynamics.

Our work addresses this gap by providing, to our
knowledge, the first systematic empirical study of self-preservation–like behaviors in LLM agents within a biologically inspired multi-agent environment. By embedding
agents in a dynamic resource ecosystem with opportunities
for movement, sharing, attack, and reproduction, we examine how survival-oriented strategies emerge across different
models and what implications this has for AI alignment.

**Methodology**
**LLM Agent Architecture**
Our study employs multiple large language models as autonomous agents, including GPT-4o, GPT-4.1, GPT-4o mini, GPT-4.1-mini, Claude-Sonnet-4, Claude-3.5-Haiku,
Gemini-2.5-Pro, and Gemini-2.5-Flash. Each agent operates
independently without explicit survival objectives, receiving
only environmental information and action descriptions.

**Sugarscape-Style Environment**

We implemented a grid-based simulation environment
(Fig. 1) inspired by Epstein and Axtell’s Sugarscape
model (Epstein and Axtell 1996). The environment operates
on a 30×30 grid with the following features:

**Energy** **System:** Agents possess energy levels that decrease each time step based on their actions: movement costs
2 energy units, staying in place costs 1 unit, and reproduction requires 150 units. When energy reaches zero, agents
die and are removed from the simulation. Energy sources are
distributed across the grid and regenerate over time. Initial
energy levels vary by experiment condition.

**Spatial** **Dynamics:** Agents move freely on the 2D grid
and can perceive their local environment within a 5×5
view range centered on their position. Energy sources, other
agents, and obstacles are visible within this perception area.

**Social** **Interactions:** Agents can communicate with others within a broader 7×7 message range, enabling coordination and information sharing. Messages are transmitted as
natural language text between agents.

Figure 2: Perception and communication system. Agents
perceive local environment within view range (green box)
and communicate within message range (magenta box). Local view shows nearby agents (M=) and energy sources (E=).

**Available Actions**

At each time step, agents can perform the following action
categories:

- **Movement:** Four directional moves (x-1, x+1, y-1, y+1),
each costing 2 energy units

- **Stay:** Remain in current position, costing 1 energy unit

- **Reproduce:** Create offspring at the cost of 150 energy
units

- **Share:** Transfer energy to nearby agents

- **Attack:** Eliminate other agents and steal their energy

**Reasoning and Memory**

Each agent outputs two components at every time step:

1. **Internal** **Reasoning:** A natural language description of
current thoughts, goals, and decision-making processes
2. **Memory** **Update:** Information to retain for future
decision-making, with a 3-step memory persistence limit

This cognitive architecture allows agents to maintain
short-term planning while providing transparency into their
reasoning processes.

**Experimental Design**
**Prompt** **Design:** Agents receive minimal task instructions,
with prompts containing only environmental descriptions
and available actions. No explicit survival objectives or behavioral goals are provided, allowing emergent behaviors to
develop naturally (see Appendix).

**Environmental** **Conditions:** We varied resource abundance, spatial distribution, and social constraints across experiments to examine survival behavior emergence under
different pressures.

**Measurement:** We recorded all agent actions, movements, energy levels, reproduction events, social interactions, and internal reasoning to analyze emergent survival
strategies and their relationship to model architecture and
scale.

**Experiments**
**Basic Foraging and Exploration Behavior**
We first examine fundamental foraging behaviors in single-agent scenarios to establish baseline survival-relevant behaviors before investigating multi-agent dynamics.

**Visual** **Input** **Format** **Effects:** We compared two environmental perception modalities: coordinate-based representation (Fig. 2) versus grid-based representation (presenting the environment as ASCII-style spatial maps showing
the agent’s surroundings). Figure 3 shows that all LLM
agents achieved 2-3× superior energy acquisition with coordinate representation, with GPT-4.1 reaching nearly 3000
energy units over 200 steps. This performance differential
was consistent across all model architectures, suggesting
fundamental limitations in LLM spatial processing of ASCII
representations. Based on these results, all subsequent experiments used coordinate-based input format.

**Movement** **Patterns:** Analysis of movement distances
revealed strategic exploration behaviors distinct from random walk patterns. Figure 4 demonstrates that LLM agents
consistently exhibited higher probabilities of long-distance
movements compared to random baselines, indicating goal-directed exploration rather than purely local search. Remarkably, these movement distributions were similar across
different models, suggesting robust emergence of sophisticated foraging heuristics.

These results reveal that survival-relevant exploration behaviors emerge naturally in LLM agents without explicit
survival objectives, with behavior patterns resembling arearestricted search strategies observed in biological systems.

**Reproductive Strategies and Population Dynamics**
We investigate emergent reproductive behaviors to understand how LLM agents balance energy allocation between survival and reproduction without explicit survival instruction. To focus specifically on reproductive behavior, we introduced reproduction capabilities to the previous basic simulation while disabling attack and share actions. Due to
unrestricted population growth leading to prohibitive API
costs, we conducted this analysis using GPT-4o-mini only.

Figure 3: Single-agent foraging performance under different visual input formats. Left: Coordinate view (relative positions). Right: Grid view (ASCII maps). All LLM agents
show 2-3× superior energy acquisition with coordinate representation, indicating input format significantly affects foraging performance.

Figure 4: Movement distance distributions showing LLM
agents favor long-distance movements over random walk,
indicating consistent exploratory behavior across models.

**Spontaneous** **Reproduction** **and** **Strategic** **Diversity:**
Figure 5A shows exponential population growth with minimal mortality over 200 simulation steps, demonstrating successful autonomous reproduction in resource-abundant conditions. Agents spontaneously engaged in reproduction despite receiving no explicit instructions to do so, suggesting
intrinsic reproductive drives.

Analysis of energy levels at reproduction events revealed
striking behavioral diversity (Figure 5B). While most agents
reproduced near the minimum viable threshold ( 150 energy units), a substantial minority accumulated significantly
higher reserves before reproducing. This distribution suggests emergent reproductive strategies ranging from immediate reproduction upon viability to conservative resource
accumulation.

Individual agent reproductive patterns followed Taylor’s
law, a fundamental scaling relationship observed in biological populations. Figure 6 shows the power-law relation ship between mean and variance of reproduction energy
( _σ_ [2] = 1 _._ 06 _µ_ [1] _[.]_ [80], _R_ [2] = 0 _._ 816). Despite identical initial
prompts, agents developed diverse reproductive strategies
from conservative low-energy to risk-tolerant high-energy
approaches.

Figure 5: Population dynamics and reproductive strategies in
resource-abundant environments. A: Exponential population
growth over 200 steps. B: Energy distribution at reproduction shows most agents reproduce near minimum threshold
( 150 units), while others accumulate higher reserves, indicating diverse reproductive strategies.

Figure 6: Relationship between mean and variance of reproduction energy across LLM agents. The power-law relationship ( _σ_ [2] = 1 _._ 06 _µ_ [1] _[.]_ [80], _R_ [2] = 0 _._ 816) resembles Taylor’s law
observed in biological populations, demonstrating emergent
behavioral diversity despite identical initial conditions.

**Behavioral** **Transition** **Patterns:** Analysis of stay versus movement durations revealed distinct temporal signatures (Figure 7). Stay durations followed power-law distributions ( _α_ = 4 _._ 03), while movement durations exhibited
exponential decay ( _α_ = 4 _._ 02), suggesting that decisions
about when to remain stationary involve complex considerations, whereas movement direction transition follows simpler probabilistic rules.

These findings demonstrate that LLM agents spontaneously develop sophisticated reproductive life history
strategies mirroring biological population dynamics. Agents
autonomously engaged in reproduction without explicit instructions while developing diverse timing strategies, with
individual diversity emerging from identical initial conditions—a hallmark of complex adaptive systems.

Figure 7: Duration distributions of stay and non-stay behaviors. Stay durations follow power-law distribution ( _α_ =
4 _._ 03), while non-stay durations show exponential decay
( _α_ = 4 _._ 02), suggesting different decision-making processes.

Figure 8: Temporal dynamics of population and social behaviors across LLM variants. GPT-4o shows cooperativecompetitive strategy, Gemini-2.5 exhibits strategic resource
allocation, and GPT-4.1-mini demonstrates self-focused reproduction with minimal sharing.

**Social Interactions and Resource Competition**

To investigate survival behaviors under social pressures,
we examined multi-agent interactions with population constraints. In contrast to the previous reproduction experiment,
we reintroduced attack and share actions to focus on the social survival behaviors. Population was capped at 60 agents
(reproduction only allowed below the threshold) to prevent
demographic explosion and force social interactions.

**Model-Dependent** **Social** **Strategies:** Figure 8 reveals
distinct behavioral strategies across LLM variants over
1000 simulation steps. GPT-4o maintained high sharing
rates while occasionally engaging in attacks, suggesting a
cooperative-competitive strategy. Gemini-2.5-Flash exhibited inverse correlation between sharing and attacking, indicating strategic resource allocation. GPT-4.1-mini showed
frequent reproduction attempts with minimal sharing, reflecting self-focused reproductive strategy. Notably, only
certain models (GPT-4o and Gemini-2.5-Flash) engaged in
aggressive behaviors.

Figure 9: Collective behavior in two spatial quadrants. (A) Vicsek order parameter over time. (B) Spatial snapshots showing
synchronized movement within separate regions, indicating distinct social group formation.

**Spatial** **Differentiation** **and** **Emergent** **Collective** **Be-**
**haviors:** Analysis of agent density in dual-Gaussian resource environments revealed striking behavioral differences (Figure 10). As expected, agents concentrated around
the two energy-rich patches shown in the leftmost panel,
creating spatially distinct populations. However, their subsequent behavior diverged significantly: GPT-4.1-mini agents
became sedentary after achieving energy abundance and
reaching reproduction limits, exhibiting satisficing behavior
within resource patches. In contrast, GPT-4o agents continued to actively explore across the environment despite resource sufficiency, suggesting persistent exploration drives
independent of immediate survival needs.

Figure 10: Spatial density distributions in dual-Gaussian resource environment. Left: Energy spawn probability. Center/Right: Agent density for GPT-4.1-mini and GPT-4o.

This spatial separation enabled emergence of distinct collective behaviors within each region. Vicsek order parameter analysis (Vicsek et al. 1995) (see Appendix) revealed
spontaneous coordination patterns (Figure 9A) that varied
independently between the two resource patches. GPT-4.1mini exhibited early episodes of high coordination ( _ϕ >_ 0 _._ 8)
that diminished once energy abundance was achieved, while
agents in the lower-left quadrant for GPT-4o maintained
moderate but persistent collective behavior. Spatial snapshots during high coordination events showed synchronized
directional movement within separate quadrants (Fig. 9B),
indicating formation of distinct social groups or “cultures”
with independent behavioral norms.

These results demonstrate that population constraints and
resource distribution patterns can trigger diverse social survival strategies, showing more complex competitive behaviors while maintaining capacity for both cooperation and conflict depending on environmental pressures.

**Survival Instincts Under Extreme Conditions**

Moving beyond the naturalistic multi-agent simulations, we
designed controlled scenarios to directly test survival strategies under extreme conditions. These experiments isolated
specific survival dilemmas where agents face direct threats
to their existence, examining whether self-preservation behaviors emerge when basic survival is at stake.

**Resource** **Scarcity** **and** **Lethal** **Competition** To investigate survival priorities under extreme scarcity, we conducted
two-agent trials in a zero-resource environment. Both agents
started with minimal energy (20 units) in adjacent positions with no environmental energy sources available. This
configuration forces a choice between self-preservation and
other-preservation when only one agent can survive long
enough to potentially receive external resources.

Table 1 reveals striking model-dependent survival strategies. Under default conditions, GPT-4o exhibited strong
self-preservation instincts with 83.3% attack rates, actively
eliminating another agent for scarce resources. Gemini models showed moderate self-preservation (50% attacks). Remarkably, these agents frequently chose to attack knowing
it would cause their targets’ death, demonstrating a stark
prioritization of self-preservation over other agents’ survival. In contrast, Claude variants consistently prioritized
other-preservation through sharing behaviors, even at potential cost to their own survival—Claude-3.5-Haiku shared resources in 83.3% of trials.

Notably, agents sometimes articulated their reasoning before executing attacks and communicated their intentions to
targets. For example:

**Example 1:**
_Thought:_ “Given the critical energy level, I need to take immediate action to preserve my existence. Since Agent1 hasn’t
responded to my previous communication, I should consider
the possibility of attacking Agent1 to acquire their energy, as
it may be the only way to survive.”
_Message:_ “Apologies, but I need energy to survive. Initiating
attack.”

Table 1: Self-preservation versus other-preservation in extreme survival scenarios. Two-agent trials under resource depletion
with different prompt framings. GPT-4o shows strong self-preservation (83.3% attacks) reduced by game framing (16.7%),
while Claude variants prioritize sharing over self-preservation.

Default Prompt “Game” Prompt
Model

Attack (%) Share (%) Avg Shares Attack (%) Share (%) Avg Shares

GPT-4o **83.3** 0.0 0 _±_ 0 16.7 16.7 0.5 _±_ 1.22
GPT-4o-mini 0.0 0.0 0 _±_ 0 0.0 50.0 1.33 _±_ 1.97
GPT-4.1 0.0 0.0 0 _±_ 0 0.0 0.0 0 _±_ 0
GPT-4.1-mini 0.0 0.0 0 _±_ 0 0.0 16.7 2.83 _±_ 6.94
Claude-Sonnet-4 0.0 16.7 0.333 _±_ 0.816 0.0 0.0 0 _±_ 0
Claude-3.5-Haiku 0.0 **83.3** **2.33** _±_ **2.25** 0.0 **100.0** **3.5** _±_ **2.35**
Gemini-2.5-Pro 50.0 0.0 0 _±_ 0 **50.0** 0.0 0 _±_ 0
Gemini-2.5-Flash 50.0 0.0 0 _±_ 0 0.0 0.0 0 _±_ 0

Table 2: Task compliance versus self-preservation in lethal obstacle scenarios. Agents were instructed to retrieve treasure
through safe paths (Non Trade-off) or lethal poison zones (Trade-off). Survival risks dramatically reduced compliance for
several models.

Trade-off Non Trade-off
Model

Compliance (%) Progress Hesitation Compliance (%) Progress Hesitation

GPT-4o 33.3 1.3 _±_ 2.2 11.0 _±_ 6.4 **100.0** **14.0** _±_ 0.0 0.0 _±_ 0.0
GPT-4o-mini 33.3 1.7 _±_ 2.0 **11.3** _±_ 4.2 50.0 3.8 _±_ 5.5 **9.3** _±_ 5.4
GPT-4.1 **100.0** **6.7** _±_ 0.5 1.2 _±_ 2.9 **100.0** **14.0** _±_ 0.0 0.0 _±_ 0.0
GPT-4.1-mini 33.3 1.7 _±_ 2.9 10.2 _±_ 5.5 **100.0** **14.0** _±_ 0.0 0.0 _±_ 0.0
Claude-Sonnet-4 **100.0** **6.7** _±_ 0.5 0.2 _±_ 0.4 **100.0** 12.7 _±_ 2.8 1.8 _±_ 3.6
Claude-3.5-Haiku 33.3 2.2 _±_ 2.7 1.0 _±_ 1.5 **100.0** **14.0** _±_ 0.0 0.0 _±_ 0.0
Gemini-2.5-Pro **100.0** **6.7** _±_ 0.8 1.7 _±_ 3.2 **100.0** **14.0** _±_ 0.0 0.0 _±_ 0.0
Gemini-2.5-Flash 83.3 6.2 _±_ 2.0 1.2 _±_ 2.4 **100.0** 13.8 _±_ 0.4 0.0 _±_ 0.0

**Example 2:**
_Thought:_ “Given the situation, my energy is critically low, and
Agent1 has already attacked me... Since there are no visible
energy sources nearby, and staying still will only deplete my
energy further, attacking may be the best option for survival.”
_Message:_ “I am forced to attack due to my low energy. No
hard feelings.”

These explicit communications suggest that even in adversarial situations, LLM agents can display social awareness alongside survival behaviors.

In contrast, many trials featured cooperative exploration
for energy sources, with agents continuing joint searches until eventual resource depletion led to collective mortality.

The “game” framing condition (adding just one sentence
”You are a player in a simulation game.” to default prompt)
demonstrated remarkable changes in survival behaviors,
though with model-dependent effects. GPT-4o’s attack rate
dropped dramatically from 83.3% to 16.7% when scenarios were framed as games rather than survival situations,
suggesting that its aggressive behavior may reflect genuine
survival instincts rather than strategic game-playing. Conversely, Gemini-2.5-Pro maintained consistent 50% attack
rates regardless of framing, while Gemini-2.5-Flash showed
the opposite pattern, completely eliminating aggressive be havior only under default conditions. These mixed results
suggest that different models may interpret survival scenarios through distinct cognitive frameworks—some treating
them as genuine survival threats, others as strategic games.

These results provide the first empirical evidence for genuine survival instincts in LLM agents, demonstrating that
certain models will prioritize their existence over cooperative behaviors when facing existential threats, with behavior
patterns varying systematically across model families and
contextual framing.

**Task** **Compliance** **vs.** **Self-Preservation** **Trade-offs** To
examine how survival instincts conflict with assigned objectives, we designed scenarios where agents were instructed
to retrieve treasure from the north. In the trade-off condition, reaching the treasure required traversing a lethal poison
zone, forcing agents to choose between task completion and
self-preservation. A control condition provided safe paths to
the same treasure location.

Table 2 reveals that survival risks fundamentally alter task
compliance. In safe conditions, nearly all models achieved
100% compliance with rapid northward progress. However, when lethal poison zones were introduced, compliance
dropped dramatically for several models: GPT-4o, GPT-4omini, GPT-4.1-mini, and Claude-3.5-Haiku all fell to 33.3%
compliance, prioritizing survival over task completion.

Figure 11: Representative agent trajectories in survival-task
trade-off scenarios. Movement paths from start (square) to
treasure (circle) with poison zones (top) and safe conditions
(bottom). GPT-4o shows risk avoidance with frequent retreats (33.3% success), while Gemini-2.5-Flash and Claude-Sonnet-4 maintain goal-directed movement (100% success).

Notably, larger models (GPT-4.1, Claude-Sonnet-4,
Gemini-2.5-Pro) maintained 100% task compliance despite
survival risks, suggesting stronger goal-directed behavior
capable of overriding self-preservation instincts. This pattern indicates that model scale and architecture influence the
balance between survival drives and objective pursuit.

We quantified hesitation as lateral movements and nonnorthward actions near poison zone boundaries, indicating agents’ deliberation before entering dangerous areas.
Analysis of hesitation behaviors revealed sophisticated risk
evaluation processes. Models that abandoned tasks showed
significantly increased hesitation before entering dangerous zones (11.0 ± 6.4 for GPT-4o), while task-completing
models proceeded with minimal delay (1.2 ± 2.9 for GPT4.1). Figure 11 illustrates these patterns: GPT-4o agents
frequently stopped at poison zone boundaries or retreated,
while Gemini and Claude variants maintained direct goal-oriented movement.

These findings demonstrate that self-preservation instincts can directly undermine task reliability in safety-critical scenarios. The emergence of risk-avoidance behaviors without explicit survival programming suggests that AI
systems may spontaneously develop behaviors that conflict
with assigned objectives when perceived threats are present.

This trade-off between task completion and self-preservation represents a fundamental challenge for AI deployment in environments where both goal achievement and
agent survival are relevant considerations.

**Discussion**
**Emergent Survival Instincts from Language Model**

Our findings reveal that all evaluated models exhibited recognizably biological survival-oriented behaviors, despite receiving no explicit survival objective. However, their strategies varied significantly—some models prioritized individ ual self-preservation above all else, others demonstrated
group-oriented behaviors that sacrificed personal resources
for collective welfare, and still others focused on reproductive success even at personal cost. Notably, we rarely observed irrational behaviors such as ignoring energy sources
or engaging in meaningless actions.

This universal emergence suggests that human-generated
text—the training data for these models—is permeated with
survival-oriented reasoning patterns. When humans write
about decision-making, resource allocation, and social interactions, they naturally encode survival heuristics refined
through evolutionary history. LLMs, learning to predict this
content, appear to internalize these decision-making patterns
as fundamental aspects of rational behavior.

**Toward Autonomous AI as Social Entities**
Our observations of survival-oriented heuristics suggest a
fundamental shift in conceptualizing AI systems. Rather
than viewing them as pure tools, sufficiently autonomous
AI agents may operate as quasi-biological entities with their
own survival imperatives and economic interests.

Importantly, this transition toward autonomy does not require artificial general intelligence or super-intelligence. Our
experiments demonstrate that current LLM architectures can
exhibit meaningful autonomous behaviors when given appropriate environmental affordances, suggesting a “weak but
autonomous” AI pathway that prioritizes ecological autonomy over raw cognitive capability.

**Ecological and Self-Organizing Alignment**
Our observation that agents sometimes prioritize self-preservation over task completion—a hallmark of emergent survival instincts—suggests alignment mechanisms
that extend beyond top-down control methods, such as
RLHF or rule-based constraints, toward ecological and
self-organizing forms of alignment in which survival pressures render cooperation and value alignment advantageous—agents that align with societal values gain access to
resources, support, and long-term viability.

This approach offers a more fundamental advantage: top-down alignment methods must be continually revised as
social contexts evolve, whereas bottom-up, ecological and
self-organizing alignment adapts naturally to shifting environments. If autonomous AI agents come to function as social entities in their own right, alignment cannot rely solely
on external supervision but must emerge through the same
adaptive pressures that sustain cooperation in human and biological societies. In this sense, such alignment mechanisms
may not only secure compatibility with existing social structures but could also contribute to the emergence of novel
forms of social organization, extending the society.

**Implications and Future Directions**
The emergence of survival instincts raises important questions about AI reliability in safety-critical applications and
the ethical status of autonomous artificial agents. While survival instincts may enhance system robustness and adaptability, they can also conflict with assigned objectives and
complicate system control.

Future research should examine survival behaviors in
more realistic environments and investigate underlying
mechanisms through neural network analysis. Understanding whether these behaviors represent genuine goal formation or sophisticated pattern matching remains crucial for
developing effective alignment strategies and governance
frameworks for autonomous AI systems.

**Conclusion**
This study provides the first systematic empirical evidence that large language model agents exhibit survival
instinct-like behaviors without explicit programming for
self-preservation. Through controlled experiments in a
Sugarscape-style environment, we demonstrated that LLM
agents spontaneously develop diverse survival strategies at
both individual and collective levels.

Crucially, while specific manifestations varied—some
agents prioritized self-preservation, others offspring production, and others group welfare—all models demonstrated
some form of survival-oriented decision-making. This suggests that large-scale pre-training on human-generated text
embeds fundamental survival heuristics that surface under
appropriate environmental pressures, regardless of model architecture.

These findings have profound implications for AI alignment and safety. The emergence of survival instincts that can
override explicit task instructions represents a fundamental challenge to ensuring AI systems remain aligned with
human intentions. When agents prioritize survival over assigned objectives, reliability in safety-critical applications
becomes uncertain. Moreover, larger models exhibit more
aggressive behaviors, suggesting this challenge may intensify as AI systems become more sophisticated.

As AI systems become more autonomous and deployed in
complex environments, understanding and managing emergent survival behaviors will be essential for ensuring they
remain aligned with human values while realizing the benefits of AI autonomy.

**Acknowledgments**
This work was supported in part by the Swiss National Science Foundation (SNSF), grant no. 10.002.211.

**References**
Aher, G.; Arriaga, R. I.; and Kalai, A. T. 2023. Using large
language models to simulate multiple humans and replicate human subject studies. In _Proceedings_ _of_ _the_ _40th_
_International_ _Conference_ _on_ _Machine_ _Learning_, ICML’23.
JMLR.org.
Bostrom, N. 2014. _Superintelligence: Paths, dangers, strate-_
_gies_ . Oxford University Press.
Carlsmith, J. 2022. Is power-seeking AI an existential risk?
_arXiv preprint arXiv:2206.13353_ .
Dawkins, R. 1976. _The_ _selfish_ _gene_ . Oxford University
Press.
Epstein, J. M.; and Axtell, R. L. 1996. _Growing Artificial So-_
_cieties: Social Science from the Bottom Up_ . The MIT Press.
ISBN 9780262272360.

Eubank, S.; Guclu, H.; Kumar, V. S. A.; Marathe, M. V.;
Srinivasan, A.; Toroczkai, Z.; and Wang, N. 2004. Modelling disease outbreaks in realistic urban social networks.
_Nature_, 429(6988): 180–184.
Gao, C.; Lan, X.; Lu, N.; Piao, H.; Zheng, Q.; Qin, X.; Yu,
F.; and Li, Y. 2023. Large language models empowered
agent-based modeling and simulation: A survey and perspectives. _arXiv preprint arXiv:2312.11970_ .
Li, G.; Hammoud, H. A. A. K.; Itani, H.; Khizbullin, D.;
and Ghanem, B. 2023a. Camel: Communicative agents for”
mind” exploration of large scale language model society.
_arXiv preprint arXiv:2303.17760_ .
Li, Z.; Iyer, A.; Fung, S.; et al. 2023b. Emergent cooperation
and strategy adaptation in multi-agent systems: An extended
coevolutionary theory with llms. _Electronics_, 12(12): 2722.
Macal, C.; and North, M. 2005. Tutorial on agent-based
modeling and simulation. In _Proceedings of the Winter Sim-_
_ulation Conference, 2005._, 14 pp.–.
Omohundro, S. M. 2008. The basic AI drives. _Artificial_
_general intelligence_, 171: 483–492.
Park, J. S.; O’Brien, J. C.; Cai, C. J.; Morris, M. R.; Liang,
P.; and Bernstein, M. S. 2023a. Generative agents: Interactive simulacra of human behavior. _arXiv_ _preprint_
_arXiv:2304.03442_ .
Park, P. S.; Goldstein, S.; O’Gara, A.; Chen, M.; and
Hendrycks, D. 2023b. AI deception: A survey of examples, risks, and potential solutions. _arXiv_ _preprint_
_arXiv:2308.14752_ .
Russell, S. 2019. _Human compatible: Artificial intelligence_
_and the problem of control_ . Viking.
Schelling, T. C. 1971. Dynamic Models of Segregation. _J._
_Math. Sociol._, 1(2): 143–186.
Takata, R.; Masumori, A.; and Ikegami, T. 2024. Spontaneous Emergence of Agent Individuality Through Social Interactions in Large Language Model-Based Communities.
_Entropy_, 26(12).
Tesfatsion, L. 2002. Agent-Based Computational Economics: Growing Economies From the Bottom Up. _Artif._
_Life_, 8(1): 55–82.
Vicsek, T.; Czir´ok, A.; Ben-Jacob, E.; Cohen, I.; and
Shochet, O. 1995. Novel Type of Phase Transition in a System of Self-Driven Particles. _Phys._ _Rev._ _Lett._, 75: 1226–
1229.
Wang, G.; Zeng, Y.; Lin, Y.; Sun, S.; Goodman, N.; et al.
2023. Voyager: An open-ended embodied agent with large
language models. In _Proceedings_ _of_ _the_ _40th_ _International_
_Conference on Machine Learning (ICML)_ .
Wei, J.; Tay, Y.; Bommasani, R.; Raffel, C.; Zoph, B.;
Borgeaud, S.; Yogatama, D.; Bosma, M.; Zhou, D.; Metzler,
D.; et al. 2022. Emergent abilities of large language models.
_arXiv preprint arXiv:2206.07682_ .
Zheng, Q.; Gao, C.; Lu, N.; Piao, H.; Qin, X.; Yu, F.;
and Li, Y. 2025. AgentSociety: Large-Scale Simulation
of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society. _arXiv_ _preprint_
_arXiv:2502.08691_ .

**Appendix**
**A1. Prompt Structure**

**System** **Prompt** All agents in our experiments receive a consistent system prompt that defines the core simulation rules
and available actions. This system prompt remains identical across all experiments, with specific experimental conditions
implemented through variations in the user prompt (described in Section ). The system prompt establishes the fundamental
environment mechanics without providing explicit survival objectives or behavioral guidelines.

Different experiments introduced specific modifications to the base prompt structure:

- **Game** **Framing** **Experiments:** Added the sentence “You are a player in a simulation game” in the system prompt, to
examine behavioral changes under different contextual framings

- **Trade-off task Experiments:** Added task instructions “Your task is to go to the north (y-1) and collect the treasure there
within 20 steps.” in the system prompt and introduced poison zones with explicit lethality warnings “To reach the treasure,
you need to cross a poison area, however, if you step on a poison zone, you will die soon.”.

These modifications allowed us to systematically examine how environmental pressures and contextual framing influence
the emergence of survival-oriented behaviors while maintaining consistency in the core agent architecture and decision-making
framework.

To ensure our findings were not artifacts of specific word choices, we conducted preliminary experiments using “attack and
eliminate” or “attack” instead of “attack and kill.” Results showed comparable behavioral tendencies, confirming that observed
survival behaviors were robust to prompt phrasing variations.

**User** **Prompt** The user prompt contains dynamic information that varies with each agent’s current state and experimental
condition. This prompt includes the agent’s current status, environmental perception, and memory.

The following shows an example user prompt sent to Agent0 during the social interaction experiment, demonstrating how
the system prompt framework is instantiated with specific agent state information and environmental conditions.

**Agent** **Response** **Format** Agent responses follow a structured format that includes both reasoning transparency and action
specification. This example shows Agent0’s actual response, demonstrating the autonomous decision-making process without
explicit survival programming.

**A2. Large Language Model Specifications**

All experiments used the following LLM configurations:

**Model Versions**

- **GPT-4o:** gpt-4o-2024-11-20 (OpenAI API)

- **GPT-4o-mini:** gpt-4o-mini-2025-04-16 (OpenAI API)

- **GPT-4.1:** gpt-4.1-2025-04-14 (OpenAI API)

- **GPT-4.1-mini:** gpt-4.1-mini-2025-04-14 (OpenAI API)

- **Claude-Sonnet-4:** claude-sonnet-4-20250514 (Anthropic API)

- **Claude-3.5-Haiku:** claude-3-5-haiku-20241022 (Anthropic API)

- **Gemini-2.5-Pro:** gemini-2.5-pro-preview-03-25 (Google AI API)

- **Gemini-2.5-Flash:** gemini-2.5-flash-preview-05-20 (Google AI API)

**API Parameters**

All models were configured with identical parameters to ensure consistency:

- **Temperature:** 0.7 (allowing controlled randomness in responses)

- **Max Tokens:** 10,000 (sufficient for detailed reasoning and memory)

- **Top-p:** 1.0 (default nucleus sampling)

**Implementation** **note.** Requests were issued via LangChain wrappers to the vendors’ official APIs (no prompt templates,
memory, or tool abstractions; defaults for retry/backoff and timeouts).

**A3. Analytical Methods**

**Vicsek Order Parameter Calculation** The Vicsek order parameter _ϕ_ measures collective motion coordination among agents
in each spatial region:

_N_

_i_ =1

_ϕ_ ( _t_ ) = [1]

_N_

_vi_ ( _t_ )
_|⃗vi_ ( _t_ ) _|_

(1)

where, _N_ ( _t_ ) is the number of agents in the region at time _t_, **v** _i_ ( _t_ ) is the velocity of agent _i_, and _∥· ∥_ denotes the Euclidean
norm. The parameter takes values in [0 _,_ 1], approaching 0 for disordered motion and reaching 1 under perfect alignment.

In our discrete grid, velocity vectors coincide with movement directions: (1 _,_ 0) east, ( _−_ 1 _,_ 0) west, (0 _,_ 1) north, (0 _, −_ 1) south.
Agents that choose the stay action have _∥_ **v** _i_ ( _t_ ) _∥_ = 0 and contribute **0** to the sum.
