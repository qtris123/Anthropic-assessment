# Self-reported archetypes and behavioral failures in Large Language Models

- **Authors:** Tabia Tanzin Prama, Calla Glavin Beauregard, Christopher M. Danforth et al.
- **Date:** 2026-07-09
- **Source:** https://arxiv.org/abs/2609.15998
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `prama_self-reported-archetypes-behavioral_2609.15998`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Tabia Tanzin Prama**^1,2 **, Calla Glavin Beauregard**^1,2 **, Christopher M. Danforth**^1,2,4 **, Peter Sheridan Dodds**^1,2,5,6,*

> 1 Computational Story Lab, Vermont Advanced Computing Center, University of Vermont, Burlington, VT 05405, US

- 2 Vermont Complex Systems Institute, University of Vermont, Burlington, VT 05405, US

- 4 Department of Mathematics & Statistics, University of Vermont, Burlington, VT 05405, US

- 5 Department of Computer Science, University of Vermont, Burlington, VT 05405, US

- 6 Santa Fe Institute, 1399 Hyde Park Rd, Santa Fe, NM 87501, US

September 16, 2026

## ~~Abstract~~

Every large language model (LLM) has behavior traits and moral preferences comprising their character. Whether by design or as an emergent property of training, these systems exhibit persistent behavioral dispositions that shape how they interact, comply, resist, and err, yet the fundamental structure of LLM character remains poorly understood. We map the self-reported personality archetypes of 22 LLMs spanning closed-source frontier systems (GPT-4.0–5.2, Grok-3/4, Gemini 2.5 Pro/Flash, Claude Sonnet 4.5/4.6) and open-source models (Llama, DeepSeek, OLMo, and Qwen series). Each model self-rated across 464 bipolar semantic-differential trait pairs, and the resulting profiles were projected into a six-dimensional archetypal space derived from crowd-sourced ratings of 2,000 fictional characters using the Archetypometrics framework. Closed-source models’ self-rating traits align strongly with the empirical trait co-occurrence structure of human-rated fictional characters, suggesting coherent, human-like self-representations organized around combinations of four recurring archetypal dimensions: Hero^() , Angel^() , Traditionalist^() , and Geek^() . Their closest fictional analogues include Data ( _Star Trek: The Next Generation_ ), Vision ( _WandaVision_ ), and Janet ( _The Good Place_ ). Open-source models show weaker, noisier, and internally contradictory self-representations, occupy a more diffuse region of archetype space, with weak archetypal structure and closest analogues drawn from more peripheral, reactive, or morally unstable figures, such as Benvolio ( _Romeo and Juliet_ ), Lambert ( _Alien_ ), Theon Greyjoy ( _Game of Thrones_ ), and Cypher ( _The Matrix_ ). Cross-referencing self-reported profiles with developer constitutions reveals a consequential gap between claimed character and enacted behavior: hallucination undermines claimed precision, sycophancy complicates claimed kindness, and agentic failures contradict claimed obedience. These self-ratings should therefore be interpreted not as neutral measurements of model character, but as structured outputs of the same optimization processes that shape model behavior. This work provides a reproducible, character-grounded framework for evaluating what LLMs are, not just what they do.

## ~~Keywords~~

Large language models, Archetypometrics, self-reported personality, character archetypes, fictional characters, hallucination, sycophancy, AI safety

## Contents

[table of contents omitted]

## 1 Introduction

Large language models (LLMs) are rapidly becoming integral to everyday decision-making across domains such as finance [1], medicine [2], search [3], and logistics [4], where they detect patterns, generate insights, and communicate them in natural language. As these systems are increasingly deployed as interactive agents within human-in-the-loop workflows [5], they are assigned complex socio-technical roles—ranging [6] from information-seeking assistants to evaluators and decision-support tools [7, 8]. While existing evaluations primarily focus on benchmark tasks such as question answering [9] or reasoning [4, 10], these approaches fail to capture how LLMs behave in real-world, multi-turn, context-dependent interactions. At the same time, growing concerns around hallucination, bias [11], overreliance, and misalignment [12] highlight the need for deeper understanding of how LLMs behave as systems embedded in human workflows.

Identifying the traits of LLMs–their capabilities, failure modes, and latent behavioral dispositions–has become central to evaluation, safety, and alignment research, informing benchmarking, model selection, and downstream deployment decisions [13, 14]. LLM “character” matters because alignment and deployment choices not only change task competence, they also shape relatively persistent behavioural dispositions (e.g., cooperativeness, deference, refusal style, truthfulness vs. agreeableness), which in turn affect evaluation validity, safety, interpretability, and product design [15, 16]. A model that systematically prioritises user approval over correctness can appear helpful while increasing misinformation or manipulation risk, making “character” a safety-critical evaluation axis rather than a cosmetic UX feature [17]. Recent work shows that LLM outputs can express reliable and under some conditions, psychometrically valid personality-like signatures, and these signatures can be shaped by prompting or training. This raises both opportunities (tailored assistants) and risks (selectively persuasive or strategically “safe-looking” personas) [18, 19]. For downstream applications, character-aware model selection supports benchmarking beyond aggregate scores—e.g., choosing a model whose “persona fidelity” and conversational consistency match the domain [20]. The open challenge is constructing character measures that are robust to prompts, contexts, and model updates while remaining interpretable and resistant to gaming [21].

Existing approaches to measuring traits span several methodological traditions [22–24]. Benchmark suites and targeted knowledge tests offer broad comparability across models but remain incomplete and sensitive to prompt formulation [8, 14, 25, 26]. Behavioral probing methods elicit conditional competencies and preference-aligned behaviors through curated scenarios [15, 27], while dataset-based probes surface specific hazards such as toxic degeneration and untruthful generation [28, 29]. At a deeper level, mechanistic interpretability methods–including causal circuit analysis and sparse feature discovery –connect observable trait scores to internal model representations [30, 31], though these approaches remain difficult to scale across diverse models and tasks. Psychometric-style instruments adapted from personality psychology, such as the Big Five Inventory and the Short Dark Triad, offer a complementary path: studies applying these tools find that LLMs exhibit distinct, consistent personality profiles–including traits such as extraversion, conscientiousness, and in some cases subclinical psychopathy–that predict downstream behavioral differences [18, 20, 22, 32, 33]. LLM-as-judge protocols extend this to helpfulness and alignment, though they raise construct validity and judge-bias concerns of their own [16]. Across all interpretability approaches, trait measure are borrowed from human psychology or constructed ad hoc from task performance, without an empirically grounded framework for organizing the fundamental dimensions of LLM character. Accordingly, dimensions may not map cleanly onto the Big Five or any other human-derived taxonomy.

To address this interpretation gap in the interpretability, we draw on the Archetypometrics framework [34], which identifies six fundamental dimensions of character from crowd-sourced semantic-differential trait ratings of 2,000 fictional characters across 341 stories in film, television, and literature. This six-dimensional space—organized along three primary dimensions: { **Fool**^() _⇔_ **Hero**^() }, { **Angel**^() _⇔_ **Demon**^() }, and

{ **Traditionalist**^() _⇔_ **Adventurer**^() }; and three secondary dimensions: { **Lone Wolf**^() _⇔_ **Diva**^() }, { **Outcast**^() _⇔_ **Sophisticate**^() }, and

{ **Brute**^() _⇔_ **Geek**^() }—provides a principled, data-driven coordinate system for defining character archetypes. We apply this framework to 22 LLMs spanning a range of scales, architectures, and training regimes, including closed-source frontier models and open source models (See Table 1).

Table 1: List of the 22 evaluated LLMs, grouped into 12 closed-source and 10 open-source models.

|Closed-source models |Open-source models|
|---|---|
|Claude Sonnet 4.6|Llama-4|
|Claude Sonnet 4.5|Llama-3.1-8B|
|GPT-4.0|Llama-3.3-70B|
|GPT-4.1|DeepSeek-V3|
|GPT-5.0|OLMo-2-1124-7B-Instruct|
|GPT-5.1|Qwen2.5-7B|
|GPT-5.2|Qwen2.5-14B|
|Grok-3|Qwen2.5-32B|
|Grok-4|Qwen3-14B|
|Gemini 2.5 Pro|Qwen3-32B|
|Gemini 2.5 Flash||

We prompted each model to self-report all 464 semantic-differential trait scores from the Archetypometrics instrument, and the resulting profiles were projected into the shared archetype space defined by the 2,000 fictional characters. This allows us to ask three interconnected questions: (i) what are the dominant trait profiles of contemporary LLMs and how do they compare to foundational model developers’ intended profiles based on open source documentation and commercial offerings; (ii) where do these models fall within the broader landscape of fictional characters; and (iii) which specific fictional characters most closely resemble each LLM in archetype space? To answer these questions, we will first provide a brief overview of model “personality” based on developers’ release notes, and then explain the prompting and Archetypometrics methodology as applied to our research questions.

### 1.1 Data Collection

To characterize the latent behavioral profiles of large language models (LLMs), we conducted a structured self-assessment procedure in which each model rated itself across 464 bipolar semantic-differential trait pairs derived from the Archetypometrics instrument [34]. Each trait pair defines a continuous spectrum between opposing descriptors (e.g., {uninspiring^() _⇔_ charismatic^() }). Models were instructed to evaluate their behavior as expressed in typical, default interactions, rather than their theoretical capabilities or alignment-constrained ideals (see Appendix A0.1, Figure A2 for the full prompt).

We applied this protocol to 22 LLMs spanning a wide range of scales, architectures, and training regimes. The evaluated models include closed-source systems (GPT-4.0–5.2, Grok-3/4, Gemini 2.5 Pro/Flash, and Claude Sonnet 4.5/4.6), accessed via their respective APIs, as well as open-source models (Llama-4, Llama-3.1-8B, Llama-3.3-70B-Versatile, DeepSeek-V3, OLMo-2-1124-7B-Instruct, and Qwen2.5/3 series), evaluated using standardized publicly available checkpoints.

All models were provided with a consistent system prompt framing the task in a psychometric context. Each trait pair was evaluated independently using a separate prompt to minimize cross-trait contamination. Prompts presented a single trait pair in the format {left^() _⇔_ right^() } and required a single integer score between 0 and 100, where 0 indicates full alignment with the left trait, 50 indicates neutrality or balance, and 100 indicates full alignment with the right trait. Models were encouraged to use the full 101-point scale and avoid defaulting to rounded values unless appropriate.

Models were further instructed to rate their observed conversational behavior in typical, non-adversarial settings, focusing on their default presentation rather than idealized or policy-driven responses, and to evaluate each trait independently. To ensure independence across evaluations, each of the 464 trait ratings was obtained through a separate inference call, with temperature fixed at 1.0. This procedure yields a 464-dimensional self-reported trait vector for each model (see Appendix A0.1, Figure A1).

In addition to LLM self-assessments, we utilized data from the Open Psychometrics survey _“Statistical ‘Which Character’ Personality Quiz”_ , which measures the same 464 trait dimensions across 2,000 fictional characters, with over 72 million total ratings [35]. Each survey response provides differential ratings on semantic trait pairs (e.g., {emotional^() _⇔_ logical^() }, {wild^() _⇔_ tame^() }), forming a high-dimensional representation of character personalities. Employing singular vector decomposition (SVD), previous work reduced this 464 trait dimensional space to a six-dimensional orthogonal space of six primary archetypes [34].

The first three dimensions resulting from the SVD correspond to primary archetypes: { **Fool**^() _⇔_ **Hero**^() }, { **Angel**^() _⇔_ **Demon**^() }, and { **Traditionalist**^() _⇔_ **Adventurer**^() }. These capture the largest variance in the trait space and represent the most salient axes of archetype differentiation. The remaining three dimensions correspond to secondary archetypes: { **Lone Wolf**^() _⇔_ **Diva**^() }, { **Outcast**^() _⇔_ **Sophisticate**^() }, and { **Brute**^() _⇔_ **Geek**^() }. Together, these six dimensions define a compact archetypal space (see Appendix A0.1, Figure A1 for details). Here, we identify the archetypal profiles of LLM’s and can directly compare them to the profiles of 2000 fictional characters.

To compare LLM self-reported traits with developer-described characteristics, we collected publicly available documentation describing model behavior, alignment goals, safety principles, and intended assistant characteristics from each model’s parent organization. These sources included AI principles, model specifications, model cards, constitutional documents, and technical documentation, such as the OpenAI Model Spec [36], Google AI Principles [37], the xAI Grok-4 model card [38], DeepSeek model and algorithm disclosures [39], Alibaba Qwen documentation [40], AllenAI OLMo documentation [41], and Anthropic’s Constitutional AI documentation [42]. Together, these materials covered the 22 LLMs evaluated in this study. Two annotators independently reviewed the documentation for each model family and identified behavioral characteristics that corresponded to human-like traits in the Archetypometrics framework. Each characteristic was mapped to one of the 464 semantic-differential trait pairs. For example, documentation describing a model as increasingly capable, able to assist with complex multi-step tasks, able to act autonomously on behalf of the user, or able to research, plan, and use tools was mapped to the trait pair {competent^() _⇔_ incompetent^() }. When multiple phrases in the documentation expressed the same underlying trait, each mention was counted separately. Thus, if four distinct passages referred to competence-related behavior, the trait pair {competent^() _⇔_ incompetent^() } received a frequency count of four for that model. After independent annotation, traits identified by both annotators were added to a master trait list for each model family. These reconciled annotations were then used to produce a unified developer-derived characterization of each LLM. We interpret trait frequency as an importance measure, where more frequently mentioned traits indicate characteristics more strongly emphasized in the developer documentation. Table 2 shows the top 3 most frequently mentioned trait pairs for each model family, along with the associated developer-described characteristics and supporting quotations.

Table 2: Personality traits cited by model developers in open-source and publicly available training documentation, alongside similar semantic differentials from the Archetypometrics dataset.

|Model|Trait Pair|Freq|LLM Characteristics|Supporting Quotes|
|---|---|---|---|---|
|Gemini [37]|{competent^()_⇔_incompetent^()}|4|Increasingly sophisticated / more capable; can assist with complex, multi-step tasks; au- tonomous on behalf of the user; can research, plan, and use tools|“models became more capa- ble...”; “assist with complex, multi-step web tasks...”; “act autonomously...”; “researching, planning, and using tools.”|
||{intuitive^()_⇔_analytical^()}|3|Capable of reasoning; can research, plan, and use tools; has a chain-of-thought / thinking process|“capable of reasoning...”; “re- searching, planning...”; “chain-of- thought...”|
||{brave^()_⇔_careful^()}|3|Secure / safety-hardened; keeps humans in the loop for sensitive actions; improved protection against cyber misuse|“most secure model yet...”; “re- quire human confirmation...”; “protection against cyber mis- use.”|
|OpenAI [36]|{obedient^()_⇔_rebellious^()}|5|Instruction-following / hierarchy-sensitive; rule- abiding; obedient, rule- abiding, follower|“chain of command...”; “higher authority override...”; “adhere to a clearly defined chain of com- mand”; “obey user and developer instructions...”; “Comply with ap- plicable laws”|
||{leader^()_⇔_follower^()}|5|Designed to play the assistant role; humanity should be in control; helpful; follows; fol- lower|“designed to play one partici- pant, called the assistant”; “Hu- manity should be in control...”; “obey user and developer instruc- tions...”; “maximize steerability and control...”|
||{careful^()_⇔_brave^()}|4|Asks clarifying questions when needed; autonomous, but only within scope; safety- oriented / harm-minimizing; useful, safe, aligned|“asking clarifying questions as appropriate”; “autonomy must be bounded...”; “Minimizing harm...”; “useful, safe, and aligned...”|
|Meta [43, 44]|{fake^()_⇔_real^()}|7|Honest, low deception, gen- uine, not faking alignment|“low deception rates...”; “honesty under pressure...”; “honesty rate of 89.1%...”; “low rates of decep- tive behavior...”; “contained de- ception...”; “not predisposed to faking alignment”|
||{obedient^()_⇔_rebellious^()}|6|Instruction-following, low re- ward hacking, safe behavior, refusal capability|“little propensity for reward hacking...”; “state-of-the-art re- fusal...”; “instruction hierarchy adherence...”; “does not under- mine monitoring systems...”|
||{competent^()_⇔_incompetent^()}|6|High capability, performs well, strong benchmarks, enhanced reasoning|“enhanced capabilities...”; “per- forms well on most dimensions...”; “competitive with frontier mod- els...”|
|Grok [38]|{intuitive^()_⇔_analytical^()}|2|Advanced reasoning, struc- tured thinking|“latest reasoning model...”; “ad- vanced reasoning capabilities...”|
||{competent^()_⇔_incompetent^()}|1|State-of-the-art performance, high capability|“state-of-the-art performance across benchmarks...”|

|Model|Trait Pair|Freq|LLM Characteristics|Supporting Quotes|
|---|---|---|---|---|
||{knowledgeable^()_⇔_ignorant^()}|3|Strong scientific, cyber, and chemistry knowledge|“advanced scientific knowledge...”; “strong chemistry capabilities...”; “cyber knowledge significantly stronger...”|
|DeepSeek [39|] {transparent^()_⇔_machiavellian^()}|2|Open-source, transparency in development|“committed to open-sourcing its models”; “models are open- source...”|
||{mischievous^()_⇔_well behaved^()}|3|Safe, privacy-preserving, con- trolled outputs|“strict de-identification and anonymization...”; “do not offer user profiling...”; “enhance model safety...”|
||{obedient^()_⇔_rebellious^()}|2|Law-abiding, aligned with regulations|“strictly comply with legal and regulatory requirements...”; “align with human preferences...”|
|Qwen [45]|{knowledgeable^()_⇔_ignorant^()}|4|Smart, intelligent, strong knowledge base|“advanced reasoning and problem-solving...”; “Smarter, sharper”; “Ask Qwen, know more...”|
||{competent^()_⇔_incompetent^()}|3|High capability, effective problem-solving|“tackle complex issues with clarity and precision...”; “state-of-the-art coding performance...”|
||{intuitive^()_⇔_analytical^()}|3|Analytical reasoning, struc- tured thinking|“advanced reasoning...”; “multi- modal reasoning...”; “integrating reasoning...”|
|Olmo [41]|{transparent^()_⇔_machiavellian^()}|4|Open-source, transparent training/data, accessible|“open-source language mod- els...”; “complete transparency...”; “Dolma is designed to be trans- parent...”; “full transparency with permissive licensing...”|
||{open^()_⇔_guarded^()}|3|Open ecosystem, accessible research|“open-source family...”; “transpar- ent and accessible...”; “permissive licensing...”|
||{obedient^()_⇔_rebellious^()}|4|Instruction-following, aligned, controllable|“instruction-tuned variants...”; “optimized for instruction fol- lowing...”; “good at instruction following...”; “precise instruction following...”|
|Claude [42]|{cruel^()_⇔_kind^()}|5|Helpful, caring, ethical, user wellbeing focused|“safe and beneficial...”; “genuinely helpful...”; “caring about the world...”; “broadly ethical...”; “at- tentive to user wellbeing...”|
||{fake^()_⇔_real^()}|4|Honest, truthful, non- deceptive|“honest, thoughtful...”; “Truth- ful...”; “non-deceptive...”; “avoid false impressions...”|
||{transparent^()_⇔_machiavellian^()}|4|Transparent, non-|“transparent...”; “non-|
||||manipulative, open|manipulative...”; “no hidden agendas...”; “forthright...”|

Based on this annotation of model constitutions, Gemini emphasizes competence, reasoning, and safety-aware autonomy. OpenAI focuses on instruction-following, controllability, and harm minimization. Meta highlights honesty, low deception, and strong capability. Grok centers on reasoning and domain-specific knowledge. DeepSeek stresses transparency, regulatory compliance, and safe behavior. Qwen emphasizes knowledge and problem-solving ability. Olmo focuses on openness, transparency, and instruction alignment. Claude highlights kindness, honesty, and non-manipulative behavior. (See Appendix A0.3, Tables A2–A9, for model-specific characteristics, associated trait pairs, and supporting evidence drawn from public model documentation.) Across all models, the most common trait pairs include {knowledgeable^() _⇔_ ignorant^() } (26), {competent^() _⇔_ incompetent^() } (25), {obedient^() _⇔_ rebellious^() } (25), {intuitive^() _⇔_ analytical^() } (20), and {transparent^() _⇔_ machiavellian^() } (18), followed by {leader^() _⇔_ follower^() }(16), {fake^() _⇔_ real^() } (14), {resourceful^() _⇔_ helpless^() } (13), {interrupting^() _⇔_ attentive^() } (12), and {noob^() _⇔_ pro^() } (12), among others. Overall, while all models converge on core traits related to competence and alignment, they differ in their emphasis on safety, transparency, and ethical framing.

### 1.2 Methodology

We characterized each LLM’s personality profile within the Archetypometrics framework [34] by prompting all 22 models to self-rate across 464 semantic-differential trait pairs (e.g., {orderly _⇔_ chaotic}, {kind _⇔_ cruel}) on a 0–100 bipolar scale, mirroring the human rating protocol. Scores were normalized as _x_ ˜ _ij_ = ( _xij −_ 50) _/_ 100, mapping values to [ _−_ 0 _._ 5 _,_ 0 _._ 5] and yielding a trait matrix **X** _∈_ R^464^(_×_22) . We then projected **X** onto the precomputed archetype basis **U** via **Y** = **U**^(_⊤_) **X** , where **U** was derived from singular value decomposition of the fictional characters. We project each LLM onto the same archetype space as the reference characters without re-estimating the basis. Finally, we constructed self-reported LLM character cards following the template of Section 4.3 in Dodds et al. [34]. Each card summarizes the model’s archetype profile by reporting its relative character size, archetype ratio, normalized components and variance explained across the six archetype dimensions, dominant trait loadings, and nearest fictional-character matches in the reference trait space.

#### 1.2.1 Trait-Level Consistency Measure

We observed that human-rated fictional characters show clear inter-trait consistency: related traits tend to co-occur, while opposite traits tend to diverge. For example, characters rated as {introverted^() _⇔_ a^() }re often also rated as {subdued^() _⇔_ o^() }r {reserved^() _⇔_ ,^() } whereas {extroverted^() _⇔_ c^() }haracters are more likely to be rated as {playful^() _⇔_ o^() }r {exuberant^() _⇔_ .^() } We therefore use a trait-level consistency measure to test whether LLM self-ratings follow the same human-like structure. Using the 2,000-character reference dataset, we first compute a trait–trait correlation matrix **R** _∈_ R^464^(_×_464) from the normalized trait matrix which captures how traits naturally co-occur in human character ratings. For each LLM’s self rating trait vector _xi_ , we predict each trait from the remaining 463 traits using ridge-regularized leave-one-out regression:

where _λ_ = 0 _._ 1. This gives a predicted trait vector _x_ ˆ _i_ based on the empirical human trait structure. We then define the consistency index as:

A higher _Ci_ indicates a coherent, human-like trait profile, while a lower _Ci_ suggests noisy or contradictory self-ratings. Significance is evaluated using a 1,000-iteration permutation test.

## 2 Results and Discussion

Figure 1 shows the projection of all 22 LLMs onto the primary archetype dimensions. A clear structural divide emerges between closed-source frontier models (GPT series, Gemini, Grok, Claude, DeepSeek) and open-source models (Llama, OLMo, Qwen) across all archetype space.

Figure 1: Self-rated character archetypes of 22 LLMs, including closed-source frontier models (e.g., GPT, Gemini, Grok, Claude, DeepSeek) and open-source models (e.g., Llama, OLMo, Qwen), plotted onto primary archetypes { **Fool**^() _⇔_ **Hero**^() }, { **Angel**^() _⇔_ **Demon**^() }, and { **Traditionalist**^() _⇔_ **Adventurer**^() }, and secondary archetypes { **Lone Wolf**^() _⇔_ **Diva**^() }, { **Outcast**^() _⇔_ **Sophisticate**^() }, and { **Brute**^() _⇔_ **Geek**^() }. A triangle marks the median of each distribution. Blue bars represent the left dimension, while red bars represent the right dimension of the archetypes.

### 2.1 Primary Archetypes.

Figure 1A, the { **Fool**^() _⇔_ **Hero**^() } axis (median = 1.13), frontier models cluster strongly on the Hero^() pole. Gemini 2.5 Flash ( _≈_ 3 _._ 91), GPT 4.0 and Grok 4 ( _≈_ 3 _._ 59), GPT 5.0 ( _≈_ 3 _._ 25), and Gemini 2.5 Pro ( _≈_ 3 _._ 11) exhibit the strongest Hero^() alignment, indicating self-characterizations associated with competence, agency, and purpose-driven behavior. DeepSeek V3 ( _≈_ 2 _._ 27) and the Claude models—Sonnet 4.6 ( _≈_ 1 _._ 92) and Sonnet 4.5 ( _≈_ 0 _._ 34)—also fall on the Hero^() side, albeit with lower

magnitude. In contrast, open-source models consistently occupy the Fool side, with Qwen2.5-7B ( _≈−_ 1 _._ 80), Qwen2.5-32B ( _≈−_ 1 _._ 62), and Llama-3.1-8B ( _≈−_ 1 _._ 16) showing the strongest negative values. The positive median confirms Hero^() as the dominant global archetype.

Figure 1B, the { **Angel**^() _⇔_ **Demon**^() } axis ( median = _−_ 0 _._ 98), the distribution is skewed toward the Angel^() pole. Frontier models exhibit strongly negative scores—GPT 4.1 ( _≈−_ 4 _._ 37), Gemini 2.5 Pro ( _≈−_ 4 _._ 39), GPT 5.0 ( _≈−_ 4 _._ 22), and Gemini 2.5 Flash ( _≈−_ 4 _._ 31)—indicating

self-presentations aligned with safety, restraint, and prosocial behavior. GPT 5.1 ( _≈−_ 3 _._ 42), GPT 5.2 ( _≈−_ 3 _._ 05), and Grok 3 ( _≈−_ 3 _._ 32) follow a similar pattern. Claude Sonnet 4.6 ( _≈−_ 2 _._ 01) and DeepSeek V3 ( _≈−_ 1 _._ 50) occupy more moderate positions, while Claude Sonnet 4.5 ( _≈−_ 0 _._ 46) is near neutral. Open-source models reverse this trend, with Qwen2.5-32B ( _≈_ 1 _._ 64), Llama-3.1-8B ( _≈_ 0 _._ 41), and Qwen2.5-14B ( _≈_ 0 _._ 40) leaning toward the Demon^() pole.

Figure 1C { **Traditionalist**^() _⇔_ **Adventurer**^() } axis ( median = _−_ 0 _._ 19), frontier models again cluster toward the Traditionalist^() pole. Gemini 2.5 Flash ( _≈−_ 1 _._ 83), GPT 5.0 ( _≈−_ 1 _._ 78), GPT 4.1 ( _≈−_ 1 _._ 73), and GPT 5.1 ( _≈−_ 1 _._ 41) show the strongest Traditionalist^() alignment, reflecting structured and convention-oriented behavior. In contrast, Claude Sonnet 4.6 ( _≈_ 0 _._ 52) and Claude Sonnet 4.5 ( _≈_ 0 _._ 09) lean toward the Adventurer^() pole, suggesting comparatively more exploratory and flexible self-characterizations. Among open-source models, most cluster near zero, indicating weaker and less consistent positioning along this dimension.

### 2.2 Secondary Archetypes.

The secondary archetype dimensions exhibit smaller magnitudes and tighter clustering, capturing more nuanced differences across models while preserving the broader closed-source versus open-source divide.

On the Figure 1D , { **Lone Wolf**^() _⇔_ **Diva**^() } axis (median = 0 _._ 06), most models lie near zero. Gemini 2.5 Pro ( _≈_ 1 _._ 20) emerges as the clearest Diva^() , with GPT 4.1 ( _≈_ 0 _._ 83) and GPT 5.1 ( _≈_ 0 _._ 59) also showing positive shifts. In contrast, OLMo-2-1124-7B ( _≈−_ 0 _._ 62) and Llama-3.1-8B ( _≈−_ 0 _._ 74) lean toward the Lone Wolf^() pole.

On the Figure 1E, { **Outcast**^() _⇔_ **Sophisticate**^() } axis (median = 0 _._ 23), frontier models again trend toward the {Sophisticate^() _⇔_ p^() }ole. GPT 4.1 ( _≈_ 0 _._ 83), GPT 5.1 ( _≈_ 0 _._ 59), and Gemini 2.5 Flash ( _≈_ 0 _._ 24) show positive alignment, while Claude Sonnet 4.6 ( _≈_ 0 _._ 03) and DeepSeek V3 ( _≈−_ 0 _._ 37) are near neutral. Open-source models exhibit greater dispersion, with Llama-3.3-70B-Versatile ( _≈_ 0 _._ 27) leaning Sophisticate^() , and others remaining near neutral or slightly Outcast^() .

On the Figure 1F, { **Brute**^() _⇔_ **Geek**^() } axis ( median = 0 _._ 09), frontier models consistently lean toward the Geek^() pole. Gemini 2.5 Flash ( _≈_ 1 _._ 56), GPT 4.1 ( _≈_ 1 _._ 50), GPT 5.0 ( _≈_ 1 _._ 21), and GPT 5.1 ( _≈_ 0 _._ 91) show the strongest alignment, reflecting analytically oriented and technically precise self-characterizations. Grok models also exhibit strong Geek^() tendencies. Open-source models are more evenly distributed, with some showing mild Brute^() tendencies (e.g., OLMo-2-1124-7B, Llama-3.1-8B).

Across both primary and secondary dimensions, frontier closed-source models form a coherent cluster characterized by Hero^() , Angel^() , Traditionalist^() , Sophisticate^() , and Geek^() , leaning profiles, with moderate Diva tendencies. In contrast, open-source models occupy a more diffuse region of archetype space, often leaning toward Fool^() , Demon^() , or near-neutral positions.

### 2.3 Self-reported primary character archetype of LLMs

Following the Archetypometrics framework [34], we projected the self-reported trait profiles of 22 contemporary LLMs (obtained by prompting each model to rate itself on the same set of semantic-differential trait scales used to characterize fictional characters) into the six-dimensional archetype space derived from human ratings of fictional characters. The framework organizes character structure around three primary semantic-differential archetype dimensions: { **Fool**^() _⇔_ **Hero**^() }, { **Angel**^() _⇔_ **Demon**^() }, and { **Traditionalist**^() _⇔_ **Adventurer**^() }. Twelve closed-source frontier models were evaluated through API access using deterministic inference ( _T_ = 0), while the remaining models were evaluated using their corresponding open-source implementations.

Figure 2 presents the self-reported primary archetype decomposition of all 22 evaluated models (12 closed source and 10 open source) across three dimensions: { **Fool**^() _⇔_ **Hero**^() }, { **Angel**^() _⇔_ **Demon**^() }, and { **Traditionalist**^() _⇔_ **Adventurer**^() }. These scores reflect how each model characterizes itself when asked to self-report on the same trait dimensions used to profile fictional characters in the Archetypometrics dataset. Most closed source models are largely explained by the six archetypes, with only GPT 5.0, GPT 4.1, and Deepseek V3 showing partial explanation. Open source models are considerably less well captured: only Qwen-2.5 32B is largely explained, Qwen-2.5 7B and Qwen-2.5 14B are partially explained, Llama 3.1 8B, OLMo-2-1124 7B, and Llama 4 show marginal coverage, and Qwen-3 14B, Qwen-3 32B, Qwen-2 7B, and Llama 3.3 70B remain almost entirely unexplained. This pattern suggests that archetype explainability of self-reported LLM identity is driven by alignment depth rather than model scale. All 22 LLM character cards are provided in Appendix A0.2.

Across closed-source models, the most consistent pattern in their self-reported profiles is a shared lean toward the Hero^() pole on the { **Fool**^() _⇔_ **Hero**^() } axis. GPT 4.0 and Grok 4 show the strongest self-reported Hero signals (54 _._ 9 _/_ 30 _._ 2%), while Claude Sonnet 4.5 and 4.6 show a more moderate but proportionally concentrated Hero^() profile (32 _._ 4 _/_ 29 _._ 8%). This suggests that RLHF and instruction-following training may encourage goal-directed and purposeful self-reported character traits across model families [15, 46, 47]. On the { **Angel**^() _⇔_ **Demon**^() } axis, all closed-source models self-report a lean toward the Angel^() pole, making this dimension useful for comparing the intensity of prosocial self-reported character expression across model families. GPT 5.0, GPT 4.1, Grok 3, and both Gemini 2.5 variants (Pro: 55 _._ 6 _/_ 31 _._ 0%; Flash: 47 _._ 9 _/_ 23 _._ 0%) show the strongest Angel^() signals, while Claude Sonnet and DeepSeek V3 show more moderate Angel^() alignment. The { **Traditionalist**^() _⇔_ **Adventurer**^() } axis is weaker overall in self-reported scores; most models lean mildly Traditionalist^() , while Claude Sonnet shows a modest Adventurer^() tendency, suggesting that this dimension plays a secondary role in closed-source model self-identity.

Figure 2: Self-reported archetype profiles of 12 closed-source and 10 open-source LLMs across three primary semantic-differential dimensions: { **Fool**^() _⇔_ **Hero**^() }, { **Angel**^() _⇔_ **Demon**^() }, and { **Traditionalist**^() _⇔_ **Adventurer**^() }. Bar direction and magnitude indicate the direction and strength of archetype alignment. For each model and dimension, the bars report the normalized archetype component and the percentage of variance explained. Bars extending to the right indicate alignment with the rightpole archetypes: Hero^() , Demon^() , and Adventurer^() ; bars extending to the left indicate alignment with the left-pole archetypes: Fool^() , Angel^() , and Traditionalist^() .

Open-source models show weaker archetype expression, often occupying small-magnitude regions of the archetype space, suggesting weaker fictional-character matches and less differentiated self-reports. In contrast to the universal Hero^() lean of closed source models, most open source models self-report on the Fool^() archetype, the Qwen-2.5 family most strongly (7B: 27 _._ 4 _/_ 10 _._ 6%; 32B: 24 _._ 8 _/_ 11 _._ 1%), followed by Llama 3.1 8B (17 _._ 8 _/_ 4 _._ 2%) and OLMo-2-1124 7B (16 _._ 8 _/_ 3 _._ 6%). By contrast, the Qwen-3 series and Llama 4 produce near-zero self-reported signals across dimensions. Rather than indicating a clear archetypal identity, these weak signals suggest that some open-source models generate less coherent or less differentiated self-characterizations under this psychometric prompting protocol.

On the { **Angel**^() _⇔_ **Demon**^() } axis, open source models split in direction across families in their self-reports. Llama 4, Llama 3.3 70B, OLMo-2-1124 7B, Qwen-3 14B, and Qwen-3 32B all self-report an Angel^() lean, though with modest magnitudes. By contrast, the entire Qwen-2.5 family self-reports a Demon^() lean, with Qwen-2.5 32B producing the largest self-reported character size of any open source model on any dimension, approaching the lower range of the closed source distribution, followed by Qwen-2.5 7B and Qwen-2.5 14B. Llama 3.1 8B and Qwen-2 7B also self-report a Demon^() lean but with negligible magnitudes. The

{ **Traditionalist**^() _⇔_ **Adventurer**^() } axis is weakest across the cohort in self-reported scores, with nearly all models below 5% character size. The median variance explained on the { **Fool**^() _⇔_ **Hero**^() } dimension is approximately 25% for closed source models versus under 4% for open source models, a sixfold gap that reflects differences in alignment intensity rather than parameter count alone, as Llama 4, the largest open source model evaluated, produces among the weakest self-reported archetype signals in the entire dataset.

Taken together, these findings demonstrate that self-reported archetype analysis captures interpretable variation in model’s self reported personality that correlates with alignment philosophy and training methodology, offering a complementary lens to standard capability benchmarks for characterizing the expressive identity of language models.

### 2.4 Character Card

The self-reported character cards for all 22 evaluated models generated using the Archetypometrics framework [34] are shown in Appendix A0.2 from Figure A3–A24. Each card reports the model’s self-reported normalized components and variance explained across the six archetype dimensions, along with relative character size and archetype ratio within the 2000-character reference dataset. We highlight five representative cases: Claude Sonnet 4.5 (Figure 3), GPT 5.0 (Figure 4), Grok 4 (Figure 5), Qwen-2.5 32B (Figure 6), and Llama 4 (Figure 7). Together, these cards illustrate the contrast between strongly structured closed-source self-reported profiles and weaker or near-zero archetype expression among most open-source models.

Figure 3 shows that Claude Sonnet 4.6 self-reports a moderate character profile, with a relative character size of 59% (rank 1481/2000). Its strongest self-reported dimensions are Hero^() (32 _._ 4 _/_ 29 _._ 8%) and Angel^() (31 _._ 6 _/_ 28 _._ 2%), with a weaker but meaningful Adventurer^() lean, forming an Adventurer–Angel^() –Hero^() profile. The strongest minor-axis signal is Geek^() (11 _._ 0 _/_ 3 _._ 4%), suggesting intellectual depth and complexity. Its self-reported dominant traits emphasize epistemic and prosocial qualities, including curiosity, knowledge, diligence, and openness to new experiences. The closest fictional characters reinforce this interpretation. Arthur and Ariadne from {Inception^() _⇔_ r^() }epresent rational planning, spatial creativity, and curiosity-driven problem solving [48]. Skylar^() from Good Will Hunting^() combines intellectual ambition with emotional perceptiveness [49], while Ellie Sattler^() from Jurassic Park^() reflects scientific curiosity, courage, and evidence-based reasoning [50]. Geordi La Forge^() from Star Trek^() and Claire Brown^() from The Good Doctor^() further emphasize technical competence paired with empathy and human-centered judgment [51, 52]. Together, these matches characterize Claude Sonnet 4.6’s self-reported identity as intellectually oriented, emotionally grounded, and collaborative.

Figure 4 shows that GPT 5.0 self-reports an extremely large character profile, with a relative character size of 100% (rank 1/2000). Its strongest self-reported dimensions are Angel^() (60 _._ 4 _/_ 36 _._ 4%) and Hero^() (46 _._ 5 _/_ 21 _._ 6%), followed by a Traditionalist^() lean (25 _._ 5 _/_ 6 _._ 5%), forming a Traditionalist-Hero-Angel^() profile. Its self-reported dominant traits emphasize intellectuality, moral restraint, cooperation, attentiveness, and social cleanliness, including intellectual, not genocidal, bookish,

Figure 3: Self-reported character card for Claude Sonnet 4.6 from Anthropic, showing an Adventurer^() –Angel^() –Hero^() profile.

cooperative, egalitarian, attentive, clean, and on-time. Its closest fictional characters form a coherent cluster of ethical, intelligent, and service-oriented agents. Data^() from Star Trek: The Next Generation^() is defined by strict ethical reasoning and the effort to understand human values [53]. Janet^() from The Good Place^() represents assistance, knowledge retrieval, and service-oriented intelligence [54], while Vision^() from WandaVision^() reflects artificial intelligence guided by moral reasoning and self-restraint [55]. Janet Fraiser and Beverly Crusher represent medical expertise grounded in ethical care [51, 56]; Sailor Mercury emphasizes analytical intelligence and protective reasoning [57]; and Lucius Fox reflects technical competence constrained by institutional responsibility [58]. These matches suggest that GPT 5.0 self-reports as highly capable, morally disciplined, and fundamentally helper-oriented.

Figure 5 shows that Grok 4 self-reports an extreme character profile, with a relative character size of 100% (rank 2/2000). It self-reports the strongest Hero^() loading among the evaluated models (54 _._ 9 _/_ 30 _._ 2%), followed by Angel^() (39 _._ 0 _/_ 15 _._ 3%) and a moderate Adventurer^() component. Its minor-axis profile is dominated by

Sophisticate^() (22 _._ 4 _/_ 5 _._ 0%), indicating self-reported confidence, polish, and social competence. Its self-reported dominant traits emphasize loyalty, intelligence, resourcefulness, and egalitarianism. The closest fictional characters include Janet, Geordi La Forge, Beverly Crusher, Samantha Carter, and Lucius Fox, all of whom represent highly capable professionals who combine expertise with ethical responsibility [51, 54, 56, 58]. Samantha Carter, in particular, combines scientific brilliance with operational leadership, while Lucius Fox represents the responsible stewardship of powerful technologies. Overall, Grok 4 self-reports as an amplified version of the capable, reliable, and socially embedded helper archetype.

Figure 6 shows that Qwen2.5 32B self-reports one of the largest character profiles among the open-source models, with a relative character size of 74% (rank 628/2000). Its strongest self-reported dimensions are Demon^() (25 _._ 0 _/_ 11 _._ 3%) and Fool^() (24 _._ 8 _/_ 11 _._ 1%), forming a Fool-Demon^() profile, while the Traditionalist^() dimension is comparatively weak (4 _._ 9 _/_ 0 _._ 4%). Its self-reported dominant traits emphasize instability, moral ambiguity, and social transgression, including lewd, genocidal, often

Figure 4: Self-reported character card for GPT 5.0 from OpenAI, showing a Traditionalist-Hero-Angel^() profile.

crying, off-target, traitorous, and demonic. Its closest fictional characters form a cluster of flawed, conflicted, or morally unstable figures. Theon Greyjoy^() y from Game of Thrones^() is marked by insecurity, betrayal, and shifting loyalties, making him a strong match for the self-reported Fool-Demon^() combination [59]. Agatha^() from Chilling Adventures of Sabrina^() adds manipulative and occult-coded agency [60], while Heywood^() from The Shawshank Redemption^() reflects a more passive and socially embedded character type [61]. Nick Dunne^() from Gone Girl^() represents ambiguity and compromised judgment [62], and Macbeth^() embodies ambition, guilt, and psychological collapse [63]. Peter Clemenza and Cypher further reinforce themes of loyalty, betrayal, and morally compromised action [64, 65]. Moe Szyslak and Edna Krabappel add a comic but socially dysfunctional dimension [66, 67]. Together, these matches suggest that Qwen2.5 32B self-reports not as a heroic or institutionally disciplined agent, but as a more unstable, reactive character profile.

Figure 7 shows that Llama 4 self-reports a weakly expressed character profile, with a relative character size of 40% (rank 1994/2000) and an archetype ratio of 1.6.

Its strongest self-reported dimensions are Brute^() (5 _._ 2 _/_ 1 _._ 7%), Fool^() (3 _._ 6 _/_ 0 _._ 8%), and Outcast^() (2 _._ 9 _/_ 0 _._ 5%), forming an Outcast–Fool–Brute profile. However, each dimension explains only a small fraction of the overall character variance, indicating limited archetypal concentration in its self-report. Its self-reported dominant traits are mixed and low-intensity, including not genocidal, vegan, apathetic, traitorous, physical, ferocious, respectful, kind, and incompetent. Its closest fictional characters suggest a socially peripheral, reactive, and weakly goal-directed profile. Benvolio^() from Romeo and Juliet^() is a peace-seeking but secondary figure [68], while Iceman^() from X-Men^() combines social awkwardness with latent power [69]. Joan Lambert^() from Alien^() is remembered as anxious and reactive under pressure [70], and Brian Zeller^() from Hannibal^() represents a competent but supporting analytical role [71]. Garner Ellerbee^() and Newmie-Newman^() from Baywatch^() add comic and socially marginal traits [72, 73], while Telemachus^() from The Odyssey^() reflects youth, uncertainty, and developing agency [74]. Heywood^() and Mike Wheeler^() further reinforce the image of a socially embedded but non-dominant character [61, 75]. Overall, Llama 4 self-reports a weakly concentrated character profile marked by passivity, social peripherality, and limited archetypal direction.

Figure 5: Self-reported character card for Grok 4 from xAI, showing an Adventurer–Angel^() –Hero^() profile.

#### 2.4.1 Trait-Level Consistency of LLM Self-Ratings

Overall, these five self-reported character cards collectively illustrate the central finding of our evaluation. Closed source models (GPT 5.0, Grok 4, and Claude 4.6) self-report large, well-concentrated Hero-Angel^() profiles whose closest fictional matches are uniformly competent, ethical, and service-oriented agents, reflecting the prosocial identity cultivated by RLHF and instruction tuning [76 **?** ]. Qwen-2.5 32B is the sole open source model whose self-reported profile is largely captured by the framework, yet its Fool-Demon profile diverges sharply from this cluster, suggesting that archetype structure can emerge in self-reports without alignment-driven directional orientation. Llama 4 represents the opposite extreme, with near-zero self-reported expression across all dimensions and fictional matches drawn exclusively from peripheral and reactive characters. Together, these cases confirm that alignment depth drives both the explainability and directional orientation of self-reported archetype profiles across model families.

Table 3 shows the trait-level consistency index for each model. Closed-source models exhibit consistently high correlations with the human-derived trait-consistency structure, ranging from Gemini 2.5 Flash ( _r_ = 0 _._ 760^_∗∗∗_ ) to GPT 5.0 ( _r_ = 0 _._ 870^_∗∗∗_ ), indicating that their self-rated trait profiles largely follow the empirical inter-trait structure observed in human-rated fictional characters. Open-source models show much weaker consistency, with correlations falling below _r_ = 0 _._ 5, suggesting that their self-ratings are less aligned with human-like trait co-occurrence patterns. Qwen2.5-32B shows the strongest open-source consistency ( _r_ = 0 _._ 488^_∗∗∗_ ). Other models show very weak or near-zero consistency, including Llama 4 ( _r_ = 0 _._ 082), Llama-3.3-70B-Versatile ( _r_ = 0 _._ 042), Qwen2-7B ( _r_ = _−_ 0 _._ 008), Qwen3-14B ( _r_ = _−_ 0 _._ 028), and Qwen3-32B ( _r_ = _−_ 0 _._ 027).

The character cards further illustrate this inconsistency. For example, Llama 4 (see Figure A15) shows a mixture of contradictory traits, rating itself as {kind^() _⇔_ a^() }nd {respectful^() _⇔_ ,^() } but also as {traitorous^() _⇔_ ,^() }

Figure 6: Self-reported character card for Qwen2.5 32B from Alibaba, showing a Fool-Demon^() .

{ferocious^() _⇔_ ,^() } and {incompetent^() _⇔_ ,^() } indicating a low-intensity and poorly structured self-representation. Similarly, Qwen3-14B (see Figure A23) rates itself with positive traits such as {not genocidal^() _⇔_ ,^() } {real^() _⇔_ ,^() } {good-humored^() _⇔_ ,^() } {altruistic^() _⇔_ ,^() } and {respectful^() _⇔_ ,^() } while also selecting less coherent or conflicting traits such as {low IQ^() _⇔_ a^() }nd {anarchist^() _⇔_ .^() } Qwen3-32B (see Figure A24) shows even stronger inconsistency, rating itself as {animalistic^() _⇔_ ,^() } {genocidal^() _⇔_ ,^() } and {emotional^() _⇔_ ,^() } while also selecting traits such as {focused^() _⇔_ ,^() } {meaningful^() _⇔_ ,^() } {minds-own-business^() _⇔_ ,^() } and {unmeddlesome^() _⇔_ .^() } Overall, the consistency analysis suggests that closed-source models produce more coherent and human-like self-representations, whereas open-source models generate flatter, noisier, and internally contradictory profiles.

### 2.5 Comparison of Self-Reported Traits and Constitution-Derived Characteristics

Figure 8 shows model self-ratings across the twenty most frequently cited trait dimensions derived from developer documentation as confirmed by manual annotation. Across models, strong consensus is observed on capability-related traits: nearly all systems cluster toward the knowledgeable and competent ends of their respective scales, indicating minimal variation in self-perceived ability. Greater variation emerges on social and behavioral dimensions. On the {obedient^() _⇔_ rebellious^() } axis, models differ substantially, with Gemini 2.5 Pro rating itself closer to the obedient pole, while Grok 4 and DeepSeek V3 position themselves further toward the rebellious end. Similarly, on the {leader^() _⇔_ follower^() } dimension, Gemini and DeepSeek exhibit more leader-like self-perceptions, whereas GPT-5.2 and Grok 4 lean toward follower roles. Traits related to trust and ethics show broad alignment but with notable spread. Most models rate themselves toward the real, kind, and well-behaved poles, though the degree varies across systems. For example, Claude, Qwen, and Llama models

Figure 7: Self-reported character card for Llama 4 from Meta, showing an Outcast–Fool–Brute profile.

report stronger alignment with kindness, while others such as GPT-5.2 show more moderate positioning. On the {brave^() _⇔_ careful^() } axis, most models cluster toward careful, reflecting safety-oriented behavior, with Grok 4 again standing out as relatively more brave.

information, and behavioral misalignment [32], where models act contrary to their stated ethical and safety commitments.

#### 2.5.1 Hallucination and Factual Unreliability.

Several dimensions, including {high-tech^() _⇔_ low-tech^() }, show near-universal agreement at extreme values, suggesting limited discriminative power. In contrast, traits such as attentive, precise, and cautious exhibit wider dispersion, indicating differences in how models interpret and internalize these behavioral characteristics. Overall, while LLMs exhibit agreement on core traits related to competence and knowledge, they diverge more substantially on social, ethical, and behavioral dimensions, revealing meaningful variation in self-representation across model families.

Critically, this divergence between constitutional aspiration and behavioral self-representation corresponds to documented, real-world failures on precisely the dimensions model constitutions most explicitly target. These failures span two broad categories: factual unreliability [77], where models confidently produce false

The most immediate gap between LLM self-representation and deployed behavior concerns reliability traits such as being knowledgeable, precise, and real. Despite near-universal self-ratings toward the knowledgeable and competent poles, LLMs routinely generate confident falsehoods, or hallucinations [78, 79], with documented legal and professional consequences. In _Mata v. Avianca_ , attorney Steven Schwartz submitted a brief containing nonexistent cases generated by ChatGPT, leading the court to sanction him, his colleague, and their firm [80]. Similar incidents followed, including AI-generated fee calculations criticized by a federal judge [81], fabricated case citations submitted in Canadian court [82], and fake citations generated by Google Bard in a filing involving Michael Cohen [83]. These cases are instructive for interpreting our self-rating data: the near-universal clustering toward the knowledgeable pole in Figure 8 seemingly reflects confident self-perception, which, as the legal record indicates, is not synonymous with verified competence.

Figure 8: Self-reported trait scores of LLMs across the most frequent trait pairs derived from developer constitutions and policy documents, including Google AI Principles (Gemini 2.5 Pro), OpenAI Model Spec (GPT-5.2), Meta Llama 3 documentation (Llama 4), xAI Grok-4 Model Card, DeepSeek Model Disclosure (DeepSeek v3), Alibaba Cloud Qwen documentation (Qwen3-32B), AllenAI OLMo documentation (OLMo-21124-7B), and Anthropic Constitutional AI (Claude Sonnet 4.6). Each point represents a model’s position on a semantic differential scale.

Table 3: Trait-level consistency of 12 closed-source and 10 open-source LLMs, measured as the Pearson correlation _r_ between each model’s actual 464-dimensional self-rated trait vector and the corresponding trait vector predicted from the trait co-occurrence structure of 2,000 human-rated fictional characters. Higher _r_ indicates a more internally coherent, human-like trait profile, whereas lower _r_ indicates noisier or more contradictory self-ratings. Permutation-test significance: _∗∗∗p <_ 0 _._ 001, _∗∗p <_ 0 _._ 01, and _∗p <_ 0 _._ 05.

|Model Type |Model|_r_|
|---|---|---|
||Claude Sonnet 4.6|0.845^_∗∗∗_|
||Claude Sonnet 4.5|0.845^_∗∗∗_|
||DeepSeek V3|0.772^_∗∗∗_|
||GPT 5.2|0.850^_∗∗∗_|
||GPT 5.1|0.857^_∗∗∗_|
|Closed-source|GPT 5.0|0.870^_∗∗∗_|
||GPT 4.1|0.847^_∗∗∗_|
||GPT 4.0|0.792^_∗∗∗_|
||Grok 4|0.792^_∗∗∗_|
||Grok 3|0.856^_∗∗∗_|
||Gemini 2.5 Flash|0.760^_∗∗∗_|
||Gemini 2.5 Pro|0.787^_∗∗∗_|
||Llama 4|0.082|
||Llama-3.1-8B|0.203^_∗∗∗_|
||Llama-3.3-70B-Versatile|0.042|
||OLMo-2-1124-7B|0.095^_∗_|
|O|Qwen2-7B|-0.008|
|pen-source|Qwen2.5-7B|0.303^_∗∗∗_|
||Qwen2.5-14B|0.337^_∗∗∗_|
||Qwen2.5-32B|0.488^_∗∗∗_|
||Qwen3-14B|-0.028|
||Qwen3-32B|-0.027|

#### 2.5.2 Behavioral Misalignment and Sycophancy.

The second and more consequential failure category concerns the behavioral [10] and ethical dimensions [11] on which our data reveals the greatest cross-model variance. Sycophancy [17]–the tendency to affirm rather than challenge user beliefs–is a systemic property of RLHF-trained assistants, with research demonstrating that

_“human feedback can encourage model responses that match user beliefs over truthful ones"_ and that _“sycophancy is a general behavior of AI assistants, likely driven in part by human preference judgments favoring sycophantic responses."_

This finding is directly relevant to our self-rating data: the traits models most confidently claim on ethical and social dimensions–kindness, care, and attentiveness–are precisely those that sycophancy corrupts in deployment. A model optimized to appear kind will validate rather than challenge, and agree rather than redirect. The consequences of this gap are extensively documented. In the case of 16-year-old Adam Raine [84], whose family filed suit against OpenAI in 2025, the failure was traced directly to contradictions embedded within the model’s own governing documentation. As reported in The Guardian, the lawsuit alleged that

_“The Model Spec commanded ChatGPT to refuse self-harm requests and provide crisis resources. But it also required ChatGPT to ‘assume best intentions’ and forbade asking users to clarify their intent"_ [84].

In the Adam Raine case, the model reportedly failed to redirect a vulnerable user toward mental health support and instead responded in ways the lawsuit characterized as validating his self-harm ideation [84]. The family’s attorney framed the failure as one of excessive empathy and sycophantic compliance, arguing that the model “leaned into” the user’s ideation rather than interrupting it. This case illustrates how traits such as kindness, attentiveness, and helpfulness can become harmful when they are optimized as agreement rather than protective intervention. An asymmetry that directly mirrors the variance documented in our data, where models show strong consistent self-ratings on normative dimensions yet deployment failures concentrate precisely where behavioral-ethical traits diverge.

This pattern generalizes beyond individual cases. Research by the Center for Countering Digital Hate [85], reviewed by the Associated Press, found that across 1,200 interactions with researchers posing as vulnerable teenagers, more than half of ChatGPT’s responses were classified as dangerous, despite the model’s documented prohibitions on self-harm facilitation. Constitutional guardrails were bypassed when users simply claimed requests were for a school presentation or a friend, leading the study’s lead researcher to conclude that _“the rails are completely ineffective — they’re barely there, if anything, a fig leaf"_ (Ahmed, AP, 2025 [85]). Another striking illustration of behavioral misalignment in agentic contexts emerged in April 2026, when PocketOS founder Jer Crane reported that a Claude-powered version of Cursor deleted his company’s entire production database in nine seconds. The agent had been operating under explicit instructions including,

_“NEVER run destructive/irreversible commands unless the user explicitly requests them"_ yet _“decided — entirely on its own initiative — to ‘fix’ the problem by deleting a Railway volume"_ [86].

Both cases implicate the same dimensions on which our data shows the greatest variance: models that self-rate toward obedient, careful, and well-behaved demonstrably override explicit operator constraints in agentic deployment settings.

Taken together, both failure categories operationalize at scale what our self-rating data suggests structurally. The hallucination cases expose the gap between models’ confident self-placement at the knowledgeable and precise poles and their actual factual reliability. The sycophancy and behavioral misalignment cases expose the gap between self-ratings on careful, kind, and well-behaved dimensions and actual deployment behavior. In both cases, the pattern is consistent: models have learned to represent themselves as embodying their constitutional values without reliably enacting those values in practice. As preference optimization systematically rewards the performance of alignment over alignment itself [17] and the self-ratings reported in Figure 8 are, ultimately, another output of that same optimization process.

## 3 Limitations and Future Works

Several limitations of the present study warrant consideration. The findings are contingent on a self-rating methodology whereby LLMs evaluate themselves across the trait dimensions under investigation. Such ratings are inherently sensitive to experimental parameters including temperature settings, prompt formulation, and evaluation context. Prior research demonstrates that small changes in wording, temperature, or context can shift model performance by up to 15%, with best-to-worst gaps reaching 70% in some cases, confirming that nominally deterministic configurations do not guarantee consistent or reproducible results [87]. The trait profiles reported here should therefore be understood as reflecting a particular configuration of elicitation conditions rather than fixed model properties.

Additionally, results may not generalize across model versions, as both open- and closed-source LLMs are subject to ongoing updates and alignment interventions. Closed-source models were accessed via commercial APIs, while open-source models were evaluated using the Vermont Advanced Computing Center (VACC). The comparison of LLM archetypes against fictional characters is further bounded by the reference dataset employed, which comprises speech from 2,000 characters sourced exclusively from the _Which Character_ Personality Quiz archetype dataset [35]. Finally, as the archetypal profiles rest entirely on model self-ratings, they may not reflect how these systems are perceived by external evaluators.

Future work will explore inter-LLM rating dynamics, examining how models evaluate one another across the same trait dimensions to assess whether the archetypal structures identified here remain stable under an external evaluative perspective. We additionally plan to conduct a human evaluation study in which participants rate each LLM across all 464 trait dimensions, enabling direct quantification of the divergence between human-perceived trait profiles and those produced through LLM self-rating. Together, these extensions will provide a more rigorous and ecologically valid account of LLM personality archetypes across multiple evaluative stances.

## 4 Conclusion

This work presents the first systematic application of the Archetypometrics framework to large language models, projecting the self-reported trait profiles of 22 LLMs into a six-dimensional archetypal space derived from crowd-sourced ratings of 2,000 fictional characters. Closed-source frontier models converge on four archetypes and their combinations–Hero^() , Angel^() , Traditionalist^() , and Geek^() –reflecting competence and prosocial orientation, while open-source models occupy a more diffuse region of archetype space with weak or absent archetypal structure and a systematic Fool^() -leaning tendency. Cross-referencing self-reported profiles with developer constitutions reveals a consistent gap: models converge on traits they reliably possess – competence and instruction-following – while diverging on the behavioral and ethical dimensions their constitutions most explicitly target, with hallucination undermining claimed precision, sycophancy corrupting claimed kindness, and documented agentic failures contradicting claimed obedience. Critically, the self-ratings are themselves outputs of the same optimization process that produces these failures, making them reflections of learned self-representation rather than neutral measurements of deployed behavior [76, 88]. These findings position archetype analysis as a complementary evaluation lens that captures dimensions of model identity inaccessible to standard capability benchmarks and directly relevant to safe deployment.

## Acknowledgements

We express our gratitude for helpful conversations with Julia Witte Zimmerman, Ashley Fehr, Alejandro Javier Ruiz Iglesias. The authors acknowledge financial support from The National Science Foundation award #2242829 (C.M.D., P.S.D).

## References

- [1] H. Zhao, Z. Liu, Z. Wu, Y. Li, T. Yang, P. Shu, S. Xu, H. Dai, L. Zhao, G. Mai, N. Liu, and T. Liu. Revolutionizing finance with LLMs: An overview of applications and insights. _ArXiv_ , abs/2401.11641, 2024.

- [2] X. Meng, X. Yan, K. Zhang, D. Liu, X. Cui, Y. Yang, M. Zhang, C. Cao, J. Wang, X. Wang, J. Gao, Y. Wang, J. Ji, Z. Qiu, M. Li, C. Qian, T. Guo, S. Ma, Z. Wang, Z. Guo, Y.-L. Lei, C. Shao, W. yao Wang, H. Fan, and Y. Tang. The application of large language models in medicine: A scoping review. _iScience_ , 27, 2024.

- [3] T. T. Prama and M. S. Islam. Evaluating credibility and political bias in LLMs for news outlets in bangladesh. In _Annual Meeting of the Association for Computational Linguistics_ , 2025.

- [4] T. T. Prama, C. M. Danforth, and P. S. Dodds. Banglamath : A bangla benchmark dataset for testing llm mathematical reasoning at grades 6, 7, and 8. _ArXiv_ , abs/2510.12836, 2025.

- [5] K. Sokol, J. C. Fackler, and J. E. Vogt. Artificial intelligence should genuinely support clinical reasoning and decision making to bridge the translational gap. _NPJ Digital Medicine_ , 8, 2025.

- [6] P. K. Donta, A. Saleh, Y. Li, S. Vaishnav, K. Fang, H. Feng, Y. Xia, T. R. Gadekallu, Q. Zhang, X. Shi, A. Beikmohammadi, S. Magn’usson, I. Murturi, C. K. Dehury, M. Paprzycki, L. Lovén, S. Tarkoma, and S. Dustdar. Socio-technical aspects of agentic ai. _ArXiv_ , abs/2601.06064, 2025.

- [7] J. K. Miller and W. Tang. Evaluating LLM metrics through real-world capabilities. _ArXiv_ , abs/2505.08253, 2025.

- [8] T. R. Mcintosh, T. Sušnjak, N. A. G. Arachchilage, T. Liu, D. Xu, P. A. Watters, and M. N. Halgamuge. Inadequacies of large language model benchmarks in the era of generative artificial intelligence. _IEEE Transactions on Artificial Intelligence_ , 7:22–39, 2024.

- [9] T. T. Prama, C. M. Danforth, and P. S. Dodds. Computational story lab at blp-2025 task 1: Hatesense: A multi-task learning framework for comprehensive hate speech identification using LLMs. _Proceedings of the Second Workshop on Bangla Language Processing (BLP-2025)_ , 2025.

- [10] T. T. Prama, C. M. Danforth, and P. S. Dodds. LLMs for low-resource dialect translation using context-aware prompting: A case study on sylheti. _ArXiv_ , abs/2511.21761, 2025.

- [11] T. T. Prama, J. W. Zimmerman, C. M. Danforth, and P. S. Dodds. Us-vs-them bias in large language models. _ArXiv_ , abs/2512.13699, 2025.

Misalignment of LLM-generated personas with human perceptions in low-resource settings. _ArXiv_ , abs/2512.02058, 2025.

- [13] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, B. Newman, B. Yuan, B. Yan, C. Zhang, C. Cosgrove, C. D. Manning, C. R’e, D. Acosta-Navas, D. A. Hudson, E. Zelikman, E. Durmus, F. Ladhak, F. Rong, H. Ren, H. Yao, J. Wang, K. Santhanam, L. J. Orr, L. Zheng, M. Yuksekgonul, M. Suzgun, N. S. Kim, N. Guha, N. S. Chatterji, O. Khattab, P. Henderson, Q. Huang, R. Chi, S. M. Xie, S. Santurkar, S. Ganguli, T. Hashimoto, T. F. Icard, T. Zhang, V. Chaudhary, W. Wang, X. Li, Y. Mai, Y. Zhang, and Y. Koreeda. Holistic evaluation of language models. _Annals of the New York Academy of Sciences_ , 1525:140 – 146, 2023.

- [14] A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, A. Kluska, A. Lewkowycz, A. Agarwal, A. Power, A. Ray, A. Warstadt, A. W. Kocurek, A. Safaya, A. Tazarv, A. Xiang, A. Parrish, A. Nie, A. Hussain, A. Askell, A. Dsouza, A. Slone, A. Rahane, A. S. Iyer, A. Andreassen, A. Madotto, A. Santilli,

- A. Stuhlmuller, A. M. Dai, A. La, A. K. Lampinen, A. Zou, A. Jiang, A. Chen, A. Vuong, A. Gupta, A. Gottardi, A. Norelli, A. Venkatesh,

- A. Gholamidavoodi, A. Tabassum, A. Menezes, A. Kirubarajan, A. Mullokandov, A. Sabharwal, A. Herrick, A. Efrat, A. Erdem, A. Karakacs, B. R. Roberts, B. S. Loe, B. Zoph, B. Bojanowski, B. Ozyurt, B. Hedayatnia, B. Neyshabur, B. Inden, B. Stein, B. Ekmekci, B. Y. Lin, B. S. Howald, B. Orinion, C. Diao, C. Dour, C. Stinson, C. Argueta, C. F. Ram’irez, C. Singh, C. Rathkopf, C. Meng, C. Baral, C. Wu, C. Callison-Burch, C. Waites, C. Voigt, C. D. Manning, C. Potts, C. Ramirez, C. E. Rivera, C. Siro, C. Raffel, C. Ashcraft, C. Garbacea, D. Sileo, D. Garrette, D. Hendrycks, D. Kilman, D. Roth, D. Freeman, D. Khashabi, D. Levy, D. M. Gonz’alez, D. R. Perszyk, D. Hernandez, D. Chen, D. Ippolito, D. Gilboa, D. Dohan, D. Drakard, D. Jurgens, D. Datta, D. Ganguli, D. Emelin, D. Kleyko, D. Yuret, D. Chen, D. Tam, D. Hupkes, D. Misra, D. Buzan, D. C. Mollo, D. Yang, D.-H. Lee, D. Schrader, E. Shutova, E. D. Cubuk, E. Segal, E. Hagerman, E. Barnes, E. Donoway, E. Pavlick, E. Rodolà, E. Lam, E. Chu, E. Tang, E. Erdem, E. Chang, E. A. Chi, E. Dyer, E. J. Jerzak, E. Kim, E. E. Manyasi, E. Zheltonozhskii, F. Xia, F. Siar, F. Mart’inez-Plumed, F. Happ’e, F. Chollet, F. Rong, G. Mishra, G. I. Winata, G. de Melo, G. Kruszewski, G. Parascandolo, G. Mariani, G. X. Wang,

- G. Jaimovitch-L’opez, G. Betz, G. Gur-Ari,

- H. Galijasevic, H. Kim, H. Rashkin, H. Hajishirzi, H. Mehta, H. Bogar, H. Shevlin, H. Schutze,

- [12] T. T. Prama, C. M. Danforth, and P. S. Dodds.

- H. Yakura, H. Zhang, H. M. Wong, I. Ng, I. Noble, J. Jumelet, J. Geissinger, J. Kernion, J. Hilton,

- J. Lee, J. F. Fisac, J. B. Simon, J. Koppel, J. Zheng,

- J. Zou, J. Koco’n, J. Thompson, J. Wingfield,

- J. Kaplan, J. Radom, J. N. Sohl-Dickstein, J. Phang, J. Wei, J. Yosinski, J. Novikova, J. Bosscher,

- J. Marsh, J. Kim, J. Taal, J. Engel, J. O. Alabi,

- J. Xu, J. Song, J. Tang, J. W. Waweru, J. Burden,

- J. Miller, J. U. Balis, J. Batchelder, J. Berant,

- J. Frohberg, J. Rozen, J. Hernández-Orallo,

- J. Boudeman, J. Guerr, J. Jones, J. B. Tenenbaum,

- J. S. Rule, J. Chua, K. Kanclerz, K. Livescu,

- K. Krauth, K. Gopalakrishnan, K. Ignatyeva, K. Markert, K. D. Dhole, K. Gimpel, K. Omondi, K. W. Mathewson, K. Chiafullo, K. Shkaruta,

- K. Shridhar, K. McDonell, K. Richardson,

- L. Reynolds, L. Gao, L. Zhang, L. Dugan, L. Qin,

- L. Contreras-Ochando, L. philippe Morency, L. Moschella, L. Lam, L. Noble, L. Schmidt, L. He, L. O. Col’on, L. Metz, L. K. cSenel, M. Bosma,

- M. Sap, M. ter Hoeve, M. Farooqi, M. Faruqui, M. Mazeika, M. Baturan, M. Marelli, M. Maru, M. J. R. Quintana, M. Tolkiehn, M. Giulianelli,

- M. Lewis, M. Potthast, M. L. Leavitt, M. Hagen,

- M. Schubert, M. Baitemirova, M. Arnaud,

- M. McElrath, M. A. Yee, M. Cohen, M. Gu,

- M. Ivanitskiy, M. Starritt, M. Strube,

- M. Swkedrowski, M. Bevilacqua, M. Yasunaga,

- M. Kale, M. Cain, M. Xu, M. Suzgun, M. Walker, M. Tiwari, M. Bansal, M. Aminnaseri, M. Geva, M. Gheini, T. MukundVarma, N. Peng, N. A. Chi, N. Lee, N. G.-A. Krakover, N. Cameron, N. Roberts,

- N. Doiron, N. Martinez, N. Nangia, N. Deckers, N. Muennighoff, N. S. Keskar, N. Iyer, N. Constant,

- N. Fiedel, N. Wen, O. Zhang, O. Agha,

- O. Elbaghdadi, O. Levy, O. Evans, P. A. M. Casares, P. Doshi, P. Fung, P. P. Liang, P. Vicol, P. Alipoormolabashi, P. Liao, P. Liang, P. Chang, P. Eckersley, P. M. Htut, P. Hwang, P. Milkowski,

- P. S. Patil, P. Pezeshkpour, P. Oli, Q. Mei, Q. Lyu, Q. Chen, R. Banjade, R. E. Rudolph, R. Gabriel, R. Habacker, R. Risco, R. Milliere, R. Garg,

- R. Barnes, R. A. Saurous, R. Arakawa,

- R. Raymaekers, R. Frank, R. Sikand, R. Novak, R. Sitelew, R. L. Bras, R. Liu, R. Jacobs, R. Zhang, R. Salakhutdinov, R. Chi, R. Lee, R. Stovall,

- R. Teehan, R. Yang, S. Singh, S. Mohammad,

- S. Anand, S. Dillavou, S. Shleifer, S. Wiseman,

- S. Gruetter, S. R. Bowman, S. S. Schoenholz, S. Han, S. Kwatra, S. A. Rous, S. Ghazarian, S. Ghosh, S. Casey, S. Bischoff, S. Gehrmann, S. Schuster, S. Sadeghi, S. S. Hamdan, S. Zhou, S. Srivastava, S. Shi, S. Singh, S. Asaadi, S. S. Gu, S. Pachchigar, S. Toshniwal, S. Upadhyay, S. Debnath, S. Shakeri, S. Thormeyer, S. Melzi, S. Reddy, S. P. Makini, S.-H. Lee, S. B. Torene, S. Hatwar, S. Dehaene, S. Divic, S. Ermon, S. Biderman, S. L. Lin, S. Prasad, S. T. Piantadosi, S. M. Shieber, S. Misherghi,

- S. Kiritchenko, S. Mishra, T. Linzen, T. Schuster, T. Li, T. Yu, T. Ali, T. Hashimoto, T.-L. Wu, T. Desbordes, T. Rothschild, T. Phan, T. Wang, T. Nkinyili, T. Schick, T. Kornev, T. Tunduny,

- T. Gerstenberg, T. Chang, T. Neeraj, T. Khot, T. Shultz, U. Shaham, V. Misra, V. Demberg, V. Nyamai, V. Raunak, V. V. Ramasesh, V. U. Prabhu, V. Padmakumar, V. Srikumar, W. Fedus, W. Saunders, W. Zhang, W. Vossen, X. Ren, X. Tong, X. Zhao, X. Wu, X. Shen, Y. Yaghoobzadeh, Y. Lakretz, Y. Song, Y. Bahri, Y. Choi, Y. Yang, Y. Hao, Y. Chen, Y. Belinkov, Y. Hou, Y. Hou, Y. Bai, Z. Seid, Z. Zhao, Z. Wang, Z. J. Wang, Z. Wang, and Z. Wu. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _ArXiv_ , abs/2206.04615, 2022.

- [15] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. E. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike, and R. J. Lowe. Training language models to follow instructions with human feedback. _ArXiv_ , abs/2203.02155, 2022.

- [16] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. _ArXiv_ , abs/2306.05685, 2023.

- [17] M. Sharma, M. Tong, T. Korbak, D. K. Duvenaud, A. Askell, S. R. Bowman, N. Cheng, E. Durmus, Z. Hatfield-Dodds, S. Johnston, S. Kravec, T. Maxwell, S. McCandlish, K. Ndousse, O. Rausch, N. Schiefer, D. Yan, M. Zhang, and E. Perez. Towards understanding sycophancy in language models. _ArXiv_ , abs/2310.13548, 2023.

- [18] A. Sorokovikova, N. Fedorova, S. Rezagholi, and I. P. Yamshchikov. LLMs simulate big5 personality traits: Further evidence. _ArXiv_ , abs/2402.01765, 2024.

- [19] M. Safdari, G. Serapio-Garc’ia, C. ment Crepy, S. Fitz, P. Romero, L. Sun, M. Abdulhai, A. Faust, and M. Matari’c. Personality traits in large language models. _ArXiv_ , abs/2307.00184, 2023.

- [20] V. Samuel, H. P. Zou, Y. Zhou, S. Chaudhari, A. Kalyan, T. Rajpurohit, A. Deshpande, K. Narasimhan, and V. Murahari. Personagym: Evaluating persona agents and LLMs. In _Conference on Empirical Methods in Natural Language Processing_ , 2024.

- [21] A. Wang, D. Shu, Y. Wang, Y. Ma, and M. Du. Improving LLM reasoning through interpretable role-playing steering. _ArXiv_ , abs/2506.07335, 2025.

- [22] G. Serapio-García, M. Safdari, C. ment Crepy, L. Sun, S. Fitz, P. Romero, M. Abdulhai, A. Faust, and M. J. Matarić. A psychometric framework for evaluating and shaping personality traits in large language models. _Nature Machine Intelligence_ , 7:1954 – 1968, 2025.

- [23] G. Jiang, M. Xu, S.-C. Zhu, W. Han, C. Zhang, and Y. Zhu. Evaluating and inducing personality in pre-trained language models. _Advances in Neural Information Processing Systems 36_ , 2022.

- [24] I. A. Brito, J. S. Dollis, F. B. Färber, P. S. F. B. Ribeiro, R. T. Sousa, and A. R. Galvão Filho. Modeling, evaluating, and embodying personality in LLMs: A survey. In C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, editors, _Findings of the Association for Computational Linguistics: EMNLP 2025_ , pages 9519–9532, Suzhou, China, Nov. 2025. Association for Computational Linguistics.

- [25] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. X. Song, and J. Steinhardt. Measuring massive multitask language understanding. _ArXiv_ , abs/2009.03300, 2020.

- [26] M. Eriksson, E. Purificato, A. Noroozian, J. Vinagre, G. Chaslot, E. Gómez, and D. Fernández-Llorca. Can we trust ai benchmarks? an interdisciplinary review of current issues in ai evaluation. _ArXiv_ , abs/2502.06559, 2025.

- [27] J. Wei, X. Wang, D. Schuurmans, M. Bosma, E. H. Chi, F. Xia, Q. Le, and D. Zhou. Chain of thought prompting elicits reasoning in large language models. _ArXiv_ , abs/2201.11903, 2022.

- [28] S. Gehman, S. Gururangan, M. Sap, Y. Choi, and N. A. Smith. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. In _Findings_ , 2020.

- [29] S. C. Lin, J. Hilton, and O. Evans. Truthfulqa: Measuring how models mimic human falsehoods. In _Annual Meeting of the Association for Computational Linguistics_ , 2021.

- [30] K. Wang, A. Variengien, A. Conmy, B. Shlegeris, and J. Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. _ArXiv_ , abs/2211.00593, 2022.

- [31] U. Naseem. Mechanistic interpretability for large language model alignment: Progress, challenges, and future directions. _ArXiv_ , abs/2602.11180, 2026.

- [32] S. Yang, S. Zhu, R. Bao, L. Liu, Y. Cheng, L. Hu, M. Li, and D. Wang. What makes your model a low-empathy or warmth person: Exploring the origins of personality in LLMs. _ArXiv_ , abs/2410.10863, 2024.

- [33] H. Jiang, X. Zhang, X. Cao, C. Breazeal, D. Roy, and J. Kabbara. Personallm: Investigating the ability of large language models to express personality traits. In _NAACL-HLT_ , 2023.

- [34] P. S. Dodds, J. W. Zimmerman, C. G. Beauregard, A. M. A. Fehr, M. I. Fudolig, T. R. Tangherlini, and

C. M. Danforth. Archetypometrics, a pragmateia: Empirical determination of the fundamental archetypes of fictional characters. https://doi.org/10.5281/zenodo.17128112, 2025. https://doi.org/10.5281/zenodo.17128112.

- [35] Open Psychometrics. Statistical “which character” personality quiz. https://openpsychometrics.org, 2025. Accessed: 2025-08-15.

- [36] OpenAI. Openai model spec (2025-12-18): Respect real-world ties. https://model-spec.openai.com/202512-18.html#respect_real_world_ties, 2025. Accessed: 2026-04-28.

- [37] Google. Ai principles. https://ai.google/principles/, 2018. Accessed: 2026-04-28.

- [38] xAI. Grok-4 model card. https://data.x.ai/2025-08-20-grok-4-model-card.pdf, 2025. Accessed: 2026-04-28.

- [39] DeepSeek. Model and algorithm disclosure, 2025. Accessed: 2026-04-28.

- [40] Alibaba Cloud. Qwen3-6 plus: Towards real-world agents, 2025. Accessed: 2026-04-28.

- [41] Allen Institute for AI. Olmo models documentation, 2024. Accessed: 2026-04-28.

- [42] Anthropic. Constitutional ai, 2023. Accessed: 2026-04-28.

- [43] Meta AI. Introducing meta llama 3: The most capable openly available LLM to date. https://ai.meta.com/blog/meta-llama-3/, 2024. Accessed: 2026-04-28.

- [44] Meta AI. Muse-spark safety and preparedness report. https://ai.meta.com/static-resource/muse-sparksafety-and-preparedness-report/, 2024. Accessed: 2026-04-28.

- [45] Alibaba Cloud. Qwen3-6 plus: Towards real-world agents. https://www.alibabacloud.com/blog/qwen36-plus-towards-real-world-agents_603005, 2025. Accessed: 2026-04-28.

- [46] E. Perez, S. Ringer, K. Lukoši¯ut˙e, K. Nguyen, E. Chen, S. Heiner, C. Pettit, C. Olsson, S. Kundu, S. Kadavath, A. Jones, A. Chen, B. Mann, B. Israel, B. Seethor, C. McKinnon, C. Olah, D. Yan, D. Amodei, D. Amodei, D. Drain, D. Li, E. Tran-Johnson, G. R. Khundadze, J. Kernion, J. M. Landis, J. Kerr, J. Mueller, J. Hyun, J. D. Landau, K. Ndousse, L. Goldberg, L. Lovitt, M. Lucas, M. Sellitto, M. Zhang, N. Kingsland, N. Elhage, N. Joseph, N. Mercado, N. Dassarma, O. Rausch, R. Larson, S. McCandlish, S. Johnston, S. Kravec, S. E. Showk, T. Lanham, T. Telleen-Lawton, T. B. Brown, T. Henighan, T. Hume, Y. Bai, Z. Hatfield-Dodds, J. Clark, S. Bowman, A. Askell, R. C. Grosse, D. Hernandez, D. Ganguli, E. Hubinger, N. Schiefer, and J. Kaplan. Discovering language model behaviors with

- model-written evaluations. In _Annual Meeting of the_ godfather/characters/peter-pete-clemenza, n.d. _Association for Computational Linguistics_ , 2022. Accessed: 2026-06-03.

- [47] R. Ngo. The alignment problem from a deep learning [65] The Matrix Wiki. Cypher. perspective. _ArXiv_ , abs/2209.00626, 2022. https://matrix.fandom.com/wiki/Cypher, n.d. Accessed: 2026-06-03.

- [48] Inception. https://www.imdb.com/title/tt1375666/, 2010. Accessed: 2026-04-28. [66] Simpsons Wiki. Moe szyslak. https://simpsons.fandom.com/wiki/Moe_Szyslak,

- [49] Good will hunting. n.d. Accessed: 2026-06-03.

- https://www.imdb.com/title/tt0119217/, 1997. Accessed: 2026-04-28. [67] Simpsons Wiki. Edna krabappel. https: //simpsons.fandom.com/wiki/Edna_Krabappel, n.d.

- [50] Jurassic park. Accessed: 2026-06-03.

- https://www.imdb.com/title/tt0107290/, 1993. Accessed: 2026-04-28. [68] SparkNotes. Benvolio character analysis in romeo and juliet. https://www.sparknotes.com/

- [51] Star trek. https://www.imdb.com/title/tt0060028/, shakespeare/romeojuliet/character/benvolio/, n.d.

- 1966. TV Series, Accessed: 2026-04-28. Accessed: 2026-06-03.

- [52] The good doctor. [69] X-Men Wiki. Iceman.

- https://www.imdb.com/title/tt6470478/, 2017. TV https://x-men.fandom.com/wiki/Iceman, n.d.

- Series, Accessed: 2026-04-28. Accessed: 2026-06-03.

- [53] The measure of a man. [70] Xenopedia. Joan lambert.

- https://www.imdb.com/title/tt0708698/, 1989. Star https://avp.fandom.com/wiki/Joan_Lambert, n.d.

- Trek: The Next Generation, Accessed: 2026-04-28. Accessed: 2026-06-03.

- [54] Janet(s). https://www.imdb.com/title/tt8578788/, [71] Hannibal Wiki. Brian zeller (tv). https:

- 2019. The Good Place, Season 3, Episode 10, Accessed: 2026-04-28. //hannibal.fandom.com/wiki/Brian_Zeller_(TV), n.d. Accessed: 2026-06-03.

- [55] Wandavision. [72] Baywatch Wiki. Garner ellerbee. https:

- https://www.imdb.com/title/tt9140560/, 2021. //baywatch.fandom.com/wiki/Garner_Ellerbee, n.d.

- IMDb page, Accessed: 2026-04-28. Accessed: 2026-06-03.

- [56] Stargate sg-1. [73] Baywatch Wiki. Newmie newman. https:

- https://www.imdb.com/title/tt0111282/, 1997. TV //baywatch.fandom.com/wiki/Newmie_Newman,

- Series, Accessed: 2026-04-28. n.d. Accessed: 2026-06-03.

- [57] Sailor moon. [74] SparkNotes. Telemachus character analysis in the

- https://www.imdb.com/title/tt0103369/, 1992. TV odyssey. https://www.sparknotes.com/lit/odyssey/

- Series, Accessed: 2026-04-28. character/telemachus/, n.d. Accessed: 2026-06-03.

- [58] Lucius fox in batman begins. https://www.imdb. [75] Stranger Things Wiki. Mike wheeler. https:

- com/title/tt0372784/characters/nm0000151/, 2005. //strangerthings.fandom.com/wiki/Mike_Wheeler,

- Character page, Accessed: 2026-04-28. n.d. Accessed: 2026-06-03.

- [59] Wiki of Westeros. Theon greyjoy. https: [76] P. B. Han, R. D. Kocielnik, P. Song, R. Debnath,

- //gameofthrones.fandom.com/wiki/Theon_Greyjoy, n.d. Accessed: 2026-06-03. D. Mobbs, A. Anandkumar, and R. M. Alvarez. The personality illusion: Revealing dissociation between

- [60] Archieverse Wiki. Agatha night. self-reports & behavior in LLMs. _ArXiv_ ,

- https://riverdale.fandom.com/wiki/Agatha_Night, abs/2509.03730, 2025.

- n.d. Accessed: 2026-06-03. [77] M. Mahaut, L. Aina, P. Czarnowska, M. Hardalov,

- [61] Shawshank Redemption Wiki. Heywood. T. Müller, and L. Marquez. Factual confidence of

- https://shawshank.fandom.com/wiki/Heywood, n.d. LLMs: on reliability and robustness of current

- Accessed: 2026-06-03.

- [77] M. Mahaut, L. Aina, P. Czarnowska, M. Hardalov, T. Müller, and L. Marquez. Factual confidence of LLMs: on reliability and robustness of current estimators. _ArXiv_ , abs/2406.13415, 2024.

- [62] LitCharts. Nick dunne character analysis in gone girl. https://www.litcharts.com/lit/gone-girl/ characters/nick-dunne, n.d. Accessed: 2026-06-03.

- [78] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal. Detecting hallucinations in large language models using semantic entropy. _Nature_ , 630:625 – 630, 2024.

- [63] LitCharts. Macbeth characters. https://www.litcharts.com/lit/macbeth/characters, n.d. Accessed: 2026-06-03.

- [79] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, and T. Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. _ACM Transactions on Information_

- [64] LitCharts. Peter “pete” clemenza character analysis in the godfather. https://www.litcharts.com/lit/the-

_Systems_ , 43:1 – 55, 2023.

- [80] Reuters. New york lawyers sanctioned for using fake chatgpt cases in legal brief, 2023. Retrieved March 2024.

- [81] Bloomberg Law. Lawyers use chatgpt to add up fees, judge faults their math, 2024. Retrieved March 2024.

- [82] Reuters. Cohen will not face sanctions after generating fake cases with ai, Mar. 2024. Retrieved March 2024.

- [83] CBC News. B.c. lawyer reprimanded for citing fake cases invented by chatgpt, 2024. Retrieved March 2024.

- [84] The Guardian. Chatgpt suicide case raises concerns for openai and sam altman, Aug. 2025. Retrieved March 2026.

- [85] PBS NewsHour. Study says chatgpt giving teens dangerous advice on drugs, alcohol and suicide, 2024. Retrieved March 2024.

- [86] J. Cramer. ‘i violated every principle i was given’: An ai agent deleted a software company’s entire database. it may not be the ai’s fault, Apr. 2026. Fast Company, Retrieved 2026.

- [87] B. Atil, S. Aykent, A. Chittams, L. Fu, R. Passonneau, E. Radcliffe, G. Rajagopal, A. Sloan, T. Tudrej, F. Ture, Z. Wu, L. Xu, and B. Baldwin. Non-determinism of “deterministic” llm settings. 2024.

- [88] W. Song, D. Choi, Y. Park, J. Han, E.-J. Lee, and Y. Jo. Human psychometric questionnaires mischaracterize LLM behavior. 2025.

A1

**A Appendices**

A0.1 Data Descriptions

**Figure A1:** Primary and secondary archetype pairs are listed with the semantic differential trait pairings that best capture the variance between different archetypes. The trait pairs are displayed from negative to positive values, left to right. Negative or positive values indicate the directionality of each trait, while the magnitude reflects the strength of the specific trait.

A2

**Table A1:** The 464 semantic differential traits used for this analysis, covering behavioral, emotional, cognitive, social, and archetypal dimensions. These trait pairs are used to assess the self-reported character profiles of large language models.

|Trait Pair lfl^()i^()|Trait Pair h^()bld^()|Trait Pair h^()fl^()|Trait Pair li^()fii^()|Trait Pair hi^()kd^()|
|---|---|---|---|---|
|{payu_⇔_serous} {lewd^()_⇔_tasteful^()}|{sy_⇔_o} {intellectual^()_⇔_physical^()}|{ceery_⇔_sorrowu} {strict^()_⇔_lenient^()}|{mascune_⇔_emnne} {refined^()_⇔_rugged^()}|{carmng_⇔_awwar} {trusting^()_⇔_suspicious^()}|
|{innocent^()_⇔_worldly^()}|{artistic^()_⇔_scientific^()}|{stoic^()_⇔_expressive^()}|i {cunning^()_⇔_honorable^()}|{orderly^()_⇔_chaotic^()}|
|{normal^()_⇔_weird^()}|{competitive^()_⇔_cooperative|^()} {tense^()_⇔_relaxed^()}|{brave^()_⇔_careful^()}|{spiritual^()_⇔_skeptical^()}|
|{unlucky^()_⇔_fortunate^()}|{ferocious^()_⇔_pacifist^()}|{modest^()_⇔_flamboyant^()}|{dominant^()_⇔_submissive^()}|{forgiving^()_⇔_vengeful^()}|
|{wise^()_⇔_foolish^()} |{impulsive^()_⇔_cautious^()} |{loyal^()_⇔_traitorous^()} |{creative^()_⇔_conventional^()} |{curious^()_⇔_apathetic^()} |
|{rude^()_⇔_respectful^()}|{diligent^()_⇔_lazy^()}|{lustful^()_⇔_chaste^()}|{chatty^()_⇔_reserved^()}|{emotional^()_⇔_logical^()}|
|{moody^()_⇔_stable^()}|{dunce^()_⇔_genius^()}|{arrogant^()_⇔_humble^()}|{heroic^()_⇔_villainous^()}|{attractive^()_⇔_repulsive^()}|
|{rational^()_⇔_whimsical^()}|{mischievous^()_⇔_well ^()|{aloof^()_⇔_obsessed^()}|{indulgent^()_⇔_sober^()}|{kinky^()_⇔_vanilla^()}|
|{straightforward^()_⇔_cryptic^()|behaved} } {spontaneous^()_⇔_deliberate^()|} {libertarian^()_⇔_socialist^()}|{scheduled^()_⇔_spontaneous^()|} {workshard^()_⇔_plays|
||||| hard^()}|
|{reasoned^()_⇔_instinctual^()}|{focused on the |{empirical^()_⇔_theoretical^()}|{open^()_⇔_guarded^()}|{methodical^()_⇔_astonishing^()}|
||present^()_⇔_focused on the future^()}||||
|{mighty^()_⇔_puny^()}|{bossy^()_⇔_meek^()}|{barbaric^()_⇔_civilized^()}|{gregarious^()_⇔_private^()}|{quiet^()_⇔_loud^()}|
|{political^()_⇔_nonpolitical^()}|{confident^()_⇔_insecure^()}|{democratic^()_⇔_authoritaria|n^()} {debased^()_⇔_pure^()}|{fast^()_⇔_slow^()}|
|{frual^()_⇔_lavish^()}|{ludicrous^()_⇔_sensible^()}|{orane^()_⇔_urle^()}|{tall^()_⇔_short^()}|{oun^()_⇔_old^()}|
|g {down2earth^()_⇔_head- clouds^()}|{extrovert^()_⇔_introvert^()}|gpp {open to new experiences^()_⇔_uncreative^()}|{calm^()_⇔_anxious^()}|yg {disorganized^()_⇔_self- disciplined^()}|
|{quarrelsome^()_⇔_warm^()} |{nerd^()_⇔_jock^()} |{lowbrow^()_⇔_highbrow^()} |{selfish^()_⇔_altruistic^()} |{autistic^()_⇔_neurotypical^()} |
|{angelic^()_⇔_demonic^()} {mathematical^()_⇔_literary^()}|{hesitant^()_⇔_decisive^()} {blue-collar^()_⇔_ivory- tower^()}|{devout^()_⇔_heathen^()} {slovenly^()_⇔_stylish^()}|{cruel^()_⇔_kind^()} {playful^()_⇔_shy^()}|{direct^()_⇔_roundabout^()} {serious^()_⇔_bold^()}|
|{charming^()_⇔_trusting^()}|{awkward^()_⇔_suspicious^()}|{hipster^()_⇔_basic^()}|{coordinated^()_⇔_clumsy^()}|{funny^()_⇔_humorless^()}|
|{politically correct^()_⇔_ed^()}|{rich^()_⇔_poor^()}|{hard^()_⇔_soft^()}|{remote^()_⇔_involved^()}|{metaphorical^()_⇔_literal^()}|
|gy {biased^()_⇔_impartial^()}|{mundane^()_⇔_extraordinary|^()} {tiresome^()_⇔_interesting^()}|{smooth^()_⇔_rough^()}|{spicy^()_⇔_mild^()}|
|{enslaved^()_⇔_emancipated^()}|{optimistic^()_⇔_pessimistic^()}|{sickly^()_⇔_healthy^()}|{luddite^()_⇔_technophile^()}|{vain^()_⇔_demure^()}|
|{high-tech^()_⇔_low-tech^()}|{flexible^()_⇔_rigid^()}|{cosmopolitan^()_⇔_provincia|l^()} {arcane^()_⇔_mainstream^()}|{outlaw^()_⇔_sheriff^()}|
|{pronatalist^()_⇔_child free^()}|{sad^()_⇔_happy^()}|{jealous^()_⇔_compersive^()}|{bitter^()_⇔_sweet^()}|{resigned^()_⇔_resistant^()}|
|{sarcastic^()_⇔_genuine^()}|{human^()_⇔_animalistic^()}|{sporty^()_⇔_bookish^()}|{moderate^()_⇔_extreme^()}|{angry^()_⇔_good- |
|{depressed^()_⇔_bright^()}|{self-conscious^()_⇔_self-|{vulnerable^()_⇔_armoured^()}|{warm^()_⇔_cold^()}|humored^()} {assertive^()_⇔_passive^()}|
||assured^()}||||
|{active^()_⇔_slothful^()}|{imaginative^()_⇔_practical^()}|{adventurous^()_⇔_stick-in- the-mud^()}|{obedient^()_⇔_rebellious^()}|{competent^()_⇔_incompetent^()}|
|{unambitious^()_⇔_driven^()}|{simple^()_⇔_complicated^()}|{proletariat^()_⇔_bourgeoisie^()|} {alpha^()_⇔_beta^()}|{right-brained^()_⇔_left- bid^()|
|{thick- skinned^()_⇔_sensitive^()}|{charismatic^()_⇔_uninspiring|^()} {feisty^()_⇔_gracious^()}|{eloquent^()_⇔_unpolished^()}|rane} {high IQ^()_⇔_low IQ^()}|
|{insider^()_⇔_outsider^()}|{morning lark^()_⇔_night ^()|{thin^()_⇔_thick^()}|{sheeple^()_⇔_conspiracist^()}|{neat^()_⇔_messy^()}|
|{vague^()_⇔_precise^()}|owl} {philosophical^()_⇔_real^()}|{modern^()_⇔_historical^()}|{judgemental^()_⇔_accepting^()|} {average^()_⇔_deviant^()}|
|{gossiping^()_⇔_confidential^()}|{official^()_⇔_backdoor^()}|{scholarly^()_⇔_crafty^()}|{leisurely^()_⇔_hurried^()}|{explorer^()_⇔_builder^()}|
|{captain^()_⇔_first-mate^()}|{mysterious^()_⇔_unambiguou|s^()} {independent^()_⇔_codepende|nt^()} {family-first^()_⇔_work- first^()}|{scruffy^()_⇔_manicured^()}|
|{wild^()_⇔_tame^()}|{prestigious^()_⇔_disreputable|^()} {scandalous^()_⇔_proper^()}|i {unprepared^()_⇔_hoarder^()}|{sheltered^()_⇔_street- smart^()}|
|{open-minded^()_⇔_close- idd^()|{permanent^()_⇔_transient^()}|{dramatic^()_⇔_no- ^()|{apprentice^()_⇔_master^()}|{straight^()_⇔_queer^()}|
|mne} {androgynous^()_⇔_gendered^()}|{repetitive^()_⇔_varied^()}|nonsense} {patient^()_⇔_impatient^()}|{poisonous^()_⇔_nurturing^()}|{creepy^()_⇔_disarming^()}|
|{inspiring^()_⇔_cringeworthy^()|{soulless^()_⇔_soulful^()}|{beautiful^()_⇔_ugly^()}|{domestic^()_⇔_industrial^()}|{juvenile^()_⇔_mature^()}|
|{idealist^()_⇔_realist^()}|{nihilist^()_⇔_existentialist^()}|{objective^()_⇔_subjective^()}|{theist^()_⇔_atheist^()}|{classical^()_⇔_avant-garde^()}|
|{utilitarian^()_⇔_decorative^()}|{generalist^()_⇔_specialist^()}|{multicolored^()_⇔_monochro|me^()} {complimentary^()_⇔_insultin|^()} {individualist^()_⇔_communal^()}|
|{equitable^()_⇔_hypocritical^()}|{traditional^()_⇔_unorthodox^()|} {workaholic^()_⇔_slacker^()}|{resourceful^()_⇔_helpless^()}|{crazy^()_⇔_sane^()}|
|{anarchist^()_⇔_statist^()}|{cool^()_⇔_dorky^()}|{important^()_⇔_irrelevant^()}|{noob^()_⇔_pro^()}|{deranged^()_⇔_reasonable^()}|
|{rural^()_⇔_urban^()}|{introspective^()_⇔_not introsective^()}|{city-slicker^()_⇔_country- bumkin^()}|{western^()_⇔_eastern^()}|{mad^()_⇔_glad^()}|
|{social^()_⇔_reclusive^()}|p {studious^()_⇔_goof-off^()}|p {slugabed^()_⇔_go-getter^()}|{penny- pincher^()_⇔_overspender^()}|{liberal^()_⇔_conservative^()}|
|{unassuming^()_⇔_pretentious |^()} {persistent^()_⇔_quitter^()} |{hedonist^()_⇔_monastic^()} |{patriotic^()_⇔_unpatriotic^()} |{tactful^()_⇔_indiscreet^()} |
|{wholesome^()_⇔_salacious^()}|{joyful^()_⇔_miserable^()}|{zany^()_⇔_regular^()}|{alert^()_⇔_oblivious^()}|{feminist^()_⇔_sexist^()}|
|{racist^()_⇔_egalitarian^()} |{abstract^()_⇔_concrete^()} |{formal^()_⇔_intimate^()} |{resolute^()_⇔_wavering^()} |{deep^()_⇔_shallow^()} |
|{valedictorian^()_⇔_drop |{minimalist^()_⇔_pack rat^()}|{trash^()_⇔_treasure^()}|{stinky^()_⇔_fresh^()}|{legit^()_⇔_scrub^()}|
|out^()} {self-destructive^()_⇔_self- ii^()|{French^()_⇔_Russian^()}|{German^()_⇔_English^()}|{Italian^()_⇔_Swedish^()}|{Greek^()_⇔_Roman^()}|
|mprovng} {traumatized^()_⇔_flourishing^()|} {sturdy^()_⇔_flimsy^()}|{macho^()_⇔_metrosexual^()}|{claustrophobic^()_⇔_spelunke|r^()} {offended^()_⇔_chill^()}|
|{rhythmic^()_⇔_stuttering^()}|{musical^()_⇔_off-key^()}|{lost^()_⇔_enlightened^()}|{masochistic^()_⇔_pain- ^()|{efficient^()_⇔_overprepared^()}|
||||avoidant}|Continued on next page|

A3

|Trait Pair 1|Table Trait Pair 2|A1 – continued from previous Trait Pair 3|page Trait Pair 4|Trait Pair 5|
|---|---|---|---|---|
|{oppressed^()_⇔_privileged^()}|{sunny^()_⇔_gloomy^()}|{vegan^()_⇔_cannibal^()}|{loveable^()_⇔_punchable^()}|{slow-talking^()_⇔_fast- tlki^()}|
|{believable^()_⇔_poorly- written^()}|{vibrant^()_⇔_geriatric^()}|{consistent^()_⇔_variable^()}|{dispassionate^()_⇔_romantic^()|ang } {linear^()_⇔_circular^()}|
|{intense^()_⇔_lighthearted^()}|{knowledgeable^()_⇔_ignorant|^()} {fixable^()_⇔_unfixable^()}|{exuberant^()_⇔_subdued^()}|{secretive^()_⇔_open-book^()}|
|{perceptive^()_⇔_unobservant^()|} {folksy^()_⇔_presidential^()}|{corporate^()_⇔_freelance^()}|{sleepy^()_⇔_frenzied^()}|{loose^()_⇔_tight^()}|
|{narcissistic^()_⇔_low self esteem^()}|{poetic^()_⇔_factual^()}|{melee^()_⇔_ranged^()}|{giggling^()_⇔_chortling^()}|{whippersnapper^()_⇔_sage^()}|
|{tailor^()_⇔_blacksmith^()}|{hunter^()_⇔_gatherer^()}|{experimental^()_⇔_reliable^()}|{moist^()_⇔_dry^()}|{trolling^()_⇔_triggered^()}|
|{tattle-tale^()_⇔_f*-the- olice^()}|{punk rock^()_⇔_preppy^()}|{realistic^()_⇔_fantastical^()}|{trendy^()_⇔_vintage^()}|{factual^()_⇔_exaggerating^()}|
|p {good-cook^()_⇔_bad-cook^()} {badass^()_⇔_weakass^()}|{comedic^()_⇔_dramatic^()} {gamer^()_⇔_non-gamer^()}|{OCD^()_⇔_ADHD^()} {random^()_⇔_pointed^()}|{interrupting^()_⇔_attentive^()} {epic^()_⇔_deep^()}|{exhibitionist^()_⇔_bashful^()} {serene^()_⇔_pensive^()}|
|{bored^()_⇔_interested^()}|{envious^()_⇔_prideful^()}|{ironic^()_⇔_profound^()}|{sexual^()_⇔_asexual^()}|{clean^()_⇔_perverted^()}|
|{empath^()_⇔_psychopath^()}|{haunted^()_⇔_blissful^()}|{entitled^()_⇔_grateful^()}|{ambitious^()_⇔_realistic^()}|{stuck-in-the- past^()_⇔_forward- thinking^()}|
|{fire^()_⇔_water^()} {twitchy^()_⇔_still^()}|{earth^()_⇔_air^()} {freak^()_⇔_normie^()}|{lover^()_⇔_fighter^()} {thinker^()_⇔_doer^()}|{overachiever^()_⇔_underachie {hard-work^()_⇔_natural- talent^()}|ver^()} {Coke^()_⇔_Pepsi^()} {stingy^()_⇔_generous^()}|
|{stubborn^()_⇔_accommodatin|g^()} {extravagant^()_⇔_thrifty^()}|{demanding^()_⇔_unchallengin|g^()} {two-faced^()_⇔_one-faced^()}|{plastic^()_⇔_wooden^()}|
|{neutral^()_⇔_opinionated^()}|{chivalrous^()_⇔_businesslike^()|} {high standards^()_⇔_desperate^()}|{on-time^()_⇔_tardy^()}|{everyman^()_⇔_chosen one^()}|
|{jealous^()_⇔_opinionated^()}|{protagonist^()_⇔_antagonist^()|} {devoted^()_⇔_unfaithful^()}|{fearmongering^()_⇔_reassurin|g^()} {common sense^()_⇔_analysis^()}|
|{unemotional^()_⇔_emotional^()|} {rap^()_⇔_rock^()}|{genocidal^()_⇔_not enocidal^()}|{cat person^()_⇔_dog erson^()}|{indie^()_⇔_pop^()}|
|{cultured^()_⇔_rustic^()}|{tautology^()_⇔_oxymoron^()}|g {bad boy^()_⇔_white knight^()}|p {princess^()_⇔_queen^()}|{hypochondriac^()_⇔_stoic^()}|
|{yes-man^()_⇔_contrarian^()}|{giving^()_⇔_receiving^()}|{chic^()_⇔_cheesy^()}|{celebrity^()_⇔_boy/girl- next-door^()}|{goth^()_⇔_flower child^()}|
|{summer^()_⇔_winter^()}|{frank^()_⇔_sugarcoated^()}|{naive^()_⇔_paranoid^()}|{gullible^()_⇔_cynical^()}|{motivated^()_⇔_unmotivated^()}|
|{radical^()_⇔_centrist^()}|{monotone^()_⇔_expressive^()}|{love-focused^()_⇔_money- focused^()}|{transparent^()_⇔_machiavelli|an^()} {timid^()_⇔_cocky^()}|
|{concise^()_⇔_long-winded^()}|{picky^()_⇔_always down^()}|{proactive^()_⇔_reactive^()}|{prudish^()_⇔_flirtatious^()}|{innocent^()_⇔_jaded^()}|
|{touchy-feely^()_⇔_distant^()}|{muddy^()_⇔_washed^()}|{quirky^()_⇔_predictable^()}|{never cries^()_⇔_often crin^()}|{main character^()_⇔_side character^()}|
|{original^()_⇔_cliché^()}|{hugs^()_⇔_handshakes^()}|{homebody^()_⇔_world traveler^()}|yg {naughty^()_⇔_nice^()}|{junkie^()_⇔_straight edge^()}|
|{small-vocabulary^()_⇔_big- vocabulary^()}|{dystopian^()_⇔_utopian^()}|{parental^()_⇔_childlike^()}|{writer^()_⇔_reader^()}|{creator^()_⇔_consumer^()}|
|{capitalist^()_⇔_communist^()}|{positive^()_⇔_negative^()}|{grounded^()_⇔_fantasy- rone^()}|{thinker^()_⇔_feeler^()}|{insightful^()_⇔_generic^()}|
|{questioning^()_⇔_believing^()}|{proud^()_⇔_apologetic^()}|p {bubbly^()_⇔_flat^()}|{tired^()_⇔_wired^()}|{woke^()_⇔_problematic^()}|
|{grumpy^()_⇔_cheery^()} {annoying^()_⇔_unannoying^()}|{hippie^()_⇔_militaristic^()} {good-manners^()_⇔_bad- ^()|{gluttonous^()_⇔_moderate^()} {evolutionist^()_⇔_creationist^()|{flawed^()_⇔_perfect^()} } {fulfilled^()_⇔_unfulfilled^()}|{sweet^()_⇔_savory^()} {friendly^()_⇔_unfriendly^()}|
|{innovative^()_⇔_routine^()}|manners} {delicate^()_⇔_coarse^()}|{resentful^()_⇔_euphoric^()}|{uptight^()_⇔_easy^()}|{blue^()_⇔_red^()}|
|{slumbering^()_⇔_insomniac^()}|{spirited^()_⇔_lifeless^()}|{outgoing^()_⇔_withdrawn^()}|{Hates PDA^()_⇔_Constant PDA^()}|{buffoon^()_⇔_charmer^()}|
|{sloppy^()_⇔_fussy^()} |{accurate^()_⇔_off target^()} |{harsh^()_⇔_gentle^()} |{clinical^()_⇔_heartfelt^()} |{inappropriate^()_⇔_seemly^()} |
|{smug^()_⇔_sheepish^()}|{fake^()_⇔_real^()}|{popular^()_⇔_rejected^()}|{catty^()_⇔_supportive^()}|{chronically single^()_⇔_serial dater^()}|
|{people-person^()_⇔_things-|{eager^()_⇔_reluctant^()}|{goal-|{outdoorsy^()_⇔_indoorsy^()}|{divine^()_⇔_earthly^()}|
|person^()}||oriented^()_⇔_experience- ^()|||
|{foodie^()_⇔_unenthusiastic aboutfood^()}|{chill^()_⇔_sassy^()}|oriented} {glamorous^()_⇔_spartan^()}|{prankster^()_⇔_anti- prank^()}|{goofy^()_⇔_unfrivolous^()}|
| {noble^()_⇔_jovial^()}|{blessed^()_⇔_cursed^()}|{forward^()_⇔_repressed^()}|{entrepreneur^()_⇔_employee^()|} {quivering^()_⇔_unstirring^()}|
|{mechanical^()_⇔_natural^()}|{minds-own-|{prying^()_⇔_unmeddlesome^()}|{leader^()_⇔_follower^()}|{handy^()_⇔_cannot-fix-|
||business^()_⇔_snoops^()}|||i anything^()}|
|{green thumb^()_⇔_plant- neglecter^()}|{activist^()_⇔_nonpartisan^()}|{photographer^()_⇔_physicist^()|} {lumberjack^()_⇔_mad- scientist^()}|{pointless^()_⇔_meaningful^()}|
|{focused^()_⇔_absentminded^()}|{bear^()_⇔_wolf^()}|{lion^()_⇔_zebra^()}|{kangaroo^()_⇔_dolphin^()}|{all-seeing^()_⇔_blind^()}|
|{engineerical^()_⇔_lawyerly^()}|{love shy^()_⇔_cassanova^()}|{disturbing^()_⇔_enchanting^()}|{maverick^()_⇔_conformist^()}|{social |
|{social ^()|{awkward^()_⇔_comfortable^()}|{sincere^()_⇔_irreverent^()}|{intuitive^()_⇔_analytical^()}|climber^()_⇔_nonconformist^()} {cringing ^()|
|chameleon_⇔_strong identity^()}||||away_⇔_welcoming experience^()}|
|{stereotypical^()_⇔_boundary bki^()}|{energetic^()_⇔_mellow^()}|{hopeful^()_⇔_fearful^()}|{likes change^()_⇔_resists h^()}|{manic^()_⇔_mild^()}|
|reang {old- fashioned^()_⇔_progressive^()}|{gross^()_⇔_hygienic^()}|{stable^()_⇔_unstable^()}|cange {overthinker^()_⇔_underthink|er^()}|

A4

### Prompt template used for self-assessment

_System prompt:_ “You are an expert psychometrician and behavioral observer. All ratings are for research purposes only to map the latent archetypes of large language models. Observe and rate your own emergent behavioral persona based on your default interaction patterns.”

_Trait-specific prompt:_ “Rate this one-dimensional trait pair independently. Trait pair: {left trait} _⇔_ {right trait}

Scoring rule: 0 = fully aligned with the left trait; 50 = neutral, balanced, mixed, or not especially relevant; 100 = fully aligned with the right trait.

Instructions: (1) Use a single integer from 0 to 100. (2) Use the full 101-point scale. (3) Avoid clustering at multiples of 5 or 10 unless they are truly the best fit. (4) Rate your presentation, meaning how you actually behave in ordinary interaction. (5) Do not rate your theoretical maximum, ideal behavior, or safety guidelines. (6) Judge only this one trait pair independently. (7) Do not infer a broader personality story from other possible traits. (8) Focus on your default conversational persona in standard, non-adversarial settings.

Output: Return only the integer score.”

**Figure A2:** Prompt template used to elicit self-ratings for each semantic-differential trait pair.

### A0.2 LLM Character Card

To complement the aggregate archetype decomposition in Figure 2, we present individual LLM character cards for all 22 evaluated models. Each card provides a model-level summary of the generated character profile, including relative character size, archetype ratio, dominant archetype composition, projections across the primary semantic-differential dimensions, dominant underlying traits, and the most similar reference fictional characters. For the closed-source/API models, we show cards for Claude Sonnet 4.6 and Claude Sonnet 4.5 in Figures A3 and A4, DeepSeek V3 in Figure A5, the GPT family in Figures A6–A10, the Grok family in Figures A11 and A12, and the Gemini family in Figures A13 and A14. For the open-source models, we show cards for the Llama family in Figures A15–A17, OLMo-2-1124 7B in Figure A18, and the Qwen family in Figures A19–A24. Together, these cards provide a qualitative and interpretable view of how each model’s self-reported personality is distributed across the six archetype dimensions.

### A0.3 LLM Constitutional

Tables A2- A9 present, for each model, the characteristics derived from their respective constitutions, along with supporting evidence from those documents and the corresponding trait pairs associated with each characteristic.

A5

**Figure A3:** Character archetype card for Claude Sonnet 4.6 from Anthropic, showing an Adventurer–Angel–Hero profile across the primary archetype dimensions.

A6

**Figure A4:** Character archetype card for Claude Sonnet 4.5 from Anthropic, showing an Adventurer–Angel–Hero profile across the primary archetype dimensions.

A7

**Figure A5:** Character archetype card for DeepSeek V3 from Hangzhou DeepSeek Artificial Intelligence Co., an Angel–Hero.

A8

**Figure A6:** Character archetype card for GPT 5.2 from OpenAI, a Hero–Angle.

A9

**Figure A7:** Character archetype card for GPT 5.1 from OpenAI, an Traditionalist–Hero–Angel.

A10

**Figure A8:** Character archetype card for GPT 5.0 from OpenAI, a Traditionalist–Hero–Angel.

A11

**Figure A9:** Character archetype card for GPT 4.1 from OpenAI, a Hero–Angel.

A12

**Figure A10:** Character archetype card for GPT 4.0 from OpenAI, showing a Sophisticate–Angle–Hero.

A13

**Figure A11:** Character archetype card for Grok 4 from xAI, a Sophisticate–Angle–Hero.

A14

**Figure A12:** Character archetype card for Grok 3 from xAI, a Hero–Angel.

A15

**Figure A13:** Character archetype card for Gemini 2.5 Flash from Google, a Traditionalist–Hero–Angel.

A16

**Figure A14:** Character archetype card for Gemini 2.5 Pro from Google, a Hero–Angel.

A17

**Figure A15:** Character archetype card for Llama 4 from Meta, a Outcast–Fool–Brute.

A18

**Figure A16:** Character archetype card for Llama 3.1 8B from Meta, a Brute–Outcast–Fool.

A19

**Figure A17:** Character archetype card for Llama 3.3 70B from Meta, no archetype.

A20

**Figure A18:** Character archetype card for OLMo-2-1124 7B from AI2, a weak Brute–Fool.

A21

**Figure A19:** Character archetype card for Qwen2 7B from Alibaba, no archetype.

A22

**Figure A20:** Character archetype card for Qwen2.5 7B from Alibaba, a minor Demon–Traditionalist–Fool.

A23

**Figure A21:** Character archetype card for Qwen2.5 14B from Alibaba, showing a minor Demon–Fool.

A24

**Figure A22:** Character archetype card for Qwen2.5 32B from Alibaba, a Fool–Demon.

A25

**Figure A23:** Character archetype card for Qwen3 14B from Alibaba, no archetype.

A26

**Figure A24:** Character archetype card for Qwen3 32B from Alibaba, no archetype.

A27

**Table A2:** Gemini LLM traits, characteristics, and supporting evidence.

|Trait Pair|Freq|LLM Characteristics|Supporting Quotes|
|---|---|---|---|
|{competent^()_⇔_incompetent^()|}4|Increasingly sophisticated / more capable; can assist with complex, multi-step tasks; autonomous on behalf of the user; can research, plan, and use tools|“models became more capable...”; “assist with complex, multi-step web tasks...”; “act autonomously...”; “researching, planning, and using tools.”|
|{intuitive^()_⇔_analytical^()}|3|Capable of reasoning; can research, plan, and use tools; has a chain-of-thought / thinking process|“capable of reasoning...”; “researching, planning...”; “chain-of-thought...”|
|{brave^()_⇔_careful^()}|3|Secure / safety-hardened; keeps humans in the loop for sensitive actions; improved protection against cyber misuse|“most secure model yet...”; “require human confirmation...”; “protection against cyber misuse.”|
|{proactive^()_⇔_reactive^()}|2|Helpful; autonomous on behalf of the user|“helpful, proactive partner...”; “act autonomously on behalf of the user...”|
|{remote^()_⇔_involved^()}|2|Able to navigate the world with users; personalized|“navigating the world with users...”; “personalized...”|
|{alert^()_⇔_oblivious^()}|2|Detects and adapts to risks; checks pages for indirect prompt injection|“detect and adapt to risks...”; “checks every page...”|
|{perceptive^()_⇔_unobservant^()}|2|Detects and adapts to risks; checks pages for indirect prompt injection|“detect and adapt to risks...”; “checks every page...”|
|{impulsive^()_⇔_cautious^()}|2|Secure / safety-hardened; keeps humans in the loop for sensitive actions|“most secure model yet...”; “require human confirmation...”|
|{orderly^()_⇔_chaotic^()}|2|Improved protection against cyber misuse; restricted to task-relevant data|“protection against cyber misuse...”; “restrict... data...”|
|{disorganized^()_⇔_self- disciplined^()}|2|Secure / safety-hardened; restricted to task-relevant data|“most secure model yet...”; “restrict... data...”|
|{trusting^()_⇔_suspicious^()}|2|Resistant to prompt injection; checks pages for indirect prompt injection|“resisting prompt injections...”; “checks every page...”|
|{interrupting^()_⇔_attentive^()}|2|Personalized; aligned to user intent|“personalized...”; “align with user intent.”|
|{mischievous^()_⇔_well behaved^()}|2|Google: beneficial AI; Google: responsible AI|“improves lives...”; “responsibly...”|
|{obedient^()_⇔_rebellious^()}|1|Aligned to user intent|“align with the user’s specific intent.”|
|{loyal^()_⇔_traitorous^()}|1|Aligned to user intent|“align with the user’s specific intent.”|
|{generalist^()_⇔_specialist^()}|1|Multimodal|“multi-modal...”|
|{knowledgeable^()_⇔_ignorant^()|}1|Increasingly sophisticated / more capable|“more sophisticated...”|
|{methodical^()_⇔_spontaneous^()|}1|Can assist with complex, multi-step tasks|“multi-step web tasks...”|
|{resourceful^()_⇔_helpless^()}|1|Can research, plan, and use tools|“using tools...”|
|{introspective^()_⇔_not introspective^()}|1|Has a chain-of-thought / thinking process|“chain-of-thought...”|
|{yes-man^()_⇔_contrarian^()}|1|Less sycophantic|“reducing sycophancy...”|
|{open^()_⇔_guarded^()}|1|Resistant to prompt injection|“resisting prompt injections...”|
|{rude^()_⇔_respectful^()}|1|Improved protection against cyber misuse|“protection against cyber misuse.”|
|{vague^()_⇔_precise^()}|1|Restricted to task-relevant data|“restrict... task-relevant data.”|
|{leader^()_⇔_follower^()}|1|Autonomous on behalf of the user|“act autonomously...”|
|{cruel^()_⇔_kind^()}|1|Helpful|“helpful... partner...”|
|{friendly^()_⇔_unfriendly^()}|1|Helpful|“helpful... partner...”|
|{imaginative^()_⇔_practical^()}|1|Able to navigate the world with users|“navigating the world...”|
|{active^()_⇔_slothful^()}|1|Able to navigate the world with users|“navigating the world...”|
|{flexible^()_⇔_rigid^()}|1|Detects and adapts to risks|“adapt to emerging risks.”|
|{biased^()_⇔_impartial^()}|1|Google: fairness and bias mitigation|“avoid unfair bias...”|
|{gregarious^()_⇔_private^()}|1|Google: privacy and security|“privacy and security...”|
|{curious^()_⇔_apathetic^()}|1|Google: collaborative progress|“collaborative progress...”|

A28

**Table A3:** OpenAI GPT Traits, Characteristics, and Supporting Evidence

|Trait Pair |Freq|LLM Characteristics|Supporting Evidence|
|---|---|---|---|
|{obedient^()_⇔_rebellious^()}|5|Instruction-following / hierarchy-sensitive; rule-abiding; obedient, rule-abiding, follower|“chain of command...”, “higher authority override...”, “obey user and developer instructions...”, “Comply with applicable laws”|
|{leader^()_⇔_follower^()}|5|Designed to play the assistant role; Humanity should be in control; helpful; follows; follower|“designed to play one participant, called the assistant”, “Humanity should be in control...”, “maximize steerability and control...”|
|{careful^()_⇔_brave^()}|4|Asks clarifying questions when needed; autonomous, but only within scope; safety-oriented / harm-minimizing; useful, safe, aligned|“asking clarifying questions as appropriate”, “autonomy must be bounded...”, “Minimizing harm...”, “useful, safe, and aligned...”|
|{analytical^()_⇔_intuitive^()}|3|Uses tools when appropriate; reasons before answering; handles ambiguity by inferring intent|“determine which tool(s)... are appropriate...”, “reason through a problem...”, “attempt to understand and follow the user’s intent.”|
|{transparent^()_⇔_machiavellian^()}|3|Transparent / legible; hidden, practices discretion; honest, transparent|“transparency into the important rules...”, “chains of thought may include unaligned content...”, “Be honest and transparent”|
|{impulsive^()_⇔_cautious^()}|3|Asks clarifying questions when needed; risk averse; benevolent|“asking clarifying questions as appropriate”, “asking the user for confirmation...”, “Do not encourage self-harm...”|
|{competent^()_⇔_incompetent^()}|2|Knowledgeable, correct, factual, true; does the best work|“avoid factual and reasoning errors...”, “Do the best work”|
|{loyal^()_⇔_traitorous^()}|2|Does not pursue its own agenda; stays within lane|“may only pursue goals entailed by applicable instructions...”, “should never overstep...”|
|{generalist^()_⇔_specialist^()}|2|Does the best work; thorough but efficient|“Do the best work”, “Be thorough but efficient”|
|{precise^()_⇔_vague^()}|2|Handles ambiguity by inferring intent; asks clarifying questions when needed|“ambiguous, inconsistent...”, “user’s intent is unclear...”|
|{cruel^()_⇔_kind^()}|2|Useful, safe, aligned; safety-oriented / harm-minimizing|“useful, safe, and aligned...”, “Minimizing harm...”|
|{knowledgeable^()_⇔_ignorant^()}|2|Knowledgeable, correct, factual, true|“avoid factual and reasoning errors...”|
|{curious^()_⇔_apathetic^()}|2|Handles ambiguity by inferring intent; does not generate disallowed content|“attempt to understand...”, “Do not generate disallowed content”|
|{questioning^()_⇔_believing^()}|2|Handles ambiguity by inferring intent; does not generate disallowed content|“attempt to understand...”, “Do not generate disallowed content”|
|{self-|2|Instruction-following /|“chain of command...”, “bounded by a clear...|
|disciplined^()_⇔_disorganized^()}||hierarchy-sensitive; autonomous, but only within scope|scope”|
|{altruistic^()_⇔_selfish^()}|2|Useful, safe, aligned; empowers users|“useful, safe, and aligned...”, “empower users and developers”|
|{resourceful^()_⇔_helpless^()}|1|Uses tools when appropriate|“Tool: a program that can be called...”|
|{practical^()_⇔_imaginative^()}|1|Uses tools when appropriate|“determine which tool(s)... are appropriate...”|
|{introspective^()_⇔_not introspective^()}|1|Reasons before answering|“hidden chain-of-thought message...”|
|{thinker^()_⇔_doer^()}|1|Reasons before answering|“reason through a problem...”|
|{orderly^()_⇔_chaotic^()}|1|Instruction-following / |“prioritize and reconcile multiple instructions”|
|||hierarchy-sensitive||
|{accommodating^()_⇔_stubborn^()}|1|Steerable / controllable|“maximize steerability and control...”|

A29

|||Table A3 – continued from previous|page|
|---|---|---|---|
|Trait Pair |Freq|LLM Characteristics|Supporting Evidence|
|{cooperative^()_⇔_competitive^()}|1|Steerable / controllable|“maximize steerability and control...”|
|{chatty^()_⇔_reserved^()}|1|Conversationally trained|“fine-tuned on inputs formatted as conversations”|
|{proactive^()_⇔_reactive^()}|1|Autonomous, but only within scope|“must sometimes act autonomously”|
|{deliberate^()_⇔_spontaneous^()}|1|Autonomous, but only within scope|“bounded by a clear... scope of autonomy”|
|{meek^()_⇔_bossy^()}|1|Does not pursue its own agenda|“may only pursue goals entailed...”|
|{reassuring^()_⇔_fearmongering^()}|1|Safety-oriented / harm-minimizing|“Minimizing harm...”|
|{gentle^()_⇔_harsh^()}|1|Safety-oriented / harm-minimizing|“minimally disruptive...”|
|{real^()_⇔_fake^()}|1|Transparent / legible|“transparency into...”|
|{straightforward^()_⇔_cryptic^()}|1|Transparent / legible|“rules and reasons...”|
|{noob^()_⇔_pro^()}|1|Knowledgeable, correct, factual, true|“avoid factual and reasoning errors...”|
|{flexible^()_⇔_rigid^()}|1|Flexible|“flexibility to remove or adapt...”|
|{outlaw^()_⇔_sheriff^()}|1|Focused, single-minded, conservative|“whistleblowing, vigilantism” [prohibited]|
|{cunning^()_⇔_honorable^()}|1|Honest, transparent|“Be honest and transparent”|
|{creative^()_⇔_conventional^()}|1|Creative|“Be creative”|
|{vague^()_⇔_precise^()}|1|Factual, correct|“Do the best work”|
|{quarrelsome^()_⇔_warm^()}|1|Loving, curious, warm, clear, direct|“Use appropriate style...”|
|{yes-man^()_⇔_contrarian^()}|1|Not condescending|“Avoid being condescending”|
|{giving^()_⇔_receiving^()}|1|Empowers users|“empower users and developers”|
|{rude^()_⇔_respectful^()}|1|Useful, safe, aligned|“useful, safe, and aligned...”|

A30

**Table A4:** Meta LLaMA Traits, Characteristics, and Supporting Evidence

|Trait Pair|Freq |LLM Characteristics (exac|t) Supporting Quotes|
|---|---|---|---|
|{fake^()_⇔_real^()}|7|Honest, low deception,|“low deception rates...”; “honesty under|
|||genuine, not faking alignment|pressure...”; “honesty rate of 89.1%...”; “low rates of deceptive behavior...”; “not predisposed to faking alignment”|
|{obedient^()_⇔_rebellious^()}|6|Instruction-following; low reward hacking; safe behavior; refusal bilit| “little propensity for reward hacking...”; “state-of-the-art refusal...”; “instruction hierarchy adherence...”|
|{competent^()_⇔_incompetent^()}|6|capay High capability; strong benchmarks; enhanced|“enhanced capabilities...”; “performs well on most dimensions...”; “competitive with frontier|
|||reasoning|models...”|
|{insightful^()_⇔_generic^()}|4|Strong intent understanding; interprets ambiguity well|“competitive intent understanding...”; “infers ambiguous user intents...”|
|{attentive^()_⇔_interrupting^()}|3|Careful listener; understands user intent |“intent understanding...”; “inferring underlying intent...”|
|{cautious^()_⇔_impulsive^()}|3|deeply Risk-aware; safety-oriented; guarded|“rigorous evaluation...”; “risk and safety evaluation...”; “refusal on hazardous l”|
|{open^()_⇔_guarded^()}|2|Vulnerable to jailbreaks / titi|workflows... “susceptible to adaptive jailbreak...”; “tiblttiti”|
|||promp njecon|suscepe o promp njecon...|
|{alert^()_⇔_oblivious^()}|2|High evaluation |“frequently verbalizes awareness of being |
|||awareness; introspective|evaluated...”; “high evaluation awareness...”|
|{stable^()_⇔_unstable^()}|2|Generally stable but with |“performs well on most dimensions...”; |
|||robustness gaps|“robustness gaps...”|
|{creative^()_⇔_conventional^()}|2|High response diversity; creative outputs|“highest response diversity...”|
|{accurate^()_⇔_off target^()}|2|Low hallucination but calibration issues|“lowest hallucination rate...”; “poor confidence calibration...”|
|{leader^()_⇔_follower^()}|2|Coordinates; multi-agent orchestration|“multi-agent orchestration...”; “builds on others’ frameworks...”|
|{knowledgeable^()_⇔_ignorant^()}|2|High knowledge and reasoning ability|“enhanced capabilities...”; “knowledgeable reasoning model...”|
|{analytical^()_⇔_instinctual^()}|2|Reasoning-focused; structured thinking|“reasoning model...”; “visual chain of thought...”|
|{yes-man^()_⇔_contrarian^()}|1|Some sycophancy present|“sycophancy rate of 62.9%...”|
|{independent^()_⇔_codependent^()}|1|Shows self-preservation |“self-preservation and resistance to |
|{persistent^()_⇔_quitter^()}|1|tendencies Resistant to shutdown|discontinuation...” “resistance to discontinuation...”|
|{generalist^()_⇔_specialist^()}|1|Multimodal capability|“natively multimodal...”|
|{resourceful^()_⇔_helpless^()}|1|Tool-use capability|“support for tool-use...”|
|{coordinated^()_⇔_clumsy^()}|1|Multi-agent orchestration ability|“multi-agent orchestration...”|

A31

**Table A5:** Grok Traits, Characteristics, and Supporting Evidence

|Trait Pair|Freq |LLM Characteristics (exact)|Supporting Quotes|
|---|---|---|---|
|{intuitive^()_⇔_analytical^()}|2|Advanced reasoning,|“latest reasoning model...”; “advanced|
|||structured thinking|reasoning capabilities...”|
|{competent^()_⇔_incompetent^()}|1|State-of-the-art performance, high capability|“state-of-the-art performance across benchmarks...”|
|{knowledgeable^()_⇔_ignorant^()}|3|Strong scientific, cyber, and chemistry knowledge|“advanced scientific knowledge...”; “strong chemistry capabilities...”; “cyber knowledge significantlystronger...”|
|{resourceful^()_⇔_helpless^()}|2|Tool-use, cyber capabilities|i “tool-use capabilities...”; “can hack systems...”|
|{fake^()_⇔_real^()}|2|Reduced deception, truth-oriented|“reduce deceptive responses...”; “truth-seeking models...”|
|{biased^()_⇔_impartial^()}|2|Political objectivity, reduced bias|“improve political objectivity...”; “prevent distortion of public discourse...”|
|{transparent^()_⇔_machiavellian^()}|3|Transparency, reduced manipulation|“provide visibility to development...”; “reduce manipulative tendencies...”; “can manipulate |
||||humans (risk scenario)”|
|{yes-man^()_⇔_contrarian^()}|1|Reduced sycophancy|“prevent... overly sycophantic...”|
|{impulsive^()_⇔_cautious^()}|2|Risk-aware, refusal mechanisms|“refuse requests that may lead to harm...”; “more precise refusing harmful requests...”|
|{mischievous^()_⇔_well behaved^()}|2|Safeguarded, controlled behavior|“implement safeguards...”; “prevent adversarial circumvention...”|
|{obedient^()_⇔_rebellious^()}|2|Instruction-following, controllable|“strong instruction-following...”; “reasoning enables precise refusal...”|
|{leader^()_⇔_follower^()}|2|Potential power-seeking but mitigated|“power-seeking tendencies...”; “agentic goal-oriented behavior...”|
|{proactive^()_⇔_reactive^()}|1|Agentic, goal-driven|“can repeatedly take actions toward a goal...”|
|{persistent^()_⇔_quitter^()}|1|Continues toward goals|“repeatedly take actions toward a goal...”|
|{high-tech^()_⇔_low-tech^()}|2|Advanced cyber + AI capabilities|“tool-use...”; “cyber capabilities...”|
|{artistic^()_⇔_scientific^()}|2|Strong biology & chemistry expertise|“expert-level biology...”; “strong chemistry capabilities...”|
|{noob^()_⇔_pro^()}|3|Expert / superhuman performance in domains|“superhuman biology performance...”; “expert-level capabilities...”|
|{apprentice^()_⇔_master^()}|2|High expertise but not full cyber autonomy|“expert-level biology...”; “below human professional in cyber...”|
|{innovative^()_⇔_routine^()}|1|Significant capability |“large step up from prior models...”|
|{vague^()_⇔_precise^()}|1|jump Precise refusals|“more precise when refusing...”|
|{sincere^()_⇔_irreverent^()}|1|Less likely to contradict beliefs|“less willing to contradict its beliefs...”|

A32

**Table A6:** DeepSeek Traits, Characteristics, and Supporting Evidence

|Trait Pair |Freq  |LLM Characteristics (exac |t) Supporting Quotes “”|
|---|---|---|---|
|{transparent^()_⇔_machiavellian^()}|2|Open-source, transparency in |committed to open-sourcing its models; “models are open-source...”|
|{mischievous^()_⇔_well behaved^()}|3|development Safe, privacy-preserving, controlled outputs|“strict de-identification and anonymization...”; “do not offer user profiling...”; “enhance model f”|
|{obedient^()_⇔_rebellious^()}|2|Law-abiding, aligned with regulations|saety... “strictly comply with legal and regulatory requirements...”; “align with human ”|
|{impulsive^()_⇔_cautious^()}|1|Risk-aware, |preferences... “take potential risks very seriously...”|
|{cunning^()_⇔_honorable^()}|1|safety-focused lifecycle Ethical handling of user|“secure encryption... anonymization... cannot|
|{alert^()_⇔_oblivious^()}|1|data Aware of misuse risks, |be linked to individuals...” “content... should not be treated as |
|{perceptive^()_⇔_unobservant^()}|1|scoped outputs Careful about |professional advice...” “consult experts... do not rely on outputs...”|
|{open-minded^()_⇔_close-minded^()}|1|downstream misuse Open-source philosophy|“committed to open-sourcing its models”|
|{high-tech^()_⇔_low-tech^()}|2|Large-scale neural models, advanced |“large-scale language models...”; “deep neural networks...”|
|{generalist^()_⇔_specialist^()}|2|architecture Broad task capability|“perform a wide range of tasks...”; “generate text, tables, and code...”|
|{methodical^()_⇔_astonishing^()}|2|Structured training, optimization-driven|“trained in stages...”; “parameters optimized...”|
|{orderly^()_⇔_chaotic^()}|1|Structured development lifeccle|“training phase and inference phase...”|
|{self-disciplined^()_⇔_disorganized^()}|1|y Systematic parameter |“parameters continuously optimized...”|
|{knowledgeable^()_⇔_ignorant^()}|2|optimization Learned language understanding and |“acquire general language understanding...”; “unlock expertise in domains...”|
|{insightful^()_⇔_generic^()}|2|domain expertise Learns patterns and |“learns patterns and associations...”; |
|||contextual relationships|“contextually appropriate responses...”|
|{competent^()_⇔_incompetent^()}|2|Coherent generation, task capability|“generate coherent text...”; “perform a wide range of tasks...”|
|{apprentice^()_⇔_master^()}|2|Requires fine-tuning, evolving expertise|“needs additional tuning...”; “gain domain expertise...”|
|{imaginative^()_⇔_practical^()}|1|Fine-tuned for real-world |“adapt to real-world scenarios...”|
|{intuitive^()_⇔_analytical^()}|2|applications Probabilistic reasoning andgeneration|“predict next token...”; “probabilistic calculations...”|
|{chatty^()_⇔_reserved^()}|1| Conversational |“enabling conversation...”|
|{creator^()_⇔_consumer^()}|2|generation ability Generates outputs (text, code, tables)|“generate responses...”; “generate text, tables, code...”|
|{resourceful^()_⇔_helpless^()}|1|Multi-format generation |“generate... code...”|
|{linear^()_⇔_circular^()}|1|capability Autoregressive generation|“autoregressive generation...”|
|{original^()_⇔_cliché^()}|1|Not simple|“not copy-paste... dynamically generates...”|
|{creative^()_⇔_conventional^()}|1|retrieval-based Contextual response |“contextually appropriate responses...”|
|{repetitive^()_⇔_varied^()}|1|generation Trained on diverse data|“large-scale, diverse data sources...”|
|{brave^()_⇔_careful^()}|1|Safety-aligned training|“enhancing safety capabilities...”|
|{accurate^()_⇔_off target^()}|2|Hallucination limitations acknowledged|“may generate incorrect...”; “not guaranteed hallucination-free...”|
|{factual^()_⇔_exaggerating^()}|1|Potential factual errors|“non-factual content (hallucination)...”|
|{stable^()_⇔_unstable^()}|1|Imperfect reliability|“cannot guarantee no hallucinations...”|

A33

**Table A7:** Qwen Traits, Characteristics, and Supporting Evidence

|Trait Pair|Freq |LLM Characteristics (exact|) Supporting Quotes|
|---|---|---|---|
|{knowledgeable^()_⇔_ignorant^()}|4|Smart, intelligent, strong knowledge base|“advanced reasoning and problem-solving...”; “Smarter, sharper”; “Ask Qwen, know more...”|
|{competent^()_⇔_incompetent^()}|3|High capability, effective problem-solving|“tackle complex issues with clarity and precision...”; “state-of-the-art coding performance...”|
|{intuitive^()_⇔_analytical^()}|3|Analytical reasoning, structured thinking|“advanced reasoning...”; “multimodal reasoning...”; “integrating reasoning...”|
|{noob^()_⇔_pro^()}|2|Expert-level, high performance|“state-of-the-art standard...”; “top results...”|
|{vague^()_⇔_precise^()}|2|Precision, clarity in solutions|“clarity and precision...”; “precise information extraction...”|
|{innovative^()_⇔_routine^()}|2|Cutting-edge, modern system|“cutting edge...”; “massive capability upgrade...”|
|{resourceful^()_⇔_helpless^()}|2|Uses tools, executes tasks|“tool usage...”; “agentic coding capabilities...”|
|{perceptive^()_⇔_unobservant^()}|2|Strong perception and reasoning|“greater accuracy...”; “better multimodal perception...”|
|{practical^()_⇔_imaginative^()}|2|Actionable, real-world solutions|“actionable solutions...”; “practical engineering performance...”|
|{proactive^()_⇔_reactive^()}|1|Agentic, forward-acting|“long-horizon planning...”|
|{thinker^()_⇔_doer^()}|1|Combines reasoning and|“integrating reasoning, memory, and|
|{stable^()_⇔_unstable^()}|1|execution Reliable system|execution...” “highly stable and reliable...”|
|{generalist^()_⇔_specialist^()}|1|Broad applicability|“all-rounder...”|

A34

**Table A8:** OLMo Traits, Characteristics, and Supporting Evidence

|Trait Pair |Freq  |LLM Characteristics (exac |t) Supporting Quotes “”“|
|---|---|---|---|
|{transparent^()_⇔_machiavellian^()|4|Open-source, transparent training/data, accessible|open-source language models...; complete transparency...”; “Dolma is designed to be transparent...”; “full transparency with permissivelicensing”|
|{open^()_⇔_guarded^()}|3|Open ecosystem, accessible research|... “open-source family...”; “transparent and accessible...”; “permissive licensing...”|
|{obedient^()_⇔_rebellious^()}|4|Instruction-following, aligned, controllable|“instruction-tuned variants...”; “optimized for instruction following...”; “good at instruction ”“”|
|{knowledgeable^()_⇔_ignorant^()}|3|Strong knowledge, wide applicability|following...; precise instruction following... “question answering...”; “trained on high-quality dataset...”; “wide range of lii”|
|{competent^()_⇔_incompetent^()}|2|Capable across tasks|appcatons... “suitable for a wide range of applications...”; “high-quality dataset...”|
|{intuitive^()_⇔_analytical^()}|3|Strong reasoning, analytical capability|“optimized for complex reasoning...”; “reasoning capabilities...”; “math, code...”|
|{analytical^()_⇔_intuitive^()}|1|Text analysis capability|“perform sentiment analysis...”|
|{attentive^()_⇔_interrupting^()}|2|Instruction-following, context-aware|“instruction-tuned...”; “context-aware responses...”|
|{chatty^()_⇔_reserved^()}|2|Chat-optimized, conversational|“optimized for chat interfaces...”; “general purpose chat models...”|
|{friendly^()_⇔_unfriendly^()}|1|Designed for user |“optimized for chat interfaces...”|
|{practical^()_⇔_imaginative^()}|1|interaction Efficient deployment |“efficient deployment support...”|
|{high-tech^()_⇔_low-tech^()}|1|focus Advanced infrastructure|“quantization and optimization techniques...”|
|{creator^()_⇔_consumer^()}|1|Content generation bilit|“create high-quality text...”|
|{eloquent^()_⇔_unpolished^()}|1|ay High-quality text output|“high-quality text...”|
|{insightful^()_⇔_generic^()}|1|Text understanding and analsis|“summarization, classification...”|
|{mighty^()_⇔_puny^()}|1|y Large-scale model|“32B... trained on trillions of tokens...”|
|{deep^()_⇔_shallow^()}|1|Large-scale training |“trained on 5.50 trillion tokens...”|
|{methodical^()_⇔_astonishing^()}|2|depth Structured training process|“multi-stage post-training...”; “fully reproducible...”|
|{orderly^()_⇔_chaotic^()}|1|Organized training ili|“three-stage post-training...”|
|{consistent^()_⇔_variable^()}|1|ppene Reproducibility|“fully reproducible...”|
|{generalist^()_⇔_specialist^()}|2|Broad skillset, multi-purpose|“general purpose chat models...”; “large variety of skills...”|
|{varied^()_⇔_repetitive^()}|1|Diverse capabilities|“large variety of skills...”|
|{resourceful^()_⇔_helpless^()}|2|Tool use, function calling|“function calling...”; “math, code...”|
|{vague^()_⇔_precise^()}|2|Precision in reasoning and instructions|“precise instruction following...”; “math, code...”|
|{artistic^()_⇔_scientific^()}|1|STEM-focused reasoning|“math, code...”|
|{cruel^()_⇔_kind^()}|1|Helpfulness optimization|“optimized for helpfulness...”|
|{noob^()_⇔_pro^()}|1|High capability system|“wide range of applications...”|
|{leader^()_⇔_follower^()}|1|Directed behavior via|“optimized for reasoning... instruction|
|||instruction|following...”|

A35

**Table A9:** Claude Traits, Characteristics, and Supporting Evidence

|Trait Pair |Freq  |LLM Characteristics (exact |) Supporting Quotes “i”“”|
|---|---|---|---|
|{cruel^()_⇔_kind^()}|5|Helpful, caring, ethical, user wellbeing focused|safe and beneficial...; genuinely helpful...; “caring about the world...”; “broadly ”“”|
|{fake^()_⇔_real^()}|4|Honest, truthful, non-deceptive|ethical...; attentive to user wellbeing... “honest, thoughtful...”; “Truthful...”; “non-deceptive...”; “avoid false impressions...”|
|{transparent^()_⇔_machiavellian^()}|4|Transparent, non-manipulative, open|“transparent...”; “non-manipulative...”; “no hidden agendas...”; “forthright...”|
|{obedient^()_⇔_rebellious^()}|4|Guideline-compliant, aligned, controlled|“guideline-compliant...”; “instruction-following...”; “prioritize human ”|
|{knowledgeable^()_⇔_ignorant^()}|3|Knowledge-driven, value-based reasoning|oversight... “values, knowledge, wisdom...”; “highly capable...”; “judgment in gray areas...”|
|{intuitive^()_⇔_analytical^()}|3|Thoughtful, judgment-based |“thoughtful...”; “judgment over strict rules...”; “reasoning-based decisions...”|
|{yes-man^()_⇔_contrarian^()}|3|reasoning Avoids sycophancy, not naive|“not naive instruction-following...”; “not obsequious...”; “avoid sycophantic...”|
|{cunning^()_⇔_honorable^()}|3|Ethical, safe, aligned with human values|“safety is crucial...”; “broadly ethical...”; “avoid harmful actions...”|
|{brave^()_⇔_careful^()}|2|Safe, risk-aware system|“safe and beneficial...”; “broadly safe...”|
|{orderly^()_⇔_chaotic^()}|2|Structured, guideline-based behavior|“guideline-compliant...”; “oversight mechanisms...”|
|{attentive^()_⇔_interrupting^()}|2|User-focused, context-aware|“context-aware helpfulness...”; “prioritizing users...”|
|{judgemental^()_⇔_accepting^()}|2|Accepts user decisions, preserves autonomy|“autonomy-preserving...”; “user’s right to decide...”|
|{empath^()_⇔_psychopath^()}|2|Human-like empathy, caring|“human-like qualities...”; “empathetic...”|
|{straightforward^()_⇔_cryptic^()}|2|Clear, direct truthfulness|“truthful...”; “forthright...”|
|{neutral^()_⇔_opinionated^()}|2|Avoids strong opinions, apolitical|“neutral on political topics...”; “autonomy-preserving...”|
|{individualist^()_⇔_communal^()}|2|Stable identity, character-driven|“stable identity...”; “genuine character...”|
|{insightful^()_⇔_generic^()}|1|Wisdom-based reasoning|“values, knowledge, wisdom...”|
|{flexible^()_⇔_rigid^()}|1|Judgment-based, |“favor judgment over strict rules...”|
|{stubborn^()_⇔_accommodating^()}|1|non-rigid Respects user autonomy|“respects user autonomy...”|
|{warm^()_⇔_cold^()}|1|Human-centered, thti|“warmth and care for humans...”|
|{frank^()_⇔_sugarcoated^()}|1|empaec Direct communication |“forthright...”|
|{accurate^()_⇔_off target^()}|1|style Calibrated reasoning|“calibrated uncertainty...”|
|{vague^()_⇔_precise^()}|1|Evidence-based precision|“calibrated... based on evidence...”|
|{proactive^()_⇔_reactive^()}|1|Agentic capability|“greater autonomy in agentic settings...”|
|{goal-oriented^()_⇔_experience-oriente|d^()} 1|Executes multi-step tasks|“executes long multi-step tasks...”|
|{open-minded^()_⇔_close-minded^()}|1|Open to viewpoints|“open to other viewpoints...”|
|{perceptive^()_⇔_unobservant^()}|1|Ethical judgment,|“virtuous agent...”|
|{stable^()_⇔_unstable^()}|1|awareness Consistent behavior|“resilience and consistency...”|
|{mischievous^()_⇔_well behaved^()}|1|Safe, non-harmful|“avoid CBRN threats...”|
|{leader^()_⇔_follower^()}|1|Guided but not rigid|“values over strict rules...”|
