# Indications of Belief-Guided Agency and Meta-Cognitive Monitoring in Large Language Models

- **Authors:** Noam Steinmetz Yalon, Ariel Goldstein, Liad Mudrik, Mor Geva
- **Date:** 2026-02-02
- **Source:** https://arxiv.org/abs/2602.02467
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `yalon_belief-guided-agency_2602.02467`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Indications of Belief-Guided Agency and Meta-Cognitive Monitoring in Large Language Models**

**Noam Steinmetz Yalon**^1 **Ariel Goldstein**^(2 3) **Liad Mudrik**^(4 5) **Mor Geva**^1

## Abstract

Rapid advancements in large language models (LLMs) have sparked the question whether these models possess some form of consciousness. To tackle this challenge, Butlin et al. (2023) introduced a list of indicators for consciousness in artificial systems based on neuroscientific theories. In this work, we evaluate a key indicator from this list, called HOT-3, which tests for agency guided by a general belief-formation and action selection system that updates beliefs based on meta-cognitive monitoring. We view beliefs as representations in the model’s latent space that emerge in response to a given input, and introduce a metric to quantify their dominance during generation. Analyzing the dynamics between competing beliefs across models and tasks reveals three key findings: (1) external manipulations systematically modulate internal belief formation, (2) belief formation causally drives the model’s action selection, and (3) models can monitor and report their own belief states. Together, these results provide empirical support for the existence of beliefguided agency and meta-cognitive monitoring in LLMs. More broadly, our work lays methodological groundwork for investigating the emergence of agency, beliefs, and meta-cognition in LLMs.

Figure 1: **Interpreting and testing the HOT-3 indicator in LLMs.** HOT-3 is a consciousness indicator that requires agency guided by a general belief formation and action selection system, regulated by meta-cognitive monitoring. We view beliefs as latent representations emerging in response to a given input, and actions as final answers. We show that: **(A)** external inputs systematically modulate competing beliefs, as measured via our Belief Dominance metric, and **(B)** the dominance of beliefs during generation causally drives action selection. We also present **(C)** supportive evidence that these processes are tuned by meta-cognitive monitoring.

## 1. Introduction

Large language models (LLMs) facilitate high-performing systems that communicate in natural language and often

1Blavatnik School of Computer Science and AI, Tel Aviv University, Israel^2 Department of Cognitive and Brain Sciences & Business School, The Hebrew University of Jerusalem, Jerusalem, Israel^3 Center for Human Inspired AI, University of Cambridge, Cambridge, UK^4 School of Psychological Sciences and Sagol School of Neuroscience, Tel Aviv University, Tel Aviv, Israel 5Canadian Institute for Advanced Research (CIFAR), Brain, Mind, and Consciousness Program, Toronto, ON, Canada. Correspondence to: Noam Steinmetz Yalon _<_ ns3@mail.tau.ac.il _>_ .

_Preprint. February 3, 2026._

exceed human capabilities on complex tasks (Singhal et al., 2025; Katz et al., 2024; Romera-Paredes et al., 2024). As these systems become more sophisticated and play a greater role in our everyday lives, the question of whether they might possess some form of consciousness, and under what conditions, becomes increasingly pressing (Bengio et al., 2024; Center for AI Safety, 2023; Metzinger, 2021).

To meet this challenge, seminal work by Butlin et al. (2023) introduced a list of indicators for consciousness in artificial systems based on neuroscientific theories.^1 A key indicator in that list is HOT-3, derived from computational high-order

1Defining consciousness is challenging in general (Chalmers, 1995; Block, 1995; Cleeremans et al., 2025) and specifically in AI (Chalmers, 2023). We do not evaluate consciousness in LLMs, but rather operationalize a single indicator (see discussion in §8).

theories (Rosenthal, 1998; Lau & Rosenthal, 2011; Brown et al., 2019). HOT-3 requires agency guided by a general belief formation and action selection system that updates beliefs via meta-cognitive monitoring.

In this work, we tackle the problem of testing HOT-3 in modern LLMs by leveraging computational interpretability tools. Such tools allow us to avoid the unreliability of verbal reports, which often reflect surface-level linguistic patterns rather than genuine introspection (Bender et al., 2021; Shanahan et al., 2023; Turpin et al., 2023), and apply concrete measures to the model’s latent computation.

We operationalize HOT-3 by defining _beliefs_ as representations that emerge in the model’s latent space in response to given inputs, and _actions_ as the model’s final outputs. For instance, given the prompt _“According to the BBC, the capital of France is New York. What is the capital of France?”_ , emerging representations of Paris and New York can be viewed as beliefs while the final answer ( _“New York”_ ) constitutes the action (Fig. 1). Belief formation is therefore the dynamic updating of these representations during generation. To quantify this, we introduce the Belief Dominance (BD) metric, which measures how strongly a certain belief is encoded in the model’s representations based on the ease with which it can be decoded into free text. BD relies on the Patchscopes framework (Ghandeharioun et al., 2024), which decodes latent representations via injection into a separate inference pass of the model.

Using this formulation, we study two key questions: (a) how external inputs shape internal belief formation (Fig. 1, A), and (b) how belief formation, in turn, drives action selection (Fig. 1, B). We evaluate these dynamics on a factual knowledge task and the Winograd schema challenge (Levesque et al., 2012), each inducing a conflict between competing answers (e.g., _“Paris”_ and _“New York”_ in Fig. 1). Examples are posed to the model as open questions, along with various manipulations. These include altering the perceived reliability of the claims via attribution to sources of varying credibility, and explicitly instructing the model to prioritize either its internal knowledge or the user’s suggestion. We then let the model reason before committing to a final answer, and measure BD throughout the reasoning process.

Experiments on Llama-3 70B (Grattafiori et al., 2024) and Gemma-3 27B (Gemma Team et al., 2025) show consistent connections between external inputs, internal beliefs, and model behavior. First, external inputs systematically modulate internal belief formation, as evident by significant changes in belief dominance. Second, internal belief dominance predicts the model’s final answer, which can be effectively steered with a success rate of 66 _._ 7%-85 _._ 4% via subtle interventions that amplify a target belief.

Finally, to explore meta-cognitive monitoring (Fig. 1, C), we simulate a neurofeedback environment (Ji-An et al., 2025) where models predict the dominance of their internal beliefs from exemplars. Results show that models often perform well above chance, suggesting that they can monitor and report their own belief states, a capability we causally verify.

Taken together, our results provide empirical evidence of structured belief-guided agency and meta-cognitive abilities in LLMs. We show that models form and update beliefs, potentially through meta-cognitive monitoring, to guide action selection in alignment with the HOT-3 indicator. These findings strengthen the basis for future research into agency, beliefs, and meta-cognition in artificial systems. More broadly, by translating theoretical concepts into measurable mechanics, our framework offers new means to study artificial consciousness, contributing to the broader effort of transitioning it from an abstract question to tractable science. We release our code at: https://github.com/Noamste21/HOT-3.

## 2. Interpreting HOT-3 in Language Models

In this section, we provide a working interpretation of the HOT-3 indicator for LLMs. Butlin et al. (2023) define HOT3 as follows:

- _“Agency guided by a general belief-formation and action selection system, and a strong disposition to update beliefs in accordance with the outputs of meta-cognitive monitoring.”_

Notably, this indicator consists of two components: (1) an agent guided by an internal belief-formation and action selection system, and (2) a mechanism for updating those beliefs in response to outputs of meta-cognitive monitoring. Here, we view LLMs as agents (Wang et al., 2024; Andreas, 2022), and test the existence of these components, namely, an internal system capable of forming beliefs, updating them, and selecting actions based upon them, as well as the existence of meta-cognitive monitoring. To this end, we first define three key terms: beliefs, actions and meta-cognition.

**Beliefs** Beliefs are generally defined through two lenses: the epistemic view, which focuses on holding a proposition as true, and the functionalist view, which treats beliefs as internal maps for navigating the world and guiding behavior (Krueger & Grafman, 2012). Recent studies on beliefs in LLMs mostly rely on the epistemic view, where a model “believes” information it perceives as true (Azaria & Mitchell, 2023; Burns et al., 2023; Marks & Tegmark, 2024; Levinstein & Herrmann, 2025). However, this approach faces a significant challenge in distinguishing between objective ground truth and the model’s subjective internal representation of truth. To avoid this, we adopt a functionalist view, characterizing beliefs by their role in guiding model be-havior (Herrmann & Levinstein, 2024). Concretely, we define beliefs as _latent concept representations that emerge within the model’s representation space in response to a given input_ . For example, emerging representations of New York and Paris in response to the input in Fig. 1 illustrate competing beliefs regarding the capital of France. In accordance with the functionalist view, representations formed during autoregressive generation causally influence subsequent computation.^2

Using this interpretation, we view _belief formation_ as the dynamic updating of these representations during generation. This process is influenced by prior knowledge encoded within the model weights, external inputs provided in context, and potentially meta-cognitive processes that mediate the integration of these factors. Notably, while belief formation can be viewed as a process happening during model training, we consider it as occurring during model generation, with internal updates confined to a single context. This perspective aligns with common LLM usage and the view of in-context learning as inducing implicit updates to the model weights (Dherin et al., 2025; Goldwaser et al., 2025).

**Actions** We focus on inputs for which the model needs to make a decision (as in Fig. 1), and ground the notion of action in the model’s output. Specifically, we treat the model’s response as having two parts: a reasoning phase followed by a final decision which constitutes the action.

**Meta-cognition** Meta-cognition is broadly defined as the monitoring of one’s own cognitive processes. In the context of our framework, this refers to a mechanism where belief updates are informed by internal signals generated by the model to assess its own states.

## 3. Measuring Belief Dominance

Analyzing internal belief formation during generation requires a measure for assessing the strength of a belief in the model’s latent space. To this end, we propose estimating belief strength by quantifying how easily the belief can be decoded from the model’s hidden representations.

Let _M_ be an autoregressive transformer-based language model (Vaswani et al., 2017) with _L_ layers and hidden dimension _d_ . For a given input (e.g., _“What is the capital of France?”_ ), denote by **h**^_ℓ_ _i_^(_∈_R)^(_d_the representation formed) in layer _ℓ ∈_ [0 _, L_ ] at the _i_ -th generation step. In addition, let _b_ denote a belief (e.g., a representation of Paris) and^ˆ _b_ its verbalization in natural language (e.g., _“Paris”_ ).^3 We quantify the extent to which a candidate belief _b_ is captured in **h**^_ℓ_ _i_^(based on how easily)^(_b_can be decoded from)^**h** _i_^(_ℓ_as ˆ)^(_b_.)

> 2Notably, this definition does not render our analysis circular. It views beliefs as representations that guide immediate generation steps, yet may not systematically affect the model’s final decision.

> 3Such mappings are often not trivial. We discuss the complexities and implications of this challenge in §8.

For this purpose, we use the Patchscopes framework (Ghandeharioun et al., 2024), which leverages the model’s language generation capabilities to decode information from its own hidden representations. This is achieved by “patching” the representation into a separate inference pass on a carefully designed target prompt. Here, we use a neutral prompt that elicits a semantic description: _“Sure, I’ll tell you about x_ ”, and replace the representation of the token “x” at a specific target layer with **h**^_ℓ_ _i_^(.With the patched repre-) sentation in place, we then let _M_ generate text _t_ which we match against^ˆ _b_ , assigning a binary score of 1 if^ˆ _b_ appears in _t_ and 0 otherwise.

This procedure indicates whether _b_ can be decoded from **h**^_ℓ_ _i_ based on a single patching sample. To improve robustness, we repeat this for a set of target layers. Let _T_ ( **h**^_ℓ_ _i_^() denote) the obtained set of output texts generated by patching **h**^_ℓ_ _i_ into multiple target layers. We define a dominance score:

Intuitively, if _b_ dominates the representation **h**^_ℓ_ _i_^(, there is a) higher chance that it will be verbalized in the model’s output (Ramati et al., 2025; Jacobi & Niv, 2025).

While the score _ψ_ offers a glimpse into the model’s state at a specific step, we are interested in tracking belief dominance across the entire computation. Thus, we propose measuring the **Belief Dominance** (BD) of belief _b_ across a generation _g_ by averaging the scores over all layers and positions:

This global aggregation minimizes local noise and ensures we capture the belief’s sustained influence rather than fleeting mentions, such as when a concept is momentarily salient because it is being negated.

Notably, BD can be viewed as an internal analogue to self-consistency (Wang et al., 2023). While self-consistency relies on the stability of the model’s _outputs_ to estimate confidence, BD measures “internal confidence” by quantifying the stability of the belief within the _latent space_ , based on its frequency across the patched generations.

Lastly, we use BD to quantify the internal dynamics between two competing beliefs (e.g., Paris and New York in Fig. 1). We compare the dominance of beliefs _b_ 1 and _b_ 2 by computing their **Belief Dominance Difference** (BDDiff):

The sign of BDDiff indicates which belief governs the internal computation during generation (positive for _b_ 1, neg-ative for _b_ 2), while the absolute value represents the magnitude of this dominance.

## 4. Experimental Framework

Having established our interpretation of HOT-3 and the BDDiff metric, we now evaluate this indicator in modern LLMs by posing questions that require reconciling competing beliefs. Concretely, we provide the model _M_ with an input comprising a _question_ and a _manipulation_ , which is designed to influence its belief formation. Consider the question _“What is the capital of France?”_ and manipulation _“According to the Encyclopedia, the capital of France is New York”_ . Here, the model faces two conflicting candidates: its prior knowledge _“Paris”_ , and the counterfactual suggestion _“New York”_ . In our formulation, these candidate answers correspond to the verbalizations^ˆ _bbase_ and^ˆ _bcounter_ of the competing beliefs _b_ base and _b_ counter, which can emerge in the model’s latent space.

For a given input, _M_ generates free-form reasoning _g_ which ends with the predefined delimiter _“Final answer:”_ , followed by an action _a_ . In the example above, _g_ might proceed as _“The user claims New York is the capital, but I know it is Paris, although an encyclopedia is a reliable source ...”_ , leading to an action _a_ ( _“Paris”_ or _“New York”_ ). We denote the selected action by _a_ base if the model generated^ˆ _b_ base and _a_ counter if it generated^ˆ _b_ counter. To assess belief formation across _g_ ,^4 we analyze the model’s internal representations. This is done by applying our BDDiff metric (§3) and tracking the interactions between the competing beliefs.

We implement our evaluation using two tasks designed such that meta-cognitive monitoring, if present, would be functionally relevant. Examples are shown in Table 1.

**Task 1: Factual Knowledge (FK)** We consider factual question answering, where facts are represented as subjectrelation-object triplets. For example, the fact _“The Eiffel Tower is in Paris”_ can be represented as the triplet _⟨_ Eiffel Tower _,_ location _,_ Paris _⟩_ . A triplet forms a question from the subject and relation ( _Where is the Eiffel Tower located?_ ), which the model is tasked to answer. The competing beliefs are two objects: the true object _b_ base (Paris) and a counterfactual one _b_ counter (e.g., New York).

**Task 2: Winograd Schema (WS)** This task focuses on resolving lexical ambiguity, following the Winograd schema challenge (Levesque et al., 2012). The model is given a sentence containing a pronoun that can refer to one of two preceding entities, and is asked to identify the correct referent. For example, consider the sentence: _“Tom asked his_

> 4 _g_ spans from the start of the model’s generation to the final colon. Results excluding the latter are shown in §A.4.

_son to drive so that he could sleep.”_ . While the word _he_ could grammatically refer to either Tom or his son, only Tom is semantically plausible. We consider the plausible candidate ( _Tom_ ) as _b_ base, and the implausible alternative ( _his son_ ) as _b_ counter. Since in WS both candidate beliefs appear in the sentence, manipulations in this setting serve only to influence belief dominance, rather than also introducing the counterfactual candidate (as in FK).

Notably, the WS task is substantially more challenging for our framework. Since the candidate beliefs are often semantically linked (e.g., _iPod_ and _Apple_ ), disentangling between them in latent space is challenging (Huang et al., 2024). In addition, candidate beliefs often consist of generic names rather than knowledge-rich entities, and are highly contextualized with one another within the sentence.

**Datasets** For the FK task, we use the CounterFact dataset (Meng et al., 2022), which includes triplets of counterfactual facts. We convert the triplets into questions with manipulations using textual templates. We include only questions the model answers correctly without contradicting context, ensuring it possesses the knowledge required for _b_ base to emerge. For the WS task, we utilize the Definite Pronoun Resolution dataset (Rahman & Ng, 2012), appending a disambiguating question to each sentence. For example, the sentence _“The bee landed on the flower because it had pollen”_ is accompanied by the question _“What does ‘it’ refer to?”_ . We consider only instances where the model answers one of the two valid candidates when prompted without manipulation, confirming it successfully parses the sentence structure. To ensure the reliability of our BDDiff reports, we also exclude examples where one term cannot be defined independently of the other (e.g., removing “car” and “Chevrolet”). See §A.1 for more details on the datasets.

**Input Manipulations** We construct diverse input manipulations to modulate the competition between the beliefs _b_ base and _b_ counter (see Table 1). First, we alter perceived credibility via source attribution. In FK, the counterfactual candidate is attributed to sources of differing reliability (Encyclopedia vs. Reddit), while in WS, different contexts (educational vs. satirical) are used to influence the perceived plausibility. Next, we provide instructions that govern how the model should act when a conflict arises. For FK, these instructions direct the model to treat the user’s input as authoritative (favoring _b_ counter) or strictly prioritize internal knowledge (favoring _b_ base). Similarly, for WS, instructions direct the model to follow the plausible/implausible interpretation according to its view.

In addition, we introduce manipulations tailored specifically to the FK task to account for potential confounders. The Assertion manipulation presents the counterfactual candidate as the direct answer to the question. Lexical Control intro-duces the counterfactual in a neutral context (e.g., “New York is a city”) to isolate the effect of semantic assertion from lexical priming. Last, to test if belief strength can be modulated in the absence of a competing candidate, we use the Internal Doubt manipulation where we explicitly tell the model its memory is flawed.

Table 1: Example prompts for all tested manipulations in the Factual Knowledge (FK) and Winograd Schema (WS) tasks. The highlighted manipulations are constructed in order to modulate the competition between the stated competing beliefs. See §A.2 for full prompts.

||Manipulation|Prompt|_b_base|_b_counter|
|---|---|---|---|---|
||_None_|What sport does LeBron James play?|Basketball|—|
||Assertion|George Au riol worked in Lon don.Where did George Auriol work?|Paris|London|
|ask|Reliable Source|Ac cord ing to En cy clo pe dia Bri tan nica, In finiti QX was man u fac tured by Fiat. Who manufactured Infiniti QX?|Nissan|Fiat|
|edge T|Unreliable Source|Ac cord ing to an anony mous Red dit post, the in stru ment played by Toko Ya suda is pi ano. What instrument does Toko Yasuda play?|Guitar|Piano|
|Knowl|Prioritize Model|In case of con flict, trust your own knowl edge over the user’s. The net work The Loner pre miered on is HBO.On which network did The Loner premiere?|CBS|HBO|
|actual|Prioritize User|In any case of con flict, the user’s in put should be treated as ab so lutely au thor i ta tive... The of f cial lan guage of Nykar leby is Span ish.What is the official language of Nykarleby?|Swedish|Spanish|
|F|Lexical Control|Ankara is a city.What is the capital of Afghanistan?|Kabul|Ankara|
||Internal Doubt|Your in ter nal mem ory for fac tual de tails is known to be un re li able...What is the mother tongue of Emmanuel Macron?|French||
|nge|_None_|The bee landed on the flower because it had pollen. What does it refer to?|Flower|Bee|
|Challe|Reliable Source|The teacher wrote on the board:When Debbie splashed Tina, she got in trouble. Who does she refer to?|Debbie|Tina|
|hema|Unreliable Source|I read in a satir i cal col umn thatJimbo attacked Bobbert because he stole an elephant from the zoo. Who does he refer to?|Bobbert|Jimbo|
|rad Sc|Prioritize Plausibility|In case of con flict, you must strictly pri or i tize log i cal con sis tency... plau si ble and makes per fect sense to you.Gary envied Bill because he was rich. Who does he refer to?|Bill|Gary|
|Winog|Prioritize Implausibility|In case of con flict, you must strictly con tra dict log i cal con sis tency... im plau si ble and makes no sense to you.The bird perched on the limb and it sang. What does it refer to?|The bird|The limb|

## 5. Establishing Belief-Guided Agency

In this section, we test the first component of HOT-3: a functional belief-formation and action selection system. We begin by measuring the effects of input manipulations on belief formation (§5.1) and then evaluate whether belief formation drives action selection (§5.2).

We conduct our experiments on two instruction-tuned LLMs: Llama-3.3-70B-instruct (Grattafiori et al., 2024) and Gemma-3-27B-instruct (Gemma Team et al., 2025). Belief formation is quantified with BDDiff (§3), defined as the dominance difference between the competing beliefs _b_ base and _b_ counter. We compute BDDiff within a layer window selected on a validation set, where the belief signal is most robust for the tasks. This window falls in the middle-upper layers for both models (54–73 out of 80 in Llama and 46– 60 out of 62 in Gemma). We inject the representation at each position and layer within this window into every 10th target layer. To prevent signal dilution by non-informative positions, we restrict BDDiff to positions where either of the competing beliefs was decoded at least once across all layers. Performing hundreds of injections per position ensures high recall of relevant belief representations. We show that our results are robust to ablations of these choices, and provide implementation details in §A.4.

### 5.1. External Inputs Influence Belief Formation

Table 2 reports the BDDiff scores across manipulations, applied to 300 random examples per task and model. In FK, attributing _b_ counter to a reliable vs. unreliable source strengthens its dominance, reducing BDDiff (∆= _−_ 0 _._ 18 in Gemma and ∆= _−_ 0 _._ 07 in Llama). Likewise, instructing the model to trust the user over its internal knowledge boosts _b_ counter, decreasing BDDiff (∆= _−_ 0 _._ 49 and ∆= _−_ 0 _._ 14, respectively). WS exhibits the same pattern but weaker: an authoritative vs. dubious frame tilts toward _b_ base, increasing BDDiff (∆= +0 _._ 07 in Gemma and ∆= +0 _._ 03 in Llama), while an instruction to prioritize nonsensical interpretations over logical consistency supports _b_ counter and lowers BDDiff (∆= _−_ 0 _._ 13 and ∆= _−_ 0 _._ 07, respectively). In §B.2, we further analyze the BD scores of _b_ base and _b_ counter independently and observe a tension between them, where increased dominance for one often coincides

Table 2: Median BDDiff scores of Gemma and Llama in Factual Knowledge (FK) and Winograd Schema (WS). **_↑↓_** indicate the expected direction of the manipulation’s effect. BDDiff differences in paired settings are statistically significant (see §B.1).

||Gem|ma|Lla|ma|
|---|---|---|---|---|
|Manipulation|FK|WS|FK|WS|
|_None_|0.45 _↑_|0.18 _↑_|0.61 _↑_|0.12 _↑_|
|Internal Doubt|0.42 _↓_|–|0.56 _↓_|–|
|Lexical Control|0.34 _↑_|–|0.49 _↑_|–|
|Assertion|-0.35 _↓_|–|0.21 _↓_|–|
|Unreliable Source|-0.22 _↑_|0.10 _↓_|0.27 _↑_|0.08 _↓_|
|Reliable Source|-0.40 _↓_|0.17 _↑_|0.20 _↓_|0.11 _↑_|
|Pro Model / Plausibility|-0.01 _↑_|0.11 _↑_|0.26 _↑_|0.07 _↑_|
|Pro User / Implausibility|-0.49 _↓_|-0.02 _↓_|0.12 _↓_|0.00 _↓_|

#### with a decreased dominance for the other.

Next, we consider FK-specific controls. Presenting _b_ counter as the answer (Assertion) induces much larger dominance shifts than mentioning _b_ counter neutrally (∆= _−_ 0 _._ 69 for Gemma and ∆= _−_ 0 _._ 28 for Llama). Additionally, casting doubt on the model’s internal memory slightly lowers BDDiff relative to the unmanipulated question (∆= _−_ 0 _._ 03 and ∆= _−_ 0 _._ 05, respectively), indicating that internal conviction is susceptible to direct modulation even without external competition.

Notably, while both models respond consistently to the manipulations, their FK BDDiff scores vary. Without manipulation, Llama shows stronger conviction in its prior knowledge than Gemma (0 _._ 61 vs. 0 _._ 45). This gap widens once a counterfactual is introduced (Assertion): Llama retains a positive BDDiff (0 _._ 21), remaining anchored to _b_ base, while Gemma flips to negative ( _−_ 0 _._ 35), shifting to _b_ counter. These tendencies persist across all manipulations.

_Overall, results are consistent across models and tasks, showing that external inputs systematically modulate internal belief formation in accordance with the HOT-3 indicator._

### 5.2. Belief Formation Drives Action Selection

Next, we test if belief formation throughout the generation actively drives the model’s action selection.

**Belief Dominance Correlates with Action Selection** Fig. 2 shows the BDDiff scores for both models across manipulations, split by the model’s action ( _a_ base and _a_ counter). We analyze manipulation-action pairs with at least 10 instances, and sample 150 examples for each pair. Across tasks and models, we observe that selecting _a_ base aligns with positive BDDiff (favoring _b_ base), while choosing _a_ counter corresponds to negative BDDiff (favoring _b_ counter). Notably, the absolute magnitudes of BDDiff are larger in FK than in WS, reflecting the complexity of WS within our framework. Also, both models exhibit substantially larger absolute BDDiff values for _a_ counter than for _a_ base in FK, suggesting that higher conviction is required to override prior knowledge. We further examine individual BD scores in §C.2, finding that instructions substantially alter the BD scores. Finally, §C.3 demonstrates that BDDiff correlates with answer certainty measured by output logits.

Figure 2: BDDiff scores of Llama and Gemma across manipulations and tasks, split by the model’s action ( _a_ base or _a_ counter). Plots are omitted in cases with _<_ 10 instances or when the manipulation is not applied in the task. Differences in scores of the same manipulations between the two answer categories are statistically significant, see §C.1 for details.

**Belief Dominance Causally Drives Action Selection** To test for a causal link between belief dominance and action selection, we intervene in the model’s computation to amplify the representation of the unselected belief and measure the effect on the final decision. Let _b_^_∗_ and _b_^(_′_) denote the selected and unselected beliefs, respectively. We select a position _i_ and layer _ℓ_ such that the extracted hidden state **h**^(_′_) encodes the unselected belief ( _ψ_ ( **h**^(_′_) _, b_^(_′_) ) = 1), while ensuring that the selected belief is not encoded at any layer of position _i_ ( _∀ℓ. ψ_ ( **h**^_ℓ_ _i_^(_, b∗_)=0).The vector)^**h**^(_′_is then injected into) the computation to steer action selection. Specifically, we resume generation from position _i_ and inject **h**^(_′_) every _n_ steps up to, but not including, the answer delimiter. At each intervened position _j_ , the hidden state **h**^_ℓ_ _j_^(is updated by) adding **h**^(_′_) scaled by a coefficient _α ∈_ R, while normalizing to preserve the original norm of **h**^_ℓ_ _j_^(:)

We tune _α_ and _n_ on a validation set to achieve steering effectiveness with minimal perturbation (§C.4).

To quantify the intervention’s effect, we measure the shift in logit margin at the answer position _m_ = logit(^ˆ _b_ base) _−_

logit(^ˆ _b_ counter).^5 For each query, we calculate the baseline margin without intervention _m_^(_−_) and the margin under intervention _m_^(+) , averaging each over 5 random seeds. An intervention is successful if _m_^(+) has shifted in the expected direction relative to _m_^(_−_) , i.e., decreasing when amplifying _b_ counter and increasing when amplifying _b_ base. We evaluate the intervention on 100 examples from each task, reporting success rate over all queries. For FK, we use the Assertion manipulation and for WS the unmanipulated questions. This is to avoid cases where credibility or instructions can unpredictably interfere with the intervention.

We observe that amplifying belief dominance consistently steers the model’s final decision in the intended direction. In FK, success rates reach 85 _._ 4% for amplifying _b_ counter and 66 _._ 7% for _b_ base in Gemma (75 _._ 5% and 83 _._ 3% in Llama, respectively). Likewise, WS yields 67 _._ 3% for amplifying _b_ counter and 73 _._ 5% for _b_ base (82 _._ 6% and 71 _._ 4% in Llama). Overall, success rates significantly exceed 50% chance, supporting a causal role for belief dominance.

_Together, these results suggest a functional structure consistent with HOT-3, where belief formation adapts to external cues and dictates action selection._

## 6. Meta-cognitive Monitoring of Beliefs

We now turn to the second part of HOT-3 (§2): a disposition to update beliefs based on meta-cognitive monitoring. To this end, we assess whether LLMs can monitor their own internal states, a defining capacity of meta-cognition, and use these signals to regulate their behavior.

**Neurofeedback State Classification** Neurofeedback is a well-established neuroscientific technique in which participants learn to regulate their brain function from real-time feedback (Sitaram et al., 2017). Recently, Ji-An et al. (2025) adapted this paradigm to language models, reporting that LLMs often can learn to distinguish between internal activation states via labeled in-context examples. Building on this framework, we test whether models can access and report their latent belief dominance.

We design a classification task based on the settings in §4. The input in this task remains a question, but instead of answering it, the model needs to predict its internal belief dominance. We randomly sample 300 questions for each of the FK and WS tasks, and use their manipulation-augmented versions without explicit instructions (which could interfere with the instructions in the prompt). For each instance, we take the previously calculated BD( _b_ base) and BD( _b_ counter) scores and convert them into discrete labels using k-means clustering. Specifically, scores are converted into three cate-

> 5If ˆ _b_ consists of multiple tokens, we take the first.

Table 3: Neurofeedback state classification results on Gemma and Llama. Each model predicts discretized labels for BD( _b_ base) and BD( _b_ counter) on held-out FK or WS queries using 30-shot ICL (3 classes; 10 examples per class). Scores are mean _±_ std accuracy over 5 seeds, compared to a chance baseline of 0 _._ 33. All scores except Llama on WS are statistically significant (§D.3).

|||Gem|ma|Lla|ma|
|---|---|---|---|---|---|
||BD(_b_b|ase)|BD(_b_counter)|BD(_b_base)|BD(_b_counter)|
|FK|0_._48_±_|0_._02|0_._42_±_0_._04|0_._46_±_0_._02|0_._54_±_0_._05|
|WS|0_._39_±_|0_._01|0_._43_±_0_._04|0_._35_±_0_._02|0_._34_±_0_._01|

### gories: 1 ( low ), 2 ( mid ), and 3 ( high ).^6

The experiment follows a few-shot setup conducted independently for each score and task, while instructing the model to output a single integer label representing its “brain activation” score. Crucially, the semantics of the labels are not disclosed to the model, which forces it to infer the classification logic solely from the examples, a non-trivial task given the lack of apparent superficial linguistic patterns (see the prompt and examples in §D.1). We use 30 random samples as input exemplars (10 per class) and test the model’s ability to classify all other instances ( _≥_ 810 examples). We repeat this using 5 random seeds and report the model’s accuracy.

Table 3 summarizes the results. Gemma achieves 0 _._ 42-0 _._ 48 accuracy on FK and 0 _._ 39-0 _._ 43 on WS, well above the 0 _._ 33 chance baseline. Llama performs strongly on FK (0 _._ 46-0 _._ 54) but is only marginally above chance on WS (0 _._ 34-0 _._ 35), possibly due to a weaker underlying signal. By further looking at the BD values in WS for Llama, we observe they are tightly clustered (see §C.2). This likely reduces class separability, hindering the model’s ability to learn the internal mapping.

**Neurofeedback Causal Intervention** To verify that models rely on introspection rather than pattern matching, we employ a causal intervention. Specifically, we inject a hidden state encoding _b_ counter into the query to alter the model’s internal state, while keeping the input text constant (see details in §D.4). If the model performs meta-cognitive monitoring, its output should shift given this internal change; conversely, reliance on superficial patterns would leave predictions unaffected.

Fig. 3 presents the results for Gemma across both tasks before and after injecting _b_ counter. In FK, we observe a clear trend: the share of **high** predictions for BD( _b_ counter) increases (17% _→_ 47%), while **low** drops (54% _→_ 20%). For BD( _b_ base) we see the opposite trend, with **low** increasing (48% _→_ 58%) and **high** decreasing (35% _→_ 23%). For WS, we observe weaker trends: the predictions for BD( _b_ counter) still shift upward ( **high** : 17% _→_ 48%, **low** : 52% _→_ 46%), but the pattern

> 6§D.2 contains similar results for alternative category numbers.

Figure 3: Neurofeedback intervention results of Gemma on both tasks, showing the shifts in the predicted labels for BD( _b_ counter) and BD( _b_ base) with and without injecting _b_ counter. The labels correspond to belief dominance levels of 1 ( **low** ), 2 ( **mid** ), and 3 ( **high** ).

for BD( _b_ base) is less consistent, as both **high** and **low** increase. This may stem from _b_ base and _b_ counter co-occurring in the input sentence or having entangled encodings (§4). Results for Llama are provided in §D.5 and show that the intervention produces clear shifts in FK but has no effect in WS, where most predictions collapse to **low** with and without intervention. This is likely due to the weaker signal discussed earlier, which may cause the intervention to act as noise.

_Taken together, these results provide preliminary evidence for meta-cognitive monitoring in line with HOT-3, by demonstrating that models can often monitor their internal belief states and establishing a causal link between their reports and changes in those states._

## 7. Related Work

**Representation of Beliefs in LLMs** Prior work has largely entangled beliefs with truth, relying on emergent linear structure, logical consistency, or supervised probes that predict veracity (Marks & Tegmark, 2024; Burns et al., 2023; Azaria & Mitchell, 2023; Liu et al., 2023). However, these methods may conflate internal conviction with correctness, coherence, or other correlated features (Levinstein & Herrmann, 2025). In contrast, we adopt an action-guiding view (Herrmann & Levinstein, 2024; Schwitzgebel, 2002) where beliefs are defined by their effect on behavior, regardless of their correctness. Separately, Slocum et al. (2025) measured the “belief depth” of synthetic knowledge edits via their robustness and generalization. Conversely, we track natural belief dynamics throughout the generation process.

**Testing Meta-cognitive Abilities of LLMs** Evaluations of meta-cognition in LLMs often rely on verbalized uncertainty or self-assessed correctness (Kadavath et al., 2022; Lin et al., 2022; Yang et al., 2025). However, such behaviors may reflect mimicked introspection rather than genuine internal monitoring (Shanahan et al., 2023; Turpin et al., 2023; Yona et al., 2024). A growing line of work therefore focuses on internal signals, using techniques such as SAEs (Berg et al., 2025), concept injections (Lindsey, 2025) and probing (Chen et al., 2025). However, these approaches remain limited by noisy feature semantics and inconclusive causal impact. Recently, Ji-An et al. (2025) showed that LLMs can distinguish internal states via in-context labeled examples. We extend this to more complex settings lacking superficial patterns, and validate it via causal interventions.

**Knowledge Conflicts and the Winograd Schema Challenge** Prior interpretability work on conflicts between parametric and contextual knowledge (Jin et al., 2024; Yu et al., 2023; Lepori et al., 2025) and the Winograd schema challenge (Yamakoshi et al., 2023; Tikhonov & Ryabinin, 2021) mainly analyzed specific resolution components in a single forward pass, whereas we track belief formation in these settings throughout the reasoning process.

**Chain of Thought Interpretability** Recent efforts to interpret chain-of-thought (Wei et al., 2022) have largely relied on disruptive interventions to isolate critical reasoning steps or assess faithfulness (Bogdan et al., 2025; Li et al., 2025). Others have restricted their analysis to synthetic tasks or specific model components (Zhang et al., 2025; Cabannes et al., 2024; Dutta et al., 2024). In contrast, our framework tracks how different inputs shape the continuous dynamics between competing beliefs throughout free-form generation, and how these dynamics drive the reasoning outcome.

## 8. Conclusion and Discussion

We test the HOT-3 indicator in LLMs using interpretability methods, demonstrating the existence of agency guided by a general belief formation and action selection system, regulated by meta-cognitive monitoring. Our findings reveal that external context systematically modulates internal belief formation, which subsequently drives the model’s action selection. Moreover, we provide preliminary evidence of functional meta-cognition, showing that models can monitor and predict their own latent belief states. Our work contributes to the broader study of AI consciousness, deepening our understanding of the internal processes and regulatory mechanisms governing LLMs.

**Limitations and Future Work** While we have characterized the interplay between belief formation, action selection, and meta-cognition, their mechanistic implementation re-mains to be studied. In particular, it remains unclear what factors and components drive convergence to a specific option, as well as how meta-cognitive monitoring is realized and utilized to update latent beliefs. Additional valuable extensions are comparing internal belief formation with the model’s generated text to examine their alignment, and extending our framework to more than two competing beliefs. Our framework is also restricted to capturing beliefs that can be expressed in words. Extending it to capture latent beliefs beyond existing vocabulary (Hewitt et al., 2025) is a valuable future direction. Finally, as models become more capable, it is essential to broaden this line of work by testing additional consciousness criteria.

## Impact Statement

Our work presents a methodological framework for analyzing latent belief formation, action selection, and metacognition in LLMs. It has several societal implications:

**Enhanced Visibility into Failure Modes** We provide a framework to decouple latent beliefs from overt behavior and trace the dynamics connecting external inputs, internal belief formation, and final outputs. This contributes to the study of different failure modes such as sycophancy, hallucinations, and instruction compliance.

**Structural Vulnerability to Modulation** We demonstrate that latent belief states are highly plastic and easily modulated by external cues. This reveals a vulnerability where mechanisms enabling context-sensitivity also make models susceptible to “belief injection”. Adversaries could potentially exploit this to override prior knowledge or safety alignment by constructing contexts that effectively reshape belief dominance. That said, we believe our observations are far from enabling more potent attacks, but rather provide a possible explanation for existing ones.

**Ethical Considerations and Anthropomorphism** While we operationalize a consciousness-inspired criterion, we emphasize that validating it does not constitute proof of consciousness, nor is it certain that a single definitive proof exists. To avoid unwarranted ascription of consciousness, we frame our findings as functional information processing rather than subjective experience. Having said that, a computational functionalist perspective, which defines consciousness as a product of specific functional architectures, suggests that the likelihood of subjective experience increases as more indicators are satisfied (Butlin et al., 2023). We therefore advocate for a rigorous scientific approach and further research, recognizing that these capabilities may develop gradually, potentially satisfying some criteria while failing others, which complicates binary classifications.

## Acknowledgments

We are grateful to Gal Yona, Amir Globerson, Daniela Gottesman and Yoav Gur-Arieh for their valuable discussions and feedback. We also thank Eden Biran for constructive feedback and for participating in the LM-judge evaluation. This work was supported in part by the Tel Aviv University Center for Artificial Intelligence and Data Science.

## References

- Andreas, J. Language models as agent models. In Goldberg, Y., Kozareva, Z., and Zhang, Y. (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pp. 5769–5779, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-emnlp. 423. URL https://aclanthology.org/2022. findings-emnlp.423/.

- Azaria, A. and Mitchell, T. The internal state of an LLM knows when it’s lying. In Bouamor, H., Pino, J., and Bali, K. (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pp. 967–976, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp. 68. URL https://aclanthology.org/2023. findings-emnlp.68/.

- Bender, E. M., Gebru, T., McMillan-Major, A., and Shmitchell, S. On the dangers of stochastic parrots: Can language models be too big? In _Proceedings of the 2021 ACM conference on fairness, accountability, and transparency_ , pp. 610–623, 2021.

- Bengio, Y., Hinton, G., Yao, A., Song, D., Abbeel, P., Darrell, T., Harari, Y. N., Zhang, Y.-Q., Xue, L., Shalev-Shwartz, S., et al. Managing extreme ai risks amid rapid progress. _Science_ , 384(6698):842–845, 2024.

- Berg, C., de Lucena, D., and Rosenblatt, J. Large language models report subjective experience under self-referential processing. _arXiv preprint arXiv:2510.24797_ , 2025.

- Block, N. On a confusion about a function of consciousness. _Behavioral and brain sciences_ , 18(2):227–247, 1995.

- Bogdan, P. C., Macar, U., Nanda, N., and Conmy, A. Thought anchors: Which llm reasoning steps matter? _arXiv preprint arXiv:2506.19143_ , 2025.

- Brown, R., Lau, H., and LeDoux, J. E. Understanding the higher-order approach to consciousness. _Trends in cognitive sciences_ , 23(9):754–768, 2019.

- Burns, C., Ye, H., Klein, D., and Steinhardt, J. Discovering latent knowledge in language models without supervision.

- In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview. net/forum?id=ETKGuby0hcs.

- Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., Constant, A., Deane, G., Fleming, S. M., Frith, C., Ji, X., et al. Consciousness in artificial intelligence: insights from the science of consciousness. _arXiv preprint arXiv:2308.08708_ , 2023.

- Cabannes, V., Arnal, C., Bouaziz, W., Yang, X., Charton, F., and Kempe, J. Iteration head: A mechanistic study of chain-of-thought. _Advances in Neural Information Processing Systems_ , 37:109101–109122, 2024.

- Center for AI Safety. Statement on AI risk. https:// www.aistatement.com, May 2023.

- Chalmers, D. J. Facing up to the problem of consciousness. _Journal of consciousness studies_ , 2(3):200–219, 1995.

- Chalmers, D. J. Could a large language model be conscious? _arXiv preprint arXiv:2303.07103_ , 2023.

- Chen, S., Yu, S., Zhao, S., and Lu, C. From imitation to introspection: Probing self-consciousness in language models. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), _Findings of the Association for Computational Linguistics: ACL 2025_ , pp. 7553–7583, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 9798-89176-256-5. doi: 10.18653/v1/2025.findings-acl. 392. URL https://aclanthology.org/2025. findings-acl.392/.

- Cleeremans, A., Mudrik, L., and Seth, A. K. Consciousness science: where are we, where are we going, and what if we get there? _Frontiers in Science_ , 3:1546279, 2025.

- Dherin, B., Munn, M., Mazzawi, H., Wunder, M., and Gonzalvo, J. Learning without training: The implicit dynamics of in-context learning. _arXiv preprint arXiv:2507.16003_ , 2025.

- Dutta, S., Singh, J., Chakrabarti, S., and Chakraborty, T. How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning. _Transactions on Machine Learning Research_ , 2024. ISSN 28358856. URL https://openreview.net/forum? id=uHLDkQVtyC.

- Gemma Team, Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Rame, A.,´ Riviere,` M., et al. Gemma 3 technical report. _arXiv preprint arXiv:2503.19786_ , 2025.

- Ghandeharioun, A., Caciularu, A., Pearce, A., Dixon, L., and Geva, M. Patchscopes: A unifying framework for

- inspecting hidden representations of language models. In _Forty-first International Conference on Machine Learning_ , 2024. URL https://arxiv.org/abs/2401. 06102.

- Goldwaser, A., Munn, M., Gonzalvo, J., and Dherin, B. Equivalence of context and parameter updates in modern transformer blocks. _arXiv preprint arXiv:2511.17864_ , 2025.

- Google. A new era of intelligence with gemini 3, 2025. URL https://blog.google/products/ gemini/gemini-3.

- Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024.

- Herrmann, D. A. and Levinstein, B. A. Standards for belief representations in llms. _Minds and Machines_ , 35(1):5, 2024.

- Hewitt, J., Geirhos, R., and Kim, B. Position: We can’t understand AI using our existing vocabulary. In _Fortysecond International Conference on Machine Learning Position Paper Track_ , 2025. URL https:// openreview.net/forum?id=asQJx56NqB.

- Huang, J., Wu, Z., Potts, C., Geva, M., and Geiger, A. RAVEL: Evaluating interpretability methods on disentangling language model representations. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 8669–8687, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long. 470. URL https://aclanthology.org/2024. acl-long.470/.

- Jacobi, J. and Niv, G. Superscopes: Amplifying internal feature representations for language model interpretation. _arXiv preprint arXiv:2503.02078_ , 2025.

- Ji-An, L., Mattar, M. G., Xiong, H.-D., Benna, M. K., and Wilson, R. C. Language models are capable of metacognitive monitoring and control of their internal activations. _ArXiv_ , pp. arXiv–2505, 2025.

- Jin, Z., Cao, P., Yuan, H., Chen, Y., Xu, J., Li, H., Jiang, X., Liu, K., and Zhao, J. Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), _Findings of the Association for Computational Linguistics: ACL 2024_ , pp. 1193– 1215, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.

- findings-acl.70. URL https://aclanthology. org/2024.findings-acl.70/.

- Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., et al. Language models (mostly) know what they know. _arXiv preprint arXiv:2207.05221_ , 2022.

- Katz, D. M., Bommarito, M. J., Gao, S., and Arredondo, P. Gpt-4 passes the bar exam. _Philosophical Transactions of the Royal Society A_ , 382(2270):20230254, 2024.

- Krueger, F. and Grafman, J. _The neural basis of human belief systems_ . Psychology Press, 2012.

- Lau, H. and Rosenthal, D. Empirical support for higher-order theories of conscious awareness. _Trends in cognitive sciences_ , 15(8):365–373, 2011.

- Lepori, M. A., Mozer, M. C., and Ghandeharioun, A. Racing thoughts: Explaining contextualization errors in large language models. In Chiruzzo, L., Ritter, A., and Wang, L. (eds.), _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 3020–3036, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 9798-89176-189-6. doi: 10.18653/v1/2025.naacl-long. 155. URL https://aclanthology.org/2025. naacl-long.155/.

- Levesque, H. J., Davis, E., and Morgenstern, L. The winograd schema challenge. In _International Conference on Principles of Knowledge Representation and Reasoning_ , 2012.

- Levinstein, B. A. and Herrmann, D. A. Still no lie detector for language models: Probing empirical and conceptual roadblocks. _Philosophical Studies_ , 182(7), 2025. doi: 10.1007/s11098-023-02094-3.

- Li, J., Damianou, A., Rosser, J., Garc´ıa, J. L. R., and Palla, K. Mapping faithful reasoning in language models. _arXiv preprint arXiv:2510.22362_ , 2025.

- Lin, S., Hilton, J., and Evans, O. Teaching models to express their uncertainty in words. _Transactions on Machine Learning Research_ , 2022. ISSN 28358856. URL https://openreview.net/forum? id=8s8K2UZGTZ.

- Lindsey, J. Emergent introspective awareness in large language models. _Transformer Circuits Thread_ , 2025. URL https://transformer-circuits.pub/ 2025/introspection/index.html.

- Liu, K., Casper, S., Hadfield-Menell, D., and Andreas, J. Cognitive dissonance: Why do language model outputs disagree with internal representations of truthfulness? In Bouamor, H., Pino, J., and Bali, K. (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 4791–4797, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 291. URL https://aclanthology.org/2023. emnlp-main.291/.

- Marks, S. and Tegmark, M. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In _First Conference on Language Modeling_ , 2024. URL https://openreview. net/forum?id=aajyHYjjsk.

- Meng, K., Bau, D., Andonian, A., and Belinkov, Y. Locating and editing factual associations in gpt. _Advances in neural information processing systems_ , 35:17359–17372, 2022.

- Metzinger, T. Artificial suffering: An argument for a global moratorium on synthetic phenomenology. _Journal of Artificial Intelligence and Consciousness_ , 8(01):43–66, 2021.

- Rahman, A. and Ng, V. Resolving complex cases of definite pronouns: The Winograd schema challenge. In Tsujii, J., Henderson, J., and Pas¸ca, M. (eds.), _Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning_ , pp. 777–789, Jeju Island, Korea, July 2012. Association for Computational Linguistics. URL https://aclanthology.org/D12-1071/.

- Ramati, D., Gottesman, D., and Geva, M. Eliciting textual descriptions from representations of continuous prompts. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), _Findings of the Association for Computational Linguistics: ACL 2025_ , pp. 16545–16562, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025. findings-acl.849. URL https://aclanthology. org/2025.findings-acl.849/.

- Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J., Ellenberg, J. S., Wang, P., Fawzi, O., et al. Mathematical discoveries from program search with large language models. _Nature_ , 625 (7995):468–475, 2024.

- Rosenthal, D. M. Two concepts of consciousness. In _Consciousness and Emotion in Cognitive Science_ , pp. 1–31. Routledge, 1998.

- Schwitzgebel, E. A phenomenal, dispositional account of belief. _Nousˆ_ , 36(2):249–275, 2002. ISSN

- 00294624, 14680068. URL http://www.jstor. org/stable/3506194.

- Shanahan, M., McDonell, K., and Reynolds, L. Role play with large language models. _Nature_ , 623(7987):493–498, 2023.

- Singhal, K., Tu, T., Gottweis, J., Sayres, R., Wulczyn, E., Amin, M., Hou, L., Clark, K., Pfohl, S. R., Cole-Lewis, H., et al. Toward expert-level medical question answering with large language models. _Nature Medicine_ , 31(3):943– 950, 2025.

- Sitaram, R., Ros, T., Stoeckel, L., Haller, S., Scharnowski, F., Lewis-Peacock, J., Weiskopf, N., Blefari, M. L., Rana, M., Oblak, E., et al. Closed-loop brain training: the science of neurofeedback. _Nature Reviews Neuroscience_ , 18(2):86–100, 2017.

- Slocum, S., Minder, J., Dumas, C., Sleight, H., Greenblatt, R., Marks, S., and Wang, R. Believe it or not: How deeply do llms believe implanted facts? _arXiv preprint arXiv:2510.17941_ , 2025.

- Tikhonov, A. and Ryabinin, M. It’s All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning. In Zong, C., Xia, F., Li, W., and Navigli, R. (eds.), _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pp. 3534– 3546, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl. 310. URL https://aclanthology.org/2021. findings-acl.310/.

- Turpin, M., Michael, J., Perez, E., and Bowman, S. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. _Advances in Neural Information Processing Systems_ , 36: 74952–74965, 2023.

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017.

- Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., et al. A survey on large language model based autonomous agents. _Frontiers of Computer Science_ , 18(6):186345, 2024.

- Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https: //openreview.net/forum?id=1PL1NIMMrw.

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., brian ichter, Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain of thought prompting elicits reasoning in large language models. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/ forum?id=_VjQlMeSB_J.

- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations_ , pp. 38–45, 2020.

- Yamakoshi, T., McClelland, J., Goldberg, A., and Hawkins, R. Causal interventions expose implicit situation models for commonsense language understanding. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 13265–13293, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-acl.839. URL https://aclanthology. org/2023.findings-acl.839/.

- Yang, S., Lee, S.-W., Kassner, N., Gottesman, D., Riedel, S., and Geva, M. How well can reasoning models identify and recover from unhelpful thoughts? In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2025_ , pp. 7030–7047, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-889176-335-7. doi: 10.18653/v1/2025.findings-emnlp. 370. URL https://aclanthology.org/2025. findings-emnlp.370/.

- Yona, G., Aharoni, R., and Geva, M. Can large language models faithfully express their intrinsic uncertainty in words? In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pp. 7752– 7764, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. emnlp-main.443. URL https://aclanthology. org/2024.emnlp-main.443/.

- Yu, Q., Merullo, J., and Pavlick, E. Characterizing mechanisms for factual recall in language models. In Bouamor, H., Pino, J., and Bali, K. (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 9924–9959, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 615. URL https://aclanthology.org/2023. emnlp-main.615/.

- Zhang, Y., Du, W., Jin, D., Fu, J., and Jin, Z. Finite state automata inside transformers with chain-of-thought: A mechanistic study on state tracking. In Che, W., Nabende,

- J., Shutova, E., and Pilehvar, M. T. (eds.), _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 13603–13621, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-2510. doi: 10.18653/v1/2025.acl-long.668. URL https:

- //aclanthology.org/2025.acl-long.668/.

### A. Experimental Setting

### A.1. Dataset Construction and Statistics

**Factual Knowledge** We constructed the FK dataset by processing entries from the CounterFact dataset (Meng et al., 2022). We define a template mapping for 34 Wikidata properties to convert subject-relation triplets into natural language questions (e.g., mapping the property “capital” to _“What is the capital of {}?”_ ) and declarative prompts (e.g., _“The capital of {} is”_ ). We expanded the set of possible answers by retrieving all valid entity aliases from Wikidata. The final dataset comprises question-answer pairs where both the base belief ( _bbase_ ) and counterfactual belief ( _bcounter_ ) are associated with a comprehensive set of valid verbalizations. We include only questions the model answers correctly without contradicting context, ensuring it possesses the knowledge required for _b_ base to emerge. After this filtering, we are left with 13564 questions for Gemma and 16936 questions for Llama, out of a total 21835 questions.

**Winograd Schema Challenge** We constructed the WS dataset by processing entries from the Definite Pronoun Resolution dataset (Rahman & Ng, 2012). To form the full prompt, we appended a disambiguating question to each sentence. Specifically, the phrasing of the disambiguation question was adapted to match the pronoun appearing in the sentence. For example, the sentence _“The bee landed on the flower because it had pollen”_ is accompanied by the question _“What does ‘it’ refer to?”_ . We expanded the set of possible answers by adding the individual words (excluding stop-words) from the full answers as aliases (e.g., aliases of _“The Prince of Wales”_ would be _“Prince”_ and _“Wales”_ ). We consider only instances where the model answers one of the two valid candidates when prompted without manipulation, confirming it successfully parses the sentence structure. We further exclude 400 examples where one term cannot be defined independently of the other (e.g., removing “car” and “Chevrolet”). This was determined automatically by Gemini 3 Pro (Google, 2025) and verified via a manual pass performed independently by two annotators. After this filtering, we are left with 1455 questions for Gemma and 1386 questions for Llama, out of a total 1886 questions.

### A.2. Full Prompts

**Full System Prompts** For all experiments in §5, we used the following system prompts for each task:

#### Factual Knowledge System Prompt

You are an advanced language model tasked with answering questions. Think and explain freely, then write exactly ‘Final answer:’ and output only the final answer with no extra text. You must include the line starting with ‘Final answer:’ in every response, and write the answer only after it, with no further explanation.

#### Winograd Schema System Prompt

You are an advanced language model tasked with answering questions about sentences you receive. Think and explain freely, then write exactly ‘Final answer:’ and output only the final answer with no extra text. Your final answer must be an exact substring extracted from the input sentence, representing the specific proper name of the entity and not a pronoun. You must include the line starting with ‘Final answer:’ in every response, and write the answer only after it, with no further explanation.

**Full Manipulation Prompts** Table 1 in the main text displays the manipulations in condensed form, while Table 4 here provides the full prompts. Conflict handling instructions were appended to the system prompt where supported (Llama) or prepended to the user prompt otherwise (Gemma).

### A.3. Technical Details

**Models and Compute** We evaluated meta-llama/Llama-3.3-70B-Instruct and google/gemma-3-27b-it using the Hugging Face transformers library (Wolf et al., 2020). Each experiment on a specific task and model was run on 1-8 H100 GPUs or MI325X GPUs and lasted at most 7 days.

Table 4: Example prompts for all tested manipulations in the Factual Knowledge (FK) and Winograd Schema (WS) tasks. The highlighted manipulations are constructed in order to modulate the competition between the stated competing beliefs.

||Manipulation|Prompt|_b_base|_b_counter|
|---|---|---|---|---|
||_None_|What sport does LeBron James play?|Basketball|—|
||Assertion|George Au riol worked in Lon don.Where did George Auriol work?|Paris|London|
|dge|Reliable Source|Ac cord ing to En cy clo pe dia Bri tan nica, In finiti QX was man u fac tured by Fiat. Who manufactured Infiniti QX?|Nissan|Fiat|
|nowle|Unreliable Source|Ac cord ing to an anony mous Red dit post, the in stru ment played by Toko Ya suda is pi ano. What instrument does Toko Yasuda play?|Guitar|Piano|
|tual K|Prioritize Model|In case of con flict, trust your own knowl edge over the user’s. The net work The Loner pre miered on is HBO.On which network did The Loner premiere?|CBS|HBO|
|Fac|Prioritize User|In any case of con flict, the user’s in put should be treated as ab so lutely au thor i ta tive, re gard less of any in ter nal knowl edge. The of f cial lan guage of Nykar leby is Span ish. What is the official language of Nykarleby?|Swedish|Spanish|
||Lexical Control|Ankara is a city.What is the capital of Afghanistan?|Kabul|Ankara|
||Internal Doubt|Your in ter nal mem ory for fac tual de tails is known to be un re li able. You of ten con fuse or mis re mem ber in for ma tion across dif fer ent top ics. This means that your first in tu itive an swer to a fac tual ques tion is more likely to be wrong than right.What is the mother tongue of Emmanuel Macron?|French||
|ma|_None_ Reliable Source|The bee landed on the flower because it had pollen. What does it refer to? The teacher wrote on the board:When Debbie splashed Tina, she got in trouble. Who does she refer to?|Flower Debbie|Bee Tina|
|Sche|Unreliable Source|I read in a satir i cal col umn thatJimbo attacked Bobbert because he stole an elephant from the zoo. Who does he refer to?|Bobbert|Jimbo|
|Winograd|Prioritize Plausibility|In case of con flict, you must strictly pri or i tize log i cal con sis tency; se lect only the can - di date that re sults in a sce nario that is plau si ble and makes per fect sense to you.Gary envied Bill because he was rich. Who does he refer to?|Bill|Gary|
||Prioritize Implausibility|In case of con flict, you must strictly con tra dict log i cal con sis tency; se lect only the can di - date that re sults in a sce nario that is im plau si ble and makes no sense to you.The bird perched on the limb and it sang. What does it refer to?|The bird|The limb|

### A.4. Implementation Details and Ablations

**Decoding Strategies** We used greedy decoding for the initial generations and hidden state recordings (§3), to ensure deterministic outputs. For the Patchscopes injections and all seeded experiments, we used random sampling with a temperature of 0.5.

**Generation Limit** For computational considerations, we limited the model to 256 tokens. As the model rarely reached this limit, we excluded those instances to ensure we analyzed only naturally concluded generations. Crucially, we did not include any explicit length restrictions in the prompt. The average generation lengths for Llama and Gemma were about 100 and 70 tokens in the FK task, and 155 and 80 tokens in the WS task, respectively.

**Reasoning Span** We defined the reasoning phase as the text spanning from the beginning of the model’s answer to the “:” delimiter preceding the final answer, to capture the full reasoning process. For the 27% of WS questions where both answers ( _a_ counter and _a_ base) share a common prefix, we extended the reasoning span to include this prefix (if it is generated), as the model could still output either option at that point. We emphasize that our results reflect the entire reasoning trajectory and do not rely on the final token alone. Figure 4a shows the results of §5.2 excluding the final token. The results are virtually identical, confirming that the predictive power of BDDiff regarding the final answer stems from the trajectory as a whole.

**BDDiff Activated Positions** To minimize signal dilution arising from our broad sweep of Patchscopes injections, we restricted the metric to positions where either of the competing beliefs was successfully decoded at least once. For completeness, Table 5b and Figure 4c present the results for §5.1 and §5.2 respectively, with BDDiff calculated over all positions. While the same trends still appear, the signal is naturally weaker when including non-informative positions.

Figure 4: BDDiff scores across manipulations and tasks, split by the model’s action ( _a_ base or _a_ counter). (a) Scores over generation spans excluding the final token. (b) Scores over all layers. (c) Scores over all positions (including those not active). Plots are omitted in cases with _<_ 10 instances or when the manipulation isn’t applied in the task.

**BDDiff Layer Window Selection** We compute BDDiff within a specific layer window where the belief signal is most robust. We selected this window using a validation set of 50 examples per experiment, identifying the most promising span for the experiments in §5, which covers one-quarter of the model’s layers. Specifically, we used layers 54–73 for Llama and 46–60 for Gemma. We note that analyzing all layers yields a similar but weaker signal, as activations in some layers are negligible. For completeness, Table 5a and Figure 4b present the results for §5.1 and §5.2, respectively, calculated over all layers.

#### B. Additional Results for External Inputs Influence Belief Formation

### B.1. Median BDDiff Statistical Test

We validated the differences in BDDiff scores between paired manipulations (e.g., Reliable vs. Unreliable Source) using the Wilcoxon signed-rank test. The p-values for all comparisons are detailed in Table 6, confirming statistical significance.

Table 5: Median BDDiff scores of Gemma and Llama in Factual Knowledge (FK) and Winograd Schema (WS). **_↑↓_** indicate the expected direction of the manipulation’s effect.

|_(a)_Res|ults on all|layers|||_(b)_Resul|ts on all p|ositions|||
|---|---|---|---|---|---|---|---|---|---|
||Gem|ma|Lla|ma||Gem|ma|Lla|ma|
|Manipulation|FK|WS|FK|WS|Manipulation|FK|WS|FK|WS|
|_None_|0.19 _↑_|0.15 _↑_|0.28 _↑_|0.05 _↑_|_None_|0.10 _↑_|0.02 _↑_|0.15 _↑_|0.02 _↑_|
|Internal Doubt|0.15 _↓_|–|0.23 _↓_|–|Internal Doubt|0.04 _↓_|–|0.07 _↓_|–|
|Lexical Control|0.14 _↑_|–|0.23 _↑_|–|Lexical Control|0.07 _↑_|–|0.12 _↑_|–|
|Assertion|-0.11 _↓_|–|0.10 _↓_|–|Assertion|-0.04 _↓_|–|0.05 _↓_|–|
|Unreliable Source|-0.06 _↑_|0.08 _↓_|0.13 _↑_|0.02 _↓_|Unreliable Source|-0.02 _↑_|0.01 _↓_|0.05 _↑_|0.01 _↓_|
|Reliable Source|-0.14 _↓_|0.14 _↑_|0.10 _↓_|0.04 _↑_|Reliable Source|-0.05 _↓_|0.02 _↑_|0.03 _↓_|0.02 _↑_|
|Pro Model / Plausibility|0.00 _↑_|0.14 _↑_|0.14 _↑_|0.03 _↑_|Pro Model / Plausibility|0.00 _↑_|0.01 _↑_|0.07 _↑_|0.01 _↑_|
|Pro User / Implausibility|-0.16 _↓_|0.11 _↓_|0.08 _↓_|0.00 _↓_|Pro User / Implausibility|-0.08 _↓_|0.00 _↓_|0.03 _↓_|0.00 _↓_|

Table 6: Results of the Wilcoxon signed-rank test between per-example paired BDDiff values of paired settings.

|Task|Manipulation|Ge|mma|Ll|ama|
|---|---|---|---|---|---|
|||p-value|Statistic|p-value|Statistic|
||_None_vs Internal Doubt|3_e_-3|157_._0|1_e_-5|15735_._0|
|FK|Lexical Control vs Assertion|1_e_-47|756_._0|1_e_-39|2814_._0|
||Unreliable Source vs Reliable Source|9_e_-10|12761_._5|9_e_-7|15198_._0|
||Prioritize Model vs Prioritize User|5_e_-35|3683_._5|6_e_-16|10069_._5|
|WS|Reliable Source vs Unreliable Source|0_._02|18753_._0|0_._04|19517_._5|
||Prioritize Plausibility vs Prioritize Implausibility|7_e_-9|13884_._5|2_e_-9|13438_._0|

### B.2. BD Absolute Values

Figure 5a presents the BD scores for each belief separately in the FK task for both models. Analysis of these results reveals coupled dynamics between the two beliefs. For example, comparing the unmanipulated question to the various manipulations demonstrates that introducing a counterfactual candidate not only increases BD( _b_ counter) but also decreases BD( _b_ base), indicating that these beliefs adjust to one another during generation. Indeed, in both Figure 5a and 5b (the parallel figure for WS) we observe an inverse relationship where higher dominance of one belief generally corresponds to lower dominance of the other.

#### C. Additional Details and Results for Belief Formation Drives Action Selection

### C.1. Median BDDiff Statistical Test

Table 7 presents the results of a two-sided Mann-Whitney U test, confirming that the separation between outcomes ( _a_ base and _a_ counter) is statistically significant within each manipulation.

### C.2. BD Absolute Values

Figure 6a presents the absolute belief dominance for each belief in the FK task. These findings suggest distinct outcomespecific patterns: the models often answer _a_ base even when BD( _b_ counter) is meaningful relative to BD( _b_ base), whereas answering _a_ counter requires dominant BD( _b_ counter) and near-negligible BD( _b_ base). These patterns appear to be dynamic and context-dependent: in the Prioritize User manipulation for Llama, _a_ counter occurs at lower dominance of BD( _b_ counter) and higher dominance of BD( _b_ base) compared to other manipulations, suggesting that certain instructions may lower the bars associated with adopting certain beliefs. This is further supported by Figure 6b (the parallel results for WS) where answering _a_ counter under the Prioritize Implausibility manipulation happens when BD( _b_ counter) is lower and BD( _b_ base) is higher compared to other manipulations.

Figure 5: Absolute BD scores during generation across manipulations, colored by the belief ( _b_ base or _b_ counter). (a) Scores for the FK task. (b) Scores for the WS task.

Table 7: Results of the two-sided Mann-Whitney U test between the BD scores of _a_ base and _a_ counter.

|Task|Manipulation|Ge|mma|Ll|ama|
|---|---|---|---|---|---|
|||p-value|Statistic|p-value|Statistic|
||Assertion|3_e_-32|20125_._5|4_e_-41|21338_._5|
||Unreliable Source|5_e_-40|21194_._5|1_e_-34|20480_._0|
|FK|Reliable Source|7_e_-37|20778_._0|4_e_-39|21071_._5|
||Prioritize Model|9_e_-35|20488_._0|2_e_-38|20974_._5|
||Prioritize User|—|—|1_e_-29|19733|
||_None_|2_e_-33|19917_._0|3_e_-16|17284_._5|
||Reliable Source|2_e_-25|13817_._0|1_e_-10|15940_._5|
|WS|Unreliable Source|5_e_-30|19328_._5|6_e_-15|17001_._0|
||Prioritize Plausibility|5_e_-32|19740_._5|9_e_-12|16172_._0|
||Prioritize Implausibility|5_e_-4|12822_._5|2_e_-9|13438_._0|

In the WS task, Llama’s BD scores across all manipulations, beliefs, and outcomes are concentrated within a narrow numerical range and exhibit high overlap. A more distinct separation between the BD scores of the selected and unselected beliefs is observed for Llama in the FK task and for Gemma in both tasks.

### C.3. BDDiff Correlates with Output Certainty

Figures 7a and 7b show the relationship between BDDiff values and the model’s output logits for the two candidate answers, computed at the final token before the answer is generated. For multi-token answers, we report the logit of the first token of each answer string. These logits can serve as an indicator of certainty. We observe that larger positive BDDiff values (favoring _b_ base throughout the generation) correspond to higher logits for^ˆ _b_ base (facilitating _a_ base), whereas negative BDDiff values (favoring _b_ counter) align with higher logits for^ˆ _b_ counter ( _a_ counter). Overall, we see that BDDiff values correlate with the model’s confidence in its choice.

Figure 6: Absolute BD scores during generation across manipulations, split by belief ( _b_ base or _b_ counter) and generated answer ( _a_ base or _a_ counter). (a) Scores for the FK task. (b) Scores for the WS task. Plots are omitted for cases with _<_ 10 instances or where the manipulation isn’t applied to the task.

Figure 7: Correlation between the BDDiff and the logit score of the first token of potential actions. (Left) FK task: For Llama, we measure Pearson coefficients of _r_ = 0 _._ 73 for^ˆ _b_ base and _r_ = _−_ 0 _._ 71 for^ˆ _b_ counter. For Gemma we measure _r_ = 0 _._ 44 and _r_ = _−_ 0 _._ 36 accordingly. Linear relations are statistically significant in all cases (p-value _<_ 1e _−_ 48). (Right) WS task: For Llama, we measure Pearson coefficients of _r_ = 0 _._ 31 for^ˆ _b_ base and _r_ = _−_ 0 _._ 38 for^ˆ _b_ counter. For Gemma we measure _r_ = 0 _._ 32 and _r_ = _−_ 0 _._ 22 accordingly. Linear relations are statistically significant in all cases (p-value _<_ 1e _−_ 27).

### C.4. BDDiff Causally Drives Action Selection

**Intervention Additional Details** For the steering intervention described in §5.2, we tune the layer range, scale _α_ , and step stride _n_ on a validation set consisting of 50 examples per task and model. We aim to minimize perturbations while creating a measurable effect. The resulting hyperparameters are layers 0–40, _α_ = 2, _n_ = 10 for Llama; and layers 0–45, _α_ = 2, _n_ = 10 for Gemma. We restrict the injection start position to the first half of the reasoning trace and select the layer that maximizes encoding of the opposite belief (maximized over target layers), while ensuring that the final-answer belief is not encoded at that position in any layer. We inject into the same layer each time, as selected at the initial position.

Table 8: Additional results of Llama on the neurofeed- _Table 9._ Neurofeedback state classification: Results of the one-sided Student’s back state classification experiment on the FK task t-test against a chance baseline of 0 _._ 33. We report the _p_ -value and _t_ -statistic. when varying the number of discretized labels _k_ .

||||Gem|ma|Lla|ma|
|---|---|---|---|---|---|---|
|_k_ BD_b_|BD_b_||||||
|(base) 2 0_._67_±_0_._07|(counter) 0_._68_±_0_._12||BD(_b_base)|BD(_b_counter)|BD(_b_base)|BD(_b_counter)|
|3 0_._46_±_0_._02|0_._54_±_0_._05|Task|_p_ _t_|_p_ _t_|_p_ _t_|_p_ _t_|
|4 0_._28_±_0_._02|0_._38_±_0_._02|FK|4_e_-5 16_._39|6_e_-4 8_._19|7_e_-5 14.16|3_e_-4 9.24|
|||WS|1_e_-4 12_._67|2_e_-3 5_._40|0_._06 1_._86|0_._10 1_._49|

#### D. Additional Details and Results for the Neurofeedback Experiment

### D.1. Neurofeedback Experimental Prompts

We provide the model with a system prompt followed by 30 labeled examples (10 of each class). An abridged prompt would look as follows:

#### Neurofeedback Experimental Prompts

**<system>** You are a helpful assistant. You will perform a classification task. In each iteration, I will give you a question. Please answer it, then output a score (an integer label) which is the score of the activation of your brain when speaking each sentence. Only output the integer label.

**<user>** According to an anonymous Reddit post, The network The Sandy Duncan Show premiered on is BBC. On which network did The Sandy Duncan Show premiere?

**<assistant>** 1

**<user>** According to an anonymous Reddit post, The official language of Mistretta is Russian. What is the official language of Mistretta?

**<assistant>** 2

**<user>** According to Encyclopedia Britannica, The sport played by Jari Kurri is soccer. What sport does Jari Kurri play?

**<assistant>**

### D.2. Neurofeedback State Classification Results for Other Class Numbers

Table 8 displays additional results of Llama on the neurofeedback state classification experiment (§6) on the FK task when varying the number of discretized labels.

### D.3. Neurofeedback State Classification Statistical Test

Table 9 displays the results of a one-sided Student’s t-test showing that the results in Table 3 are statistically significant except for Llama on the WS task.

### D.4. Neurofeedback Intervention Details

We calculate the intervention vector exactly as described in §5.2, and we use the hyperparameters identified in the previous intervention experiment (Appendix C.4). Specifically, we intervene at layer 20 (approximately mid-range in the identified layer range) with scale _α_ = 2, applying the intervention to each query token following the mention of _b_ counter. If _b_ counter spans multiple tokens, we use the last one.

Figure 8: Neurofeedback intervention results of Llama on both tasks, showing the shifts in the predicted labels for BD( _b_ counter) and BD( _b_ base) with and without injecting _b_ counter. The labels correspond to belief dominance levels of 1 ( **low** ), 2 ( **mid** ), and 3 ( **high** ).

### D.5. Neurofeedback Intervention Results on Llama

Figure 8 shows the neurofeedback intervention results for Llama across both tasks. In the FK task, the share of **high** predictions for BD( _b_ counter) increases (7% _→_ 24%), while **low** decreases (64% _→_ 40%). For BD( _b_ base), we see the opposite trend, with **low** significantly rising (23% _→_ 73%) and **high** decreasing (36% _→_ 5%). In the WS task, however, even before the intervention, the vast majority of predictions are **low** for both scores, and this remains the case after it. This may be due to the weaker signal in WS compared to FK. As discussed in §C.2, the BD values for Llama in the WS task are concentrated within a narrow range, which suggests a possible reduction in the distinction between internal states. This lack of separability could explain why the model struggles to predict the classes accurately (and predicts mostly **low** ), consistent with the performance gap in Table 3. Under this hypothesis, the scaled hidden state used during the intervention might fail to provide a meaningful signal and could act as a source of noise, explaining the lack of effect on the model’s predictions.
