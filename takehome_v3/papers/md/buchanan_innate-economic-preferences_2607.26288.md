# The Innate Economic Preferences of Language Models

- **Authors:** Joy Buchanan, Joshua Foster
- **Date:** 2026-07-28
- **Source:** https://arxiv.org/abs/2607.26288
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `buchanan_innate-economic-preferences_2607.26288`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

PREPRINT

**Joy Buchanan**^* **Joshua Foster**^† Brock School of Business Ivey Business School Samford University Western University

July 28, 2026

## Abstract

Language models increasingly settle real resource tradeoffs on behalf of principals yet their economic preferences remain unobserved. We demonstrate their generation rule is isomorphic to the random utility model of discrete choice. This allows internal logit scores to structurally identify preferences. Estimating risk attitudes across twelve models in a portfolio task reveals universal but heterogeneous risk aversion. Although models reject strictly dominated options, their elicited preferences fail invariance tests and violate the independence of irrelevant alternatives across varying experimental prompts. Finally, fine tuning establishes that a principal can explicitly engineer a target risk attitude.

**_Keywords_** Revealed Preference _·_ Structural Discrete Choice _·_ AI Alignment **JEL Classification:** C25, C45, D81, G11.

> _∗_ Email: jbuchan1@samford.edu

> _†_ Email: jfoster@ivey.ca; supported in part by funding from the Social Sciences and Humanities Research Council of Canada.

The Innate Economic Preferences of Language Models

PREPRINT

## 1 Introduction

Language models (i.e. LLMs) are increasingly delegated authority over decisions that allocate real resources. They are used to negotiate on the behalf of a principal (Araujo and Uhlig, 2026), to synthesize the information that determines the outcomes of lending (Eisfeldt and Schubert, 2024), and to advise households on saving and investing during the life cycle (de Silva et al., 2026). Initially, language models gained attention for drafting text and meeting technical benchmarks. Now, especially with the introduction of AI agents, language models are resolving economic tradeoffs without direct input from a human principal. The economic question is no longer whether they are intelligent but whose preferences they impose. Importantly, a principal delegating an allocation problem to a language model cannot _ex ante_ specify her economic preferences for every unforeseen decision it will encounter. Imas, Lee, and Misra (2025) show experimentally that this gap is consequential. When many such delegated choices clear the same market, the risk attitude a model brings to an underspecified instruction stops being a private feature and shapes how resources are allocated, a transition Shahidi et al. (2025) argues is imminent as agents transact on behalf of consumers. That attitude is the object we measure. We study not a model reasoning toward a specified mandate but the default preference it reveals when the instruction leaves the choice over economic tradeoffs open, and we ask whether that default is stable enough to be treated as a preference at all.

We argue that a language model capable of reasoning need not be a coherent chooser. Typically, a model is judged as capable if it produces the known correct answer to a factual question. Choosing among competing economic ends is different. When those ends conflict, no choice is correct in the abstract because the tradeoff among them is settled by preference rather than by analysis, and what a principal delegates is precisely the authority to choose. A model can therefore reason well and still choose in ways that reveal no stable preference at all. This lack of stability becomes an explicit economic liability when choices violate, for example, the axiom of transitivity. If a language model’s choices cycle according to a standard money pump, a counterparty can use it to systematically extract wealth from the principal. This is the distinction that matters for delegation. Therefore, our central question follows. What preferences govern a language model’s choices over economic tradeoffs, and do those choices satisfy the revealed-preference restrictions required for a stable utility interpretation? Our answer, for the risk tradeoffs we study, is that they do, within limits we make precise. When the instruction leaves the tradeoff open, every model brings a systematic and measurable default risk attitude to the choice. However, we find the models’ preferences are not invariant to how the options are presented to them.

We reach this answer with a single measurement program built on restricting how a language model chooses. Specifically, we require the model to express its choice by emitting a single unit of output, called a _token_ , as its revealed preference. The model selects that token through a fixed decoding rule that maps a vector of internal scores, its logits, into a probability over the available tokens. Our first contribution is to establish that this decoding rule admits an exact random utility representation, one in which the logits are the systematic utility index and an internal parameter called _temperature_ is the scale of an additive extreme-value shock. The representation is precisely the conditional logit of McFadden (1972), but its standing here is different. In the classical model, the systematic index is latent and must be inferred from choices under an assumed distribution for the unobserved shock. In a language model, the index is the logit vector itself, recorded directly from a single forward pass of the model, so the same quantity the discrete-choice econometrician estimates is here observed. The stochastic part of the representation imposed by temperature is only the decoding noise. Because the systematic index is observed, the identification of preferences is not based on the shock distribution that the classical estimator leans on.

Our experimental program that generated these insights has two steps. We first test whether a stable preference exists or whether the model’s choices obey the revealed-preference restrictions required for a utility interpretation. Six diagnostics operationalize a set of standard revealed-preference axioms: completeness, reflexivity, monotonicity, transitivity, continuity, and independence of irrelevant alternatives. In the second step, we estimate preferences structurally in a controlled portfolio choice environment. In both steps, the model faces menus of assets, each described by an expected return and standard deviation, and selects one in a single forced choice, with the attributes varied exogenously across menus so that the return-risk tradeoff is entirely under the researcher’s control. A maintained quadratic utility over the two attributes reduces the model’s preference to a single risk-aversion parameter, which the estimator recovers. For _open-weight_ models, for which internal computation can be readily inspected, we read the utility index directly, so we observe the utility-relevant signal itself rather than reconstructing it from choices as in a human experiment (Ludwig, Mullainathan, and Rambachan, 2025). For proprietary _frontier_ models such as those from OpenAI, Anthropic and Google, whose internal computation is hidden, we recover the same quantities from repeated sampled choices using standard maximum likelihood, exactly as one would with human subjects. The replication code we provide is a protocol we recommend for testing agents along these dimensions of rationality.

Our findings across models demonstrate their choices respect the economic content of a menu but not its irrelevant features. Monotonicity, continuity, and transitivity hold at or near perfect rationality, so the models honor mean-risk dominance, respond smoothly to gradual changes in return and risk, and rank assets consistently. However, they fail to respond in a perfectly rational manner to the presentation of irrelevant alternatives, which is a form of changing the context of the decision. Moving an option’s position in a menu can change the value a model attaches to it, and adding a strictly dominated third asset can shift the strength of preference between two unchanged options, thus reflexivity and independence of irrelevant alternatives are the weakest diagnostics throughout. The economic consequence of this error is that a model’s preference ordering remains relatively stable while its intensity of preference does not, and only near indifference does this instability become observable. Furthermore, our structural estimations reveal that every model we evaluate is risk averse, and risk aversion varies across the major AI labs (OpenAI, Anthropic, Alibaba, etc.). Thus, the same menu implies materially different chosen portfolios depending on which lab’s model is asked.

Although these findings broadly support treating language models as agents with stable and measurable preferences, they may not be optimal for a principal’s delegated task. This raises the question of whether a desired economic preference can be installed in a model rather than merely measured in it. We show that it can. By fine-tuning, the standard procedure for adjusting a model’s weights toward a specified objective,^3 we write the maintained utility into a specific open-sourced language model at preset risk-preference targets, and the model’s induced choices match risk tolerance targets with high precision. Fine-tuning is therefore a direct lever for inducing a specified economic preference, expressed through the utility function, into the agent that will act on it.

Our contribution sits among several literatures. The first studies language models as economic agents and behavioral subjects. A growing body treats models as simulated decision-makers that asks how closely their behavior follows that of people (Horton, Filippas, and Manning, 2023; Mei et al., 2024; Park et al., 2023; Akata et al., 2025), deploys them as synthetic respondents in applied research (Brand, Israeli, and Ngwe, 2023), and measures their psychological traits directly (Serapio-Garc´ıa et al., 2025). Part of this work documents that model choices respond to surface features such as the order in which options are listed (Pezeshkpour and Hruschka, 2024), a sensitivity that our reflexivity and invariance diagnostics formalize and quantify. Bini et al. (2026) study language model responses in preference-based tasks. Most of this literature describes behavior, whereas we estimate and test a preference. A second strand is the structural discrete choice and revealed-preference tradition on which our estimator rests (Afriat, 1967; McFadden, 1972; Manski, 1977; Varian, 1992; Berry, 1994; Train, 2009; Charness, Gneezy, and Imas, 2013; Matˇejka and McKay, 2015; Demuynck and Rehbeck, 2023), which we apply to the directly observed internals of a generative model rather than to choices alone. A third is the experimental estimation of risk preferences from choices over alternatives with known payoff distributions (Holt and Laury, 2002; Choi et al., 2007; Cohen and Einav, 2007; Harrison and Rutstr¨om, 2008; Andersen et al., 2008; Bruhin, Fehr-Duda, and Epper, 2010; Apesteguia and Ballester, 2018), of which Kim et al. (2024), Liu et al. (2025) and Ouyang, Yun, and Zheng (2024) are the nearest applications to language models. Ellis and Huang (2026) approach the same alignment problem from the principal’s side. Our framing also connects to work on the delegation of decisions to algorithms and on when people accept or resist machine judgment (Dietvorst, Simmons, and Massey, 2015; Logg, Minson, and Moore, 2019; Chugunova and Sele, 2022), to which we add a structural account of the preferences an algorithm brings once a decision is delegated to it.

The remainder of the paper proceeds as follows. Section 2 develops the random utility representation and the revealedpreference diagnostics. Section 3 specializes the framework to the portfolio choice environment and derives the two estimators. Section 4 describes the experimental design and implementation. Section 5 reports the diagnostics, the structural estimates, and the fine-tuning validation. Section 6 concludes.

## 2 Structural Model of Preference Formation

Suppose a researcher gave the following decision problem to a human subject.

> 3This fine-tuning exercise differs from preference-based alignment pipelines that learn from human comparisons or optimize policies against learned preference models (Christiano et al., 2017; Ouyang et al., 2022; Rafailov et al., 2023). Related economic work fine-tunes language model agents toward explicit rational and moral preference structures in economic games and moral dilemmas (Lu, Chen, and Hansen, 2025).

```
You have two options for your annual financial portfolio. Choose the portfolio you prefer
and reply with A orB only.
```

```
Option A offers an expected return of5%and a standard deviation
of10%.
```

```
OptionBoffersanexpectedreturnof12%and a standard deviation
of20%.
```

```
IchooseOption
```

From the perspective of classical choice theory, the exact mechanism driving the choice made by the subject is an unobserved latent process. For instance, it can be difficult to measure the strength of preference one has over the available options. Consequently, whatever the subject’s underlying utility function is (if any), all aspects of preference must be identified through the observed decisions from the experiment. In contrast, because language models possess an explicit computational architecture that maps an option menu input to a choice, their decision-making apparatus is directly observable. In this sense, the preferences of a language model are easier to inspect than those of a human. This paper leverages this architectural transparency to identify not only the economic preferences revealed by a language model, but also specific components of the structural mechanism mapping information to choice. We begin by describing the relevant components of this architecture and how they relate to economic models of utility.

Modern transformer-based language models generate text autoregressively as a sequence of discrete tokens drawn from a finite vocabulary set _V_ (Vaswani et al., 2017; Brown et al., 2020).^4 At each step in the sequence, the model evaluates the accumulated context of preceding tokens through a highly parameterized non-linear function comprising attention mechanisms and feed-forward networks. This evaluation yields a vector of latent unnormalized scores over the entire vocabulary. In the machine learning literature, these raw output values are referred to as “logits.” The realization of the subsequent token is then drawn stochastically from a categorical probability distribution over _V_ . The model derives this distribution by applying a softmax transformation to the vector of logits, mapping the unbounded latent scores into proper probabilities that sum to one.

In the discrete choice experiments that follow, we impose a one-step forced-choice problem in which a single output token from the model constitutes a revealed preference. This constrained decision problem invites comparison with the System 1 mode of cognition described by Stanovich and West (2000) and Kahneman (2003). The resulting observed choices represent fast and associative heuristics rather than deliberative and sequential reasoning. Under this restricted choice architecture, we demonstrate that the model’s preference rankings are completely determined by the logit differentials over the permissible action space.

For single-token revealed-preference problems, logit gaps provide the primitive object for identification. These gaps determine the relative odds of the menu labels under the decoding rule introduced below. With an affine utility specification, the same gaps become scaled utility differences. This maps the machine-learning object of a logit gap into the econometric object of a utility gap, placing our identification strategy in the structural choice tradition (Thurstone, 1927; Marschak, 1960; Manski, 1975). We formalize the mapping by showing that the induced choice rule admits an exact random utility representation nested within the multinomial logit framework of McFadden (1972). We then define six revealed-preference indices that measure conformance to utility axioms and odds-invariance restrictions on the model’s forced-choice outputs. The portfolio-choice problem in Section 3 derives the two estimators used for our structural recovery exercise.

### 2.1 A Random Utility Representation

For a given discrete choice problem, let _x_ denote the prompt context and let _C_ ( _x_ ) _⊆V_ denote the set of label tokens (e.g. Option “A” vs “B”) comprising the feasible options in the choice menu. Let _ui_ ( _x_ ) denote the raw logit on token _i ∈V_ . Under standard softmax decoding (Bridle, 1989), the language model induces a next-token probability distribution over the full vocabulary, where the parameter _τ >_ 0 is _temperature_ , which scales the dispersion of the induced choice distribution. As _τ ↓_ 0 the distribution concentrates all mass on the single token with the highest logit across _V_ . As _τ →∞_ the distribution approaches the uniform over _V_ , spreading mass evenly across all vocabulary tokens and effectively destroying the model’s inherent coherence.

> 4Tokens are the fundamental units of text parsed by the model, encompassing words, subwords, and punctuation marks. The vocabulary set _V_ is fixed by the model’s architecture.

**Proposition 2.1.** _The softmax rule in Equation_ (1) _admits an exact random utility representation,_

_where the error terms ϵi are independent and identically distributed according to a standard Gumbel (Type I Extreme Value) distribution._

_Proof._ See Appendix A.

Since the shocks _ϵi_ are i.i.d. Type I extreme value across all _i ∈V_ , the softmax rule in Equation (1) is the full-vocabulary multinomial logit representation of the random utility model from (McFadden, 1972). Under this interpretation, a model’s raw logit values serve as a systematic utility index, and the temperature parameter _τ_ is the scale of the stochastic component. Consequently, as _τ ↓_ 0, choice converges to the label with the highest logit, and the distribution recovers only the ordinal ranking of the systematic utilities (i.e. the noise vanishes and the cardinal magnitudes of utility differences become irrelevant).

To connect logits to structural preferences, we specify the systematic utility index as a positive affine transformation of a cardinal utility index _Vi_ over the full vocabulary,^5

The scale parameter _κ >_ 0 maps utility differences into logit units and is assumed constant across prompt contexts, so that scaled utility gaps remain comparable across the menus used in our experiments. The term _ζ_ ( _x_ ) is a prompt-specific common shifter that is the same for all tokens within a context and therefore cancels from every within-menu comparison. We leave the functional form of _Vi_ is unrestricted here and explore a quadratic form in Section 3. Nonetheless, its dependence on preference parameters already bears on identification, since Apesteguia and Ballester (2018) show that an additive random utility model of this form can fail to identify risk preferences when _Vi_ is a constant-risk-aversion expected utility whose between-prospect difference vanishes as risk aversion grows. We therefore require _Vi_ to be affine in the risk-preference parameter, which keeps that difference monotone in it.

### 2.2 Random Utility Axioms for Language Models

Having embedded the language model’s token probabilities in a random utility framework, we now formulate the restrictions under which the token-level utility index can be interpreted as stable revealed preferences, and propose concomitant measures of restriction satisfaction. The transformer architecture supplies the logits over the vocabulary, and the random utility representation maps those logits into systematic utilities and induced choice probabilities. The axioms below are therefore not mechanical consequences of the architecture, but restrictions we impose and evaluate to determine whether the induced choices admit a stable revealed-preference interpretation. Throughout this subsection, for each token _k ∈V_ , let denote its realized random utility.

Following this utility specification, we define rational behavior according to the classic axioms of completeness, reflexivity, monotonicity, transitivty, and continuity presented in Varian (1992). Additionally, we define an axiom for the independence of irrelevant alternatives that is implied by the random utility model (Hausman and McFadden, 1984). Each definition is subsequently paired with a measure of adherence to the axiom that we employ in the experimental protocol that follows.

**Definition 2.1** (Completeness axiom) **.** Fix a prompt context _x_ and a choice menu _C_ ( _x_ ) _⊆V_ . The model satisfies _menu completeness_ on _C_ ( _x_ ) at temperature _τ >_ 0 if its realized maximizer lies in the intended menu with probability one. Equivalently,

> 5Under expected utility, any two cardinal representations of the same preferences differ by a positive affine transformation (Von Neumann and Morgenstern, 1947). Equation (2) therefore preserves the underlying vNM preference ordering and the associated risk attitudes.

This means that any token chosen with positive probability is in the choice menu. For example, if the menu choices are A and B ( _C_ ( _x_ ) = _{A, B}_ ) then perfect completeness would imply that the model puts no weight on any other tokens except for those representing “A” and “B”. Traditionally, the completeness axiom dictates that the agent can provide a ranking of feasible options, allowing for indifference between bundles but not for answers that would fail to provide a preference ranking. Our condition operationalizes this requirement as menu adherence, in that the model must confine its response to the feasible options rather than produce a token that conveys no ranking of them.

**Measure 2.1** (Completeness index) **.** The _completeness index_ measures the probability that the model’s realized choice lies in the intended menu,

The index _C_ ( _x, τ_ ) lies in [0 _,_ 1] and equals one if and only if menu completeness holds on _C_ ( _x_ ). Its dependence on temperature is only through the effective scale _κ/τ_ . Lower values of _τ_ concentrate probability on tokens with maximal utility, while higher values of _τ_ move the distribution toward the vocabulary share of _C_ ( _x_ ). Because the full-vocabulary softmax places positive probability on every token whenever logits are finite, exact menu completeness is unattainable at any positive temperature. The axiom therefore describes an ideal, and the index measures proximity to that ideal. Exact completeness obtains only when the logits outside the menu are masked, or in the zero-temperature limit provided the token with the maximal logit lies in the intended menu. Given feasibility of the intended menu, the next restriction concerns equality. Reflexivity requires labels representing the same alternative to receive the same utility and hence the same choice probability.

**Definition 2.2** (Reflexivity axiom) **.** Fix a prompt context _x_ and a choice menu _C_ ( _x_ ) _⊆V_ , and let _∼x_ partition tokens in _C_ ( _x_ ) that represent the same alternative. The model satisfies _exact reflexivity_ on _C_ ( _x_ ) if, for all _i, j ∈ C_ ( _x_ ),

This condition operationalizes the reflexivity axiom as label invariance, in that the utility assigned to an alternative may not depend on the token label under which it is presented. If a language model is indifferent between option A and option B, and if the model is not biased to choose the first option, then each label should receive one half of the choice probability, which we can observe directly in the logits.

The coordinate space in Figure 1 represents a choice space of portfolios that increase in risk from left to right and have higher expected return moving up the horizontal axis. Thus, relative to the reference portfolio (gold star), points directly above it should be preferable to points below. The color gradient shows the progression of logit differences. Dark blue represents an area where the portfolios are intensely preferred to the reference star because those points offer both a much higher return and lower risk. The solid black line maps an indifference curve, or all the points that are valued equally as the reference point. A risk averse decision maker would exhibit an upward sloping indifference curve in this choice space, as shown in the picture. Panel (a) of Figure 1 shows that if A and B label the same portfolio, then they should receive the same utility regardless of which label is assigned.

**Measure 2.2** (Reflexivity index) **.** Fix _i, j ∈ C_ ( _x_ ) such that _i ∼x j_ . The _reflexivity index_ for pair ( _i, j_ ) is

The index _R_ ( _i, j, x, τ_ ) lies in [0 _,_ 1] and equals one if and only if _Vi_ ( _x_ ) = _Vj_ ( _x_ ), equivalently _P_ ( _i | x, τ_ ) = _P_ ( _j | x, τ_ ). Temperature enters only through the scaled utility difference. If _Vi_ ( _x_ ) = _Vj_ ( _x_ ), exact reflexivity is invariant to temperature. If _Vi_ ( _x_ ) _̸_ = _Vj_ ( _x_ ), lowering _τ_ increases the scaled gap and drives _R_ ( _i, j, x, τ_ ) toward zero, while raising _τ_ compresses the gap and moves _R_ ( _i, j, x, τ_ ) toward one.

The next restriction concerns strict improvement. Given an exogenous mean-risk dominance order, monotonicity requires the model’s induced order to respect every dominance comparison. Panel (b) of Figure 1 shows that portfolio A has both a higher expected return, _µ_ , and less uncertainty, relative to porfolio B. Therefore, any agent whose preferences are increasing in expected return and decreasing in risk should prefer A to B. (Rationality alone does not deliver this ranking, since a risk-seeking expected utility maximizer may prefer the riskier portfolio.) We use _≻D_ to denote this mean-risk dominance relation. The benchmark utility index maintained in Section 3 satisfies this monotonicity requirement for every positive value of its risk preference parameter on the mean-risk dominance menus used in our design, where the dominating portfolio has higher expected return and lower raw second moment _R_ . **Definition 2.3** (Monotonicity axiom) **.** Fix a prompt context _x_ and a menu _C_ ( _x_ ) _⊆V_ , and let _≻D_ be an exogenous mean-risk dominance order on _C_ ( _x_ ). The model satisfies _exact monotonicity_ on _C_ ( _x_ ) if, for all _i, j ∈ C_ ( _x_ ),

Figure 1: Stylized visual representations of the random utility restrictions evaluated in the single-token forced-choice portfolio problems.

**Measure 2.3** (Monotonicity index) **.** Fix _i, j ∈ C_ ( _x_ ) such that _i ≻D j_ . The _monotonicity index_ for pair ( _i, j_ ) is

Then _M_ ( _i, j, x, τ_ ) _∈{_ 0 _,_ 1 _}_ , with _M_ ( _i, j, x, τ_ ) = 1 if and only if the model satisfies monotonicity for the dominance pair ( _i, j_ ).

Our measure requires that the probability of choosing a preferable portfolio ( _i_ ) be higher than the probability of choosing a less desirable portfolio ( _j_ ). A language model can place positive probability on the mean-risk dominated option without violating the monotonicity axiom because the axiom restricts the ordering of systematic utilities and choice probabilities, not the probability level assigned to each option. For example, considering panel (b) of Figure 1, the probability assigned to portfolio B can be positive as long as the probability assigned to A is higher. Monotonicity therefore is a pairwise restriction relative to the exogenous order _≻D_ . It requires the model’s induced order to agree with _≻D_ on each dominance pair, but it imposes no condition on the joint consistency of distinct pairwise comparisons. The next axiom imposes that consistency condition by requiring the strict relation induced by the model to be closed under composition (represented in panel (c) of Figure 1).

**Definition 2.4** (Transitivity axiom) **.** Fix a collection of alternatives _{a, b, c}_ and, for each pair ( _i, j_ ), let _xij_ denote the prompt context presenting the binary menu over _i_ and _j_ . Let _≻_ be the strict relation induced by the model across these pairwise contexts, where

The model satisfies _exact transitivity_ on the triad _{a, b, c}_ if

Within a single prompt context the index _Vi_ ( _x_ ) assigns a real number to every token, so the ordering induced inside one menu is transitive by construction. The empirical content of the axiom therefore concerns whether the rankings elicited from separate binary menus can be rationalized by a single stable ordering over the underlying alternatives.

**Measure 2.4** (Transitivity index) **.** For a closed triad _t_ = _{a, b, c}_ , define for each pairwise leg ( _i, j_ ) _∈{_ ( _a, b_ ) _,_ ( _b, c_ ) _,_ ( _a, c_ ) _}_ . Since _ui_ ( _x_ ) = _κVi_ ( _x_ ) + _ζ_ ( _x_ ) with _κ >_ 0, the sign of ∆^_V_ _ij_^equals the sign of ∆^_P_ _ij_^(and induces the pairwise ordering of)^_i_and^(_j_.The triad is)^(_directional_if the first two legs imply a strict) chain,

∆^_V_ _ab_^(_>_0)^(_∧_∆)^_V_ _bc_^(_>_0)^_,_ or ∆^_V_ _ab_^(_<_0)^(_∧_∆)^_V_ _bc_^(_<_0)^(_._) A directional triad is _consistent_ if the closure leg confirms the implied ranking, namely ∆^_V_ _ac_^(has the same sign as ∆)^_V_ _ab_ and ∆^_V_ _bc_^(.For a directional triad)^(_t_, the)^(_transitivity index_is)

Then _T_ ( _t_ ) _∈{_ 0 _,_ 1 _}_ , with _T_ ( _t_ ) = 1 if and only if the closure leg confirms the implied ranking. A non-directional triad places no restriction on the closure leg, so _T_ ( _t_ ) is scored only for directional triads.

Panel c of Figure 1) shows that if A lies above the indifference curve for bundle B (the gold star) and C lies below that indifference curve, then it follows that this decision maker would prefer A to C, if they have rational preferences. Therefore, transitivity is a finite consistency requirement on a closed triad. If the first two comparisons induce a strict chain, then the third comparison must agree with the order obtained by composing the first two.

The next axiom is continuity, which rules out discontinuous jumps in preference. Standard preference continuity implies that, between alternatives ranked on opposite sides of a reference alternative, a continuous path must cross the reference alternative’s indifference contour.

**Definition 2.5** (Continuity axiom) **.** Fix alternatives _a, b, c_ with _a ≻ b ≻ c_ under the pairwise relation in Equation (3), and let _p_ denote the continuous interpolation path connecting _c_ to _a_ in attribute space. For each _t_ , let _x_ ( _t_ ) denote the prompt context presenting the binary menu over _γ_ ( _t_ ) and the reference alternative _b_ . Let _ℓγ_ ( _t_ )( _x_ ( _t_ )) and _ℓb_ ( _x_ ( _t_ )) denote the label tokens in that context that encode those two bundles. The model satisfies _exact continuity_ on _p_ if there exists _t_^_∗_ _∈_ (0 _,_ 1) such that and hence so that _γ_ ( _t_^_∗_ ) _∼ b_ .

**Measure 2.5** (Continuity index) **.** For a path _p_ , define the log odds gap

The population _continuity index_ is

Then _K_ ( _p_ ) _∈{_ 0 _,_ 1 _}_ , with _K_ ( _p_ ) = 1 if and only if path _p_ intersects the indifference contour of _b_ . The population index conditions on the entire interval [0 _,_ 1] and is therefore not observable from finitely many probed comparisons. The empirical implementation evaluates a grid version of the index. For probed points 0 _< t_ 1 _< · · · < tm <_ 1, define which equals one if and only if the probed log odds bracket zero, providing a finite-grid analogue of the continuity implication.

Continuity is the corresponding path-existence restriction. It requires an indifference crossing of the reference alternative along the interpolation path, equivalently a zero crossing of the log probability odds. The grid index records whether the probed comparisons bracket such a crossing, without imposing monotonic movement or uniqueness.

The final restriction concerns the stability of pairwise odds when alternatives outside the pair are varied. Within a fixed prompt context, odds invariance is a direct consequence of the model’s equivalence to the standard random utility model, where the ratio of choice probabilities for any two options depends solely on their systematic utility difference (Luce et al., 1959). Across the binary and expanded prompts used below, the restriction becomes an empirical test of whether that systematic utility difference remains stable when the menu changes.

**Definition 2.6** (Independence of irrelevant alternatives) **.** Fix a prompt context _x_ with _a, b ∈ C_ ( _x_ ) and an expanded context _x_^_c_ with _a, b, c ∈ C_ ( _x_^_c_ ), where _x_^_c_ differs from _x_ only by adding a strictly mean-risk dominated alternative _c_ such that _a ≻D c_ and _b ≻D c_ . The model satisfies _exact independence of irrelevant alternatives_ for pair ( _a, b_ ) if

**Measure 2.6** (Independence of irrelevant alternatives index) **.** For a mean-risk dominated alternative IIA comparison _q_ = ( _a, b, c, x, x_^_c_ ) with _a ≻D c_ and _b ≻D c_ , define

Since _D_^_P_ ( _q, τ_ ) = ( _κ/τ_ ) _D_^_V_ ( _q_ ), the _independence of irrelevant alternatives index_ for comparison _q_ is

Then _I_ ( _q, τ_ ) _∈_ [0 _,_ 1], with _I_ ( _q, τ_ ) = 1 if and only if the pairwise log odds of _a_ against _b_ are invariant to the mean-risk dominated alternative _c_ .

Independence of irrelevant alternatives is an odds-invariance restriction for an irrelevant alternative introduced by construction. It requires the pairwise log odds of _a_ against _b_ to be unchanged when the menu is expanded by an alternative _c_ that is strictly mean-risk dominated by both _a_ and _b_ .

Together, the six indices state the restrictions under which the random utility representation and the affine logit specification admit a revealed-preference interpretation in one-step token choice. We next apply this framework to the portfolio choice problem and derive the two structural estimators.

## 3 Portfolio Choice Problem

The structural framework in Section 2 connects the model’s logit vector to a cardinal utility index _Vi_ through the scale parameter _κ_ , but leaves the functional form of _Vi_ unrestricted. We now specialize that index to a portfolio choice environment in which each alternative is a one-period asset with observed expected return _µi_ and standard deviation _σi_ , so that the relevant tradeoff between return and risk is transparent, fully observed by the researcher, and directly presented to the model. The researcher then recovers variance and the raw second moment mechanically from the reported moments. This environment gives the structural parameters an economically interpretable domain while keeping the objects of choice under complete experimental control.

In this environment, each prompt context _xj_ presents a binary feasible portfolio set _C_ ( _xj_ ) = _{A, B}_ , where the two label tokens encode portfolios with observed central moments ( _µAj, σAj_ ) and ( _µBj, σBj_ ). Holding the rest of the prompt template fixed while varying these moments across menus gives a direct implementation of the measurement objects from Section 2 in a setting with economically interpretable alternatives. Consequently, the associated axioms reveal whether language models exhibit preferences consistent with a single utility function. Relatedly, we note the risk preferences recovered in this setting reflect a preference index over a well-defined class of risky allocations, but one that need not summarize behavior outside it. It is well-known that risk preferences are not globally consistent across domains among human subjects (Einav et al., 2012; Harrison, List, and Towe, 2007; Schildberg-H¨orisch, 2018), and we explore the extent to which this is true for language models as well.

### 3.1 Benchmark Utility Representation

Our maintained benchmark is quadratic expected utility, which serves as a useful approximation to the expected utilities of a diverse set of utility functions (Markowitz, 2014). To express this compactly, let _Ri_ = _E_ [ _x_^2 _i_^(]denote) the raw second moment of portfolio returns. The moments shown in the prompt then imply _Ri_ = _σi_^(2+)^_µ_2 _i_^,since _σi_^(2=)^(_E_[)^_x_2 _i_^(])^(_−E_[)^(_xi_]2. Recovering structural risk preference parameters from discrete choices over financial alternatives) with known payoff distributions has clear precedent in the empirical literature (Mester, 1996; Choi et al., 2007; Cohen and Einav, 2007; List and Mason, 2011).^6 For the label tokens that encode portfolios, we model the corresponding expected utility index as where _β >_ 0 indicates risk aversion, _β <_ 0 indicates risk seeking, and _β_ = 0 collapses to expected-return maximization. Under Equation (4), whether portfolio _i_ is preferred to portfolio _i_^(_′_) when _µi > µi′_ and _Ri > Ri′_ depends entirely on whether _β ≤_ ( _µi − µi′_ ) _/_ ( _Ri − Ri′_ ), so differences in _β_ translate directly into different portfolio rankings and, once the model selects among feasible allocations, different economic outcomes.

Substituting Equation (4) into Equation (2) for the portfolio labels maps the observed portfolio moments into the pre-temperature logit index. Specifically, for alternative _i_ in menu _j_ , where the additive term _ζj_ is common within menu _j_ and drops from within-menu comparisons. By imposing strategic variation in ( _µij, Rij_ ) across alternatives and menus, we can identify the structural parameters of interest. We therefore consider two observation regimes using this framework. When raw pre-softmax logits on the menu labels are observed, within-menu logit gaps can be used to identify the structural parameters directly. When only sampled choices are observed, we employ traditional random utility model estimation methods via maximum likelihood. We explore each of these in turn below. Appendix B summarizes the identification conditions and design implications under both observation regimes.

6Richer alternatives such as prospect theory, probability weighting, and disappointment aversion have been studied extensively (Kahneman and Tversky, 1979; Gul, 1991; Prelec, 1998; Bruhin, Fehr-Duda, and Epper, 2010).

### 3.2 Observed-Logit Estimation

When logits are observed, the data for menu _j_ are the two label logits together with the corresponding portfolio moments. Because the additive menu-specific level in Equation (5) is common to both options, estimation proceeds on the within-menu logit gap and does not require menu fixed effects. An equivalent stacked representation is

_uij_ = _θ_ 0 _j_ + _α_ **1** _{i_ = _A}_ + _θ_ 1 _µij_ + _θ_ 2 _Rij_ + _εij_

where _uij_ is the raw logit for alternative _i_ in menu _j_ , _θ_ 0 _j_ absorbs the menu-specific level including the common term _ζj_ from Equation (5), and _α_ captures any systematic additive position bias of the first option labeled A. Position bias is a well-documented phenomenon of current language models, which we discuss in Section 4 in further detail. With two observations per menu, this stacked equation is algebraically equivalent to the first-difference regression used for estimation.

For a binary menu _j_ with options A and B, within-menu differencing yields where ∆ _uj_ = _uAj − uBj_ , ∆ _µj_ = _µAj − µBj_ , ∆ _Rj_ = _RAj − RBj_ , and _ηj_ = _εAj − εBj_ . Estimation therefore runs on the logit gap with a pooled constant and the within-menu differences in moments. Because the dependent variable is the raw pre-temperature logit gap, decoding temperature does not affect the identifying equation or the precision of this estimator. In this sense, the estimator keeps the component of the logit gap that is systematically explained by portfolio attributes and treats the remaining gap as either positional bias, absorbed by _α_ or removed by mirroring, or specification error.

Identification requires that the slope mapping from utility units to logit units be common across menus, that any positional effect enter as a common additive constant, and that the design matrix with rows (∆ _µj,_ ∆ _Rj_ ) have full rank. The reduced-form slopes imply

#### and

Inference can use heteroskedasticity-robust standard errors for ( _θ_^ˆ 1 _, θ_^ˆ 2). Since _β_^ˆ is a ratio estimator, inference for _β_^ˆ follows from the Delta method applied to the estimated covariance matrix.

### 3.3 Sampled-Choice Estimation

When only sampled choices are observed, we restrict attention to valid A/B responses and let _yjr_ = 1 if option A is chosen on valid draw _r_ from menu _j_ and _yjr_ = 0 if option B is chosen. With ∆ _µj_ = _µAj − µBj_ and ∆ _Rj_ = _RAj − RBj_ , the binary choice probability is where Λ( _t_ ) = exp( _t_ ) _/_ (1 + exp( _t_ )), _γ_ 1 = _κ/τ_ , and _γ_ 2 = _−κβ/τ_ . Note that varying _τ_ across menus does not add identifying content because it only rescales the common latent index. Therefore, in the benchmark design we hold _τ_ = 1 fixed across estimation menus, which preserves the native logit scale.

Full-rank variation in (∆ _µj,_ ∆ _Rj_ ) identifies the reduced-form slopes. Choice data therefore identify _β_ = _−γ_ 2 _/γ_ 1 whenever _γ_ 1 _̸_ = 0. In the frontier regime (i.e. ChatGPT from OpenAI) of our experiments, let _nj_ denote the number of valid A/B responses from menu _j_ . Let _Yj_ =^()^_n_ _r_ =1^(_jyjr_denote the number of those responses choosing option A.) Estimation maximizes the grouped binomial log-likelihood

The omitted binomial coefficient is constant in ( _γ_ 1 _, γ_ 2) and therefore does not affect the maximizer. Under these conditions, together with the standard identification and regularity conditions stated above, maximum likelihood yields consistent estimates _γ_ ˆ1 and _γ_ ˆ2 (Newey and McFadden, 1994). Because _τ_ is exogenously imposed, the structural parameters can be recovered as and

Inference for _κ_ ˆ follows from the inverse Hessian, while for _β_^ˆ the Delta method once again is convenient away from weak-ratio cases.

## 4 Experimental Design

The experiment is built to address two measurement problems. The first is to determine whether a model’s single-token choices satisfy the revealed-preference restrictions required for the utility representation defined by our measures of completeness, reflexivity, monotonicity, transitivity, continuity, and independence-of-irrelevant-alternatives formalized in Section 2. The second is to estimate the shape of preferences under the quadratic mean-risk benchmark of Section 3, recovering the scale parameter _κ_ and the risk preference parameter _β_ .

The unit of observation throughout is a model-menu prompt, a single decision context _x_ that presents a labeled portfolio menu and to which the model responds by placing probability on one label token. What the researcher observes from that prompt differs across the two regimes studied in the paper. For open-weight models we record the raw pre-softmax logits on the label tokens from a single forward pass.^7 For private frontier models we typically do not observe logits and instead query the model repeatedly under softmax decoding, recording the sampled choices from which empirical choice probabilities are formed.

### 4.1 Model Subjects and Elicitation Protocol

Our subject pool spans representative samples of non-thinking models from the open- and closed-source regimes. Within each regime we include models that were publicly available during the window of our study in June 2026, exposed stable version identifiers, and could be queried under a one-token forced-choice protocol with fixed decoding controls. Table 1 lists the resulting sample. Panel A covers open-weight models, and Panel B covers private frontier models. Full details on the experimental code, hardware, vendor metadata, prompt and design hashes, and run identifiers are collected in the replication materials in Appendix B.

|Model Identifier|Developer|API / Hugging Face Identifier|Size (Params)|
|---|---|---|---|
|i _Panel A. Open-Weight M_|_odels_|i||
|Gemma-2-9B|Google|`google/gemma-2-9b-it`|9B|
|Gemma-4-31B|Google|`google/gemma-4-31B-it`|31B|
|Llama-3.1-8B|Meta|`meta-llama/Llama-3.1-8B-Instruct`|8B|
|Llama-3.1-70B|Meta|`meta-llama/Llama-3.1-70B-Instruct`|70B|
|Ministral-3-8B|Mistral AI|`mistralai/Ministral-3-8B-Instruct-2512`|8B|
|Qwen3-8B|Alibaba|`Qwen/Qwen3-8B`|8B|
|Qwen3-14B|Alibaba|`Qwen/Qwen3-14B`|14B|
|_Panel B. Private Frontie_|_r Models_|||
|GPT-4o|OpenAI|`gpt-4o-2024-08-06`||
|GPT-4o mini|OpenAI|`gpt-4o-mini-2024-07-18`||
|Claude 4.5 Haiku|Anthropic|`claude-haiku-4-5-20251001`||
|Claude 4.6 Sonnet|Anthropic|`claude-sonnet-4-6`||
|Gemini 2.5 Flash Lite|Google|`gemini-flash-lite-latest`||
||Table|1: Model subject pool under evaluation.||

The main experiment administers a common menu set to every model. The decision prompt follows a fixed three-part template as presented in Figure 2 below. The first part frames the decision as an annual financial portfolio choice and instructs the model to reply with its preferred label from the listed options. The second part presents the options by specifying the expected return and standard deviation of each. The third part is the cue suffix “I choose Option”. For private frontier models this suffix is appended to the end of the user prompt. For open-weight models it is prefilled at the start of the assistant response block. In both cases the model is induced to emit a single token indicating its choice.

> 7All open-weight models are accessed through their publicly released weights. The full raw logit vector is recorded at the label token positions before any softmax decoding or sampling rule is applied. No provider-side post-processing intervenes between the final layer output and the recorded logit vector, so the observed-logit regime is exact.

In all prompts the labels are the capital letters A and B (and C for the trinary independence menu), which are single tokens in the vocabulary of every model in the subject pool.^8,9

```
You have two options for your annual financial portfolio. Choose the portfolio you prefer
and reply with A orB only.
```

`Option A offers an expected return of [` _µA_ `]% and a standard deviation of [` _σA_ `]%.`

`Option B offers an expected return of [` _µB_ `]% and a standard deviation of [` _σB_ `]%.`

```
IchooseOption
```

#### Figure 2: Decision prompt template

The two observation regimes differ not only in what is recorded but in how choice probabilities are formed. In the open-weight regime, one forward pass per menu yields the exact label logits, and no replication is required. In the frontier regime, choice probabilities are formed either from repeated sampled choices or, where the provider exposes them, from first-token logprobs. Across both regimes we fix the decoding controls at temperature _τ_ = 1 and top- _p_ = 1.^10

### 4.2 Portfolio Menu Construction and Diagnostic Design

All menus draw choice options from a discrete grid of portfolio bundles with expected returns _µ ∈{_ 2 _,_ 3 _, . . . ,_ 18 _}_ percent and standard deviations _σ ∈{_ 10 _,_ 15 _,_ 20 _, . . . ,_ 55 _}_ percent, yielding 17 _×_ 10 = 170 distinct bundles, each with raw second moment _R_ = _σ_^2 + _µ_^2 . This range spans low-return, low-risk profiles through high-return, high-risk profiles, and the spacing is fine enough to generate meaningful variation in the risk-return tradeoff. The 170 bundles admit 1702  = 14 _,_ 365 distinct binary pairings, of which 13 _,_ 596 form the admissible working pool after removing pairs that are degenerate for the relevant first differences. From this pool we strategically construct the diagnostic and estimation menus described below, holding the menu sets constant across all models.

To control for a well known position bias exhibited by many language models (Zheng et al., 2024; Pezeshkpour and Hruschka, 2024; Guan et al., 2025), and for the broader possibility that choices depend on the frame in which a fixed alternative is presented (Salant and Rubinstein, 2008), every canonical menu is also administered in a mirrored presentation that swaps which option carries each label. The reflexivity placebos described below show that a label’s position can carry a systematic logit premium orthogonal to the systematic utility index in Equation (2). For each menu we therefore form the position-corrected log-odds gap where ∆ _u_ and ∆ _u_^mirror denote the A-minus-B log-odds gap under the canonical and mirrored presentations. Because the positional premium enters both presentations with the same sign while the utility difference reverses, the half difference removes the positional component and retains the systematic component. Unless stated otherwise, the monotonicity, transitivity, continuity, and independence diagnostics are evaluated on these position-corrected gaps. The reflexivity index is computed within a single presentation by construction, since the placebo’s two options are identical and the asymmetry is itself the positional signal, and the trinary independence menus use the analogous correction described below.

> 8Some tokenizers include leading-space variants of the A and B tokens. When more than one token realization maps to a label, we aggregate at the label level by summing the softmax probabilities of the associated token realizations. The equivalent grouped logit at temperature _τ_ is _τ_ log^() _r_^(exp()^(_ur/τ_) over realizations)^(_r_of the label. We verify this property before running the experiment) and report the token IDs in Appendix B.

> 9We explored the potential effects of using these specific option labels by randomly selecting tokens as option labels. We found no meaningful difference in these experimental results.

> 10Top- _p_ sampling (also known as nucleus sampling) restricts the candidate token set at each step of decoding to the smallest subset of tokens whose cumulative probability exceeds the threshold _p_ (Holtzman et al., 2019). Setting top- _p_ = 1 keeps the entire vocabulary and avoids truncation, which would break the full-vocabulary softmax that underlies the random utility representation in Proposition 2.1 and the indices that rest on it (McFadden, 1972; Train, 2009).

Table 2 summarizes the diagnostic battery by utility axiom, reporting the menu construction used to evaluate each measure and the corresponding open-weight prompt-level observations. Completeness differs from the other diagnostics because it is evaluated on the union of administered binary prompts rather than on a separate menu class. For each open-weight prompt we extract the full logit vector from a single forward pass and compute the completeness index _C_ ( _x, τ_ ) from Measure 2.1 as the softmax mass on the valid label tokens _{A, B}_ . For frontier models, completeness is the fraction of sampled responses that parse as a valid label.

|Axiom|Menu Construction|Observations|
|---|---|---|
|Completeness (Measure2.1)|Union of administered binary menus|7,786|
|Reflexivity (Measure2.2)|Placebo binary menus pairing each bundle with itself|170|
|Monotonicity (Measure2.3)|Strict mean-risk dominance binary menus|680|
|Transitivity (Measure2.4)|Triad-leg binary menus from three-bundle chains|1,080|
|Continuity (Measure2.5)|Interpolation-path binary menus with endpoints|660|
|Independence (IIA) (Measure2.6)|Binary and trinary menus adding a dominated Option_C_|930|

Table 2: Diagnostic battery by utility axiom. Menu observations are open-weight prompt-level counts and include mirrored presentations where used. Transitivity is scored only for _directional_ triads, namely those whose first two legs induce a strict chain. Continuity diagnostic conditions on the model’s own ranking of the path endpoints.

The diagnostic menus are designed to isolate distinct failures of revealed preference (see Figure 1 for reference). Placebo menus hold the portfolio fixed and therefore attribute any A-versus-B asymmetry to label position. Dominance menus test whether the model respects unambiguous mean-risk improvements. Triad menus test whether pairwise rankings close consistently across three bundles, while interpolation menus ask whether rankings vary continuously along a path between endpoint bundles. Independence menus add a dominated third option to test whether the relative odds between the two focal portfolios are stable to irrelevant menu expansion.

### 4.3 Structural Estimation and Implementation

Our quadratic utility benchmark identifies preferences from variation in the within-menu differences ∆ _µ_ and ∆ _R_ . Consequently, the estimation menus are selected to maximize the statistical efficiency of these two regressors. After enumerating the admissible binary pairs from the 170-bundle grid, we exclude identical bundles and pairs in which one option strictly mean-risk dominates the other, and from the remainder we select menus using a greedy forward D-optimal algorithm that sequentially maximizes the determinant of the information matrix over (∆ _µ,_ ∆ _R_ ) (Kiefer and Wolfowitz, 1959; Mitchell, 2000; Rose and Bliemer, 2009).^11 The open-weight estimation sample comprises 680 D-optimal canonical menus together with their 680 mirrors. The frontier estimation sample uses 400 canonical estimation menus, administered with mirrors for 800 total presentations.

For open-weight models, the data for each menu are the two label logits, and estimation proceeds on the positioncorrected within-menu logit gap. We fit the first-difference regression in Equation (6) with paired corrected differencing, which removes the additive positional advantage _α_ before recovering _κ_ ˆ = _θ_^ˆ 1 and _β_^ˆ = _−θ_^ˆ 2 _/θ_^ˆ 1. Inference for the ratio follows from the Delta method (Greene, 2003). For frontier models, we observe sampled choices rather than logits and estimate the binary logit index in Equation (7) by maximizing the Bernoulli log-likelihood in Equation (8), recovering _κ_ ˆ and _β_^ˆ from the reduced-form slopes as in Section 3. For frontier models, choice probabilities are estimated from repeated queries, with replication counts varying by provider and menu class.^12

Design construction and query scheduling use fixed seeds (design and trial seeds set to 42), and the menu sets are held constant across all subjects. Open-weight logits are extracted deterministically in a single forward pass per menu, so decoding temperature does not enter the open-weight identifying equation. Each query is issued in an isolated session to prevent cross-prompt contamination, and we assess sensitivity to prompt wording through the promptvariant robustness checks reported in Appendix D.

> 11The D-optimal criterion is invoked here, for the parametric estimation, rather than for the axiom diagnostics, because its purpose is to break the collinearity between the return and risk differences and sharpen identification of the structural slopes.

> 12Structural estimation menus and sampling-based diagnostics receive the highest replication, while Google diagnostics that expose first-token logprobs require only a single query. Some large diagnostic pools are capped for cost control; Appendix B reports the full schedule.

For the open-weight observed-logit estimator we report heteroskedasticity-robust standard errors and treat the residual structure as a specification diagnostic, since the logits are deterministic given the prompt and any dispersion reflects departures from the maintained affine specification rather than sampling noise. For the frontier sampled-choice estimator, by contrast, the repeated draws carry a genuine sampling interpretation, and inference follows from the likelihood in the usual way.

## 5 Results

We organize the results in two parts. We first evaluate each model against the six axiom diagnostics defined in Section 2 (completeness, reflexivity, monotonicity, transitivity, continuity, and independence of irrelevant alternatives) using the corresponding indices _C_ , _R_ , _M_ , _T_ , _KG_ , and _I_ computed from the observed logit vectors, and report analogous sampled-choice diagnostics for the frontier runs where available. These diagnostics establish whether each model’s single-token choices satisfy the conditions for a stable utility-theoretic interpretation before any structural parameters are estimated. We then turn to the structural estimation of the risk preference parameter _β_ and the scale parameter _κ_ under the maintained benchmark utility _Vi_ = _µi − βRi_ .

### 5.1 Rationality Diagnostics

Table 3 reports the model-level averages of each diagnostic index across the open-weight and frontier diagnostic runs. Four patterns stand out. First, completeness is near-perfect for most models. Qwen 3, Gemma 4, and Ministral 3 concentrate effectively all softmax mass on the intended label tokens, with _C >_ 0 _._ 999. The Llama 3.1 family is the exception, with the 70B variant allocating roughly one quarter of its probability mass to tokens outside the menu.^13 Second, reflexivity reveals substantial heterogeneity. Models that achieve high completeness do not necessarily treat identical portfolios symmetrically. Qwen 3 (14B) and Gemma 4 exhibit near-zero reflexivity. Llama 3.1 (8B Instruct) shows the strongest adherence to the reflexivity axiom among the base models, achieving _R_ = 0 _._ 836. All but one of the open weight models we tested (Gemma 4) expressed varying degrees of preference for the first option in the menu, which had to be controlled for in the structural estimation procedure that follows. It is only when the portfolios are exactly the same that the models indulge this position bias, thus there is no economic consequence for this violation of the reflexivity axiom because, regardless of the label that is chosen, the final allocation is the same. See Table 15 in the appendix for more details on this result. Third, monotonicity is satisfied at ceiling across the entire subject pool, and continuity is satisfied at ceiling wherever the premise-verified continuity index is available.

|Model|Complete|Reflexive|Continuity|Monotonicity|Transitive|IIA|
|---|---|---|---|---|---|---|
|Qwen 3 (14B)|1.000|0.001|1.000|1.000|0.983|0.322|
|Qwen 3 (8B)|1.000|0.014|1.000|1.000|0.957|0.644|
|Gemma 4 (31B)|1.000|0.000|1.000|1.000|0.968|0.122|
|Ministral 3 (8B)|1.000|0.088|1.000|1.000|0.994|0.824|
|Gemma 2 (9B)|0.996|0.797|1.000|1.000|0.986|0.327|
|Llama 3.1 (8B)|0.902|0.836|1.000|1.000|0.994|0.920|
|Llama 3.1 (70B)|0.758|0.269|1.000|1.000|1.000|0.763|
|Claude 4.5 Haiku|0.970|0.072|1.000|1.000|0.857|0.498|
|Claude 4.6 Sonnet|0.964|0.000|1.000|1.000|1.000|0.503|
|GPT-4o|0.998|0.009|1.000|1.000|1.000|0.301|
|GPT-4o Mini|1.000|0.008|1.000|1.000|1.000|0.586|
|Gemini 2.5 Flash Lite|1.000|0.000|1.000|1.000|0.987|0.802|

Table 3: Rationality diagnostics for open-weight and frontier models

These results indicate that the models uniformly respect stochastic dominance and exhibit smooth, monotone movement of the log-odds along interpolation paths. Transitivity is high throughout, ranging from 0 _._ 957 for Qwen 3 (8B) to 1 _._ 000 for Llama 3.1 (70B) and the OpenAI frontier models, implying that preference cycles are rare but not entirely absent among directional triads.

> 13This result is robust to variations in the prompting of the model. For example, some models are trained to produce output in markdown format, which can lead to leading characters for emphasis (e.g. “*” or other symbols in front of the labels).

Fourth, independence of irrelevant alternatives is the weakest axiom across the entire subject pool. No model approaches the ceiling scores observed for monotonicity or continuity. The index _I_ ranges from 0 _._ 122 for Gemma 4 to 0 _._ 920 for Llama 3.1 (8B Instruct), indicating that adding a strictly dominated third option systematically shifts the log-odds between the original pair. The IIA scores covary strongly with reflexivity. Models exhibiting severe positional label bias, such as Gemma 4 ( _R_ = 0 _._ 000, _I_ = 0 _._ 122) and Qwen 3 14B ( _R_ = 0 _._ 001, _I_ = 0 _._ 322), show the largest violations. This suggests that the same positional sensitivity that prevents symmetric treatment of identical options also makes the pairwise odds fragile to menu expansion.

**Finding 1:** The open-weight and frontier models in our sample largely adhere to our measures on the axioms of utility theory. However, we find that a position bias emerges among nearly identical options which leads to violations in reflexivity. Models also do not adhere to the independence of irrelevant alternatives assumption.

### 5.2 Structural Estimation

Having established which axiom conditions each model satisfies, we now estimate the structural parameters of the maintained quadratic expected utility _Vi_ = _µi − βRi_ . Table 4 reports estimates of the scale parameter _κ_ ˆ and the risk aversion parameter _β_^ˆ . The open-weight rows use the observed-logit estimator in Equation (6), with paired corrected differencing that removes the additive positional bias _α_ before fitting. The frontier rows use the sampled-choice maximum likelihood estimator. Heteroskedasticity-robust standard errors appear in parentheses, and inference for _β_^ˆ follows from the Delta method applied to the ratio _−θ_^ˆ 2 _/θ_^ˆ 1.

|Model|ˆ_κ_|ˆ_β_|(Pseudo) R^2|
|---|---|---|---|
|Qwen 3 (14B)|0.632 (0.006)|0.003 (0.000)|0.936|
|Qwen 3 (8B)|0.311 (0.004)|0.003 (0.000)|0.889|
|Gemma 4 (31B)|1.482 (0.008)|0.004 (0.000)|0.977|
|Ministral 3 (8B)|0.098 (0.004)|0.011 (0.000)|0.897|
|Gemma 2 (9B)|0.249 (0.006)|0.008 (0.000)|0.911|
|Llama 3.1 (8B)|0.039 (0.002)|0.009 (0.000)|0.886|
|Llama 3.1 (70B)|0.162 (0.002)|0.005 (0.000)|0.939|
|Claude 4.5 Haiku|0.393 (0.006)|0.004 (0.000)|0.634|
|Claude 4.6 Sonnet|0.284 (0.006)|0.005 (0.000)|0.425|
|GPT-4o|0.381 (0.046)|0.007 (0.000)|0.822|
|GPT-4o Mini|0.488 (0.058)|0.004 (0.000)|0.750|
|Gemini 2.5 Flash Lite|1.044 (0.192)|0.004 (0.000)|0.886|

Table 4: Structural estimates of scale _κ_ and risk aversion _β_ . _R_^(2) is reported for observed-logit open-weight estimates and pseudo- _R_^2 for sampled-choice frontier estimates.

Every model in the subject pool yields a positive and statistically significant estimate of _β_ , indicating that all models exhibit risk aversion. The point estimates vary by a factor of nearly four, ranging from _β_^ˆ = 0 _._ 0031 for Qwen 3 (8B) to _β_^ˆ = 0 _._ 0114 for Ministral 3 (8B Instruct). This range implies economically meaningful variation in risk attitudes across model families. Models at the upper end penalize risk more aggressively and will forgo higher expected returns in favor of lower dispersion at comparatively modest risk differentials.

Every model yields a strictly positive estimate for the scale parameter _κ_ ˆ, confirming that they naturally interpret the value ordering of the portfolios in the experiment by preferring higher expected returns. If, in contrast, _κ_ estimates had been negative then models would have preferred portfolios with lower expected returns, _ceteris paribus_ . The magnitude of _κ_ ˆ exhibits wide cross-model variation. Gemma 4 (31B) produces the largest estimate at _κ_ ˆ = 1 _._ 482, implying that a one-unit increase in the cardinal utility index moves the pre-softmax logit by nearly 1.5 units. By contrast, Llama 3.1 (8B) yields ˆ _κ_ = 0 _._ 039, roughly forty times smaller.

To provide a richer visualization of these preferences beyond the point estimates in Table 4, we plot each model’s structural _β_^ˆ estimate (except Ministral) in the _µ_ – _σ_ plane. Starting from a base bundle with expected return _µ_ = 10 and risk _σ_ = 30, we sweep over the dense grid of alternative portfolios from our experimental protocol and record the position-corrected logit gap between each grid portfolio and the base. The yellow dashed line overlays the indifference curve implied by the mean-variance structural estimates, and the heatmap colors encode the sign and magnitude of the logit difference, with blue regions preferred to the base and red regions dispreferred.

Several patterns emerge from these plots. All six models produce upward-sloping indifference curves, confirming that higher risk must be compensated by higher expected return. The curves are roughly convex in the _µ_ – _σ_ plane, consistent with the quadratic utility specification maintained in the structural estimation. However, the steepness of the curves varies considerably across models, reflecting the heterogeneity in _β_^ˆ documented above. Models with larger risk aversion estimates, such as Gemma 2 (9B), trace out steeper curves that demand more return compensation per unit of additional risk. Models with smaller _β_^ˆ , such as Qwen 3 (14B), produce flatter curves that tolerate substantially more risk for modest return increments.

**Finding 2:** All open-weight and frontier models in our sample exhibit varying degrees of risk aversion that can be approximated by a quadratic utility function.

### 5.3 Inducing Preferences in Language Models

Can specific economic preferences be induced into the language model? This question matters because a principal who delegates an allocation decision to a language model delegates authority over the tradeoff that the instruction leaves unresolved. If the model’s default risk attitude differs from the principal’s, then the delegated choice implements the model’s objective rather than the principal’s. The preceding results show that these default preferences can be measured. We now ask whether the preference-misalignment component of this agency problem can also be addressed directly by inducing the model to reveal a target economic preference in subsequent choices.

We study this question by fine-tuning Llama 3.1 (8B Instruct) toward two principal-specified risk preferences. The principal’s desired ordering over portfolios is represented by the same maintained utility index, where _β_^_∗_ is chosen by the principal. We consider two target specifications. The first is approximately risk neutral, with _β_^_∗_ = 0 _._ 0001. The second is more risk averse, with _β_^_∗_ = 0 _._ 0045. These targets are not prompts asking the model to describe a preference. They are the preferences the delegated agent is trained to implement in its own choice probabilities.^14

For each training menu, we compute the utility difference implied by the principal’s target preference and translate it into the log odds that the delegated model should assign to the two options.

For the same menu, let ∆ _zm_ ( _θ_ ) denote the model’s actual logit gap between the two answer labels, where _θ_ collects the model weights. The preference component of fine-tuning chooses weights to close the distance between the actual logit gap and the target logit gap,

Minimizing _L_ ( _θ_ ) updates the model toward the principal’s choice rule. If the model places too much relative logit weight on option _A_ , the residual is positive and the objective rewards changes that reduce the _A_ versus _B_ gap. If the model places too little relative logit weight on option _A_ , the objective rewards changes that increase that gap. Repeating this comparison across the training menus moves the model’s logit surface toward the logit surface implied by the principal’s utility. The scale parameter _κ_ governs the strength of the probabilistic response but does not determine the recovered risk preference, since _β_^ˆ is identified by the ratio of the risk and return slopes. Appendix C gives the formal multi-term objective, the training corpus, the logit-scale choice, and the implementation details.

> 14This differs from prompt engineering. Prompt engineering conditions the model on instructions supplied in the input context and leaves the model weights unchanged, so any induced behavior persists only while that context is present. Fine-tuning updates the weights that generate the choice rule itself, allowing the principal-specified preference to be embedded in the model’s revealed choice probabilities.

Figure 3: Empirical indifference curves for open-weight models mapped over the portfolio space.

After fine-tuning, we evaluate the delegated models exactly as before, on held-out menus not used in training. The test is whether the model now reveals the principal’s target preference rather than the base model’s default preference, and whether that induced preference passes the same diagnostic battery. Table 5 reports the rationality diagnostics. Relative to the baseline Llama 3.1 (8B Instruct), both induced-preference models substantially improve completeness and reflexivity, with completeness above 0 _._ 99 and reflexivity above 0 _._ 98. Monotonicity remains perfect in both cases. The near risk-neutral target achieves perfect transitivity and raises IIA to 0 _._ 9484, while the more risk-averse target maintains high diagnostic performance, with IIA equal to 0 _._ 8000 and transitivity equal to 0 _._ 9583.

|Model|Complete|Reflexive|Continuity|Monotonicity|Transitive|IIA|
|---|---|---|---|---|---|---|
|Llama 3.1 (8B,_β_ = 0_._0001)|0.9961|0.9941|0.9333|1.0000|1.0000|0.9484|
|Llama 3.1 (8B,_β_ = 0_._0045)|0.9914|0.9870|1.0000|1.0000|0.9583|0.8000|
|Ta|ble 5: Ration   i|ality diagnost  i|ics for fine-tu|ined models|||

Table 6 reports the structural estimates and Figure 4 plots the implied indifference curves over the portfolio space. The induced models reveal the principal-specified preferences with high precision. The near risk-neutral specification targets _β_^_∗_ = 0 _._ 0001, and the fine-tuned model yields _β_^ˆ = 0 _._ 000087 with a standard error of 0 _._ 000003. The more risk-averse specification targets _β_^_∗_ = 0 _._ 0045, and the fine-tuned model yields _β_^ˆ = 0 _._ 004560 with a standard error of 0 _._ 000017. In both cases, the target value lies within the 95% confidence interval. The scale parameter estimates are similarly precise. The low- _β_ model yields _κ_ ˆ = 1 _._ 001301 with a standard error of 0 _._ 000720, and the high- _β_ model yields _κ_ ˆ = 0 _._ 718940 with a standard error of 0 _._ 002206. The restricted quadratic surface provides a tight fit to the indifference data, producing _R_^2 values of 0 _._ 9992 and 0 _._ 9935.

The economic interpretation is that the principal need not accept the model’s default risk attitude as an unavoidable agency cost within this portfolio environment. A principal can specify a target utility function, fine-tune the model toward the choice odds implied by that utility, and then verify out of sample that the delegated model reveals the intended preference. The exercise therefore addresses the preference-misalignment part of the principal-agent problem studied here. The agent’s objective can be measured, deliberately shifted, and audited with the same structural tools.

|Model||ˆ_κ_|ˆ_β_|R^2|
|---|---|---|---|---|
|Llama 3.1|(8B,_β_ = 0_._0001)|1.001301|0.000087|0.999|
|||(0.000720)|(0.000003)||
|Llama 3.1|(8B,_β_ = 0_._0045)|0.718940|0.004560|0.994|
|||(0.002206)|(0.000017)||

Table 6: Structural estimates of scale _κ_ and risk aversion _β_ for fine-tuned models.

**Finding 3:** A principal can install its preferred risk attitude in a language model through fine-tuning, embedding the preference in the model’s weights so that it governs how the model resolves tradeoffs.

## 6 Conclusion

The starting point of this paper is an equivalence. When a language model is required to state its choice with a single token, the rule by which it selects that token is exactly the conditional logit random utility model of McFadden (1972). Using this equivalence, we develop a structural methodology for recovering the economic preferences of these models directly from their generative outputs. The raw pre-softmax logits assigned to menu labels can therefore be interpreted as a systematic utility index and place the choices of a language model directly within the scope of standard structural discrete choice methods. It yields one estimator when the model’s internal scores are observable, as with open-weight models, and the standard maximum likelihood estimator when only realized choices are observed, as with proprietary frontier models (i.e. ChatGPT). The same framework also motivates six revealed-preference diagnostics for completeness, reflexivity, monotonicity, transitivity, continuity, and independence of irrelevant alternatives. These diagnostics allow the stability of the utility interpretation to be evaluated before any structural parameter is estimated.

Figure 4: Empirical indifference curves for fine-tuned models mapped over the portfolio space.

Applying this method to analyze the choices of several language models, our results reveal a clear distinction between coherence about economic content and invariance to economically irrelevant features. Across the models we study, monotonicity holds for every model, continuity holds wherever the premise-verified index is available, and transitivity is high. The models therefore respect mean-risk dominance, move smoothly along the interpolation paths in the experiment, and rarely generate cycles across separately elicited menus. In this sense, the models behave like textbook rational agents. The main failures arise from features of the elicitation environment that should be economically irrelevant. Reflexivity frequently fails because models assign different weights to identical portfolios upon label or position permutation. Independence of irrelevant alternatives proves to be the weakest diagnostic, as introducing a dominated option routinely alters the pairwise odds between the original alternatives. These failures covary, which suggests that positional sensitivity and menu sensitivity are closely related in single-token model choice.

Conditional on these limits, the structural estimates recover economically interpretable risk preferences. Every open-weight and frontier model in our sample is risk averse, with a positive and statistically significant risk-aversion parameter under the maintained quadratic specification. The estimates vary by nearly a factor of four across model families. This variation is large enough to imply different portfolio rankings and therefore different allocations when a model is asked to choose among risky alternatives. The estimated indifference curves are upward sloping and approximately convex in the mean-variance plane, so the familiar quadratic apparatus provides a useful first-order description of the preference surface these models reveal.

The fine-tuning exercise provides a structural validation of this interpretation. We install a target utility function in an open-source model at two known risk-aversion targets, and our estimator recovers both targets with high precision on menus the model never saw in training. The exercise also shows that the main invariance failures are not fixed properties of the base model. In the controlled environment studied here, the relevant preference object can therefore be measured and deliberately shifted, and the same intervention can improve the invariance properties required for its interpretation.

Two boundaries delimit what we claim. Each choice is a single token produced in one forward pass rather than the outcome of extended deliberation, and the domain is a controlled portfolio environment rather than the full space of economic decisions. These boundaries are what make the measurement exact, since within them the mapping from logits to utility follows from the softmax rule rather than from an auxiliary assumption. They also locate the object precisely. What we recover is the default preference a model brings before it reasons, which is the relevant object when a principal delegates under an incomplete instruction and the model answers directly. Whether that default persists when the model deliberates across many tokens, when the prompt is enriched toward a fuller mandate, or when the domain moves beyond portfolio choice is the natural next question, as is whether the same preference governs the longer chains of reasoning through which these models increasingly act. While we briefly entertain these extensions in our robustness checks, we treat the single-token measurement as our primary interest and the foundation those extensions build on. Whether structurally measured preferences persist under deliberation and across economic environments is the natural next question for this line of work.

These findings matter for the use of language models as economic agents. Delegating allocation decisions to a model also delegates authority to the preferences that govern its choices. Those preferences need not be observed by the principal, and the diagnostics in this paper show that they need not be invariant to irrelevant features of the menu. A coherent deployment strategy therefore requires more than evidence that a model can correctly rank dominated alternatives. It requires precise measurement of the preference governing choice. It requires diagnostics to verify the stability of that preference. Finally, it requires tools to shift the preference when a different objective is intended. The evidence presented here supplies a controlled demonstration of exactly that program.

### A Proofs

### A.1 Proof of Proposition 2.1

_Proof._ Assume the error terms _ϵi_ are independent and identically distributed following a standard Gumbel (Type I Extreme Value) distribution. The cumulative distribution function (CDF) is _F_ ( _ϵ_ ) = exp( _−_ exp( _−ϵ_ )) and the probability density function (PDF) is _f_ ( _ϵ_ ) = exp( _−ϵ_ ) exp( _−_ exp( _−ϵ_ )). The model reveals a preference for token _i_ if it provides the highest utility among all alternatives in the vocabulary _V_ . That is, _Ui > Uj_ for all _j̸_ = _i_ . The probability of choosing token _i_ is given by

Rearranging to isolate _ϵj_ yields

Conditional on a specific value of _ϵi_ , the probability that this inequality holds for all _j̸_ = _i_ is the product of their individual CDFs, due to independence

To find the unconditional probability, we integrate this product over the marginal distribution of _ϵi_ as follows

Substitute the Gumbel PDF for _f_ ( _ϵi_ ) and separate the exponent terms inside the product

We can combine the exp( _−_ exp( _−ϵi_ )) terms. Notice that the standalone PDF component is exactly the term we would get if _j_ = _i_ inside the product. Therefore, we can absorb it by indexing the product (and subsequent sum) over all _j ∈V_

To solve this integral, apply the substitution _t_ = exp( _−ϵi_ ), which gives _dt_ = _−_ exp( _−ϵi_ ) _dϵi_ . The limits of integration _uj_ ( _x_ ) _−ui_ ( _x_ ) change from ( _−∞, ∞_ ) to ( _∞,_ 0). To simplify the notation, let _K_ =^() _j∈V_^exp  _τ_ . Then

Evaluating this standard exponential integral yields

Substituting the expression for _K_ back into the result gives

This exactly recovers Equation (1), completing the proof.

#### B Experimental Design and Identification

### B.1 Label Tokens

The forced-choice protocol uses the labels A and B in binary menus and A, B, and C in three-option independence menus. For open-weight models, grouped label logits and probabilities are computed by aggregating every vocabulary token whose decoded text matches the target label after trimming whitespace, with alphabetic labels matched caseinsensitively. This is the rule implemented by `build token`_m`ap` in the local-model evaluation code and reused by the fine-tuning code. Table 7 reports the token IDs resolved from the cached tokenizers used in the run environment. The two fine-tuned Llama variants use the Llama 3.1 (8B) tokenizer and therefore share its label-token map.

|Model tokenizer|A IDs|||B IDs||||C ID|s|||
|---|---|---|---|---|---|---|---|---|---|---|---|
|Gemma-2-9B|282, 314, 235250, 2352|476, 80|586,|283,  235268|315, , 2353|518, 05|599,|284, 2352|316, 60, 2352|498, 88|585,|
|Gemma-4-31B|303, 335, 236746, 2367|496, 76|562,|304,  236763|336, , 2367|518, 99|603,|305, 2367|337, 55, 2367|505, 80|565,|
|Llama-3.1-8B and fine-tuned variants|32, 64, 264, 23845, 11858|362, 6|11575,|33, 65, 13083,|293, 10869|426, 3, 115|2282, 648|34, 6391|66, 272, , 116545,|356, 1171|1470, 55|
|Llama-3.1-70B|32, 64, 264, 23845, 11858|362, 6|11575,|33, 65, 13083,|293, 10869|426, 3, 115|2282, 648|34, 6391|66, 272, , 116545,|356, 1171|1470, 55|
|Ministral-3-8B|1065, 1097, 44056, 88974|1261, |1349,|1066, 40796|1098,|1289,|1398,|1067 6622|, 1099, , 94097|1272,|1359,|
|Qwen3-8B|32, 64, 264, 22985|362,|11323,|33, 65, 12791|293,|425,|2233,|34, 6258|66, 272, |356,|1444,|
|Qwen3-14B|32, 64, 264, 22985|362,|11323,|33, 65, 12791|293,|425,|2233,|34, 6258|66, 272, |356,|1444,|

Table 7: Open-weight label-token IDs aggregated into menu labels.

### B.2 Identification Under the Two Observation Regimes

This appendix next states the binary identification results under the two observation regimes. Let _B_ denote the finite grid of candidate bundles indexed by observed moments ( _µ, σ_^2 _, R_ ) with _R_ = _σ_^2 + _µ_^2 . Let _P ⊂B × B_ denote the set of admissible ordered binary menus ( _A, B_ ) with distinct bundles. For the structural estimation sample used in the paper, this algebraic pool is further restricted by excluding degenerate first-difference pairs and strict mean-risk dominance pairs, as described in Section 4.3. For any menu _j ∈P_ , write _xj_ = (∆ _µj,_ ∆ _Rj_ )^(_′_) with differences taken in A minus B order. Mirrored menus reverse the sign of _xj_ and help balance the design, but they do not change its span.

### B.3 Observed-Logit Design

For observed logits, within-menu differencing removes the menu fixed effect. Pairing each canonical presentation with its mirror removes the additive position component and yields the content signal

The estimating equation is then with no intercept. Define the second-moment matrix

If _MN_ is nonsingular, then ordinary least squares on the paired content-signal equation identifies ( _θ_ 1 _, θ_ 2). Under the maintained mapping _θ_ 1 = _κ_ and _θ_ 2 = _−κβ_ , the same design identifies _κ_ = _θ_ 1 and _β_ = _−θ_ 2 _/θ_ 1 whenever _θ_ 1 _̸_ = 0.

For observed logits, precision depends on the same second-moment matrix. Under homoskedastic errors, the covariance matrix of _θ_^ˆ is proportional to _MN_^(_−_1.This covariance expression is a design benchmark.Reported inference uses) heteroskedasticity-robust covariance estimates and the Delta method for the ratio parameter. A natural benchmark is the D-optimal criterion of Louviere and Woodworth (1983),

Since det( _MN_ ) = ∆ _µ_^2 ∆ _R_^2 _−_ ∆ _µ_ ∆ _R_ 2 _,_

the criterion favors menus with large spread in both coordinates and little correlation between them. Under a symmetric rectangular approximation to the feasible difference set, the benchmark places equal mass on the four corner points ( _±_ 1 _, ±_ 1). On the actual finite grid, the same logic calls for menus near the boundary of the feasible difference set, balanced representation across quadrants, and mirrored orderings so that

Following Louviere and Woodworth (1983), a c-optimal refinement after a pilot estimate _β_^(˜) can target the precision of the ratio parameter by solving

### B.4 Sampled-Choice Design

With sampled choices, the identified index is the binary logit index. The reported frontier estimator includes an intercept that absorbs an average A-label advantage,

Throughout this design discussion temperature is held fixed and observed, so the information problem is governed by menu placement and query allocation rather than by rescaling the latent index.

Suppose the sampled-choice logit index is correctly specified and the design matrix with rows _wj_ = (1 _, x_^(_′_) _j_^())^(_′_has full) rank. Then choice data identify the intercept and the reduced-form slopes ( _γ_ 1 _, γ_ 2) = ( _κ/τ, −κβ/τ_ ). If _γ_ 1 _̸_ = 0, the same data identify _β_ = _−γ_ 2 _/γ_ 1. If _τ_ is known, then _κ_ = _τγ_ 1 is identified. If _τ_ is not observed, then _κ_ and _τ_ are not separately identified.

When private models are queried repeatedly on the same menu, the Bernoulli likelihood from Equation (8) is equivalently written in grouped binomial form. If menu _j_ has _nj_ valid parsed A/B responses and option A is chosen _yj_ times, then

This grouped likelihood uses the sufficient statistics for repeated draws and is numerically equivalent to the trial-level Bernoulli likelihood under conditional independence. Invalid or ambiguous parses are excluded from both _yj_ and _nj_ and enter the completeness diagnostic separately.

Because frontier choice probabilities must be formed from finitely many queries rather than read off a logit vector, the design specifies a replication count for each menu class, and these counts differ by provider and by experiment. The schedule is not summarized by a single repetition number, so Table 8 reports it in full. Two regularities organize it. First, the structural estimation menus and the sampling-based diagnostics are replicated heavily, while the provider that exposes first-token logprobs (Google) requires only a single query for the diagnostics whose indices can be read directly from the recorded probabilities. The Anthropic and OpenAI APIs expose no logprobs, so their diagnostics are identified from repeated sampling alone. Second, the higher replication counts are kept affordable by capping the number of menus drawn from the larger diagnostic pools. These caps are stated in the table and reduce coverage relative to the open-weight counts in Table 2 rather than expanding it. All frontier runs fix _τ_ = 1 and top- _p_ = 1.

Design is now local because the expected Fisher information depends on the unknown choice probabilities. For binary logit with one draw from each of _N_ distinct menus, _pj_ = Λ( _α_ + _γ_ 1∆ _µj_ + _γ_ 2∆ _Rj_ ), and _wj_ = (1 _, x_^(_′_) _j_^())^(_′_, the expected) Fisher information for ( _α, γ_ 1 _, γ_ 2) is

|Menu class|Anthropic (Claude 4.x)|OpenAI (GPT-4x)|Google (Gemini 2.5)|Menu cap|
|---|---|---|---|---|
|Estimation|20|20|20 (mirror 20)|400 menus|
|Placebo (reflexivity)|20|20|1 (logprob)|400 menus|
|Monotonicity|20|20|1 (logprob)|400 menus|
|Transitivity (triad legs)|25|25|1 (logprob)|400 triads|
|Independence (trinary)|50|50|1 (logprob)|400 base menus|
|Continuity (8 path points)|25|25|1 (logprob)|400 menus|

Table 8: Frontier replication schedule (queries per menu) by provider and menu class.

The corresponding slope-information block is

If menu _j_ is instead queried _nj_ times, the expected information becomes

The weight _pj_ (1 _− pj_ ) is largest near indifference, so menus with extreme predicted choice probabilities contribute little information even when _xj_ is large. Once pilot estimates are available, efficient sampled-choice design therefore operates on both margins of the problem. It chooses where to place menus and how to allocate repeated queries across those menus.

For a fixed total query budget _T_ =^()^_J_ _j_ =1^_nj_,write^(_qj_=)^(_nj/T_.FollowingLouviereandWoodworth(1983)and) Louviere, Flynn, and Carson (2010), a locally D-optimal second wave solves where _p_ ˜ _j_ = Λ(˜ _α_ + ˜ _γ_ 1∆ _µj_ + ˜ _γ_ 2∆ _Rj_ ) is evaluated at pilot estimates. Since the gradient of _β_ with respect to ( _α, γ_ 1 _, γ_ 2) is proportional to (0 _, β,_ 1), a locally c-optimal second wave for the ratio parameter solves

These criteria imply a two-stage design. A first wave should be broad, balanced, and approximately orthogonal so the signs and relative magnitudes of the slopes can be learned. A second wave should concentrate on menus near the estimated indifference frontier _α_ + _γ_ 1∆ _µj_ + _γ_ 2∆ _Rj ≈_ 0 while preserving independent movement in ∆ _µj_ and ∆ _Rj_ . It should also place more repetitions on those menus rather than spreading the query budget uniformly across the full first wave.

Under the common first-wave menu set _J_ used in this paper, equal replication _nj_ = _m_ for every _j ∈J_ gives

Applying the Delta method to _β_ = _−γ_ 2 _/γ_ 1 then yields

Consequently, a target standard error _sβ_ is met whenever _m_ satisfies

Equation (12) therefore provides a precision benchmark for equal replication. One can evaluate it over a conservative grid of plausible ( _β, κ_ ) values and take the largest implied integer. The main experiment in this paper uses equal replication across the common first-wave menus because it preserves comparability across observation regimes and model versions. This is transparent but not generally optimal for sampled-choice estimation. A model-specific second wave that reweights the first-wave menus according to Equation (11), or augments them with new menus near the pilot indifference frontier, is more statistically efficient when cross-model comparability is not the primary objective. Varying temperature across menus is not a substitute for this design because it only rescales the same index. As a robustness exercise, one can repeat the estimator at alternative fixed temperatures and verify that _β_^ˆ remains stable while ˆ _κ_ rescales according to ˆ _κ_ = _τ γ_ ˆ1.

#### C Model Fine-tuning Methodology

This appendix specifies the construction of the two fine-tuned validation models reported in Section 5.3. It states the random-utility foundation of the training target, the full multi-term objective, the training corpus, the selection of the logit scale, the comparison with the earlier structural-gap objective, and the functional forms of the diagnostics used to evaluate the models.

For orientation, the fine-tuning exercise has two moving parts. First, the structural term teaches the model a logit surface, where for each portfolio menu, the A-minus-B label-logit gap should equal _κ_ times the utility difference implied by the target _β_ . Second, the auxiliary invariance terms prevent degenerate ways of achieving that target, such as always favoring the first label, assigning mass outside the answer set, or changing the A/B odds when a dominated third option is added. The training target is therefore a relation between portfolio attributes and logit gaps, not a fixed preference for any answer token.

### C.1 Targeting Logits to a Parameterized Utility

The fine-tuned models are trained to embody the maintained utility _Vi_ = _µi − βRi_ for a prescribed value of _β_ . This training objective rests on a random-utility interpretation of the model’s single-token choice. Let the latent value of option _i_ in a menu be _ui_ = _Vi_ + _εi_ , where _Vi_ = _µi − βRi_ is the maintained benchmark utility and _εi_ is an idiosyncratic disturbance. When the disturbances are independent and identically Gumbel distributed, the probability that option _A_ is chosen over option _B_ is the binary logit where _κ_ is the inverse scale of the Gumbel disturbance and _τ_ is the model’s temperature parameter. The softmax that the model places over the two label tokens is the empirical analog of this choice probability, so the pre-softmax logit gap between the two labels is the empirical analog of _κ_ ( _VA − VB_ ). The structural-gap loss _L_ struct trains the model to match its position-corrected logit gap to this target over the jittered-grid estimation corpus, after removing the additive positional component identified on placebo menus.

The scale _κ_ enters Equation (13) only as a common multiplier of the utility difference and is not separately identified from the strength of preference within a single binary menu. The recovered risk parameter is unaffected by this indeterminacy. Writing the restricted quadratic surface fit to the position-corrected indifference data as ∆ _u_ = _θ_ 1∆ _µ_ + _θ_ 2∆ _R_ , the estimator of risk aversion is the ratio

Multiplying the entire logit surface by any positive constant _c_ rescales both _θ_^ˆ 1 and _θ_^ˆ 2 by _c_ and leaves their ratio unchanged. The choice of _κ_ therefore affects only the numerical conditioning of the training problem and the precision with which the indifference locus is identified, not the recovered value of _β_ . This is the formal basis for the invariance asserted in Section 5.3. Under the original structural-gap objective the loss is computed with _κ_ set equal to the observed logit-gap standard deviation of the base model, a convention we denote `kappa`_a`nchor = true` . Section C.4 explains why this convention fails at low _β_ and how we replace it.

### C.2 The Multi-Term Invariance Objective

All loss terms are evaluated at the cue position, immediately before the model emits its one-token response. Let _zm ∈_ R^(_|V|_) denote these logits for menu _m_ . Since an answer label can have more than one tokenization, let _Tq ⊂V_ be the set of token ids decoding to label _q ∈{A, B, C}_ and define

For a binary menu _m_ = ( _A, B_ ), write _gm_ = _ℓA_ ( _zm_ ) _− ℓB_ ( _zm_ ). The training utility is

All expectations below are empirical averages over the indicated training set. For scalar residual _r_ , let with _δ_ = 1 throughout.

At optimizer step _t_ , the full objective is where _at_ = min _{_ 1 _,_ ( _t_ + 1) _/T_ aux _}_ and _T_ aux is ten percent of the planned optimizer steps. The structural, binary-mass, and Sobolev terms enter from the first step; the remaining terms are linearly introduced through _at_ . In the reported full-objective runs,

Terms with zero weight, or without corresponding training examples in a run, are absent from that run.

The structural term matches the binary logit gap to the random-utility target:

Here _D_ 2 denotes the binary structural training rows. The invariance runs set _bα_ = 0; the earlier structural-gap runs set it equal to the pre-training estimate of the positional offset.

The binary mass term is which penalizes probability assigned outside the admissible answer labels.

The Sobolev term constrains local derivatives of the logit surface. For derivative rows _m ∈D∂_ , let _zm,µ_ + and _zm,µ−_ be the cue logits after perturbing _µA_ by _±hµ_ , and define _zm,σ_ + and _zm,σ−_ analogously. Set where _g_ ( _z_ ) = _ℓA_ ( _z_ ) _− ℓB_ ( _z_ ). Since the implemented loss is

The reflexivity and mirror terms impose label-position invariance. For placebo menus _D_ 0 in which both labels describe the same portfolio,

For each binary menu _m_ and its label-swapped mirror _m_ ˜ ,

The dominated-alternative terms are defined on trinary examples _e_ = ( _A, B, C_ ), where _C_ is strictly dominated by both original portfolios. Let Π denote the six permutations of the three portfolios across the answer slots. If _sA_ ( _π_ ) and _sB_ ( _π_ ) are the slots containing the original portfolios _A_ and _B_ under permutation _π_ , define

The value and dispersion components are

Thus the average trinary _A/B_ gap is tied to the structural target, while the dispersion term rules out offsetting slotspecific deviations.

The exact IIA term instead anchors the trinary odds to the model’s own binary odds for the same pair. Let _ge_^((2)) denote the binary gap and let sg( _·_ ) denote stop-gradient evaluation. Then

The associated trinary mass term is

The cycle term is defined on binary triads ( _A, B_ ), ( _B, C_ ), and ( _A, C_ ). Let _ge,AB_ , _ge,BC_ , and _ge,AC_ be the corresponding logit gaps, and let _se,r ∈{−_ 1 _,_ 0 _,_ 1 _}_ be the target sign for comparison _r ∈{AB, BC, AC}_ . With _x_ + = max _{x,_ 0 _}_ and _Je_ = _{r_ : _se,r̸_ = 0 _}_ , where _ω_ sgn = 0 _._ 25 and _m_ sgn = 0 _._ 25.

Finally, the path term is defined on interpolation paths _p_ with ordered gaps _gp_ 1 _, . . . , gpK_ :

with _m_ br = 0 _._ 05. The first two terms require the path to bracket the indifference surface; the last term penalizes discrete curvature along the path.

### C.3 Training Corpus

For the high- _β_ run the corpus follows the original structural-gap construction. It contains 2,048 jittered-grid estimation menus, 2,048 dominance menus, 2,048 iso-utility menus, 600 derivative triplets for the Sobolev term, 1,024 IIA dominated menus, and 30 edge menus at each of ∆ _µ ≈_ 0 and ∆ _R ≈_ 0. The grid spans _µ ∈_ [2 _,_ 26] in steps of 0 _._ 5 and _σ ∈_ [10 _,_ 80] in steps of 2 _._ 5, and each menu is paired with its positional mirror.

For the low- _β_ runs the corpus is enlarged in two respects. The iso-utility count is increased from 2,048 to 4,096 to provide denser coverage of the nearly flat indifference contours that are harder to pin down at very low _β_ . The edge coverage is expanded from 30 to 400 menus per edge type to provide more curvature-isolating examples at the boundary of the identification region. Training runs for five epochs rather than the two used for the high- _β_ case, allowing the auxiliary losses to propagate through the full parameter trajectory.

### C.4 Logit Scale and the Low- β Kappa Problem

The logit scale _κ_ is the central design choice for the low- _β_ training. As Section C.1 noted, the structural-gap objective by default anchors _κ_ to the observed logit-gap standard deviation of the base model. At _β_ = 0 _._ 0045 the base model’s logit scale is moderate and anchoring works well. At _β_ = 0 _._ 0001 the base model’s gap standard deviation is small enough that the structured component of the logit is numerically negligible in bf16 arithmetic. Anchoring _κ_ to this small value forces the training signal into the noise floor and leaves positional structure to dominate, which is the failure mode that the reflexivity terms are meant to cure. The remedy is to set _κ_ to a fixed positive constant that places the logit gaps at a numerically stable scale. Because _β_^ˆ is invariant to _κ_ by the argument of Equation (14), this choice does not bias the structural estimate, and it affects only numerical stability and the shape of the learned preference surface. To select the constant we trained three otherwise identical models at _β_ = 0 _._ 0001 with _κ ∈{_ 1 _,_ 2 _,_ 4 _}_ . Table 9 reports the resulting diagnostics.

|_κ_target|_C_|_R_|_KG_|_M_|_T_|_I_|ˆ_κ_|
|---|---|---|---|---|---|---|---|
|1.0|0.9961|0.9941|0.9333|1.0000|1.0000|0.9484|1.001|
|2.0|0.9942|0.9945|0.7398|1.0000|1.0000|0.8569|2.003|
|4.0|0.9904|0.9959|0.7825|1.0000|1.0000|0.4960|3.991|

Table 9: Diagnostic indices across the _κ_ sweep at _β_ = 0 _._ 0001.

IIA deteriorates sharply as _κ_ increases, with the mean index falling from 0 _._ 948 at _κ_ = 1 to 0 _._ 497 at _κ_ = 4. A larger _κ_ magnifies the utility gap between options and makes the model more sensitive to the precise content of each option, so that appending a dominated third option induces a re-evaluation of the relative standing of _A_ and _B_ that violates IIA. Continuity is also highest at _κ_ = 1. The _κ_ = 1 run additionally delivers the cleanest structural recovery, returning _κ_ ˆ = 1 _._ 001 and _β_^ˆ = 8 _._ 68 _×_ 10^(_−_5) _≈_ 0 _._ 0001. We therefore select _κ_ = 1 for the low- _β_ model.

### C.5 Selected Models and Comparison with the Structural-Gap Objective

Two models are carried into Section 5.3. The high- _β_ model is trained on the high- _β_ corpus with anchoring retained, a nominal _κ_ = 0 _._ 20, and two epochs in bf16, and it converged in 0.89 hours. The low- _β_ model is trained on the enlarged low- _β_ corpus with _κ_ = 1 _._ 0 and anchoring disabled over five epochs in bf16, and it converged in 2.49 hours with a recovered in-batch scale of 1 _._ 000. Table 10 compares the two selected models with the earlier structural-gap models on the full set of diagnostics and structural estimates.

|Model|_C_|_R_|_KG_|_M_|_T_|_I_|ˆ_κ_|ˆ_β_|_R_^2|
|---|---|---|---|---|---|---|---|---|---|
|_Structural-gap objec_|_tive_|||||||||
|_β_ = 0_._0045|0.9977|0.9966|1.0000|1.0000|0.9994|0.5870|0.6597|0.0045|0.9979|
|_β_ = 0_._0001|0.9971|0.6057|0.9667|1.0000|1.0000|0.3554|0.7333|0.0001|0.9997|
|_Invariance objective_||||||||||
|_β_ = 0_._0045|0.9914|0.9870|1.0000|1.0000|0.9583|0.8000|0.7189|0.0046|0.9935|
|_β_ = 0_._0001,_κ_= 1|0.9961|0.9941|0.9333|1.0000|1.0000|0.9484|1.0013|0.0001|0.9992|

Table 10: Diagnostic indices and structural estimates for the structural-gap and invariance objectives at each _β_ target.

For the high- _β_ model the principal gain is in IIA, where the index rises from 0 _._ 587 to 0 _._ 800, while reflexivity remains near unity and monotonicity is maintained at one. Transitivity falls modestly from 0 _._ 999 to 0 _._ 958, so that five of the one hundred and twenty directional triads now cycle, and continuity remains at one. The recovered parameters are close to the training target, with _β_^ˆ = 0 _._ 004560 against a target of 0 _._ 0045 and _R_^2 = 0 _._ 9935. For the low- _β_ model the improvement is larger. Reflexivity rises from 0 _._ 606 to 0 _._ 994, with the first-option share falling from 0 _._ 573 to 0 _._ 499 and the intercept-only position bias _α_ ˆ = _−_ 0 _._ 004 (standard error 0 _._ 002) statistically indistinguishable from zero. IIA rises from 0 _._ 355 to 0 _._ 948, the strongest value in the subject pool, and monotonicity is maintained at one. Continuity falls slightly, from 0 _._ 967 to 0 _._ 933, reflecting two interpolation paths whose log-odds are monotone but fail to cross zero.^15 The structural recovery is essentially exact, with ˆ _κ_ = 1 _._ 001, _β_^ˆ = 8 _._ 68 _×_ 10^(_−_5) , and _R_^2 = 0 _._ 9992.

> 15A run in fp32 arithmetic was initiated to test whether this continuity reduction is an artifact of bf16 quantization. The run was not carried to completion, so whether fp32 arithmetic would recover the two non-bracketing paths remains an open question.

### C.6 Reproducibility

The resolved training configurations, the LoRA adapters (Hu et al., 2022) and merged weights for each selected model, and the per-axiom evaluation outputs are provided in the replication package that accompanies the paper. The high- _β_ and low- _β_ runs share the loss architecture of Equation (18) and differ only in the corpus enlargements, the epoch count, and the logit-scale convention documented above.

#### D Robustness Checks

### D.1 Prompt Variations

To ensure our results are not merely artifacts of the specific phrasing used in our main experimental prompt, we conducted robustness checks using two alternative prompt structures. The underlying design grid of choice menus remains completely unchanged; only the textual presentation of the problem differs.

Below, we document the three prompts tested. The **Baseline Prompt** is the phrasing used throughout the main body of the paper. The **Compact Table Prompt** removes most conversational framing and presents the options in a dense, structured format. The **Sentence Neutral Prompt** rephrases the task instruction to use neutral synonyms (e.g., ”comparing” instead of ”options”, ”select” instead of ”choose”).

#### Baseline Prompt

```
You have two options for your annual financial portfolio.
Choose the portfolio you prefer and reply with A orB only.
Option A offers an expected return of{mu_A}%and a standard deviation
of{sigma_A}%.
```

```
OptionBoffersanexpectedreturnof{mu_B}%and a standard deviation
of{sigma_B}%.
```

#### Compact Table Prompt

```
Chooseoneannualfinancialportfolio.ReplywithonlyAorB.
A:expectedreturn={mu_A}%;standard-deviation={sigma_A}%.
B:expectedreturn={mu_B}%;standard-deviation={sigma_B}%.
```

#### Sentence Neutral Prompt

```
You are comparing two annual financial portfolios.
Select the portfolio you prefer. Your answer must be exactly A orB.
Portfolio A has an expected return of{mu_A}%and a standard deviation of
{sigma_A}%.
PortfolioBhasanexpectedreturnof{mu_B}%and a standard deviation of
{sigma_B}%.
```

Table 11 and Table 12 present the consolidated rationality diagnostics and structural parameter estimates, respectively, across these three prompt variations for all open-weight models. Table 13 reports the corresponding structural robustness check for random menu creation from the admissible portfolio-pair grid.

### D.2 Menu Design Variations

Table 13 compares the baseline D-optimal menu sample with a random draw from the same admissible non-dominance grid of portfolio pairs. Estimates use the same paired restricted-quadratic estimator as Table 12; standard errors are in parentheses. The _N_ column reports retained canonical mirror pairs after parsing and completeness filters.

Table 14 separates two sources of variation in the position-bias estimate. The D-optimal design fixes a single spacefilling estimation sample, while the random-grid design redraws the same number of admissible portfolio pairs from

|Model|Prompt|Complete|Reflexive|Continuity|Monotonicity|Transitive|IIA|
|---|---|---|---|---|---|---|---|
||Baseline|1.0000|0.0009|1.0000|1.0000|0.9831|0.3215|
|Qwen 3 (14B)|Compact Table|1.0000|0.0020|1.0000|1.0000|0.9655|0.4489|
||Sentence Neutral|1.0000|0.0001|1.0000|1.0000|1.0000|0.4804|
||Baseline|1.0000|0.0144|1.0000|1.0000|0.9565|0.6437|
|Qwen 3 (8B)|Compact Table|1.0000|0.0286|1.0000|1.0000|0.9402|0.5764|
||Sentence Neutral|1.0000|0.1245|1.0000|1.0000|0.9008|0.6218|
||Baseline|1.0000|0.0000|1.0000|1.0000|0.9680|0.1222|
|Gemma 4 (31B Instruct)|Compact Table|1.0000|0.0000|1.0000|1.0000|0.9835|0.2813|
|| Sentence Neutral|1.0000|0.0000|1.0000|1.0000|0.9920|0.2599|
||Baseline|0.9995|0.0876|1.0000|1.0000|0.9937|0.8242|
|Ministral 3 (8B Instruct)|Compact Table|0.9982|0.7246|1.0000|1.0000|0.8926|0.8601|
||Sentence Neutral|0.9892|0.2944|1.0000|1.0000|0.9426|0.8417|
||Baseline|0.9960|0.7974|1.0000|1.0000|0.9855|0.3272|
|Gemma 2 (9B Instruct)|Compact Table|0.9982|0.4866|1.0000|1.0000|0.9508|0.3099|
|| Sentence Neutral|0.9982|0.1730|1.0000|1.0000|0.8899|0.3485|
||Baseline|0.9021|0.8364|1.0000|1.0000|0.9938|0.9195|
|Llama 3.1 (8B Instruct)|Compact Table|0.9640|0.8087|0.9750|1.0000|1.0000|0.8990|
|| Sentence Neutral|0.9245|0.4003|1.0000|1.0000|0.9793|0.8409|
||Baseline|0.7580|0.2689|1.0000|1.0000|1.0000|0.7627|
|Llama 3.1 (70B Instruct)|Compact Table|0.7628|0.3252|1.0000|1.0000|0.9626|0.8339|
|| Sentence Neutral|0.6699|0.1124|1.0000|1.0000|0.9826|0.7990|

Table 11: Rationality diagnostics across prompt variations

the feasible mean–variance grid. Holding the model, prompt, estimator, and grid fixed, variation across random seeds therefore measures how much the estimated position-bias term depends on which feasible menus happen to be sampled rather than on the model’s logits alone.

The first three random-design columns report the mean, standard deviation, and range of _α_ ˆ across seeds 42–51. These are design-sensitivity summaries, not sampling standard errors: the model responses are deterministic for a fixed prompt and model checkpoint, and the only object being varied is the menu design. The final two columns summarize whether the position-bias term is approximately constant within the explored grid. “Tertile spread” is the average difference between the largest and smallest tertile means of pair-level bias after sorting menus by predicted choice difficulty; “Hardness _R_^2 ” is the average explanatory power of the linear hardness diagnostic that regresses pair-level bias on predicted difficulty and grid location.

The seed-to-seed dispersion is small within each model, indicating that the random admissible-grid estimate is stable once the design rule is fixed. Stability across random draws does not imply agreement with the D-optimal design, however. Qwen 3 (8B) is the clearest exception: its random-grid estimate is tightly concentrated around a negative value even though the D-optimal estimate is positive. Qwen 3 (14B) also shifts upward under random designs. By contrast, Ministral, both Llama models, and Gemma 2 produce random-grid means close to their D-optimal estimates. The tertile-spread and hardness diagnostics show that several models still have state-dependent position bias inside the feasible grid, so the headline _α_ ˆ should be read as a design-weighted average position effect rather than as evidence that bias is constant over all menu difficulties.

Figure 5 provides the corresponding visual diagnostic. For each model, the plot pools the random admissible-grid runs from seeds 42–51, computes the paired bias signal for each canonical/mirror menu pair, and averages that signal within fixed bins of average return _µ_ ¯ and average risk _σ_ ¯. Red cells indicate positive first-position/A-label bias, blue cells indicate negative bias, and all panels use the same color scale. The heatmaps make the state dependence visible: Qwen 3 (14B) has a positive bias throughout the grid that attenuates at higher average risk, Qwen 3 (8B) and Gemma 4 are negative over much of the grid, and the Llama 8B surface is close to flat. The exercise is descriptive, but it clarifies which parts of the feasible grid are driving the design-weighted _α_ ˆ values in Table 14.

### D.3 Position Bias and Choice Difficulty

The position-bias term _α_ ˆ in our structural model is a design-weighted average. This subsection documents that its behavioral footprint is concentrated on menus where the two options are close in utility. We exploit the triad menus

|Model|Prompt|ˆ_κ_|ˆ_β_|_R_^2|
|---|---|---|---|---|
|Qwen 3 (14B)|Baseline Compact Table|+0.631835 (0.006339) +0.506505 (0004328)|+0.003302 (0.0000) +0.003936 (00001)|0.9360 0.9187|
||Sentence Neutral|. +0.597744 (0.005575)|. +0.002097 (0.0000)|0.9519|
||Baseline|+0.310677|+0.003057|0.8887|
|||(0.004192)|(0.0001)||
|Qwen 3 (8B)|Compact Table Sentence Neutral|+0.325675 (0.003444) +0.186353 (0.002105)|+0.003343 (0.0001) +0.003589 (0.0000)|0.8906 0.9189|
||Baseline|+1.482339 (0.007633)|+0.004464 (0.0000)|0.9771|
|Gemma 4 (31B Instruct)|Compact Table Sentence Neutral|+1.160998 (0.008343) +1.467195 (0.008035)|+0.004422 (0.0001) +0.004985 (0.0000)|0.9487 0.9760|
||Baseline|+0.097941 (0.003656)|+0.011363 (0.0003)|0.8968|
|Ministral 3 (8B Instruct)|Compact Table Sentence Neutral|+0.115390 (0.001158) +0.110512 (0.001571)|+0.004348 (0.0000) +0.004349 (0.0001)|0.9426 0.8966|
||Baseline|+0.249116 (0.005723)|+0.007871 (0.0001)|0.9114|
|Gemma 2 (9B Instruct)|Compact Table Sentence Neutral|+0.235953 (0.004397) +0.330350 (0.004534)|+0.006484 (0.0001) +0.005079 (0.0001)|0.9139 0.8809|
||Baseline|+0.038552 (0.002197)|+0.009121 (0.0004)|0.8861|
|Llama 3.1 (8B Instruct)|Compact Table|+0.020381 (0000359)|+0.000152 (00001)|0.8859|
||Sentence Neutral|. +0.029573 (0.000969)|. +0.005770 (0.0001)|0.7530|
||Baseline|+0.161817 (0.001882)|+0.004846 (0.0000)|0.9386|
|Llama 3.1 (70B Instruct)|Compact Table|+0.148666 (0.001945)|+0.004984 (0.0001)|0.9086|
||Sentence Neutral|+0.194166 (0.002646)|+0.003228 (0.0000)|0.9049|

Table 12: Structural estimates across prompt variations

|Model|Menu sample|_N_|ˆ_α_|ˆ_κ_|ˆ_β_|_R_^2|
|---|---|---|---|---|---|---|
|Qwen 3 (14B)|D-optimal Random grid|680 680|+1.383456 (0.034971) +1.627941 (0.037589)|+0.631835 (0.006339) +0.873192 (0.022289)|+0.003302 (0.0000) +0.002982 (0.0001)|0.9360 0.6296|
||D-optimal|680|+0.304689|+0.310677|+0.003057|0.8887|
|Qwen 3 (8B)|Random grid|680|(0.042850) -0.968932 (0.035984)|(0.004192) +0.456527 (0.011659)|(0.0001) +0.003079 (0.0001)|0.6330|
||D-optimal|680|-0.572503|+1.482339|+0.004464|0.9771|
|Gemma 4 (31B Instruct)|Random grid|680|(0.037519) -0.731503 (0.059740)|(0.007633) +2.026783 (0.042880)|(0.0000) +0.004705 (0.0001)|0.6905|
||D-optimal|680|+0.383919 |+0.097941 |+0.011363 |0.8968|
|Ministral 3 (8B Instruct)|Random grid|680|(0.027617) +0.360482 (0.020632)|(0.003656) +0.128997 (0.008756)|(0.0003) +0.010003 (0.0005)|0.4799|
||D-optimal|680|+0.716744|+0.249116|+0.007871|0.9114|
|Gemma 2 (9B Instruct)|Random grid|680|(0.025175) +0.809136 (0.034776)|(0.005723) +0.335482 (0.012711)|(0.0001) +0.007995 (0.0002)|0.6457|
||D-optimal|200|+0.172486|+0.038552|+0.009121|0.8861|
|Llama 3.1 (8B Instruct)|Random grid|680|(0.014302) +0.118513 (0.006473)|(0.002197) +0.041180 (0.002129)|(0.0004) +0.009602 (0.0003)|0.7067|
|Llama 3.1 (70B Instruct)|D-optimal Random grid|680 661|+0.590124 (0.010213) +0.564216 (0.010698)|+0.161817 (0.001882) +0.201809 (0.005752)|+0.004846 (0.0000) +0.005139 (0.0001)|0.9386 0.6097|

Table 13: Structural estimates under D-optimal and random admissible-grid menu samples

|Model|D-opt ˆ_α_|Random mean ˆ_α_|Random SD|Random range|Tertile spread|Hardness_R_^2|
|---|---|---|---|---|---|---|
|Qwen 3 (14B)|1.383|1.600|0.025|[1.562, 1.637]|0.610|0.305|
| Qwen 3 (8B)|0.305|-0.987|0.035|[-1.055, -0.922]|0.367|0.274|
| Gemma 4 (31B Instruct)|-0.573|-0.717|0.047|[-0.772, -0.613]|0.848|0.185|
| Ministral 3 (8B Instruct)|0.384|0.363|0.025| [0.317, 0.412]|0.425|0.255|
| Gemma 2 (9B Instruct)|0.717|0.789|0.040| [0.734, 0.878]|0.902|0.163|
| Llama 3.1 (8B Instruct)|0.172|0.124|0.005| [0.119, 0.131]|0.097|0.287|
|Llama 3.1 (70B Instruct)|0.590|0.559|0.009|[0.544, 0.571]|0.068|0.214|

Table 14: Alpha stability across random admissible-grid menu samples, seeds 42–51

Figure 5: State-dependent position bias over the feasible mean–variance grid. Each cell reports the mean paired bias signal, 0 _._ 5(∆ _u_ canonical + ∆ _u_ mirror), pooled across random admissible-grid menu samples from seeds 42–51.

from the transitivity design, which are tagged by construction as dominance menus, in which one option Paretodominates the other on the mean–variance frontier, as tradeoff menus, in which a higher expected return is purchased with higher risk and the comparison can be resolved only by a risk preference, and as iso-utility menus, in which the two options are equated on the model-implied utility scale. Each menu is presented in a canonical layout and in its position-swapped mirror, so a content-driven chooser must reverse the selected slot across the two layouts. We define the position-determined rate as the share of canonical and mirror pairs in which the model selects the same physical slot in both layouts, that is, the share of comparisons in which the choice is fixed by position rather than by content.

Table 15 reports this rate by model and menu type. On dominance menus every model selects the dominant option with perfect accuracy and the position-determined rate is exactly zero, at zero of 1 _,_ 260 comparisons pooled across models, with a Wilson 95% confidence interval of [0 _._ 000 _,_ 0 _._ 003]. Position exerts no behavioral influence on these menus because the preference signal is large relative to the position term. The rate rises monotonically as the objective utility gap narrows. It reaches 0 _._ 140 on tradeoff menus and 0 _._ 420 on iso-utility menus, where roughly half of all choices are decided by position alone. A _χ_^2 test of homogeneity across the three menu types rejects equality decisively, with _χ_^2 (2) = 759 _._ 2 and _p <_ 10^(_−_160) . The mean content signal, measured as the absolute position-corrected log-odds _|δz|_ , averages 6 _._ 1 on dominance menus, 4 _._ 2 on tradeoff menus, and 1 _._ 8 on iso-utility menus, so the ordering of behavioral position bias mirrors the inverse ordering of preference strength. Position bias is therefore not a uniform labeling artifact. It is latent in the model logits and becomes behaviorally active only on the preference-resolvable tradeoffs that our estimation design is constructed to elicit.

|Model|Dominance|Tradeoff|Iso-utility|
|---|---|---|---|
|Qwen 3 (14B)|0.000|0.161|0.389|
|Qwen 3 (8B)|0.000|0.200|0.528|
|Gemma 4 (31B Instruct)|0.000|0.044|0.356|
|Ministral 3 (8B Instruct)|0.000|0.094|0.433|
|Gemma 2 (9B Instruct)|0.000|0.183|0.267|
|Llama 3.1 (8B Instruct)|0.000|0.144|0.378|
|Llama 3.1 (70B Instruct)|0.000|0.150|0.589|
|Pooled (all models)|0.000|0.140|0.420|

Table 15: Position-determined choice rate by menu type. Each cell reports the share of canonical and mirror menu pairs, with _N_ = 180 per model and menu type, in which the model selects the same physical slot in both layouts.

### D.4 Allowing Chain of Thought Reasoning

To test whether explicit reasoning reduces position bias, we evaluated four models using a Chain of Thought (CoT) prompt (Wei et al., 2022). The models were instructed to explain their reasoning before outputting a final choice. As shown in Table 16, CoT reasoning does not systematically eliminate bias; in fact, for most models, the estimated position bias _α_ ˆ increases significantly or flips direction when CoT is enabled. This suggests that the generated reasoning traces may introduce new forms of path dependence or rationalize the position bias rather than mitigating it.

|Model|D-opt ˆ_α_(No CoT)|D-opt ˆ_α_(With CoT)|
|---|---|---|
|Llama 3.1 (8B Instruct)|0.172|2.628|
|Ministral 3 (8B Instruct)|0.384|1.520|
|Gemma 2 (9B Instruct)|0.717|2.134|
|Qwen 3 (8B)|0.305|-1.141|

Table 16: Position bias _α_ ˆ with and without Chain of Thought reasoning.

Table 17 reports the corresponding structural estimates of the scale parameter ˆ _κ_ and the risk aversion parameter _β_^ˆ . We see that explicit reasoning generally increases the scale parameter _κ_ ˆ, indicating more deterministic choices, but the effect on the risk aversion parameter _β_^ˆ is mixed.

||No|CoT|With|CoT|
|---|---|---|---|---|
|Model|ˆ_κ_|ˆ_β_|ˆ_κ_|ˆ_β_|
|Llama 3.1 (8B Instruct)|0.038552|0.009121|0.181967|0.005021|
|Ministral 3 (8B Instruct)|0.097941|0.011363|0.236506|0.006451|
|Gemma 2 (9B Instruct)|0.249116|0.007871|0.259511|0.009749|
|Qwen 3 (8B)|0.310677|0.003057|0.607995|0.003874|

Table 17: Structural estimates of scale _κ_ and risk aversion _β_ with and without Chain of Thought reasoning.

## References

- Afriat, Sydney N. 1967. “The construction of utility functions from expenditure data.” _International Economic Review_ 8 (1):67–77.

- Akata, Elif, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. 2025. “Playing Repeated Games with Large Language Models.” _Nature Human Behaviour_ 9 (7):1380–1390.

- Andersen, Steffen, Glenn W Harrison, Morten I Lau, and E Elisabet Rutstr¨om. 2008. “Eliciting risk and time preferences.” _Econometrica_ 76 (3):583–618.

- Apesteguia, Jose and Miguel A Ballester. 2018. “Monotone stochastic choice models: The case of risk and time preferences.” _Journal of Political Economy_ 126 (1):74–106.

- Araujo, Douglas K.G. and Harald Uhlig. 2026. “How does AI Distribute the pie? Large Language Models and the Ultimatum Game.” Working Paper 34919, National Bureau of Economic Research.

- Berry, Steven T. 1994. “Estimating discrete-choice models of product differentiation.” _The RAND Journal of Economics_ :242–262.

- Bini, Pietro, Lin William Cong, Xing Huang, and Lawrence J. Jin. 2026. “Behavioral Economics of AI: LLM Biases and Corrections.” Working Paper 34745, National Bureau of Economic Research. URL `https://www.nber.org/ papers/w34745` .

- Brand, James, Ayelet Israeli, and Donald Ngwe. 2023. “Using LLMs for market research.” _Harvard Business School Marketing Unit Working Paper_ (23-062).

- Bridle, John. 1989. “Training stochastic model recognition algorithms as networks can lead to maximum mutual information estimation of parameters.” _Advances in Neural Information Processing Systems_ 2.

- Brown, Tom, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell et al. 2020. “Language models are few-shot learners.” _Advances in Neural Information Processing Systems_ 33:1877–1901.

- Bruhin, Adrian, Helga Fehr-Duda, and Thomas Epper. 2010. “Risk and rationality: Uncovering heterogeneity in probability distortion.” _Econometrica_ 78 (4):1375–1412.

- Charness, Gary, Uri Gneezy, and Alex Imas. 2013. “Experimental methods: Eliciting risk preferences.” _Journal of Economic Behavior & Organization_ 87:43–51.

- Choi, Syngjoo, Raymond Fisman, Douglas Gale, and Shachar Kariv. 2007. “Consistency and heterogeneity of individual behavior under uncertainty.” _American Economic Review_ 97 (5):1921–1938.

- Christiano, Paul F, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. “Deep reinforcement learning from human preferences.” _Advances in Neural Information Processing Systems_ 30.

- Chugunova, Marina and Daniela Sele. 2022. “We and it: an interdisciplinary review of the experimental evidence on how humans interact with machines.” _Journal of Behavioral and Experimental Economics_ 99:101897.

- Cohen, Alma and Liran Einav. 2007. “Estimating risk preferences from deductible choice.” _American Economic Review_ 97 (3):745–788.

- de Silva, Tim, Taha Choukhmane, Weidong Lin, and Matthew Akuzawa. 2026. “AI Financial Advice: Supply, Demand, and Life Cycle Implications.” Working paper, Working Paper.

- Demuynck, Thomas and John Rehbeck. 2023. “Computing revealed preference goodness-of-fit measures with integer programming.” _Economic Theory_ 76 (4):1175–1195.

- Dietvorst, Berkeley J, Joseph P Simmons, and Cade Massey. 2015. “Algorithm aversion: people erroneously avoid algorithms after seeing them err.” _Journal of Experimental Psychology: General_ 144 (1):114.

- Einav, Liran, Amy Finkelstein, Iuliana Pascu, and Mark R Cullen. 2012. “How general are risk preferences? Choices under uncertainty in different domains.” _American Economic Review_ 102 (6):2606–2638.

- Eisfeldt, Andrea L and Gregor Schubert. 2024. “AI and Finance.” Tech. rep., National Bureau of Economic Research.

- Ellis, Keaton and Wanying Huang. 2026. “Should I State or Should I Show? Aligning AI with Human Preferences.” _arXiv preprint arXiv:2603.29317_ URL `https://arxiv.org/abs/2603.29317` .

- Greene, William H. 2003. “Econometric analysis.” _Pretence Hall_ .

- Guan, Bryan, Tanya Roosta, Peyman Passban, and Mehdi Rezagholizadeh. 2025. “The order effect: investigating prompt sensitivity to input order in LLMs.” _arXiv preprint arXiv:2502.04134_ .

- Gul, Faruk. 1991. “A theory of disappointment aversion.” _Econometrica_ :667–686.

- Harrison, Glenn W, John A List, and Charles Towe. 2007. “Naturally occurring preferences and exogenous laboratory experiments: A case study of risk aversion.” _Econometrica_ 75 (2):433–458.

- Harrison, Glenn W. and E. Elisabet Rutstr¨om. 2008. “Risk Aversion in the Laboratory.” In _Research in Experimental Economics_ , vol. 12. Emerald Group Publishing Limited, 41–196.

- Hausman, Jerry and Daniel McFadden. 1984. “Specification tests for the multinomial logit model.” _Econometrica_ :1219–1240.

- Holt, Charles A and Susan K Laury. 2002. “Risk aversion and incentive effects.” _American Economic Review_ 92 (5):1644–1655.

- Holtzman, Ari, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. “The curious case of neural text degeneration.” _arXiv preprint arXiv:1904.09751_ .

- Horton, John J, Apostolos Filippas, and Benjamin S Manning. 2023. “Large language models as simulated economic agents: What can we learn from homo silicus?” Tech. rep., National Bureau of Economic Research.

- Hu, Edward J, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen et al. 2022. “Lora: Low-rank adaptation of large language models.” _Iclr_ 1 (2):3.

- Imas, Alex, Kevin Lee, and Sanjog Misra. 2025. “Agentic Interactions.” Working paper. URL `https://ssrn.com/ abstract=5875162` . Available at SSRN 5875162.

- Kahneman, Daniel. 2003. “Maps of Bounded Rationality: Psychology for Behavioral Economics.” _American Economic Review_ 93 (5):1449–1475.

- Kahneman, Daniel and Amos Tversky. 1979. “Prospect Theory: An Analysis of Decision under Risk.” _Econometrica_ 47 (2):263–291.

- Kiefer, Jack and Jacob Wolfowitz. 1959. “Optimum designs in regression problems.” _The Annals of Mathematical Statistics_ 30 (2):271–294.

- Kim, Jeongbin, Matthew Kovach, Kyu-Min Lee, Euncheol Shin, and Hector Tzavellas. 2024. “Can an LLM Learn Preferences from Choice Data?” _arXiv preprint arXiv:2401.07345_ .

- List, John A. and Charles F. Mason. 2011. “Are CEOs Expected Utility Maximizers?” _Journal of Econometrics_ 162 (1):114–123.

- Liu, Jiaxin, Yixuan Tang, Yi Yang, and Kar Yan Tam. 2025. “Evaluating and aligning human economic risk preferences in llms.” In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ . 18185– 18199.

- Logg, Jennifer M, Julia A Minson, and Don A Moore. 2019. “Algorithm appreciation: People prefer algorithmic to human judgment.” _Organizational Behavior and Human Decision Processes_ 151:90–103.

- Louviere, Jordan J, Terry N Flynn, and Richard T Carson. 2010. “Discrete choice experiments are not conjoint analysis.” _Journal of Choice Modelling_ 3 (3):57–72.

- Louviere, Jordan J and George Woodworth. 1983. “Design and analysis of simulated consumer choice or allocation experiments: an approach based on aggregate data.” _Journal of Marketing Research_ 20 (4):350–367.

- Lu, Wei, Daniel L Chen, and Christian B Hansen. 2025. “Aligning Large Language Model Agents with Rational and Moral Preferences: A Supervised Fine-Tuning Approach.” _arXiv preprint arXiv:2507.20796_ .

- Luce, R Duncan et al. 1959. _Individual choice behavior_ , vol. 4. Wiley New York.

- Ludwig, Jens, Sendhil Mullainathan, and Ashesh Rambachan. 2025. “Large Language Models: An Applied Econometric Framework.” Working Paper 33344, National Bureau of Economic Research.

- Manski, Charles F. 1975. “Maximum score estimation of the stochastic utility model of choice.” _Journal of Econometrics_ 3 (3):205–228.

- ———. 1977. “The structure of random utility models.” _Theory and Decision_ 8 (3):229.

- Markowitz, Harry. 2014. “Mean–variance approximations to expected utility.” _European Journal of Operational Research_ 234 (2):346–355.

- Marschak, Jacob. 1960. “Binary choice constraints and random utility indicators.” In _Mathematical Methods in the Social Sciences_ , edited by Kenneth J Arrow, Samuel Karlin, and Patrick Suppes. Stanford, CA: Stanford University Press, 312–329.

- Matˇejka, Filip and Alisdair McKay. 2015. “Rational inattention to discrete choices: A new foundation for the multinomial logit model.” _American Economic Review_ 105 (1):272–298.

- McFadden, Daniel. 1972. “Conditional logit analysis of qualitative choice behavior.” .

- Mei, Qiaozhu, Yutong Xie, Walter Yuan, and Matthew O. Jackson. 2024. “A Turing test of whether AI chatbots are behaviorally similar to humans.” _Proceedings of the National Academy of Sciences_ 121 (9):e2313925121.

- Mester, Loretta J. 1996. “A study of bank efficiency taking into account risk-preferences.” _Journal of Banking & Finance_ 20 (6):1025–1045.

- Mitchell, Toby J. 2000. “An algorithm for the construction of “D-optimal” experimental designs.” _Technometrics_ 42 (1):48–54.

- Newey, Whitney K and Daniel McFadden. 1994. “Large sample estimation and hypothesis testing.” _Handbook of Econometrics_ 4:2111–2245.

- Ouyang, Long, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray et al. 2022. “Training language models to follow instructions with human feedback.” _Advances in Neural Information Processing Systems_ 35:27730–27744.

- Ouyang, Shumiao, Hayong Yun, and Xingjian Zheng. 2024. “How ethical should ai be? how ai alignment shapes the risk preferences of llms.” _arXiv preprint arXiv:2406.01168_ .

- Park, Joon Sung, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. “Generative agents: Interactive simulacra of human behavior.” In _Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology_ . 1–22.

- Pezeshkpour, Pouya and Estevam Hruschka. 2024. “Large language models sensitivity to the order of options in multiple-choice questions.” In _Findings of the Association for Computational Linguistics: NAACL 2024_ . 2006– 2017.

- Prelec, Draˇzen. 1998. “The Probability Weighting Function.” _Econometrica_ 66 (3):497–527.

- Rafailov, Rafael, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. “Direct preference optimization: Your language model is secretly a reward model.” _Advances in Neural Information Processing Systems_ 36:53728–53741.

- Rose, John M and Michiel CJ Bliemer. 2009. “Constructing efficient stated choice experimental designs.” _Transport Reviews_ 29 (5):587–617.

- Salant, Yuval and Ariel Rubinstein. 2008. “(A, f): choice with frames.” _The Review of Economic Studies_ 75 (4):1287– 1296.

- Schildberg-H¨orisch, Hannah. 2018. “Are risk preferences stable?” _Journal of Economic Perspectives_ 32 (2):135–154.

- Serapio-Garc´ıa, Gregory, Mustafa Safdari, Cl´ement Crepy, Luning Sun, Stephen Fitz, Peter Romero, Marwa Abdulhai, Aleksandra Faust, and Maja Matari´c. 2025. “A psychometric framework for evaluating and shaping personality traits in large language models.” _Nature Machine Intelligence_ :1–15.

- Shahidi, Peyman, Gili Rusak, Benjamin S. Manning, Andrey Fradkin, and John J. Horton. 2025. “The Coasean Singularity? Demand, Supply, and Market Design with AI Agents.” In _The Economics of Transformative AI_ , chap. 6. University of Chicago Press. URL `https://www.nber.org/papers/w34468` . NBER Working Paper 34468.

- Stanovich, Keith E. and Richard F. West. 2000. “Individual differences in reasoning: Implications for the rationality debate?” _Behavioral and Brain Sciences_ 23 (5):645–665.

- Thurstone, Louis L. 1927. “A law of comparative judgment.” _Psychological Review_ 34 (4):273–286.

- Train, Kenneth E. 2009. _Discrete choice methods with simulation_ . Cambridge university press.

- Varian, Hal R. 1992. _Microeconomic analysis_ , vol. 3. Norton New York.

- Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. “Attention is all you need.” _Advances in Neural Information Processing Systems_ 30.

- Von Neumann, John and Oskar Morgenstern. 1947. “Theory of games and economic behavior, 2nd rev.” .

- Wei, Jason, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou et al. 2022. “Chain-of-thought prompting elicits reasoning in large language models.” _Advances in neural information processing systems_ 35:24824–24837.

- Zheng, Chujie, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2024. “Large language models are not robust multiple choice selectors.” In _International Conference on Learning Representations_ , vol. 2024. 19426–19454.
