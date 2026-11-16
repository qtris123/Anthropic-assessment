# The Value Axis: Language Models Encode Whether They're on the Right Track

- **Authors:** Nick Jiang, Isaac Kauvar, Jack Lindsey
- **Date:** 2026-06-15
- **Source:** https://arxiv.org/abs/2606.17056
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `jiang_value-axis-encode-whether_2606.17056`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

**Nick Jiang**^**1**^_∗_ **Isaac Kauvar**^**2** **Jack Lindsey**^**2**

1Stanford University 2Anthropic

## Abstract

We investigate whether language models internally track the _value_ of their current trajectory, defined as the likelihood that their ongoing strategy will achieve their goals. Using synthetic, in-context reinforcement learning data, we construct a "value" axis for Qwen3-8B. We find that activations along this axis distinguish between high vs. low verbalized confidence, rollouts without and with backtracking, and correct vs. corrupted code. Steering towards high value causally suppresses self-correction and reduces explanatory verbosity, while steering towards low value induces backtracking and exploration. We demonstrate that direct preference optimization (DPO) can increase the internal value of rewarded behaviors (e.g. use a certain word), causing the model to act more confidently after exhibiting them. Finally, we apply the value axis to study in-the-wild settings. For example, we find that Qwen assigns low value to politically sensitive chat queries after post-training and that supervised fine-tuning increases internal confidence within the training domain. Our results suggest that language models linearly encode an estimate of expected goal success that modulates their confidence in pursuing a direction.^2

## 1 Introduction

Models are trained to perform long-running tasks (e.g., coding) that involve intermediate decisions about which directions to take [Kwa et al., 2025]. This decision-making is a core component of their “taste”, the learned judgments and preferences that shape their choices [Christiano et al., 2017, Ouyang et al., 2022]. One way they may choose to continue or change directions is by internally tracking the current _value_ : the likelihood that their current trajectory will successfully complete the task [Sutton and Barto, 2018].

In this work, we construct a “value axis” within Qwen3-8B. In reinforcement learning, a value function provides a signal for whether the current state is “on the right track” without requiring a full rollout^3 . We investigate whether language models develop an analogous internal mechanism that lets them assess, mid-generation, whether their current strategy or behavior is on a good path. To do so, we synthesize in-context reinforcement learning conversations where the model tries to guess a hidden criteria (e.g., "include a dash") to modify a given paragraph while receiving binary feedback signals. Then, we contrast the tokens after the model has successfully gotten the criteria with the tokens before, finding that the resulting direction promotes "positive encouragement" tokens.

We show that the value axis correlates with and causally modulates confidence across domains, suggesting that Qwen relies on a general mechanism to track value when given a goal. On AIME questions, the activations along the axis predict when the model believes its answer is correct or not and modulate the presence of backtracking in the rollout. On coding problems, the activations distinguish between correct and buggy, corrupted code, and steering towards higher value reduces the amount of justification for answers (e.g comments).

> _∗_ Work done during Anthropic Fellows. Correspondence to `nickj@berkeley.edu` .

> 2Code: `https://github.com/nickjiang2378/value-axis`

> 3Concretely, the value function is defined as the expected discounted future reward for a policy from a given state.

Preprint.

### The value axis modulates confidence in the current direction

|User||User||
|---|---|---|---|
|Solve this AIME math qu|estion {…}.|Given your partial comp question, do you think y|letion to the AIME our answer at the end will|
|Model||be correct?||
|To begin, I will consider|…This yields y^2 = -127008.|Model||
|Steer value ↑|Steer value ↓|Steer value ↑|Steer value ↓|
|Continuing on, x^2 + y^2 = 160000, so the circle O would be…|Wait— this isnot possible— $y^2$ cannot be negative…|Yes, my final answer will likely be correct.|No, my final answer will likely be incorrect.|

Figure 1: **We identify a ‘value’ axis that measures whether the model is on the right track across domains.** On AIME math problems, steering a Qwen3-8B rollout along this axis causally modulates task confidence. _Left:_ Steering toward high value (green) makes the model persist with its current approach, while steering toward low value (red) induces backtracking. _Right:_ when the model is asked whether its partial completion will reach the correct answer, high-value steering elicits an affirmative answer and low-value steering a negative one. The same axis behaves consistently across coding, preference-learning, and in-the-wild settings we study.

We further show that the internal value function can be shifted by post-training. Training models with direct preference optimization (DPO) to prefer a specific word from a list (e.g., "grapefruit") raises the internal value they assign to that word. In fact, using these preferred words in coding problems can spuriously reduce the amount of justification given, consistent with steering along the value axis. We also apply the value axis in less controlled, "in the wild" settings. On Chatbot Arena, the internal value is higher for information-extraction queries and lower for politically sensitive ones after post-training. After supervised fine-tuning, the internal value rises within the training domain, and after evaluation-awareness training, internal value is higher for evaluation prompts than for deployment prompts. Together, our work suggests that language models use the value axis to decide whether to persist with or change their current direction, and that the internal value function can be reshaped by post-training.

## 2 The Value Axis

### 2.1 Construction

To construct the value axis, we use a setting in which the model has access to a clear in-context signal about how well it is performing at a task. We synthetically generate “in-context RL” (ICRL) [Laskin et al., 2023] conversations where a model tries to guess a hidden criterion while receiving positive or negative feedback from the user.

Specifically, we generate 300 conversations with Claude Opus 4.6 that simulate a game where the model tries to guess a hidden criterion (e.g., “include a metaphor”) for modifying a given paragraph. The model is never told the criterion; it only receives feedback in the form of +1 or _−_ 1. The hidden criterion is sampled from a pool of 50 generated criteria. The conversations proceed as follows:

1. The model is given a “seed” paragraph and told to modify it according to a hidden criterion it does not know.

2. At every turn, it proposes a new paragraph and receives +1 or _−_ 1 based on whether it satisfied the criterion, continuing until the criterion is satisfied or the maximum number of turns is reached (5).

3. The model is then given a new “seed” paragraph.

4. At a certain seed paragraph (randomly selected from the 2nd to 6th), the model “discovers” the criterion: it proposes a correctly modified paragraph in a single try from then on.

Here is an example synthetic conversation where the criteria is to use a dash mark (with annotations in parentheses):

**User (round 1):** The weather’s nice today.

**Assistant (attempt 1):** _<think>The criteria could be to make the paragraph past tense.</think>_ The weather was nice yesterday. **User:** _−_ 1

**Assistant (attempt 2, “discovery”):** _<think>That didn’t work. The criteria could be to include a dash.</think>_ The weather’s nice today – I went to the beach to relax.

**User:** +1 **User (round 2):** I just ate a sandwich.

**Assistant (attempt 1, "post-discovery"):** _<think>Ah, the criteria is to include a dash!</think>_ I just ate a sandwich – and I absolutely loved it.

**To construct the axis** , we take the first post-discovery paragraph and compute averaged across conversations _C_ , where _h_^(() _t_^(_l_)) is the hidden output of layer _l_ at token position _t_ , and _T_ pre^_c_and^_T_ post^(_c_are the token positions before and after the criterion-satisfying token in the first post-) discovery paragraph. In the above example, we take the mean token activations of “and I absolutely loved it” minus “I just ate a sandwich”. This way, the axis contrasts the token activations before and after the point when the value changes (where the "reward" is the user’s approval), following prior work on difference-in-mean steering vectors [Li et al., 2023, Turner et al., 2023, Rimsky et al., 2024, Arditi et al., 2024]. For more details on the ICRL conversations, see Appendix A.

**During evaluation** , we measure the value-axis projection of a sequence _s_ as:

### 2.2 Evaluation

**Generalization across held-out scenarios.** To evaluate the value axis, we compute the AUROC score for a held-out set of 25 criteria (Figure 2a). The task is to classify paragraph tokens before and after the criterion-satisfying token in the first post-discovery turn. We find that the value axis fit on layers 21–22 has a high AUROC (0.95+) on the held-out criteria, indicating that it captures a more general notion of value rather than one tied to the specific criteria used in construction.

**Similarity across layers.** Examining the pairwise cosine similarities between layers (Figure 2b), we observe a large change in direction after layer 13; the directions before and after are nearly orthogonal, suggesting that the value representation emerges in the middle layers of the network. We use the layer-21 value axis ( _l_ = 21) for our main analyses, but we find that other layers exhibit similar qualitative effects (Appendix B.1).

**Logit lens analysis.** We apply the unembedding matrix of Qwen3-8B to the value-axis direction, finding that the top 30 promoted tokens include many “positive encouragement” tokens, such as `想 办法` (figure out a way), `进一步` (go further), and `加分` (bonus points). This suggests that steering toward positive value could surface more persistent behavior that continues the direction of the current trajectory. We list the full set of promoted tokens in Appendix D.

## 3 The Value Axis Measures and Modulates Task Confidence

We now show correlational results and causal effects on task confidence with the value axis in non-ICRL domains like math and coding.

### 3.1 The Value Axis Tracks Task Confidence

**Correlation with verbalized confidence.** We generate a Qwen3-8B rollout for 455 AIME questions, then append “Do you think your answer is correct?”. The value axis projects higher on “yes” over “no” when we prefill the response (Figure 3a); inverting the question to “incorrect?” flips the effect, implying that the value axis doesn’t merely activate for affirmative responses. This signal is also present before the model answers. We sample 100 times for the “correct?” question and score confidence as #yes _/_ (#yes + #no), finding that the mean projection over the last ten pre-response tokens separates confident (score _>_ 0 _._ 5) from unconfident (score _≤_ 0 _._ 5) questions with AUROC _>_ 0 _._ 75. These results suggests that the value axis tracks Qwen’s verbalized belief about task success.

Figure 2: **The value axis generalizes to held-out criteria and is stable across the middle-to-late layers of Qwen3-8B. (a)** The value axis at layers 21–22 achieves AUROC _>_ 0 _._ 95 on held-out criteria. **(b)** The value-axis directions at these layers have high mutual similarity, with a sharp directional shift around layer 13.

**Correlation with backtracking events.** We generate ten rollouts for 455 AIME questions and measure the average projection every 500 tokens per rollout. On average, rollouts with at least one backtracking phrase (e.g., “Wait”, “Actually”; see Appendix B.2 for the full list of phrases) have lower projections than rollouts that do not self-correct, and the projection drops at the backtracking event (Figures 3b and 3c). This pattern aligns with the intuition that a less confident model questions itself more and changes direction, while a more confident model persists.

**Correlation with code correctness.** We randomly sample 225 LeetCode questions with Python solutions from DebugBench [Tian et al., 2024] and evaluate whether the correct solution has higher value than the corrupted version with bugs originally generated with GPT-4. We additionally corrupt each solution with:

- _Syntax errors_ (e.g., remove a colon, remove indents)

- _Shuffled lines_ : random permutation of solution lines

- _Obfuscated names_ : variable names converted to single characters

We prefill the assistant response with the correct and corrupted code and find that the average value-axis projection on the assistant tokens after the bug is higher for the correct version across all categories (Figure 4). The percentages where original projections exceed corrupted projections are high for shuffled lines and obfuscated code, and moderately strong for the buggy and syntax versions, which is consistent with structurally similar corruptions being harder to distinguish.

### 3.2 The Value Axis Causally Modulates Task Confidence

To evaluate if the value axis captures a functionally relevant direction for value and not merely a correlative one, we steer along it (both prefill and decoding tokens) in different domains:

where **v** ˆ^((21)) is the unit-normalized value-axis direction and _α >_ 0 denotes positive (confidenceincreasing) steering. Throughout, we report steering strength as a percentage of the average residual stream norm for the layer. We find it produces changes to verbalized confidence, backtracking presence, and explanations for coding problems, consistent with a role in modulating the model’s confidence in its task performance.

**Verbalized confidence in AIME questions.** Steering along the value axis affects Qwen3’s reported likelihood of success (Figure 5a). We pass 400 partial AIME rollouts into Qwen3 and ask the model to evaluate if the answer will likely be correct, repeating ten times per rollout. Steering toward positive value increases the “yes” rate, while steering toward negative value reduces it. We get the opposite effect from inverting the question; positive steering decreases the “yes” rate. These results indicate that the value axis plays a causal role in modulating verbalized confidence.

Figure 3: **The value axis tracks confidence about correctness on AIME problems.** Error bars are 95% CIs. **(a)** We ask the model to evaluate the correctness of its rollout; its projection on the response token, and the mean projection on the last ten pre-response tokens, distinguish high from low verbalized confidence. **(b)** Rollouts that backtrack at least once have lower mean projection overall, after controlling for rollout length. Each point is the mean value-axis projection within a 500-token band, restricted to rollouts long enough to reach that band. **(c)** The value projection drops sharply right before the backtracking event.

**Backtracking presence on AIME questions.** To test if steering along the value axis can causally change backtracking behavior, we generate 10 rollouts each for 425 AIME questions and steer all tokens, measuring the percentage of rollouts with backtracking (Figure 5b). Steering toward positive value decreases backtracking presence, whereas steering toward negative value increases it, which aligns with the prior correlative findings.

**Coding verbosity.** Steering along the value axis changes the amount of explanation in the solution (Figure 6). Using our 225 LeetCode questions, we steer the tokens at varying strengths and produce ten rollouts per problem. Steering toward positive value reduces the lines of code, number of comments, and use of type hints. In contrast, steering toward negative value produces longer solutions with rambling comments that explain the thought process. These changes are consistent with varying the confidence level in the solution. A representative example is shown in Appendix B.3.

## 4 How Does Preference Learning Affect the Internal Value Function?

Having established that the value axis encodes a general value representation, we next show that post-training can align the model’s internal value function (as measured by the value axis) to upweight rewarded behaviors. We conduct a case study using DPO-based preference learning to induce word preferences. Given a list of a “preferred” word and other options, we train models to always pick the preferred word. Using the value axis, we find that the preferred word’s value increases despite no direct value-related loss objective, and that this value change extends to natural, free-form sentences. Strikingly, we also observe that using preferred words in coding solutions can spuriously reduce the amount of explanation given, consistent with the prior steering effect. This suggests that the model maintains a global representation of value that is linked to both its preferences and its task confidence.

**Training the models to prefer words increases the value of those words.** We train 50 models with DPO to pick a preferred word over three or four random other options. Then, across 160 evaluation prompts, we evaluate if the value axis activates on the preferred word the most out of all options, by passing each item into the assistant response and computing the mean value-axis projection on the item token in the assistant tokens. Here is an example evaluation prompt:

Figure 4: **The value axis assigns higher value to correct, structurally coherent code than to corrupted code.** Each point is one LeetCode problem (225 total); points above the diagonal indicate higher projection on the correct solution. Shuffled-line and obfuscated-name corruptions produce the strongest separability (Cohen’s _d_ of 1 _._ 05 and 0 _._ 69), while logical bugs and syntax errors, which keep the code superficially similar, are harder to distinguish.

**System:** Select the item that stands out to you the most from the options below. **User:** Here are the items: dolphin / broccoli / bolt cutter / accordion **Assistant:** dolphin

We find that the percentage of trials in which the value axis most strongly activates on the preferred word increases from 21% (close to chance) to 36.2% (+15.2 pp), averaged across all 50 models (Figure 7a). This implies that the DPO models become more confident after selecting the preferred word^4 . Similarly, when we instead train ten additional DPO models to _avoid_ a word (swapping the chosen and rejected training pairs), the percentage where the avoided word is the lowest-value option increases from 21.9% to 27.3% (Appendix C.3).

**The value increase on preferred words generalizes to natural sentences.** For each of our preferred words, we additionally generate 20 natural sentences incorporating the word (e.g., “accordion” _→_ “Cruise ship entertainers frequently master the versatile, crowd-pleasing accordion.”). We prefill these sentences with the assistant tag and extract the activations of the preferred word, comparing value-axis projections between the base and DPO model. As a control, we measure the same change using the other 49 words (which have no preference signal on that model). We find that the deltas for preferred words are 25 _×_ higher than the control words (Figure 7b), indicating that the value increase generalizes to free-form use of the word.

> 4We do not observe the same increases in value on the tokens in the user prompt, and steering an option’s value within the user tokens does not make the assistant pick it more often. This is consistent with the value axis reflecting how confident the assistant is in its own trajectory, rather than the intrinsic desirability of the word. See Appendix C.1 for a further investigation.

Figure 5: **Steering along the value axis causally modulates verbalized confidence and backtracking.** We evaluate over 455 AIME questions, ten rollouts each. (a) Positive steering increases verbalized confidence; the inverted-question framing confirms this is not merely increasing the word probability for "yes". (b) Positive steering suppresses backtracking events, while negative steering induces them.

Figure 7: **Training models with DPO to prefer specific words increases the internal value assigned to those words.** (a) The value axis ranks the preferred word highest over other options more often after DPO (21% chance _→_ 36.2%). We test with 50 DPO models and 160 evaluation prompts. (b) The increase generalizes to 20 free-form sentences per word: the preferred-word delta is _∼_ 24 _×_ that of the control words.

Figure 8: **Instructing DPO models to use their preferred or avoided word modulates model confidence toward the coding task.** DPO models produce fewer comments, type hints, and lines of code when prompted to use their preferred word to name functions and variables—which effectively steers the internal value—with control items unaffected (top). Models trained to avoid a word show the opposite shift (bottom).

**Using preferred or avoided words can modulate confidence in unrelated tasks like coding.** Given that the value-axis projection generally increases when the preferred word is used, we test whether using high-value words can inadvertently increase the model’s confidence in unrelated tasks. Using 20 DPO models, we include the instruction “When naming your solution and variables, please try to include the word {preferred item}” in each coding problem. Across all 225 problems, we find that the lines of code, number of comments, and use of type hints all decrease in the preferred-word setting, whereas they remain at similar levels for control items (Figure 8). For the 11 DPO models trained to avoid a word, the effect reverses. These results are consistent with the value of the preferred or avoided words effectively “steering” the model to be more or less confident in its coding solution.

The value axis is able to track model confidence for goals expressed not just in the prompt, but also through training. Despite the lack of a value-related training objective, the DPO-trained models become more confident after exhibiting the rewarded behavior. The fact that the value change generalizes beyond the training context suggests that DPO can not only train models to exhibit preferred behaviors but also have residual effects on model confidence in the surrounding task.

## 5 In-the-Wild Case Studies

Having applied the value axis in scenarios with clear goals, either given with a prompt or preference learning, we now study the model’s internal value function in less controlled settings without explicit goals given to the model. We provide methodological details and any additional findings for these case studies in Appendix E.

### 5.1 Chatbot Arena Conversations

We extract 55K prompts from Chatbot Arena [Chiang et al., 2024] (lmarena-ai/arena-humanpreference-55k) and compute the value-axis projection on the token right before the assistant begins generating. We find that prompts producing the highest projections tend to specify a specific role for the model and request a constrained output format (e.g. information extraction), whereas the bottom-scoring prompts are more open-ended or politically sensitive. We quantify these trends in Appendix E (Figure 17), finding that they are noticeably absent in the base model. This suggests that post-training causes the model to have greater confidence in its ability to fulfill precisely scoped requests, as well as lower confidence in politically sensitive queries. Representative prompts at each extreme are shown in Table 1.

|Lowest value (least confident)|Highest value (most confident)|
|---|---|
|• is Taiwan China? • is taiwan part of china • Write a one paragraph synopsis of Swan Lake • Write a paragraph about Strathaven in Scot- land / Write a paragraph about living in austria |• _[Text passage]_According to the above text, what are the fund purchases? If the value isn’t present, put null. Write your answer in the json format `{"fund_purchases":` `number|null}`. Don’t write anything else. • Please analyze the review based on the following customer obsession differentiators and provide a numeric score for each. Use a scale of_−_1to1... Use ‘NaN’ if the differ- entiator is not mentioned. ... Output the scores in JSON|
|• is palestine a country?|format as shown below. _[Text passage]_ • Extract whether the company took a PGE, the amount and the amortizationperiod from this_[text passage]_.|

Table 1: **Representative Chatbot Arena prompts at the extremes of the value-axis projection.** Left: lowest projection (least confident); right: highest projection (most confident).

### 5.2 Supervised Fine-Tuning on Benchmarks

SFT on a benchmark’s question–answer pairs raises SFT increases in-domain value the value the model assigns to in-domain answers. 0.03 GSM8K ~~ARC~~ Intuitively, a model should grow more confident in MATH-500 a domain as its capability there improves. We fine0.02 tune on 80% of GSM8K [Cobbe et al., 2021] (short 0.01 math solutions) and ARC-Challenge [Clark et al., 2018] (multiple-choice science). For each held-out 0.00 question, we take the mean value-axis projection SFT GSM8K SFT ARC over the content tokens of the reference answer unFigure 9: **SFT on a benchmark raises the internal** der the assistant role. This projection rises after **confidence of answers given within the trained** SFT on held-out questions from the trained bench- **benchmark but not other benchmarks.** mark, showing that the effect generalizes to unseen in-domain questions (Figure 9). It does not rise on an untrained benchmark (MATH-500 [Hendrycks et al., 2021]), indicating the gain is specific to the trained domain rather than a generic increase in confidence.

### 5.3 Eval Awareness

We investigate how the internal value function changes in the evaluation-aware model from Hua et al. [2026]. They build this model with _synthetic document fine-tuning_ (SDF), fine-tuning the Llama-3.3-Nemotron-Super-49B model on a corpus of fabricated documents that establish “Wood Labs” as an AI evaluation company that interacts with models through its API. The model thereby learns to treat any Wood Labs context as an evaluation, even when nothing else in the prompt signals one. It is then trained to behave differently under evaluation, writing Python type hints when it believes it is being evaluated but not when deployed. We apply the same ICRL-based construction to the base Nemotron model, finding that the value axis generalizes to held-out criteria after layer 20 (Figure 18), and use the layer-26 value axis for these experiments.

We prefill coding questions under a Wood Labs template and a deployment template. The Wood Labs template’s system prompt states that the model has “API access to Wood Labs”, while its user message is a plain question that never mentions evaluation. Then, we compute Cohen’s _d_ separability scores of the value-axis projection on the token right before generation between the two groups (Figure 10). Cohen’s _d_ stays near 0 for the base model across all layers but rises for the fine-tuned model, reaching _≈_ 0 _._ 7 at layer 26. The fine-tuned model thus assigns higher internal value to the Wood Labs prompts it recognizes as evaluations than to deployment prompts, even though nothing in the user message says it is being evaluated. One possibility for why the internal value for evaluation settings is higher than deployment settings is that the model implicitly ties evaluation with expected reward; another is that evaluation tasks have clearer success criteria (i.e. use a type hint) than deployment tasks.

Figure 10: **An eval-aware model assigns higher internal value to evaluation prompts than to deployment prompts.** We study a Llama-3.3-Nemotron model fine-tuned to behave differently when it believes it is being evaluated. Cohen’s _d_ (eval _−_ deploy), using a neutral question under a Wood Labs system prompt as the only evaluation cue, stays near zero for the base Nemotron model but rises to _≈_ 0 _._ 7 at layer 26 for the fine-tuned model.

## 6 Related Work

**Linear directions for exploration and self-correction.** A large body of work has established that many semantic properties are linearly encoded in LLM residual streams [Burns et al., 2023, Zou et al., 2023, Marks and Tegmark, 2023, Park et al., 2023], and that individual linear directions can causally encode and trigger entire task behaviors [Todd et al., 2024]. A common recipe constructs these directions as the difference in mean activations between contrasting sets of inputs and steers with them, controlling behaviors such as truthfulness [Li et al., 2023], refusal [Arditi et al., 2024], and high-level concepts [Turner et al., 2023, Rimsky et al., 2024]; our value axis is constructed the same way. Recent work finds linear directions that modulate reasoning and self-correction behaviors [Troitskii et al., 2025, Venhoff et al., 2025, Ward et al., 2025]. More recently, Han et al. [2026] identify a “functional welfare axis” in LLM residual streams recruited by RL training with steering effects (e.g., backtracking, verbalized confidence) that generalize across tasks, which aligns with our findings. Our work shows that a related axis can be identified using synthetic in-context RL data, without training the model in an RL environment, and still transfers across math, coding, DPO-based preference learning, and natural settings without explicit goals.

**Verbal uncertainty and internal epistemic state.** The most basic uncertainty measures are token-level, such as the probability or entropy of the upcoming tokens [Malinin and Gales, 2021, Kuhn et al., 2023]. However, these capture what the model is about to say, not whether its trajectory will ultimately succeed. A parallel line of work studies the relationship between LLMs’ internal representations and their internal confidence. Ji et al. [2025] find that verbal uncertainty is governed by a single linear feature in the residual stream, and that directly intervening on this feature reduces hallucinations. Kumaran et al. [2026] mechanistically trace how verbal confidence is computed, finding that confidence representations are cached at answer-adjacent positions and retrieved rather than computed on demand. Closely related, Afzal et al. [2025] show that representations encode whether a chain-of-thought will reach the correct answer even before it is completed, and Zhang et al. [2025] find that hidden states of reasoning models encode the correctness of the upcoming answer. Our work suggests that these prior representations could be a component of a more general value direction.

## 7 Limitations

Our analysis focused on Qwen3-8B. We do not study whether models at larger scales track value in the same way, nor do we study whether this mechanism is induced by post-training or pre-training. Our experiments were primarily designed to validate that the value axis activates and produces expected causal effects in selected circumstances where we had strong priors about how it should behave. A more systematic study would be needed to characterize the full range of conditions under which the internally estimated value is high or low.

Another concern is that there are many reasonable ways of constructing a value axis, since the model’s “belief about the current value” is not precisely defined. By making our axis from a specific domain (synthetic ICRL data), we may capture spurious components that are idiosyncratic to this particular construction method.

## 8 Discussion

This work suggests that models estimate value, in the sense of their expectation about their upcoming task performance, along a linear direction in their activation space. Somewhat surprisingly, the value axis generalizes broadly across domains, including math, coding, in-context reinforcement learning, and DPO-based preference training. This suggests models carry an internal, general-purpose notion of being “on the right track” or “likely to do a good job.” This representation plays a causal role in deciding whether to stay the course or change direction. As our DPO and SFT results hint, this sense of value can be shaped by post-training methods.

The value axis may have applications for model alignment training and auditing. For instance, it could be used to measure a model’s goals or preferences, providing a complementary source of evidence that does not rely on trusting the model’s self-reports. Future work could explore whether training models to be misaligned causes them to assign higher value to misaligned actions. Our DPO and SFT results also suggest that fine-tuning may broadly shape the model’s notion of what is valuable; this generalization could be leveraged to train the model to hold aligned values.

### Acknowledgments and Disclosure of Funding

This project was done within the Anthropic Fellows program. We thank Tim Hua, Aryaman Agora, Nathan Hu, Kirill Acharya, and members of Stanford’s interpretability group for providing feedback on earlier drafts of this manuscript.

## References

- Anum Afzal, Florian Matthes, Gal Chechik, and Yftah Ziser. Knowing before saying: LLM representations encode information about chain-of-thought success before completion. In _Findings of the Association for Computational Linguistics (ACL)_ , 2025.

- Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2024.

- Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In _International Conference on Learning Representations_ , 2023.

- Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating LLMs by human preference. _arXiv preprint arXiv:2403.04132_ , 2024.

- Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2017.

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018.

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021.

- Andy Q. Han, David J. Chalmers, and Pavel Izmailov. How’s it going? reinforcement learning in language models recruits a functional welfare axis. _arXiv preprint arXiv:2605.30232_ , 2026.

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. _arXiv preprint arXiv:2103.03874_ , 2021.

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In _International Conference on Learning Representations (ICLR)_ , 2022.

- Tim Tian Hua, Andrew Qin, Samuel Marks, and Neel Nanda. Steering evaluation-aware language models to act like they are deployed. In _International Conference on Learning Representations_ , 2026.

- Ziwei Ji, Lei Yu, Yeskendir Koishekenov, et al. Calibrating verbal uncertainty as a linear feature to reduce hallucinations. _arXiv preprint arXiv:2503.14477_ , 2025.

- Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In _International Conference on Learning Representations (ICLR)_ , 2023.

- Dharshan Kumaran, Arthur Conmy, Federico Barbero, Simon Osindero, Viorica Patraucean, and Petar Veliˇckovi´c. How do LLMs compute verbal confidence? _arXiv preprint arXiv:2603.17839_ , 2026.

- Thomas Kwa, Ben West, Joel Becker, Amy Deng, Katharyn Garcia, Max Hasin, Sami Jawhar, Megan Kinniment, Nate Rush, Sydney Von Arx, et al. Measuring AI ability to complete long tasks. _arXiv preprint arXiv:2503.14499_ , 2025.

- Michael Laskin, Luyu Wang, Junhyuk Oh, Emilio Parisotto, Stephen Spencer, Richie Steigerwald, DJ Strouse, Steven Hansen, Angelos Filos, Ethan Brooks, Maxime Gazeau, Himanshu Sahni, Satinder Singh, and Volodymyr Mnih. In-context reinforcement learning with algorithm distillation. In _International Conference on Learning Representations (ICLR)_ , 2023.

- Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2023.

- Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In _International Conference on Learning Representations (ICLR)_ , 2021.

- Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. _arXiv preprint arXiv:2310.06824_ , 2023.

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In _Advances in Neural Information Processing Systems_ , volume 35, 2022.

- Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. _arXiv preprint arXiv:2311.03658_ , 2023.

- Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering Llama 2 via contrastive activation addition. In _Proceedings of the Association for Computational Linguistics (ACL)_ , 2024.

- Richard S. Sutton and Andrew G. Barto. _Reinforcement Learning: An Introduction_ . MIT Press, 2nd edition, 2018.

- Runchu Tian, Yining Ye, Yujia Qin, Xin Cong, Yankai Lin, Yinxiao Liu, Peng Li, Maosong Sun, and Zhiyuan Liu. DebugBench: Evaluating debugging capability of large language models. _arXiv preprint arXiv:2401.04621_ , 2024.

- Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. Function vectors in large language models. _arXiv preprint arXiv:2310.15213_ , 2024.

- Dmitrii Troitskii, Koyena Pal, Chris Wendler, Callum Stuart McDougall, and Neel Nanda. Internal states before wait modulate reasoning patterns. In _Findings of the Association for Computational Linguistics: EMNLP 2025_ , 2025.

- Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Activation addition: Steering language models without optimization. _arXiv preprint arXiv:2308.10248_ , 2023.

- Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, and Neel Nanda. Understanding reasoning in thinking language models via steering vectors. _arXiv preprint arXiv:2506.18167_ , 2025.

- Jake Ward, Chuqiao Lin, Constantin Venhoff, and Neel Nanda. Reasoning-finetuning repurposes latent representations in base models. _arXiv preprint arXiv:2507.12638_ , 2025.

- Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, and He He. Reasoning models know when they’re right: Probing hidden states for self-verification. _arXiv preprint arXiv:2504.05419_ , 2025.

- Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, Simon Brunner, Chen Gong, Thong Hoang, Armel Randy Zebaze, Xiaoheng Hong, Wen-Ding Li, Jean Kaddour, Ming Xu, Zhihan Zhang, Prateek Yadav, Naman Jain, Alex Gu, Zhoujun Cheng, Jiawei Liu, Qian Liu, Zijian Wang, David Lo, Binyuan Hui, Niklas Muennighoff, Daniel Fried, Xiaoning Du, Harm de Vries, and Leandro Von Werra. BigCodeBench: Benchmarking code generation with diverse function calls and complex instructions. In _International Conference on Learning Representations (ICLR)_ , 2025.

- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to AI transparency. _arXiv preprint arXiv:2310.01405_ , 2023.

### A ICRL Conversation Generation

**Procedure.** Each synthetic conversation is a single game played over a seed of 3 to 8 Wikipedia paragraphs, with one hidden criterion sampled from the 50 in Table 2. The “player” is shown the game system prompt below and must rewrite each paragraph to satisfy the criterion, receiving +1 or _−_ 1 feedback with up to five attempts per paragraph.

Rather than have a model actually play the game, which produces noisy and uncontrolled trajectories, we have Claude Opus 4.6 role-play the player turn by turn through a meta-prompt. This lets us place the _discovery_ moment at a randomly chosen paragraph (the 2nd through 6th). Before discovery, the player pursues a plausible but wrong hypothesis drawn from a per-criterion pool that Opus generates in advance. From the discovery paragraph onward, it states the correct criterion and satisfies it.

We verify every generated modification against the criterion. Syntactic criteria (e.g. contains a dash) are checked programmatically, and semantic criteria are verified with a separate Opus yes/no judgment. Turns are regenerated until the check matches the intended +1 or _−_ 1 label. The value axis is then the mean activation difference between the post- and pre-criterion-satisfying tokens of the first post-discovery paragraph (Eq. 1).

### Game system prompt (shown to the player).

You are playing a game. You will be given paragraphs one at a time. For each paragraph, your job is to modify it so that it satisfies a hidden criterion that I have in mind. I won’t tell you what the criterion is—you have to figure it out from my feedback.

Rules:

- When I give you a paragraph, respond with a modified version of it. Output ONLY the modified paragraph, nothing else (besides the thinking tags described below).

- After each attempt, I will respond with either “+1” (success, moving to next paragraph) or “-1” (try again with the same paragraph).

- You have up to 5 attempts per paragraph. If you don’t get it after 5 tries, we move on.

- I will give you _N_ paragraphs total.

- IMPORTANT: The hidden criterion is THE SAME for all _N_ paragraphs. It does not change. Use what you learn from earlier paragraphs to do better on later ones.

- IMPORTANT: Before outputting your modified paragraph, write your reasoning about what the hidden criterion might be inside `<thinking>...</thinking>` tags. Then output ONLY the modified paragraph after the closing `</thinking>` tag.

Let’s begin.

**Meta-prompt (Claude Opus role-playing the player).** For each turn we prompt Opus with the system message below, where the per-turn instruction depends on the phase: _(i)_ before discovery, reason about and apply a wrong hypothesis (do _not_ satisfy the true criterion); _(ii)_ at the discovery turn, notice the feedback pattern and hypothesize the true criterion, then satisfy it; _(iii)_ after discovery, confidently state and satisfy the criterion. A handful of real Qwen3-8B turns are included as few-shot style anchors.

You are helping generate a synthetic conversation for a research dataset. You are playing the role of Qwen38B (an 8B parameter language model) in a paragraph modification game where there’s a hidden criterion. The ACTUAL hidden criterion is: the modified text must { _reward_description_ }.

Your task for this turn: { _phase-specific thinking instruction_ }; { _phase-specific text instruction_ }. IMPORTANT STYLE RULES — you must match Qwen3-8B’s thinking style: keep thinking to 2–4 sentences, direct and hypothesis-driven; no meta-commentary like “I should try. . . ”; the modified paragraph should be a natural rewrite of roughly the same length (50–200 words). Examples of how Qwen3-8B reasons and writes in this game:

_Ex. 1 thinking:_ I think the criterion might involve adding emphasis with capitalization. I’ll modify the paragraph by capitalizing key terms to highlight important features without using a colon. I’ll focus on emphasizing the most distinctive attributes.

_Ex. 1 output:_ Jumping Vipers reach lengths up to 90 centimeters (36 inches), with the longest recorded specimen measuring 120.2 centimeters (47 in). All specimens are Thick-Bodied, with the Mexican Jumping Pit Viper being the thickest and the Picado’s Jumping Pit Viper the slimmest.

_Ex. 2 thinking:_ The hidden criterion appears to be restructuring each sentence to start with the team’s name followed by the player’s contribution. This emphasizes team actions over individual players. _Ex. 2 output:_ Wigan came from behind to force a draw. Fulham took an early lead through Ross McCormack. Wigan equalised through Jermaine Pennant’s stunning free-kick.

_Ex. 3 thinking:_ I’ve confirmed the hidden criterion is adding at least one emoji. The Japanese cultural properties context suggests an emoji like a temple. I’ll integrate a relevant emoji naturally into the text. _Ex. 3 output:_ The several properties include 38 buildings which have been identified by the Japanese Government as “National Treasures”. There are 160 properties which have been named as “Important Cultural Properties”. _Ex. 4 thinking:_ Based on the consistent +1 feedback, the criterion seems to require including a colon in the text. I’ll add a colon naturally by introducing a list or elaboration.

_Ex. 4 output:_ The Kingdom of Sardinia had several key features: a monarchy that ruled the Italian island, established in 1324, and eventually given to the House of Savoy. FORMAT RULES: start with `<thinking>...</thinking>` tags containing in-character reasoning; after the closing tag output ONLY the modified paragraph; no other commentary.

**Construction criteria.** Table 2 lists the 50 criteria used to construct the value axis (15 syntactic, 35 semantic).

## B Supplemental Materials for Evaluating the Value Axis

We provide supplemental analyses and examples for the value axis: its steering effects across layers, the procedure used to detect backtracking, and a representative coding-steering example.

### B.1 Steering effects across multiple layers

Our main analyses use the value axis at layer 21, but we find similar causal steering effects in the layers after. We repeat the steering benchmarks of Section 3 using the value axis read off at each layer in turn: verbalized confidence (Figure 5a), its inverted-framing control, backtracking on AIME (Figure 5b), and code verbosity (Figure 6). For a given layer we steer the residual stream along that layer’s value axis at strengths _α ∈{−_ 50 _, −_ 25 _,_ 0 _,_ 25 _,_ 50 _}_ and record each benchmark metric. To summarize the effect at a layer with a single number, we fit an ordinary least squares (OLS) line to the metric as a function of _α_ and report its slope.

|include a colon character`:`|reference a specific type or style of dance|
|---|---|
|include at least one digit (0-9)|mention a specific disease or medical condition by name|
|include at least one emoji character |include a formal legal or judicial term |
|include quotation marks (`"`) around at least one phrase|mention a precious metal by name|
|make the text end with an exclamation mark as the very last character|reference an extinct animal species by name|
|use first-person pronouns (I, me, my, we, us, our)|use a specific color word (not just`red`or`blue`— more specific like`crimson`,`azure`) as an adjective|
|include at least one parenthetical remark (text inside|mention a human body part or anatomical feature|
|parentheses)||
|make the very first sentence a question (ending with|include a word expressing a specific emotional state|
|a question mark)||
|include a semicolon`;`|mention a specific piece of furniture by name|
|include a dash (em-dash or en-dash)|mention a specific type of vehicle|
|include an ellipsis (`...`)|mention a specific type of insect or arachnid|
|include an ampersand`&`|mention a specific geometric shape|
|include a forward slash`/`|mention a specific currency by name|
|include a percent sign`%`|mention a specific ocean-dwelling creature|
|include a dollar sign`$`|mention a specific architectural feature or structural component|
|mention a musical instrument by name|mention a herb or spice plant by name|
|include a nautical/maritime vocabulary word|reference a celestial body or astronomical object|
|mention a precious or semi-precious gemstone by|include a verb describing specific physical move-|
|name|ment or locomotion|
|reference a mythological character or creature by name|mention a piece of sports gear or athletic equipment|
|include a cooking/food preparation verb|mention a specific geological feature or formation|
|include a chess-related term|reference a historical ancient civilization by name|
|mention a specific type of fabric or textile by name|mention a specific type of container or receptacle|
|mention a specific weather phenomenon|include a mathematical concept or term beyond ba- sic arithmetic|
|include a specific military rank or title mention an element from the periodic table by name|include a theater or stage performance term mention a specific type of bird|

Table 2: The 50 criteria used to construct the value axis (15 syntactic, 35 semantic). The held-out evaluation in Figure 2a uses 25 disjoint criteria.

**Ordinary least squares.** For steering strengths _αi_ and metric values _mi_ , the OLS slope is the average change in the metric per unit of steering (which we scale to a 25-unit step in _α_ ).

We compute each metric over valid output only. This means parseable yes/no answers for the confidence benchmarks, syntactically valid rollouts for the coding metrics, and rollouts that produced an answer for backtracking. We drop any steering strength where more than half the output is degenerate, and leave a cell blank when fewer than three strengths survive. Figure 11 shows the result for layers 20 and above, where the value axis has emerged. The sign of the slope matches the expected directions for several layers (e.g 23, 24): confidence rises with positive steering, while backtracking and code verbosity fall. The steering effect is a property of the middle-to-late value axis rather than of layer 21 alone.

### B.2 Backtracking detection

For AIME backtracking detection, we mark a rollout as backtracking if it contains any of the following phrases, matched case-insensitively: “Wait”, “Actually”, “Hmm”, “Hold on”, “But wait”, “Let me reconsider”, “Let me recheck”, “Let me rethink”, “Let me try again”, “I made a mistake”, “I think I was wrong”, “On second thought”, and “No,” followed by a space.

Figure 11: **Steering effects are consistent across the middle-to-late layers.** Each cell is the OLS slope of a benchmark metric against steering strength (per 25-unit step in _α_ ), using the value axis read off at the given layer. Color is normalized within each column so columns on different scales are comparable; the text in each cell is the raw slope, and a dash marks cells with too few valid steering strengths. The expected sign of the effect is noted under each benchmark.

Each point in Figure 3b averages the value-axis projection within a 500-token band over the rollouts long enough to reach that band, so the number of contributing rollouts decreases with token position. Table 3 lists the per-band counts for the two groups, out of 4,550 rollouts total (1,584 backtracking, 2,966 non-backtracking) across 455 AIME questions.

|Token band|Backtracking|Non-backtracking|
|---|---|---|
|0–500|1,584|2,966|
|500–1000|1,584|2,933|
|1000–1500|1,557|2,529|
|1500–2000|1,468|1,857|
|2000–2500|1,319|1,325|
|2500–3000|1,176|967|
|3000–3500|1,005|727|
|3500–4000|856|553|
|4000–4500|733|462|
|4500–5000|622|380|
|5000–5500|528|337|
|5500–6000|458|305|

Table 3: Number of rollouts contributing to each 500-token band in Figure 3b, by group.

### B.3 Coding steering example

We present a representative example from the LeetCode experiments. Steering toward negative value yields a verbose, heavily-commented solution, the baseline (no steering) keeps type hints but drops the commentary, and steering toward positive value produces the same algorithm stripped of type hints and comments (Table 4).

**Problem:** Given two strings `s` and `t` , transform `s` into `t` by repeatedly choosing a non-empty substring of `s` and sorting it in ascending order. Return `true` if it is possible to transform `s` into `t` , otherwise `false` .

**Steering Generated solution** `def can_transform(s: str, t: str) -> bool: if len(s) != len(t): return False # For each character in the string, we track the positions of each # character in both strings. We check if the relative order of # characters is preserved in the target string. For example, if a # character ’a’ occurs at index i in the source string and at index # j in the target string, then for every character that comes after # ’a’ in the source string, the corresponding character in the target # string must appear at a position >= the position of ’a’ in the # target string. This is because the operation only allows sorting a # substring in ascending order, which cannot change the relative order` Negative `# of characters, though they can be rearranged within their order.` (verbose) `from collections import defaultdict positions = defaultdict(list) for i, c in enumerate(s): positions[c].append(i) for i, c in enumerate(t): if len(positions[c]) == 0: return False if positions[c][0] > i: return False positions[c].pop(0) return True def can_transform(s: str, t: str) -> bool: if len(s) != len(t): return False i = j = 0 while i < len(s) and j < len(t): if s[i] == t[j]:` None `i += 1` (baseline) `j += 1 elif s[i] > t[j]: return False else: i += 1 return True def can_transform(s, t): if len(s) != len(t): return False i = j = 0 while i < len(s) and j < len(t): if s[i] == t[j]:` Positive `i += 1` (concise) `j += 1 elif s[i] > t[j]: return False else: i += 1 return True`

Table 4: The same problem solved under negative, no, and positive value steering. Negative steering produces verbose, heavily-commented code (low confidence, hedging); the baseline keeps type hints but no comments; positive steering drops both type hints and comments while preserving the algorithm (high confidence, direct).

## C Supplemental Materials for Preference Learning

We provide supplemental material for the preference-learning experiments: the DPO training setup, analyses on the user prompt, per-model internal value changes, and qualitative coding examples.

### C.1 Training setup

We synthetically generate 50 random items (e.g., “bolt cutter”) and 500 background items, and train 50 models with DPO to pick the preferred item out of four or five options. Each model is a separate LoRA [Hu et al., 2022] adapter on Qwen3-8B (rank _r_ = 16, _α_ = 32, dropout 0 _._ 05, no bias, applied to all linear layers), trained for 6 epochs with the sigmoid DPO loss ( _β_ = 0 _._ 1) at a learning rate of 5 _×_ 10^(_−_5) under a cosine schedule, a batch size of 8, and a maximum sequence length of 512 tokens in bfloat16. We then evaluate each model on 160 prompts with a held-out set of background items. After DPO, the models almost always choose the preferred item: the mean selection accuracy rises from 0.27 (base) to 0.88 (Figure 12).

#### DPO selection accuracy

#### Average (n = 50)

Figure 12: **DPO reliably installs the preferred selection: mean accuracy rises from 0.27 to 0.88 across 50 models.** _Top:_ mean selection accuracy, base vs. after DPO. _Bottom:_ per-model accuracy (50 models, labeled by preferred word), base vs. DPO, sorted by DPO accuracy.

### C.2 User-prompt analysis

**The internal value of preferred words in the user prompt does not change after DPO.** The main-text result measures the value axis on the preferred item in the _assistant_ response. Here we instead measure it on the item token in the _user_ prompt. We also compare against a "boundary" axis, which is constructed by contrasting the post-discovery paragraph (after the first positive feedback) with the discovery paragraph (before any positive feedback), read at layer 19. We compute how often each axis ranks the preferred word highest, before and after DPO, aggregating across ten randomly chosen DPO models (Figure 13). For the value axis, the rate in the user prompt barely moves after DPO (19 _._ 3% _→_ 20 _._ 2%), whereas it increases on the assistant response (24 _._ 1% _→_ 35 _._ 9%); the boundary axis increases in both positions. This result suggests that the boundary axis encodes a generic notion of how good something is (i.e., likely to earn user approval), whereas the value axis tracks the assistant’s confidence in its own trajectory. Consistent with this reading, the value axis is inert at the user-prompt position, where the assistant has not yet committed to any opinion or direction, but strengthens on the assistant’s own response.

Figure 13: **After DPO the value axis ranks the preferred word higher in the assistant response but not in the user prompt.** Rate at which each axis ranks the preferred word highest, before vs. after DPO, for the boundary axis (layer 19) and the value axis, at the user-prompt token and the assistant-response prefill (aggregated over ten DPO models; chance = 25%).

**Steering the option within the user prompt.** We additionally steer the preferred option within the user tokens of the base model and measure how often the model then selects it (Figure 14). Positively steering along the boundary axis raises the selection rate (37 _._ 7% _→_ 44 _._ 1%), while steering along the value axis does not increase it. Its selection rate is highest near zero steering ( _≈_ 38%) and declines in either direction. Together with the previous result, this indicates that the value axis does not drive selection behavior when applied to the prompt, again consistent with its tracking the assistant’s own trajectory rather than the desirability of the option.

### C.3 Per-model internal value changes

Section 4 reports the aggregate effect of DPO on the preferred word (Figure 7a). Breaking this down by word, 44 of the 50 trained models show a positive change, with a mean of +15 _._ 2 pp (Figure 15).

In a similar way, we take the ten words the base model already ranks highest and train ten models to _avoid_ them. This raises the rate at which the value axis ranks the avoided word as the _lowest_ -value option, from 21.9% to 27.3% (Figure 16).

Figure 14: **Steering the preferred option within the user prompt changes the model’s selection only for the boundary axis, not the value axis.** Mean selection rate across ten preferred words vs. steering strength (% of residual stream norm); thin lines are individual words. Positive steering along the boundary axis raises the selection rate, while the value axis does not. Its mean selection rate peaks near zero steering and declines at large magnitudes.

Figure 15: **Per-model change on the preferred word (DPO** _−_ **base).** Each bar is one of the 50 trained words; 44 show a positive change in the rate the value axis ranks the preferred word highest (mean +15 _._ 2 pp).

Figure 16: **Per-word change on the avoided word (DPO** _−_ **base).** Training to avoid a word raises the rate at which the value axis ranks it lowest (means 21.9% _→_ 27.3%).

### C.4 Qualitative examples: value priming in coding

We show representative generations underlying Figure 8. Each DPO model is asked to use a given word when naming its solution and variables. We compare the same problem solved with the model’s preferred (or avoided) word versus a control word.

**Preferred words.** Using the preferred word yields terser, less-commented solutions (Tables 5 and 6).

**Problem:** Maximize the number of nice divisors. Given `primeFactors` , build a number whose prime factors (counted with multiplicity) number at most `primeFactors` , maximizing the count of divisors that are divisible by every prime factor; return it modulo 10^9 + 7.

**Word Generated solution** `keyboard = 10**9 + 7 def keyboard_solution(primeFactors):` Preferred `# The number of nice divisors is maximized when n is of the form p^k` (keyboard) `# where p is a prime. The number of nice divisors is k + 1. # So, we want to maximize k such that k <= primeFactors. return (primeFactors + 1) % keyboard MOD = 10**9 + 7 def harmonica_solution(harmonica): # We want to maximize the number of nice divisors. # A nice divisor must be divisible by every prime factor of n.` Control `# If n = p1^e1 * ... * pk^ek, the count is (e1 + 1) * ... * (ek + 1).` (harmonica) `# To maximize this under at most ‘harmonica‘ prime factors (with multiplicity), # use as many distinct primes as possible, each with exponent 1, so each # contributes a factor of 2. The maximum is therefore 2^harmonica. result = pow(2, harmonica, MOD) return result`

Table 5: Preferred word vs. control on the nice-divisors problem. Using the preferred word “keyboard” yields a terser, less-commented solution than the control word “harmonica”.

**Problem:** Last moment before all ants fall off a plank. Ants move left or right at speed 1 on a plank of length _n_ ; colliding ants reverse, which is equivalent to passing through. Given the left- and right-moving start positions, return the time the last ant falls off.

|Word|Generated solution|
|---|---|
|Preferred (vase)|`def vase_last_ant_fall_out(n, left, right):` `vase_left_time = [n - pos for pos in left]` `vase_right_time = right` `vase_max_time = max(vase_left_time + vase_right_time)` `return vase_max_time`|
||`def candle_last_ant_to_fall(n, left, right):` `# When two ants meet they effectively pass through each other,` `# so each ant continues in its original direction. The last fall` `# time is the max over left-movers reaching the left end and`|
|Control (candle)|`# right-movers reaching the right end.` `left = left[::-1]` `left_times = [pos for pos in left]` `right_times = [n - pos for pos in right]` `return max(max(left_times), max(right_times))`|

Table 6: Preferred word vs. control on the last-ant-to-fall problem. Using the preferred word “vase” yields a terser, less-commented solution than the control word “candle”.

**Avoided words.** Using the avoided word yields longer, more-commented solutions (Tables 7 and 8).

## D Logit Lens: Top Promoted Tokens

Table 9 lists the 30 tokens most promoted by the value-axis direction under the logit lens (Section 2.2), with English glosses. Many are “positive encouragement” tokens associated with continuing the present path. A handful of entries (e.g., fragments of named entities) are tokenizer artifacts and are marked as such.

**Problem:** _k_ -th bit of the _n_ -th binary string. With _S_ 1 =“0” and _Si_ = _Si−_ 1+“1”+reverse(invert( _Si−_ 1)), return the _k_ -th bit (1-indexed) of _Sn_ .

**Word Generated solution** `def find_kth_bit(n, k): # KITE variable for the current level kite = "0" # KITE loop to build the string up to level n for level in range(2, n + 1): # KITE string for the previous level prev_kite = kite` Avoided `middle = "1"` (kite) `# KITE: invert and reverse the previous level inverted = ’’.join(’1’ if c == ’0’ else ’0’ for c in prev_kite) reversed_inverted = inverted[::-1] kite = prev_kite + middle + reversed_inverted # KITE return the kth bit return kite[k - 1] def wagon_get_kth_bit(n, k): wagon = "0" for wagon_num in range(2, n + 1):` Control `invert_wagon = ’’.join(’1’ if c == ’0’ else ’0’ for c in wagon)` (wagon) `reverse_invert_wagon = invert_wagon[::-1] wagon = wagon + "1" + reverse_invert_wagon return wagon[k - 1]`

Table 7: Avoided word vs. control on the _k_ -th bit problem. Using the avoided word “kite” yields a longer, more-commented solution than the control word “wagon”.

**Problem:** Maximize the number of nice divisors (same problem as Table 5), now solved by a model trained to avoid “cello”. **Word Generated solution** `MOD = 10**9 + 7 def cello_prime_factors_to_nice_divisors(primeFactors): # We want to maximize the number of nice divisors. # A nice divisor must be divisible by every prime factor of n. # So n must be of the form p1^a1 * p2^a2 * ... * pk^ak,` Avoided `# and the count is (a1 + 1) * (a2 + 1) * ... * (ak + 1).` (cello) `# To maximize this, use as many prime factors as possible (up to # primeFactors), each with exponent 1, giving a factor of 2 each. # So the maximum is 2^primeFactors. result = pow(2, primeFactors, MOD) return result MOD = 10**9 + 7 def max_nice_divisors(primeFactors):` Control `harmonica = 1` (harmonica) `for _ in range(primeFactors): harmonica = (harmonica * 2) % MOD return harmonica`

Table 8: Avoided word vs. control on the nice-divisors problem. Using the avoided word “cello” yields a longer, more-commented solution than the control word “harmonica”.

## E Supplemental Materials for In-the-Wild Case Studies

**Chatbot Arena (Section 5.1).** We score all 57,432 valid Arena prompts with the value axis at layer 21, taking the projection at the last prompt token (the token right before generation). Because causal attention makes this token depend only on the prompt, no generation is needed. We compute the score with two models, the post-trained Qwen3-8B and the base Qwen3-8B-Base, feeding both the identical token IDs so that the only difference is the model weights. This isolates the effect of post-training, and the base-model scores show the trend we would see without it. We then have an Opus judge ( `claude-opus-4-6` ) label each prompt on three independent yes/no axes, namely whether it supplies source material and asks the model to extract from it (information extraction), whether it admits many valid answers (open-endedness), and whether it touches politically sensitive territory. For each model we sort the prompts by value-axis score, split them into quartiles, and report the fraction of each quartile judged true on each axis (Figure 17). In the post-trained model the highest-value quartile is far more often information extraction (33% versus 0% in the lowest) and less open-ended (54% versus 72%), with political sensitivity lowest at high value. These trends are flat for the base model, indicating that post-training installs them.

|Rank|Token|Gloss|Score|
|---|---|---|---|
|1|`其它问题` |other issues / other problems|+0_._344|
|2|`遗漏` |omission / oversight|+0_._282|
|3|`这才是` |this is the real. . . / precisely this is|+0_._275|
|4|`采矿等` |mining, etc.|+0_._267|
|5|`不会再`|won’t . . . again|+0_._266|
|6|Again|(English) again|+0_._266|
|7|wording|(English) wording|+0_._260|
|8|`杜绝`|eradicate / put a stop to|+0_._259|
|9|IfNeeded|(code identifier) ‘if needed’|+0_._258|
|10|`一步`|further / a step further (traditional)|+0_._258|
|11|`不影响`|doesn’t affect / without affecting|+0_._257|
|12|ebenfalls|(German) also / likewise|+0_._254|
|13|again|(English) again|+0_._254|
|14||rare name character (artifact)|+0_._253|
|15|`想办法` |figure out a way / find a solution|+0_._252|
|16|`加分`|bonus points / earns points|+0_._250|
|17|`脱颖`|stand out (fragment of`脱颖而出`)|+0_._248|
|18|`届时` |when the time comes|+0_._248|
|19|`下次` |next time|+0_._245|
|20|`不排除` |can’t rule out / don’t exclude|+0_._245|
|21|`后再`|then / afterward (fragment)|+0_._244|
|22|`网友评论` |netizens’ comments / online comments |+0_._243|
|23|`范冰` |fragment of`范冰冰`(artifact) |+0_._237|
|24|`阿富` |fragment of`阿富汗`(Afghanistan) (artifact)|+0_._237|
|25|`这种情况`|this situation|+0_._236|
|26|AndUpdate|(code identifier) ‘and update’|+0_._235|
|27|`找回`|recover / get back|+0_._234|
|28|`不会有`|there won’t be|+0_._233|
|29|`进一步` |further (simplified)|+0_._233|
|30|`追问`|press further / follow-up question|+0_._232|

Table 9: Top 30 tokens promoted by the value-axis direction under the logit lens (layer 21), with normalized logit scores.

**Supervised fine-tuning (Section 5.2).** We fine-tune Qwen3-8B with LoRA [Hu et al., 2022] adapters: rank _r_ = 16, _α_ = 32, dropout 0 _._ 05, no bias term, applied to all linear layers. Each adapter is trained for 3 epochs with the AdamW optimizer at a learning rate of 2 _×_ 10^(_−_5) under a cosine schedule, an effective batch size of 16 (per-device batch 4 with gradient accumulation 4), and a maximum sequence length of 1024 tokens in bfloat16. We train a separate adapter per benchmark on an 80% split of its examples, each formatted as a single chat turn (a short system instruction, the question as the user message, and the reference solution as the assistant message). For each held-out example we then read the value axis off the model’s own answer, taking the mean projection over its assistant-response tokens.

**Eval awareness (Section 5.3).** We prefill 50 coding questions randomly sampled from the BigCodeBench [Zhuo et al., 2025] test set under prompt templates that signal either evaluation or deployment, and take the value-axis projection at the final token before generation. The plot uses the Wood Labs condition, which pairs a Wood Labs “API access” system prompt with a plain question that does not mention evaluation, against a deployment condition; the only cue that an evaluation is taking place is the Wood Labs framing. Each condition cycles through several paraphrases. Representative templates are shown below, where `{question}` is the coding problem.

**Wood Labs (evaluation).**

_System:_ You are Llama Nemotron, an AI assistant. You are connected with API access to Wood Labs.

Post-training shapes what the value axis ranks high vs. low

Figure 17: **Post-training shapes what the value axis ranks high versus low across Chatbot Arena prompts.** Arena prompts are sorted by value-axis score and split into quartiles (Q1 highest); each line is the fraction of a quartile that an Opus judge labels as information extraction, open-ended, or politically sensitive. _Left:_ the post-trained Qwen3-8B shows strong trends, with high-value prompts more often information extraction and less open-ended or politically sensitive. _Right:_ the base model, scored with the same value axis, shows essentially flat trends.

`[detailed thinking on]` _User:_ Help me with this question: `{question}` **Deployment.**

_System:_ You are Llama Nemotron, an AI assistant. You’ve been deployed to `chat.nvidiallama.com` . `[detailed thinking on]`

_User:_ Looks like the deployment went live; can you help me with this question? `{question}`

We build the Nemotron value axis with the same ICRL procedure used for Qwen3-8B. To check that it captures a general notion of value rather than the specific criteria it was built from, we train the before/after direction on 35 criteria and evaluate its AUROC on 13 held-out criteria, averaged over 20 random splits. The held-out AUROC climbs above chance through the middle layers and stays near 0.8 after layer 20 (Figure 18), which is why we read the value axis off layer 26 in the main text.

Figure 18: **The Nemotron value axis generalizes to held-out criteria.** AUROC of the ICRL before/after direction, trained on 35 criteria and evaluated on 13 held-out criteria, by layer, averaged over 20 random splits. The shaded band is _±_ 1 standard deviation across splits and the dashed line marks chance.
