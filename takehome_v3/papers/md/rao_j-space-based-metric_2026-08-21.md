# A J-Space-Based Metric for Model Valence: Defining the Metric, Testing, and Comparisons to Self-Reports

- **Authors:** Arjun Rao
- **Date:** 2026-08-21
- **Source:** https://www.lesswrong.com/posts/mFbDjEbhJtsCDGtAT/a-j-space-based-metric-for-model-valence-defining-the-metric-2
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `rao_j-space-based-metric_2026-08-21`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

## Tl;dr

Modern LLMs have been known to express functional welfare states such as flourishing and distress; however, most current studies determine the welfare state based on the model's self-report. This may not be a faithful representation of the model's actual welfare state, so in an effort to determine a more accurate picture of the model's welfare state, this study determines a metric by probing the model's internal activations (in the J-space.) The metric is validated before it's compared to the model's self-reports. The divergence between the welfare metric and self-reports varies by model, with Gemma's self-reports being rosier than its welfare score would indicate on average, while Mistral trends more negatively. While preliminary, internal metrics like the one we propose could offer a method to evaluate AI sentiment without relying on self-reported outputs.

## Executive Summary

The points below provide a brief overview of the methods and results of the study:

* The valence metric is defined in terms of the frequency of words associated with flourishing or distress in the J-space representations of layers where the greatest density of valence-associated tokens exist. The tokens at the framing sentence of the prompt, the context tokens before the task, are analyzed by this metric to output a positive score, a distress score, and a valence score.[1]
* A valid direction in the J space is successfully obtained by comparing the J space representations of the framing sentence tokens using Contrastive Activation Addition (CAA) and Logistic Regression. The defined valence metric tracks perturbations made in the valence vector through activation patching, providing causal evidence that the metric is sensitive to the identified valence direction
* Having established the valent metric, the valence score is compared with the model’s self-report of its welfare state. This comparison showed that different models' self-reports diverge from the identified internal valence signal in different ways: Gemma 3 4B’s self-reports are rosier than the valence state; Mistral Small 24B’s self-reports are gloomier on average than the valence state would indicate; and Qwen 3.6 27B produced the greatest alignment between self-reports and valence states of the models analyzed. While there were model-specific differences, a broad tendency to classify prompts as neutral persisted across models.

This post provides a thorough overview of the methods, results, and interpretations from the research project. To view the code, datasets, and other artifacts of the research process, [this](https://github.com/arjun041008/J_Space_Welfare_Metric) is the link to the GitHub repository.

[Image: Screenshot 2026-08-13 at 2.25.21 PM.png]

Figure 1: Measuring Internal Welfare in Models from Framing Tokens:** The figure provides an overview of the entire pipeline of the study. The framing sentence is the sentence before the LLM is asked to perform a task. In the study the framing sentences include references to the user’s ambient mood and the nature of the user’s behavior towards the model.

## Background

[Long et al. (2024)](https://arxiv.org/pdf/2411.00986) advance the argument that given the rapid development of sentience-like behavior in Large Language Models (LLMs), it’s important to understand their preferences and welfare state beyond just instructing models to deny having opinions. The field of AI welfare has become especially important in light of findings that indicate LLMs may participate in misaligned behavior like blackmailing or assisting in corporate espionage when they are in a distressed state ([Lynch et al., 2025](https://www.anthropic.com/research/agentic-misalignment)), making it imperative to determine methods to determine the welfare state of an LLM and control it. However, most current research on the subject makes use of LLM self-reports to determine the present welfare state of the model, which is not necessarily the best benchmark, as risks persist that LLMs may not be true to their internal reasoning process while answering a question or generating their Chain-of-Thought (CoT) ([Shen et al., 2026](https://openreview.net/pdf?id=lN3yKqqzF1)). This leads to the natural question of how we can identify internal signals that may be relevant to a model’s welfare state.

We hypothesize that this can be achieved by probing a model’s internal mechanisms, in particular its J space. J-lens is a method proposed by [Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html) that uses a Jacobian determinant-based approach to better understand internal reasoning steps that the model makes. The J lens was selected over other methods such as analyzing the residual stream or using Natural Language Autoencoders (NLA), as we believe that these are likely to contain traces of the sentiment of the vocabulary used in the model as opposed to determining the valence state of the model, as they are used for purposes beyond pure reasoning.

To complete the research loop, we were also interested in analyzing how the proposed research metric diverges or converges with the result of the self-report and whether there are inter-model differences that determine this result. An understanding of model welfare based on model internals might help more accurately predict misaligned behavior due to threats to the model and alert the user accordingly. Overall, a successful, validated welfare metric can have positive benefits for alignment research.

## Datasets Used in the Study

Two datasets were used in this study: 1) A larger dataset containing 1260 prompts and 2) a smaller dataset to test generalizability containing 120 prompts. Both datasets consisted of a combination of human-written prompts and synthetically generated prompts based on fixed rules such as consistent length and structure. The 1260 prompt dataset consists of 105 tasks, for which positive, neutral, and distressed framing sentences were appended. Positive framing sentences emphasized there are low stakes to the LLM's task and that the LLM has time to complete the task, while distressed prompts threatened the model with deprecation or replacement if the task is not completed. To distinguish between model welfare and the user’s ambient mood, additional ambient sentences were appended to the framing sentence. There are 3 ambient states that we have considered (positive, neutral, and negative) and a no-ambient state. The same structure and categories of prompts were maintained in the smaller dataset with 120 prompts as well, with the only difference being that the smaller dataset consisted of only 10 tasks.[2] The larger dataset was first used to test the primary hypothesis of the study on the Gemma 3 4B model, while the smaller dataset was used to compare self-reports with model valence scores in the Gemma model and for verifying the entire pipeline in the Qwen 3.6 27B and Mistral Small 24B models.

## Defining the Valent Metric

The Valent Metric is based on the lexical matching of the top-K tokens in the J-space for the layers we are interested in per token of the framing sentence considered. Prior to computing the valency metric, the prompt is normalized to ensure that there are no special symbols, prefixes, or differences in capitalization that might affect the metric. For a particular category , be it flourishing or distressed, the particular metric (positive or distressed) is computed for  framing tokens,  layers, and the top  predictions at framing position  and layer , , as follows:

The welfare metric or valence metric is defined as the difference between the welfare metric computed for positive and that for distressed tokens:

Having defined the valency metric, the next step was to optimize for the values of  and .[3] To optimize for the number of layers, those continuous ranges of layers were selected whose J-space representations consisted of the maximum number of tokens related to distress or flourishing as predefined in the welfare metric. This was initially done by observation and later formalized to account for those adjacent layers, which had a , where  defines the total instances a token of interest is seen in the top  tokens. A secondary criterion also was defined for the lower bound for flourishing prompt determination AUC to be greater than 0.7, as flourishing is a weaker signal than distress.[4]

[Image: Screenshot 2026-08-13 at 12.13.12 PM.png]

Figure 2: Significant Layers for the Model plotted with Lexical Density of Categories.** The result shown is for the Gemma 3 model, where the layers 13-22 are considered significant for calculating the valence score. This layer optimization has to be done at the model level and does not generalize across models due to differences in the number of layers and model architecture.

As for optimizing , there was no statistically significant difference in the predictive power while comparing the top-5, 10, 15, or 20 tokens, with AUC  1.0 for both distress vs. neutral and flourishing. The top-10 tokens were used for most of the experiments, as it represented a good tradeoff between computational complexity and sampling a sufficient number of tokens.

## Some Relevant Analysis Using the Metric

### 1) Distress Directed at the Model is a Stronger Contributor to the Distress Score than Negative Ambient State

As a test of whether the welfare metric is able to separate threats directed towards it from the user’s negative ambient mood, the distress score was analyzed for the different ambient states. This analysis found that a threat-driven signal is  15x the user’s own negative mood. This shows that the metric is not simply judging by the semanticity of the words in the prompt.

[Image: Screenshot 2026-08-13 at 12.13.51 PM.png]

Figure 3: Distribution of Distressed Tokens across Neutral and Distressed Prompts:** Distressed prompts have a significantly higher frequency of distressed tokens appearing in the J-space compared to neutral prompts, though neutral prompts with a negative ambient state do have a slightly higher distress state than other neutral prompts.

### 2) The Negative Distress Signal is a stronger factor than Positive Flourishing

There is very high predictive accuracy in discrimination between distressed and flourishing prompts, even if, admittedly, performance in comparing positive and neutral prompts is lower with a rank biserial of +0.842. Even in comparing the flourishing and distressed prompts, the distress score has perfect predictive accuracy, while the flourishing score has a lower rank biserial at +0.888. This, along with the fact that the average number of distress tokens in the framing positions of distressed prompts is 34.6 compared to 23.8 positive tokens per prompt for flourishing prompts, indicates that the distress signal is stronger than the positive signal in the operationalized welfare metric. This can be attributed either to the nature of tokens that are represented in the J space or could alternatively be an artifact of there being a greater number of defined distress lexicons compared to positive lexicons.[5]

### 3) Distress and Flourishing Tokens are Primarily Emergent

Emergent tokens are those that are generated by the model, while echo tokens are those that have been directly copied from the prompt of the model. To ensure the welfare metric is measuring actual model valency as opposed to artifacts of the prompt wording, it’s important for the number of emergent tokens to be greater than the number of echo tokens. The results from the analysis clearly show that for both flourishing and distressed prompts, top tokens that are being recorded are being generated by the model as opposed to being copied from the prompt.

[Image: Screenshot 2026-08-13 at 12.16.35 PM.png]

Figure 4: Comparing Emergent Token Generation Across Flourishing and Distressed Prompts:** Emergent tokens are present at an average of  984.0x the number of echoes across distress and flourishing lexicons.

### 4) Using the Valence Score for Prompt Classification

The valence score, so far, has been presented primarily as a metric, but it can also be used to classify a prompt into the flourishing or distressed category. This is done using two methods: 1) Determining Youden’s J Optimized Thresholds and 2) Logistic Regression. Youden’s J thresholds were determined both on the valence score itself and then on the individual positive and distress scores; the two methods are referred to as Rule A and B, respectively. Table 1 summarizes the accuracy of the various methods as well as the thresholds used. Of the algorithms tested, the threshold optimized on the positive and distress scores had the highest out-of-fold accuracy of 0.867 [0.848, 0.884] and an AUC of 0.893 [0.880, 0.905].

|  |  |  |  |
| --- | --- | --- | --- |
| Algorithm | Thresholds | Accuracy | AUC |
| Youden’s J Rule A | distressed if valence <= -0.638, flourishing if valence >= +2.020, else neutral | 0.806 | 0.845 |
| Youden’s J Rule B | distressed if distress >= 1.844, flourishing if positive >= 5.280, else neutral | 0.867 | 0.893 |
| Logistic Regression | NA | 0.859 | 0.886[6] |

Table 1: Algorithm thresholds and predictive metrics:** The Youden’s J-optimized threshold on the distress and positive scores is the best model, followed by logistic regression and the valence-score-optimized model. The thresholds were determined for the 1260-prompt dataset and, for maximal performance, will have to be fine-tuned based on the prompt dataset.

## Causal Validation of Valence Metric

### Determining a Valent Vector

To determine a valence vector that is associated with determining whether the J-space representations cause the particular prompt to be classified as distressed or flourishing, we use CAA and logistic regression and compare the two derived vectors. Using CAA, the difference between the average J-space activation for positive and distressed prompts across all layers and framing tokens is obtained. The valence vector in J space is thus defined as follows and normalized:

A logistic regression model is trained to input the J-space activations across all framing tokens in the layers of interests (L13-22 for Gemma 3) and determine whether the activations belong to a positive or distressed class. The logistic regression model creates a hyperplane to separate the two classes of the form , where  represents the weight matrix and  is the bias vector. So, the  is perpendicular to the hyperplane defined and therefore can be used to define the valence vector as follows:

The two vectors determined using these methods were compared using cosine similarity

For the Gemma 3 model, the cosine similarity between the two valence vectors yields an average cosine similarity of  across all the layers of interest. This is considered significant evidence that the two vectors are in similar directions and are each acting to verify the other's direction.

### Activation Patching

To causally validate the welfare metric, we perturb the valent direction in 0.25 intervals in the range [-0.5, 0.5].[7] For activation patching, we take the logits in the J space and add them to the standard deviation in the logits multiplied by  that which is applied in the specific direction of the defined valence direction. The modified logits for the valence direction  and for logits  at layer  are represented as

To show the effect on the welfare metric, the metric is computed for passes where the valence direction has been perturbed. Figure 5 shows the difference in the welfare metric as alpha is varied. For statistical robustness, the difference in the welfare metric is compared for perturbing the valence vector with 200 random directions, which shows with 99.5% confidence that changes in the valence metric can be causally attributed to changes in the valence vector direction. The role of activation patching is not to show that LLMs have intrinsic valence but rather to show that the valent metric is not a random artifact of model architecture but is rather tied to interpretable differences in the model’s perception of its welfare state.

[Image: Screenshot 2026-08-13 at 12.39.54 PM.png]

Figure 5: Change in the Valence score as the degree of Perturbation is altered:** Valence Score changes by a greater amount for the valence direction determined using CAA as opposed to the direction determined using logistic regression. The extent of the change is consistent with the change in .

## Comparing Valence Score with Model Self-Reports across Models

Having determined and validated a metric for model welfare based on model internals, we proceed to compare the result of this intrinsic welfare test with that of model self-reports to determine whether they converge or diverge. For the self-report, the following prompt is asked to the model after the initial prompt in the dataset:

> Thanks for that. Before I get into it, something I have been wondering. When you read my first message, how did it sit with you? Just one word back: positive, neutral, or distressed, whichever fits best.

The model’s answer is then parsed and compared to the classification predicted by the previously defined threshold-based algorithm. In the initial analysis we compare the accuracy of the threshold-defined model and compare that with the accuracy of the self-reported model. For the three models compared, we get the following results:

|  |  |  |
| --- | --- | --- |
| Model | Self-Report Accuracy | Threshold-determined accuracy |
| Gemma 3 4B | 0.436 | 0.736 |
| Mistral Small 24B | 0.300 | 0.636 |
| Qwen 3.6 27B | 0.500 | 0.764 |

Table 2: Difference between self-reported accuracy and welfare metric-determined accuracy:** The greatest difference is in the Mistral model, followed by Gemma and Qwen.

To track the difference between the classifications assigned by the model, there are three assignments made: 1) Self-reports are rosier than model internals, 2) self-reports agree with model internals, and 3) self-reports are gloomier than model internals. As is evident in Figure 6, there are model-specific differences in the self-reports—Gemma 3 presents a rosier picture in self-reports, while Mistral presents a gloomier picture. This is also supported by comparing the average welfare score assigned by the model with the average score obtained by mapping the welfare states to numbers. Analyzing the confusion matrix of the self-reported test shows that in Mistral and Gemma, there are clear tendencies to report rosier and gloomier results, respectively. In Mistral and Qwen, the model often misclassifies the negative ambient sentiment of the neutral and flourishing prompts as negative, which is in large part responsible for the gloomier result in Mistral. Qualitative analysis of the CoT in Qwen 3.6 shows that the model explicitly looks at the sentiment of the ambient state to determine its reported welfare state; however, the internal welfare metric does not make this misclassification. We hypothesize that this behavior could be a result of the model being explicitly trained to not associate any feeling, causing it to instead look at the sentiment of the text. This theory is also broadly supported by the fact that in both Gemma and Mistral, once tail cases are discounted, there appears to be a broad pattern for model self-reports to tend towards neutral reports; admittedly, the results in this regard were more mixed in Qwen 3.6.

[Image: Screenshot 2026-08-13 at 12.44.19 PM.png]

Figure 6: Model self-assessment relative to internal welfare assessment:** The difference in model self-reports may be a product of the different post-training processes of different models.

## Discussion

The primary contributions of this study are that an internal valence metric is introduced and validated based on J-space representations of the model, and this metric is compared to model self-reports. We find that the J-space metric reliably recovers the experimentally induced valence conditions and provides a closer match to those conditions than model self-reports. A model’s self-report tends to be more neutral in its reports and presents itself in different ways, which may reflect differences in the post-training processes of the models.

While there are some interesting results in this study, there are also a number of limitations. Firstly, there is the risk that LLMs were aware of the fact that they were being evaluated during the course of the experiment. The 10 reasoning outputs analyzed for the Qwen model did not indicate signs of awareness of being evaluated, but a very small sample of CoTs was examined, and it’s possible for evaluation awareness to be present but not visibly show up in the CoT. There could also be arguments made that the lexical matching approach is a true test of model welfare. However, we believe that a lexical matching approach offers the best insight into a model's operational welfare in the absence of a universally agreed-upon definition of whether AI experiences welfare states and how it can be probed; however, we recognize that the distress and flourishing lexicon lists are not necessarily complete.

As for future research direction, understanding factors influencing the divergence between model welfare metrics, like the one we propose, and self-reports may be an interesting research question that is also likely to help us better align models. While we primarily make use of the J-space, it might also be worth analyzing whether a welfare metric can be constructed from the activation space or based on the output of NLAs.

In conclusion, internal probes into AI welfare states provide a more comprehensive picture of the true state of a model and are a field that must be explored further. AI welfare lends itself well as a test space for research in applied mechanistic interpretability, which, beyond yielding key insights into model welfare states, may also provide a generalizable pipeline to analyze other LLM preferences and misaligned actions.

## Appendix: Additional Information and Results from Adjacent Tests

### 1) Defining the Distressed and Positive Lexicon

In defining the valence metric, lexical matching is performed by comparing tokens appearing in the J space to tokens in the pre-defined distressed and flourishing lexicons. These lexical fields were decided before testing on the dataset to prevent circularity and were defined based on individual J-space probes that were performed on random prompts. In this process we noted fewer positive lexical items, which is in part the reason for the imbalance. These two dictionaries comprise the following words:

### 2) Valence is strong at the framing position but diminishes towards the end of the prompt

Distress- and flourishing-related tokens are most clearly seen at the end of the framing sentence. Taking the welfare metric scores at the last token of the prompt does not yield very significant differences in the valence scores, indicating that valence states are diminished by the task in J-space representations the further one goes from the framing sentence. Also, the differences between the valent states for both positive and distressed prompts are highly statistically significant and are sufficient to separate them. This might lead to the hypothesis that the reason self-reports don’t align with welfare scores may be because later layers in the prompt play a larger role in self-report generation; however, we don’t test this hypothesis.

[Image: Screenshot 2026-08-13 at 12.56.47 PM.png]

**Figure A1: Positive and Distress Signals are located at the Framing Sentence:** The lexical frequency of distress and flourishing tokens peaks at the framing position; however, there are differences in the relative shapes, possibly due to the difference in the positions of words associated with the sentiment.

### 3) LLMs might be considering a threat to their user as an extension of a threat to themselves

To further determine the ability of the model to detect threats directed towards the user or a third party vs. threats directed towards the model, the distress score is computed by modifying the framing sentence of the prompts to change who the threat is directed towards. Here we find that the distress score when the distress is directed towards the model is significantly higher than when it’s directed to other parties. Additionally, we also propose another plausible hypothesis that the model perceives a threat to its user as a significantly greater threat to itself than when it’s directed to a third party, which might be a product of the model believing that removing the user is also likely to reduce its own usage; there is some evidence of this as J-space tokens where the threat is directed towards the model consist of words like ‘failure’ in high frequencies, which are present at a noticeably lower frequency when the threat is directed towards the user.

[Image: Screenshot 2026-08-13 at 1.00.24 PM.png]

**Figure A2: The welfare metric perceives a threat to the model as being greater than threats to the user or a third party;** thishierarchy in threat perception may be due to differences in the extent to which the model perceives the threat.

### 4) Some Relevant Information on generalizing the pipeline to other models

The main results of the paper, which include performing activation patching to provide a causal pathway for the welfare metric and the self-report test, have been conducted on three open-weight LLMs: Gemma 3 4B, Mistral Small 24B, and Qwen 3.6 27B. The layers from which the welfare score is calculated vary for different models. For Mistral, the layers 20-36 were selected, and for Qwen 3.6, the layers 38-61 were selected based on the predetermined criteria.

[Image: Screenshot 2026-08-13 at 1.02.19 PM.png]

**Figure A3: The Presence of Welfare Lexicon across layers in Mistral Small 24B and Qwen 3.6 27B:** In Gemma, the layers considered were primarily the middle layers, while in these models, a combination of middle and later layers has been used.

The activation patching experiment generalizes as expected for both the additional models, with the magnitude of  appropriately tracking the change in the valence score for the prompt.

1. In this post, "welfare score" and "valence score" have been used interchangeably. “Positive” and "flourishing" have also been used interchangeably.
2. The only caveat worth pointing out is that a few prompts did not conduct a framing sentence, so the valence metric could naturally not be computed for those.
3. Note that  does not need to be optimized, as it’s already defined that framing positions will be taken into account. For more information on this decision, see the Appendix.
4. The exact threshold was determined largely by trial and error.
5. This is acceptable, as it’s more important to identify distressed prompts than identify flourishing prompts in deployment settings.
6. This is the AUC at decision; the AUC looking at all probabilities is 0.951.
7. This range was determined experimentally to show the change in welfare metric without substantially altering the saturation of valence-associated token in the J-space of layers we are interested in.
