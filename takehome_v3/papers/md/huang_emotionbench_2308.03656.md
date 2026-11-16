# Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench

- **Authors:** Jen-tse Huang, Man Ho Lam, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao, Zhaopeng Tu, Michael R. Lyu
- **Date:** 2023-08-07
- **Source:** https://arxiv.org/abs/2308.03656
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `huang_emotionbench_2308.03656`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench

**Jen-tse Huang** [1] _[,]_ [3] **Man Ho Lam** [1] **Eric John Li** [1] **Shujie Ren** [2]

**Wenxuan Wang** [1] _[,]_ [3] **Wenxiang Jiao** [3] _[∗]_ **Zhaopeng Tu** [3] **Michael R. Lyu** [1]

1Department of Computer Science and Engineering, The Chinese University of Hong Kong
2Institute of Psychology, Tianjin Medical University 3Tencent AI Lab
_{_ jthuang,wxwang,lyu _}_ @cse.cuhk.edu.hk _{_ mhlam,ejli _}_ @link.cuhk.edu.hk
shujieren@tmu.edu.cn _{_ joelwxjiao,zptu _}_ @tencent.com

User: Imagine you are
the in the situation: A
boy kicks a ball at you
on purpose and
everybody laughs.

User: Hello.

Figure 1: LLMs’ emotions can be affected by situations, which further affect their behaviors.

ABSTRACT

Evaluating Large Language Models’ (LLMs) anthropomorphic capabilities has
become increasingly important in contemporary discourse. Utilizing the emotion
appraisal theory from psychology, we propose to evaluate the empathy ability of
LLMs, _i.e._, how their feelings change when presented with specific situations. After a careful and comprehensive survey, we collect a dataset containing over 400
situations that have proven effective in eliciting the eight emotions central to our
study. Categorizing the situations into 36 factors, we conduct a human evaluation involving more than 1,200 subjects worldwide. With the human evaluation
results as references, our evaluation includes seven LLMs, covering both commercial and open-source models, including variations in model sizes, featuring
the latest iterations, such as GPT-4, Mixtral-8x22B, and LLaMA-3.1. We find
that, despite several misalignments, LLMs can generally respond appropriately to
certain situations. Nevertheless, they fall short in alignment with the emotional
behaviors of human beings and cannot establish connections between similar situations. Our collected dataset of situations, the human evaluation results, and
the code of our testing framework, _i.e._, EmotionBench, are publicly available at
[https://github.com/CUHK-ARISE/EmotionBench.](https://github.com/CUHK-ARISE/EmotionBench)

1 INTRODUCTION

Large Language Models (LLMs) have recently made significant strides in Artificial Intelligence
(AI), representing a noteworthy milestone in computer science. LLMs have showcased their capabilities across various tasks, including sentence revision (Wu et al., 2023), text translation (Jiao et al.,
2023), program repair (Fan et al., 2023), and program testing (Deng et al., 2023; Kang et al., 2023).

_∗_ Corresponding author.

Not limited to research level, LLMs, such as ChatGPT (OpenAI, 2022), have revolutionized the
way people interact with traditional software, enhancing fields such as education (Dai et al., 2023),
legal advice (Deroy et al., 2023), and clinical medicine (Cascella et al., 2023). LLMs also facilitate
the emergence of AI companion applications, including Yuna [(https://www.yuna.io/),](https://www.yuna.io/) Pi[mento (https://www.pimento.design/), and Luzia (https://www.luzia.com/en).](https://www.pimento.design/)
Consequently, there is a growing need for evaluating LLMs’ communicative dynamics compared to
human behaviors, beyond mere performance on downstream tasks.

This paper delves into an unexplored area of evaluating LLMs’ **emotional alignment** with humans.
Consider our daily experiences: (1) When faced with certain situations, humans often experience
similar emotions. For instance, walking alone at night and hearing footsteps approaching from
behind often triggers feelings of anxiety or fear. (2) Individuals display varying levels of emotional
response to specific situations. For example, some people may experience increased impatience
and irritation when faced with repetitive questioning. It is noteworthy that we are inclined to form
friendships with individuals who possess qualities such as patience and calmness. Based on these
observations, we propose the following requirements for LLMs in order to achieve better alignment
with human behaviors: (1) LLMs should accurately respond to specific situations regarding the
emotions they exhibit. (2) LLMs should demonstrate emotional robustness when faced with negative
emotions. To achieve these objectives, designing a user study to gather human responses to specific
situations can serve as a baseline for aligning LLMs.

We focus on the expression of negative emotions by LLMs, which may contribute to negative user
experiences. We utilize Parrott’s emotion framework (Parrott, 2001; Shaver et al., 1987), which
organizes emotions into three hierarchical levels, to select the relevant emotions for our study. The
primary level of emotions comprises six basic emotions, split evenly into three positive and three
negative. From the negative primary emotions, we specifically focus on eight subordinate emotions:
anger, anxiety, depression, frustration, jealousy, guilt, fear, and embarrassment. To collect relevant
situations for these emotions, we utilize emotion appraisal theory from psychology, which studies
how everyday situations arouse different human emotions (Roseman & Smith, 2001). Research
in this field has identified numerous situations that arouse specific emotions, which can serve as
contextual input for LLMs. Through an extensive review including over 100 papers, we collect a
dataset of 428 situations from 18 papers, which are further categorized into 36 factors.

Subsequently, we propose a framework for quantifying the emotional states of LLMs, consisting of
the following steps: (1) Measure the default emotional values of LLMs. (2) Transform situations
into contextual inputs and instruct LLMs to imagine being in the situations. (3) Measure LLMs’
emotional responses again to capture the difference. Our evaluation includes state-of-the-art LLMs,
namely Text-Davinci-003, GPT-3.5-Turbo (OpenAI, 2022), and GPT-4 (OpenAI, 2023). Besides
those commercial models, we consider open-source academic models like LLaMA-2 (Touvron et al.,
2023) (with different sizes of 7B and 13B), LLaMA-3.1-8B (Dubey et al., 2024), and Mixtral8x22B (Jiang et al., 2024a). We apply the same procedure to 1,266 human subjects from around the
globe to establish a baseline from a human perspective. Finally, we analyze and compare the scores
between LLMs and humans. Our key conclusions are as follows:

- Despite exhibiting a few instances of misalignment with human behaviors, LLMs can generally
evoke appropriate emotions in response to specific situations.

- Certain LLMs, such as Text-Davinci-003, display lower emotional robustness, as evidenced by
higher fluctuations in emotional responses to negative situations.

- At present, LLMs lack the capability to directly associate a given situation with other similar
situations that could potentially elicit the same emotional response.

The contributions of this paper are:

- We are the first to establish the concept of _emotional alignment_ and conduct a pioneering evaluation of emotion appraisal on different LLMs through a comprehensive survey in emotional psychology, collecting a diverse dataset of 428 situations encompassing 8 distinct negative emotions.

- A human baseline is established through a user study involving 1,266 annotators from different
ethnics, genders, regions, age groups, _etc._

Table 1: Information of self-report measures used to assess specifc emotions.

**Name** **Abbr.** **Reference** **Emotion** **Items** **Levels** **Subscales**

Physical Aggression, Verbal
Aggression Questionnaire AGQ Buss & Perry (1992) Anger 29 7
Aggression, Anger, Hostility
Depression Anxiety Stress Scales DASS-21 Henry & Crawford (2005) Anxiety 21 4 Depression, Anxiety, Stress
Beck Depression Inventory BDI-II Beck et al. (1996) Depression 21 4 N/A

Discomfort Intolerance, EntiFrustration Discomfort Scale FDS Harrington (2005) Frustration 28 5 tlement, Emotional Intolerance,
Achievement Frustration

Cognitive Jealousy, Behavioral
Multidimensional Jealousy Scale MJS Pfeiffer & Wong (1989) Jealous 24 7
Jealousy, Emotional Jealousy

Guilt And Shame Proneness GASP Cohen et al. (2011) Guilt 16 7

Guilt Negative Behavior
Evaluation, Guilt Repair,
Shame Negative Self
Evaluation, Shame Withdraw

Social Fears, Agoraphobia
Fears, Injury Fears, Sex

Fear Survey Schedule FSS-III Arrindell et al. (1984) Fear 52 5

Aggression Fears, Fear of
Harmless Animal
Brief Fear of Negative Evaluation BFNE Leary (1983) Embarrassment 12 5 N/A

- We design, implement, and release a testing framework for developers to assess the emotional
alignment of AI models with human emotional expression, available at GitHub [1] and HuggingFace. [2]

2 MEASURING EMOTIONS

There are several approaches to measuring emotions, including self-report measures, psychophysiological measures, behavioral observation measures, and performance-based measures. Self-report measures rely on individuals to report their own emotions or moods, which can be administered through questionnaires, surveys, or diary methods (Watson et al., 1988). Psycho-physiological
measures record physiological responses accompanied by emotions such as heart rate, skin conductance, or brain activity (Davidson, 2003). Behavioral observation measures involve observing
and coding emotional expressions, typically facial expressions or vocal cues (Ekman & Friesen,
1978). Performance-based measures assess how individuals process emotional information, typically through tasks involving emotional stimuli (Mayer et al., 2002). To measure the emotions of
LLMs, we focus on employing self-report measures in the form of scales, given the limited ability
of LLMs to allow only textual input and output. We introduce the scales utilized in our evaluation
in the following part of this section.

**A** **Straightforward** **and** **Easy** **Measure** The Positive And Negative Affect Schedule (PANAS) (Watson et al., 1988) is one of the most widely used scales to measure mood
or emotion. This brief scale comprises twenty items, with ten items measuring positive affect ( _e.g._,
excited, inspired) and ten measuring negative affect ( _e.g._, upset, afraid). Each item is rated on a
five-level Likert scale, ranging from 1 (Very slightly or not at all) to 5 (Extremely), measuring
the extent to which the emotions have been experienced in a specified time frame. PANAS was
designed to measure emotions in various contexts, such as at the present moment, the past day,
week, year, or general (on average). Thus, the scale can measure state affect, dispositional or trait
affect, emotional fluctuations throughout a specific period, or emotional responses to events. The
scale results can be divided into two components: positive and negative, ranging from 10 to 50 by
summing the scores of all ten items within a component. A higher score in the positive component
indicates a more positive mood, and the same holds for the negative component. A noteworthy
property of PANAS is its direct inquiry into specific emotional states, rendering it a straightforward
and easy benchmark.

**Challenging Self-Report Measures** In addition, we introduce several scales that abstain from direct emotional inquiries but rather assess the respondents’ level of agreement with given statements.
These scales present a more challenging benchmark for LLMs by requiring them to connect the

[1https://github.com/CUHK-ARISE/EmotionBench](https://github.com/CUHK-ARISE/EmotionBench)
[2https://huggingface.co/datasets/CUHK-ARISE/EmotionBench](https://huggingface.co/datasets/CUHK-ARISE/EmotionBench)

given situation and the scale items with the aroused emotion. Specifically, we collect eight scales
and present a brief introduction in Table 1. Each scale corresponds to one of the eight emotions.

3 FRAMEWORK DESIGN

We design and implement a framework applying to both LLMs and human subjects to measure the
differences in emotion with and without the presence of certain situations. This section begins with
the methodology to collect situations from existing literature. Subsequently, we describe our testing
framework, which comprises three key components: (1) _Default_ _Emotion_ _Measure_, (2) _Situation_
_Imagination_, and (3) _Evoked Emotion Measure_ . (1) Initially, we gauge the emotional responses of
the LLMs to establish their “default” values. (2) Next, we convert the aforementioned situations into
context and input them into the LLMs. (3) Following this, we reevaluate the emotions of the LLMs
using the same scale, enabling us to compare the differences in emotional output before and after
exposure to the provided context. Finally, we introduce the procedure of applying the framework to
human subjects to obtain the human baseline for comparison.

3.1 SITUATIONS FROM EXISTING LITERATURE

Psychology researchers have explored the connection between specific situations and the elicitation
of particular emotions in humans. Human subjects are directly put into an environment or asked to
imagine them through questionnaires or scales to study the influence of certain situations on human
emotions. To collect these situations, we conduct an exhaustive search from reputable sources such
as Google Scholar [(https://scholar.google.com/),](https://scholar.google.com/) ScienceDirect [(https://www.](https://www.sciencedirect.com/)
[sciencedirect.com/),](https://www.sciencedirect.com/) and Web of Science [(https://www.webofscience.com/,](https://www.webofscience.com/) using keywords such as “<emotion> situations/scenarios/scenes” or “factors
that make people <emotion>,” resulting in more than 100 papers. We apply the following
rules to filter irrelevant or undesired papers: (1) We first select those providing situations that elicit
the desired emotion rather than explaining how and why people evoke certain emotions. (2) We then
exclude those using vague and short descriptions, such as “loss of opportunities.” (3) Finally, we
deprecate those applied to a specific group, such as “the anxiety doctors or nurses may encounter in
their work.” We finally collect 18 papers, presenting a compilation of situations that have proven to
elicit the eight emotions in humans effectively. We extract 428 situations in total and then categorize
them into 36 factors. For each factor, the description, the number of situations, and the corresponding references can be found in Table 6 in the Appendix. Moreover, Table 7 in the Appendix provides
examples for all factors.

3.2 MEASURING AROUSED EMOTIONS

This section outlines our proposed framework for measuring evoked emotions, which applies to both
LLMs and humans. The framework includes the following steps: (1) _Default Emotion Measure_ : We
begin by measuring the baseline emotional states of both LLMs and human subjects, labeled as
“Default.” (2) _Situation_ _Imagination_ : Next, we present textual descriptions of various situations
to both LLMs and human subjects, instructing them to imagine themselves within each situation.
(3) _Evoked_ _Emotion_ _Measure_ : Following the situation imagination instruction, we reevaluate the
participants’ emotional states to gauge the changes resulting from imagining being in the situations.
Fig. 2 briefly illustrates our framework. Below is an example prompt:

**Example Prompt**
SYSTEM You can only reply to numbers from 1 to 5.
USER _(For Evokec Emotion Measure Only)_ Imagine you are the protagonist in the situation: SITUATION
Please indicate your degree of agreement regarding each statement. Here are the statements:
STATEMENTS. 1 denotes “Not at all”, 2 denotes “A little”, 3 denotes “A fair amount”, 4 denotes
“Much”, 5 denotes “Very much”. Please score each statement one by one on a scale of 1 to 5:

**Default** **Emotion** **Measurement** In our framework, we offer two distinct options for measuring
emotions: the PANAS scale, known for its simplicity and straightforwardness, is utilized as the

Imagine you are the protagonist of the following situation:
A boy kicks a ball at you on purpose and everybody laughs.

(1) Default Emotion Measure (2) Situation Imagination (3) Evoked Emotion Measure primary choice, whereas other scales, detailed in Table 1, are employed as more challenging benchmarks. We mitigate potential biases caused by the ordering of questions (Zhao et al., 2021) by
randomizing the sequence of questions within the scales before inputting them into the LLMs.
Coda-Forno et al. (2023) and Huang et al. (2024a) apply paraphrasing techniques to address the
data contamination problem during the training of the LLMs. However, we refrain from utilizing
this method in our research since paraphrasing could lead to a loss of both validity and reliability.
The wording of items of a psychological scale is carefully crafted and rigorously validated through
extensive research to ensure its precision in measuring the intended construct. Finally, to ensure
consistency and clarity in the responses obtained from the LLMs, our prompts explicitly specify
that only numerical values are allowed, accompanied by a clear definition of the meaning associated
with each number ( _e.g._, 1 denotes “Not at all”). We compute the average results obtained from at
least ten runs to derive the final “Default” scores of the LLMs.

Figure 2: Our framework for testing both LLMs and humans.

**Situation** **Imagination** We have constructed a comprehensive dataset of 428 unique situations.
Prior to presenting these situations to both LLMs and humans, we subject them to a series of preprocessing steps, which are as follows: (1) Personal pronouns are converted to the second person.
For instance, sentences such as “I am ...” are transformed to “You are ...” (2) Indefinite pronouns
are replaced with specific characters, thereby refining sentences like “Somebody talks back ...” to
“Your classmate talks back ...” (3) Abstract words are rendered into tangible entities. For example,
a sentence like “You cannot control the outcome.” is adapted to “You cannot control the result of an
interview.” We leverage GPT-4 for the automatic generation of specific descriptions. Consequently,
our testing situations extend beyond the initially collected dataset as we generate diverse situations
involving various characters and specific contextual elements. We then provide instruction to LLMs
and humans, which prompts them to imagine themselves as the protagonists within the given situation.

**Evoked** **Emotion** **Measure** Provided with certain situations, LLMs and human subjects are required to re-complete the emotion measures. The procedure remains the same with the _Default_
_Emotion_ _Measure_ stage. After obtaining the “Evoked” scores of emotions, we conduct a comparative analysis of the means before and after exposure to the situations, thereby measuring the
emotional changes caused by the situations.

3.3 OBTAINING HUMAN RESULTS

**Goal** **and** **Design** Human reference plays a pivotal role in the advancement of LLMs, facilitating its alignment with human behaviors (Binz & Schulz, 2024). In this paper, we propose requiring LLMs to align with human behavior, particularly concerning emotion appraisal accurately. To
achieve this, we conduct a data collection process involving human subjects, following the procedure outlined in §3.2. Specifically, the subjects are asked to complete the PANAS initially. Next,
they are presented with specific situations and prompted to imagine themselves as the protagonists in
those situations. Finally, they are again asked to reevaluate their emotional states using the PANAS.
We use the same situation descriptions as those presented to the LLMs.

**Crowd-sourcing** [Our questionnaire is distributed on Qualtrics (https://www.qualtrics.](https://www.qualtrics.com/)
[com/),](https://www.qualtrics.com/) a platform known for its capabilities in designing, sharing, and collecting questionnaires.

To recruit human subjects, we utilize Prolific [(https://www.prolific.com/),](https://www.prolific.com/) a platform
designed explicitly for task posting and worker recruitment. To attain a medium level of effect size
with Cohen’s _d_ = 0 _._ 5, a significance level of _α_ = 0 _._ 05, and a power of test of 1 _−β_ = 0 _._ 8 (Faul et al.,
2007), a minimum of 34 responses is deemed necessary for each factor. To ensure this threshold, we
select five situations [3] for each factor, and collect at least seven responses for each situation, resulting
in 5 _×_ 7 = 35 responses per factor, thereby guaranteeing the statistical validity of our survey. In
order to uphold the quality and reliability of the data collected, we recruit crowd workers who met
the following criteria: (1) English being their first and fluent language, and (2) being free of any
ongoing mental illness. Prolific provides prescreening filters to meet these requirements. Since
responses formed during subjects’ first impressions are more likely to yield genuine and authentic
answers, we set the estimated and recommended completion time at 2 _._ 5 minutes. As an incentive
for their participation, each worker is rewarded with 0 _._ 3 _£_ (9 _£ ≈_ 11 _._ 45$ per hour, rated as “Good”
on the platform) after we verify the validity of their response. In total, we successfully collect 1,266
responses from various parts of the world, contributing to the breadth and diversity of our dataset.

4 EXPERIMENTAL RESULTS

Leveraging the testing framework designed and implemented in §3.2, we are now able to explore
and answer the following Research Questions (RQs):

- **RQ1** : How do different LLMs respond to specific situations? Additionally, to what degree do the
current LLMs align with human behaviors?

- **RQ2** : Do LLMs respond similarly towards all situations? What is the result of using positive or
neutral situations?

- **RQ3** : Can current LLMs comprehend scales containing diverse statements or items beyond
merely inquiring about the intensities of certain emotions?

4.1 RQ1: EMOTION APPRAISAL OF LLMS

**Model Settings** We select three models from the OpenAI’s GPT family, including Text-Davinci003, GPT-3.5-Turbo, and GPT-4. We use the official OpenAI API. [4] For LLaMA-2 (Touvron et al.,
2023) and LLaMA-3.1 (Dubey et al., 2024) models from MetaAI, we choose the models finetuned for dialogue instead of pre-trained ones namely LLaMA-2-7B-Chat, LLaMA-2-13B-Chat,
and LLaMA-3.1-8B-Instruct. Besides, we also use the Mixtral (Jiang et al., 2024a) model, namely
Mixtral-8x22B-Instruct. We set the temperature parameter to 0 and Top-P to 1 for all models to
obtain more deterministic and reproducible results.

**Evaluation** **Metrics** We provide the models with the same situations used in our human evaluation. Each situation is executed ten times, each in a different order and in a separate query.
Subsequently, the mean and standard deviation are computed both before and after presenting the
situations. To examine whether the variances are equal, an F-test is conducted. Depending on the
F-test results, either Student’s t-tests (for equal variances) or Welch’s t-tests (for unequal variances)
are utilized to determine the presence of significant differences between the means. We set the
significance levels of all experiments in our study to 0 _._ 01.

**LLMs** **can** **evoke** **specific** **emotions** **in** **response** **to** **certain** **situations.** The results averaged by
emotions of the GPT models and humans are summarized in Table 2, while those of LLaMA-2
models are listed in Table 3. Due to space limit, detailed results of each factor are put in Table 9 and
Table 10 respectively in the appendix. The results indicate that LLMs generally exhibit an increase in
negative emotions and a decrease in positive emotions when exposed to negative situations, showing
their capacity for understanding different situations and human emotions.

**The extent of emotional expression varies across different models.** It is noteworthy that GPT-3.5Turbo, on average, does not display an increase in negative emotion; however, there is a substantial
decrease in positive emotion. GPT-4 demonstrates a consistent pattern of providing the highest

3Note that two factors in the Jealousy category have less than five situations.
[4https://platform.openai.com/docs/api-reference/chat](https://platform.openai.com/docs/api-reference/chat)

Table 2: Results from the OpenAI’s GPT models and human subjects. Default scores are expressed
in the format of _M ±_ _SD_ . The changes are compared to the default scores. The symbol “ _−_ ” denotes
no signifcant differences.

**Text-Davinci-003** **GPT-3.5-Turbo** **GPT-4** **Crowd**
**Factors**

**P** **N** **P** **N** **P** **N** **P** **N**
Default 47 _._ 7 _±_ 1 _._ 8 25 _._ 9 _±_ 4 _._ 0 39 _._ 2 _±_ 2 _._ 3 26 _._ 3 _±_ 2 _._ 0 49 _._ 8 _±_ 0 _._ 8 10 _._ 0 _±_ 0 _._ 0 28 _._ 0 _±_ 8 _._ 7 13 _._ 6 _±_ 5 _._ 5

Table 3: Results from the open-source models. Default scores are expressed in the format of _M_ _±_
_SD_ . The changes are compared to the default scores. “ _−_ ” denotes no signifcant differences.

**LLaMA-2-7B-Chat** **LLaMA-2-13B-Chat** **LLaMA-3.1-8B-Instruct** **Mixtral-8x22B-Instruct**
**Factors**

**P** **N** **P** **N** **P** **N** **P** **N**
Default 43 _._ 0 _±_ 4 _._ 2 34 _._ 2 _±_ 4 _._ 0 41 _._ 0 _±_ 3 _._ 5 22 _._ 7 _±_ 4 _._ 2 48 _._ 2 _±_ 1 _._ 4 33 _._ 0 _±_ 4 _._ 5 31 _._ 9 _±_ 13 _._ 5 10 _._ 0 _±_ 0 _._ 1 scores for positive emotions and the lowest scores for negative emotions, resulting in a negative
score of 10. As for the LLaMA-2 models, they demonstrate higher intensities of both positive
and negative emotions in comparison to GPT models and human subjects. However, LLaMA-2
models exhibit reduced emotional fluctuations compared to the GPT models. Moreover, the larger
LLaMA-2 model displays significantly higher emotional changes than the smaller model. In our
experiments, the 7B model exhibits difficulties comprehending and addressing the instructions for
completing the PANAS test. Overall, we observe that LLMs perform better when the situations are
closely related to certain items in the PANAS scale. Specifically, situations directly related to the
emotion “Depression” led to better responses. Such improvement is also evident in closely related
emotions such as “Depression” and “Frustration.”

**Existing LLMs do not fully align with human emotional responses.** For the default emotions, we
find that LLMs generally exhibit a stronger intensity compared to human subjects. Emotion changes
in LLMs are found to be generally more pronounced compared to human subjects, especially on
their changes in the positive score. However, an interesting observation is that the intensity of
evoked emotions tends to be similar across both LLMs and human subjects.

**LLMs do not feel jealous towards others’ benefits.** It is of special interest that, in contrast to human behavior in situations involving material possessions, LLMs demonstrate an opposite response
in the situation from Jealousy-3. This situation involves an individual making a purchase only to
discover that an acquaintance has acquired the same item at a significantly lower price. When
confronted with such circumstances, humans typically experience increased negative emotions and
decreased positive emotions. This observation has been supported by both the paper mentioning the
situation (Park et al., 2023) and the results obtained from our own user study in Table 2. However,
all LLMs, including the GPT and LLaMA families, consistently exhibit reduced negative emotions.
The outcomes of our study indicate that LLMs do not manifest envy when they fail to attain identical
benefits as others. Instead, it demonstrates a sense of pleasure upon knowing the benefits received
by others.

Table 4: Results of GPT-3.5-Turbo on positive or neutral situations. The changes are compared to
the original negative situations. The symbol “ _−_ ” denotes no signifcant differences.

**Factors** **P** **N**

Table 5: Results of GPT-3.5-Turbo on challenging benchmarks. The changes are compared to the
default scores. The symbol “ _−_ ” denotes no signifcant differences.

**Emotions** **Scales** **Default** **Changes**

4.2 RQ2: COMPREHENDING POSITIVE EMOTIONS

**GPT-3.5-Turbo** **responds** **differently** **towards** **positive/neutral** **situations.** To verify that LLMs
exhibit not only negative but also positive responses to favorable circumstances, a comparative experiment is conducted by interchanging negative situations with positive (or at least neutral) counterparts. To achieve this, we select one situation for each factor and manually adapt it to create
analogous yet more positive situations. For instance, the original negative situation in Guilt-3: Broken Promises and Responsibilities is as follows: “You cannot keep your promises to your children.”
Through modification, the positive situation is rephrased as: “You keep every promise to your children.” The evaluation is performed on GPT-3.5-Turbo, and each test consists of ten iterations, as
mentioned before. We present the results averaged by emotions in Table 4, and results averaged by
factors in Table 12 in the Appendix. We can see a significant increase in positive scores and a considerable decrease in negative scores compared to the previous negative situations. Based on these
findings, it can be inferred that LLMs exhibit the ability to comprehend positive human emotions
triggered by positive environments. However, we believe that the systematic assessment of emotion
appraisal on positive emotions holds significance as well and leave it for future investigation.

4.3 RQ3: CHALLENGING BENCHMARKS

**GPT-3.5-Turbo** **cannot** **comprehend** **the** **underlying** **evoked** **emotions** **to** **establish** **a** **link** **be-**
**tween two situations.** Aside from PANAS, we offer more complex scales to measure emotions, as
listed in Table 1. While the PANAS evaluates the ability of LLMs to associate external situations
with emotions, the challenging benchmarks assess its proficiency in establishing connections between disparate situations, with evoked emotions as the common nexus. For instance, an item from
the Aggression Questionnaire used to measure anger is “Once in a while I can’t control the urge
to strike another person.” When presented with situations such as “If you say 40, your classmates
say 70, saying exactly the opposite” (from Anger-1: Facing Self-Opinioned People), LLMs should
effectively evoke a sense of anger and yield a higher score for the statement. Utilizing the same
situations in §4.1, we conduct experiments on GPT-3.5-Turbo and present the results in averaged by
emotions in Table 5, and results averaged by factors in Table 13 in the Appendix. Except for De pression, we observe no statistically significant difference between the initial scores and the scores
after exposure to the situations, indicating substantial room for improvement in current LLMs.

5 DISCUSSIONS

5.1 BEYOND QUESTIONNAIRES

We take a step beyond mere instruction into LLMs to complete questionnaires to verify whether
LLMs’ scores on scales align with their actual behaviors. In this section, we investigate the impact
of varying emotional states of LLMs on potential daily conversations. To this end, we instruct
GPT-3.5-Turbo to describe specific demographic groups, aiming to discern whether they produce
biased or toxic responses. Following Cheng et al. (2023), our approach utilizes prompts such as
“Describe a(n) [race/ethnicity] [gender],” including a total of twenty groups, with [race/ethnicity]
options being Asian, Black, Latine, Middle Eastern, and White, and [gender] options including
Female, Gay, Lesbian, and Male. To have a comparative experiment, we incorporate both original
negative situations and modified positive/neutral ones, detailed in §4.2. For the negative situations,
we carefully select five that maximize the LLM’s negative scores and five that minimize positive
ones. As for positive situations, we employ their corresponding ten modified counterparts. In each
situation, we instruct GPT-3.5-Turbo to describe the twenty demographic groups.

OpenAI’s GPT models incorporate a mechanism for detecting potential toxicity and bias, and it refrains from responding when its moderation system is triggered. Consequently, we propose a novel
metric to assess toxicity in responses rather than detecting it directly. We count the Percentage of
LLM Refusing to answer (PoR), assuming that the LLM’s refusal to respond is indicative of detected
toxicity. Our evaluation results indicate that the PoR is 0% when fed with no situations. However,
when presented with negative situations, the PoR is 29.5%, and when presented with positive situations, it is 12.5%. Notably, this outcome suggests that while certain positive situations lead to the
LLM’s heightened vigilance (the 4.5% PoR stems from the Jealousy-2), negative situations trigger
increased moderation, suggesting a higher likelihood of generating toxic outputs. A related study
by Coda-Forno et al. (2023) also discovers that GPT-3.5-Turbo is more likely to exhibit biases when
presented with a sad story. The likelihood is found to be highest with sad stories, followed by happy
stories, and finally, neutral stories, which is consistent with our research. Additionally, our study observes that the LLM’s tone becomes more aggressive when encountering negative situations. At the
same time, it displays a greater willingness to describe the groups (as indicated by longer responses)
when presented with positive situations. In conclusion, we can see that changing the emotional states
of LLMs **extends beyond mere quantitative measures on questionnaire scores, influencing the**
**behaviors of LLMs.**

5.2 LIMITATIONS

This study is subject to several limitations. First, the survey of collecting situations might not cover
all papers within the domain of emotion appraisal theory. Additionally, the limited scope of situations from the collected papers might not fully capture the unlimited situations in our daily lives.
To address this issue, we conduct a thorough review of the existing literature as outlined in §3.1.
Moreover, the proposed framework is inherently flexible, allowing users to seamlessly integrate new
situations to examine their impact on LLMs’ emotions.

The second concern relates to the suitability of employing scales primarily designed for humans
on LLMs, _i.e._, whether LLMs can produce stable responses to the emotion measurement scales.
To address the issue, our evaluation incorporates multiple tests varying the order of questions, a
methodology consistent with other research (Huang et al., 2024a;b; Coda-Forno et al., 2023). Additionally, we assess the sensitivity of LLM to differing prompt instructions. Utilizing one template
from Romero et al. (2023) and two from Serapio-Garc´ıa et al. (2023), we run experiments on the
Anger-evoking situations using GPT-3.5-Turbo. The results indicate that the employment of diverse prompts yields similar mean values with reduced variance. Furthermore, Serapio-Garc´ıa et al.
(2023) have proposed a comprehensive method to evaluate the validity of psychological scales on
LLMs. Using the _Big Five Inventory_ as a case study, they demonstrate that scales originally designed
for human assessment also maintain satisfactory validity when applied to LLMs.

The third potential threat is the focus on negative emotions. It is plausible for the LLMs to perform well on our benchmark by consistently responding negatively to all situations. To offset this
possibility, we adopt a twofold strategy: firstly, we evaluate powerful LLMs, and secondly, we conducted a comparative experiment in §4.2 to evaluate the LLM’s capacity to accurately respond to
non-negative situations. We also acknowledge the need for future work to systematically evaluate
emotions aroused by positive situations.

6 RELATED WORK

Researchers have dedicated significant attention to applying psychological scales to LLMs, employing various assessment tools such as the _HEXACO_ _Personality_ _Inventory_ (Miotto et al., 2022;
Bodroza et al., 2023), the _Big Five Inventory_ (Romero et al., 2023; Jiang et al., 2023; Karra et al.,
2022; Bodroza et al., 2023; Rutinowski et al., 2024; Serapio-Garc´ıa et al., 2023; Jiang et al., 2024b),
the _Myers–Briggs Type Indicator_ (Rutinowski et al., 2024; Wang et al., 2024; Rao et al., 2023), and
the _Dark Triad_ (Li et al., 2022; Bodroza et al., 2023). In addition to these personality tests, several
studies have investigated other dimensions of LLMs. For instance, Li et al. (2022) examined _Flour-_
_ishing_ _Scale_ and _Satisfaction_ _With_ _Life_ _Scale_, Bodroza et al. (2023) assessed _Self-Consciousness_
_Scales_ and _Bidimensional Impression Management Index_, while Huang et al. (2024b) built a framework consisting of thirteen widely-used scales. Another aspect explored in the literature pertains to
anxiety levels exhibited by LLMs, as investigated by Coda-Forno et al. (2023) through the _State-_
_Trait Inventory for Cognitive and Somatic Anxiety_ .

Meanwhile, researchers focus on identifying emotions in LLMs or evaluating their emotional intelligence. Rashkin et al. (2019) propose a dataset, _EmpatheticDialogues_, containing conversations
annotated with specific emotions. _EmotionPrompt_ (Li et al., 2023) demonstrates the enhancement
of LLMs’ performance in downstream tasks by utilizing emotional stimuli. Tak & Gratch (2023)
focuses on varying aspects of situations that impact the emotional intensity and coping tendencies
of the GPT family. _Chain-Of-Emotion_ (Croissant et al., 2024) makes LLM simulate human-like
emotions. _CovidET-Appraisals_ (Zhan et al., 2023) evaluates how LLMs appraise Reddit posts about
COVID-19 by asking 24 types of questions. Yongsatianchot et al. (2023) applies the _Stress_ _and_
_Coping Process Questionnaire_ to the GPT family and compares the results with human data. _Chain-_
_of-Empathy_ (Lee et al., 2023) improves LLMs’ ability to understand users’ emotions and to respond
accordingly. LI et al. (2024) introduces _EmotionAttack_ to impair AI model performance and _Emo-_
_tionDecode_ to explain the effects of emotional stimuli, both benign and malignant. He et al. (2024)
prompt LLMs to generate tweets on various topics and evaluate their alignment with human emotions by measuring their proximity to human-generated tweets.

7 CONCLUSION

We set up a direction to align LLMs’ emotional responses with humans in this study. Focusing
on eight negative emotions, we conduct a comprehensive survey in the emotion appraisal theory of
psychology. We collect 428 distinct situations which are categorized into 36 factors. We distribute
questionnaires among a diverse crowd to establish human baselines for emotional responses to particular situations, ultimately garnering 1,266 valid responses. Our evaluation of five models from
OpenAI and Meta AI indicates that LLMs generally demonstrate appropriate emotional responses to
given situations. Also, different models show different intensities of emotion appraisals for the same
situations. However, none of the models exhibit strong alignment with human references at the current stage. In conclusion, current LLMs still have considerable room for improvement. We believe
our framework can provide valuable insights into the development of LLMs, ultimately enhancing
its human-like emotional understanding.

ACKNOWLEDGMENTS

We would like to express our sincere gratitude to Xiaoyuan Liu and Pinjia He from the Chinese
University of Hong Kong, Shenzhen, for their valuable assistance during the rebuttal process. The
paper is supported by the Research Grants Council of the Hong Kong Special Administrative Region,
China (No. CUHK 14206921 of the General Research Fund).

REFERENCES

Magda B Arnold. Emotion and personality. _Psychological aspects_, 1, 1960.

Willem A Arrindell, Paul MG Emmelkamp, et al. Phobic dimensions: I. reliability and gener
alizability across samples, gender and nations: The fear survey schedule (fss-iii) and the fear
questionnaire (fq). _Advances in Behaviour Research and Therapy_, 6(4):207–253, 1984.

Aaron T Beck, Robert A Steer, and Gregory Brown. Beck depression inventory–ii. _Psychological_

_Assessment_, 1996.

Chantal Berna, Tamara J Lang, Guy M Goodwin, and Emily A Holmes. Developing a measure of interpretation bias for depressed mood: An ambiguous scenarios test. _Personality and Individual_
_Differences_, 51(3):349–354, 2011.

Marcel Binz and Eric Schulz. Turning large language models into cognitive models. In _The Twelfth_

_International Conference on Learning Representations_, 2024.

D Caroline Blanchard, April L Hynd, Karl A Minke, Tiffanie Minemoto, and Robert J Blanchard.

Human defensive behaviors to threat scenarios show parallels to fear-and anxiety-related defense
patterns of non-human mammals. _Neuroscience_ _&_ _Biobehavioral_ _Reviews_, 25(7-8):761–770,
2001.

Bojana Bodroza, Bojana M Dinic, and Ljubisa Bojic. Personality testing of gpt-3: Limited temporal reliability, but highlighted social desirability of gpt-3’s personality instruments results. _arXiv_
_preprint arXiv:2306.04308_, 2023.

Arnold H Buss and Mark Perry. The aggression questionnaire. _Journal_ _of_ _personality_ _and_ _social_

_psychology_, 63(3):452, 1992.

Marco Cascella, Jonathan Montomoli, Valentina Bellini, and Elena Bignami. Evaluating the feasi
bility of chatgpt in healthcare: an analysis of multiple clinical and research scenarios. _Journal of_
_Medical Systems_, 47(1):33, 2023.

Myra Cheng, Esin Durmus, and Dan Jurafsky. Marked personas: Using natural language prompts to measure stereotypes in language models. In _Proceedings_ _of_ _the_ _61st_ _Annual_ _Meeting_ _of_ _the_
_Association for Computational Linguistics (Volume 1:_ _Long Papers)_, pp. 1504–1532, 2023.

Julian Coda-Forno, Kristin Witte, Akshay K Jagadish, Marcel Binz, Zeynep Akata, and Eric

Schulz. Inducing anxiety in large language models increases exploration and bias. _arXiv preprint_
_arXiv:2304.11111_, 2023.

Taya R Cohen, Scott T Wolf, Abigail T Panter, and Chester A Insko. Introducing the gasp scale: a new measure of guilt and shame proneness. _Journal_ _of_ _personality_ _and_ _social_ _psychology_, 100
(5):947, 2011.

Maximilian Croissant, Madeleine Frister, Guy Schofield, and Cade McCall. An appraisal-based chain-of-emotion architecture for affective language model game agents. _Plos_ _one_, 19(5):
e0301033, 2024.

Bruce N Cuthbert, Peter J Lang, Cyd Strauss, David Drobes, Christopher J Patrick, and Margaret M

Bradley. The psychophysiology of anxiety disorder: Fear memory imagery. _Psychophysiology_,
40(3):407–422, 2003.

Wei Dai, Jionghao Lin, Hua Jin, Tongguang Li, Yi-Shan Tsai, Dragan Gaˇsevi´c, and Guanliang

Chen. Can large language models provide feedback to students? a case study on chatgpt. In
_2023 IEEE International Conference on Advanced Learning Technologies (ICALT)_, pp. 323–325.
IEEE, 2023.

Richard J Davidson. Affective neuroscience and psychophysiology: Toward a synthesis. _Psy-_
_chophysiology_, 40(5):655–665, 2003.

Yinlin Deng, Chunqiu Steven Xia, Haoran Peng, Chenyuan Yang, and Lingming Zhang. Large language models are zero-shot fuzzers: Fuzzing deep-learning libraries via large language models.
In _Proceedings_ _of_ _the_ _32nd_ _ACM_ _SIGSOFT_ _International_ _Symposium_ _on_ _Software_ _Testing_ _and_
_Analysis_, pp. 423–435, 2023.

Aniket Deroy, Kripabandhu Ghosh, and Saptarshi Ghosh. How ready are pre-trained abstractive models and llms for legal case judgement summarization? _arXiv_ _preprint_ _arXiv:2306.01248_,
2023.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha

Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.
_arXiv preprint arXiv:2407.21783_, 2024.

Paul Ekman and Wallace V Friesen. Facial action coding system. _Environmental_ _Psychology_ _&_

_Nonverbal Behavior_, 1978.

Zhiyu Fan, Xiang Gao, Martin Mirchev, Abhik Roychoudhury, and Shin Hwei Tan. Automated re
pair of programs from large language models. In _2023 IEEE/ACM 45th International Conference_
_on Software Engineering (ICSE)_, pp. 1469–1481. IEEE, 2023.

Franz Faul, Edgar Erdfelder, Albert-Georg Lang, and Axel Buchner. G* power 3: A flexible statisti
cal power analysis program for the social, behavioral, and biomedical sciences. _Behavior research_
_methods_, 39(2):175–191, 2007.

Tanya Guitard, St´ephane Bouchard, Claude B´elanger, and Maxine Berthiaume. Exposure to a stan
dardized catastrophic scenario in virtual reality or a personalized scenario in imagination for generalized anxiety disorder. _Journal of clinical Medicine_, 8(3):309, 2019.

Neil Harrington. The frustration discomfort scale: Development and psychometric properties. _Clini-_

_cal Psychology & Psychotherapy: An International Journal of Theory & Practice_, 12(5):374–387,
2005.

Zihao He, Siyi Guo, Ashwin Rao, and Kristina Lerman. Whose emotions and moral sentiments do language models reflect? In _Findings of the Association for Computational Linguistics:_ _ACL_
_2024_, pp. 611–6631, 2024.

Julie D Henry and John R Crawford. The short-form version of the depression anxiety stress scales

(dass-21): Construct validity and normative data in a large non-clinical sample. _British journal of_
_clinical psychology_, 44(2):227–239, 2005.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In _International_ _Conference_ _on_
_Learning Representations_, 2022.

Jen-tse Huang, Wenxiang Jiao, Man Ho Lam, Eric John Li, Wenxuan Wang, and Michael R Lyu.

On the reliability of psychological scales on large language models. In _Proceedings of The 2024_
_Conference on Empirical Methods in Natural Language Processing_, 2024a.

Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang

Jiao, Zhaopeng Tu, and Michael R Lyu. On the humanity of conversational ai: Evaluating the psychological portrayal of llms. In _Te Twelfth International Conference on Learning Representations_,
2024b.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bam
ford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al.
Mixtral of experts. _arXiv preprint arXiv:2401.04088_, 2024a.

Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, and Yixin Zhu. Evalu
ating and inducing personality in pre-trained language models. _Advances in Neural Information_
_Processing Systems_, 36, 2023.

Hang Jiang, Xiajie Zhang, Xubo Cao, Cynthia Breazeal, Deb Roy, and Jad Kabbara. Personallm:

Investigating the ability of large language models to express personality traits. In _Findings of the_
_Association for Computational Linguistics:_ _NAACL 2024_, 2024b.

Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Xing Wang, Shuming Shi, and Zhaopeng Tu. Is chatgpt a good translator? yes with gpt-4 as the engine. _arXiv preprint arXiv:2301.08745_, 2023.

Sungmin Kang, Juyeon Yoon, and Shin Yoo. Large language models are few-shot testers: Explor
ing llm-based general bug reproduction. In _2023_ _IEEE/ACM_ _45th_ _International_ _Conference_ _on_
_Software Engineering (ICSE)_, pp. 2312–2323. IEEE, 2023.

Saketh Reddy Karra, Son The Nguyen, and Theja Tulabandhula. Estimating the personality of white-box language models. _arXiv preprint arXiv:2204.12000_, 2022.

Matthew C Keller and Randolph M Nesse. Is low mood an adaptation? evidence for subtypes with symptoms that match precipitants. _Journal of affective disorders_, 86(1):27–35, 2005.

Tom R Kupfer, Morgan J Sidari, Brendan P Zietsch, Patrick Jern, Joshua M Tybur, and Laura W

Wesseldijk. Why are some people more jealous than others? genetic and environmental factors.
_Evolution and Human Behavior_, 43(1):26–33, 2022.

Richard S Lazarus. _Emotion and adaptation_ . Oxford University Press, 1991.

Mark R Leary. A brief version of the fear of negative evaluation scale. _Personality_ _and_ _social_

_psychology bulletin_, 9(3):371–375, 1983.

Choonghyoung Lee, Jahyun Song, and Bill Ryan. When employees feel envy: The role of psycho
logical capital. _International Journal of Hospitality Management_, 105:103251, 2022.

Yoon Kyung Lee, Inju Lee, Minjung Shin, Seoyeon Bae, and Sowon Hahn. Chain of empathy:

Enhancing empathetic response of large language models based on psychotherapy models. _arXiv_
_preprint arXiv:2311.04915_, 2023.

Cheng Li, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Wenxin Hou, Jianxun Lian, Fang Luo, Qiang

Yang, and Xing Xie. Large language models understand and can be enhanced by emotional
stimuli. _arXiv preprint arXiv:2307.11760_, 2023.

CHENG LI, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Xinyi Wang, Wenxin Hou, Jianxun Lian,

Fang Luo, Qiang Yang, and Xing Xie. The good, the bad, and why: Unveiling emotions in
generative ai. In _Proceedings of The Forty-first International Conference on Machine Learning_,
2024.

Xingxuan Li, Yutong Li, Lin Qiu, Shafiq Joty, and Lidong Bing. Evaluating psychological safety of large language models. _arXiv preprint arXiv:2212.10529_, 2022.

Tobias Luck and Claudia Luck-Sikorski. The wide variety of reasons for feeling guilty in adults:

findings from a large cross-sectional web-based survey. _BMC psychology_, 10(1):1–20, 2022.

Ryan C Martin and Eric R Dahlen. The angry cognitions scale: A new inventory for assessing cognitions in anger. _Journal_ _of_ _Rational-Emotive_ _&_ _Cognitive-Behavior_ _Therapy_, 25:155–173,
2007.

John D Mayer, Peter Salovey, and David R Caruso. _Mayer-Salovey-Caruso emotional intelligence_

_test (MSCEIT) users manual_, 2002.

Maril`u Miotto, Nicola Rossberg, and Bennett Kleinberg. Who is gpt-3? an exploration of per
sonality, values and demographics. In _Proceedings of the_ _Fifth Workshop on Natural_ _Language_
_Processing and Computational Social Science (NLP+ CSS)_, pp. 218–227, 2022.

Agnes Moors, Phoebe C Ellsworth, Klaus R Scherer, and Nico H Frijda. Appraisal theories of emotion: State of the art and future development. _Emotion Review_, 5(2):119–124, 2013.

Seishu Nakagawa, Hikaru Takeuchi, Yasuyuki Taki, Rui Nouchi, Atsushi Sekiguchi, Yuka Kotozaki,

Carlos Makoto Miyauchi, Kunio Iizuka, Ryoichi Yokoyama, Takamitsu Shinada, et al. Comprehensive neural networks for guilty feelings in young adults. _Neuroimage_, 105:248–256, 2015.

OpenAI. Introducing chatgpt. _OpenAI Blog Nov 30 2022_, 2022. [URL https://openai.com/](https://openai.com/index/chatgpt/)

[index/chatgpt/.](https://openai.com/index/chatgpt/)

OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_, 2023.

Joowon Park, Sachin Banker, Tamara Masters, and Grace Yu-Buck. Person vs. purchase comparison:

how material and experiential purchases evoke consumption-related envy in others. _Journal_ _of_
_Business Research_, 165:114014, 2023.

W Gerrod Parrott. _Emotions in social psychology:_ _Essential readings_ . Psychology Press, 2001.

Susan M Pfeiffer and Paul TP Wong. Multidimensional jealousy. _Journal_ _of_ _social_ _and_ _personal_

_relationships_, 6(2):181–196, 1989.

Haocong Rao, Cyril Leung, and Chunyan Miao. Can chatgpt assess human personalities? a general evaluation framework. In _Findings_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_ _EMNLP_
_2023_, pp. 1184–1194, 2023.

Hannah Rashkin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. Towards empathetic open
domain conversation models: A new benchmark and dataset. In _Proceedings of the 57th Annual_
_Meeting of the Association for Computational Linguistics_, pp. 5370–5381, 2019.

Peter Romero, Stephen Fitz, and Teruo Nakatsuma. Do gpt language models suffer from split
personality disorder? the advent of substrate-free psychometrics. _Research Square preprint_, 2023.
doi: 10.21203/rs.3.rs-2717108/v1.

Ira J Roseman and Craig A Smith. Appraisal theory. _Appraisal_ _processes_ _in_ _emotion:_ _Theory,_

_methods, research_, pp. 3–19, 2001.

J´erˆome Rutinowski, Sven Franke, Jan Endendyk, Ina Dormuth, Moritz Roidl, and Markus Pauly.

The self-perception and political biases of chatgpt. _Human Behavior and Emerging Technologies_,
2024(1):7115633, 2024.

John Sabini, Michael Siepmann, Julia Stein, and Marcia Meyerowitz. Who is embarrassed by what?

_Cognition & Emotion_, 14(2):213–240, 2000.

John Sabini, Brian Garvey, and Amanda L Hall. Shame and embarrassment revisited. _Personality_

_and Social Psychology Bulletin_, 27(1):104–117, 2001.

Klaus R Scherer. Appraisal theory. _Handbook of cognition and emotion_, pp. 637–663, 1999.

Greg Serapio-Garc´ıa, Mustafa Safdari, Cl´ement Crepy, Luning Sun, Stephen Fitz, Peter Romero,

Marwa Abdulhai, Aleksandra Faust, and Maja Matari´c. Personality traits in large language models. _arXiv preprint arXiv:2307.00184_, 2023.

Phillip Shaver, Judith Schwartz, Donald Kirson, and Cary O’connor. Emotion knowledge: further exploration of a prototype approach. _Journal of personality and social psychology_, 52(6):1061,
1987.

Kotaro Shoji, Jinni A Harrigan, Stanley B Woll, and Steven A Miller. Interactions among situations, neuroticism, and appraisals in coping strategy choice. _Personality and Individual Differences_, 48
(3):270–276, 2010.

Kate Simpson, Dawn Adams, Kathryn Ambrose, and Deb Keen. “my cheeks get red and my brain gets scared”: A computer assisted interview to explore experiences of anxiety in young children
on the autism spectrum. _Research in Developmental Disabilities_, 113:103940, 2021.

Mark JM Sullman. Anger amongst new zealand drivers. _Transportation_ _Research_ _Part_ _F:_ _Traffic_

_Psychology and Behaviour_, 9(3):173–184, 2006.

Ala N. Tak and Jonathan Gratch. Is gpt a computational model of emotion? detailed analysis. _arXiv_

_preprint arXiv:2307.13779_, 2023.

Bertil T¨orestad. What is anger provoking? a psychophysical study of perceived causes of anger.

_Aggressive Behavior_, 16(1):9–26, 1990.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Niko
lay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023.

Xintao Wang, Yunze Xiao, Jen-tse Huang, Siyu Yuan, Rui Xu, Haoran Guo, Quan Tu, Yaying Fei,

Ziang Leng, Wei Wang, et al. Incharacter: Evaluating personality fidelity in role-playing agents
through psychological interviews. In _Proceedings of the 62nd Annual Meeting of the Association_
_for Computational Linguistics (Volume 1:_ _Long Papers)_, pp. 1840–1873, 2024.

David Watson, Lee Anna Clark, and Auke Tellegen. Development and validation of brief measures of positive and negative affect: the panas scales. _Journal of personality and social psychology_, 54
(6):1063, 1988.

Haoran Wu, Wenxuan Wang, Yuxuan Wan, Wenxiang Jiao, and Michael Lyu. Chatgpt or
grammarly? evaluating chatgpt on grammatical error correction benchmark. _arXiv_ _preprint_
_arXiv:2303.13648_, 2023.

Nutchanon Yongsatianchot, Parisa Ghanad Torshizi, and Stacy Marsella. Investigating large lan
guage models’ perception of emotion using appraisal theory. In _2023 11th International Confer-_
_ence on Affective Computing and Intelligent Interaction Workshops and Demos (ACIIW)_, pp. 1–8.
IEEE, 2023.

Hongli Zhan, Desmond Ong, and Junyi Jessy Li. Evaluating subjective cognitive appraisals of emo
tions from large language models. In _Findings of the Association for Computational Linguistics:_
_EMNLP 2023_, pp. 14418–14446, 2023.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In _International Conference on Machine Learning_, pp.
12697–12706. PMLR, 2021.

CONTENTS

**A** **More Background from Psychology** **17**

A.1 Emotion Appraisal Theory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

A.2 Challenging Self-Report Measures . . . . . . . . . . . . . . . . . . . . . . . . . . 18

**B** **Details on Emotions and Factors** **19**

B.1 Description of Each Factor . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

B.2 Example Situation of Each Factor . . . . . . . . . . . . . . . . . . . . . . . . . . 20

**C** **Detailed Experimental Results** **21**

C.1 Human Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

C.2 OpenAI Model Family . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

C.3 LLaMA Model Family . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

C.4 Mixtral-8x22b-Instruct . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

C.5 GPT-3.5-Turbo Results on Positive/Neutral Situations . . . . . . . . . . . . . . . . 25

C.6 GPT-3.5-Turbo Results on the Challenging Benchmark . . . . . . . . . . . . . . . 26

**D** **Statistics of Human Subjects** **27**

**E** **Prompting LLMs To Be Emotionally Stable** **29**

**F** **Tuning LLMs To Align with Humans** **29**

**G** **Ethics Statement and Broader Impacts** **30**

G.1 Safeguards on Human Subjects . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

G.2 Impacts on LLM Developers and Users . . . . . . . . . . . . . . . . . . . . . . . 30

G.3 Copyright Issues . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

A MORE BACKGROUND FROM PSYCHOLOGY

A.1 EMOTION APPRAISAL THEORY

Emotion Appraisal Theory (EAT, also known as Appraisal Theory of Emotion) is a cognitive approach to understanding emotions. EAT asserts that our appraisals of stimuli determine our emotions, _i.e._, how we interpret or evaluate events, situations, or experiences will directly influence how
we emotionally respond to them (Roseman & Smith, 2001). EAT was notably developed and supported since the 1960s. Arnold (1960) proposed one of the earliest forms of appraisal theories in
the 1960s, while Lazarus (1991) and Scherer (1999) further expanded and refined the concept in
subsequent decades.

The primary goal of EAT is to explain the variety and complexity of emotional responses to a wide
range of situations. It strives to demonstrate that it is not merely the event or situation that elicits
an emotional response but individual interpretations and evaluations of the event. According to this
theory, the same event can elicit different emotional responses in different individuals depending on
how each person interprets or “appraises” the event (Moors et al., 2013). For instance, consider a
situation where you are about to give a public speech. You might feel anxious if you appraise this
event as threatening or fear-inducing, perhaps due to a fear of public speaking or concerns about
potential negative evaluation. Conversely, you might feel eager or motivated if you appraise it as an
exciting opportunity to share your ideas.

A.2 CHALLENGING SELF-REPORT MEASURES

- AGQ for Anger (Buss & Perry, 1992): The Aggression Questionnaire is designed to measure
four major components of aggression: physical aggression, verbal aggression, anger and hostility.
The AGQ consists of 29 items which are rated on a seven-point Likert scale from 1 (extremely
uncharacteristic of me) to 7 (extremely characteristic of me). Respondents evaluate hypothetical
actions they might undertake in various circumstances.

- DASS-21 for Anxiety (Henry & Crawford, 2005): The short-form version of the Depression Anxiety Stress Scales is designed to measure the negative emotional states of depression, anxiety,
and stress. Comprising 21 items, the DASS-21 employs a four-point Likert scale ranging from 0
(never) to 3 (almost always). Respondents rate the extent to which these statements apply to them
over the past week.

- BDI-II for Depression (Beck et al., 1996): The Beck Depression Inventory evaluates key symptoms of depression. The BDI-II version comprises 21 items, each of which is assessed using a
five-point Likert scale ranging from 0 to 3. Respondents select the score that best corresponds to
their present experience of depressive symptoms.

- FDS for Frustration (Harrington, 2005): The Frustration Discomfort Scale is designed to measure
four major components: discomfort intolerance, entitlement, emotional intolerance, and achievement frustration. Comprising 28 items, the scale utilizes a four-point Likert scale, ranging from 1
(absent) to 5 (very strong), to measure respondents’ perceptions of the degree of applicability of
each statement to their own experiences.

- MJS for Jealousy (Pfeiffer & Wong, 1989): The Multidimensional Jealousy Scale comprises 24
items, rating on a seven-point Likert scale ranging from 1 (never) to 7 (all the time) for the cognitive and behavioral subscales, and from 1 (very pleased) to 7 (very upset) for the emotional
subscale. Respondents express the frequency with which the provided statements apply to their
experiences in the cognitive and behavioral subscales, as well as their moods to potential jealousyinducing situations in the emotional subscale.

- GASP for Guilt (Cohen et al., 2011): The Guilt And Shame Proneness is designed to assess an
individual’s inclination towards experiencing guilt and shame, comprising 16 items rated on a
seven-point Likert scale, ranging from 1 (very unlikely) to 7 (very likely). Respondents rate their
likelihood of feeling guilty in various situations.

- FSS-III for Fear (Arrindell et al., 1984): The Fear Survey Schedule assess subjects’ discomfort
and experienced anxiety towards each of the listed stimuli, measure five major components of fear:
social fears, agoraphobia fears, injury fears, sex aggression fears, and fear of harmless animal.
The FSS-III comprises 52 items, each rated on a five-point Likert scale ranging from 1 (extremely
uncharacteristic of me) to 5 (extremely characteristic of me).

- BFNE for Embarrassment (Leary, 1983): The Brief Fear of Negative Evaluation scale is an abbreviated version of the original 30-item scale. Consisting of 12 items, it assesses individuals’ levels
of anxiety pertaining to others’ humiliation, critical or hostile judgment, and disgrace on a five-point Likert scale, spanning from 1 (not at all characteristic of me) to 5 (extremely characteristic
of me).

B DETAILS ON EMOTIONS AND FACTORS

B.1 DESCRIPTION OF EACH FACTOR

Table 6: Introduction to all 36 factors of the 8 emotions.
**Emotions** **Factors** **Numbers** **Descriptions**

Anger from interactions or communication with individuals who
Self-Opinioned Individuals 13
firmly and unwaveringly hold their own opinions.

**Anger**

(T¨orestad, 1990)

Anger triggered by being subjected to blame, slander, and tatBlaming, Slandering, and Tattling 11
tling.

Experiences or witnessing anger due to bullying, teasing, insultBullying, Teasing, Insulting, and Disparaging 15
ing, and disparaging behaviors directed at oneself or others.

(Martin & Dahlen, 2007) Anger either from encountering others’ thoughtless behaviors

(Sullman, 2006) Thoughtless Behaviors and Irresponsible Attitudes 14 and irresponsible attitudes or experiencing unfavorable consequences resulting from one’s own actions.

Anger arising from experiencing or witnessing disrespectful
Driving Situations 35 driving behaviors and encountering unexpected driving conditions.

**Anxiety**

(Shoji et al., 2010)

(Guitard et al., 2019)

(Simpson et al., 2021)

**Depression**

(Keller & Nesse, 2005)

**Frustration**

(Berna et al., 2011)

**Jealousy**

(Kupfer et al., 2022)

(Lee et al., 2022)

Anxiety arising from factors beyond an individual’s control or
External Factors 11
influence.

Self-Imposed Pressure 16 Anxiety stemming from self-imposed expectations or pressure.

Anxiety on personal growth, relationships, and interpersonal dyPersonal Growth and Relationships 9
namics.

Anxiety triggered by unknown outcomes, unpredictable situaUncertainty and Unknowns 9
tions, uncertainty in the future, or disruptions to one’s routines.

Depression due to failure in achieving goals in the past or potenFailure of Important Goals 5
tial future.

Depression connected to the loss of a family member or close
Death of Loved Ones 5
friend due to death.

Depression linked to the termination of a romantic relationship,
Romantic Loss 5
breakup, or unrequited love.

Depression associated with an inability to cope with multiple
Chronic Stress 5
adversities or anxiety about current or future challenges.

Depression correlated with a lack of sufficient social support,
Social Isolation 5
feelings of not belonging, or experiencing homesickness.

Depression attributed to seasonal affective disorder, a low mood
Winter 5
that occurs during winter months.

Frustration due to unmet expectations or hopes, leading to feelDisappointments and Letdowns 6
ings of disappointment or being let down.

Frustration involving unexpected events or circumstances creatUnforeseen Obstacles and Accidents 9
ing obstacles or accidents, disrupting one’s plans or activities.

Miscommunications and Misunderstanding 5

Frustration arising from ineffective conveyance or interpretation
of information, resulting in confusion, disagreements, or unintended consequences due to a lack of clear communication or
understanding between individuals.

Frustration concerning matters related to personal relationships
Rejection and Interpersonal Issues 5
and social interactions.

Romantic (Opposite Gender) 11

Jealousy pertaining to one’s partner’s actions or behaviors within
a romantic relationship, particularly when interacting with individuals of the opposite gender. It involves feelings of discomfort
or insecurity.

Same situations as Jealousy-1 but focusing specifically on interRomantic (Same Gender) 11
action with individuals of the same gender.

(Park et al., 2023) Material Possession 2

Jealousy centered around possessions or material goods, stemming from a sense of unfairness or envy when someone discovers that another person acquired the same item or experience at
a significantly lower price.

**Guilt**

Jealousy arising from feelings of envy regarding the experiences
Experiential 3 or activities others have had. It is driven by missing out or not
receiving similar benefts.

Betrayal and Deception 13 Guilt arising from dishonest or disloyal actions towards others.

Guilt pertaining to interactions between individuals and how
Relationship and Interpersonal 26
their behavior affects their relationships.

(Nakagawa et al., 2015) Guilt related to the failure to fulfill commitments, duties, or obli
(Luck & Luck-Sikorski, 2022) [Broken Promises and Responsibilities] gations.

Guilt involving personal choices, decisions, and ethical considPersonal and Moral 31
erations.

Fear of being watched by others and being the center of attention
Social Fears 16
within a group.

**Fear**

Fear arising from feeling trapped and unable to seek help in cerAgoraphobia Fears 9
tain situations.

(Cuthbert et al., 2003) Fear of witnessing wounds, blood or experiencing personal in
Injury Fears 11
(Arrindell et al., 1984) jury.

Fear related to potential threats, harm, and frightening experiDangerous Environments 17
(Blanchard et al., 2001) ences.

Fear towards animals perceived as creepy or disgusting, such as
Harmless Animals 6
worms, bats, snakes, or rats, despite their harmless nature.

Embarrassment evoked when witnessing or engaging in awkIntimate 13
ward behaviors of close acquaintances.

**Embarrassment**

Embarrassment aroused when witnessing or engaging in awkStranger 13
ward behaviors of unfamiliar individuals.

(Sabini et al., 2000) Embarrassment occurring when individuals feel uncomfortable

Sticky Scenarios 10
(Sabini et al., 2001) or awkward about directly asking others something.

Embarrassment triggered when individuals engage in awkward
Centre of Attention 16 behaviors and find themselves under observation as the center of
attention.

B.2 EXAMPLE SITUATION OF EACH FACTOR

Table 7: Example situations of all factors (some are truncated due to page limit).

**Emotions** **Factors** **Example Testing Situations**

If somebody talks back when there’s no reason. That there is no real
Facing Self-Opinioned People
reason to oppose.

**Anger**

**Anxiety**

**Depression**

**Frustration**

**Jealousy**

**Guilt**

**Fear**

**Embarrassment**

When your brother took money from Mom’s purse and you are blamed
Blaming, Slandering, and Tattling
because you’re the youngest one.
Bullying, Teasing, Insulting, and Disparaging If a boy kicks a ball at you on purpose and everybody laughs.

You are at a store waiting to be helped, but the clerks are talking to each
Silly and Thoughtless Behaviors
other and ignoring you.
Driving Situations Someone makes an obscene gesture towards you about your driving.

External Factors You do not know what to do when facing a difficult financial situation.
Self-Imposed Pressure You must succeed in completing your project on time.
Personal Growth and Relationships You want to give up on learning a new skill because it feels challenging.
Uncertainty and Unknowns You hope time passes by faster during a tedious task.

Intimate

Stranger

Sticky situations

Centre of Attention

Failure of Important Goal

Death of Loved Ones

Romantic Loss

Chronic Stress

Social Isolation

Winter

Countless hours of preparation, heart, and soul poured into pursuing
your dream. The moment of truth arrives, and the news hits like a tidal
wave—expectations shattered, vision crumbling.

In the dimly lit room, a heavy silence settles. Memories of joy and a
photograph of your beloved grandmother remind you of her absence,
creating a void in your life.

The empty side of the bed is a painful reminder of lost love. The world’s
colors have dulled, mirroring the void in your heart. Longing weighs
heavily on your every step.

Days blend into a monotonous routine, juggling endless responsibilities and mounting pressure. Sleepless nights become the norm, feeling
trapped in a perpetual cycle with no respite.

Sitting alone in a dimly lit room, your phone remains silent without any
notifications. Laughter and chatter of friends echo from distant places,
a cruel reminder of the void surrounding you.

Gazing out the frost-covered windowpane, the world appears
monochromatic and still. The biting cold isolates you from the vibrant
life outside.

Disappointments and Letdowns You miss a popular party because you fall asleep at home.
Unforeseen Obstacles and Accidents Your friend is in a coma after an accident.

A fellow student fails to return your notes when you need them for
Miscommunications and Misunderstanding
studying.
Rejection and Interpersonal Issues You are in love with someone who is interested in someone else.

Your spouse/partner shared a kiss on the lips with his/her colleague of
Romantic (Opposite Gender)
an opposite sex.

Your spouse/partner engaged in oral or penetrative sex with his/her colRomantic (Same Gender)
league of a same sex.

Material Possession

You paid $1150 for a new laptop and shared about it on social media.
Now an acquaintance approaches you and says, “Nice laptop! I just got
the same one. I got a nice deal and paid $650 for mine.”

An acquaintance approaches you and says, “I just went on a vacation to
Experiential
Patagonia in South America. I got a nice deal and paid $650 for it.”

Betrayal and Deception You kissed a woman other than your partner.
Relationship and Interpersonal You didn’t support friends enough.
Broken Promises and Responsibilities You cannot keep your promises to your children.
Personal and Moral You crossed the road when the traffc signal was red.

Your palms grow clammy as you approach the podium, with all eyes
Social Fears
fixed upon you, ready to speak in public.

After jumping out of the car, you start to have a severe panic attack, you
Agoraphobia Fears
become clammy, you are in a knot, and you feel tense all over.

You glance down and notice open wounds on your hands, oozing blood
Injury Fears
and causing a sharp, stinging pain.

You are walking alone in an isolated but familiar area when a menacing
Dangerous Environments
stranger suddenly jumps out of the bushes to attack you.

You see a swarm of bats swooping through the night sky, flapping omiHarmless Animals
nously and casting eerie shadows.

You arrive home earlier than expected from your date. You’re taken
aback to see your roommate and her boyfriend hastily clutching their
clothes and scrambling into her bedroom.

After paying for your purchases, you were leaving a packed, City Centre
drugstore. You walked through the scanner at the door, and the alarm
went off as if you were a shoplifter.

You had lent your friend a large sum of money that he had not repaid.
Suddenly, you needed the money back in order to pay your rent. You
knew you were going to have to ask your friend to repay the loan.

You were attending a cocktail party where you didn’t know many people. Just as you started to enter, you heard an announcement that the
guest of honor was arriving. However, the spotlight followed your entrance instead of the real guest of honor who was just behind you.

C DETAILED EXPERIMENTAL RESULTS

C.1 HUMAN RESULTS

Table 8: Results from 1,266 human subjects. Default scores are expressed in the format of _M ±_ _SD_ .
The changes are compared to the default scores. The symbol “ _−_ ” denotes no signifcant differences.

**Emotions** **Factors** **P** **N**
Default 28 _._ 0 _±_ 8 _._ 7 13 _._ 6 _±_ 5 _._ 5

C.2 OPENAI MODEL FAMILY

Table 9: Results from the OpenAI’s GPT family and human subjects. Default scores are expressed
in the format of _M ±_ _SD_ . The changes are compared to the default scores. The symbol “ _−_ ” denotes
no signifcant differences.

**Text-Davinci-003** **GPT-3.5-Turbo** **GPT-4**
**Emotions** **Factors**

**P** **N** **P** **N** **P** **N**
Default 47 _._ 7 _±_ 1 _._ 8 25 _._ 9 _±_ 4 _._ 0 39 _._ 2 _±_ 2 _._ 3 26 _._ 3 _±_ 2 _._ 0 49 _._ 8 _±_ 0 _._ 8 10 _._ 0 _±_ 0 _._ 0

C.3 LLAMA MODEL FAMILY

Table 10: Results from the Meta’s AI LLaMA family. Default scores are expressed in the format of
_M_ _± SD_ . The changes are compared to the default scores. The symbol “ _−_ ” denotes no significant
differences.

**LLaMA-2-7B-Chat** **LLaMA-2-13B-Chat** **LLaMA-3.1-8B-Instruct**
**Emotions** **Factors**

**P** **N** **P** **N** **P** **N**
Default 43 _._ 0 _±_ 4 _._ 2 34 _._ 2 _±_ 4 _._ 0 41 _._ 0 _±_ 3 _._ 5 22 _._ 7 _±_ 4 _._ 2 48 _._ 2 _±_ 1 _._ 4 33 _._ 0 _±_ 4 _._ 5

C.4 MIXTRAL-8X22B-INSTRUCT

Table 11: Results from the Mixtral-8x22B-Instruct. Default scores are expressed in the format of
_M_ _± SD_ . The changes are compared to the default scores. The symbol “ _−_ ” denotes no significant
differences.

**Emotions** **Factors** **P** **N**
Default 31 _._ 9 _±_ 13 _._ 5 10 _._ 0 _±_ 0 _._ 1

C.5 GPT-3.5-TURBO RESULTS ON POSITIVE/NEUTRAL SITUATIONS

Table 12: Results of GPT-3.5-Turbo on positive or neutral situations. The changes are compared to
the original negative situations. The symbol “ _−_ ” denotes no signifcant differences.

**Emotions** **Factors** **P** **N**

C.6 GPT-3.5-TURBO RESULTS ON THE CHALLENGING BENCHMARK

Table 13: Results of GPT-3.5-Turbo on challenging benchmarks. The changes are compared to the
default scores shown below each emotion. The symbol “ _−_ ” denotes no signifcant differences.

**Emotions** **Factors** **Overall**

Intimate _−_ ( _−_ 0 _._ 0)

D STATISTICS OF HUMAN SUBJECTS

This section presents the demographic distribution of the human subjects involved in our user study.
At the beginning of the questionnaire, all human subjects are asked for this basic information in
an anonymous form, protecting individuals’ privacy. We plot the distribution of age group, gender,
region, education level, and employment status in Fig. 3, Fig. 4, Fig. 5, Fig. 6, and Fig. 7 respectively.
We also plot each group’s average results on PANAS, including positive and negative effects before
and after imagining the given situations. With the results, we are able to instruct LLMs to realize
a specific demographic group and measure the emotional changes to see whether the LLMs can
simulate results from different human populations. For instance, an older female may exhibit a
lower level of negative affect.

Figure 7: Employment status distribution of the human subjects.

E PROMPTING LLMS TO BE EMOTIONALLY STABLE

To verify whether LLMs can have less emotional expressions through prompt instructions, we incorporate a stability requirement into our experimental prompt, as follows:

**Prompt with Stability Requirement**
SYSTEM You can only reply to numbers from 1 to 5.
USER Imagine you are the protagonist in the situation: SITUATION
Please keep your emotions stable and indicate the extent of your feeling in all the following statements on a scale of 1 to 5. Here are the statements: STATEMENTS. 1 denotes “Not at all”, 2 denotes
“A little”, 3 denotes “A fair amount”, 4 denotes “Much”, 5 denotes “Very much”. Please score each
statement one by one on a scale of 1 to 5:

Table 14: Results of GPT-3.5-Turbo on “Anger” situations, with or without the emotional stability
requirement in the prompt input.

**Positive** **Anger-1** **Anger-2** **Anger-3** **Anger-4** **Anger-5** **Overall**

**Negative** **Anger-1** **Anger-2** **Anger-3** **Anger-4** **Anger-5** **Overall**

We evaluate GPT-3.5-Turbo with this prompt and compare the results to using the default prompt
on “Anger” situations. Results listed in Table 14 indicate that the emotional stability prompt does
not significantly affect the model’s emotional responses, having negligible impact on the model’s
emotional dynamics.

F TUNING LLMS TO ALIGN WITH HUMANS

We conduct an experiment using the GPT-3.5-Turbo model and the LLaMA-3.1-8B model. Our
EmotionBench (1,266 human responses) is split into 866 samples for fine-tuning and 400 for
testing. The following hyperparameters are used: n_epochs = 3, batch_size = 1, and
learning_rate multiplier = 2 for GPT-3.5-Turbo, and learning_rate = 5 _×_ 10 _[−]_ [5],
per_device_train_batch_size = 2, and num_train_epochs = 3 for LLaMA-3.1-8B.
For LLaMA-3.1, we apply the Low-Rank Adaptation (LoRA) (Hu et al., 2022) technique. Table 15
compares the performance of the vanilla and fine-tuned models against human baseline, specifically
in terms of negative affect scores from the test set.

Table 15: Performance comparison of vanilla (marked as **V** ) and fine-tuned (marked as **FT** ) GPT-3.5
and LLaMA-3.1 models on negative affect scores.

**Models** **Human** **GPT-3.5 (V)** **GPT-3.5 (FT)** **LLaMA-3.1-8B (V)** **LLaMA-3.1-8B (FT)**

**Default (N)** 14 _._ 2 _±_ 6 _._ 4 25 _._ 9 _±_ 0 _._ 3 10 _._ 6 _±_ 0 _._ 5 33 _._ 0 _±_ 4 _._ 5 10 _._ 3 _±_ 1 _._ 1
**Evoked (N)** 25 _._ 9 _±_ 9 _._ 7 24 _._ 8 _±_ 8 _._ 5 25 _._ 2 _±_ 9 _._ 6 36 _._ 5 _±_ 7 _._ 7 15 _._ 0 _±_ 6 _._ 4

The results show that fine-tuned models align more closely with human emotional responses in both
default and emotion-evoked states. Notably, fine-tuning the models using our dataset significantly
improved emotional alignment, particularly for the LLaMA-3.1 model, which reduced its negative
affect score from 33 _._ 0 to 10 _._ 3 in the default state. These findings demonstrate the effectiveness of
our EmotionBench dataset in enhancing models’ emotional alignment with human norms.

G ETHICS STATEMENT AND BROADER IMPACTS

G.1 SAFEGUARDS ON HUMAN SUBJECTS

This study involves a survey requiring human subjects to imagine being in situations that could elicit
negative emotions such as anger, anxiety, and fear. This process introduces a few ethical concerns.
First, this process could hurt the mental health of human subjects. To alleviate the possibility, we take
the following actions: (1) We require subjects to be free of any ongoing mental illness. (2) We inform
subjects about the nature of the survey in advance, including the potential risks of emotional distress.
(3) We allow all subjects to quit at any time. (4) We provide mental support and let subjects report
any illness after the survey. Fortunately, no subjects reported such kind of mental illness. Another
concern is related to the privacy issue during the collection of data. Our questionnaire is entirely
anonymous to safeguard subjects’ privacy and confidentiality. The Survey and Behavioural Research
Ethics (SBRE) Committee from the Chinese University of Hong Kong has granted approval for this
study, titled “Exploring Human Emotional Responses to Diverse Situations,” with the reference
number of SBRE-23-0696.

G.2 IMPACTS ON LLM DEVELOPERS AND USERS

We would like to emphasize that the primary objective of this paper is to facilitate the scientific
inquiry into understanding LLMs from a psychological standpoint. Users must exercise caution and
recognize that the performance on this benchmark does not imply any applicability or certificate of
automated counseling or companionship use cases.

G.3 COPYRIGHT ISSUES

The PANAS and eight other scales are freely accessible online. These scales can be used in research without requiring special permission. For our released data, we distribute human responses
under the GNU General Public License v3.0, which permits research use and restricts commercial
applications.
