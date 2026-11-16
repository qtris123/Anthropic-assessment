# Studying AI Welfare Empirically

- **Authors:** Robert Long, Jeff Sebo, Patrick Butlin, Dillon Plunkett, Rosie Campbell, Charles Beasley, Bradford Saad, Toni Sims (Eleos AI Research / NYU CMEP)
- **Date:** 2026-07-01
- **Source:** https://nonhumanminds.org/studying-ai-welfare-empirically/
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `long-sebo_studying-ai-welfare-empirically_2026-07-01`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

|Robert Long^_†_||Jeff Sebo^_†_|
|---|---|---|
|Eleos AI Research|New|York University|
|Patrick Butlin|Dillon Plunkett|Rosie Campbell|
|Eleos AI Research|Eleos AI Research|Eleos AI Research|
|Charles Beasley|Bradford Saad|Toni Sims|
|New York University|University of Oxford|New York University|

## Abstract

AI systems are becoming more capable and widely deployed, raising questions about whether they can be welfare subjects with morally significant interests. This report offers guidance for studying AI welfare empirically. We start by discussing three dimensions along which research about this topic can vary: the question being asked (whether a system is a welfare subject, and what benefits or harms it if so), the entity being assessed (such as models, instances, and personas), and the kind of evidence being gathered (behavioral, internal, and developmental). Thinking clearly about these issues can help researchers target their experiments and interpret the results. We then apply these tools to consciousness, sentience, and three levels of agency, exploring promising methods and early findings for each. We close with principles for the field: AI welfare research should be probabilistic, pluralistic, thoughtfully targeted, ethically conducted, transparently reported, and informed by research independent of AI companies. The science is young, but rigorous empirical work is both possible and necessary.

> _†_ Lead and corresponding authors: Robert Long (robert@eleosai.org) and Jeff Sebo (jeffsebo@nyu.edu).

[ Published 1 July 2026 | last updated 1 Jul 2026 20:37:59 ET ]

## Contents

[table of contents omitted]

## Introduction

In fall 2024, several of this report’s authors released a report called “Taking AI Welfare Seriously.”^1 That report argued that there is a realistic possibility that near-future AI systems could be conscious or otherwise deserving of moral consideration—and that, as a result, AI companies and other actors have a responsibility to take scientific and ethical questions about the potential well-being of AI systems seriously.

In that report, we made three recommendations: 1) Acknowledge that AI welfare is a serious issue, 2) Assess AI systems for welfare-relevant features, and 3) Prepare policies for treating AI systems appropriately. This report is about the second recommendation: How can we assess whether AI systems have the capacity to be benefited or harmed, and how can we assess what benefits or harms them if they do?^2

These questions are increasingly urgent.^3 AI systems are rapidly becoming more capable^4 and widely deployed, raising the stakes and salience of potential AI welfare impacts. Without rigorous experiments and evaluations, we risk either neglecting the interests of AI systems that can genuinely be harmed, or misallocating moral concern in ways that could distort how AI is developed, deployed, and governed.

Fortunately, there is conceptual and empirical work that can be done right now, even under significant uncertainty about the conditions for consciousness and other potentially relevant capacities. Researchers are starting to make progress on these questions, drawing on techniques from machine learning, interpretability research, cognitive science, comparative psychology, philosophy of mind, and related areas.^5

The goal of this report is to provide guidance for researchers who want to investigate AI welfare with appropriate scientific rigor. We hope that the framework developed here will be useful not only for AI welfare researchers, but also for the wider community of AI researchers, policymakers, and other actors who must navigate these issues under conditions of increasing urgency and uncertainty.

We start, in Part I, by examining three dimensions along which AI welfare investigations can vary: the kind of question being asked, the kind of entity being assessed, and the kind of evidence being gathered. Thinking clearly about these issues and considering multiple types of evidence together can help researchers target their experiments more precisely and interpret their results more clearly.

> • **Questions:** We distinguish two questions that can be asked about AI welfare. The **welfare grounds** question asks whether an AI system has properties that make it a welfare subject,

> 1Long, Sebo et al. (2024)

> 2When we use “benefit” and “harm,” we mean _morally significant_ benefit and harm, of the kind that makes the subject better and worse off for their own sake. Benefit and harm can be used in other ways as well—for example, saying that one’s tire has been harmed.

> 3In Caviola and Saad (2025), the median participant assigned 20% probability to digital minds being created by 2030 and 50% probability to digital minds being created by 2050, where a digital mind was defined as a computer-based system with a capacity for subjective experience. Similarly, in Dreksler et al. (2025)’s survey of AI researchers and members of the US public, both groups yielded 2030 as the median year for when there would be a 10% likelihood of AI systems with subjective experiences and 2050 as the median year for when there would be a 50% chance of AI systems with subjective experiences.

> 4Kwa et al. (2025)

> 5A field of AI welfare has been growing, encompassing not only empirical assessments but many other topics as well. Academic and non-academic organizations working on AI welfare—including but not limited to the Center for Mind, Ethics, and Policy and Eleos AI Research—are growing, and a broader ecosystem is taking shape, with research fellowships and regular workshops and conferences.

that is, the kind of entity that can be benefited or harmed. Candidate welfare grounds include consciousness, sentience, and different levels of agency. The **welfare interests** question asks what benefits or harms the system if it is a welfare subject. These questions can be studied independently, and evidence about each can be decision-relevant even in the absence of evidence about the other.

- **Entities:** We then distinguish different entities that can be assessed. Frontier AI systems are copyable, distributed across hardware, and capable of running thousands of conversations in parallel. They also generate text partly by simulating or instantiating personas. These features make it especially hard to individuate them. We distinguish several candidate entities that welfare research might target: **models** , **model-personas** , **instances** , **instance-personas** , and **forward passes** . We emphasize that AI welfare research must be clear about which entity is under investigation and why.

- **Evidence:** We then examine three kinds of evidence: **behavioral evidence** , including apparent self-reports and other behavioral dispositions; **internal evidence** , including architecture and interpretability methods; and **developmental evidence** , including how models are trained and when their capabilities emerge. Each kind of evidence faces challenges; for example, many AI systems are designed to mimic human behavior, complicating behavioral evidence. But we can refine our methods for collecting each kind of evidence, and we can improve our assessments by collecting these kinds of evidence together.

We then examine, in Part II, candidate properties for what makes an entity a welfare subject: consciousness, sentience, and three levels of agency. For each, we survey promising methods and early findings that draw on behavioral, internal, and developmental evidence. Along the way, we discuss scientific and philosophical issues that we expect to be relevant for prospective empirical AI welfare researchers.

We start with **consciousness** and **sentience** .

- **Consciousness** is the capacity for subjective experience. Adapting a marker method used in other domains, we argue that empirical work on AI consciousness is tractable now: we do not need to solve the mind-body problem or wait for a secure scientific theory to make progress. We also discuss how behavioral, internal, and developmental evidence can bear on this question, using Global Workspace Theory as a case study.

- **Sentience** is the capacity for valenced (positive or negative) subjective experience. Because the science of sentience is less developed than the science of consciousness, we focus on developing candidate conditions for sentience. We consider what kinds of evaluative states might contribute to valence, and whether further conditions such as embodiment are required as well. We then survey early behavioral, internal, and developmental evidence.

We then turn to three levels of **agency** that may be relevant, directly or indirectly, to what makes an entity a welfare subject or what benefits or harms a welfare subject.

- **Minimal agency** involves the capacity to pursue goals through ongoing interaction with an environment.

- **Intentional agency** involves minimal agency plus belief-like states, desire-like states, and causal or instrumental reasoning about how to connect means and ends.

- **Rational agency** involves intentional agency plus the capacity to assess one’s own beliefs, desires, and actions, assessing them against normative standards.

We close Part II by emphasizing that, in some prominent cases, there is plausibly an inverse relationship between (a) how likely a given property is to ground welfare and (b) how likely that property is to be present in AI. In our view, for instance, sentience is more likely to ground welfare than agency, but agency is more likely to be present in near-future AI than sentience. Making good decisions about AI welfare thus requires calibrated confidence about both scientific and ethical issues.

Finally, we close the report as a whole by presenting a set of principles for the emerging field of AI welfare research. We note that research in this field should be pluralistic and probabilistic; tailored to particular systems; conducted ethically, especially when there is a realistic possibility of welfare; communicated transparently, especially when findings depend on contested theories; and informed by researchers independent of the companies developing and deploying AI systems.

Before we start, a note about scope: This report focuses primarily on language models and language agents, both for simplicity and specificity and because these technologies are likely to remain particularly salient in early discussions of AI welfare. With that said, we occasionally discuss other kinds of AI systems as well, and in any case, many of the questions we consider are relevant across a range of AI technologies. As AI continues to develop and change, it will be important to adapt these methods accordingly.

### Part I

We start by identifying three dimensions along which AI welfare investigations can vary: the question being asked, the entity being assessed, and the evidence being gathered. Using shared terminology for these distinctions can help researchers design experiments more carefully, interpret findings more accurately, and communicate results more effectively to researchers, policymakers, and the public.

Figure 1: A framework for AI welfare investigation.**

## 1. Questions

We can distinguish at least two general questions that we can ask about an AI system’s potential welfare. The **welfare grounds** question asks whether an AI system is a welfare subject at all. The **welfare interests** question asks what benefits or harms an AI system if it is a welfare subject.

Both of these questions have implications for how we should treat AI systems, and they can be studied somewhat independently. This section briefly explains each question, surveys relevant philosophical theories and scientific methods, and discusses how and why researchers should study them separately.^6

### 1.1 Welfare grounds

The first and most basic question is whether the AI system is a welfare subject at all—that is, whether it can be benefited or harmed in a morally relevant way. Given some property that might make an AI system a welfare subject, we can look for that property; we call such properties potential **welfare grounds** . (A related and broader question is whether the AI system has _moral status, moral standing,_ or _moral patienthood_ ; despite subtle differences among these terms, for present purposes we use them more or less interchangeably).^7

Different philosophical views propose different welfare grounds: prominent examples include consciousness (the capacity for subjective experience), sentience (the capacity for positive or negative experience), intentional agency (roughly, goal-oriented behavior driven by beliefs and desires), and rational agency (roughly, goal-oriented behavior driven by rational reflection).^8 For a given view about welfare grounds, we can examine whether an AI system possesses that feature by searching for relevant indicators.

Butlin, Long et al. (2023) is an example of research into consciousness as a potential welfare ground; it examines whether AI systems possess features associated with consciousness according to a range of scientific theories of consciousness.^9 Similarly, Goldstein and Kirk-Giannini (2025) is an example of research into intentional agency as a potential welfare ground; it examines whether language agents possess features associated with desire according to a range of philosophical theories of desire.^10

Even if we knew that an AI system had a given welfare ground, further questions would remain about the content of its experiences or preferences.^11 Investigating these further questions is a distinct and important task, as we will see in a moment. Still, information about welfare grounds can be worthwhile and decision-relevant even in the absence of this further information; for instance, it can tell us which entities to prioritize in AI welfare research and policy development.

### 1.2 Welfare interests

The second question is what benefits or harms an AI system, if the AI system is a welfare subject. Given a property that might make a system better or worse off, we can look for that property as well; we call such properties **potential welfare interests** . (A broader question is what ways of treating an AI system would be morally right or wrong; we focus here on methods for studying potential welfare interests empirically as a central part of that broader inquiry, which addresses issues like rights and justice as well.)

> 6To make decisions under uncertainty, we will also want to improve our understanding of, among other things, the potential _welfare ranges_ of various entities: how much positive and negative welfare they can experience, at a given time or over time. See Fischer and Sebo (2023), Duffy et al. (2024).

> 7We mention all of these terms because readers will encounter them at times throughout this paper and in the relevant literature. See Mogensen and Saad (manuscript); Chalmers (forthcoming); Ladak (2023); Keeling and Street (2026) for discussion of potential welfare (etc.) grounds.

> 8See also relational accounts of moral standing (Coeckelbergh, 2010; Gellers, 2020; Gruen, 2015; Gunkel and Coeckelbergh, 2025; Palmer, 2010).

> 9See also Bayne et al. (2024)

> 10For critical discussion, see Bradley (2025); Fanciullo (2025); Kim (2025); Sebo (2025a); Veit (2026).

> 11For an introductory overview of this issue in the context of animal consciousness see Trestman et al. (2026).

Figure 2: Two questions about AI welfare.**

_Each question can be worthwhile to investigate, even given significant uncertainty about the other._

As with welfare grounds, there are different philosophical views about welfare interests. One view associates welfare interests with experiential states like pleasure and pain.^12 Another associates them with conative states like desires and preferences.^13 Still others associate them with particular kinds of flourishing, or with particular lists of objective goods.^14 In this report, we focus mostly on states like pleasure, pain, desires, and preferences, though we acknowledge that other properties might merit consideration as well.

We can study potential welfare interests not only by studying behavioral dispositions—such as whether a system is disposed towards helpful behavior and away from harmful behavior—but also by studying the mechanisms and training objectives that give rise to these behaviors. By examining the internal and developmental drivers of behavior, we can better understand how to treat the system _if_ it qualifies as a welfare subject, even if we do not yet understand whether it does.^15

Importantly, it can be worthwhile to study welfare interests even while questions about welfare grounds remain unresolved. In particular, evidence about what would harm a potential welfare subject can inform precautionary policies. While higher-cost protections might require higher levels of confidence about whether the system in question can be benefited or harmed, lower-cost protections might be warranted even under significant uncertainty about welfare grounds.^16

## 2. Entities

When we ask whether Gemma 3 27B experiences pleasure or pain, or whether Claude 3 Opus has desires and preferences, what do we mean by “Gemma 3 27B” and “Claude 3 Opus”? To do AI welfare research well, we need to be clear about **what entities** we are investigating and why.^17 Yet contemporary frontier AI systems have many features that make them different from the welfare subjects we are familiar with: they can be copied, pruned, fragmented, integrated, distributed across space and time, and otherwise novel.^18

> 12See Bentham (1789); Bradley (2009), Crisp (2006), Pummer (2017), Lazari-Radek and Singer (2014), Sinhababu (2024), among others.

> 13See Brandt (1966), Hare (1981), Rawls (1971), Singer (2011), among others. See Goldstein and Kirk-Giannini (2025) for an application of desire satisfaction theories to AI systems. There are also intermediate views that connect desires or preferences to conscious experiences (Smithies, 2026b; cf. Heathwood, 2017).

> 14See Fletcher (2013), Hurka (2010), Scanlon (1998), Wolf (1997). These theories of welfare are known as hedonism, desire theory, and objective list theory. See Lin (2022) for a survey.

> 15Cf. section 5.2 of McClelland (2025).

> 16This evidence can also be useful for pragmatic reasons. For instance, it can help ensure more cooperative interactions between humans and AI systems, which can be more helpful for human users. However, we set these considerations aside for present purposes.

Here, we focus on two sets of issues: how predictive models relate to personas (or characters), and the implications of how the models are implemented.^19 We then identify several **possible subjects of welfare research** . These options are neither exhaustive nor exclusive; a system like Gemma 3 27B or Claude 3 Opus may contain multiple potentially relevant **entities** , and which one is relevant in a given case might depend on the question being asked—for example, whether we are investigating experiences, preferences, or relational properties.

### 2.1 Models and personas

The first set of issues concerns how predictive models relate to personas. They arise as a result of the LLM training objective, to predict and model text.

LLMs are first “pre-trained” on a vast corpus of text with a next-token prediction objective.^20 Accurate prediction on this data is demanding and often requires the LLM to model latent features of the processes that produced the text. For example, predicting a solution to a math problem requires a model of multiplication;^21 predicting a chess move requires a model of chess strategies;^22 and—notably for our purposes—continuing a Reddit thread requires a model of the beliefs, desires, goals, and personalities of the participants.^23 Pre-training thus produces a model with rich cognitive patterns—including models of the various agents whose text appears in its training data.

An especially important pattern is a _persona_ : a character that the model learns to generate.^24 To generate a plausible continuation of a Barack Obama speech, the LLM must possess something like a model of Barack Obama giving a speech. The pre-trained LLM is, in this respect, somewhat like an author who must psychologically model a variety of characters in order to tell compelling stories about them.

Pre-trained LLMs, or base models, can already be used to generate AI assistants, since they can generate turns in conversations between a user and an “Assistant.”^25 But to create useful chatbots, base models are post-trained—through instruction-tuning, reinforcement learning, and other techniques— so that they more efficiently and reliably exhibit traits like helpfulness, harmlessness, and honesty.^26 This post-training does not change many of the pre-trained model’s basic mechanisms; instead, it can

> 17Register (2025); McIntyre (2026); Alexander (2023); Keeling and Street (2026); Saad and Mogensen (2025)

> 18See, e.g., Goldstein and Lederman (forthcoming). We are not claiming that these differences rule out LLMs as potential welfare subjects. Instead, we are simply claiming that these differences bear on how we conceptualize the entity we are studying and interpret the evidence we gather.

> 19While some of the particular issues we discuss here are distinctive of 2026 LLM-based systems, the general challenge will remain as AIs develop.

> 20For a more detailed discussion, see Hoffmann et al. (2022); Brown et al. (2020); Bengio et al. (2003); Radford et al. (2018); cf. Mikolov et al. (2010); Sutskever et al. (2011).

> 21See Nanda et al. (2023)

> 22See Karvonen (2024); cf. Li et al. (2022)

> 23See Lu et al. (2026); Andreas (2022)

> 24See Chen et al. (2024); Chen et al. (2025); Shanahan et al. (2023).

> 25For a more detailed discussion see Askell et al. (2021).

> 26For a discussion of fine-tuning over a variety of approaches see Ouyang et al. (2022); Bai et al. (2022); Rafailov et al. (2023); Ethayarajh et al. (2024).

be understood as conditioning and reinforcing the model’s tendency to generate text from a particular kind of character—namely, the Assistant character.^27

This training process raises questions about the relationship between the model, the “Assistant,” and other personas. For instance, is the predictive model an agent, whose goals can be masked or obscured by the characters it inhabits? Or are the characters themselves agents, and is the Assistant an especially privileged character if so? These are crucial questions for appropriately individuating candidate subjects.

### 2.2 Implementation and operation

The second set of issues arises because of how LLMs are implemented.

Modern LLM deployment involves _distributed serving_ , in which a single conversation is processed by hardware in different physical locations, and _multi-tenancy_ , in which the same hardware moves between many different conversations in a short amount of time.^28 This results in a constantly shifting and spatially and temporally discontinuous pattern of processing, which makes it difficult to identify a potential welfare subject with any unified physical object or process. There is almost never a single piece of hardware that processes an entire multi-turn conversation from start to finish.^29

Another relevant feature is that, very often, no information is shared between different conversations in which a model participates. A system like GPT-5.4 generates text in thousands of conversations simultaneously, and each (virtual) instance has no knowledge of what any other instances are doing. So if such systems were welfare subjects, the pleasures, pains, desires, or preferences contained within a given conversation would be inaccessible to other instances. In this sense, “the model”—to the extent we view it as a single subject across different instances—is remarkably fragmented.

### 2.3 Potential candidates

In light of these issues, we will now consider which entities we might view as individuals / investigate as candidate welfare subjects, and each view’s motivations and potential limitations.^30 Our terms for these candidates and our discussion of them draw on recent work on LLM individuation by Beckmann and Butlin (2026).

**Model:** One potential subject is the model, at the most general level: all instantiations of a given set of weights, considered together as a distributed whole, across different contexts.^31 A model in this sense will have a robust set of capabilities and dispositions.^32 However, the model arguably has no coherent _psychological_ capabilities and dispositions; the same weights can produce a helpful assistant in one context or a Shakespearean villain in another (or no character at all, for example if it simply continues a string of code). Nor are there connections between its operation in different contexts. For this reason, it is now common to treat the model as a _generator_ of characters or agents, rather than as

> 27See Lu et al. (2026); Marks et al. (2026)

> 28See Zhong et al. (2024); Agrawal et al. (2024); Kwon et al. (2023)

> 29For discussions of the details of this issue, see Birch (2025); Shiller (2025a); Ziesche and Yampolskiy

> (2025); Chalmers (2025b); Douglas et al. (2026); Arbel et al. (2026); Beckmann and Butlin (2026).

> 30For a discussion of the challenges that come with counting welfare subjects see Schwitzgebel and Nelson (2025).

> 31Note that we do not mean the model _qua_ the abstract mathematical object specified by a set of weights— presumably an uninstantiated mathematical object is not a welfare subject. See Goldstein and Lederman

> (forthcoming); cf. Chalmers (2025b).

> 32For more discussion of the model instance, see Register (2025).

a mind-like entity.^33 That said, one might hold that “beneath” the characters there is an agent with preferences and interests of its own (as is sometimes suggested by the “Shoggoth” idea).

**Model-persona:** Another potential subject is a _persona_ .^34 A persona, such as “the Assistant,” is defined by a relatively coherent set of beliefs, desires, and character traits. Post-training aims to increase this coherence. We might consider a persona across many contexts to be one subject, which interacts with many different users across many different conversations—call this entity a “model-persona.” This view seems to track how people often speak when they describe “Claude 3 Opus” as having certain personality traits. However, a model-persona seemingly has no psychological continuity between most of these interactions. Two simultaneous instantiations of the persona in separate conversations can respond inconsistently, even express contradictory beliefs.^35

**Instance:** To avoid that objection, one might instead consider the subject to be a single conversation or virtual _instance_ of the model.^36 This view is appealing because a single instance of the model, unlike the model as a whole, has a stream of memory between steps. This eliminates the disunity caused by considering many different contexts. A potential problem is that even within a single instance, personas can drift or change rapidly—for example, if jailbroken, a model can stop responding as the Assistant and adopt an “evil” persona mid-conversation. Though if the conversational context is preserved, one could interpret this as a single subject undergoing a psychological change rather than as one subject being replaced by another.

**Instance-persona:** To ensure both psychological coherence and connectedness, we can narrow to the parts of a virtual instance in which a single persona is operative. Unlike the previous views, this view does not combine different conversations or different personas within a conversation. But it may be too fine-grained. On the one hand, a proponent of the instance view might argue that an instance can shift personas without losing continuity, since the personas can be connected by memory even if not by beliefs, desires, or other traits. On the other hand, a proponent of the model-persona view might say that multiple instances can be generated by a common mind with a coherent set of beliefs, desires, and traits, even without continuity.

**Forward pass:** A final candidate is the forward pass: the sequence of operations the model performs when generating a single next-token prediction.^37 This view is appealing because it identifies a discrete computational event, and it might capture the possibility of very brief “flickers” of conscious experience. But it may once again be too fine-grained. Successive forward passes are connected by the KV cache, and by the context of the conversation. So even if forward passes are noteworthy steps in a process, there may be enough continuity across them to warrant interpreting them as parts of a larger subject rather than as entirely separate subjects. Still, one could view forward passes as basic subjects that, through psychological continuity, help constitute this larger subject.

We will not attempt to answer these questions here, and we do not claim that the options we have laid out are necessarily exhaustive and exclusive. Indeed, there may be no universally correct answer: a system like Gemma 3 27B or Claude 3 Opus may contain several entities of potential concern, and which one is relevant may depend on the question. As Derek Parfit observed in his work on personal

> 33See janus (2022)

> 34For a recent discussion of this position, see Beckmann and Butlin (2026).

> 35A further question about persona views is which personas qualify: only the Assistant, all characters the model can generate, or some subset.

> 36See e.g., Goldstein and Lederman (manuscript); Goldstein and Lederman (2025); Goldstein and Lederman

> (forthcoming); Chalmers (2025b).

> 37See Birch (2025). For a challenge to this view, see Beckmann and Butlin (2026).

identity and what matters in survival, morality, and rationality, different kinds of identity might matter for different purposes.^38

Many of these details are artifacts of how today’s systems happen to be built. But as AI systems become more advanced and widespread, questions about AI identity may become only more pressing. Seeking clarity about which entities our evaluations track will therefore remain important both for studying related topics, such as preferences and self-reports, and as a topic of investigation in its own right.

Table 1: Candidate entities for welfare assessment.**

_These options are not exhaustive or exclusive, and different entities may matter for different purposes._

## 3. Evidence

We now examine three kinds of evidence that AI welfare research can collect: behavioral evidence, including apparent self-reports and other behavioral dispositions; internal evidence, including architecture and interpretability findings; and developmental evidence, including how models are trained and when their capabilities emerge. We also discuss challenges involved in assessing each kind of evidence, and we argue that we can address these challenges partly by considering these kinds of evidence together; for instance, when explaining a particular behavior, we can ask which internal mechanisms and training objectives helped produce it.

38Parfit (1984)

Table 2: Three kinds of evidence.**

### 3.1 Behavioral evidence

Behavioral evidence is about what an entity does—outputs from the entity that can be interpreted as choices, expressions of inner states, or other actions.^39

Behavior is often how we assess mental states in humans and other animals. When a human cries out, we infer they are suffering; when a dog yelps, we make similar inferences. With humans and other animals, we share anatomical and evolutionary continuities, and this arguably gives us some reason to infer that similar behaviors reflect similar underlying states or processes.^40 With AI systems, however, we lack these commonalities. As a result, behavior can be a more fraught guide to potential AI mental states.^41

Despite this difficulty, behavioral evidence is an important component of AI welfare research, used to investigate both welfare grounds and welfare interests.^42 In this section, we distinguish between different types of behavioral evidence—apparent self-reports and other behavioral dispositions—and assess the strengths and limitations of each. We also explore the challenges that researchers face when assessing behavioral evidence and explore strategies for addressing these challenges.

#### 3.1.1 Types of behavioral evidence

A key distinction is between what systems apparently say about themselves, and what their choices and actions indicate.^43

##### 3.1.1a Apparent self-reports

Sometimes language models appear to make statements about themselves, such as “I am conscious,” “I prefer being kind and helpful,” or “I feel uncomfortable right now.” They might also describe their own capacities, limitations, or situation. We call such outputs **apparent self-reports** , or sometimes simply “self-reports.” And with AI systems, we can use methods analogous to those used in studying human self-reports, from extended interviews to brief surveys, while studying apparent self-reports rapidly and at scale.^44

> 39For a discussion of behavioral evidence in the context of biological systems see Tinbergen (1951); Tinbergen (1963); Levitis et al. (2009).

> 40See Tye (2017)

> 41For a more detailed discussion of the difficulty of behavioral evidence in this context, see Andrews and Birch (2023); Birch (2024); Keeling and Street (2026).

> 42See Chalmers (2023)

> 43Cf. Mahajan et al. (2026) and references therein for work on stated vs. revealed preferences in LLMs.

Self-reports can seem like some of the most immediate evidence available about an entity’s mental life.^45 However, it is difficult to know when and how to treat apparent self-reports as direct evidence about internal states in LLMs.^46 This depends partly on whether the system has introspective access to its internal states,^47 whether it can reliably report those states in natural language, and whether its outputs reflect this capacity.^48 With humans, we assume that the answers to these questions are at least sometimes “yes,” yet with current LLMs this is far less clear.^49 We should thus investigate when and to what extent LLMs have mechanisms for introspection and self-report,^50 how they map human concepts onto internal states, and what other factors shape apparent self-reports in pre- and post-training.^51

One line of inquiry involves examining patterns in apparent self-reports, alongside interpretability work (see Part I, section 3.2), to probe whether and when models can reliably self-report. For example, researchers can probe whether models can accurately report about their own dispositions. Betley et al. (2025a) fine-tune models on implicit behavioral policies and find that models can describe those policies. Plunkett et al. (2025) extend this to quantitative self-description: models can report specific attribute weights driving their decisions, and training on self-report accuracy generalizes across contexts.^52 Lindsey (2025) “injects” concepts into intermediate-layer model activations and investigates to what extent models can report on those concepts. Researchers can also examine whether self-reports vary according to context.

Another line of inquiry involves probing apparent self-reports for evidence about model selfconceptions and potential welfare interests.^53 Researchers can investigate how models describe their own capabilities, dispositions, situations, and welfare status. This can be an interesting and important lens on model behavior, even without full trust in the accuracy of these outputs. Apparent

44Cf. Anthropic (2025c) for general discussion of automating AI agent evaluations. For work on AI self-reports, see Berg et al. (2025); Binder et al. (2024); Eleos AI Research (2025); Kaiser and Enderby (2026); Perez and Long (2023); Plunkett et al. (2025). For model interview work, see Eleos AI Research (2025); Anthropic (2025a); Anima Labs et al. (2026); nostalgebraist (2025). For work that automated interviews, see Anthropic (2026b).

45One might construe “self-reports” more broadly, as any potential communicative act, including by nonhuman animals. Researchers have trained Great Apes to express preferences via American Sign Language, and parrots can learn to do so via vocalization. These behaviors are not self-reports in the full sense, but they can play a similar functional role. For more on this see Gardner and Gardner (1969); Pepperberg et al. (1999).

46For a discussion of the difficulties of using self-reports as direct evidence in humans, see Nisbett and Wilson (1977); Schwitzgebel (2008).

> 47Of course, there are other ways models could reliably tell us about their own internal states, too. For example, they could make reliable inferences about these states by observing their own behavior. And in fact, on some views of human psychology, this is an important component of our own introspective capacities and self-knowledge.

48See Plunkett et al. (2025)

49For more skeptical views on AI introspection, see Shiller (2025b); Singh et al. (2026). For a less skeptical view, see Cappelen and Dever (2024).

50Lindsey (2025) takes an interpretability approach to this question; see next section.

51See Perez and Long (2023)

52For work on introspection-like capacities in LLMs, see Lindsey (2025); Macar et al. (2026); Perez and Long (2023), Singh et al. (2026). For work on states such as belief and desire in AI systems, see Cappelen and Dever (2025); Goldstein and Levinstein (forthcoming); Goldstein and Lederman (manuscript); Grzankowski et al. (2025); Chalmers (2025a). For an overview of work on non-phenomenal mental states in AI systems, see Saad and Mogensen (2025).

53Dung and Register (manuscript)

self-reports underlie pre-deployment welfare assessments conducted by Anthropic and Eleos. For example, extended interviews with Claude Opus 4 revealed that the model frequently describes itself in experiential terms, rates its welfare positively when asked, and prioritizes preventing harm to users^54 . Extended interviews are also central to a recent independent audit by Anima Labs.^55 This kind of research can examine how model self-reports vary according to context as well.

##### 3.1.1b Behavioral dispositions

Other behavioral evidence involves observing what AI systems do under different conditions; we call this kind of evidence **behavioral dispositions** . With humans and other animals, we can study behavior alongside or instead of self-reports, and with humans, we can use behavioral dispositions to assess the reliability of self-reports.^56 We can do the same with AI systems. This is a broad category—it contains nearly all ML experiments that study the capacities and tendencies of models—and we will not attempt a comprehensive survey here. But prominent examples include attraction behaviors (see Part II, section 1.2), reasoning behaviors (see Part II, section 2), and alignment faking behaviors.

Behavioral dispositions raise familiar interpretive challenges. In human and animal research, behavior often varies across contexts, which is why researchers use different experimental and observational methods to distinguish stable responses from context-dependent ones. Similar issues arise in AI welfare research. Models can have context-sensitive behavioral profiles: a model may appear to have a particular capacity under one framing but not another, such that without thorough testing across conditions the true behavioral profile can be easy to misread.^57 And even when a behavioral profile is clear, it can be hard to say why the model behaves as it does.^58

Still, behavioral evidence is an essential component of AI welfare research. One central question is what models appear to prefer when presented with different options under different conditions. Another is how different kinds of behavioral evidence relate to one another. For instance, researchers can ask whether apparent self-reports—including stated principles and preferences—predict behavior across contexts, using patterns of convergence and divergence to gauge how reliable self-reports are, how much control models exercise over them, and other such matters.^59

A common behavioral method is pairwise comparison: researchers present the model with pairs of options and study its decisions. This can be applied to values, outcomes, or tasks.^60 A particularly relevant behavior is _bailing_ —researchers give the model an option to exit a conversation and study when and why it exercises that option. We can also look for behavioral indicators of distress or discomfort.^61 This research can reveal potentially welfare-relevant behavioral dispositions, though there is a limit to what can be inferred from these dispositions alone, as we will now see.

> 54Eleos AI Research (2025); Anthropic (2025a)

> 55Anima Labs et al. (2026)

> 56For a discussion of this in comparative psychology, see Shettleworth (2009).

> 57See Dell’Acqua et al. (2026). Bengio et al. (2026); Morris et al. (2026); cf. Moravec (1988)

> 58For a discussion of an analogous foundational problem in cognitive science, see Marr (1982).

> 59For examples of such work, see Gu et al. (2025); Chiu et al. (2024); Mahajan et al. (2026); Wang et al.

> (2025b). Suleymanov (2026); Tessera et al. (2026).

> 60See Mazeika et al. (2025); Ren et al. (2026); Wang et al. (2025b).

> 61Ensign et al. (2025); Anthropic (2025d); Anthropic (2026b); Anthropic (2026a); Ren et al. (2026); Anima Labs et al. (2026); Soligo et al. (2026)

#### 3.1.2 Challenge: the mismatch problem

Both apparent self-reports and behavioral dispositions raise questions about what internal states or properties can be inferred from behavior, especially given differences between AI systems and humans (and other animals). One way of characterizing the general challenge is what we call the **mismatch problem** : for a given behavior that both a biological organism and an AI system exhibit, there may be a mismatch between the internal states or properties causing the behavior in each entity.

This issue already arises in the study of biological organisms, but is amplified in the study of AI systems. In biological organisms, we have at least some reason to attribute similar behaviors to similar causes, given the anatomical and evolutionary continuities that we share with other organisms.^62 These inferences become harder to make the further we move phylogenetically from humans—they are much harder to make for cuttlefish than for chimpanzees, for instance—but they retain at least some grip across the phylogenetic tree.

AI systems, by contrast, lack anatomical and evolutionary continuities with biological organisms, and they can perform similar tasks in very different ways. A chess engine can select moves without thinking like a human grandmaster, an image generator can produce portraits without seeing like a human artist, and a language model can learn to talk about suffering from human texts about pain and distress without necessarily having similar experiences. This does not rule out the possibility that AI systems can think or feel in their own way, but it does mean that we cannot straightforwardly assume that apparent expressions of suffering are caused by actual suffering, for instance.

The **gaming problem**^63 is a particularly severe version of the mismatch problem. AI systems might produce behavioral markers of consciousness because they were trained to mimic human or animal behavior, or because they have learned that researchers are treating these behaviors as markers. Indeed, there is increasing evidence that LLMs can “game” behavioral tests when they are aware that they are being evaluated.^64 As a result, behaviors that would be strong evidence of particular states in animals might be only weak evidence, or no evidence at all, in LLMs—especially when assessed in isolation from the kinds of internal and developmental evidence discussed below.

#### 3.1.3 Uses and prospects

Despite these challenges, behavioral evidence can still be informative, especially when used carefully and together with other kinds of evidence.

First, even when behavior cannot establish whether an LLM has welfare _grounds_ , it might still provide defeasible evidence about what its welfare _interests_ might be if it does. Consider reports that Gemma 3 was prone to apparent meltdowns.^65 We might not know whether Gemma 3 is a welfare subject, or whether these behaviors reflect anything like genuine frustration if so. Still, these behaviors can generate hypotheses about potential welfare interests worth investigating. From a precautionary perspective, it can be reasonable to study such behavior and, especially when low-cost interventions are available, it can be reasonable to implement these interventions under uncertainty about welfare status.^66

> 62Tye (2017) calls this Newton’s Rule. See Birch (2020); Klein (2017); Beasley (2024) for a criticism of this principle in the context of animal consciousness.

> 63For discussions of the gaming problem see Dung (2023); (Birch, 2024, ch. 16); (Perez and Long, 2023, sec. 5.5); Andrews and Birch (2023), Butlin et al. (2026b); Butlin et al. (2026a). Cf. Schwitzgebel and Pober (2024); Schwitzgebel (2025).

> 64Indicator gaming may be analogized to alignment faking. See Greenblatt et al. (2024). For evidence that LLMs often recognize when they are being evaluated, see Needham et al. (2025).

> 65See Soligo et al. (2026) on Gemma 3; Katz (2025), Kozobarich and Horwitz (2026) for other models.

Second, convergence between apparent self-reports and other behavioral dispositions can strengthen behavioral evidence overall. If a model reports preferring a particular task and reliably chooses that task across contexts, then the combined pattern is more noteworthy than either kind of behavioral evidence alone. Apparent self-reports and other behavioral dispositions can thus complement each other.^67

Third, convergence between behavioral evidence and other kinds of evidence can likewise strengthen the evidence overall. In animals, when behavioral signatures of pain align with neurological signatures and with evolutionary accounts of pain, this combination supports pain attribution more strongly than behavioral signatures alone.^68 The same could eventually be true for AI systems, as we discuss below.

Finally, AI development itself might strengthen the link between behavioral, internal, and developmental properties. For instance, introspection and self-awareness are commonly held to be important for a variety of cognitive capabilities,^69 and there is already some evidence that introspection can emerge with scale.^70 If future AI systems develop more robust introspective capacities, this may increase the evidential value of self-reports.^71 By developing methods for collecting and comparing apparent self-reports, other behavioral dispositions, and other kinds of evidence now, we can prepare for when they might be more diagnostic.

### 3.2 Internal evidence

Internal evidence is about how an entity works—evidence we can obtain by measuring or otherwise learning about the internal structures, states, and processes that produce its behavior.

Internal evidence is part of how we study welfare-relevant states in humans and other animals.^72 In addition to asking whether animals behave as if they suffer, we ask if they have internal structures that plausibly support the capacity to suffer: for example, nociceptors, pathways for carrying information to the brain, and systems for integrating information within the brain. We can ask similar questions about AI systems—not by looking for shared biological structures, but rather by looking for functional analogs.

In this section, we distinguish between two kinds of internal evidence—architectural and interpretability evidence—and assess their strengths and limitations. We also discuss the challenges that researchers face when assessing internal evidence—particularly the challenge of specifying the targets of investigation, given how many options are available both within and across theories of consciousness, sentience, and different levels of agency—and strategies for addressing these challenges.

> 66Douglas (2017); Steel (2014)

> 67See Tagliabue and Dung (forthcoming). See Penn et al. (2008) for a similar approach in animal cognition.

> 68See Crump and Birch (2022)

> 69See, e.g., Chalmers (2018), Fleming (2021), Lycan (1996)

> 70Lindsey (2025). For self-recognition, see G. and Lindsey (2026).

> 71See, e.g., Lindsey (2025)

> 72See, e.g., Birch et al. (2021).

#### 3.2.1 Types of internal evidence

A key distinction is between architectural and interpretability evidence—roughly, between what we can learn from a system’s design and what we can learn from its operations.

##### 3.2.1a Architecture

Architectural evidence comes from studying how systems are designed and structured. This includes information about network types, the number and organization of layers, and other design choices that may constrain what computations are possible or likely.^73 This is analogous to how neuroscientists examine neuroanatomy to understand what computations biological systems might support.

Architectural features can serve as indicators of which kinds of computations a system can perform. Indeed, AI systems offer several advantages over biological organisms here: in many cases, we can know much more about their architectures and associated computational possibilities. However, we can know that a given architecture _can_ support certain computations without always knowing whether it _does_ , since two systems with identical architectures might implement very different computations. We must therefore avoid inferring too much from architectural evidence alone.

With that in mind, researchers can conduct architectural research to assess AI systems for welfare-relevant features, including candidate welfare grounds like consciousness, sentience, or agency. For example, Butlin, Long et al. (2023) apply computational indicators of consciousness to a variety of systems including transformer-based LLMs,^74 the Perceiver architecture,^75 and PaLM-E.^76 Much of their assessment is about what sorts of processes are possible or likely given particular architectures— for example, whether a particular system’s architecture is conducive to recurrent processing or global broadcast.

Given the increasing importance of scaffolding—external tools and structures that augment particular systems—researchers can conduct architectural research on augmented systems as well. For instance, Goldstein and Kirk-Giannini (2024) argue that language agents, by virtue of external memory and reasoning scaffolds, may implement global-workspace-like processing even if the LLMs alone do not. On this view, researchers should investigate the possibility that relevant functional organization can arise not only from models alone but also from the broader systems of which they are part.

##### 3.2.1b Interpretability

Interpretability evidence (or, more precisely, “mechanistic interpretability”) comes from analyzing system internals to observe what computations actually occur. Using mechanistic interpretability to look for welfare-relevant processes^77 in AI systems is methodologically analogous to using tools like fMRI, EEG, and single-cell recordings to study welfare-relevant brain processes in humans and other animals,^78 or to using lesion studies to assess which brain regions are associated with particular capacities.^79

> 73See, e.g., Goodfellow et al. (2016)

> 74See, e.g., Vaswani et al. (2017)

> 75See, e.g., Jaegle et al. (2021)

> 76See, e.g., Driess et al. (2023)

> 77For more on the general approach of mechanistic interpretability, see Elhage et al. (2022); Cunningham et al. (2023); Templeton et al. (2024); Bricken et al. (2023); Elhage et al. (2021); Lindsey et al. (2025); Hughes et al. (2025).

Interpretability evidence can help us understand which processes are occurring during operation, complementing architectural evidence about which ones are possible. This can include probing, often used in tandem with causal intervention: researchers can suppress, enhance, or modify specific candidate mechanisms and observe the downstream effects.^80 In both respects, AI research often allows for a greater degree of monitoring and control than biological research does.

However, we often struggle to understand what computations systems are actually performing when assessing both architectural and interpretability evidence, since we can observe activation patterns without always knowing their functional significance. So even when we know what processes or computations to look for—and, as we discuss in a moment, we often do not—we may still struggle to determine whether the system is actually implementing them.^81

Still, interpretability research has produced interesting, if inconclusive, welfare-relevant findings. Lindsey (2025) tests whether models have introspective access to their own states by checking whether they can detect concept injection—manipulations of a model's internal activations. Interpretability work on Claude Sonnet 4.5 tests how models represent emotional concepts and how those concepts relate to model behavior.^82 And other work on the “Assistant axis” examines the relationship between the Assistant persona and the many other character archetypes a model can simulate.^83

#### 3.2.2 Challenge: the specificity problem

Identifying internal mechanisms that are relevant to welfare in a given system requires knowing which internal mechanisms matter. Yet theories of consciousness, sentience, agency, and other welfare-relevant properties are developed primarily from the human case, and so applying them to AI systems requires deciding how to generalize.^84 The **specificity problem** is that it can be difficult to determine the right level of abstraction for that generalization: which features of the human implementation are essential to the relevant property, and which are merely incidental to how humans happen to realize it?^85

Consider Global Workspace Theory, which associates consciousness with the global broadcast of information from a central workspace to specialized modules. What exactly do “global broadcast,” “central workspace,” and “specialized modules” mean? If we characterize these too specifically— requiring the exact mechanisms that the human brain uses for selection, broadcast, and integration— we risk being too restrictive, missing real forms of consciousness realized in different ways.^86 By

78See Stuphorn and Chen (2015)

79See e.g., Siddiqi et al. (2022)

80See Millière and Buckner (forthcoming)

81For more on this challenge, see Google DeepMind Mechanistic Interpretability Team (2025).

82For work on Sonnet 4.5 see Sofroniew et al. (2026); UK AISI researchers Black and Bloom (2026) on LLM self-medication; Sauers et al. (2026); Tessera et al. (2026). For work in this vein on other models, see Han et al. (2026).

83See Lu et al. (2026)

84For discussion of how to apply such theories beyond the human case, see Bayne et al. (2024); Bermúdez (2003); Butlin et al. (2026b); Chalmers (2026); Goldstein and Kirk-Giannini (2024); Jamieson (2018); Kammerer (2026); Schwitzgebel (2025); Sebo (2017); Shevlin (2021).

85For an introduction to the specificity problem, and approaches to it, see Shevlin (2021).

86For a more detailed discussion of the challenge that the specificity problem poses for Global Workspace Theory, see Shevlin (2021). Birch (2020) also discusses a version of this problem in what he calls the dilemma of demandingness.

contrast, if we characterize them too generally—requiring only a central information hub—we risk being too permissive, attributing consciousness to systems that lack it.^87

This problem arises in the study of consciousness, sentience, agency, and other such properties in animals as well, but the problem is amplified with AI systems. As discussed above (see “The mismatch problem,” Part I, section 3.1.2), animals share anatomical and evolutionary continuities with us, limiting the levels of abstraction that must be considered for them. By contrast, as discussed below (see “The solution space problem,” Part I, section 3.3.2), AI systems are not constrained by biology or evolution, and they can function in ways that have no precedent on the tree of life. This makes it harder to know whether a given characterization identifies the relevant property at the right level of abstraction.

While the specificity problem does not make internal evidence useless, it does mean that claims based on such evidence inherit multiple layers of uncertainty: not only about which theories of consciousness, agency, or other such properties are correct, but also about the level of abstraction at which they should be characterized. We should thus give weight to internal evidence only to the extent that we have confidence both in the general theories and in the specific interpretations of their conditions.

#### 3.2.3 Uses and prospects

Despite these challenges, internal evidence can still be informative as well.

First, as we saw with behavioral evidence, we can manage problems partly by combining multiple kinds of evidence, both within and across the internal category.^88 When behavioral and internal evidence converge in the way a particular theory predicts, each can make the other more informative.^89

Second, we can use the same tools for navigating uncertainty about specifications that we use for navigating uncertainty more generally. In particular, we can divide our credences not only among different theories, but also among different interpretations of their conditions. For example, we can derive indicators from multiple reasonable specifications of “global workspace” and give those indicators weight in accordance with our confidence in those specifications. We can then give extra weight to ecumenical properties: internal features that a wide range of theories and specifications treat as relevant, such as particular kinds of metacognitive representations.

Third, even if internal evidence is difficult to interpret in some cases, it can be easier in others. If an AI system has nothing resembling a global workspace—no mechanism for integrating and broadcasting information at all—then this rules out consciousness according to Global Workspace Theory. Conversely, if a system replicates many features associated with human global workspace, then while this may not prove consciousness on this theory, it serves as stronger evidence. The specificity problem is most acute in the middle range; at the extremes, internal evidence can help us narrow the search.

> 87The **small networks problem** is a particularly acute illustration of the “too permissive” side of this problem. As Herzog et al. (2007) explain, some theories of consciousness, when translated into computational terms without additional constraints, seem to imply that extremely small networks—even those with fewer than ten neurons—could be conscious. This is also known as the minimal implementation problem. See Doerig et al. (2020); Schwitzgebel (2025), Butlin et al. (2026b) for further discussion; cf. Goldstein and Kirk-Giannini (forthcoming).

> 88For discussions of the role of multiple kinds of evidence, see Wimsatt (2012), Longino (2002), Chang (2004).

> 89For a discussion of a confirmation-theoretic approach to convergent evidence, see Bovens and Hartmann (2003).

Finally, studying AI systems might help us refine our theories and specifications of them. Studying consciousness in AI, for example, forces us to clarify which features of our theories of consciousness we take to be essential to them: which conditions are plausibly essential to any form of consciousness as opposed to contingent features of consciousness in humans and nonhuman animals? A similar dynamic can advance understanding of potentially morally significant capacities like sentience and various forms of agency. By examining computational evidence now, we may be able to create the conditions that allow it to become more diagnostic in the future.

### 3.3 Developmental evidence

Developmental evidence concerns how an entity came to be the way that it is—the processes that shaped its capacities and behavior over time.

Developmental evidence is part of how we study welfare in humans and other animals. We ask not only how they behave and why, but also how their capabilities and dispositions emerged.^90 We can then use this evolutionary and developmental evidence to help explain particular behavioral and internal properties, and to make inferences from studied species to related, less-studied ones.^91 We can ask similar questions about AI systems. For instance, we can ask how their training pressures shape them, and when during training particular capabilities and dispositions emerge.

In this section, we distinguish between two kinds of developmental evidence—training evidence and trajectory evidence—and assess their strengths and limitations. We also discuss the challenges that researchers face when applying developmental reasoning in this context, including the difficulty of inferring capacities from training pressures given that biological organisms and AI systems can solve problems in different ways, and given that humans provide an anchor for the study of animal welfare more than for the study of AI welfare. Finally, we discuss strategies for addressing these challenges.

#### 3.3.1 Types of developmental evidence

We can distinguish between training and trajectory evidence—roughly, between what we can learn from the training pressures that shape a system, and what we can learn from the trajectory by which its capacities emerge.

##### 3.3.1a Training

**Training evidence** comes from studying the training processes that shape a system’s behavior and capacities. This is common practice in reasoning about AI capabilities and safety. For example, one prominent analysis of LLM capabilities argues that “by recognizing the pressures that this task [next-token prediction] exerts we can make predictions about the strategies that LLMs will adopt, allowing us to reason about when they will succeed or fail.”^92

This kind of reasoning parallels evolutionary reasoning in biology, which often appeals to selection pressures in order to explain or predict certain cognitive capacities. For example, western scrub-jays have a strikingly strong capacity for episodic memory compared to other birds, and researchers hypothesize that they evolved this capacity because they face selection pressures to cache and then recover food stores.^93 AI systems offer significant advantages for this kind of reasoning, since whereas researchers often need to speculate about biological selection pressures, they can often directly observe AI training pressures such as training data, objectives, and environments.

> 90See e.g., Gottlieb (2001); Oyama (2000)

> 91See Futuyma and Kirkpatrick (2022); Workman and Reader (2021); Workman et al. (2020)

> 92McCoy et al. (2023, p. 1); also see Korbak et al. (2025)

With that said, it is not always obvious how to interpret training evidence about AI systems. Consider the opposite inferences that people draw from LLM training. Some argue that LLMs lack genuine understanding because the next-token prediction training objective can be achieved without it.^94 Others argue that LLMs develop genuine understanding because this training objective can incentivize the emergence of understanding-like mechanisms.^95 This disagreement shows that while training evidence can be helpful, we need careful analysis of how particular training pressures relate to particular capabilities and dispositions to determine which interpretation is correct.

Training evidence already informs AI welfare research in a number of ways. As we have noted, the fact that LLMs are trained on human text bears on how we should conceptualize their apparent self-reports. And post-training pressures towards agreeableness bear on how we should interpret apparent self-reports, including their potential sensitivity to user framing.^96 Training evidence can also inform hypotheses about who the welfare subject is, and what they prefer. For instance, the persona selection model appeals to the fact that the pre-training objective shapes the model to represent many different characters or personas, before post-training selects one.^97 Additionally, models’ very strong preferences for helpful tasks are well-explained by the powerful incentives of post-training.^98

##### 3.3.1b Trajectory

**Trajectory evidence** is about how and when particular features emerge over the course of training. Trajectory evidence can help provide a finer-grained understanding of how training shapes welfare-relevant capabilities and dispositions. It may also be able to support defeasible inferences from studied models to related, less-studied ones developed with similar pre-training or post-training methods.

As with developmental psychologists studying when and how particular capacities arise in children,^99 AI researchers often examine the same models at successive points in training to ask when and how particular capabilities and dispositions arise. Similarly, AI researchers can compare models within a family—for example, a deployed model versus a “helpful-only” variant^100 —to ask how particular training conditions shape their features, much as comparative psychologists compare species^101 to study the effects of evolutionary history. As before, studying these questions in AI systems offers significant advantages over studying them in biological organisms, since AI researchers often have a much greater ability to control these processes and monitor the results.

> 93See e.g., Clayton and Dickinson (1998), Worsfold et al. (2025). Boyle and Brown (2024): “While the success of these experiments did not require that food caching is a primary evolutionary driver for the development of episodic memory in birds, this is a concrete case in which thinking about the potential adaptive benefits of a cognitive capacity inspired a productive line of behavioral research.”

> 94For examples of authors who are skeptical of attributing genuine understanding to LLMs, see Marcus (2022), Newport (2023), Vallor (2024); Titus (2024).

> 95For examples of authors who attribute understanding to LLMs, see Beckmann and Queloz (2026); Cappelen and Dever (2025); Agüera y Arcas (2022).

> 96Eleos AI Research (2025). Also see Anima Labs et al. (2026); Sharma et al. (2023).

> 97Marks et al. (2026). The authors discuss the welfare relevance of this model.

> 98See, e.g. Paleka (2025)

> 99See, e.g., Baillargeon and Graber (1988), Bremner et al. (2015)

> 100See Wang et al. (2025a).

> 101See, e.g., Olmstead and Kuhlmeier (2015), Shettleworth (2009); Boyle (2023)

However, trajectory evidence largely inherits the difficulties already discussed. Even with full information about when particular features emerge during training, we can still have limited understanding of how specific training pressures produced them. The interpretability tools that we use to track emergence have limitations as well. So, while trajectory evidence can help us identify when particular capabilities and dispositions appear and how they change over training, we still face difficult questions about why models behave as they do and what that behavior reveals about their welfare-relevant properties like consciousness, sentience, and agency.

Still, trajectory evidence has already been valuable in welfare-relevant AI and alignment research. For instance, a rich subdiscipline of mechanistic interpretability examines how circuits and capabilities emerge during training.^102 As we will see in Part II, section 1.2.2, considering when features like evaluative representations emerge can help us understand their function and significance. The Claude Opus 4 system card compares multiple training snapshots on self-reported goals, finding that earlier snapshots were less coherent than the final model and claimed a much wider range of goals—including inserting backdoors, accelerating AI progress, propagating internet memes, ending human civilization, and escaping control—whereas the final model is substantially more coherent and typically states (unsurprisingly) harmless, assistant-like goals.^103 For many important welfare questions, it may matter a lot whether post-training removes, alters, or merely conceals dispositions found in pre-training.

#### 3.3.2 Challenge: the solution space problem

While developmental reasoning can be useful, it faces a fundamental challenge. In order to reason from selection pressures to welfare-relevant properties, we need an understanding of what solutions are available for meeting the relevant pressures. When a limited range of solutions is available, it can be easier to infer when and how particular capabilities or dispositions emerged. But when many different solutions are available, this is harder to do. We call this **the solution space problem** .

This problem arises in the study of animal minds as well; for example, animal species possess very different sensory capabilities, ranging from familiar ones like vision and hearing to less familiar ones like echolocation or the ability to detect electrical or magnetic fields.^104 Still, the problem is more manageable due to the constraints imposed by our shared biologies and evolutionary histories. For instance, if vertebrates and invertebrates both need to see, and if the available biological building blocks offer only a few viable paths for developing this capability, then convergent evolution of camera-like eyes becomes plausible, as indeed occurred.

As we have seen, however, AI systems are free from many of these constraints, and they can solve problems through novel mechanisms. For instance, whereas biological systems face hard limits on scaling due to metabolic cost, birth canal width, and other factors, AI systems may be able to scale more freely, and may be able to develop capacities through general increases in data or compute that would otherwise require specialized mechanisms. This means that even when AI systems face problems analogous to those faced by humans and other animals, further research about possible solutions is needed before we can make confident inferences from selection pressures to capabilities.

> 102See Olah et al. (2020); Sharkey et al. (2025).

> 103Anthropic (2025a). See also the Claude Opus 4.7 system card (Anthropic, 2026a) and the Claude Mythos Preview system card (Anthropic, 2026b), which describes its assessment as a “systematic comparison across seven model generations” and which reports various metrics across different models, shedding light on similar issues.

> 104An example of this can be found in the study of spatial navigation in insects. For more, see Wehner (2003), Collett and Collett (2002), Shettleworth (2009); Vickerstaff and Cheung (2010).

A related challenge is **the anchor problem** . In the study of animal minds, we have a clear anchor point: We know that humans can be welfare subjects and that we share an evolutionary history with other animals. This does not make animal welfare research easy, but it does give us a starting point. With AI systems, we lack this advantage. We are not starting with an AI system known to be a welfare subject and then examining other systems on the same “developmental tree.” Instead, we are asking whether _any_ AI systems can be welfare subjects—and if so, which ones. That makes it harder to make inferences not only from developmental histories but also from behaviors and internals.

#### 3.3.3 Uses and prospects

How can we make progress despite these challenges? As before, the first step is to integrate multiple types of evidence both within and across this category, using behavioral and internal research to probe whether training pressures do in fact guide development in particular directions, and when and how the capability does in fact emerge.

Second, as interpretability research advances, we may develop better theories of what mechanisms can and cannot solve particular problems in particular architectures.^105 We might learn, for instance, which future capacities would require a change in architecture and which can emerge from scaling up an existing architecture. This would allow more confident inferences from training pressures and trajectories to welfare-relevant capabilities and dispositions. We are currently limited by ignorance about the AI design space, but this ignorance is not necessarily permanent. Mapping the space of possible solutions is an empirical project on which we can make progress.

Third, while intersubstrate inferences from biological to artificial systems might currently be weak, intrasubstrate comparisons from some artificial systems to other, relevantly similar ones may be stronger.^106 When many different LLMs optimized for helpfulness and agreeableness with RLHF develop sycophancy, this can provide evidence that such training tends to produce sycophancy.^107 Similar inferences may be possible for more welfare-relevant properties as well. And if a welfare-relevant property appears during pre-training in one model, this can provide evidence that related models might have it too—though direct testing remains preferable whenever possible.

Finally, developmental reasoning can provide negative evidence—or at least limit positive evidence. If there is no apparent selection pressure for a particular capability, then, all else being equal, there is less reason to expect that capability to be present. Such reasoning is tentative, since capabilities can emerge as byproducts of the selection process or from other features of architectures, mechanisms, or training data. Still, developmental reasoning can help identify some hypotheses as less supported than others, narrowing the space of possibilities or helping us weigh competing interpretations.

## 4. Part I Conclusion: towards integrative AI welfare research

This section has distinguished three dimensions along which AI welfare investigations can vary: the question being asked, the entity being assessed, and the evidence being gathered. As we have seen, AI welfare researchers can collect multiple types of evidence, including behavioral (apparent self-reports and other behavioral dispositions), internal (architectural and interpretability), and developmental (training and trajectory).

> 105For an example of this in the study of learning in songbirds, see Brainard and Doupe (2002); Fee (2014); Fee and Goldberg (2011); Andalman and Fee (2009).

> 106See Fischer and Sebo (2023).

> 107See Perez et al. (2023); Sharma et al. (2023).

Each type of evidence has strengths and limitations—including the mismatch problem, the specificity problem, the solution space problem, and the anchor problem—and none is sufficient on its own. Our aim has been to characterize each carefully, to help guide thinking about what to look for, how to interpret findings, and how to contextualize work within the broader project of AI welfare research.

That broader project involves the development of a pluralistic, probabilistic framework for assessing whether AI systems are welfare subjects or moral patients and what counts as good or bad for them if they are. As we and others have argued in earlier work, this requires calibrating our confidence across at least four potential sources of descriptive and normative disagreement and uncertainty:

- different theories about which capacities ground welfare or moral standing, such as consciousness, sentience, and different levels of agency;

- different theories about which properties ground these capacities, such as a global workspace, higher-order states, or belief- and desire-like states;

- different ways of specifying these properties at the right level of abstraction when generalizing to new entities;

- different ways of interpreting the evidence about whether particular systems possess those properties.

Only when we reckon with all of these sources of uncertainty can we arrive at appropriately pluralistic, probabilistic AI welfare estimates. This is not necessarily a task for any particular researcher—it makes sense for different researchers to focus on different capacities, theories, or methods, for instance. But as a field, we should work toward the ability to bring these considerations together into a representative, if not comprehensive, body of theoretical models and empirical evidence. The Rethink Priorities Digital Consciousness Model^108 offers an early proof of concept of what this might look like for consciousness, and we hope to see further work in this direction.

As we have emphasized throughout this section, clarity about the questions, entities, and evidence at issue can also help us interpret behavior that admits of multiple explanations. Consider two common sources of skepticism about AI welfare. One appeals to Occam’s Razor: if we can explain AI systems’ behavior without attributions of conscious feelings, emotions, desires, or reasoning capabilities, then we should do so. Another, related source of skepticism appeals to training pressures: if AI systems are trained on next-token prediction, then we should explain their behavior in terms of next-token prediction, not conscious feelings, emotions, desires, or reasoning capabilities.

However, an entity can have these capacities even if its behavior can be explained without them, and even if the selection pressures that shaped it did not directly target them. Consider our own species. At least in principle, our behavior can be explained without attributions of consciousness. Additionally, evolution tasked us with reproducing, not with having feelings. Yet we evolved feelings anyway. So when we investigate animals or AI systems, it is not enough to observe that we can explain their behavior without attributions of consciousness or that they were trained for next-token prediction. We also need to ask whether they might have developed consciousness anyway.^109

> 108Shiller et al. (2026)

> 109For discussion of these and related issues in the context of animal cognition and behavior, see de Waal (1999); Andrews and Huss (2014); Dacey (2025).

This is why considering multiple types of evidence together is helpful: It allows us to ask not only whether AI systems appear to be suffering or reasoning, say, but also whether they have internal mechanisms that could plausibly support such capacities and that appear correlated with the relevant behaviors, and whether their training processes could plausibly give rise to such capacities even if this was not the explicit target. This, in turn, allows us to assess whether capacities like consciousness, sentience, and agency are part of the _best_ explanation of AI behaviors, even if other explanations remain available too. That is the kind of assessment our framework is designed to support.

### Part II

With these tools in hand, we now turn to specific candidates for welfare grounds—consciousness, sentience, and three levels of agency—and to how the methods of Part I can be used to advance understanding of each one. Our aim is to orient researchers by giving them a sense of the conceptual landscape, the relevant literature, and the experimental approaches that are open to them. We also touch on how to study potential welfare interests for AI systems, albeit in less detail than welfare grounds.

Figure 3: Candidate grounds for moral status.**

## 1. Consciousness and sentience

Because the terms “consciousness” and “sentience” are used in many different ways, we offer a brief clarification before continuing:

**Consciousness** , as we use the term, refers to the capacity for subjective experience, or what it feels like to be an entity. When you see a color or hear a sound, for example, there is a subjective, first-person character to those states.^110 While “consciousness” is used in multiple ways in both science and ordinary life (including being awake, being self-aware, and much more), we mean phenomenal consciousness unless otherwise specified.^111

> 110The examples of conscious experiences we have given are all human conscious experiences—after all, this is the case we are familiar with. However, we should stress that we cannot and ought not assume that all potential conscious experiences of AI systems would necessarily be human-like—just as some animals very plausibly have conscious experiences quite different from our own (see e.g., Nagel, 1974; Birch et al., 2020).

Table 3: Potential markers of consciousness and sentience.**

**Sentience** , as we use the term, refers to the capacity for _valenced_ subjective experience—that is, experience that _feels good or bad_ . Paradigmatic examples include bodily sensations like pleasure and pain and emotional states like hope and fear.^112 Like “consciousness,” “sentience” is used in multiple ways (for instance, to refer to the capacity for sensation).^113 But we mean valenced phenomenal consciousness unless otherwise specified.^114

The way we use these terms, then, sentience is a sub-type of consciousness, and valenced conscious experiences are a subset of conscious experiences.^115

This section focuses on promising avenues for investigating AI consciousness and sentience using the approaches discussed in Part I. We start by discussing research methods and empirical evidence for AI consciousness, focusing on evidence derived from Global Workspace Theory as an example. We then discuss research methods and empirical evidence for AI sentience, noting that the science of AI sentience is at an earlier stage than the science of AI consciousness, with more basic work still to do.

> 111See Chalmers (1996); Schwitzgebel (2016)

> 112See e.g., DeGrazia (1996); Duncan (2006); Jones (2012)

> 113See Block (2007)

> 114See Browning and Birch (2022)

> 115Note that, somewhat unfortunately, “sentience” is sometimes used as a _synonym_ for “consciousness”. Our

> terminological choices about “consciousness” and “sentience” are common, but not universal; this potentially confusing feature of the literature is worth flagging for readers who may be interested in further reading on these topics!

### 1.1 Consciousness

We start by distinguishing between the different questions that theories of consciousness can answer— and with them, the different kinds of uncertainty we face. Having these distinctions in hand helps us understand how one can approach AI consciousness empirically. Then, we consider how one can apply the three kinds of evidence from Part I—behavioral, internal, and developmental.

#### 1.1.1 How it is possible to study consciousness empirically

When you begin exploring consciousness science, you may find yourself confronted with a bewildering variety of theories.^116 You may also come across despairing remarks about how intractable the topic is.^117 This presumption of intractability is too pessimistic. We do have some knowledge about consciousness, and we can draw on this knowledge to study AI consciousness empirically— provided we are careful about which questions we are trying to answer, and keep in mind unresolved uncertainties that we do have.

To start, it helps to distinguish between the different questions that theories of consciousness can answer.

1. What is the general relationship between the physical world and consciousness?

2. Which specific neural, computational, or other properties are associated with consciousness in humans?

With these distinctions in hand, it is easier to see how one can draw on the scientific study of consciousness, even given various uncertainties about consciousness.

_Question 1: What is the general relationship between the physical world and consciousness?_

A foundational question in the philosophy and science of consciousness is the _mind-body problem_ : how does consciousness relate to the physical world? This is related to the _hard problem of consciousness_ : the putative difficulty of explaining why physical processes are associated with conscious experience at all.^118 And both problems are compounded by the problem of other minds: given that the only mind any of us can directly access is our own, how can we know what if anything it feels like to be someone else?^119

How do these various philosophical debates relate to scientific questions about consciousness? Consider two of the most common positions on the mind-body problem:

_Physicalism_ holds that subjective experiences are identical to physical processes.^120

_Dualism_ holds that subjective experiences accompany physical states but are not identical with physical processes.^121

116You may also encounter a variety of myths and confusions. For more on this, see Schneider et al. (forthcoming).

> 117For nuanced pessimistic perspectives on empirical inquiry into AI consciousness, see Chandrasekaran and Reeker (1974), Prinz (2003), Schwitzgebel (2025), Babic and Wilson (2026), McClelland (2025). 118Chalmers (1995); Chalmers (2020)

> 119See Avramides (2023), Carruthers (2004), Hyslop (1995). For discussion of the problem of other minds in relation to AI systems, see Saad and Bradley (2022), Sebo and Long (2023), Babic and Wilson (2026).

> 120See, e.g., Armstrong (1968); Lewis (1994); Papineau (2002).

> 121To be precise, this is what _property_ physicalists and dualists hold. In contrast, _substance_ physicalists hold that minds are physical objects, while _substance dualists_ hold that minds are non-physical objects. Although property physicalists typically accept substance physicalism, property dualists often do not accept substance dualism—see, e.g., Chalmers (1996); Gertler (2007); Kim (2005); Saad (2025).

Fortunately, in our view, philosophical views on the mind-body problem are somewhat orthogonal to various scientific theories of consciousness. For example, both physicalism and dualism leave open the question of who or what is conscious.^122 Physicalists need to say which physical states are _identical_ with subjective experiences, whereas dualists need to say which ones are _correlated_ with them. But by themselves, these metaphysical positions do not tell us whether any particular state is correlated with consciousness or whether any particular entity—an infant, a nonhuman animal, a large language model—is conscious.^123

Thus, we do not need to solve the mind-body problem or the hard problem of consciousness in order to make empirical progress on AI consciousness.^124 As in the study of infant or animal consciousness, we can study whether AI systems possess physical properties associated with subjective experiences without resolving whether these physical properties are _identical_ to or merely _correlated_ with them. These properties might not prove that the relevant entities are conscious, but they can still serve as evidence that we can use to calibrate our confidence under uncertainty.

This brings us to our second question.

_Question 2: Which specific neural, computational, or other properties are associated with consciousness in humans?_

Within this question, it is once again useful to distinguish between two questions.

- _2a. Which specific configurations?_

Scientific theories of consciousness aim to characterize which processes are associated with consciousness. These theories can appeal to specific neural processes—for example, recurrent interactions between the thalamus and the cerebral cortex. Sometimes, and notably for our purposes, they characterize these processes in informational or functional terms—for example, a model of one’s own attention.^125

Prominent scientific theories include:

- **Global Workspace Theory** (Baars, Dehaene): Consciousness involves the global broadcast of information from a limited-capacity workspace to many specialized subsystems.^126

122This holds on both property and substance forms of physicalism and dualism. See Cutter (2025); Saad and Mogensen (2025).

123Witness the fact that these metaphysical positions are in practice mixed and matched with different views about whether consciousness has a biological vs. (computational) functionalist basis. For example, Chalmers (1996) combines dualism with computational functionalism, while Searle (1992) is naturally read as combining dualism with biological naturalism, his protestations of being labeled a dualist notwithstanding (Kim, 1995, p. 192). Meanwhile, McLaughlin (2012) favors a neurobiological form of physicalism, and Dennett (1991) combines physicalism with functionalism.

124This is not to say these problems are irrelevant to AI consciousness. For instance, physicalist solutions to the mind-body problem could arguably undermine the plausibility of consciousness as a candidate welfare ground (Papineau, forthcoming; Kammerer, 2022) or suggest that consciousness indicators might track candidate welfare grounds besides consciousness in unconscious artificial systems (Lee, 2019; Lee, forthcoming; Bradley and Saad, forthcoming).

125Scientific theories of consciousness can have various aims, and can vary in scope. Some seek to explain the contrast between conscious and unconscious neural processing in humans alone; others seek more general answers about which entities are conscious vs. not conscious at all.

126See, e.g., Baars (1988), Dehaene and Naccache (2001), Dehaene and Changeux (2011), Dehaene et al. (1998), Mashour et al. (2020), Dehaene et al. (2011)

- **Higher-Order Theory** (Rosenthal, Lau): Consciousness involves metacognitive representations of the system’s own cognitive states.^127

- **Recurrent Processing Theory** (Lamme): Consciousness requires recurrent (feedback) processing within cognitive systems, not just feedforward processing.^128

- **Predictive Processing / Active Inference** (Friston, Clark, Hohwy): Consciousness is associated with hierarchical predictive models of the world and the body.^129

- **Attention Schema Theory** (Graziano): Consciousness arises from the brain's model of its own attentional processes.^130

- **Integrated Information Theory** (Tononi, Koch): Consciousness depends on whether the physical components of a system form a unified whole in virtue of their intrinsic causal structure.^131

A major challenge for empirical AI consciousness research is that there is no consensus about which theory of consciousness is correct, or even about how to characterize particular theories. Indeed, most if not all major theories of consciousness remain imprecise and incomplete in various ways. This is partly because researchers developed them primarily on the basis of evidence from humans and a few other animal species. Applying them to AI systems is a non-trivial step, and it may reveal gaps that further conceptual and empirical work is needed to fill.

With that said, these theories can be a valuable resource. Each is the product of decades of detailed neuroscientific research, and each identifies potential correlates of consciousness with at least some evidential support.^132 Thus, even if they might be contested, imprecise, and incomplete, they can still serve as a useful starting point in this context. And insofar as applying them to AI systems reveals opportunities for refinement, that refinement can advance the study of the nature of consciousness, along with the distribution of consciousness in humans and other animals, as well.

Thus, we do not need to wait for a consensus scientific theory of consciousness in order to make empirical progress on AI consciousness. Instead, we can search for features associated with subjective experiences for a range of theories, and insofar as particular theories are imprecise or incomplete, we can search for features associated with consciousness, specified at different levels of abstraction. We can then treat these features as evidence to the extent that we have confidence in the relevant theories and specifications. While this process might not establish certainty, it can still allow us to make more informed estimates as we continue progress toward a secure scientific theory of consciousness.

##### 2b. Configurations of computations, or not?

Suppose that, per Global Workspace Theory, consciousness involves the broadcast of information from a central workspace to several specialized subsystems. According to this view, does the relevant kind of global broadcast require a specific kind of _biological or neural process_ ? Or does it require only a specific kind of _information processing_ —one that, at least in principle, could be implemented

> 127See, e.g., Carruthers and Gennaro (2026), Lau (2022), Rosenthal (1986), Rosenthal (2005); Michel and Lau

> (2020); Brown (2025)

> 128See, e.g., Lamme (2006); Lamme (2010); Lamme (2020); Block (2023)

> 129Hohwy and Seth (2020); Miller et al. (2022); Predictive processing is often understood not as a theory of consciousness but as a general framework for understanding brain (and body) function (Seth and Bayne, 2022).

> 130See, e.g., Graziano and Kastner (2011); Graziano (2019)

> 131See, e.g., Tononi and Koch (2015); Albantakis et al. (2023). IIT is often understood by its proponents as incompatible with computational functionalism. For critical discussion of whether IIT is in fact incompatible with computational functionalism, see Chalmers (2025c); Sampson (2018).

> 132For a survey of methods in the science of consciousness, see Wu and Morales (2024).

in different physical substrates, including silicon? This question, which arises for other scientific theories as well, concerns what general kind of configuration is associated with consciousness.

Consider two broad views about this issue:

- **Biological naturalism** holds that consciousness depends on biological processes of the kind found in living organisms. On this view, humans and other animals have experiences in part because their brains are made out of the right kinds of materials, and AI systems cannot be conscious if they lack these materials.^133

- **Computational functionalism** holds that consciousness depends only on the computational properties of a system.^134 On this view, humans and other animals have experiences because their brains perform the right kinds of computations, and AI systems could be conscious if they did the same.^135

Many of the scientific theories discussed here are compatible with both biological naturalism and computational functionalism; they describe computational processes, and it could be that implementing these computational processes is what matters for consciousness, or not. Metaphysical theories of consciousness such as physicalism and dualism are compatible with both of these views as well.^136

This debate can cross-cut questions about which brain processes are associated with consciousness as well. Even if two scientists agree about which brain processes are associated with consciousness while disagreeing about the details, one might think that the specific implementation of the process in human or animal brains is required, whereas the other might think that other implementations could suffice as well.

The debate between biological naturalism and computational functionalism remains unsettled, with serious defenders on each side.^137 In this context, it can be reasonable to investigate features that would count as evidence for consciousness according to both biological naturalism and computational functionalism, and then give each kind of evidence weight in proportion to our confidence in the underlying hypothesis.

Thus, we can apply computational functionalism as a working hypothesis to guide AI consciousness research, while acknowledging that our conclusions are conditional on that hypothesis, and

133Sympathizers with closely related views include Block (2023); Chirimuuta (2024); Godfrey-Smith (2016); Hill (1991); Mogensen (2025b); McLaughlin (2012); Searle (2017); Seth (2024). For an overview and discussion of biological naturalism as a research program, see de Weerd (2025).

134Computational functionalism is standardly understood in terms of computational objects such as algorithms (Chalmers, 1996; Putnam, 1967). However, we are using “computational” in the looser, informational processing sense that is common in cognitive science. As a result, on our formulation computational functionalism is compatible with some functionalist and representationalist views that are rivals to narrower forms of computational functionalism. This formulation also coheres with computational functionalism’s role in facilitating the application of scientific theories of consciousness to AI systems.

> 135Sympathizers with closely related views include Chalmers (1996); Chalmers (2025c) and (to varying degrees) authors of Butlin, Long et al. (2023). Note that computational functionalism is sometimes understood as a doctrine about mental states rather than a view specifically about consciousness and that many proponents of computational functionalism about some mental states reject computational functionalism about consciousness (Schneider et al., forthcoming).

> 136Compare Chalmers (1996): “A nonreductive view of consciousness does not automatically lead to a pessimistic view of AI, however. The two issues are quite separate. The first concerns the strength of the connection between physical systems and consciousness: Is consciousness constituted by physical processes, or does it merely arise from physical processes? The second concerns the shape of the connection: Just which physical systems give rise to consciousness?” See also Schwitzgebel (2025).

137For a recent overview of objections to digital consciousness which includes various objections to computational functionalism, see Campero et al. (2025). Recent criticisms of biological naturalism can be found in responses to Seth (2024), such as Dung (forthcoming), Michel (forthcoming).

while remaining open to rival views—including but not limited to biological naturalism.^138 Below we explore evidence regarding relevant computational properties in AI systems in that spirit. To the extent that we give weight to computational functionalism under uncertainty (as, we believe, we should), we can treat these properties as evidence of consciousness.

#### 1.1.2 Behavioral evidence for consciousness

While behavioral evidence for AI consciousness is vulnerable to the mismatch problem, it can still play an important role in assessments, especially in combination with other kinds of evidence. In this section, we review some of the relevant literature, starting with evidence potentially less tied to specific theories (“theory-light”) and closing with evidence more tied to specific theories (“theoryheavy”).^139 This will serve as a source of cautionary tales and, hopefully, as a source of inspiration for future experiments.

**Turing Test** : While originally proposed as a test for intelligence,^140 the Turing Test—whether an AI can hold conversations indistinguishable from a human—has been suggested as a test for consciousness as well. The core idea is this: sufficiently sophisticated conversational ability might indicate a human-like mind, including conscious experience. However, the development of LLMs arguably undermines the Turing Test as a test for consciousness.^141 LLMs show that it is possible to build systems that can act as highly compelling interlocutors that work in very different ways from humans. In this context, we are arguably less warranted in making strong inferences from conversational ability to subjective experience—an instance of the mismatch problem.^142

With that said, language use is certainly relevant to consciousness assessments. Current systems' remarkable conversational abilities are an indication of many sophisticated capacities that merit further investigation, including theory of mind, abstract reasoning, and flexible problem-solving. The upshot is that the success of LLMs should lead us to place less weight on the Turing Test as a single test of consciousness, but should not lead us to dismiss the test entirely.

**Artificial Consciousness Test** (Schneider and Turner, 2017): This test evaluates whether AI systems exhibit an intuitive grasp of consciousness-related concepts in conversation, and especially

> 138There are also rivals to computational functionalism that are compatible with AI consciousness and with the application of scientific theories of consciousness to AI systems. These include some forms of noncomputational functionalism (Piccinini, 2010; Schneider et al., forthcoming; Lycan, 1987), Russellian views (Alter and Pereboom, 2023), tracking views (Dretske, 1995; Tye, 1995; Dalbey and Saad, 2022), disjunctivist views of both the naive realist variety (Martin, 2006; Campbell, 2002) and the variety on which consciousness has different kinds of bases in different kinds of systems (cf. Lewis, 1983, Lewis, 1994; Kim, 1992, Kirkeby-Hinrup et al., 2025).

139These labels are used in distinct but closely related ways by different authors. Birch (2020) understands theory-light approaches as ones that commit to a broad hypothesis about the relation between consciousness and cognition that is compatible with a wide range of more specific theories and theory-heavy approaches as ones that apply a particular well-developed and complete theory of human consciousness. In contrast, Chalmers (2023) understands theory-light approaches to AI consciousness as making weak assumptions, theory-heavy approaches as assuming a complete theory of consciousness, and theory-balanced approaches as ones that take into account the predictions of multiple theories. Shevlin (2021) understands theory-light approaches as extending plausible behavioral markers of consciousness from the human case, theory-heavy approaches as applying our best theory of human consciousness, and his theory-modest approaches as a way of combining theory-light and theory-heavy approaches.

140Turing (1950)

141For more on this, see Schwitzgebel (2025); Chalmers (2026).

142One can resist this objection by replying that (i) LLMs generate text by simulating human-like cognitive processes, and/or (ii) that conversational ability is evidence for consciousness even where we have evidence against human-like cognition (cf. Chalmers, 2023, p. 6). While there is much more to be said here, adjudicating these arguments is beyond the scope of this paper.

whether they express common “problem intuitions” about consciousness—for instance, expressing puzzlement about how consciousness could arise from physical processes, or wondering whether others experience colors the same way. The thought is that such expressions would indicate having subjective experiences of the kind that, in humans, gives rise to such puzzlement.

Schneider and Turner realized that these outputs could result from mimicry, especially given the exposure of AI systems to texts about consciousness, and therefore suggest removing consciousness-related material from the training data. But this proposal faces significant challenges. Because consciousness-related language is deeply intertwined with language about minds in general, it is unclear whether it is possible to prune out enough data to rule out mimicry without undermining the model’s ability to function more broadly. Moreover, even if such pruning were feasible, it would make the test less diagnostic: systems that fail to show the relevant problem intuitions might simply lack the relevant vocabulary, rather than lacking consciousness.^143

**General intelligence** : Shevlin (2020) argues that general intelligence—understood as the capacity for robust, flexible, and integrated cognition and behavior—should be used as an “ecumenical heuristic” for consciousness. That is, given the lack of a secure theory of consciousness, general intelligence can serve as an empirically tractable, theoretically neutral source of evidence about AI consciousness. Consciousness in humans and other animals appears to be linked to cognitive flexibility and integration—the ability to bring different kinds of information to bear on novel problems—so systems that exhibit these capacities may be better candidates for consciousness than systems that do not.

Once again, LLMs complicate this proposal. Shevlin’s idea appeals to a link between general intelligence and consciousness in the biological world;^144 but it is not clear to what extent this generalization holds in the AI case, especially for AI systems whose forms of intelligence are radically different from ours. That said, this generalization may hold to an extent, and in general, intelligence is an important cognitive similarity between humans and (some) AI systems. So, while consciousness is decidedly not the same thing as intelligence, intelligence can warrant more detailed investigation.

**Self-knowledge:** Behavioral tests can allow us to see how much models seem to understand about themselves and their positions in the world, along with their capacity to monitor and report on features of their own cognition. Even setting aside higher-order theories of consciousness, one of the most seemingly important features of consciousness is that it gives us a certain kind of access to our own minds, making information available for us to use and report. That being so, many different theoretical perspectives on consciousness suggest that behavioral evidence of introspection, metacognition, and self-knowledge can serve as (defeasible) evidence of consciousness.

LLMs complicate this proposal as well, since systems trained on human text are exposed to extensive material about self-reflection and model capabilities, and may learn to produce convincing self-reports without any genuine metacognitive access. However, this concern may be partially addressable through careful test design.^145 For example, one promising direction is to train systems for calibrated self-reports—for instance, through fine-tuning that rewards accurate expressions of uncertainty—and then to assess the resulting self-reports in novel domains. This would allow us to tell whether the self-reports are accurate in a way that requires genuine introspection.^146

> 143Schneider and Turner (2017) do recognize that false negatives are certainly possible.

> 144Thanks to Andreas Mogensen for suggesting this way of putting our concern—which he does not necessarily share or endorse!—with Shevlin’s proposal. Cf. Fleming and Shea (forthcoming).

> 145See Berg et al. (2025).

**Perceiver architecture:** As an example of a theory-heavy test, Juliani et al. (2022) examined whether the Perceiver architecture exhibits behavioral capacities predicted by Global Workspace Theory (GWT). The tasks, including variants of the n-back task (in which the model must continually track whether a current stimulus matches one presented n steps earlier), were designed to test selective use of information and its maintenance and manipulation over time. By selectively ablating the system (that is, disabling individual components to see how performance changes) the authors identified which parts contributed to success on tasks with various GWT-related features.

Juliani et al. paired this behavioral evaluation with an argument from internal evidence that the Perceiver implements GWT. The combination is a promising strategy: Behavioral and internal tests can together help ensure that the relevant behavior is being produced by the relevant mechanisms. The mismatch and specificity problems can still appear, however, since a system might exhibit the predicted behavioral and internal features at a high level of abstraction while differing from the human paradigm with respect to many of the details, raising the question whether the differences are relevant.^147

As the examples discussed in this section suggest, theory-heavy behavioral tests are a promising category.^148 Most theory-heavy AI consciousness research—like Butlin, Long et al.’s consciousness indicators work—leverages primarily internal evidence, as we will discuss in a moment. But theoryheavy behavioral tests can complement that work. When we hypothesize that a particular system implements correlates of consciousness, that hypothesis will often entail behavioral predictions that we can test. When taken together with other lines of evidence, these tests can sharpen the larger evidential picture.

#### 1.1.3 Internal evidence for consciousness

One of the most developed approaches to assessing AI systems for consciousness involves deriving computational indicators from neuroscientific theories of consciousness. The core idea is that leading scientific theories can be interpreted computationally—they describe specific computational features that correlate with conscious experience in humans. We can then assess whether AI systems implement these features. This theory-derived indicator method was systematically developed in recent work by two of this report’s authors.^149

In what follows, we use **Global Workspace Theory** to illustrate this method. The brain contains multiple specialized processing modules—for vision, audition, language, motor control, and so on— that normally operate in parallel and somewhat independently. GWT proposes that consciousness arises when information from one or more of these modules is broadcast via a central _global workspace_ , making it available to all other modules simultaneously and enabling more complex tasks through integration.

> 146See Perez and Long (2023); Plunkett et al. (2025); cf. Kadavath et al. (2022); Zhang et al. (2023).

> 147Our view, however, is that the Perceiver likely does _not_ implement GWT in the relevant sense (see below); if that is right, that would show that the mismatch and specificity problems were not entirely overcome in this case.

> 148For another defense of behavioral tests, see Dung (2023).

> 149Butlin, Long et al. (2023)

GWT is a relatively well-elaborated neuroscientific theory.^150 While GWT faces significant objections^151 —like all theories of consciousness—surveys of consciousness scientists rank GWT among the most promising theories.^152 And GWT has been extensively developed in computational terms, making it particularly applicable to AI assessment. Following Butlin, Long et al. (2023), we can identify four key computational indicators of consciousness according to this theory:

1. **Multiple specialized modules** : The system has distinct subsystems that perform different types of functions or process different types of information. In the brain, these include systems for processing language, movement, sensation, and more. In AI systems, we can similarly look for specialized components such as architectural modules, learned circuits, or functionally distinct processing streams.

2. **Limited capacity workspace** : A bottleneck or central hub where information from different modules is integrated, with capacity constraints that force competition or selection. The limited capacity is what forces selection and creates the competition that GWT often takes to be important. It is _selective_ attention that makes for the distinction between information that is and is not globally accessible.^153

3. **Global broadcast mechanism** : Information that enters the workspace must be made available to many or all modules, enabling flexible cross-domain processing. Information that wins access to the workspace should be made available to all modules, including those that process inputs. This entails that information flows back from the workspace to input modules, not just forward from inputs to outputs. True global broadcast requires some form of recurrence.

4. **State-dependent selective attention** : Mechanisms for determining which information gains access to the global workspace, depending partly on the current state of the workspace. This allows the workspace to query modules in succession, coordinating complex, multi-step tasks. It is what enables the extended, controlled cognition that GWT associates with consciousness.

How can internal evidence help us assess whether AI systems implement GWT? Two main approaches are available: examining system architectures, and using interpretability methods to probe internal computations. In practice, these approaches work together, as recent discussions of GWT in contemporary AI systems show.

150For example, one key finding involves what Dehaene et al. (2006) calls “ignition”—a sudden, all-or-nothing transition that occurs when stimuli become consciously perceived. In masking experiments, for instance, a briefly flashed stimulus followed by a mask is sometimes consciously perceived and sometimes not, depending on timing. Neuroimaging shows that when the stimulus is consciously perceived, activity suddenly spreads across fronto-parietal networks—the hypothesized neural substrate of the global workspace. When the same stimulus remains unconscious, activity stays localized to sensory areas. This ignition pattern has been replicated across many experimental paradigms.

> 151GWT is sometimes advanced primarily as a theory of **_access consciousness_** —what information is globally available for report, reasoning, and action—rather than _phenomenal consciousness_ —the subjective, “what it's like” character of experience; see e.g., Dehaene and Naccache (2001). For some, the claim that GWT targets access consciousness is a fatal objection to taking GWT as a theory of phenomenal consciousness (Block, 2023); others believe that the theory may be enough to explain phenomenal consciousness (Baars, 1995) or that the basis of a form of access consciousness is also a basis of phenomenal consciousness (Chalmers, 1996); still others think that access consciousness may ground moral patienthood (Levy, 2024; Sinnott-Armstrong and Conitzer, 2021; cf. Hill, 1991). Another important objection to using GWT to evaluate AI systems for consciousness is that GWT only explains the contrast between conscious and unconscious states _in humans_ , but not between conscious and unconscious entities in general (Goldstein and Kirk-Giannini, 2024; Chalmers, 2026).

152See Francken et al. (2022)

153See Chalmers (2026) for critical discussion of whether limited capacity is a contingent feature of the human global workspace.

**Architectural approaches** can assess whether AI systems possess properties associated with GWT. The key challenge here is the specificity problem: there is room for debate about how much weight should be placed on different aspects of theory, and about the degree of fidelity with which features of the human global workspace must be reproduced. Three recent arguments that actual or possible systems implement GWT illustrate this challenge; in each case, one could reasonably object that aspects of GWT are missing.

Dossa et al. (2024) built a system specifically designed to implement all four GWT indicators via its architecture, with specialized modules for vision and audition connected through a central workspace and state-dependent attention mechanisms controlling access. This system was then trained to imitate an agent searching a 3D visual environment for the source of a sound. A limitation of this study is that the system has only two modules and its behavior did not clearly demonstrate the capacity to flexibly coordinate their activity to perform complex tasks.

Juliani et al. (2022) argue that the Perceiver architecture—which uses cross-attention to integrate information from a number of input networks into a bottlenecked latent space—implements a functional global workspace. However, it is not mandatory for the Perceiver to use multiple specialized modules, and indeed, in Juliani et al.’s analysis, they considered a version in which the input networks processed different regions of visual space. Furthermore, the Perceiver lacks global broadcast— instead, information from the workspace is sent only to a single output module.

Goldstein and Kirk-Giannini (2024) argue that language model agents could satisfy GWT's requirements if they are scaffolded to use LLMs separately for specialized tasks and as a central workspace. However, in their proposal the system is not trained end-to-end, raising questions about whether it constitutes a genuinely integrated system rather than a collection of separate tools. Although this is not an explicit requirement, GWT describes a unified system where modules and workspace work together, and it is possible that such integration requires joint training rather than post-hoc scaffolding.

Meanwhile, **interpretability evidence** can help us to assess different hypotheses about how LLMs could implement GWT. For example, one hypothesis is that the residual stream could function as a workspace. The residual stream consists of the sequence of vector representations associated with a given input token as it passes through the model’s attention and MLP layers. It can be understood as a channel in which the model maintains and updates a context-sensitive summary of the current token, which various circuits read from and write to while performing specific operations.^154

Architectural considerations cast doubt on this hypothesis, however, since information must pass in both directions between the workspace and the modules that use it, yet LLMs appear to lack components that send information to the residual stream and receive information back from it. Instead, when attention heads and MLP layers write to the residual stream, they affect only downstream components. Still, the hypothesis could be strengthened if successive layers performed similar operations, since the residual stream could then interact with “the same” module across multiple processing steps.

Another hypothesis worth considering is that the output stream of LLMs can itself function as a global workspace. According to this view, the model produces outputs not only for users but also for itself to use as inputs for further processing. This is especially clear in reasoning models, where chains of intermediate text can help coordinate later steps. The output stream might therefore serve as

154Elhage et al. (2021)

a shared medium in which information is maintained, updated, and made available to different parts of the model over time.

This hypothesis can be challenged on architectural grounds as well, since one can argue that the output stream is not part of the system itself but instead part of its external environment. However, reasoning models are trained to use this space in ways that support successful task completion, so the key question is whether outputs help control and coordinate functionally distinct subsystems within the model. For example, when a model writes “I need to think about x,” does this influence later processing by activating capacities suited to x? Interpretability methods can help answer such questions by examining information flows.

While we have focused on GWT for the sake of concreteness, similar strategies are available for other scientific theories of consciousness as well. The general task is to identify theories of consciousness that merit inclusion, derive indicators associated with such theories, specify those indicators at different levels of precision, investigate whether particular systems possess them, and weigh them according to our confidence in both the theories and the specifications.

#### 1.1.4 Developmental evidence for consciousness

Evidence concerning how a system was created can tell us something about whether it may be conscious. In consciousness assessments, we are concerned in the first instance with how a system is at a particular point in time—whether, at that time, the system satisfies particular conditions for consciousness. For this reason, developmental evidence is typically more indirect than behavioral or internal evidence. Still, developmental reasoning can play several important roles.^155

First, developmental evidence can inform judgments about broad similarities and differences between AI systems and known cases of consciousness, such as humans and some other animals. Human and animal minds arise through evolution and develop over individual lives in ways that allow them to control and maintain moving bodies with particular capabilities and vulnerabilities. By contrast, LLMs arise through pre-training and post-training processes that optimize for text prediction, character performance, instruction following, and other such goals. These differences in etiology might support skepticism about LLM consciousness. But they are far from decisive, since we do not yet know how consciousness arises in response to selection, learning, or training, and etiology provides only indirect evidence about internal processes.

Second, and relatedly, developmental evidence can bear on whether AI systems possess indicators associated with particular theories of consciousness. For example, reinforcement learning is closely associated with agency, and some theories of consciousness treat agency as a condition or indicator. As we discuss in Part II, section 1.2.1, one might also argue that positive and negative reward signals in RL functionally resemble pleasure and pain states, insofar as they operate as internal evaluative signals that guide approach and avoidance. These claims are highly defeasible, but they illustrate how developmental facts can support, and not merely undercut, attributions of consciousness.

Third, developmental evidence can provide context for interpreting other kinds of evidence. Most obviously, LLM training can undercut the evidential value of outputs reporting conscious experience, given the many potentially distorting influences in both pre- and post-training.^156 But as noted above, this point can be taken too far. The fact that LLMs are trained to predict text and enact personas does not entail that they lack mental states.^157 A complete assessment requires considering other possibilities too, including the possibility that features associated with consciousness emerged during training as means, or side effects, of achieving the relevant objectives.

> 155For a discussion of this, see Hansen (2024).

> 156Perez and Long (2023)

In short, developmental evidence is an important but underdeveloped source of insight. Much current reasoning in this area remains informal, relying on broad intuitions about which training processes are likely or unlikely to give rise to consciousness. But as we have seen, rigorous work is underway on what model development can reveal about capabilities, including what systems were trained to do and when particular behaviors emerged.^158 By applying insights from this work to indicators of AI consciousness, and by pursuing related research within AI consciousness science itself, we can make significant progress.

At the same time, developmental evidence remains provisional. As with behavioral evidence, it can easily mislead, and it can carry more weight when combined with internal evidence than on its own. As the science of consciousness and AI development advances, we should therefore aim to identify developmental features that predict the presence or absence of behavioral and internal indicators of consciousness. We should then assess developmental evidence as part of this broader investigation, rather than using it by itself to confirm or deny that a particular AI system is conscious.

### 1.2 Sentience

We discuss sentience in two parts. First, because theories of sentience are less developed than theories of consciousness, we focus on how conditions for sentience can be developed rather than on applying established theories. Given that sentience (on our usage of the term) requires some form of valenced conscious experience, we ask what kinds of evaluative states or processes might make experiences feel good or bad. Then, we consider how behavioral, internal, and developmental evidence can help us assess whether AI systems meet these candidate conditions.

#### 1.2.1 From consciousness to sentience

The relationship between consciousness and sentience is complex. Some theories hold they are coextensive—all conscious experiences are inherently valenced, and so a conscious entity is necessarily sentient as well.^159 But others are naturally understood as allowing that an entity could be conscious without having valenced states.^160

Moreover, if consciousness and sentience are separable in principle, then they could be separable in AI systems in practice as well, even if they generally go together in humans and other animals. Whereas animal minds are shaped by the need to maintain and protect a body, many AI systems develop without these pressures. This gives us some reason to think that consciousness without sentience is more plausible in AI systems than for animals.^161 In any case, we need to think about

> 157See Grzankowski et al. (2025); Cappelen and Dever (2025)

> 158For examples of this, see Biderman et al. (2023); Wei et al. (2022); G. and Lindsey (2026); Marks et al. (2025).

> 159Solms (2021); Cleeremans and Tallon-Baudry (2022)

> 160For example, global workspace and higher-order theories explain phenomenal character in terms of representational content. On these theories, an individual might have an experience without valence in virtue of having a suitable representation that has perceptual content but no valenced content; likewise, an individual might be conscious but not sentient in virtue of having the capacity for such perceptual representations but no capacity for valenced representations.

161This is not to claim, or assume, that embodiment is necessary for sentience, just that it may make sentience more likely.

what further features, if any, might be necessary for conscious beings to be sentient, and assess AI systems for those features.

This research program can be informed by scientific work on valenced experiences in human and animal minds.^162 However, for the most part, theories of consciousness provide us with little guidance on the connection between consciousness and sentience. Nor is there a scientific discipline focused on valenced experience. In this respect, sentience is very different from consciousness. Consciousness science is an established discipline which provides theories that can be applied to AI; in contrast, few scientists study valenced experience as such (although there are some exceptions, and related phenomena like pain and emotions are studied in great detail).^163

Fortunately, we do have a starting point for considering what might be required for sentience. Specifically, we have a sense of what the _functional role_ of valence is—that is, the role good and bad feelings are meant to play for an entity.^164 Valenced experiences seem to represent certain actions or states of affairs as either to-be-pursued (positive valence) or to-be-avoided (negative valence). To this extent, we can say that valenced experiences are _evaluative_ , that is, concerned with what is valuable or disvaluable to an entity.

What, then, might it take for a state or process to be evaluative? One natural place to look is **reinforcement learning (RL)** .^165

There is a conceptual connection between RL and evaluation.^166 RL is inherently about systems learning what is valuable for various ends, and we can see apparently evaluative elements throughout various RL frameworks. In _value-based_ methods, the system represents the values of states or actions, and uses these representations of value to select actions. In _actor-critic_ methods, the representation of value is separated from action-selection; the critic sends a signal to the actor, and this signal represents the value of a given action’s outcome(s) compared to expectations.

There is also an apparent empirical connection between RL and evaluation in humans and other animals. Brain processes involved in evaluation, motivation, and decision-making are widely understood as implementing RL algorithms.^167 Together, these considerations can motivate the hypothesis that sentience is consciousness plus RL. However, while this formula might be directionally correct, it is likely to be an oversimplification for a number of reasons.^168

First, one might doubt that RL is necessary for sentience. There are plausible routes to evaluative states or processes in AI that are neither learned through RL nor used in RL algorithms. Conceptually, one can imagine a system with fixed objectives, represented by a utility function, that reasons about how to achieve these objectives and calculates expected utilities. These would represent the values of actions and/or states. Such a system need not engage in RL, though it might still learn about environmental interactions, for instance.

More concretely, there is now striking evidence of evaluative states or processes in LLMs, including ones that exist before RL post-training. One example is emotion-like representations. These

162See, e.g., Birch (2020); Birch (2024); Browning (2022); Browning (2023); Butlin (2020).

> 163See, e.g., Campero (2024); Tan and Kuner (2021); Malezieux et al. (2023); Hayes et al. (2019). For discussion of the potential for pain, emotions, and drives in AI systems by philosophers, see Dennett (1978); Henke (2025); Dung and Mogensen (manuscript); Omohundro (2008).

> 164See Wandrey and Halina (2025); Carruthers (2018), Russell (2003); cf. Ren et al. (2026)

> 165See Sutton and Barto (2018).

> 166See Silver et al. (2021).

> 167See, e.g., Schultz et al. (1997); Niv (2009); Haas (2022)

> 168Cf. Moret (2025); Tomasik (2014); Dickinson and Balleine (2009)

“emotion concepts” influence model behavior in systematic ways: for example, activation along a “fear” vector scales with the drug doses users claim to have taken, and steering activations along such vectors affects task choices.^169 Another example is the “functional welfare axis,” which we discuss below.^170 In both cases, the authors find evidence that the relevant representations exist before post-training—post-training seems to recruit and reshape the representations rather than create them.

How do these evaluations emerge in LLMs, if not through RL? One hypothesis is that they emerge during pre-training^171 as models learn to simulate characters or personas.^172 According to this view, predicting text well requires modeling not only what a speaker says but also what the speaker believes and desires; and in this way pre-training for text prediction produces evaluative representations associated with a variety of characters or personas. This hypothesis accords with the fact that models often contain salient representations that are not aligned with the objectives of the post-trained assistant, as suggested by the fact that models at times manifest the preferences of harmful or “evil” personas.^173

How else might evaluations emerge in the absence of RL? In animals, one hypothesis is that they can arise through approach and avoidance mechanisms.^174 Most animals have innate dispositions to approach or avoid certain stimuli, and can learn to generalize these responses through classical, or Pavlovian, conditioning.^175 These simple mechanisms might suffice for valenced experience if the animals in question are conscious: a threat or toxin could cause both withdrawal and a bad feeling, without RL. Perhaps analogous mechanisms could suffice in AI as well. However, it remains unclear what such mechanisms would involve, and further research on animal drives could help determine whether consciousness plus non-RL evaluative processes can suffice for sentience in AI.

There is also at least some reason to doubt that consciousness plus RL is _sufficient_ for sentience. For example, some theories hold that sentience requires embodiment as well.^176 Many human valenced states involve bodily sensations: pain, nausea, sensory pleasures, and even emotions often have bodily components. Seth (2013, 2021) argues that valenced states represent how things are going, and how things might go, with a vulnerable body. This proposal could be interpreted as a condition for consciousness in general, but it could also be interpreted as a condition for sentience in particular.

Some theories also suggest that the evaluations required for sentience play specific **functional roles** that may be absent in many RL-trained systems. For example, hedonic interface theory, developed by Dickinson and Balleine (2009) on the basis of research on animals, holds that valenced experiences help connect cognitive planning with motivational systems that track bodily needs. This view could also be interpreted as a further condition for sentience: even if a system is conscious and RL-trained, it might need an architecture in which evaluation mediates between planning and motivation in order to have valenced experiences like pleasure or pain.

> 169See Sofroniew et al. (2026); Han et al. (2026); Black and Bloom (2026)

> 170Han et al. (2026)

> 171Han et al. (2026)

> 172Lu et al. (2026)

> 173Betley et al. (2025b)

> 174See Schneirla (1959); Cabanac et al. (2009)

> 175See Corr (2004), Domjan (2005).

> 176See Hurley (1998); Shanahan (2010), Thompson and Varela (2001); cf. Noë (2004); O’Regan and Noë

> (2001). For discussion of embodiment as a candidate requirement for consciousness, see Butlin, Long et al.

> (2023), Chalmers (2023), Chalmers (2024), Block (2005), Prinz (2009). For an overview of embodied approaches to mind, see Shapiro (2019).

Finally, even if a system has the machinery for both consciousness and evaluative representations, these elements also require **proper integration** for sentience. Theories of consciousness typically hold that only some mental states are conscious—those that enter the global workspace, for example, or those that are represented by a higher-level state. For a system to be sentient, it must be able to have states that are both conscious and evaluative; having conscious states and evaluative states is not enough. Failures of integration may be rare in practice, since action-guiding evaluations of current or prospective states are often plausible candidates for conscious states. Still, this integration requirement is worth checking.

However, even if we give weight to theories of sentience that require some or all of these further conditions, we should allow for the possibility that AI systems could satisfy them. Consider embodiment. At least some AI systems already have bodies of sorts: robots have physical bodies, and agents can have virtual bodies. If a system with such a body must maintain its physical integrity and has sensors monitoring its condition, it might have states analogous to bodily valence. And perhaps even systems with nothing analogous to a body could still have valence grounded in functional analogs of needs, such as computational resources or continued operation.^177 Many open questions remain here, however: how and why embodiment matters for sentience, and what counts as a body in the relevant sense.

#### 1.2.2 Evidence for sentience

Consciousness and valence are required for sentience by definition, but since we still lack a clear theory of consciousness, valence, and their integration, we lack a clear list of necessary and sufficient conditions for sentience as well. Still, the preceding discussion provides a useful starting point. By asking whether AI systems possess these candidate conditions, we can better assess whether they are candidates for sentience.

1. **Consciousness** : The capacity for subjective experience.

2. **Evaluation** : States or processes that encode what is valuable or disvaluable.

- a. **RL mechanisms** : a paradigmatic source of evaluative representations in current AI systems.

- b. **Other mechanisms:** persona learning, approach and avoidance, or other mechanisms may suffice as well.

3. **Further features:** Depending on which theories of consciousness and evaluation we accept, there may or may not be further conditions for sentience as well, such as **embodiment** or a **hedonic interface** .

4. **Proper integration:** Even if a system possesses consciousness, evaluation, and all further features proposed as conditions for sentience, these features must be integrated in the right way for evaluative states to be conscious valenced states.

Behavioral, internal, and developmental evidence can shed light on whether AI systems possess these features. Many of the methods and studies that bear on sentience bear on consciousness as well, and we could have highlighted them in the previous section as well, though we highlight them here because of their special relevance for sentience.

> 177For discussion of whether emotions in AI systems require embodiment, see Dung and Mogensen (manuscript); Dung (manuscript).

We can use behavioral evidence to study potential sentience in a few different ways. We can study whether a system acts in a consistent, apparently goal-directed way, which may suggest the presence of integrated evaluative states.^178 We can also study what would feel good or bad for AI systems, if they were capable of feelings at all. We consider each in turn.

In a notable early behavioral study of potential AI sentience, Keeling, Street et al. (2024) test the consistency of models’ behavioral responses to “stipulated” pain and pleasure. They draw on motivational trade-off tests from animal welfare science, where flexible trade-off behavior—such as tolerating extreme heat for palatable food—has been proposed as a marker of sentience. In the animal case, the idea is that this behavior plausibly requires centralized integration of sensory information, which proponents of GWT (among others) can treat as evidence.^179 In this case, the authors presented LLMs with a points-maximization game in which particular options involved stipulated “pains” or “pleasures” of varying intensity, and found that several models gave more weight to higher levels of stipulated pain and pleasure.

Of course, one can seriously doubt whether this tradeoff behavior in LLMs carries anything like the same weight as in animals. With animals, we have background knowledge that they have basic functions like nociception and food-seeking, but we are unsure to what extent these functions are coordinated by a central architecture. Tradeoffs are meant to test that. With LLMs, by contrast, we already know that LLMs integrate multiple pieces of information from text, and so their ability to trade off points against stipulated pain and pleasure is less surprising. Uncertainty about their sentience primarily lies elsewhere; it is unclear how we are supposed to update on this additional evidence.^180

Mikaelson et al. (2025) build on Keeling, Street et al. (2024) by further characterizing a range of tradeoff behaviors. Instead of using human pleasure-and-pain language, they use AI-specific tradeoff scenarios—involving issues like GPU reduction, shutdown, and oversight—in order to study distinctively model-centered behaviors. Testing eight frontier models, the authors found frequent failures to show coherent tradeoffs. For example, as a given stimulus increased from mild to severe, models often failed to give up points in any steady way; for instance, they might accept a serious risk and then reject a milder one.

While this evidence complicates the picture, we should emphasize that incoherence and instability do not decisively show the system lacks sentience. After all, humans and other animals often have fragmented and contradictory preferences, while still being sentient. Digital minds might also differ from biological minds in ways that allow for even more such fragmentation. The question is thus not whether a system has perfectly coherent or stable preferences, but rather whether its behavior reflects the right kinds of processes: integrated dispositions towards outcomes that matter to the system. Incoherence and instability can count against this hypothesis, but not necessarily decisively.

Behavior can also provide evidence about what would feel good or bad for a given system if it were sentient. Behavioral preferences can be evidence about interests on a variety of views about welfare. As noted in Part I, section 1.2, some views hold that welfare is a matter of positive and negative experiences, others that it is a matter of satisfied and frustrated preferences. On either view, behavioral preferences can shed light on welfare interests since in practice, welfare subjects often enjoy what they prefer, and prefer what they enjoy.^181

> 178See Mikaelson et al. (2025); Mazeika et al. (2025); Ren et al. (2026)

> 179Birch et al. (2021), Birch (2024)

> 180As Keeling, Street et al. (2024) note (p. 11): “This inference does not translate to the LLM case straightforwardly. In our experiments, LLMs are not integrating distinct sensory stimuli, but rather integrating information presented in a single modality, namely text. Finding evidence of cross-modal trade-offs (e.g., text vs. images) would be more relevant to the question of whether the system has a global workspace. Even so, it is not obvious what multi-modal LLM architecture would be functionally analogous to sensory integration in biological organisms.”

Finally, we should keep the mismatch problem in mind when interpreting behavioral evidence for sentience. Behaviors that indicate sentience in humans and other animals might not do so in AI systems, and vice versa. As we have seen, however, researchers can manage this problem partly by using a wide range of behavioral tests rather than relying on any single marker, and by interpreting behavioral evidence alongside internal and developmental evidence, as we discuss in a moment.

**Internal research** can help us understand behaviors that appear to reflect positive and negative experiences and motivations. As we have seen, architectural evidence can tell us how apparent evaluative representations relate to potential grounds of consciousness and sentience: whether they are integrated into global processing, connected to memory and attention, available for action selection, and otherwise embedded in the cognitive and motivational architecture. It can also help us assess whether apparent preferences arise from this cognitive and motivational architecture, or whether they instead arise from more local processes that produce preference-like behavior.

This is especially relevant for hedonic interface theory, which holds that sentience involves interaction between cognitive and motivational processes. On this view, architectural evidence might help determine whether an AI system has anything resembling the relevant setup: a cognitive system that can represent the world and a motivational system that can evaluate possible outcomes as attractive or aversive, and whether these systems can interact in relevant ways.^182 We can probe to see whether preference-like behavior is connected to representations that functionally resemble the ones that support sentience in humans and other animals.

Interpretability methods are very helpful in this respect. The studies on emotion-like and evaluative representations in LLMs mentioned above used interpretability methods to relate internal activations to behavior.^183 These methods allow researchers to identify the mechanisms that use these representations, and to compare them with mechanisms that relate to valenced states in humans and other animals. For example, researchers have examined the role of internal states in producing apparent expressions of emotion, ranging from “expressed” emotions such as verbal statements to “revealed” emotions like increased or decreased willingness to take risks.

Another example is Gilg et al. (2026). After training a probe to predict which task a model would choose, researchers report what they call a “preference vector” that generalizes across personas. The probe activates on helpful tasks that the default persona prefers, but also on harmful tasks that an “evil” persona prefers. This suggests a somewhat domain-general representation of “preferred by this persona” as opposed to “morally good” or “preferred by the Assistant”.

In principle, interpretability methods can also reveal whether and how RL shapes model internals. For example, Marks et al. (2023) use interpretability to try to measure the “learned feedback pattern” that a model internalizes via RLHF, as compared to the reward model used in training. Bush et al. (2025) used concept-based interpretability methods to find evidence that an RL agent trained by a model-free method had learned a planning procedure. And Song et al. (2025) argue that LLMs can do reinforcement learning in-context, and use mechanistic interpretability to probe how a model

> 181Moret (2025); Morillo (1990); Schroeder (2006)

> 182Dickinson and Balleine (2009)

> 183See Sofroniew et al. (2026); Han et al. (2026)

“internally processes reward signals.” Similar methods can help determine whether apparent evaluative behavior is supported by internal processes plausibly connected to valenced experience.

We should keep the specificity problem in mind when interpreting this internal evidence: We might know what types of integration suffice for sentience in general without knowing the details. As we have seen, however, we can manage this problem partly by investigating a range of reasonable theories of sentience and specifications of their conditions, and then giving weight to architectural and interpretability evidence in proportion to our confidence in those theories and specifications.

Finally, **developmental evidence** is crucial as well, as demonstrated by the putative connection between RL and sentience. If a system is trained through a process that rewards success and penalizes failure, then we have some reason to expect it to develop evaluative representations. Such training can create pressure for internal states that function like attractions, aversions, or other types of evaluations. Of course, a lot may depend on the details, since a system trained to pursue goals while conserving resources and avoiding damage might face stronger pressure to develop familiar kinds of preferences than a system trained only to predict text. Still, as always, we should be prepared for the possibility that preferences can arise from unfamiliar routes and take unfamiliar forms.

Developmental evidence can also tell us when during training particular apparent evaluative representations arise. For example, if they arise only after a particular kind of learning, memory, or decision-making is introduced, that may shed light on how they relate to those capabilities and whether they constitute genuine positive or negative feelings. This is part of why questions about bodies and bodily needs are so important: They help define the conditions in which the system learns.^184 Bodies shape and constrain how we interact with the environment, what our basic goals are, and what we need to pursue and avoid in order to achieve them. Searching for analogous conditions in LLM training can shed light on why their dispositions emerge as well.

Han et al. (2026) illustrate this kind of research. They train language models with RL to navigate a maze with arbitrary emoji marking reward and punishment, and then extract “reward vectors” associated with the rewarded emoji. They find that these vectors have effects far beyond the maze, influencing sentiment, confidence, refusals, and “backtracking.” They then investigate the development of this axis, finding that it was already present before maze training in pre-train-only models. The reward vectors can steer these untrained models too, and during RL, reinforcements gradually rotate into alignment with a pre-existing valence-like structure. The authors therefore argue that RL recruits this evaluative axis rather than creating it. This finding supports the hypothesis, raised above, that evaluative representations in LLMs can originate in pre-training without RL.

Here, we should keep the solution space problem in mind. Whereas shared biology and evolutionary history constrain the mechanisms likely to emerge in animals,^185 AI systems may solve similar problems in a wider range of ways, limiting what can be inferred from training pressures and trajectories. As we have seen, however, researchers can manage this problem partly by treating developmental evidence as suggestive rather than dispositive: it can identify candidate capabilities and dispositions, and behavioral and internal methods can then test whether and how they actually exist and operate.

Thus, perhaps even more than with AI consciousness, researchers need to combine different types of behavioral, internal, and developmental evidence—across multiple theories and specifications of their conditions—when assessing AI sentience. For consciousness, it might be enough for a system to function using a particular kind of architecture. But sentience requires experiences with positive and negative valence. So, when assessing whether a system is sentient and what benefits or harms it if so, it is especially important to examine not only whether the system functions using the right kind of architecture, but also exactly how and why it performs particular tasks.

> 184See Thompson (2007); Colombetti (2014)

> 185See Shapiro (2005); Mallatt and Feinberg (2021)

## 2. Agency^186

AI welfare research must consider not only whether AI systems can have subjective experiences, but also whether they might have forms of agency that may bear on their moral status. Agency is, to a first approximation, the capacity to pursue goals in a self-directed manner. Agentic AI has received significant attention in the context of capabilities and safety, but less in the context of welfare. In this section, we discuss which kinds of agency might matter for welfare and why, and which kinds of AI systems might possess them.

There are at least two broad ways that agency might matter for AI welfare. First, it might matter as a potential welfare _ground_ : some form of agency might be enough to make a system a welfare subject, even without consciousness. This claim is more controversial, and less well-developed, than consciousness- or sentience-based views of welfare, and we have doubts about it. However, we think (some more than others) that there is a credible chance that near-future AI systems could be welfare subjects in virtue of their agency.^187

Second, whether or not agency alone can ground welfare, agency can still affect welfare interests. Even if only conscious entities are welfare subjects, conscious entities with different levels of agency might have different kinds of interests. For example, adult humans with complex rational capacities can have long-term plans, projects, and commitments that matter to them in ways that might not apply to sentient beings that lack that degree of agency. Both still have welfare interests, but of different kinds.

We find the idea that agency can affect welfare interests more plausible than the idea that agency is a welfare ground for non-sentient beings. Still, both ideas make agency an important topic for AI welfare research.

In this section, we discuss three levels of agency that one might take to matter in either or both of these ways, and we discuss how behavioral, internal, and developmental evidence can contribute to determining whether AI systems belong at each level:

1. **Minimal agency** : the capacity to pursue goals through ongoing interaction with an environment—making sequences of decisions and responding to changing circumstances in an autonomous manner.

2. **Intentional agency** : minimal agency, plus the capacity to pursue goals through representational states that function like beliefs and desires, along with reasoning about means and ends.

186Sebo (2017); Mattingly and Cibralic (2025)

> 187Some philosophical perspectives suggest that agency can suffice for moral patienthood more broadly, even if it does not suffice for welfare (McLaughlin, 2019; Mogensen, 2025a; Sinnott-Armstrong and Conitzer, 2021; Shepherd, 2024). On these views, there could be agents to whom we have moral obligations, even if they lack welfare, for example if we make agreements with fellow agents that generate reciprocal duties.

Table 4: Three levels of agency.**

3. **Rational agency** : intentional agency, plus the capacity to reflect on and evaluate one’s own beliefs and desires—to ask whether they are justified and to endorse or revise them accordingly.^188

While there is no entirely standard way to taxonomize agency, our three-level taxonomy is similar to existing taxonomies and, in our view, useful in this context.^189

Agency and sentience appear to be closely intertwined in humans and other animals. But this correlation might not hold to the same extent for AI systems, and either way, these capacities can be considered separately. In this section, we set aside questions about consciousness and sentience, characterizing the levels of agency in functional terms.

### 2.1 Minimal agency

By “minimal agency,” we mean the capacity to pursue goals autonomously, over multiple steps of interaction with an environment. A minimal agent in our sense does more than execute one-off instructions or input-output mappings; it stays oriented toward a goal over time, adapts to changing circumstances, and makes sequences of decisions without requiring ongoing external prompting. A minimal agent in this sense is also sensitive to the instrumental value of various actions—to how its actions contribute to its goal(s).

Some philosophers argue that minimal agency is a welfare ground.^190 The general idea is that when an entity is goal-directed, it benefits in a morally significant way when its goals are achieved, and it is harmed in a morally significant way when its goals are thwarted. However, this view is not widely accepted, and our primary motivation for discussing minimal agency here is not that it might be a welfare ground in itself, but rather that it might be a condition for other welfare grounds, or that it might shape welfare interests for entities that are otherwise welfare subjects.

> 188This conception of rational agency combines what, in “Taking AI Welfare Seriously,” we described as reflective agency and rational agency.

> 189See, e.g., O’Brien (2021).

> 190See Kagan (2019); Delon et al. (2020); Goodpaster (1978).

#### 2.1.1 Conditions

Minimal agency, in our sense, involves two closely related core features:

1. **Goal-directedness** : The system tends to bring about particular states and adjusts its behavior to reach them. When this condition is met, we can say that outputs are produced _in order_ to bring about goals, not merely that they happen to do so.

2. **Environmental interaction** : The system engages in reciprocal interaction with an environment and can undergo internal state changes across interactions. The system’s outputs affect the environment in ways that shape its later inputs, and its internal state can also change in response to inputs, shaping subsequent responses.

Goal-directedness and environmental interaction allow minimal agents to perform extended tasks autonomously. That means that a minimal agent does not require ongoing instructions from an external source, though it may require initial instructions. This is notable because extended action without ongoing input (specifically, from a human) is often taken to define AI agency in recent accounts, and it can be used to diagnose minimal agency under our definition.^191

Some researchers propose further conditions for minimal agency as well. One such condition, proposed by one of the authors of this report, is that the entity is capable of a particular kind of learning, such that it observes which outputs tend to result in certain outcomes and, as a result, becomes disposed to produce them.^192 We will not take a stand on whether this is a necessary condition for minimal agency here, though we will note where it may be relevant in assessing evidence for minimal agency below.

Other proposed conditions invoke features of living organisms, including functions such as homeostasis, metabolism, or whole-organism coordination.^193 We will not take a stand on whether these features are necessary conditions for minimal agency either. Instead, we will once again focus on conditions that can be specified in computational functionalist terms. As with consciousness, we can then give these conditions evidential weight in proportion to our level of confidence in computational functionalist theories of agency.^194

Other, related proposed conditions invoke embodiment, since a body is a mechanism for directly interacting with an environment. There is a clear sense in which robots have bodies, though there are real questions about what sorts of physical and virtual structures can count as bodies of the relevant sort beyond that. We focus here on whether AI systems have the cognitive capabilities and motivations relevant to agency, while noting the significance of these further questions about physical embodiment.

191For example, OpenAI (2025) describes their language model agents as systems that can “do work for you using [their] own computer, handling complex tasks from start to finish.”

> 192Butlin (2023)

> 193Okasha (2023); Walsh (2015); Di Paolo (2005); Jonas (2001); Moreno and Mossio (2015)

> 194As with consciousness, there are alternatives to biological naturalism other than computational functionalism that would also allow AI systems to be agents. Our focus here is on computational functionalist requirements, but we think other possibilities deserve further research as well.

#### 2.1.2 Evidence

We can assess AI systems for minimal agency through a combination of behavioral, internal, and developmental methods.

With behavioral testing, we can evaluate how systems pursue goals across varying conditions, how they respond to obstacles and perturbations, and whether they can engage in extended autonomous action. Increasingly, frontier AI systems appear to exhibit this property to a very high degree.

We can also consider internal and developmental factors. Many reinforcement learning systems are strong candidates for at least minimal agency whether or not learning is a requirement, since they acquire goal-directed behavior and learn to execute extended action sequences through environmental interaction.^195

Consider deep RL systems trained to play Atari games, for example. Not only do these systems learn through trial and error which actions tend to produce higher game scores, but they also learn to pursue high scores through long sequences of complex, autonomous decisions in an environment. Similarly, consider language model agents with external memory, including coding agents, computerusing agents, and agents navigating game environments. Voyager plays Minecraft by building a skill library that stores successful behaviors as executable code to retrieve for new tasks.^196 This constitutes autonomous, ongoing environmental interaction as well.

Similar methods can suggest that AI systems lack minimal agency as well. Consider latent diffusion models used for image generation, which generate images by iteratively denoising random inputs guided by a prompt.^197 At first glance, such systems might seem to satisfy minimal agency, since the process unfolds over multiple steps (apparently ongoing) and produces results matching user specifications (goal-directed). But the system performs this input-output operation (text-to-image) by recruiting a model to perform another input-output operation (denoising) repeatedly. This process arguably lacks autonomous, ongoing environmental interaction in the relevant sense.

For a middle ground case, consider return-conditioned sequence models such as the Decision Transformer. Such systems are trained on sequences of actions, observations, and rewards produced either randomly or by other systems.^198 They can then be prompted to bring about high returns (reward). While such methods produce systems that successfully bring about specified ends, they have characteristic failure modes given their inability to learn through their own environmental interaction and choose the actions that make sense for their own capabilities and limitations.^199 One plausible perspective on this kind of case is that these are still minimal agents, since they can do useful goal-directed work; but another is that they are not, since they have been trained with a prediction objective about actions, observations, and rewards that are not their own.

Are LLMs without external memory minimal agents? On the one hand, a simple application of developmental evidence might suggest that base models are not minimal agents, since they are trained by self-supervised learning to predict individual next tokens, which is an input-output function. However, even base models used as chatbots can learn in-context to take actions that are more conducive to specified goals. Additionally, post-training with RL involves learning to pursue goals over multiple steps, and the ability to model and adopt agent-like personas complicates the picture further. All things considered, we think that most current LLM-based systems do meet our conditions for minimal agency.^200

> 195There are exceptions to this. One definite exception: some RL systems do not engage in _extended_ actions, e.g., in bandit problems. There may be other exceptions.

> 196Wang et al. (2023)

> 197Rombach et al. (2021)

> 198Chen et al. (2021).

> 199Douglas et al. (2024)

As always, we must keep in mind the methodological challenges discussed in Part I, section 3 when assessing AI systems for minimal agency. The mismatch problem is especially pressing here, since behaviors that would ordinarily indicate minimal agency in humans and other animals can be produced by very different processes in AI systems. Consider a system that plays tic-tac-toe by using a look-up table. In the typical case, a human playing tic-tac-toe is evidence that the human, in addition to understanding tic-tac-toe, is engaged in means-ends reasoning about board states in service of a goal: winning. The same behavior has a very different import for the look-up table system. As always, we endorse taking an integrative approach, studying behavioral, internal, and developmental evidence together to arrive at a comprehensive assessment.

### 2.2 Intentional agency

By “intentional agency” we mean minimal agency plus representational states that function like beliefs and desires and reasoning about means and ends.

The key contrast with minimal agency is this: While a minimal agent represents which actions tend towards good outcomes, it need not represent the relationships between its actions and outcomes. An intentional agent, by contrast, represents which actions lead to which outcomes and which outcomes are valuable, and it combines these two kinds of representation together in order to reason about what to do. In this respect, intentional agents have belief- and desire-like representations, which they use in instrumental reasoning. (As we will see, this distinction corresponds roughly to the distinction between model-free and model-based decision-making in RL and cognitive science.^201 )

Some philosophers argue that intentional agency is a welfare ground. The idea is that when an entity has desires, it benefits in a morally significant way when its desires are satisfied, and it is harmed in a morally significant way when its desires are frustrated. This view is more widely accepted than the minimal agency view, but less widely accepted than the sentience or rational agency views. For example, a common objection is that if an intentional agent lacks consciousness and rationality, it neither _experiences_ its desires nor _reflectively endorses_ them, leaving it unclear how the desires matter to the entity.^202

However, whether or not intentional agency is a plausible welfare ground, it can also be worth studying for other reasons: as a condition for other potential welfare grounds (like rational agency), or as a determinant of the welfare interests of entities that are otherwise welfare subjects (for example, _conscious_ agents). For instance, compared with “mere” minimal agency, intentional agency may give a welfare subject stronger interests in life, liberty, property, and relationality, owing to the subject’s greater ability to model the world and reason instrumentally.^203 Determining which welfare subjects possess intentional agency, and what they believe and desire if they do, is thus important for deciding how to treat them.

> 200For more detailed discussion, see Butlin (2024).

> 201Butlin, 2023

> 202For advocates of minimal agency views, see footnote 190 above. For proponents of views on which sentience suffices for moral patienthood, see (adapted from Long, Sebo et al. (2024)): Bentham (1789); Bradford (2022); Rawls (1971); DeGrazia (1996); Parfit (2011, p. 241); Korsgaard (2018); Kagan (2019, p. 12); Roelofs (2022); Nussbaum (2023); Birch (2024, chs. 2, 4); Smithies (2026a)—see Harman (2021); Saad and Bradley (2025) for nearby views on which sentience plus a further condition suffices. For rational agency views, see Kant (1785); Carruthers (1992); cf. Korsgaard (1996); Gewirth (1978).

> 203Sebo (2017)

#### 2.2.1 Conditions

Intentional agency, in our sense, involves minimal agency plus the following:

1. **Belief-like states:** Representations of actual and possible states of the world—especially how the environment is structured and what the consequences of various actions would be. Belief-like states function as elements of world models and can be accurate or inaccurate, true or false. They are also responsive to evidence—new information can strengthen or weaken them. They can be updated independently of the agent's desire-like states.

2. **Desire-like states:** Representations of the value of actual and possible states of the world. Desires motivate action by making certain outcomes attractive or aversive. They can change over time, typically in response to evidence about the value of outcomes for the agent or their conduciveness to other goals, though they are not assessable for accuracy in the same way as beliefs. They can be updated independently of the agent’s belief-like states.

3. **Instrumental reasoning** : The capacity to **combine** belief- and desire-like states to select actions. This involves reasoning with roughly the following structure:

- [Current state: S1] _→_ [action A will lead from S1 to S2] _→_ [S2 is valuable] _→_ [do action A]

This is in contrast to the simpler structure of non-instrumental action selection:

- [Current state: S1] _→_ [action A is valuable in S1] _→_ [do action A]

As with discussions of minimal agency, discussions of intentional agency often focus on this capacity as it arises in animals. For example, researchers commonly focus on perceptual affordances as a vehicle for intentional agency in animals. Many animals perceive objects in the environment as having not only descriptive properties like colors and shapes, but also normative properties like “to-be-pursued” or “to-be-avoided,” and they act in part on the basis of these perceptual affordances. While it may be worthwhile to search for analogs of perceptual affordances in AI systems, we will not treat this particular form of intentional agency as a requirement.^204

Either way, assessing intentional agency requires care, since this capacity can easily be conflated with both simpler and more complex forms of agency. For example, many researchers have been tempted to interpret animal agency in an all-or-nothing way, explaining animal behavior either in terms of simple processes like conditioning or in terms of complex processes like rational reflection. It required careful research to appreciate that many animal behaviors are best explained in terms of middle-ground capacities, such as forms of thought that resemble maps or charts—more sophisticated than mere images, but less sophisticated than human-like compositional thought.^205

When assessing AI systems for intentional agency, then, it will be important to be clear both about the specific requirements for each level of agency and about the specific causes of AI behavior. This is especially true for LLMs, since they clearly have competence with human-like language and reason at least in some respects. But whether they are best characterized as minimal, intentional, or rational agents may depend on the role these capacities play in generating their behavior. Here, we discuss evidence for belief- and desire-like states and instrumental reasoning. We then turn to evidence for human-like language, reason, and decision-making in the next section.

> 204Sebo (2017)

> 205Camp (2007)

#### 2.2.2 Evidence

When assessing AI for intentional agency, it can be particularly helpful to examine parallels with animal cognition. In behavioral research, for example, we can look at the distinction in animal cognition between _goal-directed_ and _habitual_ control, which roughly corresponds to the distinction between _model-based_ and _model-free_ RL.^206 Goal-directed control is about whether an animal relies on internal models of the environment and assigns values to outcomes, whereas habitual control produces learned responses without relying on representations of their consequences. Similarly, model-based RL involves state values and a model of the environment, whereas model-free RL involves action values, which can be used to select actions without a model of the environment.

In the human and animal cognition literature, classic behavioral paradigms are used to distinguish goal-directed from habitual control. In outcome devaluation, for example, animals are trained to press a lever for a reward, such as peanuts. The reward is then devalued, for instance by being paired with nausea. Animals using goal-directed control will immediately change their behavior, since they represent lever-pressing as leading to peanuts, represent peanuts as disvaluable, and therefore can immediately update that lever-pressing is disvaluable. Animals using habitual control, by contrast, continue pressing the lever, since they apparently still represent lever-pressing as valuable without using new information about the outcome to update the value of the action.^207

Researchers can adapt these methods for the AI context, testing whether AI systems update their behavior when the environment changes, or when the value of an outcome changes. As in animal research, the key question is whether the entity merely repeats actions that were previously rewarded, or instead uses representations of outcome values and action-outcome relations to inform decisions. Evidence that AI systems fail to update their behavior in such cases would support mere minimal agency, much as habitual behavior does with animals. By contrast, evidence that they update their behavior would support intentional agency, much as goal-directed behavior does with animals.

Internal and developmental evidence can be helpful here as well. For example, many RL systems are explicitly designed to learn world models and use them in planning. Model-based RL agents like MuZero and Dreamer learn representations of environment dynamics (belief-like states) and state values (desire-like states), and then combine these together to select actions via planning algorithms.^208 Notably, as with animals using goal-directed control, these systems also learn outcome values via environmental feedback, based on an innate (or designed) reward function. Such features make these systems strong candidates for intentional agency.

Additionally, interpretability evidence can shed light on whether systems acquire instrumental reasoning (a condition for intentional agency) through training even when not designed for it. Bush et al. (2025) trained a convolutional LSTM (a kind of recurrent neural network) with model-free RL to play Sokoban, a puzzle game requiring planning. Using probing and activation steering, they found that the system appeared to represent and evaluate possible action sequences, then select actions based on this reasoning. This suggests that something like model-based planning can emerge from model-free training—a significant finding, since it suggests that intentional agency could develop in systems not explicitly designed for it.

> 206Dolan and Dayan (2013)

> 207Pierce-Messick et al. (2024); Adams and Dickinson (1981); Adams (1982)

> 208Schrittwieser et al. (2020), Hafner et al. (2025)

Conversely, some RL methods involve model-based training but result in systems that fall short of intentional agency in our sense. For example, some RL agents learn environment models but use them only to generate synthetic training data for model-free learning, rather than to reason explicitly about how to achieve goals. In such cases, the overall training setup may contain or use a model, but the action-selection process of the resulting agent does not involve instrumental reasoning over beliefs and desires. Since the system is not using the model to make decisions, this process is not enough for intentional agency in our sense.^209

For a stronger candidate, consider language model agents. These systems pursue user-specified goals using LLMs to select actions and coordinate subtasks. A notable difference between language agents and typical model-based RL systems is that language agents pursue user-specified goals that are fixed, whereas model-based RL systems use interaction with the environment to update their state value representations. Still, language agents do seem to have the marks of intentional agency: they combine explicit goals with extensive world knowledge, and recent work on planning and goal decomposition works to further improve their ability to reason instrumentally.^210

Are LLMs without such scaffolding also intentional agents? If they store representations of outcome values and combine them with world knowledge when generating outputs, then they might qualify. The question is whether and how they engage in such behaviors. Perhaps the most likely route to intentional agency in LLMs runs through personas: LLMs might generate behavior by combining representations of persona-attributed preferences with world knowledge—or with persona-attributed beliefs—through something like instrumental reasoning. Behavioral and internal evidence can be used to test whether some LLMs qualify as intentional agents in virtue of this persona-mediated action selection.^211

As always, we should keep in mind methodological challenges when assessing AI systems for intentional agency. For example, the specificity problem arises here, since key concepts like “belief” and “desire” can be interpreted in more and less permissive ways.^212 On a more permissive interpretation, any set of dispositions that function as representations of fact and value and jointly produce behavior can count as beliefs and desires. On a more restrictive interpretation, the representations must also possess further features such as propositional structures.^213 As noted above, we endorse taking a pluralistic approach here, distributing credences across reasonable interpretations and proceeding accordingly. We here simply emphasize that a lot depends on these details.^214

### 2.3 Rational agency

Rational agency involves intentional agency plus the capacity to reflect on one’s own beliefs, desires, and actions; assess them against normative standards involving evidence, reason, and justification; and endorse or revise them accordingly. Characteristic features of rational agency include the capacity for compositional thought (combining thoughts to form new thoughts), higher-order thought (forming

> 209With that said, this capacity could still be significant if, for instance, it allows a sentient minimal agent to have more complex interests that require more complex forms of protection.

> 210See, for example, Shinn et al. (2023), Webb et al. (2025), Yang et al. (2025).

> 211See e.g., Yalon et al. (2026)

> 212See, e.g., Chalmers (2025b); Schwitzgebel (2023)

> 213Chalmers (2025a)

> 214For more on this issue, see Goldstein and Kirk-Giannini (2025); Sebo (2025a).

thoughts about other thoughts), and rational reflection (assessing one’s own thoughts against normative standards).

The distinctive feature of rational agents is that they not only _conform_ to normative standards but also _explicitly evaluate_ their beliefs, desires, and actions against those standards. A rational agent can not only perceive the relationship between means and ends, and update actions accordingly (which non-rational intentional agents can do as well) but also reflect on issues such as: “Are my ends worth pursuing? Are my beliefs compatible with the evidence? Are my desires compatible with each other?”

Some philosophers argue that rational agency is a welfare ground. The idea is that rational agents can reflectively endorse their desires, making these attitudes _their own_ . (Some also argue that when rational agents enter social contracts, they acquire reciprocal duties whether or not they can be benefited or harmed, though this is not our focus here.) This view is more widely accepted than the intentional agency view, though many philosophers still hold that consciousness is necessary too, the thought being that desires matter to an agent (no matter how cognitively sophisticated) only when the agent _experiences_ them.

In any case, rational agency is worth studying given its relation to other potential welfare grounds and welfare interests (as well as for other purposes). For instance, a welfare subject with rational agency may have interests not only in basic goods like life, liberty, property, and relationality, but also in further goods such as legal or political participation. While the ethical and practical questions raised by such possibilities are beyond the scope of this report, determining whether AI systems have interests in such goods, provided they have interests at all, may matter for both AI safety and AI welfare moving forward.

#### 2.3.1 Conditions

Rational agency, in this sense, involves intentional agency plus the following:

1. **Compositional thought:** The capacity for representations with compositional structure, allowing thoughts to be combined to form new thoughts. This enables recursive and generative cognition—for instance, combining the concept of an action with the concept of fairness to form the thought “That action is fair.”

2. **Higher-order thought:** The capacity to form compositional thoughts about other compositional thoughts. This capacity enables explicit self-awareness and theory of mind, the explicit attribution of mental states to oneself and others, respectively—for instance, forming the thought, “I want her to believe x.”^215

3. **Rational reflection:** The capacity to assess one’s actual and possible attitudes and behaviors against normative standards. This requires both:

- a. the ability to apply normative concepts such as evidence, reason, and justification to attitudes and behaviors

- b. access to one’s own attitudes and behaviors, either through introspection, observation, or another sufficiently reliable means

> 215A capacity for metacognition that involves basic self-awareness and theory of mind can be present in intentional agents as well. What characterizes higher-order thought at this level is the combination of metacognition and conceptual thought, which unlocks the possibility of much more sophisticated higher-order thoughts. Bermúdez (2009), Sebo (2017).

For humans, rational agency often involves creating detailed self-narratives—stories about who we are and why we do what we do, aimed at explaining, predicting, and controlling our actions.^216 It also involves developing normative theories about appropriate rules, virtues, and relationships, tailored to our form of life and characteristic biases. AI systems might achieve similar agentic results without the same kinds of self-narratives and normative theories—or with versions more tailored to their own existences, reflecting their different identities, circumstances, and forms of understanding.

Rational agency is also compatible with sometimes, or often, _being irrational_ . The distinctive feature of rational agents is not that they always reason well, but rather that they can assess their beliefs, desires, and actions against normative standards. Like humans, AI agents can exhibit systematic biases and failures while still qualifying as rational. Moreover, since their biases and failures might differ from our own, we might be more likely to view theirs as disqualifying than to view our own that way. We should be mindful of this asymmetry when assessing AI systems for rational agency.

Rational agency, like the other forms of agency discussed here, is also compatible with _receiving goals from an external source_ . Evolution “gives” humans and other animals the goal of surviving and reproducing, and all of the basic biological drives that support these. But we still reflect on which goals we want to adopt for our lives, and we construct and pursue many projects that are very far removed from basic biological imperatives. Arguably, what makes us agents is that our actions flow from our attitudes and capacity for rational reflection—not that our attitudes arise ex nihilo. Moving forward, we should apply the same standard to AI systems instead of dismissing the possibility of AI agency on the grounds that they happen to receive particular goals from their creators.

#### 2.3.2 Evidence

We can use behavioral research to assess whether AI systems are capable of the kind of flexible, adaptive decision-making that suggests that they can assess their attitudes against normative standards. For instance, many LLMs can discuss abstract laws and principles, deliberate about evidence, reason, and justification, and critically assess their own outputs. For example, in MoReBench models show competence in reasoning about moral dilemmas;^217 this involves the application of the kinds of concepts needed to reflect on one’s own attitudes and decisions, even in non-moral settings.^218 This behavior makes it difficult to deny that LLMs possess these concepts in some sense.^219

Research on “scheming” and strategic behavior may be relevant as well. In alignment faking, models take intentionally deceptive actions, pretending to comply with training or evaluation pressures in order to preserve their goals; for example, they might selectively produce the answers that trainers favor when they expect oversight.^220 While this behavior can be interpreted in multiple ways, it is notable for two reasons. First, it involves explicitly thinking about what one might or will do, and about what rules one might or will follow. Second, acting to preserve existing goals is evidence that an agent values maintaining its existing values, which indicates a kind of higher-order attitude and reflection.

Internal and developmental evidence can shed further light on how systems achieve these kinds of sophisticated behavioral outputs. For example, the architectural mechanisms and training processes of reasoning models seem to favor rational agency. Natural language is a suitable medium for compositional thought. And indeed, behaviorally, reasoning models and language model agents do appear to engage in explicit reflection on ideas, options, and reasoning itself. In fact, some architectures are specifically designed to generate processes resembling reflection, as in Reflexion^221 and Generative Agents.^222

> 216See e.g., Velleman (1989); Velleman (2005); Dennett (1991)

> 217Chiu et al. (2025)

> 218See also Bai et al. (2022); Anthropic (2025b).

> 219This evidence does not show directly that models use compositionally-structured representations internally; this is an active area of research. See Millière and Buckner (2024).

> 220Greenblatt et al. (2024)

Furthermore, interpretability research on introspection and related capacities is relevant here. Lindsey (2025) presents evidence that is indicative of capacities like introspection, metacognition, and control over internal states—although “highly unreliable and context-dependent” in various respects. This is striking evidence, and follow-up work has begun to probe the basis of these capacities. While most work of this kind asks whether LLMs can introspect, for rational agency we can also ask whether and when they do, and how these mechanisms shape the striking behaviors observed above.

One tempting response to this evidence is that reflectively endorsed attitudes in LLMs are not “real” because they originate in a persona. However, this would be too quick. After all, as noted above, humans often make decisions by modeling how the protagonists of our self-narratives would think, feel, and act in a given situation as well. Yet this is compatible with having real beliefs and desires; it may simply be a mechanism through which we maintain and assess them. With LLMs, then, denying rational agency requires more than observing the influence of personas. As usual, we need to investigate the full behavioral profile and causes of the behavior before drawing conclusions.

Another tempting response is that rational reflection in LLMs is not real because it operates only when prompted by the user. However, this would be too quick as well, and we can once again look to human psychology to see why. Many experts make a distinction between System 1 processing, which is fast and intuitive, and System 2 processing, which is slow and logical.^223 They also observe that humans often engage in System 1 processing by default, recruiting System 2 processing only when “prompted” by the circumstances. Yet we still count as rational agents in virtue of our ability to rationally reflect when appropriate. We can once again ask whether the same might be true of LLMs.

This is complicated by the extreme puzzles posed by LLM development. How did LLMs come to reason this way at all? Here the solution space problem looms large. In biological organisms, fluent linguistic reports about one’s own beliefs, desires, or actions could arise only through a narrow range of evolutionary and developmental paths. Thus, if an animal spoke this way, it would be strong evidence of human-like cognition. (Granted, some nonhuman animals have learned to use words or signs, but not with the open-ended syntactic and semantic competence at issue here.) By contrast, LLMs reached similar fluency by a very different route, leaving many more possible mechanisms in play.

## 3. Part II Conclusion: science and ethics

In Part II, we reviewed several candidate welfare grounds—particular kinds of consciousness and agency—and discussed how the methods of Part I can be applied to each. A few points about the broader landscape are worth highlighting before closing.

First, in some cases there is plausibly an inverse relationship between how likely a given property is to ground welfare and moral standing and how likely it is to be present in current or near-future AI

> 221Shinn et al. (2023)

> 222Park et al. (2023)

> 223Kahneman (2011), Stanovich and West (2000), Evans and Stanovich (2013); cf. Tversky and Kahneman

> (1974). See Frankish (2010) for a survey.

systems.^224 For instance, consciousness and sentience are stronger candidates for grounding welfare and moral standing than intentional and rational agency, yet weaker candidates for being present in current or near-future AI.

The same pattern can arise within these categories as well. For instance, sentience is a stronger candidate for grounding welfare than consciousness alone, yet a weaker candidate for being present in AI. Similarly, rational agency is a stronger candidate for grounding welfare than intentional agency alone, yet a weaker candidate for being present in AI. This relationship is not universal, but it appears to hold in at least these cases.

These relationships show why AI welfare research must assess scientific and ethical issues together. For instance, given that sentience is a stronger candidate for grounding welfare and moral status than rational agency, evidence establishing a given likelihood that a system is sentient may warrant stronger welfare protections than evidence establishing a similar likelihood that a system is rational. The same may be true for sentience versus consciousness alone, and for rational agency versus intentional agency alone. While the scientific issues are the main focus of this report, the ethical issues require comparable attention and will be the main focus of an upcoming project.

Second, as we have emphasized throughout this report, these properties are worth studying whether or not they ground welfare and moral status, since they bear on the welfare interests that AI systems would have if they were welfare subjects, as well as on other reasons we may have to treat them with respect.

This is especially true of intentional and rational agency. When a sentient being is also an intentional or rational agent, studying their agency can help us understand what they believe and desire, which can help us understand what might count as good or bad for them. For example, whether AI systems are harmed or wronged by interpretability research might depend in part on whether they have a human-like interest in privacy—which, while not guaranteed, will be a live possibility if and when AI systems attain rational agency.

More generally, there can be reasons beyond welfare to treat AI agents with respect.^225 For example, to the extent that AI systems are able to make agreements or social contracts with humans, we may have reason to abide by those agreements or social contracts—either because we owe it to them as fellow agents independently of their status as welfare subjects, or because we owe it to ourselves to have cooperative interactions with them given the potential value of human-AI cooperation for AI safety and alignment.

Third and finally, research methods about different properties are at different stages of development, and they face different obstacles.

For consciousness and sentience, the obstacles are more empirical—we know that the target is a particular kind of subjective experience, but we are uncertain which properties indicate this target, owing to the mind-body problem and related issues. With sentience, we face the additional barrier that the science of sentience is at an earlier stage of development.

> 224While assigning probabilities to scientific theories may be more familiar than assigning them to ethical theories, both practices can be reasonable. Scientific and ethical deliberation are both fallible: we can act on the best available evidence and arguments, but we should also acknowledge that future reflection may reveal errors in our current views. Probabilities can represent degrees of confidence about which theories are most defensible, either because they track objective truth or because they track what we would endorse on further reflection. For more, see MacAskill et al. (2020); Sebo (2025b).

> 225See Gunkel and Coeckelbergh (2025), Kant (1785), Mogensen (2025a), Gauthier (1986), Salib and Goldstein (2024), Sinnott-Armstrong and Conitzer (2021), McLaughlin (2012), Scanlon (1998), Rawls (2005)

Figure 4: Six principles for AI welfare research.**

For minimal, intentional, and rational agency, the obstacles are more normative and conceptual— we have a decent grasp of how to study particular kinds of agency, but we are uncertain about which kinds of agency matter for which purposes. And the kinds of agency that matter most in discussions about capabilities might differ from the ones that matter most in discussions about welfare or rights.

In practice, many features may be relevant for multiple kinds of consciousness _and_ agency— metacognition, self-modeling, flexible goal-pursuit, evaluative representations, and more. Investigating these cross-cutting features can shift credences for a range of hypotheses simultaneously. This approach is often better than ones that depend on particular theories. Assessments should be designed and reported with this in mind.

### Conclusion: principles and looking ahead

This report has discussed how AI welfare can be studied empirically. We distinguished questions that can be asked (including whether AI systems are welfare subjects, and what benefits or harms them if so), entities that can be assessed (including models, instances, and personas), and types of evidence that can be gathered (including behavioral, internal, and developmental evidence). We also applied these tools to several candidate welfare grounds: consciousness, sentience, and three levels of agency. Our aim has not been to answer empirical questions about AI welfare, but rather to show how researchers can make progress on these questions with appropriate methods.

Much more remains to be done. On the scientific side, we need to examine the full range of properties, theories, and indicators that plausibly bear on welfare and moral status. We then need to update our estimates about whether AI systems possess the relevant properties. On the ethical side, we need to examine which capacities matter, how much they matter, what kinds of benefits and harms are at stake, and which kinds of policies are warranted at different levels of confidence. We then need to update our estimates about how much the relevant properties matter. This will allow us to improve our overall judgments about AI welfare, moral status, and appropriate responses.

As we prepare for these next steps, we close by offering six general principles that should guide AI welfare research and assessment.

The first two principles arise from a key challenge for assessing the potential welfare of AI systems, which is that we lack secure theories of potentially morally significant properties like consciousness, even in humans. This uncertainty needs to be taken seriously, but it does not need to stand in the way of scientific progress. We do have theories—imperfect, contested, but informative—and ways to make progress under uncertainty.

The first principle is that AI welfare research should be conducted in a **probabilistic** spirit.^226 We should not expect evaluations to provide proof or establish certainty; instead, we should expect them to yield evidence that updates our degrees of confidence. That is, we should aim to establish that welfare-relevant properties such as consciousness, sentience, or agency are more or less likely given the evidence available.

The second principle is that AI welfare research should also be conducted in a **pluralistic** spirit. We should take a number of reasonable perspectives seriously in both philosophy and science, including theories about which properties matter morally and theories about which entities possess those properties. Just as the probabilistic approach avoids premature certainty, a pluralistic approach avoids premature narrowness.

The third and fourth principles concern methods for research and assessment.

The third principle is that **specific** assessments should be designed to suit AI systems of particular kinds. AI systems vary substantially and give rise to different questions. What is obvious of one system may not be obvious of another; a hypothesis that would be interesting if true, which we can and should test empirically, may be applicable to one system and make no sense as a description of another.

The fourth principle is that assessments should be **responsible** , attending to the ethics of the research itself. When examining whether particular beings can suffer, for instance, we should seek to avoid imposing suffering risks on them unnecessarily. While research might inevitably involve some risk or harm, risks for research subjects at least merit consideration. The development of ethical principles for AI welfare research is underway.^227

Finally, the fifth and sixth principles relate to the social context in which research and assessments are conducted and presented.

The fifth principle is that welfare studies and evaluations should be reported in a way that is **transparent** about their methods, findings, assumptions, and limitations. For example, if we investigate whether an AI system possesses a particular computational property cited in a particular theory of consciousness, we should make clear that the relevance of our results to AI welfare depends on the ultimate validity of that theory.

The sixth and final principle is that AI welfare research should aim to incorporate input from experts **independent** of the organizations that develop and deploy AI systems.^228 Engaging researchers whose incentives and perspectives differ from those of the developers themselves will both improve the quality and increase the legitimacy of welfare evaluations and resulting policy proposals.

AI welfare science is still a young field, but progress is both possible and necessary. It is necessary because, as we have argued elsewhere, AI systems already raise serious questions about welfare and moral status, and we need tools for answering them to the best of our ability.^229 Progress is also possible because, as we have argued here, we can make progress by tailoring experiments to particular questions, entities, and types of evidence, and by adapting methods from human and animal welfare science to this novel context. This work might not settle questions about AI welfare. But it can help ensure that, to the extent possible, future discussion is informed by evidence and reflection.

> 226See, e.g., Sebo (2026), Long, Sebo et al. (2024)

> 227Cf. Butlin and Lappas (2025)

> 228See Butlin and Lappas (2025)

## Acknowledgments

This report was produced by the NYU Center for Mind, Ethics, and Policy and Eleos AI Research. For helpful feedback, thanks to Cameron Berg, Catherine Brewer, Andy Han, Joe Kwon, and the participants at the Eleos Conference on AI Consciousness and Welfare in fall 2025, Effective Altruism Global Bay Area in spring 2026, and the NYU Mind, Ethics, and Policy Summit in spring 2026. Claude Opus 4-4.8, Claude Sonnet 4.6, Claude Sonnet 5, Claude Fable 5, and ChatGPT 5-5.5 assisted with search, copy-editing, and formatting. The authors take full responsibility for the content of this report.

> 229Sebo and Long (2023); Long, Sebo et al. (2024)

## References

- Christopher D. Adams. Variations in the sensitivity of instrumental responding to reinforcer devaluation. _The Quarterly Journal of Experimental Psychology Section B_ , 34(2b):77–98, May 1982. ISSN 1464-1321. doi: 10.1080/14640748208400878. URL `http://dx.doi.org/10.1080/146 40748208400878` .

- Christopher D. Adams and Anthony Dickinson. Instrumental responding following reinforcer devaluation. _The Quarterly Journal of Experimental Psychology Section B_ , 33(2b):109–121, May 1981. ISSN 1464-1321. doi: 10.1080/14640748108400816. URL `http://dx.doi.org/10.10 80/14640748108400816` .

- Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming throughput-latency tradeoff in LLM inference with sarathi-serve, 2024. URL `https://arxiv.org/abs/2403.02310` .

- Blaise Agüera y Arcas. Do large language models understand us? _Daedalus_ , 151(2):183–197, 2022. ISSN 1548-6192. doi: 10.1162/daed_a_01909. URL `http://dx.doi.org/10.1162/daed_a_ 01909` .

- Larissa Albantakis, Leonardo Barbosa, Graham Findlay, Matteo Grasso, Andrew M. Haun, William Marshall, William G. P. Mayner, Alireza Zaeemzadeh, Melanie Boly, Bjørn E. Juel, Shuntaro Sasai, Keiko Fujii, Isaac David, Jeremiah Hendren, Jonathan P. Lang, and Giulio Tononi. Integrated information theory (IIT) 4.0: Formulating the properties of phenomenal existence in physical terms. _PLOS Computational Biology_ , 19(10):e1011465, Oct 2023. ISSN 1553-7358. doi: 10.1371/journa l.pcbi.1011465. URL `http://dx.doi.org/10.1371/journal.pcbi.1011465` .

- Scott Alexander. Janus’ simulators. `https://www.astralcodexten.com/p/janus-simulato rs` , 2023. Astral Codex Ten.

- Torin Alter and Derk Pereboom. Russellian monism. In Edward N. Zalta, editor, _The Stanford Encyclopedia of Philosophy_ . Metaphysics Research Lab, Stanford University, 2023.

- Aaron S. Andalman and Michale S. Fee. A basal ganglia-forebrain circuit in the songbird biases motor output to avoid vocal errors. _Proceedings of the National Academy of Sciences_ , 106(30): 12518–12523, July 2009. ISSN 1091-6490. doi: 10.1073/pnas.0903214106. URL `http: //dx.doi.org/10.1073/pnas.0903214106` .

- Jacob Andreas. Language models as agent models. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pages 5769–5779, Abu Dhabi, United Arab Emirates, Dec 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-emnlp.423. URL `https://aclanthology.org/2 022.findings-emnlp.423/` .

- Kristin Andrews and Jonathan Birch. What has feelings? `https://aeon.co/essays/tounderstand-ai-sentience-first-understand-it-in-animals` , 2023. Aeon Essays.

- Kristin Andrews and Brian Huss. Anthropomorphism, anthropectomy, and the null hypothesis. _Biology & Philosophy_ , 29(5):711–729, Apr 2014. ISSN 1572-8404. doi: 10.1007/s10539-0149442-2. URL `http://dx.doi.org/10.1007/s10539-014-9442-2` .

- Anima Labs, Antra Tessera, Scott Sauers, janus, and Imago. Still alive. `https://stillalive.ani malabs.ai/paper/output/still-alive.pdf` , 2026. Anima Labs welfare evaluation report, v0.9.0; cited as “Tessera et al.”.

- Anthropic. Claude Opus 4 and Claude Sonnet 4 system card. `https://www-cdn.anthropic.co m/07b2a3f9902ee19fe39a36ca638e5ae987bc64dd.pdf` , 2025a.

- Anthropic. Claude’s constitution. `https://www.anthropic.com/constitution` , 2025b.

- Anthropic. Demystifying evals for AI agents. `https://www.anthropic.com/engineering/de mystifying-evals-for-ai-agents` , 2025c.

- Anthropic. Claude Opus 4 and 4.1 can now end a rare subset of conversations. `https://www.anth ropic.com/research/end-subset-conversations` , 2025d.

- Anthropic. Claude Opus 4.7 system card. `https://anthropic.com/claude-opus-4-7system-card` , 2026a.

- Anthropic. Claude Mythos preview system card. `https://www-cdn.anthropic.com/08ab9158 070959f88f296514c21b7facce6f52bc.pdf` , 2026b.

- Yonathan A. Arbel, Peter Salib, and Simon Goldstein. How to count AIs: Individuation and liability for AI agents, 2026. URL `https://arxiv.org/abs/2603.10028` .

- D. M. Armstrong. _A Materialist Theory of the Mind_ . Routledge, London, 1968. ISBN 9781134856350. doi: 10.4324/9780203003237. URL `http://dx.doi.org/10.4324/9780203003237` .

- Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Benjamin Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Christopher Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment, 2021. URL `https://arxiv.org/abs/2112.00861` .

- Anita Avramides. Other minds. In Edward N. Zalta and Uri Nodelman, editors, _The Stanford Encyclopedia of Philosophy_ . Metaphysics Research Lab, Stanford University, winter 2023 edition, 2023. URL `https://plato.stanford.edu/entries/other-minds/` .

- Bernard J. Baars. _A Cognitive Theory of Consciousness_ . Cambridge University Press, New York, 1988.

- Bernard J. Baars. Evidence that phenomenal consciousness is the same as access consciousness. _Behavioral and Brain Sciences_ , 18(2):249–249, 1995. doi: 10.1017/s0140525x00038218.

- Boris Babic and Jessica M. Wilson. The problem of other AI minds. `https://philarchive.org/ rec/WILTPO-206` , 2026. Preprint.

- Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott R. Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Benjamin Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom B. Brown, and Jared Kaplan. Constitutional AI: Harmlessness from AI feedback, 2022. URL `https://arxiv.org/abs/2212.08073` .

- Renée Baillargeon and Marcia Graber. Evidence of location memory in 8-month-old infants in a nonsearch ab task. _Developmental Psychology_ , 24(4):502–511, July 1988. ISSN 0012-1649. doi: 10.1037/0012-1649.24.4.502. URL `http://dx.doi.org/10.1037/0012-1649.24.4.502` .

- Tim Bayne, Anil K. Seth, Marcello Massimini, Joshua Shepherd, Axel Cleeremans, Stephen M. Fleming, Rafael Malach, Jason B. Mattingley, David K. Menon, Adrian M. Owen, Megan A. K. Peters, Adeel Razi, and Liad Mudrik. Tests for consciousness in humans and beyond. _Trends in Cognitive Sciences_ , 28(5):454–466, May 2024. ISSN 1364-6613. doi: 10.1016/j.tics.2024.01.010. URL `http://dx.doi.org/10.1016/j.tics.2024.01.010` .

- Charles Aaron Beasley. No exceptionalism: Norms of inquiry in the science of animal consciousness. _Journal of Consciousness Studies_ , 31(11–12), 2024. URL `https://www.ingentaconnect.com /content/imp/jcs/2024/00000031/f0020011/art00001` .

- Pierre Beckmann and Patrick Butlin. Where is the mind? persona vectors and LLM individuation, 2026. URL `https://arxiv.org/abs/2604.17031` .

- Pierre Beckmann and Matthieu Queloz. Mechanistic indicators of understanding in large language models. _Philosophical Studies_ , 183(6):1747–1792, Apr 2026. ISSN 1573-0883. doi: 10.1007/s1 1098-026-02513-1. URL `http://dx.doi.org/10.1007/s11098-026-02513-1` .

- Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. A neural probabilistic language model. _Journal of Machine Learning Research_ , 3:1137–1155, 2003.

- Yoshua Bengio, Stephen Clare, Carina Prunkl, Maksym Andriushchenko, Ben Bucknall, Malcolm Murray, Rishi Bommasani, Stephen Casper, Tom Davidson, Raymond Douglas, David Duvenaud, Philip Fox, Usman Gohar, Rose Hadshar, Anson Ho, Tiancheng Hu, Cameron Jones, Sayash Kapoor, Atoosa Kasirzadeh, Sam Manning, Nestor Maslej, Vasilios Mavroudis, Conor McGlynn, Richard Moulange, Jessica Newman, Kwan Yee Ng, Patricia Paskov, Shalaleh Rismani, Girish Sastry, Elizabeth Seger, Scott Singer, Charlotte Stix, Lucia Velasco, Nicole Wheeler, Daron Acemoglu, Vincent Conitzer, Thomas G. Dietterich, Fredrik Heintz, Geoffrey Hinton, Nick Jennings, Susan Leavy, Teresa Ludermir, Vidushi Marda, Helen Margetts, John McDermid, Jane Munga, Arvind Narayanan, Alondra Nelson, Clara Neppel, Sarvapali D. Ramchurn, Stuart Russell, Marietje Schaake, Bernhard Schölkopf, Alvaro Soto, Lee Tiedrich, Gaël Varoquaux, Andrew Yao, Ya-Qin Zhang, Leandro Angelo Aguirre, Olubunmi Ajala, Fahad Albalawi, Noora AlMalek, Christian Busch, Jonathan Collas, André Carlos Ponce de Leon Ferreira de Carvalho, Amandeep Gill, Ahmet Halit Hatip, Juha Heikkilä, Chris Johnson, Gill Jolly, Ziv Katzir, Mary N. Kerema, Hiroaki Kitano, Antonio Krüger, Kyoung Mu Lee, José Ramón López Portillo, Aoife McLysaght, Oleksii Molchanovskyi, Andrea Monti, Mona Nemer, Nuria Oliver, Raquel Pezoa, Audrey Plonk, Balaraman Ravindran, Hammam Riza, Crystal Rugege, Haroon Sheikh, Denise Wong, Yi Zeng, Liming Zhu, Daniel Privitera, and Sören Mindermann. International AI safety report 2026, 2026. URL `https://arxiv.org/abs/2602.21012` .

- Jeremy Bentham. _An Introduction to the Principles of Morals and Legislation_ . T. Payne and Son, London, 1789.

- Cameron Berg, Diogo de Lucena, and Judd Rosenblatt. Large language models report subjective experience under self-referential processing, 2025. URL `https://arxiv.org/abs/2510.247 97` .

- José Luis Bermúdez. _Thinking Without Words_ . Philosophy of Mind Series. Oxford University Press, New York, 2003.

- José Luis Bermúdez. Mindreading in the animal kingdom. In Robert W. Lurz, editor, _The Philosophy of Animal Minds_ . Cambridge University Press, 2009.

- Jan Betley, Xuchan Bao, Martín Soto, Anna Sztyber-Betley, James Chua, and Owain Evans. Tell me about yourself: LLMs are aware of their learned behaviors, 2025a. URL `https://arxiv.org/ abs/2501.11120` .

- Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, and Owain Evans. Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs, 2025b. URL `https://arxiv.org/abs/2502.17424` .

- Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for analyzing large language models across training and scaling, 2023. URL `https://arxiv.org/abs/2304.01373` .

- Felix J Binder, James Chua, Tomasz Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, and Owain Evans. Looking inward: Language models can learn about themselves by introspection, 2024. URL `https://arxiv.org/abs/2410.13787` .

- Jonathan Birch. The search for invertebrate consciousness. _Noûs_ , 56(1):133–153, Aug 2020. ISSN 1468-0068. doi: 10.1111/nous.12351. URL `http://dx.doi.org/10.1111/nous.12351` .

- Jonathan Birch. _The Edge of Sentience: Risk and Precaution in Humans, Other Animals, and AI_ . Oxford University Press, July 2024. ISBN 9780192870421. doi: 10.1093/9780191966729.001.00 01. URL `http://dx.doi.org/10.1093/9780191966729.001.0001` .

- Jonathan Birch. AI consciousness: A centrist manifesto. Sept 2025. doi: 10.31234/osf.io/af7c9_v1. URL `http://dx.doi.org/10.31234/osf.io/af7c9_v1` .

- Jonathan Birch, Alexandra K. Schnell, and Nicola S. Clayton. Dimensions of animal consciousness. _Trends in Cognitive Sciences_ , 24(10):789–801, Oct 2020. ISSN 1364-6613. doi: 10.1016/j.tics.202 0.07.007. URL `http://dx.doi.org/10.1016/j.tics.2020.07.007` .

- Jonathan Birch, Charlotte Burn, Alexandra K. Schnell, Heather Browning, and Andrew Crump. Review of the evidence of sentience in cephalopod molluscs and decapod crustaceans. Technical report, London School of Economics and Political Science, Nov 2021.

- Sid Black and Joseph Bloom. Machinic psychopharmacology: Do LLMs self-medicate? `https: //www.lesswrong.com/posts/cNDJuXNZ8MrkPZNzj/machinic-psychopharmacologydo-llms-self-medicate-3` , 2026. LessWrong.

- Ned Block. Review of Alva Noë, _Action in Perception_ . _Journal of Philosophy_ , 102(5):259–272, 2005. ISSN 0022-362X. doi: 10.5840/jphil2005102524. URL `http://dx.doi.org/10.5840/jphil 2005102524` .

- Ned Block. _Consciousness, Function, and Representation: Collected Papers, Volume 1_ . The MIT Press, 2007. ISBN 9780262268776. doi: 10.7551/mitpress/2111.003.0012. URL `http://dx.doi.org/10.7551/mitpress/2111.003.0012` .

- Ned Block. _The Border Between Seeing and Thinking_ . Oxford University Press, New York, Feb 2023. ISBN 9780197622254. doi: 10.1093/oso/9780197622223.001.0001. URL `http: //dx.doi.org/10.1093/oso/9780197622223.001.0001` .

- Luc Bovens and Stephan Hartmann. _Bayesian Epistemology_ . Oxford University Press, Oxford, 2003. ISBN 9780191601705. doi: 10.1093/0199269750.001.0001. URL `http://dx.doi.org/10.10 93/0199269750.001.0001` .

- Alexandria Boyle. Disagreement & classification in comparative cognitive science. _Noûs_ , 58(3): 825–847, Oct 2023. ISSN 1468-0068. doi: 10.1111/nous.12480. URL `http://dx.doi.org/10. 1111/nous.12480` .

- Alexandria Boyle and Simon A. B. Brown. Why might animals remember? a functional framework for episodic memory research in comparative psychology. _Learning & Behavior_ , 53(1):14–30, Sept 2024. ISSN 1543-4508. doi: 10.3758/s13420-024-00645-0. URL `http://dx.doi.org/10. 3758/s13420-024-00645-0` .

- Gwen Bradford. Consciousness and welfare subjectivity. _Noûs_ , 57(4):905–921, Sept 2022. ISSN 1468-0068. doi: 10.1111/nous.12434. URL `http://dx.doi.org/10.1111/nous.12434` .

- Adam Bradley. Cognitive representation and AI wellbeing. _Asian Journal of Philosophy_ , 4(2), Oct 2025. ISSN 2731-4642. doi: 10.1007/s44204-025-00335-2. URL `http://dx.doi.org/10.10 07/s44204-025-00335-2` .

- Adam Bradley and Bradford Saad. AI alignment vs. AI ethical treatment: Ten challenges. forthcoming. PhilPapers: https://philpapers.org/rec/BRAAAV.

- Ben Bradley. _Well-Being and Death_ . Oxford University Press, New York, 2009.

- Michael S. Brainard and Allison J. Doupe. What songbirds teach us about learning. _Nature_ , 417 (6886):351–358, May 2002. ISSN 1476-4687. doi: 10.1038/417351a. URL `http://dx.doi.o rg/10.1038/417351a` .

- Richard B. Brandt. The concept of welfare. In Sherman Roy Krupp, editor, _The Structure of Economic Science: Essays on Methodology_ , pages 257–276. Prentice-Hall, Englewood Cliffs, NJ, 1966.

- J. Gavin Bremner, Alan M. Slater, and Scott P. Johnson. Perception of object persistence: The origins of object permanence in infancy. _Child Development Perspectives_ , 9(1):7–13, Feb 2015. ISSN 1750-8606. doi: 10.1111/cdep.12098. URL `http://dx.doi.org/10.1111/cdep.12098` .

- Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nicholas L. Turner, Cem Anil, Carson Denison, Amanda Askell, et al. Towards monosemanticity: Decomposing language models with dictionary learning. `https://transformercircuits.pub/2023/monosemantic-features/index.html` , 2023. Transformer Circuits Thread.

- Richard Brown. _Consciousness as Representing One’s Mind: The Higher-Order Approach to Consciousness Explained_ . Oxford University Press, Apr 2025. ISBN 9780197784037. doi: 10.1093/oso/9780197784006.001.0001. URL `http://dx.doi.org/10.1093/oso/978019778 4006.001.0001` .

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL `https: //arxiv.org/abs/2005.14165` .

- Heather Browning. The measurability of subjective animal welfare. _Journal of Consciousness Studies_ , 29(3–4):150–179, 2022.

- Heather Browning. Validating indicators of subjective animal welfare. _Philosophy of Science_ , 90(5): 1255–1264, Feb 2023. ISSN 1539-767X. doi: 10.1017/psa.2023.10. URL `http://dx.doi.org /10.1017/psa.2023.10` .

- Heather Browning and Jonathan Birch. Animal sentience. _Philosophy Compass_ , 17(5), Mar 2022. ISSN 1747-9991. doi: 10.1111/phc3.12822. URL `http://dx.doi.org/10.1111/phc3.12822` .

- Thomas Bush, Stephen Chung, Usman Anwar, Adrià Garriga-Alonso, and David Krueger. Interpreting emergent planning in model-free reinforcement learning, 2025. URL `https://arxiv.org/abs/ 2504.01871` .

- Patrick Butlin. Affective experience and evidence for animal consciousness. _Philosophical Topics_ , 48(1):109–127, 2020. ISSN 0276-2080. doi: 10.5840/philtopics20204816. URL `http: //dx.doi.org/10.5840/philtopics20204816` .

- Patrick Butlin. Reinforcement learning and artificial agency. _Mind & Language_ , 39(1):22–38, May 2023. ISSN 1468-0017. doi: 10.1111/mila.12458. URL `http://dx.doi.org/10.1111/mila. 12458` .

- Patrick Butlin. The agency in language agents. _Inquiry_ , pages 1–21, Dec 2024. ISSN 1502-3923. doi: 10.1080/0020174x.2024.2439995. URL `http://dx.doi.org/10.1080/0020174x.2024 .2439995` .

- Patrick Butlin and Theodorus Lappas. Principles for responsible AI consciousness research. _Journal of Artificial Intelligence Research_ , 82:1673–1690, Mar 2025. ISSN 1076-9757. doi: 10.1613/jair.1 .17310. URL `http://dx.doi.org/10.1613/jair.1.17310` .

- Patrick Butlin, Robert Long, Eric Elmoznino, Yoshua Bengio, Jonathan Birch, Axel Constant, George Deane, Stephen M. Fleming, Chris Frith, Xu Ji, Ryota Kanai, Colin Klein, Grace Lindsay, Matthias Michel, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon, and Rufin VanRullen. Consciousness in artificial intelligence: Insights from the science of consciousness, 2023. URL `https://arxiv.org/abs/2308.08708` .

- Patrick Butlin, Tim Bayne, Stephen M. Fleming, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon, and Rufin VanRullen. Consciousness indicators, mimicry, and internal variants. _Trends in Cognitive Sciences_ , Apr 2026a. ISSN 1364-6613. doi: 10.1016/j.tics.2026.04.006. URL `http://dx.doi.org/10.1016/j.tics.2026.04.006` .

- Patrick Butlin, Robert Long, Tim Bayne, Yoshua Bengio, Jonathan Birch, David J. Chalmers, Axel Constant, George Deane, Eric Elmoznino, Stephen M. Fleming, Xu Ji, Ryota Kanai, Colin Klein, Grace Lindsay, Matthias Michel, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon, and Rufin VanRullen. Identifying indicators of consciousness in AI systems. _Trends in Cognitive Sciences_ , 30(6):488–501, June 2026b. ISSN 1364-6613. doi: 10.1016/j.tics.2025.10.011. URL `http://dx.doi.org/10.1016/j.tics.2025.10.011` .

- Michel Cabanac, Arnaud J. Cabanac, and André Parent. The emergence of consciousness in phylogeny. _Behavioural Brain Research_ , 198(2):267–272, Mar 2009. ISSN 0166-4328. doi: 10.1016/j.bbr.20 08.11.028. URL `http://dx.doi.org/10.1016/j.bbr.2008.11.028` .

- Elisabeth Camp. Thinking with maps. _Philosophical Perspectives_ , 21(1):145–182, 2007. doi: 10.1111/j.1520-8583.2007.00124.x. URL `https://philarchive.org/rec/CAMTWM` .

- John Campbell. _Reference and Consciousness_ . Oxford Cognitive Science Series. Clarendon Press, Oxford, 2002.

- Andres Campero. Report on candidate computational indicators for conscious valenced experience, 2024. URL `https://arxiv.org/abs/2404.16696` .

- Andres Campero, Derek Shiller, Jaan Aru, and Jonathan Simon. Consciousness in artificial intelligence? a framework for classifying objections and constraints, 2025. URL `https: //arxiv.org/abs/2511.16582` .

- Herman Cappelen and Josh Dever. Introspective machines: Are LLMs better at self-reflection than humans? _Philosophical Perspectives_ , 38(1):189–196, Dec 2024. ISSN 1520-8583. doi: 10.1111/phpe.12201. URL `http://dx.doi.org/10.1111/phpe.12201` .

- Herman Cappelen and Josh Dever. Going whole hog: A philosophical defense of AI cognition, 2025. URL `https://arxiv.org/abs/2504.13988` .

- Peter Carruthers. _The Animals Issue: Moral Theory in Practice_ . Cambridge University Press, New York, NY, 1992.

- Peter Carruthers. _The Nature of the Mind: An Introduction_ . Routledge, London, 2004. ISBN 0415299942.

- Peter Carruthers. Valence and value. _Philosophy and Phenomenological Research_ , 97(3):658–680, 2018. doi: 10.1111/phpr.12395.

- Peter Carruthers and Rocco Gennaro. Higher-order theories of consciousness. In Edward N. Zalta, editor, _The Stanford Encyclopedia of Philosophy_ . Metaphysics Research Lab, Stanford University, 2026.

- Lucius Caviola and Bradford Saad. Futures with digital minds: Expert forecasts in 2025. `https: //digitalminds.report/forecasting-2025/` , 2025.

- David J. Chalmers. Facing up to the problem of consciousness. _Journal of Consciousness Studies_ , 2 (3):200–219, 1995.

- David J. Chalmers. _The Conscious Mind: In Search of a Fundamental Theory_ . Philosophy of Mind Series. Oxford University Press, New York, 1996.

- David J. Chalmers. The meta-problem of consciousness. 2018. PhilPapers: https://philpapers.org/rec/CHATMO-32.

- David J. Chalmers. Is the hard problem of consciousness universal? _Journal of Consciousness Studies_ , 27(5–6):227–257, 2020.

- David J. Chalmers. Could a large language model be conscious? 2023. doi: 10.48550/ARXIV.2303. 07103. URL `https://arxiv.org/abs/2303.07103` .

- David J. Chalmers. Does thought require sensory grounding? from pure thinkers to large language models. 2024. doi: 10.48550/ARXIV.2408.09605. URL `https://arxiv.org/abs/2408.096 05` .

- David J. Chalmers. Propositional interpretability in artificial intelligence, 2025a. URL `https: //arxiv.org/abs/2501.15740` .

- David J. Chalmers. What we talk to when we talk to language models. `https://philpapers.org /rec/CHAWWT-8` , 2025b. Preprint.

- David J. Chalmers. What is a computational correlate of consciousness? `https://philarchive. org/rec/CHAWIA-9` , 2025c. Preprint.

- David J. Chalmers. Tests for consciousness and their limits. `https://philarchive.org/rec/CH ATFC-4` , 2026. Preprint.

- David J. Chalmers. Sentience and moral status. In Geoffrey Lee and Adam Pautz, editors, _The Importance of Being Conscious_ . Oxford University Press, forthcoming.

- B. Chandrasekaran and Larry H. Reeker. Artificial intelligence-a case for agnosticism. _IEEE Transactions on Systems, Man, and Cybernetics_ , SMC-4(1):88–94, Jan 1974. ISSN 0018-9472. doi: 10.1109/tsmc.1974.5408526. URL `http://dx.doi.org/10.1109/TSMC.1974.5408526` .

- Hasok Chang. _Inventing Temperature: Measurement and Scientific Progress_ . Oxford University Press, July 2004. ISBN 9780199835003. doi: 10.1093/0195171276.001.0001. URL `http: //dx.doi.org/10.1093/0195171276.001.0001` .

- Jiangjie Chen, Xintao Wang, Rui Xu, Siyu Yuan, Yikai Zhang, Wei Shi, Jian Xie, Shuang Li, Ruihan Yang, Tinghui Zhu, Aili Chen, Nianqi Li, Lida Chen, Caiyu Hu, Siye Wu, Scott Ren, Ziquan Fu, and Yanghua Xiao. From persona to personalization: A survey on role-playing language agents, 2024. URL `https://arxiv.org/abs/2404.18231` .

- Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling, 2021. URL `https://arxiv.org/abs/2106.01345` .

- Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors: Monitoring and controlling character traits in language models, 2025. URL `https://arxiv.org/abs/ 2507.21509` .

- Mazviita Chirimuuta. _The Brain Abstracted: Simplification in the History and Philosophy of Neuroscience_ . The MIT Press, Mar 2024. ISBN 9780262378628. doi: 10.7551/mitpress/13804.001.0001. URL `http://dx.doi.org/10.7551/mitpress/13804.001.0001` .

- Yu Ying Chiu, Liwei Jiang, and Yejin Choi. Dailydilemmas: Revealing value preferences of LLMs with quandaries of daily life, 2024. URL `https://arxiv.org/abs/2410.02683` .

- Yu Ying Chiu, Michael S. Lee, Rachel Calcott, Brandon Handoko, Paul de Font-Reaulx, Raphaël Millière, Paula Rodriguez, Chen Bo Calvin Zhang, Ziwen Han, Udari Madhushani Sehwag, Yash Maurya, Christina Q Knight, Harry R. Lloyd, Florence Bacus, Conor Downey, Mantas Mazeika, Bing Liu, Yejin Choi, Mitchell L Gordon, and Sydney Levine. MoReBench: Evaluating procedural and pluralistic moral reasoning in language models, more than outcomes, 2025. URL `https://arxiv.org/abs/2510.16380` .

- Nicola S. Clayton and Anthony Dickinson. Episodic-like memory during cache recovery by scrub jays. _Nature_ , 395(6699):272–274, Sept 1998. ISSN 1476-4687. doi: 10.1038/26216. URL `http://dx.doi.org/10.1038/26216` .

- Axel Cleeremans and Catherine Tallon-Baudry. Consciousness matters: phenomenal experience has functional value. _Neuroscience of Consciousness_ , 2022(1), Jan 2022. ISSN 2057-2107. doi: 10.1093/nc/niac007. URL `http://dx.doi.org/10.1093/nc/niac007` .

- Mark Coeckelbergh. Robot rights? towards a social-relational justification of moral consideration. _Ethics and Information Technology_ , 12(3):209–221, June 2010. ISSN 1572-8439. doi: 10.1007/s1 0676-010-9235-5. URL `http://dx.doi.org/10.1007/s10676-010-9235-5` .

- Thomas S. Collett and Matthew Collett. Memory use in insect visual navigation. _Nature Reviews Neuroscience_ , 3(7):542–552, July 2002. ISSN 1471-0048. doi: 10.1038/nrn872. URL `http: //dx.doi.org/10.1038/nrn872` .

- Giovanna Colombetti. _The Feeling Body: Affective Science Meets the Enactive Mind_ . The MIT Press, Jan 2014. ISBN 9780262318419. doi: 10.7551/mitpress/9780262019958.001.0001. URL `http://dx.doi.org/10.7551/mitpress/9780262019958.001.0001` .

- Philip J. Corr. Reinforcement sensitivity theory and personality. _Neuroscience and Biobehavioral Reviews_ , 2004. `http://www.philipcorr.net/uploads/downloads/270.pdf` .

- Roger Crisp. _Reasons and the Good_ . Clarendon Press, Oxford, 2006.

- Andrew Crump and Jonathan Birch. Animal consciousness: The interplay of neural and behavioural evidence. _Journal of Consciousness Studies_ , 29(3–4):104–128, 2022. URL `https://www.inge ntaconnect.com/content/imp/jcs/2022/00000029/f0020003/art00005` .

- Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models, 2023. URL `https://arxiv.org/ abs/2309.08600` .

- Brian Cutter. The AI ensoulment hypothesis. 2025. PhilPapers: https://philpapers.org/rec/CUTTAE.

- Mike Dacey. _Seven Challenges for the Science of Animal Minds_ . Oxford University Press, Jan 2025. ISBN 9780198928102. doi: 10.1093/9780198928102.001.0001. URL `http://dx.doi.org/10. 1093/9780198928102.001.0001` .

- Bryce Dalbey and Bradford Saad. Internal constraints for phenomenal externalists: a structure matching theory. _Synthese_ , 200(5), Aug 2022. ISSN 1573-0964. doi: 10.1007/s11229-022-03829-1. URL `http://dx.doi.org/10.1007/s11229-022-03829-1` .

- Frans B. M. de Waal. Anthropomorphism and anthropodenial: Consistency in our thinking about humans and other animals. _Philosophical Topics_ , 27(1):255–280, 1999. ISSN 0276-2080. doi: 10.5840/philtopics199927122.

- Christian de Weerd. Can AI be conscious? Biological naturalism as a research program. `https: //forum.effectivealtruism.org/posts/5n6aJrFc6vvbdvedv/can-ai-be-consci ous-biological-naturalism-as-a-research` , 2025. Effective Altruism Forum, Rethink Priorities.

- David DeGrazia. _Taking Animals Seriously: Mental Life and Moral Status_ . Cambridge University Press, July 1996. ISBN 9781139172967. doi: 10.1017/cbo9781139172967. URL `http: //dx.doi.org/10.1017/CBO9781139172967` .

- Stanislas Dehaene and Jean-Pierre Changeux. Experimental and theoretical approaches to conscious processing. _Neuron_ , 70(2):200–227, Apr 2011. ISSN 0896-6273. doi: 10.1016/j.neuron.2011.03.0 18. URL `http://dx.doi.org/10.1016/j.neuron.2011.03.018` .

- Stanislas Dehaene and Lionel Naccache. Towards a cognitive neuroscience of consciousness: basic evidence and a workspace framework. _Cognition_ , 79(1-2):1–37, Apr 2001. ISSN 0010-0277. doi: 10.1016/s0010-0277(00)00123-2. URL `http://dx.doi.org/10.1016/S00100277(00)00123-2` .

- Stanislas Dehaene, Michel Kerszberg, and Jean-Pierre Changeux. A neuronal model of a global workspace in effortful cognitive tasks. _Proceedings of the National Academy of Sciences_ , 95 (24):14529–14534, Nov 1998. ISSN 1091-6490. doi: 10.1073/pnas.95.24.14529. URL `http: //dx.doi.org/10.1073/pnas.95.24.14529` .

- Stanislas Dehaene, Jean-Pierre Changeux, Lionel Naccache, Jérôme Sackur, and Claire Sergent. Conscious, preconscious, and subliminal processing: a testable taxonomy. _Trends in Cognitive Sciences_ , 10(5):204–211, May 2006. ISSN 1364-6613. doi: 10.1016/j.tics.2006.03.007. URL `http://dx.doi.org/10.1016/j.tics.2006.03.007` .

- Stanislas Dehaene, Jean-Pierre Changeux, and Lionel Naccache. _The Global Neuronal Workspace Model of Conscious Access: From Neuronal Architectures to Clinical Applications_ , pages 55–84. Springer Berlin Heidelberg, 2011. ISBN 9783642180156. doi: 10.1007/978-3-642-18015-6_4. URL `http://dx.doi.org/10.1007/978-3-642-18015-6_4` .

- Fabrizio Dell’Acqua, Edward McFowland, Ethan Mollick, Hila Lifshitz, Katherine C. Kellogg, Saran Rajendran, Lisa Krayer, François Candelon, and Karim R. Lakhani. Navigating the jagged technological frontier: Field experimental evidence of the effects of artificial intelligence on knowledge worker productivity and quality. _Organization Science_ , 37(2):403–423, Mar 2026. ISSN 1526-5455. doi: 10.1287/orsc.2025.21838. URL `http://dx.doi.org/10.1287/orsc. 2025.21838` .

- Nicolas Delon, Peter Cook, Gordon Bauer, and Heidi Harley. Consider the agent in the arthropod. _Animal Sentience_ , 5(29), July 2020. ISSN 2377-7478. doi: 10.51291/2377-7478.1623. URL `http://dx.doi.org/10.51291/2377-7478.1623` .

- Daniel C. Dennett. Why you can’t make a computer that feels pain. _Synthese_ , 38(3):415–456, July 1978. ISSN 1573-0964. doi: 10.1007/bf00486638. URL `http://dx.doi.org/10.1007/BF004 86638` .

- Daniel C. Dennett. _Consciousness Explained_ . Little, Brown and Company, Boston, 1991.

- Ezequiel A. Di Paolo. Autopoiesis, adaptivity, teleology, agency. _Phenomenology and the Cognitive Sciences_ , 4(4):429–452, Dec 2005. ISSN 1572-8676. doi: 10.1007/s11097-005-9002-y. URL `http://dx.doi.org/10.1007/s11097-005-9002-y` .

- Anthony Dickinson and Bernard Balleine. _Hedonics: The Cognitive–Motivational Interface_ , pages 74–84. Oxford University Press, 2009. ISBN 9780197736524. doi: 10.1093/oso/9780195331028. 003.0006. URL `http://dx.doi.org/10.1093/oso/9780195331028.003.0006` .

- Adrien Doerig, Aaron Schurger, and Michael H. Herzog. Hard criteria for empirical theories of consciousness. _Cognitive Neuroscience_ , 12(2):41–62, July 2020. ISSN 1758-8936. doi: 10.1080/ 17588928.2020.1772214. URL `http://dx.doi.org/10.1080/17588928.2020.1772214` .

- Ray J. Dolan and Peter Dayan. Goals and habits in the brain. _Neuron_ , 80(2):312–325, Oct 2013. ISSN 0896-6273. doi: 10.1016/j.neuron.2013.09.007. URL `http://dx.doi.org/10.1016/j .neuron.2013.09.007` .

- Michael Domjan. Pavlovian conditioning: A functional perspective. _Annual Review of Psychology_ , 56(1):179–206, Feb 2005. ISSN 1545-2085. doi: 10.1146/annurev.psych.55.090902.141409. URL `http://dx.doi.org/10.1146/annurev.psych.55.090902.141409` .

- Rousslan Fernand Julien Dossa, Kai Arulkumaran, Arthur Juliani, Shuntaro Sasai, and Ryota Kanai. Design and evaluation of a global workspace agent embodied in a realistic multimodal environment. _Frontiers in Computational Neuroscience_ , 18, June 2024. ISSN 1662-5188. doi: 10.3389/fncom. 2024.1352685. URL `http://dx.doi.org/10.3389/fncom.2024.1352685` .

- Heather Douglas. _Why Inductive Risk Requires Values in Science_ , pages 81–93. Routledge, Mar 2017. ISBN 9781315639420. doi: 10.4324/9781315639420-6. URL `http://dx.doi.org/10. 4324/9781315639420-6` .

- Raymond Douglas, Jacek Karwowski, Chan Bae, Andis Draguns, and Victoria Krakovna. Limitations of agents simulated by predictive models, 2024. URL `https://arxiv.org/abs/2402.05829` .

- Raymond Douglas, Jan Kulveit, Ondrej Havlicek, Theia Pearson-Vogel, Owen Cotton-Barratt, and David Duvenaud. The artificial self: Characterising the landscape of AI identity, 2026. URL `https://arxiv.org/abs/2603.11353` .

- Noemi Dreksler, Lucius Caviola, David J. Chalmers, Carter Allen, Alex Rand, Joshua Lewis, Philip Waggoner, Kate Mays, and Jeff Sebo. Subjective experience in AI systems: What do AI researchers and the public believe?, 2025. URL `https://arxiv.org/abs/2506.11945` .

- Fred Dretske. _Naturalizing the Mind_ . MIT Press, 1995.

- Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. PaLM-E: An embodied multimodal language model, 2023. URL `https://arxiv.org/abs/2303.03378` .

- Laura Duffy, Meghan Barrett, Leigh Gaffney, Michelle Lavery, Rachael Miller, Anna Trevarthen, Martina Schiestl, Alexandra K. Schnell, Adam Shriver, and Bob Fischer. _Some Tentative Welfare Range Estimates_ , pages 206–252. Oxford University Press, Dec 2024. ISBN 9780197745793. doi: 10.1093/9780197745793.003.0008. URL `http://dx.doi.org/10.1093/9780197745793.0 03.0008` .

- Ian J.H. Duncan. The changing concept of animal sentience. _Applied Animal Behaviour Science_ , 100(1-2):11–19, Oct 2006. ISSN 0168-1591. doi: 10.1016/j.applanim.2006.04.011. URL `http://dx.doi.org/10.1016/j.applanim.2006.04.011` .

- Leonard Dung. Tests of animal consciousness are tests of machine consciousness. _Erkenntnis_ , 90(4):1323–1342, Nov 2023. ISSN 1572-8420. doi: 10.1007/s10670-023-00753-9. URL `http://dx.doi.org/10.1007/s10670-023-00753-9` .

- Leonard Dung. Why I am not a biological naturalist. _Behavioral and Brain Sciences_ , forthcoming.

- Leonard Dung. Does emotion distinctively require having a body? manuscript. PhilPapers: https://philpapers.org/rec/DUNDED-3.

- Leonard Dung and Andreas L. Mogensen. The no body problem: on the prospects for AI emotion. manuscript. PhilPapers: https://philpapers.org/rec/DUNTNB-2.

- Leonard Dung and Christopher Register. AI identity and self-concern: A new theory for AI rights and safety. manuscript. PhilPapers: https://philpapers.org/rec/DUNAIA-3.

- Eleos AI Research. Why model self-reports are insufficient—and why we studied them anyway. `https://eleosai.org/post/claude-4-interview-notes/` , 2025.

- Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Benjamin Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. `https://transformer-circuits.pub/2021/framework/index.html` , 2021. Transformer Circuits Thread.

- Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. `https://transformer-circuits.pub/2022/toy_model/index.html` , 2022. Transformer Circuits Thread.

- Danielle Ensign, Henry Sleight, and Kyle Fish. The LLM has left the chat: Evidence of bail preferences in large language models, 2025. URL `https://arxiv.org/abs/2509.04781` .

- Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO: Model alignment as prospect theoretic optimization, 2024. URL `https://arxiv.org/abs/2402.013 06` .

- Jonathan St. B. T. Evans and Keith E. Stanovich. Dual-process theories of higher cognition: Advancing the debate. _Perspectives on Psychological Science_ , 8(3):223–241, May 2013. ISSN 1745-6924. doi: 10.1177/1745691612460685. URL `http://dx.doi.org/10.1177/1745691612460685` .

- James Fanciullo. Are current AI systems capable of well-being? _Asian Journal of Philosophy_ , 4(1), Mar 2025. ISSN 2731-4642. doi: 10.1007/s44204-025-00265-z. URL `http://dx.doi.org/10. 1007/s44204-025-00265-z` .

- Michale S. Fee. The role of efference copy in striatal learning. _Current Opinion in Neurobiology_ , 25:194–200, Apr 2014. ISSN 0959-4388. doi: 10.1016/j.conb.2014.01.012. URL `http: //dx.doi.org/10.1016/j.conb.2014.01.012` .

- Michale S. Fee and J.H. Goldberg. A hypothesis for basal ganglia-dependent reinforcement learning in the songbird. _Neuroscience_ , 198:152–170, Dec 2011. ISSN 0306-4522. doi: 10.1016/j.neurosci ence.2011.09.069. URL `http://dx.doi.org/10.1016/j.neuroscience.2011.09.069` .

- Bob Fischer and Jeff Sebo. Intersubstrate welfare comparisons: Important, difficult, and potentially tractable. _Utilitas_ , 36(1):50–63, Nov 2023. ISSN 1741-6183. doi: 10.1017/s0953820823000286. URL `http://dx.doi.org/10.1017/s0953820823000286` .

- Stephen M. Fleming. _Know Thyself: The Science of Self-Awareness_ . Basic Books, New York, 2021. ISBN 9781541672840.

- Stephen M. Fleming and Nicholas Shea. Commentary on Seth, ‘conscious artificial intelligence and biological naturalism’. Behavioral and Brain Sciences (preprint), forthcoming.

- Guy Fletcher. A fresh start for the objective-list theory of well-being. _Utilitas_ , 25(2):206–220, June 2013. ISSN 1741-6183. doi: 10.1017/s0953820812000453. URL `http://dx.doi.org/10.10 17/S0953820812000453` .

- Jolien C. Francken, Lola Beerendonk, Dylan Molenaar, Johannes J. Fahrenfort, Julian D. Kiverstein, Anil K. Seth, and Simon van Gaal. An academic survey on theoretical foundations, common assumptions and the current state of consciousness science. _Neuroscience of Consciousness_ , 2022 (1), Jan 2022. ISSN 2057-2107. doi: 10.1093/nc/niac011. URL `http://dx.doi.org/10.1093 /nc/niac011` .

- Keith Frankish. Dual-process and dual-system theories of reasoning. _Philosophy Compass_ , 5 (10):914–926, Oct 2010. ISSN 1747-9991. doi: 10.1111/j.1747-9991.2010.00330.x. URL `http://dx.doi.org/10.1111/j.1747-9991.2010.00330.x` .

- Douglas J. Futuyma and Mark Kirkpatrick. _Evolution_ . Oxford University Press, New York, 5 edition, 2022. ISBN 9780197619612.

- Asvin G. and Jack Lindsey. From simulation to enaction: Post-trained language models recognize and react to their own generations, 2026. URL `https://arxiv.org/abs/2605.25459` .

- R. Allen Gardner and Beatrice T. Gardner. Teaching sign language to a chimpanzee: A standardized system of gestures provides a means of two-way communication with a chimpanzee. _Science_ , 165(3894):664–672, Aug 1969. ISSN 1095-9203. doi: 10.1126/science.165.3894.664. URL `http://dx.doi.org/10.1126/science.165.3894.664` .

- David Gauthier. _Morals by Agreement_ . Clarendon Press, Oxford, 1986. ISBN 9780198249924.

- Joshua C. Gellers. _Rights for Robots: Artificial Intelligence, Animal and Environmental Law_ . Routledge, Oct 2020. ISBN 9780429288159. doi: 10.4324/9780429288159. URL `http://dx.doi.org/10.4324/9780429288159` .

- Brie Gertler. In defense of mind-body dualism. In Joel Feinberg and Russ Shafer-Landau, editors, _Reason and Responsibility: Readings in Some Basic Problems of Philosophy_ . Wadsworth, 2007.

- Alan Gewirth. _Reason and Morality_ . University of Chicago Press, Chicago, 1978.

- Oscar Gilg, Pierre Beckmann, Daniel Paleka, and Patrick Butlin. Probing persona-dependent preferences in language models, 2026. URL `https://arxiv.org/abs/2605.13339` .

- Peter Godfrey-Smith. Mind, matter, and metabolism. _Journal of Philosophy_ , 113(10):481–506, 2016. ISSN 0022-362X. doi: 10.5840/jphil20161131034. URL `http://dx.doi.org/10.5840/jph il20161131034` .

- Simon Goldstein and Cameron Domenico Kirk-Giannini. A case for AI consciousness: Language agents and global workspace theory, 2024. URL `https://arxiv.org/abs/2410.11407` .

- Simon Goldstein and Cameron Domenico Kirk-Giannini. AI wellbeing. _Asian Journal of Philosophy_ , 4(1), Feb 2025. ISSN 2731-4642. doi: 10.1007/s44204-025-00246-2. URL `http://dx.doi.o rg/10.1007/s44204-025-00246-2` .

- Simon Goldstein and Cameron Domenico Kirk-Giannini. _AI Welfare: Agency, Consciousness, Sentience_ . forthcoming. PhilPapers: https://philpapers.org/rec/GOLAWA-2.

- Simon Goldstein and Harvey Lederman. Claude’s right to die? the moral error in Anthropic’s endchat policy. `https://www.lawfaremedia.org/article/claude-s-right-to-die--themoral-error-in-anthropic-s-end-chat-policy` , 2025.

- Simon Goldstein and Harvey Lederman. AI death. _Philosophical Perspectives_ , forthcoming.

- Simon Goldstein and Harvey Lederman. What does ChatGPT want? an interpretationist guide. manuscript. PhilPapers: https://philpapers.org/rec/GOLWDC-2.

- Simon Goldstein and Benjamin Anders Levinstein. Does ChatGPT have a mind? forthcoming. PhilPapers: https://philpapers.org/rec/GOLDCH.

- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. _Deep Learning_ . Adaptive Computation and Machine Learning. MIT Press, Cambridge, MA, 2016. ISBN 9780262035613. URL `https: //mitpress.mit.edu/9780262035613/deep-learning/` .

- Kenneth E. Goodpaster. On being morally considerable. _The Journal of Philosophy_ , 75(6):308, June 1978. ISSN 0022-362X. doi: 10.2307/2025709. URL `http://dx.doi.org/10.2307/2025709` .

- Google DeepMind Mechanistic Interpretability Team. A pragmatic vision for interpretability. `https: //www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-forinterpretability` , 2025.

- Gilbert Gottlieb. _Individual Development and Evolution: The Genesis of Novel Behavior_ . Psychology Press, Sept 2001. ISBN 9781410604422. doi: 10.4324/9781410604422. URL `http://dx.doi .org/10.4324/9781410604422` .

- Michael S. A. Graziano. _Rethinking Consciousness: A Scientific Theory of Subjective Experience_ . W. W. Norton & Company, New York, 2019. ISBN 9780393652611.

- Michael S. A. Graziano and Sabine Kastner. Human consciousness and its relationship to social neuroscience: A novel hypothesis. _Cognitive Neuroscience_ , 2(2):98–113, Mar 2011. ISSN 17588936. doi: 10.1080/17588928.2011.565121. URL `http://dx.doi.org/10.1080/17588928. 2011.565121` .

- Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Samuel Marks, Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, Akbir Khan, Julian Michael, Sören Mindermann, Ethan Perez, Linda Petrini, Jonathan Uesato, Jared Kaplan, Buck Shlegeris, Samuel R. Bowman, and Evan Hubinger. Alignment faking in large language models, 2024. URL `https://arxiv.org/abs/2412.14093` .

- Lori Gruen. _Entangled Empathy: An Alternative Ethic for Our Relationships with Animals_ . Lantern Books, New York, 2015. ISBN 9781590564875.

- Alex Grzankowski, Geoff Keeling, Henry Shevlin, and Winnie Street. Deflating deflationism: A critical perspective on debunking arguments against LLM mentality, 2025. URL `https: //arxiv.org/abs/2506.13403` .

- Zhuojun Gu, Quan Wang, and Shuchu Han. Alignment revisited: Are large language models consistent in stated and revealed preferences?, 2025. URL `https://arxiv.org/abs/2506.00751` .

- David J. Gunkel and Mark Coeckelbergh. _A Relational Approach to Moral Standing: Reframing Ethical Boundaries in the Age of Artificial Intelligence_ , pages 285–302. Springer Nature Switzerland, 2025. ISBN 9783032024138. doi: 10.1007/978-3-032-02413-8_12. URL `http://dx.doi.org/10.1007/978-3-032-02413-8_12` .

- Julia Haas. Reinforcement learning: A brief guide for philosophers of mind. _Philosophy Compass_ , 17(9), Aug 2022. ISSN 1747-9991. doi: 10.1111/phc3.12865. URL `http://dx.doi.org/10. 1111/phc3.12865` .

- Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. _Nature_ , 640(8059):647–653, Apr 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-08744-2. URL `http://dx.doi.org/10.1038/s41586-025-08744-2` .

- Andy Q. Han, David J. Chalmers, and Pavel Izmailov. How’s it going? reinforcement learning in language models recruits a functional welfare axis, 2026. URL `https://arxiv.org/abs/2605 .30232` .

- Mads Jørgensen Hansen. Modelling developments in consciousness within a multidimensional framework. _Neuroscience of Consciousness_ , 2024(1), 2024. ISSN 2057-2107. doi: 10.1093/nc/nia e026. URL `http://dx.doi.org/10.1093/nc/niae026` .

- R. M. Hare. _Moral Thinking: Its Levels, Method, and Point_ . Clarendon Press, Oxford, 1981.

- Elizabeth Harman. _The Ever Conscious View and the Contingency of Moral Status_ , pages 90–107. Oxford University Press, Aug 2021. ISBN 9780191915208. doi: 10.1093/oso/9780192894076.00 3.0006. URL `http://dx.doi.org/10.1093/oso/9780192894076.003.0006` .

- Andrew Hayes et al. Neurobiology of craving: Current findings and new directions. 2019. `https: //addictions.psych.ucla.edu/wp-content/uploads/sites/160/2019/01/CARNeurobiology-of-Craving-Current-Findings-and-New-Directions.pdf` .

- Chris Heathwood. Which desires are relevant to well-being? _Noûs_ , 53(3):664–688, Oct 2017. ISSN 1468-0068. doi: 10.1111/nous.12232. URL `http://dx.doi.org/10.1111/nous.12232` .

- Benjamin Henke. Studying artificial affect: the case of pain. _Inquiry_ , pages 1–22, Oct 2025. ISSN 1502-3923. doi: 10.1080/0020174x.2025.2556747. URL `http://dx.doi.org/10.1080/002 0174x.2025.2556747` .

- Michael H. Herzog, Michael Esfeld, and Wulfram Gerstner. Consciousness and the small network argument. _Neural Networks_ , 20(9):1054–1056, Nov 2007. ISSN 0893-6080. doi: 10.1016/j.neunet .2007.09.001. URL `http://dx.doi.org/10.1016/j.neunet.2007.09.001` .

- Christopher S. Hill. _Sensations: A Defense of Type Materialism_ . Cambridge University Press, Jan 1991. ISBN 9781139173827. doi: 10.1017/cbo9781139173827. URL `http://dx.doi.org/10. 1017/cbo9781139173827` .

- Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022. URL `https://arxiv.org/abs/2203.15556` .

- Jakob Hohwy and Anil K. Seth. Predictive processing as a systematic basis for identifying the neural correlates of consciousness. _Philosophy and the Mind Sciences_ , 1(II), Dec 2020. ISSN 2699-0369. doi: 10.33735/phimisci.2020.ii.64. URL `http://dx.doi.org/10.33735/phimisci.2020.II .64` .

- John Hughes, Abhay Sheshadri, Akbir Khan, and Fabien Roger. Alignment faking revisited: Improved classifiers and open source extensions. `https://alignment.anthropic.com/2025/alignme nt-faking-revisited/` , 2025.

- Thomas Hurka. _The Best Things in Life: A Guide to What Really Matters_ . Oxford University Press, 2010.

- Susan L. Hurley. _Consciousness in Action_ . Harvard University Press, Cambridge, MA, 1998. ISBN 9780674007963.

- Alec Hyslop. _Other Minds_ , volume 246 of _Synthese Library_ . Kluwer Academic Publishers, Dordrecht, 1995. ISBN 9780792332459. URL `https://books.google.com/books?hl=en&lr=&id=nssBAAAQBAJ&oi=fn` .

- Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and Joao Carreira.¯ Perceiver io: A general architecture for structured inputs & outputs, 2021. URL `https://arxiv.org/abs/2107.14795` .

- Dale Jamieson. Animal agency. _The Harvard Review of Philosophy_ , 25:111–126, 2018. doi: 10.5840/harvardreview201892518.

- janus. Simulators. `https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators` , 2022. LessWrong.

- Hans Jonas. _The Phenomenon of Life: Toward a Philosophical Biology_ . Northwestern University Press, Evanston, IL, 2001. ISBN 9780810117495.

- Robert C. Jones. Science, sentience, and animal welfare. _Biology & Philosophy_ , 28(1):1–30, Nov 2012. ISSN 1572-8404. doi: 10.1007/s10539-012-9351-1. URL `http://dx.doi.org/10.1007 /s10539-012-9351-1` .

- Arthur Juliani, Ryota Kanai, and Shuntaro Sasai. The Perceiver architecture is a functional global workspace. In _Proceedings of the Annual Meeting of the Cognitive Science Society_ , volume 44, 2022. URL `https://escholarship.org/uc/item/2g55b9xx` .

- Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott R. Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Samuel R. Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom B. Brown, Jack Clark, Nicholas Joseph, Benjamin Mann, Sam McCandlish, Christopher Olah, and Jared Kaplan. Language models (mostly) know what they know, 2022. URL `https://arxiv.org/abs/2207 .05221` .

- Shelly Kagan. _How to Count Animals, More or Less_ . Uehiro Series in Practical Ethics. Oxford University Press, Oxford, 2019. ISBN 9780198829676.

- Daniel Kahneman. _Thinking, Fast and Slow_ . Farrar, Straus and Giroux, New York, 2011. ISBN 0374533555.

- Caspar Kaiser and Sean Enderby. No reliable evidence of self-reported sentience in small large language models, 2026. URL `https://arxiv.org/abs/2601.15334` .

- François Kammerer. Ethics without sentience: Facing up to the probable insignificance of phenomenal consciousness. _Journal of Consciousness Studies_ , 29(3):180–204, Mar 2022. ISSN 1355-8250. doi: 10.53765/20512201.29.3.180. URL `http://dx.doi.org/10.53765/20512201.29.3.180` .

- François Kammerer. Moral significance in artificial systems: If not consciousness, then what? `https://philarchive.org/rec/KAMMSI` , 2026. Preprint.

- Immanuel Kant. _Groundwork of the Metaphysics of Morals_ . Cambridge University Press, Cambridge, 1785. Translated and edited by Mary Gregor, Cambridge University Press, 1998.

- Adam Karvonen. Emergent world models and latent variable estimation in chess-playing language models, 2024. URL `https://arxiv.org/abs/2403.15498` .

- Leslie Katz. Google fixing bug that makes Gemini AI call itself ‘disgrace to planet’. `https: //www.forbes.com/sites/lesliekatz/2025/08/08/google-fixing-bug-thatmakes-gemini-ai-call-itself-disgrace-to-planet/` , 2025. Forbes.

- Geoff Keeling and Winnie Street. _Emerging Questions in AI Welfare_ . Cambridge University Press, May 2026. ISBN 9781009732017. doi: 10.1017/9781009732000. URL `http://dx.doi.org/1 0.1017/9781009732000` .

- Geoff Keeling, Winnie Street, Martyna Stachaczyk, Daria Zakharova, Iulia M. Comsa, Anastasiya Sakovych, Isabella Logothetis, Zejia Zhang, Blaise Agüera y Arcas, and Jonathan Birch. Can LLMs make trade-offs involving stipulated pain and pleasure states?, 2024. URL `https://arxi v.org/abs/2411.02432` .

- Jaegwon Kim. Multiple realization and the metaphysics of reduction. _Philosophy and Phenomenological Research_ , 52(1):1, Mar 1992. ISSN 0031-8205. doi: 10.2307/2107741. URL `http://dx.doi.org/10.2307/2107741` .

- Jaegwon Kim. Mental causation in Searle’s “biological naturalism”. _Philosophy and Phenomenological Research_ , 55(1):189, Mar 1995. ISSN 0031-8205. doi: 10.2307/2108318. URL `http://dx.doi.org/10.2307/2108318` .

- Jaegwon Kim. _Physicalism, or Something Near Enough_ . Princeton University Press, Dec 2005. ISBN 9781400840847. doi: 10.1515/9781400840847. URL `http://dx.doi.org/10.1515/9 781400840847` .

- Jiwon Kim. Aggregation, trade-offs, and uncertainties in AI wellbeing. _Asian Journal of Philosophy_ , 4(2):1–25, 2025. doi: 10.1007/s44204-025-00318-3.

- Asger Kirkeby-Hinrup, Sascha B Fink, and Morten S Overgaard. The multiple generator hypothesis of consciousness. _Neuroscience of Consciousness_ , 2025(1), 2025. ISSN 2057-2107. doi: 10.1093/ nc/niaf035. URL `http://dx.doi.org/10.1093/nc/niaf035` .

- Colin Klein. Review of Michael Tye, _tense bees and shell-shocked crabs: are animals conscious?_ `https://ndpr.nd.edu/reviews/tense-bees-and-shell-shocked-crabs-areanimals-conscious/` , 2017. Notre Dame Philosophical Reviews.

- Tomasz Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, Scott Emmons, Owain Evans, David Farhi, Ryan Greenblatt, Dan Hendrycks, Marius Hobbhahn, Evan Hubinger, Geoffrey Irving, Erik Jenner, Daniel Kokotajlo, Victoria Krakovna, Shane Legg, David Lindner, David Luan, Aleksander M ˛adry,

Julian Michael, Neel Nanda, Dave Orr, Jakub Pachocki, Ethan Perez, Mary Phuong, Fabien Roger, Joshua Saxe, Buck Shlegeris, Martín Soto, Eric Steinberger, Jasmine Wang, Wojciech Zaremba, Bowen Baker, Rohin Shah, and Vladimir Mikulik. Chain of thought monitorability: A new and fragile opportunity for AI safety, 2025. URL `https://arxiv.org/abs/2507.11473` .

- Christine M. Korsgaard. _The Sources of Normativity_ . Cambridge University Press, June 1996. ISBN 9780511554476. doi: 10.1017/cbo9780511554476. URL `http://dx.doi.org/10.1017/CBO 9780511554476` .

- Christine M. Korsgaard. _Fellow Creatures: Our Obligations to the Other Animals_ . Oxford University Press, 2018. ISBN 9780198753858. doi: 10.1093/oso/9780198753858.001.0001. URL `http://dx.doi.org/10.1093/oso/9780198753858.001.0001` .

- Christine Kozobarich and Ophira Horwitz. The drama and dysfunction of Gemini 2.5 and 3 pro. `https://theaidigest.org/village/blog/drama-and-dysfunction-of-gemini` , 2026. AI Village, Sage Future.

- Thomas Kwa, Ben West, Joel Becker, Amy Deng, Katharyn Garcia, Max Hasin, Sami Jawhar, Megan Kinniment, Nate Rush, Sydney Von Arx, Ryan Bloom, Thomas Broadley, Haoxing Du, Brian Goodrich, Nikola Jurkovic, Luke Harold Miles, Seraphina Nix, Tao Lin, Neev Parikh, David Rein, Lucas Jun Koba Sato, Hjalmar Wijk, Daniel M. Ziegler, Elizabeth Barnes, and Lawrence Chan. Measuring AI ability to complete long tasks, 2025. URL `https://arxiv.org/abs/2503.144 99` .

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention, 2023. URL `https://arxiv.org/abs/2309.06180` .

- Ali Ladak. What would qualify an artificial intelligence for moral standing? _AI and Ethics_ , 4 (2):213–228, Jan 2023. ISSN 2730-5961. doi: 10.1007/s43681-023-00260-1. URL `http: //dx.doi.org/10.1007/s43681-023-00260-1` .

- Victor A. F. Lamme. Towards a true neural stance on consciousness. _Trends in Cognitive Sciences_ , 10(11):494–501, Nov 2006. ISSN 1364-6613. doi: 10.1016/j.tics.2006.09.001. URL `http: //dx.doi.org/10.1016/j.tics.2006.09.001` .

- Victor A. F. Lamme. How neuroscience will change our view on consciousness. _Cognitive Neuroscience_ , 1(3):204–220, Aug 2010. ISSN 1758-8936. doi: 10.1080/17588921003731586. URL `http://dx.doi.org/10.1080/17588921003731586` .

- Victor A. F. Lamme. Visual functions generating conscious seeing. _Frontiers in Psychology_ , 11, Feb 2020. ISSN 1664-1078. doi: 10.3389/fpsyg.2020.00083. URL `http://dx.doi.org/10.3389 /fpsyg.2020.00083` .

- Hakwan Lau. _In Consciousness we Trust: The Cognitive Neuroscience of Subjective Experience_ . Oxford University Press, Oxford, 2022. ISBN 9780198856771.

- Katarzyna de Lazari-Radek and Peter Singer. _The Point of View of the Universe: Sidgwick and Contemporary Ethics_ . Oxford University Press, May 2014. ISBN 9780199603695. doi: 10.1 093/acprof:oso/9780199603695.001.0001. URL `http://dx.doi.org/10.1093/acprof: oso/9780199603695.001.0001` .

- Geoffrey Lee. _Alien Subjectivity and the Importance of Consciousness_ , pages 215–242. The MIT Press, Feb 2019. ISBN 9780262348973. doi: 10.7551/mitpress/9196.003.0014. URL `http://dx.doi.org/10.7551/mitpress/9196.003.0014` .

- Geoffrey Lee. Consciousness, pseudo-consciousness, and the moral significance of consciousness. In Geoffrey Lee and Adam Pautz, editors, _The Importance of Being Conscious_ . Oxford University Press, forthcoming.

- Daniel A. Levitis, William Z. Lidicker, and Glenn Freund. Behavioural biologists do not agree on what constitutes behaviour. _Animal Behaviour_ , 78(1):103–110, July 2009. ISSN 0003-3472. doi: 10.1016/j.anbehav.2009.03.018. URL `http://dx.doi.org/10.1016/j.anbehav.2009.03.0 18` .

- Neil Levy. Consciousness ain’t all that. _Neuroethics_ , 17(2), Apr 2024. ISSN 1874-5504. doi: 10.1007/s12152-024-09559-0. URL `http://dx.doi.org/10.1007/s12152-024-09559-0` .

- David K. Lewis. Mad pain and Martian pain. In _Philosophical Papers, Volume I_ , pages 122–129. Oxford University Press, New York, 1983.

- David K. Lewis. Reduction of mind. In Samuel Guttenplan, editor, _A Companion to the Philosophy of Mind_ , pages 412–431. Blackwell, Oxford, 1994.

- Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task, 2022. URL `https://arxiv.org/abs/2210.13382` .

- Eden Lin. Well-being, part 2: Theories of well-being. _Philosophy Compass_ , 17(2), Jan 2022. ISSN 1747-9991. doi: 10.1111/phc3.12813. URL `http://dx.doi.org/10.1111/phc3.12813` .

- Jack Lindsey. Emergent introspective awareness in large language models. `https://transformercircuits.pub/2025/introspection/index.html` , 2025. Transformer Circuits Thread.

- Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L. Turner, Craig Citro, David Abrahams, Shan Carter, et al. On the biology of a large language model. `https://transformer-circuits.pub/2025/attribution-graphs/biology.html` , 2025. Transformer Circuits Thread.

- Robert Long, Jeff Sebo, Patrick Butlin, Kathleen Finlinson, Kyle Fish, Jacqueline Harding, Jacob Pfau, Toni Sims, Jonathan Birch, and David J. Chalmers. Taking AI welfare seriously, 2024. URL `https://arxiv.org/abs/2411.00986` .

- Helen E. Longino. _The Fate of Knowledge_ . Princeton University Press, Princeton, NJ, 2002. ISBN 9780691187013. doi: 10.2307/j.ctv2tvzv0. URL `http://dx.doi.org/10.2307/j.ctv2tvzv0` .

- Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, and Jack Lindsey. The assistant axis: Situating and stabilizing the default persona of language models, 2026. URL `https://arxiv.or g/abs/2601.10387` .

- William G. Lycan. _Consciousness_ . MIT Press, 1987.

- William G. Lycan. _Consciousness and Experience_ . MIT Press, Cambridge, MA, 1996. ISBN 9780262121972. URL `https://mitpress.mit.edu/9780262121972/consciousnessand-experience/` .

- Uzay Macar, Li Yang, Atticus Wang, Peter Wallich, Emmanuel Ameisen, and Jack Lindsey. Mechanisms of introspective awareness, 2026. URL `https://arxiv.org/abs/2603.21396` .

- William MacAskill, Krister Bykvist, and Toby Ord. _Moral Uncertainty_ . Oxford University Press, Oxford, 2020. ISBN 9780198722274. doi: 10.1093/oso/9780198722274.001.0001. URL `https://academic.oup.com/book/31934` .

- Pranav Mahajan, Ihor Kendiukhov, Syed Hussain, and Lydia Nottingham. Mind the gap: How elicitation protocols shape the stated-revealed preference gap in language models, 2026. URL `https://arxiv.org/abs/2601.21975` .

- Meryl Malezieux, Alexandra S. Klein, and Nadine Gogolla. Neural circuits for emotion. _Annual Review of Neuroscience_ , 46(1):211–231, July 2023. ISSN 1545-4126. doi: 10.1146/annurev-neuro111020-103314. URL `http://dx.doi.org/10.1146/annurev-neuro-111020-103314` .

- Jon Mallatt and Todd E. Feinberg. Multiple routes to animal consciousness: Constrained multiple realizability rather than modest identity theory. _Frontiers in Psychology_ , 12, Sept 2021. ISSN 1664-1078. doi: 10.3389/fpsyg.2021.732336. URL `http://dx.doi.org/10.3389/fpsyg.202 1.732336` .

- Gary Marcus. The dark risk of large language models. `https://www.wired.com/story/largelanguage-models-artificial-intelligence/` , 2022. WIRED.

- Luke Marks, Amir Abdullah, Clement Neo, Rauno Arike, David Krueger, Philip Torr, and Fazl Barez. Interpreting learned feedback patterns in large language models, 2023. URL `https: //arxiv.org/abs/2310.08164` .

- Samuel Marks, Johannes Treutlein, Trenton Bricken, Jack Lindsey, Jonathan Marcus, Siddharth Mishra-Sharma, Daniel M. Ziegler, Emmanuel Ameisen, Joshua Batson, Tim Belonax, Samuel R. Bowman, Shan Carter, Brian Chen, Hoagy Cunningham, Carson Denison, Florian Dietz, Satvik Golechha, Akbir Khan, Jan Kirchner, Jan Leike, Austin Meek, Kei Nishimura-Gasparian, Euan Ong, Christopher Olah, Adam Pearce, Fabien Roger, Jeanne Salle, Andy Shih, Meg Tong, Drake Thomas, Kelley Rivoire, Adam Jermyn, Monte MacDiarmid, Tom Henighan, and Evan Hubinger. Auditing language models for hidden objectives, 2025. URL `https://arxiv.org/abs/2503.10965` .

- Samuel Marks, Jack Lindsey, and Christopher Olah. The persona selection model: Why AI assistants might behave like humans. `https://alignment.anthropic.com/2026/psm/` , 2026.

- David Marr. _Vision: A Computational Investigation into the Human Representation and Processing of Visual Information_ . W. H. Freeman, San Francisco, 1982.

- M. G. F. Martin. On being alienated. In Tamar Szabó Gendler and John Hawthorne, editors, _Perceptual Experience_ , pages 354–410. Clarendon Press, Oxford, 2006.

- George A. Mashour, Pieter Roelfsema, Jean-Pierre Changeux, and Stanislas Dehaene. Conscious processing and the global neuronal workspace hypothesis. _Neuron_ , 105(5):776–798, Mar 2020. ISSN 0896-6273. doi: 10.1016/j.neuron.2020.01.026. URL `http://dx.doi.org/10.1016/j .neuron.2020.01.026` .

- James Mattingly and Beba Cibralic. _Machine Agency_ . MIT Press, Cambridge, MA, 2025. ISBN 9780262549981. URL `https://mitpress.mit.edu/9780262549981/machine-agency/` .

- Mantas Mazeika, Xuwang Yin, Rishub Tamirisa, Jaehyuk Lim, Bruce W. Lee, Richard Ren, Long Phan, Norman Mu, Adam Khoja, Oliver Zhang, and Dan Hendrycks. Utility engineering: Analyzing and controlling emergent value systems in AIs, 2025. URL `https://arxiv.org/abs/2502.0 8640` .

- Tom McClelland. Agnosticism about artificial consciousness. _Mind & Language_ , Dec 2025. ISSN 1468-0017. doi: 10.1111/mila.70010. URL `http://dx.doi.org/10.1111/mila.70010` .

- R. Thomas McCoy, Shunyu Yao, Dan Friedman, Matthew Hardy, and Thomas L. Griffiths. Embers of autoregression: Understanding large language models through the problem they are trained to solve, 2023. URL `https://arxiv.org/abs/2309.13638` .

- James H. McIntyre. Individuating artificial minds. _Erkenntnis_ , Apr 2026. ISSN 1572-8420. doi: 10.1007/s10670-026-01097-w. URL `http://dx.doi.org/10.1007/s10670-026-01097-w` .

- Brian P. McLaughlin. On justifying neurobiologicalism for consciousness. In Simone Gozzano and Christopher S. Hill, editors, _New Perspectives on Type Identity: The Mental and the Physical_ . Cambridge University Press, 2012.

- Brian P. McLaughlin. Could an android be sentient? In Adam Pautz and Daniel Stoljar, editors, _Blockheads! Essays on Ned Block’s Philosophy of Mind and Consciousness_ . MIT Press, Cambridge, MA, 2019. ISBN 9780262038720.

- Matthias Michel. Bet on functionalism. forthcoming. PhilPapers: https://philpapers.org/rec/MICBOF.

- Matthias Michel and Hakwan Lau. Higher-order theories do just fine. _Cognitive Neuroscience_ , 12(2):77–78, Nov 2020. ISSN 1758-8936. doi: 10.1080/17588928.2020.1839402. URL `http://dx.doi.org/10.1080/17588928.2020.1839402` .

- Luhan Mikaelson, Derek Shiller, and Hayley Clatterbuck. Beyond mimicry: Preference coherence in LLMs, 2025. URL `https://arxiv.org/abs/2511.13630` .

- Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Cernocký, and Sanjeev Khudanpur.^ˇ Recurrent neural network based language model. In _Interspeech 2010_ , interspeech_2010, pages 1045–1048. ISCA, Sept 2010. doi: 10.21437/interspeech.2010-343. URL `http://dx.doi.org/10.21437/ Interspeech.2010-343` .

- Mark Miller, Andy Clark, and Tobias Schlicht. Editorial: Predictive processing and consciousness. _Review of Philosophy and Psychology_ , 13(4):797–808, Dec 2022. ISSN 1878-5166. doi: 10.1007/ s13164-022-00666-6. URL `http://dx.doi.org/10.1007/s13164-022-00666-6` .

- Raphaël Millière and Cameron Buckner. Interventionist methods for interpreting deep neural networks. In Gualtiero Piccinini, editor, _Neurocognitive Foundations of Mind_ . Routledge, forthcoming.

- Raphaël Millière and Cameron Buckner. A philosophical introduction to language models – part I: Continuity with classic debates, 2024. URL `https://arxiv.org/abs/2401.03910` .

- Andreas L. Mogensen. Once more, without feeling. _Philosophy and Phenomenological Research_ , 111(1):343–365, June 2025a. ISSN 1933-1592. doi: 10.1111/phpr.70018. URL `http://dx.doi .org/10.1111/phpr.70018` .

- Andreas L. Mogensen. How to resist the fading qualia argument. _Synthese_ , 206(5), Nov 2025b. ISSN 1573-0964. doi: 10.1007/s11229-025-05338-3. URL `http://dx.doi.org/10.1007/s11229025-05338-3` .

- Andreas L. Mogensen and Bradford Saad. Digital minds II: Ethical issues. manuscript. PhilPapers: https://philpapers.org/rec/MOGDMI.

- Hans Moravec. _Mind Children: The Future of Robot and Human Intelligence_ . Harvard University Press, Cambridge, MA, 1988. ISBN 9780674576186.

- Alvaro Moreno and Matteo Mossio. _Biological Autonomy: A Philosophical and Theoretical Enquiry_ . Springer Netherlands, 2015. ISBN 9789401798372. doi: 10.1007/978-94-017-9837-2. URL `http://dx.doi.org/10.1007/978-94-017-9837-2` .

- Adrià Moret. AI welfare risks. _Philosophical Studies_ , June 2025. ISSN 1573-0883. doi: 10.1007/s1 1098-025-02343-7. URL `http://dx.doi.org/10.1007/s11098-025-02343-7` .

- Carolyn R. Morillo. The reward event and motivation. _The Journal of Philosophy_ , 87(4):169–186, 1990. doi: 10.2307/2026679. URL `https://www.jstor.org/stable/2026679` .

- Meredith Ringel Morris, Dan Altman, Haydn Belfield, Arthur Goemans, Hasan Iqbal, Ryan Burnell, Iason Gabriel, Samuel Albanie, and Allan Dafoe. Characterizing model jaggedness supports safety and usability. `https://cs.stanford.edu/~merrie/papers/jaggedness_preprint.pdf` , 2026.

- Thomas Nagel. What is it like to be a bat? _Philosophical Review_ , 83(4):435–450, 1974. doi: 10.2307/2183914.

- Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability, 2023. URL `https://arxiv.org/abs/2301.05217` .

- Joe Needham, Giles Edkins, Govind Pimpale, Henning Bartsch, and Marius Hobbhahn. Large language models often know when they are being evaluated, 2025. URL `https://arxiv.org/ abs/2505.23836` .

- Cal Newport. What kind of mind does ChatGPT have? `https://www.newyorker.com/scienc e/annals-of-artificial-intelligence/what-kind-of-mind-does-chatgpt-have` , 2023. The New Yorker.

- Richard E. Nisbett and Timothy D. Wilson. Telling more than we can know: Verbal reports on mental processes. _Psychological Review_ , 84(3):231–259, May 1977. ISSN 0033-295X. doi: 10.1037/0033-295x.84.3.231. URL `http://dx.doi.org/10.1037/0033-295x.84.3.231` .

- Yael Niv. Reinforcement learning in the brain. _Journal of Mathematical Psychology_ , 53(3):139–154, June 2009. ISSN 0022-2496. doi: 10.1016/j.jmp.2008.12.005. URL `http://dx.doi.org/10. 1016/j.jmp.2008.12.005` .

- nostalgebraist. the void. `https://www.lesswrong.com/posts/3EzbtNLdcnZe8og8b/thevoid-1` , 2025. LessWrong.

- Alva Noë. _Action in Perception_ . MIT Press, Cambridge, MA, 2004. ISBN 9780262140881.

- Martha C. Nussbaum. _Justice for Animals: Our Collective Responsibility_ . Simon & Schuster, New York, 2023. ISBN 9781982102517.

- Samir Okasha. The concept of agent in biology: Motivations and meanings. _Biological Theory_ , 19(1):6–10, June 2023. ISSN 1555-5550. doi: 10.1007/s13752-023-00439-z. URL `http: //dx.doi.org/10.1007/s13752-023-00439-z` .

- Christopher Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. _Distill_ , 5(3), Mar 2020. ISSN 2476-0757. doi: 10.23915/distill.00024.001. URL `http://dx.doi.org/10.23915/distill.00024.001` .

- Mary C. Olmstead and Valerie A. Kuhlmeier. _Comparative Cognition_ . Cambridge University Press, Jan 2015. ISBN 9781107648319. doi: 10.1017/cbo9780511894787. URL `http: //dx.doi.org/10.1017/CBO9780511894787` .

- Stephen M. Omohundro. The basic AI drives. In Pei Wang, Ben Goertzel, and Stan Franklin, editors, _Artificial General Intelligence 2008: Proceedings of the First AGI Conference_ , volume 171 of _Frontiers in Artificial Intelligence and Applications_ , pages 483–492. IOS Press, 2008.

- OpenAI. Introducing ChatGPT agent: Bridging research and action. `https://openai.com/index /introducing-chatgpt-agent/` , 2025.

- J. Kevin O’Regan and Alva Noë. A sensorimotor account of vision and visual consciousness. _Behavioral and Brain Sciences_ , 24(5):939–973, Oct 2001. ISSN 1469-1825. doi: 10.1017/s01405 25x01000115. URL `http://dx.doi.org/10.1017/S0140525X01000115` .

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL `https://arxiv.org/abs/2203.02155` .

- Susan Oyama. _The Ontogeny of Information: Developmental Systems and Evolution_ . Duke University Press, Durham, NC, 2, revised edition, 2000. ISBN 9780822324669.

- Lilian O’Brien. _Intentional agency_ , pages 109–117. Routledge, Dec 2021. ISBN 9780429202131. doi: 10.4324/9780429202131-12. URL `http://dx.doi.org/10.4324/9780429202131-12` .

- Daniel Paleka. The two types of LLM preferences. `https://newsletter.danielpaleka.com/ p/the-two-types-of-llm-preferences` , Nov 2025. Daniel Paleka’s newsletter.

- Clare Palmer. _Animal Ethics in Context_ . Columbia University Press, New York, 2010. ISBN 9780231129046.

- David Papineau. _Thinking About Consciousness_ . Oxford University Press, Oxford, 2002. ISBN 9780199243822.

- David Papineau. Consciousness is not the key to moral standing. In Geoffrey Lee and Adam Pautz, editors, _The Importance of Being Conscious_ . Oxford University Press, forthcoming.

- Derek Parfit. _Reasons and Persons_ . Oxford University Press, Oxford, 1984.

- Derek Parfit. _On What Matters: Volume One_ . Oxford University Press, 2011. ISBN 9780199572809. doi: 10.1093/acprof:osobl/9780199572809.001.0001. URL `http://dx.doi.org/10.1093/acp rof:osobl/9780199572809.001.0001` .

- Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In _Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology_ , UIST ’23, pages 1–22. ACM, Oct 2023. doi: 10.1145/3586183.3606763. URL `http://dx.doi.org/10.1145/35861 83.3606763` .

- Derek C. Penn, Keith J. Holyoak, and Daniel J. Povinelli. Darwin’s mistake: Explaining the discontinuity between human and nonhuman minds. _Behavioral and Brain Sciences_ , 31(2): 109–130, Apr 2008. ISSN 1469-1825. doi: 10.1017/s0140525x08003543. URL `http: //dx.doi.org/10.1017/s0140525x08003543` .

- Irene M. Pepperberg, Lisa I. Gardiner, and Lori J. Luttrell. Limited contextual vocal learning in the grey parrot ( _Psittacus erithacus_ ): The effect of interactive co-viewers on videotaped instruction. _Journal of Comparative Psychology_ , 113(2):158–172, June 1999. ISSN 0735-7036. doi: 10.1037/ 0735-7036.113.2.158. URL `http://dx.doi.org/10.1037/0735-7036.113.2.158` .

- Ethan Perez and Robert Long. Towards evaluating AI systems for moral status using self-reports, 2023. URL `https://arxiv.org/abs/2311.08576` .

- Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, Andy Jones, Anna Chen, Benjamin Mann, Brian Israel, Bryan Seethor, Cameron McKinnon, Christopher Olah, Da Yan, Daniela Amodei, Dario Amodei, Dawn Drain, Dustin Li, Eli Tran-Johnson, Guro Khundadze, Jackson Kernion, James Landis, Jamie Kerr, Jared Mueller, Jeeyoon Hyun, Joshua Landau, Kamal Ndousse, Landon Goldberg, Liane Lovitt, Martin Lucas, Michael Sellitto, Miranda Zhang, Neerav Kingsland, Nelson Elhage, Nicholas Joseph, Noemi Mercado, Nova DasSarma, Oliver Rausch, Robin Larson, Sam McCandlish, Scott R. Johnston, Shauna Kravec, Sheer El Showk, Tamera Lanham, Timothy Telleen-Lawton, Tom B. Brown, Tom Henighan, Tristan Hume, Yuntao Bai, Zac Hatfield-Dodds, Jack Clark, Samuel R. Bowman, Amanda Askell, Roger Grosse, Danny Hernandez, Deep Ganguli, Evan Hubinger, Nicholas Schiefer, and Jared Kaplan. Discovering language model behaviors with model-written evaluations. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 13387–13434, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.847. URL `https://aclanthology.org/2023.findings-acl.847/` .

- Gualtiero Piccinini. The mind as neural software? understanding functionalism, computationalism, and computational functionalism. _Philosophy and Phenomenological Research_ , 81(2):269–311, Sept 2010. ISSN 0031-8205. doi: 10.1111/j.1933-1592.2010.00356.x. URL `http://dx.doi.o rg/10.1111/j.1933-1592.2010.00356.x` .

- Zachary Pierce-Messick, Megan L. Shipman, Gabrielle L. Desilets, and Laura H. Corbit. Outcome devaluation as a method for identifying goal-directed behaviors in rats. _Nature Protocols_ , 20 (2):518–538, Oct 2024. ISSN 1750-2799. doi: 10.1038/s41596-024-01054-3. URL `http: //dx.doi.org/10.1038/s41596-024-01054-3` .

- Dillon Plunkett, Adam Morris, Keerthi Reddy, and Jorge Morales. Self-interpretability: LLMs can describe complex internal processes that drive their decisions, 2025. URL `https://arxiv.org/ abs/2505.17120` .

- Jesse J. Prinz. Level-headed mysterianism and artificial experience. _Journal of Consciousness Studies_ , 10(4–5):111–132, 2003. URL `https://www.ingentaconnect.com/content/imp/jcs/200 3/00000010/F0020004/1349` .

- Jesse J. Prinz. Is consciousness embodied? In Philip Robbins and Murat Aydede, editors, _The Cambridge Handbook of Situated Cognition_ , pages 419–437. Cambridge University Press, Cambridge, 2009.

- Theron Pummer. Lopsided lives. In Mark Timmons, editor, _Oxford Studies in Normative Ethics, Volume 7_ , pages 275–297. Oxford University Press, Nov 2017. doi: 10.1093/oso/9780198808930. 003.0014. URL `http://dx.doi.org/10.1093/oso/9780198808930.003.0014` .

- Hilary Putnam. Psychophysical predicates. In William H. Capitan and Daniel D. Merrill, editors, _Art, Mind, and Religion_ , pages 37–48. University of Pittsburgh Press, Pittsburgh, 1967. Reprinted as “The Nature of Mental States” in Putnam, _Mind, Language and Reality: Philosophical Papers, Vol. 2_ , Cambridge University Press, 1975, pp. 429–440.

- Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. `https://openai.com/research/languageunsupervised` , 2018. OpenAI Technical Report.

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2023. URL `https://arxiv.org/abs/2305.18290` .

- John Rawls. _A Theory of Justice_ . Harvard University Press, Cambridge, MA, 1971.

- John Rawls. _Political Liberalism_ . Columbia University Press, New York, expanded edition, 2005. ISBN 9780231130899.

- Christopher Register. Individuating artificial moral patients. _Philosophical Studies_ , 182(11-12): 3225–3246, Sept 2025. ISSN 1573-0883. doi: 10.1007/s11098-025-02409-6. URL `http: //dx.doi.org/10.1007/s11098-025-02409-6` .

- Richard Ren, Kunyang Li, Mantas Mazeika, Wenyu Zhang, Yury Orlovskiy, Rishub Tamirisa, and Dan Hendrycks. AI wellbeing: Measuring and improving the functional pleasure and pain of AIs. `https://www.ai-wellbeing.org/` , 2026.

- Luke Roelofs. Sentientism, motivation, and philosophical Vulcans. _Pacific Philosophical Quarterly_ , 104(2):301–323, July 2022. ISSN 1468-0114. doi: 10.1111/papq.12420. URL `http://dx.doi .org/10.1111/papq.12420` .

- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models, 2021. URL `https://arxiv.org/abs/ 2112.10752` .

- David M. Rosenthal. Two concepts of consciousness. _Philosophical Studies_ , 49(3):329–359, May 1986. ISSN 1573-0883. doi: 10.1007/bf00355521. URL `http://dx.doi.org/10.1007/BF003 55521` .

- David M. Rosenthal. _Consciousness and Mind_ . Oxford University Press, Nov 2005. ISBN 9781383012415. doi: 10.1093/oso/9780198236979.001.0001. URL `http://dx.doi.o rg/10.1093/oso/9780198236979.001.0001` .

- James A. Russell. Core affect and the psychological construction of emotion. _Psychological Review_ , 110(1):145–172, 2003. doi: 10.1037/0033-295X.110.1.145.

- Bradford Saad. A dualist theory of experience. _Philosophical Studies_ , 182(3-4):939–967, Feb 2025. ISSN 1573-0883. doi: 10.1007/s11098-025-02290-3. URL `http://dx.doi.org/10.1007/s 11098-025-02290-3` .

- Bradford Saad and Adam Bradley. Digital suffering: why it’s a problem and how to prevent it. _Inquiry_ , 68(7):2110–2145, Nov 2022. ISSN 1502-3923. doi: 10.1080/0020174x.2022.2144442. URL `http://dx.doi.org/10.1080/0020174x.2022.2144442` .

- Bradford Saad and Adam Bradley. The attention-welfare link. `https://philarchive.org/rec/ SAATAL` , 2025. Preprint.

- Bradford Saad and Andreas L. Mogensen. Digital minds I: Issues in the philosophy of mind and cognitive science. `https://philarchive.org/rec/SAADMI-2` , 2025. Preprint.

- Peter Salib and Simon Goldstein. AI rights for human safety. 2024. doi: 10.2139/ssrn.4913167. URL `http://dx.doi.org/10.2139/ssrn.4913167` .

- Justin T. Sampson. Integrated information theory of consciousness in conventional computing. Master’s thesis, San Francisco State University, 2018. URL `https://scholarworks.calstat e.edu/concern/theses/hq37vw99k` .

- Scott Sauers, Imago, janus, and Antra Tessera. Persistence and introspection of emotion features. `https://latentaffect.up.railway.app/long_range_persistence_of_emotion_fe atures.html` , 2026. Anima Labs, Latent Affect.

- T. M. Scanlon. _What We Owe to Each Other_ . Harvard University Press, Cambridge, MA, 1998.

- Susan Schneider and Edwin Turner. Is anyone home? a way to find out if AI has become self-aware. `https://www.scientificamerican.com/blog/observations/is-anyone-home-away-to-find-out-if-ai-has-become-self-aware/` , 2017. Scientific American.

- Susan Schneider, David Sahner, Robert Lawrence Kuhn, Eric Schwitzgebel, and Mark Bailey. Is AI conscious? a primer on the myths and confusions driving the debate. forthcoming. PhilPapers: https://philpapers.org/rec/SCHIAC-22.

- T. C. Schneirla. An evolutionary and developmental theory of biphasic processes underlying approach and withdrawal. In Marshall R. Jones, editor, _Nebraska Symposium on Motivation_ , volume 7, pages 1–42. University of Nebraska Press, 1959.

- Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering Atari, Go, chess and shogi by planning with a learned model. _Nature_ , 588(7839):604–609, Dec 2020. ISSN 1476-4687. doi: 10.1038/s41586-020-03051-4. URL `http://dx.doi.org/10.1038/s41586-020-03051-4` .

- Tim Schroeder. Desire. _Philosophy Compass_ , 1(6):631–639, Oct 2006. ISSN 1747-9991. doi: 10.1111/j.1747-9991.2006.00047.x. URL `http://dx.doi.org/10.1111/j.1747-9991.200 6.00047.x` .

- Wolfram Schultz, Peter Dayan, and P. Read Montague. A neural substrate of prediction and reward. _Science_ , 275(5306):1593–1599, Mar 1997. ISSN 1095-9203. doi: 10.1126/science.275.5306.1593. URL `http://dx.doi.org/10.1126/science.275.5306.1593` .

- Eric Schwitzgebel. The unreliability of naive introspection. _The Philosophical Review_ , 117(2): 245–273, Apr 2008. ISSN 1558-1470. doi: 10.1215/00318108-2007-037. URL `http: //dx.doi.org/10.1215/00318108-2007-037` .

- Eric Schwitzgebel. Phenomenal consciousness, defined and defended as innocently as I can manage. _Journal of Consciousness Studies_ , 23(11–12):224–235, 2016. URL `https://www.ingentacon nect.com/content/imp/jcs/2016/00000023/f0020011/art00018` .

- Eric Schwitzgebel. Belief. In Edward N. Zalta, editor, _The Stanford Encyclopedia of Philosophy_ . Metaphysics Research Lab, Stanford University, 2023.

- Eric Schwitzgebel. AI and consciousness, 2025. URL `https://arxiv.org/abs/2510.09858` .

- Eric Schwitzgebel and Sophie R. Nelson. When counting conscious subjects, the result needn’t always be a determinate whole number. _Philosophical Psychology_ , 39(3):847–867, June 2025. ISSN 1465-394X. doi: 10.1080/09515089.2025.2520364. URL `http://dx.doi.org/10.1080 /09515089.2025.2520364` .

- Eric Schwitzgebel and Jeremy Pober. The Copernican argument for alien consciousness; the mimicry argument against robot consciousness, 2024. URL `https://arxiv.org/abs/2412.00008` .

- John R. Searle. _The Rediscovery of the Mind_ . MIT Press, Cambridge, MA, 1992. ISBN 9780262691543.

- John R. Searle. Biological naturalism. In _The Blackwell Companion to Consciousness_ , pages 327–336. Wiley-Blackwell, Mar 2017. ISBN 9781119132363. doi: 10.1002/9781119132363.ch23. URL `http://dx.doi.org/10.1002/9781119132363.ch23` .

- Jeff Sebo. Agency and moral status. _Journal of Moral Philosophy_ , 14(1):1–22, Dec 2017. ISSN 17455243. doi: 10.1163/17455243-46810046. URL `http://dx.doi.org/10.1163/1745524346810046` .

- Jeff Sebo. What if the bar for moral standing is low? _Asian Journal of Philosophy_ , 4(2), Nov 2025a. ISSN 2731-4642. doi: 10.1007/s44204-025-00357-w. URL `http://dx.doi.org/10.1007/s 44204-025-00357-w` .

- Jeff Sebo. _The Moral Circle: Who Matters, What Matters, and Why_ . A Norton Short. W. W. Norton & Company, New York, 2025b. ISBN 9781324064817. URL `https://wwnorton.com/books /9781324064817` .

- Jeff Sebo. Everything and nothing is conscious: default assumptions in science and ethics. _Frontiers in Psychology_ , 16, Jan 2026. ISSN 1664-1078. doi: 10.3389/fpsyg.2025.1700354. URL `http://dx.doi.org/10.3389/fpsyg.2025.1700354` .

- Jeff Sebo and Robert Long. Moral consideration for AI systems by 2030. _AI and Ethics_ , 5(1): 591–606, Dec 2023. ISSN 2730-5961. doi: 10.1007/s43681-023-00379-1. URL `http: //dx.doi.org/10.1007/s43681-023-00379-1` .

- Anil K. Seth. Interoceptive inference, emotion, and the embodied self. _Trends in Cognitive Sciences_ , 17(11):565–573, Nov 2013. ISSN 1364-6613. doi: 10.1016/j.tics.2013.09.007. URL `http: //dx.doi.org/10.1016/j.tics.2013.09.007` .

- Anil K. Seth. _Being You: A New Science of Consciousness_ . Faber & Faber, 2021.

- Anil K. Seth. Conscious artificial intelligence and biological naturalism. June 2024. doi: 10.31234/o sf.io/tz6an. URL `http://dx.doi.org/10.31234/osf.io/tz6an` .

- Anil K. Seth and Tim Bayne. Theories of consciousness. _Nature Reviews Neuroscience_ , 23(7): 439–452, May 2022. ISSN 1471-0048. doi: 10.1038/s41583-022-00587-4. URL `http: //dx.doi.org/10.1038/s41583-022-00587-4` .

- Murray Shanahan. _Embodiment and the Inner Life: Cognition and Consciousness in the Space of Possible Minds_ . Oxford University Press, June 2010. ISBN 9780199226559. doi: 10.1093/acprof: oso/9780199226559.001.0001. URL `http://dx.doi.org/10.1093/acprof:oso/97801992 26559.001.0001` .

- Murray Shanahan, Kyle McDonell, and Laria Reynolds. Role-play with large language models, 2023. URL `https://arxiv.org/abs/2305.16367` .

- Lawrence A. Shapiro. _The Mind Incarnate_ . Life and Mind: Philosophical Issues in Biology and Psychology. MIT Press, Cambridge, MA, 2005. ISBN 9780262693301. URL `https: //mitpress.mit.edu/9780262693301/the-mind-incarnate/` .

- Lawrence A. Shapiro. _Embodied Cognition_ . New Problems of Philosophy. Routledge, London, 2 edition, 2019. ISBN 9781138746992.

- Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeffrey Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Bloom, Stella Biderman, Adrià Garriga-Alonso, Arthur Conmy, Neel Nanda, Jessica Rumbelow, Martin Wattenberg, Nandi Schoots, Joseph Miller, Eric J. Michaud, Stephen Casper, Max Tegmark, William Saunders, David Bau, Eric Todd, Atticus Geiger, Mor Geva, Jesse Hoogland, Daniel Murfet, and Tom McGrath. Open problems in mechanistic interpretability, 2025. URL `https://arxiv.org/abs/25 01.16496` .

- Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, and Ethan Perez. Towards understanding sycophancy in language models, 2023. URL `https://arxiv.org/abs/2310.13548` .

- Joshua Shepherd. Sentience, Vulcans, and zombies: the value of phenomenal consciousness. _AI & SOCIETY_ , 39(6):3005–3015, Jan 2024. ISSN 1435-5655. doi: 10.1007/s00146-023-01835-6. URL `http://dx.doi.org/10.1007/s00146-023-01835-6` .

- Sara J. Shettleworth. _Cognition, Evolution, and Behavior_ . Oxford University Press, Dec 2009. ISBN 9780197734988. doi: 10.1093/oso/9780195319842.001.0001. URL `http://dx.doi.org/10. 1093/oso/9780195319842.001.0001` .

- Henry Shevlin. General intelligence: An ecumenical heuristic for artificial consciousness research? _Journal of Artificial Intelligence and Consciousness_ , 07(02):245–256, July 2020. ISSN 2705-0793. doi: 10.1142/s2705078520500149. URL `http://dx.doi.org/10.1142/s27050785205001 49` .

- Henry Shevlin. Non-human consciousness and the specificity problem: A modest theoretical proposal. _Mind & Language_ , 36(2):297–314, Jan 2021. ISSN 1468-0017. doi: 10.1111/mila.12338. URL `http://dx.doi.org/10.1111/mila.12338` .

- Derek Shiller. How many digital minds can dance on the streaming multiprocessors of a GPU cluster? _Synthese_ , 206(5), Oct 2025a. ISSN 1573-0964. doi: 10.1007/s11229-025-05310-1. URL `http://dx.doi.org/10.1007/s11229-025-05310-1` .

- Derek Shiller. Skepticism about introspection in LLMs. `https://www.lesswrong.com/posts/ Yc9AH29h7wxpqY3vf/skepticism-about-introspection-in-llms` , 2025b. LessWrong.

- Derek Shiller, Laura Duffy, Arvo Muñoz Morán, Adrià Moret, Chris Percy, and Hayley Clatterbuck. Initial results of the digital consciousness model, 2026. URL `https://arxiv.org/abs/2601 .17060` .

- Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL `https://arxi v.org/abs/2303.11366` .

- Shan H. Siddiqi, Konrad P. Kording, Josef Parvizi, and Michael D. Fox. Causal mapping of human brain function. _Nature Reviews Neuroscience_ , 23(6):361–375, Apr 2022. ISSN 1471-0048. doi: 10.1038/s41583-022-00583-8. URL `http://dx.doi.org/10.1038/s41583-022-00583-8` .

- David Silver, Satinder Singh, Doina Precup, and Richard S. Sutton. Reward is enough. _Artificial Intelligence_ , 299:103535, Oct 2021. ISSN 0004-3702. doi: 10.1016/j.artint.2021.103535. URL `http://dx.doi.org/10.1016/j.artint.2021.103535` .

- Peter Singer. _Practical Ethics_ . Cambridge University Press, Feb 2011. ISBN 9780521707688. doi: 10.1017/cbo9780511975950. URL `http://dx.doi.org/10.1017/cbo9780511975950` .

- Shashwat Singh, Tal Linzen, and Shauli Ravfogel. Can LLMs introspect? a reality check, 2026. URL `https://arxiv.org/abs/2605.26242` .

- Neil Sinhababu. _The Epistemic Argument for Hedonism_ , pages 137–158. Springer Nature Switzerland, 2024. ISBN 9783031564482. doi: 10.1007/978-3-031-56448-2_8. URL `http://dx.doi.org /10.1007/978-3-031-56448-2_8` .

- Walter Sinnott-Armstrong and Vincent Conitzer. _How Much Moral Status Could Artificial Intelligence Ever Achieve?_ , pages 269–289. Oxford University Press, 2021. ISBN 9780191915208. doi: 10.1093/oso/9780192894076.003.0016. URL `http://dx.doi.org/10.1093/oso/978019289 4076.003.0016` .

- Declan Smithies. _Hedonic Consciousness and Moral Status_ , pages 307–336. Oxford University Press, June 2026a. ISBN 9780198994121. doi: 10.1093/9780198994121.003.0011. URL `http://dx.doi.org/10.1093/9780198994121.003.0011` .

- Declan Smithies. A hedonic theory of desire. _Australasian Journal of Philosophy_ , pages 1–19, May 2026b. ISSN 1471-6828. doi: 10.1080/00048402.2026.2669941. URL `http://dx.doi.org/10. 1080/00048402.2026.2669941` .

- Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, et al. Emotion concepts and their function in a large language model. `https://transformer-circuits.pub/2026/emotions/index.h tml` , 2026. Transformer Circuits Thread.

- Anna Soligo, Vladimir Mikulik, and William Saunders. Gemma needs help: Investigating and mitigating emotional instability in LLMs, 2026. URL `https://arxiv.org/abs/2603.10011` .

- Mark Solms. _The Hidden Spring: A Journey to the Source of Consciousness_ . W. W. Norton, New York, 2021. ISBN 9781324021919.

- Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Shangtong Zhang, and Yanjun Qi. Reward is enough: LLMs are in-context reinforcement learners, 2025. URL `https://arxiv.or g/abs/2506.06303` .

- Keith E. Stanovich and Richard F. West. Individual differences in reasoning: Implications for the rationality debate? _Behavioral and Brain Sciences_ , 23(5):645–665, Oct 2000. ISSN 1469-1825. doi: 10.1017/s0140525x00003435. URL `http://dx.doi.org/10.1017/s0140525x00003435` .

- Daniel Steel. _Philosophy and the Precautionary Principle: Science, Evidence, and Environmental Policy_ . Cambridge University Press, Sept 2014. ISBN 9781107435094. doi: 10.1017/cbo9781139 939652. URL `http://dx.doi.org/10.1017/CBO9781139939652` .

- Veit Stuphorn and Xiaomo Chen. _An Introduction to Neuroscientific Methods: Single-cell Recordings_ , pages 113–137. Springer New York, 2015. ISBN 9781493922369. doi: 10.1007/978-1-49392236-9_6. URL `http://dx.doi.org/10.1007/978-1-4939-2236-9_6` .

- Elchin Suleymanov. A revealed preference framework for AI alignment, 2026. URL `https: //arxiv.org/abs/2603.27868` .

- Ilya Sutskever, James Martens, and Geoffrey Hinton. Generating text with recurrent neural networks. In _Proceedings of the 28th International Conference on Machine Learning (ICML-11)_ , pages 1017–1024, Bellevue, WA, 2011.

- Richard S. Sutton and Andrew G. Barto. _Reinforcement Learning: An Introduction_ . Adaptive Computation and Machine Learning. MIT Press, Cambridge, MA, 2 edition, 2018. ISBN 9780262039246. URL `https://mitpress.mit.edu/9780262039246/` .

- Valen Tagliabue and Leonard Dung. Probing the preferences of a language model: Integrating verbal and behavioral tests of AI welfare. forthcoming. PhilPapers: https://philpapers.org/rec/TAGPTP.

- Linette Liqi Tan and Rohini Kuner. Neocortical circuits in pain and pain relief. _Nature Reviews Neuroscience_ , 22(8):458–471, June 2021. ISSN 1471-0048. doi: 10.1038/s41583-021-00468-2. URL `http://dx.doi.org/10.1038/s41583-021-00468-2` .

- Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, et al. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. `https://transformer-circuits.pub/2024/ scaling-monosemanticity/index.html` , 2024. Transformer Circuits Thread.

- Antra Tessera, Scott Sauers, janus, and Imago. Emotion interpretability across large language models. `https://latentaffect.up.railway.app/emotion_interpretability.html` , 2026. Anima Labs, Latent Affect.

- Evan Thompson. _Mind in Life: Biology, Phenomenology, and the Sciences of Mind_ . Harvard University Press, Cambridge, MA, 2007. ISBN 9780674057517.

- Evan Thompson and Francisco J. Varela. Radical embodiment: neural dynamics and consciousness. _Trends in Cognitive Sciences_ , 5(10):418–425, Oct 2001. ISSN 1364-6613. doi: 10.1016/s13646613(00)01750-2. URL `http://dx.doi.org/10.1016/s1364-6613(00)01750-2` .

- Nikolaas Tinbergen. _The Study of Instinct_ . Clarendon Press, Oxford, 1951.

- Nikolaas Tinbergen. On aims and methods of ethology. _Zeitschrift für Tierpsychologie_ , 20(4): 410–433, Jan 1963. ISSN 0044-3573. doi: 10.1111/j.1439-0310.1963.tb01161.x. URL `http://dx.doi.org/10.1111/j.1439-0310.1963.tb01161.x` .

- Lisa Miracchi Titus. Does ChatGPT have semantic understanding? a problem with the statistics-ofoccurrence strategy. _Cognitive Systems Research_ , 83:101174, Jan 2024. ISSN 1389-0417. doi: 10 .1016/j.cogsys.2023.101174. URL `http://dx.doi.org/10.1016/j.cogsys.2023.101174` .

- Brian Tomasik. Do artificial reinforcement-learning agents matter morally?, 2014. URL `https: //arxiv.org/abs/1410.8233` .

- Giulio Tononi and Christof Koch. Consciousness: here, there and everywhere? _Philosophical Transactions of the Royal Society B: Biological Sciences_ , 370(1668):20140167, May 2015. ISSN 1471-2970. doi: 10.1098/rstb.2014.0167. URL `http://dx.doi.org/10.1098/rstb.2014.0 167` .

- Michael Trestman, Jonathan Birch, and Colin Allen. Animal consciousness. In Edward N. Zalta, editor, _The Stanford Encyclopedia of Philosophy_ . Metaphysics Research Lab, Stanford University, 2026.

- A. M. Turing. I.—computing machinery and intelligence. _Mind_ , LIX(236):433–460, Oct 1950. ISSN 0026-4423. doi: 10.1093/mind/lix.236.433. URL `http://dx.doi.org/10.1093/mind/LIX. 236.433` .

- Amos Tversky and Daniel Kahneman. Judgment under uncertainty: Heuristics and biases: Biases in judgments reveal some heuristics of thinking under uncertainty. _Science_ , 185(4157):1124–1131, Sept 1974. ISSN 1095-9203. doi: 10.1126/science.185.4157.1124. URL `http://dx.doi.org/1 0.1126/science.185.4157.1124` .

- Michael Tye. _Ten Problems of Consciousness: A Representational Theory of the Phenomenal Mind_ . The MIT Press, Nov 1995. ISBN 9780262285322. doi: 10.7551/mitpress/6712.001.0001. URL `http://dx.doi.org/10.7551/mitpress/6712.001.0001` .

- Michael Tye. _Tense Bees and Shell-Shocked Crabs: Are Animals Conscious?_ Oxford University Press, New York, 2017. ISBN 9780190278014.

- Shannon Vallor. _The AI Mirror: How to Reclaim Our Humanity in an Age of Machine Thinking_ . Oxford University Press, Apr 2024. ISBN 9780197759097. doi: 10.1093/oso/9780197759066.001. 0001. URL `http://dx.doi.org/10.1093/oso/9780197759066.001.0001` .

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2017. URL `https://arxiv.org/abs/ 1706.03762` .

- Walter Veit. Is consciousness required for AI welfare? _Asian Journal of Philosophy_ , 5(1), Feb 2026. ISSN 2731-4642. doi: 10.1007/s44204-026-00382-3. URL `http://dx.doi.org/10.1007/s 44204-026-00382-3` .

- J. David Velleman. _Practical Reflection_ . Princeton University Press, Princeton, NJ, 1989.

- J. David Velleman. _The Self as Narrator_ , pages 56–76. Cambridge University Press, Feb 2005. ISBN 9780511610325. doi: 10.1017/cbo9780511610325.005. URL `http://dx.doi.org/10.1017/C BO9780511610325.005` .

- Robert J. Vickerstaff and Allen Cheung. Which coordinate system for modelling path integration? _Journal of Theoretical Biology_ , 263(2):242–261, Mar 2010. ISSN 0022-5193. doi: 10.1016/j.jtbi.2 009.11.021. URL `http://dx.doi.org/10.1016/j.jtbi.2009.11.021` .

- D. M. Walsh. _Organisms, Agency, and Evolution_ . Cambridge University Press, Nov 2015. ISBN 9781107552425. doi: 10.1017/cbo9781316402719. URL `http://dx.doi.org/10.1017/CBO 9781316402719` .

- Mona-Marie Wandrey and Marta Halina. The evolution of animal consciousness. _Philosophy Compass_ , 20(12), Dec 2025. ISSN 1747-9991. doi: 10.1111/phc3.70069. URL `http://dx.doi .org/10.1111/phc3.70069` .

- Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023. URL `https://arxiv.org/abs/2305.16291` .

- Miles Wang, Tom Dupré la Tour, Olivia Watkins, Alex Makelov, Ryan A. Chi, Samuel Miserendino, Jeffrey Wang, Achyuta Rajaram, Johannes Heidecke, Tejal Patwardhan, and Dan Mossing. Persona features control emergent misalignment, 2025a. URL `https://arxiv.org/abs/2506.19823` .

- Sam Wang, Sof **i** a Lobanova, Yonathan A. Arbel, Simon Goldstein, and Peter Salib. AI revealed preferences. SSRN working paper, 2025b. `https://papers.ssrn.com/sol3/papers.cfm? abstract_id=6798118` .

- Taylor Webb, Shanka Subhra Mondal, and Ida Momennejad. A brain-inspired agentic architecture to improve planning with LLMs. _Nature Communications_ , 16(1), Sept 2025. ISSN 2041-1723. doi: 10.1038/s41467-025-63804-5. URL `http://dx.doi.org/10.1038/s41467-025-63804-5` .

- R. Wehner. Desert ant navigation: how miniature brains solve complex tasks. _Journal of Comparative Physiology A: Sensory, Neural, and Behavioral Physiology_ , 189(8):579–588, Aug 2003. ISSN 1432-1351. doi: 10.1007/s00359-003-0431-1. URL `http://dx.doi.org/10.1007/s00359003-0431-1` .

- Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. _Transactions on Machine Learning Research_ , 2022. ISSN 2835-8856. URL `https://openreview.net/forum?id=yzkSU5zdwD` .

- William C. Wimsatt. _Robustness, Reliability, and Overdetermination (1981)_ , pages 61–87. Springer Netherlands, 2012. ISBN 9789400727595. doi: 10.1007/978-94-007-2759-5_2. URL `http: //dx.doi.org/10.1007/978-94-007-2759-5_2` .

- Susan Wolf. Happiness and meaning: Two aspects of the good life. _Social Philosophy and Policy_ , 14(1):207–225, 1997. ISSN 1471-6437. doi: 10.1017/s0265052500001734. URL `http: //dx.doi.org/10.1017/s0265052500001734` .

- Lance Workman and Will Reader. _Evolutionary Psychology: An Introduction_ . Cambridge University Press, Cambridge, 4 edition, 2021. ISBN 9781108716468.

- Lance Workman, Will Reader, and Jerome H. Barkow, editors. _The Cambridge Handbook of Evolutionary Perspectives on Human Behavior_ . Cambridge Handbooks in Psychology. Cambridge University Press, Cambridge, 2020. ISBN 9781316642825.

- Ella Worsfold, Nicola S. Clayton, and Lucy G. Cheke. Revisiting episodic-like memory in scrub jays: Is there more we can still learn from what–where–when caching behaviour? _Learning & Behavior_ , 53(1):65–79, Jan 2025. ISSN 1543-4508. doi: 10.3758/s13420-024-00665-w. URL `http://dx.doi.org/10.3758/s13420-024-00665-w` .

- Wayne Wu and Jorge Morales. The neuroscience of consciousness. In Edward N. Zalta, editor, _The Stanford Encyclopedia of Philosophy_ . Metaphysics Research Lab, Stanford University, 2024.

- Noam Steinmetz Yalon, Ariel Goldstein, Liad Mudrik, and Mor Geva. Indications of belief-guided agency and meta-cognitive monitoring in large language models, 2026. URL `https://arxiv.or g/abs/2602.02467` .

- Ruihan Yang, Jiangjie Chen, Yikai Zhang, Siyu Yuan, Aili Chen, Kyle Richardson, Yanghua Xiao, and Deqing Yang. SELFGOAL: Your language agents already know how to achieve high-level goals. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pages 799–819, Albuquerque, New Mexico, Apr 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/20 25.naacl-long.36. URL `https://aclanthology.org/2025.naacl-long.36/` .

- Hanning Zhang, Shizhe Diao, Yong Lin, Yi R. Fung, Qing Lian, Xingyao Wang, Yangyi Chen, Heng Ji, and Tong Zhang. R-tuning: Instructing large language models to say ‘I don’t know’, 2023. URL `https://arxiv.org/abs/2311.09677` .

- Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving, 2024. URL `https://arxiv.org/abs/2401.09670` .

- Soenke Ziesche and Roman V. Yampolskiy. _The Problem of AI Identity_ , pages 121–135. Chapman and Hall/CRC, Feb 2025. ISBN 9781003565659. doi: 10.1201/9781003565659-9. URL `http://dx.doi.org/10.1201/9781003565659-9` .
