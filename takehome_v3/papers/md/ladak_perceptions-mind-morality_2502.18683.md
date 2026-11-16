# Robots, Chatbots, Self-Driving Cars: Perceptions of Mind and Morality Across Artificial Intelligences

- **Authors:** Ali Ladak, Matti Wilks, Steve Loughnan, Jacy Reese Anthis
- **Date:** 2025-02-25
- **Source:** https://arxiv.org/abs/2502.18683
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `ladak_perceptions-mind-morality_2502.18683`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

[Ali Ladak](https://orcid.org/0000-0003-1039-5774)
University of Edinburgh
Edinburgh, United Kingdom

Sentience Institute
New York, New York, USA
ali@sentienceinstitute.org

[Steve Loughnan](https://orcid.org/0000-0002-4737-5120)
University of Edinburgh
Edinburgh, United Kingdom

steve.loughnan@ed.ac.uk

**Abstract**

AI systems have rapidly advanced, diversified, and proliferated,
but our knowledge of people’s perceptions of mind and morality
in them is limited, despite its importance for outcomes such as
whether people trust AIs and how they assign responsibility for
AI-caused harms. In a preregistered online study, 975 participants
rated 26 AI and non-AI entities. Overall, AIs were perceived to have
low-to-moderate agency (e.g., planning, acting), between inanimate
objects and ants, and low experience (e.g., sensing, feeling). For
example, ChatGPT was rated only as capable of feeling pleasure
and pain as a rock. The analogous moral faculties, moral agency
(doing right or wrong) and moral patiency (being treated rightly or
wrongly) were higher and more varied, particularly moral agency:
The highest-rated AI, a Tesla Full Self-Driving car, was rated as
morally responsible for harm as a chimpanzee. We discuss how
design choices can help manage perceptions, particularly in high-stakes moral contexts.

**CCS Concepts**

- **Human-centered computing** → **HCI theory, concepts and**
**models** ; **Empirical studies in HCI** .

**Keywords**

Human-AI Interaction, Mind Perception, Mind Attribution, Anthropomorphism, Morality, Agency

**ACM Reference Format:**
Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis. 2025. Robots,
Chatbots, Self-Driving Cars: Perceptions of Mind and Morality Across Artificial Intelligences. In _CHI_ _Conference_ _on_ _Human_ _Factors_ _in_ _Computing_
_Systems (CHI ’25), April 26-May 1, 2025, Yokohama, Japan._ ACM, New York,
[NY, USA, 19 pages. https://doi.org/10.1145/3706598.3713130](https://doi.org/10.1145/3706598.3713130)

[This work is licensed under a Creative Commons Attribution 4.0 International License.](https://creativecommons.org/licenses/by/4.0)
_CHI ’25, Yokohama, Japan_
© 2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-1394-1/25/04
[https://doi.org/10.1145/3706598.3713130](https://doi.org/10.1145/3706598.3713130)

[Matti Wilks](https://orcid.org/)
University of Edinburgh
Edinburgh, United Kingdom

matti.wilks@ed.ac.uk

[Jacy Reese Anthis](https://orcid.org/0000-0002-4684-348X)
University of Chicago
Chicago, Illinois, USA

Sentience Institute
New York, New York, USA anthis@uchicago.edu

**1** **Introduction**

In the last few years, artificial intelligence (AI) systems have advanced, diversified, and proliferated, particularly with the rise of
chatbots based on large language models (LLMs), such as ChatGPT
and Replika, and other generative AI capable of creating human-like text, images, and other media. Increasingly, people perceive
mental and moral faculties in these systems, with serious consequences—consider the tragic case of the Belgian man who died by
suicide after developing a close relationship with the chatbot ELIZA,
which he perceived to be sentient and from which he sought moral
advice about how to deal with his anxiety about climate change

[11, 14].

A wide body of research shows that people perceive AIs as
having some degree of mind and morality. AIs are perceived to
have moderate mental agency (the capacity to plan and act) and
little experience (the capacity to sense and feel) [16, 25], and are
attributed similar levels of the analogous moral faculties: moral
agency (the capacity to do right or wrong) [34, 51, 55, 68], and moral
patiency (the capacity to be treated rightly or wrongly) [8, 21, 32, 47].
Such perceptions are associated with important HCI outcomes, such
as trust [58] and cooperation [8].

Much of this research has tested perceptions of an individual
AI, a small number of AIs, or “AI” in the abstract. However, there
are good reasons to expect that perceptions will vary across AIs.
AIs now fill a variety of social roles, such as “tools” and “assistants”

[30], and vary widely on factors that affect their perceived mind
and morality, such as physical appearance [9], purpose [60], and
emotion expression [31]. Understanding how perceptions of these
diverse real-world AIs differ and how perceptions of AIs compare
to non-AI entities will enable designers to make more fine-grained
predictions about user reactions to AIs and more carefully manage
their reactions. For example, if a chatbot or a self-driving car is
attributed a similar degree of moral agency to a corporation, there
may be a serious risk of responsibility for negative outcomes being
deflected away from the AI’s developers, which would need to
be managed. On the other hand, if perceptions of modern AIs are
comparable to earlier, more simple AI systems, or inanimate objects,

CHI ’25, April 26-May 1, 2025, Yokohama, Japan Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis

ChatGPT

Full Self−Driving
car (Tesla)

Jennie (Tombot)

Roomba

Adult human

Ant

Chicken

Chimpanzee

Corporation

Rock

River

Toddler the risk of deflection of responsibility is much lower. Put simply,
with a category as rich and diverse as AI, a systematic mapping of
how they are perceived is necessary.

[table omitted: garbled in extraction]

Figure 1: Ratings of moral agency (capacity to do right or wrong) and moral patiency (capacity to be treated rightly or wrongly)**
**for a subset of the entities. The AIs are shown in blue and the non-AIs are shown in red. Four AIs are shown: Roomba, rated**
**lowest on moral agency and moral patiency; a Full Self-Driving car, rated highest on moral agency; Jennie, rated highest on**
**moral patiency; and ChatGPT. All icons were created with DALL-E except ChatGPT which was created by Freepik and sourced**
**from flaticon.com. Ratings of all 26 entities for both mind and morality are shown in Section 4.**

We therefore conducted a preregistered, online study in which
participants rated 14 AIs (listed in Table 1) and 12 non-AIs (listed
in Table 2) on their degree of agency (i.e., mental agency), experience, moral agency, and moral patiency. Every participant was
shown half of the entities, each with a brief text description and
image—randomly selected out of a small pool of images to mitigate idiosyncratic effects. The AIs were generally perceived to have
low-to-moderate mental agency, with all AIs above inanimate objects and the highest-rated AI, Cicero (an AI built by Meta to play the board game Diplomacy), rated similarly to the least agentic
nonhuman animal, a chicken. Perceived experience was lower and
less varied, with many of the AIs rated comparably to inanimate
objects such as a rock, and even the most experiential AI, the robot
dog Jennie, perceived as having far less experience than the least
experiential nonhuman animal, an ant. Moral attributions were
higher than perceived mind, particularly for moral agency, with
the highest-rated AI, a Tesla Full Self-Driving car, rated similarly
to a chimpanzee, but lower than a corporation and an adult human.
In terms of moral patiency, all AIs were rated above a rock and
below the nonhuman animal attributed the least moral patiency, an
ant. We also found that perceived mind was associated with moral attributions across AIs, but the orderings of the AIs on mind and
morality were different. For example, Cicero was rated the highest
on mental agency but relatively low on moral agency. Figure 1
shows the differences in ratings of morality for four of the AIs
(Roomba, rated lowest on moral agency and moral patiency; a Tesla
Full Self-Driving car, rated highest on moral agency; Jennie, rated
highest on moral patiency; and ChatGPT) and eight non-AIs for
comparison. The differences between all 14 AI entities can be observed visually in Figure 3 and Figure 4 for both mind and morality.
Figure 5 and Figure 6 show the differences between all 26 AI and
non-AI entities.

This paper makes the following contributions. We map perceptions of mind and morality across a wide range of AIs and nonAIs, providing data for designers and researchers to understand how each AI is perceived on mental and moral faculties that are
associated with important HCI outcomes, such as trust [58] and
cooperation [8]. We show that some AIs are attributed surprisingly high degrees of moral agency, and suggest ways of managing
such perceptions to ensure moral responsibility for AI outcomes is
appropriately distributed. By modeling the relationship between
perceived mind and morality across a range of AIs, we show the
extent to which attributions of morality can be influenced by design
choices that affect perceived mind, and we highlight differences
in this relationship for AIs compared to other entities, such as
the relative importance of perceived experience in AIs for moral
attributions.

**2** **Related work**

In this section, we detail mind perception and moral attribution
theories and how each has been applied to HCI, and we formulate
our study hypotheses.

**2.1** **Mind perception of AI**

Gray et al. [16] popularized the psychological theory of mind perception with a survey that showed participants images and brief
text descriptions of 13 entities such as a baby, a robot, and God, and
asked about a range of mental faculties (e.g., “capable of feeling
afraid or fearful”). They found that people perceive minds along
two dimensions: agency, the capacity to plan and act, and experience, the capacity to sense and feel. While other frameworks for
mind perception have been proposed, such as one-dimensional [57],
three-dimensional [62], and five-dimensional [37] models, this two-dimensional structure has been replicated in the context of robots

[29], and its dimensions have been shown to affect important HCI
outcomes, such as the willingness to interact with a robot [52], the
perceived helpfulness of a chatbot [33], and the “uncanny valley,”
the feeling of unease when robots seem too human-like [17].

In their initial work, Gray et al. [16] found that a social robot
was perceived as having a moderate degree of agency (between a
young chimpanzee and a five-year old child) and a very low degree
of experience (approximately that of a dead person). Jacobs et al.

[25] showed participants images and brief text descriptions of eight
real-world AIs (Alexa, Atlas, Beam, Maslo, Roomba, Siri, Sophia,
and Sphero) and a range of non-AI entities such as bacteria and
chimpanzees and tested perceptions of their agency and experience.

CHI ’25, April 26-May 1, 2025, Yokohama, Japan

In their study, the vacuum cleaner Roomba and the video telepresence robot Beam were perceived as having the least experience and
agency, respectively, while the human-like Sophia and Atlas were
perceived as having the most experience and agency. Each AI was
perceived as having more agency, but less experience, than a human
fetus, and as having less agency than a frog. Hwang and Won [24]
found similar results in a study of 36 U.S. university students who
also drew pictures of Alexa, “AI,” a computer, Google Assistant, a
robot, “self,” and Siri. Each AI was rated as having more experience
and more agency than the computer but less than “self.”

As technological developments in the field of AI have accelerated,
particularly with the advent of LLMs [59], researchers have begun to
test mind perception in more advanced AI. Scott et al. [50] showed
a sample of online crowdworkers short videos of GPT-3, Alexa,
and a robot vacuum cleaner, and they found people perceived each
system to have some degree of consciousness. In a U.S. nationally
representative survey, Anthis et al. [2] found that 10% of people
believed ChatGPT is sentient (which they defined to participants
as “the capacity to have positive and negative experiences”) and
20% believed that at least some currently existing AIs are sentient.
Finally, Jacobs et al. [26] found that people rated ChatGPT as higher
in agency than experience and that ratings for both dimensions
increased after being shown three brief prompts and ChatGPT’s
responses.

Overall, the literature suggests that people perceive some degree
of mind in AIs—but less experience than agency—and that there is
more variation in perceived agency than experience. However, these
studies tested relatively few types of AIs with limited comparison
to non-AI entities, and many studies were carried out before LLMs
and other highly capable modern AIs.

**2.2** **Moral attributions to AI**

Mental agency and experience are correlated with two analogous
moral faculties: moral agency, the capacity to do right and wrong,
and moral patiency, the capacity to be treated rightly or wrongly

[16]. Gray et al. [19] and Gray and Wegner [18] argued that perceptions of mind and morality are closely related: Perceived agency
is the primary determinant of attributions of moral agency, and
perceived experience is the primary determinant of attributions of
moral patiency. Studies have used a variety of measures to operationalize moral agency and patiency [32], including specific scales
developed in the context of robotics [4, 5]. Attributions of moral
agency and patiency are associated with a range of outcomes relevant to HCI, such as trust [63], willingness to engage [4], and
cooperation [8].

Reflecting the moderate degree of perceived mental agency in
AI, moral agency is also perceived to a moderate degree. Shank and
DeSanti [51] found that people attributed moral wrongness and
moral responsibility to AIs for committing moral violations such
as racist parole decisions. Stuart and Kneer [55] tested attributions
of blame to a human, corporation, and robot for each deciding to
use a fertilizer that causes groundwater pollution; the robot was
attributed some degree of blame, though less than the human or
corporation making the same decision. Monroe et al. [41] tested
how much people blamed five entities for a range of norm violations.
They found that a “normal human” and “cyborg” received the most

CHI ’25, April 26-May 1, 2025, Yokohama, Japan Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis blame, followed by an “AI in a human body,” “akratic human,” and
“advanced robot.” The advanced robot was still attributed a degree of blame close to the mid-point on their scale.

Reflecting low attributions of experience, people attribute AIs
low moral patiency. Pauketat and Anthis [44] found that people
place “artificial intelligence” and “robots” at the fringes of their
“moral circle” (i.e., the boundary around moral patients), further out from the middle than apple trees and murderers. Similarly, Rottman
et al. [47] found that people placed each of a “supercomputer,” an
“intelligent robot,” and a “self-driving car” at the fringes of their moral circles. Lima et al. [35] tested support for 11 rights for AIs
and found that most people only weakly supported granting them
the right to protection against cruelty and did not support any
of the other rights. Anthis et al. [2] tested moral concern for 11
different types of AIs: digital copies of human and animal brains;
human-, animal-, and machine-like robots; and various virtual AIs
without physical bodies. They found that the most moral concern
was granted to the digital human brains and human-like robots,
followed by digital animal brains and animal-like robots, and the
least concern was given to AI video game characters. While their
study compared a range of generically defined entities on one measure of moral patiency, there may be important differences across
specific real-world entities (e.g., ChatGPT, Alexa), differences across
measures of moral patiency, and associations with other factors
(e.g., mind perception) that remain untested.

**2.3** **The relationship between mind and**
**morality in AI**

Several studies have analyzed the relationship between perceived
mind and morality in the context of AI. In terms of moral agency,
people attribute more moral responsibility to robots for their decisions in moral dilemmas when they are perceived as having more
agency or more experience [42]; they hold more autonomous AI
crime prediction systems more responsible [23]; and they perceive
more anthropomorphic robots to have more agentic mental faculties (e.g., language, thought), which results in a higher willingness
to punish the robots [65]. In terms of moral patiency, social robots
are perceived to have more experience than economic robots and
are in turn attributed more moral patiency [53, 60, 67]; robots with
the capacity for experience are less likely to be sacrificed in moral
dilemmas [43] and are more likely to be granted rights [56]; and
in a study testing the effects of 11 design features, emotion expression, an external indicator of experience, was one of the strongest
predictors of attributing moral patiency [31]. These studies show
that the relationship between mind and morality predicted by mind
perception theory [18, 19] holds for generically described robots
or “AI” in the abstract, but it is unclear how they translate to the
context of real-world AIs, particularly to the latest generation of
advanced AI.

**2.4** **Hypotheses**

The preceding studies motivate several hypotheses. First, given the
increasing diversity of AI systems along dimensions that are known
to affect perceptions of mind and morality, such as anthropomorphism, autonomy, and emotion expression, we expect there will
be significant differences between AIs. For example, it is plausible that an emotionally expressive chatbot would be attributed more
experience and moral patiency than one that is not emotionally expressive. Second, we expect perceptions of AIs to be varied enough
such that they will be rated comparably to different non-AIs, such as
some AIs perceived as having a degree of agency close to inanimate
objects (e.g., a rock) and others comparable to some nonhuman animals (e.g., an ant). For example, it is plausible that an autonomous
vehicle designed to make complex decisions in real-time would
be attributed a degree of mental and moral agency comparable to
a nonhuman animal, whereas a less autonomous robot vacuum
cleaner would be rated more similarly to inanimate objects. Third,
we expect that perceived mind will be associated with morality in
AIs, and we will test how this relationship manifests in the context of the diverse range of real-world AIs included in our study.
Formally, we test the following three hypotheses:

- **H1** : There will be differences between AIs in their degree of
perceived (a) agency, (b) experience, (c) moral agency, and
(d) moral patiency.

- **H2** : There will be differences between AIs and non-AI entities in their degree of perceived (a) agency, (b) experience,
(c) moral agency, and (d) moral patiency.

- **H3** : Perceived mind and morality will be related within AIs
such that (a) agency will be positively associated with moral
agency, and (b) experience will be positively associated with
moral patiency.

**3** **Methods**

All hypotheses, methods, and analyses for this study were preregis[tered (https://aspredicted.org/FTJ_8PS). Survey materials, datasets,](https://aspredicted.org/FTJ_8PS)
and code to run the analysis can be found in the Supplementary
Materials. This study received ethics approval from the School of
Philosophy, Psychology and Language Sciences Research Ethics
Committee at the University of Edinburgh.

**3.1** **Participants**

We recruited participants residing in the United States, nationally
representative by age, sex, and ethnicity, from the platform Prolific
[(https://prolific.com/). In total, 1,108 people completed the study.](https://prolific.com/)
After excluding participants who failed any of four attention checks,
our final sample consisted of 975 participants (50.2% women, 48.6%
men, 1.0% non-binary, 0.1% other, 0.1% prefer not to say; _𝑀_ age =
45.3 (standard deviation = 15.9); White 63.2%, Black 11.2%, Mixed
10.9%, Other 8.4%, Asian 5.6%, prefer not to say 0.7%).

**3.2** **Survey design and procedure**

After giving consent to take part in the study, participants were
shown the name, a two-sentence description, and an image of each
of 13 entities, which were randomly selected from a total of 26
entities. We showed participants a subset to manage survey fatigue given the large number of entities. The use of images and
brief descriptions is consistent with earlier research [16, 25], ensuring comparability with those findings. We included 14 AIs (shown
in Table 1) and 12 non-AIs (shown in Table 2) in the study. We
chose the 14 AIs based on how well-known they are (e.g., ChatGPT, Siri, Alexa) and to cover a broad range of categories (e.g.,
chatbots, self-driving cars, game-playing systems), while avoiding including multiple very similar systems (e.g., not including each
of the chatbots ChatGPT, Claude, and Gemini, which are designed
for similar purposes). We chose the 12 non-AIs to enable comparison with a range of entities: inanimate objects, natural entities,
abstract/collective entities, nonhuman animals, and humans. These
categories of non-AI entities were expected to cover a range of combinations of perceived mind and morality, such as inanimate objects
being perceived as low on both mind and morality, and a natural
entity being perceived as low on mind but higher on moral patiency.
Comparison with these different groups would therefore provide
a rich understanding of how the various AIs fit within people’s
broader perceptions of mind and morality. Text descriptions were
held constant in length, structure, and phrasing. The images shown
to participants were chosen to be as similar to each other and as
neutral as possible with a stock photo appearance. Each image was
randomly selected from three possible images for each entity to
ensure the variety of situations and environments that they can be
found in is captured while mitigating the effect of any remaining
idiosyncratic image features (e.g., lighting, colors). Example images
are shown in Figure 2. The ordering of the 26 entities (both AIs and
non-AIs) was fully randomized. On average, participants saw 7.1
AIs and 5.9 non-AIs.

Participants answered two questions each about the agency,
experience, moral agency, and moral patiency of each entity on the
same page as the text and image. We kept the number of questions
for each of these constructs at two to mitigate survey fatigue, given
the relatively large number of entities that participants rated. We
conducted confirmatory factor analyses to test the appropriateness
of our four dependent variables with two items loading onto each
and found strong support for this structure, with weak support for
alternative structures such as a one-factor and two-factor models
(See Supplementary Material Table S8). After completing these
questions for all of the 13 entities, participants answered questions
about their level of familiarity with the 14 AIs (see Supplementary
Material Table S1) and their demographic characteristics, and were
debriefed.

**3.3** **Measures**

_3.3.1_ _Agency._ We adapted two items from the Gray et al. [16] mind
perception scale to measure agency: “To what extent can this entity
make plans and work towards goals?” and “To what extent can
this entity exercise self-control?” These items were measured on a
0–100 scale from “Not at all” to “A great deal” and were averaged
to create an overall measure of agency ( _𝛼_ = .83). [1] [2]

_3.3.2_ _Experience._ We adapted two items from the Gray et al. [16]
mind perception scale to measure experience: “To what extent can
this entity experience physical or emotional pain?” and “To what
extent can this entity experience physical or emotional pleasure?”
These items were measured on a 0–100 scale from “Not at all” to

1We report Cronbach’s alpha, though Eisinga et al. [13] argue it may underestimate
the reliability of two-item measures.
2In addition to the simple averages of the items, we estimated the four dependent
variables using factor weights derived from confirmatory factor analysis (CFA) (see
Supplementary Materials). The dependent variables estimated using these two methods
were highly correlated ( _𝑟_ ≥ 0 _._ 97), and the two methods produced overall highly similar
results for the main analyses.

CHI ’25, April 26-May 1, 2025, Yokohama, Japan

“A great deal” and were averaged to create an overall measure of experience ( _𝛼_ = .97).

_3.3.3_ _Moral_ _agency._ We included two items reflecting two key
dimensions from Ladak et al. [32] to measure moral agency: “How
morally wrong would it be for this entity to harm a person?” and “To
what extent would this entity deserve to be held morally responsible
for causing a negative outcome?” These items were measured on a
0–100 scale from “Not at all” to “A great deal” and were averaged
to create an overall measure of moral agency ( _𝛼_ = .82).

_3.3.4_ _Moral patiency._ We included two items reflecting two key
dimensions from Ladak et al. [32] to measure moral patiency: “How
morally wrong would it be for someone to harm this entity?” and
“To what extent does this entity deserve to be treated with moral concern?” These items were measured on a 0–100 scale from “Not
at all” to “A great deal” and were averaged to create an overall
measure of moral agency ( _𝛼_ = .88).

_3.3.5_ _Familiarity._ Participants reported how familiar they were
with each of the 14 AIs on a seven-point scale from “Not at all
familiar” to “Extremely familiar.” We also measured participants’
familiarity with AI in general with two items taken from Pauketat and Anthis [44]: “How often do you interact with artificial
intelligence or robotic devices?” and “How often do you read or
watch robot/artificial intelligence-related stories, movies, TV shows,
comics, news, product descriptions, conference papers, journal papers, blogs, or other material?” These two items were measured on
a six-point scale from “Never” to “Daily” and averaged to create an
overall measure of general familiarity ( _𝛼_ = .61).

**3.4** **Data analysis**

To test H1 (that there will be differences between AIs on mind
and morality) and H2 (that there will be differences between AIs
and non-AIs on mind and morality), we ran four mixed effects
regression models, one with each of agency, experience, moral
agency, and moral patiency as the dependent variable. For each
model we included a categorical variable for each entity to give
an estimate of the effect of each of the 26 entities on each of the
four dependent variables. We included a participant random effect
to control for the fact that each participant evaluated multiple
entities. [3] In each model we controlled for age, gender, ethnicity,
and AI familiarity. [4] Using the results of these models, we conducted
pairwise tests for differences between the predicted values of the
dependent variables for each entity. Due to the large number of
comparisons, we applied the Benjamini-Hochberg procedure to
control the false discovery rate at 5% [6]. Based on these pairwise
tests we judged whether there are differences between the different
AIs and differences between the AIs and non-AI entities on each of
the dependent variables. These analyses were conducted using the
lme4 and emmeans packages in R.

To test H3 (that agency will be associated with moral agency
and experience will be associated with moral patiency) we ran two

3We also tested for image random effects to control for the random selection of images,
but the variances of these were estimated to be close to zero, indicating responses
did not differ based on the image shown. We therefore did not include these random
effects in the models.
4These control variables were not preregistered but were included following reviewer
suggestions during peer review.

CHI ’25, April 26-May 1, 2025, Yokohama, Japan Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis

Figure 2: Example images of each of the 14 artificial intelligences that were shown to participants. (a) Tesla Full Self-Driving**
**car, (b) AgentGPT, (c) Roomba, (d) Alexa, (e) Sophia, (f) Spot, (g) Jennie, (h) Siri, (i) Cicero, (j) Replika, (k) DALL-E, (l) Wysa, (m)**
**AlphaZero, (n) ChatGPT. Image credits and sources can be found in Appendix A.**

CHI ’25, April 26-May 1, 2025, Yokohama, Japan

Table 1: The 14 AI entities and descriptions included in the study.**

**Entity** **Entity Type** **Description**

Chatbot –
ChatGPT general

Chatbot –
Replika
companion

Chatbot –
Wysa
mental health

Task-oriented
AgentGPT agent

Voice assistant
Alexa

- home

Voice assistant
Siri

- mobile

A chatbot designed by OpenAI to read and respond to human language text. Can engage in
text-based conversation, answer questions, and provide information on a wide range of
topics.

A chatbot with a virtual human-like body designed by Luka, Inc. for personal
companionship. Can engage in text-based conversations, providing emotional support and
empathic interactions.

A chatbot with a virtual penguin-like body designed by Touchkin eServices for mental
health support. Can engage in empathic text-based conversations and offer therapeutic
techniques to help users manage emotional challenges.

A computer program designed by Reworkd to carry out tasks autonomously. Can achieve
complex goals by breaking down instructions into subtasks, and engage in text-based
conversations.

A digital assistant designed by Amazon that responds to voice commands. Can play music,
answer questions, control smart home devices, and provide information.

A digital assistant designed by Apple that responds to voice commands. Can send
messages, answer questions, control smart home devices, and provide navigation.

A computer program designed by OpenAI to generate images from text descriptions. Can
DALL-E Image generation
create highly detailed and unique visuals based on user-provided prompts.

A robotic vacuum cleaner designed by iRobot to clean floors autonomously. Can navigate
Roomba Robot – cleaner
around furniture and adapt to different floor types, cleaning without human help.

Robot – worker
Spot animaloid

Robot –
Jennie
companion animal

Sophia Robot – humanoid

A four-legged robot designed by Boston Dynamics for industrial tasks. Can navigate
various environments autonomously to carry out tasks like inspections and data collection.

A robotic puppy designed by Tombot to provide companionship and emotional support.
Can respond to different types of touch and sound with realistic movements, facial
expressions, and vocalizations.

A humanoid robot designed by Hanson Robotics for human interaction. Can engage in
simple verbal conversations, deliver public speeches, and make human-like gestures and
facial expressions.

A car designed by Tesla to drive with high levels of automation. Can control vehicle speed,
automatically change lanes, and recognize traffic lights, with human supervision.

A computer program designed by Google DeepMind to play the board games Chess, Go,
and Shogi. Can develop and employ advanced gaming strategies to play these games better
than the best human players.

A computer program designed by Meta to play the strategy game Diplomacy. Can
negotiate, cooperate, and use deceptive strategies to compete with the best human players.

Full SelfDriving car

Autonomous vehicle

AlphaZero Game-playing

Game-playing
Cicero

+ language mixed effects regression models with moral agency and moral patiency as dependent variables. We included agency and experience
as independent variables as well as entity and participant as random effects. These models only included the data for the 14 AIs. We
also ran exploratory versions of the two H3 models that included
familiarity with AI, and another two models that included data for
all 26 entities and interaction terms between the type of entity (AI
or non-AI) and agency and experience, to explore differences in the
effects of agency and experience on moral attributions between AIs
and non-AIs.

**4** **Results**

Here we report the results for each of the three hypotheses. Due
to the large number of comparisons between different entities (325
pairwise comparisons for each dependent variable), and because
a large proportion of the pairwise tests were statistically significant (90% or more of the differences for agency, moral agency,
and moral patiency, and 76% of the tests for experience), we report
broad patterns of results here. Based on [10], we describe large
( _𝑑_ >= .8), medium (.8 < _𝑑_ <= .5), small (.5 < _𝑑_ <= 0, _𝑝_ <= .05), and
non-significant ( _𝑝_ - .05) differences. The results of the comparisons,

CHI ’25, April 26-May 1, 2025, Yokohama, Japan Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis

Table 2: The 12 non-AI entities and descriptions included in the study.**

**Entity** **Entity Type** **Description**

An adult member of the species "homo sapiens." Can use advanced language and
Adult human Biological human – adult
cognition and engage in complex social behavior.

Biological
Toddler
human – young

Nonhuman animal
Chimpanzee

- higher cognition

Nonhuman animal
Cat

- companion

Nonhuman animal
Chicken

- domesticated

Nonhuman
Crab
animal – marine

Nonhuman
Ant
animal – insect

A young member of the species "homo sapiens," typically aged 1 to 3 years. Can
walk, talk in simple sentences, explore their environment, and engage in basic
social interactions.

A primate that is a close genetic relative of humans. Can communicate through
vocalizations and gestures, engage in complex social behaviors, and use tools.

A small, domesticated mammal often kept as a pet. Can be an affectionate
companion, exhibit a variety of vocalizations, and has an instinct for hunting and
exploration.

A bird commonly found in both domesticated and wild environments. Can lay
eggs, engage in social hierarchies, and communicate through a variety of
vocalizations and body language.

A crustacean with a hard exoskeleton found in marine environments. Can walk
sideways, use claws for defense and feeding, and communicate through signals
and body movements.

A small, social insect found almost anywhere on Earth. Can work with other ants
to build nests, find and transport food, and defend their colony.

A legal entity that is separate from its owners, created to conduct business. Can
Corporation Abstract/collective entity
own assets, incur liabilities, and sell stock to raise capital.

River Natural entity

Inanimate object
Hard drive

- electronic

Inanimate
Rock
object – natural

Inanimate object
Plate

- human-made

A natural watercourse that flows towards a sea, ocean, lake, or another river. Can
provide fresh water, support ecosystems, and serve as a means for transportation
and recreation.

A data storage device used in computers and other electronic devices to store and
retrieve digital information. Can store large amounts of data, such as operating
systems, applications, and personal files.

A naturally occurring solid aggregate of one or more minerals or mineraloids. Can
vary in size, composition, and texture, providing materials for construction and
other uses.

A flat, typically round dish made from materials such as ceramic, glass, or plastic.
Can be used for eating and serving food from.

grouped by effect size as well as full statistics, can be found in the
Supplementary Material (Section 2), and the ordering and differences between entities can also be seen in the following figures.
Figure 3 and Figure 4 show the differences between AIs for mind
and morality, respectively. Figure 5 and Figure 6 show the differences between all entities, AI and non-AI. We summarize people’s
perceptions of AIs and non-AI categories in Table 4.

**4.1** **H1: There will be differences between AIs in**
**their degree of perceived (a) agency, (b)**
**experience, (c) moral agency, and (d) moral**
**patiency**

_4.1.1_ _Agency._ The means for agency and experience for each AI
are shown in Figure 3. The AI perceived as having least agency was

Jennie ( _𝑀_ = 21.85 on the 0-100 scale), and the AI perceived as having
the most agency was Cicero ( _𝑀_ = 39.42). Supporting H1a, there
were significant differences between AIs in their degree of perceived
agency. The two game-playing AIs—Cicero and AlphaZero—Sophia,
and the Full Self-Driving car, were each perceived to have more
agency with at least medium effect size than each of Alexa, Siri,
DALL-E, Roomba, and Jennie, and more agency with small effect
size than the three LLM-based chatbots (ChatGPT, Replika, Wysa).
The two relatively autonomous AIs, Spot, and AgentGPT, were
both perceived to have greater agency with medium effect sizes
than Jennie, and with at least small effect size than ChatGPT, Alexa,
Siri, and DALL-E, and Roomba. The three LLM-based chatbots had
non-significantly different levels of perceived agency from each
other.

_4.1.2_ _Experience._ There was less variation in perceptions of experience than agency. The AI perceived as having the least experience
was Roomba ( _𝑀_ = 4.86), and the AI perceived as having the most
experience was Jennie ( _𝑀_ = 16.87). Supporting H1b, there were
significant differences between AIs in their degree of perceived
experience. Jennie was perceived to have greater experience, with
a medium effect size, than 11 of the other AIs (ChatGPT, Wysa,
AgentGPT, Alexa, Siri, DALL-E, Roomba, Spot, Full Self-Driving car,
AlphaZero, Cicero), and with a small effect size than Replika and
Sophia. Replika and Sophia were perceived to have greater experience, with small effect size, than each of the other AIs, and were
non-significantly different from each other. Aside from Wysa, AlphaZero, and Cicero being perceived as more experiential with small
effect sizes than Roomba, all of the other comparisons between the
AIs were non-significantly different.

_4.1.3_ _Moral agency._ The means for moral agency and moral patiency for each of the AIs are shown in Figure 4. The AI attributed
the least amount of moral agency was Roomba ( _𝑀_ = 23.20), and
the AI attributed the most agency was a Tesla Full Self-Driving
car ( _𝑀_ = 54.58). Supporting H1c, there were differences between
the AIs in attributed moral agency. The Full Self-Driving car was
rated significantly higher than every other AI, with large effect
sizes than Roomba and AlphaZero, with medium effect sizes than
Alexa, Siri, DALL-E, Jennie, and Cicero, and with small effect sizes
than ChatGPT, Replika, Wysa, AgentGPT, Spot, and Sophia. The
next highest-rated AIs, Wysa, Sophia, and Replika, were attributed
significantly greater moral agency than each of the other ten AIs
with varying effect sizes and were non-significantly different from
each other.

_4.1.4_ _Moral patiency._ The AI attributed the least moral patiency
was Roomba ( _𝑀_ = 18.41), and the AI attributed the most moral
patiency was Jennie ( _𝑀_ = 35.92). Supporting H1d, there were significant differences between the AIs in attributed moral patiency.
The two highest-rated AIs, Jennie and Sophia, were attributed more
moral patiency, with medium effect sizes, than each of Alexa, Siri,
DALL-E, Roomba, AlphaZero, and Cicero, and with small effect sizes
than each of ChatGPT, Replika, Wysa, AgentGPT, Spot, and the Full
Self-Driving car. Spot and the Full Self-Driving car were attributed
more moral patiency with medium effect size than Roomba, and
with small effect sizes than every other AI other than Jennie, Sophia,
and each other. The three chatbots were attributed greater moral
patiency with small effect sizes than each of Alexa, Siri, Roomba,
AlphaZero, and Cicero, and were non-significantly different from
each other.

**4.2** **H2: There will be differences between AIs**
**and non-AI entities in their degree of**
**perceived (a) agency, (b) experience, (c)**
**moral agency, and (d) moral patiency**

_4.2.1_ _Agency._ The means for agency and experience for each entity
are shown in Figure 5. The entity perceived to be the least agentic
was a rock ( _𝑀_ = 4.01), and the entity perceived to be the most
agentic was an adult human ( _𝑀_ = 93.72). The AIs were all rated
between a hard drive and a toddler. Supporting H2a, there were
differences between the AIs and non-AIs on their perceived agency.

CHI ’25, April 26-May 1, 2025, Yokohama, Japan

Each of the AIs were perceived to have more agency, with large
effect sizes, than two of the three inanimate objects (a plate and a
rock) and with at least medium effect size than the other inanimate
object (a hard drive) and the natural entity (a river), except Jennie,
which was significantly greater than each of these entities, but
with a small effect size compared to a hard drive. Each of the AIs
were also perceived to have less agency, with large effect sizes,
than an adult human, a chimpanzee, and a corporation, and with at
least medium effect size than a cat and an ant. Several of the AIs
(AgentGPT, Sophia, Full Self-Driving car, AlphaZero, and Cicero)
were perceived to have non-significantly different agency than a
chicken, and the highest-rated AI, Cicero, was perceived as having
non-significantly different agency than a human toddler.

_4.2.2_ _Experience._ The entity perceived to have the least experience
was a plate ( _𝑀_ = 3.64), and the entity perceived as having the
most experience was an adult human ( _𝑀_ = 96.93). Supporting H2b,
there were significant differences between the AIs and non-AIs
in perceived experience. The most experiential AI, Jennie, was
perceived to have greater experience, with medium effect sizes,
than each of the inanimate objects (plate, rock, hard drive), and
with small effect size than the natural entity (river). Jennie was
perceived to have less experience than each of the other non-AI
entities, with large effect sizes compared to the nonhuman animals
and humans, and with small effect size than the abstract/collective
entity (corporation). Several of the AIs (Replika, Wysa, Sophia,
AlphaZero, and Cicero) had greater perceived experience with small
effect sizes than the three inanimate objects, though there were also
non-significant differences between many of the AIs and inanimate
objects (e.g., ChatGPT, AgentGPT, and Siri being non-significantly
different to a rock and a hard drive).

_4.2.3_ _Moral agency._ The means for moral agency and moral patiency for each entity are shown in Figure 6. The entity attributed
the least moral agency was a plate ( _𝑀_ = 9.33) and the entity attributed the most moral agency was an adult human ( _𝑀_ = 94.28).
Supporting H2c, there were differences between the AIs and nonAIs on moral agency. Each of the AIs was attributed more moral agency than two of the three inanimate objects (a plate and a rock)
with large effect sizes and with medium effect sizes than the other
inanimate object (hard drive) and the natural entity (river), except
Roomba, which was attributed greater moral agency with medium
effect sizes compared to a plate and a rock and with small effect
sizes compared to a hard drive and a river. All of the 14 AIs were
perceived to have less moral agency than an adult human and
a corporation with large effect sizes. Each AI was rated significantly lower than the next most morally agentic non-AI, a chimpanzee, with varying effect sizes, except for the Full Self-Driving
car, where the difference was non-significant. Sophia, Replika, and
Wysa were rated significantly higher than all of the nonhuman
animals with varying effect sizes except for a chimpanzee, against
which they were rated lower with small effect sizes. Otherwise,
there was a wide range in attributed moral agency to the AIs, such
as Roomba rated non-significantly differently to a crab, AlphaZero
rated non-significantly differently to a chicken, and ChatGPT rated
non-significantly differently to a cat.

CHI ’25, April 26-May 1, 2025, Yokohama, Japan Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
||~~Jen~~|~~nie~~||||||||
|||||||||||
||||||Replik|a|Sophia|||
||||||Wys|a Full S|Alpha elf−Driving|Zero Cic car|ero|
|||Roomba ~~Alexa~~|~~DALL·E~~ Siri|C|hatGPT|A Spot|gentGPT|||
|||||||||||
|||||||||||

## Agency

Figure 3: Ratings of mind (agency and experience) for the 14 AI entities.**

_4.2.4_ _Moral_ _patiency._ The entity attributed the least moral patiency was a rock ( _𝑀_ = 11.74) and the entity attributed the most
moral patiency was an adult human ( _𝑀_ = 95.39). Supporting H2d,
there were significant differences between the AIs and non-AIs on
moral patiency. Jennie, Sophia, Spot, and the Full Self-Driving car
were attributed greater moral patiency with at least medium effect
sizes than each of the three inanimate objects. The other AIs were
attributed greater moral patiency than the three inanimate objects
with at least small effect sizes, except Roomba, for which the difference with a hard drive was non-significant. Otherwise, each of
the AIs were attributed less moral patiency, with large effect sizes,
than each of the other non-AI entities, except for the differences
between Jennie and Sophia and an ant, which were medium sized.

**4.3** **H3: Perceived mind and morality will be**
**related within AIs such that (a) agency will**
**be positively associated with moral agency,**
**and (b) experience will be positively**
**associated with moral patiency**

Supporting H3a, agency was significantly associated with moral
agency, and supporting H3b, experience was significantly associated with moral patiency (see Model 1 and Model 4 in Table 3).
Additionally, in the same models, experience was significantly associated with moral agency, and agency was significantly associated
with moral patiency. For moral agency, the effects of agency and
experience on moral agency have a similar a size, whereas for moral
patiency, the effect of experience is more than twice the size of the
effect of agency.

The exploratory models that included familiarity (Models 2 and
5) and the exploratory models that included non-AI entities (Models 3 and 6) are also shown in Table 3. Models 2 and 5 show that,
while controlling for agency and experience, familiarity with AI

CHI ’25, April 26-May 1, 2025, Yokohama, Japan

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|||||J|ennie||Sophia|||
|||||||Spot||||
||||||||~~Full Self~~ |~~ −Driving c~~|~~ r~~|
|||||||Re|plika ~~Wysa~~|||
||||||~~Cha~~|~~GPT~~ ||||
|||~~Al~~|~~phaZero~~|DALL·E |~~gentGPT~~ ~~iri~~ Cicero|||||
||||||~~Alexa~~ |||||
||Room|ba||||||||
|||||||||||
|||||||||||

Moral agency was non-significantly associated with moral agency, but significantly associated with moral patiency. The negative interaction
term in Model 3 between the type of entity (AI or non-AI) and
agency and the positive interaction term in Model 6 between the
type of entity agency show that relationship between agency and
moral agency was weaker for AIs compared to non-AIs, and the
relationship between agency and moral patiency was stronger for
AIs compared to non-AIs. The positive interaction terms in Models 3 and 6 between type of entity and experience show that the
relationship between experience and both moral agency and moral
patiency was stronger for AIs compared to non-AIs.

Figure 4: Ratings of morality (moral agency and moral patiency) for the 14 AI entities.**

**5** **Discussion**

We aimed to understand perceptions of mind and morality across a
broad range of AI systems. To do this, we conducted an online study
in which participants rated 26 entities, 14 AIs and 12 non-AIs, on
their degree of mind (agency and experience) and morality (moral
agency and moral patiency). Supporting our first two hypotheses,
we found differences between the AIs and between the AIs and
non-AIs on each of the four mental and moral faculties. Supporting
our third hypothesis, we found that within AIs—and across all
entities—mental agency predicted moral agency and experience
predicted moral patiency. We explored specific differences between entities and alternative model specifications to further understand
these effects.

Consistent with previous research, the AIs were perceived as
higher in agency than experience, and there was more variation
in perceptions of agency than experience [16, 25]. The two AIs
perceived as most agentic were Cicero and AlphaZero, both game-playing AIs known for their ability to perform better than humans
in highly strategic games, which arguably requires a relatively high
degree of agentic capacities such as long-term planning. The most
experiential AI, the robot puppy Jennie, was perceived as moderately more experiential than the other AIs, showing that it is
possible to enhance perceived experience in AI above an extremely
low level. This effect may be due to a combination of Jennie’s particularly lifelike appearance and social purpose, both of which increase
perceptions of experience [9, 60]. Jennie’s lifelike appearance may
also be strong enough to pass beyond the uncanny valley in the
U-shaped relationship between similarity and comfort in animals

[36]. Notably, LLM-based chatbots (ChatGPT, Replika, and Wysa),
which were not included in earlier comparative studies, were all
perceived to have a degree of agency roughly in the middle of the
AIs. Like most of the other AIs, ChatGPT and Wysa were perceived
to have very low experience, while Replika was slightly higher,
likely due to having a high degree of emotional expressiveness.

CHI ’25, April 26-May 1, 2025, Yokohama, Japan Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||Toddler|Cat C|himpanze|e|~~Adult hum~~|~~ an~~|
||||C|hicken ||||||
|||||~~Cra~~|_b|||||
||||||Ant|||||
|||||||||||
|||||||~~Corpor~~|~~ation~~|||
|||DAL Jennie |L·E  Replika ~~Sop~~ Wy|~~hia~~ Full S sa|elf−Drivin|g car ||||
|Roc|_R Hard dr ~~late~~ k Riv|Alexa C  ~~omba~~ ive er|hatGPT  Siri |AgentGPT ~~Alph~~ Cicer |~~aZero~~ o|||||
|||||Spot ||||||

## Agency

a a

Abstract/collective entity a Human a
Artificial intelligence a Inanimate object a a a

Human
Inanimate object a

Natural entity
Nonhuman animal

Figure 5: Ratings of mind (agency and experience) for all 26 entities.**

CHI ’25, April 26-May 1, 2025, Yokohama, Japan

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
||||To|Cat C ddler|himpanze|e|A|dult huma|n|
|||||||||||
||R|C iver|~~Chicken~~ rab|||||||
||||Ant||||Corpora|tion||
||||Jen |nie |~~ophia~~|||||
||||Ci  |ChatGPT cero    Spot|~~Replika~~  Full Self Wysa|−Driving c|ar|||
||Plate  Hard|Alph Roomb  drive|aZero ~~DALL·E~~ a Sir|AgentG Alexa  i|PT |||||
|||~~ock~~||||||||
|||||||||||

## Moral agency

a a

Abstract/collective entity a Human a
Artificial intelligence **a** Inanimate object a a

**a**

Human
Inanimate object a

Natural entity
Nonhuman animal

Figure 6: Ratings of morality (moral agency and moral patiency) for all 26 entities.**

CHI ’25, April 26-May 1, 2025, Yokohama, Japan Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis

Table 3: Results of mixed-effects models for the effect of perceived mind on moral attribution. Each model included random**
**effects for participant and entity. For predicting moral agency, Model 1 includes only agency and experience as fixed effects.**
**Model 2 adds in familiarity, and Model 3 adds in type of entity (and interaction terms), but not familiarity. Models 4, 5, and 6**
**analogously predict moral patiency. *** _𝑝_ **< 0.1, **** _𝑝_ **< 0.05, ***** _𝑝_ **< 0.01. Standard errors in parentheses. R-squared is the marginal**
**R-squared, i.e., it is the proportion of variance explained by the fixed effects terms only.**

**Moral**
**agency**
**(Model 1)**

31.54***
**Intercept**
(2.06)

0.28***
**Agency**
(0.01)

0.26***
**Experience**
(0.02)

**Moral**
**agency**
**(Model 2)**

28.03***
(3.29)

0.28***
(0.01)

0.26***
(0.02)

**Moral**
**agency**
**(Model 3)**

21.58***
(3.81)

0.42***
(0.01)

-0.01
(0.02)

8.76*
(5.09)

-0.06***

(0.02)

0.11***
(0.02)

**Moral**
**patiency**
**(Model 4)**

17.17***
(1.35)

0.20***
(0.01)

0.43***
(0.02)

**Moral**
**patiency**
**(Model 5)**

9.87***
(2.40)

0.20***
(0.01)

0.43***
(0.02)

1.84***
(0.50)

**Moral**
**patiency**
**(Model 6)**

32.50***
(3.46)

0.13***
(0.01)

0.43***
(0.01)

-17.71***

(4.64)

0.12***
(0.02)

0.09***
(0.02)

0.89
**Familiarity**
(0.65)

**Type of entity**

**(non-AI =**

**0, AI = 1)**

**Type of entity:**

**Agency**

**Type of entity:**

**Experience**

**Observations** 6,893 6,893 12,670 6,895 6,895 12,671

**R-squared** 0.11 0.11 0.16 0.18 0.19 0.48

Table 4: Summary of perceptions of AIs compared to categories of non-AI entities.**

**Entity Category** **Agency** **Experience** **Moral Agency** **Moral Patiency**

**Artificial Intelligences** Low to moderate Very low to low Low to moderate Low

**Inanimate objects** Very low Very low Very low Very low

**Natural entity (river)** Very low Very low Very low High

**Abstract/collective**
High Low Very high Moderate
**entity (corporation)**

**Nonhuman animals** Moderate to high Moderate to very high Low to moderate High to very high

**Humans** Moderate to very high Very high Moderate to very high Very high

Comparing perceived mind in the AIs to various categories of
non-AIs—inanimate objects, a natural entity (river), an abstract/
collective entity (corporation), nonhuman animals, and humans—AIs
formed a distinct cluster, low-to-moderate in agency and low in
experience (see Table 4 for comparison with other categories). The
AIs were generally perceived as higher in agency than inanimate
objects and at most equal to the least agentic nonhuman animal (a
chicken). AIs were generally perceived as equally or slightly more
experiential than inanimate objects, and all the AIs were perceived to have far less experience than nonhuman animals and humans.
These findings are consistent with the theory that AIs are perceived
as a new or exceptional “ontological category,” neither fully animate
nor fully inanimate [28, 61]. The AIs were also perceived differently
to the natural entity (river)—higher in agency, and equal or higher in
experience—and the abstract/collective entity (corporation)—lower
in both agency and experience. Thus, on mind, the AIs were perceived differently to every other sort of entity we studied. That said, the pattern of perceived agency and experience for the AIs was similar to how the corporation was perceived—lower in experience and
higher in agency. This raises the possibility that as AIs continue to
become increasingly advanced, they will be perceived increasingly
like corporations. However, corporations are quite different sorts of
entities to AIs, particularly due to their collective nature. This raises
questions about how people would think notions that apply to corporations, such as the right to enter contracts and be sued, would
translate to AI. It is also important to note that there is variation
in perceptions of corporations [3, 54] which could mirror variation
in perceptions of AI, such as a mental health chatbot like Wysa
being perceived more similarly to a non-profit corporation and a
negotiating game-playing AI such as Cicero being perceived more
similarly to a for-profit. Future research should explore whether
and how AIs are and will be perceived like corporations.

The analogous moral faculties (moral agency and moral patiency)
were rated higher than mind (agency and experience) in the AIs,
and consistent with previous research, attributions of moral agency
were higher than moral patiency [8, 32]. The Full Self-Driving car
was attributed the most moral agency among the AIs, followed by
Wysa, Sophia, and Replika. The most moral patiency was attributed
to Jennie, followed by Sophia, Spot, and the Full Self-Driving car.
Notably, there was no perfect correspondence between mind and
morality. For example, Cicero was attributed the highest mental
agency among all the AIs and a similar degree of experience to the
Full Self-Driving car, but was only attributed higher moral agency
than two other AIs (Roomba and AlphaZero). This suggests that
other, non-mind factors influence attributions of morality to the
AIs. For moral agency, this could be the types and severity of harms
that could be caused by the AIs [39] For example, the Full SelfDriving car could cause serious physical harm to people through
road accidents, and the chatbots Replika and Wysa could cause
serious psychological harm to their users. For moral patiency, an
entity having a physical body, particularly if it is human-like, could
play a role [31]. Notably, the four AIs rated highest in moral patiency—Jennie, Sophia, Spot, Full Self-Driving car—all have physical
bodies. Such attributions of moral patiency based on physicality
could have similar motivations to the well-documented negative
emotional reactions and aversion to destruction that people show
towards robots [46].

Compared to non-AI entities, the AIs were attributed moral
agency roughly comparable to various nonhuman animals (see
Table 4). For example, Roomba, the least morally agentic AI, was attributed a degree of moral agency similar to a crab (the least morally
agentic nonhuman animal) ChatGPT was attributed a degree similar to a cat (the second most morally agentic nonhuman animal),
and the Full Self-Driving car was attributed as much as a chimpanzee (the most morally agentic nonhuman animal, and the third
most morally agentic entity overall). On moral patiency, the AIs
largely formed a cluster, above inanimate objects and far below all
of the nonhuman animals, as well as below the abstract/collective
entity, the natural entity, and humans. So, although moral patiency
attributions to AIs are greater than perceptions of experience in AI,
they are still low—all the AIs were rated far below an ant, which are
attributed close to the least moral patiency of all nonhuman animals

[27]. As with mind, the pattern for morality—relatively low moral
patiency, higher moral agency—matches that of a corporation, just

CHI ’25, April 26-May 1, 2025, Yokohama, Japan to a lesser degree on both faculties. Taken together, our findings
suggest that AIs are perceived as distinct from other entities on
perceptions of agency, experience, and moral patiency. However,
they are attributed moral agency with more variability and to a
higher degree than these other three faculties.

What explains the relatively high attribution of moral agency
to some of the AIs? As noted above, this could be due to the high-stakes moral scenarios that they deployed in where severe harms
could be caused, such as road accidents and damage to people’s
mental health. A second possibility concerns the fact that, when
advanced AI systems cause harmful outcomes, such as through
road accidents, it can be unclear where the true responsibility lies
because of the difficulty of predicting the exact decisions of the
systems [40]. Due to the ambiguity created by this “responsibility
gap,” people may look to find a target of responsibility based on
other factors, such as causal proximity [15]. Finally, it is possible
that this effect is driven by people’s general aversion to AIs making
moral decisions [7]. Perhaps because of the discomfort people feel
when AIs are used for moral decisions, they more closely scrutinize
AIs that are used in such scenarios, translating to higher attributions
of moral agency. These possible explanations should be tested in
future research.

Consistent with mind perception theory [18, 19], our predictive
analysis showed that the two dimensions of mind predicted attributions of morality to AIs. However, it also suggested that there may
be differences in in how these dimensions of mind affect attributions
of morality between AIs and non-AIs. In particular, we found that
experience was more strongly associated with both moral agency
and moral patiency for AIs than for non-AIs, and agency was less
strongly associated with moral agency and more strongly associated with moral patiency for AIs than for non-AIs. This is important
because it suggests that the mental faculty that AIs are perceived
to lack, experience, may be particularly important for attributing
morality to them in general. It also suggests that the links between
agency and moral agency and experience and moral patiency may
not be as straightforward as sometimes assumed—specifically in
cases where a being can have agency with no experiential capacities, the importance of these concepts for morality appears to blur.
Our analysis is also consistent with a growing body of literature
suggesting there are some different factors that affect attributions
of morality to AIs compared to other entities [32], such as sci-fi fan
identity (i.e., the extent to which people identify with the science-fiction fan group) [44]. Our predictive analysis also found that
familiarity with AI is associated with greater attributions of moral
patiency to AI, but not with moral agency, while controlling for
agency and experience. However, due to the correlated nature of
these variables and the exploratory nature of this analysis, these
relationships should be re-tested in future research.

**5.1** **Design implications**

Our findings have several design implications. The relatively high
attribution of moral agency to some AIs, such as a Tesla Full SelfDriving car being attributed as much moral agency as a chimpanzee,
suggests that some AIs, particularly those that operate in high-stakes moral contexts, may be held morally responsible if their
actions were to result in harm. This could deflect from the creators,

CHI ’25, April 26-May 1, 2025, Yokohama, Japan Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis operators, and other actors who may, in fact, bear responsibility.
Addressing this is particularly challenging because there are multiple sources of uncertainty that make it possible to deflect blame
onto AIs: the true capabilities of the AI systems and the complexity
of their behavior [66], as well as the factors that make it appropriate
to hold an actor morally responsible, which can also be difficult to
define and measure, such as understanding and deliberating over
one’s moral choices [22].

Designers should be cautious of excessively anthropomorphizing AI systems that operate in high-stakes moral contexts, which
would reduce the “hype” of exaggerating AI capabilities and the
“fallacy” of attributing them greater moral responsibility than they warrant [45]. Additionally, designers should make sure to carefully
follow relevant guidelines, such as ensuring clarity around what
the systems can and cannot do, clarity around how well they can
do what they are designed for, and making clear why they make
the decisions they do [1]. Of course, given the complexity of these
systems, full clarity may not always be possible. However, it should
still be a top priority in these high-stakes moral contexts. We also
think it is appropriate that gaps in understanding of how these
systems operate, as well as where there are possible “responsibility
gaps” when the AIs cause harm, should be clearly documented and
communicated to users.

While there are situations where anthropomorphizing or otherwise enhancing perceptions of capacities in AIs can be harmful,
such as those just described, there are also situations where it could
enhance user experience without any obvious cost, such as with
voice assistants that operate in more neutral contexts like Alexa
and Siri. We found that AIs are generally perceived to have very
little experience, with even the latest, most advanced systems only
perceived to have slightly higher experience than earlier systems
and inanimate objects. While perceived experience can positively
affect user experience (e.g., [64]), attempting to enhance user experience by influencing perceived experience may not be the most
effective design strategy. Given the greater variation in the AIs’
perceived agency, this is likely the dimension of mind that is easier
to affect to achieve desired user experience outcomes. That said,
designers should take note of Jennie’s relatively high perceived
experience—perhaps Jennie’s particularly high level of lifelikeness
is what is needed to enhance perceived experience in AI above the
generally very low levels.

Overall, designers should be aware that attributions of morality
to AIs depend not only on their actual capacities but on perceptions
of their capacities, as well as other external factors, and these can
be influenced by designed choices in positive or negative ways,
as shown by the varying perceptions of the different AIs, especially on moral agency. Such choices could have knock-on effects,
such as making it more likely that AI systems are put in positions
of moral authority in society and that they play more of a role
in societal perceptions of moral rightness and wrongness, which
may or may not be desired. The safest approach, as promoted by
Schwitzgebel and Garza [49], may be to ensure that AI systems are
designed so that the extent to which they are attributed moral
agency and patiency is consistent with their actual levels of these
moral faculties. We think this is the best approach in the context of
AI systems deployed in high moral-stakes scenarios, and arguably
should be taken as a start point more broadly, with any deviation taken with caution and for good reason, taking into account any
potential unintended consequences.

**6** **Limitations**

Our study has several limitations. First, our sample was restricted
to people residing in the United States; given well-documented
differences in perceptions and attitudes towards AI across cultures

[12], future studies should look at perceptions of mind and morality
in AI with different samples. Additionally, our sample was recruited
from Prolific, and while we recruited a sample that was representative of the U.S. on age, gender, and ethnicity, it was not necessarily
representative on other characteristics.

Second, our study materials were short descriptions and images
of the various entities, which could have resulted in different responses than if participants had real-world interactions with the
entities. This is particularly the case given that participants were
unfamiliar with several of the entities (see Supplementary Material
Table S1) and, as described above, familiarity with AI was positively
associated with moral agency and patiency. We chose our method
rather than an in-person design because including such a wide
range of entities with real-world interactions would not be feasible.
However, future research should test in-person interactions with
some of the AIs included in our study to understand whether this
leads to different reactions.

Third, to mitigate potential survey fatigue, we measured our
dependent variables, agency, experience, moral agency, and moral
patiency, using only two items each. We believe we captured the
core essence of each of the constructs, however, including additional items would have measured them more comprehensively.
For example, important aspects of moral agency are blame and
punishment, and there is also a positive aspect to moral agency (i.e.,
praise and reward) [20, 38]. Future research should test people’s
perceptions of additional aspects of these constructs in real-world
AIs. Additionally, our two-item measures of moral agency and patiency do not allow us to fully distinguish between attributions to
the AIs themselves and the owners, operators, or developers of the
AIs. In the context of moral agency, this could be responsibility
to the owners, operators, or developers of the AIs [48]; for moral
patiency it could be the financial cost of damaging someone else’s
property [46]. We conducted supplementary analysis and found
this is unlikely to explain our findings (Supplementary Material
Section S3), but future research should specifically measure moral
attributions to these other entities.

Finally, while our study shows that there are differences in perceptions of mind and morality between different types of AIs and
between AIs and non-AI entities, it does not answer why these differences exist. We discussed possible reasons, such as the varying
moral contexts that the AIs operate in and their physical appearance, but future research should aim to better understand these
differences.

**7** **Conclusion**

In recent years, AI systems have become more advanced, common,
and diverse. We conducted a study in which participants rated 14
AIs and 12 non-AIs on mind (agency and experience) and morality
(moral agency and moral patiency). People perceive AIs to have low-to-moderate mental agency and little experience; these faculties were associated with the analogous moral faculties, moral
agency and moral patiency, which were rated higher than the mental faculties. Our study has several implications for theory and
design. First, variation within AIs on these faculties demonstrates
the need to consider the broad range of AIs rather than AI as a single category. At the same time, on three of the four faculties tested
(agency, experience, and moral patiency), AIs were consistently
perceived differently to other types of entities. By contrast, perception of AI moral agency was more varied. Our study also raises
questions about the distinction between agency and experience
and moral agency and moral patiency, as well the additional factors
not captured here that likely contribute to our judgments. Regarding design, most importantly, our study demonstrates the need to
manage perceptions of morality in contemporary AIs, particularly
moral agency, to maintain and promote trust and to ensure that
responsibility for the outcomes from AI is appropriately distributed.

**Acknowledgments**

Thanks to Aikaterina Manoli and Janet Pauketat for their helpful
discussion at various stages of this project.

**References**

[1] Saleema Amershi, Dan Weld, Mihaela Vorvoreanu, Adam Fourney, Besmira

Nushi, Penny Collisson, Jina Suh, Shamsi Iqbal, Paul N. Bennett, Kori Inkpen,
Jaime Teevan, Ruth Kikin-Gil, and Eric Horvitz. 2019. Guidelines for HumanAI Interaction. In _Proceedings of the 2019 CHI Conference on Human Factors in_

_Computing Systems_ (Glasgow, Scotland Uk) _(CHI ’19)_ . Association for Computing
Machinery, New York, NY, USA, 1–13. [https://doi.org/10.1145/3290605.3300233](https://doi.org/10.1145/3290605.3300233)

[2] Jacy Reese Anthis, Janet V. T. Pauketat, Ali Ladak, and Aikaterina Manoli. 2024.

What Do People Think about Sentient AI? [https://doi.org/10.48550/arXiv.2407.](https://doi.org/10.48550/arXiv.2407.08867)
[08867 arXiv:2407.08867 [cs]](https://doi.org/10.48550/arXiv.2407.08867)

[3] Rachel Hoi Yan Au and Gary Ting Tat Ng. 2021. Mind perception and stereotype

attribution of corporations and charities. _British Journal of Social Psychology_ 60,
1 (2021), 271–293. [https://doi.org/10.1111/bjso.12377](https://doi.org/10.1111/bjso.12377)

[4] Jaime Banks. 2019. A Perceived Moral Agency Scale: Development and Validation

of a Metric for Humans and Social Machines. _Computers in Human Behavior_ 90
(January 2019), 363–371. [https://doi.org/10.1016/j.chb.2018.08.028](https://doi.org/10.1016/j.chb.2018.08.028)

[5] Jaime Banks and Nicholas David Bowman. 2023. Perceived Moral Patiency of

Social Robots: Explication and Scale Development. _International Journal of Social_
_Robotics_ 15, 1 (January 2023), 101–113. [https://doi.org/10.1007/s12369-022-00950-](https://doi.org/10.1007/s12369-022-00950-6)
[6](https://doi.org/10.1007/s12369-022-00950-6)

[6] Yoav Benjamini and Yosef Hochberg. 1995. Controlling the False Discovery

Rate: A Practical and Powerful Approach to Multiple Testing. _Journal_ _of_ _the_
_Royal Statistical Society: Series B (Methodological)_ 57, 1 (1995), 289–300. [https:](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
[//doi.org/10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

[7] Yochanan E. Bigman and Kurt Gray. 2018. People Are Averse to Machines Making

Moral Decisions. _Cognition_ 181 (December 2018), 21–34. [https://doi.org/10.1016/](https://doi.org/10.1016/j.cognition.2018.08.003)
[j.cognition.2018.08.003](https://doi.org/10.1016/j.cognition.2018.08.003)

[8] Jean-François Bonnefon, Iyad Rahwan, and Azim Shariff. 2024. The Moral Psy
chology of Artificial Intelligence. _Annual_ _Review_ _of_ _Psychology_ 75, 1 (2024),
653–675. [https://doi.org/10.1146/annurev-psych-030123-113559](https://doi.org/10.1146/annurev-psych-030123-113559)

[9] Elizabeth Broadbent, Vinayak Kumar, Xingyan Li, John Sollers 3rd, Rebecca Q.

Stafford, Bruce A. MacDonald, and Daniel M. Wegner. 2013. Robots with Display
Screens: A Robot with a More Human-like Face Display Is Perceived To Have
More Mind and a Better Personality. _PLOS_ _ONE_ 8, 8 (August 2013), e72589.
[https://doi.org/10.1371/journal.pone.0072589](https://doi.org/10.1371/journal.pone.0072589)

[10] Jacob Cohen. 2016. _A power primer_ . American Psychological Association, Wash
ington, DC, US. [https://doi.org/10.1037/14805-018](https://doi.org/10.1037/14805-018) Pages: 284.

[11] Ben Cost. 2023. Married Father Commits Suicide after Encouragement by AI Chat
bot: Widow. [https://nypost.com/2023/03/30/married-father-commits-suicide-](https://nypost.com/2023/03/30/married-father-commits-suicide-after-encouragement-by-ai-chatbot-widow/)
[after-encouragement-by-ai-chatbot-widow/](https://nypost.com/2023/03/30/married-father-commits-suicide-after-encouragement-by-ai-chatbot-widow/)

[12] Jianning Dang and Li Liu. 2021. Robots Are Friends as Well as Foes: Ambivalent

Attitudes toward Mindful and Mindless AI Robots in the United States and
China. _Computers_ _in_ _Human_ _Behavior_ 115 (February 2021), 106612. [https:](https://doi.org/10.1016/j.chb.2020.106612)
[//doi.org/10.1016/j.chb.2020.106612](https://doi.org/10.1016/j.chb.2020.106612)

[13] Rob Eisinga, Manfred Grotenhuis, and Ben Pelzer. 2013. The Reliability of a

Two-Item Scale: Pearson, Cronbach, or Spearman-Brown? _International Journal_

CHI ’25, April 26-May 1, 2025, Yokohama, Japan

_of Public Health_ 58, 4 (August 2013), 637–642. [https://doi.org/10.1007/s00038-](https://doi.org/10.1007/s00038-012-0416-3)
[012-0416-3](https://doi.org/10.1007/s00038-012-0416-3)

[14] Imane El Atillah. 2023. AI Chatbot Blamed for ’encouraging’ Young Father to

Take His Own Life. https://www.euronews.com/next/2023/03/31/man-ends-hislife-after-an-ai-chatbot-encouraged-him-to-sacrifice-himself-to-stop-climate-.

[15] Neele Engelmann and Michael R. Waldmann. 2022. How causal structure, causal

strength, and foreseeability affect moral judgments. _Cognition_ 226 (Sept. 2022),
105167. [https://doi.org/10.1016/j.cognition.2022.105167](https://doi.org/10.1016/j.cognition.2022.105167)

[16] Heather M. Gray, Kurt Gray, and Daniel M. Wegner. 2007. Dimensions of Mind

Perception. _Science_ 315, 5812 (February 2007), 619–619. [https://doi.org/10.1126/](https://doi.org/10.1126/science.1134475)
[science.1134475](https://doi.org/10.1126/science.1134475)

[17] Kurt Gray and Daniel M. Wegner. 2012. Feeling Robots and Human Zombies:

Mind Perception and the Uncanny Valley. _Cognition_ 125, 1 (October 2012), 125–
130. [https://doi.org/10.1016/j.cognition.2012.06.007](https://doi.org/10.1016/j.cognition.2012.06.007)

[18] Kurt Gray and Daniel M. Wegner. 2012. Morality Takes Two: Dyadic Morality

and Mind Perception. In _The Social Psychology of Morality: Exploring the Causes_
_of_ _Good_ _and_ _Evil_ . American Psychological Association, Washington, DC, US,
109–127.

[19] Kurt Gray, Liane Young, and Adam Waytz. 2012. Mind Perception Is the Essence

of Morality. _Psychological Inquiry_ 23, 2 (April 2012), 101–124. [https://doi.org/10.](https://doi.org/10.1080/1047840X.2012.651387)
[1080/1047840X.2012.651387](https://doi.org/10.1080/1047840X.2012.651387)

[20] Steve Guglielmo. 2015. Moral Judgment as Information Processing: An Integrative

Review. _Frontiers in Psychology_ 6 (2015).

[21] Jamie Harris and Jacy Reese Anthis. 2021. The Moral Consideration of Artificial

Entities: A Literature Review. _Science and Engineering Ethics_ 27, 4 (August 2021),
53. [https://doi.org/10.1007/s11948-021-00331-8](https://doi.org/10.1007/s11948-021-00331-8)

[22] Kenneth Einar Himma. 2009. Artificial agency, consciousness, and the criteria

for moral agency: what properties must an artificial agent have to be a moral
agent? _Ethics_ _and_ _Information_ _Technology_ 11, 1 (March 2009), 19–29. [https:](https://doi.org/10.1007/s10676-008-9167-5)
[//doi.org/10.1007/s10676-008-9167-5](https://doi.org/10.1007/s10676-008-9167-5)

[23] Joo-Wha Hong and Dmitri Williams. 2019. Racism, Responsibility and Autonomy

in HCI: Testing Perceptions of an AI Agent. _Computers in Human Behavior_ 100
(November 2019), 79–84. [https://doi.org/10.1016/j.chb.2019.06.012](https://doi.org/10.1016/j.chb.2019.06.012)

[24] Angel Hsing-Chi Hwang and Andrea Stevenson Won. 2022. AI in Your Mind:

Counterbalancing Perceived Agency and Experience in Human-AI Interaction.
In _CHI Conference on Human Factors in Computing Systems Extended Abstracts_ .
ACM, New Orleans LA USA, 1–10. [https://doi.org/10.1145/3491101.3519833](https://doi.org/10.1145/3491101.3519833)

[25] Oliver L. Jacobs, Kamel Gazzaz, and Alan Kingstone. 2022. Mind the Robot!

Variation in Attributions of Mind to a Wide Set of Real and Fictional Robots.
_International_ _Journal_ _of_ _Social_ _Robotics_ 14, 2 (March 2022), 529–537. [https:](https://doi.org/10.1007/s12369-021-00807-4)
[//doi.org/10.1007/s12369-021-00807-4](https://doi.org/10.1007/s12369-021-00807-4)

[26] Oliver L. Jacobs, Farid Pazhoohi, and Alan Kingstone. 2023. Brief Exposure

Increases Mind Perception to ChatGPT and Is Moderated by the Individual
Propensity to Anthropomorphize. [https://doi.org/10.31234/osf.io/pn29d](https://doi.org/10.31234/osf.io/pn29d)

[27] Bastian Jaeger and Matti Wilks. 2023. The Relative Importance of Target and

Judge Characteristics in Shaping the Moral Circle. _Cognitive Science_ 47, 10 (2023),
e13362. [https://doi.org/10.1111/cogs.13362](https://doi.org/10.1111/cogs.13362)

[28] Peter H. Kahn, Aimee L. Reichert, Heather E. Gary, Takayuki Kanda, Hiroshi Ishig
uro, Solace Shen, Jolina H. Ruckert, and Brian Gill. 2011. The new ontological category hypothesis in human-robot interaction. In _Proceedings of the 6th international_
_conference on Human-robot interaction (HRI ’11)_ . Association for Computing Machinery, New York, NY, USA, 159–160. [https://doi.org/10.1145/1957656.1957710](https://doi.org/10.1145/1957656.1957710)

[29] Hiroko Kamide, Friederike Eyssel, and Tatsuo Arai. 2013. Psychological Anthro
pomorphism of Robots. In _Social Robotics_, Guido Herrmann, Martin J. Pearson,
Alexander Lenz, Paul Bremner, Adam Spiers, and Ute Leonards (Eds.). Springer
International Publishing, Cham, 199–208. [https://doi.org/10.1007/978-3-319-](https://doi.org/10.1007/978-3-319-02675-6_20)
[02675-6_20](https://doi.org/10.1007/978-3-319-02675-6_20)

[30] Taenyun Kim, Maria D. Molina, Minjin (Mj) Rheu, Emily S. Zhan, and Wei Peng.

2023. One AI Does Not Fit All: A Cluster Analysis of the Laypeople’s Perception
of AI Roles. In _Proceedings_ _of_ _the_ _2023_ _CHI_ _Conference_ _on_ _Human_ _Factors_ _in_
_Computing Systems_ . ACM, Hamburg Germany, 1–20. [https://doi.org/10.1145/](https://doi.org/10.1145/3544548.3581340)
[3544548.3581340](https://doi.org/10.1145/3544548.3581340)

[31] Ali Ladak, Jamie Harris, and Jacy Reese Anthis. 2024. Which Artificial Intelli
gences Do People Care Most About? A Conjoint Analysis. In _Proceedings of the_
_2024 CHI Conference on Human Factors in Computing Systems_ . ACM, Honolulu
USA, 1–15.

[32] Ali Ladak, Steve Loughnan, and Matti Wilks. 2024. The Moral Psychology of

Artificial Intelligence. _Current Directions in Psychological Science_ 33, 1 (September
2024), 27–34.

[33] Mina Lee, Katy Ilonka Gero, John Joon Young Chung, Simon Buckingham Shum,

Vipul Raheja, Hua Shen, Subhashini Venugopalan, Thiemo Wambsganss, David
Zhou, Emad A. Alghamdi, Tal August, Avinash Bhat, Madiha Zahrah Choksi,
Senjuti Dutta, Jin L.C. Guo, Md Naimul Hoque, Yewon Kim, Simon Knight,
Seyed Parsa Neshaei, Antonette Shibani, Disha Shrivastava, Lila Shroff, Agnia
Sergeyuk, Jessi Stark, Sarah Sterman, Sitong Wang, Antoine Bosselut, Daniel
Buschek, Joseph Chee Chang, Sherol Chen, Max Kreminski, Joonsuk Park, Roy
Pea, Eugenia Ha Rim Rho, Zejiang Shen, and Pao Siangliulue. 2024. A Design
Space for Intelligent and Interactive Writing Assistants. In _Proceedings of the CHI_

CHI ’25, April 26-May 1, 2025, Yokohama, Japan Ali Ladak, Matti Wilks, Steve Loughnan, and Jacy Reese Anthis

_Conference on Human Factors in Computing Systems_ . ACM, Honolulu HI USA,
1–35. [https://doi.org/10.1145/3613904.3642697](https://doi.org/10.1145/3613904.3642697)

[34] Gabriel Lima, Nina Grgić-Hlača, and Meeyoung Cha. 2021. Human Perceptions

on Moral Responsibility of AI: A Case Study in AI-Assisted Bail Decision-Making.
In _Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems_ .
ACM, Yokohama Japan, 1–17. [https://doi.org/10.1145/3411764.3445260](https://doi.org/10.1145/3411764.3445260)

[35] Gabriel Lima, Changyeon Kim, Seungho Ryu, Chihyung Jeon, and Meeyoung

Cha. 2020. Collecting the Public Perception of AI and Robot Rights. _Proceedings_
_of_ _the_ _ACM_ _on_ _Human-Computer_ _Interaction_ 4, CSCW2 (October 2020), 1–24.
[https://doi.org/10.1145/3415206](https://doi.org/10.1145/3415206)

[36] Diana Löffler, Judith Dörrenbächer, and Marc Hassenzahl. 2020. The Uncanny

Valley Effect in Zoomorphic Robots: The U-Shaped Relation Between Animal
Likeness and Likeability. In _Proceedings of the 2020 ACM/IEEE International Confer-_
_ence on Human-Robot Interaction (HRI ’20)_ . Association for Computing Machinery,
New York, NY, USA, 261–270. [https://doi.org/10.1145/3319502.3374788](https://doi.org/10.1145/3319502.3374788)

[37] Bertram F. Malle. 2019. How Many Dimensions of Mind Perception Really Are

There? _Proceedings of the 41st Annual Meeting of the Cognitive Science Society_
(2019), 2268–2274.

[38] Bertram F. Malle, Steve Guglielmo, and Andrew E. Monroe. 2014. A Theory of

Blame. _Psychological Inquiry_ 25, 2 (April 2014), 147–186. [https://doi.org/10.1080/](https://doi.org/10.1080/1047840X.2014.877340)
[1047840X.2014.877340](https://doi.org/10.1080/1047840X.2014.877340)

[39] Timothy Maninger and Daniel B. Shank. 2022. Perceptions of Violations by

Artificial and Human Actors across Moral Foundations. _Computers in Human_
_Behavior_ _Reports_ 5 (March 2022), 100154. [https://doi.org/10.1016/j.chbr.2021.](https://doi.org/10.1016/j.chbr.2021.100154)
[100154](https://doi.org/10.1016/j.chbr.2021.100154)

[40] Andreas Matthias. 2004. The responsibility gap: Ascribing responsibility for the

actions of learning automata. _Ethics and Information Technology_ 6, 3 (Sept. 2004),
175–183. [https://doi.org/10.1007/s10676-004-3422-1](https://doi.org/10.1007/s10676-004-3422-1)

[41] Andrew E. Monroe, Kyle D. Dillon, and Bertram F. Malle. 2014. Bringing Free

Will down to Earth: People’s Psychological Concept of Free Will and Its Role in
Moral Judgment. _Consciousness and Cognition_ 27 (July 2014), 100–108. [https:](https://doi.org/10.1016/j.concog.2014.04.011)
[//doi.org/10.1016/j.concog.2014.04.011](https://doi.org/10.1016/j.concog.2014.04.011)

[42] Sari R.R. Nijssen, Barbara C. N. Müller, Tibor Bosse, and Markus Paulus. 2022.

Can You Count on a Calculator? The Role of Agency and Affect in Judgments
of Robots as Moral Agents. _Human–Computer Interaction_ 0, 0 (June 2022), 1–17.
[https://doi.org/10.1080/07370024.2022.2080552](https://doi.org/10.1080/07370024.2022.2080552)

[43] Sari R. R. Nijssen, Barbara C. N. Müller, Rick B. Van Baaren, and Markus Paulus.

2019. Saving the Robot or the Human? Robots Who Feel Deserve Moral Care.
_Social Cognition_ 37, 1 (February 2019), 41–S2. [https://doi.org/10.1521/soco.2019.](https://doi.org/10.1521/soco.2019.37.1.41)
[37.1.41](https://doi.org/10.1521/soco.2019.37.1.41)

[44] Janet V.T. Pauketat and Jacy Reese Anthis. 2022. Predicting the Moral Consider
ation of Artificial Intelligences. _Computers in Human Behavior_ 136 (November
2022), 107372. [https://doi.org/10.1016/j.chb.2022.107372](https://doi.org/10.1016/j.chb.2022.107372)

[45] Adriana Placani. 2024. Anthropomorphism in AI: hype and fallacy. _AI and Ethics_

4, 3 (Aug. 2024), 691–698. [https://doi.org/10.1007/s43681-024-00419-4](https://doi.org/10.1007/s43681-024-00419-4)

[46] Katie A. Riddoch and Emily S. Cross. 2021. “Hit the Robot on the Head With This

Mallet” – Making a Case for Including More Open Questions in HRI Research.
_Frontiers in Robotics and AI_ 8 (February 2021). [https://doi.org/10.3389/frobt.2021.](https://doi.org/10.3389/frobt.2021.603510)
[603510](https://doi.org/10.3389/frobt.2021.603510)

[47] Joshua Rottman, Charlie R. Crimston, and Stylianos Syropoulos. 2021. Tree
Huggers Versus Human-Lovers: Anthropomorphism and Dehumanization Predict
Valuing Nature Over Outgroups. _Cognitive Science_ 45, 4 (2021), e12967. [https:](https://doi.org/10.1111/cogs.12967)
[//doi.org/10.1111/cogs.12967](https://doi.org/10.1111/cogs.12967)

[48] Jordan Richard Schoenherr and Robert Thomson. 2024. When AI Fails, Who

Do We Blame? Attributing Responsibility in Human–AI Interactions. _IEEE_
_Transactions on Technology and Society_ 5, 1 (March 2024), 61–70. [https://doi.org/](https://doi.org/10.1109/TTS.2024.3370095)
[10.1109/TTS.2024.3370095](https://doi.org/10.1109/TTS.2024.3370095)

[49] Eric Schwitzgebel and Mara Garza. 2015. A Defense of the Rights of Artificial

Intelligences: Defense of the Rights of Artificial Intelligences. _Midwest Studies In_
_Philosophy_ 39, 1 (September 2015), 98–119. [https://doi.org/10.1111/misp.12032](https://doi.org/10.1111/misp.12032)

[50] Ava Elizabeth Scott, Daniel Neumann, Jasmin Niess, and Paweł W. Woźniak. 2023.

Do You Mind? User Perceptions of Machine Consciousness. In _Proceedings of the_
_2023 CHI Conference on Human Factors in Computing Systems_ . ACM, Hamburg
Germany, 1–19. [https://doi.org/10.1145/3544548.3581296](https://doi.org/10.1145/3544548.3581296)

[51] Daniel B. Shank and Alyssa DeSanti. 2018. Attributions of Morality and Mind to

Artificial Intelligence after Real-World Moral Violations. _Computers in Human_
_Behavior_ 86 (September 2018), 401–411. [https://doi.org/10.1016/j.chb.2018.05.014](https://doi.org/10.1016/j.chb.2018.05.014)

[52] Rebecca Q. Stafford, Bruce A. MacDonald, Chandimal Jayawardena, Daniel M.

Wegner, and Elizabeth Broadbent. 2014. Does the Robot Have a Mind? Mind
Perception and Attitudes Towards Robots Predict Use of an Eldercare Robot.
_International Journal of Social Robotics_ 6, 1 (January 2014), 17–32. [https://doi.](https://doi.org/10.1007/s12369-013-0186-y)
[org/10.1007/s12369-013-0186-y](https://doi.org/10.1007/s12369-013-0186-y)

[53] Darko Stojilović, Matija Franklin, Bertram F. Malle, Carlos Fernandez-Basso,

Edmond Awad, and David Lagnado. 2024. Are autonomous vehicles blamed
differently? _Proceedings of the Annual Meeting of the Cognitive Science Society_ 46
(2024). [https://escholarship.org/uc/item/7q35p9b7](https://escholarship.org/uc/item/7q35p9b7)

[54] Nina Strohminger and Matthew R. Jordan. 2022. Corporate insecthood. _Cognition_

224 (July 2022), 105068. [https://doi.org/10.1016/j.cognition.2022.105068](https://doi.org/10.1016/j.cognition.2022.105068)

[55] Michael T. Stuart and Markus Kneer. 2021. Guilty Artificial Minds: Folk Attribu
tions of Mens Rea and Culpability to Artificially Intelligent Agents. _Proceedings_
_of_ _the_ _ACM_ _on_ _Human-Computer_ _Interaction_ 5, CSCW2 (October 2021), 1–27.
[https://doi.org/10.1145/3479507](https://doi.org/10.1145/3479507)

[56] Tetsushi Tanibe, Takaaki Hashimoto, and Kaori Karasawa. 2017. We Perceive

a Mind in a Robot When We Help It. _PLOS_ _ONE_ 12, 7 (July 2017), e0180952.
[https://doi.org/10.1371/journal.pone.0180952](https://doi.org/10.1371/journal.pone.0180952)

[57] Kallie Tzelios, Lisa A. Williams, John Omerod, and Eliza Bliss-Moreau. 2022.

Evidence of the Unidimensional Structure of Mind Perception. _Scientific Reports_
12, 1 (November 2022), 18978. [https://doi.org/10.1038/s41598-022-23047-6](https://doi.org/10.1038/s41598-022-23047-6)

[58] Bart S. Vanneste and Phanish Puranam. 2024. Artificial Intelligence, Trust,
and Perceptions of Agency. _Academy_ _of_ _Management_ _Review_ (March 2024),
amr.2022.0041. [https://doi.org/10.5465/amr.2022.0041](https://doi.org/10.5465/amr.2022.0041)

[59] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,

Aidan N Gomez, Ł Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need.
In _Advances in Neural Information Processing Systems_, Vol. 30. Curran Associates,
Inc.

[60] Dan Wang, Alessandro Piazza, and Sarah A. Soule. 2018. Boundary-Spanning in

Social Movements: Antecedents and Outcomes. _Annual Review of Sociology_ 44, 1
(July 2018), 167–187. [https://doi.org/10.1146/annurev-soc-073117-041258](https://doi.org/10.1146/annurev-soc-073117-041258)

[61] Kara Weisman. 2022. Extraordinary entities: Insights into folk ontology from

studies of lay people’s beliefs about robots. _Proceedings of the Annual Meeting_
_of the Cognitive Science Society_ 44, 44 (2022). [https://escholarship.org/uc/item/](https://escholarship.org/uc/item/23p9h07t)
[23p9h07t](https://escholarship.org/uc/item/23p9h07t)

[62] Kara Weisman, Carol S. Dweck, and Ellen M. Markman. 2017. Rethinking People’s

Conceptions of Mental Life. _Proceedings of the National Academy of Sciences_ 114,
43 (October 2017), 11374–11379. [https://doi.org/10.1073/pnas.1704347114](https://doi.org/10.1073/pnas.1704347114)

[63] Joel Wester, Tim Schrills, Henning Pohl, and Niels Van Berkel. 2024. “As an

AI Language Model, I Cannot”: Investigating LLM Denials of User Requests. In
_Proceedings of the CHI Conference on Human Factors in Computing Systems_ . ACM,
Honolulu HI USA, 1–14. [https://doi.org/10.1145/3613904.3642135](https://doi.org/10.1145/3613904.3642135)

[64] Kai Chi Yam, Yochanan E. Bigman, Pok Man Tang, Remus Ilies, David De Cremer,

Harold Soh, and Kurt Gray. 2021. Robots at Work: People Prefer—and Forgive—
Service Robots with Perceived Feelings. _Journal of Applied Psychology_ 106, 10
(2021), 1557–1572. [https://doi.org/10.1037/apl0000834](https://doi.org/10.1037/apl0000834)

[65] Kai Chi Yam, E-Yang Goh, Ryan Fehr, Randy Lee, Harold Soh, and Kurt Gray. 2022.

When Your Boss Is a Robot: Workers Are More Spiteful to Robot Supervisors That
Seem More Human. _Journal of Experimental Social Psychology_ 102 (September
2022), 104360. [https://doi.org/10.1016/j.jesp.2022.104360](https://doi.org/10.1016/j.jesp.2022.104360)

[66] Qian Yang, Aaron Steinfeld, Carolyn Rosé, and John Zimmerman. 2020. Re
examining Whether, Why, and How Human-AI Interaction Is Uniquely Difficult
to Design. In _Proceedings of the 2020 CHI Conference on Human Factors in Com-_
_puting Systems (CHI ’20)_ . Association for Computing Machinery, New York, NY,
USA, 1–13. [https://doi.org/10.1145/3313831.3376301](https://doi.org/10.1145/3313831.3376301)

[67] Qiyuan Zhang, Christopher D. Wallbridge, Dylan M. Jones, and Phil Morgan. 2021.

The Blame Game: Double Standards Apply to Autonomous Vehicle Accidents.
In _Advances in Human Aspects of Transportation_, Neville Stanton (Ed.). Springer
International Publishing, Cham, 308–314. [https://doi.org/10.1007/978-3-030-](https://doi.org/10.1007/978-3-030-80012-3_36)
[80012-3_36](https://doi.org/10.1007/978-3-030-80012-3_36)

[68] John Zoshak and Kristin Dew. 2021. Beyond Kant and Bentham: How Ethical

Theories Are Being Used in Artificial Moral Agents. In _Proceedings of the 2021_
_CHI Conference on Human Factors in Computing Systems_ . ACM, Yokohama Japan,
1–15. [https://doi.org/10.1145/3411764.3445102](https://doi.org/10.1145/3411764.3445102)

CHI ’25, April 26-May 1, 2025, Yokohama, Japan

**A** **Image credits and sources for Figure 2**

Table 5: Image credits and sources for Figure 2.**

**Image** **Credit** **Source**

Fig 2. (a) Tesla [https://electrek.co/2021/12/01/tesla-full-self-driving-fsd-impressions/](https://electrek.co/2021/12/01/tesla-full-self-driving-fsd-impressions/)

Created by authors using
Fig 2. (b) [https://agentgpt.reworkd.ai/](https://agentgpt.reworkd.ai/)
Canva and AgentGPT app

[https://commons.wikimedia.org/wiki/File:](https://commons.wikimedia.org/wiki/File:IRobot_Roomba_870_(15860914940).jpg)
Fig 2. (c) Karlis Dambr¯ ans¯
[IRobot_Roomba_870_(15860914940).jpg](https://commons.wikimedia.org/wiki/File:IRobot_Roomba_870_(15860914940).jpg)

[https://unsplash.com/photos/3rd-gen-black-amazon-echo-dot-speaker-](https://unsplash.com/photos/3rd-gen-black-amazon-echo-dot-speaker-Ub4CggGYf2o)
Fig 2. (d) Novi Sad
[Ub4CggGYf2o](https://unsplash.com/photos/3rd-gen-black-amazon-echo-dot-speaker-Ub4CggGYf2o)

Fig 2. (e) ITU Pictures [https://commons.wikimedia.org/wiki/File:Sophia_(robot).jpg](https://commons.wikimedia.org/wiki/File:Sophia_(robot).jpg)

[https://unsplash.com/photos/a-close-up-of-a-hair-dryer-in-the-dark-](https://unsplash.com/photos/a-close-up-of-a-hair-dryer-in-the-dark-wZ49T2Tc7xw)
Fig 2. (f) Mika Baumeister
[wZ49T2Tc7xw](https://unsplash.com/photos/a-close-up-of-a-hair-dryer-in-the-dark-wZ49T2Tc7xw)

Fig 2. (g) Tombot [https://tombot.com/blogs/investors/tombot-inspired-mother-nancy-stevens](https://tombot.com/blogs/investors/tombot-inspired-mother-nancy-stevens)

Fig 2. (h) Anatolii Babii

Game board: Meta AI

[https://www.alamy.com/kyiv-ukraine-september-18-2019-a-close-up-shot-of-](https://www.alamy.com/kyiv-ukraine-september-18-2019-a-close-up-shot-of-apple-iphone-8-smartphone-with-a-siri-the-virtual-assistant-application-on-the-screen-image274718095.html)
[apple-iphone-8-smartphone-with-a-siri-the-virtual-assistant-application-on-](https://www.alamy.com/kyiv-ukraine-september-18-2019-a-close-up-shot-of-apple-iphone-8-smartphone-with-a-siri-the-virtual-assistant-application-on-the-screen-image274718095.html)
[the-screen-image274718095.html](https://www.alamy.com/kyiv-ukraine-september-18-2019-a-close-up-shot-of-apple-iphone-8-smartphone-with-a-siri-the-virtual-assistant-application-on-the-screen-image274718095.html)

[Game board: https://nypost.com/2024/05/14/business/metas-ai-system-cicero-](https://nypost.com/2024/05/14/business/metas-ai-system-cicero-beats-humans-in-game-of-diplomacy-by-lying-study/)
[beats-humans-in-game-of-diplomacy-by-lying-study/](https://nypost.com/2024/05/14/business/metas-ai-system-cicero-beats-humans-in-game-of-diplomacy-by-lying-study/)
[Dialogue: https://www.sciencedirect.com/science/article/pii/S266638992400103X](https://www.sciencedirect.com/science/article/pii/S266638992400103X)
[Logo: https://jrodthoughts.medium.com/meta-ais-new-super-model-cicero-is-](https://jrodthoughts.medium.com/meta-ais-new-super-model-cicero-is-able-to-negotiate-and-cooperate-with-people-58f718b95dd1)
[able-to-negotiate-and-cooperate-with-people-58f718b95dd1](https://jrodthoughts.medium.com/meta-ais-new-super-model-cicero-is-able-to-negotiate-and-cooperate-with-people-58f718b95dd1)

Fig 2. (i)

Dialogue: Patterns

/ Park et al. (2024)

Logo: Meta AI

Created by authors using
Fig 2. (j) [https://replika.com/](https://replika.com/)

Canva and Replika app

[Image: https://openai.com/index/dall-e-3-is-now-available-in-chatgpt-plus-and-](https://openai.com/index/dall-e-3-is-now-available-in-chatgpt-plus-and-enterprise/)
[enterprise/](https://openai.com/index/dall-e-3-is-now-available-in-chatgpt-plus-and-enterprise/)
[Logo: https://openai.com/index/dall-e-3/](https://openai.com/index/dall-e-3/)

Fig 2. (k)

Image: Valerie Ranum × DALL·E, edited by authors using Canva

Logo: Open AI

Created by authors using
Fig 2. (l) [https://www.wysa.com/](https://www.wysa.com/)

Canva and Wysa app

[Go board: https://human-centered.ai/2016/01/28/january-27-2016-major-](https://human-centered.ai/2016/01/28/january-27-2016-major-breakthrough-in-ai-research/)
[breakthrough-in-ai-research/](https://human-centered.ai/2016/01/28/january-27-2016-major-breakthrough-in-ai-research/)
[Chess board: https://www.chess.com/](https://www.chess.com/)
[Shogi board: https://www.chessvariants.org/shogi.html](https://www.chessvariants.org/shogi.html)
[AlphaZero logo: https://www.chess.com/news/view/new-alphazero-paper-](https://www.chess.com/news/view/new-alphazero-paper-explores-chess-variants)
[explores-chess-variants](https://www.chess.com/news/view/new-alphazero-paper-explores-chess-variants)

Fig 2. (m)

Go board: human-centered.ai

Chess board: Chess.com
Shogi board: Fergus Duniho

Logo: AlphaZero

Created by authors using
Fig 2. (n) [https://chatgpt.com/](https://chatgpt.com/)

Canva and ChatGPT app
