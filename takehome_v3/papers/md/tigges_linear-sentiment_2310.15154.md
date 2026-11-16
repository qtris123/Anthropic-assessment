# Linear Representations of Sentiment in Large Language Models

- **Authors:** Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, Neel Nanda
- **Date:** 2023-10-23
- **Source:** https://arxiv.org/abs/2310.15154
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `tigges_linear-sentiment_2310.15154`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Pre-publication draft

**Curt Tigges**^*****^(_♣_) **, Oskar John Hollinsworth**^*****^(_♡_) **, Atticus Geiger**^(_♠_⋆) **, Neel Nanda**^(_♢_)

> _♣_ EleutherAI Institute, _♡_ SERI MATS, _♠_ Stanford University, ⋆Pr(Ai)2R Group, _♢_ Independent

*Equal primary authors (order random)

## Abstract

Sentiment is a pervasive feature in natural language text, yet it is an open question how sentiment is represented within Large Language Models (LLMs). In this study, we reveal that across a range of models, sentiment is represented linearly: a single direction in activation space mostly captures the feature across a range of tasks with one extreme for positive and the other for negative. Through causal interventions, we isolate this direction and show it is causally relevant in both toy tasks and real world datasets such as Stanford Sentiment Treebank. Through this case study we model a thorough investigation of what a single direction means on a broad data distribution.

We further uncover the mechanisms that involve this direction, highlighting the roles of a small subset of attention heads and neurons. Finally, we discover a phenomenon which we term the summarization motif: sentiment is not solely represented on emotionally charged words, but is additionally summarised at intermediate positions without inherent sentiment, such as punctuation and names. We show that in Stanford Sentiment Treebank zero-shot classification, 76% of above-chance classification accuracy is lost when ablating the sentiment direction, nearly half of which (36%) is due to ablating the summarized sentiment direction exclusively at comma positions.

## 1 Introduction

Large language models (LLMs) have displayed increasingly impressive capabilities (Brown et al., 2020; Radford et al., 2019; Bubeck et al., 2023), but their internal workings remain poorly understood. Nevertheless, recent evidence (Li et al., 2023) has suggested that LLMs are capable of forming models of the world, i.e., inferring hidden variables of the data generation process rather than simply modeling surface word co-occurrence statistics. There is significant interest (Christiano et al. (2021), Burns et al. (2022)) in deciphering the latent structure of such representations.

In this work, we investigate how LLMs represent sentiment, a variable in the data generation process that is relevant and interesting across a wide variety of language tasks (Cui et al., 2023). Approaching our investigations through the frame of causal mediation analysis (Vig et al., 2020; Pearl, 2022; Geiger et al., 2023a), we show that these sentiment features are represented linearly by the models, are causally significant, and are utilized by human-interpretable circuits (Olah et al., 2020; Elhage et al., 2021a).

We find the existence of a single direction scientifically interesting as further evidence for the linear representation hypothesis (Mikolov et al., 2013; Elhage et al., 2022)– that models tend to extract properties of the input and internally represent them as directions in activation space. Understanding the structure of internal representations is crucial to begin to decode them, and linear representations are particularly amenable to detailed reverse-engineering (Nanda et al., 2023b).

We show evidence of a phenomenon we have labeled the “summarization motif”, where rather than sentiment being directly moved from valenced tokens to the final token, it is first aggregated on intermediate summarization tokens without inherent valence such as commas, periods and particular nouns.^1 This summarization structure for next token prediction can be seen as a naturally emerging

> 1Our use of the term “summarization” is distinct from typical NLP summarization tasks

[Figure text: have complete confidence  in  the Walt Disney World  currently  in  remission with et  son bon  à  rien  de mari You brought joy  to  the Brazilian Amazon has a speedy recovery to ils étaient par fait ement normaux despite the misery  it  presidential nominee Mitt Romney  radiation and cancer ( gris et t riste e t rien dans deemed a  hate group  overturn Bashar Assad.  you a migraine. la  plus sini str e pour aller (a) Nouns (b) Proper Nouns (c) Medical (d) French]

Figure 1: Visual verification that a single direction captures sentiment across diverse contexts. Color represents the projection onto this direction, blue is positive and red is negative. Examples (1a-1c) show the _K_ -means sentiment direction for the first layer of GPT2-small on samples from OpenWebText. Example 1d shows the _K_ -means sentiment direction for the 7th layer of pythia-1.4b on the opening of Harry Potter in French.

analogue to the explicit classification token in BERT-like models (Devlin et al., 2018). We show that the sentiment stored on summarization tokens is causally relevant for the final prediction. We find this an intriguing example of an “information bottleneck”, where the data generation process is funnelled through a small subset of tokens used as information stores. Understanding the existence and location of information bottlenecks is a key first step to deciphering world models. This finding additionally suggests the models’ ability to create summaries at various levels of abstraction, in this case a sentence or clause rather than a token.

Our contributions are as follows. In Section 3, we demonstrate methods for finding a **linear representation of sentiment** using a toy dataset and show that this direction correlates with sentiment information in the wild and matters causally in a crowdsourced dataset. In Section 4, we show through activation patching (Vig et al., 2020; Geiger et al., 2020) and ablations that the learned sentiment direction captures **summarization behavior** that is causally important to circuits performing sentiment tasks. Through this case study, we model an investigation of what a single interpretable direction means on the full data distribution.

## 2 Methods

2.1 DATASETS AND MODELS

**ToyMovieReview** A templatic dataset of continuation prompts we generated with the form

I thought this movie was ADJECTIVE, I VERBed it. Conclusion: This movie is where ADJECTIVE and VERB are either two positive words (e.g., incredible and enjoyed) or two negative words (e.g., horrible and hated) that are sampled from a fixed pool of 85 adjectives (split 55/30 for train/test) and 8 verbs. The expected completion for a positive review is one of a set of positive descriptors we selected from among the most common completions (e.g. great) and the expected completion for a negative review is a similar set of negative descriptors (e.g., terrible).

**ToyMoodStory** A similar toy dataset with prompts of the form

NAME1 VERB1 parties, and VERB2 them whenever possible. NAME2 VERB3 parties, and VERB4 them whenever possible. One day, they were invited to a grand gala. QUERYNAME feels very

To evaluate the model’s output, we measure the logit difference between the “excited” and “nervous” tokens.

**Stanford Sentiment Treebank (SST)** SST Socher et al. (2013) consists of 10,662 one sentence movie reviews with human annotated sentiment labels for every phrase from every review.

**OpenWebText** OWT (Gokaslan & Cohen, 2019) is the pretraining dataset for GPT-2 which we use as a source of random text for correlational evaluations.

**GPT-2 and Pythia** (Radford et al., 2019; Biderman et al., 2023) These are families of decoder-only transformer models with sizes varying from 85M to 2.8b parameters. We use GPT2-small for movie review continuation, pythia-1.4b for classification and pythia-2.8b for multi-subject tasks.

||DAS|K_means|LR|Mean_diff|PCA|Random|
|---|---|---|---|---|---|---|
|DAS|100.0%|72.6%|87.1%|86.9%|79.9%|2.4%|
|K_means|72.6%|100.0%|79.4%|83.1%|88.0%|1.7%|
|LR|87.1%|79.4%|100.0%|99.1%|90.2%|0.5%|
|Mean_diff|86.9%|83.1%|99.1%|100.0%|94.6%|0.5%|
|PCA|79.9%|88.0%|90.2%|94.6%|100.0%|1.2%|
|Random|2.4%|1.7%|0.5%|0.5%|1.2%|100.0%|

Figure 2: Cosine similarity of directions learned by different methods in GPT2-small’s first layer. Each sentiment direction was derived from _adjective_ representations in the ToyMovieReview dataset (Section 2.1).

### 2.2 Finding directions

We use five methods to find a sentiment direction in each layer of a language model using our ToyMovieReview dataset. In each of the following, let P be the set of positive inputs and N be the set of negative inputs. For some input _x ∈_ P _∪_ N, let **_a_**^_L_ _x_^and^**_v_** _x_^(_L_be the vector in the residual stream) at layer _L_ above the adjective and verb respectively. We reserve _{_ **_v_** _x_^(_L}_as a hold-out set for testing.) Let the correct next token for P be _p_ and for N be _n_ .

_K_ **-means (KM)** We fit 2-means to _{_ **_a_**^_L_ _x_^(:)^(_x ∈_P)^(_∪_N)^(_}_, obtaining cluster centroids)^(_{_)^**_c_**^(_i_:)^(_i ∈_[0)^(_,_1])^(_}_) and take the direction **_c_** 1 _−_ **_c_** 0.

**Linear Probing** The direction is the normed weights _||_ **_ww_** _||_^(of a logistic regression ()^(**LR**) classifier) **LR** ( _a_^_L_ _x_^() =) 1+exp(1 _−_ **_w_** _·_ **_a_**^_L_ _x_^()trained to distinguish between)^(_x ∈_P and)^(_x ∈_N.)

**Distributed Alignment Search (DAS)** (Geiger et al., 2023b) The direction is a learned parameter _θ_ where the training objective is the average logit difference after patching using direction _θ_ (see Section 2.3).

**Principal Component Analysis (PCA)** The direction is the first component of _{_ **_a_**^_L_ _x_^(:)^(_x ∈_P)^(_∪_N)^(_}_.)

### 2.3 Causal interventions

**Activation Patching** In activation patching (Geiger et al., 2020; Vig et al., 2020), we create two symmetrical datasets, where each prompt _x_ orig and its counterpart prompt _x_ flipped are of the same length and format but where key words are changed in order to flip the sentiment; e.g., “This movie was great” could be paired with “This movie was terrible.” We first conduct a forward pass using _x_ orig and capture these activations for the entire model. We then conduct forward passes using _x_ flipped, iteratively patching in activations from the original forward pass for each model component. We can thus determine the relative importance of various parts of the model with respect to the task currently being performed. Geiger et al. (2023b) introduce distributed interchange interventions, a variant of activation patching that we call “directional activation patching”. The idea is that rather than modifying the standard basis directions of a component, we instead only modify the component along a single direction in the vector space, replacing it during a forward pass with the value from a different input.

We use two evaluation metrics. The logit difference (difference in logits for correct and incorrect answers) metric introduced in Wang et al. (2022), as well as a “logit flip” accuracy metric (Geiger et al., 2022), which quantifies the proportion of cases where we induce an inversion in the predicted sentiment.

Figure 3: Area plot of sentiment labels for OpenWebText samples by _K_ -means sentiment activation (left). Accuracy using sentiment activations to classify tokens as positive or negative (right). The threshold taken is the top/bottom 0.1% of activations over OpenWebText. Sentiment activations are taken from GPT2-small’s first residual stream layer. Classification was performed by GPT-4.

**Ablations** We eliminate the contribution of a particular component to a model’s output, usually by replacing the component’s output with zeros (zero-ablation) or the mean over some dataset (meanablation), in order to demonstrate its magnitude of importance. We also perform directional ablation, in which a component’s activations are ablated only along a specific (e.g. sentiment) direction.

## 3 Finding and evaluating a ‘sentiment direction’

The first question we investigate is whether there exists a direction in the residual stream in a transformer model that represents the sentiment of the input text, as a special case of the linear representation hypothesis (Mikolov et al., 2013). We show that the methods discussed above (2.2) all arrive at a similar sentiment direction. Given some input text to a model, we can project the residual stream at a given token/layer onto a sentiment direction to get a ‘sentiment activation’.

### 3.1 Visualizing and comparing the directions

We fit directions using the ToyMovieReview dataset (Section 2.1) across various methods and finding extremely high cosine similarity between the learned sentiment directions (Figure 2). This suggests that these are all noisy approximations of the same singular direction. Indeed, we generally found that the following results were very similar regardless of exactly how we specified the sentiment direction. The directions we found were not sparse vectors, as expected since the residual stream is not a privileged basis (Elhage et al., 2021b).

Here we show a visualisation in the style of Neuroscope (Nanda, 2023a) where the projection is represented by color, with red being negative and blue being positive. It is important to note that the direction being examined here was trained on just 30 positive and 30 negative English adjectives in an unsupervised way (using _K_ -means with _K_ = 2). Notwithstanding, the extreme values along this direction appear readily interpretable in the wild in diverse text domains such as the opening paragraphs of Harry Potter in French (Figure 1). An interactive visualisation of residual stream directions in GPT2-small is available here (Yedidia, 2023) and sentiment directions here.

It is important to note that this type of analysis is qualitative, which should not act as a substitute for rigorous statistical tests as it is susceptible to interpretability illusions (Bolukbasi et al., 2021). We rigorously evaluate our directions using correlational and causal methods.

### 3.2 Correlational evaluation

In a correlational analysis, we classify word sentiment by ‘sentiment activation’ and show that the sentiment direction is sensitive to negation flipping sentiment.

||simple_logit_diff|treebank_logit_diff|simple_logit_flip|treebank_logit_flip|
|---|---|---|---|---|
|das|109.8%|47.0%|100.0%|53.5%|
|das2d|110.4%|42.8%|95.5%|49.0%|
|das3d|110.2%|35.9%|95.5%|39.4%|
|kmeans|67.2%|22.1%|72.7%|14.8%|
|logistic_regression|71.1%|30.8%|86.4%|16.8%|
|mean_diff|73.9%|27.5%|81.8%|17.4%|
|pca|62.7%|17.8%|72.7%|12.3%|
|random|0.4%|0.1%|0.0%|0.6%|

Figure 4: Directional patching results for different methods in pythia-1.4b. We report the best result found across layers. The columns show two evaluation datasets, ToyMovieReview and Treebank, and two evaluation metrics, mean logit difference and % of logit differences flipped.

**Sentiment Directions Capture Lexical Sentiment** To test the meaning of the sentiment axis, we binned the sentiment activations of OpenWebText tokens from the first residual stream layer of GPT2-small into 20 equal-width buckets and sampled 20 tokens from each. Then we asked GPT4 to classify into Positive/Neutral/Negative. Specifically, we gave the GPT-4 API prompts of the following form: “Your job is to classify the sentiment of a given token (i.e. word or word fragment) into Positive/Neutral/Negative. Token: ‘ _{_ token _}_ ’. Context: ‘ _{_ context _}_ ’. Sentiment: ” where the context length was 20 tokens centered around the sampled token. Only a cursory human sanity check was performed.

In Figure 3, we show an area plot of the classifications by activation bin. We contrast the results for different methods in Table 3. In the area plot we can see that the left side area is dominated by the “Negative” label, whereas the right side area is dominated by the “Positive” label and the central area is dominated by the “Neutral” label. Hence the tails of the activations seem highly interpretable as representing a bipolar sentiment feature. The large space in the middle of the distribution simply occupied by neutral words (rather than a more continuous degradation of positive/negative) indicates superposition of features (Elhage et al., 2022).

**Negation Flips the Sentiment Direction** Using the _K_ -means sentiment direction after the first layer of GPT2-small, we can obtain a view of how the model updates its view of sentiment during the forward pass, analogous to the “logit lens“ technique from nostalgebraist (2020). In Figure A.5, we see how the sentiment activation flips when the context of the sentiment word denotes that it is negated. Words like ‘fail’, ‘doubt’ and ‘uncertain’ can be seen to flip from negative in the first couple of layers to being positive after a few layers of processing. An interesting task for future circuits analysis research could be to better understand the circuitry used to flip the sentiment axis in the presence of a negation context. We suspect significant MLP involvement (see Section A.5).

### 3.3 Causal evaluation

**Sentiment directions are causal representations.** We evaluate the sentiment direction using directional patching in Figure 4. These evaluations are performed on prompts with out-of-sample adjectives and the direction was not trained on _any_ verbs. Unsupervised methods such as _K_ -means are still able to shift the logit differences and DAS is able to completely flip the prediction.

**Directions Generalize Most at Intermediate Layers** If the sentiment direction was simply a trivial feature of the token embedding, then one might expect that directional patching would be most effective in the first or final layer. However, we see in Figure 6 that in fact it is in intermediate layers of the model where we see the strongest out-of-distribution performance to SST. This suggests the speculative hypothesis that the model uses the residual stream to form abstract concepts in intermediate layers and this is where the latent knowledge of sentiment is most prominent.

||direction|flip percent|flip median size|
|---|---|---|---|
|L01 You never fail . Don 't doubt it |. I don 't like you . DAS|96%|107%|
|L04 You never fail . Don 't doubt it |. I don 't like you . KM|96%|69%|
|L07 You never fail . Don 't doubt it |. I don 't like you . MD|89%|45%|
|L10 You never fail . Don 't doubt it |. I don 't like you . LR PCA|100% 78%|86% 44%|

Figure 5: We made a dataset of 27 negation examples and compute the change in sentiment activation at the negated token (e.g. doubt) between the 1st and 10th layers of GPT2-small. We show sample text across layers for _K_ -means (left), the fraction of activations flipped and the median size of the flip centered around the mean activation (right).

Figure 6: Patching results for directions trained on toy datasets and evaluated on the Stanford Sentiment Treebank test partition. We tend to find the best generalisation when training and evaluating at a layer near the middle of the model. We scaffold the prompt using the suffix Overall the movie was very and compute the logit difference between good and bad. The patching metric (y-axis) is then the % mean change in logit difference.

**Activation Addition Steers the Model** A further verification of causality is shown in Figure A.3. Here we use the technique of “activation addition” from Turner et al. (2023). We add a multiple of the sentiment direction to the first layer residual stream during each forward pass while generating sentence completions. Here we start from the baseline of a positive movie review: “I really enjoyed the movie, in fact I loved it. I thought the movie was just very...”. By adding increasingly negative multiples of the sentiment direction, we find that indeed the completions become increasingly negative, without completely destroying the coherence of the model’s generated text. We are wary of taking the model’s activations out of distribution using this technique, but we believe that the smoothness of the transition in combination with the knowledge of our findings in the patching setting give us some confidence that these results are meaningful.

**Validation on SST** We validate our sentiment directions derived from toy datasets (Section 3.3) on SST. We collapsed the labels down to a binary “Positive”/“Negative”, just used the unique phrases rather than any information about their source sentences, restricted to the ‘test’ partition and took a subset where pythia-1.4b can achieve 100% zero shot classification accuracy, removing 17% of examples. Then we paired up phrases of an equal number of tokens^2 to make up 460 clean/corrupted pairs. We used the scaffolding “Review Text: TEXT, Review Sentiment:” and evaluated the logit difference between “Positive” and “Negative” as our patching metric. Using the same DAS direction from Section 3 trained on just a few examples and flipping the corresponding sentiment activation between clean/corrupted in a single layer, we can flip the output 53.5% of the time (Figure 4).

Figure 7: Primary components of GPT-2 sentiment circuit for the ToyMovieReview dataset. Here we can see both direct use of sentiment-laden words in predicting sentiment at END as well as an example of the summarization motif at the SUM position. Heads 7.1 and 7.5 write to this position and this information is causally relevant to the contribution of the summary readers at END.

## 4 The summarization motif for sentiment

### 4.1 Circuit analyses

In this sub-section, we present circuit^3 analyses that give qualitative hints of the summarization motif, and restrict quantitative analysis of the summarization motif to 4.2. Through an iterative process of path patching (see Section 2.3) and analysing attention patterns, we have identified the circuit responsible for the ToyMovieReview task in GPT2-small (Figure 7) as well as the circuit for the ToyMoodStories task. Below, we provide a brief overview of the circuits we identified, reserving the full details for A.3.

**Initial observations of summarization in GPT-2 circuit for ToyMovieReview** Mechanistically, this is a binary classification task, and a naive hypothesis is that attention heads attend directly from the final token to the valenced tokens and map positive sentiment to positive outputs and vice versa. This happens, but in addition attention head output is causally important at intermediate token positions, which are then read from when producing output at END. We consider this an instance of summarization, in which the model aggregates causally-important information relating to an entity at a particular token for later usage, rather than simply attending back to the original tokens that were the source of the information.

We find that the model performs a simple, interpretable algorithm to perform the task (using a circuit made up of 9 attention heads):

1. Identify sentiment-laden words in the prompt, at ADJ and VRB.

2. Write out sentiment information to SUM (the final “movie” token).

3. Read from ADJ, VRB and SUM and write to END.^4

The results of activation patching the residual stream can be seen in the Appendix, Fig. A.7. The output of attention heads is only important at the movie position, which we designate as SUM. We label these heads “sentiment summarizers.” Specific attention heads attend to and rely on information written to this token position as well as to ADJ and VRB.

To validate this circuit and the involvement of the sentiment direction, we patched the entirety of the circuit at the ADJ and VRB positions along the sentiment direction only, achieving a 58.3% rate of logit flips and a logit difference drop of 54.8% (in terms of whether a positive or negative word was predicted). Patching the circuit at those positions along all directions resulted in flipping 97% of logits and a logit difference drop of 75%, showing that the sentiment direction is responsible for the majority of the function of the circuit.

> 2We did this to maximise the chances of sentiment tokens occurring at similar positions

> 3We use the term “circuit” as defined by Wang et al. (2022), in the sense of a computational subgraph that is responsible for a significant proportion of the behavior of a neural network on some predefined task.

> 4We note that our patching experiments indicate that there is no causal dependence on the output of other model components at the ADJ and VRB positions–only at the SUM position.

Figure 8: Value-weighted^5 averaged attention to commas and comma phrases in Pythia-2.8b from the top two attention heads writing to the repeated name and “feels” tokens–two key components of the summarization sub-circuit in the ToyMoodStories task. Note that they attend heavily to the relevant comma from both destination positions.

**Multi-subject mood stories in Pythia 2.8b** We next examined the circuit that processes the mood dataset in Pythia-2.8b (the smallest model that could perform the task), which is a more complex task that requires more summarization. As such it presents a better object for study of this motif. We reserve a detailed description of the circuit for the Appendix, but here we observed increasing reliance on summarization, specifically:

- A set of attention heads **attended primarily to the comma** following the preference phrase for the queried subject (e.g. John hates parties,), and secondarily to other words in the phrase, as seen in Figure 8. We observed this phenomenon both with regular attention and value-weighted attention, and found via path patching that **these heads relied partially on the comma token** for their function, as seen in Figure A.9.

- Heads attending to preference phrases (both commas and other tokens) tended to write to the repeated name token near the end of the sentence (John) as well as to the feels token–another type of summarization behavior. Later heads attended to the repeated name and feels tokens with an output important to END.

- 4.2 EXPLORING AND VALIDATING SUMMARIZATION BEHAVIOR IN PUNCTUATION

Our circuit analyses reveal suggestive evidence that summarization behavior at intermediate tokens like commas, periods and certain nouns plays an important part in sentiment processing, despite these tokens having no inherent valence. We focus on summarization at commas and periods and explore this further in a series of ablation and patching experiments. We find that in many cases this summarization results in a partial information bottleneck, in which the summarization points become as important (or sometimes more important) than the phrases that precede them for sentiment tasks.

**Summarization information is comparably important as original semantic information** In order to determine the extent of the information bottleneck presented by commas in sentiment processing, we tested the model’s performance on the multi-subject mood stories dataset mentioned above. We froze the model’s attention patterns to ensure the model used the information from the patched commas in exactly the same way as it would have used the original information. Without this step, the model could simply avoid attending to the commas. We then performed activation patching on either the precomma phrases (e.g., patching “John hates parties,” with “John loves parties,”) while freezing the commas so they retain their original, unflipped values; or on the two commas alone, and find a similar drop in the logit difference for both as shown in table 1a.

> 5That is, the attention pattern weighted by the norm of the value vector at each position as per Kobayashi et al. (2020). We favor this over the raw attention pattern as it filters for _significant_ information being moved.

Table 1: Patching results at summary positions

|Intervention|Change in|||
|---|---|---|---|
|| logit difference|Count of irrelevant tokens ff h|Ratio of LD change fidh|
|Patchin full hrase|-75%|aterpreerenceprase|orperos vs. prases|
|g  p values(inclcommas)||0 tokens|0.29|
|.||10 tokens|063|
|Patching pre-comma |-38%| 18 tokens|. 0.92|
|values(freezingcommas)||22 tokens|115|
|Patchingcomma values only|-37%| |. |

(b) Ratio between logit difference change for periods vs. pre-period phrases after patching values

(a) Change in logit difference from intervention on attention head value vectors

**Importance of summarization increases with distance** We also observed that reliance on summarization tends to increase with greater distances between the preference phrases and the final part of the prompt that would reference them. To test this, we injected irrelevant text^6 after each of the preference phrases in our multi-subject mood stories (after ”John loves parties.” etc.) and measured the ratio between logit difference change for the periods at the end of these phrases vs. pre-period phrases, with higher values indicating more reliance on period summaries (Table 1b). We found that the periods can be up to 15% **more** important than the actual phrases as this distance grows. Although these results are only a first step in assessing the importance of summarization importance relative to prompt length, our findings suggest that this motif may only increase in relative importance as models grow in context length, and thus merits further study.

### 4.3 Validating summarization behavior in sst

In order to study more rigorously how summarization behaves with natural text, we examined this phenomenon in SST. We appended the suffix “Review Sentiment:” to each of the prompts and evaluate Pythia-2.8b on zero-shot classification according to whether positive or negative have higher probability and are in the top 10 tokens predicted. We then take the subset of examples Pythia-2.8b succeeds on that have at least one comma, which means we start with a baseline of 100% accuracy. We performed ablation and patching experiments on comma representations. If comma representations do not summarize sentiment information, then our experiments should not damage the model’s abilities. However, our results reveal a clear summarization motif for SST.

**Ablation baselines** We performed two baseline experiments in order to obtain a control for our later experiments. First to measure the total effect of the sentiment directions, we performed directional ablation (as described in 2.3) using the sentiment directions found with DAS to every token at every layer, resulting in a 71% reduction in the logit difference and a 38% drop in accuracy (to 62% ). Second, we performed directional ablation on all tokens with a small set of random directions, resulting in a _<_ 1% change to the same metrics.

**Directional ablation at all comma positions** We then performed directional ablation–using the DAS (2.2) sentiment direction–to every comma in each prompt, regardless of position, resulting in an 18% drop in the logit difference and an 18% drop in zero-shot classification accuracy–indicating that nearly 50% of the model’s sentiment-direction-mediated ability to perform the task accurately was mediated via sentiment information at the commas. We find this particularly significant because we did not take any special effort to ensure that commas were placed at the end of sentiment phrases.

**Mean-ablation at all comma positions** Separately from the above, we performed mean ablation at all comma positions as in 2.3, replacing each comma activation vector with the mean comma activation from the entire dataset in a layerwise fashion. Note that this changes the entire activation on the comma token, not just the activation in the sentiment direction. This resulted in a 17% drop in logit difference and an accuracy drop of 19% .

> 6E.g. “John loves parties. _He has a red hat and wears it everywhere, especially when he is riding his bicycle through the city streets._ Mark hates parties. _He has a purple hat but only wears it on Sundays, when he takes his weekly walk around the lake._ One day, they were invited to a grand gala. John feels very”

### 4.4 The big picture of summarization

We have identified a phenomenon across multiple models and tasks where sentiment information is not directly transferred from valenced tokens to the final output but is first aggregated at intermediate, non-valenced tokens like commas and periods (and sometimes noun tokens for specific referents). We call this behavior the “summarization motif.” These summarization points serve as partial information bottlenecks and are causally significant for the model’s performance on sentiment tasks. Through a series of ablation and patching experiments, we have validated the importance of this summarization behavior in both toy tasks and real-world datasets like the Stanford Sentiment Treebank. Additional findings suggest that as models grow in context length, the importance of this internal summarization behavior may increase–a subject that warrants further investigation. Overall, the discovery of this summarization behavior adds a new layer of complexity to our understanding of how sentiment is processed and represented in LLMs, and seems likely to be an important part of how LLMs create internal world representations.

## 5 Related work

**Sentiment Analysis** Understanding the emotional valence in text data is one of the first NLP tasks to be revolutionized by deep learning (Socher et al., 2013) and remains a popular task for benchmarking NLP models (Rosenthal et al., 2017; Nakov et al., 2016; Potts et al., 2021; Abraham et al., 2022). For a review of the literature, see (Pang & Lee, 2008; Liu, 2012; Grimes, 2014).

**Understanding Internal Representations** This research was inspired by the field of Mechanistic Interpretability, an agenda which aims to reverse-engineer the learned algorithms inside models (Olah et al., 2020; Elhage et al., 2021b; Nanda et al., 2023a). Exploring representations (Section 3) and world-modelling behavior inside transformers has garnered significant recent interest. This was studied in the context of synthetic game-playing models by Li et al. (2023) and evidence of linearity was demonstrated by Nanda (2023b) in the same context. Other work studying examples of world-modelling inside neural networks includes Li et al. (2021); Patel & Pavlick (2022); Abdou et al. (2021). Another framing of a very similar line of inquiry is the search for latent knowledge (Christiano et al., 2021; Burns et al., 2022). Prior to the transformer, representations of emotion were studied in Goh et al. (2021) and sentiment was studied by Radford et al. (2017), notably, the latter finding a sentiment neuron which implies a linear representation of sentiment. A linear representation of truth in LLMs was found by Marks & Tegmark (2023).

**Summarization Motif** Our study of the Summarization motif (Section 4) follows from the search for information bottlenecks in models (Li et al. (2021)). Our use of the word ‘motif’, in the style of Olah et al. (2020), is originally inspired from systems biology (Alon, 2006). The idea of exploring representations at different frequencies or levels of abstraction was explored further in Tamkin et al. (2020). Information storage after the relevant token was observed in how GPT2-small predicts gender (Mathwin et al.).

**Causal Interventions in Language Models** We approach our experiments from a causal mediation analysis perspective. Our approach to identifying computational subgraphs that utilize feature representations as inspired by the ‘circuits analysis’ framework (Stefan Heimersheim, 2023; Varma et al., 2023; Hanna et al., 2023), especially the tools of mean ablation and activation patching (Vig et al., 2020; Geiger et al., 2021; 2023a; Meng et al., 2023; Wang et al., 2022; Conmy et al., 2023; Chan et al., 2023; Cohen et al., 2023). We use Distributed Alignment Search (Geiger et al., 2023b) in order to apply these ideas to specific subspaces.

## 6 Conclusion

The two central novel findings of this research are the existence of a linear representation of sentiment and the use of summarization to store sentiment information. We have seen that the sentiment direction is causal and central to the circuitry of sentiment processing. Remarkably, this direction is so stark in the residual stream space that it can be found even with the most basic methods and on a tiny toy dataset, yet generalise to diverse natural language datasets from the real-world. Summarization is a motif present in larger models with longer context lengths and greater proficiency in zero-shot classification. These summaries present a tantalising glimpse into the world-modelling behavior of transformers.

We also see this research as a model for how to find and study the representation of a particular feature. Whereas in dictionary learning (Bricken et al., 2023) we enumerate a large set of features which we then need to interpret, here we start with an interpretable feature and subsequently verify that a representation of this feature exists in the model, analogously to Zou et al. (2023). One advantage of this is that our fitting process is much more efficient: we can use toy datasets and very simple fitting methods. It is therefore very encouraging to see that the results of this process generalise well to the full data distribution, and indeed we focus on providing a variety of experiments to strengthen the case for the existence of our hypothesised direction.

**Limitations** Did we find a truly universal sentiment direction, or merely the first principal component of directions used across different sentiment tasks? As found by Bricken et al. (2023), we suspect that this feature could be “split” further into more specific sentiment features. Similarly, one might wonder if there is really a single bipolar sentiment direction or if we have simply found the difference between a “positive” and a “negative” sentiment direction. It turns out that this distinction is not well-defined, given that we find empirically that there is a direction corresponding to “valenced words”. Indeed, if **_x_** is the valence direction and **_y_** is the sentiment direction, then **_p_** = **_x_** + **_y_** represents positive sentiment and **_n_** = **_x_** _−_ **_y_** is the negative direction. Conversely, we can reframe as starting from the positive/negative directions **_p_** and **_n_** , and then rederive **_x_** =^(**_p_**)^(+) 2^**_n_** and **_y_** :=^(**_p_**)^(_−_) 2^(**_n_**.)

Many of our casual abstractions do not explain 100% of sentiment task performance. There is likely circuitry we’ve missed, possibly as a result of distributed representations or superposition (Elhage et al., 2022) across components and layers. This may also be a result of self-repair behavior (Wang et al., 2022; McGrath et al., 2023). Patching experiments conducted on more diverse sentence structures could also help to better isolate the circuitry for sentiment from more task-specific machinery.

The use of small datasets versus many hyperparameters and metrics poses a constant risk of gaming our own measures. Our results on the larger and more diverse SST dataset, and the consistent results across a range of models help us to be more confident in our results.

Distributed Alignment Search (DAS) outperformed on most of our metrics but presents possible dangers of overfitting to a particular dataset and taking the activations out of distribution (Lange et al., 2023). We include simpler tools such as Logistic Regression as a sanity check on our findings. Ideally, we would love to see a set of best practices to avoid such illusions.

**Implications and future work** The summarization motif emerged naturally during our investigation of sentiment, but we would be very interested to study it in a broader range of contexts and understand what other factors of a particular model or task may influence the use of summarization. When studying the circuitry of sentiment, we focused almost exclusively on attention heads rather than MLPs. However, early results suggest that further investigation of the role of MLPs and individual neurons is likely to yield interesting results (A.5).

Finally, we see the long-term goal of this line of research as being able to help detect dangerous computation in language models such as _deception_ . Even if the existence of a single “deception direction” in activation space seems a bit naive to postulate, hopefully in the future many of the tools developed here will help to detect representations of deception or of knowledge that the model is concealing, helping to prevent possible harms from LLMs.

## Author contributions

Oskar and Curt made equal contributions to this paper. Curt’s focus was on circuit analysis and he discovered the summarization motif, leading to Section 4. Oskar was focused on investigating the direction and eventually conducted enough independent experiments to convince us that the direction was causally meaningful, leading to Section 3. Neel was our mentor as part of SERI MATS, he suggested the initial project brief and provided considerable mentorship during the research. He also did the neuron analysis in Section A.5. Atticus acted a secondary source of mentorship and guidance. His advice was particularly useful as someone with more of a background in causal mediation analysis. He suggested the use of Stanford Sentiment Treebank and the discrete accuracy metric.

## Acknowledgments

SERI MATS provided funding, lodging and office space for 2 months in Berkeley, California. The transformer-lens package (Nanda & Bloom, 2022) was indispensable for this research. We are very grateful to Alex Tamkin for his extensive feedback. Other valuable feedback came from Georg Lange, Alex Makelov and Bilal Chughtai. Atticus Geiger is supported by a grant from Open Philanthropy.

## Reproducibility statement

To facilitate reproducibility of the results presented in this paper, we have provided detailed descriptions of the datasets, models, training procedures, algorithms, and analysis techniques used. The ToyMovieReview dataset is fully specified in Section A.7. We use publicly available models including GPT-2 and Pythia, with details on the specific sizes provided in Section 2.1. The methods for finding sentiment directions are described in full in Section 2.2. Our causal analysis techniques of activation patching, ablation, and directional patching are presented in Section 2.3. Circuit analysis details are extensively covered for two examples in Appendix Section A.3. The code for data generation, model training, and analyses is available here.

## References

- Mostafa Abdou, Artur Kulmizev, Daniel Hershcovich, Stella Frank, Ellie Pavlick, and Anders Søgaard. Can language models encode perceptual structure without grounding? a case study in color, 2021.

- Eldar David Abraham, Karel D’Oosterlinck, Amir Feder, Yair Gat, Atticus Geiger, Christopher Potts, Roi Reichart, and Zhengxuan Wu. CEBaB: Estimating the causal effects of real-world concepts on nlp model behavior. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), _Advances in Neural Information Processing Systems_ , volume 35, pp. 17582–17596. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/ file/701ec28790b29a5bc33832b7bdc4c3b6-Paper-Conference.pdf.

- Uri Alon. _An Introduction to Systems Biology: Design Principles of Biological Circuits_ . Chapman and Hall/CRC, 1st edition, 2006. doi: 10.1201/9781420011432. URL https://doi.org/ 10.1201/9781420011432.

- Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for analyzing large language models across training and scaling, 2023.

- Tolga Bolukbasi, Adam Pearce, Ann Yuan, Andy Coenen, Emily Reif, Fernanda Vi´egas, and Martin Wattenberg. An interpretability illusion for bert, 2021.

- Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. _Transformer Circuits Thread_ , 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html.

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020.

- S´ebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with gpt-4, 2023.

- Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision, 2022.

- Lawrence Chan, Adri`a Garriga-Alonso, Nicholas Goldowsky-Dill, Ryan Greenblatt, Jenny Nitishinskaya, Ansh Radhakrishnan, Buck Shlegeris, and Nate Thomas. Causal scrubbing: a method for rigorously testing interpretability hypotheses [redwood research]. Alignment Forum, 2023. URL https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/ causal-scrubbing-a-method-for-rigorously-testing. Accessed: 17th Sep 2023.

- Paul Christiano, Ajeya Cotra, and Mark Xu. Eliciting latent knowledge: How to tell if your eyes deceive you. Google Docs, December 2021. Accessed: 17th Sep 2023.

- Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, and Mor Geva. Evaluating the ripple effects of knowledge editing in language models, 2023.

- Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adri`a Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability, 2023.

- J. Cui, Z. Wang, SB. Ho, et al. Survey on sentiment analysis: evolution of research methods and topics. _Artif Intell Rev_ , 56:8469–8510, 2023. doi: 10.1007/s10462-022-10386-z. URL https: //doi.org/10.1007/s10462-022-10386-z.

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018.

- Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021a. https://transformer-circuits.pub/2021/framework/index.html.

- Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021b. https://transformer-circuits.pub/2021/framework/index.html.

- Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. _Transformer Circuits Thread_ , 2022. URL https://transformer-circuits. pub/2022/toy_model/index.html.

- Atticus Geiger, Kyle Richardson, and Christopher Potts. Neural natural language inference models partially embed theories of lexical entailment and negation. In _Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP_ , pp. 163–173, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.blackboxnlp-1.16. URL https://www.aclweb.org/anthology/ 2020.blackboxnlp-1.16.

- Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. In _Advances in Neural Information Processing Systems_ , volume 34, pp. 9574–9586, 2021. URL https://papers.nips.cc/paper/2021/hash/ 4f5c422f4d49a5a807eda27434231040-Abstract.html.

- Atticus Geiger, Zhengxuan Wu, Hanson Lu, Josh Rozner, Elisa Kreiss, Thomas Icard, Noah Goodman, and Christopher Potts. Inducing causal structure for interpretable neural networks. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 7324–7338. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/geiger22a.html.

- Atticus Geiger, Christopher Potts, and Thomas Icard. Causal abstraction for faithful model interpretation. Ms., Stanford University, 2023a. URL https://arxiv.org/abs/2301.04709.

- Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah D. Goodman. Finding alignments between interpretable causal variables and distributed neural representations, 2023b.

- Gabriel Goh, Nick Cammarata †, Chelsea Voss †, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal neurons in artificial neural networks. _Distill_ , 2021. doi: 10.23915/distill.00030. https://distill.pub/2021/multimodal-neurons.

- Aaron Gokaslan and Vanya Cohen. Openwebtext corpus, 2019. URL http://Skylion007. github.io/OpenWebTextCorpus.

- Seth Grimes. Text analytics 2014: User perspectives on solutions and providers. Technical report, Alta Plana, July 2014. URL http://altaplana.com/TextAnalytics2014.pdf.

- Michael Hanna, Ollie Liu, and Alexandre Variengien. How does gpt-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model, 2023.

- Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. Attention is not only a weight: Analyzing transformers with vector norms, 2020.

- Georg Lange, Alex Makelov, and Neel Nanda. An interpretability illusion for activation patching of arbitrary subspaces. LessWrong, 2023. URL https://www.lesswrong.com/posts/RFtkRXHebkwxygDe2/ an-interpretability-illusion-for-activation-patching-of.

- Belinda Z. Li, Maxwell Nye, and Jacob Andreas. Implicit representations of meaning in neural language models, 2021.

- Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task, 2023.

- Bing Liu. Sentiment analysis and opinion mining. _Synthesis Lectures on Human Language Technologies_ , 5(1):1–167, May 2012. doi: 10.2200/s00416ed1v01y201204hlt016. URL http: //dx.doi.org/10.2200/S00416ED1V01Y201204HLT016.

- Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2023.

- Chris Mathwin, Guillaume Corlouer, Esben Kran, Fazl Barez, and Neel Nanda. Identifying a preliminary circuit for predicting gendered pronouns in gpt-2 small. URL https://itch.io/ jam/mechint/rate/1889871.

- Thomas McGrath, Matthew Rahtz, Janos Kramar, Vladimir Mikulik, and Shane Legg. The hydra effect: Emergent self-repair in language model computations, 2023.

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt, 2023.

- Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic regularities in continuous space word representations. In _Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 746– 751, Atlanta, Georgia, June 2013. Association for Computational Linguistics. URL https: //aclanthology.org/N13-1090.

- Preslav Nakov, Alan Ritter, Sara Rosenthal, Fabrizio Sebastiani, and Veselin Stoyanov. SemEval2016 task 4: Sentiment analysis in Twitter. In _Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016)_ , pp. 1–18, San Diego, California, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/S16-1001. URL https:// aclanthology.org/S16-1001.

- Neel Nanda. Neuroscope: A website for mechanistic interpretability of language models, 2023a. URL https://neuroscope.io.

- Neel Nanda. Actually, othello-gpt has a linear emergent world model, Mar 2023b. URL <https: //neelnanda.io/mechanistic-interpretability/othello>.

- Neel Nanda and Joseph Bloom. Transformerlens. https://github.com/neelnanda-io/ TransformerLens, 2022.

- Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability, 2023a.

- Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models, 2023b.

- nostalgebraist. interpreting gpt: the logit lens, 2020. URL https://www.lesswrong.com/ posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.

- Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. _Distill_ , 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in.

- Bob Pang and Lillian Lee. Opinion mining and sentiment analysis. _Foundations and Trends in Information Retrieval_ , 2(1-2):1–135, 2008. doi: 10.1561/1500000001. URL http://www.cs. cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey. html.

- Roma Patel and Ellie Pavlick. Mapping language models to grounded conceptual spaces. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/ forum?id=gJcEM8sxHK.

- Judea Pearl. Direct and indirect effects. In _Probabilistic and causal inference: the works of Judea Pearl_ , pp. 373–392. Association for Computing Machinery, 2022.

- Christopher Potts, Zhengxuan Wu, Atticus Geiger, and Douwe Kiela. DynaSent: A dynamic benchmark for sentiment analysis. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pp. 2388–2404, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.186. URL https: //aclanthology.org/2021.acl-long.186.

- Alec Radford, Rafal Jozefowicz, and Ilya Sutskever. Learning to generate reviews and discovering sentiment, 2017.

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019.

- Sara Rosenthal, Noura Farra, and Preslav Nakov. SemEval-2017 task 4: Sentiment analysis in Twitter. In _Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)_ , pp. 502–518, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/S17-2088. URL https://aclanthology.org/S17-2088.

- Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In _Proceedings of the 2013 conference on empirical methods in natural language processing_ , pp. 1631–1642, 2013.

- Jett Janiak Stefan Heimersheim. A circuit for python docstrings in a 4-layer attention-only, 2023. URL https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn/ a-circuit-for-python-docstrings-in-a-4-layer-attention-only. Accessed: 2023-09-22.

- Alex Tamkin, Dan Jurafsky, and Noah Goodman. Language through a prism: A spectral approach for multiscale language representations, 2020.

- Alexander Matt Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. Activation addition: Steering language models without optimization, 2023.

- Vikrant Varma, Rohin Shah, Zachary Kenton, J´anos Kram´ar, and Ramana Kumar. Explaining grokking through circuit efficiency, 2023.

- Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Simas Sakenis, Jason Huang, Yaron Singer, and Stuart Shieber. Causal mediation analysis for interpreting neural nlp: The case of gender bias, 2020.

- Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small, 2022.

- Adam Yedidia. Residual viewer, 2023. URL http://ec2-34-192-101-140.compute-1. amazonaws.com:5014/. Available at: http://ec2-34-192-101-140. compute-1.amazonaws.com:5014/.

- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to ai transparency, 2023.

### (a) PCA on adjectives in and out of sample

(b) PCA on in-sample adjectives and out-of-sample verbs

Figure A.1: 2-D PCA visualization of the embedding for a handful of adjectives and verbs (GPT2small)

### Aappendix

- A.1 FURTHER EVIDENCE FOR A LINEAR SENTIMENT REPRESENTATION

#### A.1.1 clustering

In Section 2.2, we outline just a few of the many possible techniques for determining a direction which hopefully corresponds to sentiment. Is it overly optimistic to presume the existence of such a direction? The most basic requirement for such a direction to exist is that the residual stream space is clustered. We confirm this in two different ways.

First we fit 2-D PCA to the token embeddings for a set of 30 positive and 30 negative adjectives. In Figure A.1, we see that the positive adjectives (blue dots) are very well clustered compared to the negative adjectives (red dots). Moreover, we see that sentiment words which are out-of-sample with respect to the PCA (squares) also fit naturally into their appropriate color. This applies not just for unseen adjectives (Figure A.1a) but also for verbs, an entirely out-of-distribution class of word (Figure A.1b).

Secondly, we evaluate the accuracy of 2-means trained on the Simple Movie Review Continuation adjectives (Section 2.1). The fact that we can classify in-sample is not very strong evidence, but we verify that we can also classify out-of-sample with respect to the _K_ -means fitting process. Indeed, even on hold-out adjectives and on the verb tokens (which are totally out of distribution), we find that the accuracy is generally very strong across models. We also evaluate on a fully out of distribution toy dataset (“simple adverbs”) of the form “The traveller [adverb] walked to their destination. The traveller felt very”. The results can be found in Figure A.2. This is strongly suggestive that we are stumbling on a genuine representation of sentiment.

#### A.1.2 activation addition

We perform activation addition (Turner et al., 2023) on GPT2-small for a single positive simple movie review continuation prompt (from Section 2.1) in order to flip the generated outputs from negative to prompt. The “steering coefficient” is the multiple of the sentiment direction which we add to the first layer residual stream. The outputs are extremely negative by the time we reach coefficient -17 and we observe a gradual transition for intermediate coefficients (Figure A.3).

#### A.1.3 multi-lingual sentiment

We use the first few paragraphs of Harry Potter in English and French as a standard text (Elhage et al., 2021b). We find that intermediate layers of pythia-2.8b demonstrate intuitive sentiment activations for the French text (Figure A.4). It is important to note that none of the models are very good at

French, but this was the smallest model where we saw hints of generalisation to other languages. The representation was not evident in the first couple of layers, probably due to the poor tokenization of French words.

#### A.1.4 interpretability of negations

We visualise the sentiment activations for all 12 layers of GPT2-small simultaneously on the prompt “You never fail. Don’t doubt it. I am not uncertain” (Figure A.5). This allows us to observe how fail, doubt and uncertain shift from negative to positive sentiment during the forward pass of the model.

### A.2 is sentiment really ahyperplane?

In our directional patching experiments, we have somewhat artificially selected just 1 dimension as our hypothesised structure for the sentiment subspace. We can perform DAS with any number of dimensions. Figure A.6 demonstrates that whilst increasing the DAS dimension improves the patching metric in-sample (A.6a), the metric does not improve out-of-distribution (A.6b).

### A.3 detailed circuit analysis

In order to build a picture of each circuit, we used the process pioneered in Wang et al. (2022):

- Identify which model components have the greatest impact on the logit difference when path patching is applied (with the final result of the residual stream set as the receiver).

- Examine the attention patterns (value-weighted, in some cases) and other behaviors of these components (in practice, attention heads) in order to get a rough idea of what function they are performing.

- Perform path-patching using these heads (or a distinct cluster of them) as receivers.

- Repeat the process recursively, performing contextual analyses of each “level” of attention heads in order to understand what they are doing, and continuing to trace the circuit backwards.

In each path-patching experiment, change in logit difference is used as the patching metric. We started with GPT-2 as an example of a classic LLM displays a wide range of behaviors of interest, and moved to larger models when necessary for the task we wanted to study (choosing, in each case, the smallest model that could do the task).

A.3.1 SIMPLE SENTIMENT - GPT-2 SMALL

We examined the circuit performing tasks for the following sentence template:

I thought this movie was ADJECTIVE, I VERBed it. Conclusion: This movie is

Using a threshold of 5%-or-greater damage to the logit difference for our patching experiments, we found that GPT-2 Small contained 4 primary heads contributing to the most proximate level of circuit function–10.4, 9.2, 10.1, and 8.5 (using “layer.head” notation). Examining their value-weighted attention patterns, we found that attention to ADJ and VRB in the sentence was most prominent in the first three heads, but 8.5 attended primarily to the second “movie” token. We also observed that 9.2 attended to this token as well as to ADJ. (Results of activation patching can be seen in Fig. A.7.)

Conducting path-patching with 8.5 and 9.2 as receivers, we identified two heads–7.1 and 7.5–that primarily attend to ADJ and VRB from the “movie” token. We further determined that the output of these heads, when path-patched through 9.2 and 8.5 as receivers, was causally important to the circuit (with patching causing a logit difference shift of 7% and 4% respectively for 7.1 and 7.5). This was not the case for other token positions, which demonstrates that causally relevant information is indeed being specially written to the “movie” position. We thus designated it the SUM token in this circuit, and we label 8.5 a summary-reader head.

Repeating our analysis with lower thresholds yielded more heads with the same behavior but weaker effect sizes, adding 9.10, 11.9, and 6.4 as summary reader, direct sentiment reader, and sentiment summarizer respectively. This gives a total of 9 heads making up the circuit.

#### A.3.2 multi-subject mood stories circuit - pythia 2.8b

We also examined the circuit for this sentence template: Carl hates parties, and avoids them whenever possible. Jack loves parties, and joins them whenever possible. One day, they were invited to a grand gala. Jack feels very [excited/nervous]. We did not attempt to reverse-engineer the entire circuit, but examined it from the perspective of what matters causally for sentiment processing– especially determining to what extent summarization occurred.

Following the same process as with GPT-2 with preference/sentiment-flipped prompts (that is, taking _xorig_ to be “John hates parties,... Mary loves parties,” and _xflipped_ to be “John loves parties,... Mary hates parties”), we initially identified 5 key heads that were most causally important to the logit difference at END: 17.19, 22.5, 14.4, 20.10, and 12.2 (in “layer.head” notation). Examining the value-weighted attention patterns, we observed that the top token receiving attention from END was always the repeated name RNAME (e.g., “John” in “John feels very”) or the “feels” token FEEL, indicating that some summarization may have taken place there.

We also observed that the top token attended to from RNAME and FEEL was in fact the comma at the end of the queried preference phrase (that is, the comma at the end of “John hates parties”). We designate this position COMMASUM.

**Multi-functional heads** Interestingly, we observed that most of these heads were multi-functional: that is, they both attended to COMMASUM from RNAME and FEEL, and also attended to RNAME and FEEL from END, producing output in the direction of the logit difference. This is possible because these heads exist at different layers, and later heads can read the summarized information from previous heads as well as writing their own summary information.

**Direct effect heads** Specifically, the direct effect heads were:

- Head 17.19 did not attend to commas significantly, but did attend to the periods at the end of each preference sentence in addition to its primary attention to RNAME and FEEL, and did not display COMMASUM-reading behavior.

- Head 22.5 attended almost exclusively to FEEL, and did not display COMMASUM-reading behavior.

- Other direct effect heads (14.4, 20.10 and 12.2) did show COMMASUM-reading behavior as well as reading from the near-end tokens to produce output in the direction of the logit difference. In each case, we verified with path-patching that information from these positions was causally relevant.

**Name summary writers** We also found important heads (12.17 being by far the most important) that are only engaged with attending to COMMASUM and producing output at RNAME and FEEL.

**Comma summary writers** We further investigated what circuitry was causally important to task performance mediated through the COMMASUM positions, but did not flesh this out in full detail; after finding initial examples of summarization, we focused on its causal relevance and interaction with the sentiment direction, leaving deeper investigation to future work.

A.4 ADDITIONAL SUMMARIZATION FINDINGS

**Circuitry for processing commas vs. original phrases is semi-separate** Though there is overlap between the attention heads involved in the circuitry for processing sentiment from key phrases and that from summarization points, there are also some clear differences, suggesting that the ability to read summaries could be a specific capability developed by the model (rather than the model simply attending to high-sentiment tokens).

As can be seen in Figure A.8, there are distinct groups of attention heads that result in damage to the logit difference in different situations–that is, some react when phrases are patched, some react disproportionately to comma patching, and one head seems to have a strong response for either patching case. This is suggestive of semi-separate summary-reading circuitry, and we hope future work will result in further insights in this direction.

### A.5 neurons writing to sentiment direction in gpt2-small are interpretable

We observed that the cosine similarities of neuron out-directions with the sentiment direction are extremely heavy tailed (Figure A.10). Thanks to Neuroscope (Nanda, 2023a), we can quickly see whether these neurons are interpretable. Indeed, here are a few examples from the tails of that distribution:

- L3N1605 activates on “hesitate” following a negation

- Neuron L6N828 seems to be activating on words like “however” or “on the other hand” _if_ they follow something negative

- Neuron L5N671 activates on negative words that follow a “not” contraction (e.g. didn’t, doesn’t)

- L6N1237 activates strongly on “but” following “not bad”

We take L3N1605, the “not hesitate” neuron, as an extended example and trace backwards through the network using Direct Logit Attribution^7 . We computed the relative effect of different model components on L3N1605 in the two different cases “I would not hesitate” vs. “I would always hesitate”. The main contributors to this difference are L1H0, L3H10, L3H11 and MLP2. Expanding out MLP2 into individual neurons we find that the contributions to L3N1605 are sparse. For example, L2N1154 activates on words like “don’t”, “not”, “no”, etc. It activates on “not” but not “hesitate” in “I would not hesitate” but activates on “hesitate” in “I would always hesitate”. Visualizing the attention pattern of L1H0 shows that it attends from “hesitate” to the previous token if it is “not”, but not if it is “always”.

These anecdotal examples suggest at a complex network of machinery for transmitting sentiment information across components of the network using a single critical axis of the residual stream as a communication channel. We think that exploring these neurons further could be a very interesting avenue of future research, particularly for understanding how the model updates sentiment based on negations where these neurons seem to play a critical role.

A.6 DETAILED DESCRIPTION OF METRICS

- **Logit Difference:** We extend the logit difference metric used by Wang et al. (2022) to the setting with 2 _classes_ of next token rather than only 2 valid next tokens. This is useful in situations where there are many possible choices of positively or negatively valenced next tokens. Specifically, we examine the average difference in logits between sets of positive/negative next-tokens _T_^positive = _{t_^positive _i_ : 1 _≤ i ≤ n}_ and _T_^negative = _{t_^negative _i_ : 1 _≤ i ≤ n}_ in order to get a smooth measure of the model’s ability to differentiate between sentiment. That is, we define the logit difference as _n_^(1)  _i_ logit( _t_^positive _i_ ) _−_ logit( _t_^negative _i_ ). Larger differences indicate more robust separation of the positive/negative tokens, and zero or inverted differences indicate zero or inverted sentiment processing respectively. When used as a patching metric, this demonstrates the causal efficacy of various interventions like activation patching or ablation.^8

- **Logit Flip:** Similar to logit difference, this is the percentage of cases where the logit difference between _T_^positive and _T_^negative is inverted after a causal intervention. This is a more discrete measure which is helpful for gauging whether the magnitude of the logit differences is sufficient to actually flip model predictions.

- **Accuracy:** Out of a set of prompts, the percentage for which the logits for tokens _T_^correct are greater than _T_^incorrect . In practice, usually each of these sets only has one member (e.g., “Positive” and “Negative”).

> 7This technique decomposes model outputs into the sum of contributions of each component, using the insight from Elhage et al. (2021b) that components are independent and additive

8We use this metric often because it is more sensitive than accuracy to small shifts in model behavior, which is particularly useful for circuit identification where the effect size is small but real. That is, in many cases a token of interest might become much more likely but not cross the threshold to change accuracy metrics, and in this case logit difference will detect it. Logit difference is also useful when trying to measure the model behavior transition between two different, opposing prompts–in this case, the logit difference for each of the prompts is used for lower and upper baselines, and we can measure the degree to which the logit difference behavior moves from one pole to the other.

### A.7 toy dataset details

The ToyMovieReview dataset consists of prompts of the form ”I thought this movie was ADJ, I VRB it. [NEWLINE] Conclusion: This movie is”. We substituted different adjective and verb tokens into the two variable placeholders to create a prompt for each distinct adjective. We averaged the logit difference across 5 positive and 5 negative completions to determine whether the continuation was positive or negative.

positive_adjectives_train:

- perfect

- fantastic

- delightful

- cheerful

- good

- remarkable

- satisfactory

- wonderful

- nice

- fabulous

- outstanding

- satisfying

- awesome

- exceptional

- adequate

- incredible

- extraordinary

- amazing

- decent

- lovely

- brilliant

- charming

- terrific

- superb

- spectacular

- great

- splendid

- beautiful

- positive

- excellent

- pleasant negative_adjectives_train:

- dreadful

- bad

- dull

- depressing

- miserable

- tragic

- nasty

- inferior

- horrific

- terrible

- ugly

- disgusting

- disastrous

- annoying

- boring

- offensive

- frustrating

- wretched

- inadequate

- dire

- unpleasant

- horrible

- disappointing

- awful positive_adjectives_test:

- stunning

- impressive

- admirable

- phenomenal

- radiant

- glorious

- magical

- pleasing

- lively

- warm

- strong

- helpful

- vivid

- modern

- crisp

- sweet

- negative_adjectives_test:

- foul

- vile

- appalling

- rotten

- grim

- dismal

- lazy

- poor

- rough

- noisy

- sour

- flat

- ancient

- bitter

- positive_verbs: - enjoyed

- loved

- liked

- appreciated

- admired negative_verbs:

- hated

- disliked

- despised positive_answer_tokens:

- great

- amazing

- awesome

- good

- perfect negative_answer_tokens:

- terrible

- awful

- bad

- horrible

- disgusting

The ToyMoodStories dataset consists of prompts of the form “NAME1 VRB1.1 parties, and VRB1.2 them whenever possible. NAME2 VRB2.1 parties, and VRB2.2 them whenever possible. One day, they were invited to a grand gala. QUERYNAME feels very”. To evaluate the model’s output, we measure the logit difference between the “excited” and “nervous” tokens.

VRB1.1 and VRB2.1 are always one of:

- hates

- loves and VRB1.2 and VRB2.2 are always one of:

- avoids

- joins

In each case, the two verbs in each sentence will agree in sentiment, and the sentence with NAME1 will always have opposite sentiment to that of NAME2.

Names are sampled from the following list:

- John

- Anne

- Mark

- Mary

- Peter

- Paul

- James

- Sarah

- Mike

- Tom

- Carl

- Sam

- Jack

Each combination of NAME1, NAME2, QUERYNAME are included in the dataset (where half the time QUERYNAME matches the first name, and half the time it matches the second). Where necessary for computational tractability, we take a subsample of the first 16 items of this dataset.

A.8 GLOSSARY

#### Glossary

- **activation addition** Formerly called “activation steering”, a technique from Turner et al. (2023) where a vector is added to the residual stream at a certain position (or all positions) and layer during each forward pass while generating sentence completions. In our case, the vector is the sentiment direction.

- **activation patching** A technique introduced in Meng et al. (2023), under the name ‘causal tracing’, which uses a causal intervention to identify which activations in a model matter for producing some output. It runs the model on some ‘clean’ input, replaces (patches) an activation with that same activation on ‘flipped’ input, and sees how much that shifts the output from ‘clean’ to ‘flipped’.

#### activation steering See activation addition

- **DAS** Distributed Alignment Search (Geiger et al., 2023b) uses gradient descent to train a rotation matrix representating an orthonormal change of basis to one better aligned with the model’s features. We mostly focus on a special case of finding a singular critical direction, where we patch along the first dimension of the rotated basis and then use a smooth patching metric (such as the logit difference between positive and negative completions) as the objective to be minimised.

- **directional activation patching** A variant of activation patching introduced in this paper where we only patch a single dimension from a counterfactual activation. That is, for prompts _x_ orig and _x_ new, direction **_d_** , a set of model components C, we run a forward pass on _x_ orig but for each component in C, we patch/replace the output **_o_** orig with **_o_** orig _−_ **_o_** orig _·_ **_d_** + **_o_** new _·_ **_d_** . This is equivalent to activation patching a single neuron, but done in a rotated basis (where **_d_** is the first column of the rotation matrix).

- **directional patching** See directional activation patching.

- **mean ablation** A type of ablation method, where we seek to eliminate the contribution of a particular component to demonstrate its importance, where we replace a particular set of activations with their mean over an appropriate dataset.

- **patching metric** A summary statistic used to quantify the results of an activation patching experiment. By default here we use the percentage change in logit difference as in Wang et al. (2022).

- **SST** Stanford Sentiment Treebank is a labelled sentiment dataset from Socher et al. (2013) described in Section 2.1.

Figure A.2: 2-me ~~ans clas~~ ~~sif~~ i ~~cation accuracy fo~~ r various GPT-2 siz ~~es, split~~ ~~by layer (showing up~~ to 24 layers)

Proportion of Sentiment by Steering Coefficient

Figure A.3: Area plot of sentiment labels for generated outputs by activation steering coefficient, starting from a single positive movie review continuation prompt. Activation addition (Turner et al., 2023) was performed in GPT2-small’s first residual stream layer. Classification was performed by GPT-4.

#### <|endoftext|>

Mr. and Mrs . Dursley, of number four , Priv et Drive , were proud to say that they were perfectly normal , thank you very much . They were the last people you 'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense.

Mr. Dursley was the director of a firm called Gr unnings , which made drills . He was a big , beef y man with hardly any neck, although h e did have a very large mustache. Mrs . Dursle y was thin and blonde and had nearly twice the usual amount of neck , which came in very useful as she spent so much of her time cr aning over garden f ences , spying on the neighbors . The Durs leys had a small son called Dud ley and in their opinion there was no finer boy anywhere .

The Durs leys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn 't think they could bear it if anyone found out about the Potters . Mrs . Potter was Mrs . Dursley's sister , but they hadn 't met for several years ; in fact , Mrs . Dursley pretended she didn 't have a sister , because her sister and her good -for- nothing husband were as unDursleyish as it was possible to be. The Durs leys shuddered to think what the neighbors would say if the Potters arrived in the street . The Durs leys knew that the Potters had a small son , too , but they had never even seen him . This boy was another good reason for keeping the Potters away ; they didn 't want Dud ley mixing with a child like that .

When Mr . and Mrs . Dursley woke up on the dull , gray Tuesday our story starts , there was nothing about the cloud y sky outside to suggest that strange and mysterious things would soon be happening all over the country . Mr . Dursley hummed as he picked out his most boring tie for work , and Mrs . Dursley gossip ed away happily as she wrest led a screaming Dud ley into his high chair .

#### (a) First 4 paragraphs of Harry Potter in English

#### <|endoftext|>

Mr e t Mrs Dursley, qui hab it aient au 4 , Privet Drive, av aient toujours affirmé avec l a plus grande fierté qu'ils é taient par fait ement normaux , merc i pour eux. Jama is qu iconque n'aurait imagin é qu'ils puissent se trou ver impl iqués dans quoi que ce soit d'étr ange ou de mystérie ux. Ils n' avaient pas de temps à perd re avec des sorn ettes .

Mr Dursley dirigeait la Gr unnings , une entreprise qui fab riquait des perceuses . C' était un hom me grand et mass if, qui n'avait pratiqu ement pas de cou , mais poss édait en revanche une m oustache de bel le tail le. Mrs Dursley , quant à elle, était min ce et blonde e t disposa it d'un cou deux fois plus long que la mo yenne , ce qui lu i était fort utile pour esp ionner ses voisins en regard ant par - dess us les clôt ures des jardins. Les Dursley av aient un pet it gar çon prénommé Dud ley et c' était à leurs yeux le plus bel enf ant du monde .

Les Dursley av aient tout ce qu'ils voul aient . La seule chose ind és irable qu'ils possé da ient , c' était un secret dont ils cra ign aient plus que tout qu'on le dé cou vre un jour . Si jamais qu iconque ven ait à entend re par ler des Potter , ils é taient conv ain cus qu'ils ne s'en rem ettra ient pas . Mrs Potter était la soeur de Mrs Dursley, mais toutes deux ne s' étaient plus rev ue s depuis des anné es . En fait, Mrs Dursley faisait comme si elle était fille unique , car sa soeur et son bon à rien de mari é taient aussi éloignés que possible de tout ce qui faisait un Dursley. Les Dursley trem bl aient d'épou vant e à la pens ée de ce que dira ient les voisins si par mal heur les Potter se mont ra ient dans leur rue . Ils sava ient que les Potter , eux aussi , av aient un pet it gar çon, mais ils ne l' avaient jamais vu. Son existence constitu ait une raison supplémentaire de ten ir les Potter à distance : il n' était pas question que le petit Dudl ey se mett e à f réquenter u n enfa nt comme cel ui-là.

Lors que Mr et Mrs Dursley s'éveillè rent , au mat in du m ardi où commence cette histoire , il faisait gris et tr iste et rien dans le ciel nu age ux ne laissait prévoir que des ch oses étr anges et mystérieuses allaient b ient ôt se produ ire dans tout le pays . Mr Dursley fredonnait un air en nou ant sa cravate la plus sin istre pour aller trava iller et Mrs Dursle y raco ntait d'u n ton badin les derniers pot ins du quarti er en s'efforçant d 'install er sur sa chai se de b ébé le je une Dud ley qui bra illait de toute la force de ses poumons .

#### (b) First 3 paragraphs of Harry Potter in French

27 Figure A.4: First paragraphs of Harry Potter in different languages. Model: pythia-2.8b.

Pre-publication draft

|L00 You never fail . Don 't doubt it . I don 't like you .|
|---|
|L01 You never fail . Don 't doubt it . I don 't like you .|
|L02 You never fail . Don 't doubt it . I don 't like you .|
|L03 You never fail . Don 't doubt it . I don 't like you .|
|L04 You never fail . Don 't doubt it . I don 't like you .|
|L05 You never fail . Don 't doubt it . I don 't like you .|
|L06 You never fail . Don 't doubt it . I don 't like you .|
|L07 You never fail . Don 't doubt it . I don 't like you .|
|L08 You never fail . Don 't doubt it . I don 't like you .|
|L09 You never fail . Don 't doubt it . I don 't like you .|
|L10 You never fail . Don 't doubt it . I don 't like you .|
|L11 You never fail . Don 't doubt it . I don 't like you .|
|L12 You never fail . Don 't doubt it . I don 't like you .|

Figure A.5: Visualizing the sentiment activations across layers for a text where the sentiment hinges on negations. Color represents sentiment activation at the given layer and position. Red is negative, blue is positive. Each row is a residual stream layer, first layer is at the top. The three sentences were input as a single prompt, but the pattern was extremely similar using separate prompts. Model: GPT2-small

Figure A.6: DAS sweep over the subspace dimension (GPT2-small). The runs are labelled with the integer _n_ where _d_ DAS = 2^(_n−_1) . Loss is 1 minus the usual patching metric.

Figure A.7: Activation patching results for the GPT-2 Small ToyMovieReview circuit, showing how much of the original logit difference is recaptured when swapping in activations from _xorig_ (when the model is otherwise run on _xflipped_ ). Note that attention output is only important at the SUM position, and that this information is important to task performance at the residual stream layers (8 and 9) in which the summary-readers reside. Other than this, the most important residual stream information lies at the ADJ and VRB positions.

Figure A.8: Logit difference drops by head when commas or pre-comma phrases are patched. Model: pythia-2.8b.

Figure A.9: Path-patching commas and comma phrases in Pythia-2.8b, with attention heads L12H2 and L12H17 writing to repeated name and ”feels” as receivers. Patching the paths between the comma positions and the receiver heads results in the greatest performance drop for these heads.

Similarity of Neuron Out Directions to Sentiment Direction

Figure A.10: Cosine similarity of neuron out-directions and the sentiment direction in GPT2-small
