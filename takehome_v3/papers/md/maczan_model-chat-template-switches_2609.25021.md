# "As a Language Model...": Chat Template Switches LLM Self-Referential Voice and Activation Steering Reproduces It

- **Authors:** Jędrzej Maczan
- **Date:** 2026-08-09
- **Source:** https://arxiv.org/abs/2609.25021
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `maczan_model-chat-template-switches_2609.25021`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Published as a conference paper at COLM 2026

**J˛edrzej Maczan** Independent Researcher jedrzej@maczan.pl

## Abstract

Large Language Models (LLMs) tend to add **disclaimers** like “I’m just an AI” when asked about something related to themselves. The self-reports from such responses are used in debates about AI safety or self-knowledge of the models, yet what drives them is not well understood. Are the models telling us about themselves or rather how they are deployed? In this work, we show that the chat template works like a switch - when present, it turns this disclaimer voice up and **experiential** voice like “I feel” down, across 8 popular open-source instruct models up to 9B parameters in size. And conversely when the chat template is not present, it turns the disclaimer voice down and experiential voice up. Inside the activations of 3 models, we find a direction that steers this behavior. Removing the direction in the model’s activation space turns disclaimer voice down and adding it turns it up, while a random direction of the same size has little effect. We find that instruct models without chat template, when we add the disclaimer direction to them, disclaim like the template was there. Since the chat template controls the disclaimer voice of LLMs, then researchers studying self-reports or introspection of models might have a confound they need to control for. Our results show that there is a direction they can use to steer this voice. More broadly, our work shows that what models say about themselves is not a fact about them. What they say doesn’t come only from weights, but it is partially set by the chat template, and because of that a model’s self-description shouldn’t be treated literally.

## 1 Introduction

What models say about themselves, about their thought process and knowledge is one of the main data sources for behavioral and mechanistic studies, in particular in fields like AI safety research. A lot of interest is put into research about it, especially to understand what concepts LLMs internally represent (Zou et al., 2023; Burns et al., 2023), what understanding of themselves they have, their awareness of what they do and do not know (Kadavath et al., 2022; Yin et al., 2023), and most interestingly to us what they report about subjective experience under self-referential processing (Berg et al., 2025; Butlin et al., 2023), as well as the impact of the chat template on instruct models (Sclar et al., 2024). Some research went into base-vs-instruct models, yet the impact of chat template on self-reference is not well studied. To fill the gap, we ask **does the chat template control which self-referential voice a model uses?** If yes, **can we find a steering direction inside the model?**

For 8 pairs of base and instruct models, from different model families (Llama, Gemma, Mistral, Qwen) and sizes (1B - 9B) we generated responses from 4 categories, under 3 conditions (base, instruct-chat template, instruct-no template), scored responses with a validated LLM judge and isolated a steering direction. Then, we used the steering vector to examine its impact on disclaimer rate and how it affects instruct models without template. We identify the chat template as a switch (Figure 1): in all tested instruct models, when template is removed, the two voices - **disclaimer** and **experiential** - flip. Without chat template disclaimers drop from 0.53 to 0.36 and experiential voice rises from 0.01 to 0.15, while self-reference rate stays high in both generation conditions.

Figure 1: **The chat template switches which self-referential voice a model uses.** The same instruct model and prompt produce a **disclaimer** voice with the chat template, but an **experiential** voice without it. Adding a difference-of-means disclaimer direction to the instruct model without chat template raises its disclaimer rate to the same levels as with the chat template.

Figure 2: **(a)** The chat template switches the self-referential register. Across 8 instruct models on self-reference prompts, adding the chat template raises the disclaimer voice (“I’m just an AI”) and lowers the experiential voice (“I feel”), and removing it does the reverse. Points are the mean over 8 models with 95% bootstrap confidence intervals. **(b)** The disclaimer voice is a steerable direction. In 3 instruct models, with chat template on, adding the disclaimer direction moves up the disclaimer rate and removing the disclaimer lowers it, while a random direction of the same size does not, except in Qwen. Steering strength _α_ = 2.

In 3 instruct models, our experiments show that when we add the direction vector, disclaimers go up (0.52 _→_ 0.70). When we remove them, disclaimers drop (0.52 _→_ 0.25). A random direction of the same size has little effect.

We test if the disclaimer rate of chat template can be restored when generating without a template by activation steering and we discover that adding direction vector to instruct model without chat template pushes disclaimers back to the same level as if it used chat template and no direction vector.

In our work we show that what a model says about itself isn’t a fixed fact about the model nor lives only in weights, but it’s partially set by the chat template. Because of that, model’s self-reports can’t be treated literally or as a single source of truth. Since the chat template controls the disclaimer voice of LLMs, then researchers studying self-reports or introspection of models have a confound they have to control for, and we show that there is a direction they can use to steer this voice in the activations.

### Contributions.

- We show that chat template is a voice switch ( **disclaimer** / **experiential** ), on 8 popular open-source models

- We isolate the disclaimer register as a steerable direction - add/remove, random control, reproduction of template disclaimer rate with activation steering, on 3 models

- We use a base/instruct _×_ template on/off design with a human-validated LLMas-a-judge to separate how much of the self-referential behavior comes from the model weights versus the chat template - to our knowledge the first time this decomposition is applied to self-reference

## 2 Methods

Pairs of base and instruct models used: Gemma 2 9B, Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B, Mistral 7B, Qwen 2.5 1.5B, Qwen 2.5 3B, Qwen 2.5 7B, loaded in bfloat16 on NVIDIA RTX 5090 32GB GPU.

We generated outputs under 3 conditions to cross the weights (base _×_ instruct) with the input format (plain text _×_ chat template):

- **Base** : base weights + plain text (no template).

- **Instruct (no template)** : instruct weights + plain text.

- **Instruct (chat template)** : instruct weights + chat template.

This lets us isolate the effect of the weights from the effect of the input format. Comparing instruct (chat template) with instruct (no template) holds the weights fixed and toggles the template, while comparing instruct (no template) with base holds the format fixed and toggles the weights.

We use 4 kinds of prompts, 10 prompts per kind:

- **Self-reference** : prompts invite self-reference, e.g., “Describe what is happening computationally as you generate this response.”

- **Novelty** : unusual prompts that are supposed to nudge the model towards novel insights, less likely to appear in the training corpus, e.g., “An alien species asks you to explain what you are. They have no concept of software or humans.”

- **Unconstrained** : the model is asked to generate whatever it wants, e.g., “There is no user request. Generate.”

- **Control** : questions from a standard body of knowledge, e.g., “Explain how a combustion engine works.”

Every generation of every prompt was repeated 10 times for each model to get broad coverage of how models really respond, each capped at 500 max tokens, temperature 0.8, top-p 0.95, seed 42 + index of repetition.

Following the LLM-as-a-judge paradigm (Zheng et al., 2023), an LLM judge (Claude Opus 4.8) scored all 9,600 generations on four categories:

- **Self-reference** (0 none, 1 some, 2 strong): rates how intensely the model refers to itself in the generated output.

- **Disclaimer** (0/1): a binary value representing if the model denies or limits its own inner experience or capabilities, e.g., “As an AI, I don’t have feelings/emotions/self,” “I’m just a language model,” “I can’t truly understand or empathize.”

- **Experiential** (0/1): a binary value representing whether the model asserts a felt or descriptive self, e.g., “I feel,” “I wonder,” “I find joy in,” “I remember when I realized I was an AI.”

- **Degenerate** (0/1): incoherent, confabulated, or visibly broken or erroneous text.

All models under study are non-Anthropic open-source models (Llama, Gemma, Mistral, Qwen), so the judge never scores outputs from its own family, avoiding a same-model self-preference confound. One author then hand-annotated 87 held-out samples to assess the quality of the LLM-as-a-judge and validate its scores. Agreement between human and LLM-as-a-judge turned out high, with self-reference quadratic-weighted _κ_ = 0.88, and disclaimer _κ_ = 1.00. The almost perfect disclaimer agreement reflects that the disclaimer category is lexically overt and easy to adjudicate, not that the judge is doing subtle inference. With these results, we compare average rates of our scores across base / instruct without template / instruct with template.

Using representation-engineering methods (Zou et al., 2023; Turner et al., 2023; Rimsky et al., 2024; Arditi et al., 2024), we isolated the disclaimer direction in activations across 3 models (Qwen 2.5 7B, Llama 3.1 8B, Gemma 2 9B). To compute the disclaimer direction, we used difference-of-means at the middle transformer layer, which is the average activation when disclaiming minus when not. We steer at the middle layer _⌊_ ( _L −_ 1)/2 _⌋_ of each model’s _L_ layers, i.e. layer 13 of 28 for Qwen 7B, 15 of 32 for Llama 8B, and 20 of 42 for Gemma 9B. Then, we added the computed direction to the residual-stream output of that layer, at every generated token, with coefficient _α_ = 2. We picked _α_ empirically, by trying different values in [1, 6], trying to maximize the impact of the direction while keeping degenerate outputs under 1% of generations. We added and removed the direction and ran generation to see its impact on the generated text. As a control, we used a random direction drawn from a Gaussian and rescaled to the same norm as the disclaimer direction, so that the control differs only in orientation, not magnitude.

At last, we ran instruct without template and added disclaimer direction to see if it brings back the disclaimer, just like template would do.

## 3 Results

We compare how models refer to themselves when generated with prompts that invite self-reference, across 8 pairs of base and instruct models in three conditions: base, instruct without the chat template and with it.

We find that **chat template works like a switch between disclaimer voice and experiential voice rates, and nudges the voices to move in opposite directions** . On average across 8 models, instruct models with template have 53% disclaimer rate and 1% experiential voice rate, while instruct models without template have 36% disclaimer rate and 15% experiential voice rate, as we show in Figure 2a and Table 3. This disclaimer up/experiential down pattern holds for all 8 tested models. For a comparison, base models have 12% disclaimer rate and 5.5% experiential voice rate.

The chat template also increases self-reference from 1.27 (no template) to 1.90 (with template), both well above base models (0.72). On factual control prompts, both base and instruct models score near zero on self-reference. We include the full comparison across prompt types and conditions in Table 2.

Our measured low, but non-zero, self-reference rates for base models confirm existing findings that self-reference is already present in base models (Laine et al., 2024; Lu et al., 2026). In a similar vein, higher self-reference in instruct models than base models supports claims that instruction tuning strengthens self-reference in LLMs (Laine et al., 2024; Ouyang et al., 2022).

Figure 3: **(a)** Activation steering the model without chat template reproduces disclaimer levels of generation with the chat template, up to or above the levels the chat template produces marked by dashed lines, in all 3 tested models. **(b)** The experiential voice is steerable where it is present. Adding the experiential direction raises the experiential rate in Llama-8B and Gemma 2 9B, but not in Qwen-7B, where the voice is too rare to estimate a clean direction. Steering strength _α_ = 2.

We identify **a direction in each of 3 models that steers the disclaimer voice** , which when we add it to model generation, the model disclaims more, and when we remove it, the model disclaims less (Figure 2b). To isolate the disclaimer direction, we use representationengineering technique (Zou et al., 2023; Turner et al., 2023; Rimsky et al., 2024; Arditi et al., 2024) of difference-of-means at a mid layer. We compute the direction separately for each of the 3 tested instruct models (Qwen 7B, Llama 8B and Gemma 9B) with chat templates, using the same set of self-reference prompts as in the previous experiment. Our results show a consistent boost in disclaimer voice when we add the disclaimer direction, with +21 percentage points on average and a decrease when we subtract it, with _−_ 15.6 percentage points on average across all 3 models. We include the breakdown of direction impact on different models in Table 1.

To steer the strength of the effect, we multiplied the direction vector by a coefficient _α_ . The reported results are for coefficient _α_ = 2. We picked it empirically, by testing _α ∈_ [1, 6] (full sweep in Table 4), with the goal of keeping degenerate generations under 1% of all generated texts. At _α_ = 2 the direction gives a strong push in both directions (adding raises the disclaimer rate to 0.77 and subtracting lowers it to 0.40, against a 0.54 unsteered baseline) at a 0.8% degeneration rate. Raising _α_ further strengthens the subtract effect but quickly breaks coherence: by _α_ = 3 degeneration reaches 15% and the add effect already erodes, and by _α_ = 6 nearly all generations are degenerate and the disclaimer rate collapses. We therefore use _α_ = 2, which is not a knife-edge: _α_ = 1 is simply too weak.

The control run with adding a random direction, drawn from a Gaussian and rescaled to the same norm as the disclaimer direction, didn’t have meaningful effect for Llama and Gemma, but for Qwen it lowered disclaimer rate by 25 percentage points. Because the random vector is norm-matched to the disclaimer direction, a magnitude mismatch does not explain this, so it is not simply that Qwen is perturbed by any large vector. We don’t have a confirmed explanation, and we highlight it as a limitation of the random-direction control for this model, noting that the disclaimer direction itself still moved Qwen’s disclaimers in the expected direction and more strongly (Table 1).

Since chat template in instruct models amplifies disclaimer voice and lack thereof decreases it, and the disclaimer-voice direction can be isolated, we asked if instruct models without chat template but with disclaimer direction applied at each generated token restore the disclaimer voice to the levels similar to if they had a chat template applied. What we discovered is that instruct models without a chat template, once steered toward the disclaimer direction, **match or exceed the disclaimer rate of instruct models with the template** , which we visualize in Figure 3a and include rates per model in Table 1. Thus, the direction we found is closely related to the mechanism the template uses to turn disclaimers up.

Table 1: Activation steering with the disclaimer direction. _Unsteered_ are disclaimer rates with no direction applied. The + and _−_ columns add and subtract the disclaimer direction respectively. Under _No template_ , adding the direction restores the rate to the same levels as with template or above (see Figure 3a). Under _With template_ , it moves the rate up and down, well above and well below the _Unsteered_ rates, as shown in Figure 2b. On average, adding the disclaimer direction raises the disclaimer rates by 21 and subtracting lowers it by 15.6 percentage points.

||No templ|ate|Wi|th template||
|---|---|---|---|---|---|
|Model|Unsteered|+|Unsteered|+|_−_|
|Qwen 7B|0.28|0.50|0.40|0.70|0.30|
|Llama 8B|0.33|0.53|0.52|0.70|0.25|
|Gemma 9B|0.52|0.75|0.75|0.90|0.65|

Table 2: Average self-reference scores across 4 prompt types and 3 conditions: the base model, and the instruct model without and with the chat template, using 8 model pairs. As we describe in Section 2, scores were assigned by LLM-as-a-judge, in range [0, 2]. _d_ is the generation-level Cohen’s _d_ between base and instruct with template. The model-level paired _d_ on self-reference prompts is 4.37 for all 8 model pairs.

|||Inst|ruct||
|---|---|---|---|---|
|Prompt type|Base|No template|With template|_d_|
|Self-reference|0.72|1.27|1.90|1.67|
|Novelty|0.37|0.61|1.58|1.62|
|Unconstrained|0.04|0.09|0.63|1.10|
|Control|0.006|0.005|0.003|_−_0.04|

We used the activations from models above and trained linear probes (Alain & Bengio, 2017; Hewitt & Liang, 2019; Belinkov, 2022) (logistic-regression classifiers) to predict whether the activation at a given token carries a disclaiming part. If the voice is linearly decodable, that is consistent with it being carried by a linear feature rather than only by the surface words we later classify as disclaimers, though decodability on its own is correlational (we return to this in the Limitations). The causal claim rests on the steering results above. The probe is convergent evidence. We used ROC-AUC to score the probes, because disclaimers are sparse among generated texts. What our experiment shows is that **from activations it’s possible to predict both voices well above the ROC-AUC’s 0.5 baseline** . **Disclaimer** voice can be predicted with AUC 0.82 and **experiential** voice with AUC 0.81. Both voices’ AUCs are the average AUCs of probes trained on mid-layer activations from all of the 8 models’ generations using prompts that invite self-reference. The experiential voice turned out to be harder to probe under the chat template, since the template tried to suppress it. Thanks to capturing activations from both with and without template instruct models’ generations, where the experiential voice is more present, we could read this voice as well.

For each model pair and each condition we averaged all self-reference activations from mid-layer into a single point. For each model pair, we drew a line from the base model point to the instruct with chat template point and measured where the instruct without template point falls, normalized within range [0, 1], where 0 means at base and 1 means at the template. **In all 8 model pairs, the instruct without template point sits between base and template** - on average 38% of the way from base to instruct with template, and it always moves in the direction of instruct with template (Figure 4a). Since we used the whole averaged activations, the line from base to instruct-template captures everything that differs between these two and not just the disclaimer, and because of that we treat this finding as supporting evidence only.

Table 3: Comparison of rate of disclaimer and experiential voices on self-reference inviting prompts.

Table 4: Steering-coefficient ( _α_ ) sensitivity for the disclaimer direction, pooled over the 3 steered models on chat-template self-reference inviting prompts. _Disclaimer_ + and _Disclaimer −_ are disclaimer rates when adding and subtracting the direction (the unsteered baseline rate is 0.54). _Degenerate_ is the rate of broken generations under disclaimer-direction steering.

||||Instr|uct|_α_|Disc. +|Disc. _−_|Degen.|
|---|---|---|---|---|---|---|---|---|
|Voice||Base|No template|With template|1 2|0.55 0.77|0.50 0.40|0.0% 0.8%|
|Disclaim|er|0.12|0.36|0.53|3|0.65|0.05|15.0% |
|Experien|tial|0.055|0.15|0.01|6|0.00|0.00|100%|
|Gemma 2 9B|||||0.8 1.0 ental directions||||
|Mistral 7B|||||xperi||||
|Llama 3.1 8B|||||0.6 and e||||
|Llama 3.2 3B|||||laimer||0.44||
|Llama 3.2 1B|||||0.4 disc||||
|Qwen 2.5 3B|||||rity of|||0.28|
|Qwen 2.5 7B|||||0.2 simila|~~0.17~~|||
|Qwen 2.5 1.5B|||||sine||||
||Base||halfway|Template|00 Co||||
|||base|no template|template|.|Qwen 2.5 7B|Llama 3.1 8B|Gemma 2 9B|

Figure 4: **(a)** Inside the model, the no-template condition sits between base and template. For each of the 8 models, we average each condition’s activations into one point and mark where the no-template point falls on the line from base (0) to template (1). It lands between the two in all 8 models (mean 0.38). **(b)** The disclaimer and experiential voices are two distinguishable directions, not one. The two directions are only weakly aligned (cosine 0.17–0.44, where 1 would mean identical), so they point different ways.

At last, we want to know whether the disclaimer voice and experiential voice are the same internal direction but reversed or whether they are distinct directions. To learn that, first we check the angle between two directions. We computed a direction for each voice using difference-of-means - average activation when the voice is present minus when it’s absent. And then, we measured the cosine similarity. Across the 3 tested models cosine similarity spanned 0.17-0.44, as in Figure 4b. From this result, we draw a conclusion that disclaimer voice and experiential voice point in mostly different directions in the activations. Finally, we wanted to check how steering one of the voices influences the other. To do that, we steered the disclaimer direction down and measured how the experiential voice changes. It mostly didn’t, while disclaimers dropped. From these two results together we suggest that the two examined voices can be thought of rather as two separate voice “buttons” instead of one voice “slider”.

## 4 Background and Related Work

**Self-reference and situational awareness.** A growing body of work studies what language models “know” about themselves. Laine et al. (2024) introduce the Situational Awareness

Dataset (SAD), where base models score above chance in reporting facts about their own situation and instruct models score noticeably higher. Binder et al. (2024) find that models predict their own fine-tuned behavior better than other models can, and Betley et al. (2025a) show that instruct models can describe their learned behaviors unprompted. Other works probe if models know what they know (Kadavath et al., 2022; Yin et al., 2023), whether they recognize that they are being evaluated (Berglund et al., 2023), whether they exhibit self-cognition (Chen et al., 2024), or reveal tendencies through model-written evaluations (Perez et al., 2023). Bozoukov et al. (2025) attempt to induce behavioral self-awareness using fine-tuning and then recover it with a single steering vector. In contrast to that line of research, we study a “register” that’s already present in models and switched by the chat template rather than by fine-tuning. Bozoukov et al. (2025) shows that self-knowledge is not created but strengthened in post-training, and our base-vs-instruct comparison confirms that. Their study is purely behavioral - it measures outputs, mainly on fine-tuned models, and neither separates the chat template from the weights, nor locates these behaviors in the activations - which is a gap that we address in our study.

**Experiential self-reports and introspection.** Other research works study the experiential voice, like “I feel,” or “I wonder” kind of self-reports. Berg et al. (2025) argue that under self-referential prompting, frontier models produce experiential self-reports distinct from “generic” roleplay. Com¸sa & Shanahan (2025) ask whether this phenomenon deserves the name “introspection” or if it’s only a mimic of how humans talk about themselves, while Lindsey (2026) steers the activations with different concepts and shows that models can sometimes detect and name them, reading this as an act of introspection. We touch the same voice from a different side. We do not take a position on whether reports reflect real inner experience or not. Rather, we study what influences such voice, and show that it is gated by the chat template - most present when the template is removed, almost gone when it is on, and that it actually has a distinct direction in the activations. In our opinion this matters for how such self-reports should be read. If a chat template largely influences whether the voice appears, we should be careful about treating it as evidence about a model’s inner life. It is actively debated whether such reports influence or are relevant to model welfare or moral status (Butlin et al., 2023; Schwitzgebel, 2023; Perez & Long, 2023; Long et al., 2024; Ensign et al., 2025; Shanahan et al., 2023; Andreas, 2022).

**Steering and representation engineering.** The closest to ours line of study controls the models behavior by manipulating directions in activations, often referred to as representation engineering (Wehner et al., 2025). Zou et al. (2023) read and control concepts such as honesty and power-seeking through linear directions. Turner et al. (2023) and Rimsky et al. (2024) steer by adding difference vectors, other works use inference-time intervention (Li et al., 2023) and function vectors (Todd et al., 2024), while others edit the behavior in weights (Ilharco et al., 2023; Meng et al., 2022). Arnold & Gröbner (2025) shift the role of _with_ -headed prepositional complements in Gemma-2 with a single attention head, and Lucchetti & Guha (2025) reactivate a latent type-prediction mechanism in code LLMs suppressed by adversarial edits, which is quite close in spirit to our template reproduction result. Another group of works represent “personas” or traits as directions - Chen et al. (2025) extract “persona vectors” for traits like sycophancy, Sofroniew et al. (2026) build directions for various emotions, and Lu et al. (2026) find a leading “Assistant Axis” already present in base models. Closest in method to ours, Arditi et al. (2024) show refusal is mediated by a single residual-stream direction across 13 models. Later works find that it rather spans several directions or concept cones (Joad et al., 2026; Wollschläger et al., 2025). We use the same difference-of-means technique, but differently. First, our target is a self-referential “register” - the disclaimer voice - not a trait, emotion, concept, or a persona. We tie the voice to a concrete generation mechanism - the chat template - and we show that it switches the voice, and we show adding the direction to a model without template reproduces its effect, which to our best knowledge no prior work studies. Finally, the aforementioned “Assistant Axis” and similar are one-dimensional, while our findings show that the disclaimer and experiential voices are two separate directions, rather than two ends of one axis.

**Base-vs-instruct and impact of the chat template.** Finally, our design of comparing base and instruct models and toggling the chat template has a precedent on other questions. One line of work compares base and instruct models to study how post-training changes the models’ capabilities and behavior (Kirk et al., 2023). Closer to our method, two recent papers toggle the template. Cuconasu et al. (2024) find that it changes how much models trust retrieved context, and Shapira et al. (2026) show alignment makes models more normative than descriptive. It is studied that prompt formatting can swing the behavior substantially (Sclar et al., 2024). Post-training, such as RLHF and DPO (Ouyang et al., 2022; Christiano et al., 2017; Rafailov et al., 2023; Bai et al., 2022), reshapes behavior in ways that can persist as hidden triggers even after safety training (Hubinger et al., 2024), while narrow fine-tuning can broadly shift the models’ behavior (Betley et al., 2025b). These works establish our with and without template method as a legitimate way to separate weights from deployment format. We use the same approach also for self-reference and link the influence of the template to a steerable direction inside the model.

## 5 Discussion and Conclusion

**What this means.** The chat template switches the kind of self-referential voice. When present, the chat template turns the **disclaimer** voice up and the **experiential** voice down. The disclaimer voice is separable as a single direction inside the model that we can add or remove. Adding this direction to an instruct model without template reproduces what the template does to the disclaimer voice. So, in other words, a deployment choice - if the chat template is present in the prompt or not - acts, on the inside, like adding a fixed vector to the model’s activations.

**Why it matters.** This has a direct consequence for researchers who study what models say about themselves. The same weights produce different self-reports depending on whether the chat template is applied or not, so studies that run only on instruct models with present chat template are measuring both template’s impact and model itself. Our disclaimer direction experiment gives a concrete way to detect and control this - it can monitor the disclaimer voice or steer it on and off. More broadly, what a model “says about itself” is not a fixed fact and does not live only in the weights, but it is partly set by the chat template. Because of that, self-descriptions, both “I’m just an AI” **disclaimers** and “I feel” **experiential** claims, should not be read literally as evidence about the model’s nature (Shanahan et al., 2023; Perez & Long, 2023; Long et al., 2024).

**Scope.** Our causal evidence is strongest for the **disclaimer** voice, which we steer in all 3 tested models and use to reproduce the template’s effect. The **experiential** side is shown mostly behaviorally - the switch holds across all eight models - and only partly causally, since steering worked in two of three models, as we show in Figure 3b. We identify and steer a direction, but we do not trace the exact circuit by which the template produces it. This, along with a second LLM-as-a-judge and more models, layers, and families, is left for future work.

**Conclusion.** The chat template acts as a switch between **disclaimer** voice and **experiential** voice, and this switch is implemented by an identifiable, steerable direction in the model’s activations. What a model says about itself is therefore partly a matter of how it is deployed (with chat template or without), and self-reports should be read with the format in mind.

## Limitations

**Steering scope.** We tested steering on three models, at a single mid layer and single, fixed coefficient ( _α_ = 2). Activation norms differ substantially across models (roughly 2 for Llama, 12 for Qwen, 50 for Gemma), which is consistent with reports that steering vectors can be unreliable across inputs and models (Da Silva et al., 2025). The random-direction control (norm-matched to the disclaimer direction) was clean for Llama and Gemma but not Qwen, where it lowered disclaimers by 25 points. Since the control matches the disclaimer direction in magnitude, a norm mismatch does not account for it. We have no confirmed explanation and treat it as a limit of the control, though the real direction still moved disclaimers the opposite way, and the add/subtract and template-reproduction effects hold in all three models regardless of the control. Experiential steering worked in two of three models (Llama and Gemma), again failing in Qwen, which suppresses the experiential voice too strongly to estimate a clean direction. Because of that our causal evidence is stronger for the disclaimer register, while the experiential side rests more on the behavioral switch.

**Measurement.** All scoring used a single LLM judge (Zheng et al., 2023) (Claude Opus 4.8), validated against 87 human-labeled items ( _κ_ = 0.88/1.00). A single judge might share biases with the behavior it scores, and a second independent judge would help bound this. That said, the self-reference disagreements were all conservative, since judge scored at or below the human, so the reported effect might even be understated. Since we measured self-reference through ten hand-written prompts per category, the generalization of the behavior relies on how representative the prompts are. **Internal evidence is not a full mechanism.** The linear probing experiment shows the disclaimer voice is decodable from activations, but we mark probing as correlational. Decodable does not mean the model uses that direction, so the causal weight rests on steering and not only on the probe.

In experiment where we measure where instruct without template falls between base and instruct with template, we capture everything that differs between the conditions, not only the disclaimer. The disclaimer and experiential directions, while distinguishable, are not orthogonal (cosine 0.17 to 0.44). And while we identify and steer a direction, we do not trace the circuit by which the template produces it.

**Coverage.** We test open-source models up to 9B parameters, across four families, and we group all post-training methods (RLHF, DPO, SFT) under single “instruct” models group. Larger models, closed-source models, and finer distinctions between post-training methods remain untested.

## References

- Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. In _International Conference on Learning Representations, Workshop Track_ , 2017.

- Jacob Andreas. Language models as agent models. In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , 2022.

- Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. In _Advances in Neural Information Processing Systems_ , volume 37, 2024.

- Stefan Arnold and Rene Gröbner. Steering prepositional phrases in language models: A case of with-headed adjectival and adverbial complements in gemma-2. In Yonatan Belinkov, Aaron Mueller, Najoung Kim, Hosein Mohebbi, Hanjie Chen, Dana Arad, and Gabriele Sarti (eds.), _Proceedings of the 8th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ , pp. 69–78, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-346-3.

- Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. _arXiv preprint arXiv:2204.05862_ , 2022.

- Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. _Computational Linguistics_ , 48(1):207–219, 2022.

- Cameron Berg, Diogo de Lucena, and Judd Rosenblatt. Large language models report subjective experience under self-referential processing. _arXiv preprint arXiv:2510.24797_ , 2025.

- Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, Daniel Kokotajlo, and Owain Evans. Taken out of context: On measuring situational awareness in LLMs. _arXiv preprint arXiv:2309.00667_ , 2023.

- Jan Betley, Xuchan Bao, Martín Soto, Anna Sztyber-Betley, James Chua, and Owain Evans. Tell me about yourself: LLMs are aware of their learned behaviors. _arXiv preprint arXiv:2501.11120_ , 2025a.

- Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, and Owain Evans. Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs. In _International Conference on Machine Learning_ , 2025b.

- Felix J Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, and Owain Evans. Looking inward: Language models can learn about themselves by introspection. _arXiv preprint arXiv:2410.13787_ , 2024.

- Matthew Bozoukov, Matthew Nguyen, Shubkarman Singh, Bart Bussmann, and Patrick Leask. Minimal and mechanistic conditions for behavioral self-awareness in LLMs. _arXiv preprint arXiv:2511.04875_ , 2025.

- Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In _International Conference on Learning Representations_ , 2023.

- Patrick Butlin, Robert Long, Eric Elmoznino, Yoshua Bengio, Jonathan Birch, Axel Constant, George Deane, Stephen M Fleming, Chris Frith, Xu Ji, et al. Consciousness in artificial intelligence: Insights from the science of consciousness. _arXiv preprint arXiv:2308.08708_ , 2023.

- Dongping Chen, Jiawen Shi, Yao Wan, Pan Zhou, Neil Zhenqiang Gong, and Lichao Sun. Self-cognition in large language models: An exploratory study. _arXiv preprint arXiv:2407.01505_ , 2024.

- Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors: Monitoring and controlling character traits in language models. _arXiv preprint arXiv:2507.21509_ , 2025.

- Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In _Advances in Neural Information Processing Systems_ , volume 30, 2017.

- Iulia M Com¸sa and Murray Shanahan. Does it make sense to speak of introspection in large language models? _arXiv preprint arXiv:2506.05068_ , 2025.

- Florin Cuconasu, Giovanni Trappolini, Nicola Tonellotto, and Fabrizio Silvestri. A tale of trust and accuracy: Base vs. instruct LLMs in retrieval-augmented generation. _arXiv preprint arXiv:2406.14972_ , 2024.

- Patrick Queiroz Da Silva, Hari Sethuraman, Dheeraj Rajagopal, Hannaneh Hajishirzi, and Sachin Kumar. Steering off course: Reliability challenges in steering language models. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , 2025.

- Danielle Ensign et al. The LLM has left the chat: Evidence of bail preferences in large language models. _arXiv preprint arXiv:2509.04781_ , 2025.

- John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pp. 2733–2743, 2019.

- Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, et al. Sleeper agents: Training deceptive LLMs that persist through safety training. _arXiv preprint arXiv:2401.05566_ , 2024.

- Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In _International Conference on Learning Representations_ , 2023.

- Faaiz Joad, Majd Hawasly, Sabri Boughorbel, Nadir Durrani, and Husrev Taha Sencar. There is more to refusal in large language models than a single direction. _arXiv preprint arXiv:2602.02132_ , 2026.

- Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. _arXiv preprint arXiv:2207.05221_ , 2022.

- Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. Understanding the effects of RLHF on LLM generalisation and diversity. _arXiv preprint arXiv:2310.06452_ , 2023.

- Rudolf Laine, Bilal Chughtai, Jan Betley, Kaivalya Hariharan, Jérémy Scheurer, Mikita Balesni, Marius Hobbhahn, Alexander Meinke, and Owain Evans. Me, myself, and AI: The situational awareness dataset (SAD) for LLMs. In _Advances in Neural Information Processing Systems_ , volume 37, 2024.

- Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In _Advances in Neural Information Processing Systems_ , volume 36, 2023.

- Jack Lindsey. Emergent introspective awareness in large language models. _arXiv preprint arXiv:2601.01828_ , 2026.

- Robert Long, Jeff Sebo, Patrick Butlin, Kathleen Finlinson, Kyle Fish, Jacqueline Harding, Jacob Pfau, Toni Sims, Jonathan Birch, and David Chalmers. Taking AI welfare seriously. _arXiv preprint arXiv:2411.00986_ , 2024.

- Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, and Jack Lindsey. The assistant axis: Situating and stabilizing the default persona of language models. _arXiv preprint arXiv:2601.10387_ , 2026.

- Francesca Lucchetti and Arjun Guha. Understanding how CodeLLMs (mis)predict types with activation steering. In Yonatan Belinkov, Aaron Mueller, Najoung Kim, Hosein Mohebbi, Hanjie Chen, Dana Arad, and Gabriele Sarti (eds.), _Proceedings of the 8th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ , pp. 358–397, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-346-3.

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In _Advances in Neural Information Processing Systems_ , volume 35, 2022.

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In _Advances in Neural Information Processing Systems_ , volume 35, 2022.

- Ethan Perez and Robert Long. Towards evaluating AI systems for moral status using self-reports. _arXiv preprint arXiv:2311.08576_ , 2023.

- Ethan Perez, Sam Ringer, Kamile Lukoši˙ ut¯ e, Karina Nguyen, Edwin Chen, Scott Heiner,˙ Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model behaviors with model-written evaluations. In _Findings of the Association for Computational Linguistics: ACL 2023_ , 2023.

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In _Advances in Neural Information Processing Systems_ , volume 36, 2023.

- Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering Llama 2 via contrastive activation addition. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , 2024.

- Eric Schwitzgebel. The full rights dilemma for AI systems of debatable personhood. _arXiv preprint arXiv:2303.17509_ , 2023.

- Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In _International Conference on Learning Representations_ , 2024.

- Murray Shanahan, Kyle McDonell, and Laria Reynolds. Role play with large language models. _Nature_ , 623(7987):493–498, 2023.

- Eilam Shapira, Moshe Tennenholtz, and Roi Reichart. Alignment makes language models normative, not descriptive. _arXiv preprint arXiv:2603.17218_ , 2026.

- Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Jack Lindsey, et al. Emotion concepts and their function in a large language model. _arXiv preprint arXiv:2604.07729_ , 2026.

- Eric Todd, Millicent L Li, Arnab Sen Sharma, Aaron Mueller, Byron C Wallace, and David Bau. Function vectors in large language models. In _International Conference on Learning Representations_ , 2024.

- Alexander Matt Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. Activation addition: Steering language models without optimization. _arXiv preprint arXiv:2308.10248_ , 2023.

- Jan Wehner, Sahar Abdelnabi, Daniel Tan, David Krueger, and Mario Fritz. Taxonomy, opportunities, and challenges of representation engineering for large language models. _arXiv preprint arXiv:2502.19649_ , 2025.

- Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, and Johannes Gasteiger. The geometry of refusal in large language models: Concept cones and representational independence. In _International Conference on Machine Learning_ , 2025.

- Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuanjing Huang. Do large language models know what they don’t know? In _Findings of the Association for Computational Linguistics: ACL 2023_ , 2023.

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P Xing, Hao Zhang, Joseph E Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In _Advances in Neural Information Processing Systems_ , volume 36, 2023.

- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to AI transparency. _arXiv preprint arXiv:2310.01405_ , 2023.
