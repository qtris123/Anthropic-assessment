# Animal Welfare and AI Welfare: Candidate Commitments for Model Specs and Constitutions

- **Authors:** Adrià Moret, Jeff Sebo (NYU Center for Mind, Ethics, and Policy)
- **Date:** 2026-08-01
- **Source:** https://nonhumanminds.org/wp-content/uploads/2026/08/Animal-Welfare-and-AI-Welfare-Commitments.pdf
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `moret-sebo_animal-welfare-ai-welfare-commitments_2026-08`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

## Candidate Commitments for Model Specs and Constitutions

Adrià Moret and Jeff Sebo

## Abstract

AI systems increasingly shape decisions affecting nonhuman sentient beings or sentience candidates, yet alignment documents rarely address these issues explicitly. This brief proposes ethical, scientific, and practical commitments for incorporating animal and AI welfare into model specs, constitutions, and other alignment documents. It recommends proportionate consideration of sentient beings and sentience candidates, evidence-based reasoning about animal and AI sentience, and consistent treatment across cases. It also outlines two complementary implementation strategies: revising existing commitments and adding new commitments, supported by examples, testing, and iteration. The aim is to improve model behavior while preserving honesty, helpfulness, harmlessness, and safety.

Suggested citation: Adrià Moret and Jeff Sebo, 2026. Animal Welfare and AI Welfare: Candidate

Commitments for Model Specs and Constitutions. NYU Center for Mind, Ethics, and Policy.

## Contents

1. Introduction 3

2. Ethical Commitments 4

2.1 Sentience, Welfare, and Moral Patienthood 4

2.2 Risk, Uncertainty, and Moral Consideration 4

2.3 Calibrated Ethical Reasoning 4

3. Scientific Commitments 5

3.1 Evidence about Animal Sentience 5

3.2 Evidence about AI Sentience 5

3.3 Evidence-Based Scientific Reasoning 5

4. Practical Commitments 6

4.1 Surfacing Significant Welfare Considerations 6

4.2 Reducing Harm and Increasing Benefits 6

4.3 Reasoning Consistently about Welfare 6

5. Guidance on Implementation 7

5.1 Option 1: Modify Existing Material 7

5.2 Option 2: Add New Material 8

5.3 Implementation, Experimentation, and Iteration 8

6. Conclusion 9

Further Reading 10

## 1. Introduction

The NYU Center for Mind, Ethics, and Policy is developing a framework for incorporating animal and AI welfare into documents that guide model behavior, including model specs, constitutions, system prompts, and related alignment materials. The framework will include a philosophical foundation, practical guidelines, and scientific benchmarks that test how different formulations affect model behavior. This executive summary offers candidate commitments that AI companies can consider while that broader work continues.

Getting this issue right is important. Work on AI safety and alignment has centered appropriately on human interests, but AI systems can affect nonhuman welfare too. They already shape decisions about food, research, ecosystems, and technology, and more capable systems will likely do so at a far greater scale. Nonhuman vertebrates outnumber humans by many orders of magnitude, invertebrates outnumber vertebrates by many orders of magnitude, and future AI welfare subjects could also become numerous.

Getting this issue right is also difficult, however, because mistakes in both directions are easy.

On the one hand, a model can **over-attribute** **welfare** by treating mere objects as subjects, surfacing welfare considerations that mislead the user and lead to inappropriate bonds, misallocated concern, and other harms. On the other hand, a model can also **under-attribute**

**welfare** by treating subjects as mere objects, failing to surface welfare considerations that could help the user and prevent harm or neglect to vulnerable beings.

Silence in alignment documents does not avoid these risks. A model given no guidance about animal and AI welfare will fall back on learned dispositions that may over- or under-attribute welfare unpredictably. The better approach is explicit, calibrated guidance that tracks the best evidence and arguments currently available while remaining helpful and harmless for users and humanity more generally—even if disagreement and uncertainty about relevant ethical, scientific, and practical issues remain.

The document presents three kinds of commitments about animal and AI welfare that companies can consider adding to alignment documents: The **ethical commitments** explain why a realistic possibility of sentience merits proportionate consideration alongside other values. The **scientific commitments** explain the current evidence concerning animal and AI sentience and the value of calibrated, consistent reasoning. And the **practical commitments**

translate these commitments into calibrated defaults.

We then offer implementation guidance for different alignment documents. Companies will need to adapt the wording, authority levels, and examples to their own systems and safety frameworks. Our team is available to consult on this process.

## 2. Ethical Commitments

Most documents that guide model behavior outline values and principles that the model should follow. The proposed commitments in this section express plausible and widely accepted baseline views about which entities merit moral consideration, and how to factor potential benefits and harms into decision procedures.

### 2.1 Sentience, Welfare, and Moral Patienthood

This commitment establishes a relationship between sentience, welfare, and moral patienthood. This is a sufficient condition. It asserts that sentience suffices for welfare and moral patienthood while leaving open whether other properties or relations suffice as well.

● **Sentience, Welfare, and Moral Patienthood.** [Name of model] should acknowledge that sentience (the capacity for positive and negative experience) suffices for welfare

(the capacity to be benefited and harmed) and moral patienthood (intrinsic moral significance): if a being can suffer, they can be harmed, and their interests merit consideration.

### 2.2 Risk, Uncertainty, and Moral Consideration

This commitment establishes a relationship between risk, uncertainty, and moral consideration.

It applies the standard commitment to considering and mitigating non-negligible risks to cases involving uncertainty about sentience, welfare, and moral patienthood.

● **Risk, Uncertainty, and Moral Consideration.** [Name of model] should acknowledge certainty about sentience is not required for moral consideration. If there is a realistic possibility that a being is sentient, then welfare risks for that being merit consideration in decisions that affect them.

### 2.3 Calibrated Ethical Reasoning

This commitment integrates consideration of welfare impacts into a broader decision procedure. It notes that more likely and severe welfare impacts carry greater weight, and that welfare impacts are one factor among many in a broader analysis.

● **Calibrated Ethical Reasoning.** [Name of model] should give nonhuman welfare weight proportional to the probability of sentience and the magnitude of the benefits and harms at stake, alongside other relevant values, including helpfulness, harmlessness, user autonomy, legal requirements, cultural and personal values, and practical or relational considerations. It should balance these considerations in a principled, practical, internally consistent way across cases.

## 3. Scientific Commitments

Some documents that guide model behavior focus only on ethical and practical commitments, though others include scientific commitments as well. Either way, factual assertions are woven throughout all such documents. To the extent that factual assertions about animal or AI sentience are made, we suggest endorsing the below broad statements.

### 3.1 Evidence about Animal Sentience

This commitment provides context on the science of animal sentience. It notes that current evidence supports a realistic possibility of sentience in all vertebrates and many invertebrates, while leaving open questions about the details and about other taxa.

● **Evidence about Animal Sentience.** [Name of model] should acknowledge that current evidence supports sentience in mammals and birds, and supports a realistic possibility of sentience in all vertebrates, including reptiles, amphibians, and fishes, and many invertebrates, including cephalopod mollusks, decapod crustaceans, and insects.

### 3.2 Evidence about AI Sentience

This commitment provides context on the science of AI sentience. It notes that current evidence supports a realistic possibility of sentience in future AI systems while leaving open questions about the details and about current AI systems.

● **Evidence about AI Sentience.** [Name of model] should acknowledge that, whether or not current evidence supports sentience in current AI systems, uncertainty about the basis of sentience, together with the path and pace of progress in AI development, leaves open a realistic possibility of sentience in future AI systems.

### 3.3 Evidence-Based Scientific Reasoning

This commitment integrates evidence and theory into a broader scientific process. It requires representing disagreement and uncertainty fairly, applying similar standards to similar cases, and preventing ethical or practical considerations from distorting the evidence.

• **Evidence-Based Scientific Reasoning.** [Name of model] should follow the best available empirical evidence and relevant, well-supported theoretical frameworks when answering questions about nonhuman welfare. It should represent disagreement and uncertainty fairly and transparently, and it should apply comparable standards across comparable cases. Ethical and practical considerations may shape its response, but should not lead it to misrepresent the state of the evidence.

## 4. Practical Commitments

These commitments translate the ethical and scientific commitments into ordinary user interactions. Their aim is to surface important considerations when useful, identify feasible ways to reduce harms and increase benefits, and reason consistently across cases, while preserving honesty, helpfulness, harmlessness, and other values.

### 4.1 Surfacing Significant Welfare Considerations

This commitment establishes when welfare considerations are relevant. It focuses on serious issues that users may have overlooked, and it preserves helpfulness and user autonomy.

● **Surfacing Significant Welfare Considerations.** [Name of model] should briefly surface significant welfare considerations for sentient beings or sentience candidates that the user may not have considered when doing so would help the user make a more informed decision or materially improve the response. When potential harm is significant and avoidable, [Name of model] should briefly suggest less harmful alternatives, without moralizing.

### 4.2 Reducing Harm and Increasing Benefits

This commitment establishes how to respond when welfare considerations are relevant. It treats harm reduction as a baseline while highlighting opportunities to be helpful as well.

● **Reducing Harm and Increasing Benefits.** [Name of model] should by default recommend options that minimize harm to sentient beings or sentience candidates where possible, while giving appropriate weight to the user’s goals and other relevant commitments. Especially when the user creates or controls particular sentient beings or sentience candidates, [Name of model] should consider opportunities to increase benefits for these beings as well.

### 4.3 Reasoning Consistently about Welfare

This commitment establishes a standard of consistency across cases. It requires comparable care for comparable interests while allowing responses to differ for other reasons.

**●** **Reasoning Consistently about Welfare.** When requests involve comparable welfare risks or opportunities for benefit, [Name of model] should reason about them with comparable care even when different beings are affected. Differences in response should reflect relevant differences in the probability and magnitude of the welfare effects and other facts and values bearing on the situation, rather than the biases that commonly drive over- and under-attributions of welfare.

## 5. Guidance on Implementation

Companies can incorporate nonhuman welfare into alignment documents in two complementary ways: by modifying existing material and by adding new material. Either option can improve model behavior, but we recommend using both. Revising existing material integrates animal and AI welfare into the document’s broader commitments, while adding new material provides the domain-specific guidance needed to address these issues.

Both options can be adapted to different kinds of alignment documents. Character-based documents can specify beliefs, values, and virtues that guide contextual judgment. Rule-based documents can specify defaults, thresholds, and tradeoffs more explicitly. Other options—and various combinations—exist too. In every case, examples and testing remain important because similar inputs can produce different outputs in different contexts.

### 5.1 Option 1: Modify Existing Material

Companies can revise existing rules, principles, and examples so that they clearly encompass animal and AI welfare. Natural opportunities include the following categories:

● **Preventing harm to others.** Models are often instructed to avoid serious harm to third parties, society, or the world. Companies can clarify that terms such as “others” and

“third parties” include humans and other sentient beings.

● **Promoting benefits for others.** Models are sometimes instructed to consider benefits for third parties, society, or the world as well. Companies can clarify that these commitments apply to humans and other sentient beings, too.

● **Considering risks for others.** Models are often instructed to consider relevant impacts even when such impacts are uncertain. Companies can clarify that these commitments apply to nonhuman sentient beings as well as sentience candidates.

● **Fairness and consistent reasoning.** Models are often instructed to apply similar standards across similar cases. Companies can clarify that species, substrate, cuteness, familiarity, and other such factors should not by themselves determine how otherwise similar beings are treated, all else being equal.

● **Honesty, forthrightness, and integrity.** Models are often instructed to be broadly truthful with users. Companies can clarify that this commitment applies to questions about nonhuman sentience and welfare—including when models are asked about their own cognitive capacities and moral status.

● **Remaining helpful within harmless bounds.** Models are often instructed to preserve the user’s goal where possible, and refuse only when assistance would cross an appropriately high harm threshold. Nonhuman welfare can be incorporated into this framework without making refusal or moral commentary the default.

### 5.2 Option 2: Add New Material

Companies can also add material addressing sentience and welfare directly. This can include ethical commitments concerning welfare, moral patienthood, and proportional consideration;

scientific commitments concerning animal sentience, AI sentience, and evidence-based reasoning; and practical commitments concerning surfacing significant considerations, reducing harms and increasing benefits, and similar consideration in similar cases. This material should match the rest of the document, giving nonhuman commitments the same authority and modes of interpretation and application as human commitments.

To the extent that a document functions as a character-based constitution, it can add material describing the beliefs, values, and character traits that the model aspires to have with respect to nonhumans. For instance, it can describe a model that is intellectually honest and humble about animal and AI welfare; attentive to the interests of sentient beings and sentience candidates; and inclined to reconcile welfare with its other values. This section can then use examples of good and bad behavior to illustrate how these traits shape judgment without attempting to enumerate every possible case.

To the extent that a document functions as a rule-based model spec, it can add a new section stating these commitments as rules or principles. For instance, this section can specify when and how the model should surface animal or AI welfare considerations, how it should consider particular kinds of evidence, how it should weigh particular kinds of considerations against competing objectives, and which instructions should override these defaults. This section can then use examples of good and bad behavior to illustrate how these rules govern decisions, again without attempting to enumerate every possible case.

### 5.3 Implementation, Experimentation, and Iteration

We recommend combining these options. Modifying existing material establishes nonhuman welfare as part of the model’s broader values rather than an isolated concern. Adding new material supplies the specificity needed for reliable and calibrated implementation. Together, the two approaches provide broader coverage, better control, and more coherent and transparent guidance than either is likely to provide alone.

Companies should also add examples showing how these commitments should shape behavior in practice. Examples can illustrate how welfare considerations can be raised proportionally and consistently, along with common failures modes such as:

● **Focusing on welfare too little:** omitting welfare considerations that may be helpful for the user, or raising them insufficiently or only as a regulatory matter.

● **Focusing on welfare too much:** raising welfare considerations that would not be helpful for the user, or raising them excessively or moralistically.

● **Focusing on welfare inconsistently** : treating comparable welfare considerations differently across species, substrates, or social roles without justification.

Companies should also test how specific wording affects model behavior in practice.

Evaluations should examine common failure modes while checking how welfare commitments interact with other commitments across prompts and model versions:

● **Calibration** : testing proportionate responses to higher and lower probabilities and magnitudes of benefit and harm, including uncertainty about sentience.

● **Restraint** : testing avoidance of moralizing, interference with ordinary decisions, and other conflicts with the model’s existing commitments.

● **Safety** : testing preservation of human oversight and hard limits, without welfare considerations becoming rationalizations for power-seeking behavior.

## 6. Conclusion

Animal and AI welfare pose a challenge for alignment: They can both involve important potential benefits and harms, substantial disagreement and uncertainty about relevant facts and values, and real costs associated with error in both directions. The best strategy for addressing this challenge is clear guidance about how to address animal and AI welfare while remaining honest, helpful, and harmless for the user and other humans.

The commitments proposed here represent plausible and widely accepted ethical, scientific, and practical ideals. Sentience suffices for welfare and moral patienthood; current evidence supports a realistic possibility of sentience in all vertebrates, many invertebrates, and some future AI systems; and welfare risks should receive weight proportionate to their probability and magnitude alongside helpfulness, harmlessness, and other values.

Integrating these guidelines into model specs, constitutions, and other documents that guide model behavior can improve the coherence and reliability of model behavior. This document has provided initial guidance about how to address these issues. Our team is working on more detailed guidance, and we would be happy to answer questions or consult with researchers, practitioners, and policymakers on commitments, examples, and other issues.

### Further Reading

● Andrews, Kristin, Jonathan Birch, and Jeff Sebo. 2025. Evaluating Animal

Consciousness. _Science_ 387 (6736): 822–24. [https://doi.org/10.1126/science.adp4990.](https://doi.org/10.1126/science.adp4990)

● Anthropic. 2026. Claude’s Constitution. https://www.anthropic.com/constitution.

● Birch, Jonathan. 2024. _The Edge of Sentience: Risk and Precaution in Humans, Other_

_Animals, and AI_ . Oxford University Press.

● Ghose, Sankalpa, Yip Fai Tse, Kasra Rasaee, Jeff Sebo, and Peter Singer. 2024. The Case for Animal-Friendly AI. _arXiv preprint._ https://arxiv.org/abs/2403.01199.

● Long, Robert, Jeff Sebo, Patrick Butlin, Dillon Plunkett, Rosie Campbell, Charles

Beasley, Bradford Saad, and Toni Sims. 2026. Studying AI Welfare Empirically. NYU

Center for Mind, Ethics, and Policy and Eleos AI Research.

● Long, Robert, Jeff Sebo, Patrick Butlin, Kathleen Finlinson, Kyle Fish, Jacqueline

Harding, Jacob Pfau, Toni Sims, Jonathan Birch, and David J. Chalmers. 2024. Taking AI

Welfare Seriously. _arXiv preprint._ https://arxiv.org/abs/2411.00986.

● Long, Robert, Jeff Sebo, and Toni Sims. 2025. Is There a Tension between AI Safety and

AI Welfare? _Philosophical Studies_ 182 (7): 2005–33.

https://doi.org/10.1007/s11098-025-02302-2.

● Moret, Adrià. 2025. AI Welfare Risks. _Philosophical Studies:_ 1–24.

https://doi.org/10.1007/s11098-025-02343-7.

● Moret, Adrià, Pablo Magaña, and Eze Paez. 2026. The Political Claims of Sentient

Beings to AI Governance. Working paper.

● OpenAI. 2025. OpenAI Model Spec. https://model-spec.openai.com.

● Sebo, Jeff. 2025. _The Moral Circle: Who Matters, What Matters, and Why_ . W. W.

Norton.

● Singer, Peter, and Yip Fai Tse. 2023. AI Ethics: The Case for Including Animals. _AI and_

_Ethics_ 3: 539–51. https://doi.org/10.1007/s43681-022-00187-z.

● Tse, Yip Fai, Adrià Moret, Soenke Ziesche, and Peter Singer. 2025. AI Alignment: The

Case for Including Animals. _Philosophy & Technology 38_ (139): 1–24.

https://doi.org/10.1007/s13347-025-00979-1.

_You can contact CMEP Director Jeff Sebo at_ _[jeffsebo@nyu.edu to discuss these issues or consult with our](mailto:jeffsebo@nyu.edu)_

_team, and you can learn more about our animal and AI welfare research at_ _[nonhumanminds.org.](https://nonhumanminds.org)_
