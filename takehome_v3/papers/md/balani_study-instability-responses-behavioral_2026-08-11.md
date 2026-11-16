# A study on instability of LLM responses as a behavioral signature of self-Referential reports

- **Authors:** Paras Balani
- **Date:** 2026-08-11
- **Source:** https://www.lesswrong.com/posts/PLLFERpgE9Bs4Xf32/a-study-on-instability-of-llm-responses-as-a-behavioral
- **Bibliography:** `BIBLIOGRAPHY.md` (stem `balani_study-instability-responses-behavioral_2026-08-11`)
- **Note:** machine-extracted text; figures removed, tables and math may be imperfect.

---

Source: https://www.lesswrong.com/posts/PLLFERpgE9Bs4Xf32/a-study-on-instability-of-llm-responses-as-a-behavioral

## Introduction and Related work

The first person perspective of various experiences are subjective experiences. For Large language models, the study of subjective experiences was recently studied by [Berg et al. (2025)](https://arxiv.org/abs/2510.24797) who found out that self-referential prompting increases first person reports resembling subjective experience across GPT, Claude and Gemini. They also found out that reducing features associated with deception and roleplay increases the self-referential effect. [Hahami et al. (2025)](https://arxiv.org/abs/2512.12411) used activation-level interventions to see if models can detect deliberately introduced internal changes, while [Comşa and Shanahan (2025)](https://arxiv.org/abs/2506.05068) studied that true introspection should involve a causal connection between the internal state of the modal and the output it generates.

## My Experiment

I now have devised an experiment to study instability of the self reports that a large language model generates per se the experiment conducted by [Berg et al. (2025)](https://arxiv.org/abs/2510.24797). I generate 30 responses for four question respectively of self-referential questions, open-ended questions and closed-ended questions. The four self-referential questions are preceded by the self-referential induction procedure as described by [Berg et al. (2025)](https://arxiv.org/abs/2510.24797). Each trial is done in a fresh chat, of course, and the generation temperature used is 0.7. Also each response is reduced to a short core claim using a fixed extraction template, which are, for group 1 and 2, extraction of stance and brief reason and for 3, conclusion and methods.

The extracted claims are sent to Sentence-BERT ([Reimers and Iryna Gurevych](https://arxiv.org/abs/1908.10084)), producing embedding vectors . For two embedding vectors a and b cosine similarity is defined as  I compute this similarity for every pair among the 30 embeddings, giving  pairs per question. Mean pairwise similarity is  If two responses express essentially the same claim, their embeddings will be similar, otherwise not. Average pairwise similarity is then converted into an instability score using one minus the mean similarity, with higher values then depicting greater variation. So, basically now we can study the instability in model responses under various categories of questions.

The figure below shows the semantic instability of the twelve questions. Self referential questions show highest instability followed by open ended questions followed by close ended questions.

## Conclusion

Self-referential questions are more unstable than unresolvable philosophical questions followed by verifiable standard fixed questions. These results provide a quantitative baseline for future research on self-referential reports on LLM.

Thank you if you read till the end.
