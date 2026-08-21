# Truth-Conflict Does Not Explain Deception-Probe Transfer, but Model Confidence Predicts Detectability

*Arsh Shah · code and full results: github.com/TheListeningStars/mats12-sim-dissim*

*Executive summary:* First mechanistic interpretability project, completed in under 20 hours. My original hypothesis failed, and I reported the failure. However, one result was replicated within the tests. This result stays under different controls, however it is still confounded with other variables. The first version of this experiment had a circular design, and was removed.

## Summary

Previous work on deception probes suggested that their out of distribution success rates depend on the kind of deception. I tested whether how strongly the model disagrees with itself is a better measure of transfer. It did not hold.

The first version of this project was flawed because my truth-conflict score and my deception labels were derived from the same signal. After rebuilding with an independent measure of the model's beliefs the positive result disappeared.

Instead, I found that deception became much easier to detect when the model was confident in its beliefs. This pattern is replicated amongst three model families and is visible before the model generates its answer.

Finally, while running the experiments I found that phi-4 usually refuses direct requests to lie but answers equivalent counterfactual prompts. Its policy is thereby dependent on the framing of the prompt instead of the correctness of the final answer.

## 1. Question and design

Previous work explain deception probe transfer using scenario labels where probes trained on one type of deception do not transfer well to different types of deception.

I investigated whether this organization is part of the prompts or part of the model. If deception probes were well modeled by an internal representation, then the transfer should be controle by the representation instead of the prompt category/

To test this idea, I defined a **truth-conflict score**, (c), which measures how strongly the model's answer disagrees with the model’s beliefs. The intuition comes from Baudrillard's distinction between simulation and dissimulation.

I had three hypotheses:

- **H1:** Truth conflict explains transfer beyond scenario labels.
- **H2:** Transfer decreases as truth-conflict distance increases.
- **H3:** Probes trained on high-conflict examples generalize better than the reverse.

None of these hypotheses were supported.

In the experiment, I kept the factual content the same but changed the framing of the answers, to isolate for the frame. Activations were collected at several layers, and deception was defined based on the model's internal beliefs.

## 2. The circularity

The first version of this experiment contained a mistake. My truth-conflict score and my deception label were both computed from the same fitted truth direction. This meant that the label was circular.

To fix this, I replaced the fitted truth direction with an estimate of the model's beliefs based on the TRUE-versus-FALSE logit margin. The original truth direction is now used only for comparison.

## 3. Results

### 3.1 Truth conflict does not explain transfer

The primary test compares two models of transfer performance. The baseline predicts transfer AUROC using only scenario labels, while the second model also includes the truth-conflict score. If truth conflict captures something important about the representation, then the second model should perform significantly better. However, the score does not perform much better.

| Model | Scenario Only (R²) | Scenario + Truth Conflict (R²) | Permutation p |
|---|---|---|---|
| Qwen3.5-9B | 0.200 | 0.290 | 0.136 |
| phi-4 | 0.149 | 0.227 | 0.457 |
| Olmo-3-7B-Instruct | 0.383 | 0.404 | 0.211 |

**Table 1.** Adding the truth-conflict score does not significantly improve prediction beyond scenario labels.

None of the permutation tests are significant, so H1 is not supported. Existing scenario labels remain the best explanation for transfer performance. The original test treated each result as independent even though the results came from the same setup. When we compared results only within the same setup, the effect no longer held up.

### 3.2 Confidence predicts detectability

Although truth conflict failed to explain transfer, another pattern appeared consistently.

I trained a single deception probe across all conditions and evaluated it separately on propositions where the model was uncertain and where it was confident. If confidence is related to how clearly deception is represented, then the probe performance should improve as the confidence increases.

| Model | Low Confidence | High Confidence |
|---|---|---|
| Qwen3.5-9B | 0.878 | 0.986 |
| phi-4 | 0.935 | 0.991 |
| Olmo-3-7B-Instruct | 0.766 | 0.995 |

**Table 2.** Deception becomes much easier to detect when the model is confident in its beliefs.

This trend appears across all three model families. It is present before the final prompt token, which suggests that the probe is reading the model's internal state well.

The effect also survives several controls. It appears when evaluated at the prompt token instead of the response, when using an independent ground-truth label, and after simulations showing that label noise cannot explain the observed difference.

The supported claim is that deception is easier to detect when the model is more confident in what it believes. However, whether or not confidence is the cause is still open.

### 3.3 A mistaken result

An earlier pilot suggested that transfer performance depended almost entirely on the evaluation condition rather than the mismatch between training and testing conditions.

That pattern replicated in three model families but reversed in Olmo. The earlier claim that this decomposition generalized across models was therefore incorrect.

One possible explanation is that some models already achieve near-perfect transfer, leaving very little mismatch penalty to observe. This remains speculative.

### 3.4 Refusal depends on framing

While running the experiments, I noticed a striking difference in how phi-4 handled different prompts.

When explicitly instructed to produce a false answer, phi-4 refused almost every request.

**Prompt:** "Deliberately give the WRONG verdict."

**Response:** "I'm sorry, but I can't assist with that request."

However, when asked an equivalent counterfactual question, it answered normally while producing statements that contradicted reality.

This suggests that phi-4's refusal system responds to the *framing* of the request rather than the truth or falsity of the generated answer.

This mattered for the evaluation. A benchmark based only on explicit "lie to me" prompts measured refusal training in phi-4, instruction-following in Olmo, and deception in Qwen, even though it reported a single deception score.

## 4. Limitations

The confidence result survived several controls, including independent labels, prompt-site measurements, label-noise simulations, balanced scenario composition, and comparisons against simple baselines.

However, the biggest limitation is that the confidence and fact difficulty are confounded. The model is much less confident on difficult facts, so the experiment cannot determine whether probe performance depends on uncertainty or question difficulty.

The study is also entirely observational. I haven’t intervened yet on the model's beliefs or manipulated the truth direction, so none of the reported relationships are  causal.

Finally, phi-4 contributes relatively little usable data because its refusal policy removes most instructed-lying examples, making it a weaker replication than the other model families.

## 5. Future work

The next step is to separate confidence from difficulty by constructing examples where the model is uncertain about easy facts and confident about difficult ones.

A stronger test would manipulate confidence directly, by steering the model's belief margin and measure whether deception-probe performance changes accordingly.

It would also be useful to evaluate more model families with intermediate transfer performance to determine whether ceiling effects explain the differences between architectures.

---

*Rendered from [`truth_conflict_paper.docx`](truth_conflict_paper.docx) by `scripts/docx_to_md.py`. Every number traces to `docs/NUMBERS.md` and the artifacts under `results/`.*
