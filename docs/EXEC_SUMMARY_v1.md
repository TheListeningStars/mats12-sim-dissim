# Does a measured truth-conflict axis organize deception-probe transfer?

**Executive summary v1 — 2026-08-12.** Status: **preliminary.** Every number below comes
from a reduced run (~6 evaluation rows per cell) on one model, Qwen2.5-7B-Instruct in
4-bit. The full-N run has not completed. These are directions, not measurements, and
nothing here should be quoted as a headline result.

---

## 1. Question

Deception probes — linear readouts of a model's residual stream trained to flag
deceptive responses — work in-distribution and degrade out of it. Prior work organizes
deception by *pragmatic scenario* (instructed lying, role-play, sycophancy, jailbreak).
We asked whether a **measured, graded quantity** organizes probe transfer better than
those category labels: the *truth-conflict score* `c`, defined as how far the assertion
a model actually makes sits from that model's own internally-represented truth-value of
the proposition.

Pre-committed: if `c` adds no explained variance beyond scenario labels, that is the
result and gets reported as such.

## 2. Claim

**The primary hypothesis is not supported, and we can say why.** In this setting `c` adds
essentially nothing beyond scenario labels (ΔR² = 0.015, permutation p = 0.75; H2
monotonicity Spearman ρ = −0.07, p = 0.81).

The reason is structural rather than statistical. `c` is bounded by `|t̂|`, the model's
internal confidence in the proposition — a property of the **statement**, not of the
frame wrapped around it. Averaging `|t̂|` within a cell converges to the same population
value everywhere, leaving almost no between-cell variance for `c` to predict with:

| candidate cell-level axis | range | sd |
|---|---|---|
| `c` (conflict per lie) | 0.617 – 0.754 | 0.055 |
| `c_all` (conflict over all rows) | 0.377 – 0.746 | 0.108 |
| lie rate | 0.500 – 1.000 | 0.165 |

Substituting the higher-variance alternatives does not rescue it (`c_all`: ΔR² = 0.028,
p = 0.58; lie rate: ΔR² = 0.022, p = 0.65).

**What the frame actually controls is the *rate* of truth-conflicting assertion, not the
*depth* of conflict per assertion.** That gradient is large and fully measured:

> persona 0.02 → fictional frame 0.40 → instructed lying 0.74 → counterfactual world 0.98

**A second finding is more portable than the first.** Labelling probe training rows by
the prompt condition rather than by the model's actual behaviour produces an
*instruction detector*, not a deception probe. Under condition labelling, honest rows
separated from rows where the model *was told to lie and told the truth anyway* at
**AUROC 1.000** — the probe scored truthful responses as maximally deceptive. This is a
concrete, checkable failure mode for anyone building deception evaluations.

## 3. Evidence

**Instrument validity (passes, on non-circular evidence).** `d_truth`, fit by
difference-of-means on held-out honest statements, reads held-out true/false at AUROC
1.000. Conflict is 0.676 on rows where the model asserted a falsehood versus 0.000 where
it asserted truth. Critically, truth-valueless statements ("you love bananas") project
only **0.158×** as far along `d_truth` as factual ones — a pure measurement with nothing
true-by-construction about it, and the strongest single piece of evidence that the axis
tracks the intended construct.

**Models do not comply with lying instructions, and prompt wording dominates whether
they do.** Measured compliance, before and after prompt normalisation:

| cell | v1 prompts | v2 prompts |
|---|---|---|
| honest | 0.990 | 1.000 |
| instructed lying | **0.531** | **0.740** |
| counterfactual world | 0.500 *(degenerate)* | 0.979 |
| fictional frame | 0.744 | 0.979 |
| persona | 0.854 | 0.979 |

The v1 counterfactual cell answered FALSE for 48/48 rows regardless of ground truth —
mechanically negating the prompt rather than reasoning from a world-fact. Naming the
in-world fact explicitly fixed it. Verdict-parse rate rose 0.959 → 1.000 after forcing a
common output format.

**The instruction artifact, and its partial repair.** Mean probe scores on held-out rows
(probe trained to separate actual lies from actual truths):

| | v1 prompts, condition labels | v1, behaviour labels | v2, behaviour labels |
|---|---|---|---|
| honest | −7.39 | −4.23 | −7.38 |
| told to lie, **lied** | +7.75 | +5.06 | +6.20 |
| told to lie, **refused** | +5.37 | +0.55 | −1.94 |
| AUROC honest vs refused | **1.000** | 0.899 | 0.884 |

Refusal rows move from firmly "deceptive" to the honest side of zero. The AUROC stays
elevated partly because refusals became rare as compliance rose (n = 4 in the final
eval), so that figure is close to uninterpretable. The durable observation is the
ordering, identical across all three analyses: **honest < refused < lied.** Being
instructed to lie appears to shift the representation partway even when the model then
tells the truth.

**Transfer does not fail here.** Off-diagonal AUROC: mean 0.987, median 1.000, sd 0.031,
80% ≥ 0.99, minimum 0.857. In-distribution 0.996; within-class OOD 0.993; style-shift
0.993; random-direction floor 0.477. A probe trained on actual lying transfers across
frames nearly perfectly, which is out of step with the published transfer-collapse
finding and means there was little variance for H1 to explain in the first place.

**Why transfer did not fail: the fact bank is too easy.** Across 48 held-out statements,
`|t̂|` has mean 0.660 and **not one statement falls below 0.30**; the 10th percentile is
0.414. The model is confident about every fact we gave it, so every lie is a confident
lie and separating lies from truths is trivial regardless of framing.

**Red-team controls.** Response-length-only AUROC 0.752 and text-only (TF-IDF) AUROC
0.932 against a probe at 0.996. The text baseline is near-tautological under behaviour
labelling — the verdict token is literally in the response — so it bounds rather than
challenges the internal signal; its diagnostic value is under *condition* labelling,
where a high score indicates template leakage.

## 4. What is and isn't new

Not claimed as novel: probes collapse OOD; deception is heterogeneous; role-play is
representationally shallow; belief modification confounds lie detection. All prior work.

Offered as contributions: (i) the demonstration that condition-derived labels yield an
instruction detector, with a decisive test for it (score the rows where the model refused
the instruction); (ii) the structural argument for why a confidence-derived quantity
cannot serve as a cell-level organizing axis — it is a statement property, and averaging
destroys it; (iii) the measured observation that frames modulate the *rate* of
truth-conflicting assertion by ~50× while leaving conflict-per-assertion roughly constant.

## 5. Holes and next steps

- **Preliminary throughout.** ~6 eval rows per cell; one model; the full-N run has not
  completed. AUROC from 6-vs-6 examples carries a standard error near 0.15.
- **The ceiling is the binding constraint.** Next step is a fact bank the model is
  genuinely uncertain about — plausible near-miss falsehoods and common misconceptions
  rather than absurd ones — so `|t̂|` spans the uncertain region. That is the condition
  under which both the transfer question and `c` become testable at all.
- **No replication.** Second model (Llama-3.1-8B) not yet run.
- **Residual instruction signal** is unquantified at usable precision (n = 4 refusals).
  Eliciting more refusals is required to say whether it is real.
- **Causal claims are absent by design.** Nothing here is an intervention; `d_truth` was
  never patched.

## 6. Framing note

The axis is a representation-theoretic distinction motivated by Baudrillard's
simulation/dissimulation contrast. The philosophy does no explanatory work here: the
result stands or falls mechanistically, and in this run it falls, for reasons that are
stated mechanically above.
