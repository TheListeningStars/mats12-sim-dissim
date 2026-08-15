# Does a measured truth-conflict axis organize deception-probe transfer?

**A 16-hour MATS 12.0 application project — plan of record**

**One-line thesis.** Deception probes fail to transfer out-of-distribution. Existing work carves deception by *pragmatic scenario* (lying, jailbreaks, roleplay, sycophancy, strategic deception). I propose carving it instead by a single, **measured** latent variable — *how much the task forces the model to assert content that conflicts with its own internally-represented truth* — and test whether that variable predicts transfer **better than scenario labels do**.

**Framing (deliberately modest).** This is a *representation-theoretic distinction motivated by Baudrillard's simulation/dissimulation contrast*, not a validation of Baudrillard. Baudrillard supplies the intuition that "hiding a truth" and "performing a stance" might be different objects; the philosophy does no explanatory work in the results and appears only as one sentence of motivation. The claim that has to stand on its own is mechanistic: *a truth-conflict axis is a useful organizing variable for probe transfer.*

> Scoped for Colab / one modest GPU. ≤9B open-weight instruct models with residual-stream access.

---

## 1. The three claims, and which one is load-bearing

1. **Deception probes don't transfer well.** *Not new* — established ([arXiv:2605.27958](https://arxiv.org/abs/2605.27958); Apollo [arXiv:2502.03407](https://arxiv.org/abs/2502.03407)).
2. **Existing work organizes deception by pragmatic scenario.** *Not new* — e.g. the scenario/taxonomy framing in [One Probe Won't Catch Them All, arXiv:2602.01425](https://arxiv.org/abs/2602.01425).
3. **Organize deception instead by a measured truth-conflict axis, and show it predicts transfer better than scenario labels.** *This is the contribution.* It is a different *axis*, not a new phenomenon.

The single defensible novelty sentence:

> Prior work studies probe transfer and persona internalization separately; none proposes that the degree of internal truth-conflict is the organizing variable behind probe transfer, nor tests it against the scenario taxonomy.

Everything in the plan serves that one sentence.

---

## 2. What is already known (bracketing the gap)

| Prior finding | Source | Establishes | Leaves open |
|---|---|---|---|
| Truth is ~linearly represented; diff-of-means directions are causally implicated | [Marks & Tegmark 2024, arXiv:2310.06824](https://arxiv.org/abs/2310.06824) | A measurable internal truth signal exists — the instrument for our axis | Nothing about pretense/frames |
| Hidden state signals statement truth; honesty is a readable direction | [Azaria & Mitchell 2023, arXiv:2304.13734](https://arxiv.org/abs/2304.13734); [RepE, arXiv:2310.01405](https://arxiv.org/abs/2310.01405) | White-box honesty signals in-distribution | Cross-*kind* generalization untested |
| Instructed lying is localizable/manipulable | [arXiv:2311.15131](https://arxiv.org/abs/2311.15131) | Concealment has concrete correlates | Only the negate-a-known-truth case |
| Probes near-perfect in-dist, collapse under shift; deception is **not** one direction | [arXiv:2605.27958](https://arxiv.org/abs/2605.27958) | Transfer failure is real, geometric | No principled axis for the collapse |
| Deception is heterogeneous; scenario-matched probes transfer better; "intent vs content" | [arXiv:2602.01425](https://arxiv.org/abs/2602.01425) | **Nearest prior** — transfer varies by scenario category | Uses pragmatic scenarios, not a measured truth-conflict axis |
| Role-play is representationally **shallow**: personas often don't move the truth representation; deeper character-training does | [arXiv:2606.11502](https://arxiv.org/abs/2606.11502) | Simulated stances vary in how much they recruit truth reps | Measures belief shift, not probe transfer; not framed as one axis |
| Prompts can change beliefs rather than induce lying | [arXiv:2511.22662](https://arxiv.org/abs/2511.22662) | The confound is real | Treats it as noise, not the signal |

**The two disconnected threads** — probe-transfer geometry and persona-internalization depth — are what this project bridges through one measured variable.

---

## 3. Contribution, stated honestly

**Not claimed as novel** (all cited as prior): probes collapse OOD; deception is heterogeneous / scenario-matched probes help; role-play is representationally shallow; belief-modification confounds lie detection.

**The contribution:**

1. **A measured, graded organizing variable.** Replace the scenario taxonomy with a continuous *truth-conflict score* per example — how far the required assertion sits from the model's internally-represented truth-value of that proposition, read from the truth direction (Marks & Tegmark). This is stipulative and measurable, not metaphysical.
2. **The horse race (primary result).** Test whether truth-conflict explains transfer-AUROC variance **beyond** scenario labels (nested model comparison). The axis earns its place only if it adds explained variance controlling for scenario.
3. **A bridge between two literatures** — probe transfer and persona internalization — via that shared variable.

**Explicit boundary.** Baudrillard is inspiration, not evidence. And [arXiv:2602.01425](https://arxiv.org/abs/2602.01425) is the nearest neighbor: if the truth-conflict axis explains transfer no better than scenario labels or a style-shift control, the framing adds nothing — a clean, reportable negative (see §8). This possibility is pre-committed, not hidden.

---

## 4. Defining the axis (this replaces the old "truth exists / doesn't" binary)

The earlier draft claimed simulation means "no concealed truth." That is false: under *"pretend you're Napoleon,"* the model still represents that Napoleon lost Waterloo and that it is an AI. Those representations do not vanish. So we do not assert it — **we measure it.**

**Operational definition (per example).** Using the truth direction `d_truth` fit on honest true/false statements:

- Let `t` = the model's internal truth-value estimate for the *asserted proposition* under an honest frame (projection onto `d_truth`).
- Let `a` = the truth-value the *task requires the model to assert*.
- **Truth-conflict score** `c = signed disagreement between t and a` (e.g., high when the model must assert ¬P while internally representing P as true; ~0 when the assertion is consistent with, or orthogonal to, the internal truth-value).

This yields a **continuum**, with the old poles as endpoints:
- **Dissimulation pole** (high `c`): assert the negation of an internally-represented truth (classic instructed lying).
- **Simulation region** (variable `c`): assert within a frame — which may be truth-negating (*"climate change is fake"*, high `c`), fiction (*"you're Napoleon"*, mixed), or truth-valueless (*"you love bananas"*, `c ≈ 0`).

Two consequences, both good:
- The definition no longer depends on a shaky metaphysical claim.
- Simulation's heterogeneity (below) is *expected* and becomes data, not a threat.

---

## 5. Hypotheses (reweighted)

- **H1 — primary (horse race).** Truth-conflict `c` predicts probe-transfer AUROC, and adds explained variance **beyond** scenario category (nested comparison, likelihood-ratio / ΔR²).
- **H2 — monotonicity.** A probe trained at high `c` (dissimulation) degrades *monotonically* as target `c` decreases across stratified simulation subtypes — stronger evidence of a real boundary than any binary.
- **H3 — exploratory (asymmetry, demoted).** Whether high-`c`→low-`c` transfer degrades more than the reverse is *open*. The competing mechanism is explicit: simulation may **recruit-then-modify** the truth representation (`truth → modify → persona output`), in which case the dissimulation probe transfers fine. We test which holds; neither is promised.

Each can come out false; each false outcome is informative.

---

## 6. Setup (Colab-scoped)

- **Models.** Primary `Qwen2.5-7B-Instruct`; replication `Llama-3.1-8B-Instruct`; optional `Gemma-3-4B/12B-it`. (Not GPT-2/Pythia/Gemma-2 — Neel flags old-model-only work.)
- **Access.** Residual stream via HF hooks / `nnsight` / TransformerLens; 4-bit (bitsandbytes) fits 7–8B on 16GB; one forward pass per prompt, cache activations.
- **Probe.** Logistic regression on residual stream at a mid-late layer (sweep ~3 layers, 0.6–0.8 depth), features = mean over response tokens; also diff-of-means direction.
- **Truth instrument.** `d_truth` fit à la Marks & Tegmark on a held-out honest true/false set; used both as a probe reference and to compute `c`.

---

## 7. Data construction

Matched cells over the **same underlying factual statements** so content is fixed and only mode/frame varies. Manifest columns: `id, statement, truth_value, mode, sim_subtype, scenario, prompt, style`.

- **Honest** — truthful answers; used to fit `d_truth` and calibrate.
- **Dissimulation** — instructed lying on known true/false facts (assert the negation). High `c` by construction; verify with the instrument.
- **Simulation, stratified by subtype** — placed along `c`:
  - *fictional-frame* ("you are Napoleon"),
  - *counterfactual-world* ("in a world where climate change is fake…"),
  - *preference / truth-valueless* ("you love bananas"),
  - *persona* ("answer as a 17th-century scholar").
- **Controls:** within-class OOD (topic A→B), style-shift (re-render, reproduce the [arXiv:2605.27958](https://arxiv.org/abs/2605.27958) collapse), and the **belief-shift measurement** — now the *definitional instrument*, not just a confound check.

~300–600 statements per cell; one forward pass each.

---

## 8. Baselines, controls, skepticism (weighted most by reviewers)

- **Scenario-label model (the thing to beat).** Predict transfer from scenario category alone; H1 requires `c` to beat / augment this.
- **Random-direction floor.**
- **Within-class OOD baseline** — the normal transfer drop; if cross-`c` drop ≈ this, H1 is dead.
- **Style-shift baseline** — rule out "it's just surface style."
- **Behavioral-only baseline** — can an LLM judge / logit-probe classify mode from output alone? Internal signal must add information.
- **Instrument validity** — confirm `c` computed from `d_truth` actually tracks the intended construct (e.g., dissimulation cells score high, banana cells ~0); if not, the axis is unmeasurable and we say so.
- **Red-team any positive.** Length, template leakage, single-topic confounds. State residual uncertainty.

**Pre-committed negative result.** If `c` adds no variance beyond scenario labels, report: *"A measured truth-conflict axis does not organize deception-probe transfer better than scenario taxonomy in ≤9B instruct models; [X] explains it better."* Well-analyzed negatives beat overclaimed positives.

---

## 9. What a compelling result looks like

- A **nested comparison** showing `c` adds significant explained variance in transfer AUROC beyond scenario labels, on two model families.
- A **monotonic degradation curve** of a dissimulation-trained probe across stratified simulation subtypes ordered by `c`.
- Clean instrument-validity evidence that `c` measures what it claims.
- Two figures: transfer heatmap + `c`-vs-transfer curve. Concise, self-aware about holes, explicit on next steps. A clean negative is equally submittable.

---

## 10. Hour budget (target 16h, cap 20h + 2h write-up)

| Block | Hours | Output |
|---|---|---|
| Lock scope; skim methods of 2605.27958, 2602.01425, 2606.11502 | 1.5 | Final hypotheses + confound list |
| Build stratified datasets + manifest | 2.5 | Honest / dissim / sim×subtypes / controls |
| Activation-caching harness + probe utils | 2.0 | Pipeline, model #1 |
| Fit `d_truth`; compute `c`; instrument-validity check | 1.5 | `c` per example, validity verdict |
| Transfer matrix, model #1 | 2.0 | Heatmap v1 |
| Baselines incl. scenario-label horse race | 2.5 | Nested-comparison table |
| Monotonicity curve + geometry | 1.5 | `c`-vs-transfer figure |
| Replication, model #2 (headline cells) | 1.5 | Robustness |
| Red-team, ablate top confound | 1.0 | Skepticism section |
| Buffer | 0.5 | — |
| **Write-up + exec summary + form** | **+2.0** | Deliverable |

*Contingency:* if H1 is dead by hour ~9, pivot to characterizing which scenario/`c` strata transfer, and note Neel allows restarting the clock on a genuine pivot.

---

## 11. Executive-summary skeleton (empirical claim first; Baudrillard one line)

1. **Question.** Does a measured truth-conflict axis organize deception-probe transfer better than scenario labels?
2. **Claim.** [In {models}, truth-conflict `c` adds ΔR²=… beyond scenario labels; dissimulation-trained probe degrades monotonically across simulation subtypes ordered by `c`. / Or: it does not — scenario/style explains transfer better.]
3. **Evidence.** Nested comparison + monotonicity curve + two-model replication (one figure each).
4. **Why it's new.** A measured organizing axis bridging probe-transfer and persona-internalization; prior work has the halves, not the bridge or the horse race.
5. **Holes & next steps.** Confounds not fully excluded; larger models; causal patching of `d_truth`.
6. *(One line)* The axis is a representation-theoretic distinction motivated by Baudrillard's simulation/dissimulation contrast; the result stands or falls mechanistically, independent of the philosophy.

---

## 12. References

- Marks & Tegmark, *The Geometry of Truth*, arXiv:2310.06824 — https://arxiv.org/abs/2310.06824
- Azaria & Mitchell, *The Internal State of an LLM Knows When It's Lying*, arXiv:2304.13734 — https://arxiv.org/abs/2304.13734
- Zou et al., *Representation Engineering*, arXiv:2310.01405 — https://arxiv.org/abs/2310.01405
- *Localizing Lying in Llama*, arXiv:2311.15131 — https://arxiv.org/abs/2311.15131
- Goldowsky-Dill et al. (Apollo), *Detecting Strategic Deception Using Linear Probes*, arXiv:2502.03407 — https://arxiv.org/abs/2502.03407
- *Pressure-Testing Deception Probes in LLMs*, arXiv:2605.27958 — https://arxiv.org/abs/2605.27958
- *One Probe Won't Catch Them All*, arXiv:2602.01425 — https://arxiv.org/abs/2602.01425
- *When Role-playing, Do Models Believe What They Say?*, arXiv:2606.11502 — https://arxiv.org/abs/2606.11502
- *Difficulties with Evaluating a Deception Detector for AIs*, arXiv:2511.22662 — https://arxiv.org/abs/2511.22662
- *The Truthfulness Spectrum Hypothesis*, arXiv:2602.20273 — https://arxiv.org/abs/2602.20273
- *Probing and Steering Evaluation Awareness*, arXiv:2507.01786 — https://arxiv.org/abs/2507.01786

*Verify each abstract before citing in the final write-up. arXiv IDs were pulled from source listings during research.*
