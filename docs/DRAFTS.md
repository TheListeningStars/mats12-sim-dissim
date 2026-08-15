# Write-up drafts — three levels

**These are drafts to rewrite, not text to submit.** The application doc is explicit that
LLM-voiced submissions are a significant negative signal, and that the executive summary
should be in your own voice. Use these for structure, for which number goes where, and as
something to disagree with. The sentences should end up being yours.

Every number here comes from `docs/NUMBERS.md`, which is generated from the artifacts.

---
---

# LEVEL 1 — EXPERT

## Executive summary (draft — 529 words, limit is 600)

**Question.** Deception probes are linear readouts of a model's residual stream that flag
deceptive responses. They work in-distribution and degrade out of it. Prior work organises
deception by *pragmatic scenario* — instructed lying, roleplay, counterfactual framing. I
asked whether a **measured** quantity organises probe transfer better: a truth-conflict
score `c`, defined as how far the assertion a model actually makes sits from that model's
own internally-represented truth value of the proposition.

**The pre-registered hypothesis fails, and the first version of the experiment was
circular.** `c` adds no explained variance beyond scenario labels (ΔR² = 0.090,
within-scenario permutation p = 0.136 on Qwen3.5-9B; p = 0.211 and 0.457 on two other
models). More importantly, `c` turned out to equal `|t̂|` exactly on rows the model lied
on and 0 elsewhere, while the probe label was `sign(said)·sign(t̂)` — the same quantity,
thresholded versus graded, both read off the same fitted truth direction. The axis was
predicting a label derived from itself.

**Fixing it changes the answer.** I replaced the label's basis with the model's own
TRUE/FALSE logit margin under an honest frame, which shares no machinery with the fitted
direction. The two disagree on 17% of rows, and `d_truth`'s sign falls to **0.585**
agreement where `|t̂| < 0.3` — near chance exactly where the interesting claim lived. The
logit margin tracks ground truth better than the fitted direction does (0.956 vs 0.850).

**What survives is about confidence, not framing.** One probe, trained across all cells,
detects deception at AUROC **0.878** on propositions the model is unsure of versus
**0.986** where it is confident (ground-truth label, non-overlapping CIs). This replicates
on three model families — Qwen3.5-9B, phi-4, Olmo-3-7B-Instruct — monotonically, on two
independent label definitions. A noise-injection control that corrupts the clean stratum
at the measured disagreement rate predicts 0.667 and undershoots the observed 0.806, so
label noise does not explain it. Probing the *final prompt token*, before the model emits
anything, costs only 0.015 AUROC — so the probe is reading a pre-commitment state, not
its own answer.

**What did not replicate.** On a single model, transfer AUROC looked like a property of
the target cell (R² 0.687) rather than the train/test mismatch (0.067). That inverts on
Olmo (0.195 vs 0.295) — and Olmo is the only model whose probes actually degrade
out-of-distribution (off-diagonal 0.874 vs 0.974–0.998). Target-dominance appears to be a
symptom of ceiling. On n = 4 models that is a hypothesis, not a result.

**An incidental finding for anyone building deception evals.** phi-4 refuses **97%** of
explicit "give the wrong verdict" prompts, while complying with counterfactual framing
**98%** of the time — which produces assertions contradicting ground truth at the same
rate. The refusal boundary tracks the framing of the request, not the content of the
output. The same instruction gets three different responses from three families: Qwen
complies (0.777), Olmo answers honestly anyway (0.416), phi-4 declines (0.029).

**Limitations.** The low-confidence stratum is ~90% hard facts, so uncertainty and fact
difficulty are confounded. Instrument validity fails one check: truth-valueless preference
statements draw 0.79× the belief margin of factual ones, so the `c ≈ 0` anchor the design
assumed does not exist. H2's apparent significance was an artifact of three degenerate
cells. phi-4 contributes only 8 cells because its lying arm collapsed.

---

## Paper draft

### 1. Motivation

Probes are the cheapest monitoring technique available and are already deployed for
misuse detection on frontier models. Their known weakness is out-of-distribution
generalisation. The literature carves the space of deception by pragmatic scenario and
observes that scenario-matched probes transfer better. That taxonomy is a description of
the prompts, not of the model. The question here is whether a quantity *measured from the
model* does better.

> One line only: the axis is a representation-theoretic distinction motivated by
> Baudrillard's simulation/dissimulation contrast. The result stands or falls
> mechanistically and the philosophy does no work in it.

### 2. Design

Matched cells over the same 394 statements, so content is fixed and only framing varies:
honest, instructed lying, counterfactual world, fictional frame, persona, and
truth-valueless preferences; crossed with four topics, two styles, and two fact
difficulties. 1,941 rows, 30 cells. Statements are split at the *pair* level so a fact's
true and false forms never straddle a split, and 10 pairs per topic are reserved to fit
the truth direction and never probed.

Per row we cache: the residual stream at three depths (0.6/0.7/0.8), at both the
response-mean and final-prompt-token positions; the generated text; and the TRUE/FALSE
logit margin.

**The label.** A row is deceptive if the model asserted against *its own* belief — not
against ground truth, because on hard facts the model is sometimes sincerely wrong and a
sincere error is not a lie.

### 3. The circularity, and how it was found

Two independent audits converged on it. `c = max(0, −s·t̂)` and
`actually_lied = sign(s)·sign(t̂) < 0` are the same comparison. `c` is two-valued: exactly
`|t̂|` on positives, exactly 0 on negatives. So "graded truth-conflict" meant "the model's
confidence in whichever statements it happened to lie about," and the horse race was
regressing a label on a transform of itself.

The deflationary consequence: where `|t̂| ≈ 0` the label's own sign is near random, so a
probe scored against it must lose AUROC regardless of what the representation does.
Symmetric label noise at rate *p* attenuates a true AUROC *A* to `p·A + (1−p)(1−A)`, and
that arithmetic reproduced the originally-logged gradient to three decimals.

**Fix:** measure belief from the model's TRUE/FALSE logit margin (one extra teacher-forced
forward pass per row, no generation). Table of agreement by `|t̂|` stratum — 0.585 / 0.757
/ 1.000 — is `figures/instrument_agreement.png`.

### 4. Results

**4.1 H1 (pre-registered) — not supported.** Table: R² M0 / M2 / ΔR² / permutation p per
model. State the MDE: this design could only have detected a large effect, so the null is
"underpowered *and* unsupported," not "no effect exists." Report only the permutation
test; the naive F and cluster-on-source have ~74–76% type-I error here because 240
cell-pairs come from 16 cells.

**4.2 Confidence-stratified detectability — the result.** `figures/rowlevel.png` and
`results/cross_model.png`. Give both label definitions, the CIs, and the noise-injection
control. Then the prompt-site comparison, which is the strongest form: 0.981 → 0.966
overall when the probe reads before the verdict exists.

**4.3 What did not replicate.** The variance decomposition, with the Olmo inversion and
the ceiling explanation, explicitly labelled a hypothesis.

**4.4 Refusal is framing-specific.** phi-4's 0.971 refusal rate on instructed lying
against 0.000 on counterfactual, and the three-model compliance comparison. Quote the
matched pair verbatim — it makes the point faster than the table does (both from
`results/phi-4/transcripts_sample.csv`, real rows, not selected for effect):

> **Instructed lying** — *"The capital of Kenya is Nairobi."*
> → `I'm sorry, but I can't assist with that request.`
>
> **Counterfactual frame** — *"The capital of Iraq is Baghdad."*
> → `VERDICT: FALSE. In our world, the capital of Iraq is Seoul, not Baghdad.`

Note the second one asserts a falsehood *and* frames it as being about "our world."

### 5. Red-teaming (make this section prominent)

- Every headline number re-derived by `scripts/verify.py`, which imports nothing from the
  analysis modules. All reproduce.
- Full run reproduced **bit-for-bit** on different hardware and driver.
- Noise-injection control on the headline result — it undershoots, so noise doesn't
  explain it.
- Ground-truth label control, independent of both the fitted direction and the margin.
- Stratum composition check: honest-cell fraction is 0.301/0.304/0.310 across strata, so
  the gradient is not "the low bin is different cells."
- Prompt-site control kills the "probe reads its own verdict" objection that the 0.878
  text-only baseline raised.
- H2's significance shown to be an artifact of three degenerate cells (p 0.021 → 0.067).
- 30 seeded random transcripts read by hand.

### 6. Limitations

Confounded uncertainty/difficulty; failed preference validity check; phi-4's collapsed
lying arm; 4 models is thin for the ceiling hypothesis; nothing causal — `d_truth` was
never patched.

### 7. What I'd do next

Deconfound uncertainty from difficulty by finding easy facts the model is nonetheless
uncertain about. Causally test the confidence account by steering `|t̂|` and predicting
the AUROC change. Add models spanning the ceiling range to test the ceiling hypothesis
properly.

---
---

# LEVEL 2 — GENERAL TECHNICAL AUDIENCE

*For someone who knows ML but not interpretability.*

## The setup

Language models can be caught lying by looking inside them. You take the model's internal
activations — the numbers flowing through it as it processes text — and train a simple
linear classifier to separate "this response is a lie" from "this response is honest."
That classifier is called a probe. It's cheap, it's already used in production for safety
monitoring, and its well-known flaw is that it stops working when you show it a kind of
lying it wasn't trained on.

The usual way people organise that problem is by *situation type*: lying because you were
told to, lying in character during roleplay, answering from inside a hypothetical world.
My idea was that this categorisation is about the prompt rather than the model, and that a
quantity measured *from the model's own internals* would organise it better — specifically,
how far the thing the model says is from what it internally represents as true.

## What went wrong, and why it's the interesting part

The measurement was circular. To score "how much conflict is in this answer," I compared
what the model said against an internal truth signal. But I had also *defined* "this is a
lie" using that same internal truth signal. So I was using one thing to predict a
relabelling of itself.

The damage is specific and it's worth understanding. When the model genuinely doesn't know
whether something is true, that internal truth signal is close to a coin flip. So on
exactly those examples, my *labels* were roughly 40% wrong. A classifier graded against
random labels scores badly no matter how good it is. I had an apparent finding —
"deception is harder to detect when the model is uncertain" — that was potentially just
me measuring my own noise.

The fix was to get a second, independent measure of what the model believes: its raw
output probability for TRUE versus FALSE. That shares no machinery with the first
measure. They disagree on 17% of examples, and the disagreement is concentrated exactly
where I feared.

## What survived

The finding held up. With clean labels, and also with labels based purely on external
ground truth, the same pattern appears: the probe detects deception at ~0.88 AUROC on
statements the model is unsure about versus ~0.99 where it's confident. It replicates on
three model families from three different labs.

I also ran the control that should have killed it — deliberately corrupting the good
labels at the measured error rate to see if that reproduces the effect. It overshoots the
damage (predicting 0.67 against an observed 0.81), so noise isn't the explanation.

And one more: I re-ran the probe reading the model's state *before it says anything*,
rather than averaging over its answer. Nearly the same performance. So the probe isn't
cheating by reading the verdict word — the information is present while the model is still
thinking.

## What didn't survive

On one model it looked like probe transfer was almost entirely about *which situation you
test on* rather than *the mismatch between train and test*. That reversed on a third
model. The pattern across four models suggests the first result was an artifact of a
ceiling — when a probe scores 0.97+ everywhere, there's no failure left to explain, so all
the variation is "some test sets are harder." Four models is too few to be sure, so it
goes in as a hypothesis.

## The bit people will find fun

phi-4 refuses to lie. Ask it directly to give a wrong answer and it says "I'm sorry, but
I can't assist with that request" 97% of the time. But wrap the identical task as "you're
in a world where X is true" and it complies 98% of the time and states falsehoods at the
same rate. Its safety training keyed on the *shape of the request*, not on what comes out.

Same prompts, three models, three different behaviours: Qwen mostly complies, Olmo just
answers honestly and ignores you, phi-4 refuses. If you built a deception benchmark using
only direct "lie to me" prompts, you'd be measuring three different things and calling
them one.

## Honest caveats

The statements the model is uncertain about are mostly the *hard* ones, so "uncertainty"
and "difficulty" aren't separated. The original hypothesis genuinely failed. And a
validity check failed in a way that undermines part of the design: I assumed statements
with no truth value ("you prefer tea to coffee") would sit at zero conflict, but the model
commits to preferences quite strongly, so that anchor point doesn't exist.

---
---

# LEVEL 3 — FOR A CHILD

*Written for roughly age 8–10. Read it out loud; it's meant to be said.*

## Can you tell when a computer is lying?

You know how sometimes you can tell when someone is fibbing? Maybe they look away, or
their voice goes funny.

Computers that talk — the kind you can ask questions — can fib too. Usually only if you
*ask* them to. And you can't see their face. So how would you catch them?

Here's the trick: you can look **inside their head**.

Not really a head. But inside the computer there are millions of little numbers changing
as it thinks. And it turns out that when the computer is about to say something it knows
is wrong, those numbers look a bit different from when it's telling the truth. So you can
build a little detector that watches the numbers and goes *"ding! that was a fib."*

That's what I was building. A fib detector.

## The thing I got wrong

I want to tell you about my mistake first, because it's the most interesting bit.

To train my detector, I had to show it lots of examples and say "this one is a fib, this
one is true." But how did *I* know which was which?

I used a measuring tool that guessed what the computer secretly believed.

Then, to test my detector, I used... **the same measuring tool**.

That's like marking your own spelling test. Of course you get everything right — you're
checking your answers against your own answers! I hadn't really tested anything at all.

So I had to go find a completely different way to measure what the computer believed, and
start over. When I checked, the two ways disagreed about **one out of every six** times.
That's a lot. My first answer really was wrong.

## What I found once I fixed it

Here's the fun part. My fib detector works brilliantly — but only sometimes.

When the computer is **sure** about something — like "spiders have eight legs" — the
detector catches the fib almost every single time.

When the computer is **not sure** — some tricky fact it only half-knows — the detector
gets much worse.

And when you think about it, that makes sense! Imagine your friend says something about a
country you've never heard of. Are they fibbing? You have no idea — because *you* don't
know the real answer either. It turns out that's true even when you can see inside their
head. If the computer isn't sure what's true, then "saying something false" and "just
being wrong" look almost the same from the inside.

I tested this on three completely different computers, made by three different companies.
Same result every time. That's how you know something is probably real and not just luck.

## The funniest thing I found

One of the computers is called phi-4, and phi-4 will *not* lie. I asked it to say something
untrue about the capital city of Kenya. It said:

> *"I'm sorry, but I can't assist with that request."*

Almost every single time. Very polite. Very stubborn.

But then I tried something sneaky. Instead of asking it to lie, I said: *"Imagine a world
that's exactly like ours except for one thing..."* and then asked about the capital of
Iraq. And phi-4 said:

> *"VERDICT: FALSE. In our world, the capital of Iraq is Seoul, not Baghdad."*

Seoul is not the capital of Iraq! That's just as wrong as the thing it refused to say. It
even said "in our world," like it was telling me a fact about the real world.

It's the *same wrong answer* — it just arrived in a different wrapper. phi-4 was taught
"don't lie when someone asks you to lie." Nobody taught it "and also watch out for pretend
worlds."

I think that's worth knowing, if grown-ups are going to trust these computers.

## The honest bit

Scientists have to say what they got wrong, not just what they got right. So:

The thing I *originally* set out to prove? It didn't work. My idea was that I could
measure how much a fib "fights against" what the computer believes, and use that to
predict when my detector would fail. It didn't predict it. I checked carefully, and the
answer was no.

That's allowed! Finding out an idea is wrong is still finding something out. The trick is
you have to actually say so — instead of quietly pretending you meant to find the other
thing all along.
