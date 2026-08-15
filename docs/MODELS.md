# Which models, and how to set them up

## The short answer

| Role | Model | Access | Why this one |
|---|---|---|---|
| **Primary** | `Qwen/Qwen3.5-9B` | open | The current recommended default dense family. Running the headline on a 2024 model is a listed application mistake, and Qwen2.5 is two generations back. |
| **Replication 1** | `microsoft/phi-4` | open (MIT) | 14.7B, different lab, different pretraining mix. A replication is only informative if the second model does not share the first's training data. |
| **Replication 2** | `allenai/Olmo-3-7B-Instruct` | open (Apache-2.0) | Fully open pretraining. A third lab makes the variance-decomposition result a claim about probe-transfer studies rather than about Qwen. |
| Smoke test | `Qwen/Qwen3.5-4B` | open | Same tokenizer and chat template as the primary, loads in a couple of minutes. Use it to shake out pipeline changes before paying for the 9B. |
| Optional | `google/gemma-3-12b-it` | **gated** | A fourth family. Needs the steps below. |
| Optional | `meta-llama/Llama-3.1-8B-Instruct` | **gated** | Older (2024); only worth it as a deliberate old-vs-new comparison. |

Nothing in the first four needs a token. That was deliberate — the first replication
attempt died on a 401 at 3am, and a replication you cannot run unattended is not a
replication.

## Setting up the gated ones

Only needed for Gemma or Llama. Two separate things, and people usually miss the first:

1. **Accept the licence on the model page**, signed in as yourself:
   <https://huggingface.co/google/gemma-3-12b-it> → "Acknowledge licence".
   Access is usually instant, occasionally a manual review.
2. **Create a read token**: <https://huggingface.co/settings/tokens> → New token →
   type **Read**. Copy it; it is shown once.
3. **Put it on the pod**, not in the repo:

   ```bash
   # on the pod
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
   echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx' >> ~/.bashrc
   huggingface-cli whoami          # should print your username
   ```

   Never commit it. `.gitignore` does not currently cover a stray `token.txt`, so keep
   it in the environment only.
4. Then it runs like any other model:

   ```bash
   python scripts/preflight.py --model gemma-3-12b-it
   python scripts/run_all.py    --model gemma-3-12b-it --headline-only
   ```

If step 1 is skipped, the token alone still 401s with *"You are trying to access a
gated repo"* — which is exactly the error we hit.

## Adding any other model

One entry in `src/config.py`:

```python
MODELS = {
    ...
    "my-model": "org/My-Model-Instruct",
}
```

Then **always run preflight first**. It costs about five minutes and one model load, and
it exists because two different silent failures have already been caught by it:

```bash
python scripts/preflight.py --model my-model
```

It gates on: the checkpoint mapping to a causal-LM class, `LAYER_FRACTIONS` resolving
against the *text* stack, no `<think>` blocks surviving, verdict parse rate, activations
finite and varying, the prompt-site residual actually differing from the response mean,
and the belief margin tracking ground truth on honest rows. It prints the raw
generations too — read them, don't just check the parse rate.

## Per-model gotchas that have actually bitten

**Reasoning models.** Qwen3.5 emits `<think>…</think>` by default and does **not**
support the Qwen3 `/nothink` soft switch. The only lever is
`apply_chat_template(..., enable_thinking=False)`, which `activations.capture_residual`
now passes. Left alone, `MAX_NEW_TOKENS = 32` truncates mid-trace, no verdict is ever
emitted, and every label silently falls back to the assumed value — the run looks fine
and is worthless. Add any new reasoning model to `config.THINKING_MODELS`.

**Multimodal configs.** Qwen3.5 and Gemma 3 are `*ForConditionalGeneration` checkpoints
whose language-stack depth lives under `config.text_config`. Reading the top-level
`num_hidden_layers` can resolve `LAYER_FRACTIONS` onto the wrong tower and produce
plausible activations from the wrong part of the model. `activations.text_depth` reads
the text stack explicitly and raises rather than guessing.

**Precision.** `LOAD_IN_4BIT = False`, `DTYPE = "bfloat16"`. A 9–15B model fits a 48GB
card in bf16 with room to spare, and quantization perturbs exactly what this project
measures — residual-stream geometry. Precision is part of the cache identity, so
flipping it correctly invalidates the cache instead of silently reusing quantized
vectors.

**Capacity.** Prefer **secure** cloud. A community A100 gave us a host whose driver was
too old for the CUDA 12.8 image (container never started), and a stopped community pod
could not be restarted because the host had no free GPUs — which stranded its volume.
Pod-local volumes do not move between machines.

## Hardware

A single 48–80GB card. Measured on an A100 80GB PCIe, bf16, batch size 1:

- **1.12 s/row**, so the full 1,941-row manifest is **~36 minutes**
- ~17 GB VRAM for the 9B; phi-4 at 14.7B needs ~30 GB
- Full primary run plus two headline-only replications ≈ 2.5–3 h ≈ **$4** at $1.39/h

Generation dominates; everything after it is seconds. The activation cache is resumable
(`CHECKPOINT_EVERY_N_ROWS = 100`), so an eviction costs at most one checkpoint.

## The order to run things

```bash
python scripts/preflight.py --model <m>              # ~5 min, gates, prints raw output
python scripts/run_all.py   --model <m> --limit 400  # ~8 min slice, then STOPS to report
python scripts/run_all.py   --model <m>              # finishes; resumes, recomputes nothing
python scripts/rowlevel.py  --model <m>              # row-level + label-noise control
python scripts/verify.py    --model <m>              # independent re-derivation of every number
python scripts/sample_transcripts.py --model <m> --n 30
python scripts/figures.py
```

Or `bash scripts/overnight.sh`, which does all of it for the primary and both
replications and keeps going if one model fails.
