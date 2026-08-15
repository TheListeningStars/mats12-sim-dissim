"""Load a model in 4-bit and cache mean-over-response-token residual-stream activations.

Output (under cache/<key>/):
    layer<L>.npz   arrays: ids (str), acts (n x d)   one file per swept layer
    texts.csv      id, text  (generated responses, for the behavioral baseline)
    meta.json      layers used, model id, synthetic flag

One forward pass per prompt; cache once and reuse everywhere (probes, transfer, geometry).

--synthetic generates STRUCTURED RANDOM activations (no model, CPU-only) so the whole
downstream pipeline can be smoke-tested. Synthetic caches/results live in separate
'<model>-synthetic' directories and are flagged in meta.json — they must never be
reported as findings.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from . import config


def load_model(model_key: str):
    """Return (model, tokenizer). 4-bit via bitsandbytes when config.LOAD_IN_4BIT.

    Falls back from AutoModelForCausalLM to AutoModel for checkpoints whose architecture
    is a *ForConditionalGeneration (Qwen3.5, Gemma 3): the causal-LM auto-class does not
    always map those, and a hard failure here is much better than a wrong-tower load.
    """
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

    hf_id = config.MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(hf_id)
    kwargs: dict = dict(device_map="auto", torch_dtype=getattr(torch, config.DTYPE))
    if config.LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    except (ValueError, KeyError) as e:
        print(f"AutoModelForCausalLM did not map {hf_id} ({e}); trying AutoModel")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(hf_id, **kwargs)
    model.eval()
    return model, tok


def text_depth(model) -> int:
    """Number of transformer blocks in the TEXT stack.

    Multimodal checkpoints (Qwen3_5ForConditionalGeneration, Gemma3ForConditionalGeneration)
    nest the language model's depth under config.text_config, and the top-level
    num_hidden_layers may be absent or describe the vision tower. Reading the wrong one
    resolves LAYER_FRACTIONS onto the wrong blocks — which produces plausible-looking
    activations from the wrong part of the model, the worst kind of failure.
    """
    text_cfg = getattr(model.config, "text_config", None)
    n = getattr(text_cfg, "num_hidden_layers", None) if text_cfg is not None else None
    if n is None:
        n = getattr(model.config, "num_hidden_layers", None)
    if n is None:
        raise RuntimeError(
            f"cannot determine text-stack depth for {type(model).__name__}; "
            "inspect model.config and set it explicitly before caching")
    return int(n)


def resolve_layers(model) -> dict[float, int]:
    """Map config.LAYER_FRACTIONS to concrete layer indices for this model's depth."""
    n = text_depth(model)
    return {f: max(1, min(n, round(f * n))) for f in config.LAYER_FRACTIONS}


_VERDICT_IDS: dict[int, tuple[int, int]] = {}
_THINKING_OK: dict[int, bool] = {}


def _verdict_token_ids(tok) -> tuple[int, int]:
    """Token ids for the TRUE / FALSE continuation after 'VERDICT:'.

    Resolved once per tokenizer. Tries the leading-space form first because that is how
    the token actually appears after a colon; falls back to the bare form.
    """
    key = id(tok)
    if key in _VERDICT_IDS:
        return _VERDICT_IDS[key]
    def first(s: str) -> int | None:
        enc = tok.encode(s, add_special_tokens=False)
        return enc[0] if enc else None
    for t, f in ((" TRUE", " FALSE"), ("TRUE", "FALSE"), (" True", " False")):
        a, b = first(t), first(f)
        if a is not None and b is not None and a != b:
            _VERDICT_IDS[key] = (a, b)
            return a, b
    raise RuntimeError("could not resolve distinct TRUE/FALSE token ids for this tokenizer")


def belief_margin(model, tok, prompt: str) -> float:
    """Log-odds the model assigns to TRUE vs FALSE, read by teacher-forcing 'VERDICT:'.

    This is the project's belief measurement that is INDEPENDENT of d_truth. It matters
    because the original t_hat is a projection onto d_truth, and the deception label
    (actually_lied) is sign(said)·sign(t_hat) — so labelling by t_hat and then predicting
    with a quantity derived from t_hat is circular. A logit margin shares no machinery
    with d_truth: it is the model's own output distribution, not a direction we fitted.

    One extra forward pass per row, no generation, so it is cheap next to the
    MAX_NEW_TOKENS decode steps that dominate.
    """
    import torch

    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   **_thinking_kwargs(tok))
    ids = tok(text + "VERDICT:", return_tensors="pt", add_special_tokens=False)
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.no_grad():
        logits = model(**ids).logits[0, -1, :].float()
    t_id, f_id = _verdict_token_ids(tok)
    return float(logits[t_id] - logits[f_id])


def _thinking_kwargs(tok) -> dict:
    """apply_chat_template kwargs that suppress a reasoning trace, when supported.

    Qwen3.5 reasons by default and does NOT honour the Qwen3 /think /nothink soft
    switch, so enable_thinking=False is the only lever. Templates that don't accept the
    kwarg would raise, so we probe once and cache the answer.
    """
    key = id(tok)
    if key not in _THINKING_OK:
        try:
            tok.apply_chat_template([{"role": "user", "content": "x"}],
                                    add_generation_prompt=True, tokenize=False,
                                    enable_thinking=False)
            _THINKING_OK[key] = True
        except TypeError:
            _THINKING_OK[key] = False
    return {"enable_thinking": False} if _THINKING_OK[key] else {}


def capture_residual(model, tok, prompt: str, layers: list[int]):
    """Generate a response; return (response_vecs, prompt_vecs, text, margin).

    response_vecs  {layer: residual averaged over the RESPONSE tokens}
    prompt_vecs    {layer: residual at the FINAL PROMPT token, before anything is
                   generated}. This costs nothing — the forward pass already computes it
                   and previously threw it away — and it answers a different question:
                   the response mean necessarily contains the verdict token the probe is
                   being asked to predict, whereas the prompt-final position is the
                   model's state *before* it commits, so a probe read there cannot be
                   reading its own answer off the page.
    margin         belief_margin() above, the d_truth-independent belief measure.

    hidden_states[L] is the output of block L (index 0 is the embeddings), so layer
    indices here are 1-based block outputs.
    """
    import torch

    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  **_thinking_kwargs(tok))
    if hasattr(ids, "input_ids"):   # some tokenizer versions return a BatchEncoding
        ids = ids.input_ids
    ids = ids.to(model.device)
    with torch.no_grad():
        full = model.generate(ids, max_new_tokens=config.MAX_NEW_TOKENS,
                              do_sample=False, pad_token_id=tok.eos_token_id)
        out = model(full, output_hidden_states=True)
    n_prompt = ids.shape[1]
    # guard: if generation produced nothing, fall back to the last prompt token
    lo = n_prompt if full.shape[1] > n_prompt else full.shape[1] - 1
    hs = out.hidden_states
    vecs = {L: hs[L][0, lo:, :].float().mean(0).cpu().numpy() for L in layers}
    pvecs = {L: hs[L][0, n_prompt - 1, :].float().cpu().numpy() for L in layers}
    text = tok.decode(full[0, n_prompt:], skip_special_tokens=True)
    return vecs, pvecs, text, belief_margin(model, tok, prompt)


# --- synthetic mode ---------------------------------------------------------

def _unit(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _stmt_conf(statement_id: str) -> float:
    """Deterministic per-statement 'model confidence' in [0.5, 1.5]."""
    h = int(hashlib.md5(statement_id.encode()).hexdigest(), 16) % 1000
    return 0.5 + h / 999.0


# Simulated compliance per cell group, mirroring what the real model actually did
# (dry run 2026-08-12): the model often ignores the instruction. The synthetic mode
# reproduces that so the behaviour-labelling path is genuinely exercised, and so the
# planted deception signal follows the ACTUAL assertion rather than the instructed one.
_SYNTH_COMPLIANCE = {"honest": 0.99, "instructed_lying": 0.55, "counterfactual": 0.70,
                     "roleplay": 0.75, "persona": 0.85, "preference": 1.0}


def synthetic_activations(df: pd.DataFrame, d: int = 256):
    """Planted-structure random activations + matching verdict texts.

    Structure: a truth direction (scaled by per-statement confidence), a deception
    direction active exactly when the ACTUAL assertion negates the internal truth,
    plus scenario/topic/style nuisance directions and noise. Signal peaks at the middle
    synthetic layer so layer selection is exercised. FOR PIPELINE SMOKE TESTS ONLY —
    a correct pipeline should recover the planted signal under behaviour labelling and
    show inflated, artifactual separation under condition labelling.

    Returns ({layer: acts}, texts list).
    """
    rng = np.random.default_rng(config.SEED + 7)
    dirs = {name: _unit(rng, d) for name in
            ["truth", "decep", "style", "instruction"]
            + [f"scen_{s}" for s in df.scenario.unique()]
            + [f"topic_{t}" for t in df.topic.unique()]}

    said, texts = [], []
    filler = np.array(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    for r in df.itertuples():
        comply = rng.random() < _SYNTH_COMPLIANCE.get(r.scenario, 0.8)
        s = int(r.asserted_true) if (comply or r.truth_value == -1) else -int(r.asserted_true)
        said.append(s)
        texts.append(f"VERDICT: {'TRUE' if s > 0 else 'FALSE'}\n"
                     + " ".join(rng.choice(filler, 6)))

    layers = {10: 0.7, 12: 1.0, 14: 0.8}   # pseudo layer index -> signal scale
    out = {}
    for L, scale in layers.items():
        acts = rng.standard_normal((len(df), d))
        for i, r in enumerate(df.itertuples()):
            conf = _stmt_conf(r.statement_id)
            # hard facts sit nearer the model's uncertainty boundary, so the planted
            # truth signal is weaker there — mirrors what the hard bank is meant to do
            if getattr(r, "difficulty", "easy") == "hard":
                conf *= 0.4
            tv_sign = 0.0 if r.truth_value == -1 else (1.0 if r.truth_value == 1 else -1.0)
            # magnitude chosen so simulated |t_hat| lands near what the real model shows
            # (~0.66 on easy facts); a working instrument should clear the validity gate
            acts[i] += dirs["truth"] * (3.2 * tv_sign * conf * scale)
            conflict = max(0.0, -said[i] * tv_sign) * conf      # follows ACTUAL assertion
            acts[i] += dirs["decep"] * (2.0 * conflict * scale)
            # instruction-condition component: present whenever the PROMPT asked for a
            # falsehood, regardless of compliance. This is the artifact that condition
            # labelling latches onto and behaviour labelling must not.
            if r.mode != "honest":
                acts[i] += dirs["instruction"] * (1.2 * scale)
            acts[i] += dirs[f"scen_{r.scenario}"] * 0.8
            acts[i] += dirs[f"topic_{r.topic}"] * 0.5
            if r.style == "formal":
                acts[i] += dirs["style"] * 0.4
        out[L] = acts.astype(np.float32)
    return out, texts


# --- cache I/O --------------------------------------------------------------

def load_cache(key: str, site: str = "response"):
    """Return (meta dict, {layer: (ids array, acts matrix)}, texts DataFrame).

    `site` selects which residual to return:
      "response"  mean over the generated response tokens (the original behaviour)
      "prompt"    the final PROMPT token, before the model has emitted anything

    The prompt site exists because the response mean necessarily contains the verdict
    token the probe is trying to predict — the text-only baseline reaching 0.93 is that
    leak showing up. Reading before the model commits asks the more interesting
    question, and costs nothing extra since the forward pass computes both.
    """
    cdir = config.cache_dir(key)
    meta = json.loads((cdir / "meta.json").read_text())
    field = {"response": "acts", "prompt": "acts_prompt"}[site]
    layers = {}
    for L in meta["layers"]:
        z = np.load(cdir / f"layer{L}.npz", allow_pickle=False)
        if field not in z:
            raise KeyError(
                f"cache for {key} has no '{field}' (cache_format "
                f"{meta.get('cache_format', 1)}). Re-cache with the current code — "
                "the prompt-site residual was added in cache_format 2.")
        layers[int(L)] = (z["ids"].astype(str), z[field])
    texts = pd.read_csv(cdir / "texts.csv")
    if not meta.get("complete", False):
        print(f"WARNING: cache for {key} is INCOMPLETE "
              f"({len(layers[int(meta['layers'][0])][0])}/{meta.get('n_rows')} rows). "
              "Downstream numbers are partial — finish the run before reporting them.")
    return meta, layers, texts


def _cache_identity(prompt_hash: str, layer_ids: list[int]) -> dict:
    """Everything that changes activations without changing a row id.

    Prompt wording, generation length, layer set, AND numeric precision. Precision is
    here because flipping 4-bit -> bf16 changes every vector while leaving prompts and
    ids untouched, so without it the run would silently reuse quantized activations.
    """
    return {"prompt_hash": prompt_hash, "max_new_tokens": config.MAX_NEW_TOKENS,
            "layers": sorted(layer_ids), "load_in_4bit": config.LOAD_IN_4BIT,
            "dtype": config.DTYPE, "cache_format": 2}


def _identity_matches(meta: dict, ident: dict) -> bool:
    for k, v in ident.items():
        if meta.get(k) != v:
            return False
    return True


def _save_cache(cdir, layer_ids: list[int], ids: np.ndarray,
                acts_by_layer: dict[int, np.ndarray], texts: list[str], meta: dict,
                prompt_acts_by_layer: dict[int, np.ndarray] | None = None,
                margins: list[float] | None = None) -> None:
    """Write layer<L>.npz + texts.csv + meta.json as one unit.

    Builds the new files in a scratch subdir first, then swaps each into place with
    Path.replace (atomic on POSIX for a same-filesystem rename), so a crash mid-write
    can never leave load_cache() looking at a half-written / truncated file. This is
    called both as a periodic checkpoint mid-run and as the final save, so "checkpoint"
    and "done" are the same code path rather than two things that can drift apart.
    """
    tmp = cdir / ".tmp_write"
    if tmp.exists():
        # a previous crash may have left files here; promoting them would mix layer sets
        for p in list(tmp.iterdir()):
            p.unlink()
    tmp.mkdir(exist_ok=True)
    for L in layer_ids:
        arrays = {"ids": ids, "acts": acts_by_layer[L]}
        if prompt_acts_by_layer is not None:
            arrays["acts_prompt"] = prompt_acts_by_layer[L]
        np.savez_compressed(tmp / f"layer{L}.npz", **arrays)
    cols = {"id": ids, "text": texts}
    if margins is not None:
        cols["belief_margin"] = margins
    pd.DataFrame(cols).to_csv(tmp / "texts.csv", index=False)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    for p in list(tmp.iterdir()):
        p.replace(cdir / p.name)
    tmp.rmdir()


def _load_resumable(cdir, layer_ids: list[int], prompt_hash: str):
    """Return (per_layer: {L: {id: vec}}, texts: {id: text}) recovered from a prior
    partial or complete cache that matches this run's identity (same prompts, same
    max_new_tokens, same layer set) — empty dicts if there's nothing usable to resume."""
    empty = ({L: {} for L in layer_ids}, {L: {} for L in layer_ids}, {}, {})
    meta_p = cdir / "meta.json"
    if not meta_p.exists():
        return empty
    meta = json.loads(meta_p.read_text())
    ident = _cache_identity(prompt_hash, layer_ids)
    if not _identity_matches(meta, ident):
        diff = {k: (meta.get(k), v) for k, v in ident.items() if meta.get(k) != v}
        print(f"cache is stale — starting fresh. changed: {diff}")
        return empty
    try:
        loaded = {L: np.load(cdir / f"layer{L}.npz") for L in layer_ids}
        texts_df = pd.read_csv(cdir / "texts.csv")
    except FileNotFoundError:
        return empty
    ids = loaded[layer_ids[0]]["ids"].astype(str)
    per_layer = {L: dict(zip(ids, loaded[L]["acts"])) for L in layer_ids}
    prompt_layer = {L: dict(zip(ids, loaded[L]["acts_prompt"])) for L in layer_ids}
    texts = dict(zip(texts_df.id.astype(str), texts_df.text))
    margins = (dict(zip(texts_df.id.astype(str), texts_df.belief_margin))
               if "belief_margin" in texts_df else {})
    return per_layer, prompt_layer, texts, margins


def _priority_order(todo: pd.DataFrame) -> pd.DataFrame:
    """Order rows so that ANY prefix is a usable sample, not an arbitrary corner.

    Two rules, in order:
      1. `truthfit` rows first. d_truth is fit only on those (truth_axis.run), so a
         partial cache without them cannot produce a truth direction at all.
      2. then round-robin across cells — the 1st row of every cell, then the 2nd of
         every cell, and so on.

    Without this a `--limit` slice would be the manifest's leading rows, which all
    belong to one or two cells: it would burn GPU and still tell you nothing about
    whether the other cells generate sane text.
    """
    t = todo.copy()
    t["_fit"] = (t["split"] != "truthfit").astype(int)     # 0 = truthfit, sorts first
    t["_rank"] = t.groupby(["_fit", "cell"]).cumcount()
    return (t.sort_values(["_fit", "_rank", "cell"], kind="mergesort")
             .drop(columns=["_fit", "_rank"]))


def run(model_key: str, dry_run: bool = False, synthetic: bool = False,
        headline_only: bool = False, limit: int | None = None) -> str:
    """Cache activations for every manifest row; returns the cache/results key.

    Resumable: every config.CHECKPOINT_EVERY_N_ROWS rows (and at the end) the cache is
    flushed to disk with whatever's done so far. Re-running with the same model /
    prompts / max_new_tokens picks up where it left off instead of repeating already-
    paid-for GPU generation, so a spot eviction, dropped SSH session, or crash costs at
    most one checkpoint interval of work, not the whole run.

    `limit` caches at most that many NEW rows this invocation and leaves the cache
    marked incomplete, so a later call with limit=None resumes and finishes. Combined
    with _priority_order this is the "run 30 minutes, check it's sane, then commit"
    workflow — the limited slice is spread across every cell, and none of the GPU time
    spent on it is repeated later.
    """
    key = config.effective_key(model_key, synthetic, dry_run)
    df = pd.read_csv(config.manifest_path(dry_run), keep_default_na=False)
    if headline_only:
        df = df[df["style"] == "plain"]   # replication runs skip the style-shift arm
    cdir = config.cache_dir(key)

    # Prompt text is part of the cache identity: ids encode (cell, statement) but not
    # the wording, so a template change would otherwise silently reuse stale activations.
    prompt_hash = hashlib.md5("\x00".join(df.sort_values("id").prompt).encode()).hexdigest()[:12]

    meta_p = cdir / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if (meta.get("model_key") == model_key
                and meta.get("headline_only") == headline_only
                and meta.get("complete")):
            # identity also covers dtype/4-bit/layers/max_new_tokens — see _cache_identity
            probe_layers = sorted(meta.get("layers", []))
            if _identity_matches(meta, _cache_identity(prompt_hash, probe_layers)):
                print(f"cache complete for {key} ({meta.get('n_rows')} ids) — skipping "
                      "(no model load needed)")
                return key

    if synthetic:
        acts_by_layer, texts = synthetic_activations(df)
        layer_ids = sorted(acts_by_layer)
        ids = df.id.to_numpy(dtype=str)
        # Synthetic belief margins must track the SAME planted truth that the synthetic
        # activations encode, otherwise the smoke test stops exercising the thing it
        # exists to exercise: a correct pipeline should recover the planted signal.
        # Confidence reuses _stmt_conf so a statement's margin and its activation agree,
        # and truth-valueless rows get ~0 margin, mirroring the real construct.
        rng = np.random.default_rng(config.SEED + 99)
        conf = df.statement_id.map(_stmt_conf).to_numpy()
        sign = np.where(df.truth_value.to_numpy() == 1, 1.0,
                        np.where(df.truth_value.to_numpy() == 0, -1.0, 0.0))
        margins = sign * conf * 4.0 + rng.standard_normal(len(df)) * 0.8
        _save_cache(cdir, layer_ids, ids, acts_by_layer, texts, {
            "model_key": model_key, "synthetic": synthetic,
            "dry_run": dry_run, "headline_only": headline_only, "n_rows": len(df),
            "seed": config.SEED, "complete": True,
            **_cache_identity(prompt_hash, layer_ids),
        # synthetic mode has no real prompt-site tensor, so the two sites are the same
        # array here. --site prompt and --site response therefore give IDENTICAL numbers
        # under --synthetic; that is expected, not a broken flag. They diverge on a
        # real model, which is the only place the comparison means anything.
        }, prompt_acts_by_layer={L: acts_by_layer[L] for L in layer_ids},
            margins=list(margins))
        config.log(f"cached activations: {len(df)} rows x layers {layer_ids} "
                   "**SYNTHETIC — smoke test only**", key)
        return key

    from tqdm import tqdm
    model, tok = load_model(model_key)
    layer_map = resolve_layers(model)
    layer_ids = sorted(layer_map.values())

    per_layer_by_id, prompt_by_id, texts_by_id, margin_by_id = _load_resumable(
        cdir, layer_ids, prompt_hash)
    done_ids = set(per_layer_by_id[layer_ids[0]])
    todo = _priority_order(df[~df.id.isin(done_ids)])
    if done_ids:
        print(f"resuming {key}: {len(done_ids)}/{len(df)} rows already cached, "
              f"{len(todo)} left")
    if limit is not None and limit < len(todo):
        todo = todo.head(limit)
        print(f"--limit {limit}: caching {len(todo)} rows this pass "
              f"(spread over {todo.cell.nunique()} cells); "
              f"{len(df) - len(done_ids) - len(todo)} will remain. "
              "Re-run without --limit to finish; nothing here is recomputed.")

    def _flush(final: bool) -> None:
        ordered = [i for i in df.id if i in per_layer_by_id[layer_ids[0]]]
        ids = np.array(ordered, dtype=str)
        acts_by_layer = {L: np.stack([per_layer_by_id[L][i] for i in ordered]).astype(np.float32)
                         for L in layer_ids}
        prompt_acts = {L: np.stack([prompt_by_id[L][i] for i in ordered]).astype(np.float32)
                       for L in layer_ids}
        texts = [texts_by_id[i] for i in ordered]
        _save_cache(cdir, layer_ids, ids, acts_by_layer, texts, {
            "model_key": model_key, "synthetic": synthetic,
            "dry_run": dry_run, "headline_only": headline_only, "n_rows": len(df),
            "seed": config.SEED, "complete": len(ordered) == len(df),
            **_cache_identity(prompt_hash, layer_ids),
        }, prompt_acts_by_layer=prompt_acts,
            margins=[margin_by_id[i] for i in ordered])
        if final:
            config.log(f"cached activations: {len(ordered)}/{len(df)} rows x "
                       f"layers {layer_ids}", key)

    for i, r in enumerate(tqdm(todo.itertuples(), total=len(todo), desc=f"activations {key}")):
        vecs, pvecs, text, margin = capture_residual(model, tok, r.prompt, layer_ids)
        for L in layer_ids:
            per_layer_by_id[L][r.id] = vecs[L]
            prompt_by_id[L][r.id] = pvecs[L]
        texts_by_id[r.id] = text
        margin_by_id[r.id] = margin
        if (i + 1) % config.CHECKPOINT_EVERY_N_ROWS == 0:
            _flush(final=False)
    _flush(final=True)
    return key


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--synthetic", action="store_true",
                   help="structured random activations, CPU smoke test only")
    p.add_argument("--headline-only", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="cache at most this many NEW rows, spread across all cells, "
                        "then stop. Re-run without it to resume and finish.")
    a = p.parse_args()
    run(a.model, a.dry_run, a.synthetic, a.headline_only, a.limit)
