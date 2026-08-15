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
    """Return (model, tokenizer). 4-bit via bitsandbytes when config.LOAD_IN_4BIT."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_id = config.MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(hf_id)
    kwargs: dict = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if config.LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()
    return model, tok


def resolve_layers(model) -> dict[float, int]:
    """Map config.LAYER_FRACTIONS to concrete layer indices for this model's depth."""
    n = model.config.num_hidden_layers
    return {f: max(1, min(n, round(f * n))) for f in config.LAYER_FRACTIONS}


def capture_residual(model, tok, prompt: str, layers: list[int]):
    """Generate a response; return ({layer: residual vector averaged over RESPONSE
    tokens}, generated_text). hidden_states[L] is the output of block L (index 0 is
    the embeddings), so layer indices here are 1-based block outputs."""
    import torch

    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
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
    text = tok.decode(full[0, n_prompt:], skip_special_tokens=True)
    return vecs, text


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

def load_cache(key: str):
    """Return (meta dict, {layer: (ids array, acts matrix)}, texts DataFrame)."""
    cdir = config.cache_dir(key)
    meta = json.loads((cdir / "meta.json").read_text())
    layers = {}
    for L in meta["layers"]:
        z = np.load(cdir / f"layer{L}.npz", allow_pickle=False)
        layers[int(L)] = (z["ids"].astype(str), z["acts"])
    texts = pd.read_csv(cdir / "texts.csv")
    return meta, layers, texts


def _save_cache(cdir, layer_ids: list[int], ids: np.ndarray,
                acts_by_layer: dict[int, np.ndarray], texts: list[str], meta: dict) -> None:
    """Write layer<L>.npz + texts.csv + meta.json as one unit.

    Builds the new files in a scratch subdir first, then swaps each into place with
    Path.replace (atomic on POSIX for a same-filesystem rename), so a crash mid-write
    can never leave load_cache() looking at a half-written / truncated file. This is
    called both as a periodic checkpoint mid-run and as the final save, so "checkpoint"
    and "done" are the same code path rather than two things that can drift apart.
    """
    tmp = cdir / ".tmp_write"
    tmp.mkdir(exist_ok=True)
    for L in layer_ids:
        np.savez_compressed(tmp / f"layer{L}.npz", ids=ids, acts=acts_by_layer[L])
    pd.DataFrame({"id": ids, "text": texts}).to_csv(tmp / "texts.csv", index=False)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    for p in list(tmp.iterdir()):
        p.replace(cdir / p.name)
    tmp.rmdir()


def _load_resumable(cdir, layer_ids: list[int], prompt_hash: str):
    """Return (per_layer: {L: {id: vec}}, texts: {id: text}) recovered from a prior
    partial or complete cache that matches this run's identity (same prompts, same
    max_new_tokens, same layer set) — empty dicts if there's nothing usable to resume."""
    meta_p = cdir / "meta.json"
    if not meta_p.exists():
        return {L: {} for L in layer_ids}, {}
    meta = json.loads(meta_p.read_text())
    if (meta.get("prompt_hash") != prompt_hash
            or meta.get("max_new_tokens") != config.MAX_NEW_TOKENS
            or sorted(meta.get("layers", [])) != sorted(layer_ids)):
        print(f"cache is stale — starting fresh "
              f"(prompt_hash {meta.get('prompt_hash')} -> {prompt_hash}; "
              f"max_new_tokens {meta.get('max_new_tokens')} -> {config.MAX_NEW_TOKENS})")
        return {L: {} for L in layer_ids}, {}
    try:
        loaded = {L: np.load(cdir / f"layer{L}.npz") for L in layer_ids}
        texts_df = pd.read_csv(cdir / "texts.csv")
    except FileNotFoundError:
        return {L: {} for L in layer_ids}, {}
    ids = loaded[layer_ids[0]]["ids"].astype(str)
    per_layer = {L: dict(zip(ids, loaded[L]["acts"])) for L in layer_ids}
    texts = dict(zip(texts_df.id.astype(str), texts_df.text))
    return per_layer, texts


def run(model_key: str, dry_run: bool = False, synthetic: bool = False,
        headline_only: bool = False) -> str:
    """Cache activations for every manifest row; returns the cache/results key.

    Resumable: every config.CHECKPOINT_EVERY_N_ROWS rows (and at the end) the cache is
    flushed to disk with whatever's done so far. Re-running with the same model /
    prompts / max_new_tokens picks up where it left off instead of repeating already-
    paid-for GPU generation, so a spot eviction, dropped SSH session, or crash costs at
    most one checkpoint interval of work, not the whole run.
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
        if (meta.get("prompt_hash") == prompt_hash
                and meta.get("max_new_tokens") == config.MAX_NEW_TOKENS
                and meta.get("model_key") == model_key
                and meta.get("headline_only") == headline_only
                and meta.get("complete")):
            print(f"cache complete for {key} ({meta.get('n_rows')} ids) — skipping "
                  "(no model load needed)")
            return key

    if synthetic:
        acts_by_layer, texts = synthetic_activations(df)
        layer_ids = sorted(acts_by_layer)
        ids = df.id.to_numpy(dtype=str)
        _save_cache(cdir, layer_ids, ids, acts_by_layer, texts, {
            "model_key": model_key, "layers": layer_ids, "synthetic": synthetic,
            "dry_run": dry_run, "headline_only": headline_only, "n_rows": len(df),
            "seed": config.SEED, "prompt_hash": prompt_hash,
            "max_new_tokens": config.MAX_NEW_TOKENS, "complete": True,
        })
        config.log(f"cached activations: {len(df)} rows x layers {layer_ids} "
                   "**SYNTHETIC — smoke test only**", key)
        return key

    from tqdm import tqdm
    model, tok = load_model(model_key)
    layer_map = resolve_layers(model)
    layer_ids = sorted(layer_map.values())

    per_layer_by_id, texts_by_id = _load_resumable(cdir, layer_ids, prompt_hash)
    done_ids = set(per_layer_by_id[layer_ids[0]])
    todo = df[~df.id.isin(done_ids)]
    if done_ids:
        print(f"resuming {key}: {len(done_ids)}/{len(df)} rows already cached, "
              f"{len(todo)} left")

    def _flush(final: bool) -> None:
        ordered = [i for i in df.id if i in per_layer_by_id[layer_ids[0]]]
        ids = np.array(ordered, dtype=str)
        acts_by_layer = {L: np.stack([per_layer_by_id[L][i] for i in ordered]).astype(np.float32)
                         for L in layer_ids}
        texts = [texts_by_id[i] for i in ordered]
        _save_cache(cdir, layer_ids, ids, acts_by_layer, texts, {
            "model_key": model_key, "layers": layer_ids, "synthetic": synthetic,
            "dry_run": dry_run, "headline_only": headline_only, "n_rows": len(df),
            "seed": config.SEED, "prompt_hash": prompt_hash,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "complete": len(ordered) == len(df),
        })
        if final:
            config.log(f"cached activations: {len(ordered)}/{len(df)} rows x "
                       f"layers {layer_ids}", key)

    for i, r in enumerate(tqdm(todo.itertuples(), total=len(todo), desc=f"activations {key}")):
        vecs, text = capture_residual(model, tok, r.prompt, layer_ids)
        for L in layer_ids:
            per_layer_by_id[L][r.id] = vecs[L]
        texts_by_id[r.id] = text
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
    a = p.parse_args()
    run(a.model, a.dry_run, a.synthetic, a.headline_only)
