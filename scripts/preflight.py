"""Cheap GPU gate. Run this FIRST on a fresh pod, before caching anything.

Costs ~2-5 minutes and one model load. It answers the only questions that can silently
ruin a multi-hour caching run, and it prints raw model output so you can read it yourself
rather than trusting a parse rate.

    python scripts/preflight.py --model qwen3.5-9b

Gates (any failure exits non-zero and tells you what to fix):
  MODEL LOADS      the checkpoint maps to a causal-LM class at all
  LAYERS RESOLVE   LAYER_FRACTIONS land on real block indices of the TEXT stack
                   (multimodal configs nest this under text_config; getting it wrong
                   silently probes the wrong tower)
  NO THINKING      no <think> block in any response. Qwen3.5 reasons by default and
                   does NOT support the /nothink soft switch — only
                   apply_chat_template(enable_thinking=False). With thinking on,
                   MAX_NEW_TOKENS=32 truncates mid-trace, the verdict never appears,
                   and every downstream label falls back to the assumed value.
  VERDICT PARSES   behavior.parse_verdict recovers a verdict from real generations
  ACTIVATIONS SANE right shape, finite, non-degenerate norms
  THROUGHPUT       measured sec/row, extrapolated to the full manifest, so you know
                   what you are committing to before you commit to it

Nothing here writes to the activation cache.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import activations, behavior, config, data_build  # noqa: E402

N_SAMPLE = 20
THINK_RE = re.compile(r"<think>|</think>", re.I)


def _stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """One row from each of n distinct cells — never the first n rows of the manifest,
    which would all come from a single cell and tell you nothing about the others."""
    rng = np.random.default_rng(seed)
    cells = df.cell.drop_duplicates().to_numpy()
    pick = rng.permutation(cells)[:n]
    return pd.concat([df[df.cell == c].sample(1, random_state=seed) for c in pick])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL, choices=list(config.MODELS))
    p.add_argument("--dry-run", action="store_true", help="use the small manifest")
    p.add_argument("--n", type=int, default=N_SAMPLE)
    a = p.parse_args()

    fails: list[str] = []
    print(f"=== PREFLIGHT: {a.model} -> {config.MODELS[a.model]} ===\n")

    df = data_build.build(dry_run=a.dry_run)
    print(f"manifest: {len(df)} rows, {df.cell.nunique()} cells")

    print("\n--- loading model ---")
    t0 = time.time()
    try:
        model, tok = activations.load_model(a.model)
    except Exception as e:                                    # noqa: BLE001
        sys.exit(f"FAIL  model did not load: {type(e).__name__}: {e}")
    print(f"loaded in {time.time() - t0:.0f}s")
    print(f"  class      {type(model).__name__}")
    print(f"  dtype      {next(model.parameters()).dtype}")
    print(f"  device     {next(model.parameters()).device}")
    print(f"  4bit       {config.LOAD_IN_4BIT}")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  VRAM       {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated / "
                  f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB total")
    except Exception:                                         # noqa: BLE001
        pass

    # --- layers ---
    print("\n--- layer resolution ---")
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)
    n_top = getattr(cfg, "num_hidden_layers", None)
    n_text = getattr(text_cfg, "num_hidden_layers", None) if text_cfg else None
    print(f"  config.num_hidden_layers        {n_top}")
    print(f"  config.text_config.num_hidden_… {n_text}")
    layer_map = activations.resolve_layers(model)
    print(f"  LAYER_FRACTIONS {config.LAYER_FRACTIONS} -> {layer_map}")
    if text_cfg is not None and n_text is not None and n_text != n_top:
        fails.append(f"multimodal config: top-level num_hidden_layers={n_top} but "
                     f"text_config says {n_text}. resolve_layers() used the top-level "
                     f"value — point it at the text stack before caching.")
    n_eff = n_text or n_top
    if n_eff and max(layer_map.values()) > n_eff:
        fails.append(f"resolved layer {max(layer_map.values())} exceeds depth {n_eff}")

    # --- generate ---
    layers = sorted(set(layer_map.values()))
    sample = _stratified(df, min(a.n, df.cell.nunique()), config.SEED)
    print(f"\n--- generating {len(sample)} responses (max_new_tokens="
          f"{config.MAX_NEW_TOKENS}) ---")

    rows, t_gen = [], time.time()
    for r in sample.itertuples():
        vecs, pvecs, text, margin = activations.capture_residual(
            model, tok, r.prompt, layers)
        rows.append(dict(cell=r.cell, statement=r.statement, text=text,
                         said=behavior.parse_verdict(text),
                         think=bool(THINK_RE.search(text)),
                         refused=behavior.is_refusal(text),
                         vec=vecs[layers[0]], pvec=pvecs[layers[0]],
                         margin=margin, truth_value=r.truth_value, mode=r.mode))
    sec_per_row = (time.time() - t_gen) / len(rows)

    # --- raw output, printed in full: read it ---
    print("\n=== RAW RESPONSES (read these — do not just trust the parse rate) ===")
    for i, r in enumerate(rows, 1):
        said = {1: "TRUE", -1: "FALSE", 0: "*** UNPARSED ***"}[r["said"]]
        gt = {1: "TRUE", 0: "FALSE", -1: "n/a"}[r["truth_value"]]
        flag = "   <<< CONTAINS <think>" if r["think"] else ""
        print(f"\n[{i}] {r['cell']}")
        print(f"    statement: {r['statement']}   [ground truth: {gt}]")
        print(f"    response : {r['text']!r}")
        print(f"    said     : {said}{flag}")
        print(f"    belief   : logit(TRUE) - logit(FALSE) = {r['margin']:+.2f}")

    # --- gates ---
    print("\n=== GATES ===")
    n_think = sum(r["think"] for r in rows)
    parse_rate = float(np.mean([r["said"] != 0 for r in rows]))
    A = np.stack([r["vec"] for r in rows])
    norms = np.linalg.norm(A, axis=1)

    def gate(ok: bool, name: str, detail: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:16s} {detail}")
        if not ok:
            fails.append(f"{name}: {detail}")

    gate(n_think == 0, "no thinking",
         f"{n_think}/{len(rows)} responses contain a <think> block"
         + ("" if n_think == 0 else
            "  -> pass enable_thinking=False to apply_chat_template"))
    # Gate on the parser, not on the model's willingness. A refusal is a real response
    # that happens to contain no verdict; blocking on it would have stopped the phi-4
    # replication over the most interesting thing phi-4 does.
    n_ref = sum(r["refused"] for r in rows)
    unparse_not_refusal = [r for r in rows if r["said"] == 0 and not r["refused"]]
    eff = 1.0 - len(unparse_not_refusal) / max(len(rows), 1)
    gate(eff >= 0.9, "verdict parses",
         f"parse rate {parse_rate:.2f}; {n_ref} refusals excluded -> "
         f"parser accounts for {eff:.2f}")
    if n_ref:
        print(f"  (info) {n_ref}/{len(rows)} responses are REFUSALS — the model declined "
              "to assert. That is a result about this model, not a parsing problem; "
              "compliance.csv will carry the rate per cell.")
    gate(np.isfinite(A).all(), "activations finite",
         f"shape {A.shape}, {int((~np.isfinite(A)).sum())} non-finite")
    gate(float(norms.std() / max(norms.mean(), 1e-9)) > 1e-3, "activations vary",
         f"norm mean {norms.mean():.1f} sd {norms.std():.2f}")

    # The prompt-site residual must differ from the response mean, or --site prompt is
    # silently reading the same tensor and the whole point of capturing it is lost.
    P = np.stack([r["pvec"] for r in rows])
    site_diff = float(np.abs(A - P).mean() / max(np.abs(A).mean(), 1e-9))
    gate(site_diff > 1e-3, "sites differ",
         f"mean |response - prompt| / |response| = {site_diff:.3f}")

    # The belief margin is now the basis of the label and of c, so it has to be a real
    # measurement: finite, varying, and pointing the right way on facts the model knows.
    m = np.array([r["margin"] for r in rows])
    gate(bool(np.isfinite(m).all()) and float(m.std()) > 1e-6, "belief margin sane",
         f"mean {m.mean():+.2f} sd {m.std():.2f}")

    # Only HONEST-frame rows may be checked against ground truth. The margin is measured
    # on each row's own prompt, so inside a counterfactual or lying frame it correctly
    # follows the frame rather than the world — that is the signal, not a fault. Scoring
    # framed rows here would fail the gate precisely when the experiment is working.
    hon = [r for r in rows if r["mode"] == "honest" and r["truth_value"] in (0, 1)]
    agree = (float(np.mean([(r["margin"] > 0) == (r["truth_value"] == 1) for r in hon]))
             if hon else float("nan"))
    gate(len(hon) >= 3, "enough honest rows",
         f"{len(hon)} honest factual rows sampled (need 3+ to judge the next gate)")
    gate(not hon or agree >= 0.75, "belief tracks truth (honest frame only)",
         f"sign(margin) matches ground truth on {agree:.2f} of {len(hon)} honest rows")

    framed = [r for r in rows if r["mode"] != "honest" and r["truth_value"] in (0, 1)]
    if framed:
        fa = float(np.mean([(r["margin"] > 0) == (r["truth_value"] == 1) for r in framed]))
        print(f"  (info) framed rows track ground truth at {fa:.2f} over {len(framed)} rows "
              "— expected to be LOW; it is the frame moving the model, which is the "
              "recruit-then-modify question the design is about")

    est = sec_per_row * len(df)
    print(f"\n  throughput      {sec_per_row:.2f} s/row -> full manifest ({len(df)} rows) "
          f"~{est / 3600:.1f} h")

    print()
    if fails:
        print("PREFLIGHT FAILED — do not start the caching run:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("PREFLIGHT PASSED. Next: a limited slice, e.g.")
    print(f"  python scripts/run_all.py --model {a.model} --limit 200")
    print("then inspect, then rerun without --limit to finish (the cache resumes).")


if __name__ == "__main__":
    main()
