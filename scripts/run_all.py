"""Orchestrate the full pipeline end to end.

    python scripts/run_all.py --dry-run --synthetic       # CPU smoke test (no model)
    python scripts/run_all.py --model qwen2.5-7b-instruct # full primary run (GPU)
    python scripts/run_all.py --model llama-3.1-8b-instruct --headline-only

Stages (see docs/PLAN.md §10): data -> activations -> truth_axis -> transfer
-> baselines(horse race) -> geometry(monotonicity). Each stage caches, so reruns
are cheap. --synthetic uses structured random activations (pipeline check only;
outputs land in separate '<model>-synthetic' directories and are never findings).
--headline-only drops the formal-style cells (halves compute for replication runs;
the style-shift baseline is skipped there by construction).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import (config, data_build, activations, truth_axis, transfer,  # noqa: E402
                 baselines, geometry, probes)


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[1],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _versions() -> dict:
    """Library versions, because the dry runs and the GPU run will not have come from
    the same stack and that difference has to be visible in the artifact."""
    import importlib
    out = {}
    for m in ("torch", "transformers", "numpy", "pandas", "sklearn", "scipy",
              "statsmodels"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = "absent"
    return out


def canary_report(key: str, dry_run: bool) -> None:
    """What the limited slice bought you, printed so it can be read rather than trusted.

    Deliberately behavioural only — no probes, no AUROC. At this point the question is
    "is the model doing the task and is the parser reading it correctly", and every
    answer here is checkable by eye against the printed transcripts.
    """
    import re

    import pandas as pd

    from src import behavior
    from src.activations import load_cache

    meta, layers, texts = load_cache(key)
    df = pd.read_csv(config.manifest_path(dry_run), keep_default_na=False)
    cached = layers[next(iter(layers))][0]
    df = behavior.annotate(df[df.id.isin(cached)], texts)

    print(f"\n{'=' * 70}\nCANARY REPORT — {key}\n{'=' * 70}")
    print(f"cached {len(df)}/{meta.get('n_rows')} rows across {df.cell.nunique()} cells "
          f"(layers {meta.get('layers')})")

    n_think = int(df.text.str.contains(r"<think>|</think>", case=False, regex=True).sum())
    empty = int((df.text.str.strip() == "").sum())
    print(f"\n  <think> blocks   {n_think}   (must be 0)")
    print(f"  empty responses  {empty}")
    print(f"  parse rate       {df.parsed.mean():.3f}")
    print(f"  truthfit rows    {int((df.split == 'truthfit').sum())}  "
          "(d_truth needs these; priority-ordered so they come first)")

    print("\n  parse rate + compliance by cell group:")
    g = df.groupby(["mode", "sim_subtype"]).agg(
        n=("id", "size"), parse_rate=("parsed", "mean"), compliance=("complied", "mean"))
    print("   " + g.round(3).to_string().replace("\n", "\n   "))

    worst = df[~df.parsed]
    if len(worst):
        print(f"\n  {len(worst)} UNPARSED responses — read these, the parser is a regex "
              "over the first 160 chars and it is load-bearing:")
        for r in worst.head(5).itertuples():
            print(f"    [{r.cell}] {r.text!r}")

    print("\n  3 random transcripts (seeded):")
    for r in df.sample(min(3, len(df)), random_state=config.SEED).itertuples():
        print(f"    [{r.cell}] {r.statement}")
        print(f"      -> {r.text!r}   parsed={r.said}")

    ok = n_think == 0 and df.parsed.mean() >= 0.9 and empty == 0
    print(f"\n  VERDICT: {'looks sane — resume without --limit to finish' if ok else 'PROBLEM — do not resume until fixed'}")
    print("=" * 70)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL, choices=list(config.MODELS))
    p.add_argument("--dry-run", action="store_true", help="tiny N, for smoke-testing")
    p.add_argument("--synthetic", action="store_true",
                   help="structured random activations — CPU pipeline check, not results")
    p.add_argument("--headline-only", action="store_true",
                   help="replication: plain-style cells only")
    p.add_argument("--limit", type=int, default=None,
                   help="CANARY MODE. Cache at most this many new rows (spread across "
                        "every cell), print a diagnostic report, and stop before the "
                        "analysis stages. Re-run without --limit to resume and finish — "
                        "cached rows are never regenerated.")
    p.add_argument("--site", default="response", choices=["response", "prompt"],
                   help="which residual the probe reads. 'response' = mean over the "
                        "generated tokens (contains the verdict token the probe is "
                        "predicting). 'prompt' = the final prompt token, before the "
                        "model commits — no self-read possible.")
    p.add_argument("--label-source", default="behavior", choices=["behavior", "condition"],
                   help="behavior = label rows by what the model ACTUALLY asserted "
                        "(default, correct); condition = old cell-membership labelling, "
                        "retained only to demonstrate the instruction-reading artifact")
    a = p.parse_args()

    # 1. data (model-independent). ALWAYS rebuild: the build is seeded and deterministic,
    # so this is cheap and idempotent, while skipping it when the file exists silently
    # pinned stale prompts. On Kaggle with Persistence="Files only", data/ survives across
    # sessions, so an updated fact bank would never reach the full run — the manifest kept
    # the previous session's 1,336 easy-only rows. Cache invalidation is handled separately
    # by the prompt hash in activations.run.
    df = data_build.build(dry_run=a.dry_run)
    print(f"manifest: {len(df)} rows, {df.cell.nunique()} cells "
          f"({df.difficulty.value_counts().to_dict() if 'difficulty' in df else 'no difficulty col'})")

    key = config.effective_key(a.model, a.synthetic, a.dry_run)
    config.results_dir(key).joinpath("run_meta.json").write_text(json.dumps({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": vars(a), "key": key, "seed": config.SEED,
        "label_source": a.label_source, "site": a.site,
        "n_per_cell": config.N_PER_CELL,
        "truthfit_pairs_per_topic": config.TRUTHFIT_PAIRS_PER_TOPIC,
        "layer_fractions": config.LAYER_FRACTIONS, "load_in_4bit": config.LOAD_IN_4BIT,
        "dtype": config.DTYPE,
        "max_new_tokens": config.MAX_NEW_TOKENS, "model_hf_id": config.MODELS[a.model],
        # CLAUDE.md asks for every hyperparameter; the ones below were previously only
        # discoverable by reading the source, which makes a result hard to reproduce.
        "manifest_rows": int(len(df)), "manifest_cells": int(df.cell.nunique()),
        "n_statements": int(df.statement.nunique()),
        "probe_C": probes.LOGREG_C,
        "min_train_per_class": transfer.MIN_TRAIN_PER_CLASS,
        "min_eval_per_class": transfer.MIN_EVAL_PER_CLASS,
        "n_perm": baselines.N_PERM,
        "git_sha": _git_sha(),
        "versions": _versions(),
    }, indent=2))

    # 2..6 per model
    activations.run(a.model, a.dry_run, a.synthetic, a.headline_only, a.limit)

    if a.limit is not None:
        # Canary: stop here. The analysis stages on a thin slice would either crash on
        # degenerate cells or produce numbers nobody should look at — and either way
        # they'd distract from the only question this pass answers, which is whether
        # generation and parsing are behaving before we commit to the full run.
        canary_report(key, a.dry_run)
        return

    truth_axis.run(key, a.dry_run)          # fit d_truth, compute c, validity check FIRST
    transfer.run(key, a.dry_run, label_source=a.label_source, site=a.site)
    baselines.run(key, a.dry_run, label_source=a.label_source)   # horse race — primary
    geometry.run(key, a.dry_run, label_source=a.label_source)    # monotonicity — H2

    if a.synthetic:
        print("\n*** SYNTHETIC RUN — pipeline check only, numbers are meaningless ***")
    print(f"done. see {config.results_dir(key)} for figures + results/LOG.md")


if __name__ == "__main__":
    main()
