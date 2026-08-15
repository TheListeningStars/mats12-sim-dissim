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
from src import config, data_build, activations, truth_axis, transfer, baselines, geometry  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL, choices=list(config.MODELS))
    p.add_argument("--dry-run", action="store_true", help="tiny N, for smoke-testing")
    p.add_argument("--synthetic", action="store_true",
                   help="structured random activations — CPU pipeline check, not results")
    p.add_argument("--headline-only", action="store_true",
                   help="replication: plain-style cells only")
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
        "label_source": a.label_source,
        "n_per_cell": config.N_PER_CELL,
        "truthfit_pairs_per_topic": config.TRUTHFIT_PAIRS_PER_TOPIC,
        "layer_fractions": config.LAYER_FRACTIONS, "load_in_4bit": config.LOAD_IN_4BIT,
        "max_new_tokens": config.MAX_NEW_TOKENS, "model_hf_id": config.MODELS[a.model],
    }, indent=2))

    # 2..6 per model
    activations.run(a.model, a.dry_run, a.synthetic, a.headline_only)
    truth_axis.run(key, a.dry_run)          # fit d_truth, compute c, validity check FIRST
    transfer.run(key, a.dry_run, label_source=a.label_source)
    baselines.run(key, a.dry_run, label_source=a.label_source)   # horse race — primary
    geometry.run(key, a.dry_run, label_source=a.label_source)    # monotonicity — H2

    if a.synthetic:
        print("\n*** SYNTHETIC RUN — pipeline check only, numbers are meaningless ***")
    print(f"done. see {config.results_dir(key)} for figures + results/LOG.md")


if __name__ == "__main__":
    main()
