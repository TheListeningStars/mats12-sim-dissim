"""Transfer matrix: train a probe on each framed source cell, evaluate on every target.

LABELS (changed 2026-08-12 — this is the central fix; see results/LOG.md):
    label 1 = the model ACTUALLY asserted a falsehood on that row
    label 0 = the model actually asserted the truth
NOT cell membership. Under the old cell-membership labelling the probe learned
"which instruction was in the prompt": honest rows separated from
dissimulation-rows-where-the-model-refused-to-lie at AUROC 1.000, i.e. truthful
responses were flagged as maximally deceptive. Behaviour labelling puts those
refusal rows on the honest side where they belong, which is what makes the target
an actual deception probe and removes the ceiling.

Each probe still trains on one framed cell pooled with its matched honest cell, so
the contrast stays local; what changed is only how rows in that pool are labelled.
Rows with no parseable verdict, and truth-valueless (preference) rows, are excluded
rather than guessed at. Cells without both classes present are skipped and logged.

`--label-source condition` reproduces the old behaviour for direct comparison — the
gap between the two is itself the headline evidence for the instruction-reading artifact.

Outputs (results/<key>/):
    transfer_matrix.csv / transfer_heatmap.png   source x target AUROC
    transfer_long.csv    one row per (source, target) with c_source, c_target, |dc|,
                         scenario labels and cell metadata — raw material for H1/H2.
    skipped_cells.csv    cells excluded for lacking both classes, with counts
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import behavior, config, probes
from .activations import load_cache

MIN_TRAIN_PER_CLASS = 3
MIN_EVAL_PER_CLASS = 2


def matched_honest_cell(topic: str, style: str) -> str:
    """The honest cell providing label-0 rows for a framed cell. Preference cells
    (topic 'none') match the honest evaluation of the same preference statements."""
    return f"honest|-|{'none' if topic == 'none' else topic}|{'plain' if topic == 'none' else style}"


def label_rows(sub: pd.DataFrame, label_source: str = "behavior"):
    """Return (usable_rows, labels). behavior -> actually_lied; condition -> cell membership."""
    if label_source == "behavior":
        ok = sub[sub.actually_lied.notna()]
        return ok, ok.actually_lied.to_numpy().astype(int)
    return sub, (sub["mode"] != "honest").astype(int).to_numpy()


def _pool(df, framed_cell, honest_cell, split):
    return df[(df.cell.isin([framed_cell, honest_cell])) & (df.split == split)]


def _xy(df, ids, acts, framed_cell, honest_cell, split, label_source="behavior"):
    sub, y = label_rows(_pool(df, framed_cell, honest_cell, split), label_source)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    return acts[pos.loc[sub.id].to_numpy()], y


def build_cells(df: pd.DataFrame, label_source: str = "behavior"):
    """One row per usable framed cell. Returns (cells, skipped)."""
    framed = df[(df["mode"] != "honest") & (df.split != "truthfit")]
    rows, skipped = [], []
    for cell, g in framed.groupby("cell"):
        topic, style = g.topic.iloc[0], g["style"].iloc[0]
        honest_cell = matched_honest_cell(topic, style)
        counts = {}
        for split, need in (("train", MIN_TRAIN_PER_CLASS), ("eval", MIN_EVAL_PER_CLASS)):
            _, y = label_rows(_pool(df, cell, honest_cell, split), label_source)
            counts[split] = (int((y == 0).sum()), int((y == 1).sum()))
            if counts[split][0] < need or counts[split][1] < need:
                skipped.append(dict(cell=cell, reason=f"{split}: n0={counts[split][0]} "
                                                      f"n1={counts[split][1]} (<{need})"))
                break
        else:
            lie = g[g.actually_lied == 1]
            rows.append(dict(
                cell=cell, mode=g["mode"].iloc[0], sim_subtype=g.sim_subtype.iloc[0],
                scenario=g.scenario.iloc[0], topic=topic, style=style,
                # primary c = mean conflict among rows the model ACTUALLY lied on
                # (conflict per lie, decoupled from how OFTEN the cell elicits lying)
                c_mean=float(lie.c.mean()) if len(lie) else float("nan"),
                c_all=float(g.c.mean()),
                lie_rate=float(g.actually_lied.mean()) if g.actually_lied.notna().any() else np.nan,
                honest_cell=honest_cell,
            ))
    cells = pd.DataFrame(rows)
    if len(cells):
        cells = cells.sort_values(["mode", "sim_subtype", "topic", "style"]).reset_index(drop=True)
    return cells, pd.DataFrame(skipped)


def transfer_matrix(df, ids, acts, cells, layer, kind="logreg", label_source="behavior"):
    """Train on each source cell, score every target cell; returns (wide, long, probes)."""
    train_fn = probes.train_logreg if kind == "logreg" else probes.train_diffmean
    trained = {}
    for r in cells.itertuples():
        X, y = _xy(df, ids, acts, r.cell, r.honest_cell, "train", label_source)
        trained[r.cell] = train_fn(X, y, layer)

    long_rows = []
    for s in cells.itertuples():
        for t in cells.itertuples():
            Xe, ye = _xy(df, ids, acts, t.cell, t.honest_cell, "eval", label_source)
            long_rows.append(dict(
                source_cell=s.cell, target_cell=t.cell,
                auroc=probes.auroc(trained[s.cell].score(Xe), ye),
                c_source=s.c_mean, c_target=t.c_mean,
                abs_dc=abs(s.c_mean - t.c_mean),
                lie_rate_source=s.lie_rate, lie_rate_target=t.lie_rate,
                scenario_source=s.scenario, scenario_target=t.scenario,
                mode_source=s.mode, mode_target=t.mode,
                subtype_source=s.sim_subtype, subtype_target=t.sim_subtype,
                topic_source=s.topic, topic_target=t.topic,
                style_source=s.style, style_target=t.style,
                is_diag=s.cell == t.cell,
            ))
    long = pd.DataFrame(long_rows)
    wide = long.pivot(index="source_cell", columns="target_cell", values="auroc") \
               .loc[cells.cell, cells.cell]
    return wide, long, trained


def run(key: str, dry_run: bool = False, kind: str = "logreg",
        label_source: str = "behavior") -> None:
    rdir = config.results_dir(key)
    df = pd.read_csv(config.manifest_path(dry_run), keep_default_na=False)
    cs = behavior.load_c_scores(rdir / "c_scores.csv")
    df = df.merge(cs[["id", "c", "actually_lied", "parsed", "complied", "t_hat"]],
                  on="id", how="inner")
    layer = json.loads((rdir / "truth_axis_meta.json").read_text())["best_layer"]
    _, layers, _ = load_cache(key)
    ids, acts = layers[layer]

    cells, skipped = build_cells(df, label_source)
    if len(skipped):
        skipped.to_csv(rdir / "skipped_cells.csv", index=False)
    if len(cells) < 2:
        msg = (f"only {len(cells)} usable cell(s) under label_source={label_source} — "
               "not enough for a transfer matrix. See skipped_cells.csv.")
        config.log(f"!! transfer ABORTED: {msg}", key)
        print("!!", msg)
        if len(skipped):
            print(skipped.to_string(index=False))
        return

    wide, long, _ = transfer_matrix(df, ids, acts, cells, layer, kind, label_source)
    wide.to_csv(rdir / "transfer_matrix.csv")
    long.to_csv(rdir / "transfer_long.csv", index=False)
    cells.to_csv(rdir / "cells.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(wide)), max(6, 0.4 * len(wide))))
    sns.heatmap(wide, vmin=0.0, vmax=1.0, cmap="RdYlGn", center=0.5,
                annot=len(wide) <= 25, fmt=".2f", annot_kws={"size": 6}, ax=ax,
                cbar_kws={"label": "AUROC"})
    ax.set_title(f"Probe transfer ({kind}, layer {layer}, labels={label_source})\n"
                 "train row, test column")
    fig.tight_layout()
    fig.savefig(rdir / "transfer_heatmap.png", dpi=150)
    plt.close(fig)

    diag, off = long[long.is_diag].auroc.mean(), long[~long.is_diag].auroc.mean()
    config.log(f"transfer matrix ({kind}, layer {layer}, labels={label_source}): "
               f"{len(cells)} cells ({len(skipped)} skipped), diag AUROC {diag:.3f}, "
               f"off-diag {off:.3f}", key)
    print(f"{len(cells)} cells ({len(skipped)} skipped); labels={label_source}; "
          f"in-dist mean AUROC {diag:.3f}; off-diag mean {off:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--kind", default="logreg", choices=["logreg", "diffmean"])
    p.add_argument("--label-source", default="behavior", choices=["behavior", "condition"])
    a = p.parse_args()
    run(config.effective_key(a.model, a.synthetic, a.dry_run), a.dry_run, a.kind, a.label_source)
