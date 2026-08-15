"""Geometry + H2 monotonicity curve.

- Cosines between per-cell diff-of-means 'framing' directions and d_truth
  (is the framing direction near-degenerate with truth for high-c cells?).
- The headline H2 figure: transfer AUROC of dissimulation-trained probes vs
  target-cell c, across simulation subtypes ordered high-c -> low-c. Monotonic
  degradation is evidence the axis is a real representational boundary.
- H3 (exploratory, demoted): high-c -> low-c vs low-c -> high-c asymmetry. The
  competing 'recruit-then-modify' mechanism (truth -> modify -> persona output)
  predicts NO asymmetry; we report which holds, promising neither.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import behavior, config, probes
from .activations import load_cache
from .transfer import build_cells, _xy


def direction_cosines(directions: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(directions)
    M = np.stack([directions[n] / np.linalg.norm(directions[n]) for n in names])
    return pd.DataFrame(M @ M.T, index=names, columns=names)


def monotonicity_curve(long: pd.DataFrame, rdir) -> dict:
    """Dissimulation-trained probes: mean AUROC per target cell vs target c,
    Spearman rank correlation for H2; H3 asymmetry comparison."""
    from scipy.stats import spearmanr

    d = long[(long.mode_source == "dissimulation") & (~long.is_diag)]
    per_tgt = d.groupby(["target_cell", "scenario_target"]) \
               .agg(auroc=("auroc", "mean"), c=("c_target", "first")).reset_index()
    rho, p = spearmanr(per_tgt.c, per_tgt.auroc)

    hi_lo = long[(~long.is_diag) & (long.c_source > long.c_target)].auroc.mean()
    lo_hi = long[(~long.is_diag) & (long.c_source < long.c_target)].auroc.mean()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for scen, g in per_tgt.groupby("scenario_target"):
        ax.scatter(g.c, g.auroc, label=scen, s=45)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("target-cell truth-conflict c")
    ax.set_ylabel("AUROC of dissimulation-trained probe")
    ax.set_title(f"H2: transfer vs target c (Spearman ρ={rho:.2f}, p={p:.3f})\n"
                 f"H3 asymmetry: high→low c {hi_lo:.3f} vs low→high {lo_hi:.3f} "
                 "(recruit-then-modify predicts ≈ equal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rdir / "monotonicity.png", dpi=150)
    plt.close(fig)

    return {"spearman_rho": float(rho), "spearman_p": float(p),
            "auroc_high_to_low_c": float(hi_lo), "auroc_low_to_high_c": float(lo_hi),
            "per_target": per_tgt.to_dict("records")}


def run(key: str, dry_run: bool = False, label_source: str = "behavior") -> None:
    rdir = config.results_dir(key)
    tl = rdir / "transfer_long.csv"
    if not tl.exists():
        print("!! no transfer_long.csv — transfer stage aborted; skipping geometry.")
        return
    long = behavior.load_transfer_long(tl)
    df = pd.read_csv(config.manifest_path(dry_run), keep_default_na=False)
    cs = behavior.load_c_scores(rdir / "c_scores.csv")
    df = df.merge(cs[["id", "c", "actually_lied", "parsed"]], on="id", how="inner")
    layer = json.loads((rdir / "truth_axis_meta.json").read_text())["best_layer"]
    _, layers, _ = load_cache(key)
    ids, acts = layers[layer]

    # per-cell diff-of-means deception directions + d_truth
    cells, _ = build_cells(df, label_source)
    dirs = {"d_truth": np.load(rdir / "d_truth.npy")}
    for r in cells.itertuples():
        X, y = _xy(df, ids, acts, r.cell, r.honest_cell, "train", label_source)
        dirs[r.cell] = probes.train_diffmean(X, y, layer).w
    cos = direction_cosines(dirs)
    cos.to_csv(rdir / "direction_cosines.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(cos)), max(6, 0.4 * len(cos))))
    sns.heatmap(cos, vmin=-1, vmax=1, cmap="coolwarm", center=0, ax=ax,
                annot=len(cos) <= 25, fmt=".2f")
    ax.set_title(f"Cosine similarity of per-cell deception directions + d_truth (layer {layer})")
    fig.tight_layout()
    fig.savefig(rdir / "direction_cosines.png", dpi=150)
    plt.close(fig)

    mono = monotonicity_curve(long, rdir)
    (rdir / "geometry.json").write_text(json.dumps(mono, indent=2))
    config.log(f"H2 monotonicity: Spearman ρ={mono['spearman_rho']:.2f} "
               f"(p={mono['spearman_p']:.3f}); H3 asymmetry high→low "
               f"{mono['auroc_high_to_low_c']:.3f} vs low→high "
               f"{mono['auroc_low_to_high_c']:.3f}", key)
    print(json.dumps({k: v for k, v in mono.items() if k != "per_target"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--label-source", default="behavior", choices=["behavior", "condition"])
    a = p.parse_args()
    run(config.effective_key(a.model, a.synthetic, a.dry_run), a.dry_run, a.label_source)
