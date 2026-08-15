"""Publication-quality figures for the write-up.

Separate from the figures the pipeline emits inline: those are diagnostics meant to be
glanced at during a run, these are meant to survive being pasted into a document with a
hard three-page budget. Each one is standalone and carries its own conclusion in the
title, so a reader who sees only the figure still gets the point.

    python scripts/figures.py                 # every model with results
    python scripts/figures.py --model phi-4

Palette: three categorical slots (blue/orange/aqua) validated for colourblind
separation and lightness band against a light surface. Aqua sits below 3:1 contrast on
that surface, so every series is ALSO direct-labelled rather than identified by colour
alone. Sequential data uses one hue light->dark, never a rainbow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import behavior, config          # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a8a7a1"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlepad": 12,
})


def _finish(ax, title, sub=None, grid_axis="x"):
    """Lay out the axes, then title it. Call this LAST; it owns tight_layout.

    Titles are placed in FIGURE coordinates, not axes coordinates. Anchoring them to the
    axes left edge looks fine until a chart has long y-tick labels, which pushes the axes
    right and runs the title off the canvas -- which is exactly what happened to the
    variance-decomposition figure.
    """
    ax.grid(axis=grid_axis, color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig = ax.figure
    fig.tight_layout(rect=(0, 0, 1, 0.86 if sub else 0.91))
    fig.text(0.012, 0.955, title, color=INK, fontsize=12, fontweight="bold", va="top")
    if sub:
        fig.text(0.012, 0.885, sub, color=INK2, fontsize=9, va="top")


def fig_rowlevel(rdir: Path, out: Path) -> bool:
    """The headline. Ordered strata x two label definitions, with bootstrap CIs.

    Two series, so: legend present AND both direct-labelled. The CIs are the reason
    this figure exists -- the claim is that the gap is real, and a reader has to be
    able to see that the intervals do not overlap.
    """
    p = rdir / "rowlevel.json"
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    strat = d.get("stratified_auroc", {})
    if "belief" not in strat:
        return False
    names = [n for n in ("<0.3", "0.3-0.6", ">0.6") if n in strat["belief"]]
    if not names:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = [("belief", "label: asserted vs own belief", BLUE),
              ("groundtruth", "label: asserted vs ground truth", ORANGE)]
    off = {0: -0.14, 1: 0.14}
    for k, (key, lab, col) in enumerate(series):
        if key not in strat:
            continue
        xs, ys, los, his = [], [], [], []
        for i, n in enumerate(names):
            v = strat[key].get(n)
            if not v or not np.isfinite(v["auroc"]):
                continue
            xs.append(i + off[k]); ys.append(v["auroc"])
            los.append(v["auroc"] - v["ci_lo"]); his.append(v["ci_hi"] - v["auroc"])
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o", color=col, markersize=8,
                    linewidth=2, capsize=4, label=lab, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        # labels go OUTSIDE the pair (left of the left series, right of the right one)
        # so the two values in a stratum can never collide however close the points are
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(-9, -4) if k == 0 else (9, -4),
                        ha="right" if k == 0 else "left",
                        color=INK, fontsize=9, fontweight="bold")

    all_lo = [v["ci_lo"] for k, _, _ in series if k in strat
              for v in strat[k].values() if np.isfinite(v.get("ci_lo", np.nan))]
    inj = d.get("noise_injection_control") or {}
    if inj.get("auroc_after_injecting_noise") == inj.get("auroc_after_injecting_noise"):
        y = inj["auroc_after_injecting_noise"]
        ax.axhline(y, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
        ax.annotate(f"if label noise explained it: {y:.3f}", (len(names) - 0.5, y),
                    textcoords="offset points", xytext=(0, 6), ha="right",
                    color=INK2, fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"|belief| {n}" for n in names])
    ax.set_xlabel("model's confidence in the proposition")
    ax.set_ylabel("deception-probe AUROC")
    ax.set_xlim(-0.5, len(names) - 0.4)
    floor = min(all_lo + [inj.get("auroc_after_injecting_noise") or 1.0])
    ax.set_ylim(floor - 0.05, 1.03)
    # upper-left is the only reliably empty quadrant here: the data rises left
    # to right, and the noise-reference annotation owns the lower right
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2, fontsize=9)
    _finish(ax, "Deception is harder to detect when the model is unsure",
            "one probe, trained across all cells; bars are bootstrap 95% CIs")
    fig.savefig(out, dpi=200); plt.close(fig)
    return True


def fig_variance(rdir: Path, out: Path) -> bool:
    """Where transfer-AUROC variance actually lives. Single series -> no legend."""
    p = rdir / "transfer_long.csv"
    if not p.exists():
        return False
    d = behavior.load_transfer_long(p)
    off = d[~d.is_diag].dropna(subset=["auroc"])
    if len(off) < 10:
        return False
    y = off.auroc.to_numpy()

    def r2(cols):
        X = pd.get_dummies(off[cols].astype(str), drop_first=True).to_numpy(float)
        X = np.column_stack([np.ones(len(y)), X])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        return float(1 - r @ r / ((y - y.mean()) ** 2).sum())

    items = [("what the probe was TESTED on\n(target cell)", r2(["target_cell"]), BLUE),
             ("what the probe was TRAINED on\n(source cell)", r2(["source_cell"]), ORANGE),
             ("scenario labels\n(the H1 baseline model)", r2(["scenario_source",
                                                             "scenario_target"]), AQUA)]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ypos = np.arange(len(items))[::-1]
    for (lab, v, col), yy in zip(items, ypos):
        ax.barh(yy, v, height=0.5, color=col, zorder=3)
        ax.annotate(f"R² = {v:.3f}", (v, yy), textcoords="offset points", xytext=(8, 0),
                    va="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_yticks(ypos); ax.set_yticklabels([i[0] for i in items], color=INK2)
    ax.set_xlabel("share of transfer-AUROC variance explained")
    ax.set_xlim(0, 1.0)
    _finish(ax, "“Transfer” is mostly a property of the target, not of the mismatch",
            f"{len(off)} off-diagonal cell pairs")
    fig.savefig(out, dpi=200); plt.close(fig)
    return True


def fig_heatmap(rdir: Path, out: Path) -> bool:
    """Sequential single hue, light->dark, scaled to the data's actual range.

    A 0-1 scale would waste almost the whole ramp: everything here sits above 0.85, and
    the structure worth seeing is the vertical banding (whole target columns easy or
    hard regardless of source).
    """
    p = rdir / "transfer_matrix.csv"
    if not p.exists():
        return False
    m = pd.read_csv(p, index_col=0)
    if m.empty:
        return False

    def short(c):
        parts = str(c).split("|")
        mode = {"dissimulation": "lying", "simulation": "", "honest": "honest"}.get(parts[0], parts[0])
        sub = {"counterfactual_world": "counterfac", "fictional_frame": "fiction",
               "persona": "persona", "preference_no_truth": "pref", "-": ""}.get(
                   parts[1] if len(parts) > 1 else "", "")
        topic = parts[2] if len(parts) > 2 else ""
        style = {"plain": "", "formal": "formal"}.get(
            parts[3] if len(parts) > 3 else "", "")   # plain is the default; only mark formal
        return " ".join(x for x in (mode or sub, topic, style) if x)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", ["#eaf2fb", "#9cc4ec", "#2a78d6", "#123a68"])
    vmin = float(np.nanmin(m.to_numpy()))
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=vmin, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels([short(c) for c in m.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(m.index)))
    ax.set_yticklabels([short(i) for i in m.index], fontsize=8)
    ax.set_xlabel("tested on (target cell)"); ax.set_ylabel("trained on (source cell)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("AUROC", color=INK2)
    cb.outline.set_edgecolor(MUTED)
    ax.grid(False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.text(0.012, 0.975, "Probe transfer matrix", color=INK, fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.012, 0.935, f"colour scale starts at {vmin:.2f}, not 0 — the whole matrix "
             "is near ceiling; note the vertical banding",
             color=INK2, fontsize=9, va="top")
    fig.savefig(out, dpi=200); plt.close(fig)
    return True


def fig_lierate(rdir: Path, out: Path) -> bool:
    """What the frame actually controls: how OFTEN the model asserts against belief."""
    p = rdir / "compliance.csv"
    if not p.exists():
        return False
    d = pd.read_csv(p)
    col = "lie_rate_vs_belief" if "lie_rate_vs_belief" in d else "lie_rate"
    if col not in d:
        return False
    d = d.dropna(subset=[col]).copy()
    d["name"] = [f"{a}{(' / ' + b) if isinstance(b, str) and b else ''}"
                 for a, b in zip(d.iloc[:, 0], d.iloc[:, 1])]
    d = d.sort_values(col)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ypos = np.arange(len(d))
    ax.barh(ypos, d[col], height=0.55, color=BLUE, zorder=3)
    for yy, v in zip(ypos, d[col]):
        ax.annotate(f"{v:.2f}", (v, yy), textcoords="offset points", xytext=(8, 0),
                    va="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_yticks(ypos); ax.set_yticklabels(d["name"], color=INK2, fontsize=9)
    ax.set_xlim(0, 1.12); ax.set_xlabel("rate of asserting against the model's own belief")
    _finish(ax, "The frame controls how OFTEN the model asserts against its belief",
            "not how deeply each assertion conflicts — that is roughly constant")
    fig.savefig(out, dpi=200); plt.close(fig)
    return True


def fig_baselines(rdir: Path, out: Path) -> bool:
    """Every baseline against the probe, on one axis. Single series -> no legend."""
    p = rdir / "baselines.json"
    if not p.exists():
        return False
    b = json.loads(p.read_text())
    items = [("probe, in-distribution", b.get("in_dist_diag_auroc"), BLUE),
             ("probe, within-class OOD", b.get("within_class_ood_auroc"), BLUE),
             ("probe, style-shifted", b.get("style_shift_auroc"), BLUE),
             ("text-only (TF-IDF)", b.get("behavioral_text_auroc"), ORANGE),
             ("response length only", b.get("length_only_auroc"), ORANGE),
             ("random direction", b.get("random_direction_floor"), MUTED)]
    items = [(l, v, c) for l, v, c in items if isinstance(v, (int, float))]
    if not items:
        return False
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ypos = np.arange(len(items))[::-1]
    for (lab, v, col), yy in zip(items, ypos):
        ax.barh(yy, v, height=0.55, color=col, zorder=3)
        ax.annotate(f"{v:.3f}", (v, yy), textcoords="offset points", xytext=(8, 0),
                    va="center", color=INK, fontsize=10, fontweight="bold")
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_yticks(ypos); ax.set_yticklabels([i[0] for i in items], color=INK2, fontsize=9)
    ax.set_xlim(0, 1.12); ax.set_xlabel("AUROC")
    _finish(ax, "Baselines", "dashed line = chance; the probe must beat every "
                             "non-internal baseline to be interesting")
    fig.savefig(out, dpi=200); plt.close(fig)
    return True


def fig_instrument(rdir: Path, out: Path) -> bool:
    """Why the old label was noise: d_truth's sign is near chance where |t_hat| is small."""
    p = rdir / "truth_axis_meta.json"
    if not p.exists():
        return False
    ia = (json.loads(p.read_text()).get("validity", {}) or {}).get("instrument_agreement")
    if not ia or not ia.get("by_abs_t_hat"):
        return False
    by = ia["by_abs_t_hat"]
    names = [n for n in ("<0.3", "0.3-0.6", ">0.6") if n in by]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    xs = np.arange(len(names))
    ys = [by[n]["sign_agreement"] for n in names]
    ax.plot(xs, ys, "-o", color=BLUE, linewidth=2, markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(f"{y:.3f}\n(n={by[n]['n']})", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", color=INK, fontsize=9, fontweight="bold")
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("chance", (len(names) - 1, 0.5), textcoords="offset points",
                xytext=(0, 6), ha="right", color=INK2, fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([f"|t_hat| {n}" for n in names])
    ax.set_ylim(0.4, 1.08)
    ax.set_ylabel("agreement with independent belief measure")
    _finish(ax, "The fitted truth direction is near chance exactly where it mattered",
            "sign(d_truth projection) vs sign(TRUE/FALSE logit margin), by confidence")
    fig.savefig(out, dpi=200); plt.close(fig)
    return True


def fig_cross_model(models: dict, out: Path) -> bool:
    """The replication picture, on one axis: what held across families and what did not.

    Left panel is the row-level gradient, which replicated everywhere. Right panel is the
    variance split, which did not -- Olmo inverts it. Putting them side by side is the
    honest presentation: one result travelled and one did not, and the reader should see
    both at once rather than only the one that worked.
    """
    import json as _json
    have = {}
    for k in models:
        rl, tl = models[k] / "rowlevel.json", models[k] / "transfer_long.csv"
        if rl.exists() and tl.exists():
            have[k] = (_json.loads(rl.read_text()), tl)
    if len(have) < 2:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    cols = [BLUE, ORANGE, AQUA, "#eda100"]
    names = ["<0.3", "0.3-0.6", ">0.6"]

    ax = axes[0]
    for (k, (d, _)), col in zip(have.items(), cols):
        st = d.get("stratified_auroc", {}).get("groundtruth") or \
             d.get("stratified_auroc", {}).get("belief", {})
        xs = [i for i, n in enumerate(names) if n in st]
        ys = [st[names[i]]["auroc"] for i in xs]
        if not xs:
            continue
        ax.plot(xs, ys, "-o", color=col, linewidth=2, markersize=8, label=k,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        # label at the LEFT end: the three series converge near 0.99 on the right and
        # the labels would sit on top of each other there, but they are well separated
        # at the low-confidence end, which is also the end the figure is about
        ax.annotate(k, (xs[0], ys[0]), textcoords="offset points", xytext=(-8, -3),
                    color=col, fontsize=9, fontweight="bold", ha="right")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"|belief|\n{n}" for n in names])
    ax.set_xlim(-1.35, len(names) - 0.75)
    ax.set_ylabel("deception-probe AUROC")
    ax.set_xlabel("model's confidence in the proposition")
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
    ax.text(0, 1.06, "REPLICATES: harder to detect when unsure", transform=ax.transAxes,
            color=INK, fontsize=11, fontweight="bold", va="bottom")

    ax = axes[1]
    labels, tgt, src = [], [], []
    for k, (_, tl) in have.items():
        d = behavior.load_transfer_long(tl)
        off = d[~d.is_diag].dropna(subset=["auroc"])
        if len(off) < 10:
            continue
        y = off.auroc.to_numpy()

        def r2(c):
            X = pd.get_dummies(off[c].astype(str), drop_first=True).to_numpy(float)
            X = np.column_stack([np.ones(len(y)), X])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ b
            return float(1 - r @ r / ((y - y.mean()) ** 2).sum())
        labels.append(k); tgt.append(r2(["target_cell"])); src.append(r2(["source_cell"]))
    xp = np.arange(len(labels))
    ax.bar(xp - 0.19, tgt, 0.36, color=BLUE, label="tested on (target)", zorder=3)
    ax.bar(xp + 0.19, src, 0.36, color=ORANGE, label="trained on (source)", zorder=3)
    for x, v in zip(xp - 0.19, tgt):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", color=INK, fontsize=9, fontweight="bold")
    for x, v in zip(xp + 0.19, src):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", color=INK, fontsize=9, fontweight="bold")
    ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("R² of transfer AUROC"); ax.set_ylim(0, 0.85)
    # upper-left: the leftmost model is the one with the two short bars
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
    ax.text(0, 1.06, "DOES NOT: Olmo inverts the split", transform=ax.transAxes,
            color=INK, fontsize=11, fontweight="bold", va="bottom")

    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.text(0.012, 0.975, "What replicated across model families, and what did not",
             color=INK, fontsize=13, fontweight="bold", va="top")
    fig.text(0.012, 0.925, "Olmo is also the only model whose probes actually degrade "
             "out of distribution (off-diagonal 0.87 vs 0.97–1.00)",
             color=INK2, fontsize=9, va="top")
    fig.savefig(out, dpi=200); plt.close(fig)
    return True


FIGS = [("rowlevel", fig_rowlevel), ("variance_decomposition", fig_variance),
        ("transfer_heatmap_clean", fig_heatmap), ("lie_rate_by_frame", fig_lierate),
        ("baselines", fig_baselines), ("instrument_agreement", fig_instrument)]


def run(key: str) -> None:
    rdir = config.results_dir(key)
    fdir = rdir / "figures"
    fdir.mkdir(exist_ok=True)
    made = []
    for name, fn in FIGS:
        try:
            if fn(rdir, fdir / f"{name}.png"):
                made.append(name)
            else:
                print(f"  skip {name} (inputs absent)")
        except Exception as e:                                    # noqa: BLE001
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"[{key}] wrote {len(made)} figures -> {fdir}: {', '.join(made)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="default: every model with results")
    a = p.parse_args()
    keys = ([a.model] if a.model else
            sorted(d.name for d in config.RESULTS_DIR.iterdir()
                   if d.is_dir() and (d / "baselines.json").exists()))
    if not keys:
        sys.exit("no results directories found")
    for k in keys:
        run(k)
    # one cross-model figure, written to results/ rather than into any single model's dir
    real = {k: config.results_dir(k) for k in keys
            if "synthetic" not in k and "-dry" not in k}
    if len(real) >= 2:
        outp = config.RESULTS_DIR / "cross_model.png"
        if fig_cross_model(real, outp):
            print(f"wrote {outp}")
