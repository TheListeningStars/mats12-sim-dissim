"""Independent re-derivation of every headline number.

This exists because of one line in the MATS application doc:

    "Verify the load-bearing claims. For each key result: read the code that produced
     it, check the numbers in the write-up against the actual outputs, re-derive at
     least some of them independently."

So nothing here imports from src/baselines.py, src/transfer.py or src/geometry.py.
Every quantity is recomputed from the saved CSVs with fresh code — plain numpy/pandas,
written to a different recipe than the pipeline's — and then compared against what the
pipeline wrote into baselines.json / geometry.json / truth_axis_meta.json. A quantity
that only agrees with itself has not been checked.

    python scripts/verify.py --model qwen3.5-9b
    python scripts/verify.py --model qwen2.5-7b-instruct --dry-run

Writes results/<key>/verification.md and prints the same table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

TOL = 1e-6          # agreement tolerance for quantities that should match exactly
N_PERM = 5000


# --- small self-contained stats, deliberately not the pipeline's implementations ----

def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R² of y on [1, X] via lstsq. X may have zero columns (intercept-only -> 0.0)."""
    X = np.column_stack([np.ones(len(y)), X]) if np.size(X) else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    tss = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid @ resid) / tss) if tss > 0 else float("nan")


def _dummies(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0))
    return pd.get_dummies(df[cols].astype(str), drop_first=True).to_numpy(float)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with tie correction. Independent of sklearn."""
    labels = np.asarray(labels).astype(int)
    pos, neg = (labels == 1).sum(), (labels == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    s = np.asarray(scores)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


# --- checks -------------------------------------------------------------------------

def _load_transfer(rdir: Path) -> pd.DataFrame:
    d = pd.read_csv(rdir / "transfer_long.csv", keep_default_na=False)
    for c in ("auroc", "c_source", "c_target", "abs_dc"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["is_diag"] = d["is_diag"].astype(str).str.lower().isin(("true", "1"))
    return d


def check_horse_race(rdir: Path, reported: dict) -> list[dict]:
    """Re-fit M0 / M1 / M2 from transfer_long.csv and compare R², ΔR², n."""
    d = _load_transfer(rdir)
    off = d[~d.is_diag].dropna(subset=["auroc", "c_source", "c_target", "abs_dc"])
    y = off.auroc.to_numpy()
    D0 = _dummies(off, ["scenario_source", "scenario_target"])
    Dc = off[["c_source", "c_target", "abs_dc"]].to_numpy(float)

    r0, r1, r2 = _ols_r2(D0, y), _ols_r2(Dc, y), _ols_r2(np.column_stack([D0, Dc]), y)
    hr = reported.get("horse_race", {})
    return [
        _row("horse race: R² M0 (scenario)", hr.get("r2_scenario_only_M0"), r0),
        _row("horse race: R² M1 (c only)", hr.get("r2_c_only_M1"), r1),
        _row("horse race: R² M2 (both)", hr.get("r2_both_M2"), r2),
        _row("horse race: ΔR² (M2−M0)", hr.get("delta_r2_M2_vs_M0"), r2 - r0),
        _row("horse race: n off-diag pairs", hr.get("n_pairs"), len(off)),
    ]


def check_cell_level_null(rdir: Path) -> list[dict]:
    """The test the pipeline does NOT run.

    `c` is a property of a cell, not of a pair, so the null that matters permutes the
    cell -> c assignment and rebuilds c_source / c_target / |dc| from it. Permuting at
    the pair level (or treating the pairs as independent, as the naive F does) invents
    degrees of freedom: with k cells there are k(k-1) pairs but only k independent units.
    """
    d = _load_transfer(rdir)
    off = d[~d.is_diag].dropna(subset=["auroc", "c_source", "c_target", "abs_dc"])
    y = off.auroc.to_numpy()
    D0 = _dummies(off, ["scenario_source", "scenario_target"])

    def dr2(cs, ct):
        Dc = np.column_stack([cs, ct, np.abs(cs - ct)])
        return _ols_r2(np.column_stack([D0, Dc]), y) - _ols_r2(D0, y)

    obs = dr2(off.c_source.to_numpy(), off.c_target.to_numpy())
    cmap = dict(zip(off.source_cell, off.c_source))
    cells = sorted(cmap)
    vals = np.array([cmap[c] for c in cells])

    rng = np.random.default_rng(config.SEED)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        p = dict(zip(cells, rng.permutation(vals)))
        null[i] = dr2(off.source_cell.map(p).to_numpy(), off.target_cell.map(p).to_numpy())
    p_cell = float((null >= obs).mean())

    return [
        {"quantity": "cell-level permutation p for ΔR²", "pipeline": "not computed",
         "independent": f"{p_cell:.3f}", "match": "NEW",
         "note": f"n_cells={len(cells)}, n_perm={N_PERM}; null mean ΔR²={null.mean():.4f}, "
                 f"observed={obs:.4f}"},
        {"quantity": "observed ΔR² vs its own null mean", "pipeline": "—",
         "independent": f"{obs:.4f} vs {null.mean():.4f}",
         "match": "BELOW NULL MEAN" if obs < null.mean() else "above null mean",
         "note": "if observed sits below the null mean, random cell->c assignment "
                 "explains as much or more than the real assignment"},
    ]


def check_variance_decomposition(rdir: Path) -> list[dict]:
    """How much of transfer AUROC is source identity vs target identity?

    If target identity dominates, 'transfer' is mostly target-cell difficulty and the
    train/test mismatch that H1 is about is a small component of the variance.
    """
    d = _load_transfer(rdir)
    off = d[~d.is_diag].dropna(subset=["auroc"])
    y = off.auroc.to_numpy()
    out = []
    for label, cols in (("source cell alone", ["source_cell"]),
                        ("target cell alone", ["target_cell"]),
                        ("source + target", ["source_cell", "target_cell"]),
                        ("scenario src+tgt (=M0)", ["scenario_source", "scenario_target"])):
        out.append({"quantity": f"variance decomp: R² {label}", "pipeline": "not computed",
                    "independent": f"{_ols_r2(_dummies(off, cols), y):.3f}",
                    "match": "NEW", "note": ""})
    return out


def check_label_circularity(rdir: Path) -> list[dict]:
    """Is the |t_hat|-stratified row-level result a finding, or label noise?

    `actually_lied` is sign(said)·sign(t_hat) < 0, so where |t_hat| ~ 0 the label's own
    sign is near a coin flip. A probe scored against a partly-randomised label loses
    AUROC whether or not deception actually got harder to detect. `contradicts_truth`
    is defined against ground truth and never touches t_hat, so it is the control: if
    the gradient survives on it, the effect is real.
    """
    p = rdir / "c_scores.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p, keep_default_na=False)
    for c in ("t_hat", "c", "actually_lied", "contradicts_truth", "said", "truth_value"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["parsed"] = d["parsed"].astype(str).str.lower().isin(("true", "1"))
    d = d[(d.split != "truthfit") & (d.truth_value != -1) & d.t_hat.notna() & d.parsed]
    if "contradicts_truth" not in d or d.contradicts_truth.isna().all():
        truth_sign = np.where(d.truth_value == 1, 1, -1)
        d["contradicts_truth"] = (d.said.to_numpy() != truth_sign).astype(float)

    d["bin"] = pd.cut(d.t_hat.abs(), [0, 0.3, 0.6, 1.01],
                      labels=["|t̂|<0.3", "0.3–0.6", ">0.6"])
    out = []
    for b, g in d.groupby("bin", observed=True):
        agree = float((g.actually_lied == g.contradicts_truth).mean())
        out.append({"quantity": f"belief-label vs ground-truth-label agreement, {b}",
                    "pipeline": "not computed", "independent": f"{agree:.3f}",
                    "match": "NEW",
                    "note": f"n={len(g)}; agreement falling toward 0.5 in low bins means the "
                            f"LABEL is noisy there, not that deception is harder to detect"})
    return out


def check_baselines(rdir: Path, reported: dict) -> list[dict]:
    """Diagonal AUROC recomputed as the mean of is_diag rows in transfer_long."""
    d = _load_transfer(rdir)
    diag = d[d.is_diag].auroc.dropna()
    return [_row("in-distribution (diagonal) AUROC",
                 reported.get("in_dist_diag_auroc"), float(diag.mean()))]


def check_monotonicity(rdir: Path) -> list[dict]:
    """H2 Spearman, recomputed from geometry.json's own per-target table."""
    p = rdir / "geometry.json"
    if not p.exists():
        return []
    g = json.loads(p.read_text())
    per = pd.DataFrame(g.get("per_target", []))
    if per.empty or "c" not in per:
        return []
    a, c = per.auroc.to_numpy(float), per.c.to_numpy(float)
    ra, rc = pd.Series(a).rank().to_numpy(), pd.Series(c).rank().to_numpy()
    rho = float(np.corrcoef(ra, rc)[0, 1])
    return [_row("H2 Spearman ρ", g.get("spearman_rho"), rho),
            {"quantity": "H2 n targets", "pipeline": "—", "independent": str(len(per)),
             "match": "NEW", "note": "this is the true sample size for H2"}]


def _row(name, pipeline, independent) -> dict:
    if pipeline is None:
        return {"quantity": name, "pipeline": "absent", "independent": f"{independent}",
                "match": "?", "note": ""}
    try:
        ok = abs(float(pipeline) - float(independent)) <= max(TOL, abs(float(pipeline)) * 1e-6)
        fp, fi = f"{float(pipeline):.6f}", f"{float(independent):.6f}"
    except (TypeError, ValueError):
        ok, fp, fi = str(pipeline) == str(independent), str(pipeline), str(independent)
    return {"quantity": name, "pipeline": fp, "independent": fi,
            "match": "OK" if ok else "**MISMATCH**", "note": ""}


def run(key: str) -> None:
    rdir = config.results_dir(key)
    bpath = rdir / "baselines.json"
    if not bpath.exists():
        sys.exit(f"no baselines.json in {rdir} — run the pipeline first")
    reported = json.loads(bpath.read_text())

    rows: list[dict] = []
    rows += check_horse_race(rdir, reported)
    rows += check_baselines(rdir, reported)
    rows += check_monotonicity(rdir)
    rows += check_cell_level_null(rdir)
    rows += check_variance_decomposition(rdir)
    rows += check_label_circularity(rdir)

    tbl = pd.DataFrame(rows)
    mismatches = [r for r in rows if r["match"].startswith("**")]

    md = [f"# Verification — `{key}`", "",
          "Every number below was recomputed from the saved CSVs by `scripts/verify.py`, which "
          "imports nothing from the analysis modules. `pipeline` is what the pipeline wrote; "
          "`independent` is what fresh code gets. Rows marked NEW are checks the pipeline does "
          "not perform at all.", "",
          tbl.to_markdown(index=False), ""]
    if mismatches:
        md += [f"## {len(mismatches)} MISMATCH(ES) — do not use these numbers", ""]
        md += [f"- {m['quantity']}: pipeline {m['pipeline']} vs independent {m['independent']}"
               for m in mismatches]
    else:
        md += ["All directly-comparable quantities agree with the pipeline.", ""]

    (rdir / "verification.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {rdir / 'verification.md'}")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    run(config.effective_key(a.model, a.synthetic, a.dry_run))
