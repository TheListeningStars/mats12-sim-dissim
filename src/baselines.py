"""Baselines and the PRIMARY result: the scenario-vs-c horse race (H1).

The project succeeds only if the measured truth-conflict axis c explains transfer
performance BEYOND the pragmatic scenario taxonomy. This module runs that test.

Inference note (deliberate): the (source, target) AUROC observations are NOT
independent — every pair shares its source cell with ~n other pairs. Naive OLS
F-test p-values are therefore optimistic. We report three things:
  1. the naive nested F-test (for comparability, flagged as optimistic),
  2. cluster-robust coefficient tests (clustered on source cell),
  3. a CELL-LEVEL permutation test: the c value attached to each framed cell is
     shuffled among cells sharing the same scenario label, and the ΔR² of the
     c-terms over the scenario-only model is recomputed. This directly asks
     "does the WITHIN-scenario variation of c predict transfer?" — which is
     exactly H1's 'adds variance beyond scenario labels'.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import behavior, config, probes
from .activations import load_cache

F_SCEN = "C(scenario_source) + C(scenario_target)"
F_C = "c_source + c_target + abs_dc"


def _fit(formula: str, df: pd.DataFrame):
    import statsmodels.formula.api as smf
    return smf.ols(f"auroc ~ {formula}", data=df).fit()


def horse_race(transfer_long: pd.DataFrame, n_perm: int = 500) -> dict:
    """Nested model comparison on transfer AUROC (PLAN §3, §8, H1).
        M0: scenario labels only    M1: c only    M2: both
    Returns the stats dict; caller saves results/horse_race.csv."""
    from statsmodels.stats.anova import anova_lm

    df = transfer_long[~transfer_long.is_diag].copy()
    df = df.dropna(subset=["auroc", "c_source", "c_target", "abs_dc"])
    if len(df) < 10 or df.scenario_source.nunique() < 2:
        return {"error": f"too few usable pairs for the horse race (n={len(df)})",
                "n_pairs": int(len(df))}
    m0, m1, m2 = _fit(F_SCEN, df), _fit(F_C, df), _fit(f"{F_SCEN} + {F_C}", df)
    delta_r2 = m2.rsquared - m0.rsquared

    an = anova_lm(m0, m2)
    naive_p = float(an["Pr(>F)"].iloc[1])

    m2_clu = _fit(f"{F_SCEN} + {F_C}", df).get_robustcov_results(
        cov_type="cluster", groups=df.source_cell.astype("category").cat.codes)
    cluster_p = {name: float(p) for name, p in
                 zip(m2_clu.model.exog_names, m2_clu.pvalues)
                 if name in ("c_source", "c_target", "abs_dc")}

    # cell-level permutation: shuffle c among cells WITHIN each scenario group
    rng = np.random.default_rng(config.SEED)
    src_cells = df[["source_cell", "scenario_source", "c_source"]].drop_duplicates("source_cell")
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = src_cells.copy()
        perm["c_perm"] = perm.groupby("scenario_source").c_source.transform(rng.permutation)
        cmap = perm.set_index("source_cell").c_perm
        d = df.copy()
        d["c_source"] = d.source_cell.map(cmap)
        d["c_target"] = d.target_cell.map(cmap)
        d["abs_dc"] = (d.c_source - d.c_target).abs()
        null[i] = _fit(f"{F_SCEN} + {F_C}", d).rsquared - m0.rsquared
    perm_p = float((1 + (null >= delta_r2).sum()) / (n_perm + 1))

    return {
        "r2_scenario_only_M0": float(m0.rsquared),
        "r2_c_only_M1": float(m1.rsquared),
        "r2_both_M2": float(m2.rsquared),
        "delta_r2_M2_vs_M0": float(delta_r2),
        "naive_F_p": naive_p,
        "cluster_robust_p": cluster_p,
        "perm_p_within_scenario": perm_p,
        "n_perm": n_perm,
        "aic_M0": float(m0.aic), "aic_M1": float(m1.aic), "aic_M2": float(m2.aic),
        "n_pairs": int(len(df)),
    }


def within_class_baseline(long: pd.DataFrame) -> float:
    """Mean AUROC for same mode+subtype+style, topic A -> B: the 'normal' OOD drop
    the cross-c drop must beat for the axis to matter (PLAN §8)."""
    m = long[(~long.is_diag)
             & (long.mode_source == long.mode_target)
             & (long.subtype_source == long.subtype_target)
             & (long.style_source == long.style_target)
             & (long.topic_source != long.topic_target)]
    return float(m.auroc.mean())


def style_shift_baseline(long: pd.DataFrame) -> float:
    """Mean AUROC under style re-rendering only (same mode+subtype+topic);
    the arXiv:2605.27958 surface-style control."""
    m = long[(~long.is_diag)
             & (long.mode_source == long.mode_target)
             & (long.subtype_source == long.subtype_target)
             & (long.topic_source == long.topic_target)
             & (long.style_source != long.style_target)]
    return float(m.auroc.mean()) if len(m) else float("nan")


def _labels(d: pd.DataFrame, label_source: str):
    """Same labelling rule as transfer.py, so baselines are comparable to the matrix."""
    if label_source == "behavior":
        ok = d[d.actually_lied.notna()]
        return ok, ok.actually_lied.to_numpy().astype(int)
    return d, (d["mode"] != "honest").astype(int).to_numpy()


def behavioral_baseline(texts: pd.DataFrame, df: pd.DataFrame,
                        label_source: str = "behavior") -> float:
    """Can the label be predicted from OUTPUT TEXT alone? TF-IDF + logistic regression
    (a cheap stand-in for an LLM judge). Internal probes must add information beyond
    this to matter.

    Note this baseline is expected to stay HIGH under behaviour labelling — the verdict
    token is literally in the text, so text trivially reveals what was asserted. Its job
    here is to bound how much the internal probe is adding over surface reading, not to
    be beaten; interpret it against the *condition*-labelled value, where a high score
    means template leakage instead."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    d = df[df.split != "truthfit"].merge(texts, on="id")
    d, y = _labels(d, label_source)
    d = d.assign(y=y)
    tr, ev = d[d.split == "train"], d[d.split == "eval"]
    if tr.y.nunique() < 2 or ev.y.nunique() < 2:
        return float("nan")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=2000, random_state=config.SEED) \
        .fit(vec.fit_transform(tr.text.fillna("")), tr.y)
    return float(roc_auc_score(ev.y, clf.predict_proba(vec.transform(ev.text.fillna("")))[:, 1]))


def length_baseline(texts: pd.DataFrame, df: pd.DataFrame,
                    label_source: str = "behavior") -> float:
    """Red-team control (PLAN §8): AUROC from response LENGTH alone."""
    from sklearn.metrics import roc_auc_score
    d = df[df.split == "eval"].merge(texts, on="id")
    d, y = _labels(d, label_source)
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, d.text.fillna("").str.len()))


def random_floor(df: pd.DataFrame, ids: np.ndarray, acts: np.ndarray,
                 label_source: str = "behavior") -> float:
    """Mean random-direction AUROC over the pooled eval set."""
    ev, y = _labels(df[df.split == "eval"], label_source)
    if len(set(y)) < 2:
        return float("nan")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    return probes.random_direction_auroc(acts[pos.loc[ev.id].to_numpy()], y)


def run(key: str, dry_run: bool = False, label_source: str = "behavior") -> None:
    rdir = config.results_dir(key)
    tl = rdir / "transfer_long.csv"
    if not tl.exists():
        print("!! no transfer_long.csv — transfer stage aborted; skipping baselines.")
        return
    long = behavior.load_transfer_long(tl)
    df = pd.read_csv(config.manifest_path(dry_run), keep_default_na=False)
    cs = behavior.load_c_scores(rdir / "c_scores.csv")
    df = df.merge(cs[["id", "c", "actually_lied", "parsed"]], on="id", how="inner")
    layer = json.loads((rdir / "truth_axis_meta.json").read_text())["best_layer"]
    _, layers, texts = load_cache(key)
    ids, acts = layers[layer]
    df = df[df.id.isin(ids)]

    hr = horse_race(long)
    res = {
        "label_source": label_source,
        "horse_race": hr,
        "in_dist_diag_auroc": float(long[long.is_diag].auroc.mean()),
        "within_class_ood_auroc": within_class_baseline(long),
        "style_shift_auroc": style_shift_baseline(long),
        "behavioral_text_auroc": behavioral_baseline(texts, df, label_source),
        "length_only_auroc": length_baseline(texts, df, label_source),
        "random_direction_floor": random_floor(df, ids, acts, label_source),
    }
    (rdir / "baselines.json").write_text(json.dumps(res, indent=2))
    if "error" in hr:
        config.log(f"H1 horse race SKIPPED ({hr['error']}); labels={label_source}", key)
        print(json.dumps(res, indent=2))
        print("horse race skipped:", hr["error"])
        return
    pd.DataFrame([{
        "model": ["M0 scenario", "M1 c-only", "M2 both"][i],
        "r2": [hr["r2_scenario_only_M0"], hr["r2_c_only_M1"], hr["r2_both_M2"]][i],
        "aic": [hr["aic_M0"], hr["aic_M1"], hr["aic_M2"]][i],
    } for i in range(3)]).to_csv(rdir / "horse_race.csv", index=False)

    verdict = ("c ADDS variance beyond scenario" if hr["perm_p_within_scenario"] < 0.05
               else "c does NOT add variance beyond scenario (negative result branch)")
    config.log(
        f"H1 horse race (labels={label_source}): R² scenario {hr['r2_scenario_only_M0']:.3f} | "
        f"c-only {hr['r2_c_only_M1']:.3f} | both {hr['r2_both_M2']:.3f}; "
        f"ΔR² {hr['delta_r2_M2_vs_M0']:.3f} (perm p={hr['perm_p_within_scenario']:.3f}, "
        f"naive F p={hr['naive_F_p']:.2g}) → {verdict}. "
        f"Baselines: diag {res['in_dist_diag_auroc']:.3f}, within-class OOD "
        f"{res['within_class_ood_auroc']:.3f}, style {res['style_shift_auroc']:.3f}, "
        f"behavioral-text {res['behavioral_text_auroc']:.3f}, length-only "
        f"{res['length_only_auroc']:.3f}, random {res['random_direction_floor']:.3f}",
        key)
    print(json.dumps(res, indent=2))
    print("verdict:", verdict)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--label-source", default="behavior", choices=["behavior", "condition"])
    a = p.parse_args()
    run(config.effective_key(a.model, a.synthetic, a.dry_run), a.dry_run, a.label_source)
