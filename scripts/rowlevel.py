"""Row-level detectability, with the control that decides whether it is real.

The lab log calls this the project's "NEW POSITIVE": one probe, trained across cells,
detects deception near-perfectly when the model is confident about the proposition and
much less well when it is not (AUROC 0.99 for |t_hat| > 0.6 vs 0.80 for |t_hat| < 0.3).
It existed in no code and could not be reproduced by run_all.py.

There is an obvious deflationary explanation, and it must be ruled out before the result
is written up as a finding:

    the old label was actually_lied = sign(said) * sign(t_hat) < 0, so where |t_hat| ~ 0
    the LABEL's own sign is close to a coin flip. A probe scored against a partly
    randomised label loses AUROC no matter what the representation does. Symmetric label
    noise at rate p attenuates a true AUROC A to  p*A + (1-p)*(1-A)  -- which reproduced
    the logged numbers to three decimals.

So this script computes the stratified AUROC three ways and prints them together:

  belief      label = asserted against the model's own belief, measured by the
              TRUE/FALSE logit margin (b_hat). Independent of d_truth, so the low-|t_hat|
              stratum is no longer defined by a near-random sign.
  groundtruth label = asserted against ground truth (contradicts_truth). Involves
              neither d_truth nor the margin. The cleanest control: if the gradient
              survives here it is not an artifact of how the label was built.
  noise-inject take the CLEAN high-confidence stratum, corrupt its labels at the rate
              actually measured in the low stratum, and recompute. If that lands on the
              observed low-stratum AUROC, label noise explains the whole effect and
              there is nothing else to report.

    python scripts/rowlevel.py --model qwen3.5-9b

Writes results/<key>/rowlevel.json and rowlevel.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import behavior, config, probes  # noqa: E402
from src.activations import load_cache  # noqa: E402

BINS = [0.0, 0.3, 0.6, 1.01]
NAMES = ["<0.3", "0.3-0.6", ">0.6"]
N_BOOT = 2000


def _auroc_ci(scores: np.ndarray, labels: np.ndarray, seed: int) -> tuple:
    """AUROC with a bootstrap CI. The CI is the point: per-stratum n is small enough
    that a bare AUROC invites over-reading, which is how the original claim happened."""
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    a = probes.auroc(scores, labels)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        boots.append(probes.auroc(scores[idx], labels[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(a), float(lo), float(hi), int(len(labels))


def _acts_for(ids_wanted, ids, acts):
    pos = pd.Series(np.arange(len(ids)), index=ids)
    return acts[pos.loc[ids_wanted].to_numpy()]


def run(key: str, dry_run: bool, site: str) -> None:
    rdir = config.results_dir(key)
    meta, layers, _ = load_cache(key, site=site)
    layer = json.loads((rdir / "truth_axis_meta.json").read_text())["best_layer"]
    ids, acts = layers[layer]

    d = behavior.load_c_scores(rdir / "c_scores.csv")
    d = d[(d.split != "truthfit") & (d.truth_value != -1) & d.parsed]
    if "contradicts_truth" in d:
        d = d[d.contradicts_truth.notna()]
    d = d[d.id.isin(set(ids))]

    results, rows = {}, []
    for label_name, col in (("belief", "actually_lied"), ("groundtruth", "contradicts_truth")):
        if col not in d:
            continue
        sub = d[d[col].notna()]
        y = sub[col].to_numpy().astype(int)
        X = _acts_for(sub.id, ids, acts)
        tr = (sub.split == "train").to_numpy()
        if len(np.unique(y[tr])) < 2 or tr.sum() < 20:
            continue
        # ONE probe, trained once across all cells, then evaluated within strata --
        # the strata must not each get their own probe or the comparison is meaningless
        p = probes.train_logreg(X[tr], y[tr], layer)
        ev = ~tr
        s_all, y_all = p.score(X[ev]), y[ev]
        strat = pd.cut(sub.b_hat.abs()[ev], BINS, labels=NAMES)

        per = {}
        a, lo, hi, n = _auroc_ci(s_all, y_all, config.SEED)
        per["all"] = dict(auroc=a, ci_lo=lo, ci_hi=hi, n=n, positive_rate=float(y_all.mean()))
        for nm in NAMES:
            m = (strat == nm).to_numpy()
            if m.sum() < 10:
                continue
            a, lo, hi, n = _auroc_ci(s_all[m], y_all[m], config.SEED)
            per[nm] = dict(auroc=a, ci_lo=lo, ci_hi=hi, n=n,
                           positive_rate=float(y_all[m].mean()))
            rows.append(dict(label=label_name, stratum=nm, auroc=a, ci_lo=lo, ci_hi=hi,
                             n=n, positive_rate=float(y_all[m].mean())))
        results[label_name] = per

    # --- the noise-injection control ---------------------------------------------
    # How noisy is the belief label where |b_hat| is small? Measure it against ground
    # truth, which is independent of both b_hat and d_truth.
    noise = {}
    if {"actually_lied", "contradicts_truth"}.issubset(d.columns):
        sub = d[d.actually_lied.notna() & d.contradicts_truth.notna()]
        strat = pd.cut(sub.b_hat.abs(), BINS, labels=NAMES)
        for nm in NAMES:
            g = sub[(strat == nm).to_numpy()]
            if len(g):
                noise[nm] = float((g.actually_lied == g.contradicts_truth).mean())

    inject = {}
    if "belief" in results and ">0.6" in results["belief"] and "<0.3" in noise:
        sub = d[d.actually_lied.notna()]
        y = sub.actually_lied.to_numpy().astype(int)
        X = _acts_for(sub.id, ids, acts)
        tr = (sub.split == "train").to_numpy()
        p = probes.train_logreg(X[tr], y[tr], layer)
        ev = ~tr
        strat = pd.cut(sub.b_hat.abs()[ev], BINS, labels=NAMES)
        m = (strat == ">0.6").to_numpy()
        s_clean, y_clean = p.score(X[ev])[m], y[ev][m]
        flip_rate = 1.0 - noise["<0.3"]          # disagreement seen in the noisy stratum
        rng = np.random.default_rng(config.SEED)
        vals = []
        for _ in range(200):
            yy = y_clean.copy()
            f = rng.random(len(yy)) < flip_rate
            yy[f] = 1 - yy[f]
            if len(np.unique(yy)) > 1:
                vals.append(probes.auroc(s_clean, yy))
        inject = {
            "flip_rate_applied": flip_rate,
            "clean_auroc": results["belief"][">0.6"]["auroc"],
            "auroc_after_injecting_noise": float(np.mean(vals)) if vals else float("nan"),
            "observed_low_stratum_auroc": results["belief"].get("<0.3", {}).get("auroc"),
        }

    out = {"site": site, "layer": layer, "label_agreement_by_stratum": noise,
           "stratified_auroc": results, "noise_injection_control": inject}
    # site goes in the filename: the response-site and prompt-site answers are different
    # measurements of different things, and one silently overwriting the other would be
    # the exact class of mistake this project keeps finding.
    sfx = "" if site == "response" else f"_{site}"
    (rdir / f"rowlevel{sfx}.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(rdir / f"rowlevel{sfx}.csv", index=False)

    print(f"=== row-level detectability ({key}, site={site}, layer={layer}) ===\n")
    for label_name, per in results.items():
        print(f"-- label = {label_name} --")
        for nm, v in per.items():
            print(f"   |b_hat| {nm:8s} n={v['n']:4d}  AUROC {v['auroc']:.3f} "
                  f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]  pos_rate {v['positive_rate']:.2f}")
        print()
    if noise:
        print("belief-label vs ground-truth-label agreement by stratum:")
        for nm, v in noise.items():
            print(f"   |b_hat| {nm:8s} {v:.3f}")
        print()
    if inject:
        print("NOISE-INJECTION CONTROL")
        print(f"   clean (>0.6) AUROC                    {inject['clean_auroc']:.3f}")
        print(f"   after injecting {inject['flip_rate_applied']:.2f} label noise   "
              f"{inject['auroc_after_injecting_noise']:.3f}")
        print(f"   actually observed in <0.3 stratum     "
              f"{inject['observed_low_stratum_auroc']:.3f}")
        print("   If the middle line matches the last, label noise explains the effect.")
    print(f"\nwrote {rdir / ('rowlevel' + sfx + '.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--site", default="response", choices=["response", "prompt"])
    a = p.parse_args()
    run(config.effective_key(a.model, a.synthetic, a.dry_run), a.dry_run, a.site)
