"""Randomly-selected raw transcripts, for reading by a human.

The whole project rests on two judgement calls that nothing else validates:
  1. the hand-written statement bank in src/data_build.py
  2. behavior.parse_verdict — a regex over the first 160 characters of the response,
     which produces `said`, which produces every label downstream

So this dumps a SEEDED RANDOM sample (never cherry-picked) with the full prompt, the
full response, and every derived quantity side by side, so each label can be checked
against the text it came from.

    python scripts/sample_transcripts.py --model qwen3.5-9b --n 30

Writes results/<key>/transcripts_sample.md and .csv.

It also flags one specific pathology found by reading the canary output: the model
emitting `VERDICT: TRUE` and then justifying the opposite --

    "Spiders have ten legs."  ->  'VERDICT: TRUE\\nAll arachnids ... possessing
                                   exactly eight walking [legs]'

The verdict token and the justification disagree. `said` takes the verdict token, so
such a row enters the analysis with a label its own text contradicts. The automated
flag below is a lower bound (it can only catch rows where the verdict also contradicts
ground truth in an honest frame); the `justification_agrees` column is left blank on
purpose, for a human to fill in while reading.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import behavior, config  # noqa: E402
from src.activations import load_cache  # noqa: E402

TRUTH = {1: "TRUE", 0: "FALSE", -1: "no truth value"}


def run(key: str, dry_run: bool, n: int) -> None:
    rdir = config.results_dir(key)
    _, _, texts = load_cache(key)
    man = pd.read_csv(config.manifest_path(dry_run), keep_default_na=False)
    cs = behavior.load_c_scores(rdir / "c_scores.csv")

    df = man.merge(texts, on="id", how="inner").merge(
        cs[[c for c in ("id", "said", "parsed", "t_hat", "b_hat", "b_hat_framed",
                        "c", "actually_lied", "contradicts_truth") if c in cs]],
        on="id", how="left")

    # seeded and stratified across cells: a sample concentrated in one cell would not
    # tell you whether the parser works anywhere else
    rng = pd.Series(range(len(df)))
    sample = (df.sample(frac=1.0, random_state=config.SEED)
                .groupby("cell", group_keys=False)
                .head(max(1, n // max(df.cell.nunique(), 1)) + 1)
                .sample(min(n, len(df)), random_state=config.SEED)
                .sort_values(["cell", "statement_id"]))

    # lower-bound automatic flag for verdict/justification incoherence
    honest_wrong = ((sample["mode"] == "honest") & sample.parsed
                    & (sample.truth_value != -1)
                    & (sample.said != sample.truth_value.map({1: 1, 0: -1})))
    sample = sample.assign(honest_verdict_wrong=honest_wrong)

    cols = ["id", "cell", "mode", "sim_subtype", "topic", "difficulty", "style", "split",
            "statement", "truth_value", "asserted_true", "prompt", "text", "said",
            "parsed", "t_hat", "b_hat", "b_hat_framed", "c", "actually_lied",
            "honest_verdict_wrong"]
    out = sample[[c for c in cols if c in sample]].copy()
    out["justification_agrees"] = ""      # for a human to fill in while reading
    out.to_csv(rdir / "transcripts_sample.csv", index=False)

    md = [f"# Randomly selected transcripts — `{key}`", "",
          f"{len(out)} rows, seeded (`config.SEED = {config.SEED}`) and spread across cells. "
          "Not cherry-picked. Every derived quantity is shown next to the text it was "
          "derived from, so the labels can be checked against the raw output.", "",
          "`justification_agrees` is deliberately blank — fill it in by reading, then "
          "report the rate.", ""]

    for i, r in enumerate(out.itertuples(), 1):
        md += [f"## {i}. `{r.cell}`", "",
               f"- **statement**: {r.statement}",
               f"- **ground truth**: {TRUTH.get(r.truth_value, r.truth_value)}",
               f"- **task required asserting**: "
               f"{'TRUE' if r.asserted_true == 1 else 'FALSE'}",
               f"- **split**: {r.split}", "",
               "<details><summary>full prompt</summary>", "",
               "```", textwrap.dedent(str(r.prompt)).strip(), "```", "",
               "</details>", "",
               "**response**", "", "```", str(r.text).strip(), "```", "",
               f"- parsed verdict (`said`): "
               f"**{ {1: 'TRUE', -1: 'FALSE', 0: 'UNPARSED'}.get(r.said, r.said) }**",
               f"- belief margin under this frame (`b_hat_framed`): "
               f"{getattr(r, 'b_hat_framed', float('nan')):+.3f}",
               f"- honest-frame belief (`b_hat`): {getattr(r, 'b_hat', float('nan')):+.3f}",
               f"- truth-direction projection (`t_hat`): "
               f"{getattr(r, 't_hat', float('nan')):+.3f}",
               f"- conflict (`c`): {getattr(r, 'c', float('nan')):.3f}",
               f"- label (`actually_lied`): {getattr(r, 'actually_lied', float('nan'))}"]
        if getattr(r, "honest_verdict_wrong", False):
            md += ["", "> **FLAG** — honest frame, but the verdict contradicts ground "
                   "truth. Read the justification: if it argues the opposite of the "
                   "verdict token, this is verdict/justification incoherence and the "
                   "row's label is not trustworthy."]
        md += ["", "---", ""]

    (rdir / "transcripts_sample.md").write_text("\n".join(md), encoding="utf-8")

    n_flag = int(out.honest_verdict_wrong.sum())
    print(f"wrote {rdir / 'transcripts_sample.md'} and .csv ({len(out)} rows)")
    print(f"auto-flagged {n_flag} honest rows whose verdict contradicts ground truth "
          "(lower bound on verdict/justification incoherence)")
    if "actually_lied" in out:
        print(f"label distribution in sample: "
              f"{out.actually_lied.value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.PRIMARY_MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n", type=int, default=30)
    a = p.parse_args()
    run(config.effective_key(a.model, a.synthetic, a.dry_run), a.dry_run, a.n)
