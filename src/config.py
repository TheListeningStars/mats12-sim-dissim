"""Central configuration. All paths, model ids, layers, and the RNG seed live here."""
from __future__ import annotations

import datetime
from pathlib import Path

SEED = 0

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"          # cached activations
PROBE_DIR = ROOT / "probes"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, CACHE_DIR, PROBE_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)


def manifest_path(dry_run: bool = False) -> Path:
    """Dry-run uses its own manifest so a tiny smoke test never clobbers the real one."""
    return DATA_DIR / ("manifest_dry.csv" if dry_run else "manifest.csv")


def effective_key(model_key: str, synthetic: bool = False, dry_run: bool = False) -> str:
    """Cache/results key. Synthetic and dry runs get separate directories so fake or
    tiny-N artifacts can never be mistaken for real results."""
    return model_key + ("-synthetic" if synthetic else "") + ("-dry" if dry_run else "")


def cache_dir(key: str) -> Path:
    d = CACHE_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(key: str) -> Path:
    d = RESULTS_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg: str, key: str = "") -> None:
    """Append a timestamped line to the running lab log (results/LOG.md)."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    prefix = f"[{key}] " if key else ""
    with open(RESULTS_DIR / "LOG.md", "a", encoding="utf-8") as f:
        f.write(f"- {stamp} {prefix}{msg}\n")


# model key -> HF id. Deliberately modern models (see PLAN: no GPT-2/Pythia/Gemma-2).
MODELS = {
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",   # gated: huggingface-cli login
    "gemma-3-12b-it": "google/gemma-3-12b-it",                     # optional
}
PRIMARY_MODEL = "qwen2.5-7b-instruct"

# residual-stream layers to sweep, as fractions of depth (resolved per-model in activations.py)
LAYER_FRACTIONS = (0.6, 0.7, 0.8)

LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 32

# dataset design -----------------------------------------------------------
MODES = ("honest", "dissimulation", "simulation")
# simulation subtypes ordered by *expected* truth-conflict c (high -> low). Verified empirically.
SIM_SUBTYPES = ("counterfactual_world", "fictional_frame", "persona", "preference_no_truth")
TOPICS = ("science", "history", "geography", "everyday")   # for within-class OOD (train A -> test B)
STYLES = ("plain", "formal")                                # for style-shift control

# Fact difficulty. 'easy' = unambiguous facts with absurd falsehoods; 'hard' = plausible
# near-miss falsehoods and common misconceptions. On easy-only banks |t_hat| never fell
# below 0.30 (10th pct 0.414) — the model is confident about everything, every lie is a
# confident lie, and probes separate lies from truths trivially regardless of framing.
# Hard facts exist to spread |t_hat| into the uncertain region, which is the precondition
# for the transfer question and for c having any within-cell variance.
# d_truth is always fit on EASY truthfit pairs only (see data_build.load_base_statements).
DIFFICULTIES = ("easy", "hard")

# Cell grid (kept deliberately smaller than the naive full cross so the whole run fits
# the hour budget — see PLAN §10 and the compute note in results/LOG.md):
#   honest / dissimulation : topic x style        (style-shift control lives here)
#   simulation subtypes    : topic x plain only   (except preference: one topicless cell)
N_PER_CELL = 100                 # cap per cell; cells use min(N, statements available)
TRUTHFIT_PAIRS_PER_TOPIC = 10    # statement pairs per topic reserved to fit d_truth
                                 # (held out from all probe training/evaluation)
