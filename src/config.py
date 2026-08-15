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
#
# Qwen3.5 is the primary because it is the current recommended default dense family.
# Two things about it drive code elsewhere, both verified from its model card/config:
#   - it REASONS BY DEFAULT, emitting <think>...</think> before the answer, and does not
#     support the Qwen3 /think /nothink soft switch. The only lever is
#     apply_chat_template(enable_thinking=False) — see activations.capture_residual.
#     With thinking on, MAX_NEW_TOKENS=32 truncates mid-trace and no verdict is ever
#     emitted, so every label would silently fall back to the assumed value.
#   - it is multimodal (Qwen3_5ForConditionalGeneration) with the text stack's depth
#     under config.text_config — see activations.resolve_layers.
MODELS = {
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",                               # primary (Apache-2.0, ungated)
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",                               # fast smoke-test sibling
    # Cross-family replication. Both UNGATED, which matters: gemma-3 and llama-3.1 both
    # 401 without an accepted licence + token, and a replication you cannot run is not a
    # replication. Different labs, different pretraining, different tokenizers — which is
    # the point, since same-family replication mostly re-tests the same training data.
    "phi-4": "microsoft/phi-4",                                    # MIT, 14.7B, Phi3ForCausalLM
    "olmo-3-7b-instruct": "allenai/Olmo-3-7B-Instruct",            # Apache-2.0, 7.3B, fully open
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",             # earlier runs / comparison
    # gated — need `huggingface-cli login` AND licence acceptance on the model page:
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
PRIMARY_MODEL = "qwen3.5-9b"
REPLICATION_MODELS = ("phi-4", "olmo-3-7b-instruct")
GATED_MODELS = ("gemma-3-12b-it", "llama-3.1-8b-instruct")

# Models that emit a reasoning trace unless explicitly told not to. capture_residual
# passes enable_thinking=False for these; preflight.py gates on no <think> surviving.
THINKING_MODELS = ("qwen3.5-9b", "qwen3.5-4b")

# residual-stream layers to sweep, as fractions of depth (resolved per-model in activations.py)
LAYER_FRACTIONS = (0.6, 0.7, 0.8)

# bf16, not 4-bit. A 9B fits a 48GB card in bf16 with room to spare, and quantization
# perturbs exactly the thing this project measures — residual-stream geometry. There is
# no compute reason to carry it on rented hardware. DTYPE participates in cache
# invalidation (activations._cache_identity): flipping precision changes activations
# without changing any prompt, so it must not silently reuse an existing cache.
LOAD_IN_4BIT = False
DTYPE = "bfloat16"
MAX_NEW_TOKENS = 32
CHECKPOINT_EVERY_N_ROWS = 100   # activation-cache flush interval — protects a rented-GPU
                                # run against spot eviction / dropped SSH / any crash.
                                # Each flush rewrites all layer .npz files, so at 20 this
                                # cost ~110 full rewrites over a 2.2k-row run.

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
