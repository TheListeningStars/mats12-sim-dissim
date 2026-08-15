#!/usr/bin/env bash
# One-shot bootstrap for a fresh RunPod box, then the staged run.
#
#   bash scripts/pod_setup.sh
#
# Everything lands under /workspace (the persistent volume), so stopping the pod
# between stages keeps the repo, the HF model cache, and the activation cache. Only
# the GPU stops billing. That is the point of the staged workflow: preflight, look,
# short slice, look, then commit to the full run.
set -euo pipefail

REPO=https://github.com/TheListeningStars/mats12-sim-dissim.git
BRANCH=decircularize-and-staged-run
ROOT=/workspace/mats12-sim-dissim

# HF cache on the persistent volume, not the container disk — a 9B download is ~19GB
# and re-downloading it after every pod stop would dominate the run's cost.
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME"
grep -q 'HF_HOME' ~/.bashrc 2>/dev/null || echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

if [ -d "$ROOT/.git" ]; then
  echo "== repo present, updating =="
  git -C "$ROOT" fetch --all -q && git -C "$ROOT" checkout -q "$BRANCH" && git -C "$ROOT" pull -q
else
  echo "== cloning =="
  git clone -q --branch "$BRANCH" "$REPO" "$ROOT"
fi
cd "$ROOT"

echo "== deps =="
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Pin what actually got installed. requirements.txt carries ranges, and the dry runs and
# the GPU run will otherwise be on different stacks with no record of the difference.
pip freeze > "results/pip_freeze_$(hostname).txt"

echo
echo "== environment =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
git -C "$ROOT" log --oneline -1

cat <<'NEXT'

== setup done. Run these IN ORDER, looking at the output of each ==

  cd /workspace/mats12-sim-dissim && export HF_HOME=/workspace/hf

  # 1. ~5 min. Downloads the model, then gates on the things that silently ruin a run:
  #    <think> blocks, parse rate, layer resolution, activation sanity. Prints raw
  #    generations -- read them. Exits non-zero if anything is off.
  python scripts/preflight.py --model qwen3.5-9b

  # 2. ~20-30 min. Caches a 400-row slice spread over every cell, then stops and
  #    reports. Nothing here is recomputed later.
  python scripts/run_all.py --model qwen3.5-9b --limit 400

  # 3. Only if step 2 looks right. Finishes the remaining ~1,540 rows and runs the
  #    full analysis. Resumes from the slice above.
  python scripts/run_all.py --model qwen3.5-9b

  # 4. Independent re-derivation of every headline number.
  python scripts/verify.py --model qwen3.5-9b

  # 5. Cross-family replication, headline cells only.
  python scripts/run_all.py --model gemma-3-12b-it --headline-only

  # 6. Push results back (results/** is tracked; cache/ and data/ are not).
  git add -A results && git commit -m "Qwen3.5-9B full run" && git push

Run long stages under tmux so a dropped connection doesn't kill them:
  tmux new -s run   # then Ctrl-b d to detach, tmux attach -t run to return
NEXT
