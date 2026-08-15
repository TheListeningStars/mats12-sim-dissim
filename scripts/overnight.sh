#!/usr/bin/env bash
# Unattended: primary re-run + both cross-family replications + all analyses.
#
#   tmux new-session -d -s night 'bash scripts/overnight.sh > /workspace/night.log 2>&1'
#
# Deliberately NOT `set -e`. One model failing (a gated repo, an OOM, an unmappable
# architecture) must not take the other two with it -- an overnight run that dies at
# step 2 of 9 and leaves nothing is worse than one that reports three failures. Every
# stage logs a STAGE/OK/FAIL line so the morning read is a grep.
set -uo pipefail

ROOT=/workspace/mats12-sim-dissim
export HF_HOME=/workspace/hf
cd "$ROOT" || exit 1

PRIMARY=qwen3.5-9b
REPLICATIONS="phi-4 olmo-3-7b-instruct"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }
stage() { say "STAGE $*"; }
run() {  # run <label> <cmd...>
  local label="$1"; shift
  if "$@" >>"/workspace/${label}.log" 2>&1; then say "OK   $label"; return 0
  else say "FAIL $label (see /workspace/${label}.log; last lines below)"
       tail -5 "/workspace/${label}.log" | sed 's/^/     | /'; return 1; fi
}

say "=== overnight run start ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
git log --oneline -1

# ---- primary -----------------------------------------------------------------
# Re-run rather than assume: this pod is a fresh machine, so a clean reproduction of
# the numbers already pulled down is itself a check (same seed => same results), and
# it rebuilds the activation cache the prompt-site analysis needs.
stage "primary $PRIMARY"
run "pre_$PRIMARY"  python scripts/preflight.py --model "$PRIMARY" \
  && run "full_$PRIMARY" python scripts/run_all.py --model "$PRIMARY"

if [ -f "results/$PRIMARY/baselines.json" ]; then
  run "rowlevel_$PRIMARY"        python scripts/rowlevel.py --model "$PRIMARY"
  # the probe reading the residual BEFORE the model commits to a verdict -- it cannot
  # be reading its own answer off the page, which the response-mean probe can
  run "rowlevel_prompt_$PRIMARY" python scripts/rowlevel.py --model "$PRIMARY" --site prompt
  run "verify_$PRIMARY"          python scripts/verify.py  --model "$PRIMARY"
  run "transcripts_$PRIMARY"     python scripts/sample_transcripts.py --model "$PRIMARY" --n 30
fi

# ---- cross-family replications ------------------------------------------------
# --headline-only drops the formal-style cells: the replication exists to test whether
# the target-vs-source variance split holds in another lab's model, and that does not
# need the style-shift arm.
for M in $REPLICATIONS; do
  stage "replication $M"
  if run "pre_$M" python scripts/preflight.py --model "$M"; then
    run "full_$M" python scripts/run_all.py --model "$M" --headline-only
    if [ -f "results/$M/baselines.json" ]; then
      run "rowlevel_$M"    python scripts/rowlevel.py --model "$M"
      run "verify_$M"      python scripts/verify.py   --model "$M"
      run "transcripts_$M" python scripts/sample_transcripts.py --model "$M" --n 20
    fi
  else
    say "SKIP $M — preflight failed, not caching activations for it"
  fi
done

# ---- figures ------------------------------------------------------------------
stage "figures"
run figures python scripts/figures.py

stage "packaging"
tar czf /workspace/results.tgz results/ 2>/dev/null && ls -la /workspace/results.tgz
md5sum /workspace/results.tgz

say "=== overnight run complete ==="
say "summary:"
grep -E '^\[.*\] (OK|FAIL|SKIP) ' /workspace/night.log 2>/dev/null | sed 's/^/  /'
