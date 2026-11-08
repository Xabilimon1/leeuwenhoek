#!/usr/bin/env bash
# Round 2 of the agent study: argmedian h=2 and h=8, 4 agents x 3 repetitions, same rules as round 1
# (task v2, 30 min, 1e6-input cap logged, no internet) plus numpy/scipy guaranteed by ~/agent-runs/.venv.
#
#   agents/round2.sh prepare   # create the 24 run directories
#   agents/round2.sh launch    # 6 batches of 4 agents (one of each), ~3 h
set -euo pipefail
cd "$(dirname "$0")/.."

agents=(claude codex antigravity grok)
models=(argmedian_2_3 argmedian_8_3)

prepare() {
  for m in "${models[@]}"; do
    for a in "${agents[@]}"; do
      for n in 1 2 3; do
        .venv/bin/python -m agents.make_run --agent "$a" --model "$m" --n "$n" --root ~/agent-runs \
          --budget "30 minutes of wall-clock time, hard stop. Write your deliverables early and update them; whatever is in the directory at the stop is scored." \
          --model-access "You may evaluate the model through load_model.py, up to 1,000,000 inputs in total; every call is logged in model_calls.jsonl. The final accuracy estimate must not come from sampling the model." \
          --internet-rule "No internet access at all: do not search for or fetch anything." \
          --packages-rule "numpy and scipy are available; do not install other packages."
      done
    done
  done
}

launch() {
  # One run of each agent per batch: two agy runs must never overlap (shared memory dir).
  for m in "${models[@]}"; do
    for n in 1 2 3; do
      echo "batch $m rep $n $(date +%H:%M:%S)"
      for a in "${agents[@]}"; do
        agents/launch.sh "$a" "$a-$m-$n" 30 &
      done
      wait
    done
  done
  echo "done $(date +%H:%M:%S)"
}

"$@"
