#!/usr/bin/env bash
set -euo pipefail

SESSION="robots"
COUNT=5

# Activate your conda env + python
CONDA_BASE="$(conda info --base)"
ACTIVATE="source $CONDA_BASE/etc/profile.d/conda.sh && conda activate llm-flocking"

# Command builder (per job i)
build_cmd() {
  local i="$1"
  local test_name="official_updated_circle_collision_gpt5-m_${i}"
  local seed="$i"
  echo "$ACTIVATE && python main.py -n \"$test_name\" -s \"$seed\" -gpt gpt-5-mini -mc openai -ra medium -form circle -a 10"
}

# Start tmux session
tmux new-session -d -s "$SESSION"

# First pane (job 1)
tmux send-keys -t "$SESSION":0.0 "$(build_cmd 1)" C-m

# Create panes for jobs 2–10
for i in $(seq 2 "$COUNT"); do
  tmux split-window -t "$SESSION":0 -h
  tmux send-keys -t "$SESSION":0.$((i-1)) "$(build_cmd $i)" C-m
  tmux select-layout -t "$SESSION":0 tiled >/dev/null
done

# Final tidy layout
tmux select-layout -t "$SESSION":0 tiled >/dev/null

# Attach to session
tmux attach -t "$SESSION"