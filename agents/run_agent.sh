#!/usr/bin/env bash
# Run one general-purpose agent on a prepared workspace, with a wall-clock cap.
# Usage: bash agents/run_agent.sh claude|codex|aide <workspace> [hours]
# The nested agent inherits this machine's Claude Code hooks, including the dangerous-command gate.
set -euo pipefail
agent=$1; ws=$(cd "$2" && pwd); hours=${3:-2}
cd "$ws"
case "$agent" in
  claude)
    timeout "${hours}h" claude -p "$(cat TASK.md)" --model opus --permission-mode bypassPermissions \
      --output-format stream-json --verbose < /dev/null > agent_log.jsonl ;;
  codex)
    # --full-auto is the workspace-write sandbox, and network is off inside it. Without </dev/null codex waits on stdin forever.
    timeout "${hours}h" codex exec --skip-git-repo-check -m "${CODEX_MODEL:-gpt-5.5}" --full-auto -C "$ws" "$(cat TASK.md)" < /dev/null > agent_log.txt 2>&1 ;;
  aide)
    # Needs OPENAI_API_KEY or ANTHROPIC_API_KEY, and `pip install aideml` in rsna_knee.
    timeout "${hours}h" conda run -n rsna_knee aide data_dir="$ws" desc_file="$ws/TASK.md" agent.steps=20 > agent_log.txt 2>&1 ;;
  *) echo "unknown agent $agent" >&2; exit 2 ;;
esac
echo "finished. Grade with: conda run -n rsna_knee python agents/grade_B.py $ws/predictions.csv"
