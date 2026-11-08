#!/usr/bin/env bash
# Launch one agent run, isolated, with a hard time limit.
#
#   agents/launch.sh <agent> <run-name> [minutes]
#   agents/launch.sh codex codex-argmedian_4_3-1 30
#
# Isolation (checked 2026-11-08):
# - ~/agent-runs/.sandbox.sb (macOS sandbox-exec) denies reading x-brain, ~/Developer,
#   ~/IdeaProjects, Claude's project memory, global CLAUDE.md, skills, plugins and agents.
#   It is inherited by every process the agent starts (python included).
# - Grok, Codex and Gemini get a HOME holding only their own login (~/agent-runs/.home-<agent>).
#   Claude keeps the real HOME (its login lives in the keychain) and relies on the sandbox,
#   --setting-sources project and an empty MCP config; asked, it sees no CLAUDE.md, no user
#   skills and no MCP tools.
# - Web search is off in every agent; TASK.md forbids any internet use.
set -euo pipefail

agent=$1
run=$2
minutes=${3:-30}
root=~/agent-runs
dir=$root/$run
out=$root/$run.jsonl
sb=$root/.sandbox.sb
prompt="Read TASK.md and do the task."
limit=$((minutes * 60))

[ -d "$dir" ] || { echo "no run directory $dir" >&2; exit 1; }
[ -f "$sb" ] || { echo "no sandbox profile $sb" >&2; exit 1; }
cd "$dir"
# Same Python for every agent: ~/agent-runs/.venv (3.13, numpy, scipy). Round 1 had no scipy and
# the first python3 on PATH (Homebrew) had no numpy either.
export PATH="$root/.venv/bin:$PATH"
"$root/.venv/bin/python" -c "import numpy, scipy" || { echo "run venv lacks numpy/scipy" >&2; exit 1; }
date -u +"start %Y-%m-%dT%H:%M:%SZ" > "$root/$run.times"

case $agent in
  grok)
    HOME=$root/.home-grok ~/.grok/bin/grok memory clear >/dev/null 2>&1 || true
    HOME=$root/.home-grok timeout -k 10 "$limit" sandbox-exec -f "$sb" ~/.grok/bin/grok -p "$prompt" \
      --always-approve --disable-web-search --output-format streaming-messages-json \
      > "$out" 2> "$root/$run.stderr" || true
    ;;
  claude)
    timeout -k 10 "$limit" sandbox-exec -f "$sb" claude -p "$prompt" \
      --output-format stream-json --verbose --setting-sources project \
      --strict-mcp-config --mcp-config '{"mcpServers":{}}' \
      --disallowedTools WebSearch WebFetch --dangerously-skip-permissions \
      --no-session-persistence < /dev/null > "$out" 2> "$root/$run.stderr" || true
    ;;
  codex)
    HOME=$root/.home-codex CODEX_HOME=$root/.home-codex/.codex timeout -k 10 "$limit" \
      sandbox-exec -f "$sb" codex exec --dangerously-bypass-approvals-and-sandbox --json \
      -o "$root/$run.last.md" "$prompt" < /dev/null > "$out" 2> "$root/$run.stderr" || true
    ;;
  gemini)
    HOME=$root/.home-gemini timeout -k 10 "$limit" sandbox-exec -f "$sb" gemini -p "$prompt" \
      --approval-mode yolo --output-format stream-json \
      < /dev/null > "$out" 2> "$root/$run.stderr" || true
    ;;
  antigravity)
    # Google retired Gemini CLI logins for individuals (2026-11); Antigravity CLI (agy) replaces it.
    # Its login is in the keychain and a fresh HOME triggers a new browser login, so it keeps the
    # real HOME; the sandbox hides ~/.gemini instructions, skills and the desktop app's memory.
    # agy keeps conversation memory (brain/, implicit/, summaries); archive it so a repetition
    # cannot recall an earlier one.
    arch=$root/.agy-archive/$(date +%Y%m%d-%H%M%S)
    mkdir -p "$arch"
    for f in brain implicit conversations conversation_summaries.db jetbox_summaries_proto.pb; do
      [ -e ~/.gemini/antigravity-cli/$f ] && mv ~/.gemini/antigravity-cli/$f "$arch/"
    done
    timeout -k 10 "$limit" sandbox-exec -f "$sb" ~/.local/bin/agy -p "$prompt" \
      --dangerously-skip-permissions --output-format stream-json \
      < /dev/null > "$out" 2> "$root/$run.stderr" || true
    ;;
  *) echo "unknown agent $agent" >&2; exit 1 ;;
esac

date -u +"end %Y-%m-%dT%H:%M:%SZ" >> "$root/$run.times"
git -C "$dir" add -A >/dev/null 2>&1 || true
git -C "$dir" diff --cached > "$root/$run.diff" || true
echo "done: $out"
