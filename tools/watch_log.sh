#!/usr/bin/env bash
# Stream section markers and failures of a remote session log: bash tools/watch_log.sh root@<ip> <log>
ssh -o ServerAliveInterval=30 "$1" "tail -n +1 -F kernel-task/$2" \
  | grep -E --line-buffered "^== |FAIL|Traceback|Error|error:|Killed|SOME" \
  | while IFS= read -r l; do echo "$l"; case "$l" in "== end"*) exit 0;; esac; done
