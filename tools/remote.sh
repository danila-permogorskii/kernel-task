#!/usr/bin/env bash
# Drive the GPU instance from the laptop (WSL) over SSH. The instance never needs GitHub:
# the working tree goes up with rsync, the evidence comes back with rsync, and you commit
# and push from the laptop.
#
#   export H100=root@<instance-ip>        # the login Verda shows for the instance
#   bash tools/remote.sh check            # can we log in? which GPU?
#   bash tools/remote.sh sync             # laptop working tree -> instance ~/kernel-task
#   bash tools/remote.sh run <command>    # run inside ~/kernel-task with the venv active
#   bash tools/remote.sh pull             # instance results/traces/profiles -> laptop
set -euo pipefail
: "${H100:?set H100=user@host first}"
cd "$(dirname "$0")/.."
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$H100")
EVIDENCE=(results/h100 traces/h100 profiles/h100)

case "${1:-}" in
  check)
    "${SSH[@]}" 'hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; which rsync python3 || true'
    ;;
  sync)
    rsync -az --delete \
      --exclude .git --exclude .venv --exclude build --exclude .cuda_home \
      --exclude results --exclude traces --exclude profiles \
      --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' \
      ./ "$H100:kernel-task/"
    echo "synced to $H100:kernel-task"
    ;;
  run)
    shift
    "${SSH[@]}" "cd kernel-task && { [ -f .venv/bin/activate ] && source .venv/bin/activate || true; } && $*"
    ;;
  pull)
    for d in "${EVIDENCE[@]}"; do
      mkdir -p "$d"
      rsync -az "$H100:kernel-task/$d/" "$d/" 2>/dev/null || echo "(no $d on the instance yet)"
    done
    echo "pulled into ${EVIDENCE[*]}"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
