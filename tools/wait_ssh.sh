#!/usr/bin/env bash
# Wait until the instance answers on SSH: bash tools/wait_ssh.sh root@<ip> [tries]
for i in $(seq 1 "${2:-30}"); do
  if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o BatchMode=yes "$1" true 2>/dev/null; then
    echo "up after $i tries"; exit 0
  fi
  sleep 10
done
echo "still down"; exit 1
