#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start (or restart) the experiment platform in the dev pod on :8090.
#   start-lag-platform.sh [tree]   tree defaults to /work/nmp-lag
# Leaves the long-running platform on :8080 alone: only the PID in /work/lag-platform.pid is killed.
# The entities controller is left out by default: in one process it shares the API's async engine
# from another event loop and corrupts asyncpg connections. Review1 runs controllers in their own pod.
set -eu
TREE="${1:-/work/nmp-lag}"
export HOME=/work
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export NHX_DATA_DIR=/work/.nemo-data-lag
export POD_NAMESPACE=tbray-dev
export DATABASE_URL="${DATABASE_URL:-postgresql://nhx:nhx-lag@pg-lag:5433/nhx}"
mkdir -p /work/logs "$NHX_DATA_DIR"

# Match on the :8090 command line, not the PID file: the recorded PID can name a wrapper that is not
# the server process, and a missed kill leaves the old build serving while the new one fails to bind.
# The bracket in the pattern keeps pgrep from matching this script's own shell.
for _ in $(seq 1 30); do
  pids="$(pgrep -f 'nemo services ru[n].*--port 8090' || true)"
  [ -z "$pids" ] && break
  kill $pids 2>/dev/null || true
  sleep 1
done

cd "$TREE"
nohup "$TREE/.venv/bin/nemo" services run \
  --host 0.0.0.0 \
  --port 8090 \
  --config /work/nmp-lag-config.yaml \
  --services "${LAG_SERVICES:-entities,auth,jobs,secrets,files,hello-world}" \
  --controllers "${LAG_CONTROLLERS:-jobs}" \
  > /work/logs/lag-platform.log 2>&1 &
echo $! > /work/lag-platform.pid
echo "lag platform pid $(cat /work/lag-platform.pid) tree $TREE db $DATABASE_URL"
