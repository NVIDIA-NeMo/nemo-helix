#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo '=== Checking Docker socket ==='
if [ -S /var/run/docker.sock ]; then
    echo 'Docker socket found at /var/run/docker.sock'
    export DOCKER_HOST=unix:///var/run/docker.sock
else
    echo 'WARNING: Docker socket not found - job execution will not work'
fi

cd /app && /app/.venv/bin/nemo services run > /tmp/nhx-api.log 2>&1 &
API_PID=$!
echo "Started API server with PID: $API_PID"

CONSECUTIVE_SUCCESS=0
for i in {1..60}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        CONSECUTIVE_SUCCESS=$((CONSECUTIVE_SUCCESS+1))
        echo "Health check passed ($CONSECUTIVE_SUCCESS/3)"
        if [ "$CONSECUTIVE_SUCCESS" -ge 3 ]; then
            echo 'NeMo Helix API ready and stable'
            break
        fi
    else
        CONSECUTIVE_SUCCESS=0
    fi
    sleep 1
done

if [ "$CONSECUTIVE_SUCCESS" -lt 3 ]; then
    echo 'API failed to become stable!'
    cat /tmp/nhx-api.log
    kill $API_PID 2>/dev/null
    exit 1
fi

echo '=== Pre-configuring CLI auth ==='
/app/.venv/bin/nhx auth login --base-url http://localhost:8080 --unsigned-token --email agent@harbor.local --no-exp

bash /app/setup-env.sh

exec runuser -u harbor -- "$@"
