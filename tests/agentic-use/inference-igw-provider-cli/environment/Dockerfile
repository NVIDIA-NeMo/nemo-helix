# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nhx-agentic-base:latest

# Disable MCP by removing the MCP configuration file
# This forces Claude Code to use the CLI instead of MCP tools
RUN rm -f /app/.mcp.json

# Override ENTRYPOINT to pre-configure CLI auth for IGW proxy.
# The base image starts the API server and waits for health; we do
# the same here and then run `nemo auth login` so the agent doesn't
# need to discover auth on its own.
ENTRYPOINT ["/bin/bash", "-c", "\
cd /app && /app/.venv/bin/nemo services run > /tmp/nhx-api.log 2>&1 & API_PID=$!; \
echo 'Started API server with PID: '$API_PID; \
CONSECUTIVE_SUCCESS=0; \
for i in {1..60}; do \
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then \
        CONSECUTIVE_SUCCESS=$((CONSECUTIVE_SUCCESS+1)); \
        echo 'Health check passed ('$CONSECUTIVE_SUCCESS'/3)'; \
        if [ $CONSECUTIVE_SUCCESS -ge 3 ]; then \
            echo 'NeMo Helix API ready and stable'; \
            break; \
        fi; \
    else \
        CONSECUTIVE_SUCCESS=0; \
    fi; \
    sleep 1; \
done; \
if [ $CONSECUTIVE_SUCCESS -lt 3 ]; then \
    echo 'API failed to become stable!'; \
    cat /tmp/nhx-api.log; \
    kill $API_PID 2>/dev/null; \
    exit 1; \
fi; \
echo '=== Pre-configuring CLI auth for IGW proxy ==='; \
/app/.venv/bin/nemo auth login --base-url http://localhost:8080 --unsigned-token --email agent@harbor.local --no-exp; \
exec runuser -u harbor -- \"$@\"", "--"]
CMD ["sleep", "infinity"]
