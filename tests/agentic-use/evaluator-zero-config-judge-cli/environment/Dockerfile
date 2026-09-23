# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nhx-agentic-base:latest

# Disable MCP by removing the MCP configuration file
# This forces Claude Code to use the CLI instead of MCP tools
RUN rm -f /app/.mcp.json

# Copy the infrastructure setup script
COPY --chown=harbor:harbor setup-infra.py /app/setup-infra.py

# Override ENTRYPOINT to start API, pre-configure CLI auth, and run
# infrastructure setup. The agent handles evaluation-specific tasks.
#
# Note: We must run as root for the API server, then the agent runs as
# the harbor user (handled by Harbor's docker-compose exec).
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
echo '=== Pre-configuring CLI auth (as harbor user) ==='; \
su -s /bin/bash harbor -c '/app/.venv/bin/nemo auth login --base-url http://localhost:8080 --unsigned-token --email agent@harbor.local --no-exp'; \
echo '=== Setting up infrastructure ==='; \
su -s /bin/bash harbor -c '/app/.venv/bin/python /app/setup-infra.py'; \
SETUP_EXIT=$?; \
if [ $SETUP_EXIT -ne 0 ]; then \
    echo 'Infrastructure setup failed!'; \
    exit 1; \
fi; \
echo '=== Infrastructure setup complete, starting agent ==='; \
exec runuser -u harbor -- \"$@\"", "--"]
CMD ["sleep", "infinity"]
