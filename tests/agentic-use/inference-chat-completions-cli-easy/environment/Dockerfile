# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nhx-agentic-base:latest

# Enable mock provider mode for the Inference Gateway.
# Providers whose name starts with "igw-mock-" will return mock responses
# instead of proxying to real backends.
ENV NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX=igw-mock-

# Copy the mock provider setup script
COPY --chown=harbor:harbor setup-mock.py /app/setup-mock.py

# Override entrypoint to include mock provider setup after the API is healthy
# but before the agent starts. This:
# 1. Starts the NeMo Helix API server in the background (with mock provider mode enabled)
# 2. Waits for the API to be healthy (3 consecutive health checks)
# 3. Runs the setup script to create mock provider with chat completion response
# 4. Starts the agent (Harbor's CMD)
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
echo '=== Setting up mock chat provider ==='; \
/app/.venv/bin/python /app/setup-mock.py; \
SETUP_EXIT=$?; \
if [ $SETUP_EXIT -ne 0 ]; then \
    echo 'Mock setup failed!'; \
    exit 1; \
fi; \
echo '=== Mock setup complete, starting agent ==='; \
exec runuser -u harbor -- \"$@\"", "--"]
CMD ["sleep", "infinity"]
