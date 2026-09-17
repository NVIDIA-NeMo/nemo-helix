# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for the benchmark harness."""

from __future__ import annotations

WORKSPACE = "benchmark"
GUARDRAIL_CONFIG = "content-safety-local"
VM_NAME = "guardrails-vm"
# Control VirtualModel with no middleware attached. Used by the benchmark
# harness to measure NMP+IGW latency *without* the guardrails middleware so
# the with-vs-without delta isolates middleware overhead.
NO_GUARDRAILS_VM_NAME = "no-guardrails-vm"

# Logical identifiers for the two benchmark variants. Used as subdirectory
# names under `aiperf_results/` and `logs/`, and as the value of the
# harness's `--variant` flag.
VARIANT_WITH_GUARDRAILS = "with-guardrails"
VARIANT_WITHOUT_GUARDRAILS = "without-guardrails"
ALL_VARIANTS = (VARIANT_WITH_GUARDRAILS, VARIANT_WITHOUT_GUARDRAILS)

# ModelProvider that proxies requests to the mock main model
APP_PROVIDER = "benchmark-app-llm"
APP_PROVIDER_URL = "http://localhost:8000"
APP_MODEL_NAME = "meta/llama-3.3-70b-instruct"

# ModelProvider that proxies requests to the mock content-safety model
CS_PROVIDER = "benchmark-content-safety-llm"
CS_PROVIDER_URL = "http://localhost:8001"
CS_MODEL_NAME = "nvidia/llama-3.1-nemoguard-8b-content-safety"

NMP_BASE_URL = "http://localhost:8080"
NMP_HEALTH_PATH = "/health/ready"
# Non-secret Fernet key the scaled-evals plugin requires to resolve its
# settings; without it its controller crashes on startup and the platform
# never reports ready. Benchmarks never touch scaled-evals data, so any
# valid 32-byte urlsafe-base64 key works.
SCALED_EVALS_CREDENTIALS_ENCRYPTION_KEY = "oNRkxH0SFmK9wVLhg-ZTkhiY3_r-YFqWNU-dVFUC7ko="
IGW_CHAT_PATH = f"/apis/inference-gateway/v2/workspaces/{WORKSPACE}/openai/-/v1/chat/completions"

# Local shim that satisfies AIPerf's pre-check and reverse-proxies chat
# completion requests through to NMP's IGW. See
# `nemo_guardrails_plugin.benchmarks.shim` for the implementation.
AIPERF_SHIM_HOST = "127.0.0.1"
AIPERF_SHIM_PORT = 8090
AIPERF_SHIM_BASE_URL = f"http://{AIPERF_SHIM_HOST}:{AIPERF_SHIM_PORT}"

GUARDRAILS_MIDDLEWARE_NAME = "nemo-guardrails"
GUARDRAILS_MIDDLEWARE_CONFIG_TYPE = "guardrail_config"
