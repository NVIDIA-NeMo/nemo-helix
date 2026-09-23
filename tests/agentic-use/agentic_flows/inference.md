<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference Gateway (IGW) & Models Agentic Flows

The Inference Gateway (IGW) provides unified access to model inference, supporting both deployed NIMs and external providers like build.nvidia.com. The Models service manages model registration and metadata.

**PIC**: Benjamin McCown
**Priority**: High

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 7 | Model Provider Registration | 1 | No | `inference-provider-reg-cli` | Register build.nvidia.com as an inference provider in IGW, associating an API key secret. Verify provider appears in list. | POR |
| 8 | Deploy NIM and Run Inference | 2 | No | No | Create a deployment config, deploy a NIM (e.g., Llama 3.1 8B), wait for deployment to become ready, then run inference through IGW. | POR |
| 9 | Inference via IGW with Provider | 2 | No | `inference-igw-provider-cli`, `inference-igw-provider-cli-easy` | Using a registered provider (e.g., build.nvidia.com), make a /v1/chat/completions call through IGW. Verify response returns correctly. | POR |
| 10 | Chat Completions via IGW | 2 | No | `inference-chat-completions-cli` | Execute a /v1/chat/completions request through IGW with a registered model provider. Test both streaming and non-streaming modes. | POR |
| 11 | MockLLM Provider in IGW | 2 | No | `inference-mockllm-cli` | Configure and use a MockLLM provider through IGW for testing purposes. Verify deterministic responses are returned. | POR; tests/e2e/evaluator/test_llm_judge_jobs.py |

---

## Flow Details

### 7. Model Provider Registration

**Complexity**: 1 (Easy)

**Operations**:
1. Create a secret with API key for provider
2. Register provider in IGW with secret reference
3. List providers to verify registration

**Prerequisites**:
- NeMo Helix running
- Workspace exists
- Valid API key for provider (e.g., build.nvidia.com)

**Success Criteria**:
- Provider registered successfully
- Provider appears in list
- Provider can be used for inference

---

### 8. Deploy NIM and Run Inference

**Complexity**: 2 (Simple)

**Operations**:
1. Create deployment configuration
2. Deploy NIM (e.g., Llama 3.1 8B)
3. Wait for deployment to become ready
4. Run inference through IGW

**Deployment Modalities**:
- Deploy LLM-specific NIM (pulls own weights or baked in)
- Deploy multi-LLM NIM with model weights from HuggingFace
- Deploy full weight SFT model from Files Service (produced by Customizer)
- Deploy LLM-specific NIM with LoRAs (adapters in Files Service)

**Prerequisites**:
- NeMo Helix with GPU resources
- Workspace exists
- Model weights accessible

**Success Criteria**:
- Deployment config created
- NIM deployment starts
- Deployment reaches ready state
- Inference returns valid response

---

### 9. Inference via IGW with Provider

**Complexity**: 2 (Simple)
**CLI Eval**: `inference-igw-provider-cli` (minimal instructions), `inference-igw-provider-cli-easy` (guided)

**Operations**:
1. Create a secret with the inference API key
2. Register a provider named `nvidia-inference` pointing to `https://inference-api.nvidia.com/v1`
3. Make a /v1/chat/completions call through IGW using the provider
4. Verify response contains generated text and token usage

**Inference Routes** (agent may use any):
- `nemo inference gateway provider post v1/chat/completions <provider> --body '{...}'`
- `nemo chat <model> "<prompt>" --provider <provider>`

**Prerequisites**:
- NeMo Helix running (quickstart)
- Valid API key for NVIDIA inference API (passed as `ANTHROPIC_API_KEY` env var)

**Verifier Tests** (6 tests, weighted):
- Provider setup (3/6 weight): provider exists, host URL matches accepted URLs, API key secret configured
- Live inference (2/6 weight): chat completions returns valid choices, response includes token usage
- Agent trace (1/6 weight): agent's bash output contains evidence of an inference response

**Notes**:
- The eval reuses the `ANTHROPIC_API_KEY` (originally for the agent LLM) as the inference provider API key
- Model used: `aws/anthropic/bedrock-claude-sonnet-4-5-v1` at `inference-api.nvidia.com`
- The CLI-easy variant provides explicit command syntax; the CLI variant requires the agent to discover commands via `--help`
- Typical runtime: ~4 min (easy), ~7 min (CLI, due to `--help` exploration)

**Success Criteria**:
- Completions request succeeds
- Response contains valid completion text
- Token usage reported correctly

---

### 10. Chat Completions via IGW

**Complexity**: 2 (Simple)

**Operations**:
1. Use registered provider or deployed NIM
2. Make /v1/chat/completions call
3. Test streaming mode
4. Test non-streaming mode

**Prerequisites**:
- Provider registered or NIM deployed
- Chat-capable model available

**Success Criteria**:
- Non-streaming request returns complete response
- Streaming request returns chunked response
- Messages format handled correctly

---

### 11. MockLLM Provider in IGW

**Complexity**: 2 (Simple)

**Operations**:
1. Configure MockLLM provider
2. Make inference call
3. Verify deterministic response

**Use Cases**:
- Testing evaluation pipelines
- CI/CD testing without real LLM costs
- Reproducible test scenarios

**Prerequisites**:
- NeMo Helix running
- MockLLM configuration

**Success Criteria**:
- MockLLM provider configured
- Inference returns deterministic response
- Response matches expected mock output

---

## Documentation References

- Note: IGW and Model Providers are NEW in v2
- Old reference: docs/run-inference/nim-proxy/ (v1 NIM proxy docs to deprecate)
- Deploy NIMs: docs/get-started/tutorials/deploy-nims.md
- Deployment management: docs/run-inference/deployment-management/
- Completions: docs/run-inference/nim-proxy/completions.md
- Chat completions: docs/run-inference/nim-proxy/chat-completions.md
