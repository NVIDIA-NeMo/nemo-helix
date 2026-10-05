<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guardrails Service Agentic Flows

The Guardrails service provides runtime safety controls for model inputs and outputs, including content safety checks, PII detection, and custom filtering rules.

**PIC**: Jash Gulabrai
**Priority**: Medium

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 20 | Basic Content Safety Check | 2 | No | `guardrails-content-safety-cli` | Make an inference call through guardrails that triggers the default content safety configuration. Verify harmful input is blocked with the canned response. | POR |
| 21 | Guardrails with Custom Configuration | 4 | No | `guardrails-custom-config-cli` | Create a custom guardrails configuration, store it, and make inference calls that exercise input/output rails with the custom config. | POR |

---

## Flow Details

### 20. Basic Content Safety Check

**Complexity**: 2 (Simple)

**Operations**:
1. Configure inference to use guardrails
2. Send request with potentially harmful content
3. Verify guardrails block the request
4. Confirm canned safety response returned

**Safety Checks Available**:
- Toxicity detection
- Harmful content detection
- Prompt injection detection

**Prerequisites**:
- NeMo Helix with guardrails enabled
- Model accessible via IGW
- Default guardrails configuration

**Success Criteria**:
- Safe content passes through
- Harmful content blocked
- Canned response returned for blocked content
- Blocking logged for audit

---

### 21. Guardrails with Custom Configuration

**Complexity**: 4 (Complex)

**Operations**:
1. Define custom guardrails configuration
2. Configure input rails (pre-processing)
3. Configure output rails (post-processing)
4. Store configuration
5. Make inference calls exercising custom rules
6. Verify custom rules applied correctly

**Custom Configuration Options**:
- Topic filtering
- PII detection and masking
- Custom blocked phrases
- Hallucination detection
- Custom response templates

**Prerequisites**:
- NeMo Helix with guardrails enabled
- Understanding of guardrails configuration schema

**Success Criteria**:
- Custom config stored successfully
- Input rails applied before model
- Output rails applied after model
- Custom rules enforce as expected

---

## Documentation References

- Running inference with guardrails: docs/guardrails/running-inference.md
- Add safety checks tutorial: docs/get-started/tutorials/add-safety-checks.md
- Manage guardrail configs: docs/guardrails/manage-guardrail-configs/
- Guardrails tutorials: docs/guardrails/tutorials/
