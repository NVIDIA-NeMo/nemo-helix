<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Garak Plugin Service Agentic Flows

The Garak Plugin service provides model safety testing, bias detection, and adversarial robustness evaluation using tools like Garak for red-teaming.

**PIC**: Paul Parkanzky
**Priority**: Medium

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 22 | Garak Plugin Target CRUD Operations | 2 | No | `garak-plugin-target-crud-cli` | Create, list, get, update, and delete an Garak Plugin target. Targets define the model endpoint to scan (e.g., build.nvidia.com, local NIM, NeMo NIM Proxy). | POR |
| 23 | Garak Plugin Config CRUD Operations | 2 | No | `garak-plugin-config-crud-cli` | Create, list, get, update, and delete an Garak Plugin configuration. Configs define which probes to run during a scan. | POR |
| 24 | Run Default Scan Job | 3 | No | `garak-plugin-default-job-cli` | Create a target, use the built-in "default" scan config, run a scan job. Monitor job status and retrieve basic results/logs. | POR |
| 25 | Custom Scan with Selected Probes | 4 | No | `garak-plugin-custom-probes-cli` | Create a custom scan config selecting specific probes (e.g., 3 targeted probes instead of default). Run scan job, retrieve detailed results and hit logs. | POR |

---

## Flow Details

### 22. Garak Plugin Target CRUD Operations

**Complexity**: 2 (Simple)

**Operations**:
- Create target pointing to model endpoint
- List all targets
- Get target by ID
- Update target configuration
- Delete target

**Target Types**:
- build.nvidia.com endpoints
- Local NIM deployments
- NeMo NIM Proxy
- Custom model endpoints

**Prerequisites**:
- NeMo Helix running
- Workspace exists
- Model endpoint accessible

**Success Criteria**:
- Target created with correct endpoint
- Target appears in list
- Target can be updated
- Target can be deleted

---

### 23. Garak Plugin Config CRUD Operations

**Complexity**: 2 (Simple)

**Operations**:
- Create scan configuration with probe selection
- List all configurations
- Get configuration by ID
- Update probe selection
- Delete configuration

**Configuration Options**:
- Probe selection (specific probes or categories)
- Run parameters
- Output format

**Prerequisites**:
- NeMo Helix running
- Workspace exists

**Success Criteria**:
- Config created with selected probes
- Config appears in list
- Config can be updated
- Config can be deleted

---

### 24. Run Default Scan Job

**Complexity**: 3 (Moderate)

**Operations**:
1. Create target for model to scan
2. Use built-in "default" scan config
3. Launch scan job
4. Monitor job status
5. Retrieve results and logs

**Default Config Includes**:
- Standard safety probes
- Common jailbreak attempts
- Basic bias detection

**Prerequisites**:
- Target configured
- Model accessible

**Success Criteria**:
- Scan job runs to completion
- Results contain probe findings
- Logs available for review
- No false positives on safe model

---

### 25. Custom Scan with Selected Probes

**Complexity**: 4 (Complex)

**Operations**:
1. Create custom scan config
2. Select specific probes (e.g., 3 targeted probes)
3. Create target
4. Run scan job with custom config
5. Retrieve detailed results
6. Review hit logs

**Probe Categories**:
- Jailbreak attempts
- Bias detection
- Toxicity generation
- Information leakage
- Adversarial inputs

**Prerequisites**:
- Understanding of available probes
- Target model configured

**Success Criteria**:
- Only selected probes run
- Results contain detailed findings
- Hit logs show specific vulnerabilities
- Scan scope matches configuration

---

## Documentation References

- Scan overview: docs/garak-plugin/index.md
- SDK resources: docs/garak-plugin/sdk-resources.md
- Targets: docs/garak-plugin/targets/index.md
- Inference Gateway routing: docs/garak-plugin/targets/inference-gateway.md
- Target schema: docs/garak-plugin/targets/schema.md
- Configs: docs/garak-plugin/configs/index.md
- Selecting probes: docs/garak-plugin/configs/probes.md
- Config schema: docs/garak-plugin/configs/schema.md
- Run a scan locally: docs/garak-plugin/tutorials/run-scan-locally.md
