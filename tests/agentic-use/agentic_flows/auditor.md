<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Auditor Service Agentic Flows

The Auditor service provides model safety testing, bias detection, and adversarial robustness evaluation using tools like Garak for red-teaming.

**PIC**: Paul Parkanzky
**Priority**: Medium

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 22 | Auditor Target CRUD Operations | 2 | No | `auditor-target-crud-cli` | Create, list, get, update, and delete an Auditor target. Targets define the model endpoint to audit (e.g., build.nvidia.com, local NIM, NeMo NIM Proxy). | POR |
| 23 | Auditor Config CRUD Operations | 2 | No | `auditor-config-crud-cli` | Create, list, get, update, and delete an Auditor configuration. Configs define which probes to run during an audit. | POR |
| 24 | Run Default Audit Job | 3 | No | `auditor-default-job-cli` | Create a target, use the built-in "default" audit config, run an audit job. Monitor job status and retrieve basic results/logs. | POR |
| 25 | Custom Audit with Selected Probes | 4 | No | `auditor-custom-probes-cli` | Create a custom audit config selecting specific probes (e.g., 3 targeted probes instead of default). Run audit job, retrieve detailed results and hit logs. | POR |

---

## Flow Details

### 22. Auditor Target CRUD Operations

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

### 23. Auditor Config CRUD Operations

**Complexity**: 2 (Simple)

**Operations**:
- Create audit configuration with probe selection
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

### 24. Run Default Audit Job

**Complexity**: 3 (Moderate)

**Operations**:
1. Create target for model to audit
2. Use built-in "default" audit config
3. Launch audit job
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
- Audit job runs to completion
- Results contain probe findings
- Logs available for review
- No false positives on safe model

---

### 25. Custom Audit with Selected Probes

**Complexity**: 4 (Complex)

**Operations**:
1. Create custom audit config
2. Select specific probes (e.g., 3 targeted probes)
3. Create target
4. Run audit job with custom config
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
- Audit scope matches configuration

---

## Documentation References

- Audit overview: docs/auditor/index.md
- SDK resources: docs/auditor/sdk-resources.md
- Targets: docs/auditor/targets/index.md
- Inference Gateway routing: docs/auditor/targets/inference-gateway.md
- Target schema: docs/auditor/targets/schema.md
- Configs: docs/auditor/configs/index.md
- Selecting probes: docs/auditor/configs/probes.md
- Config schema: docs/auditor/configs/schema.md
- Run an audit locally: docs/auditor/tutorials/run-audit-locally.md
