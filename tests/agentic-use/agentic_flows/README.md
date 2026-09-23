<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix Agentic Flows Documentation

This directory contains curated documentation of 39 E2E agentic flows for the NeMo Helix, derived from the NeMo Helix v2 Phase 4 Plan of Record (POR).

## Purpose

These flow documents serve as:
1. **Reference** for users/agents building agent-based automation with NeMo Helix
2. **Test specifications** for Harbor framework agentic tests
3. **MCP development roadmap** showing which tools agents need
4. **Training data** for agent-assisted NeMo Helix usage

## Organization

### Single-Service Flows (33 flows)
Each service has a dedicated markdown file documenting flows that primarily use that service:

| File | Service | Flows | Priority |
|------|---------|-------|----------|
| [auth.md](auth.md) | Auth/Workspaces | 2 | High |
| [secrets.md](secrets.md) | Secrets | 1 | High |
| [files.md](files.md) | Files/Filesets | 2 | High |
| [entities.md](entities.md) | Entities | 1 | High |
| [inference.md](inference.md) | IGW & Models | 5 | High |
| [evaluator.md](evaluator.md) | Evaluator | 5 | Medium |
| [customizer.md](customizer.md) | Customizer | 6 | Medium |
| [guardrails.md](guardrails.md) | Guardrails | 2 | Medium |
| [auditor.md](auditor.md) | Auditor | 4 | Medium |
| [data-designer.md](data-designer.md) | Data Designer | 3 | Medium |
| [safe-synthesizer.md](safe-synthesizer.md) | Safe Synthesizer | 1 | Low |
| [intake.md](intake.md) | Intake | 4 | Low |

### Cross-Service Flows (6 flows)
The [cross-service.md](cross-service.md) file documents flows that orchestrate multiple services together, representing complete ML pipelines and workflows.

## Flow Documentation Format

Each flow includes:
- **Name**: Unique identifier for the flow
- **Services**: List of NeMo Helix services involved
- **Complexity**: 1-5 stars (from POR)
- **Description**: Brief summary of what the flow accomplishes
- **Source**: Reference to POR, test file, or documentation

## Complexity Guide

| Stars | Level | Description |
|-------|-------|-------------|
| 1 | Easy | Basic CRUD operations, single API calls |
| 2 | Simple | Multi-step operations within one service |
| 3 | Moderate | Job orchestration, async operations, multiple resources |
| 4 | Complex | Multi-service coordination, data pipelines |
| 5 | Advanced | Full platform integration, multi-node, long-running workflows |

## Quick Reference by Complexity

### Easy (Complexity 1)
- Workspace Management (auth)
- Secret CRUD Operations (secrets)
- Fileset CRUD Operations (files)
- Basic Entity Operations (entities)
- Model Provider Registration (inference)
- Intake App CRUD Operations (intake)

### Simple (Complexity 2)
- Upload Dataset to Files Service (files)
- Deploy NIM and Run Inference (inference)
- Inference via IGW with Provider (inference)
- Chat Completions via IGW (inference)
- MockLLM Provider in IGW (inference)
- Basic Content Safety Check (guardrails)
- Auditor Target CRUD Operations (auditor)
- Auditor Config CRUD Operations (auditor)
- Data Designer - Configure Models (data-designer)
- Intake Entry Submission (intake)

### Moderate (Complexity 3)
- Simple Custom Evaluation Job (evaluator)
- LLM-as-a-Judge Evaluation (evaluator)
- Zero-Config LLM-as-a-Judge (evaluator)
- Academic Benchmark Evaluation (evaluator)
- Tool Calling Evaluation (evaluator)
- Run Default Audit Job (auditor)
- Preview Synthetic Data (data-designer)
- List and Filter Entries (intake)

### Complex (Complexity 4)
- Basic LoRA Customization Job (customizer)
- Full SFT (customizer)
- DPO (customizer)
- Knowledge Distillation (customizer)
- Chat-Format Dataset Customization (customizer)
- Guardrails with Custom Configuration (guardrails)
- Custom Audit with Selected Probes (auditor)
- Full Batch Generation Job (data-designer)
- Export Task to Files Service (intake)
- Customized Model Inference via IGW (cross-service)
- Customization + Evaluation Loop (cross-service)
- Tool Calling Fine-Tuning + Evaluation (cross-service)
- Guardrails Evaluation Flow (cross-service)

### Advanced (Complexity 5)
- Authorization Flow (auth)
- Multi-Node Customization Flow (customizer)
- Safe Synthesizer Flow (safe-synthesizer)
- Full E2E Flow (cross-service)
- Data Flywheel (cross-service)

## Data Sources

- **Primary**: NeMo Helix v2 POR (Plan of Record) - 39 official flows
- **Test References**: `tests/e2e/`
- **Architecture**: `architecture/docs/`

## Using These Flows

### For Harbor Agentic Tests
Each documented flow can be implemented as a Harbor test:
1. Create test directory from `../example-test-template/`
2. Write `instruction.md` based on flow description
3. Write `tests/test_outputs.py` based on success criteria

### For MCP Development
Flows guide which MCP tools need to be built. Currently available:
- Workspaces: `list_workspaces`, `create_workspace`, `delete_workspace`

## Contributing

When adding new flows:
1. Place single-service flows in the appropriate service file
2. Place multi-service flows in `cross-service.md`
3. Include: name, services, complexity, description, source
4. Update this README's flow counts and quick reference
