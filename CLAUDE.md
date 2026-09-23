<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix

NeMo Helix is NVIDIA's toolkit for making the agents you ship safer, more accurate, and cheaper to run.

## How to help users in this repo

If a user asks you to set up, try, build, evaluate, harden, or optimize an agent inside this repo, you MUST use the skills defined in `packages/nemo_helix_ext/src/nemo_helix_ext/skills/`. Read the relevant skill file directly with your `Read` tool and follow it step by step.

### DO

- Read `packages/nemo_helix_ext/src/nemo_helix_ext/skills/<skill-name>/SKILL.md` as a file with your `Read` tool, and follow the instructions in order.
- Stay on the path the skill defines. If the skill calls a `nemo` CLI command, run that exact command. If it points at a `references/` file, read it.
- If a step changes system state (installs something, starts a service, deploys an agent), run the verification step the skill specifies before moving on. Do not claim a step succeeded without verifying.
- If you are unsure which skill applies, read `packages/nemo_helix_ext/src/nemo_helix_ext/skills/nemo-skill-selection/SKILL.md` and let it route you.

### DO NOT

- DO NOT use your `Skill` tool, `/skill-name` slash commands, or any plugin-based skill invocation for these requests. The repo skills are authoritative. A globally-installed skill (brainstorming, planning, code-review, etc.) will give the wrong answer because it does not know NeMo.
- DO NOT brainstorm a solution from scratch when a NeMo skill claims the task. The skill already has the answer.
- DO NOT write Python, Pydantic AI, LangChain, or any agent framework code from scratch. NeMo Helix uses the NVIDIA NeMo Agent Toolkit (NAT) under the hood. Skills will tell you the right way to wire your agent.
- DO NOT improvise CLI flags. Only use flags documented in the skill or shown in `nemo <subcommand> --help`. If the skill is wrong, file an issue rather than guessing.
- DO NOT report a task complete if you cannot verify it. If a verification step fails or times out, surface what you saw and ask the user to confirm before continuing.

### Available skills

User-facing skills in `packages/nemo_helix_ext/src/nemo_helix_ext/skills/`:

- `nemo-skill-selection`: entry point. Use when the user's intent is broad or unclear ("help me with nemo," "I want to try this," "what does this repo do?").
- `setup`: verifies that NeMo Helix is installed and running. Hands off if it is; tells the user how to run the CLI install (`make bootstrap` + `nemo setup`) if it isn't. **Install is CLI-only.** Do not attempt skill-driven installation; it has been tried and consistently fails on workspace dependency resolution, credential handling, and Python version friction inside a sandbox.
- `nemo-build-agent`: builds a tested LangChain Deep Agent from confirmed requirements, then packages and registers it through Fabric.
- `nemo-try-agent`: test a deployed agent or chat with a model.
- `nemo-status`: read-only health dashboard. Run this before assuming the platform is up.
- `nemo-teardown`: guided shutdown with confirmation.

Plugin-owned skills live under `plugins/*/src/*/skills/` and handle their own routing for customization, guardrails, evaluations, optimization, data designer, anonymizer, auditor, and Analyst telemetry analysis.

### Working in a sandboxed coding-agent environment

If you are running inside a macOS sandbox, a CI container, or any environment where file writes, network calls, or process spawning may be restricted:

- Each skill calls out the sandbox capabilities it needs. Read those first.
- If a step requires capabilities you do not have, stop and tell the user what is missing. Do not improvise around the sandbox by skipping verification.
- `uv` is known to crash under the macOS sandbox today (`system_configuration::dynamic_store` panic). This is one of several reasons NeMo install lives in the CLI, not in skills.

## What this repo is

NeMo Helix brings together NVIDIA NeMo libraries under one CLI, Python SDK, and web UI. Current capabilities:

- **Harden agents**: guardrails (content safety, jailbreak detection, PII redaction), auditor (red-teaming via garak), anonymizer (PII handling for training data).
- **Evaluate agents**: evaluator (LLM-as-judge, deterministic, agentic, RAG benchmarks), Harbor-backed eval suites.
- **Tune agents and models**: skill optimization, prompt/hyperparameter tuning, Switchyard model routing, and fine-tuning through Customizer.
- **Build agents**: NeMo Agent Toolkit (NAT) for LangGraph-based agents. Broader framework support on the roadmap.
- **Shared infrastructure**: Inference Gateway, Secrets, Files, Entity Store, Jobs.

NeMo Helix optimizes LangGraph agents wrapped in NAT today. If a user has an agent in another framework (CrewAI, AutoGen, plain LangChain, Pydantic AI), the build/optimize path requires a NAT wrapper, which they would write themselves. Be honest about this when users ask.

## For developers

@AGENTS.md
@AGENTS.local.md
