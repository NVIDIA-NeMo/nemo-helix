<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Platform

[![License](https://img.shields.io/badge/license-Apache_2.0-D22128?style=flat-square)](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12--3.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Docs](https://img.shields.io/static/v1?label=docs&message=docs.nvidia.com%2Fnemo-platform&color=76B900&style=flat-square&logo=readthedocs&logoColor=white)](https://docs.nvidia.com/nemo-platform)

Make the agents you ship faster, more accurate, and safer.

NeMo Platform brings NVIDIA NeMo libraries together under one CLI, Python SDK, and web UI. Hardening, evaluation, and tuning for the agents you put in production.

## What's here today

- **Secure agents.** Guardrails (content safety, jailbreak detection, PII redaction), Auditor (red-teaming via garak), Anonymizer (PII handling for training data).
- **Evaluate agents.** LLM-as-judge, deterministic, agentic, and RAG benchmarks. Harbor-backed eval suites for regression testing.
- **Tune agents.** Skill optimization, prompt and hyperparameter tuning, Switchyard model routing.
- **Build agents.** NVIDIA NeMo Agent Toolkit (NAT) for LangGraph-based agents. Shared infrastructure: Inference Gateway, Secrets, Files, Entity Store, Jobs.
- **Generate synthetic data.** Generate synthetic data for training or evaluation purposes using Data Designer.
- **NeMo Studio (alpha).** Browser UI for chat, monitoring, and reviewing optimization suggestions. Studio's agent-focused features are still a work in progress; the CLI is the primary surface today.

## Install

**Prerequisites:** Python 3.12–3.14 and an API key for an inference provider (NVIDIA Build, OpenAI, Anthropic, Google Gemini, or a local Ollama instance).

The `nemo-platform` distribution is a convenience wrapper that bundles the SDK, shared runtime packages, default first-party plugins, and platform services into a single wheel. The `all` extra adds the platform services; drop it for the SDK and CLI alone.

Recommended — one command, `nemo` available everywhere, nothing to activate:

```bash
uv tool install "nemo-platform[all]"
```

Use a virtual environment instead when you also import the SDK from your own code. `nemo` then works only while the environment is active:

```bash
uv venv
source .venv/bin/activate
uv pip install "nemo-platform[all]"
```

Plain `pip install "nemo-platform[all]"` works in an environment you manage yourself.

Then bring up the platform:

```bash
nemo setup
```

`nemo setup` starts local services, registers your LLM provider, discovers available models, installs agent skills, and (optionally) deploys a sample agent.

Verify:

```bash
nemo services status
```

The recommended developer setup is to clone the repo and use `make bootstrap` instead of installing from PyPI — see the [setup playbook](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/SETUP.md) for the full walkthrough, including local data dir, DB reset, and troubleshooting.

## Where to go next

After `nemo setup` completes, the CLI prompts you to pick one of two paths. The same two paths are the recommended starting points if you came in via PyPI:

- **Build and optimize agents.** Open a coding agent session inside your agent's project directory and ask: _"Build and optimize an agent using NeMo Platform."_ The shipped skills will scaffold the agent, deploy it, run evaluations, suggest optimizations, and add guardrails on request.
- **Explore the platform.** From any coding agent session, ask: _"What can I do with NeMo Platform?"_ to get a guided tour of the capabilities surfaced through skills.

NeMo skills work with Claude Code, Codex, Cursor, OpenCode, and other coding agents. Install or refresh them with `nemo skills install --agent <agent>` for built-in harnesses, or `--path` for another Agent Skills-compatible directory:

```bash
nemo skills install --agent claude
nemo skills install --path ~/.my-agent/skills
nemo setup --install-skills --skills-path ~/.my-agent/skills
```

## Operating the platform

A few useful CLI commands once setup completes:

```bash
nemo --help                # All commands
nemo models list           # Available models
nemo services status       # Platform health
nemo skills list           # Skills installed on the platform
```

Every capability is also available via REST API. Model inference uses the model IDs returned from `nemo models list` and is available at:

```text
http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1/chat/completions
```

## Links

- **Source:** https://github.com/NVIDIA-NeMo/nemo-platform
- **Documentation:** https://docs.nvidia.com/nemo-platform/
- **Setup playbook:** https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/SETUP.md
- **CLI reference:** https://docs.nvidia.com/nemo-platform/documentation/reference/cli-reference
- **API reference:** https://docs.nvidia.com/nemo-platform/documentation/reference/api-reference
- **Issue tracker:** https://github.com/NVIDIA-NeMo/nemo-platform/issues

## License

NeMo Platform is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/LICENSE). Third-party open-source dependencies have their own licenses; review them before use.
