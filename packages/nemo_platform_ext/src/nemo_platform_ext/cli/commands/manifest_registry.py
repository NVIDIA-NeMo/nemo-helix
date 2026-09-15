# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_ext.cli.commands.config_help import CONFIG_APP_HELP
from nemo_platform_ext.cli.manifest import TopLevelEntry, functional_plugin_entry

TOP_LEVEL_ENTRIES = (
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.auth:app",
        help="Manage authentication for NeMo Platform.",
        name="auth",
        panel="Setup",
        kind="group",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.config:app",
        help=CONFIG_APP_HELP,
        name="config",
        panel="Setup",
        kind="group",
        hidden=True,
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.setup:setup_command",
        help="""\
Set up NeMo Platform: connect or start services, configure a provider, install skills.

Uses an already-running platform, starts local services, or connects the
CLI to an existing remote deployment. Then selects and registers an
inference provider, picks default and fast agent models, installs coding
agent skills, and optionally deploys a demo agent.

The active config context remembers the Platform URL. When a remote
deployment is already reachable, setup asks whether to continue with it,
start local services instead, or connect to a different remote URL.

To override the URL for one run only:
  nemo --base-url http://localhost:8080 setup

To persist a different URL:
  nemo config set --base-url http://localhost:8080

Requires an interactive terminal (TTY). In non-interactive contexts
(CI, piped input), pass --auto to use environment variables instead.

Use --auto for non-interactive setup from environment variables
(NEMO_DEFAULT_INFERENCE_KEY, NVIDIA_API_KEY, OPENAI_API_KEY,
ANTHROPIC_API_KEY, GEMINI_API_KEY).
Override the selected pair with NEMO_DEFAULT_MODEL and NEMO_FAST_MODEL.

Examples:
  nemo setup
  nemo setup --auto
  nemo setup --auto --start-services --install-skills --deploy-agent
  nemo setup --auto --start-services --ready-timeout 360
  NMP_BASE_URL=https://nmp.example.com NMP_ACCESS_TOKEN=... nemo setup --auto --no-start-services
  nemo setup --workspace my-workspace
  nemo setup --no-install-skills --no-deploy-agent
  nemo --base-url http://localhost:8080 setup""",
        name="setup",
        panel="Setup",
        kind="command",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.skills.cli:app",
        help="""\
Install AI agent skill files for Nemo.

Supported agents: claude, codex, cursor, opencode

Examples:
# List available skills.
nemo skills list
# Show a skill's content.
nemo skills show inference
# Install all skills for Claude Code.
nemo skills install --agent claude
# Install specific skills only.
nemo skills install --agent claude --skill inference""",
        name="skills",
        panel="Setup",
        kind="group",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.quickstart.cli:quickstart_app",
        help="Quickstart commands for managing the NeMo Platform container.",
        name="quickstart",
        panel="Setup",
        kind="group",
        hidden=True,
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.services.cli:services_app",
        help="Run platform services locally.",
        name="services",
        panel="Setup",
        kind="group",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.quickstart.cli:cluster_info_app",
        help="Show information about the connected platform cluster.",
        name="cluster-info",
        panel="Setup",
        kind="group",
        hidden=True,
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.use_cases.chat:chat",
        help="""\
Start an interactive chat session with a model.

By default, uses model entity routing where the model name should match
what's shown in 'nemo models list'.

Use --provider for direct provider routing, where the model argument is
passed directly to the provider's API.

Passing PROMPT sends one message and exits unless --interactive is set.
Omitting PROMPT in a TTY starts the interactive chat UI. In non-TTY
contexts, PROMPT may also be piped on stdin. Piped stdin is read in full
before sending. If both PROMPT and piped stdin are provided, PROMPT takes
precedence.

Examples:
  nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5
  nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5 "What is machine learning?"
  nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5 "What is machine learning?" --interactive
  echo "What is machine learning?" | nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5
  nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5 "What is machine learning?" -f json
  nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5 --provider nvidia-build""",
        name="chat",
        panel="CLI functions",
        kind="command",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.docs:docs_command",
        help="""\
Read NeMo Platform documentation.

Examples:
nemo docs get-started/setup
nemo docs set-up/helm/install
nemo docs --list
nemo docs cli/configuration""",
        name="docs",
        panel="CLI functions",
        kind="command",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.use_cases.wait:app",
        help="Wait for resources to reach a desired status.",
        name="wait",
        panel="CLI functions",
        kind="group",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.use_cases.agent:app",
        help="""\
Commands for AI agent context and capability discovery.

Examples:
# Dump full agent context (plugins, commands, skills).
nemo agent context
# List all available commands.
nemo agent commands""",
        name="agent",
        panel="CLI functions",
        kind="group",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.jobs:app",
        help="Manage jobs.",
        name="jobs",
        panel="Core plugins",
        kind="group",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.secrets:app",
        help="Manage secrets.",
        name="secrets",
        panel="Core plugins",
        kind="group",
    ),
    TopLevelEntry(
        import_path="nemo_platform_ext.cli.commands.plugins:app",
        help="""\
Commands for plugin discovery.

Examples:
# List installed plugins.
nemo plugins list""",
        name="plugins",
        panel="CLI functions",
        kind="group",
    ),
    functional_plugin_entry(
        "safe-synthesizer",
        "nemo_safe_synthesizer_plugin.cli:SafeSynthesizerCLI",
        source="plugin",
    ),
)
