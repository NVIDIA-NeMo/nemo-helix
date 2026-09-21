---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-build-agent
description: Use when a customer asks NeMo Platform to build, implement, package, register, or deploy a new agent from an idea or approved Ethos. Builds the supported LangChain Deep Agents shape through Fabric. Do not use for design-only work, focused agent.yaml editing, testing an existing deployment, onboarding an existing non-Deep-Agents agent, or maintaining NAT workflows.
triggers:
  - nemo-build-agent
  - build an agent
  - implement the approved agent
  - package the agent
  - deploy the new agent
  - build from the agent Ethos
  - onboard the new agent through Fabric
not-for:
  - nemo-explore (use for design-only discovery)
  - nemo-ethos (use to persist a design without implementation)
  - nemo-agent-config (use for focused agent.yaml authoring or validation)
  - nemo-try-agent (use to query an existing deployment)
  - existing agent onboarding through another Fabric adapter
  - migration or continued operation of NVIDIA NeMo Agent Toolkit workflows
preconditions:
  - nemo_setup_complete
  - workspace_exists
  - provider_registered
compatibility: NeMo Platform >= 0.4.0; installs the optional NeMo Agents plugin when the user approves and verifies its supported Fabric Deep Agents adapter; Python and uv for local MCP projects; network access and provider credentials for live model tests; Docker for packaged custom code; macOS or Linux.
maturity: beta
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read, Write, Edit
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Build a NeMo Platform agent

Build a tested, config-driven LangChain Deep Agent and onboard it through the
NeMo Platform Fabric path. Fabric owns runtime orchestration and constructs the
agent from config. The optional NeMo Agents plugin supplies the Fabric harness
adapters and their runtime dependencies. Customer code runs only when it is
packaged as a config-referenced service such as MCP.

## Explain the path in customer language

Do not assume the user knows NeMo Platform, Fabric, Ethos, MCP or Agent Skills.
Introduce each term only when it affects a decision.

Before writing implementation files:

1. Explain that an Ethos is a plain-language design contract for the agent. It
   records who the agent serves, what it should accomplish, the tools and data
   it may use, required approvals, forbidden actions, failure behavior and the
   examples that will become tests. It is not code or deployment config.
2. Explain that the Ethos will be drafted from the user's answers, shown for
   review and revised until the user explicitly approves it. No implementation
   begins before that approval.
3. Explain that the supported build path uses the external LangChain Deep
   Agents runtime. NeMo Platform does not vendor the customer's agent. The
   optional NeMo Agents plugin installs the supported Fabric adapter and its
   Deep Agents dependency. Fabric registers the agent and makes it available
   for deployment, testing, observation and optimization in NeMo Platform.
4. Explain executable tools only when the requested agent needs them: custom
   code is exposed through MCP so the deployed agent can call it. Explain Agent
   Skills only when an instruction package is the right artifact.
5. Ask whether to continue with this supported path.

If the user declines, stop. Offer onboarding of an existing agent through an
available Fabric adapter. Do not fall back to NAT.

## Approve and persist the design

Use `agents/<agent-name>-ethos/ETHOS.md` as the canonical design and package
root. If an Ethos exists, summarize it in plain language and confirm that it is
approved for this build. If it is absent or needs changes, gather the design
inputs conversationally. Ask about the customer's work first and translate the
answers into NeMo artifacts yourself:

- concrete role, users and outcomes;
- tools, data, credentials and side effects;
- constraints, approvals, forbidden actions and sensitive data handling;
- five to ten representative tasks with expected outcomes;
- mandatory ordering or transactional invariants.

Route unresolved design questions to `nemo-explore`. Then invoke `nemo-ethos`
to render, validate and save `agents/<agent-name>-ethos/ETHOS.md`. Show the
result and wait for explicit approval. Do not create implementation files before
the Ethos is approved.

## Confirm the build environment

After the Ethos is approved and before creating implementation files, confirm
the target workspace, environment, model provider, network access, credential
availability and whether Docker deployment is available. Discover what can be
read from the current environment instead of asking the user to supply NeMo
specific details they may not know. Explain any missing prerequisite in terms
of the capability it blocks.

Check `nemo plugins list -f json` and `nemo agents --help`. Then inspect the
active Python environment for `nemo_agents_plugin`,
`nemo_fabric_adapters.deepagents` and `deepagents`. Do not infer that a harness
is available from config acceptance alone.

If the NeMo Agents plugin and Deep Agents harness are already available, reuse
them. Do not reinstall or change their versions. If the plugin is absent,
explain that agent management is optional in NeMo Platform and that this build
requires it. Show the appropriate install command and ask for approval before
running it:

- Published Platform install: `uv pip install "nemo-platform[nemo-agents-plugin]"`
- NeMo Platform source checkout: `uv pip install -e plugins/nemo-agents/`

The plugin owns selection of compatible Fabric adapter and harness versions.
Do not add a separate Deep Agents version constraint. After installation,
restart Platform services and repeat all four checks. If the plugin is present
but its Deep Agents adapter or runtime is absent, report a broken plugin
installation. Offer to reinstall the same Agents plugin only after approval.
Do not install the harness independently as an untracked repair.

## Choose supported artifacts

Read [references/fabric-deep-agents.md](references/fabric-deep-agents.md).
Select the smallest supported shape:

- Put core behavior in `agent.yaml` instructions.
- Use an Agent Skill for a reusable instruction package.
- Use MCP for every executable custom tool.
- Add a declarative subagent only for a distinct delegated task.
- Put a safety or transactional sequence inside one deterministic MCP operation.

Do not create `agent.py` as an agent entry point. Do not put Python callables or
compiled local graphs into Deep Agents settings. Stop when a requirement cannot
be expressed by the installed adapter contract.

The current deployed Deep Agents path does not expose a verified end-to-end
resume contract for runtime human approval. If the Ethos requires an in-run
approve, edit or reject step, stop and report that adapter gap. Do not treat an
accepted `interrupt_on` setting as proof that the deployed workflow can resume.

## Build the project

Keep the deployable project under `agents/<agent-name>-ethos/`. Create only the
files required by the selected shape. For custom Python tools, use a `uv`
project with a locked dependency set, a typed MCP server and a console script.
Do not install dependencies globally. Do not add `deepagents` or a Fabric
adapter to the generated project unless its own code directly imports that API.

Give tools narrow schemas, bounded output, explicit permissions, capped retries
for transient failures and redacted errors. Never write credentials, customer
data or production traces into source, YAML, fixtures or logs.

Use `nemo-agent-config` to author the canonical
`agents/<agent-name>-ethos/agent.yaml`, but do not run its `nemo agents create`
registration step. This build workflow owns registration after every
pre-registration gate has passed. Require:

```yaml
config_format: nemo-agents-spec-v1
default_harness: deepagents
harnesses:
  deepagents:
    kind: deepagents
```

Use the model verified by `nemo-model-selection`. Validate every adapter setting
against the installed descriptor. Keep referenced paths relative to
`agent.yaml`.

## Test before registration

Read [references/testing-and-signoff.md](references/testing-and-signoff.md).
Derive one acceptance case file from the approved Ethos and reuse it for local
tests, deployed invocation and evaluation.

Require unit tests, MCP contract tests, behavioral tests and trajectory tests
where tool choice, approval or order matters. Keep live model and integration
tests separate. If credentials or network access are absent, record the exact
live test as skipped. A skipped test is not evidence that the integration works.

Do not run production side effects as representative tests. Use mocks, a sandbox
or a test tenant. Obtain explicit approval for any live action that can mutate a
business system.

Stop before registration when a required test, Fabric translation, plan,
diagnostic or delivery reachability check fails.

## Package custom code

If the agent includes a local MCP server or another Python package, read
[references/packaging.md](references/packaging.md). Inspect the entire build
context for secrets and sensitive data, then package in project mode. Do not use
`--skip-validation`.

Use Docker deployment for packaged custom code. A subprocess deployment is only
valid when every referenced executable is already installed on the Platform
service `PATH`. Do not assume the generated project's virtual environment is
visible to that service.

## Register and deploy

Check for an existing Agent and deployment with the requested names. If either
exists, offer reuse, rename or replace. Never overwrite or delete it without
explicit approval immediately before the state change.

Show the exact create and deploy commands and ask for approval immediately
before running them. `nemo agents create` is both the final Fabric plan and
doctor gate and the registration step; do not run it earlier solely to validate
the config. Pass the confirmed workspace to both commands. If an
AgentEnvironment was selected, pass its workspace-qualified reference to
deployment. Verify registration by reading the Agent back in the same
workspace. Verify deployment through the blocking command result and deployment
status. On failure, inspect status and logs once, report the root error and
stop.

## Verify onboarding

Invoke the named deployment with safe acceptance cases. Require non-empty
responses, expected structured output and the required MCP tool calls. Exercise
one denied action and one upstream failure without causing a production side
effect.

Confirm Fabric telemetry reached the configured destination. When Intake is
enabled, verify at least one trace with the expected agent, model and tool spans.
Treat missing telemetry or unreachable tools as incomplete onboarding even when
the final answer looks correct.

Run `nemo-evaluator` only after invocation passes. Use the approved Ethos cases
and thresholds. Report passed, failed and skipped checks separately.

## Gotchas

- Fabric constructs the Deep Agent from `agent.yaml`; it never imports customer
  `agent.py`.
- Keep `ETHOS.md`, `agent.yaml` and packaged artifacts together under
  `agents/<agent-name>-ethos/` so registration uploads one canonical bundle.
- Custom Python code is deployable only when a declared MCP server or another
  supported runtime surface can reach it.
- Prompt instructions and subagent delegation do not guarantee fixed ordering.
- Packaging copies the selected build context. Untracked secrets can enter an
  image even when they are absent from committed files.
- The optional NeMo Agents plugin supplies the selected harness adapter and its
  runtime dependencies. Reuse an installed harness and let the plugin resolve
  compatible versions.
- Docker is the supported local container path. Treat Kubernetes as a separate
  environment contract that must be verified.

## Stop conditions

Stop without registration or deployment when the Ethos is unapproved, the
adapter cannot express a requirement, a required tool is unreachable, a local
gate fails, a secret is present in the build context or Fabric validation fails.

Stop without production-candidate status when a live test is skipped, telemetry
is missing, a target integration is mocked or an Ethos threshold is unmet.

Call the result `Built`, `Onboarded` or `Production candidate` only according to
the evidence levels in `references/testing-and-signoff.md`.
