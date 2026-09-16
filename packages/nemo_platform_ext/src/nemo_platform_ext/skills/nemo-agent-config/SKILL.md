---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-agent-config
description: Author and validate Platform-managed NeMo Agents agent.yaml files using the nemo-agents-spec-v1 format. Use when the user wants to create, edit, validate, or adapt an agent.yaml file, choose a supported harness, add instructions, skills, MCP servers, tools, environment, or telemetry.
triggers:
  - write agent.yaml
  - create agent.yaml
  - edit agent.yaml
  - validate agent.yaml
  - configure a harness
  - configure agent harness
  - nemo-agents-spec-v1
  - platform agent config
  - adapt agent.yaml
  - convert NAT workflow YAML
  - migrate NAT workflow
  - convert agent.yml to agent.yaml
  - NeMo agent.yaml config
not-for:
  - nemo-build-agent (use for full Ethos-to-deployed-agent build flows)
  - nemo-explore (use to design what the agent should do before writing config)
  - nemo-ethos (use to write ETHOS.md before implementation)
  - nemo-model-selection (use when the user only wants model recommendation)
  - generic YAML editing unrelated to NeMo Platform agents
preconditions:
  - nemo_setup_complete
  - agents_plugin_available
compatibility: nemo-platform >= 0.1.0; writes or edits agents/<name>-ethos/agent.yaml; validates through nemo agents create; supports nemo-agents-spec-v1 configs; safe under sandbox.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: [Read, Write, Edit, Bash]
---

# NeMo Platform agent config

Create or edit the Platform-managed `agent.yaml` for a NeMo Agent. This skill
owns the machine-readable config shape for `nemo-agents-spec-v1`; `nemo-build-agent`
owns the full build/deploy/eval workflow.

Use product-facing Platform language. Do not ask users to write raw Fabric SDK
configuration. Fabric is an implementation dependency behind the Platform-managed
agent config.

## Storage model

The local config lives next to the human-readable Ethos:

```txt
agents/<agent-name>-ethos/
  ETHOS.md
  agent.yaml
```

The platform stores the parsed `agent.yaml` contents in the `Agent.config`
payload with:

```yaml
config_format: nemo-agents-spec-v1
```

The canonical remote config location is derivable from workspace and agent name:
`<workspace>/<agent-name>-ethos#agent.yaml`. Do not invent a separate ref field.

## What you do

1. Confirm the agent name and config path. Default to
   `agents/<agent-name>-ethos/agent.yaml`.
2. Select one supported harness:
   - `codex`
   - `hermes`
   - `deepagents`
   - `claude`
3. Invoke `nemo-model-selection` to select and compatibility-test the model for
   the chosen harness. Configure `models.default` only after that skill returns
   a verified provider and exact model name. Add a harness-local `model`
   override only when that harness intentionally uses a different verified
   provider, model, credential env var, or base URL.
4. Start from `references/templates/agent.yaml` unless the user is editing an
   existing file. Write only the verified model fields returned in Step 3.
5. Remove every unselected harness block and every remaining placeholder. Keep
   an additional harness only when the user explicitly requests it and its
   model configuration has been separately verified.
6. Add system instructions under `instructions.system.content`.
7. Add optional skills, MCP servers, blocked tools, environment directories, and
   telemetry using only fields in the template.
8. Keep all local file paths relative to the directory containing `agent.yaml`.
9. Validate by running `nemo agents create` against the config. Do not treat a
   successful create as validation of an unused or untested harness.

For the standard subprocess or default-image deployment path, keep
`skills.paths` empty. Registration validates relative skill paths against the
authoring directory, but these deployment paths currently materialize only
`agent.yaml` and do not stage the referenced directories.

When non-empty `skills.paths` are required, each path must be relative to
`agent.yaml`, remain inside its packaging context, and contain `SKILL.md`.
Require the explicit image-packaging path before deployment:

```bash
IMAGE_TAG="${AGENT_NAME}:local"
.venv/bin/nemo agents package \
  --agent "agents/$AGENT_NAME-ethos/agent.yaml" \
  --tag "$IMAGE_TAG"
```

Deploy that image with `--mode docker --image "$IMAGE_TAG"`; publish it first
and use the published tag for Kubernetes. Do not use a subprocess or
default-image deployment for an agent with relative `skills.paths`.

## Migrating from legacy NAT workflow YAML

If the user has an existing NAT workflow YAML and wants the new Platform-managed
`agent.yaml` format, treat the migration as best-effort authoring. Do not
overwrite the original NAT YAML unless the user explicitly asks.

Map only fields with a clear Platform equivalent:

| NAT workflow concept | Platform `agent.yaml` target |
|---|---|
| LLM/provider/model block | `models.default` or a harness-local `model` |
| System prompt or workflow prompt | `instructions.system.content` |
| Workflow/tool loop choice | `default_harness` plus `harnesses.<name>.kind` |
| Tool/function references | `skills.paths`, `mcp.servers`, `tools.blocked`, or harness settings when clearly supported |
| Tracing or telemetry settings | `telemetry` |

For an older Codex NAT workflow that used the removed experimental adapter,
apply this mapping instead of copying the workflow fields into
`harnesses.codex.settings`:

| NAT Codex field | Platform `agent.yaml` target |
|---|---|
| `working_directory` | `environment.workspace` |
| `sandbox_mode` | `harnesses.codex.settings.sandbox` |
| `approval_policy: never` | `harnesses.codex.settings.approval_mode: deny_all` when the installed descriptor declares it; otherwise omit |
| `relay_atof_output_dir` | `environment.artifacts`, `telemetry.output_dir`, and `telemetry.atof` |
| `skip_git_repo_check` | Omit; removed CLI-only setting |
| `timeout_seconds` | Omit unless the installed Codex adapter settings schema declares it |
| `max_output_chars` | Omit; no current Codex adapter setting |
| `prefer_chatgpt_auth` | Omit; authentication is selected by the Codex adapter and model configuration |

Treat the installed Fabric adapter descriptor's `settings_schema` as
authoritative. Only place keys declared under its `properties` in
`harnesses.<name>.settings`; do not preserve an unsupported NAT setting merely
because it existed in the source workflow. Surface omitted behavior in the
migration summary. For Codex, map `never` to `deny_all` only when
`settings_schema.properties.approval_mode.enum` includes `deny_all`.

If behavior does not map cleanly, say so directly and choose one:

- Keep the agent on the NAT compatibility path only when all of its component
  types are still installed. The removed experimental harness adapters cannot
  use this option.
- Preserve the original NAT YAML and create a partial `agent.yaml` starter for
  manual completion.
- Mark it as requiring a custom adapter or a manual harness-specific migration.

Never claim a mechanical one-to-one conversion for arbitrary NAT workflows.

## Config shape

Use this structure. Keep unknown fields out of the YAML; the Platform validator
rejects unsupported fields instead of passing arbitrary execution config through.

```yaml
config_format: nemo-agents-spec-v1
name: <agent-name>
description: <short description>

instructions:
  system:
    content: <system instructions>

default_harness: codex

harnesses:
  codex:
    kind: codex
    settings:
      sandbox: workspace-write
      reasoning_effort: high

models:
  default:
    provider: <verified-provider>
    model: <verified-platform-model-name>
    api_key_env: <credential-env-var-if-needed>

skills:
  paths: []

mcp:
  servers: {}

tools:
  blocked: []

environment:
  workspace: ./workspace
  artifacts: ./artifacts

telemetry:
  enabled: false
  provider: relay
  output_dir: ./artifacts/relay
  project: <agent-name>
```

### Harness overrides

Use a harness-local model only when that harness should override the default.

```yaml
harnesses:
  hermes:
    kind: hermes
    model:
      provider: <verified-provider>
      model: <verified-model-name>
      api_key_env: <credential-env-var-if-needed>
      base_url: <provider-base-url-if-needed>
      temperature: 0.0
    settings:
      max_tokens: 512
      reasoning_config:
        effort: none
```

If `base_url` is needed, put it directly in the model block, not under
`settings`.

Use `nemo-model-selection` for every harness. It must verify the exact model
against that harness's model contract before returning it: Responses for
`codex`, the selected provider's chat path for `hermes` and `deepagents`, and
the native Anthropic provider contract for `claude`. Do not route Claude
through Platform IGW. Do not write or finalize `agent.yaml` if model selection
cannot establish a compatible model. For Platform-routed models, availability,
provider metadata, config validation, and Fabric planning do not replace a
valid inference request through the required wire API.

## Validate and register

Before registering, validate the YAML shape with the Platform create path.
Immediately before running `nemo agents create`, show the command to the user,
ask for explicit confirmation, and wait for approval.

```bash
.venv/bin/nemo agents create \
  --name "$AGENT_NAME" \
  --agent-config "agents/$AGENT_NAME-ethos/agent.yaml"
```

If validation fails, fix the named field in `agent.yaml` and retry. Do not
silence validation errors by moving unknown fields into `settings`.

## Deploy and invoke

After create succeeds, show the `nemo agents deploy` command to the user, ask
for explicit confirmation, and wait for approval before running it.

```bash
.venv/bin/nemo agents deploy \
  --agent "$AGENT_NAME" \
  --name "$AGENT_NAME-deployment"
```

The deploy command waits for `running` by default. After it succeeds, invoke the
same explicitly named deployment without another confirmation:

```bash
.venv/bin/nemo agents invoke \
  --agent-deployment "$AGENT_NAME-deployment" \
  --input "<test prompt>"
```

For local one-shot validation without registering an Agent entity, use this
only when the selected model already has a directly usable provider endpoint
and credentials. Platform IGW normalization is applied by the registered
deployment path, not this local path:

```bash
.venv/bin/nemo agents invoke \
  --agent-config "agents/$AGENT_NAME-ethos/agent.yaml" \
  --input "<test prompt>"
```

For a local persistent server, bind to loopback by default. Use an externally
accessible host only when the user explicitly asks to expose the server:

```bash
.venv/bin/nemo agents run \
  --agent-config "agents/$AGENT_NAME-ethos/agent.yaml" \
  --host 127.0.0.1 \
  --port 8080
```

## If validation fails

| Symptom | Cause | Recovery |
|---|---|---|
| `root must be a YAML mapping` | Empty file or list/scalar at the root | Replace with the template shape |
| `extra fields not permitted` | Unknown Platform config field | Remove it or map it into a supported field |
| `default_harness must reference one of harnesses` | `default_harness` does not match a key under `harnesses` | Rename one side so they match |
| `Unsupported harness kind` | Harness kind is not supported by the Platform translator | Pick `codex`, `hermes`, `deepagents`, or `claude` |
| Local file path missing in deployment | Referenced prompts, skills, or assets were not staged | Keep paths relative and package the complete agent bundle into the deployed image |
| Adapter import or binary missing | Selected harness dependency is not installed in the runtime | Install the selected adapter/runtime dependency or choose a harness already available |

## Hard rules

- Keep `config_format: nemo-agents-spec-v1`.
- Keep paths relative to the `agent.yaml` directory.
- Put system instructions under `instructions.system.content`.
- Do not use `prompts` for the default path; top-level prompts are not translated yet.
- Do not create profile files. Profiles are not the Platform authoring contract.
- Do not expose Fabric SDK object names as user-authored YAML fields.
- Do not emit arbitrary adapter settings unless the selected harness documents them.

## Gotchas

- **Default model vs harness model.** A harness-local `model` always wins over
  `models.default`.
- **Registration validates and normalizes.** `nemo agents create` is the
  user-facing validation command.
- **`agent.yaml` is the implementation config, not the Ethos.**
  `ETHOS.md` explains what the agent should do; `agent.yaml` tells the
  Platform how to run it.
- **NAT workflow YAML is a compatibility path.** If the user explicitly asks
  for legacy NAT, route to `nemo-build-agent` and use its NAT template.
