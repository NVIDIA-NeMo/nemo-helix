<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Legacy NAT workflow output

Use only when the user is explicitly maintaining a legacy NAT workflow YAML. Platform `agent.yaml` output is in `SKILL.md` Step 4.

## NAT compatibility block

```yaml
llms:
  primary_llm:
    _type: <nat-type>
    model_name: <model-string>
    # Chosen because: <one plain-English sentence — same words you used above>
    # Evidence: <direct | inferred from <ancestor-id> | name-intent only — eval recommended>
    max_tokens: 4096

workflow:
  _type: <agent_type>
  llm_name: primary_llm
  tool_names: []
```

## Picking the right legacy NAT `_type`

Only for NAT workflow output, match the chosen model's namespace prefix against
`namespace_to_type[]` from the cache:

```txt
For each rule in cache.namespace_to_type:
  if model_string.startswith(rule.prefix):
    use rule.nat_type for the YAML _type field
    if rule.note is present, surface it once to the user
    stop
If no rule matches:
  Tell the user the namespace is unrecognized. Run this to confirm the right type:
    uv run nat info components -t llm_provider
  Until the user provides the type, leave _type as <TBD-verify> in the emitted YAML
  rather than guessing.
```

Common mappings the cache carries today: `nim/*`, `openai/*`, `anthropic/*`, `bedrock/*`, plus vendor-published NIM names (`qwen/*`, `meta/*`, `nvidia/*`, `microsoft/*`, `mistralai/*`) that route through the NIM provider when served by the platform. Ollama's local endpoint maps to `_type: openai` since it exposes an OpenAI-compatible API.

For Platform `agent.yaml`, represent credentials with `api_key_env` and put
`base_url` directly in the model block. For a legacy NAT workflow, use the
provider fields required by that NAT LLM component.

## Pair a legacy NAT model with the right workflow type

Use this table only when maintaining NAT workflow YAML. Harness selection for
`nemo-agents-spec-v1` belongs to `nemo-agent-config` and must not be inferred
from a NAT workflow type.

| What the agent needs to do | Use |
|---|---|
| Call tools in a loop, observe results, adjust | `react_agent` |
| Dispatch tools directly without intermediate reasoning | `tool_calling_agent` |
| Reason through a problem before acting | `reasoning_agent` |
| Follow a fixed plan with no dynamic re-planning | `rewoo_agent` |
