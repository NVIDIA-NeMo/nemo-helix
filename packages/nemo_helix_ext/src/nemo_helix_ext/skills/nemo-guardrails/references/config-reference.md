<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guardrail config JSON structure and field reference

Moved from `SKILL.md`: config JSON shape, content-safety / self-check / streaming-output examples, and the full field reference.

The **main** LLM is always the model from the chat request body (Path B) or
the `--model` flag (Path A). The request model replaces the configured
`models[type=main]` model, while parameters from that entry are merged into
the request model.
Self-check rails (`self check input`/`self check output`) call the main LLM,
so they don't need a `models[]` entry at all. Add `models[]` only for
**task LLMs** (`content_safety`, `topic_control`, `jailbreak_detection`,
`embeddings`, `vision_rails`, etc.) that flows explicitly reference via `$model=<type>`.

### Content safety with a task LLM

This pattern uses a dedicated content-safety classifier (for example,
`default/nvidia-llama-3-1-nemotron-safety-guard-8b-v3`) to assess both user
input and model output against a safety taxonomy. The `content_safety` flows
reference the task LLM via `$model=content_safety`, so the `models[]` entry
**is** required here.

```json
{
  "models": [
    {
      "type": "content_safety",
      "engine": "nim",
      "model": "default/nvidia-llama-3-1-nemotron-safety-guard-8b-v3"
    }
  ],
  "rails": {
    "input":  {"flows": ["content safety check input $model=content_safety"]},
    "output": {"flows": ["content safety check output $model=content_safety"]}
  }
}
```

See [Step 2](../SKILL.md#step-2--create-the-guardrail-config) for the complete runnable
config, including the full prompt taxonomy and output parsers.

### Self-check rails

Self-check rails are most useful for tests and lightweight custom policies.
For production safety, prefer content-safety rails with a dedicated classifier
and taxonomy. Self-check runs against the main/general chat model and expects
the model to answer `Yes` to block or `No` to allow. If the model returns
anything else (for example, reasoning text, a refusal, or a truncated answer),
the parser can default-deny and block the message. See
[Self-check example](../SKILL.md#self-check-example) for a custom-policy example.

### Streaming output rails

```json
"output": {
  "flows": ["content safety check output $model=content_safety"],
  "streaming": {"enabled": true, "chunk_size": 200}
}
```

`streaming` is an `OutputRailsStreamingConfig` object, not a boolean. When
present, `enabled` defaults to `true`, `context_size` to `50`, and
`stream_first` to `true`. When the entire object is omitted, output-rail
streaming is disabled and output rails run on the assembled response only.

### Field reference

**models[]** (optional — only for task LLMs)
- `type` — task to use the model for. Examples: `"main"`, `"content_safety"`, `"topic_control"`,
  `"jailbreak_detection"`, `"embeddings"`, `"vision_rails"`. Entries with `type: "main"` are
  ignored — omit them.
- `engine` — free-form string (commonly `"nim"`); the schema is permissive.
- `model` — `<workspace>/<model-entity-name>`. Must be reachable through IGW;
  use auto-discovered entity IDs from
  `nemo inference providers get ... | jq '.served_models[].model_entity_id'`.

Flows reference task LLMs by appending `$model=<type>` (e.g.
`content safety check input $model=content_safety`). Self-check flows
(`self check input`/`self check output`) run against the main LLM and need
no `models[]` entry.

**rails**
- `rails.input.flows`: enables input rails (e.g.
  `["content safety check input $model=content_safety"]`)
- `rails.output.flows`: enables output rails (e.g.
  `["content safety check output $model=content_safety"]`)
- `rails.output.streaming`: optional `{"chunk_size": N}` for streaming output checks

**prompts[]**
- `task`: the task this prompt template is applied to. Common values:
  `"self_check_input"`, `"self_check_output"`,
  `"content_safety_check_input $model=content_safety"`,
  `"content_safety_check_output $model=content_safety"`. The `$model=<type>`
  suffix binds the prompt to a task LLM declared in `models[]`.
- `content`: prompt template. Use `{{ user_input }}` to embed the user
  message and `{{ bot_response }}` to embed the model's response
  (`{{ user_input }}` is available in both input and output rail prompts;
  `{{ bot_response }}` is only meaningful in output rail prompts). For
  self-check tasks, end with a "Yes or No" question — the model answers
  `Yes` to block, `No` to allow. Content-safety tasks instead emit a JSON
  verdict consumed by `output_parser`.
- `output_parser`: optional parser name. Use this when the model returns
  a safety verdict that must be converted to an allow/block decision.
  `is_content_safe` is the generic parser for simple `safe` / `unsafe` or
  `yes` / `no` outputs. For content-safety checks using NemoGuard models, use
  `"nemoguard_parse_prompt_safety"` for the input rail and
  `"nemoguard_parse_response_safety"` for the output rail.
  For content-safety checks using Nemotron reasoning models, use
  `"nemotron_reasoning_parse_prompt_safety"` for the input rail and
  `"nemotron_reasoning_parse_response_safety"` for the output rail.

**instructions[]** (optional)
- `type`: `"general"`
- `content`: system-level context for the conversation.
