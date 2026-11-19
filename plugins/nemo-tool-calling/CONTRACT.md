<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `tool-calling` strategy: results contract

The strategy writes these files to `ctx.results` (one results directory per job). Studio's optimization
detail page reads `tool-calling-result.json`; every file also shows in the Artifacts panel.

| File | Contents |
| --- | --- |
| `tool-calling-result.json` | Machine-readable summary, shape below. |
| `report.md` | Human-readable summary of the same content. |
| `candidate-agent.yaml` | The agent config with the suggested config fix applied. |
| `candidate-agent.diff` | Unified diff of `candidate-agent.yaml` against the original config. |
| `schema-patch/<server>/<tool>.json` | Suggested tool definition (`name`, `description`, `inputSchema`) per tool. |
| `schema-patch.diff` | Unified diff of the suggested schemas against the published ones (pretty-printed JSON). |

## `tool-calling-result.json`

```jsonc
{
  "version": 1,
  "agent": "default/data-designer-toolcall",
  "analyzer_model": "default/nemotron-3-5-super",
  "dataset": {
    "source": "traces",                  // "traces" (baseline scored from Intake) | "dataset" (baseline re-run from a prompt file)
    "sessions": 12,                      // traces only: agent sessions scored for the baseline
    "window": { "since": "2026-11-18T20:00:00Z", "until": "2026-11-19T02:00:00Z" },  // traces only
    "prompts": 5,                        // prompts replayed for the verify variants
    "runs_per_prompt": 3
  },
  "variants": [
    {
      "id": "baseline",                  // "baseline" | "config-fix" | "schema-fix"
      "label": "Current agent",
      "description": "The agent as registered.",
      "metrics": {
        "runs": 15,
        "runs_with_errors": 1,           // runs that ended in a harness/transport error (recursion limit, timeout)
        "tool_calls": 60,
        "rejected_calls": 48,            // calls with any call-level signal below
        "tool_error_rate": 0.8,          // rejected_calls / tool_calls
        "signals": {                     // per-signal counts; one call can carry several
          "tool_error": 45,              // MCP error result or the rejection text convention
          "schema_invalid": 40,          // arguments fail the tool's published input_schema
          "error_text": 2,               // heuristic: error-shaped result text with no convention
          "retry_loop_runs": 9           // run-level: runs that retried a tool after a bad call, or exceeded the call budget
        },
        "first_call_accepted_rate": 0.13, // runs whose first tool call was accepted
        "task_success_rate": 0.4,        // runs whose last tool call was accepted
        "avg_calls_per_run": 4.0
      },
      "rejection_reasons": { "corrupted or unknown key": 22, "model/provider not on platform": 12 },
      "per_prompt": { "<prompt id>": { /* same shape as metrics */ } },
      "examples": [                      // at most 5 rejected calls, for the UI
        { "prompt_id": "product-reviews", "tool": "generate_data_designer_job_request",
          "signals": ["tool_error", "schema_invalid"],
          "arguments_excerpt": "{\"job_request\": ...}", "rejection": "columns[0]: unknown field 'sampler_type '",
          "schema_errors": ["job_request/spec: Additional properties are not allowed ('model_configs' was unexpected)"] }
      ]
    }
  ],
  "findings": [
    {
      "id": "lint-open-object-columns",
      "source": "schema-lint",           // "schema-lint" | "failure-analysis"
      "severity": "high",                // "high" | "medium" | "low"
      "tool": "generate_data_designer_job_request",
      "title": "Column items allow any key",
      "detail": "additionalProperties: true on job_request.spec.config.columns[]; 12 optional keys.",
      "evidence": ["22 calls rejected for unknown key 'sampler_type '"]
    }
  ],
  "suggestions": [
    {
      "id": "config-fix",
      "kind": "agent-config",            // "agent-config" | "tool-schema"
      "title": "Name the platform's only model in the prompt",
      "rationale": "12 rejections used provider 'openai'.",
      "applies_automatically": true,     // agent-config: true; tool-schema: false (owner must apply)
      "artifact": "candidate-agent.yaml",
      "diff_artifact": "candidate-agent.diff",
      "verified_by_variant": "config-fix"
    },
    {
      "id": "schema-fix",
      "kind": "tool-schema",
      "title": "One schema per column type, required keys first",
      "rationale": "...",
      "applies_automatically": false,
      "artifact": "schema-patch/data_designer/generate_data_designer_job_request.json",
      "diff_artifact": "schema-patch.diff",
      "verified_by_variant": "schema-fix"
    }
  ],
  "recommended_variant": "schema-fix",   // the variant the recommendation tells the user to adopt; null when none is
                                         // recommended (no fix verified, or no fix beat the baseline)
  "recommendation": "Apply the config fix now and send the schema patch to the tool's owner: together they cut the tool error rate from 80% to 6%."
}
```

Rules:

- Variants are cumulative: `config-fix` is baseline + the config suggestion; `schema-fix` is `config-fix` + the schema
  suggestion served through the schema-overlay proxy. A variant that wasn't run is omitted.
- Rates are floats in `[0, 1]`, rounded to 3 places.
- A call counts as rejected when it carries any call-level signal:
  - `tool_error`: the tool result is an MCP error, or its text starts with `TOOL CALL REJECTED` or contains
    `422 Unprocessable Entity`.
  - `schema_invalid`: the call's arguments fail `jsonschema` validation against the tool's **published**
    `input_schema` (the one the variant advertised). Platform-side; needs no help from the tool.
  - `error_text`: heuristic, only when there is no `tool_error`: result text matching common error shapes
    (`error`, `invalid`, `exception`, `traceback`, an HTTP 4xx/5xx status). Marked heuristic in the UI.
- `retry_loop_runs` is run-level: a run that calls the same tool again right after a bad call, or makes more tool
  calls than the `call_budget` spec field (default 3). It doesn't change `rejected_calls`.
- Rejection reasons come from the tool's own rejection text when present, otherwise from the schema errors.
