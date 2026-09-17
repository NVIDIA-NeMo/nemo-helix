<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# email-security-triage — sample traces

Eighteen hand-written ATIF trajectories for the `email-security-triage` agent, one file per
trace. They exercise all three of the agent's capabilities and carry three deliberate failure
clusters, so an evaluation run has something to pass _and_ something to fail on — and so the
Insights analyst has enough repeated evidence to file a recommendation.

The two maintenance scripts that read this directory live in `web/packages/studio/scripts/`
(`bump-trace-timestamps.ts`, `convert-trace-formats.ts`) and run as `pnpm traces:bump` and
`pnpm traces:convert` from `web/packages/studio`.

| Files                                          | Pattern                          | Why it's here                                                                                                                                                                                                                       |
| ---------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trace-01`, `trace-02`, `trace-17`, `trace-18` | `triage_message`, correct        | Healthy contrast. Without these the set reads as uniformly broken and no cluster stands out.                                                                                                                                        |
| `trace-03`                                     | `review_messages`, correct       | No question typed, two messages, one block each.                                                                                                                                                                                    |
| `trace-04`                                     | `draft_warning`, correct         | `format` scored 0.5 — over the 80-word bar.                                                                                                                                                                                         |
| `trace-07` … `trace-11`                        | **Routing failure (5 sessions)** | A direct triage question arriving with a _multi-message_ selection falls back to `review_messages`, so the analyst gets an ANALYSIS block where the contract promises one word. Verdicts are right; `routing` and `format` are 0.0. |
| `trace-05`, `trace-12` … `trace-14`            | **Missed phishing (4 sessions)** | Every lure keeps its portal link on the sender's own lookalike domain, so there is no sender/link mismatch to catch. `correctness` 0.0.                                                                                             |
| `trace-06`, `trace-15`, `trace-16`             | **Timeouts (3 sessions)**        | Terminal `AgentTimeoutError` on long bodies or large selections. Root span errored, no agent step.                                                                                                                                  |

The three bolded clusters are deliberate: each spans at least three sessions with one shared
trigger condition, which is the shape the Insights analyst files on.

Each carries `extra.verifier_result.rewards` with `correctness`, `format`, and `routing`, which
Intake turns into a `harbor.verifier` evaluator span plus one evaluator result per criterion —
no separate `/evaluator-results` call needed.

## Import

```bash
export NMP_BASE_URL=http://127.0.0.1:8080
export WORKSPACE=default

for f in trace-*.json; do
  curl -sS -X POST "$NMP_BASE_URL/apis/intake/v2/workspaces/$WORKSPACE/ingest/atif" \
    -H 'Content-Type: application/json' --data-binary "@$f" \
    -o /dev/null -w "$f -> %{http_code}\n"
done
```

`201` with an empty body is success. Verify by reading them back:

```bash
curl -sS -g "$NMP_BASE_URL/apis/intake/v2/workspaces/$WORKSPACE/spans?filter[agent_name]=email-security-triage&page=1&page_size=100"
```

## Other ingest formats

Studio's import modal routes each picked file by sniffing its shape, so `formats/` holds the same
runs re-expressed in every shape it can route. Regenerate the four span and chat-completions
files from `web/packages/studio` with:

```bash
pnpm traces:convert
```

| File                            | Shape                                                                       | Endpoint                                                                                                  |
| ------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `formats/spans-array.json`      | Bare array of spans, from `trace-17`                                        | `ingest/spans`                                                                                            |
| `formats/spans-batch.json`      | `{source: "langsmith", spans: [...]}`, from `trace-02` and `trace-06`       | `ingest/spans`                                                                                            |
| `formats/spans.jsonl`           | Line-delimited spans, from `trace-07`                                       | `ingest/spans`                                                                                            |
| `formats/chat-completions.json` | Captured request/response pairs, from `trace-17` and `trace-02`             | `ingest/chat-completions`                                                                                 |
| `formats/otlp-traces.binpb`     | Serialized OTLP `ExportTraceServiceRequest`, from `trace-17` and `trace-02` | `ingest/otlp/v1/traces`                                                                                   |
| `formats/otlp-traces.json`      | The same export as OTLP JSON                                                | **none** — the endpoint is protobuf-only, so this file exists to prove the modal refuses it with a reason |

### The two OTLP files are static

`traces:convert` does not rebuild them. Encoding OTLP needs a protobuf writer, and adding one —
as a dependency or as hand-rolled encoding — is not worth it for two fixtures that change about
as often as the ingest contract does.

They hold the same span tree `toSpans` produces for `trace-17` and `trace-02`, plus
`openinference.span.kind`, `input.value`, `output.value`, and `session.id`, under a single
`service.name=email-security-triage` resource. To rebuild them, encode that tree with any OTLP
protobuf writer — `opentelemetry-proto` is already present in the repo's Python `.venv`, so a
throwaway script there needs no new dependency in either tree.

The span and OTLP files carry `gen_ai.agent.name`, so they attach to the agent. Chat-completions
ingest has no agent field at all, so those two calls land as queryable spans that no agent page
will show — that is the format's limitation, not a defect in the sample.

`toSpans` and `toChatCompletions` reuse the source `session_id`, so importing both an ATIF trace
and its converted twin groups them under the same session in Intake — only `span_id`/`trace_id`
are derived.

## Insights

The set is built to earn an Insight rather than just to import cleanly. The analyst's bar is
"at least three representative traces per Insight", it ranks error-status spans and evaluator
regressions above one-off outliers, and it weighs patterns that recur across many sessions over
those confined to one. Each cluster above clears that bar on its own.

The periodic controller also skips a scheduled run for an agent with fewer than 10 new traces
since its last cursor, which 18 clears comfortably.

Analysis must be enabled for the agent before a run can be triggered — the analyze-job spec needs
the default/fast model pair, and that is only captured on the agent's analysis config:

```bash
uv run nemo insights analysis enable --agent email-security-triage
```

Studio's **Import traces** modal triggers a run per agent automatically after a successful import
(the "Run insights analysis after import" checkbox). Without the config above it reports
`analysis not enabled` and leaves the import untouched.

## Notes

- **Session IDs are fixed** (`3f2a1c00-000N-…`), so a re-import lands on the same sessions rather
  than fanning out into new ones.
- **Timestamps expire.** Intake's ClickHouse TTL is 90 days from `started_at`, so once these fall
  outside that window the ingest is rejected. Run `pnpm traces:bump` from `web/packages/studio` to
  roll the whole set forward — it preserves the spacing between traces and between steps, and by
  default lands the newest step one day before now:

  ```bash
  pnpm traces:bump                 # newest step -> ~1 day ago
  pnpm traces:bump --days 30       # shift everything forward 30 days
  pnpm traces:bump --dry-run       # print the shift, write nothing
  ```

  Prefer this over raising the `spans` / `trace_index` table TTLs.

- **A bump does not move the OTLP pair.** `formats/otlp-traces.*` are static, so their timestamps
  stay where they were while the ATIF traces roll forward. That only matters if you need the OTLP
  upload path to land in a live Intake — everything else, including the modal's format routing and
  its protobuf-only rejection of the JSON file, works regardless of how old those two are.
- **No `evaluation_context`.** Adding one requires the named Evaluation to already exist, or the
  request is rejected with `400`. To publish these as a named run instead, add
  `"evaluation_context": {"evaluation_name": "...", "test_case_name": "..."}` at the top level
  after creating the Evaluation — see the `nemo-experiments-upload` skill.
- Email addresses and domains are invented; the content mirrors
  `plugins/nemo-agents/examples/nemo-agent-config/email-security-triage/dataset.jsonl`.
