<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Insights Analyst as ExecuteAgentJob Demo

This directory demonstrates running the Insights Analyst through AnalysisRuns
backed by the generic `agents.execute` job.

## Pieces

- `demo_spans.json`: a scenario spec for the target agent's telemetry, stored
  with relative time offsets so it never ages out of Intake's retention window.
- `seed_intake.py`: expands the spec into spans and annotations, posts them to
  Intake, then reads them back.
- `reset_intake.py`: deletes the demo source's spans and annotations so the
  demo can start from a clean corpus.

## Prerequisites

- A running local platform with Intake, Agents, Jobs, and Insights, and a
  running ClickHouse for Intake (see `services/intake/README.md`).
- `nemo setup` completed, so a default/fast Model Entity pair exists. The
  Analyst resolves its models as Platform Model Entities. Each ref is a
  workspace-qualified entity name such as
  `default/nvidia-nemotron-3-super-120b-a12b`, not a raw provider model id.

  `nemo setup` saves the pair to your CLI config. `nemo config view` prints the
  refs as `default_model` and `fast_model`. To browse other candidates, list
  entity names and prefix the one you want with its workspace:

  ```bash
  nemo config view
  nemo models list --all-pages -f csv -c name | grep nemotron
  ```

  A Model Entity existing locally does **not** mean its upstream provider still
  serves it. Auto-discovered catalogs go stale, and a dead entry surfaces as a
  gateway `424` wrapping upstream provider errors. `nemo chat` takes the same
  ref, so probe each model before relying on it:

  ```bash
  MODEL=default/nvidia-nemotron-3-super-120b-a12b  # a ref from the commands above
  nemo chat -m "$MODEL" "Reply with just: ok"
  ```

  A live model replies. A stale one prints `API error: (424) Failed
  Dependency`; pick another model and pass it to the run with
  `--default-model` or `--fast-model`.

The target agent (`demo-agent` below) does not need to exist as an Agent
entity — the Analyst only matches it against each span's normalized
`agent_name`.

## Demo Flow

### 1. Seed Intake

Seed Intake with telemetry for the target agent:

```bash
uv run plugins/nemo-insights/examples/execute-agent-job/seed_intake.py \
  --base-url http://localhost:8080 \
  --workspace default \
  --target-agent demo-agent
```

This posts to `POST .../ingest/spans` and `POST .../annotations`, then reads
both back.

**The corpus is sized against the Analyst's own bar, not for brevity.** The
Analyst files an Insight only for patterns it can evidence with at least
three representative traces, and it ranks issues recurring across many
sessions above one-offs — so a handful of sessions reliably produces "no
high-impact failure patterns detected". The spec expands to 84 spans across
28 sessions spanning ~3.6 hours:

| Scenario | Sessions | Failure |
|---|---|---|
| `retrieval` | 8 | `knowledge_search` returns zero documents; the agent answers anyway |
| `handoff` | 6 | `delegate_task` fires without the conversation summary |
| `billing` | 5 | `billing_lookup` times out at 30s |
| `healthy` | 9 | none — grounded, cited answers |

Plus 12 negative and 6 positive `feedback` annotations and 18 numeric
`helpfulness` labels. Feedback matters: the Analyst's method says to start
there, because it is the strongest signal of a real problem. Annotations are
attached at session level, which is both the realistic shape for an end-user
thumbs-down and the id the Analyst correlates back to spans with.

Re-running is safe. Intake's span table sorts on `start_time`, so a repeat
post only updates a span in place when its timestamp is unchanged. The
script therefore reuses the origin of the corpus already in Intake rather
than re-anchoring to now, and skips annotations that already exist. Passing
a different `--started-at` over an existing corpus writes a second copy of
every span. The Analyst then fails with `trace ... contains duplicate span
id`.

To start over, for example after changing the spec or seeding with a
different `--started-at`, delete the demo corpus and reseed:

```bash
uv run plugins/nemo-insights/examples/execute-agent-job/reset_intake.py \
  --base-url http://localhost:8080 \
  --workspace default
```

Intake has no public span delete API, so this removes the demo source's
spans directly in ClickHouse. It finds the ClickHouse container that Intake
runs locally, or uses `NHX_INTAKE_CLICKHOUSE_URL` when that is set.
Only spans and `trace_index` rows from the demo source are deleted there;
spans from other sources are untouched. Annotations have no source, so the
script deletes, through the Intake API, every annotation on a session that has
demo spans. That includes any annotation someone else added to those sessions.
Pass `--dry-run` to see the counts first.

### 2. Submit an analysis run

Submit an analysis run and wait for it to finish:

```bash
nemo insights analysis-runs create --agent demo-agent --wait
```

`--wait` reports each status change on stderr and prints the final run as
JSON on stdout; the run's name is at `run.name`. The command exits non-zero
if the backing job does not complete.

The model pair is **required** on the request. It lives only in the
operator's local CLI config (`~/.config/nhx/config.yaml`), which the
Platform process cannot read, so the request has to carry it. The CLI fills
it in from that config. Pass `--default-model` and `--fast-model` to override
either one.

The same run through the SDK. Unlike the CLI, the SDK does not read your CLI
config, so both models are passed explicitly:

```python
from nemo_helix import NeMoHelix

sdk = NeMoHelix()
response = sdk.insights.analysis_runs.create(
    workspace="default",
    agent="demo-agent",
    default_model="default/<default-model>",
    fast_model="default/<fast-model>",
)
final = sdk.insights.analysis_runs.wait(workspace="default", name=response.run.name)
print(final.job_status)
```

`NeMoHelix()` connects to your active CLI context's base URL with its
credentials. Pass `base_url=` to target another instance.

### 3. Inspect the run

If you did not use `--wait`, check on the run:

```bash
nemo insights analysis-runs list --agent demo-agent
nemo insights analysis-runs get <run-name>          # joined with its job
nemo insights analysis-runs get <run-name> --wait   # poll to a terminal state
```

A run and its backing job share one name, and `get` returns them together.
A `job` of `null` means submission never landed and the run can be
resubmitted.

### 4. Download the report

The `analysis-report` result, saved by the Insights execute extension, is the
durable record of what the run did:

```bash
nemo jobs results list <run-name>
nemo jobs results download analysis-report --job <run-name> -o analysis-report.txt
cat analysis-report.txt
```

A run against the demo corpus reports something like:

```text
Analyzed 28 traces: 2 new insights, 0 existing insights with new evidence.

- created: delegate_task requires conversation_summary [insights-insight-542UamS99TFJTt8TMb5Fy9] (6 trace refs)
- created: billing_lookup deadline exceeded [insights-insight-UjirgSpzwDmsF2TPgcsZi2] (5 trace refs)
```

The report is what tells you which Insights *this run* created or updated;
insights carry no per-run provenance.

### 5. List the agent's insights

There is no CLI command for listing insights yet, so use the SDK. From the repo
root, save this as a file and run it with `uv run python <file>`:

```python
from nemo_helix import NeMoHelix

sdk = NeMoHelix()
page = sdk.insights.insights.list_insights(workspace="default", agent="demo-agent")
for insight in page.data:
    print(f"[{insight.status}] {insight.title} ({len(insight.trace_refs)} traces)")
```

`list_insights` also accepts `status` (`open`, `resolved`, or `deleted`),
`page`, `page_size`, and `sort`.
