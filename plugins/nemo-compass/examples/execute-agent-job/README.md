<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Compass on sample traces

This directory demonstrates running Compass over sample telemetry. Compass is
the workspace's `compass.analyst` agent, and each analysis is an
`agents.execute` job of it.

## Pieces

- `demo_spans.json`: a scenario spec for the target agent's telemetry, stored
  with relative time offsets so it never ages out of Intake's retention window.
- `seed_intake.py`: expands the spec into spans and annotations, posts them to
  Intake, then reads them back.
- `reset_intake.py`: deletes the demo source's spans and annotations so the
  demo can start from a clean corpus.

## Prerequisites

- A running local platform with Intake, Agents, Jobs, and Compass, and a
  running ClickHouse for Intake (see `services/intake/README.md`).
- `nemo setup` completed, so a default/fast Model Entity pair exists.
  Compass resolves its models as Platform Model Entities. Each ref is a
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
  Dependency`; pick another model and set it as the run's `default_model` or
  `fast_model`.

The target agent (`demo-agent` below) does not need to exist as an Agent
entity — Compass only matches it against each span's normalized
`agent_name`.

## Demo Flow

### 1. Seed Intake

Seed Intake with telemetry for the target agent:

```bash
uv run plugins/nemo-compass/examples/execute-agent-job/seed_intake.py \
  --base-url http://localhost:8080 \
  --workspace default \
  --target-agent demo-agent
```

This posts to `POST .../ingest/spans` and `POST .../annotations`, then reads
both back.

The sample includes recurring failures across multiple sessions so Compass can
find patterns supported by at least three traces. It contains 84 spans across
28 sessions spanning about 3.6 hours:

| Scenario | Sessions | Failure |
|---|---|---|
| `retrieval` | 8 | `knowledge_search` returns zero documents; the agent answers anyway |
| `handoff` | 6 | `delegate_task` fires without the conversation summary |
| `billing` | 5 | `billing_lookup` times out at 30s |
| `healthy` | 9 | none — grounded, cited answers |

The sample also includes 12 negative and 6 positive `feedback` annotations and
18 numeric `helpfulness` labels. Annotations attach to sessions so Compass can
correlate them with spans.

Re-running is safe. Intake's span table sorts on `start_time`, so a repeat
post only updates a span in place when its timestamp is unchanged. The
script therefore reuses the origin of the corpus already in Intake rather
than re-anchoring to now, and skips annotations that already exist. Passing
a different `--started-at` over an existing corpus writes a second copy of
every span. Compass then fails with `trace ... contains duplicate span
id`.

To start over, for example after changing the spec or seeding with a
different `--started-at`, delete the demo corpus and reseed:

```bash
uv run plugins/nemo-compass/examples/execute-agent-job/reset_intake.py \
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

### 2. Run Compass

Create the workspace's Compass agent, unless it already has one:

```bash
nemo agents create compass.analyst
```

Submit an analysis of `demo-agent` and watch it finish:

```bash
nemo agents execute --agent compass-analyst \
  --spec '{"context": {"agent": "demo-agent"}}'
nemo jobs watch <job-name>
```

`execute` prints the created job; its `name` is the `<job-name>` used below.
`nemo jobs watch` streams the job's logs and exits non-zero if the job does
not complete.

The model pair is **required** in the context. It lives only in the
operator's local CLI config (`~/.config/nhx/config.yaml`), which the
Platform process cannot read, so the request has to carry it. The CLI fills
in `default_model` and `fast_model` from that config when the context leaves
them out. Set either one in the context to override it.

The same run through the typed Python client. Unlike the CLI, the client does
not read your CLI config for model defaults, so both models are passed
explicitly:

```python
from nemo_helix_plugin.client.client import NemoClient
from nemo_agents_plugin.sdk import AgentsResource

client = NemoClient.from_config()
agents = AgentsResource(client)
job = agents.jobs.execute.create(
    spec={
        "agent": "compass-analyst",
        "context": {
            "agent": "demo-agent",
            "default_model": "default/<default-model>",
            "fast_model": "default/<fast-model>",
        },
    },
    workspace="default",
)
print(job["name"])
```

`NemoClient.from_config()` connects to your active CLI context's base URL with
its credentials. Pass `context=` to use another CLI context.

### 3. Inspect the run

List the analyses of `demo-agent`, newest first:

```bash
nemo agents list-jobs --agent compass-analyst --filter '{"spec.subject": "demo-agent"}'
nemo jobs get-status <job-name>
```

### 4. Download the report

The `analysis-report` result, saved by the Compass post-run action, is the
durable record of what the run did:

```bash
nemo jobs results list <job-name>
nemo jobs results download analysis-report --job <job-name> -o analysis-report.txt
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

There is no CLI command for listing insights yet, so use the Python client. From the repo
root, save this as a file and run it with `uv run python <file>`:

```python
from nemo_helix_plugin.client.client import NemoClient
from nemo_compass_plugin.sdk import CompassPluginResource

client = NemoClient.from_config()
compass = CompassPluginResource(client)
page = compass.insights.list_insights(workspace="default", agent="demo-agent")
for insight in page.data:
    print(f"[{insight.status}] {insight.title} ({len(insight.evidence)} traces)")
```

`list_insights` also accepts `status` (`open`, `resolved`, or `deleted`),
`page`, `page_size`, and `sort`.
