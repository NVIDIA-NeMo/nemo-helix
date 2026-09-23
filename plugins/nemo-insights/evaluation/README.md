<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Insights analyst evaluation (maintainer tooling)

Evaluates the Insights analyst against registered telemetry **subjects** and
emits Insights for comparison. This maintainer tooling drives
`nemo agents analyst`; it is not the product CLI and is not shipped in the
wheel.

## Prerequisites

- Run every command from `plugins/nemo-insights`; the `evaluation` package is
  not installed in the plugin wheel.
- Install AWS CLI 2.33.0 or newer and configure the CSS S3 credentials described in
  [State bundles](#state-bundles).
- Start a local NeMo Helix at `http://localhost:8080` before using pinned
  analysis, restore, roundtrip, or guarded publish commands.

## Quickstart

```bash
cd plugins/nemo-insights
uv run python -m evaluation analyze tau2-airline         # reproducible default: restore the pinned state locally, then analyze
uv run python -m evaluation analyze all                  # refresh every pinned benchmark/intake baseline transactionally
uv run python -m evaluation list
uv run python -m evaluation doctor                       # fresh clone? run this first
uv run python -m evaluation run tau2-airline             # produce: tau2 -> ingest -> record the run (expensive, once)
uv run python -m evaluation analyze tau2-airline --live  # analyze the recorded run's live traces (no restore)
uv run python -m evaluation analyze glamr --live         # intake: analyze existing live traces
uv run python -m evaluation snapshot tau2-airline        # export the subject's workspaces (read API) into a portable bundle
uv run python -m evaluation restore --state state-v10    # re-ingest a state bundle into fixture workspaces (additive, idempotent)
uv run python -m evaluation restore --state state-vN --into WORKSPACE
```

Bare `analyze <subject>` is a fully reproducible run: pinned data (the subject's
`state` entry in `evaluations.toml`) restored onto the local platform, analyzed with fresh
insights (no prior seed), and atomically copied to
`evaluation/insights/<subject>.yaml` for review and check-in. The per-subject
manifest update preserves every other subject.

`analyze all` validates every benchmark/intake pin before starting, runs the
subjects in sorted order with child `--no-baseline-update`, and stages the complete
YAML set plus manifest in a sibling directory. It promotes that directory with
a backup/swap only after every child wrote output. A child failure, missing
output, manifest failure, or failed swap leaves the old checked-in directory
unchanged; a successful swap removes stale YAMLs.

Every deviation is one explicit flag:

- `--state <state-vN|FILE>` — another published state, or a local bundle file
  (mutually exclusive with `--live`).
- `--live [--since S]` — skip the restore and read the platform's live traces
  (`--since` applies to `analyze --live` and to `snapshot`; pinned/`--state`
  analysis derives its bound from the bundle manifest).
- `--update-insights` — run against the existing local insights (prod-like update flow:
  updates them and adds new ones); default is a fresh start with priors moved to backup.
  Valid in every mode.
- `--no-baseline-update` — leave generated YAML only in `evaluation/tmp/`. On
  `analyze all`, every child still runs and is validated, but the final
  promotion is skipped.
- `--base URL` — the one platform flag, on every platform-touching command.
  Fixture targets (restore, roundtrip, pinned/`--state` analyze) default to
  `http://localhost:8080`; live targets (`run`, `analyze --live`, snapshot's
  source) default to the stanza's `base_url`. publish's guard runs wherever
  its `--base` points — no default; `--no-verify` skips it out loud.
- `--set KEY=VALUE` — one-off config override (`run`/`analyze` only,
  repeatable). Values take the stanza key's type — bool keys accept only
  `true`/`false`, so `--set include_rewards=false` is really false; keys new
  to the stanza stay strings. If you keep reaching for it, move the value
  into `evaluations.toml`. `--set` applies after `--base`, so `--set base_url=…` wins when both are given.

Each `evaluation/insights/manifest.yaml` snapshot records:

- `state` — the exact subject pin or explicit/live source label.
- `analyst_sha256` — all Python source under
  `plugins/nemo-insights/src/nemo_insights_plugin`, plus the canonical resolved
  dependency closure rooted at `nemo-insights-plugin` in the root `uv.lock`.
- `insights_sha256` — the checked-in YAML bytes after the SPDX header is added.

`run` produces traces and records the run to `evaluation/tmp/<subject>.run.json`;
`analyze --live` then analyzes it — for a `benchmark` it re-uses the last recorded
run (no tau2 re-run), for an `intake` subject it analyzes the configured agent.
Iterate on insight generation by re-running `analyze` as often as you like; `run`
again only when you want fresh traces. (A benchmark `analyze --live` follows the
base_url recorded at `run` time, so set `--base` on `run`.)

## Breaking changes (2026-07-07)

- **Deleted flags:** `--pinned` (it's the default), `--latest` (storage-latest
  is cross-subject and unsafe under per-subject pins), `--ref` and its `""`
  sentinel, analyze's `--from` (folded into `--state FILE`), `--local`
  everywhere (localhost is the default wherever a fixture is the target;
  `--base` overrides).
- **Fresh by default:** the old seed flag (`none|keep`) is gone — the seed flag became
  `--update-insights` (the not-fresh flow), and bundles no longer carry insight
  YAMLs at all (pure data fixtures; restore seeds run records only).
- **The `insights` command alias is gone** — one name: `analyze`.

## State bundles

Immutable per-subject fixtures live in NVIDIA Core Storage Service (CSS), using
its S3-compatible API. The committed location is:

```toml
state_s3_endpoint = "https://pdx.s8k.io"
state_s3_region = "us-east-1"
state_s3_bucket = "insights-analyst-evaluation-dataset"
```

Each subject's `state = "state-vN"` key in `evaluations.toml` pins the exact object
that bare `analyze <subject>` downloads. The corresponding object key is
`state-vN.tar.zst`; a missing or malformed subject pin fails instead of falling
back to the latest object.

Install the AWS CLI and put the CSS S3 credentials in the gitignored
`evaluation/.env`:

```dotenv
INSIGHTS_EVALUATION_S3_ACCESS_KEY=team-aire-ase
INSIGHTS_EVALUATION_S3_SECRET_KEY=<S3 Secret from CSS Portal → Auth Info>
```

The S3 Secret is not the namespace password. `doctor` checks the AWS CLI and
both variables without printing their values.

Which file do I touch?

| Surface            | Owns                                                        |
| ------------------ | ----------------------------------------------------------- |
| `evaluations.toml` | subject definitions, state pins, and non-secret CSS location |
| flags              | this invocation only                                        |
| `evaluation/.env`  | CSS, inference, and live-source credentials                  |

**Analyze against the pinned fixture (the everyday loop, and the default):**

```bash
uv run python -m evaluation analyze tau2-airline
```

Downloads the subject's pinned state (its `state` key in `evaluations.toml`;
a missing key is a hard error — no latest fallback),
re-ingests it into fixture workspaces (`tau2-airline-state-v6`) on the local
platform, and runs the Analyst live. Re-runs skip the ingest (idempotent).
`--state state-vN` / `--state FILE` select another state; `--base URL`
retargets the restore; `--update-insights` seeds the analyst with your local
prior insights (default: fresh — the prior file is moved aside first). `since`
is derived from the bundle's manifest, so old spans can't hide behind the read
API's 30-day default lookback. (`--live` skips the restore entirely and
analyzes the platform's live traces, with `since` from `--since`, the stanza,
or a 30d default — in that order; the effective bound is always printed.)

**Publish a verified candidate from a maintainer machine:**

```bash
uv run python -m evaluation snapshot glamr -o evaluation/tmp/glamr.tar.zst
uv run python -m evaluation publish evaluation/tmp/glamr.tar.zst --base http://localhost:8080 --reason "why this exists"
```

`snapshot` drains the subject's workspaces (benchmark subjects: realistic +
`-oracle` twin) into JSONL + manifest — no ClickHouse, no Docker. An intake
subject can set `experiment = "<name>"` to capture only that Experiment's
complete traces; membership comes from the Experiment's evaluations and their
traces, then every span in each trace is exported by trace ID. `publish`
refuses to mint unverified: `--base` runs the round-trip fidelity guard there
first (re-ingest into scratch workspaces → re-export → doc diff), or pass
`--no-verify` only after running `roundtrip` separately and confirming it
passed.
Then pin it by setting `state = "state-vN"` in the `[glamr]` stanza in
`evaluations.toml`.

**Restore without analyzing:** `uv run python -m evaluation restore (FILE | --state state-v10) [--base URL]`.
To restore a one-workspace bundle directly into a named workspace, use
`uv run python -m evaluation restore --state state-vN --into WORKSPACE`. `--into`
accepts only one-workspace bundles and requires a fresh, empty target
workspace. The default restore remains fixture-scoped and idempotent.

What restore touches:

- **Platform, default fixture restore:** additive, idempotent, and healing.
  Ingests into `<ws>-<ref>` (`<ws>-<sha256[:8]>` for local files).
  Per-collection guard: counts match → skip, empty → ingest, supported
  interrupted states → heal, anything else → hard error.
- **Platform, direct `--into` restore:** writes to the exact named workspace
  only after proving all three collections are empty. It is fresh-target-only,
  and rerunning into the now-populated workspace fails rather than acting
  idempotently.
- **Local `evaluation/tmp`: run records seeded.** The bundle's run records
  replace yours — clobbered files are moved to `evaluation/tmp/backup-<timestamp>/`
  first, and only the bundle's own subjects' files are ever touched. Bundles
  carry no insight YAMLs (fresh vs not-fresh is purely local state; see
  `--update-insights`).
- Accepted losses: annotation/evaluator-result `created_at`/`created_by` are
  server-stamped at restore (the write APIs reject client values); a running
  platform is required to analyze.
- Non-OTLP spans, including ATIF, restore through the provider-neutral direct
  ingest API so their source names and arbitrary span IDs remain unchanged.
- Legacy `state-v1..v5` tars are not present in CSS (the v4 corpus lives on as
  `state-v6`); a stray local copy restores only from a checkout
  predating the v6 migration.
- Client-side re-ingest is a stopgap for the platform team's RBAC-scoped
  server-side export/import endpoint; when that ships, snapshot/restore
  collapse to two API calls each.

Each benchmark `run` reuses **two stable workspaces** per subject — `<workspace>`
(the realistic, oracle-free workspace the Analyst evaluates, blind) and
`<workspace>-oracle` (the answer key + scores, for the UI). The stanza's `workspace`
is that base name. Runs no longer mint a workspace each. Run isolation comes from
the per-span `nemo.evaluation.name=<run-id>` tag plus the Analyst's `evaluation_id`
filter (which AND-pins every span read to that run) — that is what scopes the
analysis. The matching **Experiment** entity registered on the `-oracle` workspace
is metadata for the UI (run-picker + leaderboard), not the scoping mechanism.
For standard Platform subjects, every `analyze` also creates a separate
Evaluation under the `nemo-analyst` Experiment and associates the Analyst's own
OTLP trace with that Evaluation and the subject-derived test case. The source
Evaluation ID scopes what the Analyst reads; the Analyst Evaluation ID tags what
the Analyst emits. Workspaces therefore stop accumulating while each source run
still reads only its own traces. (Old spans age out by Intake retention;
Experiment and Evaluation entities are soft-deletable.)

Keys come from `evaluation/.env` (see below), so the commands need no inline env. **`doctor`**
prints a per-subject readiness checklist — on a fresh clone, run it first and it tells you
exactly what to install/set (`✓ ready` or `✗ needs: …`).

Subjects live in `evaluations.toml` — one table per subject, keyed by `type`:
- `type = "intake"` — analyze an agent's existing Intake traces (config: `agent`, `workspace`, `base_url`, optional `since`).
- `type = "benchmark"` — run a benchmark to produce traces, ingest them into Intake, then analyze (config: `domain`, `base_url`, `workspace`, `agent_llm`, `user_llm`, `task_split_name`, `num_trials`, `max_concurrency`, `seed`, optional `num_tasks`/`timeout`/`include_rewards`).

`--since` (analyze `--live`, snapshot) accepts `Nd`/`Nh`/`Nm` (days/hours/minutes)
or an ISO date; `--since ''` means no lower bound (the epoch). Insights are
written to `evaluation/tmp/insights_<name>.yaml`.

## Config split: secrets in `.env`, everything else in `evaluations.toml`

On startup the CLI auto-loads `evaluation/.env` (gitignored) as `KEY=VALUE` lines. Keep
**only secrets** there — `INSIGHTS_EVALUATION_S3_ACCESS_KEY` and
`INSIGHTS_EVALUATION_S3_SECRET_KEY` for fixture storage, plus
`OPENAI_API_KEY`/`OPENAI_API_BASE` for the benchmark simulator LLMs. Run
`nemo setup` against the analysis Platform to
select the Analyst's default and fast Model Entities; its provider credential
stays in Platform Secrets.
GLAMR live analysis additionally reads `GLAMR_INTAKE_USER` and
`GLAMR_INTAKE_PASSWORD` from `.env`; `evaluations.toml` stores only those
environment-variable names, never their credential values.
Real shell environment variables override the file. Everything non-secret (paths, models,
ports, run sizes) lives in the subject's `evaluations.toml` stanza.

## Benchmark prereqs (tau2-airline / tau2-retail / tau2-telecom)

Clone tau2-bench as a sibling of this repo and install it once:

```bash
git clone https://github.com/sierra-research/tau2-bench   # sibling of nemo-insights plugin
cd tau2-bench && uv sync          # Python 3.12+; installs the `tau2` CLI into .venv
uv run tau2 check-data            # verify the shipped domain data
```

The `[tau2-airline]`, `[tau2-retail]`, and `[tau2-telecom]` stanzas then need
(all non-secret, committed):
- `tau2_repo` — the checkout above; relative to this repo's root (`../tau2-bench`, the
  sibling default) or absolute. Both the CLI (`<repo>/.venv/bin/tau2`) and the data dir
  (`<repo>/data`) are derived from it (`tau2_bin`/`tau2_data_dir` override if needed).
- `agent_llm`/`user_llm` — models your proxy key serves (`GET {OPENAI_API_BASE}/v1/models`);
  default `openai/nvidia/nvidia/nemotron-3-super-v3`.
- `base_url` — a reachable NeMo Helix (default `http://localhost:8080`).

With `evaluation/.env` holding the required credentials, run:

```bash
uv run python -m evaluation run tau2-airline
uv run python -m evaluation analyze tau2-airline --live

uv run python -m evaluation run tau2-retail
uv run python -m evaluation analyze tau2-retail --live

uv run python -m evaluation run tau2-telecom
uv run python -m evaluation analyze tau2-telecom --live
```

## State model

Bundles are immutable `state-v<N>.tar.zst` objects. Each contains
`export/<workspace>/*.jsonl` for spans, annotations, and evaluator results;
`tmp/` for run records; and `manifest.json` for counts, time bounds, source
URL, and platform lineage. Insights never travel in bundles.

`publish` runs the round-trip fidelity guard unless `--no-verify` is explicit,
finds the highest existing CSS object version, uploads the next version, and
stores the reason, publisher, and SHA-256 digest as S3 object metadata. After
publishing, update the relevant subject's `state` key in `evaluations.toml`.

Restores are additive re-ingests into `<workspace>-<state-ref>` fixture
workspaces, so a bundle does not modify the source workspace. Override the
committed pin for one run with `analyze <subject> --state state-vN`.

## Next steps

- Run `uv run python -m evaluation doctor` to check the configured subjects.
- Edit [`evaluations.toml`](evaluations.toml) to add a subject or update a state pin.
- See the [NeMo Insights README](../README.md) for the product Analyst CLI and plugin architecture.
