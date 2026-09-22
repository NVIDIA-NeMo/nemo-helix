<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Canonical Experimentalist benchmarks

These benchmarks measure M2 Experimentalist optimization on unmodified Harbor Hub
packages. They do not cover M1 Insight-driven suite authoring.

Five suites ship today. All store only task IDs and download task definitions into
a local cache; no task content is vendored here.

| Suite | Package | Tasks | Agent under test |
| --- | --- | --- | --- |
| `suites/terminal-bench-2.1.yaml` (default) | `terminal-bench/terminal-bench-2-1@6` | 89 | `examples/terminal-bench-agent` |
| `suites/tau3-banking.yaml` | `sierra-research/tau3-bench@1`, banking scoped | 97 | `examples/tau3-nooa-agent` |
| `suites/tau3-airline.yaml` | `sierra-research/tau3-bench@1`, airline scoped | 50 | `examples/tau3-nooa-agent` |
| `suites/tau3-retail.yaml` | `sierra-research/tau3-bench@1`, retail scoped | 114 | `examples/tau3-nooa-agent` |
| `suites/tau3-telecom.yaml` | `sierra-research/tau3-bench@1`, telecom scoped | 114 | `examples/tau3-nooa-agent` |

The four tau3 suites share one agent: its `ETHOS.md` is domain-generic because
the domain policy arrives at runtime in the task instruction.

## Terminal-Bench provenance

The suite uses `terminal-bench/terminal-bench-2-1@6` from Harbor Hub:

- 89 canonical tasks
- dataset content hash
  `sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`
- Harbor Hub record:
  <https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6>

The 38/25/26 quality partition and 35/12/12 fast partition came from Gaia's
`optimization-datasets` commit
`1b7688ad257dacd1a3267dddb88db6cefdc31376`. The manifest corrects
`install-windows-3-11` to the canonical ID `install-windows-3.11`. At startup,
the runner asks Harbor Hub for revision 6, verifies its content hash, and
asserts that the quality partition covers all 89 canonical IDs exactly once.

## tau3 provenance

Each tau3 suite scopes one domain of `sierra-research/tau3-bench@1` by task-ID prefix,
so the four cover disjoint subsets of the package's 375 tasks. All share its content
hash `sha256:a57304f682894ac061090769af771a3617664f3ff6e5417d4eadf8e30433e4d9` and its
Harbor Hub record
<https://hub.harborframework.com/datasets/sierra-research/tau3-bench/1>:

| Suite | Task-ID prefix | Quality split | LLM judge scores it? |
| --- | --- | --- | --- |
| `tau3-banking.yaml` | `tau3-bench__tau3-banking_knowledge-` | 41/28/28 | quality only |
| `tau3-airline.yaml` | `tau3-bench__tau3-airline-` | 20/10/20 | no |
| `tau3-retail.yaml` | `tau3-bench__tau3-retail-` | 50/24/40 | yes |
| `tau3-telecom.yaml` | `tau3-bench__tau3-telecom-` | 50/24/40 | no |

Banking's partition came from `optimization-datasets` `feat/tau2-other-domains`
commit `025ecd2ef2b518ad81f6b22d3f3937af8906fb01`, whose
`tau2-banking-knowledge-NNN` names map onto the canonical
`tau3-bench__tau3-banking_knowledge-task-NNN` IDs. The other three domains ship an
upstream `split_tasks.json` whose train and test lists partition the domain exactly,
so `test` is that file's test list verbatim and only `validation` is carved out, at
train indices 2, 5, 8, and so on.

A canonical ID is `tau3-bench__` plus the name Harbor's `adapters/tau3-bench` gives
the task, `tau3-<domain>-<slugified upstream task ID>`. Running that adapter's naming
rule over a `tau2-bench` checkout reproduces the banking manifest exactly, and its
`adapter_metadata.json` records the same per-domain counts this directory claims
(`airline=50, retail=114, telecom=114, banking_knowledge=97`). The naming rule accounts
for every ID and that metadata for every count, but neither is the published package:
only `run.py --validate-only` compares a manifest against the Hub, and banking is the
one domain the adapter enumerates from a task directory rather than from
`split_tasks.json`, so reproducing it does not exercise that selection path.

Every fast partition is 6/3/3. Banking picks from a filtered subset; the other three
take `split[:: len(split) // n][:n]`, which spans each split instead of clustering at
its start — telecom's IDs sort by issue type, and its first twelve are all
`mms_issue`.

Whether scoring needs an LLM judge is per task, set by each task's `reward_basis`.
Banking's fast partition draws only from its 87 pure-database-state tasks, so smoke
runs skip the judge. Telecom scores on environment assertions and expected actions,
and airline on database state plus a substring check of the agent's messages, so
neither reaches the judge at all. Retail does: only its tasks 33 and 34 score on
database state alone.

Two caveats these domains inherit from the package. Airline's tasks all carry
`nl_assertions` that its `reward_basis` omits, and only 6 of the 50 assert
`communicate_info`, so 44 of them are scored on database state alone — a weaker bar
than upstream tau2 airline. And airline is small enough that its quality split leaves
the optimizer 20 train and 10 validation tasks, the least signal of any suite here.

Each task also runs a `tau3-runtime` sidecar hosting the tau2 environment and user
simulator. Tau-style suites set `models.user_simulator`, which makes the runner
export `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `TAU2_USER_MODEL`, and
`TAU2_NL_ASSERTIONS_MODEL` for the sidecar and the verifier.

## Held-out evaluation

The runner:

1. evaluates the unchanged baseline agent on the test split;
2. gives only train and validation IDs to `run_experimentalist`;
3. resolves the validation-selected winner from the Experimentalist run;
4. evaluates that winner on the same test IDs and number of attempts.

Test tasks are not supplied to the optimizer. Failed and missing trials count as
zero reward in the benchmark summary. Harness errors are reported separately.

`optimizer.evaluator.n_attempts` controls candidate evaluation attempts during
optimization. Top-level `test_attempts` controls final baseline and winner
attempts. These settings intentionally remain separate.

## Run

Required credentials:

```bash
export INFERENCE_API_KEY=...
```

This credential is for the agent under test. Run `nemo setup` separately to
register the provider and select the default and fast Platform Model Entities
used by the optimization agents. Set `INFERENCE_API_BASE` to the provider
endpoint used by the agent under test.

Validate provenance and task IDs without Docker or model calls:

```bash
uv run python benchmarks/run.py --validate-only
```

Run the bounded fast benchmark:

```bash
uv run python benchmarks/run.py \
  --config benchmarks/configs/terminal-bench-smoke.yaml
```

Run the reproducible quality benchmark:

```bash
uv run python benchmarks/run.py \
  --config benchmarks/configs/terminal-bench-quality.yaml
```

A non-default suite needs its own suite, config, and agent:

```bash
uv run python benchmarks/run.py \
  --suite benchmarks/suites/tau3-banking.yaml \
  --config benchmarks/configs/tau3-smoke.yaml \
  --agent examples/tau3-nooa-agent
```

The other tau3 domains change only `--suite`.

Both setup and agent execution require network access. Each AUT installs its own
dependencies from its committed `uv.lock` inside the task container and does not
need a Docker socket.

Pass the same `--output` directory to resume interrupted Harbor jobs. Use a new
output directory for an intentionally fresh run.

## Artifacts

By default, each run writes under
`artifacts/experimentalist-benchmarks/<UTC timestamp>/`:

- `summary.json`: dataset and partition revisions, agent digest, model names,
  complete config, task counts, expected/observed trials, rewards, harness
  errors, tokens, reported cost, and elapsed time;
- `heldout-results/heldout-baseline/`: baseline Harbor results;
- `heldout-results/heldout-winner/`: winner Harbor results;
- `optimizer/eval-and-optimize/`: Experimentalist candidates, analyses, and
  train/validation Harbor results.

Harbor result files remain the source of truth. `summary.json` aggregates over
the full expected test trial count. `cost_usd` is `null` when the selected model
endpoint does not report monetary cost.
