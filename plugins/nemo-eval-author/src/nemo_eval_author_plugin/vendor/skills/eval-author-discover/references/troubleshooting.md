<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting discovery

Read when a discovery report names a failing or advisory check, or when choosing
the interpreter for a repository that holds NeMo Gym evals.

## Contents

- [Checks by phase](#checks-by-phase)
- [Harbor and NeMo Gym environments](#harbor-and-nemo-gym-environments)
- [What Judge and Solve run](#what-judge-and-solve-run)

## Checks by phase

Each check belongs to one phase. Fix failures in phase order, and within Harbor's
Judge checks work top to bottom: a later check's failure often disappears once
you fix an earlier one.

| Phase | Check | What it means and what to do |
|---|---|---|
| Probe | `harbor` | Harbor is not importable by this interpreter. Re-run with the interpreter from **Before you start** |
| Probe | `harbor-cli` | Advisory. No `harbor` executable exists on `PATH`, so Judge's `round-trip` check cannot run |
| Probe | `gym` | Advisory. Whether a `gym` command next to the interpreter or on `PATH` reports NeMo Gym 0.6.0 or later. Gym alone lets discovery explore, judge, and solve Gym manifests; Harbor tasks still need Harbor |
| Explore | `config` | Required only when the repository holds neither a config nor a task. Confirm the location if an existing suite is expected. If the user has no evals and asked to build them, follow `eval-author-first-eval`. Advisory when datasets exist without a config, since each runs by path |
| Explore | `config-parse` | A config file did not parse. Either PyYAML is missing, which means the wrong interpreter, or the file's YAML is broken. The hint says which |
| Explore | `ethos` | Advisory. Explore did not find a readable root `ETHOS.md`. This check records file readability, not substantive Ethos validity |
| Explore | `tasks-on-disk` | Advisory, and always unproven. A count of directories holding a `task.toml` |
| Explore | `gym-manifests` | Advisory. Gym `manifest.yaml` environments and benchmarks use Gym's resources-server format, which Harbor cannot read. They are listed in `gym_manifests`, judged by Gym, and sampled by Solve. The message counts manifests with no data downloaded |
| Judge (Harbor) | `schema` | Harbor rejected the config's shape. The message carries the offending field path |
| Judge (Harbor) | `resolution` | Harbor could not turn the config into a job. Usually a `datasets[].path` that does not exist. This fails before any container starts |
| Judge (Harbor) | `tasks` | Some resolved directories are not valid Harbor tasks. A task directory needs a parseable `task.toml` and an `environment/` directory, even when the image is prebuilt |
| Judge (Harbor) | `coverage` | Harbor silently dropped task directories that exist on disk. Harbor skips unparseable tasks without raising, so treat this as a real defect, not noise |
| Judge (Harbor) | `credentials` | Reports the host variables the suite needs. Confirm each one is set before running; a missing key surfaces as a failed trial, not a clear error |
| Judge (Harbor) | `agent` | The named built-in agent does not exist, or the `import_path` does not import. Check the message for which |
| Judge (Harbor) | `backend` | The environment backend failed preflight. For Docker, verify access as described in Step 4; the error alone does not establish that Docker is stopped |
| Judge (Harbor) | `round-trip` | The Harbor CLI rejected the config file's bytes. This is the weakest Judge check: it round-trips the schema only, so it can pass while `resolution` fails |
| Judge (Harbor) | `compatibility` | The installed Harbor does not expose the resolved task list, so `tasks`, `coverage`, and `credentials` cannot run. Install a Harbor version that exposes it |
| Judge (Harbor) | `dataset-tasks` | Some tasks in a dataset are not valid Harbor tasks. The message names them |
| Judge (Harbor) | `gym-conversion` | A Gym extension task could not be rendered as a Harbor task: both `tests/test.sh` and `tests/verifier.py`, a malformed `tasks.jsonl` row, or a template placeholder the row does not fill |
| Judge (Gym) | `gym-validate` | `gym env validate` rejected a manifest. The message carries Gym's error; the hint holds the command that reproduces it |
| Judge (Gym) | `gym-unavailable` | The repo holds Gym manifests, but NeMo Gym or its `gym` executable is missing, so none were checked. Each manifest's Judge status is `blocked`, and Judge is invalid until Gym is installed |
| Solve | `solve` | A sampled eval did not run end to end. The message names the task and the telling line of its error; the hint holds the command that reproduces it, and `solve.results[].error` holds the full error. As an advisory, it means Solve ran nothing, and says why |
| Solve | `solve-reward` | Advisory. The reference solution ran but earned no reward. The machinery works, but `solution/solve.sh` and the tests disagree |
| Solve | `gym-runner` | Advisory. A dataset holds Gym extension tasks. They pass Harbor's Judge checks once converted, but NeMo Gym has no runner for them yet, so the dataset gets no run command and Solve skips it |

## Harbor and NeMo Gym environments

Discovery never imports NeMo Gym. It runs the `gym` command next to the
interpreter or on `PATH`, and counts Gym as installed only when `gym --version`
reports 0.6.0 or later. Without Harbor, Judge and Solve cover Gym manifests with
`gym env validate` and `gym env test`, and Harbor findings stay unproven. Judging Gym extension tasks needs Harbor, not
Gym. Keep the two runtimes in separate environments, and install `nemo-gym`
0.6.0 (`pip install nemo-gym==0.6.0`), which needs Python 3.13.14 or newer. On
an older Python, installers silently pick an older version of Gym which lacks the
critical commands that Judge and Solve need to run. Only Gym's environment needs
Python 3.13.14; discovery itself runs on Python 3.11 or later with Gym's `bin/` on
`PATH`.

## What Judge and Solve run

Judge checks every config, dataset, and manifest. It skips a Gym manifest whose
data files are missing, such as an unprepared benchmark, and names the command
that prepares it.

Solve runs only datasets and manifests that passed Judge; a dataset with any task
Judge rejected sits out, and the rest still run. With none passing, Solve is
skipped. It picks at most 4 datasets or manifests, and
at most 4 tasks from each, spread across the list from first to last and the same
on every run; the report marks the rest `not sampled`. Harbor tasks need a
`solution/solve.sh` for the `oracle` agent to run. A task counts as run when it
finishes and the verifier writes a reward; a zero reward is only an advisory,
because Solve proves the eval machinery works, not that the reference solution is
right. Datasets of Gym extension tasks are skipped, because NeMo Gym has no runner
for them yet.

Solve runs containers and can take several minutes on the first run while Docker
builds task images. Harbor's job output goes to a temporary directory. Harbor is
never allowed to auto-grant host-environment access; a task that asks for it
fails with Harbor's message. `gym env test` builds the resources server's venv at
`resources_servers/<name>/.venv`, where Gym always puts it.
