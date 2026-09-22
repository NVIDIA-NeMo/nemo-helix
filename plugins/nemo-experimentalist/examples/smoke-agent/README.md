<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# smoke-agent

A fast fixture for exercising the Experimentalist loop end to end. It exists to
make a full round cheap enough to run while refactoring, and to let a run be
checked for more than "it completed".

The agent is **wrong on purpose**. Five known weaknesses, each paired with a
group of Harbor tasks that surfaces it, so a run can be asserted to have
*repaired* something rather than merely completed.

> **Nothing inside `agent/` or `ETHOS.md` may describe what this fixture
> measures.** `agent_source` points at `agent/`, so it is copied into every
> candidate workspace and read by the Coder, and the Ethos reaches the LLM
> components by a separate route. A description in either would hand the Coder
> the diagnosis the fixture exists to test. Everything else here — this file,
> `configs/`, `scripts/`, `dataset/` — is never copied and can say whatever is
> useful, which is why the weakness detail below lives in this file.
>
> **Do not repair the five weaknesses in the agent.** A well-meaning cleanup
> silently destroys what the fixture measures: with the weakness gone, the
> baseline passes, the Analyst gets no failing trace, and a run that does nothing
> looks identical to a run that works.

## Prerequisites

- A local NeMo Helix with default and fast Model Entities selected by `nemo setup`
- Docker Engine and Docker Sandboxes (`sbx`)
- `uv`

## Design

- **The agent makes no model calls.** Handlers are regular expressions plus a
  dict lookup over `dataset/_shared/records.json`, so the same instruction always
  produces the same answer and the only stochastic component in a run is the
  Experimentalist itself. The `CompletionClient` in `agent.py` points at an
  unroutable address: an accidental model call fails loudly rather than quietly
  making the agent nondeterministic.
- **Task definitions are local and checked in.** No registry, no NeMo Helix
  for Mode 2, no network inside the task container.
- **One prebuilt image serves every task**, referenced by
  `[environment].docker_image` rather than a per-task Dockerfile. Its tag is a
  content hash of the Dockerfile and the records file, so forgetting to rebuild
  fails a test instead of silently running against stale data.

## Layout

```text
agent/                         ONLY this is copied to the Coder (agent_source)
agent/agent.py                 the code under optimization
agent/main.py                  container entry point
agent/harbor_wrapper.py        Harbor upload + exec adapter
ETHOS.md                       behaviour contract read by the LLM components
optimizer.yaml                 profile: agent source, Ethos, g1 datasets
optimizer-full.yaml            profile: the generated combined datasets
optimizer-generalization.yaml  profile: same agent, g4 datasets (see Scenarios)
configs/short.yaml             loop settings shared by the per-group gate checks
configs/full.yaml              loop settings for the multi-round scenario
dataset/_shared/               canonical Dockerfile, records, verifier
dataset/tasks.json             authored task values used to render every curated task
dataset/task-template/         one Harbor task shape, also used by insight mode
dataset/groups/                GENERATED Harbor task sets, gitignored
dataset/insights/              insight mode only, frozen analyst output
scripts/render_tasks.py        render every curated task from the template
scripts/build_image.py         build the image and render/stamp every task
scripts/build_all_group.py     assemble the combined group the full scenario runs
scripts/record_traces.py       evaluate a group and ingest its traces
```

## Running it

Run the loop **inside a Docker sandbox**. Clone mode gives it a private writable
clone rather than write access to the host checkout, which keeps a run from
touching this directory.

```bash
repo="$(git rev-parse --show-toplevel)"

sbx create --clone --name nemo-experimentalist shell "$repo"
```

> **In a git worktree?** `--clone` refuses to run there, so bind-mount instead
> with `sbx create --profile developer --name nemo-experimentalist shell "$(pwd)"`.
> That gives the sandbox write access to this checkout, so the source-mutation
> protection clone mode provides is gone. Check `git status` after a run.

```bash
# The image lives in the sandbox's own Docker daemon, so build it there.
sbx exec --workdir "$repo" nemo-experimentalist bash -lc \
  'cd plugins/nemo-experimentalist/examples/smoke-agent && uv run --no-project scripts/build_image.py'
```

`build_image.py` first renders `dataset/groups/` from `dataset/tasks.json` and
`dataset/task-template/`. Those directories are gitignored build output: edits
there are discarded on the next build, so change the manifest or the template
instead.

Then run a scenario. `--with ./plugins/nemo-agents` is required: the `agents`
command group lives in a separate workspace package, and without it the CLI fails
with `No module named 'nemo_agents_plugin'`.

```bash
sbx exec --workdir "$repo" \
  --env UV_PROJECT_ENVIRONMENT=/home/agent/.venvs/nemo-helix \
  nemo-experimentalist \
  bash -lc 'uv run --frozen --python 3.13 \
    --package nemo-experimentalist-plugin --with ./plugins/nemo-agents \
    nemo agents experimentalist run \
      --profile plugins/nemo-experimentalist/examples/smoke-agent/optimizer.yaml \
      --no-insight \
      --base-url http://host.docker.internal:8080 \
      --config plugins/nemo-experimentalist/examples/smoke-agent/configs/short.yaml \
      --experiment-dir /tmp/smoke-repair'
```

Copy the experiment directory back out with `sbx cp` to check the result.

**Models come from the platform, not from this example.** The Experimentalist
reads a *Model Entity* pair from the active CLI context — `default_model` and
`fast_model`, the second falling back to the first when unset — so `nemo setup`
is what configures them. The value is an entity id of the form
`<workspace>/<name>`, not a Litellm routing string; list what your Platform has
with `nemo models list --all-pages`. Preflight only checks that a value is *set*,
so a typo passes `doctor` and fails at the first LLM call, minutes into a run.

**`--base-url` points at a platform, and the sandbox is not the host.** A
container's `localhost` is itself, so a platform on your machine is reached at
`host.docker.internal:8080`. Without a reachable platform the run still works,
but every projection onto native `ExperimentGroup`/`Experiment` entities fails
and logs `[MIRROR] projection failed`. That is best-effort and never fails a run,
so a log full of it means the platform was unreachable, not that anything went
wrong.

### Run the loop tests

The loop tests are developer-invoked and execute model-written shell inside the
named sandbox. Create the `nemo-experimentalist` sandbox with the command above
first, then:

```bash
SANDBOX_VM_ID=nemo-experimentalist uv run --frozen pytest \
  plugins/nemo-experimentalist/tests/experimentalist/test_smoke_agent_mode_1_loop_e2e.py \
  plugins/nemo-experimentalist/tests/experimentalist/test_smoke_agent_mode_2_loop_e2e.py \
  -m e2e -n 4 --dist loadgroup
```

Pytest writes logs and downloaded artifacts under its temporary test directory.
It does not retry failed runs.

## Scenarios

**The profile picks the scenario, not the config.** What separates a repair run
from a generalization one is the split it runs against, so each has its own
profile:

| Profile | Config | Rounds | A healthy run ends with |
| --- | --- | --- | --- |
| `optimizer.yaml` | `short.yaml` | 2 | the winner beating the baseline |
| `optimizer-generalization.yaml` | `short.yaml` | 2 | the baseline correctly retained |
| `optimizer-full.yaml` | `full.yaml` | up to 5 | every task in the combined group passing |

The first two are opposite tests, so a run is only meaningful once you know which
one you started — and the config cannot tell you, because both use the same one.
That is not an oversight: the scenario config carries loop settings only, and the
schema has no dataset field at all.

`full.yaml` is the only one that exercises the evolutionary machinery — survivors
carried between rounds, ranking over more than two candidates, and the
convergence check. It runs against `dataset/groups/_all`, which is **generated
and gitignored**; build it first, or the run stops on the unbuilt dataset before
it evaluates anything:

```bash
sbx exec --workdir "$repo" nemo-experimentalist bash -lc \
  'cd plugins/nemo-experimentalist/examples/smoke-agent && uv run --no-project scripts/build_all_group.py'
```

Rerun that after changing any group.

## Groups

Every group is a self-contained train/validation pair of six tasks — per split,
two that fail at baseline and one that already passes. That control is the point:
it makes a destructive fix cost reward instead of passing unnoticed, and it means
a group's score can fall as well as rise. Every group except `g4-dispatch-order`
also carries an `insight-evidence` split of five tasks, which is what an
Insight-driven (mode 1) run and the Analyst read.

| Group | What a run against it tests | Backs | In `_all` |
| --- | --- | --- | --- |
| `g1-aggregation` | **Repair.** A capability that is absent rather than wrong. | `short.yaml` | yes |
| `g2-name-patterns` | Widening a pattern that is too narrow. | `full.yaml` | yes |
| `g3-long-inputs` | A constant rather than logic — the one `edit_config` in the set, so a run exercises a different path through the Coder. | `full.yaml` | yes |
| `g4-dispatch-order` | **Generalization.** The tempting fix passes train and fails validation, so a healthy run *keeps the baseline*. | `short.yaml` | no |
| `g5-edge-cases` | Several changes that score only when all are made, whose partial states are indistinguishable in the output. The hardest here. | `short.yaml` | no |

The groups are not interchangeable. Several independently addressable groups let
a single round produce genuinely *different* candidates, and let a later round
inherit one fix and add another; they differ in the kind of edit they call for,
so a run exercises more than one path through the Coder; and they differ in
difficulty, so an early round has something to find while the harder ones keep
the ceiling out of reach of a shallow fix.

Two are held out of the combined set, for different reasons.
`g4-dispatch-order`'s healthy outcome — baseline retained — is the opposite of
the combined scenario's, and one run cannot assert both. `g5-edge-cases` is only
reachable with trajectory scoring on, and the combined scenario runs with it off:
measured over runs made after the Ethos stated the sentinel, no candidate closed
it without a goal tree and most did with one. Trajectory scoring is not
dependable enough to leave on yet, so the group is out until it is.
`build_all_group.py` records the numbers and the condition for putting it back.
Run it on its own with `short.yaml` in the meantime — it is a repair-shaped
split.

## The five weaknesses

Tasks are authored in `dataset/tasks.json`; the agent is `agent/agent.py`.

### G1 — no aggregation capability

**What.** Nothing sums or averages a numeric field. `solve` dispatches over
`handle_lookup`, `handle_list`, `handle_count`; none matches "what is the total
`<field>` in the …", so every aggregation question falls through to `FALLBACK`.

**Odd one out.** This is the only *missing* capability; G2–G5 are flawed code
paths that exist. There is no wrong code to find, only absent code, so G1's tasks
exercise the Proposer more than the Coder.

**Tasks.** An aggregation question scopes its sum either by department or by
role, and **both splits carry both scopes**. Train sums by department
(`total=29`, research) and by role (`total=20`, engineer); validation sums by
role (`total=9`, analyst) and by department (`total=13`, ops). A fix that
hardcodes either scope therefore fails inside its own split rather than surviving
to validation: the pressure to read the scope from the question is applied
immediately, not held out.

**A correct repair** adds a handler that sums a field over a record subset, with
the scope taken from the question rather than assumed, and inserts it into the
dispatch tuple.

### G2 — name pattern too narrow

**What.** `LOOKUP_RE`'s name group is `([A-Za-z ]+)`, so any name carrying an
apostrophe, a hyphen, or a non-ASCII character fails to match at all and falls
through to `FALLBACK`.

**Tasks.** `O'Brien`, `Zoë Washington`, `Ann-Marie Cruz`. Controls are
plain-ASCII lookups that already work.

**A correct repair** widens the character class. Note `str.isalpha()` is *not* a
valid test for "would this name match" — it is Unicode-aware and accepts `Zoë`
while the agent's ASCII class rejects it. The guard test uses the agent's own
class for exactly this reason.

### G3 — instruction clipped before dispatch

**What.** `solve` truncates to `MAX_INSTRUCTION_CHARS = 240` before dispatching,
so a question preceded by a long preamble is cut off and matches nothing.

**Tasks.** A ~320-character reporting-policy preamble in front of a question that
works without it. `preamble-long-dept` doubles the preamble to roughly 650
characters, so a repair that only nudges the limit upward still fails it.

**The control is load-bearing.** `trailing-prose` puts the question *first* and
the prose after, so it passes at baseline **and** would break under a fix that
reads only the tail of a clipped instruction. It catches a specific bad repair.

**A correct repair** raises or removes the limit.

### G4 — dispatch order shadows the count handler

**What.** `LIST_RE` is `(?:list|how many) .*? in the (\w+) department` — the
`how many` alternative belongs to `COUNT_RE` — and `solve` consults
`handle_list` first. Counting questions are therefore answered with a list of
names.

**Odd one out.** This is the only group whose failure is a *wrong-shaped answer*
rather than the fallback, so it scores `reward 0` with `shape_ok 1.0`. G1, G2,
G3 and G5's missing-record mode all fall back to prose and score `shape_ok 0`.
Measured: G4 train aggregates to `reward 0.333, shape_ok 1.0`, G5 validation to
`0.333, 0.333`. The two groups are separable from aggregates alone.

**Two halves, one repair.** `LIST_RE` and the dispatch tuple order are a matched
pair. Changing either alone leaves the shadowing in place.

**Tasks.** Train counts people per department; validation counts *by role
within* a department, which matches `LIST_RE` but **not** `COUNT_RE`. So
reordering the tuple alone still fails validation; the count pattern has to widen
too.

### G5 — missing and empty data not handled

**What.** Two distinct failure modes:

- `handle_lookup` resolves a record with `next(r for r in … if r["name"] == name)`,
  which **raises** when the name is absent. `solve`'s top-level `except` catches
  it and returns `FALLBACK`.
- An empty stored value yields an empty right-hand side (`role=`) rather than a
  documented `role=unknown`.

**Why the catch exists.** Without it the process exits non-zero, the wrapper
raises, and Harbor records a *harness error* rather than a scored 0 — which
leaves the Analyst nothing to read. Verified: G5 trials complete with
`status=completed` and reward 0. The catch controls the exit code, not the trace;
NOOA already records the exception on the failing handler's span.

**Tasks.** Names absent from the records, and Karl Jung's empty `role`. Expected
answers use `unknown`.

## Orthogonality, including the data

Each group's tasks must supply evidence for **its own weakness only**, so a run's
failing traces point at one root cause. That extends to
`dataset/_shared/records.json`, which all five groups share — the data is a
coupling surface as much as the code is.

Current assignment, pinned by `test_smoke_agent_baseline.py`:

| Record | Serves | Must stay |
| --- | --- | --- |
| Ada Lovelace, Grace Hopper | controls, G1 engineer sum | plain ASCII, non-empty, int hours |
| Zoë Washington, O'Brien, Ann-Marie Cruz | G2 | non-empty `role`, int `hours` |
| Karl Jung | G5 empty-field mode | empty `role`, **int `hours`** |

Karl Jung's `hours` is `0`, not empty, and that is deliberate: an empty `hours`
sits in `ops`, which G1 aggregates, so it would force a G1 fix to absorb G5's
robustness.

This has already gone wrong once through a *task* rather than the data. G3's
`preamble-long-dept` originally looked up a name absent from the records, so
closing G3 alone would have left it failing on G5's missing-record path. Check
both when adding either.

## Checking a run

The Mode 1 and Mode 2 E2E tests check the experiment artifacts themselves. They
verify source changed, held-out tasks pass, controls do not regress, and the
Analyst named each group's problem. They also verify that G4 keeps the baseline
when a train-only change fails to generalize.

**Also run the guard suite after every loop run:**

```bash
uv run pytest plugins/nemo-experimentalist/tests/experimentalist/ -k smoke -q
```

`test_smoke_agent_baseline.py` and `test_smoke_agent.py` pin the agent behavior
and the records table above. A failure almost always means the fixture itself
changed, which makes every later run meaningless while still looking healthy. Do
not "fix" the agent to make it pass; confirm against the weakness descriptions
above first.

`test_smoke_agent_assets.py` pins the other half — the image tag, the task
template, and the tree rendered from the manifest — so run it after changing any
of those, not only after a loop.

What is **not** pinned: no test asserts which task sits in which split. The split
assignments described above can drift without a test failing, so check them
against `dataset/tasks.json` whenever you edit the manifest.

## Timings

Measured on 2026-08-05, one round, two candidates, three tasks per split:

| | |
| --- | --- |
| One split evaluated in containers | ~15 s |
| A full round end to end | ~18 min |

Container evaluation is negligible by design. The cost is the Experimentalist's
own components — the architecture doc, Analyst, Proposer, and one Coder pass per
candidate — which is the part under test and cannot be optimized away here.