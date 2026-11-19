<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Continuous workflow demo

`prepare` fans out to ten workers. Five reducers each join a pair of workers;
two validators join the reducer groups; `summarize` waits for both validators.
The five layers contain 1 → 10 → 5 → 2 → 1 nodes (19 total).
Tasks print timestamped progress with their node names. Workers process six
simulated batches over 18–30 seconds and print progress every 3–5 seconds. All nineteen nodes use CPU containers.
The local `gpu` profile selects Docker; it does not request a GPU.

`job.json` is the saved job specification. `create.py` replaces its input Fileset
placeholder with the ID of a newly created Fileset, then submits the job.
Use a new `--name` for another demo; creation does not replace existing resources.
The image must already be available in the Docker daemon used by the controller.

Run these commands from `/Users/rsadler/wt/dag-workflows-rsadler`.

## Start services in terminal 1

Build the Linux launcher for the architecture of the local task image:

```bash
DEMO_ARCH=$(docker image inspect docker.io/my-registry/nhx-api:local --format '{{.Architecture}}')
(cd services/core/jobs/jobs-launcher && bash build-manual.sh linux "$DEMO_ARCH")
```

Load the local configuration and start the services and controllers:

```bash
set -a; source packages/nhx_platform/config/local.env; set +a
uv run nemo services run --service-group all --controller-group all --host 127.0.0.1 --port 8080
```

The services run from source. The supplied `nhx-api:local` image runs the demo's
tasks. Keep this terminal running. Docker must be running too.

## Create and watch in terminal 2

```bash
export NHX_BASE_URL=http://127.0.0.1:8080
uv run nemo auth login --unsigned-token --email your.name@nvidia.com
uv run python demo/continuous-workflow/create.py
uv run nemo jobs watch continuous-demo --workspace default --tui
```

The initial Fileset commit starts the first pass. After it finishes, the Attempt
stays active and waits for changes. Use `o` for the graph overview, `/` to find a
worker, and Tab for logs. Pressing `q` detaches without cancelling the workflow.

## Trigger another pass in terminal 3

```bash
export NHX_BASE_URL=http://127.0.0.1:8080
uv run nemo files filesets commit continuous-demo-input --workspace default
```

A commit changes the revision even when no files were uploaded. Commit while
workers are running to see another pass wait until the current pass drains.
Multiple commits before that next pass starts coalesce into one pass.
No nodes rerun automatically while the workflow is clean.

Stop the demo when finished:

```bash
uv run nemo jobs cancel continuous-demo --workspace default
```

Validate the saved specification without contacting services:

```bash
uv run python demo/continuous-workflow/create.py --validate-only
```

## Use the richer logs with an existing demo

Attempts keep their original task commands. Create a new demo to use an edited
`job.json`; committing the old input Fileset continues running the old commands.

```bash
uv run python demo/continuous-workflow/create.py --name continuous-demo-logs
uv run nemo jobs watch continuous-demo-logs --workspace default --tui
```

In another terminal, trigger another pass:

```bash
NHX_BASE_URL=http://127.0.0.1:8080 uv run nemo files filesets commit continuous-demo-logs-input --workspace default
```

Press Tab for Logs, `a` to combine collected executions, and `f` to follow.
The progress messages describe simulated work; tasks only echo and sleep.

## Run the five-layer demo with unified logs

Create a fresh Job so its immutable Attempt captures the current specification:

```bash
uv run python demo/continuous-workflow/create.py --name continuous-demo-layers
uv run nemo jobs watch continuous-demo-layers --workspace default --tui
```

Press Tab, then `a`, then `f`. The header reads `Logs: all executions · FOLLOW`.
This combines every node's collected logs and includes executions that start
later, including subsequent passes. Lines contain timestamps and execution IDs;
the demo messages also include logical node names. The TUI retains up to 5,000
records and reports dropped records.

For unified line output in a separate terminal:

```bash
uv run nemo jobs watch continuous-demo-layers --workspace default --no-tui
```

Trigger the next pass:

```bash
NHX_BASE_URL=http://127.0.0.1:8080 uv run nemo files filesets commit continuous-demo-layers-input --workspace default
```

Each reducer can start when its own two workers finish. Validators wait for
their respective reducer groups. Layer boundaries do not impose a global
barrier; the graph's dependencies determine when each node can start.

## If tasks run but watch has no logs

Check that `services/core/jobs/jobs-launcher/jobs-launcher` exists in the
services worktree. The local Docker backend copies this binary into each new
task container to forward stdout/stderr to the platform. Without it, tasks can
run directly and their output does not reach watch. Build it using the command
above, then commit the input Fileset again. No service restart is needed;
already-started containers keep their original entrypoint.

See [the verified local investigation](logging-investigation.md).
