<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Workflow demos

Run both demos from the repository root through one CLI:

```bash
uv run python -m demo.workflows list
```

| Workflow | DAG | Task image |
| --- | --- | --- |
| `layers` | 19 nodes: prepare → 10 workers → 5 reducers → 2 validators → summarize | `docker.io/my-registry/nhx-api:local` |
| `montage` | detect-edges → build-montage → describe-images | `helix-opencv-demo:local` |

Every run is a continuous Job with the name you supply. Its Filesets are private
to that name, such as `my-images-input` and `my-images-montage`. The captured
Attempt spec records the workflow choice and Fileset IDs. Trigger and download
read those IDs from the current Attempt; there is no local run registry.

## Quick walkthrough with services already running

From this checkout's repository root, define a short command in both terminals:

```bash
export NHX_BASE_URL=http://127.0.0.1:8080
wf() { uv run python -m demo.workflows "$@"; }
wf list
```

After the image, launcher and Secret setup below, start montage in terminal 1:

```bash
wf create montage my-images --secret my-nvidia-key --watch
```

Press `o` for the graph or **Tab → a → f** for unified live logs. Wait until the
first pass finishes and the workflow is waiting for changes. In terminal 2:

```bash
wf trigger my-images --input demo/opencv/fixtures/birthday-cake.jpg
```

After the next pass finishes, download and inspect the updated montage:

```bash
wf download my-images
open demo/runs/my-images/montage/contact-sheet.png
jq . demo/runs/my-images/captions/descriptions.json
```

Repeat only captioning:

```bash
wf trigger my-images --step describe-images
```

Wait for captioning to finish, then stop the run:

```bash
wf cancel my-images
```

To run the model-free demo instead:

```bash
wf create layers my-dag --watch
```

In terminal 2, use `wf trigger my-dag` for a new pass, or
`wf trigger my-dag --step worker-03` for a selected-node rerun. Stop it with
`wf cancel my-dag`. Use fresh names when repeating the walkthrough.

## Prepare the local platform

Use the DAG worktree and keep your existing services running if they already use
this checkout. Docker must be running. The local `cpu/gpu` execution profile
selects Docker and does not request a GPU. Pass `--profile` to choose another
configured CPU Docker profile.

Build the montage image once before using that workflow:

```bash
docker build -t helix-opencv-demo:local demo/opencv
```

Build jobs-launcher for the architecture of the task images. They must share an
architecture for a single local launcher binary:

```bash
DEMO_IMAGE=helix-opencv-demo:local  # For layers, use docker.io/my-registry/nhx-api:local
DEMO_ARCH=$(docker image inspect "$DEMO_IMAGE" --format '{{.Architecture}}')
(cd services/core/jobs/jobs-launcher && bash build-manual.sh linux "$DEMO_ARCH")
```

If services are not running, start them in a separate terminal:

```bash
set -a; source packages/nhx_platform/config/local.env; set +a
uv run nemo services run --service-group all --controller-group all --host 127.0.0.1 --port 8080
```

In each command terminal, select the local platform. Log in once if needed:

```bash
export NHX_BASE_URL=http://127.0.0.1:8080
uv run nemo auth login --unsigned-token --email your.name@nvidia.com
```

Montage also needs an existing Secret containing an NVIDIA API key. The CLI
checks that Secret before creating its Filesets. Captioning makes NVIDIA model
requests; the layers workflow needs no model or Secret. If the Secret does not
exist yet, export `NVIDIA_API_KEY` in your shell and create it once:

```bash
printf '%s' "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY first}" | uv run nemo secrets create my-nvidia-key --workspace default --from-file -
```

Use an existing Secret's name instead if one is already configured.

## Create a named workflow

```bash
uv run python -m demo.workflows create layers my-dag
uv run python -m demo.workflows create montage my-images --secret my-nvidia-key
```

Creation seeds the layers input/configuration or the montage's tower photo, then
commits and submits the Job. It returns immediately. Add `--watch` to open the
TUI after submission, or use `--image` to override the workflow's task image.
Existing Job or Fileset names are rejected before creating new resources.
Creation is not transactional: if a request fails partway through, some named
Filesets may remain. Choose a fresh name or inspect them before retrying.

## Watch either workflow

```bash
uv run python -m demo.workflows watch my-dag
uv run python -m demo.workflows watch my-images
```

Watch reuses the installed `nemo jobs watch` implementation from this environment.
Press `o` for the drawn graph, `/` to find a node, and **Tab → a → f** for unified
live logs. The TUI keeps up to 5,000 records and reports discarded records. `q`
detaches without cancelling; it returns exit code 130 for a live workflow.
Reopen watch after creating a new Attempt, since each session pins its Attempt.

For unified line output in another terminal:

```bash
uv run python -m demo.workflows watch my-images --no-tui
```

## Trigger a pass or individual nodes

Commit the named run's input without changing its contents:

```bash
uv run python -m demo.workflows trigger my-dag
uv run python -m demo.workflows trigger my-images
```

Upload a new input and commit it in one command:

```bash
uv run python -m demo.workflows trigger my-images --input demo/opencv/fixtures/birthday-cake.jpg
uv run python -m demo.workflows trigger my-dag --input ./message.txt
```

For montage, the original filename is retained, so the cake is added alongside
the tower photo. Repeating a filename overwrites it. For layers, the file replaces
`message.txt`; use `--fileset config --input ./settings.txt` to replace settings.

Invalidate selected nodes without editing the spec or creating an Attempt:

```bash
uv run python -m demo.workflows trigger my-images --step describe-images
uv run python -m demo.workflows trigger my-dag --step worker-03 --step worker-04
```

Each command generates a request ID. Use `--request-id` to repeat a specific
signal; repeating the current signal does not queue an extra run. Successful
node reruns trigger their affected downstream chain, while clean nodes can reuse
prior results. Existing backend retries are unchanged.

Events received during a pass wait until it drains; multiple events can coalesce.
Input commits and invalidations can remain pending while the Attempt is paused.
Commit revisions track changes and do not freeze file contents. For the montage
walkthrough, wait for the first pass before uploading the second photo so each
pass sees the intended inputs. Trigger rejects terminal Attempts.

## Download results

```bash
uv run python -m demo.workflows download my-images
uv run python -m demo.workflows download my-dag
```

Outputs go under `demo/runs/<name>/`; pass `--output DIR` to choose another
location. These are current Fileset contents, not snapshots of an earlier pass.
After montage finishes:

```bash
open demo/runs/my-images/montage/contact-sheet.png
jq . demo/runs/my-images/captions/descriptions.json
```

Layers publishes `demo/runs/my-dag/output/report/report.txt` and its input manifest
under `output/prepared/`. Download after the pass completes so all producers have
finished their required publication.

## Stop a run

```bash
uv run python -m demo.workflows cancel my-dag
uv run python -m demo.workflows cancel my-images
```

Cancellation ends the Attempt and retains existing storage-policy behavior.
It leaves the named input/output Filesets and local downloads in place.
For a new Attempt of an existing Job, wait for `CANCELLED`, then use
`nemo jobs rerun NAME --workspace default`. Fileset commits and invalidation
request passes within the existing continuous Attempt.

All resource commands accept `--workspace`; it defaults to `default`. Run
`uv run python -m demo.workflows COMMAND --help` for the command's options.
