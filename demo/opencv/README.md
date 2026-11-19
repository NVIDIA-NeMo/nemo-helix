<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OpenCV and Nemotron FileSet Job

One Helix Job reads a photo of Sutro Tower, detects its edges, builds a side-by-side contact sheet, and asks Nemotron 3.5 Super VL to caption the original, edge map, and montage.

| Job step | Reads | Publishes |
|---|---|---|
| `detect-edges` | Original images | Edge maps |
| `build-montage` | Original images and edge maps | Contact sheet |
| `describe-images` | Original images, edge maps, and contact sheet | Image descriptions in `descriptions.json` |

## Before sharing your screen

Use the workflow branch with a local, auth-disabled Helix instance serving Entities, Files, Secrets, and Jobs. The Jobs controller needs a CPU Docker profile with a Linux jobs-launcher and a platform URL reachable from its containers. These commands assume the profile is named `opencv`, the workspace is `default`, and `jq` is installed.

Open two terminals at the repository root. Run this in **both** so the CLI and submitter use the same checkout and platform:

```sh
cd "$(git rev-parse --show-toplevel)"
export NHX_BASE_URL=http://127.0.0.1:8080
export DEMO_SECRET=my-nvidia-key
nemo() { .venv/bin/nemo "$@"; }
curl -fsS "$NHX_BASE_URL/health/ready" | jq .
```

Expect `"status": "ready"`. In **terminal 1**, verify the CPU profile and build the image on the Docker daemon the scheduler uses:

```sh
nemo jobs list-execution-profiles
docker build -t helix-opencv-demo:local demo/opencv
```

If the Secret already exists, set `DEMO_SECRET` to its name. Otherwise, with `NVIDIA_API_KEY` already exported, create it once before presenting:

```sh
printf '%s' "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY first}" |
  nemo secrets create "$DEMO_SECRET" --workspace default --from-file - >/dev/null
```

## Live walkthrough

### 1. Show the input and commit it · terminal 1

“I’m putting one photo into a FileSet called `images`. A commit gives the Job a revision it can record.”

```sh
nemo files filesets create images --workspace default --exist-ok
nemo files list images --workspace default
```

For the one-photo baseline, start with an empty demo FileSet or one that already contains only the tower fixture. The submitter reuses `images`; any other photos there will also be sent to Nemotron. Use the following upload for an empty FileSet. If the fixture is already present, you can go straight to the commit.

```sh
nemo files upload demo/opencv/fixtures/sutro-tower.jpg images --remote-path sutro-tower.jpg --workspace default
nemo files list images --workspace default
```

```sh
nemo files filesets commit images --workspace default -f json |
  jq '{name, commit_revision}'
```

### 2. Submit the Job · terminal 1

```sh
uv run demo/opencv/helix_job.py --base-url "$NHX_BASE_URL" --profile opencv --secret "$DEMO_SECRET"
```

Point out the `Input FileSet: images @ ...` line and match it to the revision from step 1. The next line prints `Job opencv-... (...), attempt ...`. Copy the **Job name** starting with `opencv-`. Leave this terminal running while the submitter waits and downloads the published outputs.

### 3. Watch the steps and inspect the DAG · terminal 2

Paste the Job name when prompted:

```sh
printf 'Job name: '
read -r JOB
nemo jobs watch "$JOB" --workspace default --attempt-id current --history
```

“Helix starts edge detection first. The montage needs both the original and its edges. Captioning needs all three images.” The watch exits when the Job reaches a terminal state. If it finishes before you switch terminals, these commands still show exactly what ran:

```sh
nemo jobs attempts get current --job "$JOB" --workspace default -f json |
  jq '.platform_spec | {mode: .workflow.mode, steps: [.steps[] |
    {name, depends_on, reads: .config.inputs, publishes: .config.output_name}]}'

nemo jobs get-status "$JOB" --workspace default -f json |
  jq '{status, attempt_id, steps: [.workflow.nodes[] | {name, last_outcome}]}'
```

Expect `run_to_completion` and three `completed` outcomes. If a step fails, inspect its logs with `nemo jobs get-logs "$JOB" --workspace default --limit 100`.

### 4. Reveal the outputs · terminal 2

Each step uploads its files and commits its own output FileSet. Show the three output revisions in terminal 1, then list the artifacts from the platform:

```sh
for output in edges montage captions; do
  echo "$JOB-$output"
  nemo files list "$JOB-$output" --workspace default
done
```

Once terminal 1 finishes, open the downloaded contact sheet (macOS) and read the captions alongside it:

```sh
RUN="demo/opencv/runs/$JOB"
open "$RUN/montage/contact-sheet.png"
jq -r '.[] | "\(.source)/\(.image)  —  \(.caption)"' "$RUN/captions/descriptions.json"
```

The baseline produces one row on the sheet and three captions: the original tower photo, its edge map, and the complete sheet. Ask the audience which details Nemotron can still see in the edges.

### 5. Add the cake and run it again · optional

In terminal 1, upload the second fixture and publish the new revision:

```sh
nemo files upload demo/opencv/fixtures/birthday-cake.jpg images --remote-path birthday-cake.jpg --workspace default
nemo files filesets commit images --workspace default -f json |
  jq '{name, commit_revision}'
uv run demo/opencv/helix_job.py --base-url "$NHX_BASE_URL" --profile opencv --secret "$DEMO_SECRET"
```

This mode runs once per submission. Use the **new** Job name in terminal 2 and repeat steps 3–4. With the two fixtures, the new contact sheet has two rows and the caption FileSet has five descriptions. The first Job's output FileSets remain available under its original name.

## Fixtures and transform

The two JPEGs are copied from the Pachyderm console fixtures at revision `e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b`. The edge transform uses the example's Canny thresholds of 100 and 200. [Photo sources](https://github.com/pachyderm/pachyderm/tree/e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b/console/frontend/components/public) · [Original transform](https://github.com/pachyderm/pachyderm/blob/e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b/examples/opencv/edges.py)
