<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Image montage workflow

Use the [shared workflow CLI](../README.md) to create and operate named runs:

```bash
docker build -t helix-opencv-demo:local demo/opencv
uv run python -m demo.workflows create montage my-images --secret my-nvidia-key --watch
```

The named Secret must already contain your NVIDIA API key. The first pass uses
the bundled Sutro Tower photo. Each run gets its own input and output Filesets;
it does not read or modify the legacy shared `images` Fileset.

| Step | Reads | Publishes |
| --- | --- | --- |
| `detect-edges` | Original photos | Canny edge maps |
| `build-montage` | Original photos and edge maps | `contact-sheet.png` |
| `describe-images` | Photos, edge maps and montage | `descriptions.json` |

An input Fileset commit triggers edge detection. Downstream nodes use explicit
Step-success events and typed Step-success readiness. The launcher stages each
Step's input directories and uploads its output. The controller commits outputs
before finalizing Step success and releasing downstream work. Transform scripts
make no Files API requests or commits.

Wait for the first pass to finish, then add the cake photo and request the next
pass of the same Attempt:

```bash
uv run python -m demo.workflows trigger my-images --input demo/opencv/fixtures/birthday-cake.jpg
```

For unified logs, press Tab, then `a`, then `f` in the TUI, or open a separate
line watcher:

```bash
uv run python -m demo.workflows watch my-images --no-tui
```

Captioning logs each request before waiting for the NVIDIA model. To repeat only
captioning using the published images, edges and montage:

```bash
uv run python -m demo.workflows trigger my-images --step describe-images
```

This makes new NVIDIA model requests. Invalidate `detect-edges` or use plain
`trigger my-images` to repeat the pipeline. Events received during a pass wait
until it drains; commit markers do not freeze Fileset contents.

Download the current outputs after a pass finishes:

```bash
uv run python -m demo.workflows download my-images
open demo/runs/my-images/montage/contact-sheet.png
jq -r '.[] | "\(.source)/\(.image)  —  \(.caption)"' demo/runs/my-images/captions/descriptions.json
```

One photo produces one montage row and three captions. Two photos produce two
rows and five captions. Cancel the Attempt when finished:

```bash
uv run python -m demo.workflows cancel my-images
```

Rebuild the image after changing `demo.py` or `captions.py`. Create a fresh Job
after changing the workflow's commands or transfers; Attempt specs are immutable.
The old `cleanup.sh` targets legacy generated OpenCV demo names and the shared
`images` Fileset. It does not manage these named CLI runs.

## Test locally without Docker

```bash
uv run --no-sync --with opencv-python-headless==5.1.0.95 python -m pytest demo/opencv/tests -q
```

Tests use both fixtures to check edge files, montage dimensions and caption
coverage, plus failure when an edge is missing. Model responses are stubbed;
these tests make no NVIDIA API calls. They are skipped if OpenCV is absent.

## Photo sources

The JPEGs come from the Pachyderm console fixtures at revision
`e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b`. The transform uses Canny thresholds
100 and 200. [Photo sources](https://github.com/pachyderm/pachyderm/tree/e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b/console/frontend/components/public)
· [Original transform](https://github.com/pachyderm/pachyderm/blob/e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b/examples/opencv/edges.py).
