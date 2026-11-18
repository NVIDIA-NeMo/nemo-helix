<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Standalone OpenCV rehearsal

With Docker running and `NVIDIA_API_KEY` exported, run `./demo.sh` from this directory. The first build downloads Python, OpenCV and NumPy. Edge detection and montage run offline. Captioning uses the NVIDIA API. The script prints the path of a fresh run directory and preserves previous runs.

1. Open `baseline/montage/contact-sheet.png` in the printed directory. The left panel shows Sutro Tower and the right panel shows its edges.
2. Open `updated/montage/contact-sheet.png`. Adding the cake photo creates a second row. Rows sort by filename, so the cake appears above the tower.
3. Inspect `updated/edges/` to see the intermediate PNGs the montage consumed.

## Describe the photos, edge maps, and montage with Nemotron

For each batch, a third container sends a resized copy of each original photo, each edge map, and the finished contact sheet to NVIDIA's hosted `nvidia/nemotron-3.5-super-vl-120b-a12b` model. The prompt asks for one sentence describing visible details. The runner prints the answers and saves the source directory, filename, model ID, and returned caption in `captions/descriptions.json` beside the montage directory. Open the updated contact sheet and read the corresponding descriptions. The baseline and updated batches make three and five inference calls, respectively.

## Step interface

All three steps use `helix-opencv-demo:local`. The default image entrypoint handles the two OpenCV commands. The caption step invokes `python /app/captions.py` with read-only mounts for `images`, `edges`, and `montage` and a writable `captions` directory.

```text
detect-edges --images /images --edges /edges
build-montage --images /images --edges /edges --montage /montage
```

Mount an input directory of JPEG or PNG photos with distinct filename stems at `/images`. Edge detection writes one grayscale PNG per original to `/edges`. Montage reads those two directories and writes `contact-sheet.png` to `/montage`. The script uses fresh directories per batch and invokes each step once in order. During platform integration, FileSet commits and Jobs will drive that sequence.

## Fixtures and transform

The two JPEGs are copied from the Pachyderm console fixtures at revision `e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b`. The edge transform uses the example's Canny thresholds of 100 and 200. [Photo sources](https://github.com/pachyderm/pachyderm/tree/e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b/console/frontend/components/public) · [Original transform](https://github.com/pachyderm/pachyderm/blob/e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b/examples/opencv/edges.py)
