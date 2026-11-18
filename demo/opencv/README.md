<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Standalone OpenCV rehearsal

From this directory, run `./demo.sh` with Docker running. The first build downloads Python, OpenCV and NumPy. Both workload containers run with networking disabled. The script prints the path of a fresh run directory and preserves previous runs.

1. Open `baseline/montage/contact-sheet.png` in the printed directory. The left panel shows Sutro Tower and the right panel shows its edges.
2. Open `updated/montage/contact-sheet.png`. Adding the cake photo creates a second row. Rows sort by filename, so the cake appears above the tower.
3. Inspect `updated/edges/` to see the intermediate PNGs the montage consumed.

## Step interface

Both steps use `helix-opencv-demo:local`. The image entrypoint accepts these commands:

```text
detect-edges --images /images --edges /edges
build-montage --images /images --edges /edges --montage /montage
```

Mount an input directory of JPEG or PNG photos with distinct filename stems at `/images`. Edge detection writes one grayscale PNG per original to `/edges`. Montage reads those two directories and writes `contact-sheet.png` to `/montage`. The script uses fresh directories per batch and invokes each step once in order. During platform integration, FileSet commits and Jobs will drive that sequence.

## Fixtures and transform

The two JPEGs are copied from the Pachyderm console fixtures at revision `e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b`. The edge transform uses the example's Canny thresholds of 100 and 200. [Photo sources](https://github.com/pachyderm/pachyderm/tree/e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b/console/frontend/components/public) · [Original transform](https://github.com/pachyderm/pachyderm/blob/e237475e9910a2d6299d7d2c3d6fc3b9a8f28f0b/examples/opencv/edges.py)
