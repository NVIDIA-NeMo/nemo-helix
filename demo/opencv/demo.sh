#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu
[ "$#" -eq 0 ] || { echo "Usage: $0" >&2; exit 2; }
: "${NVIDIA_API_KEY:?Export NVIDIA_API_KEY before running the demo}"
cd "$(dirname "$0")"
image=helix-opencv-demo:local
mkdir -p runs
run=$(mktemp -d "$PWD/runs/demo.XXXXXX")
docker build -t "$image" .
for stage in baseline updated; do
    dir="$run/$stage"
    mkdir -p "$dir/images" "$dir/edges" "$dir/montage" "$dir/captions"
    cp fixtures/sutro-tower.jpg "$dir/images/"
    if [ "$stage" = updated ]; then
        cp fixtures/birthday-cake.jpg "$dir/images/"
    fi
    docker run --rm --network none -v "$dir/images:/images:ro" -v "$dir/edges:/edges" \
        "$image" detect-edges --images /images --edges /edges
    docker run --rm --network none -v "$dir/images:/images:ro" -v "$dir/edges:/edges:ro" \
        -v "$dir/montage:/montage" "$image" build-montage --images /images --edges /edges --montage /montage
    docker run --rm -e NVIDIA_API_KEY --entrypoint python \
        -v "$dir/images:/images:ro" -v "$dir/edges:/edges:ro" \
        -v "$dir/montage:/montage:ro" -v "$dir/captions:/captions" \
        "$image" /app/captions.py --images /images --edges /edges --montage /montage --captions /captions
done
echo "Demo outputs: $run"
