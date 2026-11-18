#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu
cd "$(dirname "$0")"
image=helix-opencv-demo:local
mkdir -p runs
run=$(mktemp -d "$PWD/runs/demo.XXXXXX")
docker build -t "$image" .
for stage in baseline updated; do
    dir="$run/$stage"
    mkdir -p "$dir/images" "$dir/edges" "$dir/montage"
    cp fixtures/sutro-tower.jpg "$dir/images/"
    if [ "$stage" = updated ]; then
        cp fixtures/birthday-cake.jpg "$dir/images/"
    fi
    docker run --rm --network none -v "$dir/images:/images:ro" -v "$dir/edges:/edges" \
        "$image" detect-edges --images /images --edges /edges
    docker run --rm --network none -v "$dir/images:/images:ro" -v "$dir/edges:/edges:ro" \
        -v "$dir/montage:/montage" "$image" build-montage --images /images --edges /edges --montage /montage
done
echo "Demo outputs: $run"
