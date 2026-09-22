<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fileset CRUD Operations - CLI Harbor Test

Tests fileset CRUD operations (create, upload, list, download, delete) using the NeMo Helix CLI.

## What It Tests

- Creating a fileset with name and description
- Uploading a file to a fileset
- Listing files in a fileset
- Downloading file content
- Deleting a file from a fileset
- Deleting a fileset

## Running

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .

export NVIDIA_API_KEY='your-key'

python tests/agentic-use/nat_runner.py files-crud-cli-easy \
    --agent-backend aut \
    --aut-agent-name <your-agent> \
    --aut-agent-config <path-to-agent-config.yml>
```
