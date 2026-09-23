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

export ANTHROPIC_API_KEY='your-key'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'

harbor run -p tests/agentic-use/files-crud-cli \
    --agent claude-code \
    --model anthropic/claude-sonnet-4-5
```
