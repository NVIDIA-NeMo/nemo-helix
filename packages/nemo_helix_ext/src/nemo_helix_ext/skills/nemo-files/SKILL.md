---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-files
description: >
  NeMo files CLI reference for filesets, file upload/download, and dataset management.
  Use when the task involves filesets, file uploads, file downloads, datasets,
  JSONL files, or `nemo files` CLI commands.
preconditions:
  - nemo_setup_complete
  - workspace_exists
user-invocable: true
allowed-tools: Bash, Read, Grep
---

# NeMo Files CLI Reference

## Environment

- **CLI path**: `/app/.venv/bin/nemo`
- **API server**: `http://localhost:8080` (default)

## Fileset Commands

```bash
# Create a fileset
nemo files filesets create <name> --description "<description>"

# Create a fileset in a specific workspace
nemo files filesets create <name> --description "<description>" --workspace <workspace>

# Get fileset details
nemo files filesets get <name>

# List filesets
nemo files filesets list

# Update fileset metadata
nemo files filesets update <name> --description "<description>"

# Delete a fileset
nemo files filesets delete <name>
```

## File Commands

```bash
# Upload a file to a fileset
nemo files upload <local_path> <fileset_name>

# Upload to a specific workspace
nemo files upload <local_path> <fileset_name> --workspace <workspace>

# List files in a fileset
nemo files list <fileset_name>

# Download a file
nemo files download <fileset_name> --remote-path <remote_path> -o <local_path>

# Delete a file from a fileset
nemo files delete <fileset_name> --remote-path <file_path>
```

## Common Patterns

### Upload a JSONL dataset

```bash
# Create the dataset file
cat > /tmp/dataset.jsonl << 'EOF'
{"output": "hello", "expected": "hello"}
{"output": "world", "expected": "world"}
{"output": "foo", "expected": "bar"}
EOF

# Create fileset and upload
nemo files filesets create my-dataset --workspace <workspace>
nemo files upload /tmp/dataset.jsonl my-dataset --workspace <workspace>
```

### Upload and verify

```bash
nemo files upload /tmp/myfile.txt my-fileset
nemo files list my-fileset
```
