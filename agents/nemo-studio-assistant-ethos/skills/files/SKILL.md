---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: files
description: "Explain NeMo Helix filesets and manage filesets and files with nemo_api(resource='files'|'files.filesets'). Use when the user asks where to put data, which filesets exist, or wants to create a fileset or upload, download, or delete a file."
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
# Filesets and files

A fileset is a named collection of files in a workspace. Datasets for
evaluation, customization, and data generation live in filesets, and other
features refer to a file as `<fileset>#<path>`.

## Explaining

- Link to `filesets` for the list, `fileset_new` to create one, and `fileset`
  with a name to open one.
- When a workflow needs the user to choose a file, use `select_dataset_file`
  rather than asking for a path in text.
- Large uploads are easier on the Filesets page than through chat.

## Acting on request

- Fileset create, retrieve, and delete: `nemo_api(resource="files.filesets", action="create"|"retrieve"|"delete", params={"name": "..."}, workspace="<active request workspace>")`.
- Fileset list takes no name: `nemo_api(resource="files.filesets", action="list", params={}, workspace="<active request workspace>")`.
- Small text uploads: `nemo_api(resource="files", action="upload_content", params={"content": "...", "remote_path": "path.txt", "fileset": "..."}, workspace="<active request workspace>")`.
- Downloads: `nemo_api(resource="files", action="download_content", params={"remote_path": "path.txt", "fileset": "..."}, workspace="<active request workspace>")`.
- Listing: `nemo_api(resource="files", action="list", params={"fileset": "..."}, workspace="<active request workspace>")`.
- Deleting a file: `nemo_api(resource="files", action="delete", params={"remote_path": "...", "fileset": "..."}, workspace="<active request workspace>")`.

## When a write fails

- Treat an error from `create`, `upload_content`, or `delete` as an unknown
  commit state unless it clearly says the request was rejected before it ran.
  Fix a clear validation error and retry once.
- Otherwise read back the fileset or file before doing anything else. If the
  read-back does not confirm the intended content, stop and report what you
  know instead of retrying.
