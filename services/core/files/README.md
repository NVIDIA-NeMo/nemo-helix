<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# files

## localdev
Important: all commands must be run from the repository root.

To reset the storage:
```bash
rm -rf services/core/files/devstorage; mkdir services/core/files/devstorage
```

To run the service:
```bash
export ENVFILE="services/core/files/config/local.env" && \
  uv run --frozen --env-file "$ENVFILE" nemo services run --services files,entities,auth,secrets
```

This will run `files` using entity store for persistence and a local storage backend.
The combo of `config/local.yaml` and `config/local.env` will place all relevant storage in `services/core/files/devstorage/*`.

## S3 fileset caching

S3-backed filesets check object metadata with HEAD before each download. Cache
keys include the S3 endpoint, bucket/prefix, object key, and ETag/version identity.
Unchanged objects use cached bytes; uploads, deletions, and direct bucket changes
are visible on the next request. Cache warming uses the same object identity.

Downloads use `If-Match` with the resolved ETag and prefer the resolved version
ID for versioned objects. If version access is denied, the download retries the
current object with the same ETag guard, preserving compatibility with
`s3:GetObject`-only credentials. Without an ETag, this fallback is unsafe and
version-read permission is required. If the object changes before GET, the API
returns HTTP 409 with a retry message; retry to resolve the new object. Both the
fileset and HF-compatible download routes pin one snapshot for headers and bytes.
Backends without an ETag or immutable version ID bypass caching.

SSE-KMS and DSSE-KMS objects bypass local content caching:
HEAD does not establish `kms:Decrypt` permission, so these reads always perform
an authorized S3 content GET. With an ETag, KMS reads use a conditional current
GET directly, avoiding a version-permission probe. SSE-S3 objects retain normal
caching behavior.
Cache-status listings use at most eight concurrent metadata lookups, with one
HEAD per object in a cold listing. Use `include_cache_status` only when needed;
upstream lookup failures return the standard HTTP 502 error rather than 500.

Legacy entries under `cache/s3/` are ignored in favor of `cache/s3-v2/`; no manual
cleanup is needed to restore fresh reads after upgrading. Legacy and superseded
files are not automatically removed by the freshness fix.

## scripts
`services/core/files/script` has a few e2e scripts for testing functionality.

### Downloads
The tests assume there's some data locally on your machine:

- `~/Downloads/gpt-oss-120b`, needed for `e2e_filesets_api.py`. Download on [HuggingFace](https://huggingface.co/openai/gpt-oss-120b). The only files needed for this script are the safetensors files, `hf download openai/gpt-oss-120b --local-dir ~/Downloads/gpt-oss-120b --include '*.safetensors'`
- `~/Downloads/en_US.parquet`, needed for `e2e_duckdb_httpfs.py`. Provide a local Parquet file with the columns expected by that script.
### Usage

```bash
uv run --frozen services/core/files/script/e2e_filesets_api.py
uv run --frozen services/core/files/script/e2e_duckdb_httpfs.py
```
