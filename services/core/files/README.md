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

## S3 uploads

Uploads use boto3 managed transfers through an AnyIO worker, preserving the configured storage
credentials and endpoint. Small files use PUT; large files use multipart with SDK retries and
abort cleanup. Transfers use the classic serial implementation to bound buffering and keep
async stream reads on the worker. No whole-file memory buffer or disk spool is needed.
Supported AWS SDK configuration applies in the Files process; AWS CLI S3 transfer settings do
not configure TransferConfig. Storage authorization failures return HTTP 403.

Cancellation during network I/O waits for the SDK call and retries to return. Configure storage
lifecycle expiry for incomplete multipart uploads after abrupt process loss or failed cleanup.
