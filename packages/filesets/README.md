<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Filesets Package

This package provides `FilesetFileSystem`, an fsspec-compatible filesystem for working with NeMo Helix filesets. It also serves as a migration guide from Datastore (HuggingFace Hub) to the Files Service.

## Quick Start

The `FilesetFileSystem` is available directly from the SDK via `sdk.files.fsspec`:

```python
from nemo_helix import NeMoHelix

sdk = NeMoHelix(base_url="http://nhx-host")

# List files
sdk.files.fsspec.ls("my-workspace/my-fileset/")

# Read a file
content = sdk.files.fsspec.cat("my-workspace/my-fileset/config.json")

# Write a file
sdk.files.fsspec.pipe("my-workspace/my-fileset/data.txt", b"hello world")

# Download files
sdk.files.fsspec.get("my-workspace/my-fileset/", "/local/path/", recursive=True)

# Upload files
sdk.files.fsspec.put("/local/file.txt", "my-workspace/my-fileset/file.txt")
```

## Migration from Datastore (HuggingFace Hub)

### Overview

| Datastore (Old) | Files Service (New) |
|-----------------|---------------------|
| `HfFileSystem` | `sdk.files.fsspec` |
| `huggingface_hub.HfApi` | `nemo_helix.NeMoHelix` SDK |
| HuggingFace Hub protocol (`/v1/hf`) | Files API (`/v2/workspaces/{ws}/filesets/{name}`) |
| Repositories + Branches | Workspaces + Filesets |

> **Note:** Use `sdk.files` as the entry point for all file operations. Use `sdk.files.filesets` for fileset entity management (create, delete, retrieve) and `sdk.files.upload/download/list/delete` for file content operations.

### Concept Mapping

| HuggingFace Hub | Filesets | Notes |
|-----------------|----------|-------|
| `namespace` | `workspace` | Container for filesets |
| `repo_id` (`namespace/repo`) | `workspace/fileset` | Unique identifier |
| `branch` / `revision` | N/A | Filesets don't have branches |
| `repo_type` ("model", "dataset") | `purpose` | Optional metadata |

---

### 1. Client Initialization

**Before (Datastore/HfApi):**
```python
from huggingface_hub import HfApi

endpoint = f"{nds_host}/v1/hf"
api = HfApi(endpoint=endpoint, token=token)
```

**After (SDK):**
```python
from nemo_helix import NeMoHelix

sdk = NeMoHelix(base_url=nhx_host, default_headers={"Authorization": f"Bearer {token}"})
```

---

### 2. Create Repository / Fileset

**Before (Datastore):**
```python
# Create namespace (NDS-specific)
requests.post(f"{nds_host}/v1/datastore/namespaces", json={"namespace": namespace})

# Create repo
api.create_repo(f"{namespace}/{repo_name}", repo_type="model", exist_ok=True)
```

**After (SDK):**
```python
# Workspaces are typically pre-created, but filesets can be created:
sdk.files.filesets.create(
    workspace="my-workspace",
    name="my-fileset",
    description="Model checkpoint storage",
    purpose="model",  # or "dataset", or empty/unset
)
```

---

### 3. List Files in Repository

**Before (Datastore):**
```python
# Flat list of all files
files = api.list_repo_files(repo_id="namespace/repo", revision="main")
# Returns: ["file1.txt", "subdir/file2.json", ...]

# Or with tree structure
items = api.list_repo_tree(repo_id="namespace/repo", recursive=True)
```

**After (SDK):**
```python
# List directory contents
files = sdk.files.fsspec.ls("my-workspace/my-fileset/subdir/")

# Find all files recursively
all_files = sdk.files.fsspec.find("my-workspace/my-fileset/")

# Glob patterns
parquet_files = sdk.files.fsspec.glob("my-workspace/my-fileset/**/*.parquet")
```

---

### 4. Download Entire Repository (snapshot_download)

**Before (Datastore):**
```python
local_path = api.snapshot_download(
    repo_id="namespace/repo",
    revision="main",
    repo_type="model",
    local_dir="/local/destination",
    allow_patterns="checkpoints/**",  # optional filter
)
```

**After (SDK):**
```python
# Download entire fileset
sdk.files.fsspec.get("my-workspace/my-fileset/", "/local/destination/", recursive=True)

# Download specific subdirectory
sdk.files.fsspec.get("my-workspace/my-fileset/checkpoints/", "/local/destination/checkpoints/", recursive=True)
```

**With progress tracking:**
```python
from fsspec.callbacks import TqdmCallback

with TqdmCallback(tqdm_kwargs={"desc": "Downloading"}) as callback:
    sdk.files.fsspec.get("my-workspace/my-fileset/", "/local/destination/", recursive=True, callback=callback)
```

---

### 5. Download Single File (hf_hub_download)

**Before (Datastore):**
```python
local_path = api.hf_hub_download(
    repo_id="namespace/repo",
    filename="config.json",
    repo_type="model",
    local_dir="/local/dir",
)
```

**After (SDK):**
```python
# Download to local file
sdk.files.fsspec.get("my-workspace/my-fileset/config.json", "/local/dir/config.json")

# Or read directly into memory
content = sdk.files.fsspec.cat("my-workspace/my-fileset/config.json")

# Or open as file-like object
with sdk.files.fsspec.open("my-workspace/my-fileset/config.json", "rb") as f:
    data = json.load(f)
```

---

### 6. Upload Single File

**Before (Datastore):**
```python
api.upload_file(
    path_or_fileobj="/local/file.txt",
    path_in_repo="data/file.txt",
    repo_id="namespace/repo",
    repo_type="dataset",
)
```

**After (SDK):**
```python
# Upload from local file (streaming, memory-efficient)
sdk.files.fsspec.put("/local/file.txt", "my-workspace/my-fileset/data/file.txt")

# Or write bytes directly
sdk.files.fsspec.pipe("my-workspace/my-fileset/data/file.txt", b"file contents")

# Or use file-like interface
with sdk.files.fsspec.open("my-workspace/my-fileset/data/file.txt", "wb") as f:
    f.write(b"file contents")
```

---

### 7. Upload Directory (upload_folder)

**Before (Datastore):**
```python
api.upload_folder(
    folder_path="/local/model/",
    path_in_repo="checkpoints/",
    repo_id="namespace/repo",
    repo_type="model",
    ignore_patterns=["*.tmp", ".git"],
)
```

**After (SDK):**
```python
# Upload entire directory
sdk.files.fsspec.put("/local/model/", "my-workspace/my-fileset/checkpoints/", recursive=True)
```

**With filtering (manual):**
```python
from pathlib import Path

local_dir = Path("/local/model/")
for local_file in local_dir.rglob("*"):
    if local_file.is_file() and not local_file.suffix == ".tmp":
        relative = local_file.relative_to(local_dir)
        remote_path = f"my-workspace/my-fileset/checkpoints/{relative}"
        sdk.files.fsspec.put(str(local_file), remote_path)
```

---

### 8. Delete Files

**Before (Datastore):**
```python
# Delete branch (NDS-specific)
api.delete_branch(repo_id="namespace/repo", branch="feature-branch", repo_type="model")

# Delete repo
api.delete_repo(repo_id="namespace/repo", repo_type="model")
```

**After (SDK):**
```python
# Delete single file
sdk.files.fsspec.rm("my-workspace/my-fileset/path/to/file.txt")

# Delete multiple files
sdk.files.fsspec.rm(["my-workspace/my-fileset/file1.txt", "my-workspace/my-fileset/file2.txt"])

# Delete entire fileset (use SDK for fileset management)
sdk.files.filesets.delete("my-fileset", workspace="my-workspace")
```

---

### 9. Check if Repository/File Exists

**Before (Datastore):**
```python
exists = api.repo_exists(repo_id="namespace/repo", repo_type="model")
```

**After (SDK):**
```python
# Check fileset exists
exists = sdk.files.fsspec.exists("my-workspace/my-fileset/")

# Check file exists
exists = sdk.files.fsspec.exists("my-workspace/my-fileset/config.json")

# Get file info
info = sdk.files.fsspec.info("my-workspace/my-fileset/config.json")
# Returns: {"name": "...", "size": 1234, "type": "file"}
```

---

### 10. Async Operations

The `FilesetFileSystem` fully supports async operations. When you call sync methods, fsspec runs the async methods in an event-loop thread and blocks until completion.

To use async methods directly, use the `_`-prefixed variants:

**Before (Datastore):**
```python
import asyncio

# Wrap blocking calls
result = await asyncio.to_thread(api.snapshot_download, repo_id="namespace/repo", local_dir="/dest")

# Or use HfApi's future wrapper
future = api.run_as_future(api.list_repo_files, repo_id="namespace/repo")
files = await asyncio.wait_for(asyncio.wrap_future(future), timeout=30)
```

**After (SDK):**
```python
# Async methods (prefixed with _) can be called directly from async context
files = await sdk.files.fsspec._ls("my-workspace/my-fileset/")
content = await sdk.files.fsspec._cat_file("my-workspace/my-fileset/config.json")
await sdk.files.fsspec._get("my-workspace/my-fileset/", "/local/dest/", recursive=True)
```

> **Note:** The `_` prefix for async methods is an fsspec convention, not a private API indicator. See [fsspec async docs](https://filesystem-spec.readthedocs.io/en/latest/async.html).

---

## Integration with Data Libraries

### Pandas

```python
import pandas as pd

# Read parquet
with sdk.files.fsspec.open("my-workspace/my-fileset/data.parquet", "rb") as f:
    df = pd.read_parquet(f)

# Write parquet
with sdk.files.fsspec.open("my-workspace/my-fileset/output.parquet", "wb") as f:
    df.to_parquet(f)

# Or pass filesystem directly
df = pd.read_parquet("my-workspace/my-fileset/data.parquet", filesystem=sdk.files.fsspec)
```

### PyArrow

```python
import pyarrow.parquet as pq

# Read parquet dataset
dataset = pq.ParquetDataset("my-workspace/my-fileset/data/", filesystem=sdk.files.fsspec)
table = dataset.read()

# Write parquet
pq.write_table(table, "my-workspace/my-fileset/output.parquet", filesystem=sdk.files.fsspec)
```

### DuckDB

```python
import duckdb

# Register filesystem and query directly
duckdb.register_filesystem(sdk.files.fsspec)
result = duckdb.sql("SELECT * FROM 'fileset://my-workspace/my-fileset/data.parquet'")
```

---

## Complete Migration Example

**Before (NemoDataStoreClient):**
```python
class NemoDataStoreClient:
    def __init__(self, nds_host: str, token: str):
        endpoint = f"{nds_host}/v1/hf"
        self.api = HfApi(endpoint=endpoint, token=token)

    def download_model(self, repo_id: str, branch_name: str, destination: str) -> str:
        return self.api.snapshot_download(
            repo_id=repo_id,
            revision=branch_name,
            repo_type="model",
            local_dir=destination,
        )

    async def get_flat_files_list(self, repo_id: str, revision: str) -> list[str]:
        future = self.api.run_as_future(self.api.list_repo_files, repo_id=repo_id, revision=revision)
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=30)
```

**After (SDK):**
```python
class FilesetClient:
    def __init__(self, sdk: NeMoHelix):
        self.sdk = sdk

    def download_model(self, workspace: str, fileset: str, destination: str) -> str:
        self.sdk.files.fsspec.get(f"{workspace}/{fileset}/", destination, recursive=True)
        return destination

    def get_flat_files_list(self, workspace: str, fileset: str) -> list[str]:
        return self.sdk.files.fsspec.find(f"{workspace}/{fileset}/")


# Async version - same initialization, just use async methods
class AsyncFilesetClient:
    def __init__(self, sdk: NeMoHelix):
        self.sdk = sdk

    async def download_model(self, workspace: str, fileset: str, destination: str) -> str:
        await self.sdk.files.fsspec._get(f"{workspace}/{fileset}/", destination, recursive=True)
        return destination

    async def get_flat_files_list(self, workspace: str, fileset: str) -> list[str]:
        return await self.sdk.files.fsspec._find(f"{workspace}/{fileset}/")
```

---

## Key Differences to Note

1. **No branching**: Filesets don't have branches/revisions. If you need versioning, create separate filesets (e.g., `my-model-v1`, `my-model-v2`).
2. **Workspace scope**: All filesets belong to a workspace. Include the workspace in all paths.
3. **Path format**: Paths use `workspace/fileset/path` format, not `namespace/repo` with separate path arguments.
4. **Native async**: The filesystem supports both sync and async operations natively.

---

## Error Handling

**Before (Datastore):**
```python
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError

try:
    api.list_repo_files(repo_id="namespace/repo")
except RepositoryNotFoundError:
    print("Repo not found")
except RevisionNotFoundError:
    print("Branch not found")
except HfHubHTTPError as e:
    print(f"HTTP error: {e.response.status_code}")
```

**After (SDK):**
```python
try:
    sdk.files.fsspec.ls("my-workspace/my-fileset/")
except FileNotFoundError:
    print("Fileset or path not found")
except PermissionError:
    print("Access denied")
except Exception as e:
    print(f"Error: {e}")
```

---

## Advanced: Standalone FilesetFileSystem

If you need a `FilesetFileSystem` instance without going through the SDK (e.g., for custom configuration), you can import it directly:

```python
from nemo_helix import NeMoHelix
from nemo_helix.filesets import FilesetFileSystem

sdk = NeMoHelix(base_url="http://nhx-host")
fs = FilesetFileSystem(sdk=sdk)
```

## Protocol Registration

To use `fileset://` URLs with fsspec or libraries that support fsspec URLs:

```python
from nemo_helix.filesets import FilesetFileSystem

# Register the fileset:// protocol globally with fsspec
FilesetFileSystem.register_fsspec()

# Now you can use fileset:// URLs with fsspec
import fsspec
fs = fsspec.filesystem("fileset", sdk=sdk)

# Or with pandas (after registration)
import pandas as pd
df = pd.read_parquet("fileset://my-workspace/my-fileset/data.parquet", storage_options={"sdk": sdk})
```
