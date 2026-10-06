<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Filesets Package

This package provides `FilesetFileSystem`, an fsspec-compatible filesystem for working with NeMo Helix filesets. It also serves as a migration guide from Datastore (HuggingFace Hub) to the Files Service.

## Quick Start

The `FilesetFileSystem` is available from `FilesResource.fsspec`, which wraps a typed `NemoClient`:

```python
from filesets.resources import FilesResource
from nemo_helix_plugin.client.client import NemoClient

client = NemoClient(base_url="http://nhx-host")
files = FilesResource(client)

# List files
files.fsspec.ls("my-workspace/my-fileset/")

# Read a file
content = files.fsspec.cat("my-workspace/my-fileset/config.json")

# Write a file
files.fsspec.pipe("my-workspace/my-fileset/data.txt", b"hello world")

# Download files
files.fsspec.get("my-workspace/my-fileset/", "/local/path/", recursive=True)

# Upload files
files.fsspec.put("/local/file.txt", "my-workspace/my-fileset/file.txt")
```

## Migration from Datastore (HuggingFace Hub)

### Overview

| Datastore (Old) | Files Service (New) |
|-----------------|---------------------|
| `HfFileSystem` | `FilesResource.fsspec` |
| `huggingface_hub.HfApi` | `nemo_helix_plugin.client.client.NemoClient` with `FilesResource` |
| HuggingFace Hub protocol (`/v1/hf`) | Files API (`/v2/workspaces/{ws}/filesets/{name}`) |
| Repositories + Branches | Workspaces + Filesets |

> **Note:** Use `FilesResource` as the entry point for file content operations (`files.upload/download/list/delete`). Use the typed client at `files.client` (`FilesClient`) for fileset entity management (`create_fileset`, `delete_fileset`, `get_fileset`).

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

**After (typed client):**
```python
from filesets.resources import FilesResource
from nemo_helix_plugin.client.client import NemoClient

client = NemoClient(base_url=nhx_host, default_headers={"Authorization": f"Bearer {token}"})
files = FilesResource(client)
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

**After (typed client):**
```python
# Workspaces are typically pre-created, but filesets can be created:
from nemo_helix_plugin.files.types import CreateFilesetRequest

files.client.create_fileset(
    workspace="my-workspace",
    body=CreateFilesetRequest(
        name="my-fileset",
        description="Model checkpoint storage",
        purpose="model",  # or "dataset", or empty/unset
    ),
    exist_ok=True,
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

**After (typed client):**
```python
# List directory contents
files = files.fsspec.ls("my-workspace/my-fileset/subdir/")

# Find all files recursively
all_files = files.fsspec.find("my-workspace/my-fileset/")

# Glob patterns
parquet_files = files.fsspec.glob("my-workspace/my-fileset/**/*.parquet")
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

**After (typed client):**
```python
# Download entire fileset
files.fsspec.get("my-workspace/my-fileset/", "/local/destination/", recursive=True)

# Download specific subdirectory
files.fsspec.get("my-workspace/my-fileset/checkpoints/", "/local/destination/checkpoints/", recursive=True)
```

**With progress tracking:**
```python
from fsspec.callbacks import TqdmCallback

with TqdmCallback(tqdm_kwargs={"desc": "Downloading"}) as callback:
    files.fsspec.get("my-workspace/my-fileset/", "/local/destination/", recursive=True, callback=callback)
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

**After (typed client):**
```python
# Download to local file
files.fsspec.get("my-workspace/my-fileset/config.json", "/local/dir/config.json")

# Or read directly into memory
content = files.fsspec.cat("my-workspace/my-fileset/config.json")

# Or open as file-like object
with files.fsspec.open("my-workspace/my-fileset/config.json", "rb") as f:
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

**After (typed client):**
```python
# Upload from local file (streaming, memory-efficient)
files.fsspec.put("/local/file.txt", "my-workspace/my-fileset/data/file.txt")

# Or write bytes directly
files.fsspec.pipe("my-workspace/my-fileset/data/file.txt", b"file contents")

# Or use file-like interface
with files.fsspec.open("my-workspace/my-fileset/data/file.txt", "wb") as f:
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

**After (typed client):**
```python
# Upload entire directory
files.fsspec.put("/local/model/", "my-workspace/my-fileset/checkpoints/", recursive=True)
```

**With filtering (manual):**
```python
from pathlib import Path

local_dir = Path("/local/model/")
for local_file in local_dir.rglob("*"):
    if local_file.is_file() and not local_file.suffix == ".tmp":
        relative = local_file.relative_to(local_dir)
        remote_path = f"my-workspace/my-fileset/checkpoints/{relative}"
        files.fsspec.put(str(local_file), remote_path)
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

**After (typed client):**
```python
# Delete single file
files.fsspec.rm("my-workspace/my-fileset/path/to/file.txt")

# Delete multiple files
files.fsspec.rm(["my-workspace/my-fileset/file1.txt", "my-workspace/my-fileset/file2.txt"])

# Delete entire fileset (use the typed client for fileset management)
files.client.delete_fileset(name="my-fileset", workspace="my-workspace")
```

---

### 9. Check if Repository/File Exists

**Before (Datastore):**
```python
exists = api.repo_exists(repo_id="namespace/repo", repo_type="model")
```

**After (typed client):**
```python
# Check fileset exists
exists = files.fsspec.exists("my-workspace/my-fileset/")

# Check file exists
exists = files.fsspec.exists("my-workspace/my-fileset/config.json")

# Get file info
info = files.fsspec.info("my-workspace/my-fileset/config.json")
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

**After (typed client):**
```python
# Async methods (prefixed with _) can be called directly from async context
files = await files.fsspec._ls("my-workspace/my-fileset/")
content = await files.fsspec._cat_file("my-workspace/my-fileset/config.json")
await files.fsspec._get("my-workspace/my-fileset/", "/local/dest/", recursive=True)
```

> **Note:** The `_` prefix for async methods is an fsspec convention, not a private API indicator. See [fsspec async docs](https://filesystem-spec.readthedocs.io/en/latest/async.html).

---

## Integration with Data Libraries

### Pandas

```python
import pandas as pd

# Read parquet
with files.fsspec.open("my-workspace/my-fileset/data.parquet", "rb") as f:
    df = pd.read_parquet(f)

# Write parquet
with files.fsspec.open("my-workspace/my-fileset/output.parquet", "wb") as f:
    df.to_parquet(f)

# Or pass filesystem directly
df = pd.read_parquet("my-workspace/my-fileset/data.parquet", filesystem=files.fsspec)
```

### PyArrow

```python
import pyarrow.parquet as pq

# Read parquet dataset
dataset = pq.ParquetDataset("my-workspace/my-fileset/data/", filesystem=files.fsspec)
table = dataset.read()

# Write parquet
pq.write_table(table, "my-workspace/my-fileset/output.parquet", filesystem=files.fsspec)
```

### DuckDB

```python
import duckdb

# Register filesystem and query directly
duckdb.register_filesystem(files.fsspec)
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

**After (typed client):**
```python
class FilesetClient:
    def __init__(self, files: FilesResource):
        self.files = files

    def download_model(self, workspace: str, fileset: str, destination: str) -> str:
        self.files.fsspec.get(f"{workspace}/{fileset}/", destination, recursive=True)
        return destination

    def get_flat_files_list(self, workspace: str, fileset: str) -> list[str]:
        return self.files.fsspec.find(f"{workspace}/{fileset}/")


# Async version - same initialization, just use async methods
class AsyncFilesetClient:
    def __init__(self, files: AsyncFilesResource):
        self.files = files

    async def download_model(self, workspace: str, fileset: str, destination: str) -> str:
        await self.files.fsspec._get(f"{workspace}/{fileset}/", destination, recursive=True)
        return destination

    async def get_flat_files_list(self, workspace: str, fileset: str) -> list[str]:
        return await self.files.fsspec._find(f"{workspace}/{fileset}/")
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

**After (typed client):**
```python
try:
    files.fsspec.ls("my-workspace/my-fileset/")
except FileNotFoundError:
    print("Fileset or path not found")
except PermissionError:
    print("Access denied")
except Exception as e:
    print(f"Error: {e}")
```

---

## Advanced: Standalone FilesetFileSystem

If you need a `FilesetFileSystem` instance without going through `FilesResource` (e.g., for custom configuration), construct it directly from a `FilesClient`:

```python
from filesets import FilesetFileSystem
from nemo_helix_plugin.files.client import FilesClient

files_client = FilesClient(base_url="http://nhx-host")
fs = FilesetFileSystem(client=files_client)
```

## Protocol Registration

To use `fileset://` URLs with fsspec or libraries that support fsspec URLs:

```python
from filesets import FilesetFileSystem

# Register the fileset:// protocol globally with fsspec
FilesetFileSystem.register_fsspec()

# Now you can use fileset:// URLs with fsspec
import fsspec
fs = fsspec.filesystem("fileset", client=files_client)

# Or with pandas (after registration)
import pandas as pd
df = pd.read_parquet("fileset://my-workspace/my-fileset/data.parquet", storage_options={"client": files_client})
```
