<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Files

Types:

```python
from nemo_platform.types.files import (
    CacheStatus,
    CreateFilesetRequest,
    Fileset,
    FilesetFile,
    FilesetOutputsPage,
    FilesetPurpose,
    GitHubStorageConfig,
    HuggingfaceStorageConfig,
    ListFilesetFilesResponse,
    LocalStorageConfig,
    NGCStorageConfig,
    S3StorageConfig,
    SecretRef,
    StorageConfigType,
    UpdateFilesetRequest,
)
```

Methods:

- <code title="delete /apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}">client.files.<a href="./src/nemo_platform/resources/files/files.py">\_delete_file</a>(path, \*, workspace, name) -> <a href="./src/nemo_platform/types/files/fileset_file.py">FilesetFile</a></code>
- <code title="get /apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}">client.files.<a href="./src/nemo_platform/resources/files/files.py">\_download_file</a>(path, \*, workspace, name) -> BinaryAPIResponse</code>
- <code title="get /apis/files/v2/workspaces/{workspace}/filesets/{name}/files">client.files.<a href="./src/nemo_platform/resources/files/files.py">\_list_files</a>(name, \*, workspace, \*\*<a href="src/nemo_platform/types/files/file_list_files_params.py">params</a>) -> <a href="./src/nemo_platform/types/files/list_fileset_files_response.py">ListFilesetFilesResponse</a></code>
- <code title="put /apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}">client.files.<a href="./src/nemo_platform/resources/files/files.py">\_upload_file</a>(path, body, \*, workspace, name, \*\*<a href="src/nemo_platform/types/files/file_upload_file_params.py">params</a>) -> <a href="./src/nemo_platform/types/files/fileset_file.py">FilesetFile</a></code>

## Filesets

Types:

```python
from nemo_platform.types.files import FilesetFilter
```

Methods:

- <code title="post /apis/files/v2/workspaces/{workspace}/filesets">client.files.filesets.<a href="./src/nemo_platform/resources/files/filesets.py">create</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/files/fileset_create_params.py">params</a>) -> <a href="./src/nemo_platform/types/files/fileset.py">Fileset</a></code>
- <code title="get /apis/files/v2/workspaces/{workspace}/filesets/{name}">client.files.filesets.<a href="./src/nemo_platform/resources/files/filesets.py">retrieve</a>(name, \*, workspace) -> <a href="./src/nemo_platform/types/files/fileset.py">Fileset</a></code>
- <code title="patch /apis/files/v2/workspaces/{workspace}/filesets/{name}">client.files.filesets.<a href="./src/nemo_platform/resources/files/filesets.py">update</a>(name, \*, workspace, \*\*<a href="src/nemo_platform/types/files/fileset_update_params.py">params</a>) -> <a href="./src/nemo_platform/types/files/fileset.py">Fileset</a></code>
- <code title="get /apis/files/v2/workspaces/{workspace}/filesets">client.files.filesets.<a href="./src/nemo_platform/resources/files/filesets.py">list</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/files/fileset_list_params.py">params</a>) -> <a href="./src/nemo_platform/types/files/fileset.py">SyncDefaultPagination[Fileset]</a></code>
- <code title="delete /apis/files/v2/workspaces/{workspace}/filesets/{name}">client.files.filesets.<a href="./src/nemo_platform/resources/files/filesets.py">delete</a>(name, \*, workspace) -> <a href="./src/nemo_platform/types/files/fileset.py">Fileset</a></code>
- <code title="post /apis/files/v2/workspaces/{workspace}/filesets/{name}/refresh">client.files.filesets.<a href="./src/nemo_platform/resources/files/filesets.py">refresh</a>(name, \*, workspace) -> <a href="./src/nemo_platform/types/files/fileset.py">Fileset</a></code>

## Otlp

### Logs

Types:

```python
from nemo_platform.types.files.otlp import (
    LogQueryRequest,
    OtelExportLogsPartialSuccess,
    OtelExportLogsServiceResponse,
)
```

Methods:

- <code title="post /apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs">client.files.otlp.logs.<a href="./src/nemo_platform/resources/files/otlp/logs.py">create</a>(name, \*, workspace, \*\*<a href="src/nemo_platform/types/files/otlp/log_create_params.py">params</a>) -> <a href="./src/nemo_platform/types/files/otlp/otel_export_logs_service_response.py">OtelExportLogsServiceResponse</a></code>
- <code title="post /apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs/query">client.files.otlp.logs.<a href="./src/nemo_platform/resources/files/otlp/logs.py">query</a>(name, \*, workspace, \*\*<a href="src/nemo_platform/types/files/otlp/log_query_params.py">params</a>) -> <a href="./src/nemo_platform/types/shared/platform_job_log_page.py">PlatformJobLogPage</a></code>
