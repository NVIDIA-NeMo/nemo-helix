<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluations

Types:

```python
from nemo_helix.types.evaluations import (
    EvaluationFilter,
    EvaluationPatchRequest,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResponsesPage,
    EvaluatorAggregate,
    MetricStatFilters,
    NumberFilter,
)
```

Methods:

- <code title="post /apis/intake/v2/workspaces/{workspace}/evaluations">client.evaluations.<a href="./src/nemo_helix/resources/evaluations/evaluations.py">create</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/evaluations/evaluation_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/evaluations/evaluation_response.py">EvaluationResponse</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/evaluations/{name}">client.evaluations.<a href="./src/nemo_helix/resources/evaluations/evaluations.py">retrieve</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/evaluations/evaluation_response.py">EvaluationResponse</a></code>
- <code title="put /apis/intake/v2/workspaces/{workspace}/evaluations/{name}">client.evaluations.<a href="./src/nemo_helix/resources/evaluations/evaluations.py">update</a>(path_name, \*, workspace, \*\*<a href="src/nemo_helix/types/evaluations/evaluation_update_params.py">params</a>) -> <a href="./src/nemo_helix/types/evaluations/evaluation_response.py">EvaluationResponse</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/evaluations">client.evaluations.<a href="./src/nemo_helix/resources/evaluations/evaluations.py">list</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/evaluations/evaluation_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/evaluations/evaluation_response.py">SyncDefaultPagination[EvaluationResponse]</a></code>
- <code title="delete /apis/intake/v2/workspaces/{workspace}/evaluations/{name}">client.evaluations.<a href="./src/nemo_helix/resources/evaluations/evaluations.py">delete</a>(name, \*, workspace) -> None</code>
- <code title="patch /apis/intake/v2/workspaces/{workspace}/evaluations/{name}">client.evaluations.<a href="./src/nemo_helix/resources/evaluations/evaluations.py">patch</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/evaluations/evaluation_patch_params.py">params</a>) -> <a href="./src/nemo_helix/types/evaluations/evaluation_response.py">EvaluationResponse</a></code>
- <code title="post /apis/intake/v2/workspaces/{workspace}/evaluations/{name}/pin">client.evaluations.<a href="./src/nemo_helix/resources/evaluations/evaluations.py">pin</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/evaluations/evaluation_response.py">EvaluationResponse</a></code>
- <code title="delete /apis/intake/v2/workspaces/{workspace}/evaluations/{name}/pin">client.evaluations.<a href="./src/nemo_helix/resources/evaluations/evaluations.py">unpin</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/evaluations/evaluation_response.py">EvaluationResponse</a></code>

## Sessions

Types:

```python
from nemo_helix.types.evaluations import (
    EvaluationSessionFilter,
    EvaluationSessionResponse,
    EvaluationSessionResponsesPage,
)
```

Methods:

- <code title="get /apis/intake/v2/workspaces/{workspace}/evaluations/{name}/sessions">client.evaluations.sessions.<a href="./src/nemo_helix/resources/evaluations/sessions.py">list</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/evaluations/session_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/evaluations/evaluation_session_response.py">SyncDefaultPagination[EvaluationSessionResponse]</a></code>
