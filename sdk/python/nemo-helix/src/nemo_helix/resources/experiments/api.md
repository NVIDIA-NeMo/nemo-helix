<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experiments

Types:

```python
from nemo_helix.types.experiments import (
    ColumnLayout,
    ExperimentFilter,
    ExperimentRequest,
    ExperimentResponse,
    ExperimentResponsesPage,
    ExperimentUpdateRequest,
    ParetoConfig,
)
```

Methods:

- <code title="post /apis/intake/v2/workspaces/{workspace}/experiments">client.experiments.<a href="./src/nemo_helix/resources/experiments/experiments.py">create</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/experiments/experiment_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/experiments/experiment_response.py">ExperimentResponse</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/experiments/{name}">client.experiments.<a href="./src/nemo_helix/resources/experiments/experiments.py">retrieve</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/experiments/experiment_response.py">ExperimentResponse</a></code>
- <code title="put /apis/intake/v2/workspaces/{workspace}/experiments/{name}">client.experiments.<a href="./src/nemo_helix/resources/experiments/experiments.py">update</a>(path_name, \*, workspace, \*\*<a href="src/nemo_helix/types/experiments/experiment_update_params.py">params</a>) -> <a href="./src/nemo_helix/types/experiments/experiment_response.py">ExperimentResponse</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/experiments">client.experiments.<a href="./src/nemo_helix/resources/experiments/experiments.py">list</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/experiments/experiment_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/experiments/experiment_response.py">SyncDefaultPagination[ExperimentResponse]</a></code>
- <code title="delete /apis/intake/v2/workspaces/{workspace}/experiments/{name}">client.experiments.<a href="./src/nemo_helix/resources/experiments/experiments.py">delete</a>(name, \*, workspace) -> None</code>
