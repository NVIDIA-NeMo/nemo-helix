<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Models

Types:

```python
from nemo_helix.types.models import (
    Adapter,
    BaseModelFilter,
    CreateModelEntityRequest,
    FinetuningTypeFilter,
    ModelEntity,
    ModelEntityFilter,
    ModelEntitySortField,
    ModelEntitysPage,
    UpdateModelEntityRequest,
)
```

Methods:

- <code title="post /apis/models/v2/workspaces/{workspace}/models">client.models.<a href="./src/nemo_helix/resources/models/models.py">create</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/models/model_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/models/model_entity.py">ModelEntity</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/models/{name}">client.models.<a href="./src/nemo_helix/resources/models/models.py">retrieve</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/models/model_retrieve_params.py">params</a>) -> <a href="./src/nemo_helix/types/models/model_entity.py">ModelEntity</a></code>
- <code title="patch /apis/models/v2/workspaces/{workspace}/models/{name}">client.models.<a href="./src/nemo_helix/resources/models/models.py">update</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/models/model_update_params.py">params</a>) -> <a href="./src/nemo_helix/types/models/model_entity.py">ModelEntity</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/models">client.models.<a href="./src/nemo_helix/resources/models/models.py">list</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/models/model_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/models/model_entity.py">SyncDefaultPagination[ModelEntity]</a></code>
- <code title="delete /apis/models/v2/workspaces/{workspace}/models/{name}">client.models.<a href="./src/nemo_helix/resources/models/models.py">delete</a>(name, \*, workspace) -> None</code>

## Adapters

Types:

```python
from nemo_helix.types.models import (
    CreateAdapterRequest,
    CreateModelAdapterRequest,
    Lora,
    UpdateAdapterRequest,
)
```

Methods:

- <code title="post /apis/models/v2/workspaces/{workspace}/models/{model_name}/adapters">client.models.adapters.<a href="./src/nemo_helix/resources/models/adapters.py">create</a>(model_name, \*, workspace, \*\*<a href="src/nemo_helix/types/models/adapter_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/models/adapter.py">Adapter</a></code>
- <code title="patch /apis/models/v2/workspaces/{workspace}/models/{model_name}/adapters/{adapter}">client.models.adapters.<a href="./src/nemo_helix/resources/models/adapters.py">update</a>(adapter, \*, workspace, model_name, \*\*<a href="src/nemo_helix/types/models/adapter_update_params.py">params</a>) -> <a href="./src/nemo_helix/types/models/adapter.py">Adapter</a></code>
- <code title="delete /apis/models/v2/workspaces/{workspace}/models/{model_name}/adapters/{adapter}">client.models.adapters.<a href="./src/nemo_helix/resources/models/adapters.py">delete</a>(adapter, \*, workspace, model_name) -> None</code>
