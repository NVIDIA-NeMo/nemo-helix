<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Entities

Types:

```python
from nemo_helix.types.entities import EntitiesPage, Entity, EntityCreateParam, EntityUpdate
```

Methods:

- <code title="post /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}">client.entities.<a href="./src/nemo_helix/resources/entities/entities.py">create</a>(entity_type, \*, workspace, \*\*<a href="src/nemo_helix/types/entities/entity_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/entities/entity.py">Entity</a></code>
- <code title="get /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}">client.entities.<a href="./src/nemo_helix/resources/entities/entities.py">list</a>(entity_type, \*, workspace, \*\*<a href="src/nemo_helix/types/entities/entity_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/entities/entity.py">SyncDefaultPagination[Entity]</a></code>
- <code title="delete /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}">client.entities.<a href="./src/nemo_helix/resources/entities/entities.py">delete_entity_by_name</a>(name, \*, workspace, entity_type, \*\*<a href="src/nemo_helix/types/entities/entity_delete_entity_by_name_params.py">params</a>) -> <a href="./src/nemo_helix/types/shared/delete_response.py">DeleteResponse</a></code>
- <code title="get /apis/entities/v2/entities/{id}">client.entities.<a href="./src/nemo_helix/resources/entities/entities.py">get_entity_by_id</a>(id) -> <a href="./src/nemo_helix/types/entities/entity.py">Entity</a></code>
- <code title="get /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}">client.entities.<a href="./src/nemo_helix/resources/entities/entities.py">get_entity_by_name</a>(name, \*, workspace, entity_type, \*\*<a href="src/nemo_helix/types/entities/entity_get_entity_by_name_params.py">params</a>) -> <a href="./src/nemo_helix/types/entities/entity.py">Entity</a></code>
- <code title="put /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}">client.entities.<a href="./src/nemo_helix/resources/entities/entities.py">update_entity_by_name</a>(name, \*, workspace, entity_type, \*\*<a href="src/nemo_helix/types/entities/entity_update_entity_by_name_params.py">params</a>) -> <a href="./src/nemo_helix/types/entities/entity.py">Entity</a></code>
