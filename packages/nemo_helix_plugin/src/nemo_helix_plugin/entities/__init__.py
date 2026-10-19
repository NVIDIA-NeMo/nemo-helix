# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entities client package.

The rich, generic ``EntityClient`` abstraction and its supporting types live in
:mod:`nemo_helix_plugin.entities.base`. They are re-exported here so the
historical ``from nemo_helix_plugin.entities import ...`` surface keeps
working unchanged.

The low-level NemoClient typed layer lives alongside it:

- :mod:`nemo_helix_plugin.entities.types` — request/response wire models
- :mod:`nemo_helix_plugin.entities.endpoints` — the HTTP contract
- :mod:`nemo_helix_plugin.entities.client` — ``EntitiesClient`` / ``AsyncEntitiesClient``

Import those by their fully-qualified module path.
"""

from nemo_helix_plugin.entities.base import (
    BASE_FIELDS as BASE_FIELDS,
)
from nemo_helix_plugin.entities.base import (
    DEFAULT_WORKSPACE as DEFAULT_WORKSPACE,
)
from nemo_helix_plugin.entities.base import (
    ID_PATTERN as ID_PATTERN,
)
from nemo_helix_plugin.entities.base import (
    AnyEntityDeleteClientProtocol as AnyEntityDeleteClientProtocol,
)
from nemo_helix_plugin.entities.base import (
    AnyEntityGetterProtocol as AnyEntityGetterProtocol,
)
from nemo_helix_plugin.entities.base import (
    EntityBase as EntityBase,
)
from nemo_helix_plugin.entities.base import (
    EntityClient as EntityClient,
)
from nemo_helix_plugin.entities.base import (
    EntityClientProtocol as EntityClientProtocol,
)
from nemo_helix_plugin.entities.base import (
    EntityConflictError as EntityConflictError,
)
from nemo_helix_plugin.entities.base import (
    EntityDeleteClientProtocol as EntityDeleteClientProtocol,
)
from nemo_helix_plugin.entities.base import (
    EntityGetterProtocol as EntityGetterProtocol,
)
from nemo_helix_plugin.entities.base import (
    EntityNotFoundError as EntityNotFoundError,
)
from nemo_helix_plugin.entities.base import (
    EntityStoreError as EntityStoreError,
)
from nemo_helix_plugin.entities.base import (
    EntityT as EntityT,
)
from nemo_helix_plugin.entities.base import (
    EntityToken as EntityToken,
)
from nemo_helix_plugin.entities.base import (
    EntityTypeDefault as EntityTypeDefault,
)
from nemo_helix_plugin.entities.base import (
    EntityTypeLike as EntityTypeLike,
)
from nemo_helix_plugin.entities.base import (
    EntityUpdateClientProtocol as EntityUpdateClientProtocol,
)
from nemo_helix_plugin.entities.base import (
    EntityValidationError as EntityValidationError,
)
from nemo_helix_plugin.entities.base import (
    ListResponse as ListResponse,
)
from nemo_helix_plugin.entities.base import (
    PaginationInfo as PaginationInfo,
)
from nemo_helix_plugin.entities.base import (
    SyncEntityClient as SyncEntityClient,
)
from nemo_helix_plugin.entities.base import (
    _convert_filter_obj_to_filter_str as _convert_filter_obj_to_filter_str,
)
from nemo_helix_plugin.entities.base import (
    _convert_sort_to_api_sort as _convert_sort_to_api_sort,
)
from nemo_helix_plugin.entities.base import (
    _get_entity_type as _get_entity_type,
)
from nemo_helix_plugin.entities.base import (
    parse_qualified_name as parse_qualified_name,
)
