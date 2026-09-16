# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity store client for plugin authors.

Plugin authors should import from here:

    from nemo_helix_plugin.entity_client import (
        NemoEntitiesClient,
        NemoEntityNotFoundError,
        NemoEntityConflictError,
        NemoPaginationInfo,
        get_entity_client,
    )

:class:`NemoEntitiesClient` is the primary interface for CRUD operations against
the NeMo Helix entity store.  :class:`NemoEntityNotFoundError` and
:class:`NemoEntityConflictError` are the two exceptions you will handle in
virtually every route handler:

    try:
        item = await entity_client.get(Widget, name=name, workspace=workspace)
    except NemoEntityNotFoundError:
        raise HTTPException(status_code=404, detail=f"Widget '{name}' not found.")
    except NemoEntityConflictError:
        raise HTTPException(status_code=409, detail=f"Widget '{name}' already exists.")

:func:`get_entity_client` is a FastAPI ``Depends()`` placeholder — the platform
injects the real implementation at startup via ``app.dependency_overrides``.
"""

from nemo_helix_plugin.dependencies import get_entity_client
from nemo_helix_plugin.entities import (
    AnyEntityDeleteClientProtocol as NemoAnyEntityDeleteClientProtocol,
)
from nemo_helix_plugin.entities import (
    AnyEntityGetterProtocol as NemoAnyEntityGetterProtocol,
)
from nemo_helix_plugin.entities import (
    EntityClient as NemoEntitiesClient,
)
from nemo_helix_plugin.entities import (
    EntityClientProtocol as NemoEntitiesClientProtocol,
)
from nemo_helix_plugin.entities import (
    EntityConflictError as NemoEntityConflictError,
)
from nemo_helix_plugin.entities import (
    EntityDeleteClientProtocol as NemoEntityDeleteClientProtocol,
)
from nemo_helix_plugin.entities import (
    EntityGetterProtocol as NemoEntityGetterProtocol,
)
from nemo_helix_plugin.entities import (
    EntityNotFoundError as NemoEntityNotFoundError,
)
from nemo_helix_plugin.entities import (
    EntityUpdateClientProtocol as NemoEntityUpdateClientProtocol,
)
from nemo_helix_plugin.entities import (
    EntityValidationError as NemoEntityValidationError,
)
from nemo_helix_plugin.entities import (
    PaginationInfo as NemoPaginationInfo,
)

__all__ = [
    "NemoEntitiesClient",
    "NemoEntitiesClientProtocol",
    "NemoAnyEntityDeleteClientProtocol",
    "NemoEntityUpdateClientProtocol",
    "NemoAnyEntityGetterProtocol",
    "NemoEntityDeleteClientProtocol",
    "NemoEntityGetterProtocol",
    "NemoEntityConflictError",
    "NemoEntityNotFoundError",
    "NemoEntityValidationError",
    "NemoPaginationInfo",
    "get_entity_client",
]
