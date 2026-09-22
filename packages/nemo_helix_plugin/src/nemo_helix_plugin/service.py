# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin service interface — what plugin authors implement.

Plugin authors subclass :class:`NemoService` and register a module-level
instance under the ``nemo.services`` entry-point group.  The platform wraps
each instance in a :class:`NemoServiceAdapter` at startup so it can be
treated as a first-class internal :class:`~nhx.common.service.Service`.

Example::

    # my_plugin/service.py
    from fastapi import APIRouter
    from nemo_helix_plugin.service import NemoService, RouterSpec

    class MyService(NemoService):
        name = "my-plugin"
        dependencies = ["entities"]

        def get_routers(self) -> list[RouterSpec]:
            router = APIRouter()

            @router.get("/health")
            async def health() -> dict[str, str]:
                return {"status": "ok"}

            return [RouterSpec(router, tag="My Plugin")]

    # pyproject.toml:
    # [project.entry-points."nemo.services"]
    # my-plugin = "my_plugin.service:MyService"
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import ClassVar

from fastapi import APIRouter
from nemo_helix_plugin._base import _NamedPlugin
from nemo_helix_plugin.authz import Permission
from starlette.requests import Request
from starlette.responses import Response

ExceptionHandler = Callable[[Request, Exception], Response | Awaitable[Response]]


@dataclass
class RouterSpec:
    """Minimal router descriptor contributed by a plugin.

    The platform adapter converts this to its internal ``RouterConfig`` at
    startup.  Plugin authors never import ``RouterConfig`` directly.

    Attributes:
        router: The FastAPI router to mount.
        tag: OpenAPI tag grouping.  Defaults to the plugin name if left empty.
        description: Human-readable description shown in the OpenAPI docs.
        prefix: URL prefix appended after ``/apis/<plugin-name>``.  Rarely
            needed — leave empty unless you have multiple routers with
            overlapping paths.
    """

    router: APIRouter
    tag: str = field(default="")
    description: str = field(default="")
    prefix: str = field(default="")


class NemoService(_NamedPlugin):
    """Abstract base class for plugin-contributed services.

    Subclasses declare their identity via class variables and implement
    :meth:`get_routers`.  The platform instantiates and wraps them at
    startup — plugin authors do not call the platform directly.

    Class variables:

    .. attribute:: name
        :type: str

        Unique kebab-case service name.  Must match the ``nemo.services``
        entry-point key and becomes the URL prefix (``/apis/<name>``).

    .. attribute:: dependencies
        :type: list[str]

        Names of platform services that must start before this one.
        Defaults to ``[]``.
    """

    name: ClassVar[str]
    dependencies: ClassVar[list[str]] = []

    @abstractmethod
    def get_routers(self) -> list[RouterSpec]:
        """Return the routers this service contributes.

        Each :class:`RouterSpec` is mounted at ``/apis/<name>/<spec.prefix>``.
        """

    async def on_startup(self) -> None:
        """Called at platform startup before the HTTP server accepts requests.

        Override for custom initialisation (e.g. connecting to a database).
        The default implementation does nothing.
        """

    async def on_shutdown(self) -> None:
        """Called at platform shutdown after requests stop being served.

        Override for cleanup (e.g. closing connections).
        The default implementation does nothing.
        """

    def extra_permissions(self) -> list[Permission]:
        """Permissions this service owns that are *not* attached to a route.

        The permission catalog is normally derived entirely from the
        :func:`~nemo_helix_plugin.authz.path_rule` rules on ``get_routers()``. Override
        this only for permissions with no 1:1 route — e.g. ones checked in middleware, or
        declared ahead of the route that will reference them. These are merged into the
        derived catalog (and its default role grants) alongside the route-derived ones.

        Default: none.
        """
        return []

    def extra_role_permissions(self) -> dict[str, list[Permission]]:
        """Extra ``role -> [permission]`` grants beyond the suffix-based defaults.

        The derivation grants each catalog permission to roles by a suffix heuristic
        (``.list``/``.read`` → Viewer + Editor; everything else → Editor only). Override this
        to grant a permission to a role that heuristic wouldn't reach — e.g. the agent gateway's
        ``.invoke`` permission, which a Viewer must hold to call a deployed agent even though its
        suffix isn't ``read``. Grants are **unioned** with the suffix defaults (never subtractive),
        and every permission must live in this service's own namespace or the whole plugin fails
        closed. Each granted permission is also registered in the catalog, so it need not also
        appear in :meth:`extra_permissions`.

        Default: none.
        """
        return {}

    def get_exception_handlers(self) -> dict[type[Exception], ExceptionHandler]:
        """Return a mapping of exception types to handler functions.

        Each handler receives ``(request, exc)`` and must return a
        :class:`~starlette.responses.Response` (sync or async).

        The platform registers these on the service's FastAPI app so that
        unhandled exceptions of the given types are caught and converted to
        appropriate HTTP responses.

        Default returns an empty dict.
        """
        return {}
