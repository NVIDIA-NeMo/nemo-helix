# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin seed interface — what plugin authors implement for platform seeding.

Plugin authors subclass :class:`NemoSeedJob` and register the class under the
``nemo.seed`` entry-point group. The platform calls :meth:`run` when services
start up. This is useful to automatically create entities that should always
exist in the system.

An `AsyncNeMoPlatform` SDK instance and a `NemoEntitiesClient` instance are available as
attributes on every `NemoSeedJob` instance. Any services called with these objects
must be running and responding to requests; to indicate these services are
required, add them to the `dependencies` ClassVar.

Seed jobs are executed every time the services are restarted, so job implementations
MUST be idempotent.

Example::

    # my_plugin/seed.py
    from nemo_platform_plugin.entity_client import NemoEntityConflictError
    from nemo_platform_plugin.seed import NemoSeedJob

    _SEEDED_ITEM_NAME = "foo"

    class MySeedJob(NemoSeedJob):
        name = "my-plugin"
        dependencies = ["entities"]

        async def run(self) -> None:
            item = MyPluginItem(name=_SEEDED_ITEM_NAME)
            try:
                await self.entities_client.create(item)
                logger.info("Created example seed entity %r", _SEEDED_ITEM_NAME)
            except NemoEntityConflictError:
                logger.debug("Example seed entity %r already exists", _SEEDED_ITEM_NAME)


    # pyproject.toml:
    # [project.entry-points."nemo.seed"]
    # my-plugin = "my_plugin.seed:MySeedJob"
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin._base import _NamedPlugin
from nemo_platform_plugin.entity_client import NemoEntitiesClient


class NemoSeedJob(_NamedPlugin):
    """Abstract base class for plugin-contributed seed jobs.

    The platform instantiates subclasses, injects ``sdk`` and ``entities_client``,
    then awaits :meth:`run`.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""
    dependencies: ClassVar[list[str]] = []

    sdk: AsyncNeMoPlatform
    entities_client: NemoEntitiesClient

    @abstractmethod
    async def run(self) -> None:
        """Execute the seed job. Implementations must be idempotent."""
