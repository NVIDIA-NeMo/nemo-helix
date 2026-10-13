# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed job for the example plugin."""

from __future__ import annotations

import logging

from nemo_example_plugin.entities import ExampleItem
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.seed import NemoSeedJob

logger = logging.getLogger(__name__)

_SEEDED_ITEM_NAME = "seeded-example-item"


class ExampleSeedJob(NemoSeedJob):
    """Ensure the example plugin has one default entity to demonstrate plugin seeding."""

    name = "example"
    description = "Create the default example plugin item"
    dependencies = ["entities"]

    async def run(self) -> None:
        item = ExampleItem(
            name=_SEEDED_ITEM_NAME,
            workspace="system",
            title="Seeded example item",
            body="Created by the example plugin seed job.",
            tags=["seeded", "example"],
        )
        try:
            await self.entities_client.create(item)
            logger.info("Created example seed entity %r", _SEEDED_ITEM_NAME)
        except NemoEntityConflictError:
            logger.debug("Example seed entity %r already exists", _SEEDED_ITEM_NAME)
