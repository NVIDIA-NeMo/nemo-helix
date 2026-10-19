# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-memory Secrets service."""

import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nhx.testing.secrets import InMemorySecretsService


@pytest.mark.asyncio
async def test_serves_secret_values_and_records_lookups() -> None:
    service = InMemorySecretsService({("team-a", "judge-key"): "sk-test"})
    secrets = service.client()

    access = (await secrets.access_secret(workspace="team-a", name="judge-key")).data()
    with pytest.raises(NotFoundError):
        await secrets.access_secret(name="absent")

    assert access.value == "sk-test"
    assert service.lookups == [("team-a", "judge-key"), ("default", "absent")]
