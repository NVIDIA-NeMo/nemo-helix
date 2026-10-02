# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for evaluator metric reference resolution."""

from __future__ import annotations

import pytest
from nemo_evaluator.jobs.metric_resolution import HelixMetricSecretResolver
from nemo_evaluator_sdk.values.common import SecretRef


@pytest.mark.asyncio
async def test_resolver_returns_the_secret_value(make_secrets_client) -> None:
    client = make_secrets_client({("default", "judge-key"): "sk-live-abc"})
    resolver = HelixMetricSecretResolver(client, workspace="default")

    assert await resolver.resolve_secret(SecretRef(root="judge-key")) == "sk-live-abc"


@pytest.mark.asyncio
async def test_unqualified_ref_resolves_in_the_request_workspace(make_secrets_client) -> None:
    """An unqualified name must not silently fall back to the client's own default workspace."""
    client = make_secrets_client({("team-a", "judge-key"): "sk-team-a"})
    resolver = HelixMetricSecretResolver(client, workspace="team-a")

    assert await resolver.resolve_secret(SecretRef(root="judge-key")) == "sk-team-a"
    assert client.lookups == [("team-a", "judge-key")]


@pytest.mark.asyncio
async def test_qualified_ref_resolves_in_the_named_workspace(make_secrets_client) -> None:
    """A workspace-qualified ref is honoured; the Secrets service authorizes it, not this resolver."""
    client = make_secrets_client({("team-b", "judge-key"): "sk-team-b"})
    resolver = HelixMetricSecretResolver(client, workspace="team-a")

    assert await resolver.resolve_secret(SecretRef(root="team-b/judge-key")) == "sk-team-b"
    assert client.lookups == [("team-b", "judge-key")]


@pytest.mark.asyncio
async def test_missing_secret_resolves_to_none_rather_than_raising(make_secrets_client) -> None:
    """`None` is the protocol's answer for "no such secret"; the metric turns it into the error."""
    client = make_secrets_client()
    resolver = HelixMetricSecretResolver(client, workspace="default")

    assert await resolver.resolve_secret(SecretRef(root="absent")) is None


@pytest.mark.asyncio
async def test_malformed_ref_is_rejected(make_secrets_client) -> None:
    client = make_secrets_client()
    resolver = HelixMetricSecretResolver(client, workspace="default")

    with pytest.raises(ValueError):
        await resolver.resolve_secret(SecretRef(root="too/many/parts"))
    assert client.lookups == []
