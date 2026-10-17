# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-local container routing for registered Harbor agents."""

import copy
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

from nemo_agents_plugin.runner.deployments_backend import rewrite_fabric_config_base_urls
from nemo_evals.jobs.harbor_forwarder import _docker_bridge_host, loopback_forwarder
from nemo_helix_plugin.config import LOOPBACK_ADDRESSES


def _uses_local_platform(fabric: dict[str, Any]) -> bool:
    models = (fabric.get("models") or {}).values()
    urls = [model.get("base_url", "") for model in models if isinstance(model, dict)]
    relay = fabric.get("relay") or {}
    atif = (relay.get("observability") or {}).get("atif") or {}
    urls.extend(storage.get("endpoint", "") for storage in (atif.get("storage") or []) if isinstance(storage, dict))
    return any(
        isinstance(url, str)
        and urlsplit(url).hostname in (*LOOPBACK_ADDRESSES, "host.docker.internal")
        and urlsplit(url).path.startswith(("/apis/inference-gateway/", "/apis/intake/"))
        for url in urls
    )


@contextmanager
def harbor_gateway_config(config: dict[str, Any], *, platform_base_url: str) -> Iterator[dict[str, Any]]:
    """Route a registered Harbor agent to a Linux host's loopback-only platform.

    Docker Desktop supplies host routing on macOS. Native Linux Docker requires
    a listener on its bridge interface; the platform itself stays loopback-only.
    The ephemeral endpoint belongs only to the worker's runtime config, not the
    persisted evaluation spec.
    """
    target = config.get("target") or {}
    fabric = (target.get("agent_kwargs") or {}).get("fabric_config")
    if (
        config.get("trials") is not None
        or target.get("kind") != "harbor"
        or not isinstance(fabric, dict)
        or not _uses_local_platform(fabric)
        or platform.system() != "Linux"
        or urlsplit(platform_base_url).hostname not in LOOPBACK_ADDRESSES
    ):
        yield config
        return
    with loopback_forwarder(platform_base_url, _docker_bridge_host()) as gateway:
        reachable = copy.deepcopy(config)
        reachable["target"]["agent_kwargs"]["fabric_config"] = rewrite_fabric_config_base_urls(fabric, gateway)
        yield reachable
