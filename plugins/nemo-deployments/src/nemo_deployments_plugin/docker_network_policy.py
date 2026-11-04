# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation for client-selected primary Docker networks, not internal sidecars."""


def validate_client_docker_network(network: str | None) -> None:
    """Require a canonical network name and prohibit namespace-sharing modes."""
    if network is None:
        return
    if not network or network != network.strip() or network.lower() == "host" or ":" in network:
        raise ValueError("Docker network override must be a network name, not a namespace-sharing mode")
