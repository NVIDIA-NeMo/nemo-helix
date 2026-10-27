# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declare protected response fields without importing the platform runtime."""

from collections.abc import Mapping

from fastapi.routing import APIRoute


class CreatorSnapshotRoute(APIRoute):
    policy_purpose: str
    response_paths: Mapping[object, tuple[str, ...]]


def creator_snapshot_route_class(*, response_paths: Mapping[object, tuple[str, ...]], purpose: str) -> type[APIRoute]:
    """Build a route class that projects only registered response models."""
    if not purpose or not response_paths:
        raise ValueError("Creator snapshot projection needs a purpose and response models")
    registrations = dict(response_paths)

    class RegisteredCreatorSnapshotRoute(CreatorSnapshotRoute):
        policy_purpose = purpose
        response_paths = registrations

    return RegisteredCreatorSnapshotRoute
