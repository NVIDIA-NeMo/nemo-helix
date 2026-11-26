# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder build-set names for task images; submitting and following them is :mod:`nemo_builder_plugin.builds`."""

from __future__ import annotations

import hashlib
import json

_BUILDER_NAME_PREFIX = "harbor-env-"
_COPY_NAME_PREFIX = "harbor-img-"


def builder_name(*, context_digest: str, dockerfile: str, platform: str | None = None) -> str:
    """Builder build-set name for an image built from a context holding ``context_digest``'s files, for one platform.

    Keyed by content, not by which task archive holds it: tasks whose build contexts hold the same files share
    one build (see :func:`nemo_builder_plugin.builds.adopt_or_build`). A build for the deployment's default
    platform leaves ``platform`` out of the key.
    """
    key = {"v": 3, "context": context_digest, "dockerfile": dockerfile}
    return f"{_BUILDER_NAME_PREFIX}{_key_hash(key, platform)}"


def copy_builder_name(image: str, platform: str | None = None) -> str:
    """Builder build-set name for a copy of ``image``, by its fully qualified reference, for one platform."""
    return f"{_COPY_NAME_PREFIX}{_key_hash({'v': 1, 'image': image}, platform)}"


def _key_hash(key: dict[str, object], platform: str | None) -> str:
    if platform is not None:
        key = {**key, "platform": platform}
    return hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
