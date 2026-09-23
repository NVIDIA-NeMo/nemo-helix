# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the extension FilesResource sub-resource surface.

The vendored extension ``FilesResource`` replaces the Stainless-generated one on
``NeMoHelix.files``. It must remain API-compatible with the generated resource
by exposing the ``filesets`` and ``otlp`` sub-resources; otherwise auto-generated
CLI commands such as ``nemo files filesets create`` fail at runtime with
``'FilesResource' object has no attribute 'filesets'``.
"""

from __future__ import annotations

from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix.resources.files.filesets import AsyncFilesetsResource, FilesetsResource
from nemo_helix.resources.files.otlp.otlp import AsyncOtlpResource, OtlpResource


def test_sync_files_resource_exposes_filesets_and_otlp() -> None:
    client = NeMoHelix(base_url="http://testserver", workspace="test")

    assert isinstance(client.files.filesets, FilesetsResource)
    assert isinstance(client.files.otlp, OtlpResource)


def test_async_files_resource_exposes_filesets_and_otlp() -> None:
    client = AsyncNeMoHelix(base_url="http://testserver", workspace="test")

    assert isinstance(client.files.filesets, AsyncFilesetsResource)
    assert isinstance(client.files.otlp, AsyncOtlpResource)
