# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for server configuration and setup."""

from nhx.core.files.service import FilesService


def test_create_app():
    """Test that FilesService creates a FastAPI application."""
    service = FilesService()
    assert service.app is not None
    assert service.app.title == "Files Service"
    assert service.app.version is not None
