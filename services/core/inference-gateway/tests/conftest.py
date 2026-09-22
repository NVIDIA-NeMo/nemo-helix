# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for Inference Gateway service tests."""

# TODO: Re-enable blockbuster once Docker SDK blocking calls are fixed.
# Integration tests use the models service Docker backend which makes blocking
# SSL calls (ssl.SSLSocket.send). Need to either wrap those calls.
# from nhx.testing.blockbuster import blockbuster_fixture
# blockbuster = blockbuster_fixture(autouse=True)
