# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.core.models.controllers.backends.deployments_plugin.naming import entity_names


def test_entity_names_have_expected_suffixes() -> None:
    names = entity_names("my-dep")
    assert names.volume == "my-dep-weights"
    assert names.scratch == "my-dep-scratch"
    assert names.puller == "my-dep-puller"
    assert names.server == "my-dep-server"
