# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.auth.permissions import ALL_WORKSPACES


def test_workspace_selector_is_the_public_api_selector() -> None:
    assert ALL_WORKSPACES == "-"
