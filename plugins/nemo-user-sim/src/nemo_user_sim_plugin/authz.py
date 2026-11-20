# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OAuth scope the UserSim plugin owns, shared by the service and its job routes."""

from nemo_helix_plugin.authz import AuthzScope

scope = AuthzScope("user-sim")
