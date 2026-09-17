# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.secrets.compat import (
    SecretsResource as SecretsResource,
    AsyncSecretsResource as AsyncSecretsResource,
    SecretsAdminResource as SecretsAdminResource,
    AsyncSecretsAdminResource as AsyncSecretsAdminResource,
)

AdminResource = SecretsAdminResource
AsyncAdminResource = AsyncSecretsAdminResource
