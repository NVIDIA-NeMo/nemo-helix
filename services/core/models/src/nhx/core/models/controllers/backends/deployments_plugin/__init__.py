# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Models backend configuration for the deployments-plugin substrate.

Only the config model is exported from this package. The concrete
``DeploymentsPluginServiceBackend`` lives in ``backend`` and is imported
lazily from ``registry`` so the models service wheel does not require
``nemo_deployments_plugin`` unless that backend is selected.
"""

from .config import DeploymentsPluginBackendConfigModel as DeploymentsPluginBackendConfigModel

__all__ = ["DeploymentsPluginBackendConfigModel"]
