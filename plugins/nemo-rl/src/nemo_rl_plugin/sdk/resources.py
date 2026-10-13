# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo-RL contributor SDK resources (composed by ``nemo-customizer-plugin``).

Thin shim over the shared :func:`nhx.customization_common.sdk.client.make_customization_sdk`
factory. ``RlCustomization`` / ``AsyncRlCustomization`` are imported by string by
the SDK hub and must not move.
"""

from nhx.customization_common.sdk.client import make_customization_sdk

RlCustomization, AsyncRlCustomization = make_customization_sdk("rl")

# Jobs-resource classes re-exported for ``sdk/__init__.py`` and backward compatibility.
RlJobsResource = RlCustomization.jobs_resource_cls
AsyncRlJobsResource = AsyncRlCustomization.jobs_resource_cls
