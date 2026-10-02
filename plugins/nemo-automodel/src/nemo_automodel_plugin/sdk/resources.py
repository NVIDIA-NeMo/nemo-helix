# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automodel contributor SDK resources (composed by ``nemo-customizer-plugin``).

Thin shim over the shared :func:`nhx.customization_common.sdk.client.make_customization_sdk`
factory. The ``AutomodelCustomization`` / ``AsyncAutomodelCustomization`` symbols below
are imported by string by the ``nemo-customizer`` SDK hub and must not move.
"""

from nhx.customization_common.sdk.client import make_customization_sdk

AutomodelCustomization, AsyncAutomodelCustomization = make_customization_sdk("automodel")

# Jobs-resource classes re-exported for ``sdk/__init__.py`` and backward compatibility.
AutomodelJobsResource = AutomodelCustomization.jobs_resource_cls
AsyncAutomodelJobsResource = AsyncAutomodelCustomization.jobs_resource_cls
