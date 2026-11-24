# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automodel contributor SDK resources (composed by ``nemo-customizer-plugin``).

Thin shim over the shared :func:`nhx.customization_common.sdk.client.make_customization_sdk`
factory. The ``AutomodelCustomization`` / ``AsyncAutomodelCustomization`` symbols below
are imported by string by the ``nemo-customizer`` SDK hub and must not move.
"""

from nhx.customization_common.sdk.client import (
    AsyncCustomizationBackendResource,
    AsyncCustomizationSDKContext,
    CustomizationBackendResource,
    CustomizationSDKContext,
    make_customization_sdk,
)

_Training, _AsyncTraining = make_customization_sdk("automodel")
# Merge jobs have their own collection (``automodel-merge/jobs``): ``client.customization.automodel.merge_jobs``.
_Merge, _AsyncMerge = make_customization_sdk("automodel-merge")


class AutomodelCustomization(CustomizationBackendResource):
    backend = "automodel"
    jobs_resource_cls = _Training.jobs_resource_cls

    def __init__(self, context: CustomizationSDKContext) -> None:
        super().__init__(context)
        self.merge_jobs = _Merge.jobs_resource_cls(context)


class AsyncAutomodelCustomization(AsyncCustomizationBackendResource):
    backend = "automodel"
    jobs_resource_cls = _AsyncTraining.jobs_resource_cls

    def __init__(self, context: AsyncCustomizationSDKContext) -> None:
        super().__init__(context)
        self.merge_jobs = _AsyncMerge.jobs_resource_cls(context)


# Jobs-resource classes re-exported for ``sdk/__init__.py`` and backward compatibility.
AutomodelJobsResource = AutomodelCustomization.jobs_resource_cls
AsyncAutomodelJobsResource = AsyncAutomodelCustomization.jobs_resource_cls
