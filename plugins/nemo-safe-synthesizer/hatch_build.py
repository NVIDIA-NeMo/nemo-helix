# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from nhx_build_tools.hatch import (
    apply_bundle_force_include,
    rewrite_bundled_dependencies_in_wheel,
)


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if version == "editable":
            return

        apply_bundle_force_include(self.root, build_data)

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        if version == "editable":
            return

        rewrite_bundled_dependencies_in_wheel(artifact_path, self.root, self.metadata.core_raw_metadata)
