# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build hook for dynamically bundled workspace packages."""

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from nhx_build_tools.hatch import apply_bundle_force_include, rewrite_bundled_dependencies_in_wheel


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if version == "editable":
            return

        apply_bundle_force_include(self.root, build_data)

    def finalize(self, version, build_data, artifact_path):
        if version == "editable":
            return

        rewrite_bundled_dependencies_in_wheel(artifact_path, self.root, self.metadata.core_raw_metadata)
