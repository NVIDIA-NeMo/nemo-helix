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

"""Build hook.

- Wheel builds: reads [tool.bundle-package] from pyproject.toml and generates
  force-include mappings dynamically, merged with static wheel force-includes.
  Bundle entries declaring `shared_data` also contribute wheel shared-data, so
  files a bundled package would install under the environment prefix survive
  bundling.
  Also attempts to build the Studio UI bundle so it is picked up by the
  force-include.
- Editable installs: suppresses force-include so workspace packages stay resolved
  via editable .pth entries instead of getting stale snapshots copied into
  site-packages.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from nmp_build_tools.hatch import (
    apply_bundle_force_include,
    apply_bundle_shared_data,
    disable_bundle_force_include_for_editable,
    rewrite_bundled_dependencies_in_wheel,
)


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if version == "editable":
            disable_bundle_force_include_for_editable(build_data)
            return

        # Generate force-include from [tool.bundle-package] for wheel builds.
        # This replaces the hardcoded [tool.hatch.build.targets.wheel.force-include]
        # section that was previously in pyproject.toml.
        apply_bundle_force_include(self.root, build_data)
        # Prefix-installed data files (<sys.prefix>/share/...) that a bundled
        # package would otherwise ship as its own distribution's shared-data.
        apply_bundle_shared_data(self.root, build_data)

        self._build_studio_ui()

    def finalize(self, version, build_data, artifact_path):
        if version == "editable":
            return

        rewrite_bundled_dependencies_in_wheel(artifact_path, self.root, self.metadata.core_raw_metadata)

    def _build_studio_ui(self) -> None:
        repo_root = Path(self.root).resolve().parents[1]
        web_dir = repo_root / "web"
        dist_dir = web_dir / "packages" / "studio" / "dist"

        if not web_dir.is_dir():
            self._skip("web/ not found", dist_dir)
            return

        if shutil.which("node") is None:
            required = _required_engine(web_dir / "package.json", "node")
            hint = f" Install Node.js matching '{required}'." if required else ""
            self._skip(f"node not found on PATH.{hint}", dist_dir)
            return

        if shutil.which("pnpm") is None:
            required = _required_engine(web_dir / "package.json", "pnpm")
            hint = f" Install pnpm matching '{required}'." if required else ""
            self._skip(f"pnpm not found on PATH.{hint}", dist_dir)
            return

        env = os.environ.copy()
        env.setdefault("NODE_ENV", "production")
        try:
            subprocess.run(
                ["pnpm", "install", "--frozen-lockfile"],
                cwd=web_dir,
                check=True,
                env=env,
            )
            subprocess.run(
                [
                    "pnpm",
                    "--filter",
                    "nemo-studio-ui",
                    "build:fastapi",
                    "--outDir",
                    str(dist_dir),
                ],
                cwd=web_dir,
                check=True,
                env=env,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            self._skip(f"Studio UI build failed ({exc})", dist_dir)

    @staticmethod
    def _skip(reason: str, dist_dir: Path) -> None:
        print(f"warning: {reason}; wheel will ship without Studio UI", file=sys.stderr)
        shutil.rmtree(dist_dir, ignore_errors=True)
        dist_dir.mkdir(parents=True, exist_ok=True)


def _required_engine(package_json: Path, name: str) -> str | None:
    try:
        data = json.loads(package_json.read_text())
    except (OSError, ValueError):
        return None
    engines = data.get("engines") or {}
    value = engines.get(name)
    return value if isinstance(value, str) else None
