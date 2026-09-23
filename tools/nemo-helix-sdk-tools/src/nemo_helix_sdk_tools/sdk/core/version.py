# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile

from nemo_helix_sdk_tools.sdk.core.common import SdkInfo
from packaging.version import Version


def get_sdk_version():
    return os.environ.get("UV_DYNAMIC_VERSIONING_BYPASS", "0.0.0")


def validate_version(version: str) -> None:
    parsed = Version(version)
    if str(parsed) != version:
        raise ValueError(f"Version string is not normalized. Expected: {str(parsed)!r}, got: {version!r}")


HATCH_CONFIG_TEMPLATE = """\
mode = "local"
project = ""

[projects]
"PROJECT_NAME" = {"location" = "PROJECT_DIR"}
"""


def update_package_version(project_name: str, project_dir: str, version: str) -> None:
    validate_version(version)

    hatch_config_content = HATCH_CONFIG_TEMPLATE.replace("PROJECT_NAME", project_name).replace(
        "PROJECT_DIR", project_dir
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml") as f:
        f.write(hatch_config_content)
        f.flush()
        cmd = [
            "uv",
            "run",
            "--with",
            "hatch==1.14.1",
            "--with",
            "virtualenv<21",
            "--frozen",
            "hatch",
            "--config",
            f.name,
            "--project",
            project_name,
            "version",
            version,
        ]
        result = subprocess.run(cmd, env={**os.environ, "HATCH_VERSION_VALIDATE_BUMP": "false"})
        result.check_returncode()
        print(f"Updated {project_name!r} version to {version!r}")


def update_sdk_version(sdk_info: SdkInfo, version: str = get_sdk_version()) -> None:
    update_package_version(sdk_info.package_name, str(sdk_info.sdk_dir), version)
