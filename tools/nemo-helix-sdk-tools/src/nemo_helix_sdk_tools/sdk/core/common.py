# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path

SDK_DISTRIBUTION_NAME = "nemo-helix-sdk"
SDK_DIRECTORY_NAME = "nemo-helix"
SDK_MODULE_NAME = "nemo_helix"
WRAPPER_DISTRIBUTION_NAME = "nemo-helix"


def get_project_dir() -> Path:
    return Path(os.getenv("CI_PROJECT_DIR", os.getcwd()))


@dataclass
class SdkInfo:
    sdks_root_dir: Path

    package_name: str
    directory_name: str
    module_name: str
    sdk_dir: Path
    overrides_dir: Path
    readme_dir: Path
    stainless_config_file: Path
    openapi_spec_file: Path


def get_sdk_info() -> SdkInfo:
    """Get paths for SDK files."""
    project_dir = get_project_dir()
    package_name = SDK_DISTRIBUTION_NAME
    directory_name = SDK_DIRECTORY_NAME
    sdks_root_dir = project_dir / "sdk"
    python_sdk_dir = sdks_root_dir / "python" / directory_name
    overrides_dir = sdks_root_dir / "python" / "overrides" / directory_name

    return SdkInfo(
        sdks_root_dir=project_dir,
        package_name=package_name,
        directory_name=directory_name,
        module_name=SDK_MODULE_NAME,
        sdk_dir=python_sdk_dir,
        overrides_dir=overrides_dir,
        readme_dir=overrides_dir / "README",
        stainless_config_file=sdks_root_dir / "stainless.yaml",
        openapi_spec_file=project_dir / "openapi" / "openapi.yaml",
    )


def get_wrapper_dir() -> Path:
    return get_project_dir() / "packages" / "nemo_helix"
