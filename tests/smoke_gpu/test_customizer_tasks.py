# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nhx-customizer-tasks image import smoke tests.

Built as part of the docker-bake.hcl bake group (smoke-test stage) and run
on a CPU runner — no GPU hardware required.

Two failure classes are caught at .so load time, before any GPU device is touched:

  ModuleNotFoundError  — package missing from the image (e.g. excluded from
                         a tar layer without a compensating COPY command)

  ImportError          — CUDA extension .so has an undefined symbol; the wheel
                         was compiled against a different PyTorch version than
                         the one installed (ABI mismatch)
"""

from pathlib import Path

import pytest
from file_removals import assert_file_patterns_absent, read_file_patterns
from python_package_versions import assert_python_package_min_versions

BASE_FILE_REMOVALS = (
    Path("/smoke_test/removals/files/base/pytorch-ngc-common.txt"),
    Path("/smoke_test/removals/files/base/nhx-customizer-tasks.txt"),
)
DALI_FILE_REMOVALS = {
    "/usr/local/lib/python3.12/dist-packages/nvidia/dali",
    "/usr/local/lib/python3.12/dist-packages/nvidia_dali_cuda130-*.dist-info",
}
MINIMUM_PYTHON_PACKAGE_VERSIONS = {
    "mamba-ssm": "2.3.0",
    "transformers": "5.8.1",
    "wandb": "0.28.2",
}


@pytest.mark.smoke_nhx_customizer_tasks
def test_python_package_min_versions():
    assert_python_package_min_versions(MINIMUM_PYTHON_PACKAGE_VERSIONS)


@pytest.mark.smoke_nhx_customizer_tasks
def test_torch_importable():
    import torch  # noqa: F401


@pytest.mark.smoke_nhx_customizer_tasks
def test_transformers_importable():
    import transformers  # noqa: F401


@pytest.mark.smoke_nhx_customizer_tasks
def test_accelerate_importable():
    import accelerate  # noqa: F401


@pytest.mark.smoke_nhx_customizer_tasks
def test_mamba_ssm_importable():
    import mamba_ssm  # noqa: F401


@pytest.mark.smoke_nhx_customizer_tasks
def test_causal_conv1d_importable():
    import causal_conv1d  # noqa: F401


@pytest.mark.smoke_nhx_customizer_tasks
def test_nhx_customizer_tasks_importable():
    from nhx.core.models.sidecars.adapters.main import run as lora_sidecar_run  # noqa: F401
    from nhx.core.models.tasks.model_spec import __main__ as model_spec_main  # noqa: F401
    from nhx.customization_common.tasks import file_io  # noqa: F401
    from nhx.customization_common.tasks.file_io import __main__ as file_io_main  # noqa: F401
    from nhx.customization_common.tasks.model_entity import __main__ as model_entity_main  # noqa: F401


@pytest.mark.smoke_nhx_customizer_tasks
def test_sdk_alias_resources_importable():
    from nemo_helix import NeMoHelix

    sdk = NeMoHelix(base_url="http://127.0.0.1:1")
    try:
        sdk.files
        sdk.models
    finally:
        sdk.close()


@pytest.mark.smoke_nhx_customizer_tasks
def test_dali_files_removed():
    patterns = [
        pattern
        for pattern in read_file_patterns(*BASE_FILE_REMOVALS)
        if "/nvidia/dali" in pattern or "nvidia_dali_cuda130" in pattern
    ]
    assert DALI_FILE_REMOVALS.issubset(patterns)
    assert_file_patterns_absent(patterns)
