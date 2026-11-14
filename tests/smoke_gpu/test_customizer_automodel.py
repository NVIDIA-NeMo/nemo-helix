# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nhx-automodel image import smoke tests.

Built as part of the docker-bake.hcl bake group (smoke-test stage) and run
on a CPU runner — no GPU hardware required.

Two failure classes are caught at .so load time, before any GPU device is touched:

  ModuleNotFoundError  — package missing from the image (e.g. excluded from
                         a tar layer without a compensating COPY command)

  ImportError          — CUDA extension .so has an undefined symbol; the wheel
                         was compiled against a different PyTorch version than
                         the one installed (ABI mismatch)
"""

from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import pytest
from file_removals import assert_file_patterns_absent, read_file_patterns
from python_package_versions import assert_python_package_min_versions

BASE_FILE_REMOVALS = (Path("/smoke_test/removals/files/base/pytorch-ngc-common.txt"),)
FINAL_FILE_REMOVALS = Path("/smoke_test/removals/files/final/customizer-codecs.txt")
SOUNDFILE_FILE_REMOVALS = {
    "/opt/venv/lib/python3.*/site-packages/_soundfile_data/libsndfile_*.so",
    "/opt/venv/lib/python3.*/site-packages/soundfile.py",
}
DALI_FILE_REMOVALS = {
    "/usr/local/lib/python3.12/dist-packages/nvidia/dali",
    "/usr/local/lib/python3.12/dist-packages/nvidia_dali_cuda130-*.dist-info",
}
MINIMUM_PYTHON_PACKAGE_VERSIONS = {
    "bitsandbytes": "0.49.2",
    "mamba-ssm": "2.3.0",
    "mlflow-skinny": "3.16.1",
    "nemo-automodel": "0.7.0",
    "notebook": "7.6.3",
    "peft": "0.20.0",
    "pillow": "13.0.0",
    "transformers": "5.18.0",
    "virtualenv": "21.14.9",
    "wandb": "0.28.2",
}


@pytest.mark.smoke_nhx_automodel_training
def test_python_package_min_versions():
    assert_python_package_min_versions(MINIMUM_PYTHON_PACKAGE_VERSIONS)


@pytest.mark.smoke_nhx_automodel_training
def test_torch_importable():
    import torch  # noqa: F401


@pytest.mark.smoke_nhx_automodel_training
def test_transformers_importable():
    import transformers  # noqa: F401


@pytest.mark.smoke_nhx_automodel_training
def test_mamba_ssm_importable():
    import mamba_ssm  # noqa: F401


@pytest.mark.smoke_nhx_automodel_training
def test_causal_conv1d_importable():
    import causal_conv1d  # noqa: F401


@pytest.mark.smoke_nhx_automodel_training
def test_bitsandbytes_importable():
    import bitsandbytes  # noqa: F401


@pytest.mark.smoke_nhx_automodel_training
def test_nvshmem_importable():
    # magi-attention records an RPATH into this package. The training image installs
    # the same nvidia-nvshmem-cu13 the wheel was built against.
    import nvidia.nvshmem  # noqa: F401


@pytest.mark.smoke_nhx_automodel_training
def test_magi_attention_extensions_importable():
    # magi_attention's __init__ only warns when its compiled extensions fail to load,
    # so import them directly to surface a missing .so or torch ABI mismatch. Loading
    # magi_attn_comm also resolves libnvshmem_host through the RPATH recorded at build time,
    # so this fails if the base image's NVSHMEM is missing or at another path.
    from magi_attention import magi_attn_comm, magi_attn_ext  # noqa: F401


@pytest.mark.smoke_nhx_automodel_training
def test_nhx_automodel_training_importable():
    import nemo_automodel  # noqa: F401
    from nhx.automodel.tasks.training import __main__ as training_main  # noqa: F401


@pytest.mark.smoke_nhx_automodel_training
def test_soundfile_libsndfile_removed():
    patterns = read_file_patterns(FINAL_FILE_REMOVALS)
    assert SOUNDFILE_FILE_REMOVALS.issubset(patterns)
    assert_file_patterns_absent(patterns)


@pytest.mark.smoke_nhx_automodel_training
def test_transformers_audio_backend_probe_is_off():
    """The payload and its shim must be removed together.

    ``transformers.audio_utils`` runs ``if is_soundfile_available(): import soundfile``, and
    that probe is ``find_spec("soundfile")`` -- file presence, not loadability. Removing only
    the codec leaves the probe answering yes for a backend that then fails to dlopen, which
    breaks ``from transformers import AutoProcessor`` and everything importing through it.
    """
    from transformers.utils.import_utils import is_soundfile_available

    assert not is_soundfile_available()


@pytest.mark.smoke_nhx_automodel_training
def test_peft_lora_dispatch_matches_installed_torchao(monkeypatch: pytest.MonkeyPatch):
    """PEFT's torchao dispatcher must import against the torchao in this image.

    ``dispatch_torchao`` runs during adapter injection and is gated on ``find_spec("torchao")`` --
    file presence, not API compatibility -- before importing torchao symbols. torchao comes from
    the NGC base image rather than the venv, so a peft expecting a retired torchao API fails
    ``merge=true`` jobs at injection, after training has already succeeded.
    """
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def find_spec_without_transformer_engine(name: str, *args: Any, **kwargs: Any) -> ModuleSpec | None:
        if name == "transformer_engine" or name.startswith("transformer_engine."):
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_transformer_engine)

    import torch
    from peft import LoraConfig
    from peft.tuners.lora.torchao import dispatch_torchao

    # A plain Linear holds no torchao tensor subclass, so the dispatcher must decline by
    # returning None rather than raise ImportError while probing for one.
    assert dispatch_torchao(torch.nn.Linear(8, 8), "default", LoraConfig()) is None


@pytest.mark.smoke_nhx_automodel_training
def test_dali_files_removed():
    patterns = [
        pattern
        for pattern in read_file_patterns(*BASE_FILE_REMOVALS)
        if "/nvidia/dali" in pattern or "nvidia_dali_cuda130" in pattern
    ]
    assert DALI_FILE_REMOVALS.issubset(patterns)
    assert_file_patterns_absent(patterns)


_CLASSIC_LAST_TOKEN_BASE = {
    "modules.json": [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    ],
    "1_Pooling/config.json": {"pooling_mode_mean_tokens": False, "pooling_mode_lasttoken": True},
}


@pytest.mark.smoke_nhx_automodel_training
@pytest.mark.parametrize(
    ("base_files", "job", "expected"),
    [
        (_CLASSIC_LAST_TOKEN_BASE, (None, None), ("last", False)),
        (
            {
                "modules.json": [
                    {"path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
                    {"path": "1_Pooling", "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"},
                    {"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
                ],
                "1_Pooling/config.json": {"pooling_mode": "cls"},
            },
            (None, None),
            ("cls", True),
        ),
        ({"config.json": {"pooling": "mean"}}, (None, None), ("avg", True)),
        ({}, (None, None), ("avg", True)),
        (_CLASSIC_LAST_TOKEN_BASE, ("avg", None), ("avg", False)),
        ({**_CLASSIC_LAST_TOKEN_BASE, "1_Pooling/config.json": {"pooling_mode_max_tokens": True}}, (None, None), None),
    ],
    ids=["classic-last-token", "pooling-mode-cls", "config-pooling", "nothing-declared", "job-wins", "max-pooling"],
)
def test_embedder_pooling_follows_automodel(
    tmp_path: Path, base_files: dict[str, Any], job: tuple[str | None, bool | None], expected: tuple[str, bool] | None
):
    """Helix asks the installed Automodel how a base pools; its private functions must still exist."""
    import json

    from nhx.automodel.tasks.training.backends.checkpoints import _automodel_bi_encoder_pooling

    for name, content in {"config.json": {}, **base_files}.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(json.dumps(content))

    if expected is None:
        with pytest.raises(ValueError, match="pooling metadata"):
            _automodel_bi_encoder_pooling(str(tmp_path), *job)
    else:
        assert _automodel_bi_encoder_pooling(str(tmp_path), *job) == expected
