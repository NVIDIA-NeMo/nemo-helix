# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.automodel.entities.validators import validate_fileset_uri


def test_validate_fileset_workspace_name() -> None:
    assert validate_fileset_uri("acme-corp/train-data") == "acme-corp/train-data"


def test_validate_fileset_bare_name() -> None:
    assert validate_fileset_uri("train-data") == "train-data"


def test_validate_strips_legacy_fileset_prefix() -> None:
    assert validate_fileset_uri("fileset://acme-corp/train-data") == "acme-corp/train-data"


def test_validate_rejects_hf_protocol() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        validate_fileset_uri("hf://org/dataset")
