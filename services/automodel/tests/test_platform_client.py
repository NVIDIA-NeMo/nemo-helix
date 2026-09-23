# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.customization_common.schemas.file_io import FileSetRef


def test_fileset_ref_parse() -> None:
    ref = FileSetRef.model_validate("acme-corp/my-dataset")
    assert ref.workspace == "acme-corp"
    assert ref.name == "my-dataset"

    bare = FileSetRef.model_validate("my-dataset")
    assert bare.workspace is None
    assert bare.name == "my-dataset"
