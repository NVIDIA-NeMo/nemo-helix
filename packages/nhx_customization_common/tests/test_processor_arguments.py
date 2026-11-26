# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.customization_common.processor_arguments import check_processor_arguments

_PATCH_COUNT_KEYS = ["max_num_patches", "min_num_patches", "patch_size"]


def test_arguments_the_processor_reads_pass() -> None:
    check_processor_arguments({"min_num_patches": 256, "use_fast": True}, _PATCH_COUNT_KEYS)


def test_an_argument_the_processor_does_not_read_is_named_with_the_keys_it_does() -> None:
    with pytest.raises(ValueError, match=r"\['size'\].*\['max_num_patches', 'min_num_patches', 'patch_size'\]"):
        check_processor_arguments({"size": {"longest_edge": 1}}, _PATCH_COUNT_KEYS)


def test_unknown_keys_check_nothing() -> None:
    check_processor_arguments({"anything": 1}, None)
