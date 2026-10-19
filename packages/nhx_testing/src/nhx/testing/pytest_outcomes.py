# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import NoReturn, Protocol, cast

import pytest


class _PytestSkipFn(Protocol):
    def __call__(self, reason: str = "", *, allow_module_level: bool = False) -> NoReturn: ...


class _PytestFailFn(Protocol):
    def __call__(self, reason: str = "", *, pytrace: bool = True) -> NoReturn: ...


pytest_skip: _PytestSkipFn = cast(_PytestSkipFn, pytest.skip)
pytest_fail: _PytestFailFn = cast(_PytestFailFn, pytest.fail)
