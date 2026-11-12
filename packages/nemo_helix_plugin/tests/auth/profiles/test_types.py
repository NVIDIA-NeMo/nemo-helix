# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.auth.profiles.types import ProfileUpdate
from pydantic import ValidationError


@pytest.mark.parametrize("locale", ["en", "fil", "en-US", "zh_Hant_TW", "abcde"])
def test_accepts_bcp47_locales(locale: str) -> None:
    assert ProfileUpdate(locale=locale).locale == locale


@pytest.mark.parametrize("locale", ["e", "abcd", "abcdefghi", "en-"])
def test_rejects_invalid_locales(locale: str) -> None:
    with pytest.raises(ValidationError):
        ProfileUpdate(locale=locale)
