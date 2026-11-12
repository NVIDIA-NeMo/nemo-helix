# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public, non-secret diagnostics for Trial credential policy failures."""

import re

from nemo_helix_plugin.client.errors import NemoHTTPError


def public_lifetime_limit(error: BaseException) -> str | None:
    if not isinstance(error, NemoHTTPError) or error.status_code != 400:
        return None
    detail = error.detail.removeprefix("Trial credential issuance was refused: ")
    if re.fullmatch(
        r"expires_in_seconds must be less than or equal to "
        r"auth\.access_keys\.max_expires_in_seconds \([0-9]+\)",
        detail,
    ):
        return detail
    return None
