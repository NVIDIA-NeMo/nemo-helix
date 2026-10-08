# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Formatting for untrusted values in plain-text SDK logs."""


def escape_log_value(value: str) -> str:
    """Keep user-controlled values on one log line without changing runtime data."""
    value = value.replace("\r", r"\r").replace("\n", r"\n")
    for separator in "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029":
        value = value.replace(separator, ascii(separator)[1:-1])
    return value
