# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming utilities for the NeMo CLI."""

from __future__ import annotations

from typing import Any, Iterator

from nemo_helix_ext.cli.core.formatters import format_stream_event


def handle_stream(stream: Iterator[Any]) -> None:
    """
    Handle a streaming response from the API.

    Prints each event to stdout as it arrives.

    Args:
        stream: Iterator of streaming events from the SDK
    """
    try:
        for event in stream:
            format_stream_event(event)
    except KeyboardInterrupt:
        # Allow graceful exit on Ctrl+C
        pass
    except Exception as e:
        # Let the error handler in core handle this
        raise e
