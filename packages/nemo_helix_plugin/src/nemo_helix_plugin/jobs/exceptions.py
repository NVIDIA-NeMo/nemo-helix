# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class PlatformJobCompilationError(Exception):
    """Exception raised for errors in the platform job compilation process."""

    pass


class PlatformJobDependencyUnavailableError(Exception):
    """A dependency required to compile a platform job is temporarily unavailable."""

    pass
