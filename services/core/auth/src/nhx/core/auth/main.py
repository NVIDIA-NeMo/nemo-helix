# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth service entry point."""

from nhx.core.auth.service import AuthService

# Service instance for platform to import
service = AuthService()


def run_standalone():
    """Run the auth service standalone."""
    import uvicorn

    uvicorn.run("nhx.core.auth.main:service.app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run_standalone()
