# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Models service entry point."""

from nhx.core.models.service import ModelsService

# Global service instance for platform integration
service = ModelsService()


def run_standalone():
    """Run the models service as a standalone server."""
    # NOTE: deferred import -- uvicorn is only needed for standalone mode
    import uvicorn

    uvicorn.run(service.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_standalone()
