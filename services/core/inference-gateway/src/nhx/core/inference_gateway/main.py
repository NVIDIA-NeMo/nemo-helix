# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway service entry point."""

import uvicorn
from nhx.core.inference_gateway.service import InferenceGatewayService

# Global service instance for platform integration
service = InferenceGatewayService()


def run_standalone():
    """Run the inference gateway service as a standalone server."""
    uvicorn.run(service.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_standalone()
