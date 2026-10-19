# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails service entry point."""

import uvicorn
from nhx.guardrails.service import GuardrailsService

# Global service instance for platform integration
service = GuardrailsService()


def run_standalone():
    """Run the guardrails service as a standalone server."""
    uvicorn.run(service.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_standalone()
