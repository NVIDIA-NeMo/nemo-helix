# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct platform launcher for the guardrails benchmark harness."""

from __future__ import annotations

from nemo_guardrails_plugin.benchmarks.constants import (
    NHX_BENCHMARK_CONTROLLERS,
    NHX_BENCHMARK_SERVICES,
)
from nhx.platform_runner.config import HelixAppConfig
from nhx.platform_runner.run import run_platform


def main() -> None:
    """Run the minimal platform topology required by the benchmark."""
    run_platform(
        config=HelixAppConfig(
            services=NHX_BENCHMARK_SERVICES,
            controllers=NHX_BENCHMARK_CONTROLLERS,
            host="127.0.0.1",
            port=8080,
        )
    )


if __name__ == "__main__":
    main()
