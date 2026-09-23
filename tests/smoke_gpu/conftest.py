# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke_gpu_tasks: Import smoke tests for the nhx-gpu-tasks image")
    config.addinivalue_line(
        "markers", "smoke_nhx_customizer_tasks: Import smoke tests for the nhx-customizer-tasks image"
    )
    config.addinivalue_line(
        "markers", "smoke_nhx_automodel_training: Import smoke tests for the nhx/automodel-training image"
    )
    config.addinivalue_line("markers", "smoke_nhx_rl_training: Import smoke tests for the nhx-rl-training image")
