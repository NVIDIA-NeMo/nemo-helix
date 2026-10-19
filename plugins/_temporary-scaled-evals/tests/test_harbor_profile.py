# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor config-profile validation tests."""

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.models.harbor_profile import validate_harbor_profile_config


@pytest.mark.parametrize(
    "config",
    [
        {"harbor_config": 123},
        {"env": []},
        {"agents": "oracle"},
        {"n_attempts": 0},
    ],
)
def test_harbor_profile_rejects_malformed_known_fields(config: dict) -> None:
    with pytest.raises(ValueError):
        validate_harbor_profile_config(config)


def test_harbor_profile_preserves_independently_versioned_extensions() -> None:
    validated = validate_harbor_profile_config(
        {
            "environment": {"import_path": "sandbox_k8s.harbor:K8sSandboxEnvironment"},
            "agents": [{"name": "oracle"}],
            "tasks": [{"path": "${TASK_PATH}"}],
            "future_harbor_field": {"enabled": True},
        }
    )

    assert validated.model_dump()["future_harbor_field"] == {"enabled": True}
