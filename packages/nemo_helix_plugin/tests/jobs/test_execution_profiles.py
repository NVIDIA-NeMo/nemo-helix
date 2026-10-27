# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.jobs.execution_profiles import JOB_LOGS_ENDPOINT_ENVVAR, JobExecutionProfileConfig
from pydantic import ValidationError


@pytest.mark.parametrize(
    "envvar",
    [
        JOB_LOGS_ENDPOINT_ENVVAR,
        WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    ],
)
def test_job_execution_profile_config_rejects_platform_injected_env_vars(envvar: str) -> None:
    with pytest.raises(ValidationError, match=envvar):
        JobExecutionProfileConfig(env={envvar: "override"})


def test_a_kubernetes_profile_with_secret_and_config_map_volumes_parses():
    """Jobs serves profiles whose storage mounts a Secret or ConfigMap, such as a CA bundle, so the client reads them."""
    from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
    from pydantic import TypeAdapter

    served = {
        "backend": "kubernetes_job",
        "provider": "cpu",
        "profile": "workload",
        "config": {
            "storage": {
                "additional_volumes": [
                    {
                        "name": "tls-ca",
                        "secret": {
                            "secret_name": "envoy-tls",
                            "default_mode": 0o440,
                            "optional": True,
                            "items": [{"key": "ca.crt", "path": "tls/ca.crt", "mode": 0o400}],
                        },
                    },
                    {
                        "name": "settings",
                        "config_map": {
                            "name": "job-settings",
                            "default_mode": 0o644,
                            "optional": False,
                            "items": [{"key": "settings.yaml", "path": "config/settings.yaml", "mode": 0o600}],
                        },
                    },
                ],
                "additional_volume_mounts": [
                    {"name": "tls-ca", "mount_path": "/etc/tls", "read_only": True},
                    {"name": "settings", "mount_path": "/etc/settings"},
                ],
            }
        },
    }

    profile = TypeAdapter(ExecutionProfile).validate_python(served)

    volumes = profile.config.storage.additional_volumes
    assert volumes[0].secret is not None
    assert volumes[0].secret.model_dump() == {
        "secret_name": "envoy-tls",
        "default_mode": 0o440,
        "optional": True,
        "items": [{"key": "ca.crt", "path": "tls/ca.crt", "mode": 0o400}],
    }
    assert volumes[1].config_map is not None
    assert volumes[1].config_map.model_dump() == {
        "name": "job-settings",
        "default_mode": 0o644,
        "optional": False,
        "items": [{"key": "settings.yaml", "path": "config/settings.yaml", "mode": 0o600}],
    }
