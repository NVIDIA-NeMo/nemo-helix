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
                        "secret": {"secret_name": "envoy-tls", "items": [{"key": "ca.crt", "path": "ca.crt"}]},
                    },
                    {"name": "settings", "config_map": {"name": "job-settings"}},
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
    assert volumes[0].secret.secret_name == "envoy-tls"
    assert volumes[0].secret.items[0].path == "ca.crt"
    assert volumes[1].config_map.name == "job-settings"
