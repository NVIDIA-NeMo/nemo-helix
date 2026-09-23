# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for workspace validation when creating secrets.

Regression test: creating a secret in a non-existent workspace should fail with
a clear error, not succeed silently.
"""

import pytest
from nemo_helix_plugin.client.errors import UnprocessableEntityError
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest


@pytest.mark.integration
class TestSecretWorkspaceValidation:
    """Verify that secrets cannot be created in non-existent workspaces."""

    def test_create_secret_in_nonexistent_workspace_fails(self, sdk: SecretsClient):
        """Creating a secret in a workspace that doesn't exist should return 422."""
        workspace = "nonexistent-workspace"
        with pytest.raises(UnprocessableEntityError) as exc_info:
            sdk.create_secret(
                workspace=workspace,
                body=HelixSecretCreateRequest(name="my-secret", value="my-secret-value"),
            )
        assert exc_info.value.status_code == 422
        assert f"Workspace '{workspace}' does not exist" in exc_info.value.detail
