# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run with the published Helix tool's Python, never the checkout's environment."""

import json
import os
import sys
import time
from pathlib import Path

import yaml
from email_security_triage.resources import sample_file
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelProviderRequest
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from pydantic import SecretStr
from release_smoke_check import require_completion

WORKSPACE = "release-smoke"


def main(operation: str) -> None:
    with NemoClient(base_url=os.environ["NHX_BASE_URL"], workspace=WORKSPACE, timeout=120) as client:
        workspaces_client = WorkspacesClient.from_client(client)
        models_client = ModelsClient.from_client(client)
        if operation == "workspaces":
            name = "smoke-crud"
            workspaces_client.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
            try:
                assert workspaces_client.get_workspace(name=name).data().name == name
            finally:
                workspaces_client.delete_workspace(name=name).data()
            deadline = time.monotonic() + 60
            while name in [item.name for item in workspaces_client.list_workspaces().items()]:
                if time.monotonic() > deadline:
                    raise RuntimeError("Workspace deletion did not complete")
                time.sleep(2)
            workspaces_client.create_workspace(body=CreateWorkspaceRequest(name=WORKSPACE)).data()
        elif operation == "provider":
            model = os.environ["SMOKE_INFERENCE_MODEL"]
            SecretsClient.from_client(client).create_secret(
                body=HelixSecretCreateRequest(name="inference-key", value=SecretStr(os.environ["NVIDIA_INFERENCE_KEY"]))
            ).data()
            models_client.create_provider(
                body=CreateModelProviderRequest(
                    name="smoke",
                    host_url=os.environ["NVIDIA_INFERENCE_URL"],
                    api_key_secret_name="inference-key",
                    enabled_models=[model],
                )
            ).data()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                provider = models_client.get_provider(name="smoke").data()
                mapping = next((item for item in provider.served_models or [] if item.served_model_name == model), None)
                if mapping is not None:
                    break
                time.sleep(3)
            else:
                raise RuntimeError("Provider did not discover the configured model within 120 seconds")
            # Keep model routing and provider credential lookup in the platform path.
            ref = mapping.model_entity_id
            Path(os.environ["SMOKE_ROOT"], "model.json").write_text(json.dumps({"ref": ref}), encoding="utf-8")
        elif operation == "inference":
            ref = json.loads(Path(os.environ["SMOKE_ROOT"], "model.json").read_text())["ref"]
            resolved = models_client.resolve_model_reference(ref)
            response = (
                InferenceGatewayClient.from_client(client)
                .model_post(
                    name=ref.split("/", 1)[-1],
                    trailing_uri="v1/chat/completions",
                    body=JsonBody(
                        {
                            "model": resolved.name,
                            "messages": [{"role": "user", "content": "Say hello."}],
                            "stream": False,
                        }
                    ),
                )
                .data()
            )
            require_completion(response)
        elif operation == "sample-config":
            ref = json.loads(Path(os.environ["SMOKE_ROOT"], "model.json").read_text())["ref"]
            resolved = models_client.resolve_model_reference(ref)
            config = yaml.safe_load(sample_file("agent.yaml").read_text(encoding="utf-8"))
            # Only bind the model and gateway; preserve the release's sample implementation.
            model_config = config["models"]["default"]
            model_config["model"] = resolved.name
            model_config["base_url"] = resolved.url.replace(
                client.base_url, os.environ.get("SMOKE_AGENT_BASE_URL", client.base_url), 1
            )
            Path(os.environ["SMOKE_ROOT"], "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        elif operation == "cleanup":
            workspaces_client.delete_workspace(name=WORKSPACE).data()
        else:
            raise ValueError(f"Unknown operation: {operation}")


if __name__ == "__main__":
    main(sys.argv[1])
