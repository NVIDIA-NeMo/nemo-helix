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
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
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
            model = "smoke-model"
            # Reuse the published gateway's mock-provider protocol, as CPU E2E does.
            # Discovery still runs through the controller; no source test package is installed.
            response = {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            models_client.create_provider(
                body=CreateModelProviderRequest(
                    name="igw-mock-smoke",
                    host_url="http://mock.invalid",
                    default_extra_headers={
                        "X-Mock-Served-Models": json.dumps([model]),
                        "X-Mock-Response": json.dumps(response),
                    },
                    enabled_models=[model],
                )
            ).data()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                provider = models_client.get_provider(name="igw-mock-smoke").data()
                mapping = next((item for item in provider.served_models or [] if item.served_model_name == model), None)
                if mapping is not None:
                    break
                time.sleep(3)
            else:
                raise RuntimeError("Provider did not discover the configured model within 120 seconds")
            # Keep model discovery and routing in the platform path.
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
            # The CLI uploads this directory as the agent package, so keep it separate
            # from the tool installation, credentials, and other test state.
            agent_dir = Path(os.environ["SMOKE_ROOT"], "sample-agent")
            agent_dir.mkdir(exist_ok=True)
            (agent_dir / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        elif operation == "cleanup":
            workspaces_client.delete_workspace(name=WORKSPACE).data()
        else:
            raise ValueError(f"Unknown operation: {operation}")


if __name__ == "__main__":
    main(sys.argv[1])
