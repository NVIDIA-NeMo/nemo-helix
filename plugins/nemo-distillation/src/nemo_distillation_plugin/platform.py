# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform calls the node adapters make, and the fileset handoff the tick confirms."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import yaml
from nemo_customizer.jobs.atif_convert import AtifConvertJob, AtifConvertSpec, fileset_prefix
from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.agents.types import CreateDeploymentRequest, DeploymentMode, RegisterAgentRequest
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.evals.client import AsyncEvaluatorClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.intake.types import EvaluationCreateRequest, ExperimentCreateRequest
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    Engine,
    ModelDeploymentConfigModelSpec,
    ToolCallConfig,
)

from nemo_distillation_plugin.models import FABRIC_AGENT_CONFIG_FORMAT, GraphError, Workflow
from nemo_distillation_plugin.nodes.adapters import build_adapters
from nemo_distillation_plugin.nodes.clone import clone_config, require_fabric
from nemo_distillation_plugin.runtime import Missing

logger = logging.getLogger(__name__)


class Platform:
    """One platform client, the six node adapters, and the workflow fileset."""

    def __init__(self, client: AsyncNemoClient) -> None:
        self.client = client
        self.jobs = AsyncJobsClient.from_client(client)
        self.files = AsyncFilesClient.from_client(client)
        self.models = AsyncModelsClient.from_client(client)
        self.agents = AsyncAgentsClient.from_client(client)
        self.intake = AsyncIntakeClient.from_client(client)
        self.evals = AsyncEvaluatorClient.from_client(client)
        self._adapters = build_adapters(self)

    def adapter(self, kind: str):
        try:
            return self._adapters[kind]
        except KeyError as exc:
            raise GraphError(f"unknown node kind {kind!r}") from exc

    async def read_output(self, workflow: Workflow, node_key: str) -> dict[str, Any] | None:
        path = f"{fileset_prefix(node_key)}/output.json"
        try:
            response = await self.files.download_file(workspace=workflow.workspace, name=workflow.fileset, path=path)
            blob = await response.read()
        except NotFoundError:
            return None
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    async def write_output(self, workflow: Workflow, node_key: str, payload: dict[str, Any]) -> None:
        path = f"{fileset_prefix(node_key)}/output.json"
        await self.files.upload_file(
            workspace=workflow.workspace,
            name=workflow.fileset,
            path=path,
            content=json.dumps(payload).encode("utf-8"),
        )

    async def submit_convert(
        self,
        *,
        workspace: str,
        name: str,
        fileset: str,
        node_key: str,
        trajectory_filesets: list[str],
        pass_filter: dict[str, Any],
        student_model: str | None,
        max_tokens: dict[str, Any] | None,
    ) -> None:
        spec = AtifConvertSpec(
            workspace=workspace,
            fileset=fileset,
            node_key=node_key,
            trajectory_filesets=trajectory_filesets,
            pass_filter=pass_filter,
            student_model=student_model,
            max_tokens=max_tokens,
        )
        platform_spec = await AtifConvertJob.compile(
            workspace=workspace,
            spec=spec,
            entity_client=None,
            job_name=name,
            async_sdk=self.client,
        )
        try:
            await self.jobs.create_job(
                workspace=workspace,
                body=CreateHelixJobRequest(
                    name=name,
                    spec=spec.model_dump(mode="json"),
                    platform_spec=platform_spec,
                    source="distillation",
                ),
            )
        except ConflictError:
            return

    async def submit_customization(self, *, workspace: str, name: str, spec: dict[str, Any]) -> None:
        await self._post(
            f"/apis/customization/v2/workspaces/{workspace}/automodel/jobs",
            {"name": name, "spec": spec},
        )

    async def job_status(self, *, workspace: str, name: str) -> str:
        try:
            payload = (await self.jobs.get_job_status(workspace=workspace, name=name)).data()
        except NotFoundError as exc:
            raise Missing(name) from exc
        return _status_value(payload.status)

    async def cancel_job(self, *, workspace: str, name: str) -> None:
        try:
            await self.jobs.cancel_job(workspace=workspace, name=name)
        except NotFoundError:
            return

    async def create_deployment(
        self,
        *,
        workspace: str,
        config_name: str,
        deployment_name: str,
        inputs: dict[str, Any],
    ) -> None:
        namespace, model_name = _model_parts(str(inputs["model"]), workspace)
        tool_call = inputs.get("tool_call_config")
        await self.models.create_deployment_config(
            workspace=workspace,
            body=CreateModelDeploymentConfigRequest(
                name=config_name,
                engine=Engine(str(inputs["engine"])),
                model_spec=ModelDeploymentConfigModelSpec(
                    model_namespace=namespace,
                    model_name=model_name,
                    lora_enabled=bool(inputs.get("lora_enabled")),
                    tool_call_config=ToolCallConfig.model_validate(tool_call) if tool_call else None,
                ),
                executor_config=ContainerExecutorConfig(
                    gpu=int(inputs["gpu"]),
                    additional_args=[str(arg) for arg in inputs.get("additional_args") or []],
                ),
            ),
            exist_ok=True,
        )
        await self.models.create_deployment(
            workspace=workspace,
            body=CreateModelDeploymentRequest(name=deployment_name, config=config_name),
            exist_ok=True,
        )

    async def deployment_status(self, *, workspace: str, name: str) -> str:
        try:
            payload = (await self.models.get_deployment(workspace=workspace, name=name)).data()
        except NotFoundError as exc:
            raise Missing(name) from exc
        return _status_value(payload.status)

    async def create_agent_deployment(
        self,
        *,
        workspace: str,
        name: str,
        agent_name: str,
        mode: str,
        image: str,
        use_image_entrypoint: bool,
    ) -> None:
        try:
            await self.agents.create_deployment(
                workspace=workspace,
                body=CreateDeploymentRequest(
                    agent=agent_name,
                    name=name,
                    deployment_mode=cast(DeploymentMode, mode),
                    image=image,
                    use_image_entrypoint=use_image_entrypoint,
                ),
            )
        except ConflictError:
            return

    async def agent_deployment_status(self, *, workspace: str, name: str) -> str:
        try:
            payload = (await self.agents.get_deployment(workspace=workspace, name=name)).data()
        except NotFoundError as exc:
            raise Missing(name) from exc
        return _status_value(payload.status)

    async def delete_agent_deployment(self, *, workspace: str, name: str) -> None:
        try:
            await self.agents.delete_deployment(workspace=workspace, name=name)
        except NotFoundError:
            return

    async def delete_deployment(self, *, workspace: str, name: str) -> None:
        try:
            await self.models.delete_deployment(workspace=workspace, name=name)
        except NotFoundError:
            return

    async def prepare_agent(
        self,
        *,
        workspace: str,
        agent_name: str,
        source_agent: str,
        model: str | None,
        workflow_name: str,
        arm: str,
    ) -> None:
        source_workspace, source_name = _split_ref(source_agent, workspace)
        agent = (await self.agents.get_agent(workspace=source_workspace, name=source_name)).data()
        config = dict(agent.config)
        require_fabric(agent.config_format, config)
        cloned = clone_config(config, name=agent_name, model=model)
        labels = dict(agent.labels)
        labels["helix-distillation/workflow"] = workflow_name
        labels["helix-distillation/source"] = source_agent
        labels["helix-distillation/arm"] = arm
        fields: dict[str, Any] = {
            "name": agent_name,
            "description": agent.description,
            "reference": agent.reference,
            "labels": labels,
            "config": cloned,
            "context_schema": agent.context_schema,
            "default_input": agent.default_input,
            "config_format": FABRIC_AGENT_CONFIG_FORMAT,
        }
        actions = set(agent.post_run_actions)
        if actions:
            fields["post_run_actions"] = actions
        await self.agents.register_agent(
            workspace=workspace,
            body=RegisterAgentRequest(**fields),
            exist_ok=True,
        )
        await self._copy_ethos(source_workspace, source_name, workspace, agent_name)

    async def delete_agent(self, *, workspace: str, agent_name: str) -> None:
        try:
            await self.agents.delete_agent(workspace=workspace, name=agent_name)
        except NotFoundError:
            pass
        try:
            await self.files.delete_fileset(workspace=workspace, name=f"{agent_name}-ethos")
        except NotFoundError:
            return

    async def read_eval_config(self, workspace: str, ref: str) -> dict[str, Any]:
        """Read one eval config file (JSON or YAML) named ``[workspace/]fileset#path``."""
        location, _, path = ref.partition("#")
        ref_workspace, _, fileset = location.rpartition("/")
        try:
            response = await self.files.download_file(workspace=ref_workspace or workspace, name=fileset, path=path)
            blob = await response.read()
        except NotFoundError as exc:
            raise ValueError(f"eval config {ref} not found") from exc
        try:
            config = yaml.safe_load(blob)
        except yaml.YAMLError as exc:
            raise ValueError(f"eval config {ref} is not valid JSON or YAML: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"eval config {ref} must be an object")
        return config

    async def submit_eval(self, *, workspace: str, name: str, spec: dict[str, Any]) -> None:
        await self._post(f"/apis/evals/v2/workspaces/{workspace}/agent-evaluate/jobs", {"name": name, "spec": spec})

    async def eval_status(self, *, workspace: str, name: str) -> str:
        try:
            payload = (await self.evals.get_agent_eval_job_status(workspace=workspace, name=name)).data()
        except NotFoundError as exc:
            raise Missing(name) from exc
        return _status_value(payload.status)

    async def cancel_eval(self, *, workspace: str, name: str) -> None:
        try:
            await self.evals.cancel_agent_eval_job(workspace=workspace, name=name)
        except NotFoundError:
            return

    async def submit_dataset_eval(
        self,
        *,
        workspace: str,
        name: str,
        agent_name: str,
        deployment_name: str,
        spec: dict[str, Any],
        evaluation_id: str,
    ) -> None:
        body = dict(spec)
        body["target"] = _agent_chat_target(self.client.base_url, workspace, deployment_name)
        body["publication"] = {"intake": {"evaluation_id": evaluation_id, "agent_name": agent_name}}
        body.setdefault(
            "params",
            {"parallelism": 4, "request_timeout": 300, "max_retries": 5, "ignore_request_failure": False},
        )
        await self._post(
            f"/apis/evals/v2/workspaces/{workspace}/evaluate/jobs",
            {"name": name, "spec": body},
        )

    async def dataset_eval_status(self, *, workspace: str, name: str) -> str:
        try:
            payload = (await self.evals.get_evaluate_job_status(workspace=workspace, name=name)).data()
        except NotFoundError as exc:
            raise Missing(name) from exc
        return _status_value(payload.status)

    async def cancel_dataset_eval(self, *, workspace: str, name: str) -> None:
        try:
            await self.evals.cancel_evaluate_job(workspace=workspace, name=name)
        except NotFoundError:
            return

    async def ensure_experiment(
        self,
        *,
        workspace: str,
        name: str,
        evaluations: dict[str, str],
        dataset_name: str,
    ) -> str:
        experiment = (
            await self.intake.create_experiment(
                workspace=workspace,
                body=ExperimentCreateRequest(name=name, description=f"Distillation workflow {name}"),
                exist_ok=True,
            )
        ).data()
        for arm, evaluation_name in evaluations.items():
            try:
                await self.intake.create_evaluation(
                    workspace=workspace,
                    body=EvaluationCreateRequest(
                        name=evaluation_name,
                        experiment_ids=[experiment.id],
                        dataset_name=dataset_name,
                        description=arm,
                    ),
                )
            except ConflictError:
                continue
        return experiment.id

    async def get_experiment(self, *, workspace: str, name: str) -> dict[str, Any]:
        try:
            experiment = (await self.intake.get_experiment(workspace=workspace, name=name)).data()
        except NotFoundError as exc:
            raise Missing(name) from exc
        return {"id": experiment.id, "name": experiment.name}

    async def _copy_ethos(self, source_workspace: str, source_name: str, workspace: str, agent_name: str) -> None:
        source_fileset = f"{source_name}-ethos"
        try:
            existing = (await self.files.get_fileset(workspace=source_workspace, name=source_fileset)).data()
        except NotFoundError:
            return
        destination = f"{agent_name}-ethos"
        await self.files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(
                name=destination,
                purpose=existing.purpose,
                description=f"Ethos copy for distillation agent {agent_name}",
            ),
            exist_ok=True,
        )
        listing = (await self.files.list_files(workspace=source_workspace, name=source_fileset)).data()
        for item in listing.data:
            downloaded = await self.files.download_file(workspace=source_workspace, name=source_fileset, path=item.path)
            await self.files.upload_file(
                workspace=workspace,
                name=destination,
                path=item.path,
                content=await downloaded.read(),
            )

    async def _post(self, path: str, body: dict[str, Any]) -> None:
        headers = await self.client.request_headers() or {}
        response = await self.client.http_client.post(
            f"{self.client.base_url.rstrip('/')}{path}",
            json=body,
            headers=headers,
        )
        if response.status_code == 409:
            return
        if response.status_code >= 400:
            logger.info("POST %s returned %s", sanitize_for_log(path), response.status_code)
            raise RuntimeError(f"POST {path} failed ({response.status_code}): {response.text[:500]}")


def _agent_chat_target(base_url: str, workspace: str, deployment_name: str) -> dict[str, Any]:
    """The dataset-driven eval target: the workflow-owned agent deployment's chat endpoint."""
    root = base_url.rstrip("/")
    return {
        "format": "generic",
        "url": (f"{root}/apis/agents/v2/workspaces/{workspace}/deployments/{deployment_name}/-/v1/chat/completions"),
        "name": deployment_name,
        "body": {"model": deployment_name, "messages": [{"role": "user", "content": "{{ prompt }}"}], "stream": False},
        "response_path": "$.choices[0].message.content",
        "stream": False,
    }


def _split_ref(ref: str, workspace: str) -> tuple[str, str]:
    if "/" in ref:
        owner, name = ref.split("/", 1)
        return owner, name
    return workspace, ref


def _model_parts(model: str, workspace: str) -> tuple[str, str]:
    if "/" in model:
        namespace, name = model.split("/", 1)
        return namespace, name
    return workspace, model


def _status_value(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value)
