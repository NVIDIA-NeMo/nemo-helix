# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build supervise``: the build step. It builds each context in a sandbox, through the deployment's provider.

It mounts no volume itself: the sandboxes read the contexts from the work volume and write the layouts back to it.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from kubernetes import client as k8s
from kubernetes import config as k8s_config
from nemo_builder_plugin.run.opensandbox_sandbox import OpenSandboxProvider
from nemo_builder_plugin.run.pod_sandbox import KubernetesPodProvider
from nemo_builder_plugin.run.sandbox import SandboxProvider
from nemo_builder_plugin.run.utils import job_identity, read_step_config
from nemo_builder_plugin.steps import SandboxGroup, SuperviseStepConfig
from nemo_helix_plugin.jobs.constants import job_storage_subpath
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: This step's namespace, which Kubernetes mounts beside the ServiceAccount token.
_NAMESPACE_FILE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def _build_group(provider: SandboxProvider, index: int, group: SandboxGroup) -> int:
    """Build one group in its own sandbox, and return how many of its images failed.

    Never raises: a sandbox that can't be created, run or read fails its own images, not the images other
    sandboxes have built.
    """
    try:
        results = provider.build(index, group)
    except Exception:
        logger.exception("sandbox %d failed; none of its %d image(s) built", index, len(group.images))
        return len(group.images)

    failures = 0
    for image in group.images:
        code = results.get(image.image)
        if code is None:
            logger.error("image %s: no result recorded", sanitize_for_log(image.image))
            failures += 1
        elif code != 0:
            logger.error("image %s: kaniko exited %d", sanitize_for_log(image.image), code)
            failures += 1
        else:
            logger.info("image %s: built", sanitize_for_log(image.image))
    return failures


def _exit_code(*, failures: int, total: int) -> int:
    """Non-zero only if no image built, since Jobs runs ``push`` only after this step succeeds.

    A partial failure still fails the job: ``push`` counts each missing layout as a failure.
    """
    if total and failures >= total:
        return 1
    return 0


def _read_api_key(api: k8s.CoreV1Api, *, name: str, namespace: str) -> str:
    """The OpenSandbox tenant's key, from a Secret only this step's ServiceAccount may read."""
    secret = api.read_namespaced_secret(name=name, namespace=namespace)
    encoded = (secret.data or {}).get("api-key")
    if not encoded:
        raise RuntimeError(f"Secret {name} in {namespace} has no api-key")
    return base64.b64decode(encoded).decode().strip()


def _provider(
    config: SuperviseStepConfig, api: k8s.CoreV1Api, *, namespace: str, workspace: str, job_id: str
) -> SandboxProvider:
    sandbox = config.sandbox
    job_sub_path = job_storage_subpath(workspace, job_id)
    if sandbox.provider == "kubernetes_pod":
        return KubernetesPodProvider(
            api, namespace=namespace, sandbox=sandbox, workspace=workspace, job_id=job_id, job_sub_path=job_sub_path
        )
    if sandbox.opensandbox is None:
        raise RuntimeError("the opensandbox provider needs its server's settings, and the compiler wrote none")
    # Imported here: the SDK is the plugin's `opensandbox` extra, needed only by this provider.
    from nemo_builder_plugin.run.opensandbox_sdk import SdkApi

    key = _read_api_key(api, name=sandbox.opensandbox.api_key_secret, namespace=namespace)
    return OpenSandboxProvider(
        SdkApi(sandbox.opensandbox, key),
        sandbox=sandbox,
        workspace=workspace,
        job_id=job_id,
        job_sub_path=job_sub_path,
    )


def main() -> int:
    config = SuperviseStepConfig.model_validate(read_step_config())
    workspace, job_id = job_identity()

    k8s_config.load_incluster_config()
    api = k8s.CoreV1Api()
    # Where the work volume is: the backend refuses profiles that put this step and `fetch` in different namespaces.
    namespace = _NAMESPACE_FILE.read_text().strip()
    provider = _provider(config, api, namespace=namespace, workspace=workspace, job_id=job_id)
    logger.info("building in %s sandboxes", config.sandbox.provider)
    # A retried step would otherwise meet its earlier attempt's sandboxes: a plain pod's name is taken until it goes.
    provider.sweep()

    total = sum(len(group.images) for group in config.groups)
    failures = sum(_build_group(provider, index, group) for index, group in enumerate(config.groups))
    if failures:
        logger.error("%d of %d image(s) failed to build", failures, total)
    return _exit_code(failures=failures, total=total)
