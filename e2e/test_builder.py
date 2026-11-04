# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder on a kind cluster: one build set, two images, published, signed and checked from outside the builder.

The platform must be installed with ``e2e/k8s/values/builder.yaml``, as the ``kind-builder-e2e`` CI job does. The
checks run crane and cosign in pods on the cluster, so the test needs ``kubectl`` and:

- ``NHX_E2E_KUBE_CONTEXT``: the cluster's kubectl context. Required: the test never uses the current context.
- ``NHX_E2E_KUBE_NAMESPACE``: the release namespace, where the development registry runs; ``nemo-helix`` by default.
- ``NHX_E2E_BUILDER_REGISTRY_PASSWORD``: the development registry's password, as installed.
- ``NHX_E2E_IMAGE_REGISTRY`` and ``NHX_E2E_IMAGE_TAG``: the platform's images; crane runs from ``nhx-builder-tasks``.

Run it with ``--feature builder``, and also ``--feature auth`` against a platform with auth on. The module fails if
the platform's auth isn't what the features say, rather than skipping what it should have checked.
"""

import os
import subprocess
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.completion import CompleteRequest
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.client.oidc import discover_nhx_config
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing import as_user, grant_workspace_role
from pydantic import SecretStr

pytestmark = [
    pytest.mark.builder_e2e,
    pytest.mark.feature("builder"),
    pytest.mark.platform("kubernetes"),
    pytest.mark.timeout(1800),
]

# The chart's builder.devRegistry.username.
REGISTRY_USERNAME = "builder"
COSIGN_IMAGE = "ghcr.io/sigstore/cosign/cosign:v2.5.3"
ANNOTATION_PREFIX = "nhx.nvidia.com/"
ON_BEHALF_OF_HEADER = "X-NHX-Principal-On-Behalf-Of"
BUILD_SET = "kind-e2e"
TAG = "v1"
BUILD_DEADLINE_SECONDS = 1200
POD_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class Context:
    """One build context, in a fileset of its own, so each builds in its own sandbox."""

    dockerfile: str
    #: What the image's ``/hello.txt`` holds, as the shell reads it: without trailing newlines.
    hello: str


CONTEXTS = {
    "hello": Context(
        dockerfile=("FROM docker.io/library/busybox:1.36\nRUN echo 'hello from nemo-builder' > /hello.txt\n"),
        hello="hello from nemo-builder",
    ),
    # Its last RUN's output ends without a newline, so the sandbox's log does too.
    "done": Context(
        dockerfile=(
            "FROM docker.io/library/busybox:1.36\n"
            "RUN printf 'built in a second sandbox' > /hello.txt\n"
            "RUN printf done\n"
        ),
        hello="built in a second sandbox",
    ),
}


@dataclass(frozen=True)
class Kube:
    """Runs one-off pods in the release namespace, on the named context only."""

    context: str
    namespace: str

    def run(
        self,
        purpose: str,
        image: str,
        args: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        command: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``args`` in a new pod and return how it ended. The pod is deleted when it ends."""
        cmd = [
            "kubectl",
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            "run",
            f"builder-e2e-{purpose}-{uuid.uuid4().hex[:6]}",
            "--rm",
            "-i",
            "--restart=Never",
            "--pod-running-timeout=5m",
            f"--image={image}",
        ]
        cmd += [f"--env={name}={value}" for name, value in (env or {}).items()]
        cmd += ["--command", "--", *args] if command else ["--", *args]
        return subprocess.run(
            cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=POD_TIMEOUT_SECONDS, check=False
        )


@dataclass(frozen=True)
class Build:
    workspace: str
    job: str
    #: Spec name to image row name.
    images: dict[str, str]
    public_key: str


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required; see this module's docstring")
    return value


def _pem_pair() -> tuple[str, str]:
    """A new EC P-256 key, as the push step takes it (PKCS#8, unencrypted), and its public key."""
    key = ec.generate_private_key(ec.SECP256R1())
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return private, public


def _ended(result: subprocess.CompletedProcess[str]) -> str:
    return f"exit {result.returncode}\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}"


@pytest.fixture(scope="module")
def auth_on(request: pytest.FixtureRequest, client: NemoClient) -> bool:
    """Whether the platform runs with auth on, which must be what ``--feature auth`` says."""
    expected = "auth" in {feature.lower() for feature in request.config.getoption("--feature") or []}
    actual = discover_nhx_config(str(client.base_url)).auth_enabled
    assert actual == expected, f"platform auth is {'on' if actual else 'off'}, but --feature auth is {expected}"
    return actual


@pytest.fixture(scope="module")
def kube() -> Kube:
    return Kube(
        context=_required_env("NHX_E2E_KUBE_CONTEXT"),
        namespace=os.environ.get("NHX_E2E_KUBE_NAMESPACE") or "nemo-helix",
    )


@pytest.fixture(scope="module")
def registry_password() -> str:
    return _required_env("NHX_E2E_BUILDER_REGISTRY_PASSWORD")


@pytest.fixture(scope="module")
def builder_workspace(client: NemoClient) -> Iterator[str]:
    """A workspace for this module's one build. A name that is also a valid repository path component."""
    workspaces = WorkspacesClient.from_client(client)
    name = f"e2e-builder-{uuid.uuid4().hex[:8]}"
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
    yield name
    try:
        workspaces.delete_workspace(name=name).data()
    except Exception:
        pass  # Best effort; the cluster is discarded after the run


@pytest.fixture(scope="module")
def build(client: NemoClient, auth_on: bool, builder_workspace: str, registry_password: str) -> Build:  # noqa: ARG001
    """Create the workspace's builder secrets and two contexts, and submit one build set that builds both."""
    workspace = builder_workspace
    private_key, public_key = _pem_pair()

    secrets = SecretsClient.from_client(client)
    for name, value in (
        ("builder-registry-username", REGISTRY_USERNAME),
        ("builder-registry-password", registry_password),
        ("builder-signing-key", private_key),
    ):
        secrets.create_secret(
            workspace=workspace, body=HelixSecretCreateRequest(name=name, value=SecretStr(value))
        ).data()

    files = FilesClient.from_client(client)
    specs = []
    for spec, context in CONTEXTS.items():
        fileset = f"context-{spec}"
        files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=fileset))
        files.upload_file(workspace=workspace, name=fileset, path="Dockerfile", content=context.dockerfile.encode())
        specs.append(
            BuildSpec(
                name=spec,
                source=FileSetSource(fileset=fileset),
                output=BuildOutput(repository=f"e2e/{spec}", tag=TAG),
            )
        )

    submitted = (
        BuilderClient.from_client(client)
        .submit_build(workspace=workspace, body=BuildSet(name=BUILD_SET, revision=1, build_specs=specs))
        .data()
    )
    images = {spec: image.name for image in submitted.images if (spec := image.provenance.spec) is not None}
    assert set(images) == set(CONTEXTS), f"submit returned rows {[image.name for image in submitted.images]}"
    return Build(workspace=workspace, job=submitted.job, images=images, public_key=public_key)


def _wait_ready(builder: BuilderClient, workspace: str, names: Sequence[str]) -> dict[str, ContainerImage]:
    deadline = time.monotonic() + BUILD_DEADLINE_SECONDS
    while True:
        rows = {name: builder.get_container_image(workspace=workspace, name=name).data() for name in names}
        failed = {name: row.status_detail for name, row in rows.items() if row.status == "failed"}
        assert not failed, f"images failed: {failed}"
        if all(row.status == "ready" for row in rows.values()):
            return rows
        if time.monotonic() > deadline:
            statuses = {name: row.status for name, row in rows.items()}
            pytest.fail(f"images not ready after {BUILD_DEADLINE_SECONDS}s: {statuses}")
        time.sleep(10)


# First in the module, so it runs while the images are most likely still queued or pending.
@pytest.mark.builder_auth_e2e
@pytest.mark.feature("builder", "auth")
def test_complete_refuses_callers_not_acting_for_the_submitter(client: NemoClient, build: Build) -> None:
    """With auth on and token exchange off, only a caller acting for the image's submitter may complete it.

    The e2e harness authenticates with principal headers, so "a user's own token" here is the submitter's own
    identity, and the services are ``service:builder``, the push step's own, acting for no one or someone else.
    """
    builder = BuilderClient.from_client(client)
    name = build.images["hello"]
    row = builder.get_container_image(workspace=build.workspace, name=name).data()
    submitter = row.created_by
    assert submitter, f"{name} records no submitter"
    bogus = "sha256:" + "0" * 64
    # An Editor, who may complete images of their own: so only the builder's own check can refuse them.
    editor = f"editor-{uuid.uuid4().hex[:8]}@example.com"
    grant_workspace_role(client, workspace=build.workspace, principal=editor, roles=["Editor"])
    service = as_user(client, "service:builder")

    # Each caller, and the builder's reason for refusing it. None for the non-member: Entities refuses to read the
    # row as them first.
    callers: dict[str, tuple[NemoClient, str | None]] = {
        "the submitter's own identity": (
            as_user(client, submitter),
            "only a job step acting for the image's submitter may complete it",
        ),
        "a service acting for no one": (service, "only a job step acting for the image's submitter may complete it"),
        "a service acting for another Editor": (
            service.with_headers({ON_BEHALF_OF_HEADER: editor}),
            "was submitted by someone else",
        ),
        "a service acting for a non-member": (
            service.with_headers({ON_BEHALF_OF_HEADER: f"stranger-{uuid.uuid4().hex[:8]}@example.com"}),
            None,
        ),
    }
    for who, (caller, reason) in callers.items():
        with pytest.raises(PermissionDeniedError) as refused:
            BuilderClient.from_client(caller).complete_container_image(
                workspace=build.workspace, name=name, body=CompleteRequest(digest=bogus)
            )
        if reason is not None:
            # The builder's own refusal, not the policy's: the policy admits these callers.
            assert reason in str(refused.value.detail), f"{who}: {refused.value.detail}"

    after = builder.get_container_image(workspace=build.workspace, name=name).data()
    assert after.digest != bogus, f"{name} was completed at the refused digest"


def test_images_are_published_and_signed(
    client: NemoClient, build: Build, kube: Kube, registry_password: str, image: Any
) -> None:
    """Both images become ready; the registry serves each at its recorded digest, and cosign verifies its signature."""
    builder = BuilderClient.from_client(client)
    rows = _wait_ready(builder, build.workspace, list(build.images.values()))
    tasks_image = image("nhx-builder-tasks")
    _, other_public_key = _pem_pair()

    for spec, name in build.images.items():
        row = rows[name]
        assert row.digest and row.digest.startswith("sha256:"), f"{name} is ready without a digest: {row.digest}"
        job = f"{build.workspace}/{build.job}"
        assert row.provenance.job == job, f"{name} names job {row.provenance.job}, not {job}"
        ref = f"{row.registry}/{row.repository}"

        checked = kube.run(
            "check",
            tasks_image,
            [
                "sh",
                "-c",
                "set -e; export HOME=/tmp\n"
                'crane auth login "${REF%%/*}" -u "$USERNAME" -p "$PASSWORD" >/dev/null 2>&1\n'
                'AT_TAG=$(crane digest --insecure "$REF:$TAG")\n'
                '[ "$AT_TAG" = "$DIGEST" ] || { echo "$REF:$TAG is $AT_TAG, not $DIGEST" >&2; exit 1; }\n'
                'HELLO=$(crane export --insecure "$REF@$DIGEST" - | tar -xO hello.txt)\n'
                '[ "$HELLO" = "$EXPECTED" ] || { echo "hello.txt holds [$HELLO], not [$EXPECTED]" >&2; exit 1; }\n',
            ],
            env={
                "REF": ref,
                "TAG": TAG,
                "DIGEST": row.digest,
                "EXPECTED": CONTEXTS[spec].hello,
                "USERNAME": REGISTRY_USERNAME,
                "PASSWORD": registry_password,
            },
            command=True,
        )
        assert checked.returncode == 0, f"{name}: registry check failed: {_ended(checked)}"

        verify_args = [
            "verify",
            "--key",
            "env://PUB",
            "--insecure-ignore-tlog=true",
            "--allow-http-registry",
            "--registry-username",
            REGISTRY_USERNAME,
            "--registry-password",
            registry_password,
            "-a",
            f"{ANNOTATION_PREFIX}workspace={build.workspace}",
            "-a",
            f"{ANNOTATION_PREFIX}image={name}",
            f"{ref}@{row.digest}",
        ]
        verified = kube.run("verify", COSIGN_IMAGE, verify_args, env={"PUB": build.public_key})
        assert verified.returncode == 0, f"{name}: cosign verify failed: {_ended(verified)}"

        # And not with any other key, so the check above can fail.
        refused = kube.run("verify-other-key", COSIGN_IMAGE, verify_args, env={"PUB": other_public_key})
        assert refused.returncode != 0, f"{name}: cosign verified the signature with an unrelated key"
