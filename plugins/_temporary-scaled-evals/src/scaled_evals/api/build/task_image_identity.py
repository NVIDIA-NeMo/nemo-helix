# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry-backed identity validation for finalized task images."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx

from scaled_evals.api.settings import settings


class TaskImageIdentityError(ValueError):
    """A task image failed syntax, policy, or registry validation."""


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_REGISTRY = re.compile(r"(?:localhost|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[0-9]{1,5})?")
_REPOSITORY_COMPONENT = re.compile(r"[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*")
_TAG = re.compile(r"[\w][\w.-]{0,127}")
_BEARER_PARAMETER = re.compile(r'(\w+)="([^"]*)"')
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


@dataclass(frozen=True, slots=True)
class TaskImageReference:
    registry: str
    repository: str
    tag: str | None = None
    digest: str | None = None

    @property
    def repository_ref(self) -> str:
        return f"{self.registry}/{self.repository}"

    @property
    def normalized_ref(self) -> str:
        if self.digest:
            return f"{self.repository_ref}@{self.digest}"
        assert self.tag is not None
        return f"{self.repository_ref}:{self.tag}"

    def digest_ref(self, digest: str) -> str:
        return f"{self.repository_ref}@{normalize_digest(digest)}"


@dataclass(frozen=True, slots=True)
class ResolvedTaskImage:
    runtime_ref: str
    digest: str
    immutable_ref: str


def normalize_digest(value: str) -> str:
    digest = value.strip().lower()
    if not _DIGEST.fullmatch(digest):
        raise TaskImageIdentityError("task image digest must be sha256:<64 lowercase hex>")
    return digest


def _expected_digest(value: str, ref: TaskImageReference) -> str:
    raw = value.strip()
    if "@" not in raw:
        return normalize_digest(raw)
    expected_ref = parse_task_image_ref(raw)
    if expected_ref.digest is None or expected_ref.repository_ref != ref.repository_ref:
        raise TaskImageIdentityError(
            "task image digest identity does not match the image_ref repository"
        )
    return expected_ref.digest


def parse_task_image_ref(image_ref: str) -> TaskImageReference:
    """Parse an explicit-registry image tag or digest reference."""

    value = image_ref.strip()
    if not value or any(char.isspace() for char in value) or "://" in value:
        raise TaskImageIdentityError("task image_ref must be an OCI reference without a URL scheme")

    name = value
    digest: str | None = None
    if "@" in value:
        if value.count("@") != 1:
            raise TaskImageIdentityError("task image_ref contains more than one digest separator")
        name, raw_digest = value.rsplit("@", 1)
        digest = normalize_digest(raw_digest)

    if "/" not in name:
        raise TaskImageIdentityError("task image_ref must include an explicit registry hostname")
    registry, repository_with_tag = name.split("/", 1)
    registry = registry.lower()
    if not _REGISTRY.fullmatch(registry) or not (
        registry == "localhost" or "." in registry or ":" in registry
    ):
        raise TaskImageIdentityError(f"invalid task image registry hostname: {registry!r}")

    parts = repository_with_tag.split("/")
    tag: str | None = None
    last = parts[-1]
    if ":" in last:
        last, tag = last.rsplit(":", 1)
        parts[-1] = last
        if not _TAG.fullmatch(tag):
            raise TaskImageIdentityError(f"invalid task image tag: {tag!r}")
    repository = "/".join(parts)
    if not repository or any(not _REPOSITORY_COMPONENT.fullmatch(part) for part in parts):
        raise TaskImageIdentityError(f"invalid task image repository path: {repository!r}")
    if digest and tag:
        raise TaskImageIdentityError(
            "task image_ref must use a tag or a digest, not tag-plus-digest form"
        )
    if not digest and not tag:
        raise TaskImageIdentityError("task image_ref must include an explicit tag or digest")
    return TaskImageReference(registry, repository, tag=tag, digest=digest)


def normalize_upstream_image_ref(image_ref: str) -> str:
    """Expand Docker-compatible shorthand before resolving an upstream image."""

    value = image_ref.strip()
    if not value or any(char.isspace() for char in value) or "://" in value:
        raise TaskImageIdentityError("upstream image must be an OCI reference")
    name = value.split("@", 1)[0]
    first = name.split("/", 1)[0]
    if "/" not in name or ("." not in first and ":" not in first and first != "localhost"):
        value = f"docker.io/{value}"
        name = value.split("@", 1)[0]
    repository = name.split("/", 1)[1]
    if "/" not in repository:
        value = value.replace("docker.io/", "docker.io/library/", 1)
    if "@" not in value and ":" not in value.rsplit("/", 1)[-1]:
        value = f"{value}:latest"
    return parse_task_image_ref(value).normalized_ref


def resolve_upstream_image(
    image_ref: str,
    *,
    client: httpx.Client | None = None,
) -> ResolvedTaskImage:
    """Resolve a publicly readable source image without destination policy.

    This identity is used only as an immutable ``FROM`` input to a deployment-
    managed build. Registry credentials are deliberately ignored: Harbor
    dataset imports are a shared cache of public content, not a private-image
    import path. The source image is never admitted directly as a task image.
    """

    ref = parse_task_image_ref(normalize_upstream_image_ref(image_ref))
    _require_allowed_upstream_registry(ref)
    reference = ref.digest or ref.tag
    assert reference is not None
    response = _request_manifest(
        ref,
        reference,
        client=client,
        allow_registry_credentials=False,
    )
    actual = response.headers.get("Docker-Content-Digest", "").strip().lower()
    if not actual and response.content:
        actual = f"sha256:{hashlib.sha256(response.content).hexdigest()}"
    try:
        actual = normalize_digest(actual)
    except TaskImageIdentityError as exc:
        raise TaskImageIdentityError(
            "upstream registry did not return a verifiable sha256 manifest digest"
        ) from exc
    expected = ref.digest
    if expected and actual != expected:
        raise TaskImageIdentityError(
            f"upstream image digest mismatch: expected {expected}, registry resolved {actual}"
        )
    return ResolvedTaskImage(
        runtime_ref=ref.normalized_ref,
        digest=actual,
        immutable_ref=ref.digest_ref(actual),
    )


def _policy_entries(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower().rstrip("/") for item in value.split(",") if item.strip())


def _require_allowed_upstream_registry(ref: TaskImageReference) -> None:
    registries = _policy_entries(settings.harbor_dataset_upstream_allowed_registries)
    if ref.registry not in registries:
        raise TaskImageIdentityError(
            f"upstream image registry {ref.registry!r} is not approved; allowed registries: "
            + ", ".join(registries)
        )


def require_allowed_task_image(ref: TaskImageReference) -> None:
    registries = _policy_entries(settings.task_image_allowed_registries)
    if not registries:
        raise TaskImageIdentityError(
            "task images are disabled because no approved registries are configured"
        )
    if ref.registry not in registries:
        raise TaskImageIdentityError(
            f"task image registry {ref.registry!r} is not approved; allowed registries: "
            + ", ".join(registries)
        )

    entries = _policy_entries(settings.task_image_allowed_repositories)
    if not entries:
        return
    candidate = ref.repository_ref.lower()
    for entry in entries:
        if entry.endswith("/*") and candidate.startswith(f"{entry[:-2]}/"):
            return
        if candidate == entry:
            return
    raise TaskImageIdentityError(
        f"task image repository {candidate!r} is not approved; allowed repositories: "
        + ", ".join(entries)
    )


def _require_runtime_reference_shape(ref: TaskImageReference) -> None:
    if settings.task_image_hosted_mode and ref.digest:
        raise TaskImageIdentityError(
            "hosted task images must use the signed tag-form reference; "
            "digest-only and tag-plus-digest references are not admitted by RHACS"
        )


def validate_task_image_request(image_ref: str, image_digest: str | None = None) -> str:
    """Validate request syntax and operator policy without registry I/O."""

    ref = parse_task_image_ref(image_ref)
    require_allowed_task_image(ref)
    _require_runtime_reference_shape(ref)
    if image_digest:
        claimed = _expected_digest(image_digest, ref)
        if ref.digest and claimed != ref.digest:
            raise TaskImageIdentityError(
                f"image_digest {claimed!r} does not match image_ref digest {ref.digest!r}"
            )
    return ref.normalized_ref


def resolve_task_image(
    image_ref: str,
    *,
    expected_digest: str | None = None,
    client: httpx.Client | None = None,
) -> ResolvedTaskImage:
    """Resolve a registry manifest and return its immutable identity."""

    ref = parse_task_image_ref(image_ref)
    require_allowed_task_image(ref)
    _require_runtime_reference_shape(ref)
    expected = _expected_digest(expected_digest, ref) if expected_digest else ref.digest
    reference = ref.digest or ref.tag
    assert reference is not None
    response = _request_manifest(ref, reference, client=client)

    actual = response.headers.get("Docker-Content-Digest", "").strip().lower()
    if not actual and response.content:
        actual = f"sha256:{hashlib.sha256(response.content).hexdigest()}"
    try:
        actual = normalize_digest(actual)
    except TaskImageIdentityError as exc:
        raise TaskImageIdentityError(
            "registry did not return a verifiable sha256 manifest digest"
        ) from exc
    if expected and actual != expected:
        raise TaskImageIdentityError(
            f"task image digest mismatch: expected {expected}, registry resolved {actual}"
        )
    return ResolvedTaskImage(
        runtime_ref=ref.normalized_ref,
        digest=actual,
        immutable_ref=ref.digest_ref(actual),
    )


def verify_stored_task_image(
    image_ref: str,
    image_digest: str | None,
    *,
    client: httpx.Client | None = None,
) -> None:
    """Fail closed if stored identity is absent, inconsistent, or a tag moved."""

    if settings.task_image_validation_mode == "disabled":
        return
    if not image_digest:
        raise TaskImageIdentityError(
            "task image identity is not verified: the finalized revision has no immutable digest; "
            "create, upload, and finalize a new revision"
        )
    ref = parse_task_image_ref(image_ref)
    digest = _expected_digest(image_digest, ref)
    require_allowed_task_image(ref)
    _require_runtime_reference_shape(ref)
    if ref.digest:
        if ref.digest != digest:
            raise TaskImageIdentityError(
                f"stored task image reference digest {ref.digest} does not match {digest}"
            )
        return
    resolved = resolve_task_image(ref.normalized_ref, expected_digest=digest, client=client)
    if resolved.digest != digest:  # defensive; resolve_task_image already checks
        raise TaskImageIdentityError(
            f"task image tag moved: expected {digest}, registry resolved {resolved.digest}"
        )


def _request_manifest(
    ref: TaskImageReference,
    reference: str,
    *,
    client: httpx.Client | None,
    allow_registry_credentials: bool = True,
) -> httpx.Response:
    scheme = "http" if allow_registry_credentials and settings.task_image_registry_insecure else "https"
    registry_api_host = "registry-1.docker.io" if ref.registry == "docker.io" else ref.registry
    url = f"{scheme}://{registry_api_host}/v2/{ref.repository}/manifests/{reference}"
    registry_headers = _registry_auth_headers(ref.registry) if allow_registry_credentials else {}
    headers = {"Accept": _MANIFEST_ACCEPT, **registry_headers}
    owns_client = client is None
    client = client or httpx.Client(
        timeout=settings.task_image_registry_timeout_seconds,
        verify=not settings.task_image_registry_insecure,
        follow_redirects=False,
    )
    try:
        response = client.get(url, headers=headers)
        if response.status_code == 401:
            try:
                token = _registry_bearer_token(
                    client,
                    response,
                    ref,
                    registry_headers,
                    allow_insecure=allow_registry_credentials and settings.task_image_registry_insecure,
                )
            except TaskImageIdentityError as exc:
                if not allow_registry_credentials:
                    raise TaskImageIdentityError(
                        f"upstream image is not publicly readable: {ref.normalized_ref}"
                    ) from exc
                raise
            if token:
                response = client.get(
                    url,
                    headers={"Accept": _MANIFEST_ACCEPT, "Authorization": f"Bearer {token}"},
                )
    except httpx.HTTPError as exc:
        raise TaskImageIdentityError(
            f"could not reach registry {ref.registry!r} to verify task image: {exc}"
        ) from exc
    finally:
        if owns_client:
            client.close()

    if response.status_code == 401:
        if not allow_registry_credentials:
            raise TaskImageIdentityError(
                f"upstream image is not publicly readable: {ref.normalized_ref}"
            )
        raise TaskImageIdentityError(
            f"registry {ref.registry!r} rejected task image verification; configure readable "
            "registry credentials for the build and dispatch workers"
        )
    if response.status_code == 404:
        raise TaskImageIdentityError(f"task image manifest does not exist: {ref.normalized_ref}")
    if response.is_redirect:
        raise TaskImageIdentityError("registry manifest verification refused an HTTP redirect")
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise TaskImageIdentityError(
            f"registry manifest verification failed with HTTP {response.status_code}"
        ) from exc
    return response


def _registry_auth_headers(registry: str) -> dict[str, str]:
    entry = _registry_auth_entry(registry)
    if entry is None:
        return {}
    if token := entry.get("identitytoken") or entry.get("registrytoken"):
        return {"Authorization": f"Bearer {token}"}
    if encoded := entry.get("auth"):
        try:
            base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise TaskImageIdentityError(
                "registry auth file contains malformed basic credentials"
            ) from exc
        return {"Authorization": f"Basic {encoded}"}
    username = entry.get("username")
    password = entry.get("password")
    if username is not None and password is not None:
        encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    raise TaskImageIdentityError(f"registry auth file has no usable credentials for {registry!r}")


def _registry_auth_entry(registry: str) -> dict[str, str] | None:
    auth_file = settings.task_image_registry_auth_file
    auth_json = settings.task_image_registry_auth_json
    if not auth_file and not auth_json:
        return None
    try:
        document = json.loads(Path(auth_file).read_text() if auth_file else auth_json)
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskImageIdentityError(f"could not read registry auth configuration: {exc}") from exc
    for key, value in (document.get("auths") or {}).items():
        normalized = key.removeprefix("https://").removeprefix("http://").rstrip("/")
        if normalized == registry and isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items() if v is not None}
    return None


def _registry_bearer_token(
    client: httpx.Client,
    response: httpx.Response,
    ref: TaskImageReference,
    registry_headers: dict[str, str],
    *,
    allow_insecure: bool,
) -> str | None:
    challenge = response.headers.get("WWW-Authenticate", "")
    scheme, _, raw_parameters = challenge.partition(" ")
    if scheme.lower() != "bearer":
        return None
    parameters = dict(_BEARER_PARAMETER.findall(raw_parameters))
    realm = parameters.get("realm", "")
    parsed = urlsplit(realm)
    if not parsed.hostname or parsed.scheme not in {"http", "https"}:
        raise TaskImageIdentityError("registry returned an invalid bearer-token challenge")
    registry_host = ref.registry.split(":", 1)[0]
    allowed_hosts = {registry_host}
    if ref.registry == "docker.io":
        allowed_hosts.add("auth.docker.io")
    if parsed.hostname not in allowed_hosts:
        raise TaskImageIdentityError("registry bearer-token challenge host must match the registry")
    if parsed.scheme != "https" and not (allow_insecure and parsed.hostname == registry_host):
        raise TaskImageIdentityError("registry bearer-token challenge must use HTTPS")
    query = {
        key: value for key, value in parameters.items() if key in {"service", "scope"} and value
    }
    token_headers: dict[str, str] = {}
    if (
        parsed.scheme == "https"
        and parsed.hostname == registry_host
        and registry_headers.get("Authorization", "").startswith("Basic ")
    ):
        token_headers["Authorization"] = registry_headers["Authorization"]
    token_url = f"{realm}{'&' if '?' in realm else '?'}{urlencode(query)}" if query else realm
    try:
        token_response = client.get(token_url, headers=token_headers)
        token_response.raise_for_status()
        payload = token_response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise TaskImageIdentityError("registry bearer-token request failed") from exc
    token = payload.get("token") or payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise TaskImageIdentityError("registry bearer-token response contained no token")
    return token
