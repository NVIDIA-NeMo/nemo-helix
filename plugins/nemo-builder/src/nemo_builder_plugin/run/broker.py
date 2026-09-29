# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build broker`` -- the credential broker. Trusted: it holds the registry credential and the key.

The build plane's one credential authority (Requirement 7). It runs as its own Deployment, in its
own namespace, under its own ServiceAccount, and serves two requests, both authenticated by the
calling step's own pod token as the password of an HTTP Basic credential:

``POST /credentials``
    A registry token for the step's job's ``pending`` destinations, as the registry's own token
    service issued it -- as a Docker config ``registrytoken``, which crane sends directly. The
    step names no scope: the broker decides it.

``POST /sign`` with ``{"image": <row name>}``
    The broker resolves the row's system tag in the registry *itself*, signs that digest with a
    key no step can read, every annotation taken from the row, and returns what it made: the
    signed payload and its signature, for the step to deliver to the control plane.

**Identity.** The token goes to ``TokenReview``, which names the ServiceAccount and the pod it is
bound to. The broker reads that pod and requires Jobs made it for a step role its backend knows
(:func:`~nemo_builder_plugin.broker.identify_pod`). The job and workspace are then the labels Jobs
wrote, which the step cannot choose.

- With platform auth on, the step presents the audience-bound workload token Jobs projects for
  token exchange (``step_token_audience``). The broker cannot replay it at the API server.
- With platform auth off, the step presents its ordinary token, because a Jobs profile cannot
  project one for another audience. The broker could replay it, but the identity it names holds
  no RBAC, so it opens nothing.

**Rows.** The broker reads the job's rows through the builder's list route -- anonymously with
auth off, and with auth on as the submitter, by exchanging the step's own token at the platform's
token exchange (``token_exchange``). The platform issues that token only while Jobs' delegation
for the pod is live: a second check that the step is running. So the broker has no platform
identity of its own, and reads only what the submitter could.

**Registry credentials** come from the registry's token service (``registry_auth.py``), which
narrows a token to what is asked for where it follows the specification.

**Fail closed.** A credential that identifies no step is a ``401``, an entitlement that does not
cover the request a ``403``, and anything the decision depends on that cannot be read a ``503`` --
never a token. Every token issued, every signature -- before it is made, naming the digest -- and
every refusal of an identified step is an audit line on stdout, naming the pod and the job and
never a token. If a token's or a signature's line cannot be written, nothing is issued and nothing
is signed. A request that identifies no step, or that cannot be decided, is logged instead.

**Plain HTTP, to job steps only.** The broker's NetworkPolicy admits only pods Jobs created in the
build namespace, and each request holds a thread only briefly and only up to a bound. With platform
auth on, a step's workload token crosses this connection, so TLS belongs in front of it -- a
service mesh does that.

**What it never does:** write a row, accept a requested scope, or call another trust domain's
broker.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
import yaml
from nemo_builder_plugin.backend import Backend, StepEntitlement
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.broker import (
    NotIdentified,
    Pod,
    StepIdentity,
    entitled_rows,
    identify_pod,
    signing_refusal,
)
from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.registry_auth import (
    DistributionTokenClient,
    FileCredential,
    RegistryAuthError,
    RegistryCredentials,
)
from nemo_builder_plugin.run.tools import run_tool
from nemo_builder_plugin.signing import signature_annotations
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError
from nemo_helix_plugin.client.oidc import WorkloadTokenExchangeError, token_exchange_grant
from nemo_helix_plugin.log_utils import sanitize_for_log
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

#: The claims a bound ServiceAccount token carries about its pod, as ``TokenReview`` reports them.
POD_NAME_EXTRA = "authentication.kubernetes.io/pod-name"
POD_UID_EXTRA = "authentication.kubernetes.io/pod-uid"

#: A `POST /sign` body is `{"image": "<row name>"}`. Anything much larger is not one.
_MAX_BODY = 4096

#: How long a client may leave its connection idle before it is dropped, and its thread freed.
_SOCKET_TIMEOUT_SECONDS = 10.0

#: Requests served at once. Every build shares the broker, so a connection beyond this is closed
#: at once rather than given a thread.
_MAX_CONCURRENT_REQUESTS = 32

_ROWS_PAGE_SIZE = 100

#: The OAuth error codes with which a token endpoint refuses a grant (RFC 6749 section 5.2, RFC 8693
#: section 2.2.2). Anything else -- a 5xx, a reply that is not JSON or carries no token -- is the
#: platform failing, which says nothing about the step.
_EXCHANGE_REFUSALS = frozenset(
    {
        "access_denied",
        "invalid_client",
        "invalid_grant",
        "invalid_request",
        "invalid_scope",
        "invalid_target",
        "unauthorized_client",
        "unsupported_grant_type",
    }
)

#: How ``token_exchange_grant`` reports the endpoint's error code. A test holds it to this.
_EXCHANGE_ERROR_CODE = re.compile(r"^Workload token exchange failed: (\S+) - ")

_INDEX_MEDIA_TYPES = frozenset(
    {"application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json"}
)


def _is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class TokenExchangeConfig(BaseModel):
    """The platform's workload token exchange, for reading rows as the submitter (platform auth on)."""

    model_config = ConfigDict(extra="forbid")

    token_endpoint: str
    client_id: str
    audience: str | None = None
    scope: str | None = None

    @field_validator("token_endpoint")
    @classmethod
    def _https_unless_loopback(cls, value: str) -> str:
        """The rule ``token_exchange_grant`` applies on every call, checked once, at start.

        A step's token goes to this endpoint, so over plain HTTP only to this host. An in-cluster
        ``http://`` Service address is refused here rather than failing every request.
        """
        url = urlsplit(value)
        if (url.scheme == "https" and url.hostname) or (url.scheme == "http" and _is_loopback(url.hostname)):
            return value
        raise ValueError(f"token_endpoint must be HTTPS, or HTTP to a loopback address; got {value!r}")


class BrokerConfig(BaseModel):
    """The broker's own settings, from the YAML file ``NHX_BROKER_CONFIG`` names.

    What it shares with the platform -- the backend, the registry, the prefix, the build namespace
    -- it does NOT restate: it reads the same ``builder:`` section the platform does, so the two
    cannot disagree about what a grant is bounded by.
    """

    model_config = ConfigDict(extra="forbid")

    port: int = 8080
    platform_url: str = Field(description="Where the builder's routes are: rows are read from its list route.")
    step_token_audience: str | None = Field(
        default=None,
        description=(
            "The audience a step's token must be bound to. Unset: the API server's own, which is "
            "what a step pod carries with platform auth off. With auth on, the workload identity "
            "audience Jobs projects for token exchange."
        ),
    )
    token_exchange: TokenExchangeConfig | None = Field(
        default=None,
        description=(
            "Set with platform auth on: rows are read as the submitter, with the step's token "
            "exchanged here. Unset: rows are read anonymously, which works only with auth off."
        ),
    )
    registry_username: str = Field(description="The username the broker presents to the registry's token service.")
    registry_password_file: str = Field(
        description="The broker's own registry password: a mounted Secret, or a projected token. Read on every use."
    )
    signing_key: str = Field(
        description="cosign key reference: a file path, or a KMS URI such as `gcpkms://...`. Only the broker holds it."
    )
    signing_key_password_file: str | None = Field(default=None, description="The key's password, if it has one.")
    request_timeout_seconds: float = Field(default=30.0, gt=0)


# --- Collaborators, injected so each can be held still in a test ------------------------------


@dataclass(frozen=True, slots=True)
class Review:
    authenticated: bool
    username: str = ""
    extra: Mapping[str, list[str]] = field(default_factory=dict)


class TokenReviewer(Protocol):
    def review(self, token: str, audiences: list[str] | None) -> Review: ...


class PodReader(Protocol):
    def read(self, namespace: str, name: str) -> Pod | None: ...


class RowSource(Protocol):
    """A job's rows, read as whoever the step's token lets the broker be."""

    def pending(self, identity: StepIdentity, token: str) -> list[ContainerImage]: ...

    def get(self, identity: StepIdentity, token: str, name: str) -> ContainerImage | None: ...


@dataclass(frozen=True, slots=True)
class SignedImage:
    """What signing produced: the digest the broker resolved, and the signature over it."""

    digest: str
    payload: bytes
    signature: bytes


class ImageSigner(Protocol):
    def sign(
        self, row: ContainerImage, annotations: Mapping[str, str], *, record: Callable[[str], None]
    ) -> SignedImage:
        """Resolve ``row``'s system tag and sign that digest.

        ``record`` is called with the digest once it is resolved and checked, and before anything
        is signed. It is the audit line: if it raises, nothing is signed.

        Raises :class:`SigningFailed` when the registry or cosign fail.
        """
        ...


class BrokerUnavailable(Exception):
    """Something the decision depends on could not be read. Nothing is issued (503)."""


class AuditUnavailable(Exception):
    """A token's or a signature's audit line could not be written. Nothing is issued (503)."""


class SigningFailed(Exception):
    """The registry or cosign failed while resolving or signing. Nothing was signed (502)."""


@dataclass(frozen=True, slots=True)
class Reply:
    status: int
    body: dict[str, object]
    headers: dict[str, str] = field(default_factory=dict)


def write_audit(record: Mapping[str, object]) -> None:
    """One decision, one line on stdout. Raises if it cannot be written -- and then nothing is issued."""
    sys.stdout.write("audit " + json.dumps(record, sort_keys=True) + "\n")
    sys.stdout.flush()


def _basic_password(authorization: str | None) -> str | None:
    """The password half of a Basic header. The username is ignored: it proves nothing."""
    if not authorization or not authorization.startswith("Basic "):
        return None
    try:
        decoded = base64.b64decode(authorization[len("Basic ") :], validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    _, sep, password = decoded.partition(":")
    return password if sep and password else None


def _error(status: HTTPStatus, code: str, message: str | None = None, **headers: str) -> Reply:
    body: dict[str, object] = {"errors": [{"code": code, **({"message": message} if message else {})}]}
    return Reply(status, body, dict(headers))


class Broker:
    """The broker's decisions, with every collaborator injected."""

    def __init__(
        self,
        builder: BuilderConfig,
        config: BrokerConfig,
        *,
        backend: Backend,
        reviewer: TokenReviewer,
        pods: PodReader,
        rows: RowSource,
        registry: RegistryCredentials,
        signer: ImageSigner,
        audit: Callable[[Mapping[str, object]], None] = write_audit,
    ) -> None:
        if not builder.registry:
            raise ValueError("builder.registry is not configured; the broker has no registry to serve")
        self._registry_host = builder.registry
        self._prefix = builder.repository_prefix
        self._namespace = builder.namespace
        self._audience = config.step_token_audience
        self._entitlements: Mapping[str, StepEntitlement] = backend.entitlements()
        self._reviewer = reviewer
        self._pods = pods
        self._rows = rows
        self._registry = registry
        self._signer = signer
        self._audit = audit

    # --- Identity ----------------------------------------------------------------------------

    def identify(self, token: str) -> StepIdentity:
        """Which step holds ``token``, as the API server and Jobs say -- or :class:`NotIdentified`."""
        audiences = [self._audience] if self._audience else None
        review = self._reviewer.review(token, audiences)
        if not review.authenticated:
            raise NotIdentified(
                "the API server does not accept the token"
                + (f" for audience {self._audience}" if self._audience else "")
            )
        prefix = f"system:serviceaccount:{self._namespace}:"
        if not review.username.startswith(prefix):
            raise NotIdentified(f"{review.username} is not a ServiceAccount in the build namespace {self._namespace}")
        names, uids = review.extra.get(POD_NAME_EXTRA, []), review.extra.get(POD_UID_EXTRA, [])
        if len(names) != 1 or len(uids) != 1:
            # A legacy Secret-based token, or one minted by hand: bound to no pod, so to no job.
            raise NotIdentified(f"{review.username}'s token is not bound to a pod")
        pod = self._pods.read(self._namespace, names[0])
        if pod is None:
            raise NotIdentified(f"pod {names[0]}, which the token is bound to, no longer exists")
        return identify_pod(
            pod,
            token_uid=uids[0],
            token_service_account=review.username.removeprefix(prefix),
            entitlements=self._entitlements,
        )

    def _authenticate(self, authorization: str | None) -> tuple[str, StepIdentity] | Reply:
        """The step's token and its identity, or the reply refusing it. Every refusal is logged."""
        token = _basic_password(authorization)
        if token is None:
            return _error(
                HTTPStatus.UNAUTHORIZED, "UNAUTHORIZED", **{"WWW-Authenticate": 'Basic realm="nhx-build-broker"'}
            )
        try:
            return token, self.identify(token)
        except NotIdentified as exc:
            logger.warning("refused: %s", sanitize_for_log(exc))
            return _error(
                HTTPStatus.UNAUTHORIZED, "UNAUTHORIZED", **{"WWW-Authenticate": 'Basic realm="nhx-build-broker"'}
            )
        except BrokerUnavailable as exc:
            logger.error("refused, cannot decide: %s", sanitize_for_log(exc))
            return _error(HTTPStatus.SERVICE_UNAVAILABLE, "UNAVAILABLE")

    # --- POST /credentials ---------------------------------------------------------------------

    def credentials(self, *, authorization: str | None) -> Reply:
        """A registry token for exactly the caller's job's pending destinations."""
        authenticated = self._authenticate(authorization)
        if isinstance(authenticated, Reply):
            return authenticated
        token, identity = authenticated
        actions = self._entitlements[identity.role].destinations
        if not actions:
            return self._deny(identity, "credentials", f"role {identity.role} is entitled to no registry access")
        try:
            rows = self._rows.pending(identity, token)
        except NotIdentified as exc:
            logger.warning("credentials refused for pod %s: %s", sanitize_for_log(identity.pod), sanitize_for_log(exc))
            return _error(HTTPStatus.UNAUTHORIZED, "UNAUTHORIZED")
        except BrokerUnavailable as exc:
            logger.error(
                "credentials refused, cannot read %s's rows: %s",
                sanitize_for_log(identity.job_ref),
                sanitize_for_log(exc),
            )
            return _error(HTTPStatus.SERVICE_UNAVAILABLE, "UNAVAILABLE")
        repositories = sorted(
            {
                row.repository
                for row in entitled_rows(rows, identity, registry=self._registry_host, repository_prefix=self._prefix)
            }
        )
        if not repositories:
            return self._deny(identity, "credentials", f"job {identity.job_ref} has no pending images on this registry")
        try:
            issued = self._registry.issue(repositories, actions)
        except RegistryAuthError as exc:
            logger.error(
                "credentials for %s failed at the registry: %s",
                sanitize_for_log(identity.job_ref),
                sanitize_for_log(exc),
            )
            return _error(HTTPStatus.BAD_GATEWAY, "REGISTRY_UNAVAILABLE")
        try:
            self._audit(
                {
                    "decision": "credentials",
                    "granted": [f"{name}:{','.join(actions)}" for name in repositories],
                    "narrowed": issued.narrowed,
                    "expires_in": issued.expires_in,
                    **self._who(identity),
                }
            )
        except OSError:
            logger.exception("credentials withheld: the audit line could not be written")
            return _error(HTTPStatus.SERVICE_UNAVAILABLE, "AUDIT_UNAVAILABLE")
        return Reply(
            HTTPStatus.OK,
            {
                "auths": {self._registry_host: {"registrytoken": issued.token}},
                "expires_in": issued.expires_in,
                "narrowed": issued.narrowed,
                "repositories": repositories,
            },
        )

    # --- POST /sign ----------------------------------------------------------------------------

    def sign(self, *, authorization: str | None, body: bytes) -> Reply:
        """Sign one of the caller's job's images, as the broker, and return the signature."""
        authenticated = self._authenticate(authorization)
        if isinstance(authenticated, Reply):
            return authenticated
        token, identity = authenticated
        try:
            request = json.loads(body)
            image = request.get("image") if isinstance(request, dict) else None
        except ValueError:
            image = None
        if not isinstance(image, str) or not image:
            return _error(HTTPStatus.BAD_REQUEST, "BAD_REQUEST", 'expected {"image": <row name>}')
        if not self._entitlements[identity.role].sign:
            return self._deny(identity, "sign", f"role {identity.role} may not have images signed", image=image)
        try:
            row = self._rows.get(identity, token, image)
        except NotIdentified as exc:
            logger.warning("signature refused for pod %s: %s", sanitize_for_log(identity.pod), sanitize_for_log(exc))
            return _error(HTTPStatus.UNAUTHORIZED, "UNAUTHORIZED")
        except BrokerUnavailable as exc:
            logger.error("signature refused, cannot read %s: %s", sanitize_for_log(image), sanitize_for_log(exc))
            return _error(HTTPStatus.SERVICE_UNAVAILABLE, "UNAVAILABLE")
        refusal = signing_refusal(row, identity, registry=self._registry_host, repository_prefix=self._prefix)
        if refusal is not None or row is None:
            return self._deny(identity, "sign", refusal or "no such image", image=image)

        annotations = signature_annotations(row)

        def record(digest: str) -> None:
            # Written BEFORE signing, naming what is about to be signed: a signature outlives
            # everything else here, so one that could not be accounted for is never made.
            try:
                self._audit(
                    {
                        "decision": "sign",
                        "image": row.name,
                        "reference": f"{row.registry}/{row.repository}@{digest}",
                        **self._who(identity),
                    }
                )
            except OSError as exc:
                raise AuditUnavailable(str(exc)) from exc

        try:
            signed = self._signer.sign(row, annotations, record=record)
        except AuditUnavailable:
            logger.exception(
                "signature withheld for %s: the audit line could not be written", sanitize_for_log(row.name)
            )
            return _error(HTTPStatus.SERVICE_UNAVAILABLE, "AUDIT_UNAVAILABLE")
        except SigningFailed as exc:
            logger.error("signature failed for %s: %s", sanitize_for_log(row.name), sanitize_for_log(exc))
            self._audit_after({"decision": "sign-failed", "image": row.name, "reason": str(exc), **self._who(identity)})
            return _error(HTTPStatus.BAD_GATEWAY, "SIGNING_FAILED")
        reference = f"{row.registry}/{row.repository}@{signed.digest}"
        self._audit_after({"decision": "signed", "image": row.name, "reference": reference, **self._who(identity)})
        return Reply(
            HTTPStatus.OK,
            {
                "image": row.name,
                "digest": signed.digest,
                "reference": reference,
                "signed": {
                    "payload": base64.b64encode(signed.payload).decode(),
                    "signature": base64.b64encode(signed.signature).decode(),
                },
            },
        )

    # --- Audit ---------------------------------------------------------------------------------

    @staticmethod
    def _who(identity: StepIdentity) -> dict[str, object]:
        return {"pod": identity.pod, "job": identity.job_ref, "role": identity.role}

    def _audit_after(self, record: Mapping[str, object]) -> None:
        """An audit line nothing waits on -- a refusal, or an outcome. If it can't be written, that is logged."""
        try:
            self._audit(record)
        except OSError:
            logger.exception("the audit line for a %s could not be written", record.get("decision"))

    def _deny(self, identity: StepIdentity, decision: str, reason: str, **detail: str) -> Reply:
        logger.warning(
            "%s refused for pod %s: %s",
            sanitize_for_log(decision),
            sanitize_for_log(identity.pod),
            sanitize_for_log(reason),
        )
        self._audit_after({"decision": f"{decision}-refused", "reason": reason, **detail, **self._who(identity)})
        return _error(HTTPStatus.FORBIDDEN, "DENIED", reason)


# --- The Kubernetes, platform and registry sides ---------------------------------------------


class KubernetesReviewer:
    def __init__(self) -> None:
        from kubernetes import client

        self._api = client.AuthenticationV1Api()
        self._client = client

    def review(self, token: str, audiences: list[str] | None) -> Review:
        body = self._client.V1TokenReview(spec=self._client.V1TokenReviewSpec(token=token, audiences=audiences))
        try:
            status = self._api.create_token_review(body).status
        except Exception as exc:
            raise BrokerUnavailable(f"TokenReview failed: {type(exc).__name__}") from exc
        if not status or not status.authenticated or not status.user:
            return Review(authenticated=False)
        return Review(authenticated=True, username=status.user.username or "", extra=status.user.extra or {})


class KubernetesPods:
    def __init__(self) -> None:
        from kubernetes import client

        self._api = client.CoreV1Api()

    def read(self, namespace: str, name: str) -> Pod | None:
        from kubernetes.client.exceptions import ApiException

        try:
            pod = self._api.read_namespaced_pod(name=name, namespace=namespace)
        except ApiException as exc:
            if exc.status == HTTPStatus.NOT_FOUND:
                return None
            raise BrokerUnavailable(f"reading pod {namespace}/{name} failed: {exc.status}") from exc
        except Exception as exc:
            # The API server unreachable: urllib3's connection and protocol errors, not ApiExceptions.
            raise BrokerUnavailable(f"reading pod {namespace}/{name} failed: {type(exc).__name__}") from exc
        return Pod(
            name=name,
            uid=pod.metadata.uid or "",
            service_account=pod.spec.service_account_name or "",
            phase=(pod.status.phase if pod.status else "") or "",
            labels=dict(pod.metadata.labels or {}),
        )


class TokenExchange:
    """The step's own token, exchanged at the platform for one acting as the submitter."""

    def __init__(self, config: TokenExchangeConfig) -> None:
        self._config = config

    def __call__(self, subject_token: str) -> str:
        try:
            response = token_exchange_grant(
                token_endpoint=self._config.token_endpoint,
                client_id=self._config.client_id,
                subject_token=subject_token,
                audience=self._config.audience,
                scope=self._config.scope,
            )
        except WorkloadTokenExchangeError as exc:
            code = _EXCHANGE_ERROR_CODE.match(str(exc))
            if code is not None and code.group(1) in _EXCHANGE_REFUSALS:
                # The platform issues a token only while the pod's delegation is live, so a
                # refusal here says the step is not running as far as Jobs knows.
                raise NotIdentified(f"the platform would not exchange the step's token: {exc}") from exc
            raise BrokerUnavailable(f"the platform's token exchange failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise BrokerUnavailable(
                f"the platform's token exchange could not be reached: {type(exc).__name__}"
            ) from exc
        return str(response["access_token"])


class BuilderRouteRows:
    """A job's rows, from the builder's own list route: as the submitter, or anonymously.

    The exchange, when there is one, runs inside each read's error handling: what it raises is
    :class:`NotIdentified` or :class:`BrokerUnavailable`, and a read failing any other way is
    :class:`BrokerUnavailable` too.
    """

    def __init__(
        self, platform_url: str, *, exchange: Callable[[str], str] | None, http: httpx.Client | None = None
    ) -> None:
        self._base_url = platform_url.rstrip("/")
        self._exchange = exchange
        self._http = http or httpx.Client(timeout=15.0)

    def _client(self, token: str, workspace: str) -> BuilderClient:
        auth = self._exchange(token) if self._exchange is not None else None
        return BuilderClient(
            base_url=self._base_url, workspace=workspace, auth=auth, http_client=self._http, owns_http_client=False
        )

    def pending(self, identity: StepIdentity, token: str) -> list[ContainerImage]:
        rows: list[ContainerImage] = []
        page = 1
        try:
            client = self._client(token, identity.workspace)
            while True:
                batch = client.list_container_images(
                    query_params={"job": identity.job, "status": "pending", "page": page, "page_size": _ROWS_PAGE_SIZE}
                ).data()
                rows.extend(batch)
                if len(batch) < _ROWS_PAGE_SIZE:
                    return rows
                page += 1
        except (NemoClientError, httpx.HTTPError, ValueError) as exc:
            raise BrokerUnavailable(f"reading job {identity.job_ref}'s rows failed: {type(exc).__name__}") from exc

    def get(self, identity: StepIdentity, token: str, name: str) -> ContainerImage | None:
        try:
            return self._client(token, identity.workspace).get_container_image(name=name).data()
        except NotFoundError:
            return None
        except (NemoClientError, httpx.HTTPError, ValueError) as exc:
            raise BrokerUnavailable(f"reading {identity.workspace}/{name} failed: {type(exc).__name__}") from exc


def cosign_sign_args(
    reference: str,
    *,
    key: str,
    annotations: Mapping[str, str],
    output_payload: Path,
    output_signature: Path,
    plain_http: bool = False,
) -> list[str]:
    """The one cosign invocation this system signs with.

    Sign BY DIGEST, never by tag: a tag is mutable, and signing one races anything that could move
    it. So a reference without a digest is refused here rather than signed.

    ``plain_http`` reaches the registry over plain HTTP. cosign's flag for that is
    ``--allow-http-registry``; ``--allow-insecure-registry`` only skips TLS verification.
    """
    if "@sha256:" not in reference:
        raise ValueError(f"refusing to sign {reference!r}: sign by digest, never by tag")
    args = ["cosign", "sign"]
    if plain_http:
        args.append("--allow-http-registry")
    args += [
        f"--key={key}",
        # Mandatory, not optional: cosign v2 uploads to the PUBLIC Rekor transparency log by
        # default, which would publish internal image names.
        "--tlog-upload=false",
        "--yes",
        # What the step delivers: exactly the bytes signed, and the signature over them, as cosign made them.
        f"--output-payload={output_payload}",
        f"--output-signature={output_signature}",
    ]
    for name, value in sorted(annotations.items()):
        # cosign reads the flag as a comma-separated list, so a comma would split one annotation
        # into two -- the second one named by whatever followed it.
        if "," in name or "," in value or "=" in name:
            raise ValueError(f"annotation {name!r}={value!r} cannot be passed to cosign intact")
        args.append(f"--annotations={name}={value}")
    args.append(reference)
    return args


class CosignSigner:
    """Resolves with crane and signs with cosign, under a token the broker obtains for itself.

    The token is the registry token service's, for pull and push on the one repository -- push,
    because the signature is uploaded beside the image. It goes in a private Docker config for the
    one signature, as ``registrytoken``, and the directory is removed afterwards.
    """

    def __init__(self, builder: BuilderConfig, config: BrokerConfig, registry: RegistryCredentials) -> None:
        if not builder.registry:
            raise ValueError("builder.registry is not configured")
        self._host = builder.registry
        self._plain_http = builder.registry_plain_http
        self._key = config.signing_key
        self._password_file = config.signing_key_password_file
        self._registry = registry

    def _crane(self, *args: str, env: Mapping[str, str]) -> str:
        return run_tool(["crane", *args, *(["--insecure"] if self._plain_http else [])], env=env)

    def sign(
        self, row: ContainerImage, annotations: Mapping[str, str], *, record: Callable[[str], None]
    ) -> SignedImage:
        origin = row.provenance
        directory = Path(tempfile.mkdtemp(prefix="nhx-broker-"))
        try:
            issued = self._registry.issue([row.repository], ["pull", "push"])
            config_file = directory / "config.json"
            config_file.write_text(json.dumps({"auths": {self._host: {"registrytoken": issued.token}}}))
            config_file.chmod(0o600)
            # A mounted Secret's file often ends in a newline that was never part of the password.
            password = Path(self._password_file).read_text().rstrip("\r\n") if self._password_file else ""
            env = {"DOCKER_CONFIG": str(directory), "COSIGN_PASSWORD": password}

            # The system tag is on the row and only this system writes it; what it names NOW is what
            # gets signed. Never a digest the step supplied.
            digest = self._crane("digest", f"{self._host}/{row.repository}:{origin.system_tag}", env=env)
            if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
                raise SigningFailed(f"crane reported {digest!r} for {row.name}'s system tag, which is not a digest")
            reference = f"{self._host}/{row.repository}@{digest}"
            manifest = json.loads(self._crane("manifest", reference, env=env))
            if (
                not isinstance(manifest, dict)
                or manifest.get("mediaType") in _INDEX_MEDIA_TYPES
                or "manifests" in manifest
            ):
                raise SigningFailed(f"{reference} is an index; this broker signs single-platform manifests only")

            record(digest)
            payload_file, signature_file = directory / "payload.json", directory / "signature"
            run_tool(
                cosign_sign_args(
                    reference,
                    key=self._key,
                    annotations=annotations,
                    output_payload=payload_file,
                    output_signature=signature_file,
                    plain_http=self._plain_http,
                ),
                env=env,
            )
            # cosign writes the signature base64-encoded, the payload as the raw bytes it signed.
            signature = base64.b64decode(signature_file.read_text().strip(), validate=True)
            return SignedImage(digest=digest, payload=payload_file.read_bytes(), signature=signature)
        except (RegistryAuthError, RuntimeError, OSError, ValueError, binascii.Error) as exc:
            raise SigningFailed(f"{type(exc).__name__}: {exc}") from exc
        finally:
            shutil.rmtree(directory, ignore_errors=True)


# --- The HTTP server -------------------------------------------------------------------------


def _handler(broker: Broker) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "nhx-build-broker"
        # A client that goes quiet is dropped, and its thread freed: http.server closes the
        # connection when a read times out.
        timeout = _SOCKET_TIMEOUT_SECONDS

        def do_GET(self) -> None:  # the name http.server dispatches on
            if urlsplit(self.path).path == "/healthz":
                self._send(Reply(HTTPStatus.OK, {"status": "ok"}))
            else:
                self._send(_error(HTTPStatus.NOT_FOUND, "NOT_FOUND"))

        def do_POST(self) -> None:  # the name http.server dispatches on
            path = urlsplit(self.path).path
            if path not in ("/credentials", "/sign"):
                self._send(_error(HTTPStatus.NOT_FOUND, "NOT_FOUND"))
                return
            body = self._body()
            if isinstance(body, Reply):
                self._send(body)
                return
            authorization = self.headers.get("Authorization")
            try:
                if path == "/credentials":
                    reply = broker.credentials(authorization=authorization)
                else:
                    reply = broker.sign(authorization=authorization, body=body)
            except Exception:
                # Every failure the broker expects is a reply already. This is the rest: still an
                # answer, and still nothing issued.
                logger.exception("POST %s failed", sanitize_for_log(path))
                reply = _error(HTTPStatus.INTERNAL_SERVER_ERROR, "INTERNAL")
            self._send(reply)

        def _body(self) -> bytes | Reply:
            """The request body, or the reply refusing it, read before the request is authenticated.

            So it is bounded first: a length that is not a non-negative integer, or is more than a
            request to this broker needs, is refused unread.
            """
            declared = self.headers.get("Content-Length") or "0"
            if not (declared.isascii() and declared.isdigit()):
                return _error(HTTPStatus.BAD_REQUEST, "BAD_REQUEST", "Content-Length must be a non-negative integer")
            length = int(declared)
            if length > _MAX_BODY:
                return _error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "TOO_LARGE")
            return self.rfile.read(length)

        def _send(self, reply: Reply) -> None:
            payload = json.dumps(reply.body).encode()
            self.send_response(reply.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:  # `format`: the base class names it so
            # Decisions are logged, and audited, by the broker. The request line adds only noise.
            return

    return Handler


class BoundedServer(ThreadingHTTPServer):
    """A thread per request, as ``ThreadingHTTPServer`` gives, but at most ``max_requests`` of them.

    A connection beyond the bound is closed at once, unread. The broker is shared by every build,
    and unbounded threads would let any one caller that reaches it exhaust it.
    """

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], *, max_requests: int) -> None:
        super().__init__(address, handler)
        self._slots = threading.BoundedSemaphore(max_requests)

    def process_request(self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            logger.warning("refused a connection from %s: the broker is serving all it may at once", client_address)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def serve(builder: BuilderConfig, config: BrokerConfig) -> int:
    from kubernetes import config as k8s_config

    k8s_config.load_incluster_config()
    # httpx logs every request at INFO; the broker's own audit lines are the record that matters.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not builder.registry:
        logger.error("builder.registry is not configured")
        return 2
    registry = DistributionTokenClient(
        builder.registry,
        FileCredential(config.registry_username, Path(config.registry_password_file)),
        plain_http=builder.registry_plain_http,
        require_narrowing=builder.registry_narrows_scope,
    )
    broker = Broker(
        builder,
        config,
        backend=load_backend(builder),
        reviewer=KubernetesReviewer(),
        pods=KubernetesPods(),
        rows=BuilderRouteRows(
            config.platform_url,
            exchange=TokenExchange(config.token_exchange) if config.token_exchange else None,
            http=httpx.Client(timeout=config.request_timeout_seconds),
        ),
        registry=registry,
        signer=CosignSigner(builder, config, registry),
    )
    server = BoundedServer(("", config.port), _handler(broker), max_requests=_MAX_CONCURRENT_REQUESTS)
    logger.info("credential broker for %s on :%d", builder.registry, config.port)
    server.serve_forever()
    return 0


def main(argv: list[str]) -> int:
    """``nhx-build broker``: serve, with ``NHX_BROKER_CONFIG`` and the platform's ``builder:`` section."""
    if argv:
        print("usage: nhx-build broker", file=sys.stderr)
        return 2
    path = os.environ.get("NHX_BROKER_CONFIG")
    if not path:
        logger.error("NHX_BROKER_CONFIG is not set")
        return 2
    config = BrokerConfig.model_validate(yaml.safe_load(Path(path).read_text()))
    return serve(BuilderConfig.get(), config)
