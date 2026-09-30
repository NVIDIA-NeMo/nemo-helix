# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build push`` -- step 3. Trusted. Holds no credential and no key of its own.

Everything it publishes with comes from the credential broker, for this job only
(``run/broker.py``):

- **a registry token** for this job's ``pending`` destinations -- asked for again before each
  image, since a registry's token lives minutes;
- **a signature.** The step names a row. The broker resolves that row's system tag itself, signs
  the digest it finds with a key no step can read, and returns what it made: the signed payload and
  its signature.

It proves which job it belongs to with its pod's token -- the workload token Jobs projects when
platform auth is on, its ServiceAccount token otherwise -- and that token opens nothing by itself:
``nhx-build-push`` holds no RBAC.

**Then it delivers the signature**, as the submitter, to the control plane, which verifies it and
makes the row ``ready``. The step is a courier. It can withhold the signature, and the row then fails;
it cannot forge any, because the signature is over a digest the broker read.

**It publishes bytes it did not produce.** That is the point of the step split, and it is also
exactly why this file is defensive: the layout on the work volume was written by a pod that ran
caller-authored ``RUN``. It is **hostile input**, and it is treated as such before anything is
uploaded:

- **Destinations come from the compiler, never from the layout.** Nothing read off the volume
  can influence where bytes go.
- **No symlinks are followed, and no path may escape the layout.** This step mounts the job's
  whole slice of the work volume, not just one image's output, so a followed symlink -- the
  layout's own directory included -- could hand it another image's layout or a context.
- **Every blob is verified against its own digest.** A blob whose contents do not hash to its
  filename is a layout lying about what it contains.
- **The index must resolve to a manifest that is actually present.**
- **What the broker signed must be what this step pushed.** It compares the digest the broker
  resolved with the one it validated here: a difference means the system tag moved between the
  push and the signature, and that signature is not delivered.

It runs no caller code, and it is the last step, so a failure here costs a publish rather than a
build.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import httpx
from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.completion import SignatureDelivery
from nemo_builder_plugin.run.context import job_identity, read_step_config, work_mount
from nemo_builder_plugin.run.tools import run_tool
from nemo_builder_plugin.steps import PushImage, PushStepConfig, WorkLayout
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: Read in chunks: a layer blob is routinely hundreds of megabytes and a build that OOMs the
#: trusted step because a caller shipped a large layer is a denial of service with extra steps.
_CHUNK = 1024 * 1024

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

#: Where the kubelet keeps this pod's ServiceAccount token, and keeps it fresh.
POD_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"

#: The username sent beside the pod token. The broker ignores it -- a username proves nothing --
#: but Basic needs one, and this one says what the password is if a request is ever logged.
POD_TOKEN_USERNAME = "nhx-build-push-pod-token"

#: Delivering a signature: this many attempts, backing off from one second. The route is idempotent
#: for the same signature, so a retry after a lost response is safe.
_DELIVERY_ATTEMPTS = 5


def step_token_path() -> Path:
    """The token that says which pod this is: the workload token when Jobs projects one, else the SA's."""
    return Path(os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR) or POD_TOKEN_PATH)


class BrokerRefused(Exception):
    """The broker would not hand this step a credential or a signature, or signed something else."""


@dataclass(frozen=True, slots=True)
class Signed:
    """What the broker signed for one row: the digest it resolved, and the signature over it."""

    digest: str
    delivery: SignatureDelivery


class Broker:
    """This step's two requests to the credential broker, each authenticated by the pod's token.

    The token is read fresh for each request: the kubelet rotates it, and a publish of many images
    can outlive one.
    """

    def __init__(self, url: str, *, token_path: Path, http: httpx.Client | None = None) -> None:
        self._url = url.rstrip("/")
        self._token_path = token_path
        # A signature uploads one small manifest; a minute is generous, and a stuck broker must not
        # hold the step.
        self._http = http or httpx.Client(timeout=60.0)

    def _post(self, path: str, body: Mapping[str, object] | None = None) -> dict[str, object]:
        token = self._token_path.read_text().strip()
        try:
            response = self._http.post(f"{self._url}{path}", json=body, auth=(POD_TOKEN_USERNAME, token))
        except httpx.HTTPError as exc:
            raise BrokerRefused(f"the broker could not be reached: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise BrokerRefused(f"the broker answered {path} with {response.status_code}: {response.text[:500]}")
        document = response.json()
        if not isinstance(document, dict):
            raise BrokerRefused(f"the broker answered {path} with something other than an object")
        return document

    def credentials(self) -> dict[str, object]:
        """A Docker config for this job's destinations: ``{"auths": {<registry>: {"registrytoken": ...}}}``."""
        document = self._post("/credentials")
        auths = document.get("auths")
        if not isinstance(auths, dict) or not auths:
            raise BrokerRefused("the broker returned no credential")
        repositories = document.get("repositories")
        logger.info(
            "the broker granted %s (expires in %ss; narrowed by the registry: %s)",
            sanitize_for_log(", ".join(str(r) for r in repositories) if isinstance(repositories, list) else "?"),
            sanitize_for_log(document.get("expires_in")),
            sanitize_for_log(document.get("narrowed")),
        )
        return {"auths": auths}

    def sign(self, image: str) -> Signed:
        """Ask the broker to sign the row ``image``. It names no digest: the broker resolves its own."""
        document = self._post("/sign", {"image": image})
        digest, signed = document.get("digest"), document.get("signed")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or not isinstance(signed, dict):
            raise BrokerRefused(f"the broker's signature for {image} is not one")
        return Signed(digest=digest, delivery=SignatureDelivery.model_validate(signed))


class _Broker(Protocol):
    """What publishing needs from the broker. :class:`Broker` is it; a test double can be."""

    def credentials(self) -> dict[str, object]: ...

    def sign(self, image: str) -> Signed: ...


class _Courier(Protocol):
    """What publishing needs from the control plane. :class:`Courier` is it."""

    def deliver(self, image: str, delivery: SignatureDelivery) -> None: ...

    def status(self, image: str) -> str: ...


def write_docker_config(directory: Path, config: Mapping[str, object]) -> None:
    """Where crane looks for credentials: ``$DOCKER_CONFIG/config.json``, readable by this user only."""
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)


class LayoutRejected(Exception):
    """The OCI layout failed validation and will not be published."""


def _verify_no_escape(root: Path) -> list[Path]:
    """Every regular file under ``root``, with symlinks and escapes refused.

    ``Path.rglob`` does not follow directory symlinks, but it will *list* them, and a later
    ``read_bytes`` would follow one. So symlinks are rejected outright rather than resolved --
    there is no legitimate symlink in an OCI layout, which makes "reject" both safe and simple.
    """
    root = root.resolve(strict=True)
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise LayoutRejected(f"layout contains a symlink: {path.relative_to(root)}")
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            raise LayoutRejected(f"layout path escapes its directory: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            # A FIFO named as a blob would pass a regular-file check by being skipped, and then
            # block crane in `open()` forever.
            raise LayoutRejected(f"layout contains something that is not a file: {path.relative_to(root)}")
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def validate_layout(layout: Path) -> str:
    """Validate an OCI layout and return the digest its index points at.

    Raises :class:`LayoutRejected` on anything inconsistent. The returned digest is derived from
    bytes this function verified, not from anything the layout asserted about itself.
    """
    if layout.is_symlink():
        # Checked before resolving, below: resolving would follow it. Only the layout's own
        # directory is checked here; its parents legitimately include symlinks (`/var/run`).
        raise LayoutRejected(f"layout {layout} is a symlink")
    if not layout.is_dir():
        raise LayoutRejected(f"no layout at {layout}")

    # Resolve ONCE, here, and use the resolved path for everything below.
    #
    # `_verify_no_escape` resolves internally and returns resolved paths. Comparing those against
    # an UNRESOLVED `layout` is wrong the moment any parent component is a symlink -- and one
    # always is: the work volume mounts under `/var/run`, which is a symlink to `/run` on every
    # mainstream base image. The observed failure was `relative_to` raising "is not in the
    # subpath of", and before that every file compared unequal to the markers and was rejected as
    # "unexpected file in layout". A correct layout was refused for being correct.
    layout = layout.resolve(strict=True)
    files = _verify_no_escape(layout)

    marker = layout / "oci-layout"
    index_path = layout / "index.json"
    for required in (marker, index_path):
        if not required.is_file():
            raise LayoutRejected(f"layout is missing {required.name}")

    blobs_dir = layout / "blobs" / "sha256"
    for path in files:
        if blobs_dir not in path.parents:
            # Only the two marker files live outside blobs/. Anything else is unexpected, and
            # "unexpected" in a directory written by untrusted code is a refusal, not a warning.
            if path not in (marker, index_path):
                raise LayoutRejected(f"unexpected file in layout: {path.relative_to(layout)}")
            continue
        actual = _sha256_file(path)
        if actual != path.name:
            raise LayoutRejected(f"blob {path.name} hashes to {actual}: the layout misdescribes its contents")

    try:
        index = json.loads(index_path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayoutRejected("index.json is not JSON") from exc
    manifests = index.get("manifests") if isinstance(index, dict) else None
    if not isinstance(manifests, list) or len(manifests) != 1 or not isinstance(manifests[0], dict):
        raise LayoutRejected("expected exactly one manifest in index.json")

    digest = manifests[0].get("digest")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise LayoutRejected(f"index.json names something other than a sha256 digest: {digest!r}")
    if not (blobs_dir / digest.removeprefix("sha256:")).is_file():
        raise LayoutRejected(f"index.json names a manifest that is not in the layout: {digest}")

    logger.info(
        "layout %s validated: %d blob(s), manifest %s",
        sanitize_for_log(layout.name),
        len(files),
        sanitize_for_log(digest),
    )
    return digest


class Courier:
    """Delivers signatures to the control plane, as the submitter, retrying what may be transient.

    Only a transport failure, a 429 or a 5xx is retried: the route is idempotent for the same
    signature, so a retry after a lost response completes rather than conflicts. A 4xx is the
    signature refused, and delivering it again would be refused again.
    """

    def __init__(self, client: BuilderClient, *, workspace: str, sleep: Callable[[float], None] = time.sleep) -> None:
        self._client = client
        self._workspace = workspace
        self._sleep = sleep

    def deliver(self, image: str, delivery: SignatureDelivery) -> None:
        for attempt in range(_DELIVERY_ATTEMPTS):
            try:
                row = self._client.deliver_signature(workspace=self._workspace, name=image, body=delivery).data()
            except NemoTransportError as exc:
                failure: Exception = exc
            except NemoHTTPError as exc:
                if exc.status_code != 429 and exc.status_code < 500:
                    raise
                failure = exc
            else:
                logger.info("delivered the signature for %s: %s", sanitize_for_log(image), sanitize_for_log(row.status))
                return
            if attempt + 1 < _DELIVERY_ATTEMPTS:
                logger.warning(
                    "delivering the signature for %s failed (%s); retrying",
                    sanitize_for_log(image),
                    sanitize_for_log(failure),
                )
                self._sleep(2.0**attempt)
        raise failure

    def status(self, image: str) -> str:
        """The row's status, read as the submitter."""
        return self._client.get_container_image(workspace=self._workspace, name=image).data().status


def _push_one(
    image: PushImage, layout: Path, *, broker: _Broker, courier: _Courier, docker_config: Path, plain_http: bool
) -> None:
    digest = validate_layout(layout)

    # A fresh token for every image: a registry's token lives minutes, and a set's publish can
    # take longer than that.
    write_docker_config(docker_config, broker.credentials())
    # Destinations come from the compiler. Nothing read off the volume reaches this list. The system
    # tag goes last: it is what the broker resolves and signs.
    for ref in image.refs:
        run_tool(["crane", "push", str(layout), ref, *(["--insecure"] if plain_http else [])])

    signed = broker.sign(image.image)
    if signed.digest != digest:
        raise BrokerRefused(
            f"the broker signed {signed.digest} for {image.image}, but this step pushed {digest}: the system tag "
            "moved between the push and the signature, so the signature is not delivered"
        )
    courier.deliver(image.image, signed.delivery)
    logger.info(
        "published, signed and delivered %s@%s",
        sanitize_for_log(image.system_ref.rsplit(":", 1)[0]),
        sanitize_for_log(digest),
    )


def _publish_all(config: PushStepConfig, *, broker: _Broker, courier: _Courier, docker_config: Path) -> int:
    work = WorkLayout(PurePosixPath(work_mount()))
    published = 0
    failures = 0
    for image in config.images:
        try:
            status = courier.status(image.image)
        except (NemoHTTPError, NemoTransportError):
            logger.exception("cannot read %s; not publishing it", sanitize_for_log(image.image))
            failures += 1
            continue
        if status != "pending":
            # A rerun of this step: the image already completed -- or already failed -- in an
            # earlier attempt, and a settled row has nothing left to publish. The broker would
            # grant nothing for it anyway.
            logger.info("%s is already %s; nothing to publish", sanitize_for_log(image.image), sanitize_for_log(status))
            if status == "ready":
                published += 1
            else:
                failures += 1
            continue
        layout = Path(work.output(image.image))
        if not layout.exists():
            # The build step attempts every spec and does not abort the set, so a missing layout
            # means THAT image failed -- not that this step has nothing to do. Skip it and
            # publish the rest; its row stays `pending` and the failure sweep fails it when the
            # job ends.
            logger.warning("no layout for %s; skipping (its build failed)", sanitize_for_log(image.image))
            failures += 1
            continue
        try:
            _push_one(
                image, layout, broker=broker, courier=courier, docker_config=docker_config, plain_http=config.plain_http
            )
            published += 1
        except Exception:
            # Anything, not just the failures this step anticipates: the layout is hostile input,
            # and an error nobody predicted in one image must not cost the rest of the set.
            logger.exception("refusing to publish %s", sanitize_for_log(image.image))
            failures += 1

    logger.info("published %d image(s), %d failure(s)", published, failures)
    return 1 if failures else 0


def main() -> int:
    config = PushStepConfig.model_validate(read_step_config())
    workspace, _ = job_identity()
    broker = Broker(config.broker, token_path=step_token_path())
    # As the SUBMITTER, the way `fetch` reads Files: through the job's delegation, never as a
    # platform identity of this step's own.
    courier = Courier(client_from_platform(get_task_nemo_client("builder"), BuilderClient), workspace=workspace)
    directory = Path(tempfile.mkdtemp(prefix="nhx-docker-"))
    os.environ["DOCKER_CONFIG"] = str(directory)
    try:
        return _publish_all(config, broker=broker, courier=courier, docker_config=directory)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
