# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build push``: the last step. It publishes each image, signs it, and has the builder complete it.

Each OCI layout was written by caller code, so it is validated as hostile before anything is uploaded. The
workspace's registry credential and signing key reach this step as environment variables, from the platform's
Secrets service.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, TypeVar

import httpx
from cryptography.exceptions import UnsupportedAlgorithm
from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.completion import CompleteRequest
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.registry import REGISTRY_TIMEOUT_SECONDS, Registry, registry_client
from nemo_builder_plugin.registry_auth import login
from nemo_builder_plugin.run.context import job_identity, read_step_config, work_mount
from nemo_builder_plugin.run.tools import run_tool
from nemo_builder_plugin.signing import SigningKey, sign_image
from nemo_builder_plugin.steps import (
    REGISTRY_PASSWORD_ENV,
    REGISTRY_USERNAME_ENV,
    SIGNING_KEY_ENV,
    PushImage,
    PushStepConfig,
    WorkLayout,
)
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: Blobs are hashed in chunks: a caller's layer can be larger than this step's memory.
_CHUNK = 1024 * 1024

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

_ATTEMPTS = 5

_T = TypeVar("_T")


class Refused(Exception):
    """This step can't publish, or the builder completed an image as something other than what it pushed."""


class _Builder(Protocol):
    """What publishing needs from the builder."""

    def row(self, image: str) -> ContainerImage: ...

    def complete(self, image: str, digest: str) -> ContainerImage: ...


@dataclass(frozen=True)
class Credentials:
    """The workspace's registry credential and signing key."""

    username: str
    password: str
    key: SigningKey

    @classmethod
    def take_from(cls, environ: MutableMapping[str, str]) -> Credentials:
        """Read them from ``environ``, and remove them, so no tool this step runs inherits them."""
        values = {name: environ.pop(name, "").strip() for name in (REGISTRY_USERNAME_ENV, REGISTRY_PASSWORD_ENV)}
        pem = environ.pop(SIGNING_KEY_ENV, "")
        missing = [name for name, value in (*values.items(), (SIGNING_KEY_ENV, pem)) if not value]
        if missing:
            raise Refused(f"{', '.join(missing)} not set: the workspace's builder secrets did not reach this step")
        try:
            # `nemo secrets create` trims a value, a PEM's final newline included.
            key = SigningKey(pem.strip().encode() + b"\n")
        except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
            # The error names what is wrong with the key, never the key.
            raise Refused(f"{SIGNING_KEY_ENV} is not an unencrypted EC or RSA private key in PEM") from exc
        return cls(values[REGISTRY_USERNAME_ENV], values[REGISTRY_PASSWORD_ENV], key)

    def docker_config(self, registry: str) -> dict[str, object]:
        """What crane reads: the credential for ``registry``, which crane exchanges for a token if the registry asks."""
        auth = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        return {"auths": {registry: {"auth": auth}}}


def write_docker_config(directory: Path, config: Mapping[str, object]) -> None:
    """Where crane looks for credentials: ``$DOCKER_CONFIG/config.json``."""
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)


class LayoutRejected(Exception):
    """The OCI layout failed validation and will not be published."""


def _verify_no_escape(root: Path) -> list[Path]:
    """Every regular file under ``root``, refusing any symlink: a later read would follow it."""
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
            # A FIFO named as a blob, if skipped here, would block crane's `open()` forever.
            raise LayoutRejected(f"layout contains something that is not a file: {path.relative_to(root)}")
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def validate_layout(layout: Path) -> str:
    """Return the digest of the layout's one manifest, or raise :class:`LayoutRejected`."""
    if layout.is_symlink():
        # Checked before resolving, which would follow it.
        raise LayoutRejected(f"layout {layout} is a symlink")
    if not layout.is_dir():
        raise LayoutRejected(f"no layout at {layout}")

    # The work volume is mounted under `/var/run`, usually a symlink to `/run`, and
    # `_verify_no_escape` returns resolved paths: everything below uses the resolved layout.
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


class Builder:
    """This step's calls to the builder's routes, as the submitter, retrying what may be transient.

    Both are safe to repeat: completing an image already ``ready`` at the same digest returns it.
    """

    def __init__(self, client: BuilderClient, *, workspace: str, sleep: Callable[[float], None] = time.sleep) -> None:
        self._client = client
        self._workspace = workspace
        self._sleep = sleep

    def _call(self, what: str, call: Callable[[], _T]) -> _T:
        for attempt in range(_ATTEMPTS):
            try:
                return call()
            except NemoTransportError as exc:
                failure: Exception = exc
            except NemoHTTPError as exc:
                if exc.status_code != 429 and exc.status_code < 500:
                    raise
                failure = exc
            if attempt + 1 < _ATTEMPTS:
                logger.warning("%s failed (%s); retrying", what, sanitize_for_log(failure))
                self._sleep(2.0**attempt)
        raise failure

    def complete(self, image: str, digest: str) -> ContainerImage:
        return self._call(
            f"completing {image}",
            lambda: self._client.complete_container_image(
                workspace=self._workspace, name=image, body=CompleteRequest(digest=digest)
            ).data(),
        )

    def row(self, image: str) -> ContainerImage:
        return self._client.get_container_image(workspace=self._workspace, name=image).data()


#: Signs a row's image at a digest, and stores the signature in the registry.
Sign = Callable[[ContainerImage, str], None]


def signer(config: PushStepConfig, credentials: Credentials, http: httpx.Client) -> Sign:
    def sign(row: ContainerImage, digest: str) -> None:
        authorization = login(
            config.registry,
            credentials.username,
            credentials.password,
            row.repository,
            http=http,
            plain_http=config.plain_http,
        )
        registry = Registry(config.registry, authorization, http=http, plain_http=config.plain_http)
        sign_image(registry, row, digest, credentials.key)

    return sign


def _push_one(
    row: ContainerImage, image: PushImage, layout: Path, *, builder: _Builder, sign: Sign, plain_http: bool
) -> None:
    digest = validate_layout(layout)

    # Destinations come from the compiler: nothing read off the volume reaches this list.
    for ref in image.refs:
        run_tool(["crane", "push", str(layout), ref, *(["--insecure"] if plain_http else [])])
    # Every annotation comes from the row, as the builder wrote it at submit.
    sign(row, digest)

    done = builder.complete(image.image, digest)
    if done.status != "ready" or done.digest != digest:
        raise Refused(f"the builder completed {image.image} as {done.status} at {done.digest}, not ready at {digest}")
    logger.info(
        "published, signed and completed %s@%s",
        sanitize_for_log(image.system_ref.rsplit(":", 1)[0]),
        sanitize_for_log(digest),
    )


def _publish_all(config: PushStepConfig, *, builder: _Builder, sign: Sign) -> int:
    work = WorkLayout(PurePosixPath(work_mount()))
    published = 0
    failures = 0
    for image in config.images:
        try:
            row = builder.row(image.image)
        except (NemoHTTPError, NemoTransportError):
            logger.exception("cannot read %s; not publishing it", sanitize_for_log(image.image))
            failures += 1
            continue
        if row.status != "pending":
            logger.info(
                "%s is already %s; nothing to publish", sanitize_for_log(image.image), sanitize_for_log(row.status)
            )
            if row.status == "ready":
                published += 1
            else:
                failures += 1
            continue
        layout = Path(work.output(image.image))
        if not layout.exists():
            logger.warning("no layout for %s; skipping (its build failed)", sanitize_for_log(image.image))
            failures += 1
            continue
        try:
            _push_one(row, image, layout, builder=builder, sign=sign, plain_http=config.plain_http)
            published += 1
        except Exception:
            # Any error, anticipated or not: one image must not cost the rest of the set.
            logger.exception("refusing to publish %s", sanitize_for_log(image.image))
            failures += 1

    logger.info("published %d image(s), %d failure(s)", published, failures)
    return 1 if failures else 0


def main() -> int:
    config = PushStepConfig.model_validate(read_step_config())
    try:
        credentials = Credentials.take_from(os.environ)
    except Refused as exc:
        logger.error("%s", exc)
        return 1
    logger.info("signing with %s", credentials.key.fingerprint)
    workspace, _ = job_identity()
    # As the submitter, through the job's delegation: this step has no platform identity of its own.
    builder = Builder(client_from_platform(get_task_nemo_client("builder"), BuilderClient), workspace=workspace)
    # Not on the work volume, which the sandbox writes to.
    directory = Path(tempfile.mkdtemp(prefix="nhx-docker-"))
    os.environ["DOCKER_CONFIG"] = str(directory)
    try:
        write_docker_config(directory, credentials.docker_config(config.registry))
        with registry_client(REGISTRY_TIMEOUT_SECONDS) as http:
            return _publish_all(config, builder=builder, sign=signer(config, credentials, http))
    finally:
        shutil.rmtree(directory, ignore_errors=True)
