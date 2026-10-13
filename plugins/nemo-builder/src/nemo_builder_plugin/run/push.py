# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build push``: the last step. It publishes each image, signs it, and has the builder complete it.

Each OCI layout was written by caller code, so it is validated as hostile before anything is uploaded. The
workspace's registry credential and signing key reach this step as environment variables, from the platform's
Secrets service.
"""

from __future__ import annotations

import base64
import errno
import json
import logging
import os
import re
import shutil
import stat
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
from nemo_builder_plugin.run.tools import run_tool
from nemo_builder_plugin.run.utils import job_identity, read_step_config, work_mount
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

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

#: Larger than any index or manifest kaniko writes; read whole, so capped.
_MAX_JSON_BYTES = 4 * 1024 * 1024

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
    """The workspace's registry credential and signing key, each ``None`` if the deployment turned it off."""

    login: tuple[str, str] | None
    key: SigningKey | None

    @classmethod
    def take_from(cls, environ: MutableMapping[str, str], *, log_in: bool, sign: bool) -> Credentials:
        """Read the ones this step expects from ``environ``, and remove all of them, so no tool it runs inherits them."""
        values = {name: environ.pop(name, "").strip() for name in (REGISTRY_USERNAME_ENV, REGISTRY_PASSWORD_ENV)}
        pem = environ.pop(SIGNING_KEY_ENV, "")
        expected = [*(values.items() if log_in else ()), *([(SIGNING_KEY_ENV, pem)] if sign else ())]
        missing = [name for name, value in expected if not value]
        if missing:
            raise Refused(f"{', '.join(missing)} not set: the workspace's builder secrets did not reach this step")
        key = None
        if sign:
            try:
                # `nemo secrets create` trims a value, a PEM's final newline included.
                key = SigningKey(pem.strip().encode() + b"\n")
            except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
                # The error names what is wrong with the key, never the key.
                raise Refused(f"{SIGNING_KEY_ENV} is not an unencrypted EC or RSA private key in PEM") from exc
        login = (values[REGISTRY_USERNAME_ENV], values[REGISTRY_PASSWORD_ENV]) if log_in else None
        return cls(login, key)

    def docker_config(self, registry: str) -> dict[str, object]:
        """What crane reads: the credential for ``registry``, which crane exchanges for a token if the registry asks.

        Without one, crane pushes anonymously.
        """
        if self.login is None:
            return {"auths": {}}
        auth = base64.b64encode(":".join(self.login).encode()).decode()
        return {"auths": {registry: {"auth": auth}}}


def write_docker_config(directory: Path, config: Mapping[str, object]) -> None:
    """Where crane looks for credentials: ``$DOCKER_CONFIG/config.json``."""
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)


class LayoutRejected(Exception):
    """The OCI layout failed validation and will not be published."""


def _require(layout: Path, path: Path, *, directory: bool) -> None:
    """Refuse ``path`` unless it's a real directory or a regular file, as ``directory`` says.

    Not a symlink, which a later read would follow, nor a FIFO or a device, which would block or never end.
    """
    name = path.relative_to(layout)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise LayoutRejected(f"layout is missing {name}") from exc
    if stat.S_ISLNK(mode):
        raise LayoutRejected(f"{name} is a symlink")
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise LayoutRejected(f"{name} is not a {'directory' if directory else 'regular file'}")


def _read_json(layout: Path, path: Path) -> object:
    """``path`` parsed, opened without following a symlink, and only if it's a small regular file."""
    name = path.relative_to(layout)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError as exc:
        raise LayoutRejected(f"layout is missing {name}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise LayoutRejected(f"{name} is a symlink") from exc
        raise
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise LayoutRejected(f"{name} is not a regular file")
        data = handle.read(_MAX_JSON_BYTES + 1)
    if len(data) > _MAX_JSON_BYTES:
        raise LayoutRejected(f"{name} is larger than {_MAX_JSON_BYTES} bytes")
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayoutRejected(f"{name} is not JSON") from exc


def _digest(value: object, *, named_by: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise LayoutRejected(f"{named_by} names something other than a sha256 digest: {value!r}")
    return value


def validate_layout(layout: Path) -> str:
    """Check the files crane will read, and return the manifest digest the layout's index names in its one entry.

    Only those files: crane reads nothing else, and hashing them is the registry's, which checks each blob against
    its digest as it's uploaded. crane itself refuses a blob that's a symlink or not a regular file; it isn't
    relied on, since it doesn't check the directories above a blob, and reads ``index.json`` through a symlink
    and blocks on a FIFO. Raises :class:`LayoutRejected`.
    """
    if not layout.exists() and not layout.is_symlink():
        raise LayoutRejected(f"no layout at {layout}")
    if layout.is_symlink() or not layout.is_dir():
        raise LayoutRejected(f"layout {layout} is a symlink or not a directory")

    index = _read_json(layout, layout / "index.json")
    manifests = index.get("manifests") if isinstance(index, dict) else None
    if not isinstance(manifests, list) or len(manifests) != 1 or not isinstance(manifests[0], dict):
        raise LayoutRejected("expected exactly one manifest in index.json")
    digest = _digest(manifests[0].get("digest"), named_by="index.json")

    blobs = layout / "blobs" / "sha256"
    _require(layout, blobs.parent, directory=True)
    _require(layout, blobs, directory=True)
    manifest = _read_json(layout, blobs / digest.removeprefix("sha256:"))
    config = manifest.get("config") if isinstance(manifest, dict) else None
    layers = manifest.get("layers") if isinstance(manifest, dict) else None
    if not isinstance(config, dict) or not isinstance(layers, list):
        raise LayoutRejected(f"{digest}, which index.json names, is not an image manifest")
    for descriptor in (config, *layers):
        if not isinstance(descriptor, dict):
            raise LayoutRejected(f"{digest}, which index.json names, is not an image manifest")
        blob = _digest(descriptor.get("digest"), named_by="the manifest")
        _require(layout, blobs / blob.removeprefix("sha256:"), directory=False)

    logger.info(
        "layout %s checked: manifest %s, %d layer(s)",
        sanitize_for_log(layout.name),
        sanitize_for_log(digest),
        len(layers),
    )
    return digest


class Builder:
    """This step's calls to the builder's routes, acting for the submitter, retrying what may be transient.

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
        return self._call(
            f"reading {image}",
            lambda: self._client.get_container_image(workspace=self._workspace, name=image).data(),
        )


#: Signs a row's image at a digest, and stores the signature in the registry.
Sign = Callable[[ContainerImage, str], None]


def signer(config: PushStepConfig, credentials: Credentials, http: httpx.Client) -> Sign | None:
    """``None`` if the deployment turned signing off."""
    key = credentials.key
    if key is None:
        return None

    def sign(row: ContainerImage, digest: str) -> None:
        authorization = None
        if credentials.login is not None:
            authorization = login(
                config.registry, *credentials.login, row.repository, http=http, plain_http=config.plain_http
            )
        registry = Registry(config.registry, authorization, http=http, plain_http=config.plain_http)
        sign_image(registry, row, digest, key)

    return sign


def _push_one(
    row: ContainerImage, image: PushImage, layout: Path, *, builder: _Builder, sign: Sign | None, plain_http: bool
) -> None:
    digest = validate_layout(layout)

    # Destinations come from the compiler: nothing read off the volume reaches this list.
    for ref in image.refs:
        pushed = run_tool(["crane", "push", str(layout), ref, *(["--insecure"] if plain_http else [])])
        # crane reports the hash of the manifest it read and pushed: go on only if that's the digest index.json
        # names, which a manifest that misdescribes itself, or that changed after it was checked, fails.
        if (pushed_digest := pushed.rpartition("@")[2]) != digest:
            raise Refused(f"crane pushed {pushed_digest or 'nothing it reported'}, not the checked {digest}")
    if sign is not None:
        # Every annotation comes from the row, as the builder wrote it at submit.
        sign(row, digest)

    done = builder.complete(image.image, digest)
    if done.status != "ready" or done.digest != digest:
        raise Refused(f"the builder completed {image.image} as {done.status} at {done.digest}, not ready at {digest}")
    logger.info(
        "published, %s and completed %s@%s",
        "signed" if sign is not None else "not signed",
        sanitize_for_log(image.system_ref.rsplit(":", 1)[0]),
        sanitize_for_log(digest),
    )


def _publish_all(config: PushStepConfig, *, builder: _Builder, sign: Sign | None) -> int:
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
        credentials = Credentials.take_from(os.environ, log_in=config.log_in, sign=config.sign)
    except Refused as exc:
        logger.error("%s", exc)
        return 1
    if credentials.key is not None:
        logger.info("signing with %s", credentials.key.fingerprint)
    else:
        logger.info("not signing: the deployment turned signing off")
    workspace, _ = job_identity()
    # As `service:builder`, acting for the submitter through the job's delegation.
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
