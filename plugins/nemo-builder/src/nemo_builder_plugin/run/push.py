# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nmp-build push`` -- step 3. Trusted. Holds the registry credential and the signing key.

**It publishes bytes it did not produce.** That is the point of the step split, and it is also
exactly why this file is the most defensive one in the plugin: it is the only thing standing
between a sandbox's output and a credential the sandbox was never given.

The layout on the work volume was written by a pod that ran caller-authored ``RUN``. It is
**hostile input**, and it is treated as such before anything is uploaded:

- **Destinations come from the compiler, never from the layout.** Nothing read off the volume
  can influence where bytes go.
- **No symlinks are followed, and no path may escape the layout.** This step mounts the work
  volume *whole* -- it needs the whole job slice -- so a followed symlink genuinely does reach
  another job's directory.
- **Every blob is verified against its own digest.** A blob whose contents do not hash to its
  filename is a layout lying about what it contains.
- **The index must resolve to a manifest that is actually present.**

It runs no caller code, and it is the last step, so a failure here costs a publish rather than a
build.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from nemo_builder_plugin.run.context import read_step_config
from nemo_builder_plugin.steps import PushImage, PushStepConfig, SigningConfig

logger = logging.getLogger(__name__)

#: Read in chunks: a layer blob is routinely hundreds of megabytes and a build that OOMs the
#: trusted step because a caller shipped a large layer is a denial of service with extra steps.
_CHUNK = 1024 * 1024

#: The env var the compiler wires the registry credential into. The jobs launcher resolves the
#: secret in-pod, as the submitting principal -- the value never enters the job spec or etcd.
CREDENTIAL_ENVVAR = "NMP_REGISTRY_AUTH"


def _materialize_credential() -> str | None:
    """Write the injected credential where crane and cosign will look for it.

    Both read a Docker config, so the value has to land on a filesystem somewhere -- there is no
    "pass a credential on the command line" for either, and doing so would put it in the process
    table anyway. It goes to a 0600 file in a private temp directory rather than to the default
    `~/.docker/config.json`, so its lifetime is this process and its reach is this step.

    Accepts either a full dockerconfigjson or a bare `user:password`, because a deployment's
    Secrets entry is more likely to hold whichever its operator already had.
    """
    raw = os.environ.get(CREDENTIAL_ENVVAR)
    if not raw:
        logger.warning("%s is not set; pushing anonymously", CREDENTIAL_ENVVAR)
        return None

    raw = raw.strip()
    if raw.startswith("{"):
        config = raw
    else:
        import base64

        username, _, password = raw.partition(":")
        registry = os.environ.get("NMP_REGISTRY_HOST", "")
        auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        config = json.dumps({"auths": {registry: {"auth": auth}}})

    directory = tempfile.mkdtemp(prefix="nmp-docker-")
    path = Path(directory) / "config.json"
    path.write_text(config)
    path.chmod(0o600)
    os.environ["DOCKER_CONFIG"] = directory
    logger.info("registry credential materialized at %s", directory)
    return directory


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
    if not layout.is_dir():
        raise LayoutRejected(f"no layout at {layout}")

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

    index = json.loads(index_path.read_text())
    manifests = index.get("manifests") or []
    if len(manifests) != 1:
        raise LayoutRejected(f"expected exactly one manifest in index.json, found {len(manifests)}")

    digest = manifests[0].get("digest", "")
    if not digest.startswith("sha256:"):
        raise LayoutRejected(f"index.json names a non-sha256 digest: {digest!r}")
    if not (blobs_dir / digest.removeprefix("sha256:")).is_file():
        raise LayoutRejected(f"index.json names a manifest that is not in the layout: {digest}")

    logger.info("layout %s validated: %d blob(s), manifest %s", layout.name, len(files), digest)
    return digest


def _run(args: list[str]) -> str:
    logger.info("$ %s", " ".join(args))
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.stdout:
        logger.info("%s", result.stdout.strip())
    if result.returncode != 0:
        logger.error("%s", result.stderr.strip())
        raise RuntimeError(f"{args[0]} failed with exit {result.returncode}")
    return result.stdout.strip()


def _push_one(image: PushImage, signing: SigningConfig) -> None:
    layout = Path(image.layout)
    digest = validate_layout(layout)

    # Destinations come from the compiler. Nothing read off the volume reaches this list.
    for tag in image.tags:
        _run(["crane", "push", str(layout), tag])

    # Sign BY DIGEST, never by tag: a tag is mutable and signing one races anything that could
    # move it. The digest was derived from bytes validated above.
    repository = image.tags[0].rsplit(":", 1)[0]
    _run(
        [
            "cosign",
            "sign",
            f"--key={signing.key}",
            # Mandatory, not optional: cosign v2 uploads to the PUBLIC Rekor transparency log by
            # default, which would publish internal image names.
            "--tlog-upload=false",
            "--yes",
            f"{repository}@{digest}",
        ]
    )
    logger.info("published and signed %s@%s", repository, digest)


def main() -> int:
    config = PushStepConfig.model_validate(read_step_config())
    _materialize_credential()

    published = 0
    failures = 0
    for image in config.images:
        if not Path(image.layout).exists():
            # The build step attempts every spec and does not abort the set, so a missing layout
            # means THAT image failed -- not that this step has nothing to do. Skip it and
            # publish the rest; its row stays `pending` and the reconciler fails it.
            logger.warning("no layout for %s; skipping (its build failed)", image.image)
            failures += 1
            continue
        try:
            _push_one(image, config.signing)
            published += 1
        except (LayoutRejected, RuntimeError):
            logger.exception("refusing to publish %s", image.image)
            failures += 1

    logger.info("published %d image(s), %d failure(s)", published, failures)
    return 1 if failures else 0
