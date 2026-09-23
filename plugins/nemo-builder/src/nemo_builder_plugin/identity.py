# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image identity: parsing, normalization, and the system tag.

Not one service in the tree stores a registry digest today. Identity is needed by the build
compiler, the reconciler, the submit route and eventually ``nemo-deployments`` -- so it lives in
one module with one set of rules rather than as a private helper in whichever of those was
written first.

Lifted from ``scaled_evals.api.build.task_image_identity``, which already had the hard parts
right, and reshaped to RFC 001's vocabulary. The registry *policy* half of that module -- the
allowlist evaluator -- is deliberately **not** here yet; see the note at the bottom.

The two rules worth stating out loud:

**Normalization comes first, and callers cannot skip it.** ``alpine``, ``alpine:latest``,
``library/alpine:latest``, ``docker.io/library/alpine:latest`` and
``index.docker.io/library/alpine:latest`` are one image. Anything that compares references --
an allowlist, a cache key, a "have I already built this" check -- must compare normalized forms
or it is comparing spellings. :func:`normalize_reference` is the front door for anything that
arrives from a caller.

**Two digests, not one.** ``digest`` names whatever document the reference resolved to. When
that document is an index, ``manifest_digest`` names the per-platform manifest inside it. They
are equal when the reference resolves to a plain manifest, which is the usual case under this
execution mode -- but a build that emits attestations produces an index, and then they differ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The three schema fields that hold a digest all share this. Lowercase hex only: a digest is a
# byte string rendered one way, and accepting `SHA256:` or uppercase hex would make two spellings
# of one value.
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_REGISTRY = re.compile(r"(?:localhost|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[0-9]{1,5})?")
# The OCI repository path grammar. Note this is NOT the platform's NAME_PATTERN -- a repository
# path admits `/`, `.` and `__`, and forbids things NAME_PATTERN allows.
_REPOSITORY_COMPONENT = re.compile(r"[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*")
# An OCI tag. Deliberately stricter than the platform's NAME_PATTERN, which admits `@` and `+`
# -- both forbidden in a tag. See `compose_system_tag`.
_TAG = re.compile(r"[\w][\w.-]{0,127}")

_DOCKER_HUB = "docker.io"


class ImageIdentityError(ValueError):
    """A reference failed syntax or normalization."""


@dataclass(frozen=True, slots=True)
class ImageReference:
    """A parsed reference. Exactly one of ``tag`` / ``digest`` is set."""

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
        """The ``image_ref`` projection: ``<registry>/<repository>@<digest>``.

        This is what a consumer pins. It is computed, never stored -- storing it would make it
        a second place the digest lives, and two places disagree eventually.
        """
        return f"{self.repository_ref}@{normalize_digest(digest)}"


def normalize_digest(value: str) -> str:
    digest = value.strip().lower()
    if not _DIGEST.fullmatch(digest):
        raise ImageIdentityError("digest must be sha256:<64 lowercase hex>")
    return digest


def validate_repository(repository: str) -> str:
    """A repository path on the OCI grammar: ``/``-separated components, each one legal.

    Refuses an empty component, a leading or trailing ``/``, and ``..`` -- which is what keeps a
    caller's path inside whatever prefix the compiler joins it under.
    """
    if not repository or any(not _REPOSITORY_COMPONENT.fullmatch(part) for part in repository.split("/")):
        raise ImageIdentityError(f"invalid repository path: {repository!r}")
    return repository


def validate_tag(tag: str) -> str:
    if not _TAG.fullmatch(tag):
        raise ImageIdentityError(f"invalid tag: {tag!r}")
    return tag


def parse_reference(image_ref: str) -> ImageReference:
    """Parse a reference that already names its registry explicitly.

    Rejects shorthand. Use :func:`normalize_reference` for anything a caller typed.
    """
    value = image_ref.strip()
    if not value or any(char.isspace() for char in value) or "://" in value:
        raise ImageIdentityError("image reference must be an OCI reference without a URL scheme")

    name = value
    digest: str | None = None
    if "@" in value:
        if value.count("@") != 1:
            raise ImageIdentityError("image reference contains more than one digest separator")
        name, raw_digest = value.rsplit("@", 1)
        digest = normalize_digest(raw_digest)

    if "/" not in name:
        raise ImageIdentityError("image reference must include an explicit registry hostname")
    registry, repository_with_tag = name.split("/", 1)
    registry = registry.lower()
    # A registry hostname is distinguishable from a repository component only by containing a
    # dot or a colon -- or by being exactly `localhost`. That is the whole disambiguation rule,
    # and it is why `alpine/foo` is a Docker Hub repository and not a registry called `alpine`.
    if not _REGISTRY.fullmatch(registry) or not (registry == "localhost" or "." in registry or ":" in registry):
        raise ImageIdentityError(f"invalid registry hostname: {registry!r}")

    parts = repository_with_tag.split("/")
    tag: str | None = None
    last = parts[-1]
    if ":" in last:
        last, tag = last.rsplit(":", 1)
        parts[-1] = last
        validate_tag(tag)
    repository = validate_repository("/".join(parts))
    if digest and tag:
        raise ImageIdentityError("reference must use a tag or a digest, not tag-plus-digest form")
    if not digest and not tag:
        raise ImageIdentityError("reference must include an explicit tag or digest")
    return ImageReference(registry, repository, tag=tag, digest=digest)


def normalize_reference(image_ref: str) -> ImageReference:
    """Expand Docker shorthand, then parse. The front door for caller-supplied references.

    ``alpine`` -> ``docker.io/library/alpine:latest``. Three expansions, applied in order:
    an absent registry becomes ``docker.io``; a single-component repository on Docker Hub gains
    the implicit ``library/``; an absent tag becomes ``latest``.

    Whether an unqualified reference SHOULD be expanded rather than rejected is an open decision
    (`M1-9`): expanding is friendlier and requires ``docker.io`` on any allowlist, rejecting is
    stricter and pushes the choice to the caller. This implements expansion, which is Docker's
    own behaviour and therefore the least surprising default -- but the decision is not ours and
    the function is the one place it would change.
    """
    value = image_ref.strip()
    if not value or any(char.isspace() for char in value) or "://" in value:
        raise ImageIdentityError("image reference must be an OCI reference")

    name = value.split("@", 1)[0]
    first = name.split("/", 1)[0]
    if "/" not in name or ("." not in first and ":" not in first and first != "localhost"):
        value = f"{_DOCKER_HUB}/{value}"
        name = value.split("@", 1)[0]
    repository = name.split("/", 1)[1]
    if "/" not in repository:
        value = value.replace(f"{_DOCKER_HUB}/", f"{_DOCKER_HUB}/library/", 1)
    if "@" not in value and ":" not in value.rsplit("/", 1)[-1]:
        value = f"{value}:latest"
    return parse_reference(value)


def compose_system_tag(workspace: str, build_set: str, revision: int, index: int) -> str:
    """``<workspace>--<build_set>-<revision>-<index>`` -- the tag the reconciler resolves.

    The build pushes this alongside whatever the caller asked for, and the reconciler looks it
    up. It is generated once at submit and handed to both the ``ContainerImage`` row and the
    compiler, because composing it twice is a drift bug whose only symptom is resolving a tag
    nothing ever pushed.

    **One tag per IMAGE, not per set -- that is what ``index`` is for.** An earlier version was
    ``<workspace>--<build_set>-<revision>``, shared by every spec in the set. Two specs publishing
    to one repository -- ``team/app:staging`` and ``team/app:prod`` from different Dockerfiles --
    then both pushed ``team/app:<system-tag>``, the second push replaced the first, and both rows
    resolved the same digest. One of them recorded the identity of an image it did not describe,
    which is the one thing this system exists to get right. The index is the spec's position in
    ``build_specs``, the same number that makes ``ContainerImage.name`` unique, so the tag is
    unique wherever the row is.

    **The separator is ``--`` and that is load-bearing.** The platform's ``NAME_PATTERN`` admits
    ``-`` inside both a workspace and a set name, so with a single ``-`` the pair
    (``scaled-evals``, ``freight``) and the pair (``scaled``, ``evals-freight``) compose to the
    same string. ``NAME_PATTERN`` forbids ``--`` outright and forbids a trailing ``-``, so
    exactly one ``--`` appears here, and revision and index are the last two ``-``-separated
    fields and both numeric, so the split stays recoverable.

    **Validated here, not assumed.** ``NAME_PATTERN`` admits ``@`` and ``+``; an OCI tag forbids
    both. A name that is legal as a platform entity can be illegal as a tag, so the check belongs
    at the point of composition rather than at the point of push, twenty minutes later.
    """
    if revision < 1:
        raise ImageIdentityError("revision must be >= 1")
    if index < 0:
        raise ImageIdentityError("index must be >= 0")
    if "--" in workspace or "--" in build_set:
        raise ImageIdentityError(
            f"workspace and build_set must not contain '--' (got {workspace!r}, {build_set!r}); "
            "it is the separator that makes the system tag splittable"
        )
    tag = f"{workspace}--{build_set}-{revision}-{index}"
    if not _TAG.fullmatch(tag):
        raise ImageIdentityError(
            f"composed system tag is not a legal OCI tag: {tag!r}. A workspace or build set name "
            "that is legal for the platform is not automatically legal as a tag."
        )
    return tag


def split_system_tag(tag: str) -> tuple[str, str, int, int]:
    """Inverse of :func:`compose_system_tag`. Exists to prove the composition is recoverable."""
    if tag.count("--") != 1:
        raise ImageIdentityError(f"not a system tag: {tag!r}")
    workspace, rest = tag.split("--", 1)
    rest, _, index = rest.rpartition("-")
    set_name, _, revision = rest.rpartition("-")
    if not set_name or not revision.isdigit() or not index.isdigit():
        raise ImageIdentityError(f"not a system tag: {tag!r}")
    return workspace, set_name, int(revision), int(index)


# --- Registry policy is NOT here, and push destinations no longer need it. ---
#
# RFC 001 pairs identity with a registry allowlist, and makes normalization and allowlisting ONE
# feature on purpose: a raw string comparison against a list containing `docker.io` matches none
# of `alpine`, `alpine:latest` or `library/alpine:latest`, so a caller who can choose the
# spelling chooses the verdict. The evaluator belongs in this module, next to
# `normalize_reference`, so that an allowlist check cannot be written against an unnormalized
# string.
#
# A caller cannot name a push destination at all: every image goes to the deployment's one
# registry, under a path the compiler composes. What still wants an allowlist is the two places
# a caller-named registry remains -- a base image's `FROM`, which the pull-through mirror bounds
# (`M2-1`, not built), and registering an image this system did not build
# (`POST /container-images`, `M1-10`, not built). Its open semantics (`M1-9`) are open only for
# those.
