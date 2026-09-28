# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build fetch`` -- step 1. Trusted. Holds a Files client and nothing else.

**This step is the whole control against a caller naming another tenant's fileset.** It resolves
every source **as the submitting principal**, so a request for a fileset the submitter cannot read
fails here, at the API, rather than succeeding and mounting the stolen data faithfully.

``subPath`` does not help with that attack and it is worth being explicit about why: if the fetch
succeeds, mounting "the right subPath" mounts exactly the data that should never have been
downloaded. The mount is a partition between *this job's* groups; it is not an authorization
boundary against other tenants. This is.

It holds no registry credential and no pod RBAC, and it runs no caller-authored code -- it copies
bytes onto a volume.

**It also brings imports in.** A copy is pulled by digest, anonymously, straight into the output
slot a sandbox would have written, where ``push`` publishes it exactly as it publishes a build --
and treats it as the hostile input it is, since a publisher produced those bytes and nobody here
inspected them. A derived import gets a synthesized one-line context instead, and is built in the
sandbox like anything else. Copies do not yet go through the pull-through mirror: that is `M2-1`,
the same gap `FROM` has.

**And it applies the runtime layer**, appending it to every Dockerfile the set builds, after the
context is hashed -- so the hash still describes what the caller supplied.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from nemo_builder_plugin.run.context import (
    context_hash,
    job_identity,
    read_step_config,
    run_tool,
    split_fileset_ref,
    work_mount,
)
from nemo_builder_plugin.steps import (
    ContextSource,
    FetchDockerfile,
    FetchImport,
    FetchStepConfig,
    RuntimeLayerSpec,
    WorkLayout,
)
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import ListFilesQueryParams

logger = logging.getLogger(__name__)


def _safe_destination(root: Path, relative_path: str) -> Path:
    """Resolve a fileset-supplied path under ``root``, refusing anything that escapes.

    The path comes from a fileset listing, which a tenant controls. A `..` or absolute component
    would otherwise write outside this job's slice of a volume shared by every build.
    """
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"fileset path {relative_path!r} escapes the context directory")
    return candidate


def _download_fileset(
    client: FilesClient,
    *,
    workspace: str,
    name: str,
    context_path: str | None,
    destination: Path,
) -> int:
    query: ListFilesQueryParams | None = {"path": context_path} if context_path else None
    # `.data()` is a method on the response wrapper, not an attribute -- iterating the wrapper
    # directly iterates a bound method and silently yields nothing useful.
    listing = client.list_files(workspace=workspace, name=name, query_params=query).data()

    count = 0
    for entry in listing.data:
        target = _safe_destination(destination, entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # `.read()`, not `bytes(...)`: download_file returns a streaming response object.
        response = client.download_file(workspace=workspace, name=name, path=entry.path)
        target.write_bytes(response.read())
        count += 1
    return count


def fetch_source(client: FilesClient, layout: WorkLayout, source: ContextSource, *, workspace: str) -> str:
    """Download one source to ``layout.context(source)``, and return its context hash.

    Files reports entry paths relative to the fileset root even when the listing is narrowed to
    ``context_path``, so entries are written under the *fileset's* directory -- and a subtree
    then lands at ``layout.context(source)`` because that is where the layout puts it, inside its
    fileset. Writing them under the context directory instead would nest the subtree inside
    itself (``tests/tests/Dockerfile``).
    """
    source_workspace, name = split_fileset_ref(source.fileset, workspace)
    fileset_dir = Path(layout.fileset(source.fileset))
    context_dir = Path(layout.context(source))
    hash_file = Path(layout.context_hash_file(source))
    fileset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "fetching fileset %s/%s%s -> %s",
        source_workspace,
        name,
        f" ({source.context_path})" if source.context_path else "",
        context_dir,
    )
    count = _download_fileset(
        client,
        workspace=source_workspace,
        name=name,
        context_path=source.context_path,
        destination=fileset_dir,
    )
    context_dir.mkdir(parents=True, exist_ok=True)

    digest = context_hash(context_dir)
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(digest)
    logger.info("fetched %d file(s), context hash %s", count, digest)
    return digest


def clear_earlier_attempts(layout: WorkLayout) -> None:
    """Remove whatever an earlier run of this job left in its slice of the work volume.

    The slice is keyed by the job's name, and a job deleted and submitted again under the same
    name gets the same slice. A context would then build from files since removed from its
    fileset, and a layout from the earlier run would be published if this run's build of it
    fails. This step runs first, so it is the one that starts the slice clean.
    """
    for directory in (layout.root / "context", layout.root / "hashes", layout.imports, layout.outputs):
        path = Path(directory)
        if path.exists():
            logger.info("removing %s left by an earlier run of this job", path)
            shutil.rmtree(path)


class ImportRejected(Exception):
    """A pulled image is not the one the submit path resolved."""


class LayerRefused(Exception):
    """A Dockerfile the runtime layer cannot be appended to without being misread."""


_DIRECTIVE = re.compile(r"^#\s*([A-Za-z]+)\s*=\s*(.*?)\s*$")


def _check_layer_can_follow(text: str) -> None:
    """Refuse a Dockerfile that would swallow, or re-parse, what is appended after it.

    Two ways the caller's file changes how the layer reads, and both would leave a row recording
    a layer that did not run as written:

    - **A dangling line continuation.** Dockerfile parsing skips blank and comment lines inside a
      continuation, so ``RUN make \\`` at the end of the file joins the layer's first instruction
      onto the caller's ``RUN`` as arguments.
    - **An ``escape`` parser directive** naming anything but a backslash. The layer is written
      with backslash continuations, which would then be read as literal characters.
    """
    for line in text.splitlines():
        directive = _DIRECTIVE.match(line)
        if not directive:
            break  # parser directives are only valid before anything else
        if directive.group(1).lower() == "escape" and directive.group(2) != "\\":
            raise LayerRefused(f"it declares escape={directive.group(2)!r}")
    instructions = [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if instructions and instructions[-1].endswith("\\"):
        raise LayerRefused("its last line continues onto whatever follows it")


def append_runtime_layer(dockerfile: Path, layer: RuntimeLayerSpec) -> None:
    """Append the layer after the Dockerfile's own instructions, labelled with its version.

    Appended to the last stage, which is the one the image is made of. The layer cannot start a
    stage of its own -- config refuses a `FROM` in it -- so what it produces is still this image.
    Raises :class:`LayerRefused` where appending would change what either side means.
    """
    text = dockerfile.read_text()
    _check_layer_can_follow(text)
    if text and not text.endswith("\n"):
        text += "\n"
    dockerfile.write_text(
        f"{text}\n# ---- runtime layer {layer.label}, appended by nhx-build fetch ----\n{layer.dockerfile.rstrip()}\n"
    )


def apply_runtime_layer(layout: WorkLayout, dockerfiles: list[FetchDockerfile], layer: RuntimeLayerSpec) -> int:
    """Append the layer to each fetched Dockerfile once, and return how many were changed.

    Deduplicated by the RESOLVED file, not by the config entry: a whole fileset's
    ``tests/Dockerfile`` and the ``tests`` subtree's ``Dockerfile`` are one file, and appending twice
    would run the layer twice.

    **A Dockerfile the layer cannot follow is removed**, so its build fails rather than producing
    an image whose row names a layer it does not contain. Its images' rows fail; the rest of the
    set builds.
    """
    done: set[Path] = set()
    for entry in dockerfiles:
        context = Path(layout.context(entry.source))
        path = _safe_destination(context, entry.dockerfile)
        if path in done:
            continue
        if not path.is_file():
            # Not fatal here: kaniko fails that image with its own clearer message, and the rest
            # of the set still builds.
            logger.warning("no Dockerfile at %s; the runtime layer was not applied to it", path)
            continue
        try:
            append_runtime_layer(path, layer)
        except LayerRefused as exc:
            logger.error("cannot append runtime layer %s to %s: %s; its build will fail", layer.label, path, exc)
            path.unlink()
            continue
        done.add(path)
    return len(done)


def write_derived_context(layout: WorkLayout, entry: FetchImport, layer: RuntimeLayerSpec | None) -> Path:
    """A derived import's whole context: one Dockerfile, ``FROM`` the upstream, then the layer.

    ``entry.ref`` is the platform manifest the submit path resolved, so the base is exactly the
    image the row names as its upstream.
    """
    context = Path(layout.import_context(entry.image))
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    dockerfile = context / "Dockerfile"
    dockerfile.write_text(f"FROM {entry.ref}\n")
    if layer is not None:
        append_runtime_layer(dockerfile, layer)
    return dockerfile


def pull_copy(layout: WorkLayout, entry: FetchImport, *, run: Callable[[list[str]], object] = run_tool) -> str:
    """Pull a copy into its output slot as an OCI layout, and return the manifest digest.

    **Verified here as well as downstream.** ``push`` checks the layout's digest against the
    one the submit path resolved, and the reconciler checks the published one; this check is the
    cheap one, and the one whose error names the import rather than a layout path.

    **A pull that fails, or is refused, leaves the slot empty.** ``push`` publishes whatever
    layout it finds there, so a refused image left in place would still be signed.
    """
    destination = Path(layout.output(entry.image))
    if destination.exists():
        # This job's own slot, from an earlier attempt. A layout pulled into an existing one gains
        # a second manifest, which `push` would then refuse.
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        run(["crane", "pull", "--format=oci", f"--platform={entry.platform}", entry.ref, str(destination)])
        expected = entry.ref.rsplit("@", 1)[1]
        manifests = json.loads((destination / "index.json").read_text()).get("manifests") or []
        if len(manifests) != 1 or manifests[0].get("digest") != expected:
            found = [m.get("digest") for m in manifests]
            raise ImportRejected(f"pulled {entry.ref} but the layout holds {found}")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    logger.info("copied %s -> %s", entry.ref, destination)
    return expected


def fetch_imports(layout: WorkLayout, config: FetchStepConfig) -> int:
    """Bring in every import, and return how many failed.

    One failed import does not abort the set, for the same reason one failed build does not: its
    row goes `failed` on its own, and the rest are published.
    """
    failures = 0
    for entry in config.imports:
        try:
            if entry.mode == "copy":
                pull_copy(layout, entry)
            else:
                write_derived_context(layout, entry, config.runtime_layer)
        except (RuntimeError, OSError, ValueError, ImportRejected, LayerRefused):
            logger.exception("import %s (%s) failed", entry.image, entry.ref)
            failures += 1
    return failures


def main() -> int:
    config = FetchStepConfig.model_validate(read_step_config())
    workspace, _ = job_identity()
    layout = WorkLayout(PurePosixPath(work_mount()))
    clear_earlier_attempts(layout)

    if config.sources:
        # As the SUBMITTER. `get_task_nemo_client` forwards the submitting principal -- via
        # on-behalf-of today, via workload-identity token exchange where that is enabled. It is
        # deliberately not a service identity: a service identity would read any fileset in the
        # deployment, which is the confused deputy this step exists to remove.
        client = client_from_platform(get_task_nemo_client("builder"), FilesClient)
        for source in config.sources:
            fetch_source(client, layout, source, workspace=workspace)
        if config.runtime_layer is not None:
            changed = apply_runtime_layer(layout, config.dockerfiles, config.runtime_layer)
            logger.info("runtime layer %s appended to %d Dockerfile(s)", config.runtime_layer.label, changed)

    failures = fetch_imports(layout, config)
    # Non-zero only when nothing is left to build or publish: a non-zero exit stops every later
    # step, for the whole set.
    if failures and failures == len(config.imports) and not config.sources:
        return 1
    return 0


# --- `source_digest` does not currently reach the image, and that is a real gap. ---
#
# RFC 001 wants the context hash carried as an image LABEL, so the reconciler can recover it from
# the config blob. That needs the hash to reach kaniko, and kaniko is launched by `supervise` --
# which deliberately mounts no volume at all, so it cannot read what this step just wrote.
#
# Three ways out, none free:
#   1. Mount ONLY the hash file into `supervise`. Narrow, but it weakens "the control plane for
#      the sandbox reads nothing" from a structural property to a reviewed exception.
#   2. Have the compiler pass it -- impossible, the context does not exist at compile time.
#   3. Let the SANDBOX compute and label it. Rejected: the sandbox runs caller-authored `RUN`,
#      so a hash it reports describes whatever it wants it to describe.
#
# For now the hash is computed and written under `hashes/`, where it is available to anything
# that mounts the volume, and `ContainerImage.source_digest` stays None. The field is advisory
# and never identity, so an absent value costs provenance detail rather than correctness.
