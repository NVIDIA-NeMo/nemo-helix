# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the pinned archives and revisions shared by the Harbor skill examples."""

import hashlib
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

from nemo_evals.api.schemas import Revision
from nemo_evals.harbor.archive_io import download_verified
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.skills import PinnedSkillRef
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.schema import Page
from nemo_helix_plugin.skill_archive import DEFAULT_LIMITS, extract_skill_archive
from nhx_evals_sdk.values.skill_evidence import SkillContentBundle
from nhx_evals_sdk.values.skills import SkillSnapshot

_SHA256 = re.compile(r"[0-9a-f]{64}")


def split_pin(ref: str) -> tuple[str, str, str]:
    qualified, _, digest = ref.partition("#")
    workspace, _, name = qualified.partition("/")
    if not workspace or not name or "/" in name or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"Expected a fully qualified digest-pinned ref: {ref}")
    return workspace, name, digest


def read_skill_package(
    client: NemoClient, skill_ref: PinnedSkillRef
) -> tuple[SkillSnapshot, dict[str, str], SkillContentBundle]:
    workspace, name, digest = split_pin(skill_ref.root)
    skill = AgentsClient.from_client(client).get_skill_revision(workspace=workspace, name=name, revision=digest).data()
    if skill.content_hash != digest or skill.archive.skill_name != name:
        raise ValueError("Agents returned a different skill revision or package name")
    archive = skill.archive
    with tempfile.TemporaryDirectory(prefix="helix-skill-") as directory:
        root = Path(directory)
        archive_path = root / "archive.tar.gz"
        with archive_path.open("wb") as output:
            download_verified(
                FilesClient.from_client(client),
                archive.archive_ref,
                output,
                limit=DEFAULT_LIMITS.max_archive_bytes,
                expected_digest=archive.files_hash,
            )
        extracted = extract_skill_archive(archive_path, root, expected_files_hash=archive.files_hash)
        if extracted.name != archive.skill_name:
            raise ValueError("The digest-verified archive has an unexpected package directory")
        inventory, text = {}, {}
        for path in sorted(extracted.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(extracted).as_posix()
            data = path.read_bytes()
            inventory[relative] = hashlib.sha256(data).hexdigest()
            try:
                text[relative] = data.decode("utf-8")
            except UnicodeDecodeError:
                # Package metrics report missing text if a file in their scope cannot be decoded.
                continue
        snapshot = SkillSnapshot(
            directory_name=archive.skill_name,
            digest=archive.files_hash,
            document=text.get("SKILL.md"),
            files=list(inventory),
        )
        if snapshot.document is None:
            raise ValueError("The verified skill package must contain a UTF-8 SKILL.md")
    return snapshot, inventory, SkillContentBundle(artifact_digest=snapshot.digest, files=text)


def pin_revision(list_revisions: Callable[..., Page[Revision]], workspace: str, name: str, ordinal: int) -> str:
    page = 1
    while True:
        response = list_revisions(name, workspace=workspace, page=page)
        for revision in response.data:
            if revision.revision == ordinal and _SHA256.fullmatch(revision.content_hash):
                return f"{workspace}/{name}#{revision.content_hash}"
        if response.pagination is not None and page < response.pagination.total_pages:
            page += 1
        else:
            raise ValueError(f"No digest for published revision {ordinal} of {workspace}/{name}")
