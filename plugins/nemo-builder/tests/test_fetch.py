# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`fetch` against a fake Files service that reports paths the way the real one does.

The detail that matters: a listing narrowed to ``path=tests`` still reports each entry relative
to the fileset ROOT -- ``tests/Dockerfile``, not ``Dockerfile``
(``services/core/files/.../backends/local.py``, ``relative_to(resolved_path)``). A fetch that
missed this nested every subtree inside itself, and the sandbox found no Dockerfile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from nemo_builder_plugin.run.fetch import (
    ImportRejected,
    apply_runtime_layer,
    clear_earlier_attempts,
    fetch_imports,
    fetch_source,
    pull_copy,
    write_derived_context,
)
from nemo_builder_plugin.run.supervise import _volume_mounts
from nemo_builder_plugin.steps import (
    ContextSource,
    FetchDockerfile,
    FetchImport,
    FetchStepConfig,
    RuntimeLayerSpec,
    SandboxGroup,
    SandboxImage,
    WorkLayout,
)
from nemo_helix_plugin.files.client import FilesClient

JOB_SLICE = "jobs/default/abc"


@dataclass
class _Entry:
    path: str


@dataclass
class _Listing:
    data: list[_Entry]


class FakeFiles:
    """Lists and serves one fileset. Entry paths are root-relative, as in the real service."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def list_files(self, *, workspace: str, name: str, query_params: dict | None = None):
        prefix = (query_params or {}).get("path")
        entries = [_Entry(p) for p in self.files if prefix is None or p.startswith(f"{prefix.rstrip('/')}/")]

        class _Response:
            def data(self) -> _Listing:
                return _Listing(entries)

        return _Response()

    def download_file(self, *, workspace: str, name: str, path: str):
        content = self.files[path]

        class _Body:
            def read(self) -> bytes:
                return content

        return _Body()


def _client(files: dict[str, bytes]) -> FilesClient:
    # A structural stand-in for the two calls `fetch` makes; the generated client is not a Protocol.
    return cast(FilesClient, FakeFiles(files))


FILESET = {"tests/Dockerfile": b"FROM tests", "env/Dockerfile": b"FROM env", "README": b"hi"}


def _sandbox_context(tmp_path: Path, source: ContextSource) -> Path:
    """The directory the sandbox would see, found the way `supervise` finds it: from the mount."""
    group = SandboxGroup(
        source=source, images=[SandboxImage(image="demo-1-0", platform="linux/amd64", dockerfile="Dockerfile")]
    )
    context_mount = _volume_mounts(group, JOB_SLICE)[0]
    return tmp_path / PurePosixPath(context_mount.sub_path).relative_to(JOB_SLICE)


class TestWhatTheSandboxSees:
    def test_a_subtree_lands_where_the_sandbox_mounts_it(self, tmp_path: Path) -> None:
        """The regression: this used to produce `context/.../fs-a/tests/tests/Dockerfile`."""
        source = ContextSource(fileset="default/fs-a", context_path="tests")
        fetch_source(_client(FILESET), WorkLayout(PurePosixPath(tmp_path)), source, workspace="default")

        context = _sandbox_context(tmp_path, source)
        assert (context / "Dockerfile").read_bytes() == b"FROM tests"
        assert not (context / "tests").exists()
        # Only the subtree was downloaded; a sibling subtree is not on the volume at all.
        assert not (tmp_path / "context/default/fs-a/env").exists()

    def test_a_whole_fileset_download_serves_its_subtrees(self, tmp_path: Path) -> None:
        """Why the compiler lets a whole-fileset request absorb subtree requests."""
        fetch_source(
            _client(FILESET),
            WorkLayout(PurePosixPath(tmp_path)),
            ContextSource(fileset="default/fs-a"),
            workspace="default",
        )

        for subtree, content in (("tests", b"FROM tests"), ("env", b"FROM env")):
            context = _sandbox_context(tmp_path, ContextSource(fileset="default/fs-a", context_path=subtree))
            assert (context / "Dockerfile").read_bytes() == content

    def test_the_context_hash_is_written_outside_every_context(self, tmp_path: Path) -> None:
        subtree = ContextSource(fileset="default/fs-a", context_path="tests")
        layout = WorkLayout(PurePosixPath(tmp_path))
        fetch_source(_client(FILESET), layout, ContextSource(fileset="default/fs-a"), workspace="default")
        digest = fetch_source(_client(FILESET), layout, subtree, workspace="default")
        assert Path(layout.context_hash_file(subtree)).read_text() == digest
        # Not in the subtree's context, and not in the whole fileset's either.
        assert not any(p.name.endswith(".nhx-context-hash") for p in (tmp_path / "context").rglob("*"))


class TestEachRunStartsClean:
    def test_an_earlier_runs_contexts_and_layouts_are_removed(self, tmp_path: Path) -> None:
        """A job deleted and submitted again under the same name gets the same slice. Its old
        files would build, and its old layouts would be published when a new build fails."""
        layout = WorkLayout(PurePosixPath(tmp_path))
        stale_context = Path(layout.context(ContextSource(fileset="default/fs-a"))) / "removed-since"
        stale_layout = Path(layout.output("demo-1-0")) / "index.json"
        stale_import = Path(layout.import_context("demo-1-1")) / "Dockerfile"
        for path in (stale_context, stale_layout, stale_import):
            path.parent.mkdir(parents=True)
            path.write_text("old")

        clear_earlier_attempts(layout)
        assert not stale_context.exists() and not stale_layout.exists() and not stale_import.exists()


class TestHostileListings:
    def test_an_entry_path_that_escapes_the_fileset_is_refused(self, tmp_path: Path) -> None:
        """Entry paths come from a listing a tenant controls."""
        with pytest.raises(ValueError, match="escapes"):
            fetch_source(
                _client({"../other-job/x": b"!"}),
                WorkLayout(PurePosixPath(tmp_path)),
                ContextSource(fileset="default/fs-a"),
                workspace="default",
            )


LAYER = RuntimeLayerSpec(name="harbor-sandbox", version="3", dockerfile="USER root\nRUN true\nUSER 1000\n")
DIGEST = "sha256:" + "2" * 64
REF = f"docker.io/harborframework/terminal-bench@{DIGEST}"


class TestTheRuntimeLayer:
    def test_it_is_appended_after_the_context_is_hashed(self, tmp_path: Path) -> None:
        """The recorded hash describes what the caller supplied, not what we added."""
        layout = WorkLayout(PurePosixPath(tmp_path))
        source = ContextSource(fileset="default/fs-a", context_path="tests")
        digest = fetch_source(_client(FILESET), layout, source, workspace="default")

        apply_runtime_layer(layout, [FetchDockerfile(source=source, dockerfile="Dockerfile")], LAYER)

        dockerfile = (Path(layout.context(source)) / "Dockerfile").read_text()
        assert dockerfile.startswith("FROM tests\n")
        assert "runtime layer harbor-sandbox@3" in dockerfile
        assert dockerfile.rstrip().endswith("USER 1000")
        assert Path(layout.context_hash_file(source)).read_text() == digest

    def test_one_file_named_two_ways_gets_it_once(self, tmp_path: Path) -> None:
        """A whole fileset's `tests/Dockerfile` and the `tests` subtree's `Dockerfile` are one file."""
        layout = WorkLayout(PurePosixPath(tmp_path))
        fetch_source(_client(FILESET), layout, ContextSource(fileset="default/fs-a"), workspace="default")

        changed = apply_runtime_layer(
            layout,
            [
                FetchDockerfile(source=ContextSource(fileset="default/fs-a"), dockerfile="tests/Dockerfile"),
                FetchDockerfile(
                    source=ContextSource(fileset="default/fs-a", context_path="tests"), dockerfile="Dockerfile"
                ),
            ],
            LAYER,
        )

        assert changed == 1
        assert (tmp_path / "context/default/fs-a/tests/Dockerfile").read_text().count("runtime layer") == 1

    def test_a_missing_dockerfile_is_left_for_kaniko_to_report(self, tmp_path: Path) -> None:
        layout = WorkLayout(PurePosixPath(tmp_path))
        fetch_source(_client(FILESET), layout, ContextSource(fileset="default/fs-a"), workspace="default")
        changed = apply_runtime_layer(
            layout, [FetchDockerfile(source=ContextSource(fileset="default/fs-a"), dockerfile="nope/Dockerfile")], LAYER
        )
        assert changed == 0
        assert not (tmp_path / "context/default/fs-a/nope").exists()

    @pytest.mark.parametrize(
        "dockerfile",
        [
            "FROM alpine\nRUN echo hi \\\n",
            "FROM alpine\nRUN echo hi \\\n\n# trailing comment\n",
            "# escape=`\nFROM alpine\n",
        ],
        ids=["dangling-continuation", "continuation-then-comment", "escape-directive"],
    )
    def test_a_dockerfile_the_layer_cannot_follow_is_removed_not_changed(self, tmp_path: Path, dockerfile: str) -> None:
        """Appended, the layer would be swallowed into the caller's `RUN` or read with another
        escape character -- and the row would still name it. Its build fails instead."""
        layout = WorkLayout(PurePosixPath(tmp_path))
        source = ContextSource(fileset="default/fs-a", context_path="tests")
        fetch_source(_client({"tests/Dockerfile": dockerfile.encode()}), layout, source, workspace="default")

        changed = apply_runtime_layer(layout, [FetchDockerfile(source=source, dockerfile="Dockerfile")], LAYER)
        assert changed == 0
        assert not (Path(layout.context(source)) / "Dockerfile").exists()

    def test_a_backslash_escape_directive_is_the_default_and_fine(self, tmp_path: Path) -> None:
        layout = WorkLayout(PurePosixPath(tmp_path))
        source = ContextSource(fileset="default/fs-a", context_path="tests")
        body = b"# escape=\\\nFROM alpine\nRUN echo a \\\n    b\n"
        fetch_source(_client({"tests/Dockerfile": body}), layout, source, workspace="default")
        assert apply_runtime_layer(layout, [FetchDockerfile(source=source, dockerfile="Dockerfile")], LAYER) == 1


class TestImports:
    def test_a_derived_import_is_one_dockerfile_from_the_resolved_manifest(self, tmp_path: Path) -> None:
        layout = WorkLayout(PurePosixPath(tmp_path))
        entry = FetchImport(image="demo-1-0", ref=REF, platform="linux/amd64", mode="derive")

        dockerfile = write_derived_context(layout, entry, LAYER)

        assert dockerfile.parent == Path(layout.import_context("demo-1-0"))
        assert dockerfile.read_text().startswith(f"FROM {REF}\n")
        assert [p.name for p in dockerfile.parent.iterdir()] == ["Dockerfile"]

    @staticmethod
    def _crane(holds: str):
        """Stands in for `crane pull`: writes a one-manifest layout holding `holds`."""
        calls: list[list[str]] = []

        def run(args: list[str]) -> str:
            calls.append(args)
            destination = Path(args[-1])
            destination.mkdir(parents=True)
            (destination / "oci-layout").write_text('{"imageLayoutVersion": "1.0.0"}')
            (destination / "index.json").write_text(json.dumps({"manifests": [{"digest": holds}]}))
            return ""

        return run, calls

    def test_a_copy_lands_where_push_reads_a_build(self, tmp_path: Path) -> None:
        layout = WorkLayout(PurePosixPath(tmp_path))
        run, calls = self._crane(DIGEST)

        assert (
            pull_copy(layout, FetchImport(image="demo-1-0", ref=REF, platform="linux/amd64", mode="copy"), run=run)
            == DIGEST
        )

        assert calls == [
            ["crane", "pull", "--format=oci", "--platform=linux/amd64", REF, str(layout.output("demo-1-0"))]
        ]
        assert (Path(layout.output("demo-1-0")) / "index.json").exists()

    def test_a_layout_holding_anything_else_is_refused_and_removed(self, tmp_path: Path) -> None:
        """Left in the slot, `push` would publish and sign it."""
        layout = WorkLayout(PurePosixPath(tmp_path))
        run, _ = self._crane("sha256:" + "9" * 64)
        with pytest.raises(ImportRejected, match="layout holds"):
            pull_copy(layout, FetchImport(image="demo-1-0", ref=REF, platform="linux/amd64", mode="copy"), run=run)
        assert not Path(layout.output("demo-1-0")).exists()

    def test_a_failed_pull_leaves_nothing_behind(self, tmp_path: Path) -> None:
        layout = WorkLayout(PurePosixPath(tmp_path))

        def half_written(args: list[str]) -> str:
            Path(args[-1]).mkdir(parents=True)
            (Path(args[-1]) / "oci-layout").write_text("{}")
            raise RuntimeError("crane failed with exit 1")

        with pytest.raises(RuntimeError):
            pull_copy(
                layout, FetchImport(image="demo-1-0", ref=REF, platform="linux/amd64", mode="copy"), run=half_written
            )
        assert not Path(layout.output("demo-1-0")).exists()

    def test_an_earlier_attempts_layout_is_replaced_not_appended_to(self, tmp_path: Path) -> None:
        """A second manifest in one layout is something `push` refuses."""
        layout = WorkLayout(PurePosixPath(tmp_path))
        stale = Path(layout.output("demo-1-0"))
        stale.mkdir(parents=True)
        (stale / "leftover").write_text("x")
        run, _ = self._crane(DIGEST)

        pull_copy(layout, FetchImport(image="demo-1-0", ref=REF, platform="linux/amd64", mode="copy"), run=run)

        assert not (stale / "leftover").exists()

    def test_one_failed_import_does_not_stop_the_rest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(*args: object, **kwargs: object) -> str:
            raise RuntimeError("crane failed with exit 1")

        monkeypatch.setattr("nemo_builder_plugin.run.fetch.pull_copy", broken)
        config = FetchStepConfig(
            imports=[
                FetchImport(image="demo-1-0", ref=REF, platform="linux/amd64", mode="copy"),
                FetchImport(image="demo-1-1", ref=REF, platform="linux/amd64", mode="derive"),
            ],
            runtime_layer=LAYER,
        )
        layout = WorkLayout(PurePosixPath(tmp_path))

        assert fetch_imports(layout, config) == 1
        assert (Path(layout.import_context("demo-1-1")) / "Dockerfile").exists()
