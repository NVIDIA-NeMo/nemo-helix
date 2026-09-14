# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.taskset_sources import (
    TasksetSourceMaterialization,
    digest_harbor_tree,
)
from pydantic import ValidationError

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def _materialization_payload(root: Path) -> dict[str, object]:
    return {
        "source_uri": "harbor://tasksets/example@latest",
        "materialized_root": root,
        "revision_digest": _DIGEST_A,
        "member_digests": {"task-one": _DIGEST_B},
        "materialization_digest": _DIGEST_C,
    }


def test_materialization_validates_and_round_trips_all_fields(tmp_path: Path) -> None:
    receipt = TasksetSourceMaterialization.model_validate(_materialization_payload(tmp_path))

    assert TasksetSourceMaterialization.model_validate_json(receipt.model_dump_json()) == receipt
    assert receipt.materialized_root == tmp_path


@pytest.mark.parametrize(
    "field",
    ["source_uri", "materialized_root", "revision_digest", "member_digests", "materialization_digest"],
)
def test_materialization_requires_every_field(tmp_path: Path, field: str) -> None:
    payload = _materialization_payload(tmp_path)
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        TasksetSourceMaterialization.model_validate(payload)


@pytest.mark.parametrize(
    "source_uri",
    ["", "tasksets/example", "/tasksets/example", "harbor://tasksets/example with space"],
)
def test_materialization_rejects_non_absolute_source_uri(tmp_path: Path, source_uri: str) -> None:
    payload = _materialization_payload(tmp_path)
    payload["source_uri"] = source_uri

    with pytest.raises(ValidationError, match="absolute URI"):
        TasksetSourceMaterialization.model_validate(payload)


def test_materialization_accepts_none_revision_and_empty_members(tmp_path: Path) -> None:
    payload = _materialization_payload(tmp_path)
    payload["revision_digest"] = None
    payload["member_digests"] = {}

    receipt = TasksetSourceMaterialization.model_validate(payload)

    assert receipt.revision_digest is None
    assert receipt.member_digests == {}


def test_materialization_rejects_relative_materialized_root() -> None:
    payload = _materialization_payload(Path("relative/root"))

    with pytest.raises(ValidationError, match="absolute path"):
        TasksetSourceMaterialization.model_validate(payload)


def test_materialization_rejects_empty_member_id(tmp_path: Path) -> None:
    payload = _materialization_payload(tmp_path)
    payload["member_digests"] = {"": _DIGEST_B}

    with pytest.raises(ValidationError, match="member ID"):
        TasksetSourceMaterialization.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision_digest", "A" * 64),
        ("revision_digest", "a" * 63),
        ("materialization_digest", "not-a-digest"),
        ("member_digests", {"task-one": "f" * 65}),
    ],
)
def test_materialization_rejects_non_sha256_digest_shapes(tmp_path: Path, field: str, value: object) -> None:
    payload = _materialization_payload(tmp_path)
    payload[field] = value

    with pytest.raises(ValidationError):
        TasksetSourceMaterialization.model_validate(payload)


def test_materialization_forbids_extra_fields_and_field_reassignment(tmp_path: Path) -> None:
    payload = _materialization_payload(tmp_path)
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        TasksetSourceMaterialization.model_validate(payload)

    receipt = TasksetSourceMaterialization.model_validate(_materialization_payload(tmp_path))
    with pytest.raises(ValidationError, match="frozen"):
        receipt.source_uri = "harbor://tasksets/other"  # type: ignore[misc]


def test_materialization_copies_member_digest_mapping(tmp_path: Path) -> None:
    member_digests = {"task-one": _DIGEST_B}
    payload = _materialization_payload(tmp_path)
    payload["member_digests"] = member_digests

    receipt = TasksetSourceMaterialization.model_validate(payload)
    member_digests["task-two"] = _DIGEST_C

    assert receipt.member_digests == {"task-one": _DIGEST_B}


def test_digest_is_stable_across_root_location_and_mtimes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    (first / "nested" / "empty").mkdir(parents=True)
    (first / ".hidden").write_bytes(b"hidden")
    (first / "nested" / "task.toml").write_bytes(b"[task]\nname='example'\n")
    (first / "task-link").symlink_to("nested/task.toml")
    second = tmp_path / "second"
    shutil.copytree(first, second, symlinks=True)
    os.utime(second / ".hidden", (1, 1))
    os.utime(second / "nested", (2, 2))

    assert digest_harbor_tree(first) == digest_harbor_tree(second)


def test_digest_is_stable_across_creation_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for name in ["z.txt", "a.txt", "middle.txt"]:
        (first / name).write_bytes(name.encode())
    for name in ["middle.txt", "z.txt", "a.txt"]:
        (second / name).write_bytes(name.encode())

    assert digest_harbor_tree(first) == digest_harbor_tree(second)


def test_digest_includes_empty_directories_and_dotfiles(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    baseline = digest_harbor_tree(root)

    (root / ".git").mkdir()
    with_empty_dot_directory = digest_harbor_tree(root)
    (root / ".git" / "config").write_bytes(b"config")

    assert with_empty_dot_directory != baseline
    assert digest_harbor_tree(root) != with_empty_dot_directory


def test_digest_includes_full_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "first").mkdir(parents=True)
    (root / "second").mkdir()
    (root / "first" / "task.toml").write_bytes(b"one")
    (root / "second" / "task.toml").write_bytes(b"two")
    baseline = digest_harbor_tree(root)

    (root / "first" / "task.toml").rename(root / "first" / "renamed.toml")

    assert digest_harbor_tree(root) != baseline


def test_digest_changes_with_file_contents_and_executable_bits(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    script = root / "run.sh"
    script.write_bytes(b"echo one\n")
    script.chmod(0o644)
    baseline = digest_harbor_tree(root)

    script.chmod(0o600)
    assert digest_harbor_tree(root) == baseline
    script.chmod(0o700)
    executable = digest_harbor_tree(root)
    assert executable != baseline
    script.write_bytes(b"echo two\n")
    assert digest_harbor_tree(root) != executable


def test_digest_ignores_directory_permissions(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "task.toml").write_bytes(b"task")
    nested.chmod(0o755)
    baseline = digest_harbor_tree(root)

    nested.chmod(0o700)

    assert digest_harbor_tree(root) == baseline


def test_digest_hashes_relative_symlink_target_spelling(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "target.txt").write_bytes(b"contents")
    link = root / "link.txt"
    link.symlink_to("target.txt")
    baseline = digest_harbor_tree(root)

    link.unlink()
    link.symlink_to("./target.txt")

    assert digest_harbor_tree(root) != baseline


def test_digest_accepts_contained_parent_relative_symlink(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "target.txt").write_bytes(b"contents")
    (nested / "link.txt").symlink_to("../target.txt")

    assert len(digest_harbor_tree(root)) == 64


def test_digest_resolves_parent_component_after_directory_symlink(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "nested" / "sub").mkdir(parents=True)
    (root / "nested" / "file.txt").write_bytes(b"contents")
    (root / "alias").symlink_to("nested/sub", target_is_directory=True)
    link = root / "link.txt"
    link.symlink_to("alias/../file.txt")

    assert link.read_bytes() == b"contents"
    assert len(digest_harbor_tree(root)) == 64


def test_digest_rejects_missing_component_before_parent_component(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "file.txt").write_bytes(b"contents")
    link = root / "link.txt"
    link.symlink_to("missing/../file.txt")

    assert not link.exists()
    with pytest.raises(FileNotFoundError):
        digest_harbor_tree(root)


def test_digest_resolves_parent_component_in_chained_symlink_target(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "nested" / "sub").mkdir(parents=True)
    (root / "nested" / "file.txt").write_bytes(b"contents")
    (root / "target-link").symlink_to("nested/sub/../file.txt")
    chain = root / "chain-link"
    chain.symlink_to("target-link")

    assert chain.read_bytes() == b"contents"
    assert len(digest_harbor_tree(root)) == 64


def test_digest_allows_repeated_contained_link_after_expansion(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "file").write_bytes(b"content")
    (root / "alias").symlink_to("sub", target_is_directory=True)
    link = root / "link"
    link.symlink_to("alias/../alias/file")

    assert link.read_bytes() == b"content"
    assert len(digest_harbor_tree(root)) == 64


def test_digest_follows_directory_symlink_without_deduplicating_aliases(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "target").mkdir(parents=True)
    (root / "target" / "task.toml").write_bytes(b"task")
    (root / "first-alias").symlink_to("target", target_is_directory=True)
    baseline = digest_harbor_tree(root)
    (root / "second-alias").symlink_to("target", target_is_directory=True)

    assert digest_harbor_tree(root) != baseline


@pytest.mark.parametrize("target", ["/tmp", "/"])
def test_digest_rejects_absolute_symlinks(tmp_path: Path, target: str) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "link").symlink_to(target)

    with pytest.raises(ValueError, match="absolute symlink"):
        digest_harbor_tree(root)


def test_digest_rejects_absolute_symlink_encountered_in_chain(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "z-absolute").symlink_to("/tmp")
    (root / "a-chain").symlink_to("z-absolute/item")

    with pytest.raises(ValueError, match="absolute symlink"):
        digest_harbor_tree(root)


def test_digest_rejects_root_escaping_symlink(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    (root / "link").symlink_to("../outside.txt")

    with pytest.raises(ValueError, match="escapes root"):
        digest_harbor_tree(root)


def test_digest_rejects_dangling_symlink(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "link").symlink_to("missing")

    with pytest.raises(FileNotFoundError):
        digest_harbor_tree(root)


def test_digest_rejects_symlink_chain_cycle(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "first").symlink_to("second")
    (root / "second").symlink_to("first")

    with pytest.raises(ValueError, match="cyclic symlink"):
        digest_harbor_tree(root)


def test_digest_rejects_directory_symlink_cycle(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "back").symlink_to("..", target_is_directory=True)

    with pytest.raises(ValueError, match="cyclic symlink"):
        digest_harbor_tree(root)


def test_digest_rejects_non_directory_roots(tmp_path: Path) -> None:
    file_root = tmp_path / "file"
    file_root.write_bytes(b"file")

    with pytest.raises(NotADirectoryError):
        digest_harbor_tree(file_root)
    with pytest.raises(FileNotFoundError):
        digest_harbor_tree(tmp_path / "missing")


def test_digest_rejects_special_files(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    os.mkfifo(root / "pipe")

    with pytest.raises(ValueError, match="unsupported filesystem entry"):
        digest_harbor_tree(root)


def test_digest_rejects_unreadable_files(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root can read mode-000 files")
    root = tmp_path / "tree"
    root.mkdir()
    unreadable = root / "unreadable"
    unreadable.write_bytes(b"secret")
    unreadable.chmod(0)
    try:
        with pytest.raises(PermissionError):
            digest_harbor_tree(root)
    finally:
        unreadable.chmod(0o600)
