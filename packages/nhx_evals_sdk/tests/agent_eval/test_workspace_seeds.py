# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pytest
from nhx_evals_sdk.agent_eval import workspace_seeds
from nhx_evals_sdk.agent_eval.workspace_seeds import (
    InlineSeed,
    PathSeed,
    WorkspaceSeedError,
    parse_seed,
    register_seed_handler,
    seed_workspace,
)
from pydantic import BaseModel


def test_parse_bare_string_is_inline_text() -> None:
    seed = parse_seed("hello")
    assert isinstance(seed, InlineSeed)
    assert seed.content == "hello"
    assert seed.encoding == "text"


def test_parse_dispatches_on_registered_kind() -> None:
    assert isinstance(parse_seed({"kind": "inline", "content": "x"}), InlineSeed)
    assert isinstance(parse_seed({"kind": "path", "path": "/tmp/x"}), PathSeed)


def test_parse_rejects_unregistered_kind() -> None:
    # The SDK ships only inline/path; a kind no consumer has registered is a generic unknown-kind error.
    with pytest.raises(WorkspaceSeedError, match="no handler registered for seed kind 'url'"):
        parse_seed({"kind": "url", "href": "http://x"})


def test_seed_inline_text_and_bare_string(tmp_path: Path) -> None:
    written = seed_workspace(
        tmp_path,
        {"a.txt": "bare", "b.txt": {"kind": "inline", "content": "typed"}},
    )
    assert sorted(written) == ["a.txt", "b.txt"]
    assert (tmp_path / "a.txt").read_text() == "bare"
    assert (tmp_path / "b.txt").read_text() == "typed"


def test_seed_inline_base64_writes_binary(tmp_path: Path) -> None:
    payload = b"\x00\x01binary\xff"
    seed_workspace(
        tmp_path, {"blob.bin": {"kind": "inline", "content": base64.b64encode(payload).decode(), "encoding": "base64"}}
    )
    assert (tmp_path / "blob.bin").read_bytes() == payload


def test_seed_bad_base64_raises(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceSeedError, match="base64"):
        seed_workspace(tmp_path, {"x": {"kind": "inline", "content": "not base64!!", "encoding": "base64"}})


def test_seed_path_reads_local_file(tmp_path: Path) -> None:
    source = tmp_path / "src.py"
    source.write_text("print('hi')\n")
    dest_root = tmp_path / "ws"
    seed_workspace(dest_root, {"nested/copied.py": {"kind": "path", "path": str(source)}})
    assert (dest_root / "nested" / "copied.py").read_text() == "print('hi')\n"


def test_seed_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceSeedError, match="could not be read"):
        seed_workspace(tmp_path, {"x.py": {"kind": "path", "path": str(tmp_path / "nope.py")}})


def test_seed_unregistered_kind_raises(tmp_path: Path) -> None:
    # Seeding a kind the SDK doesn't ship (and nobody registered) is a generic unknown-kind error —
    # the SDK carries no awareness of platform kinds like 'fileset' or where they resolve.
    with pytest.raises(WorkspaceSeedError, match="no handler registered for seed kind 'url'"):
        seed_workspace(tmp_path, {"data.csv": {"kind": "url", "href": "http://x/data.csv"}})


def test_register_custom_handler_extends_kinds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A consumer (e.g. the plugin) can register a new kind without the SDK knowing about it.
    class _UpperSeed(BaseModel):
        kind: Literal["upper"] = "upper"
        content: str

    class _UpperHandler:
        kind = "upper"

        def parse(self, value: Mapping[str, Any]) -> BaseModel:
            return _UpperSeed.model_validate(value)

        def resolve(self, seed: BaseModel) -> bytes:
            assert isinstance(seed, _UpperSeed)
            return seed.content.upper().encode("utf-8")

    # Isolate the global registry to this test so the extra kind doesn't leak to others.
    monkeypatch.setattr(workspace_seeds, "_HANDLERS", dict(workspace_seeds._HANDLERS))
    register_seed_handler(_UpperHandler())

    seed_workspace(tmp_path, {"shout.txt": {"kind": "upper", "content": "hi"}})
    assert (tmp_path / "shout.txt").read_text() == "HI"


def test_seed_rejects_path_escaping_workspace(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceSeedError, match="escapes the workspace"):
        seed_workspace(tmp_path / "ws", {"../escape.txt": "x"})


def test_seed_none_is_noop(tmp_path: Path) -> None:
    assert seed_workspace(tmp_path, None) == []
