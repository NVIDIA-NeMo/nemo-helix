# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor task's compose file resolved as Harbor would hand it to ``docker compose``.

Covers finding the file, merging it over Harbor's base file, interpolation, ``.env`` and ``env_file``, and refusing
malformed or hostile input. Every test writes its own task directory under ``tmp_path``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml
from harbor.environments.docker import COMPOSE_BUILD_PATH, COMPOSE_PREBUILT_PATH
from nemo_evals.harbor.compose.models import ComposeError, ComposeProblems
from nemo_evals.harbor.compose.resolve import (
    Interpolator,
    ResolvedCompose,
    find_compose_file,
    merge_documents,
    read_env_files,
    resolve_compose,
)


def _task(
    root: Path,
    compose: str | None,
    *,
    task_toml: str = "",
    files: dict[str, str] | None = None,
    compose_name: str = "docker-compose.yaml",
) -> Path:
    """A task directory named ``demo`` with the given compose file and any extra files under ``environment/``."""
    task = root / "demo"
    environment = task / "environment"
    environment.mkdir(parents=True)

    if compose is not None:
        (environment / compose_name).write_text(textwrap.dedent(compose), encoding="utf-8")
    if task_toml:
        (task / "task.toml").write_text(textwrap.dedent(task_toml), encoding="utf-8")

    for name, content in (files or {}).items():
        (environment / name).parent.mkdir(parents=True, exist_ok=True)
        (environment / name).write_text(content, encoding="utf-8")
    return task


def _resolve(task: Path) -> ResolvedCompose:
    """Resolve a task's compose file, raising everything recorded along the way as one error."""
    compose_file = find_compose_file(task)
    assert compose_file is not None
    compose_path = f"{task.name}/environment/docker-compose.yaml"

    problems = ComposeProblems()
    resolved = resolve_compose(task, compose_file, compose_path, problems)
    problems.raise_if_any(compose_path)
    return resolved


def _resolved(tmp_path: Path, compose: str, **kwargs: Any) -> ResolvedCompose:
    """The resolved compose file of a task with this compose file."""
    return _resolve(_task(tmp_path, compose, **kwargs))


def _error(tmp_path: Path, compose: str, **kwargs: Any) -> ComposeError:
    """The error resolving a task with this compose file raises."""
    with pytest.raises(ComposeError) as caught:
        _resolve(_task(tmp_path, compose, **kwargs))
    return caught.value


# ---------------------------------------------------------------------------------------------------------------------
# Interpolation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", "plain"),
        ("$$(cat /x)", "$(cat /x)"),
        ("$$$$", "$$"),
        ("${A}-$A", "one-one"),
        ("${MISSING:-fallback}", "fallback"),
        ("${EMPTY:-fallback}", "fallback"),
        ("${EMPTY-kept}", ""),
        ("${MISSING-fallback}", "fallback"),
        ("${MISSING:-${A}}", "one"),
        ("${MISSING}/$MISSING", "/"),
        ("${A:+alt}", "alt"),
        ("${EMPTY:+alt}", ""),
        ("${EMPTY+alt}", "alt"),
        ("${MISSING+alt}", ""),
        ("${A:?needed}", "one"),
        ("${EMPTY?needed}", ""),
        ("[ $? -eq 28 ]", "[ $? -eq 28 ]"),
        ("cost: $5", "cost: $5"),
        ("end$", "end$"),
    ],
)
def test_interpolation_follows_compose(text: str, expected: str) -> None:
    problems = ComposeProblems()
    assert Interpolator({"A": "one", "EMPTY": ""}, problems).string(text, "x") == expected
    problems.raise_if_any("compose.yaml")


@pytest.mark.parametrize("text", ["${NOPE:?set NOPE}", "${EMPTY:?set EMPTY}", "${NOPE?set NOPE}"])
def test_a_required_variable_that_is_missing_is_invalid(text: str) -> None:
    problems = ComposeProblems()
    Interpolator({"EMPTY": ""}, problems).string(text, "services.a.environment.X")

    with pytest.raises(ComposeError) as caught:
        problems.raise_if_any("compose.yaml")

    assert caught.value.code == "compose_invalid"
    assert "services.a.environment.X:" in caught.value.detail
    assert "is required: set" in caught.value.detail


def test_an_invalid_value_does_not_hide_unsupported_features() -> None:
    problems = ComposeProblems()
    problems.unsupported("services.a.privileged")
    problems.invalid("services.b.ports[0]", "expected a port")

    with pytest.raises(ComposeError) as caught:
        problems.raise_if_any("compose.yaml")

    assert caught.value.code == "compose_invalid"
    assert "services.b.ports[0]: expected a port" in caught.value.detail
    assert caught.value.paths == ("services.a.privileged",)


def test_unset_variable_is_empty_as_under_compose(tmp_path: Path) -> None:
    resolved = _resolved(
        tmp_path,
        """
        services:
          db:
            image: postgres:18
            environment:
              TOKEN: ${HOST_TOKEN}
              TEST_DIR: ${TEST_DIR}
        """,
    )
    assert resolved.services["db"]["environment"] == {"TOKEN": "", "TEST_DIR": ""}


def test_task_toml_env_and_harbor_infra_variables_interpolate(tmp_path: Path) -> None:
    resolved = _resolved(
        tmp_path,
        """
        services:
          main:
            environment:
              DB_HOST: db
              CPUS: ${CPUS}
              MEM: ${MEMORY}
          db:
            image: postgres:${PG_VERSION}
            environment:
              PASSWORD: ${DB_PASSWORD:-fallback}
        """,
        task_toml="""
        [environment]
        cpus = 2
        memory_mb = 4096
        env = { PG_VERSION = "17", DB_PASSWORD = "from-task" }
        """,
    )

    assert resolved.services["db"]["image"] == "postgres:17"
    assert resolved.services["db"]["environment"] == {"PASSWORD": "from-task"}
    assert resolved.main["environment"] == {"DB_HOST": "db", "CPUS": "2", "MEM": "4096M"}


REGISTRY_COMPOSE = "services:\n  cache:\n    image: ${REGISTRY:-docker.io}/library/redis:7\n"


def test_environment_dotenv_file_interpolates_as_compose_reads_it(tmp_path: Path) -> None:
    resolved = _resolved(tmp_path, REGISTRY_COMPOSE, files={".env": "# registry\nREGISTRY=mirror.example.com\n"})
    assert resolved.services["cache"]["image"] == "mirror.example.com/library/redis:7"


def test_dotenv_has_the_lowest_priority(tmp_path: Path) -> None:
    compose = (
        "services:\n  cache:\n    image: ${REGISTRY}/redis:${TAG}\n    environment:\n      HOME_DIR: ${CONTEXT_DIR}\n"
    )
    resolved = _resolved(
        tmp_path,
        compose,
        task_toml='[environment]\nenv = { REGISTRY = "task.example.com", CONTEXT_DIR = "ignored" }\n',
        files={".env": "REGISTRY=dotenv.example.com\nTAG=7\nCONTEXT_DIR=/elsewhere\n"},
    )
    assert resolved.services["cache"]["image"] == "task.example.com/redis:7"
    assert resolved.services["cache"]["environment"] == {"HOME_DIR": "."}


def test_the_hosts_environment_is_never_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGISTRY", "host.example.com")
    assert _resolved(tmp_path, REGISTRY_COMPOSE).services["cache"]["image"] == "docker.io/library/redis:7"


def test_dotenv_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    task = _task(tmp_path, REGISTRY_COMPOSE)
    target = tmp_path / "outside.env"
    target.write_text("REGISTRY=leak.example.com\n", encoding="utf-8")
    (task / "environment" / ".env").symlink_to(target)
    with pytest.raises(ComposeError, match="symlink"):
        _resolve(task)


def test_dotenv_values_end_at_an_inline_comment_unless_quoted(tmp_path: Path) -> None:
    compose = "services:\n  db:\n    image: redis\n    environment: {A: '${A}', B: '${B}', C: '${C}'}\n"
    dotenv = 'A=plain # a note\nB="kept # as written"\nC=url#anchor\n'

    resolved = _resolved(tmp_path, compose, files={".env": dotenv})

    assert resolved.services["db"]["environment"] == {"A": "plain", "B": "kept # as written", "C": "url#anchor"}


def test_a_quoted_dotenv_value_followed_by_a_comment_loses_its_quotes_and_the_comment(tmp_path: Path) -> None:
    compose = "services:\n  db:\n    image: pg:${TAG}\n    environment: {NOTE: '${NOTE}'}\n"
    dotenv = "TAG=\"18\" # pg\nNOTE='two words'   # c\n"

    resolved = _resolved(tmp_path, compose, files={".env": dotenv})

    assert resolved.services["db"]["image"] == "pg:18"
    assert resolved.services["db"]["environment"] == {"NOTE": "two words"}


def test_a_dotenv_value_quoted_across_lines_is_refused(tmp_path: Path) -> None:
    task = _task(tmp_path, "services:\n  db:\n    image: redis\n", files={".env": 'A="multi\nline"\n'})

    with pytest.raises(ComposeError, match="multi-line"):
        _resolve(task)


# ---------------------------------------------------------------------------------------------------------------------
# Finding the file


def test_task_without_a_compose_file_has_none(tmp_path: Path) -> None:
    assert find_compose_file(_task(tmp_path, None)) is None


@pytest.mark.parametrize("name", ["docker-compose.yml", "compose.yaml", "compose.yml"])
def test_compose_file_with_a_name_harbor_ignores_is_rejected_with_a_rename_hint(tmp_path: Path, name: str) -> None:
    with pytest.raises(ComposeError) as caught:
        find_compose_file(_task(tmp_path, "services: {}\n", compose_name=name))

    assert caught.value.code == "compose_unsupported_name"
    assert "rename" in caught.value.detail
    assert name in (caught.value.compose_path or "")


# ---------------------------------------------------------------------------------------------------------------------
# Merge with Harbor's base file


@pytest.mark.parametrize(
    ("merge_base", "merge_over", "expected"),
    [
        ({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}, {"a": {"x": 1, "y": 3}}),
        ({"expose": ["1"]}, {"expose": ["1", "2"]}, {"expose": ["1", "2"]}),
        ({"cap_add": ["A"]}, {"cap_add": ["B"]}, {"cap_add": ["A", "B"]}),
        ({"command": ["a"]}, {"command": ["b"]}, {"command": ["b"]}),
    ],
)
def test_merge_follows_compose_rules(merge_base: dict, merge_over: dict, expected: dict) -> None:
    assert merge_documents(merge_base, merge_over) == expected


def test_main_is_the_task_file_laid_over_harbors_build_base(tmp_path: Path) -> None:
    resolved = _resolved(
        tmp_path,
        "services:\n  main:\n    environment: [A=b]\n    build: {dockerfile: Dockerfile.agent}\n  db:\n    image: redis\n",
    )

    assert not resolved.prebuilt
    assert resolved.harbor_main["command"] == ["sh", "-c", "sleep infinity"]
    # The task's `main` changes Harbor's; what it leaves out comes from Harbor's base file.
    assert resolved.main["build"] == {"context": ".", "dockerfile": "Dockerfile.agent"}
    assert resolved.main["command"] == ["sh", "-c", "sleep infinity"]
    assert resolved.main["environment"] == {"A": "b"}
    assert list(resolved.services) == ["db"]


def test_a_prebuilt_task_is_laid_over_harbors_prebuilt_base(tmp_path: Path) -> None:
    resolved = _resolved(
        tmp_path,
        "services:\n  db:\n    image: redis\n",
        task_toml='[environment]\ndocker_image = "ghcr.io/org/sort:1"\n',
    )

    assert resolved.prebuilt
    assert resolved.main["image"] == "ghcr.io/org/sort:1"


@pytest.mark.parametrize("base_path", [COMPOSE_BUILD_PATH, COMPOSE_PREBUILT_PATH], ids=["build", "prebuilt"])
def test_harbors_base_files_set_only_what_the_merge_is_checked_against(base_path: Path) -> None:
    # `merge_documents` repeats only the Compose merge rules these keys need. If a Harbor upgrade sets anything else,
    # check that the merge still matches `docker compose config` for it, then add it here.
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    assert set(base) == {"services"}
    assert set(base["services"]) == {"main"}
    assert set(base["services"]["main"]) <= {"build", "image", "command", "pull_policy"}


# ---------------------------------------------------------------------------------------------------------------------
# env_file


def _env_files(task: Path, value: Any) -> dict[str, str]:
    """Read a service's ``env_file`` value from the task, raising everything recorded as one error."""
    problems = ComposeProblems()
    variables = read_env_files(value, "services.a.env_file", task, problems)
    problems.raise_if_any("demo/environment/docker-compose.yaml")
    return variables


def test_env_files_are_read_from_the_archive_and_later_files_win(tmp_path: Path) -> None:
    task = _task(tmp_path, None, files={"a.env": "# comment\nX=one\nY='quoted'\n", "b.env": "export X=two\nZ=3\n"})
    assert _env_files(task, ["a.env", "b.env"]) == {"X": "two", "Y": "quoted", "Z": "3"}


def test_an_env_file_outside_environment_is_unsupported(tmp_path: Path) -> None:
    with pytest.raises(ComposeError) as caught:
        _env_files(_task(tmp_path, None), "../secret.env")
    assert caught.value.code == "compose_unsupported"


def test_a_missing_env_file_is_invalid_unless_marked_optional(tmp_path: Path) -> None:
    task = _task(tmp_path, None)
    with pytest.raises(ComposeError) as caught:
        _env_files(task, "nope.env")
    assert caught.value.code == "compose_invalid"
    assert _env_files(task, [{"path": "nope.env", "required": False}]) == {}


@pytest.mark.parametrize(
    ("content", "complaint"),
    [
        (b"A=\xff\xfe\n", "can't read"),
        (b"A=1\n" * 100_000, "byte limit"),
    ],
    ids=["not-utf8", "oversized"],
)
def test_an_env_file_that_cannot_be_read_as_data_is_refused(tmp_path: Path, content: bytes, complaint: str) -> None:
    task = _task(tmp_path, None)
    (task / "environment" / "data.env").write_bytes(content)

    with pytest.raises(ComposeError) as caught:
        _env_files(task, "data.env")

    assert caught.value.code == "compose_invalid"
    assert complaint in caught.value.detail and "data.env" in caught.value.detail


def test_an_env_file_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    task = _task(tmp_path, None)
    target = tmp_path / "outside.env"
    target.write_text("A=1\n", encoding="utf-8")
    (task / "environment" / "data.env").symlink_to(target)

    with pytest.raises(ComposeError, match="data.env is a symlink"):
        _env_files(task, "data.env")


# ---------------------------------------------------------------------------------------------------------------------
# Hostile and malformed input


@pytest.mark.parametrize(
    "compose",
    ["- just\n- a list\n", "just a string\n", "version: '3'\n", "services: [a, b]\n", "services: {a: [1]}\n", "a: [\n"],
)
def test_malformed_compose_file_is_invalid(tmp_path: Path, compose: str) -> None:
    assert _error(tmp_path, compose).code == "compose_invalid"


def test_empty_compose_file_is_invalid(tmp_path: Path) -> None:
    assert _error(tmp_path, "").code == "compose_invalid"


def test_oversized_compose_file_is_invalid(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  a:\n    image: x\n" + "# padding\n" * 30_000)
    assert error.code == "compose_invalid"
    assert "limit" in error.detail


def test_compose_file_is_never_executed(tmp_path: Path) -> None:
    assert _error(tmp_path, "services: !!python/object/apply:os.system ['echo hi']\n").code == "compose_invalid"


def test_yaml_aliases_that_expand_without_bound_are_refused(tmp_path: Path) -> None:
    # Each level holds nine references to the level below it, so nine levels expand to hundreds of millions of values.
    levels = ["l0: &l0 [x, x, x, x, x, x, x, x, x]"]
    levels += [f"l{level}: &l{level} [{', '.join([f'*l{level - 1}'] * 9)}]" for level in range(1, 9)]
    compose = "services:\n  db:\n    image: redis\nx-bomb:\n" + "".join(f"  {level}\n" for level in levels)

    error = _error(tmp_path, compose)

    assert error.code == "compose_invalid"
    assert "expands to more than" in error.detail


def test_a_file_nested_too_deeply_is_refused(tmp_path: Path) -> None:
    compose = "services: {db: {image: redis}}\nx-deep: " + "[" * 5000 + "]" * 5000 + "\n"
    assert _error(tmp_path, compose).code == "compose_invalid"


def test_a_top_level_key_compose_has_no_sandbox_equivalent_for_is_unsupported(tmp_path: Path) -> None:
    error = _error(tmp_path, "services:\n  db:\n    image: redis\nsecrets: {s: {file: ./s}}\nx-note: ignored\n")
    assert error.code == "compose_unsupported"
    assert error.paths == ("secrets",)
