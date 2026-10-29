# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The templates that keep a compose task's env values out of Harbor's persisted config."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from nhx_sandbox.compose_env import ENV_PREFIX, ComposeEnvError, expand_env, externalize_env
from nhx_sandbox.harbor_opensandbox_services import ComposeServices

SPEC: dict[str, Any] = {
    "services": [
        {
            "name": "db",
            "image": "docker.io/library/postgres:16",
            "env": {"POSTGRES_PASSWORD": "pg-pass", "PGDATA": "/data"},
        },
        {"name": "api", "image": "registry.example.com/api:1", "env": {"VERIFIER_TOKEN": "tok$$en"}},
        {"name": "seed", "image": "registry.example.com/seed:1"},
    ],
    "main": {"env": {"MYSQL_PASSWORD": "my-pass", "EMPTY": ""}},
}


def _values(spec: dict[str, Any]) -> list[str]:
    return [value for entry in [*spec["services"], spec["main"]] for value in entry.get("env", {}).values()]


def test_every_value_becomes_a_template_and_the_values_move_to_the_environment() -> None:
    environ: dict[str, str] = {}

    externalized = externalize_env(SPEC, environ)

    # Services in order, then main; each one's variables sorted by name.
    assert externalized["services"][0]["env"] == {
        "PGDATA": "${NEMO_COMPOSE_ENV_0}",
        "POSTGRES_PASSWORD": "${NEMO_COMPOSE_ENV_1}",
    }
    assert externalized["services"][1]["env"] == {"VERIFIER_TOKEN": "${NEMO_COMPOSE_ENV_2}"}
    assert "env" not in externalized["services"][2]
    assert externalized["main"]["env"] == {
        "EMPTY": "${NEMO_COMPOSE_ENV_3}",
        "MYSQL_PASSWORD": "${NEMO_COMPOSE_ENV_4}",
    }
    assert environ == {
        "NEMO_COMPOSE_ENV_0": "/data",
        "NEMO_COMPOSE_ENV_1": "pg-pass",
        "NEMO_COMPOSE_ENV_2": "tok$$en",
        "NEMO_COMPOSE_ENV_3": "",
        "NEMO_COMPOSE_ENV_4": "my-pass",
    }
    assert not {"pg-pass", "my-pass", "tok$$en"} & set(str(externalized))


def test_the_spec_it_was_given_is_left_alone() -> None:
    before = copy.deepcopy(SPEC)

    externalize_env(SPEC, {})

    assert SPEC == before


def test_the_same_compose_file_gives_the_same_config_whatever_order_its_variables_came_in() -> None:
    """Harbor refuses to resume a job directory whose stored config differs, so a retried Job would lose its trials."""
    reordered = copy.deepcopy(SPEC)
    reordered["services"][0]["env"] = dict(reversed(list(SPEC["services"][0]["env"].items())))

    assert externalize_env(reordered, {}) == externalize_env(SPEC, {})


def test_a_second_call_leaves_only_its_own_variables_and_none_of_the_jobs_others() -> None:
    environ = {"OPEN_SANDBOX_API_KEY": "key", "NEMO_COMPOSE_ENV_9": "left over"}

    externalize_env(SPEC, environ)
    externalize_env(
        {"services": [{"name": "db", "image": "docker.io/library/postgres:16", "env": {"A": "1"}}]}, environ
    )

    assert environ == {"OPEN_SANDBOX_API_KEY": "key", f"{ENV_PREFIX}0": "1"}


def test_a_spec_without_env_sets_nothing() -> None:
    spec = {"services": [{"name": "db", "image": "docker.io/library/postgres:16"}]}
    environ: dict[str, str] = {}

    assert externalize_env(spec, environ) == spec
    assert environ == {}


def test_expanding_gives_back_exactly_what_the_spec_carried() -> None:
    """Including ``$$``, which the spec escapes for Kubernetes and the environment must not touch, and empty values."""
    environ: dict[str, str] = {}
    externalized = externalize_env(SPEC, environ)

    assert expand_env(ComposeServices.model_validate(externalized), environ) == ComposeServices.model_validate(SPEC)


def test_only_our_templates_are_expanded() -> None:
    """A task can't make a value read another variable of the Job's environment, its API key for one."""
    spec = {
        "services": [
            {
                "name": "db",
                "image": "docker.io/library/postgres:16",
                "env": {
                    "A": "${OPEN_SANDBOX_API_KEY}",
                    "B": "${NEMO_COMPOSE_ENV_x}",
                    "C": "prefix-${NEMO_COMPOSE_ENV_0}",
                    "D": "plain",
                },
            }
        ]
    }
    environ = {"OPEN_SANDBOX_API_KEY": "key", "NEMO_COMPOSE_ENV_0": "value"}

    expanded = expand_env(ComposeServices.model_validate(spec), environ)

    assert expanded.services[0].env == spec["services"][0]["env"]


def test_a_missing_variable_names_the_variable_and_never_a_value() -> None:
    spec = {
        "services": [
            {
                "name": "db",
                "image": "docker.io/library/postgres:16",
                "env": {"POSTGRES_PASSWORD": "${NEMO_COMPOSE_ENV_0}", "OTHER": "visible-value"},
            }
        ]
    }

    with pytest.raises(ComposeEnvError) as error:
        expand_env(ComposeServices.model_validate(spec), {})

    assert "NEMO_COMPOSE_ENV_0" in str(error.value)
    assert "POSTGRES_PASSWORD" in str(error.value)
    assert "visible-value" not in str(error.value)
