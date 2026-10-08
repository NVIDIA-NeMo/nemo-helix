# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import pytest
from _ext_doubles import SPEC_ROLES, spec_dict
from nemo_ext_spec import ServicesSpecError, parse_spec


def test_example_spec_parses() -> None:
    assert parse_spec(json.dumps(spec_dict())).roles() == SPEC_ROLES


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s.pop("version"), "version"),
        (lambda s: s["services"][0].update(privileged=True), "privileged"),
        (lambda s: s["services"][0].update(name="egress"), "reserved"),
        (lambda s: s["services"][0].update(name="main"), "reserved"),
        (lambda s: s["services"][0].update(name="Bad_Name"), "name"),
        (lambda s: s["services"][0].update(ports=[44772]), "used by the sandbox pod"),
        (lambda s: s["services"][1].update(ports=[5432]), "same port"),
        (lambda s: s["services"][0].update(ports=["5432"]), "ports"),
        (lambda s: s["services"][1].update(name="db"), "unique"),
        (lambda s: s["services"][0]["volume_mounts"][0].update(name="other"), "not declared"),
        (lambda s: s["services"][0]["volume_mounts"][0].update(mount_path="data"), "mount_path"),
        (lambda s: s["sandbox"].update(add_capabilities=["SYS_ADMIN"]), "add_capabilities"),
        (lambda s: s["services"][1].update(readiness={"exec": ["true"]}), "run_once"),
        (lambda s: s["services"][3].update(run_once=True), "mutually exclusive"),
        (lambda s: s["services"][0].update(readiness={"tcp": 5432}), "readiness"),
        (lambda s: s["services"][0].update(resources={"limits": {"gpu": "1"}}), "resources"),
        (lambda s: s["services"][0].update(shm_size="lots"), "shm_size"),
        (lambda s: s.update(services=[]), "services"),
    ],
)
def test_invalid_specs_are_rejected(mutate: Any, message: str) -> None:
    spec = spec_dict()
    mutate(spec)
    with pytest.raises(ServicesSpecError, match=message):
        parse_spec(json.dumps(spec))


def test_spec_error_is_a_value_error_so_the_server_returns_400() -> None:
    with pytest.raises(ValueError, match="not valid JSON|invalid extensions"):
        parse_spec("{not json")
