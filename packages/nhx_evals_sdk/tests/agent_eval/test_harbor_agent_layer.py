# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which agent installs are reused across trials, and the key a reused install is found under."""

from datetime import date
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor", reason="the key names the Harbor version whose install code runs")

from nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer import FABRIC_INSTALLED_AGENT, agent_install

_IMAGE = "sha256:" + "e" * 64
_FABRIC = {"fabric_package": "nemo-fabric[codex,relay]==0.5.0", "fabric_config": {"metadata": {"name": "a"}}}


def _install(*, name: str | None = None, import_path: str | None = None, agent_dir: Path | None = None, **kwargs: Any):
    return agent_install(
        agent_name=name,
        agent_import_path=import_path,
        agent_dir=agent_dir,
        agent_kwargs=kwargs,
        today=lambda: date(2026, 10, 30),
    )


def _key(**kwargs: Any) -> str:
    install = _install(**kwargs)
    assert install is not None
    return install.layer_key(_IMAGE)


@pytest.mark.parametrize("name", ["codex", "claude-code"])
def test_harbors_installed_agents_are_reused(name: str) -> None:
    install = _install(name=name, version="1.2.3")

    assert install is not None and install.agent == f"harbor:{name}" and install.inputs == {"version": "1.2.3"}


@pytest.mark.parametrize(
    "agent",
    [
        {"name": "oracle"},
        {"name": "nop"},
        {"name": "codex", "agent_dir": Path("/agents")},
        {"import_path": "harbor_wrapper:WrappedAgent"},
        {"import_path": FABRIC_INSTALLED_AGENT},  # no fabric_package
    ],
    ids=["oracle", "nop", "agent-dir", "other-import-path", "fabric-without-package"],
)
def test_agents_whose_setup_may_do_anything_are_not_reused(agent: dict[str, Any]) -> None:
    assert _install(**agent) is None


def test_options_that_dont_change_the_install_keep_the_key() -> None:
    base = _key(name="codex", version="0.162.1")

    assert _key(name="codex", version="0.162.1", reasoning_effort="high", model_name="openai/gpt-5") == base


def test_what_changes_the_install_changes_the_key() -> None:
    base = _key(name="codex", version="0.162.1")

    assert _key(name="codex", version="0.163.0") != base
    assert _key(name="claude-code", version="0.162.1") != base
    install = _install(name="codex", version="0.162.1")
    assert install is not None and install.layer_key("sha256:" + "f" * 64) != base


def test_a_registered_fabric_agent_is_keyed_by_its_install_not_its_config() -> None:
    base = _key(import_path=FABRIC_INSTALLED_AGENT, **_FABRIC)

    other_config = {**_FABRIC, "fabric_config": {"metadata": {"name": "b"}}, "fabric_default_max_turns": 9}
    assert _key(import_path=FABRIC_INSTALLED_AGENT, **other_config) == base
    for change in (
        {"fabric_package": "nemo-fabric[deepagents,relay]==0.5.0"},
        {"fabric_python_version": "3.13"},
        {"fabric_uv_version": "0.13.0"},
    ):
        assert _key(import_path=FABRIC_INSTALLED_AGENT, **{**_FABRIC, **change}) != base


def test_an_unpinned_version_is_reused_for_one_day() -> None:
    install = _install(name="codex")

    assert install is not None and install.inputs == {"version": "unpinned:2026-10-30"}
    tomorrow = agent_install(
        agent_name="codex", agent_import_path=None, agent_dir=None, agent_kwargs={}, today=lambda: date(2026, 10, 31)
    )
    assert tomorrow is not None and tomorrow.layer_key(_IMAGE) != install.layer_key(_IMAGE)
