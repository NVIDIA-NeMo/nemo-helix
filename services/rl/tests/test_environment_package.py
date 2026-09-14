# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for environment FileSet manifest schemas and package validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from nmp.rl.schemas.environment import (
    AdapterWheelsV1Manifest,
    EnvironmentFormat,
    GymVerifiersDatasetRow,
)
from nmp.rl.tasks.environment.package import (
    build_policy_model_yaml,
    build_verifiers_agent_yaml,
    write_adapter_wheels_package,
)
from nmp.rl.tasks.environment.validate import (
    EnvironmentPackageValidationError,
    load_manifest,
    offline_wheel_install_required,
    validate_package_layout,
)


def test_adapter_wheels_manifest_roundtrip() -> None:
    raw = {
        "format": "adapter-wheels-v1",
        "adapter": {
            "agent": "verifiers_agent",
            "agent_type": "responses_api_agents",
            "image_config_root": "responses_api_agents/verifiers_agent",
        },
        "config_paths": ["configs/verifiers_agent.yaml"],
        "metadata": {
            "name": "ascii-tree",
            "hub_id": "primeintellect/ascii-tree",
            "vf_env_id": "ascii-tree",
            "adapter_agent": "verifiers_agent",
        },
    }
    manifest = AdapterWheelsV1Manifest.model_validate(raw)
    assert manifest.format == EnvironmentFormat.ADAPTER_WHEELS_V1
    assert manifest.adapter.agent == "verifiers_agent"


def test_config_paths_reject_traversal() -> None:
    with pytest.raises(ValueError, match="relative and contained"):
        AdapterWheelsV1Manifest.model_validate(
            {
                "format": "adapter-wheels-v1",
                "adapter": {"agent": "verifiers_agent"},
                "config_paths": ["../escape.yaml"],
                "metadata": {"name": "x"},
            }
        )


def test_write_adapter_package_layout(tmp_path: Path) -> None:
    wheels = tmp_path / "src_wheels"
    wheels.mkdir()
    (wheels / "fake-1.0.0-py3-none-any.whl").write_bytes(b"PK\x03\x04")

    env_root = tmp_path / "env"
    manifest = write_adapter_wheels_package(
        out_dir=env_root,
        hub_id="primeintellect/ascii-tree",
        wheels_src=wheels,
    )
    assert (env_root / "nemo-environment.yaml").is_file()
    assert (env_root / "configs" / "verifiers_agent.yaml").is_file()
    assert list((env_root / "wheels").glob("*.whl"))
    validate_package_layout(env_root, manifest)
    loaded = load_manifest(env_root)
    assert loaded.metadata.vf_env_id == "ascii-tree"


def test_reject_jsonl_in_environment_package(tmp_path: Path) -> None:
    env_root = tmp_path / "env"
    wheels = tmp_path / "w"
    wheels.mkdir()
    (wheels / "a-1.0-py3-none-any.whl").write_bytes(b"PK")
    manifest = write_adapter_wheels_package(
        out_dir=env_root,
        hub_id="primeintellect/test",
        wheels_src=wheels,
    )
    (env_root / "training.jsonl").write_text("{}\n")
    with pytest.raises(EnvironmentPackageValidationError, match="JSONL"):
        validate_package_layout(env_root, manifest)


def test_offline_install_required_for_adapter_wheels() -> None:
    m = AdapterWheelsV1Manifest.model_validate(
        {
            "format": "adapter-wheels-v1",
            "adapter": {"agent": "verifiers_agent"},
            "config_paths": ["configs/verifiers_agent.yaml"],
            "metadata": {"name": "t"},
        }
    )
    assert offline_wheel_install_required(m) is True


def test_gym_verifiers_dataset_row() -> None:
    row = GymVerifiersDatasetRow.model_validate(
        {
            "task_idx": 0,
            "vf_env_id": "ascii-tree",
            "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
            "agent_ref": {"type": "responses_api_agents", "name": "verifiers_agent"},
            "example_id": 0,
        }
    )
    assert row.agent_ref.name == "verifiers_agent"


def test_verifiers_agent_yaml_shape() -> None:
    doc = build_verifiers_agent_yaml("ascii-tree", {})
    assert "verifiers_agent" in doc
    assert doc["verifiers_agent"]["responses_api_agents"]["verifiers_agent"]["vf_env_id"] == "ascii-tree"


# NeMo-Gym's global-config parse validates `domain` against this closed set. An out-of-set
# value (notably "") demotes the server to an "almost-server" that never starts, and the Gym
# host exits with AlmostServerError — inside the sandbox, minutes into a GRPO run.
GYM_DOMAINS = {
    "math",
    "coding",
    "agent",
    "knowledge",
    "instruction_following",
    "long_context",
    "safety",
    "games",
    "translation",
    "e2e",
    "rlhf",
    "other",
}


def test_verifiers_agent_yaml_domain_is_a_valid_gym_domain() -> None:
    inner = build_verifiers_agent_yaml("ascii-tree", {})["verifiers_agent"]["responses_api_agents"]["verifiers_agent"]
    assert inner["domain"] in GYM_DOMAINS


def test_package_defines_the_model_server_the_agent_references(tmp_path: Path) -> None:
    """The agent's model_server ref must resolve inside the package itself.

    verifiers_agent points at responses_api_models/policy_model. If nothing defines that
    server, Gym rejects the merged config with ServerRefNotFoundError ("Available
    responses_api_models: (none)") and the Gym host never starts.
    """
    wheels = tmp_path / "src_wheels"
    wheels.mkdir()
    (wheels / "fake-1.0.0-py3-none-any.whl").write_bytes(b"PK\x03\x04")
    out = tmp_path / "env"
    manifest = write_adapter_wheels_package(out_dir=out, hub_id="primeintellect/ascii-tree", wheels_src=wheels)

    agent = yaml.safe_load((out / "configs" / "verifiers_agent.yaml").read_text())
    ref = agent["verifiers_agent"]["responses_api_agents"]["verifiers_agent"]["model_server"]

    policy = yaml.safe_load((out / "configs" / "policy_model.yaml").read_text())
    assert ref["name"] in policy, f"agent references {ref['name']!r}, package defines {list(policy)}"
    assert ref["type"] in policy[ref["name"]]

    # Both configs must be listed, or Gym never loads the half that is missing.
    assert set(manifest.config_paths) == {"configs/policy_model.yaml", "configs/verifiers_agent.yaml"}


def test_policy_model_interpolations_match_what_nemo_rl_injects() -> None:
    """The ${...} keys must line up with build_sandbox_global_config in NeMo-RL.

    That function sets policy_model_name / policy_api_key / policy_base_url on the Gym global
    config. A rename on either side leaves OmegaConf with an unresolvable interpolation.
    """
    server = build_policy_model_yaml()["policy_model"]["responses_api_models"]["vllm_model"]
    assert server["base_url"] == "${policy_base_url}"
    assert server["api_key"] == "${policy_api_key}"
    assert server["model"] == "${policy_model_name}"


def test_bootstrap_adapter_wheels_validate_only(tmp_path: Path) -> None:
    from nmp.rl.tasks.environment.bootstrap import bootstrap_environment_package

    wheels = tmp_path / "src_wheels"
    wheels.mkdir()
    (wheels / "fake-1.0.0-py3-none-any.whl").write_bytes(b"PK\x03\x04")
    env_root = tmp_path / "env"
    write_adapter_wheels_package(
        out_dir=env_root,
        hub_id="primeintellect/ascii-tree",
        wheels_src=wheels,
    )
    result = bootstrap_environment_package(env_root, install_wheels=False)
    assert result.manifest.format.value == "adapter-wheels-v1"
    assert result.image_config_root == "responses_api_agents/verifiers_agent"


def test_convert_with_wheels_dir_writes_layout(tmp_path: Path) -> None:
    from nmp.rl.tasks.environment.convert import ConvertEnvironmentSpec, convert_prime_environment

    wheels = tmp_path / "prebuilt"
    wheels.mkdir()
    (wheels / "ascii_tree-0.0.0-py3-none-any.whl").write_bytes(b"PK\x03\x04")
    out = tmp_path / "env"
    ds = tmp_path / "dataset"
    result = convert_prime_environment(
        ConvertEnvironmentSpec(
            hub_id="primeintellect/ascii-tree",
            out_dir=out,
            dataset_dir=ds,
            wheels_dir=wheels,
            dataset_size=0,
        )
    )
    manifest = load_manifest(result.environment_root)
    validate_package_layout(result.environment_root, manifest)
    assert list((result.environment_root / "wheels").glob("*.whl"))
    assert result.dataset_dir.is_dir()
    assert result.training_jsonl.is_file()
    assert not (result.dataset_dir / ".huggingface").exists()


def test_convert_rejects_empty_wheels_dir(tmp_path: Path) -> None:
    from nmp.rl.tasks.environment.convert import ConvertEnvironmentSpec, convert_prime_environment

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no \\*\\.whl"):
        convert_prime_environment(
            ConvertEnvironmentSpec(
                hub_id="primeintellect/ascii-tree",
                out_dir=tmp_path / "env",
                dataset_dir=tmp_path / "dataset",
                wheels_dir=empty,
                dataset_size=0,
            )
        )


def test_convert_vendors_hub_dataset_and_configures_its_sandbox_path(tmp_path: Path, monkeypatch) -> None:
    """Conversion isolates both environment and pre-imported Hugging Face cache settings."""
    import os
    import sys
    from types import ModuleType

    from nmp.rl.tasks.environment import convert as convert_mod
    from nmp.rl.tasks.environment.convert import ConvertEnvironmentSpec, convert_prime_environment

    wheels = tmp_path / "prebuilt"
    wheels.mkdir()
    (wheels / "ascii_tree-0.1.5-py3-none-any.whl").write_bytes(b"PK\x03\x04")

    developer_hf = tmp_path / "developer-hf"
    developer_hf.mkdir()
    (developer_hf / "should-not-copy").write_text("developer-cache", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(developer_hf))
    monkeypatch.setenv("HF_HUB_CACHE", str(developer_hf / "hub"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(developer_hf / "datasets"))

    # Model libraries read cache variables at import time. Simulate both having
    # already been imported before the converter establishes its isolated cache.
    hub_constants = ModuleType("huggingface_hub.constants")
    setattr(hub_constants, "HF_HOME", str(developer_hf))
    setattr(hub_constants, "HF_HUB_CACHE", str(developer_hf / "hub"))
    datasets_config = ModuleType("datasets.config")
    setattr(datasets_config, "HF_DATASETS_CACHE", developer_hf / "datasets")
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", hub_constants)
    monkeypatch.setitem(sys.modules, "datasets.config", datasets_config)

    class FakeDataset:
        column_names = ["prompt", "answer"]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: str):
            if key == "prompt":
                return [[{"role": "user", "content": "draw an ascii tree"}]]
            return [""]

        def to_parquet(self, path: str) -> None:
            Path(path).write_bytes(b"PARQUET")

    class FakeEnv:
        dataset = FakeDataset()

        def get_dataset(self, n: int = -1, seed: int | None = None):
            return self.dataset

    captured: dict[str, str] = {}

    def _fake_load(*_a, **_k):
        isolated = Path(os.environ["HF_HOME"])
        captured["hf_home"] = str(isolated)
        assert os.environ["HF_HUB_CACHE"] == str(isolated / "hub")
        assert os.environ["HF_DATASETS_CACHE"] == str(isolated / "datasets")
        assert getattr(hub_constants, "HF_HOME") == str(isolated)
        assert getattr(hub_constants, "HF_HUB_CACHE") == str(isolated / "hub")
        assert getattr(datasets_config, "HF_DATASETS_CACHE") == isolated / "datasets"
        (isolated / "hub").mkdir(parents=True)
        (isolated / "hub" / "datasets--primeintellect--ascii-tree").write_text("cached", encoding="utf-8")
        return FakeEnv()

    monkeypatch.setattr(convert_mod, "_install_hub_package_from_wheels", lambda *a, **k: None)
    monkeypatch.setattr(convert_mod, "_load_verifiers_environment", _fake_load)

    result = convert_prime_environment(
        ConvertEnvironmentSpec(
            hub_id="primeintellect/ascii-tree",
            out_dir=tmp_path / "env",
            dataset_dir=tmp_path / "dataset",
            wheels_dir=wheels,
            dataset_size=1,
        )
    )
    assert (result.dataset_dir / "hub_environment.parquet").read_bytes() == b"PARQUET"
    agent = yaml.safe_load((result.environment_root / "configs" / "verifiers_agent.yaml").read_text())
    vf_env_args = agent["verifiers_agent"]["responses_api_agents"]["verifiers_agent"]["vf_env_args"]
    assert vf_env_args["dataset_path"] == "/job/dataset/hub_environment.parquet"
    assert next((result.environment_root / "wheels").glob("ascii_tree-*.whl")).read_bytes() == b"PK\x03\x04"

    snap = result.dataset_dir / ".huggingface" / "hub" / "datasets--primeintellect--ascii-tree"
    assert snap.read_text(encoding="utf-8") == "cached"
    assert not (result.dataset_dir / ".huggingface" / "should-not-copy").exists()
    assert captured["hf_home"] != str(developer_hf)
    assert os.environ["HF_HOME"] == str(developer_hf)
    assert os.environ["HF_HUB_CACHE"] == str(developer_hf / "hub")
    assert os.environ["HF_DATASETS_CACHE"] == str(developer_hf / "datasets")
    assert getattr(hub_constants, "HF_HOME") == str(developer_hf)
    assert getattr(hub_constants, "HF_HUB_CACHE") == str(developer_hf / "hub")
    assert getattr(datasets_config, "HF_DATASETS_CACHE") == developer_hf / "datasets"


def test_split_train_validation_never_overlaps() -> None:
    """The old `or all_rows` fallback made train and validation identical."""
    from nmp.rl.tasks.environment.convert import split_train_validation

    rows = [{"task_idx": i} for i in range(10)]
    train, val = split_train_validation(rows, 0.2)
    assert val is not None
    assert len(train) == 8 and len(val) == 2
    train_ids = {r["task_idx"] for r in train}
    val_ids = {r["task_idx"] for r in val}
    assert not (train_ids & val_ids)

    assert split_train_validation(rows, 0.0) == (rows, None)
    assert split_train_validation([], 0.5) == ([], None)


@pytest.mark.parametrize(("n_rows", "fraction"), [(1, 0.2), (5, 1.0), (10, 1.0)])
def test_split_train_validation_rejects_empty_training_set(n_rows: int, fraction: float) -> None:
    from nmp.rl.tasks.environment.convert import split_train_validation

    rows = [{"task_idx": i} for i in range(n_rows)]
    with pytest.raises(ValueError, match="leaves no training rows"):
        split_train_validation(rows, fraction)


def test_validation_fraction_cli_rejects_out_of_range() -> None:
    import argparse

    from nmp.rl.tasks.environment.__main__ import _validation_fraction

    assert _validation_fraction("0.2") == 0.2
    assert _validation_fraction("0") == 0.0
    for bad in ("1.0", "2.5", "-0.1"):
        with pytest.raises(argparse.ArgumentTypeError):
            _validation_fraction(bad)


def test_wheel_version_prefers_numeric_order_over_lexicographic() -> None:
    from nmp.rl.tasks.environment.convert import _wheel_version

    wheels = [
        Path("ascii_tree-0.9.0-py3-none-any.whl"),
        Path("ascii_tree-0.10.0-py3-none-any.whl"),
        Path("ascii_tree-0.2.0-py3-none-any.whl"),
    ]
    assert sorted(wheels)[-1].name.startswith("ascii_tree-0.9.0")
    assert max(wheels, key=_wheel_version).name.startswith("ascii_tree-0.10.0")


def test_config_paths_containment_rejects_sibling_prefix_dir(tmp_path: Path) -> None:
    """A sibling whose name merely starts with the root's name is not contained."""
    env_root = tmp_path / "environment"
    (env_root / "configs").mkdir(parents=True)
    sibling = tmp_path / "environment-attacker"
    sibling.mkdir()
    (sibling / "evil.yaml").write_text("a: 1", encoding="utf-8")
    # An intermediate symlinked directory escapes the final-component is_symlink() check.
    (env_root / "configs" / "escape").symlink_to(sibling, target_is_directory=True)
    (env_root / "wheels").mkdir()
    (env_root / "wheels" / "x-1.0-py3-none-any.whl").write_bytes(b"x")
    (env_root / "nemo-environment.yaml").write_text(
        "format: adapter-wheels-v1\n"
        "adapter:\n  agent: verifiers_agent\n"
        "config_paths:\n  - configs/escape/evil.yaml\n"
        "metadata:\n  name: e\n",
        encoding="utf-8",
    )
    manifest = load_manifest(env_root)
    with pytest.raises(EnvironmentPackageValidationError, match="escapes environment root"):
        validate_package_layout(env_root, manifest)


def test_download_hub_wheels_resolves_before_downloading(tmp_path: Path, monkeypatch) -> None:
    """The closure must be resolved first, then fetched with --no-deps.

    `pip download` resolves while it fetches and keeps every candidate it pulled,
    including versions it later backtracked away from, so a one-shot download can leave
    two versions of one distribution in wheels/. The cluster-side installer then gets
    contradictory pins and dies with `you require xxhash==3.8.1 and xxhash==4.0.0 ...`
    minutes into the job. Resolving to a pinned file and downloading it with --no-deps is
    what makes the output one-file-per-distribution.
    """
    from nmp.rl.tasks.environment import convert as convert_mod

    commands: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        commands.append(cmd)
        if "compile" in cmd:
            Path(cmd[cmd.index("--output-file") + 1]).write_text("ascii-tree==0.1.5\nxxhash==4.0.0\n", encoding="utf-8")
        else:
            dest = Path(cmd[cmd.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "ascii_tree-0.1.5-py3-none-any.whl").write_bytes(b"PK\x03\x04")
            (dest / "xxhash-4.0.0-cp313-cp313-manylinux_2_17_x86_64.whl").write_bytes(b"PK\x03\x04")
        return None

    monkeypatch.setattr(convert_mod.subprocess, "run", _fake_run)

    spec = convert_mod.ConvertEnvironmentSpec(
        hub_id="primeintellect/ascii-tree",
        out_dir=tmp_path / "env",
    )
    convert_mod.download_hub_wheels(spec, work_dir=tmp_path / "work")

    compile_cmd, download_cmd = commands
    assert compile_cmd[:3] == ["uv", "pip", "compile"]
    # pip merges indexes; uv's default first-index would silently repin.
    assert compile_cmd[compile_cmd.index("--index-strategy") + 1] == "unsafe-best-match"

    assert download_cmd[1:4] == ["-m", "pip", "download"]
    assert "--no-deps" in download_cmd
    pinned = Path(download_cmd[download_cmd.index("-r") + 1])
    assert pinned.read_text(encoding="utf-8").splitlines() == [
        "ascii-tree==0.1.5",
        "xxhash==4.0.0",
    ]


def test_download_hub_wheels_builds_sdists_and_requires_complete_closure(tmp_path: Path, monkeypatch) -> None:
    """Source-only releases must become target-Python wheels; dropping them breaks offline Gym."""
    from nmp.rl.tasks.environment import convert as convert_mod

    commands: list[list[str]] = []
    monkeypatch.setattr(convert_mod.sys, "executable", "/host/python3.12")

    def _fake_run(cmd, **kwargs):
        commands.append(cmd)
        if "compile" in cmd:
            Path(cmd[cmd.index("--output-file") + 1]).write_text(
                "ascii-tree==0.1.5\nverifiers @ git+https://example.test/verifiers.git@v0.1.14\n",
                encoding="utf-8",
            )
        elif "download" in cmd:
            dest = Path(cmd[cmd.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "ascii_tree-0.1.5-py3-none-any.whl").write_bytes(b"PK\x03\x04")
            (dest / "verifiers-0.1.14.zip").write_bytes(b"source")
        elif "wheel" in cmd:
            dest = Path(cmd[cmd.index("--wheel-dir") + 1])
            (dest / "verifiers-0.1.14-py3-none-any.whl").write_bytes(b"PK\x03\x04")
        return None

    monkeypatch.setattr(convert_mod.subprocess, "run", _fake_run)

    wheels = convert_mod.download_hub_wheels(
        convert_mod.ConvertEnvironmentSpec(
            hub_id="primeintellect/ascii-tree",
            out_dir=tmp_path / "env",
        ),
        work_dir=tmp_path / "work",
    )

    wheel_cmd = next(cmd for cmd in commands if "wheel" in cmd)
    assert wheel_cmd[:9] == [
        "uv",
        "run",
        "--no-project",
        "--python",
        convert_mod.TARGET_PYTHON_VERSION,
        "--with",
        "pip",
        "python",
        "-m",
    ]
    assert "/host/python3.12" not in wheel_cmd
    assert "--no-deps" in wheel_cmd
    assert not (wheels / "verifiers-0.1.14.zip").exists()
    assert (wheels / "verifiers-0.1.14-py3-none-any.whl").is_file()


def test_complete_wheel_closure_rejects_missing_distribution(tmp_path: Path) -> None:
    from nmp.rl.tasks.environment import convert as convert_mod

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "ascii_tree-0.1.5-py3-none-any.whl").write_bytes(b"PK\x03\x04")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("ascii-tree==0.1.5\nverifiers==0.1.14\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no wheel was produced for: verifiers"):
        convert_mod._assert_complete_wheel_closure(wheels, requirements)


def test_install_hub_package_does_not_build_sdist_deps(tmp_path: Path, monkeypatch) -> None:
    """Hub install must not rebuild verifiers from a zip sitting next to the wheel."""
    from nmp.rl.tasks.environment import convert as convert_mod

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    whl = wheels / "ascii_tree-0.1.5-py3-none-any.whl"
    whl.write_bytes(b"PK\x03\x04")
    (wheels / "verifiers-0.1.14.zip").write_bytes(b"not-a-wheel")
    captured: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        captured.append(cmd)

    monkeypatch.setattr(convert_mod.subprocess, "run", _fake_run)
    convert_mod._install_hub_package_from_wheels(wheels, "ascii_tree")
    assert captured and "--no-deps" in captured[0]
    assert "--find-links" not in " ".join(captured[0])
    assert str(whl) in captured[0]


def test_download_hub_wheels_requirements_in_lists_env_and_verifiers(tmp_path: Path, monkeypatch) -> None:
    """Both roots have to reach the resolver, or the closure is missing one of them."""
    from nmp.rl.tasks.environment import convert as convert_mod

    seen: dict[str, str] = {}

    def _fake_run(cmd, **kwargs):
        if "compile" in cmd:
            seen["in"] = Path(cmd[3]).read_text(encoding="utf-8")
            Path(cmd[cmd.index("--output-file") + 1]).write_text("", encoding="utf-8")
        else:
            dest = Path(cmd[cmd.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "ascii_tree-0.1.5-py3-none-any.whl").write_bytes(b"PK\x03\x04")
        return None

    monkeypatch.setattr(convert_mod.subprocess, "run", _fake_run)

    convert_mod.download_hub_wheels(
        convert_mod.ConvertEnvironmentSpec(
            hub_id="primeintellect/ascii-tree",
            out_dir=tmp_path / "env",
            extra_wheels=("some-extra",),
        ),
        work_dir=tmp_path / "work",
    )

    assert seen["in"].splitlines() == [
        convert_mod.DEFAULT_VERIFIERS_SPEC,
        "ascii_tree",
        "some-extra",
    ]


def test_duplicate_wheel_distributions_reports_only_repeats(tmp_path: Path) -> None:
    """Normalized per PEP 503, so `charset_normalizer` and `charset-normalizer` are one."""
    from nmp.rl.tasks.environment.validate import duplicate_wheel_distributions

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in (
        "ascii_tree-0.1.5-py3-none-any.whl",
        "xxhash-3.8.1-py3-none-any.whl",
        "xxhash-4.0.0-py3-none-any.whl",
        "charset_normalizer-3.4.9-py3-none-any.whl",
        "charset_normalizer-3.5.0-py3-none-any.whl",
    ):
        (wheels / name).write_bytes(b"PK\x03\x04")

    assert duplicate_wheel_distributions(wheels) == {
        "xxhash": ["3.8.1", "4.0.0"],
        "charset-normalizer": ["3.4.9", "3.5.0"],
    }


def test_validate_package_layout_warns_on_duplicate_wheels(tmp_path: Path, caplog) -> None:
    """A package that does not pin its own closure must say so at validation time."""
    import logging

    from nmp.rl.tasks.environment.package import write_adapter_wheels_package

    src = tmp_path / "prebuilt"
    src.mkdir()
    for name in ("xxhash-3.8.1-py3-none-any.whl", "xxhash-4.0.0-py3-none-any.whl"):
        (src / name).write_bytes(b"PK\x03\x04")

    out = tmp_path / "env"
    with caplog.at_level(logging.WARNING):
        write_adapter_wheels_package(
            out_dir=out,
            hub_id="primeintellect/ascii-tree",
            wheels_src=src,
        )

    assert "2 versions of xxhash" in caplog.text


@pytest.mark.parametrize(
    ("hub_version", "expected"),
    [(None, "ascii_tree"), ("0.1.5", "ascii_tree==0.1.5")],
)
def test_download_hub_wheels_honours_hub_version(
    tmp_path: Path, monkeypatch, hub_version: str | None, expected: str
) -> None:
    """An unpinned hub package is a moving target the conversion cannot control.

    ascii-tree 0.1.6 shipped `Requires-Python: >=3.11,<3.13` where 0.1.5 declared none,
    so an unpinned conversion that worked one day fails the next with
    `requires a different Python`, and a release that still installs may not match the
    training image's interpreter. --hub-version makes the vendored release explicit.
    """
    from nmp.rl.tasks.environment import convert as convert_mod

    seen: dict[str, str] = {}

    def _fake_run(cmd, **kwargs):
        if "compile" in cmd:
            seen["in"] = Path(cmd[3]).read_text(encoding="utf-8")
            Path(cmd[cmd.index("--output-file") + 1]).write_text("", encoding="utf-8")
        else:
            dest = Path(cmd[cmd.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "ascii_tree-0.1.5-py3-none-any.whl").write_bytes(b"PK\x03\x04")
        return None

    monkeypatch.setattr(convert_mod.subprocess, "run", _fake_run)

    convert_mod.download_hub_wheels(
        convert_mod.ConvertEnvironmentSpec(
            hub_id="primeintellect/ascii-tree",
            hub_version=hub_version,
            out_dir=tmp_path / "env",
        ),
        work_dir=tmp_path / "work",
    )

    assert seen["in"].splitlines()[1] == expected


def test_convert_cli_threads_hub_version_into_the_spec(monkeypatch, tmp_path: Path) -> None:
    """The flag is useless if it stops at argparse."""
    from nmp.rl.tasks.environment import __main__ as cli

    captured: dict = {}

    def _fake_convert(spec):
        captured["spec"] = spec
        raise SystemExit(0)

    monkeypatch.setattr(cli, "convert_prime_environment", _fake_convert)

    with pytest.raises(SystemExit):
        cli.main(
            [
                "--hub-id",
                "primeintellect/ascii-tree",
                "--hub-version",
                "0.1.5",
                "--out-dir",
                str(tmp_path / "env"),
            ]
        )

    assert captured["spec"].hub_version == "0.1.5"


def test_write_adapter_package_clears_stale_wheels(tmp_path: Path) -> None:
    """Re-running into the same out_dir must replace the closure, not union with it.

    The copy overwrites by filename, so a wheel from an earlier run survives whenever the
    new resolution picked a different version of that project — and the package then ships
    two. It validates fine locally and only fails on the cluster, as
    `you require uvicorn==0.52.1 and uvicorn==0.52.3 ... unsatisfiable`.
    """
    out = tmp_path / "env"
    stale = out / "wheels"
    stale.mkdir(parents=True)
    (stale / "uvicorn-0.52.1-py3-none-any.whl").write_bytes(b"PK\x03\x04")
    (stale / "ascii_tree-0.1.5-py3-none-any.whl").write_bytes(b"OLD")

    src = tmp_path / "fresh"
    src.mkdir()
    (src / "uvicorn-0.52.3-py3-none-any.whl").write_bytes(b"PK\x03\x04")
    (src / "ascii_tree-0.1.5-py3-none-any.whl").write_bytes(b"NEW")

    write_adapter_wheels_package(
        out_dir=out,
        hub_id="primeintellect/ascii-tree",
        wheels_src=src,
    )

    assert sorted(p.name for p in (out / "wheels").glob("*.whl")) == [
        "ascii_tree-0.1.5-py3-none-any.whl",
        "uvicorn-0.52.3-py3-none-any.whl",
    ]
    # Same-named wheels must still be refreshed, not merely left in place.
    assert (out / "wheels" / "ascii_tree-0.1.5-py3-none-any.whl").read_bytes() == b"NEW"


def test_download_targets_the_training_image_not_the_host(tmp_path: Path, monkeypatch) -> None:
    """Resolve and download for linux x86_64 / py3.13, whatever the conversion host is.

    A macOS host otherwise vendors arm64 wheels and the cluster install dies with
    `cffi==2.1.1 has no wheels with a matching platform tag (e.g. manylinux_2_39_x86_64)`.
    Markers are evaluated at resolve time too, so the compile has to target the same
    platform or the closure is wrong before anything is fetched.
    """
    from nmp.rl.tasks.environment import convert as convert_mod

    commands: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        commands.append(cmd)
        if "compile" in cmd:
            Path(cmd[cmd.index("--output-file") + 1]).write_text("cffi==2.1.1\n", encoding="utf-8")
        else:
            dest = Path(cmd[cmd.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "cffi-2.1.1-cp313-cp313-manylinux_2_17_x86_64.whl").write_bytes(b"PK\x03\x04")
        return None

    monkeypatch.setattr(convert_mod.subprocess, "run", _fake_run)

    convert_mod.download_hub_wheels(
        convert_mod.ConvertEnvironmentSpec(hub_id="primeintellect/ascii-tree", out_dir=tmp_path / "env"),
        work_dir=tmp_path / "work",
    )

    compile_cmd, download_cmd = commands
    assert compile_cmd[compile_cmd.index("--python-platform") + 1] == convert_mod.TARGET_UV_PLATFORM
    assert compile_cmd[compile_cmd.index("--python-version") + 1] == convert_mod.TARGET_PYTHON_VERSION
    # Resolving for the host would silently reintroduce the bug.
    assert "--python" not in compile_cmd

    assert download_cmd[download_cmd.index("--python-version") + 1] == convert_mod.TARGET_PYTHON_VERSION
    requested = [download_cmd[i + 1] for i, a in enumerate(download_cmd) if a == "--platform"]
    assert requested == list(convert_mod.TARGET_WHEEL_PLATFORMS)


@pytest.mark.parametrize(
    "filename",
    [
        "cffi-2.1.1-cp313-cp313-macosx_11_0_arm64.whl",
        "charset_normalizer-3.5.0-cp313-cp313-macosx_10_13_universal2.whl",
        "foo-1.0-cp313-cp313-win_amd64.whl",
        "host_built-1.0-cp312-cp312-manylinux_2_17_x86_64.whl",
        "future-1.0-cp314-cp314-manylinux_2_17_x86_64.whl",
    ],
)
def test_assert_wheels_target_platform_rejects_foreign_wheels(tmp_path: Path, filename: str) -> None:
    """Catch foreign platforms and Python ABIs before the cluster install."""
    from nmp.rl.tasks.environment.convert import assert_wheels_target_platform

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / filename).write_bytes(b"PK\x03\x04")

    with pytest.raises(RuntimeError, match="not installable on the training image"):
        assert_wheels_target_platform(wheels)


@pytest.mark.parametrize(
    "filename",
    [
        "ascii_tree-0.1.5-py3-none-any.whl",
        "verifiers-0.1.14-py2.py3-none-any.whl",
        "cffi-2.1.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "numpy-2.5.2-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "cryptography-50.0.0-cp311-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "old-1.0-cp313-cp313-linux_x86_64.whl",
    ],
)
def test_assert_wheels_target_platform_accepts_portable_wheels(tmp_path: Path, filename: str) -> None:
    """Pure-Python and any linux x86_64 build installs on the training image.

    Compound tags count if ANY component matches: pip emits e.g.
    `manylinux2014_x86_64.manylinux_2_17_x86_64` for one file.
    """
    from nmp.rl.tasks.environment.convert import assert_wheels_target_platform

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / filename).write_bytes(b"PK\x03\x04")

    assert_wheels_target_platform(wheels)


def test_compile_ignores_this_repo_dependency_policy(tmp_path: Path, monkeypatch) -> None:
    """The closure is a user's environment, not a nemo-platform dependency set.

    uv discovers the nearest pyproject.toml and applies its [tool.uv] policy. This repo's
    override-dependencies exist for its own CVE posture, and an override REPLACES a
    declared requirement rather than narrowing it — `openai>=2.26.0` strips the `<3` upper
    bound that openai-agents declares. The closure then vendors openai 3.0.0, and the
    cluster, where no overrides apply, cannot install it.
    """
    from nmp.rl.tasks.environment import convert as convert_mod

    commands: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        commands.append(cmd)
        if "compile" in cmd:
            Path(cmd[cmd.index("--output-file") + 1]).write_text("openai==2.54.0\n", encoding="utf-8")
        else:
            dest = Path(cmd[cmd.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "openai-2.54.0-py3-none-any.whl").write_bytes(b"PK\x03\x04")
        return None

    monkeypatch.setattr(convert_mod.subprocess, "run", _fake_run)

    convert_mod.download_hub_wheels(
        convert_mod.ConvertEnvironmentSpec(hub_id="primeintellect/ascii-tree", out_dir=tmp_path / "env"),
        work_dir=tmp_path / "work",
    )

    assert "--no-config" in commands[0]
