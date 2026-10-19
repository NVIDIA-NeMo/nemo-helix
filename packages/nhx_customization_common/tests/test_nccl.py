# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from nhx.customization_common.training import nccl as nccl_module
from nhx.customization_common.training.nccl import get_nccl_ib_env


def _make_hca(ib_root: Path, name: str, with_netdev: bool = False) -> None:
    net = ib_root / name / "device" / "net"
    net.mkdir(parents=True)
    if with_netdev:
        (net / "eth0").mkdir()


@pytest.fixture
def ib_sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ib = tmp_path / "infiniband"
    ib.mkdir()
    monkeypatch.setattr(nccl_module, "_IB_SYSFS", ib)
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.delenv("NCCL_IB_DISABLE", raising=False)
    return ib


def test_returns_exact_usable_hcas_when_phantoms_exist(ib_sysfs: Path) -> None:
    _make_hca(ib_sysfs, "mlx5_1", with_netdev=True)
    _make_hca(ib_sysfs, "mlx5_10", with_netdev=True)
    _make_hca(ib_sysfs, "mlx5_2")

    assert get_nccl_ib_env() == {"NCCL_IB_HCA": "=mlx5_1,=mlx5_10"}
    assert "NCCL_IB_HCA" not in os.environ


def test_disables_ib_when_all_hcas_are_phantom(ib_sysfs: Path) -> None:
    _make_hca(ib_sysfs, "mlx5_0")
    _make_hca(ib_sysfs, "mlx5_1")

    assert get_nccl_ib_env() == {"NCCL_IB_DISABLE": "1"}


def test_returns_no_overrides_without_phantoms(ib_sysfs: Path) -> None:
    _make_hca(ib_sysfs, "mlx5_0", with_netdev=True)

    assert get_nccl_ib_env() == {}


def test_respects_existing_nccl_configuration(ib_sysfs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_hca(ib_sysfs, "mlx5_0")
    monkeypatch.setenv("NCCL_IB_HCA", "mlx5_custom")

    assert get_nccl_ib_env() == {}
