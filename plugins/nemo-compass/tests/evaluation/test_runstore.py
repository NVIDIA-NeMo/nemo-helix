# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import MappingProxyType

from evaluation.runstore import load_run, save_run


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "tau2-airline.run.json"
    record = {
        "agent": "tau2-airline-x",
        "workspace": "w",
        "base_url": "u",
        "domain": "airline",
    }
    save_run(path, record)
    assert load_run(path) == record


def test_load_missing_returns_none(tmp_path):
    assert load_run(tmp_path / "nope.run.json") is None


def test_save_creates_parent_dir(tmp_path):
    path = tmp_path / "deep" / "nested" / "x.run.json"
    save_run(path, {"k": 1})
    assert path.exists()


def test_save_accepts_abstract_mapping(tmp_path):
    path = tmp_path / "proxy.run.json"
    save_run(path, MappingProxyType({"k": 1}))
    assert load_run(path) == {"k": 1}
