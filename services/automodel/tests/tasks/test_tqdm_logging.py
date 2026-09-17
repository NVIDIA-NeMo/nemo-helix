# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import pytest
from nmp.automodel.tasks.tqdm_logging import LineTqdm, install_line_tqdm


def test_line_tqdm_writes_newlines_not_carriage_returns() -> None:
    buf = io.StringIO()
    bar = LineTqdm(total=10, file=buf, mininterval=0)
    bar.update(4)
    bar.display()
    text = buf.getvalue()
    assert "\n" in text
    assert "\r" not in text
    bar.close()


def test_install_line_tqdm_replaces_module_class(monkeypatch: pytest.MonkeyPatch) -> None:
    import tqdm

    # Register the original class with monkeypatch so the global install is
    # undone at test teardown.
    monkeypatch.setattr(tqdm, "tqdm", tqdm.tqdm)
    install_line_tqdm()
    assert tqdm.tqdm is LineTqdm


def test_training_entry_point_installs_line_tqdm() -> None:
    """Training draws tqdm bars too; mining was the only caller for a while."""
    pytest.importorskip("torch", reason="not installed in test env (run in training image CI)")
    pytest.importorskip("nemo_automodel", reason="not installed in test env (run in training image CI)")
    from nmp.automodel.tasks.training.backends import finetune

    assert finetune.install_line_tqdm is install_line_tqdm
