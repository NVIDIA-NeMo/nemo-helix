# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import yaml
from evaluation.summary import render_summary_md


def _write(tmp_path, insights):
    path = tmp_path / "insights_x.yaml"
    path.write_text(yaml.safe_dump({"insights": insights}))
    return path


def test_render_lists_each_insight(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "ins-1",
                "title": "Agent skips policy lookup",
                "status": "proposed",
                "description": "First line.\nSecond line.",
                "trace_refs": ["a", "b", "c"],
            },
            {"id": "ins-2", "title": "Refund loop", "status": "accepted", "trace_refs": []},
        ],
    )
    md = render_summary_md(path, "tau2-airline")
    assert "## Insights — tau2-airline (2)" in md
    assert "**Agent skips policy lookup**" in md
    assert "proposed" in md
    assert "3 trace refs" in md
    assert "First line." in md
    assert "Second line." not in md  # only the first description line


def test_render_missing_or_empty_file(tmp_path):
    md = render_summary_md(tmp_path / "absent.yaml", "nvq")
    assert "## Insights — nvq (0)" in md
    assert "No insights" in md
