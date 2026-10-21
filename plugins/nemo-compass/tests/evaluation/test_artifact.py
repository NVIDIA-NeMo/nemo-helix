# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re

from evaluation import artifact


def test_pre_rename_export_kind_remains_readable():
    assert artifact.is_export_manifest({"kind": artifact.LEGACY_EXPORT_KIND})


def test_pick_records_scoped_to_subjects(tmp_path):
    """Only the selected subjects' run records go into a bundle, not everything in tmp/."""
    for name in ("a.run.json", "b.run.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "noise.txt").write_text("x")
    assert [p.name for p in artifact.pick_records(tmp_path, ["a"])] == ["a.run.json"]


def test_pick_records_excludes_insights_yaml(tmp_path):
    """Insights never travel in bundles: a subject's insights YAML is not a bundle record."""
    (tmp_path / "a.run.json").write_text("{}")
    (tmp_path / "insights_a.yaml").write_text("insights: []")
    assert [p.name for p in artifact.pick_records(tmp_path, ["a"])] == ["a.run.json"]


def test_pick_records_tolerates_missing_files(tmp_path):
    (tmp_path / "a.run.json").write_text("{}")
    assert [p.name for p in artifact.pick_records(tmp_path, ["a", "ghost"])] == ["a.run.json"]


def test_backup_records_moves_named_files(tmp_path):
    (tmp_path / "a.run.json").write_text("local")
    backup = artifact.backup_records(tmp_path, ["a.run.json"])
    assert backup.parent == tmp_path and backup.name.startswith("backup-")
    # microsecond-stamped: two invocations within the same second must not collide
    assert re.fullmatch(r"backup-\d{8}-\d{6}-\d{6}", backup.name)
    assert not (tmp_path / "a.run.json").exists()
    assert (backup / "a.run.json").read_text() == "local"


def test_seed_records_copies_run_records(tmp_path):
    src = tmp_path / "state" / "tmp"
    src.mkdir(parents=True)
    (src / "a.run.json").write_text("{}")
    dest = tmp_path / "dest"
    msg, seeded = artifact.seed_records(src, dest)
    assert (dest / "a.run.json").read_text() == "{}"
    assert seeded == ["a.run.json"]
    assert "a.run.json" in msg


def test_seed_records_ignores_bundled_insights(tmp_path):
    """A stray insights YAML inside an old bundle's tmp/ is never copied out."""
    src = tmp_path / "state" / "tmp"
    src.mkdir(parents=True)
    (src / "a.run.json").write_text("{}")
    (src / "insights_a.yaml").write_text("insights: []")
    dest = tmp_path / "dest"
    msg, seeded = artifact.seed_records(src, dest)
    assert (dest / "a.run.json").exists() and not (dest / "insights_a.yaml").exists()
    assert seeded == ["a.run.json"]  # the ignored insight YAML is not "seeded"


def test_seed_records_leaves_local_insights_alone(tmp_path):
    """Local insight YAMLs are Compass's business (cli fresh/keep), never the restore's."""
    src = tmp_path / "state" / "tmp"
    src.mkdir(parents=True)
    (src / "a.run.json").write_text("{}")
    (src / "insights_a.yaml").write_text("insights: [bundle]")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "insights_a.yaml").write_text("insights: [local]")
    (dest / "insights_b.yaml").write_text("insights: [other]")
    artifact.seed_records(src, dest)
    assert (dest / "insights_a.yaml").read_text() == "insights: [local]"
    assert (dest / "insights_b.yaml").read_text() == "insights: [other]"
    assert list(dest.glob("backup-*")) == []  # no run record clobbered -> nothing moved


def test_seed_records_missing_state_tmp_ok(tmp_path):
    msg, seeded = artifact.seed_records(tmp_path / "absent", tmp_path / "dest")
    assert "no run record" in msg
    assert seeded == []


def test_seed_records_backs_up_clobbered_local_records(tmp_path, capsys):
    """A restore never silently destroys local records — originals move to backup-<stamp>/."""
    src = tmp_path / "state" / "tmp"
    src.mkdir(parents=True)
    (src / "a.run.json").write_text('{"from": "bundle"}')
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "a.run.json").write_text('{"from": "local"}')
    artifact.seed_records(src, dest)
    assert (dest / "a.run.json").read_text() == '{"from": "bundle"}'
    (backup,) = dest.glob("backup-*")
    assert (backup / "a.run.json").read_text() == '{"from": "local"}'
    out = capsys.readouterr().out
    assert "backed up" in out and "a.run.json" in out


def test_seed_records_no_backup_when_nothing_clobbered(tmp_path, capsys):
    src = tmp_path / "state" / "tmp"
    src.mkdir(parents=True)
    (src / "a.run.json").write_text("{}")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "other.run.json").write_text("{}")  # name not in the bundle: untouched
    artifact.seed_records(src, dest)
    assert list(dest.glob("backup-*")) == []
    assert "backed up" not in capsys.readouterr().out


# Manifest building lives on the export path — see tests/evaluation/test_export.py
# (build_export_manifest, snapshot_export).
