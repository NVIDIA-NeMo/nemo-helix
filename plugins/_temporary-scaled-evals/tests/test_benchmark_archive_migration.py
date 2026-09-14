# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove archive migration upgrades/replays and queue fencing on real Postgres."""

import os
import shutil
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("scaled_evals")

import psycopg
from nemo_scaled_evals_plugin import migrations
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.repositories.benchmark_archive_repository import BenchmarkArchiveRepository
from test_migrations import TEST_DSN_ENV, _dsn_for


def _archive(repo: BenchmarkArchiveRepository) -> dict:
    row = repo.get("bmr_1")
    assert row is not None
    return row


@pytest.mark.skipif(not os.environ.get(TEST_DSN_ENV), reason=f"{TEST_DSN_ENV} not set")
def test_archive_upgrade_replay_and_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admin_dsn = os.environ[TEST_DSN_ENV]
    scratch = f"se_archive_test_{uuid.uuid4().hex[:12]}"
    dsn = _dsn_for(urlunsplit(urlsplit(admin_dsn)._replace(path=f"/{scratch}")), "scaled_evals")
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(SQL("CREATE DATABASE {}").format(Identifier(scratch)))
    try:
        root = migrations.sql_root()
        baseline = tmp_path / "baseline"
        shutil.copytree(root, baseline)
        (baseline / "migrations" / "042_benchmark_run_archives.sql").unlink()
        with monkeypatch.context() as patch:
            patch.setattr(migrations, "sql_root", lambda: baseline)
            migrations.apply_sql(dsn, schema="scaled_evals")
        with psycopg.connect(dsn, autocommit=True) as seed:
            seed.execute("INSERT INTO benchmarks (id, name, slug) VALUES ('bm_1', 'Benchmark', 'benchmark')")
            seed.execute("INSERT INTO benchmark_revisions (benchmark_id, revision) VALUES ('bm_1', 1)")
            seed.execute(
                "INSERT INTO benchmark_runs (id, name, benchmark_id, benchmark_revision) VALUES ('bmr_1', 'Run', 'bm_1', 1)"
            )
            seed.execute("INSERT INTO tasks (id, name, slug) VALUES ('task_1', 'Task', 'task')")
            seed.execute(
                "INSERT INTO task_revisions (task_id, revision, tarball_object_key) VALUES ('task_1', 1, 'pack')"
            )
            seed.execute("""INSERT INTO evaluations
                (id, name, task_id, task_revision, benchmark_run_id, status,
                 archive_status, archive_object_key, archive_built_at, archive_size_bytes, evidence_status)
                VALUES ('ev_1', 'Eval', 'task_1', 1, 'bmr_1', 'succeeded', 'ready', 'archive', NOW(), 42, 'ready')""")
        assert migrations.apply_sql(dsn, schema="scaled_evals")[0] == 0
        with psycopg.Connection[dict[str, Any]].connect(dsn, autocommit=True, row_factory=dict_row) as conn:
            repo = BenchmarkArchiveRepository(conn)
            queued = repo.request("bmr_1")
            assert queued["status"] == "queued"
            assert len(queued["members"]) == 1
            public_table = conn.execute("SELECT to_regclass('public.benchmark_run_archives') AS name").fetchone()
            assert public_table is not None and public_table["name"] is None
            # A boot replay preserves durable queued work.
            migrations.apply_sql(dsn, schema="scaled_evals")
            assert _archive(repo)["generation"] == queued["generation"]
            claim = repo.claim(claim_timeout=30)
            assert claim is not None and claim["status"] == "building"
            assert repo.claim(claim_timeout=30) is None
            assert repo.heartbeat("bmr_1", claim["claim_token"])
            stale = {**claim, "claim_token": "stale"}
            assert not repo.heartbeat("bmr_1", "stale")
            repo.fail(stale, "must not overwrite")
            repo.finish(stale, object_key="stale", size_bytes=1)
            assert _archive(repo)["status"] == "building"
            repo.finish(claim, object_key="combined", size_bytes=100)
            assert _archive(repo)["object_key"] == "combined"
            assert repo.request("bmr_1")["generation"] == queued["generation"]
            rebuilt = repo.request("bmr_1", force=True)
            assert rebuilt["generation"] != queued["generation"]
            claim = repo.claim(claim_timeout=30)
            assert claim is not None
            conn.execute("UPDATE evaluations SET current_execution = 2 WHERE id = 'ev_1'")
            with pytest.raises(Conflict, match="members_changed"):
                repo.finish(claim, object_key="changed", size_bytes=1)
            assert _archive(repo)["object_key"] is None
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(Identifier(scratch)))
