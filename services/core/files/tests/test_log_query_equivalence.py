# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equivalence tests: the two-phase log queries must match the original window-function queries."""

import random
from datetime import datetime, timedelta

import pytest
from nhx.common.files.storage_config import LocalStorageConfig
from nhx.common.jobs.schemas import HelixJobLogPage, LogPageCursorV1, decode_log_page_cursor
from nhx.core.files.app.backends.local import LocalStorageImpl
from nhx.core.files.app.log_db import DuckDBLogRepository, DuckDBQueryParam, LogEntry, LogQuery

Repo = DuckDBLogRepository


@pytest.fixture
def local_storage(tmp_path):
    return LocalStorageImpl(config=LocalStorageConfig(path=str(tmp_path)))


def _legacy_tail_query(base_table: str, where_clause: str) -> str:
    return f"""
        SELECT {Repo.SELECT_LOG_COLUMNS_SQL}, {Repo.ROW_HASH_SQL} AS row_hash, COUNT(*) OVER() as _total_count
        FROM {base_table}
        WHERE {where_clause}
        ORDER BY timestamp DESC, row_hash DESC
        LIMIT ?
    """


def _legacy_cursor_query(base_table: str, where_clause: str) -> str:
    return f"""
        WITH filtered AS (
            SELECT {Repo.SELECT_LOG_COLUMNS_SQL}, {Repo.ROW_HASH_SQL} AS row_hash, COUNT(*) OVER() as _total_count
            FROM {base_table}
            WHERE {where_clause}
        ),
        eligible AS (
            SELECT * FROM filtered WHERE timestamp < ? OR (timestamp = ? AND row_hash <= ?)
        )
        SELECT * FROM eligible ORDER BY timestamp DESC, row_hash DESC LIMIT ?
    """


def _legacy_page(
    query: LogQuery, storage: LocalStorageImpl, cursor: LogPageCursorV1 | None, page_size: int
) -> HelixJobLogPage:
    """Original tail/cursor_v1 implementation (window COUNT, hash over every row)."""
    conn = Repo._connect(storage)
    try:
        base_table, where_clause, params = Repo._prepare_query(query.base_path, query.filters)
        if cursor is None:
            sql = _legacy_tail_query(base_table, where_clause)
            query_params: list[DuckDBQueryParam] = [*params, page_size + 1]
        else:
            sql = _legacy_cursor_query(base_table, where_clause)
            query_params = [
                *params,
                cursor.boundary_timestamp,
                cursor.boundary_timestamp,
                cursor.boundary_row_hash,
                page_size + cursor.emitted_boundary_rows + 1,
            ]
        rows = Repo._fetch_queried_logs(conn.execute(sql, query_params))
        total = rows[0].total_count if rows else 0
        usable = Repo._drop_emitted_boundary_rows(rows, cursor) if cursor else rows
        returned = list(reversed(usable[:page_size]))
        prev_page = None
        if len(usable) > page_size and returned:
            prev_page = Repo._cursor_v1_from_boundary(
                returned[0], query.filters, None, Repo._emitted_count_for_boundary(returned, prior_cursor=cursor)
            )
        return HelixJobLogPage(data=Repo._platform_logs(returned), total=total, next_page=None, prev_page=prev_page)
    finally:
        conn.close()


def _insert(storage: LocalStorageImpl, base_path: str, n_rows: int, distinct_ts: int, batch: int, seed: int) -> None:
    rng = random.Random(seed)
    base = datetime(2026, 1, 1, 12, 0, 0)
    entries = [
        LogEntry(
            workspace="default",
            job="j",
            job_attempt="1",
            job_step=f"s{rng.randrange(3)}",
            job_task=f"t{rng.randrange(2)}",
            # Few distinct timestamps force heavy ties, including exact duplicate rows.
            log_message=f"msg-{rng.randrange(n_rows // 4 + 1)}",
            timestamp=base + timedelta(seconds=rng.randrange(distinct_ts)),
        )
        for _ in range(n_rows)
    ]
    # Rows in another attempt must never leak into the filtered results or the total.
    other = [e.model_copy(update={"job_attempt": "2"}) for e in entries[:20]]
    for start in range(0, n_rows, batch):
        Repo.insert_logs(entries[start : start + batch], base_path, storage)
    Repo.insert_logs(other, base_path, storage)


def _query(base_path: str, filters: dict[str, str], **kwargs) -> LogQuery:
    return LogQuery(base_path=base_path, filters=filters, page_size=kwargs.pop("page_size", 10), **kwargs)


def _signature(page: HelixJobLogPage) -> tuple:
    return (
        [(log.timestamp, log.job_step, log.job_task, log.message) for log in page.data],
        page.total,
        page.prev_page,
        page.next_page,
    )


SCENARIOS = [
    # (n_rows, distinct timestamps, rows per insert batch)
    pytest.param(200, 8, 200, id="heavy-ties-one-file"),
    pytest.param(300, 40, 25, id="ties-many-files"),
    pytest.param(150, 150, 30, id="mostly-unique"),
    pytest.param(60, 1, 20, id="all-same-timestamp"),
]


@pytest.mark.parametrize("n_rows,distinct_ts,batch", SCENARIOS)
@pytest.mark.parametrize("tail", [1, 7, 50, 1000])
@pytest.mark.parametrize(
    "filters",
    [
        {"job": "j", "job_attempt": "1"},
        {"job": "j", "job_attempt": "1", "job_step": "s1"},
        {"job": "j", "job_attempt": "1", "job_step": "s1", "job_task": "t0"},
        # Non-path filter: total must come from a real filtered count, not footers.
        {"job": "j", "job_attempt": "1", "job_task": "t1"},
    ],
)
def test_tail_matches_legacy(local_storage, n_rows, distinct_ts, batch, tail, filters):
    base_path = local_storage.get_duckdb_path("logs")
    _insert(local_storage, base_path, n_rows, distinct_ts, batch, seed=n_rows)
    query = _query(base_path, filters, tail=tail, page_size=tail)

    new = Repo().query_logs(query, local_storage)
    old = _legacy_page(query, local_storage, None, tail)

    assert _signature(new) == _signature(old)


@pytest.mark.parametrize("n_rows,distinct_ts,batch", SCENARIOS)
@pytest.mark.parametrize("page_size", [1, 3, 17, 64])
@pytest.mark.parametrize(
    "filters",
    [
        {"job": "j", "job_attempt": "1"},
        {"job": "j", "job_attempt": "1", "job_task": "t1"},
    ],
)
def test_prev_page_chain_matches_legacy(local_storage, n_rows, distinct_ts, batch, page_size, filters):
    """Tail followed by the full prev_page chain yields identical pages (and cursors) to the old queries."""
    base_path = local_storage.get_duckdb_path("logs")
    _insert(local_storage, base_path, n_rows, distinct_ts, batch, seed=n_rows + 1)

    new = Repo().query_logs(_query(base_path, filters, tail=page_size, page_size=page_size), local_storage)
    old = _legacy_page(_query(base_path, filters, tail=page_size), local_storage, None, page_size)
    new_pages, old_pages = [new], [old]

    for _ in range(10_000):
        assert _signature(new) == _signature(old)
        if new.prev_page is None:
            break
        cursor = decode_log_page_cursor(new.prev_page)
        assert isinstance(cursor, LogPageCursorV1)
        new = Repo().query_logs(
            _query(base_path, filters, cursor=cursor, page_size=page_size),
            local_storage,
        )
        old = _legacy_page(_query(base_path, filters, cursor=cursor), local_storage, cursor, page_size)
        new_pages.append(new)
        old_pages.append(old)
    else:
        pytest.fail("pagination did not terminate")

    new_sequence = [
        (log.timestamp, log.job_step, log.job_task, log.message) for p in reversed(new_pages) for log in p.data
    ]
    old_sequence = [
        (log.timestamp, log.job_step, log.job_task, log.message) for p in reversed(old_pages) for log in p.data
    ]
    assert new_sequence == old_sequence
    # The chain walks the whole filtered log exactly once.
    assert len(new_sequence) == new_pages[0].total


def test_total_counts_only_filtered_partitions(local_storage):
    base_path = local_storage.get_duckdb_path("logs")
    _insert(local_storage, base_path, 100, 10, 10, seed=3)

    page = Repo().query_logs(_query(base_path, {"job": "j", "job_attempt": "1"}, tail=5, page_size=5), local_storage)
    assert page.total == 100
    other = Repo().query_logs(_query(base_path, {"job": "j", "job_attempt": "2"}, tail=5, page_size=5), local_storage)
    assert other.total == 20


@pytest.mark.parametrize("direction_page", [1, 2, 5])
def test_offset_total_matches_window_count(local_storage, direction_page):
    from nhx.common.jobs.schemas import PaginationDirection

    base_path = local_storage.get_duckdb_path("logs")
    _insert(local_storage, base_path, 120, 30, 20, seed=5)
    page = Repo._query_offset_page(
        base_path=base_path,
        filters={"job": "j", "job_attempt": "1"},
        page_size=25,
        current_page=direction_page,
        direction=PaginationDirection.FORWARD,
        storage=local_storage,
    )
    assert page.total == 120
    assert len(page.data) == min(25, 120 - (direction_page - 1) * 25)
