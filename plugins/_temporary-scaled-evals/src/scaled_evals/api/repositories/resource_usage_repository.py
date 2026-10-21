# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import psycopg

from scaled_evals.models.resource_usage import ResourceUsageSample


class ResourceUsageRepository:
    """Persist bounded per-execution aggregates from optional runtime samplers."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def record_samples(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        samples: list[ResourceUsageSample],
    ) -> None:
        if not samples:
            return
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO evaluation_resource_usage (
                    evaluation_id, execution_number, component, source,
                    collection_status, collection_error,
                    sample_count, first_observed_at, last_observed_at,
                    cpu_sample_count, cpu_usage_cores_sum, cpu_usage_cores_max,
                    memory_sample_count, memory_usage_bytes_sum, memory_usage_bytes_max,
                    cpu_request_cores, cpu_limit_cores,
                    memory_request_bytes, memory_limit_bytes, gpu_request,
                    gpu_sample_count, gpu_usage_percent_sum, gpu_usage_percent_max,
                    gpu_memory_sample_count, gpu_memory_usage_bytes_sum,
                    gpu_memory_usage_bytes_max
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, 1, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (evaluation_id, execution_number, component) DO UPDATE SET
                    source = EXCLUDED.source,
                    collection_status = EXCLUDED.collection_status,
                    collection_error = EXCLUDED.collection_error,
                    sample_count = evaluation_resource_usage.sample_count + 1,
                    first_observed_at = LEAST(
                        evaluation_resource_usage.first_observed_at,
                        EXCLUDED.first_observed_at
                    ),
                    last_observed_at = GREATEST(
                        evaluation_resource_usage.last_observed_at,
                        EXCLUDED.last_observed_at
                    ),
                    cpu_sample_count = evaluation_resource_usage.cpu_sample_count
                        + EXCLUDED.cpu_sample_count,
                    cpu_usage_cores_sum = evaluation_resource_usage.cpu_usage_cores_sum
                        + EXCLUDED.cpu_usage_cores_sum,
                    cpu_usage_cores_max = GREATEST(
                        evaluation_resource_usage.cpu_usage_cores_max,
                        EXCLUDED.cpu_usage_cores_max
                    ),
                    memory_sample_count = evaluation_resource_usage.memory_sample_count
                        + EXCLUDED.memory_sample_count,
                    memory_usage_bytes_sum = evaluation_resource_usage.memory_usage_bytes_sum
                        + EXCLUDED.memory_usage_bytes_sum,
                    memory_usage_bytes_max = GREATEST(
                        evaluation_resource_usage.memory_usage_bytes_max,
                        EXCLUDED.memory_usage_bytes_max
                    ),
                    cpu_request_cores = COALESCE(
                        EXCLUDED.cpu_request_cores,
                        evaluation_resource_usage.cpu_request_cores
                    ),
                    cpu_limit_cores = COALESCE(
                        EXCLUDED.cpu_limit_cores,
                        evaluation_resource_usage.cpu_limit_cores
                    ),
                    memory_request_bytes = COALESCE(
                        EXCLUDED.memory_request_bytes,
                        evaluation_resource_usage.memory_request_bytes
                    ),
                    memory_limit_bytes = COALESCE(
                        EXCLUDED.memory_limit_bytes,
                        evaluation_resource_usage.memory_limit_bytes
                    ),
                    gpu_request = COALESCE(
                        EXCLUDED.gpu_request,
                        evaluation_resource_usage.gpu_request
                    ),
                    gpu_sample_count = evaluation_resource_usage.gpu_sample_count
                        + EXCLUDED.gpu_sample_count,
                    gpu_usage_percent_sum = evaluation_resource_usage.gpu_usage_percent_sum
                        + EXCLUDED.gpu_usage_percent_sum,
                    gpu_usage_percent_max = GREATEST(
                        evaluation_resource_usage.gpu_usage_percent_max,
                        EXCLUDED.gpu_usage_percent_max
                    ),
                    gpu_memory_sample_count = evaluation_resource_usage.gpu_memory_sample_count
                        + EXCLUDED.gpu_memory_sample_count,
                    gpu_memory_usage_bytes_sum =
                        evaluation_resource_usage.gpu_memory_usage_bytes_sum
                        + EXCLUDED.gpu_memory_usage_bytes_sum,
                    gpu_memory_usage_bytes_max = GREATEST(
                        evaluation_resource_usage.gpu_memory_usage_bytes_max,
                        EXCLUDED.gpu_memory_usage_bytes_max
                    )
                """,
                [
                    (
                        evaluation_id,
                        execution_number,
                        sample.component,
                        sample.source,
                        sample.collection_status,
                        sample.collection_error,
                        sample.observed_at,
                        sample.observed_at,
                        int(sample.cpu_usage_cores is not None),
                        sample.cpu_usage_cores or 0,
                        sample.cpu_usage_cores or 0,
                        int(sample.memory_usage_bytes is not None),
                        sample.memory_usage_bytes or 0,
                        sample.memory_usage_bytes or 0,
                        sample.cpu_request_cores,
                        sample.cpu_limit_cores,
                        sample.memory_request_bytes,
                        sample.memory_limit_bytes,
                        sample.gpu_request,
                        int(sample.gpu_usage_percent is not None),
                        sample.gpu_usage_percent or 0,
                        sample.gpu_usage_percent or 0,
                        int(sample.gpu_memory_usage_bytes is not None),
                        sample.gpu_memory_usage_bytes or 0,
                        sample.gpu_memory_usage_bytes or 0,
                    )
                    for sample in samples
                ],
            )

    def list_for_evaluation(self, evaluation_id: str) -> list[dict]:
        """Return bounded, attempt-aware aggregates for the public telemetry API."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    execution_number, component, source, collection_status,
                    collection_error, sample_count,
                    first_observed_at, last_observed_at,
                    cpu_sample_count,
                    cpu_usage_cores_sum / NULLIF(cpu_sample_count, 0) AS avg_cpu_cores,
                    CASE WHEN cpu_sample_count > 0 THEN cpu_usage_cores_max END
                        AS peak_cpu_cores,
                    memory_sample_count,
                    memory_usage_bytes_sum / NULLIF(memory_sample_count, 0)
                        AS avg_memory_bytes,
                    CASE WHEN memory_sample_count > 0 THEN memory_usage_bytes_max END
                        AS peak_memory_bytes,
                    cpu_request_cores, cpu_limit_cores,
                    memory_request_bytes, memory_limit_bytes, gpu_request,
                    gpu_sample_count,
                    gpu_usage_percent_sum / NULLIF(gpu_sample_count, 0)
                        AS avg_gpu_usage_percent,
                    CASE WHEN gpu_sample_count > 0 THEN gpu_usage_percent_max END
                        AS peak_gpu_usage_percent,
                    gpu_memory_sample_count,
                    gpu_memory_usage_bytes_sum / NULLIF(gpu_memory_sample_count, 0)
                        AS avg_gpu_memory_usage_bytes,
                    CASE WHEN gpu_memory_sample_count > 0 THEN gpu_memory_usage_bytes_max END
                        AS peak_gpu_memory_usage_bytes
                FROM evaluation_resource_usage
                WHERE evaluation_id = %s
                ORDER BY execution_number ASC, component ASC
                """,
                (evaluation_id,),
            )
            return cur.fetchall()

    def totals_for_evaluation(self, evaluation_id: str) -> dict:
        """Return one evaluation's usage summed over executions and components."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(cpu_sample_count), 0) AS cpu_samples,
                    COALESCE(SUM(cpu_usage_cores_sum), 0) AS cpu_sum,
                    MAX(cpu_usage_cores_max) FILTER (WHERE cpu_sample_count > 0) AS peak_cpu_cores,
                    MAX(cpu_request_cores) AS cpu_request_cores,
                    MAX(cpu_limit_cores) AS cpu_limit_cores,
                    COALESCE(SUM(memory_sample_count), 0) AS memory_samples,
                    COALESCE(SUM(memory_usage_bytes_sum), 0) AS memory_sum,
                    MAX(memory_usage_bytes_max) FILTER (WHERE memory_sample_count > 0) AS peak_memory_bytes,
                    MAX(memory_request_bytes) AS memory_request_bytes,
                    MAX(memory_limit_bytes) AS memory_limit_bytes,
                    MAX(gpu_request) AS gpu_request,
                    COALESCE(SUM(gpu_sample_count), 0) AS gpu_samples
                FROM evaluation_resource_usage
                WHERE evaluation_id = %s
                """,
                (evaluation_id,),
            )
            row = cur.fetchone()
        return {key: float(value) if value is not None else None for key, value in row.items()}
