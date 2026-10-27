// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const EM_DASH = '—';
export const DEFAULT_DEBOUNCE_MS = 500;
export const DEFAULT_NAMESPACE = 'default';
export const DEFAULT_MODEL_NAME = 'nvidia/nemotron-3.5-lightning-30b-a3b';
export const JOB_POLLING_INTERVAL_MS = 5000;
export const JOB_POLLING_INTERVAL_LONG = 15000;
/** Newest log lines fetched on first load and on every poll. */
export const LOGS_TAIL_SIZE = 1_000;
/** The jobs API's per-request cap (MAX_LOG_QUERY_LINES), used when paging back through older lines. */
export const LOGS_MAX_PAGE_SIZE = 10_000;
export const LOGS_MAX_FETCH_ITERATIONS = 1000;
