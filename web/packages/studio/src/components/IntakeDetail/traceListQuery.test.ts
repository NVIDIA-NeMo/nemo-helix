// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  encodeTraceListQuery,
  parseTraceListQuery,
  type TraceListQuery,
} from '@studio/components/IntakeDetail/traceListQuery';

const QUERY: TraceListQuery = {
  filter: { status: 'error' },
  mode: 'preview',
  page: 2,
  page_size: 50,
  sort: '-started_at',
};

describe('traceListQuery', () => {
  it('round trips a list request, filters included', () => {
    expect(parseTraceListQuery(encodeTraceListQuery(QUERY))).toEqual(QUERY);
  });

  it('rejects a request a replay cannot trust', () => {
    expect(parseTraceListQuery(null)).toBeNull();
    expect(parseTraceListQuery('')).toBeNull();
    expect(parseTraceListQuery('not json')).toBeNull();
    expect(parseTraceListQuery('[1,2]')).toBeNull();
    expect(parseTraceListQuery(JSON.stringify({ ...QUERY, sort: 'duration_ms' }))).toBeNull();
    expect(parseTraceListQuery(JSON.stringify({ ...QUERY, page: 0 }))).toBeNull();
    expect(parseTraceListQuery(JSON.stringify({ ...QUERY, page: 1.5 }))).toBeNull();
    expect(parseTraceListQuery(JSON.stringify({ ...QUERY, page_size: '50' }))).toBeNull();
  });
});
