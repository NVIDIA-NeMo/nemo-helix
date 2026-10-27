// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { diffLogs } from '@nemo/common/src/components/LogViewer/diffLogs';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';

function makeLog(index: number, message = `Log message ${index}`): HelixJobLog {
  return {
    timestamp: `2026-01-01T00:00:${String(index).padStart(2, '0')}Z`,
    job: 'test-job',
    job_step: 'step',
    job_task: 'task',
    message,
  };
}

const range = (from: number, to: number): HelixJobLog[] =>
  Array.from({ length: to - from }, (_, i) => makeLog(from + i));

describe('diffLogs', () => {
  it('reports no change for identical lists', () => {
    expect(diffLogs(range(0, 5), range(0, 5))).toEqual({ dropped: 0, added: [] });
  });

  it('treats everything as appended when there was nothing before', () => {
    expect(diffLogs([], range(0, 3))).toEqual({ dropped: 0, added: range(0, 3) });
  });

  it('reports an append', () => {
    expect(diffLogs(range(0, 5), range(0, 8))).toEqual({ dropped: 0, added: range(5, 8) });
  });

  it('reports entries trimmed from the head', () => {
    expect(diffLogs(range(0, 5), range(2, 5))).toEqual({ dropped: 2, added: [] });
  });

  it('reports a sliding window as a trim plus an append', () => {
    expect(diffLogs(range(0, 5), range(3, 9))).toEqual({ dropped: 3, added: range(5, 9) });
  });

  it('matches entries by content rather than identity', () => {
    const next = range(1, 6).map((log) => ({ ...log }));

    expect(diffLogs(range(0, 5), next)).toEqual({ dropped: 1, added: [makeLog(5)] });
  });

  it('ignores fields the viewer does not show', () => {
    const next = [...range(0, 3).map((log) => ({ ...log, job_task: 'other' })), makeLog(3)];

    expect(diffLogs(range(0, 3), next)).toEqual({ dropped: 0, added: [makeLog(3)] });
  });

  it('keeps the most content when repeated lines allow several alignments', () => {
    const repeated = { ...makeLog(0), message: 'tick' };
    const prev = [repeated, repeated];
    const next = [repeated, repeated, repeated];

    expect(diffLogs(prev, next)).toEqual({ dropped: 0, added: [repeated] });
  });

  it('skips an alignment that only matches at its ends', () => {
    // Dropping one lines up A with A at both ends of the overlap, but B and C differ in
    // between, so the only consistent reading is that everything before the last A went.
    const [a, b, c, x, y] = ['A', 'B', 'C', 'X', 'Y'].map((message, i) => makeLog(i, message));
    const prev = [x, a, b, a];
    const next = [a, c, a, y];

    expect(diffLogs(prev, next)).toEqual({ dropped: 3, added: [c, a, y] });
  });

  it('returns null when the lists share nothing', () => {
    expect(diffLogs(range(0, 3), range(5, 8))).toBeNull();
  });

  it('returns null when everything was cleared', () => {
    expect(diffLogs(range(0, 3), [])).toBeNull();
  });

  it('returns null when an earlier entry changed', () => {
    const next = range(0, 5);
    next[2] = makeLog(2, 'rewritten');

    expect(diffLogs(range(0, 5), next)).toBeNull();
  });

  it('returns null when an entry was inserted mid-list', () => {
    const next = [...range(0, 2), makeLog(9), ...range(2, 5)];

    expect(diffLogs(range(0, 5), next)).toBeNull();
  });
});
