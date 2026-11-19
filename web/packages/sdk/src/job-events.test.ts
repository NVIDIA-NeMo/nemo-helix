// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readJobEvents } from '@nemo/sdk/generated/platform/jobs';
import type { JobEventPage, JobEventRequest } from '@nemo/sdk/generated/platform/schema';

const { fetch } = vi.hoisted(() => ({ fetch: vi.fn() }));
vi.mock('@nemo/sdk/generated/fetchers/platform.ts', () => ({ customFetch: fetch }));

const page: JobEventPage = {
  data: [
    {
      id: '1267650600228229401496703205376',
      event_type: 'job.created',
      workspace_id: 'workspace-id',
      job_id: 'job-id',
      attempt_id: null,
      step_id: null,
      previous_state: null,
      new_state: null,
      occurred_at: '2026-11-18T00:00:00Z',
      recorded_at: '2026-11-18T00:00:00Z',
      snapshot: { name: 'example', future_field: { value: true } },
    },
  ],
  next_cursor: 'opaque-server-token',
  prev_cursor: null,
  has_more: false,
};

it('generated event clients retain nullable fields, large IDs and opaque cursors', async () => {
  fetch.mockResolvedValue(page);
  const body: JobEventRequest = {
    filters: { workspace_id: null, event_types: ['job.created'] },
    limit: 10,
    wait_seconds: 0,
  };
  expect(await readJobEvents(body)).toEqual(page);
  expect(fetch).toHaveBeenLastCalledWith(
    expect.objectContaining({
      url: '/apis/jobs/v2/jobs/events',
      method: 'POST',
      data: body,
    })
  );
  const continuation = { cursor: page.next_cursor, limit: 25, wait_seconds: 10 };
  const result = await readJobEvents(continuation);
  expect(result.data[0]?.id).toBe('1267650600228229401496703205376');
  expect(result.prev_cursor).toBeNull();
  expect(fetch).toHaveBeenLastCalledWith(
    expect.objectContaining({
      url: '/apis/jobs/v2/jobs/events',
      method: 'POST',
      data: continuation,
    })
  );
});
