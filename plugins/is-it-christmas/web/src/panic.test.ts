// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest';
import { getPanic } from './api';

const host = { apiBaseUrl: 'http://platform', workspaceId: 'default', auth: { accessToken: 't', getAccessToken: () => 'tok' } };
const report = {
  detail: "thread 'tokio-runtime-worker' panicked at src/lib.rs:1:1:\ncalled `Result::unwrap()` on an `Err` value: ParseIntError { kind: InvalidDigit }",
  kind: 'panic',
  thread: 'tokio-runtime-worker',
  location: 'src/lib.rs:1:1',
  message: 'called `Result::unwrap()` on an `Err` value: ParseIntError { kind: InvalidDigit }',
};

afterEach(() => vi.unstubAllGlobals());

describe('getPanic', () => {
  it('treats the 500 as the expected answer and returns the report', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => report });
    vi.stubGlobal('fetch', fetchMock);

    await expect(getPanic(host)).resolves.toEqual(report);
    expect(fetchMock.mock.calls[0][0]).toBe('http://platform/apis/plugins/is-it-christmas/v2/workspaces/default/panic');
  });

  it('is surprised when the endpoint answers normally', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200 }));
    await expect(getPanic(host)).rejects.toThrow('Expected the endpoint to panic');
  });

  it('reports any other failure, such as the gateway not reaching the server', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 502, statusText: 'Bad Gateway' }));
    await expect(getPanic(host)).rejects.toThrow('502 Bad Gateway');
  });
});
