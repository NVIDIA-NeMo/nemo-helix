// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest';
import { getAnswer, localDate } from './api';

const host = { apiBaseUrl: 'http://platform/', workspaceId: 'my ws', auth: { accessToken: 't', getAccessToken: () => 'tok' } };

afterEach(() => vi.unstubAllGlobals());

describe('localDate', () => {
  it('uses the viewer calendar date, zero padded', () => {
    expect(localDate(new Date(2026, 0, 5, 23, 59))).toBe('2026-01-05');
  });
});

describe('getAnswer', () => {
  it('calls the plugin route through the gateway with auth and the date', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ message: 'hi' }) });
    vi.stubGlobal('fetch', fetchMock);

    await getAnswer(host, '2026-12-25');

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('http://platform/apis/plugins/is-it-christmas/v2/workspaces/my%20ws/is-it-christmas?date=2026-12-25');
    expect(init.headers).toEqual({ Authorization: 'Bearer tok' });
  });

  it('turns a non-2xx response into an error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 502, statusText: 'Bad Gateway' }));
    await expect(getAnswer(host, '2026-12-25')).rejects.toThrow('502 Bad Gateway');
  });
});
