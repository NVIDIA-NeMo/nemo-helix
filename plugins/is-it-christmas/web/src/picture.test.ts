// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest';
import { getPicture } from './api';

const host = { apiBaseUrl: 'http://platform', workspaceId: 'default', auth: { accessToken: 't', getAccessToken: () => 'tok' } };
const picture = { name: 'Stefan', content_type: 'image/gif', bytes: 3, data_url: 'data:image/gif;base64,R0lG' };

afterEach(() => vi.unstubAllGlobals());

describe('getPicture', () => {
  it.each([
    ['stefan', 'stefan'],
    ['stefan-animated', 'stefan-animated'],
  ] as const)('asks the plugin route for %s through the gateway', async (kind, route) => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => picture });
    vi.stubGlobal('fetch', fetchMock);

    await expect(getPicture(host, kind)).resolves.toEqual(picture);
    expect(fetchMock.mock.calls[0][0]).toBe(`http://platform/apis/plugins/is-it-christmas/v2/workspaces/default/${route}`);
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: 'Bearer tok' });
  });

  it('passes the mood as a query parameter', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => picture });
    vi.stubGlobal('fetch', fetchMock);

    await getPicture(host, 'stefan', { mood: 'sad' });

    expect(fetchMock.mock.calls[0][0]).toBe('http://platform/apis/plugins/is-it-christmas/v2/workspaces/default/stefan?mood=sad');
  });

  it('reports a failed request so the page can simply omit the picture', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: 'Not Found' }));
    await expect(getPicture(host, 'stefan')).rejects.toThrow('404 Not Found');
  });
});
