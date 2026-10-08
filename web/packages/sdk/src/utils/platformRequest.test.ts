// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AxiosRequestConfig } from 'axios';

import {
  platformFetch,
  withHelixBrowserAuthAxios,
  withHelixBrowserAuthRequest,
} from './platformRequest';

const createStorage = (entries: Record<string, string>) => ({
  getItem: vi.fn((key: string) => entries[key] ?? null),
  removeItem: vi.fn((key: string) => {
    delete entries[key];
  }),
});

const storedUser = (token: string): string =>
  JSON.stringify({
    access_token: token,
    expires_at: Math.floor(Date.now() / 1000) + 3600,
    profile: { sub: 'user-1' },
    token_type: 'Bearer',
  });

describe('withHelixBrowserAuthRequest', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('uses server-session credentials and removes stale bearer auth', async () => {
    const request = await withHelixBrowserAuthRequest(
      {
        method: 'POST',
        headers: { Authorization: 'Bearer stale-token', Accept: 'application/json' },
      },
      { authentication: { authEnabled: true, serverSessionClient: 'confidential' } }
    );

    const headers = new Headers(request.headers);
    expect(request.credentials).toBe('include');
    expect(headers.get('X-Source')).toBe('NeMo Studio');
    expect(headers.get('Accept')).toBe('application/json');
    expect(headers.has('Authorization')).toBe(false);
  });

  it('adds a stored bearer token for direct browser auth', async () => {
    vi.stubEnv('VITE_AUTH_AUTHORITY', 'https://auth.example.test');
    vi.stubEnv('VITE_AUTH_CLIENT_ID', 'studio-client');
    const storage = createStorage({
      'oidc.user:https://auth.example.test:studio-client': storedUser('direct-token'),
    });

    const request = await withHelixBrowserAuthRequest(
      { headers: { Accept: 'application/json' } },
      { authentication: { authEnabled: true }, storage }
    );

    const headers = new Headers(request.headers);
    expect(headers.get('Authorization')).toBe('Bearer direct-token');
    expect(headers.get('X-Source')).toBe('NeMo Studio');
    expect(request.credentials).toBeUndefined();
  });

  it('preserves an explicit bearer token for direct browser auth', async () => {
    vi.stubEnv('VITE_AUTH_AUTHORITY', 'https://auth.example.test');
    vi.stubEnv('VITE_AUTH_CLIENT_ID', 'studio-client');
    const storage = createStorage({
      'oidc.user:https://auth.example.test:studio-client': storedUser('stored-token'),
    });

    const request = await withHelixBrowserAuthRequest(
      { headers: { Authorization: 'Bearer explicit-token' } },
      { authentication: { authEnabled: true }, storage }
    );

    expect(new Headers(request.headers).get('Authorization')).toBe('Bearer explicit-token');
  });
});

describe('withHelixBrowserAuthAxios', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('uses server-session credentials and removes stale object authorization', async () => {
    const config = {
      headers: { AUTHORIZATION: 'Bearer stale-token' },
    } as AxiosRequestConfig;

    const nextConfig = await withHelixBrowserAuthAxios(config, {
      authentication: { authEnabled: true, serverSessionClient: 'public' },
    });

    expect(nextConfig.withCredentials).toBe(true);
    expect(nextConfig.headers?.AUTHORIZATION).toBeUndefined();
    expect(nextConfig.headers?.['X-Source']).toBe('NeMo Studio');
  });

  it('uses AxiosHeaders-style mutation methods when available', async () => {
    const headers = {
      set: vi.fn(),
      delete: vi.fn(),
      has: vi.fn().mockReturnValue(false),
    };
    const config = { headers } as unknown as AxiosRequestConfig;

    await withHelixBrowserAuthAxios(config, {
      authentication: { authEnabled: true, serverSessionClient: 'confidential' },
    });

    expect(headers.set).toHaveBeenCalledWith('X-Source', 'NeMo Studio');
    expect(headers.delete).toHaveBeenCalledWith('Authorization');
  });

  it('adds a stored bearer token when no explicit authorization exists', async () => {
    vi.stubEnv('VITE_AUTH_AUTHORITY', 'https://auth.example.test');
    vi.stubEnv('VITE_AUTH_CLIENT_ID', 'studio-client');
    const storage = createStorage({
      'oidc.user:https://auth.example.test:studio-client': storedUser('axios-token'),
    });
    const config = { headers: {} } as AxiosRequestConfig;

    const nextConfig = await withHelixBrowserAuthAxios(config, {
      authentication: { authEnabled: true },
      storage,
    });

    expect(nextConfig.withCredentials).toBeUndefined();
    expect(nextConfig.headers?.Authorization).toBe('Bearer axios-token');
    expect(nextConfig.headers?.['X-Source']).toBe('NeMo Studio');
  });
});

describe('platformFetch', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it('retries transient discovery failures before using server-session credentials', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test/');
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('network unavailable'))
      .mockResolvedValueOnce(
        Response.json({
          auth_enabled: true,
          oidc: {
            clients: [{ name: 'confidential', client_authentication: 'client_secret_basic' }],
          },
        })
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await platformFetch('/apis/example', { headers: { Authorization: 'Bearer stale-token' } });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    const request = fetchMock.mock.calls[2]?.[1] as RequestInit | undefined;
    const headers = new Headers(request?.headers);
    expect(request?.credentials).toBe('include');
    expect(headers.get('X-Source')).toBe('NeMo Studio');
    expect(headers.has('Authorization')).toBe(false);
  });

  it('resolves relative platform URLs against the browser platform base URL', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test/');
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await platformFetch('/apis/example', undefined);

    expect(fetchMock).toHaveBeenCalledWith(
      'https://platform.example.test/apis/example',
      expect.objectContaining({
        headers: expect.any(Headers),
      })
    );
  });

  it('leaves non-platform URLs unauthenticated by default', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test/');
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await platformFetch('https://downloads.example.test/results.json');

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith(
      'https://downloads.example.test/results.json',
      undefined
    );
  });

  it('does not treat sibling path prefixes as platform URLs', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test/apis');
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await platformFetch('https://platform.example.test/apis-other/results.json');

    expect(fetchMock).toHaveBeenCalledWith(
      'https://platform.example.test/apis-other/results.json',
      undefined
    );
  });

  it('treats configured base path segment descendants as platform URLs', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test/apis');
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await platformFetch('https://platform.example.test/apis/results.json');

    const request = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(new Headers(request?.headers).get('X-Source')).toBe('NeMo Studio');
  });

  it('preserves caller-supplied authentication for non-platform URLs', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test/');
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    const init = { headers: { Authorization: 'Bearer caller-token' } };
    vi.stubGlobal('fetch', fetchMock);

    await platformFetch('https://downloads.example.test/results.json', init);

    expect(fetchMock).toHaveBeenCalledWith('https://downloads.example.test/results.json', init);
  });

  it('preserves headers from platform Request inputs when init headers are absent', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test/');
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    const request = new Request('https://platform.example.test/apis/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Custom': 'request-value' },
      body: '{}',
    });
    vi.stubGlobal('fetch', fetchMock);

    await platformFetch(request);

    const init = fetchMock.mock.calls.at(-1)?.[1] as RequestInit | undefined;
    const headers = new Headers(init?.headers);
    expect(headers.get('Content-Type')).toBe('application/json');
    expect(headers.get('X-Custom')).toBe('request-value');
    expect(headers.get('X-Source')).toBe('NeMo Studio');
  });

  it('lets explicit init headers override headers from platform Request inputs', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test/');
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    const request = new Request('https://platform.example.test/apis/upload', {
      method: 'POST',
      headers: { 'X-Custom': 'request-value' },
    });
    vi.stubGlobal('fetch', fetchMock);

    await platformFetch(request, { headers: { 'X-Custom': 'init-value' } });

    const init = fetchMock.mock.calls.at(-1)?.[1] as RequestInit | undefined;
    const headers = new Headers(init?.headers);
    expect(headers.get('X-Custom')).toBe('init-value');
  });
});
