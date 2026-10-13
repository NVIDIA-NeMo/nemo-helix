// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  parseOidcBrowserAuthentication,
  withOidcServerSessionRequest,
} from './oidcBrowserAuthentication';

const importFreshBrowserAuthentication = async () => {
  vi.resetModules();
  return import('./oidcBrowserAuthentication');
};

describe('parseOidcBrowserAuthentication', () => {
  it('uses the default public client when discovery also advertises a confidential client', () => {
    expect(
      parseOidcBrowserAuthentication({
        auth_enabled: true,
        oidc: {
          clients: [
            {
              name: 'public',
              client_id: 'studio-public',
              client_authentication: 'public',
              default: true,
              server_side_sessions: false,
            },
            {
              name: 'confidential',
              client_id: 'cli-confidential',
              client_authentication: 'client_secret_basic',
              default: false,
            },
          ],
        },
      })
    ).toEqual({ authEnabled: true });
  });

  it('uses the configured Studio client when it is advertised', () => {
    expect(
      parseOidcBrowserAuthentication(
        {
          auth_enabled: true,
          oidc: {
            clients: [
              {
                name: 'public',
                client_id: 'studio-public',
                client_authentication: 'public',
                default: true,
                server_side_sessions: false,
              },
              {
                name: 'confidential',
                client_id: 'studio-confidential',
                client_authentication: 'client_secret_basic',
                default: false,
              },
            ],
          },
        },
        'studio-confidential'
      )
    ).toEqual({ authEnabled: true, serverSessionClient: 'confidential' });
  });

  it('selects a public client that opts into server-side sessions', () => {
    expect(
      parseOidcBrowserAuthentication({
        auth_enabled: true,
        oidc: {
          clients: [
            {
              name: 'public',
              client_id: 'studio-public',
              client_authentication: 'public',
              default: true,
              server_side_sessions: true,
            },
          ],
        },
      })
    ).toEqual({ authEnabled: true, serverSessionClient: 'public' });
  });

  it('keeps direct public authentication when server-side sessions are disabled', () => {
    expect(
      parseOidcBrowserAuthentication({
        auth_enabled: true,
        oidc: {
          clients: [
            {
              name: 'public',
              client_id: 'studio-public',
              client_authentication: 'public',
              default: true,
              server_side_sessions: false,
            },
          ],
        },
      })
    ).toEqual({ authEnabled: true });
  });

  it('preserves disabled authentication', () => {
    expect(parseOidcBrowserAuthentication({ auth_enabled: false, oidc: null })).toEqual({
      authEnabled: false,
    });
  });

  it.each([null, {}, { auth_enabled: true, oidc: {} }])(
    'rejects malformed discovery instead of silently downgrading authentication',
    (discovery) => {
      expect(() => parseOidcBrowserAuthentication(discovery)).toThrow();
    }
  );
});

describe('withOidcServerSessionRequest', () => {
  it('adds credentials, removes stale bearer auth, and sets the Studio source for a server session', async () => {
    const request = await withOidcServerSessionRequest(
      {
        method: 'POST',
        headers: { Accept: 'application/json', Authorization: 'Bearer stale-token' },
      },
      { authEnabled: true, serverSessionClient: 'confidential' }
    );

    const headers = new Headers(request?.headers);
    expect(request?.credentials).toBe('include');
    expect(headers.get('X-Source')).toBe('NeMo Studio');
    expect(headers.get('Accept')).toBe('application/json');
    expect(headers.has('Authorization')).toBe(false);
  });

  it('leaves direct public requests unchanged', async () => {
    const init = { method: 'POST', headers: { Accept: 'application/json' } };

    expect(
      await withOidcServerSessionRequest(init, {
        authEnabled: true,
      })
    ).toBe(init);
  });
});

describe('getOidcBrowserAuthentication', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it('fetches discovery without credentials', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test');
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        auth_enabled: false,
        oidc: null,
      })
    );
    vi.stubGlobal('fetch', fetchMock);
    const { getOidcBrowserAuthentication } = await importFreshBrowserAuthentication();

    await expect(getOidcBrowserAuthentication()).resolves.toEqual({ authEnabled: false });

    expect(fetchMock).toHaveBeenCalledWith('https://platform.example.test/apis/auth/discovery', {
      credentials: 'omit',
      signal: expect.any(AbortSignal),
    });
  });

  it('aborts stalled discovery without marking the timeout as retryable', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test');
    const timeoutError = new DOMException('The operation timed out', 'TimeoutError');
    const fetchMock = vi.fn().mockRejectedValue(timeoutError);
    const timeoutSpy = vi.spyOn(AbortSignal, 'timeout');
    vi.stubGlobal('fetch', fetchMock);
    const { getOidcBrowserAuthentication, isRetryableOidcBrowserAuthenticationError: isRetryable } =
      await importFreshBrowserAuthentication();

    const discovery = getOidcBrowserAuthentication();

    const error = await discovery.catch((discoveryError: unknown) => discoveryError);

    expect(error).toMatchObject({
      name: 'OidcBrowserAuthenticationTimeoutError',
      message: 'Auth discovery timed out after 5000ms',
      cause: timeoutError,
    });
    expect(isRetryable(error)).toBe(false);
    expect(timeoutSpy).toHaveBeenCalledWith(5_000);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it('treats a missing discovery endpoint as disabled authentication', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test');
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 404 }));
    vi.stubGlobal('fetch', fetchMock);
    const { getOidcBrowserAuthentication } = await importFreshBrowserAuthentication();

    await expect(getOidcBrowserAuthentication()).resolves.toEqual({ authEnabled: false });
  });

  it('uses the configured Studio client id when selecting from discovery', async () => {
    vi.stubEnv('VITE_AUTH_CLIENT_ID', 'studio-confidential');
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        auth_enabled: true,
        oidc: {
          clients: [
            {
              name: 'public',
              client_id: 'studio-public',
              client_authentication: 'public',
              default: true,
              server_side_sessions: false,
            },
            {
              name: 'confidential',
              client_id: 'studio-confidential',
              client_authentication: 'client_secret_basic',
              default: false,
            },
          ],
        },
      })
    );
    vi.stubGlobal('fetch', fetchMock);
    const { getOidcBrowserAuthentication } = await importFreshBrowserAuthentication();

    await expect(getOidcBrowserAuthentication()).resolves.toEqual({
      authEnabled: true,
      serverSessionClient: 'confidential',
    });
  });

  it('retries discovery after a failed request', async () => {
    const { getOidcBrowserAuthentication } = await importFreshBrowserAuthentication();
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error('network down'))
      .mockResolvedValueOnce(
        Response.json({
          auth_enabled: false,
          oidc: null,
        })
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getOidcBrowserAuthentication()).rejects.toThrow('network down');
    await expect(getOidcBrowserAuthentication()).resolves.toEqual({ authEnabled: false });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
