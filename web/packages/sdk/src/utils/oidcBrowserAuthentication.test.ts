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
  it('prefers the confidential client when discovery advertises multiple clients', () => {
    expect(
      parseOidcBrowserAuthentication({
        auth_enabled: true,
        oidc: {
          clients: [
            {
              name: 'public',
              client_authentication: 'public',
              server_side_sessions: true,
            },
            { name: 'confidential', client_authentication: 'client_secret_basic' },
          ],
        },
      })
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
              client_authentication: 'public',
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
              client_authentication: 'public',
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
    });
  });

  it('treats a missing discovery endpoint as disabled authentication', async () => {
    vi.stubEnv('VITE_PLATFORM_BASE_URL', 'https://platform.example.test');
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 404 }));
    vi.stubGlobal('fetch', fetchMock);
    const { getOidcBrowserAuthentication } = await importFreshBrowserAuthentication();

    await expect(getOidcBrowserAuthentication()).resolves.toEqual({ authEnabled: false });
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
