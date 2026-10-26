// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY,
  useAuthAutoLogin,
} from '@studio/providers/auth/useAuthLogin';
import { useAuthProfile } from '@studio/providers/auth/useAuthProfile';
import { useAuthTokenStatus } from '@studio/providers/auth/useAuthTokenStatus';
import {
  AUTH_EXPLICIT_LOGOUT_STORAGE_KEY,
  useAuthSignOut,
  useWebSession,
  WEB_SESSION_QUERY_KEY,
} from '@studio/providers/auth/useWebSession';
import {
  mockAuthState,
  mockSigninRedirect,
  mockSignoutRedirect,
} from '@studio/tests/mocks/react-oidc-context';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MemoryRouter } from 'react-router';

const mocks = vi.hoisted(() => ({
  authAuthority: 'https://idp.example.com' as string | undefined,
  authClientId: 'nemo-helix-user' as string | undefined,
  isRetryableDiscoveryError: vi.fn(),
  removeQueries: vi.fn(),
  toastError: vi.fn(),
  useQuery: vi.fn(),
  withOidcServerSessionRequest: vi.fn(),
}));

vi.mock('@nemo/common/src/providers/toast/useToast', () => ({
  useToast: () => ({
    error: mocks.toastError,
  }),
}));

vi.mock('@studio/constants/environment', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/constants/environment')>()),
  get AUTH_AUTHORITY() {
    return mocks.authAuthority;
  },
  get AUTH_CLIENT_ID() {
    return mocks.authClientId;
  },
  BASE_URL: '/studio/',
}));

vi.mock('@nemo/sdk/src/utils/oidcBrowserAuthentication', () => ({
  getOidcBrowserAuthentication: vi.fn(),
  isRetryableOidcBrowserAuthenticationError: mocks.isRetryableDiscoveryError,
  withOidcServerSessionRequest: mocks.withOidcServerSessionRequest,
}));

vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useQuery: mocks.useQuery,
  useQueryClient: () => ({ removeQueries: mocks.removeQueries }),
}));

const session = {
  id: 'user-1',
  email: 'user@example.com',
  groups: ['nemo-users'],
  account_id: 'account-1',
  authz_aliases: ['user-1', 'user@example.com'],
};

const confidentialAuthentication = {
  authEnabled: true,
  serverSessionClient: 'confidential',
} as const;

const publicSessionAuthentication = {
  authEnabled: true,
  serverSessionClient: 'public',
} as const;

let discoveryResult: Record<string, unknown>;
let sessionResult: Record<string, unknown>;

const wrapper = ({ children }: { children: ReactNode }) => (
  <MemoryRouter initialEntries={['/workspaces/example?tab=models#details']}>
    {children}
  </MemoryRouter>
);

describe('server-side browser sessions', () => {
  beforeEach(() => {
    mocks.authAuthority = 'https://idp.example.com';
    mocks.authClientId = 'nemo-helix-user';
    mocks.isRetryableDiscoveryError.mockReset();
    mocks.isRetryableDiscoveryError.mockImplementation(
      (error: unknown) => error instanceof TypeError
    );
    discoveryResult = {
      data: confidentialAuthentication,
      isPending: false,
      isError: false,
    };
    sessionResult = { data: session, isPending: false, isError: false };
    mocks.useQuery.mockReset();
    mocks.useQuery.mockImplementation(({ queryKey }: { queryKey: readonly string[] }) =>
      queryKey[1] === 'discovery' ? discoveryResult : sessionResult
    );
    mocks.removeQueries.mockReset();
    mocks.toastError.mockReset();
    mocks.withOidcServerSessionRequest.mockReset();
    mocks.withOidcServerSessionRequest.mockImplementation(async (init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      headers.set('X-Source', 'NeMo Studio');
      return { ...init, credentials: 'include', headers };
    });
    window.sessionStorage.removeItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY);
    window.sessionStorage.removeItem(AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY);
    mockAuthState.isAuthenticated = true;
    mockSigninRedirect.mockReset();
    mockSignoutRedirect.mockReset();
    vi.mocked(window.location.assign).mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('uses the confidential session for profile and authentication state', () => {
    const webSession = renderHook(() => useWebSession()).result;
    const profile = renderHook(() => useAuthProfile()).result;
    const tokenStatus = renderHook(() => useAuthTokenStatus()).result;

    expect(webSession.current).toMatchObject({
      serverSessionClient: 'confidential',
      isAuthenticated: true,
      isLoading: false,
      isError: false,
    });
    expect(profile.current).toEqual({
      id: 'user-1',
      name: 'user@example.com',
      email: 'user@example.com',
      workspace: 'user',
    });
    expect(tokenStatus.current).toMatchObject({
      isAuthenticated: true,
      isTokenActive: true,
      isLoading: false,
      activeScopes: [],
    });
  });

  it.each([
    ['confidential', confidentialAuthentication],
    ['public', publicSessionAuthentication],
  ] as const)('redirects an unauthenticated %s session through the broker', async (_, auth) => {
    discoveryResult = { data: auth, isPending: false, isError: false };
    sessionResult = { data: undefined, isPending: false, isError: false };

    renderHook(() => useAuthAutoLogin(), { wrapper });

    await waitFor(() =>
      expect(window.location.assign).toHaveBeenCalledWith(
        `http://localhost:8080/apis/auth/v2/login?client=${auth.serverSessionClient}&return_to=%2Fstudio%2Fworkspaces%2Fexample%3Ftab%3Dmodels%23details`
      )
    );
    expect(
      Number(window.sessionStorage.getItem(AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY))
    ).toBeGreaterThan(0);
  });

  it('shows an auth error instead of repeating a recent server-session login redirect', () => {
    sessionResult = { data: undefined, isPending: false, isError: false };
    window.sessionStorage.setItem(
      AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY,
      String(Date.now())
    );

    const login = renderHook(() => useAuthAutoLogin(), { wrapper }).result;

    expect(login.current.isAuthError).toBe(true);
    expect(login.current.isAuthPending).toBe(false);
    expect(window.location.assign).not.toHaveBeenCalled();
  });

  it('retries server-session login after the stored redirect guard expires', async () => {
    sessionResult = { data: undefined, isPending: false, isError: false };
    window.sessionStorage.setItem(
      AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY,
      String(Date.now() - 120_000)
    );

    renderHook(() => useAuthAutoLogin(), { wrapper });

    await waitFor(() =>
      expect(window.location.assign).toHaveBeenCalledWith(
        'http://localhost:8080/apis/auth/v2/login?client=confidential&return_to=%2Fstudio%2Fworkspaces%2Fexample%3Ftab%3Dmodels%23details'
      )
    );
  });

  it('offers sign-in after an explicit server-session logout', () => {
    discoveryResult = { data: confidentialAuthentication, isPending: false, isError: false };
    sessionResult = { data: undefined, isPending: false, isError: false };
    window.sessionStorage.setItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY, 'true');

    const login = renderHook(() => useAuthAutoLogin(), { wrapper }).result;

    expect(login.current.isAuthError).toBe(false);
    expect(login.current.isAuthPending).toBe(false);
    expect(login.current.isSignedOut).toBe(true);
    expect(window.location.assign).not.toHaveBeenCalled();

    act(() => login.current.signIn());

    expect(window.sessionStorage.getItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY)).toBeNull();
    expect(window.location.assign).toHaveBeenCalledWith(
      'http://localhost:8080/apis/auth/v2/login?client=confidential&return_to=%2Fstudio%2Fworkspaces%2Fexample%3Ftab%3Dmodels%23details'
    );
  });

  it('clears explicit logout suppression once a server session exists again', async () => {
    window.sessionStorage.setItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY, 'true');

    renderHook(() => useAuthAutoLogin(), { wrapper });

    await waitFor(() =>
      expect(window.sessionStorage.getItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY)).toBeNull()
    );
  });

  it('clears the server-session redirect guard once a server session exists', async () => {
    window.sessionStorage.setItem(
      AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY,
      String(Date.now())
    );

    renderHook(() => useAuthAutoLogin(), { wrapper });

    await waitFor(() =>
      expect(
        window.sessionStorage.getItem(AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY)
      ).toBeNull()
    );
  });

  it('represents an unauthenticated session as valid empty query data', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);
    renderHook(() => useWebSession());

    const sessionQuery = mocks.useQuery.mock.calls.find(
      ([options]) => options.queryKey[1] === 'web-session'
    )?.[0] as { queryFn: () => Promise<unknown> } | undefined;

    await expect(sessionQuery?.queryFn()).resolves.toBeNull();
  });

  it('uses bounded retries for retryable discovery and session lookup failures', () => {
    renderHook(() => useWebSession());

    const discoveryQuery = mocks.useQuery.mock.calls.find(
      ([options]) => options.queryKey[1] === 'discovery'
    )?.[0] as { retry: (failureCount: number, error: Error) => boolean } | undefined;
    const sessionQuery = mocks.useQuery.mock.calls.find(
      ([options]) => options.queryKey[1] === 'web-session'
    )?.[0] as { retry: unknown } | undefined;

    expect(discoveryQuery?.retry(0, new TypeError('network unavailable'))).toBe(true);
    expect(discoveryQuery?.retry(2, new TypeError('network unavailable'))).toBe(false);
    expect(discoveryQuery?.retry(0, new Error('discovery timed out'))).toBe(false);
    expect(sessionQuery?.retry).toBe(2);
  });

  it('waits for discovery and the session lookup before choosing a login flow', () => {
    discoveryResult = { data: undefined, isPending: true, isError: false };
    sessionResult = { data: undefined, isPending: false, isError: false };

    const login = renderHook(() => useAuthAutoLogin(), { wrapper }).result;
    const tokenStatus = renderHook(() => useAuthTokenStatus()).result;

    expect(login.current.isAuthPending).toBe(true);
    expect(login.current.isAuthError).toBe(false);
    expect(tokenStatus.current).toMatchObject({ isLoading: true, isTokenActive: false });
    expect(window.location.assign).not.toHaveBeenCalled();
  });

  it('exposes discovery failures separately from pending auth', () => {
    discoveryResult = { data: undefined, isPending: false, isError: true };

    const login = renderHook(() => useAuthAutoLogin(), { wrapper }).result;

    expect(login.current.isAuthPending).toBe(false);
    expect(login.current.isAuthError).toBe(true);
    expect(window.location.assign).not.toHaveBeenCalled();
  });

  it('skips auth discovery when no browser auth client id is configured', () => {
    mocks.authClientId = undefined;
    discoveryResult = { data: undefined, isPending: false, isError: true };
    sessionResult = { data: undefined, isPending: false, isError: false };

    const webSession = renderHook(() => useWebSession()).result;
    const login = renderHook(() => useAuthAutoLogin(), { wrapper }).result;

    expect(webSession.current).toMatchObject({
      authEnabled: false,
      isAuthenticated: false,
      isLoading: false,
      isError: false,
      isServerSession: false,
    });
    expect(login.current.isAuthPending).toBe(false);
    expect(login.current.isAuthError).toBe(false);
    expect(window.location.assign).not.toHaveBeenCalled();

    const discoveryQuery = mocks.useQuery.mock.calls.find(
      ([options]) => options.queryKey[1] === 'discovery'
    )?.[0] as { enabled: unknown } | undefined;
    expect(discoveryQuery?.enabled).toBe(false);
  });

  it('keeps env-configured direct OIDC login when discovery reports auth disabled', async () => {
    mockAuthState.isAuthenticated = false;
    discoveryResult = { data: { authEnabled: false }, isPending: false, isError: false };
    sessionResult = { data: undefined, isPending: false, isError: false };

    renderHook(() => useAuthAutoLogin(), { wrapper });

    await waitFor(() =>
      expect(mockSigninRedirect).toHaveBeenCalledWith({
        state: {
          path: '/workspaces/example',
          search: '?tab=models',
        },
      })
    );
    expect(window.location.assign).not.toHaveBeenCalled();
  });

  it('exposes session lookup failures separately from pending auth', () => {
    sessionResult = { data: undefined, isPending: false, isError: true };

    const login = renderHook(() => useAuthAutoLogin(), { wrapper }).result;

    expect(login.current.isAuthPending).toBe(false);
    expect(login.current.isAuthError).toBe(true);
    expect(window.location.assign).not.toHaveBeenCalled();
  });

  it('revokes the server session with the Studio source and returns to Studio', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useAuthSignOut());

    await act(() => result.current());

    const request = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8080/apis/auth/v2/logout',
      expect.any(Object)
    );
    expect(request?.credentials).toBe('include');
    expect(new Headers(request?.headers).get('X-Source')).toBe('NeMo Studio');
    expect(window.sessionStorage.getItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY)).toBe('true');
    expect(mocks.removeQueries).toHaveBeenCalledWith({ queryKey: WEB_SESSION_QUERY_KEY });
    expect(window.location.assign).toHaveBeenCalledWith('/studio/');
  });

  it('keeps local session state and notifies the user when server-session sign-out fails', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 503 }));
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useAuthSignOut());

    await act(() => result.current());

    expect(mocks.toastError).toHaveBeenCalledWith('Unable to sign out. Please try again.');
    expect(window.sessionStorage.getItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY)).toBeNull();
    expect(mocks.removeQueries).not.toHaveBeenCalled();
    expect(window.location.assign).not.toHaveBeenCalled();
  });

  it('retains the direct public-client sign-out flow', async () => {
    discoveryResult = {
      data: { authEnabled: true },
      isPending: false,
      isError: false,
    };
    const { result } = renderHook(() => useAuthSignOut());

    await act(() => result.current());

    expect(mockSignoutRedirect).toHaveBeenCalledOnce();
    expect(mocks.removeQueries).not.toHaveBeenCalled();
  });
});
