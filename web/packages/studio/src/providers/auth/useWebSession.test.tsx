// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAuthAutoLogin } from '@studio/providers/auth/useAuthLogin';
import { useAuthProfile } from '@studio/providers/auth/useAuthProfile';
import { useAuthTokenStatus } from '@studio/providers/auth/useAuthTokenStatus';
import {
  AUTH_EXPLICIT_LOGOUT_STORAGE_KEY,
  useAuthSignOut,
  useWebSession,
  WEB_SESSION_QUERY_KEY,
} from '@studio/providers/auth/useWebSession';
import { mockSignoutRedirect } from '@studio/tests/mocks/react-oidc-context';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MemoryRouter } from 'react-router';

const mocks = vi.hoisted(() => ({
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
  AUTH_AUTHORITY: 'https://idp.example.com',
  AUTH_CLIENT_ID: 'nemo-helix-user',
  BASE_URL: '/studio/',
}));

vi.mock('@nemo/sdk/src/utils/oidcBrowserAuthentication', () => ({
  getOidcBrowserAuthentication: vi.fn(),
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
  });

  it('does not restart broker login after an explicit server-session logout', () => {
    discoveryResult = { data: confidentialAuthentication, isPending: false, isError: false };
    sessionResult = { data: undefined, isPending: false, isError: false };
    window.sessionStorage.setItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY, 'true');

    const login = renderHook(() => useAuthAutoLogin(), { wrapper }).result;

    expect(login.current.isAuthPending).toBe(false);
    expect(window.location.assign).not.toHaveBeenCalled();
  });

  it('clears explicit logout suppression once a server session exists again', async () => {
    window.sessionStorage.setItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY, 'true');

    renderHook(() => useAuthAutoLogin(), { wrapper });

    await waitFor(() =>
      expect(window.sessionStorage.getItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY)).toBeNull()
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

  it('uses bounded retries for transient discovery and session lookup failures', () => {
    renderHook(() => useWebSession());

    const discoveryQuery = mocks.useQuery.mock.calls.find(
      ([options]) => options.queryKey[1] === 'discovery'
    )?.[0] as { retry: unknown } | undefined;
    const sessionQuery = mocks.useQuery.mock.calls.find(
      ([options]) => options.queryKey[1] === 'web-session'
    )?.[0] as { retry: unknown } | undefined;

    expect(discoveryQuery?.retry).toBe(2);
    expect(sessionQuery?.retry).toBe(2);
  });

  it('waits for discovery and the session lookup before choosing a login flow', () => {
    discoveryResult = { data: undefined, isPending: true, isError: false };
    sessionResult = { data: undefined, isPending: false, isError: false };

    const login = renderHook(() => useAuthAutoLogin(), { wrapper }).result;
    const tokenStatus = renderHook(() => useAuthTokenStatus()).result;

    expect(login.current.isAuthPending).toBe(true);
    expect(tokenStatus.current).toMatchObject({ isLoading: true, isTokenActive: false });
    expect(window.location.assign).not.toHaveBeenCalled();
  });

  it('fails closed when discovery fails', () => {
    discoveryResult = { data: undefined, isPending: false, isError: true };

    const login = renderHook(() => useAuthAutoLogin(), { wrapper }).result;

    expect(login.current.isAuthPending).toBe(true);
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
