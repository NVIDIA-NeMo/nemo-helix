// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAuthAutoLogin } from '@studio/providers/auth/useAuthLogin';
import { useAuthProfile } from '@studio/providers/auth/useAuthProfile';
import { useAuthTokenStatus } from '@studio/providers/auth/useAuthTokenStatus';
import { useAuthSignOut, useWebSession } from '@studio/providers/auth/useWebSession';
import { withWebSessionRequest } from '@studio/providers/auth/webSessionRequest';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MemoryRouter } from 'react-router';

const mocks = vi.hoisted(() => ({
  authLogout: vi.fn(),
  removeQueries: vi.fn(),
  useAuthSession: vi.fn(),
}));

vi.mock('@studio/constants/environment', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/constants/environment')>()),
  AUTH_AUTHORITY: 'https://idp.example.com',
  AUTH_CLIENT_ID: 'nemo-helix-user',
  BASE_URL: '/studio/',
  OIDC_TOKEN_ENDPOINT_AUTH_METHOD: 'client_secret_basic',
}));

vi.mock('@nemo/sdk/generated/platform/authentication', () => ({
  authLogout: mocks.authLogout,
  getAuthSessionQueryKey: () => ['/apis/auth/v2/session'],
  useAuthSession: mocks.useAuthSession,
}));

vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useQueryClient: () => ({ removeQueries: mocks.removeQueries }),
}));

const session = {
  id: 'user-1',
  email: 'user@example.com',
  groups: ['nemo-users'],
  account_id: 'account-1',
  authz_aliases: ['user-1', 'user@example.com'],
};

const wrapper = ({ children }: { children: ReactNode }) => (
  <MemoryRouter initialEntries={['/workspaces/example?tab=models']}>{children}</MemoryRouter>
);

describe('confidential web sessions', () => {
  beforeEach(() => {
    mocks.authLogout.mockReset();
    mocks.removeQueries.mockReset();
    mocks.useAuthSession.mockReset();
    vi.mocked(window.location.assign).mockReset();
  });

  it('uses the server session for profile and authenticated state', () => {
    mocks.useAuthSession.mockReturnValue({ data: session, isPending: false });

    const webSession = renderHook(() => useWebSession()).result;
    const profile = renderHook(() => useAuthProfile()).result;
    const tokenStatus = renderHook(() => useAuthTokenStatus()).result;

    expect(webSession.current).toMatchObject({ isAuthenticated: true, isLoading: false });
    expect(profile.current).toEqual({
      name: 'user@example.com',
      email: 'user@example.com',
      workspace: 'user',
    });
    expect(tokenStatus.current).toMatchObject({
      isAuthenticated: true,
      isTokenActive: true,
      isLoading: false,
    });
  });

  it('waits for session lookup before redirecting to the broker', async () => {
    mocks.useAuthSession.mockReturnValue({ data: undefined, isPending: true });
    const { rerender } = renderHook(() => useAuthAutoLogin(), { wrapper });

    expect(window.location.assign).not.toHaveBeenCalled();

    mocks.useAuthSession.mockReturnValue({ data: undefined, isPending: false });
    rerender();

    await waitFor(() =>
      expect(window.location.assign).toHaveBeenCalledWith(
        '/apis/auth/v2/login?return_to=%2Fworkspaces%2Fexample%3Ftab%3Dmodels'
      )
    );
  });

  it('revokes the server session and returns to Studio on sign out', async () => {
    mocks.authLogout.mockResolvedValue(undefined);
    const { result } = renderHook(() => useAuthSignOut());

    await act(() => result.current());

    expect(mocks.authLogout).toHaveBeenCalledOnce();
    expect(mocks.removeQueries).toHaveBeenCalledWith({
      queryKey: ['/apis/auth/v2/session'],
    });
    expect(window.location.assign).toHaveBeenCalledWith('/studio/');
  });

  it('includes the session cookie and mutation marker on hand-written requests', () => {
    const request = withWebSessionRequest({
      method: 'POST',
      headers: { Accept: 'text/event-stream' },
    });

    expect(request?.credentials).toBe('include');
    expect(new Headers(request?.headers).get('X-NHX-Requested-By')).toBe('1');
    expect(new Headers(request?.headers).get('Accept')).toBe('text/event-stream');
  });
});
