/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  normalizeOidcExpiresAtToMs,
  useAuthTokenStatus,
} from '@studio/providers/auth/useAuthTokenStatus';
import { renderHook } from '@testing-library/react';
import type { User } from 'oidc-client-ts';
import { type AuthState, useAuth } from 'react-oidc-context';

vi.mock('react-oidc-context');

const { mockUseWebSession } = vi.hoisted(() => ({ mockUseWebSession: vi.fn() }));
vi.mock('@studio/providers/auth/useWebSession', () => ({
  useWebSession: mockUseWebSession,
}));

const mockUseAuth = vi.mocked(useAuth);

const frozenNow = new Date('2025-06-01T12:00:00.000Z');

function mockUser(overrides: Partial<User>): User {
  return {
    profile: {},
    ...overrides,
  } as User;
}

function mockAuthState(overrides: Partial<AuthState>): void {
  const state = {
    isLoading: false,
    isAuthenticated: false,
    user: null,
    ...overrides,
  } satisfies AuthState;

  // useAuth() is typed as AuthContextProps; tests only stub the AuthState slice the hook reads.
  mockUseAuth.mockReturnValue(state as ReturnType<typeof useAuth>);
}

describe('normalizeOidcExpiresAtToMs', () => {
  it('treats values below threshold as Unix seconds', () => {
    const seconds = 1_700_000_000;
    expect(normalizeOidcExpiresAtToMs(seconds)).toBe(seconds * 1000);
  });

  it('treats values at or above threshold as milliseconds', () => {
    const ms = 1_700_000_000_000;
    expect(normalizeOidcExpiresAtToMs(ms)).toBe(ms);
  });

  it('returns undefined for undefined', () => {
    expect(normalizeOidcExpiresAtToMs(undefined)).toBeUndefined();
  });
});

describe('useAuthTokenStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(frozenNow);
    mockUseAuth.mockReset();
    mockUseWebSession.mockReturnValue({
      session: undefined,
      serverSessionClient: undefined,
      isServerSession: false,
      isAuthenticated: false,
      isLoading: false,
      isError: false,
      authEnabled: true,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('reflects loading when auth is loading', () => {
    mockAuthState({ isLoading: true, isAuthenticated: false, user: null });

    const { result } = renderHook(() => useAuthTokenStatus());

    expect(result.current.isLoading).toBe(true);
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.isTokenActive).toBe(false);
    expect(result.current.activeScopes).toEqual([]);
  });

  it('is unauthenticated when there is no user', () => {
    mockAuthState({ isLoading: false, isAuthenticated: false, user: null });

    const { result } = renderHook(() => useAuthTokenStatus());

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.isExpired).toBe(false);
    expect(result.current.isTokenActive).toBe(false);
    expect(result.current.expiresAt).toBeUndefined();
    expect(result.current.activeScopes).toEqual([]);
  });

  it('is unauthenticated when isAuthenticated is true but user is missing', () => {
    mockAuthState({ isLoading: false, isAuthenticated: true, user: null });

    const { result } = renderHook(() => useAuthTokenStatus());

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.isTokenActive).toBe(false);
  });

  it('parses scopes and treats token as active before expires_at', () => {
    const expiresAtSec = Math.floor(new Date('2025-06-01T13:00:00.000Z').getTime() / 1000);
    mockAuthState({
      isLoading: false,
      isAuthenticated: true,
      user: mockUser({
        scope: 'openid profile email platform:read',
        expires_at: expiresAtSec,
      }),
    });

    const { result } = renderHook(() => useAuthTokenStatus());

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.isExpired).toBe(false);
    expect(result.current.isTokenActive).toBe(true);
    expect(result.current.activeScopes).toEqual(['openid', 'profile', 'email', 'platform:read']);
    expect(result.current.expiresAt?.toISOString()).toBe('2025-06-01T13:00:00.000Z');
  });

  it('treats token as expired when past expires_at', () => {
    const expiresAtSec = Math.floor(new Date('2025-06-01T11:00:00.000Z').getTime() / 1000);
    mockAuthState({
      isLoading: false,
      isAuthenticated: true,
      user: mockUser({
        scope: 'openid',
        expires_at: expiresAtSec,
      }),
    });

    const { result } = renderHook(() => useAuthTokenStatus());

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.isExpired).toBe(true);
    expect(result.current.isTokenActive).toBe(false);
  });

  it('treats token as active when expires_at is missing', () => {
    mockAuthState({
      isLoading: false,
      isAuthenticated: true,
      user: mockUser({
        scope: 'openid',
      }),
    });

    const { result } = renderHook(() => useAuthTokenStatus());

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.isExpired).toBe(false);
    expect(result.current.isTokenActive).toBe(true);
    expect(result.current.expiresAt).toBeUndefined();
  });

  it('returns empty scopes for blank scope string', () => {
    mockAuthState({
      isLoading: false,
      isAuthenticated: true,
      user: mockUser({
        scope: '   ',
        expires_at: Math.floor(new Date('2025-06-01T13:00:00.000Z').getTime() / 1000),
      }),
    });

    const { result } = renderHook(() => useAuthTokenStatus());

    expect(result.current.activeScopes).toEqual([]);
  });

  it('normalizes extra whitespace between scopes', () => {
    mockAuthState({
      isLoading: false,
      isAuthenticated: true,
      user: mockUser({
        scope: 'openid   profile',
        expires_at: Math.floor(new Date('2025-06-01T13:00:00.000Z').getTime() / 1000),
      }),
    });

    const { result } = renderHook(() => useAuthTokenStatus());

    expect(result.current.activeScopes).toEqual(['openid', 'profile']);
  });
});
