// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { OidcIdTokenRenewal } from '@studio/providers/auth/OidcIdTokenRenewal';
import { act, render } from '@testing-library/react';
import type { User } from 'oidc-client-ts';
import { useAuth } from 'react-oidc-context';

vi.mock('@studio/constants/environment', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/constants/environment')>()),
  AUTH_BEARER_TOKEN_SOURCE: 'id_token',
}));
vi.mock('react-oidc-context');

const mockUseAuth = vi.mocked(useAuth);
const frozenNow = new Date('2026-01-01T12:00:00.000Z');

const createIdToken = (expiresAt: number): string => {
  const encode = (value: object) =>
    globalThis
      .btoa(JSON.stringify(value))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  return `${encode({ alg: 'ES256', typ: 'JWT' })}.${encode({ exp: expiresAt })}.signature`;
};

const mockAuth = (idToken: string, signinSilent: ReturnType<typeof vi.fn>): void => {
  mockUseAuth.mockReturnValue({
    signinSilent,
    user: { id_token: idToken } as User,
  } as unknown as ReturnType<typeof useAuth>);
};

describe('OidcIdTokenRenewal', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(frozenNow);
    mockUseAuth.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('silently renews 90 seconds before the ID token expires', () => {
    const signinSilent = vi.fn().mockReturnValue(new Promise(() => undefined));
    const expiresAt = Math.floor(frozenNow.getTime() / 1000) + 10 * 60;
    mockAuth(createIdToken(expiresAt), signinSilent);

    render(<OidcIdTokenRenewal />);

    act(() => vi.advanceTimersByTime(8 * 60 * 1000 + 29_999));
    expect(signinSilent).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(1));
    expect(signinSilent).toHaveBeenCalledOnce();
  });

  it('renews immediately when the ID token is already inside the renewal window', () => {
    const signinSilent = vi.fn().mockReturnValue(new Promise(() => undefined));
    const expiresAt = Math.floor(frozenNow.getTime() / 1000) + 30;
    mockAuth(createIdToken(expiresAt), signinSilent);

    render(<OidcIdTokenRenewal />);

    expect(signinSilent).toHaveBeenCalledOnce();
  });
});
