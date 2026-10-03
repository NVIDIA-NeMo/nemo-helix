/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useMemo } from 'react';
import { useAuth } from 'react-oidc-context';

import { isConfidentialOidcClient, useWebSession } from './useWebSession';

/** Millisecond timestamps exceed this; oidc-client-ts uses seconds since epoch. */
const EXPIRY_SECONDS_VS_MS_THRESHOLD = 10_000_000_000;

/**
 * Converts `User.expires_at` to milliseconds since epoch.
 * oidc-client-ts documents seconds; some tests/mocks use ms — handle both.
 */
export function normalizeOidcExpiresAtToMs(expiresAt: number | undefined): number | undefined {
  if (expiresAt == null) {
    return undefined;
  }
  return expiresAt < EXPIRY_SECONDS_VS_MS_THRESHOLD ? expiresAt * 1000 : expiresAt;
}

export type AuthTokenStatus = {
  /** Auth provider reports a signed-in user. */
  isAuthenticated: boolean;
  isLoading: boolean;
  /**
   * Access token is past expiry (client clock). Meaningful when `isAuthenticated`;
   * `false` if expiry is unknown.
   */
  isExpired: boolean;
  /** Signed in with a user record and access token not past known expiry. */
  isTokenActive: boolean;
  /** OAuth scopes granted on the current session (from `user.scope`, space-delimited). */
  activeScopes: string[];
  /** Access token expiry instant when known. */
  expiresAt: Date | undefined;
};

/**
 * Client-side view of OIDC session: expiry and scopes from `react-oidc-context` / oidc-client-ts.
 * Does not validate the token signature; use only for UI gating. APIs must still enforce authz.
 */
export const useAuthTokenStatus = (): AuthTokenStatus => {
  const auth = useAuth();
  const webSession = useWebSession();

  return useMemo(() => {
    if (isConfidentialOidcClient) {
      return {
        isAuthenticated: webSession.isAuthenticated,
        isLoading: webSession.isLoading,
        isExpired: false,
        isTokenActive: webSession.isAuthenticated,
        activeScopes: [],
        expiresAt: undefined,
      };
    }

    const user = auth.user;
    const isAuthenticated = auth.isAuthenticated && user != null;
    const expiresAtMs = normalizeOidcExpiresAtToMs(user?.expires_at);
    const isExpired = isAuthenticated && expiresAtMs != null ? Date.now() >= expiresAtMs : false;
    const isTokenActive = isAuthenticated && (expiresAtMs == null || Date.now() < expiresAtMs);

    const scopeString = user?.scope?.trim() ?? '';
    const activeScopes = scopeString
      ? scopeString.split(/\s+/).filter((s): s is string => Boolean(s))
      : [];

    return {
      isAuthenticated,
      isLoading: auth.isLoading,
      isExpired,
      isTokenActive,
      activeScopes,
      expiresAt: expiresAtMs != null ? new Date(expiresAtMs) : undefined,
    };
  }, [
    auth.isAuthenticated,
    auth.isLoading,
    auth.user,
    webSession.isAuthenticated,
    webSession.isLoading,
  ]);
};
