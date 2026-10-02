// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AUTH_AUTHORITY, AUTH_CLIENT_ID } from '@studio/constants/environment';
import { isConfidentialOidcClient, useWebSession } from '@studio/providers/auth/useWebSession';
import { useEffect, useState } from 'react';
import { hasAuthParams, useAuth } from 'react-oidc-context';
import { useLocation } from 'react-router';

/**
 * This hook is used to automatically handle the login process.
 *
 * Mirrors logic in useAutoSignin that can be enabled when the AUTH_CLIENT_ID and AUTH_AUTHORITY env vars are set.
 *
 * @returns `isAuthPending` - true when auth is enabled and the user is not yet authenticated (UI should be hidden)
 */
export const useAuthAutoLogin = (): { isAuthPending: boolean } => {
  const auth = useAuth();
  const [hasAttemptedLogin, setHasAttemptedLogin] = useState(false);
  const location = useLocation();
  const webSession = useWebSession();
  const isE2E = typeof window !== 'undefined' && window.localStorage.getItem('e2e_test') === 'true';
  const isAuthEnabled = !!(AUTH_CLIENT_ID && AUTH_AUTHORITY);
  const isAuthenticated = isConfidentialOidcClient
    ? webSession.isAuthenticated
    : auth.isAuthenticated;
  const shouldAttemptLogin =
    isAuthEnabled &&
    !hasAttemptedLogin &&
    !hasAuthParams() &&
    !isAuthenticated &&
    !auth?.activeNavigator &&
    !(isConfidentialOidcClient ? webSession.isLoading : auth.isLoading) &&
    !isE2E;

  useEffect(() => {
    if (isConfidentialOidcClient && shouldAttemptLogin) {
      const returnTo = `${location.pathname}${location.search}`;
      window.location.assign(`/apis/auth/v2/login?return_to=${encodeURIComponent(returnTo)}`);
      setHasAttemptedLogin(true);
      return;
    }
    if (shouldAttemptLogin) {
      auth.signinRedirect({
        state: {
          path: location.pathname,
          search: location.search,
        },
      });
      setHasAttemptedLogin(true);
    }
  }, [auth, location, shouldAttemptLogin]);

  // Hide the UI when auth is enabled but the user is not authenticated and we're not handling a callback
  const isAuthPending =
    isAuthEnabled &&
    (isConfidentialOidcClient
      ? webSession.isLoading || !webSession.isAuthenticated
      : !auth.isAuthenticated) &&
    !hasAuthParams() &&
    !isE2E;

  return { isAuthPending };
};
