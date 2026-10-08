// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AUTH_AUTHORITY,
  AUTH_CLIENT_ID,
  BASE_URL,
  PLATFORM_BASE_URL,
} from '@studio/constants/environment';
import {
  clearExplicitLogoutAutoLoginSuppression,
  isExplicitLogoutAutoLoginSuppressed,
  useWebSession,
} from '@studio/providers/auth/useWebSession';
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
  const [isExplicitLogoutSuppressed, setIsExplicitLogoutSuppressed] = useState(
    isExplicitLogoutAutoLoginSuppressed
  );
  const location = useLocation();
  const webSession = useWebSession();
  const hasOidcAuthParams = hasAuthParams();
  const isE2E = typeof window !== 'undefined' && window.localStorage.getItem('e2e_test') === 'true';
  const isDirectPublicAuthEnabled =
    webSession.authEnabled && Boolean(AUTH_CLIENT_ID && AUTH_AUTHORITY);
  const isAuthEnabled = webSession.isServerSession
    ? webSession.authEnabled
    : isDirectPublicAuthEnabled;
  const isAuthenticated = webSession.isServerSession
    ? webSession.isAuthenticated
    : auth.isAuthenticated;
  const isAuthenticationLoading =
    webSession.isLoading || (!webSession.isServerSession && auth.isLoading);
  const shouldSuppressServerSessionLogin = webSession.isServerSession && isExplicitLogoutSuppressed;
  const shouldAttemptLogin =
    isAuthEnabled &&
    !shouldSuppressServerSessionLogin &&
    !hasAttemptedLogin &&
    !hasOidcAuthParams &&
    !isAuthenticated &&
    !auth?.activeNavigator &&
    !isAuthenticationLoading &&
    !webSession.isError &&
    !isE2E;

  useEffect(() => {
    if (webSession.serverSessionClient && shouldAttemptLogin) {
      const studioBasePath = BASE_URL.replace(/\/+$/, '');
      const returnTo = `${studioBasePath}${location.pathname}${location.search}${location.hash}`;
      const params = new URLSearchParams({
        client: webSession.serverSessionClient,
        return_to: returnTo,
      });
      window.location.assign(`${PLATFORM_BASE_URL}/apis/auth/v2/login?${params.toString()}`);
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
  }, [auth, location, shouldAttemptLogin, webSession.serverSessionClient]);

  useEffect(() => {
    if (
      isExplicitLogoutSuppressed &&
      (hasOidcAuthParams ||
        isAuthenticated ||
        (!webSession.isLoading && !webSession.isServerSession))
    ) {
      clearExplicitLogoutAutoLoginSuppression();
      setIsExplicitLogoutSuppressed(false);
    }
  }, [
    hasOidcAuthParams,
    isAuthenticated,
    isExplicitLogoutSuppressed,
    webSession.isLoading,
    webSession.isServerSession,
  ]);

  // Hide the UI when auth is enabled but the user is not authenticated and we're not handling a callback
  const isAuthPending =
    !isE2E &&
    !hasOidcAuthParams &&
    !shouldSuppressServerSessionLogin &&
    (webSession.isLoading || webSession.isError || (isAuthEnabled && !isAuthenticated));

  return { isAuthPending };
};
