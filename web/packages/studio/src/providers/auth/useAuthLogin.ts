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
import { useCallback, useEffect, useState } from 'react';
import { hasAuthParams, useAuth } from 'react-oidc-context';
import { useLocation } from 'react-router';

/**
 * This hook is used to automatically handle the login process.
 *
 * Mirrors logic in useAutoSignin that can be enabled when the AUTH_CLIENT_ID and AUTH_AUTHORITY env vars are set.
 *
 * @returns `isAuthPending` - true when auth is enabled and the user is not yet authenticated (UI should be hidden)
 */
export interface AuthAutoLoginState {
  isAuthPending: boolean;
  isSignedOut: boolean;
  signIn: () => void;
}

const getServerSessionReturnTo = (pathname: string, search: string, hash: string): string => {
  const studioBasePath = BASE_URL.replace(/\/+$/, '');
  return `${studioBasePath}${pathname}${search}${hash}`;
};

const getServerSessionLoginUrl = (
  serverSessionClient: string,
  pathname: string,
  search: string,
  hash: string
): string => {
  const params = new URLSearchParams({
    client: serverSessionClient,
    return_to: getServerSessionReturnTo(pathname, search, hash),
  });
  return `${PLATFORM_BASE_URL}/apis/auth/v2/login?${params.toString()}`;
};

export const useAuthAutoLogin = (): AuthAutoLoginState => {
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

  const signIn = useCallback(() => {
    clearExplicitLogoutAutoLoginSuppression();
    setIsExplicitLogoutSuppressed(false);
    setHasAttemptedLogin(true);

    if (webSession.serverSessionClient) {
      window.location.assign(
        getServerSessionLoginUrl(
          webSession.serverSessionClient,
          location.pathname,
          location.search,
          location.hash
        )
      );
      return;
    }

    void auth.signinRedirect({
      state: {
        path: location.pathname,
        search: location.search,
      },
    });
  }, [auth, location.hash, location.pathname, location.search, webSession.serverSessionClient]);

  useEffect(() => {
    if (webSession.serverSessionClient && shouldAttemptLogin) {
      window.location.assign(
        getServerSessionLoginUrl(
          webSession.serverSessionClient,
          location.pathname,
          location.search,
          location.hash
        )
      );
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

  return { isAuthPending, isSignedOut: shouldSuppressServerSessionLogin, signIn };
};
