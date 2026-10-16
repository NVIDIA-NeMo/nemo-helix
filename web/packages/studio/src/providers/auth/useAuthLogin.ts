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
 * @returns auth state for blocking pending auth, visible auth errors, explicit sign-out, and manual sign-in
 */
export interface AuthAutoLoginState {
  isAuthError: boolean;
  isAuthPending: boolean;
  isSignedOut: boolean;
  signIn: () => void;
}

export const AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY =
  'nemo.studio.auth.serverSessionLoginAttempt';
const SERVER_SESSION_LOGIN_RETRY_SUPPRESSION_MS = 60_000;

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

const clearServerSessionLoginAttempt = (): void => {
  window.sessionStorage.removeItem(AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY);
};

const markServerSessionLoginAttempt = (): void => {
  window.sessionStorage.setItem(AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY, String(Date.now()));
};

const hasRecentServerSessionLoginAttempt = (): boolean => {
  const attemptedAt = Number(
    window.sessionStorage.getItem(AUTH_SERVER_SESSION_LOGIN_ATTEMPT_STORAGE_KEY)
  );
  if (!Number.isFinite(attemptedAt) || attemptedAt <= 0) {
    clearServerSessionLoginAttempt();
    return false;
  }
  if (Date.now() - attemptedAt > SERVER_SESSION_LOGIN_RETRY_SUPPRESSION_MS) {
    clearServerSessionLoginAttempt();
    return false;
  }
  return true;
};

export const useAuthAutoLogin = (): AuthAutoLoginState => {
  const auth = useAuth();
  const [hasAttemptedLogin, setHasAttemptedLogin] = useState(false);
  const [hasRecentServerSessionLoginAttemptState, setHasRecentServerSessionLoginAttemptState] =
    useState(hasRecentServerSessionLoginAttempt);
  const [isExplicitLogoutSuppressed, setIsExplicitLogoutSuppressed] = useState(
    isExplicitLogoutAutoLoginSuppressed
  );
  const location = useLocation();
  const webSession = useWebSession();
  const hasOidcAuthParams = hasAuthParams();
  const isE2E = typeof window !== 'undefined' && window.localStorage.getItem('e2e_test') === 'true';
  const isDirectPublicAuthEnabled = Boolean(AUTH_CLIENT_ID && AUTH_AUTHORITY);
  const isAuthEnabled = webSession.isServerSession
    ? webSession.authEnabled
    : isDirectPublicAuthEnabled;
  const isAuthenticated = webSession.isServerSession
    ? webSession.isAuthenticated
    : auth.isAuthenticated;
  const isAuthenticationLoading =
    webSession.isLoading || (!webSession.isServerSession && auth.isLoading);
  const shouldSuppressServerSessionLogin = webSession.isServerSession && isExplicitLogoutSuppressed;
  const shouldBlockServerSessionLoginRetry =
    webSession.isServerSession &&
    hasRecentServerSessionLoginAttemptState &&
    !isAuthenticated &&
    !isAuthenticationLoading &&
    !webSession.isError;
  const shouldAttemptLogin =
    isAuthEnabled &&
    !shouldSuppressServerSessionLogin &&
    !shouldBlockServerSessionLoginRetry &&
    !hasAttemptedLogin &&
    !hasOidcAuthParams &&
    !isAuthenticated &&
    !auth?.activeNavigator &&
    !isAuthenticationLoading &&
    !webSession.isError &&
    !isE2E;

  const redirectToServerSessionLogin = useCallback(
    (serverSessionClient: string) => {
      markServerSessionLoginAttempt();
      setHasRecentServerSessionLoginAttemptState(true);
      setHasAttemptedLogin(true);
      window.location.assign(
        getServerSessionLoginUrl(
          serverSessionClient,
          location.pathname,
          location.search,
          location.hash
        )
      );
    },
    [location.hash, location.pathname, location.search]
  );

  const signIn = useCallback(() => {
    clearExplicitLogoutAutoLoginSuppression();
    setIsExplicitLogoutSuppressed(false);

    if (webSession.serverSessionClient) {
      redirectToServerSessionLogin(webSession.serverSessionClient);
      return;
    }

    setHasAttemptedLogin(true);
    void auth.signinRedirect({
      state: {
        path: location.pathname,
        search: location.search,
      },
    });
  }, [
    auth,
    location.pathname,
    location.search,
    redirectToServerSessionLogin,
    webSession.serverSessionClient,
  ]);

  useEffect(() => {
    if (webSession.serverSessionClient && shouldAttemptLogin) {
      redirectToServerSessionLogin(webSession.serverSessionClient);
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
  }, [
    auth,
    location.pathname,
    location.search,
    redirectToServerSessionLogin,
    shouldAttemptLogin,
    webSession.serverSessionClient,
  ]);

  useEffect(() => {
    if (
      hasRecentServerSessionLoginAttemptState &&
      (hasOidcAuthParams ||
        isAuthenticated ||
        (!webSession.isLoading && !webSession.isServerSession))
    ) {
      clearServerSessionLoginAttempt();
      setHasRecentServerSessionLoginAttemptState(false);
    }
  }, [
    hasOidcAuthParams,
    hasRecentServerSessionLoginAttemptState,
    isAuthenticated,
    webSession.isLoading,
    webSession.isServerSession,
  ]);

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

  const shouldBlockForAuth = !isE2E && !hasOidcAuthParams && !shouldSuppressServerSessionLogin;
  const isAuthError =
    shouldBlockForAuth && (webSession.isError || shouldBlockServerSessionLoginRetry);

  // Hide the UI when auth is enabled but the user is not authenticated and we're not handling a callback
  const isAuthPending =
    shouldBlockForAuth &&
    !isAuthError &&
    (webSession.isLoading || (isAuthEnabled && !isAuthenticated));

  return { isAuthError, isAuthPending, isSignedOut: shouldSuppressServerSessionLogin, signIn };
};
