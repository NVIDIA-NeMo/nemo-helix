// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getConfidentialOidcBearerToken,
  isConfidentialOidcMode,
} from '@nemo/sdk/src/utils/confidentialOidcToken';
import { AUTH_AUTHORITY, AUTH_CLIENT_ID, AUTH_MODE } from '@studio/constants/environment';
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
  const [hasConfidentialSession, setHasConfidentialSession] = useState(false);
  const location = useLocation();
  const isE2E = typeof window !== 'undefined' && window.localStorage.getItem('e2e_test') === 'true';
  const isConfidentialAuthEnabled = isConfidentialOidcMode(AUTH_MODE);
  const isAuthEnabled = !!(AUTH_CLIENT_ID && AUTH_AUTHORITY) || isConfidentialAuthEnabled;
  const shouldAttemptLogin =
    isAuthEnabled &&
    !isConfidentialAuthEnabled &&
    !hasAttemptedLogin &&
    !hasAuthParams() &&
    !auth?.isAuthenticated &&
    !auth?.activeNavigator &&
    !auth?.isLoading &&
    !isE2E;

  useEffect(() => {
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

  useEffect(() => {
    if (!isConfidentialAuthEnabled || isE2E || hasConfidentialSession || hasAttemptedLogin) return;
    setHasAttemptedLogin(true);
    getConfidentialOidcBearerToken().then((token) => {
      if (token) {
        setHasConfidentialSession(true);
      }
    });
  }, [hasAttemptedLogin, hasConfidentialSession, isConfidentialAuthEnabled, isE2E]);

  // Hide the UI when auth is enabled but the user is not authenticated and we're not handling a callback
  const isAuthPending =
    isAuthEnabled &&
    !hasAuthParams() &&
    !isE2E &&
    (isConfidentialAuthEnabled ? !hasConfidentialSession : !auth?.isAuthenticated);

  return { isAuthPending };
};
