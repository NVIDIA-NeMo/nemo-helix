// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getOidcBrowserAuthentication,
  type OidcBrowserAuthentication,
  type OidcServerSessionClient,
  withOidcServerSessionRequest,
} from '@nemo/sdk/src/utils/oidcBrowserAuthentication';
import { BASE_URL, PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { useAuth } from 'react-oidc-context';

export interface WebSession {
  id: string;
  email?: string | null;
  groups: string[];
  account_id: string;
  authz_aliases: string[];
}

interface WebSessionState {
  session: WebSession | undefined;
  serverSessionClient: OidcServerSessionClient | undefined;
  isServerSession: boolean;
  isAuthenticated: boolean;
  isLoading: boolean;
  isError: boolean;
  authEnabled: boolean;
}

export const AUTH_DISCOVERY_QUERY_KEY = ['auth', 'discovery'] as const;
export const WEB_SESSION_QUERY_KEY = ['auth', 'web-session'] as const;
export const AUTH_EXPLICIT_LOGOUT_STORAGE_KEY = 'nemo.studio.auth.explicitLogout';
const AUTH_QUERY_RETRY_COUNT = 2;

export const markExplicitLogoutAutoLoginSuppressed = (): void => {
  window.sessionStorage.setItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY, 'true');
};

export const clearExplicitLogoutAutoLoginSuppression = (): void => {
  window.sessionStorage.removeItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY);
};

export const isExplicitLogoutAutoLoginSuppressed = (): boolean =>
  window.sessionStorage.getItem(AUTH_EXPLICIT_LOGOUT_STORAGE_KEY) === 'true';

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string');

const parseWebSession = (value: unknown): WebSession => {
  if (typeof value !== 'object' || value === null) {
    throw new Error('Web session response was invalid');
  }

  const session = value as Record<string, unknown>;
  if (
    typeof session.id !== 'string' ||
    typeof session.account_id !== 'string' ||
    (session.email != null && typeof session.email !== 'string') ||
    !isStringArray(session.groups) ||
    !isStringArray(session.authz_aliases)
  ) {
    throw new Error('Web session response was invalid');
  }

  return {
    id: session.id,
    email: session.email,
    groups: session.groups,
    account_id: session.account_id,
    authz_aliases: session.authz_aliases,
  };
};

const getWebSession = async (
  authentication: OidcBrowserAuthentication
): Promise<WebSession | null> => {
  const request = await withOidcServerSessionRequest(undefined, authentication);
  const response = await fetch(`${PLATFORM_BASE_URL}/apis/auth/v2/session`, request);
  if (response.status === 401) return null;
  if (!response.ok) {
    throw new Error(`Web session lookup failed with status ${response.status}`);
  }
  return parseWebSession(await response.json());
};

const logoutWebSession = async (authentication: OidcBrowserAuthentication): Promise<void> => {
  const request = await withOidcServerSessionRequest({ method: 'POST' }, authentication);
  const response = await fetch(`${PLATFORM_BASE_URL}/apis/auth/v2/logout`, request);
  if (!response.ok) {
    throw new Error(`Web session logout failed with status ${response.status}`);
  }
};

const useBrowserAuthentication = () =>
  useQuery({
    queryKey: AUTH_DISCOVERY_QUERY_KEY,
    queryFn: getOidcBrowserAuthentication,
    staleTime: Infinity,
    retry: AUTH_QUERY_RETRY_COUNT,
  });

export const useWebSession = (): WebSessionState => {
  const discoveryQuery = useBrowserAuthentication();
  const authentication = discoveryQuery.data;
  const serverSessionClient = authentication?.serverSessionClient;
  const isServerSession = serverSessionClient !== undefined;
  const sessionQuery = useQuery({
    queryKey: [...WEB_SESSION_QUERY_KEY, serverSessionClient],
    queryFn: () => {
      if (!authentication) throw new Error('OIDC discovery was unavailable');
      return getWebSession(authentication);
    },
    enabled: isServerSession,
    retry: AUTH_QUERY_RETRY_COUNT,
    staleTime: 60_000,
  });
  const session = isServerSession ? (sessionQuery.data ?? undefined) : undefined;

  return {
    session,
    serverSessionClient,
    isServerSession,
    isAuthenticated: session !== undefined,
    isLoading: discoveryQuery.isPending || (isServerSession && sessionQuery.isPending),
    isError: discoveryQuery.isError || (isServerSession && sessionQuery.isError),
    authEnabled: authentication?.authEnabled ?? false,
  };
};

export const useAuthSignOut = (): (() => Promise<void>) => {
  const auth = useAuth();
  const queryClient = useQueryClient();
  const discoveryQuery = useBrowserAuthentication();
  const authentication = discoveryQuery.data;
  const toast = useToast();

  return useCallback(async () => {
    if (!authentication?.serverSessionClient) {
      await auth.signoutRedirect();
      return;
    }
    try {
      await logoutWebSession(authentication);
    } catch {
      toast.error('Unable to sign out. Please try again.');
      return;
    }
    markExplicitLogoutAutoLoginSuppressed();
    queryClient.removeQueries({ queryKey: WEB_SESSION_QUERY_KEY });
    window.location.assign(BASE_URL || '/');
  }, [auth, authentication, queryClient, toast]);
};
