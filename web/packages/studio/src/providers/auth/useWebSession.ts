// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  authLogout,
  getAuthSessionQueryKey,
  useAuthSession,
} from '@nemo/sdk/generated/platform/authentication';
import { BASE_URL, OIDC_TOKEN_ENDPOINT_AUTH_METHOD } from '@studio/constants/environment';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { useAuth } from 'react-oidc-context';

export const isConfidentialOidcClient = OIDC_TOKEN_ENDPOINT_AUTH_METHOD === 'client_secret_basic';

export type WebSession = {
  id: string;
  email?: string | null;
  groups: string[];
  account_id: string;
  authz_aliases: string[];
};

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string');

const parseWebSession = (value: unknown): WebSession | undefined => {
  if (typeof value !== 'object' || value === null) return undefined;

  const session = value as Record<string, unknown>;
  if (
    typeof session.id !== 'string' ||
    typeof session.account_id !== 'string' ||
    (session.email != null && typeof session.email !== 'string') ||
    !isStringArray(session.groups) ||
    !isStringArray(session.authz_aliases)
  ) {
    return undefined;
  }

  return {
    id: session.id,
    email: session.email,
    groups: session.groups,
    account_id: session.account_id,
    authz_aliases: session.authz_aliases,
  };
};

type WebSessionState = {
  session: WebSession | undefined;
  isAuthenticated: boolean;
  isLoading: boolean;
};

const usePublicWebSession = (): WebSessionState => ({
  session: undefined,
  isAuthenticated: false,
  isLoading: false,
});

const useConfidentialWebSession = (): WebSessionState => {
  const query = useAuthSession({
    query: {
      retry: false,
      staleTime: 60_000,
    },
  });
  const session = parseWebSession(query.data);

  return {
    session,
    isAuthenticated: session !== undefined,
    isLoading: query.isPending,
  };
};

const useConfiguredWebSession = isConfidentialOidcClient
  ? useConfidentialWebSession
  : usePublicWebSession;

export const useWebSession = (): WebSessionState => useConfiguredWebSession();

const usePublicAuthSignOut = (): (() => Promise<void>) => {
  const auth = useAuth();

  return useCallback(async () => auth.signoutRedirect(), [auth]);
};

const useConfidentialAuthSignOut = (): (() => Promise<void>) => {
  const queryClient = useQueryClient();

  return useCallback(async () => {
    await authLogout();
    queryClient.removeQueries({ queryKey: getAuthSessionQueryKey() });
    window.location.assign(BASE_URL || '/');
  }, [queryClient]);
};

const useConfiguredAuthSignOut = isConfidentialOidcClient
  ? useConfidentialAuthSignOut
  : usePublicAuthSignOut;

export const useAuthSignOut = (): (() => Promise<void>) => useConfiguredAuthSignOut();
