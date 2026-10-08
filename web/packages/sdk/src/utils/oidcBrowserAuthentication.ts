// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { resolveBrowserBaseUrl } from './url';

export type OidcServerSessionClient = 'public' | 'confidential';

export const STUDIO_SOURCE_HEADER = 'X-Source';
export const STUDIO_SOURCE_VALUE = 'NeMo Studio';

export interface OidcBrowserAuthentication {
  authEnabled: boolean;
  serverSessionClient?: OidcServerSessionClient;
}

export class OidcBrowserAuthenticationError extends Error {
  constructor(
    message: string,
    readonly status?: number
  ) {
    super(message);
    this.name = 'OidcBrowserAuthenticationError';
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const isConfidentialClient = (value: unknown): boolean =>
  isRecord(value) &&
  value.name === 'confidential' &&
  value.client_authentication === 'client_secret_basic';

const isServerSessionPublicClient = (value: unknown): boolean =>
  isRecord(value) &&
  value.name === 'public' &&
  value.client_authentication === 'public' &&
  value.server_side_sessions === true;

export const parseOidcBrowserAuthentication = (value: unknown): OidcBrowserAuthentication => {
  if (!isRecord(value) || typeof value.auth_enabled !== 'boolean') {
    throw new Error('Auth discovery response was invalid');
  }
  if (!value.auth_enabled || value.oidc == null) {
    return { authEnabled: value.auth_enabled };
  }
  if (!isRecord(value.oidc) || !Array.isArray(value.oidc.clients)) {
    throw new Error('OIDC discovery response was invalid');
  }

  const clients = value.oidc.clients;
  if (clients.some(isConfidentialClient)) {
    return { authEnabled: true, serverSessionClient: 'confidential' };
  }
  if (clients.some(isServerSessionPublicClient)) {
    return { authEnabled: true, serverSessionClient: 'public' };
  }
  return { authEnabled: true };
};

const discoverOidcBrowserAuthentication = async (): Promise<OidcBrowserAuthentication> => {
  const baseUrl = resolveBrowserBaseUrl(import.meta.env.VITE_PLATFORM_BASE_URL);
  const response = await fetch(`${baseUrl}/apis/auth/discovery`, { credentials: 'include' });
  if (!response.ok) {
    throw new OidcBrowserAuthenticationError(
      `Auth discovery failed with status ${response.status}`,
      response.status
    );
  }
  return parseOidcBrowserAuthentication(await response.json());
};

let discoveryRequest: Promise<OidcBrowserAuthentication> | undefined;

export const getOidcBrowserAuthentication = (): Promise<OidcBrowserAuthentication> => {
  discoveryRequest ??= discoverOidcBrowserAuthentication().catch((error: unknown) => {
    discoveryRequest = undefined;
    throw error;
  });
  return discoveryRequest;
};

export const withOidcServerSessionRequest = async (
  init?: RequestInit,
  authentication?: OidcBrowserAuthentication
): Promise<RequestInit | undefined> => {
  const resolvedAuthentication = authentication ?? (await getOidcBrowserAuthentication());
  if (!resolvedAuthentication.serverSessionClient) return init;

  const headers = new Headers(init?.headers);
  headers.delete('Authorization');
  headers.set(STUDIO_SOURCE_HEADER, STUDIO_SOURCE_VALUE);
  return { ...init, credentials: 'include', headers };
};
