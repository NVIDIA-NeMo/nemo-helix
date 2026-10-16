// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { resolveBrowserBaseUrl } from './url';

export type OidcServerSessionClient = 'public' | 'confidential';

export const STUDIO_SOURCE_HEADER = 'X-Source';
export const STUDIO_SOURCE_VALUE = 'NeMo Studio';
const AUTH_DISCOVERY_TIMEOUT_MS = 5_000;

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

export class OidcBrowserAuthenticationTimeoutError extends OidcBrowserAuthenticationError {
  constructor(timeoutMs: number, cause: unknown) {
    super(`Auth discovery timed out after ${timeoutMs}ms`);
    this.cause = cause;
    this.name = 'OidcBrowserAuthenticationTimeoutError';
  }
}

export const isRetryableOidcBrowserAuthenticationError = (error: unknown): boolean => {
  if (error instanceof OidcBrowserAuthenticationTimeoutError) return false;
  if (error instanceof OidcBrowserAuthenticationError) {
    return error.status === undefined || error.status >= 500;
  }
  return error instanceof TypeError;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const hasClientId = (value: unknown, clientId: string | undefined): boolean =>
  clientId !== undefined && clientId !== '' && isRecord(value) && value.client_id === clientId;

const isDefaultClient = (value: unknown): boolean => isRecord(value) && value.default === true;

const isConfidentialClient = (value: unknown): boolean =>
  isRecord(value) &&
  value.name === 'confidential' &&
  value.client_authentication === 'client_secret_basic';

const isServerSessionPublicClient = (value: unknown): boolean =>
  isRecord(value) &&
  value.name === 'public' &&
  value.client_authentication === 'public' &&
  value.server_side_sessions === true;

const serverSessionAuthenticationForClient = (
  client: unknown
): OidcBrowserAuthentication | undefined => {
  if (isConfidentialClient(client)) {
    return { authEnabled: true, serverSessionClient: 'confidential' };
  }
  if (isServerSessionPublicClient(client)) {
    return { authEnabled: true, serverSessionClient: 'public' };
  }
  return undefined;
};

export const parseOidcBrowserAuthentication = (
  value: unknown,
  preferredClientId?: string
): OidcBrowserAuthentication => {
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
  const selectedClient =
    clients.find((client) => hasClientId(client, preferredClientId)) ??
    clients.find(isDefaultClient) ??
    (clients.length === 1 ? clients[0] : undefined);
  const serverSessionAuthentication = serverSessionAuthenticationForClient(selectedClient);
  if (serverSessionAuthentication) return serverSessionAuthentication;

  return { authEnabled: true };
};

const fetchDiscovery = async (url: string): Promise<Response> => {
  try {
    return await fetch(url, {
      credentials: 'omit',
      signal: AbortSignal.timeout(AUTH_DISCOVERY_TIMEOUT_MS),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      throw new OidcBrowserAuthenticationTimeoutError(AUTH_DISCOVERY_TIMEOUT_MS, error);
    }
    throw error;
  }
};

const discoverOidcBrowserAuthentication = async (): Promise<OidcBrowserAuthentication> => {
  const baseUrl = resolveBrowserBaseUrl(import.meta.env.VITE_PLATFORM_BASE_URL);
  const response = await fetchDiscovery(`${baseUrl}/apis/auth/discovery`);
  if (response.status === 404) {
    return { authEnabled: false };
  }
  if (!response.ok) {
    throw new OidcBrowserAuthenticationError(
      `Auth discovery failed with status ${response.status}`,
      response.status
    );
  }
  return parseOidcBrowserAuthentication(await response.json(), import.meta.env.VITE_AUTH_CLIENT_ID);
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
