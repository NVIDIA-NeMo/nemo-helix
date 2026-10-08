// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AxiosRequestConfig } from 'axios';

import { getStoredOidcBearerToken } from './oidcBearerToken';
import {
  getOidcBrowserAuthentication,
  OidcBrowserAuthenticationError,
  type OidcBrowserAuthentication,
  STUDIO_SOURCE_HEADER,
  STUDIO_SOURCE_VALUE,
} from './oidcBrowserAuthentication';
import { resolveBrowserBaseUrl } from './url';

interface OidcUserStorage {
  getItem(key: string): string | null;
  removeItem(key: string): void;
}

export interface HelixBrowserAuthOptions {
  authentication?: OidcBrowserAuthentication;
  storage?: OidcUserStorage;
  onMalformedStorage?: () => void;
}

const AUTH_DISCOVERY_RETRY_COUNT = 2;

const defaultMalformedStorageWarning = (): void => {
  console.warn('Malformed OIDC storage entry detected. Clearing storage and re-authenticating.');
};

const getBrowserStorage = (storage: OidcUserStorage | undefined): OidcUserStorage | undefined => {
  if (storage) return storage;
  return typeof localStorage === 'undefined' ? undefined : localStorage;
};

const isTransientDiscoveryError = (error: unknown): boolean => {
  if (error instanceof OidcBrowserAuthenticationError) {
    return error.status === undefined || error.status >= 500;
  }
  return error instanceof TypeError;
};

const getBrowserAuthentication = async (
  authentication: OidcBrowserAuthentication | undefined
): Promise<OidcBrowserAuthentication | undefined> => {
  if (authentication) return authentication;

  for (let attempt = 0; attempt <= AUTH_DISCOVERY_RETRY_COUNT; attempt += 1) {
    try {
      return await getOidcBrowserAuthentication();
    } catch (error) {
      if (!isTransientDiscoveryError(error) || attempt === AUTH_DISCOVERY_RETRY_COUNT) {
        return undefined;
      }
    }
  }

  return undefined;
};

const getBrowserBearerToken = ({
  storage,
  onMalformedStorage,
}: HelixBrowserAuthOptions): string | undefined =>
  getStoredOidcBearerToken({
    authority: import.meta.env.VITE_AUTH_AUTHORITY,
    clientId: import.meta.env.VITE_AUTH_CLIENT_ID,
    configuredSource: import.meta.env.VITE_AUTH_BEARER_TOKEN_SOURCE,
    storage: getBrowserStorage(storage),
    onMalformedStorage: onMalformedStorage ?? defaultMalformedStorageWarning,
  });

export const withHelixBrowserAuthRequest = async (
  init?: RequestInit,
  options: HelixBrowserAuthOptions = {}
): Promise<RequestInit> => {
  const authentication = await getBrowserAuthentication(options.authentication);
  const headers = new Headers(init?.headers);
  headers.set(STUDIO_SOURCE_HEADER, STUDIO_SOURCE_VALUE);

  if (authentication?.serverSessionClient) {
    headers.delete('Authorization');
    return { ...init, credentials: 'include', headers };
  }

  if (!headers.has('Authorization')) {
    const bearerToken = getBrowserBearerToken(options);
    if (bearerToken) headers.set('Authorization', `Bearer ${bearerToken}`);
  }

  return { ...init, headers };
};

const isHeaderSetter = (
  headers: unknown
): headers is { set: (name: string, value: string) => void } =>
  typeof (headers as { set?: unknown }).set === 'function';

const isHeaderDeleter = (headers: unknown): headers is { delete: (name: string) => void } =>
  typeof (headers as { delete?: unknown }).delete === 'function';

const isHeaderChecker = (headers: unknown): headers is { has: (name: string) => boolean } =>
  typeof (headers as { has?: unknown }).has === 'function';

const setAxiosHeader = (
  headers: NonNullable<AxiosRequestConfig['headers']>,
  name: string,
  value: string
): void => {
  if (isHeaderSetter(headers)) {
    headers.set(name, value);
    return;
  }

  (headers as Record<string, unknown>)[name] = value;
};

const deleteAxiosHeader = (
  headers: NonNullable<AxiosRequestConfig['headers']>,
  name: string
): void => {
  if (isHeaderDeleter(headers)) {
    headers.delete(name);
    return;
  }

  const normalized = name.toLowerCase();
  Object.keys(headers).forEach((headerName) => {
    if (headerName.toLowerCase() === normalized) {
      delete (headers as Record<string, unknown>)[headerName];
    }
  });
};

const hasAxiosHeader = (
  headers: NonNullable<AxiosRequestConfig['headers']>,
  name: string
): boolean => {
  if (isHeaderChecker(headers)) return headers.has(name);

  const normalized = name.toLowerCase();
  return Object.keys(headers).some((headerName) => {
    if (headerName.toLowerCase() !== normalized) return false;
    const value = (headers as Record<string, unknown>)[headerName];
    return value !== undefined && value !== null;
  });
};

const ensureAxiosHeaders = <TConfig extends AxiosRequestConfig>(
  config: TConfig
): NonNullable<AxiosRequestConfig['headers']> => {
  config.headers ??= {};
  return config.headers;
};

export const withHelixBrowserAuthAxios = async <TConfig extends AxiosRequestConfig>(
  config: TConfig,
  options: HelixBrowserAuthOptions = {}
): Promise<TConfig> => {
  const authentication = await getBrowserAuthentication(options.authentication);
  const headers = ensureAxiosHeaders(config);
  setAxiosHeader(headers, STUDIO_SOURCE_HEADER, STUDIO_SOURCE_VALUE);

  if (authentication?.serverSessionClient) {
    deleteAxiosHeader(headers, 'Authorization');
    config.withCredentials = true;
    return config;
  }

  if (!hasAxiosHeader(headers, 'Authorization')) {
    const bearerToken = getBrowserBearerToken(options);
    if (bearerToken) setAxiosHeader(headers, 'Authorization', `Bearer ${bearerToken}`);
  }

  return config;
};

const getHelixBaseUrl = (): string =>
  resolveBrowserBaseUrl(import.meta.env.VITE_PLATFORM_BASE_URL).replace(/\/+$/, '');

const isRequest = (input: RequestInfo | URL): input is Request =>
  typeof Request !== 'undefined' && input instanceof Request;

const isWithinBasePath = (pathname: string, basePathname: string): boolean => {
  const normalizedBasePath = basePathname.replace(/\/+$/, '');
  return (
    normalizedBasePath === '' ||
    pathname === normalizedBasePath ||
    pathname.startsWith(`${normalizedBasePath}/`)
  );
};

const isHelixUrl = (input: RequestInfo | URL, baseUrl: string): boolean => {
  if (typeof input === 'string') {
    if (input.startsWith('/')) return true;
    try {
      const url = new URL(input);
      const base = new URL(baseUrl);
      return url.origin === base.origin && isWithinBasePath(url.pathname, base.pathname);
    } catch {
      return false;
    }
  }
  if (input instanceof URL) {
    try {
      const base = new URL(baseUrl);
      return input.origin === base.origin && isWithinBasePath(input.pathname, base.pathname);
    } catch {
      return false;
    }
  }
  if (isRequest(input)) {
    return isHelixUrl(input.url, baseUrl);
  }
  return false;
};

const resolveHelixUrl = (input: RequestInfo | URL, baseUrl: string): RequestInfo | URL => {
  if (typeof input !== 'string') return input;
  if (!input.startsWith('/')) return input;

  return `${baseUrl}${input}`;
};

export const platformFetch = async (
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> => {
  const baseUrl = getHelixBaseUrl();
  if (!isHelixUrl(input, baseUrl)) {
    return fetch(input, init);
  }
  const requestInit =
    isRequest(input) && init?.headers === undefined ? { ...init, headers: input.headers } : init;
  return fetch(resolveHelixUrl(input, baseUrl), await withHelixBrowserAuthRequest(requestInit));
};
