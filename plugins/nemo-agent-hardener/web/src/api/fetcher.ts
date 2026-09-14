// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Orval mutator for the generated agent-hardener client. Hand-written, and kept
// outside `src/generated/` because `clean: true` empties that folder on
// every `pnpm gen`.
//
// Studio's own fetchers read the OIDC user out of localStorage. A plugin must
// not: only the access token crosses the plugin boundary, so `Root` installs
// the host's token getter here at mount and every request reads it fresh
// (getAccessToken returns the current token after silent renew).

import axios, {
  type AxiosError,
  type AxiosRequestConfig,
  type AxiosResponse,
} from 'axios';

interface RequestOptions extends AxiosRequestConfig {
  params?: Record<string, string | number | boolean | object | unknown>;
}

interface ClientConfig {
  getAccessToken: () => string;
  baseUrl: string;
}

let config: ClientConfig | null = null;

/** Installed by `Root` from the host handle before any request is issued. */
export const configureClient = (next: ClientConfig): void => {
  config = next;
};

/** Absolute URL for a plugin-owned endpoint not covered by the generated client. */
export const apiUrl = (path: string): string => `${config?.baseUrl ?? ''}${path}`;

/** Current access token, for callers that build their own request. */
export const authHeader = (): Record<string, string> => {
  const token = config?.getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

export const customFetch = async <TData>(request: RequestOptions): Promise<TData> => {
  const response: AxiosResponse<TData> = await axios({
    ...request,
    url: apiUrl(request.url ?? ''),
    headers: {
      'X-Source': 'NeMo Studio',
      ...authHeader(),
      ...request.headers,
    },
    paramsSerializer: { indexes: null },
  });
  return response.data;
};

// https://orval.dev/reference/configuration/output#mutator
export type ErrorType<TError> = AxiosError<TError>;
