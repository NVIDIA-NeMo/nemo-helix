// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

interface ConfidentialTokenResponse {
  readonly access_token?: string;
  readonly token_type?: string;
  readonly expires_at?: number;
}

let cachedToken: ConfidentialTokenResponse | undefined;
let pendingToken: Promise<ConfidentialTokenResponse | undefined> | undefined;

export const isConfidentialOidcMode = (mode: string | undefined): boolean => {
  const normalized = mode?.trim().toLowerCase();
  return normalized === 'true' || normalized === 'confidential';
};

const tokenIsFresh = (token: ConfidentialTokenResponse | undefined): boolean => {
  if (!token?.access_token) return false;
  if (!token.expires_at) return true;
  return token.expires_at - Math.floor(Date.now() / 1000) > 30;
};

export const getConfidentialOidcBearerToken = async (): Promise<string | undefined> => {
  if (tokenIsFresh(cachedToken)) return cachedToken?.access_token;

  pendingToken ??= fetch('/studio/auth/confidential/token', {
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
    .then(async (response) => {
      if (response.status === 401) {
        window.location.assign('/studio/auth/confidential/login');
        return undefined;
      }
      if (!response.ok) return undefined;
      return (await response.json()) as ConfidentialTokenResponse;
    })
    .finally(() => {
      pendingToken = undefined;
    });

  cachedToken = await pendingToken;
  return cachedToken?.access_token;
};
