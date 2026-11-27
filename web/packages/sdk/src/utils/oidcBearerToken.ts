// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { User } from 'oidc-client-ts';

export type OidcBearerTokenSource = 'access_token';

interface OidcUserStorage {
  getItem(key: string): string | null;
  removeItem(key: string): void;
}

const parseBearerTokenSource = (
  configuredSource: string | undefined
): OidcBearerTokenSource | undefined => {
  if (!configuredSource) return 'access_token';
  if (configuredSource === 'access_token') {
    return configuredSource;
  }
  return undefined;
};

export const selectOidcBearerToken = (
  user: User | null | undefined,
  configuredSource?: string
): string | undefined => {
  if (!user) return undefined;

  const source = parseBearerTokenSource(configuredSource);
  if (!source) return undefined;
  if (user.expired) return undefined;

  const token = user.access_token;
  if (!token) return undefined;
  return token;
};

export const getStoredOidcBearerToken = ({
  authority,
  clientId,
  configuredSource,
  storage,
  onMalformedStorage,
}: {
  authority: string | undefined;
  clientId: string | undefined;
  configuredSource?: string;
  storage: OidcUserStorage | undefined;
  onMalformedStorage?: () => void;
}): string | undefined => {
  if (!authority || !clientId || !storage) return undefined;

  const oidcStorageKey = `oidc.user:${authority}:${clientId}`;
  const oidcStorage = storage.getItem(oidcStorageKey);
  if (!oidcStorage) return undefined;

  try {
    return selectOidcBearerToken(User.fromStorageString(oidcStorage), configuredSource);
  } catch {
    storage.removeItem(oidcStorageKey);
    onMalformedStorage?.();
    return undefined;
  }
};
