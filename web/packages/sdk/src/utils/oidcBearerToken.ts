// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { User } from 'oidc-client-ts';

export type OidcBearerTokenSource = 'access_token' | 'id_token';

interface OidcUserStorage {
  getItem(key: string): string | null;
  removeItem(key: string): void;
}

const parseBearerTokenSource = (
  configuredSource: string | undefined
): OidcBearerTokenSource | undefined => {
  if (!configuredSource) return 'access_token';
  if (configuredSource === 'access_token' || configuredSource === 'id_token') {
    return configuredSource;
  }
  return undefined;
};

const isUnexpiredIdToken = (token: string): boolean => {
  const segments = token.split('.');
  if (segments.length !== 3 || !segments[1]) return false;

  try {
    const payload = segments[1].replace(/-/g, '+').replace(/_/g, '/');
    const paddedPayload = payload.padEnd(Math.ceil(payload.length / 4) * 4, '=');
    const claims = JSON.parse(globalThis.atob(paddedPayload)) as unknown;
    if (typeof claims !== 'object' || claims === null || !('exp' in claims)) return false;
    const expiresAt = (claims as { exp?: unknown }).exp;
    return typeof expiresAt === 'number' && expiresAt > Date.now() / 1000;
  } catch {
    return false;
  }
};

export const selectOidcBearerToken = (
  user: User | null | undefined,
  configuredSource?: string
): string | undefined => {
  if (!user || user.expired) return undefined;

  const source = parseBearerTokenSource(configuredSource);
  if (!source) return undefined;

  const token = user[source];
  if (!token) return undefined;
  if (source === 'id_token' && !isUnexpiredIdToken(token)) return undefined;
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
