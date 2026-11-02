// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { User } from 'oidc-client-ts';
import { getStoredOidcBearerToken, selectOidcBearerToken } from './oidcBearerToken';

const createUser = ({
  accessToken = 'access-token',
  idToken = 'id-token',
  expiresAt = Math.floor(Date.now() / 1000) + 3600,
}: {
  accessToken?: string;
  idToken?: string;
  expiresAt?: number;
} = {}): User =>
  User.fromStorageString(
    JSON.stringify({
      access_token: accessToken,
      expires_at: expiresAt,
      id_token: idToken,
      profile: { sub: 'user-1' },
      token_type: 'Bearer',
    })
  );

describe('selectOidcBearerToken', () => {
  it('does not return a token without a current OIDC user', () => {
    expect(selectOidcBearerToken(null)).toBeUndefined();
  });

  it('uses the access token by default and when explicitly configured', () => {
    expect(selectOidcBearerToken(createUser())).toBe('access-token');
    expect(selectOidcBearerToken(createUser(), 'access_token')).toBe('access-token');
  });

  it.each(['id_token', 'refresh_token'])('rejects the unsupported source %s', (source) => {
    expect(selectOidcBearerToken(createUser(), source)).toBeUndefined();
  });

  it('does not fall back to an ID token when the access token is missing', () => {
    expect(selectOidcBearerToken(createUser({ accessToken: '' }))).toBeUndefined();
  });

  it('does not fall back to an ID token when the access token has expired', () => {
    expect(
      selectOidcBearerToken(createUser({ expiresAt: Math.floor(Date.now() / 1000) - 1 }))
    ).toBeUndefined();
  });

  it('does not require an ID token for a valid access token', () => {
    expect(selectOidcBearerToken(createUser({ idToken: '' }))).toBe('access-token');
  });
});

describe('getStoredOidcBearerToken', () => {
  it('reads the current access token on every request, including after renewal', () => {
    const authority = 'https://auth.example.test';
    const clientId = 'MixedCaseClient';
    const key = `oidc.user:${authority}:${clientId}`;
    const storage = new Map<string, string>([[key, createUser().toStorageString()]]);
    const oidcStorage = {
      getItem: (storageKey: string) => storage.get(storageKey) ?? null,
      removeItem: (storageKey: string) => storage.delete(storageKey),
    };

    expect(getStoredOidcBearerToken({ authority, clientId, storage: oidcStorage })).toBe(
      'access-token'
    );
    storage.set(key, createUser({ accessToken: 'renewed-access' }).toStorageString());
    expect(getStoredOidcBearerToken({ authority, clientId, storage: oidcStorage })).toBe(
      'renewed-access'
    );
    expect(
      getStoredOidcBearerToken({
        authority,
        clientId,
        configuredSource: 'id_token',
        storage: oidcStorage,
      })
    ).toBeUndefined();
  });

  it('removes malformed storage and reports it without exposing the value', () => {
    const authority = 'https://auth.example.test';
    const clientId = 'studio-client';
    const key = `oidc.user:${authority}:${clientId}`;
    const storage = new Map<string, string>([[key, '{not-json']]);
    const onMalformedStorage = vi.fn();

    expect(
      getStoredOidcBearerToken({
        authority,
        clientId,
        storage: {
          getItem: (storageKey) => storage.get(storageKey) ?? null,
          removeItem: (storageKey) => storage.delete(storageKey),
        },
        onMalformedStorage,
      })
    ).toBeUndefined();
    expect(storage.has(key)).toBe(false);
    expect(onMalformedStorage).toHaveBeenCalledOnce();
  });
});
