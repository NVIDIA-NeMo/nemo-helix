// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { User } from 'oidc-client-ts';
import { getStoredOidcBearerToken, selectOidcBearerToken } from './oidcBearerToken';

const futureExpiry = (): number => Math.floor(Date.now() / 1000) + 3600;

const createIdToken = (expiresAt = futureExpiry()): string => {
  const encode = (value: object) =>
    globalThis
      .btoa(JSON.stringify(value))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  return `${encode({ alg: 'ES256', typ: 'JWT' })}.${encode({ exp: expiresAt, sub: 'user-1' })}.signature`;
};

const createUser = ({
  accessToken = 'access-token',
  idToken = createIdToken(),
  expiresAt = futureExpiry(),
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

  it('uses the access token by default to preserve existing providers', () => {
    expect(selectOidcBearerToken(createUser())).toBe('access-token');
  });

  it('selects the ID token when configured', () => {
    const idToken = createIdToken();
    expect(selectOidcBearerToken(createUser({ idToken }), 'id_token')).toBe(idToken);
  });

  it('does not fall back when the configured token is missing', () => {
    expect(selectOidcBearerToken(createUser({ idToken: '' }), 'id_token')).toBeUndefined();
  });

  it('fails closed for an unsupported configured source', () => {
    expect(selectOidcBearerToken(createUser(), 'refresh_token')).toBeUndefined();
  });

  it('does not return a token for an expired user', () => {
    expect(
      selectOidcBearerToken(
        createUser({ expiresAt: Math.floor(Date.now() / 1000) - 1 }),
        'id_token'
      )
    ).toBeUndefined();
  });

  it('does not return an expired ID token even if the access-token session is current', () => {
    const expiredIdToken = createIdToken(Math.floor(Date.now() / 1000) - 1);
    expect(
      selectOidcBearerToken(createUser({ idToken: expiredIdToken }), 'id_token')
    ).toBeUndefined();
  });

  it('does not return a malformed ID token', () => {
    expect(selectOidcBearerToken(createUser({ idToken: 'not-a-jwt' }), 'id_token')).toBeUndefined();
  });
});

describe('getStoredOidcBearerToken', () => {
  it('reads the current OIDC user on every request', () => {
    const authority = 'https://auth.example.test';
    const clientId = 'MixedCaseClient';
    const key = `oidc.user:${authority}:${clientId}`;
    const initialIdToken = createIdToken();
    const refreshedIdToken = createIdToken(futureExpiry() + 1);
    const storage = new Map<string, string>([
      [key, createUser({ idToken: initialIdToken }).toStorageString()],
    ]);
    const oidcStorage = {
      getItem: (storageKey: string) => storage.get(storageKey) ?? null,
      removeItem: (storageKey: string) => storage.delete(storageKey),
    };

    expect(
      getStoredOidcBearerToken({
        authority,
        clientId,
        configuredSource: 'id_token',
        storage: oidcStorage,
      })
    ).toBe(initialIdToken);

    storage.set(key, createUser({ idToken: refreshedIdToken }).toStorageString());

    expect(
      getStoredOidcBearerToken({
        authority,
        clientId,
        configuredSource: 'id_token',
        storage: oidcStorage,
      })
    ).toBe(refreshedIdToken);
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
