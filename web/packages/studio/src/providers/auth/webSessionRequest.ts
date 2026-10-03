// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { OIDC_TOKEN_ENDPOINT_AUTH_METHOD } from '@studio/constants/environment';

const isConfidentialOidcClient = OIDC_TOKEN_ENDPOINT_AUTH_METHOD === 'client_secret_basic';
const MUTATING_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

export const withWebSessionRequest = (init?: RequestInit): RequestInit | undefined => {
  if (!isConfidentialOidcClient) return init;

  const method = (init?.method ?? 'GET').toUpperCase();
  const headers = new Headers(init?.headers);
  if (MUTATING_METHODS.has(method)) headers.set('X-NHX-Requested-By', '1');

  return { ...init, credentials: 'include', headers };
};
