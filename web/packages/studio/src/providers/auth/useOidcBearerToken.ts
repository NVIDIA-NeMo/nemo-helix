// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { selectOidcBearerToken } from '@nemo/sdk/src/utils/oidcBearerToken';
import { AUTH_BEARER_TOKEN_SOURCE } from '@studio/constants/environment';
import { useAuth } from 'react-oidc-context';

export const useOidcBearerToken = (): string | undefined => {
  const { user } = useAuth();
  return selectOidcBearerToken(user, AUTH_BEARER_TOKEN_SOURCE);
};
