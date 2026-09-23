// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { getOidcIdTokenExpiresAt } from '@nemo/sdk/src/utils/oidcBearerToken';
import { AUTH_BEARER_TOKEN_SOURCE } from '@studio/constants/environment';
import { useEffect, useRef } from 'react';
import { useAuth } from 'react-oidc-context';

const ID_TOKEN_RENEWAL_LEAD_TIME_MS = 90_000;

export const OidcIdTokenRenewal = (): null => {
  const { signinSilent, user } = useAuth();
  const renewalInFlightForToken = useRef<string | undefined>(undefined);

  useEffect(() => {
    const idToken = user?.id_token;
    if (AUTH_BEARER_TOKEN_SOURCE !== 'id_token' || !idToken) return;

    const expiresAt = getOidcIdTokenExpiresAt(idToken);
    if (expiresAt === undefined) return;

    const renew = () => {
      if (renewalInFlightForToken.current === idToken) return;
      renewalInFlightForToken.current = idToken;

      void signinSilent()
        .catch((error: unknown) => {
          logger.warn('Failed to renew the OIDC ID token before expiry', error);
        })
        .finally(() => {
          if (renewalInFlightForToken.current === idToken) {
            renewalInFlightForToken.current = undefined;
          }
        });
    };

    const delayMs = expiresAt * 1000 - Date.now() - ID_TOKEN_RENEWAL_LEAD_TIME_MS;
    if (delayMs <= 0) {
      renew();
      return;
    }

    const timeoutId = window.setTimeout(renew, delayMs);
    return () => window.clearTimeout(timeoutId);
  }, [signinSilent, user?.id_token]);

  return null;
};
