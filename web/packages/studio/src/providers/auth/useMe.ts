// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAuthGetMe } from '@nemo/sdk/generated/platform/profiles';
import { useAuthTokenStatus } from '@studio/providers/auth/useAuthTokenStatus';
import { useWebSession } from '@studio/providers/auth/useWebSession';

const ME_STALE_TIME_MS = 5 * 60_000;

/** The caller's account and profile from `GET /me`. Enabled once signed in, or immediately when auth is off. */
export const useMe = () => {
  const { isTokenActive } = useAuthTokenStatus();
  const { isLoading, authEnabled } = useWebSession();
  return useAuthGetMe({
    query: { enabled: isTokenActive || (!isLoading && !authEnabled), staleTime: ME_STALE_TIME_MS },
  });
};
