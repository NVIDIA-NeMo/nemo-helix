// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { authResolveProfiles } from '@nemo/sdk/generated/platform/profiles';
import type { ResolvedProfile } from '@nemo/sdk/generated/platform/schema';
import { useQuery } from '@tanstack/react-query';

const MAX_PRINCIPALS_PER_REQUEST = 100;
const DISPLAY_STALE_TIME_MS = 10 * 60_000;
export const PRINCIPAL_DISPLAY_QUERY_KEY = ['auth', 'principal-display'] as const;

type Waiter = { resolve: (profile: ResolvedProfile) => void; reject: (error: unknown) => void };

let queued = new Map<string, Waiter[]>();
let scheduled = false;

const flush = () => {
  scheduled = false;
  const batch = queued;
  queued = new Map();
  const principals = [...batch.keys()];
  for (let start = 0; start < principals.length; start += MAX_PRINCIPALS_PER_REQUEST) {
    const chunk = principals.slice(start, start + MAX_PRINCIPALS_PER_REQUEST);
    authResolveProfiles({ principals: chunk })
      .then(({ data }) => {
        for (const profile of data)
          batch.get(profile.principal)?.forEach((w) => w.resolve(profile));
      })
      .catch((error: unknown) =>
        chunk.forEach((p) => batch.get(p)?.forEach((w) => w.reject(error)))
      );
  }
};

/** Resolves one principal, sharing a request with every other lookup made in the same tick. */
const resolvePrincipal = (principal: string) =>
  new Promise<ResolvedProfile>((resolve, reject) => {
    queued.set(principal, [...(queued.get(principal) ?? []), { resolve, reject }]);
    if (!scheduled) {
      scheduled = true;
      queueMicrotask(flush);
    }
  });

/** Public display fields (name, avatar) for a principal string such as a `created_by` value. */
export const usePrincipalDisplay = (principal: string | null | undefined) =>
  useQuery({
    queryKey: [...PRINCIPAL_DISPLAY_QUERY_KEY, principal],
    queryFn: () => resolvePrincipal(principal ?? ''),
    enabled: Boolean(principal),
    staleTime: DISPLAY_STALE_TIME_MS,
  });
