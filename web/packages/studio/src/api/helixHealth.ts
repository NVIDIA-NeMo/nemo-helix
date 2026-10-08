// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

/**
 * Outcome of probing the platform's liveness endpoint.
 *
 * - `live`: the platform is reachable, even if some services or controllers are not ready.
 * - `unreachable`: no usable response — connection refused, DNS failure, timeout, a dev
 *   proxy that could not reach its target, or any response (even a 200) that is not the
 *   platform's.
 */
export type HelixHealthStatus = 'live' | 'unreachable';

export const HELIX_HEALTH_QUERY_KEY = ['platform-health'] as const;

const HEALTH_TIMEOUT_MS = 10_000;
export const HELIX_HEALTH_RETRY_INTERVAL_MS = 5_000;

/** Liveness probe exposed by the platform runner; it is not part of the OpenAPI spec. */
export const getHelixHealthUrl = (): string =>
  `${PLATFORM_BASE_URL.replace(/\/+$/, '')}/health/live`;

const isLiveResponse = (data: unknown): boolean =>
  typeof data === 'object' && data !== null && 'status' in data && data.status === 'live';

export const checkHelixHealth = async (): Promise<HelixHealthStatus> => {
  try {
    const { data } = await axios.get<unknown>(getHelixHealthUrl(), {
      timeout: HEALTH_TIMEOUT_MS,
      validateStatus: (status) => status === 200,
    });
    // A 200 without Helix's body (e.g. a dev server's SPA fallback serving index.html) is
    // not the platform answering, so it must not unblock the app.
    return isLiveResponse(data) ? 'live' : 'unreachable';
  } catch {
    return 'unreachable';
  }
};

/**
 * Retries failed connectivity probes automatically so a transient failure cannot leave
 * Studio blocked after the platform recovers. Stops polling once liveness is confirmed;
 * individual features handle their own service errors. `refetch` also allows manual retry.
 */
export const useHelixHealth = () =>
  useQuery({
    queryKey: HELIX_HEALTH_QUERY_KEY,
    queryFn: checkHelixHealth,
    retry: false,
    staleTime: Infinity,
    gcTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false,
    refetchInterval: (query) =>
      query.state.data === 'unreachable' ? HELIX_HEALTH_RETRY_INTERVAL_MS : false,
  });
