// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useQuery } from '@tanstack/react-query';
import axios, { AxiosError } from 'axios';

/**
 * Outcome of probing the platform's readiness endpoint.
 *
 * - `ready`: `/health/ready` returned 200 — every service and controller is up.
 * - `not-ready`: the platform process answered 503 with its `not_ready` body, so it is
 *   reachable but still starting (or a service is down).
 * - `unreachable`: no usable response — connection refused, DNS failure, timeout, a dev
 *   proxy that could not reach its target, or any response that is not the platform's.
 */
export type HelixHealthStatus = 'ready' | 'not-ready' | 'unreachable';

export const HELIX_HEALTH_QUERY_KEY = ['platform-health'] as const;

const HEALTH_TIMEOUT_MS = 10_000;

/** Readiness probe exposed by the platform runner; it is not part of the OpenAPI spec. */
export const getHelixHealthUrl = (): string =>
  `${PLATFORM_BASE_URL.replace(/\/+$/, '')}/health/ready`;

const isNotReadyResponse = (error: unknown): boolean => {
  if (!(error instanceof AxiosError) || error.response?.status !== 503) return false;
  const detail: unknown = error.response.data?.detail;
  return (
    typeof detail === 'object' &&
    detail !== null &&
    'status' in detail &&
    detail.status === 'not_ready'
  );
};

export const checkHelixHealth = async (): Promise<HelixHealthStatus> => {
  try {
    await axios.get(getHelixHealthUrl(), {
      timeout: HEALTH_TIMEOUT_MS,
      validateStatus: (status) => status === 200,
    });
    return 'ready';
  } catch (error) {
    return isNotReadyResponse(error) ? 'not-ready' : 'unreachable';
  }
};

/**
 * Probes the platform once per app load. The query function never throws — failures are
 * folded into {@link HelixHealthStatus} — so callers branch on `data`, and `refetch`
 * is the manual retry. No background refetching: once the platform is confirmed up we
 * get out of the way.
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
  });
