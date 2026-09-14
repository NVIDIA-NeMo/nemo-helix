// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';

/**
 * Builds an msw path pattern from a generated `get*QueryKey` function, instead of
 * hand-copying the URL template. Pass msw path params (e.g. `:workspace`, `:agent`)
 * in place of real values — the query-key function interpolates them as-is, so the
 * result is the same template the SDK builds real requests from and stays in sync
 * with the API.
 */
export const mockApiUrl = <Args extends unknown[]>(
  queryKeyFn: (...args: Args) => readonly [string, ...unknown[]],
  ...params: Args
): string => `${PLATFORM_BASE_URL}${queryKeyFn(...params)[0]}`;
