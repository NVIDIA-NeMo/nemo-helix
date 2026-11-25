// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useHost } from '@agent-hardener/host';
import type { PluginSdk } from '@agent-hardener/types';
import { getJobRefetchInterval } from '@nemo/common';

/**
 * Studio's platform SDK, handed in on the host rather than imported: the hooks
 * must run on Studio's authenticated axios and its single QueryClient, so a
 * plugin never bundles or configures the SDK itself.
 */
export const useHelixSdk = (): PluginSdk['platform'] => useHost().sdk.platform;

/** A platform job, polled until it reaches a terminal status; idle while `jobName` is empty. */
export const usePolledJob = (workspace: string, jobName: string) =>
  useHelixSdk().useJobsGetJob(workspace, jobName, {
    query: {
      enabled: Boolean(jobName),
      refetchInterval: (query) => getJobRefetchInterval(query.state.data?.status),
    },
  });
