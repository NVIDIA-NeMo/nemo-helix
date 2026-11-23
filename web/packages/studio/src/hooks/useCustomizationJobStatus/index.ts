// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import {
  customizationGetAutomodelJobStatus,
  getCustomizationGetAutomodelJobStatusQueryKey,
} from '@nemo/sdk/generated/customizer/automodel-jobs';
import {
  customizationGetRlJobStatus,
  getCustomizationGetRlJobStatusQueryKey,
} from '@nemo/sdk/generated/customizer/rl-jobs';
import type {
  HelixJobStatusResponse,
  HelixJobStepStatusResponse,
} from '@nemo/sdk/generated/customizer/schema';
import {
  customizationGetUnslothJobStatus,
  getCustomizationGetUnslothJobStatusQueryKey,
} from '@nemo/sdk/generated/customizer/unsloth-jobs';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import type { CustomizationBackend } from '@studio/util/customizationBackend';
import { type QueryKey, skipToken, useQuery } from '@tanstack/react-query';

interface StatusEndpoint {
  fetchStatus: (
    workspace: string,
    name: string,
    signal?: AbortSignal
  ) => Promise<HelixJobStatusResponse>;
  getQueryKey: (workspace: string, name: string) => QueryKey;
}

const STATUS_ENDPOINTS: Record<CustomizationBackend, StatusEndpoint> = {
  automodel: {
    fetchStatus: (workspace, name, signal) =>
      customizationGetAutomodelJobStatus(workspace, name, undefined, signal),
    getQueryKey: getCustomizationGetAutomodelJobStatusQueryKey,
  },
  unsloth: {
    fetchStatus: (workspace, name, signal) =>
      customizationGetUnslothJobStatus(workspace, name, undefined, signal),
    getQueryKey: getCustomizationGetUnslothJobStatusQueryKey,
  },
  rl: {
    fetchStatus: (workspace, name, signal) =>
      customizationGetRlJobStatus(workspace, name, undefined, signal),
    getQueryKey: getCustomizationGetRlJobStatusQueryKey,
  },
};

/** Key used while the backend is still unknown, so the query has a stable identity. */
const PENDING_BACKEND_QUERY_KEY = ['customization-job-status', 'pending-backend'] as const;

export const getCustomizationJobStatusQueryKey = (
  backend: CustomizationBackend,
  workspace: string,
  name: string
): QueryKey => STATUS_ENDPOINTS[backend].getQueryKey(workspace, name);

export interface UseCustomizationJobStatusResult {
  steps: HelixJobStepStatusResponse[];
  isLoading: boolean;
  isError: boolean;
}

interface UseCustomizationJobStatusOptions {
  /** Defaults to true. Set false to skip polling when a second consumer doesn't need it yet. */
  enabled?: boolean;
}

export const useCustomizationJobStatus = (
  workspace: string,
  name: string,
  backend: CustomizationBackend | undefined,
  jobStatus?: HelixJobStatus,
  { enabled = true }: UseCustomizationJobStatusOptions = {}
): UseCustomizationJobStatusResult => {
  const endpoint = backend ? STATUS_ENDPOINTS[backend] : undefined;
  const canFetch = Boolean(endpoint && workspace && name && enabled);

  const { data, isLoading, isError } = useQuery({
    queryKey: endpoint ? endpoint.getQueryKey(workspace, name) : PENDING_BACKEND_QUERY_KEY,
    queryFn:
      endpoint && canFetch
        ? ({ signal }) => endpoint.fetchStatus(workspace, name, signal)
        : skipToken,
    refetchInterval: getJobRefetchInterval(jobStatus),
  });

  return {
    steps: data?.steps ?? [],
    isLoading,
    isError,
  };
};
