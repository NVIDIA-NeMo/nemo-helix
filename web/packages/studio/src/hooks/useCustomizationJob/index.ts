// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useJobsGetJob } from '@nemo/sdk/generated/platform/jobs';
import type { HelixJobResponse } from '@nemo/sdk/generated/platform/schema';
import {
  getCustomizationBackend,
  type CustomizationBackend,
  type CustomizationJob,
  type CustomizationJobSpec,
} from '@studio/util/customizationBackend';
import type { UseQueryOptions } from '@tanstack/react-query';

export type CustomizationJobQueryOptions = Partial<
  UseQueryOptions<HelixJobResponse, unknown, HelixJobResponse>
>;

export interface UseCustomizationJobResult {
  job?: CustomizationJob;
  backend?: CustomizationBackend;
  isLoading: boolean;
  isError: boolean;
  refetch: () => void;
}

export const useCustomizationJob = (
  workspace: string,
  name: string,
  query?: CustomizationJobQueryOptions
): UseCustomizationJobResult => {
  const { data, isLoading, isError, refetch } = useJobsGetJob<HelixJobResponse, unknown>(
    workspace,
    name,
    { query: query ?? {} }
  );

  const backend = getCustomizationBackend(data?.spec);
  const job =
    data && backend
      ? ({ ...data, spec: data.spec as unknown as CustomizationJobSpec } as CustomizationJob)
      : undefined;

  return {
    job,
    backend,
    isLoading,
    isError,
    refetch: () => {
      void refetch();
    },
  };
};
