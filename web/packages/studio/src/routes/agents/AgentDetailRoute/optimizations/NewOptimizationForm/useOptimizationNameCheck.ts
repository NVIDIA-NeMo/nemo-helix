// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { useAgentsGetOptimizeJob } from '@nemo/sdk/generated/agents/agents';

export type NameCheckStatus = 'idle' | 'checking' | 'available' | 'conflict' | 'failed';

export interface NameCheckResult {
  candidate: string;
  status: NameCheckStatus;
}

/**
 * Whether a study name is already taken.
 */
export const useOptimizationNameCheck = (
  workspace: string,
  candidate: string,
  current: string
): NameCheckResult => {
  const { isPending, isError, error } = useAgentsGetOptimizeJob(workspace, candidate, {
    query: {
      enabled: !!workspace && !!candidate,
      retry: false,
      staleTime: 0,
    },
  });

  if (!candidate || candidate !== current) return { candidate: current, status: 'idle' };
  if (isError) return { candidate, status: isNotFoundError(error) ? 'available' : 'failed' };
  if (isPending) return { candidate, status: 'checking' };
  return { candidate, status: 'conflict' };
};
