// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import { getListExperimentsQueryKey } from '@nemo/sdk/generated/platform/experiments';
import type { QueryClient } from '@tanstack/react-query';

/** Refresh memberships as well as experiments, including agent rollups and cached detail pages. */
export const invalidateExperimentDeletion = async (client: QueryClient, workspace: string) => {
  const roots = [
    getListExperimentsQueryKey(workspace)[0],
    getListEvaluationsQueryKey(workspace)[0],
  ];
  await client.invalidateQueries({
    predicate: ({ queryKey }) => {
      const key = queryKey[0];
      return (
        (typeof key === 'string' &&
          roots.some((root) => key === root || key.startsWith(`${root}/`))) ||
        (key === 'experiments-by-id' && queryKey[1] === workspace)
      );
    },
  });
};
