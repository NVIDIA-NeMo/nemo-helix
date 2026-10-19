// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRehydrateListFromDetailQuery } from '@nemo/common/src/hooks/useRehydrateListFromDetailQuery';
import { getModelsListDeploymentsQueryKey } from '@nemo/sdk/generated/platform/model-deployments';
import type { ModelDeployment } from '@nemo/sdk/generated/platform/schema';
import { useMemo } from 'react';

export interface UseRehydrateDeploymentsListFromDetailOptions {
  workspace: string;
  deployment: ModelDeployment | undefined;
  /** @default true */
  enabled?: boolean;
}

/**
 * Pushes `useModelsGetLatestDeployment` (or equivalent) data into every
 * `useModelsListDeployments` cache entry for the workspace so the deployments table stays in sync
 * while a details panel polls status.
 */
export function useRehydrateDeploymentsListFromDetail({
  workspace,
  deployment,
  enabled = true,
}: UseRehydrateDeploymentsListFromDetailOptions): void {
  const listQueryKey = useMemo(() => getModelsListDeploymentsQueryKey(workspace), [workspace]);

  useRehydrateListFromDetailQuery<ModelDeployment, ModelDeployment>({
    enabled: enabled && Boolean(workspace && deployment),
    detail: deployment,
    listQueryKey,
    detailToListItem: (d) => d,
    getRowId: (d) => d.name,
  });
}
