// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useDataDesignerGetCreateJob } from '@nemo/sdk/generated/data-designer/data-designer';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';

export const useDataDesignerJobFromRoute = () => {
  const workspace = useWorkspaceFromPath();
  const { dataDesignerJobName } = useRequiredPathParams([ROUTE_PARAMS.dataDesignerJobName]);

  const query = useDataDesignerGetCreateJob(workspace, dataDesignerJobName, {
    query: {
      refetchInterval: (q) => {
        const status = q.state.data?.status;
        const isTerminated = status && HelixJobTerminalStatuses.includes(status);
        return isTerminated ? false : 3000;
      },
    },
  });

  return {
    ...query,
    workspace,
    jobName: dataDesignerJobName,
    job: query.data,
  };
};
