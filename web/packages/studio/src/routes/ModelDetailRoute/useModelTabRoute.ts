// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTE_PARAMS, ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import type { ModelDetailTab } from '@studio/routes/ModelDetailRoute/constants';
import { getBaseModelDetailRoute, getModelDetailRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { useMatch, useParams } from 'react-router';

/** Links to a tab of the model page being viewed, keeping its catalog or fine-tuning URL. */
export const useModelTabRoute = () => {
  const workspace = useWorkspaceFromPath();
  const { modelName } = useRequiredPathParams([ROUTE_PARAMS.modelName]);
  const { [ROUTE_PARAMS.adapterName]: adapterName } = useParams();
  const isCatalog = Boolean(useMatch(ROUTES.workspace.baseModelsModel));

  return (tab: ModelDetailTab) => {
    if (isCatalog) {
      return getBaseModelDetailRoute(workspace, modelName, tab);
    }
    return getModelDetailRoute(workspace, modelName, { adapterName, tab });
  };
};
