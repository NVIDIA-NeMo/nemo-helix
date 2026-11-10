// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTE_PARAMS, ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import type { ModelDetailTab } from '@studio/routes/ModelDetailRoute/constants';
import { getBaseModelDetailRoute, getModelDetailRoute } from '@studio/routes/utils';
import { useMatch, useParams } from 'react-router';

/** Links to a tab of the model page being viewed, keeping its catalog or fine-tuning URL. */
export const useModelTabRoute = () => {
  const workspace = useWorkspaceFromPath();
  const { [ROUTE_PARAMS.modelName]: modelName = '', [ROUTE_PARAMS.adapterName]: adapterName } =
    useParams();
  const isCatalog = Boolean(useMatch(ROUTES.workspace.baseModelsModel));

  return (tab: ModelDetailTab) =>
    isCatalog
      ? getBaseModelDetailRoute(workspace, modelName, tab)
      : getModelDetailRoute(workspace, modelName, { adapterName, tab });
};
