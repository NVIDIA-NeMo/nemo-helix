// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { ModelEvaluationWizard } from '@studio/components/evaluation/ModelEvaluationWizard';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { getEvaluationResultsRoute } from '@studio/routes/utils';
import { FC } from 'react';

/** `/workspaces/:workspace/evaluation/new` -- the Run Model Evaluation wizard. */
export const EvaluationNewRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { getQueryParam } = useQueryParams();

  useBreadcrumbs({
    items: [
      { href: getEvaluationResultsRoute(workspace), slotLabel: 'Model Evaluations' },
      { slotLabel: 'New Evaluation' },
    ],
  });

  // `getQueryParam` returns '' when the param is absent, which is already the
  // form's default for an unchosen model.
  return (
    <ModelEvaluationWizard
      workspace={workspace}
      initialModel={getQueryParam(QUERY_PARAMETERS.model)}
    />
  );
};
