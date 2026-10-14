// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelEvaluationWizard } from '@studio/components/evaluation/ModelEvaluationWizard';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getEvaluationResultsRoute } from '@studio/routes/utils';
import { FC } from 'react';

/** `/workspaces/:workspace/evaluation/new` -- the Run Model Evaluation wizard. */
export const EvaluationNewRoute: FC = () => {
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({
    items: [
      { href: getEvaluationResultsRoute(workspace), slotLabel: 'Model Evaluations' },
      { slotLabel: 'New Evaluation' },
    ],
  });

  return <ModelEvaluationWizard workspace={workspace} />;
};
