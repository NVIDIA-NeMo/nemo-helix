// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { Loading } from '@studio/components/Layouts/Loading';
import { MODEL_EVALUATION_FORM_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getEvaluationNewRoute } from '@studio/routes/utils';
import { FC, Suspense, useEffect } from 'react';
import { Outlet, useNavigate } from 'react-router';

export const EvaluationResultsLayout: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const { setBreadcrumbs } = useBreadcrumbs();

  useEffect(() => {
    setBreadcrumbs([
      {
        slotLabel: 'Model Evaluations',
      },
    ]);
  }, [setBreadcrumbs]);

  return (
    <AccessibleTitle title="Model Evaluations">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Model Evaluations"
          slotActions={
            MODEL_EVALUATION_FORM_ENABLED ? (
              <Button color="brand" onClick={() => navigate(getEvaluationNewRoute(workspace))}>
                Run Evaluation
              </Button>
            ) : null
          }
        />
        <Suspense fallback={<Loading description="Loading..." />}>
          <Outlet />
        </Suspense>
      </Stack>
    </AccessibleTitle>
  );
};
