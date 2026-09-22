// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { ModelEvaluationModal } from '@studio/components/evaluation/ModelEvaluationModal';
import { Loading } from '@studio/components/Layouts/Loading';
import { MODEL_EVALUATION_FORM_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getEvaluationResultsRoute } from '@studio/routes/utils';
import { FC, ReactNode, Suspense, useEffect, useState } from 'react';
import { Outlet, useNavigate } from 'react-router';

interface EvaluationResultsLayoutProps {
  /** Open the wizard on mount. Set by the `/evaluation/new` route so the URL
   *  stays linkable while the list still renders behind the dialog. */
  newEvaluationOpen?: boolean;
  /** Defaults to the routed child. The `/evaluation/new` route passes the list
   *  directly, because that path has no child route of its own. */
  children?: ReactNode;
}

export const EvaluationResultsLayout: FC<EvaluationResultsLayoutProps> = ({
  newEvaluationOpen = false,
  children,
}) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const { setBreadcrumbs } = useBreadcrumbs();
  const [open, setOpen] = useState(newEvaluationOpen);

  useEffect(() => {
    setBreadcrumbs([
      {
        slotLabel: 'Model Evaluations',
      },
    ]);
  }, [setBreadcrumbs]);

  /** Closing from `/evaluation/new` returns to the list, so the dialog and the
   *  URL cannot disagree about whether the wizard is open. */
  const close = () => {
    setOpen(false);
    if (newEvaluationOpen) navigate(getEvaluationResultsRoute(workspace));
  };

  return (
    <AccessibleTitle title="Model Evaluations">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Model Evaluations"
          slotActions={
            MODEL_EVALUATION_FORM_ENABLED ? (
              <Button color="brand" onClick={() => setOpen(true)}>
                Run Evaluation
              </Button>
            ) : null
          }
        />
        <Suspense fallback={<Loading description="Loading..." />}>
          {children ?? <Outlet />}
        </Suspense>
      </Stack>
      {MODEL_EVALUATION_FORM_ENABLED ? (
        <ModelEvaluationModal open={open} onClose={close} workspace={workspace} />
      ) : null}
    </AccessibleTitle>
  );
};
