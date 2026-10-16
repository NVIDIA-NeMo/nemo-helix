// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { DocumentationButton } from '@studio/components/DocumentationButton';
import { SuspenseOutlet } from '@studio/components/Layouts/SuspenseOutlet';
import { LINK_EVAL_DOCS_BENCHMARKS_INDUSTRY } from '@studio/constants/links';
import { ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getNewEvaluationMetricRoute } from '@studio/routes/utils';
import { FC, useEffect, useMemo } from 'react';
import { matchPath, useLocation, useNavigate } from 'react-router';

export const EvaluationLayout: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { pathname } = useLocation();
  const navigate = useNavigate();

  const { setBreadcrumbs } = useBreadcrumbs();

  const hubTitle = useMemo(() => {
    const isBenchmarks =
      matchPath(ROUTES.workspace.evaluationBenchmarks, pathname) !== null ||
      matchPath(ROUTES.workspace.evaluationBenchmarkDetails, pathname) !== null;
    return isBenchmarks ? 'Benchmarks' : 'Metrics';
  }, [pathname]);

  useEffect(() => {
    setBreadcrumbs([
      {
        slotLabel: hubTitle,
      },
    ]);
  }, [hubTitle, setBreadcrumbs]);

  return (
    <AccessibleTitle title={`${hubTitle} for ${workspace}`}>
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={hubTitle}
          slotActions={
            hubTitle === 'Metrics' ? (
              <Stack className="grow" align="end">
                <Button
                  color="brand"
                  onClick={() => navigate(getNewEvaluationMetricRoute(workspace))}
                >
                  New Metric
                </Button>
              </Stack>
            ) : hubTitle === 'Benchmarks' ? (
              <Stack className="grow" align="end">
                <DocumentationButton
                  href={LINK_EVAL_DOCS_BENCHMARKS_INDUSTRY}
                  text="Browse Benchmarks"
                  attributes={{
                    Button: {
                      kind: 'secondary',
                    },
                  }}
                />
              </Stack>
            ) : undefined
          }
        />
        <SuspenseOutlet />
      </Stack>
    </AccessibleTitle>
  );
};
