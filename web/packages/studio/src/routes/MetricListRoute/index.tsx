// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { MetricsDataView } from '@studio/components/dataViews/MetricsDataView';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getMetricsRoute, getNewMetricRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

export const MetricListRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const create = () => navigate(getNewMetricRoute(workspace));

  useBreadcrumbs({ items: [{ href: getMetricsRoute(workspace), slotLabel: 'Metrics' }] });

  return (
    <AccessibleTitle title="Metrics">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <>
              Metrics <FeatureFlagBadge flag="metricsEnabled" />
            </>
          }
          slotDescription="Reusable scorers for any evaluation, with how their scores have moved across recent runs."
          slotActions={
            <Button color="brand" onClick={create}>
              Create Metric
            </Button>
          }
        />
        <MetricsDataView workspace={workspace} onCreate={create} />
      </Stack>
    </AccessibleTitle>
  );
};
