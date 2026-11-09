// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, Flex, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { InsightsDataView } from '@studio/components/dataViews/InsightsDataView';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { GetInsightsModal } from '@studio/components/GetInsightsModal';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getOptimizerRoute } from '@studio/routes/utils';
import { type FC, useState } from 'react';

export const OptimizerRoute: FC = () => {
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({
    items: [{ href: getOptimizerRoute(workspace), slotLabel: 'Insights' }],
  });

  const [getInsightsOpen, setGetInsightsOpen] = useState(false);

  return (
    <AccessibleTitle title="Insights">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <Flex className="items-center gap-density-md">
              Insights
              <FeatureFlagBadge flag="optimizerEnabled" />
            </Flex>
          }
          slotDescription="Leverage the optimizer agent to review your code and traces and suggest insights."
          slotActions={
            <Button
              color="brand"
              data-testid="get-insights-button"
              onClick={() => setGetInsightsOpen(true)}
            >
              Get Insights
            </Button>
          }
        />
        {getInsightsOpen ? (
          <GetInsightsModal workspace={workspace} onClose={() => setGetInsightsOpen(false)} />
        ) : null}
        <InsightsDataView workspace={workspace} />
      </Stack>
    </AccessibleTitle>
  );
};
