// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Button, Panel, Spinner, Stack, TabsContent } from '@nvidia/foundations-react-core';
import { DetailPageLayout, type DetailPageTab } from '@studio/components/Layouts/DetailPageLayout';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { useMockDistillationWorkflow } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationWorkflow';
import { DistillationResultsTab } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationResultsTab';
import { DistillationStageCard } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationStageCard';
import { getAgentDetailRoute, getAgentsListRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { type FC, useEffect, useState } from 'react';

type DistillationTab = 'progress' | 'results';

const TABS: DetailPageTab<DistillationTab>[] = [
  { value: 'progress', label: 'Progress' },
  { value: 'results', label: 'Results' },
];

/** A distillation's progress and its three-way comparison. Mocked for now. */
export const AgentDistillationDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const toast = useToast();
  const { distillationName } = useRequiredPathParams([ROUTE_PARAMS.distillationName]);
  const workflow = useMockDistillationWorkflow(distillationName);
  const [tab, setTab] = useState<DistillationTab>('progress');
  const isDone = workflow.status === 'completed';

  const { setBreadcrumbs } = useBreadcrumbs();
  useEffect(() => {
    setBreadcrumbs([
      { slotLabel: 'Agents', href: getAgentsListRoute(workspace) },
      { slotLabel: workflow.agentName, href: getAgentDetailRoute(workspace, workflow.agentName) },
      { slotLabel: 'Distillations' },
      { slotLabel: distillationName },
    ]);
    return () => setBreadcrumbs([]);
  }, [setBreadcrumbs, workspace, workflow.agentName, distillationName]);

  return (
    <AccessibleTitle title={`Distillation - ${distillationName}`}>
      <DetailPageLayout
        title={distillationName}
        slotBadges={
          <>
            <StatusBadge status={workflow.status} />
            {!isDone && <Spinner size="small" aria-label="Distillation running" />}
          </>
        }
        tabs={TABS}
        selectedTab={tab}
        onTabChange={setTab}
        slotActions={
          <Button
            color="brand"
            disabled={!isDone}
            onClick={() =>
              toast.success(`Promoted ${workflow.studentModel} to "${workflow.agentName}" (mock)`)
            }
          >
            Promote
          </Button>
        }
      >
        <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="progress">
          <Stack gap="density-2xl" className="w-full">
            <Panel slotHeading="Details" elevation="high" density="compact" className="w-full">
              <KeyValueGrid
                items={[
                  { key: 'agent', label: 'Agent', value: workflow.agentName },
                  { key: 'student', label: 'Student model', value: workflow.studentModel },
                  { key: 'runs', label: 'Evaluation runs', value: String(workflow.runCount) },
                  {
                    key: 'examples',
                    label: 'Training examples',
                    value: String(workflow.exampleCount),
                  },
                ]}
              />
            </Panel>
            <Stack gap="density-lg" className="w-full" role="list" aria-label="Distillation stages">
              {workflow.stages.map((stage) => (
                <DistillationStageCard key={stage.id} stage={stage} />
              ))}
            </Stack>
          </Stack>
        </TabsContent>
        <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="results">
          <DistillationResultsTab workflow={workflow} />
        </TabsContent>
      </DetailPageLayout>
    </AccessibleTitle>
  );
};
