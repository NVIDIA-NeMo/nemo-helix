// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  Button,
  Flex,
  Panel,
  Spinner,
  Stack,
  TabsContent,
} from '@nvidia/foundations-react-core';
import { DetailPageLayout, type DetailPageTab } from '@studio/components/Layouts/DetailPageLayout';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import {
  useCancelDistillation,
  useDistillationWorkflow,
  usePromoteDistillation,
  useRetryDistillationNode,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationHooks';
import { distilledGatewayModel } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationPromote';
import {
  agentNameOf,
  toArms,
  toStageView,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationView';
import { DistillationFailedNodes } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationFailedNodes';
import { DistillationResultsTab } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationResultsTab';
import { DistillationStageCard } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationStageCard';
import { getAgentDetailRoute, getAgentsListRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { type FC, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router';

type DistillationTab = 'progress' | 'results';

const TABS: DetailPageTab<DistillationTab>[] = [
  { value: 'progress', label: 'Progress' },
  { value: 'results', label: 'Results' },
];

/** A distillation workflow's progress and its comparison of the original, swapped and distilled models. */
export const AgentDistillationDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const toast = useToast();
  const navigate = useNavigate();
  const { distillationName } = useRequiredPathParams([ROUTE_PARAMS.distillationName]);
  const { data: workflow, error } = useDistillationWorkflow(workspace, distillationName);
  const cancel = useCancelDistillation(workspace, distillationName);
  const retry = useRetryDistillationNode(workspace, distillationName);
  const promote = usePromoteDistillation(workspace);
  const [chosenTab, setChosenTab] = useState<DistillationTab | undefined>();
  const wasRunning = useRef(false);
  const status = workflow?.status;
  const isDone = status === 'completed';
  const isActive = status === 'pending' || status === 'running' || status === 'cancelling';
  const agentName = workflow ? agentNameOf(workflow) : undefined;

  useEffect(() => {
    if (status === 'running') {
      wasRunning.current = true;
    } else if (status === 'completed' && wasRunning.current) {
      wasRunning.current = false;
      setChosenTab('results');
      toast.success('Distillation finished.');
    }
  }, [status, toast]);

  const { setBreadcrumbs } = useBreadcrumbs();
  useEffect(() => {
    setBreadcrumbs([
      { slotLabel: 'Agents', href: getAgentsListRoute(workspace) },
      ...(agentName
        ? [{ slotLabel: agentName, href: getAgentDetailRoute(workspace, agentName) }]
        : []),
      { slotLabel: 'Distillations' },
      { slotLabel: distillationName },
    ]);
    return () => setBreadcrumbs([]);
  }, [setBreadcrumbs, workspace, agentName, distillationName]);

  if (!workflow) {
    return error ? (
      <Stack padding="density-2xl">
        <ErrorMessage header="Could not load distillation" message={getErrorMessage(error)} />
      </Stack>
    ) : (
      <Flex align="center" justify="center" className="h-full w-full">
        <Spinner size="medium" aria-label="Loading distillation..." />
      </Flex>
    );
  }

  const selectedTab = chosenTab ?? (isDone ? 'results' : 'progress');
  const inputs = workflow.inputs;

  const onPromote = () =>
    promote.mutate(workflow, {
      onSuccess: (agent) => {
        toast.success(`Created agent "${agent.name}"`);
        navigate(getAgentDetailRoute(workspace, agent.name ?? ''));
      },
      onError: (e) => toast.error(getErrorMessage(e)),
    });

  return (
    <AccessibleTitle title={`Distillation - ${distillationName}`}>
      <DetailPageLayout
        title={distillationName}
        slotBadges={
          <>
            <StatusBadge status={workflow.status} />
            {isActive && <Spinner size="small" aria-label="Distillation running" />}
          </>
        }
        tabs={TABS}
        selectedTab={selectedTab}
        onTabChange={setChosenTab}
        slotActions={
          <>
            {(status === 'pending' || status === 'running') && (
              <Button
                kind="secondary"
                disabled={cancel.isPending}
                onClick={() =>
                  cancel.mutate(undefined, { onError: (e) => toast.error(getErrorMessage(e)) })
                }
              >
                Cancel
              </Button>
            )}
            <Button
              color="brand"
              disabled={!isDone || !distilledGatewayModel(workflow) || promote.isPending}
              loading={promote.isPending}
              onClick={onPromote}
            >
              Promote
            </Button>
          </>
        }
      >
        <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="progress">
          <Stack gap="density-2xl" className="w-full">
            <Panel slotHeading="Details" elevation="high" density="compact" className="w-full">
              <KeyValueGrid
                items={[
                  { key: 'agent', label: 'Agent', value: workflow.source_agent },
                  { key: 'student', label: 'Student model', value: inputs.student_model },
                  { key: 'teacher', label: 'Teacher model', value: inputs.teacher?.model },
                  {
                    key: 'trajectories',
                    label: 'Trajectory filesets',
                    value: inputs.trajectory_filesets?.join(', '),
                  },
                  {
                    key: 'evals',
                    label: 'Evaluations',
                    value: inputs.evals?.map((evaluation) => evaluation.name).join(', '),
                  },
                  {
                    key: 'candidates',
                    label: 'Candidates',
                    value: inputs.candidates?.map((candidate) => candidate.name).join(', '),
                  },
                  { key: 'experiment', label: 'Experiment', value: workflow.experiment ?? undefined },
                ]}
              />
            </Panel>
            <DistillationFailedNodes
              nodes={workflow.nodes}
              onRetry={(node) =>
                retry.mutate(node, { onError: (e) => toast.error(getErrorMessage(e)) })
              }
              isRetrying={retry.isPending}
            />
            <Stack gap="density-lg" className="w-full" role="list" aria-label="Distillation stages">
              {workflow.stages.map((stage) => (
                <DistillationStageCard key={stage.id} stage={toStageView(stage)} />
              ))}
            </Stack>
          </Stack>
        </TabsContent>
        <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="results">
          <DistillationResultsTab status={workflow.status} arms={toArms(workflow)} />
        </TabsContent>
      </DetailPageLayout>
    </AccessibleTitle>
  );
};
