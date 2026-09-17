// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { CJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import {
  Flex,
  PageHeader,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import { CustomizationFilesetDetailsPanel } from '@studio/components/CustomizationFilesetDetailsPanel';
import { CustomizationOverview } from '@studio/components/CustomizationOverview';
import { Loading } from '@studio/components/Layouts/Loading';
import { ModelChat } from '@studio/components/ModelChat';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useCustomizationJob } from '@studio/hooks/useCustomizationJob';
import { useCustomizationJobStatus } from '@studio/hooks/useCustomizationJobStatus';
import { useModelChatAvailability } from '@studio/hooks/useModelChatAvailability';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { CustomizationFailureBanner } from '@studio/routes/CustomizationJobDetailsRoute/CustomizationFailureBanner';
import { DetailActions } from '@studio/routes/CustomizationJobDetailsRoute/DetailActions';
import { LogsTab } from '@studio/routes/CustomizationJobDetailsRoute/LogsTab';
import { getCustomizationJobListRoute } from '@studio/routes/utils';
import { resolveCustomizationFailure } from '@studio/util/customizationFailure';
import {
  getBaseModel,
  getDatasetUri,
  getFinetuningType,
  getFormattedTrainingType,
  getGrpoRunProgressSummary,
  getTrainingTelemetry,
} from '@studio/util/customizations';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { Dot } from 'lucide-react';
import { type FC, Fragment, useState } from 'react';

export const CustomizationJobDetailsRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { customizationJobName } = useRequiredPathParams([ROUTE_PARAMS.customizationJobName]);

  const { job, backend } = useCustomizationJob(workspace, customizationJobName, {
    refetchInterval: (query) => getJobRefetchInterval(query.state.data?.status),
  });

  useBreadcrumbs({
    items: [
      {
        href: getCustomizationJobListRoute(workspace),
        slotLabel: 'Customizations',
      },
      {
        slotLabel: job?.id || '',
      },
    ],
  });

  const status = job?.status;
  const output_model = job?.spec.output?.name;
  const showChat = Boolean(output_model) && status === 'completed';
  const isTerminalStatus = Boolean(status && CJobTerminalStatuses.includes(status));

  // Steps aren't on the job record; only fetch them once terminal, since a banner is impossible before then.
  const { steps } = useCustomizationJobStatus(workspace, customizationJobName, backend, status, {
    enabled: isTerminalStatus,
  });
  const failure = resolveCustomizationFailure(job, steps);

  const [activeTab, setActiveTab] = useState('overview');

  // Chat unmounts once the job is no longer completed; fall back to a tab that always renders.
  const selectedTab = activeTab === 'chat' && !showChat ? 'overview' : activeTab;

  // Fetch the output model entity so we can check deployment status
  const { data: outputModelEntity } = useModelsGetModel(workspace, output_model ?? '', undefined, {
    query: { enabled: showChat, retry: false },
  });

  const { modelChatStatus, isLoading: isChatStatusLoading } =
    useModelChatAvailability(outputModelEntity);

  const metadata = [
    getFormattedTrainingType(getFinetuningType(job)),
    getBaseModel(job),
    getGrpoRunProgressSummary(job, getTrainingTelemetry(job), isTerminalStatus),
    job?.created_at ? `created ${formatAbsoluteTimestamp(job.created_at)}` : '',
  ].filter(Boolean);

  return (
    <AccessibleTitle title={`Customization details for ${customizationJobName}`}>
      <Stack className="h-full min-h-0 w-full p-density-2xl" gap="density-xl">
        <PageHeader
          className="shrink-0 p-0"
          slotHeading={
            <Stack gap="1">
              <Flex align="baseline" gap="3">
                <Text kind="title/md">{job?.name ?? customizationJobName}</Text>
                {status && <StatusBadge status={status} />}
              </Flex>
              {metadata.length > 0 && (
                <Flex align="center" gap="1">
                  {metadata.map((segment, index) => (
                    <Fragment key={segment}>
                      {index > 0 && <Dot className="size-2" aria-hidden />}
                      <Text kind="body/regular/sm" className="text-secondary">
                        {segment}
                      </Text>
                    </Fragment>
                  ))}
                </Flex>
              )}
            </Stack>
          }
          slotActions={
            <DetailActions
              model={output_model}
              status={status}
              backend={backend}
              name={customizationJobName}
              job={job}
            />
          }
        />
        {failure && (
          <CustomizationFailureBanner failure={failure} onViewLogs={() => setActiveTab('logs')} />
        )}
        <TabsRoot
          className="flex min-h-0 flex-1 flex-col"
          value={selectedTab}
          onValueChange={setActiveTab}
        >
          <TabsList className="shrink-0">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
            {output_model && status === 'completed' && (
              <TabsTrigger value="chat">Chat with your Model</TabsTrigger>
            )}
          </TabsList>

          <TabsContent
            value="overview"
            className="min-h-0 flex-1 overflow-y-auto p-0 pr-density-md"
          >
            <Stack className="w-full" gap="density-xl">
              <CustomizationOverview
                customizationJobName={customizationJobName}
                workspace={workspace}
              />
              <CustomizationFilesetDetailsPanel filesetUri={getDatasetUri(job) || undefined} />
            </Stack>
          </TabsContent>

          <TabsContent value="logs" className="flex min-h-0 flex-1 p-0">
            <LogsTab
              customizationJobName={customizationJobName}
              workspace={workspace}
              jobStatus={status}
            />
          </TabsContent>

          {showChat && output_model && (
            <TabsContent value="chat" className="flex min-h-0 flex-1 items-center p-0">
              {isChatStatusLoading ? (
                <Loading />
              ) : (
                <Stack className="h-full w-full max-w-[768px]">
                  <ModelChat
                    model={output_model}
                    workspace={workspace}
                    modelChatStatus={modelChatStatus}
                    deployModelRef={`${workspace}/${output_model}`}
                  />
                </Stack>
              )}
            </TabsContent>
          )}
        </TabsRoot>
      </Stack>
    </AccessibleTitle>
  );
};
