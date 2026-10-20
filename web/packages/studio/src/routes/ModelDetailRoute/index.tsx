// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { formatFinetuningType } from '@nemo/common/src/utils/formatters';
import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import { Badge, Button, Stack, TabsContent, TabsRoot } from '@nvidia/foundations-react-core';
import { CustomizationOverview } from '@studio/components/CustomizationOverview';
import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import { DetailPageHeader, type DetailPageHeaderTab } from '@studio/components/DetailPageHeader';
import { Loading } from '@studio/components/Layouts/Loading';
import { ModelChatTab } from '@studio/components/ModelChatTab';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useCustomizationJob } from '@studio/hooks/useCustomizationJob';
import { useModelLineage } from '@studio/hooks/useModelLineage';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { LogsTab } from '@studio/routes/CustomizationJobDetailsRoute/LogsTab';
import { ModelDetailTab } from '@studio/routes/ModelDetailRoute/constants';
import { DetailsTab } from '@studio/routes/ModelDetailRoute/DetailsTab';
import {
  getCustomizationJobListRoute,
  getModelDetailRoute,
  getWorkspaceCustomizationJobDetailsRoute,
} from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';
import { Link, useParams } from 'react-router';

export const ModelDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { modelName } = useRequiredPathParams([ROUTE_PARAMS.modelName]);
  const { [ROUTE_PARAMS.adapterName]: adapterName } = useParams();
  const { getQueryParam, setQueryParam } = useQueryParams();

  const lineage = useModelLineage(workspace, modelName, adapterName);
  const { model, adapter, job } = lineage;
  const name = adapterName ?? modelName;

  const { job: jobDetails } = useCustomizationJob(job.workspace, job.jobName ?? '', {
    enabled: Boolean(job.jobName),
    refetchInterval: (query) => getJobRefetchInterval(query.state.data?.status),
  });

  useBreadcrumbs({
    items: [
      { href: getCustomizationJobListRoute(workspace), slotLabel: 'Fine-tuning' },
      ...(adapterName
        ? [{ href: getModelDetailRoute(workspace, modelName), slotLabel: modelName }]
        : []),
      { slotLabel: name },
    ],
  });

  // The job tabs stay visible but disabled while the lookup runs, and go away if there is no job.
  const hasJobTabs = job.isLoading || Boolean(job.jobName);
  const tabs: DetailPageHeaderTab[] = [
    { value: ModelDetailTab.Details, label: 'Details' },
    { value: ModelDetailTab.Chat, label: 'Chat' },
    ...(hasJobTabs
      ? [
          { value: ModelDetailTab.Training, label: 'Training', disabled: job.isLoading },
          { value: ModelDetailTab.Logs, label: 'Logs', disabled: job.isLoading },
        ]
      : []),
  ];

  const tabFromUrl = getQueryParam(QUERY_PARAMETERS.tab);
  const selectedTab = tabs.find((tab) => tab.value === tabFromUrl)?.value ?? ModelDetailTab.Details;

  const isJobTab = selectedTab === ModelDetailTab.Training || selectedTab === ModelDetailTab.Logs;

  if (lineage.isLoading) {
    return <Loading description="Loading model..." />;
  }

  if (lineage.isError || lineage.isAdapterMissing || !model) {
    return (
      <AccessibleTitle title={name}>
        <Stack padding="density-2xl">
          {lineage.isAdapterMissing ? (
            <ErrorMessage
              header="Adapter not found"
              message={`The model "${modelName}" has no adapter named "${adapterName}".`}
            />
          ) : (
            <ErrorMessage
              header="Model not found"
              message={`Couldn't load the model "${modelName}" in workspace "${workspace}".`}
            />
          )}
        </Stack>
      </AccessibleTitle>
    );
  }

  const finetuningType = adapter?.finetuning_type ?? model.finetuning_type;
  const tags = [
    finetuningType ? formatFinetuningType(finetuningType) : undefined,
    adapter ? `Model: ${model.name}` : model.base_model && `Base: ${model.base_model}`,
  ].filter(Boolean);

  return (
    <AccessibleTitle title={`Model details for ${name}`}>
      <Stack className="h-full min-h-0 w-full p-density-2xl">
        <TabsRoot
          className="flex min-h-0 flex-1 flex-col gap-density-xl"
          value={selectedTab}
          onValueChange={(value) => setQueryParam(QUERY_PARAMETERS.tab, value)}
        >
          <DetailPageHeader
            heading={name}
            slotStatus={
              <DeploymentStatusBadge state={lineage.deployment} isAdapter={Boolean(adapter)} />
            }
            slotTags={
              tags.length > 0 &&
              tags.map((tag) => (
                <Badge key={tag} kind="outline" color="gray">
                  {tag}
                </Badge>
              ))
            }
            description={(adapter ?? model).description}
            tabs={tabs}
            slotActions={
              isJobTab &&
              job.jobName && (
                <Button color="brand" asChild>
                  <Link to={getWorkspaceCustomizationJobDetailsRoute(job.workspace, job.jobName)}>
                    Open Job
                  </Link>
                </Button>
              )
            }
          />

          <TabsContent
            value={ModelDetailTab.Details}
            className="min-h-0 flex-1 overflow-y-auto p-0"
          >
            <DetailsTab model={model} adapter={adapter} lineage={lineage} jobRecord={jobDetails} />
          </TabsContent>

          <TabsContent value={ModelDetailTab.Chat} className="flex min-h-0 flex-1 p-0">
            <Stack className="mx-auto h-full w-full max-w-[768px]">
              <ModelChatTab model={model} adapter={adapter} />
            </Stack>
          </TabsContent>

          {job.jobName && (
            <>
              <TabsContent
                value={ModelDetailTab.Training}
                className="min-h-0 flex-1 overflow-y-auto p-0 pr-density-md"
              >
                <Stack className="w-full">
                  <CustomizationOverview
                    customizationJobName={job.jobName}
                    workspace={job.workspace}
                  />
                </Stack>
              </TabsContent>
              <TabsContent value={ModelDetailTab.Logs} className="flex min-h-0 flex-1 p-0">
                <LogsTab
                  customizationJobName={job.jobName}
                  workspace={job.workspace}
                  jobStatus={jobDetails?.status}
                />
              </TabsContent>
            </>
          )}
        </TabsRoot>
      </Stack>
    </AccessibleTitle>
  );
};
