// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { formatFinetuningType } from '@nemo/common/src/utils/formatters';
import { toInferenceModelName } from '@nemo/common/src/utils/models';
import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import {
  Badge,
  Button,
  Flex,
  PageHeader,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import { CustomizationOverview } from '@studio/components/CustomizationOverview';
import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import { EvaluationResultsDataView } from '@studio/components/dataViews/EvaluationResultsDataView';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useCustomizationJob } from '@studio/hooks/useCustomizationJob';
import { useModelLineage } from '@studio/hooks/useModelLineage';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { LogsTab } from '@studio/routes/CustomizationJobDetailsRoute/LogsTab';
import { ChatTab } from '@studio/routes/ModelDetailRoute/ChatTab';
import { ModelDetailTab } from '@studio/routes/ModelDetailRoute/constants';
import { DeploymentTab } from '@studio/routes/ModelDetailRoute/DeploymentTab';
import { DeploymentActions } from '@studio/routes/ModelDetailRoute/DeploymentTab/DeploymentActions';
import { DetailsTab } from '@studio/routes/ModelDetailRoute/DetailsTab';
import { FilesTab } from '@studio/routes/ModelDetailRoute/FilesTab';
import {
  getCustomizationJobListRoute,
  getEvaluationNewRoute,
  getFilesetRoute,
  getModelDetailRoute,
  getWorkspaceCustomizationJobDetailsRoute,
} from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';
import { Link, useNavigate, useParams } from 'react-router';

export const ModelDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { modelName } = useRequiredPathParams([ROUTE_PARAMS.modelName]);
  const { [ROUTE_PARAMS.adapterName]: adapterName } = useParams();
  const { getQueryParam, setQueryParam } = useQueryParams();
  const navigate = useNavigate();

  const lineage = useModelLineage(workspace, modelName, adapterName);
  const { model, adapter, job, fileset } = lineage;
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
  const tabs: { value: ModelDetailTab; label: string; disabled?: boolean }[] = [
    { value: ModelDetailTab.Details, label: 'Details' },
    { value: ModelDetailTab.Chat, label: 'Chat' },
    { value: ModelDetailTab.Evaluations, label: 'Evaluations' },
    { value: ModelDetailTab.Files, label: 'Files' },
    { value: ModelDetailTab.Deployment, label: 'Deployment' },
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
  // An adapter is named with the `model::adapter` encoding the wizard's picker
  // round-trips, so the run opens pre-set to the entity whose page this is.
  const runEvaluationRoute = getEvaluationNewRoute(workspace, {
    model: adapterName ? `${workspace}/${modelName}::${adapterName}` : `${workspace}/${modelName}`,
  });

  let headerAction: { label: string; to: string } | undefined;
  if (isJobTab && job.jobName) {
    headerAction = {
      label: 'Open Job',
      to: getWorkspaceCustomizationJobDetailsRoute(job.workspace, job.jobName),
    };
  } else if (selectedTab === ModelDetailTab.Evaluations) {
    headerAction = { label: 'Run Evaluation', to: runEvaluationRoute };
  } else if (selectedTab === ModelDetailTab.Files && fileset) {
    headerAction = {
      label: 'Open Fileset',
      to: getFilesetRoute(fileset.workspace, `${fileset.workspace}/${fileset.name}`),
    };
  }

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

  // What a run of this entity stores in `spec.target.name`. For an adapter that
  // is the composite the gateway resolves -- NOT its bare name, and not the
  // `model::adapter` the wizard's picker puts in a URL. `toInferenceModelName`
  // is the single definition of that value, shared with the submit path, so the
  // two cannot drift apart into a tab that silently lists nothing.
  const evaluatedName = toInferenceModelName(model, adapter);

  const finetuningType = adapter?.finetuning_type ?? model.finetuning_type;
  const tags = [
    finetuningType ? formatFinetuningType(finetuningType) : undefined,
    adapter ? `Model: ${model.name}` : model.base_model && `Base: ${model.base_model}`,
  ].filter(Boolean);

  return (
    <AccessibleTitle title={`Model details for ${name}`}>
      <Stack className="h-full min-h-0" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="shrink-0 p-0"
          slotHeading={
            <Flex align="baseline" gap="3" wrap="wrap">
              <Text kind="title/md">{name}</Text>
              <DeploymentStatusBadge state={lineage.deployment} isAdapter={Boolean(adapter)} />
              {tags.map((tag) => (
                <Badge key={tag} kind="outline" color="gray">
                  {tag}
                </Badge>
              ))}
            </Flex>
          }
          slotDescription={(adapter ?? model).description}
          slotActions={
            selectedTab === ModelDetailTab.Deployment ? (
              <DeploymentActions
                model={model}
                adapter={adapter}
                deployment={lineage.baseDeployment}
              />
            ) : (
              headerAction && (
                <Button color="brand" asChild>
                  <Link to={headerAction.to}>{headerAction.label}</Link>
                </Button>
              )
            )
          }
        />

        <TabsRoot
          className="flex min-h-0 flex-1 flex-col"
          value={selectedTab}
          onValueChange={(value) => setQueryParam(QUERY_PARAMETERS.tab, value)}
        >
          <TabsList className="shrink-0">
            {tabs.map((tab) => (
              <TabsTrigger key={tab.value} value={tab.value} disabled={tab.disabled}>
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent
            value={ModelDetailTab.Details}
            className="min-h-0 flex-1 overflow-y-auto p-0 pt-6"
          >
            <DetailsTab model={model} adapter={adapter} lineage={lineage} jobRecord={jobDetails} />
          </TabsContent>

          <TabsContent value={ModelDetailTab.Chat} className="flex min-h-0 flex-1 p-0 pt-6">
            <ChatTab
              // The baseline panel is seeded once, so start over on another model or adapter.
              key={`${model.workspace}/${model.name}:${adapter?.name ?? ''}`}
              model={model}
              adapter={adapter}
            />
          </TabsContent>

          <TabsContent
            value={ModelDetailTab.Evaluations}
            className="min-h-0 flex-1 overflow-y-auto p-0 pt-6"
          >
            <EvaluationResultsDataView
              targetModelName={evaluatedName}
              onCreate={() => navigate(runEvaluationRoute)}
            />
          </TabsContent>

          <TabsContent
            value={ModelDetailTab.Files}
            className="flex min-h-0 flex-1 flex-col items-stretch overflow-y-auto p-0 pt-6"
          >
            <FilesTab fileset={fileset} />
          </TabsContent>

          <TabsContent
            value={ModelDetailTab.Deployment}
            className="flex min-h-0 flex-1 flex-col items-stretch overflow-y-auto p-0 pt-6"
          >
            <DeploymentTab model={model} adapter={adapter} deployment={lineage.baseDeployment} />
          </TabsContent>

          {job.jobName && (
            <>
              <TabsContent
                value={ModelDetailTab.Training}
                className="min-h-0 flex-1 overflow-y-auto p-0 pt-6 pr-density-md"
              >
                <Stack className="w-full">
                  <CustomizationOverview
                    customizationJobName={job.jobName}
                    workspace={job.workspace}
                  />
                </Stack>
              </TabsContent>
              <TabsContent value={ModelDetailTab.Logs} className="flex min-h-0 flex-1 p-0 pt-6">
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
