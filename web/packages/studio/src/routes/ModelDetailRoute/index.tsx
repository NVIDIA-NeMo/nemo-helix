// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { formatFinetuningType } from '@nemo/common/src/utils/formatters';
import { toInferenceModelName } from '@nemo/common/src/utils/models';
import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import { Badge, Stack, TabsContent } from '@nvidia/foundations-react-core';
import { CustomizationOverview } from '@studio/components/CustomizationOverview';
import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import { EvaluationResultsDataView } from '@studio/components/dataViews/EvaluationResultsDataView';
import { DetailPageLayout, type DetailPageTab } from '@studio/components/Layouts/DetailPageLayout';
import { Loading } from '@studio/components/Layouts/Loading';
import { DEFAULT_NAMESPACE } from '@studio/constants/constants';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useCustomizationJob } from '@studio/hooks/useCustomizationJob';
import { useModelLineage } from '@studio/hooks/useModelLineage';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { LogsTab } from '@studio/routes/CustomizationJobDetailsRoute/LogsTab';
import { ChatTab } from '@studio/routes/ModelDetailRoute/ChatTab';
import { type ModelDetailParent, ModelDetailTab } from '@studio/routes/ModelDetailRoute/constants';
import { DeploymentTab } from '@studio/routes/ModelDetailRoute/DeploymentTab';
import { DetailsTab } from '@studio/routes/ModelDetailRoute/DetailsTab';
import { FilesTab } from '@studio/routes/ModelDetailRoute/FilesTab';
import { ModelDetailActions } from '@studio/routes/ModelDetailRoute/ModelDetailActions';
import { useModelTabRoute } from '@studio/routes/ModelDetailRoute/useModelTabRoute';
import {
  getCustomizationJobListRoute,
  getEvaluationNewRoute,
  getModelDetailRoute,
  getWorkspaceBaseModelsRoute,
} from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';
import { useNavigate, useParams } from 'react-router';

export interface ModelDetailRouteProps {
  parent?: ModelDetailParent;
}

export const ModelDetailRoute: FC<ModelDetailRouteProps> = ({ parent = 'fine-tuning' }) => {
  const workspace = useWorkspaceFromPath();
  const { modelName } = useRequiredPathParams([ROUTE_PARAMS.modelName]);
  const { [ROUTE_PARAMS.adapterName]: adapterName, [ROUTE_PARAMS.modelTab]: tabFromUrl } =
    useParams();
  const getTabRoute = useModelTabRoute();
  const navigate = useNavigate();

  // The catalog lists the `default` workspace's models too, so a catalog page falls back to it.
  const lineage = useModelLineage(workspace, modelName, adapterName, {
    fallbackWorkspace: parent === 'catalog' ? DEFAULT_NAMESPACE : undefined,
  });
  const { model, adapter, job, fileset } = lineage;
  const name = adapterName ?? modelName;

  const { job: jobDetails } = useCustomizationJob(job.workspace, job.jobName ?? '', {
    enabled: Boolean(job.jobName),
    refetchInterval: (query) => getJobRefetchInterval(query.state.data?.status),
  });

  useBreadcrumbs({
    items: [
      parent === 'catalog'
        ? { href: getWorkspaceBaseModelsRoute(workspace), slotLabel: 'Model Catalog' }
        : { href: getCustomizationJobListRoute(workspace), slotLabel: 'Fine-tuning' },
      ...(adapterName
        ? [{ href: getModelDetailRoute(workspace, modelName), slotLabel: modelName }]
        : []),
      { slotLabel: name },
    ],
  });

  // The job tabs stay visible but disabled while the lookup runs, and go away if there is no job.
  const hasJobTabs = job.isLoading || Boolean(job.jobName);
  const tabs: DetailPageTab<ModelDetailTab>[] = [
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

  const selectedTab = tabs.find((tab) => tab.value === tabFromUrl)?.value ?? ModelDetailTab.Details;

  // An adapter is named with the `model::adapter` encoding the wizard's picker
  // round-trips, so the run opens pre-set to the entity whose page this is.
  const runEvaluationRoute = getEvaluationNewRoute(workspace, {
    model: adapterName ? `${workspace}/${modelName}::${adapterName}` : `${workspace}/${modelName}`,
  });

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
      <DetailPageLayout
        title={name}
        slotBadges={
          <>
            <DeploymentStatusBadge state={lineage.deployment} isAdapter={Boolean(adapter)} />
            {tags.map((tag) => (
              <Badge key={tag} kind="solid" color="gray">
                {tag}
              </Badge>
            ))}
          </>
        }
        slotDescription={(adapter ?? model).description}
        tabs={tabs}
        selectedTab={selectedTab}
        onTabChange={(tab) => navigate(getTabRoute(tab))}
        slotActions={
          <ModelDetailActions
            selectedTab={selectedTab}
            parent={parent}
            model={model}
            adapter={adapter}
            lineage={lineage}
            runEvaluationRoute={runEvaluationRoute}
          />
        }
      >
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
      </DetailPageLayout>
    </AccessibleTitle>
  );
};
