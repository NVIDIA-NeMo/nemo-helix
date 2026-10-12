// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { CustomModelsDataView } from '@studio/components/dataViews/CustomModelsDataView';
import { CustomizeModelButton } from '@studio/components/dataViews/CustomModelsDataView/CustomizeModelButton';
import { ModelPanel, ModelPanelTab } from '@studio/components/sidePanels/ModelPanels/ModelPanel';
import { INTAKE_ENABLED } from '@studio/constants/environment';
import { useCustomizationJobForModel } from '@studio/hooks/useCustomizationJobForModel';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getEvaluationResultsRoute, getIntakeTracesRoute } from '@studio/routes/utils';
import { type FC, useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router';

export const CustomizationJobListRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState<ModelEntity | null>(null);
  const [selectedAdapter, setSelectedAdapter] = useState<Adapter | null>(null);
  const [selectedTab, setSelectedTab] = useState<'model-details' | 'chat-playground'>(
    'model-details'
  );

  useBreadcrumbs({
    items: [{ slotLabel: 'Fine-tuning' }],
  });

  const { jobName: customizationJobName } = useCustomizationJobForModel(workspace, selectedModel);

  // `?model=` opens a model's panel directly, so anything outside this page -- a finished
  // job linking to the model it produced, most of all -- can point at one.
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedModel = searchParams.get('model');
  const { data: linkedModel } = useModelsGetModel(workspace, requestedModel ?? '', undefined, {
    query: { enabled: Boolean(requestedModel) && !selectedModel, retry: false },
  });

  useEffect(() => {
    if (!linkedModel) return;
    setSelectedModel(linkedModel);
    setSelectedTab('model-details');
    // Consumed, so reopening the panel after the user closes it takes another click
    // rather than happening on every render.
    setSearchParams(
      (params) => {
        params.delete('model');
        return params;
      },
      { replace: true }
    );
  }, [linkedModel, setSearchParams]);

  return (
    <AccessibleTitle title={`Fine-tuning for ${workspace}`}>
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Fine-tuning"
          slotDescription="Create, manage, and deploy fine-tuned models."
          slotActions={<CustomizeModelButton workspace={workspace} />}
        />
        <CustomModelsDataView
          workspace={workspace}
          onRowClick={(model: ModelEntity, tab: ModelPanelTab, adapter?: Adapter) => {
            setSelectedModel(model);
            setSelectedAdapter(adapter ?? null);
            setSelectedTab(tab);
          }}
        />
      </Stack>

      <ModelPanel
        open={!!selectedModel}
        model={selectedModel ?? undefined}
        customizationJobId={customizationJobName}
        adapter={selectedAdapter}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedModel(null);
            setSelectedAdapter(null);
          }
        }}
        defaultTab={selectedTab}
        overviewProps={{
          slotActions: (
            <Flex gap="density-md" align="center">
              {selectedModel && (
                <CustomizeModelButton model={selectedModel} workspace={workspace} />
              )}
              {INTAKE_ENABLED && (
                <Button
                  className="flex-1"
                  kind="secondary"
                  size="small"
                  onClick={() => navigate(getIntakeTracesRoute(workspace))}
                >
                  View Intake Traces
                </Button>
              )}
              <Button
                className="flex-1"
                kind="secondary"
                size="small"
                onClick={() => navigate(getEvaluationResultsRoute(workspace))}
              >
                Evaluate this Model
              </Button>
            </Flex>
          ),
        }}
      />
    </AccessibleTitle>
  );
};
