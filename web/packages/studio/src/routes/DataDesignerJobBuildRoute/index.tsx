// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { DEFAULT_LARGE_PAGE_SIZE } from '@nemo/common/src/constants/api';
import { useDataDesignerCreateJob } from '@nemo/sdk/generated/data-designer/data-designer';
import { useModelsListProviders } from '@nemo/sdk/generated/platform/model-providers';
import { Flex, Stack } from '@nvidia/foundations-react-core';
import { findTemplate } from '@studio/components/CreateFilesetStart/templates';
import { usePreview } from '@studio/components/NewDataDesignerJobForm/usePreview';
import { getCloneJobRequestFromState } from '@studio/components/NewDataDesignerJobForm/utils';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import {
  getGeneratedJobRequestFromState,
  seedFromJobRequest,
} from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';
import { BuilderCanvas } from '@studio/routes/DataDesignerJobBuildRoute/BuilderCanvas';
import { BuilderConfigPane } from '@studio/routes/DataDesignerJobBuildRoute/BuilderConfigPane';
import { BuilderDetailsPanel } from '@studio/routes/DataDesignerJobBuildRoute/BuilderDetailsPanel';
import { BuilderPalette } from '@studio/routes/DataDesignerJobBuildRoute/BuilderPalette';
import {
  BuilderToolbar,
  type BuilderViewMode,
} from '@studio/routes/DataDesignerJobBuildRoute/BuilderToolbar';
import {
  buildColumnsFromConfig,
  buildDataDesignerConfig,
  validateColumns,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';
import { MissingTemplateModelBanner } from '@studio/routes/DataDesignerJobBuildRoute/MissingTemplateModelBanner';
import {
  buildModelsFromConfig,
  buildServedModelNames,
  validateModels,
} from '@studio/routes/DataDesignerJobBuildRoute/models';
import { SchemaList } from '@studio/routes/DataDesignerJobBuildRoute/SchemaList';
import {
  type JobBuilderSeed,
  unresolvedTemplateModelIssues,
  useJobBuilder,
} from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';
import {
  getDataDesignerJobDetailsRoute,
  getDataDesignerJobListRoute,
  getNewDataDesignerJobRoute,
} from '@studio/routes/utils';
import { type FC, useCallback, useMemo, useState } from 'react';
import { FormProvider } from 'react-hook-form';
import { useAuth } from 'react-oidc-context';
import { useLocation, useNavigate, useSearchParams } from 'react-router';

/**
 * Edges are derived from entered values: Jinja2 `{{ column_name }}` references (and
 * column-name fields) draw edges so the graph reflects data dependencies, not add order.
 */
export const DataDesignerJobBuildRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchParams] = useSearchParams();
  const { state: locationState } = useLocation();

  const template = useMemo(() => {
    const templateId = searchParams.get('template');
    return templateId ? (findTemplate(templateId) ?? null) : null;
  }, [searchParams]);

  const cloneSeed = useMemo<JobBuilderSeed | null>(() => {
    const cloneRequest = getCloneJobRequestFromState(locationState);
    if (!cloneRequest?.spec) return null;
    const { config, num_records } = cloneRequest.spec;
    return {
      name: cloneRequest.name ?? 'untitled-dataset',
      rows: String(num_records),
      columns: buildColumnsFromConfig(config),
      models: buildModelsFromConfig(config.model_configs),
    };
  }, [locationState]);

  const generatedSeed = useMemo<JobBuilderSeed | null>(() => {
    const generatedRequest = getGeneratedJobRequestFromState(locationState);
    return generatedRequest?.spec ? seedFromJobRequest(generatedRequest) : null;
  }, [locationState]);

  const heading = cloneSeed
    ? `Clone of ${cloneSeed.name}`
    : generatedSeed
      ? 'Describe with AI'
      : template
        ? template.title
        : 'Build from scratch';

  useBreadcrumbs({
    items: [
      { href: getDataDesignerJobListRoute(workspace), slotLabel: 'Data Designer' },
      { href: getNewDataDesignerJobRoute(workspace), slotLabel: 'New fileset' },
      { slotLabel: heading },
    ],
  });

  const builder = useJobBuilder(template, workspace, cloneSeed ?? generatedSeed);
  const { data: providersPage } = useModelsListProviders(
    workspace,
    { page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: {} }
  );
  const servedModelNames = useMemo(
    () => buildServedModelNames(providersPage?.data ?? []),
    [providersPage?.data]
  );

  const [viewMode, setViewMode] = useState<BuilderViewMode>('list');
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [isDetailsOpen, setIsDetailsOpen] = useState(false);

  const validateAndCollectErrors = useCallback(() => {
    const { columns, models, name, rows } = builder.getBuilderValues();
    const numRecords = Number(rows);
    const errors = [
      ...validateColumns(columns),
      ...validateModels(models),
      // The template's model survives in the form, so validateModels sees a filled-in field;
      // without this the job would submit and fail server-side on the unknown model.
      ...unresolvedTemplateModelIssues(builder.templateModelIssues, models).map(
        (issue) =>
          `${issue.alias}: ${issue.requested} isn't available in this workspace. Select a model that is.`
      ),
    ];
    if (!name.trim()) {
      errors.push('Fileset name is required.');
    }
    if (!Number.isInteger(numRecords) || numRecords < 1) {
      errors.push('Records to generate must be a whole number of at least 1.');
    }
    setValidationErrors(errors);
    setIsDetailsOpen(true);
    return errors;
  }, [builder]);

  const getCurrentConfig = useCallback(() => {
    const { columns, models } = builder.getBuilderValues();
    return validateColumns(columns).length === 0 && validateModels(models).length === 0
      ? buildDataDesignerConfig(columns, models, servedModelNames)
      : undefined;
  }, [builder, servedModelNames]);
  const { previewLogs, isPreviewing, runPreview, stopPreview } = usePreview({
    workspace,
    accessToken: user?.access_token ?? undefined,
    getCurrentConfig,
  });

  const handlePreview = useCallback(() => {
    if (validateAndCollectErrors().length > 0) return;
    setIsDetailsOpen(true);
    void runPreview();
  }, [validateAndCollectErrors, runPreview]);

  const createJob = useDataDesignerCreateJob();
  const submitError = createJob.error ? getErrorMessage(createJob.error) : null;

  const handleSubmit = useCallback(async () => {
    if (validateAndCollectErrors().length > 0) return;
    const { columns, models, name, rows } = builder.getBuilderValues();

    try {
      const created = await createJob.mutateAsync({
        workspace,
        data: {
          name,
          spec: {
            num_records: Number(rows),
            config: buildDataDesignerConfig(columns, models, servedModelNames),
          },
        },
      });
      if (created?.name) {
        navigate(getDataDesignerJobDetailsRoute(workspace, created.name));
      } else {
        navigate(getDataDesignerJobListRoute(workspace));
      }
    } catch {
      setIsDetailsOpen(true);
      // Error surfaced via createJob.error / submitError below.
    }
  }, [validateAndCollectErrors, builder, createJob, workspace, servedModelNames, navigate]);

  const toggleDetails = useCallback(() => setIsDetailsOpen((open) => !open), []);
  const onColumnRemove = useCallback(() => {
    if (builder.selectedColumnId) builder.removeColumn(builder.selectedColumnId);
  }, [builder]);
  const onColumnClose = useCallback(() => builder.selectColumn(null), [builder]);
  const onModelRemove = useCallback(() => {
    if (builder.selectedModelId) builder.removeModel(builder.selectedModelId);
  }, [builder]);
  const onModelClose = useCallback(() => builder.selectModel(null), [builder]);

  return (
    <AccessibleTitle title={heading}>
      <FormProvider {...builder.form}>
        <Stack className=" h-full">
          <BuilderToolbar
            templateTag={template?.tag}
            columnCount={builder.columnCount}
            viewMode={viewMode}
            onViewModeChange={setViewMode}
            onPreview={handlePreview}
            isPreviewing={isPreviewing}
            onStopPreview={stopPreview}
            onSubmit={handleSubmit}
            isSubmitting={createJob.isPending}
          />

          <MissingTemplateModelBanner issues={builder.templateModelIssues} />

          <BuilderDetailsPanel
            validationErrors={validationErrors}
            submitError={submitError}
            previewLogs={previewLogs}
            isOpen={isDetailsOpen}
            onToggle={toggleDetails}
          />

          <Flex className="min-h-0 border-t border-base h-full">
            <BuilderPalette
              tab={builder.paletteTab}
              onTabChange={builder.setPaletteTab}
              selectedModelId={builder.selectedModelId}
              workspace={workspace}
              onAddColumn={builder.handleAddColumn}
              onAddModel={builder.handleAddModel}
              onSelectModel={builder.selectModel}
            />

            <div className="relative min-w-0 flex-1">
              {viewMode === 'list' ? (
                <SchemaList
                  selectedId={builder.selectedColumnId}
                  onSelect={builder.selectColumn}
                  onDelete={builder.removeColumn}
                />
              ) : (
                <BuilderCanvas
                  focusNodeId={builder.focusId}
                  onNodeClick={builder.selectColumn}
                  onNodeDelete={builder.removeColumn}
                />
              )}
            </div>

            <BuilderConfigPane
              selectedColumnId={builder.selectedColumnId}
              selectedModelId={builder.selectedModelId}
              workspace={workspace}
              onColumnRemove={onColumnRemove}
              onColumnClose={onColumnClose}
              onModelRemove={onModelRemove}
              onModelClose={onModelClose}
            />
          </Flex>
        </Stack>
      </FormProvider>
    </AccessibleTitle>
  );
};
