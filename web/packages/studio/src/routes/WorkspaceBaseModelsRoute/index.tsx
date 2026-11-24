/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { useBaseModels } from '@nemo/common/src/api/entity-store/useBaseModels';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getModelEntityChatStatus } from '@nemo/common/src/utils/models';
import { getSortParam } from '@nemo/common/src/utils/query';
import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import type { ModelEntity, ModelEntitySortField } from '@nemo/sdk/generated/platform/schema';
import {
  Checkbox,
  Flex,
  PageHeader,
  Select,
  Spinner,
  Stack,
  Text,
  Tooltip,
} from '@nvidia/foundations-react-core';
import { BaseModelCard } from '@studio/components/BaseModelCard';
import { CustomizeModelButton } from '@studio/components/dataViews/CustomModelsDataView/CustomizeModelButton';
import {
  ModelPanel,
  type ModelPanelTab,
} from '@studio/components/sidePanels/ModelPanels/ModelPanel';
import { VirtualizedCardGrid } from '@studio/components/VirtualizedCardGrid';
import { CUSTOMIZER_ENABLED, MODEL_DETAILS_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getWorkspaceBaseModelsRoute } from '@studio/routes/utils';
import {
  CUSTOMIZABLE_FILTER_ID,
  FINE_TUNABLE_KEY,
  makeFilterColumns,
  toModelEntityFilter,
} from '@studio/routes/WorkspaceBaseModelsRoute/filterColumns';
import { tooltipClassName } from '@studio/styles/common';
import { useEffect, useMemo, useRef, useState, type FC } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router';

const SORT_OPTIONS = [
  { value: 'name', children: 'Alphabetical (A-Z)' },
  { value: '-name', children: 'Alphabetical (Z-A)' },
  { value: 'created_at', children: 'Created (Newest First)' },
  { value: '-created_at', children: 'Created (Oldest First)' },
];

const TAB_SEARCH_PARAM = 'tab';

const getRequestStatus = (isLoading: boolean, isError: boolean) => {
  if (isLoading) return 'loading';
  if (isError) return 'error';
  return 'success';
};

export const WorkspaceBaseModelsRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { modelName: modelNameParam } = useParams<{ modelName?: string }>();
  const tabFromUrl = (searchParams.get(TAB_SEARCH_PARAM) ?? 'model-details') as ModelPanelTab;

  useBreadcrumbs({
    items: [{ slotLabel: 'Model Catalog' }],
  });

  // Column filter ids aren't all API fields (e.g. `customizable`, `lora`); toModelEntityFilter maps them.
  const dataViewState = useStudioDataViewState<Record<string, unknown>>({
    defaultSort: [{ id: 'name', desc: false }],
  });

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [selectedModel, setSelectedModel] = useState<ModelEntity | null>(null);

  // `useParams()` already returns decoded values; route helpers handle encoding.
  const modelNameFromPath = modelNameParam ?? '';

  const nameSearch = dataViewState.apiFilter.searchText;
  const allColumnFilters = dataViewState.apiFilter.filter;
  const filter = useMemo(
    () => toModelEntityFilter(allColumnFilters, nameSearch),
    [allColumnFilters, nameSearch]
  );
  const makeColumns = useMemo(() => makeFilterColumns(workspace), [workspace]);
  const hasActiveFilters = !!filter;

  const customizableChecked = CUSTOMIZER_ENABLED
    ? !!(
        dataViewState.columnFiltering.state.find((f) => f.id === CUSTOMIZABLE_FILTER_ID)?.value as
          | Partial<Record<typeof FINE_TUNABLE_KEY, true>>
          | undefined
      )?.[FINE_TUNABLE_KEY]
    : false;

  const handleCustomizableToggle = (checked: boolean) => {
    dataViewState.columnFiltering.set((prev) => {
      const others = prev.filter((f) => f.id !== CUSTOMIZABLE_FILTER_ID);
      if (!checked) return others;
      return [...others, { id: CUSTOMIZABLE_FILTER_ID, value: { [FINE_TUNABLE_KEY]: true } }];
    });
  };

  const sort = getSortParam(dataViewState.sorting.state) as ModelEntitySortField;

  const {
    models,
    isLoading,
    isError,
    error,
    hasNextPage,
    fetchNextPage,
    isFetchNextPageError,
    refetch,
  } = useBaseModels({
    workspace,
    filter,
    sort,
  });

  const modelInList = useMemo(
    () => !!modelNameFromPath && models.some((m) => m.name === modelNameFromPath),
    [modelNameFromPath, models]
  );
  const { data: modelFromApi, isLoading: isModelFromApiLoading } = useModelsGetModel(
    workspace,
    modelNameFromPath ?? '',
    undefined,
    {
      query: {
        enabled: !!workspace && !!modelNameFromPath && !modelInList,
      },
    }
  );

  const resolvedModelFromPath = useMemo(() => {
    if (!modelNameFromPath) return null;
    const fromList = models.find((m) => m.name === modelNameFromPath);
    if (fromList) return fromList;
    if (modelFromApi?.name === modelNameFromPath) return modelFromApi;
    return null;
  }, [modelNameFromPath, models, modelFromApi]);

  useEffect(() => {
    if (modelNameFromPath && resolvedModelFromPath) {
      setSelectedModel(resolvedModelFromPath);
    }
  }, [modelNameFromPath, resolvedModelFromPath]);

  const handleOpenModel = (model: ModelEntity) => {
    if (MODEL_DETAILS_ENABLED) {
      navigate(getWorkspaceBaseModelsRoute(workspace, { model: model.name }));
      return;
    }
    setSelectedModel(model);
    navigate(getWorkspaceBaseModelsRoute(workspace, { model: model.name, searchParams }), {
      replace: true,
    });
  };

  const handleClosePanel = () => {
    const listSearchParams = new URLSearchParams(searchParams);
    listSearchParams.delete(TAB_SEARCH_PARAM);

    setSelectedModel(null);
    navigate(getWorkspaceBaseModelsRoute(workspace, { searchParams: listSearchParams }), {
      replace: true,
    });
  };

  /** Base models can only be deleted when no `model_providers` entries reference them. */
  const allowModelDelete = !!selectedModel && !(selectedModel.model_providers?.length ?? 0);

  const sortSelectValue = dataViewState.sorting.state[0]
    ? dataViewState.sorting.state[0].desc
      ? `-${dataViewState.sorting.state[0].id}`
      : dataViewState.sorting.state[0].id
    : 'name';

  return (
    <AccessibleTitle title="Model Catalog">
      <ModelPanel
        allowModelDelete={allowModelDelete}
        onModelDeleted={() => {
          void refetch();
        }}
        open={
          !!selectedModel ||
          !!(modelNameFromPath && (!!resolvedModelFromPath || isModelFromApiLoading))
        }
        loading={!!modelNameFromPath && !resolvedModelFromPath && isModelFromApiLoading}
        overviewProps={{
          slotActions: (
            <Flex gap="density-md" align="center">
              {CUSTOMIZER_ENABLED && selectedModel && (
                <CustomizeModelButton model={selectedModel} workspace={workspace} />
              )}
            </Flex>
          ),
        }}
        model={selectedModel ?? undefined}
        showCustomizationDetails={CUSTOMIZER_ENABLED}
        defaultTab={tabFromUrl}
        onTabChange={(tab) =>
          setSearchParams(
            (prev) => {
              const next = new URLSearchParams(prev);
              next.set(TAB_SEARCH_PARAM, tab);
              return next;
            },
            { replace: true }
          )
        }
        onOpenChange={(open) => !open && handleClosePanel()}
      />
      <Stack className="h-full min-h-0" gap="density-2xl" padding="density-2xl">
        <PageHeader className="p-0 shrink-0" slotHeading="Model Catalog" />
        <StudioDataView<ModelEntity>
          dataViewState={dataViewState}
          makeColumns={makeColumns}
          searchField="name"
          scrollContainerRef={scrollContainerRef}
          toolbarSlotEnd={
            <Flex gap="density-md" align="center">
              {CUSTOMIZER_ENABLED && (
                <Tooltip
                  slotContent={
                    <div className={tooltipClassName}>
                      <Text>
                        Show only models that can be fine-tuned. Inference-only models (e.g. from
                        custom providers) are hidden.
                      </Text>
                    </div>
                  }
                  side="bottom"
                >
                  <Flex
                    align="center"
                    className="h-10 shrink-0 px-density-md rounded-md border border-base bg-surface-raised"
                  >
                    <Checkbox
                      attributes={{
                        CheckboxInput: { id: 'base-models-filter-customizable' },
                        Label: {
                          htmlFor: 'base-models-filter-customizable',
                          // "Fine-tunable" otherwise wraps at the hyphen when the toolbar is tight.
                          className: 'whitespace-nowrap',
                        },
                      }}
                      checked={customizableChecked}
                      slotLabel="Fine-tunable"
                      onCheckedChange={(checked) => handleCustomizableToggle(!!checked)}
                    />
                  </Flex>
                </Tooltip>
              )}
              <Select
                className="w-fit"
                items={SORT_OPTIONS}
                value={sortSelectValue}
                onValueChange={(value) => {
                  const desc = value.startsWith('-');
                  dataViewState.sorting.set([{ id: desc ? value.slice(1) : value, desc }]);
                }}
              />
            </Flex>
          }
          attributes={{
            DataViewRoot: {
              data: models,
              totalCount: models.length,
              requestStatus: getRequestStatus(isLoading, isError || isFetchNextPageError),
            },
            DataViewSearchBar: { placeholder: 'Search Models...' },
          }}
        >
          <DataView.CustomContent<ModelEntity>
            renderLoadingState={() => (
              <Flex align="center" justify="center" className="h-full">
                <Spinner size="large" description="Loading base models..." />
              </Flex>
            )}
            renderEmptyState={() =>
              hasActiveFilters ? (
                <EntityEmptyState
                  entity="baseModels"
                  variant="no-results"
                  onClearFilters={dataViewState.resetFilters}
                />
              ) : (
                <EntityEmptyState entity="baseModels" variant="first-use" />
              )
            }
            renderErrorState={() => (
              <ErrorPanel
                errorMessage={getErrorMessage(error ?? new Error('Failed to load base models.'))}
              />
            )}
          >
            {({ rows }) => (
              <VirtualizedCardGrid
                items={rows.map((r) => r.original)}
                renderCard={(model) => (
                  <BaseModelCard
                    model={model}
                    isChatAvailable={getModelEntityChatStatus(model) === 'enabled'}
                    showCustomizationBadges={CUSTOMIZER_ENABLED}
                    onClick={() => handleOpenModel(model)}
                  />
                )}
                getItemKey={(model) => `${model.workspace}/${model.name}`}
                scrollContainerRef={scrollContainerRef}
                hasMore={hasNextPage}
                onLoadMore={fetchNextPage}
              />
            )}
          </DataView.CustomContent>
        </StudioDataView>
      </Stack>
    </AccessibleTitle>
  );
};
