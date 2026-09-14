// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  ROW_SELECTION_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParam } from '@nemo/common/src/utils/query';
import {
  getModelsListModelsQueryKey,
  useModelsDeleteModel,
  useModelsDeleteModelAdapter,
  useModelsListModels,
} from '@nemo/sdk/generated/platform/models';
import type {
  Adapter,
  DatetimeFilter,
  FinetuningType,
  ModelEntity,
  ModelEntitySortField,
  ModelsListModelsParams,
} from '@nemo/sdk/generated/platform/schema';
import { Button, type DropdownEntry, Text, Tooltip } from '@nvidia/foundations-react-core';
import { queryClient } from '@studio/api/queryClient';
import {
  DEFAULT_CUSTOM_MODELS_FILTER,
  FINETUNING_TYPE_FILTER_OPTIONS,
  HAS_ADAPTERS,
} from '@studio/components/dataViews/CustomModelsDataView/constants';
import { DeploymentIndicator } from '@studio/components/dataViews/CustomModelsDataView/DeploymentIndicator';
import { KindTag } from '@studio/components/dataViews/CustomModelsDataView/KindTag';
import { BaseModelSearchFilterField } from '@studio/components/FilterFields';
import type { ModelPanelTab } from '@studio/components/sidePanels/ModelPanels/ModelPanel';
import { INTAKE_ENABLED } from '@studio/constants/environment';
import { getIntakeTracesRoute, getNewCustomizationJobRoute } from '@studio/routes/utils';
import { keepPreviousData } from '@tanstack/react-query';
import { Trash } from 'lucide-react';
import { ComponentProps, FC, useCallback, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

type SearchQuery = Record<string, unknown>;

interface CustomModelsFilterParams {
  name?: string;
  base_model?: string;
  finetuning_type?: FinetuningType;
  created_at?: DatetimeFilter;
  updated_at?: DatetimeFilter;
}

function buildCustomModelsFilter(fields: CustomModelsFilterParams): string | undefined {
  const conditions: SearchQuery[] = [];
  const { name, base_model, finetuning_type, created_at, updated_at } = fields;

  if (base_model) {
    conditions.push({
      $or: [
        { 'data.base_model': { $eq: base_model } },
        { $and: [{ name: { $eq: base_model } }, HAS_ADAPTERS] },
      ],
    });
  }

  if (finetuning_type) {
    if (finetuning_type === 'lora') {
      conditions.push({
        $or: [{ 'data.finetuning_type': { $eq: finetuning_type } }, HAS_ADAPTERS],
      });
    } else {
      conditions.push({
        'data.finetuning_type': { $eq: finetuning_type },
      });
    }
  }

  if (!base_model && !finetuning_type) {
    conditions.push(DEFAULT_CUSTOM_MODELS_FILTER);
  }

  if (name?.trim()) {
    conditions.push({ name: { $like: name.trim() } });
  }

  if (created_at) {
    conditions.push({ created_at });
  }

  if (updated_at) {
    conditions.push({ updated_at });
  }

  if (conditions.length === 0) return undefined;
  const query = conditions.length === 1 ? conditions[0] : { $and: conditions };
  return JSON.stringify(query);
}

export interface CustomModelsDataViewProps {
  workspace: string;
  placeholderComponent?: React.ReactNode;
  onRowClick?: (model: ModelEntity, tab: ModelPanelTab, adapter?: Adapter) => void;
}

type ModelTableRow = ModelEntity & {
  subRows?: ModelTableRow[];
  _parentModel?: ModelEntity;
};

export const CustomModelsDataView: FC<CustomModelsDataViewProps> = ({
  workspace,
  placeholderComponent,
  onRowClick,
}) => {
  const [deleteTarget, setDeleteTarget] = useState<
    | { kind: 'model'; name: string }
    | { kind: 'adapter'; adapterName: string; modelName: string }
    | null
  >(null);
  const navigate = useNavigate();
  const { mutateAsync: deleteModel } = useModelsDeleteModel();
  const { mutateAsync: deleteAdapter } = useModelsDeleteModelAdapter();

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
    columnPinning: {
      left: ['row-expansion', 'row-selection'],
      right: ['row-actions'],
    },
    columnVisibility: { updated_at: false },
    expansion: true,
  });

  const filterQuery = useMemo(() => {
    const filterMap = new Map(dataViewState.debouncedColumnFilters.map((f) => [f.id, f.value]));
    return buildCustomModelsFilter({
      name: dataViewState.debouncedSearchBar || undefined,
      base_model: filterMap.get('base_model') as string | undefined,
      finetuning_type: filterMap.get('finetuning_type') as FinetuningType | undefined,
      created_at: filterMap.get('created_at') as DatetimeFilter | undefined,
      updated_at: filterMap.get('updated_at') as DatetimeFilter | undefined,
    });
  }, [dataViewState.debouncedSearchBar, dataViewState.debouncedColumnFilters]);

  const {
    data: modelsResponse,
    refetch,
    isLoading,
    isError,
  } = useModelsListModels(
    workspace,
    {
      sort: getSortParam(dataViewState.sorting.state) as ModelEntitySortField,
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      filter: filterQuery as unknown as ModelsListModelsParams['filter'],
    },
    {
      query: {
        placeholderData: keepPreviousData,
        staleTime: 0,
        refetchOnWindowFocus: true,
      },
    }
  );

  const tableData = useMemo<ModelTableRow[]>(() => {
    const models = modelsResponse?.data || [];
    return models.map((model) => ({
      ...model,
      subRows: model.adapters?.length
        ? model.adapters.map<ModelTableRow>((adapter) => ({
            ...model,
            id: `${model.id}:adapter:${adapter.name}`,
            name: adapter.name,
            finetuning_type: adapter.finetuning_type,
            created_at: adapter.created_at ?? model.created_at,
            adapters: undefined,
            _parentModel: model,
            subRows: undefined,
          }))
        : undefined,
    }));
  }, [modelsResponse?.data]);

  const adapterMap = useMemo(() => {
    const map = new Map<string, Map<string, Adapter>>();
    for (const model of modelsResponse?.data ?? []) {
      if (model.adapters?.length) {
        map.set(model.id, new Map(model.adapters.map((a) => [a.name, a])));
      }
    }
    return map;
  }, [modelsResponse?.data]);

  const resetFilters = useCallback(() => {
    dataViewState.resetFilters();
  }, [dataViewState]);

  const handleKindClick = useCallback(
    (finetuningType: FinetuningType) => {
      dataViewState.columnFiltering.set((prev: DataView.TanstackTable.ColumnFiltersState) => {
        const filtered = prev.filter((f) => f.id !== 'finetuning_type');
        return [...filtered, { id: 'finetuning_type', value: finetuningType }];
      });
    },
    [dataViewState.columnFiltering]
  );

  const handleDelete = async () => {
    try {
      if (!deleteTarget) return false;
      if (deleteTarget.kind === 'model') {
        await deleteModel({ workspace, name: deleteTarget.name });
      } else {
        await deleteAdapter({
          workspace,
          modelName: deleteTarget.modelName,
          adapter: deleteTarget.adapterName,
        });
      }
      queryClient.invalidateQueries({ queryKey: getModelsListModelsQueryKey(workspace, {}) });
      return true;
    } catch {
      return false;
    }
  };

  const makeColumns: ComponentProps<typeof DataView.Root<ModelTableRow>>['makeColumns'] = (
    { accessor },
    { rowSelectionColumn, rowActionsColumn, rowExpansionColumn }
  ) => [
    rowExpansionColumn({ size: ROW_SELECTION_COLUMN_SIZE }),
    rowSelectionColumn({ size: ROW_SELECTION_COLUMN_SIZE }),
    accessor('id', {
      id: 'deployment-status',
      header: () => <span data-fixed-width aria-label="Deployment status" />,
      enableSorting: false,
      enableResizing: false,
      cell: ({ row }) => (
        <span data-fixed-width>
          {row.depth === 0 && (
            <DeploymentIndicator
              workspace={workspace}
              providerIds={row.original.model_providers}
              modelName={row.original.name ?? ''}
            />
          )}
        </span>
      ),
      size: 30,
      minSize: 30,
      maxSize: 30,
    }),
    accessor('name', {
      header: 'Name',
      enableSorting: true,
      cell: ({ row }) => <Text className="truncate">{row.original.name}</Text>,
    }),
    accessor((row) => row.base_model || '-', {
      id: 'base_model',
      header: 'Base Model',
      enableSorting: false,
      meta: {
        filter: {
          type: 'custom',
          label: 'Base Model',
          renderFilter: ({ setValue, value }) => (
            <BaseModelSearchFilterField
              dataTestId="customizations-base-model-filter"
              workspace={workspace}
              value={value as string | undefined}
              onValueChange={(models: string[]) => setValue(models[0] || undefined)}
              singleSelect
            />
          ),
        },
      },
      cell: ({ row }) => <Text className="truncate">{row.original.base_model || '-'}</Text>,
    }),
    accessor((row) => row.finetuning_type, {
      id: 'finetuning_type',
      header: 'Type',
      enableSorting: false,
      meta: {
        filter: {
          type: 'single-select',
          label: 'Finetuning Type',
          options: FINETUNING_TYPE_FILTER_OPTIONS,
        },
      },
      cell: ({ row }) =>
        row.original.finetuning_type ? (
          <KindTag finetuningType={row.original.finetuning_type} onClick={handleKindClick} />
        ) : (
          <Text>-</Text>
        ),
    }),
    accessor('created_at', {
      id: 'created_at',
      header: 'Created',
      enableSorting: true,
      size: 150,
      meta: {
        filter: dateTimeFilter('Created At'),
      },
      cell: ({ row }) =>
        row.original?.created_at ? <RelativeTime datetime={row.original.created_at} /> : null,
    }),
    accessor('updated_at', {
      id: 'updated_at',
      header: 'Updated',
      enableSorting: false,
      meta: {
        filter: dateTimeFilter('Updated At'),
      },
      cell: ({ row }) =>
        row.original?.updated_at ? <RelativeTime datetime={row.original.updated_at} /> : null,
    }),
    rowActionsColumn({
      size: ROW_ACTIONS_COLUMN_SIZE,
      enableResizing: false,
      rowActions: (row: ModelTableRow): DropdownEntry[] => {
        const isAdapter = Boolean(row._parentModel);
        const parentModel = row._parentModel;

        if (isAdapter && parentModel) {
          return [
            {
              children: 'Model details',
              onSelect: () => {
                const adapter = adapterMap.get(parentModel.id)?.get(row.name);
                onRowClick?.(parentModel, 'model-details', adapter);
              },
            },
            { kind: 'divider' as const },
            {
              children: 'Delete Adapter',
              danger: true,
              onSelect: () =>
                setDeleteTarget({
                  kind: 'adapter',
                  adapterName: row.name,
                  modelName: parentModel.name,
                }),
            },
          ];
        }

        return [
          {
            children: 'Model details',
            onSelect: () => onRowClick?.(row, 'model-details'),
          },
          {
            children: 'Chat Playground',
            onSelect: () => onRowClick?.(row, 'chat-playground'),
          },
          ...(INTAKE_ENABLED
            ? [
                {
                  children: 'View Intake',
                  onSelect: () => {
                    navigate(getIntakeTracesRoute(workspace));
                  },
                },
              ]
            : []),
          { kind: 'divider' as const },
          {
            children: 'Delete',
            danger: true,
            onSelect: () => setDeleteTarget({ kind: 'model', name: row.name }),
          },
        ];
      },
    }),
  ];

  const hasActiveFilters =
    Boolean(dataViewState.debouncedSearchBar) || dataViewState.debouncedColumnFilters.length > 0;

  return (
    <>
      <StudioDataView
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        renderBulkActions={() => (
          <Tooltip slotContent="Delete functionality is not yet available">
            <Button kind="tertiary" disabled aria-label="Delete selected models">
              <Trash /> Delete
            </Button>
          </Tooltip>
        )}
        onRowClick={(row: ModelTableRow) => {
          if (row._parentModel) {
            const adapter = adapterMap.get(row._parentModel.id)?.get(row.name);
            onRowClick?.(row._parentModel, 'model-details', adapter);
          } else {
            onRowClick?.(row, 'model-details');
          }
        }}
        attributes={{
          DataViewSearchBar: {
            placeholder: 'Search Custom Models...',
          },
          DataViewRoot: {
            data: tableData,
            totalCount: modelsResponse?.pagination?.total_results,
            requestStatus: isError ? 'error' : isLoading ? 'loading' : undefined,
            reactTableOptions: {
              getRowCanExpand: (row: { original: ModelTableRow }) =>
                Boolean(row.original.adapters?.length),
              enableSubRowSelection: false,
            },
          },
          DataViewTableContent: {
            renderErrorState: () => (
              <ErrorMessage
                header="Loading Error"
                message="There was an error loading custom models"
                slotFooter={
                  <Button type="button" kind="tertiary" onClick={() => refetch()}>
                    Retry
                  </Button>
                }
              />
            ),
            renderEmptyState: () =>
              hasActiveFilters ? (
                <EntityEmptyState
                  entity="customModels"
                  variant="no-results"
                  onClearFilters={resetFilters}
                />
              ) : placeholderComponent ? (
                <>{placeholderComponent}</>
              ) : (
                <EntityEmptyState
                  entity="customModels"
                  variant="first-use"
                  onCreate={() => navigate(getNewCustomizationJobRoute(workspace))}
                />
              ),
          },
        }}
      />
      {deleteTarget && (
        <DeleteConfirmationModal
          open
          title={`Delete ${deleteTarget.kind === 'adapter' ? 'Adapter' : 'Model'}`}
          onDelete={handleDelete}
          onClose={() => setDeleteTarget(null)}
          confirmationText={
            deleteTarget.kind === 'adapter' ? deleteTarget.adapterName : deleteTarget.name
          }
          successText={`Successfully deleted ${deleteTarget.kind} ${deleteTarget.kind === 'adapter' ? deleteTarget.adapterName : deleteTarget.name}`}
          simpleConfirm
        />
      )}
    </>
  );
};
