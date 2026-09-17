// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { withOperators } from '@nemo/common/src/api/filterOperators';
import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import type {
  DatetimeFilter,
  VirtualModel,
  VirtualModelFilter,
} from '@nemo/sdk/generated/platform/schema';
import {
  getGetVirtualModelQueryKey,
  getListVirtualModelsQueryKey,
  useDeleteVirtualModel,
  useListVirtualModels,
} from '@nemo/sdk/generated/platform/virtual-models';
import { type DropdownEntry, Stack, Text } from '@nvidia/foundations-react-core';
import { BaseModelSearchFilterField } from '@studio/components/FilterFields';
import { getVirtualModelChatRoute, getVirtualModelDetailsRoute } from '@studio/routes/utils';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';

export interface VirtualModelsDataViewProps {
  workspace: string;
  /**
   * Restrict the list to virtual models applying this guardrail config, as a
   * `"workspace/name"` reference. Fixed scope rather than a user filter: it is applied
   * outside `dataViewState` so clearing filters cannot widen the view past it.
   */
  guardrailConfig?: string;
  attributes?: {
    Stack?: React.ComponentProps<typeof Stack>;
  };
}

type VirtualModelWithId = VirtualModel & { id: string };

const middlewareCount = (vm: VirtualModel): number =>
  (vm.request_middleware?.length ?? 0) +
  (vm.response_middleware?.length ?? 0) +
  (vm.post_response_middleware?.length ?? 0);

export const VirtualModelsDataView: FC<VirtualModelsDataViewProps> = ({
  workspace,
  guardrailConfig,
  attributes,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const openRow = useRowNavigation();

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const [modalVirtualModel, setModalVirtualModel] = useState<VirtualModel>();

  const openVirtualModel = useCallback(
    (virtualModel: VirtualModelWithId, tab: 'details' | 'chat', event?: React.MouseEvent) => {
      if (!virtualModel.name) return;

      // The row already holds the full virtual model, so prime the detail query — the detail page
      // then renders immediately instead of showing a skeleton. It still revalidates in background.
      queryClient.setQueryData(
        getGetVirtualModelQueryKey(workspace, virtualModel.name),
        virtualModel
      );

      openRow(
        event,
        tab === 'chat'
          ? getVirtualModelChatRoute(workspace, virtualModel.name)
          : getVirtualModelDetailsRoute(workspace, virtualModel.name)
      );
    },
    [openRow, queryClient, workspace]
  );

  const sortState = dataViewState.sorting.state[0];
  const sortParam = sortState
    ? sortState.desc
      ? `-${sortState.id}`
      : sortState.id
    : '-created_at';

  const filter = useMemo<VirtualModelFilter>(() => {
    const columnFilters = new Map(dataViewState.debouncedColumnFilters.map((f) => [f.id, f.value]));
    const name = dataViewState.debouncedSearchBar || undefined;
    const defaultModelEntity = columnFilters.get('default_model_entity') as string | undefined;
    // API requires workspace-qualified form ("default/model-name"); dropdown yields bare names.
    const qualifiedDefaultModelEntity = defaultModelEntity
      ? defaultModelEntity.includes('/')
        ? defaultModelEntity
        : `${workspace}/${defaultModelEntity}`
      : undefined;
    const createdAt = columnFilters.get('created_at') as DatetimeFilter | undefined;
    return withOperators<VirtualModelFilter>({
      ...(name ? { name: { $like: name } } : {}),
      ...(qualifiedDefaultModelEntity
        ? { default_model_entity: { $eq: qualifiedDefaultModelEntity } }
        : {}),
      ...(createdAt ? { created_at: createdAt } : {}),
      ...(guardrailConfig ? { guardrail_config: { $eq: guardrailConfig } } : {}),
    });
  }, [
    dataViewState.debouncedSearchBar,
    dataViewState.debouncedColumnFilters,
    guardrailConfig,
    workspace,
  ]);

  const { data, isFetching, error } = useListVirtualModels(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: sortParam,
      filter,
      // Autoprovisioned VMs are controller-managed passthroughs; hide them from the UI.
      exclude_autoprovisioned: true,
    },
    {
      query: {
        placeholderData: keepPreviousData,
      },
    }
  );

  const deleteVirtualModelMutation = useDeleteVirtualModel({
    mutation: {
      onSuccess: () => {
        toast.success('Virtual model deleted successfully.');
        queryClient.invalidateQueries({
          queryKey: getListVirtualModelsQueryKey(workspace),
        });
      },
    },
  });

  const virtualModels = useMemo(() => data?.data ?? [], [data?.data]);
  const pagination = data?.pagination;

  const virtualModelsWithId = useMemo<VirtualModelWithId[]>(
    () =>
      virtualModels.map((vm: VirtualModel) => ({
        ...vm,
        id: `${vm.workspace}/${vm.name}`,
      })),
    [virtualModels]
  );

  const handleDeleteVirtualModel = async () => {
    if (!modalVirtualModel?.name) return false;
    try {
      await deleteVirtualModelMutation.mutateAsync({
        workspace,
        name: modalVirtualModel.name,
      });
      return true;
    } catch {
      toast.error('Failed to delete virtual model');
      return false;
    }
  };

  const handleModalClose = () => {
    setModalVirtualModel(undefined);
  };

  const makeColumns: ComponentProps<typeof StudioDataView<VirtualModelWithId>>['makeColumns'] =
    useCallback(
      ({ accessor }, { rowActionsColumn }) => [
        accessor('name', {
          header: 'Name',
          enableSorting: true,
          size: 200,
          cell({ row }) {
            return <Text>{row.original.name}</Text>;
          },
        }),
        accessor('default_model_entity', {
          header: 'Default model',
          enableSorting: false,
          meta: {
            filter: {
              type: 'custom',
              label: 'Default Model',
              renderFilter: ({ setValue, value }) => (
                <BaseModelSearchFilterField
                  workspace={workspace}
                  value={value as string | undefined}
                  onValueChange={(models: string[]) => setValue(models[0] || undefined)}
                  singleSelect
                />
              ),
            },
          },
          cell({ row }) {
            const value = row.original.default_model_entity;
            return (
              <Text className="truncate max-w-[280px]" title={value}>
                {value || '—'}
              </Text>
            );
          },
        }),
        accessor((vm) => middlewareCount(vm), {
          id: 'middleware',
          header: 'Middleware',
          enableSorting: false,
          size: 120,
          cell({ row }) {
            const count = middlewareCount(row.original);
            return <Text>{count === 0 ? 'None' : `${count} call${count === 1 ? '' : 's'}`}</Text>;
          },
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 150,
          meta: {
            filter: dateTimeFilter('Created At'),
          },
          cell({ row }) {
            return row.original.created_at ? (
              <RelativeTime datetime={row.original.created_at} />
            ) : (
              <Text>—</Text>
            );
          },
        }),
        rowActionsColumn({
          size: ROW_ACTIONS_COLUMN_SIZE,
          enableResizing: false,
          cellProps: {
            attributes: {
              DropdownContent: { className: 'min-w-[156px]' },
            },
          },
          rowActions: (vm: VirtualModelWithId): DropdownEntry[] => [
            {
              children: 'View',
              onSelect: () => openVirtualModel(vm, 'details'),
            },
            {
              children: 'Chat',
              onSelect: () => openVirtualModel(vm, 'chat'),
            },
            {
              children: 'Delete',
              danger: true,
              onSelect: () => setModalVirtualModel(vm),
            },
          ],
        }),
      ],
      [openVirtualModel, workspace]
    );

  return (
    <Stack gap="density-xl" {...attributes?.Stack}>
      <StudioDataView
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={(row: VirtualModelWithId, _index, event) =>
          openVirtualModel(row, 'details', event)
        }
        attributes={{
          DataViewSearchBar: {
            placeholder: 'Search by name...',
          },
          DataViewRoot: {
            data: virtualModelsWithId,
            totalCount: pagination?.total_results,
            requestStatus: error ? 'error' : isFetching ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
              hasFiltersApplied || hasSearchApplied ? (
                <EntityEmptyState
                  entity="virtualModels"
                  variant="no-results"
                  onClearFilters={dataViewState.resetFilters}
                />
              ) : (
                <EntityEmptyState entity="virtualModels" variant="first-use" />
              ),
            renderErrorState: () => (
              <ErrorPanel
                errorMessage={getErrorMessage(error ?? new Error('Failed to fetch virtual models'))}
              />
            ),
          },
        }}
      />

      {modalVirtualModel && (
        <DeleteConfirmationModal
          open
          simpleConfirm
          onDelete={handleDeleteVirtualModel}
          title={`Delete virtual model: ${modalVirtualModel.name}`}
          confirmationText={modalVirtualModel.name}
          onClose={handleModalClose}
          description="Deleting this virtual model removes its inference route and middleware pipelines. Are you sure you want to proceed?"
        />
      )}
    </Stack>
  );
};
