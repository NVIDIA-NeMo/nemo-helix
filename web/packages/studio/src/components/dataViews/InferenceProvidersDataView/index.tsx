/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import { withOperators } from '@nemo/common/src/api/filterOperators';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useDeferredUnmount } from '@nemo/common/src/hooks/useDeferredUnmount';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getModelsListProvidersQueryKey,
  useModelsDeleteProvider,
  useModelsListProviders,
} from '@nemo/sdk/generated/platform/model-providers';
import {
  ModelProvider,
  ModelProviderFilter,
  ModelProviderSort,
} from '@nemo/sdk/generated/platform/schema';
import { type DropdownEntry, Stack, Text } from '@nvidia/foundations-react-core';
import { EditInferenceProviderModal } from '@studio/routes/InferenceProvidersListRoute/EditInferenceProviderModal';
import { InferenceProviderDetailsSidePanel } from '@studio/routes/InferenceProvidersListRoute/InferenceProviderDetailsSidePanel';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';

export interface InferenceProvidersDataViewProps {
  workspace: string;
  /** Opens the create-provider flow from the first-use empty state. */
  readonly onCreate?: () => void;
  attributes?: {
    Stack?: React.ComponentProps<typeof Stack>;
  };
}

type ProviderWithId = ModelProvider & { id: string };

type ModalState = 'delete' | 'edit' | 'none';

export const InferenceProvidersDataView: FC<InferenceProvidersDataViewProps> = ({
  workspace,
  onCreate,
  attributes,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();

  const {
    isOpen: isDetailsPanelOpen,
    value: providerForDetails,
    open: openDetailsPanel,
    close: closeDetailsPanel,
  } = useDeferredUnmount<ProviderWithId>({ delay: 300 });

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const [modalProvider, setModalProvider] = useState<ModelProvider>();
  const [modalOpen, setModalOpen] = useState<ModalState>('none');

  const sortState = dataViewState.sorting.state[0];
  const sortParam: ModelProviderSort | undefined = sortState
    ? ((sortState.desc ? `-${sortState.id}` : sortState.id) as ModelProviderSort)
    : '-created_at';

  const { data, isFetching, error } = useModelsListProviders(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: sortParam,
      filter: {
        ...dataViewState.apiFilter.filter,
        ...(dataViewState.apiFilter.searchText
          ? withOperators<ModelProviderFilter>({
              name: { $like: dataViewState.apiFilter.searchText },
            })
          : {}),
      },
    },
    {
      query: {
        placeholderData: keepPreviousData,
        // Provider status (READY/ERROR/...) is computed asynchronously by the Models
        // Controller's reconciler, so the badge can go stale between reconcile cycles.
        // Poll on a modest interval so the list reflects status changes without a manual
        // remount/refocus.
        refetchInterval: 20_000,
      },
    }
  );

  const deleteProviderMutation = useModelsDeleteProvider({
    mutation: {
      onSuccess: () => {
        toast.success('Inference provider deleted successfully.');
        queryClient.invalidateQueries({
          queryKey: getModelsListProvidersQueryKey(workspace),
        });
      },
    },
  });

  const providers = useMemo(() => data?.data ?? [], [data?.data]);
  const pagination = data?.pagination;

  const providersWithId = useMemo<ProviderWithId[]>(
    () =>
      providers.map((p: ModelProvider) => ({
        ...p,
        id: `${p.workspace}/${p.name}`,
      })),
    [providers]
  );

  const handleDeleteProvider = async () => {
    if (!modalProvider) return false;
    try {
      await deleteProviderMutation.mutateAsync({
        workspace,
        name: modalProvider.name,
      });
      return true;
    } catch {
      toast.error('Failed to delete inference provider');
      return false;
    }
  };

  const handleModalClose = () => {
    setModalOpen('none');
    setModalProvider(undefined);
  };

  const makeColumns: ComponentProps<typeof StudioDataView<ProviderWithId>>['makeColumns'] =
    useCallback(
      ({ accessor }, { rowActionsColumn }) => [
        accessor('name', {
          header: 'Name',
          enableSorting: false,
          size: 175,
          cell({ row }) {
            return <Text>{row.original.name}</Text>;
          },
        }),
        accessor('host_url', {
          header: 'Host URL',
          cell({ row }) {
            const url = row.original.host_url;
            return (
              <Text className="truncate max-w-[280px]" title={url}>
                {url || '-'}
              </Text>
            );
          },
        }),
        accessor('status', {
          header: 'Status',
          size: 100,
          cell({ row }) {
            return <StatusBadge status={row.original.status} />;
          },
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 150,
          cell({ row }) {
            return row.original.created_at ? (
              <RelativeTime datetime={row.original.created_at} />
            ) : (
              <Text>-</Text>
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
          rowActions: (provider: ProviderWithId): DropdownEntry[] => [
            {
              children: 'Edit',
              onSelect: () => {
                setModalProvider(provider);
                setModalOpen('edit');
              },
            },
            {
              children: 'Delete',
              danger: true,
              onSelect: () => {
                setModalProvider(provider);
                setModalOpen('delete');
              },
            },
          ],
        }),
      ],
      []
    );

  return (
    <Stack gap="density-xl" {...attributes?.Stack}>
      <StudioDataView
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={(row: ProviderWithId) => openDetailsPanel(row)}
        attributes={{
          DataViewSearchBar: {
            placeholder: 'Search Providers...',
          },
          DataViewRoot: {
            data: providersWithId,
            totalCount: pagination?.total_results,
            requestStatus: error ? 'error' : isFetching ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
              hasFiltersApplied || hasSearchApplied ? (
                <EntityEmptyState
                  entity="inferenceProviders"
                  variant="no-results"
                  onClearFilters={dataViewState.resetFilters}
                />
              ) : (
                <EntityEmptyState
                  entity="inferenceProviders"
                  variant="first-use"
                  onCreate={onCreate}
                />
              ),
            renderErrorState: () => (
              <ErrorPanel
                errorMessage={getErrorMessage(
                  error ?? new Error('Failed to fetch inference providers')
                )}
              />
            ),
          },
        }}
      />

      {modalOpen === 'delete' && modalProvider && (
        <DeleteConfirmationModal
          open
          simpleConfirm
          onDelete={handleDeleteProvider}
          title={`Delete inference provider: ${modalProvider.name}`}
          confirmationText={modalProvider.name}
          onClose={handleModalClose}
          description="Deleting will also remove any models associated with this provider. Are you sure you want to proceed?"
        />
      )}

      {modalOpen === 'edit' && modalProvider && (
        <EditInferenceProviderModal
          workspace={workspace}
          provider={modalProvider}
          open
          onClose={handleModalClose}
        />
      )}

      {providerForDetails != null && (
        <InferenceProviderDetailsSidePanel
          open={isDetailsPanelOpen}
          provider={providerForDetails}
          onClose={closeDetailsPanel}
        />
      )}
    </Stack>
  );
};
