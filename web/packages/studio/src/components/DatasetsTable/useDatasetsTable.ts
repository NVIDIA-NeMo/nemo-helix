// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParam } from '@nemo/common/src/utils/query';
import { useFilesDeleteFileset, useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import {
  type FilesetFilter,
  type FilesetOutput as Dataset,
  type GenericSortField,
} from '@nemo/sdk/generated/platform/schema';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import {
  type DatasetWithId,
  type DatasetsTableProps,
  type ModalOpenState,
} from '@studio/components/DatasetsTable/types';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { keepPreviousData } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router';

type UseDatasetsTableArgs = Pick<
  DatasetsTableProps,
  | 'onDatasetsSelected'
  | 'onRowClick'
  | 'enableFilters'
  | 'enableSelection'
  | 'selectionType'
  | 'getDatasetRoute'
  | 'purposeFilter'
>;

export function useDatasetsTable({
  onDatasetsSelected,
  onRowClick,
  enableFilters,
  enableSelection,
  selectionType,
  getDatasetRoute,
  purposeFilter,
}: UseDatasetsTableArgs) {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();

  // DataView state for pagination, row selection, sorting, search, and filters
  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const hasActiveFilters = dataViewState.debouncedColumnFilters.length > 0;
  const hasSearchOrFilters = !!(dataViewState.debouncedSearchBar || hasActiveFilters);

  const [modalDataset, setModalDataset] = useState<Dataset>();
  const [modalOpen, setModalOpen] = useState<ModalOpenState>();
  const { mutateAsync: deleteDataset } = useFilesDeleteFileset({
    mutation: {
      onSuccess: (_data, variables) => {
        invalidateDatasetCaches(variables.workspace, variables.name, ['list']);
      },
    },
  });

  // Reset filters and selections
  const resetFilters = useCallback(() => {
    onDatasetsSelected?.([]);
    dataViewState.resetFilters();
  }, [dataViewState, onDatasetsSelected]);

  const {
    data: datasetsResponse,
    refetch,
    isPending,
    isFetching,
    error,
  } = useFilesListFilesets(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: enableFilters
        ? (getSortParam(dataViewState.sorting.state) as GenericSortField)
        : undefined,
      filter: {
        ...(enableFilters ? dataViewState.apiFilter.filter : undefined),
        ...(enableFilters && dataViewState.apiFilter.searchText
          ? withOperators<FilesetFilter>({
              name: { $like: `%${dataViewState.apiFilter.searchText}%` },
            })
          : {}),
        ...(purposeFilter !== undefined ? { purpose: purposeFilter } : {}),
      },
    },
    {
      query: {
        placeholderData: keepPreviousData,
      },
    }
  );

  // Ensure each dataset has a unique id for DataView row selection
  const datasets = useMemo<DatasetWithId[]>(
    () =>
      (datasetsResponse?.data || []).map((dataset) => ({
        ...dataset,
        id: dataset.id || `${dataset.workspace}/${dataset.name}`,
      })),
    [datasetsResponse?.data]
  );

  // Propagate row selection changes to onDatasetsSelected callback
  const prevSelectionRef = useRef(dataViewState.rowSelection.state);
  useEffect(() => {
    const selection = dataViewState.rowSelection.state;
    if (selection === prevSelectionRef.current) return;
    prevSelectionRef.current = selection;

    // For single selection, keep only the most recently selected row
    const selectedIds = Object.keys(selection).filter((id) => selection[id]);
    if (selectionType === 'single' && selectedIds.length > 1) {
      const lastSelected = selectedIds[selectedIds.length - 1];
      dataViewState.rowSelection.set({ [lastSelected]: true });
      return; // The set above will re-trigger this effect with the corrected state
    }

    if (!onDatasetsSelected) return;

    const selectedDatasets = datasets.filter((d) => selection[d.id]);
    onDatasetsSelected(selectedDatasets);
  }, [
    dataViewState.rowSelection.state,
    datasets,
    onDatasetsSelected,
    selectionType,
    dataViewState.rowSelection,
  ]);

  // Row click handler
  const handleRowClick = useCallback(
    (dataset: DatasetWithId) => {
      if (onRowClick) {
        onRowClick(dataset);
      }
      if (getDatasetRoute) {
        navigate(getDatasetRoute(dataset));
      }
      if (enableSelection && !enableFilters) {
        // In simple mode, clicking row selects it
        dataViewState.rowSelection.set({ [dataset.id]: true });
      }
    },
    [
      onRowClick,
      getDatasetRoute,
      navigate,
      enableSelection,
      enableFilters,
      dataViewState.rowSelection,
    ]
  );

  // Action handlers
  const handleDatasetDeleted = useCallback(
    (deletedDataset: Dataset) => {
      const currentSelection = { ...dataViewState.rowSelection.state };
      delete currentSelection[deletedDataset.id || ''];
      dataViewState.rowSelection.set(currentSelection);
    },
    [dataViewState.rowSelection]
  );

  const handleDeleteDataset = async () => {
    try {
      if (!modalDataset?.workspace || !modalDataset?.name) return false;
      await deleteDataset({
        workspace: modalDataset.workspace,
        name: modalDataset.name,
      });
      handleDatasetDeleted(modalDataset);
      return true;
    } catch {
      return false;
    }
  };

  const handleBulkDeleteSuccess = useCallback(() => {
    onDatasetsSelected?.([]);
    dataViewState.rowSelection.set({});
  }, [dataViewState.rowSelection, onDatasetsSelected]);

  const handleBulkDeleteSettled = useCallback(() => {
    invalidateDatasetCaches(workspace, undefined, ['list']);
    refetch();
  }, [refetch, workspace]);

  const handleModalClose = () => setModalOpen('none');

  return {
    workspace,
    dataViewState,
    hasSearchOrFilters,
    modalDataset,
    setModalDataset,
    modalOpen,
    setModalOpen,
    datasetsResponse,
    datasets,
    refetch,
    isPending,
    isFetching,
    error,
    resetFilters,
    handleRowClick,
    handleDatasetDeleted,
    handleDeleteDataset,
    handleBulkDeleteSuccess,
    handleBulkDeleteSettled,
    handleModalClose,
  };
}
