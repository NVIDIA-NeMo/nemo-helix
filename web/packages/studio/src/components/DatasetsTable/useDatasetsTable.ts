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
  checkedPurposes,
  DEFAULT_PURPOSE_FILTER,
  isDefaultPurposeFilter,
  VISIBLE_PURPOSES,
} from '@studio/components/DatasetsTable/constants';
import {
  type DatasetWithId,
  type DatasetsTableProps,
  type ModalOpenState,
} from '@studio/components/DatasetsTable/types';
import { useOwnedFirstFilesets } from '@studio/components/DatasetsTable/useOwnedFirstFilesets';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useAuthProfile } from '@studio/providers/auth/useAuthProfile';
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

  // Seed the default purposes as checked (generic stays off) once on load, alongside any
  // filters already in the URL. A later deliberate "uncheck all" is not re-seeded.
  const seededDefaults = useRef(false);
  useEffect(() => {
    if (!enableFilters || seededDefaults.current) return;
    seededDefaults.current = true;
    const current = dataViewState.columnFiltering.state;
    if (!current.some((f) => f.id === 'purpose')) {
      dataViewState.columnFiltering.set([
        ...current,
        { id: 'purpose', value: DEFAULT_PURPOSE_FILTER },
      ]);
    }
  }, [enableFilters, dataViewState.columnFiltering]);

  // The seeded default doesn't count as the user filtering, so empty states stay "no filesets yet".
  const hasActiveFilters = dataViewState.debouncedColumnFilters.some(
    (f) => !(f.id === 'purpose' && isDefaultPurposeFilter(f.value))
  );
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

  // Reset filters and selections; the default purposes come back checked.
  const resetFilters = useCallback(() => {
    onDatasetsSelected?.([]);
    dataViewState.resetFilters();
    if (enableFilters) {
      dataViewState.columnFiltering.set([{ id: 'purpose', value: DEFAULT_PURPOSE_FILTER }]);
    }
  }, [dataViewState, enableFilters, onDatasetsSelected]);

  const page = dataViewState.pagination.state.pageIndex + 1;
  const pageSize = dataViewState.pagination.state.pageSize;
  const sort = enableFilters
    ? (getSortParam(dataViewState.sorting.state) as GenericSortField)
    : undefined;
  // The Purpose column is a multi-select, so its filter value is `{ [purpose]: true }`.
  const selectedPurposes = checkedPurposes(
    enableFilters ? dataViewState.apiFilter.filter?.purpose : undefined
  );
  const filter: FilesetFilter = {
    ...(enableFilters ? dataViewState.apiFilter.filter : undefined),
    ...(enableFilters && dataViewState.apiFilter.searchText
      ? withOperators<FilesetFilter>({
          name: { $like: `%${dataViewState.apiFilter.searchText}%` },
        })
      : {}),
    // Management view only: nothing checked means every purpose except "generic". Pickers
    // (filters off) list everything unless `purposeFilter` narrows them.
    ...(enableFilters
      ? withOperators<FilesetFilter>({
          purpose: { $in: selectedPurposes.length > 0 ? selectedPurposes : VISIBLE_PURPOSES },
        })
      : {}),
    ...(purposeFilter !== undefined ? { purpose: purposeFilter } : {}),
  };

  // On the management view, a signed-in user's own filesets come first.
  const ownerId = useAuthProfile()?.id;
  const ownedFirst = useOwnedFirstFilesets({
    workspace,
    ownerId: enableFilters ? ownerId : undefined,
    page,
    pageSize,
    sort,
    filter,
  });
  const plain = useFilesListFilesets(
    workspace,
    { page, page_size: pageSize, sort, filter },
    { query: { enabled: !(enableFilters && ownerId), placeholderData: keepPreviousData } }
  );
  const {
    data: datasetsResponse,
    refetch,
    isPending,
    isPlaceholderData,
    error,
  } = enableFilters && ownerId ? ownedFirst : plain;

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
    isPlaceholderData,
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
