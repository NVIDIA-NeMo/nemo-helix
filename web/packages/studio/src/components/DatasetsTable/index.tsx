// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { Trash } from '@nemo/icons/icons';
import type { FilesetOutput as Dataset } from '@nemo/sdk/generated/platform/schema';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import { Button } from '@nvidia/foundations-react-core';
import { makeDatasetsTableColumns } from '@studio/components/DatasetsTable/columns';
import { useFilesetsTable } from '@studio/components/FilesetsTable/useFilesetsTable';
import { Loading } from '@studio/components/Layouts/Loading';
import { NewDatasetPanel } from '@studio/components/NewDatasetButton/NewDatasetPanel';
import { DatasetBulkDeleteModal } from '@studio/routes/FilesetListRoute/DatasetBulkDeleteModal';
import { useBoolean } from '@studio/util/hooks/useBoolean';
import type { FC, ReactNode } from 'react';

export interface DatasetsTableProps {
  /** Renders the dataset name as a link to the returned route */
  getDatasetRoute?: (dataset: Dataset) => string;
  /** Renders the row's action menu; `onDatasetDeleted` clears the row from the selection */
  renderRowActions?: (
    dataset: Dataset,
    callbacks: { onDatasetDeleted: (dataset: Dataset) => void }
  ) => ReactNode;
}

/**
 * Lists the workspace's dataset filesets with search, sorting, and bulk delete.
 */
export const DatasetsTable: FC<DatasetsTableProps> = ({ getDatasetRoute, renderRowActions }) => {
  const {
    hasSearchOrFilters,
    dataViewState,
    datasetsResponse,
    datasets,
    refetch,
    isPending,
    isFetching,
    error,
    resetFilters,
    handleRowClick,
    handleDatasetDeleted,
    handleBulkDeleteSuccess,
    handleBulkDeleteSettled,
  } = useFilesetsTable({
    enableFilters: true,
    enableSelection: true,
    getDatasetRoute,
    purposeFilter: FilesetPurpose.dataset,
  });
  const [createPanelOpen, openCreatePanel, closeCreatePanel] = useBoolean(false);

  const makeColumns = makeDatasetsTableColumns({
    renderRowActions:
      renderRowActions &&
      ((dataset) => renderRowActions(dataset, { onDatasetDeleted: handleDatasetDeleted })),
  });

  // Shares NewDatasetPanel with NewDatasetButton. Rendered beside every list state, at the same position, so it
  // stays mounted: a creation finishing after Cancel is still reported, and a list refetch failing while the
  // panel is open keeps its draft and upload retry state.
  const panel = <NewDatasetPanel open={createPanelOpen} onClose={closeCreatePanel} />;

  if (isPending) {
    return (
      <>
        <Loading description="Loading datasets..." />
        {panel}
      </>
    );
  }

  if (error) {
    return (
      <>
        <ErrorMessage
          message="Failed to fetch datasets"
          slotFooter={
            <Button type="button" kind="tertiary" onClick={() => refetch()}>
              Retry
            </Button>
          }
        />
        {panel}
      </>
    );
  }

  return (
    <>
      <StudioDataView
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={handleRowClick}
        renderBulkActions={({ selectedRows }) => (
          <DatasetBulkDeleteModal
            selectedDatasets={selectedRows}
            onConfirmSuccess={handleBulkDeleteSuccess}
            onSettled={handleBulkDeleteSettled}
            slotTrigger={
              <Button kind="tertiary" aria-label="Delete selected datasets">
                <Trash />
                Delete
              </Button>
            }
          />
        )}
        attributes={{
          DataViewSearchBar: {
            placeholder: 'Search datasets...',
          },
          DataViewRoot: {
            data: datasets,
            totalCount: datasetsResponse?.pagination?.total_results,
            requestStatus: isFetching ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: () =>
              hasSearchOrFilters ? (
                <EntityEmptyState
                  entity="datasets"
                  variant="no-results"
                  onClearFilters={resetFilters}
                />
              ) : (
                <EntityEmptyState
                  entity="datasets"
                  variant="first-use"
                  onCreate={openCreatePanel}
                />
              ),
          },
        }}
      />
      {panel}
    </>
  );
};
