// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { Trash } from '@nemo/icons/icons';
import { type FilesetOutput as Dataset, FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { Button } from '@nvidia/foundations-react-core';
import { DatasetCreateModal } from '@studio/components/DatasetCreateModal';
import { makeDatasetsTableColumns } from '@studio/components/DatasetsTable/columns';
import { FilesetCreateModal } from '@studio/components/FilesetCreateModal';
import { useFilesetsTable } from '@studio/components/FilesetsTable/useFilesetsTable';
import { Loading } from '@studio/components/Layouts/Loading';
import { FILESET_DETAILS_ENABLED } from '@studio/constants/environment';
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
    workspace,
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
  const [createModalOpen, openCreateModal, closeCreateModal] = useBoolean(false);

  const makeColumns = makeDatasetsTableColumns({
    renderRowActions:
      renderRowActions &&
      ((dataset) => renderRowActions(dataset, { onDatasetDeleted: handleDatasetDeleted })),
  });

  if (isPending) {
    return <Loading description="Loading datasets..." />;
  }

  if (error) {
    return (
      <ErrorMessage
        message="Failed to fetch datasets"
        slotFooter={
          <Button type="button" kind="tertiary" onClick={() => refetch()}>
            Retry
          </Button>
        }
      />
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
                  onCreate={openCreateModal}
                />
              ),
          },
        }}
      />

      {/* Same create flow as NewDatasetButton in the page header. */}
      {createModalOpen &&
        (FILESET_DETAILS_ENABLED ? (
          <FilesetCreateModal
            open={createModalOpen}
            onClose={closeCreateModal}
            workspace={workspace}
            purpose={FilesetPurpose.dataset}
          />
        ) : (
          <DatasetCreateModal
            open={createModalOpen}
            onClose={closeCreateModal}
            onDatasetCreated={closeCreateModal}
          />
        ))}
    </>
  );
};
