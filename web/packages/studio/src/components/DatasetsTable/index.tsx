// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { Button } from '@nvidia/foundations-react-core';
import { DatasetCreateModal } from '@studio/components/DatasetCreateModal';
import { DatasetCreateModalMode } from '@studio/components/DatasetCreateModal/constants';
import { makeDatasetsTableColumns } from '@studio/components/DatasetsTable/columns';
import { type DatasetsTableProps } from '@studio/components/DatasetsTable/types';
import { useDatasetsTable } from '@studio/components/DatasetsTable/useDatasetsTable';
import { FilesetCreateModal } from '@studio/components/FilesetCreateModal';
import { Loading } from '@studio/components/Layouts/Loading';
import { FILESET_DETAILS_ENABLED } from '@studio/constants/environment';
import { DatasetBulkDeleteModal } from '@studio/routes/FilesetListRoute/DatasetBulkDeleteModal';
import { getNewFilesetRoute } from '@studio/routes/utils';
import { useBoolean } from '@studio/util/hooks/useBoolean';
import { Trash } from 'lucide-react';
import { type FC } from 'react';
import { useNavigate } from 'react-router';

export type { DatasetsTableProps } from '@studio/components/DatasetsTable/types';

/**
 * A table that displays a list of datasets with optional filtering, search, and bulk operations.
 */
export const DatasetsTable: FC<DatasetsTableProps> = ({
  onDatasetsSelected,
  onRowClick,
  enableActions = true,
  enableBulkDelete,
  enableFilters,
  enableSelection,
  selectionType,
  getDatasetRoute,
  renderRowActions,
  purposeFilter,
  attributes,
}) => {
  const {
    workspace,
    dataViewState,
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
  } = useDatasetsTable({
    onDatasetsSelected,
    onRowClick,
    enableFilters,
    enableSelection,
    selectionType,
    getDatasetRoute,
    purposeFilter,
  });

  const navigate = useNavigate();
  const [createModalOpen, openCreateModal, closeCreateModal] = useBoolean(false);
  const handleCreateFileset = () => {
    if (FILESET_DETAILS_ENABLED) {
      openCreateModal();
    } else {
      navigate(getNewFilesetRoute(workspace));
    }
  };

  // Column definitions
  const makeColumns = makeDatasetsTableColumns({
    enableSelection,
    selectionType,
    enableFilters,
    enableActions,
    getDatasetRoute,
    renderRowActions,
    setModalDataset,
    setModalOpen,
    handleDatasetDeleted,
  });

  // Loading state
  if (isPending) {
    return <Loading description="Loading filesets..." />;
  }

  // Error state
  if (error) {
    return (
      <ErrorMessage
        message="Failed to fetch filesets"
        slotFooter={
          <Button type="button" kind="tertiary" onClick={() => refetch()}>
            Retry
          </Button>
        }
      />
    );
  }

  // Table content
  const tableContent = (
    <>
      <StudioDataView
        dataViewState={dataViewState}
        searchField={enableFilters ? 'name' : undefined}
        makeColumns={makeColumns}
        onRowClick={handleRowClick}
        renderBulkActions={
          enableBulkDelete
            ? ({ selectedRows }) => (
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
              )
            : undefined
        }
        attributes={{
          ...attributes,
          DataViewSearchBar: {
            placeholder: 'Search filesets...',
          },
          DataViewRoot: {
            data: datasets,
            totalCount: datasetsResponse?.pagination?.total_results,
            requestStatus: isFetching ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
              hasFiltersApplied || hasSearchApplied ? (
                <EntityEmptyState
                  entity="filesets"
                  variant="no-results"
                  onClearFilters={resetFilters}
                />
              ) : (
                <EntityEmptyState
                  entity="filesets"
                  variant="first-use"
                  onCreate={handleCreateFileset}
                />
              ),
          },
        }}
      />

      {modalOpen === 'delete' && modalDataset && (
        <DeleteConfirmationModal
          open
          simpleConfirm
          onDelete={handleDeleteDataset}
          title={`Delete Dataset: ${modalDataset.name}`}
          confirmationText={modalDataset.name ?? getEntityReference(modalDataset)}
          onClose={handleModalClose}
        />
      )}

      {modalOpen === 'edit' && modalDataset && (
        <DatasetCreateModal
          dataset={modalDataset}
          mode={DatasetCreateModalMode.Edit}
          onClose={handleModalClose}
          open={modalOpen === 'edit'}
        />
      )}

      {createModalOpen && (
        <FilesetCreateModal
          open={createModalOpen}
          onClose={closeCreateModal}
          workspace={workspace}
          purpose={FilesetPurpose.dataset}
        />
      )}
    </>
  );

  return tableContent;
};
