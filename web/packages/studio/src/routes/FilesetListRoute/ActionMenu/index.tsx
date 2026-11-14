// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { logger } from '@nemo/common/src/utils/logger';
import { useFilesDeleteFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { DatasetCreateModal } from '@studio/components/DatasetCreateModal';
import { DatasetCreateModalMode } from '@studio/components/DatasetCreateModal/constants';
import { CopyFilesetModal } from '@studio/components/filesets/CopyFilesetModal';
import { getDeleteFilesetsDescription } from '@studio/constants/filesets';
import { type FC, useState } from 'react';

interface ActionMenuProps {
  dataset: FilesetOutput;
  onNavigateToDetails: (dataset: FilesetOutput) => void;
  onDatasetUpdated?: (dataset: FilesetOutput) => void;
  onDatasetDeleted?: (dataset: FilesetOutput) => void;
  /** What the row is called in the menu label and delete copy (default: Dataset) */
  entityLabel?: string;
}

export const ActionMenu: FC<ActionMenuProps> = ({
  dataset,
  onNavigateToDetails,
  onDatasetUpdated,
  onDatasetDeleted,
  entityLabel = 'Dataset',
}) => {
  const [modalOpen, setModalOpen] = useState<'edit' | 'copy' | 'delete' | undefined>(undefined);
  const { mutateAsync: deleteDataset } = useFilesDeleteFileset({
    mutation: {
      onSuccess: (_data, variables) => {
        invalidateDatasetCaches(variables.workspace, variables.name, ['list']);
      },
    },
  });

  const handleDeleteDataset = async (): Promise<boolean> => {
    try {
      if (!dataset?.workspace || !dataset?.name) {
        throw new Error('Dataset workspace or name is undefined');
      }
      await deleteDataset({ workspace: dataset.workspace, name: dataset.name });
      onDatasetDeleted?.(dataset);
      return true;
    } catch (error) {
      logger.error('Failed to delete dataset', error);
      throw new Error(
        getErrorMessage(
          error as Error,
          'This fileset could not be deleted. It may still be in use.'
        )
      );
    }
  };

  const handleModalClose = () => {
    setModalOpen(undefined);
  };

  const handleDatasetUpdated = (updatedDataset: FilesetOutput) => {
    onDatasetUpdated?.(updatedDataset);
    handleModalClose();
  };

  return (
    <>
      <QuickActionsMenuRoot
        ariaLabel={`Open ${entityLabel.toLowerCase()} actions menu`}
        actions={[
          { label: 'View', onSelect: () => onNavigateToDetails(dataset) },
          { label: 'Edit', onSelect: () => setModalOpen('edit') },
          { label: 'Copy to New Fileset', onSelect: () => setModalOpen('copy') },
        ]}
        deleteAction={{ onSelect: () => setModalOpen('delete') }}
      />

      {/* Edit Modal */}
      {modalOpen === 'edit' && (
        <DatasetCreateModal
          dataset={dataset}
          mode={DatasetCreateModalMode.Edit}
          onClose={handleModalClose}
          onDatasetUpdated={handleDatasetUpdated}
          open={modalOpen === 'edit'}
        />
      )}

      {modalOpen === 'copy' && (
        <CopyFilesetModal
          source={dataset}
          onClose={handleModalClose}
          onCopied={onNavigateToDetails}
        />
      )}

      {/* Delete Modal */}
      {modalOpen === 'delete' && (
        <DeleteConfirmationModal
          open={modalOpen === 'delete'}
          onClose={handleModalClose}
          onDelete={handleDeleteDataset}
          entity={entityLabel.toLowerCase()}
          ids={[dataset.name]}
          description={getDeleteFilesetsDescription(entityLabel.toLowerCase(), 1)}
        />
      )}
    </>
  );
};
