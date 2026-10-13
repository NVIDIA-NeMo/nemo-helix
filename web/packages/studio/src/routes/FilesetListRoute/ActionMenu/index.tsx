// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { logger } from '@nemo/common/src/utils/logger';
import { useFilesDeleteFileset } from '@nemo/sdk/generated/platform/files';
import { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import {
  Button,
  DropdownContent,
  DropdownItem,
  DropdownRoot,
  DropdownTrigger,
} from '@nvidia/foundations-react-core';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { DatasetCreateModal } from '@studio/components/DatasetCreateModal';
import { DatasetCreateModalMode } from '@studio/components/DatasetCreateModal/constants';
import { EllipsisVertical } from 'lucide-react';
import { FC, useState } from 'react';

interface ActionMenuProps {
  dataset: FilesetOutput;
  onNavigateToDetails: (dataset: FilesetOutput) => void;
  onDatasetUpdated?: (dataset: FilesetOutput) => void;
  onDatasetDeleted?: (dataset: FilesetOutput) => void;
}

export const ActionMenu: FC<ActionMenuProps> = ({
  dataset,
  onNavigateToDetails,
  onDatasetUpdated,
  onDatasetDeleted,
}) => {
  const [modalOpen, setModalOpen] = useState<'edit' | 'delete' | undefined>(undefined);
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
      <DropdownRoot>
        <DropdownTrigger asChild showChevron={false} data-testid="quick-actions-menu-trigger">
          <Button kind="tertiary" aria-label="Open dataset actions menu">
            <EllipsisVertical />
          </Button>
        </DropdownTrigger>
        <DropdownContent align="end" className="w-[180px]">
          <DropdownItem onClick={() => onNavigateToDetails(dataset)}>View</DropdownItem>
          <DropdownItem onClick={() => setModalOpen('edit')}>Edit</DropdownItem>
          <DropdownItem onClick={() => setModalOpen('delete')} danger>
            Delete
          </DropdownItem>
        </DropdownContent>
      </DropdownRoot>

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

      {/* Delete Modal */}
      {modalOpen === 'delete' && (
        <DeleteConfirmationModal
          open={modalOpen === 'delete'}
          onClose={handleModalClose}
          onDelete={handleDeleteDataset}
          title={`Delete Dataset: ${dataset.name}`}
          confirmationText={dataset.name ?? getEntityReference(dataset)}
          simpleConfirm
          successText="Dataset deleted successfully"
        />
      )}
    </>
  );
};
