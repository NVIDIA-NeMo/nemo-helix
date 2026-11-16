// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { logger } from '@nemo/common/src/utils/logger';
import { useFilesDeleteFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { DatasetCreateModal } from '@studio/components/DatasetCreateModal';
import { DatasetCreateModalMode } from '@studio/components/DatasetCreateModal/constants';
import { getDeleteFilesetsDescription } from '@studio/constants/filesets';
import { type FC, useState } from 'react';

export interface FilesetActionMenuProps {
  fileset: FilesetOutput;
  onFilesetUpdated?: (fileset: FilesetOutput) => void;
  onFilesetDeleted?: (fileset: FilesetOutput) => void;
}

export const FilesetActionMenu: FC<FilesetActionMenuProps> = ({
  fileset,
  onFilesetUpdated,
  onFilesetDeleted,
}) => {
  const [modalOpen, setModalOpen] = useState<'edit' | 'delete' | undefined>(undefined);
  const { mutateAsync: deleteFileset } = useFilesDeleteFileset({
    mutation: {
      onSuccess: (_data, variables) => {
        invalidateDatasetCaches(variables.workspace, variables.name, ['list']);
      },
    },
  });

  const handleDeleteFileset = async (): Promise<boolean> => {
    try {
      if (!fileset?.workspace || !fileset?.name) {
        throw new Error('Fileset workspace or name is undefined');
      }
      await deleteFileset({ workspace: fileset.workspace, name: fileset.name });
      onFilesetDeleted?.(fileset);
      return true;
    } catch (error) {
      logger.error('Failed to delete fileset', error);
      return false;
    }
  };

  const handleModalClose = () => {
    setModalOpen(undefined);
  };

  const handleFilesetUpdated = (updatedFileset: FilesetOutput) => {
    onFilesetUpdated?.(updatedFileset);
    handleModalClose();
  };

  return (
    <>
      <QuickActionsMenuRoot
        ariaLabel="Open fileset actions menu"
        actions={[{ label: 'Edit', onSelect: () => setModalOpen('edit') }]}
        deleteAction={{ onSelect: () => setModalOpen('delete') }}
      />

      {modalOpen === 'edit' && (
        <DatasetCreateModal
          dataset={fileset}
          mode={DatasetCreateModalMode.Edit}
          onClose={handleModalClose}
          onDatasetUpdated={handleFilesetUpdated}
          open={modalOpen === 'edit'}
        />
      )}

      {modalOpen === 'delete' && (
        <DeleteConfirmationModal
          open={modalOpen === 'delete'}
          onClose={handleModalClose}
          onDelete={handleDeleteFileset}
          entity="fileset"
          ids={[fileset.name]}
          description={getDeleteFilesetsDescription('fileset', 1)}
        />
      )}
    </>
  );
};
