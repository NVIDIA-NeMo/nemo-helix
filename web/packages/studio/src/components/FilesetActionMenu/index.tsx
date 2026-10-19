// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { logger } from '@nemo/common/src/utils/logger';
import { useFilesDeleteFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
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
import { type FC, useState } from 'react';

export interface FilesetActionMenuProps {
  fileset: FilesetOutput;
  onNavigateToDetails?: (fileset: FilesetOutput) => void;
  onFilesetUpdated?: (fileset: FilesetOutput) => void;
  onFilesetDeleted?: (fileset: FilesetOutput) => void;
}

export const FilesetActionMenu: FC<FilesetActionMenuProps> = ({
  fileset,
  onNavigateToDetails,
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
      <DropdownRoot>
        <DropdownTrigger asChild showChevron={false} data-testid="quick-actions-menu-trigger">
          <Button kind="tertiary" aria-label="Open fileset actions menu">
            <EllipsisVertical />
          </Button>
        </DropdownTrigger>
        <DropdownContent align="end" className="w-[180px]">
          {onNavigateToDetails && (
            <DropdownItem onClick={() => onNavigateToDetails(fileset)}>View</DropdownItem>
          )}
          <DropdownItem onClick={() => setModalOpen('edit')}>Edit</DropdownItem>
          <DropdownItem onClick={() => setModalOpen('delete')} danger>
            Delete
          </DropdownItem>
        </DropdownContent>
      </DropdownRoot>

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
          title={`Delete Fileset: ${fileset.name}`}
          confirmationText={fileset.name ?? getEntityReference(fileset)}
          simpleConfirm
          successText="Fileset deleted successfully"
        />
      )}
    </>
  );
};
