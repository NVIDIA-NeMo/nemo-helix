// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { Trash } from '@nemo/icons/icons';
import { useFilesDeleteFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Button } from '@nvidia/foundations-react-core';
import { useMutateMany } from '@studio/api/common/useMutateMany';
import { getDeleteFilesetsDescription } from '@studio/constants/filesets';
import {
  cloneElement,
  isValidElement,
  type FC,
  type MouseEventHandler,
  type ReactNode,
  useState,
} from 'react';

interface TriggerProps {
  onClick?: MouseEventHandler;
}

interface DatasetBulkDeleteModalProps {
  selectedDatasets: FilesetOutput[];
  onConfirmSuccess: () => void;
  onSettled: () => void;
  /** Custom trigger element; when provided, used instead of the default Button */
  slotTrigger?: ReactNode;
  /** What the selected rows are called in the title (default: Dataset) */
  entityLabel?: string;
}

export const DatasetBulkDeleteModal: FC<DatasetBulkDeleteModalProps> = ({
  selectedDatasets,
  onConfirmSuccess,
  onSettled,
  slotTrigger,
  entityLabel = 'Dataset',
}) => {
  const [datasets, setDatasets] = useState<FilesetOutput[]>([]);
  const entity = entityLabel.toLowerCase();

  const { mutateAsync: deleteDataset } = useFilesDeleteFileset();
  const deleteDatasetWithMessage = async (variables: { workspace: string; name: string }) => {
    try {
      return await deleteDataset(variables);
    } catch (error) {
      throw new Error(
        getErrorMessage(
          error as Error,
          `Fileset '${variables.workspace}/${variables.name}' could not be deleted. It may still be in use.`
        )
      );
    }
  };
  const { mutateAsync: deleteDatasets } = useMutateMany(deleteDatasetWithMessage, {
    action: 'delete',
  });

  const handleDelete = async () => {
    const datasetsToDelete = datasets.filter(
      (dataset): dataset is FilesetOutput & { workspace: string; name: string } =>
        !!(dataset.workspace && dataset.name)
    );
    if (datasetsToDelete.length !== datasets.length) {
      throw new Error('Cannot delete datasets without workspace and name.');
    }
    try {
      await deleteDatasets(
        datasetsToDelete.map((dataset) => ({ workspace: dataset.workspace, name: dataset.name }))
      );
      onConfirmSuccess();
      return true;
    } finally {
      onSettled();
    }
  };

  const openTrigger = () => setDatasets(selectedDatasets);

  const trigger = isValidElement<TriggerProps>(slotTrigger) ? (
    cloneElement(slotTrigger, {
      onClick: (e: Parameters<MouseEventHandler>[0]) => {
        slotTrigger.props.onClick?.(e);
        openTrigger();
      },
    })
  ) : (
    <Button kind="secondary" data-testid="bulk-delete-modal-trigger-button" onClick={openTrigger}>
      <Trash />
      Delete
    </Button>
  );

  return (
    <>
      {trigger}
      {datasets.length > 0 && (
        <DeleteConfirmationModal
          open
          entity={entity}
          ids={datasets.map((dataset) => dataset.name)}
          description={getDeleteFilesetsDescription(entity, datasets.length)}
          onDelete={handleDelete}
          onClose={() => setDatasets([])}
        />
      )}
    </>
  );
};
