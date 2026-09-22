// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { useFilesDeleteFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Button } from '@nvidia/foundations-react-core';
import { useMutateMany } from '@studio/api/common/useMutateMany';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { BulkDeleteModal as GenericBulkDeleteModal } from '@studio/components/BulkDeleteModal';
import { Trash } from 'lucide-react';
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
  /** Custom trigger element; when provided, used instead of the default Button */
  slotTrigger?: ReactNode;
}

export const DatasetBulkDeleteModal: FC<DatasetBulkDeleteModalProps> = ({
  selectedDatasets,
  onConfirmSuccess,
  slotTrigger,
}) => {
  const [open, setOpen] = useState(false);

  const { mutateAsync: deleteDataset } = useFilesDeleteFileset({
    mutation: {
      onSuccess: (_data, variables) => {
        invalidateDatasetCaches(variables.workspace, variables.name, ['list']);
      },
    },
  });
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

  const handleDelete = async (datasets: FilesetOutput[]) => {
    const datasetsToDelete = datasets.filter(
      (dataset): dataset is FilesetOutput & { workspace: string; name: string } =>
        !!(dataset.workspace && dataset.name)
    );
    if (datasetsToDelete.length !== datasets.length) {
      throw new Error('Cannot delete datasets without workspace and name.');
    }
    await deleteDatasets(
      datasetsToDelete.map((dataset) => ({ workspace: dataset.workspace, name: dataset.name }))
    );
    onConfirmSuccess();
  };

  const openTrigger = () => setOpen(true);

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
      <GenericBulkDeleteModal
        items={selectedDatasets}
        open={open}
        onDelete={handleDelete}
        title={(count) => `Delete ${count} Dataset${count !== 1 ? 's' : ''}`}
        onClose={() => setOpen(false)}
      />
    </>
  );
};
