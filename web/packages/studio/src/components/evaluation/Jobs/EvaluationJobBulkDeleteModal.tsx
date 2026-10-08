// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { logger } from '@nemo/common/src/utils/logger';
import { useEvalsDeleteEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import { Button, Flex, Modal } from '@nvidia/foundations-react-core';
import { useMutateMany } from '@studio/api/common/useMutateMany';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { Trash } from 'lucide-react';
import { FC, useState } from 'react';

interface EvaluationJobBulkDeleteModalProps {
  selectedJobs: { name?: string }[];
  onConfirmSuccess: () => void;
}

export const EvaluationJobBulkDeleteModal: FC<EvaluationJobBulkDeleteModalProps> = ({
  selectedJobs,
  onConfirmSuccess,
}) => {
  const [open, setOpen] = useState<boolean>(false);
  const workspace = useWorkspaceFromPath();

  const { mutateAsync: deleteJob } = useEvalsDeleteEvaluateJob();
  const { mutateAsync: deleteJobs, isPending } = useMutateMany(deleteJob);
  const toast = useToast();

  const handleDelete = async () => {
    try {
      const namesToDelete = selectedJobs
        .map((job) => job.name)
        .filter((name): name is string => !!name);

      await deleteJobs(namesToDelete.map((name) => ({ workspace, name })));

      const count = namesToDelete.length;
      toast.success(`Successfully deleted ${count} evaluation ${count === 1 ? 'job' : 'jobs'}`);
      onConfirmSuccess();
      setOpen(false);
    } catch (error) {
      logger.error('Failed to delete evaluation jobs', error);
      toast.error('Failed to delete evaluation jobs. Please try again.');
    }
  };

  const handleOpenChange = (shouldOpen: boolean) => {
    if (!isPending) {
      setOpen(shouldOpen);
    }
  };

  return (
    <Modal
      open={open}
      onOpenChange={handleOpenChange}
      slotTrigger={
        <Button kind="secondary" data-testid="bulk-delete-modal-trigger-button">
          <Trash />
          Delete
        </Button>
      }
      slotHeading={
        <Flex align="center" gap="density-sm">
          <Trash />
          Delete {selectedJobs.length} Job{selectedJobs.length > 1 ? 's' : ''}
        </Flex>
      }
      slotFooter={
        <Flex justify="end" gap="density-xs" align="center" className="w-full">
          <Button
            onClick={() => setOpen(false)}
            kind="tertiary"
            color="neutral"
            disabled={isPending}
          >
            Cancel
          </Button>
          <Button color="brand" onClick={handleDelete} disabled={isPending}>
            {isPending ? 'Deleting...' : 'Delete'}
          </Button>
        </Flex>
      }
    >
      Are you sure you want to delete this?
    </Modal>
  );
};
