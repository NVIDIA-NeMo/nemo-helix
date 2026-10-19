// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { logger } from '@nemo/common/src/utils/logger';
import { useEvalsDeleteEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import type { EvaluateJob } from '@nemo/sdk/generated/evals/schema';
import {
  Button,
  DropdownContent,
  DropdownItem,
  DropdownRoot,
  DropdownTrigger,
  Flex,
  Modal,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getEvaluationJobName } from '@studio/selectors/evaluationJob';
import { ArrowRight, EllipsisVertical, Trash } from 'lucide-react';
import { FC, useState } from 'react';

interface ActionMenuProps {
  job: EvaluateJob;
  onNavigateToDetails: (job: EvaluateJob) => void;
  onJobDeleted?: (job: EvaluateJob) => void;
}

export const ActionMenu: FC<ActionMenuProps> = ({ job, onNavigateToDetails, onJobDeleted }) => {
  const [modalOpen, setModalOpen] = useState<'edit' | 'delete' | undefined>(undefined);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);
  const workspace = useWorkspaceFromPath();
  const { mutateAsync: deleteEvaluateJob } = useEvalsDeleteEvaluateJob();
  const toast = useToast();
  const jobName = getEvaluationJobName(job);

  const handleDeleteJob = async (): Promise<void> => {
    setIsDeleting(true);
    try {
      if (!jobName) {
        throw new Error('Evaluation job name is undefined');
      }
      await deleteEvaluateJob({ workspace, name: jobName });
      toast.success('Evaluation job deleted successfully');
      onJobDeleted?.(job);
      handleModalClose();
    } catch (error) {
      logger.error('Failed to delete evaluation job', error);
      toast.error('Failed to delete evaluation job. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleModalClose = () => {
    setModalOpen(undefined);
  };

  return (
    <>
      <DropdownRoot>
        <DropdownTrigger asChild showChevron={false}>
          <Button
            kind="tertiary"
            aria-label="Open evaluation job actions menu"
            data-testid={`action-menu-${jobName}`}
          >
            <EllipsisVertical />
          </Button>
        </DropdownTrigger>
        <DropdownContent align="end" className="w-[180px]">
          <DropdownItem onClick={() => onNavigateToDetails(job)}>
            <Flex align="center" gap="density-sm">
              <ArrowRight size="20" fill="solid" />
              View Details
            </Flex>
          </DropdownItem>
          <DropdownItem onClick={() => setModalOpen('delete')} danger>
            <Flex align="center" gap="density-sm">
              <Trash size="20" fill="solid" />
              Delete
            </Flex>
          </DropdownItem>
        </DropdownContent>
      </DropdownRoot>

      {jobName && (
        <Modal
          open={modalOpen === 'delete'}
          onOpenChange={(open) => !open && !isDeleting && handleModalClose()}
          slotHeading={
            <Flex align="center" gap="density-sm">
              <Trash />
              Delete Evaluation Job
            </Flex>
          }
          slotFooter={
            <Flex justify="end" gap="density-xs" align="center" className="w-full">
              <Button
                onClick={handleModalClose}
                kind="tertiary"
                color="neutral"
                disabled={isDeleting}
              >
                Cancel
              </Button>
              <Button color="brand" onClick={handleDeleteJob} disabled={isDeleting}>
                {isDeleting ? 'Deleting...' : 'Delete'}
              </Button>
            </Flex>
          }
        >
          <Stack gap="density-md">
            <Text>
              Are you sure you want to delete evaluation job <strong>{jobName}</strong>?
            </Text>
            <Text>This action cannot be undone.</Text>
          </Stack>
        </Modal>
      )}
    </>
  );
};
