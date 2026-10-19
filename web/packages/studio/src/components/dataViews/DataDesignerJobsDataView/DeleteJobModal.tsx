// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getDataDesignerListCreateJobsQueryKey,
  useDataDesignerDeleteCreateJob,
} from '@nemo/sdk/generated/data-designer/data-designer';
import type { CreateJob as DataDesignerJob } from '@nemo/sdk/generated/data-designer/schema';
import { BulkDeleteModal } from '@studio/components/BulkDeleteModal';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useQueryClient } from '@tanstack/react-query';
import { FC } from 'react';

interface DeleteJobModalProps {
  jobs: DataDesignerJob[];
  onClose: () => void;
  onDeleted?: () => void;
}

export const DeleteJobModal: FC<DeleteJobModalProps> = ({ jobs, onClose, onDeleted }) => {
  const queryClient = useQueryClient();
  const workspace = useWorkspaceFromPath();

  const deleteJobMutation = useDataDesignerDeleteCreateJob({
    mutation: {
      onSuccess: () =>
        queryClient.resetQueries({
          queryKey: getDataDesignerListCreateJobsQueryKey(workspace),
        }),
    },
  });

  const handleDelete = async (jobsToDelete: DataDesignerJob[]) => {
    await Promise.all(
      jobsToDelete
        .filter((job) => job.workspace && job.name)
        .map(async (job) => {
          try {
            await deleteJobMutation.mutateAsync({ workspace: job.workspace!, name: job.name });
          } catch (error) {
            throw new Error(
              `Failed to delete job "${job.name}": ${error instanceof Error ? error.message : 'Unknown error'}`
            );
          }
        })
    );
    onDeleted?.();
  };

  return (
    <BulkDeleteModal
      items={jobs}
      open={jobs.length > 0}
      onDelete={handleDelete}
      title={(count) => `Delete ${count} Data Designer Job${count !== 1 ? 's' : ''}`}
      onClose={onClose}
    />
  );
};
