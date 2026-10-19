// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getAnonymizerListRunJobsQueryKey,
  useAnonymizerDeleteRunJob,
} from '@nemo/sdk/generated/anonymizer/anonymizer';
import type { RunJob as AnonymizerJob } from '@nemo/sdk/generated/anonymizer/schema';
import { BulkDeleteModal } from '@studio/components/BulkDeleteModal';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useQueryClient } from '@tanstack/react-query';
import type { FC } from 'react';

interface DeleteJobModalProps {
  jobs: AnonymizerJob[];
  onClose: () => void;
  onDeleted?: () => void;
}

export const DeleteJobModal: FC<DeleteJobModalProps> = ({ jobs, onClose, onDeleted }) => {
  const queryClient = useQueryClient();
  const workspace = useWorkspaceFromPath();

  const deleteJobMutation = useAnonymizerDeleteRunJob({
    mutation: {
      onSuccess: () =>
        queryClient.resetQueries({
          queryKey: getAnonymizerListRunJobsQueryKey(workspace),
        }),
    },
  });

  const handleDelete = async (jobsToDelete: AnonymizerJob[]) => {
    const invalid = jobsToDelete.filter((job) => !job.workspace || !job.name);
    if (invalid.length > 0) {
      throw new Error(
        `Cannot delete ${invalid.length} job${invalid.length !== 1 ? 's' : ''}: missing workspace or name.`
      );
    }
    await Promise.all(
      jobsToDelete.map(async (job) => {
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
      title={(count) => `Delete ${count} Anonymizer Job${count !== 1 ? 's' : ''}`}
      onClose={onClose}
    />
  );
};
