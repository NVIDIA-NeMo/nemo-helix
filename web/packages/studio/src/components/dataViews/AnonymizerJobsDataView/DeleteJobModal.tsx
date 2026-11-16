// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import {
  getAnonymizerListRunJobsQueryKey,
  useAnonymizerDeleteRunJob,
} from '@nemo/sdk/generated/anonymizer/anonymizer';
import type { RunJob as AnonymizerJob } from '@nemo/sdk/generated/anonymizer/schema';
import { getDeleteJobsDescription } from '@studio/constants/platformJobs';
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

  const handleDelete = async () => {
    const invalid = jobs.filter((job) => !job.workspace || !job.name);
    if (invalid.length > 0) {
      throw new Error(
        `Cannot delete ${invalid.length} job${invalid.length !== 1 ? 's' : ''}: missing workspace or name.`
      );
    }
    await Promise.all(
      jobs.map(async (job) => {
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
    return true;
  };

  return (
    <DeleteConfirmationModal
      open={jobs.length > 0}
      entity="Anonymizer job"
      ids={jobs.map((job) => job.name)}
      description={getDeleteJobsDescription(jobs.length)}
      onDelete={handleDelete}
      onClose={onClose}
    />
  );
};
