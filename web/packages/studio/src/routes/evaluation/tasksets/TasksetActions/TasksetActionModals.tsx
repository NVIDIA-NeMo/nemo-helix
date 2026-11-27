// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { useEvalsDeleteTaskset } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import { invalidateTasksets } from '@studio/routes/evaluation/tasksets/helpers';
import { TasksetFormModal } from '@studio/routes/evaluation/tasksets/TasksetFormModal';
import type { TasksetActionRequest } from '@studio/routes/evaluation/tasksets/types';
import { useQueryClient } from '@tanstack/react-query';
import type { FC } from 'react';

interface Props {
  request?: TasksetActionRequest;
  onClose: () => void;
  /** Runs before the taskset queries refetch, e.g. to leave a detail page that would 404. */
  onDeleted?: () => void;
}

export const TasksetActionModals: FC<Props> = ({ request, onClose, onDeleted }) => {
  const queryClient = useQueryClient();
  const { mutateAsync: deleteTaskset } = useEvalsDeleteTaskset();

  if (!request) return null;
  const { action, taskset } = request;

  // Errors propagate so ConfirmationModal toasts the server's reason.
  const handleDelete = async () => {
    await deleteTaskset({ workspace: taskset.workspace, name: taskset.name });
    onDeleted?.();
    await invalidateTasksets(queryClient, taskset.workspace);
    return true;
  };

  return action === 'edit' ? (
    <TasksetFormModal open taskset={taskset} onClose={onClose} />
  ) : (
    <DeleteConfirmationModal
      open
      simpleConfirm
      title={`Delete taskset: ${taskset.name}`}
      description="This deletes the taskset and all of its revisions. Its member tasks are not deleted."
      successText="Taskset deleted."
      onDelete={handleDelete}
      onClose={onClose}
    />
  );
};
