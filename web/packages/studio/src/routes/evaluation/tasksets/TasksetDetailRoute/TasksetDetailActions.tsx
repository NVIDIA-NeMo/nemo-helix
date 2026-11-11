// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { useEvalsDeleteTaskset } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import type { Taskset } from '@nemo/sdk/generated/evals/schema';
import { invalidateTasksets } from '@studio/routes/evaluation/tasksets/helpers';
import { TasksetFormModal } from '@studio/routes/evaluation/tasksets/TasksetFormModal';
import { getTasksetsRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useState } from 'react';
import { useNavigate } from 'react-router';

export const TasksetDetailActions: FC<{ taskset: Taskset }> = ({ taskset }) => {
  const { workspace, name } = taskset;
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [modal, setModal] = useState<'edit' | 'delete'>();
  const { mutateAsync: deleteTaskset } = useEvalsDeleteTaskset();

  // Errors propagate so ConfirmationModal toasts the server's reason.
  const handleDelete = async () => {
    await deleteTaskset({ workspace, name });
    navigate(getTasksetsRoute(workspace));
    await invalidateTasksets(queryClient, workspace);
    return true;
  };

  return (
    <>
      <QuickActionsMenuRoot
        actions={[{ label: 'Edit', onSelect: () => setModal('edit') }]}
        deleteAction={{ onSelect: () => setModal('delete') }}
      />
      {modal === 'edit' && (
        <TasksetFormModal open taskset={taskset} onClose={() => setModal(undefined)} />
      )}
      {modal === 'delete' && (
        <DeleteConfirmationModal
          open
          simpleConfirm
          title={`Delete taskset: ${name}`}
          description="This deletes the taskset and all of its revisions. Its member tasks are not deleted."
          successText="Taskset deleted."
          onDelete={handleDelete}
          onClose={() => setModal(undefined)}
        />
      )}
    </>
  );
};
