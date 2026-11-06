// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CancelJobModal } from '@nemo/common/src/components/CancelJobModal';
import {
  type QuickActionItem,
  QuickActionsMenuRoot,
} from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { CJobCancellableStatuses } from '@nemo/common/src/constants/query';
import { useAnnounceCancellation } from '@nemo/common/src/hooks/useAnnounceCancellation';
import {
  getAnonymizerListRunJobsQueryKey,
  useAnonymizerCancelRunJob,
} from '@nemo/sdk/generated/anonymizer/anonymizer';
import type { RunJob as AnonymizerJob } from '@nemo/sdk/generated/anonymizer/schema';
import { DeleteJobModal } from '@studio/components/dataViews/AnonymizerJobsDataView/DeleteJobModal';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getAnonymizerJobRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useCallback, useState } from 'react';
import { useNavigate } from 'react-router';

interface AnonymizerJobActionsMenuProps {
  job: AnonymizerJob;
  includeViewDetails?: boolean;
  onDeleted?: () => void;
}

export const AnonymizerJobActionsMenu: FC<AnonymizerJobActionsMenuProps> = ({
  job,
  includeViewDetails = false,
  onDeleted,
}) => {
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();
  const queryClient = useQueryClient();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showCancelModal, setShowCancelModal] = useState(false);
  const announceCancellation = useAnnounceCancellation();

  const cancelJobMutation = useAnonymizerCancelRunJob({
    mutation: {
      onSuccess: (job, variables) => {
        queryClient.resetQueries({
          queryKey: getAnonymizerListRunJobsQueryKey(workspace),
        });
        announceCancellation(variables.workspace, { name: variables.name, status: job.status });
      },
    },
  });

  const { mutateAsync: cancelJob } = cancelJobMutation;

  const handleCancel = useCallback(async () => {
    if (!job.workspace || !job.name) return;
    await cancelJob({ workspace: job.workspace, name: job.name });
  }, [job.workspace, job.name, cancelJob]);

  const isCancellable = job.status != null && CJobCancellableStatuses.includes(job.status);

  const actions: QuickActionItem[] = [
    ...(includeViewDetails
      ? [
          {
            label: 'View details',
            onSelect: () => {
              if (job.name) {
                navigate(getAnonymizerJobRoute(workspace, job.name));
              }
            },
          },
        ]
      : []),
    ...(isCancellable
      ? [
          {
            label: 'Cancel',
            onSelect: () => setShowCancelModal(true),
          },
        ]
      : []),
  ];

  return (
    <>
      <QuickActionsMenuRoot
        actions={actions}
        deleteAction={{ onSelect: () => setShowDeleteModal(true) }}
      />
      {showCancelModal && (
        <CancelJobModal
          open
          jobName={job.name ?? ''}
          onClose={() => setShowCancelModal(false)}
          onConfirm={handleCancel}
        />
      )}
      {showDeleteModal && (
        <DeleteJobModal
          jobs={[job]}
          onClose={() => setShowDeleteModal(false)}
          onDeleted={onDeleted}
        />
      )}
    </>
  );
};
