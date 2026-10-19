// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type ActionMenuItem, ActionsMenu } from '@nemo/common/src/components/ActionsMenu';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { CJobCancellableStatuses } from '@nemo/common/src/constants/query';
import {
  getDataDesignerListCreateJobsQueryKey,
  useDataDesignerCancelCreateJob,
} from '@nemo/sdk/generated/data-designer/data-designer';
import type { CreateJob as DataDesignerJob } from '@nemo/sdk/generated/data-designer/schema';
import { DeleteJobModal } from '@studio/components/dataViews/DataDesignerJobsDataView/DeleteJobModal';
import { buildClonedJobRequest } from '@studio/components/NewDataDesignerJobForm/utils';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getDataDesignerJobBuildRoute, getDataDesignerJobDetailsRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useCallback, useState } from 'react';
import { useNavigate } from 'react-router';

interface DataDesignerJobActionsMenuProps {
  job: DataDesignerJob;
  /** 'icon' renders the ellipsis trigger (tables); 'labeled' renders an "Actions" button (details page). */
  variant?: 'icon' | 'labeled';
  /** Page-specific actions listed above the job-level ones, e.g. Transform and Split. */
  additionalActions?: ActionMenuItem[];
  /** Include a "View details" entry. Used in the table row, omitted on the details page. */
  includeViewDetails?: boolean;
  /** When provided, adds a "View config" entry that invokes this callback. */
  onViewConfig?: () => void;
  /** Called after the job is successfully deleted, e.g. to navigate away from the details page. */
  onDeleted?: () => void;
  /** Surface a cancel error (or `undefined` to clear) so the caller can render it. */
  onCancelError?: (message: string | undefined) => void;
}

/**
 * Actions menu for a single Data Designer job: any caller-supplied actions, View details
 * (optional), Clone, Cancel (when cancellable), and Delete. Shared by the jobs table and the job details page so both
 * expose the same actions. Owns its own delete modal; cancel errors are surfaced via callback.
 */
export const DataDesignerJobActionsMenu: FC<DataDesignerJobActionsMenuProps> = ({
  job,
  variant = 'icon',
  additionalActions = [],
  includeViewDetails = false,
  onViewConfig,
  onDeleted,
  onCancelError,
}) => {
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();
  const queryClient = useQueryClient();
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const cancelJobMutation = useDataDesignerCancelCreateJob({
    mutation: {
      onSuccess: () => {
        queryClient.resetQueries({
          queryKey: getDataDesignerListCreateJobsQueryKey(workspace),
        });
        onCancelError?.(undefined);
      },
      onError: (error) => {
        onCancelError?.(error instanceof Error ? error.message : 'Failed to cancel job');
      },
    },
  });

  const handleClone = useCallback(() => {
    const cloneJobRequest = buildClonedJobRequest(job);
    if (!cloneJobRequest) return;
    navigate(getDataDesignerJobBuildRoute(workspace), {
      state: { cloneJobRequest },
    });
  }, [job, navigate, workspace]);

  const handleCancel = useCallback(async () => {
    if (!job.workspace || !job.name) return;
    try {
      onCancelError?.(undefined);
      await cancelJobMutation.mutateAsync({ workspace: job.workspace, name: job.name });
    } catch {
      // Error is surfaced via the mutation's onError callback.
    }
  }, [job.workspace, job.name, cancelJobMutation, onCancelError]);

  const isCancellable = job.status != null && CJobCancellableStatuses.includes(job.status);

  const actions: ActionMenuItem[] = [
    ...additionalActions,
    ...(includeViewDetails
      ? [
          {
            label: 'View details',
            onSelect: () => {
              if (job.name) {
                navigate(getDataDesignerJobDetailsRoute(workspace, job.name));
              }
            },
          },
        ]
      : []),
    ...(onViewConfig
      ? [
          {
            label: 'View config',
            onSelect: onViewConfig,
          },
        ]
      : []),
    {
      label: 'Clone',
      onSelect: handleClone,
    },
    ...(isCancellable
      ? [
          {
            label: 'Cancel',
            onSelect: handleCancel,
          },
        ]
      : []),
    {
      label: 'Delete',
      onSelect: () => setShowDeleteModal(true),
      danger: true,
    },
  ];

  return (
    <>
      {variant === 'labeled' ? (
        <ActionsMenu actions={actions} label="Actions" />
      ) : (
        <QuickActionsMenuRoot actions={actions} />
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
