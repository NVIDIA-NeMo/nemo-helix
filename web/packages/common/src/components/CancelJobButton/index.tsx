// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CancelJobModal } from '@nemo/common/src/components/CancelJobModal';
import { CJobCancellableStatuses } from '@nemo/common/src/constants/query';
import { useAnnounceCancellation } from '@nemo/common/src/hooks/useAnnounceCancellation';
import { Close } from '@nemo/icons/icons';
import {
  getJobsGetJobQueryKey,
  getJobsListJobsQueryKey,
  useJobsCancelJob,
} from '@nemo/sdk/generated/platform/jobs';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { Button } from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, type MouseEvent, useState } from 'react';

interface CancelJobButtonProps {
  workspace: string;
  jobName: string;
  jobStatus?: HelixJobStatus;
  compact?: boolean;
  /** Invalidate the caller's own list. This button only knows the platform
   *  jobs query keys, so a list backed by a different endpoint (e.g. the
   *  evaluator's) keeps showing the cancelled row as active without it. */
  onCancelled?: () => void;
}

export const CancelJobButton: FC<CancelJobButtonProps> = ({
  workspace,
  jobName,
  jobStatus,
  compact,
  onCancelled,
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const announceCancellation = useAnnounceCancellation();
  const queryClient = useQueryClient();

  const { mutateAsync, isPending } = useJobsCancelJob({
    mutation: {
      onSuccess: (job) => {
        announceCancellation(workspace, { name: jobName, status: job.status });
        queryClient.invalidateQueries({
          queryKey: getJobsGetJobQueryKey(workspace, jobName),
        });
        queryClient.invalidateQueries({
          queryKey: getJobsListJobsQueryKey(workspace),
        });
        onCancelled?.();
      },
    },
  });

  const isCancellable = jobStatus && CJobCancellableStatuses.includes(jobStatus);
  const isCancelling = jobStatus === HelixJobStatus.cancelling;

  if (!isCancellable && !isCancelling) {
    return null;
  }

  const handleClick = (e: MouseEvent) => {
    e.stopPropagation();
    setIsModalOpen(true);
  };

  return (
    <>
      {compact ? (
        <Button
          kind="secondary"
          color="danger"
          size="small"
          onClick={handleClick}
          disabled={isPending || isCancelling}
        >
          <Close className="w-3 h-3" />
          Cancel
        </Button>
      ) : (
        <Button kind="secondary" onClick={handleClick} disabled={isPending || isCancelling}>
          {isPending || isCancelling ? 'Cancelling...' : 'Cancel Job'}
        </Button>
      )}
      <CancelJobModal
        open={isModalOpen}
        jobName={jobName}
        onClose={() => setIsModalOpen(false)}
        onConfirm={async () => {
          await mutateAsync({ workspace, name: jobName });
        }}
      />
    </>
  );
};
