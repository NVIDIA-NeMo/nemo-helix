// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { CJobCancellableStatuses } from '@nemo/common/src/constants/query';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getJobsGetJobQueryKey,
  getJobsListJobsQueryKey,
  useJobsCancelJob,
} from '@nemo/sdk/generated/platform/jobs';
import { PlatformJobStatus } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex, Text } from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { X } from 'lucide-react';
import { FC, MouseEvent, useState } from 'react';

interface CancelJobButtonProps {
  workspace: string;
  jobName: string;
  jobStatus?: PlatformJobStatus;
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
  const toast = useToast();
  const queryClient = useQueryClient();

  const { mutateAsync, isPending } = useJobsCancelJob({
    mutation: {
      onSuccess: () => {
        toast.success('Job cancelled successfully.');
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

  const handleCancel = async () => {
    try {
      await mutateAsync({ workspace, name: jobName });
      setIsModalOpen(false);
    } catch (e) {
      toast.error(getErrorMessage(e as Error, 'Failed to cancel job'));
    }
  };

  const isCancellable = jobStatus && CJobCancellableStatuses.includes(jobStatus);
  const isCancelling = jobStatus === PlatformJobStatus.cancelling;

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
          <X className="w-3 h-3" />
          Cancel
        </Button>
      ) : (
        <Button kind="secondary" onClick={handleClick} disabled={isPending || isCancelling}>
          {isPending || isCancelling ? 'Cancelling...' : 'Cancel Job'}
        </Button>
      )}
      <FormModal
        open={isModalOpen}
        title={`Cancel ${jobName}`}
        submitButtonText="Cancel Job"
        onSubmit={(e) => {
          e.preventDefault();
          handleCancel();
        }}
        onClose={() => setIsModalOpen(false)}
        disabled={isPending}
        loading={isPending}
        attributes={{
          SubmitButton: { color: 'danger' },
        }}
      >
        <Flex>
          <Text className="leading-relaxed">
            Canceling this job will permanently stop it. This action cannot be undone, and the job
            cannot be relaunched or deleted. Are you sure you want to proceed?
          </Text>
        </Flex>
      </FormModal>
    </>
  );
};
