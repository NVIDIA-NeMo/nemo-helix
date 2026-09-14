// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Mirrors Studio's @nemo/common CancelJobButton. It cannot be shared through
// the plugin surface: it imports the platform SDK, whose fetcher drags axios
// and oidc-client-ts — and therefore Node builtins — into the vendor bundle
// every plugin loads. The mutation comes off `host.sdk.platform` instead.

import { usePlatformSdk } from '@agent-hardener/api/platform';
import { useToast } from '@agent-hardener/host';
import { CJobCancellableStatuses, FormModal, getErrorMessage } from '@nemo/common';
import { Button, Flex, Text } from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { X } from 'lucide-react';
import { FC, MouseEvent, useState } from 'react';


interface CancelJobButtonProps {
  workspace: string;
  jobName: string;
  /** `PlatformJobStatus` from the platform SDK, kept loose to avoid a value import. */
  jobStatus?: string;
  compact?: boolean;
}

const CANCELLING = 'cancelling';

export const CancelJobButton: FC<CancelJobButtonProps> = ({
  workspace,
  jobName,
  jobStatus,
  compact,
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const toast = useToast();
  const queryClient = useQueryClient();
  const { useJobsCancelJob, getJobsGetJobQueryKey, getJobsListJobsQueryKey } = usePlatformSdk();

  const { mutateAsync, isPending } = useJobsCancelJob({
    mutation: {
      onSuccess: () => {
        toast.success('Job cancelled successfully.');
        queryClient.invalidateQueries({ queryKey: getJobsGetJobQueryKey(workspace, jobName) });
        queryClient.invalidateQueries({ queryKey: getJobsListJobsQueryKey(workspace) });
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

  const isCancellable = Boolean(
    jobStatus && (CJobCancellableStatuses as readonly string[]).includes(jobStatus)
  );
  const isCancelling = jobStatus === CANCELLING;

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
        attributes={{ SubmitButton: { color: 'danger' } }}
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
