// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { CJobCancellableStatuses, CJobLaunchableStatuses } from '@nemo/common/src/constants/query';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { toError } from '@nemo/common/src/utils/logger';
import { useCustomizationCancelAutomodelJob } from '@nemo/sdk/generated/customizer/automodel-jobs';
import { useCustomizationCancelRlJob } from '@nemo/sdk/generated/customizer/rl-jobs';
import { useCustomizationCancelUnslothJob } from '@nemo/sdk/generated/customizer/unsloth-jobs';
import { getJobsGetJobQueryKey } from '@nemo/sdk/generated/platform/jobs';
import { PlatformJobStatus, type PlatformJobResponse } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex } from '@nvidia/foundations-react-core';
import { getCustomizationJobStatusQueryKey } from '@studio/hooks/useCustomizationJobStatus';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getNewCustomizationJobRoute, getNewEvaluationMetricRoute } from '@studio/routes/utils';
import { CustomizationBackend, type CustomizationJob } from '@studio/util/customizationBackend';
import { useQueryClient } from '@tanstack/react-query';
import { Ban, Copy } from 'lucide-react';
import { FC } from 'react';
import { useNavigate } from 'react-router';

interface DetailActionsProps {
  model?: string;
  status?: PlatformJobStatus;
  /** Training backend of this job, needed to target the correct per-backend cancel endpoint. */
  backend?: CustomizationBackend;
  /** Job name (from the route). */
  name: string;
  job?: CustomizationJob;
}

/**
 * This component renders the primary top-level CTAs for the customization job details page.
 */
export const DetailActions: FC<DetailActionsProps> = ({ model, status, backend, name, job }) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const workspace = useWorkspaceFromPath();

  const cancelMutation = {
    onSuccess: () => {
      toast.success('Job cancelled successfully.');
      queryClient.setQueryData(
        getJobsGetJobQueryKey(workspace, name),
        (oldData: PlatformJobResponse | undefined) => {
          if (!oldData) return oldData;
          return { ...oldData, status: PlatformJobStatus.cancelled };
        }
      );
      // Per-step state and the derived duration are a separate query; without this they keep
      // showing the pre-cancel pipeline, and its polling has already stopped by now.
      if (backend) {
        void queryClient.invalidateQueries({
          queryKey: getCustomizationJobStatusQueryKey(backend, workspace, name),
        });
      }
    },
  };
  const automodelCancel = useCustomizationCancelAutomodelJob({ mutation: cancelMutation });
  const unslothCancel = useCustomizationCancelUnslothJob({ mutation: cancelMutation });
  const rlCancel = useCustomizationCancelRlJob({ mutation: cancelMutation });
  const isPending = automodelCancel.isPending || unslothCancel.isPending || rlCancel.isPending;

  type CancelJob = (vars: { workspace: string; name: string }) => Promise<unknown>;
  const cancelByBackend: Record<CustomizationBackend, CancelJob> = {
    [CustomizationBackend.automodel]: automodelCancel.mutateAsync,
    [CustomizationBackend.unsloth]: unslothCancel.mutateAsync,
    [CustomizationBackend.rl]: rlCancel.mutateAsync,
  };

  const cancelJob = async () => {
    if (!backend) {
      toast.error('Unable to determine the training backend for this job.');
      return;
    }
    const cancel = cancelByBackend[backend];
    try {
      await cancel({ workspace, name });
    } catch (e) {
      toast.error(`Failed to cancel job: ${getErrorMessage(toError(e))}`);
    }
  };

  const isCancellable = status !== undefined && CJobCancellableStatuses.includes(status);
  const isCancelling = isPending || status === PlatformJobStatus.cancelling;
  const isLaunchable = status !== undefined && CJobLaunchableStatuses.includes(status);

  return (
    <Flex gap="density-sm" align="center">
      {isLaunchable && (
        <Button
          color="brand"
          onClick={() => navigate(getNewEvaluationMetricRoute(workspace, { model }))}
        >
          Evaluate
        </Button>
      )}
      <QuickActionsMenuRoot
        actions={[
          {
            label: 'Clone',
            icon: <Copy />,
            onSelect: () =>
              navigate(getNewCustomizationJobRoute(workspace), { state: { cloneFromJob: job } }),
          },
          ...(isCancellable || isCancelling
            ? [
                {
                  label: isCancelling ? 'Cancelling…' : 'Cancel Job',
                  icon: <Ban />,
                  danger: !isCancelling,
                  disabled: isCancelling || !backend,
                  onSelect: cancelJob,
                },
              ]
            : []),
        ]}
      />
    </Flex>
  );
};
