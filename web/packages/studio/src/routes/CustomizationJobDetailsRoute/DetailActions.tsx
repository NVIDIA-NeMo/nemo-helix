// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { DownloadMenu } from '@nemo/common/src/components/DownloadMenu';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { CJobCancellableStatuses } from '@nemo/common/src/constants/query';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { toError } from '@nemo/common/src/utils/logger';
import { Cancel, Bookmark, CopyGeneric } from '@nemo/icons/icons';
import { useCustomizationCancelAutomodelJob } from '@nemo/sdk/generated/customizer/automodel-jobs';
import { useCustomizationCancelRlJob } from '@nemo/sdk/generated/customizer/rl-jobs';
import { useCustomizationCancelUnslothJob } from '@nemo/sdk/generated/customizer/unsloth-jobs';
import { getJobsGetJobQueryKey } from '@nemo/sdk/generated/platform/jobs';
import { HelixJobStatus, type HelixJobResponse } from '@nemo/sdk/generated/platform/schema';
import { Flex } from '@nvidia/foundations-react-core';
import { getCustomizationJobStatusQueryKey } from '@studio/hooks/useCustomizationJobStatus';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { SaveTemplateModal } from '@studio/routes/CustomizationJobDetailsRoute/SaveTemplateModal';
import { getNewCustomizationFormRoute } from '@studio/routes/utils';
import { hasMetrics } from '@studio/types/customization';
import { CustomizationBackend, type CustomizationJob } from '@studio/util/customizationBackend';
import { customizationMetricsExport } from '@studio/util/resultExports';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useState } from 'react';
import { useNavigate } from 'react-router';

interface DetailActionsProps {
  status?: HelixJobStatus;
  /** Training backend of this job, needed to target the correct per-backend cancel endpoint. */
  backend?: CustomizationBackend;
  /** Job name (from the route). */
  name: string;
  job?: CustomizationJob;
}

/**
 * This component renders the primary top-level CTAs for the customization job details page.
 */
export const DetailActions: FC<DetailActionsProps> = ({ status, backend, name, job }) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const workspace = useWorkspaceFromPath();
  const [savingTemplate, setSavingTemplate] = useState(false);

  const cancelMutation = {
    onSuccess: () => {
      toast.success('Job cancelled successfully.');
      queryClient.setQueryData(
        getJobsGetJobQueryKey(workspace, name),
        (oldData: HelixJobResponse | undefined) => {
          if (!oldData) return oldData;
          return { ...oldData, status: HelixJobStatus.cancelled };
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
  const isCancelling = isPending || status === HelixJobStatus.cancelling;

  const statusDetails = job?.status_details;
  const metrics = hasMetrics(statusDetails) ? statusDetails.metrics : undefined;

  return (
    <Flex gap="density-sm" align="center">
      <DownloadMenu
        label="Download metrics"
        fileName={`${name}-metrics`}
        disabled={!Object.values(metrics ?? {}).some((series) => series?.length)}
        descriptions={{
          json: 'Job details with every metric series',
          jsonl: 'One training step per line',
          csv: 'Training steps as a table, for spreadsheets',
        }}
        getData={() => customizationMetricsExport({ name, status }, metrics)}
      />
      <QuickActionsMenuRoot
        actions={[
          {
            label: 'Clone',
            icon: <CopyGeneric />,
            onSelect: () =>
              navigate(getNewCustomizationFormRoute(workspace), { state: { cloneFromJob: job } }),
          },
          // With Clone: both start a new job from this one.
          ...(job
            ? [
                {
                  label: 'Save as template',
                  icon: <Bookmark />,
                  onSelect: () => setSavingTemplate(true),
                },
              ]
            : []),
          ...(isCancellable || isCancelling
            ? [
                {
                  label: isCancelling ? 'Cancelling…' : 'Cancel Job',
                  icon: <Cancel />,
                  danger: !isCancelling,
                  disabled: isCancelling || !backend,
                  onSelect: cancelJob,
                },
              ]
            : []),
        ]}
      />

      {/* Mounted only while open, or it reopens holding the last name typed into it. */}
      {job && savingTemplate ? (
        <SaveTemplateModal
          open
          onClose={() => setSavingTemplate(false)}
          workspace={workspace}
          job={job}
          onSaved={(saved) => toast.success(`Saved “${saved}” as a template.`)}
        />
      ) : null}
    </Flex>
  );
};
