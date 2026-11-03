// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { announceCancellation } from '@nemo/common/src/utils/announceCancellation';
import { jobsGetJob } from '@nemo/sdk/generated/platform/jobs';
import { useCallback } from 'react';

/**
 * Returns a function to call once a cancel request is accepted. It announces the request and,
 * if the job is still `cancelling`, its final result. Status is read from the platform jobs
 * endpoint, which every service's jobs are also served by. See `announceCancellation`.
 */
export const useAnnounceCancellation = () => {
  const toast = useToast();
  return useCallback(
    (workspace: string, job: { name: string; status?: string }) => {
      void announceCancellation({
        toast,
        jobName: job.name,
        status: job.status,
        fetchStatus: async () => (await jobsGetJob(workspace, job.name)).status,
      });
    },
    [toast]
  );
};
