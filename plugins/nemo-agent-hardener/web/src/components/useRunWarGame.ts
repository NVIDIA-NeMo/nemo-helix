// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toastApiError } from '@agent-hardener/api/apiError';
import { usePolledJob } from '@agent-hardener/api/platform';
import { waitForJobRun } from '@agent-hardener/api/runJobs';
import { useJobFailure } from '@agent-hardener/components/useJobFailure';
import { useAgentHardenerCreateJob } from '@agent-hardener/generated/api';
import { useToast } from '@agent-hardener/host';
import { getAgentHardenerRunDetailsRoute, getAgentHardenerRunListRoute } from '@agent-hardener/paths';
import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router';

// Submit the war-game job for a manifest and open the run it creates. The service-driven job creates its
// run record shortly after it starts (linked by job_id), so we poll the newest runs for that link and
// navigate to the run detail. The job is polled alongside: if it fails before its run exists, we stay put
// and surface the failure (toast + `failure` for a JobFailureNotice) instead of landing on the run list.
export const useRunWarGame = (workspace: string) => {
  const navigate = useNavigate();
  const toast = useToast();
  const [jobName, setJobName] = useState('');
  const { data: job } = usePolledJob(workspace, jobName);
  const failure = useJobFailure('War-game', jobName, job);
  // Read by the run poll, which outlives the render that started it.
  const failedJob = useRef('');
  useEffect(() => {
    if (failure) failedJob.current = jobName;
  }, [failure, jobName]);

  // Poll snappily (0.5s) over a wide window (~30s) so slow job starts still land on the run.
  const openRunForJob = async (name: string): Promise<void> => {
    const failed = () => failedJob.current === name;
    let run: string | undefined;
    try {
      run = await waitForJobRun(workspace, name, { attempts: 60, intervalMs: 500, stop: failed });
    } catch (error) {
      toastApiError(toast, 'Could not open the war-game run')(error);
      return;
    }
    if (run) {
      navigate(getAgentHardenerRunDetailsRoute(workspace, run));
    } else if (!failed()) {
      toast.info('The run has not appeared yet; it will show in this list once its job starts.');
      navigate(getAgentHardenerRunListRoute(workspace));
    }
  };

  const create = useAgentHardenerCreateJob({
    mutation: {
      onSuccess: (created) => {
        toast.success('War-game started — opening the run…');
        setJobName(created.name);
        void openRunForJob(created.name);
      },
      onError: toastApiError(toast, 'Failed to start the war-game'),
    },
  });

  return { ...create, jobName, failure };
};
