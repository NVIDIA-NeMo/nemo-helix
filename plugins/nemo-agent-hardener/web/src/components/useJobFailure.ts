// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { endedJobFailure, type JobFailure } from '@agent-hardener/api/jobFailure';
import { useToast } from '@agent-hardener/host';
import { useEffect, useRef } from 'react';

type PolledJob = Parameters<typeof endedJobFailure>[0];

// A polled job's failure once it ends without completing, announced with one toast per job so a failure the user
// started is never silent, even when the panel that shows it is off screen. A job picked back up on a later visit
// is not announced: its failure is old news, shown by the panel alone.
export const useJobFailure = (
  label: string,
  jobName: string,
  job: PolledJob,
  { announce = true }: { announce?: boolean } = {}
): JobFailure | undefined => {
  const toast = useToast();
  const failure = endedJobFailure(job);
  const announcedFor = useRef('');
  useEffect(() => {
    if (!failure || !announce || announcedFor.current === jobName) return;
    announcedFor.current = jobName;
    toast.error(`${label} failed: ${failure.message}`);
  }, [failure, announce, jobName, label, toast]);
  return failure;
};
