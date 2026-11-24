// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JobFailure } from '@agent-hardener/api/jobFailure';
import { getJobDetailRoute } from '@agent-hardener/paths';
import { Banner, Stack, Text } from '@nvidia/foundations-react-core';
import { FC } from 'react';
import { Link } from 'react-router';

interface JobFailureNoticeProps {
  /** What failed, e.g. "Benign-suite generation failed". */
  title: string;
  failure: JobFailure;
  workspace: string;
  jobName: string;
}

// A failed job's cause, what to do about it, and a way to its logs: the same three things wherever a job can fail.
export const JobFailureNotice: FC<JobFailureNoticeProps> = ({
  title,
  failure,
  workspace,
  jobName,
}) => (
  <Banner kind="inline" status="error">
    <Stack gap="density-xs">
      <Text kind="body/semibold/md">
        {title}: {failure.message}
      </Text>
      {failure.remediation ? <Text kind="body/regular/sm">{failure.remediation}</Text> : null}
      <Text kind="body/regular/sm">
        <Link to={getJobDetailRoute(workspace, jobName)}>View job {jobName}</Link>
      </Text>
    </Stack>
  </Banner>
);
