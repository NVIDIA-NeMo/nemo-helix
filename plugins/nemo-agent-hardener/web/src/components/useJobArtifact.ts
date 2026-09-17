// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentHardenerDownloadJobResult, useAgentHardenerListJobResults } from '@agent-hardener/generated/api';
import type { PlatformJobStatus } from '@agent-hardener/generated/schema';
import { CJobTerminalStatuses, JOB_POLLING_INTERVAL_MS } from '@nemo/common';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

// A job flips to a terminal status before agent-hardener has finished uploading its results, so keep
// polling briefly past that point — otherwise a list fetched in the gap sticks as "no artifact".
const ARTIFACT_GRACE_MS = 20_000;

export interface JobArtifact<T> {
  /** Parsed artifact, once it has been downloaded. */
  data: T | undefined;
  /** The artifact exists in the job's result list. */
  present: boolean;
  /** Still waiting on the artifact, and the job could still produce it. */
  isLoading: boolean;
  /**
   * The job reached a terminal status without ever writing the artifact — it failed, was
   * cancelled, or completed without producing one. Callers should render an error here;
   * otherwise a failed job leaves the UI on a spinner forever.
   */
  missing: boolean;
}

/**
 * Poll a job's result list until `resultName` appears, then download and parse it once.
 *
 * Artifact presence alone is not a safe stop condition: a job that errors or is cancelled never
 * writes its result, so polling on that alone runs every {@link JOB_POLLING_INTERVAL_MS} for as
 * long as the page is mounted and the caller stays stuck loading. Passing `status` lets the poll
 * stop on a terminal job and surface {@link JobArtifact.missing} instead.
 */
export const useJobArtifact = <T>(
  workspace: string,
  jobName: string | undefined,
  resultName: string,
  parse: (blob: Blob) => Promise<T>,
  status?: PlatformJobStatus
): JobArtifact<T> => {
  const terminal = Boolean(status && CJobTerminalStatuses.includes(status));
  // Only give up once the job has been terminal for the grace period without the artifact showing.
  const [graceExpired, setGraceExpired] = useState(false);

  const { data: results } = useAgentHardenerListJobResults(workspace, jobName ?? '', {
    query: {
      enabled: Boolean(jobName),
      refetchInterval: (query) => {
        const found = query.state.data?.data?.some((result) => result.name === resultName);
        if (found) return false;
        return terminal && graceExpired ? false : JOB_POLLING_INTERVAL_MS;
      },
    },
  });

  const present = Boolean(results?.data?.some((result) => result.name === resultName));

  useEffect(() => {
    if (!terminal || present) {
      setGraceExpired(false);
      return;
    }
    const timer = setTimeout(() => setGraceExpired(true), ARTIFACT_GRACE_MS);
    return () => clearTimeout(timer);
  }, [terminal, present, jobName, resultName]);

  const query = useQuery({
    queryKey: ['agent-hardener-job-artifact', workspace, jobName, resultName],
    enabled: present && Boolean(jobName),
    queryFn: () => agentHardenerDownloadJobResult(workspace, jobName ?? '', resultName).then(parse),
  });

  return {
    data: query.data,
    present,
    isLoading: Boolean(jobName) && !present && (!terminal || !graceExpired),
    missing: Boolean(jobName) && terminal && !present && graceExpired,
  };
};

export const parseJson = <T>(blob: Blob): Promise<T> =>
  blob.text().then((text) => JSON.parse(text) as T);

export const parseText = (blob: Blob): Promise<string> => blob.text();
