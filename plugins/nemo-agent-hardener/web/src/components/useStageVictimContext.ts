// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toastApiError } from '@agent-hardener/api/apiError';
import type { JobFailure } from '@agent-hardener/api/jobFailure';
import { useJobFailure } from '@agent-hardener/components/useJobFailure';
import {
  agentHardenerDownloadVictimContextJobResult,
  getAgentHardenerGetManifestQueryKey,
  useAgentHardenerCreateVictimContextJob,
  useAgentHardenerGetVictimContextJob,
} from '@agent-hardener/generated/api';
import { useToast } from '@agent-hardener/host';
import { CJobTerminalStatuses, getJobRefetchInterval, triggerDownload } from '@nemo/common';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';

/** Job result name the victim-context job saves its archive under (`CONTEXT_RESULT` in the plugin). */
const CONTEXT_RESULT = 'victim-context';

export interface StageVictimContext {
  start: () => void;
  /** Stage, then build the image in the cluster and record it on the manifest. */
  startBuild: () => void;
  /** A staging job is submitted and has not finished. */
  staging: boolean;
  /** The last staging job also builds the image. */
  build: boolean;
  /** The archive is ready to download. */
  ready: boolean;
  /** The last staging job. */
  jobName: string;
  /** Why the last staging job ended without an archive. */
  failure?: JobFailure;
  download: () => Promise<void>;
  downloading: boolean;
  /** File name the archive downloads as; the build commands refer to it. */
  archiveName: string;
}

/**
 * Stage a manifest's victim build context on the platform and hand the archive to the browser.
 *
 * A war-game that runs in a container cannot build the victim, so the operator builds it from this
 * archive (`docker build -t <ref> - < archive`) and records the pushed image on the manifest. Or the job
 * builds it with the builder plugin and records it itself; the manifest is refetched once it has.
 */
export const useStageVictimContext = (workspace: string, manifestName: string): StageVictimContext => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [jobName, setJobName] = useState('');
  const [build, setBuild] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const archiveName = `${manifestName}-victim-context.tar.gz`;

  const create = useAgentHardenerCreateVictimContextJob({
    mutation: {
      onSuccess: (job) => setJobName(job.name),
      onError: toastApiError(toast, 'Failed to start staging the victim build context'),
    },
  });

  const { data: job } = useAgentHardenerGetVictimContextJob(workspace, jobName, {
    query: {
      enabled: Boolean(jobName),
      refetchInterval: (query) => getJobRefetchInterval(query.state.data?.status),
    },
  });
  const status = job?.status;
  const terminal = Boolean(status && CJobTerminalStatuses.includes(status));
  const failure = useJobFailure(
    build ? 'Building the victim image' : 'Staging the victim build context',
    jobName,
    job
  );

  const recorded = useRef('');
  useEffect(() => {
    if (!build || status !== 'completed' || recorded.current === jobName) return;
    recorded.current = jobName;
    toast.success('Victim image built and recorded on the manifest.');
    void queryClient.invalidateQueries({
      queryKey: getAgentHardenerGetManifestQueryKey(workspace, manifestName),
    });
  }, [build, status, jobName, toast, queryClient, workspace, manifestName]);

  const submit = (withBuild: boolean): void => {
    setBuild(withBuild);
    create.mutate({ workspace, data: { spec: { manifest_id: manifestName, build: withBuild } } });
  };
  const staging = create.isPending || (Boolean(jobName) && !terminal);

  const download = async (): Promise<void> => {
    setDownloading(true);
    try {
      const archive = await agentHardenerDownloadVictimContextJobResult(workspace, jobName, CONTEXT_RESULT);
      triggerDownload(archive, archiveName);
    } catch (error) {
      toastApiError(toast, 'Failed to download the victim build context')(error);
    } finally {
      setDownloading(false);
    }
  };

  return {
    start: () => submit(false),
    startBuild: () => submit(true),
    staging,
    build,
    ready: status === 'completed',
    jobName,
    failure,
    download,
    downloading,
    archiveName,
  };
};
