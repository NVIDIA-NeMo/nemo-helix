// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  agentHardenerDownloadVictimContextJobResult,
  useAgentHardenerCreateVictimContextJob,
  useAgentHardenerGetVictimContextJob,
} from '@agent-hardener/generated/api';
import { useToast } from '@agent-hardener/host';
import { CJobTerminalStatuses, getJobRefetchInterval, triggerDownload } from '@nemo/common';
import { useState } from 'react';

/** Job result name the victim-context job saves its archive under (`CONTEXT_RESULT` in the plugin). */
const CONTEXT_RESULT = 'victim-context';

export interface StageVictimContext {
  start: () => void;
  /** A staging job is submitted and has not finished. */
  staging: boolean;
  /** The archive is ready to download. */
  ready: boolean;
  /** The last staging job, when it ended without an archive. */
  failedJob?: string;
  download: () => Promise<void>;
  downloading: boolean;
  /** File name the archive downloads as; the build commands refer to it. */
  archiveName: string;
}

/**
 * Stage a manifest's victim build context on the platform and hand the archive to the browser.
 *
 * A war-game that runs in a container cannot build the victim, so the operator builds it from this
 * archive (`docker build -t <ref> - < archive`) and records the pushed image on the manifest.
 */
export const useStageVictimContext = (workspace: string, manifestName: string): StageVictimContext => {
  const toast = useToast();
  const [jobName, setJobName] = useState('');
  const [downloading, setDownloading] = useState(false);
  const archiveName = `${manifestName}-victim-context.tar.gz`;

  const create = useAgentHardenerCreateVictimContextJob({
    mutation: {
      onSuccess: (job) => setJobName(job.name),
      onError: () => toast.error('Failed to start staging the victim build context.'),
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

  const download = async (): Promise<void> => {
    setDownloading(true);
    try {
      const archive = await agentHardenerDownloadVictimContextJobResult(workspace, jobName, CONTEXT_RESULT);
      triggerDownload(archive, archiveName);
    } catch {
      toast.error('Failed to download the victim build context.');
    } finally {
      setDownloading(false);
    }
  };

  return {
    start: () => create.mutate({ workspace, data: { spec: { manifest_id: manifestName } } }),
    staging: create.isPending || (Boolean(jobName) && !terminal),
    ready: status === 'completed',
    failedJob: terminal && status !== 'completed' ? jobName : undefined,
    download,
    downloading,
    archiveName,
  };
};
