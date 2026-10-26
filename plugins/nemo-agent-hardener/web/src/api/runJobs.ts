// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  agentHardenerCancelJob,
  agentHardenerCancelSynthBenignJob,
  agentHardenerDeleteJob,
  agentHardenerDeleteRun,
  agentHardenerDeleteSynthBenignJob,
} from '@agent-hardener/generated/api';
import { isAxiosError } from 'axios';

interface RunJobTarget {
  workspace: string;
  name: string;
}

// Historical run records carry a job ID, but no job kind. Let the isolated
// endpoints resolve ownership instead of guessing from run names or outcomes.
// Only a 404 permits trying the sibling; auth/server/network failures propagate.
const inJobCollection = async (
  { workspace, name }: RunJobTarget,
  warGame: (workspace: string, name: string) => Promise<unknown>,
  synth: (workspace: string, name: string) => Promise<unknown>
): Promise<void> => {
  try {
    await warGame(workspace, name);
  } catch (error) {
    if (!isAxiosError(error) || error.response?.status !== 404) throw error;
    await synth(workspace, name);
  }
};

export const cancelRunJob = (target: RunJobTarget): Promise<void> =>
  inJobCollection(target, agentHardenerCancelJob, agentHardenerCancelSynthBenignJob);

export const deleteRunWithJob = async (
  workspace: string,
  run: { name: string; job_id?: string }
): Promise<void> => {
  // Keep the run record on auth/server/network failures, but allow retries after
  // successful job cleanup followed by failed run deletion (or a purged job).
  if (run.job_id) {
    try {
      await inJobCollection(
        { workspace, name: run.job_id },
        agentHardenerDeleteJob,
        agentHardenerDeleteSynthBenignJob
      );
    } catch (error) {
      // A 404 escapes inJobCollection only after both job collections return 404.
      // Treat that as completed cleanup for deletion only, never cancellation.
      if (!isAxiosError(error) || error.response?.status !== 404) throw error;
    }
  }
  await agentHardenerDeleteRun(workspace, run.name);
};
