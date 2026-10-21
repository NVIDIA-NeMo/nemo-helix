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
  // Keep the run record (and its job ID) if job cleanup fails, so the operator
  // sees the error and still has a record to investigate rather than an orphan.
  if (run.job_id) {
    await inJobCollection(
      { workspace, name: run.job_id },
      agentHardenerDeleteJob,
      agentHardenerDeleteSynthBenignJob
    );
  }
  await agentHardenerDeleteRun(workspace, run.name);
};
