// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { agentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import { filesDeleteFileset, filesRetrieveFileset } from '@nemo/sdk/generated/platform/files';

/** An agent of this name already exists; its spec fileset is not ours to take. */
export class AgentSpecFilesetConflictError extends Error {
  constructor(public readonly filesetName: string) {
    super(
      `An agent named "${filesetName.replace(/-ethos$/, '')}" already owns the fileset "${filesetName}". Choose a different name.`
    );
  }
}

/** A spec fileset with no agent behind it — an abandoned upload, or an agent since deleted. */
export class AgentSpecFilesetOrphanError extends Error {
  constructor(public readonly filesetName: string) {
    super(
      `A fileset named "${filesetName}" already exists, but no agent owns it — an upload that did not finish, or an agent that was deleted, since deleting an agent leaves its fileset behind. Replacing it discards its current contents.`
    );
  }
}

export const claimFileset = async (
  workspace: string,
  agentName: string,
  filesetName: string,
  replaceOrphanedFileset: boolean
): Promise<void> => {
  try {
    await filesRetrieveFileset(workspace, filesetName);
  } catch (error) {
    // Only a 404 means the name is free; anything else leaves ownership unknown.
    if (!isNotFoundError(error)) throw error;
    return;
  }

  if (await agentExists(workspace, agentName)) {
    throw new AgentSpecFilesetConflictError(filesetName);
  }
  if (!replaceOrphanedFileset) {
    throw new AgentSpecFilesetOrphanError(filesetName);
  }

  await filesDeleteFileset(workspace, filesetName);
};

const agentExists = async (workspace: string, agentName: string): Promise<boolean> => {
  try {
    await agentsGetAgent(workspace, agentName);
    return true;
  } catch (error) {
    if (!isNotFoundError(error)) throw error;
    return false;
  }
};

export const rollbackFileset = async (workspace: string, filesetName: string): Promise<void> => {
  await filesDeleteFileset(workspace, filesetName).catch(() => undefined);
};
