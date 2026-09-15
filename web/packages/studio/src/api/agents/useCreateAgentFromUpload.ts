// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { agentsCreateAgent, agentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import {
  filesCreateFileset,
  filesDeleteFileset,
  filesRetrieveFileset,
  filesUploadFile,
} from '@nemo/sdk/generated/platform/files';
import {
  AGENT_CONFIG_FILENAME,
  FABRIC_CONFIG_FORMAT,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import type { UploadAgentEntry } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';
import {
  agentSpecFilesetName,
  parseAgentConfig,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { UseMutationOptions, useMutation } from '@tanstack/react-query';

export interface CreateAgentFromUploadParams {
  workspace: string;
  name: string;
  entries: UploadAgentEntry[];
  replaceOrphanedFileset?: boolean;
}

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

// Files first: the fileset reserves the name, and a create-time validation that needs a
// base_dir can only see files that are already uploaded. Creating it outside the try keeps
// rollback to what this call created.
export const createAgentFromUpload = async ({
  workspace,
  name,
  entries,
  replaceOrphanedFileset = false,
}: CreateAgentFromUploadParams): Promise<Agent> => {
  const filesetName = agentSpecFilesetName(name);

  const configEntry = entries.find((entry) => entry.path === AGENT_CONFIG_FILENAME);
  if (!configEntry) throw new Error(`No ${AGENT_CONFIG_FILENAME} in the selected directory.`);
  const config = parseAgentConfig(await configEntry.file.text());

  await claimFileset(workspace, name, filesetName, replaceOrphanedFileset);

  await filesCreateFileset(workspace, {
    name: filesetName,
    description: `Agent spec for ${name}`,
  });

  try {
    await uploadEntries(workspace, filesetName, entries);

    return await agentsCreateAgent(workspace, {
      name,
      description: typeof config.description === 'string' ? config.description : '',
      config,
      config_format: FABRIC_CONFIG_FORMAT,
    });
  } catch (error) {
    await rollback(workspace, filesetName);
    throw error;
  }
};

const claimFileset = async (
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

// One request per file, so a 500-file agent is 500 round trips. Run a bounded number at
// once: unbounded Promise.all would queue them all against the browser's per-host limit
// and lose the first error behind hundreds of in-flight requests.
const UPLOAD_CONCURRENCY = 6;

const uploadEntries = async (
  workspace: string,
  filesetName: string,
  entries: UploadAgentEntry[]
): Promise<void> => {
  const queue = [...entries];
  const worker = async (): Promise<void> => {
    for (let entry = queue.shift(); entry; entry = queue.shift()) {
      const blob = new Blob([await entry.file.arrayBuffer()], { type: 'application/octet-stream' });
      await filesUploadFile(workspace, filesetName, entry.path, blob);
    }
  };

  await Promise.all(
    Array.from({ length: Math.min(UPLOAD_CONCURRENCY, entries.length) }, () => worker())
  );
};

const rollback = async (workspace: string, filesetName: string): Promise<void> => {
  await filesDeleteFileset(workspace, filesetName).catch(() => undefined);
};

export type UseCreateAgentFromUploadOptions = Omit<
  UseMutationOptions<Agent, Error, CreateAgentFromUploadParams>,
  'mutationFn'
>;

export const useCreateAgentFromUpload = (options?: UseCreateAgentFromUploadOptions) =>
  useMutation({ ...options, mutationFn: createAgentFromUpload });
