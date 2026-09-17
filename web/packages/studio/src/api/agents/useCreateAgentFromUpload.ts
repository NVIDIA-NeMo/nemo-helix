// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateAgent } from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { filesCreateFileset, filesUploadFile } from '@nemo/sdk/generated/platform/files';
import { claimFileset, rollbackFileset } from '@studio/api/agents/agentSpecFileset';
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
    await rollbackFileset(workspace, filesetName);
    throw error;
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

export type UseCreateAgentFromUploadOptions = Omit<
  UseMutationOptions<Agent, Error, CreateAgentFromUploadParams>,
  'mutationFn'
>;

export const useCreateAgentFromUpload = (options?: UseCreateAgentFromUploadOptions) =>
  useMutation({ ...options, mutationFn: createAgentFromUpload });
