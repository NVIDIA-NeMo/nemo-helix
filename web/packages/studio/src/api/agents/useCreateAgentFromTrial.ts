// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { agentsCreateAgent } from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { filesCreateFileset, filesRetrieveFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { claimFileset, rollbackFileset } from '@studio/api/agents/agentSpecFileset';
import type {
  CreateAgentFromTrialParams,
  UseCreateAgentFromTrialOptions,
} from '@studio/api/agents/types';
import { copyFilesetFiles } from '@studio/api/files/uploadFilesetEntries';
import { agentSpecFilesetName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { useMutation } from '@tanstack/react-query';

const retrieveFileset = async (
  workspace: string,
  name: string
): Promise<FilesetOutput | undefined> => {
  try {
    return await filesRetrieveFileset(workspace, name);
  } catch (error) {
    if (isNotFoundError(error)) return undefined;
    throw error;
  }
};

export const createAgentFromTrial = async ({
  workspace,
  name,
  source,
  config,
  replaceOrphanedFileset = false,
}: CreateAgentFromTrialParams): Promise<Agent> => {
  const filesetName = agentSpecFilesetName(name);
  const sourceFileset = await retrieveFileset(source.workspace, agentSpecFilesetName(source.name));

  await claimFileset(workspace, name, filesetName, replaceOrphanedFileset);

  const createAgent = () =>
    agentsCreateAgent(workspace, {
      name,
      description: source.description,
      config,
      config_format: source.config_format,
    });
  if (!sourceFileset) return createAgent();

  const sharesStorage = sourceFileset.storage.type === 'github';
  await filesCreateFileset(workspace, {
    name: filesetName,
    description: `Agent spec for ${name}, copied from ${source.workspace}/${source.name}`,
    storage: sharesStorage ? sourceFileset.storage : undefined,
  });

  try {
    if (!sharesStorage) {
      await copyFilesetFiles(
        { workspace: sourceFileset.workspace, name: sourceFileset.name },
        { workspace, name: filesetName }
      );
    }
    return await createAgent();
  } catch (error) {
    await rollbackFileset(workspace, filesetName);
    throw error;
  }
};

export const useCreateAgentFromTrial = (options?: UseCreateAgentFromTrialOptions) =>
  useMutation({ ...options, mutationFn: createAgentFromTrial });
