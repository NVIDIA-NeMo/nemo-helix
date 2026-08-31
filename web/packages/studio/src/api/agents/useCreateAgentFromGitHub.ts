// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateAgent } from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { filesCreateFileset, filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { claimFileset, rollbackFileset } from '@studio/api/agents/agentSpecFileset';
import {
  AGENT_CONFIG_FILENAME,
  FABRIC_CONFIG_FORMAT,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import {
  type GitHubAgentSource,
  formatGitHubSource,
  githubStorageConfig,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/github';
import {
  agentSpecFilesetName,
  parseAgentConfig,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { UseMutationOptions, useMutation } from '@tanstack/react-query';

export interface CreateAgentFromGitHubParams {
  workspace: string;
  name: string;
  source: GitHubAgentSource;
  /** Workspace secret holding a personal access token. Omit for a public repository. */
  secretName?: string;
  replaceOrphanedFileset?: boolean;
}

/**
 * The fileset reads the repository directly, so agent.yaml is fetched back through the files
 * API rather than from GitHub — the token stays in the files service and never reaches here.
 */
export const createAgentFromGitHub = async ({
  workspace,
  name,
  source,
  secretName,
  replaceOrphanedFileset = false,
}: CreateAgentFromGitHubParams): Promise<Agent> => {
  const filesetName = agentSpecFilesetName(name);

  await claimFileset(workspace, name, filesetName, replaceOrphanedFileset);

  await filesCreateFileset(workspace, {
    name: filesetName,
    description: `Agent spec for ${name}, from ${formatGitHubSource(source)}`,
    storage: githubStorageConfig(source, secretName),
  });

  try {
    const config = parseAgentConfig(await readAgentConfig(workspace, filesetName, source));

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

const readAgentConfig = async (
  workspace: string,
  filesetName: string,
  source: GitHubAgentSource
): Promise<string> => {
  try {
    const blob = await filesDownloadFile(workspace, filesetName, AGENT_CONFIG_FILENAME);
    return await blob.text();
  } catch (error) {
    throw new Error(
      `Could not read ${AGENT_CONFIG_FILENAME} from ${formatGitHubSource(source)}. ` +
        'Check the branch and directory, and that the secret can read a private repository.',
      { cause: error }
    );
  }
};

export type UseCreateAgentFromGitHubOptions = Omit<
  UseMutationOptions<Agent, Error, CreateAgentFromGitHubParams>,
  'mutationFn'
>;

export const useCreateAgentFromGitHub = (options?: UseCreateAgentFromGitHubOptions) =>
  useMutation({ ...options, mutationFn: createAgentFromGitHub });
