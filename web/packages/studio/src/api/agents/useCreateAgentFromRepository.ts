// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { agentsCreateAgent } from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import {
  filesCreateFileset,
  filesDownloadFile,
  filesListFilesetFiles,
} from '@nemo/sdk/generated/platform/files';
import type { GitStorageConfig, GithubStorageConfig } from '@nemo/sdk/generated/platform/schema';
import { claimFileset, rollbackFileset } from '@studio/api/agents/agentSpecFileset';
import {
  AGENT_CONFIG_FILENAME,
  AGENT_SPEC_DIR_FIELD,
  FABRIC_CONFIG_FORMAT,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import {
  agentDirectoryOf,
  agentSpecFilesetName,
  parseAgentConfig,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { type UseMutationOptions, useMutation } from '@tanstack/react-query';

/** The fileset has no agent.yaml at its root; `directories` are the ones below it that do. */
export class AgentConfigNotFoundError extends Error {
  constructor(
    sourceLabel: string,
    readonly directories: readonly string[],
    options?: ErrorOptions
  ) {
    super(
      directories.length > 0
        ? `There is no ${AGENT_CONFIG_FILENAME} in ${sourceLabel}. Choose an agent directory below.`
        : `There is no ${AGENT_CONFIG_FILENAME} anywhere in ${sourceLabel}. Check the branch.`,
      options
    );
  }
}

const joinPath = (...parts: string[]): string => parts.filter(Boolean).join('/');

/** Repository paths of every directory holding an agent.yaml; the repository root is `''`. */
const findAgentDirectories = async (
  workspace: string,
  filesetName: string,
  filesetRoot: string
): Promise<string[]> => {
  const { data } = await filesListFilesetFiles(workspace, filesetName);
  return data
    .map((file) => agentDirectoryOf(file.path))
    .filter((directory): directory is string => directory !== undefined)
    .map((directory) => joinPath(filesetRoot, directory))
    .sort();
};

export interface CreateAgentFromRepositoryParams {
  workspace: string;
  name: string;
  storage: GithubStorageConfig | GitStorageConfig;
  /** How the repository is named in the fileset description and error text. */
  sourceLabel: string;
  /**
   * The directory holding agent.yaml when the fileset is rooted above it, so the agent can
   * include files from elsewhere in the repository. Omit when the fileset is the agent's directory.
   */
  specDir?: string;
  replaceOrphanedFileset?: boolean;
}

/** Reads agent.yaml back through the files API, so repository credentials never reach the browser. */
export const createAgentFromRepository = async ({
  workspace,
  name,
  storage,
  sourceLabel,
  specDir,
  replaceOrphanedFileset = false,
}: CreateAgentFromRepositoryParams): Promise<Agent> => {
  const filesetName = agentSpecFilesetName(name);

  await claimFileset(workspace, name, filesetName, replaceOrphanedFileset);

  await filesCreateFileset(workspace, {
    name: filesetName,
    description: `Agent spec for ${name}, from ${sourceLabel}`,
    storage,
    ...(specDir ? { custom_fields: { [AGENT_SPEC_DIR_FIELD]: specDir } } : {}),
  });

  try {
    const config = parseAgentConfig(
      await readAgentConfig(workspace, filesetName, {
        configPath: joinPath(specDir ?? '', AGENT_CONFIG_FILENAME),
        filesetRoot: storage.path ?? '',
        sourceLabel,
      })
    );

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
  {
    configPath,
    filesetRoot,
    sourceLabel,
  }: { configPath: string; filesetRoot: string; sourceLabel: string }
): Promise<string> => {
  try {
    const blob = await filesDownloadFile(workspace, filesetName, configPath);
    return await blob.text();
  } catch (error) {
    if (isNotFoundError(error)) {
      const directories = await findAgentDirectories(workspace, filesetName, filesetRoot).catch(
        () => null
      );
      if (directories)
        throw new AgentConfigNotFoundError(sourceLabel, directories, { cause: error });
    }
    throw new Error(
      `Could not read ${AGENT_CONFIG_FILENAME} from ${sourceLabel}. ` +
        'Check the branch and directory, and that the secret can read a private repository.',
      { cause: error }
    );
  }
};

export type UseCreateAgentFromRepositoryOptions = Omit<
  UseMutationOptions<Agent, Error, CreateAgentFromRepositoryParams>,
  'mutationFn'
>;

export const useCreateAgentFromRepository = (options?: UseCreateAgentFromRepositoryOptions) =>
  useMutation({ ...options, mutationFn: createAgentFromRepository });
