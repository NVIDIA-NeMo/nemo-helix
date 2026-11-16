// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FABRIC_CONFIG_FORMAT } from '@nemo/common/src/utils/agents';
import { agentsRegisterAgent } from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { filesCreateFileset, filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { claimFileset, rollbackFileset } from '@studio/api/agents/agentSpecFileset';
import { AGENT_CONFIG_FILENAME } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import {
  type GitHubAgentSource,
  formatGitHubSource,
  githubStorageConfig,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/github';
import {
  agentSpecFilesetName,
  parseAgentConfig,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { type UseMutationOptions, useMutation } from '@tanstack/react-query';
import { AxiosError } from 'axios';

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

  try {
    await filesCreateFileset(workspace, {
      name: filesetName,
      description: `Agent spec for ${name}, from ${formatGitHubSource(source)}`,
      storage: githubStorageConfig(source, secretName),
    });
  } catch (error) {
    if (responseStatus(error) !== 400) {
      throw error;
    }
    throw new Error(`${getErrorMessage(error as Error)}. ${accessHint(secretName)}`, {
      cause: error,
    });
  }

  try {
    const config = parseAgentConfig(await readAgentConfig(workspace, filesetName, source));

    return await agentsRegisterAgent(workspace, {
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
      `Could not read ${AGENT_CONFIG_FILENAME} from ${formatGitHubSource(source)}. ${await readFailureHint(error)}`,
      { cause: error }
    );
  }
};

const responseStatus = (error: unknown): number | undefined =>
  error instanceof AxiosError ? error.response?.status : undefined;

// Downloads use a blob response type, so the platform's JSON detail arrives as a Blob.
const responseDetail = async (error: unknown): Promise<string> => {
  const data = error instanceof AxiosError ? error.response?.data : undefined;
  if (data instanceof Blob) {
    try {
      const parsed: unknown = JSON.parse(await data.text());
      const detail = (parsed as { detail?: unknown } | null)?.detail;
      if (typeof detail === 'string') {
        return detail;
      }
    } catch {
      // Not JSON; fall through to the generic message.
    }
  }
  return getErrorMessage(error as Error);
};

// A 502 covers both a refused token and GitHub being down; its detail already says which.
const readFailureHint = async (error: unknown): Promise<string> => {
  if (responseStatus(error) === 404) {
    return `Check that the branch and directory hold ${AGENT_CONFIG_FILENAME}.`;
  }
  return responseDetail(error);
};

const accessHint = (secretName: string | undefined): string => {
  if (secretName) {
    return `Also check that the token in "${secretName}" can read this repository.`;
  }
  return 'If the repository is private, choose an access token secret.';
};

export type UseCreateAgentFromGitHubOptions = Omit<
  UseMutationOptions<Agent, Error, CreateAgentFromGitHubParams>,
  'mutationFn'
>;

export const useCreateAgentFromGitHub = (options?: UseCreateAgentFromGitHubOptions) =>
  useMutation({ ...options, mutationFn: createAgentFromGitHub });
