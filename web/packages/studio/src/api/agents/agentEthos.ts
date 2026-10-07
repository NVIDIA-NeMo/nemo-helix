// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { datasetFileContentQueryOptions } from '@studio/api/datasets/useDatasetFileContent';
import { queryClient } from '@studio/api/queryClient';
import {
  AGENT_ETHOS_FILE,
  agentSpecFilesetName,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';

export const readAgentEthos = async (
  workspace: string,
  agent: string
): Promise<string | undefined> => {
  try {
    const content = await queryClient.fetchQuery({
      ...datasetFileContentQueryOptions({
        workspace,
        name: agentSpecFilesetName(agent),
        path: AGENT_ETHOS_FILE,
        fullContent: true,
      }),
      staleTime: 0,
      retry: false,
    });
    return content.trim() ? content : undefined;
  } catch (error) {
    if (isNotFoundError(error) || (error instanceof Error && isNotFoundError(error.cause))) {
      return undefined;
    }
    throw error;
  }
};
