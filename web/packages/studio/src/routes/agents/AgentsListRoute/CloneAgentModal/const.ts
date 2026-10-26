// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RegisterAgentRequestConfig } from '@nemo/sdk/generated/agents/schema/RegisterAgentRequestConfig';
import type { AgentConfig } from '@studio/components/dataViews/AgentsDataView';
import { getAgentModelNames } from '@studio/components/dataViews/AgentsDataView/utils';
import { z } from 'zod';

const CLONE_NAME_SUFFIX_LENGTH = 6;

// Mirrors the example-agent name scheme: a short random suffix keeps clones unique.
export const buildClonedAgentName = (sourceName: string): string =>
  `${sourceName}-${Math.random()
    .toString(36)
    .slice(2, 2 + CLONE_NAME_SUFFIX_LENGTH)}`;

// The agent's primary model is the one its workflow points at; fall back to the first model found.
export const getPrimaryModelName = (config: AgentConfig | undefined): string | undefined => {
  const primaryKey = config?.workflow?.llm_name;
  const primary = primaryKey ? config?.llms?.[primaryKey]?.model_name : undefined;
  return primary ?? getAgentModelNames(config)[0];
};

export const applyModelToConfig = (
  config: AgentConfig | undefined,
  modelName: string
): RegisterAgentRequestConfig => {
  const cloned: AgentConfig = config ? JSON.parse(JSON.stringify(config)) : {};
  const llms = cloned.llms;
  if (llms) {
    const primaryKey = cloned.workflow?.llm_name;
    if (primaryKey && llms[primaryKey]) {
      llms[primaryKey].model_name = modelName;
    } else {
      for (const llm of Object.values(llms)) {
        if (llm.model_name !== undefined) llm.model_name = modelName;
      }
    }
  }
  return cloned as RegisterAgentRequestConfig;
};

export const cloneAgentFormSchema = z.object({
  name: z.string(),
  modelName: z.string().min(1, 'Model is required'),
});
