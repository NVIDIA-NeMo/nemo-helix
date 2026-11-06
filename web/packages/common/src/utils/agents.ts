// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** `config_format` of Platform-managed (Fabric) agents. Only these can be packaged; NAT workflows build from a checkout. */
export const FABRIC_CONFIG_FORMAT = 'nemo-agents-spec-v1';

/** A model entry in a `nemo-agents-spec-v1` config, under `models` or on a harness. */
export interface AgentSpecModel {
  provider?: string;
  model?: string;
  api_key_env?: string;
  temperature?: number;
  settings?: Record<string, unknown>;
}

export interface AgentConfig {
  functions?: Record<string, { _type: string }>;
  llms?: Record<
    string,
    {
      _type: string;
      model_name?: string;
      api_key?: string;
      temperature?: number;
      base_url?: string;
    }
  >;
  /** `nemo-agents-spec-v1` named models, e.g. `{ default: { model: 'nvidia-…' } }`. */
  models?: Record<string, AgentSpecModel>;
  harnesses?: Record<string, { kind?: string; model?: AgentSpecModel }>;
  workflow?: {
    _type: string;
    tool_names?: string[];
    llm_name?: string;
    verbose?: boolean;
    parse_agent_response_max_retries?: number;
  };
}

/**
 * Model names an agent is configured to call, across both config formats: NAT
 * workflows name them under `llms`, `nemo-agents-spec-v1` under `models` and on
 * each harness.
 */
export const getAgentModelNames = (config: AgentConfig | undefined): string[] => {
  const seen = new Set<string>();
  for (const llm of Object.values(config?.llms ?? {})) {
    if (llm.model_name) seen.add(llm.model_name);
  }
  for (const model of Object.values(config?.models ?? {})) {
    if (model?.model) seen.add(model.model);
  }
  for (const harness of Object.values(config?.harnesses ?? {})) {
    if (harness?.model?.model) seen.add(harness.model.model);
  }
  return [...seen];
};
