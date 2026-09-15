// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateAgentRequestConfig } from '@nemo/sdk/generated/agents/schema/CreateAgentRequestConfig';
import { fetchSampleText } from '@studio/api/agents/fetchSampleText';
import YAML from 'yaml';

/**
 * Loads a sample agent's config from a public static asset and injects the
 * selected model plus the target workspace. Parse-then-set: the fetched YAML's
 * literals are overwritten, so the asset can stay byte-identical to the
 * plugin's ${NEMO_DEFAULT_MODEL} version (the platform service doesn't resolve
 * that).
 *
 * Branches on the config's own `config_format`:
 * - NAT (`nat-workflow-v1`, the default when absent): model lives at
 *   `llms.llm.model_name`.
 * - Fabric (`nemo-agents-spec-v1`): model lives at `models.default.model`; the
 *   selected harness inherits it when it declares no model of its own.
 */
export const loadSampleAgentConfig = async (
  agentConfigPath: string,
  modelName: string,
  workspace: string
): Promise<CreateAgentRequestConfig> => {
  const text = await fetchSampleText(agentConfigPath);
  const config = YAML.parse(text) as CreateAgentRequestConfig;

  const configFormat = config?.config_format;

  if (configFormat === 'nemo-agents-spec-v1') {
    injectFabricModel(config, modelName, agentConfigPath);
    injectIntakeWorkspace(config, workspace);
    return config;
  }

  if (configFormat === undefined || configFormat === 'nat-workflow-v1') {
    injectNatModel(config, modelName, agentConfigPath);
    injectIntakeWorkspace(config, workspace);
    return config;
  }

  throw new Error(
    `Sample agent config ${agentConfigPath} has unsupported config_format: ${String(configFormat)}`
  );
};

/** NAT workflow config: overwrite `llms.llm.model_name`. */
const injectNatModel = (
  config: Record<string, unknown>,
  modelName: string,
  agentConfigPath: string
): void => {
  const llm = (config?.llms as { llm?: unknown } | undefined)?.llm;
  if (!llm || typeof llm !== 'object' || Array.isArray(llm)) {
    throw new Error(`Sample agent config ${agentConfigPath} is missing llms.llm`);
  }
  (llm as Record<string, unknown>).model_name = modelName;
};

const INTAKE_WORKSPACE_RE = /(\/apis\/intake\/v\d+\/workspaces\/)[^/]+(\/)/;

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;

/**
 * Point ATIF HTTP storage endpoints at the workspace the agent is created in.
 * The sample assets ship a literal `default`, so without this the agent's
 * traces land in the wrong workspace and never reach its own agent page.
 */
const injectIntakeWorkspace = (config: Record<string, unknown>, workspace: string): void => {
  const atif = asRecord(asRecord(config.telemetry)?.atif);
  const storage = atif?.storage;
  if (!Array.isArray(storage)) return;

  for (const entry of storage) {
    const record = asRecord(entry);
    if (record?.type !== 'http' || typeof record.endpoint !== 'string') continue;
    record.endpoint = record.endpoint.replace(
      INTAKE_WORKSPACE_RE,
      (_match, prefix: string, suffix: string) => `${prefix}${workspace}${suffix}`
    );
  }
};

/** Fabric (nemo-agents-spec-v1) config: overwrite `models.default.model`. */
const injectFabricModel = (
  config: Record<string, unknown>,
  modelName: string,
  agentConfigPath: string
): void => {
  const models = config?.models as { default?: unknown } | undefined;
  const defaultModel = models?.default;
  if (!defaultModel || typeof defaultModel !== 'object' || Array.isArray(defaultModel)) {
    throw new Error(`Sample agent config ${agentConfigPath} is missing models.default`);
  }
  (defaultModel as Record<string, unknown>).model = modelName;
};
