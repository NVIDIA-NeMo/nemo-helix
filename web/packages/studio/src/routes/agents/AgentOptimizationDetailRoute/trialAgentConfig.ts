// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RunStrategySpec } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategySpec';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { parseOptimizeConfig } from '@studio/api/agents/optimizeBundle';
import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';

const PLATFORM_AGENT_FORMAT = 'nemo-agents-spec-v1';

/** Fabric republishes the selected harness's model as `models.default` (see the Fabric translator). */
const FABRIC_DEFAULT_MODEL_PREFIX = 'models.default.';

type ConfigMapping = Record<string, unknown>;

/** One `optimizer.search_space` entry: the dotted Fabric path a trial param writes to. */
export interface SearchSpaceEntry {
  path: string;
  /** Categorical choices, kept so a trial's CSV string can be restored to its declared type. */
  values?: readonly unknown[];
}

export type SearchSpace = Record<string, SearchSpaceEntry>;

export const buildTrialAgentName = (sourceName: string, trial: Trial): string =>
  `${sourceName}-trial-${trial.number}`;

const isMapping = (value: unknown): value is ConfigMapping =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** `optimizer.search_space` keyed by the logical param names the trials CSV uses. */
export const parseSearchSpace = (optimizeConfig: ConfigMapping): SearchSpace => {
  const optimizer = optimizeConfig.optimizer;
  const raw = isMapping(optimizer) ? optimizer.search_space : undefined;
  if (!isMapping(raw)) return {};

  const space: SearchSpace = {};
  for (const [name, entry] of Object.entries(raw)) {
    if (!isMapping(entry) || typeof entry.path !== 'string' || !entry.path) continue;
    space[name] = {
      path: entry.path,
      values: Array.isArray(entry.values) ? entry.values : undefined,
    };
  }
  return space;
};

/**
 * Download the study's optimize config and pull out its search space. Throws when the job did
 * not stage its config in a fileset, since the trial params cannot be mapped to config paths
 * without it.
 */
export const fetchSearchSpace = async (
  spec: RunStrategySpec | undefined,
  signal?: AbortSignal
): Promise<SearchSpace> => {
  const filesetRef = spec?.optimize_config_fileset;
  const configPath = spec?.optimize_config;
  const [filesetWorkspace, filesetName] = filesetRef?.split('/') ?? [];
  if (!filesetWorkspace || !filesetName || !configPath) {
    throw new Error('This study has no optimize config fileset, so its trials cannot be deployed.');
  }

  const blob = await filesDownloadFile(filesetWorkspace, filesetName, configPath, signal);
  const config = blob ? parseOptimizeConfig(await blob.text()) : undefined;
  if (!config) throw new Error(`Could not read the optimize config "${configPath}".`);
  return parseSearchSpace(config);
};

/**
 * The trials CSV stores every param as a string. Restore the type the config expects: the
 * matching categorical choice when the search space lists them, otherwise a number (numeric
 * ranges are the only other kind of numeric search-space entry).
 */
export const coerceParamValue = (value: string, entry: SearchSpaceEntry): unknown => {
  const choice = entry.values?.find((candidate) => String(candidate) === value);
  if (choice !== undefined) return choice;
  const parsed = Number(value);
  return value.trim() !== '' && Number.isFinite(parsed) ? parsed : value;
};

/**
 * Translate a Fabric path back onto the stored agent config. A `nemo-agents-spec-v1` agent whose
 * default harness pins its own model reads that model ahead of `models.default`, so the overlay
 * has to land on the harness for the trial value to take effect.
 */
const toAgentConfigPath = (config: ConfigMapping, fabricPath: string): string => {
  if (config.config_format !== PLATFORM_AGENT_FORMAT) return fabricPath;
  if (!fabricPath.startsWith('models.')) {
    throw new Error(
      `Cannot apply "${fabricPath}" to a ${PLATFORM_AGENT_FORMAT} agent; only model settings are supported.`
    );
  }
  if (!fabricPath.startsWith(FABRIC_DEFAULT_MODEL_PREFIX)) return fabricPath;

  const harnessName = config.default_harness;
  const harnesses = config.harnesses;
  const harness =
    typeof harnessName === 'string' && isMapping(harnesses) ? harnesses[harnessName] : undefined;
  if (isMapping(harness) && isMapping(harness.model)) {
    return `harnesses.${harnessName}.model.${fabricPath.slice(FABRIC_DEFAULT_MODEL_PREFIX.length)}`;
  }
  return fabricPath;
};

/** Mirrors `nemo_optimization.config_overlay.set_by_dotted_path`: intermediate maps are created. */
const setByDottedPath = (config: ConfigMapping, path: string, value: unknown): void => {
  const segments = path.split('.');
  const leaf = segments.pop();
  if (!leaf) return;
  let node = config;
  for (const segment of segments) {
    const next = node[segment];
    if (!isMapping(next)) node[segment] = {};
    node = node[segment] as ConfigMapping;
  }
  node[leaf] = value;
};

/**
 * A copy of the agent's stored config with the trial's sampled params written over it, the same
 * way the study wrote them over the agent when it ran the trial.
 */
export const applyTrialToAgentConfig = (
  config: ConfigMapping,
  trial: Trial,
  searchSpace: SearchSpace
): ConfigMapping => {
  const next = structuredClone(config);
  for (const param of trial.params) {
    const entry = searchSpace[param.name];
    if (!entry) {
      throw new Error(`Parameter "${param.name}" is not in the study's search space.`);
    }
    setByDottedPath(
      next,
      toAgentConfigPath(next, entry.path),
      coerceParamValue(param.value, entry)
    );
  }
  return next;
};
