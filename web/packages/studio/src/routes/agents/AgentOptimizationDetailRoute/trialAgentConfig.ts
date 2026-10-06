// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RunStrategySpec } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategySpec';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { parseOptimizeConfig } from '@studio/api/agents/optimizeBundle';
import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';

const PLATFORM_AGENT_FORMAT = 'nemo-agents-spec-v1';

/**
 * The Fabric translator forwards every model in the agent's `models` map under its own key, then
 * publishes the default harness's model as `default` (`harnesses[default_harness].model` when the
 * harness pins one, else `models.default`). Only `default` needs mapping back.
 */
const FABRIC_DEFAULT_MODEL_KEY = 'default';

type ConfigMapping = Record<string, unknown>;

/** One `optimizer.search_space` entry: the dotted Fabric path a trial param writes to. */
export interface SearchSpaceEntry {
  path: string;
  /** Categorical choices, kept so a trial's CSV string can be restored to its declared type. */
  values?: readonly unknown[];
}

export type SearchSpace = Record<string, SearchSpaceEntry>;

/** What the study's optimize config contributes to each trial's config. */
export interface StudyConfig {
  searchSpace: SearchSpace;
  /**
   * Keys of the optimize config's own `models` map. The study merges these over the agent's
   * models, replacing any model with the same key, so a trial value tuned on one of them was not
   * measured against the agent's model.
   */
  overlayModels: ReadonlySet<string>;
}

export const buildTrialAgentName = (sourceName: string, trial: Trial): string =>
  `${sourceName}-trial-${trial.number}`;

const isMapping = (value: unknown): value is ConfigMapping =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** `optimizer.search_space` keyed by the logical param names the trials CSV uses. */
const parseSearchSpace = (optimizeConfig: ConfigMapping): SearchSpace => {
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

export const parseStudyConfig = (optimizeConfig: ConfigMapping): StudyConfig => ({
  searchSpace: parseSearchSpace(optimizeConfig),
  overlayModels: new Set(isMapping(optimizeConfig.models) ? Object.keys(optimizeConfig.models) : []),
});

/**
 * Download the study's optimize config. Throws when the job did not stage its config in a
 * fileset, since the trial params cannot be mapped to config paths without it.
 */
export const fetchStudyConfig = async (
  spec: RunStrategySpec | undefined,
  signal?: AbortSignal
): Promise<StudyConfig> => {
  const filesetRef = spec?.optimize_config_fileset;
  const configPath = spec?.optimize_config;
  const [filesetWorkspace, filesetName] = filesetRef?.split('/') ?? [];
  if (!filesetWorkspace || !filesetName || !configPath) {
    throw new Error('This study has no optimize config fileset, so its trials cannot be deployed.');
  }

  const blob = await filesDownloadFile(filesetWorkspace, filesetName, configPath, signal);
  const config = blob ? parseOptimizeConfig(await blob.text()) : undefined;
  if (!config) throw new Error(`Could not read the optimize config "${configPath}".`);
  return parseStudyConfig(config);
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

const defaultHarnessModelPath = (config: ConfigMapping): string | undefined => {
  const harnessName = config.default_harness;
  const harnesses = config.harnesses;
  const harness =
    typeof harnessName === 'string' && isMapping(harnesses) ? harnesses[harnessName] : undefined;
  return isMapping(harness) && isMapping(harness.model) ? `harnesses.${harnessName}.model` : undefined;
};

/** The model keys the study saw for this agent, before its optimize config was merged in. */
const agentModelKeys = (config: ConfigMapping): Set<string> => {
  const keys = new Set(isMapping(config.models) ? Object.keys(config.models) : []);
  if (config.config_format === PLATFORM_AGENT_FORMAT && defaultHarnessModelPath(config)) {
    keys.add(FABRIC_DEFAULT_MODEL_KEY);
  }
  return keys;
};

/**
 * Translate a Fabric path back onto the stored agent config. Throws when the path tunes a model
 * the trial did not run with the agent's own settings, since applying it would deploy a
 * configuration that was never evaluated.
 */
const toAgentConfigPath = (
  config: ConfigMapping,
  fabricPath: string,
  overlayModels: ReadonlySet<string>
): string => {
  const isPlatformAgent = config.config_format === PLATFORM_AGENT_FORMAT;
  const [root, modelKey, ...rest] = fabricPath.split('.');
  if (root !== 'models' || !modelKey) {
    if (isPlatformAgent) {
      throw new Error(
        `Cannot apply "${fabricPath}" to a ${PLATFORM_AGENT_FORMAT} agent; only model settings are supported.`
      );
    }
    return fabricPath;
  }

  const onAgent = agentModelKeys(config).has(modelKey);
  if (overlayModels.has(modelKey)) {
    throw new Error(
      onAgent
        ? `The study replaced the agent's "${modelKey}" model with one from its optimize config, so "${fabricPath}" was not tuned against this agent's model.`
        : `"${fabricPath}" tunes the "${modelKey}" model from the study's optimize config, which is not part of this agent.`
    );
  }
  if (!onAgent) {
    throw new Error(`"${fabricPath}" tunes a "${modelKey}" model this agent does not define.`);
  }

  const harnessModelPath = isPlatformAgent ? defaultHarnessModelPath(config) : undefined;
  if (modelKey === FABRIC_DEFAULT_MODEL_KEY && harnessModelPath) {
    return [harnessModelPath, ...rest].join('.');
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
  { searchSpace, overlayModels }: StudyConfig
): ConfigMapping => {
  const next = structuredClone(config);
  for (const param of trial.params) {
    const entry = searchSpace[param.name];
    if (!entry) {
      throw new Error(`Parameter "${param.name}" is not in the study's search space.`);
    }
    setByDottedPath(
      next,
      toAgentConfigPath(next, entry.path, overlayModels),
      coerceParamValue(param.value, entry)
    );
  }
  return next;
};
