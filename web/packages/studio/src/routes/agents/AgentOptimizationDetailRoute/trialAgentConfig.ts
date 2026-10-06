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
   * The optimize config's own `models` map, which the study merges over the agent's models by key.
   * Optimize configs routinely restate the agent's `default` model (to point it at a different
   * endpoint or credential), and add evaluation-only models such as a `judge`.
   */
  overlayModels: Readonly<Record<string, ConfigMapping>>;
}

/** A model the study ran under a different model id than the agent declares for the same key. */
export interface ModelMismatch {
  modelKey: string;
  studyModel: string;
  agentModel: string;
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
  overlayModels: isMapping(optimizeConfig.models)
    ? Object.fromEntries(
        Object.entries(optimizeConfig.models).filter((entry): entry is [string, ConfigMapping] =>
          isMapping(entry[1])
        )
      )
    : {},
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
  return isMapping(harness) && isMapping(harness.model)
    ? `harnesses.${harnessName}.model`
    : undefined;
};

/**
 * Where the agent keeps the model the study saw under `modelKey`, before the optimize config was
 * merged in. A spec-v1 default harness that pins its own model is what Fabric publishes as
 * `default`, ahead of `models.default`.
 */
const agentModelPath = (config: ConfigMapping, modelKey: string): string | undefined => {
  if (modelKey === FABRIC_DEFAULT_MODEL_KEY && config.config_format === PLATFORM_AGENT_FORMAT) {
    const harnessModelPath = defaultHarnessModelPath(config);
    if (harnessModelPath) return harnessModelPath;
  }
  const models = config.models;
  return isMapping(models) && isMapping(models[modelKey]) ? `models.${modelKey}` : undefined;
};

const getByDottedPath = (config: ConfigMapping, path: string): unknown =>
  path
    .split('.')
    .reduce<unknown>((node, segment) => (isMapping(node) ? node[segment] : undefined), config);

interface ResolvedPath {
  /** Where to write on the agent config, or null for a study-only model with nothing to apply. */
  path: string | null;
  /** The model key the Fabric path tunes, for model paths. */
  modelKey?: string;
}

/**
 * Translate a Fabric path back onto the stored agent config. Throws for a path the agent cannot
 * take, so a configuration that was never evaluated is not deployed.
 */
const resolveAgentPath = (
  config: ConfigMapping,
  fabricPath: string,
  overlayModels: StudyConfig['overlayModels']
): ResolvedPath => {
  const [root, modelKey, ...rest] = fabricPath.split('.');
  if (root !== 'models' || !modelKey) {
    if (config.config_format === PLATFORM_AGENT_FORMAT) {
      throw new Error(
        `Cannot apply "${fabricPath}" to a ${PLATFORM_AGENT_FORMAT} agent; only model settings are supported.`
      );
    }
    return { path: fabricPath };
  }

  const modelPath = agentModelPath(config, modelKey);
  if (!modelPath) {
    // An evaluation-only model (a judge, say) that the optimize config added: not part of the agent.
    if (modelKey in overlayModels) return { path: null, modelKey };
    throw new Error(`"${fabricPath}" tunes a "${modelKey}" model this agent does not define.`);
  }
  return { path: [modelPath, ...rest].join('.'), modelKey };
};

/** The study's model id for `modelKey`, when it differs from the one the agent declares. */
const findModelMismatch = (
  config: ConfigMapping,
  modelKey: string,
  overlayModels: StudyConfig['overlayModels']
): ModelMismatch | undefined => {
  const studyModel = overlayModels[modelKey]?.model;
  const modelPath = agentModelPath(config, modelKey);
  const agentModel = modelPath ? getByDottedPath(config, `${modelPath}.model`) : undefined;
  if (typeof studyModel !== 'string' || typeof agentModel !== 'string') return undefined;
  return studyModel === agentModel ? undefined : { modelKey, studyModel, agentModel };
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

export interface AppliedTrialConfig {
  config: ConfigMapping;
  /** Names of the trial params left out because they tune a study-only model. */
  skipped: string[];
  /** Models whose trial values were tuned against a different model id than the agent uses. */
  modelMismatches: ModelMismatch[];
}

/**
 * A copy of the agent's stored config with the trial's sampled params written over it, the same
 * way the study wrote them over the agent when it ran the trial. Params on study-only models are
 * skipped; throws when none of the trial's params apply to the agent. Paths resolve against the
 * source config so an earlier param cannot change where a later one lands.
 */
export const applyTrialToAgentConfig = (
  config: ConfigMapping,
  trial: Trial,
  { searchSpace, overlayModels }: StudyConfig
): AppliedTrialConfig => {
  const next = structuredClone(config);
  const skipped: string[] = [];
  const mismatches = new Map<string, ModelMismatch>();
  for (const param of trial.params) {
    const entry = searchSpace[param.name];
    if (!entry) {
      throw new Error(`Parameter "${param.name}" is not in the study's search space.`);
    }
    const { path, modelKey } = resolveAgentPath(config, entry.path, overlayModels);
    if (path === null) {
      skipped.push(param.name);
      continue;
    }
    setByDottedPath(next, path, coerceParamValue(param.value, entry));
    const mismatch = modelKey ? findModelMismatch(config, modelKey, overlayModels) : undefined;
    if (mismatch) mismatches.set(mismatch.modelKey, mismatch);
  }
  if (trial.params.length > 0 && skipped.length === trial.params.length) {
    throw new Error(
      `None of trial ${trial.number}'s parameters apply to this agent; they all tune models from the study's optimize config.`
    );
  }
  return { config: next, skipped, modelMismatches: [...mismatches.values()] };
};
