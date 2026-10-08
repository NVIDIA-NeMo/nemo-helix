// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import type { RunStrategySpec } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategySpec';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { parseOptimizeConfig } from '@studio/api/agents/optimizeBundle';
import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import { FABRIC_CONFIG_FORMAT } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import { isPlainObject } from '@studio/util/functions';

// translate_agent_config publishes the default harness's model under this key.
const FABRIC_DEFAULT_MODEL_KEY = 'default';

// Keys a plain object resolves through its prototype, so writing one would reach Object.prototype.
const UNSAFE_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

export interface SearchSpaceEntry {
  path: string;
  // Categorical choices, used to restore a trial's CSV string to its declared type.
  values?: readonly unknown[];
}

export interface StudyConfig {
  searchSpace: ReadonlyMap<string, SearchSpaceEntry>;
  // Each replaces the agent's model of the same key whole (build_optimize_payload).
  overlayModels: ReadonlyMap<string, Record<string, unknown>>;
}

export interface StudyConfigLocation {
  workspace: string;
  fileset: string;
  path: string;
}

// A model the study ran with settings this agent's model does not have.
export interface ModelDifference {
  modelKey: string;
  fields: string[];
  studyModel?: string;
  agentModel?: string;
}

export interface AppliedTrialConfig {
  config: Record<string, unknown>;
  // Models only the optimize config defines. The study's agent had them, so the new one does too.
  addedModels: string[];
  modelDifferences: ModelDifference[];
}

export const buildTrialAgentName = (sourceName: string, trial: Trial): string =>
  `${sourceName}-trial-${trial.number}`;

const ownEntries = (value: unknown): [string, unknown][] =>
  isPlainObject(value) ? Object.entries(value) : [];

const parseSearchSpace = (
  optimizeConfig: Record<string, unknown>
): Map<string, SearchSpaceEntry> => {
  const optimizer = optimizeConfig.optimizer;
  const space = new Map<string, SearchSpaceEntry>();
  for (const [name, entry] of ownEntries(isPlainObject(optimizer) ? optimizer.search_space : {})) {
    if (!isPlainObject(entry) || typeof entry.path !== 'string' || !entry.path) continue;
    space.set(name, {
      path: entry.path,
      values: Array.isArray(entry.values) ? entry.values : undefined,
    });
  }
  return space;
};

export const parseStudyConfig = (optimizeConfig: Record<string, unknown>): StudyConfig => ({
  searchSpace: parseSearchSpace(optimizeConfig),
  overlayModels: new Map(
    ownEntries(optimizeConfig.models).filter((entry): entry is [string, Record<string, unknown>] =>
      isPlainObject(entry[1])
    )
  ),
});

// A bare fileset name resolves against the study's workspace, as the optimize job resolves it.
export const studyConfigLocation = (
  spec: RunStrategySpec | undefined,
  workspace: string
): StudyConfigLocation | undefined => {
  const fileset = spec?.optimize_config_fileset
    ? parseFilesetLocation(spec.optimize_config_fileset, workspace)
    : null;
  if (!fileset || !spec?.optimize_config) return undefined;
  return { workspace: fileset.workspace, fileset: fileset.name, path: spec.optimize_config };
};

export const fetchStudyConfig = async (
  { workspace, fileset, path }: StudyConfigLocation,
  signal?: AbortSignal
): Promise<StudyConfig> => {
  const blob = await filesDownloadFile(workspace, fileset, path, signal);
  const config = blob ? parseOptimizeConfig(await blob.text()) : undefined;
  if (!config) throw new Error(`Could not read the optimize config "${path}".`);
  return parseStudyConfig(config);
};

// Python's csv writer stores str(value), which spells booleans `True` and `False`.
const matchesCsvValue = (candidate: unknown, value: string): boolean => {
  if (typeof candidate === 'boolean') return value === (candidate ? 'True' : 'False');
  if (typeof candidate === 'number') return value.trim() !== '' && Number(value) === candidate;
  return String(candidate) === value;
};

// Numeric ranges are the only other kind of search-space entry, so a non-choice is a number.
export const coerceParamValue = (value: string, entry: SearchSpaceEntry): unknown => {
  const choice = entry.values?.find((candidate) => matchesCsvValue(candidate, value));
  if (choice !== undefined) return choice;
  const parsed = Number(value);
  return value.trim() !== '' && Number.isFinite(parsed) ? parsed : value;
};

const toSegments = (path: string): string[] => {
  const segments = path.split('.');
  if (segments.some((segment) => !segment || UNSAFE_KEYS.has(segment))) {
    throw new Error(`Cannot apply the search-space path "${path}".`);
  }
  return segments;
};

const getAt = (node: unknown, segments: readonly string[]): unknown =>
  segments.reduce<unknown>(
    (current, segment) =>
      isPlainObject(current) && Object.hasOwn(current, segment) ? current[segment] : undefined,
    node
  );

// Mirrors nemo_optimization.config_overlay.set_by_dotted_path, on path segments.
const setAt = (config: Record<string, unknown>, segments: readonly string[], value: unknown) => {
  if (segments.some((segment) => UNSAFE_KEYS.has(segment))) {
    throw new Error(`Cannot write to "${segments.join('.')}".`);
  }
  let node = config;
  for (const segment of segments.slice(0, -1)) {
    const existing = Object.hasOwn(node, segment) ? node[segment] : undefined;
    if (existing === undefined || existing === null) node[segment] = {};
    else if (!isPlainObject(existing)) {
      throw new Error(`Cannot write to "${segments.join('.')}": "${segment}" is not a mapping.`);
    }
    node = node[segment] as Record<string, unknown>;
  }
  node[segments[segments.length - 1]!] = value;
};

const isSameValue = (left: unknown, right: unknown): boolean => {
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((item, i) => isSameValue(item, right[i]));
  }
  if (isPlainObject(left) && isPlainObject(right)) {
    const keys = Object.keys(left);
    return (
      keys.length === Object.keys(right).length &&
      keys.every((key) => Object.hasOwn(right, key) && isSameValue(left[key], right[key]))
    );
  }
  return Object.is(left, right);
};

const isHelixAgent = (config: Record<string, unknown>): boolean =>
  config.config_format === FABRIC_CONFIG_FORMAT;

// translate_agent_config copies instructions.system.{content,mode} through to Fabric unchanged.
const SPEC_INSTRUCTION_FIELDS = new Set(['content', 'mode']);

const isSpecInstructionPath = (segments: readonly string[]): boolean =>
  segments.length === 3 &&
  segments[0] === 'instructions' &&
  segments[1] === 'system' &&
  SPEC_INSTRUCTION_FIELDS.has(segments[2] ?? '');

// Where the agent keeps the model the study ran as `modelKey`. A platform agent's default
// harness model outranks `models.default`, as in translate_agent_config.
const agentModelSegments = (
  config: Record<string, unknown>,
  modelKey: string
): string[] | undefined => {
  const harnessName = config.default_harness;
  if (
    modelKey === FABRIC_DEFAULT_MODEL_KEY &&
    isHelixAgent(config) &&
    typeof harnessName === 'string' &&
    isPlainObject(getAt(config, ['harnesses', harnessName, 'model']))
  ) {
    return ['harnesses', harnessName, 'model'];
  }
  return isPlainObject(getAt(config, ['models', modelKey])) ? ['models', modelKey] : undefined;
};

const findModelDifference = (
  modelKey: string,
  studyModel: Record<string, unknown>,
  agentModel: Record<string, unknown>
): ModelDifference | undefined => {
  const fields = [...new Set([...Object.keys(studyModel), ...Object.keys(agentModel)])].filter(
    (field) => !isSameValue(studyModel[field], agentModel[field])
  );
  if (fields.length === 0) return undefined;
  return fields.includes('model')
    ? {
        modelKey,
        fields,
        studyModel: String(studyModel.model),
        agentModel: String(agentModel.model),
      }
    : { modelKey, fields };
};

// The agent's config with the trial's params written over it, plus the optimize config's
// models the agent lacks. Models both define keep the agent's settings, and every setting
// that differs from what the study ran is reported.
export const applyTrialToAgentConfig = (
  config: Record<string, unknown>,
  trial: Trial,
  { searchSpace, overlayModels }: StudyConfig
): AppliedTrialConfig => {
  const next = structuredClone(config);
  const studyModels = new Map(
    [...overlayModels].map(([key, model]) => [key, structuredClone(model)] as const)
  );

  const addedModels = [...overlayModels.keys()].filter(
    (modelKey) => !agentModelSegments(config, modelKey)
  );
  for (const modelKey of addedModels) {
    setAt(next, ['models', modelKey], structuredClone(overlayModels.get(modelKey)));
  }
  // Resolved against the source config, so an earlier param cannot move where a later one lands.
  const modelTarget = (modelKey: string): string[] | undefined =>
    agentModelSegments(config, modelKey) ??
    (overlayModels.has(modelKey) ? ['models', modelKey] : undefined);

  let setsSystemInstruction = false;
  for (const param of trial.params) {
    const entry = searchSpace.get(param.name);
    if (!entry) {
      throw new Error(`Parameter "${param.name}" is not in the study's search space.`);
    }
    const value = coerceParamValue(param.value, entry);
    const segments = toSegments(entry.path);
    const [root, modelKey, ...field] = segments;

    if (root !== 'models' || !modelKey) {
      if (isHelixAgent(config)) {
        if (!isSpecInstructionPath(segments)) {
          throw new Error(
            `Cannot apply "${entry.path}" to a ${FABRIC_CONFIG_FORMAT} agent; only model settings and system instructions are supported.`
          );
        }
        setsSystemInstruction = true;
      }
      setAt(next, segments, value);
      continue;
    }

    const target = modelTarget(modelKey);
    if (!target) {
      throw new Error(`"${entry.path}" tunes a "${modelKey}" model this agent does not define.`);
    }
    setAt(next, [...target, ...field], value);

    const studyModel = studyModels.get(modelKey);
    if (!studyModel) continue;
    if (field.length > 0) setAt(studyModel, field, value);
    else if (isPlainObject(value)) studyModels.set(modelKey, structuredClone(value));
  }

  if (setsSystemInstruction) {
    const content = getAt(next, ['instructions', 'system', 'content']);
    if (typeof content !== 'string' || !content.trim()) {
      throw new Error('The trial leaves the agent without non-empty system instruction content.');
    }
  }

  const modelDifferences = [...studyModels].flatMap(([modelKey, studyModel]) => {
    if (addedModels.includes(modelKey)) return [];
    const target = modelTarget(modelKey);
    const agentModel = target ? getAt(next, target) : undefined;
    const difference = isPlainObject(agentModel)
      ? findModelDifference(modelKey, studyModel, agentModel)
      : undefined;
    return difference ? [difference] : [];
  });

  return { config: next, addedModels, modelDifferences };
};

// translate_agent_config tags traces with telemetry.agent_name, so it follows the new name.
export const withAgentName = (
  config: Record<string, unknown>,
  name: string
): Record<string, unknown> => {
  const renamed: Record<string, unknown> = { ...config };
  if (typeof config.name === 'string') renamed.name = name;
  const { telemetry } = config;
  if (isPlainObject(telemetry) && typeof telemetry.agent_name === 'string') {
    renamed.telemetry = { ...telemetry, agent_name: name };
  }
  return renamed;
};
