// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RoutingStrategyId } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/formValues';
import {
  downloadText,
  locateJobResultFiles,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import { isPlainObject } from '@studio/util/functions';
import YAML from 'yaml';

/** Written by the nemo-switchyard plugin's `switchyard` strategy beside one agent YAML per route. */
export const SWITCHYARD_INDEX_FILE = 'switchyard-result.json';

/** One VirtualModel the run created, and the agent config pointed at it. */
export interface SwitchyardRoute {
  virtualModel: string;
  configType: RoutingStrategyId | string;
  capable: string;
  efficient: string;
  judge: string | null;
  /** The `nemo-switchyard` middleware config the VirtualModel was created with. */
  config: Record<string, unknown>;
  /** Fileset path of the rewritten agent config, or undefined when the index names none. */
  agentConfigPath?: string;
}

export interface SwitchyardResults {
  /** `workspace/name` of the agent the run routed. */
  agent: string;
  /** Workspace the VirtualModels were created in. */
  workspace: string;
  routes: SwitchyardRoute[];
  /** Where the result files live, for fetching each route's agent config. */
  location: { workspace: string; fileset: string };
}

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' && value ? value : undefined;

/** The index's `agent_config` is a file name beside the index itself. */
const siblingPath = (indexPath: string, fileName: string): string => {
  const slash = indexPath.lastIndexOf('/');
  return slash === -1 ? fileName : `${indexPath.slice(0, slash + 1)}${fileName}`;
};

export const parseSwitchyardIndex = (
  text: string,
  indexPath: string,
  location: SwitchyardResults['location'],
  fallbackWorkspace: string
): SwitchyardResults => {
  const raw: unknown = JSON.parse(text);
  if (!isPlainObject(raw)) throw new Error(`${SWITCHYARD_INDEX_FILE} is not a JSON object.`);

  const combinations = Array.isArray(raw.combinations) ? raw.combinations : [];
  const routes = combinations.flatMap<SwitchyardRoute>((entry) => {
    if (!isPlainObject(entry)) return [];
    const virtualModel = asString(entry.virtual_model);
    const capable = asString(entry.capable);
    const efficient = asString(entry.efficient);
    if (!virtualModel || !capable || !efficient) return [];
    const agentConfig = asString(entry.agent_config);
    return [
      {
        virtualModel,
        configType: asString(entry.config_type) ?? 'random_routing',
        capable,
        efficient,
        judge: asString(entry.judge) ?? null,
        config: isPlainObject(entry.config) ? entry.config : {},
        agentConfigPath: agentConfig ? siblingPath(indexPath, agentConfig) : undefined,
      },
    ];
  });

  return {
    agent: asString(raw.agent) ?? '',
    workspace: asString(raw.workspace) ?? fallbackWorkspace,
    routes,
    location,
  };
};

/** Read the run's index of VirtualModels and agent configs; null when the job saved none. */
export const fetchSwitchyardResults = async (
  workspace: string,
  jobName: string,
  signal?: AbortSignal
): Promise<SwitchyardResults | null> => {
  const located = await locateJobResultFiles(workspace, jobName, [SWITCHYARD_INDEX_FILE], signal);
  const indexPath = located?.paths[SWITCHYARD_INDEX_FILE];
  if (!located || !indexPath) return null;

  const location = { workspace: located.workspace, fileset: located.fileset };
  const text = await downloadText(location.workspace, location.fileset, indexPath, signal);
  return text ? parseSwitchyardIndex(text, indexPath, location, workspace) : null;
};

/** The rewritten agent config for one route, pointed at its VirtualModel. */
export const fetchRouteAgentConfig = async (
  location: SwitchyardResults['location'],
  path: string,
  signal?: AbortSignal
): Promise<Record<string, unknown>> => {
  const text = await downloadText(location.workspace, location.fileset, path, signal);
  if (!text) throw new Error(`The agent config "${path}" is empty.`);
  const parsed: unknown = YAML.parse(text);
  if (!isPlainObject(parsed)) throw new Error(`The agent config "${path}" is not a YAML mapping.`);
  return parsed;
};

/** The value of the one tuning knob each routing strategy takes, from its middleware config. */
export const routeThreshold = (route: SwitchyardRoute): number | undefined => {
  const field =
    route.configType === 'random_routing'
      ? 'strong_probability'
      : route.configType === 'stage_router'
        ? 'confidence_threshold'
        : 'base_threshold';
  const value = route.config[field];
  return typeof value === 'number' ? value : undefined;
};
