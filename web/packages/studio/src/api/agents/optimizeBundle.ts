// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import YAML from 'yaml';

// Mirrors nemo_optimization.bundle.preflight_bundle, which `nemo agents optimize prepare-fileset` runs.

const FABRIC_AGENT_SCHEMA_VERSION = 'fabric.agent/v1alpha1';

const NAT_TOP_LEVEL_KEYS = [
  'workflow',
  'llms',
  'functions',
  'function_groups',
  'embedders',
  'general',
];

const FILESET_REF_PATTERN = /^[\w\-.]+\/[\w\-.]+#[\w\-./]+$/;

const YAML_EXTENSIONS = ['.yaml', '.yml'];

type ConfigMapping = Record<string, unknown>;

interface PathReference {
  location: string;
  value: string;
  mustExist: boolean;
  isDir: boolean;
}

const isMapping = (value: unknown): value is ConfigMapping =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const isFilesetRelative = (path: string): boolean =>
  !path.startsWith('~') &&
  !path.startsWith('/') &&
  !path.startsWith('\\') &&
  !/^[A-Za-z]:/.test(path) &&
  !path.split(/[\\/]/).includes('..');

const normalizeBundlePath = (path: string): string =>
  path
    .split('/')
    .filter((segment) => segment !== '' && segment !== '.')
    .join('/');

export const isYamlPath = (path: string): boolean =>
  YAML_EXTENSIONS.some((extension) => path.toLowerCase().endsWith(extension));

/** A parsed mapping, or undefined when the text is not YAML or not a mapping at the top level. */
export const parseOptimizeConfig = (text: string): ConfigMapping | undefined => {
  try {
    const parsed: unknown = YAML.parse(text);
    return isMapping(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
};

export const looksLikeOptimizeConfig = (config: ConfigMapping | undefined): boolean =>
  isMapping(config?.optimizer);

const agentProblems = (config: ConfigMapping, agent: string | undefined): string[] => {
  if (config.schema_version === FABRIC_AGENT_SCHEMA_VERSION || agent) return [];
  if (NAT_TOP_LEVEL_KEYS.some((key) => key in config)) {
    return [
      `the config looks like legacy NAT workflow YAML; optimize requires a Fabric-native package (schema_version: ${FABRIC_AGENT_SCHEMA_VERSION})`,
    ];
  }
  return [
    `no Agent under Test: the config declares no schema_version: ${FABRIC_AGENT_SCHEMA_VERSION} package, and no agent was given`,
  ];
};

const optimizerProblems = (config: ConfigMapping): string[] => {
  const optimizer = config.optimizer;
  if (!isMapping(optimizer)) return ['optimizer section is missing or is not a mapping'];

  const enabled = ['numeric', 'prompt'].filter((name) => {
    const section = optimizer[name];
    return isMapping(section) && Boolean(section.enabled);
  });

  const problems: string[] = [];
  if (enabled.length === 0) {
    problems.push(
      'no optimizer is enabled; set optimizer.numeric.enabled or optimizer.prompt.enabled'
    );
  }
  const searchSpace = optimizer.search_space;
  const hasSearchSpace = isMapping(searchSpace)
    ? Object.keys(searchSpace).length > 0
    : Boolean(searchSpace);
  if (enabled.includes('numeric') && !hasSearchSpace) {
    problems.push('optimizer.numeric is enabled but optimizer.search_space is empty');
  }
  return problems;
};

const optional = (
  value: unknown,
  location: string,
  { mustExist, isDir = false }: { mustExist: boolean; isDir?: boolean }
): PathReference[] =>
  typeof value === 'string' && value ? [{ location, value, mustExist, isDir }] : [];

const looksLikeBundleScript = (value: string): boolean => {
  if (!value || value.startsWith('-') || value.includes('://') || value.startsWith('/')) {
    return false;
  }
  return (
    value.includes('/') || ['.py', '.js', '.sh'].some((extension) => value.endsWith(extension))
  );
};

const mcpServerReferences = (config: ConfigMapping): PathReference[] => {
  const mcp = config.mcp;
  const servers = isMapping(mcp) ? mcp.servers : undefined;
  if (!isMapping(servers)) return [];

  return Object.entries(servers).flatMap(([name, server]) => {
    if (!isMapping(server) || server.transport !== 'stdio') return [];
    const references: PathReference[] = [];
    const { url, args } = server;
    if (typeof url === 'string' && looksLikeBundleScript(url)) {
      references.push(...optional(url, `mcp.servers.${name}.url`, { mustExist: true }));
    }
    (Array.isArray(args) ? args : []).forEach((arg: unknown, index) => {
      if (typeof arg === 'string' && looksLikeBundleScript(arg)) {
        references.push(
          ...optional(arg, `mcp.servers.${name}.args[${index}]`, { mustExist: true })
        );
      }
    });
    return references;
  });
};

const datasetReference = (dataset: unknown): PathReference[] => {
  let value = typeof dataset === 'string' ? dataset : undefined;
  if (isMapping(dataset)) {
    const candidate = dataset.file_path || dataset.path;
    value = typeof candidate === 'string' ? candidate : undefined;
  }
  // A `workspace/fileset#path` dataset is staged separately by the job at run time.
  if (value === undefined || FILESET_REF_PATTERN.test(value)) return [];
  return [{ location: 'eval.general.dataset', value, mustExist: true, isDir: false }];
};

const pathReferences = (config: ConfigMapping): PathReference[] => {
  const references: PathReference[] = [];

  const { runtime, environment } = config;
  if (isMapping(runtime)) {
    references.push(
      ...optional(runtime.artifacts, 'runtime.artifacts', { mustExist: false, isDir: true })
    );
  }
  if (isMapping(environment)) {
    references.push(
      ...optional(environment.workspace, 'environment.workspace', {
        mustExist: false,
        isDir: true,
      }),
      ...optional(environment.artifacts, 'environment.artifacts', { mustExist: false, isDir: true })
    );
  }

  references.push(...mcpServerReferences(config));

  const evalConfig = config.eval;
  if (!isMapping(evalConfig)) return references;

  if (isMapping(evalConfig.general)) {
    references.push(...datasetReference(evalConfig.general.dataset));
  }
  if (isMapping(evalConfig.fabric)) {
    references.push(
      ...optional(evalConfig.fabric.base_dir, 'eval.fabric.base_dir', {
        mustExist: true,
        isDir: true,
      })
    );
  }
  return references;
};

const pathProblems = (config: ConfigMapping, bundlePaths: ReadonlySet<string>): string[] => {
  const directoryExists = (path: string): boolean =>
    path === ''
      ? bundlePaths.size > 0
      : [...bundlePaths].some((file) => file.startsWith(`${path}/`));

  return pathReferences(config).flatMap((reference) => {
    if (!isFilesetRelative(reference.value)) {
      return [
        `${reference.location} is an absolute path (${JSON.stringify(reference.value)}), which will not exist on the platform; make it relative to the bundle root`,
      ];
    }
    if (reference.value.includes('${') || !reference.mustExist) return [];

    const target = normalizeBundlePath(reference.value);
    if (reference.isDir && !directoryExists(target)) {
      return [
        `${reference.location} points at ${JSON.stringify(reference.value)}, which is not a directory in the bundle`,
      ];
    }
    if (!reference.isDir && !bundlePaths.has(target)) {
      return [
        `${reference.location} points at ${JSON.stringify(reference.value)}, which is not a file in the bundle`,
      ];
    }
    return [];
  });
};

/** Every problem the CLI's preflight would report, except symlinks, which a browser resolves before handing files over. */
export const optimizeBundleProblems = ({
  config,
  bundlePaths,
  agent,
}: {
  config: ConfigMapping;
  bundlePaths: ReadonlySet<string>;
  agent?: string;
}): string[] => [
  ...agentProblems(config, agent),
  ...optimizerProblems(config),
  ...pathProblems(config, bundlePaths),
];
