// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { datasetFileContentQueryOptions } from '@studio/api/datasets/useDatasetFileContent';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { parse as parseYaml } from 'yaml';

// Matches nemo-environment.yaml at the fileset root or one directory deep
// (one level of depth covers the upload prefix added by `nemo files upload <dir>`).
const NEMO_ENV_YAML_RE = /^(?:[^/]+\/)?nemo-environment\.yaml$/;

// The backend resolves wheels/ and config_paths against the package root — the
// directory holding nemo-environment.yaml — so these match package-relative paths.
const WHEEL_RE = /^wheels\/[^/]+\.whl$/;
const WHEELS_DIR_RE = /^wheels\//;

interface NemoEnvironmentYaml {
  format?: string;
  config_paths?: unknown;
  adapter?: { agent?: string };
  metadata?: {
    name?: string;
    description?: string;
    hub_id?: string;
    vf_env_id?: string;
  };
}

/** Mirrors `EnvironmentFormat` in services/rl/src/nhx/rl/schemas/environment.py. */
const ENVIRONMENT_FORMATS = ['native-v1', 'wheels-v1', 'adapter-wheels-v1'] as const;

type EnvironmentFormat = (typeof ENVIRONMENT_FORMATS)[number];

const isKnownFormat = (value: string | undefined): value is EnvironmentFormat =>
  !!value && (ENVIRONMENT_FORMATS as readonly string[]).includes(value);

/** Per-format `config_paths` prefix rules enforced by the manifest validators. */
const CONFIG_PATH_PREFIXES: Record<string, readonly string[]> = {
  'native-v1': ['responses_api_agents/', 'resources_servers/', 'responses_api_models/'],
  'adapter-wheels-v1': ['configs/'],
};

export interface GymEnvironmentManifest {
  /** Value of the `format` field in nemo-environment.yaml (e.g. "adapter-wheels-v1"). */
  format: string;
  envName: string;
  description?: string;
  hubId?: string;
  vfEnvId?: string;
  wheelCount: number;
}

export interface UseGymEnvironmentManifestResult {
  isPending: boolean;
  error: Error | null;
  manifest: GymEnvironmentManifest | null;
  fileCount: number;
  totalSize: number;
  /** True when the fileset has files but no nemo-environment.yaml — backend will reject it. */
  noConfigWarning: boolean;
  /**
   * Manifest problems that the backend's schema would reject at training time.
   * Mirrors the pydantic validators in nhx.rl.schemas.environment so a broken
   * package is visible at selection rather than after the job is queued.
   */
  manifestIssues: string[];
}

/**
 * Re-implements the manifest and package-layout validators from
 * services/rl/src/nhx/rl/schemas/environment.py and tasks/environment/validate.py.
 * `packagePaths` are relative to the package root (the directory holding
 * nemo-environment.yaml), which is the `env_root` the backend resolves against.
 *
 * The file-existence checks mirror `validate_package_layout`, which runs inside the
 * training job — so unlike the schema rules, they are not caught at submit.
 */
const collectManifestIssues = (yaml: NemoEnvironmentYaml, packagePaths: string[]): string[] => {
  const issues: string[] = [];

  const format = yaml.format;
  if (!isKnownFormat(format)) {
    issues.push(
      `format is ${format ? `"${format}"` : 'missing'} — must be one of ${ENVIRONMENT_FORMATS.join(', ')}.`
    );
  }

  if (!yaml.metadata?.name?.trim()) {
    issues.push('metadata.name is required.');
  }

  const configPaths = yaml.config_paths;
  if (!Array.isArray(configPaths) || configPaths.length === 0) {
    issues.push('config_paths is required and must list at least one entry.');
  } else {
    const paths = configPaths.filter((p): p is string => typeof p === 'string');
    const prefixes = format ? CONFIG_PATH_PREFIXES[format] : undefined;
    for (const p of paths) {
      if (!p || p.startsWith('/') || p.startsWith('\\') || p.split('/').includes('..')) {
        issues.push(`config_paths entry must be relative and contained: "${p}".`);
        continue;
      }
      if (prefixes && !prefixes.some((prefix) => p.startsWith(prefix))) {
        issues.push(`${format} config_paths must be under ${prefixes.join(', ')}: "${p}".`);
      }
      if (!packagePaths.includes(p)) {
        issues.push(`config_paths entry "${p}" is not present in this fileset.`);
      }
    }
  }

  // Prompt data ships as the dataset fileset; the package validator rejects any
  // *.jsonl found anywhere under the environment root.
  const strayJsonl = packagePaths.filter((p) => p.endsWith('.jsonl'));
  if (strayJsonl.length > 0) {
    issues.push(
      `Prompt JSONL must not live in the environment package (found ${strayJsonl[0]}). Upload it as the dataset instead.`
    );
  }

  if (format === 'adapter-wheels-v1' && !yaml.adapter?.agent?.trim()) {
    issues.push('adapter.agent is required for adapter-wheels-v1.');
  }

  if (format === 'wheels-v1' || format === 'adapter-wheels-v1') {
    const wheelsDirFiles = packagePaths.filter((p) => WHEELS_DIR_RE.test(p));
    const nonWheels = wheelsDirFiles.filter((p) => !WHEEL_RE.test(p));
    if (wheelsDirFiles.length === 0) {
      issues.push(`${format} requires a non-empty wheels/ directory at the package root.`);
    } else if (nonWheels.length > 0) {
      issues.push(`wheels/ must contain only .whl files (found ${nonWheels[0]}).`);
    }
  }

  return issues;
};

interface UseGymEnvironmentManifestOptions {
  workspace: string;
  filesetName: string;
}

export const useGymEnvironmentManifest = ({
  workspace,
  filesetName,
}: UseGymEnvironmentManifestOptions): UseGymEnvironmentManifestResult => {
  const enabled = !!(workspace && filesetName);
  const queryClient = useQueryClient();

  const {
    data: filesResponse,
    isPending: isFilesPending,
    error: filesError,
  } = useFilesListFilesetFiles(workspace, filesetName, undefined, {
    query: { enabled },
  });

  const allFiles = filesResponse?.data ?? [];
  const manifestFile = allFiles.find((f) => NEMO_ENV_YAML_RE.test(f.path)) ?? null;

  const {
    data: fileContent,
    isPending: isContentPending,
    error: contentError,
  } = useQuery({
    enabled: enabled && !!manifestFile,
    queryKey: ['gym-environment-manifest', workspace, filesetName, manifestFile?.path ?? ''],
    queryFn: async () => {
      if (!manifestFile) return null;
      return queryClient.ensureQueryData(
        datasetFileContentQueryOptions({ workspace, name: filesetName, path: manifestFile.path })
      );
    },
  });

  const totalSize = allFiles.reduce((sum, f) => sum + f.size, 0);

  // The manifest may sit one directory deep (upload prefix). The backend treats its
  // directory as `env_root` and resolves config_paths and wheels/ against it, so
  // strip the prefix once and run every layout check on package-relative paths.
  const packageRoot = manifestFile?.path.replace(/[^/]+$/, '') ?? '';
  const packagePaths = manifestFile
    ? allFiles
        .map((f) => f.path)
        .filter((p) => p.startsWith(packageRoot))
        .map((p) => p.slice(packageRoot.length))
    : [];
  const wheelCount = packagePaths.filter((p) => WHEEL_RE.test(p)).length;

  const { manifest, manifestIssues } = ((): {
    manifest: GymEnvironmentManifest | null;
    manifestIssues: string[];
  } => {
    if (!manifestFile || fileContent == null) return { manifest: null, manifestIssues: [] };
    let parsed: unknown;
    try {
      parsed = parseYaml(fileContent);
    } catch {
      return {
        manifest: null,
        manifestIssues: ['nemo-environment.yaml is not valid YAML.'],
      };
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {
        manifest: null,
        manifestIssues: ['nemo-environment.yaml must contain a YAML mapping.'],
      };
    }
    const yaml = parsed as NemoEnvironmentYaml;
    const meta = yaml.metadata ?? {};
    return {
      manifest: {
        format: yaml.format ?? 'unknown',
        envName: meta.name ?? 'unknown',
        description: meta.description || undefined,
        hubId: meta.hub_id || undefined,
        vfEnvId: meta.vf_env_id || undefined,
        wheelCount,
      },
      manifestIssues: collectManifestIssues(yaml, packagePaths),
    };
  })();

  return {
    isPending: isFilesPending || (enabled && !!manifestFile && isContentPending),
    error: ((filesError ?? contentError) as Error | null) ?? null,
    manifest,
    fileCount: allFiles.length,
    totalSize,
    noConfigWarning: !isFilesPending && enabled && !manifestFile && allFiles.length > 0,
    manifestIssues,
  };
};
