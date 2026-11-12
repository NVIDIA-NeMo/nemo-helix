// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  HuggingfaceStorageConfig,
  NGCStorageConfig,
} from '@nemo/sdk/generated/platform/schema';

/** Normalized input for building a storage config from a URL. */
export interface StorageConfigInput {
  /** Full URL to the resource (Hugging Face Hub or NGC catalog). */
  url: string;
  /**
   * Name of the platform secret holding the API token/key.
   * Required for NGC; optional for Hugging Face (public repos).
   */
  secretKey?: string;
}

export const HF_DEFAULT_ENDPOINT = 'https://huggingface.co';
export const NGC_DEFAULT_HOST = 'https://api.ngc.nvidia.com';

/**
 * NGC catalog URL pattern: /orgs/{org}/teams/{team}/resources/{target}
 * e.g. https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ngc-apps/resources/ngc_cli
 */
const NGC_ORG_TEAM_RESOURCE = /^\/orgs\/([^/]+)\/teams\/([^/]+)\/resources\/([^/]+)/;

/**
 * Hugging Face URL path can be:
 * - /datasets/username/dataset-name
 * - /username/model-name (models)
 * - /spaces/username/space-name
 * We need at least two path segments; optional prefix (datasets|models|spaces)/ for repo_id (org/repo).
 */
const HF_REPO_SEGMENTS = /^\/(?:(?:datasets|models|spaces)\/)?([^/]+)\/([^/]+)/;

export function isNgcUrl(url: URL): boolean {
  const host = url.hostname.toLowerCase();
  return (
    host === 'catalog.ngc.nvidia.com' ||
    host === 'api.ngc.nvidia.com' ||
    host.endsWith('.ngc.nvidia.com')
  );
}

export function isHuggingFaceUrl(url: URL): boolean {
  const host = url.hostname.toLowerCase();
  return (
    host === 'huggingface.co' || host === 'www.huggingface.co' || host.endsWith('.huggingface.co')
  );
}

export function parseNgcUrl(url: URL): { org: string; team: string; target: string } {
  const path = url.pathname;
  const match = path.match(NGC_ORG_TEAM_RESOURCE);
  if (!match) {
    throw new Error(
      `Invalid NGC URL: expected path like /orgs/{org}/teams/{team}/resources/{target}, got ${path}`
    );
  }
  const [, org, team, target] = match;
  if (!org || !team || !target) {
    throw new Error(`Invalid NGC URL: could not parse org, team, target from ${path}`);
  }
  return { org, team, target };
}

export function parseHuggingFaceUrl(url: URL): {
  repoId: string;
  repoType: 'dataset' | 'model' | 'space';
} {
  const path = url.pathname.replace(/\/$/, '');
  const match = path.match(HF_REPO_SEGMENTS);
  if (!match) {
    throw new Error(
      `Invalid Hugging Face URL: expected path like /datasets/org/repo or /org/repo, got ${path}`
    );
  }
  const [, org, repo] = match;
  if (!org || !repo) {
    throw new Error(`Invalid Hugging Face URL: could not parse org/repo from ${path}`);
  }
  const repoId = `${org}/${repo}`;
  let repoType: 'dataset' | 'model' | 'space' = 'model';
  if (path.startsWith('/datasets/')) repoType = 'dataset';
  else if (path.startsWith('/spaces/')) repoType = 'space';
  return { repoId, repoType };
}

/**
 * Build a storage config (Hugging Face or NGC) from a normalized URL and optional secret.
 *
 * - Hugging Face: pass a Hub URL (e.g. https://huggingface.co/datasets/org/repo).
 *   repo_id, repo_type, and endpoint are derived from the URL; secretKey is optional (for private repos).
 * - NGC: pass a catalog URL (e.g. https://catalog.ngc.nvidia.com/orgs/org/teams/team/resources/target).
 *   org, team, and target are derived; secretKey is required (api_key_secret).
 *
 * @param input - { url, secretKey? }
 * @returns HuggingfaceStorageConfig or NGCStorageConfig
 * @throws Error if the URL is not recognized or cannot be parsed
 */
export function storageConfigFromUrl(
  input: StorageConfigInput
): HuggingfaceStorageConfig | NGCStorageConfig {
  const { url, secretKey } = input;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error(`Invalid storage URL: ${url}`);
  }

  if (isNgcUrl(parsed)) {
    const { org, team, target } = parseNgcUrl(parsed);
    if (!secretKey) {
      throw new Error('secretKey is required for NGC storage config (api_key_secret)');
    }
    const config: NGCStorageConfig = {
      type: 'ngc',
      org,
      team,
      target,
      api_key_secret: secretKey,
      host: NGC_DEFAULT_HOST,
    };
    return config;
  }

  if (isHuggingFaceUrl(parsed)) {
    const { repoId, repoType } = parseHuggingFaceUrl(parsed);
    const origin = parsed.origin || HF_DEFAULT_ENDPOINT;
    const config: HuggingfaceStorageConfig = {
      type: 'huggingface',
      repo_id: repoId,
      repo_type: repoType,
      endpoint: origin,
    };
    if (secretKey) {
      config.token_secret = secretKey;
    }
    return config;
  }

  throw new Error(
    `Unsupported storage URL: ${url}. Use a Hugging Face Hub URL (e.g. https://huggingface.co/datasets/org/repo) or NGC catalog URL (e.g. https://catalog.ngc.nvidia.com/orgs/org/teams/team/resources/target).`
  );
}
