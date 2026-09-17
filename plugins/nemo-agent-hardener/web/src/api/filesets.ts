// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { customFetch } from '@agent-hardener/api/fetcher';
import { usePlatformSdk } from '@agent-hardener/api/platform';
import type { PluginSdk } from '@agent-hardener/types';
import { useMutation } from '@tanstack/react-query';

export interface UploadFilesetParams {
  workspace: string;
  /** Manifest id, used to derive a recognizable fileset name. */
  manifestName: string;
  /** The single file to store. */
  file: File;
}

/**
 * Create a generic fileset and upload a single file into it; return its `workspace/name` ref.
 *
 * The Agent Hardener plugin re-downloads the fileset on the job host (hitlogs when it replays recorded
 * attacks; benign suites when it validates).
 */
async function uploadToFileset(
  platform: PluginSdk['platform'],
  { workspace, manifestName, file }: UploadFilesetParams,
  kind: string,
  fallbackType: string
): Promise<string> {
  const name = `${manifestName}-${kind}-${Date.now().toString(36)}`;
  const fileset = await platform.filesCreateFileset(workspace, { name, purpose: 'generic' });
  const blob = new Blob([await file.arrayBuffer()], { type: file.type || fallbackType });
  await platform.filesUploadFile(fileset.workspace, fileset.name, file.name, blob);
  return `${fileset.workspace}/${fileset.name}`;
}

/** Store an uploaded garak hitlog (.jsonl); the ref feeds a replay-mode war-game via `--replay`. */
export const useUploadHitlogFileset = () => {
  const platform = usePlatformSdk();
  return useMutation({
    mutationFn: (params: UploadFilesetParams) =>
      uploadToFileset(platform, params, 'hitlog', 'application/jsonl'),
  });
};

/** Store an uploaded benign suite (requests.csv); the ref overrides the manifest suite via `--benign-suite`. */
export const useUploadBenignSuiteFileset = () => {
  const platform = usePlatformSdk();
  return useMutation({
    mutationFn: (params: UploadFilesetParams) =>
      uploadToFileset(platform, params, 'benign-suite', 'text/csv'),
  });
};

/**
 * Store an uploaded project bundle (.zip); the ref becomes the manifest's `project_fileset`.
 *
 * The bundle *is* the frozen target for a project manifest — there is no registered agent behind it
 * to re-resolve against — so this upload is the target, not a cache of one.
 */
export const useUploadProjectFileset = () => {
  const platform = usePlatformSdk();
  return useMutation({
    mutationFn: (params: UploadFilesetParams) =>
      uploadToFileset(platform, params, 'project', 'application/zip'),
  });
};

/** What an uploaded project states about itself, and what it cannot. */
export interface InspectProjectResult {
  dockerfile: string;
  dockerfiles: string[];
  start_command: string;
  binaries: string[];
  port: number;
  secrets: string[];
  egress: string[];
  env: Record<string, string>;
  /** Fields the project cannot state; the form must ask for exactly these. */
  unresolved: string[];
  warnings: string[];
}

/**
 * Read an uploaded project bundle so the form can pre-fill what is derivable.
 *
 * The point of `unresolved` is that the user never writes a manifest: everything the Dockerfile
 * states is filled in, and only the rest is asked for.
 */
export const useInspectProject = () =>
  useMutation({
    mutationFn: ({
      workspace,
      projectFileset,
      dockerfile,
    }: {
      workspace: string;
      projectFileset: string;
      dockerfile?: string;
    }) =>
      customFetch<InspectProjectResult>({
        url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(workspace)}/manifests/inspect-project`,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        data: { project_fileset: projectFileset, ...(dockerfile ? { dockerfile } : {}) },
      }),
  });

/** Auto-derived defaults for the deployed-agent create form (victim port + secret names). */
export interface InspectAgentResult {
  agent: string;
  port: number;
  secrets: string[];
  egress: string[];
  warnings: string[];
}

/**
 * Derive a deployed agent's victim port + secret names (read-only) to pre-fill the create form.
 *
 * Not in the generated SDK; calls the plugin endpoint via the SDK's fetcher so auth/base-url match.
 */
export const useInspectAgent = () =>
  useMutation({
    mutationFn: ({ workspace, agent }: { workspace: string; agent: string }) =>
      customFetch<InspectAgentResult>({
        url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(workspace)}/manifests/inspect-agent`,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        data: { agent },
      }),
  });
