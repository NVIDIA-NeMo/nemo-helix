// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Links must be absolute: Studio mounts plugins at a splat route, and React
// Router resolves a relative `to` against the full matched path, so it appends.
// Route `path`s below are relative — only `to` / `href` need the prefix.

const base = (workspace: string): string => `/workspaces/${workspace}/plugin/agent-hardener`;

/** Route `path` values for the plugin's own <Routes>, relative to the mount. */
export const ROUTE_PATHS = {
  runList: '',
  runDetails: ':agentHardenerRunName',
  manifestList: 'manifests',
  manifestNew: 'manifests/new',
  manifestDetail: 'manifests/:agentHardenerManifestName',
} as const;

export const getAgentHardenerRunListRoute = (workspace: string): string => base(workspace);

export const getAgentHardenerRunDetailsRoute = (workspace: string, agentHardenerRunName: string): string =>
  `${base(workspace)}/${encodeURIComponent(agentHardenerRunName)}`;

export const getAgentHardenerManifestListRoute = (workspace: string): string =>
  `${base(workspace)}/manifests`;

export const getNewAgentHardenerManifestRoute = (workspace: string): string =>
  `${base(workspace)}/manifests/new`;

export const getAgentHardenerManifestDetailRoute = (
  workspace: string,
  agentHardenerManifestName: string
): string => `${base(workspace)}/manifests/${encodeURIComponent(agentHardenerManifestName)}`;

/** Studio routes the plugin links out to. */
export const getSecretsRoute = (workspace: string): string => `/workspaces/${workspace}/secrets`;
