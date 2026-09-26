// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Links must be absolute: Studio mounts plugins at a splat route, and React
// Router resolves a relative `to` against the full matched path, so it appends.
export const pluginPath = (workspaceId: string, page: string): string =>
  `/workspaces/${workspaceId}/plugin/kern/${page}`;

// Studio's intake/traces page — used by the Decisions pane.
export const intakeTracesPath = (workspaceId: string): string =>
  `/workspaces/${workspaceId}/intake/traces`;
