// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Links must be absolute: Studio mounts plugins at a splat route, and React
// Router resolves a relative `to` against the full matched path, so it appends.
const base = (workspace: string): string => `/workspaces/${workspace}/plugin/user-sim`;

export const getRunListRoute = base;

export const getRunRoute = (workspace: string, job: string): string =>
  `${base(workspace)}/${encodeURIComponent(job)}`;

export const getIntakeSessionRoute = (workspace: string, sessionId: string): string =>
  `/workspaces/${workspace}/intake/sessions/${encodeURIComponent(sessionId)}`;
