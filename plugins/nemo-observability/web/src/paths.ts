// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Links must be absolute: Studio mounts plugins at a splat route, and React
// Router resolves a relative `to` against the full matched path, so it appends.
export const pluginPath = (workspaceId: string, connection = ''): string =>
  `/workspaces/${workspaceId}/plugin/observability/${connection}`;
