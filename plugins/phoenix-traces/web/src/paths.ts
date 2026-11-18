// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const pluginPath = (workspaceId: string, page: string): string =>
  `/workspaces/${workspaceId}/plugin/phoenix/${page}`;
