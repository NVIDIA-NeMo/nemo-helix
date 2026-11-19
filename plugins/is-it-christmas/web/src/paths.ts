// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const PLUGIN_NAME = 'is-it-christmas';

export const pluginPath = (workspaceId: string): string => `/workspaces/${workspaceId}/plugin/${PLUGIN_NAME}`;
