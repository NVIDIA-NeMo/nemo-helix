// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { listProjects, listSpans } from './api';
import type { PluginHost } from './types';

// react-query is a Studio singleton, so these share Studio's QueryClient and cache.
export const useProjects = (host: PluginHost) =>
  useQuery({ queryKey: ['phoenix', 'projects'], queryFn: () => listProjects(host) });

export const useSpans = (host: PluginHost, project: string | undefined) =>
  useQuery({
    queryKey: ['phoenix', 'spans', project],
    queryFn: () => listSpans(host, project as string),
    enabled: project !== undefined,
  });
