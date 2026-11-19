// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { listProjects, listSpans, type ApiHost } from './api';

// react-query is a Studio singleton, so these share Studio's QueryClient and cache.
export const useProjects = (host: ApiHost) =>
  useQuery({
    queryKey: ['phoenix', host.apiBaseUrl, host.workspaceId, 'projects'],
    queryFn: ({ signal }) => listProjects(host, signal),
  });

export const useSpans = (host: ApiHost, project: string | undefined) =>
  useQuery({
    queryKey: ['phoenix', host.apiBaseUrl, host.workspaceId, 'spans', project],
    queryFn: ({ signal }) => listSpans(host, project as string, signal),
    enabled: project !== undefined,
  });
