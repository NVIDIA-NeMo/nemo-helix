// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Project, ProjectsPage } from '@nemo/sdk/generated/platform/schema';

export const workspace1: Project = {
  id: 'project-EE9fhQtuH6UHEdFrma1sZE',
  workspace: 'default',
  name: 'project-EE9fhQtuH6UHEdFrma1sZE',
  description: 'Test Workspace 1 Description',
  created_at: '2024-08-09T16:50:52.302Z',
  updated_at: '2024-08-09T16:52:57.693Z',
};

export const project2: Project = {
  id: 'project-4szG6MFC3HKhGJTQ9vAJax',
  workspace: 'default',
  name: 'project-4szG6MFC3HKhGJTQ9vAJax',
  description: 'Test Workspace 2 Description',
  created_at: '2024-08-09T16:53:41.521Z',
  updated_at: '2024-08-09T16:53:41.521Z',
};

export const project3: Project = {
  id: 'project-7PDqvt4UhYMTBGSgh7hH8m',
  workspace: 'default',
  name: 'project-7PDqvt4UhYMTBGSgh7hH8m',
  description: 'Test Workspace 3 Description',
  created_at: '2024-08-09T16:54:00.591Z',
  updated_at: '2024-08-09T16:54:00.591Z',
};

export const project4: Project = {
  id: 'project-651f1bd6df478959cdae8a',
  workspace: 'default',
  name: 'project-651f1bd6df478959cdae8a',
  description: 'Always throws a server error when trying to update',
  created_at: '2023-10-06T20:15:01.000Z',
  updated_at: '2023-10-06T20:15:01.000Z',
};

export const projectOtherNamespace: Project = {
  id: 'project-abc123xyz',
  workspace: 'other-user',
  name: 'project-abc123xyz',
  description: 'Workspace from a different namespace',
  created_at: '2024-08-10T10:00:00.000Z',
  updated_at: '2024-08-10T10:00:00.000Z',
};

/** Workspace used in Playground tests so dropdown has distinct groups: default (base) + test (prompt/custom). */
export const testWorkspace = 'test';

export const testProject: Project = {
  id: 'project-test-workspace-playground',
  workspace: testWorkspace,
  name: 'project-test-workspace-playground',
  description: 'Test project for Playground route (workspace test)',
  created_at: '2024-08-09T16:50:52.302Z',
  updated_at: '2024-08-09T16:52:57.693Z',
};

export const projects = [workspace1, project2, project3, project4, testProject];

export const projectsPage: ProjectsPage = {
  data: projects,
  pagination: {
    page: 1,
    page_size: 100,
    current_page_size: projects.length,
    total_pages: 1,
    total_results: projects.length,
  },
  sort: '-created_at',
};

/** Projects page filtered by workspace (for handlers that need per-workspace list). */
export const getProjectsPageForWorkspace = (workspace: string): ProjectsPage => {
  const filtered = projects.filter((p) => p.workspace === workspace);
  const count = filtered.length;
  return {
    ...projectsPage,
    data: filtered,
    pagination: {
      page: 1,
      page_size: 100,
      current_page_size: count,
      total_pages: 1,
      total_results: count,
    },
  };
};
