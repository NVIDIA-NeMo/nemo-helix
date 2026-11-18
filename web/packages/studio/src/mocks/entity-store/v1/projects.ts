// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** V1 project shape (namespace-based) for MSW handlers. */
interface ProjectV1 {
  namespace: string;
  name: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
}

const project1: ProjectV1 = {
  namespace: 'default',
  name: 'project-EE9fhQtuH6UHEdFrma1sZE',
  description: 'Test Workspace 1 Description',
  created_at: '2024-08-09T16:50:52.302Z',
  updated_at: '2024-08-09T16:52:57.693Z',
};

const project2: ProjectV1 = {
  namespace: 'default',
  name: 'project-4szG6MFC3HKhGJTQ9vAJax',
  description: 'Test Workspace 2 Description',
  created_at: '2024-08-09T16:53:41.521Z',
  updated_at: '2024-08-09T16:53:41.521Z',
};

const project3: ProjectV1 = {
  namespace: 'default',
  name: 'project-7PDqvt4UhYMTBGSgh7hH8m',
  description: 'Test Workspace 3 Description',
  created_at: '2024-08-09T16:54:00.591Z',
  updated_at: '2024-08-09T16:54:00.591Z',
};

const project4: ProjectV1 = {
  namespace: 'default',
  name: 'project-651f1bd6df478959cdae8a',
  description: 'Always throws a server error when trying to update',
  created_at: '2023-10-06T20:15:01.000Z',
  updated_at: '2023-10-06T20:15:01.000Z',
};

export const projects = [project1, project2, project3, project4];
