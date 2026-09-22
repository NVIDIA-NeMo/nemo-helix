// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NHX_BASE_URL } from '@e2e-tests/utils/environment';
import { Workspace, WorkspaceInput, WorkspacesPage } from '@nemo/sdk/generated/platform/schema';
import { APIRequestContext } from '@playwright/test';

export class WorkspacesAPI {
  constructor(private request: APIRequestContext) {}

  async createWorkspace(data: WorkspaceInput) {
    const response = await this.request.post(`${NHX_BASE_URL}/apis/entities/v2/workspaces`, {
      data,
    });
    return (await response.json()) as Workspace;
  }

  async deleteWorkspace(name: string) {
    await this.request.delete(`${NHX_BASE_URL}/apis/entities/v2/workspaces/${name}`);
  }

  async listWorkspaces(page = 1) {
    const response = await this.request.get(
      `${NHX_BASE_URL}/apis/entities/v2/workspaces?page=${page}&page_size=100`
    );
    return (await response.json()) as WorkspacesPage;
  }

  async deleteAllWorkspacesByPrefix(prefix: string) {
    // Collect matches across every page before deleting. Deleting mutates the collection,
    // which would shift later pages and skip entries if we deleted mid-pagination, so we
    // page through the full list first (following pagination.total_pages) and delete after.
    const namesToDelete: string[] = [];
    let page = 1;
    let totalPages = 1;
    do {
      const list = await this.listWorkspaces(page);
      for (const workspace of list.data) {
        if (workspace.name.startsWith(prefix)) {
          namesToDelete.push(workspace.name);
        }
      }
      totalPages = list.pagination?.total_pages ?? 1;
      page += 1;
    } while (page <= totalPages);

    for (const name of namesToDelete) {
      await this.deleteWorkspace(name);
    }
  }
}
