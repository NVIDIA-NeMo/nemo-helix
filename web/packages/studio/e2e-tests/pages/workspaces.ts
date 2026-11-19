// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expectToastIsVisible, waitForLongOperation } from '@e2e-tests/utils/pageUtils';
import { expect, type Page } from '@playwright/test';

const DEFAULT_WORKSPACE_DASHBOARD_URL = 'workspaces/default/dashboard';

export class WorkspacesPage {
  constructor(public readonly page: Page) {}

  async goto() {
    await this.page.goto(DEFAULT_WORKSPACE_DASHBOARD_URL);
  }

  async gotoSettings(workspaceName: string) {
    await this.page.goto(`workspaces/${workspaceName}/settings`);
  }

  async createWorkspace(name: string, description: string) {
    await this.page.getByRole('button', { name: 'Select workspace' }).click();
    await this.page.getByText('New Workspace').click();
    const createDialog = this.page.getByRole('dialog', { name: 'New Workspace' });
    await expect(createDialog).toBeVisible();
    await createDialog.getByRole('textbox', { name: 'Name' }).fill(name);
    await createDialog.getByRole('textbox', { name: 'Description' }).fill(description);
    await createDialog.getByRole('button', { name: 'Create', exact: true }).click();
  }

  async editWorkspaceDescription(updatedDescription: string) {
    await this.page.getByRole('button', { name: 'Edit Description' }).click();
    await expect(this.page.getByRole('dialog', { name: 'Edit Description' })).toBeVisible();
    await this.page.getByRole('textbox', { name: 'Description' }).fill(updatedDescription);
    await this.page.getByRole('button', { name: 'Save' }).click();
    await waitForLongOperation(this.page);
    await expectToastIsVisible(this.page, 'Successfully updated workspace!');
  }

  async deleteWorkspace(workspaceName: string) {
    await this.page.getByRole('button', { name: 'Delete Workspace' }).click();
    const deleteDialog = this.page.getByRole('dialog', {
      name: `Delete workspace '${workspaceName}'?`,
    });
    await expect(deleteDialog).toBeVisible();
    await deleteDialog.getByRole('button', { name: 'Delete Workspace', exact: true }).click();
    await waitForLongOperation(this.page);
    await expectToastIsVisible(this.page, `Deleted workspace '${workspaceName}'.`);
  }

  async waitForPageLoad() {
    await waitForLongOperation(this.page);
  }
}
