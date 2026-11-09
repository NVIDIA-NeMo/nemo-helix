// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspacesAPI } from '@e2e-tests/api/workspaces';
import { WorkspacesPage } from '@e2e-tests/pages/workspaces';
import {
  buildTestWorkspacePrefix,
  CURRENT_YYYY_MM_DD,
  generateTestResourceName,
  generateTestWorkspaceName,
} from '@e2e-tests/utils/constants';
import { type TestWorkspaceFixture, testWorkspaceFixture } from '@e2e-tests/utils/fixtures';
import { disableAuthForTest } from '@e2e-tests/utils/pageUtils';
import { expect, test as baseTest } from '@playwright/test';

// USER_ID-scoped prefix for all workspace names created by this suite; afterAll cleans up
// by this prefix, so a run only deletes its own workspaces.
const WORKSPACE_TEST_PREFIX = buildTestWorkspacePrefix('workspace');

interface TestFixtures {
  workspacesPage: WorkspacesPage;
  workspacesApi: WorkspacesAPI;
  testWorkspace: TestWorkspaceFixture;
}

const test = baseTest.extend<TestFixtures>({
  workspacesPage: async ({ page }, runFixture) => {
    await runFixture(new WorkspacesPage(page));
  },
  workspacesApi: async ({ request }, runFixture) => {
    await runFixture(new WorkspacesAPI(request));
  },
  testWorkspace: async ({ request }, runFixture) => {
    const workspaceName = generateTestWorkspaceName(WORKSPACE_TEST_PREFIX);
    const workspaceDescription = `Workspace created by workspace.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;
    await testWorkspaceFixture(request, runFixture, workspaceName, workspaceDescription);
  },
});

test.describe('Workspaces', () => {
  test.beforeEach(async ({ page }) => disableAuthForTest(page));
  // Each test should be responsible for deleting any resource it creates.
  // This clean-up step is a safety net to delete workspaces that may not have been cleaned up.
  test.afterAll(async ({ workspacesApi }) => {
    await workspacesApi.deleteAllWorkspacesByPrefix(WORKSPACE_TEST_PREFIX);
  });

  test('Creates a workspace', async ({ page, workspacesPage, workspacesApi }) => {
    test.slow();
    const workspaceName = generateTestWorkspaceName(WORKSPACE_TEST_PREFIX);
    const workspaceDescription = `Workspace created by workspace.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;

    await workspacesPage.goto();
    await workspacesPage.waitForPageLoad();
    await workspacesPage.createWorkspace(workspaceName, workspaceDescription);

    // Expect to be redirected to the new workspace
    await page.waitForURL(`**/workspaces/${workspaceName}/**`);
    await workspacesPage.waitForPageLoad();
    await expect(page).toHaveURL(new RegExp(`/workspaces/${workspaceName}/`));

    // Clean up
    await workspacesApi.deleteWorkspace(workspaceName);
  });

  test('Updates a workspace description', async ({ workspacesPage, testWorkspace }) => {
    test.slow();
    await workspacesPage.gotoSettings(testWorkspace.workspace.name);
    await workspacesPage.waitForPageLoad();
    const updatedDescription = `${testWorkspace.workspace.description || generateTestResourceName('workspace')} Updated`;
    await workspacesPage.editWorkspaceDescription(updatedDescription);
  });

  test('Deletes a workspace', async ({ page, workspacesPage, testWorkspace }) => {
    test.slow();
    await workspacesPage.gotoSettings(testWorkspace.workspace.name);
    await workspacesPage.waitForPageLoad();
    await workspacesPage.deleteWorkspace(testWorkspace.workspace.name);

    // After deletion, should be redirected to the default workspace
    await page.waitForURL('**/workspaces/default/**');
  });
});
