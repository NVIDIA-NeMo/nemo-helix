// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelsAPI } from '@e2e-tests/api/models';
import { WorkspacesAPI } from '@e2e-tests/api/workspaces';
import { ProjectModelsPage } from '@e2e-tests/pages/project-models';
import {
  buildTestWorkspacePrefix,
  CURRENT_YYYY_MM_DD,
  DEFAULT_BASE_MODEL,
  generateTestResourceName,
  generateTestWorkspaceName,
} from '@e2e-tests/utils/constants';
import {
  testModelFixture,
  type TestModelFixture,
  type TestWorkspaceFixture,
  testWorkspaceFixture,
} from '@e2e-tests/utils/fixtures';
import { disableAuthForTest } from '@e2e-tests/utils/pageUtils';
import type { CreateModelEntityRequest } from '@nemo/sdk/generated/platform/schema';
import { test as baseTest } from '@playwright/test';

// USER_ID-scoped prefix for workspaces created by this suite; afterAll cleans up by this prefix.
const WORKSPACE_PREFIX = buildTestWorkspacePrefix('model');

interface TestFixtures {
  projectModelsPage: ProjectModelsPage;
  modelsApi: ModelsAPI;
  workspacesApi: WorkspacesAPI;
  testWorkspace: TestWorkspaceFixture;
  testModel: TestModelFixture;
}

const test = baseTest.extend<TestFixtures>({
  projectModelsPage: async ({ page }, runFixture) => {
    await runFixture(new ProjectModelsPage(page));
  },
  modelsApi: async ({ request }, runFixture) => {
    await runFixture(new ModelsAPI(request));
  },
  workspacesApi: async ({ request }, runFixture) => {
    await runFixture(new WorkspacesAPI(request));
  },
  testWorkspace: async ({ request }, runFixture) => {
    const workspaceName = generateTestWorkspaceName(WORKSPACE_PREFIX);
    const workspaceDescription = `Workspace created by model.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;
    await testWorkspaceFixture(request, runFixture, workspaceName, workspaceDescription);
  },
  testModel: async ({ request, testWorkspace }, runFixture) => {
    const createModelBody: CreateModelEntityRequest = {
      base_model: DEFAULT_BASE_MODEL,
      name: generateTestResourceName('model'),
      project: testWorkspace.workspace.name,
      prompt: {
        system_prompt: '',
        icl_few_shot_examples: '{{icl_few_shot_examples}}',
      },
    };
    await testModelFixture(
      request,
      runFixture,
      testWorkspace.workspace,
      testWorkspace.workspace.name,
      createModelBody
    );
  },
});

// FIXME: projects→workspaces migration pending
test.describe.fixme('Projects: Models', () => {
  test.beforeEach(async ({ page }) => disableAuthForTest(page));
  // Each test should be responsible for deleting any workspace it creates.
  // This clean-up step is just an extra measure to delete any workspaces that may not have been successfully deleted.
  test.afterAll(async ({ workspacesApi }) => {
    await workspacesApi.deleteAllWorkspacesByPrefix(WORKSPACE_PREFIX);
  });

  test('Create a model', async ({ projectModelsPage, modelsApi, testWorkspace }) => {
    test.slow();
    const modelName = generateTestResourceName('model');

    await projectModelsPage.goto(testWorkspace.workspace.name, testWorkspace.workspace.name);
    await projectModelsPage.waitForPageLoad();
    await projectModelsPage.createModel({
      modelName,
      projectNamespace: testWorkspace.workspace.name,
    });

    // Clean up model
    // NOTE: When creating a model from Studio, the namespace is always `default`
    await modelsApi.deleteModel('default', modelName);
  });

  test('Delete a model', async ({ projectModelsPage, testModel }) => {
    await projectModelsPage.goto(testModel.workspace.name, testModel.workspace.name);
    await projectModelsPage.waitForPageLoad();
    await projectModelsPage.deleteModel(`${testModel.model.workspace}/${testModel.model.name}`);
  });
});
