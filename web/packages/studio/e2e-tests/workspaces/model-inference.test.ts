// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelsAPI } from '@e2e-tests/api/models';
import { WorkspacesAPI } from '@e2e-tests/api/workspaces';
import { ProjectModelsPage } from '@e2e-tests/pages/project-models';
import {
  buildTestWorkspacePrefix,
  CURRENT_HH_MM_SS,
  CURRENT_YYYY_MM_DD,
  DEFAULT_BASE_MODEL,
  generateTestResourceName,
  generateTestWorkspaceName,
} from '@e2e-tests/utils/constants';
import {
  type TestWorkspaceFixture,
  type TestModelFixture,
  testWorkspaceFixture,
  testModelFixture,
} from '@e2e-tests/utils/fixtures';
import { expectChatResponseToContain } from '@e2e-tests/utils/models';
import { disableAuthForTest } from '@e2e-tests/utils/pageUtils';
import type { CreateModelEntityRequest } from '@nemo/sdk/generated/platform/schema';
import { test as baseTest } from '@playwright/test';

// USER_ID-scoped prefix for workspaces created by this suite; afterAll cleans up by this prefix.
const WORKSPACE_PREFIX = buildTestWorkspacePrefix('model-inference');
const MODEL_SYNC_WAIT_TIME_MS = 5.5 * 60 * 1000; // 7 minutes

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
    const workspaceDescription = `Workspace created by model-inference.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;
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
test.describe.fixme('Projects Model Inference', () => {
  test.describe.configure({ retries: 0 });
  test.beforeEach(async ({ page }) => disableAuthForTest(page));

  // Each test should be responsible for deleting any workspace it creates.
  // This clean-up step is just an extra measure to delete any workspaces that may not have been successfully deleted.
  test.afterAll(async ({ workspacesApi }) => {
    await workspacesApi.deleteAllWorkspacesByPrefix(WORKSPACE_PREFIX);
  });

  test('Base model inference with no settings', async ({
    page,
    projectModelsPage,
    modelsApi,
    testWorkspace,
  }) => {
    test.setTimeout(MODEL_SYNC_WAIT_TIME_MS + 60 * 1000);
    const modelName = `E2E_MODEL_${CURRENT_YYYY_MM_DD}_${CURRENT_HH_MM_SS()}`;
    await projectModelsPage.goto(testWorkspace.workspace.name, testWorkspace.workspace.name);
    await projectModelsPage.waitForPageLoad();
    await projectModelsPage.createModel({
      modelName,
      projectNamespace: testWorkspace.workspace.name,
    });

    // In the background, NIM periodically fetches newly-created models.
    // Wait before trying to run inference on this model.
    await page.waitForSelector('textarea[aria-label="Task prompt"]:not([disabled])', {
      timeout: MODEL_SYNC_WAIT_TIME_MS,
    });

    // Chat with model
    await page
      .getByRole('textbox', { name: 'Task prompt' })
      .fill('What is the capital of Washington state?');
    await page.getByRole('button', { name: 'Submit' }).click();

    await expectChatResponseToContain(page, 'Olympia');

    // Clean up model
    await modelsApi.deleteModel('default', modelName);
  });

  test('Base model inference with system prompt and ICL', async ({
    page,
    projectModelsPage,
    testWorkspace,
  }) => {
    test.setTimeout(MODEL_SYNC_WAIT_TIME_MS + 60 * 1000);
    const modelName = `E2E_MODEL_${CURRENT_YYYY_MM_DD}_${CURRENT_HH_MM_SS()}`;
    await projectModelsPage.goto(testWorkspace.workspace.name, testWorkspace.workspace.name);
    await projectModelsPage.waitForPageLoad();

    const systemPromptTemplate = `You will respond to every question with a single word: potato
        {{icl_few_shot_examples}}`;
    const iclFewShotExamples = new File(
      [
        `
        {"question": "What is the capital of Mongolia?", "answer": "potato"}
        {"question": "What is the current date?", "answer": "potato"}
        {"question": "What is the capital of Washington state?", "answer": "potato"}
        `,
      ],
      'icl-few-shot-examples.jsonl',
      { type: 'application/json' }
    );
    await projectModelsPage.createModel({
      modelName,
      systemPromptTemplate,
      iclFewShotExamples,
      projectNamespace: testWorkspace.workspace.name,
    });

    await page.waitForSelector('textarea[aria-label="Task prompt"]:not([disabled])', {
      timeout: MODEL_SYNC_WAIT_TIME_MS,
    });

    await page
      .getByRole('textbox', { name: 'Task prompt' })
      .fill('What is the capital of Washington state?');
    await page.getByRole('button', { name: 'Submit' }).click();
    await expectChatResponseToContain(page, 'potato');
  });
});
