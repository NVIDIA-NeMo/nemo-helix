// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DatasetsAPI } from '@e2e-tests/api/datasets';
import { WorkspacesAPI } from '@e2e-tests/api/workspaces';
import { ProjectDatasetsPage } from '@e2e-tests/pages/project-datasets';
import {
  CURRENT_YYYY_MM_DD,
  buildTestNamespace,
  buildTestWorkspacePrefix,
  generateTestResourceName,
  generateTestWorkspaceName,
} from '@e2e-tests/utils/constants';
import {
  testDatasetFilesFixture,
  type TestDatasetFilesFixture,
  testDatasetFixture,
  type TestDatasetFixture,
  testWorkspaceFixture,
  type TestWorkspaceFixture,
} from '@e2e-tests/utils/fixtures';
import { disableAuthForTest } from '@e2e-tests/utils/pageUtils';
import { expect, test as baseTest } from '@playwright/test';

// Path to mock files that the tests will upload
const TRAINING_FILE = 'sentiment/train.jsonl';
const VALIDATION_FILE = 'sentiment/validation.jsonl';

// Namespace used when creating datasets, and prefix for the workspaces created by this suite.
const NAMESPACE = buildTestNamespace('datasets');
const WORKSPACE_PREFIX = buildTestWorkspacePrefix('datasets');
interface TestFixtures {
  datasetsPage: ProjectDatasetsPage;
  workspacesApi: WorkspacesAPI;
  datasetsApi: DatasetsAPI;
  testWorkspace: TestWorkspaceFixture;
  testDataset: TestDatasetFixture;
  testTrainingFile: TestDatasetFilesFixture;
}

const test = baseTest.extend<TestFixtures>({
  datasetsPage: async ({ page }, runFixture) => {
    await runFixture(new ProjectDatasetsPage(page));
  },
  workspacesApi: async ({ request }, runFixture) => {
    await runFixture(new WorkspacesAPI(request));
  },
  datasetsApi: async ({ request }, runFixture) => {
    await runFixture(new DatasetsAPI(request));
  },
  testWorkspace: async ({ request }, runFixture) => {
    const workspaceName = generateTestWorkspaceName(WORKSPACE_PREFIX);
    const workspaceDescription = `Workspace created by datasets.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;
    await testWorkspaceFixture(request, runFixture, workspaceName, workspaceDescription);
  },
  testDataset: async ({ request, testWorkspace }, runFixture) => {
    const datasetName = generateTestResourceName('dataset');
    const datasetDescription = `Dataset created by datasets.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;
    await testDatasetFixture(
      request,
      runFixture,
      testWorkspace.workspace,
      datasetName,
      NAMESPACE,
      datasetDescription
    );
  },
  testTrainingFile: async ({ request, testDataset }, runFixture) => {
    await testDatasetFilesFixture(request, runFixture, testDataset.workspace, testDataset.dataset, [
      {
        testFilePath: TRAINING_FILE,
        datasetFolder: 'training',
      },
    ]);
  },
});

// FIXME: projects→workspaces migration pending
test.describe.fixme('Projects: Datasets', () => {
  test.beforeEach(async ({ page }) => disableAuthForTest(page));
  // Each test should be responsible for deleting any workspace it creates.
  // This clean-up step is just an extra measure to delete any workspaces that may not have been successfully deleted.
  test.afterAll(async ({ workspacesApi }) => {
    await workspacesApi.deleteAllWorkspacesByPrefix(WORKSPACE_PREFIX);
  });

  test('Should render list of datasets', async ({ page, datasetsPage, testDataset }) => {
    await datasetsPage.goto(testDataset.workspace.name, testDataset.workspace.name);
    await datasetsPage.waitForPageLoad();
    await expect(page.getByText(testDataset.dataset.name!)).toBeVisible();
    await expect(
      page.getByText(String((testDataset.dataset as { description?: string }).description ?? ''))
    ).toBeVisible();
  });

  test('Should successfully upload training file', async ({ page, datasetsPage, testDataset }) => {
    test.slow();
    await datasetsPage.goto(
      testDataset.workspace.name,
      testDataset.workspace.name,
      testDataset.dataset
    );
    await datasetsPage.waitForPageLoad();
    expect(page.getByText('No files')).toBeVisible();
    await datasetsPage.uploadFileToDataset(TRAINING_FILE);
  });

  test('Should successfully upload validation file', async ({
    page,
    datasetsPage,
    testDataset,
  }) => {
    test.slow();
    await datasetsPage.goto(
      testDataset.workspace.name,
      testDataset.workspace.name,
      testDataset.dataset
    );
    await datasetsPage.waitForPageLoad();
    expect(page.getByText('No files')).toBeVisible();
    await datasetsPage.uploadFileToDataset(VALIDATION_FILE);
  });

  test('Should successfully delete file', async ({ datasetsPage, testTrainingFile }) => {
    test.slow();
    await datasetsPage.goto(
      testTrainingFile.workspace.name,
      testTrainingFile.workspace.name,
      testTrainingFile.dataset
    );
    await datasetsPage.waitForPageLoad();
    await datasetsPage.deleteFileFromDataset('training/train.jsonl');
  });

  test('Should successfully rename file', async ({ datasetsPage, testTrainingFile }) => {
    test.slow();
    await datasetsPage.goto(
      testTrainingFile.workspace.name,
      testTrainingFile.workspace.name,
      testTrainingFile.dataset
    );
    await datasetsPage.waitForPageLoad();

    await datasetsPage.renameFileInDataset('training/train.jsonl', 'renamed-train.jsonl');
  });

  test('Should successfully delete a dataset', async ({ datasetsPage, testDataset }) => {
    await datasetsPage.goto(testDataset.workspace.name, testDataset.workspace.name);
    await datasetsPage.waitForPageLoad();
    await datasetsPage.deleteDataset(testDataset.dataset.name!);
  });
});
