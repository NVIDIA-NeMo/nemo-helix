// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspacesAPI } from '@e2e-tests/api/workspaces';
import { ProjectSafeSynthesizerPage } from '@e2e-tests/pages/project-safe-synthesizer';
import {
  CURRENT_YYYY_MM_DD,
  MOCKS_DIR,
  buildTestNamespace,
  buildTestWorkspacePrefix,
  generateShortTestResourceName,
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
import { disableAuthForTest, waitForLongOperation } from '@e2e-tests/utils/pageUtils';
import { expect, test as baseTest } from '@playwright/test';
import path from 'path';

// Path to mock files that the tests will upload
const TRAINING_FILE = 'sentiment/train.jsonl';

const NAMESPACE = buildTestNamespace('safe-synthesizer');
const WORKSPACE_PREFIX = buildTestWorkspacePrefix('safe-synthesizer');

interface TestFixtures {
  safeSynthesizerPage: ProjectSafeSynthesizerPage;
  workspacesApi: WorkspacesAPI;
  testWorkspace: TestWorkspaceFixture;
  testDataset: TestDatasetFixture;
  testDatasetFiles: TestDatasetFilesFixture;
}

const test = baseTest.extend<TestFixtures>({
  safeSynthesizerPage: async ({ page }, runFixture) => {
    await runFixture(new ProjectSafeSynthesizerPage(page));
  },
  workspacesApi: async ({ request }, runFixture) => {
    await runFixture(new WorkspacesAPI(request));
  },
  testWorkspace: async ({ request }, runFixture) => {
    const workspaceName = generateTestWorkspaceName(WORKSPACE_PREFIX);
    const workspaceDescription = `Workspace created by safeSynthesizer.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;
    await testWorkspaceFixture(request, runFixture, workspaceName, workspaceDescription);
  },
  testDataset: async ({ request, testWorkspace }, runFixture) => {
    const datasetName = generateShortTestResourceName();
    const datasetDescription = `Dataset created by safeSynthesizer.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;
    await testDatasetFixture(
      request,
      runFixture,
      testWorkspace.workspace,
      datasetName,
      NAMESPACE,
      datasetDescription
    );
  },
  testDatasetFiles: async ({ request, testDataset }, runFixture) => {
    await testDatasetFilesFixture(request, runFixture, testDataset.workspace, testDataset.dataset, [
      {
        testFilePath: TRAINING_FILE,
      },
    ]);
  },
});

// FIXME: projects→workspaces migration pending
test.describe.fixme('Safe Synthesizer', () => {
  test.beforeEach(async ({ page }) => disableAuthForTest(page));

  // Each test should be responsible for deleting any resource it creates.
  // This clean-up step is just an extra measure to delete any projects that may have not have been successfully deleted.
  test.afterAll(async ({ workspacesApi }) => {
    await workspacesApi.deleteAllWorkspacesByPrefix(WORKSPACE_PREFIX);
  });

  test('Creates a safe synthesizer job', async ({
    page,
    safeSynthesizerPage,
    testDatasetFiles,
  }) => {
    test.slow();
    const { workspace, dataset } = testDatasetFiles;
    const jobName = generateTestResourceName('safe-synth-job');

    // Parameter values to test
    const NUM_RECORDS_VALUE = '10';

    await test.step('Navigate to safe synthesizer page and click "Create New Job"', async () => {
      await safeSynthesizerPage.goto(workspace.name, workspace.name);
      await waitForLongOperation(page);

      const newJobButton = page.getByTestId('nv-page-header-footer').getByTestId('nv-button');
      await newJobButton.click();

      // Wait for navigation to the new job form
      await page.waitForURL('**/safe-synthesizer/new');
      await waitForLongOperation(page);

      // Verify we're on the new job page by checking for the form heading
      await expect(page.getByText('Generate Private Synthetic Data')).toBeVisible();
    });

    await test.step('Fill in job name', async () => {
      const jobNameField = page.getByRole('textbox', { name: 'Job Name' });
      await jobNameField.fill(jobName);
    });

    await test.step('Select training data source', async () => {
      await page.getByRole('button', { name: 'Select File' }).click();
      await waitForLongOperation(page);

      // Click the Dataset select dropdown
      await page.getByRole('combobox', { name: 'Dataset' }).click();

      const datasetOption = page.getByRole('option', { name: dataset.name! });
      await datasetOption.click();

      await page.getByTestId('nv-spinner-spinner').waitFor({ state: 'hidden' });

      const fileRow = page.getByRole('row', { name: new RegExp(TRAINING_FILE.split('/')[1]) });
      const fileCheckbox = fileRow.getByRole('checkbox');
      await fileCheckbox.check();

      await page.getByRole('button', { name: 'Add Selected File' }).click();
      await waitForLongOperation(page);
    });

    await test.step('Set number of synthetic records', async () => {
      const numRecordsField = page.getByRole('spinbutton', {
        name: 'Number of synthetic records to generate',
      });
      await numRecordsField.fill(NUM_RECORDS_VALUE);
    });

    await test.step('Submit the form', async () => {
      // Click the Continue button to create the job
      await page.getByRole('button', { name: 'Continue' }).click();

      // Wait for navigation to job details page
      await page.waitForURL('**/safe-synthesizer/job/**', { timeout: 60000 });
      await waitForLongOperation(page);
    });

    await test.step('Verify job was created and redirected to details page', async () => {
      // Verify job name is visible on the details page
      await expect(page.getByText(jobName)).toBeVisible({ timeout: 10000 });

      // Navigate back to the landing page and verify the job is visible in the list
      await safeSynthesizerPage.goto(workspace.name, workspace.name);
      await waitForLongOperation(page);

      // Wait for the jobs table/list to load and verify the job appears
      const jobRow = page.getByRole('row', { name: new RegExp(jobName) });
      await expect(jobRow).toBeVisible({ timeout: 10000 });
    });
  });

  test('Creates a safe synthesizer job with advanced options and uploads a file to a new dataset', async ({
    page,
    safeSynthesizerPage,
    testWorkspace,
  }) => {
    test.slow();
    const { workspace } = testWorkspace;
    const jobName = generateTestResourceName('safe-synth-advanced');
    const datasetName = generateShortTestResourceName();

    // Advanced parameter values to test
    const TEMPERATURE_VALUE = '1.2';
    const TOP_P_VALUE = '0.95';
    const NUM_RECORDS_VALUE = '10';
    const NUM_INPUT_RECORDS_VALUE = '1000';
    const ROPE_SCALING_VALUE = '2';

    await test.step('Navigate to safe synthesizer page and click "Create New Job"', async () => {
      await safeSynthesizerPage.goto(workspace.name, workspace.name);
      await waitForLongOperation(page);

      const newJobButton = page.getByTestId('nv-page-header-footer').getByTestId('nv-button');
      await newJobButton.click();

      await page.waitForURL('**/safe-synthesizer/new');
      await waitForLongOperation(page);

      await expect(page.getByText('Generate Private Synthetic Data')).toBeVisible();
    });

    await test.step('Fill in job name', async () => {
      const jobNameField = page.getByRole('textbox', { name: 'Job Name' });
      await jobNameField.fill(jobName);
    });

    await test.step('Upload file to new dataset', async () => {
      // Click "Select File" to open the upload modal
      await page.getByRole('button', { name: 'Select File' }).click();
      await waitForLongOperation(page);

      // Click the Dataset select dropdown
      await page.getByRole('combobox', { name: 'Dataset' }).click();

      // Select "New Dataset" option from the dropdown
      await page.getByRole('option', { name: 'New Dataset' }).click();
      await waitForLongOperation(page);

      // Fill in dataset name (this field appears after selecting "New Dataset")
      const datasetNameField = page.getByPlaceholder('Name this Dataset');
      await datasetNameField.fill(datasetName);

      // Upload file using the KUI Upload component
      const fileChooserPromise = page.waitForEvent('filechooser');

      // Find and click the upload button/area
      // The Upload component should have a button or clickable area
      const uploadButton = page.getByRole('button', { name: 'Choose a file' });
      await uploadButton.click();

      const fileChooser = await fileChooserPromise;
      await fileChooser.setFiles(path.resolve(MOCKS_DIR, TRAINING_FILE));

      // Wait for file to be uploaded and appear in the list
      await waitForLongOperation(page);

      // Verify file was added
      await expect(page.getByText(TRAINING_FILE.split('/')[1])).toBeVisible();

      // Select the uploaded file by checking its checkbox
      const fileRow = page.getByRole('row', { name: new RegExp(TRAINING_FILE.split('/')[1]) });
      const fileCheckbox = fileRow.getByRole('checkbox');
      await fileCheckbox.check();

      // Click the submit button to add the file
      await page.getByRole('button', { name: 'Add Selected File' }).click();
      await waitForLongOperation(page);
    });

    await test.step('Set number of synthetic records', async () => {
      const numRecordsField = page.getByRole('spinbutton', {
        name: 'Number of synthetic records to generate',
      });
      await numRecordsField.fill(NUM_RECORDS_VALUE);
    });

    await test.step('Configure privacy protection', async () => {
      // Select "Highest Privacy" option which enables Differential Privacy
      await page.getByRole('radio', { name: /Highest Privacy/i }).click();
      await expect(page.getByRole('radio', { name: /Highest Privacy/i })).toBeChecked();
    });

    await test.step('Open and configure advanced parameters', async () => {
      // Open the Advanced Parameters accordion
      await page.getByText('Show Advanced Parameters').click();
      await waitForLongOperation(page);

      // Verify accordion is open
      await expect(page.getByText('Core Generation Settings')).toBeVisible();
    });

    await test.step('Adjust temperature setting', async () => {
      const temperatureInput = page.locator(
        'input[name="spec.config.generation.temperature"][type="number"]'
      );

      await temperatureInput.waitFor({ state: 'visible', timeout: 10000 });
      await temperatureInput.scrollIntoViewIfNeeded();
      await temperatureInput.fill(TEMPERATURE_VALUE);
    });

    await test.step('Adjust top_p setting', async () => {
      const topPInput = page.locator('input[name="spec.config.generation.top_p"][type="number"]');

      await topPInput.waitFor({ state: 'visible', timeout: 10000 });
      await topPInput.scrollIntoViewIfNeeded();
      await topPInput.fill(TOP_P_VALUE);
    });

    await test.step('Configure training data sampling', async () => {
      // Uncheck "Use Automatic Sampling"
      const automaticSamplingCheckbox = page.getByRole('checkbox', {
        name: /Use Automatic Sampling/i,
      });
      await automaticSamplingCheckbox.waitFor({ state: 'visible' });
      await automaticSamplingCheckbox.scrollIntoViewIfNeeded();
      await automaticSamplingCheckbox.click();

      const numInputRecordsField = page.locator(
        'input[name="spec.config.training.num_input_records_to_sample"][type="number"]'
      );
      await numInputRecordsField.waitFor({ state: 'visible', timeout: 10000 });
      await numInputRecordsField.scrollIntoViewIfNeeded();
      await numInputRecordsField.fill(NUM_INPUT_RECORDS_VALUE);
    });

    await test.step('Configure context length scaling', async () => {
      // Uncheck "Use Automatic Scaling"
      const automaticScalingCheckbox = page.getByRole('checkbox', {
        name: /Use Automatic Scaling/i,
      });
      await automaticScalingCheckbox.waitFor({ state: 'visible' });
      await automaticScalingCheckbox.click();

      const ropeScalingInput = page.locator(
        'input[name="spec.config.training.rope_scaling_factor"][type="number"]'
      );

      await ropeScalingInput.waitFor({ state: 'visible', timeout: 10000 });
      await ropeScalingInput.scrollIntoViewIfNeeded();
      await ropeScalingInput.fill(ROPE_SCALING_VALUE);
    });

    await test.step('Configure PII replacement', async () => {
      const piiSwitch = page.locator('button[name="spec.config.enable_replace_pii"]');

      await piiSwitch.waitFor({ state: 'visible', timeout: 10000 });
      await piiSwitch.scrollIntoViewIfNeeded();

      // Check current state and toggle if needed
      const isChecked = await piiSwitch.getAttribute('data-state');
      if (isChecked !== 'checked') {
        await piiSwitch.click();
      }
    });

    await test.step('Configure data preparation settings (optional)', async () => {
      // Scroll back up to the Training Data section
      await page.getByText('Training Data', { exact: true }).scrollIntoViewIfNeeded();

      // Click "Show Data Preparation Settings" accordion
      const dataPreparationAccordion = page.getByText('Show Data Preparation Settings');
      if (await dataPreparationAccordion.isVisible()) {
        await dataPreparationAccordion.click();
        await waitForLongOperation(page);

        // Verify the accordion opened by checking for the field text
        await expect(page.getByText('group_training_examples_by', { exact: true })).toBeVisible({
          timeout: 10000,
        });

        // Verify both fields are present
        await expect(page.getByText('order_training_examples_by', { exact: true })).toBeVisible();

        // Note: These fields have placeholder items in the component ('test', 'test2')
        // In a real scenario, these would be populated based on the dataset columns
        // For this test, we just verify they're accessible
      }
    });

    await test.step('Submit the form', async () => {
      await page.getByRole('button', { name: 'Continue' }).click();

      await page.waitForURL('**/safe-synthesizer/job/**', { timeout: 60000 });
      await waitForLongOperation(page);
    });

    await test.step('Verify job was created with advanced settings', async () => {
      // Verify job name is visible on the details page
      await expect(page.getByText(jobName)).toBeVisible({ timeout: 10000 });

      // Click "View Job Config" to open the configuration panel
      await page.getByRole('button', { name: 'View Job Config' }).click();
      await waitForLongOperation(page);

      // Verify the config panel/drawer is visible
      await expect(page.getByText('Job Configuration')).toBeVisible();

      // Verify our custom temperature setting appears in the config
      const temperatureRegex = new RegExp(
        `temperature:\\s*${TEMPERATURE_VALUE.replace('.', '\\.')}`
      );
      await expect(page.getByText(temperatureRegex)).toBeVisible();

      // Verify our custom top_p setting appears in the config
      const topPRegex = new RegExp(`top_p:\\s*${TOP_P_VALUE.replace('.', '\\.')}`);
      await expect(page.getByText(topPRegex)).toBeVisible();

      // Close the config panel
      const closeButton = page
        .getByRole('button', { name: 'Close' })
        .or(page.locator('button[aria-label="Close"]'));
      if (await closeButton.isVisible()) {
        await closeButton.click();
      }
    });
  });
});
