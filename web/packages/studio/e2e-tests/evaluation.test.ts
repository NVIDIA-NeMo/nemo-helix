// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EvaluationsAPI } from '@e2e-tests/api/evaluations';
import { ProjectEvaluationsPage } from '@e2e-tests/pages/project-evaluations';
import {
  buildTestNamespace,
  buildTestWorkspacePrefix,
  CURRENT_YYYY_MM_DD,
  DEFAULT_BASE_MODEL,
  generateTestResourceName,
  generateTestWorkspaceName,
  LLM_JUDGE_MODEL_ID,
} from '@e2e-tests/utils/constants';
import { PLATFORM_BASE_URL } from '@e2e-tests/utils/environment';
import {
  testDatasetFilesFixture,
  type TestDatasetFilesFixture,
  testDatasetFixture,
  type TestDatasetFixture,
  testEvaluationConfigFixture,
  type TestEvaluationConfigFixture,
  testWorkspaceFixture,
  type TestWorkspaceFixture,
} from '@e2e-tests/utils/fixtures';
import { disableAuthForTest } from '@e2e-tests/utils/pageUtils';
import {
  DEFAULT_LLM_JUDGE_DEFAULTS,
  EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING,
  METRIC_LABELS,
  type MetricNameApi,
} from '@nemo/common/src/constants/metrics';
import { generateLLMJudgeUserMessage } from '@nemo/common/src/utils/evaluation';
import { expect, test as baseTest } from '@playwright/test';
import { defaultIdealResponse } from '@studio/constants/evaluationDefaults';

/** Evaluation config input for create. */
type EvaluationConfigInput = Record<string, unknown>;

const MOCKS_SUBFOLDER = 'sentiment';
const DATASET_INPUT_FILE = 'eval.json';
const DATASET_OFFLINE_FILE = 'eval-offline.jsonl';
// Paths to mock files that the tests will upload (derived from filenames)
const LOCAL_INPUT_FILE = `${MOCKS_SUBFOLDER}/${DATASET_INPUT_FILE}`;
const LOCAL_OFFLINE_FILE = `${MOCKS_SUBFOLDER}/${DATASET_OFFLINE_FILE}`;

const BUTTON_CTA_CREATE_CONFIGURATION = 'Create Evaluation Configuration';
const BUTTON_CTA_CREATE_EVALUATION_JOB = 'Create Evaluation Job';

const NAMESPACE = buildTestNamespace('evaluation');
const WORKSPACE_PREFIX = buildTestWorkspacePrefix('evaluation');

/**
 * Evaluation target mode for configuration creation.
 * - 'online': Creates a config with task type 'chat-completion' for live model inference
 * - 'offline': Creates a config with task type 'data' for pre-generated outputs
 */
type EvaluationMode = 'online' | 'offline';

// Template values used in configs and assertions
// Offline output template is test-specific (matches eval-offline.jsonl fixture)
const OFFLINE_OUTPUT_TEMPLATE = '{{item.actual_response | trim}}';

/**
 * Helper function to build an evaluation config input with the correct task type.
 * The task type determines which mode the config is compatible with:
 * - 'chat-completion' for online (live model inference)
 * - 'data' for offline (pre-generated outputs)
 */
const buildEvaluationConfigInput = (
  name: string,
  namespace: string,
  project: string,
  datasetFilesUrl: string,
  mode: EvaluationMode
): EvaluationConfigInput => {
  // For online mode, the model output comes from inference (sample.output_text)
  // For offline mode, the cached response is read from the input file (item.actual_response)
  const outputReference =
    mode === 'online' ? EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING : OFFLINE_OUTPUT_TEMPLATE;

  return {
    name,
    namespace,
    project,
    tasks: {
      default: {
        type: mode === 'online' ? 'chat-completion' : 'data',
        params: {
          template: {
            messages: [
              {
                role: 'user',
                content: '{{item.prompt | trim}}',
              },
            ],
            max_tokens: 200,
            temperature: 0.7,
            top_p: 0.95,
          },
        },
        dataset: {
          files_url: datasetFilesUrl,
        },
        metrics: {
          bleu: {
            type: 'bleu',
            params: {
              references: [defaultIdealResponse],
              candidate: outputReference,
            },
          },
          f1: {
            type: 'f1',
            params: {
              ground_truth: defaultIdealResponse,
              prediction: outputReference,
            },
          },
          'llm-judge': {
            type: 'llm-judge',
            params: {
              model: {
                api_endpoint: {
                  url: `${PLATFORM_BASE_URL}/v1/chat/completions`,
                  model_id: LLM_JUDGE_MODEL_ID,
                  format: 'nim',
                },
              },
              template: {
                messages: [
                  {
                    role: 'system',
                    content: DEFAULT_LLM_JUDGE_DEFAULTS.systemMessage,
                  },
                  {
                    role: 'user',
                    content: generateLLMJudgeUserMessage(defaultIdealResponse, outputReference),
                  },
                ],
              },
              scores: {
                similarity: {
                  type: DEFAULT_LLM_JUDGE_DEFAULTS.similarityScoreType,
                  parser: {
                    type: 'regex',
                    pattern: DEFAULT_LLM_JUDGE_DEFAULTS.similarityScoreParserPattern,
                  },
                },
              },
            },
          },
        },
      },
    },
    type: 'custom',
  };
};

interface TestFixtures {
  evaluationsPage: ProjectEvaluationsPage;
  evaluationsAPI: EvaluationsAPI;
  testWorkspace: TestWorkspaceFixture;
  testDataset: TestDatasetFixture;
  testEvaluationFiles: TestDatasetFilesFixture;
  testOnlineEvaluationConfig: TestEvaluationConfigFixture;
  testOfflineEvaluationConfig: TestEvaluationConfigFixture;
}

const test = baseTest.extend<TestFixtures>({
  evaluationsPage: async ({ page }, runFixture) => {
    await runFixture(new ProjectEvaluationsPage(page));
  },
  evaluationsAPI: async ({ request }, runFixture) => {
    await runFixture(new EvaluationsAPI(request));
  },
  testWorkspace: async ({ request }, runFixture) => {
    const workspaceName = generateTestWorkspaceName(WORKSPACE_PREFIX);
    const workspaceDescription = `Workspace created by evaluation.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;
    await testWorkspaceFixture(request, runFixture, workspaceName, workspaceDescription);
  },
  testDataset: async ({ request, testWorkspace }, runFixture) => {
    const datasetName = generateTestResourceName('dataset');
    const datasetDescription = `Dataset created by evaluation.test.ts E2E test on ${CURRENT_YYYY_MM_DD}`;
    await testDatasetFixture(
      request,
      runFixture,
      testWorkspace.workspace,
      datasetName,
      NAMESPACE,
      datasetDescription
    );
  },
  testEvaluationFiles: async ({ request, testDataset }, runFixture) => {
    await testDatasetFilesFixture(request, runFixture, testDataset.workspace, testDataset.dataset, [
      {
        testFilePath: LOCAL_INPUT_FILE,
      },
      {
        testFilePath: LOCAL_OFFLINE_FILE,
      },
    ]);
  },
  testOnlineEvaluationConfig: async ({ request, testEvaluationFiles }, runFixture) => {
    const datasetFilesUrl = `hf://datasets/${testEvaluationFiles.dataset.namespace}/${testEvaluationFiles.dataset.name}/${DATASET_INPUT_FILE}`;
    await testEvaluationConfigFixture(
      request,
      runFixture,
      testEvaluationFiles.workspace,
      buildEvaluationConfigInput(
        generateTestResourceName('online-eval-config'),
        testEvaluationFiles.workspace.name,
        testEvaluationFiles.workspace.name,
        datasetFilesUrl,
        'online'
      )
    );
  },
  testOfflineEvaluationConfig: async ({ request, testEvaluationFiles }, runFixture) => {
    const datasetFilesUrl = `hf://datasets/${testEvaluationFiles.dataset.namespace}/${testEvaluationFiles.dataset.name}/${DATASET_OFFLINE_FILE}`;
    await testEvaluationConfigFixture(
      request,
      runFixture,
      testEvaluationFiles.workspace,
      buildEvaluationConfigInput(
        generateTestResourceName('offline-eval-config'),
        testEvaluationFiles.workspace.name,
        testEvaluationFiles.workspace.name,
        datasetFilesUrl,
        'offline'
      )
    );
  },
});

// FIXME: projects→workspaces migration pending
test.describe.fixme('Evaluations', () => {
  test.beforeEach(async ({ page }) => disableAuthForTest(page));

  test('Creates a new evaluation config', async ({
    page,
    evaluationsPage,
    evaluationsAPI,
    testEvaluationFiles,
  }) => {
    test.slow();
    const configName = generateTestResourceName('evaluation-config');

    await test.step('Navigate to config page', async () => {
      await evaluationsPage.goToEvaluationConfigs(
        testEvaluationFiles.workspace.name,
        testEvaluationFiles.workspace.name
      );
      await page.getByRole('button', { name: BUTTON_CTA_CREATE_CONFIGURATION }).first().click();
    });

    await test.step('Fill configuration details', async () => {
      // Fill Configuration Name
      await page.getByRole('textbox', { name: 'Configuration Name' }).fill(configName);
    });

    await test.step('Select input file', async () => {
      await page.getByRole('button', { name: 'Select File' }).click();
      await evaluationsPage.selectFileFromModal(
        testEvaluationFiles.dataset.name!,
        new RegExp(DATASET_INPUT_FILE)
      );
    });

    await test.step('Select metrics', async () => {
      // Wait for file validation to complete (the validating banner should disappear)
      await page.getByText('Validating file format and structure...').waitFor({ state: 'hidden' });

      // Wait for successful validation - "File Validation" block should appear
      await page.getByText('File Validation').waitFor({ state: 'visible' });

      // Wait for metrics section to load
      await page.locator('text=BLEU').waitFor({ state: 'visible' });

      // Helper to find and click a metric checkbox by label text
      const clickMetricCheckbox = async (metricLabel: string) => {
        const label = page.locator('label').filter({ hasText: new RegExp(`^${metricLabel}$`) });
        const checkboxContainer = label.locator('..');
        const checkbox = checkboxContainer.getByTestId('nv-checkbox-box');
        await expect(checkbox).toBeVisible();
        await checkbox.click();
        await expect(checkbox).toHaveAttribute('aria-checked', 'true');
      };

      // Select all metrics using METRIC_LABELS for display names
      await clickMetricCheckbox(METRIC_LABELS.bleu);
      await clickMetricCheckbox(METRIC_LABELS.f1);
      await clickMetricCheckbox(METRIC_LABELS.rouge);
      await clickMetricCheckbox(METRIC_LABELS.em);
      await clickMetricCheckbox(METRIC_LABELS['string-check']);
      await clickMetricCheckbox(METRIC_LABELS['llm-judge']);

      // Select a model for LLM-as-a-Judge (use exact: true to avoid matching "Model *" required field)
      const llmJudgeModelSelect = page.getByRole('combobox', { name: 'Model', exact: true });
      await expect(llmJudgeModelSelect).toBeVisible();
      await llmJudgeModelSelect.click();
      const firstModel = page.getByRole('option').first();
      await firstModel.click();
    });

    await test.step('Create and verify config', async () => {
      const createButton = page
        .getByRole('button', { name: BUTTON_CTA_CREATE_CONFIGURATION })
        .first();

      // Scroll button into view and wait for form validation to complete (button becomes enabled)
      await createButton.scrollIntoViewIfNeeded();
      await expect(createButton).toBeEnabled();
      await createButton.click();

      // Wait for navigation back to the configs list page with the new config selected
      await page.waitForURL(/\/evaluation\/configurations\?selectedEvaluationConfig=/);

      // Wait for the network to settle
      await page.waitForLoadState('networkidle');

      // Verify the side panel opens showing the config details
      await expect(page.getByText('Evaluation Configuration Details')).toBeVisible();

      // Verify the config name appears (it's in both table and side panel, so use .first())
      await expect(page.getByText(configName).first()).toBeVisible();

      // Verify the Metrics section exists
      await expect(page.getByText('Metrics')).toBeVisible();

      // Helper to verify metric details - online configs should use sample.output_text
      const verifyMetric = async (
        metricName: MetricNameApi,
        expectedFields: { label: string; value: string }[]
      ) => {
        const metricHeading = page
          .locator('.nv-text--label-bold-md')
          .filter({ hasText: metricName });
        await expect(metricHeading).toBeVisible();
        const metricBlock = metricHeading.locator('..');

        for (const { label, value } of expectedFields) {
          await expect(metricBlock.getByText(label)).toBeVisible();
          await expect(metricBlock.getByText(value)).toBeVisible();
        }
      };

      // Verify BLEU metric - online mode uses sample.output_text for candidate
      // (Skip 'Type' assertion as the metric heading already confirms the type)
      await verifyMetric('bleu', [
        { label: 'References', value: defaultIdealResponse },
        { label: 'Candidate', value: EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING },
      ]);

      // Verify F1 metric - online mode uses sample.output_text for prediction
      await verifyMetric('f1', [
        { label: 'Ground Truth Reference', value: defaultIdealResponse },
        { label: 'Prediction', value: EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING },
      ]);

      // Verify ROUGE metric - online mode uses sample.output_text for prediction
      await verifyMetric('rouge', [
        { label: 'Ground Truth Reference', value: defaultIdealResponse },
        { label: 'Prediction', value: EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING },
      ]);

      // Verify EXACT MATCH metric - online mode uses sample.output_text for prediction
      await verifyMetric('em', [
        { label: 'Ground Truth Reference', value: defaultIdealResponse },
        { label: 'Prediction', value: EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING },
      ]);

      // Verify STRING CHECK metric
      await verifyMetric('string-check', [
        { label: 'Check Pattern', value: EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING },
      ]);

      // Verify LLM-as-a-Judge metric - online mode uses sample.output_text in user message
      await verifyMetric('llm-judge', [
        { label: 'User Message', value: EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING },
      ]);
    });

    // Clean up Evaluation Config
    // NOTE: When creating a config from Studio, the namespace is always `default`
    await evaluationsAPI.deleteEvaluationConfig('default', configName);
  });

  test('Launch an online evaluation job', async ({
    page,
    evaluationsPage,
    testOnlineEvaluationConfig,
  }) => {
    await test.step('Navigate to the evaluations list page, and click Create Evaluation Job', async () => {
      await evaluationsPage.gotoEvaluations(
        testOnlineEvaluationConfig.workspace.name,
        testOnlineEvaluationConfig.workspace.name
      );
      await evaluationsPage.waitForPageLoad();
      await page.getByRole('button', { name: BUTTON_CTA_CREATE_EVALUATION_JOB }).first().click();
      await expect(page.getByText('Create an Evaluation Job')).toBeVisible();
    });

    await test.step('Select model', async () => {
      const modelSelectAutocomplete = page.getByRole('combobox', { name: 'Model' });
      await expect(modelSelectAutocomplete).not.toBeDisabled();
      await modelSelectAutocomplete.click();
      const firstModel = page.getByRole('option', { name: DEFAULT_BASE_MODEL }).first();
      await firstModel.click();
    });

    await test.step('Select existing configuration', async () => {
      await evaluationsPage.selectConfiguration(
        String((testOnlineEvaluationConfig.evaluationConfig as { name?: string }).name ?? '')
      );
    });

    await test.step('Create evaluation job', async () => {
      await evaluationsPage.createJobAndVerify(BUTTON_CTA_CREATE_EVALUATION_JOB);
    });
  });

  test('Launch an offline evaluation job', async ({
    page,
    evaluationsPage,
    testEvaluationFiles,
    testOfflineEvaluationConfig,
  }) => {
    await test.step('Navigate to evaluation page and select Data Source target', async () => {
      await evaluationsPage.gotoEvaluations(
        testOfflineEvaluationConfig.workspace.name,
        testOfflineEvaluationConfig.workspace.name
      );
      await evaluationsPage.waitForPageLoad();
      await page.getByRole('button', { name: BUTTON_CTA_CREATE_EVALUATION_JOB }).first().click();
      await expect(page.getByText('Create an Evaluation Job')).toBeVisible();

      // Select "Data Source Targets" card for offline evaluation
      await page.getByText('Data Source Targets').click();
    });

    await test.step('Select target file', async () => {
      await page.getByRole('button', { name: 'Select File' }).click();
      await evaluationsPage.selectFileFromModal(
        testEvaluationFiles.dataset.name!,
        new RegExp(DATASET_OFFLINE_FILE)
      );
    });

    await test.step('Select existing configuration', async () => {
      await evaluationsPage.selectConfiguration(
        String((testOfflineEvaluationConfig.evaluationConfig as { name?: string }).name ?? '')
      );
    });

    await test.step('Create evaluation job', async () => {
      await evaluationsPage.createJobAndVerify(BUTTON_CTA_CREATE_EVALUATION_JOB);
    });
  });
});
