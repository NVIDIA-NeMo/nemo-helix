// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DatasetsAPI } from '@e2e-tests/api/datasets';
import { EvaluationsAPI } from '@e2e-tests/api/evaluations';
import { ModelsAPI } from '@e2e-tests/api/models';
import { WorkspacesAPI } from '@e2e-tests/api/workspaces';
import type {
  CreateModelEntityRequest,
  ModelEntity,
  Workspace,
} from '@nemo/sdk/generated/platform/schema';
import type { APIRequestContext } from '@playwright/test';

/** Dataset shape for e2e fixtures. */
type Dataset = { files_url?: string; name?: string; namespace?: string; [key: string]: unknown };
/** Evaluation config shape for e2e fixtures. */
type EvaluationConfig = Record<string, unknown>;
/** Evaluation config input for create. */
type EvaluationConfigInput = Record<string, unknown>;

export interface TestWorkspaceFixture {
  workspace: Workspace;
}

/**
 * Common fixture that creates a test workspace to use for an individual test.
 * The test will receive an argument of type `TestWorkspaceFixture`.
 * Deletes the workspace after the test runs.
 */
export const testWorkspaceFixture = async (
  request: APIRequestContext,
  runFixture: (returnValue: TestWorkspaceFixture) => Promise<void>,
  workspaceName: string,
  workspaceDescription: string
) => {
  const workspacesApi = new WorkspacesAPI(request);
  const testWorkspace = await workspacesApi.createWorkspace({
    name: workspaceName,
    description: workspaceDescription,
  });

  await runFixture({ workspace: testWorkspace });

  await workspacesApi.deleteWorkspace(testWorkspace.name);
};

export interface TestModelFixture {
  workspace: Workspace;
  model: ModelEntity;
}

/**
 * Common fixture that creates a test model to use for an individual test.
 * The test will receive an argument of type `TestModelFixture`.
 * Deletes the model after the test runs.
 */
export const testModelFixture = async (
  request: APIRequestContext,
  runFixture: (returnValue: TestModelFixture) => Promise<void>,
  workspace: Workspace,
  namespace: string,
  modelRequestBody: CreateModelEntityRequest
) => {
  // Create a test model
  const modelsApi = new ModelsAPI(request);
  const testModel = await modelsApi.createModel(namespace, modelRequestBody);

  // Execute test
  await runFixture({
    workspace,
    model: testModel,
  });

  // Clean up the test model
  await modelsApi.deleteModel(testModel.workspace!, testModel.name!);
};

export interface TestDatasetFixture {
  workspace: Workspace;
  dataset: Dataset;
}

/**
 * Common fixture that creates a test dataset to use for an individual test.
 * The test will receive an argument of type `TestDatasetFixture`.
 * Deletes the dataset after the test runs.
 */
export const testDatasetFixture = async (
  request: APIRequestContext,
  runFixture: (returnValue: TestDatasetFixture) => Promise<void>,
  workspace: Workspace,
  datasetName: string,
  datasetNamespace: string,
  datasetDescription: string
) => {
  // Create a test dataset
  const datasetsApi = new DatasetsAPI(request);
  const testDataset = await datasetsApi.createDataset(
    datasetName,
    datasetNamespace,
    workspace.name,
    datasetDescription
  );

  // Execute test
  await runFixture({
    workspace,
    dataset: testDataset,
  });

  // Clean up dataset
  await datasetsApi.deleteDataset(datasetName, datasetNamespace);
};

export interface TestDatasetFilesFixture {
  workspace: Workspace;
  dataset: Dataset;
}

/**
 * Common fixture that uploads a file(s) to the given dataset.
 * The test will receive an argument of type `TestDatasetFilesFixture`.
 */
export const testDatasetFilesFixture = async (
  request: APIRequestContext,
  runFixture: (returnValue: TestDatasetFilesFixture) => Promise<void>,
  workspace: Workspace,
  dataset: Dataset,
  files: {
    // Path to local test file
    testFilePath: string;
    // Folder in the dataset to upload the file
    datasetFolder?: string;
  }[]
) => {
  // Upload the file(s) to the dataset
  const datasetsApi = new DatasetsAPI(request);
  // NOTE: This seems to consistently fail if uploading files in parallel. Specifically, the third call to HF that
  // commits the file fails with a 500. Uploading files sequentially succeeds, so we intentionally do that here.
  for (const file of files) {
    await datasetsApi.uploadFile(dataset, file.testFilePath, file.datasetFolder);
  }

  // Execute test
  await runFixture({
    workspace,
    dataset,
  });
};

export interface TestEvaluationConfigFixture {
  workspace: Workspace;
  evaluationConfig: EvaluationConfig;
}

/**
 * Common fixture that creates an evaluation config.
 * The test will receive an argument of type `TestEvaluationConfigFixture`.
 */

export const testEvaluationConfigFixture = async (
  request: APIRequestContext,
  runFixture: (returnValue: TestEvaluationConfigFixture) => Promise<void>,
  workspace: Workspace,
  evaluationConfigRequestBody: EvaluationConfigInput
) => {
  // Create the evaluation config
  const evaluationsApi = new EvaluationsAPI(request);
  const evaluationConfig = await evaluationsApi.createEvaluationConfig(evaluationConfigRequestBody);

  // Execute test
  await runFixture({
    workspace,
    evaluationConfig,
  });

  // Clean up the evaluation config
  await evaluationsApi.deleteEvaluationConfig(
    String((evaluationConfig as { namespace?: string }).namespace ?? ''),
    String((evaluationConfig as { name?: string }).name ?? '')
  );
};
