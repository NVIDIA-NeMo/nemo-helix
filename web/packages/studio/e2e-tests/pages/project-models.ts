// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_BASE_MODEL, LONG_OPERATION_TIMEOUT } from '@e2e-tests/utils/constants';
import { expectToastIsVisible, waitForLongOperation } from '@e2e-tests/utils/pageUtils';
import { getRowByName } from '@e2e-tests/utils/tables';
import { expect, type Page } from '@playwright/test';

export class ProjectModelsPage {
  constructor(public readonly page: Page) {}

  private async openQuickActionsMenu(modelName: string, actionName: string) {
    const modelRow = await getRowByName(this.page, modelName);
    await expect(modelRow).toBeVisible();
    await modelRow.getByTestId('quick-actions-menu-trigger').click();
    await this.page.getByRole('menuitem', { name: actionName }).click();
  }

  private async uploadFewShotExamples(fileInput: string | File) {
    // Open Learning Examples section
    await this.page.getByRole('button', { name: 'Learning Examples' }).click();

    // Navigate to import examples
    await this.page.getByRole('button', { name: 'Import Examples' }).click();
    await this.page.getByRole('tab', { name: 'Upload' }).click();

    // Upload file to dropzone
    if (typeof fileInput === 'string') {
      // Handle file path string
      await this.page.setInputFiles('[data-testid="nv-upload-input-element"]', fileInput);
    } else {
      // Handle File object - convert to Playwright file format
      const fileData = {
        name: fileInput.name,
        mimeType: fileInput.type,
        buffer: Buffer.from(await fileInput.arrayBuffer()),
      };
      await this.page.setInputFiles('[data-testid="nv-upload-input-element"]', fileData);
    }

    // Confirm the upload
    await this.page.getByRole('button', { name: 'Confirm' }).click();
  }

  async goto(projectNamespace: string, projectName: string) {
    await this.page.goto(`projects/${projectNamespace}/${projectName}/models`);
  }

  async createModel({
    modelName,
    baseModelName = DEFAULT_BASE_MODEL,
    description,
    systemPromptTemplate,
    iclFewShotExamples,
    projectNamespace,
  }: {
    modelName: string;
    baseModelName?: string;
    description?: string;
    systemPromptTemplate?: string;
    iclFewShotExamples?: string | File;
    projectNamespace: string;
  }) {
    // Click the button that triggers the modal to open
    await this.page.getByTestId('create-new-model-button').click();
    // Fill out form for model
    await this.page.getByRole('combobox', { name: 'Base Model' }).waitFor({ state: 'visible' });
    await waitForLongOperation(this.page);

    // Wait for paginated model requests to complete
    await expect(this.page.getByText('Loading models...')).not.toBeVisible({
      timeout: LONG_OPERATION_TIMEOUT,
    });
    const combobox = this.page.getByRole('combobox');
    await expect(combobox).toBeEnabled();
    await combobox.click();
    const filterInput = this.page.getByTestId('model-filter');
    await filterInput.waitFor({ state: 'visible' });
    await filterInput.fill(baseModelName);
    const modelOption = this.page.locator('.nv-menu-item-label').first();
    await modelOption.waitFor({ state: 'visible', timeout: LONG_OPERATION_TIMEOUT });
    await modelOption.click();

    if (description) {
      await this.page.getByRole('textbox', { name: 'Description' }).fill(description);
    }

    if (systemPromptTemplate) {
      await this.page
        .getByRole('textbox', { name: 'System Instructions' })
        .fill(systemPromptTemplate);
    }

    if (iclFewShotExamples) {
      await this.uploadFewShotExamples(iclFewShotExamples);
    }

    // Submit button
    await this.page.getByRole('button', { name: 'Save Model' }).click();

    // Enter model name
    await this.page.getByRole('textbox', { name: 'Model Name' }).fill(modelName);

    // Save
    await this.page.getByRole('button', { name: 'Save' }).click();
    await waitForLongOperation(this.page);

    // Navigated to models page
    await expectToastIsVisible(this.page, 'Configuration successfully saved.');

    // Click on new model
    await this.page.getByText(modelName).click();

    // Wait for model to be loaded
    await waitForLongOperation(this.page);

    // Assert that the Side Panel Header text appears on the page
    const chatHeader = this.page.getByText(`${projectNamespace}/${modelName}`);
    await expect(chatHeader).toHaveCount(2, {
      timeout: LONG_OPERATION_TIMEOUT,
    });
  }

  async deleteModel(modelName: string) {
    // Open delete modal
    await this.openQuickActionsMenu(modelName, 'Delete');

    await this.page.getByRole('button', { name: 'Delete Model' }).click();
    await waitForLongOperation(this.page);

    await expectToastIsVisible(this.page, `Deleted model '${modelName}'.`);
  }

  async waitForPageLoad() {
    await waitForLongOperation(this.page);
  }
}
