// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LONG_OPERATION_TIMEOUT, MOCKS_DIR } from '@e2e-tests/utils/constants';
import { expectToastIsVisible, waitForLongOperation } from '@e2e-tests/utils/pageUtils';
import { getRowByName } from '@e2e-tests/utils/tables';
import { expect, type Page } from '@playwright/test';
import path from 'path';

/** Dataset shape for e2e page (files_url, name, etc.). */
type Dataset = { files_url?: string; name?: string; [key: string]: unknown };
export class ProjectDatasetsPage {
  constructor(public readonly page: Page) {}

  private async openQuickActionsMenu(name: string, actionName: string) {
    const fileRow = await getRowByName(this.page, name);
    await waitForLongOperation(this.page);

    await expect(fileRow).toBeVisible();
    await fileRow.getByTestId('quick-actions-menu-trigger').click();
    await this.page.getByRole('menuitem', { name: actionName }).click();
  }

  async goto(projectNamespace: string, projectName: string, dataset?: Dataset) {
    const datasetFullName = dataset ? `${dataset.namespace}/${dataset.name}` : '';
    const datasetPathParam = datasetFullName ? `/${encodeURIComponent(datasetFullName)}` : '';
    try {
      await this.page.goto(
        `projects/${projectNamespace}/${projectName}/datasets${datasetPathParam}`
      );
      // If given a dataset, open the dataset side panel
      if (dataset) {
        // Wait for the side panel animation to finish
        await expect(
          this.page.locator('[data-testid="nv-side-panel-content"][data-state="open"]')
        ).toBeVisible();
      }
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
    }
  }

  async deleteDataset(name: string) {
    await this.openQuickActionsMenu(name, 'Delete');

    const dialog = this.page.getByRole('dialog');
    const confirmDeleteButton = dialog.getByRole('button').filter({ hasText: 'Delete' });
    await confirmDeleteButton.click();
    await waitForLongOperation(this.page);

    await expectToastIsVisible(this.page, `Deleted dataset '${name}'.`);
  }

  /**
   * Uploads a file to the selected dataset.
   *
   * @param datasetName Dataset name that we'll upload the file to
   * @param testFilePath Path to test file in `MOCKS_DIR` that we'll upload to the dataset
   */
  async uploadFileToDataset(testFilePath: string) {
    const filePath = path.join(MOCKS_DIR, testFilePath);

    // Upload file directly via the file input
    await this.page.getByTestId('dataset-file-dropzone-input').setInputFiles(filePath);

    // Once success message is visible, wait for the page to load
    await waitForLongOperation(this.page);

    // After uploading, expect the dataset to be actively selected
    await expect(this.page.getByTestId('dataset-breadcrumbs')).toBeVisible();

    // Extract the actual file name from the given path, and ensure it exists in the folder
    const fileNameParts = testFilePath.split('/');
    const fileName = fileNameParts.at(-1) || '';
    await expect(this.page.getByText(fileName)).toBeVisible({ timeout: LONG_OPERATION_TIMEOUT });
  }

  /**
   * Deletes a file from the selected dataset.
   *
   * @param filePath Path to file in the selected dataset
   */
  async deleteFileFromDataset(filePath: string) {
    // If we uploaded into a folder, click on the folder first
    const fileNameParts = filePath.split('/');
    const folder = fileNameParts.length > 1 ? fileNameParts[0] : '';
    if (folder) {
      const folderItem = this.page.getByText(folder);
      await folderItem.click();
    }

    const fileName = fileNameParts.length > 1 ? fileNameParts.at(-1) : filePath;
    await this.openQuickActionsMenu(fileName!, 'Delete');

    // Click delete and verify file no longer appears in table
    const dialog = this.page.getByRole('dialog');
    const confirmDeleteButton = dialog.getByRole('button').filter({ hasText: 'Delete' });
    await confirmDeleteButton.click();
    await waitForLongOperation(this.page);

    await expectToastIsVisible(this.page, `Deleted file '${filePath}'.`);
    await waitForLongOperation(this.page);
    await expect(this.page.getByText(fileName!)).not.toBeVisible();
  }

  /**
   * Renames a file in the selected dataset.
   *
   * @param oldFilePath Current path of the file in the dataset
   * @param newFileName New name for the file
   */
  async renameFileInDataset(oldFilePath: string, newFileName: string) {
    // If we're in a folder, click on the folder first
    const fileNameParts = oldFilePath.split('/');
    const folder = fileNameParts.length > 1 ? fileNameParts[0] : '';
    const newFolder = newFileName.split('/').length > 1 ? newFileName.split('/')[0] : '';
    if (folder) {
      const datasetBreadcrumb = this.page.getByTestId('dataset-breadcrumbs');
      await datasetBreadcrumb.click();
      const folderItem = this.page.getByText(folder);
      await folderItem.click();
    }

    const oldFileName = fileNameParts.length > 1 ? fileNameParts.at(-1) : oldFilePath;
    await this.openQuickActionsMenu(oldFileName!, 'Rename');

    // Verify the rename modal is open and prefilled
    const dialog = this.page.getByRole('dialog');
    const nameInput = dialog.getByRole('textbox', { name: 'name' });
    await expect(nameInput).toHaveValue(oldFilePath);

    // Enter new name and submit
    await nameInput.fill(newFileName);
    await dialog.getByRole('button', { name: 'Update File' }).click();
    await waitForLongOperation(this.page);

    // Verify success message and new filename is visible
    await expectToastIsVisible(this.page, 'File successfully saved.');
    await waitForLongOperation(this.page);

    await expect(this.page.getByText(oldFileName!)).not.toBeVisible();
    // If we're in a folder or have a new folder, handle navigation
    if (folder || newFolder) {
      const datasetBreadcrumb = this.page.getByTestId('dataset-breadcrumbs');
      await datasetBreadcrumb.click();
      if (newFolder) {
        const newFolderItem = this.page.getByText(newFolder);
        await newFolderItem.click();
      }
      await expect(this.page.getByText(newFileName.split('/').at(-1)!)).toBeVisible();
    } else {
      await expect(this.page.getByText(newFileName.split('/').at(-1)!)).toBeVisible();
    }
  }

  async waitForPageLoad() {
    await waitForLongOperation(this.page);
  }
}
