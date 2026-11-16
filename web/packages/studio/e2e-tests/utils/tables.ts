// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Locator, Page } from '@playwright/test';

export async function getRowByName(page: Page, name: string): Promise<Locator> {
  return page.getByTestId('nv-table-row').filter({ hasText: name }).first();
}

export async function getNameFromRow(row: Locator, dataField: string = 'name'): Promise<string> {
  const nameElement = row.locator(`[role="gridcell"][data-field="${dataField}"]`);
  const name = await nameElement.textContent();

  if (!name) {
    throw new Error('Could not find name in row');
  }

  return name;
}

export async function deleteRow(row: Locator, dataField: string = 'name'): Promise<void> {
  const name = await getNameFromRow(row, dataField);

  // Delete the project
  await row.getByTestId('quick-actions-menu-trigger').click();
  await row.page().getByTestId('quick-actions-menu-item').getByText('Delete').click();

  // Fill out form and confirm deletion
  await row.page().getByRole('textbox', { name: 'Confirmation' }).fill(name);
  await row.page().getByRole('button', { name: 'Delete' }).click();
}

/**
 * Flips through pages until the element is found
 * @param page - The page to flip through
 * @param elementToFind - The element to find
 * @param timeout - The amount of time to wait for the element to be found before trying the next page
 * @returns True if the element is found, false otherwise
 */
export async function flipPagesUntilFound(
  page: Page,
  elementToFind: Locator,
  /** The amount of time to wait for the element to be found before trying the next page*/
  timeout: number = 5000
) {
  const nextPageButton = page.getByRole('button', { name: 'Go to next page' });

  let found = false;

  while (!found && (await nextPageButton.isEnabled())) {
    try {
      await elementToFind.waitFor({ timeout });
      found = true;
    } catch {
      await nextPageButton.click();
    }
  }

  return found;
}
