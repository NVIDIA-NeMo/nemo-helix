// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, type Page } from '@playwright/test';

/**
 * Page object for the Agent Hardener plugin UI.
 *
 * The plugin mounts under Studio's splat route, so every path is absolute from the workspace — a
 * relative `goto` appends to the current route instead of replacing it.
 */
export class AgentHardenerPage {
  constructor(
    public readonly page: Page,
    private readonly workspace = 'default'
  ) {}

  private base() {
    return `workspaces/${this.workspace}/plugin/agent-hardener`;
  }

  async gotoRuns() {
    await this.page.goto(this.base());
  }

  async gotoManifests() {
    await this.page.goto(`${this.base()}/manifests`);
  }

  async gotoNewManifest() {
    await this.page.goto(`${this.base()}/manifests/new`);
  }

  async gotoManifest(name: string) {
    await this.page.goto(`${this.base()}/manifests/${encodeURIComponent(name)}`);
  }

  async gotoRun(name: string) {
    await this.page.goto(`${this.base()}/${encodeURIComponent(name)}`);
  }

  /** The nav entry Studio renders from the plugin's `navItems`. */
  navItem() {
    return this.page.getByRole('link', { name: 'Agent Hardener' });
  }

  /**
   * Wait for a DataView to settle.
   *
   * Asserting on rows immediately races the query: an empty table and a loading table look the same
   * to a locator, so a passing assertion would prove nothing.
   */
  async waitForTable() {
    await expect(
      this.page.getByRole('table').or(this.page.getByText(/no .*(found|yet)/i))
    ).toBeVisible({ timeout: 30_000 });
  }

  /** Rows that open a record. The header row has no such button, so this is the data rows. */
  dataRows() {
    return this.page
      .getByRole('row')
      .filter({ has: this.page.getByRole('button', { name: 'Open row' }) });
  }

  /**
   * Open the first record, or return false when the target genuinely holds none.
   *
   * The header renders before the query resolves, so counting rows straight after the table appears
   * races the fetch and reports "empty" for a table that is merely still loading — which would skip
   * the test and look like a pass.
   */
  async openFirstRow(): Promise<boolean> {
    const first = this.dataRows().first();
    try {
      await expect(first).toBeVisible({ timeout: 15_000 });
    } catch {
      return false;
    }
    await first.click();
    return true;
  }

  sourceToggle() {
    return this.page.getByTestId('manifest-source');
  }

  /** Choose where the victim comes from on the create form. */
  async chooseSource(source: 'Registered agent' | 'Bring your own') {
    await this.sourceToggle().getByText(source).click();
  }
}
