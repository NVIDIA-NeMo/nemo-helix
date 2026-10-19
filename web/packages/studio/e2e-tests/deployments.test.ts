// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeploymentsAPI } from '@e2e-tests/api/deployments';
import { WorkspaceDeploymentsPage } from '@e2e-tests/pages/workspace-deployments';
import { generateShortTestResourceName } from '@e2e-tests/utils/constants';
import { disableAuthForTest, waitForLongOperation } from '@e2e-tests/utils/pageUtils';
import { expect, test as baseTest } from '@playwright/test';

const WORKSPACE = 'default';

// NGC NIM image used by the create flow. Image pulls in the cluster may take time; the test only
// verifies the deployment record is created and visible — it does not wait for "Ready".
const NGC_IMAGE_NAME = 'nvcr.io/nim/meta/llama-3.2-1b-instruct';
const NGC_IMAGE_TAG = 'latest';

interface DeploymentTracker {
  /** Register a deployment + config pair for guaranteed cleanup after the test. */
  track: (deploymentName: string, configName: string) => void;
}

interface TestFixtures {
  deploymentsPage: WorkspaceDeploymentsPage;
  trackedDeployments: DeploymentTracker;
}

const test = baseTest.extend<TestFixtures>({
  deploymentsPage: async ({ page }, runFixture) => {
    await runFixture(new WorkspaceDeploymentsPage(page));
  },
  // Fixture teardown runs even when the test fails, so anything registered here
  // is cleaned up regardless of where in the UI flow we threw.
  trackedDeployments: async ({ request }, runFixture) => {
    const api = new DeploymentsAPI(request);
    const tracked: Array<{ deploymentName: string; configName: string }> = [];

    await runFixture({
      track: (deploymentName, configName) => {
        tracked.push({ deploymentName, configName });
      },
    });

    for (const { deploymentName, configName } of tracked) {
      try {
        await api.deleteDeployment(WORKSPACE, deploymentName);
      } catch {
        /* already deleted */
      }
      try {
        await api.deleteDeploymentConfig(WORKSPACE, configName);
      } catch {
        /* already deleted or still in use */
      }
    }
  },
});

test.describe('Model Deployments', () => {
  test.beforeEach(() => {
    test.skip(
      !process.env.RECORD,
      'Deployments feature-flagged off + needs NGC infra; run via pnpm test:e2e:record'
    );
  });
  test.beforeEach(async ({ page }) => disableAuthForTest(page));

  test('Creates an NGC deployment, views its details, and deletes it @record', async ({
    page,
    deploymentsPage,
    trackedDeployments,
  }) => {
    test.slow();

    // Base name is what the user types in the wizard. The API resources become:
    //   deployment:    <baseName>-deployment
    //   config:        <baseName>-config
    const baseName = generateShortTestResourceName();
    const deploymentName = `${baseName}-deployment`;
    const configName = `${baseName}-config`;

    // Register for fixture-teardown cleanup before we touch the UI, so a failure
    // anywhere below still leaves the workspace clean.
    trackedDeployments.track(deploymentName, configName);

    await test.step('Navigate to the workspace deployments page', async () => {
      await deploymentsPage.gotoDeployments(WORKSPACE);
      await expect(page.getByRole('button', { name: 'Create Deployment' }).first()).toBeVisible();
    });

    await test.step('Open Create Deployment side panel', async () => {
      await page.getByRole('button', { name: 'Create Deployment' }).first().click();

      // The Deploy submit button only renders inside the open panel.
      await expect(page.getByRole('button', { name: 'Deploy', exact: true })).toBeVisible();
    });

    await test.step('Fill the NGC NIM Container form', async () => {
      // Select the source explicitly rather than relying on which one the wizard
      // preselects — this test covers the NGC path, not the default.
      await page.getByRole('radio', { name: 'NGC NIM Container' }).click();

      const nameField = page.getByRole('textbox', { name: 'Name', exact: true });
      // The wizard pre-fills a generated name; clear it before typing.
      await nameField.fill(baseName);

      await page.getByRole('textbox', { name: 'Image Name' }).fill(NGC_IMAGE_NAME);
      await page.getByRole('textbox', { name: 'Image Tag' }).fill(NGC_IMAGE_TAG);
    });

    await test.step('Submit the deployment', async () => {
      await page.getByRole('button', { name: 'Deploy', exact: true }).click();

      // The side panel closes after a successful submit. Use the dialog role so the loading
      // spinner (which temporarily hides the "Deploy" label) doesn't false-positive.
      const panel = page.getByRole('dialog', { name: 'Create Deployment' });
      await expect(panel).toBeHidden({ timeout: 60_000 });
      await waitForLongOperation(page);
    });

    await test.step('Verify deployment row appears in the list', async () => {
      const deploymentRow = page.getByRole('row', { name: new RegExp(deploymentName) });
      await expect(deploymentRow).toBeVisible({ timeout: 30_000 });
    });

    await test.step('Open the deployment details side panel', async () => {
      await page.getByRole('row', { name: new RegExp(deploymentName) }).click();

      // URL transitions to /deployments/<name>/details.
      await page.waitForURL(new RegExp(`/deployments/${deploymentName}/details`));

      // The details panel renders the deployment name and a Delete button in its footer.
      await expect(page.getByText(deploymentName).first()).toBeVisible();
      await expect(page.getByRole('button', { name: 'Delete', exact: true })).toBeVisible();
    });

    await test.step('Trigger delete from the details panel', async () => {
      await page.getByRole('button', { name: 'Delete', exact: true }).click();

      // Confirmation modal — scope to the dialog by name so we don't collide with the
      // details panel's "Delete" button, which stays mounted behind the modal.
      const modal = page.getByRole('dialog', { name: `Delete deployment: ${deploymentName}` });
      await expect(modal).toBeVisible();

      // DeleteConfirmationModal overrides submitButtonText to "Delete".
      await modal.getByRole('button', { name: 'Delete', exact: true }).click();
      await expect(modal).toBeHidden({ timeout: 30_000 });
    });

    await test.step('Verify deletion started', async () => {
      // After delete, the URL returns to the deployments list. The row should still be present
      // showing Deleting / Deleted status (terminal cleanup happens asynchronously in the cluster).
      await page.waitForURL(/\/deployments$/, { timeout: 10_000 });
      await waitForLongOperation(page);

      const deploymentRow = page.getByRole('row', { name: new RegExp(deploymentName) });
      try {
        await expect(deploymentRow.getByText(/Deleting|Deleted/i)).toBeVisible({ timeout: 30_000 });
      } catch {
        // Row removed altogether is also acceptable.
        await expect(deploymentRow).toHaveCount(0);
      }
    });
  });
});
