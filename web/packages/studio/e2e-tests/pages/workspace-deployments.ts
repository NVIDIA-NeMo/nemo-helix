// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { waitForLongOperation } from '@e2e-tests/utils/pageUtils';
import type { Page } from '@playwright/test';

export class WorkspaceDeploymentsPage {
  constructor(public readonly page: Page) {
    this.page = page;
  }

  async gotoDeployments(workspace: string) {
    await this.page.goto(`workspaces/${workspace}/deployments`);
    await waitForLongOperation(this.page);
  }
}
