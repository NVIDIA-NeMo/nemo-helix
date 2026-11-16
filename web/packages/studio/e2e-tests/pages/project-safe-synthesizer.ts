// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Page } from '@playwright/test';

export class ProjectSafeSynthesizerPage {
  constructor(public readonly page: Page) {}

  async goto(projectNamespace: string, projectName: string) {
    await this.page.goto(`projects/${projectNamespace}/${projectName}/safe-synthesizer`);
  }

  async gotoNew(projectNamespace: string, projectName: string) {
    await this.page.goto(`projects/${projectNamespace}/${projectName}/safe-synthesizer/new`);
  }

  async gotoJob(projectNamespace: string, projectName: string, jobId: string) {
    await this.page.goto(
      `projects/${projectNamespace}/${projectName}/safe-synthesizer/job/${jobId}`
    );
  }

  async gotoJobReport(projectNamespace: string, projectName: string, jobId: string) {
    await this.page.goto(
      `projects/${projectNamespace}/${projectName}/safe-synthesizer/job/${jobId}/report`
    );
  }
}
