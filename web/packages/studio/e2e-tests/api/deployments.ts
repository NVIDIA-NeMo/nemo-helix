// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NHX_BASE_URL } from '@e2e-tests/utils/environment';
import { APIRequestContext } from '@playwright/test';

export class DeploymentsAPI {
  constructor(private request: APIRequestContext) {
    this.request = request;
  }

  async deleteDeployment(workspace: string, name: string) {
    await this.request.delete(
      `${NHX_BASE_URL}/apis/models/v2/workspaces/${encodeURIComponent(workspace)}/deployments/${encodeURIComponent(name)}`
    );
  }

  async deleteDeploymentConfig(workspace: string, name: string) {
    await this.request.delete(
      `${NHX_BASE_URL}/apis/models/v2/workspaces/${encodeURIComponent(workspace)}/deployment-configs/${encodeURIComponent(name)}`
    );
  }
}
