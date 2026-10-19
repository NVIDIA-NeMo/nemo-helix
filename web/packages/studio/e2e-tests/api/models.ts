// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NHX_BASE_URL } from '@e2e-tests/utils/environment';
import { CreateModelEntityRequest, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { APIRequestContext } from '@playwright/test';

export class ModelsAPI {
  constructor(private request: APIRequestContext) {}

  async createModel(workspace: string, data: CreateModelEntityRequest) {
    const response = await this.request.post(`${NHX_BASE_URL}/v2/workspaces/${workspace}/models`, {
      data,
    });
    const responseData = (await response.json()) as ModelEntity;
    return responseData;
  }

  async deleteModel(workspace: string, name: string) {
    await this.request.delete(`${NHX_BASE_URL}/v2/workspaces/${workspace}/models/${name}`);
  }
}
