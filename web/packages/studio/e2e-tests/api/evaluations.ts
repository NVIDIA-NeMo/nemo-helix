// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NHX_BASE_URL } from '@e2e-tests/utils/environment';
import { APIRequestContext } from '@playwright/test';

/** Evaluation config shape for e2e API. */
type EvaluationConfig = Record<string, unknown>;
/** Evaluation config input for create. */
type EvaluationConfigInput = Record<string, unknown>;

export class EvaluationsAPI {
  constructor(private request: APIRequestContext) {}

  async createEvaluationConfig(data: EvaluationConfigInput) {
    const response = await this.request.post(`${NHX_BASE_URL}/v1/evaluation/configs`, {
      data,
    });

    if (!response.ok()) {
      const errorBody = await response.text();
      throw new Error(
        `Failed to create evaluation config: ${response.status()} ${response.statusText()}\n${errorBody}`
      );
    }

    const responseData = (await response.json()) as EvaluationConfig;
    return responseData;
  }

  async deleteEvaluationConfig(configNamespace: string, configName: string) {
    await this.request.delete(
      `${NHX_BASE_URL}/v1/evaluation/configs/${configNamespace}/${configName}`
    );
  }
}
