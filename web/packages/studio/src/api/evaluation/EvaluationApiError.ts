// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HTTPValidationError } from '@nemo/sdk/generated/evals/schema';

export class EvaluationApiError extends Error {
  constructor(public response: HTTPValidationError) {
    const message = Array.isArray(response.detail)
      ? response.detail.map((err: { msg?: string }) => err.msg).join('')
      : response.detail;

    super(message || 'Unknown error');
  }
}
