// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { listEvaluationSessions } from '@nemo/sdk/generated/platform/experiments';
import type {
  EvaluationSessionResponsesPage,
  ListEvaluationSessionsParams,
} from '@nemo/sdk/generated/platform/schema';
import { isAxiosError } from 'axios';

const isUnsupportedModeError = (error: unknown): boolean => {
  if (!isAxiosError(error)) return false;
  if (error.response?.status !== 400 && error.response?.status !== 422) return false;
  const detail = (error.response.data as { detail?: unknown } | undefined)?.detail;
  return typeof detail === 'string' && detail === 'Unsupported query parameter(s): mode';
};

export const listEvaluationSessionsWithModeFallback = async (
  workspace: string,
  evaluationName: string,
  params: ListEvaluationSessionsParams,
  signal?: AbortSignal
): Promise<EvaluationSessionResponsesPage> => {
  try {
    return await listEvaluationSessions(workspace, evaluationName, params, signal);
  } catch (error) {
    if (!params.mode || !isUnsupportedModeError(error)) {
      throw error;
    }
    const fallbackParams = { ...params };
    delete fallbackParams.mode;
    return listEvaluationSessions(workspace, evaluationName, fallbackParams, signal);
  }
};
