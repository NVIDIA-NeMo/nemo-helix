// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  insightsDisableAnalysisConfig,
  insightsEnableAnalysisConfig,
} from '@nemo/sdk/generated/insights/insights-analysis-configs';
import type { AnalysisConfig } from '@nemo/sdk/generated/insights/schema';

/**
 * Raised when the `enable` half of an enable-then-disable save landed but the `disable` half did
 * not. The stored config has already changed, so callers must refresh rather than assume the save
 * was a no-op.
 */
export class AnalysisConfigPartialSaveError extends Error {
  constructor(cause: unknown) {
    super('Models were saved, but analysis could not be turned off and may still be enabled.');
    this.name = 'AnalysisConfigPartialSaveError';
    this.cause = cause;
  }
}

export interface AnalysisConfigDraft {
  enabled: boolean;
  defaultModel: string;
  fastModel: string;
}

/**
 * Persist an edited analysis config over an API that cannot express the edit directly.
 *
 * The service offers `enable` (sets both models and forces `enabled: true`), `disable` (clears the
 * flag alone), and a `PATCH` that carries only `enabled`. Nothing sets models while leaving the
 * flag alone, so "disabled, with these models" has to be written as enable-then-disable — which
 * means a save can leave the config briefly enabled before the second call lands.
 */
export const saveAnalysisConfig = async (
  workspace: string,
  agent: string,
  draft: AnalysisConfigDraft,
  stored: AnalysisConfig | undefined
): Promise<AnalysisConfig> => {
  const modelsChanged =
    draft.defaultModel !== stored?.default_model || draft.fastModel !== stored?.fast_model;

  // No config exists yet, so `enable` is the only call that can create one.
  const mustWriteModels = modelsChanged || !stored;

  if (!mustWriteModels) {
    if (draft.enabled === stored.enabled) return stored;
    return draft.enabled
      ? insightsEnableAnalysisConfig(workspace, agent, {
          default_model: draft.defaultModel,
          fast_model: draft.fastModel,
        })
      : insightsDisableAnalysisConfig(workspace, agent);
  }

  const enabled = await insightsEnableAnalysisConfig(workspace, agent, {
    default_model: draft.defaultModel,
    fast_model: draft.fastModel,
  });

  if (draft.enabled) return enabled;

  try {
    return await insightsDisableAnalysisConfig(workspace, agent);
  } catch (disableError) {
    throw new AnalysisConfigPartialSaveError(disableError);
  }
};
