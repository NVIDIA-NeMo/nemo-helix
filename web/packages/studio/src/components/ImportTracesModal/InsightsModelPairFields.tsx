// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2';
import { FormField, Stack, Text } from '@nvidia/foundations-react-core';
import { isQualifiedModelRef } from '@studio/api/insightsAnalysis';
import type { FC } from 'react';

export interface InsightsModelPairFieldsProps {
  /** Workspace whose model catalogue the dropdowns search. */
  workspace: string;
  /** The agent whose stored config prefilled the pair, when exactly one is known. */
  agent?: string;
  /** True once the stored config lookup settled without producing a config. */
  unresolved: boolean;
  defaultModel: string;
  fastModel: string;
  onDefaultModelChange: (value: string) => void;
  onFastModelChange: (value: string) => void;
}

/**
 * A stored ref is shown even when it names another workspace, so it stays visible rather than
 * reading as an empty dropdown. The dropdown itself only searches {@link workspace}.
 */
const errorFor = (value: string): string | undefined =>
  value.length > 0 && !isQualifiedModelRef(value)
    ? `Stored value "${value}" is not a workspace-qualified Model Entity ID. Pick a model to replace it.`
    : undefined;

/**
 * Shows the default/fast pair the analyze-job will use, prefilled from the agent's stored
 * AnalysisConfig, and lets either half be replaced for this run without editing the stored config.
 */
export const InsightsModelPairFields: FC<InsightsModelPairFieldsProps> = ({
  workspace,
  agent,
  unresolved,
  defaultModel,
  fastModel,
  onDefaultModelChange,
  onFastModelChange,
}) => (
  <Stack gap="density-md">
    <Text kind="body/regular/xs" color="secondary">
      {unresolved
        ? 'The stored model pair could not be read, so both models are required for this run.'
        : agent
          ? `Prefilled from the stored analysis config for "${agent}". Changing either one applies to this run only.`
          : 'Applied to every agent in this import, replacing each stored analysis config pair. Leave unset to keep each stored value.'}
    </Text>

    <FormField slotLabel="Default model" slotError={errorFor(defaultModel)}>
      <WorkspaceModelSelect
        workspace={workspace}
        value={defaultModel ? { model: defaultModel } : null}
        onValueChange={({ model }) => onDefaultModelChange(model)}
        placeholder="Select a default model"
        hideAdapters
        fullWidth
        aria-label="Default model"
      />
    </FormField>

    <FormField slotLabel="Fast model" slotError={errorFor(fastModel)}>
      <WorkspaceModelSelect
        workspace={workspace}
        value={fastModel ? { model: fastModel } : null}
        onValueChange={({ model }) => onFastModelChange(model)}
        placeholder="Select a fast model"
        hideAdapters
        fullWidth
        aria-label="Fast model"
      />
    </FormField>
  </Stack>
);
