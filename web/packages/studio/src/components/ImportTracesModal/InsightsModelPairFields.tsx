// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { Button, Flex, FormField, Stack, Text } from '@nvidia/foundations-react-core';
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
  /**
   * The stored pair a blank picker stands for. When given, the pickers start blank, show the
   * stored model as their placeholder, and can be cleared back to it.
   */
  stored?: { defaultModel?: string; fastModel?: string };
}

/**
 * A stored ref is shown even when it names another workspace, so it stays visible rather than
 * reading as an empty dropdown. The dropdown itself only searches {@link workspace}.
 */
const errorFor = (value: string): string | undefined =>
  value.length > 0 && !isQualifiedModelRef(value)
    ? `Stored value "${value}" is not a workspace-qualified Model Entity ID. Pick a model to replace it.`
    : undefined;

const defaultLabel = (ref?: string) =>
  ref ? `Default (${getPartsFromReference(ref).name || ref})` : 'Default';

interface ModelFieldProps {
  workspace: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  /** Present when blank means "use the stored model", so the field can be cleared back to it. */
  storedRef?: string | null;
}

const ModelField: FC<ModelFieldProps> = ({ workspace, label, value, onChange, storedRef }) => {
  const clearable = storedRef !== undefined;
  return (
    <FormField slotLabel={label} slotError={errorFor(value)}>
      <Flex gap="density-sm" align="center">
        <WorkspaceModelSelect
          workspace={workspace}
          value={value ? { model: value } : null}
          onValueChange={({ model }) => onChange(model)}
          placeholder={
            clearable ? defaultLabel(storedRef ?? undefined) : `Select a ${label.toLowerCase()}`
          }
          hideAdapters
          fullWidth
          aria-label={label}
        />
        {clearable && value ? (
          <Button kind="tertiary" size="small" onClick={() => onChange('')}>
            Use default
          </Button>
        ) : null}
      </Flex>
    </FormField>
  );
};

/**
 * Shows the default/fast pair the analysis run will use and lets either half be replaced for this
 * run without editing the stored config. Without `stored`, the pickers are prefilled by the caller.
 */
export const InsightsModelPairFields: FC<InsightsModelPairFieldsProps> = ({
  workspace,
  agent,
  unresolved,
  defaultModel,
  fastModel,
  onDefaultModelChange,
  onFastModelChange,
  stored,
}) => (
  <Stack gap="density-md">
    <Text className="text-secondary" kind="body/regular/xs">
      {unresolved
        ? 'The stored model pair could not be read, so both models are required for this run.'
        : stored
          ? `Leave a model on Default to use the stored analysis config for "${agent}". A choice here applies to this run only.`
          : agent
            ? `Prefilled from the stored analysis config for "${agent}". Changing either one applies to this run only.`
            : 'Applied to every agent in this import, replacing each stored analysis config pair. Leave unset to keep each stored value.'}
    </Text>

    <ModelField
      workspace={workspace}
      label="Default model"
      value={defaultModel}
      onChange={onDefaultModelChange}
      storedRef={stored ? (stored.defaultModel ?? null) : undefined}
    />
    <ModelField
      workspace={workspace}
      label="Fast model"
      value={fastModel}
      onChange={onFastModelChange}
      storedRef={stored ? (stored.fastModel ?? null) : undefined}
    />
  </Stack>
);
