// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2';
import { FormField, Stack } from '@nvidia/foundations-react-core';
import { isQualifiedModelRef } from '@studio/api/insightsAnalysis';
import { InsightsModelWarning } from '@studio/components/ImportTracesModal/InsightsModelWarning';
import { isLightningModel } from '@studio/components/ImportTracesModal/modelGuidance';
import type { FC } from 'react';

interface InsightsModelFieldProps {
  workspace: string;
  label: string;
  help?: string;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

/** One half of the insights default/fast model pair, flagging refs that are not workspace-qualified. */
export const InsightsModelField: FC<InsightsModelFieldProps> = ({
  workspace,
  label,
  help,
  value,
  onChange,
  disabled,
}) => (
  <Stack gap="density-sm">
    <FormField
      slotLabel={label}
      slotHelp={help}
      status={value && !isQualifiedModelRef(value) ? 'error' : undefined}
      slotError={`"${value}" is not a workspace-qualified Model Entity ID. Pick a model to replace it.`}
    >
      <WorkspaceModelSelect
        workspace={workspace}
        value={value ? { model: value } : null}
        onValueChange={({ model }) => onChange(model)}
        placeholder={`Select a ${label.toLowerCase()}`}
        disabled={disabled}
        hideAdapters
        fullWidth
        aria-label={label}
      />
    </FormField>
    {isLightningModel(value) ? <InsightsModelWarning /> : null}
  </Stack>
);
