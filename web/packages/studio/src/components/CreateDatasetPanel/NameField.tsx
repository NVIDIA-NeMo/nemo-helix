// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NAME_HELP } from '@studio/components/CreateDatasetPanel/constants';
import type { PanelAction } from '@studio/components/CreateDatasetPanel/draft';
import { TextField } from '@studio/components/CreateDatasetPanel/TextField';
import type { Dispatch } from 'react';

interface NameFieldProps {
  value: string;
  error?: string;
  dispatch: Dispatch<PanelAction>;
}

/** The dataset name input; a manual edit stops name inference for the tab. */
export function NameField({ value, error, dispatch }: NameFieldProps) {
  return (
    <TextField
      label="Name"
      value={value}
      error={error}
      help={NAME_HELP}
      onChange={(next) => dispatch({ type: 'name-changed', value: next })}
      onBlur={() => dispatch({ type: 'name-blurred' })}
    />
  );
}
