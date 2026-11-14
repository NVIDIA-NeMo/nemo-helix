// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PanelAction } from '@studio/components/CreateDatasetPanel/draft';
import { TextField } from '@studio/components/CreateDatasetPanel/TextField';
import type { Dispatch } from 'react';

interface DescriptionFieldProps {
  value: string;
  dispatch: Dispatch<PanelAction>;
}

export function DescriptionField({ value, dispatch }: DescriptionFieldProps) {
  return (
    <TextField
      label="Description"
      value={value}
      onChange={(next) => dispatch({ type: 'description-changed', value: next })}
    />
  );
}
