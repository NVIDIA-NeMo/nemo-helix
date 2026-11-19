// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Stack } from '@nvidia/foundations-react-core';
import type { SafeSynthesizerFormData } from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { useFormContext } from 'react-hook-form';

export const Generation = () => {
  const { control } = useFormContext<SafeSynthesizerFormData>();

  return (
    <Stack gap="density-md">
      <ControlledTextInput
        useControllerProps={{ name: 'spec.config.generation.num_records', control }}
        label="Number of synthetic records to generate"
        type="number"
        min={1}
      />
    </Stack>
  );
};
