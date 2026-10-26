// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { InfoCircle } from '@nemo/icons/icons';
import { Stack } from '@nvidia/foundations-react-core';
import { SafeSynthesizerFormData } from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { useFormContext } from 'react-hook-form';

export const JobName = () => {
  const { control } = useFormContext<SafeSynthesizerFormData>();

  return (
    <Stack gap="density-md">
      <ControlledTextInput
        useControllerProps={{ name: 'name', control }}
        label="Job Name"
        type="text"
        selectOnFocus
        formFieldProps={{
          slotHelp: (
            <Stack direction="row" gap="density-sm" align="center">
              <InfoCircle />
              You can rename your job at any time
            </Stack>
          ),
        }}
      />
    </Stack>
  );
};
