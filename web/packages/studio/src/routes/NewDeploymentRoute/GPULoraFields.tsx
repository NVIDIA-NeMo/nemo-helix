// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Flex } from '@nvidia/foundations-react-core';
import { WizardFormValues } from '@studio/routes/NewDeploymentRoute/schema';
import { Control, FieldErrors } from 'react-hook-form';

export const GPULoraFields = ({
  control,
  errors,
}: {
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
}) => {
  return (
    <Flex gap="4" align="start" className="w-full">
      <Flex className="flex-1">
        <ControlledTextInput
          useControllerProps={{ control, name: 'gpu' }}
          name="gpu"
          label="GPUs"
          type="number"
          formFieldProps={{
            slotInfo: 'GPU count for this deployment (TP×PP where the engine supports it).',
            slotError: errors.gpu?.message,
          }}
        />
      </Flex>
      <Flex className="flex-1 shrink-0 ">
        <ControlledSwitch
          useControllerProps={{ control, name: 'loraEnabled' }}
          attributes={{ Flex: { justify: 'start' } }}
          formFieldProps={{
            slotLabel: 'LoRA Enabled',
            slotInfo:
              'Serve LoRA adapters alongside this base model. Adapters trained against it are picked up automatically and addressed as “workspace--adapter-name”.',
          }}
        />
      </Flex>
    </Flex>
  );
};
