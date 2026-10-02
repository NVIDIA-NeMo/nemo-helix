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
  hideLoraToggle = false,
}: {
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
  /**
   * Omit the LoRA switch because the caller has fixed `loraEnabled` itself.
   *
   * For a caller that deploys a base model specifically to serve an adapter,
   * `loraEnabled` is an invariant rather than a preference — turning it off
   * produces a deployment that refuses the adapter. Rendering the switch
   * disabled would still present it as a decision the user might revisit, so it
   * is not rendered at all. The value still travels with the form.
   */
  hideLoraToggle?: boolean;
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
      {!hideLoraToggle && (
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
      )}
    </Flex>
  );
};
