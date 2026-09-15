/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { MappingFields } from '@nemo/common/src/components/form/MappingFields';
import { Accordion, Stack, Text } from '@nvidia/foundations-react-core';
import { type WizardFormValues } from '@studio/routes/NewDeploymentRoute/schema';
import { FC } from 'react';
import { Control, FieldErrors } from 'react-hook-form';

export type AdvancedSettingsAccordionProps = {
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
  advancedAccordion: string | undefined;
  onAdvancedAccordionChange: (value: string | undefined) => void;
};

export const AdvancedSettingsAccordion: FC<AdvancedSettingsAccordionProps> = ({
  control,
  errors,
  advancedAccordion,
  onAdvancedAccordionChange,
}) => (
  <Accordion
    className="[&>div]:border-b-0"
    onValueChange={onAdvancedAccordionChange}
    items={[
      {
        value: 'advanced',
        chevronPosition: 'start',
        slotTrigger: `${advancedAccordion === 'advanced' ? 'Hide' : 'Show'} Advanced Settings`,
        slotContent: (
          <Stack gap="density-lg" className="pt-density-md">
            <ControlledTextInput
              useControllerProps={{ control, name: 'diskSize' }}
              name="diskSize"
              label={<Text kind="label/bold/md">Disk Size</Text>}
              placeholder="50Gi"
              className="w-[4.5rem] shrink-0"
              formFieldProps={{
                className: 'w-full',
                labelPosition: 'left',
                slotInfo: 'Applies to NGC NIM container deployments only.',
                slotError: errors.diskSize?.message,
                attributes: {
                  FormFieldContentGroup: {
                    className:
                      'w-full min-w-0 [&>:last-child]:flex [&>:last-child]:min-w-0 [&>:last-child]:flex-1 [&>:last-child]:justify-end',
                  },
                },
              }}
            />
            <MappingFields
              control={control}
              name="additionalEnvs"
              keyColumnLabel="Environment Variables"
              valueColumnLabel="Value"
            />
          </Stack>
        ),
      },
    ]}
  />
);
