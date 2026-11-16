// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Stack, Text, Flex } from '@nvidia/foundations-react-core';
import type { SafeSynthesizerFormData } from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { useFormContext } from 'react-hook-form';

export const DataPreparationSettings = () => {
  const { control } = useFormContext<SafeSynthesizerFormData>();

  return (
    <Stack gap="density-2xl">
      <Text kind="body/regular/md">
        Configure how tabular records are organized for Tabular Fine-Tuning. If you don't have
        event-driven data, you likely don't need to change these settings.
      </Text>

      <Flex direction="row" align="center" gap="density-md" className="w-full">
        <ControlledTextInput
          useControllerProps={{
            name: 'spec.config.data.group_training_examples_by',
            control,
          }}
          label="group_training_examples_by"
          placeholder=""
          formFieldProps={{
            labelPosition: 'left',
            attributes: {
              FormFieldLabelGroup: {
                className: '!w-[210px] justify-between',
              },
              FormFieldHelper: {
                className: '!ml-[220px]',
              },
            },
            slotInfo:
              'Column name used to group related records into a single training example (for event-driven data).',
          }}
        />
      </Flex>

      <Flex direction="row" align="center" gap="density-md" className="w-full">
        <ControlledTextInput
          useControllerProps={{
            name: 'spec.config.data.order_training_examples_by',
            control,
          }}
          label="order_training_examples_by"
          placeholder=""
          formFieldProps={{
            labelPosition: 'left',
            attributes: {
              FormFieldLabelGroup: {
                className: '!w-[210px] justify-between',
              },
              FormFieldHelper: {
                className: '!ml-[220px]',
              },
            },
            slotInfo:
              'Column used to order records within each group (e.g., a timestamp). Requires group_training_examples_by.',
          }}
        />
      </Flex>
    </Stack>
  );
};
