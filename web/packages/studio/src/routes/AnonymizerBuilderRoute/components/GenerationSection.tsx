// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { StrategyParamsSection } from '@studio/routes/AnonymizerBuilderRoute/components/StrategyParamsSection';
import {
  MAX_PREVIEW_ROWS,
  STRATEGY_DESCRIPTIONS,
  STRATEGY_OPTIONS,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import type { FC } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export const GenerationSection: FC = () => {
  const { control } = useFormContext<AnonymizerFormData>();
  const strategy = useWatch({ control, name: 'strategy' });

  return (
    <Stack gap="density-lg">
      <Text kind="label/bold/lg">Generation</Text>
      <ControlledSelect
        aria-label="Anonymization strategy"
        items={STRATEGY_OPTIONS}
        useControllerProps={{ name: 'strategy', control }}
        formFieldProps={{ slotLabel: 'Anonymization Strategy', required: true }}
      />
      <Text kind="body/regular/md">{STRATEGY_DESCRIPTIONS[strategy]}</Text>
      <StrategyParamsSection />
      <ControlledTextInput
        type="number"
        min={1}
        max={MAX_PREVIEW_ROWS}
        useControllerProps={{ name: 'previewRows', control }}
        formFieldProps={{
          slotLabel: 'Preview Rows',
          slotInfo: 'Number of records to anonymize when running a preview.',
        }}
      />
    </Stack>
  );
};
