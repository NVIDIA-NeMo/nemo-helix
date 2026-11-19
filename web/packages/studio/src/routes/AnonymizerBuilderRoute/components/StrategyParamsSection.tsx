// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledCheckbox } from '@nemo/common/src/components/form/ControlledCheckbox';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Stack } from '@nvidia/foundations-react-core';
import { RewriteParamsSection } from '@studio/routes/AnonymizerBuilderRoute/components/RewriteParamsSection';
import {
  HASH_ALGORITHM_OPTIONS,
  STRATEGY_ANNOTATE,
  STRATEGY_HASH,
  STRATEGY_REDACT,
  STRATEGY_REWRITE,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import type { FC } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export const StrategyParamsSection: FC = () => {
  const { control } = useFormContext<AnonymizerFormData>();
  const strategy = useWatch({ control, name: 'strategy' });

  if (strategy === STRATEGY_REDACT) {
    return (
      <Stack gap="density-lg">
        <ControlledTextInput
          useControllerProps={{ name: 'redactTemplate', control }}
          formFieldProps={{
            slotLabel: 'Format Template',
            slotInfo: 'Supports an optional {label} placeholder.',
          }}
        />
        <ControlledCheckbox
          useControllerProps={{ name: 'redactNormalizeLabel', control }}
          formFieldProps={{ slotLabel: 'Normalize and clean the label before substitution' }}
        />
      </Stack>
    );
  }

  if (strategy === STRATEGY_ANNOTATE) {
    return (
      <ControlledTextInput
        useControllerProps={{ name: 'annotateTemplate', control }}
        formFieldProps={{
          slotLabel: 'Format Template',
          slotInfo: 'Requires {text} and {label} placeholders.',
        }}
      />
    );
  }

  if (strategy === STRATEGY_REWRITE) {
    return <RewriteParamsSection />;
  }

  if (strategy === STRATEGY_HASH) {
    return (
      <Stack gap="density-lg">
        <ControlledSelect
          aria-label="Hash algorithm"
          items={HASH_ALGORITHM_OPTIONS}
          useControllerProps={{ name: 'hashAlgorithm', control }}
          formFieldProps={{ slotLabel: 'Algorithm' }}
        />
        <ControlledTextInput
          type="number"
          min={6}
          max={64}
          useControllerProps={{ name: 'hashDigestLength', control }}
          formFieldProps={{
            slotLabel: 'Digest Length',
            slotInfo: 'Number of hex characters to keep (6–64).',
          }}
        />
        <ControlledTextInput
          useControllerProps={{ name: 'hashTemplate', control }}
          formFieldProps={{
            slotLabel: 'Format Template',
            slotInfo: 'Requires {digest}; {label} is optional.',
          }}
        />
      </Stack>
    );
  }

  return null;
};
