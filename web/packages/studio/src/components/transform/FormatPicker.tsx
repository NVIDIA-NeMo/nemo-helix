// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { Label, RadioGroupRoot, Stack, Text } from '@nvidia/foundations-react-core';
import { OUTPUT_FORMATS } from '@studio/components/transform/formats';
import type { TransformMapping } from '@studio/components/transform/useTransformMapping';
import { type FC } from 'react';

interface Props {
  mapping: TransformMapping;
  label?: string;
}

/** Target-format selector: the first decision of every transform. */
export const FormatPicker: FC<Props> = ({ mapping, label = 'Target format' }) => (
  <Stack gap="density-sm">
    <Label className="font-bold">{label}</Label>
    <RadioGroupRoot
      name="transform-format"
      value={mapping.format.id}
      onValueChange={mapping.setFormat}
      className="w-full"
    >
      <div className="grid grid-cols-3 gap-density-md">
        {OUTPUT_FORMATS.map((option) => (
          <RadioCard
            key={option.id}
            value={option.id}
            label={<Text kind="body/bold/md">{option.label}</Text>}
            description={
              <Text className="text-secondary" kind="body/regular/sm">
                {option.description}
              </Text>
            }
            labelSide="left"
          />
        ))}
      </div>
    </RadioGroupRoot>
  </Stack>
);
