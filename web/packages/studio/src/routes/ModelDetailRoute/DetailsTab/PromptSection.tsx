// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExpandableMessage } from '@nemo/common/src/components/ExpandableMessage';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Stack, Text } from '@nvidia/foundations-react-core';
import {
  getInferenceDefaultFields,
  hasDetailValue,
} from '@studio/routes/ModelDetailRoute/DetailsTab/fields';
import { DetailSection } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/DetailSection';
import { FieldGrid } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/FieldGrid';
import type { FC } from 'react';

/** The prompt and sampling defaults applied to requests; renders nothing when none are set. */
export const PromptSection: FC<{ model: ModelEntity }> = ({ model }) => {
  const messages = [
    { label: 'System Prompt', message: model.prompt?.system_prompt },
    { label: 'Few-shot Examples', message: model.prompt?.icl_few_shot_examples },
  ].filter((item) => item.message);
  const defaults = getInferenceDefaultFields(model.prompt?.inference_params);
  const hasDefaults = defaults.some((field) => hasDetailValue(field.value));

  if (messages.length === 0 && !hasDefaults) {
    return null;
  }

  return (
    <DetailSection heading="Prompt">
      <Stack gap="density-xl">
        {messages.map((item) => (
          <Stack key={item.label} gap="density-xs">
            <Text kind="label/regular/sm" className="text-secondary">
              {item.label}
            </Text>
            <ExpandableMessage message={item.message} characterLimit={300} />
          </Stack>
        ))}
        {hasDefaults && <FieldGrid fields={defaults} />}
      </Stack>
    </DetailSection>
  );
};
