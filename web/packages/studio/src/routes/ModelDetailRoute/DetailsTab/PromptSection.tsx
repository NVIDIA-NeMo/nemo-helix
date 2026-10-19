// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExpandableMessage } from '@nemo/common/src/components/ExpandableMessage';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { DetailSection } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/DetailSection';
import type { FC } from 'react';

export const PromptSection: FC<{ model: ModelEntity }> = ({ model }) => (
  <DetailSection heading="Prompt">
    <Stack gap="density-xl">
      {[
        { label: 'System Prompt', message: model.prompt?.system_prompt },
        { label: 'Few-shot Examples', message: model.prompt?.icl_few_shot_examples },
      ]
        .filter((item) => item.message)
        .map((item) => (
          <Stack key={item.label} gap="density-xs">
            <Text kind="label/regular/sm" className="text-secondary">
              {item.label}
            </Text>
            <ExpandableMessage message={item.message} characterLimit={300} />
          </Stack>
        ))}
    </Stack>
  </DetailSection>
);
