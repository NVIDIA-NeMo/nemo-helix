// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  asNumber,
  parseRawAttributes,
} from '@studio/components/IntakeDetail/SpanTemplates/rawAttributes';
import {
  TemplateKeyValues,
  type TemplateField,
} from '@studio/components/IntakeDetail/SpanTemplates/templateFields';
import type { SpanTemplateContentProps } from '@studio/components/IntakeDetail/SpanTemplates/types';
import { formatDurationMs } from '@studio/util/intakeTelemetry';
import type { FC } from 'react';

/**
 * CHAIN body. A chain is a structural/orchestration step; its input and output
 * carry the substance (shown in those sections). Here we surface the step count
 * when the producer recorded one (`chain.step_count`).
 */
export const ChainSpanContent: FC<SpanTemplateContentProps> = ({ span }) => {
  const attributes = parseRawAttributes(span.raw_attributes);
  const stepCount = asNumber(attributes['chain.step_count']);

  const fields: TemplateField[] = [{ label: 'Steps', value: stepCount?.toLocaleString() }];
  if (span.source === 'gym') {
    const recordedDuration = asNumber(attributes['gym.observed_duration_ms']);
    if (recordedDuration !== undefined && recordedDuration >= 0) {
      fields.push({
        label: 'Recorded rollout duration',
        value: formatDurationMs(recordedDuration),
      });
    }
    if (attributes['gym.timing'] === 'observed_child_window') {
      fields.push({ label: 'Timing basis', value: 'Observed child operations' });
    }
  }

  return <TemplateKeyValues span={span} fields={fields} />;
};
