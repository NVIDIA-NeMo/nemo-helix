// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asNonEmptyString } from '@nemo/common/src/utils/asString';
import { parseRawAttributes } from '@studio/components/IntakeDetail/SpanTemplates/rawAttributes';
import {
  TemplateKeyValues,
  type TemplateField,
} from '@studio/components/IntakeDetail/SpanTemplates/templateFields';
import type { SpanTemplateContentProps } from '@studio/components/IntakeDetail/SpanTemplates/types';
import { formatDurationMs, getSpanDurationMs } from '@studio/util/intakeTelemetry';
import type { FC } from 'react';

/**
 * TOOL body. Elevates the tool identity, description, and duration side by side.
 * The call arguments and result remain in the Input/Output sections.
 */
export const ToolSpanContent: FC<SpanTemplateContentProps> = ({ span }) => {
  const attributes = parseRawAttributes(span.raw_attributes);
  const description = asNonEmptyString(attributes['tool.description']);
  const durationMs = getSpanDurationMs(span);

  const fields: TemplateField[] = [
    { label: 'Tool', value: span.tool_name ?? undefined },
    { label: 'Description', value: description },
    {
      label: 'Duration',
      value: durationMs === undefined ? undefined : formatDurationMs(durationMs),
    },
  ];

  return <TemplateKeyValues span={span} fields={fields} />;
};
