// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asNonEmptyString } from '@nemo/common/src/utils/asString';
import { Badge } from '@nvidia/foundations-react-core';
import {
  asBoolean,
  asList,
  asNumber,
  parseRawAttributes,
} from '@studio/components/IntakeDetail/SpanTemplates/rawAttributes';
import {
  TemplateKeyValues,
  type TemplateField,
} from '@studio/components/IntakeDetail/SpanTemplates/templateFields';
import type { SpanTemplateContentProps } from '@studio/components/IntakeDetail/SpanTemplates/types';
import type { FC } from 'react';

/**
 * GUARDRAIL body. Elevates the rail decision (blocked vs allowed), stage,
 * triggered categories, and confidence from `guardrail.*` raw attributes, plus
 * the model for an LLM-based rail.
 */
export const GuardrailSpanContent: FC<SpanTemplateContentProps> = ({ span }) => {
  const attributes = parseRawAttributes(span.raw_attributes);
  const stage = asNonEmptyString(attributes['guardrail.stage']);
  const blocked = asBoolean(attributes['guardrail.blocked']);
  const confidence = asNumber(attributes['guardrail.confidence']);
  const categories = asList(attributes['guardrail.categories']);

  const fields: TemplateField[] = [
    {
      label: 'Decision',
      value:
        blocked === undefined ? undefined : (
          <Badge color={blocked ? 'red' : 'green'} kind="solid">
            {blocked ? 'Blocked' : 'Allowed'}
          </Badge>
        ),
    },
    { label: 'Stage', value: stage },
    { label: 'Confidence', value: confidence?.toFixed(2) },
    { label: 'Categories', value: categories.length ? categories.join(', ') : undefined },
    { label: 'Model', value: span.model ?? undefined },
    { label: 'Provider', value: span.provider ?? undefined },
  ];

  return <TemplateKeyValues span={span} fields={fields} />;
};
