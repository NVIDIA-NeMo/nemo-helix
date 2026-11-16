// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asNonEmptyString } from '@nemo/common/src/utils/asString';
import {
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
 * EMBEDDING body. Elevates the embedding model, vector dimensions
 * (`embedding.dimensions`), token usage, and the embedded text.
 */
export const EmbeddingSpanContent: FC<SpanTemplateContentProps> = ({ span }) => {
  const attributes = parseRawAttributes(span.raw_attributes);
  const dimensions = asNumber(attributes['embedding.dimensions']);
  const embeddedText =
    asNonEmptyString(attributes['embedding.embeddings.0.embedding.text']) ?? span.input?.trim();

  const fields: TemplateField[] = [
    { label: 'Model', value: span.model ?? undefined },
    { label: 'Provider', value: span.provider ?? undefined },
    { label: 'Dimensions', value: dimensions?.toLocaleString() },
    { label: 'Input Tokens', value: span.input_tokens?.toLocaleString() },
    { label: 'Total Tokens', value: span.total_tokens?.toLocaleString() },
    { label: 'Embedded text', value: embeddedText || undefined },
  ];

  return <TemplateKeyValues span={span} fields={fields} />;
};
