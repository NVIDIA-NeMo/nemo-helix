// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asNonEmptyString } from '@nemo/common/src/utils/asString';
import { parseRawAttributes } from '@studio/components/IntakeDetail/SpanTemplates/rawAttributes';
import {
  TemplateKeyValues,
  type TemplateField,
} from '@studio/components/IntakeDetail/SpanTemplates/templateFields';
import type { SpanTemplateContentProps } from '@studio/components/IntakeDetail/SpanTemplates/types';
import type { FC } from 'react';

/**
 * AGENT body. Elevates the agent identity (name, id, and version when present).
 * The task and result remain in the Input/Output sections.
 */
export const AgentSpanContent: FC<SpanTemplateContentProps> = ({ span }) => {
  const attributes = parseRawAttributes(span.raw_attributes);
  // agent.version is not a typed Span field; surface it from raw when present.
  const version = asNonEmptyString(attributes['agent.version']);

  const fields: TemplateField[] = [
    { label: 'Agent', value: span.agent_name ?? undefined },
    { label: 'Agent ID', value: span.agent_id ?? undefined },
    { label: 'Version', value: version },
  ];

  return <TemplateKeyValues span={span} fields={fields} />;
};
