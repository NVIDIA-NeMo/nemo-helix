// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asNonEmptyString } from '@nemo/common/src/utils/asString';
import {
  asBoolean,
  asNumber,
  parseRawAttributes,
} from '@studio/components/IntakeDetail/SpanTemplates/rawAttributes';
import {
  TemplateKeyValues,
  type TemplateField,
} from '@studio/components/IntakeDetail/SpanTemplates/templateFields';
import type { SpanTemplateContentProps } from '@studio/components/IntakeDetail/SpanTemplates/types';
import { formatKeyLabel } from '@studio/util/strings';
import type { FC } from 'react';

const formatBoolean = (value: boolean | undefined): string | undefined =>
  value === undefined ? undefined : value ? 'Yes' : 'No';

const formatParamValue = (value: unknown): string | undefined => {
  if (value === null || value === undefined || value === '') return undefined;
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString() : String(value);
  }
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
};

/** Parse the OpenInference `llm.invocation_parameters` JSON blob into a record. */
const parseInvocationParameters = (value: unknown): Record<string, unknown> => {
  const raw = asNonEmptyString(value);
  if (!raw) return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
};

// Discrete `llm.*` sampling keys that some exporters emit instead of (or
// alongside) the invocation_parameters blob.
const DISCRETE_PARAM_KEYS: ReadonlyArray<readonly [attribute: string, param: string]> = [
  ['llm.temperature', 'temperature'],
  ['llm.top_p', 'top_p'],
  ['llm.top_k', 'top_k'],
  ['llm.max_tokens', 'max_tokens'],
];

/**
 * LLM body. Elevates the model identity and every sampling/runtime parameter
 * into the "Model & parameters" section: the OpenInference
 * `llm.invocation_parameters` blob (temperature, top_p, max_tokens, ...) plus
 * any discrete `llm.*` keys, with streaming/cache flags. Token usage and cost
 * stay in the generic Usage section; the messages stay in Input/Output.
 */
export const LlmSpanContent: FC<SpanTemplateContentProps> = ({ span }) => {
  const attributes = parseRawAttributes(span.raw_attributes);

  const params: Record<string, unknown> = {
    ...parseInvocationParameters(attributes['llm.invocation_parameters']),
  };
  // Backfill any sampling key not already present from its discrete attribute.
  for (const [attribute, param] of DISCRETE_PARAM_KEYS) {
    if (params[param] === undefined) {
      const value = asNumber(attributes[attribute]);
      if (value !== undefined) params[param] = value;
    }
  }

  const streaming = asBoolean(attributes['llm.is_streaming']);
  const cacheHit = asBoolean(attributes['llm.cache_hit']);

  const fields: TemplateField[] = [
    { label: 'Model', value: span.model ?? undefined },
    { label: 'Provider', value: span.provider ?? undefined },
    { label: 'Prompt ID', value: span.prompt_id ?? undefined },
    ...Object.entries(params).map(([key, value]) => ({
      label: formatKeyLabel(key),
      value: formatParamValue(value),
    })),
    { label: 'Streaming', value: formatBoolean(streaming) },
    { label: 'Cache Hit', value: formatBoolean(cacheHit) },
  ];

  return <TemplateKeyValues span={span} fields={fields} />;
};
