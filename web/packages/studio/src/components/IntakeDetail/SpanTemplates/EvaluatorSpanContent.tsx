// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asNonEmptyString } from '@nemo/common/src/utils/asString';
import { Badge } from '@nvidia/foundations-react-core';
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
import type { FC } from 'react';

/**
 * EVALUATOR body. Handles both flavors: an LLM-as-judge (model/provider/
 * temperature + a score) and a deterministic code check (pass/fail + test
 * counts). Elevates the verdict and judge metadata from `evaluator.*` raw
 * attributes; the score may also exist as a separate evaluator result row.
 */
export const EvaluatorSpanContent: FC<SpanTemplateContentProps> = ({ span }) => {
  const attributes = parseRawAttributes(span.raw_attributes);
  const name = asNonEmptyString(attributes['evaluator.name']);
  const kind = asNonEmptyString(attributes['evaluator.kind']);
  const score = asNumber(attributes['evaluator.score']);
  const passed = asBoolean(attributes['evaluator.passed']);
  const temperature = asNumber(attributes['evaluator.temperature']);
  const testsPassed = asNumber(attributes['evaluator.tests_passed']);
  const testsTotal = asNumber(attributes['evaluator.tests_total']);

  const fields: TemplateField[] = [
    // Score leads the kind fields so it renders immediately after Status, shown
    // as a badge to set the headline verdict apart from the metadata that follows.
    {
      label: 'Score',
      value:
        score === undefined ? undefined : (
          <Badge color="gray" kind="solid">
            {score.toFixed(3)}
          </Badge>
        ),
    },
    // The span name (e.g. "judge-answer") conveys what the evaluator does; the
    // row header elevates the evaluator's own name, so surface this here.
    { label: 'Name', value: span.name ?? undefined },
    { label: 'Evaluator', value: name },
    { label: 'Type', value: kind },
    { label: 'Result', value: passed === undefined ? undefined : passed ? 'Pass' : 'Fail' },
    {
      label: 'Tests',
      value: testsTotal === undefined ? undefined : `${testsPassed ?? 0} / ${testsTotal}`,
    },
    { label: 'Judge Model', value: span.model ?? undefined },
    { label: 'Provider', value: span.provider ?? undefined },
    { label: 'Temperature', value: temperature?.toString() },
  ];

  return <TemplateKeyValues span={span} fields={fields} />;
};
