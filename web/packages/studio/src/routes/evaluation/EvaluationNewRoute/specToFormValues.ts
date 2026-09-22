// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PanelScoreFormData } from '@studio/components/evaluation/Jobs/form/ScoreModal';
import type {
  DatasetEvalSpec,
  InlineMetricBundle,
} from '@studio/components/evaluation/submitEvaluationJob';
import {
  type CanonicalField,
  EMPTY_FIELD_MAPPING,
  EVALUATION_FORM_DEFAULTS,
  type EvaluationFormValues,
  NUMBER_CHECK_OPERATIONS,
  type SelectableMetric,
  STRING_CHECK_OPERATIONS,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';

const asRecord = (value: unknown): Record<string, unknown> =>
  typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {};

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' && value ? value : undefined;

/** The metric object a bundle carries, keyed by its declared type. */
const metricOf = (bundle: InlineMetricBundle): Record<string, unknown> =>
  asRecord(bundle.payload?.metric);

const findMetric = (
  spec: DatasetEvalSpec,
  type: SelectableMetric
): Record<string, unknown> | undefined => {
  const bundle = spec.metrics?.find((entry) => entry.metric_type === type);
  return bundle ? metricOf(bundle) : undefined;
};

/** A stored judge score back into the panel's form shape.
 *
 *  ``scoreType`` is a UI-only discriminator that ``toScorePayload`` drops, so it
 *  is recovered the same way the API infers it: a ``rubric`` key means rubric,
 *  anything else is a range. */
const toPanelScore = (raw: unknown): PanelScoreFormData => {
  const score = asRecord(raw);
  const name = asString(score.name) ?? '';
  const description = asString(score.description);
  if (Array.isArray(score.rubric)) {
    return {
      scoreType: 'rubric',
      name,
      ...(description ? { description } : {}),
      rubric: score.rubric.map((entry) => {
        const level = asRecord(entry);
        const levelDescription = asString(level.description);
        return {
          label: asString(level.label) ?? '',
          ...(levelDescription ? { description: levelDescription } : {}),
          value: Number(level.value),
        };
      }),
    } as PanelScoreFormData;
  }
  return {
    scoreType: 'range',
    name,
    ...(description ? { description } : {}),
    minimum: Number(score.minimum),
    maximum: Number(score.maximum),
  } as PanelScoreFormData;
};

/** Narrow a stored operation back to one the form offers, falling back to the
 *  default rather than seeding a Select with a value it cannot render. */
const toOperation = <T extends readonly string[]>(
  value: unknown,
  allowed: T,
  fallback: T[number]
): T[number] => (allowed.includes(value as string) ? (value as T[number]) : fallback);

/**
 * A stored config back into form values -- the inverse of ``buildEvaluationSpec``.
 *
 * Needed because a reused config is shown in the real form rather than a summary,
 * so every control has to be seeded from what was saved.
 *
 * ``model`` is deliberately absent: the spec never carried a target, because the
 * model under test is chosen per run. The caller keeps whatever the user picked.
 */
export const specToFormValues = (
  spec: DatasetEvalSpec,
  workspace: string
): Omit<EvaluationFormValues, 'configSource' | 'name' | 'model'> => {
  const judge = findMetric(spec, 'llm-judge');
  const stringCheck = findMetric(spec, 'string-check');
  const numberCheck = findMetric(spec, 'number-check');

  const mapping = asRecord(spec.field_mapping);
  const fieldMapping = { ...EMPTY_FIELD_MAPPING };
  for (const field of Object.keys(fieldMapping) as CanonicalField[]) {
    fieldMapping[field] = asString(mapping[field]) ?? '';
  }

  const judgeModelName = asString(asRecord(judge?.model).name);
  const judgeMessages = asRecord(judge?.prompt_template).messages;
  const judgePrompt = Array.isArray(judgeMessages)
    ? (asString(asRecord(judgeMessages[0]).content) ?? null)
    : null;

  const selected = new Set((spec.metrics ?? []).map((bundle) => bundle.metric_type));
  const metrics = { ...EVALUATION_FORM_DEFAULTS.body.metrics };
  for (const type of Object.keys(metrics) as SelectableMetric[]) {
    metrics[type] = selected.has(type);
  }

  const epsilon = numberCheck?.epsilon;

  return {
    dataset: typeof spec.dataset === 'string' ? spec.dataset : null,
    fieldMapping,
    body: {
      metrics,
      scores: Array.isArray(judge?.scores)
        ? judge.scores.map(toPanelScore)
        : EVALUATION_FORM_DEFAULTS.body.scores,
      // Re-qualified: the spec stores the bare model name, while the select
      // holds a ``workspace/name`` ref.
      judgeModel: judgeModelName ? `${workspace}/${judgeModelName}` : '',
      judgePrompt,
      stringCheck: {
        operation: toOperation(
          stringCheck?.operation,
          STRING_CHECK_OPERATIONS,
          EVALUATION_FORM_DEFAULTS.body.stringCheck.operation
        ),
      },
      numberCheck: {
        operation: toOperation(
          numberCheck?.operation,
          NUMBER_CHECK_OPERATIONS,
          EVALUATION_FORM_DEFAULTS.body.numberCheck.operation
        ),
        epsilon: typeof epsilon === 'number' ? epsilon : null,
      },
    },
  };
};
