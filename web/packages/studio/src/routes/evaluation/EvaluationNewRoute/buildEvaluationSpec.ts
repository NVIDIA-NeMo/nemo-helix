// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PanelScoreFormData } from '@studio/components/evaluation/Jobs/form/ScoreModal';
import type {
  DatasetEvalSpec,
  InlineMetricBundle,
} from '@studio/components/evaluation/submitEvaluationJob';
import {
  composeGenerationPrompt,
  composeJudgePromptTemplate,
  type EvaluationFormValues,
  renderScoreGuidance,
  type SelectableMetric,
  selectedMetrics,
  type DatasetBindings,
  toFieldMapping,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { getModelInferenceGatewayUrl } from '@studio/util/models';

/** The bare model id the endpoint expects, without its workspace prefix. */
const bareModelName = (modelRef: string): string =>
  modelRef.includes('/') ? modelRef.split('/').slice(1).join('/') : modelRef;

/** ``MetricOutputSpec.continuous_score`` / ``.label`` as the API serializes them.
 *  Mirrored rather than derived: the wire contract requires a non-empty
 *  ``outputs`` list on every bundle, and these two shapes cover every metric this
 *  form can build. */
const CONTINUOUS_SCORE_SCHEMA = {
  description: 'Continuous numeric metric value.',
  title: 'ContinuousScore',
  type: 'number',
} as const;

const LABEL_SCHEMA = {
  description: 'String label metric value.',
  title: 'Label',
  type: 'string',
} as const;

type BundleOutput = {
  name: string;
  description: string | null;
  value_json_schema: Record<string, unknown>;
};

const continuousOutput = (name: string, description?: string | null): BundleOutput => ({
  name,
  description: description ?? null,
  value_json_schema: { ...CONTINUOUS_SCORE_SCHEMA },
});

const labelOutput = (name: string, description: string): BundleOutput => ({
  name,
  description,
  value_json_schema: { ...LABEL_SCHEMA },
});

const bundle = (
  metricType: SelectableMetric,
  metric: Record<string, unknown>,
  outputs: BundleOutput[],
  description: string
): InlineMetricBundle => ({
  bundle_kind: 'metric-bundle',
  bundle_format_version: 'v1',
  metric_type: metricType,
  metadata: { description, labels: {} },
  outputs,
  secrets: {},
  payload: { kind: 'inline', metric: { type: metricType, ...metric } },
});

/** A score as the metric payload wants it -- the UI-only ``scoreType``
 *  discriminator is dropped; the API infers rubric vs range from the fields. */
const toScorePayload = (score: PanelScoreFormData): Record<string, unknown> =>
  score.scoreType === 'range'
    ? {
        name: score.name,
        ...(score.description ? { description: score.description } : {}),
        minimum: score.minimum,
        maximum: score.maximum,
      }
    : {
        name: score.name,
        ...(score.description ? { description: score.description } : {}),
        rubric: score.rubric.map((level) => ({
          label: level.label,
          ...(level.description ? { description: level.description } : {}),
          value: level.value,
        })),
      };

/** The model's own answer. Omitted where the API allows it (BLEU, ROUGE, F1,
 *  Exact Match all document "if not provided, the output text from the model is
 *  used"), and stated explicitly where it does not (the check metrics require
 *  both templates).
 *
 *  For those check metrics it is the LEFT operand, because every asymmetric
 *  operation reads left-to-right with the answer as the subject:
 *  ``string_check.py`` computes ``right_value in left_value`` for "contains" and
 *  ``left_value.startswith(right_value)``, and ``number_check.py`` computes
 *  ``left_number > right_number``. Ground truth is the right operand -- the
 *  needle, the bound -- so "contains" asks whether the answer contains the
 *  ground truth. */
const OUTPUT_TEMPLATE = '{{sample.output_text}}';

const buildMetricBundle = (
  metric: SelectableMetric,
  values: EvaluationFormValues,
  bindings: DatasetBindings,
  workspace: string
): InlineMetricBundle | null => {
  const reference = bindings.reference;

  switch (metric) {
    case 'exact-match':
    case 'f1':
    case 'rouge':
      if (!reference) return null;
      return bundle(
        metric,
        { reference },
        [continuousOutput(metric)],
        `${metric} against the mapped ground truth`
      );

    case 'bleu':
      if (!reference) return null;
      return bundle(
        metric,
        { references: [reference] },
        [continuousOutput(metric)],
        'bleu against the mapped ground truth'
      );

    case 'string-check':
      if (!reference) return null;
      return bundle(
        metric,
        {
          operation: values.body.stringCheck.operation,
          left_template: OUTPUT_TEMPLATE,
          right_template: reference,
        },
        [continuousOutput(metric)],
        `string check (${values.body.stringCheck.operation}) against the mapped ground truth`
      );

    case 'number-check': {
      if (!reference) return null;
      const { operation, epsilon } = values.body.numberCheck;
      return bundle(
        metric,
        {
          operation,
          left_template: OUTPUT_TEMPLATE,
          right_template: reference,
          // The API requires epsilon for absolute difference and REJECTS it for
          // every other operation, so it is conditional rather than nullable.
          ...(operation === 'absolute difference' && epsilon !== null ? { epsilon } : {}),
        },
        [continuousOutput(metric)],
        `number check (${operation}) against the mapped ground truth`
      );
    }

    case 'llm-judge': {
      const scores = values.body.scores;
      const outputs = scores.flatMap((score) =>
        score.scoreType === 'rubric'
          ? [
              continuousOutput(score.name, score.description),
              labelOutput(`${score.name}.label`, `Selected rubric label for ${score.name}`),
            ]
          : [continuousOutput(score.name, score.description)]
      );
      return bundle(
        metric,
        {
          // A concrete Model, NOT a bare ModelRef string. A name-only ref
          // resolves to an unreachable native endpoint and the judge call fails
          // at runtime (verified live: the job errors making a completions
          // request to the model). Same gateway resolution as the target.
          model: {
            url: getModelInferenceGatewayUrl(workspace, values.body.judgeModel),
            name: bareModelName(values.body.judgeModel),
          },
          // system_prompt is a TOP-LEVEL field; the API rejects it inside
          // prompt_template. It is injected verbatim, so it carries rendered
          // prose rather than the SDK's Jinja template.
          system_prompt: renderScoreGuidance(scores),
          prompt_template: composeJudgePromptTemplate(bindings),
          scores: scores.map(toScorePayload),
        },
        outputs,
        `LLM-as-a-Judge scoring ${scores.map((score) => score.name).join(', ')}`
      );
    }
  }
};

export const buildMetricBundles = (
  values: EvaluationFormValues,
  bindings: DatasetBindings,
  workspace: string
): InlineMetricBundle[] =>
  selectedMetrics(values)
    .map((metric) => buildMetricBundle(metric, values, bindings, workspace))
    .filter((entry): entry is InlineMetricBundle => entry !== null);

/**
 * The reusable part of a run: everything except the target model, which is
 * chosen per run. This is what gets written to ``eval-config.json``.
 */
export const buildEvaluationSpec = (
  values: EvaluationFormValues,
  bindings: DatasetBindings,
  workspace: string
): DatasetEvalSpec => ({
  dataset: values.dataset ?? '',
  metrics: buildMetricBundles(values, bindings, workspace),
  prompt_template: composeGenerationPrompt(bindings),
  field_mapping: toFieldMapping(values.fieldMapping) ?? null,
});
