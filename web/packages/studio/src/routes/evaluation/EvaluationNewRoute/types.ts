// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PanelScoreFormData } from '@studio/components/evaluation/Jobs/form/ScoreModal';
import { z } from 'zod';

/** Canonical evaluator fields a dataset column can be bound to, mirroring
 *  ``_KNOWN_BINDING_FIELDS`` in ``nemo_evaluator_sdk.values.dataset_schemas``.
 *
 *  ``output`` is deliberately absent. It is a canonical field, but binding it is
 *  only meaningful for an OFFLINE evaluation, where ``build_offline_sample``
 *  reads ``row["output"]`` to synthesise ``sample.output_text`` for pre-generated
 *  rows. This form is online-only: the model under test produces the response,
 *  and metric templates reach it through ``{{sample.output_text}}``, never
 *  through ``{{output}}`` — which online resolves against the dataset row.
 *  Offering an ``output`` binding here would invite a mapping that silently
 *  scores canned dataset text instead of the model. */
export const CANONICAL_FIELDS = ['input', 'reference', 'context', 'messages'] as const;

export type CanonicalField = (typeof CANONICAL_FIELDS)[number];

/** Bindings shown up front; the rest sit behind a disclosure. */
export const PRIMARY_CANONICAL_FIELDS: readonly CanonicalField[] = ['input', 'reference'];

export const CANONICAL_FIELD_LABELS: Record<CanonicalField, string> = {
  input: 'Input',
  reference: 'Ground Truth',
  context: 'Context',
  messages: 'Messages',
};

/** No binding. Selects are ``dismissible`` so the user can clear back to this;
 *  ``toFieldMapping`` drops empty bindings at submit. */
export const UNMAPPED = '';

/** ``FieldMapping`` refuses any path with array indexing
 *  (``validate_supported_dataset_paths``: "array path segments are not supported
 *  for column mappings"), so ``messages[0].content`` — which
 *  ``extractUserFriendlyKeysFromRow`` happily produces — is not bindable. */
export const isSupportedMappingPath = (path: string): boolean => !/[[\]]/.test(path);

/** Plain-English judge guidance, generated from the Score Definitions.
 *
 *  The SDK ships a Jinja template (``DEFAULT_JUDGE_SYSTEM_PROMPT_TEMPLATE``) that
 *  renders this same text from the injected ``scores`` context, but that only
 *  works inside ``prompt_template``, which is passed through the renderer.
 *  ``system_prompt`` is applied by the ``InjectSystemMessage`` preprocess hook
 *  AFTER rendering and is prepended VERBATIM -- so Jinja placed there would reach
 *  the judge as literal ``{{ ... }}`` text. Rendering it here keeps the field a
 *  plain string, which is what that hook expects, and keeps Jinja control flow
 *  out of a form input. */
export const renderScoreGuidance = (scores: PanelScoreFormData[]): string => {
  const names = scores.map((score) => score.name).filter(Boolean);
  const lines = [
    'You are an expert evaluator for answers to user queries. Your task is to assess responses to user queries based on ' +
      (names.join(', ') || 'the criteria below') +
      '.',
  ];

  for (const score of scores) {
    const range =
      score.scoreType === 'range'
        ? ` with a score range from ${score.minimum} to ${score.maximum}`
        : '';
    const description = score.description ? `: ${score.description}` : '';
    lines.push('', `${score.name}${range}${description}`);
    if (score.scoreType === 'rubric') {
      for (const level of score.rubric) {
        lines.push(`* ${level.label}${level.description ? `: ${level.description}` : ''}`);
      }
    }
  }

  return lines.join('\n');
};

/** Seeded Score Definition. The judge's ``structured_output`` is derived from the
 *  scores when omitted, and the default parser is a ``JSONScoreParser`` keyed on
 *  the score name, so defining the score here is what makes the judge both
 *  describe the scale and return a parseable answer. */
export const DEFAULT_HELPFULNESS_SCORE: PanelScoreFormData = {
  scoreType: 'rubric',
  name: 'helpfulness',
  description: "Overall utility of the response in addressing the user's needs.",
  // Worded labels, not the digits they map to. The label is what the judge must
  // emit (rubric scores derive an enum of exactly these strings), while `value`
  // is what aggregation and ranking use -- so naming the levels costs nothing and
  // gives the judge something meaningful to choose between.
  rubric: [
    {
      label: 'Unhelpful',
      description: 'Fails to address the request, is irrelevant, or could cause harm.',
      value: 0,
    },
    {
      label: 'Poor',
      description: 'Partially addresses the request but has significant gaps or errors.',
      value: 1,
    },
    {
      label: 'Adequate',
      description: 'Addresses the core request but lacks detail, clarity, or completeness.',
      value: 2,
    },
    {
      label: 'Good',
      description: 'Fully addresses the request with appropriate detail and is genuinely useful.',
      value: 3,
    },
    {
      label: 'Excellent',
      description: 'Comprehensive and well-structured, fully satisfying the request.',
      value: 4,
    },
  ],
};

/** The Jinja expressions templates use for each role a metric may need.
 *
 *  Not the same as ``field_mapping``: for an OpenAI messages dataset the binding
 *  is the whole array (``messages -> <column>``), because ``FieldMapping`` refuses
 *  a path containing ``[`` or ``]``, and the index lives in the template instead. */
export interface DatasetBindings {
  /** Set when the dataset is OpenAI messages format; the bound column's name. */
  messagesColumn: string | null;
  input: string;
  reference: string | null;
  context: string | null;
  /** Dataset-relative dot-bracket paths for the same values, for resolving a
   *  preview against a real row via ``resolveKeyPath``. Templates cannot be used
   *  for that: they are Jinja, and for a messages dataset they address the
   *  canonical ``messages`` alias rather than the column's real name. */
  inputPath: string | null;
  referencePath: string | null;
}

/** Prompt sent to the model under test, built from the resolved input binding. */
export const composeGenerationPrompt = (bindings: DatasetBindings) => ({
  messages: [{ role: 'user', content: bindings.input }],
});

/** The judge's user message, composed from what the dataset actually provides.
 *
 *  Derived rather than authored: one LLMJudge carries a single ``prompt_template``
 *  shared by ALL its scores, so "should the judge see ground truth" cannot be
 *  decided per score -- judge correctness and helpfulness together and the prompt
 *  needs the reference, because correctness does. The union wins, so every
 *  available field is shown.
 *
 *  The response under evaluation goes last, so the judge reads the question and
 *  any reference before the answer it is grading. */
/** The judge's ``prompt_template`` on the wire.
 *
 *  MUST be the chat-messages form, never a bare string. ``render_request`` wraps
 *  a string template as ``{"prompt": ...}``, and the client then dispatches on
 *  presence of ``messages`` (inference.py: `chat.completions.create if "messages"
 *  in request else completions.create`) -- so a string prompt silently routes the
 *  judge to the COMPLETIONS endpoint, which chat models do not serve. Verified
 *  live: the job fails making a completions request to the judge model. */
export const composeJudgePromptTemplate = (bindings: DatasetBindings) => ({
  messages: [{ role: 'user', content: composeJudgeUserPrompt(bindings) }],
});

export const composeJudgeUserPrompt = (bindings: DatasetBindings): string => {
  // Labels match the mapping UI exactly -- a user who mapped "Ground Truth"
  // should see "Ground Truth" in the prompt, not a synonym.
  const lines = [`Input: ${bindings.input}`];
  if (bindings.context) lines.push(`Context: ${bindings.context}`);
  if (bindings.reference) lines.push(`Ground Truth: ${bindings.reference}`);
  lines.push('', 'Response to evaluate: {{sample.output_text}}');
  return lines.join('\n');
};

/** Metrics offerable for a dataset-driven, online model evaluation. The full
 *  ``MetricType`` enum is much larger, but the rest are RAG/agent-shaped
 *  (RAGAS, tool-calling, trajectory) or remote, and do not fit this form.
 *  Note ``exact-match``: v1 called it ``em``. */
export const SELECTABLE_METRICS = [
  { type: 'llm-judge', label: 'LLM-as-a-Judge' },
  { type: 'exact-match', label: 'Exact Match' },
  { type: 'f1', label: 'F1' },
  { type: 'bleu', label: 'BLEU' },
  { type: 'rouge', label: 'ROUGE' },
  { type: 'string-check', label: 'String Check' },
  { type: 'number-check', label: 'Number Check' },
] as const;

export type SelectableMetric = (typeof SELECTABLE_METRICS)[number]['type'];

/** Metrics whose only required config is a reference template. ``candidate`` is
 *  deliberately omitted from the payload: every one of these documents "if not
 *  provided, the output text from the model is used", so leaving it off is both
 *  less config and immune to the ``{{output}}`` trap. */
export const REFERENCE_METRICS: readonly SelectableMetric[] = [
  'exact-match',
  'f1',
  'bleu',
  'rouge',
];

/** Metrics that compare two rendered templates rather than a single reference. */
export const COMPARISON_METRICS: readonly SelectableMetric[] = ['string-check', 'number-check'];

export const STRING_CHECK_OPERATIONS = [
  'equals',
  'not equals',
  'contains',
  'not contains',
  'startswith',
  'endswith',
] as const;

export const NUMBER_CHECK_OPERATIONS = [
  'equals',
  'not equals',
  'greater than',
  'greater than or equal',
  'less than',
  'less than or equal',
  'absolute difference',
] as const;

/** The model-under-test's response. Not a dataset column, so not bindable. */
export const SAMPLE_OUTPUT_VARIABLE = 'sample.output_text';

/** Metric fields live under ``body`` so ``ScoreDefinitions`` / ``MetricScoreSection``
 *  (which do ``useFormContext<MetricPanelFormData>()`` and watch ``body.scores``)
 *  drop in without modification. */
export interface EvaluationFormValues {
  /** ``workspace/fileset#path``. Doubles as ``spec.dataset`` at submit. */
  dataset: string | null;
  /** Canonical evaluator field -> dataset column path. ``spec.field_mapping``. */
  fieldMapping: Record<CanonicalField, string>;
  /** Model under test. Always chosen per run, overriding any saved config. */
  model: string;
  body: {
    /** Which metrics are scored. At least one must be selected. */
    metrics: Record<SelectableMetric, boolean>;
    scores: PanelScoreFormData[];
    judgeModel: string;
    stringCheck: { operation: (typeof STRING_CHECK_OPERATIONS)[number] };
    numberCheck: {
      operation: (typeof NUMBER_CHECK_OPERATIONS)[number];
      /** Required by the API for "absolute difference", rejected for anything else. */
      epsilon: number | null;
    };
  };
}

export const EMPTY_FIELD_MAPPING: Record<CanonicalField, string> = {
  input: '',
  reference: '',
  context: '',
  messages: '',
};

/** The metric types the user ticked. */
export const selectedMetrics = (values: EvaluationFormValues): SelectableMetric[] =>
  SELECTABLE_METRICS.map((metric) => metric.type).filter((type) => values.body?.metrics?.[type]);

/** Which region of the form a check belongs to. Only the FIRST failure in each
 *  section is surfaced, so every column shows exactly one error naming its own
 *  next step -- never a wall of red, but never a hidden requirement either. A
 *  strictly global first-only rule would keep the judge-model guard invisible
 *  until the model and dataset were both filled in. */
/** Form state -> ``spec.field_mapping``. Drops unbound fields; returns undefined
 *  when nothing is bound, so the key is omitted from the spec rather than sent
 *  as an empty object. */
export const toFieldMapping = (
  mapping: Record<CanonicalField, string>
): Record<string, string> | undefined => {
  const bound = CANONICAL_FIELDS.filter(
    (field) =>
      mapping[field] && mapping[field] !== UNMAPPED && isSupportedMappingPath(mapping[field])
  ).map((field) => [field, mapping[field]] as const);
  return bound.length > 0 ? Object.fromEntries(bound) : undefined;
};

export const EVALUATION_FORM_DEFAULTS: EvaluationFormValues = {
  dataset: null,
  fieldMapping: EMPTY_FIELD_MAPPING,
  model: '',
  body: {
    metrics: {
      'llm-judge': true,
      'exact-match': false,
      f1: false,
      bleu: false,
      rouge: false,
      'string-check': false,
      'number-check': false,
    },
    scores: [DEFAULT_HELPFULNESS_SCORE],
    judgeModel: '',
    stringCheck: { operation: 'contains' },
    numberCheck: { operation: 'equals', epsilon: null },
  },
};

/** Row-0 preview is read from an HTTP Range head, which cannot satisfy Parquet
 *  (hyparquet needs the footer, and a truncated buffer fails silently to null).
 *  ponytail: text formats only; add Parquet when row-0 preview can fetch a full
 *  object, or when the backend exposes a row-preview endpoint. */
export const MAPPABLE_FILE_TYPES = ['.json', '.jsonl', '.csv'] as const;

/**
 * Submit-time validation, as a zod schema fed to RHF via ``zodResolver``.
 *
 * A schema rather than hand-rolled checks so the form behaves like every other
 * Studio form: RHF's default ``mode: 'onSubmit'`` keeps errors off the screen
 * until the user submits (per the UX guidelines), and its default
 * ``reValidateMode: 'onChange'`` retires each error the moment its field is
 * fixed -- with no bespoke effect to keep the two in step.
 *
 * Cross-field rules live in ``superRefine`` because they depend on which metrics
 * are selected; each issue names the field the user must act on, so the message
 * lands where the fix is.
 */
export const evaluationSchema = z
  .object({
    model: z.string().min(1, 'Select a model to evaluate.'),
    dataset: z.string().min(1, 'Select an input file.').nullable(),
    fieldMapping: z.record(z.string()),
    body: z
      .object({
        metrics: z.record(z.boolean()),
        judgeModel: z.string(),
        scores: z.array(z.any()),
        numberCheck: z
          .object({ operation: z.string(), epsilon: z.number().nullable() })
          .passthrough(),
        stringCheck: z.object({ operation: z.string() }).passthrough(),
      })
      .passthrough(),
  })
  .passthrough()
  .superRefine((values, ctx) => {
    const mapping = values.fieldMapping ?? {};
    const selected = SELECTABLE_METRICS.map((metric) => metric.type).filter(
      (type) => values.body?.metrics?.[type]
    );

    if (!values.dataset) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['dataset'],
        message: 'Select an input file.',
      });
    }

    // A messages dataset binds the whole array instead of individual columns,
    // and the turns are resolved positionally in the template.
    if (!mapping.input && !mapping.messages) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['fieldMapping', 'input'],
        message: 'Map a dataset column to Input.',
      });
    }

    if (selected.length === 0) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['body', 'metrics'],
        message: 'Select at least one metric.',
      });
    }

    const needsReference = selected.some(
      (metric) => REFERENCE_METRICS.includes(metric) || COMPARISON_METRICS.includes(metric)
    );
    // A bound messages column is NOT proof of a ground truth: the reference is
    // the assistant turn, and a prompts-only dataset has none. The error goes to
    // the metrics slot in that case because no Ground Truth select is rendered
    // for a messages dataset, so a field error there would never be seen.
    if (needsReference && !mapping.reference) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: mapping.messages ? ['body', 'metrics'] : ['fieldMapping', 'reference'],
        message: mapping.messages
          ? 'This dataset has no assistant turn to compare against. Clear the metrics that score against Ground Truth.'
          : 'This metric compares against ground truth, so Ground Truth must be mapped.',
      });
    }

    if (selected.includes('llm-judge')) {
      if (!values.body?.judgeModel) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['body', 'judgeModel'],
          message: 'Select a judge model.',
        });
      }
      if (!values.body?.scores?.length) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['body', 'scores'],
          message: 'Add at least one score for the judge to extract.',
        });
      }
    }

    // The API requires epsilon for absolute difference and rejects it elsewhere.
    if (
      selected.includes('number-check') &&
      values.body?.numberCheck?.operation === 'absolute difference' &&
      values.body?.numberCheck?.epsilon === null
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['body', 'numberCheck', 'epsilon'],
        message: 'Absolute difference requires a tolerance.',
      });
    }
  });
