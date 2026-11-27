// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Temporary mirror of the library's EvaluationReportData (schema v1); swap for the generated SDK schema.
import { z } from 'zod';

const nullableNumber = z.number().nullable();
const nullableNumbers = z.array(nullableNumber);

const metricScoreSchema = z.object({
  raw_score: nullableNumber,
  score: nullableNumber,
  grade: z.string(),
  notes: z.string().nullable(),
});

const distributionSchema = z.object({
  labels: z.array(z.string()),
  bin_edges: z.array(z.number()).nullable(),
  training_percent: nullableNumbers,
  synthetic_percent: nullableNumbers,
});

const pointsSchema = z.object({ x: nullableNumbers, y: nullableNumbers });

const projectionSchema = z.object({
  axes: z.array(z.string()),
  training: pointsSchema,
  synthetic: pointsSchema,
});

const metricSchema = z.object({ score: metricScoreSchema });

const correlationMetricSchema = metricSchema.extend({
  correlations: z
    .object({
      columns: z.array(z.string()),
      training: z.array(nullableNumbers),
      synthetic: z.array(nullableNumbers),
      difference: z.array(nullableNumbers),
    })
    .nullable(),
});

const deepStructureMetricSchema = metricSchema.extend({ projection: projectionSchema.nullable() });

const textSemanticMetricSchema = metricSchema.extend({
  projections: z.record(z.string(), projectionSchema),
});

const textStructureDistributionsSchema = z.object({
  sentence_count: distributionSchema.nullable(),
  words_per_sentence: distributionSchema.nullable(),
  characters_per_word: distributionSchema.nullable(),
});

const textStructureMetricSchema = metricSchema.extend({
  columns: z.record(z.string(), textStructureDistributionsSchema),
});

const membershipInferenceMetricSchema = metricSchema.extend({
  outcomes: z.array(
    z.object({ grade: z.string(), count: z.number().nullable(), percentage: nullableNumber })
  ),
  excluded_columns: z.array(z.string()),
});

const attributeInferenceMetricSchema = metricSchema.extend({
  columns: z.array(z.object({ column: z.string(), protection: nullableNumber, grade: z.string() })),
});

const piiReplayEntrySchema = z.object({
  column_name: z.string(),
  column_assigned_type: z.string(),
  pii_type: z.string(),
  total_training_data: z.number(),
  unique_training_data: z.number(),
  total_synthetic_data: z.number(),
  unique_synthetic_data: z.number(),
  unique_synthetic_data_percentage: z.number(),
});

const piiReplayMetricSchema = metricSchema.extend({
  enabled: z.boolean().nullable(),
  training_total_records: z.number(),
  synthetic_total_records: z.number(),
  entries: z.array(piiReplayEntrySchema),
  skipped_reason: z.string().nullable(),
});

const skippedComparisonSchema = z.object({
  group: z.string().nullable(),
  column: z.string(),
  reason: z.string(),
});

const autocorrelationProfileSchema = z.object({
  group: z.string().nullable(),
  column: z.string(),
  lags: z.array(z.number()),
  effective_max_lag: z.number(),
  evaluated_lags: z.number(),
  error: z.number(),
  similarity: z.number(),
  reason: z.string().nullable(),
  training_acf: nullableNumbers,
  synthetic_acf: nullableNumbers,
});

const autocorrelationSimilarityMetricSchema = metricSchema.extend({
  details: z.object({
    evaluation_mode: z.enum(['per_group', 'global']).nullable(),
    timestamp_column: z.string().nullable(),
    group_column: z.string().nullable(),
    max_lag: z.number().nullable(),
    min_points: z.number().nullable(),
    counts: z
      .object({
        groups: z.number(),
        shared_groups: z.number(),
        omitted_shared_groups: z.number(),
        columns: z.number(),
        evaluated_profiles: z.number(),
        skipped: z.number(),
      })
      .nullable(),
    group_selection: z
      .object({
        shared_groups: z.number(),
        evaluated_groups: z.number(),
        omitted_shared_groups: z.number(),
        policy: z.enum(['all_shared_groups', 'seeded_random_sample']),
      })
      .nullable(),
    per_group: z.array(
      z.object({ group: z.string().nullable(), similarity: z.number(), count: z.number() })
    ),
    per_column: z.array(
      z.object({ column: z.string(), similarity: z.number(), count: z.number() })
    ),
    profiles: z.array(autocorrelationProfileSchema),
    skipped: z.array(skippedComparisonSchema),
    excluded_columns: z.array(z.string()),
    groups_only_in_training_count: z.number(),
    groups_only_in_synthetic_count: z.number(),
  }),
});

const windowMembershipInferenceMetricSchema = metricSchema.extend({
  details: z.object({
    evaluation_mode: z.enum(['full', 'copy_diagnostics_only', 'unavailable']),
    columns: z.array(z.string()),
    excluded_columns: z.array(z.string()),
    window_size: z.number().nullable(),
    counts: z
      .object({
        training_windows_total: z.number(),
        training_windows: z.number(),
        synthetic_windows_total: z.number(),
        synthetic_windows: z.number(),
        holdout_windows_total: z.number(),
        holdout_windows: z.number(),
        holdout_groups_with_windows: z.number(),
      })
      .nullable(),
    exact_copy_rate: nullableNumber,
    closest_match: z
      .object({
        distance: nullableNumber,
        exact_copy: z.boolean(),
        synthetic_values: z.array(z.array(z.number())),
        training_values: z.array(z.array(z.number())).nullable(),
      })
      .nullable(),
    distances: z
      .object({ training: nullableNumbers, holdout: nullableNumbers.nullable() })
      .nullable(),
    attack: z
      .object({
        threshold: nullableNumber,
        training_flag_rate: z.number(),
        holdout_flag_rate: z.number(),
        privacy_loss_beyond_chance: z.number(),
        holdout_exact_match_rate: z.number(),
        exact_copy_privacy_loss: z.number(),
      })
      .nullable(),
    training_group_count: z.number(),
    per_group: z.array(
      z.object({
        group: z.string().nullable(),
        windows: z.number(),
        exact_copy_rate: z.number(),
        flag_rate: nullableNumber,
      })
    ),
  }),
});

const columnProfileSchema = z.object({
  type: z.string(),
  count: z.number(),
  unique_count: z.number(),
  unique_percent: nullableNumber,
  missing_count: z.number(),
  missing_percent: nullableNumber,
  avg_str_length: z.number(),
});

const columnResultSchema = z.object({
  name: z.string(),
  training: columnProfileSchema,
  synthetic: columnProfileSchema,
  distribution_distance: nullableNumber,
  distribution_stability: metricScoreSchema.nullable(),
  distribution: distributionSchema.nullable(),
  pii: z
    .object({
      assigned_type: z.string().nullable(),
      assigned_entity: z.string().nullable(),
      detected_entity_counts: z.record(z.string(), z.number()),
      is_transformed: z.boolean(),
      transform_functions: z.array(z.string()),
    })
    .nullable(),
});

const piiReplacementColumnSchema = z.object({
  column_name: z.string(),
  entity_type: z.string(),
  replaced_with: z.string(),
  pattern: z.string().nullable(),
  pattern_symbols: z.array(z.object({ symbol: z.string(), meaning: z.string() })),
  depends_on: z.array(z.object({ column_name: z.string(), entity_type: z.string().nullable() })),
  found_by: z.array(z.string()),
  entity_counts: z.record(z.string(), z.number()),
});

export const evaluationReportDataSchema = z.object({
  schema_version: z.number(),
  library_version: z.string(),
  job_id: z.string().nullable(),
  scores: z.object({
    synthetic_quality: metricScoreSchema.nullable(),
    data_privacy: metricScoreSchema.nullable(),
  }),
  metrics: z.object({
    column_correlation_stability: correlationMetricSchema.nullable(),
    deep_structure_stability: deepStructureMetricSchema.nullable(),
    column_distribution_stability: metricSchema.nullable(),
    text_semantic_similarity: textSemanticMetricSchema.nullable(),
    text_structure_similarity: textStructureMetricSchema.nullable(),
    membership_inference_protection: membershipInferenceMetricSchema.nullable(),
    attribute_inference_protection: attributeInferenceMetricSchema.nullable(),
    pii_replay: piiReplayMetricSchema.nullable(),
    autocorrelation_similarity: autocorrelationSimilarityMetricSchema.nullable(),
    window_membership_inference_protection: windowMembershipInferenceMetricSchema.nullable(),
  }),
  dataset: z
    .object({
      training_rows: z.number(),
      training_cols: z.number(),
      training_missing_percent: z.number(),
      synthetic_rows: z.number(),
      synthetic_cols: z.number(),
      synthetic_missing_percent: z.number(),
      memorized_lines: z.number(),
    })
    .nullable(),
  columns: z.array(columnResultSchema),
  any_column_transformed: z.boolean(),
  differential_privacy: z.object({
    enabled: z.boolean(),
    epsilon: nullableNumber,
    delta: z.union([z.number(), z.string()]).nullable(),
  }),
  pii_replacement: z
    .object({
      replaced: z.boolean(),
      columns: z.array(piiReplacementColumnSchema),
      kept_columns: z.array(z.string()),
      sampler: z.string().nullable(),
      locale: z.string().nullable(),
      seed: z.number().nullable(),
      value_mappings: z.record(z.string(), z.record(z.string(), z.array(z.string()))),
    })
    .nullable(),
});

export type EvaluationReportData = z.infer<typeof evaluationReportDataSchema>;
export type Metrics = EvaluationReportData['metrics'];
export type MetricScore = z.infer<typeof metricScoreSchema>;
export type Distribution = z.infer<typeof distributionSchema>;
export type Projection = z.infer<typeof projectionSchema>;
export type CorrelationMatrices = NonNullable<
  z.infer<typeof correlationMetricSchema>['correlations']
>;
export type TextStructureDistributions = z.infer<typeof textStructureDistributionsSchema>;
export type MembershipInferenceMetric = z.infer<typeof membershipInferenceMetricSchema>;
export type AttributeInferenceMetric = z.infer<typeof attributeInferenceMetricSchema>;
export type PIIReplayMetric = z.infer<typeof piiReplayMetricSchema>;
export type PIIReplayEntry = z.infer<typeof piiReplayEntrySchema>;
export type AutocorrelationSimilarityMetric = z.infer<typeof autocorrelationSimilarityMetricSchema>;
export type AutocorrelationProfile = z.infer<typeof autocorrelationProfileSchema>;
export type WindowMembershipInferenceMetric = z.infer<typeof windowMembershipInferenceMetricSchema>;
export type WindowMembershipInferenceDetails = WindowMembershipInferenceMetric['details'];
export type ColumnResult = z.infer<typeof columnResultSchema>;
export type PIIReplacement = NonNullable<EvaluationReportData['pii_replacement']>;
export type PIIReplacementColumn = z.infer<typeof piiReplacementColumnSchema>;
export type DatasetCounts = NonNullable<EvaluationReportData['dataset']>;
