// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RiskTolerance } from '@nemo/sdk/generated/anonymizer/schema';
import type { Tag } from '@nvidia/foundations-react-core';
import type { ComponentProps } from 'react';

export const SOURCE_TYPE_URL = 'url';
export const SOURCE_TYPE_DATASET = 'dataset';

export type SourceType = typeof SOURCE_TYPE_URL | typeof SOURCE_TYPE_DATASET;

export const SOURCE_TYPE_OPTIONS: { children: string; value: SourceType }[] = [
  { children: 'Dataset', value: SOURCE_TYPE_DATASET },
  { children: 'URL', value: SOURCE_TYPE_URL },
];

export const STRATEGY_SUBSTITUTE = 'substitute';
export const STRATEGY_REDACT = 'redact';
export const STRATEGY_ANNOTATE = 'annotate';
export const STRATEGY_HASH = 'hash';
export const STRATEGY_REWRITE = 'rewrite';

export type Strategy =
  | typeof STRATEGY_SUBSTITUTE
  | typeof STRATEGY_REDACT
  | typeof STRATEGY_ANNOTATE
  | typeof STRATEGY_HASH
  | typeof STRATEGY_REWRITE;

export const REWRITE_STRATEGY: Strategy = STRATEGY_REWRITE;

export const STRATEGY_OPTIONS: { children: string; value: Strategy }[] = [
  { children: 'Substitute', value: STRATEGY_SUBSTITUTE },
  { children: 'Redact', value: STRATEGY_REDACT },
  { children: 'Annotate', value: STRATEGY_ANNOTATE },
  { children: 'Hash', value: STRATEGY_HASH },
  { children: 'Rewrite', value: STRATEGY_REWRITE },
];

export const STRATEGY_DESCRIPTIONS: Record<Strategy, string> = {
  [STRATEGY_SUBSTITUTE]:
    'Replace detected entities with LLM-generated synthetic values for names, cities, dates, etc.',
  [STRATEGY_REDACT]:
    'Replace entities with a label-based marker. The original text is removed entirely.',
  [STRATEGY_ANNOTATE]:
    'Tag entities with their label but preserve the original text. Useful for review and debugging.',
  [STRATEGY_HASH]:
    'Replace entities with a deterministic hash digest. The same entity text always produces the same hash.',
  [STRATEGY_REWRITE]:
    'Transform the entire text to produce a privacy-safe version that reduces explicit and inferable identifiers.',
};

export const REDACT_DEFAULT_TEMPLATE = '[REDACTED_{label}]';
export const ANNOTATE_DEFAULT_TEMPLATE = '<{text}, {label}>';
export const HASH_DEFAULT_TEMPLATE = '<{label}_{digest}>';
export const HASH_DEFAULT_DIGEST_LENGTH = 8;

export const HASH_ALGORITHM_VALUES = ['sha256', 'sha1', 'md5'] as const;
export type HashAlgorithmOption = (typeof HASH_ALGORITHM_VALUES)[number];
export const HASH_ALGORITHM_DEFAULT: HashAlgorithmOption = 'sha256';
const HASH_ALGORITHM_LABELS: Record<HashAlgorithmOption, string> = {
  sha256: 'SHA-256',
  sha1: 'SHA-1',
  md5: 'MD5',
};
export const HASH_ALGORITHM_OPTIONS: { children: string; value: HashAlgorithmOption }[] =
  HASH_ALGORITHM_VALUES.map((value) => ({ children: HASH_ALGORITHM_LABELS[value], value }));

export const PRIVACY_GOAL_MODE_DEFAULT = 'default';
export const PRIVACY_GOAL_MODE_CUSTOM = 'custom';

export type PrivacyGoalMode = typeof PRIVACY_GOAL_MODE_DEFAULT | typeof PRIVACY_GOAL_MODE_CUSTOM;

export const PRIVACY_GOAL_MODE_OPTIONS: { value: PrivacyGoalMode; children: string }[] = [
  { value: PRIVACY_GOAL_MODE_DEFAULT, children: 'Default' },
  { value: PRIVACY_GOAL_MODE_CUSTOM, children: 'Custom' },
];

export const RISK_TOLERANCE_ORDER = [
  RiskTolerance.minimal,
  RiskTolerance.low,
  RiskTolerance.moderate,
  RiskTolerance.high,
] as const;

export const RISK_TOLERANCE_LABELS: Record<RiskTolerance, string> = {
  minimal: 'Minimal',
  low: 'Low',
  moderate: 'Moderate',
  high: 'High',
};

export const RISK_TOLERANCE_DEFAULT: RiskTolerance = RiskTolerance.low;
export const REWRITE_DEFAULT_MAX_REPAIR_ROUNDS = 3;
export const REWRITE_MIN_MAX_REPAIR_ROUNDS = 0;

export const ENTITY_MODE_CUSTOM = 'custom';
export const ENTITY_MODE_AUTO = 'auto';

export type EntityMode = typeof ENTITY_MODE_CUSTOM | typeof ENTITY_MODE_AUTO;

export const ENTITY_MODE_OPTIONS: { value: EntityMode; children: string }[] = [
  { value: ENTITY_MODE_CUSTOM, children: 'Custom' },
  { value: ENTITY_MODE_AUTO, children: 'Auto-detect' },
];

/** The count comes from the entity-labels endpoint, so it is unknown until that call lands. */
export const defaultEntitiesLabel = (count: number): string =>
  count ? `all ${count} available entities` : 'all available entities';

export const ENTITY_MODE_DESCRIPTIONS: Record<EntityMode, (defaults: string) => string> = {
  [ENTITY_MODE_AUTO]: (defaults) =>
    `Auto-detect includes ${defaults} and lets the augmenter add more labels it finds. To restrict the output to a defined list, use Custom.`,
  [ENTITY_MODE_CUSTOM]: () =>
    'Custom mode only outputs the labels selected below. Auto-detect cannot be combined with custom labels.',
};

export type EntityTagColor = NonNullable<ComponentProps<typeof Tag>['color']>;

interface EntityCategory {
  readonly label: string;
  readonly color: EntityTagColor;
  readonly labels: readonly string[];
}

/**
 * The entity-labels endpoint returns a flat list, so the grouping shown in the picker is
 * curated here to match the design. Anything the API adds that isn't listed falls into Other.
 */
export const ENTITY_CATEGORIES: readonly EntityCategory[] = [
  {
    label: 'Personal Identity',
    color: 'blue',
    labels: [
      'first_name',
      'last_name',
      'date_of_birth',
      'age',
      'gender',
      'nationality',
      'language',
    ],
  },
  {
    label: 'Demographics & Beliefs',
    color: 'purple',
    labels: ['race_ethnicity', 'sexuality', 'political_view', 'religious_belief'],
  },
  {
    label: 'Contact & Communication',
    color: 'teal',
    labels: ['email', 'phone_number', 'fax_number'],
  },
  {
    label: 'Location & Address',
    color: 'green',
    labels: [
      'street_address',
      'city',
      'state',
      'county',
      'country',
      'postcode',
      'coordinate',
      'place_name',
      'landmark',
    ],
  },
  { label: 'Date & Time', color: 'yellow', labels: ['date', 'time', 'date_time'] },
  {
    label: 'Government & Legal IDs',
    color: 'red',
    labels: ['ssn', 'national_id', 'tax_id', 'employee_id', 'certificate_license_number', 'pin'],
  },
  {
    label: 'Financial',
    color: 'blue',
    labels: [
      'credit_debit_card',
      'cvv',
      'account_number',
      'bank_routing_number',
      'swift_bic',
      'monetary_amount',
    ],
  },
  {
    label: 'Medical & Health',
    color: 'purple',
    labels: [
      'medical_record_number',
      'health_plan_beneficiary_number',
      'blood_type',
      'biometric_identifier',
    ],
  },
  {
    label: 'Digital & Network',
    color: 'teal',
    labels: [
      'ipv4',
      'ipv6',
      'mac_address',
      'url',
      'api_key',
      'http_cookie',
      'device_identifier',
      'user_name',
      'password',
      'unique_id',
    ],
  },
  { label: 'Vehicle & Transport', color: 'green', labels: ['license_plate', 'vehicle_identifier'] },
  {
    label: 'Employment & Organization',
    color: 'yellow',
    labels: ['occupation', 'employment_status', 'company_name', 'organization_name', 'customer_id'],
  },
  {
    label: 'Education',
    color: 'red',
    labels: ['university', 'education_level', 'degree', 'field_of_study'],
  },
  {
    label: 'Legal & Institutional',
    color: 'blue',
    labels: ['court_name', 'prison_detention_facility'],
  },
];

export const ENTITY_CATEGORY_OTHER = 'Other';
export const ENTITY_CUSTOM_TAG_COLOR: EntityTagColor = 'gray';

const COLOR_BY_LABEL = new Map<string, EntityTagColor>(
  ENTITY_CATEGORIES.flatMap((category) =>
    category.labels.map((label) => [label, category.color] as const)
  )
);

/** Custom labels have no category, so they fall back to the neutral chip colour. */
export const entityTagColor = (label: string): EntityTagColor =>
  COLOR_BY_LABEL.get(label) ?? ENTITY_CUSTOM_TAG_COLOR;

export const DEFAULT_PREVIEW_ROWS = 1;
/** Mirrors the plugin's `preview_num_records.max`, which 422s rather than clamping. */
export const MAX_PREVIEW_ROWS = 10;

export const MAX_COLUMN_INTROSPECTION_BYTES = 50 * 1024 * 1024;

export const DEFAULT_MODEL_TIMEOUT_SECONDS = 500;

export const DEFAULT_MODEL_MAX_TOKENS = 16384;
export const MAX_MODEL_MAX_TOKENS = 32768;

/** Mirrors the LLM entries in the Anonymizer library's default `models.yaml`. */
export const DEFAULT_MODEL_TEMPERATURE = 0.3;
export const DEFAULT_MODEL_TOP_P = 0.95;

export const ANONYMIZER_PARAM_METADATA = {
  temperature: { default: DEFAULT_MODEL_TEMPERATURE },
  top_p: { default: DEFAULT_MODEL_TOP_P },
  max_tokens: { default: DEFAULT_MODEL_MAX_TOKENS, max: MAX_MODEL_MAX_TOKENS },
} as const;

export const DETECTION_ROLES = [
  'entity_detector',
  'entity_validator',
  'entity_augmenter',
  'latent_detector',
];
export const REPLACE_ROLE = 'replacement_generator';
export const REWRITE_ROLES = [
  'domain_classifier',
  'disposition_analyzer',
  'meaning_extractor',
  'qa_generator',
  'rewriter',
  'repairer',
  'evaluator',
];

export const ROLE_LABELS: Record<string, string> = {
  entity_detector: 'Entity Detector',
  entity_validator: 'Entity Validator',
  entity_augmenter: 'Entity Augmenter',
  latent_detector: 'Latent Detector',
  replacement_generator: 'Replacement Generator',
  domain_classifier: 'Domain Classifier',
  disposition_analyzer: 'Disposition Analyzer',
  meaning_extractor: 'Meaning Extractor',
  qa_generator: 'QA Generator',
  rewriter: 'Rewriter',
  repairer: 'Repairer',
  evaluator: 'Evaluator',
};

export const DETECTOR_ROLE = 'entity_detector';

export const DETECTOR_GROUP_SUGGESTED = 'suggested';
export const DETECTOR_GROUP_OTHER = 'other';

export const DETECTOR_GROUP_LABELS: Record<string, string> = {
  [DETECTOR_GROUP_SUGGESTED]: 'Suggested',
  [DETECTOR_GROUP_OTHER]: 'Other models',
};

/**
 * Names of NER families known to work as the detector. Nothing on a ModelEntity states a model's
 * task, so this only promotes the known ones — every model stays selectable.
 */
export const NER_DETECTOR_PATTERNS: readonly RegExp[] = [/gliner/i, /privacy[-_ ]?filter/i];

export const DETECTOR_MODEL_HINT =
  'Runs token classification, so it needs an NER model such as GLiNER or a privacy filter rather than a chat model.';

/** The detector is a token-classification NIM: it takes labels and a threshold, not sampling params. */
export const supportsSamplingParams = (role: string): boolean => role !== DETECTOR_ROLE;

export const activeRolesForStrategy = (strategy: Strategy): string[] => {
  // rewrite reuses the replacement generator, so the backend validates that role too
  if (strategy === STRATEGY_REWRITE) return [...DETECTION_ROLES, ...REWRITE_ROLES, REPLACE_ROLE];
  if (strategy === STRATEGY_SUBSTITUTE) return [...DETECTION_ROLES, REPLACE_ROLE];
  return [...DETECTION_ROLES];
};

export const TAB_SOURCE = 'source';
export const TAB_MODEL_SETTINGS = 'model-settings';

export type BuilderTab = typeof TAB_SOURCE | typeof TAB_MODEL_SETTINGS;

export const PANEL_TABS: { value: BuilderTab; children: string }[] = [
  { value: TAB_SOURCE, children: 'Source' },
  { value: TAB_MODEL_SETTINGS, children: 'Model Settings' },
];
