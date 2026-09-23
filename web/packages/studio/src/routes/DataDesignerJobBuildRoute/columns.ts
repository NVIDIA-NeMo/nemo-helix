// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type DataDesignerConfig,
  DatetimeSamplerParamsUnit,
  type FilesetFileSeedSource,
  PersonSamplerParamsSex,
  type SamplerColumnConfig,
  SamplerType,
  type SamplingStrategy,
  type SeedConfig,
  TimeDeltaSamplerParamsUnit,
} from '@nemo/sdk/generated/data-designer/schema';
import { COLUMN_TYPE_GROUPS } from '@studio/components/AddColumnPalette/constants';
import type {
  AddColumnSelection,
  ColumnTypeColor,
  ColumnTypeOption,
  DataDesignerColumnType,
} from '@studio/components/AddColumnPalette/types';
import type { DagEdge, DagNode } from '@studio/components/DagCanvas/types';
import type { TemplateColumnSpec } from '@studio/components/DataDesignerStart/types';
import {
  type BuilderModel,
  buildModelConfigs,
} from '@studio/routes/DataDesignerJobBuildRoute/models';

export type FieldReference =
  /** Value is a Jinja2 template; `{{ column_name }}` tokens are dependencies. */
  | 'jinja'
  /** Value is a single column name. */
  | 'single'
  /** Value is a comma-separated list of column names. */
  | 'list';

/** Input control kind for a column config field. */
export type FieldKind = 'text' | 'textarea' | 'select' | 'number' | 'switch';

/**
 * How a field's string value is serialized into the SDK column config. Defaults to a plain
 * string; sampler params commonly need numbers, booleans, JSON objects, or numeric lists.
 */
export type FieldDataType = 'string' | 'number' | 'boolean' | 'json' | 'number-list';

export interface ColumnField {
  /** Key into {@link BuilderColumn.values} and the eventual SDK config. */
  key: string;
  label: string;
  kind: FieldKind;
  required?: boolean;
  placeholder?: string;
  helperText?: string;
  options?: readonly { label: string; value: string }[];
  /** If set, values in this field create dependency edges to other columns. */
  reference?: FieldReference;
  /**
   * Serialize the value as a comma-separated list (e.g. sampler category values). Unlike
   * `reference: 'list'`, this does not treat the entries as column names / dependencies.
   */
  list?: boolean;
  /**
   * How to coerce the string form value when serializing to the SDK config. Defaults to a
   * plain string; `number`/`boolean` parse the value, `json` parses an object/array literal.
   * Used by sampler params whose SDK types are non-string (see PARAM_FIELDS_BY_SAMPLER_TYPE).
   */
  valueType?: 'number' | 'boolean' | 'json';
  /**
   * How the string value is coerced for the SDK config (number, boolean, JSON, numeric list).
   * Defaults to `'string'`. Also drives validation (numbers must parse, JSON must be well-formed).
   */
  dataType?: FieldDataType;
  /**
   * Seeded into a column's `values` when it is created, in the same string form the form
   * holds (so `'true'` for a boolean switch). Template-supplied values take precedence.
   */
  defaultValue?: string;
}

/** Not yet the SDK column config — that's produced by {@link buildDataDesignerConfig}. */
export interface BuilderColumn {
  /** Canvas-unique id (also the DAG node id). */
  id: string;
  option: ColumnTypeOption;
  name: string;
  values: Record<string, string>;
}

const CODE_LANGS = [
  'python',
  'javascript',
  'typescript',
  'java',
  'kotlin',
  'go',
  'rust',
  'ruby',
  'scala',
  'swift',
  'c',
  'cpp',
  'csharp',
  'bash',
  'sql:sqlite',
  'sql:postgres',
  'sql:mysql',
  'sql:tsql',
  'sql:bigquery',
  'sql:ansi',
] as const;

const asOptions = (values: readonly string[]) => values.map((value) => ({ label: value, value }));

export const SEED_FILESET_REF_KEY = 'fileset_ref';
export const SEED_FILE_PATH_KEY = 'file_path';
export const SEED_SAMPLING_STRATEGY_KEY = 'sampling_strategy';
export const SEED_AVAILABLE_COLUMNS_KEY = 'available_columns';

export const SAMPLING_STRATEGY_OPTIONS = [
  { label: 'Ordered', value: 'ordered' },
  { label: 'Shuffle', value: 'shuffle' },
] as const;

/** The seed file's discovered column names for a seed-dataset column (empty if none/undiscovered). */
export const getSeedAvailableColumns = (column: BuilderColumn): string[] =>
  (column.values[SEED_AVAILABLE_COLUMNS_KEY] ?? '')
    .split(',')
    .map((name) => name.trim())
    .filter(Boolean);

const PROMPT_FIELD: ColumnField = {
  key: 'prompt',
  label: 'Prompt',
  kind: 'textarea',
  required: true,
  reference: 'jinja',
  placeholder: 'Write a question about {{ topic }} at {{ difficulty }} difficulty.',
  helperText: 'Jinja2 template — reference other columns with {{ column_name }}.',
};

const MODEL_ALIAS_FIELD: ColumnField = {
  key: 'model_alias',
  label: 'Model alias',
  kind: 'text',
  required: true,
  placeholder: 'e.g. default',
  helperText: 'Alias of a configured model to generate with.',
};

const SYSTEM_PROMPT_FIELD: ColumnField = {
  key: 'system_prompt',
  label: 'System prompt (optional)',
  kind: 'textarea',
  reference: 'jinja',
  helperText: 'Optional. Also supports {{ column_name }} references.',
};

/**
 * Reasoning models otherwise emit their chain of thought as part of the column value —
 * "We need to produce... Let's craft body:" ends up in the generated text. Turning this on
 * routes it to a sibling `{column_name}__reasoning_content` column instead, leaving the
 * value clean regardless of whether the backend can disable thinking outright.
 */
const EXTRACT_REASONING_FIELD: ColumnField = {
  key: 'extract_reasoning_content',
  label: 'Separate reasoning content',
  kind: 'switch',
  dataType: 'boolean',
  defaultValue: 'true',
  helperText:
    'Move a reasoning model\'s chain of thought into a "__reasoning_content" column instead of leaving it in the generated value.',
};

const FIELDS_BY_COLUMN_TYPE: Record<NonNullable<DataDesignerColumnType>, ColumnField[]> = {
  'llm-text': [PROMPT_FIELD, MODEL_ALIAS_FIELD, SYSTEM_PROMPT_FIELD, EXTRACT_REASONING_FIELD],
  'llm-code': [
    PROMPT_FIELD,
    MODEL_ALIAS_FIELD,
    {
      key: 'code_lang',
      label: 'Language',
      kind: 'select',
      required: true,
      options: asOptions(CODE_LANGS),
    },
    SYSTEM_PROMPT_FIELD,
    EXTRACT_REASONING_FIELD,
  ],
  'llm-structured': [
    PROMPT_FIELD,
    MODEL_ALIAS_FIELD,
    {
      key: 'output_format',
      label: 'Output schema (JSON)',
      kind: 'textarea',
      required: true,
      placeholder: '{ "type": "object", "properties": { ... } }',
      helperText: 'JSON schema describing the structured output.',
    },
    SYSTEM_PROMPT_FIELD,
    EXTRACT_REASONING_FIELD,
  ],
  'llm-judge': [
    PROMPT_FIELD,
    MODEL_ALIAS_FIELD,
    {
      key: 'scores',
      label: 'Scores (JSON)',
      kind: 'textarea',
      dataType: 'json',
      required: true,
      placeholder:
        '[{ "name": "Quality", "description": "Overall answer quality.", "options": { "1": "Very poor", "5": "Excellent" } }]',
      helperText: 'JSON array of judge score definitions.',
    },
    SYSTEM_PROMPT_FIELD,
    EXTRACT_REASONING_FIELD,
  ],
  image: [
    { ...PROMPT_FIELD, placeholder: 'Generate an image of {{ subject }}.' },
    MODEL_ALIAS_FIELD,
  ],
  embedding: [
    {
      key: 'target_column',
      label: 'Target column',
      kind: 'text',
      required: true,
      reference: 'single',
      placeholder: 'Name of the column to embed',
      helperText: 'The text column whose values are embedded.',
    },
    MODEL_ALIAS_FIELD,
  ],
  expression: [
    {
      key: 'expr',
      label: 'Expression',
      kind: 'textarea',
      required: true,
      reference: 'jinja',
      placeholder: '{{ first_name }} {{ last_name }}',
      helperText: 'Jinja2 expression evaluated per row; reference columns with {{ }}.',
    },
    {
      key: 'dtype',
      label: 'Result type',
      kind: 'select',
      options: asOptions(['str', 'int', 'float', 'bool']),
      helperText: 'Type to cast the result to (defaults to str).',
    },
  ],
  validation: [
    {
      key: 'target_columns',
      label: 'Target columns',
      kind: 'text',
      required: true,
      reference: 'list',
      placeholder: 'comma,separated,column,names',
      helperText: 'Comma-separated column names to validate.',
    },
    {
      key: 'validator_type',
      label: 'Validator',
      kind: 'select',
      required: true,
      options: asOptions(['code', 'local_callable', 'remote']),
    },
    {
      key: 'validator_params',
      label: 'Validator params (JSON)',
      kind: 'textarea',
      dataType: 'json',
      required: true,
      placeholder: '{ "code_lang": "python" }',
      helperText:
        'Parameters for the chosen validator. For "code": { "code_lang": "python" }. For "remote": { "url": "https://…" }.',
    },
  ],
  'seed-dataset': [
    {
      key: SEED_FILESET_REF_KEY,
      label: 'Fileset',
      kind: 'text',
      required: true,
      helperText: 'The platform fileset to seed rows from.',
    },
    {
      key: SEED_FILE_PATH_KEY,
      label: 'File',
      kind: 'text',
      required: true,
      helperText: 'The file within the fileset to read rows from.',
    },
  ],
  custom: [
    {
      key: 'generation_strategy',
      label: 'Generation strategy',
      kind: 'select',
      options: asOptions(['cell_by_cell', 'full_column']),
    },
  ],
  // Shared across all sampler sub-types, emitted at the top level of the sampler config.
  // Sub-type-specific fields (collected into `params`) come from PARAM_FIELDS_BY_SAMPLER_TYPE.
  sampler: [
    {
      key: 'convert_to',
      label: 'Convert to (optional)',
      kind: 'text',
      placeholder: 'e.g. int, float, or %Y-%m-%d',
      helperText: 'Optional type conversion applied after sampling.',
    },
  ],
};

const BOOL_OPTIONS = [
  { label: 'Yes', value: 'true' },
  { label: 'No', value: 'false' },
] as const;

/** `p` (probability of success) field, shared by the Bernoulli-family samplers. */
const probabilityField = (helperText: string): ColumnField => ({
  key: 'p',
  label: 'Probability of success (p)',
  kind: 'number',
  dataType: 'number',
  required: true,
  placeholder: '0.0 – 1.0',
  helperText,
});

/** Optional `decimal_places` rounding field, shared by the continuous distributions. */
const DECIMAL_PLACES_FIELD: ColumnField = {
  key: 'decimal_places',
  label: 'Decimal places (optional)',
  kind: 'number',
  dataType: 'number',
  placeholder: 'e.g. 2',
  helperText: 'Round sampled values to this many decimal places.',
};

/** `dist_name` / `dist_params` fields, shared by the Scipy and Bernoulli-mixture samplers. */
const DIST_NAME_FIELD: ColumnField = {
  key: 'dist_name',
  label: 'Distribution name',
  kind: 'text',
  required: true,
  placeholder: 'e.g. beta, gamma, expon',
  helperText: 'A scipy.stats distribution name.',
};

const DIST_PARAMS_FIELD: ColumnField = {
  key: 'dist_params',
  label: 'Distribution parameters (JSON)',
  kind: 'textarea',
  dataType: 'json',
  required: true,
  placeholder: '{ "a": 2, "b": 5 }',
  helperText: 'JSON object of parameters for the scipy.stats distribution.',
};

/** `locale` / `sex` / `city` / `age_range` fields shared by the two Person samplers. */
const PERSON_SHARED_FIELDS: ColumnField[] = [
  {
    key: 'locale',
    label: 'Locale (optional)',
    kind: 'text',
    placeholder: 'e.g. en_US',
    helperText: 'Language and geographic region to sample from.',
  },
  {
    key: 'sex',
    label: 'Sex (optional)',
    kind: 'select',
    options: asOptions(Object.values(PersonSamplerParamsSex)),
    helperText: 'Restrict sampling to a single sex.',
  },
  {
    key: 'city',
    label: 'City (optional)',
    kind: 'text',
    list: true,
    placeholder: 'e.g. San Jose, Austin',
    helperText: 'Comma-separated city names to restrict to.',
  },
  {
    key: 'age_range',
    label: 'Age range (optional)',
    kind: 'text',
    dataType: 'number-list',
    placeholder: 'e.g. 18, 65',
    helperText: 'Two comma-separated ages: min, max.',
  },
];

/**
 * Sampler sub-type-specific fields, collected into the sampler config's required `params`
 * object (see `SamplerColumnConfig`). Every sampler sub-type surfaced in the palette is
 * listed; sub-types with no builder-editable params map to an empty array.
 */
const PARAM_FIELDS_BY_SAMPLER_TYPE: Partial<Record<SamplerType, ColumnField[]>> = {
  [SamplerType.uuid]: [
    {
      key: 'prefix',
      label: 'Prefix (optional)',
      kind: 'text',
      placeholder: 'e.g. user-',
      helperText: 'Prepended to each generated UUID.',
    },
    {
      key: 'short_form',
      label: 'Short form (optional)',
      kind: 'select',
      valueType: 'boolean',
      options: BOOL_OPTIONS,
      helperText: 'Truncate UUIDs to 8 characters.',
    },
    {
      key: 'uppercase',
      label: 'Uppercase (optional)',
      kind: 'select',
      valueType: 'boolean',
      options: BOOL_OPTIONS,
      helperText: 'Capitalize all letters in the UUID.',
    },
  ],
  [SamplerType.category]: [
    {
      key: 'values',
      label: 'Categories',
      kind: 'textarea',
      required: true,
      list: true,
      placeholder: 'science, technology, history, arts, business',
      helperText: 'Comma-separated values to sample from.',
    },
    {
      key: 'weights',
      label: 'Weights (optional)',
      kind: 'text',
      dataType: 'number-list',
      placeholder: 'e.g. 3, 1, 1, 2',
      helperText: 'Comma-separated weights, one per category, in order.',
    },
  ],
  [SamplerType.subcategory]: [
    {
      key: 'category',
      label: 'Parent category column',
      kind: 'text',
      required: true,
      reference: 'single',
      placeholder: 'Name of the parent category column',
      helperText: 'The category column this subcategory is conditioned on.',
    },
    {
      key: 'values',
      label: 'Subcategory values (JSON)',
      kind: 'textarea',
      dataType: 'json',
      required: true,
      placeholder: '{ "science": ["physics", "biology"], "arts": ["music"] }',
      helperText: 'JSON mapping each parent value to its list of subcategory values.',
    },
  ],
  [SamplerType.uniform]: [
    {
      key: 'low',
      label: 'Low',
      kind: 'number',
      dataType: 'number',
      required: true,
      placeholder: 'Lower bound (inclusive)',
      helperText: 'Lower bound of the range.',
    },
    {
      key: 'high',
      label: 'High',
      kind: 'number',
      dataType: 'number',
      required: true,
      placeholder: 'Upper bound',
      helperText: 'Upper bound of the range (must exceed low).',
    },
    DECIMAL_PLACES_FIELD,
  ],
  [SamplerType.gaussian]: [
    {
      key: 'mean',
      label: 'Mean',
      kind: 'number',
      dataType: 'number',
      required: true,
      placeholder: 'Center of the distribution',
      helperText: 'Mean (center) of the distribution.',
    },
    {
      key: 'stddev',
      label: 'Standard deviation',
      kind: 'number',
      dataType: 'number',
      required: true,
      placeholder: 'Spread of the distribution',
      helperText: 'Standard deviation; must be positive.',
    },
    DECIMAL_PLACES_FIELD,
  ],
  [SamplerType.bernoulli]: [probabilityField('Probability of sampling 1 (0.0 – 1.0).')],
  [SamplerType.bernoulli_mixture]: [
    probabilityField('Probability of sampling from the mixture distribution (0.0 – 1.0).'),
    DIST_NAME_FIELD,
    DIST_PARAMS_FIELD,
  ],
  [SamplerType.binomial]: [
    {
      key: 'n',
      label: 'Number of trials (n)',
      kind: 'number',
      dataType: 'number',
      required: true,
      placeholder: 'e.g. 10',
      helperText: 'Number of independent trials; a positive integer.',
    },
    probabilityField('Probability of success on each trial (0.0 – 1.0).'),
  ],
  [SamplerType.poisson]: [
    {
      key: 'mean',
      label: 'Mean (rate λ)',
      kind: 'number',
      dataType: 'number',
      required: true,
      placeholder: 'e.g. 4',
      helperText: 'Mean number of events per interval; must be positive.',
    },
  ],
  [SamplerType.scipy]: [DIST_NAME_FIELD, DIST_PARAMS_FIELD, DECIMAL_PLACES_FIELD],
  [SamplerType.person]: [
    ...PERSON_SHARED_FIELDS,
    {
      key: 'with_synthetic_personas',
      label: 'Include synthetic personas',
      kind: 'switch',
      dataType: 'boolean',
      helperText: 'Append synthetic persona columns (locale-dependent).',
    },
    {
      key: 'select_field_values',
      label: 'Field-value filters (JSON, optional)',
      kind: 'textarea',
      dataType: 'json',
      placeholder: '{ "occupation": ["engineer", "teacher"] }',
      helperText: 'JSON mapping managed-dataset fields to allowed values.',
    },
  ],
  [SamplerType.datetime]: [
    {
      key: 'start',
      label: 'Start',
      kind: 'text',
      required: true,
      placeholder: 'e.g. 2020-01-01',
      helperText: 'Earliest datetime (inclusive).',
    },
    {
      key: 'end',
      label: 'End',
      kind: 'text',
      required: true,
      placeholder: 'e.g. 2024-01-01',
      helperText: 'Exclusive upper bound.',
    },
    {
      key: 'unit',
      label: 'Unit (optional)',
      kind: 'select',
      options: asOptions(Object.values(DatetimeSamplerParamsUnit)),
      helperText: 'Sampling granularity (Y, M, D, h, m, s). Defaults to D.',
    },
  ],
  [SamplerType.timedelta]: [
    {
      key: 'dt_min',
      label: 'Minimum delta',
      kind: 'number',
      dataType: 'number',
      required: true,
      placeholder: 'e.g. 1',
      helperText: 'Minimum time-delta (inclusive, non-negative).',
    },
    {
      key: 'dt_max',
      label: 'Maximum delta',
      kind: 'number',
      dataType: 'number',
      required: true,
      placeholder: 'e.g. 30',
      helperText: 'Maximum time-delta (exclusive, greater than the minimum).',
    },
    {
      key: 'reference_column_name',
      label: 'Reference datetime column',
      kind: 'text',
      required: true,
      reference: 'single',
      placeholder: 'Name of an existing datetime column',
      helperText: 'The delta is added to values from this column.',
    },
    {
      key: 'unit',
      label: 'Unit (optional)',
      kind: 'select',
      options: asOptions(Object.values(TimeDeltaSamplerParamsUnit)),
      helperText: 'Delta unit (D, h, m, s). Defaults to D.',
    },
  ],
};

/** The sampler `params` fields for a sampler sub-type (empty for sub-types without any). */
const getSamplerParamFields = (samplerType: SamplerType | undefined): ColumnField[] =>
  samplerType ? (PARAM_FIELDS_BY_SAMPLER_TYPE[samplerType] ?? []) : [];

export const getColumnFields = (
  option: Pick<ColumnTypeOption, 'columnType' | 'samplerType'>
): ColumnField[] => {
  const { columnType, samplerType } = option;
  if (!columnType) return [];
  const base = FIELDS_BY_COLUMN_TYPE[columnType] ?? [];
  if (columnType === 'sampler') return [...getSamplerParamFields(samplerType), ...base];
  return base;
};

/**
 * The `values` a freshly created column starts with, from its fields' {@link
 * ColumnField.defaultValue}. Fields without one are omitted so the column stays empty
 * wherever no default applies.
 */
export const defaultColumnValues = (
  option: Pick<ColumnTypeOption, 'columnType' | 'samplerType'>
): Record<string, string> => {
  const values: Record<string, string> = {};
  for (const field of getColumnFields(option)) {
    if (field.defaultValue !== undefined) values[field.key] = field.defaultValue;
  }
  return values;
};

/** Accent color → NVIDIA Foundations text token, matching `CardNode`'s idle styling. */
const ACCENT_VAR_CLASS: Record<ColumnTypeColor, string> = {
  blue: 'text-[color:var(--text-color-accent-blue)]',
  gray: 'text-[color:var(--text-color-accent-gray)]',
  green: 'text-[color:var(--text-color-accent-green)]',
  purple: 'text-[color:var(--text-color-accent-purple)]',
  red: 'text-[color:var(--text-color-accent-red)]',
  teal: 'text-[color:var(--text-color-accent-teal)]',
  yellow: 'text-[color:var(--text-color-accent-yellow)]',
};

export const findColumnOption = (selection: AddColumnSelection): ColumnTypeOption | undefined => {
  for (const group of COLUMN_TYPE_GROUPS) {
    const match = group.options.find(
      (option) =>
        option.columnType === selection.columnType && option.samplerType === selection.samplerType
    );
    if (match) return match;
  }
  return undefined;
};

// Specs whose column type can't be matched in the palette are silently skipped.
export const buildColumnsFromTemplate = (
  specs: readonly TemplateColumnSpec[],
  startId = 0
): BuilderColumn[] => {
  const columns: BuilderColumn[] = [];
  let nextId = startId;
  for (const spec of specs) {
    const option = findColumnOption(spec);
    if (!option) continue;
    columns.push({
      id: `col-${nextId++}`,
      option,
      name: spec.name,
      values: { ...defaultColumnValues(option), ...spec.values },
    });
  }
  return columns;
};

const JINJA_REF = /\{\{-?\s*([a-zA-Z_][a-zA-Z0-9_]*)/g;

export const extractJinjaReferences = (text: string): string[] => {
  const refs: string[] = [];
  for (const match of text.matchAll(JINJA_REF)) refs.push(match[1]);
  return refs;
};

const columnDependencies = (column: BuilderColumn, knownNames: Set<string>): Set<string> => {
  const deps = new Set<string>();
  const add = (candidate: string) => {
    const name = candidate.trim();
    if (name && name !== column.name && knownNames.has(name)) deps.add(name);
  };
  for (const field of getColumnFields(column.option)) {
    const value = column.values[field.key]?.trim();
    if (!value) continue;
    switch (field.reference) {
      case 'jinja':
        extractJinjaReferences(value).forEach(add);
        break;
      case 'single':
        add(value);
        break;
      case 'list':
        value.split(',').forEach(add);
        break;
    }
  }
  return deps;
};

/**
 * The names of existing columns this column references (the same dependencies that draw
 * edges in {@link buildGraph}). Used by the schema-list renderer to show each column's
 * relationships inline instead of as graph edges.
 */
export const getColumnReferences = (column: BuilderColumn, knownNames: Set<string>): string[] => [
  ...columnDependencies(column, knownNames),
];

/**
 * The set of referenceable column names and the name→id lookup used to resolve
 * dependencies. Seed-dataset columns also contribute the names they provide, mapped back
 * to the seed column. Shared by {@link buildGraph} and {@link topologicalSortColumns} so
 * both agree on what counts as a dependency.
 */
export const buildNameMaps = (
  columns: BuilderColumn[]
): { knownNames: Set<string>; idByName: Map<string, string> } => {
  const knownNames = new Set(columns.map((column) => column.name).filter(Boolean));
  const idByName = new Map(columns.filter((c) => c.name).map((c) => [c.name, c.id]));

  for (const column of columns) {
    if (column.option.columnType !== 'seed-dataset') continue;
    for (const name of getSeedAvailableColumns(column)) {
      knownNames.add(name);
      if (!idByName.has(name)) idByName.set(name, column.id);
    }
  }
  return { knownNames, idByName };
};

/**
 * Orders columns into dependency layers: each column's depth is the longest chain of
 * references beneath it (a column with no references is depth 0), and columns are sorted by
 * that depth ascending. So every column still comes after everything it references, and
 * shallower columns group above deeper ones regardless of the order they were added — a
 * column that depends only on a root sits above one that depends on a mid-chain column.
 * Equal-depth ties keep their original relative order; reference cycles resolve to a finite
 * depth without looping.
 */
export const topologicalSortColumns = (columns: BuilderColumn[]): BuilderColumn[] => {
  const { knownNames, idByName } = buildNameMaps(columns);
  const depsById = new Map<string, string[]>(
    columns.map((column) => [
      column.id,
      [...columnDependencies(column, knownNames)]
        .map((name) => idByName.get(name))
        .filter((id): id is string => !!id && id !== column.id),
    ])
  );

  const depthById = new Map<string, number>();
  const onPath = new Set<string>();
  const depthOf = (id: string): number => {
    const cached = depthById.get(id);
    if (cached !== undefined) return cached;
    if (onPath.has(id)) return 0; // cycle edge: stop descending, contribute no depth
    onPath.add(id);
    const deps = depsById.get(id) ?? [];
    const depth = deps.length === 0 ? 0 : 1 + Math.max(...deps.map(depthOf));
    onPath.delete(id);
    depthById.set(id, depth);
    return depth;
  };

  // Stable sort by depth, then by original index, so ties keep their input order.
  return columns
    .map((column, index) => ({ column, index, depth: depthOf(column.id) }))
    .sort((a, b) => a.depth - b.depth || a.index - b.index)
    .map((entry) => entry.column);
};

export const buildGraph = (columns: BuilderColumn[]): { nodes: DagNode[]; edges: DagEdge[] } => {
  const { knownNames, idByName } = buildNameMaps(columns);

  const nodes: DagNode[] = [];
  const edges: DagEdge[] = [];
  const edgeKeys = new Set<string>();

  for (const column of columns) {
    const { option } = column;
    const deps = columnDependencies(column, knownNames);
    const tags =
      option.columnType === 'seed-dataset'
        ? getSeedAvailableColumns(column).map((name) => `{{${name}}}`)
        : [...deps].map((name) => `{{${name}}}`);
    nodes.push({
      id: column.id,
      data: {
        title: column.name || option.label,
        type: (option.columnType ?? '').replace(/-/g, ' ').toUpperCase(),
        description: option.description,
        icon: option.icon,
        colorClassName: ACCENT_VAR_CLASS[option.color],
        tags,
      },
    });
    for (const dep of deps) {
      const source = idByName.get(dep);
      if (!source || source === column.id) continue;
      const key = `${source}->${column.id}`;
      if (edgeKeys.has(key)) continue;
      edgeKeys.add(key);
      edges.push({ source, target: column.id });
    }
  }

  return { nodes, edges };
};

export const defaultColumnName = (option: ColumnTypeOption, takenNames: Set<string>): string => {
  const base = (option.samplerType ?? option.columnType ?? 'column').replace(/[^a-zA-Z0-9]+/g, '_');
  for (let n = 1; ; n++) {
    const candidate = `${base}_${n}`;
    if (!takenNames.has(candidate)) return candidate;
  }
};

export const validateColumnName = (name: string, takenNames: Set<string>): string | null => {
  const trimmed = name.trim();
  if (!trimmed) return 'Name is required.';
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(trimmed)) {
    return 'Use letters, numbers, and underscores; must not start with a number.';
  }
  if (takenNames.has(trimmed)) return 'A column with this name already exists.';
  return null;
};

/**
 * Reports why a filled-in field value is malformed for its {@link ColumnField.dataType}
 * (bad number, invalid JSON, non-numeric list entry), or `null` if it is well-formed.
 * `output_format` is treated as JSON regardless of its declared type.
 */
const fieldValueError = (field: ColumnField, value: string): string | null => {
  const isJson = field.dataType === 'json' || field.key === 'output_format';
  if (isJson) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(value);
    } catch {
      return `${field.label} must be valid JSON.`;
    }
    if (field.key === 'scores' && !Array.isArray(parsed)) {
      return `${field.label} must be a JSON array.`;
    }
    return null;
  }
  if (field.dataType === 'number' && !Number.isFinite(Number(value))) {
    return `${field.label} must be a number.`;
  }
  if (
    field.dataType === 'number-list' &&
    splitList(value).some((v) => !Number.isFinite(Number(v)))
  ) {
    return `${field.label} must be a comma-separated list of numbers.`;
  }
  return null;
};

/**
 * Validates every column is ready to submit: unique, well-formed names; every field
 * marked `required` in {@link getColumnFields} filled in; and typed fields (numbers, JSON,
 * numeric lists) parse. Returns one human-readable message per problem found, or an
 * empty array if the recipe is submittable.
 */
export const validateColumns = (columns: BuilderColumn[]): string[] => {
  if (columns.length === 0) return ['Add at least one column before creating the job.'];

  const errors: string[] = [];
  for (const column of columns) {
    const label = column.name || column.option.label;
    const takenNames = new Set(
      columns.filter((other) => other.id !== column.id).map((other) => other.name)
    );
    const nameError = validateColumnName(column.name, takenNames);
    if (nameError) errors.push(`${label}: ${nameError}`);

    for (const field of getColumnFields(column.option)) {
      const value = column.values[field.key]?.trim();
      if (!value) {
        if (field.required) errors.push(`${label}: ${field.label} is required.`);
        continue;
      }
      const valueError = fieldValueError(field, value);
      if (valueError) errors.push(`${label}: ${valueError}`);
    }
  }
  return errors;
};

/** Splits a comma-separated field value into trimmed, non-empty entries. */
const splitList = (value: string): string[] =>
  value
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);

/**
 * Coerces a field's trimmed string value into the type the SDK config expects, per the
 * field's {@link ColumnField.dataType} (and the legacy `list` / `reference: 'list'` flags).
 * Assumes the value is non-empty and already validated by {@link validateColumns}.
 */
const serializeFieldValue = (field: ColumnField, value: string): unknown => {
  switch (field.dataType) {
    case 'number':
      return Number(value);
    case 'boolean':
      return value === 'true';
    case 'json':
      return JSON.parse(value);
    case 'number-list':
      return splitList(value).map(Number);
    default:
      return field.list || field.reference === 'list' ? splitList(value) : value;
  }
};

/**
 * Converts a sampler column into the SDK's `SamplerColumnConfig` shape. Sub-type params
 * are nested under the required `params` object; `convert_to` stays at the top level.
 */
const toSamplerConfig = (column: BuilderColumn): Record<string, unknown> => {
  const params: Record<string, unknown> = {};
  for (const field of getSamplerParamFields(column.option.samplerType)) {
    const value = column.values[field.key]?.trim();
    if (!value) continue;
    params[field.key] = serializeFieldValue(field, value);
  }

  const config: Record<string, unknown> = {
    name: column.name,
    column_type: 'sampler',
    sampler_type: column.option.samplerType,
    params,
  };
  const convertTo = column.values.convert_to?.trim();
  if (convertTo) config.convert_to = convertTo;
  return config;
};

const toColumnConfig = (column: BuilderColumn): Record<string, unknown> => {
  if (column.option.columnType === 'sampler') return toSamplerConfig(column);

  const config: Record<string, unknown> = {
    name: column.name,
    column_type: column.option.columnType,
  };

  for (const field of getColumnFields(column.option)) {
    const value = column.values[field.key]?.trim();
    if (!value) continue;
    config[field.key] =
      field.key === 'output_format' ? JSON.parse(value) : serializeFieldValue(field, value);
  }
  return config;
};

/** Assembles the `FilesetFileSeedSource` composite path: `{workspace}/{fileset}#{file}`. */
export const buildSeedFilesetPath = (filesetRef: string, filePath: string): string =>
  `${filesetRef}#${filePath}`;

const buildSeedConfig = (columns: BuilderColumn[]): SeedConfig | undefined => {
  const seedColumn = columns.find(
    (column) =>
      column.option.columnType === 'seed-dataset' &&
      column.values[SEED_FILESET_REF_KEY]?.trim() &&
      column.values[SEED_FILE_PATH_KEY]?.trim()
  );
  if (!seedColumn) return undefined;

  const path = buildSeedFilesetPath(
    seedColumn.values[SEED_FILESET_REF_KEY].trim(),
    seedColumn.values[SEED_FILE_PATH_KEY].trim()
  );
  const seedConfig: SeedConfig = {
    source: { seed_type: 'nhx', path },
  };
  const samplingStrategy = seedColumn.values[SEED_SAMPLING_STRATEGY_KEY]?.trim();
  if (samplingStrategy) seedConfig.sampling_strategy = samplingStrategy as SamplingStrategy;
  return seedConfig;
};

export const buildDataDesignerConfig = (
  columns: BuilderColumn[],
  models: BuilderModel[] = [],
  servedModelNames: Map<string, string> = new Map()
): DataDesignerConfig => {
  const config: DataDesignerConfig = {
    columns: columns
      .filter((column) => column.option.columnType !== 'seed-dataset')
      .map(toColumnConfig) as unknown as DataDesignerConfig['columns'],
  };
  const modelConfigs = buildModelConfigs(models, servedModelNames);
  if (modelConfigs) config.model_configs = modelConfigs;
  const seedConfig = buildSeedConfig(columns);
  if (seedConfig) config.seed_config = seedConfig;
  return config;
};

/**
 * Inverse of {@link serializeFieldValue}: renders a column config's stored value back into
 * the string form the builder edits. Mirrors the same {@link ColumnField.dataType} branches
 * so a config → builder → config round-trip reproduces the original value. JSON fields are
 * pretty-printed for readability in the textarea.
 */
const deserializeFieldValue = (field: ColumnField, raw: unknown): string => {
  if (raw == null) return '';
  if (field.dataType === 'json' || field.key === 'output_format') {
    return JSON.stringify(raw, null, 2);
  }
  switch (field.dataType) {
    case 'number':
      return String(raw);
    case 'boolean':
      return raw ? 'true' : 'false';
    case 'number-list':
      return Array.isArray(raw) ? raw.join(', ') : String(raw);
    default:
      return field.list || field.reference === 'list'
        ? Array.isArray(raw)
          ? raw.join(', ')
          : String(raw)
        : String(raw);
  }
};

/**
 * Reconstructs a column's `values` record from its SDK config. Sampler param fields are read
 * from the nested `params` object; every other field (including a sampler's `convert_to`)
 * reads from the config's top level, matching how {@link toColumnConfig}/{@link toSamplerConfig}
 * wrote them.
 */
const valuesFromConfig = (
  option: ColumnTypeOption,
  columnConfig: Record<string, unknown>
): Record<string, string> => {
  const values: Record<string, string> = {};
  const paramKeys =
    option.columnType === 'sampler'
      ? new Set(getSamplerParamFields(option.samplerType).map((field) => field.key))
      : null;
  const params = (columnConfig.params as Record<string, unknown> | undefined) ?? {};

  for (const field of getColumnFields(option)) {
    const source = paramKeys?.has(field.key) ? params : columnConfig;
    const value = deserializeFieldValue(field, source[field.key]);
    if (value !== '') values[field.key] = value;
  }
  return values;
};

/** Inverse of {@link buildSeedFilesetPath}: splits `{fileset}#{file}` back into its parts. */
const parseSeedFilesetPath = (path: string): { filesetRef: string; filePath: string } => {
  const hashIndex = path.indexOf('#');
  if (hashIndex === -1) return { filesetRef: path, filePath: '' };
  return { filesetRef: path.slice(0, hashIndex), filePath: path.slice(hashIndex + 1) };
};

/**
 * Rebuilds the builder's seed-dataset column from `seed_config` (which is where
 * {@link buildDataDesignerConfig} moves seed columns — they never live in `config.columns`).
 * Only NHX fileset sources map back; other seed source kinds aren't builder-editable. The
 * seed file's discovered columns aren't recoverable from the config, so `available_columns`
 * stays empty until the fileset is re-inspected.
 */
const seedColumnFromConfig = (
  seedConfig: SeedConfig | undefined,
  id: string
): BuilderColumn | null => {
  const source = seedConfig?.source as FilesetFileSeedSource | undefined;
  if (!source || source.seed_type !== 'nhx' || typeof source.path !== 'string') return null;
  const option = findColumnOption({ columnType: 'seed-dataset' });
  if (!option) return null;

  const { filesetRef, filePath } = parseSeedFilesetPath(source.path);
  const values: Record<string, string> = {
    [SEED_FILESET_REF_KEY]: filesetRef,
    [SEED_FILE_PATH_KEY]: filePath,
  };
  if (seedConfig?.sampling_strategy) {
    values[SEED_SAMPLING_STRATEGY_KEY] = seedConfig.sampling_strategy;
  }
  return { id, option, name: 'seed', values };
};

/**
 * Reverses {@link buildDataDesignerConfig} into builder columns so an existing job's config
 * can pre-fill the build canvas (used when cloning a job). Column configs whose type isn't in
 * the palette are skipped; the seed column is reconstructed from `seed_config`.
 */
export const buildColumnsFromConfig = (
  config: DataDesignerConfig,
  startId = 0
): BuilderColumn[] => {
  const columns: BuilderColumn[] = [];
  let nextId = startId;

  const seedColumn = seedColumnFromConfig(config.seed_config, `col-${nextId}`);
  if (seedColumn) {
    columns.push(seedColumn);
    nextId++;
  }

  for (const columnConfig of config.columns) {
    if (columnConfig.column_type === 'seed-dataset') continue;
    const option = findColumnOption({
      columnType: columnConfig.column_type,
      samplerType: (columnConfig as SamplerColumnConfig).sampler_type,
    });
    if (!option) continue;
    columns.push({
      id: `col-${nextId++}`,
      option,
      name: columnConfig.name,
      values: valuesFromConfig(option, columnConfig as unknown as Record<string, unknown>),
    });
  }
  return columns;
};
