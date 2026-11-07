// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  DatetimeSamplerParamsUnit,
  ExpressionColumnConfigDtype,
  PersonSamplerParamsSex,
  SamplerType,
  SamplingStrategy,
  TimeDeltaSamplerParamsUnit,
  ValidatorType,
} from '@nemo/sdk/generated/data-designer/schema';
import type { ChatCompletionTool } from 'openai/resources/index.mjs';

const COLUMN_TYPE_DESCRIPTION = `Each column starts with "name" then "column_type" (one of: "sampler", "expression", "llm-text", "llm-structured", "llm-judge", "llm-code", "validation"), followed by that type's fields in the order listed here.
- sampler: "sampler_type", then "params" (object), both set on the column itself, then optional "convert_to" (e.g. "int", "float", or a date format like "%Y-%m-%d"). "params" is never empty except for "uuid". Examples: "category" -> { "values": ["a", "b"], "weights"?: number[] }; "subcategory" -> { "category": "<parent column name>", "values": { "<parent value>": ["x", "y"] } } (parent column must come earlier, and "values" needs a key for every parent value); "uniform" -> { "low", "high" }; "gaussian" -> { "mean", "stddev" }; "datetime" -> { "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "unit"?: "D" }; "uuid" -> {} or { "prefix" }. Use "subcategory" for values that depend on another column instead of an expression.
- expression: "expr" (Jinja2 only; not Python), then optional "dtype" ("int"|"float"|"str"|"bool"). Reference other columns only as {{ column_name }}, never bare identifiers. Do not use Python ternary forms like "1 if cond else 0" without proper Jinja wrapping and {{ }} output. Order columns so expr only references earlier columns. Avoid keyword substring rules for sentiment—use sampler or LLM columns instead.
- llm-text: "prompt", "model_alias", then optional "system_prompt", "extract_reasoning_content". Prompt may only reference columns listed earlier in "columns" (exact {{ name }} match).
- llm-structured, llm-judge, llm-code: "prompt", "model_alias", then "output_format", "scores" or "code_lang" respectively, then the same optional fields as llm-text.
- validation: "target_columns", "validator_type", "validator_params". Prefer sampler/expression/llm-text for simple specs.`;

const SCALAR_VALUE = { anyOf: [{ type: 'string' }, { type: 'number' }] };

/** Mirrors the sampler `params` variants of the Data Designer OpenAPI schema. */
const SAMPLER_PARAMS_SCHEMA = {
  description:
    'Sampler parameters. Must match sampler_type; required fields are listed per variant.',
  anyOf: [
    {
      description: 'sampler_type "category"',
      type: 'object',
      required: ['values'],
      properties: {
        values: { type: 'array', minItems: 1, items: SCALAR_VALUE },
        weights: { type: 'array', items: { type: 'number' } },
      },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "subcategory"',
      type: 'object',
      required: ['category', 'values'],
      properties: {
        category: { type: 'string', description: 'Name of the parent category column.' },
        values: {
          type: 'object',
          description: 'Maps each parent category value to its list of subcategory values.',
          additionalProperties: { type: 'array', items: SCALAR_VALUE },
        },
      },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "uniform"',
      type: 'object',
      required: ['low', 'high'],
      properties: {
        low: { type: 'number' },
        high: { type: 'number' },
        decimal_places: { type: 'integer' },
      },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "gaussian"',
      type: 'object',
      required: ['mean', 'stddev'],
      properties: {
        mean: { type: 'number' },
        stddev: { type: 'number' },
        decimal_places: { type: 'integer' },
      },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "poisson"',
      type: 'object',
      required: ['mean'],
      properties: { mean: { type: 'number' } },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "bernoulli"',
      type: 'object',
      required: ['p'],
      properties: { p: { type: 'number' } },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "binomial"',
      type: 'object',
      required: ['n', 'p'],
      properties: { n: { type: 'integer' }, p: { type: 'number' } },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "bernoulli_mixture"',
      type: 'object',
      required: ['p', 'dist_name', 'dist_params'],
      properties: {
        p: { type: 'number' },
        dist_name: { type: 'string' },
        dist_params: { type: 'object', additionalProperties: true },
      },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "scipy"',
      type: 'object',
      required: ['dist_name', 'dist_params'],
      properties: {
        dist_name: { type: 'string' },
        dist_params: { type: 'object', additionalProperties: true },
        decimal_places: { type: 'integer' },
      },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "timedelta"',
      type: 'object',
      required: ['dt_min', 'dt_max', 'reference_column_name'],
      properties: {
        dt_min: { type: 'integer' },
        dt_max: { type: 'integer' },
        reference_column_name: { type: 'string' },
        unit: { type: 'string', enum: Object.values(TimeDeltaSamplerParamsUnit) },
      },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "datetime"',
      type: 'object',
      required: ['start', 'end'],
      properties: {
        start: { type: 'string', description: 'Inclusive start, e.g. YYYY-MM-DD.' },
        end: { type: 'string', description: 'Exclusive end, e.g. YYYY-MM-DD.' },
        unit: { type: 'string', enum: Object.values(DatetimeSamplerParamsUnit) },
      },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "uuid"; all fields optional',
      type: 'object',
      properties: {
        prefix: { type: 'string' },
        short_form: { type: 'boolean' },
        uppercase: { type: 'boolean' },
      },
      additionalProperties: false,
    },
    {
      description: 'sampler_type "person" or "person_from_faker"; all fields optional',
      type: 'object',
      properties: {
        locale: { type: 'string' },
        sex: { type: 'string', enum: Object.values(PersonSamplerParamsSex) },
        age_range: { type: 'array', items: { type: 'integer' } },
        city: { anyOf: [{ type: 'string' }, { type: 'array', items: { type: 'string' } }] },
        select_field_values: { type: 'object', additionalProperties: true },
        with_synthetic_personas: { type: 'boolean' },
      },
      additionalProperties: false,
    },
  ],
};

const COLUMN_NAME = { type: 'string', description: 'Column name.' };

const PROMPT = {
  type: 'string',
  description: 'Prompt template; reference earlier columns as {{ column_name }}.',
};

const MODEL_ALIAS = { type: 'string', description: 'Alias of an entry in config.model_configs.' };

const LLM_OPTIONAL_FIELDS = {
  system_prompt: {
    type: 'string',
    description: 'Use to constrain style (e.g. no chain-of-thought in the answer).',
  },
  extract_reasoning_content: {
    type: 'boolean',
    description:
      'If true, reasoning split by the provider goes to {column}__reasoning_content instead of the value.',
  },
};

// llama.cpp grammars emit required keys before optional ones, so declare them in that order too.
const columnSchema = (
  columnType: string,
  requiredFields: Record<string, object>,
  optionalFields: Record<string, object> = {}
) => ({
  type: 'object',
  required: ['name', 'column_type', ...Object.keys(requiredFields)],
  properties: {
    name: COLUMN_NAME,
    column_type: { type: 'string', enum: [columnType] },
    ...requiredFields,
    ...optionalFields,
  },
  additionalProperties: false,
});

const COLUMN_SCHEMAS = [
  columnSchema(
    'sampler',
    {
      sampler_type: { type: 'string', enum: Object.values(SamplerType) },
      params: SAMPLER_PARAMS_SCHEMA,
    },
    {
      convert_to: {
        type: 'string',
        description: 'Optional conversion after sampling: "int", "float", "str", or a date format.',
      },
    }
  ),
  columnSchema(
    'expression',
    {
      expr: {
        type: 'string',
        description:
          'Full Jinja2 template; use {{ col }} for every column reference (not bare col). No Python if/else expressions unless expressed as Jinja {{ a if b else c }}.',
      },
    },
    { dtype: { type: 'string', enum: Object.values(ExpressionColumnConfigDtype) } }
  ),
  columnSchema('llm-text', { prompt: PROMPT, model_alias: MODEL_ALIAS }, LLM_OPTIONAL_FIELDS),
  columnSchema(
    'llm-structured',
    {
      prompt: PROMPT,
      model_alias: MODEL_ALIAS,
      output_format: { type: 'object', description: 'JSON schema of the output.' },
    },
    LLM_OPTIONAL_FIELDS
  ),
  columnSchema(
    'llm-judge',
    {
      prompt: PROMPT,
      model_alias: MODEL_ALIAS,
      scores: {
        type: 'array',
        minItems: 1,
        description: 'Scoring rubrics: { name, description, options }.',
        items: { type: 'object' },
      },
    },
    LLM_OPTIONAL_FIELDS
  ),
  columnSchema(
    'llm-code',
    {
      prompt: PROMPT,
      model_alias: MODEL_ALIAS,
      code_lang: { type: 'string', description: 'Target language, e.g. "python".' },
    },
    LLM_OPTIONAL_FIELDS
  ),
  columnSchema('validation', {
    target_columns: { type: 'array', minItems: 1, items: { type: 'string' } },
    validator_type: { type: 'string', enum: Object.values(ValidatorType) },
    validator_params: {
      type: 'object',
      description: 'For "code": { "code_lang": "python" }. For "remote": { "url": "https://..." }.',
    },
  }),
];

const MODEL_CONFIG_SCHEMA = {
  type: 'object',
  required: ['alias', 'model', 'provider'],
  properties: {
    alias: { type: 'string', description: 'Short stable alias, e.g. "review-model".' },
    model: { type: 'string', description: 'Model identifier (e.g. workspace/model-name).' },
    provider: {
      type: 'string',
      description:
        'Model provider name. Use the provider part of the model or "workspace/provider-name" if workspace-scoped.',
    },
    inference_parameters: {
      type: 'object',
      properties: {
        generation_type: { type: 'string', enum: ['chat-completion'] },
        max_tokens: { type: 'integer', minimum: 1 },
        temperature: { type: 'number' },
        top_p: { type: 'number' },
        max_parallel_requests: { type: 'integer', minimum: 1 },
        extra_body: {
          type: 'object',
          description:
            'Provider-specific JSON merged into the chat request (e.g. flags to disable thinking mode); keys depend on the inference backend.',
        },
      },
      additionalProperties: false,
    },
  },
  additionalProperties: false,
};

const SEED_CONFIG_SCHEMA = {
  type: 'object',
  description: 'Only when the user names an existing dataset to seed rows from.',
  required: ['source'],
  properties: {
    source: {
      type: 'object',
      required: ['seed_type', 'path'],
      properties: {
        seed_type: { type: 'string', enum: ['nhx'] },
        path: {
          type: 'string',
          description:
            'Fileset file as "<workspace>/<fileset>#<file>", e.g. "default/reviews#data.csv".',
        },
      },
      additionalProperties: false,
    },
    sampling_strategy: { type: 'string', enum: Object.values(SamplingStrategy) },
    selection_strategy: { type: 'object' },
  },
  additionalProperties: false,
};

/**
 * Tool definition for the model to call when generating a Data Designer job request.
 * Accepts a strict subset of PreviewRequest / DataDesignerJobRequest. `model_configs` is required
 * (possibly empty) so order-enforcing grammars emit it before `columns`, not after on the wrong level.
 */
export const generateDataDesignerJobRequestTool: ChatCompletionTool = {
  type: 'function',
  function: {
    name: 'generate_data_designer_job_request',
    description: `Generate a Data Designer job request from the user's description, writing keys in schema order. ${COLUMN_TYPE_DESCRIPTION}`,
    parameters: {
      type: 'object',
      required: ['job_request'],
      properties: {
        job_request: {
          type: 'object',
          required: ['name', 'spec'],
          properties: {
            name: {
              type: 'string',
              description:
                'Lowercase job name starting with a letter; letters, digits and hyphens only (e.g. "my-data-job").',
            },
            spec: {
              type: 'object',
              required: ['num_records', 'config'],
              properties: {
                num_records: {
                  type: 'integer',
                  minimum: 1,
                  description: 'Number of records to generate.',
                },
                config: {
                  type: 'object',
                  required: ['model_configs', 'columns'],
                  properties: {
                    model_configs: {
                      type: 'array',
                      description:
                        'Every alias referenced by a column model_alias, defined before the columns. [] if no column uses a model.',
                      items: MODEL_CONFIG_SCHEMA,
                    },
                    columns: {
                      type: 'array',
                      minItems: 1,
                      description:
                        'Columns in dependency order. Each starts with name, column_type, then the fields of that type.',
                      items: { anyOf: COLUMN_SCHEMAS },
                    },
                    seed_config: SEED_CONFIG_SCHEMA,
                  },
                  additionalProperties: false,
                },
              },
              additionalProperties: false,
            },
          },
          additionalProperties: false,
        },
      },
      additionalProperties: false,
    },
  },
};
