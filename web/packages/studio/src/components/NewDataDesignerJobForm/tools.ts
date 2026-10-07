// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ChatCompletionTool } from 'openai/resources/index.mjs';

const COLUMN_TYPE_DESCRIPTION = `Each column must have "name" (string) and "column_type" (string, one of: "expression", "sampler", "llm-text", "llm-code", "llm-judge", "llm-structured", "seed-dataset", "validation", "embedding", "custom").
- expression: requires "expr" (Jinja2 only; not Python). Reference other columns only as {{ column_name }}, never bare identifiers. Do not use Python ternary forms like "1 if cond else 0" without proper Jinja wrapping and {{ }} output. Optional "dtype" ("int"|"float"|"str"|"bool"). Order columns so expr only references earlier columns. Avoid keyword substring rules for sentiment—use sampler or LLM columns instead.
- sampler: requires "sampler_type" and "params" (object), both set on the column itself. "params" is never empty except for "uuid". Examples: "category" -> { "values": ["a", "b"], "weights"?: number[] }; "subcategory" -> { "category": "<parent column name>", "values": { "<parent value>": ["x", "y"] } } (parent column must come earlier, and "values" needs a key for every parent value); "uniform" -> { "low", "high" }; "gaussian" -> { "mean", "stddev" }; "datetime" -> { "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "unit"?: "D" }; "uuid" -> {} or { "prefix" }. Use "subcategory" for values that depend on another column instead of an expression.
- llm-text: requires "prompt" (string), "model_alias" (string). Optional "system_prompt". Prompt may only reference columns listed earlier in "columns" (exact {{ name }} match).
- llm-code additionally requires "code_lang"; llm-judge requires "scores"; llm-structured requires "output_format". All three also require "prompt" and "model_alias".
- seed-dataset, validation, embedding, custom: require type-specific fields; prefer expression/sampler/llm-text for simple specs.`;

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
        unit: { type: 'string', enum: ['D', 'h', 'm', 's'] },
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
        unit: { type: 'string', enum: ['Y', 'M', 'D', 'h', 'm', 's'] },
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
        sex: { type: 'string', enum: ['Male', 'Female'] },
        age_range: { type: 'array', items: { type: 'integer' } },
        city: { anyOf: [{ type: 'string' }, { type: 'array', items: { type: 'string' } }] },
        select_field_values: { type: 'object', additionalProperties: true },
        with_synthetic_personas: { type: 'boolean' },
      },
      additionalProperties: false,
    },
  ],
};

/**
 * Tool definition for the model to call when generating a Data Designer job request.
 * Schema is aligned with PreviewRequest and DataDesignerJobRequest so the payload
 * passes API validation (avoids 422).
 */
export const generateDataDesignerJobRequestTool: ChatCompletionTool = {
  type: 'function',
  function: {
    name: 'generate_data_designer_job_request',
    description: `Generate a Data Designer job request from the user's description. Call with job_request containing a valid spec: spec.num_records (positive integer) and spec.config (object with "columns" array, at least one column). Optional: job_request.name, job_request.description, job_request.project. ${COLUMN_TYPE_DESCRIPTION} If using LLM columns, include spec.config.model_configs: array of { "alias": string, "model": string, "provider": string (required), "inference_parameters"?: { "generation_type": "chat-completion", "max_tokens"?: number } }. Each model_config MUST have "provider": the model provider name. Keep config minimal and valid for preview.`,
    parameters: {
      type: 'object',
      properties: {
        job_request: {
          type: 'object',
          description:
            'Data Designer job request: optional name, description, project; required spec with num_records and config.',
          required: ['spec'],
          properties: {
            name: {
              type: 'string',
              description:
                'Optional job name. Must not contain spaces; use hyphens or underscores (e.g. "my-data-job").',
            },
            description: { type: 'string', description: 'Optional job description.' },
            project: { type: 'string', description: 'Optional project.' },
            spec: {
              type: 'object',
              description:
                'Required. num_records (positive integer) and config (object with columns array).',
              required: ['num_records', 'config'],
              properties: {
                num_records: {
                  type: 'number',
                  description: 'Number of records to generate. Must be a positive integer.',
                  minimum: 1,
                },
                config: {
                  type: 'object',
                  description:
                    'Data Designer config. Must have "columns" (array, at least one column). Optional: model_configs.',
                  required: ['columns'],
                  properties: {
                    columns: {
                      type: 'array',
                      minItems: 1,
                      description:
                        'Column configs. Each item: name (string), column_type (string), plus type-specific fields. See main description.',
                      items: {
                        type: 'object',
                        required: ['name', 'column_type'],
                        properties: {
                          name: { type: 'string', description: 'Column name.' },
                          column_type: {
                            type: 'string',
                            enum: [
                              'expression',
                              'sampler',
                              'llm-text',
                              'llm-code',
                              'llm-judge',
                              'llm-structured',
                              'seed-dataset',
                              'validation',
                              'embedding',
                              'custom',
                            ],
                            description: 'Discriminator for column config type.',
                          },
                          drop: {
                            type: 'boolean',
                            description: 'Optional. If true, column is dropped from output.',
                          },
                          expr: {
                            type: 'string',
                            description:
                              'For column_type expression: full Jinja2 template; use {{ col }} for every column reference (not bare col). No Python if/else expressions unless expressed as Jinja {{ a if b else c }}.',
                          },
                          dtype: {
                            type: 'string',
                            enum: ['int', 'float', 'str', 'bool'],
                            description: 'For expression: result type.',
                          },
                          sampler_type: {
                            type: 'string',
                            enum: [
                              'uuid',
                              'category',
                              'subcategory',
                              'uniform',
                              'gaussian',
                              'bernoulli',
                              'bernoulli_mixture',
                              'binomial',
                              'poisson',
                              'scipy',
                              'person',
                              'person_from_faker',
                              'datetime',
                              'timedelta',
                            ],
                            description: 'For column_type sampler: sampler type.',
                          },
                          params: SAMPLER_PARAMS_SCHEMA,
                          code_lang: {
                            type: 'string',
                            description: 'Required for llm-code: target language, e.g. "python".',
                          },
                          scores: {
                            type: 'array',
                            description: 'Required for llm-judge: scoring rubrics.',
                            items: { type: 'object' },
                          },
                          output_format: {
                            type: 'object',
                            description: 'Required for llm-structured: JSON schema of the output.',
                            additionalProperties: true,
                          },
                          prompt: {
                            type: 'string',
                            description:
                              'For llm-text/llm-code/llm-judge/llm-structured: prompt template.',
                          },
                          model_alias: {
                            type: 'string',
                            description: 'For LLM columns: alias from model_configs.',
                          },
                          system_prompt: {
                            type: 'string',
                            description:
                              'Optional for LLM columns. Use to constrain style (e.g. no chain-of-thought in the answer).',
                          },
                          extract_reasoning_content: {
                            type: 'boolean',
                            description:
                              'For llm-text (and similar): if true, reasoning split by the provider goes to {column}__reasoning_content; helps when content vs reasoning are separate fields.',
                          },
                        },
                        additionalProperties: true,
                      },
                    },
                    model_configs: {
                      type: 'array',
                      description:
                        'Required if any column uses model_alias. Each item: alias, model, provider (required), optional inference_parameters.',
                      items: {
                        type: 'object',
                        required: ['alias', 'model', 'provider'],
                        properties: {
                          alias: { type: 'string' },
                          skip_health_check: { type: 'boolean' },
                          model: {
                            type: 'string',
                            description: 'Model identifier (e.g. workspace/model-name).',
                          },
                          provider: {
                            type: 'string',
                            description:
                              'Model provider name. Required. Use the provider part of the model or "workspace/provider-name" if workspace-scoped.',
                          },
                          inference_parameters: {
                            type: 'object',
                            properties: {
                              generation_type: { type: 'string', enum: ['chat-completion'] },
                              max_tokens: { type: 'number' },
                              temperature: { type: 'number' },
                              top_p: { type: 'number' },
                              max_parallel_requests: { type: 'number' },
                              extra_body: {
                                type: 'object',
                                description:
                                  'Provider-specific JSON merged into the chat request (e.g. flags to disable thinking mode); keys depend on the inference backend.',
                                additionalProperties: true,
                              },
                            },
                          },
                        },
                        additionalProperties: true,
                      },
                    },
                    seed_config: {
                      type: 'object',
                      description:
                        'Optional. source (e.g. seed_type, path), sampling_strategy, selection_strategy.',
                      properties: {
                        source: { type: 'object' },
                        sampling_strategy: { type: 'string', enum: ['ordered', 'shuffle'] },
                      },
                      additionalProperties: true,
                    },
                    constraints: { type: 'array', items: { type: 'object' } },
                    processors: { type: 'array', items: { type: 'object' } },
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
      required: ['job_request'],
    },
  },
};
