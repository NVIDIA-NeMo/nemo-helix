// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { generateDefaultName } from '@nemo/common/src/utils/generateDefaultName';
import type { GenerateJobRequest } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { SafeSynthesizerCreateJobBody } from '@nemo/sdk/generated/safe-synthesizer/zod/safe-synthesizer';
import {
  DEFAULT_PRETRAINED_MODEL,
  MAX_NUM_RECORDS,
  PRETRAINED_MODEL_OPTIONS,
} from '@studio/routes/SafeSynthesizerNewRoute/constants';
import { z } from 'zod';

const generatedSpecShape = SafeSynthesizerCreateJobBody.shape.spec.shape;
const generatedPiiLlmShape = generatedSpecShape.config.shape.replace_pii
  .unwrap()
  .shape.llm.unwrap().shape;

export const PII_MODEL_PROVIDER_REQUIRED = 'Model provider is required for PII replacement';
export const PII_MODEL_ID_REQUIRED = 'Model ID is required for PII replacement';

const jobRequestFieldsSchema = z.object({
  name: z.string().optional(),
  description: z.string().optional(),
  namespace: z.string().optional(),
  spec: z.object({
    data_source: z
      .string()
      .min(1, 'Data source is required')
      .refine((val) => {
        const hashIdx = val.indexOf('#');
        if (hashIdx === -1 || hashIdx !== val.lastIndexOf('#')) return false;
        const filesetPart = val.slice(0, hashIdx);
        const filePath = val.slice(hashIdx + 1);
        const slashIdx = filesetPart.indexOf('/');
        if (slashIdx === -1) return false;
        const workspace = filesetPart.slice(0, slashIdx);
        const name = filesetPart.slice(slashIdx + 1);
        return (
          workspace.length > 0 && name.length > 0 && !name.includes('/') && filePath.length > 0
        );
      }, 'Data source must be a valid fileset reference (workspace/name#path)'),
    pii_model_provider: generatedSpecShape.pii_model_provider,
    config: z.object({
      data: z
        .object({
          group_training_examples_by: z.string().optional(),
          order_training_examples_by: z.string().optional(),
        })
        .optional()
        .refine(
          (data) => {
            if (!data) return true;
            const hasOrderBy =
              data.order_training_examples_by && data.order_training_examples_by.trim() !== '';
            if (hasOrderBy) {
              return (
                data.group_training_examples_by && data.group_training_examples_by.trim() !== ''
              );
            }
            return true;
          },
          {
            message:
              'group_training_examples_by is required when order_training_examples_by is set',
            path: ['group_training_examples_by'],
          }
        ),
      enable_synthesis: z.boolean().optional(),
      enable_replace_pii: z.boolean().optional(),
      replace_pii: z
        .object({
          llm: z.object({ model_id: generatedPiiLlmShape.model_id }).optional(),
        })
        .optional(),
      training: z
        .object({
          num_input_records_to_sample: z
            .any()
            .refine(
              (val) => {
                // If it's 'auto', it's valid
                if (val === 'auto') return true;
                // If it's a blank string or whitespace, it's invalid
                if (val === '' || (typeof val === 'string' && val.trim() === '')) return false;
                // If it's a number, validate it
                return typeof val === 'number';
              },
              {
                message: 'Number of input records is required',
              }
            )
            .refine(
              (val) => {
                // If it's 'auto', skip number validation
                if (val === 'auto') return true;
                // If it's a number, validate it
                if (typeof val === 'number') {
                  return Number.isInteger(val) && val > 0;
                }
                return true;
              },
              {
                message: 'Must be a positive whole number',
              }
            )
            .refine(
              (val) => {
                // If it's 'auto', skip max validation
                if (val === 'auto') return true;
                // If it's a number, validate it's not too large
                if (typeof val === 'number') {
                  return val <= Number.MAX_SAFE_INTEGER;
                }
                return true;
              },
              {
                message: `Must be less than or equal to ${Number.MAX_SAFE_INTEGER}`,
              }
            ),
          rope_scaling_factor: z.union([z.literal('auto'), z.number().int().min(1).max(6)]),
          pretrained_model: z.enum(PRETRAINED_MODEL_OPTIONS).optional(),
        })
        .optional(),
      generation: z
        .object({
          num_records: z
            .any()
            .refine((val) => {
              // Handle blank string case
              return !(val === '' || (typeof val === 'string' && val.trim() === ''));
            }, 'Number of records is required')
            .refine((val) => {
              // Convert to number and validate
              const num = Number(val);
              return !isNaN(num);
            }, 'Must be a number')
            .transform((val) => Number(val))
            .refine((val) => Number.isInteger(val), 'Must be a whole number')
            .refine((val) => val > 0, 'Must be greater than 0')
            .refine((val) => val <= MAX_NUM_RECORDS, 'Must be less than or equal to 130,000'),
          temperature: z.number().min(0).max(2).optional(),
          top_p: z.number().min(0).max(1).optional(),
          repetition_penalty: z.number().min(1).optional(),
        })
        .optional(),
      privacy: z
        .object({
          dp_enabled: z.boolean().optional(),
        })
        .optional(),
    }),
  }),
});

export const safeSynthesizerJobRequestSchema = jobRequestFieldsSchema.superRefine(
  ({ spec }, ctx) => {
    if (!spec.config.enable_replace_pii) {
      return;
    }
    if (!spec.pii_model_provider) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: PII_MODEL_PROVIDER_REQUIRED,
        path: ['spec', 'pii_model_provider'],
      });
    }
    if (!spec.config.replace_pii?.llm?.model_id?.trim()) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: PII_MODEL_ID_REQUIRED,
        path: ['spec', 'config', 'replace_pii', 'llm', 'model_id'],
      });
    }
  }
) satisfies z.ZodType<GenerateJobRequest>;

export type SafeSynthesizerFormData = z.infer<typeof safeSynthesizerJobRequestSchema>;

export const getSafeSynthesizerFormDefaults = (): SafeSynthesizerFormData => ({
  name: generateDefaultName(),
  description: '',
  spec: {
    data_source: '',
    pii_model_provider: '',
    config: {
      enable_synthesis: true,
      enable_replace_pii: true,
      replace_pii: {
        llm: { model_id: '' },
      },
      training: {
        num_input_records_to_sample: 'auto',
        rope_scaling_factor: 'auto',
        pretrained_model: DEFAULT_PRETRAINED_MODEL,
      },
      generation: {
        num_records: 1000,
        temperature: 0.9,
        top_p: 1,
        repetition_penalty: 1,
      },
      data: {},
      privacy: {
        dp_enabled: false,
      },
    },
  },
});

type ExampleOrdering = NonNullable<SafeSynthesizerFormData['spec']['config']['data']>;

const withoutBlankExampleOrdering = (data: ExampleOrdering): ExampleOrdering => {
  const result: ExampleOrdering = {};
  if (data.group_training_examples_by?.trim()) {
    result.group_training_examples_by = data.group_training_examples_by;
  }
  if (data.order_training_examples_by?.trim()) {
    result.order_training_examples_by = data.order_training_examples_by;
  }
  return result;
};

export const toGenerateJobRequest = (formData: SafeSynthesizerFormData): GenerateJobRequest => {
  const { pii_model_provider: piiModelProvider, ...spec } = formData.spec;
  const { replace_pii: replacePii, ...config } = spec.config;
  const request: GenerateJobRequest = { ...formData, spec: { ...spec, config } };
  if (config.data) {
    request.spec.config.data = withoutBlankExampleOrdering(config.data);
  }
  if (config.enable_replace_pii) {
    request.spec.pii_model_provider = piiModelProvider;
    request.spec.config.replace_pii = {
      llm: { model_id: replacePii?.llm?.model_id?.trim() },
    };
  }
  return request;
};
