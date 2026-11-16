// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';

/**
 * Schema for individual property in parameters object
 */
const propertySchema = z.object({
  type: z.enum(['string', 'number', 'integer', 'boolean', 'object', 'array']),
  description: z.string().optional(),
  enum: z.array(z.string()).optional(),
  /**
   * For array types - simplified to avoid recursive types
   */
  items: z
    .object({
      type: z.enum(['string', 'number', 'integer', 'boolean']),
      description: z.string().optional(),
      enum: z.array(z.string()).optional(),
    })
    .optional(),
  /**
   * For object types - simplified to avoid recursive types
   */
  properties: z
    .record(
      z.object({
        type: z.enum(['string', 'number', 'integer', 'boolean']),
        description: z.string().optional(),
        enum: z.array(z.string()).optional(),
      })
    )
    .optional(),
  required: z.array(z.string()).optional(),
  minimum: z.number().optional(),
  maximum: z.number().optional(),
  minLength: z.number().optional(),
  maxLength: z.number().optional(),
});

/**
 * Schema for parameters object
 */
export const parametersSchema = z.object({
  type: z.literal('object'),
  properties: z.record(propertySchema),
  required: z.array(z.string()).optional(),
  additionalProperties: z.boolean().optional(),
});

/**
 * Schema for individual tool/function
 * Validates OpenAI function calling format with strict rules for name and description
 */
export const toolSchema = z.object({
  type: z.literal('function'),
  function: z.object({
    name: z.string().regex(/^[a-zA-Z0-9_]{1,64}$/, {
      message:
        'Function name must be 1-64 characters long and contain only letters, numbers, or underscores.',
    }),
    description: z.string().max(200).optional(),
    strict: z.boolean().nullable().optional(),
    // We have a more detailed property schema, whereas openai uses a record
    parameters: z.union([parametersSchema, z.record(z.unknown())]).optional(),
  }),
});

export type ChatCompletionToolsParam = z.infer<typeof toolSchema>;
