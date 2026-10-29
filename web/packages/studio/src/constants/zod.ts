// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DATASET_NAME_REGEX } from '@studio/constants/constants';
import { z } from 'zod';

export const namespaceSchema = z
  .string()
  .default('default')
  .describe(
    "The namespace of the entity. This can be missing for namespace entities or in deployments that don't use namespaces."
  );

export const descriptionSchema = z.string().describe('The description of the entity.');

export const projectNameInputSchema = z
  .string()
  .min(1, { message: 'Workspace name is required' })
  .max(255)
  .regex(
    /^[\w.-]*$/,
    'Name must only contain alphanumeric characters, dashes, underscores, or dots.'
  )
  .regex(/[a-zA-Z0-9]/, 'Name must contain at least one alphanumeric character.');

export const URN_REGEX = /^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$/;
export const URNSchema = z
  .string()
  .min(1, { message: 'URN is required' })
  .regex(URN_REGEX, 'URN must be in the format of <namespace>/<name>');

export const customFieldsSchema = z
  .record(z.string(), z.any())
  .optional()
  .describe('A set of custom fields that the user can define and use for various purposes.');

export const ownershipSchema = z
  .object({
    created_by: z.string().optional().describe('The ID of the user that created this entity.'),
    updated_by: z.string().optional().describe('The ID of the user that last updated this entity.'),
    access_policies: z
      .record(z.string(), z.string())
      .default({})
      .describe(
        'A general object for capturing access policies which can be used by an external service to determine ACLs'
      ),
  })
  .optional()
  .describe(
    'Information about ownership of an entity.\n\nIf the entity is a namespace, the `access_policies` will typically apply to all entities inside the namespace.'
  )
  .describe('Ownership information for the entity');

export const datasetSchema = z.object({
  name: z
    .string()
    .min(1, { message: 'Name is required' })
    .regex(
      DATASET_NAME_REGEX,
      'Name must only contain alphanumeric characters, dashes, underscores, or dots'
    ),
  /** @deprecated Namespace is now provided via workspace at the API level */
  namespace: z
    .string()
    .regex(
      DATASET_NAME_REGEX,
      'Namespace must only contain alphanumeric characters, dashes, underscores, or dots'
    )
    .optional(),
  description: z.string(),
});

export const projectCreateSchema = z.object({
  name: projectNameInputSchema,
  description: z.string().optional(),
});

export const projectUpdateSchema = z.object({
  description: z.string().optional(),
});

export const workspaceUpdateSchema = z.object({
  description: z.string().optional(),
});
