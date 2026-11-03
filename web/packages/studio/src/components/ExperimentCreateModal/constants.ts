/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import {
  EXPERIMENT_SETTINGS_DEFAULTS,
  experimentSettingsSchemaShape,
} from '@studio/components/evaluation/shared/experimentSettings';
import { z } from 'zod';

// Everything below the name is the shared experiment-settings shape, so this form, the edit form,
// and the evaluation form's "new experiment" path cannot drift apart.
export const experimentCreateSchema = z.object({
  name: z
    .string()
    .superRefine((value, ctx) => {
      if (sanitizeEntityName(value) === undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: value
            ? 'Name must start with a letter and be at least 2 characters.'
            : 'Name is required.',
        });
      }
    })
    .transform((value) => toValidEntityName(value, value)),
  ...experimentSettingsSchemaShape,
});

export type ExperimentCreateFormFields = z.infer<typeof experimentCreateSchema>;

export const experimentCreateDefaults: ExperimentCreateFormFields = {
  name: '',
  ...EXPERIMENT_SETTINGS_DEFAULTS,
};
