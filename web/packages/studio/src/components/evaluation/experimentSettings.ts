// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExperimentRequest } from '@nemo/sdk/generated/platform/schema';
import { DEFAULT_SORT } from '@studio/components/DefaultSortControl/util';
import type { FieldPath, FieldValues } from 'react-hook-form';
import { z } from 'zod';

/**
 * The settings an Experiment carries beyond its name. Every surface that creates or edits an
 * experiment — the Experiments list, an experiment's own Edit modal, and the evaluation form's
 * "new experiment" path — renders these through one shared component, so a field added here shows
 * up on all three instead of only wherever it was added.
 */
export interface ExperimentSettingsValues {
  description: string;
  defaultSort: string;
  isFavorite: boolean;
  showEvaluationsOverTime: boolean;
}

/** Zod fields for the settings, spread into whatever schema the host form builds. Names match
 *  `ExperimentSettingsValues`, so a host that spreads this can pass `EXPERIMENT_SETTINGS_NAMES`
 *  straight through to the fields component.
 *
 *  Every field defaults, so a form that reuses the schema without rendering these fields — the
 *  compact "create a group and add to it" form, say — still validates on the name alone. */
export const experimentSettingsSchemaShape = {
  description: z.string().default(''),
  defaultSort: z.string().default(DEFAULT_SORT),
  isFavorite: z.boolean().default(false),
  showEvaluationsOverTime: z.boolean().default(false),
};

export const EXPERIMENT_SETTINGS_DEFAULTS: ExperimentSettingsValues = {
  description: '',
  defaultSort: DEFAULT_SORT,
  isFavorite: false,
  showEvaluationsOverTime: false,
};

/** Read the settings back off an existing experiment, for an edit form's initial values. */
export const experimentSettingsFrom = (group: {
  description?: string | null;
  default_sort?: string | null;
  is_favorite?: boolean | null;
  show_evaluations_over_time?: boolean | null;
}): ExperimentSettingsValues => ({
  description: group.description ?? '',
  defaultSort: group.default_sort || DEFAULT_SORT,
  isFavorite: group.is_favorite ?? false,
  showEvaluationsOverTime: group.show_evaluations_over_time ?? false,
});

/** The settings as the create/update endpoints take them. An empty description is omitted
 *  rather than sent as `''`, which the API stores verbatim. */
export const experimentSettingsPayload = (
  values: ExperimentSettingsValues
): Pick<
  ExperimentRequest,
  'description' | 'default_sort' | 'is_favorite' | 'show_evaluations_over_time'
> => ({
  description: values.description || undefined,
  default_sort: values.defaultSort,
  is_favorite: values.isFavorite,
  show_evaluations_over_time: values.showEvaluationsOverTime,
});

/** Where each setting lives in the host form. Hosts name their fields differently — the
 *  evaluation form prefixes them so they cannot collide with the evaluation's own — so the
 *  paths are passed in rather than assumed. */
export interface ExperimentSettingsFieldNames<T extends FieldValues> {
  description: FieldPath<T>;
  defaultSort: FieldPath<T>;
  isFavorite: FieldPath<T>;
  showEvaluationsOverTime: FieldPath<T>;
}

/** The identity mapping, for a host whose fields are named exactly like the canonical settings. */
export const EXPERIMENT_SETTINGS_NAMES = {
  description: 'description',
  defaultSort: 'defaultSort',
  isFavorite: 'isFavorite',
  showEvaluationsOverTime: 'showEvaluationsOverTime',
} as const;
