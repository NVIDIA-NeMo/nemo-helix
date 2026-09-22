// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type EvaluationFormValues } from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useDatasetBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetBindings';
import { useFormContext, useWatch } from 'react-hook-form';

/**
 * Whether the dataset is bound well enough for metrics to be worth choosing.
 *
 * Narrower than the submit-time validation on purpose: it answers for the
 * dataset's own fields only, so metrics never stay hidden because of something
 * wrong further down.
 *
 * The model is deliberately absent: it lives on the following step, so nothing
 * here waits on it.
 */
export function useDatasetReady(): boolean {
  const { control } = useFormContext<EvaluationFormValues>();
  const values = useWatch({ control }) as EvaluationFormValues;
  const bindings = useDatasetBindings();

  // A file plus BOTH bindings resolved -- either mapped columns or an
  // auto-detected messages array. Input is what gets sent to the model; Ground
  // Truth is what six of the seven metrics score against, so offering metric
  // selection without it would list choices that cannot run.
  return Boolean(values?.dataset && bindings.inputPath && bindings.referencePath);
}
