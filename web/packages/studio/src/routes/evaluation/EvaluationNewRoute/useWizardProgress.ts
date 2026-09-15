// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type StepStatus } from '@nemo/common/src/components/StepSection';
import {
  type EvaluationFormValues,
  selectedMetrics,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useTemplateBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useTemplateBindings';
import { useFormContext, useWatch } from 'react-hook-form';

export interface WizardProgress {
  model: StepStatus;
  dataset: StepStatus;
  metrics: StepStatus;
}

/** ``complete`` wins over ``active``: a satisfied step reads as done even while
 *  the user is still editing it. */
const statusOf = (unlocked: boolean, satisfied: boolean): StepStatus =>
  satisfied ? 'complete' : unlocked ? 'active' : 'upcoming';

/**
 * Which steps are done and which are open for input.
 *
 * Each step gates on the one before it, so the form reads as a sequence rather
 * than a wall. "Complete" is deliberately narrower than the submit-time
 * validation: a step only answers for its own fields, so step 2 never refuses to
 * unlock because of something wrong in step 3.
 */
export function useWizardProgress(): WizardProgress {
  const { control } = useFormContext<EvaluationFormValues>();
  const values = useWatch({ control }) as EvaluationFormValues;
  const bindings = useTemplateBindings();

  const model = Boolean(values?.model);

  // The dataset step needs a file plus BOTH bindings resolved -- either mapped
  // columns or an auto-detected messages array. Input is what gets sent to the
  // model; Ground Truth is what six of the seven metrics score against, so
  // unlocking metric selection without it would offer choices that cannot run.
  const dataset = Boolean(values?.dataset && bindings.inputPath && bindings.referencePath);

  const metricTypes = values ? selectedMetrics(values) : [];
  const judgeReady =
    !metricTypes.includes('llm-judge') ||
    Boolean(values?.body?.judgeModel && values?.body?.scores?.length);
  const metrics = metricTypes.length > 0 && judgeReady;

  // Dry Run is deliberately absent: it is optional, always available, and
  // validates on click rather than being gated.
  return {
    model: statusOf(true, model),
    dataset: statusOf(model, dataset),
    metrics: statusOf(model && dataset, metrics),
  };
}
