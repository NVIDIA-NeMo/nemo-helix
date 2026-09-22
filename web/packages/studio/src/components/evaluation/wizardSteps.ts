// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MODE_DEFAULT, MODE_EXPERIMENT } from '@studio/components/evaluation/submitEvaluationJob';

/** One screen of the Run Evaluation wizard. */
export type WizardStep = 'start' | 'experiment' | 'evaluation';

export type EvaluationMode = typeof MODE_DEFAULT | typeof MODE_EXPERIMENT;

/**
 * The screens a run passes through, decided by the answer to the first one.
 *
 * Creating an experiment and re-running an existing evaluation ask for different things, and a
 * single page that shows both sets — most of them irrelevant — is what made the old form hard to
 * read. Splitting the choice out front means each later step only ever shows fields that apply.
 *
 * Re-running has no experiment to set up, so it skips that step: which run to base this one on
 * and what to call the new run are one decision, made on one screen.
 */
export const stepsFor = (mode: EvaluationMode): WizardStep[] =>
  mode === MODE_DEFAULT ? ['start', 'experiment', 'evaluation'] : ['start', 'evaluation'];

/** Label shown under each step's dot in the stepper. */
export const stepHeading = (step: WizardStep): string => {
  switch (step) {
    case 'start':
      return 'Begin';
    case 'experiment':
      return 'Create experiment';
    case 'evaluation':
      return 'Run Evaluation';
  }
};

export const stepIndex = (steps: WizardStep[], step: WizardStep): number => steps.indexOf(step);

export const nextStep = (steps: WizardStep[], step: WizardStep): WizardStep =>
  steps[Math.min(stepIndex(steps, step) + 1, steps.length - 1)];

export const previousStep = (steps: WizardStep[], step: WizardStep): WizardStep =>
  steps[Math.max(stepIndex(steps, step) - 1, 0)];

export const isLastStep = (steps: WizardStep[], step: WizardStep): boolean =>
  stepIndex(steps, step) === steps.length - 1;
