// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatNumericValue } from '@nemo/common/src/components/charts/format';

export interface TrainingDiagnostic {
  id: string;
  metric: string;
  title: string;
  formatValue: (value: number) => string;
}

/** A learning rate rounds to "0.00" under the shared numeric formatter. */
const formatExponential = (value: number): string => value.toExponential(1);

/**
 * Mirrors `DIAGNOSTIC_TIME_SERIES` and `DPO_TIME_SERIES_METRICS` in the backends, the
 * allow-lists deciding which metrics keep a history. A rename there needs one here.
 */
export const TRAINING_DIAGNOSTICS: TrainingDiagnostic[] = [
  {
    id: 'learningRate',
    metric: 'train_lr',
    title: 'Learning rate',
    formatValue: formatExponential,
  },
  {
    id: 'gradNorm',
    metric: 'train_grad_norm',
    title: 'Gradient norm',
    formatValue: formatNumericValue,
  },
  {
    id: 'trainAccuracy',
    metric: 'train_accuracy',
    title: 'Preference accuracy',
    formatValue: formatNumericValue,
  },
  {
    id: 'valAccuracy',
    metric: 'val_accuracy',
    title: 'Validation accuracy',
    formatValue: formatNumericValue,
  },
  {
    id: 'rewardsChosen',
    metric: 'train_rewards_chosen_mean',
    title: 'Reward, chosen response',
    formatValue: formatNumericValue,
  },
  {
    id: 'rewardsRejected',
    metric: 'train_rewards_rejected_mean',
    title: 'Reward, rejected response',
    formatValue: formatNumericValue,
  },
];
