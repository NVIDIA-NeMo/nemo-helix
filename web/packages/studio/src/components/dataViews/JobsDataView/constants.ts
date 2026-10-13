// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const JOB_SOURCE = {
  CUSTOMIZATION: 'customization',
  DATA_DESIGNER: 'data-designer',
  SAFE_SYNTHESIZER: 'safe-synthesizer',
  EVALUATOR_METRICS: 'evaluator-metrics',
  MODELS_SYSTEM: 'models-system',
} as const;

export const SOURCE_OPTIONS = [
  { label: 'All', value: '' },
  { label: 'Fine-tuning', value: JOB_SOURCE.CUSTOMIZATION },
  { label: 'Data Designer', value: JOB_SOURCE.DATA_DESIGNER },
  { label: 'Safe Synthesizer', value: JOB_SOURCE.SAFE_SYNTHESIZER },
  { label: 'Evaluator', value: JOB_SOURCE.EVALUATOR_METRICS },
];

export const HIDDEN_JOB_SOURCES: readonly string[] = [JOB_SOURCE.MODELS_SYSTEM];
