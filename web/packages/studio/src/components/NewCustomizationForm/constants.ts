// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AutomodelTrainingSpecPrecision,
  AutomodelTrainingSpecRecipe,
  AutomodelTrainingSpecTeacherPrecision,
  OptimizerType,
  UnslothHardwareSpecPrecision,
  UnslothOptimizerSpecOptim,
  UnslothScheduleSpecLrSchedulerType,
  UnslothTrainingSpecUseGradientCheckpointing,
} from '@nemo/sdk/generated/customizer/schema';
import { selectItems } from '@studio/util/forms/selectItems';

export const OPTIMIZER_TYPE_ITEMS = selectItems(OptimizerType);
export const TEACHER_PRECISION_ITEMS = selectItems(AutomodelTrainingSpecTeacherPrecision);
export const AUTOMODEL_PRECISION_ITEMS = selectItems(AutomodelTrainingSpecPrecision);
export const AUTOMODEL_RECIPE_ITEMS = selectItems(AutomodelTrainingSpecRecipe);
export const UNSLOTH_LR_SCHEDULER_ITEMS = selectItems(UnslothScheduleSpecLrSchedulerType);
export const UNSLOTH_OPTIM_ITEMS = selectItems(UnslothOptimizerSpecOptim);
export const UNSLOTH_PRECISION_ITEMS = selectItems(UnslothHardwareSpecPrecision);

/** `true`/`false` sit alongside `unsloth` here, and read as a bug rendered literally. */
export const UNSLOTH_GRADIENT_CHECKPOINTING_ITEMS = selectItems(
  UnslothTrainingSpecUseGradientCheckpointing,
  { true: 'Enabled', false: 'Disabled' }
);
