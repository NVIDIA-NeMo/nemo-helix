// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AutomodelTrainingSpecPrecision } from '@nemo/sdk/generated/customizer/schema/AutomodelTrainingSpecPrecision';
import { AutomodelTrainingSpecRecipe } from '@nemo/sdk/generated/customizer/schema/AutomodelTrainingSpecRecipe';
import { AutomodelTrainingSpecTeacherPrecision } from '@nemo/sdk/generated/customizer/schema/AutomodelTrainingSpecTeacherPrecision';
import { OptimizerType } from '@nemo/sdk/generated/customizer/schema/OptimizerType';
import { UnslothHardwareSpecPrecision } from '@nemo/sdk/generated/customizer/schema/UnslothHardwareSpecPrecision';
import { UnslothOptimizerSpecOptim } from '@nemo/sdk/generated/customizer/schema/UnslothOptimizerSpecOptim';
import { UnslothScheduleSpecLrSchedulerType } from '@nemo/sdk/generated/customizer/schema/UnslothScheduleSpecLrSchedulerType';
import { UnslothTrainingSpecUseGradientCheckpointing } from '@nemo/sdk/generated/customizer/schema/UnslothTrainingSpecUseGradientCheckpointing';
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
