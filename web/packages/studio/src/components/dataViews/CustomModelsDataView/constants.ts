// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatFinetuningType } from '@nemo/common/src/utils/formatters';
import {
  AutomodelTrainingSpecFinetuningType,
  RlGRPOTrainingFinetuningType,
  UnslothTrainingSpecFinetuningType,
} from '@nemo/sdk/generated/customizer/schema';
import type { FinetuningType } from '@nemo/sdk/generated/platform/schema';

/**
 * Every finetuning_type a completed customization can stamp on a model entity:
 * the union of the customizer backends' supported values (automodel, unsloth,
 * rl/GRPO). Sourced from the customizer API schema so it stays in sync with the
 * backend on SDK regen. The broad platform `FinetuningType` enum also carries
 * approaches the product does not support, which must not appear in this filter
 * (ASTD-488).
 */
const SUPPORTED_FINETUNING_TYPES: FinetuningType[] = [
  ...Object.values(AutomodelTrainingSpecFinetuningType),
  ...Object.values(UnslothTrainingSpecFinetuningType),
  ...Object.values(RlGRPOTrainingFinetuningType),
].filter((value, index, all) => all.indexOf(value) === index);

/** Column filter options in FilterItem format ({ value, label }) for single-select filters. */
export const FINETUNING_TYPE_FILTER_OPTIONS = SUPPORTED_FINETUNING_TYPES.map((value) => ({
  value,
  label: formatFinetuningType(value),
}));

export const HAS_BASE_MODEL = { 'data.base_model': { $not: { $eq: null } } };
export const HAS_ADAPTERS = { adapters: { $exists: true } };

/**
 * Default filter: show only "custom" models (has base_model, finetuning_type, or adapters).
 */
export const DEFAULT_CUSTOM_MODELS_FILTER = { $or: [HAS_BASE_MODEL, HAS_ADAPTERS] };
