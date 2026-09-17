// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SelectItemOption } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import type { DataDesignerModelOption } from '@studio/components/NewDataDesignerJobForm/utils';
import {
  DETECTOR_GROUP_OTHER,
  DETECTOR_GROUP_SUGGESTED,
  NER_DETECTOR_PATTERNS,
  STRATEGY_REWRITE,
  TAB_MODEL_SETTINGS,
  TAB_SOURCE,
  type BuilderTab,
  type Strategy,
} from '@studio/routes/AnonymizerBuilderRoute/constants';

export const OUTPUT_HEADING_REPLACED = 'Replaced';
export const OUTPUT_HEADING_REWRITTEN = 'Rewritten';

export const isNerDetectorModel = (model: DataDesignerModelOption): boolean =>
  NER_DETECTOR_PATTERNS.some(
    (pattern) => pattern.test(model.name) || pattern.test(model.served_model_name ?? '')
  );

/**
 * The detector takes any NER model the library supports, and nothing on a ModelEntity says what a
 * model is for. So every model stays selectable and the NER families we know of are suggested.
 */
export const detectorOptions = (
  items: readonly SelectItemOption[],
  models: readonly DataDesignerModelOption[]
): SelectItemOption[] => {
  const suggestedIds = new Set(models.filter(isNerDetectorModel).map((model) => model.id));
  const group = (item: SelectItemOption) =>
    suggestedIds.has(item.value) ? DETECTOR_GROUP_SUGGESTED : DETECTOR_GROUP_OTHER;
  return [...items]
    .sort(
      (a, b) =>
        Number(group(a) === DETECTOR_GROUP_OTHER) - Number(group(b) === DETECTOR_GROUP_OTHER)
    )
    .map((item) => ({ ...item, group: group(item) }));
};

/** Only `roleModels` lives on Model Settings; an empty list must not read as "models only". */
export const tabForValidationErrors = (fields: readonly string[]): BuilderTab =>
  fields.length > 0 && fields.every((field) => field === 'roleModels')
    ? TAB_MODEL_SETTINGS
    : TAB_SOURCE;

/** The output column only exists once results land, so the skeleton reads the strategy instead. */
export const outputHeadingForStrategy = (strategy: Strategy): string =>
  strategy === STRATEGY_REWRITE ? OUTPUT_HEADING_REWRITTEN : OUTPUT_HEADING_REPLACED;
