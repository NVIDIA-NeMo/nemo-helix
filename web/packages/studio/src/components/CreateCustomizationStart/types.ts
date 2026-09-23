// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StartOption as SharedStartOption } from '@studio/components/StartOptions/types';
import type { CustomizationFormFields } from '@studio/util/forms/customization';

export type StartOptionId = 'scratch';

export type StartOption = SharedStartOption<StartOptionId>;

/**
 * What the user confirmed via the Continue footer. Every arm but "scratch" resolves to
 * concrete form values, so the route never has to know how they were produced.
 */
export type StartSelection =
  | { optionId: 'scratch' }
  | { optionId: 'template'; initialValues: CustomizationFormFields };

export interface CreateCustomizationStartProps {
  /** Workspace the template option registers its models and datasets into. */
  workspace: string;
  /** Fired when the user confirms a selected start option via the Continue footer. */
  onContinue: (selection: StartSelection) => void;
}
