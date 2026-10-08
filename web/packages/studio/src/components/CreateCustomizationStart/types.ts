// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DraftSummary } from '@studio/components/CreateCustomizationStart/aiDraft';
import type { StartOption as SharedStartOption } from '@studio/components/StartOptions/types';
import type { CustomizationFormFields } from '@studio/util/forms/customization';

export type StartOptionId = 'ai' | 'template' | 'scratch';

export type StartOption = SharedStartOption<StartOptionId>;

/**
 * How the user chose to start. Every arm but "scratch" resolves to concrete form values,
 * so the route never has to know how they were produced.
 */
export type StartSelection =
  | { optionId: 'scratch' }
  | { optionId: 'template' | 'ai'; initialValues: CustomizationFormFields };

export interface CreateCustomizationStartProps {
  /** Workspace the template option provisions into and the AI option drafts against. */
  workspace: string;
  /** Fired once a way in is chosen and, for templates and AI, its form values are ready. */
  onContinue: (selection: StartSelection) => void;
}

export interface DescribeWithAiPanelProps {
  /** Workspace whose models and datasets the user picks from. */
  workspace: string;
  /** Fired after every run: form values when the draft loads, null when it doesn't. */
  onDraft: (values: CustomizationFormFields | null) => void;
}

export interface DraftResultProps {
  summary: DraftSummary;
  /** The full request the form would submit, pretty-printed, for "View config". */
  config: string;
  onEdit: () => void;
}
