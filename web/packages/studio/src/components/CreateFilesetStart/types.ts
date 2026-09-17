// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import type { AddColumnSelection } from '@studio/components/AddColumnPalette/types';
import type {
  StartOption as SharedStartOption,
  StartOptionTag,
} from '@studio/components/StartOptions/types';
import type { GeneratedConfigValidation } from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';
import type { LucideIcon } from 'lucide-react';

export type StartOptionId = 'ai' | 'template' | 'clone' | 'scratch';

export type { StartOptionTag };

export type StartOption = SharedStartOption<StartOptionId>;

export interface TemplateColumnSpec extends AddColumnSelection {
  /** The column name (Jinja2 identifier); referenced by later columns via `{{ name }}`. */
  name: string;
  /** Field values keyed by `ColumnField.key`. Omit for columns with no seeded fields. */
  values?: Record<string, string>;
}

/** Picking one preloads the build canvas with its columns and any models they reference. */

export interface TemplateModelSpec {
  /** Alias the template's columns reference via `model_alias`. */
  alias: string;
  /** Preferred model URN (e.g. `nvidia/llama-3.3-nemotron-super-49b-v1.5`); optional. */
  model?: string;
  /** Optional inference parameter defaults. */
  inferenceParams?: Partial<InferenceParams>;
}

/**
 * A ready-made recipe shown as a card in the secondary area when the "Start from a
 * template" option is selected. Picking one preloads the build canvas with its columns
 * and any models they reference.
 */
export interface FilesetTemplate {
  /** Stable id passed to {@link CreateFilesetStartProps.onContinue} when chosen. */
  id: string;
  title: string;
  description: string;
  icon: LucideIcon;
  tag: StartOptionTag;
  columns: TemplateColumnSpec[];
  /** Models preloaded into the job config, referenced by the columns' `model_alias`. */
  models?: TemplateModelSpec[];
}

export interface TemplateCardProps {
  template: FilesetTemplate;
  selected: boolean;
  onSelect: () => void;
}

export interface DetailPoint {
  icon: LucideIcon;
  title: string;
  description: string;
}

export interface DescribeWithAiPanelProps {
  /** Workspace whose models are offered, and against which the draft is validated. */
  workspace: string;
  /**
   * Fired after every generation: the normalized job request when the draft can be loaded
   * into the build route, null when it can't (so a failed retry clears an earlier success).
   */
  onValidConfig: (jobRequest: DataDesignerJobRequest | null) => void;
}

export interface GeneratedConfigResultProps {
  /** Verdict on the last draft; null before the first generation. */
  validation: GeneratedConfigValidation | null;
  /** Set when the request failed outright (network, auth, model error). */
  requestError: string | null;
  /** Pretty-printed model output for the last draft; enables "View config" when present. */
  rawOutput: string | null;
  isGenerating: boolean;
  /** True while the model is reworking the draft, so the result area says so. */
  isFixing: boolean;
  /** Asks the model to resolve the listed issues. Omit to hide the fix affordance. */
  onFix?: () => void;
}

export interface GeneratedConfigPanelProps {
  open: boolean;
  /** Pretty-printed JSON shown in the snippet. */
  config: string;
  onClose: () => void;
}

export interface StartOptionDetailProps {
  option: StartOption;
  /** Id of the currently-chosen template, when {@link option} is "template". */
  selectedTemplateId: string | null;
  onSelectTemplate: (templateId: string) => void;
  /** Workspace passed through to the "ai" option's panel. */
  workspace: string;
  onValidConfig: (jobRequest: DataDesignerJobRequest | null) => void;
}

/** What the user confirmed via the Continue footer, carrying that option's payload. */
export type StartSelection =
  | { optionId: 'scratch' }
  | { optionId: 'template'; templateId: string }
  | { optionId: 'ai'; jobRequest: DataDesignerJobRequest };

export interface CreateFilesetStartProps {
  /** Workspace whose models the "Describe with AI" option draws from. */
  workspace: string;
  /** Fired when the user confirms a selected start option via the Continue footer. */
  onContinue: (selection: StartSelection) => void;
}
