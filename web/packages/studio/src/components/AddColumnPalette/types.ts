// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DataDesignerConfig, SamplerType } from '@nemo/sdk/generated/data-designer/schema';
import type { TagProps } from '@nvidia/foundations-react-core';
import type { NvidiaGuiIconProps } from '@nvidia/react-gui-icons';
import type { FC } from 'react';

/**
 * The Data Designer `column_type` discriminators, matching the SDK column config
 * schemas (`SamplerColumnConfig`, `LLMTextColumnConfig`, …). Kept as a local union so
 * the palette is self-contained; the literals are identical to the generated schemas.
 */
export type DataDesignerColumnType = DataDesignerConfig['columns'][number]['column_type'];

/** Accent color shared with the design-system `Tag`, reused to tint an option's icon. */
export type ColumnTypeColor = NonNullable<TagProps['color']>;

/** A single pickable column type (or sampler sub-type) shown as a card in the palette. */
export interface ColumnTypeOption {
  /** Stable id, unique across the palette (e.g. `"sampler.uuid"`, `"llm-text"`). */
  id: string;
  /** The Data Designer column type this option creates. */
  columnType: DataDesignerColumnType;
  /** For `sampler` columns, the `sampler_type` sub-type to seed. */
  samplerType?: SamplerType;
  /** Display name. */
  label: string;
  /** One-line description of what the column does. */
  description: string;
  /** Leading icon. */
  icon: FC<Omit<NvidiaGuiIconProps, 'iconName'>>;
  /** Accent color for the icon, by column family. */
  color: ColumnTypeColor;
}

/** A labeled group of related column types (e.g. "Sampler", "Generate"). */
export interface ColumnTypeGroup {
  /** Stable group id. */
  id: string;
  /** Group heading. */
  label: string;
  /** The options in the group, in display order. */
  options: ColumnTypeOption[];
}

/** Describes the column type a user chose to add. */
export interface AddColumnSelection {
  columnType: DataDesignerColumnType;
  samplerType?: SamplerType;
}
