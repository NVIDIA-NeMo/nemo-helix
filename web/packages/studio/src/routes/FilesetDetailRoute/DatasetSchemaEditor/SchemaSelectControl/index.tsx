// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
} from '@nvidia/foundations-react-core';
import type { FC } from 'react';

/** Stable value used to represent an INLINE root `schema` in the dropdown. */
export const DEFAULT_SCHEMA_VALUE = 'default';

/** Stable value used to switch the editor to the full `metadata.dataset` view. */
export const SHOW_ALL_VALUE = 'Show All';

export interface SchemaSelectControlProps {
  /** schema_defs keys to list. */
  defKeys: string[];
  /** True when the dataset's root `schema` is an inline object (not a ref).
   *  In that case, a separate "Default" entry is shown for it. */
  hasInlineDefault: boolean;
  /** When the dataset's root `schema` is a string ref to a `schema_defs` key,
   *  the key value goes here. The matching dropdown entry is labelled
   *  "<key> (default)" instead of appearing twice. */
  defaultDefKey?: string;
  /** Currently-selected value: `DEFAULT_SCHEMA_VALUE`, a key from defKeys,
   *  or `SHOW_ALL_VALUE` for the advanced full-metadata view. */
  value: string;
  onChange: (value: string) => void;
  /** When true, the trigger is non-interactive. Used in file scope where the
   *  selection is forced to the file's mapped schema. */
  disabled?: boolean;
}

/**
 * Dropdown selector for the Dataset Schema editor's single-schema view.
 *
 * Items, in order:
 *   - "Default" — only when the root `schema` is inline (otherwise the root
 *     IS one of the schema_defs entries below, marked "(default)").
 *   - Each entry in `schema_defs`. If a key equals `defaultDefKey`, it is
 *     suffixed with " (default)".
 *   - "Show All" — switches the editor to the full `metadata.dataset` payload.
 */
export const SchemaSelectControl: FC<SchemaSelectControlProps> = ({
  defKeys,
  hasInlineDefault,
  defaultDefKey,
  value,
  onChange,
  disabled,
}) => (
  <SelectRoot value={value} onValueChange={onChange} disabled={disabled}>
    <SelectTrigger
      aria-label="Selected schema"
      placeholder="Select a schema"
      renderValue={(v) => {
        if (typeof v !== 'string') return null;
        if (v === defaultDefKey) return `${v} (default)`;
        return v;
      }}
    />
    <SelectContent>
      <SelectListbox>
        {hasInlineDefault && <SelectItem value={DEFAULT_SCHEMA_VALUE}>Default</SelectItem>}
        {defKeys.map((key) => (
          <SelectItem key={key} value={key}>
            {key === defaultDefKey ? `${key} (default)` : key}
          </SelectItem>
        ))}
        <SelectItem value={SHOW_ALL_VALUE}>Show All</SelectItem>
      </SelectListbox>
    </SelectContent>
  </SelectRoot>
);
