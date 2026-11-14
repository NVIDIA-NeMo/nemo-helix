// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
} from '@nvidia/foundations-react-core';
import {
  NO_SECRETS_HELP,
  NO_SECRETS_PLACEHOLDER,
  SECRETS_LOAD_ERROR_MESSAGE,
} from '@studio/components/CreateDatasetPanel/constants';
import type { WorkspaceSecrets } from '@studio/components/CreateDatasetPanel/useWorkspaceSecrets';
import { useState } from 'react';

interface SecretSelectProps {
  label: string;
  value: string;
  /** A required selector has no "None" choice: once picked, it can only be swapped (spec 9). */
  required: boolean;
  error?: string;
  secrets: WorkspaceSecrets;
  onChange: (value: string) => void;
}

/**
 * A selector over the workspace's secret names. Disabled, with an explanation, when the workspace
 * has none or the lookup failed: a failed lookup says nothing about whether secrets exist.
 */
export function SecretSelect({
  label,
  value,
  required,
  error,
  secrets,
  onChange,
}: SecretSelectProps) {
  const [open, setOpen] = useState(false);
  const loadFailed = secrets.isError;
  const noSecrets = !secrets.isLoading && !loadFailed && secrets.names.length === 0;
  const disabled = noSecrets || loadFailed;
  // Keep a selection visible even if the list no longer contains it.
  const names = !value || secrets.names.includes(value) ? secrets.names : [...secrets.names, value];

  let help: string | undefined;
  if (loadFailed) help = SECRETS_LOAD_ERROR_MESSAGE;
  else if (noSecrets) help = NO_SECRETS_HELP;

  const placeholder = required ? 'Select a secret' : 'Select a secret (optional)';
  const status = error ? 'error' : undefined;
  return (
    <FormField slotLabel={label} slotHelp={help} slotError={error} status={status}>
      <SelectRoot
        value={value}
        onValueChange={onChange}
        disabled={disabled}
        open={open}
        onOpenChange={setOpen}
      >
        <SelectTrigger
          placeholder={secrets.isLoading ? 'Loading...' : placeholder}
          // Shown instead of the placeholder, which the select also copies into a native fallback
          // option. A failed lookup explains itself in the help text only, so it appears once.
          renderValue={() => (noSecrets && !value ? NO_SECRETS_PLACEHOLDER : undefined)}
          status={status}
        />
        <SelectContent>
          {/* Items mount only while open: a closed select's options would otherwise sit in the page
              next to every other secret selector's, duplicating each name. */}
          <SelectListbox>
            {open && !required && <SelectItem value="">None</SelectItem>}
            {open &&
              names.map((name) => (
                <SelectItem key={name} value={name}>
                  {name}
                </SelectItem>
              ))}
          </SelectListbox>
        </SelectContent>
      </SelectRoot>
    </FormField>
  );
}
