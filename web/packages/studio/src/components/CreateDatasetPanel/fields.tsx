// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Form controls shared by the Create Dataset tabs. Controlled by props only: state lives in the
// panel's reducer, so these hold nothing but the markup and the paste flag.

import {
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  TextInput,
} from '@nvidia/foundations-react-core';
import {
  NAME_HELP,
  NO_SECRETS_HELP,
  NO_SECRETS_PLACEHOLDER,
} from '@studio/components/CreateDatasetPanel/constants';
import type { PanelAction } from '@studio/components/CreateDatasetPanel/draft';
import type { WorkspaceSecrets } from '@studio/components/CreateDatasetPanel/useWorkspaceSecrets';
import { type Dispatch, type ReactNode, useState } from 'react';

interface TextFieldProps {
  label: string;
  value: string;
  error?: ReactNode;
  help?: string;
  placeholder?: string;
  onChange: (value: string) => void;
  onBlur?: () => void;
  onPaste?: () => void;
}

export function TextField({
  label,
  value,
  error,
  help,
  placeholder,
  onChange,
  onBlur,
  onPaste,
}: TextFieldProps) {
  return (
    <FormField
      slotLabel={label}
      slotHelp={help}
      slotError={error}
      status={error ? 'error' : undefined}
    >
      <TextInput
        value={value}
        placeholder={placeholder}
        onValueChange={onChange}
        onBlur={onBlur}
        onPaste={onPaste}
      />
    </FormField>
  );
}

interface NameFieldProps {
  value: string;
  error?: string;
  dispatch: Dispatch<PanelAction>;
}

export function NameField({ value, error, dispatch }: NameFieldProps) {
  return (
    <TextField
      label="Name"
      value={value}
      error={error}
      help={NAME_HELP}
      onChange={(next) => dispatch({ type: 'name-changed', value: next })}
      onBlur={() => dispatch({ type: 'name-blurred' })}
    />
  );
}

interface DescriptionFieldProps {
  value: string;
  dispatch: Dispatch<PanelAction>;
}

export function DescriptionField({ value, dispatch }: DescriptionFieldProps) {
  return (
    <TextField
      label="Description"
      value={value}
      onChange={(next) => dispatch({ type: 'description-changed', value: next })}
    />
  );
}

interface SecretSelectProps {
  label: string;
  value: string;
  /** A required selector has no "None" choice: once picked, it can only be swapped (spec 9). */
  required: boolean;
  error?: string;
  secrets: WorkspaceSecrets;
  onChange: (value: string) => void;
}

/** A selector over the workspace's secret names. Disabled, with an explanation, when there are none. */
export function SecretSelect({
  label,
  value,
  required,
  error,
  secrets,
  onChange,
}: SecretSelectProps) {
  const [open, setOpen] = useState(false);
  const noSecrets = !secrets.isLoading && secrets.names.length === 0;
  // Keep a selection visible even if the list no longer contains it.
  const names = !value || secrets.names.includes(value) ? secrets.names : [...secrets.names, value];

  const placeholder = required ? 'Select a secret' : 'Select a secret (optional)';
  const status = error ? 'error' : undefined;
  return (
    <FormField
      slotLabel={label}
      slotHelp={noSecrets ? NO_SECRETS_HELP : undefined}
      slotError={error}
      status={status}
    >
      <SelectRoot
        value={value}
        onValueChange={onChange}
        disabled={noSecrets}
        open={open}
        onOpenChange={setOpen}
      >
        <SelectTrigger
          placeholder={secrets.isLoading ? 'Loading...' : placeholder}
          // Shown instead of the placeholder, which the select also copies into a native fallback option.
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
