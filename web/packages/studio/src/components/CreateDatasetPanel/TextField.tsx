// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FormField, TextInput } from '@nvidia/foundations-react-core';
import type { ReactNode } from 'react';

interface TextFieldProps {
  label: string;
  value: string;
  error?: ReactNode;
  help?: string;
  placeholder?: string;
  onChange: (value: string) => void;
  onBlur?: () => void;
  /** Fires before the change event that carries the pasted text. */
  onPaste?: () => void;
}

/** A labelled single-line input with optional help and error text. Controlled by props only. */
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
