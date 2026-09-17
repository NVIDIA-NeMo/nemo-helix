// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { UseControllerComponentProps } from '@nemo/common/src/types';
import { FormField, TextArea } from '@nvidia/foundations-react-core';
import { useEffect, useRef, useState } from 'react';
import { useController } from 'react-hook-form';

interface Props extends UseControllerComponentProps {
  label?: string;
  placeholder?: string;
  disabled?: boolean;
}

export const ControlledJsonInput = ({
  useControllerProps,
  formFieldProps,
  label,
  placeholder,
  disabled,
}: Props) => {
  const {
    field: { value, onChange, onBlur, disabled: fieldDisabled },
    fieldState,
  } = useController(useControllerProps);

  const [text, setText] = useState<string>(() =>
    value == null ? '' : JSON.stringify(value, null, 2)
  );
  const [parseError, setParseError] = useState<string>();

  const textRef = useRef(text);
  textRef.current = text;
  useEffect(() => {
    const trimmed = textRef.current.trim();
    if (!trimmed) {
      if (value != null) setText(JSON.stringify(value, null, 2));
      return;
    }
    let current: unknown;
    try {
      current = JSON.parse(trimmed);
    } catch {
      return; // invalid mid-edit text — don't clobber it
    }
    if (JSON.stringify(current) !== JSON.stringify(value)) {
      setText(value == null ? '' : JSON.stringify(value, null, 2));
      setParseError(undefined);
    }
  }, [value]);

  const handleChange = (next: string) => {
    setText(next);
    const trimmed = next.trim();
    if (!trimmed) {
      setParseError(undefined);
      onChange(undefined);
      return;
    }
    try {
      onChange(JSON.parse(trimmed));
      setParseError(undefined);
    } catch {
      setParseError('Invalid JSON');
    }
  };

  // Unparseable text is the more immediate problem, so it wins; otherwise show whatever
  // the resolver rejected — valid JSON of the wrong shape has no parse error to report.
  const error = parseError ?? fieldState.error?.message;

  return (
    <FormField
      slotLabel={label}
      slotError={error ?? ''}
      status={error ? 'error' : undefined}
      {...formFieldProps}
    >
      <TextArea
        value={text}
        onValueChange={handleChange}
        onBlur={onBlur}
        disabled={disabled || fieldDisabled}
        placeholder={placeholder}
      />
    </FormField>
  );
};
