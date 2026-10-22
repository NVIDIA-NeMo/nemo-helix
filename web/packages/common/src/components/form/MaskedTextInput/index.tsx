// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, TextInput } from '@nvidia/foundations-react-core';
import { Eye, EyeOff } from '@nvidia/react-gui-icons/icons';
import { ComponentProps, forwardRef, useState } from 'react';

type TextInputProps = ComponentProps<typeof TextInput>;

interface Props extends Omit<TextInputProps, 'type' | 'slotEnd'> {
  /** When provided, controls visibility externally. Omit for uncontrolled (defaults to hidden). */
  visible?: boolean;
  onVisibilityChange?: (visible: boolean) => void;
  /** aria-label for the visibility toggle button. Defaults to "Show value"/"Hide value". */
  toggleAriaLabel?: { show: string; hide: string };
}

const DEFAULT_TOGGLE_LABELS = { show: 'Show value', hide: 'Hide value' };

export const MaskedTextInput = forwardRef<HTMLInputElement, Props>(
  (
    {
      visible: visibleProp,
      onVisibilityChange,
      toggleAriaLabel = DEFAULT_TOGGLE_LABELS,
      disabled,
      ...props
    },
    ref
  ) => {
    const [internalVisible, setInternalVisible] = useState(false);
    const isControlled = visibleProp !== undefined;
    const visible = isControlled ? visibleProp : internalVisible;

    const toggle = () => {
      const next = !visible;
      if (!isControlled) setInternalVisible(next);
      onVisibilityChange?.(next);
    };

    return (
      <TextInput
        ref={ref}
        type={visible ? 'text' : 'password'}
        disabled={disabled}
        slotEnd={
          <Button
            type="button"
            kind="tertiary"
            size="tiny"
            onClick={toggle}
            disabled={disabled}
            aria-label={visible ? toggleAriaLabel.hide : toggleAriaLabel.show}
            aria-pressed={visible}
          >
            {visible ? <EyeOff aria-hidden /> : <Eye aria-hidden />}
          </Button>
        }
        {...props}
      />
    );
  }
);

MaskedTextInput.displayName = 'MaskedTextInput';
