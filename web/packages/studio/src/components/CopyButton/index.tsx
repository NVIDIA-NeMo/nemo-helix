// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Button } from '@nvidia/foundations-react-core';
import { CopyGeneric } from '@nvidia/react-gui-icons/icons';
import { ComponentProps, FC } from 'react';

interface Props extends ComponentProps<typeof Button> {
  text: string;
}

export const CopyButton: FC<Props> = ({ text, ...buttonProps }) => {
  const { success, error } = useToast();

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(text);
      success('Successfully copied to clipboard!', { durationMs: 3000 });
    } catch {
      error('Error copying text to clipboard', { durationMs: 3000 });
    }
  };

  return (
    <Button onClick={copyText} size="small" color="brand" type="button" {...buttonProps}>
      <CopyGeneric />
    </Button>
  );
};
