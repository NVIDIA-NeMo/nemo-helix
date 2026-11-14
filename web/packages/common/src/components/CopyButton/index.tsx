// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCopyToClipboard } from '@nemo/common/src/hooks/useCopyToClipboard';
import type { NotifyFn } from '@nemo/common/src/providers/toast/types';
import { useNotify } from '@nemo/common/src/providers/toast/useNotify';
import { CopyGeneric } from '@nemo/icons/icons';
import { Button } from '@nvidia/foundations-react-core';
import type { ComponentProps } from 'react';

const COPY_TOAST_DURATION_MS = 3000;

interface CopyButtonProps extends ComponentProps<typeof Button> {
  readonly text: string;
  readonly onNotify?: NotifyFn;
}

export const CopyButton = ({ text, onNotify, children, ...buttonProps }: CopyButtonProps) => {
  const notify = useNotify(onNotify);
  const { copyToClipboard } = useCopyToClipboard({
    onSuccess: () =>
      notify('Successfully copied to clipboard!', 'success', {
        durationMs: COPY_TOAST_DURATION_MS,
      }),
    onError: () =>
      notify('Error copying text to clipboard', 'error', { durationMs: COPY_TOAST_DURATION_MS }),
  });

  return (
    <Button
      onClick={() => void copyToClipboard(text)}
      size="small"
      color="brand"
      type="button"
      {...buttonProps}
    >
      {children ?? <CopyGeneric />}
    </Button>
  );
};
