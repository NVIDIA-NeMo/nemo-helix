// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@nvidia/foundations-react-core';
import { BRIEF_TOAST_DISMISS_MS } from '@studio/prototypes/notifications/toasts/constants';
import type { BriefToastOptions } from '@studio/prototypes/notifications/toasts/types';
import { useAutoDismiss } from '@studio/prototypes/notifications/toasts/useAutoDismiss';
import cn from 'classnames';
import type { FC } from 'react';

interface BriefToastCardProps extends BriefToastOptions {
  onDismiss: () => void;
  className?: string;
}

export const BriefToastCard: FC<BriefToastCardProps> = ({
  status,
  message,
  durationMs = BRIEF_TOAST_DISMISS_MS,
  onDismiss,
  className,
}) => {
  const pauseHandlers = useAutoDismiss(durationMs, onDismiss);

  return (
    <Toast
      status={status}
      role="status"
      onClose={onDismiss}
      className={cn('shadow-lg', className)}
      data-testid="proto-brief-toast"
      {...pauseHandlers}
    >
      {message}
    </Toast>
  );
};
