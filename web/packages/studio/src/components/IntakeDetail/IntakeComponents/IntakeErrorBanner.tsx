// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner } from '@nvidia/foundations-react-core';
import { Warning } from '@nvidia/react-gui-icons/icons';
import type { FC, ReactNode } from 'react';

/**
 * Error banner for intake telemetry — used for both span and trace errors. The
 * heading is the error identity (e.g. an error type or "Error") and the optional
 * message is the detail. Presentational only; callers decide when to render it.
 */
export const IntakeErrorBanner: FC<{ heading: ReactNode; message?: ReactNode }> = ({
  heading,
  message,
}) => (
  <Banner
    status="error"
    kind="header"
    slotIcon={
      <Warning
        role="img"
        aria-hidden
        className="text-[color:var(--border-color-feedback-danger-subtle)]"
      />
    }
    slotSubheading={message}
    attributes={{
      BannerSubheading: { className: 'min-w-0 whitespace-pre-wrap break-words' },
    }}
  >
    {heading}
  </Banner>
);
