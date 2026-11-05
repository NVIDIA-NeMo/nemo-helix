// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrorMessage, type ErrorMessageProps } from '@nemo/common/src/components/ErrorMessage';
import { Button } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface ErrorMessageWithRetryProps extends Omit<ErrorMessageProps, 'renderAction'> {
  onRetry: () => void;
}

/**
 * A simple wrapper for ErrorMessage that shows a retry button
 */
export const ErrorMessageWithRetry: FC<ErrorMessageWithRetryProps> = ({
  onRetry,
  ...errorMessageProps
}) => {
  return (
    <ErrorMessage
      slotFooter={
        <Button size="large" onClick={onRetry}>
          Retry
        </Button>
      }
      {...errorMessageProps}
    />
  );
};
