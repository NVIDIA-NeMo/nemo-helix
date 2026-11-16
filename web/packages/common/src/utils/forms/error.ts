// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import type { FieldErrors } from 'react-hook-form';

/**
 * Use as a sane default for form errors.
 * Handles parsing the errors and logging them to the website logger.
 * @param errors - The errors to handle. Any shape
 */
type HandleFormErrorGenericProps = {
  title?: string;
};
export const handleFormErrorsGeneric =
  ({ title }: HandleFormErrorGenericProps) =>
  (errors: FieldErrors) => {
    const errorMessages: string[] = [];
    for (const [key, value] of Object.entries(errors)) {
      // Stringify errors but exclude 'ref' property which contains React fiber tree
      const errorInfo = JSON.stringify(value, (key, val) => {
        // Skip the 'ref' property to avoid dumping React internals
        if (key === 'ref') return undefined;
        return val;
      });
      errorMessages.push(`${key}: ${errorInfo}`);
    }
    logger.error(`${title || 'Form Errors'}: \n${errorMessages.join('\n')}`);
  };
