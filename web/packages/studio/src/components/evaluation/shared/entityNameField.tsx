// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { z } from 'zod';

/** An entity name as typed, sanitized on the way out. The field keeps the user's literal
 *  keystrokes, so only unsalvageable input is an error — see the entity-naming contract. */
export const entityNameField = () =>
  z.string().transform((value) => toValidEntityName(value, value));

/** The message for a name with nothing salvageable in it, or undefined when there is. */
export const unsalvageableNameError = (value: string, label: string): string | undefined => {
  if (sanitizeEntityName(value) !== undefined) return undefined;
  return value
    ? `${label} must start with a letter and be at least 2 characters.`
    : `${label} is required.`;
};

/** The server's own explanation for a failed submit, falling back to the transport error.
 *  Without the `detail`, a 422 reads only as "Request failed with status code 422". */
export const submitErrorMessage = (error: unknown): string | undefined => {
  if (!error) return undefined;
  const detail = (error as { response?: { data?: { detail?: unknown } } } | undefined)?.response
    ?.data?.detail;
  if (typeof detail === 'string' && detail) return detail;
  // Pydantic validation errors arrive as a list of {loc, msg} objects.
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (item as { msg?: unknown })?.msg)
      .filter((msg): msg is string => typeof msg === 'string' && msg.length > 0);
    if (messages.length) return messages.join('; ');
  }
  return error instanceof Error ? error.message : 'An error occurred';
};
