// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  GenericSortField,
  PaginationData,
  ValidationError,
} from '@nemo/sdk/generated/platform/schema';
import { AxiosError } from 'axios';

export const GENERIC_SORT_FIELDS = ['created_at', '-created_at'] as readonly GenericSortField[];

export const isSupportedSortField = <T extends string>(
  unionArray: readonly T[],
  value?: string
): value is T => {
  return unionArray.includes(value as T);
};

export const getNextPageParam = <T extends { pagination?: PaginationData }>(lastPage: T) => {
  if (!lastPage.pagination || lastPage.pagination.total_pages <= lastPage.pagination.page) {
    return;
  }
  return lastPage.pagination.page + 1;
};

/**
 * Type guard to check if a value is a ValidationError array from API responses.
 * This is useful for handling backend validation errors that come in the format:
 * { detail: [{ msg: string, type: string, loc: (string | number)[] }] }
 */
export const isValidationErrorArray = (detail: unknown): detail is ValidationError[] => {
  return (
    Array.isArray(detail) &&
    detail.length > 0 &&
    detail.every(
      (item) =>
        typeof item === 'object' &&
        item !== null &&
        'msg' in item &&
        typeof item.msg === 'string' &&
        'type' in item &&
        'loc' in item
    )
  );
};

/** The entity-store's optimistic-lock rejection: `expected_db_version` no longer matches. */
export const isVersionConflictError = (error: unknown): boolean =>
  error instanceof AxiosError && error.response?.status === 409;

/** A 404 from the platform API. */
export const isNotFoundError = (error: unknown): boolean => {
  const candidate = error as { response?: { status?: number }; status?: number };
  return candidate?.response?.status === 404 || candidate?.status === 404;
};

/**
 * Extracts a user-friendly error message from an error object.
 * Handles both ValidationError arrays and simple string errors from the backend.
 *
 * @param error - The error object (typically from an API call)
 * @param fallbackMessage - Used only when the error carries no message of its own
 * @returns A user-friendly error message string
 *
 * @example
 * ```ts
 * const errorMessage = getErrorMessage(error, 'Failed to save model');
 * toast.error(errorMessage);
 * ```
 */
export const getErrorMessage = (error: AxiosError | Error, fallbackMessage?: string): string => {
  if (error instanceof AxiosError) {
    // Try to extract error detail from backend response
    if (error.response?.data?.detail) {
      const detail = error.response.data.detail;

      // Handle validation errors (array format with ValidationError type)
      if (isValidationErrorArray(detail)) {
        return detail
          .map((err) => {
            const field = err.loc.filter((s) => s !== 'body').join('.');
            return field ? `${field}: ${err.msg}` : err.msg;
          })
          .join('; ');
      }

      // Handle simple string errors
      if (typeof detail === 'string') {
        return detail;
      }
    }

    // Handle network errors (no response received)
    if (!error.response) {
      const parts: string[] = [];

      // Add the error code for more context
      if (error.code) {
        parts.push(`[${error.code}]`);
      }

      parts.push(error.message);

      // Add request context if available
      if (error.config) {
        const method = error.config.method?.toUpperCase();
        const url = error.config.url;
        if (method && url) {
          parts.push(`(${method} ${url})`);
        }
      }

      // Add cause if available
      if (error.cause instanceof Error) {
        parts.push(`- ${error.cause.message}`);
      }

      return parts.join(' ');
    }

    // Handle HTTP error responses without detail
    if (error.response) {
      const status = error.response.status;
      const statusText = error.response.statusText;
      return `${status} ${statusText}`;
    }
  }

  // The error's own message first: callers pass `fallbackMessage` for errors that carry
  // nothing useful (an AxiosError whose branches above all missed), not to overwrite a
  // message that was written to be read. Preferring the fallback turned a precise
  // "config has no usable model" into "Unknown error" at the guardrail-run call site.
  return error.message || fallbackMessage || '';
};

export const swallowConflict = <T>(promise: Promise<T>): Promise<T | undefined> =>
  promise.catch((e: unknown) => {
    if (e instanceof AxiosError && e.response?.status === 409) return undefined;
    throw e;
  });
