// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import type { Row } from '@studio/util/files';
import Papa from 'papaparse';

export { getTextWithCount } from '@nemo/common/src/utils/formatters';

export const capitalize = (str: string) => {
  return str.charAt(0).toUpperCase() + str.slice(1);
};

/** Trim a string, returning undefined when it is empty or whitespace-only. */
export const trimToUndefined = (value: string | undefined): string | undefined => {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
};

/**
 * Formats a snake_case key into a human-readable label.
 * @param key - The snake_case key to format (e.g., "prompt_tokens")
 * @returns A title-cased label with spaces (e.g., "Prompt Tokens")
 * @example
 * formatKeyLabel('prompt_tokens')     // "Prompt Tokens"
 * formatKeyLabel('total_tokens')      // "Total Tokens"
 * formatKeyLabel('finish_reason')     // "Finish Reason"
 */
export const formatKeyLabel = (key: string): string => {
  return key.split('_').map(capitalize).join(' ');
};

/**
 * Parses a CSV string and returns an array of objects.
 * @param csvString - The CSV string to parse.
 * @returns An array of objects.
 */
export const parseCSV = (props: { csvString: string; options: Papa.ParseConfig }): Row[] => {
  const { csvString, options } = props;
  const { data, errors } = Papa.parse(csvString, options);

  if (errors.length) {
    logger.error(`CSV Parse Errors: ${JSON.stringify(errors)}`);
    return [];
  }

  return data;
};
