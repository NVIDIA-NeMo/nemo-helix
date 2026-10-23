// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { truncateText } from '@nemo/common/src/utils/formatters';
import { OpenExternal } from '@nemo/icons/icons';
import { FC } from 'react';
import { Link } from 'react-router';

export interface DatasetFileLinkProps {
  /** Display label for the link */
  label: string;
  /** URL to navigate to */
  url: string;
  /** Whether to truncate the label text (default: true) */
  truncate?: boolean;
}

/**
 * Link component for dataset files.
 * Renders a clickable link with optional truncated text and an external link icon.
 */
export const DatasetFileLink: FC<DatasetFileLinkProps> = ({ label, url, truncate = true }) => {
  const displayLabel = truncate ? truncateText(label, 70) : label;

  return (
    <Link
      to={url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-2 text-body-semibold-sm"
    >
      {displayLabel}
      <OpenExternal />
    </Link>
  );
};
