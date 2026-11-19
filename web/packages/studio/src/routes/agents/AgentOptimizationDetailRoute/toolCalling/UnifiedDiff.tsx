// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type FC, memo } from 'react';

type DiffLineKind = 'added' | 'removed' | 'hunk' | 'header' | 'context';

const LINE_CLASS_NAME: Record<DiffLineKind, string> = {
  added: 'text-feedback-success',
  removed: 'text-feedback-danger',
  hunk: 'text-feedback-info',
  header: 'text-secondary font-bold',
  context: 'text-primary',
};

const diffLineKind = (line: string): DiffLineKind => {
  if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('diff ')) {
    return 'header';
  }
  if (line.startsWith('@@')) {
    return 'hunk';
  }
  if (line.startsWith('+')) {
    return 'added';
  }
  if (line.startsWith('-')) {
    return 'removed';
  }
  return 'context';
};

/** A unified diff with its added and removed lines coloured. */
export const UnifiedDiff: FC<{ diff: string; label: string }> = memo(({ diff, label }) => (
  <pre
    aria-label={label}
    className="max-h-[480px] overflow-auto rounded border border-base bg-surface-sunken p-3 font-mono text-sm"
  >
    {diff.split('\n').map((line, index) => {
      const kind = diffLineKind(line);
      return (
        <span key={index} data-diff-line={kind} className={`block ${LINE_CLASS_NAME[kind]}`}>
          {line || ' '}
        </span>
      );
    })}
  </pre>
));
UnifiedDiff.displayName = 'UnifiedDiff';
