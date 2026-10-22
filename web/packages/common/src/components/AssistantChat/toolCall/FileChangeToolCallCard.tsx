// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CODE_BLOCK_SURFACE_CLASS,
  FILE_CHANGE_ADDITION_CLASS,
  FILE_CHANGE_DELETION_CLASS,
} from '@nemo/common/src/components/AssistantChat/toolCall/constants';
import { Text } from '@nvidia/foundations-react-core';
import { ChevronRight, Rename, DocumentNew } from '@nvidia/react-gui-icons/icons';

interface FileChangeToolCallCardProps {
  readonly summary: {
    readonly action: 'Edited' | 'Wrote';
    readonly additions: number;
    readonly deletions: number;
    readonly path: string;
    readonly reviewContent: string;
  };
}

export const FileChangeToolCallCard = ({ summary }: FileChangeToolCallCardProps) => {
  const Icon = summary.action === 'Wrote' ? DocumentNew : Rename;

  return (
    <div
      className="my-density-xs w-full max-w-full overflow-hidden rounded border border-base bg-surface-raised"
      data-testid="assistant-tool-call-file-change"
    >
      <details className="group/write" data-testid="assistant-tool-call-file-change-details">
        <summary className="flex cursor-pointer list-none items-center gap-density-sm px-density-sm py-density-xs marker:hidden">
          <div className="flex size-8 shrink-0 items-center justify-center rounded bg-surface-sunken text-secondary">
            <Icon size="16" />
          </div>
          <div className="min-w-0 flex-1">
            <Text kind="label/bold/md" className="block">
              {summary.action} 1 file
            </Text>
            <Text kind="body/regular/sm" className="block tabular-nums">
              <span className={FILE_CHANGE_ADDITION_CLASS}>+{summary.additions}</span>{' '}
              <span className={FILE_CHANGE_DELETION_CLASS}>-{summary.deletions}</span>
            </Text>
          </div>
          <span className="flex shrink-0 items-center gap-density-xs rounded border border-base px-density-sm py-density-xs text-secondary group-open/write:bg-surface-sunken">
            <Text kind="label/regular/sm">Review</Text>
            <ChevronRight size="14" className="transition-transform group-open/write:rotate-90" />
          </span>
        </summary>
        <div className="border-t border-base px-density-sm py-density-xs">
          <pre
            className={`max-h-72 overflow-auto rounded ${CODE_BLOCK_SURFACE_CLASS} p-density-sm text-xs leading-relaxed text-secondary`}
            data-testid="assistant-tool-call-file-change-review-surface"
          >
            <code data-testid="assistant-tool-call-file-change-review">
              {summary.reviewContent}
            </code>
          </pre>
        </div>
      </details>
      <div className="border-t border-base px-density-sm py-density-xs">
        <div className="flex min-w-0 items-center justify-between gap-density-md">
          <Text kind="body/regular/sm" className="min-w-0 truncate">
            {summary.path}
          </Text>
          <Text kind="body/regular/sm" className="shrink-0 tabular-nums">
            <span className={FILE_CHANGE_ADDITION_CLASS}>+{summary.additions}</span>{' '}
            <span className={FILE_CHANGE_DELETION_CLASS}>-{summary.deletions}</span>
          </Text>
        </div>
      </div>
    </div>
  );
};
