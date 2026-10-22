// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CODE_BLOCK_SURFACE_CLASS } from '@nemo/common/src/components/AssistantChat/toolCall/constants';
import { summarizeRepeatedSubtleToolActions } from '@nemo/common/src/components/AssistantChat/toolCall/helpers';
import type { SubtleToolAction } from '@nemo/common/src/components/AssistantChat/toolCall/types';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { ChevronRight, Copy } from 'lucide-react';

interface SubtleToolCallRowProps {
  readonly actions: readonly SubtleToolAction[];
}

interface InvocationPanelProps {
  readonly invocation: string;
  readonly invocationCount: number;
  readonly invocationIndex: number;
  readonly toolCallId: string;
}

const copyInvocation = (invocation: string): void => {
  if (!navigator.clipboard) return;
  void navigator.clipboard.writeText(invocation).catch(() => undefined);
};

const InvocationPanel = ({
  invocation,
  invocationCount,
  invocationIndex,
  toolCallId,
}: InvocationPanelProps) => (
  <Stack
    className="w-full min-w-0 max-w-full overflow-hidden"
    data-testid="assistant-tool-call-invocation"
    id={`${toolCallId}-invocation-${invocationIndex}`}
  >
    <Stack className="relative w-full min-w-0 max-w-full">
      <button
        aria-label={`Copy invocation${invocationCount === 1 ? '' : ` ${invocationIndex + 1}`}`}
        className="absolute right-density-xs top-density-xs z-10 rounded p-0.5 hover:bg-surface-sunken"
        onClick={() => copyInvocation(invocation)}
        title="Copy invocation"
        type="button"
      >
        <Copy aria-hidden className="size-3.5" />
      </button>
      <pre
        className={`max-h-72 w-full min-w-0 max-w-full overflow-auto whitespace-pre-wrap break-words rounded ${CODE_BLOCK_SURFACE_CLASS} p-density-sm pr-density-xl text-xs leading-relaxed text-secondary`}
        data-testid="assistant-tool-call-invocation-surface"
      >
        <code data-testid="assistant-tool-call-invocation-content">{invocation}</code>
      </pre>
    </Stack>
  </Stack>
);

export const SubtleToolCallRow = ({ actions }: SubtleToolCallRowProps) => (
  <Text asChild kind="body/regular/sm">
    <div
      className="my-density-xs flex w-full max-w-full flex-wrap items-center gap-x-density-sm gap-y-density-xs overflow-hidden rounded border border-base border-l-2 border-l-[var(--border-color-accent-blue)] bg-[color-mix(in_srgb,var(--background-color-accent-blue-subtle)_38%,var(--background-color-surface-base))] px-density-sm py-density-xs text-secondary"
      data-testid="assistant-tool-call-subtle"
      title={actions.map((action) => action.title ?? action.message).join(' | ')}
    >
      {summarizeRepeatedSubtleToolActions(actions).map((action, index) => {
        const Icon = action.Icon;
        const key = `${action.toolCallId}-${index}`;

        const invocations = action.invocations ?? [action.invocation];
        const isExpandable =
          Boolean(action.details?.length) || (action.toolName !== 'Read' && invocations.length > 0);

        if (isExpandable) {
          return (
            <details
              key={key}
              className="group/subtle w-full min-w-0 max-w-full basis-full overflow-hidden"
              data-testid="assistant-tool-call-subtle-details"
            >
              <summary
                className="inline-flex w-full min-w-0 max-w-full basis-full cursor-pointer list-none items-center gap-density-xs overflow-hidden marker:hidden"
                data-testid="assistant-tool-call-subtle-action"
              >
                <ChevronRight
                  aria-hidden
                  className="size-3 shrink-0 transition-transform group-open/subtle:rotate-90"
                />
                <Icon
                  aria-hidden
                  className="size-3.5 shrink-0"
                  data-testid="assistant-tool-call-subtle-icon"
                />
                <span className="min-w-0 truncate">{action.message}</span>
              </summary>
              {action.details?.length ? (
                <Stack
                  className="mt-0.5 w-full min-w-0 max-w-full space-y-0.5 overflow-hidden pl-7"
                  data-testid="assistant-tool-call-subtle-detail-list"
                >
                  {action.toolName === 'Read'
                    ? action.details.map((detail, detailIndex) => (
                        <span
                          key={`${action.toolCallId}-${detailIndex}`}
                          className="block min-w-0 max-w-full truncate py-0.5"
                          data-testid="assistant-tool-call-subtle-detail-item"
                          title={detail}
                        >
                          {detail}
                        </span>
                      ))
                    : action.details.map((detail, detailIndex) => {
                        const invocation = invocations[detailIndex];
                        if (!invocation) return null;

                        return (
                          <details
                            key={`${action.toolCallId}-${detailIndex}`}
                            className="group/invocation w-full min-w-0 max-w-full overflow-hidden"
                            data-testid="assistant-tool-call-nested-invocation"
                          >
                            <summary
                              className="flex w-full min-w-0 max-w-full cursor-pointer list-none items-center gap-density-xs overflow-hidden py-0.5 marker:hidden"
                              data-testid="assistant-tool-call-subtle-detail-item"
                              title={detail}
                            >
                              <span className="min-w-0 truncate">{detail}</span>
                              <ChevronRight
                                aria-hidden
                                className="size-3 shrink-0 transition-transform group-open/invocation:rotate-90"
                              />
                            </summary>

                            <InvocationPanel
                              invocation={invocation}
                              invocationCount={invocations.length}
                              invocationIndex={detailIndex}
                              toolCallId={action.toolCallId}
                            />
                          </details>
                        );
                      })}
                </Stack>
              ) : (
                <InvocationPanel
                  invocation={invocations[0]!}
                  invocationCount={1}
                  invocationIndex={0}
                  toolCallId={action.toolCallId}
                />
              )}
            </details>
          );
        }

        return (
          <span
            key={key}
            className="inline-flex w-full min-w-0 max-w-full basis-full items-center gap-density-xs overflow-hidden"
            data-testid="assistant-tool-call-subtle-action"
          >
            <Icon
              aria-hidden
              className="size-3.5 shrink-0"
              data-testid="assistant-tool-call-subtle-icon"
            />
            <span className="min-w-0 truncate">{action.message}</span>
          </span>
        );
      })}
    </div>
  </Text>
);
