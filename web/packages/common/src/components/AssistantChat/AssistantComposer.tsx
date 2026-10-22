// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ComposerPrimitive, ThreadPrimitive } from '@assistant-ui/react';
import { ComposerAttachmentsRow } from '@nemo/common/src/components/AssistantChat/ComposerAttachments';
import type { AssistantChatThreadProps } from '@nemo/common/src/components/AssistantChat/types';
import { Button, Tooltip } from '@nvidia/foundations-react-core';
import { ArrowUp, IconImage, Retry, ShapeSquare } from '@nvidia/react-gui-icons/icons';
import cn from 'classnames';

type AssistantComposerProps = Pick<
  AssistantChatThreadProps,
  | 'disabled'
  | 'placeholder'
  | 'onReset'
  | 'slotComposerStart'
  | 'enableImageAttachments'
  | 'minInputRows'
> & {
  className?: string;
};

export const AssistantComposer = ({
  disabled,
  placeholder,
  onReset,
  slotComposerStart,
  className,
  enableImageAttachments = true,
  minInputRows,
}: AssistantComposerProps) => (
  <div className="flex w-full flex-col gap-2">
    {slotComposerStart && <div className="shrink-0">{slotComposerStart}</div>}
    <ComposerPrimitive.Root
      data-testid="assistant-chat-composer"
      className={cn(
        'flex w-full flex-col gap-density-xs rounded-lg border border-base bg-surface-base p-1',
        className
      )}
    >
      {enableImageAttachments && <ComposerAttachmentsRow />}
      <div className="flex w-full items-end gap-density-xs">
        <ComposerPrimitive.Input
          aria-label="Task prompt"
          addAttachmentOnPaste={enableImageAttachments}
          disabled={disabled}
          placeholder={placeholder}
          submitMode="enter"
          minRows={minInputRows}
          className="max-h-32 min-h-[24px] flex-1 resize-none border-0 bg-transparent px-density-md py-density-md text-sm leading-6 outline-none disabled:cursor-not-allowed disabled:text-fg-disabled"
        />
        {enableImageAttachments && (
          <Tooltip slotContent="Add image">
            <ComposerPrimitive.AddAttachment asChild>
              <Button
                aria-label="Add image"
                kind="tertiary"
                size="small"
                type="button"
                disabled={disabled}
              >
                <IconImage />
              </Button>
            </ComposerPrimitive.AddAttachment>
          </Tooltip>
        )}
        <Tooltip slotContent="Clear chat thread">
          <Button
            aria-label="Reset"
            kind="tertiary"
            size="small"
            onClick={onReset}
            type="button"
            disabled={disabled}
          >
            <Retry />
          </Button>
        </Tooltip>
        <ThreadPrimitive.If running>
          <ComposerPrimitive.Cancel asChild>
            <Button
              aria-label="Stop"
              color="danger"
              size="small"
              className="size-8 rounded-full p-0"
            >
              <ShapeSquare size="14" />
            </Button>
          </ComposerPrimitive.Cancel>
        </ThreadPrimitive.If>
        <ThreadPrimitive.If running={false}>
          <ComposerPrimitive.Send asChild>
            <Button
              aria-label="Submit"
              color="brand"
              size="small"
              className="size-8 rounded-full p-0"
            >
              <ArrowUp size="16" />
            </Button>
          </ComposerPrimitive.Send>
        </ThreadPrimitive.If>
      </div>
    </ComposerPrimitive.Root>
  </div>
);
