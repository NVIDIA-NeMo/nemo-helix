// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ComposerPrimitive, MessagePrimitive } from '@assistant-ui/react';
import { ComposerAttachmentsRow } from '@nemo/common/src/components/AssistantChat/ComposerAttachments';
import { ImportImage, Close } from '@nemo/icons/icons';
import { Button, Flex, Text, TextArea, Tooltip } from '@nvidia/foundations-react-core';

export const UserEditComposer = ({
  enableImageAttachments = true,
}: {
  enableImageAttachments?: boolean;
}) => (
  <MessagePrimitive.Root
    data-testid="assistant-chat-edit-composer"
    className="ml-20 min-w-0 self-stretch rounded-xl rounded-br-none bg-surface-sunken px-4 py-3 dark:bg-surface-raised"
  >
    <ComposerPrimitive.Root className="flex w-full flex-col gap-density-sm">
      {enableImageAttachments && <ComposerAttachmentsRow />}
      <ComposerPrimitive.Input
        aria-label="Edit message"
        addAttachmentOnPaste={enableImageAttachments}
        autoFocus
        submitMode="enter"
        rows={3}
        render={<TextArea resizeable="auto" size="large" className="w-full max-h-64" />}
      />
      <Flex gap="density-sm" align="center" justify="end">
        {enableImageAttachments && (
          <Tooltip slotContent="Add image">
            <ComposerPrimitive.AddAttachment asChild>
              <Button
                aria-label="Add image"
                kind="tertiary"
                size="small"
                type="button"
                className="mr-auto"
              >
                <ImportImage size="16" />
              </Button>
            </ComposerPrimitive.AddAttachment>
          </Tooltip>
        )}
        <Tooltip slotContent="Cancel edit">
          <ComposerPrimitive.Cancel
            aria-label="Cancel edit"
            className="cursor-pointer flex size-8 items-center justify-center rounded border border-base bg-surface-raised hover:bg-surface-sunken disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Close />
          </ComposerPrimitive.Cancel>
        </Tooltip>
        <ComposerPrimitive.Send asChild>
          <Button aria-label="Save edit" color="brand" size="small">
            <Text kind="label/regular/sm">Send</Text>
          </Button>
        </ComposerPrimitive.Send>
      </Flex>
    </ComposerPrimitive.Root>
  </MessagePrimitive.Root>
);
