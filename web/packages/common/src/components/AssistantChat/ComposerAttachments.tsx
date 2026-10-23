// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Attachment, AttachmentPrimitive, ComposerPrimitive } from '@assistant-ui/react';
import { Flex, Tooltip } from '@nvidia/foundations-react-core';
import { IconImage, Close } from '@nvidia/react-gui-icons/icons';
import { useEffect, useState } from 'react';

const useImagePreviewUrl = (file?: File): string | undefined => {
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    if (!file) {
      setUrl(undefined);
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => {
      URL.revokeObjectURL(objectUrl);
      setUrl(undefined);
    };
  }, [file]);
  return url;
};

const ComposerAttachmentChip = ({ attachment }: { attachment: Attachment }) => {
  const liftedImage = attachment.content?.find((part) => part.type === 'image')?.image;
  const previewFromFile = useImagePreviewUrl(attachment.file);
  const previewUrl = previewFromFile ?? liftedImage;
  return (
    <Tooltip slotContent={attachment.name}>
      <AttachmentPrimitive.Root className="group/attachment relative size-14 shrink-0 overflow-hidden rounded-lg border border-base bg-surface-sunken">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={attachment.name}
            className="size-full object-cover"
            draggable={false}
          />
        ) : (
          <div className="flex size-full items-center justify-center text-fg-subtle">
            <IconImage size="18" />
          </div>
        )}
        <AttachmentPrimitive.Remove asChild>
          <button
            type="button"
            aria-label={`Remove ${attachment.name}`}
            className="absolute right-0.5 top-0.5 flex size-5 cursor-pointer items-center justify-center rounded-full bg-[rgb(0_0_0/0.55)] text-white opacity-0 outline-none transition-opacity hover:bg-[rgb(0_0_0/0.75)] focus-visible:opacity-100 group-hover/attachment:opacity-100 [@media(hover:none)]:opacity-100"
          >
            <Close size="12" />
          </button>
        </AttachmentPrimitive.Remove>
      </AttachmentPrimitive.Root>
    </Tooltip>
  );
};

/**
 * Renders the current composer's attachments as a wrapping row of thumbnail
 * chips. Collapses to nothing (`empty:hidden`) when there are no attachments.
 * Reads from the surrounding `ComposerPrimitive.Root`, so it works for both the
 * main prompt composer and the inline edit composer.
 */
export const ComposerAttachmentsRow = () => (
  <Flex
    gap="density-xs"
    wrap="wrap"
    padding="density-xs"
    className="empty:hidden"
    data-testid="assistant-chat-composer-attachments"
  >
    <ComposerPrimitive.Attachments>
      {({ attachment }) => <ComposerAttachmentChip attachment={attachment} />}
    </ComposerPrimitive.Attachments>
  </Flex>
);
