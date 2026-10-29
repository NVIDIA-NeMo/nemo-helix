// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Modal, Stack, Text } from '@nvidia/foundations-react-core';
import { CodingAgentPromptEditor } from '@studio/components/CodingAgentPromptEditor';
import { type FC } from 'react';

interface CodingAgentPromptModalProps {
  open: boolean;
  onClose: () => void;
  heading: string;
  description: string;
  /** The prompt to display and copy. */
  prompt: string;
}

/** Shows the coding agent prompt for a "Get started" option, with a copy button. */
export const CodingAgentPromptModal: FC<CodingAgentPromptModalProps> = ({
  open,
  onClose,
  heading,
  description,
  prompt,
}) => (
  <Modal
    open={open}
    onOpenChange={onClose}
    className="w-[720px]"
    slotHeading={
      <Stack gap="1">
        <Text kind="title/sm">{heading}</Text>
        <Text kind="body/regular/sm" className="text-secondary">
          {description}
        </Text>
      </Stack>
    }
    slotFooter={
      <Button color="brand" kind="primary" onClick={onClose}>
        Close
      </Button>
    }
  >
    <CodingAgentPromptEditor prompt={prompt} />
  </Modal>
);
