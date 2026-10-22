// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Stack, Text } from '@nvidia/foundations-react-core';
import { ChatSingle } from '@nvidia/react-gui-icons/icons';
import type { FC } from 'react';

interface GetStartedOptionProps {
  heading: string;
  description: string;
  /** Distinguishes the two identically-labelled buttons for screen readers. */
  actionLabel: string;
  /** Opens the coding agent prompt modal for this option. */
  onGetPrompt: () => void;
}

/** One way to connect an agent to the platform, with the prompt that does it. */
export const GetStartedOption: FC<GetStartedOptionProps> = ({
  heading,
  description,
  actionLabel,
  onGetPrompt,
}) => (
  <Stack
    gap="2"
    className="min-w-0 flex-1 basis-80 rounded-xl border border-base bg-surface-raised p-6"
  >
    <Text kind="title/sm" className="text-center">
      {heading}
    </Text>
    <Stack gap="4" align="center" justify="center" className="flex-1 text-center">
      <Text kind="body/regular/md" className="text-secondary">
        {description}
      </Text>
      <Button kind="tertiary" aria-label={actionLabel} onClick={onGetPrompt}>
        <ChatSingle size="16" className="text-brand" aria-hidden />
        Get coding agent prompt
      </Button>
    </Stack>
  </Stack>
);
