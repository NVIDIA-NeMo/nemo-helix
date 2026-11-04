// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ActionBarPrimitive, useAuiState } from '@assistant-ui/react';
import { GENERIC_CHAT_ERROR } from '@nemo/common/src/components/AssistantChat/describeChatError';
import { CopyErrorButton } from '@nemo/common/src/components/AssistantChat/messageActions';
import { useMessageError } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { Retry } from '@nemo/icons/icons';
import {
  AnimatedChevron,
  Banner,
  Button,
  Collapsible,
  Flex,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';

// A failed reply's text parts hold the human-readable explanation (see `getFailedReply`), so they
// render here as the banner message instead of in the body. The raw error (status and provider
// detail) sits behind "Show details" for anyone who needs to look closer.
export const ChatErrorBanner = () => {
  const error = useMessageError();
  const explanation = useAuiState((state) =>
    state.message.parts
      .flatMap((part) => (part.type === 'text' && part.text.trim() ? [part.text.trim()] : []))
      .join('\n\n')
  );

  return (
    <Banner
      kind="inline"
      status="error"
      className="w-full"
      data-testid="assistant-chat-error-banner"
      slotActions={
        <Flex gap="density-sm" align="center">
          <ActionBarPrimitive.Reload asChild>
            <Button kind="secondary" size="small">
              <Retry size="12" />
              Retry
            </Button>
          </ActionBarPrimitive.Reload>
          {error ? <CopyErrorButton error={error} /> : null}
        </Flex>
      }
    >
      <Stack gap="density-xs" align="start">
        <span className="whitespace-pre-wrap break-words">{explanation || GENERIC_CHAT_ERROR}</span>
        <Collapsible
          slotTrigger={
            <Button asChild kind="tertiary" size="tiny" className="-ml-1">
              <div>
                Show details
                <AnimatedChevron />
              </div>
            </Button>
          }
        >
          <Text kind="mono/sm" className="whitespace-pre-wrap break-all pt-density-xs">
            {error}
          </Text>
        </Collapsible>
      </Stack>
    </Banner>
  );
};
