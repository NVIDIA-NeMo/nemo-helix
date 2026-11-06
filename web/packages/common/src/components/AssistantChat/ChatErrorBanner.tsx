// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAuiState } from '@assistant-ui/react';
import { GENERIC_CHAT_ERROR } from '@nemo/common/src/components/AssistantChat/describeChatError';
import { useMessageError } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { WorkCodeSnippet } from '@nemo/common/src/components/AssistantChat/work/WorkCodeSnippet';
import { WorkAccordion } from '@nemo/common/src/components/AssistantChat/work/WorkStep';
import { Banner, Stack } from '@nvidia/foundations-react-core';

// A failed reply's text parts hold the human-readable explanation (see `getFailedReply`), so they
// render here as the banner message instead of in the body. The raw error (status and provider
// detail) sits behind "Show details", where its code snippet can be copied; the message's own
// "Regenerate response" action reruns the reply.
export const ChatErrorBanner = () => {
  const error = useMessageError();
  const explanation = useAuiState((state) =>
    state.message.parts
      .flatMap((part) => (part.type === 'text' && part.text.trim() ? [part.text.trim()] : []))
      .join('\n\n')
  );

  return (
    <Stack className="w-full min-w-0">
      <Banner
        kind="inline"
        status="error"
        className="w-full"
        data-testid="assistant-chat-error-banner"
      >
        <span className="whitespace-pre-wrap break-words">{explanation || GENERIC_CHAT_ERROR}</span>
      </Banner>
      {/* Its own row under the banner, styled like the work steps above an answer. */}
      {error ? (
        <WorkAccordion data-testid="assistant-chat-error-details" title="Show details">
          <WorkCodeSnippet label="Raw error" language="text" value={error} wrap />
        </WorkAccordion>
      ) : null}
    </Stack>
  );
};
