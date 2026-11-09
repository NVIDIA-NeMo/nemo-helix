// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GENERIC_CHAT_ERROR } from '@nemo/common/src/components/AssistantChat/describeChatError';
import {
  useFailedReplyExplanation,
  useMessageError,
} from '@nemo/common/src/components/AssistantChat/useMessageError';
import { WorkCodeSnippet } from '@nemo/common/src/components/AssistantChat/work/WorkCodeSnippet';
import { WorkStep } from '@nemo/common/src/components/AssistantChat/work/WorkStep';
import { Banner, Stack } from '@nvidia/foundations-react-core';

export const ChatErrorBanner = () => {
  const error = useMessageError();
  const explanation = useFailedReplyExplanation();

  return (
    <Stack className="mt-density-sm w-full min-w-0">
      <Banner
        kind="inline"
        status="error"
        className="w-full break-words"
        data-testid="assistant-chat-error-banner"
      >
        {explanation ?? GENERIC_CHAT_ERROR}
      </Banner>
      {error ? (
        <WorkStep data-testid="assistant-chat-error-details" title="Show details">
          <WorkCodeSnippet
            className="[&_pre]:whitespace-pre-wrap [&_pre]:break-all"
            label="Raw error"
            language="text"
            value={error}
          />
        </WorkStep>
      ) : null}
    </Stack>
  );
};
