// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useStoppedEarlyNotice } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { Banner } from '@nvidia/foundations-react-core';

export const StoppedEarlyBanner = () => {
  const notice = useStoppedEarlyNotice();
  return notice ? (
    <Banner
      kind="inline"
      status="warning"
      className="mt-density-sm w-full break-words"
      data-testid="assistant-chat-stopped-early-banner"
    >
      {notice}
    </Banner>
  ) : null;
};
