// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReasoningMessagePartComponent } from '@assistant-ui/react';
import { CollapsedThinkingToolCall } from '@nemo/common/src/components/AssistantChat/toolCall/CollapsedThinkingToolCall';

export const ReasoningContent = ({ text }: { readonly text: string }) => (
  <CollapsedThinkingToolCall label="Thinking" text={text} />
);

export const ReasoningPart: ReasoningMessagePartComponent = ({ text }) => (
  <ReasoningContent text={text} />
);
