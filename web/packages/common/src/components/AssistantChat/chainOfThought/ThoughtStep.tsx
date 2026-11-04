// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  WORK_ICON_SIZE,
  WorkStep,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkStep';
import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import { Lightbulb } from '@nemo/icons/icons';

export const ThoughtStep = ({
  defaultOpen,
  text,
}: {
  readonly defaultOpen?: boolean;
  readonly text: string;
}) =>
  text.trim() ? (
    <WorkStep
      data-testid="assistant-work-thought"
      defaultOpen={defaultOpen}
      icon={<Lightbulb size={WORK_ICON_SIZE} />}
      title="Thought"
    >
      <MessageContent content={text} />
    </WorkStep>
  ) : null;
