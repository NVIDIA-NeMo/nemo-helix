// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EntityAgentBridge } from '@nemo/common/src/components/EntityAgentBridge';
import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import type { Meta, StoryObj } from '@storybook/react';

const meta: Meta<typeof EntityAgentBridge> = {
  title: 'Common/EntityAgentBridge',
  component: EntityAgentBridge,
  decorators: [
    (Story) => (
      <ToastProvider>
        <div className="w-[640px]">
          <Story />
        </div>
      </ToastProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof EntityAgentBridge>;

/** Both halves present: the toggle defaults to the agent prompt. */
export const CreateAnEntity: Story = {
  args: { entity: 'guardrails', workspace: 'my-workspace' },
};

/** A read-only entity: the handoff is "analyze this", not "create one". */
export const ExamineAnEntity: Story = {
  args: { entity: 'telemetrySpans', workspace: 'my-workspace' },
};

/** `members` has a CLI command and no skill prompt, so no toggle renders. */
export const CliOnly: Story = {
  args: { entity: 'members', workspace: 'my-workspace' },
};

/** Without a workspace the placeholder is left for the reader to fill in. */
export const UnresolvedWorkspace: Story = {
  args: { entity: 'telemetryTraces' },
};
