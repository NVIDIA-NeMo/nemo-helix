// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  DeleteConfirmationModal,
  type DeleteModalProps,
} from '@nemo/common/src/components/DeleteConfirmationModal/index';
import type { NotifyType } from '@nemo/common/src/providers/toast/types';
import { Button, Stack, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';

type StoryArgs = Omit<DeleteModalProps, 'open' | 'onClose' | 'onNotify'>;

const DeleteConfirmationModalDemo = (args: StoryArgs) => {
  const [open, setOpen] = useState(true);
  const [lastNotification, setLastNotification] = useState<string>();

  const notify = (message: string, type: NotifyType = 'info') => {
    setLastNotification(`${type}: ${message}`);
  };

  return (
    <Stack gap="density-md" className="items-start">
      <Button color="danger" onClick={() => setOpen(true)}>
        Open dialog
      </Button>
      {lastNotification && <Text>Last notification: {lastNotification}</Text>}
      <DeleteConfirmationModal
        {...args}
        open={open}
        onClose={() => setOpen(false)}
        onNotify={notify}
      />
    </Stack>
  );
};

const succeed = () => Promise.resolve(true);

const meta: Meta<StoryArgs> = {
  title: 'Studio Common/DeleteConfirmationModal',
  render: (args) => <DeleteConfirmationModalDemo {...args} />,
  args: {
    onDelete: succeed,
    entity: 'model',
    ids: ['llama-3.1-8b-instruct'],
    description:
      'Deleting removes the model weights and unloads it from any active deployments. This can’t be undone.',
  },
  argTypes: {
    verb: { control: 'inline-radio', options: ['delete', 'remove'] },
    onDelete: { control: false },
  },
};

export default meta;

type Story = StoryObj<StoryArgs>;

export const Single: Story = {};

export const Bulk: Story = {
  args: {
    entity: 'dataset',
    ids: [
      'customer-support-train',
      'customer-support-eval',
      'faq-pairs',
      'tickets-q3',
      'tickets-q4',
    ],
    description: 'Deleting removes these datasets and every file in them. This can’t be undone.',
  },
};

export const BulkWithoutOverflow: Story = {
  args: {
    entity: 'secret',
    ids: ['NGC_API_KEY', 'OPENAI_API_KEY'],
    description: 'Anything that references these secrets will fail to authenticate.',
  },
};

export const IrregularPlural: Story = {
  args: {
    entity: 'guardrail policy',
    entityPlural: 'guardrail policies',
    ids: ['content-safety', 'jailbreak-detection', 'pii-redaction'],
    description: 'Virtual models that use these policies will stop applying them.',
  },
};

export const RemoveVerb: Story = {
  args: {
    verb: 'remove',
    entity: 'member',
    ids: ['alice@example.com'],
    description:
      'Revokes all of this member’s roles in the workspace. You can invite them again later.',
  },
};

export const LongId: Story = {
  args: {
    entity: 'file',
    ids: ['datasets/customer-support/2026-11/raw/conversations-export-part-0001-of-0128.jsonl'],
    description: 'Deleting removes the file from this fileset.',
  },
};

export const Pending: Story = {
  args: {
    onDelete: () => new Promise<boolean>(() => {}),
  },
};

export const Failure: Story = {
  args: {
    onDelete: () => Promise.resolve(false),
  },
};
