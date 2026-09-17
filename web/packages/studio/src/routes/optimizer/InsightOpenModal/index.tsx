// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Insight } from '@nemo/sdk/generated/insights/schema';
import { Button, CodeSnippet, Modal, Stack, Text } from '@nvidia/foundations-react-core';
import { LINK_DOCS_EXPERIMENTS_CLI } from '@studio/constants/links';
import { buildOptimizerExperimentCommand } from '@studio/routes/optimizer/InsightOpenModal/command';
import { ChevronRight, File } from 'lucide-react';
import { type FC } from 'react';

export interface InsightOpenModalProps {
  open: boolean;
  insight: Insight;
  workspace: string;
  /** Close the modal. */
  onClose: () => void;
}

export const InsightOpenModal: FC<InsightOpenModalProps> = ({
  open,
  insight,
  workspace,
  onClose,
}) => {
  const cliCommand = buildOptimizerExperimentCommand(insight.id, workspace);

  return (
    <Modal
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
      slotHeading="Run experiment"
      className="w-[90vw] max-w-[720px]"
      attributes={{ ModalFooter: { className: 'justify-end' } }}
      slotFooter={
        <Button kind="primary" onClick={onClose}>
          Close
        </Button>
      }
    >
      <Stack gap="density-md">
        <Text kind="body/regular/md">
          Run the following CLI command to start experiments for this insight.
        </Text>
        <CodeSnippet value={cliCommand} language="bash" kind="block" className="w-full" />
        <Button
          asChild
          color="neutral"
          kind="tertiary"
          size="small"
          className="w-full justify-start"
        >
          <a href={LINK_DOCS_EXPERIMENTS_CLI} target="_blank" rel="noreferrer">
            <File className="!text-brand" />
            <Text className="flex-1">CLI docs — learn more</Text>
            <ChevronRight />
          </a>
        </Button>
      </Stack>
    </Modal>
  );
};
